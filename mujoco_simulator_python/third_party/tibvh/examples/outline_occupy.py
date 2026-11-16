import os
import time
import tqdm
import argparse
import numpy as np

import torch
import taichi as ti

def filter_surface_points_tile(
    m_grid_points: np.ndarray,
    resolution: float,
    aabb_min: np.ndarray,
    grid_size: np.ndarray,
    tile_size: int = 64,
    use_halo: bool = True,
):
    """
    使用三维瓦片 + halo 的GPU算法，删除内部点，仅保留外部点（六邻域缺失即外部）。

    参数:
        m_grid_points: (N, 3) float32，体素中心坐标
        resolution: 标量体素尺寸
        aabb_min: (3,) float32，网格AABB最小点
        grid_size: (3,) int32，总体素尺寸 (gx, gy, gz)
        tile_size: 每个瓦片边长 B（建议 64/96）
        use_halo: 是否装载六面 halo 保证跨瓦片邻居正确
        device: 'cuda' 或 'cpu'（建议 'cuda'）

    返回:
        surface_points: (M, 3) float32，仅外部点
    """

    device: str = 'cuda',
    assert m_grid_points.ndim == 2 and m_grid_points.shape[1] == 3

    # 输入 -> Torch (GPU) 加速离散化与分桶
    pts = torch.from_numpy(m_grid_points.astype(np.float32))
    pts = pts.cuda(non_blocking=True)

    aabb_min_t = torch.as_tensor(aabb_min, dtype=torch.float32, device=pts.device)
    res_t = torch.tensor(resolution, dtype=torch.float32, device=pts.device)
    grid_size_t = torch.as_tensor(grid_size, dtype=torch.int32, device=pts.device)

    # 映射到整型网格索引（中心对齐规则：x = min + 0.5*res + i*res）
    # i = round((x - (min + 0.5*res)) / res)
    ijk = torch.round((pts - (aabb_min_t + 0.5 * res_t)) / res_t).to(torch.int32)

    # 分离 in-grid 与 out-of-grid（越界点一律保留为外部）
    in_mask = (
        (ijk[:, 0] >= 0) & (ijk[:, 0] < grid_size_t[0]) &
        (ijk[:, 1] >= 0) & (ijk[:, 1] < grid_size_t[1]) &
        (ijk[:, 2] >= 0) & (ijk[:, 2] < grid_size_t[2])
    )
    outside_points = pts[~in_mask]

    if in_mask.any():
        ijk_in = ijk[in_mask]
    else:
        # 全部越界，直接返回
        return outside_points.detach().cpu().numpy()

    B = int(tile_size)
    gx, gy, gz = int(grid_size[0]), int(grid_size[1]), int(grid_size[2])
    Tx = (gx + B - 1) // B
    Ty = (gy + B - 1) // B
    Tz = (gz + B - 1) // B
    num_tiles = Tx * Ty * Tz

    # 计算瓦片坐标与局部坐标
    tx = torch.div(ijk_in[:, 0], B, rounding_mode='floor')
    ty = torch.div(ijk_in[:, 1], B, rounding_mode='floor')
    tz = torch.div(ijk_in[:, 2], B, rounding_mode='floor')
    li = ijk_in[:, 0] - tx * B
    lj = ijk_in[:, 1] - ty * B
    lk = ijk_in[:, 2] - tz * B

    # tile_id = (tz*Ty + ty)*Tx + tx
    tile_id = (tz.to(torch.int64) * Ty + ty.to(torch.int64)) * Tx + tx.to(torch.int64)

    # 基础版：按 tile_id 排序将点分组（GPU）
    order = torch.argsort(tile_id)
    tile_id_sorted = tile_id[order]
    loc_sorted = torch.stack([li[order], lj[order], lk[order]], dim=1).to(torch.int32)

    # 计算每个 tile 的起止 offset（GPU）
    counts = torch.bincount(tile_id_sorted, minlength=num_tiles)
    offsets = torch.zeros(num_tiles + 1, dtype=torch.int64, device=pts.device)
    offsets[1:] = torch.cumsum(counts, dim=0)

    # 将需要的元数据拷回CPU以供 Taichi 字段初始化
    N_in = loc_sorted.shape[0]

    # Taichi 字段与常量
    sorted_local = ti.Vector.field(3, dtype=ti.i32, shape=N_in)
    tile_offsets = ti.field(dtype=ti.i64, shape=num_tiles + 1)
    out_points = ti.Vector.field(3, dtype=ti.f32, shape=N_in)  # 仅 in-grid 部分的上限
    out_count = ti.field(dtype=ti.i32, shape=())

    # 局部占据体（本块+halo）：(B+2)^3
    occ = ti.field(dtype=ti.u1, shape=(num_tiles, B + 2, B + 2, B + 2))
    interior = ti.field(dtype=ti.u1, shape=(num_tiles, B + 2, B + 2, B + 2))

    # 常量参数
    Tx_i, Ty_i, Tz_i = int(Tx), int(Ty), int(Tz)
    gx_i, gy_i, gz_i = int(gx), int(gy), int(gz)
    res_f = float(resolution)
    aabb_min_f = np.asarray(aabb_min, dtype=np.float32)

    # 数据导入 Taichi
    sorted_local.from_torch(loc_sorted)
    tile_offsets.from_torch(offsets)

    aabb_min_ti = ti.Vector.field(3, dtype=ti.f32, shape=())
    aabb_min_ti[None] = ti.Vector(aabb_min_f.tolist())
    res_ti = ti.field(dtype=ti.f32, shape=())
    res_ti[None] = res_f

    @ti.func
    def tile_coords_from_id(t):
        tx = t % Tx_i
        ty = (t // Tx_i) % Ty_i
        tz = (t // (Tx_i * Ty_i))
        return ti.Vector([tx, ty, tz])

    @ti.func
    def tile_id_from_coords(tx, ty, tz):
        return (tz * Ty_i + ty) * Tx_i + tx

    @ti.kernel
    def process_all_tiles():
        out_count[None] = 0
        for i_tile in ti.ndrange(num_tiles):
            # 清零占据
            for x, y, z in ti.ndrange(B + 2, B + 2, B + 2):
                occ[i_tile, x, y, z] = False
                interior[i_tile, x, y, z] = False

            # 当前 tile 尺寸（最后一块可能不足 B）
            tc = tile_coords_from_id(i_tile)
            tx = tc[0]; ty = tc[1]; tz = tc[2]
            size_x = ti.min(B, gx_i - tx * B)
            size_y = ti.min(B, gy_i - ty * B)
            size_z = ti.min(B, gz_i - tz * B)
            if size_x <= 0 or size_y <= 0 or size_z <= 0:
                continue

            # 当前 tile 的点段（将 i64 偏移转换为 i32 以用于 range）
            s0 = ti.i32(tile_offsets[i_tile])
            s1 = ti.i32(tile_offsets[i_tile + 1])

            # 写入核心占据
            for s in range(s0, s1):
                v = sorted_local[s]
                lx = v[0] + 1
                ly = v[1] + 1
                lz = v[2] + 1
                if lx >= 1 and lx <= size_x and ly >= 1 and ly <= size_y and lz >= 1 and lz <= size_z:
                    occ[i_tile, lx, ly, lz] = True

            if ti.static(use_halo):
                # 六个方向装载 halo：仅边界一层
                # -x 邻：当前的 occ[i_tile, 0, :, :] 由左侧 tile 的 li==B-1 填充
                if tx > 0:
                    tn = tile_id_from_coords(tx - 1, ty, tz)
                    n0 = ti.i32(tile_offsets[tn])
                    n1 = ti.i32(tile_offsets[tn + 1])
                    for s in range(n0, n1):
                        v = sorted_local[s]
                        if v[0] == B - 1:
                            ly2 = v[1] + 1
                            lz2 = v[2] + 1
                            if ly2 >= 1 and ly2 <= size_y and lz2 >= 1 and lz2 <= size_z:
                                occ[i_tile, 0, ly2, lz2] = True
                # +x 邻：右侧 tile 的 li==0 -> occ[i_tile, B+1, :, :]
                if tx + 1 < Tx_i:
                    tn = tile_id_from_coords(tx + 1, ty, tz)
                    n0 = ti.i32(tile_offsets[tn])
                    n1 = ti.i32(tile_offsets[tn + 1])
                    for s in range(n0, n1):
                        v = sorted_local[s]
                        if v[0] == 0:
                            ly2 = v[1] + 1
                            lz2 = v[2] + 1
                            if ly2 >= 1 and ly2 <= size_y and lz2 >= 1 and lz2 <= size_z:
                                occ[i_tile, size_x + 1, ly2, lz2] = True
                # -y 邻：lj==B-1 -> occ[i_tile, :,0,:]
                if ty > 0:
                    tn = tile_id_from_coords(tx, ty - 1, tz)
                    n0 = ti.i32(tile_offsets[tn])
                    n1 = ti.i32(tile_offsets[tn + 1])
                    for s in range(n0, n1):
                        v = sorted_local[s]
                        if v[1] == B - 1:
                            lx2 = v[0] + 1
                            lz2 = v[2] + 1
                            if lx2 >= 1 and lx2 <= size_x and lz2 >= 1 and lz2 <= size_z:
                                occ[i_tile, lx2, 0, lz2] = True
                # +y 邻：lj==0 -> occ[i_tile, :, B+1, :]
                if ty + 1 < Ty_i:
                    tn = tile_id_from_coords(tx, ty + 1, tz)
                    n0 = ti.i32(tile_offsets[tn])
                    n1 = ti.i32(tile_offsets[tn + 1])
                    for s in range(n0, n1):
                        v = sorted_local[s]
                        if v[1] == 0:
                            lx2 = v[0] + 1
                            lz2 = v[2] + 1
                            if lx2 >= 1 and lx2 <= size_x and lz2 >= 1 and lz2 <= size_z:
                                occ[i_tile, lx2, size_y + 1, lz2] = True
                # -z 邻：lk==B-1 -> occ[i_tile, :,:,0]
                if tz > 0:
                    tn = tile_id_from_coords(tx, ty, tz - 1)
                    n0 = ti.i32(tile_offsets[tn])
                    n1 = ti.i32(tile_offsets[tn + 1])
                    for s in range(n0, n1):
                        v = sorted_local[s]
                        if v[2] == B - 1:
                            lx2 = v[0] + 1
                            ly2 = v[1] + 1
                            if lx2 >= 1 and lx2 <= size_x and ly2 >= 1 and ly2 <= size_y:
                                occ[i_tile, lx2, ly2, 0] = True
                # +z 邻：lk==0 -> occ[i_tile, :,:,B+1]
                if tz + 1 < Tz_i:
                    tn = tile_id_from_coords(tx, ty, tz + 1)
                    n0 = ti.i32(tile_offsets[tn])
                    n1 = ti.i32(tile_offsets[tn + 1])
                    for s in range(n0, n1):
                        v = sorted_local[s]
                        if v[2] == 0:
                            lx2 = v[0] + 1
                            ly2 = v[1] + 1
                            if lx2 >= 1 and lx2 <= size_x and ly2 >= 1 and ly2 <= size_y:
                                occ[i_tile, lx2, ly2, size_z + 1] = True

            # 计算 interior（核心区域 1..size_*）
            for lx, ly, lz in ti.ndrange((1, size_x + 1), (1, size_y + 1), (1, size_z + 1)):
                v = occ[i_tile, lx, ly, lz]
                xm = occ[i_tile, lx - 1, ly, lz]
                xp = occ[i_tile, lx + 1, ly, lz]
                ym = occ[i_tile, lx, ly - 1, lz]
                yp = occ[i_tile, lx, ly + 1, lz]
                zm = occ[i_tile, lx, ly, lz - 1]
                zp = occ[i_tile, lx, ly, lz + 1]
                interior[i_tile, lx, ly, lz] = v and xm and xp and ym and yp and zm and zp

            # 输出当前 tile 的外部点
            for s in range(s0, s1):
                v = sorted_local[s]
                lx = v[0] + 1
                ly = v[1] + 1
                lz = v[2] + 1
                if lx >= 1 and lx <= size_x and ly >= 1 and ly <= size_y and lz >= 1 and lz <= size_z:
                    is_occ = occ[i_tile, lx, ly, lz]
                    is_interior = interior[i_tile, lx, ly, lz]
                    if is_occ and (not is_interior):
                        # 反算世界坐标：x = min + 0.5*res + (tx*B + (lx-1))*res
                        wx = aabb_min_ti[None][0] + 0.5 * res_ti[None] + ti.cast(tx * B + (lx - 1), ti.f32) * res_ti[None]
                        wy = aabb_min_ti[None][1] + 0.5 * res_ti[None] + ti.cast(ty * B + (ly - 1), ti.f32) * res_ti[None]
                        wz = aabb_min_ti[None][2] + 0.5 * res_ti[None] + ti.cast(tz * B + (lz - 1), ti.f32) * res_ti[None]
                        oi = ti.atomic_add(out_count[None], 1)
                        if oi < ti.static(out_points.shape[0]):
                            out_points[oi] = ti.Vector([wx, wy, wz])

    # 运行内核
    process_all_tiles()
    ti.sync()

    # 收集 in-grid 外部点
    out_n = int(out_count.to_numpy())
    surface_in = out_points.to_numpy()[:out_n]

    # 合并 out-of-grid（它们天然为外部）
    if outside_points.numel() > 0:
        surface_outside = outside_points.detach().cpu().numpy()
        surface_points = np.concatenate([surface_in, surface_outside], axis=0)
    else:
        surface_points = surface_in

    return surface_points


def infer_grid_from_points(points: np.ndarray, resolution: float):
    """
    尝试从点云与分辨率推断 aabb_min 和 grid_size（依赖中心对齐规则）。
    注意：更稳妥做法是从上游传入这两个参数。
    """
    pmin = points.min(axis=0)
    pmax = points.max(axis=0)
    aabb_min = pmin - 0.5 * resolution
    # 估算 grid_size（向上取整）
    size = np.round((pmax - aabb_min - 0.5 * resolution) / resolution).astype(np.int64) + 1
    size = np.maximum(size, 1)
    return aabb_min.astype(np.float32), size.astype(np.int32)

if __name__ == "__main__":
    ti.init(
        arch=ti.gpu,
        kernel_profiler=True,
        advanced_optimization=True,
        offline_cache=True,
    )

    parser = argparse.ArgumentParser(description="Filter surface points using tile+halo GPU algorithm")
    parser.add_argument('--npy', type=str, default='/home/tatp/Desktop/grids.npy', help='Input npy path (N,4) or (N,3)')
    parser.add_argument('--resolution', type=float, default=0.033, help='Voxel resolution')
    parser.add_argument('--tile', type=int, default=64, help='Tile size B')
    args = parser.parse_args()

    arr = np.load(args.npy)
    pts = arr[:, :3].astype(np.float32)
    
    aabb_min, grid_size = infer_grid_from_points(pts, args.resolution)

    print(f"Input points: {pts.shape[0]}")
    print(f"resolution={args.resolution}, aabb_min={aabb_min}, grid_size={grid_size}")

    # surface_pts = filter_surface_points_tile(
    #     pts,
    #     resolution=args.resolution,
    #     aabb_min=aabb_min,
    #     grid_size=grid_size,
    #     tile_size=args.tile,
    #     use_halo=True,
    # )
    ##############################################################
    merged_points = pts
    resolution = args.resolution

    # 分片策略：沿 z 轴将 merged_points 均匀分成 N 份，相邻分片在 z 方向重叠 3×resolution
    # N 依据 z 方向 tile 数 Tz 估算：每片约含 4 个 z-tiles
    start_time = time.time()
    B = int(args.tile)
    gz = int(grid_size[2])
    Tz = (gz + B - 1) // B
    tiles_per_slice = 1
    N = max(1, int(np.ceil(Tz / tiles_per_slice)))

    z_vals = merged_points[:, 2]
    zmin_all = float(z_vals.min())
    zmax_all = float(z_vals.max())
    overlap = 3.0 * float(resolution)

    surface_parts = []
    for k in tqdm.trange(N):
        base_z0 = zmin_all + (zmax_all - zmin_all) * (k / N)
        base_z1 = zmin_all + (zmax_all - zmin_all) * ((k + 1) / N)
        z0 = max(zmin_all, base_z0 - overlap)
        z1 = min(zmax_all, base_z1 + overlap)
        mask_k = (z_vals >= z0) & (z_vals <= z1)
        part_pts = merged_points[mask_k]
        print(part_pts.shape)
        if part_pts.shape[0] == 0:
            continue
        sub_surface = filter_surface_points_tile(
            part_pts,
            resolution=args.resolution,
            aabb_min=aabb_min,
            grid_size=grid_size,
            tile_size=args.tile,
        )
        if sub_surface is not None and sub_surface.size > 0:
            surface_parts.append(sub_surface)

    if len(surface_parts) == 0:
        surface_pts = np.empty((0, 3), dtype=np.float32)
    else:
        surface_concat = np.concatenate(surface_parts, axis=0)
        # 拼接后的整体再执行一次全局过滤，去除跨分片边界处的内部点/离散点
        surface_pts = filter_surface_points_tile(
            surface_concat,
            resolution=args.resolution,
            aabb_min=aabb_min,
            grid_size=grid_size,
            tile_size=args.tile,
        )
    ##############################################################

    print(f"Surface points: {surface_pts.shape[0]}")
    out_path = os.path.splitext(args.npy)[0] + f"_surface.npy"
    np.save(out_path, surface_pts.astype(np.float32))
    print(f"Saved: {out_path}")

    from ply_model import xyz2ply, xyz2supersplat    
    xyz2ply(surface_pts, os.path.splitext(args.npy)[0] + f"_surface.ply")
    xyz2supersplat(surface_pts, os.path.splitext(args.npy)[0] + f"_surface_supersplat.ply", base_scale=args.resolution)

