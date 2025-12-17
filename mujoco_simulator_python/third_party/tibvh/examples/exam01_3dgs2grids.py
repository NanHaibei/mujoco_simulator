import os
import sys
import tqdm
import time
import argparse

import numpy as np
import taichi as ti
from tibvh import AABB, LBVH

ti.init(
    arch=ti.gpu,
    kernel_profiler=True,
    advanced_optimization=True,
    offline_cache=True,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.')))
from gaussian_model import Gaussian, load_gaussians
from ply_model import xyz2ply, xyz2supersplat
from outline_occupy import filter_surface_points_tile

#######################################################################################
@ti.func
def opacity_func(x):
    # 自己随便实现了一个，可以试试换成别的
    # 将透明的物体尽量映射到 0

    # return x
    # return x ** 2
    return x ** 4

@ti.func
def gaussian_density(pos:ti.types.vector(3, ti.f32), gaussian:Gaussian):
    """
    计算单个3D高斯体在指定位置的密度值
    """
    diff = pos - gaussian.position
    inner = gaussian.cov_inv @ diff
    exponent = -0.5 * diff.dot(inner)
    norm_factor = 15.749609945722419 * gaussian.sqrt_det
    density = ti.exp(exponent) / norm_factor
    # return density --- IGNORE ---
    # 计算不透明度加权密度
    density_opc = density * opacity_func(gaussian.opacity)
    return density_opc
#######################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('ply_file', type=str, help='Input PLY file path')
    parser.add_argument('-o', '--output', type=str, default=None, help='Output folder path')
    parser.add_argument('-c', '--cutoff', type=float, default=3.0, help='Cutoff ratio for Gaussian influence')
    parser.add_argument('-r', '--resolution', type=float, default=0.02, help='Voxel resolution, meter')
    parser.add_argument('-d', '--density-threshold', type=float, default=1.0, help='Density threshold for filtering')
    parser.add_argument('-bg', '--grid-batch', type=int, default=25, help='Grid points batch size (log2)')
    parser.add_argument('-bq', '--max-query-results', type=int, default=29, help='Max query results for LBVH (log2)')
    parser.add_argument('-md', '--max-gaussian-det', type=float, default=0.05, help='Max Gaussian determinant (sx*sy*sz)')
    parser.add_argument('--tile', type=int, default=64, help='Tile size B')
    args = parser.parse_args()

    ply_file = args.ply_file
    output_path = args.output if args.output else os.path.join(os.path.dirname(ply_file), "output")
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    scale_cutoff_ratio = args.cutoff
    resolution = args.resolution
    density_threshold = args.density_threshold
    grid_points_batch_size = 2**args.grid_batch
    max_query_results = 2**args.max_query_results
    max_gaussian_det = args.max_gaussian_det

    start_time = time.time()
    gaussians, _ = load_gaussians(ply_file, cutoff=scale_cutoff_ratio, max_det=max_gaussian_det, verbose=True)
    print(f"加载高斯体时间: {(time.time()-start_time)*1e3:.2f} 毫秒")
    gaussian_points_np = gaussians.position.to_numpy()

    n_gaussians = gaussians.shape[0]
    aabb_manager = AABB(max_n_aabbs=n_gaussians)

    @ti.kernel
    def fill_aabb_manager():
        for i in ti.ndrange(n_gaussians):
            aabb_manager.aabbs[i].min = gaussians[i].position - gaussians[i].radius
            aabb_manager.aabbs[i].max = gaussians[i].position + gaussians[i].radius
    fill_aabb_manager()

    lbvh = LBVH(aabb_manager, max_query_results=max_query_results, profiling=False)

    start_time = time.time()
    lbvh.build()  # 预热运行
    ti.sync()     # 确保GPU操作完成
    print(f"LBVH 构建时间: {(time.time()-start_time)*1e3:.2f} 毫秒")

    aabb_min = gaussian_points_np.min(axis=0) - resolution
    aabb_max = gaussian_points_np.max(axis=0) + resolution
    print(aabb_min, aabb_max)

    aabb_size = aabb_max - aabb_min
    print("AABB Min:", aabb_min)
    print("AABB Max:", aabb_max)
    print("AABB Size:", aabb_size)

    grid_size = (aabb_size / resolution).astype(np.int32)
    print("Grid Size:", grid_size)
    print("Total Voxels:", grid_size.prod())

    # 计算所有体素中心点位置
    xs = np.linspace(aabb_min[0] + resolution * 0.5, aabb_max[0] - resolution * 0.5, grid_size[0], dtype=np.float32)
    ys = np.linspace(aabb_min[1] + resolution * 0.5, aabb_max[1] - resolution * 0.5, grid_size[1], dtype=np.float32)
    zs = np.linspace(aabb_min[2] + resolution * 0.5, aabb_max[2] - resolution * 0.5, grid_size[2], dtype=np.float32)
    grid_x, grid_y, grid_z = np.meshgrid(xs, ys, zs, indexing='ij')
    grid_points = np.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3)  # (N, 3)
    n_points = grid_points.shape[0]
    densities_all = np.zeros((n_points,), dtype=np.float32)
    print("Total Grid Points:", n_points)

    n_batches = (n_points + grid_points_batch_size - 1) // grid_points_batch_size

    # 预分配并复用 GPU 缓冲，避免每轮创建 Field 触发显存增长
    max_batch_size = grid_points_batch_size
    grid_points_ti = ti.Vector.field(3, dtype=ti.f32, shape=max_batch_size)
    densities_ti = ti.field(dtype=ti.f32, shape=max_batch_size)

    # 预分配一次 host 缓冲用于 from_numpy（填充到 max_batch_size）
    upload_buffer = np.zeros((max_batch_size, 3), dtype=np.float32)

    @ti.kernel
    def accumulate(current_batch_size: int) -> int:
        # 清零当前批次的密度
        for i in range(current_batch_size):
            densities_ti[i] = 0.0

        # 进行批量查询
        overflow = lbvh.query(grid_points_ti)

        # 按查询结果累加密度
        for k in range(lbvh.query_result_count[None]):
            # 结果为 (aabb_id, query_id)
            aabb_id = lbvh.query_result[k][0]
            qid = lbvh.query_result[k][1]
            if 0 <= aabb_id < n_gaussians and 0 <= qid < current_batch_size:
                pos = grid_points_ti[qid]
                density = gaussian_density(pos, gaussians[aabb_id])
                ti.atomic_add(densities_ti[qid], density)
        return overflow

    start_time = time.time()
    for i in tqdm.tqdm(range(n_batches)):
        start_idx = i * grid_points_batch_size
        end_idx = min((i + 1) * grid_points_batch_size, n_points)
        current_batch_size = end_idx - start_idx

        # 填充 host 缓冲的前 current_batch_size 段，其余保持 0
        upload_buffer[:current_batch_size, :] = grid_points[start_idx:end_idx]
        if current_batch_size < max_batch_size:
            # 可选：清零尾部，已由初始零化满足
            pass
        grid_points_ti.from_numpy(upload_buffer)

        overflow = accumulate(current_batch_size)
        ti.sync()
        assert not overflow, "LBVH 查询结果溢出，考虑增加 max_query_results 参数"

        densities_batch = densities_ti.to_numpy()[:current_batch_size]
        densities_all[start_idx:end_idx] = densities_batch
    print(f"处理时间: {(time.time()-start_time):.2f} 秒")

    mask = densities_all > density_threshold

    m_grid_points = grid_points[mask]
    m_densities_all = densities_all[mask]
    print(f"筛选后体素数: {m_grid_points.shape[0]} / {n_points}, 阈值: {density_threshold}")

    grids_with_density = np.concatenate([m_grid_points, m_densities_all[:, None]], axis=-1)  # (N, 4)
    output_file = os.path.join(output_path, "grids.npy")
    np.save(output_file, grids_with_density)
    print(f"保存网格数据到 {output_file}")

    merged_points = np.concatenate([gaussian_points_np, m_grid_points], axis=0)
    ##############################################################
    start_time = time.time()
    surface_pts = filter_surface_points_tile(
        merged_points, # m_grid_points
        resolution=args.resolution,
        aabb_min=aabb_min,
        grid_size=grid_size,
        tile_size=args.tile,
    )
    ##############################################################
    print(f"提取表面点: {(time.time()-start_time)*1e3:.2f} 毫秒")

    print(f"提取表面点数: {surface_pts.shape[0]} / {m_grid_points.shape[0]}")
    output_file = os.path.join(output_path, "surface_points.npy")
    np.save(output_file, surface_pts)
    print(f"保存表面点数据到 {output_file}")
    xyz2ply(surface_pts, os.path.join(output_path, "surface_points_bin.ply"))

    # map surface points to gaussian AABB indices and derive colors per-point
    try:
        total_sp = surface_pts.shape[0]
        voxel_batch_size = 2**20 if total_sp > (1<<20) else total_sp
        sp_ti = ti.Vector.field(3, dtype=ti.f32, shape=voxel_batch_size)
        sp_to_aabb = ti.field(dtype=ti.i32, shape=voxel_batch_size)

        @ti.kernel
        def map_surface(batch_size: int):
            for i in range(batch_size):
                sp_to_aabb[i] = -1
            _ = lbvh.query(sp_ti)
            for k in range(lbvh.query_result_count[None]):
                aid = lbvh.query_result[k][0]
                qid = lbvh.query_result[k][1]
                if 0 <= qid < batch_size and sp_to_aabb[qid] == -1:
                    sp_to_aabb[qid] = aid

        # attempt to read SH colors from PLY
        sh_colors = None
        try:
            import plyfile
            plydata = plyfile.PlyData.read(ply_file)
            if 'chunk' not in plydata:
                vtx = plydata['vertex']
                if all(k in vtx.data.dtype.names for k in ('f_dc_0', 'f_dc_1', 'f_dc_2')):
                    f0 = vtx['f_dc_0'].astype(np.float32)
                    f1 = vtx['f_dc_1'].astype(np.float32)
                    f2 = vtx['f_dc_2'].astype(np.float32)
                    SH = np.stack([f0, f1, f2], axis=1)
                    C0 = 0.28209479177387814
                    rgb = SH * C0 + 0.5
                    rgb = np.clip(rgb, 0.0, 1.0)
                    scale_props = [p.name for p in vtx.properties if p.name.startswith('scale_')]
                    if len(scale_props) > 0:
                        pscale = np.stack([vtx[p] for p in scale_props], axis=1).astype(np.float32)
                        scales_tmp = np.exp(pscale)
                        scale_det = np.abs(scales_tmp[:,0] * scales_tmp[:,1] * scales_tmp[:,2])
                        valid_args = (scale_det < 0.5)
                        rgb = rgb[valid_args]
                    sh_colors = rgb
        except Exception:
            sh_colors = None

        sp_colors = np.zeros((total_sp, 3), dtype=np.float32)
        for i in range((total_sp + voxel_batch_size - 1) // voxel_batch_size):
            s = i * voxel_batch_size
            e = min((i + 1) * voxel_batch_size, total_sp)
            cur = e - s
            upload = np.zeros((voxel_batch_size, 3), dtype=np.float32)
            upload[:cur] = surface_pts[s:e]
            sp_ti.from_numpy(upload)
            map_surface(cur)
            ti.sync()
            aabb_map = sp_to_aabb.to_numpy()[:cur]
            for j in range(cur):
                aid = int(aabb_map[j])
                if aid >= 0 and sh_colors is not None and aid < sh_colors.shape[0]:
                    sp_colors[s + j] = sh_colors[aid]
                elif aid >= 0:
                    hashv = (aid * 1664525) & 0xFFFFFFFF
                    r = ((hashv >> 0) & 0xFF) / 255.0
                    g = ((hashv >> 8) & 0xFF) / 255.0
                    b = ((hashv >> 16) & 0xFF) / 255.0
                    sp_colors[s + j] = [r, g, b]
                else:
                    sp_colors[s + j] = [0.7, 0.7, 0.7]

        xyz2supersplat(surface_pts, os.path.join(output_path, "surface_points_supersplat.ply"), base_scale=resolution, colors=sp_colors)
    except Exception as e:
        print("Failed to compute per-surface colors, falling back to random: ", e)
        xyz2supersplat(surface_pts, os.path.join(output_path, "surface_points_supersplat.ply"), base_scale=resolution)