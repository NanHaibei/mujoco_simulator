import torch
import plyfile
import numpy as np
import taichi as ti

@ti.dataclass
class Gaussian:
    """
    3D高斯体类，表示场景中的一个高斯分布点
    用于在LiDAR仿真中表示物体表面
    """
    position: ti.math.vec3  # 高斯体中心位置
    cov_inv: ti.math.mat3   # 协方差矩阵的逆矩阵，描述高斯体的形状和方向
    opacity: ti.f32         # 不透明度/强度
    scale: ti.math.vec3     # 高斯体在局部坐标系三个主轴方向上的尺度
    radius: ti.math.vec3    # 高斯体在三个主轴方向上的包围半径，用于BVH构建，注意不是椭球体的半径，而是沿世界坐标系下x,y,z轴的包围半径，和scale区分
    
    # 下面添加2D高斯椭圆相交所需的预计算字段
    normal: ti.math.vec3      # 椭圆平面法向量（旋转矩阵的第三列）
    rot_matrix: ti.math.mat3  # 旋转矩阵
    sqrt_det: ti.f32     # 协方差矩阵行列式的平方根，sqrt(|Σ|)
    # sqrt(|Σ|) = sx * sy * sz （Σ = R S S^T R^T, S=diag(sx,sy,sz)）

# ================== torch 工具函数 ==================
def RGB2SH(rgb):
    C0 = 0.28209479177387814
    return (rgb - 0.5).astype(np.float32) / C0

def quat_to_rot(q: torch.Tensor) -> torch.Tensor:
    """
    将四元数转换为旋转矩阵
    
    参数:
        q: 四元数张量 [N, 4] (w, x, y, z)
    
    返回:
        旋转矩阵张量 [N, 3, 3]
    """
    q = q / q.norm(dim=-1, keepdim=True)  # 四元数归一化
    w, x, y, z = q.unbind(-1)  # 分解四元数分量
    # 构建旋转矩阵
    return torch.stack([
        1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)
    ], dim=-1).view(-1, 3, 3)

def decode_supersplat_ply(plydata):
    vtx = plydata['vertex'].data  # structured array
    chk = plydata['chunk'].data   # structured array

    # 每256个vertex对应一个chunk（按顺序）
    num_vertex = vtx.shape[0]
    if num_vertex == 0:
        empty = np.zeros((0, 3), dtype=np.float32)
        return empty, np.zeros((0, 4), dtype=np.float32), empty.copy(), empty.copy(), np.zeros((0,), dtype=np.float32)

    chunk_idx = (np.arange(num_vertex) // 256).astype(np.int64)
    # 防御性裁剪（以防最后一个 chunk 未满或越界情况）
    chunk_idx = np.clip(chunk_idx, 0, chk.shape[0] - 1)

    # 拉取每个点对应 chunk 的标量边界
    def gather_chunk(field):
        return chk[field][chunk_idx]

    # 解码位置（11/10/11）
    ppos = vtx['packed_position'].astype(np.uint32)
    xbits = (ppos >> 21) & 0x7FF
    ybits = (ppos >> 11) & 0x3FF
    zbits = ppos & 0x7FF
    fx = xbits.astype(np.float32) / 2047.0 * (gather_chunk('max_x') - gather_chunk('min_x')) + gather_chunk('min_x')
    fy = ybits.astype(np.float32) / 1023.0 * (gather_chunk('max_y') - gather_chunk('min_y')) + gather_chunk('min_y')
    fz = zbits.astype(np.float32) / 2047.0 * (gather_chunk('max_z') - gather_chunk('min_z')) + gather_chunk('min_z')
    positions = np.stack([fx, fy, fz], axis=1).astype(np.float32)

    # 解码尺度（11/10/11），并指数还原
    pscale = vtx['packed_scale'].astype(np.uint32)
    sxb = (pscale >> 21) & 0x7FF
    syb = (pscale >> 11) & 0x3FF
    szb = pscale & 0x7FF
    sx = sxb.astype(np.float32) / 2047.0 * (gather_chunk('max_scale_x') - gather_chunk('min_scale_x')) + gather_chunk('min_scale_x')
    sy = syb.astype(np.float32) / 1023.0 * (gather_chunk('max_scale_y') - gather_chunk('min_scale_y')) + gather_chunk('min_scale_y')
    sz = szb.astype(np.float32) / 2047.0 * (gather_chunk('max_scale_z') - gather_chunk('min_scale_z')) + gather_chunk('min_scale_z')
    scales = np.exp(np.stack([sx, sy, sz], axis=1)).astype(np.float32)

    # 解码颜色和不透明度（8/8/8/8）
    pcol = vtx['packed_color'].astype(np.uint32)
    r8 = (pcol >> 24) & 0xFF
    g8 = (pcol >> 16) & 0xFF
    b8 = (pcol >> 8) & 0xFF
    a8 = pcol & 0xFF
    fr = r8.astype(np.float32) / 255.0 * (gather_chunk('max_r') - gather_chunk('min_r')) + gather_chunk('min_r')
    fg = g8.astype(np.float32) / 255.0 * (gather_chunk('max_g') - gather_chunk('min_g')) + gather_chunk('min_g')
    fb = b8.astype(np.float32) / 255.0 * (gather_chunk('max_b') - gather_chunk('min_b')) + gather_chunk('min_b')
    SH_C0 = 0.28209479177387814
    fr = (fr - 0.5) / SH_C0
    fg = (fg - 0.5) / SH_C0
    fb = (fb - 0.5) / SH_C0
    opacity = a8.astype(np.float32) / 255.0
    # opacity = 1.0 / (1.0 + np.exp(-opacity))
    colors = np.stack([fr, fg, fb], axis=1).astype(np.float32)
    opacities = opacity.astype(np.float32)

    # 解码旋转（最大分量索引 + 3×10bit）
    prot = vtx['packed_rotation'].astype(np.uint32)
    largest = (prot >> 30) & 0x3  # 0..3
    v0 = (prot >> 20) & 0x3FF
    v1 = (prot >> 10) & 0x3FF
    v2 = prot & 0x3FF
    norm = np.sqrt(2.0) * 0.5
    vals = np.stack([v0, v1, v2], axis=1).astype(np.float32)
    vals = (vals / 1023.0 - 0.5) / norm
    # 映射到四元数的非最大分量（顺序依 index 增序，略过 largest）
    q = np.zeros((num_vertex, 4), dtype=np.float32)

    # Masks for largest index
    m0 = (largest == 0)
    m1 = (largest == 1)
    m2 = (largest == 2)
    m3 = (largest == 3)

    # 对应关系见说明：
    # largest=0: (1,2,3) <= (v0,v1,v2)
    q[m0, 1] = vals[m0, 0]
    q[m0, 2] = vals[m0, 1]
    q[m0, 3] = vals[m0, 2]
    # largest=1: (0,2,3) <= (v0,v1,v2)
    q[m1, 0] = vals[m1, 0]
    q[m1, 2] = vals[m1, 1]
    q[m1, 3] = vals[m1, 2]
    # largest=2: (0,1,3) <= (v0,v1,v2)
    q[m2, 0] = vals[m2, 0]
    q[m2, 1] = vals[m2, 1]
    q[m2, 3] = vals[m2, 2]
    # largest=3: (0,1,2) <= (v0,v1,v2)
    q[m3, 0] = vals[m3, 0]
    q[m3, 1] = vals[m3, 1]
    q[m3, 2] = vals[m3, 2]

    # 复原最大分量
    sum_sq = np.sum(q * q, axis=1)
    max_comp = np.sqrt(np.clip(1.0 - sum_sq, 0.0, 1.0)).astype(np.float32)
    # 写回到对应的 largest 位置（0:w, 1:x, 2:y, 3:z）
    q[m0, 0] = max_comp[m0]
    q[m1, 1] = max_comp[m1]
    q[m2, 2] = max_comp[m2]
    q[m3, 3] = max_comp[m3]
    quats = q.astype(np.float32)
    return positions, quats, scales, colors, opacities

# ================== 数据加载 ==================
def load_gaussians(ply_path, cutoff=3.0, max_det=0.5, verbose=False):
    """
    从PLY文件加载3D高斯模型
    
    参数:
        ply_path: PLY文件路径
        
    返回:
        高斯体场(Gaussian.field)
    """
    # 读取PLY文件
    if verbose:
        print(f"加载高斯模型: {ply_path}")
    plydata = plyfile.PlyData.read(ply_path)

    try:
        plydata['chunk']
        super_splat_format = True
    except KeyError:
        super_splat_format = False

    if super_splat_format:
        positions, quats, scales, _colors, opacities = decode_supersplat_ply(plydata)
        is_2dgs = False
    else:
        vertex = plydata['vertex']
        scale_properties = [p.name for p in vertex.properties if p.name.startswith('scale_')]
        is_2dgs = True if len(scale_properties) == 2 else False

        # 提取高斯体参数
        positions = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=-1).astype(np.float32)  # 位置
        quats = np.stack([vertex[f'rot_{i}'] for i in range(4)], axis=-1).astype(np.float32)  # 旋转四元数 wxyz
        scales = np.exp(np.stack([vertex[p] for p in scale_properties], axis=-1).astype(np.float32))  # 尺度
        opacities = (1./(1.+np.exp(-vertex['opacity']))).astype(np.float32) # 不透明度

    # 过滤过大的高斯体
    scale_det = np.abs(scales[:, 0] * scales[:, 1] * scales[:, 2])
    valid_args = (scale_det < max_det)
    positions = positions[valid_args]
    quats = quats[valid_args]
    scales = scales[valid_args]
    opacities = opacities[valid_args]
    print(f"过滤掉过大高斯体: {np.sum(~valid_args)} / {len(valid_args)}")

    if is_2dgs:
        scales = np.hstack([scales, 1e-9*np.ones_like(scales[:, :1])])
    elif scales[:, 2].max() < 1.1e-6:
        scales[:, 2] = 1e-9
        is_2dgs = True

    if verbose:
        print(">" * 100)
        print(f"高斯模型统计: {len(positions)}个点")
        print(f"位置范围: [{positions.min(axis=0)}, {positions.max(axis=0)}]")
        print(f"尺度范围: [{scales.min(axis=0)}, {scales.max(axis=0)}]")
        print(f"旋转四元数范围: [{quats.min(axis=0)}, {quats.max(axis=0)}]")
        print(f"不透明度范围: [{opacities.min()}, {opacities.max()}]")
        print(f"是否为2DGS: {is_2dgs}")
        print(f"是否单位四元数: {np.all(np.isclose(np.linalg.norm(quats, axis=-1), 1))}")
        print("<" * 100)

    # 创建Taichi高斯体场
    gaussians = Gaussian.field(shape=len(positions))
    # 批量上传数据到GPU，避免多次小数据传输
    gaussians.position.from_numpy(positions)  # 设置位置

    # 计算协方差矩阵及其逆矩阵
    with torch.no_grad():
        R = quat_to_rot(torch.from_numpy(quats))  # 四元数转旋转矩阵
        scales_cu = torch.from_numpy(scales)
        S = torch.diag_embed(scales_cu)  # 尺度矩阵
        cov = R @ S @ S.transpose(1,2) @ R.transpose(1,2)  # 计算协方差矩阵
        cov[:] += torch.eye(3, device=cov.device) * 1e-9  # 防止协方差矩阵奇异
        cov_inv_cu = torch.inverse(cov)  # 计算协方差矩阵逆矩阵
        sqrt_det_cu = torch.sqrt(torch.det(cov))  # 计算协方差矩阵行列式的平方根
    
        # 对每个高斯体计算三维半径
        radius_xyz_cu = torch.zeros((len(positions), 3), dtype=torch.float32)
        # 截断比例
        opa_cu = torch.from_numpy(opacities)
        cutoff_ratio = torch.sqrt(torch.max((cutoff**2) + 2.0 * torch.log(opa_cu), 1e-6*torch.ones_like(opa_cu)))

        # 将旋转矩阵和尺度结合，计算每个轴的最大半径
        for i in range(3):  # 对x, y, z轴分别计算
            # 计算在旋转后空间中的投影
            proj_vec = R[:, i, :]
            # 计算每个方向的半径（元素级乘法后再求和的平方根）
            radius_xyz_cu[:, i] = cutoff_ratio * torch.sqrt(torch.sum(
                (proj_vec * scales_cu)**2, dim=1
            ))

        gaussians.cov_inv.from_torch(cov_inv_cu.float())  # 上传协方差矩阵逆矩阵
        gaussians.opacity.from_torch(opa_cu)  # 设置不透明度
        gaussians.scale.from_torch(scales_cu)  # 设置尺度
        gaussians.radius.from_torch(radius_xyz_cu)  # 设置三维包围半径

        gaussians.rot_matrix.from_torch(R)  # 设置旋转矩阵
        gaussians.sqrt_det.from_torch(sqrt_det_cu.float())  # 上传协方差矩阵行列式的平方根

    # 为2DGS椭圆计算预处理数据
    if is_2dgs:
        normals = R[:, :, 2].numpy()  # 旋转矩阵的第三列是法向量

    # 如果是2DGS，设置预计算的椭圆参数
    if is_2dgs:
        gaussians.normal.from_numpy(normals.astype(np.float32))

    # 限制max_det，如果超过这个范围的gaossians会被过滤掉
    if max_det > 0:
        pass
    
    return gaussians, is_2dgs