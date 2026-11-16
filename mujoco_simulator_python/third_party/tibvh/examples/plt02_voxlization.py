import os
import sys
import time
import argparse

import numpy as np
import taichi as ti
from tibvh import AABB, LBVH

ti.init(arch=ti.gpu, kernel_profiler=True, advanced_optimization=True, offline_cache=True)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.')))
from gaussian_model import Gaussian, load_gaussians


def parse_bg(s: str):
    try:
        parts = [float(x) for x in s.split(',')]
        if len(parts) != 3:
            return [0.0, 0.0, 0.0]
        return parts
    except Exception:
        return [0.0, 0.0, 0.0]


@ti.func
def opacity_func(x):
    return x ** 2


@ti.func
def gaussian_density(pos, gaussian):
    diff = pos - gaussian.position
    inner = gaussian.cov_inv @ diff
    exponent = -0.5 * diff.dot(inner)
    norm_factor = 15.749609945722419 * gaussian.sqrt_det
    density = ti.exp(exponent) / norm_factor
    return density * opacity_func(gaussian.opacity)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ply_file', type=str)
    parser.add_argument('-r', '--resolution', type=float, default=0.02)
    parser.add_argument('--bg-color', type=str, default='0,0,0')
    parser.add_argument('--density-threshold', type=float, default=1.0)
    parser.add_argument('--vis-limit', type=int, default=200000, help='Max number of voxels to visualize (0=no limit)')
    parser.add_argument('--point-size', type=float, default=1, help='Point size when rendering voxels as points')
    parser.add_argument('--surface-filter', action='store_true', help='After voxel filtering, run filter_surface_points_tile and visualize resulting surface points instead of full voxels')
    parser.add_argument('--tile', type=int, default=64, help='Tile size to pass to filter_surface_points_tile')
    args = parser.parse_args()

    ply_file = args.ply_file
    resolution = args.resolution
    density_threshold = args.density_threshold

    # reuse load_gaussians to perform same filtering
    gaussians, is_2dgs = load_gaussians(ply_file, cutoff=3.0, verbose=False)
    gaussian_points_np = gaussians.position.to_numpy()
    n_gaussians = gaussians.shape[0]
    aabb_manager = AABB(max_n_aabbs=n_gaussians)

    @ti.kernel
    def fill_aabb_manager():
        for i in ti.ndrange(n_gaussians):
            aabb_manager.aabbs[i].min = gaussians[i].position - gaussians[i].radius
            aabb_manager.aabbs[i].max = gaussians[i].position + gaussians[i].radius

    fill_aabb_manager()

    lbvh = LBVH(aabb_manager, profiling=False)
    lbvh.build()
    ti.sync()

    aabb_min = gaussian_points_np.min(axis=0) - resolution
    aabb_max = gaussian_points_np.max(axis=0) + resolution
    aabb_size = aabb_max - aabb_min
    grid_size = (aabb_size / resolution).astype(np.int32)

    xs = np.linspace(aabb_min[0] + resolution * 0.5, aabb_max[0] - resolution * 0.5, grid_size[0], dtype=np.float32)
    ys = np.linspace(aabb_min[1] + resolution * 0.5, aabb_max[1] - resolution * 0.5, grid_size[1], dtype=np.float32)
    zs = np.linspace(aabb_min[2] + resolution * 0.5, aabb_max[2] - resolution * 0.5, grid_size[2], dtype=np.float32)
    grid_x, grid_y, grid_z = np.meshgrid(xs, ys, zs, indexing='ij')
    grid_points = np.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3)

    n_points = grid_points.shape[0]
    print(f"Total grid points: {n_points}")

    # reuse accumulate kernel pattern to compute densities
    grid_points_batch_size = 2**22 if n_points > (1<<22) else n_points
    max_batch_size = int(grid_points_batch_size)
    grid_points_ti = ti.Vector.field(3, dtype=ti.f32, shape=max_batch_size)
    densities_ti = ti.field(dtype=ti.f32, shape=max_batch_size)
    upload_buffer = np.zeros((max_batch_size, 3), dtype=np.float32)

    @ti.kernel
    def accumulate(current_batch_size: int) -> int:
        for i in range(current_batch_size):
            densities_ti[i] = 0.0
        overflow = lbvh.query(grid_points_ti)
        for k in range(lbvh.query_result_count[None]):
            aabb_id = lbvh.query_result[k][0]
            qid = lbvh.query_result[k][1]
            if 0 <= aabb_id < n_gaussians and 0 <= qid < current_batch_size:
                pos = grid_points_ti[qid]
                d = gaussian_density(pos, gaussians[aabb_id])
                ti.atomic_add(densities_ti[qid], d)
        return overflow

    densities_all = np.zeros((n_points,), dtype=np.float32)
    n_batches = (n_points + max_batch_size - 1) // max_batch_size
    print(f"Computing densities in {n_batches} batches, batch size {max_batch_size}")
    for i in range(n_batches):
        s = i * max_batch_size
        e = min((i + 1) * max_batch_size, n_points)
        cur = e - s
        upload_buffer[:cur, :] = grid_points[s:e]
        grid_points_ti.from_numpy(upload_buffer)
        overflow = accumulate(cur)
        ti.sync()
        densities_all[s:e] = densities_ti.to_numpy()[:cur]

    mask = densities_all > density_threshold
    m_grid_points = grid_points[mask]
    print(f"Filtered voxels: {m_grid_points.shape[0]}")

    if m_grid_points.shape[0] == 0:
        print("No voxels to visualize")
        return

    # optionally run surface extraction and replace voxels with surface points
    if args.surface_filter:
        try:
            from outline_occupy import filter_surface_points_tile
            merged_points = np.concatenate([gaussian_points_np, m_grid_points], axis=0)
            print("Running surface filter (filter_surface_points_tile)...")
            start_sf = time.time()
            surface_pts = filter_surface_points_tile(
                merged_points,
                resolution=resolution,
                aabb_min=aabb_min,
                grid_size=grid_size,
                tile_size=int(args.tile),
            )
            print(f"Surface filter done in {(time.time()-start_sf):.2f}s, got {surface_pts.shape[0]} points")
            if surface_pts.shape[0] == 0:
                print("No surface points produced")
                return
            # use surface points as visualization points (render small cubes centered at points)
            m_grid_points = surface_pts
            total_vox = m_grid_points.shape[0]
        except Exception as e:
            print("Failed to run surface filter:", e)
            # continue with voxel visualization

    # Map voxels to aabb (gaussian indices) in batches
    voxel_colors = []
    voxel_positions = []

    voxel_batch_size = max_batch_size
    voxel_ti = ti.Vector.field(3, dtype=ti.f32, shape=voxel_batch_size)
    voxel_to_aabb = ti.field(dtype=ti.i32, shape=voxel_batch_size)

    @ti.kernel
    def map_voxels(batch_size: int):
        for i in range(batch_size):
            voxel_to_aabb[i] = -1
        _ = lbvh.query(voxel_ti)
        for k in range(lbvh.query_result_count[None]):
            aid = lbvh.query_result[k][0]
            qid = lbvh.query_result[k][1]
            if 0 <= qid < batch_size and voxel_to_aabb[qid] == -1:
                voxel_to_aabb[qid] = aid

    # prepare colors per gaussian: try read SH from ply
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
                # apply same filtering as load_gaussians: compute scales and filter by default max_det in load_gaussians
                scale_props = [p.name for p in vtx.properties if p.name.startswith('scale_')]
                if len(scale_props) > 0:
                    pscale = np.stack([vtx[p] for p in scale_props], axis=1).astype(np.float32)
                    scales_tmp = np.exp(pscale)
                    scale_det = np.abs(scales_tmp[:,0] * scales_tmp[:,1] * scales_tmp[:,2])
                    # default max_det used in load_gaussians is 0.5
                    valid_args = (scale_det < 0.5)
                    rgb = rgb[valid_args]
                sh_colors = rgb
    except Exception:
        sh_colors = None

    # prepare Open3D geometry list
    try:
        import open3d as o3d
    except Exception:
        o3d = None

    if o3d is None:
        print("open3d not installed, install with: pip install open3d")
        return

    total_vox = m_grid_points.shape[0]
    vis_limit = int(args.vis_limit)
    if vis_limit > 0 and total_vox > vis_limit:
        # sample
        rng = np.random.default_rng()
        idxs = rng.choice(total_vox, size=vis_limit, replace=False)
        m_grid_points = m_grid_points[idxs]
        total_vox = vis_limit

    # batch map
    for i in range((total_vox + voxel_batch_size - 1) // voxel_batch_size):
        s = i * voxel_batch_size
        e = min((i + 1) * voxel_batch_size, total_vox)
        cur = e - s
        upload = np.zeros((voxel_batch_size, 3), dtype=np.float32)
        upload[:cur] = m_grid_points[s:e]
        voxel_ti.from_numpy(upload)
        map_voxels(cur)
        ti.sync()
        aabb_map = voxel_to_aabb.to_numpy()[:cur]
        for j in range(cur):
            aid = int(aabb_map[j])
            pos = m_grid_points[s + j]
            if aid >= 0:
                if sh_colors is not None and aid < sh_colors.shape[0]:
                    color = sh_colors[aid].tolist()
                else:
                    # fallback: color from hash
                    hashv = (aid * 1664525) & 0xFFFFFFFF
                    r = ((hashv >> 0) & 0xFF) / 255.0
                    g = ((hashv >> 8) & 0xFF) / 255.0
                    b = ((hashv >> 16) & 0xFF) / 255.0
                    color = [r, g, b]
            else:
                color = [0.7, 0.7, 0.7]
            voxel_positions.append(pos)
            voxel_colors.append(color)

    # If surface_filter is disabled, render voxels as an Open3D PointCloud (faster)
    meshes = []
    # if not args.surface_filter:
    if args.surface_filter:
        pts = np.array(voxel_positions, dtype=np.float64)
        cols = np.array(voxel_colors, dtype=np.float64)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(cols)
    else:
        for pos, col in zip(voxel_positions, voxel_colors):
            box = o3d.geometry.TriangleMesh.create_box(width=resolution, height=resolution, depth=resolution)
            box.compute_vertex_normals()
            m = box.translate((pos[0] - resolution/2, pos[1] - resolution/2, pos[2] - resolution/2), relative=False)
            m.paint_uniform_color(col)
            meshes.append(m)

    # visualize
    bg = parse_bg(args.bg_color)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    opt = vis.get_render_option()
    opt.background_color = np.array(bg, dtype=np.float64)
    # if not args.surface_filter:
    if args.surface_filter:
        vis.add_geometry(pcd)
        # try to set point size (may be ignored depending on Open3D backend)
        opt.point_size = float(args.point_size)
    else:
        for m in meshes:
            vis.add_geometry(m)
    vis.poll_events()
    vis.update_renderer()
    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    main()
