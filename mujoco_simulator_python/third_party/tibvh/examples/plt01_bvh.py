import os
import sys
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('ply_file', type=str, help='Input PLY file path')
    parser.add_argument('--start-layer', type=int, default=0, help='')
    parser.add_argument('--vis-layers', type=int, default=3, help='Number of BVH top layers to visualize')
    parser.add_argument('--leaves-only', action='store_true', help='Visualize only leaf nodes, colored by SH (f_dc_0/1/2)')
    parser.add_argument('--vis-count', type=int, default=0, help='If >0, randomly sample this many boxes to visualize')
    parser.add_argument('--bg-color', type=str, default='0,0,0', help='Background color as "r,g,b" in 0..1')
    args = parser.parse_args()

    ply_file = args.ply_file

    scale_cutoff_ratio = 3.0

    start_time = time.time()
    gaussians, _ = load_gaussians(ply_file, cutoff=scale_cutoff_ratio, verbose=True)
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

    lbvh = LBVH(aabb_manager, profiling=False)

    start_time = time.time()
    lbvh.build()  # 预热运行
    ti.sync()     # 确保GPU操作完成
    print(f"LBVH 构建时间: {(time.time()-start_time)*1e3:.2f} 毫秒")

    # 可视化前 N 层 BVH 节点 AABB（线框）
    start_layer = int(args.start_layer)
    vis_layers = int(args.vis_layers)
    try:
        import open3d as o3d
    except Exception:
        o3d = None

    leaves_only = bool(args.leaves_only)

    if o3d is not None and (vis_layers > 0 or leaves_only):
        # 同步 Taichi 数据到 CPU
        ti.sync()
        nodes_min = lbvh.nodes.aabb_min.to_numpy()
        nodes_max = lbvh.nodes.aabb_max.to_numpy()
        left = lbvh.nodes.left.to_numpy()
        right = lbvh.nodes.right.to_numpy()
        element_id = lbvh.nodes.element_id.to_numpy()

        # BFS 收集前 vis_layers 层节点（根层为 0）
        n_nodes = lbvh.n_aabbs * 2 - 1
        root_idx = None
        parents = lbvh.nodes.parent.to_numpy()
        for i in range(n_nodes):
            if parents[i] == -1:
                root_idx = i
                break
        if root_idx is None:
            print("无法找到根节点，跳过 BVH 可视化")
        else:
            from collections import deque
            collected = []
            if leaves_only:
                # collect all leaf node indices
                n_nodes = lbvh.n_aabbs * 2 - 1
                internal = lbvh.n_aabbs - 1
                for li in range(internal, n_nodes):
                    collected.append(li)
            else:
                q = deque()
                q.append((root_idx, 0))
                while q:
                    idx, depth = q.popleft()
                    if depth >= vis_layers:
                        continue
                    if depth >= start_layer:
                        collected.append(idx)
                    l = int(left[idx]) if idx < left.shape[0] else -1
                    r = int(right[idx]) if idx < right.shape[0] else -1
                    if l >= 0:
                        q.append((l, depth + 1))
                    if r >= 0:
                        q.append((r, depth + 1))

            # 随机采样部分 box（如果用户指定 vis_count）
            vis_count = int(args.vis_count)
            if vis_count > 0 and len(collected) > vis_count:
                rng = np.random.default_rng()
                sampled_idx = rng.choice(len(collected), size=vis_count, replace=False)
                collected = [collected[i] for i in sampled_idx]

            # 构建 Open3D LineSet（每个 box 的 12 条边）
            lines = []
            points = []
            line_colors = []

            # if leaves_only and non-super_splat_format, try to read SH from PLY
            sh_colors = None
            if leaves_only:
                try:
                    import plyfile
                    plydata = plyfile.PlyData.read(ply_file)
                    if 'chunk' in plydata:
                        # super_splat_format: skip SH coloring
                        sh_colors = None
                    else:
                        vtx = plydata['vertex']
                        # check f_dc properties
                        if all(k in vtx.data.dtype.names for k in ('f_dc_0', 'f_dc_1', 'f_dc_2')):
                            f0 = vtx['f_dc_0'].astype(np.float32)
                            f1 = vtx['f_dc_1'].astype(np.float32)
                            f2 = vtx['f_dc_2'].astype(np.float32)
                            SH = np.stack([f0, f1, f2], axis=1)
                            C0 = 0.28209479177387814
                            rgb = SH * C0 + 0.5
                            rgb = np.clip(rgb, 0.0, 1.0)
                            # apply same filtering as load_gaussians: compute scale_properties and max_det
                            scale_props = [p.name for p in vtx.properties if p.name.startswith('scale_')]
                            if len(scale_props) > 0:
                                pscale = np.stack([vtx[p] for p in scale_props], axis=1).astype(np.float32)
                                scales_tmp = np.exp(pscale)
                                scale_det = np.abs(scales_tmp[:,0] * scales_tmp[:,1] * scales_tmp[:,2])
                                max_det = 0.5
                                valid_args = (scale_det < max_det)
                                rgb = rgb[valid_args]
                            sh_colors = rgb
                except Exception:
                    sh_colors = None

            for bidx in collected:
                bmin = nodes_min[bidx]
                bmax = nodes_max[bidx]
                # box 8 corners
                corners = np.array([
                    [bmin[0], bmin[1], bmin[2]],
                    [bmax[0], bmin[1], bmin[2]],
                    [bmax[0], bmax[1], bmin[2]],
                    [bmin[0], bmax[1], bmin[2]],
                    [bmin[0], bmin[1], bmax[2]],
                    [bmax[0], bmin[1], bmax[2]],
                    [bmax[0], bmax[1], bmax[2]],
                    [bmin[0], bmax[1], bmax[2]],
                ], dtype=np.float32)
                base_idx = len(points)
                points.extend(corners.tolist())
                # edges (pairs of corner indices)
                edges = [
                    (0,1),(1,2),(2,3),(3,0),
                    (4,5),(5,6),(6,7),(7,4),
                    (0,4),(1,5),(2,6),(3,7)
                ]
                # determine color for this box
                box_color = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                if leaves_only and sh_colors is not None:
                    # map from leaf node to element id
                    if bidx < element_id.shape[0]:
                        eid = int(element_id[bidx])
                        if eid >= 0 and eid < sh_colors.shape[0]:
                            box_color = sh_colors[eid].astype(np.float64)
                for e in edges:
                    lines.append((base_idx + e[0], base_idx + e[1]))
                    line_colors.append(box_color.tolist())

            if len(points) > 0:
                ls = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector(np.array(points, dtype=np.float64)),
                    lines=o3d.utility.Vector2iVector(np.array(lines, dtype=np.int32))
                )
                if len(line_colors) == len(lines):
                    ls.colors = o3d.utility.Vector3dVector(np.array(line_colors, dtype=np.float64))
                else:
                    ls.colors = o3d.utility.Vector3dVector(np.tile(np.array([[1.0,0.0,0.0]]), (len(lines),1)))
                print(f"可视化 {len(collected)} 个节点的 AABB，点数 {len(points)}，边数 {len(lines)}")
                # 使用 Visualizer 设置背景色
                try:
                    bg = [float(x) for x in args.bg_color.split(',')]
                    if len(bg) != 3:
                        raise ValueError
                except Exception:
                    bg = [0.0, 0.0, 0.0]
                vis = o3d.visualization.Visualizer()
                vis.create_window()
                opt = vis.get_render_option()
                opt.background_color = np.array(bg, dtype=np.float64)
                vis.add_geometry(ls)
                vis.poll_events()
                vis.update_renderer()
                vis.run()
                vis.destroy_window()
            else:
                print("无 AABB 可视化数据")

    