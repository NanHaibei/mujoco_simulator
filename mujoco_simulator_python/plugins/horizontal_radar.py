from __future__ import annotations
import mujoco
import numpy as np
import colorsys
from std_msgs.msg import Float32MultiArray
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..mujoco_simulator_python import mujoco_simulator

from .base import BasePlugin


class HorizontalRadar(BasePlugin):
    """水平雷达插件
    
    以某个site为参考，在site的高度往水平面均匀发射多条射线，
    计算每条射线碰撞到的障碍物的距离。
    
    使用 Alias 插值算法将射线结果映射到 16 个 bin 中，
    并进行距离编码，输出范围为 [0, 1]。
    
    编码规则：
    - sensor = max(0, (max_dist - dist) / max_dist)
    - 距离越近值越大，超出最大感知范围值为 0
    
    Bin 布局（逆时针递增）：
    - bin[0] = 正前方
    - bin[4] = 正左方  
    - bin[8] = 正后方
    - bin[12] = 正右方
    """
    
    def __init__(self, plugin_config: dict, simulator: mujoco_simulator):
        """初始化水平雷达插件"""
        super().__init__(plugin_config, simulator)
        
        # 读取配置参数
        self.site_name = plugin_config.get("site_name", "radar_site")
        self.num_rays = plugin_config.get("num_rays", 64)
        self.num_bins = plugin_config.get("num_bins", 16)
        self.max_distance = plugin_config.get("max_distance", 3.0)  # 默认 3.0m
        self.topic_name = plugin_config.get("topic_name", "/horizontal_radar")
        self.enable_visualization = plugin_config.get("enable_visualization", True)
        
        # offset 参数：在 site 本地坐标系下的位置偏移 [x, y, z]
        offset_config = plugin_config.get("offset", [0.0, 0.0, 0.0])
        self.offset = np.array(offset_config, dtype=np.float64)
        
        # 计算 bin 相关参数
        self.bin_size = 2 * np.pi / self.num_bins  # 每个 bin 的角度范围 (22.5°)
        
        # 查找site
        self.site_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_SITE, self.site_name
        )
        if self.site_id < 0:
            self.simulator.get_logger().warn(
                f"未找到 {self.site_name}，水平雷达插件禁用"
            )
            self.enabled = False
            return
        else:
            self.enabled = True
        
        # 计算每条射线的角度间隔（360度均匀分布，从正前方开始）
        self.ray_angles = np.linspace(0, 2 * np.pi, self.num_rays, endpoint=False)
        
        # 预分配射线方向数组
        self.ray_directions = np.zeros((self.num_rays, 3), dtype=np.float64)
        for i, angle in enumerate(self.ray_angles):
            self.ray_directions[i, 0] = np.cos(angle)  # x
            self.ray_directions[i, 1] = np.sin(angle)  # y
            self.ray_directions[i, 2] = 0.0             # z (水平)
        
        # 设置geomgroup（控制哪些几何体组可见）
        # 默认检测所有几何体
        self.geomgroup = (False, False, True, False, False, False)
        
        # 创建ROS发布者
        self.radar_pub = self.simulator.create_publisher(
            Float32MultiArray, self.topic_name, 10
        )
        
        # 存储结果
        self.distances = np.full(self.num_rays, self.max_distance, dtype=np.float64)
        self.lidar_output = np.zeros(self.num_bins, dtype=np.float32)
        
        # 可视化相关
        self.closest_ray_per_bin = np.zeros(self.num_bins, dtype=np.int32)  # 每个 bin 中最近障碍物的射线索引
        self.current_site_pos = None  # 当前帧的 site 位置
        
        self.simulator.get_logger().info(
            f"水平雷达插件已初始化: site={self.site_name}, "
            f"rays={self.num_rays}, bins={self.num_bins}, "
            f"max_dist={self.max_distance}, offset={self.offset.tolist()}"
        )
    
    def _encode_distance(self, dist: float) -> float:
        """距离编码函数
        
        编码公式: sensor = max(0, (max_dist - dist) / max_dist)
        - 距离越近值越大
        - 超出最大感知范围值为 0
        
        Args:
            dist: 实际距离
            
        Returns:
            编码后的值 [0, 1]
        """
        return max(0.0, (self.max_distance - dist) / self.max_distance)
    
    def _apply_alias_interpolation(self, angle: float, sensor: float, obs: np.ndarray):
        """应用 Alias 插值算法
        
        当一个物理点落入 bin[k] 内，且在 bin 内的相对角度偏移比例为 α 时：
        - 主 bin 更新: obs[k] = max(obs[k], sensor)
        - 逆时针相邻 bin 更新: obs[(k+1) % n] = max(..., α * sensor)
        - 顺时针相邻 bin 更新: obs[(k-1) % n] = max(..., (1-α) * sensor)
        
        Args:
            angle: 射线角度 (弧度)
            sensor: 编码后的传感器值
            obs: 输出数组 (num_bins,)
        """
        # 确保角度在 [0, 2π) 范围内
        angle = angle % (2 * np.pi)
        
        # 计算 bin 索引
        k = int(angle / self.bin_size)
        
        # 计算偏移比例 α
        alpha = (angle - k * self.bin_size) / self.bin_size
        
        # 更新主 bin
        obs[k] = max(obs[k], sensor)
        
        # 更新逆时针相邻 bin (左侧)
        k_ccw = (k + 1) % self.num_bins
        obs[k_ccw] = max(obs[k_ccw], alpha * sensor)
        
        # 更新顺时针相邻 bin (右侧)
        k_cw = (k - 1) % self.num_bins
        obs[k_cw] = max(obs[k_cw], (1 - alpha) * sensor)
    
    def execute(self):
        """执行射线追踪并发布结果"""
        if not hasattr(self, 'enabled') or not self.enabled:
            return
        
        # 获取site的世界坐标位置和旋转矩阵
        site_pos = self.mj_data.site_xpos[self.site_id].copy()
        site_mat = self.mj_data.site_xmat[self.site_id].reshape(3, 3).copy()
        
        # 将 offset 从 site 本地坐标系转换到世界坐标系
        if np.any(self.offset != 0):
            # 使用旋转矩阵将本地坐标转换到世界坐标
            offset_world = site_mat @ self.offset
            ray_origin = site_pos + offset_world
        else:
            ray_origin = site_pos
        
        self.current_site_pos = ray_origin  # 保存用于可视化
        
        # 计算机器人的 yaw 角（从旋转矩阵提取）
        # site_mat 是 3x3 旋转矩阵，yaw = atan2(mat[1,0], mat[0,0])
        yaw = np.arctan2(site_mat[1, 0], site_mat[0, 0])
        
        # 根据机器人的 yaw 角更新射线方向
        for i, base_angle in enumerate(self.ray_angles):
            # 将基础角度加上机器人的 yaw 角
            world_angle = base_angle + yaw
            self.ray_directions[i, 0] = np.cos(world_angle)  # x
            self.ray_directions[i, 1] = np.sin(world_angle)  # y
            self.ray_directions[i, 2] = 0.0                   # z (水平)
        
        # 存储距离结果
        dists = np.full(self.num_rays, self.max_distance, dtype=np.float64)
        
        # 使用射线追踪计算距离
        self._compute_distances(ray_origin, dists)
        
        # 保存距离结果用于可视化
        self.distances = dists.copy()
        
        # 初始化 lidar 输出为 0
        self.lidar_output.fill(0.0)
        
        # 记录每个 bin 中最近障碍物的射线索引（用于可视化）
        min_dist_per_bin = np.full(self.num_bins, self.max_distance)
        self.closest_ray_per_bin.fill(-1)
        
        # 对每条射线应用 Alias 插值
        for i in range(self.num_rays):
            # 距离编码
            sensor = self._encode_distance(dists[i])
            
            # 如果传感器值大于 0，应用插值
            if sensor > 0:
                self._apply_alias_interpolation(self.ray_angles[i], sensor, self.lidar_output)
            
            # 记录每个 bin 中最近障碍物的射线（使用整数运算与 visualize 保持一致）
            bin_idx = (i * self.num_bins) // self.num_rays
            if dists[i] < min_dist_per_bin[bin_idx]:
                min_dist_per_bin[bin_idx] = dists[i]
                self.closest_ray_per_bin[bin_idx] = i
        
        # 发布结果
        msg = Float32MultiArray()
        msg.data = self.lidar_output.tolist()
        self.radar_pub.publish(msg)
    
    def _get_bin_color(self, bin_idx: int, is_closest: bool = False) -> list:
        """根据 bin 索引获取颜色
        
        使用黄金角度分布的色相，使相邻 bin 颜色差异最大化：
        - 黄金角度 ≈ 0.618（色相环上的最优分布）
        - 相邻 bin 在色轮上相距约 222°，视觉差异明显
        
        Args:
            bin_idx: bin 索引 [0, num_bins-1]
            is_closest: 是否是该 bin 中最近障碍物的射线
            
        Returns:
            RGBA 颜色列表 [r, g, b, a]
        """
        # 最近障碍物的射线使用白色
        if is_closest:
            return [1.0, 1.0, 1.0, 1.0]  # 白色
        
        # 使用黄金角度分布色相，使相邻 bin 颜色差异最大化
        # 黄金比例 ≈ 0.618033988749895
        golden_ratio = 0.618033988749895
        hue = (bin_idx * golden_ratio) % 1.0
        
        # 饱和度和明度
        saturation = 0.8
        value = 0.9
        
        # HSV 转 RGB
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        
        return [r, g, b, 1.0]

    def _compute_distances(self, pnt: np.ndarray, dists: np.ndarray):
        """计算射线距离
        
        Args:
            pnt: 射线起点 (3,)
            dists: 距离结果数组 (num_rays,)
        """
        # 预分配 geomid 数组（mj_ray 需要这个参数）
        geomid = np.array([-1], dtype=np.int32)
        
        for i in range(self.num_rays):
            # mj_ray 返回距离，geomid 会被修改为碰撞的 geom id
            hit_dist = mujoco.mj_ray(
                self.mj_model,
                self.mj_data,
                pnt,
                self.ray_directions[i],
                self.geomgroup,
                1,
                -1,
                geomid
            )
            
            # hit_dist >= 0 表示有碰撞，返回的是距离值
            if hit_dist >= 0:
                dists[i] = min(hit_dist, self.max_distance)
    
    def visualize(self, viewer):
        """可视化射线
        
        绘制 64 根射线，不同 bin 使用不同颜色（HSV 色轮）：
        - 每个 bin 的射线使用不同的色相
        - 最近障碍物的射线更亮，其他射线稍暗
        
        颜色布局：
        - bin[0]（正前方）= 红色
        - bin[4]（正左方）= 绿色
        - bin[8]（正后方）= 青色
        - bin[12]（正右方）= 蓝色
        
        Args:
            viewer: mujoco viewer 实例
        """
        if not self.enable_visualization:
            return
        
        if not hasattr(self, 'enabled') or not self.enabled:
            return
        
        if self.current_site_pos is None:
            return
        
        # 保留场景中已有的几何体（如机器人模型等）
        viewer.user_scn.ngeom = self.mj_model.ngeom
        
        # 获取 closest_ray_per_bin 的集合
        closest_rays = set(self.closest_ray_per_bin[self.closest_ray_per_bin >= 0])
        
        # 绘制每条射线
        for i in range(self.num_rays):
            # 计算射线所属的 bin（使用整数运算避免浮点精度问题）
            bin_idx = (i * self.num_bins) // self.num_rays
            
            # 判断是否是该 bin 中最近障碍物的射线
            is_closest = i in closest_rays
            
            # 获取颜色
            color = self._get_bin_color(bin_idx, is_closest)
            
            # 计算射线起点和终点
            start = self.current_site_pos
            dist = self.distances[i]
            end = start + self.ray_directions[i] * dist
            
            # 增加几何体计数
            viewer.user_scn.ngeom += 1
            
            # 初始化线条几何体
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[viewer.user_scn.ngeom - 1],
                type=mujoco.mjtGeom.mjGEOM_LINE,
                size=[0.005, 0, 0],  # 线条粗细
                pos=start,
                mat=np.eye(3).flatten(),
                rgba=np.array(color, dtype=np.float32)
            )
            
            # 使用 mjv_connector 设置线条连接
            mujoco.mjv_connector(
                viewer.user_scn.geoms[viewer.user_scn.ngeom - 1],
                mujoco.mjtGeom.mjGEOM_LINE,
                3.03,  # width: 线条宽度
                start.astype(np.float64),  # from_: 起点 [3]
                end.astype(np.float64)     # to: 终点 [3]
            )
