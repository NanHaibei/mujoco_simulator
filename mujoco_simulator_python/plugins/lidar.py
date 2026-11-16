from __future__ import annotations
import mujoco
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..mujoco_simulator_python import mujoco_simulator

from .base import BasePlugin


class Lidar(BasePlugin):
    """激光雷达插件
    
    负责获取点云信息，发送接口暂时留空。
    """
    
    def __init__(self, plugin_config: dict, simulator: mujoco_simulator):
        """初始化激光雷达插件"""
        super().__init__(plugin_config, simulator)
        
        # 查找lidar_site
        lidar_site_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "lidar_site")
        if lidar_site_id < 0:
            self.simulator.get_logger().warn("未找到 lidar_site，激光雷达插件禁用")
            self.enabled = False
            return
        
        # 获取传感器的位置和姿态
        self.lidar_site_pos = self.mj_model.site_pos[lidar_site_id].copy()  # [x, y, z]
        self.lidar_site_quat = self.mj_model.site_quat[lidar_site_id].copy()  # [w, x, y, z]
        
        # 设置雷达类型
        # 使用随仓库迁入的库，无需安装整套示例依赖。
        import sys
        from pathlib import Path
        library_path = str(Path(__file__).resolve().parents[1] / "third_party" / "MuJoCo-LiDAR")
        if library_path not in sys.path:
            sys.path.insert(0, library_path)
        from mujoco_lidar.scan_gen import LivoxGenerator
        self.livox_generator = LivoxGenerator("mid360")
        self.rays_theta, self.rays_phi = self.livox_generator.sample_ray_angles()
        
        # 设置geomgroup（控制哪些几何体组可见）
        geomgroup = np.array([1, 0, 1, 0, 0, 0], dtype=np.uint8)
        
        # 创建雷达句柄
        from mujoco_lidar.lidar_wrapper import MjLidarWrapper
        self.lidar_sim = MjLidarWrapper(
            self.mj_model,
            site_name="lidar_site",
            backend="cpu",
            cutoff_dist=100.0,
            args={
                'bodyexclude': -1,
                'geomgroup': geomgroup,
                'max_candidates': 64,
                'ti_init_args': {'device_memory_GB': 4.0}
            }
        )

        self.enabled = True
        self.last_points = None

    def after_step(self):
        """获取点云并交给空发送接口"""
        if not self.should_execute():
            return
        if not self.enabled:
            return
        # 更新雷达射线角度（动态扫描）
        self.rays_theta, self.rays_phi = self.livox_generator.sample_ray_angles()
        
        # 执行光线追踪
        self.lidar_sim.trace_rays(self.mj_data, self.rays_theta, self.rays_phi)
        
        # 获取击中点
        hit_points = self.lidar_sim.get_hit_points()
        world_points = hit_points
        
        self.last_points = np.asarray(world_points, dtype=np.float32).copy()
        self.send(self.last_points)

    def send(self, state):
        """发送本插件的采样结果，外部通信待本插件接入。"""
        pass

    def reset(self):
        if not self.enabled:
            return
        self.last_points = None
        self.livox_generator.currStartIndex = 0
