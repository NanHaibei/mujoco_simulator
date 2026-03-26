from __future__ import annotations

import mujoco
import numpy as np
from std_msgs.msg import Float32MultiArray
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..mujoco_simulator_python import mujoco_simulator

from .base import BasePlugin


class DlioRadarVisualizer(BasePlugin):
    """DLIO 雷达可视化插件（16方向 mjray 风格）

    订阅 DLIO 发布的 /horizontal_radar：
    - 前 16 维: goal_lidar
    - 后 16 维: hazards_lidar

    可视化方式：
    - hazards: 按 16 个 bin 画 16 根水平线，未命中时也显示到 max_distance
    - goal: 只取最强的一个 bin，用一根红线表示目标方向
    """

    def __init__(self, plugin_config: dict, simulator: mujoco_simulator):
        super().__init__(plugin_config, simulator)

        self.topic_name = plugin_config.get("topic_name", "/horizontal_radar")
        self.site_name = plugin_config.get("site_name", "lidar_site")
        self.num_bins = plugin_config.get("num_bins", 16)
        self.max_distance = plugin_config.get("max_distance", 3.0)
        self.goal_max_distance = plugin_config.get("goal_max_distance", 15.0)
        self.enable_visualization = plugin_config.get("enable_visualization", True)
        self.show_goal = plugin_config.get("show_goal", True)
        self.show_hazards = plugin_config.get("show_hazards", True)
        self.hazard_width = plugin_config.get("hazard_width", 3.0)
        self.goal_width = plugin_config.get("goal_width", 6.0)
        self.hazard_alpha = plugin_config.get("hazard_alpha", 1.0)
        self.goal_alpha = plugin_config.get("goal_alpha", 1.0)
        self.goal_single_line = plugin_config.get("goal_single_line", True)

        offset_config = plugin_config.get("offset", [0.0, 0.0, 0.0])
        self.offset = np.array(offset_config, dtype=np.float64)

        self.site_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_SITE, self.site_name
        )
        if self.site_id < 0:
            self.simulator.get_logger().warn(
                f"未找到 {self.site_name}，DLIO 雷达可视化插件禁用"
            )
            self.enabled = False
            return

        self.enabled = True
        self.bin_size = 2.0 * np.pi / self.num_bins
        self.bin_angles = np.arange(self.num_bins, dtype=np.float64) * self.bin_size

        self.goal_bins = np.zeros(self.num_bins, dtype=np.float32)
        self.hazard_bins = np.zeros(self.num_bins, dtype=np.float32)
        self.current_site_pos = None
        self.current_yaw = 0.0

        self.radar_sub = self.simulator.create_subscription(
            Float32MultiArray, self.topic_name, self.radar_callback, 10
        )

        self.simulator.get_logger().info(
            f"DLIO雷达可视化插件已初始化: site={self.site_name}, topic={self.topic_name}, bins={self.num_bins}"
        )

    def radar_callback(self, msg: Float32MultiArray):
        expected_dim = self.num_bins * 2
        if len(msg.data) < expected_dim:
            self.simulator.get_logger().warn(
                f"收到 /horizontal_radar 维度不足: {len(msg.data)} < {expected_dim}"
            )
            return

        data = np.asarray(msg.data[:expected_dim], dtype=np.float32)
        self.goal_bins = data[:self.num_bins].copy()
        self.hazard_bins = data[self.num_bins:expected_dim].copy()

    def execute(self):
        if not self.enabled:
            return

        site_pos = self.mj_data.site_xpos[self.site_id].copy()
        site_mat = self.mj_data.site_xmat[self.site_id].reshape(3, 3).copy()

        if np.any(self.offset != 0):
            self.current_site_pos = site_pos + site_mat @ self.offset
        else:
            self.current_site_pos = site_pos

        self.current_yaw = np.arctan2(site_mat[1, 0], site_mat[0, 0])

    def _decode_distance(self, value: float, max_distance: float) -> float:
        value = float(np.clip(value, 0.0, 1.0))
        return max_distance * (1.0 - value)

    def _draw_line(self, viewer, start: np.ndarray, end: np.ndarray, color: list[float], width: float):
        viewer.user_scn.ngeom += 1
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[viewer.user_scn.ngeom - 1],
            type=mujoco.mjtGeom.mjGEOM_LINE,
            size=[0.005, 0, 0],
            pos=start,
            mat=np.eye(3).flatten(),
            rgba=np.array(color, dtype=np.float32)
        )
        mujoco.mjv_connector(
            viewer.user_scn.geoms[viewer.user_scn.ngeom - 1],
            mujoco.mjtGeom.mjGEOM_LINE,
            width,
            start.astype(np.float64),
            end.astype(np.float64)
        )

    def visualize(self, viewer):
        if not self.enable_visualization or not self.enabled:
            return

        if self.current_site_pos is None:
            return

        viewer.user_scn.ngeom = self.mj_model.ngeom
        start = self.current_site_pos.astype(np.float64)

        if self.show_hazards:
            for i in range(self.num_bins):
                sensor = float(np.clip(self.hazard_bins[i], 0.0, 1.0))
                angle = self.current_yaw + self.bin_angles[i]
                direction = np.array([np.cos(angle), np.sin(angle), 0.0], dtype=np.float64)
                dist = self._decode_distance(sensor, self.max_distance)
                end = start + direction * dist
                color = [1.0, 1.0, 1.0, self.hazard_alpha]
                self._draw_line(viewer, start, end, color, self.hazard_width)

        if self.show_goal:
            if self.goal_single_line:
                i = int(np.argmax(self.goal_bins))
                sensor = float(np.clip(self.goal_bins[i], 0.0, 1.0))
                if sensor > 1e-4:
                    angle = self.current_yaw + self.bin_angles[i]
                    direction = np.array([np.cos(angle), np.sin(angle), 0.0], dtype=np.float64)
                    dist = self._decode_distance(sensor, self.goal_max_distance)
                    end = start + direction * dist
                    color = [1.0, 0.0, 0.0, self.goal_alpha]
                    self._draw_line(viewer, start, end, color, self.goal_width)
            else:
                for i in range(self.num_bins):
                    sensor = float(np.clip(self.goal_bins[i], 0.0, 1.0))
                    if sensor <= 1e-4:
                        continue
                    angle = self.current_yaw + self.bin_angles[i]
                    direction = np.array([np.cos(angle), np.sin(angle), 0.0], dtype=np.float64)
                    dist = self._decode_distance(sensor, self.goal_max_distance)
                    end = start + direction * dist
                    color = [1.0, 0.0, 0.0, self.goal_alpha]
                    self._draw_line(viewer, start, end, color, self.goal_width)
