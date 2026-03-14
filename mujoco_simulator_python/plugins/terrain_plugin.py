from __future__ import annotations
from visualization_msgs.msg import Marker, MarkerArray
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..mujoco_simulator_python import mujoco_simulator

from .base_plugin import BasePlugin


class TerrainPlugin(BasePlugin):
    """地形可视化插件
    
    负责发布地形障碍物信息用于可视化。
    """
    
    def __init__(self, name: str, plugin_config: dict, simulator: mujoco_simulator):
        """初始化地形可视化插件"""
        super().__init__(name, plugin_config, simulator)
        # 创建发布者
        self.marker_array_pub = self.simulator.create_publisher(
            MarkerArray, '/visualization_marker_array', 10
        )
    
    def execute(self):
        """执行地形可视化发布"""
        if not self.enabled:
            return
        
        # 如果模型读取有错误，则不执行操作
        if self.simulator.read_error_flag:
            return
        
        # 发布地形可视化信息
        marker_array = MarkerArray()
        marker_id = 0
        
        for i in range(len(self.simulator.terrain_pos)):
            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = self.simulator.get_clock().now().to_msg()
            marker.ns = "mujoco"
            marker.id = marker_id
            marker_id += 1
            
            # 根据地形类型设置marker类型
            if self.simulator.terrain_type[i] == 'box':
                marker.type = Marker.CUBE
            elif self.simulator.terrain_type[i] == 'cylinder':
                marker.type = Marker.CYLINDER
            else:
                marker.type = Marker.CUBE
            
            marker.action = Marker.ADD
            
            marker.pose.position.x = self.simulator.terrain_pos[i][0]
            marker.pose.position.y = self.simulator.terrain_pos[i][1]
            marker.pose.position.z = self.simulator.terrain_pos[i][2]
            
            marker.pose.orientation.w = self.simulator.terrain_quat[i][0]
            marker.pose.orientation.x = self.simulator.terrain_quat[i][1]
            marker.pose.orientation.y = self.simulator.terrain_quat[i][2]
            marker.pose.orientation.z = self.simulator.terrain_quat[i][3]
            
            marker.scale.x = self.simulator.terrain_size[i][0]
            marker.scale.y = self.simulator.terrain_size[i][1]
            marker.scale.z = self.simulator.terrain_size[i][2]
            
            marker.color.r = float(self.simulator.terrain_rgba[i][0])
            marker.color.g = float(self.simulator.terrain_rgba[i][1])
            marker.color.b = float(self.simulator.terrain_rgba[i][2])
            marker.color.a = float(self.simulator.terrain_rgba[i][3])
            
            marker_array.markers.append(marker)
        
        self.marker_array_pub.publish(marker_array)
