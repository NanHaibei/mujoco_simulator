from __future__ import annotations
from tf2_msgs.msg import TFMessage
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..mujoco_simulator_python import mujoco_simulator

from .base import BasePlugin


class MapFrame(BasePlugin):
    """Map坐标系插件
    
    负责处理map坐标系的变换发布。
    监听/tf_static话题，在收到odom坐标系时发布world->map的静态变换。
    """
    
    def __init__(self, plugin_config: dict, simulator: mujoco_simulator):
        """初始化Map坐标系插件"""
        super().__init__(plugin_config, simulator)
        # 初始化状态
        self.map_triggered = False
        
        # 订阅tf信息（用于map坐标系）
        self.tf_sub = self.simulator.create_subscription(
            TFMessage, '/tf_static', self.map_tf_callback, 10
        )
        
        self.simulator.get_logger().info("Map坐标系插件已启用")
    
    def map_tf_callback(self, msg: TFMessage):
        """处理map坐标系变换"""
        if self.map_triggered:
            return
        
        for transform in msg.transforms:
            child = transform.child_frame_id
            if child == "odom":
                self.simulator.get_logger().info(f"第一次收到 {child} 的 tf，执行函数！")
                self.broadcaster = StaticTransformBroadcaster(self.simulator)

                t = TransformStamped()
                t.header.stamp = self.simulator.get_clock().now().to_msg()
                t.header.frame_id = 'world'
                t.child_frame_id = 'map'

                t.transform.translation.x = float(self.simulator.sensor_data_list[self.simulator.real_pos_head_id + 0])
                t.transform.translation.y = float(self.simulator.sensor_data_list[self.simulator.real_pos_head_id + 1])
                t.transform.translation.z = 0.0

                t.transform.rotation.w = float(self.simulator.sensor_data_list[self.simulator.imu_quat_head_id + 0])
                t.transform.rotation.x = float(self.simulator.sensor_data_list[self.simulator.imu_quat_head_id + 1])
                t.transform.rotation.y = float(self.simulator.sensor_data_list[self.simulator.imu_quat_head_id + 2])
                t.transform.rotation.z = float(self.simulator.sensor_data_list[self.simulator.imu_quat_head_id + 3])

                self.broadcaster.sendTransform(t)
                self.simulator.get_logger().info("发布了静态坐标变换 world -> map")
                self.map_triggered = True
                break
    
    def execute(self):
        """执行函数 - map坐标系变换在回调中执行，这里不需要做任何事"""
        pass