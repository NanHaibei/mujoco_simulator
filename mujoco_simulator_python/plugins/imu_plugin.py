from __future__ import annotations
from sensor_msgs.msg import Imu
from geometry_msgs.msg import TransformStamped
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..mujoco_simulator_python import mujoco_simulator

from .base_plugin import BasePlugin


class ImuPlugin(BasePlugin):
    """IMU数据发布插件
    
    负责发布IMU数据用于感知模块，并发布IMU的TF变换。
    """
    
    def __init__(self, name: str, plugin_config: dict, simulator: mujoco_simulator):
        """初始化IMU插件"""
        super().__init__(name, plugin_config, simulator)
        # 读取配置参数
        self.imu_topic = plugin_config.get("imuTopic", "/imu")
        self.g_unit = plugin_config.get("g_unit", "g")
        
        # IMU发布者
        self.imu_pub = self.simulator.create_publisher(Imu, self.imu_topic, 10)
    
    def execute(self):
        """执行IMU数据发布"""
        if not self.enabled:
            return
        
        time_stamp = self.simulator.get_clock().now().to_msg()
        
        # 获取IMU数据
        imu_data_msg = self.simulator.low_state_msg.imu
        imu_data_msg.header.frame_id = "imu_frame"
        imu_data_msg.header.stamp = time_stamp
        
        # 根据重力单位进行转换
        if self.g_unit == "g":
            imu_data_msg.linear_acceleration.x /= 9.80665
            imu_data_msg.linear_acceleration.y /= 9.80665
            imu_data_msg.linear_acceleration.z /= 9.80665
        elif self.g_unit == "m/s^2":
            pass
        else:
            self.simulator.get_logger().error(
                f"未知的重力单位: {self.g_unit}, 请检查参数设置"
            )
            return
        
        self.imu_pub.publish(imu_data_msg)
        
        # 发布IMU的TF变换
        imu_transform = TransformStamped()
        imu_transform.header.stamp = time_stamp
        imu_transform.header.frame_id = self.simulator.first_link_name
        imu_transform.child_frame_id = "imu_frame"
        imu_transform.transform.translation.x = 0.0
        imu_transform.transform.translation.y = 0.0
        imu_transform.transform.translation.z = 0.0
        imu_transform.transform.rotation.w = 1.0
        imu_transform.transform.rotation.x = 0.0
        imu_transform.transform.rotation.y = 0.0
        imu_transform.transform.rotation.z = 0.0
        self.simulator.tf_broadcaster.sendTransform(imu_transform)