from __future__ import annotations
from sensor_msgs.msg import Imu as ImuMsg
from geometry_msgs.msg import TransformStamped
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..mujoco_simulator_python import mujoco_simulator

from .base import BasePlugin


class Imu(BasePlugin):
    """IMU数据发布插件
    
    负责发布IMU数据用于感知模块，并发布IMU的TF变换。
    直接从MuJoCo传感器数据读取，不依赖其他插件。
    """
    
    def __init__(self, plugin_config: dict, simulator: mujoco_simulator):
        """初始化IMU插件"""
        super().__init__(plugin_config, simulator)
        # 读取配置参数
        self.imu_topic = plugin_config.get("imuTopic", "/imu")
        self.g_unit = plugin_config.get("g_unit", "g")

        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        
        # IMU发布者
        self.imu_pub = self.simulator.create_publisher(ImuMsg, self.imu_topic, qos)
        
        # 初始化IMU消息
        self.imu_msg = ImuMsg()
    
    def execute(self):
        """执行IMU数据发布"""
        # 检查传感器数据是否有效
        if self.simulator.read_error_flag:
            return
        
        time_stamp = self.simulator.get_clock().now().to_msg()
        
        # 更新传感器数据列表
        sensor_data = list(self.mj_data.sensordata)
        
        # 从传感器数据直接读取IMU数据
        self.imu_msg.header.frame_id = "imu_frame"
        self.imu_msg.header.stamp = time_stamp
        
        # 四元数 (w, x, y, z)
        self.imu_msg.orientation.w = sensor_data[self.simulator.imu_quat_head_id + 0]
        self.imu_msg.orientation.x = sensor_data[self.simulator.imu_quat_head_id + 1]
        self.imu_msg.orientation.y = sensor_data[self.simulator.imu_quat_head_id + 2]
        self.imu_msg.orientation.z = sensor_data[self.simulator.imu_quat_head_id + 3]
        
        # 角速度
        self.imu_msg.angular_velocity.x = sensor_data[self.simulator.imu_gyro_head_id + 0]
        self.imu_msg.angular_velocity.y = sensor_data[self.simulator.imu_gyro_head_id + 1]
        self.imu_msg.angular_velocity.z = sensor_data[self.simulator.imu_gyro_head_id + 2]
        
        # 线性加速度
        self.imu_msg.linear_acceleration.x = sensor_data[self.simulator.imu_acc_head_id + 0]
        self.imu_msg.linear_acceleration.y = sensor_data[self.simulator.imu_acc_head_id + 1]
        self.imu_msg.linear_acceleration.z = sensor_data[self.simulator.imu_acc_head_id + 2]
        
        # 根据重力单位进行转换
        if self.g_unit == "g":
            self.imu_msg.linear_acceleration.x /= 9.80665
            self.imu_msg.linear_acceleration.y /= 9.80665
            self.imu_msg.linear_acceleration.z /= 9.80665
        elif self.g_unit == "m/s^2":
            pass
        else:
            self.simulator.get_logger().error(
                f"未知的重力单位: {self.g_unit}, 请检查参数设置"
            )
            return
        
        self.imu_pub.publish(self.imu_msg)
        
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