from __future__ import annotations
from sensor_msgs.msg import Imu as ImuMsg
from geometry_msgs.msg import TransformStamped
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..mujoco_simulator_python import mujoco_simulator

from .base import BasePlugin


class Imu2(BasePlugin):
    """第二套IMU数据发布插件

    从 MuJoCo 传感器列表中查找名称前缀为 imu2 的四元数、角速度、线加速度，
    并发布为 ROS IMU 消息及对应 TF。
    """

    def __init__(self, plugin_config: dict, simulator: mujoco_simulator):
        """初始化第二套IMU插件"""
        super().__init__(plugin_config, simulator)

        self.imu_topic = plugin_config.get("imu2_topic", "/secondary_imu/converted")
        self.frame_id = plugin_config.get("frame_id", "imu2_frame")
        self.g_unit = plugin_config.get("g_unit", "g")
        self.sensor_prefix = plugin_config.get("sensor_prefix", "imu2")

        self.imu_pub = self.simulator.create_publisher(ImuMsg, self.imu_topic, 10)
        self.imu_msg = ImuMsg()

        self.imu2_quat_head_id = self._find_sensor_head("imu quat", expected_suffix="_w")
        self.imu2_gyro_head_id = self._find_sensor_head("imu gyro", expected_suffix="_x")
        self.imu2_acc_head_id = self._find_sensor_head("imu linear acc", expected_suffix="_x")

        self._sensor_ready = all(idx >= 0 for idx in [
            self.imu2_quat_head_id,
            self.imu2_gyro_head_id,
            self.imu2_acc_head_id,
        ])

        if not self._sensor_ready:
            self.simulator.get_logger().error(
                f"未找到前缀为 {self.sensor_prefix} 的完整IMU传感器，请检查MJCF中的sensor配置"
            )

    def _find_sensor_head(self, sensor_kind: str, expected_suffix: Optional[str] = None) -> int:
        """在 sensor_type 列表中查找指定前缀与类型的起始索引"""
        prefix = f"{self.sensor_prefix}_"
        for i, sensor in enumerate(self.simulator.sensor_type):
            name, kind, _ = sensor
            if kind != sensor_kind:
                continue
            if not name.startswith(prefix):
                continue
            if expected_suffix is not None and not name.endswith(expected_suffix):
                continue
            return i
        return -1

    def execute(self):
        """执行第二套IMU数据发布"""
        if self.simulator.read_error_flag or not self._sensor_ready:
            return

        time_stamp = self.simulator.get_clock().now().to_msg()
        sensor_data = list(self.mj_data.sensordata)

        self.imu_msg.header.frame_id = self.frame_id
        self.imu_msg.header.stamp = time_stamp

        self.imu_msg.orientation.w = sensor_data[self.imu2_quat_head_id + 0]
        self.imu_msg.orientation.x = sensor_data[self.imu2_quat_head_id + 1]
        self.imu_msg.orientation.y = sensor_data[self.imu2_quat_head_id + 2]
        self.imu_msg.orientation.z = sensor_data[self.imu2_quat_head_id + 3]

        self.imu_msg.angular_velocity.x = sensor_data[self.imu2_gyro_head_id + 0]
        self.imu_msg.angular_velocity.y = sensor_data[self.imu2_gyro_head_id + 1]
        self.imu_msg.angular_velocity.z = sensor_data[self.imu2_gyro_head_id + 2]

        self.imu_msg.linear_acceleration.x = sensor_data[self.imu2_acc_head_id + 0]
        self.imu_msg.linear_acceleration.y = sensor_data[self.imu2_acc_head_id + 1]
        self.imu_msg.linear_acceleration.z = sensor_data[self.imu2_acc_head_id + 2]

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

        imu_transform = TransformStamped()
        imu_transform.header.stamp = time_stamp
        imu_transform.header.frame_id = self.simulator.first_link_name
        imu_transform.child_frame_id = self.frame_id
        imu_transform.transform.translation.x = 0.0
        imu_transform.transform.translation.y = 0.0
        imu_transform.transform.translation.z = 0.0
        imu_transform.transform.rotation.w = 1.0
        imu_transform.transform.rotation.x = 0.0
        imu_transform.transform.rotation.y = 0.0
        imu_transform.transform.rotation.z = 0.0
        self.simulator.tf_broadcaster.sendTransform(imu_transform)