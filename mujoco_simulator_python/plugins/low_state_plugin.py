import copy
import numpy as np
from collections import deque
from sensor_msgs.msg import Imu
from mit_msgs.msg import MITLowState, MITJointCommand, MITJointCommands

from .base_plugin import BasePlugin


class LowStatePlugin(BasePlugin):
    """低状态发布插件
    
    负责发布机器人电机与IMU状态信息。
    同时负责初始化 low_state_msg 和 low_cmd_msg。
    """
    
    def init(self):
        """初始化低状态插件"""
        # 读取配置
        self.low_state_topic = self.simulator.param.get("lowStateTopic", "/low_state")
        self.imu2_topic = self.simulator.param.get("imu2Topic", "/secondary_imu/converted")
        
        # ==================== 初始化 low_state_msg ====================
        self.simulator.low_state_msg = MITLowState()
        self.simulator.low_state_msg.joint_states.position = [0.0 for _ in range(self.mj_model.nu)]
        self.simulator.low_state_msg.joint_states.velocity = [0.0 for _ in range(self.mj_model.nu)]
        self.simulator.low_state_msg.joint_states.effort = [0.0 for _ in range(self.mj_model.nu)]
        
        # ==================== 初始化 low_cmd_msg ====================
        self.simulator.low_cmd_msg = MITJointCommands()
        self.simulator.low_cmd_msg.commands = [MITJointCommand() for _ in range(self.mj_model.nu)]
        
        # ==================== 初始化 state_deque ====================
        self.simulator.state_deque = deque()
        
        # 读取延迟配置
        self.state_delay = self.simulator.param.get("stateDelay", 0)
        
        # 填充延迟队列
        for _ in range(self.state_delay):
            self.simulator.state_deque.append(copy.deepcopy(self.simulator.low_state_msg))
        
        # 创建发布者
        self.lowState_pub = self.simulator.create_publisher(MITLowState, self.low_state_topic, 10)
        self.imu2_pub = self.simulator.create_publisher(Imu, self.imu2_topic, 10)
        self.imu2_normalized_pub = self.simulator.create_publisher(
            Imu, self.imu2_topic + "_normalized", 10
        )
    
    def execute(self):
        """执行低状态发布"""
        if not self.enabled:
            return
        
        # 更新传感器数据列表
        self.simulator.sensor_data_list = list(self.mj_data.sensordata)
        
        # 如果读取错误标志为真，则不发布状态
        if self.simulator.read_error_flag:
            return
        
        # 更新电机状态
        self.simulator.low_state_msg.joint_states.position = self.simulator.sensor_data_list[
            self.simulator.joint_pos_head_id : self.simulator.joint_pos_head_id + self.mj_model.nu
        ]
        self.simulator.low_state_msg.joint_states.velocity = self.simulator.sensor_data_list[
            self.simulator.joint_vel_head_id : self.simulator.joint_vel_head_id + self.mj_model.nu
        ]
        self.simulator.low_state_msg.joint_states.effort = self.simulator.sensor_data_list[
            self.simulator.joint_tor_head_id : self.simulator.joint_tor_head_id + self.mj_model.nu
        ]
        
        # 更新IMU状态
        self.simulator.low_state_msg.imu.orientation.w = self.simulator.sensor_data_list[self.simulator.imu_quat_head_id + 0]
        self.simulator.low_state_msg.imu.orientation.x = self.simulator.sensor_data_list[self.simulator.imu_quat_head_id + 1]
        self.simulator.low_state_msg.imu.orientation.y = self.simulator.sensor_data_list[self.simulator.imu_quat_head_id + 2]
        self.simulator.low_state_msg.imu.orientation.z = self.simulator.sensor_data_list[self.simulator.imu_quat_head_id + 3]
        self.simulator.low_state_msg.imu.angular_velocity.x = self.simulator.sensor_data_list[self.simulator.imu_gyro_head_id + 0]
        self.simulator.low_state_msg.imu.angular_velocity.y = self.simulator.sensor_data_list[self.simulator.imu_gyro_head_id + 1]
        self.simulator.low_state_msg.imu.angular_velocity.z = self.simulator.sensor_data_list[self.simulator.imu_gyro_head_id + 2]
        self.simulator.low_state_msg.imu.linear_acceleration.x = self.simulator.sensor_data_list[self.simulator.imu_acc_head_id + 0]
        self.simulator.low_state_msg.imu.linear_acceleration.y = self.simulator.sensor_data_list[self.simulator.imu_acc_head_id + 1]
        self.simulator.low_state_msg.imu.linear_acceleration.z = self.simulator.sensor_data_list[self.simulator.imu_acc_head_id + 2]
        
        # 如果存在 imu2 传感器，则发布 imu2 数据
        if self.simulator.imu2_quat_head_id != 999999:
            imu2_msg = Imu()
            imu2_msg.header.stamp = self.simulator.get_clock().now().to_msg()
            imu2_msg.header.frame_id = "imu2_frame"
            imu2_msg.orientation.w = self.simulator.sensor_data_list[self.simulator.imu2_quat_head_id + 0]
            imu2_msg.orientation.x = self.simulator.sensor_data_list[self.simulator.imu2_quat_head_id + 1]
            imu2_msg.orientation.y = self.simulator.sensor_data_list[self.simulator.imu2_quat_head_id + 2]
            imu2_msg.orientation.z = self.simulator.sensor_data_list[self.simulator.imu2_quat_head_id + 3]
            imu2_msg.angular_velocity.x = self.simulator.sensor_data_list[self.simulator.imu2_gyro_head_id + 0]
            imu2_msg.angular_velocity.y = self.simulator.sensor_data_list[self.simulator.imu2_gyro_head_id + 1]
            imu2_msg.angular_velocity.z = self.simulator.sensor_data_list[self.simulator.imu2_gyro_head_id + 2]
            imu2_msg.linear_acceleration.x = self.simulator.sensor_data_list[self.simulator.imu2_acc_head_id + 0]
            imu2_msg.linear_acceleration.y = self.simulator.sensor_data_list[self.simulator.imu2_acc_head_id + 1]
            imu2_msg.linear_acceleration.z = self.simulator.sensor_data_list[self.simulator.imu2_acc_head_id + 2]
            
            # 添加噪声
            imu2_msg.angular_velocity.x += np.random.uniform(
                -self.simulator.param.get("noise_imu_angle_acc", 0.0),
                self.simulator.param.get("noise_imu_angle_acc", 0.0)
            )
            imu2_msg.angular_velocity.y += np.random.uniform(
                -self.simulator.param.get("noise_imu_angle_acc", 0.0),
                self.simulator.param.get("noise_imu_angle_acc", 0.0)
            )
            imu2_msg.angular_velocity.z += np.random.uniform(
                -self.simulator.param.get("noise_imu_angle_acc", 0.0),
                self.simulator.param.get("noise_imu_angle_acc", 0.0)
            )
            noisy_ori2 = self._add_quat_noise_uniform(
                np.array([
                    imu2_msg.orientation.w,
                    imu2_msg.orientation.x,
                    imu2_msg.orientation.y,
                    imu2_msg.orientation.z
                ]),
                angle_range=self.simulator.param.get("noise_imu_gravity", 0.0)
            )
            imu2_msg.orientation.w = float(noisy_ori2[0])
            imu2_msg.orientation.x = float(noisy_ori2[1])
            imu2_msg.orientation.y = float(noisy_ori2[2])
            imu2_msg.orientation.z = float(noisy_ori2[3])
            
            # 发布原始 IMU2 数据
            self.imu2_pub.publish(imu2_msg)
            
            # 发布归一化的 IMU2 数据
            imu2_normalized_msg = Imu()
            imu2_normalized_msg.header = imu2_msg.header
            imu2_normalized_msg.orientation = imu2_msg.orientation
            imu2_normalized_msg.angular_velocity = imu2_msg.angular_velocity
            imu2_normalized_msg.linear_acceleration.x = imu2_msg.linear_acceleration.x / 9.80665
            imu2_normalized_msg.linear_acceleration.y = imu2_msg.linear_acceleration.y / 9.80665
            imu2_normalized_msg.linear_acceleration.z = imu2_msg.linear_acceleration.z / 9.80665
            self.imu2_normalized_pub.publish(imu2_normalized_msg)
        
        # 给传感器添加噪声
        self.simulator.low_state_msg.joint_states.position = (
            np.array(self.simulator.low_state_msg.joint_states.position, dtype=float) +
            np.random.uniform(
                -self.simulator.param.get("noise_joint_pos", 0.0),
                self.simulator.param.get("noise_joint_pos", 0.0),
                self.mj_model.nu
            )
        ).tolist()
        self.simulator.low_state_msg.joint_states.velocity = (
            np.array(self.simulator.low_state_msg.joint_states.velocity, dtype=float) +
            np.random.uniform(
                -self.simulator.param.get("noise_joint_vel", 0.0),
                self.simulator.param.get("noise_joint_vel", 0.0),
                self.mj_model.nu
            )
        ).tolist()
        self.simulator.low_state_msg.imu.angular_velocity.x += np.random.uniform(
            -self.simulator.param.get("noise_imu_angle_acc", 0.0),
            self.simulator.param.get("noise_imu_angle_acc", 0.0)
        )
        self.simulator.low_state_msg.imu.angular_velocity.y += np.random.uniform(
            -self.simulator.param.get("noise_imu_angle_acc", 0.0),
            self.simulator.param.get("noise_imu_angle_acc", 0.0)
        )
        self.simulator.low_state_msg.imu.angular_velocity.z += np.random.uniform(
            -self.simulator.param.get("noise_imu_angle_acc", 0.0),
            self.simulator.param.get("noise_imu_angle_acc", 0.0)
        )
        noisy_ori = self._add_quat_noise_uniform(
            np.array([
                self.simulator.low_state_msg.imu.orientation.w,
                self.simulator.low_state_msg.imu.orientation.x,
                self.simulator.low_state_msg.imu.orientation.y,
                self.simulator.low_state_msg.imu.orientation.z
            ]),
            angle_range=self.simulator.param.get("noise_imu_gravity", 0.0)
        )
        self.simulator.low_state_msg.imu.orientation.w = float(noisy_ori[0])
        self.simulator.low_state_msg.imu.orientation.x = float(noisy_ori[1])
        self.simulator.low_state_msg.imu.orientation.y = float(noisy_ori[2])
        self.simulator.low_state_msg.imu.orientation.z = float(noisy_ori[3])
        
        # 更新时间戳
        self.simulator.low_state_msg.stamp = self.simulator.get_clock().now().to_msg()
        self.simulator.low_state_msg.joint_states.header.stamp = self.simulator.low_state_msg.stamp
        self.simulator.low_state_msg.imu.header.stamp = self.simulator.low_state_msg.stamp
        
        # 存储拷贝，避免共享引用
        self.simulator.state_deque.append(copy.deepcopy(self.simulator.low_state_msg))
        
        self.lowState_pub.publish(self.simulator.state_deque.popleft())
    
    def _add_quat_noise_uniform(self, q, angle_range=0.01):
        """
        给四元数添加均匀分布的小旋转噪声
        
        参数:
            q: ndarray, shape=(4,), 输入四元数 (w, x, y, z)，必须是单位四元数
            angle_range: float, 噪声角度范围（弧度）
        
        返回:
            noisy_q: ndarray, shape=(4,), 加了噪声并归一化后的四元数
        """
        # 随机旋转轴
        axis = np.random.randn(3)
        axis /= np.linalg.norm(axis)
        
        # 均匀噪声角度
        angle = np.random.uniform(-angle_range, angle_range)
        
        # 构造扰动四元数 dq
        half_sin = np.sin(angle / 2.0)
        dq = np.array([
            np.cos(angle / 2.0),
            axis[0] * half_sin,
            axis[1] * half_sin,
            axis[2] * half_sin
        ])
        
        # 四元数乘法 q * dq
        w1, x1, y1, z1 = q
        w2, x2, y2, z2 = dq
        noisy_q = np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
        
        # 归一化，保持单位四元数
        return noisy_q / np.linalg.norm(noisy_q)