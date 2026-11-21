"""通过宇树 SDK2 接收 G1 命令，在物理步前控制、物理步后发布状态。"""

from collections import deque
from copy import deepcopy
import math
import struct
from threading import Lock

import numpy as np

from .base import BasePlugin


class UnitreeG1SDK(BasePlugin):
    """收发 HG LowCmd_/LowState_；actuator 和传感器对应 SDK 的前 nu 个电机。

    插件配置支持 domain_id（默认 0）、interface（默认自动选择）、
    command_topic（默认 rt/lowcmd）和 state_topic（默认 rt/lowstate）。
    控制端停发后持续使用最后一条有效命令，尚无命令时也正常发布状态。
    cmdDelay/stateDelay 表示控制/状态更新次数，每 step_interval 个物理步更新一次。
    DDS 工厂由 SDK 在进程内初始化一次，其他 SDK 插件应使用相同配置。
    """

    def __init__(self, plugin_config, simulator):
        super().__init__(plugin_config, simulator)
        self.cmd_delay = int(plugin_config.get("cmdDelay", 0))
        self.state_delay = int(plugin_config.get("stateDelay", 0))
        if self.state_delay < 0:
            raise ValueError("stateDelay 不能小于 0")
        self.state_deque = deque()
        self.last_state = None

        self.command = None
        self.cmd_deque = deque()
        self._command_lock = Lock()
        self._latest_command = None

        # 延迟导入，未启用该插件时无需安装 SDK。
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber, ChannelPublisher
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
        from unitree_sdk2py.utils.crc import CRC

        domain_id = int(plugin_config.get("domain_id", 0))

        # 初始化SDK DDS 工厂，必须在所有 SDK 插件使用前调用一次。
        ChannelFactoryInitialize(domain_id, plugin_config.get("interface"))
        self._crc = CRC()
        self._make_low_state = unitree_hg_msg_dds__LowState_
        self.low_state_msg = self._make_low_state()
        self.subscriber = ChannelSubscriber(plugin_config.get("command_topic", "rt/lowcmd"), LowCmd_)
        self.publisher = ChannelPublisher(plugin_config.get("state_topic", "rt/lowstate"), LowState_)
        try:
            self.publisher.Init()
            self.subscriber.Init(self._on_lowcmd, 1)
        except BaseException:
            self.close()
            raise

    def _on_lowcmd(self, msg):
        """接收SDK发来的电机命令。校验并缓存 SDK 消息；DDS 回调线程不读写 MuJoCo 仿真状态。"""
        try:
            # 当前模型使用 pitch/roll 关节，不直接接受 A/B 并联电机命令。
            if msg.mode_pr != 0 or len(msg.motor_cmd) != 35:
                return
            if msg.crc != self._crc.Crc(msg):
                return
            command = deepcopy(msg)
        except (AttributeError, TypeError, ValueError, OverflowError, struct.error):
            # 无效消息不能让 SDK 接收线程退出，也不覆盖最后一条有效命令。
            return
        with self._command_lock:
            self._latest_command = command

    def before_step(self):
        """按控制间隔处理命令，直接读取 SDK 字段计算并写入 PD 力矩。"""
        if not self.should_execute():
            return

        # 获取最新的命令
        with self._command_lock:
            command = self._latest_command

        # 把最新命令塞进延迟队列种
        self.cmd_deque.append(command)
        if len(self.cmd_deque) > self.cmd_delay:
            self.command = self.cmd_deque.popleft()
        if self.command is None:
            return

        # 开始计算 PD 力矩
        pos_head = self.simulator.joint_pos_head_id
        vel_head = self.simulator.joint_vel_head_id
        sensor_pos = self.mj_data.sensordata[pos_head:pos_head + self.mj_model.nu]
        sensor_vel = self.mj_data.sensordata[vel_head:vel_head + self.mj_model.nu]
        motor_cmd = self.command.motor_cmd
        torque = np.zeros(self.mj_model.nu)
        for i in range(self.mj_model.nu):
            if motor_cmd[i].mode == 1:
                torque[i] = (
                    motor_cmd[i].kp * (motor_cmd[i].q - sensor_pos[i])
                    + motor_cmd[i].kd * (motor_cmd[i].dq - sensor_vel[i])
                    + motor_cmd[i].tau
                )

        limited = self.mj_model.actuator_ctrllimited.astype(bool)
        torque[limited] = np.clip(
            torque[limited],
            self.mj_model.actuator_ctrlrange[limited, 0],
            self.mj_model.actuator_ctrlrange[limited, 1],
        )
        self.mj_data.ctrl[:] = torque

    def after_step(self):
        """
        更新电机与 IMU 状态，添加噪声后按延迟配置发送。
        目前还是使用单线程，测试过改成多线程并没有明显提升。
        """
        if not self.should_execute():
            return

        # 更新传感器数据列表
        self.simulator.sensor_data_list = self.mj_data.sensordata.tolist()

        # 更新电机状态
        for i in range(self.mj_model.nu):
            self.low_state_msg.motor_state[i].mode = 1
            self.low_state_msg.motor_state[i].q = self.simulator.sensor_data_list[
                self.simulator.joint_pos_head_id + i
            ]
            self.low_state_msg.motor_state[i].dq = self.simulator.sensor_data_list[
                self.simulator.joint_vel_head_id + i
            ]
            self.low_state_msg.motor_state[i].tau_est = self.simulator.sensor_data_list[
                self.simulator.joint_tor_head_id + i
            ]

        # 更新 IMU 状态，四元数顺序为 w、x、y、z
        self.low_state_msg.imu_state.quaternion = self.simulator.sensor_data_list[
            self.simulator.imu_quat_head_id:self.simulator.imu_quat_head_id + 4
        ]
        self.low_state_msg.imu_state.gyroscope = self.simulator.sensor_data_list[
            self.simulator.imu_gyro_head_id:self.simulator.imu_gyro_head_id + 3
        ]
        self.low_state_msg.imu_state.accelerometer = self.simulator.sensor_data_list[
            self.simulator.imu_acc_head_id:self.simulator.imu_acc_head_id + 3
        ]

        # 给关节位置、速度添加噪声
        noise_joint_pos = float(self.plugin_config.get("noise_joint_pos", 0.0))
        noise_joint_vel = float(self.plugin_config.get("noise_joint_vel", 0.0))
        position_noise = np.random.uniform(-noise_joint_pos, noise_joint_pos, self.mj_model.nu)
        velocity_noise = np.random.uniform(-noise_joint_vel, noise_joint_vel, self.mj_model.nu)
        for i in range(self.mj_model.nu):
            self.low_state_msg.motor_state[i].q += float(position_noise[i])
            self.low_state_msg.motor_state[i].dq += float(velocity_noise[i])

        # 给 IMU 角速度添加噪声
        noise_imu_angle_acc = float(self.plugin_config.get("noise_imu_angle_acc", 0.0))
        self.low_state_msg.imu_state.gyroscope = (
            np.array(self.low_state_msg.imu_state.gyroscope, dtype=float)
            + np.random.uniform(-noise_imu_angle_acc, noise_imu_angle_acc, 3)
        ).tolist()

        # 给 IMU 姿态添加小旋转噪声
        noise_imu_gravity = float(self.plugin_config.get("noise_imu_gravity", 0.0))
        if noise_imu_gravity:
            self.low_state_msg.imu_state.quaternion = self._add_quat_noise_uniform(
                np.array(self.low_state_msg.imu_state.quaternion, dtype=float),
                angle_range=noise_imu_gravity,
            ).tolist()

        # 根据添加噪声后的四元数更新欧拉角，单位为弧度
        w, x, y, z = self.low_state_msg.imu_state.quaternion
        self.low_state_msg.imu_state.rpy = [
            math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
            math.asin(float(np.clip(2 * (w * y - z * x), -1.0, 1.0))),
            math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)),
        ]

        # 更新时间戳和 CRC，时间戳使用采样时的仿真毫秒数
        self.low_state_msg.tick = int(self.mj_data.time * 1000) & 0xFFFFFFFF
        self.low_state_msg.crc = self._crc.Crc(self.low_state_msg)

        # 存储消息副本，避免后续更新影响延迟队列中的历史状态
        self.state_deque.append(deepcopy(self.low_state_msg))
        if len(self.state_deque) <= self.state_delay:
            return

        # 发布延迟后的状态，并保存本次发布的 SDK 消息
        self.last_state = self.state_deque.popleft()
        self.publisher.Write(self.last_state)

    def reset(self):
        """清空命令、状态及延迟队列，撤销旧力矩；保留 SDK 收发通道。"""
        with self._command_lock:
            self._latest_command = None
        self.command = None
        self.cmd_deque.clear()
        self.state_deque.clear()
        self.last_state = None
        self.low_state_msg = self._make_low_state()
        self.mj_data.ctrl[:] = 0.0

    def close(self):
        """关闭 SDK 订阅和发布器，接收线程停止后清空命令缓存。"""
        try:
            self.subscriber.Close()
        finally:
            self.publisher.Close()
        with self._command_lock:
            self._latest_command = None

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
