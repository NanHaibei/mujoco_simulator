"""通过宇树 SDK2 收发 B2-W 底盘的低层控制与状态。"""

from collections import deque
from copy import deepcopy
import math
import struct
from threading import Lock

import mujoco
import numpy as np

from .base import BasePlugin


_LEG_ORDER = ("FR", "FL", "RR", "RL")
_MOTOR_COUNT = 16
_DDS_MOTOR_COUNT = 20
_SERVO_MODE = 0x01
_POS_STOP = 2.146e9
_VEL_STOP = 16000.0
_NOISE_KEYS = ("noise_joint_pos", "noise_joint_vel", "noise_imu_angle_acc", "noise_imu_gravity")


class UnitreeB2WSDK(BasePlugin):
    """直接使用 MJCF 的前 16 个执行器，对应 DDS 的电机索引 0 到 15。

    先按 FR、FL、RR、RL 各排列 hip、thigh、calf，再按相同顺序排列四个轮子。
    ModelInfo 需先加载。关节位置、速度、力矩传感器各自连续排列，且每组的
    前 16 个传感器与执行器同序。Piper 的执行器放在底盘执行器之后。
    默认订阅 rt/lowcmd，发布 rt/lowstate。cmdDelay 和 stateDelay 的单位为
    插件更新次数，控制端停发时继续使用最后一条有效命令。支持可选的观测噪声配置
    noise_joint_pos、noise_joint_vel、noise_imu_angle_acc 和 noise_imu_gravity
    （默认均为 0.0），噪声仅作用于对外发布的 LowState_，不修改 mj_data.sensordata。
    """

    def __init__(self, plugin_config, simulator):
        super().__init__(plugin_config, simulator)
        self.cmd_delay = int(plugin_config.get("cmdDelay", 0))
        self.state_delay = int(plugin_config.get("stateDelay", 0))
        if min(self.cmd_delay, self.state_delay) < 0:
            raise ValueError("cmdDelay 和 stateDelay 必须大于等于 0")
        for key in _NOISE_KEYS:
            self._noise_level(key)
        self._bind_sensors()

        limits = self.mj_model.actuator_ctrlrange[:_MOTOR_COUNT]
        limited = self.mj_model.actuator_ctrllimited[:_MOTOR_COUNT]
        self._ctrl_min = np.where(limited, limits[:, 0], -np.inf)
        self._ctrl_max = np.where(limited, limits[:, 1], np.inf)

        self._command_lock = Lock()
        self._latest_command = None
        self._command_queue = deque()
        self._state_queue = deque()
        self.last_state = None

        # 只有启用本插件才需要宇树 SDK2 和 CycloneDDS。
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC

        self._crc = CRC()
        self._make_low_state = unitree_go_msg_dds__LowState_
        ChannelFactoryInitialize(int(plugin_config.get("domain_id", 0)), plugin_config.get("interface"))
        self.publisher = None
        self.subscriber = None
        try:
            self.publisher = ChannelPublisher(plugin_config.get("state_topic", "rt/lowstate"), LowState_)
            self.publisher.Init()
            self.subscriber = ChannelSubscriber(plugin_config.get("command_topic", "rt/lowcmd"), LowCmd_)
            self.subscriber.Init(self._on_lowcmd, 1)
        except BaseException:
            try:
                self.close()
            except Exception:
                self.simulator.get_logger().exception("关闭 B2-W DDS 通道失败")
            raise

    def _bind_sensors(self):
        """校验底盘执行器和传感器的顺序，保存 sensordata 切片。"""
        model = self.mj_model
        joint_ids = model.actuator_trnid[:_MOTOR_COUNT, 0]
        joint_transmissions = model.actuator_trntype[:_MOTOR_COUNT] == mujoco.mjtTrn.mjTRN_JOINT
        if model.nu < _MOTOR_COUNT or not joint_transmissions.all():
            raise ValueError("MJCF 的前 16 个执行器必须是 B2-W 底盘关节电机")

        joint_names = [model.joint(int(joint_id)).name for joint_id in joint_ids]
        leg_joints = [f"{leg}_{part}_joint" for leg in _LEG_ORDER for part in ("hip", "thigh", "calf")]
        wheel_joints = [f"{leg}_wheel_joint" for leg in _LEG_ORDER]
        foot_joints = [f"{leg}_foot_joint" for leg in _LEG_ORDER]
        if joint_names[:12] != leg_joints or joint_names[12:] not in (wheel_joints, foot_joints):
            raise ValueError("B2-W 执行器应先按 FR、FL、RR、RL 排列腿关节，再按相同顺序排列轮关节")

        self._position = self._joint_sensor_slice("joint_pos_head_id", mujoco.mjtSensor.mjSENS_JOINTPOS, joint_ids)
        self._velocity = self._joint_sensor_slice("joint_vel_head_id", mujoco.mjtSensor.mjSENS_JOINTVEL, joint_ids)
        self._torque = self._joint_sensor_slice("joint_tor_head_id", mujoco.mjtSensor.mjSENS_JOINTACTFRC, joint_ids)

        imu = self.simulator
        self._quaternion = slice(imu.imu_quat_head_id, imu.imu_quat_head_id + 4)
        self._gyroscope = slice(imu.imu_gyro_head_id, imu.imu_gyro_head_id + 3)
        self._accelerometer = slice(imu.imu_acc_head_id, imu.imu_acc_head_id + 3)

    def _joint_sensor_slice(self, head_attr, sensor_type, joint_ids):
        start = getattr(self.simulator, head_attr, -1)
        if start < 0:
            raise ValueError("UnitreeB2WSDK 需要先加载 ModelInfo 并提供底盘关节传感器")

        model = self.mj_model
        stop = start + _MOTOR_COUNT
        sensor_ids = np.flatnonzero((model.sensor_adr >= start) & (model.sensor_adr < stop))
        if not (
            np.array_equal(model.sensor_adr[sensor_ids], np.arange(start, stop))
            and np.all(model.sensor_type[sensor_ids] == sensor_type)
            and np.array_equal(model.sensor_objid[sensor_ids], joint_ids)
        ):
            raise ValueError(f"MJCF 的 {sensor_type.name} 传感器必须连续排列，并与 B2-W 执行器顺序一致")
        return slice(start, stop)

    def _on_lowcmd(self, msg):
        """校验并缓存命令，DDS 线程不读写仿真状态。"""
        try:
            if len(msg.motor_cmd) != _DDS_MOTOR_COUNT or msg.crc != self._crc.Crc(msg):
                return
            command = msg.motor_cmd[:_MOTOR_COUNT]
            if any(
                motor.mode == _SERVO_MODE
                and not all(map(math.isfinite, (motor.q, motor.dq, motor.kp, motor.kd, motor.tau)))
                for motor in command
            ):
                return
            command = deepcopy(command)
        except (AttributeError, TypeError, ValueError, OverflowError, struct.error):
            return
        with self._command_lock:
            self._latest_command = command

    def before_step(self):
        if not self.should_execute():
            return

        with self._command_lock:
            self._command_queue.append(self._latest_command)
        if len(self._command_queue) <= self.cmd_delay:
            return
        command = self._command_queue.popleft()
        if command is None:
            return

        position = self.mj_data.sensordata[self._position]
        velocity = self.mj_data.sensordata[self._velocity]
        torque = np.zeros(_MOTOR_COUNT)
        for index, motor in enumerate(command):
            if motor.mode != _SERVO_MODE:
                continue
            effort = motor.tau
            if motor.q != _POS_STOP:
                effort += motor.kp * (motor.q - position[index])
            if motor.dq != _VEL_STOP:
                effort += motor.kd * (motor.dq - velocity[index])
            torque[index] = effort
        self.mj_data.ctrl[:_MOTOR_COUNT] = np.clip(torque, self._ctrl_min, self._ctrl_max)

    def after_step(self):
        if not self.should_execute():
            return

        sensors = self.mj_data.sensordata
        positions = sensors[self._position].astype(float, copy=True)
        velocities = sensors[self._velocity].astype(float, copy=True)
        torques = sensors[self._torque].tolist()
        quaternion = sensors[self._quaternion].astype(float, copy=True)
        gyroscope = sensors[self._gyroscope].astype(float, copy=True)
        accelerometer = sensors[self._accelerometer].tolist()

        noise_joint_pos = self._noise_level("noise_joint_pos")
        if noise_joint_pos > 0.0:
            positions += np.random.uniform(-noise_joint_pos, noise_joint_pos, _MOTOR_COUNT)

        noise_joint_vel = self._noise_level("noise_joint_vel")
        if noise_joint_vel > 0.0:
            velocities += np.random.uniform(-noise_joint_vel, noise_joint_vel, _MOTOR_COUNT)

        noise_imu_angle_acc = self._noise_level("noise_imu_angle_acc")
        if noise_imu_angle_acc > 0.0:
            gyroscope += np.random.uniform(-noise_imu_angle_acc, noise_imu_angle_acc, 3)

        noise_imu_gravity = self._noise_level("noise_imu_gravity")
        if noise_imu_gravity > 0.0:
            quaternion = self._add_quat_noise_uniform(quaternion, noise_imu_gravity)

        state = self._make_low_state()
        for motor, position, velocity, torque in zip(
            state.motor_state[:_MOTOR_COUNT],
            positions.tolist(),
            velocities.tolist(),
            torques,
            strict=True,
        ):
            motor.mode = _SERVO_MODE
            motor.q, motor.dq, motor.tau_est = position, velocity, torque

        imu = state.imu_state
        imu.quaternion = quaternion.tolist()
        imu.gyroscope = gyroscope.tolist()
        imu.accelerometer = accelerometer
        w, x, y, z = imu.quaternion
        pitch = 2 * (w * y - z * x)
        imu.rpy = [
            math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
            math.asin(max(-1.0, min(1.0, pitch))),
            math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)),
        ]
        state.tick = round(self.mj_data.time * 1000) & 0xFFFFFFFF
        state.crc = self._crc.Crc(state)

        self._state_queue.append(state)
        if len(self._state_queue) > self.state_delay:
            self.last_state = self._state_queue.popleft()
            self.publisher.Write(self.last_state)

    def _noise_level(self, key: str) -> float:
        """读取并校验非负有限的噪声幅度配置。"""
        value = float(self.plugin_config.get(key, 0.0))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{key} 必须为大于等于 0 的有限数值")
        return value

    def _add_quat_noise_uniform(self, q: np.ndarray, angle_range: float) -> np.ndarray:
        """给单位四元数 (w, x, y, z) 添加均匀分布的小角度随机旋转噪声。"""
        axis = np.random.randn(3)
        norm = float(np.linalg.norm(axis))
        while norm == 0.0:
            axis = np.random.randn(3)
            norm = float(np.linalg.norm(axis))
        axis /= norm

        angle = float(np.random.uniform(-angle_range, angle_range))
        half_angle = angle * 0.5
        half_sin = math.sin(half_angle)
        dq = np.array([
            math.cos(half_angle),
            axis[0] * half_sin,
            axis[1] * half_sin,
            axis[2] * half_sin,
        ])

        w1, x1, y1, z1 = q
        w2, x2, y2, z2 = dq
        noisy_q = np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ])
        return noisy_q / np.linalg.norm(noisy_q)

    def reset(self):
        """清空命令与状态缓存，只归零底盘控制量。"""
        with self._command_lock:
            self._latest_command = None
        self._command_queue.clear()
        self._state_queue.clear()
        self.last_state = None
        self.mj_data.ctrl[:_MOTOR_COUNT] = 0.0

    def close(self):
        """关闭 DDS 收发通道。"""
        subscriber, self.subscriber = self.subscriber, None
        publisher, self.publisher = self.publisher, None
        try:
            if subscriber is not None:
                subscriber.Close()
        finally:
            if publisher is not None:
                publisher.Close()
