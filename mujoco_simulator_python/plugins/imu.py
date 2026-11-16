"""读取 IMU 传感器，发送接口暂时留空。"""

import mujoco
from .base import BasePlugin


class Imu(BasePlugin):
    default_prefix = "imu"

    def __init__(self, plugin_config, simulator):
        super().__init__(plugin_config, simulator)
        self.prefix = plugin_config.get("sensor_prefix", self.default_prefix)
        self.g_unit = plugin_config.get("g_unit", "g")
        if self.g_unit not in ("g", "m/s^2"):
            raise ValueError(f"未知的重力单位: {self.g_unit}")
        self.sensor_ids = {}
        for key, kind in (("orientation", mujoco.mjtSensor.mjSENS_FRAMEQUAT),
                          ("angular_velocity", mujoco.mjtSensor.mjSENS_GYRO),
                          ("linear_acceleration", mujoco.mjtSensor.mjSENS_ACCELEROMETER)):
            self.sensor_ids[key] = next((i for i in range(self.mj_model.nsensor)
                if int(self.mj_model.sensor_type[i]) == int(kind)
                and (mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, i) or "").startswith(self.prefix + "_")), None)
        self.enabled = all(i is not None for i in self.sensor_ids.values())
        self.last_state = None
        if not self.enabled:
            self.simulator.get_logger().warning("未找到完整的 %s 传感器，IMU 采样禁用", self.prefix)

    def after_step(self):
        """按步间隔采样当前 IMU 状态。"""
        if not self.should_execute():
            return
        if not self.enabled:
            return
        self.last_state = {"time": float(self.mj_data.time)}
        for key, i in self.sensor_ids.items():
            start = self.mj_model.sensor_adr[i]
            self.last_state[key] = self.mj_data.sensordata[start:start + self.mj_model.sensor_dim[i]].copy()
        if self.g_unit == "g":
            self.last_state["linear_acceleration"] /= 9.80665
        self.send(self.last_state)

    def send(self, state):
        """发送本插件的采样结果，外部通信待本插件接入。"""
        pass

    def reset(self):
        self.last_state = None
