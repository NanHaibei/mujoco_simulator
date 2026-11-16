"""采样世界坐标系下的位姿和速度，发送接口暂时留空。"""

from .base import BasePlugin


class Odom(BasePlugin):
    def __init__(self, plugin_config, simulator):
        super().__init__(plugin_config, simulator)
        self.last_state = None

    def after_step(self):
        """按步间隔采样当前世界坐标系下的位姿和速度。"""
        if not self.should_execute():
            return
        sim = self.simulator
        if min(sim.real_pos_head_id, sim.real_vel_head_id, sim.imu_quat_head_id, sim.imu_gyro_head_id) < 0:
            return
        sensor = self.mj_data.sensordata
        self.last_state = {
            "time": float(self.mj_data.time),
            "position": sensor[sim.real_pos_head_id:sim.real_pos_head_id + 3].copy(),
            "orientation": sensor[sim.imu_quat_head_id:sim.imu_quat_head_id + 4].copy(),
            "linear_velocity": sensor[sim.real_vel_head_id:sim.real_vel_head_id + 3].copy(),
            "angular_velocity": sensor[sim.imu_gyro_head_id:sim.imu_gyro_head_id + 3].copy(),
        }
        self.send(self.last_state)

    def send(self, state):
        """发送本插件的采样结果，外部通信待本插件接入。"""
        pass

    def reset(self):
        self.last_state = None
