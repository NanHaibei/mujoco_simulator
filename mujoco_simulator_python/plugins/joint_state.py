"""采样关节状态，发送接口暂时留空。"""

import mujoco
import numpy as np
from .base import BasePlugin


class JointStates(BasePlugin):
    def __init__(self, plugin_config, simulator):
        super().__init__(plugin_config, simulator)
        self.joints = np.flatnonzero(np.isin(self.mj_model.jnt_type,
            [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE]))
        self.last_state = None

    def after_step(self):
        """按步间隔采样当前关节状态。"""
        if not self.should_execute():
            return
        qpos = self.mj_model.jnt_qposadr[self.joints]
        dofs = self.mj_model.jnt_dofadr[self.joints]
        self.last_state = {
            "time": float(self.mj_data.time),
            "name": [mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, int(j)) for j in self.joints],
            "position": self.mj_data.qpos[qpos].copy(),
            "velocity": self.mj_data.qvel[dofs].copy(),
            "effort": self.mj_data.qfrc_actuator[dofs].copy(),
        }
        self.send(self.last_state)

    def send(self, state):
        """发送本插件的采样结果，外部通信待本插件接入。"""
        pass

    def reset(self):
        self.last_state = None
