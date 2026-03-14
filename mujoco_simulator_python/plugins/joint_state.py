from __future__ import annotations
from sensor_msgs.msg import JointState
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..mujoco_simulator_python import mujoco_simulator

from .base import BasePlugin


class JointStates(BasePlugin):
    """关节状态发布插件
    
    负责发布关节状态用于可视化（如 rviz）。
    """
    
    def __init__(self, plugin_config: dict, simulator: mujoco_simulator):
        """初始化关节状态插件"""
        super().__init__(plugin_config, simulator)
        # 创建发布者
        self.joint_state_pub = self.simulator.create_publisher(
            JointState, "/joint_states", 10
        )
    
    def execute(self):
        """发布关节状态"""
        if self.simulator.read_error_flag:
            return
        
        joint_state = JointState()
        joint_state.header.stamp = self.simulator.get_clock().now().to_msg()
        joint_state.name = self.simulator.joint_name.copy()
        joint_state.position = self.simulator.sensor_data_list[
            self.simulator.joint_pos_head_id : self.simulator.joint_pos_head_id + self.mj_model.nu
        ]
        joint_state.velocity = self.simulator.sensor_data_list[
            self.simulator.joint_vel_head_id : self.simulator.joint_vel_head_id + self.mj_model.nu
        ]
        joint_state.effort = self.simulator.sensor_data_list[
            self.simulator.joint_tor_head_id : self.simulator.joint_tor_head_id + self.mj_model.nu
        ]
        self.joint_state_pub.publish(joint_state)
