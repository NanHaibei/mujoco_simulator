from __future__ import annotations
import mujoco
import numpy as np
from collections import deque
from mit_msgs.msg import MITJointCommands, MITJointCommand
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..mujoco_simulator_python import mujoco_simulator

from .base import BasePlugin


class LowCommand(BasePlugin):
    """低级命令插件
    
    负责初始化 low_cmd_msg，接收控制命令并计算关节力矩。
    """
    
    def __init__(self, plugin_config: dict, simulator: mujoco_simulator):
        """初始化低级命令插件"""
        super().__init__(plugin_config, simulator)
        # 读取配置
        self.joint_commands_topic = plugin_config.get("jointCommandsTopic", "/joint_commands")
        self.cmd_delay = plugin_config.get("cmdDelay", 0)
        
        # ==================== 初始化 low_cmd_msg ====================
        self.low_cmd_msg = MITJointCommands()
        self.low_cmd_msg.commands = [MITJointCommand() for _ in range(self.mj_model.nu)]
        
        # 初始化命令延迟队列
        self.cmd_deque = deque()
        
        # 填充延迟队列
        for _ in range(self.cmd_delay):
            self.cmd_deque.append(self.low_cmd_msg)
        
        # 订阅控制器命令
        self.joint_command_sub = self.simulator.create_subscription(
            MITJointCommands, self.joint_commands_topic, self.low_cmd_callback, 10
        )
        
        # 注册mujoco控制回调
        mujoco.set_mjcb_control(self.pd_controller)
        
        self.simulator.get_logger().info(
            f"低级命令插件已启用，命令话题: {self.joint_commands_topic}, 延迟: {self.cmd_delay}"
        )
    
    def pd_controller(self, model, data):
        """mujoco控制回调，根据命令值计算力矩"""
        if self.low_cmd_msg is None:
            return
        
        kp_cmd_list = np.array([cmd.kp for cmd in self.low_cmd_msg.commands])
        kd_cmd_list = np.array([cmd.kd for cmd in self.low_cmd_msg.commands])
        pos_cmd_list = np.array([cmd.pos for cmd in self.low_cmd_msg.commands])
        vel_cmd_list = np.array([cmd.vel for cmd in self.low_cmd_msg.commands])
        eff_cmd_list = np.array([cmd.eff for cmd in self.low_cmd_msg.commands])
        
        sensor_pos = np.array(self.simulator.sensor_data_list[
            self.simulator.joint_pos_head_id : self.simulator.joint_pos_head_id + self.mj_model.nu
        ])
        sensor_vel = np.array(self.simulator.sensor_data_list[
            self.simulator.joint_vel_head_id : self.simulator.joint_vel_head_id + self.mj_model.nu
        ])
        
        ctrl_torque = kp_cmd_list * (pos_cmd_list - sensor_pos) + kd_cmd_list * (vel_cmd_list - sensor_vel) + eff_cmd_list
        data.ctrl = np.clip(ctrl_torque, -10000.0, 10000.0)
    
    def low_cmd_callback(self, msg: MITJointCommands):
        """控制器命令回调函数"""
        if self.simulator.read_error_flag:
            return
        if len(msg.commands) != self.mj_model.nu:
            self.simulator.get_logger().error(
                f"命令长度 {len(msg.commands)} 不等于模型关节数 {self.mj_model.nu}，请检查"
            )
            return
        
        # 添加到延迟队列
        self.cmd_deque.append(msg)
        self.low_cmd_msg = self.cmd_deque.popleft()
    
    def execute(self):
        """执行函数 - PD控制在回调中执行，这里不需要做任何事"""
        pass