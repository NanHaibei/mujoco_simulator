from __future__ import annotations
import mujoco
import numpy as np
from collections import deque
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
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
        self.low_cmd_msg.cmds.kp = [0.0 for _ in range(self.mj_model.nu)]
        self.low_cmd_msg.cmds.kd = [0.0 for _ in range(self.mj_model.nu)]
        self.low_cmd_msg.cmds.pos = [0.0 for _ in range(self.mj_model.nu)]
        self.low_cmd_msg.cmds.vel = [0.0 for _ in range(self.mj_model.nu)]
        self.low_cmd_msg.cmds.eff = [0.0 for _ in range(self.mj_model.nu)]
        
        # 初始化命令延迟队列
        self.cmd_deque = deque()
        
        # 填充延迟队列
        for _ in range(self.cmd_delay):
            self.cmd_deque.append(self.low_cmd_msg)

        mit_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        
        # 订阅控制器命令
        self.joint_command_sub = self.simulator.create_subscription(
            MITJointCommands, self.joint_commands_topic, self.low_cmd_callback, mit_qos
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
        
        kp_cmd_list = np.array(self.low_cmd_msg.cmds.kp)
        kd_cmd_list = np.array(self.low_cmd_msg.cmds.kd)
        pos_cmd_list = np.array(self.low_cmd_msg.cmds.pos)
        vel_cmd_list = np.array(self.low_cmd_msg.cmds.vel)
        eff_cmd_list = np.array(self.low_cmd_msg.cmds.eff)
        
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
        if len(msg.cmds.pos) != self.mj_model.nu:
            self.simulator.get_logger().error(
                f"命令长度 {len(msg.cmds.pos)} 不等于模型关节数 {self.mj_model.nu}，请检查"
            )
            return
        
        # 添加到延迟队列
        self.cmd_deque.append(msg)
        self.low_cmd_msg = self.cmd_deque.popleft()
    
    def execute(self):
        """执行函数 - PD控制在回调中执行，这里不需要做任何事"""
        pass