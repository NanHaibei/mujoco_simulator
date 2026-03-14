from __future__ import annotations
import mujoco
from std_srvs.srv import Empty
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..mujoco_simulator_python import mujoco_simulator

from .base_plugin import BasePlugin


class SimulationControlPlugin(BasePlugin):
    """仿真控制插件
    
    负责仿真暂停/启动控制。
    """
    
    def __init__(self, name: str, plugin_config: dict, simulator: mujoco_simulator):
        """初始化仿真控制插件"""
        super().__init__(name, plugin_config, simulator)
        # 读取配置
        self.unpause_service_name = plugin_config.get("unPauseService", "/unpause")
        self.init_pause_flag = plugin_config.get("initPauseFlag", True)
        
        # 设置初始暂停状态
        self.simulator.pause = True if self.init_pause_flag else False
        
        # 创建仿真启动服务
        self.unpause_server = self.simulator.create_service(
            Empty, self.unpause_service_name, self.unpause_callback
        )
        
        self.simulator.get_logger().info(
            f"仿真控制插件已启用，初始暂停: {self.simulator.pause}, 服务: {self.unpause_service_name}"
        )
    
    def unpause_callback(self, request, response):
        """仿真启动回调"""
        self.simulator.get_logger().info("Unpause service called")
        self.simulator.pause = False
        if self.simulator.keyframe_count > 0:
            mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, 0)
        return response
    
    def execute(self):
        """执行函数 - 仿真控制在回调中执行，这里不需要做任何事"""
        pass