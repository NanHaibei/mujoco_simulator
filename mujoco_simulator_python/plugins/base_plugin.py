from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..mujoco_simulator_python import mujoco_simulator


class BasePlugin(ABC):
    """MuJoCo仿真器插件基类
    
    所有插件必须继承此类并实现 execute() 方法。
    插件在仿真主循环中被调用，可以根据 step_interval 控制执行频率。
    """
    
    def __init__(self, name: str, plugin_config: dict, simulator: mujoco_simulator):
        """
        初始化插件
        
        Args:
            name: 插件名称
            plugin_config: 插件配置参数（从YAML读取）
            simulator: mujoco_simulator 实例的引用
        """
        self.name = name
        self.plugin_config = plugin_config
        self.simulator = simulator
        self.mj_model = simulator.mj_model
        self.mj_data = simulator.mj_data
        self.step_interval = plugin_config["step_interval"]
        self.enabled = plugin_config["enabled"]
    
    @abstractmethod
    def execute(self):
        """
        插件执行函数，子类必须实现
        
        在仿真主循环中被调用，执行插件的核心功能。
        """
        pass
    
    def should_execute(self) -> bool:
        """
        判断当前步是否应该执行
        
        使用全局 step_counter 判断
        
        Returns:
            bool: 是否执行
        """
        return self.simulator.step_counter % self.step_interval == 0
    
    def update(self):
        """
        更新函数，在主循环中调用
        
        内部会判断是否需要执行，如果需要则调用 execute()。
        """
        if self.enabled and self.should_execute():
            self.execute()
    
    def visualize(self, viewer):
        """
        可视化函数（可选）
        
        在渲染时调用，用于在viewer中绘制可视化内容。
        子类可以重写此方法实现自定义可视化。
        
        Args:
            viewer: mujoco viewer实例
        """
        pass
    
    def log(self):
        """
        日志输出函数（可选）
        
        在execute()执行后调用，用于输出插件的调试/状态日志。
        子类可以重写此方法实现自定义日志输出。
        只有当 simulator.enable_log_output 为 True 时才会被调用。
        """
        pass
