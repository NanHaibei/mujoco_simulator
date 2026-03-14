from abc import ABC, abstractmethod


class BasePlugin(ABC):
    """MuJoCo仿真器插件基类
    
    所有插件必须继承此类并实现 execute() 方法。
    插件在仿真主循环中被调用，可以根据 step_interval 控制执行频率。
    """
    
    def __init__(self, name: str, plugin_config: dict, simulator):
        """
        初始化插件
        
        Args:
            name: 插件名称
            plugin_config: 插件配置参数（从YAML读取）
            simulator: mujoco_simulator 实例的引用
        """
        self.name = name
        self.config = plugin_config
        self.simulator = simulator
        self.mj_model = simulator.mj_model
        self.mj_data = simulator.mj_data
        self.step_interval = plugin_config.get("step_interval", 1)  # 执行间隔，默认每步都执行
        self.step_counter = 0
        self.enabled = plugin_config.get("enabled", True)
        
        # 调用子类的初始化方法
        self.init()
    
    def init(self):
        """子类重写此方法进行初始化
        
        在构造函数中被调用，用于初始化插件特有的资源。
        """
        pass
    
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
        
        Returns:
            bool: 是否执行
        """
        self.step_counter += 1
        if self.step_counter >= self.step_interval:
            self.step_counter = 0
            return True
        return False
    
    def update(self):
        """
        更新函数，在主循环中调用
        
        内部会判断是否需要执行，如果需要则调用 execute()。
        """
        if self.enabled and self.should_execute():
            self.execute()
    
    def reset(self):
        """
        重置插件状态
        
        子类可以重写此方法以在仿真重置时执行特定操作。
        """
        self.step_counter = 0
    
    def visualize(self, viewer):
        """
        可视化函数（可选）
        
        在渲染时调用，用于在viewer中绘制可视化内容。
        子类可以重写此方法实现自定义可视化。
        
        Args:
            viewer: mujoco viewer实例
        """
        pass
