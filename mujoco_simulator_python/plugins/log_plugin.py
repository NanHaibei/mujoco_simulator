from .base_plugin import BasePlugin


class LogPlugin(BasePlugin):
    """日志输出插件
    
    负责输出仿真运行时的日志信息。
    """
    
    def init(self):
        """初始化日志插件"""
        pass
    
    def execute(self):
        """执行日志输出"""
        if not self.enabled:
            return
        
        # 检查是否启用log输出
        if not self.simulator.param.get("enableLogOutput", True):
            return
        
        # 输出step耗时统计
        if self.simulator.step_count > 0:
            mean_time = self.simulator.step_time_sum / self.simulator.step_count
            self.simulator.get_logger().info(
                f"runtime[min/mean/max] {self.simulator.step_time_min:.2f}/{mean_time:.2f}/{self.simulator.step_time_max:.2f} ms"
            )