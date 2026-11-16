from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..mujoco_simulator_python import MujocoSimulator


def _class_name_to_plugin_name(class_name: str) -> str:
    """将类名转换为下划线分隔的插件名，例如 UnitreeG1SDK 转为 unitree_g1_sdk。"""
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', class_name)
    s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)
    return s2.lower()


class BasePlugin:
    """仿真插件基类，提供可选的步前、步后和辅助接口。

    主循环每个物理步都会调用 before_step() 和 after_step()。
    插件只需重写有业务的接口，并在接口内部用 should_execute() 判断间隔。
    通信收发方法由需要通信的子类自行定义，基类不提供 send/receive。
    """

    def __init__(self, plugin_config: dict, simulator: MujocoSimulator):
        """保存插件配置以及模型、仿真状态的引用。

        Args:
            plugin_config (dict): YAML 中的插件配置，step_interval 表示物理步间隔。
            simulator (MujocoSimulator): 当前仿真器实例。
        """
        self.name = _class_name_to_plugin_name(self.__class__.__name__)
        self.plugin_config = plugin_config
        self.simulator = simulator
        self.mj_model = simulator.mj_model
        self.mj_data = simulator.mj_data
        self.step_interval = int(plugin_config.get("step_interval", 1))
        if self.step_interval <= 0:
            raise ValueError(f"{self.name}: step_interval 必须大于 0")

    def should_execute(self) -> bool:
        """判断当前物理步是否达到插件的执行间隔。

        主循环在物理步开始时递增 step_counter，步前、步后使用同一个步编号。
        例如 step_interval 为 3 时，在第 3、6、9 步执行。

        Returns:
            bool: 当前物理步是否为执行时刻。
        """
        return self.simulator.step_counter % self.step_interval == 0

    def before_step(self):
        """物理步前的可选接口，重写时在插件内部判断执行间隔。"""
        pass

    def after_step(self):
        """物理步后的可选接口，重写时在插件内部判断执行间隔。"""
        pass

    def reset(self):
        """重置插件缓存，可由插件覆盖。"""
        pass

    def close(self):
        """释放插件资源，可由插件覆盖。"""
        pass

    def visualize(self, geoms):
        """向 Studio 额外几何体列表追加插件的可视化内容。

        Args:
            geoms (list): 本次可视化快照的几何体列表。
        """
        pass

    def log(self):
        """按插件的物理步间隔输出可选的调试信息。

        主循环开启 enable_log_output 时调用此方法。子类重写时也应在方法
        内部判断 should_execute()，再执行自己的日志输出。
        """
        if not self.should_execute():
            return
