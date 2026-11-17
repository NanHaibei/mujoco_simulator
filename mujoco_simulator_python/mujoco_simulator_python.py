"""MuJoCo 4.0 Studio 仿真入口，通信接口留待接入宇树 SDK。"""

import argparse
from collections import deque
from dataclasses import dataclass, field
import importlib
import logging
import math
from pathlib import Path
import time

import mujoco
from mujoco.studio import app, launch_web, messages, plugin, sim, sim_history, step_control
from mujoco.studio import viewer_protocol
from rich.console import Console
from rich.logging import RichHandler
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class OverlaySnapshot(messages.Snapshot):
    geoms: list[mujoco.MjvGeom]


class PluginVisualization(plugin.ViewerPlugin):
    @messages.handler
    def _on_overlay(self, snapshot: OverlaySnapshot):
        self.viewer.extra_geoms[:] = snapshot.geoms


@dataclass
class StepTimeStats:
    """统计计算耗时（毫秒），不包含实时节奏的 sleep。"""

    times: deque = field(default_factory=lambda: deque(maxlen=1000))
    min_val: float = float("inf")
    max_val: float = 0.0
    sum_val: float = 0.0
    count: int = 0
    _start_time: float = 0.0
    _step_time: float = 0.0

    def tic(self):
        """记录本次耗时统计的起始时刻。"""
        self._start_time = time.perf_counter()

    def toc(self):
        """结束本次计时，并更新累计耗时统计。

        最近样本、最小值、最大值和累计值以毫秒保存，返回值以秒表示。

        Returns:
            float: 本次计时经过的时间，单位为秒。
        """
        self._step_time = time.perf_counter() - self._start_time
        elapsed_ms = self._step_time * 1000.0
        self.times.append(elapsed_ms)
        self.min_val = min(self.min_val, elapsed_ms)
        self.max_val = max(self.max_val, elapsed_ms)
        self.sum_val += elapsed_ms
        self.count += 1
        return self._step_time

    @property
    def step_time(self):
        """获取最近一次计时的耗时。

        Returns:
            float: 最近一次计算耗时，单位为秒；尚未计时时为 0.0。
        """
        return self._step_time

    @property
    def mean(self):
        """计算全部累计计时样本的平均耗时。

        Returns:
            float: 平均耗时，单位为毫秒；没有样本时为 0.0。
        """
        return self.sum_val / self.count if self.count else 0.0

    def format_summary(self):
        """将累计耗时统计格式化为日志摘要。

        Returns:
            str: 按“最小值/平均值/最大值”排列的字符串，各值保留两位小数，
                单位为毫秒。
        """
        return f"{self.min_val:.2f}/{self.mean:.2f}/{self.max_val:.2f}"


class MujocoSimulator(step_control.StepControl):
    """替换 Studio 默认步进器，每个 StepEvent 最多执行一个物理步。"""

    def __init__(self, config_path=None):
        """读取 YAML 配置，加载 MuJoCo 模型并初始化仿真器。

        配置顶层使用 mujoco_simulator。配置中的相对 MJCF 路径以 YAML
        文件所在目录为基准解析，插件随后按配置中的顺序初始化。

        Args:
            config_path (str | pathlib.Path, optional): YAML 配置文件路径。
                默认为 None，调用时必须提供。
        """
        super().__init__()

        # 读取yaml文件
        if config_path is None:
            raise ValueError("必须提供配置文件路径 config_path")
        config_path = Path(config_path).expanduser().resolve()
        with config_path.open(encoding="utf-8") as stream:
            contents = yaml.safe_load(stream)
        self.param = contents["mujoco_simulator"]

        self.logger = logging.getLogger("mujoco_simulator")

        # 读取mjcf文件
        self.model_path = Path(self.param["mjcfPath"]).expanduser()
        if not self.model_path.is_absolute():
            self.model_path = config_path.parent / self.model_path
        self.model_path = self.model_path.resolve()

        # 日志输出相关变量
        self.enable_log_output = bool(self.param.get("enableLogOutput", False))
        self.log_output_interval = int(self.param.get("logOutputInterval", 50))

        # 初始化mjcf模型
        self._initialize_model(mujoco.MjModel.from_xml_path(str(self.model_path)))

    def get_logger(self):
        """获取仿真器使用的日志对象。

        Returns:
            logging.Logger: 名为 mujoco_simulator 的日志对象，供仿真器和插件使用。
        """
        return self.logger

    def _initialize_model(self, model, data=None):
        """绑定模型和仿真状态，设置步长并按配置初始化插件。

        重置步计数器、耗时统计和插件列表，最后执行 mj_forward 更新派生量
        与传感器读数。此方法用于程序启动时的模型初始化。

        Args:
            model (mujoco.MjModel): 已编译的 MuJoCo 模型。
            data (mujoco.MjData, optional): 与模型对应的仿真状态。
                默认为 None，此时根据 model 新建状态。
        """
        # 从mjcf路径中读取mj_model和mj_data，并设置仿真频率和渲染频率。
        self.mj_model = model
        self.mj_data = mujoco.MjData(model) if data is None else data
        sim_rate = float(self.param.get("simRate", 1.0 / model.opt.timestep))
        render_rate = float(self.param.get("renderRate", 60.0))

        model.opt.timestep = 1.0 / sim_rate
        self.render_decimation = max(1, round(sim_rate / render_rate))
        self.step_counter = 0
        self.step_stats = StepTimeStats()
        # TODO：确认这个flag是否必须
        self.read_error_flag = False

        # 初始化插件
        self.plugins = []
        for config in self.param.get("plugins", []):
            module_name, class_name = config["path"].split(":", 1)
            module = importlib.import_module(f".plugins.{module_name}", package=__package__)
            plugin_class = getattr(module, class_name)
            self.plugins.append(plugin_class(plugin_config=config, simulator=self))
        mujoco.mj_forward(model, self.mj_data)

    @messages.handler
    def _on_post_model(self, event: messages.PostModelEvent):
        """处理 Studio 初始模型加载通知并同步步进控制状态。

        执行父类的步进控制初始化，再将步进控制状态快照发送给 Viewer。

        Args:
            event (messages.PostModelEvent): 包含启动时模型、仿真状态和来源路径的
                模型加载完成事件。
        """
        super()._on_post_model(event)
        self.handle.send_to_viewer(self._snapshot())

    @messages.handler
    def reset_plugins(self, event: messages.ResetEvent):
        """在 Studio 重置物理状态后清空插件缓存和统计信息。

        实际物理状态已由 ViewerHandle 按事件中的关键帧设置完成重置。
        此处重置步计数器、耗时统计和插件缓存，并发送空几何体快照。

        Args:
            event (messages.ResetEvent): Studio 发出的重置通知；物理状态的
                重置已由 ViewerHandle 处理。
        """
        # ViewerHandle 已按 event.key 重置物理状态；这里清空插件缓存。
        self.step_counter = 0
        self.step_stats = StepTimeStats()
        for plugin in self.plugins:
            plugin.reset()
        self.handle.send_to_viewer(OverlaySnapshot([]))

    def _physics_step(self):
        """执行一个物理步，并更新传感器、插件和耗时统计。

        物理步开始时先递增步编号，再依次调用 before_step、mj_step、
        mj_forward 和 after_step；插件在自身接口中判断执行间隔。
        再增加耗时统计并按配置输出日志。计时不包括随后发送的几何体快照、
        实时节奏的 sleep，以及耗时统计日志自身的输出。
        """
        self.step_stats.tic()
        # 步前、步后使用同一个物理步编号，执行间隔由插件内部判断。
        self.step_counter += 1

        # 运行插件的before_step函数
        for plugin in self.plugins:
            plugin.before_step()
        mujoco.mj_step(self.mj_model, self.mj_data)

        # 更新派生量，使采样和画面对应当前 qpos/qvel。
        mujoco.mj_forward(self.mj_model, self.mj_data)
        # 运行插件的after_step函数和log函数
        for plugin in self.plugins:
            plugin.after_step()
            if self.enable_log_output:
                plugin.log()
        self.step_stats.toc()

        # 输出耗时统计日志
        if self.enable_log_output and self.step_counter % self.log_output_interval == 0:
            self.logger.info("runtime[min/mean/max] %s ms", self.step_stats.format_summary())

    def _geometry_snapshot(self):
        """收集所有插件生成的额外几何体并创建 Viewer 快照。

        Returns:
            OverlaySnapshot: 本次生成的几何体快照；使用新建的列表收集几何体，
                交给 Viewer 进行可视化。
        """
        # 新建几何体再发送快照，避免线程间共享可变对象。
        geoms = []
        for plugin in self.plugins:
            plugin.visualize(geoms)
        return OverlaySnapshot(geoms)

    @messages.handler
    def _on_step(self, event: messages.StepEvent):
        """处理 Studio 步进事件，执行暂停、单步和速度控制。

        暂停且没有单步请求时仅短暂让出执行时间；其他情况下最多执行一个
        物理步，按渲染抽取间隔发送几何体快照，再按速度百分比控制 sleep。

        Args:
            event (messages.StepEvent): 携带当前模型和仿真状态的 Studio 步进事件。
        """
        started = time.perf_counter()
        single_step = self.step_control.consume_single_step_request()

        # 检查是否暂停仿真
        paused = self.step_control.get_pause_state() != sim.PauseState.UNPAUSED
        if paused and not single_step:
            time.sleep(0.001)
            return
        
        # 执行一个物理步
        self._physics_step()

        # 渲染插件要绘制的图形
        if self.step_counter == 1 or self.step_counter % self.render_decimation == 0:
            self.handle.send_to_viewer(self._geometry_snapshot())

        # 安装物理步的实际耗时，按速度百分比计算 sleep 时间，保证仿真节奏。
        period = float(self.mj_model.opt.timestep) * 100.0 / self.step_control.get_speed()
        time.sleep(max(0.0, period - (time.perf_counter() - started)))

    def run(self, *, port=0):
        """启动 Studio Web Viewer 并阻塞运行仿真循环。

        注册当前仿真器作为唯一的物理步进插件，同时启用应用界面、历史记录
        和插件几何体可视化。网页地址由 Studio 输出，浏览器不自动打开。
        正常结束、Ctrl+C 或异常退出时，按初始化逆序调用所有插件的 close()。

        Args:
            port (int, optional): Web Viewer 的 HTTP 端口。
                默认为 0，由 Studio 自动选择可用端口。
        """
        try:
            launch_web.run(
                viewer_protocol.ViewerConfig(
                    title="G1 MuJoCo Simulator", gfx="web", http_port=port,
                    open_browser=False,
                ),
                model=self.mj_model,
                data=self.mj_data,
                # 本类已继承 StepControl，不能再加入 app.SIM_PLUGINS。
                plugins=[app.App(), self, sim_history.SimHistory(), PluginVisualization()],
            )
        finally:
            # 按初始化的逆序释放资源；单个插件失败不影响其他插件关闭。
            for plugin in reversed(self.plugins):
                try:
                    plugin.close()
                except Exception:
                    self.logger.exception("关闭插件 %s 失败", plugin.name)

def main(args=None):
    """解析命令行参数，读取指定配置并启动 Studio 仿真器。

    --config 只接受本项目 config 目录下已有的 YAML 文件名，必须显式提供。
    --port 用于选择 Web Viewer 的 HTTP 端口。

    Args:
        args (list[str], optional): 要解析的命令行参数列表。
            默认为 None，此时读取 sys.argv[1:]。
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=None,
        choices=sorted(path.name for path in (PROJECT_ROOT / "config").glob("*.yaml")),
        help="config 目录中的 YAML 文件名",
    )
    parser.add_argument("--port", type=int, default=0)
    options = parser.parse_args(args)
    if options.config is None:
        raise ValueError("请通过 --config 指定 config 目录中的 YAML 文件名")

    # 设置logger信息输出方式
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=Console(stderr=True), show_time=False, show_path=False)],
    )

    # 拼接yaml文件路径
    config_path = PROJECT_ROOT / "config" / options.config

    # 实例化并仿真类并开始仿真
    simulator = MujocoSimulator(config_path)
    simulator.run(port=options.port)


if __name__ == "__main__":
    import sys

    # 通过包内入口启动，保证插件和快照类型使用统一的模块路径。
    sys.path.insert(0, str(PROJECT_ROOT))
    from mujoco_simulator_python.mujoco_simulator_python import main

    main()
