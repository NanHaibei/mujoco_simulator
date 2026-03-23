import mujoco.viewer
import mujoco
import time
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from mit_msgs.msg import MITLowState, MITJointCommand, MITJointCommands
import yaml
import threading
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from std_srvs.srv import Empty
import numpy as np
from tf2_ros import TransformBroadcaster
from tf2_msgs.msg import TFMessage
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import copy
import os

# 导入插件系统
from .plugins import *

@dataclass
class StepTimeStats:
    """仿真步耗时统计"""
    times: deque = field(default_factory=lambda: deque(maxlen=1000))
    min_val: float = float('inf')
    max_val: float = 0.0
    sum_val: float = 0.0
    count: int = 0
    _start_time: float = 0.0
    _step_time: float = 0.0
    
    def tic(self):
        """开始计时"""
        self._start_time = time.time()
    
    def toc(self):
        """结束计时并更新统计数据"""
        self._step_time = time.time() - self._start_time
        step_time_ms = self._step_time * 1e3
        self.times.append(step_time_ms)
        self.min_val = min(self.min_val, step_time_ms)
        self.max_val = max(self.max_val, step_time_ms)
        self.sum_val += step_time_ms
        self.count += 1
        return self._step_time
    
    @property
    def step_time(self) -> float:
        """获取最近一次步耗时（秒）"""
        return self._step_time
    
    @property
    def mean(self) -> float:
        """计算平均值"""
        return self.sum_val / self.count if self.count > 0 else 0.0
    
    def format_summary(self) -> str:
        """格式化输出摘要"""
        return f"{self.min_val:.2f}/{self.mean:.2f}/{self.max_val:.2f}"

class mujoco_simulator(Node):
    """调用mujoco物理仿真, 收发ros2消息
    
    重构版本：使用插件系统组织功能模块
    """
    
    def __init__(self):
        super().__init__('mujoco_simulator')

        # 读取launch中传来的参数
        self.declare_parameter('yaml_path', " ")
        self.declare_parameter('mjcf_path', " ")
        yaml_path = self.get_parameter('yaml_path').get_parameter_value().string_value
        mjcf_path = self.get_parameter('mjcf_path').get_parameter_value().string_value

        # 读取yaml文件
        with open(yaml_path, 'r') as f:
            try:
                param = yaml.safe_load(f)
                self.param = param["mujoco_simulator"]
            except yaml.YAMLError as e:
                self.get_logger().error(f"YAML解析失败: {e}")
                raise

        # 保存mujoco的model和data
        self.mj_model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.sensor_data_list = list(self.mj_data.sensordata)

        # ==================== 初始化变量 ====================
        self.read_error_flag = False
        self.tf_broadcaster = TransformBroadcaster(self) # 给各个插件使用
        
        # 全局仿真步计数器
        self.step_counter = 0
        
        # step耗时统计
        self.step_stats = StepTimeStats()
        
        # 日志输出控制
        self.enable_log_output = self.param.get("enableLogOutput", False)
        self.log_output_interval = self.param.get("logOutputInterval", 50)  # 默认每50步输出一次

        # 计算渲染和状态发布的抽取频率
        self.render_decimation = int((1.0 / self.mj_model.opt.timestep) / self.param["renderRate"])

        # ==================== 初始是否暂停 ====================
        self.pause = self.param.get("initPauseFlag", False)
        self.unpause_service_name = self.param.get("unPauseService", "/unpause_mujoco")
        self.unpause_server = self.create_service(
            Empty, self.unpause_service_name, self._unpause_callback
        )

        # ==================== 加载插件系统 ====================
        self.plugins = []
        self._load_plugins()

        # ==================== ROS 回调线程 ====================
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self)
        self._spin_thread = None
        self._spin_running = threading.Event()

    def _spin_executor(self):
        """后台线程持续处理 ROS 回调。"""
        while self._spin_running.is_set() and rclpy.ok():
            self._executor.spin_once(timeout_sec=0.1)

    def _start_ros_spin_thread(self):
        """启动独立 ROS 回调线程。"""
        if self._spin_thread is not None and self._spin_thread.is_alive():
            return

        self._spin_running.set()
        self._spin_thread = threading.Thread(
            target=self._spin_executor,
            name='mujoco_simulator_ros_spin',
            daemon=True,
        )
        self._spin_thread.start()

    def _stop_ros_spin_thread(self):
        """停止独立 ROS 回调线程并清理执行器。"""
        self._spin_running.clear()

        if self._executor is not None:
            self._executor.wake()

        if self._spin_thread is not None and self._spin_thread.is_alive():
            self._spin_thread.join(timeout=1.0)

        if self._executor is not None:
            try:
                self._executor.remove_node(self)
            except Exception:
                pass
            self._executor.shutdown()
            self._executor = None

        self._spin_thread = None

    def run(self):
        """物理仿真主循环, 默认500Hz"""
        self._start_ros_spin_thread()
        
        # 开启mujoco窗口
        with mujoco.viewer.launch_passive(self.mj_model, self.mj_data) as viewer:
            # 默认关闭0号组的可视化（通常是地面平面）
            viewer.opt.geomgroup[0] = 0
            
            while viewer.is_running():
                # 开始计时
                self.step_stats.tic()

                # 进行物理仿真
                if not self.pause:
                    mujoco.mj_step(self.mj_model, self.mj_data)
                
                # 间隔一定step次数进行一次画面渲染
                if self.step_counter % self.render_decimation == 0:
                    
                    # 调用所有插件的可视化方法
                    for plugin in self.plugins:
                        try:
                            plugin.visualize(viewer)
                        except Exception as e:
                            self.get_logger().error(f"插件 {plugin.name} 可视化失败: {e}")

                    viewer.sync()
                # ==================== 调用所有插件的update方法 ====================
                for plugin in self.plugins:
                    try:
                        plugin.update()
                        # 执行插件的log输出
                        if self.enable_log_output and plugin.should_execute():
                            plugin.log()
                    except Exception as e:
                        self.get_logger().error(f"插件 {plugin.name} 执行失败: {e}")

                # 递增全局仿真步计数器
                self.step_counter += 1

                # 结束计时并更新统计数据
                self.step_stats.toc()
                
                # 输出仿真统计日志
                self._log_step_stats()

                # sleep 以保证仿真实时
                time_until_next_step = self.mj_model.opt.timestep - self.step_stats.step_time
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

    def _unpause_callback(self, request, response):
        """仿真启动回调"""
        self.get_logger().info("Unpause service called")
        self.pause = False
        if hasattr(self, 'keyframe_count') and self.keyframe_count > 0:
            mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, 0)
        return response

    def _log_step_stats(self):
        """输出仿真步耗时统计日志"""
        if self.enable_log_output and self.step_stats.count % self.log_output_interval == 0:
            self.get_logger().info(
                f"runtime[min/mean/max] {self.step_stats.format_summary()} ms"
            )

    def _load_plugins(self):
        """加载插件系统
        
        从simulate.yaml读取配置并按顺序加载插件
        配置格式: 列表中每个元素包含 path 字段，格式为 "module:class"
        """
        import importlib
        
        # 从 self.param 中读取插件配置
        plugins_config = self.param.get("plugins", [])
        
        self.get_logger().info(f"开始加载 {len(plugins_config)} 个插件...")
        
        for plugin_cfg in plugins_config:
            # 从 path 字段中解析 module 和 class
            plugin_path = plugin_cfg.get("path", "")
            if not plugin_path or ":" not in plugin_path:
                self.get_logger().warn(f"插件配置 path 格式错误，应为 'module:class': {plugin_path}")
                continue
            
            module_name, class_name = plugin_path.split(":", 1)
            
            # 动态导入插件类（使用相对导入）
            try:
                # 添加 . 前缀实现相对导入
                module = importlib.import_module(f".{module_name}", package="mujoco_simulator_python.plugins")
                plugin_class = getattr(module, class_name)
                plugin_instance = plugin_class(
                    plugin_config=plugin_cfg,
                    simulator=self
                )
                self.plugins.append(plugin_instance)
                self.get_logger().info(
                    f"插件 {plugin_instance.name} 加载成功 (step_interval={plugin_cfg.get('step_interval', 1)})"
                )
            except ImportError as e:
                self.get_logger().error(f"插件 {module_name}:{class_name} 模块导入失败: {e}")
            except AttributeError as e:
                self.get_logger().error(f"插件 {module_name}:{class_name} 类未找到: {e}")
            except Exception as e:
                self.get_logger().error(f"插件 {module_name}:{class_name} 加载失败: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = mujoco_simulator()
    try:
        node.run()
    finally:
        node._stop_ros_spin_thread()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()