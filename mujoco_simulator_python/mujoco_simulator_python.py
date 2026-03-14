import mujoco.viewer
import mujoco
import time
import rclpy
from rclpy.node import Node
from mit_msgs.msg import MITLowState, MITJointCommand, MITJointCommands
import yaml
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

# ==================== 常量定义 ====================
INVALID_ID = -1  # 无效ID标识

# 必需的传感器配置：(属性名, 中文描述)
REQUIRED_SENSORS = [
    ("joint_pos_head_id", "关节位置传感器"),
    ("joint_vel_head_id", "关节速度传感器"),
    ("joint_tor_head_id", "关节力矩传感器"),
    ("imu_quat_head_id", "四元数传感器"),
    ("imu_gyro_head_id", "角速度传感器"),
    ("imu_acc_head_id", "线加速度传感器"),
]

# head_id 名称映射
HEAD_ID_NAMES = {
    "joint_pos_head_id": "joint pos head",
    "joint_vel_head_id": "joint vel head",
    "joint_tor_head_id": "joint torque head",
    "imu_quat_head_id": "imu quat head",
    "imu_gyro_head_id": "imu gyro head",
    "imu_acc_head_id": "imu acc head",
    "real_pos_head_id": "real pos head",
    "real_vel_head_id": "real vel head",
}

@dataclass
class SensorHeadIDs:
    """传感器头ID数据类"""
    joint_pos: int = INVALID_ID
    joint_vel: int = INVALID_ID
    joint_tor: int = INVALID_ID
    imu_quat: int = INVALID_ID
    imu_gyro: int = INVALID_ID
    imu_acc: int = INVALID_ID
    real_pos: int = INVALID_ID
    real_vel: int = INVALID_ID

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

        # 读取当前模型信息并输出
        self.read_model()
        if self.param["modelTableFlag"]: self.show_model()

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

    def run(self):
        """物理仿真主循环, 默认500Hz"""
        
        # 开启mujoco窗口
        with mujoco.viewer.launch_passive(self.mj_model, self.mj_data) as viewer:
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

                # 处理ROS回调（非阻塞）
                rclpy.spin_once(self, timeout_sec=0.0)

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
        if self.keyframe_count > 0:
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

    def _build_head_id_reverse_map(self) -> Dict[int, str]:
        """构建 head_id 到名称的反向映射"""
        reverse_map = {}
        for attr_name, display_name in HEAD_ID_NAMES.items():
            head_id = getattr(self, attr_name, INVALID_ID)
            if head_id != INVALID_ID:
                reverse_map[head_id] = display_name
        return reverse_map

    def _validate_required_sensors(self, console: Console) -> bool:
        """验证必需的传感器是否存在
        
        Returns:
            True 如果所有必需传感器都存在，否则 False
        """
        has_error = False
        
        for attr_name, description in REQUIRED_SENSORS:
            if getattr(self, attr_name, INVALID_ID) == INVALID_ID:
                console.print(f"[bold red][ERROR][/bold red] 未发现{description},请检查模型")
                has_error = True
        
        return has_error

    def show_model(self):
        """输出读取到的机器人模型信息（重构版本）"""
        console = Console()

        # 标题
        console.print("[bold cyan]------------读取到的环境与模型信息如下------------[/bold cyan]")
        console.print(
            f"[bold]model name:[/bold] [green]{self.model_name}[/green]   "
            f"[bold]time step:[/bold] [yellow]{self.time_step:.6f}s[/yellow]"
        )

        # Joint 信息表格
        joint_table = Table(
            title="Joints information",
            header_style="bold white",
            box=box.HEAVY_EDGE,
            show_lines=False,
            pad_edge=False
        )
        headers = ["ID", "name", "posLimit(rad)", "torLimit(Nm)", "friction", "damping"]
        for h in headers:
            joint_table.add_column(h, justify="center", no_wrap=True)
        
        for i, name in enumerate(self.joint_name):
            pos_range = f"{self.joint_pos_range[i][0]:.2f} ~ {self.joint_pos_range[i][1]:.2f}"
            tor_range = f"{self.joint_torque_range[i][0]:.2f} ~ {self.joint_torque_range[i][1]:.2f}"
            joint_table.add_row(
                str(i), name, pos_range, tor_range,
                f"{self.joint_friction[i]:.2f}", f"{self.joint_damping[i]:.2f}"
            )
        console.print(joint_table)

        # Link 信息表格
        link_table = Table(
            title="Links information",
            header_style="bold white",
            box=box.HEAVY_EDGE,
            show_lines=False,
            pad_edge=False
        )
        for h in ["ID", "name", "mass(kg)"]:
            link_table.add_column(h, justify="center", no_wrap=True)
        
        for i, name in enumerate(self.link_name):
            link_table.add_row(str(i), name, f"{self.link_mass[i]:.2f}")
        console.print(link_table)

        # Sensor 信息表格
        sensor_table = Table(
            title="Sensors information",
            header_style="bold white",
            box=box.HEAVY_EDGE,
            show_lines=False,
            pad_edge=False
        )
        for h in ["ID", "name", "type", "attach", "head"]:
            sensor_table.add_column(h, justify="center", no_wrap=True)
        
        # 使用反向映射简化 head_id_name 查找
        head_id_map = self._build_head_id_reverse_map()
        
        for i, sensor in enumerate(self.sensor_type):
            head_id_name = head_id_map.get(i, "")
            sensor_table.add_row(str(i), sensor[0], sensor[1], sensor[2], head_id_name)
        console.print(sensor_table)

        # 提示信息
        console.print(
            Panel("[bold green]如果仿真遇到问题,请检查上述信息是否正确,物理仿真进行中...[/bold green]")
        )

        # Keyframe 检查
        if self.keyframe_count == 0:
            console.print("[bold yellow][WARN][/bold yellow] 未发现keyframe,请检查模型")

        # 传感器错误检查（使用统一的验证方法）
        self.read_error_flag = self._validate_required_sensors(console)

        if self.read_error_flag:
            console.print("[bold red][ERROR][/bold red] 传感器参数缺失,将不会进行ROS通信")

    def _get_body_name_by_sensor(self, sensor_id: int) -> Optional[str]:
        """根据传感器ID获取其附着的body名称"""
        obj_id = self.mj_model.sensor_objid[sensor_id]
        body_id = self.mj_model.site_bodyid[obj_id] + 1
        return mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id)

    def _add_vector_sensor(self, name: str, sensor_type: str, attach: str, 
                           head_id_attr: str = None, exclude_filter: str = None) -> int:
        """添加向量类型传感器（展开为 x/y/z 三个分量）
        
        Args:
            name: 传感器名称
            sensor_type: 传感器类型描述
            attach: 附着体名称
            head_id_attr: 要设置的 head_id 属性名（如 "imu_gyro_head_id"）
            exclude_filter: 排除过滤字符串（如 "imu2"）
            
        Returns:
            起始的 sensor_type 列表索引
        """
        head_idx = len(self.sensor_type)
        
        # 如果指定了排除过滤且名称包含该字符串，则不设置 head_id
        if head_id_attr and (exclude_filter is None or exclude_filter not in name.lower()):
            setattr(self, head_id_attr, head_idx)
        
        # 添加 x/y/z 分量
        for comp in ['_x', '_y', '_z']:
            self.sensor_type.append([name + comp, sensor_type, attach])
        
        return head_idx

    def _add_quat_sensor(self, name: str, sensor_type: str, attach: str,
                         head_id_attr: str = None, exclude_filter: str = None) -> int:
        """添加四元数类型传感器（展开为 w/x/y/z 四个分量）"""
        head_idx = len(self.sensor_type)
        
        if head_id_attr and (exclude_filter is None or exclude_filter not in name.lower()):
            setattr(self, head_id_attr, head_idx)
        
        # 四元数顺序：w, x, y, z
        for comp in ['_w', '_x', '_y', '_z']:
            self.sensor_type.append([name + comp, sensor_type, attach])
        
        return head_idx

    def _read_joints(self):
        """读取所有关节信息"""
        for i in range(self.mj_model.njnt):
            if self.mj_model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            
            self.joint_name.append(
                mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
            )
            self.joint_pos_range.append(self.mj_model.jnt_range[i])
            self.joint_torque_range.append(self.mj_model.jnt_actfrcrange[i])
            
            dofadr = self.mj_model.jnt_dofadr[i]
            self.joint_friction.append(self.mj_model.dof_frictionloss[dofadr])
            self.joint_damping.append(self.mj_model.dof_damping[dofadr])

    def _read_links(self):
        """读取所有连杆信息"""
        for i in range(self.mj_model.nbody):
            name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name == "world":
                continue
            self.link_name.append(name)
            self.link_mass.append(self.mj_model.body_mass[i])

    def _read_sensors(self):
        """读取所有传感器信息"""
        # 简单传感器类型映射：(mujoco_type, type_name, head_id_attr)
        SIMPLE_SENSOR_TYPES = {
            mujoco.mjtSensor.mjSENS_JOINTPOS: ("joint pos", "joint_pos_head_id"),
            mujoco.mjtSensor.mjSENS_JOINTVEL: ("joint vel", "joint_vel_head_id"),
            mujoco.mjtSensor.mjSENS_JOINTACTFRC: ("joint torque", "joint_tor_head_id"),
        }
        
        for i in range(self.mj_model.nsensor):
            sensor_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            sensor_type = self.mj_model.sensor_type[i]
            
            # 处理简单传感器
            if sensor_type in SIMPLE_SENSOR_TYPES:
                type_name, head_attr = SIMPLE_SENSOR_TYPES[sensor_type]
                # 只在未设置时设置 head_id
                if getattr(self, head_attr) == INVALID_ID:
                    setattr(self, head_attr, len(self.sensor_type))
                
                attach = mujoco.mj_id2name(
                    self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, 
                    self.mj_model.sensor_objid[i]
                )
                self.sensor_type.append([sensor_name, type_name, attach])
            
            # 处理四元数传感器
            elif sensor_type == mujoco.mjtSensor.mjSENS_FRAMEQUAT:
                attach = self._get_body_name_by_sensor(i)
                self._add_quat_sensor(
                    sensor_name, "imu quat", attach,
                    "imu_quat_head_id", "imu2"
                )
            
            # 处理陀螺仪
            elif sensor_type == mujoco.mjtSensor.mjSENS_GYRO:
                attach = self._get_body_name_by_sensor(i)
                self._add_vector_sensor(
                    sensor_name, "imu gyro", attach,
                    "imu_gyro_head_id", "imu2"
                )
            
            # 处理加速度计
            elif sensor_type == mujoco.mjtSensor.mjSENS_ACCELEROMETER:
                attach = self._get_body_name_by_sensor(i)
                self._add_vector_sensor(
                    sensor_name, "imu linear acc", attach,
                    "imu_acc_head_id", "imu2"
                )
            
            # 处理位置传感器
            elif sensor_type == mujoco.mjtSensor.mjSENS_FRAMEPOS:
                attach = self._get_body_name_by_sensor(i)
                self._add_vector_sensor(sensor_name, "real position", attach, "real_pos_head_id")
            
            # 处理速度传感器
            elif sensor_type == mujoco.mjtSensor.mjSENS_FRAMELINVEL:
                attach = self._get_body_name_by_sensor(i)
                self._add_vector_sensor(sensor_name, "real velocity", attach, "real_vel_head_id")
            
            # 未知类型
            else:
                attach = mujoco.mj_id2name(
                    self.mj_model, mujoco.mjtObj.mjOBJ_JOINT,
                    self.mj_model.sensor_objid[i]
                )
                self.sensor_type.append([sensor_name, "unknown", attach])

    def _read_terrain(self):
        """读取障碍物/地形信息"""
        for geom_id in range(self.mj_model.ngeom):
            geom_type = self.mj_model.geom_type[geom_id]
            pos = self.mj_model.geom_pos[geom_id].copy()
            quat = self.mj_model.geom_quat[geom_id].copy()
            rgba = self.mj_model.geom_rgba[geom_id].copy()
            
            if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                self.terrain_pos.append(pos)
                self.terrain_quat.append(quat)
                self.terrain_size.append(self.mj_model.geom_size[geom_id].copy() * 2)
                self.terrain_rgba.append(rgba)
                self.terrain_type.append('box')
            
            elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                size = self.mj_model.geom_size[geom_id].copy()
                self.terrain_pos.append(pos)
                self.terrain_quat.append(quat)
                self.terrain_size.append([size[0] * 2, size[0] * 2, size[1] * 2])
                self.terrain_rgba.append(rgba)
                self.terrain_type.append('cylinder')

    def read_model(self):
        """从mjcf中读取模型信息（重构版本）"""
        # 初始化列表
        self.joint_name: List[str] = []
        self.joint_pos_range: List[np.ndarray] = []
        self.joint_torque_range: List[np.ndarray] = []
        self.joint_friction: List[float] = []
        self.joint_damping: List[float] = []
        self.link_name: List[str] = []
        self.link_mass: List[float] = []
        self.sensor_type: List[List[str]] = []
        
        # 初始化 head_id（使用常量 INVALID_ID）
        self.joint_pos_head_id = INVALID_ID
        self.joint_vel_head_id = INVALID_ID
        self.joint_tor_head_id = INVALID_ID
        self.imu_quat_head_id = INVALID_ID
        self.imu_gyro_head_id = INVALID_ID
        self.imu_acc_head_id = INVALID_ID
        self.real_pos_head_id = INVALID_ID
        self.real_vel_head_id = INVALID_ID
        
        # 初始化地形信息
        self.terrain_pos: List[np.ndarray] = []
        self.terrain_size: List[np.ndarray] = []
        self.terrain_quat: List[np.ndarray] = []
        self.terrain_rgba: List[np.ndarray] = []
        self.terrain_type: List[str] = []
        
        # 读取基本信息
        self.model_name = self.mj_model.names.split(b'\x00', 1)[0].decode('utf-8')
        self.time_step = self.mj_model.opt.timestep
        
        # 加载关键帧
        self.keyframe_count = self.mj_model.nkey
        if self.keyframe_count > 0:
            mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, 0)
        
        # 分模块读取
        self._read_joints()
        self._read_links()
        self.first_link_name = self.link_name[0] if self.link_name else None
        self._read_sensors()
        self._read_terrain()


def main(args=None):
    rclpy.init(args=args)
    node = mujoco_simulator()
    node.run()


if __name__ == '__main__':
    main()