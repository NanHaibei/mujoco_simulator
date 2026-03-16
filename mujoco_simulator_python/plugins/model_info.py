from __future__ import annotations
import mujoco
import numpy as np
from typing import TYPE_CHECKING, List, Optional, Dict

if TYPE_CHECKING:
    from ..mujoco_simulator_python import mujoco_simulator

from .base import BasePlugin


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


class ModelInfo(BasePlugin):
    """模型信息插件
    
    负责：
    1. 从MuJoCo模型中读取joint/link/sensor/terrain信息
    2. 将这些信息存储到simulator实例上供其他插件使用
    3. 可选地输出格式化的模型信息表格
    
    必须是第一个加载的插件，因为其他插件依赖 head_id 等信息。
    """
    
    def __init__(self, plugin_config: dict, simulator: mujoco_simulator):
        """初始化模型信息插件"""
        super().__init__(plugin_config, simulator)
        
        # 读取配置
        self.show_table = plugin_config.get("show_table", True)
        
        # 执行模型读取（在初始化时立即执行）
        self._read_model()
        
        # 可选显示模型信息表格
        if self.show_table:
            self._show_model()
        
        self.simulator.get_logger().info(
            f"模型信息插件已初始化: joints={len(self.simulator.joint_name)}, "
            f"links={len(self.simulator.link_name)}, "
            f"sensors={len(self.simulator.sensor_type)}"
        )
    
    def execute(self):
        """插件执行函数 - 模型信息只需读取一次，不需要周期执行"""
        pass
    
    def _read_model(self):
        """从mjcf中读取模型信息"""
        simulator = self.simulator
        
        # 初始化列表
        simulator.joint_name: List[str] = []
        simulator.joint_pos_range: List[np.ndarray] = []
        simulator.joint_torque_range: List[np.ndarray] = []
        simulator.joint_friction: List[float] = []
        simulator.joint_damping: List[float] = []
        simulator.link_name: List[str] = []
        simulator.link_mass: List[float] = []
        simulator.sensor_type: List[List[str]] = []
        
        # 初始化 head_id（使用常量 INVALID_ID）
        simulator.joint_pos_head_id = INVALID_ID
        simulator.joint_vel_head_id = INVALID_ID
        simulator.joint_tor_head_id = INVALID_ID
        simulator.imu_quat_head_id = INVALID_ID
        simulator.imu_gyro_head_id = INVALID_ID
        simulator.imu_acc_head_id = INVALID_ID
        simulator.real_pos_head_id = INVALID_ID
        simulator.real_vel_head_id = INVALID_ID
        
        # 初始化地形信息
        simulator.terrain_pos: List[np.ndarray] = []
        simulator.terrain_size: List[np.ndarray] = []
        simulator.terrain_quat: List[np.ndarray] = []
        simulator.terrain_rgba: List[np.ndarray] = []
        simulator.terrain_type: List[str] = []
        
        # 读取基本信息
        simulator.model_name = self.mj_model.names.split(b'\x00', 1)[0].decode('utf-8')
        simulator.time_step = self.mj_model.opt.timestep
        
        # 加载关键帧
        simulator.keyframe_count = self.mj_model.nkey
        if simulator.keyframe_count > 0:
            mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, 0)
        
        # 分模块读取
        self._read_joints()
        self._read_links()
        simulator.first_link_name = simulator.link_name[0] if simulator.link_name else None
        self._read_sensors()
        self._read_terrain()
    
    def _read_joints(self):
        """读取所有关节信息"""
        simulator = self.simulator
        
        for i in range(self.mj_model.njnt):
            if self.mj_model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            
            simulator.joint_name.append(
                mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
            )
            simulator.joint_pos_range.append(self.mj_model.jnt_range[i])
            simulator.joint_torque_range.append(self.mj_model.jnt_actfrcrange[i])
            
            dofadr = self.mj_model.jnt_dofadr[i]
            simulator.joint_friction.append(self.mj_model.dof_frictionloss[dofadr])
            simulator.joint_damping.append(self.mj_model.dof_damping[dofadr])
    
    def _read_links(self):
        """读取所有连杆信息"""
        simulator = self.simulator
        
        for i in range(self.mj_model.nbody):
            name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name == "world":
                continue
            simulator.link_name.append(name)
            simulator.link_mass.append(self.mj_model.body_mass[i])
    
    def _read_sensors(self):
        """读取所有传感器信息"""
        # 简单传感器类型映射：(mujoco_type, type_name, head_id_attr)
        SIMPLE_SENSOR_TYPES = {
            mujoco.mjtSensor.mjSENS_JOINTPOS: ("joint pos", "joint_pos_head_id"),
            mujoco.mjtSensor.mjSENS_JOINTVEL: ("joint vel", "joint_vel_head_id"),
            mujoco.mjtSensor.mjSENS_JOINTACTFRC: ("joint torque", "joint_tor_head_id"),
        }
        
        simulator = self.simulator
        
        for i in range(self.mj_model.nsensor):
            sensor_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            sensor_type = self.mj_model.sensor_type[i]
            
            # 处理简单传感器
            if sensor_type in SIMPLE_SENSOR_TYPES:
                type_name, head_attr = SIMPLE_SENSOR_TYPES[sensor_type]
                # 只在未设置时设置 head_id
                if getattr(simulator, head_attr) == INVALID_ID:
                    setattr(simulator, head_attr, len(simulator.sensor_type))
                
                attach = mujoco.mj_id2name(
                    self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, 
                    self.mj_model.sensor_objid[i]
                )
                simulator.sensor_type.append([sensor_name, type_name, attach])
            
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
                simulator.sensor_type.append([sensor_name, "unknown", attach])
    
    def _read_terrain(self):
        """读取障碍物/地形信息"""
        simulator = self.simulator
        
        for geom_id in range(self.mj_model.ngeom):
            geom_type = self.mj_model.geom_type[geom_id]
            pos = self.mj_model.geom_pos[geom_id].copy()
            quat = self.mj_model.geom_quat[geom_id].copy()
            rgba = self.mj_model.geom_rgba[geom_id].copy()
            
            if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                simulator.terrain_pos.append(pos)
                simulator.terrain_quat.append(quat)
                simulator.terrain_size.append(self.mj_model.geom_size[geom_id].copy() * 2)
                simulator.terrain_rgba.append(rgba)
                simulator.terrain_type.append('box')
            
            elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                size = self.mj_model.geom_size[geom_id].copy()
                simulator.terrain_pos.append(pos)
                simulator.terrain_quat.append(quat)
                simulator.terrain_size.append([size[0] * 2, size[0] * 2, size[1] * 2])
                simulator.terrain_rgba.append(rgba)
                simulator.terrain_type.append('cylinder')
    
    def _get_body_name_by_sensor(self, sensor_id: int) -> Optional[str]:
        """根据传感器ID获取其附着的body名称"""
        obj_id = self.mj_model.sensor_objid[sensor_id]
        body_id = self.mj_model.site_bodyid[obj_id] + 1
        return mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    
    def _add_vector_sensor(self, name: str, sensor_type: str, attach: str, 
                           head_id_attr: str = None, exclude_filter: str = None) -> int:
        """添加向量类型传感器（展开为 x/y/z 三个分量）"""
        simulator = self.simulator
        head_idx = len(simulator.sensor_type)
        
        # 如果指定了排除过滤且名称包含该字符串，则不设置 head_id
        if head_id_attr and (exclude_filter is None or exclude_filter not in name.lower()):
            setattr(simulator, head_id_attr, head_idx)
        
        # 添加 x/y/z 分量
        for comp in ['_x', '_y', '_z']:
            simulator.sensor_type.append([name + comp, sensor_type, attach])
        
        return head_idx
    
    def _add_quat_sensor(self, name: str, sensor_type: str, attach: str,
                         head_id_attr: str = None, exclude_filter: str = None) -> int:
        """添加四元数类型传感器（展开为 w/x/y/z 四个分量）"""
        simulator = self.simulator
        head_idx = len(simulator.sensor_type)
        
        if head_id_attr and (exclude_filter is None or exclude_filter not in name.lower()):
            setattr(simulator, head_id_attr, head_idx)
        
        # 四元数顺序：w, x, y, z
        for comp in ['_w', '_x', '_y', '_z']:
            simulator.sensor_type.append([name + comp, sensor_type, attach])
        
        return head_idx
    
    def _show_model(self):
        """输出读取到的机器人模型信息"""
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich import box
        
        console = Console()
        simulator = self.simulator

        # 标题
        console.print("[bold cyan]------------读取到的环境与模型信息如下------------[/bold cyan]")
        console.print(
            f"[bold]model name:[/bold] [green]{simulator.model_name}[/green]   "
            f"[bold]time step:[/bold] [yellow]{simulator.time_step:.6f}s[/yellow]"
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
        
        for i, name in enumerate(simulator.joint_name):
            pos_range = f"{simulator.joint_pos_range[i][0]:.2f} ~ {simulator.joint_pos_range[i][1]:.2f}"
            tor_range = f"{simulator.joint_torque_range[i][0]:.2f} ~ {simulator.joint_torque_range[i][1]:.2f}"
            joint_table.add_row(
                str(i), name, pos_range, tor_range,
                f"{simulator.joint_friction[i]:.2f}", f"{simulator.joint_damping[i]:.2f}"
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
        
        for i, name in enumerate(simulator.link_name):
            link_table.add_row(str(i), name, f"{simulator.link_mass[i]:.2f}")
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
        
        for i, sensor in enumerate(simulator.sensor_type):
            head_id_name = head_id_map.get(i, "")
            sensor_table.add_row(str(i), sensor[0], sensor[1], sensor[2], head_id_name)
        console.print(sensor_table)

        # 提示信息
        console.print(
            Panel("[bold green]如果仿真遇到问题,请检查上述信息是否正确,物理仿真进行中...[/bold green]")
        )

        # Keyframe 检查
        if simulator.keyframe_count == 0:
            console.print("[bold yellow][WARN][/bold yellow] 未发现keyframe,请检查模型")

        # 传感器错误检查
        self.simulator.read_error_flag = self._validate_required_sensors(console)

        if self.simulator.read_error_flag:
            console.print("[bold red][ERROR][/bold red] 传感器参数缺失,将不会进行ROS通信")
    
    def _build_head_id_reverse_map(self) -> Dict[int, str]:
        """构建 head_id 到名称的反向映射"""
        reverse_map = {}
        for attr_name, display_name in HEAD_ID_NAMES.items():
            head_id = getattr(self.simulator, attr_name, INVALID_ID)
            if head_id != INVALID_ID:
                reverse_map[head_id] = display_name
        return reverse_map
    
    def _validate_required_sensors(self, console) -> bool:
        """验证必需的传感器是否存在
        
        Returns:
            True 如果所有必需传感器都存在，否则 False
        """
        has_error = False
        
        for attr_name, description in REQUIRED_SENSORS:
            if getattr(self.simulator, attr_name, INVALID_ID) == INVALID_ID:
                console.print(f"[bold red][ERROR][/bold red] 未发现{description},请检查模型")
                has_error = True
        
        return has_error