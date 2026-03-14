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
import copy
import os

# 导入插件系统
from .plugins import *


class mujoco_simulator(Node):
    """调用mujoco物理仿真, 收发ros2消息
    
    重构版本：使用插件系统组织功能模块
    """
    
    def __init__(self):
        super().__init__('mujoco_simulator')

        # 读取launch中传来的参数
        self.declare_parameter('yaml_path', " ")  # launch文件中的参数
        self.declare_parameter('mjcf_path', " ")  # launch文件中的参数
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

        # 实例化mujoco的model和data
        self.mj_model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.sensor_data_list = list(self.mj_data.sensordata)

        # 读取模型信息并输出
        self.read_model()
        if self.param["modelTableFlag"]: self.show_model()

        # TF广播器
        self.tf_broadcaster = TransformBroadcaster(self)

        # ==================== 初始化变量 ====================
        # 注意：以下变量由各插件初始化：
        # - low_state_msg, low_cmd_msg, state_deque: LowStatePlugin
        # - cmd_deque: PdControllerPlugin
        # - pause: SimulationControlPlugin
        # - map_triggered: MapFramePlugin
        
        self.read_error_flag = False
        self.mujoco_step_time = 0.0
        
        # step耗时统计变量
        self.step_times = deque(maxlen=1000)
        self.step_time_min = float('inf')
        self.step_time_max = 0.0
        self.step_time_sum = 0.0
        self.step_count = 0

        # 计算渲染和状态发布的抽取频率
        self.render_decimation = int((1.0 / self.mj_model.opt.timestep) / self.param["renderRate"])

        # ==================== 加载插件系统 ====================
        self.plugins = []
        self._load_plugins(yaml_path)

        # 打印 IMU 传感器信息
        self._print_imu_info()

    def _load_plugins(self, yaml_path):
        """加载插件系统
        
        从plugin_config.yaml读取配置并按顺序加载插件
        使用动态导入从配置中的module和class字段加载插件类
        """
        import importlib
        
        # 读取插件配置文件
        config_dir = os.path.dirname(yaml_path)
        plugin_config_path = os.path.join(config_dir, "plugin_config.yaml")
        
        try:
            with open(plugin_config_path, 'r') as f:
                plugin_config = yaml.safe_load(f)
        except FileNotFoundError:
            self.get_logger().warn(f"插件配置文件未找到: {plugin_config_path}")
            return
        except yaml.YAMLError as e:
            self.get_logger().error(f"插件配置解析失败: {e}")
            return
        
        plugins_config = plugin_config.get("plugins", [])
        
        self.get_logger().info(f"开始加载 {len(plugins_config)} 个插件...")
        
        for plugin_cfg in plugins_config:
            plugin_name = plugin_cfg.get("name", "")
            if not plugin_name:
                continue
            
            # 检查是否启用
            if not plugin_cfg.get("enabled", True):
                self.get_logger().info(f"插件 {plugin_name} 已禁用，跳过")
                continue
            
            # 从配置中获取模块和类名
            module_name = plugin_cfg.get("module", "")
            class_name = plugin_cfg.get("class", "")
            
            if not module_name or not class_name:
                self.get_logger().warn(f"插件 {plugin_name} 缺少 module 或 class 配置")
                continue
            
            # 动态导入插件类
            try:
                module = importlib.import_module(module_name, package="mujoco_simulator_python.plugins")
                plugin_class = getattr(module, class_name)
                plugin_instance = plugin_class(
                    name=plugin_name,
                    plugin_config=plugin_cfg,
                    simulator=self
                )
                self.plugins.append(plugin_instance)
                self.get_logger().info(
                    f"插件 {plugin_name} 加载成功 (step_interval={plugin_cfg.get('step_interval', 1)})"
                )
            except ImportError as e:
                self.get_logger().error(f"插件 {plugin_name} 模块导入失败: {e}")
            except AttributeError as e:
                self.get_logger().error(f"插件 {plugin_name} 类 {class_name} 未找到: {e}")
            except Exception as e:
                self.get_logger().error(f"插件 {plugin_name} 加载失败: {e}")

    def _print_imu_info(self):
        """打印IMU传感器信息"""
        self.imu_attach_body_name = None
        self.imu_attach_body_id = -1
        
        for sensor_id in range(self.mj_model.nsensor):
            sensor_type = self.mj_model.sensor_type[sensor_id]
            sensor_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_id)
            
            if sensor_type == mujoco.mjtSensor.mjSENS_FRAMEQUAT and "imu2" not in sensor_name.lower():
                site_id = self.mj_model.sensor_objid[sensor_id]
                if site_id >= 0:
                    self.imu_attach_body_id = self.mj_model.site_bodyid[site_id]
                    self.imu_attach_body_name = mujoco.mj_id2name(
                        self.mj_model, mujoco.mjtObj.mjOBJ_BODY, self.imu_attach_body_id
                    )
                    site_pos = self.mj_model.site_pos[site_id].copy()
                    self.imu_pos_offset = (float(site_pos[0]), float(site_pos[1]), float(site_pos[2]))
                    self.get_logger().info(
                        f"IMU 传感器附着到 body: {self.imu_attach_body_name}, 位置偏移: {self.imu_pos_offset}"
                    )
                    break

        # 打印 torso_link 到 pelvis 的偏移
        try:
            torso_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
            pelvis_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
            if torso_id >= 0 and pelvis_id >= 0:
                mujoco.mj_forward(self.mj_model, self.mj_data)
                torso_pos = self.mj_data.xpos[torso_id].copy()
                pelvis_pos = self.mj_data.xpos[pelvis_id].copy()
                offset = torso_pos - pelvis_pos
                self.get_logger().info(
                    f"torso_link 到 pelvis 的偏移（初始配置）: ({offset[0]:.6f}, {offset[1]:.6f}, {offset[2]:.6f})"
                )
        except Exception as e:
            pass

    def run(self):
        """物理仿真主循环, 默认500Hz"""
        rander_count = 0
        
        # 开启mujoco窗口
        with mujoco.viewer.launch_passive(self.mj_model, self.mj_data) as viewer:
            while viewer.is_running():
                # 记录当前运行时间
                self.temp_time1 = time.time()

                # 进行物理仿真
                if not self.pause:
                    mujoco.mj_step(self.mj_model, self.mj_data)
                
                # 间隔一定step次数进行一次画面渲染
                if rander_count % self.render_decimation == 0:
                    viewer.sync()
                    
                    # 调用所有插件的可视化方法
                    for plugin in self.plugins:
                        try:
                            plugin.visualize(viewer)
                        except Exception as e:
                            self.get_logger().error(f"插件 {plugin.name} 可视化失败: {e}")

                # 处理ROS回调（非阻塞）
                rclpy.spin_once(self, timeout_sec=0.0)

                # ==================== 调用所有插件的update方法 ====================
                for plugin in self.plugins:
                    try:
                        plugin.update()
                    except Exception as e:
                        self.get_logger().error(f"插件 {plugin.name} 执行失败: {e}")

                rander_count += 1

                self.temp_time2 = time.time()
                self.mujoco_step_time = self.temp_time2 - self.temp_time1
                
                # 统计step耗时（转换为毫秒）
                step_time_ms = self.mujoco_step_time * 1e3
                self.step_times.append(step_time_ms)
                self.step_time_min = min(self.step_time_min, step_time_ms)
                self.step_time_max = max(self.step_time_max, step_time_ms)
                self.step_time_sum += step_time_ms
                self.step_count += 1

                # sleep 以保证仿真实时
                time_until_next_step = self.mj_model.opt.timestep - self.mujoco_step_time
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

    def show_model(self):
        """输出读取到的机器人模型信息"""
        console = Console()

        console.print("[bold cyan]------------读取到的环境与模型信息如下------------[/bold cyan]")
        console.print(f"[bold]model name:[/bold] [green]{self.model_name}[/green]   [bold]time step:[/bold] [yellow]{self.time_step:.6f}s[/yellow]")

        # Joint 信息
        joint_table = Table(title="Joints information", 
                            header_style="bold white",
                            box=box.HEAVY_EDGE, 
                            show_lines=False, 
                            pad_edge=False)
        for h in ["ID", "name", "posLimit(rad)", "torLimit(Nm)", "friction", "damping"]:
            joint_table.add_column(h, justify="center", no_wrap=True)
        for i, name in enumerate(self.joint_name):
            pos_range = f"{self.joint_pos_range[i][0]:.2f} ~ {self.joint_pos_range[i][1]:.2f}"
            tor_range = f"{self.joint_torque_range[i][0]:.2f} ~ {self.joint_torque_range[i][1]:.2f}"
            joint_table.add_row(
                str(i),
                name,
                pos_range,
                tor_range,
                f"{self.joint_friction[i]:.2f}",
                f"{self.joint_damping[i]:.2f}"
            )
        console.print(joint_table)

        # Link 信息
        link_table = Table(title="Links information", 
                        header_style="bold white", 
                        box=box.HEAVY_EDGE,
                        show_lines=False,
                        pad_edge=False)
        for h in ["ID", "name", "mass(kg)"]:
            link_table.add_column(h, justify="center", no_wrap=True)
        for i, name in enumerate(self.link_name):
            link_table.add_row(str(i), name, f"{self.link_mass[i]:.2f}")
        console.print(link_table)

        # Sensor 信息
        sensor_table = Table(title="Sensors information", 
                            header_style="bold white", 
                            box=box.HEAVY_EDGE, 
                            show_lines=False,
                            pad_edge=False)
        for h in ["ID", "name", "type", "attach", "head"]:
            sensor_table.add_column(h, justify="center", no_wrap=True)
        for i, sensor in enumerate(self.sensor_type):
            head_id_name = ""
            if i == self.joint_pos_head_id: head_id_name = "joint pos head"
            elif i == self.joint_vel_head_id: head_id_name = "joint vel head"
            elif i == self.joint_tor_head_id: head_id_name = "joint torque head"
            elif i == self.imu_quat_head_id: head_id_name = "imu quat head"
            elif i == self.imu_gyro_head_id: head_id_name = "imu gyro head"
            elif i == self.imu_acc_head_id: head_id_name = "imu acc head"
            elif i == self.imu2_quat_head_id: head_id_name = "imu2 quat head"
            elif i == self.imu2_gyro_head_id: head_id_name = "imu2 gyro head"
            elif i == self.imu2_acc_head_id: head_id_name = "imu2 acc head"
            elif i == self.real_pos_head_id: head_id_name = "real pos head"
            elif i == self.real_vel_head_id: head_id_name = "real vel head"
            sensor_table.add_row(str(i), sensor[0], sensor[1], sensor[2], head_id_name)
        console.print(sensor_table)

        console.print(Panel("[bold green]如果仿真遇到问题,请检查上述信息是否正确,物理仿真进行中...[/bold green]"))

        # Keyframe 检查
        if self.keyframe_count == 0:
            console.print("[bold yellow][WARN][/bold yellow] 未发现keyframe,请检查模型")

        # 传感器错误检查
        self.read_error_flag = False
        def err(msg): 
            console.print(f"[bold red][ERROR][/bold red] {msg}")
            return True

        if self.joint_pos_head_id == 999999: self.read_error_flag = err("未发现关节位置传感器,请检查模型")
        if self.joint_vel_head_id == 999999: self.read_error_flag = err("未发现关节速度传感器,请检查模型")
        if self.joint_tor_head_id == 999999: self.read_error_flag = err("未发现关节力矩传感器,请检查模型")
        if self.imu_quat_head_id == 999999: self.read_error_flag = err("未发现四元数传感器,请检查模型")
        if self.imu_gyro_head_id == 999999: self.read_error_flag = err("未发现角速度传感器,请检查模型")
        if self.imu_acc_head_id == 999999: self.read_error_flag = err("未发现线加速度传感器,请检查模型")

        if self.read_error_flag:
            console.print("[bold red][ERROR][/bold red] 传感器参数缺失,将不会进行ROS通信")

    def read_model(self):
        """从mjcf中读取模型信息"""
        # 初始化变量
        self.joint_name = []
        self.joint_pos_range = []
        self.joint_torque_range = []
        self.joint_friction = []
        self.joint_damping = []
        self.link_name = []
        self.link_mass = []
        self.sensor_type = []
        self.joint_pos_head_id = 999999
        self.joint_vel_head_id = 999999
        self.joint_tor_head_id = 999999
        self.imu_quat_head_id = 999999
        self.imu_gyro_head_id = 999999
        self.imu_acc_head_id = 999999
        self.imu2_quat_head_id = 999999
        self.imu2_gyro_head_id = 999999
        self.imu2_acc_head_id = 999999
        self.real_pos_head_id = 999999
        self.real_vel_head_id = 999999
        self.terrain_pos = []
        self.terrain_size = []
        self.terrain_quat = []
        self.terrain_rgba = []
        self.terrain_type = []
        
        # 读取模型名称
        self.model_name = self.mj_model.names.split(b'\x00', 1)[0].decode('utf-8')
        self.time_step = self.mj_model.opt.timestep
        
        # 加载关键帧
        self.keyframe_count = self.mj_model.nkey
        if self.keyframe_count > 0:
            mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, 0)

        # 遍历所有joint
        for i in range(self.mj_model.njnt):
            if(self.mj_model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE):
                continue
            self.joint_name.append((mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)))
            self.joint_pos_range.append(self.mj_model.jnt_range[i])
            self.joint_torque_range.append(self.mj_model.jnt_actfrcrange[i])
            joint_dofadr = self.mj_model.jnt_dofadr[i]
            self.joint_friction.append(self.mj_model.dof_frictionloss[joint_dofadr])
            self.joint_damping.append(self.mj_model.dof_damping[joint_dofadr])

        # 遍历所有link
        for i in range(self.mj_model.nbody):
            if(mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, i) == "world"):
                continue
            self.link_name.append((mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, i)))
            self.link_mass.append(self.mj_model.body_mass[i])

        self.first_link_name = self.link_name[0]

        # 遍历所有sensor
        for i in range(self.mj_model.nsensor):
            temp_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            temp_attch = ""
            if (self.mj_model.sensor_type[i] == mujoco.mjtSensor.mjSENS_JOINTPOS):
                temp_type = "joint pos"
                self.joint_pos_head_id = len(self.sensor_type) if self.joint_pos_head_id == 999999 else self.joint_pos_head_id
            elif (self.mj_model.sensor_type[i] == mujoco.mjtSensor.mjSENS_JOINTVEL):
                temp_type = "joint vel"
                self.joint_vel_head_id = len(self.sensor_type) if self.joint_vel_head_id == 999999 else self.joint_vel_head_id
            elif (self.mj_model.sensor_type[i] == mujoco.mjtSensor.mjSENS_JOINTACTFRC):
                temp_type = "joint torque"
                self.joint_tor_head_id = len(self.sensor_type) if self.joint_tor_head_id == 999999 else self.joint_tor_head_id
            elif (self.mj_model.sensor_type[i] == mujoco.mjtSensor.mjSENS_FRAMEQUAT):
                temp_type = "imu quat"
                temp_attch = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, self.mj_model.sensor_objid[i]+1)
                if "imu2" in temp_name.lower():
                    self.imu2_quat_head_id = len(self.sensor_type)
                else:
                    self.imu_quat_head_id = len(self.sensor_type)
                self.sensor_type.append([temp_name+"_w", temp_type, temp_attch])
                self.sensor_type.append([temp_name+"_x", temp_type, temp_attch])
                self.sensor_type.append([temp_name+"_y", temp_type, temp_attch])
                self.sensor_type.append([temp_name+"_z", temp_type, temp_attch])
                continue
            elif (self.mj_model.sensor_type[i] == mujoco.mjtSensor.mjSENS_GYRO):
                temp_type = "imu gyro"
                temp_attch = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, self.mj_model.sensor_objid[i]+1)
                if "imu2" in temp_name.lower():
                    self.imu2_gyro_head_id = len(self.sensor_type)
                else:
                    self.imu_gyro_head_id = len(self.sensor_type)
                self.sensor_type.append([temp_name+"_x", temp_type, temp_attch])
                self.sensor_type.append([temp_name+"_y", temp_type, temp_attch])
                self.sensor_type.append([temp_name+"_z", temp_type, temp_attch])
                continue
            elif (self.mj_model.sensor_type[i] == mujoco.mjtSensor.mjSENS_ACCELEROMETER):
                temp_type = "imu linear acc"
                temp_attch = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, self.mj_model.sensor_objid[i]+1)
                if "imu2" in temp_name.lower():
                    self.imu2_acc_head_id = len(self.sensor_type)
                else:
                    self.imu_acc_head_id = len(self.sensor_type)
                self.sensor_type.append([temp_name+"_x", temp_type, temp_attch])
                self.sensor_type.append([temp_name+"_y", temp_type, temp_attch])
                self.sensor_type.append([temp_name+"_z", temp_type, temp_attch])
                continue
            elif (self.mj_model.sensor_type[i] == mujoco.mjtSensor.mjSENS_FRAMEPOS):
                temp_type = "real position"
                temp_attch = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, self.mj_model.sensor_objid[i]+1)
                self.real_pos_head_id = len(self.sensor_type)
                self.sensor_type.append([temp_name+"_x", temp_type, temp_attch])
                self.sensor_type.append([temp_name+"_y", temp_type, temp_attch])
                self.sensor_type.append([temp_name+"_z", temp_type, temp_attch])
                continue
            elif (self.mj_model.sensor_type[i] == mujoco.mjtSensor.mjSENS_FRAMELINVEL):
                temp_type = "real velocity"
                temp_attch = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, self.mj_model.sensor_objid[i]+1)
                self.real_vel_head_id = len(self.sensor_type)
                self.sensor_type.append([temp_name+"_x", temp_type, temp_attch])
                self.sensor_type.append([temp_name+"_y", temp_type, temp_attch])
                self.sensor_type.append([temp_name+"_z", temp_type, temp_attch])
                continue
            else:
                temp_type = "unknown"
            temp_attch = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, self.mj_model.sensor_objid[i])
            self.sensor_type.append([temp_name, temp_type, temp_attch])
        
        # 读取障碍物信息
        for geom_id in range(self.mj_model.ngeom):
            geom_type = self.mj_model.geom_type[geom_id]
            if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                self.terrain_pos.append(self.mj_model.geom_pos[geom_id].copy())
                self.terrain_quat.append(self.mj_model.geom_quat[geom_id].copy())
                self.terrain_size.append(self.mj_model.geom_size[geom_id].copy() * 2)
                self.terrain_rgba.append(self.mj_model.geom_rgba[geom_id].copy())
                self.terrain_type.append('box')
            elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                self.terrain_pos.append(self.mj_model.geom_pos[geom_id].copy())
                self.terrain_quat.append(self.mj_model.geom_quat[geom_id].copy())
                size = self.mj_model.geom_size[geom_id].copy()
                self.terrain_size.append([size[0] * 2, size[0] * 2, size[1] * 2])
                self.terrain_rgba.append(self.mj_model.geom_rgba[geom_id].copy())
                self.terrain_type.append('cylinder')


def main(args=None):
    rclpy.init(args=args)
    node = mujoco_simulator()
    node.run()


if __name__ == '__main__':
    main()