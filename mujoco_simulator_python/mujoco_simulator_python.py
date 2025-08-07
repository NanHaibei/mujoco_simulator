import mujoco.viewer
import mujoco
import time
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from mit_msgs.msg import MITLowState, MITJointCommand, MITJointCommands
import os
import yaml
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from std_srvs.srv import Empty
from rclpy.executors import MultiThreadedExecutor
import numpy as np
from mujoco_lidar.scan_gen import LivoxGenerator
from mujoco_lidar.scan_gen import generate_grid_scan_pattern
from mujoco_lidar.lidar_wrapper import MjLidarWrapper
from sensor_msgs.msg import PointCloud2, PointField, JointState
from std_msgs.msg import Header
from geometry_msgs.msg import TransformStamped, Vector3Stamped
from tf2_ros import TransformBroadcaster


class mujoco_simulator(Node):
    """调用mujoco物理仿真, 收发ros2消息"""
    def __init__(self):
        super().__init__('mujoco_simulator')

        # 读取launch中传来的参数
        self.declare_parameter('use_lidar', False)  # launch文件中的参数
        self.declare_parameter('yaml_path', " ")  # launch文件中的参数
        self.declare_parameter('sensor_yaml_path', " ")  # launch文件中的参数
        self.declare_parameter('mjcf_path', " ")  # launch文件中的参数
        self.use_lidar = self.get_parameter('use_lidar').get_parameter_value().bool_value
        yaml_path = self.get_parameter('yaml_path').get_parameter_value().string_value
        mjcf_path = self.get_parameter('mjcf_path').get_parameter_value().string_value
        sensor_yaml_path = self.get_parameter('sensor_yaml_path').get_parameter_value().string_value

        # 读取yaml文件
        with open(yaml_path, 'r') as f:
            try:
                param = yaml.safe_load(f)  # 返回字典/列表[3](@ref)
                self.param = param["mujoco_simulator"]
            except yaml.YAMLError as e:
                print(f"YAML解析失败: {e}")

        # 读取传感器yaml文件
        with open(sensor_yaml_path, 'r') as f:
            try:
                param = yaml.safe_load(f)  # 返回字典/列表[3](@ref)
                print(param)
                self.param.update(param["sensor_cfg"])
            except yaml.YAMLError as e:
                print(f"YAML解析失败: {e}")


        # 实例化mujoco的model和data
        self.mj_model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.sensor_data_list = list(self.mj_data.sensordata)

        # 读取模型信息并输出
        self.read_model()
        if self.param["modelTableFlag"]: self.show_model()

        # 设置控制回调
        mujoco.set_mjcb_control(self.pd_controller)

        # 声明ROS2接口
        self.lowState_pub = self.create_publisher( # 发布电机与IMU信息
            MITLowState, self.param["lowStateTopic"], 10
        )
        self.jointCommand_sub = self.create_subscription( # 接收控制器命令
            MITJointCommands, self.param["jointCommandsTopic"], self.low_cmd_callback, 10
        )
        self.unpause_server = self.create_service( # 仿真启动服务
            Empty, self.param["unPauseService"], self.unpause_callback
        )   
        self.joint_state_pub = self.create_publisher( # 发布关节状态
            JointState, "/joint_states", 10
        )
        self.real_vel_pub = self.create_publisher( # 发布线速度真值
            Vector3Stamped, "/sim_real_vel", 10
        )
        self.create_timer(1.0/10.0, self.show_log) # 10Hz输出log信息
        self.create_timer(1.0/10.0, self.publish_sim_states) # 10Hz发布真值信息
        self.tf_broadcaster = TransformBroadcaster(self)  # 发布tf变换

        # 初始化变量
        self.low_state_msg = MITLowState()
        self.low_state_msg.joint_states.position = [0.0 for _ in range(self.mj_model.nu)]
        self.low_state_msg.joint_states.velocity = [0.0 for _ in range(self.mj_model.nu)]
        self.low_state_msg.joint_states.effort = [0.0 for _ in range(self.mj_model.nu)]
        self.low_cmd_msg = MITJointCommands()
        self.low_cmd_msg.commands = [MITJointCommand() for _ in range(self.mj_model.nu)]
        self.read_error_flag = False  # 传感器读取错误标志
        self.pause = True if self.param["initPauseFlag"] else False
        self.mujoco_step_time = 0.0

        # 如果使用雷达
        if self.use_lidar:
            # 检查是否有lidar_site
            lidar_site_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "lidar_site")
            if lidar_site_id <= 0:
                raise ValueError("运行雷达仿真但是MJCF文件中未找到lidar_site")
            # 设置雷达类型
            livox_generator = LivoxGenerator("mid360")
            self.rays_theta, self.rays_phi = livox_generator.sample_ray_angles()
            # 创建雷达句柄
            self.lidar_sim = MjLidarWrapper(
                self.mj_model, 
                self.mj_data, 
                site_name="lidar_site",  # 与MJCF中的<site name="...">匹配
                args={
                    "enable_profiling": False, # 启用性能分析（可选）
                    "verbose": False           # 显示详细信息（可选）
                }
            )
            
            # 发布点云与坐标系
            self.point_cloud_pub = self.create_publisher(
                PointCloud2, "/point_cloud", 100
            )
            self.create_timer(1.0/10.0, self.lidar_callback) # 10Hz发布点云消息

    def run(self):
        """物理仿真主循环, 默认500Hz"""

        # 开启mujoco窗口
        with mujoco.viewer.launch_passive(self.mj_model, self.mj_data) as viewer:
            while viewer.is_running():
                # 记录当前运行时间
                self.temp_time1 = time.time()

                # 进行物理仿真，渲染画面
                if not self.pause: mujoco.mj_step(self.mj_model, self.mj_data)
                # mujoco.mj_step(self.mj_model, self.mj_data)
                viewer.sync() 

                # 处理ROS回调（非阻塞）
                rclpy.spin_once(self, timeout_sec=0.0)
    
                self.temp_time2 = time.time()
                self.mujoco_step_time = self.temp_time2 - self.temp_time1

                # 发布当前状态
                self.publish_low_state() # 200us

                # sleep 以保证仿真实时
                time_until_next_step = self.mj_model.opt.timestep - (time.time() - self.temp_time1)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

    def publish_sim_states(self):
        """发布关节状态和世界坐标信息"""
        # 如果模型读取有错误，则不执行操作
        if self.read_error_flag: return
            
        # 发布关节信息
        joint_state = JointState()
        joint_state.header.stamp = self.get_clock().now().to_msg()
        joint_state.name = self.joint_name.copy()
        joint_state.position = self.sensor_data_list[self.joint_pos_head_id : self.joint_pos_head_id + self.mj_model.nu]
        joint_state.velocity = self.sensor_data_list[self.joint_vel_head_id : self.joint_vel_head_id + self.mj_model.nu]
        joint_state.effort = self.sensor_data_list[self.joint_tor_head_id : self.joint_tor_head_id + self.mj_model.nu]
        self.joint_state_pub.publish(joint_state)
        
        # 发布世界坐标信息
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = "world"
        transform.child_frame_id = self.first_link_name
        transform.transform.translation.x = float(self.sensor_data_list[self.real_pos_head_id + 0])
        transform.transform.translation.y = float(self.sensor_data_list[self.real_pos_head_id + 1])
        transform.transform.translation.z = float(self.sensor_data_list[self.real_pos_head_id + 2])
        transform.transform.rotation.w = float(self.sensor_data_list[self.imu_quat_head_id + 0])
        transform.transform.rotation.x = float(self.sensor_data_list[self.imu_quat_head_id + 1])
        transform.transform.rotation.y = float(self.sensor_data_list[self.imu_quat_head_id + 2])
        transform.transform.rotation.z = float(self.sensor_data_list[self.imu_quat_head_id + 3])
        self.tf_broadcaster.sendTransform(transform)
        
        # 发布实际速度信息
        if self.real_vel_head_id != 999999:  # 检查是否有实际速度传感器
            real_vel = Vector3Stamped()
            real_vel.header.stamp = self.get_clock().now().to_msg()
            real_vel.header.frame_id = self.first_link_name
            real_vel.vector.x = float(self.sensor_data_list[self.real_vel_head_id + 0])
            real_vel.vector.y = float(self.sensor_data_list[self.real_vel_head_id + 1])
            real_vel.vector.z = float(self.sensor_data_list[self.real_vel_head_id + 2])
            self.real_vel_pub.publish(real_vel)


    def lidar_callback(self):
        """获取点云信息并发布"""

        # 获取点云信息
        self.lidar_sim.update_scene(self.mj_model, self.mj_data)
        points = self.lidar_sim.get_lidar_points(self.rays_phi, self.rays_theta, self.mj_data)
        
        # 设置并发布点云信息
        point_cloud_msg = PointCloud2()
        point_cloud_msg.header = Header()
        point_cloud_msg.header.frame_id = 'lidar' 
        point_cloud_msg.fields = [ # 定义点云字段 (x, y, z)
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
        ]
        point_cloud_msg.header.stamp = self.get_clock().now().to_msg()
        point_cloud_msg.is_bigendian = False
        point_cloud_msg.point_step = 12  # 每个点12字节 (3个float32 * 4字节)
        point_cloud_msg.row_step = point_cloud_msg.point_step * len(points)
        point_cloud_msg.height = 1  # 非结构化点云
        point_cloud_msg.width = len(points)
        point_cloud_msg.is_dense = True  # 没有无效点
        point_cloud_msg.data = points.astype(np.float32).tobytes() # 将numpy数组转换为字节数据
        self.point_cloud_pub.publish(point_cloud_msg) # 发布点云信息

        # 点云相对于base_link的坐标转换
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg() # 设置消息头
        t.header.frame_id = self.first_link_name # 设置父坐标系
        t.child_frame_id = 'lidar'
        t.transform.translation.x = self.param["lidar"]["translation"]["x"] # 设置变换
        t.transform.translation.y = self.param["lidar"]["translation"]["y"]
        t.transform.translation.z = self.param["lidar"]["translation"]["z"]
        t.transform.rotation.x = self.param["lidar"]["rotation"]["x"]
        t.transform.rotation.y = self.param["lidar"]["rotation"]["y"]
        t.transform.rotation.z = self.param["lidar"]["rotation"]["z"]
        t.transform.rotation.w = self.param["lidar"]["rotation"]["w"]
        self.tf_broadcaster.sendTransform(t) # 发布变换

    def show_log(self):
        """输出日志"""
        self.get_logger().info(f"物理仿真渲染耗时: {self.mujoco_step_time:.4f} 秒")

    def pd_controller(self, model, data):
        """mujoco控制回调,根据命令值计算力矩

        Args:
            model : mj_model
            data : mj_data
        """
        # 向量化一次完成运算
        kp_cmd_list = np.array([cmd.kp for cmd in self.low_cmd_msg.commands])
        kd_cmd_list = np.array([cmd.kd for cmd in self.low_cmd_msg.commands])
        pos_cmd_list = np.array([cmd.pos for cmd in self.low_cmd_msg.commands])
        vel_cmd_list = np.array([cmd.vel for cmd in self.low_cmd_msg.commands])
        eff_cmd_list = np.array([cmd.eff for cmd in self.low_cmd_msg.commands])
        sensor_pos = np.array(self.sensor_data_list[self.joint_pos_head_id:self.joint_pos_head_id + self.mj_model.nu])
        sensor_vel = np.array(self.sensor_data_list[self.joint_vel_head_id:self.joint_vel_head_id + self.mj_model.nu])

        data.ctrl = kp_cmd_list * (pos_cmd_list - sensor_pos) + kd_cmd_list * (vel_cmd_list - sensor_vel) + eff_cmd_list

    def low_cmd_callback(self, msg: MITJointCommands):
        """控制器命令回调函数

        Args:
            msg (MITJointCommands): 控制器命令
        """
        # 如果读取错误标志为真，则不处理命令
        if self.read_error_flag: return
        if len(msg.commands) != self.mj_model.nu:
            self.get_logger().error(f"命令长度 {len(msg.commands)} 不等于模型关节数 {self.mj_model.nu} ,请检查")
            return

        # 将命令值保存到成员变量
        self.low_cmd_msg = msg

    def publish_low_state(self):
        """发布机器人状态"""

        # 将数据切片一次操作
        self.sensor_data_list = list(self.mj_data.sensordata)

        # 如果读取错误标志为真，则不发布状态
        if self.read_error_flag: return
        
        # 更新电机状态
        self.low_state_msg.joint_states.position = self.sensor_data_list[self.joint_pos_head_id : self.joint_pos_head_id + self.mj_model.nu]
        self.low_state_msg.joint_states.velocity = self.sensor_data_list[self.joint_vel_head_id : self.joint_vel_head_id + self.mj_model.nu]
        self.low_state_msg.joint_states.effort = self.sensor_data_list[self.joint_tor_head_id : self.joint_tor_head_id + self.mj_model.nu]
        # 更新IMU状态
        self.low_state_msg.imu.orientation.w = self.sensor_data_list[self.imu_quat_head_id + 0]
        self.low_state_msg.imu.orientation.x = self.sensor_data_list[self.imu_quat_head_id + 1]
        self.low_state_msg.imu.orientation.y = self.sensor_data_list[self.imu_quat_head_id + 2]
        self.low_state_msg.imu.orientation.z = self.sensor_data_list[self.imu_quat_head_id + 3]
        self.low_state_msg.imu.angular_velocity.x = self.sensor_data_list[self.imu_gyro_head_id + 0]
        self.low_state_msg.imu.angular_velocity.y = self.sensor_data_list[self.imu_gyro_head_id + 1]
        self.low_state_msg.imu.angular_velocity.z = self.sensor_data_list[self.imu_gyro_head_id + 2]
        self.low_state_msg.imu.linear_acceleration.x = self.sensor_data_list[self.imu_acc_head_id + 0]
        self.low_state_msg.imu.linear_acceleration.y = self.sensor_data_list[self.imu_acc_head_id + 1]
        self.low_state_msg.imu.linear_acceleration.z = self.sensor_data_list[self.imu_acc_head_id + 2]
    
        # 更新时间戳
        self.low_state_msg.stamp = self.get_clock().now().to_msg()
        self.low_state_msg.joint_states.header.stamp = self.low_state_msg.stamp
        self.low_state_msg.imu.header.stamp = self.low_state_msg.stamp

        # 发布当前状态
        self.lowState_pub.publish(self.low_state_msg)

    def unpause_callback(self, request, response):
        """仿真启动回调"""
        # 处理 unpause 请求
        self.get_logger().info("Unpause service called")
        # 启动物理仿真
        self.pause = False
        # 应用第一个关键帧作为初始位
        if self.keyframe_count > 0: mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, 0)

        return response  # 返回空响应
         
    def show_model(self):
        """
        输出读取到的机器人模型信息
        GPT写的,没检查过
        """

        console = Console()

        # --------- 总信息 ---------
        console.print("[bold cyan]------------读取到的环境与模型信息如下------------[/bold cyan]")
        console.print(f"[bold]model name:[/bold] [green]{self.model_name}[/green]   [bold]time step:[/bold] [yellow]{self.time_step:.6f}s[/yellow]")

        # --------- Joint 信息 ---------
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

        # --------- Link 信息 ---------
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

        # --------- Sensor 信息 ---------
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
            elif i == self.real_pos_head_id: head_id_name = "real pos head"
            elif i == self.real_vel_head_id: head_id_name = "real vel head"
            sensor_table.add_row(str(i), sensor[0], sensor[1], sensor[2], head_id_name)
        console.print(sensor_table)

        console.print(Panel("[bold green]如果仿真遇到问题,请检查上述信息是否正确,物理仿真进行中...[/bold green]"))

        # --------- Keyframe 检查 ---------
        if self.keyframe_count == 0:
            console.print("[bold yellow][WARN][/bold yellow] 未发现keyframe,请检查模型")

        # --------- 传感器错误检查 ---------
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
        self.real_pos_head_id = 999999
        self.real_vel_head_id = 999999
        # 读取模型名称
        self.model_name = self.mj_model.names.split(b'\x00', 1)[0].decode('utf-8')
        # 读取加载的模型时间步长
        self.time_step = self.mj_model.opt.timestep
        # 加载关键帧
        self.keyframe_count = self.mj_model.nkey
        if self.keyframe_count > 0:
            mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, 0)

        # 遍历所有joint,读取参数
        for i in range(self.mj_model.njnt):
            # 忽略自由关节
            if(self.mj_model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE):
                continue
            self.joint_name.append((mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)))
            self.joint_pos_range.append(self.mj_model.jnt_range[i])
            self.joint_torque_range.append(self.mj_model.jnt_actfrcrange[i])
            joint_dofadr = self.mj_model.jnt_dofadr[i]
            self.joint_friction.append(self.mj_model.dof_frictionloss[joint_dofadr])
            self.joint_damping.append(self.mj_model.dof_damping[joint_dofadr])

        # 遍历所有link,读取参数
        for i in range(self.mj_model.nbody):
            # 忽略world link
            if(mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, i) == "world"):
                continue
            self.link_name.append((mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, i)))
            self.link_mass.append(self.mj_model.body_mass[i])

        # 记录第一个link的名字作为base_link，不同机器人base_link名称不同
        self.first_link_name = self.link_name[0]

        # 遍历所有sensor,读取参数
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
                self.imu_quat_head_id = len(self.sensor_type) 
                self.sensor_type.append([temp_name+"_w", temp_type, temp_attch])
                self.sensor_type.append([temp_name+"_x", temp_type, temp_attch])
                self.sensor_type.append([temp_name+"_y", temp_type, temp_attch])
                self.sensor_type.append([temp_name+"_z", temp_type, temp_attch])
                continue
            elif (self.mj_model.sensor_type[i] == mujoco.mjtSensor.mjSENS_GYRO):
                temp_type = "imu gyro"
                temp_attch = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, self.mj_model.sensor_objid[i]+1)
                self.imu_gyro_head_id = len(self.sensor_type)
                self.sensor_type.append([temp_name+"_x", temp_type, temp_attch])
                self.sensor_type.append([temp_name+"_y", temp_type, temp_attch])
                self.sensor_type.append([temp_name+"_z", temp_type, temp_attch])
                continue
            elif (self.mj_model.sensor_type[i] == mujoco.mjtSensor.mjSENS_ACCELEROMETER):
                temp_type = "imu linear acc"
                temp_attch = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, self.mj_model.sensor_objid[i]+1)
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

def main(args=None):
    rclpy.init(args=args)
    node = mujoco_simulator()
    node.run()

if __name__ == '__main__':
    main()
