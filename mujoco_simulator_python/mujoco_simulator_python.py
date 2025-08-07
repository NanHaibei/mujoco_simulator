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
from threading import Thread
from rclpy.executors import MultiThreadedExecutor
import numpy as np
from mujoco_lidar.scan_gen import LivoxGenerator
from mujoco_lidar.scan_gen import generate_grid_scan_pattern
from mujoco_lidar.lidar_wrapper import MjLidarWrapper
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


class mujoco_simulator(Node):
    """调用mujoco物理仿真, 收发ros2消息"""
    def __init__(self, yaml_path):
        super().__init__('mujoco_simulator')

        # 读取yaml文件
        with open(yaml_path, 'r') as f:
            try:
                param = yaml.safe_load(f)  # 返回字典/列表[3](@ref)
                self.param = param["mujoco_simulator"]
            except yaml.YAMLError as e:
                print(f"YAML解析失败: {e}")

        # 获取mjcf路径
        robot_pkg_path = os.path.join(get_package_share_directory('robot_description'))
        model_name = self.param["modelName"]
        if "G1" in model_name:
            model_type = "G1"
        elif "S1" in model_name:
            model_type = "S1"
        elif "S2" in model_name:
            model_type = "S2"
        else:
            raise ValueError("Unsupported model type in mujoco_simulator_python.py")
        mjcf_path = robot_pkg_path + "/" + model_type + "/mjcf/scene_" + model_name + ".xml"

        # 实例化mujoco的model和data
        self.mj_model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.sensor_data_list = list(self.mj_data.sensordata)

        # 读取模型信息并输出
        self.read_model()
        if self.param["modelTableFlag"]: self.show_model()

        # 设置控制回调
        mujoco.set_mjcb_control(self.pd_controller)

        # 底层信息发布
        self.lowState_pub = self.create_publisher(
            MITLowState,  # 假设低频状态消息类型为String
            self.param["lowStateTopic"],  # 替换为实际的低频状态话题名称
            10  # 队列大小
        )

        # 底层命令接收
        self.jointCommand_sub = self.create_subscription(
            MITJointCommands,  # 假设底层命令消息类型为String
            self.param["jointCommandsTopic"],  # 替换为实际的底层命令话题名称
            self.low_cmd_callback,
            10  # 队列大小
        )
        # 仿真启动服务
        self.unpause_server = self.create_service(
            Empty,  # 假设服务类型为Unpause
            self.param["unPauseService"],  # 替换为实际的服务名称
            self.unpause_callback
        )

        # 初始化变量
        self.low_state_msg = MITLowState()
        self.low_state_msg.joint_states.position = [0.0 for _ in range(self.mj_model.nu)]
        self.low_state_msg.joint_states.velocity = [0.0 for _ in range(self.mj_model.nu)]
        self.low_state_msg.joint_states.effort = [0.0 for _ in range(self.mj_model.nu)]
        self.low_cmd_msg = MITJointCommands()
        self.low_cmd_msg.commands = [MITJointCommand() for _ in range(self.mj_model.nu)]
        self.read_error_flag = False  # 传感器读取错误标志
        if self.param["initPauseFlag"]: self.pause = True

        # 测试雷达
        livox_generator = LivoxGenerator("mid360")
        self.rays_theta, self.rays_phi = livox_generator.sample_ray_angles()
        # self.rays_theta, self.rays_phi = generate_grid_scan_pattern(
        #     num_ray_cols=360,  # 水平分辨率
        #     num_ray_rows=64,   # 垂直分辨率
        #     theta_range=(-np.pi, np.pi),    # 水平扫描范围（弧度）
        #     phi_range=(-np.pi, np.pi)   # 垂直扫描范围（弧度）
        # )
        self.lidar_sim = MjLidarWrapper(
            self.mj_model, 
            self.mj_data, 
            site_name="lidar_site",  # 与MJCF中的<site name="...">匹配
            args={
                "enable_profiling": False, # 启用性能分析（可选）
                "verbose": False           # 显示详细信息（可选）
            }
        )
        self.lidar_count = 0
        self.point_cloud_pub = self.create_publisher(
            PointCloud2,
            "/point_cloud",
            100
        )
        self.tf_broadcaster = TransformBroadcaster(self)


        self.mujoco_step_time = 0.0

        self.create_timer(1.0/10.0, self.lidar_callback)
        self.create_timer(1.0/10.0, self.show_log)

        

    def run(self):
        """物理仿真主循环, 默认500Hz"""

        # 开启mujoco窗口
        with mujoco.viewer.launch_passive(self.mj_model, self.mj_data) as viewer:
            while viewer.is_running():
                # 记录当前运行时间
                self.temp_time1 = time.time()

                # 进行物理仿真，渲染画面
                # if not self.pause: mujoco.mj_step(self.mj_model, self.mj_data)
                mujoco.mj_step(self.mj_model, self.mj_data)
                viewer.sync() # 影响实时性的大头

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
                 

    def lidar_callback(self):
        self.lidar_sim.update_scene(self.mj_model, self.mj_data)
        points = self.lidar_sim.get_lidar_points(self.rays_phi, self.rays_theta, self.mj_data)
        self.point_cloud_pub.publish(self.numpy_to_pointcloud2(points))
        self.broadcast_timer_callback()

    def show_log(self):
        """输出日志"""
        self.get_logger().info(f"物理仿真渲染耗时: {self.mujoco_step_time:.4f} 秒")

    def numpy_to_pointcloud2(self, points):
        """将numpy数组转换为PointCloud2消息"""
        # 创建PointCloud2消息对象
        msg = PointCloud2()
        
        # 设置消息头
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'  # 或者你需要的坐标系
        
        # 定义点云字段 (x, y, z)
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
        ]
        
        # 设置点云属性
        msg.is_bigendian = False
        msg.point_step = 12  # 每个点12字节 (3个float32 * 4字节)
        msg.row_step = msg.point_step * len(points)
        msg.height = 1  # 非结构化点云
        msg.width = len(points)
        msg.is_dense = True  # 没有无效点
        
        # 关键：将numpy数组转换为字节数据
        msg.data = points.astype(np.float32).tobytes()
        
        return msg

    def broadcast_timer_callback(self):
        # 创建变换消息
        t = TransformStamped()
        
        # 设置消息头
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'world'
        
        # 设置变换（这里设置为原点，无旋转）
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 1.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 0.0
        
        # 发布变换
        self.tf_broadcaster.sendTransform(t)

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
         
    def ros_spin(self, node):
        """放在第二个线程中运行,执行ros2回调"""
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        executor.spin()

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
    yaml_path = os.path.join(
        get_package_share_directory('mujoco_simulator_python'),
        'config', 'simulate.yaml'
    )
    node = mujoco_simulator(yaml_path)
    node.run()

if __name__ == '__main__':
    main()
