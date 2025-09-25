import mujoco.viewer
import mujoco
import time
import rclpy
from rclpy.node import Node
from mit_msgs.msg import MITLowState, MITJointCommand, MITJointCommands
from geometry_msgs.msg import Pose, Twist
import yaml
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from std_srvs.srv import Empty
import numpy as np
from mujoco_lidar.scan_gen import LivoxGenerator
from mujoco_lidar.lidar_wrapper import MjLidarWrapper
from sensor_msgs.msg import PointCloud2, PointField, JointState, Imu
from std_msgs.msg import Header
from geometry_msgs.msg import TransformStamped, Vector3Stamped
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from tf2_ros import TransformBroadcaster
import math
from grid_map_msgs.msg import GridMap
from std_msgs.msg import Float32MultiArray
from tf2_ros import Buffer, TransformListener
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import MultiArrayLayout, MultiArrayDimension
from rosgraph_msgs.msg import Clock
from .mujoco_RayCaster import RayCaster


class mujoco_simulator(Node):
    """调用mujoco物理仿真, 收发ros2消息"""
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
                param = yaml.safe_load(f)  # 返回字典/列表[3](@ref)
                self.param = param["mujoco_simulator"]
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
        self.imu_pub = self.create_publisher( # 发布电机与IMU信息
            Imu, "/imu", 10
        )
        self.marker_array_pub = self.create_publisher( # 发布障碍信息
            MarkerArray, '/visualization_marker_array', 10
        )
        self.tf_sub = self.create_subscription( # 订阅tf信息
            TFMessage, '/tf_static', self.map_tf_callback, 10
        )
        self.create_timer(1.0/10.0, self.show_log) # 10Hz输出log信息
        self.create_timer(1.0/60.0, self.publish_sim_states) # 60Hz发布真值信息
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
        self.map_triggered = False

        # ==================== 实现 Height Scan 功能 ====================
        # self.elevation_map_params = self.param.get("elevation_map", {})
        # self.map_enabled = self.elevation_map_params.get("enabled", False)
        if self.param["elevation_map"]["enabled"]:
            self.get_logger().info("Height Scan (高程图) 功能已启用。")
            
            # 读取具体参数
            # self.map_topic = self.elevation_map_params.get("topic", "/mujoco_elevation_map")
            self.map_topic = self.param["elevation_map"]["topic"]
            # self.map_frame = self.elevation_map_params.get("frame_id", "world")
            self.robot_base_footprint_frame = self.param["elevation_map"]["robot_base_footprint_frame_id"]
            # self.robot_base_frame = self.first_link_name  # 使用 read_model() 中获取的机器人基座名
            self.map_size = self.param["elevation_map"]["size"]
            self.map_resolution = self.param["elevation_map"]["resolution"]
            self.elevation_map_debug = self.param["elevation_map"]["debug_info"]
            self.robot_base_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, self.first_link_name)
            update_rate = self.param["elevation_map"]["update_rate"]
            self.raycaster = RayCaster(
                self.mj_data,
                self.mj_model,
                offset_pos=(0.0, 0.0, 0.0),  # 相对于机器人基座的偏移位置
                offset_rot=(0.0, 0.0, 0.0, 1.0),  # 相对于机器人基座的偏移旋转（四元数）
                resolution=0.1,
                size=(0.8,0.5),
                debug_vis=True
            )

            
            # 计算网格维度
            self.grid_size_x = round(self.map_size[0] / self.map_resolution)
            self.grid_size_y = round(self.map_size[1] / self.map_resolution)
            # self.grid_map_pub = self.create_publisher(GridMap, self.map_topic, 100)
            self.elevation_sample_point = np.zeros((self.grid_size_x * self.grid_size_y, 3), dtype=np.float32)
            self.map_timer = self.create_timer(1.0 / update_rate, self.generate_and_publish_elevation_map)
            # 创建 TF 监听器以获取机器人实时位姿
            # self.tf_buffer = Buffer()
            # self.tf_listener = TransformListener(self.tf_buffer, self)
        # ==================== 实现 Height Scan 功能 ====================

        # 如果mjcf中有lidar_site，则读取雷达信息
        lidar_site_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "lidar_site")
        if lidar_site_id > 0:
            # 获取传感器的位置
            self.lidar_site_pos = self.mj_model.site_pos[lidar_site_id].copy()  # [x, y, z]
            self.lidar_site_quat = self.mj_model.site_quat[lidar_site_id].copy() # [w, x, y, z]
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
            # 点云发布者
            self.point_cloud_pub = self.create_publisher(
                PointCloud2, "/point_cloud", 100
            )
            self.create_timer(1.0/10.0, self.lidar_callback) # 10Hz发布点云消息

    def run(self):
        """物理仿真主循环, 默认500Hz"""
        rander_count = 0
        rander_decimation = int((1.0 / self.mj_model.opt.timestep) / 60.0)
        # 开启mujoco窗口
        with mujoco.viewer.launch_passive(self.mj_model, self.mj_data) as viewer:
            while viewer.is_running():
                # 记录当前运行时间
                self.temp_time1 = time.time()

                
                viewer.user_scn.ngeom = self.mj_model.ngeom  # 重置几何体数量，避免重复添加
                # 初始化新添加的几何体（这里是一个小球）
                for i in range(self.raycaster.num_x_points * self.raycaster.num_y_points):
                    # 增加场景中的几何体数量
                    viewer.user_scn.ngeom += 1
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[viewer.user_scn.ngeom - 1],  # 获取最后一个几何体的索引
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,                 # 几何体类型为球体
                        size=[0.02, 0, 0],                                 # 小球半径，后两个参数忽略
                        pos=self.elevation_sample_point[i],                                 # 小球的位置
                        mat=np.eye(3).flatten(),                           # 朝向矩阵（单位矩阵表示无旋转）
                        rgba=[1.0, 0.0, 0.0, 1.0]                         # 颜色和透明度（红色不透明）
                    )

                # 进行物理仿真
                if not self.pause: mujoco.mj_step(self.mj_model, self.mj_data)
                # 间隔一定step次数进行一次画面渲染，确保60fps
                if rander_count % rander_decimation == 0:
                    viewer.sync() 
                rander_count += 1

                # 处理ROS回调（非阻塞）
                rclpy.spin_once(self, timeout_sec=0.0)
    
                # 发布当前状态
                self.publish_low_state() # 200us

                self.temp_time2 = time.time()
                self.mujoco_step_time = self.temp_time2 - self.temp_time1

                # sleep 以保证仿真实时
                time_until_next_step = self.mj_model.opt.timestep - self.mujoco_step_time
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

    def publish_sim_states(self):
        """发布关节状态和世界坐标信息"""
        # 如果模型读取有错误，则不执行操作
        if self.read_error_flag: return

        # 发布地形信息
        self.publish_terrain()
            
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

        # robot_base_footprint
        robot_pos = self.mj_data.xpos[self.robot_base_id]  # 位置 [x, y, z]
        robot_quat = self.mj_data.xquat[self.robot_base_id]  # 四元数 [w, x, y, z]
        full_rotation = R.from_quat([robot_quat[1], robot_quat[2], robot_quat[3], robot_quat[0]])
        euler_angles = full_rotation.as_euler('xyz', degrees=False)
        yaw_angle = euler_angles[2]
        yaw_only_rotation = R.from_euler('z', yaw_angle, degrees=False)
        quat_yaw_only = yaw_only_rotation.as_quat()
        # --- 修改結束 ---
        footprint_transform = TransformStamped()
        footprint_transform.header.stamp = self.get_clock().now().to_msg()
        footprint_transform.header.frame_id = "world"
        footprint_transform.child_frame_id = self.robot_base_footprint_frame
        footprint_transform.transform.translation.x = float(robot_pos[0]) #
        footprint_transform.transform.translation.y = float(robot_pos[1]) #
        footprint_transform.transform.translation.z = 0.0 #
        footprint_transform.transform.rotation.w = float(quat_yaw_only[3])  # w
        footprint_transform.transform.rotation.x = float(quat_yaw_only[0])  # x
        footprint_transform.transform.rotation.y = float(quat_yaw_only[1])  # y
        footprint_transform.transform.rotation.z = float(quat_yaw_only[2])  # z
        self.tf_broadcaster.sendTransform(footprint_transform)
        
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

        time_stamp = self.get_clock().now().to_msg()
        
        # 设置并发布点云信息
        point_cloud_msg = PointCloud2()
        point_cloud_msg.header = Header()
        point_cloud_msg.header.frame_id = 'lidar' 
        point_cloud_msg.fields = [ # 定义点云字段 (x, y, z)
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
        ]
        point_cloud_msg.header.stamp = time_stamp
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
        t.header.stamp = time_stamp # 设置消息头
        t.header.frame_id = self.first_link_name # 设置父坐标系
        t.child_frame_id = 'lidar'
        t.transform.translation.x = self.lidar_site_pos[0] # 设置变换
        t.transform.translation.y = self.lidar_site_pos[1]
        t.transform.translation.z = self.lidar_site_pos[2]
        t.transform.rotation.w = self.lidar_site_quat[0]
        t.transform.rotation.x = self.lidar_site_quat[1]
        t.transform.rotation.y = self.lidar_site_quat[2]
        t.transform.rotation.z = self.lidar_site_quat[3]
        self.tf_broadcaster.sendTransform(t) # 发布变换

        # 单独发布imu数据给感知用
        imu_data_msg = self.low_state_msg.imu
        imu_data_msg.header.frame_id = "imu_frame"
        imu_data_msg.header.stamp = time_stamp
        if self.param["g_unit"] == "g":
            imu_data_msg.linear_acceleration.x /= 9.80665
            imu_data_msg.linear_acceleration.y /= 9.80665
            imu_data_msg.linear_acceleration.z /= 9.80665
        elif self.param["g_unit"] == "m/s^2":
            pass
        else:
            self.get_logger().error(f"未知的重力单位: {self.param['g_unit']}, 请检查参数设置")
            return
        self.imu_pub.publish(imu_data_msg)
        # ======================= 关键补充：发布IMU的TF变换 =======================
        # 这个变换告诉系统 "imu_frame" 在机器人上的确切位置
        imu_transform = TransformStamped()
        imu_transform.header.stamp = time_stamp
        imu_transform.header.frame_id = self.first_link_name  # 父坐标系：机器人基座
        imu_transform.child_frame_id = "imu_frame"            # 子坐标系：IMU
        # IMU在机器人中心，所以平移为0，旋转为单位四元数
        imu_transform.transform.translation.x = 0.0
        imu_transform.transform.translation.y = 0.0
        imu_transform.transform.translation.z = 0.0
        imu_transform.transform.rotation.w = 1.0
        imu_transform.transform.rotation.x = 0.0
        imu_transform.transform.rotation.y = 0.0
        imu_transform.transform.rotation.z = 0.0
        self.tf_broadcaster.sendTransform(imu_transform)
        # ====================================================================

    def show_log(self):
        """输出日志"""
        pass
        # self.get_logger().info(f"物理仿真渲染耗时: {self.mujoco_step_time:.4f} 秒")

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
        # 根据PD控制器计算输出力矩
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

    def publish_terrain(self):
        marker_array = MarkerArray()
        id = 0

        for i in range(len(self.terrain_pos)):  # boxes 里存放你的障碍物信息
            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "mujoco"
            marker.id = id   # 每个 marker 必须有唯一 id
            id += 1
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            marker.pose.position.x = self.terrain_pos[i][0]
            marker.pose.position.y = self.terrain_pos[i][1]
            marker.pose.position.z = self.terrain_pos[i][2]

            marker.pose.orientation.w = self.terrain_quat[i][0]
            marker.pose.orientation.x = self.terrain_quat[i][1]
            marker.pose.orientation.y = self.terrain_quat[i][2]
            marker.pose.orientation.z = self.terrain_quat[i][3]

            marker.scale.x = self.terrain_size[i][0]
            marker.scale.y = self.terrain_size[i][1]
            marker.scale.z = self.terrain_size[i][2]

            marker.color.r = float(self.terrain_rgba[i][0])
            marker.color.g = float(self.terrain_rgba[i][1])
            marker.color.b = float(self.terrain_rgba[i][2])
            marker.color.a = float(self.terrain_rgba[i][3])

            marker_array.markers.append(marker)

        self.marker_array_pub.publish(marker_array)

    def map_tf_callback(self, msg: TFMessage):
        if self.map_triggered:
            return
        
        for transform in msg.transforms:
            child = transform.child_frame_id

            # print(transform)
            if child == "odom":
                self.get_logger().info(f"第一次收到 {child} 的 tf，执行函数！")
                self.broadcaster = StaticTransformBroadcaster(self)

                # 定义一个静态变换
                t = TransformStamped()
                t.header.stamp = self.get_clock().now().to_msg()
                t.header.frame_id = 'world'        # 父坐标系
                t.child_frame_id = 'map'   # 子坐标系

                # 平移 (x, y, z)
                t.transform.translation.x = float(self.sensor_data_list[self.real_pos_head_id + 0])
                t.transform.translation.y = float(self.sensor_data_list[self.real_pos_head_id + 1])
                t.transform.translation.z = 0.0

                # 四元数 (w, x, y, z)
                t.transform.rotation.w = float(self.sensor_data_list[self.imu_quat_head_id + 0])
                t.transform.rotation.x = float(self.sensor_data_list[self.imu_quat_head_id + 1])
                t.transform.rotation.y = float(self.sensor_data_list[self.imu_quat_head_id + 2])
                t.transform.rotation.z = float(self.sensor_data_list[self.imu_quat_head_id + 3])

                # 发布一次即可
                self.broadcaster.sendTransform(t)
                self.get_logger().info("发布了静态坐标变换 world -> camera_link")
                self.map_triggered = True
                break

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

    def generate_and_publish_elevation_map(self):
        """
        通过循环调用 MuJoCo 射线投射功能，并直接从 mj_data 读取机器人位姿，生成并发布高程图。
        """
        # 读取pelvis的pos和quat
        robot_pos = self.mj_data.xpos[self.robot_base_id]  # 位置 [x, y, z]
        robot_quat = self.mj_data.xquat[self.robot_base_id]  # 四元数 [w, x, y, z]

        self.elevation_sample_point = self.raycaster.update_elevation_data(robot_pos,robot_quat)

        print(self.elevation_sample_point[:,2])
        
        # r = R.from_quat([robot_quat[1], robot_quat[2], robot_quat[3], robot_quat[0]])
        # yaw = r.as_euler('xyz', degrees=False)[2]

        # elevation_data = np.full((self.grid_size_x, self.grid_size_y), -1.0, dtype=np.float32)
        # cos_yaw = math.cos(yaw)
        # sin_yaw = math.sin(yaw)

        # # 用于存放射线检测（raycast）命中的几何体（geom）ID 的占位数组
        # geomid_placeholder = np.array([-1], dtype=np.int32)

        # for i in range(self.grid_size_x):
        #     for j in range(self.grid_size_y):
        #         px_robot = (self.grid_size_x / 2.0 - i) * self.map_resolution
        #         py_robot = (self.grid_size_y / 2.0 - j) * self.map_resolution
        #         px_world = robot_pos[0] + px_robot * cos_yaw - py_robot * sin_yaw
        #         py_world = robot_pos[1] + px_robot * sin_yaw + py_robot * cos_yaw
        #         # 固定机器人正上方离地3m，发射一条垂直向下的射线
        #         ray_start = np.array([px_world, py_world, 3.0], dtype=np.float64).reshape(3, 1)
        #         ray_dir = np.array([0, 0, -1.0], dtype=np.float64).reshape(3, 1)

        #         # 屏蔽掉机器人
        #         # G1的gemo全部设置成group(1)，group(0)是默认,group(2)是地形相关的
        #         # 对应顺序geomgroup = (group(0), group(1), group(2), group(3), group(4), group(5))
        #         geomgroup = (False, False, True, False, False, False)
        #         hit_dist = mujoco.mj_ray(self.mj_data.model, self.mj_data, ray_start, ray_dir,
        #                                  geomgroup, 1, -1, geomid_placeholder)

        #         if hit_dist > 0:
        #             height = ray_start[2] - hit_dist
        #             elevation_data[i, j] = height

        # # 打印高层图数据
        # if self.elevation_map_debug:
        #     print("--- 高程图数据 ---")
        #     # 遍历每一行并格式化打印
        #     for i, row in enumerate(elevation_data):
        #         # 将行内每个浮点数格式化为保留两位的字符串，并用空格连接
        #         row_str = " ".join([f"{val:6.2f}" for val in row])
        #         print(f"第 {i:02d} 行: {row_str}")
        #     print("--------------------")
        #     print("--------------------")

        # 创建并填充 GridMap 消息
        # grid_map_msg = GridMap()
        # grid_map_msg.header.stamp = self.get_clock().now().to_msg()
        # grid_map_msg.header.frame_id = "robot_base_footprint"
        # grid_map_msg.info.resolution = self.map_resolution
        # grid_map_msg.info.length_x = self.map_size[0]
        # grid_map_msg.info.length_y = self.map_size[1]
        # grid_map_msg.info.pose.position.x = robot_pos[0]
        # grid_map_msg.info.pose.position.y = robot_pos[1]
        # grid_map_msg.info.pose.position.z = 0.0
        # grid_map_msg.layers = ['elevation']

        # # 填充 elevation_layer 的 layout
        # elevation_layer = Float32MultiArray()
        # layout = MultiArrayLayout()
        # # 维度0 (行)
        # dim_row = MultiArrayDimension()
        # dim_row.label = "row"
        # dim_row.size = self.grid_size_x
        # dim_row.stride = self.grid_size_x * self.grid_size_y
        # layout.dim.append(dim_row)
        # # 维度1 (列)
        # dim_col = MultiArrayDimension()
        # dim_col.label = "column"
        # dim_col.size = self.grid_size_y
        # dim_col.stride = self.grid_size_y
        # layout.dim.append(dim_col)
        # layout.data_offset = 0
        # elevation_layer.layout = layout
        # elevation_layer.data = elevation_data.flatten(order='F').tolist()
        # grid_map_msg.data.append(elevation_layer)
        # self.grid_map_pub.publish(grid_map_msg)

    def unpause_callback(self, request, response):
        """仿真启动回调"""
        # 处理 unpause 请求
        self.get_logger().info("Unpause service called")
        # 启动物理仿真
        self.pause = False
        # 应用第一个关键帧作为初始位
        if self.keyframe_count > 0: mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, 0)
        # 返回空响应
        return response  
         
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
        self.terrain_pos = [] # 记录场景中的box，目前还不支持其他的形状
        self.terrain_size = []
        self.terrain_quat = []
        self.terrain_rgba = []
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
        
        # 读取所有的box障碍物
        
        for geom_id in range(self.mj_model.ngeom):
            geom_type = self.mj_model.geom_type[geom_id]
            if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                self.terrain_pos.append(self.mj_model.geom_pos[geom_id].copy())
                self.terrain_quat.append(self.mj_model.geom_quat[geom_id].copy())
                self.terrain_size.append(self.mj_model.geom_size[geom_id].copy() * 2) # mjcf中是半尺寸，这里改为全尺寸
                self.terrain_rgba.append(self.mj_model.geom_rgba[geom_id].copy())

def main(args=None):
    rclpy.init(args=args)
    node = mujoco_simulator()
    node.run()

if __name__ == '__main__':
    main()


