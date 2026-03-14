import mujoco
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField, Imu
from std_msgs.msg import Header
from geometry_msgs.msg import TransformStamped

from .base_plugin import BasePlugin


class LidarPlugin(BasePlugin):
    """激光雷达插件
    
    负责获取点云信息并发布。
    """
    
    def init(self):
        """初始化激光雷达插件"""
        # 检查是否启用激光雷达
        if not self.simulator.param.get("enableLidar", True):
            self.enabled = False
            return
        
        # 查找lidar_site
        lidar_site_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "lidar_site")
        if lidar_site_id < 0:
            self.simulator.get_logger().warn("未找到 lidar_site，激光雷达插件禁用")
            self.enabled = False
            return
        
        # 获取传感器的位置和姿态
        self.lidar_site_pos = self.mj_model.site_pos[lidar_site_id].copy()  # [x, y, z]
        self.lidar_site_quat = self.mj_model.site_quat[lidar_site_id].copy()  # [w, x, y, z]
        
        # 设置雷达类型
        from mujoco_lidar.scan_gen import LivoxGenerator
        self.livox_generator = LivoxGenerator("mid360")
        self.rays_theta, self.rays_phi = self.livox_generator.sample_ray_angles()
        
        # 设置geomgroup（控制哪些几何体组可见）
        geomgroup = np.array([1, 0, 1, 0, 0, 0], dtype=np.uint8)
        
        # 创建雷达句柄
        from mujoco_lidar.lidar_wrapper import MjLidarWrapper
        self.lidar_sim = MjLidarWrapper(
            self.mj_model,
            site_name="lidar_site",
            backend="cpu",
            cutoff_dist=100.0,
            args={
                'bodyexclude': -1,
                'geomgroup': geomgroup,
                'max_candidates': 64,
                'ti_init_args': {'device_memory_GB': 4.0}
            }
        )
        
        # 点云发布者
        self.point_cloud_pub = self.simulator.create_publisher(
            PointCloud2, "/point_cloud", 100
        )
        
        # IMU发布者（用于感知）
        self.imu_pub = self.simulator.create_publisher(Imu, "/imu", 10)
    
    def execute(self):
        """执行点云获取和发布"""
        if not self.enabled:
            return
        
        # 更新雷达射线角度（动态扫描）
        self.rays_theta, self.rays_phi = self.livox_generator.sample_ray_angles()
        
        # 执行光线追踪
        self.lidar_sim.trace_rays(self.mj_data, self.rays_theta, self.rays_phi)
        
        # 获取击中点
        hit_points = self.lidar_sim.get_hit_points()
        world_points = hit_points
        
        time_stamp = self.simulator.get_clock().now().to_msg()
        
        # 设置并发布点云信息
        point_cloud_msg = PointCloud2()
        point_cloud_msg.header = Header()
        point_cloud_msg.header.frame_id = 'lidar'
        point_cloud_msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
        ]
        point_cloud_msg.header.stamp = time_stamp
        point_cloud_msg.is_bigendian = False
        point_cloud_msg.point_step = 12
        point_cloud_msg.row_step = point_cloud_msg.point_step * len(world_points)
        point_cloud_msg.height = 1
        point_cloud_msg.width = len(world_points)
        point_cloud_msg.is_dense = True
        point_cloud_msg.data = world_points.astype(np.float32).tobytes()
        self.point_cloud_pub.publish(point_cloud_msg)
        
        # 点云相对于base_link的坐标转换
        t = TransformStamped()
        t.header.stamp = time_stamp
        t.header.frame_id = self.simulator.first_link_name
        t.child_frame_id = 'lidar'
        t.transform.translation.x = self.lidar_site_pos[0]
        t.transform.translation.y = self.lidar_site_pos[1]
        t.transform.translation.z = self.lidar_site_pos[2]
        t.transform.rotation.w = self.lidar_site_quat[0]
        t.transform.rotation.x = self.lidar_site_quat[1]
        t.transform.rotation.y = self.lidar_site_quat[2]
        t.transform.rotation.z = self.lidar_site_quat[3]
        self.simulator.tf_broadcaster.sendTransform(t)
        
        # 单独发布imu数据给感知用
        imu_data_msg = self.simulator.low_state_msg.imu
        imu_data_msg.header.frame_id = "imu_frame"
        imu_data_msg.header.stamp = time_stamp
        if self.simulator.param.get("g_unit", "g") == "g":
            imu_data_msg.linear_acceleration.x /= 9.80665
            imu_data_msg.linear_acceleration.y /= 9.80665
            imu_data_msg.linear_acceleration.z /= 9.80665
        elif self.simulator.param.get("g_unit", "g") == "m/s^2":
            pass
        else:
            self.simulator.get_logger().error(
                f"未知的重力单位: {self.simulator.param.get('g_unit', 'g')}, 请检查参数设置"
            )
            return
        self.imu_pub.publish(imu_data_msg)
        
        # 发布IMU的TF变换
        imu_transform = TransformStamped()
        imu_transform.header.stamp = time_stamp
        imu_transform.header.frame_id = self.simulator.first_link_name
        imu_transform.child_frame_id = "imu_frame"
        imu_transform.transform.translation.x = 0.0
        imu_transform.transform.translation.y = 0.0
        imu_transform.transform.translation.z = 0.0
        imu_transform.transform.rotation.w = 1.0
        imu_transform.transform.rotation.x = 0.0
        imu_transform.transform.rotation.y = 0.0
        imu_transform.transform.rotation.z = 0.0
        self.simulator.tf_broadcaster.sendTransform(imu_transform)