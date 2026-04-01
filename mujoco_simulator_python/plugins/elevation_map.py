from __future__ import annotations
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Float32MultiArray
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..mujoco_simulator_python import mujoco_simulator

from .base import BasePlugin


class RayCaster:
    """射线传感器类
    
    负责通过射线投射获取高程数据
    """
    
    def __init__(
            self, 
            mj_data,
            mj_model,
            attach_link_id: int,
            pos_offset: tuple[float, float, float], 
            yaw_offset: float, 
            resolution: float, 
            size: tuple[float, float], 
    ):
        """射线传感器

        Args:
            mj_data: mujoco数据句柄
            mj_model: mujoco模型句柄
            attach_link_id (int): 传感器依附的link在mj_model中的id,高程图基于该基座坐标系获取
            pos_offset (tuple[float, float, float]): 高程图采样点相对于机器人基座的偏移位置
            yaw_offset (float): 高程图采样点相对于机器人基座的偏航角
            resolution (float): 高程图的分辨率
            size (tuple[float, float]): 高程图的大小
        """
        
        self.mj_data = mj_data
        self.mj_model = mj_model
        self.attach_link_id = attach_link_id
        self.offset_pos = pos_offset
        self.offset_rot = yaw_offset
        self.resolution = resolution
        self.size = size

        # 计算长宽点数
        self.num_x_points = round(self.size[0] / self.resolution) + 1
        self.num_y_points = round(self.size[1] / self.resolution) + 1
        # 初始化高程图数组，-1表示无效值
        self._data = np.zeros((self.num_x_points * self.num_y_points, 3), dtype=np.float32) - 1  
        # 生成机器人坐标系下，x和y轴的采样点
        self.x_sample_points = np.linspace(-self.size[0]/2, self.size[0]/2, self.num_x_points)
        self.y_sample_points = np.linspace(-self.size[1]/2, self.size[1]/2, self.num_y_points)
        # 设置射线探测的碰撞组
        self.geomgroup = (False, False, True, False, False, False)
    
    def update_elevation_data(self):
        """更新高程数据

        Returns:
            np.ndarray: 高程数据数组，形状为 (num_points, 3)，每行为 [x, y, height]
        """
        # 读取base link的pos和quat
        robot_pos = self.mj_data.xpos[self.attach_link_id]  # 位置 [x, y, z]
        robot_rot = self.mj_data.xquat[self.attach_link_id]  # 四元数 [w, x, y, z]

        # 将offset_pos应用到robot_pos上
        # 需要将offset_pos从body坐标系转换到世界坐标系
        r = R.from_quat([robot_rot[1], robot_rot[2], robot_rot[3], robot_rot[0]])
        offset_world = r.apply(self.offset_pos)
        adjusted_robot_pos = robot_pos + offset_world

        # 计算世界坐标系下采样矩阵的坐标
        world_coords, _ = self._transform_points_to_world_yaw_only(
            self.x_sample_points, self.y_sample_points, adjusted_robot_pos, robot_rot
        )

        # 填充_data的x和y坐标
        self._data[:,:2] = world_coords[:, :2]

        # 获取射线发射的起始高度（lidar_site的世界坐标Z值）
        ray_start_height = world_coords[0, 2]  # 所有采样点的Z坐标相同，都是robot_pos[2]
        
        # 对每个采样点进行射线投射
        for i in range(self.num_x_points * self.num_y_points):
            hit_dist = mujoco.mj_ray(
                self.mj_model, 
                self.mj_data, 
                [world_coords[i, 0], world_coords[i, 1], ray_start_height], # 从lidar_site高度垂直向下发射射线
                [0, 0, -1], # 方向为垂直向下
                self.geomgroup, # 碰撞组
                1, # 包含静态物体
                -1,  # 包含所有body
                np.array([-1], dtype=np.int32) # 占位符
            )
            # 计算地形高度
            self._data[i,2] = hit_dist
                
        return self._data

    def _transform_points_to_world_yaw_only(self, x_sample_points, y_sample_points, robot_pos, robot_quat):
        """
        将机器人坐标系下的采样点转换到世界坐标系，仅考虑机器人的Yaw角（偏航角）。

        参数:
            x_sample_points: 一维数组，x方向的采样点（机器人坐标系下）。
            y_sample_points: 一维数组，y方向的采样点（机器人坐标系下）。
            robot_pos: 机器人位置 [x, y, z]（世界坐标系下）。
            robot_quat: 机器人的四元数 [w, x, y, z]。

        返回:
            world_points: 世界坐标系下的点云，形状为 (num_points, 3)。
            shape: 采样网格的形状
        """
        # 1. 创建网格采样点
        X, Y = np.meshgrid(x_sample_points, y_sample_points, indexing='ij')
        # 将二维网格点转换为三维点，假设采样点在机器人坐标系中 z=0
        points_robot = np.column_stack((X.ravel(), Y.ravel(), np.zeros(X.size)))

        r = R.from_quat([robot_quat[1], robot_quat[2], robot_quat[3], robot_quat[0]])
        yaw = r.as_euler('xyz', degrees=False)[2]

        # 2. 构建仅考虑Yaw的简化齐次变换矩阵
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        
        # 绕Z轴的旋转矩阵 (2x2)
        R_z = np.array([[cos_yaw, -sin_yaw],
                        [sin_yaw,  cos_yaw]])
    
        # 构建4x4齐次变换矩阵
        T = np.eye(4)
        T[:2, :2] = R_z  # 将2D旋转矩阵放入左上角的2x2部分
        T[:2, 3] = robot_pos[:2]  # 平移部分为机器人的x, y坐标
        # Z轴平移设为0，因为我们将在后面单独处理高度

        # 3. 将点转换为齐次坐标（添加一列1）
        points_homo = np.hstack((points_robot, np.ones((points_robot.shape[0], 1))))

        # 4. 应用变换：世界坐标 = T × 机器人坐标
        world_points_homo = (T @ points_homo.T).T

        # 5. 转换回三维坐标（去掉齐次坐标的最后一维）
        world_points = world_points_homo[:, :3]

        # 6. 设置Z坐标：所有点的高度与机器人底座高度相同（或加上采样点的原始Z坐标，这里原始Z=0）
        world_points[:, 2] = robot_pos[2]  # 将机器人的高度赋予所有点

        return world_points, X.shape


class ElevationMap(BasePlugin):
    """高程图插件
    
    负责生成和发布高程图信息。
    """
    
    def __init__(self, plugin_config: dict, simulator: mujoco_simulator):
        """初始化高程图插件"""
        super().__init__(plugin_config, simulator)
        # 从插件配置中读取高程图参数
        self.map_topic = plugin_config.get("topic", "/elevation/map_array")
        self.map_size = plugin_config.get("size", [1.35, 0.95])
        self.map_resolution = plugin_config.get("resolution", 0.05)
        attach_link_name = plugin_config.get("attach_link_name", "lidar_site")
        
        self.simulator.get_logger().info("Height Scan (高程图) 功能已启用。")
        
        # 获取attach_link的body ID和位置偏移
        attach_site_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, attach_link_name)
        if attach_site_id >= 0:
            # 从site获取对应的body id和位置
            self.elevation_attach_body_id = self.mj_model.site_bodyid[attach_site_id]
            attach_body_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, self.elevation_attach_body_id)
            site_pos = self.mj_model.site_pos[attach_site_id].copy()  # [x, y, z]
            self.elevation_pos_offset = (float(site_pos[0]), float(site_pos[1]), float(site_pos[2]))
            self.simulator.get_logger().info(
                f"高程图将附着到site: {attach_link_name}, body: {attach_body_name}, 位置偏移: {self.elevation_pos_offset}"
            )
        else:
            # 如果没有找到site，报错
            self.simulator.get_logger().error(f"未找到site '{attach_link_name}'，请检查MJCF文件和YAML配置！")
            raise ValueError(f"Site '{attach_link_name}' not found in the model!")
        
        # 实例化RayCaster传感器（直接使用内嵌类）
        self.raycaster = RayCaster(
            self.mj_data,
            self.mj_model,
            attach_link_id=self.elevation_attach_body_id,
            pos_offset=self.elevation_pos_offset,
            yaw_offset=0.0,
            resolution=self.map_resolution,
            size=(self.map_size[0], self.map_size[1]),
        )
        
        # 初始化网格尺寸
        self.grid_size_x = round(self.map_size[0] / self.map_resolution) + 1
        self.grid_size_y = round(self.map_size[1] / self.map_resolution) + 1
        self.elevation_sample_point = np.zeros((self.grid_size_x * self.grid_size_y, 3), dtype=np.float32)

        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        
        # 声明elevation map发布者
        self.elevation_pub = self.simulator.create_publisher(Float32MultiArray, self.map_topic, qos)
    
    def execute(self):
        """执行高程图生成和发布"""
        # 获取采样点
        self.elevation_sample_point = self.raycaster.update_elevation_data()
        
        # 添加噪声
        noise = np.random.uniform(
            -self.plugin_config.get("noise_elevation_map", 0.0),
            self.plugin_config.get("noise_elevation_map", 0.0),
            size=len(self.elevation_sample_point)
        )
        self.elevation_sample_point[:, 2] += noise
        
        # 生成并发布高程图
        msg = Float32MultiArray()
        msg.data = self.elevation_sample_point[:, 2].astype(np.float32).tolist()
        self.elevation_pub.publish(msg)
    
    def visualize(self, viewer):
        """高程图可视化（在渲染时调用）"""
        if self.elevation_attach_body_id < 0:
            return
        
        sample_points = self.elevation_sample_point
        pos_offset = self.elevation_pos_offset
        
        viewer.user_scn.ngeom = self.mj_model.ngeom
        
        # 获取当前lidar_site的世界坐标高度
        current_robot_pos = self.mj_data.xpos[self.elevation_attach_body_id]
        current_robot_rot = self.mj_data.xquat[self.elevation_attach_body_id]
        r = R.from_quat([current_robot_rot[1], current_robot_rot[2], current_robot_rot[3], current_robot_rot[0]])
        offset_world = r.apply(pos_offset)
        lidar_height = current_robot_pos[2] + offset_world[2]
        
        # 初始化新添加的几何体
        for i in range(len(sample_points)):
            viewer.user_scn.ngeom += 1
            terrain_height = lidar_height - sample_points[i, 2]
            sphere_pos = [
                sample_points[i, 0],
                sample_points[i, 1],
                terrain_height
            ]
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[viewer.user_scn.ngeom - 1],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.02, 0, 0],
                pos=sphere_pos,
                mat=np.eye(3).flatten(),
                rgba=[1.0, 0.0, 0.0, 1.0]
            )