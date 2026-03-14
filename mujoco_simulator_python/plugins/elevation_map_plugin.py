import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Float32MultiArray

from .base_plugin import BasePlugin


class ElevationMapPlugin(BasePlugin):
    """高程图插件
    
    负责生成和发布高程图信息。
    """
    
    def init(self):
        """初始化高程图插件"""
        # 从主配置中读取高程图参数
        elevation_config = self.simulator.param.get("elevation_map", {})
        
        if not elevation_config.get("enabled", False):
            self.enabled = False
            return
        
        self.simulator.get_logger().info("Height Scan (高程图) 功能已启用。")
        
        # 读取具体参数
        self.map_topic = elevation_config.get("topic", "/elevation/map_array")
        self.map_size = elevation_config.get("size", [1.35, 0.95])
        self.map_resolution = elevation_config.get("resolution", 0.05)
        self.elevation_map_debug = elevation_config.get("debug_info", True)
        attach_link_name = elevation_config.get("attach_link_name", "lidar_site")
        
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
        
        # 实例化RayCaster传感器
        from ..mujoco_RayCaster import RayCaster
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
        
        # 声明elevation map发布者
        self.elevation_pub = self.simulator.create_publisher(Float32MultiArray, self.map_topic, 1)
    
    def execute(self):
        """执行高程图生成和发布"""
        if not self.enabled:
            return
        
        # 获取采样点
        self.elevation_sample_point = self.raycaster.update_elevation_data()
        
        # 添加噪声
        noise = np.random.uniform(
            -self.simulator.param.get("noise_elevation_map", 0.0),
            self.simulator.param.get("noise_elevation_map", 0.0),
            size=len(self.elevation_sample_point)
        )
        self.elevation_sample_point[:, 2] += noise
        
        # 生成并发布高程图
        msg = Float32MultiArray()
        msg.data = self.elevation_sample_point[:, 2].astype(np.float32).tolist()
        self.elevation_pub.publish(msg)
    
    def get_sample_points(self):
        """获取当前采样点（用于可视化）"""
        return self.elevation_sample_point
    
    def get_attach_body_id(self):
        """获取附着body ID（用于可视化）"""
        return getattr(self, 'elevation_attach_body_id', -1)
    
    def get_pos_offset(self):
        """获取位置偏移（用于可视化）"""
        return getattr(self, 'elevation_pos_offset', (0, 0, 0))
    
    def visualize(self, viewer):
        """高程图可视化（在渲染时调用）"""
        if not self.enabled:
            return
        
        # 获取采样点和偏移信息
        sample_points = self.get_sample_points()
        attach_body_id = self.get_attach_body_id()
        pos_offset = self.get_pos_offset()
        
        if attach_body_id < 0:
            return
        
        viewer.user_scn.ngeom = self.mj_model.ngeom
        
        # 获取当前lidar_site的世界坐标高度
        current_robot_pos = self.mj_data.xpos[attach_body_id]
        current_robot_rot = self.mj_data.xquat[attach_body_id]
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
