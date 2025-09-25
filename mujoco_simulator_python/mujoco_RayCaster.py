import numpy as np
from scipy.spatial.transform import Rotation as R
import mujoco

class RayCaster:
    def __init__(
            self, 
            mj_data,
            mj_model,
            offset_pos: tuple[float, float, float], 
            offset_rot: tuple[float, float, float, float], 
            resolution: float, 
            size: tuple[float, float], 
            debug_vis: bool
    ):
        self.mj_data = mj_data
        self.mj_model = mj_model
        self.offset_pos = offset_pos
        self.offset_rot = offset_rot
        self.resolution = resolution
        self.size = size
        self.debug_vis = debug_vis

        # 计算长宽点数
        self.num_x_points = round(self.size[0] / self.resolution) + 1
        self.num_y_points = round(self.size[1] / self.resolution) + 1

        self._data = np.zeros((self.num_x_points * self.num_y_points, 3), dtype=np.float32) - 1  # 初始化为-1，表示无效值

    @property
    def data(self):
        # update sensors if needed
        # self._update_elevation_data()
        # return the data
        return self._data
    
    def update_elevation_data(self, robot_pos, robot_rot):

        # 生成机器人坐标系下，x和y轴的采样点
        x_sample_points = np.linspace(-self.size[0]/2, self.size[0]/2, self.num_x_points)
        y_sample_points = np.linspace(-self.size[1]/2, self.size[1]/2, self.num_y_points)

        # print(x_sample_points, y_sample_points)

        # 计算世界坐标系下采样矩阵的坐标
        world_coords, original_shape = self._transform_points_to_world_yaw_only(
            x_sample_points, y_sample_points, robot_pos, robot_rot
        )
        for i in range(self.num_x_points * self.num_y_points):
            # px_world = 
            # py_world = world_coords[i, 1]

            # 固定机器人正上方离地3m，发射一条垂直向下的射线
            # ray_start = np.array([px_world, py_world, 3.0], dtype=np.float64).reshape(3, 1)
            # ray_dir = np.array([0, 0, -1.0], dtype=np.float64).reshape(3, 1)

            # 屏蔽掉机器人
            # G1的gemo全部设置成group(1)，group(0)是默认,group(2)是地形相关的
            # 对应顺序geomgroup = (group(0), group(1), group(2), group(3), group(4), group(5))
            geomgroup = (False, False, True, False, False, False)
            hit_dist = mujoco.mj_ray(
                self.mj_model, 
                self.mj_data, 
                [world_coords[i, 0], world_coords[i, 1], 3.0],
                [0, 0, -1],
                geomgroup, 
                1, 
                -1, 
                np.array([-1], dtype=np.int32)
            )
            

            if hit_dist > 0:
                height = 3 - hit_dist
                self._data[i,0] = world_coords[i, 0]
                self._data[i,1] = world_coords[i, 1]
                self._data[i,2] = height
                

        return self._data



    def _transform_points_to_world_yaw_only(self, x_sample_points, y_sample_points, robot_pos, robot_quat):
        """
        将机器人坐标系下的采样点转换到世界坐标系，仅考虑机器人的Yaw角（偏航角）。

        参数:
            x_sample_points: 一维数组，x方向的采样点（机器人坐标系下）。
            y_sample_points: 一维数组，y方向的采样点（机器人坐标系下）。
            robot_pos: 机器人位置 [x, y, z]（世界坐标系下）。
            robot_yaw: 机器人的偏航角（yaw，弧度制），绕Z轴旋转的角度。

        返回:
            world_points: 世界坐标系下的点云，形状为 (num_points, 3)。
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

        # print(world_points)
        # print(world_points.shape)

        return world_points, X.shape

    