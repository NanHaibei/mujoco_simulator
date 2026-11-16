import os
import sys
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from pynput import keyboard

def create_demo_scene(scene="primitive"):
    """创建一个用于测试的mujoco场景，包含所有支持的几何体类型"""
    if scene == "floor":
        xml = """
        <mujoco>
          <worldbody>
            <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
            <!-- 激光雷达位置 -->
            <body name="lidar_base" pos="0 0 1">
              <site name="lidar_site" size="0.005" rgba="1 0 0 1"/>
            </body>
            <!-- 平面 -->
            <geom name="ground" type="plane" size="10 10 0.1" pos="0 0 0" rgba="0.9 0.9 0.9 1"/>
          </worldbody>
        </mujoco>
        """

    elif scene == "primitive":
        xml = """
        <mujoco>
            <worldbody>
                <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
                
                <!-- 激光雷达位置 -->
                <body name="lidar_base" pos="0 0 1">
                    <site name="lidar_site" size="0.005" rgba="1 0 0 1"/>
                </body>
                
                <!-- 平面 -->
                <geom name="ground" type="plane" size="10 10 0.1" pos="0 0 0" rgba="0.9 0.9 0.9 1"/>
                <geom name="plane1" type="plane" size="5 5 0.1" pos="6 0 5" euler="0 -60 0" rgba="0.9 0.9 0.9 1"/>
                <geom name="plane2" type="plane" size="5 3 0.1" pos="-1 -4 3" euler="-90 0 0" rgba="0.9 0.9 0.9 1"/>
                
                <!-- 盒子 -->
                <geom name="box1" type="box" size="0.5 0.5 0.5" pos="2 0 0.5" euler="45 -45 0" rgba="1 0 0 1"/>
                <geom name="box2" type="box" size="0.3 0.8 0.2" pos="-2 -1 0.2" rgba="1 0 0 0.7"/>
                <geom name="box3" type="box" size="0.2 0.2 0.3" pos="2.4 -2 0.3" rgba="1 0 0 0.7"/>
                <geom name="box4" type="box" size="0.2 0.2 0.4" pos="0 -2.2 0.4" rgba="1 0 0 0.7"/>
                
                <!-- 球体 -->
                <geom name="sphere1" type="sphere" size="0.5" pos="0 2 0.5" rgba="0 1 0 1"/>
                <geom name="sphere2" type="sphere" size="0.3" pos="-1 2 0.3" rgba="0 1 0 0.7"/>
                
                <!-- 圆柱体 -->
                <geom name="cylinder1" type="cylinder" size="0.4 0.6" pos="0 -2 0.4" euler="0 90 0" rgba="0 0 1 1"/>
                <geom name="cylinder2" type="cylinder" size="0.2 0.3" pos="2 -2 0.3" rgba="0 0 1 0.7"/>
                
                <!-- 椭球体 -->
                <geom name="ellipsoid1" type="ellipsoid" size="0.4 0.3 0.5" pos="3 2 0.5" rgba="1 1 0 1"/>
                <geom name="ellipsoid2" type="ellipsoid" size="0.2 0.4 0.3" pos="3 -1 0.3" rgba="1 1 0 0.7"/>
                
                <!-- 胶囊体 -->
                <geom name="capsule1" type="capsule" size="0.3 0.5" pos="-3 1 0.8" euler="45 0 0" rgba="1 0 1 1"/>
                <geom name="capsule2" type="capsule" size="0.2 0.4" pos="-3 -2 0.6" euler="45 0 45" rgba="1 0 1 0.7"/>
                
                <!-- 角落放置一组排列的几何体 -->
                <body pos="-3 3 1" quat="0.707 0.707 0 0">
                    <geom name="corner_box" type="box" size="0.2 0.2 0.2" pos="0 0 0.2" quat="0.1 0.2 0.3 0.4" rgba="0.5 0.5 0.5 1"/>
                    <geom name="corner_sphere" type="sphere" size="0.2" pos="0.5 0 0.2" quat="0.1 0.4 0.2 0.3" rgba="0.7 0.7 0.7 1"/>
                    <geom name="corner_cylinder" type="cylinder" size="0.2 0.2" pos="0 0.5 0.2" quat="0.9 0.4 0.2 0.3" rgba="0.6 0.6 0.6 1"/>
                    <geom name="corner_capsule" type="capsule" size="0.1 0.3" pos="0.5 0.5 0.3" rgba="0.8 0.8 0.8 1" euler="0 1.57 1.1"/>
                </body>
            </worldbody>
        </mujoco>
        """

    elif scene == "mesh_obj":
        xml = """
        <mujoco>
            <default>
                <default class="mesh_geom">
                    <geom group="0" type="mesh"/>
                </default>
            </default>

            <asset>
                <mesh name="eight" file="../models/eight.obj"/>
            </asset>

            <worldbody>
                <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>

                <!-- mesh -->
                <geom mesh="eight" pos="0 0 1" euler="0 0 0" rgba="0 1 0 1" class="mesh_geom"/>
                
                <!-- 激光雷达位置 -->
                <body name="lidar_base" pos="0 0 1">
                    <site name="lidar_site" size="0.005" rgba="1 0 0 1"/>
                </body>
                
                <!-- 平面 -->
                <geom name="ground" type="plane" size="10 10 0.1" pos="0 0 0" rgba="0.9 0.9 0.9 1"/>

            </worldbody>
        </mujoco>
        """

    elif scene == "mesh_scene":
        xml = """
        <mujoco>
            <default>
                <default class="mesh_geom">
                    <geom group="0" type="mesh"/>
                </default>
            </default>

            <asset>
                <mesh name="scene" file="../models/scene.obj"/>
            </asset>

            <worldbody>
                <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>

                <!-- mesh -->
                <geom mesh="scene" pos="0 0 1" euler="0 0 0" rgba="0 1 0 1" class="mesh_geom"/>

                <!-- 激光雷达位置 -->
                <body name="lidar_base" pos="0 0 1">
                    <site name="lidar_site" size="0.005" rgba="1 0 0 1"/>
                </body>
                
                <!-- 平面 -->
                <geom name="ground" type="plane" size="10 10 0.1" pos="0 0 0" rgba="0.9 0.9 0.9 1"/>

            </worldbody>
        </mujoco>
        """

    wk_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"wk_dir = {wk_dir}")
    os.chdir(wk_dir)
    print("current working dir: ", os.getcwd())

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    # mujoco.mj_saveLastXML("test.xml", model)
    return model, data

class KeyboardListener:
    def __init__(self, lidar_position, lidar_orientation):
        # 存储激光雷达的位置和方向
        self.lidar_position = lidar_position
        self.lidar_orientation = lidar_orientation
        
        # 存储当前的欧拉角（俯仰、偏航）
        self.euler_angles = Rotation.from_quat(lidar_orientation).as_euler('xyz')
        
        # 移动和旋转的速度
        self.move_speed = 1.  # 平移速度 (米/秒)
        self.rotate_speed = 1.  # 旋转速度 (弧度/秒)
        self.height_speed = 0.5  # 高度调整速度 (米/秒)
        
        # 当前按下的键
        self.pressed_keys = set()

        self.running = True
        # 启动键盘监听器
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()
        
        print("键盘控制已启动:")
        print("  选中python终端窗口，按下以下键盘按键控制")
        print("  WASD: 控制水平移动")
        print("  Q/E: 控制高度上升/下降")
        print("  方向键上/下: 控制俯仰角")
        print("  方向键左/右: 控制偏航角")
        print("  ESC: 退出程序")

        # 设置在终端隐藏键盘输入
        os.system('stty -echo')
    
    def __del__(self):
        # 恢复终端显示键盘输入
        os.system('stty echo')

    def on_press(self, key):
        """处理按键按下事件"""
        try:
            # 将按键添加到已按下的键集合中
            if hasattr(key, 'char'):
                self.pressed_keys.add(key.char.lower())
            else:
                self.pressed_keys.add(key)
            
            # 如果按下ESC键，则停止监听
            if key == keyboard.Key.esc:
                print('ESC 键按下，退出程序')
                self.running = False
                return False
        except AttributeError:
            # 忽略特殊键的AttributeError
            pass

    def on_release(self, key):
        """处理按键释放事件"""
        try:
            # 从已按下的键集合中移除释放的键
            if hasattr(key, 'char'):
                self.pressed_keys.discard(key.char.lower())
            else:
                self.pressed_keys.discard(key)
        except AttributeError:
            # 忽略特殊键的AttributeError
            pass

    def update_lidar_pose(self, dt):
        """根据当前按下的键更新激光雷达的位置和姿态"""
        # 获取当前的欧拉角
        pitch = self.euler_angles[1]
        yaw = self.euler_angles[2]
        
        # 计算前进方向（基于当前的yaw角）
        forward_dir = np.array([np.cos(yaw), np.sin(yaw), 0])
        right_dir = np.array([np.sin(yaw), -np.cos(yaw), 0])
        
        # 处理平移（WASD）
        move_delta = np.zeros(3)
        
        if 'w' in self.pressed_keys:
            move_delta += forward_dir * self.move_speed * dt
        if 's' in self.pressed_keys:
            move_delta -= forward_dir * self.move_speed * dt
        if 'd' in self.pressed_keys:
            move_delta += right_dir * self.move_speed * dt
        if 'a' in self.pressed_keys:
            move_delta -= right_dir * self.move_speed * dt
            
        # 处理高度调整（QE）
        if 'q' in self.pressed_keys:
            move_delta[2] += self.height_speed * dt
        if 'e' in self.pressed_keys:
            move_delta[2] -= self.height_speed * dt
            
        # 应用平移
        self.lidar_position += move_delta
        
        # 确保激光雷达不会低于地面
        self.lidar_position[2] = max(0.1, self.lidar_position[2])
        
        # 处理旋转（方向键）
        if keyboard.Key.up in self.pressed_keys:
            pitch += self.rotate_speed * dt
        if keyboard.Key.down in self.pressed_keys:
            pitch -= self.rotate_speed * dt
        if keyboard.Key.left in self.pressed_keys:
            yaw += self.rotate_speed * dt
        if keyboard.Key.right in self.pressed_keys:
            yaw -= self.rotate_speed * dt
            
        # 限制俯仰角范围
        pitch = np.clip(pitch, -np.pi/2 + 0.1, np.pi/2 - 0.1)
        
        # 更新欧拉角
        self.euler_angles[1] = pitch
        self.euler_angles[2] = yaw
        
        # 将欧拉角转换为四元数
        self.lidar_orientation = Rotation.from_euler('xyz', self.euler_angles).as_quat()
        
        # 返回更新后的位置和姿态
        return self.lidar_position.copy(), self.lidar_orientation.copy()
