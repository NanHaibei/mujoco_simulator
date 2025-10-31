# mujoco_simulator

## 1. 介绍

本仓库使用mujoco的pythonAPI进行开发，在保持物理仿真基本功能的同时添加了ROS2通信接口与模型信息输出功能。可以在不改动代码的基础上支持各种MJCF模型

本仓库在下列环境中测试通过
1. Ubuntu 22.04
2. Windows WSL2 Ubuntu 22.04


## 2. 安装教程

安装ROS2，推荐使用鱼香ROS一键安装脚本

```shell
wget http://fishros.com/install -O fishros && . fishros
```

安装foxglove包用于可视化

```shell
sudo apt install ros-humble-foxglove-bridge
```

安装mujoco-python

```shell
pip install mujoco
```

新建ros2工作空间

```shell
cd ~/project # 换成您自己的路径
mkdir -p mujoco_simulator/src
cd mujoco_simulator/src
```

克隆本仓库和相关仓库到ros2工作空间中

```shell
# mujoco仿真库本体
git clone https://github.com/NanHaibei/mujoco_simulator.git
# 实验室机器人模型库
git clone https://gitee.com/coralab/robot_description_coral.git 
# ros2 自定义消息
git clone https://gitee.com/LockedFlysher/common_msgs.git
cd common_msgs
git checkout ros2_version # 切换到ros2分支
cd ..
# 在mujoco中支持激光雷达 
# 该仓库貌似不能放到ros2工作空间中【已修复】
# 如果ros2编译报错，请将 project.toml 的requires修改为  ["setuptools>=64", "wheel"]
git clone https://github.com/TATP-233/MuJoCo-LiDAR.git
cd MuJoCo-LiDAR
python3 -m pip install .
```

开始编译

```shell
cd ..
colcon build
```

如果编译成功就说明安装没有问题

## 3. 使用说明

### 3.1 启动mujoco仿真

仓库的主要超参数保存在`config/simulate.yaml`中，modelName设置了仿真使用的机器人模型

```yaml
mujoco_simulator:
    # 使用的模型名称 目前的可选项如下
    # G1_29dof_float， 
    # G1_29dof，
    # S1_20dof_simp_col
    # S1_20dof_simp_col_float， 
    # S2_12dof_lock_arm_simp_col_float， 
    # S2_12dof_lock_arm_simp_col
    # S2_20dof_simp_col_float
    # S2_20dof_simp_col

    modelName: "G1_29dof" 
    
    # 电机状态与命令topic
    lowStateTopic: "/human_lower_state" 
    jointCommandsTopic: "/human_lower_command" 

    # 加载模型后是否暂停
    unPauseService: "/unpause_mujoco" # 启动服务名称
    initPauseFlag: true # 1是会暂停，0是不会暂停

    # 是否输出模型信息表格
    modelTableFlag: true # 1是会，0是不会

```
由于ros2抽象的执行逻辑，修改任何文件后都需要进行编译才能生效

```shell
colcon build
```

确认参数无误后使用下面的命令启动仿真

```shell
source install/setup.bash
ros2 launch mujoco_simulator_python simulate.launch.py 
```

成功启动后的界面如下图所示

![](https://picgo-nanhaibei.oss-cn-beijing.aliyuncs.com/20250808152024.png)

默认启动mujoco后物理仿真处于暂停状态，可以通过修改配置文件设置不暂停，也可以调用ROS2服务启动仿真

在python中使用服务启动仿真的示例代码如下

```python
from std_srvs.srv import Empty

self.mujoco_unpause_client = self.create_client(
    Empty， 
    "/unpause_mujoco"
)
unpause = Empty.Request()
self.mujoco_unpause_client.call_async(unpause)
```

### 3.2 控制仿真环境中的机器人

启动launch后ros2 topic中会出现`/human_lower_command`、`/human_lower_state`两个话题，前者是命令消息，后者是状态消息

只需要往`/human_lower_command`中发送`MITJointCommands`类型的消息即可控制机器人关节运动

需要注意：命令的长度必须与机器人的关节数量一致，否则仿真程序会认为用户运行了错误的控制器，控制命令不予生效。
此外，命令、状态中电机的顺序分别为MJCF中执行器、传感器的顺序

具体控制过程见[作者的强化学习运动控制库](https://gitee.com/nanhaibei/rl_deploy_python)

### 3.3 MJCF文件的导出

#### 3.3.1 从URDF导出MJCF

理论上本节内容在ubuntu中也能完成，但是由于笔者使用的是WSL，不便测试，所以本节内容均在纯Windows系统中进行。

以实验室的神农机器人为例，假设现在已经拿到从solidworks中导出的urdf文件，首先需要在`robot`标签下添加mesh文件路径，让mujoco能找到mesh文件放在哪里

```xml
<robot
  name="ShenNong">
  <mujoco>
    <compiler
    	meshdir="../meshes/"
    	balanceinertia="true"
    	discardvisual="false" />
  </mujoco>
  <link
    name="base_link">
  <!-- 下面的link和joint省略 -->
```

接下来在[官方仓库](https://github.com/google-deepmind/mujoco/releases)中下载mujoco的Windows版本，版本没有要求，下载最新的即可。下载解压后双击`bin/simulate.exe`打开mujoco界面，然后把urdf文件直接拖到mujoco界面中，成功加载后的界面如下图所示

![](https://picgo-nanhaibei.oss-cn-beijing.aliyuncs.com/20250404155834.png)

如果加载URDF时出现下面的报错，一般是某个mesh面数超过200000，mujoco的compiler不支持，需要手动减少mesh面数后再尝试导入。减少面数的方法见[这篇博文]((https://blog.csdn.net/qq_37389133/article/details/125050981))

```
Error: number of faces should be between 1 and 200000 in STL file '/home/coral-jyz/project/robot-urdf/ShenNong/urdf/../meshes/base_link.STL'; perhaps this is an ASCII file?
```

此时按下空格键启动仿真，可能会发现机器人关节乱动

![](https://picgo-nanhaibei.oss-cn-beijing.aliyuncs.com/20250404160016.png)

这一般是由于机器人的碰撞体积发生干涉，需要简化urdf中的碰撞体积。

下面给出一个修改例子，原始urdf中link标签的内容如下，需要用简单几何体替换mesh作为碰撞体积

```xml
  <link
    name="LL1_link">
    <inertial>
    <!-- ... -->
    </inertial>
    <visual>
      <!-- ... -->
    </visual>
    <!-- 👇这里是碰撞体积标签👇 -->
    <collision> 
      <origin
        xyz="0 0 0"
        rpy="0 0 0" />
      <geometry>
        <mesh
          filename="package://robot_urdf/ShenNong/meshes/LL1_link.STL" />
      </geometry>
    </collision>
    <!-- 👆这里是碰撞体积标签👆 -->
  </link>
```

作如下修改

```xml
  <link
    name="LL1_link">
    <inertial>
    <!-- ... -->
    </inertial>
    <visual>
      <!-- ... -->
    </visual>
    <collision>
        <origin
            xyz="-0.0028 3.5e-05 -0.002062"
            rpy="0 0 0"/>
        <geometry>
            <!-- 使用圆柱替代原来的碰撞体积 -->
            <cylinder radius="0.06" length="0.05"/>
        </geometry>
    </collision>
  </link>
```

碰撞体积的简化方式视任务而定，在一般的行走任务中，可以使用长方体作为base_link和脚掌的碰撞体积，其他所有link的碰撞体积都可以删除

tips:在mujoco中按下数字`0`和`1`分别是打开/隐藏碰撞体积与可视化mesh

#### 3.3.2 MJCF文件的完善

**添加base_link**

当机器人物理仿真正常后，可以点击mujoco界面左上角的`Save xml`导出MJCF文件了，文件会生成在打开mujoco的终端所在的路径下。需要注意，mujoco默认将base_link与第二个link合并，为了保持正确的link树，需要手动创建base_link

以神农机器人为例，导出的MJCF的body部分如下，第一个body是`LL1_link`而不是`base_link`

```xml
 <worldbody>
    <geom type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="0.752941 0.752941 0.752941 1" mesh="base_link_sim"/>
    <geom type="mesh" rgba="0.752941 0.752941 0.752941 1" mesh="base_link_sim"/>
    <geom size="0.0075 0.0075 0.002" pos="0 0 -0.004" quat="0.999998 0 0 -0.002" type="box" contype="0" conaffinity="0" group="1" density="0"/>
    <body name="LL1_link" pos="0.04142 0.119977 -0.266431">
      <inertial pos="-0.0028 3.5e-05 -0.002062" quat="-0.0066869 0.703683 0.0022928 0.710479" mass="1.905" diaginertia="0.00597639 0.00442126 0.00244036"/>
      <joint name="l_leg_hpx" pos="0 0 0" axis="1 0 0" range="-0.2 0.4" actuatorfrcrange="-160 160"/>
      <geom type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="0.752941 0.752941 0.752941 1" mesh="LL1_link"/>
      <geom type="mesh" rgba="0.752941 0.752941 0.752941 1" mesh="LL1_link"/>
      <body name="LL2_link" pos="0 0 -0.0473">
      <!-- ... -->
<worldbody/>
```

需要作如下修改

```xml
<worldbody>
    <body name="base_link" pos="0 0 1.2">
      <inertial pos="0 0 0" quat="1 0 0 0" mass="12.42979" diaginertia="0.61166 0.544877 0.139043" />
      <joint name="float_base_joint" type="free" limited="false" actuatorfrclimited="false"/>
      <geom type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="1 1 1 1"
        mesh="base_link_sim" />
      <geom size="0.0075 0.0075 0.002" pos="0 0 -0.004" quat="0.999998 0 0 -0.002" type="box"
        contype="0" conaffinity="0" group="1" density="0" />
        <!-- 记得在这里添加site，后面要用 -->
      <site name="imu" size="0.01" pos="0 0 0" />
      <body name="LL1_link" pos="0.04142 0.119977 -0.266431">
        <inertial pos="-0.0028 3.5e-05 -0.002062" quat="-0.0066869 0.703683 0.0022928 0.710479"
          mass="1.905" diaginertia="0.00597639 0.00442126 0.00244036" />
        <joint name="l_leg_hpx" pos="0 0 0" axis="1 0 0" range="-0.2 0.4"
          actuatorfrcrange="-160 160" frictionloss="0.2" damping="0.1"/>
        <geom type="mesh" contype="0" conaffinity="0" group="1" density="0"
          rgba="0.752941 0.752941 0.752941 1" mesh="LL1_link" />
        <geom size="0.06 0.025" pos="-0.0028 3.5e-05 -0.002062" type="cylinder"
          rgba="0.752941 0.752941 0.752941 1" />
        <body name="LL2_link" pos="0 0 -0.0473">
        <!-- ... -->
```

其中`base_link`的`pos`指机器人初始生成的位置，可自行给定，`inertial`标签下的`pos`，`quat`，`mass`可直接从URDF中读取，`diaginertia`有三个元素，分别是URDF中惯性矩阵的`ixx`，`iyy`，`izz`

**添加actuator与sensor**

在MJCF中添加`<actuator>`与`<sensor>`标签后，才能发送电机命令与读取传感器状态

以神农机器人为例，在`<mujoco>`标签下添加下面的内容

```xml
  <!-- 力控执行器 -->
  <actuator>
    <motor name="M_l_leg_hpx" joint="l_leg_hpx" ctrlrange="-160 160" />
    <motor name="M_l_leg_hpz" joint="l_leg_hpz" ctrlrange="-120 120" />
    <motor name="M_l_leg_hpy" joint="l_leg_hpy" ctrlrange="-160 160" />
    <!-- ... -->
  </actuator>

  <sensor>
    <!-- 电机位置传感器 -->
    <jointpos name="l_leg_hpx_pos" joint="l_leg_hpx" />
    <jointpos name="l_leg_hpz_pos" joint="l_leg_hpz" />
    <jointpos name="l_leg_hpy_pos" joint="l_leg_hpy" />
    <!-- ... -->

    <!-- 电机速度传感器 -->
    <jointvel name="l_leg_hpx_vel" joint="l_leg_hpx" />
    <jointvel name="l_leg_hpz_vel" joint="l_leg_hpz" />
    <jointvel name="l_leg_hpy_vel" joint="l_leg_hpy" />
    <!-- ... -->

    <!-- 电机力矩传钢琴 -->
    <jointactuatorfrc name="l_leg_hpx_torque" joint="l_leg_hpx" />
    <jointactuatorfrc name="l_leg_hpz_torque" joint="l_leg_hpz" />
    <jointactuatorfrc name="l_leg_hpy_torque" joint="l_leg_hpy" />
    <!-- ... -->

    <!-- 建立一个返回四元数的传感器 -->
    <framequat name="imu_quat" objtype="site" objname="imu" />
    <!-- 建立一个返回角速度的传感器 -->
    <gyro name="imu_gyro" site="imu" />
    <!-- 建立一个返回线加速度的传感器 -->
    <accelerometer name="imu_acc" site="imu" />
    <!-- 返回3D真实位置 -->
    <framepos name="frame_pos" objtype="site" objname="imu" />
    <!-- 返回3D真实线速度 -->
    <framelinvel name="frame_vel" objtype="site" objname="imu" />
  </sensor>
```

添加完成后，用mujoco打开该MJCF，在`Option`中开启`Sensor`，在右侧边栏中打开`Control`，如果看到如下图所示的画面，说明标签添加成功

![](https://picgo-nanhaibei.oss-cn-beijing.aliyuncs.com/20250405104500.png)

**给关节添加阻尼摩擦力**

默认的关节阻尼和摩擦都是0，可能会导致控制发散，有两种方法给关节添加阻尼和摩擦：

第一种是直接在`<joint>`标签下添加摩擦`frictionloss`和阻尼`damping`

```xml
<joint name="l_leg_hpx" pos="0 0 0" axis="1 0 0" range="-0.2 0.4"
       actuatorfrcrange="-160 160" frictionloss="0.2" damping="0.1"/>
```

第二种是先在mjcf文件开头声明`class`属性

```xml
<default>
  <default class="motor_a">
    <joint damping="0.05" armature="0.01" frictionloss="0.2"/>
  </default>
</default>
```

然后对每一个joint设置`class`属性

```xml
<joint name="left_ankle_pitch_joint" pos="0 0 0" axis="0 1 0"
        range="-0.87267 0.5236" actuatorfrcrange="-50 50"  class="ankle_motor"/>
```

**添加天空和地板**

上诉的MJCF只包含机器人本身，启动物理仿真后机器人会坠入无尽虚空，为了解决该问题，需要在MJCF中添加地板和天空

可以新建一个xml文件，名称必须为为`scene_<机器人mjcf名称>.xml`，内容如下

```xml
<mujoco model="g1_29dof scene">
    <!-- 此处引用机器人mjcf文件 -->
  <include file="G1_29dof.xml"/> 

  <statistic center="0 0 0.5" extent="2.0"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="-130" elevation="-20"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>

  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="50 50 0.05" type="plane" material="groundplane"/>
  </worldbody>
</mujoco>
```

完成了上述工作后，我们就得到了完善的MJCF文件，可以开始仿真了

### 3.4 激光雷达仿真

目前新版本的Mujoco-LiDAR库支持扫描mesh，即雷达可以探测到机器人本身，这原本是好事，但是探测mesh将会带来极大的时间消耗。

建议将该库回退到2025年4月16日的commit `22c082640ed0f8c244c23ec284691300389be836`，以加速雷达仿真

![](https://picgo-nanhaibei.oss-cn-beijing.aliyuncs.com/20250808154706.png)

只要在机器人的mjcf中添加`lidar_site`，程序就会自动执行激光雷达仿真与点云信息发布，下面是一个例子

以G1为例，在名为`pelvis`的body下添加`lidar_site`

```xml
<worldbody>
    <body name="pelvis" pos="0 0 0.793">
      <site name="imu" size="0.01" pos="0.0 0.0 0.0" />
      <!-- site 的type、size可以自行调整，pos、euler确定了雷达原点的位姿，需要精确设置 -->
      <site name="lidar_site" size="0.03" type='capsule' pos="0.0 0.0 0.416" euler="3.14 0 0"/> 
      <inertial pos="0 0 -0.07605" quat="1 0 -0.000399148 0" mass="3.813"
        diaginertia="0.010549 0.0093089 0.0079184" />
      <joint name="floating_base_joint" type="free" limited="false" actuatorfrclimited="false" />
      <geom type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="0.2 0.2 0.2 1"
        mesh="pelvis" />
      <geom type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="0.7 0.7 0.7 1"
        mesh="pelvis_contour_link" />
      <geom type="mesh" rgba="0.7 0.7 0.7 1" mesh="pelvis_contour_link" />
      <body name="left_hip_pitch_link" pos="0 0.064452 -0.1027">
```

此时启动仿真后，tos2 topic中将会出现点云信息`/point_clound`

```shell
$ ros2 topic list
/clicked_point
/diagnostics
/human_lower_command
/human_lower_state
/initialpose
/joint_states
/joint_states_panel
/move_base_simple/goal
/parameter_events
/point_cloud
/rl_cmd_vel
/robot_description
/rosout
/sim_real_vel
/tf
/tf_static
```

### 3.5 foxglove可视化

下载[foxglove](https://foxglove.dev/download)

启动`simulate.launch.py`后直接打开foxglove，选择`Open connection`，然后点击`Open`，此时就可以在foxglove中看到所有ros2消息了，foxglove中的面板需要自行设置

![](https://picgo-nanhaibei.oss-cn-beijing.aliyuncs.com/20250808162523.png)

![](https://picgo-nanhaibei.oss-cn-beijing.aliyuncs.com/20250808162556.png)


