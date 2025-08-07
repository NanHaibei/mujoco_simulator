import launch
import launch_ros
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.actions import ExecuteProcess
from launch.substitutions import PathJoinSubstitution
from ament_index_python import get_package_share_directory
import os
import yaml
from datetime import datetime
from launch_ros.actions import Node
from launch.conditions import IfCondition

def generate_launch_description():

    # 声明launch参数
    use_lidar_arg = DeclareLaunchArgument(
        'use_lidar',
        default_value='true',
        description='Enable or disable lidar simulation'
    )
    use_foxglove_arg = DeclareLaunchArgument(
        'use_foxglove',
        default_value='true',
        description='Enable or disable foxglove bridge'
    )
    record_bag_arg = DeclareLaunchArgument(
        'record_bag',
        default_value='false',
        description='Enable rosbag recording'
    )

    # 获取yaml路径以读取urdf文件
    mujoco_pkg_path = get_package_share_directory('mujoco_simulator_python')
    yaml_path = mujoco_pkg_path + "/config/simulate.yaml"
    if not os.path.exists(yaml_path): raise FileNotFoundError(f"yaml未找到: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as file:
        yaml_config = yaml.safe_load(file)
    # 获取urdf文件路径
    robot_pkg_path = get_package_share_directory('robot_description')
    model_name = yaml_config["mujoco_simulator"]["modelName"]
    if "S1" in model_name:
        model_type = "S1"
    elif "S2" in model_name:
        model_type = "S2"
    elif "G1" in model_name:
        model_type = "G1"
    elif "Pegasus" in model_name:
        model_type = "Pegasus"

    mjcf_path = robot_pkg_path + "/" + model_type + "/mjcf/scene_" + model_name + ".xml"
    model_name = model_name.replace("_float", "") # 删除float字段
    model_name = model_name.replace("scene_", "") # 删除scene字段
    urdf_path = robot_pkg_path + "/" + model_type + "/urdf/" + model_name + ".urdf"

    if not os.path.exists(urdf_path): raise FileNotFoundError(f"URDF文件未找到: {urdf_path}") 

    # 传感器配置文件路径
    sensor_yaml_path = mujoco_pkg_path + "/config/" + model_type + "_sensor_cfg.yaml"
    if not os.path.exists(sensor_yaml_path): raise FileNotFoundError(f"传感器配置文件未找到: {sensor_yaml_path}")

    # 获取foxglove的launch文件路径
    foxglove_pkg_path = get_package_share_directory('foxglove_bridge')
    xml_launch_path = PathJoinSubstitution([
        foxglove_pkg_path, 'launch', 'foxglove_bridge_launch.xml'
    ])

    # 获取当前时间，用于确定bags的保存路径
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bag_folder_path = "bags/" + f"{timestamp}"

    # 在launch文件中使用参数
    nodes = [
        Node(
            package='mujoco_simulator_python',
            executable='mujoco_simulator_python',
            name='mujoco_simulator',
            output='both',
            emulate_tty=True,
            parameters=[
                {
                    'use_lidar': LaunchConfiguration('use_lidar'),  # 使用launch参数
                    'yaml_path': yaml_path,
                    'sensor_yaml_path': sensor_yaml_path,
                    'mjcf_path': mjcf_path,
                },
            ]
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='both',
            parameters=[{'robot_description': open(urdf_path).read()}]
        ),
    ]
    
    # 条件性启动foxglove
    nodes.append(
        IncludeLaunchDescription(
            AnyLaunchDescriptionSource(xml_launch_path),
            launch_arguments={
                "output": "log"
            }.items(),
            condition=IfCondition(LaunchConfiguration('use_foxglove'))  # 条件启动
        )
    )
    
    # 条件性录制rosbag
    nodes.append(
        ExecuteProcess(
            cmd=['ros2', 'bag', 'record', '-o', bag_folder_path, '-a', '-s', 'mcap'],
            output='screen',
            condition=IfCondition(LaunchConfiguration('record_bag'))  # 条件启动
        )
    )

    return LaunchDescription([
        use_lidar_arg,
        use_foxglove_arg,
        record_bag_arg,
        *nodes
    ])
