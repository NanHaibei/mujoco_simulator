from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python import get_package_share_directory
import os
from datetime import datetime

def find_matching_file(base_path, pattern, file_extension):
    """在robot_description目录中搜索匹配的文件"""
    import glob
    
    # 搜索所有子目录中的文件
    search_pattern = os.path.join(base_path, "**", f"*{pattern}*.{file_extension}")
    matches = glob.glob(search_pattern, recursive=True)
    
    # 优先选择scene_开头的文件
    scene_matches = [m for m in matches if "scene_" in os.path.basename(m)]
    if scene_matches:
        # 进一步精确匹配：完全匹配的优先
        exact_matches = [m for m in scene_matches if f"scene_{pattern}.{file_extension}" == os.path.basename(m)]
        if exact_matches:
            return exact_matches[0]
        return scene_matches[0]
    
    # 如果没有scene_前缀的，选择普通匹配
    if matches:
        # 精确匹配优先
        exact_matches = [m for m in matches if f"{pattern}.{file_extension}" == os.path.basename(m)]
        if exact_matches:
            return exact_matches[0]
        return matches[0]
    
    return None

def scan_available_models(robot_pkg_path):
    """扫描robot_description目录，自动发现可用的机器人模型"""
    import glob
    
    available_models = []
    
    # 搜索所有 scene_*.xml 文件
    mjcf_pattern = os.path.join(robot_pkg_path, "**", "mjcf", "scene_*.xml")
    mjcf_files = glob.glob(mjcf_pattern, recursive=True)
    
    for mjcf_file in mjcf_files:
        # 从文件名提取模型名称: scene_ModelName.xml -> ModelName
        basename = os.path.basename(mjcf_file)
        if basename.startswith("scene_") and basename.endswith(".xml"):
            model_name = basename[6:-4]  # 去掉 "scene_" 前缀和 ".xml" 后缀
            
            # 检查是否存在对应的 urdf 文件（清理后的名称）
            clean_model_name = model_name.replace("_float", "").replace("_bind", "")
            urdf_pattern = os.path.join(robot_pkg_path, "**", "urdf", f"*{clean_model_name}*.urdf")
            urdf_files = glob.glob(urdf_pattern, recursive=True)
            
            # 只有同时存在 mjcf 和 urdf 文件的模型才添加到列表
            if urdf_files:
                available_models.append(model_name)
    
    # 排序以便更好地显示
    available_models.sort()
    
    return available_models

def interactive_select_model(available_models):
    """交互式选择机器人模型"""
    print("\n" + "="*60)
    print("      MuJoCo 机器人仿真启动器")
    print("="*60)
    print("\n可用的机器人模型：\n")
    
    for idx, model in enumerate(available_models, 1):
        print(f"  [{idx:2d}] {model}")
    
    print(f"\n  [ 0] 退出")
    print("\n" + "="*60)
    
    while True:
        try:
            choice = input("\n请输入数字选择机器人模型 (0-{}): ".format(len(available_models)))
            choice = int(choice)
            
            if choice == 0:
                print("\n退出启动器...")
                exit(0)
            
            if 1 <= choice <= len(available_models):
                selected_model = available_models[choice - 1]
                print(f"\n✅ 您选择了: {selected_model}\n")
                return selected_model
            else:
                print(f"❌ 无效的选择！请输入 0 到 {len(available_models)} 之间的数字。")
        
        except ValueError:
            print("❌ 请输入有效的数字！")
        except KeyboardInterrupt:
            print("\n\n用户中断，退出...")
            exit(0)
        except EOFError:
            print("\n\n检测到非交互式环境，将使用默认配置...")
            return None

def generate_launch_description():
    
    # 获取yaml路径以读取urdf文件
    mujoco_pkg_path = get_package_share_directory('mujoco_simulator_python')
    yaml_path = mujoco_pkg_path + "/config/simulate.yaml"
    if not os.path.exists(yaml_path): raise FileNotFoundError(f"yaml未找到: {yaml_path}")
    
    # 获取robot_description路径
    robot_pkg_path = get_package_share_directory('robot_description')
    
    # 自动扫描robot_description目录获取可用的机器人模型列表
    available_models = scan_available_models(robot_pkg_path)
    print(f"  🔍 自动扫描发现 {len(available_models)} 个模型")
    
    if not available_models:
        raise RuntimeError(f"未找到任何可用的机器人模型！请检查 {robot_pkg_path}")
    
    # 交互式选择机器人模型
    import sys
    if sys.stdin.isatty():
        model_name = interactive_select_model(available_models)
        if not model_name:
            print("\n❌ 未选择模型，退出启动器...")
            exit(0)
    else:
        print("\n❌ 检测到非交互式环境，无法启动！")
        print("提示：本启动器仅支持交互式选择模型。")
        exit(1)
    
    print(f"\n{'='*60}")
    print(f"  🤖 启动机器人模型: {model_name}")
    print(f"{'='*60}\n")
    
    # 搜索mjcf文件
    mjcf_path = find_matching_file(robot_pkg_path, model_name, "xml")
    if mjcf_path is None:
        raise FileNotFoundError(f"未找到匹配的MJCF文件: {model_name}.xml 在 {robot_pkg_path}")
    
    print(f"  📁 找到MJCF文件: {mjcf_path}")
    
    # 搜索urdf文件（需要清理model_name中的特殊字段）
    clean_model_name = model_name.replace("_float", "").replace("scene_", "").replace("_bind", "")
    urdf_path = find_matching_file(robot_pkg_path, clean_model_name, "urdf")
    
    if urdf_path is None:
        raise FileNotFoundError(f"未找到匹配的URDF文件: {clean_model_name}.urdf 在 {robot_pkg_path}")
    
    print(f"  📁 找到URDF文件: {urdf_path}")
    print(f"{'='*60}\n") 

    # 获取当前时间，用于确定bags的保存路径
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bag_folder_path = "bags/" + f"{timestamp}"

    return LaunchDescription([
        Node(
            package='mujoco_simulator_python',
            executable='mujoco_simulator_python',
            name='mujoco_simulator',
            output='both',
            emulate_tty=True,
            parameters=[
                {
                    'yaml_path': yaml_path,
                    'mjcf_path': mjcf_path,
                },
            ]
        ),
        # 发布机器人状态以可视化
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output={'stdout': 'log', 'stderr': 'log'},
            arguments=['--ros-args', '--log-level', 'FATAL'],
            parameters=[{'robot_description': open(urdf_path).read()}]
        ),
        # foxglove节点
        Node(
            package='foxglove_bridge',
            executable='foxglove_bridge',
            name='foxglove_bridge',
            output={'stdout': 'log', 'stderr': 'log'},
            arguments=['--ros-args', '--log-level', 'FATAL'],
        ),
        # 进行rosbag2录制
        # ExecuteProcess(
        #     cmd=['ros2', 'bag', 'record', '-o', bag_folder_path, '-a', '-s', 'mcap'],  # 录制所有话题到 my_bag 目录
        #     output='screen'  # 显示录制日志
        # ),
    ])
