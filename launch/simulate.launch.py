from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python import get_package_share_directory
import os
from datetime import datetime
import re
import glob

def extract_urdf_binding(mjcf_path):
    """从MJCF文件的注释中提取URDF绑定信息
    
    查找格式: <!-- urdf_bind: filename.urdf -->
    
    Args:
        mjcf_path: MJCF文件的完整路径
        
    Returns:
        str: 绑定的URDF文件名，如果未找到则返回None
    """
    with open(mjcf_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找 <!-- urdf_bind: filename.urdf --> 格式的注释
    match = re.search(r'<!--\s*urdf_bind:\s*(.+?\.urdf)\s*-->', content)
    if match:
        return match.group(1).strip()
    return None

def extract_included_files(mjcf_path):
    """从MJCF文件中提取所有include的文件路径
    
    查找格式: <include file="filename.xml"/>
    
    Args:
        mjcf_path: MJCF文件的完整路径
        
    Returns:
        list: include的文件名列表
    """
    with open(mjcf_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找所有 <include file="xxx.xml"/> 或 <include file="xxx.xml" /> 格式
    matches = re.findall(r'<include\s+file="([^"]+)"\s*/?>', content)
    return matches

def find_urdf_binding_recursive(mjcf_path, robot_pkg_path, visited=None):
    """递归地从MJCF文件及其include的文件中查找URDF绑定
    
    Args:
        mjcf_path: MJCF文件的完整路径
        robot_pkg_path: robot_description包的路径
        visited: 已访问的文件集合（防止循环引用）
        
    Returns:
        str: 绑定的URDF文件名，如果未找到则返回None
    """
    if visited is None:
        visited = set()
    
    # 规范化路径以避免重复访问
    mjcf_path = os.path.normpath(mjcf_path)
    if mjcf_path in visited:
        return None
    visited.add(mjcf_path)
    
    # 首先检查当前文件是否有urdf_bind注释
    urdf_binding = extract_urdf_binding(mjcf_path)
    if urdf_binding:
        return urdf_binding
    
    # 如果没有，查找include的文件
    included_files = extract_included_files(mjcf_path)
    mjcf_dir = os.path.dirname(mjcf_path)
    
    for included_file in included_files:
        # 构建include文件的完整路径
        included_path = os.path.join(mjcf_dir, included_file)
        
        if os.path.exists(included_path):
            # 递归查找
            urdf_binding = find_urdf_binding_recursive(included_path, robot_pkg_path, visited)
            if urdf_binding:
                return urdf_binding
    
    return None

def find_urdf_by_binding(robot_pkg_path, urdf_filename):
    """根据绑定的URDF文件名查找完整路径
    
    Args:
        robot_pkg_path: robot_description包的路径
        urdf_filename: 绑定的URDF文件名
        
    Returns:
        str: URDF文件的完整路径，如果未找到则返回None
    """
    # 在所有子目录中搜索指定的URDF文件
    search_pattern = os.path.join(robot_pkg_path, "**", urdf_filename)
    matches = glob.glob(search_pattern, recursive=True)
    
    if matches:
        return matches[0]
    return None

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
    """扫描robot_description目录，自动发现可用的机器人模型
    
    只返回包含有效URDF绑定的MJCF场景文件（包括递归查找include的文件）
    """
    available_models = []
    
    # 搜索所有 scene_*.xml 文件
    mjcf_pattern = os.path.join(robot_pkg_path, "**", "mjcf", "scene_*.xml")
    mjcf_files = glob.glob(mjcf_pattern, recursive=True)
    
    for mjcf_file in mjcf_files:
        # 从文件名提取模型名称: scene_ModelName.xml -> ModelName
        basename = os.path.basename(mjcf_file)
        if basename.startswith("scene_") and basename.endswith(".xml"):
            model_name = basename[6:-4]  # 去掉 "scene_" 前缀和 ".xml" 后缀
            
            # 递归检查MJCF文件及其include的文件中是否有URDF绑定注释
            urdf_binding = find_urdf_binding_recursive(mjcf_file, robot_pkg_path)
            
            if urdf_binding:
                # 验证绑定的URDF文件是否存在
                urdf_path = find_urdf_by_binding(robot_pkg_path, urdf_binding)
                if urdf_path:
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
    
    # 递归从MJCF文件及其include的文件中提取URDF绑定信息
    urdf_binding = find_urdf_binding_recursive(mjcf_path, robot_pkg_path)
    
    if urdf_binding is None:
        print(f"\n❌ 错误：MJCF文件及其include的文件中未找到URDF绑定信息！")
        print(f"   文件: {mjcf_path}")
        print(f"   请在MJCF文件或其include的文件中添加绑定注释，格式如下：")
        print(f"   <!-- urdf_bind: your_robot.urdf -->")
        raise RuntimeError(f"未找到URDF绑定信息，请在MJCF文件或其include的文件中添加 '<!-- urdf_bind: xxx.urdf -->' 注释")
    
    print(f"  📎 URDF绑定: {urdf_binding}")
    
    # 根据绑定信息查找URDF文件
    urdf_path = find_urdf_by_binding(robot_pkg_path, urdf_binding)
    
    if urdf_path is None:
        raise FileNotFoundError(f"未找到绑定的URDF文件: {urdf_binding} 在 {robot_pkg_path}")
    
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
