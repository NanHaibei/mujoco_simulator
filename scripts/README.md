# MuJoCo 仿真启动说明

## 使用方式

### 方式 1：交互式选择（推荐）⭐

直接运行 launch 文件，会自动弹出交互式选择菜单：

```bash
ros2 launch mujoco_simulator_python simulate.launch.py
```

程序会显示可用的机器人模型列表，输入对应数字即可启动。

### 方式 2：命令行直接指定模型

通过命令行参数直接指定要启动的机器人模型：

```bash
ros2 launch mujoco_simulator_python simulate.launch.py model_name:=G1_29dof_full_collision
```

### 优先级说明

模型选择优先级：**命令行参数 > 交互式选择 > YAML配置**

1. 如果指定了 `model_name:=xxx` 参数，直接使用该模型
2. 如果未指定参数且在交互式终端中运行，显示选择菜单
3. 如果在非交互式环境（如脚本、后台运行），从 `simulate.yaml` 读取默认配置

## 配置文件

可用的机器人模型列表现在配置在 `simulate.yaml` 中：

```yaml
mujoco_simulator:
    # 可用的机器人模型列表（用于交互式选择）
    availableModels:
        - "G1_29dof_full_collision"
        - "G1_29dof_box_foot"
        - "G1_29dof_float"
        - "G1_12dof_box"
        - "G1_12dof"
        - "S3_22dof"
        - "S2_22dof"
        - "S2_20dof_simp_col"
        - "S1_20dof_simp_col"
        - "zsl1"
        - "L1"
        - "Pegasus2"
        - "Go2"
    
    # 默认使用的模型名称
    modelName: "G1_29dof_full_collision"
```

您可以通过编辑 YAML 文件来添加、删除或重新排序可用的机器人模型。

## 使用示例

### 交互式启动
```bash
# 直接运行，会弹出选择菜单
ros2 launch mujoco_simulator_python simulate.launch.py

# 输出示例：
# ============================================================
#       MuJoCo 机器人仿真启动器
# ============================================================
# 
# 可用的机器人模型：
# 
#   [ 1] G1_29dof_full_collision
#   [ 2] G1_29dof_box_foot
#   ...
#   [ 0] 退出
# 
# ============================================================
# 请输入数字选择机器人模型 (0-13): 
```

### 命令行指定
```bash
# 启动 G1 机器人（完整碰撞模型）
ros2 launch mujoco_simulator_python simulate.launch.py model_name:=G1_29dof_full_collision

# 启动 S3 机器人
ros2 launch mujoco_simulator_python simulate.launch.py model_name:=S3_22dof

# 启动 Go2 四足机器人
ros2 launch mujoco_simulator_python simulate.launch.py model_name:=Go2
```

### 非交互式环境
```bash
# 在脚本中使用，会自动读取yaml配置
./my_script.sh  # 里面调用 ros2 launch
```

## 添加新机器人模型

要添加新的机器人模型到选择列表中，只需编辑 `simulate.yaml` 文件：

1. 打开 `config/simulate.yaml`
2. 在 `availableModels` 列表中添加新模型名称
3. 保存文件

例如：
```yaml
availableModels:
    - "G1_29dof_full_collision"
    - "MyNewRobot"  # 添加新模型
    - "AnotherRobot"  # 添加另一个新模型
```

## 修改内容

修改了以下文件：

1. **simulate.yaml** - 添加了 `availableModels` 列表配置
2. **simulate.launch.py** - 从YAML读取可用模型列表，实现三种启动方式：
   - ✅ 交互式选择：在交互式终端中自动显示选择菜单
   - ✅ 命令行参数：支持 `model_name:=xxx` 参数快速指定
   - ✅ 配置文件：向后兼容，非交互式环境使用yaml配置

无需任何额外脚本，所有功能都集成在 launch 文件中，配置集中在 YAML 文件中！
