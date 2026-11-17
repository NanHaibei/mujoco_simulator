# MuJoCo Studio 仿真器

从旧 `mujoco_simulator` 迁入，使用 MuJoCo 4.0.0 Studio Web Viewer。保留模型信息、PD 控制、状态采样、高程图、雷达和地形插件的本地计算。原 ROS 发布、订阅、服务、TF、执行器线程和消息类型已移除，G1 控制与状态通过宇树 SDK2 的 DDS 通道收发。

## 运行

在 `src/mujoco_simulator` 中执行：

```bash
uv run mujoco_simulator_python/mujoco_simulator_python.py --config g1_base.yaml
uv run mujoco_simulator_python/mujoco_simulator_python.py --config simulate.yaml
```

必须通过 `--config` 指定本项目 `config/` 目录下已有的 `.yaml` 文件名；未指定时抛出 `ValueError`。在工作区根目录运行：

```bash
uv run --project src/mujoco_simulator \
  src/mujoco_simulator/mujoco_simulator_python/mujoco_simulator_python.py \
  --config g1_base.yaml
```

终端会输出网页地址，手动打开即可。可用 `--port 8080` 指定端口。启动入口统一使用 `mujoco_simulator_python/mujoco_simulator_python.py`。

`g1_base.yaml` 配置加载本工作区的 G1 29 自由度平地场景。模型路径仅从 YAML 的 `mjcfPath` 读取；相对路径以 YAML 所在目录为基准解析，也可以在 YAML 中使用绝对路径。MJCF 仅在程序启动时加载，更换模型需修改配置并重新启动程序。

无需浏览器的检查方式：

```bash
uv run python -m unittest discover -s test -v
```

## 配置与步进

`main()` 使用 `argparse` 读取配置选择。配置顶层统一使用 `mujoco_simulator`：

- `simRate`：物理频率，设置 `model.opt.timestep = 1 / simRate`；省略则采用 MJCF 设置。
- `renderRate`：插件额外几何体快照的近似目标频率，通过整数步抽取；网页实际渲染由 Studio 管理。
- `plugins`：列表中的插件全部按顺序加载，使用 `path: "module:Class"`、`step_interval` 和计算参数。模型信息插件应放在最前面。

`MujocoSimulator` 继承 `step_control.StepControl`，通过 `launch_web.run(...)` 注册自身的 `@messages.handler def _on_step(self, event: messages.StepEvent)`。每次最多推进一个物理步，先调用插件 `before_step()`，再进行 `mj_step`、`mj_forward` 和插件 `after_step()`。暂停时不采样、不增加步计数；单步仍可执行一个物理步。速度按 Studio 的百分比设置控制 sleep。此实现采用简单实时节奏，不使用默认步进器的追赶步进和控制噪声。

插件的周期性业务仅使用 `before_step()` 和 `after_step()` 两个接口，默认都是空实现。主循环每个物理步都会调用它们，由具体插件在接口开头通过 `self.should_execute()` 判断 `step_interval`，未到执行时刻就返回。步前、步后共享同一个物理步编号；例如 `step_interval: 3` 时，均在第 3、6、9 步执行。间隔以实际物理步数为准，暂停时不会推进；重置后重新计数。重写 `log()` 时也应在插件内部判断执行间隔。

不要再同时注册 `app.SIM_PLUGINS`，否则会出现重复步进。插件额外几何体通过 Studio 的快照接口传给 Viewer，不再使用旧被动窗口的 `user_scn`。

## 宇树 SDK 与通信接口

`BasePlugin` 不定义 `send()` 或 `receive()`；通信方法由各子类自行定义。G1 插件已实现 SDK 命令接收与状态发送；其余插件在各自类中保留尚未接入外部通信的收发占位方法，`Imu2` 复用具体 IMU 插件 `Imu` 的实现。

- `UnitreeG1SDK`（`unitree_g1_sdk:UnitreeG1SDK`）订阅 `command_topic`（默认 `rt/lowcmd`）的 HG `LowCmd_`，在 `before_step()` 中读取缓存命令并计算 PD 力矩；控制端停发后持续使用最后一条有效命令。`after_step()` 从 `sensordata` 采样电机位置、速度、力矩及 IMU 四元数、角速度、加速度，转换为 HG `LowState_` 并发布至 `state_topic`（默认 `rt/lowstate`）；尚未收到命令时也正常发布状态。状态四元数采用 `(w,x,y,z)`，`rpy` 单位为弧度，`tick` 为采样时的仿真毫秒数，发送前计算 CRC。保留 `cmdDelay`、`stateDelay`、力矩限幅及四项状态噪声参数；收发共用 `step_interval`。`reset()` 清空收发缓存和延迟队列，退出时 `close()` 释放订阅器和发布器。`g1_base.yaml` 和 `simulate.yaml` 不再单独加载 `LowState`。运行该插件需要在所用 Python 环境中安装 `unitree_sdk2py` 及其依赖。
- `LowState`、`Imu`、`Imu2`、`JointStates`、`Odom` 保留本地 `last_state` 快照，再调用各插件自行定义的 `send()` 占位方法。时间使用仿真秒数，四元数采用 MuJoCo 的 `(w,x,y,z)` 顺序。
- 高程图和水平雷达保留本地数组与 Studio 可视化。`DlioRadarVisualizer` 在自身类中定义 `receive()`，当前返回 `None`，为外部输入预留接入位置。
- LiDAR 默认使用随仓库迁入的 `third_party/MuJoCo-LiDAR` CPU 后端，输出传感器局部坐标点云至 `last_points`。Taichi/JAX 后端及第三方独立示例的额外依赖不在默认运行依赖中。
- `MapFrame` 仅保留空插件，原 TF 通信逻辑已删除。

本目录使用 `pyproject.toml` 和 `uv.lock` 管理依赖。原 ROS 专用的 `package.xml`、ament 资源、launch 文件和 lint 测试没有保留；第三方 ROS 示例和 RViz 配置也已移除。第三方计算后端、模型、扫描数据及许可证保留。正式目录已由验证版本完整覆盖，覆盖前文件的本地备份位于工作区 `agent_ws/mujoco_publish_20261116_p08ppv3c/backup_target/`。
