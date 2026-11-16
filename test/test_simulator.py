"""对真实 G1 模型验证步进、插件及配置入口，无需 ROS 或浏览器。"""

import importlib
import unittest
from unittest.mock import Mock, patch

from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_

import mujoco
from mujoco.studio import messages, sim, step_control, viewer_handle
import numpy as np

from mujoco_simulator_python import mujoco_simulator_python as entry
from mujoco_simulator_python.plugins.base import BasePlugin

class SimulatorFixture:
    def setUp(self):
        for name in ("ChannelFactoryInitialize", "ChannelSubscriber", "ChannelPublisher"):
            channel_patch = patch("unitree_sdk2py.core.channel." + name)
            setattr(self, name, channel_patch.start())
            self.addCleanup(channel_patch.stop)
        self.publisher = self.ChannelPublisher.return_value
        self.subscriber = self.ChannelSubscriber.return_value
        self.simulator = entry.MujocoSimulator(entry.PROJECT_ROOT / "config" / "g1_base.yaml")
        self.simulator.handle = Mock(spec=viewer_handle.ViewerHandle)
        self.sdk = self.simulator.plugins[1]
        self.addCleanup(self.sdk.close)

    def send_command(self, tau=0.0, kp=0.0, kd=0.0, q=0.0, dq=0.0):
        command = unitree_hg_msg_dds__LowCmd_()
        for motor in command.motor_cmd[:self.simulator.mj_model.nu]:
            motor.mode = 1
            motor.tau, motor.kp, motor.kd = tau, kp, kd
            motor.q, motor.dq = q, dq
        command.crc = self.sdk._crc.Crc(command)
        self.sdk._on_lowcmd(command)
        return command


class SimulatorTests(SimulatorFixture, unittest.TestCase):
    def test_g1_state_matches_current_physics(self):
        simulator = self.simulator
        for _ in range(12):
            simulator._physics_step()
        self.assertEqual(len(simulator.plugins), 2)
        self.assertEqual(simulator.mj_model.nu, 29)
        self.assertEqual(simulator.step_counter, 12)
        self.assertAlmostEqual(simulator.mj_data.time, 12 / 200)
        state = self.sdk.last_state
        np.testing.assert_allclose([motor.q for motor in state.motor_state[:29]], simulator.mj_data.sensordata[:29])
        self.assertEqual(state.tick, int(simulator.mj_data.time * 1000))
        self.assertFalse(simulator.read_error_flag)
        self.assertIsNone(self.sdk.command)
        self.assertEqual(self.publisher.Write.call_count, 12)
        message = self.publisher.Write.call_args.args[0]
        self.assertIs(message, state)

    def test_pause_single_step_and_speed(self):
        simulator = self.simulator
        event = messages.StepEvent(simulator.mj_model, simulator.mj_data)
        simulator.step_control.set_pause_state(sim.PauseState.NORMAL_PAUSED)
        with patch.object(entry.time, "sleep") as sleep:
            simulator._on_step(event)
            self.assertEqual(simulator.step_counter, 0)
            simulator.step_control.request_single_step()
            simulator._on_step(event)
            self.assertEqual(simulator.step_counter, 1)
            simulator._on_step(event)
            self.assertEqual(simulator.step_counter, 1)
            self.assertEqual(simulator.step_stats.count, 1)
            simulator.step_control.set_pause_state(sim.PauseState.UNPAUSED)
            simulator.step_control.set_speed(50.0)
            simulator._on_step(event)
            self.assertEqual(simulator.step_counter, 2)
            self.assertGreater(sleep.call_args.args[0], 0)

    def test_control_applied_before_step_and_clipped(self):
        simulator = self.simulator
        self.send_command(tau=1e6)
        simulator._physics_step()
        np.testing.assert_allclose(simulator.mj_data.ctrl, simulator.mj_model.actuator_ctrlrange[:, 1])
        self.assertTrue(np.any(simulator.mj_data.qvel != 0))
        simulator.reset_plugins(messages.ResetEvent())
        self.assertIsNone(self.sdk.command)
        self.assertIsNone(self.sdk.last_state)
        self.assertEqual(simulator.step_counter, 0)

    def test_no_command_preserves_studio_control(self):
        simulator = self.simulator
        simulator.step_counter = 1
        simulator.mj_data.ctrl[:] = 0.25
        self.sdk.before_step()
        self.assertIsNone(self.sdk.command)
        np.testing.assert_array_equal(simulator.mj_data.ctrl, 0.25)

    def test_pd_feedback_comes_from_joint_sensors(self):
        simulator = self.simulator
        model, data = simulator.mj_model, simulator.mj_data
        sensor_pos = np.linspace(-0.2, 0.2, model.nu)
        sensor_vel = np.linspace(-0.1, 0.1, model.nu)
        pos_head, vel_head = simulator.joint_pos_head_id, simulator.joint_vel_head_id
        data.sensordata[pos_head:pos_head + model.nu] = sensor_pos
        data.sensordata[vel_head:vel_head + model.nu] = sensor_vel
        data.qpos[:] = 7.0
        data.qvel[:] = -9.0
        self.send_command(kp=2.0, kd=0.5, q=0.1, dq=0.2, tau=0.3)
        simulator.step_counter = 1
        self.sdk.before_step()
        expected = 2.0 * (0.1 - sensor_pos) + 0.5 * (0.2 - sensor_vel) + 0.3
        np.testing.assert_allclose(data.ctrl, expected)
        np.testing.assert_allclose(data.sensordata[pos_head:pos_head + model.nu], sensor_pos)
        np.testing.assert_allclose(data.sensordata[vel_head:vel_head + model.nu], sensor_vel)

    def test_decimation_counts_physical_steps(self):
        class Probe(BasePlugin):
            def __init__(self, simulator):
                super().__init__({"step_interval": 3}, simulator)
                self.before = []
                self.after = []
                self.logged = []

            def before_step(self):
                if not self.should_execute():
                    return
                self.before.append((self.simulator.step_counter, float(self.mj_data.time)))

            def after_step(self):
                if not self.should_execute():
                    return
                self.after.append((self.simulator.step_counter, float(self.mj_data.time)))

            def log(self):
                if not self.should_execute():
                    return
                self.logged.append(self.simulator.step_counter)

        simulator = self.simulator
        probe = Probe(simulator)
        simulator.plugins.append(probe)
        simulator.enable_log_output = True
        simulator.logger = Mock()
        for _ in range(10):
            simulator._physics_step()
        self.assertEqual([step for step, _ in probe.before], [3, 6, 9])
        self.assertEqual([step for step, _ in probe.after], [3, 6, 9])
        self.assertEqual(probe.logged, [3, 6, 9])
        timestep = simulator.mj_model.opt.timestep
        np.testing.assert_allclose([time for _, time in probe.before], np.array([2, 5, 8]) * timestep)
        np.testing.assert_allclose([time for _, time in probe.after], np.array([3, 6, 9]) * timestep)

    def test_control_and_state_hooks_respect_intervals(self):
        simulator = self.simulator
        self.sdk.step_interval = 3
        self.send_command(tau=0.5)
        initial_ctrl = simulator.mj_data.ctrl.copy()
        for step in range(1, 8):
            simulator._physics_step()
            self.assertEqual(self.publisher.Write.call_count, step // 3)
            if step < 3:
                self.assertIsNone(self.sdk.last_state)
                self.assertIsNone(self.sdk.command)
                np.testing.assert_allclose(simulator.mj_data.ctrl, initial_ctrl)
            else:
                np.testing.assert_allclose(simulator.mj_data.ctrl, 0.5)
                self.assertEqual(self.sdk.last_state.tick, int((step // 3) * 3 * simulator.mj_model.opt.timestep * 1000))

    def test_interval_waits_while_paused_and_resumes_on_single_steps(self):
        simulator = self.simulator
        self.sdk.step_interval = 3
        self.send_command()
        simulator.step_control.set_pause_state(sim.PauseState.NORMAL_PAUSED)
        event = messages.StepEvent(simulator.mj_model, simulator.mj_data)
        with patch.object(entry.time, "sleep"):
            for _ in range(5):
                simulator._on_step(event)
            self.assertEqual(simulator.step_counter, 0)
            self.publisher.Write.assert_not_called()
            for step in range(1, 4):
                simulator.step_control.request_single_step()
                simulator._on_step(event)
                simulator._on_step(event)
                self.assertEqual(simulator.step_counter, step)
                self.assertEqual(self.publisher.Write.call_count, step // 3)
        self.assertEqual(self.sdk.last_state.tick, int(3 * simulator.mj_model.opt.timestep * 1000))

    def test_all_optional_plugins_and_studio_geometries(self):
        simulator = self.simulator
        for path in ("elevation_map:ElevationMap", "horizontal_radar:HorizontalRadar",
                     "dlio_radar_visualizer:DlioRadarVisualizer", "imu:Imu", "imu2:Imu2",
                     "joint_state:JointStates", "odom:Odom", "terrain:Terrain",
                     "map_frame:MapFrame", "lidar:Lidar"):
            module, name = path.split(":")
            cls = getattr(importlib.import_module("mujoco_simulator_python.plugins." + module), name)
            config = {"step_interval": 2, "site_name": "lidar_site", "goal_position": [2.0, 0.0]}
            instance = cls(config, simulator)
            if module == "lidar":
                instance.livox_generator.samples = 32
            simulator.plugins.append(instance)
        simulator._physics_step()
        self.assertEqual(simulator._geometry_snapshot().geoms, [])
        self.assertIsNone(simulator.plugins[-1].last_points)
        simulator._physics_step()
        overlays = simulator._geometry_snapshot().geoms
        self.assertGreater(len(overlays), 500)
        self.assertTrue(all(isinstance(geom, mujoco.MjvGeom) for geom in overlays))
        lidar = simulator.plugins[-1]
        self.assertEqual(lidar.last_points.shape, (32, 3))
        for plugin in simulator.plugins:
            plugin.reset()
        self.assertIsNone(lidar.last_points)
        self.assertFalse(simulator.plugins[2].has_sample)

    def test_cli_resolves_config_filename(self):
        with patch.object(entry, "MujocoSimulator") as cls:
            entry.main(["--config", "simulate.yaml", "--port", "8099"])
            self.assertEqual(cls.call_args.args[0], entry.PROJECT_ROOT / "config/simulate.yaml")
            cls.return_value.run.assert_called_once_with(port=8099)

    def test_web_run_has_one_physics_plugin(self):
        with patch.object(entry.launch_web, "run") as run:
            self.simulator.run()
        plugins = run.call_args.kwargs["plugins"]
        self.assertEqual(sum(isinstance(p, step_control.StepControl) for p in plugins), 1)
        self.assertIs(run.call_args.kwargs["data"], self.simulator.mj_data)

    def test_sensor_addresses_with_multidimensional_sensor(self):
        # 未识别的三维传感器排在关节传感器前面时，地址仍应正确。
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <body name="base"><joint name="j"/><geom size="0.1"/>
          <site name="imu"/></body></worldbody><actuator><motor joint="j"/></actuator>
          <sensor><velocimeter name="extra" site="imu"/>
          <jointpos name="j_pos" joint="j"/><jointvel name="j_vel" joint="j"/>
          <jointactuatorfrc name="j_force" joint="j"/>
          <framequat name="imu_quat" objtype="site" objname="imu"/>
          <gyro name="imu_gyro" site="imu"/><accelerometer name="imu_acc" site="imu"/>
          </sensor></mujoco>''')
        self.simulator._initialize_model(model)
        self.assertEqual(self.simulator.joint_pos_head_id, 3)
        self.assertEqual(self.simulator.imu_quat_head_id, 6)
        self.assertEqual(len(self.simulator.sensor_type), model.nsensordata)
        self.assertEqual(self.simulator.plugins[0]._get_body_name_by_sensor(4), "base")


if __name__ == "__main__":
    unittest.main()
