"""使用真实 G1 模型、SDK 消息和 CRC 验证收发；DDS 网络由独立检查覆盖。"""

from copy import deepcopy
import math
import unittest
from unittest.mock import patch

import numpy as np
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
import yaml

from mujoco_simulator_python.plugins import UnitreeG1SDK
from mujoco_simulator_python import mujoco_simulator_python as entry
from test_simulator import SimulatorFixture


class UnitreeG1SDKTests(SimulatorFixture, unittest.TestCase):
    def test_initializes_both_topics_once(self):
        self.ChannelFactoryInitialize.assert_called_once_with(0, None)
        self.ChannelSubscriber.assert_called_once_with("rt/lowcmd", LowCmd_)
        self.ChannelPublisher.assert_called_once_with("rt/lowstate", LowState_)
        self.subscriber.Init.assert_called_once_with(self.sdk._on_lowcmd, 1)
        self.publisher.Init.assert_called_once_with()

    def test_custom_transport_configuration(self):
        sdk = UnitreeG1SDK({"domain_id": 3, "interface": "lo", "command_topic": "test/cmd", "state_topic": "test/state"}, self.simulator)
        self.addCleanup(sdk.close)
        self.ChannelFactoryInitialize.assert_called_with(3, "lo")
        self.ChannelSubscriber.assert_called_with("test/cmd", LowCmd_)
        self.ChannelPublisher.assert_called_with("test/state", LowState_)

    def test_state_fields_crc_and_wire_round_trip_without_commands(self):
        sim = self.simulator
        data = sim.mj_data
        del sim.read_error_flag
        positions = np.linspace(-0.2, 0.2, 29)
        velocities = np.linspace(-0.3, 0.3, 29)
        efforts = np.linspace(-2.0, 2.0, 29)
        data.sensordata[sim.joint_pos_head_id:sim.joint_pos_head_id + 29] = positions
        data.sensordata[sim.joint_vel_head_id:sim.joint_vel_head_id + 29] = velocities
        data.sensordata[sim.joint_tor_head_id:sim.joint_tor_head_id + 29] = efforts
        quaternion = [math.cos(0.25), 0.0, 0.0, math.sin(0.25)]
        data.sensordata[sim.imu_quat_head_id:sim.imu_quat_head_id + 4] = quaternion
        data.sensordata[sim.imu_gyro_head_id:sim.imu_gyro_head_id + 3] = [0.1, -0.2, 0.3]
        data.sensordata[sim.imu_acc_head_id:sim.imu_acc_head_id + 3] = [1.0, 2.0, 9.81]
        data.time = 1.234
        sim.step_counter = 1
        self.sdk.after_step()
        self.assertIsNone(self.sdk.command)
        message = self.publisher.Write.call_args.args[0]
        self.assertIsInstance(message, LowState_)
        self.assertEqual(message.tick, 1234)
        self.assertEqual((message.mode_pr, message.mode_machine), (0, 0))
        np.testing.assert_allclose([m.q for m in message.motor_state[:29]], positions)
        np.testing.assert_allclose([m.dq for m in message.motor_state[:29]], velocities)
        np.testing.assert_allclose([m.tau_est for m in message.motor_state[:29]], efforts)
        self.assertTrue(all(m.mode == 1 for m in message.motor_state[:29]))
        self.assertEqual(message.motor_state[29:], unitree_hg_msg_dds__LowState_().motor_state[29:])
        np.testing.assert_allclose(message.imu_state.quaternion, quaternion)
        np.testing.assert_allclose(message.imu_state.gyroscope, [0.1, -0.2, 0.3])
        np.testing.assert_allclose(message.imu_state.accelerometer, [1.0, 2.0, 9.81])
        np.testing.assert_allclose(message.imu_state.rpy, [0.0, 0.0, 0.5], atol=1e-12)
        self.assertEqual(message.crc, self.sdk._crc.Crc(message))
        decoded = LowState_.deserialize(message.serialize())
        np.testing.assert_allclose([m.q for m in decoded.motor_state[:29]], positions, atol=1e-7)
        self.assertEqual(decoded.crc, self.sdk._crc.Crc(decoded))

    def test_state_delay_retains_sample_time_and_independent_snapshots(self):
        sim = self.simulator
        self.sdk.state_delay = 2
        for step in range(1, 5):
            sim.step_counter = step
            sim.mj_data.time = step * 0.01
            head = sim.joint_pos_head_id
            sim.mj_data.sensordata[head:head + 29] = step * 0.1
            self.sdk.after_step()
            self.assertEqual(self.publisher.Write.call_count, max(0, step - 2))
        first, second = [call.args[0] for call in self.publisher.Write.call_args_list]
        self.assertIsNot(first, second)
        self.assertEqual((first.tick, second.tick), (10, 20))
        np.testing.assert_allclose([m.q for m in first.motor_state[:29]], 0.1)
        np.testing.assert_allclose([m.q for m in second.motor_state[:29]], 0.2)
        np.testing.assert_allclose([motor.q for motor in self.sdk.last_state.motor_state[:29]], 0.2)
        self.assertEqual(len(self.sdk.state_deque), 2)
        self.assertEqual(first.crc, self.sdk._crc.Crc(first))

    def test_noise_is_applied_to_sent_state_without_changing_sensors(self):
        sim = self.simulator
        self.sdk.plugin_config.update(noise_joint_pos=0.1, noise_joint_vel=0.2, noise_imu_angle_acc=0.3, noise_imu_gravity=0.05)
        head = sim.imu_quat_head_id
        sim.mj_data.sensordata[head:head + 4] = [1.0, 0.0, 0.0, 0.0]
        original = sim.mj_data.sensordata.copy()
        def upper(low, high, size=None):
            return high if size is None else np.full(size, high)
        with patch("numpy.random.uniform", side_effect=upper), patch("numpy.random.randn", return_value=np.array([1.0, 0.0, 0.0])):
            self.sdk.after_step()
        np.testing.assert_array_equal(sim.mj_data.sensordata, original)
        np.testing.assert_allclose([motor.q for motor in self.sdk.last_state.motor_state[:29]], original[sim.joint_pos_head_id:sim.joint_pos_head_id + 29] + 0.1)
        np.testing.assert_allclose([motor.dq for motor in self.sdk.last_state.motor_state[:29]], original[sim.joint_vel_head_id:sim.joint_vel_head_id + 29] + 0.2)
        np.testing.assert_allclose(self.sdk.last_state.imu_state.gyroscope, original[sim.imu_gyro_head_id:sim.imu_gyro_head_id + 3] + 0.3)
        message = self.publisher.Write.call_args.args[0]
        np.testing.assert_allclose(message.imu_state.rpy, [0.05, 0.0, 0.0], atol=1e-12)
        self.assertAlmostEqual(np.linalg.norm(message.imu_state.quaternion), 1.0)
        self.assertEqual(message.crc, self.sdk._crc.Crc(message))

    def test_command_delay_keeps_last_message_when_sender_stops(self):
        sim = self.simulator
        self.sdk.cmd_delay = 2
        original = self.send_command(tau=0.7)
        original.motor_cmd[0].tau = 100.0
        for step in range(1, 7):
            sim._physics_step()
            if step < 3:
                self.assertIsNone(self.sdk.command)
            else:
                np.testing.assert_allclose(sim.mj_data.ctrl, 0.7)
        self.assertEqual(self.sdk.command.motor_cmd[0].tau, 0.7)
        self.assertEqual(self.publisher.Write.call_count, 6)

    def test_bad_crc_and_pr_mode_keep_last_command(self):
        command = self.send_command(tau=0.7)
        bad = deepcopy(command)
        bad.motor_cmd[0].tau = 5.0
        self.sdk._on_lowcmd(bad)
        bad.mode_pr = 1
        bad.crc = self.sdk._crc.Crc(bad)
        self.sdk._on_lowcmd(bad)
        self.simulator._physics_step()
        np.testing.assert_allclose(self.simulator.mj_data.ctrl, 0.7)

    def test_disabled_motor_and_limits(self):
        command = self.send_command(tau=1e6)
        command.motor_cmd[0].mode = 0
        command.crc = self.sdk._crc.Crc(command)
        self.sdk._on_lowcmd(command)
        self.simulator._physics_step()
        expected = self.simulator.mj_model.actuator_ctrlrange[:, 1].copy()
        expected[0] = 0.0
        np.testing.assert_allclose(self.simulator.mj_data.ctrl, expected)

    def test_reset_clears_both_delay_queues_and_waits_for_new_samples(self):
        sim = self.simulator
        self.sdk.cmd_delay = 1
        self.sdk.state_delay = 2
        self.send_command(tau=0.7)
        for _ in range(3):
            sim._physics_step()
        self.assertIsNotNone(self.sdk.last_state)
        self.sdk.reset()
        self.assertIsNone(self.sdk.command)
        self.assertIsNone(self.sdk._latest_command)
        self.assertIsNone(self.sdk.last_state)
        self.assertFalse(self.sdk.cmd_deque)
        self.assertFalse(self.sdk.state_deque)
        np.testing.assert_array_equal(sim.mj_data.ctrl, 0.0)
        self.publisher.Write.reset_mock()
        for _ in range(2):
            sim._physics_step()
        self.publisher.Write.assert_not_called()
        sim._physics_step()
        self.publisher.Write.assert_called_once()
        self.assertIsNone(self.sdk.command)
        np.testing.assert_array_equal(sim.mj_data.ctrl, 0.0)

    def test_close_attempts_both_channels_when_subscriber_fails(self):
        self.subscriber.Close.side_effect = RuntimeError("subscriber close failed")
        try:
            with self.assertRaisesRegex(RuntimeError, "subscriber close failed"):
                self.sdk.close()
            self.publisher.Close.assert_called_once_with()
        finally:
            self.subscriber.Close.side_effect = None

    def test_constructor_failure_releases_channels(self):
        self.publisher.Init.side_effect = RuntimeError("publisher init failed")
        try:
            with self.assertRaisesRegex(RuntimeError, "publisher init failed"):
                UnitreeG1SDK({}, self.simulator)
            self.subscriber.Close.assert_called_once_with()
            self.publisher.Close.assert_called_once_with()
        finally:
            self.publisher.Init.side_effect = None

    def test_ctrl_c_from_simulation_releases_both_channels(self):
        with patch.object(entry.launch_web, "run", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.simulator.run()
        self.subscriber.Close.assert_called_once_with()
        self.publisher.Close.assert_called_once_with()

    def test_both_configurations_load_a_single_sdk_plugin(self):
        for filename in ("g1_base.yaml", "simulate.yaml"):
            with self.subTest(filename=filename):
                config = yaml.safe_load((entry.PROJECT_ROOT / "config" / filename).read_text())
                plugins = config["mujoco_simulator"]["plugins"]
                self.assertEqual(plugins[0]["path"], "model_info:ModelInfo")
                sdk_plugins = [p for p in plugins if p["path"] == "unitree_g1_sdk:UnitreeG1SDK"]
                self.assertEqual(len(sdk_plugins), 1)
                self.assertEqual(sdk_plugins[0]["state_topic"], "rt/lowstate")
                self.assertFalse(any(p["path"] == "low_state:LowState" for p in plugins))


if __name__ == "__main__":
    unittest.main()
