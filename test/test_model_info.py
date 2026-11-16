"""验证必需传感器缺失时终止初始化，且校验不依赖表格显示。"""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import mujoco
import yaml

from mujoco_simulator_python import mujoco_simulator_python as entry
from mujoco_simulator_python.plugins.model_info import ModelInfo, REQUIRED_SENSORS


SENSORS = {
    "joint_pos_head_id": '<jointpos name="joint_pos" joint="joint"/>',
    "joint_vel_head_id": '<jointvel name="joint_vel" joint="joint"/>',
    "joint_tor_head_id": '<jointactuatorfrc name="joint_torque" joint="joint"/>',
    "imu_quat_head_id": '<framequat name="imu_quat" objtype="site" objname="imu"/>',
    "imu_gyro_head_id": '<gyro name="imu_gyro" site="imu"/>',
    "imu_acc_head_id": '<accelerometer name="imu_acc" site="imu"/>',
}


def model_xml(missing=()):
    sensors = "".join(xml for attr, xml in SENSORS.items() if attr not in missing)
    return '''<mujoco><worldbody><body name="base">
        <joint name="joint"/><geom type="sphere" size="0.1"/>
        <site name="imu"/></body></worldbody>
        <actuator><motor joint="joint"/></actuator>
        <sensor>''' + sensors + '</sensor></mujoco>'


def simulator_for(model):
    logger = Mock()
    return SimpleNamespace(mj_model=model, mj_data=mujoco.MjData(model), step_counter=0, get_logger=lambda: logger)


class ModelInfoTests(unittest.TestCase):
    def test_each_missing_sensor_fails_with_table_enabled_or_disabled(self):
        for attr, description in REQUIRED_SENSORS:
            for show_table in (False, True):
                with self.subTest(sensor=attr, show_table=show_table):
                    model = mujoco.MjModel.from_xml_string(model_xml({attr}))
                    simulator = simulator_for(model)
                    with self.assertRaisesRegex(ValueError, description):
                        ModelInfo({"show_table": show_table}, simulator)

    def test_error_lists_all_missing_sensors(self):
        model = mujoco.MjModel.from_xml_string(model_xml(SENSORS))
        simulator = simulator_for(model)
        with self.assertRaises(ValueError) as caught:
            ModelInfo({"show_table": False}, simulator)
        for _, description in REQUIRED_SENSORS:
            self.assertIn(description, str(caught.exception))
        self.assertIn("MJCF", str(caught.exception))
        simulator.get_logger().info.assert_not_called()

    def test_complete_model_initializes_in_both_display_modes(self):
        for show_table in (False, True):
            with self.subTest(show_table=show_table):
                model = mujoco.MjModel.from_xml_string(model_xml())
                simulator = simulator_for(model)
                with patch.object(ModelInfo, "_show_model") as show:
                    ModelInfo({"show_table": show_table}, simulator)
                self.assertEqual(show.call_count, int(show_table))
                for attr, _ in REQUIRED_SENSORS:
                    self.assertGreaterEqual(getattr(simulator, attr), 0)

    def test_invalid_model_stops_startup_before_sdk_initialization(self):
        with TemporaryDirectory() as directory:
            directory = Path(directory)
            model_path = directory / "missing_gyro.xml"
            model_path.write_text(model_xml({"imu_gyro_head_id"}))
            config_path = directory / "missing_gyro.yaml"
            config_path.write_text(yaml.safe_dump({"mujoco_simulator": {
                "mjcfPath": str(model_path),
                "plugins": [
                    {"path": "model_info:ModelInfo", "show_table": False},
                    {"path": "unitree_g1_sdk:UnitreeG1SDK"},
                ],
            }}))
            with patch("unitree_sdk2py.core.channel.ChannelFactoryInitialize") as initialize, \
                    patch("unitree_sdk2py.core.channel.ChannelPublisher") as publisher, \
                    patch("unitree_sdk2py.core.channel.ChannelSubscriber") as subscriber:
                with self.assertRaisesRegex(ValueError, "角速度传感器"):
                    entry.MujocoSimulator(config_path)
            initialize.assert_not_called()
            publisher.assert_not_called()
            subscriber.assert_not_called()


if __name__ == "__main__":
    unittest.main()
