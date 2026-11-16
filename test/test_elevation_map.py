"""用真实射线检测验证采样间隔内的高程图显示保持固定。"""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import mujoco
import numpy as np

from mujoco_simulator_python.plugins.elevation_map import ElevationMap


class ElevationMapTests(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
            <geom name="floor" type="plane" size="5 5 0.1" group="2"/>
            <geom name="step" type="box" pos="0.1 -0.05 0.15" size="0.09 0.09 0.15" group="2"/>
            <body name="base" pos="0 0 1"><freejoint/>
                <geom type="sphere" size="0.05" group="1"/>
                <site name="lidar_site" pos="0.1 -0.05 0.2"/>
            </body></worldbody></mujoco>''')
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.simulator = SimpleNamespace(mj_model=self.model, mj_data=self.data, step_counter=1, get_logger=lambda: Mock())
        self.plugin = ElevationMap({"step_interval": 8, "size": [0.4, 0.4], "resolution": 0.2}, self.simulator)
        self.plugin.send = Mock()

    def positions(self):
        geoms = []
        self.plugin.visualize(geoms)
        return np.array([geom.pos.copy() for geom in geoms])

    def sample(self, step):
        self.simulator.step_counter = step
        self.plugin.after_step()

    def test_no_geometry_before_first_scan(self):
        self.sample(1)
        self.assertFalse(self.plugin.has_sample)
        self.assertEqual(self.positions().size, 0)

    def test_cached_world_points_do_not_move_between_scans(self):
        self.sample(8)
        original_data = self.plugin.elevation_sample_point.copy()
        original_positions = self.positions()
        self.assertEqual(original_positions.shape, (9, 3))
        self.assertAlmostEqual(original_positions[:, 2].min(), 0.0, places=6)
        self.assertAlmostEqual(original_positions[:, 2].max(), 0.3, places=6)

        # 同时改变机器人高度和姿态，下一次采样前历史世界坐标应保持不变。
        self.data.qpos[2] += 0.2
        self.data.qpos[3:7] = [np.cos(0.25), np.sin(0.25), 0.0, 0.0]
        mujoco.mj_forward(self.model, self.data)
        self.sample(9)
        np.testing.assert_array_equal(self.plugin.elevation_sample_point, original_data)
        np.testing.assert_allclose(self.positions(), original_positions, atol=1e-7)

    def test_new_scan_updates_distances_without_changing_terrain_height(self):
        self.sample(8)
        original_distances = self.plugin.elevation_sample_point[:, 2].copy()
        original_positions = self.positions()
        self.data.qpos[2] += 0.4
        mujoco.mj_forward(self.model, self.data)
        self.sample(16)
        np.testing.assert_allclose(self.plugin.elevation_sample_point[:, 2] - original_distances, 0.4, atol=1e-6)
        np.testing.assert_allclose(self.positions(), original_positions, atol=1e-6)
        # 对外数据继续使用射线距离。
        np.testing.assert_array_equal(self.plugin.send.call_args.args[0], self.plugin.elevation_sample_point[:, 2])

    def test_reset_hides_previous_scan_and_allows_resampling(self):
        self.sample(8)
        expected = self.positions()
        self.plugin.reset()
        self.assertEqual(self.positions().size, 0)
        self.data.qpos[2] -= 0.1
        mujoco.mj_forward(self.model, self.data)
        self.sample(16)
        np.testing.assert_allclose(self.positions(), expected, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
