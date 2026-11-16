"""保留本地地形信息，发送接口暂时留空。"""

from .base import BasePlugin


class Terrain(BasePlugin):
    def after_step(self):
        """按步间隔整理地形信息并交给发送接口。"""
        if not self.should_execute():
            return
        sim = self.simulator
        self.last_state = [
            {"position": pos.copy(), "quaternion": quat.copy(),
             "size": size.copy(), "rgba": rgba.copy(), "type": kind}
            for pos, quat, size, rgba, kind in zip(sim.terrain_pos, sim.terrain_quat,
                sim.terrain_size, sim.terrain_rgba, sim.terrain_type)
        ]
        self.send(self.last_state)

    def send(self, state):
        """发送本插件的采样结果，外部通信待本插件接入。"""
        pass
