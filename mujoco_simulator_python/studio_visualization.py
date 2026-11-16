"""通过 Studio 快照传递插件额外几何体。"""

from dataclasses import dataclass

import mujoco
from mujoco.studio import messages, plugin


@dataclass(frozen=True)
class OverlaySnapshot(messages.Snapshot):
    geoms: list[mujoco.MjvGeom]


class PluginVisualization(plugin.ViewerPlugin):
    @messages.handler
    def _on_overlay(self, snapshot: OverlaySnapshot):
        self.viewer.extra_geoms[:] = snapshot.geoms
