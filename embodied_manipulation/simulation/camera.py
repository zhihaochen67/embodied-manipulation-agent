"""Fixed external camera configuration for the Phase 1 scene."""

from dataclasses import dataclass
from typing import Any

import pybullet


@dataclass(frozen=True, slots=True)
class FixedCamera:
    """A fixed camera suitable for inspecting the tabletop scene."""

    target: tuple[float, float, float] = (0.5, 0.0, 0.35)
    distance: float = 1.8
    yaw: float = 45.0
    pitch: float = -30.0
    roll: float = 0.0
    field_of_view: float = 60.0
    near_plane: float = 0.1
    far_plane: float = 4.0
    width: int = 640
    height: int = 480

    def view_matrix(self) -> tuple[float, ...]:
        return tuple(
            pybullet.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=self.target,
                distance=self.distance,
                yaw=self.yaw,
                pitch=self.pitch,
                roll=self.roll,
                upAxisIndex=2,
            )
        )

    def projection_matrix(self) -> tuple[float, ...]:
        return tuple(
            pybullet.computeProjectionMatrixFOV(
                fov=self.field_of_view,
                aspect=self.width / self.height,
                nearVal=self.near_plane,
                farVal=self.far_plane,
            )
        )

    def render(self, client_id: int) -> tuple[int, int, Any, Any, Any]:
        """Render the scene with PyBullet's headless-capable tiny renderer."""
        return pybullet.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=self.view_matrix(),
            projectionMatrix=self.projection_matrix(),
            renderer=pybullet.ER_TINY_RENDERER,
            physicsClientId=client_id,
        )

    def configure_debug_view(self, client_id: int) -> None:
        """Apply this view to a GUI client's debug visualizer."""
        pybullet.resetDebugVisualizerCamera(
            cameraDistance=self.distance,
            cameraYaw=self.yaw,
            cameraPitch=self.pitch,
            cameraTargetPosition=self.target,
            physicsClientId=client_id,
        )
