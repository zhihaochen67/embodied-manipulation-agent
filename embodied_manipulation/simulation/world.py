"""Minimal deterministic PyBullet world for Phase 1."""

from dataclasses import dataclass

import numpy as np
import pybullet
import pybullet_data

from .camera import FixedCamera
from .objects import CUBE_HALF_EXTENT, GROUND_Z, TABLE_SURFACE_Z, create_cube, create_table
from .robot import load_franka


@dataclass(frozen=True, slots=True)
class SceneState:
    """Body IDs and initial cube position for the current scene."""

    plane_id: int
    table_id: int
    robot_id: int
    cube_id: int
    cube_initial_position: tuple[float, float, float]


class World:
    """Own a PyBullet client and assemble the minimal Phase 1 scene."""

    def __init__(self, gui: bool = False, time_step: float = 1.0 / 240.0) -> None:
        self.gui = gui
        self.time_step = time_step
        connection_mode = pybullet.GUI if gui else pybullet.DIRECT
        self.client_id = pybullet.connect(connection_mode)
        if self.client_id < 0:
            raise RuntimeError("Could not connect to PyBullet")

        self.camera = FixedCamera()
        self.scene: SceneState | None = None

    @property
    def is_connected(self) -> bool:
        return bool(pybullet.isConnected(self.client_id))

    def reset(self, seed: int = 0) -> SceneState:
        """Reset and deterministically assemble the minimal tabletop scene."""
        if not self.is_connected:
            raise RuntimeError("PyBullet client is disconnected")

        rng = np.random.default_rng(seed)
        pybullet.resetSimulation(physicsClientId=self.client_id)
        pybullet.setAdditionalSearchPath(
            pybullet_data.getDataPath(),
            physicsClientId=self.client_id,
        )
        pybullet.setRealTimeSimulation(0, physicsClientId=self.client_id)
        pybullet.setTimeStep(self.time_step, physicsClientId=self.client_id)
        pybullet.setGravity(0.0, 0.0, -9.81, physicsClientId=self.client_id)
        pybullet.setPhysicsEngineParameter(
            deterministicOverlappingPairs=1,
            physicsClientId=self.client_id,
        )

        plane_id = pybullet.loadURDF(
            "plane.urdf",
            basePosition=(0.0, 0.0, GROUND_Z),
            useFixedBase=True,
            physicsClientId=self.client_id,
        )
        table_id = create_table(self.client_id)
        robot_id = load_franka(self.client_id)

        cube_position = (
            float(rng.uniform(0.45, 0.60)),
            float(rng.uniform(-0.12, 0.12)),
            TABLE_SURFACE_Z + CUBE_HALF_EXTENT,
        )
        cube_id = create_cube(self.client_id, cube_position)

        self.scene = SceneState(
            plane_id=plane_id,
            table_id=table_id,
            robot_id=robot_id,
            cube_id=cube_id,
            cube_initial_position=cube_position,
        )
        if self.gui:
            self.camera.configure_debug_view(self.client_id)
        return self.scene

    def get_cube_pose(
        self,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        """Return the cube's current world-frame position and orientation."""
        scene = self._require_scene()
        position, orientation = pybullet.getBasePositionAndOrientation(
            scene.cube_id,
            physicsClientId=self.client_id,
        )
        return tuple(position), tuple(orientation)

    def render(self) -> tuple[int, int, object, object, object]:
        """Render the fixed external camera without interpreting the image."""
        self._require_scene()
        return self.camera.render(self.client_id)

    def step(self) -> None:
        pybullet.stepSimulation(physicsClientId=self.client_id)

    def disconnect(self) -> None:
        if self.is_connected:
            pybullet.disconnect(physicsClientId=self.client_id)
        self.scene = None

    def _require_scene(self) -> SceneState:
        if self.scene is None:
            raise RuntimeError("World must be reset before accessing the scene")
        return self.scene

    def __enter__(self) -> "World":
        return self

    def __exit__(self, *_: object) -> None:
        self.disconnect()
