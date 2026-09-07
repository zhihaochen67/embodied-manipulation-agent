"""Minimal deterministic PyBullet world for the current manipulation phases."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pybullet
import pybullet_data

from .camera import FixedCamera
from .objects import (
    CUBE_HALF_EXTENT,
    COLOR_RGBA,
    GROUND_Z,
    TABLE_SURFACE_Z,
    Tray,
    create_cube,
    create_table,
    create_tray,
)
from .robot import load_franka

if TYPE_CHECKING:
    from embodied_manipulation.benchmark.scenarios import Scenario


@dataclass(frozen=True, slots=True)
class SceneState:
    """Body IDs and minimal geometry for the current scene."""

    plane_id: int
    table_id: int
    robot_id: int
    cube_id: int
    cube_initial_position: tuple[float, float, float]
    tray: Tray | None = None
    scenario: Scenario | None = None
    object_body_ids: dict[str, tuple[int, ...]] = field(default_factory=dict)


class World:
    """Own a PyBullet client and assemble the deterministic tabletop scene."""

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

    def reset(self, seed: int = 0, *, include_tray: bool = False) -> SceneState:
        """Reset and deterministically assemble the minimal tabletop scene."""
        rng = np.random.default_rng(seed)
        plane_id, table_id, robot_id = self._reset_base_scene()

        cube_position = (
            float(rng.uniform(0.45, 0.60)),
            float(rng.uniform(-0.12, 0.12)),
            TABLE_SURFACE_Z + CUBE_HALF_EXTENT,
        )
        cube_id = create_cube(self.client_id, cube_position)
        tray = create_tray(self.client_id) if include_tray else None

        self.scene = SceneState(
            plane_id=plane_id,
            table_id=table_id,
            robot_id=robot_id,
            cube_id=cube_id,
            cube_initial_position=cube_position,
            tray=tray,
        )
        if self.gui:
            self.camera.configure_debug_view(self.client_id)
        return self.scene

    def reset_from_scenario(self, scenario: Scenario) -> SceneState:
        """Reset and instantiate one structured scenario deterministically."""
        from embodied_manipulation.benchmark.scenarios import (
            CubeSpec,
            Scenario,
            TraySpec,
            validate_scenario,
        )

        if not isinstance(scenario, Scenario):
            raise TypeError("scenario must be a Scenario")
        validate_scenario(scenario)
        plane_id, table_id, robot_id = self._reset_base_scene()

        source = scenario.source
        cube_id = create_cube(
            self.client_id,
            source.initial_position,
            color=COLOR_RGBA[source.color],
            half_extent=source.size / 2.0,
        )
        target = scenario.target
        tray = create_tray(
            self.client_id,
            center=target.initial_position[:2],
            inner_half_extents=(
                target.inner_size[0] / 2.0,
                target.inner_size[1] / 2.0,
            ),
            floor_thickness=target.floor_thickness,
            wall_height=target.wall_height,
            wall_thickness=target.wall_thickness,
            color=COLOR_RGBA[target.color],
        )
        object_body_ids = {
            source.object_id: (cube_id,),
            target.object_id: tray.body_ids,
        }
        for distractor in scenario.distractors:
            if isinstance(distractor, CubeSpec):
                body_id = create_cube(
                    self.client_id,
                    distractor.initial_position,
                    color=COLOR_RGBA[distractor.color],
                    half_extent=distractor.size / 2.0,
                )
                object_body_ids[distractor.object_id] = (body_id,)
            elif isinstance(distractor, TraySpec):
                distractor_tray = create_tray(
                    self.client_id,
                    center=distractor.initial_position[:2],
                    inner_half_extents=(
                        distractor.inner_size[0] / 2.0,
                        distractor.inner_size[1] / 2.0,
                    ),
                    floor_thickness=distractor.floor_thickness,
                    wall_height=distractor.wall_height,
                    wall_thickness=distractor.wall_thickness,
                    color=COLOR_RGBA[distractor.color],
                )
                object_body_ids[distractor.object_id] = distractor_tray.body_ids
            else:
                raise TypeError("Unsupported scenario object specification")

        self.scene = SceneState(
            plane_id=plane_id,
            table_id=table_id,
            robot_id=robot_id,
            cube_id=cube_id,
            cube_initial_position=source.initial_position,
            tray=tray,
            scenario=scenario,
            object_body_ids=object_body_ids,
        )
        if self.gui:
            self.camera.configure_debug_view(self.client_id)
        return self.scene

    def get_object_body_ids(self, object_id: str) -> tuple[int, ...]:
        """Return the PyBullet bodies spawned for one scenario object ID."""
        scene = self._require_scene()
        try:
            return scene.object_body_ids[object_id]
        except KeyError as error:
            raise KeyError(f"Unknown scenario object ID: {object_id}") from error

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

    def get_tray_pose(
        self,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        """Return the tray floor body's current world pose."""
        scene = self._require_scene()
        if scene.tray is None:
            raise RuntimeError("Current scene does not include a tray")
        position, orientation = pybullet.getBasePositionAndOrientation(
            scene.tray.floor_id,
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

    def _reset_base_scene(self) -> tuple[int, int, int]:
        if not self.is_connected:
            raise RuntimeError("PyBullet client is disconnected")
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
        return plane_id, table_id, robot_id

    def __enter__(self) -> "World":
        return self

    def __exit__(self, *_: object) -> None:
        self.disconnect()
