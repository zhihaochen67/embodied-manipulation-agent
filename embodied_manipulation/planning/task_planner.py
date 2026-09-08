"""Deterministic geometry planner for Phase 8 vision grounding."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from embodied_manipulation.control.pick import (
    GRASP_TARGET_Z_OFFSET,
    LIFT_DISTANCE,
    PREGRASP_CLEARANCE,
)
from embodied_manipulation.control.pick_place import (
    RELEASE_BOTTOM_CLEARANCE,
    TRANSPORT_WALL_CLEARANCE,
)
from embodied_manipulation.language import ObjectRef, Task
from embodied_manipulation.perception import LocalizationResult
from embodied_manipulation.simulation.objects import (
    CUBE_HALF_EXTENT,
    TRAY_FLOOR_THICKNESS,
    TRAY_WALL_HEIGHT,
)

from .primitives import ManipulationPlan, PlanStep, Position, PrimitiveKind


class PlanningError(RuntimeError):
    """Raised when grounded task semantics cannot produce a Phase 8 plan."""


class UnsupportedTaskError(PlanningError):
    """Raised for actions intentionally deferred beyond Phase 8."""


@dataclass(frozen=True, slots=True)
class PlannerConfig:
    """Known primitive geometry used without simulator pose queries."""

    cube_half_extent: float = CUBE_HALF_EXTENT
    tray_floor_thickness: float = TRAY_FLOOR_THICKNESS
    tray_wall_height: float = TRAY_WALL_HEIGHT
    pregrasp_clearance: float = PREGRASP_CLEARANCE
    grasp_target_z_offset: float = GRASP_TARGET_Z_OFFSET
    lift_distance: float = LIFT_DISTANCE
    transport_wall_clearance: float = TRANSPORT_WALL_CLEARANCE
    release_bottom_clearance: float = RELEASE_BOTTOM_CLEARANCE

    def __post_init__(self) -> None:
        for value, name in (
            (self.cube_half_extent, "cube_half_extent"),
            (self.tray_floor_thickness, "tray_floor_thickness"),
            (self.tray_wall_height, "tray_wall_height"),
            (self.pregrasp_clearance, "pregrasp_clearance"),
            (self.lift_distance, "lift_distance"),
            (self.transport_wall_clearance, "transport_wall_clearance"),
            (self.release_bottom_clearance, "release_bottom_clearance"),
        ):
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if not isfinite(self.grasp_target_z_offset):
            raise ValueError("grasp_target_z_offset must be finite")


class TaskPlanner:
    """Convert vision localizations into the fixed Phase 8 motion geometry."""

    def __init__(self, config: PlannerConfig | None = None) -> None:
        self.config = config or PlannerConfig()

    def plan(
        self,
        task: Task,
        source_detection: LocalizationResult,
        target_detection: LocalizationResult | None,
    ) -> ManipulationPlan:
        """Plan one pick-and-place directly from vision-derived coordinates."""
        if task.action != "pick_and_place":
            raise UnsupportedTaskError(
                "Phase 8 executes pick-and-place tasks only; pick-only execution "
                "is intentionally deferred"
            )
        if task.target is None or target_detection is None:
            raise PlanningError("Pick-and-place requires a grounded target")
        self._validate_detection(task.source, source_detection, "source", "cube")
        self._validate_detection(task.target, target_detection, "target", "tray")

        source = source_detection.world_position
        target = target_detection.world_position
        config = self.config
        grasp = (
            source[0],
            source[1],
            source[2] + config.grasp_target_z_offset,
        )
        # Nominal top-down grasp model: the cube remains on the planned grasp
        # axis. This is geometry-derived and is never replaced by a measured
        # post-grasp cube pose.
        cube_to_ee = tuple(
            cube - ee for cube, ee in zip(source, grasp, strict=True)
        )
        pregrasp = (
            source[0],
            source[1],
            source[2] + config.pregrasp_clearance,
        )
        lift = (grasp[0], grasp[1], grasp[2] + config.lift_distance)

        tray_floor_top = target[2] + config.tray_floor_thickness / 2.0
        tray_wall_top = tray_floor_top + config.tray_wall_height
        predicted_lifted_cube_z = lift[2] + cube_to_ee[2]
        transport_cube_z = max(
            predicted_lifted_cube_z,
            tray_wall_top
            + config.cube_half_extent
            + config.transport_wall_clearance,
        )
        transport_ee_z = transport_cube_z - cube_to_ee[2]
        safe_transport = (lift[0], lift[1], transport_ee_z)
        above_target = (
            target[0] - cube_to_ee[0],
            target[1] - cube_to_ee[1],
            transport_ee_z,
        )
        expected_release_cube_center: Position = (
            target[0],
            target[1],
            tray_floor_top
            + config.cube_half_extent
            + config.release_bottom_clearance,
        )
        release = tuple(
            cube - offset
            for cube, offset in zip(
                expected_release_cube_center,
                cube_to_ee,
                strict=True,
            )
        )
        retreat = above_target

        steps = (
            PlanStep(PrimitiveKind.OPEN_GRIPPER),
            PlanStep(PrimitiveKind.MOVE_TO_PREGRASP, pregrasp),
            PlanStep(PrimitiveKind.APPROACH, grasp),
            PlanStep(PrimitiveKind.CLOSE_GRIPPER),
            PlanStep(PrimitiveKind.LIFT, lift),
            PlanStep(PrimitiveKind.RAISE_FOR_TRANSPORT, safe_transport),
            PlanStep(PrimitiveKind.TRANSPORT, above_target),
            PlanStep(PrimitiveKind.DESCEND_TO_RELEASE, release),
            PlanStep(PrimitiveKind.RELEASE),
            PlanStep(PrimitiveKind.RETREAT, retreat),
        )
        return ManipulationPlan(
            task=task,
            source_estimated_center=source,
            target_estimated_center=target,
            nominal_cube_to_ee_offset=cube_to_ee,
            pregrasp_target=pregrasp,
            grasp_target=grasp,
            lift_target=lift,
            safe_transport_target=safe_transport,
            above_target=above_target,
            release_target=release,
            retreat_target=retreat,
            steps=steps,
        )

    @staticmethod
    def _validate_detection(
        expected: ObjectRef,
        detection: LocalizationResult,
        role: str,
        object_type: str,
    ) -> None:
        if detection.source != "vision":
            raise PlanningError(f"{role} detection must come from vision")
        if detection.object_ref != expected:
            raise PlanningError(f"{role} detection does not match parsed semantics")
        if detection.object_ref.object_type != object_type:
            raise PlanningError(f"{role} must be a {object_type}")
        if len(detection.world_position) != 3 or not all(
            isfinite(value) for value in detection.world_position
        ):
            raise PlanningError(f"{role} position must contain three finite values")
