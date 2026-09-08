"""Small typed representation of a deterministic manipulation plan."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite

from embodied_manipulation.language import Task

Position = tuple[float, float, float]


class PrimitiveKind(str, Enum):
    """Phase 8's fixed physical primitives, in execution order."""

    OPEN_GRIPPER = "open_gripper"
    MOVE_TO_PREGRASP = "move_to_pregrasp"
    APPROACH = "approach"
    CLOSE_GRIPPER = "close_gripper"
    LIFT = "lift"
    RAISE_FOR_TRANSPORT = "raise_for_transport"
    TRANSPORT = "transport"
    DESCEND_TO_RELEASE = "descend_to_release"
    RELEASE = "release"
    RETREAT = "retreat"


@dataclass(frozen=True, slots=True)
class PlanStep:
    """One typed primitive with an optional Cartesian end-effector target."""

    primitive: PrimitiveKind
    target_position: Position | None = None

    def __post_init__(self) -> None:
        if self.target_position is not None:
            _validate_position(self.target_position, "target_position")


@dataclass(frozen=True, slots=True)
class ManipulationPlan:
    """Vision-grounded Cartesian targets for one open-loop pick-and-place."""

    task: Task
    source_estimated_center: Position
    target_estimated_center: Position
    nominal_cube_to_ee_offset: Position
    pregrasp_target: Position
    grasp_target: Position
    lift_target: Position
    safe_transport_target: Position
    above_target: Position
    release_target: Position
    retreat_target: Position
    steps: tuple[PlanStep, ...]

    def __post_init__(self) -> None:
        for value, name in (
            (self.source_estimated_center, "source_estimated_center"),
            (self.target_estimated_center, "target_estimated_center"),
            (self.nominal_cube_to_ee_offset, "nominal_cube_to_ee_offset"),
            (self.pregrasp_target, "pregrasp_target"),
            (self.grasp_target, "grasp_target"),
            (self.lift_target, "lift_target"),
            (self.safe_transport_target, "safe_transport_target"),
            (self.above_target, "above_target"),
            (self.release_target, "release_target"),
            (self.retreat_target, "retreat_target"),
        ):
            _validate_position(value, name)
        if not self.steps:
            raise ValueError("A manipulation plan must contain at least one step")


def _validate_position(position: Position, name: str) -> None:
    if len(position) != 3 or not all(isfinite(value) for value in position):
        raise ValueError(f"{name} must contain exactly three finite values")
