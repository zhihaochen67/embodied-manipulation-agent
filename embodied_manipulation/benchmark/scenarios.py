"""Deterministic Phase 5 task and scenario generation."""

from dataclasses import dataclass
from math import hypot
from typing import TypeAlias

import numpy as np

from embodied_manipulation.language import ObjectRef, Task
from embodied_manipulation.simulation.objects import (
    CUBE_HALF_EXTENT,
    TABLE_SURFACE_Z,
    TRAY_FLOOR_THICKNESS,
    TRAY_INNER_HALF_EXTENTS,
    TRAY_WALL_HEIGHT,
    TRAY_WALL_THICKNESS,
)

COLOR_NAMES = ("red", "blue", "yellow", "green")
INSTRUCTION_TEMPLATES = (
    "Put the {source_color} cube in the {target_color} tray.",
    "Place the {source_color} cube into the {target_color} tray.",
    "Move the {source_color} cube to the {target_color} tray.",
)

# These conservative regions stay near poses already exercised by Phases 2–4.
SOURCE_X_RANGE = (0.48, 0.57)
SOURCE_Y_RANGE = (-0.10, 0.02)
TARGET_X_RANGE = (0.49, 0.55)
TARGET_Y_RANGE = (0.25, 0.31)
REACHABLE_WORKSPACE_BOUNDS = (0.40, 0.75, -0.20, 0.40)
MINIMUM_SOURCE_TARGET_CENTER_DISTANCE = 0.22

DISTRACTOR_CENTERS = ((0.68, -0.12), (0.68, 0.10))
DISTRACTOR_X_JITTER = 0.01
DISTRACTOR_Y_JITTER = 0.015


@dataclass(frozen=True, slots=True)
class CubeSpec:
    """Simulator-independent metadata for one cube."""

    object_id: str
    color: str
    initial_position: tuple[float, float, float]
    role: str
    size: float = 2.0 * CUBE_HALF_EXTENT
    object_type: str = "cube"

    @property
    def footprint_bounds(self) -> tuple[float, float, float, float]:
        half = self.size / 2.0
        return (
            self.initial_position[0] - half,
            self.initial_position[0] + half,
            self.initial_position[1] - half,
            self.initial_position[1] + half,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "object_id": self.object_id,
            "type": self.object_type,
            "color": self.color,
            "initial_position": list(self.initial_position),
            "role": self.role,
            "size": self.size,
        }


@dataclass(frozen=True, slots=True)
class TraySpec:
    """Simulator-independent metadata for one primitive tray."""

    object_id: str
    color: str
    initial_position: tuple[float, float, float]
    role: str
    inner_size: tuple[float, float] = (
        2.0 * TRAY_INNER_HALF_EXTENTS[0],
        2.0 * TRAY_INNER_HALF_EXTENTS[1],
    )
    floor_thickness: float = TRAY_FLOOR_THICKNESS
    wall_height: float = TRAY_WALL_HEIGHT
    wall_thickness: float = TRAY_WALL_THICKNESS
    object_type: str = "tray"

    @property
    def outer_size(self) -> tuple[float, float]:
        return (
            self.inner_size[0] + 2.0 * self.wall_thickness,
            self.inner_size[1] + 2.0 * self.wall_thickness,
        )

    @property
    def footprint_bounds(self) -> tuple[float, float, float, float]:
        return (
            self.initial_position[0] - self.outer_size[0] / 2.0,
            self.initial_position[0] + self.outer_size[0] / 2.0,
            self.initial_position[1] - self.outer_size[1] / 2.0,
            self.initial_position[1] + self.outer_size[1] / 2.0,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "object_id": self.object_id,
            "type": self.object_type,
            "color": self.color,
            "initial_position": list(self.initial_position),
            "role": self.role,
            "inner_size": list(self.inner_size),
            "floor_thickness": self.floor_thickness,
            "wall_height": self.wall_height,
            "wall_thickness": self.wall_thickness,
        }


SceneObjectSpec: TypeAlias = CubeSpec | TraySpec


@dataclass(frozen=True, slots=True)
class Scenario:
    """One reproducible structured pick-and-place scene."""

    seed: int
    difficulty: str
    task: Task
    source: CubeSpec
    target: TraySpec
    distractors: tuple[SceneObjectSpec, ...] = ()

    @property
    def objects(self) -> tuple[SceneObjectSpec, ...]:
        return (self.source, self.target, *self.distractors)

    @property
    def source_target_center_distance(self) -> float:
        return hypot(
            self.source.initial_position[0] - self.target.initial_position[0],
            self.source.initial_position[1] - self.target.initial_position[1],
        )

    @property
    def source_target_clearance(self) -> float:
        return _footprint_distance(
            self.source.footprint_bounds,
            self.target.footprint_bounds,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "difficulty": self.difficulty,
            "task": self.task.to_dict(),
            "source": self.source.to_dict(),
            "target": self.target.to_dict(),
            "distractors": [item.to_dict() for item in self.distractors],
        }


def generate_scenario(
    seed: int,
    difficulty: str = "basic",
    *,
    distractor_count: int = 0,
) -> Scenario:
    """Generate one valid scenario directly from a local seeded RNG."""
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if difficulty != "basic":
        raise ValueError("Phase 5 supports only difficulty='basic'")
    if (
        not isinstance(distractor_count, int)
        or isinstance(distractor_count, bool)
        or not 0 <= distractor_count <= 2
    ):
        raise ValueError("distractor_count must be 0, 1, or 2")

    rng = np.random.default_rng(seed)
    source_position = (
        float(rng.uniform(*SOURCE_X_RANGE)),
        float(rng.uniform(*SOURCE_Y_RANGE)),
        TABLE_SURFACE_Z + CUBE_HALF_EXTENT,
    )
    target_position = (
        float(rng.uniform(*TARGET_X_RANGE)),
        float(rng.uniform(*TARGET_Y_RANGE)),
        TABLE_SURFACE_Z + TRAY_FLOOR_THICKNESS / 2.0,
    )
    colors = tuple(str(color) for color in rng.permutation(COLOR_NAMES))
    source = CubeSpec(
        object_id="source_cube",
        color=colors[0],
        initial_position=source_position,
        role="source",
    )
    target = TraySpec(
        object_id="target_tray",
        color=colors[1],
        initial_position=target_position,
        role="destination",
    )
    distractors = tuple(
        CubeSpec(
            object_id=f"distractor_cube_{index + 1}",
            color=colors[index + 2],
            initial_position=(
                DISTRACTOR_CENTERS[index][0]
                + float(rng.uniform(-DISTRACTOR_X_JITTER, DISTRACTOR_X_JITTER)),
                DISTRACTOR_CENTERS[index][1]
                + float(rng.uniform(-DISTRACTOR_Y_JITTER, DISTRACTOR_Y_JITTER)),
                TABLE_SURFACE_Z + CUBE_HALF_EXTENT,
            ),
            role="distractor",
        )
        for index in range(distractor_count)
    )
    instruction = INSTRUCTION_TEMPLATES[
        int(rng.integers(0, len(INSTRUCTION_TEMPLATES)))
    ].format(
        source_color=source.color,
        target_color=target.color,
    )
    task = Task(
        action="pick_and_place",
        source=ObjectRef(source.object_id, source.object_type, source.color),
        target=ObjectRef(target.object_id, target.object_type, target.color),
        instruction=instruction,
    )
    scenario = Scenario(
        seed=seed,
        difficulty=difficulty,
        task=task,
        source=source,
        target=target,
        distractors=distractors,
    )
    validate_scenario(scenario)
    return scenario


def validate_scenario(scenario: Scenario) -> None:
    """Raise if a scenario violates the explicit Phase 5 geometry rules."""
    if scenario.source_target_center_distance < (
        MINIMUM_SOURCE_TARGET_CENTER_DISTANCE
    ):
        raise ValueError("source and target are too close")
    if scenario.source.size >= min(scenario.target.inner_size):
        raise ValueError("tray interior must be larger than the source cube")

    identities = [item.object_id for item in scenario.objects]
    if len(set(identities)) != len(identities):
        raise ValueError("scenario object IDs must be unique")
    semantics = [(item.object_type, item.color) for item in scenario.objects]
    if len(set(semantics)) != len(semantics):
        raise ValueError("scenario type/color descriptions must be unique")
    if any(item.color not in COLOR_NAMES for item in scenario.objects):
        raise ValueError("scenario contains an unsupported color")
    if scenario.task.action != "pick_and_place":
        raise ValueError("Phase 5 supports only pick_and_place tasks")
    if scenario.task.source != ObjectRef(
        scenario.source.object_id,
        scenario.source.object_type,
        scenario.source.color,
    ):
        raise ValueError("task source does not match source metadata")
    if scenario.task.target != ObjectRef(
        scenario.target.object_id,
        scenario.target.object_type,
        scenario.target.color,
    ):
        raise ValueError("task target does not match target metadata")

    for item in scenario.objects:
        if not _inside_workspace(item.footprint_bounds):
            raise ValueError(f"{item.object_id} is outside the reachable workspace")
        expected_z = (
            TABLE_SURFACE_Z + item.size / 2.0
            if isinstance(item, CubeSpec)
            else TABLE_SURFACE_Z + item.floor_thickness / 2.0
        )
        if item.initial_position[2] != expected_z:
            raise ValueError(f"{item.object_id} is not on the tabletop")

    for index, first in enumerate(scenario.objects):
        for second in scenario.objects[index + 1 :]:
            if _footprints_overlap(
                first.footprint_bounds,
                second.footprint_bounds,
            ):
                raise ValueError(
                    f"{first.object_id} overlaps {second.object_id}"
                )


def _inside_workspace(bounds: tuple[float, float, float, float]) -> bool:
    min_x, max_x, min_y, max_y = REACHABLE_WORKSPACE_BOUNDS
    return (
        bounds[0] >= min_x
        and bounds[1] <= max_x
        and bounds[2] >= min_y
        and bounds[3] <= max_y
    )


def _footprints_overlap(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> bool:
    return not (
        first[1] <= second[0]
        or second[1] <= first[0]
        or first[3] <= second[2]
        or second[3] <= first[2]
    )


def _footprint_distance(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    x_gap = max(first[0] - second[1], second[0] - first[1], 0.0)
    y_gap = max(first[2] - second[3], second[2] - first[3], 0.0)
    return hypot(x_gap, y_gap)
