"""Pre-registered, method-independent Phase 14B benchmark perturbations."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from typing import Any, Iterable, Literal

from embodied_manipulation.planning import ManipulationPlan, PrimitiveKind, TaskPlanner

PerturbationStage = Literal["grasp", "placement"]


@dataclass(frozen=True, slots=True)
class BenchmarkPerturbation:
    """One frozen geometric intervention on the first manipulation attempt."""

    name: str
    stage: PerturbationStage
    axis: Literal["x"]
    offset_m: float
    first_attempt_only: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Perturbation name must be non-empty")
        if self.stage not in {"grasp", "placement"}:
            raise ValueError("Perturbation stage must be grasp or placement")
        if self.axis != "x":
            raise ValueError("Phase 14B supports only X-axis perturbations")
        if not isfinite(self.offset_m):
            raise ValueError("Perturbation offset must be finite")
        if not self.first_attempt_only:
            raise ValueError("Phase 14B perturbations must be first-attempt-only")

    @property
    def grasp_xy_offset(self) -> tuple[float, float]:
        return (self.offset_m, 0.0) if self.stage == "grasp" else (0.0, 0.0)

    @property
    def placement_xy_offset(self) -> tuple[float, float]:
        return (
            (self.offset_m, 0.0)
            if self.stage == "placement"
            else (0.0, 0.0)
        )

    def to_dict(self) -> dict[str, str | float | bool]:
        return {
            "name": self.name,
            "stage": self.stage,
            "axis": self.axis,
            "offset_m": self.offset_m,
            "first_attempt_only": self.first_attempt_only,
        }

    def apply_to_plan(self, plan: ManipulationPlan) -> ManipulationPlan:
        """Return a shifted immutable plan without mutating scenario metadata."""
        grasp_offset = self.grasp_xy_offset
        placement_offset = self.placement_xy_offset

        def shifted(
            position: tuple[float, float, float],
            offset: tuple[float, float],
        ) -> tuple[float, float, float]:
            return position[0] + offset[0], position[1] + offset[1], position[2]

        grasp_primitives = {
            PrimitiveKind.MOVE_TO_PREGRASP,
            PrimitiveKind.APPROACH,
            PrimitiveKind.LIFT,
            PrimitiveKind.RAISE_FOR_TRANSPORT,
        }
        placement_primitives = {
            PrimitiveKind.TRANSPORT,
            PrimitiveKind.DESCEND_TO_RELEASE,
            PrimitiveKind.RETREAT,
        }
        steps = []
        for step in plan.steps:
            target = step.target_position
            if target is not None and step.primitive in grasp_primitives:
                target = shifted(target, grasp_offset)
            if target is not None and step.primitive in placement_primitives:
                target = shifted(target, placement_offset)
            steps.append(replace(step, target_position=target))
        return replace(
            plan,
            pregrasp_target=shifted(plan.pregrasp_target, grasp_offset),
            grasp_target=shifted(plan.grasp_target, grasp_offset),
            lift_target=shifted(plan.lift_target, grasp_offset),
            safe_transport_target=shifted(plan.safe_transport_target, grasp_offset),
            above_target=shifted(plan.above_target, placement_offset),
            release_target=shifted(plan.release_target, placement_offset),
            retreat_target=shifted(plan.retreat_target, placement_offset),
            steps=tuple(steps),
        )


PERTURBATION_REGISTRY = (
    BenchmarkPerturbation(
        name="grasp_shift_x_120mm",
        stage="grasp",
        axis="x",
        offset_m=0.12,
    ),
    BenchmarkPerturbation(
        name="placement_shift_x_180mm",
        stage="placement",
        axis="x",
        offset_m=0.18,
    ),
)
PERTURBATION_NAMES = tuple(item.name for item in PERTURBATION_REGISTRY)
_PERTURBATIONS_BY_NAME = {item.name: item for item in PERTURBATION_REGISTRY}


def ordered_perturbations(
    perturbations: Iterable[str | BenchmarkPerturbation] | None,
) -> tuple[BenchmarkPerturbation, ...]:
    """Validate conditions and return them in frozen registry order."""
    if perturbations is None:
        return ()
    requested = tuple(
        _PERTURBATIONS_BY_NAME[item] if isinstance(item, str) else item
        for item in perturbations
    )
    names = tuple(item.name for item in requested)
    if not names:
        raise ValueError("At least one perturbation is required")
    if len(set(names)) != len(names):
        raise ValueError("Benchmark perturbations must not be repeated")
    unknown = set(names).difference(PERTURBATION_NAMES)
    if unknown:
        raise ValueError(f"Unknown benchmark perturbations: {sorted(unknown)}")
    if any(_PERTURBATIONS_BY_NAME[item.name] != item for item in requested):
        raise ValueError("Benchmark perturbations must match the frozen registry")
    requested_names = set(names)
    return tuple(
        item for item in PERTURBATION_REGISTRY if item.name in requested_names
    )


class FirstAttemptPerturbingPlanner:
    """Shift only the first successful plan; recovery replans stay nominal."""

    def __init__(
        self,
        perturbation: BenchmarkPerturbation,
        planner: TaskPlanner | None = None,
    ) -> None:
        self._perturbation = perturbation
        self._planner = planner or TaskPlanner()
        self._first_plan_pending = True

    def plan(self, *args: Any, **kwargs: Any) -> ManipulationPlan:
        plan = self._planner.plan(*args, **kwargs)
        if not self._first_plan_pending:
            return plan
        self._first_plan_pending = False
        return self._perturbation.apply_to_plan(plan)
