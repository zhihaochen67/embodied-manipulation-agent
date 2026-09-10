"""Stable, analysis-friendly metrics for the Phase 11A benchmark."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import fsum
from typing import Any, Iterable

BENCHMARK_VERSION = "phase14d-perturb-v3"
METHODS = (
    "oracle_scripted",
    "vision_open_loop",
    "vision_closed_loop",
    "vision_recovery",
)
VISION_METHODS = frozenset(METHODS[1:])
VERIFICATION_METHODS = frozenset(METHODS[2:])

# Keep this order stable: it is the public, flat episodes.csv schema.
CSV_FIELDS = (
    "benchmark_version",
    "method",
    "seed",
    "difficulty",
    "distractor_count",
    "instruction",
    "source_color",
    "source_type",
    "target_color",
    "target_type",
    "perturbation_name",
    "perturbation_stage",
    "perturbation_axis",
    "perturbation_offset_m",
    "perturbation_first_attempt_only",
    "agent_success",
    "task_success",
    "grasp_success",
    "placement_success",
    "simulation_steps",
    "observation_count",
    "recovery_activated",
    "recovery_attempts",
    "recovery_success",
    "recovery_stage",
    "recovered",
    "recovery_entry_type",
    "initial_execution_failure_code",
    "initial_execution_failure_stage",
    "initial_execution_failure_reason",
    "initial_failure_reason",
    "recovery_diagnosis",
    "final_failure_reason",
    "retry_execution_failure_code",
    "retry_execution_failure_stage",
    "retry_execution_failure_reason",
    "initial_failed_reach_failure_code",
    "initial_failed_reach_ik_position_residual",
    "initial_failed_reach_required_position_tolerance",
    "initial_gripper_termination_reason",
    "initial_gripper_timeout_diagnostic",
    "retry_failed_reach_failure_code",
    "retry_failed_reach_ik_position_residual",
    "retry_failed_reach_required_position_tolerance",
    "retry_gripper_termination_reason",
    "retry_gripper_timeout_diagnostic",
    "recovery_final_verification_result",
    "grasp_verification_result",
    "placement_verification_result",
    "initial_grasp_verification_result",
    "initial_placement_verification_result",
    "retry_grasp_verification_result",
    "retry_placement_verification_result",
    "grasp_verification_failure_count",
    "placement_verification_failure_count",
    "verification_failure_count",
    "source_grounding_success",
    "target_grounding_success",
    "source_position_error",
    "target_position_error",
    "failure_stage",
    "failure_reason",
    "wall_clock_seconds",
    "infrastructure_error",
)

CONDITION_SUMMARY_FIELDS = (
    "condition",
    "method",
    "episodes",
    "task_success_count",
    "grasp_success_count",
    "placement_success_count",
    "task_success_rate",
    "grasp_success_rate",
    "placement_success_rate",
    "verification_failure_count",
    "grasp_verification_failure_count",
    "placement_verification_failure_count",
    "recovery_activation_count",
    "recovery_attempt_count",
    "recovery_success_count",
    "recovery_failure_count",
    "pre_verification_recovery_count",
    "post_verification_recovery_count",
    "mean_simulation_steps",
    "median_simulation_steps",
    "std_simulation_steps",
    "min_simulation_steps",
    "max_simulation_steps",
    "mean_runtime_seconds",
    "total_runtime_seconds",
)

PAIRED_SUCCESS_FIELDS = (
    "condition",
    "seed",
    "oracle_task_success",
    "oracle_grasp_success",
    "oracle_placement_success",
    "vision_open_task_success",
    "vision_open_grasp_success",
    "vision_open_placement_success",
    "vision_closed_task_success",
    "vision_closed_grasp_success",
    "vision_closed_placement_success",
    "vision_closed_verification_failure",
    "vision_closed_verification_failure_count",
    "vision_closed_failure_stage",
    "vision_recovery_task_success",
    "vision_recovery_grasp_success",
    "vision_recovery_placement_success",
    "vision_recovery_verification_failure_occurred",
    "vision_recovery_verification_failure_count",
    "vision_recovery_failure_stage",
    "vision_recovery_recovery_activated",
    "vision_recovery_recovery_entry_type",
    "vision_recovery_recovered",
    "vision_recovery_recovery_success",
    "vision_recovery_final_success",
)


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """One JSON-serializable episode with method-independent semantics.

    ``task_success`` and ``placement_success`` come from the common objective
    evaluator. ``grasp_success`` means the cube was physically lifted/held at
    least once. Visual gate outcomes remain separate from those physical facts.
    Recovery-only fields are null for other methods. Low-level ``initial_*``
    reach/gripper diagnostics describe the sole attempt for non-recovery methods
    and the first attempt for the recovery method; ``retry_*`` is recovery-only.
    """

    benchmark_version: str
    method: str
    seed: int
    difficulty: str
    distractor_count: int
    instruction: str
    source_color: str
    source_type: str
    target_color: str
    target_type: str
    perturbation_name: str | None
    perturbation_stage: str | None
    perturbation_axis: str | None
    perturbation_offset_m: float | None
    perturbation_first_attempt_only: bool | None
    agent_success: bool | None
    task_success: bool
    grasp_success: bool | None
    placement_success: bool | None
    simulation_steps: int | None
    observation_count: int | None
    recovery_activated: bool | None
    recovery_attempts: int | None
    recovery_success: bool | None
    recovery_stage: str | None
    recovered: bool | None
    recovery_final_verification_result: bool | None
    grasp_verification_result: bool | None
    placement_verification_result: bool | None
    verification_failure_count: int
    source_grounding_success: bool | None
    target_grounding_success: bool | None
    source_position_error: float | None
    target_position_error: float | None
    failure_stage: str | None
    failure_reason: str | None
    wall_clock_seconds: float
    infrastructure_error: str | None = None
    recovery_entry_type: str | None = None
    initial_execution_failure_code: str | None = None
    initial_execution_failure_stage: str | None = None
    initial_execution_failure_reason: str | None = None
    initial_failure_reason: str | None = None
    recovery_diagnosis: str | None = None
    final_failure_reason: str | None = None
    retry_execution_failure_code: str | None = None
    retry_execution_failure_stage: str | None = None
    retry_execution_failure_reason: str | None = None
    initial_failed_reach_failure_code: str | None = None
    initial_failed_reach_ik_position_residual: float | None = None
    initial_failed_reach_required_position_tolerance: float | None = None
    initial_gripper_termination_reason: str | None = None
    initial_gripper_timeout_diagnostic: str | None = None
    retry_failed_reach_failure_code: str | None = None
    retry_failed_reach_ik_position_residual: float | None = None
    retry_failed_reach_required_position_tolerance: float | None = None
    retry_gripper_termination_reason: str | None = None
    retry_gripper_timeout_diagnostic: str | None = None
    initial_grasp_verification_result: bool | None = None
    initial_placement_verification_result: bool | None = None
    retry_grasp_verification_result: bool | None = None
    retry_placement_verification_result: bool | None = None
    grasp_verification_failure_count: int = 0
    placement_verification_failure_count: int = 0

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"Unknown benchmark method: {self.method}")
        if self.benchmark_version != BENCHMARK_VERSION:
            raise ValueError("EpisodeResult benchmark_version is not current")
        if self.verification_failure_count < 0:
            raise ValueError("verification_failure_count must be nonnegative")
        if self.grasp_verification_failure_count < 0:
            raise ValueError("grasp_verification_failure_count must be nonnegative")
        if self.placement_verification_failure_count < 0:
            raise ValueError("placement_verification_failure_count must be nonnegative")
        if self.verification_failure_count != (
            self.grasp_verification_failure_count
            + self.placement_verification_failure_count
        ):
            raise ValueError("Verification failure counts must agree by stage")
        if self.wall_clock_seconds < 0.0:
            raise ValueError("wall_clock_seconds must be nonnegative")
        if self.recovery_entry_type not in {
            None,
            "pre_verification",
            "post_verification",
        }:
            raise ValueError("Unknown recovery_entry_type")
        perturbation_values = (
            self.perturbation_name,
            self.perturbation_stage,
            self.perturbation_axis,
            self.perturbation_offset_m,
            self.perturbation_first_attempt_only,
        )
        if any(value is not None for value in perturbation_values) and any(
            value is None for value in perturbation_values
        ):
            raise ValueError("Perturbation fields must be all populated or all null")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/CSV-safe flat mapping in the stable schema."""
        data = asdict(self)
        return {field: data[field] for field in CSV_FIELDS}

    def logical_dict(self) -> dict[str, Any]:
        """Return reproducible content with wall-clock timing removed."""
        data = self.to_dict()
        data.pop("wall_clock_seconds")
        return data


def summarize_episodes(
    episodes: Iterable[EpisodeResult],
) -> dict[str, dict[str, int | float | None]]:
    """Aggregate metrics with null for every genuinely undefined ratio."""
    grouped = {method: [] for method in METHODS}
    for episode in episodes:
        grouped[episode.method].append(episode)
    return {
        method: _summarize_method(records)
        for method, records in grouped.items()
        if records
    }


def _summarize_method(
    records: list[EpisodeResult],
) -> dict[str, int | float | None]:
    method = records[0].method
    vision = method in VISION_METHODS
    verification = method in VERIFICATION_METHODS
    recovery = method == "vision_recovery"
    return {
        "episodes": len(records),
        "task_success_rate": _rate(item.task_success for item in records),
        "grasp_success_rate": _rate(item.grasp_success for item in records),
        "placement_success_rate": _rate(
            item.placement_success for item in records
        ),
        "mean_simulation_steps": _mean(
            item.simulation_steps for item in records
        ),
        "mean_wall_clock_seconds": _mean(
            item.wall_clock_seconds for item in records
        ),
        "source_grounding_success_rate": (
            _rate(item.source_grounding_success for item in records)
            if vision
            else None
        ),
        "target_grounding_success_rate": (
            _rate(item.target_grounding_success for item in records)
            if vision
            else None
        ),
        "mean_source_position_error": (
            _mean(item.source_position_error for item in records)
            if vision
            else None
        ),
        "mean_target_position_error": (
            _mean(item.target_position_error for item in records)
            if vision
            else None
        ),
        "mean_verification_failure_count": (
            _mean(item.verification_failure_count for item in records)
            if verification
            else None
        ),
        "recovery_activation_rate": (
            _rate(item.recovery_activated for item in records)
            if recovery
            else None
        ),
        "recovery_success_rate": (
            _rate(item.recovery_success for item in records)
            if recovery
            else None
        ),
        "mean_recovery_attempts": (
            _mean(item.recovery_attempts for item in records)
            if recovery
            else None
        ),
    }


def _rate(values: Iterable[bool | None]) -> float | None:
    available = [value for value in values if value is not None]
    if not available:
        return None
    return fsum(1.0 for value in available if value) / len(available)


def _mean(values: Iterable[int | float | None]) -> float | None:
    available = [float(value) for value in values if value is not None]
    if not available:
        return None
    return fsum(available) / len(available)


def validate_episode_matrix(
    episodes: Iterable[EpisodeResult],
    *,
    expected_record_count: int | None = None,
    expected_conditions: Iterable[str | None],
    expected_seeds: Iterable[int],
    expected_methods: Iterable[str] = METHODS,
) -> None:
    """Validate a complete condition/seed/method matrix without simulation."""
    records = tuple(episodes)
    conditions = tuple(expected_conditions)
    seeds = tuple(expected_seeds)
    methods = tuple(expected_methods)
    if expected_record_count is not None and len(records) != expected_record_count:
        raise ValueError(
            "Episode matrix record count mismatch: "
            f"expected {expected_record_count}, got {len(records)}"
        )
    for values, name in (
        (conditions, "conditions"),
        (seeds, "seeds"),
        (methods, "methods"),
    ):
        if len(values) != len(set(values)):
            raise ValueError(f"Expected {name} must not contain duplicates")
    expected = {
        (condition, seed, method)
        for condition in conditions
        for seed in seeds
        for method in methods
    }
    keys = [
        (record.perturbation_name, record.seed, record.method)
        for record in records
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate condition/seed/method episode records")
    actual = set(keys)
    if actual != expected:
        missing = sorted(expected - actual, key=repr)
        unexpected = sorted(actual - expected, key=repr)
        raise ValueError(
            "Episode matrix coverage mismatch: "
            f"expected {len(expected)} records, got {len(records)}; "
            f"missing={missing}; unexpected={unexpected}"
        )
