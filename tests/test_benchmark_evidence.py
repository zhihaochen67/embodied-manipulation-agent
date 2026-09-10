"""Phase 14D-2A benchmark evidence-contract regression tests."""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from embodied_manipulation.benchmark import (
    BENCHMARK_VERSION,
    CONDITION_SUMMARY_FIELDS,
    CSV_FIELDS,
    METHODS,
    PAIRED_SUCCESS_FIELDS,
    PERTURBATION_NAMES,
    PERTURBATION_REGISTRY,
    EpisodeResult,
    generate_scenario,
    validate_episode_matrix,
)
from embodied_manipulation.benchmark import runner as benchmark_runner
from embodied_manipulation.benchmark.metrics import summarize_episodes
from embodied_manipulation.control import GripperResult, ReachResult
from embodied_manipulation.control.open_loop import OpenLoopExecutionResult
from embodied_manipulation.control.pick_place import PlacementEvaluation


EXPECTED_CONDITION_SUMMARY_FIELDS = (
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

EXPECTED_PAIRED_SUCCESS_FIELDS = (
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

PAIRED_METHOD_PREFIXES = {
    "oracle_scripted": "oracle",
    "vision_open_loop": "vision_open",
    "vision_closed_loop": "vision_closed",
    "vision_recovery": "vision_recovery",
}


def _evaluation(success: bool = True) -> PlacementEvaluation:
    return PlacementEvaluation(
        success=success,
        cube_position=(0.50, 0.28, 0.033),
        cube_linear_velocity=(0.0, 0.0, 0.0),
        cube_angular_velocity=(0.0, 0.0, 0.0),
        cube_inside_tray=success,
        cube_near_floor=success,
        cube_released=True,
        stable=True,
        cube_to_tray_center_distance=0.0,
        failure_reason=None if success else "cube_outside_tray",
    )


def _reach_failure(
    code: str,
    *,
    residual: float = 0.005,
    tolerance: float = 0.004,
) -> ReachResult:
    return ReachResult(
        success=False,
        steps=0,
        target_position=(0.55, 0.10, 0.20),
        final_position=(0.50, 0.00, 0.40),
        final_orientation=(1.0, 0.0, 0.0, 0.0),
        position_error=0.05,
        target_joint_positions=(0.0,) * 7,
        final_joint_positions=(0.0,) * 7,
        failure_reason="opaque structured reach failure",
        failure_code=code,
        ik_position_residual=residual,
        required_position_tolerance=tolerance,
        ik_joint_solution_valid=True,
    )


def _timeout_gripper(code: str) -> GripperResult:
    return GripperResult(
        success=False,
        steps=240,
        target_opening_width=0.0,
        final_opening_width=0.05,
        target_finger_positions=(0.0, 0.0),
        final_finger_positions=(0.025, 0.025),
        max_finger_position_error=0.025,
        reached_target=False,
        stalled=False,
        failure_reason="opaque gripper failure",
        termination_reason="timeout",
        timeout_diagnostic=code,
    )


def _execution(
    success: bool,
    *,
    grasp_success: bool,
    placement_success: bool,
    failure_stage: str | None = None,
    failure_reason: str | None = None,
    failed_reach: ReachResult | None = None,
    close_result: GripperResult | None = None,
) -> OpenLoopExecutionResult:
    return OpenLoopExecutionResult(
        success=success,
        grasp_success=grasp_success,
        transport_success=placement_success,
        placement_success=placement_success,
        arm_steps=5,
        gripper_steps=3,
        settling_steps=2,
        total_steps=10,
        constraint_count_before=0,
        constraint_count_after=0,
        transport_finger_targets=(0.02, 0.02) if grasp_success else None,
        close_result=close_result,
        failure_stage=failure_stage,
        failure_reason=failure_reason,
        failed_reach_result=failed_reach,
    )


def _raw_recovery_result(
    *,
    initial_execution: OpenLoopExecutionResult,
    retry_execution: OpenLoopExecutionResult | None,
    trace: object,
    success: bool,
    initial_grasp: bool | None,
    initial_placement: bool | None,
    retry_grasp: bool | None,
    retry_placement: bool | None,
) -> object:
    verification = lambda value: (
        None if value is None else SimpleNamespace(verified=value)
    )
    trace.recovery_grasp_verification = verification(retry_grasp)
    trace.recovery_placement_verification = verification(retry_placement)
    return SimpleNamespace(
        success=success,
        total_steps=20 if retry_execution is not None else 10,
        observation_count=4,
        recovery_activated=True,
        recovery_attempts=1,
        recovery_stage=trace.recovery_stage,
        recovered=success,
        recovery_trace=trace,
        initial_execution_result=initial_execution,
        recovery_execution_result=retry_execution,
        execution_result=retry_execution or initial_execution,
        initial_grasp_verification=verification(initial_grasp),
        initial_placement_verification=verification(initial_placement),
        grasp_verification=verification(retry_grasp),
        placement_verification=verification(retry_placement),
        source_detection=None,
        target_detection=None,
        initial_failure_reason=trace.initial_failure_reason,
        final_failure_reason=trace.final_failure_reason,
        failure_stage=None if success else "recovery_execution",
        failure_reason=None if success else trace.final_failure_reason,
    )


def _trace(
    *,
    entry: str,
    stage: str,
    code: str | None,
    initial_stage: str | None,
    final_reason: str | None,
) -> object:
    return SimpleNamespace(
        recovery_entry_type=entry,
        recovery_stage=stage,
        recovery_attempts=1,
        recovered=final_reason is None,
        initial_execution_failure_code=code,
        initial_execution_failure_stage=initial_stage,
        initial_failure_reason="opaque initial failure text",
        diagnosis="structured_diagnosis",
        final_failure_reason=final_reason,
        recovery_grasp_verification=None,
        recovery_placement_verification=None,
    )


def _synthetic_episode(
    method: str,
    scenario: object,
    perturbation: object,
) -> EpisodeResult:
    seed = scenario.seed
    recovery = method == "vision_recovery"
    closed = method == "vision_closed_loop"
    verification = closed or recovery
    task_success = not (seed == 1 and method in {"vision_closed_loop", "vision_recovery"})
    grasp_failures = 1 if closed and seed == 0 else 0
    placement_failures = (
        2 if recovery and seed == 1
        else 1 if closed and seed == 1
        else 0
    )
    initial_grasp_verification = (
        None if recovery and seed == 0
        else False if grasp_failures
        else True if verification
        else None
    )
    initial_placement_verification = (
        None if (recovery and seed == 0) or (closed and seed == 0)
        else False if placement_failures
        else True if verification
        else None
    )
    return EpisodeResult(
        benchmark_version=BENCHMARK_VERSION,
        method=method,
        seed=seed,
        difficulty=scenario.difficulty,
        distractor_count=len(scenario.distractors),
        instruction=scenario.task.instruction,
        source_color=scenario.source.color,
        source_type=scenario.source.object_type,
        target_color=scenario.target.color,
        target_type=scenario.target.object_type,
        perturbation_name=perturbation.name,
        perturbation_stage=perturbation.stage,
        perturbation_axis=perturbation.axis,
        perturbation_offset_m=perturbation.offset_m,
        perturbation_first_attempt_only=perturbation.first_attempt_only,
        agent_success=task_success,
        task_success=task_success,
        grasp_success=True,
        placement_success=task_success,
        simulation_steps=100 + seed,
        observation_count=3 if method != "oracle_scripted" else 0,
        recovery_activated=True if recovery else None,
        recovery_attempts=1 if recovery else None,
        recovery_success=(seed == 0) if recovery else None,
        recovery_stage=(
            "pre_verification_execution" if recovery and seed == 0
            else "placement_verification" if recovery
            else None
        ),
        recovered=(seed == 0) if recovery else None,
        recovery_final_verification_result=(seed == 0) if recovery else None,
        grasp_verification_result=None,
        placement_verification_result=None,
        verification_failure_count=grasp_failures + placement_failures,
        source_grounding_success=(method != "oracle_scripted"),
        target_grounding_success=(method != "oracle_scripted"),
        source_position_error=0.002 if method != "oracle_scripted" else None,
        target_position_error=0.003 if method != "oracle_scripted" else None,
        failure_stage=(
            None if task_success
            else "recovery_placement_verification" if recovery
            else "placement_verification"
        ),
        failure_reason=None if task_success else "synthetic failure",
        wall_clock_seconds=0.1 + seed * 0.1,
        recovery_entry_type=(
            "pre_verification" if recovery and seed == 0
            else "post_verification" if recovery
            else None
        ),
        initial_failure_reason="initial failure" if recovery else None,
        recovery_diagnosis="synthetic diagnosis" if recovery else None,
        final_failure_reason=None if task_success else "synthetic failure",
        initial_grasp_verification_result=initial_grasp_verification,
        initial_placement_verification_result=initial_placement_verification,
        retry_grasp_verification_result=True if recovery else None,
        retry_placement_verification_result=(seed == 0) if recovery else None,
        grasp_verification_failure_count=grasp_failures,
        placement_verification_failure_count=placement_failures,
    )


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _csv_value(value: object) -> str:
    return "" if value is None else str(value)


def test_successful_preverification_recovery_preserves_initial_evidence(
    tmp_path: Path,
) -> None:
    scenario = generate_scenario(56, distractor_count=2)
    initial = _execution(
        False,
        grasp_success=False,
        placement_success=False,
        failure_stage="grasp",
        failure_reason="human text deliberately contains no structured code",
        close_result=_timeout_gripper("persistent_bilateral_non_target_contact"),
    )
    retry = _execution(True, grasp_success=True, placement_success=True)
    trace = _trace(
        entry="pre_verification",
        stage="pre_verification_execution",
        code="persistent_bilateral_non_target_contact",
        initial_stage="grasp",
        final_reason=None,
    )
    raw = _raw_recovery_result(
        initial_execution=initial,
        retry_execution=retry,
        trace=trace,
        success=True,
        initial_grasp=None,
        initial_placement=None,
        retry_grasp=True,
        retry_placement=True,
    )
    episode = benchmark_runner._normalize_result(
        "vision_recovery",
        scenario,
        raw,
        _evaluation(),
        0.25,
        PERTURBATION_REGISTRY[0],
    )

    assert episode.task_success and episode.recovered
    assert episode.recovery_entry_type == "pre_verification"
    assert episode.initial_execution_failure_code == (
        "persistent_bilateral_non_target_contact"
    )
    assert episode.initial_execution_failure_stage == "grasp"
    assert episode.initial_failure_reason == "opaque initial failure text"
    assert episode.recovery_diagnosis == "structured_diagnosis"
    assert episode.final_failure_reason is None
    assert episode.initial_gripper_termination_reason == "timeout"
    assert episode.initial_gripper_timeout_diagnostic == (
        "persistent_bilateral_non_target_contact"
    )

    run = benchmark_runner.run_benchmark(
        start_seed=56,
        episodes=1,
        methods=("vision_recovery",),
        perturbations=(PERTURBATION_NAMES[0],),
        output_root=tmp_path,
        run_id="pre-verification",
        episode_runner=lambda *_: episode,
    )
    assert run.output_dir is not None
    json_row = json.loads((run.output_dir / "episodes.jsonl").read_text())
    csv_row = _csv_rows(run.output_dir / "episodes.csv")[0]
    assert set(json_row) == set(CSV_FIELDS)
    for field, value in episode.to_dict().items():
        assert json_row[field] == value
        assert csv_row[field] == ("" if value is None else str(value))


def test_failed_recovery_preserves_retry_and_final_failure_details() -> None:
    scenario = generate_scenario(10, distractor_count=1)
    initial_reach = _reach_failure("ik_residual_exceeds_execution_tolerance")
    retry_reach = _reach_failure(
        "execution_timeout",
        residual=0.003,
        tolerance=0.004,
    )
    initial = _execution(
        False,
        grasp_success=False,
        placement_success=False,
        failure_stage="grasp",
        failure_reason="initial execution stopped",
        failed_reach=initial_reach,
    )
    retry = _execution(
        False,
        grasp_success=False,
        placement_success=False,
        failure_stage="grasp",
        failure_reason="retry execution stopped",
        failed_reach=retry_reach,
    )
    trace = _trace(
        entry="pre_verification",
        stage="pre_verification_execution",
        code="ik_residual_exceeds_execution_tolerance",
        initial_stage="grasp",
        final_reason="retry execution stopped",
    )
    raw = _raw_recovery_result(
        initial_execution=initial,
        retry_execution=retry,
        trace=trace,
        success=False,
        initial_grasp=None,
        initial_placement=None,
        retry_grasp=None,
        retry_placement=None,
    )
    episode = benchmark_runner._normalize_result(
        "vision_recovery",
        scenario,
        raw,
        _evaluation(False),
        0.3,
        PERTURBATION_REGISTRY[0],
    )

    assert episode.initial_failed_reach_failure_code == (
        "ik_residual_exceeds_execution_tolerance"
    )
    assert episode.initial_failed_reach_ik_position_residual == pytest.approx(0.005)
    assert episode.initial_failed_reach_required_position_tolerance == pytest.approx(
        0.004
    )
    assert episode.retry_execution_failure_code == "execution_timeout"
    assert episode.retry_execution_failure_stage == "grasp"
    assert episode.retry_execution_failure_reason == "retry execution stopped"
    assert episode.retry_failed_reach_failure_code == "execution_timeout"
    assert episode.retry_failed_reach_ik_position_residual == pytest.approx(0.003)
    assert episode.retry_failed_reach_required_position_tolerance == pytest.approx(
        0.004
    )
    assert episode.final_failure_reason == "retry execution stopped"


@pytest.mark.parametrize(
    ("stage", "initial_grasp", "initial_placement", "grasp_failures", "placement_failures"),
    [
        ("grasp_verification", False, None, 1, 0),
        ("placement_verification", True, False, 0, 1),
    ],
)
def test_postverification_recovery_stage_remains_distinguishable(
    stage: str,
    initial_grasp: bool | None,
    initial_placement: bool | None,
    grasp_failures: int,
    placement_failures: int,
) -> None:
    scenario = generate_scenario(0)
    initial = _execution(False, grasp_success=True, placement_success=False)
    retry = _execution(True, grasp_success=True, placement_success=True)
    trace = _trace(
        entry="post_verification",
        stage=stage,
        code=None,
        initial_stage=None,
        final_reason=None,
    )
    raw = _raw_recovery_result(
        initial_execution=initial,
        retry_execution=retry,
        trace=trace,
        success=True,
        initial_grasp=initial_grasp,
        initial_placement=initial_placement,
        retry_grasp=True,
        retry_placement=True,
    )
    episode = benchmark_runner._normalize_result(
        "vision_recovery",
        scenario,
        raw,
        _evaluation(),
        0.2,
    )

    assert episode.recovery_entry_type == "post_verification"
    assert episode.recovery_stage == stage
    assert episode.initial_execution_failure_code is None
    assert episode.initial_execution_failure_stage is None
    assert episode.grasp_verification_failure_count == grasp_failures
    assert episode.placement_verification_failure_count == placement_failures
    assert episode.verification_failure_count == grasp_failures + placement_failures


def test_non_recovery_methods_have_null_recovery_evidence() -> None:
    scenario = generate_scenario(0)
    failed_reach = _reach_failure("execution_timeout")
    execution = _execution(
        False,
        grasp_success=False,
        placement_success=False,
        failure_stage="grasp",
        failure_reason="structured execution failed",
        failed_reach=failed_reach,
    )
    vision_raw = SimpleNamespace(
        success=False,
        total_steps=10,
        observation_count=1,
        execution_result=execution,
        grasp_verification=None,
        placement_verification=None,
        source_detection=None,
        target_detection=None,
        failure_stage="grasp",
        failure_reason="structured execution failed",
    )
    oracle_raw = SimpleNamespace(
        success=False,
        pick_success=False,
        total_steps=10,
        failure_reason="oracle pick failed",
        pick_result=SimpleNamespace(
            pregrasp_result=failed_reach,
            approach_result=None,
            lift_result=None,
            open_result=None,
            close_result=None,
        ),
        release_result=None,
    )
    raw_by_method = {
        "oracle_scripted": oracle_raw,
        "vision_open_loop": vision_raw,
        "vision_closed_loop": vision_raw,
    }
    for method, raw in raw_by_method.items():
        episode = benchmark_runner._normalize_result(
            method,
            scenario,
            raw,
            _evaluation(False),
            0.1,
            PERTURBATION_REGISTRY[0],
        )
        payload = episode.to_dict()
        assert payload["recovery_activated"] is None
        assert payload["recovery_attempts"] is None
        assert payload["recovery_success"] is None
        assert payload["recovery_stage"] is None
        assert payload["recovered"] is None
        assert payload["recovery_entry_type"] is None
        assert payload["initial_execution_failure_code"] is None
        assert payload["initial_execution_failure_stage"] is None
        assert payload["initial_execution_failure_reason"] is None
        assert payload["initial_failure_reason"] is None
        assert payload["recovery_diagnosis"] is None
        assert payload["final_failure_reason"] is None
        assert payload["retry_execution_failure_code"] is None
        assert payload["retry_execution_failure_stage"] is None
        assert payload["retry_execution_failure_reason"] is None
        assert payload["initial_failed_reach_failure_code"] == "execution_timeout"


def test_failure_text_is_never_parsed_for_structured_codes() -> None:
    scenario = generate_scenario(0)
    reason = "text mentions ik_residual_exceeds_execution_tolerance only"
    execution = _execution(
        False,
        grasp_success=False,
        placement_success=False,
        failure_stage="grasp",
        failure_reason=reason,
    )
    raw = SimpleNamespace(
        success=False,
        total_steps=10,
        observation_count=1,
        recovery_activated=False,
        recovery_attempts=0,
        recovery_stage=None,
        recovered=False,
        recovery_trace=None,
        initial_execution_result=execution,
        recovery_execution_result=None,
        execution_result=execution,
        initial_grasp_verification=None,
        initial_placement_verification=None,
        grasp_verification=None,
        placement_verification=None,
        source_detection=None,
        target_detection=None,
        initial_failure_reason=None,
        final_failure_reason=reason,
        failure_stage="grasp",
        failure_reason=reason,
    )
    episode = benchmark_runner._normalize_result(
        "vision_recovery",
        scenario,
        raw,
        _evaluation(False),
        0.1,
    )

    assert episode.initial_execution_failure_code is None
    assert episode.initial_failed_reach_failure_code is None
    assert episode.initial_gripper_timeout_diagnostic is None


def test_condition_and_paired_artifacts_are_complete_and_deterministic(
    tmp_path: Path,
) -> None:
    run = benchmark_runner.run_benchmark(
        episodes=2,
        perturbations=PERTURBATION_NAMES,
        output_root=tmp_path,
        run_id="evidence-primary",
        episode_runner=_synthetic_episode,
    )
    assert run.output_dir is not None
    assert {path.name for path in run.output_dir.iterdir()} == {
        "episodes.jsonl",
        "episodes.csv",
        "summary.json",
        "metadata.json",
        "condition_summary.csv",
        "paired_success.csv",
    }

    jsonl_rows = [
        json.loads(line)
        for line in (run.output_dir / "episodes.jsonl").read_text().splitlines()
    ]
    episode_csv_rows = _csv_rows(run.output_dir / "episodes.csv")
    assert len(jsonl_rows) == len(episode_csv_rows) == len(run.episodes) == 16
    for episode, jsonl_row, csv_row in zip(
        run.episodes,
        jsonl_rows,
        episode_csv_rows,
        strict=True,
    ):
        expected = episode.to_dict()
        assert jsonl_row == expected
        assert csv_row == {
            field: _csv_value(value) for field, value in expected.items()
        }
    stored_summary = json.loads((run.output_dir / "summary.json").read_text())
    assert stored_summary["benchmark_version"] == BENCHMARK_VERSION
    assert stored_summary["methods"] == summarize_episodes(run.episodes)
    assert stored_summary["perturbations"] == run.perturbation_summary

    summary_rows = _csv_rows(run.output_dir / "condition_summary.csv")
    assert CONDITION_SUMMARY_FIELDS == EXPECTED_CONDITION_SUMMARY_FIELDS
    assert tuple(summary_rows[0]) == EXPECTED_CONDITION_SUMMARY_FIELDS
    assert [(row["condition"], row["method"]) for row in summary_rows] == [
        (condition, method)
        for condition in PERTURBATION_NAMES
        for method in METHODS
    ]
    assert len(summary_rows) == 8
    by_episode_group = {
        (condition, method): [
            item
            for item in run.episodes
            if item.perturbation_name == condition and item.method == method
        ]
        for condition in PERTURBATION_NAMES
        for method in METHODS
    }
    for row in summary_rows:
        grouped = by_episode_group[(row["condition"], row["method"])]
        assert int(row["episodes"]) == len(grouped)
        assert int(row["task_success_count"]) == sum(
            item.task_success for item in grouped
        )
        assert int(row["grasp_success_count"]) == sum(
            item.grasp_success is True for item in grouped
        )
        assert int(row["placement_success_count"]) == sum(
            item.placement_success is True for item in grouped
        )
        assert float(row["task_success_rate"]) == pytest.approx(
            sum(item.task_success for item in grouped) / len(grouped)
        )
        assert float(row["grasp_success_rate"]) == pytest.approx(
            sum(item.grasp_success is True for item in grouped) / len(grouped)
        )
        assert float(row["placement_success_rate"]) == pytest.approx(
            sum(item.placement_success is True for item in grouped) / len(grouped)
        )
        if row["method"] in {"vision_closed_loop", "vision_recovery"}:
            expected_grasp_failures = sum(
                item.grasp_verification_failure_count for item in grouped
            )
            expected_placement_failures = sum(
                item.placement_verification_failure_count for item in grouped
            )
            assert int(row["grasp_verification_failure_count"]) == (
                expected_grasp_failures
            )
            assert int(row["placement_verification_failure_count"]) == (
                expected_placement_failures
            )
            assert int(row["verification_failure_count"]) == (
                expected_grasp_failures + expected_placement_failures
            )
        else:
            assert row["verification_failure_count"] == ""
            assert row["grasp_verification_failure_count"] == ""
            assert row["placement_verification_failure_count"] == ""
        if row["method"] == "vision_recovery":
            assert int(row["recovery_activation_count"]) == sum(
                item.recovery_activated is True for item in grouped
            )
            assert int(row["recovery_attempt_count"]) == sum(
                item.recovery_attempts or 0 for item in grouped
            )
            assert int(row["recovery_success_count"]) == sum(
                item.recovery_success is True for item in grouped
            )
            assert int(row["recovery_failure_count"]) == sum(
                item.recovery_activated is True
                and item.recovery_success is False
                for item in grouped
            )
            assert int(row["pre_verification_recovery_count"]) == sum(
                item.recovery_activated is True
                and item.recovery_entry_type == "pre_verification"
                for item in grouped
            )
            assert int(row["post_verification_recovery_count"]) == sum(
                item.recovery_activated is True
                and item.recovery_entry_type == "post_verification"
                for item in grouped
            )
        else:
            for field in (
                "recovery_activation_count",
                "recovery_attempt_count",
                "recovery_success_count",
                "recovery_failure_count",
                "pre_verification_recovery_count",
                "post_verification_recovery_count",
            ):
                assert row[field] == ""
        assert float(row["mean_simulation_steps"]) == pytest.approx(100.5)
        assert float(row["median_simulation_steps"]) == pytest.approx(100.5)
        assert float(row["std_simulation_steps"]) == pytest.approx(0.5)
        assert int(row["min_simulation_steps"]) == 100
        assert int(row["max_simulation_steps"]) == 101
        assert float(row["mean_runtime_seconds"]) == pytest.approx(0.15)
        assert float(row["total_runtime_seconds"]) == pytest.approx(0.3)

    recovery_rows = [
        row for row in summary_rows if row["method"] == "vision_recovery"
    ]
    assert all(row["recovery_activation_count"] == "2" for row in recovery_rows)
    assert all(row["recovery_attempt_count"] == "2" for row in recovery_rows)
    assert all(row["recovery_success_count"] == "1" for row in recovery_rows)
    assert all(row["recovery_failure_count"] == "1" for row in recovery_rows)
    assert all(
        row["pre_verification_recovery_count"] == "1"
        for row in recovery_rows
    )
    assert all(
        row["post_verification_recovery_count"] == "1"
        for row in recovery_rows
    )

    paired_rows = _csv_rows(run.output_dir / "paired_success.csv")
    assert PAIRED_SUCCESS_FIELDS == EXPECTED_PAIRED_SUCCESS_FIELDS
    assert tuple(paired_rows[0]) == EXPECTED_PAIRED_SUCCESS_FIELDS
    assert [(row["condition"], int(row["seed"])) for row in paired_rows] == [
        (condition, seed)
        for condition in PERTURBATION_NAMES
        for seed in range(2)
    ]
    assert len(paired_rows) == 4
    for row in paired_rows:
        for method, prefix in PAIRED_METHOD_PREFIXES.items():
            source = next(
                item
                for item in run.episodes
                if item.perturbation_name == row["condition"]
                and item.seed == int(row["seed"])
                and item.method == method
            )
            for metric in ("task_success", "grasp_success", "placement_success"):
                assert row[f"{prefix}_{metric}"] == str(getattr(source, metric))
        closed = next(
            item
            for item in run.episodes
            if item.perturbation_name == row["condition"]
            and item.seed == int(row["seed"])
            and item.method == "vision_closed_loop"
        )
        assert row["vision_closed_verification_failure"] == str(
            closed.verification_failure_count > 0
        )
        assert row["vision_closed_verification_failure_count"] == str(
            closed.verification_failure_count
        )
        assert row["vision_closed_failure_stage"] == _csv_value(
            closed.failure_stage
        )
        recovery = next(
            item
            for item in run.episodes
            if item.perturbation_name == row["condition"]
            and item.seed == int(row["seed"])
            and item.method == "vision_recovery"
        )
        assert row["vision_recovery_verification_failure_occurred"] == str(
            recovery.verification_failure_count > 0
        )
        assert row["vision_recovery_verification_failure_count"] == str(
            recovery.verification_failure_count
        )
        assert row["vision_recovery_failure_stage"] == _csv_value(
            recovery.failure_stage
        )
        assert row["vision_recovery_recovery_activated"] == _csv_value(
            recovery.recovery_activated
        )
        assert row["vision_recovery_recovery_entry_type"] == _csv_value(
            recovery.recovery_entry_type
        )
        assert row["vision_recovery_recovered"] == _csv_value(recovery.recovered)
        assert row["vision_recovery_recovery_success"] == _csv_value(
            recovery.recovery_success
        )
        assert row["vision_recovery_final_success"] == str(
            recovery.task_success
        )

    reversed_run = replace(run, episodes=tuple(reversed(run.episodes)))
    copy_dir = benchmark_runner.save_run(reversed_run, tmp_path, "evidence-copy")
    for filename in ("condition_summary.csv", "paired_success.csv"):
        assert (run.output_dir / filename).read_bytes() == (
            copy_dir / filename
        ).read_bytes()
    with pytest.raises(ValueError, match="record count mismatch"):
        benchmark_runner.save_run(
            replace(run, episodes=run.episodes[:-1]),
            tmp_path,
            "incomplete",
        )
    assert not (tmp_path / "incomplete").exists()


def test_condition_summary_recomputes_primary_metrics_independently() -> None:
    scenario = generate_scenario(0)
    first = _synthetic_episode(
        "oracle_scripted",
        scenario,
        PERTURBATION_REGISTRY[0],
    )
    second = replace(
        first,
        seed=1,
        task_success=True,
        grasp_success=False,
        placement_success=False,
        simulation_steps=None,
        wall_clock_seconds=0.2,
    )
    third = replace(
        first,
        seed=2,
        task_success=True,
        grasp_success=True,
        placement_success=False,
        simulation_steps=102,
        wall_clock_seconds=0.3,
    )
    fourth = replace(
        first,
        seed=3,
        task_success=False,
        grasp_success=None,
        placement_success=False,
        simulation_steps=104,
        wall_clock_seconds=0.4,
    )

    row = benchmark_runner._condition_summary_rows(
        (first, second, third, fourth)
    )[0]
    assert row["episodes"] == 4
    assert row["task_success_count"] == 3
    assert row["task_success_rate"] == pytest.approx(0.75)
    assert row["grasp_success_count"] == 2
    assert row["grasp_success_rate"] == pytest.approx(2 / 3)
    assert row["placement_success_count"] == 1
    assert row["placement_success_rate"] == pytest.approx(0.25)
    assert row["mean_simulation_steps"] == pytest.approx(102.0)
    assert row["median_simulation_steps"] == pytest.approx(102.0)
    assert row["std_simulation_steps"] == pytest.approx((8 / 3) ** 0.5)
    assert row["min_simulation_steps"] == 100
    assert row["max_simulation_steps"] == 104
    assert row["mean_runtime_seconds"] == pytest.approx(0.25)
    assert row["total_runtime_seconds"] == pytest.approx(1.0)


def test_condition_summary_uses_persisted_recovery_outcomes_and_attempts() -> None:
    scenario = generate_scenario(0)
    base = _synthetic_episode(
        "vision_recovery",
        scenario,
        PERTURBATION_REGISTRY[0],
    )
    records = (
        replace(
            base,
            seed=0,
            recovery_activated=False,
            recovery_attempts=0,
            recovery_success=None,
            recovered=False,
            recovery_entry_type=None,
        ),
        replace(
            base,
            seed=1,
            recovery_activated=True,
            recovery_attempts=2,
            recovery_success=True,
            recovered=True,
            recovery_entry_type="pre_verification",
        ),
        replace(
            base,
            seed=2,
            recovery_activated=True,
            recovery_attempts=3,
            recovery_success=False,
            recovered=False,
            recovery_entry_type="post_verification",
        ),
        replace(
            base,
            seed=3,
            recovery_activated=True,
            recovery_attempts=1,
            recovery_success=False,
            recovered=True,
            recovery_entry_type="post_verification",
        ),
    )

    row = benchmark_runner._condition_summary_rows(records)[0]
    assert row["recovery_activation_count"] == 3
    assert row["recovery_attempt_count"] == 6
    assert row["recovery_success_count"] == 1
    assert row["recovery_failure_count"] == 2
    assert row["pre_verification_recovery_count"] == 1
    assert row["post_verification_recovery_count"] == 2


def test_paired_success_preserves_explicit_episode_fields() -> None:
    scenario = generate_scenario(0)
    perturbation = PERTURBATION_REGISTRY[0]
    records = (
        replace(
            _synthetic_episode("oracle_scripted", scenario, perturbation),
            seed=7,
            task_success=True,
            grasp_success=False,
            placement_success=None,
        ),
        replace(
            _synthetic_episode("vision_open_loop", scenario, perturbation),
            seed=7,
            task_success=False,
            grasp_success=True,
            placement_success=None,
        ),
        replace(
            _synthetic_episode("vision_closed_loop", scenario, perturbation),
            seed=7,
            task_success=True,
            grasp_success=None,
            placement_success=False,
            verification_failure_count=2,
            grasp_verification_failure_count=1,
            placement_verification_failure_count=1,
            failure_stage="placement_verification",
        ),
        replace(
            _synthetic_episode("vision_recovery", scenario, perturbation),
            seed=7,
            task_success=False,
            grasp_success=True,
            placement_success=None,
            recovery_activated=True,
            recovery_entry_type="pre_verification",
            recovered=True,
            recovery_success=False,
            failure_stage="objective_after_visual_recovery",
        ),
    )

    assert benchmark_runner._paired_success_rows(records) == [
        {
            "condition": perturbation.name,
            "seed": 7,
            "oracle_task_success": True,
            "oracle_grasp_success": False,
            "oracle_placement_success": None,
            "vision_open_task_success": False,
            "vision_open_grasp_success": True,
            "vision_open_placement_success": None,
            "vision_closed_task_success": True,
            "vision_closed_grasp_success": None,
            "vision_closed_placement_success": False,
            "vision_closed_verification_failure": True,
            "vision_closed_verification_failure_count": 2,
            "vision_closed_failure_stage": "placement_verification",
            "vision_recovery_task_success": False,
            "vision_recovery_grasp_success": True,
            "vision_recovery_placement_success": None,
            "vision_recovery_verification_failure_occurred": False,
            "vision_recovery_verification_failure_count": 0,
            "vision_recovery_failure_stage": "objective_after_visual_recovery",
            "vision_recovery_recovery_activated": True,
            "vision_recovery_recovery_entry_type": "pre_verification",
            "vision_recovery_recovered": True,
            "vision_recovery_recovery_success": False,
            "vision_recovery_final_success": False,
        }
    ]


def test_metadata_has_authoritative_boolean_dirty_worktree_and_full_sha(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    expected_sha = "a" * 40
    monkeypatch.setattr(benchmark_runner, "_git_commit", lambda: expected_sha)
    monkeypatch.setattr(benchmark_runner, "_git_worktree_dirty", lambda: False)
    run = benchmark_runner.run_benchmark(
        episodes=1,
        methods=("oracle_scripted",),
        perturbations=(PERTURBATION_NAMES[0],),
        output_root=tmp_path,
        run_id="metadata",
        episode_runner=lambda method, current, perturbation: _synthetic_episode(
            method,
            current,
            perturbation,
        ),
    )

    assert run.metadata["benchmark_version"] == BENCHMARK_VERSION
    assert run.metadata["git_commit"] == expected_sha
    assert run.metadata["dirty_worktree"] is False
    assert run.metadata["git_worktree_dirty"] is run.metadata["dirty_worktree"]
    assert run.metadata["seeds"] == [0]
    assert run.metadata["methods"] == ["oracle_scripted"]
    assert run.metadata["scenario_difficulty"] == "basic"
    assert run.metadata["perturbations"] == [PERTURBATION_REGISTRY[0].to_dict()]
    assert run.metadata["max_recovery_attempts"] == 1
    assert run.output_dir is not None
    persisted = json.loads((run.output_dir / "metadata.json").read_text())
    assert persisted["benchmark_version"] == BENCHMARK_VERSION
    assert persisted["git_commit"] == expected_sha
    assert persisted["dirty_worktree"] is False
    assert persisted["git_worktree_dirty"] is persisted["dirty_worktree"]


def test_episode_matrix_validator_accepts_complete_and_rejects_gaps() -> None:
    records = []
    for perturbation in PERTURBATION_REGISTRY:
        for seed in range(2):
            scenario = generate_scenario(seed, distractor_count=seed % 3)
            records.extend(
                _synthetic_episode(method, scenario, perturbation)
                for method in METHODS
            )
    validate_episode_matrix(
        records,
        expected_conditions=PERTURBATION_NAMES,
        expected_seeds=range(2),
        expected_methods=METHODS,
    )
    with pytest.raises(ValueError, match="coverage mismatch"):
        validate_episode_matrix(
            records[:-1],
            expected_conditions=PERTURBATION_NAMES,
            expected_seeds=range(2),
            expected_methods=METHODS,
        )
    with pytest.raises(ValueError, match="Duplicate"):
        validate_episode_matrix(
            [*records, records[0]],
            expected_conditions=PERTURBATION_NAMES,
            expected_seeds=range(2),
            expected_methods=METHODS,
        )
