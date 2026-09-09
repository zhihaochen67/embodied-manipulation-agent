"""Focused Phase 14B perturbation protocol and serialization tests."""

from __future__ import annotations

import csv
import io
import json
from types import SimpleNamespace

import pytest

from embodied_manipulation.benchmark import (
    BENCHMARK_VERSION,
    CSV_FIELDS,
    METHODS,
    PERTURBATION_NAMES,
    PERTURBATION_REGISTRY,
    EpisodeResult,
    FirstAttemptPerturbingPlanner,
    generate_scenario,
    ordered_perturbations,
)
from embodied_manipulation.benchmark import runner as benchmark_runner
from embodied_manipulation.perception import LocalizationResult
from embodied_manipulation.planning import TaskPlanner


def _grounding(seed: int = 0) -> tuple[object, LocalizationResult, LocalizationResult]:
    scenario = generate_scenario(seed, distractor_count=seed % 3)
    source = LocalizationResult(
        scenario.task.source,
        scenario.source.initial_position,
        "object_center",
        "vision",
    )
    assert scenario.task.target is not None
    target = LocalizationResult(
        scenario.task.target,
        scenario.target.initial_position,
        "tray_floor_body_center",
        "vision",
    )
    return scenario, source, target


def _episode(method: str, scenario: object, perturbation: object) -> EpisodeResult:
    vision = method != "oracle_scripted"
    recovery = method == "vision_recovery"
    return EpisodeResult(
        benchmark_version=BENCHMARK_VERSION,
        method=method,
        seed=scenario.seed,
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
        agent_success=True,
        task_success=True,
        grasp_success=True,
        placement_success=True,
        simulation_steps=100,
        observation_count=1 if vision else 0,
        recovery_activated=False,
        recovery_attempts=0,
        recovery_success=None,
        recovery_stage=None,
        recovered=False if recovery else None,
        recovery_final_verification_result=None,
        grasp_verification_result=None,
        placement_verification_result=None,
        verification_failure_count=0,
        source_grounding_success=True if vision else None,
        target_grounding_success=True if vision else None,
        source_position_error=0.002 if vision else None,
        target_position_error=0.003 if vision else None,
        failure_stage=None,
        failure_reason=None,
        wall_clock_seconds=0.0,
    )


def test_frozen_perturbation_registry_has_exact_names_and_magnitudes() -> None:
    assert PERTURBATION_NAMES == (
        "grasp_shift_x_120mm",
        "placement_shift_x_180mm",
    )
    assert tuple(
        (item.stage, item.axis, item.offset_m, item.first_attempt_only)
        for item in PERTURBATION_REGISTRY
    ) == (
        ("grasp", "x", 0.12, True),
        ("placement", "x", 0.18, True),
    )
    assert ordered_perturbations(reversed(PERTURBATION_NAMES)) == (
        PERTURBATION_REGISTRY
    )


@pytest.mark.parametrize("perturbation", PERTURBATION_REGISTRY)
def test_first_plan_is_perturbed_and_recovery_replan_is_nominal(
    perturbation: object,
) -> None:
    scenario, source, target = _grounding()
    metadata_before = scenario.to_dict()
    nominal = TaskPlanner().plan(scenario.task, source, target)
    planner = FirstAttemptPerturbingPlanner(perturbation)

    first = planner.plan(scenario.task, source, target)
    recovery = planner.plan(scenario.task, source, target)

    expected_grasp_x = nominal.grasp_target[0] + (
        perturbation.offset_m if perturbation.stage == "grasp" else 0.0
    )
    expected_release_x = nominal.release_target[0] + (
        perturbation.offset_m if perturbation.stage == "placement" else 0.0
    )
    assert first.grasp_target[0] == pytest.approx(expected_grasp_x)
    assert first.release_target[0] == pytest.approx(expected_release_x)
    assert recovery == nominal
    assert scenario.to_dict() == metadata_before


@pytest.mark.parametrize("perturbation", PERTURBATION_REGISTRY)
def test_same_geometry_is_dispatched_to_every_method_without_recovery_fault_injection(
    monkeypatch: pytest.MonkeyPatch,
    perturbation: object,
) -> None:
    scenario, source, target = _grounding()
    nominal = TaskPlanner().plan(scenario.task, source, target)
    captured: dict[str, object] = {}

    def fake_oracle(world: object, **kwargs: object) -> object:
        captured["oracle_kwargs"] = kwargs
        return object()

    def agent_type(label: str) -> type:
        class FakeAgent:
            def __init__(self, *, planner: object | None = None) -> None:
                captured[f"{label}_planner"] = planner

            def run(
                self,
                instruction: str,
                world: object,
                **kwargs: object,
            ) -> object:
                captured[f"{label}_run_kwargs"] = kwargs
                return object()

        return FakeAgent

    monkeypatch.setattr(benchmark_runner, "oracle_pick_place", fake_oracle)
    monkeypatch.setattr(
        benchmark_runner,
        "VisionOpenLoopAgent",
        agent_type("vision_open_loop"),
    )
    monkeypatch.setattr(
        benchmark_runner,
        "VisionClosedLoopAgent",
        agent_type("vision_closed_loop"),
    )
    monkeypatch.setattr(
        benchmark_runner,
        "VisionRecoveryAgent",
        agent_type("vision_recovery"),
    )

    for method in METHODS:
        benchmark_runner._run_method(method, scenario, object(), perturbation)

    assert captured["oracle_kwargs"] == {
        "grasp_xy_offset": perturbation.grasp_xy_offset,
        "placement_xy_offset": perturbation.placement_xy_offset,
    }
    vision_plans = []
    for method in METHODS[1:]:
        planner = captured[f"{method}_planner"]
        vision_plans.append(planner.plan(scenario.task, source, target))
        assert captured[f"{method}_run_kwargs"] == {}
    assert vision_plans[0] == vision_plans[1] == vision_plans[2]
    assert vision_plans[0] == perturbation.apply_to_plan(nominal)


def test_clean_dispatch_uses_no_perturbation_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = generate_scenario(0)
    captured: dict[str, object] = {}

    def fake_oracle(world: object, **kwargs: object) -> object:
        captured["oracle_kwargs"] = kwargs
        return object()

    class FakeAgent:
        def __init__(self, *, planner: object | None = None) -> None:
            captured.setdefault("planners", []).append(planner)

        def run(self, instruction: str, world: object, **kwargs: object) -> object:
            captured.setdefault("run_kwargs", []).append(kwargs)
            return object()

    monkeypatch.setattr(benchmark_runner, "oracle_pick_place", fake_oracle)
    monkeypatch.setattr(benchmark_runner, "VisionOpenLoopAgent", FakeAgent)
    monkeypatch.setattr(benchmark_runner, "VisionClosedLoopAgent", FakeAgent)
    monkeypatch.setattr(benchmark_runner, "VisionRecoveryAgent", FakeAgent)

    for method in METHODS:
        benchmark_runner._run_method(method, scenario, object())

    assert captured["oracle_kwargs"] == {}
    assert captured["planners"] == [None, None, None]
    assert captured["run_kwargs"] == [{}, {}, {}]


def test_perturbation_fields_serialize_to_flat_json_and_csv() -> None:
    scenario = generate_scenario(0)
    episode = _episode("vision_recovery", scenario, PERTURBATION_REGISTRY[0])
    payload = episode.to_dict()
    encoded = json.loads(json.dumps(payload))
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
    writer.writeheader()
    writer.writerow(payload)

    assert tuple(payload) == CSV_FIELDS
    assert encoded["perturbation_name"] == "grasp_shift_x_120mm"
    assert encoded["perturbation_offset_m"] == pytest.approx(0.12)
    assert "perturbation_first_attempt_only" in stream.getvalue()


def test_perturbed_tiny_run_is_deterministic_and_has_no_benchmark_retry() -> None:
    calls: list[tuple[str, int, str, int, dict[str, object]]] = []

    def run_one(
        method: str,
        scenario: object,
        perturbation: object,
    ) -> EpisodeResult:
        calls.append(
            (
                perturbation.name,
                scenario.seed,
                method,
                id(scenario),
                scenario.to_dict(),
            )
        )
        return _episode(method, scenario, perturbation)

    first = benchmark_runner.run_benchmark(
        episodes=2,
        perturbations=PERTURBATION_NAMES,
        save_outputs=False,
        episode_runner=run_one,
    )
    first_calls = list(calls)
    calls.clear()
    second = benchmark_runner.run_benchmark(
        episodes=2,
        perturbations=reversed(PERTURBATION_NAMES),
        save_outputs=False,
        episode_runner=run_one,
    )

    expected = [
        (condition, seed, method)
        for condition in PERTURBATION_NAMES
        for seed in range(2)
        for method in METHODS
    ]
    assert [(item[0], item[1], item[2]) for item in first_calls] == expected
    assert len(first_calls) == len(set(expected)) == 16
    assert [item.logical_dict() for item in first.episodes] == [
        item.logical_dict() for item in second.episodes
    ]
    for condition in PERTURBATION_NAMES:
        for seed in range(2):
            rows = [
                row
                for row in first_calls
                if row[0] == condition and row[1] == seed
            ]
            assert len({row[3] for row in rows}) == 1
            assert all(row[4] == rows[0][4] for row in rows)
    assert first.metadata["episode_order"] == (
        "perturbation_major_seed_major_method_minor"
    )
    assert first.metadata["perturbation_protocol"] == {
        "condition_order": list(PERTURBATION_NAMES),
        "application": "first relevant manipulation attempt only",
        "recovery_attempts_perturbed": False,
        "benchmark_level_retries": 0,
    }
    assert tuple(first.perturbation_summary) == PERTURBATION_NAMES
