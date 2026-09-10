"""Phase 11A benchmark protocol and persistence regression tests."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from embodied_manipulation.benchmark import (
    BENCHMARK_VERSION,
    CONDITION_SUMMARY_FIELDS,
    CSV_FIELDS,
    METHODS,
    PAIRED_SUCCESS_FIELDS,
    EpisodeResult,
    generate_scenario,
)
from embodied_manipulation.benchmark import runner as benchmark_runner
from embodied_manipulation.benchmark.metrics import summarize_episodes
from embodied_manipulation.control.pick_place import PlacementEvaluation
from embodied_manipulation.perception import LocalizationResult


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


def _episode(
    method: str,
    scenario: object,
    **overrides: object,
) -> EpisodeResult:
    vision = method != "oracle_scripted"
    recovery = method == "vision_recovery"
    values = dict(
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
        perturbation_name=None,
        perturbation_stage=None,
        perturbation_axis=None,
        perturbation_offset_m=None,
        perturbation_first_attempt_only=None,
        agent_success=True,
        task_success=True,
        grasp_success=True,
        placement_success=True,
        simulation_steps=100,
        observation_count=1 if vision else 0,
        recovery_activated=False if recovery else None,
        recovery_attempts=0 if recovery else None,
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
        wall_clock_seconds=0.1,
        infrastructure_error=None,
    )
    values.update(overrides)
    return EpisodeResult(**values)


def test_method_registry_and_csv_schema_are_stable() -> None:
    assert METHODS == (
        "oracle_scripted",
        "vision_open_loop",
        "vision_closed_loop",
        "vision_recovery",
    )
    assert CSV_FIELDS[0:5] == (
        "benchmark_version",
        "method",
        "seed",
        "difficulty",
        "distractor_count",
    )
    assert CSV_FIELDS[-2:] == ("wall_clock_seconds", "infrastructure_error")


def test_benchmark_metadata_uses_tracked_version() -> None:
    scenario = generate_scenario(0)
    run = benchmark_runner.run_benchmark(
        episodes=1,
        methods=("oracle_scripted",),
        save_outputs=False,
        episode_runner=lambda method, _: _episode(method, scenario),
    )

    assert BENCHMARK_VERSION == "phase14d-perturb-v2"
    assert run.metadata["benchmark_version"] == BENCHMARK_VERSION


def test_episode_result_is_json_serializable_and_oracle_metrics_are_na() -> None:
    episode = _episode("oracle_scripted", generate_scenario(0))
    encoded = json.dumps(episode.to_dict(), sort_keys=True)

    assert json.loads(encoded)["method"] == "oracle_scripted"
    assert tuple(episode.to_dict()) == CSV_FIELDS
    assert episode.observation_count == 0
    assert episode.source_grounding_success is None
    assert episode.target_grounding_success is None
    assert episode.source_position_error is None
    assert episode.target_position_error is None


def test_recovery_aggregate_denominators_and_nulls_are_correct() -> None:
    scenario = generate_scenario(0)
    records = [
        _episode("vision_recovery", scenario),
        _episode(
            "vision_recovery",
            scenario,
            recovery_activated=True,
            recovery_attempts=1,
            recovery_success=True,
        ),
        _episode(
            "vision_recovery",
            scenario,
            task_success=False,
            placement_success=False,
            recovery_activated=True,
            recovery_attempts=1,
            recovery_success=False,
            grasp_verification_failure_count=1,
            placement_verification_failure_count=1,
            verification_failure_count=2,
        ),
    ]

    summary = summarize_episodes(records)["vision_recovery"]

    assert summary["task_success_rate"] == pytest.approx(2.0 / 3.0)
    assert summary["recovery_activation_rate"] == pytest.approx(2.0 / 3.0)
    assert summary["recovery_success_rate"] == pytest.approx(0.5)
    assert summary["mean_recovery_attempts"] == pytest.approx(2.0 / 3.0)


def test_seed_major_order_reuses_one_logical_scenario_and_is_reproducible() -> None:
    seen: list[tuple[int, str, int, dict[str, object]]] = []

    def fake_episode(method: str, scenario: object) -> EpisodeResult:
        seen.append((scenario.seed, method, id(scenario), scenario.to_dict()))
        return _episode(method, scenario, wall_clock_seconds=len(seen) / 100.0)

    first = benchmark_runner.run_benchmark(
        episodes=2,
        save_outputs=False,
        episode_runner=fake_episode,
    )
    first_seen = list(seen)
    seen.clear()
    second = benchmark_runner.run_benchmark(
        episodes=2,
        save_outputs=False,
        episode_runner=fake_episode,
    )

    assert [(item.seed, item.method) for item in first.episodes] == [
        (seed, method) for seed in range(2) for method in METHODS
    ]
    for seed in range(2):
        rows = [row for row in first_seen if row[0] == seed]
        assert len({row[2] for row in rows}) == 1
        assert all(row[3] == rows[0][3] for row in rows)
    assert [item.logical_dict() for item in first.episodes] == [
        item.logical_dict() for item in second.episodes
    ]


def test_saved_artifacts_have_stable_flat_schema(tmp_path: Path) -> None:
    scenario = generate_scenario(0)
    run = benchmark_runner.run_benchmark(
        episodes=1,
        methods=("vision_open_loop",),
        output_root=tmp_path,
        run_id="schema-test",
        episode_runner=lambda method, _: _episode(method, scenario),
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
    with (run.output_dir / "episodes.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == CSV_FIELDS
        assert len(list(reader)) == 1
    payload = json.loads((run.output_dir / "episodes.jsonl").read_text())
    assert set(payload) == set(CSV_FIELDS)
    metadata = json.loads((run.output_dir / "metadata.json").read_text())
    assert "git_commit" in metadata
    assert "git_worktree_dirty" in metadata
    assert isinstance(metadata["dirty_worktree"], bool)
    assert metadata["git_worktree_dirty"] == metadata["dirty_worktree"]
    with (run.output_dir / "condition_summary.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == CONDITION_SUMMARY_FIELDS
        assert len(list(reader)) == 1
    with (run.output_dir / "paired_success.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == PAIRED_SUCCESS_FIELDS
        paired_rows = list(reader)
        assert len(paired_rows) == 1
        assert paired_rows[0]["vision_open_task_success"] == "True"
        assert paired_rows[0]["vision_open_grasp_success"] == "True"
        assert paired_rows[0]["vision_open_placement_success"] == "True"
        for prefix in ("oracle", "vision_closed", "vision_recovery"):
            assert paired_rows[0][f"{prefix}_task_success"] == ""
            assert paired_rows[0][f"{prefix}_grasp_success"] == ""
            assert paired_rows[0][f"{prefix}_placement_success"] == ""


def test_fresh_worlds_and_common_objective_success_semantics() -> None:
    scenario = generate_scenario(0)
    worlds: list[object] = []

    class FakeWorld:
        def __init__(self) -> None:
            worlds.append(self)
            self.scenario = None
            self.closed = False

        def __enter__(self) -> object:
            return self

        def __exit__(self, *_: object) -> None:
            self.closed = True

        def reset_from_scenario(self, value: object) -> None:
            self.scenario = value

    raw = SimpleNamespace(
        success=True,
        pick_success=True,
        total_steps=12,
        failure_reason=None,
    )
    for _ in range(2):
        result = benchmark_runner.run_episode(
            "oracle_scripted",
            scenario,
            world_factory=FakeWorld,
            method_runner=lambda *_: raw,
            objective_evaluator=lambda _: _evaluation(False),
        )
        assert result.agent_success
        assert not result.task_success
        assert result.failure_stage == "placement"
    assert len(worlds) == 2 and worlds[0] is not worlds[1]
    assert all(world.scenario is scenario and world.closed for world in worlds)


def test_exception_is_recorded_and_does_not_abort_small_run() -> None:
    def episode(method: str, scenario: object) -> EpisodeResult:
        if method == "oracle_scripted":
            raise RuntimeError("controlled infrastructure failure")
        return _episode(method, scenario)

    run = benchmark_runner.run_benchmark(
        episodes=1,
        methods=("oracle_scripted", "vision_open_loop"),
        save_outputs=False,
        episode_runner=episode,
    )

    assert len(run.episodes) == 2
    assert run.episodes[0].failure_stage == "benchmark_exception"
    assert "controlled infrastructure failure" in (
        run.episodes[0].infrastructure_error or ""
    )
    assert run.episodes[1].task_success


def test_clean_recovery_dispatch_never_passes_fault_injection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = generate_scenario(0)
    calls: list[dict[str, object]] = []
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
    execution = SimpleNamespace(grasp_success=True)
    raw = SimpleNamespace(
        success=True,
        total_steps=10,
        observation_count=3,
        recovery_activated=False,
        recovery_attempts=0,
        grasp_verification=None,
        placement_verification=None,
        initial_grasp_verification=None,
        initial_placement_verification=None,
        recovery_trace=None,
        initial_execution_result=execution,
        recovery_execution_result=None,
        source_detection=source,
        target_detection=target,
        failure_stage=None,
        failure_reason=None,
    )

    class FakeWorld:
        def __enter__(self) -> object:
            return self

        def __exit__(self, *_: object) -> None:
            pass

        def reset_from_scenario(self, _: object) -> None:
            pass

    def fake_run(
        self: object,
        instruction: str,
        world: object,
        **kwargs: object,
    ) -> object:
        calls.append(kwargs)
        return raw

    monkeypatch.setattr(benchmark_runner.VisionRecoveryAgent, "run", fake_run)
    result = benchmark_runner.run_episode(
        "vision_recovery",
        scenario,
        world_factory=FakeWorld,
        objective_evaluator=lambda _: _evaluation(),
    )

    assert result.task_success
    assert calls == [{}]


def test_one_seed_real_integration_preserves_all_method_semantics() -> None:
    run = benchmark_runner.run_benchmark(episodes=1, save_outputs=False)

    assert len(run.episodes) == 4
    assert all(item.task_success for item in run.episodes)
    assert [item.observation_count for item in run.episodes] == [0, 1, 3, 3]
    assert run.episodes[0].source_grounding_success is None
    assert all(
        item.source_position_error is not None for item in run.episodes[1:]
    )
    assert not run.episodes[-1].recovery_activated
