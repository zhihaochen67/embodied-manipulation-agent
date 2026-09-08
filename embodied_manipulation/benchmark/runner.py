"""Sequential, reproducible Phase 11A four-method benchmark runner."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from math import dist
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterable, Mapping

from embodied_manipulation.agent import (
    MAX_RECOVERY_ATTEMPTS,
    VisionClosedLoopAgent,
    VisionOpenLoopAgent,
    VisionRecoveryAgent,
)
from embodied_manipulation.control import evaluate_placement, oracle_pick_place
from embodied_manipulation.control.pick_place import PlacementEvaluation
from embodied_manipulation.simulation import World

from .metrics import (
    BENCHMARK_VERSION,
    CSV_FIELDS,
    METHODS,
    EpisodeResult,
    summarize_episodes,
)
from .scenarios import Scenario, generate_scenario

DEFAULT_START_SEED = 0
DEFAULT_EPISODES = 12
DEFAULT_DIFFICULTY = "basic"
DEFAULT_OUTPUT_ROOT = Path("outputs/benchmarks")

WorldFactory = Callable[[], Any]
MethodRunner = Callable[[str, Scenario, Any], Any]
EpisodeRunner = Callable[[str, Scenario], EpisodeResult]


@dataclass(frozen=True, slots=True)
class BenchmarkRun:
    """In-memory benchmark records plus optional persisted artifact path."""

    episodes: tuple[EpisodeResult, ...]
    summary: Mapping[str, Mapping[str, int | float | None]]
    metadata: Mapping[str, Any]
    output_dir: Path | None = None


def run_episode(
    method: str,
    scenario: Scenario,
    *,
    world_factory: WorldFactory | None = None,
    method_runner: MethodRunner | None = None,
    objective_evaluator: Callable[[Any], PlacementEvaluation] = evaluate_placement,
    clock: Callable[[], float] = perf_counter,
) -> EpisodeResult:
    """Run one method in a fresh world and normalize its result.

    Scenario metadata is ground truth available only after method execution for
    benchmark diagnostics. It is never passed into a vision agent's planner.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown benchmark method: {method}")
    make_world = world_factory or (lambda: World(gui=False))
    execute = method_runner or _run_method
    started = clock()
    try:
        with make_world() as world:
            world.reset_from_scenario(scenario)
            raw_result = execute(method, scenario, world)
            objective = objective_evaluator(world)
    except Exception as error:
        elapsed = max(0.0, clock() - started)
        return _exception_result(method, scenario, error, elapsed)
    elapsed = max(0.0, clock() - started)
    return _normalize_result(method, scenario, raw_result, objective, elapsed)


def run_benchmark(
    *,
    start_seed: int = DEFAULT_START_SEED,
    episodes: int = DEFAULT_EPISODES,
    methods: Iterable[str] = METHODS,
    difficulty: str = DEFAULT_DIFFICULTY,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    run_id: str | None = None,
    save_outputs: bool = True,
    episode_runner: EpisodeRunner | None = None,
    clock: Callable[[], float] = perf_counter,
) -> BenchmarkRun:
    """Run seed-major/method-minor episodes sequentially without retries."""
    if not isinstance(start_seed, int) or isinstance(start_seed, bool):
        raise ValueError("start_seed must be an integer")
    if not isinstance(episodes, int) or isinstance(episodes, bool) or episodes <= 0:
        raise ValueError("episodes must be a positive integer")
    selected_methods = _ordered_methods(methods)
    run_one = episode_runner or (
        lambda method, scenario: run_episode(method, scenario)
    )
    seeds = tuple(range(start_seed, start_seed + episodes))
    records: list[EpisodeResult] = []
    started = clock()
    for seed in seeds:
        scenario = generate_scenario(
            seed,
            difficulty,
            distractor_count=seed % 3,
        )
        for method in selected_methods:
            try:
                record = run_one(method, scenario)
            except Exception as error:
                record = _exception_result(method, scenario, error, 0.0)
            records.append(record)
    total_runtime = max(0.0, clock() - started)
    created_at = datetime.now(UTC)
    metadata = {
        "benchmark_version": BENCHMARK_VERSION,
        "created_at_utc": created_at.isoformat(),
        "start_seed": start_seed,
        "seeds": list(seeds),
        "methods": list(selected_methods),
        "scenario_difficulty": difficulty,
        "distractor_rule": "seed % 3",
        "episode_order": "seed_major_method_minor",
        "scenario_count": len(seeds),
        "episode_count": len(records),
        "max_recovery_attempts": MAX_RECOVERY_ATTEMPTS,
        "git_commit": _git_commit(),
        "git_worktree_dirty": _git_worktree_dirty(),
        "python_version": platform.python_version(),
        "total_wall_clock_seconds": total_runtime,
    }
    run = BenchmarkRun(
        episodes=tuple(records),
        summary=summarize_episodes(records),
        metadata=metadata,
    )
    if save_outputs:
        identifier = run_id or _default_run_id(start_seed, episodes, created_at)
        destination = save_run(run, output_root, identifier)
        run = replace(run, output_dir=destination)
    return run


def save_run(
    run: BenchmarkRun,
    output_root: str | Path,
    run_id: str,
) -> Path:
    """Persist JSONL, flat CSV, summary, and reproducibility metadata."""
    if (
        not run_id
        or run_id in {".", ".."}
        or any(character in run_id for character in ("/", "\\"))
    ):
        raise ValueError("run_id must be one non-empty path component")
    destination = Path(output_root) / run_id
    destination.mkdir(parents=True, exist_ok=False)
    with (destination / "episodes.jsonl").open("w", encoding="utf-8") as handle:
        for episode in run.episodes:
            handle.write(json.dumps(episode.to_dict(), sort_keys=True))
            handle.write("\n")
    with (destination / "episodes.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(episode.to_dict() for episode in run.episodes)
    _write_json(
        destination / "summary.json",
        {"benchmark_version": BENCHMARK_VERSION, "methods": run.summary},
    )
    _write_json(destination / "metadata.json", dict(run.metadata))
    return destination


def _run_method(method: str, scenario: Scenario, world: World) -> Any:
    if method == "oracle_scripted":
        return oracle_pick_place(world)
    instruction = scenario.task.instruction
    if method == "vision_open_loop":
        return VisionOpenLoopAgent().run(instruction, world)
    if method == "vision_closed_loop":
        return VisionClosedLoopAgent().run(instruction, world)
    if method == "vision_recovery":
        # Deliberately no RecoveryFaultInjection: Phase 11A is clean.
        return VisionRecoveryAgent().run(instruction, world)
    raise ValueError(f"Unknown benchmark method: {method}")


def _normalize_result(
    method: str,
    scenario: Scenario,
    result: Any,
    objective: PlacementEvaluation,
    elapsed: float,
) -> EpisodeResult:
    if method == "oracle_scripted":
        agent_success = bool(result.success)
        grasp_success = bool(result.pick_success)
        steps = int(result.total_steps)
        observation_count = 0
        recovery_activated = False
        recovery_attempts = 0
        recovery_success = None
        grasp_verification = None
        placement_verification = None
        verification_failures = 0
        source_grounded = None
        target_grounded = None
        source_error = None
        target_error = None
        failure_stage = None if result.success else (
            "grasp" if not result.pick_success else "placement"
        )
        failure_reason = result.failure_reason
    else:
        agent_success = bool(result.success)
        executions = [getattr(result, "execution_result", None)]
        if method == "vision_recovery":
            executions = [
                getattr(result, "initial_execution_result", None),
                getattr(result, "recovery_execution_result", None),
            ]
        grasp_success = any(
            execution is not None and execution.grasp_success
            for execution in executions
        )
        steps = int(result.total_steps)
        observation_count = int(result.observation_count)
        recovery_activated = bool(
            getattr(result, "recovery_activated", False)
        )
        recovery_attempts = int(getattr(result, "recovery_attempts", 0))
        recovery_success = (
            bool(result.success and objective.success)
            if method == "vision_recovery" and recovery_activated
            else None
        )
        grasp_verification = _verified(
            getattr(result, "grasp_verification", None)
        )
        placement_verification = _verified(
            getattr(result, "placement_verification", None)
        )
        verification_failures = _verification_failure_count(method, result)
        source_detection = getattr(result, "source_detection", None)
        target_detection = getattr(result, "target_detection", None)
        source_grounded = source_detection is not None
        target_grounded = target_detection is not None
        source_error = (
            dist(source_detection.world_position, scenario.source.initial_position)
            if source_detection is not None
            else None
        )
        target_error = (
            dist(target_detection.world_position, scenario.target.initial_position)
            if target_detection is not None
            else None
        )
        failure_stage = result.failure_stage
        failure_reason = result.failure_reason

    if not objective.success and failure_stage is None:
        failure_stage = "placement"
        failure_reason = objective.failure_reason
    return EpisodeResult(
        **_identity(method, scenario),
        agent_success=agent_success,
        task_success=bool(objective.success),
        grasp_success=grasp_success,
        placement_success=bool(objective.success),
        simulation_steps=steps,
        observation_count=observation_count,
        recovery_activated=recovery_activated,
        recovery_attempts=recovery_attempts,
        recovery_success=recovery_success,
        grasp_verification_result=grasp_verification,
        placement_verification_result=placement_verification,
        verification_failure_count=verification_failures,
        source_grounding_success=source_grounded,
        target_grounding_success=target_grounded,
        source_position_error=source_error,
        target_position_error=target_error,
        failure_stage=failure_stage,
        failure_reason=failure_reason,
        wall_clock_seconds=elapsed,
    )


def _verification_failure_count(method: str, result: Any) -> int:
    if method not in {"vision_closed_loop", "vision_recovery"}:
        return 0
    if method == "vision_closed_loop":
        values = [result.grasp_verification, result.placement_verification]
    else:
        values = [
            result.initial_grasp_verification,
            result.initial_placement_verification,
        ]
        trace = result.recovery_trace
        if trace is not None:
            values.extend(
                [
                    trace.recovery_grasp_verification,
                    trace.recovery_placement_verification,
                ]
            )
    return sum(value is not None and not value.verified for value in values)


def _verified(value: Any) -> bool | None:
    return None if value is None else bool(value.verified)


def _exception_result(
    method: str,
    scenario: Scenario,
    error: Exception,
    elapsed: float,
) -> EpisodeResult:
    detail = f"{type(error).__name__}: {error}"
    return EpisodeResult(
        **_identity(method, scenario),
        agent_success=None,
        task_success=False,
        grasp_success=None,
        placement_success=None,
        simulation_steps=None,
        observation_count=None,
        recovery_activated=None,
        recovery_attempts=None,
        recovery_success=None,
        grasp_verification_result=None,
        placement_verification_result=None,
        verification_failure_count=0,
        source_grounding_success=None,
        target_grounding_success=None,
        source_position_error=None,
        target_position_error=None,
        failure_stage="benchmark_exception",
        failure_reason=detail,
        wall_clock_seconds=elapsed,
        infrastructure_error=detail,
    )


def _identity(method: str, scenario: Scenario) -> dict[str, Any]:
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "method": method,
        "seed": scenario.seed,
        "difficulty": scenario.difficulty,
        "distractor_count": len(scenario.distractors),
        "instruction": scenario.task.instruction,
        "source_color": scenario.source.color,
        "source_type": scenario.source.object_type,
        "target_color": scenario.target.color,
        "target_type": scenario.target.object_type,
    }


def _ordered_methods(methods: Iterable[str]) -> tuple[str, ...]:
    requested = tuple(methods)
    if not requested:
        raise ValueError("At least one benchmark method is required")
    if len(set(requested)) != len(requested):
        raise ValueError("Benchmark methods must not be repeated")
    unknown = set(requested).difference(METHODS)
    if unknown:
        raise ValueError(f"Unknown benchmark methods: {sorted(unknown)}")
    return tuple(method for method in METHODS if method in requested)


def _default_run_id(start_seed: int, episodes: int, created_at: datetime) -> str:
    end_seed = start_seed + episodes - 1
    timestamp = created_at.strftime("%Y%m%dT%H%M%SZ")
    return f"phase11a-seeds-{start_seed}-{end_seed}-{timestamp}"


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _git_worktree_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(result.stdout.strip())


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _print_report(run: BenchmarkRun) -> None:
    print("method\tepisodes\ttask\tgrasp\tplacement\tmean_steps\tmean_seconds")
    for method, summary in run.summary.items():
        print(
            f"{method}\t{summary['episodes']}\t"
            f"{_percent(summary['task_success_rate'])}\t"
            f"{_percent(summary['grasp_success_rate'])}\t"
            f"{_percent(summary['placement_success_rate'])}\t"
            f"{_number(summary['mean_simulation_steps'])}\t"
            f"{_number(summary['mean_wall_clock_seconds'])}"
        )
    print("\nseed\toracle\topen_loop\tclosed_loop\trecovery")
    by_key = {(item.seed, item.method): item for item in run.episodes}
    for seed in run.metadata["seeds"]:
        cells = []
        for method in METHODS:
            item = by_key.get((seed, method))
            cells.append(
                "-" if item is None else ("S" if item.task_success else "F")
            )
        print(f"{seed}\t" + "\t".join(cells))
    failures = [item for item in run.episodes if not item.task_success]
    print("\nfailed episodes:")
    if not failures:
        print("none")
    for item in failures:
        print(
            f"seed={item.seed} method={item.method} "
            f"stage={item.failure_stage} reason={item.failure_reason} "
            f"instruction={item.instruction}"
        )
    print(
        "\ntotal wall-clock seconds: "
        f"{run.metadata['total_wall_clock_seconds']:.6f}"
    )
    for method in run.summary:
        seconds = sum(
            item.wall_clock_seconds
            for item in run.episodes
            if item.method == method
        )
        print(f"{method} wall-clock seconds: {seconds:.6f}")
    if run.output_dir is not None:
        print(f"output directory: {run.output_dir}")


def _percent(value: int | float | None) -> str:
    return "N/A" if value is None else f"{100.0 * float(value):.1f}%"


def _number(value: int | float | None) -> str:
    return "N/A" if value is None else f"{float(value):.6f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-seed", type=int, default=DEFAULT_START_SEED)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(METHODS),
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    run = run_benchmark(
        start_seed=args.start_seed,
        episodes=args.episodes,
        methods=args.methods,
        output_root=args.output_root,
        run_id=args.run_id,
    )
    _print_report(run)
    return 1 if any(item.infrastructure_error for item in run.episodes) else 0


if __name__ == "__main__":
    sys.exit(main())
