"""Sequential, reproducible clean and Phase 14B perturbation benchmark runner."""

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
from .perturbations import (
    PERTURBATION_NAMES,
    BenchmarkPerturbation,
    FirstAttemptPerturbingPlanner,
    ordered_perturbations,
)
from .scenarios import Scenario, generate_scenario

DEFAULT_START_SEED = 0
DEFAULT_EPISODES = 12
DEFAULT_DIFFICULTY = "basic"
DEFAULT_OUTPUT_ROOT = Path("outputs/benchmarks")

WorldFactory = Callable[[], Any]
MethodRunner = Callable[[str, Scenario, Any], Any]
EpisodeRunner = Callable[..., EpisodeResult]


@dataclass(frozen=True, slots=True)
class BenchmarkRun:
    """In-memory benchmark records plus optional persisted artifact path."""

    episodes: tuple[EpisodeResult, ...]
    summary: Mapping[str, Mapping[str, int | float | None]]
    perturbation_summary: Mapping[
        str,
        Mapping[str, Mapping[str, int | float | None]],
    ]
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
    perturbation: BenchmarkPerturbation | None = None,
) -> EpisodeResult:
    """Run one method in a fresh world and normalize its result.

    Scenario metadata is ground truth available only after method execution for
    benchmark diagnostics. It is never passed into a vision agent's planner.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown benchmark method: {method}")
    make_world = world_factory or (lambda: World(gui=False))
    started = clock()
    try:
        with make_world() as world:
            world.reset_from_scenario(scenario)
            raw_result = (
                method_runner(method, scenario, world)
                if method_runner is not None
                else _run_method(method, scenario, world, perturbation)
            )
            objective = objective_evaluator(world)
    except Exception as error:
        elapsed = max(0.0, clock() - started)
        return _exception_result(method, scenario, error, elapsed, perturbation)
    elapsed = max(0.0, clock() - started)
    return _normalize_result(
        method,
        scenario,
        raw_result,
        objective,
        elapsed,
        perturbation,
    )


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
    perturbations: Iterable[str | BenchmarkPerturbation] | None = None,
) -> BenchmarkRun:
    """Run deterministic episodes sequentially without benchmark-level retries."""
    if not isinstance(start_seed, int) or isinstance(start_seed, bool):
        raise ValueError("start_seed must be an integer")
    if not isinstance(episodes, int) or isinstance(episodes, bool) or episodes <= 0:
        raise ValueError("episodes must be a positive integer")
    selected_methods = _ordered_methods(methods)
    selected_perturbations = ordered_perturbations(perturbations)
    conditions: tuple[BenchmarkPerturbation | None, ...] = (
        selected_perturbations if selected_perturbations else (None,)
    )
    seeds = tuple(range(start_seed, start_seed + episodes))
    records: list[EpisodeResult] = []
    started = clock()
    for perturbation in conditions:
        for seed in seeds:
            scenario = generate_scenario(
                seed,
                difficulty,
                distractor_count=seed % 3,
            )
            for method in selected_methods:
                try:
                    if episode_runner is None:
                        record = run_episode(
                            method,
                            scenario,
                            perturbation=perturbation,
                        )
                    elif perturbation is None:
                        record = episode_runner(method, scenario)
                    else:
                        record = episode_runner(method, scenario, perturbation)
                except Exception as error:
                    record = _exception_result(
                        method,
                        scenario,
                        error,
                        0.0,
                        perturbation,
                    )
                records.append(record)
    total_runtime = max(0.0, clock() - started)
    created_at = datetime.now(UTC)
    metadata = {
        "benchmark_version": BENCHMARK_VERSION,
        "created_at_utc": created_at.isoformat(),
        "start_seed": start_seed,
        "seeds": list(seeds),
        "methods": list(selected_methods),
        "perturbations": [
            perturbation.to_dict()
            for perturbation in selected_perturbations
        ],
        "perturbation_protocol": (
            None
            if not selected_perturbations
            else {
                "condition_order": [
                    perturbation.name
                    for perturbation in selected_perturbations
                ],
                "application": "first relevant manipulation attempt only",
                "recovery_attempts_perturbed": False,
                "benchmark_level_retries": 0,
            }
        ),
        "scenario_difficulty": difficulty,
        "distractor_rule": "seed % 3",
        "episode_order": (
            "perturbation_major_seed_major_method_minor"
            if selected_perturbations
            else "seed_major_method_minor"
        ),
        "scenario_count": len(seeds),
        "condition_count": len(selected_perturbations),
        "scenario_condition_count": len(seeds) * len(conditions),
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
        perturbation_summary=_summarize_perturbations(records),
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
    summary_payload: dict[str, Any] = {
        "benchmark_version": BENCHMARK_VERSION,
        "methods": run.summary,
    }
    if run.perturbation_summary:
        summary_payload["perturbations"] = run.perturbation_summary
    _write_json(destination / "summary.json", summary_payload)
    _write_json(destination / "metadata.json", dict(run.metadata))
    return destination


def _run_method(
    method: str,
    scenario: Scenario,
    world: World,
    perturbation: BenchmarkPerturbation | None = None,
) -> Any:
    if method == "oracle_scripted":
        if perturbation is None:
            return oracle_pick_place(world)
        return oracle_pick_place(
            world,
            grasp_xy_offset=perturbation.grasp_xy_offset,
            placement_xy_offset=perturbation.placement_xy_offset,
        )
    instruction = scenario.task.instruction
    planner = (
        FirstAttemptPerturbingPlanner(perturbation)
        if perturbation is not None
        else None
    )
    if method == "vision_open_loop":
        agent = VisionOpenLoopAgent() if planner is None else VisionOpenLoopAgent(
            planner=planner
        )
        return agent.run(instruction, world)
    if method == "vision_closed_loop":
        agent = (
            VisionClosedLoopAgent()
            if planner is None
            else VisionClosedLoopAgent(planner=planner)
        )
        return agent.run(instruction, world)
    if method == "vision_recovery":
        # The benchmark planner shifts only its first plan. The agent's fresh
        # recovery replan is nominal, and RecoveryFaultInjection is never used.
        agent = (
            VisionRecoveryAgent()
            if planner is None
            else VisionRecoveryAgent(planner=planner)
        )
        return agent.run(instruction, world)
    raise ValueError(f"Unknown benchmark method: {method}")


def _normalize_result(
    method: str,
    scenario: Scenario,
    result: Any,
    objective: PlacementEvaluation,
    elapsed: float,
    perturbation: BenchmarkPerturbation | None = None,
) -> EpisodeResult:
    if method == "oracle_scripted":
        agent_success = bool(result.success)
        grasp_success = bool(result.pick_success)
        steps = int(result.total_steps)
        observation_count = 0
        recovery_activated = False
        recovery_attempts = 0
        recovery_success = None
        recovery_stage = None
        recovered = None
        recovery_final_verification = None
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
        recovery_stage = (
            getattr(result, "recovery_stage", None)
            if method == "vision_recovery"
            else None
        )
        recovered = (
            bool(getattr(result, "recovered", False))
            if method == "vision_recovery"
            else None
        )
        recovery_final_verification = (
            _recovery_final_verification(result)
            if method == "vision_recovery"
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
        **_identity(method, scenario, perturbation),
        agent_success=agent_success,
        task_success=bool(objective.success),
        grasp_success=grasp_success,
        placement_success=bool(objective.success),
        simulation_steps=steps,
        observation_count=observation_count,
        recovery_activated=recovery_activated,
        recovery_attempts=recovery_attempts,
        recovery_success=recovery_success,
        recovery_stage=recovery_stage,
        recovered=recovered,
        recovery_final_verification_result=recovery_final_verification,
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


def _recovery_final_verification(result: Any) -> bool | None:
    trace = getattr(result, "recovery_trace", None)
    if trace is None:
        return None
    for value in (
        trace.recovery_placement_verification,
        trace.recovery_grasp_verification,
    ):
        if value is not None:
            return bool(value.verified)
    return None


def _exception_result(
    method: str,
    scenario: Scenario,
    error: Exception,
    elapsed: float,
    perturbation: BenchmarkPerturbation | None = None,
) -> EpisodeResult:
    detail = f"{type(error).__name__}: {error}"
    return EpisodeResult(
        **_identity(method, scenario, perturbation),
        agent_success=None,
        task_success=False,
        grasp_success=None,
        placement_success=None,
        simulation_steps=None,
        observation_count=None,
        recovery_activated=None,
        recovery_attempts=None,
        recovery_success=None,
        recovery_stage=None,
        recovered=None,
        recovery_final_verification_result=None,
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


def _identity(
    method: str,
    scenario: Scenario,
    perturbation: BenchmarkPerturbation | None = None,
) -> dict[str, Any]:
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
        "perturbation_name": (
            None if perturbation is None else perturbation.name
        ),
        "perturbation_stage": (
            None if perturbation is None else perturbation.stage
        ),
        "perturbation_axis": (
            None if perturbation is None else perturbation.axis
        ),
        "perturbation_offset_m": (
            None if perturbation is None else perturbation.offset_m
        ),
        "perturbation_first_attempt_only": (
            None if perturbation is None else perturbation.first_attempt_only
        ),
    }


def _summarize_perturbations(
    records: Iterable[EpisodeResult],
) -> dict[str, dict[str, dict[str, int | float | None]]]:
    names = tuple(
        dict.fromkeys(
            item.perturbation_name
            for item in records
            if item.perturbation_name is not None
        )
    )
    summaries: dict[str, dict[str, dict[str, int | float | None]]] = {}
    for name in names:
        grouped = [item for item in records if item.perturbation_name == name]
        method_summaries = summarize_episodes(grouped)
        for method, summary in method_summaries.items():
            method_records = [item for item in grouped if item.method == method]
            summary["verification_failures"] = sum(
                item.verification_failure_count for item in method_records
            )
            summary["recovery_activations"] = sum(
                item.recovery_activated is True for item in method_records
            )
            summary["successful_recoveries"] = sum(
                item.recovery_success is True for item in method_records
            )
            summary["intended_verification_failures"] = sum(
                (
                    item.failure_stage == f"{item.perturbation_stage}_verification"
                    or item.recovery_stage
                    == f"{item.perturbation_stage}_verification"
                )
                for item in method_records
            )
        summaries[name] = method_summaries
    return summaries


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
    return f"{BENCHMARK_VERSION}-seeds-{start_seed}-{end_seed}-{timestamp}"


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
    conditions = (
        tuple(run.perturbation_summary)
        if run.perturbation_summary
        else (None,)
    )
    by_key = {
        (item.perturbation_name, item.seed, item.method): item
        for item in run.episodes
    }
    for condition in conditions:
        if condition is not None:
            print(f"\nperturbation: {condition}")
        print("seed\toracle\topen_loop\tclosed_loop\trecovery")
        for seed in run.metadata["seeds"]:
            cells = []
            for method in METHODS:
                item = by_key.get((condition, seed, method))
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
            f"perturbation={item.perturbation_name} "
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
    parser.add_argument(
        "--perturbations",
        nargs="+",
        choices=PERTURBATION_NAMES,
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    run = run_benchmark(
        start_seed=args.start_seed,
        episodes=args.episodes,
        methods=args.methods,
        perturbations=args.perturbations,
        output_root=args.output_root,
        run_id=args.run_id,
    )
    _print_report(run)
    return 1 if any(item.infrastructure_error for item in run.episodes) else 0


if __name__ == "__main__":
    sys.exit(main())
