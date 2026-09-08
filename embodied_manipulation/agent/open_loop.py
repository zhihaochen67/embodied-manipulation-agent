"""Observe-once, vision-conditioned open-loop manipulation agent."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from embodied_manipulation.control.open_loop import (
    OpenLoopExecutionResult,
    execute_open_loop_plan,
)
from embodied_manipulation.control.pick_place import (
    PlacementEvaluation,
    evaluate_placement,
)
from embodied_manipulation.language import (
    InstructionParseError,
    Task,
    parse_instruction,
)
from embodied_manipulation.perception import (
    LocalizationResult,
    Perception,
    PerceptionError,
    RGBDObservation,
    VisionPerception,
)
from embodied_manipulation.planning import (
    ManipulationPlan,
    PlanningError,
    TaskPlanner,
    UnsupportedTaskError,
)
from embodied_manipulation.simulation import World

StageCallback = Callable[[str, object], None]


@dataclass(frozen=True, slots=True)
class VisionOpenLoopResult:
    """Compact result separating vision execution from oracle evaluation."""

    success: bool
    instruction: str
    parsed_task: Task | None
    source_detection: LocalizationResult | None
    target_detection: LocalizationResult | None
    plan: ManipulationPlan | None
    execution_result: OpenLoopExecutionResult | None
    execution_success: bool
    final_evaluation: PlacementEvaluation | None
    failure_stage: str | None
    failure_reason: str | None
    total_steps: int
    observation_count: int
    execution_source: str = "vision"
    final_evaluation_source: str | None = None


class VisionOpenLoopAgent:
    """First complete language-to-physical-action agent for the project."""

    def __init__(
        self,
        *,
        capture_observation: Callable[[World], RGBDObservation] | None = None,
        vision_factory: Callable[[RGBDObservation], Perception] | None = None,
        planner: TaskPlanner | None = None,
        executor: Callable[..., OpenLoopExecutionResult] | None = None,
        evaluator: Callable[[World], PlacementEvaluation] | None = None,
    ) -> None:
        # Narrow injection points make single-observation and oracle-integrity
        # behavior directly testable without introducing a generic framework.
        self._capture_observation = capture_observation or _capture_rgbd
        self._vision_factory = vision_factory or VisionPerception
        self._planner = planner or TaskPlanner()
        self._executor = executor or execute_open_loop_plan
        self._evaluator = evaluator or evaluate_placement

    def run(
        self,
        instruction: str,
        world: World,
        *,
        step_delay: float = 0.0,
        stage_pause: float = 0.0,
        stage_callback: StageCallback | None = None,
    ) -> VisionOpenLoopResult:
        """Parse, observe once, ground, plan, act, then evaluate once."""
        task: Task | None = None
        source: LocalizationResult | None = None
        target: LocalizationResult | None = None
        plan: ManipulationPlan | None = None
        observation_count = 0

        try:
            task = parse_instruction(instruction)
        except InstructionParseError as error:
            return _failure(
                instruction,
                failure_stage="parse",
                failure_reason=str(error),
            )
        _notify(stage_callback, "parsed", task)

        if task.action != "pick_and_place":
            error = UnsupportedTaskError(
                "Phase 8 executes pick-and-place tasks only; pick-only execution "
                "is intentionally deferred"
            )
            return _failure(
                instruction,
                parsed_task=task,
                failure_stage="planning",
                failure_reason=str(error),
            )

        try:
            observation = self._capture_observation(world)
            observation_count = 1
        except (PerceptionError, RuntimeError, ValueError) as error:
            return _failure(
                instruction,
                parsed_task=task,
                failure_stage="source_grounding",
                failure_reason=f"Initial RGB-D capture failed: {error}",
            )
        _notify(stage_callback, "observation_captured", observation)

        perception = self._vision_factory(observation)
        try:
            source = perception.locate(task.source)
        except PerceptionError as error:
            return _failure(
                instruction,
                parsed_task=task,
                failure_stage="source_grounding",
                failure_reason=str(error),
                observation_count=observation_count,
            )
        _notify(stage_callback, "source_grounded", source)

        assert task.target is not None
        try:
            target = perception.locate(task.target)
        except PerceptionError as error:
            return _failure(
                instruction,
                parsed_task=task,
                source_detection=source,
                failure_stage="target_grounding",
                failure_reason=str(error),
                observation_count=observation_count,
            )
        _notify(stage_callback, "target_grounded", target)

        try:
            plan = self._planner.plan(task, source, target)
        except (PlanningError, ValueError) as error:
            return _failure(
                instruction,
                parsed_task=task,
                source_detection=source,
                target_detection=target,
                failure_stage="planning",
                failure_reason=str(error),
                observation_count=observation_count,
            )
        _notify(stage_callback, "planned", plan)

        try:
            execution = self._executor(
                world,
                plan,
                step_delay=step_delay,
                stage_pause=stage_pause,
            )
        except (RuntimeError, ValueError) as error:
            return _failure(
                instruction,
                parsed_task=task,
                source_detection=source,
                target_detection=target,
                plan=plan,
                failure_stage="grasp",
                failure_reason=f"Execution failed before completion: {error}",
                observation_count=observation_count,
            )
        _notify(stage_callback, "executed", execution)

        # This is the sole ground-truth stage. It runs after the executor has
        # returned and therefore cannot alter any action target or trajectory.
        try:
            evaluation = self._evaluator(world)
        except RuntimeError as error:
            return _failure(
                instruction,
                parsed_task=task,
                source_detection=source,
                target_detection=target,
                plan=plan,
                execution_result=execution,
                failure_stage=execution.failure_stage or "placement",
                failure_reason=f"Final evaluation failed: {error}",
                observation_count=observation_count,
            )
        _notify(stage_callback, "evaluated", evaluation)

        success = execution.success and evaluation.success
        if not execution.success:
            failure_stage = execution.failure_stage
            failure_reason = execution.failure_reason
        elif not evaluation.success:
            failure_stage = "placement"
            failure_reason = evaluation.failure_reason
        else:
            failure_stage = None
            failure_reason = None
        return VisionOpenLoopResult(
            success=success,
            instruction=instruction,
            parsed_task=task,
            source_detection=source,
            target_detection=target,
            plan=plan,
            execution_result=execution,
            execution_success=execution.success,
            final_evaluation=evaluation,
            failure_stage=failure_stage,
            failure_reason=failure_reason,
            total_steps=execution.total_steps,
            observation_count=observation_count,
            final_evaluation_source="simulator_ground_truth",
        )


def _capture_rgbd(world: World) -> RGBDObservation:
    if world.scene is None:
        raise RuntimeError("World must be reset before RGB-D capture")
    return RGBDObservation.capture(world.camera, world.client_id)


def _notify(
    callback: StageCallback | None,
    stage: str,
    value: Any,
) -> None:
    if callback is not None:
        callback(stage, value)


def _failure(
    instruction: str,
    *,
    parsed_task: Task | None = None,
    source_detection: LocalizationResult | None = None,
    target_detection: LocalizationResult | None = None,
    plan: ManipulationPlan | None = None,
    execution_result: OpenLoopExecutionResult | None = None,
    failure_stage: str,
    failure_reason: str,
    observation_count: int = 0,
) -> VisionOpenLoopResult:
    return VisionOpenLoopResult(
        success=False,
        instruction=instruction,
        parsed_task=parsed_task,
        source_detection=source_detection,
        target_detection=target_detection,
        plan=plan,
        execution_result=execution_result,
        execution_success=bool(execution_result and execution_result.success),
        final_evaluation=None,
        failure_stage=failure_stage,
        failure_reason=failure_reason,
        total_steps=execution_result.total_steps if execution_result else 0,
        observation_count=observation_count,
    )
