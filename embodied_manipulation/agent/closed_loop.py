"""Stage-level RGB-D verification around the fixed Phase 8 plan."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from embodied_manipulation.control.open_loop import (
    OpenLoopExecutionResult,
    StageGateResult,
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

from .open_loop import StageCallback, _capture_rgbd, _notify
from .verification import (
    GraspVerificationResult,
    PlacementVerificationResult,
    VerificationConfig,
    verify_grasp,
    verify_placement,
)


@dataclass(frozen=True, slots=True)
class VisionClosedLoopResult:
    """Phase 9 result with visual and objective outcomes kept separate."""

    success: bool
    instruction: str
    parsed_task: Task | None
    initial_observation: RGBDObservation | None
    source_detection: LocalizationResult | None
    target_detection: LocalizationResult | None
    plan: ManipulationPlan | None
    grasp_verification: GraspVerificationResult | None
    placement_verification: PlacementVerificationResult | None
    execution_result: OpenLoopExecutionResult | None
    execution_success: bool
    final_evaluation: PlacementEvaluation | None
    failure_stage: str | None
    failure_reason: str | None
    total_steps: int
    observation_count: int
    execution_source: str = "vision"
    verification_source: str = "fresh_rgbd"
    final_evaluation_source: str | None = None


class VisionClosedLoopAgent:
    """Execute one immutable vision plan with two visual continue/stop gates."""

    def __init__(
        self,
        *,
        capture_observation: Callable[[World], RGBDObservation] | None = None,
        vision_factory: Callable[[RGBDObservation], Any] | None = None,
        planner: TaskPlanner | None = None,
        executor: Callable[..., OpenLoopExecutionResult] | None = None,
        evaluator: Callable[[World], PlacementEvaluation] | None = None,
        verification_config: VerificationConfig | None = None,
    ) -> None:
        self._capture_observation = capture_observation or _capture_rgbd
        self._vision_factory = vision_factory or VisionPerception
        self._planner = planner or TaskPlanner()
        self._executor = executor or execute_open_loop_plan
        self._evaluator = evaluator or evaluate_placement
        self._verification_config = verification_config or VerificationConfig()

    def run(
        self,
        instruction: str,
        world: World,
        *,
        step_delay: float = 0.0,
        stage_pause: float = 0.0,
        stage_callback: StageCallback | None = None,
    ) -> VisionClosedLoopResult:
        """Observe, plan once, verify grasp/place visually, then evaluate."""
        task: Task | None = None
        initial_observation: RGBDObservation | None = None
        source: LocalizationResult | None = None
        target: LocalizationResult | None = None
        plan: ManipulationPlan | None = None
        grasp_result: GraspVerificationResult | None = None
        placement_result: PlacementVerificationResult | None = None
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
                "Phase 9 executes pick-and-place tasks only; pick-only execution "
                "is intentionally deferred"
            )
            return _failure(
                instruction,
                parsed_task=task,
                failure_stage="planning",
                failure_reason=str(error),
            )

        try:
            initial_observation = self._capture_observation(world)
            observation_count = 1
        except (PerceptionError, RuntimeError, ValueError) as error:
            return _failure(
                instruction,
                parsed_task=task,
                failure_stage="source_grounding",
                failure_reason=f"Initial RGB-D capture failed: {error}",
            )
        _notify(
            stage_callback,
            "initial_observation_captured",
            initial_observation,
        )

        perception = self._vision_factory(initial_observation)
        try:
            source = perception.locate(task.source)
        except PerceptionError as error:
            return _failure(
                instruction,
                parsed_task=task,
                initial_observation=initial_observation,
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
                initial_observation=initial_observation,
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
                initial_observation=initial_observation,
                source_detection=source,
                target_detection=target,
                failure_stage="planning",
                failure_reason=str(error),
                observation_count=observation_count,
            )
        _notify(stage_callback, "planned", plan)

        def after_grasp(
            end_effector_position: tuple[float, float, float],
            bilateral_contact: bool,
        ) -> StageGateResult:
            nonlocal observation_count, grasp_result
            try:
                observation = self._capture_observation(world)
                observation_count += 1
                _notify(
                    stage_callback,
                    "post_grasp_observation_captured",
                    observation,
                )
                verification_perception = self._vision_factory(observation)
                try:
                    cube = verification_perception.locate(
                        task.source,
                        cube_center_mode="dynamic",
                    )
                except PerceptionError:
                    cube = None
                grasp_result = verify_grasp(
                    cube,
                    initial_cube_position=source.world_position,
                    end_effector_position=end_effector_position,
                    bilateral_contact=bilateral_contact,
                    config=self._verification_config,
                )
            except (PerceptionError, RuntimeError, ValueError) as error:
                grasp_result = GraspVerificationResult(
                    verified=False,
                    cube_detected=False,
                    estimated_cube_position=None,
                    end_effector_position=end_effector_position,
                    cube_height_above_table=None,
                    cube_to_ee_distance=None,
                    visual_displacement_from_initial=None,
                    bilateral_contact=bilateral_contact,
                    reason=f"Post-grasp RGB-D verification failed: {error}",
                )
            _notify(stage_callback, "grasp_verified", grasp_result)
            return StageGateResult(grasp_result.verified, grasp_result.reason)

        def after_placement(
            end_effector_position: tuple[float, float, float],
        ) -> StageGateResult:
            nonlocal observation_count, placement_result
            try:
                observation = self._capture_observation(world)
                observation_count += 1
                _notify(
                    stage_callback,
                    "post_placement_observation_captured",
                    observation,
                )
                verification_perception = self._vision_factory(observation)
                try:
                    cube = verification_perception.locate(
                        task.source,
                        cube_center_mode="dynamic",
                    )
                except PerceptionError:
                    cube = None
                try:
                    tray = verification_perception.locate(task.target)
                except PerceptionError:
                    tray = None
                placement_result = verify_placement(
                    cube,
                    tray,
                    end_effector_position=end_effector_position,
                    config=self._verification_config,
                )
            except (PerceptionError, RuntimeError, ValueError) as error:
                placement_result = PlacementVerificationResult(
                    verified=False,
                    cube_detected=False,
                    tray_detected=False,
                    estimated_cube_position=None,
                    estimated_tray_position=None,
                    end_effector_position=end_effector_position,
                    cube_inside_tray=False,
                    cube_near_tray_floor=False,
                    cube_released=False,
                    cube_to_ee_distance=None,
                    reason=f"Post-placement RGB-D verification failed: {error}",
                )
            _notify(stage_callback, "placement_verified", placement_result)
            return StageGateResult(
                placement_result.verified,
                placement_result.reason,
            )

        try:
            execution = self._executor(
                world,
                plan,
                step_delay=step_delay,
                stage_pause=stage_pause,
                after_grasp=after_grasp,
                after_placement=after_placement,
            )
        except (RuntimeError, ValueError) as error:
            return _failure(
                instruction,
                parsed_task=task,
                initial_observation=initial_observation,
                source_detection=source,
                target_detection=target,
                plan=plan,
                grasp_verification=grasp_result,
                placement_verification=placement_result,
                failure_stage="grasp",
                failure_reason=f"Execution failed before completion: {error}",
                observation_count=observation_count,
            )
        _notify(stage_callback, "executed", execution)

        if grasp_result is None:
            return _failure(
                instruction,
                parsed_task=task,
                initial_observation=initial_observation,
                source_detection=source,
                target_detection=target,
                plan=plan,
                execution_result=execution,
                failure_stage=execution.failure_stage or "grasp",
                failure_reason=execution.failure_reason
                or "Execution stopped before grasp verification",
                observation_count=observation_count,
            )
        if not grasp_result.verified:
            return _failure(
                instruction,
                parsed_task=task,
                initial_observation=initial_observation,
                source_detection=source,
                target_detection=target,
                plan=plan,
                grasp_verification=grasp_result,
                execution_result=execution,
                failure_stage="grasp_verification",
                failure_reason=grasp_result.reason,
                observation_count=observation_count,
            )
        if placement_result is None:
            return _failure(
                instruction,
                parsed_task=task,
                initial_observation=initial_observation,
                source_detection=source,
                target_detection=target,
                plan=plan,
                grasp_verification=grasp_result,
                execution_result=execution,
                failure_stage=execution.failure_stage or "placement",
                failure_reason=execution.failure_reason
                or "Execution stopped before placement verification",
                observation_count=observation_count,
            )

        # Objective truth is read only after the third observation and visual
        # placement decision. It cannot modify the immutable initial plan.
        try:
            evaluation = self._evaluator(world)
        except RuntimeError as error:
            return _failure(
                instruction,
                parsed_task=task,
                initial_observation=initial_observation,
                source_detection=source,
                target_detection=target,
                plan=plan,
                grasp_verification=grasp_result,
                placement_verification=placement_result,
                execution_result=execution,
                failure_stage="placement",
                failure_reason=f"Final evaluation failed: {error}",
                observation_count=observation_count,
            )
        _notify(stage_callback, "evaluated", evaluation)

        success = (
            execution.success
            and grasp_result.verified
            and placement_result.verified
            and evaluation.success
        )
        if not placement_result.verified:
            failure_stage = "placement_verification"
            failure_reason = placement_result.reason
        elif not execution.success:
            failure_stage = execution.failure_stage
            failure_reason = execution.failure_reason
        elif not evaluation.success:
            failure_stage = "placement"
            failure_reason = evaluation.failure_reason
        else:
            failure_stage = None
            failure_reason = None
        return VisionClosedLoopResult(
            success=success,
            instruction=instruction,
            parsed_task=task,
            initial_observation=initial_observation,
            source_detection=source,
            target_detection=target,
            plan=plan,
            grasp_verification=grasp_result,
            placement_verification=placement_result,
            execution_result=execution,
            execution_success=execution.success,
            final_evaluation=evaluation,
            failure_stage=failure_stage,
            failure_reason=failure_reason,
            total_steps=execution.total_steps,
            observation_count=observation_count,
            final_evaluation_source="simulator_ground_truth",
        )


def _failure(
    instruction: str,
    *,
    parsed_task: Task | None = None,
    initial_observation: RGBDObservation | None = None,
    source_detection: LocalizationResult | None = None,
    target_detection: LocalizationResult | None = None,
    plan: ManipulationPlan | None = None,
    grasp_verification: GraspVerificationResult | None = None,
    placement_verification: PlacementVerificationResult | None = None,
    execution_result: OpenLoopExecutionResult | None = None,
    failure_stage: str,
    failure_reason: str,
    observation_count: int = 0,
) -> VisionClosedLoopResult:
    return VisionClosedLoopResult(
        success=False,
        instruction=instruction,
        parsed_task=parsed_task,
        initial_observation=initial_observation,
        source_detection=source_detection,
        target_detection=target_detection,
        plan=plan,
        grasp_verification=grasp_verification,
        placement_verification=placement_verification,
        execution_result=execution_result,
        execution_success=bool(execution_result and execution_result.success),
        final_evaluation=None,
        failure_stage=failure_stage,
        failure_reason=failure_reason,
        total_steps=execution_result.total_steps if execution_result else 0,
        observation_count=observation_count,
    )
