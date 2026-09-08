"""Bounded visual failure recovery layered on Phase 9."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from math import isfinite
from typing import Any

from embodied_manipulation.control.open_loop import (
    OpenLoopExecutionResult,
    StageGateResult,
    execute_open_loop_plan,
)
from embodied_manipulation.control.pick_place import PlacementEvaluation, evaluate_placement
from embodied_manipulation.language import InstructionParseError, Task, parse_instruction
from embodied_manipulation.perception import (
    LocalizationResult,
    PerceptionError,
    RGBDObservation,
    VisionPerception,
)
from embodied_manipulation.planning import (
    ManipulationPlan,
    PlanningError,
    PrimitiveKind,
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

Position = tuple[float, float, float]
XYOffset = tuple[float, float]

MAX_RECOVERY_ATTEMPTS = 1
CONTROLLED_GRASP_MISS_XY_OFFSET: XYOffset = (0.12, 0.0)
CONTROLLED_PLACEMENT_MISS_XY_OFFSET: XYOffset = (0.18, 0.0)


@dataclass(frozen=True, slots=True)
class RecoveryFaultInjection:
    """Development-only perturbation of the first plan, never the retry."""

    initial_grasp_xy_offset: XYOffset | None = None
    initial_placement_xy_offset: XYOffset | None = None

    def __post_init__(self) -> None:
        for offset, name in (
            (self.initial_grasp_xy_offset, "initial_grasp_xy_offset"),
            (self.initial_placement_xy_offset, "initial_placement_xy_offset"),
        ):
            if offset is not None and (
                len(offset) != 2 or not all(isfinite(value) for value in offset)
            ):
                raise ValueError(f"{name} must contain exactly two finite values")

    @classmethod
    def grasp_miss(cls) -> RecoveryFaultInjection:
        return cls(initial_grasp_xy_offset=CONTROLLED_GRASP_MISS_XY_OFFSET)

    @classmethod
    def placement_miss(cls) -> RecoveryFaultInjection:
        return cls(initial_placement_xy_offset=CONTROLLED_PLACEMENT_MISS_XY_OFFSET)


@dataclass(frozen=True, slots=True)
class RecoveryTrace:
    """One inspectable diagnosis/replan/retry record."""

    recovery_stage: str
    recovery_attempts: int
    initial_failure_reason: str
    diagnosis: str
    recovery_observation: RGBDObservation | None
    recovery_source_detection: LocalizationResult | None
    recovery_target_detection: LocalizationResult | None
    recovery_plan: ManipulationPlan | None
    recovery_grasp_verification: GraspVerificationResult | None
    recovery_placement_verification: PlacementVerificationResult | None
    recovered: bool
    final_failure_reason: str | None


@dataclass(frozen=True, slots=True)
class VisionRecoveryResult:
    """Phase 10 outcome with visual recovery and objective truth separated."""

    success: bool
    instruction: str
    parsed_task: Task | None = None
    initial_observation: RGBDObservation | None = None
    source_detection: LocalizationResult | None = None
    target_detection: LocalizationResult | None = None
    plan: ManipulationPlan | None = None
    initial_grasp_verification: GraspVerificationResult | None = None
    initial_placement_verification: PlacementVerificationResult | None = None
    grasp_verification: GraspVerificationResult | None = None
    placement_verification: PlacementVerificationResult | None = None
    initial_execution_result: OpenLoopExecutionResult | None = None
    recovery_execution_result: OpenLoopExecutionResult | None = None
    execution_result: OpenLoopExecutionResult | None = None
    execution_success: bool = False
    final_evaluation: PlacementEvaluation | None = None
    recovery_trace: RecoveryTrace | None = None
    recovery_activated: bool = False
    recovery_stage: str | None = None
    recovery_attempts: int = 0
    recovered: bool = False
    initial_failure_reason: str | None = None
    final_failure_reason: str | None = None
    failure_stage: str | None = None
    failure_reason: str | None = None
    total_steps: int = 0
    observation_count: int = 0
    execution_source: str = "vision"
    verification_source: str = "fresh_rgbd"
    recovery_source: str | None = None
    final_evaluation_source: str | None = None


@dataclass(slots=True)
class _Attempt:
    execution: OpenLoopExecutionResult
    grasp_observation: RGBDObservation | None
    grasp_verification: GraspVerificationResult | None
    placement_observation: RGBDObservation | None
    placement_verification: PlacementVerificationResult | None
    observation_count: int


class VisionRecoveryAgent:
    """Run Phase 9 visual gates with at most one explicit recovery attempt."""

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
        self._capture = capture_observation or _capture_rgbd
        self._vision_factory = vision_factory or VisionPerception
        self._planner = planner or TaskPlanner()
        self._executor = executor or execute_open_loop_plan
        self._evaluator = evaluator or evaluate_placement
        self._config = verification_config or VerificationConfig()

    def run(
        self,
        instruction: str,
        world: World,
        *,
        step_delay: float = 0.0,
        stage_pause: float = 0.0,
        stage_callback: StageCallback | None = None,
        fault_injection: RecoveryFaultInjection | None = None,
    ) -> VisionRecoveryResult:
        """Observe, act, verify, and perform no more than one visual retry."""
        try:
            task = parse_instruction(instruction)
        except InstructionParseError as error:
            return _failed(instruction, "parse", str(error))
        _notify(stage_callback, "parsed", task)
        if task.action != "pick_and_place":
            error = UnsupportedTaskError(
                "Phase 10 executes pick-and-place tasks only; pick-only execution "
                "is intentionally deferred"
            )
            return _failed(instruction, "planning", str(error), parsed_task=task)

        try:
            observation = self._capture(world)
        except (PerceptionError, RuntimeError, ValueError) as error:
            return _failed(
                instruction,
                "source_grounding",
                f"Initial RGB-D capture failed: {error}",
                parsed_task=task,
            )
        observation_count = 1
        _notify(stage_callback, "initial_observation_captured", observation)
        perception = self._vision_factory(observation)
        try:
            source = perception.locate(task.source)
        except PerceptionError as error:
            return _failed(
                instruction,
                "source_grounding",
                str(error),
                parsed_task=task,
                initial_observation=observation,
                observation_count=1,
            )
        _notify(stage_callback, "source_grounded", source)
        assert task.target is not None
        try:
            target = perception.locate(task.target)
        except PerceptionError as error:
            return _failed(
                instruction,
                "target_grounding",
                str(error),
                parsed_task=task,
                initial_observation=observation,
                source_detection=source,
                observation_count=1,
            )
        _notify(stage_callback, "target_grounded", target)
        common = dict(
            parsed_task=task,
            initial_observation=observation,
            source_detection=source,
            target_detection=target,
        )
        try:
            plan = self._planner.plan(task, source, target)
            if fault_injection is not None:
                plan = _inject_first_attempt_fault(plan, fault_injection)
        except (PlanningError, ValueError) as error:
            return _failed(
                instruction,
                "planning",
                str(error),
                observation_count=1,
                **common,
            )
        common["plan"] = plan
        _notify(stage_callback, "planned", plan)
        try:
            first = self._execute_attempt(
                world,
                task,
                plan,
                initial_cube_position=source.world_position,
                recovery=False,
                step_delay=step_delay,
                stage_pause=stage_pause,
                stage_callback=stage_callback,
            )
        except (RuntimeError, ValueError) as error:
            return _failed(
                instruction,
                "grasp",
                f"Execution failed before completion: {error}",
                observation_count=1,
                **common,
            )
        observation_count += first.observation_count
        _notify(stage_callback, "executed", first.execution)
        attempt_fields = dict(
            initial_grasp_verification=first.grasp_verification,
            initial_placement_verification=first.placement_verification,
            grasp_verification=first.grasp_verification,
            placement_verification=first.placement_verification,
            initial_execution_result=first.execution,
            execution_result=first.execution,
        )
        if first.grasp_verification is None:
            return _failed(
                instruction,
                first.execution.failure_stage or "grasp",
                first.execution.failure_reason
                or "Execution stopped before grasp verification",
                observation_count=observation_count,
                **common,
                **attempt_fields,
            )
        if not first.grasp_verification.verified:
            return self._recover(
                instruction,
                world,
                task,
                first,
                recovery_stage="grasp_verification",
                failed_observation=first.grasp_observation,
                diagnosis=_diagnose_grasp(first.grasp_verification, self._config),
                observation_count=observation_count,
                step_delay=step_delay,
                stage_pause=stage_pause,
                stage_callback=stage_callback,
                common=common,
            )
        if first.placement_verification is None:
            return _failed(
                instruction,
                first.execution.failure_stage or "placement",
                first.execution.failure_reason
                or "Execution stopped before placement verification",
                observation_count=observation_count,
                **common,
                **attempt_fields,
            )
        if not first.placement_verification.verified:
            return self._recover(
                instruction,
                world,
                task,
                first,
                recovery_stage="placement_verification",
                failed_observation=first.placement_observation,
                diagnosis=_diagnose_placement(first.placement_verification),
                observation_count=observation_count,
                step_delay=step_delay,
                stage_pause=stage_pause,
                stage_callback=stage_callback,
                common=common,
            )
        return self._finalize(
            instruction,
            world,
            first,
            first,
            None,
            observation_count,
            stage_callback,
            common,
        )

    def _execute_attempt(
        self,
        world: World,
        task: Task,
        plan: ManipulationPlan,
        *,
        initial_cube_position: Position,
        recovery: bool,
        step_delay: float,
        stage_pause: float,
        stage_callback: StageCallback | None,
    ) -> _Attempt:
        grasp_observation: RGBDObservation | None = None
        grasp_result: GraspVerificationResult | None = None
        placement_observation: RGBDObservation | None = None
        placement_result: PlacementVerificationResult | None = None
        count = 0
        prefix = "recovery_" if recovery else ""

        def after_grasp(
            end_effector_position: Position,
            bilateral_contact: bool,
        ) -> StageGateResult:
            nonlocal grasp_observation, grasp_result, count
            try:
                grasp_observation = self._capture(world)
                count += 1
                _notify(
                    stage_callback,
                    f"{prefix}post_grasp_observation_captured",
                    grasp_observation,
                )
                perception = self._vision_factory(grasp_observation)
                try:
                    cube = perception.locate(
                        task.source,
                        cube_center_mode="dynamic",
                    )
                except PerceptionError:
                    cube = None
                grasp_result = verify_grasp(
                    cube,
                    initial_cube_position=initial_cube_position,
                    end_effector_position=end_effector_position,
                    bilateral_contact=bilateral_contact,
                    config=self._config,
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
            _notify(stage_callback, f"{prefix}grasp_verified", grasp_result)
            return StageGateResult(grasp_result.verified, grasp_result.reason)

        def after_placement(end_effector_position: Position) -> StageGateResult:
            nonlocal placement_observation, placement_result, count
            try:
                placement_observation = self._capture(world)
                count += 1
                _notify(
                    stage_callback,
                    f"{prefix}post_placement_observation_captured",
                    placement_observation,
                )
                perception = self._vision_factory(placement_observation)
                try:
                    cube = perception.locate(
                        task.source,
                        cube_center_mode="dynamic",
                    )
                except PerceptionError:
                    cube = None
                assert task.target is not None
                try:
                    tray = perception.locate(task.target)
                except PerceptionError:
                    tray = None
                placement_result = verify_placement(
                    cube,
                    tray,
                    end_effector_position=end_effector_position,
                    config=self._config,
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
            _notify(stage_callback, f"{prefix}placement_verified", placement_result)
            return StageGateResult(placement_result.verified, placement_result.reason)

        execution = self._executor(
            world,
            plan,
            step_delay=step_delay,
            stage_pause=stage_pause,
            after_grasp=after_grasp,
            after_placement=after_placement,
        )
        return _Attempt(
            execution,
            grasp_observation,
            grasp_result,
            placement_observation,
            placement_result,
            count,
        )

    def _recover(
        self,
        instruction: str,
        world: World,
        task: Task,
        first: _Attempt,
        *,
        recovery_stage: str,
        failed_observation: RGBDObservation | None,
        diagnosis: str,
        observation_count: int,
        step_delay: float,
        stage_pause: float,
        stage_callback: StageCallback | None,
        common: dict[str, Any],
    ) -> VisionRecoveryResult:
        failed_result = (
            first.grasp_verification
            if recovery_stage == "grasp_verification"
            else first.placement_verification
        )
        assert failed_result is not None
        initial_reason = failed_result.reason
        _notify(stage_callback, "recovery_activated", recovery_stage)
        _notify(stage_callback, "failure_diagnosed", diagnosis)
        recovery_source: LocalizationResult | None = None
        recovery_target: LocalizationResult | None = None
        recovery_plan: ManipulationPlan | None = None
        replan_failure: str | None = None

        if failed_observation is None:
            replan_failure = "Failed verification produced no reusable RGB-D observation"
        else:
            perception = self._vision_factory(failed_observation)
            try:
                recovery_source = perception.locate(
                    task.source,
                    cube_center_mode="dynamic",
                )
            except PerceptionError:
                replan_failure = "Target cube could not be re-grounded for recovery"
            assert task.target is not None
            try:
                recovery_target = perception.locate(task.target)
            except PerceptionError:
                if replan_failure is None:
                    replan_failure = "Target tray could not be re-grounded for recovery"
        if recovery_source is not None:
            _notify(stage_callback, "recovery_source_grounded", recovery_source)
        if recovery_target is not None:
            _notify(stage_callback, "recovery_target_grounded", recovery_target)
        if replan_failure is None:
            assert recovery_source is not None and recovery_target is not None
            try:
                recovery_plan = self._planner.plan(
                    task,
                    recovery_source,
                    recovery_target,
                )
            except (PlanningError, ValueError) as error:
                replan_failure = f"Recovery planning failed: {error}"

        first_fields = dict(
            initial_grasp_verification=first.grasp_verification,
            initial_placement_verification=first.placement_verification,
            grasp_verification=first.grasp_verification,
            placement_verification=first.placement_verification,
            initial_execution_result=first.execution,
            execution_result=first.execution,
        )
        if recovery_plan is None:
            trace = _trace(
                recovery_stage,
                initial_reason,
                diagnosis,
                failed_observation,
                recovery_source,
                recovery_target,
                final_reason=replan_failure,
            )
            return _failed(
                instruction,
                "recovery_planning",
                replan_failure or "Recovery planning failed",
                observation_count=observation_count,
                recovery_trace=trace,
                **common,
                **first_fields,
            )

        _notify(stage_callback, "recovery_planned", recovery_plan)
        assert recovery_source is not None
        try:
            retry = self._execute_attempt(
                world,
                task,
                recovery_plan,
                initial_cube_position=recovery_source.world_position,
                recovery=True,
                step_delay=step_delay,
                stage_pause=stage_pause,
                stage_callback=stage_callback,
            )
        except (RuntimeError, ValueError) as error:
            reason = f"Recovery execution failed: {error}"
            trace = _trace(
                recovery_stage,
                initial_reason,
                diagnosis,
                failed_observation,
                recovery_source,
                recovery_target,
                recovery_plan,
                final_reason=reason,
            )
            return _failed(
                instruction,
                "recovery_execution",
                reason,
                observation_count=observation_count,
                recovery_trace=trace,
                **common,
                **first_fields,
            )
        observation_count += retry.observation_count
        _notify(stage_callback, "recovery_executed", retry.execution)

        if retry.grasp_verification is None or not retry.grasp_verification.verified:
            reason = (
                retry.grasp_verification.reason
                if retry.grasp_verification is not None
                else retry.execution.failure_reason
                or "Recovery stopped before grasp verification"
            )
            trace = _trace(
                recovery_stage,
                initial_reason,
                diagnosis,
                failed_observation,
                recovery_source,
                recovery_target,
                recovery_plan,
                retry.grasp_verification,
                retry.placement_verification,
                final_reason=reason,
            )
            return _failed(
                instruction,
                "recovery_grasp_verification",
                reason,
                observation_count=observation_count,
                recovery_trace=trace,
                recovery_execution_result=retry.execution,
                execution_result=retry.execution,
                grasp_verification=retry.grasp_verification,
                placement_verification=retry.placement_verification,
                initial_grasp_verification=first.grasp_verification,
                initial_placement_verification=first.placement_verification,
                initial_execution_result=first.execution,
                **common,
            )
        if retry.placement_verification is None:
            reason = retry.execution.failure_reason or (
                "Recovery stopped before placement verification"
            )
            trace = _trace(
                recovery_stage,
                initial_reason,
                diagnosis,
                failed_observation,
                recovery_source,
                recovery_target,
                recovery_plan,
                retry.grasp_verification,
                final_reason=reason,
            )
            return _failed(
                instruction,
                retry.execution.failure_stage or "recovery_placement",
                reason,
                observation_count=observation_count,
                recovery_trace=trace,
                recovery_execution_result=retry.execution,
                execution_result=retry.execution,
                grasp_verification=retry.grasp_verification,
                initial_grasp_verification=first.grasp_verification,
                initial_placement_verification=first.placement_verification,
                initial_execution_result=first.execution,
                **common,
            )
        trace = _trace(
            recovery_stage,
            initial_reason,
            diagnosis,
            failed_observation,
            recovery_source,
            recovery_target,
            recovery_plan,
            retry.grasp_verification,
            retry.placement_verification,
            recovered=retry.placement_verification.verified,
            final_reason=(
                None
                if retry.placement_verification.verified
                else retry.placement_verification.reason
            ),
        )
        return self._finalize(
            instruction,
            world,
            first,
            retry,
            trace,
            observation_count,
            stage_callback,
            common,
        )

    def _finalize(
        self,
        instruction: str,
        world: World,
        first: _Attempt,
        terminal: _Attempt,
        trace: RecoveryTrace | None,
        observation_count: int,
        stage_callback: StageCallback | None,
        common: dict[str, Any],
    ) -> VisionRecoveryResult:
        placement = terminal.placement_verification
        assert placement is not None
        try:
            evaluation = self._evaluator(world)
        except RuntimeError as error:
            return _failed(
                instruction,
                "placement",
                f"Final evaluation failed: {error}",
                observation_count=observation_count,
                recovery_trace=trace,
                initial_grasp_verification=first.grasp_verification,
                initial_placement_verification=first.placement_verification,
                grasp_verification=terminal.grasp_verification,
                placement_verification=placement,
                initial_execution_result=first.execution,
                recovery_execution_result=(terminal.execution if trace else None),
                execution_result=terminal.execution,
                **common,
            )
        _notify(stage_callback, "evaluated", evaluation)
        success = (
            terminal.execution.success
            and bool(
                terminal.grasp_verification
                and terminal.grasp_verification.verified
            )
            and placement.verified
            and evaluation.success
        )
        if not placement.verified:
            failure_stage = (
                "recovery_placement_verification"
                if trace
                else "placement_verification"
            )
            failure_reason = placement.reason
        elif not terminal.execution.success:
            failure_stage = terminal.execution.failure_stage
            failure_reason = terminal.execution.failure_reason
        elif not evaluation.success:
            failure_stage = "placement"
            failure_reason = evaluation.failure_reason
        else:
            failure_stage = None
            failure_reason = None
        total_steps = first.execution.total_steps
        if trace is not None:
            total_steps += terminal.execution.total_steps
        return VisionRecoveryResult(
            success=success,
            instruction=instruction,
            initial_grasp_verification=first.grasp_verification,
            initial_placement_verification=first.placement_verification,
            grasp_verification=terminal.grasp_verification,
            placement_verification=placement,
            initial_execution_result=first.execution,
            recovery_execution_result=terminal.execution if trace else None,
            execution_result=terminal.execution,
            execution_success=terminal.execution.success,
            final_evaluation=evaluation,
            recovery_trace=trace,
            recovery_activated=trace is not None,
            recovery_stage=trace.recovery_stage if trace else None,
            recovery_attempts=trace.recovery_attempts if trace else 0,
            recovered=bool(trace and trace.recovered),
            initial_failure_reason=(trace.initial_failure_reason if trace else None),
            final_failure_reason=failure_reason,
            failure_stage=failure_stage,
            failure_reason=failure_reason,
            total_steps=total_steps,
            observation_count=observation_count,
            recovery_source="failed_verification_rgbd" if trace else None,
            final_evaluation_source="simulator_ground_truth",
            **common,
        )


def _trace(
    stage: str,
    initial_reason: str,
    diagnosis: str,
    observation: RGBDObservation | None,
    source: LocalizationResult | None,
    target: LocalizationResult | None,
    plan: ManipulationPlan | None = None,
    grasp: GraspVerificationResult | None = None,
    placement: PlacementVerificationResult | None = None,
    *,
    recovered: bool = False,
    final_reason: str | None,
) -> RecoveryTrace:
    return RecoveryTrace(
        recovery_stage=stage,
        recovery_attempts=MAX_RECOVERY_ATTEMPTS,
        initial_failure_reason=initial_reason,
        diagnosis=diagnosis,
        recovery_observation=observation,
        recovery_source_detection=source,
        recovery_target_detection=target,
        recovery_plan=plan,
        recovery_grasp_verification=grasp,
        recovery_placement_verification=placement,
        recovered=recovered,
        final_failure_reason=final_reason,
    )


def _diagnose_grasp(
    result: GraspVerificationResult,
    config: VerificationConfig,
) -> str:
    if not result.cube_detected:
        return "target_cube_not_detected"
    if (
        result.cube_height_above_table is not None
        and result.cube_height_above_table < config.minimum_lift_clearance
    ):
        return "cube_not_lifted"
    if (
        result.cube_to_ee_distance is not None
        and result.cube_to_ee_distance > config.maximum_cube_to_ee_distance
    ):
        return "cube_not_near_end_effector"
    return "grasp_verification_failed"


def _diagnose_placement(result: PlacementVerificationResult) -> str:
    if not result.cube_detected:
        return "cube_not_detected"
    if not result.tray_detected:
        return "tray_not_detected"
    if not result.cube_inside_tray:
        return "cube_outside_tray"
    if not result.cube_released:
        return "cube_still_near_gripper"
    return "placement_verification_failed"


def _inject_first_attempt_fault(
    plan: ManipulationPlan,
    fault: RecoveryFaultInjection,
) -> ManipulationPlan:
    grasp_offset = fault.initial_grasp_xy_offset
    placement_offset = fault.initial_placement_xy_offset

    def shifted(position: Position, offset: XYOffset | None) -> Position:
        if offset is None:
            return position
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


def _failed(
    instruction: str,
    stage: str,
    reason: str,
    *,
    recovery_trace: RecoveryTrace | None = None,
    initial_execution_result: OpenLoopExecutionResult | None = None,
    recovery_execution_result: OpenLoopExecutionResult | None = None,
    execution_result: OpenLoopExecutionResult | None = None,
    **fields: Any,
) -> VisionRecoveryResult:
    total_steps = 0
    if initial_execution_result is not None:
        total_steps += initial_execution_result.total_steps
    if recovery_execution_result is not None:
        total_steps += recovery_execution_result.total_steps
    trace = recovery_trace
    return VisionRecoveryResult(
        success=False,
        instruction=instruction,
        initial_execution_result=initial_execution_result,
        recovery_execution_result=recovery_execution_result,
        execution_result=execution_result,
        execution_success=bool(execution_result and execution_result.success),
        recovery_trace=trace,
        recovery_activated=trace is not None,
        recovery_stage=trace.recovery_stage if trace else None,
        recovery_attempts=trace.recovery_attempts if trace else 0,
        recovered=False,
        initial_failure_reason=trace.initial_failure_reason if trace else None,
        final_failure_reason=reason,
        failure_stage=stage,
        failure_reason=reason,
        total_steps=total_steps,
        recovery_source="failed_verification_rgbd" if trace else None,
        **fields,
    )
