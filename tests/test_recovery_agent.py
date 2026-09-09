"""Phase 10 bounded visual recovery regression tests."""

from __future__ import annotations

from dataclasses import replace

import pybullet
import pytest

from embodied_manipulation.agent import (
    MAX_RECOVERY_ATTEMPTS,
    PREVERIFICATION_RECOVERY_ELIGIBLE_FAILURE_CODES,
    RecoveryFaultInjection,
    VisionRecoveryAgent,
    is_preverification_recovery_eligible,
)
from embodied_manipulation.benchmark import generate_scenario
from embodied_manipulation.control import (
    IK_RESIDUAL_EXCEEDS_EXECUTION_TOLERANCE,
    GripperResult,
    ReachResult,
)
from embodied_manipulation.control.open_loop import OpenLoopExecutionResult
from embodied_manipulation.control.pick_place import PlacementEvaluation
from embodied_manipulation.language import parse_instruction
from embodied_manipulation.perception import LocalizationResult, PerceptionError
from embodied_manipulation.planning import PlanningError, TaskPlanner
from embodied_manipulation.simulation import World


def _evaluation_success() -> PlacementEvaluation:
    return PlacementEvaluation(
        success=True,
        cube_position=(0.50, 0.28, 0.033),
        cube_linear_velocity=(0.0, 0.0, 0.0),
        cube_angular_velocity=(0.0, 0.0, 0.0),
        cube_inside_tray=True,
        cube_near_floor=True,
        cube_released=True,
        stable=True,
        cube_to_tray_center_distance=0.0,
    )


def _execution(
    success: bool,
    *,
    grasp_success: bool,
    placement_success: bool,
    failure_stage: str | None = None,
    failure_reason: str | None = None,
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
        failure_stage=failure_stage,
        failure_reason=failure_reason,
    )


def _ik_failure_execution(
    failure_code: str = IK_RESIDUAL_EXCEEDS_EXECUTION_TOLERANCE,
) -> OpenLoopExecutionResult:
    target = (0.55, 0.10, 0.20)
    failed_reach = ReachResult(
        success=False,
        steps=0,
        target_position=target,
        final_position=(0.50, 0.00, 0.40),
        final_orientation=(1.0, 0.0, 0.0, 0.0),
        position_error=0.05,
        target_joint_positions=(0.0,) * 7,
        final_joint_positions=(0.0,) * 7,
        failure_reason="Structured IK failure",
        failure_code=failure_code,
        ik_position_residual=0.005379,
        required_position_tolerance=0.004,
        ik_joint_solution_valid=(
            failure_code == IK_RESIDUAL_EXCEEDS_EXECUTION_TOLERANCE
        ),
    )
    return replace(
        _execution(
            False,
            grasp_success=False,
            placement_success=False,
            failure_stage="grasp",
            failure_reason="Grasp approach failed: Structured IK failure",
        ),
        failed_reach_result=failed_reach,
    )


def _gripper_timeout_execution(
    timeout_diagnostic: str,
) -> OpenLoopExecutionResult:
    close_result = GripperResult(
        success=False,
        steps=240,
        target_opening_width=0.0,
        final_opening_width=0.05,
        target_finger_positions=(0.0, 0.0),
        final_finger_positions=(0.025, 0.025),
        max_finger_position_error=0.025,
        reached_target=False,
        stalled=False,
        failure_reason="Timed out after 240 simulation steps",
        termination_reason="timeout",
        target_contact_observed=False,
        bilateral_target_contact_observed=False,
        persistent_bilateral_non_target_contact_observed=(
            timeout_diagnostic == "persistent_bilateral_non_target_contact"
        ),
        persistent_bilateral_non_target_body_ids=(7,),
        timeout_diagnostic=timeout_diagnostic,
    )
    return replace(
        _execution(
            False,
            grasp_success=False,
            placement_success=False,
            failure_stage="grasp",
            failure_reason="Gripper close failed: timeout",
        ),
        close_result=close_result,
    )


def test_clean_episode_keeps_phase9_schedule_without_recovery() -> None:
    scenario = generate_scenario(0, distractor_count=2)
    events: list[str] = []
    with World(gui=False) as world:
        world.reset_from_scenario(scenario)
        result = VisionRecoveryAgent().run(
            scenario.task.instruction,
            world,
            stage_callback=lambda stage, _: events.append(stage),
        )

    assert result.success, result.failure_reason
    assert not result.recovery_activated
    assert result.recovery_attempts == 0
    assert result.recovery_trace is None
    assert result.observation_count == 3
    assert "recovery_activated" not in events
    assert result.initial_execution_result is result.execution_result
    assert result.recovery_execution_result is None


def test_seed_89_clean_path_succeeds_without_recovery() -> None:
    scenario = generate_scenario(89, distractor_count=2)
    with World(gui=False) as world:
        world.reset_from_scenario(scenario)
        result = VisionRecoveryAgent().run(scenario.task.instruction, world)

    assert result.success, result.failure_reason
    assert not result.recovery_activated
    assert result.recovery_attempts == 0
    assert result.initial_execution_result is not None
    assert result.initial_execution_result.close_result is not None
    assert (
        result.initial_execution_result.close_result.termination_reason
        == "bilateral_target_contact"
    )


def test_controlled_grasp_miss_recovers_from_reused_visual_frame() -> None:
    scenario = generate_scenario(0)
    events: list[str] = []
    with World(gui=False) as world:
        world.reset_from_scenario(scenario)
        result = VisionRecoveryAgent().run(
            scenario.task.instruction,
            world,
            fault_injection=RecoveryFaultInjection.grasp_miss(),
            stage_callback=lambda stage, _: events.append(stage),
        )

    assert result.success, result.failure_reason
    assert result.recovery_activated and result.recovered
    assert result.recovery_stage == "grasp_verification"
    assert result.recovery_attempts == MAX_RECOVERY_ATTEMPTS == 1
    assert result.observation_count == 4
    assert result.initial_grasp_verification is not None
    assert not result.initial_grasp_verification.verified
    assert result.grasp_verification is not None
    assert result.grasp_verification.verified
    assert result.placement_verification is not None
    assert result.placement_verification.verified
    trace = result.recovery_trace
    assert trace is not None
    assert trace.recovery_entry_type == "post_verification"
    assert trace.fresh_recovery_observation is None
    assert trace.diagnosis in {"cube_not_lifted", "cube_not_near_end_effector"}
    assert trace.recovery_observation is not None
    assert trace.recovery_source_detection is not None
    assert trace.recovery_plan is not None
    assert (
        trace.recovery_plan.source_estimated_center
        == trace.recovery_source_detection.world_position
    )
    assert trace.recovery_plan.grasp_target != result.plan.grasp_target
    assert events.count("recovery_planned") == 1
    assert events.count("recovery_post_grasp_observation_captured") == 1
    assert events.count("recovery_post_placement_observation_captured") == 1
    assert result.final_evaluation is not None and result.final_evaluation.success
    assert result.initial_execution_result is not None
    assert result.recovery_execution_result is not None
    assert result.initial_execution_result.constraint_count_after == 0
    assert result.recovery_execution_result.constraint_count_after == 0


def test_displaced_cube_changes_recovery_source_without_an_extra_observation() -> None:
    task = parse_instruction("Put the red cube in the blue tray.")
    assert task.target is not None
    observations = [object() for _ in range(4)]
    captures = iter(observations)
    vision_inputs: list[object] = []
    executor_calls = 0
    initial_cube = (0.45, 0.00, 0.025)
    displaced_cube = (0.50, 0.02, 0.025)
    tray_position = (0.60, 0.20, 0.004)

    class Vision:
        def __init__(self, observation: object) -> None:
            self.observation = observation
            vision_inputs.append(observation)

        def locate(self, ref: object, **_: object) -> LocalizationResult:
            if ref == task.target:
                position = tray_position
            elif self.observation is observations[0]:
                position = initial_cube
            elif self.observation is observations[1]:
                position = displaced_cube
            elif self.observation is observations[2]:
                position = (displaced_cube[0], displaced_cube[1], 0.20)
            else:
                position = (tray_position[0], tray_position[1], 0.033)
            return LocalizationResult(ref, position, "object_center", "vision")

    def executor(_: object, __: object, **kwargs: object) -> OpenLoopExecutionResult:
        nonlocal executor_calls
        executor_calls += 1
        after_grasp = kwargs["after_grasp"]
        after_placement = kwargs["after_placement"]
        if executor_calls == 1:
            gate = after_grasp((0.57, 0.00, 0.20), False)
            assert not gate.proceed
            return _execution(
                False,
                grasp_success=False,
                placement_success=False,
                failure_stage="grasp_verification",
                failure_reason=gate.reason,
            )
        grasp_gate = after_grasp((0.50, 0.02, 0.20), True)
        assert grasp_gate.proceed
        placement_gate = after_placement((0.60, 0.20, 0.20))
        assert placement_gate.proceed
        return _execution(True, grasp_success=True, placement_success=True)

    result = VisionRecoveryAgent(
        capture_observation=lambda _: next(captures),
        vision_factory=Vision,
        executor=executor,
        evaluator=lambda _: _evaluation_success(),
    ).run(task.instruction, object())

    assert result.success
    assert result.observation_count == 4
    assert executor_calls == 2
    assert result.source_detection is not None
    assert result.source_detection.world_position == initial_cube
    assert result.recovery_trace is not None
    assert result.recovery_trace.recovery_source_detection is not None
    assert (
        result.recovery_trace.recovery_source_detection.world_position
        == displaced_cube
    )
    assert result.recovery_trace.recovery_plan is not None
    assert result.recovery_trace.recovery_plan.source_estimated_center == displaced_cube
    assert vision_inputs.count(observations[1]) == 2


def test_controlled_placement_miss_regrasps_and_replaces_from_fresh_vision() -> None:
    scenario = generate_scenario(0)
    events: list[str] = []
    with World(gui=False) as world:
        world.reset_from_scenario(scenario)
        result = VisionRecoveryAgent().run(
            scenario.task.instruction,
            world,
            fault_injection=RecoveryFaultInjection.placement_miss(),
            stage_callback=lambda stage, _: events.append(stage),
        )

    assert result.success, result.failure_reason
    assert result.recovery_stage == "placement_verification"
    assert result.recovery_attempts == 1
    assert result.observation_count == 5
    assert result.initial_grasp_verification is not None
    assert result.initial_grasp_verification.verified
    assert result.initial_placement_verification is not None
    assert not result.initial_placement_verification.verified
    trace = result.recovery_trace
    assert trace is not None
    assert trace.recovery_entry_type == "post_verification"
    assert trace.fresh_recovery_observation is None
    assert trace.diagnosis == "cube_outside_tray"
    assert trace.recovery_source_detection is not None
    assert trace.recovery_target_detection is not None
    assert trace.recovery_plan is not None
    assert (
        trace.recovery_plan.source_estimated_center
        == trace.recovery_source_detection.world_position
    )
    assert (
        trace.recovery_plan.target_estimated_center
        == trace.recovery_target_detection.world_position
    )
    assert trace.recovery_grasp_verification is not None
    assert trace.recovery_grasp_verification.verified
    assert trace.recovery_placement_verification is not None
    assert trace.recovery_placement_verification.verified
    assert events.count("recovery_post_grasp_observation_captured") == 1
    assert events.count("recovery_post_placement_observation_captured") == 1


class _AlwaysMissPlanner(TaskPlanner):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def plan(self, *args: object, **kwargs: object) -> object:
        self.calls += 1
        plan = super().plan(*args, **kwargs)
        offset = 0.12

        def shifted(position: tuple[float, float, float]) -> tuple[float, float, float]:
            return position[0] + offset, position[1], position[2]

        return replace(
            plan,
            pregrasp_target=shifted(plan.pregrasp_target),
            grasp_target=shifted(plan.grasp_target),
            lift_target=shifted(plan.lift_target),
            safe_transport_target=shifted(plan.safe_transport_target),
        )


def test_failed_recovery_stops_after_one_attempt_without_evaluation() -> None:
    scenario = generate_scenario(0)
    planner = _AlwaysMissPlanner()
    evaluated = False

    def forbidden_evaluation(_: World) -> PlacementEvaluation:
        nonlocal evaluated
        evaluated = True
        raise AssertionError("No placement occurred, so evaluation is forbidden")

    with World(gui=False) as world:
        world.reset_from_scenario(scenario)
        result = VisionRecoveryAgent(
            planner=planner,
            evaluator=forbidden_evaluation,
        ).run(scenario.task.instruction, world)

    assert not result.success
    assert result.recovery_activated
    assert result.recovery_attempts == 1
    assert result.failure_stage == "recovery_grasp_verification"
    assert result.observation_count == 3
    assert planner.calls == 2
    assert not evaluated
    assert result.final_evaluation is None


def test_preverification_ik_failure_gets_exactly_one_failed_recovery_attempt() -> None:
    scenario = generate_scenario(10, distractor_count=1)
    execution = _ik_failure_execution()
    execution_calls = 0
    evaluated = False
    events: list[str] = []

    def execute(*_: object, **__: object) -> OpenLoopExecutionResult:
        nonlocal execution_calls
        execution_calls += 1
        return execution

    def forbidden_evaluation(_: World) -> PlacementEvaluation:
        nonlocal evaluated
        evaluated = True
        raise AssertionError("Pre-verification failure must not be evaluated")

    with World(gui=False) as world:
        world.reset_from_scenario(scenario)
        result = VisionRecoveryAgent(
            executor=execute,
            evaluator=forbidden_evaluation,
        ).run(
            scenario.task.instruction,
            world,
            stage_callback=lambda stage, _: events.append(stage),
        )

    assert not result.success
    assert execution_calls == 2
    assert not evaluated
    assert result.initial_execution_result is execution
    assert result.execution_result is execution
    assert result.execution_result.failed_reach_result is not None
    assert result.initial_grasp_verification is None
    assert result.initial_placement_verification is None
    assert "grasp_verified" not in events
    assert "placement_verified" not in events
    assert result.recovery_activated
    assert result.recovery_attempts == MAX_RECOVERY_ATTEMPTS == 1
    assert result.observation_count == 2
    assert events.count("fresh_recovery_observation_captured") == 1
    trace = result.recovery_trace
    assert trace is not None
    assert trace.recovery_entry_type == "pre_verification"
    assert (
        trace.initial_execution_failure_code
        == IK_RESIDUAL_EXCEEDS_EXECUTION_TOLERANCE
    )
    assert trace.initial_execution_failure_stage == "grasp"
    assert trace.fresh_recovery_observation is not None
    assert trace.fresh_source_detection is not None
    assert trace.fresh_target_detection is not None
    assert trace.recovery_plan is not None
    assert trace.recovery_grasp_verification is None
    assert not trace.recovered
    assert result.failure_stage == "recovery_execution"


def test_persistent_non_target_contact_is_preverification_recoverable() -> None:
    scenario = generate_scenario(56, distractor_count=2)
    execution = _gripper_timeout_execution(
        "persistent_bilateral_non_target_contact"
    )
    execution_calls = 0

    def execute(*_: object, **__: object) -> OpenLoopExecutionResult:
        nonlocal execution_calls
        execution_calls += 1
        return execution

    with World(gui=False) as world:
        world.reset_from_scenario(scenario)
        result = VisionRecoveryAgent(
            executor=execute,
        ).run(scenario.task.instruction, world)

    assert not result.success
    assert execution_calls == 2
    assert result.initial_execution_result is execution
    assert result.execution_result is execution
    assert result.execution_result.close_result is execution.close_result
    assert result.initial_grasp_verification is None
    assert result.initial_placement_verification is None
    assert result.recovery_activated
    assert result.recovery_attempts == 1
    assert result.observation_count == 2
    assert result.recovery_trace is not None
    assert result.recovery_trace.recovery_entry_type == "pre_verification"
    assert (
        result.recovery_trace.initial_execution_failure_code
        == "persistent_bilateral_non_target_contact"
    )
    assert result.final_evaluation is None


def test_preverification_recovery_eligibility_whitelist_is_exact() -> None:
    assert PREVERIFICATION_RECOVERY_ELIGIBLE_FAILURE_CODES == {
        IK_RESIDUAL_EXCEEDS_EXECUTION_TOLERANCE,
        "persistent_bilateral_non_target_contact",
    }
    assert is_preverification_recovery_eligible(_ik_failure_execution())
    assert is_preverification_recovery_eligible(
        _gripper_timeout_execution("persistent_bilateral_non_target_contact")
    )

    assert not is_preverification_recovery_eligible(
        _execution(
            False,
            grasp_success=False,
            placement_success=False,
            failure_stage="grasp",
            failure_reason="Unknown execution failure",
        )
    )
    assert not is_preverification_recovery_eligible(
        _ik_failure_execution("ik_solution_invalid")
    )
    assert not is_preverification_recovery_eligible(
        _ik_failure_execution("joint_limit_invalid")
    )
    assert not is_preverification_recovery_eligible(
        _gripper_timeout_execution("generic_timeout")
    )
    assert not is_preverification_recovery_eligible(
        _gripper_timeout_execution("no_target_contact")
    )


@pytest.mark.parametrize(
    "execution",
    [
        _execution(
            False,
            grasp_success=False,
            placement_success=False,
            failure_stage="grasp",
            failure_reason="Unknown execution failure",
        ),
        _ik_failure_execution("ik_solution_invalid"),
        _gripper_timeout_execution("generic_timeout"),
    ],
    ids=("unknown", "invalid_ik", "generic_gripper_timeout"),
)
def test_noneligible_preverification_failure_stops_without_fresh_capture(
    execution: OpenLoopExecutionResult,
) -> None:
    task = parse_instruction("Put the red cube in the blue tray.")
    assert task.target is not None
    observation = object()
    capture_count = 0
    execution_count = 0

    class Vision:
        def __init__(self, seen: object) -> None:
            assert seen is observation

        def locate(self, ref: object, **_: object) -> LocalizationResult:
            position = (
                (0.60, 0.20, 0.004)
                if ref == task.target
                else (0.45, 0.00, 0.025)
            )
            return LocalizationResult(ref, position, "object_center", "vision")

    def capture(_: object) -> object:
        nonlocal capture_count
        capture_count += 1
        return observation

    def execute(*_: object, **__: object) -> OpenLoopExecutionResult:
        nonlocal execution_count
        execution_count += 1
        return execution

    result = VisionRecoveryAgent(
        capture_observation=capture,
        vision_factory=Vision,
        executor=execute,
    ).run(task.instruction, object())

    assert not result.success
    assert capture_count == 1
    assert execution_count == 1
    assert result.observation_count == 1
    assert not result.recovery_activated
    assert result.recovery_trace is None


def test_preverification_recovery_uses_fresh_vision_and_nominal_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = parse_instruction("Put the red cube in the blue tray.")
    assert task.target is not None
    observations = [object() for _ in range(4)]
    captures = iter(observations)
    vision_inputs: list[object] = []
    executor_calls = 0
    events: list[str] = []
    initial_source = (0.45, 0.00, 0.025)
    fresh_source = (0.47, 0.01, 0.025)
    target = (0.60, 0.20, 0.004)

    class Vision:
        def __init__(self, observation: object) -> None:
            self.observation = observation
            vision_inputs.append(observation)

        def locate(self, ref: object, **_: object) -> LocalizationResult:
            if ref == task.target:
                position = target
            elif self.observation is observations[0]:
                position = initial_source
            elif self.observation is observations[1]:
                position = fresh_source
            elif self.observation is observations[2]:
                position = (fresh_source[0], fresh_source[1], 0.20)
            else:
                position = (target[0], target[1], 0.033)
            return LocalizationResult(ref, position, "object_center", "vision")

    def execute(
        _: object,
        plan: object,
        **kwargs: object,
    ) -> OpenLoopExecutionResult:
        nonlocal executor_calls
        executor_calls += 1
        if executor_calls == 1:
            assert plan.source_estimated_center == initial_source
            assert plan.grasp_target[0] == pytest.approx(initial_source[0] + 0.12)
            return _ik_failure_execution()

        assert plan.source_estimated_center == fresh_source
        assert plan.grasp_target[0] == pytest.approx(fresh_source[0])
        grasp_gate = kwargs["after_grasp"](
            (fresh_source[0], fresh_source[1], 0.20),
            True,
        )
        assert grasp_gate.proceed
        placement_gate = kwargs["after_placement"]((0.60, 0.20, 0.20))
        assert placement_gate.proceed
        return _execution(True, grasp_success=True, placement_success=True)

    monkeypatch.setattr(
        pybullet,
        "getBasePositionAndOrientation",
        lambda *_args, **_kwargs: pytest.fail(
            "Ground-truth object pose entered pre-verification recovery"
        ),
    )
    result = VisionRecoveryAgent(
        capture_observation=lambda _: next(captures),
        vision_factory=Vision,
        executor=execute,
        evaluator=lambda _: _evaluation_success(),
    ).run(
        task.instruction,
        object(),
        fault_injection=RecoveryFaultInjection.grasp_miss(),
        stage_callback=lambda stage, _: events.append(stage),
    )

    assert result.success, result.failure_reason
    assert executor_calls == 2
    assert result.recovery_activated and result.recovered
    assert result.recovery_attempts == MAX_RECOVERY_ATTEMPTS == 1
    assert result.recovery_stage == "pre_verification_execution"
    assert result.recovery_source == "fresh_recovery_rgbd"
    assert result.observation_count == 4
    assert result.initial_grasp_verification is None
    assert result.initial_placement_verification is None
    assert result.grasp_verification is not None
    assert result.grasp_verification.verified
    assert result.placement_verification is not None
    assert result.placement_verification.verified
    trace = result.recovery_trace
    assert trace is not None
    assert trace.recovery_entry_type == "pre_verification"
    assert (
        trace.initial_execution_failure_code
        == IK_RESIDUAL_EXCEEDS_EXECUTION_TOLERANCE
    )
    assert trace.initial_execution_failure_stage == "grasp"
    assert trace.recovery_observation is observations[1]
    assert trace.fresh_recovery_observation is observations[1]
    assert trace.recovery_source_detection is trace.fresh_source_detection
    assert trace.recovery_target_detection is trace.fresh_target_detection
    assert trace.fresh_source_detection is not None
    assert trace.fresh_source_detection.world_position == fresh_source
    assert trace.recovery_plan is not None
    assert trace.recovery_plan.source_estimated_center == fresh_source
    assert trace.recovery_plan.grasp_target[0] == pytest.approx(fresh_source[0])
    assert trace.recovery_grasp_verification is result.grasp_verification
    assert trace.recovery_grasp_verification.verified
    assert trace.recovered
    assert trace.final_failure_reason is None
    assert vision_inputs == [
        observations[0],
        observations[1],
        observations[2],
        observations[3],
    ]
    assert events.count("fresh_recovery_observation_captured") == 1
    assert events.count("recovery_post_grasp_observation_captured") == 1
    assert events.count("recovery_post_placement_observation_captured") == 1


def test_parse_grounding_and_planning_failures_never_activate_recovery() -> None:
    parse_result = VisionRecoveryAgent(
        capture_observation=lambda _: pytest.fail("parse failure must stop")
    ).run("This is not a manipulation instruction.", object())
    assert parse_result.failure_stage == "parse"
    assert not parse_result.recovery_activated

    task = parse_instruction("Put the red cube in the blue tray.")
    observation = object()

    class GroundingFailureVision:
        def __init__(self, _: object) -> None:
            pass

        def locate(self, *_: object, **__: object) -> LocalizationResult:
            raise PerceptionError("No source")

    grounding_result = VisionRecoveryAgent(
        capture_observation=lambda _: observation,
        vision_factory=GroundingFailureVision,
        executor=lambda *_args, **_kwargs: pytest.fail(
            "grounding failure must stop"
        ),
    ).run(task.instruction, object())
    assert grounding_result.failure_stage == "source_grounding"
    assert not grounding_result.recovery_activated

    class Vision:
        def __init__(self, _: object) -> None:
            pass

        def locate(self, ref: object, **_: object) -> LocalizationResult:
            position = (
                (0.60, 0.20, 0.004)
                if ref == task.target
                else (0.45, 0.00, 0.025)
            )
            return LocalizationResult(ref, position, "object_center", "vision")

    class FailingPlanner:
        def plan(self, *_: object, **__: object) -> object:
            raise PlanningError("No plan")

    planning_result = VisionRecoveryAgent(
        capture_observation=lambda _: observation,
        vision_factory=Vision,
        planner=FailingPlanner(),
        executor=lambda *_args, **_kwargs: pytest.fail(
            "planning failure must stop"
        ),
    ).run(task.instruction, object())
    assert planning_result.failure_stage == "planning"
    assert not planning_result.recovery_activated


def test_recovery_actions_do_not_query_simulator_object_pose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = generate_scenario(0)
    events: list[str] = []
    with World(gui=False) as world:
        world.reset_from_scenario(scenario)

        def forbidden_pose(*_: object, **__: object) -> object:
            raise AssertionError("Ground-truth object pose entered recovery")

        monkeypatch.setattr(pybullet, "getBasePositionAndOrientation", forbidden_pose)
        result = VisionRecoveryAgent(
            evaluator=lambda _: _evaluation_success(),
        ).run(
            scenario.task.instruction,
            world,
            fault_injection=RecoveryFaultInjection.grasp_miss(),
            stage_callback=lambda stage, _: events.append(stage),
        )

    assert result.success, result.failure_reason
    assert result.recovery_activated
    assert result.final_evaluation_source == "simulator_ground_truth"
    assert events.index("recovery_placement_verified") < events.index("evaluated")


def test_fault_injection_validation_is_explicit_and_finite() -> None:
    assert MAX_RECOVERY_ATTEMPTS == 1
    with pytest.raises(ValueError, match="initial_grasp_xy_offset"):
        RecoveryFaultInjection(initial_grasp_xy_offset=(float("nan"), 0.0))
