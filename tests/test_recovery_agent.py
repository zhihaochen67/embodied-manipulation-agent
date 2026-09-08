"""Phase 10 bounded visual recovery regression tests."""

from __future__ import annotations

from dataclasses import replace

import pybullet
import pytest

from embodied_manipulation.agent import (
    MAX_RECOVERY_ATTEMPTS,
    RecoveryFaultInjection,
    VisionRecoveryAgent,
)
from embodied_manipulation.benchmark import generate_scenario
from embodied_manipulation.control.open_loop import OpenLoopExecutionResult
from embodied_manipulation.control.pick_place import PlacementEvaluation
from embodied_manipulation.language import parse_instruction
from embodied_manipulation.perception import LocalizationResult
from embodied_manipulation.planning import TaskPlanner
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
