"""Phase 8 vision-conditioned open-loop agent regression tests."""

from __future__ import annotations

from math import isfinite

import pybullet
import pytest

from embodied_manipulation.agent import VisionOpenLoopAgent
from embodied_manipulation.benchmark import generate_scenario
from embodied_manipulation.control.open_loop import (
    OpenLoopExecutionResult,
    execute_open_loop_plan,
)
from embodied_manipulation.control.pick_place import PlacementEvaluation
from embodied_manipulation.language import parse_instruction
from embodied_manipulation.perception import (
    LocalizationResult,
    ObjectNotFoundError,
    OraclePerception,
    RGBDObservation,
    VisionPerception,
)
from embodied_manipulation.planning import (
    PlanningError,
    PrimitiveKind,
    TaskPlanner,
)
from embodied_manipulation.simulation import World


def _execution_success() -> OpenLoopExecutionResult:
    return OpenLoopExecutionResult(
        success=True,
        grasp_success=True,
        transport_success=True,
        placement_success=True,
        arm_steps=10,
        gripper_steps=5,
        settling_steps=2,
        total_steps=17,
        constraint_count_before=0,
        constraint_count_after=0,
        transport_finger_targets=(0.02, 0.02),
    )


def _evaluation_success() -> PlacementEvaluation:
    return PlacementEvaluation(
        success=True,
        cube_position=(0.5, 0.28, 0.033),
        cube_linear_velocity=(0.0, 0.0, 0.0),
        cube_angular_velocity=(0.0, 0.0, 0.0),
        cube_inside_tray=True,
        cube_near_floor=True,
        cube_released=True,
        stable=True,
        cube_to_tray_center_distance=0.0,
    )


def test_agent_parses_observes_once_and_grounds_with_vision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = generate_scenario(0, distractor_count=2)
    captures = 0
    events: list[str] = []

    def capture(world: World) -> RGBDObservation:
        nonlocal captures
        captures += 1
        assert world.scene is not None
        return RGBDObservation.capture(world.camera, world.client_id)

    def forbidden_oracle(*_: object, **__: object) -> LocalizationResult:
        raise AssertionError("OraclePerception must not select Phase 8 actions")

    def execute(
        world: World,
        plan: object,
        **_: object,
    ) -> OpenLoopExecutionResult:
        events.append("executor")
        assert world.scene is not None
        return _execution_success()

    def evaluate(_: World) -> PlacementEvaluation:
        events.append("evaluator")
        assert "executor" in events
        return _evaluation_success()

    monkeypatch.setattr(OraclePerception, "locate", forbidden_oracle)
    with World(gui=False) as world:
        world.reset_from_scenario(scenario)
        result = VisionOpenLoopAgent(
            capture_observation=capture,
            executor=execute,
            evaluator=evaluate,
        ).run(
            scenario.task.instruction,
            world,
            stage_callback=lambda stage, _: events.append(stage),
        )

    assert result.success
    assert captures == result.observation_count == 1
    assert result.parsed_task is not None
    assert result.parsed_task.action == scenario.task.action
    assert result.parsed_task.source.color == scenario.task.source.color
    assert result.parsed_task.target is not None
    assert result.parsed_task.target.color == scenario.task.target.color
    assert result.source_detection is not None
    assert result.target_detection is not None
    assert result.source_detection.source == "vision"
    assert result.target_detection.source == "vision"
    assert result.plan is not None
    assert result.plan.source_estimated_center == (
        result.source_detection.world_position
    )
    assert result.plan.target_estimated_center == (
        result.target_detection.world_position
    )
    assert events.count("observation_captured") == 1
    assert events.index("executor") < events.index("evaluator")
    assert result.execution_source == "vision"
    assert result.final_evaluation_source == "simulator_ground_truth"


def test_planner_maps_visual_estimates_to_finite_cartesian_targets() -> None:
    task = parse_instruction("Put the red cube in the blue tray.")
    assert task.target is not None
    source = LocalizationResult(
        object_ref=task.source,
        world_position=(0.5123, -0.0432, 0.025),
        reference="object_center",
        source="vision",
    )
    target = LocalizationResult(
        object_ref=task.target,
        world_position=(0.5312, 0.2743, 0.004),
        reference="tray_floor_body_center",
        source="vision",
    )

    plan = TaskPlanner().plan(task, source, target)

    assert plan.source_estimated_center == source.world_position
    assert plan.target_estimated_center == target.world_position
    assert plan.grasp_target[:2] == source.world_position[:2]
    assert plan.above_target[:2] == target.world_position[:2]
    assert plan.release_target[:2] == target.world_position[:2]
    assert plan.nominal_cube_to_ee_offset == (0.0, 0.0, 0.0)
    positions = (
        plan.pregrasp_target,
        plan.grasp_target,
        plan.lift_target,
        plan.safe_transport_target,
        plan.above_target,
        plan.release_target,
        plan.retreat_target,
    )
    assert all(isfinite(value) for position in positions for value in position)
    assert tuple(step.primitive for step in plan.steps) == (
        PrimitiveKind.OPEN_GRIPPER,
        PrimitiveKind.MOVE_TO_PREGRASP,
        PrimitiveKind.APPROACH,
        PrimitiveKind.CLOSE_GRIPPER,
        PrimitiveKind.LIFT,
        PrimitiveKind.RAISE_FOR_TRANSPORT,
        PrimitiveKind.TRANSPORT,
        PrimitiveKind.DESCEND_TO_RELEASE,
        PrimitiveKind.RELEASE,
        PrimitiveKind.RETREAT,
    )


def test_planner_rejects_oracle_localization() -> None:
    task = parse_instruction("Put the red cube in the blue tray.")
    assert task.target is not None
    source = LocalizationResult(
        task.source,
        (0.5, 0.0, 0.025),
        "object_center",
        "oracle",
    )
    target = LocalizationResult(
        task.target,
        (0.5, 0.28, 0.004),
        "tray_floor_body_center",
        "vision",
    )

    with pytest.raises(PlanningError, match="source detection must come from vision"):
        TaskPlanner().plan(task, source, target)


def test_executor_never_reads_object_poses_or_creates_constraint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = generate_scenario(0, distractor_count=2)
    with World(gui=False) as world:
        world.reset_from_scenario(scenario)
        observation = RGBDObservation.capture(world.camera, world.client_id)
        perception = VisionPerception(observation)
        task = parse_instruction(scenario.task.instruction)
        source = perception.locate(task.source)
        assert task.target is not None
        target = perception.locate(task.target)

        def forbidden(*_: object, **__: object) -> object:
            raise AssertionError("Object ground-truth pose entered open-loop execution")

        monkeypatch.setattr(world, "get_cube_pose", forbidden)
        monkeypatch.setattr(world, "get_tray_pose", forbidden)
        monkeypatch.setattr(
            pybullet,
            "getBasePositionAndOrientation",
            forbidden,
        )
        monkeypatch.setattr(pybullet, "createConstraint", forbidden)

        # Planning also occurs after forbidden pose lookup instrumentation.
        plan = TaskPlanner().plan(task, source, target)
        result = execute_open_loop_plan(world, plan)

    assert result.success, result.failure_reason
    assert result.grasp_success
    assert result.transport_success
    assert result.placement_success
    assert result.constraint_count_before == result.constraint_count_after == 0


def test_seed_zero_executes_full_vision_open_loop_reproducibly() -> None:
    scenario = generate_scenario(0, distractor_count=2)

    def run_once() -> object:
        with World(gui=False) as world:
            world.reset_from_scenario(scenario)
            return VisionOpenLoopAgent().run(scenario.task.instruction, world)

    first = run_once()
    second = run_once()

    assert first.success, first.failure_reason
    assert second.success, second.failure_reason
    assert first.execution_success and second.execution_success
    assert first.final_evaluation is not None
    assert second.final_evaluation is not None
    assert first.final_evaluation.success
    assert second.final_evaluation.success
    assert first.plan == second.plan
    assert first.total_steps == second.total_steps
    assert first.observation_count == second.observation_count == 1
    assert first.execution_result is not None
    assert first.execution_result.constraint_count_before == 0
    assert first.execution_result.constraint_count_after == 0


def test_impossible_perception_query_fails_cleanly() -> None:
    class MissingVision:
        def locate(self, _: object) -> LocalizationResult:
            raise ObjectNotFoundError("synthetic missing object")

    def forbidden_execution(*_: object, **__: object) -> OpenLoopExecutionResult:
        raise AssertionError("Execution must not start after grounding failure")

    result = VisionOpenLoopAgent(
        capture_observation=lambda _: object(),
        vision_factory=lambda _: MissingVision(),
        executor=forbidden_execution,
    ).run("Put the red cube in the blue tray.", object())

    assert not result.success
    assert result.failure_stage == "source_grounding"
    assert result.failure_reason == "synthetic missing object"
    assert result.source_detection is None
    assert result.plan is None
    assert result.observation_count == 1
    assert result.total_steps == 0


def test_pick_only_returns_explicit_unsupported_result_without_observing() -> None:
    def forbidden_capture(_: object) -> RGBDObservation:
        raise AssertionError("Unsupported pick-only task should not start perception")

    result = VisionOpenLoopAgent(
        capture_observation=forbidden_capture,
    ).run("Pick up the red cube.", object())

    assert not result.success
    assert result.parsed_task is not None
    assert result.parsed_task.action == "pick"
    assert result.failure_stage == "planning"
    assert "pick-only execution is intentionally deferred" in result.failure_reason
    assert result.observation_count == 0
