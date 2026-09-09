"""Phase 9 closed-loop visual verification regression tests."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pybullet
import pytest

from embodied_manipulation.agent import (
    VerificationConfig,
    VisionClosedLoopAgent,
    verify_grasp,
    verify_placement,
)
from embodied_manipulation.benchmark import generate_scenario
from embodied_manipulation.control.pick_place import (
    PlacementEvaluation,
    evaluate_placement,
)
from embodied_manipulation.language import parse_instruction
from embodied_manipulation.perception import (
    LocalizationResult,
    RGBDObservation,
    VisionPerception,
)
from embodied_manipulation.planning import TaskPlanner
from embodied_manipulation.simulation import World
from embodied_manipulation.simulation.objects import TABLE_SURFACE_Z


def _detections() -> tuple[LocalizationResult, LocalizationResult]:
    task = parse_instruction("Put the red cube in the blue tray.")
    assert task.target is not None
    cube = LocalizationResult(
        object_ref=task.source,
        world_position=(0.50, 0.28, 0.033),
        reference="object_center",
        source="vision",
    )
    tray = LocalizationResult(
        object_ref=task.target,
        world_position=(0.50, 0.28, 0.004),
        reference="tray_floor_body_center",
        source="vision",
    )
    return cube, tray


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


def test_grasp_verification_uses_visual_height_proximity_and_motion() -> None:
    cube, _ = _detections()
    lifted = replace(cube, world_position=(0.51, -0.02, 0.20))

    result = verify_grasp(
        lifted,
        initial_cube_position=(0.51, -0.02, 0.025),
        end_effector_position=(0.51, -0.02, 0.202),
        bilateral_contact=True,
    )

    assert result.verified
    assert result.cube_detected
    assert result.cube_height_above_table == pytest.approx(0.175)
    assert result.cube_to_ee_distance == pytest.approx(0.002)
    assert result.visual_displacement_from_initial == pytest.approx(0.175)


def test_grasp_verification_rejects_visually_stationary_tabletop_cube() -> None:
    cube, _ = _detections()
    tabletop = replace(cube, world_position=(0.51, -0.02, 0.025))

    result = verify_grasp(
        tabletop,
        initial_cube_position=tabletop.world_position,
        end_effector_position=(0.63, -0.02, 0.205),
        bilateral_contact=False,
    )

    assert not result.verified
    assert result.cube_height_above_table == pytest.approx(0.0)
    assert "clearance" in result.reason


def test_placement_verification_accepts_contained_released_cube() -> None:
    cube, tray = _detections()

    result = verify_placement(
        cube,
        tray,
        end_effector_position=(0.50, 0.28, 0.205),
    )

    assert result.verified
    assert result.cube_inside_tray
    assert result.cube_near_tray_floor
    assert result.cube_released


def test_placement_verification_rejects_outside_tray() -> None:
    cube, tray = _detections()
    outside = replace(cube, world_position=(0.68, 0.28, 0.025))

    result = verify_placement(
        outside,
        tray,
        end_effector_position=(0.68, 0.28, 0.205),
    )

    assert not result.verified
    assert not result.cube_inside_tray
    assert "outside" in result.reason


def test_dynamic_cube_localization_recovers_non_tabletop_height() -> None:
    scenario = generate_scenario(0)
    commanded = (
        scenario.source.initial_position[0],
        scenario.source.initial_position[1],
        0.15,
    )
    with World(gui=False) as world:
        world.reset_from_scenario(scenario)
        assert world.scene is not None
        pybullet.resetBasePositionAndOrientation(
            world.scene.cube_id,
            commanded,
            (0.0, 0.0, 0.0, 1.0),
            physicsClientId=world.client_id,
        )
        vision = VisionPerception(
            RGBDObservation.capture(world.camera, world.client_id)
        )
        dynamic = vision.locate(
            scenario.task.source,
            cube_center_mode="dynamic",
        )
        tabletop = vision.locate(scenario.task.source)

    np.testing.assert_allclose(dynamic.world_position, commanded, atol=0.005)
    assert tabletop.world_position[2] == pytest.approx(TABLE_SURFACE_Z + 0.025)
    assert abs(dynamic.world_position[2] - tabletop.world_position[2]) > 0.10


def test_successful_closed_loop_uses_three_observations_and_immutable_plan() -> None:
    scenario = generate_scenario(0, distractor_count=2)
    captures = 0
    events: list[str] = []
    planned: list[object] = []

    def capture(world: World) -> RGBDObservation:
        nonlocal captures
        captures += 1
        return RGBDObservation.capture(world.camera, world.client_id)

    def event(stage: str, value: object) -> None:
        events.append(stage)
        if stage == "planned":
            planned.append(value)

    def evaluate(world: World) -> PlacementEvaluation:
        events.append("evaluator_called")
        return evaluate_placement(world)

    with World(gui=False) as world:
        world.reset_from_scenario(scenario)
        result = VisionClosedLoopAgent(
            capture_observation=capture,
            evaluator=evaluate,
        ).run(
            scenario.task.instruction,
            world,
            stage_callback=event,
        )

    assert result.success, result.failure_reason
    assert captures == result.observation_count == 3
    assert result.source_detection is not None
    assert result.target_detection is not None
    assert result.source_detection.source == "vision"
    assert result.target_detection.source == "vision"
    assert result.grasp_verification is not None
    assert result.grasp_verification.verified
    assert result.placement_verification is not None
    assert result.placement_verification.verified
    assert result.final_evaluation is not None
    assert result.final_evaluation.success
    assert result.plan is planned[0]
    assert events.count("post_grasp_observation_captured") == 1
    assert events.count("post_placement_observation_captured") == 1
    assert events.index("grasp_verified") < events.index("placement_verified")
    assert events.index("placement_verified") < events.index("evaluator_called")
    assert result.execution_result is not None
    assert result.execution_result.constraint_count_before == 0
    assert result.execution_result.constraint_count_after == 0


def test_seed_89_closed_loop_visually_verifies_contact_supported_grasp() -> None:
    scenario = generate_scenario(89, distractor_count=2)
    with World(gui=False) as world:
        world.reset_from_scenario(scenario)
        result = VisionClosedLoopAgent().run(scenario.task.instruction, world)

    assert result.success, result.failure_reason
    assert result.grasp_verification is not None
    assert result.grasp_verification.verified
    assert result.execution_result is not None
    assert result.execution_result.close_result is not None
    assert (
        result.execution_result.close_result.termination_reason
        == "bilateral_target_contact"
    )


class _MissedGraspPlanner(TaskPlanner):
    calls = 0

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


class _OutsidePlacementPlanner(TaskPlanner):
    def plan(self, *args: object, **kwargs: object) -> object:
        plan = super().plan(*args, **kwargs)
        offset = 0.18

        def shifted(position: tuple[float, float, float]) -> tuple[float, float, float]:
            return position[0] + offset, position[1], position[2]

        return replace(
            plan,
            above_target=shifted(plan.above_target),
            release_target=shifted(plan.release_target),
            retreat_target=shifted(plan.retreat_target),
        )


def test_controlled_missed_grasp_is_rejected_without_retry_or_evaluation() -> None:
    scenario = generate_scenario(0)
    planner = _MissedGraspPlanner()
    evaluated = False

    def forbidden_evaluation(_: World) -> PlacementEvaluation:
        nonlocal evaluated
        evaluated = True
        raise AssertionError("Grasp verification failure must stop before evaluation")

    with World(gui=False) as world:
        world.reset_from_scenario(scenario)
        result = VisionClosedLoopAgent(
            planner=planner,
            evaluator=forbidden_evaluation,
        ).run(scenario.task.instruction, world)

    assert not result.success
    assert result.failure_stage == "grasp_verification"
    assert result.observation_count == 2
    assert result.grasp_verification is not None
    assert not result.grasp_verification.verified
    assert result.placement_verification is None
    assert result.final_evaluation is None
    assert planner.calls == 1
    assert not evaluated


def test_controlled_outside_placement_is_visually_rejected_and_evaluated() -> None:
    scenario = generate_scenario(0)
    with World(gui=False) as world:
        world.reset_from_scenario(scenario)
        result = VisionClosedLoopAgent(
            planner=_OutsidePlacementPlanner(),
        ).run(scenario.task.instruction, world)

    assert not result.success
    assert result.failure_stage == "placement_verification"
    assert result.observation_count == 3
    assert result.grasp_verification is not None
    assert result.grasp_verification.verified
    assert result.placement_verification is not None
    assert not result.placement_verification.verified
    assert not result.placement_verification.cube_inside_tray
    assert result.final_evaluation is not None
    assert not result.final_evaluation.success


def test_visual_decisions_do_not_query_simulator_object_pose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = generate_scenario(0)
    with World(gui=False) as world:
        world.reset_from_scenario(scenario)

        def forbidden_pose(*_: object, **__: object) -> object:
            raise AssertionError("Ground-truth object pose entered visual verification")

        monkeypatch.setattr(
            pybullet,
            "getBasePositionAndOrientation",
            forbidden_pose,
        )
        result = VisionClosedLoopAgent(
            evaluator=lambda _: _evaluation_success(),
        ).run(scenario.task.instruction, world)

    assert result.success, result.failure_reason
    assert result.observation_count == 3
    assert result.final_evaluation_source == "simulator_ground_truth"


def test_verification_thresholds_reject_invalid_geometry() -> None:
    with pytest.raises(ValueError, match="minimum_lift_clearance"):
        VerificationConfig(minimum_lift_clearance=0.0)
