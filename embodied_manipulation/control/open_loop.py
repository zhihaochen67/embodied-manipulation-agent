"""Physical execution of a precomputed vision-grounded open-loop plan."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from time import sleep

import pybullet

from embodied_manipulation.planning import ManipulationPlan, Position
from embodied_manipulation.simulation import World
from embodied_manipulation.simulation.robot import PANDA_FINGER_JOINT_INDICES

from .arm_controller import ArmController, ReachResult
from .gripper import PandaGripper
from .pick import (
    CARTESIAN_WAYPOINT_COUNT,
    CONTACT_SETTLE_STEPS,
    PICK_POSITION_TOLERANCE,
    _finger_cube_contacts,
    _linear_waypoints,
    _move_through_waypoints,
    _vertical_waypoints,
)
from .pick_place import POST_RETREAT_STEPS, SETTLING_STEPS


@dataclass(frozen=True, slots=True)
class StageGateResult:
    """Whether a stage-level observer permits the fixed plan to continue."""

    proceed: bool
    reason: str


GraspStageGate = Callable[[Position, bool], StageGateResult]
PlacementStageGate = Callable[[Position], StageGateResult]


@dataclass(frozen=True, slots=True)
class OpenLoopExecutionResult:
    """Procedural result without any cube/tray world-pose measurements."""

    success: bool
    grasp_success: bool
    transport_success: bool
    placement_success: bool
    arm_steps: int
    gripper_steps: int
    settling_steps: int
    total_steps: int
    constraint_count_before: int
    constraint_count_after: int
    transport_finger_targets: tuple[float, float] | None
    failure_stage: str | None = None
    failure_reason: str | None = None


def execute_open_loop_plan(
    world: World,
    plan: ManipulationPlan,
    *,
    arm_max_steps: int = 480,
    waypoint_count: int = CARTESIAN_WAYPOINT_COUNT,
    settling_steps: int = SETTLING_STEPS,
    post_retreat_steps: int = POST_RETREAT_STEPS,
    contact_settle_steps: int = CONTACT_SETTLE_STEPS,
    position_tolerance: float = PICK_POSITION_TOLERANCE,
    step_delay: float = 0.0,
    stage_pause: float = 0.0,
    after_grasp: GraspStageGate | None = None,
    after_placement: PlacementStageGate | None = None,
) -> OpenLoopExecutionResult:
    """Execute fixed targets without observing or measuring cube/tray poses.

    End-effector and joint proprioception, gripper state, and finger/cube
    contacts are used by the physical controllers. Object world poses are not
    read here; objective task evaluation is the agent's later, separate stage.
    """
    if world.scene is None:
        raise RuntimeError("World must be reset before plan execution")
    if world.scene.tray is None:
        raise RuntimeError("Open-loop pick-and-place requires a tray")
    if plan.task.action != "pick_and_place":
        raise ValueError("Open-loop executor requires a pick-and-place plan")
    if after_grasp is not None and not callable(after_grasp):
        raise ValueError("after_grasp must be callable")
    if after_placement is not None and not callable(after_placement):
        raise ValueError("after_placement must be callable")
    for value, name in (
        (arm_max_steps, "arm_max_steps"),
        (waypoint_count, "waypoint_count"),
        (settling_steps, "settling_steps"),
        (post_retreat_steps, "post_retreat_steps"),
        (contact_settle_steps, "contact_settle_steps"),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for value, name in (
        (position_tolerance, "position_tolerance"),
        (step_delay, "step_delay"),
        (stage_pause, "stage_pause"),
    ):
        if not isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be nonnegative and finite")
    if position_tolerance == 0.0:
        raise ValueError("position_tolerance must be positive")

    scene = world.scene
    arm = ArmController(world.client_id, scene.robot_id)
    gripper = PandaGripper(world.client_id, scene.robot_id)
    constraint_count_before = pybullet.getNumConstraints(
        physicsClientId=world.client_id
    )
    arm_results: list[ReachResult] = []
    gripper_steps = 0
    completed_settling_steps = 0
    grasp_success = False
    transport_success = False
    placement_success = False
    transport_finger_targets: tuple[float, float] | None = None

    def finish(
        success: bool,
        failure_stage: str | None,
        failure_reason: str | None,
    ) -> OpenLoopExecutionResult:
        constraint_count_after = pybullet.getNumConstraints(
            physicsClientId=world.client_id
        )
        if constraint_count_after != constraint_count_before:
            success = False
            failure_stage = failure_stage or "placement"
            failure_reason = "A simulation constraint was created during execution"
        arm_steps = sum(result.steps for result in arm_results)
        total_steps = arm_steps + gripper_steps + completed_settling_steps
        return OpenLoopExecutionResult(
            success=success,
            grasp_success=grasp_success,
            transport_success=transport_success,
            placement_success=placement_success,
            arm_steps=arm_steps,
            gripper_steps=gripper_steps,
            settling_steps=completed_settling_steps,
            total_steps=total_steps,
            constraint_count_before=constraint_count_before,
            constraint_count_after=constraint_count_after,
            transport_finger_targets=transport_finger_targets,
            failure_stage=failure_stage,
            failure_reason=failure_reason,
        )

    open_result = gripper.open(step_delay=step_delay)
    gripper_steps += open_result.steps
    _pause(stage_pause)
    if not open_result.success:
        return finish(False, "grasp", f"Gripper open failed: {open_result.failure_reason}")

    pregrasp_results, _ = _move_through_waypoints(
        arm,
        _linear_waypoints(
            arm.end_effector_pose()[0],
            plan.pregrasp_target,
            waypoint_count,
        ),
        position_tolerance=position_tolerance,
        max_steps=arm_max_steps,
        step_delay=step_delay,
    )
    arm_results.extend(pregrasp_results)
    _pause(stage_pause)
    if not pregrasp_results[-1].success:
        return finish(
            False,
            "grasp",
            f"Pre-grasp move failed: {pregrasp_results[-1].failure_reason}",
        )

    approach_results, _ = _move_through_waypoints(
        arm,
        _vertical_waypoints(
            arm.end_effector_pose()[0],
            plan.grasp_target,
            waypoint_count,
        ),
        position_tolerance=position_tolerance,
        max_steps=arm_max_steps,
        step_delay=step_delay,
    )
    arm_results.extend(approach_results)
    _pause(stage_pause)
    if not approach_results[-1].success:
        return finish(
            False,
            "grasp",
            f"Grasp approach failed: {approach_results[-1].failure_reason}",
        )

    close_result = gripper.close(step_delay=step_delay)
    gripper_steps += close_result.steps
    if not close_result.success:
        return finish(
            False,
            "grasp",
            f"Gripper close failed: {close_result.failure_reason}",
        )
    hold_result = gripper.hold(
        steps=contact_settle_steps,
        step_delay=step_delay,
    )
    gripper_steps += hold_result.steps
    initial_bilateral_contact = _has_bilateral_contact(world)
    if not initial_bilateral_contact and after_grasp is None:
        return finish(False, "grasp", "Cube was not contacted by both fingers")

    transport_finger_targets = gripper.capture_hold()
    lift_results, _ = _move_through_waypoints(
        arm,
        _vertical_waypoints(
            arm.end_effector_pose()[0],
            plan.lift_target,
            waypoint_count,
        ),
        position_tolerance=position_tolerance,
        max_steps=arm_max_steps,
        step_delay=step_delay,
        before_step=gripper.maintain,
    )
    arm_results.extend(lift_results)
    _pause(stage_pause)
    if not lift_results[-1].success:
        return finish(False, "grasp", f"Lift failed: {lift_results[-1].failure_reason}")
    bilateral_contact = _has_bilateral_contact(world)
    grasp_success = bilateral_contact
    if after_grasp is not None:
        gate = after_grasp(arm.end_effector_pose()[0], bilateral_contact)
        if not gate.proceed:
            return finish(
                False,
                "grasp_verification",
                gate.reason,
            )
    if not bilateral_contact:
        return finish(False, "grasp", "Cube contact was lost during lift")

    safe_results, _ = _move_through_waypoints(
        arm,
        _vertical_waypoints(
            arm.end_effector_pose()[0],
            plan.safe_transport_target,
            max(2, waypoint_count // 2),
        ),
        position_tolerance=position_tolerance,
        max_steps=arm_max_steps,
        step_delay=step_delay,
        before_step=gripper.maintain,
    )
    arm_results.extend(safe_results)
    if not safe_results[-1].success:
        return finish(
            False,
            "transport",
            f"Transport raise failed: {safe_results[-1].failure_reason}",
        )

    transport_results, _ = _move_through_waypoints(
        arm,
        _linear_waypoints(
            arm.end_effector_pose()[0],
            plan.above_target,
            waypoint_count,
        ),
        position_tolerance=position_tolerance,
        max_steps=arm_max_steps,
        step_delay=step_delay,
        before_step=gripper.maintain,
    )
    arm_results.extend(transport_results)
    _pause(stage_pause)
    if not transport_results[-1].success:
        return finish(
            False,
            "transport",
            f"Transport move failed: {transport_results[-1].failure_reason}",
        )
    if not _has_bilateral_contact(world):
        return finish(False, "transport", "Cube contact was lost during transport")
    transport_success = True

    placement_results, _ = _move_through_waypoints(
        arm,
        _vertical_waypoints(
            arm.end_effector_pose()[0],
            plan.release_target,
            waypoint_count,
        ),
        position_tolerance=position_tolerance,
        max_steps=arm_max_steps,
        step_delay=step_delay,
        before_step=gripper.maintain,
    )
    arm_results.extend(placement_results)
    if not placement_results[-1].success:
        return finish(
            False,
            "placement",
            f"Placement descent failed: {placement_results[-1].failure_reason}",
        )
    if not _has_bilateral_contact(world):
        return finish(False, "placement", "Cube contact was lost before release")

    release_result = gripper.open(step_delay=step_delay)
    gripper_steps += release_result.steps
    if not release_result.success:
        return finish(
            False,
            "placement",
            f"Gripper release failed: {release_result.failure_reason}",
        )
    _pause(stage_pause)

    _step_world(world, settling_steps, step_delay)
    completed_settling_steps += settling_steps
    retreat_results, _ = _move_through_waypoints(
        arm,
        _vertical_waypoints(
            arm.end_effector_pose()[0],
            plan.retreat_target,
            waypoint_count,
        ),
        position_tolerance=position_tolerance,
        max_steps=arm_max_steps,
        step_delay=step_delay,
    )
    arm_results.extend(retreat_results)
    if not retreat_results[-1].success:
        return finish(
            False,
            "placement",
            f"Retreat failed: {retreat_results[-1].failure_reason}",
        )

    _step_world(world, post_retreat_steps, step_delay)
    completed_settling_steps += post_retreat_steps
    _pause(stage_pause)
    placement_success = True
    if after_placement is not None:
        gate = after_placement(arm.end_effector_pose()[0])
        if not gate.proceed:
            return finish(
                False,
                "placement_verification",
                gate.reason,
            )
    return finish(True, None, None)


def _has_bilateral_contact(world: World) -> bool:
    """Use only the permitted robot/object contact signal."""
    assert world.scene is not None
    links = {
        contact.finger_link_index
        for contact in _finger_cube_contacts(
            world.client_id,
            world.scene.robot_id,
            world.scene.cube_id,
        )
    }
    return links == set(PANDA_FINGER_JOINT_INDICES)


def _step_world(world: World, steps: int, step_delay: float) -> None:
    for _ in range(steps):
        world.step()
        if step_delay > 0.0:
            sleep(step_delay)


def _pause(duration: float) -> None:
    if duration > 0.0:
        sleep(duration)
