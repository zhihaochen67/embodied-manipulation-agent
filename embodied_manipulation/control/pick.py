"""Fixed oracle top-down pick for the deterministic Phase 3 scene."""

from dataclasses import dataclass
from math import dist, hypot, isfinite
from time import sleep

import pybullet

from embodied_manipulation.simulation import World
from embodied_manipulation.simulation.robot import (
    PANDA_FINGER_JOINT_INDICES,
    get_end_effector_pose,
)

from .arm_controller import ArmController, ReachResult
from .gripper import GripperResult, PandaGripper
from .ik import DEFAULT_END_EFFECTOR_ORIENTATION

# With the default downward orientation, the loaded finger collision geometry
# extends about 11 mm below panda_grasptarget. Aligning that frame with the
# 25 mm-high cube centre therefore leaves about 14 mm of table clearance.
PREGRASP_CLEARANCE = 0.20
GRASP_TARGET_Z_OFFSET = 0.0
LIFT_DISTANCE = 0.18
PICK_POSITION_TOLERANCE = 0.004
MINIMUM_SUCCESSFUL_LIFT = 0.08
MAXIMUM_CUBE_TO_EE_DISTANCE = 0.08
MAXIMUM_HORIZONTAL_DISPLACEMENT = 0.05
CONTACT_SETTLE_STEPS = 30
FINAL_HOLD_STEPS = 30
CARTESIAN_WAYPOINT_COUNT = 10


@dataclass(frozen=True, slots=True)
class FingerContact:
    """A measured contact between one Panda finger and the cube."""

    finger_link_index: int
    position_on_finger: tuple[float, float, float]
    position_on_cube: tuple[float, float, float]
    contact_distance: float
    normal_force: float


@dataclass(frozen=True, slots=True)
class PickResult:
    """Measured outcome of one fixed oracle pick sequence."""

    success: bool
    initial_cube_position: tuple[float, float, float]
    pregrasp_target_position: tuple[float, float, float]
    grasp_target_position: tuple[float, float, float]
    lift_target_position: tuple[float, float, float]
    pregrasp_trajectory: tuple[tuple[float, float, float], ...]
    approach_trajectory: tuple[tuple[float, float, float], ...]
    lift_trajectory: tuple[tuple[float, float, float], ...]
    end_effector_position_at_grasp: tuple[float, float, float]
    final_end_effector_position: tuple[float, float, float]
    final_cube_position: tuple[float, float, float]
    cube_lift_distance: float
    final_cube_to_end_effector_distance: float
    horizontal_cube_displacement: float
    grasp_contacts: tuple[FingerContact, ...]
    final_contacts: tuple[FingerContact, ...]
    open_result: GripperResult | None
    close_result: GripperResult | None
    pregrasp_result: ReachResult | None
    approach_result: ReachResult | None
    lift_result: ReachResult | None
    arm_steps: int
    gripper_steps: int
    total_steps: int
    constraint_count_before: int
    constraint_count_after: int
    failure_reason: str | None = None


def oracle_pick(
    world: World,
    *,
    pregrasp_clearance: float = PREGRASP_CLEARANCE,
    grasp_target_z_offset: float = GRASP_TARGET_Z_OFFSET,
    lift_distance: float = LIFT_DISTANCE,
    position_tolerance: float = PICK_POSITION_TOLERANCE,
    arm_max_steps: int = 480,
    contact_settle_steps: int = CONTACT_SETTLE_STEPS,
    final_hold_steps: int = FINAL_HOLD_STEPS,
    waypoint_count: int = CARTESIAN_WAYPOINT_COUNT,
    step_delay: float = 0.0,
    stage_pause: float = 0.0,
) -> PickResult:
    """Execute one oracle open, approach, close, lift, and hold sequence."""
    if world.scene is None:
        raise RuntimeError("World must be reset before oracle_pick")
    for value, name in (
        (pregrasp_clearance, "pregrasp_clearance"),
        (lift_distance, "lift_distance"),
        (position_tolerance, "position_tolerance"),
    ):
        if not isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    if not isfinite(grasp_target_z_offset):
        raise ValueError("grasp_target_z_offset must be finite")
    if not isfinite(step_delay) or step_delay < 0.0:
        raise ValueError("step_delay must be nonnegative and finite")
    if not isfinite(stage_pause) or stage_pause < 0.0:
        raise ValueError("stage_pause must be nonnegative and finite")
    for value, name in (
        (arm_max_steps, "arm_max_steps"),
        (contact_settle_steps, "contact_settle_steps"),
        (final_hold_steps, "final_hold_steps"),
        (waypoint_count, "waypoint_count"),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    scene = world.scene
    initial_cube_position, _ = world.get_cube_pose()
    pregrasp_target = (
        initial_cube_position[0],
        initial_cube_position[1],
        initial_cube_position[2] + pregrasp_clearance,
    )
    grasp_target = (
        initial_cube_position[0],
        initial_cube_position[1],
        initial_cube_position[2] + grasp_target_z_offset,
    )
    lift_target = (
        grasp_target[0],
        grasp_target[1],
        grasp_target[2] + lift_distance,
    )

    arm = ArmController(world.client_id, scene.robot_id)
    gripper = PandaGripper(world.client_id, scene.robot_id)
    constraint_count_before = pybullet.getNumConstraints(
        physicsClientId=world.client_id
    )
    open_result: GripperResult | None = None
    close_result: GripperResult | None = None
    pregrasp_result: ReachResult | None = None
    approach_result: ReachResult | None = None
    lift_result: ReachResult | None = None
    grasp_contacts: tuple[FingerContact, ...] = ()
    end_effector_at_grasp = arm.end_effector_pose()[0]
    pregrasp_trajectory = (end_effector_at_grasp,)
    approach_trajectory = (end_effector_at_grasp,)
    lift_trajectory = (end_effector_at_grasp,)
    pregrasp_waypoint_results: tuple[ReachResult, ...] = ()
    approach_waypoint_results: tuple[ReachResult, ...] = ()
    lift_waypoint_results: tuple[ReachResult, ...] = ()
    gripper_steps = 0

    def finish(success: bool, failure_reason: str | None) -> PickResult:
        final_cube_position, _ = world.get_cube_pose()
        final_end_effector_position, _ = get_end_effector_pose(
            world.client_id,
            scene.robot_id,
        )
        final_contacts = _finger_cube_contacts(
            world.client_id,
            scene.robot_id,
            scene.cube_id,
        )
        cube_lift = final_cube_position[2] - initial_cube_position[2]
        horizontal_displacement = hypot(
            final_cube_position[0] - initial_cube_position[0],
            final_cube_position[1] - initial_cube_position[1],
        )
        arm_steps = sum(
            result.steps
            for result in (
                *pregrasp_waypoint_results,
                *approach_waypoint_results,
                *lift_waypoint_results,
            )
        )
        constraint_count_after = pybullet.getNumConstraints(
            physicsClientId=world.client_id
        )
        return PickResult(
            success=success,
            initial_cube_position=initial_cube_position,
            pregrasp_target_position=pregrasp_target,
            grasp_target_position=grasp_target,
            lift_target_position=lift_target,
            pregrasp_trajectory=pregrasp_trajectory,
            approach_trajectory=approach_trajectory,
            lift_trajectory=lift_trajectory,
            end_effector_position_at_grasp=end_effector_at_grasp,
            final_end_effector_position=final_end_effector_position,
            final_cube_position=final_cube_position,
            cube_lift_distance=cube_lift,
            final_cube_to_end_effector_distance=dist(
                final_cube_position,
                final_end_effector_position,
            ),
            horizontal_cube_displacement=horizontal_displacement,
            grasp_contacts=grasp_contacts,
            final_contacts=final_contacts,
            open_result=open_result,
            close_result=close_result,
            pregrasp_result=pregrasp_result,
            approach_result=approach_result,
            lift_result=lift_result,
            arm_steps=arm_steps,
            gripper_steps=gripper_steps,
            total_steps=arm_steps + gripper_steps,
            constraint_count_before=constraint_count_before,
            constraint_count_after=constraint_count_after,
            failure_reason=failure_reason,
        )

    open_result = gripper.open(step_delay=step_delay)
    gripper_steps += open_result.steps
    _stage_pause(stage_pause)
    if not open_result.success:
        return finish(False, f"Gripper open failed: {open_result.failure_reason}")

    pregrasp_start = arm.end_effector_pose()[0]
    pregrasp_waypoint_results, pregrasp_trajectory = _move_through_waypoints(
        arm,
        _linear_waypoints(pregrasp_start, pregrasp_target, waypoint_count),
        position_tolerance=position_tolerance,
        max_steps=arm_max_steps,
        step_delay=step_delay,
    )
    pregrasp_result = pregrasp_waypoint_results[-1]
    _stage_pause(stage_pause)
    if not pregrasp_result.success:
        return finish(False, f"Pre-grasp approach failed: {pregrasp_result.failure_reason}")

    approach_start = arm.end_effector_pose()[0]
    approach_waypoint_results, approach_trajectory = _move_through_waypoints(
        arm,
        _vertical_waypoints(approach_start, grasp_target, waypoint_count),
        position_tolerance=position_tolerance,
        max_steps=arm_max_steps,
        step_delay=step_delay,
    )
    approach_result = approach_waypoint_results[-1]
    end_effector_at_grasp = approach_result.final_position
    _stage_pause(stage_pause)
    if not approach_result.success:
        return finish(False, f"Grasp approach failed: {approach_result.failure_reason}")

    close_result = gripper.close(step_delay=step_delay)
    gripper_steps += close_result.steps
    if not close_result.success:
        return finish(False, f"Gripper close failed: {close_result.failure_reason}")

    settle_result = gripper.hold(
        steps=contact_settle_steps,
        step_delay=step_delay,
    )
    gripper_steps += settle_result.steps
    grasp_contacts = _finger_cube_contacts(
        world.client_id,
        scene.robot_id,
        scene.cube_id,
    )
    _stage_pause(stage_pause)
    contact_links = {contact.finger_link_index for contact in grasp_contacts}
    if contact_links != set(PANDA_FINGER_JOINT_INDICES):
        return finish(False, "Cube was not contacted by both fingers")

    lift_start = arm.end_effector_pose()[0]
    lift_waypoint_results, lift_trajectory = _move_through_waypoints(
        arm,
        _vertical_waypoints(lift_start, lift_target, waypoint_count),
        position_tolerance=position_tolerance,
        max_steps=arm_max_steps,
        step_delay=step_delay,
    )
    lift_result = lift_waypoint_results[-1]
    if not lift_result.success:
        return finish(False, f"Lift failed: {lift_result.failure_reason}")

    hold_result = gripper.hold(steps=final_hold_steps, step_delay=step_delay)
    gripper_steps += hold_result.steps
    _stage_pause(stage_pause)
    result = finish(True, None)
    if result.constraint_count_after != result.constraint_count_before:
        return finish(False, "A simulation constraint was created during the pick")
    if result.cube_lift_distance < MINIMUM_SUCCESSFUL_LIFT:
        return finish(False, "Cube was not lifted by the required distance")
    if result.final_cube_to_end_effector_distance > MAXIMUM_CUBE_TO_EE_DISTANCE:
        return finish(False, "Cube did not remain close to the end effector")
    if result.horizontal_cube_displacement > MAXIMUM_HORIZONTAL_DISPLACEMENT:
        return finish(False, "Cube was knocked too far from its initial horizontal position")
    return result


def _linear_waypoints(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    count: int,
) -> tuple[tuple[float, float, float], ...]:
    return tuple(
        (
            start[0] + (end[0] - start[0]) * step / count,
            start[1] + (end[1] - start[1]) * step / count,
            start[2] + (end[2] - start[2]) * step / count,
        )
        for step in range(1, count + 1)
    )


def _vertical_waypoints(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    count: int,
) -> tuple[tuple[float, float, float], ...]:
    return tuple(
        (
            end[0],
            end[1],
            start[2] + (end[2] - start[2]) * step / count,
        )
        for step in range(1, count + 1)
    )


def _move_through_waypoints(
    arm: ArmController,
    waypoints: tuple[tuple[float, float, float], ...],
    *,
    position_tolerance: float,
    max_steps: int,
    step_delay: float,
) -> tuple[tuple[ReachResult, ...], tuple[tuple[float, float, float], ...]]:
    results: list[ReachResult] = []
    measured_positions = [arm.end_effector_pose()[0]]
    for waypoint in waypoints:
        result = arm.move_end_effector(
            waypoint,
            DEFAULT_END_EFFECTOR_ORIENTATION,
            position_tolerance=position_tolerance,
            max_steps=max_steps,
            step_delay=step_delay,
        )
        results.append(result)
        measured_positions.append(result.final_position)
        if not result.success:
            break
    return tuple(results), tuple(measured_positions)


def _finger_cube_contacts(
    client_id: int,
    robot_id: int,
    cube_id: int,
) -> tuple[FingerContact, ...]:
    contacts = pybullet.getContactPoints(
        bodyA=robot_id,
        bodyB=cube_id,
        physicsClientId=client_id,
    )
    return tuple(
        FingerContact(
            finger_link_index=int(contact[3]),
            position_on_finger=tuple(float(value) for value in contact[5]),
            position_on_cube=tuple(float(value) for value in contact[6]),
            contact_distance=float(contact[8]),
            normal_force=float(contact[9]),
        )
        for contact in contacts
        if int(contact[3]) in PANDA_FINGER_JOINT_INDICES
    )


def _stage_pause(duration: float) -> None:
    if duration > 0.0:
        sleep(duration)
