"""Oracle physical pick-and-place for the deterministic Phase 4 scene."""

from dataclasses import dataclass
from math import dist, hypot, isfinite
from time import sleep

import pybullet

from embodied_manipulation.simulation import Tray, World
from embodied_manipulation.simulation.objects import CUBE_HALF_EXTENT
from embodied_manipulation.simulation.robot import (
    PANDA_FINGER_JOINT_INDICES,
    get_end_effector_pose,
)

from .arm_controller import ArmController, ReachResult
from .gripper import GripperResult, PandaGripper
from .pick import (
    CARTESIAN_WAYPOINT_COUNT,
    MAXIMUM_CUBE_TO_EE_DISTANCE,
    PickResult,
    _finger_cube_contacts,
    _linear_waypoints,
    _move_through_waypoints,
    _vertical_waypoints,
    oracle_pick,
)

TRANSPORT_WALL_CLEARANCE = 0.12
TRANSPORT_HOLD_INWARD_PRELOAD = 0.001
MAXIMUM_TRANSPORT_FINGER_TRACKING_ERROR = 0.003
RELEASE_BOTTOM_CLEARANCE = 0.015
RELEASE_OBSERVATION_STEPS = 4
PLACEMENT_POSITION_TOLERANCE = 0.004
SETTLING_STEPS = 240
POST_RETREAT_STEPS = 60
HORIZONTAL_CONTAINMENT_TOLERANCE = 0.002
VERTICAL_POSITION_TOLERANCE = 0.012
LINEAR_SPEED_THRESHOLD = 0.02
ANGULAR_SPEED_THRESHOLD = 0.10


@dataclass(frozen=True, slots=True)
class PlacementEvaluation:
    """Measured geometric and dynamic placement checks."""

    success: bool
    cube_position: tuple[float, float, float]
    cube_linear_velocity: tuple[float, float, float]
    cube_angular_velocity: tuple[float, float, float]
    cube_inside_tray: bool
    cube_near_floor: bool
    cube_released: bool
    stable: bool
    cube_to_tray_center_distance: float
    failure_reason: str | None = None


@dataclass(frozen=True, slots=True)
class PickPlaceResult:
    """Outcome and measured state of one oracle pick-and-place sequence."""

    success: bool
    pick_success: bool
    initial_cube_position: tuple[float, float, float]
    tray_center: tuple[float, float]
    tray_inner_bounds: tuple[float, float, float, float]
    tray_floor_height: float
    pick_result: PickResult
    transport_targets: tuple[tuple[float, float, float], ...]
    above_tray_target: tuple[float, float, float]
    release_target: tuple[float, float, float]
    expected_cube_center_at_release: tuple[float, float, float]
    release_clearance_above_floor: float
    cube_held_during_transport: bool
    transport_finger_targets: tuple[float, float] | None
    transport_finger_position_ranges: (
        tuple[tuple[float, float], tuple[float, float]] | None
    )
    maximum_transport_finger_tracking_errors: tuple[float, float] | None
    transport_bilateral_contact_fraction: float
    maximum_transport_cube_to_ee_deviation: float | None
    release_result: GripperResult | None
    cube_position_before_release: tuple[float, float, float] | None
    cube_bottom_before_release: float | None
    actual_release_clearance_above_floor: float | None
    cube_position_shortly_after_release: tuple[float, float, float] | None
    cube_position_after_release: tuple[float, float, float] | None
    observed_vertical_fall_distance: float | None
    final_cube_position: tuple[float, float, float]
    final_cube_linear_velocity: tuple[float, float, float]
    final_cube_angular_velocity: tuple[float, float, float]
    cube_inside_tray: bool
    cube_near_floor: bool
    cube_released: bool
    stable: bool
    cube_to_tray_center_distance: float
    arm_steps: int
    gripper_steps: int
    settling_steps: int
    total_steps: int
    constraint_count_before: int
    constraint_count_after: int
    failure_reason: str | None = None


def oracle_pick_place(
    world: World,
    *,
    arm_max_steps: int = 480,
    waypoint_count: int = CARTESIAN_WAYPOINT_COUNT,
    settling_steps: int = SETTLING_STEPS,
    post_retreat_steps: int = POST_RETREAT_STEPS,
    step_delay: float = 0.0,
    stage_pause: float = 0.0,
) -> PickPlaceResult:
    """Pick the cube, transport it over the tray, release, settle, and retreat."""
    if world.scene is None:
        raise RuntimeError("World must be reset before oracle_pick_place")
    if world.scene.tray is None:
        raise RuntimeError("oracle_pick_place requires reset(include_tray=True)")
    for value, name in (
        (arm_max_steps, "arm_max_steps"),
        (waypoint_count, "waypoint_count"),
        (settling_steps, "settling_steps"),
        (post_retreat_steps, "post_retreat_steps"),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for value, name in (
        (step_delay, "step_delay"),
        (stage_pause, "stage_pause"),
    ):
        if not isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be nonnegative and finite")

    scene = world.scene
    tray = scene.tray
    (
        tray_center,
        tray_inner_bounds,
        tray_floor_height,
        tray_wall_top_height,
    ) = _measured_tray_geometry(world, tray)
    constraint_count_before = pybullet.getNumConstraints(
        physicsClientId=world.client_id
    )
    pick_result = oracle_pick(
        world,
        arm_max_steps=arm_max_steps,
        waypoint_count=waypoint_count,
        step_delay=step_delay,
        stage_pause=stage_pause,
    )

    arm = ArmController(world.client_id, scene.robot_id)
    gripper = PandaGripper(world.client_id, scene.robot_id)
    arm_results: list[ReachResult] = []
    release_result: GripperResult | None = None
    cube_position_after_release: tuple[float, float, float] | None = None
    completed_settling_steps = 0
    cube_held_during_transport = False
    transport_finger_targets: tuple[float, float] | None = None
    transport_samples: list[
        tuple[
            tuple[float, float],
            set[int],
            tuple[float, float, float],
        ]
    ] = []
    transport_finger_position_ranges: (
        tuple[tuple[float, float], tuple[float, float]] | None
    ) = None
    maximum_transport_finger_tracking_errors: tuple[float, float] | None = None
    transport_bilateral_contact_fraction = 0.0
    maximum_transport_cube_to_ee_deviation: float | None = None
    cube_position_before_release: tuple[float, float, float] | None = None
    cube_bottom_before_release: float | None = None
    actual_release_clearance_above_floor: float | None = None
    cube_position_shortly_after_release: tuple[float, float, float] | None = None
    release_samples: list[
        tuple[tuple[float, float, float], set[int]]
    ] = []

    final_ee_position = arm.end_effector_pose()[0]
    final_cube_position, _ = world.get_cube_pose()
    cube_to_ee_offset = tuple(
        cube - ee
        for cube, ee in zip(final_cube_position, final_ee_position, strict=True)
    )
    expected_release_center = (
        tray_center[0],
        tray_center[1],
        tray_floor_height + CUBE_HALF_EXTENT + RELEASE_BOTTOM_CLEARANCE,
    )
    release_target = tuple(
        cube - offset
        for cube, offset in zip(
            expected_release_center,
            cube_to_ee_offset,
            strict=True,
        )
    )
    minimum_transport_cube_z = (
        tray_wall_top_height + CUBE_HALF_EXTENT + TRANSPORT_WALL_CLEARANCE
    )
    transport_cube_z = max(final_cube_position[2], minimum_transport_cube_z)
    transport_ee_z = transport_cube_z - cube_to_ee_offset[2]
    safe_transport_target = (
        final_ee_position[0],
        final_ee_position[1],
        transport_ee_z,
    )
    above_tray_target = (
        tray_center[0] - cube_to_ee_offset[0],
        tray_center[1] - cube_to_ee_offset[1],
        transport_ee_z,
    )
    transport_targets = (safe_transport_target, above_tray_target)

    def finish(success: bool, failure_reason: str | None) -> PickPlaceResult:
        evaluation = evaluate_placement(world)
        constraint_count_after = pybullet.getNumConstraints(
            physicsClientId=world.client_id
        )
        arm_steps = pick_result.arm_steps + sum(
            result.steps for result in arm_results
        )
        gripper_steps = pick_result.gripper_steps
        if release_result is not None:
            gripper_steps += release_result.steps
        return PickPlaceResult(
            success=success,
            pick_success=pick_result.success,
            initial_cube_position=pick_result.initial_cube_position,
            tray_center=tray_center,
            tray_inner_bounds=tray_inner_bounds,
            tray_floor_height=tray_floor_height,
            pick_result=pick_result,
            transport_targets=transport_targets,
            above_tray_target=above_tray_target,
            release_target=release_target,
            expected_cube_center_at_release=expected_release_center,
            release_clearance_above_floor=RELEASE_BOTTOM_CLEARANCE,
            cube_held_during_transport=cube_held_during_transport,
            transport_finger_targets=transport_finger_targets,
            transport_finger_position_ranges=transport_finger_position_ranges,
            maximum_transport_finger_tracking_errors=(
                maximum_transport_finger_tracking_errors
            ),
            transport_bilateral_contact_fraction=(
                transport_bilateral_contact_fraction
            ),
            maximum_transport_cube_to_ee_deviation=(
                maximum_transport_cube_to_ee_deviation
            ),
            release_result=release_result,
            cube_position_before_release=cube_position_before_release,
            cube_bottom_before_release=cube_bottom_before_release,
            actual_release_clearance_above_floor=(
                actual_release_clearance_above_floor
            ),
            cube_position_shortly_after_release=(
                cube_position_shortly_after_release
            ),
            cube_position_after_release=cube_position_after_release,
            observed_vertical_fall_distance=(
                None
                if cube_position_before_release is None
                else max(
                    0.0,
                    cube_position_before_release[2] - evaluation.cube_position[2],
                )
            ),
            final_cube_position=evaluation.cube_position,
            final_cube_linear_velocity=evaluation.cube_linear_velocity,
            final_cube_angular_velocity=evaluation.cube_angular_velocity,
            cube_inside_tray=evaluation.cube_inside_tray,
            cube_near_floor=evaluation.cube_near_floor,
            cube_released=evaluation.cube_released,
            stable=evaluation.stable,
            cube_to_tray_center_distance=evaluation.cube_to_tray_center_distance,
            arm_steps=arm_steps,
            gripper_steps=gripper_steps,
            settling_steps=completed_settling_steps,
            total_steps=arm_steps + gripper_steps + completed_settling_steps,
            constraint_count_before=constraint_count_before,
            constraint_count_after=constraint_count_after,
            failure_reason=failure_reason,
        )

    if not pick_result.success:
        return finish(False, "pick_failed")

    transport_finger_targets = gripper.capture_hold(
        inward_preload=TRANSPORT_HOLD_INWARD_PRELOAD
    )
    initial_transport_relative_position = tuple(
        cube - ee
        for cube, ee in zip(
            world.get_cube_pose()[0],
            arm.end_effector_pose()[0],
            strict=True,
        )
    )

    def record_transport_state() -> None:
        positions = gripper.finger_joint_positions()
        contacts = {
            contact.finger_link_index
            for contact in _finger_cube_contacts(
                world.client_id,
                scene.robot_id,
                scene.cube_id,
            )
        }
        cube_position, _ = world.get_cube_pose()
        end_effector_position = arm.end_effector_pose()[0]
        relative_position = tuple(
            cube - ee
            for cube, ee in zip(
                cube_position,
                end_effector_position,
                strict=True,
            )
        )
        transport_samples.append((positions, contacts, relative_position))

    safe_results, _ = _move_through_waypoints(
        arm,
        _vertical_waypoints(
            arm.end_effector_pose()[0],
            safe_transport_target,
            max(2, waypoint_count // 2),
        ),
        position_tolerance=PLACEMENT_POSITION_TOLERANCE,
        max_steps=arm_max_steps,
        step_delay=step_delay,
        before_step=gripper.maintain,
        after_step=record_transport_state,
    )
    arm_results.extend(safe_results)
    if not safe_results[-1].success:
        return finish(False, "transport_failed")

    horizontal_results, _ = _move_through_waypoints(
        arm,
        _linear_waypoints(
            arm.end_effector_pose()[0],
            above_tray_target,
            waypoint_count,
        ),
        position_tolerance=PLACEMENT_POSITION_TOLERANCE,
        max_steps=arm_max_steps,
        step_delay=step_delay,
        before_step=gripper.maintain,
        after_step=record_transport_state,
    )
    arm_results.extend(horizontal_results)
    if not horizontal_results[-1].success:
        return finish(False, "transport_failed")

    maximum_transport_finger_tracking_errors = tuple(
        max(
            abs(sample[0][finger] - transport_finger_targets[finger])
            for sample in transport_samples
        )
        for finger in range(2)
    )
    transport_finger_position_ranges = tuple(
        (
            min(sample[0][finger] for sample in transport_samples),
            max(sample[0][finger] for sample in transport_samples),
        )
        for finger in range(2)
    )
    transport_bilateral_contact_fraction = sum(
        sample[1] == set(PANDA_FINGER_JOINT_INDICES)
        for sample in transport_samples
    ) / len(transport_samples)
    maximum_transport_cube_to_ee_deviation = max(
        dist(initial_transport_relative_position, sample[2])
        for sample in transport_samples
    )
    transport_contacts = _finger_cube_contacts(
        world.client_id,
        scene.robot_id,
        scene.cube_id,
    )
    transported_cube_position, _ = world.get_cube_pose()
    transported_ee_position = arm.end_effector_pose()[0]
    cube_held_during_transport = (
        {contact.finger_link_index for contact in transport_contacts}
        == set(PANDA_FINGER_JOINT_INDICES)
        and dist(transported_cube_position, transported_ee_position)
        <= MAXIMUM_CUBE_TO_EE_DISTANCE
    )
    _pause(stage_pause)
    if not cube_held_during_transport:
        return finish(False, "transport_failed")

    placement_results, _ = _move_through_waypoints(
        arm,
        _vertical_waypoints(
            arm.end_effector_pose()[0],
            release_target,
            waypoint_count,
        ),
        position_tolerance=PLACEMENT_POSITION_TOLERANCE,
        max_steps=arm_max_steps,
        step_delay=step_delay,
        before_step=gripper.maintain,
    )
    arm_results.extend(placement_results)
    if not placement_results[-1].success:
        return finish(False, "placement_approach_failed")

    before_release_contacts = _finger_cube_contacts(
        world.client_id,
        scene.robot_id,
        scene.cube_id,
    )
    if {
        contact.finger_link_index for contact in before_release_contacts
    } != set(PANDA_FINGER_JOINT_INDICES):
        return finish(False, "placement_approach_failed")

    cube_position_before_release, _ = world.get_cube_pose()
    cube_bottom_before_release = float(
        pybullet.getAABB(
            scene.cube_id,
            physicsClientId=world.client_id,
        )[0][2]
    )
    actual_release_clearance_above_floor = (
        cube_bottom_before_release - tray_floor_height
    )

    def record_release_state() -> None:
        cube_position, _ = world.get_cube_pose()
        contacts = {
            contact.finger_link_index
            for contact in _finger_cube_contacts(
                world.client_id,
                scene.robot_id,
                scene.cube_id,
            )
        }
        release_samples.append((cube_position, contacts))

    release_result = gripper.open(
        step_delay=step_delay,
        step_observer=record_release_state,
    )
    if not release_result.success:
        return finish(False, "gripper_release_failed")
    cube_position_after_release, _ = world.get_cube_pose()
    first_free_sample = next(
        (
            index
            for index, sample in enumerate(release_samples)
            if not sample[1]
        ),
        None,
    )
    if first_free_sample is not None:
        observation_index = min(
            first_free_sample + RELEASE_OBSERVATION_STEPS,
            len(release_samples) - 1,
        )
        cube_position_shortly_after_release = release_samples[
            observation_index
        ][0]
    _pause(stage_pause)

    _step_world(world, settling_steps, step_delay)
    completed_settling_steps += settling_steps

    retreat_results, _ = _move_through_waypoints(
        arm,
        _vertical_waypoints(
            arm.end_effector_pose()[0],
            above_tray_target,
            waypoint_count,
        ),
        position_tolerance=PLACEMENT_POSITION_TOLERANCE,
        max_steps=arm_max_steps,
        step_delay=step_delay,
    )
    arm_results.extend(retreat_results)
    if not retreat_results[-1].success:
        return finish(False, "placement_approach_failed")

    _step_world(world, post_retreat_steps, step_delay)
    completed_settling_steps += post_retreat_steps
    _pause(stage_pause)

    evaluation = evaluate_placement(world)
    constraint_count_after = pybullet.getNumConstraints(
        physicsClientId=world.client_id
    )
    if constraint_count_after != constraint_count_before:
        return finish(False, "constraint_count_changed")
    return finish(evaluation.success, evaluation.failure_reason)


def evaluate_placement(world: World) -> PlacementEvaluation:
    """Evaluate complete containment, floor height, release, and stability."""
    if world.scene is None or world.scene.tray is None:
        raise RuntimeError("Placement evaluation requires a scene with a tray")
    scene = world.scene
    tray = scene.tray
    tray_center, tray_inner_bounds, tray_floor_height, _ = (
        _measured_tray_geometry(world, tray)
    )
    cube_position, _ = world.get_cube_pose()
    linear_velocity, angular_velocity = pybullet.getBaseVelocity(
        scene.cube_id,
        physicsClientId=world.client_id,
    )
    cube_aabb = pybullet.getAABB(
        scene.cube_id,
        physicsClientId=world.client_id,
    )
    min_x, max_x, min_y, max_y = tray_inner_bounds
    cube_inside_tray = (
        cube_aabb[0][0] >= min_x - HORIZONTAL_CONTAINMENT_TOLERANCE
        and cube_aabb[1][0] <= max_x + HORIZONTAL_CONTAINMENT_TOLERANCE
        and cube_aabb[0][1] >= min_y - HORIZONTAL_CONTAINMENT_TOLERANCE
        and cube_aabb[1][1] <= max_y + HORIZONTAL_CONTAINMENT_TOLERANCE
    )
    cube_near_floor = (
        abs(cube_aabb[0][2] - tray_floor_height)
        <= VERTICAL_POSITION_TOLERANCE
    )
    linear_speed = _norm(linear_velocity)
    angular_speed = _norm(angular_velocity)
    stable = (
        linear_speed <= LINEAR_SPEED_THRESHOLD
        and angular_speed <= ANGULAR_SPEED_THRESHOLD
    )
    finger_contacts = _finger_cube_contacts(
        world.client_id,
        scene.robot_id,
        scene.cube_id,
    )
    end_effector_position, _ = get_end_effector_pose(
        world.client_id,
        scene.robot_id,
    )
    cube_released = (
        not finger_contacts
        and dist(cube_position, end_effector_position)
        > MAXIMUM_CUBE_TO_EE_DISTANCE
    )
    failure_reason = None
    if not cube_inside_tray:
        failure_reason = "cube_outside_tray"
    elif not cube_near_floor:
        failure_reason = "cube_not_near_tray_floor"
    elif not cube_released:
        failure_reason = "gripper_release_failed"
    elif not stable:
        failure_reason = "cube_unstable"
    return PlacementEvaluation(
        success=failure_reason is None,
        cube_position=tuple(float(value) for value in cube_position),
        cube_linear_velocity=tuple(float(value) for value in linear_velocity),
        cube_angular_velocity=tuple(float(value) for value in angular_velocity),
        cube_inside_tray=cube_inside_tray,
        cube_near_floor=cube_near_floor,
        cube_released=cube_released,
        stable=stable,
        cube_to_tray_center_distance=hypot(
            cube_position[0] - tray_center[0],
            cube_position[1] - tray_center[1],
        ),
        failure_reason=failure_reason,
    )


def _norm(vector: tuple[float, ...]) -> float:
    return sum(component * component for component in vector) ** 0.5


def _measured_tray_geometry(
    world: World,
    tray: Tray,
) -> tuple[
    tuple[float, float],
    tuple[float, float, float, float],
    float,
    float,
]:
    floor_pose, _ = world.get_tray_pose()
    wall_aabbs = tuple(
        pybullet.getAABB(body_id, physicsClientId=world.client_id)
        for body_id in tray.wall_ids
    )
    center = (float(floor_pose[0]), float(floor_pose[1]))
    inner_bounds = (
        float(wall_aabbs[0][1][0]),
        float(wall_aabbs[1][0][0]),
        float(wall_aabbs[2][1][1]),
        float(wall_aabbs[3][0][1]),
    )
    floor_height = float(
        pybullet.getAABB(
            tray.floor_id,
            physicsClientId=world.client_id,
        )[1][2]
    )
    wall_top_height = max(float(bounds[1][2]) for bounds in wall_aabbs)
    return center, inner_bounds, floor_height, wall_top_height


def _step_world(world: World, steps: int, step_delay: float) -> None:
    for _ in range(steps):
        world.step()
        if step_delay > 0.0:
            sleep(step_delay)


def _pause(duration: float) -> None:
    if duration > 0.0:
        sleep(duration)
