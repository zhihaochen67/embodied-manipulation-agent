from collections.abc import Iterator

import numpy as np
import pybullet
import pytest

from embodied_manipulation.control.ik import solve_inverse_kinematics
from embodied_manipulation.control.pick_place import (
    ANGULAR_SPEED_THRESHOLD,
    HORIZONTAL_CONTAINMENT_TOLERANCE,
    LINEAR_SPEED_THRESHOLD,
    MAXIMUM_TRANSPORT_FINGER_TRACKING_ERROR,
    RELEASE_BOTTOM_CLEARANCE,
    VERTICAL_POSITION_TOLERANCE,
    evaluate_placement,
    oracle_pick_place,
)
from embodied_manipulation.simulation import World
from embodied_manipulation.simulation.objects import (
    CUBE_HALF_EXTENT,
    TABLE_SURFACE_Z,
    TRAY_CENTER,
    TRAY_FLOOR_THICKNESS,
    TRAY_INNER_HALF_EXTENTS,
    TRAY_WALL_HEIGHT,
    TRAY_WALL_THICKNESS,
)
from embodied_manipulation.simulation.robot import PANDA_FINGER_JOINT_INDICES


@pytest.fixture
def world() -> Iterator[World]:
    simulation = World(gui=False)
    simulation.reset(seed=0, include_tray=True)
    try:
        yield simulation
    finally:
        simulation.disconnect()


def test_tray_geometry_is_valid_and_deterministic(world: World) -> None:
    assert world.scene is not None and world.scene.tray is not None
    first = world.scene.tray
    first_pose = world.get_tray_pose()

    assert len(set(first.body_ids)) == 5
    assert first.center == TRAY_CENTER
    assert first.inner_bounds == pytest.approx(
        (
            TRAY_CENTER[0] - TRAY_INNER_HALF_EXTENTS[0],
            TRAY_CENTER[0] + TRAY_INNER_HALF_EXTENTS[0],
            TRAY_CENTER[1] - TRAY_INNER_HALF_EXTENTS[1],
            TRAY_CENTER[1] + TRAY_INNER_HALF_EXTENTS[1],
        )
    )
    assert first.floor_height == pytest.approx(
        TABLE_SURFACE_Z + TRAY_FLOOR_THICKNESS
    )
    assert first.wall_height == pytest.approx(TRAY_WALL_HEIGHT)
    assert first.wall_thickness == pytest.approx(TRAY_WALL_THICKNESS)
    assert 2.0 * TRAY_INNER_HALF_EXTENTS[0] > 2.0 * CUBE_HALF_EXTENT
    assert 2.0 * TRAY_INNER_HALF_EXTENTS[1] > 2.0 * CUBE_HALF_EXTENT

    second = world.reset(seed=0, include_tray=True).tray
    assert second is not None
    assert second.center == first.center
    assert second.inner_bounds == pytest.approx(first.inner_bounds)
    assert second.floor_height == pytest.approx(first.floor_height)
    second_pose = world.get_tray_pose()
    np.testing.assert_allclose(second_pose[0], first_pose[0], atol=0.0)
    np.testing.assert_allclose(second_pose[1], first_pose[1], atol=0.0)


def test_cube_and_tray_do_not_overlap_initially(world: World) -> None:
    assert world.scene is not None and world.scene.tray is not None
    for tray_body_id in world.scene.tray.body_ids:
        assert not pybullet.getClosestPoints(
            world.scene.cube_id,
            tray_body_id,
            distance=0.0,
            physicsClientId=world.client_id,
        )


def test_tray_center_release_target_has_valid_ik(world: World) -> None:
    assert world.scene is not None and world.scene.tray is not None
    tray = world.scene.tray
    release_target = (
        tray.center[0],
        tray.center[1],
        tray.floor_height + CUBE_HALF_EXTENT + RELEASE_BOTTOM_CLEARANCE,
    )
    targets = solve_inverse_kinematics(
        world.client_id,
        world.scene.robot_id,
        release_target,
    )

    assert len(targets) == 7
    assert np.all(np.isfinite(targets))


def test_seed_zero_oracle_pick_place_succeeds_physically(world: World) -> None:
    assert world.scene is not None and world.scene.tray is not None
    result = oracle_pick_place(world)
    tray = world.scene.tray
    linear_speed = np.linalg.norm(result.final_cube_linear_velocity)
    angular_speed = np.linalg.norm(result.final_cube_angular_velocity)

    assert result.success, result.failure_reason
    assert result.pick_success
    assert result.cube_held_during_transport
    assert result.release_result is not None and result.release_result.success
    assert result.release_result.reached_target
    assert result.cube_position_after_release is not None
    assert result.cube_inside_tray
    assert result.cube_near_floor
    assert result.cube_released
    assert result.stable
    assert linear_speed <= LINEAR_SPEED_THRESHOLD
    assert angular_speed <= ANGULAR_SPEED_THRESHOLD
    assert (
        abs(result.final_cube_position[2] - (tray.floor_height + CUBE_HALF_EXTENT))
        <= VERTICAL_POSITION_TOLERANCE
    )
    assert result.release_clearance_above_floor == pytest.approx(
        RELEASE_BOTTOM_CLEARANCE
    )
    assert result.expected_cube_center_at_release[2] == pytest.approx(
        tray.floor_height + CUBE_HALF_EXTENT + RELEASE_BOTTOM_CLEARANCE
    )


def test_transport_finger_hold_tracks_captured_targets(world: World) -> None:
    result = oracle_pick_place(world)

    assert result.success, result.failure_reason
    assert result.transport_finger_targets is not None
    assert result.maximum_transport_finger_tracking_errors is not None
    assert max(result.maximum_transport_finger_tracking_errors) <= (
        MAXIMUM_TRANSPORT_FINGER_TRACKING_ERROR
    )
    assert result.transport_bilateral_contact_fraction >= 0.99
    assert result.cube_held_during_transport


def test_release_leaves_no_finger_hold_or_artificial_constraint(
    world: World,
) -> None:
    assert world.scene is not None
    result = oracle_pick_place(world)
    contacts = pybullet.getContactPoints(
        bodyA=world.scene.robot_id,
        bodyB=world.scene.cube_id,
        physicsClientId=world.client_id,
    )
    finger_contacts = [
        contact
        for contact in contacts
        if int(contact[3]) in PANDA_FINGER_JOINT_INDICES
    ]

    assert result.success, result.failure_reason
    assert result.cube_position_before_release is not None
    assert result.cube_bottom_before_release is not None
    assert result.actual_release_clearance_above_floor is not None
    assert result.cube_position_shortly_after_release is not None
    assert result.observed_vertical_fall_distance is not None
    assert result.cube_bottom_before_release > result.tray_floor_height
    assert result.actual_release_clearance_above_floor == pytest.approx(
        RELEASE_BOTTOM_CLEARANCE,
        abs=0.005,
    )
    assert result.cube_position_shortly_after_release[2] < (
        result.cube_position_before_release[2]
    )
    assert result.observed_vertical_fall_distance >= 0.01
    assert result.cube_inside_tray
    assert result.cube_near_floor
    assert not finger_contacts
    assert result.constraint_count_before == 0
    assert result.constraint_count_after == result.constraint_count_before
    assert pybullet.getNumConstraints(physicsClientId=world.client_id) == 0


def test_seed_zero_pick_place_is_reproducible() -> None:
    results = []
    for _ in range(2):
        with World(gui=False) as world:
            world.reset(seed=0, include_tray=True)
            results.append(oracle_pick_place(world))

    assert results[0].success and results[1].success
    np.testing.assert_allclose(
        results[0].final_cube_position,
        results[1].final_cube_position,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        results[0].final_cube_linear_velocity,
        results[1].final_cube_linear_velocity,
        atol=1e-8,
    )
    assert results[1].total_steps == results[0].total_steps


def test_placement_evaluator_rejects_cube_outside_tray(world: World) -> None:
    assert world.scene is not None and world.scene.tray is not None
    tray = world.scene.tray
    outside_position = (
        tray.inner_bounds[1]
        + CUBE_HALF_EXTENT
        + HORIZONTAL_CONTAINMENT_TOLERANCE
        + 0.02,
        tray.center[1],
        TABLE_SURFACE_Z + CUBE_HALF_EXTENT,
    )
    pybullet.resetBasePositionAndOrientation(
        world.scene.cube_id,
        outside_position,
        (0.0, 0.0, 0.0, 1.0),
        physicsClientId=world.client_id,
    )
    for _ in range(60):
        world.step()

    evaluation = evaluate_placement(world)

    assert not evaluation.success
    assert not evaluation.cube_inside_tray
    assert evaluation.failure_reason == "cube_outside_tray"
