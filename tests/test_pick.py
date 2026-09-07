from collections.abc import Iterator

import numpy as np
import pybullet
import pytest

from embodied_manipulation.control.gripper import PandaGripper
from embodied_manipulation.control.pick import (
    CARTESIAN_WAYPOINT_COUNT,
    MAXIMUM_CUBE_TO_EE_DISTANCE,
    MINIMUM_SUCCESSFUL_LIFT,
    oracle_pick,
)
from embodied_manipulation.simulation import World
from embodied_manipulation.simulation.objects import CUBE_HALF_EXTENT
from embodied_manipulation.simulation.robot import (
    PANDA_FINGER_JOINT_INDICES,
    PANDA_FINGER_JOINT_NAMES,
)


@pytest.fixture
def world() -> Iterator[World]:
    simulation = World(gui=False)
    simulation.reset(seed=0)
    try:
        yield simulation
    finally:
        simulation.disconnect()


def test_gripper_metadata_and_limits_match_loaded_model(world: World) -> None:
    assert world.scene is not None
    gripper = PandaGripper(world.client_id, world.scene.robot_id)
    info = tuple(
        pybullet.getJointInfo(
            world.scene.robot_id,
            index,
            physicsClientId=world.client_id,
        )
        for index in PANDA_FINGER_JOINT_INDICES
    )

    assert tuple(item[1].decode() for item in info) == PANDA_FINGER_JOINT_NAMES
    assert all(item[2] == pybullet.JOINT_PRISMATIC for item in info)
    assert gripper.finger_lower_limits == pytest.approx((0.0, 0.0))
    assert gripper.finger_upper_limits == pytest.approx((0.04, 0.04))
    assert gripper.minimum_opening_width == pytest.approx(0.0)
    assert gripper.maximum_opening_width == pytest.approx(0.08)
    assert all(force > 0.0 for force in gripper.finger_forces)
    assert all(velocity > 0.0 for velocity in gripper.finger_max_velocities)


def test_gripper_opens_wider_than_cube_with_unambiguous_width(world: World) -> None:
    assert world.scene is not None
    gripper = PandaGripper(world.client_id, world.scene.robot_id)
    result = gripper.open()

    assert result.success
    assert result.reached_target
    assert result.final_opening_width > 2.0 * CUBE_HALF_EXTENT
    assert result.final_opening_width == pytest.approx(
        sum(result.final_finger_positions)
    )
    assert result.target_finger_positions == pytest.approx((0.04, 0.04))
    finger_aabbs = sorted(
        (
            pybullet.getAABB(
                world.scene.robot_id,
                index,
                physicsClientId=world.client_id,
            )
            for index in PANDA_FINGER_JOINT_INDICES
        ),
        key=lambda bounds: (bounds[0][1] + bounds[1][1]) / 2.0,
    )
    collision_gap = finger_aabbs[1][0][1] - finger_aabbs[0][1][1]
    assert collision_gap > 2.0 * CUBE_HALF_EXTENT


def test_gripper_closes_deterministically_without_hanging(world: World) -> None:
    assert world.scene is not None
    first_gripper = PandaGripper(world.client_id, world.scene.robot_id)
    first_gripper.open()
    first = first_gripper.close()

    second_scene = world.reset(seed=0)
    second_gripper = PandaGripper(world.client_id, second_scene.robot_id)
    second_gripper.open()
    second = second_gripper.close()

    assert first.success and second.success
    assert first.steps <= 240 and second.steps <= 240
    assert first.reached_target and second.reached_target
    np.testing.assert_allclose(
        first.final_finger_positions,
        second.final_finger_positions,
        atol=1e-8,
    )


def test_oracle_pick_physically_lifts_and_holds_cube(world: World) -> None:
    result = oracle_pick(world)

    assert result.success, result.failure_reason
    assert result.cube_lift_distance >= MINIMUM_SUCCESSFUL_LIFT
    assert result.final_cube_to_end_effector_distance <= MAXIMUM_CUBE_TO_EE_DISTANCE
    assert {contact.finger_link_index for contact in result.grasp_contacts} == set(
        PANDA_FINGER_JOINT_INDICES
    )
    assert {contact.finger_link_index for contact in result.final_contacts} == set(
        PANDA_FINGER_JOINT_INDICES
    )
    assert result.close_result is not None and result.close_result.stalled
    assert result.constraint_count_before == 0
    assert result.constraint_count_after == result.constraint_count_before


def test_oracle_pick_approach_and_lift_waypoints_are_monotonic(
    world: World,
) -> None:
    result = oracle_pick(world)
    pregrasp = np.asarray(result.pregrasp_trajectory)
    approach = np.asarray(result.approach_trajectory)
    lift = np.asarray(result.lift_trajectory)
    grasp_axis = np.asarray(result.grasp_target_position[:2])

    assert result.success, result.failure_reason
    assert len(pregrasp) == CARTESIAN_WAYPOINT_COUNT + 1
    assert len(approach) == CARTESIAN_WAYPOINT_COUNT + 1
    assert len(lift) == CARTESIAN_WAYPOINT_COUNT + 1
    assert result.pregrasp_result is not None
    assert abs(result.pregrasp_result.final_orientation[0]) > 0.999
    assert np.all(np.diff(pregrasp[:, 2]) <= 1e-3)
    assert np.all(np.diff(approach[:, 2]) <= 1e-4)
    assert np.max(np.linalg.norm(approach[:, :2] - grasp_axis, axis=1)) <= 0.006
    assert np.all(np.diff(lift[:, 2]) >= -1e-4)
    assert np.max(np.linalg.norm(lift[:, :2] - grasp_axis, axis=1)) <= 0.006


def test_oracle_pick_is_reproducible() -> None:
    results = []
    for _ in range(2):
        with World(gui=False) as world:
            world.reset(seed=0)
            results.append(oracle_pick(world))

    assert results[0].success and results[1].success
    np.testing.assert_allclose(
        results[0].final_cube_position,
        results[1].final_cube_position,
        atol=1e-8,
    )
    assert results[1].cube_lift_distance == pytest.approx(
        results[0].cube_lift_distance,
        abs=1e-8,
    )
    assert results[1].total_steps == results[0].total_steps


def test_unusable_pick_budget_returns_failure_without_hanging(world: World) -> None:
    result = oracle_pick(world, arm_max_steps=1)

    assert not result.success
    assert result.pregrasp_result is not None
    assert result.pregrasp_result.steps == 1
    assert result.lift_result is None
    assert result.failure_reason is not None
    assert "Pre-grasp approach failed" in result.failure_reason
    assert result.total_steps == 2
