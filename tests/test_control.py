from collections.abc import Iterator

import numpy as np
import pybullet
import pytest

from embodied_manipulation.control import (
    ArmController,
    IKError,
    solve_inverse_kinematics,
    target_above_cube,
)
from embodied_manipulation.simulation import World
from embodied_manipulation.simulation.robot import (
    PANDA_ARM_JOINT_INDICES,
    PANDA_ARM_JOINT_NAMES,
    PANDA_END_EFFECTOR_LINK_INDEX,
    PANDA_END_EFFECTOR_LINK_NAME,
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


def test_panda_control_metadata_matches_loaded_model(world: World) -> None:
    assert world.scene is not None
    robot_id = world.scene.robot_id
    assert len(PANDA_ARM_JOINT_INDICES) == 7
    assert PANDA_END_EFFECTOR_LINK_INDEX < pybullet.getNumJoints(
        robot_id,
        physicsClientId=world.client_id,
    )

    arm_info = [
        pybullet.getJointInfo(robot_id, index, physicsClientId=world.client_id)
        for index in PANDA_ARM_JOINT_INDICES
    ]
    finger_info = [
        pybullet.getJointInfo(robot_id, index, physicsClientId=world.client_id)
        for index in PANDA_FINGER_JOINT_INDICES
    ]
    end_effector_info = pybullet.getJointInfo(
        robot_id,
        PANDA_END_EFFECTOR_LINK_INDEX,
        physicsClientId=world.client_id,
    )

    assert tuple(info[1].decode() for info in arm_info) == PANDA_ARM_JOINT_NAMES
    assert all(info[2] == pybullet.JOINT_REVOLUTE for info in arm_info)
    assert tuple(info[1].decode() for info in finger_info) == PANDA_FINGER_JOINT_NAMES
    assert all(info[2] == pybullet.JOINT_PRISMATIC for info in finger_info)
    assert end_effector_info[12].decode() == PANDA_END_EFFECTOR_LINK_NAME
    assert end_effector_info[2] == pybullet.JOINT_FIXED


@pytest.mark.parametrize(
    "target",
    [
        (0.50, 0.00, 0.40),
        (0.40, 0.20, 0.35),
    ],
)
def test_ik_returns_finite_seven_joint_targets(
    world: World,
    target: tuple[float, float, float],
) -> None:
    assert world.scene is not None
    targets = solve_inverse_kinematics(
        world.client_id,
        world.scene.robot_id,
        target,
    )

    assert len(targets) == 7
    assert np.all(np.isfinite(targets))
    for joint_index, target_value in zip(
        PANDA_ARM_JOINT_INDICES,
        targets,
        strict=True,
    ):
        info = pybullet.getJointInfo(
            world.scene.robot_id,
            joint_index,
            physicsClientId=world.client_id,
        )
        assert info[8] <= target_value <= info[9]


def test_reaches_known_target(world: World) -> None:
    assert world.scene is not None
    controller = ArmController(world.client_id, world.scene.robot_id)
    result = controller.move_end_effector((0.50, 0.00, 0.40))

    assert result.success
    assert result.steps <= 480
    assert result.position_error <= 0.01
    assert len(result.final_joint_positions) == 7


def test_reaches_above_cube_without_moving_cube(world: World) -> None:
    assert world.scene is not None
    controller = ArmController(world.client_id, world.scene.robot_id)
    cube_before, _ = world.get_cube_pose()
    target = target_above_cube(cube_before)
    result = controller.move_end_effector(target)
    cube_after, _ = world.get_cube_pose()

    assert result.success
    assert result.position_error <= 0.01
    assert target[2] == pytest.approx(cube_before[2] + 0.25)
    np.testing.assert_allclose(cube_after, cube_before, atol=1e-4)


def test_reach_is_deterministic(world: World) -> None:
    assert world.scene is not None
    cube_position, _ = world.get_cube_pose()
    target = target_above_cube(cube_position)
    first = ArmController(
        world.client_id,
        world.scene.robot_id,
    ).move_end_effector(target)

    second_scene = world.reset(seed=0)
    second = ArmController(
        world.client_id,
        second_scene.robot_id,
    ).move_end_effector(target)

    assert first.success and second.success
    np.testing.assert_allclose(
        first.final_joint_positions,
        second.final_joint_positions,
        atol=1e-8,
    )
    assert second.position_error == pytest.approx(first.position_error, abs=1e-8)


def test_invalid_ik_inputs_fail_cleanly(world: World) -> None:
    assert world.scene is not None
    robot_id = world.scene.robot_id
    with pytest.raises(ValueError, match="finite"):
        solve_inverse_kinematics(
            world.client_id,
            robot_id,
            (float("nan"), 0.0, 0.4),
        )
    with pytest.raises(ValueError, match="exactly three"):
        ArmController(world.client_id, robot_id).move_end_effector((0.5, 0.0))
    with pytest.raises(ValueError, match="nonzero"):
        solve_inverse_kinematics(
            world.client_id,
            robot_id,
            (0.5, 0.0, 0.4),
            (0.0, 0.0, 0.0, 0.0),
        )


def test_unreachable_ik_target_fails_clearly(world: World) -> None:
    assert world.scene is not None
    with pytest.raises(IKError, match="residual"):
        solve_inverse_kinematics(
            world.client_id,
            world.scene.robot_id,
            (5.0, 5.0, 5.0),
        )


def test_controller_timeout_does_not_hang(world: World) -> None:
    assert world.scene is not None
    controller = ArmController(world.client_id, world.scene.robot_id)
    result = controller.move_end_effector(
        (0.50, 0.00, 0.40),
        max_steps=1,
    )

    assert not result.success
    assert result.steps == 1
    assert result.failure_reason is not None
    assert "Timed out" in result.failure_reason
