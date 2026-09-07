"""Franka Panda loading, metadata, and deterministic initialization."""

from math import pi
from pathlib import Path

import pybullet
import pybullet_data

PANDA_BASE_POSITION = (0.0, 0.0, 0.0)

# Verified against pybullet_data/franka_panda/panda.urdf.
PANDA_ARM_JOINT_INDICES = (0, 1, 2, 3, 4, 5, 6)
PANDA_ARM_JOINT_NAMES = tuple(f"panda_joint{number}" for number in range(1, 8))
PANDA_FINGER_JOINT_INDICES = (9, 10)
PANDA_FINGER_JOINT_NAMES = ("panda_finger_joint1", "panda_finger_joint2")
PANDA_END_EFFECTOR_LINK_INDEX = 11
PANDA_END_EFFECTOR_LINK_NAME = "panda_grasptarget"

PANDA_INITIAL_ARM_POSITIONS = (
    0.0,
    -pi / 4.0,
    0.0,
    -3.0 * pi / 4.0,
    0.0,
    pi / 2.0,
    pi / 4.0,
)
PANDA_INITIAL_FINGER_POSITIONS = (0.04, 0.04)
PANDA_INITIAL_JOINT_POSITIONS = (
    *PANDA_INITIAL_ARM_POSITIONS,
    *PANDA_INITIAL_FINGER_POSITIONS,
)
PANDA_INITIAL_JOINT_INDICES = (
    *PANDA_ARM_JOINT_INDICES,
    *PANDA_FINGER_JOINT_INDICES,
)


def load_franka(client_id: int) -> int:
    """Load a fixed-base Panda and reset it to the initial joint state."""
    urdf_path = Path(pybullet_data.getDataPath()) / "franka_panda" / "panda.urdf"
    robot_id = pybullet.loadURDF(
        str(urdf_path),
        basePosition=PANDA_BASE_POSITION,
        useFixedBase=True,
        physicsClientId=client_id,
    )
    validate_franka_model(client_id, robot_id)
    for joint_index, joint_position in zip(
        PANDA_INITIAL_JOINT_INDICES,
        PANDA_INITIAL_JOINT_POSITIONS,
        strict=True,
    ):
        pybullet.resetJointState(
            robot_id,
            joint_index,
            targetValue=joint_position,
            targetVelocity=0.0,
            physicsClientId=client_id,
        )
    return robot_id


def validate_franka_model(client_id: int, robot_id: int) -> None:
    """Ensure the loaded URDF matches the joint/link metadata used here."""
    joint_count = pybullet.getNumJoints(robot_id, physicsClientId=client_id)
    if joint_count <= PANDA_END_EFFECTOR_LINK_INDEX:
        raise RuntimeError(f"Panda URDF has only {joint_count} joints")

    arm_info = [
        pybullet.getJointInfo(robot_id, index, physicsClientId=client_id)
        for index in PANDA_ARM_JOINT_INDICES
    ]
    arm_names = tuple(info[1].decode("utf-8") for info in arm_info)
    arm_types = tuple(info[2] for info in arm_info)
    if arm_names != PANDA_ARM_JOINT_NAMES or any(
        joint_type != pybullet.JOINT_REVOLUTE for joint_type in arm_types
    ):
        raise RuntimeError("Loaded Panda arm metadata does not match expected URDF")

    finger_info = [
        pybullet.getJointInfo(robot_id, index, physicsClientId=client_id)
        for index in PANDA_FINGER_JOINT_INDICES
    ]
    finger_names = tuple(info[1].decode("utf-8") for info in finger_info)
    finger_types = tuple(info[2] for info in finger_info)
    if finger_names != PANDA_FINGER_JOINT_NAMES or any(
        joint_type != pybullet.JOINT_PRISMATIC for joint_type in finger_types
    ):
        raise RuntimeError("Loaded Panda finger metadata does not match expected URDF")

    end_effector_info = pybullet.getJointInfo(
        robot_id,
        PANDA_END_EFFECTOR_LINK_INDEX,
        physicsClientId=client_id,
    )
    end_effector_name = end_effector_info[12].decode("utf-8")
    if (
        end_effector_name != PANDA_END_EFFECTOR_LINK_NAME
        or end_effector_info[2] != pybullet.JOINT_FIXED
    ):
        raise RuntimeError("Loaded Panda end-effector metadata does not match expected URDF")


def get_arm_joint_positions(client_id: int, robot_id: int) -> tuple[float, ...]:
    """Return the current seven arm-joint positions."""
    return tuple(
        state[0]
        for state in pybullet.getJointStates(
            robot_id,
            PANDA_ARM_JOINT_INDICES,
            physicsClientId=client_id,
        )
    )


def get_end_effector_pose(
    client_id: int,
    robot_id: int,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Return the world pose of the Panda grasp-target link."""
    link_state = pybullet.getLinkState(
        robot_id,
        PANDA_END_EFFECTOR_LINK_INDEX,
        computeForwardKinematics=True,
        physicsClientId=client_id,
    )
    return tuple(link_state[4]), tuple(link_state[5])


def get_initial_joint_positions(client_id: int, robot_id: int) -> tuple[float, ...]:
    """Return the initialized arm and finger joint positions."""
    return tuple(
        pybullet.getJointState(
            robot_id,
            joint_index,
            physicsClientId=client_id,
        )[0]
        for joint_index in PANDA_INITIAL_JOINT_INDICES
    )
