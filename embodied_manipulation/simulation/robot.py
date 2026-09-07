"""Franka Panda loading and deterministic initialization."""

from math import pi
from pathlib import Path

import pybullet
import pybullet_data

PANDA_BASE_POSITION = (0.0, 0.0, 0.0)
PANDA_ARM_JOINT_INDICES = (0, 1, 2, 3, 4, 5, 6)
PANDA_FINGER_JOINT_INDICES = (9, 10)
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
    """Load a fixed-base Panda and reset it to the Phase 1 joint state."""
    urdf_path = Path(pybullet_data.getDataPath()) / "franka_panda" / "panda.urdf"
    robot_id = pybullet.loadURDF(
        str(urdf_path),
        basePosition=PANDA_BASE_POSITION,
        useFixedBase=True,
        physicsClientId=client_id,
    )
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
