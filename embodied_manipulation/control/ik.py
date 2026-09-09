"""Inverse kinematics for the seven Panda arm joints."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import acos, dist, isfinite, pi, sqrt

import pybullet

from embodied_manipulation.simulation.robot import (
    PANDA_ARM_JOINT_INDICES,
    PANDA_END_EFFECTOR_LINK_INDEX,
)

# PyBullet quaternions use (x, y, z, w). A pi rotation about world X
# makes the grasp-target tool axis point down toward the tabletop.
DEFAULT_END_EFFECTOR_ORIENTATION = (1.0, 0.0, 0.0, 0.0)


class IKError(RuntimeError):
    """Raised when PyBullet does not produce an acceptable IK solution."""


@dataclass(frozen=True, slots=True)
class IKSolution:
    """A broadly valid IK candidate and its measured forward-kinematics pose."""

    joint_positions: tuple[float, ...]
    candidate_position: tuple[float, float, float]
    candidate_orientation: tuple[float, float, float, float]
    position_residual: float
    orientation_residual: float


def solve_inverse_kinematics(
    client_id: int,
    robot_id: int,
    target_position: Sequence[float],
    target_orientation: Sequence[float] | None = None,
    *,
    max_position_error: float = 0.03,
    max_orientation_error: float = 0.15,
) -> tuple[float, ...]:
    """Return finite, limit-clamped targets for only the seven arm joints."""
    return solve_inverse_kinematics_with_diagnostics(
        client_id,
        robot_id,
        target_position,
        target_orientation,
        max_position_error=max_position_error,
        max_orientation_error=max_orientation_error,
    ).joint_positions


def solve_inverse_kinematics_with_diagnostics(
    client_id: int,
    robot_id: int,
    target_position: Sequence[float],
    target_orientation: Sequence[float] | None = None,
    *,
    max_position_error: float = 0.03,
    max_orientation_error: float = 0.15,
) -> IKSolution:
    """Return one broadly valid IK candidate with its FK residuals.

    The broad residual thresholds remain solver-level sanity checks. Callers with
    stricter completion predicates must separately decide whether this single
    candidate is admissible for their execution contract.
    """
    position = _finite_vector(target_position, 3, "target_position")
    orientation = _normalized_quaternion(
        DEFAULT_END_EFFECTOR_ORIENTATION
        if target_orientation is None
        else target_orientation
    )
    _positive_finite(max_position_error, "max_position_error")
    _positive_finite(max_orientation_error, "max_orientation_error")

    movable_joints = _movable_joints_in_ik_order(client_id, robot_id)
    lower_limits: list[float] = []
    upper_limits: list[float] = []
    joint_ranges: list[float] = []
    rest_positions: list[float] = []
    for joint_index in movable_joints:
        info = pybullet.getJointInfo(robot_id, joint_index, physicsClientId=client_id)
        lower, upper = float(info[8]), float(info[9])
        if lower > upper:
            lower, upper = -pi, pi
        lower_limits.append(lower)
        upper_limits.append(upper)
        joint_ranges.append(upper - lower)
        rest_positions.append(
            pybullet.getJointState(
                robot_id,
                joint_index,
                physicsClientId=client_id,
            )[0]
        )

    raw_solution = pybullet.calculateInverseKinematics(
        robot_id,
        PANDA_END_EFFECTOR_LINK_INDEX,
        targetPosition=position,
        targetOrientation=orientation,
        lowerLimits=lower_limits,
        upperLimits=upper_limits,
        jointRanges=joint_ranges,
        restPoses=rest_positions,
        maxNumIterations=200,
        residualThreshold=1e-5,
        physicsClientId=client_id,
    )
    if len(raw_solution) != len(movable_joints):
        raise IKError(
            "Unexpected IK result size: "
            f"expected {len(movable_joints)}, got {len(raw_solution)}"
        )

    solution_by_joint = dict(zip(movable_joints, raw_solution, strict=True))
    arm_targets: list[float] = []
    for joint_index in PANDA_ARM_JOINT_INDICES:
        if joint_index not in solution_by_joint:
            raise IKError(f"IK result omitted arm joint {joint_index}")
        target = float(solution_by_joint[joint_index])
        info = pybullet.getJointInfo(robot_id, joint_index, physicsClientId=client_id)
        lower, upper = float(info[8]), float(info[9])
        target = min(max(target, lower), upper)
        if not isfinite(target):
            raise IKError(f"IK returned a non-finite value for arm joint {joint_index}")
        arm_targets.append(target)

    candidate_position, candidate_orientation = _candidate_pose(
        client_id,
        robot_id,
        tuple(arm_targets),
    )
    position_error = dist(position, candidate_position)
    orientation_error = _quaternion_angle(orientation, candidate_orientation)
    if position_error > max_position_error:
        raise IKError(
            f"IK position residual {position_error:.4f} m exceeds "
            f"{max_position_error:.4f} m"
        )
    if orientation_error > max_orientation_error:
        raise IKError(
            f"IK orientation residual {orientation_error:.4f} rad exceeds "
            f"{max_orientation_error:.4f} rad"
        )
    return IKSolution(
        joint_positions=tuple(arm_targets),
        candidate_position=candidate_position,
        candidate_orientation=candidate_orientation,
        position_residual=position_error,
        orientation_residual=orientation_error,
    )


def _movable_joints_in_ik_order(client_id: int, robot_id: int) -> tuple[int, ...]:
    joints: list[tuple[int, int]] = []
    joint_count = pybullet.getNumJoints(robot_id, physicsClientId=client_id)
    for joint_index in range(joint_count):
        info = pybullet.getJointInfo(robot_id, joint_index, physicsClientId=client_id)
        q_index = int(info[3])
        if q_index >= 0:
            joints.append((q_index, joint_index))
    joints.sort()
    return tuple(joint_index for _, joint_index in joints)


def _candidate_pose(
    client_id: int,
    robot_id: int,
    arm_targets: tuple[float, ...],
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    previous_states = pybullet.getJointStates(
        robot_id,
        PANDA_ARM_JOINT_INDICES,
        physicsClientId=client_id,
    )
    try:
        for joint_index, target in zip(
            PANDA_ARM_JOINT_INDICES,
            arm_targets,
            strict=True,
        ):
            pybullet.resetJointState(
                robot_id,
                joint_index,
                targetValue=target,
                targetVelocity=0.0,
                physicsClientId=client_id,
            )
        state = pybullet.getLinkState(
            robot_id,
            PANDA_END_EFFECTOR_LINK_INDEX,
            computeForwardKinematics=True,
            physicsClientId=client_id,
        )
        return tuple(state[4]), tuple(state[5])
    finally:
        for joint_index, state in zip(
            PANDA_ARM_JOINT_INDICES,
            previous_states,
            strict=True,
        ):
            pybullet.resetJointState(
                robot_id,
                joint_index,
                targetValue=state[0],
                targetVelocity=state[1],
                physicsClientId=client_id,
            )


def _finite_vector(
    values: Sequence[float],
    size: int,
    name: str,
) -> tuple[float, ...]:
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain {size} numeric values") from error
    if len(result) != size:
        raise ValueError(f"{name} must contain exactly {size} values")
    if not all(isfinite(value) for value in result):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _normalized_quaternion(values: Sequence[float]) -> tuple[float, float, float, float]:
    quaternion = _finite_vector(values, 4, "target_orientation")
    norm = sqrt(sum(value * value for value in quaternion))
    if norm <= 1e-12:
        raise ValueError("target_orientation must have nonzero magnitude")
    return tuple(value / norm for value in quaternion)


def _quaternion_angle(first: Sequence[float], second: Sequence[float]) -> float:
    normalized_second = _normalized_quaternion(second)
    dot = abs(sum(a * b for a, b in zip(first, normalized_second, strict=True)))
    return 2.0 * acos(min(1.0, max(-1.0, dot)))


def _positive_finite(value: float, name: str) -> None:
    if not isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
