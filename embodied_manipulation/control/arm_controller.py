"""Continuous joint-space execution for Panda end-effector reaches."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import dist, isfinite
from time import sleep

import pybullet

from embodied_manipulation.simulation.robot import (
    PANDA_ARM_JOINT_INDICES,
    get_arm_joint_positions,
    get_end_effector_pose,
    validate_franka_model,
)

from .ik import IKError, solve_inverse_kinematics

DEFAULT_CUBE_CLEARANCE = 0.25


@dataclass(frozen=True, slots=True)
class ReachResult:
    """Outcome and final measured state of one reach command."""

    success: bool
    steps: int
    target_position: tuple[float, float, float]
    final_position: tuple[float, float, float]
    final_orientation: tuple[float, float, float, float]
    position_error: float
    target_joint_positions: tuple[float, ...] | None
    final_joint_positions: tuple[float, ...]
    failure_reason: str | None = None


class ArmController:
    """Move the Panda arm with bounded POSITION_CONTROL setpoint updates."""

    def __init__(self, client_id: int, robot_id: int) -> None:
        validate_franka_model(client_id, robot_id)
        self.client_id = client_id
        self.robot_id = robot_id
        self._joint_forces = tuple(
            float(
                pybullet.getJointInfo(
                    robot_id,
                    joint_index,
                    physicsClientId=client_id,
                )[10]
            )
            for joint_index in PANDA_ARM_JOINT_INDICES
        )

    def end_effector_pose(
        self,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        return get_end_effector_pose(self.client_id, self.robot_id)

    def move_end_effector(
        self,
        target_position: Sequence[float],
        target_orientation: Sequence[float] | None = None,
        *,
        max_joint_increment: float = 0.025,
        position_tolerance: float = 0.01,
        joint_tolerance: float = 0.02,
        max_steps: int = 480,
        step_delay: float = 0.0,
        before_step: Callable[[], None] | None = None,
        after_step: Callable[[], None] | None = None,
    ) -> ReachResult:
        """Reach a pose through bounded joint targets and simulation steps."""
        target = _position(target_position)
        _positive_finite(max_joint_increment, "max_joint_increment")
        _positive_finite(position_tolerance, "position_tolerance")
        _positive_finite(joint_tolerance, "joint_tolerance")
        if not isfinite(step_delay) or step_delay < 0.0:
            raise ValueError("step_delay must be nonnegative and finite")
        if not isinstance(max_steps, int) or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        if before_step is not None and not callable(before_step):
            raise ValueError("before_step must be callable")
        if after_step is not None and not callable(after_step):
            raise ValueError("after_step must be callable")

        try:
            joint_targets = solve_inverse_kinematics(
                self.client_id,
                self.robot_id,
                target,
                target_orientation,
            )
        except IKError as error:
            return self._result(
                success=False,
                steps=0,
                target=target,
                joint_targets=None,
                failure_reason=str(error),
            )

        commanded = list(get_arm_joint_positions(self.client_id, self.robot_id))
        for step in range(1, max_steps + 1):
            for index, target_value in enumerate(joint_targets):
                delta = target_value - commanded[index]
                delta = min(max(delta, -max_joint_increment), max_joint_increment)
                commanded[index] += delta

            pybullet.setJointMotorControlArray(
                self.robot_id,
                PANDA_ARM_JOINT_INDICES,
                pybullet.POSITION_CONTROL,
                targetPositions=commanded,
                forces=self._joint_forces,
                physicsClientId=self.client_id,
            )
            if before_step is not None:
                before_step()
            pybullet.stepSimulation(physicsClientId=self.client_id)
            if after_step is not None:
                after_step()
            if step_delay > 0.0:
                sleep(step_delay)

            final_position, _ = self.end_effector_pose()
            position_error = dist(target, final_position)
            actual_joints = get_arm_joint_positions(self.client_id, self.robot_id)
            joint_error = max(
                abs(actual - desired)
                for actual, desired in zip(actual_joints, joint_targets, strict=True)
            )
            if position_error <= position_tolerance and joint_error <= joint_tolerance:
                return self._result(
                    success=True,
                    steps=step,
                    target=target,
                    joint_targets=joint_targets,
                    failure_reason=None,
                )

        return self._result(
            success=False,
            steps=max_steps,
            target=target,
            joint_targets=joint_targets,
            failure_reason=f"Timed out after {max_steps} simulation steps",
        )

    def _result(
        self,
        *,
        success: bool,
        steps: int,
        target: tuple[float, float, float],
        joint_targets: tuple[float, ...] | None,
        failure_reason: str | None,
    ) -> ReachResult:
        final_position, final_orientation = self.end_effector_pose()
        return ReachResult(
            success=success,
            steps=steps,
            target_position=target,
            final_position=final_position,
            final_orientation=final_orientation,
            position_error=dist(target, final_position),
            target_joint_positions=joint_targets,
            final_joint_positions=get_arm_joint_positions(
                self.client_id,
                self.robot_id,
            ),
            failure_reason=failure_reason,
        )


def target_above_cube(
    cube_position: Sequence[float],
    vertical_offset: float = DEFAULT_CUBE_CLEARANCE,
) -> tuple[float, float, float]:
    """Return a point safely above a cube without commanding contact."""
    cube = _position(cube_position)
    _positive_finite(vertical_offset, "vertical_offset")
    return cube[0], cube[1], cube[2] + vertical_offset


def _position(values: Sequence[float]) -> tuple[float, float, float]:
    try:
        position = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ValueError("target position must contain three numeric values") from error
    if len(position) != 3:
        raise ValueError("target position must contain exactly three values")
    if not all(isfinite(value) for value in position):
        raise ValueError("target position must contain only finite values")
    return position


def _positive_finite(value: float, name: str) -> None:
    if not isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
