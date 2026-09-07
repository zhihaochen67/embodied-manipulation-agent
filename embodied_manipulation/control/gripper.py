"""Deterministic position control for the Panda parallel gripper."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from time import sleep

import pybullet

from embodied_manipulation.simulation.robot import (
    PANDA_FINGER_JOINT_INDICES,
    PANDA_FINGER_JOINT_NAMES,
    validate_franka_model,
)


@dataclass(frozen=True, slots=True)
class GripperResult:
    """Measured outcome of one gripper command.

    Opening widths are the sum of the two symmetric finger-joint positions.
    ``target_finger_positions`` and ``final_finger_positions`` are the two
    individual prismatic-joint positions.
    """

    success: bool
    steps: int
    target_opening_width: float
    final_opening_width: float
    target_finger_positions: tuple[float, float]
    final_finger_positions: tuple[float, float]
    max_finger_position_error: float
    reached_target: bool
    stalled: bool
    failure_reason: str | None = None


class PandaGripper:
    """Control the two loaded Panda finger joints symmetrically."""

    def __init__(self, client_id: int, robot_id: int) -> None:
        validate_franka_model(client_id, robot_id)
        self.client_id = client_id
        self.robot_id = robot_id

        joint_info = tuple(
            pybullet.getJointInfo(
                robot_id,
                joint_index,
                physicsClientId=client_id,
            )
            for joint_index in PANDA_FINGER_JOINT_INDICES
        )
        names = tuple(info[1].decode("utf-8") for info in joint_info)
        if names != PANDA_FINGER_JOINT_NAMES:
            raise RuntimeError("Loaded Panda finger names do not match metadata")

        self.finger_lower_limits = tuple(float(info[8]) for info in joint_info)
        self.finger_upper_limits = tuple(float(info[9]) for info in joint_info)
        self.finger_forces = tuple(float(info[10]) for info in joint_info)
        self.finger_max_velocities = tuple(float(info[11]) for info in joint_info)

        # A total opening width is split equally between the two fingers.
        self.minimum_opening_width = 2.0 * max(self.finger_lower_limits)
        self.maximum_opening_width = 2.0 * min(self.finger_upper_limits)
        if (
            self.minimum_opening_width < 0.0
            or self.maximum_opening_width <= self.minimum_opening_width
            or any(force <= 0.0 for force in self.finger_forces)
        ):
            raise RuntimeError("Loaded Panda finger limits or efforts are invalid")

        self._target_opening_width = self.opening_width()

    def finger_joint_positions(self) -> tuple[float, float]:
        """Return the two individual finger-joint positions in metres."""
        states = pybullet.getJointStates(
            self.robot_id,
            PANDA_FINGER_JOINT_INDICES,
            physicsClientId=self.client_id,
        )
        return float(states[0][0]), float(states[1][0])

    def opening_width(self) -> float:
        """Return total gripper opening: finger joint 1 plus joint 2."""
        return sum(self.finger_joint_positions())

    def open(
        self,
        opening_width: float | None = None,
        *,
        tolerance: float = 5e-4,
        max_steps: int = 240,
        step_delay: float = 0.0,
    ) -> GripperResult:
        """Open to a total width, defaulting to the loaded-model maximum."""
        target = self.maximum_opening_width if opening_width is None else opening_width
        return self._move(
            target,
            tolerance=tolerance,
            max_steps=max_steps,
            step_delay=step_delay,
            allow_stall=False,
        )

    def close(
        self,
        opening_width: float | None = None,
        *,
        tolerance: float = 5e-4,
        max_steps: int = 240,
        step_delay: float = 0.0,
    ) -> GripperResult:
        """Close toward a total width, defaulting to fully closed.

        Contact with an object can prevent the position target from being
        reached. A stable, low-velocity contact stall is therefore a normal
        successful completion mode and the motor target remains active.
        """
        target = self.minimum_opening_width if opening_width is None else opening_width
        return self._move(
            target,
            tolerance=tolerance,
            max_steps=max_steps,
            step_delay=step_delay,
            allow_stall=True,
        )

    def hold(self, *, steps: int = 1, step_delay: float = 0.0) -> GripperResult:
        """Keep the last open/close motor target active for a fixed duration."""
        _positive_integer(steps, "steps")
        _nonnegative_finite(step_delay, "step_delay")
        targets = self._symmetric_targets(self._target_opening_width)
        for _ in range(steps):
            self._command(targets)
            pybullet.stepSimulation(physicsClientId=self.client_id)
            if step_delay > 0.0:
                sleep(step_delay)
        return self._result(
            success=True,
            steps=steps,
            target_opening_width=self._target_opening_width,
            targets=targets,
            tolerance=0.0,
            reached_target=False,
            stalled=False,
            failure_reason=None,
        )

    def _move(
        self,
        opening_width: float,
        *,
        tolerance: float,
        max_steps: int,
        step_delay: float,
        allow_stall: bool,
    ) -> GripperResult:
        target_width = _finite(opening_width, "opening_width")
        if not self.minimum_opening_width <= target_width <= self.maximum_opening_width:
            raise ValueError(
                "opening_width must be within loaded Panda limits "
                f"[{self.minimum_opening_width:.6f}, "
                f"{self.maximum_opening_width:.6f}] m"
            )
        _positive_finite(tolerance, "tolerance")
        _positive_integer(max_steps, "max_steps")
        _nonnegative_finite(step_delay, "step_delay")

        targets = self._symmetric_targets(target_width)
        self._target_opening_width = target_width
        previous_width = self.opening_width()
        stalled_steps = 0
        for step in range(1, max_steps + 1):
            self._command(targets)
            pybullet.stepSimulation(physicsClientId=self.client_id)
            if step_delay > 0.0:
                sleep(step_delay)

            positions = self.finger_joint_positions()
            position_error = max(
                abs(actual - target)
                for actual, target in zip(positions, targets, strict=True)
            )
            if position_error <= tolerance:
                return self._result(
                    success=True,
                    steps=step,
                    target_opening_width=target_width,
                    targets=targets,
                    tolerance=tolerance,
                    reached_target=True,
                    stalled=False,
                    failure_reason=None,
                )

            states = pybullet.getJointStates(
                self.robot_id,
                PANDA_FINGER_JOINT_INDICES,
                physicsClientId=self.client_id,
            )
            closing_velocity = abs(sum(float(state[1]) for state in states))
            width_movement = abs(sum(positions) - previous_width)
            # A grasped object can translate slightly between the fingers while
            # their total separation has converged. Detect closure convergence
            # from total width, not from either finger in isolation.
            if (
                allow_stall
                and width_movement <= 2e-6
                and closing_velocity <= 5e-4
            ):
                stalled_steps += 1
            else:
                stalled_steps = 0
            if allow_stall and step >= 20 and stalled_steps >= 12:
                return self._result(
                    success=True,
                    steps=step,
                    target_opening_width=target_width,
                    targets=targets,
                    tolerance=tolerance,
                    reached_target=False,
                    stalled=True,
                    failure_reason=None,
                )
            previous_width = sum(positions)

        return self._result(
            success=False,
            steps=max_steps,
            target_opening_width=target_width,
            targets=targets,
            tolerance=tolerance,
            reached_target=False,
            stalled=False,
            failure_reason=f"Timed out after {max_steps} simulation steps",
        )

    def _symmetric_targets(self, opening_width: float) -> tuple[float, float]:
        finger_target = opening_width / 2.0
        targets = (finger_target, finger_target)
        for target, lower, upper in zip(
            targets,
            self.finger_lower_limits,
            self.finger_upper_limits,
            strict=True,
        ):
            if not lower <= target <= upper:
                raise ValueError("symmetric target exceeds an individual finger limit")
        return targets

    def _command(self, targets: Sequence[float]) -> None:
        pybullet.setJointMotorControlArray(
            self.robot_id,
            PANDA_FINGER_JOINT_INDICES,
            pybullet.POSITION_CONTROL,
            targetPositions=targets,
            forces=self.finger_forces,
            physicsClientId=self.client_id,
        )

    def _result(
        self,
        *,
        success: bool,
        steps: int,
        target_opening_width: float,
        targets: tuple[float, float],
        tolerance: float,
        reached_target: bool,
        stalled: bool,
        failure_reason: str | None,
    ) -> GripperResult:
        positions = self.finger_joint_positions()
        error = max(
            abs(actual - target)
            for actual, target in zip(positions, targets, strict=True)
        )
        return GripperResult(
            success=success,
            steps=steps,
            target_opening_width=target_opening_width,
            final_opening_width=sum(positions),
            target_finger_positions=targets,
            final_finger_positions=positions,
            max_finger_position_error=error,
            reached_target=reached_target or error <= tolerance,
            stalled=stalled,
            failure_reason=failure_reason,
        )


def _finite(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_finite(value: float, name: str) -> None:
    if _finite(value, name) <= 0.0:
        raise ValueError(f"{name} must be positive and finite")


def _nonnegative_finite(value: float, name: str) -> None:
    if _finite(value, name) < 0.0:
        raise ValueError(f"{name} must be nonnegative and finite")


def _positive_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
