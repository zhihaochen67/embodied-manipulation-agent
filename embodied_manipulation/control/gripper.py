"""Deterministic position control for the Panda parallel gripper."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import isfinite
from time import sleep

import pybullet

from embodied_manipulation.simulation.robot import (
    PANDA_FINGER_JOINT_INDICES,
    PANDA_FINGER_JOINT_NAMES,
    validate_franka_model,
)


CLOSE_MINIMUM_COMPLETION_STEPS = 20
CLOSE_CONSECUTIVE_COMPLETION_STEPS = 12


@dataclass(frozen=True, slots=True)
class GripperResult:
    """Measured outcome of one gripper command.

    Opening widths are the sum of the two finger-joint positions.
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
    termination_reason: str | None = None
    target_contact_observed: bool | None = None
    bilateral_target_contact_observed: bool | None = None
    persistent_bilateral_non_target_contact_observed: bool | None = None
    persistent_bilateral_non_target_body_ids: tuple[int, ...] = ()
    timeout_diagnostic: str | None = None


class PandaGripper:
    """Control the two loaded Panda parallel finger joints."""

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

        self._target_finger_positions = self.finger_joint_positions()
        self._target_opening_width = sum(self._target_finger_positions)

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
        step_observer: Callable[[], None] | None = None,
    ) -> GripperResult:
        """Open to a total width, defaulting to the loaded-model maximum."""
        target = self.maximum_opening_width if opening_width is None else opening_width
        return self._move(
            target,
            tolerance=tolerance,
            max_steps=max_steps,
            step_delay=step_delay,
            allow_stall=False,
            target_body_id=None,
            step_observer=step_observer,
        )

    def close(
        self,
        opening_width: float | None = None,
        *,
        target_body_id: int | None = None,
        tolerance: float = 5e-4,
        max_steps: int = 240,
        step_delay: float = 0.0,
        step_observer: Callable[[], None] | None = None,
    ) -> GripperResult:
        """Close toward a total width, defaulting to fully closed.

        Contact with an object can prevent the position target from being
        reached. A stable, low-velocity contact stall is therefore a normal
        successful completion mode. When ``target_body_id`` is supplied,
        sustained contact between that body and both finger links is also a
        normal completion mode. Completion means post-close settling may
        proceed; it does not establish physical grasp success. The motor target
        remains active in every case.
        """
        target = self.minimum_opening_width if opening_width is None else opening_width
        return self._move(
            target,
            tolerance=tolerance,
            max_steps=max_steps,
            step_delay=step_delay,
            allow_stall=True,
            target_body_id=target_body_id,
            step_observer=step_observer,
        )

    def capture_hold(self, *, inward_preload: float = 0.001) -> tuple[float, float]:
        """Hold the measured finger configuration with a small inward preload."""
        _nonnegative_finite(inward_preload, "inward_preload")
        positions = self.finger_joint_positions()
        targets = tuple(
            max(lower, position - inward_preload)
            for position, lower in zip(
                positions,
                self.finger_lower_limits,
                strict=True,
            )
        )
        self._target_finger_positions = targets
        self._target_opening_width = sum(targets)
        self._command(targets)
        return targets

    def maintain(self) -> None:
        """Reissue the active physical finger POSITION_CONTROL command."""
        self._command(self._target_finger_positions)

    def hold(self, *, steps: int = 1, step_delay: float = 0.0) -> GripperResult:
        """Keep the last open/close motor target active for a fixed duration."""
        _positive_integer(steps, "steps")
        _nonnegative_finite(step_delay, "step_delay")
        targets = self._target_finger_positions
        for _ in range(steps):
            self.maintain()
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
            termination_reason="hold_complete",
        )

    def _move(
        self,
        opening_width: float,
        *,
        tolerance: float,
        max_steps: int,
        step_delay: float,
        allow_stall: bool,
        target_body_id: int | None,
        step_observer: Callable[[], None] | None,
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
        _optional_nonnegative_integer(target_body_id, "target_body_id")
        if step_observer is not None and not callable(step_observer):
            raise ValueError("step_observer must be callable")

        targets = self._symmetric_targets(target_width)
        self._target_opening_width = target_width
        self._target_finger_positions = targets
        previous_width = self.opening_width()
        stalled_steps = 0
        bilateral_target_contact_steps = 0
        contact_diagnostics_enabled = allow_stall and target_body_id is not None
        target_contact_observed: bool | None = (
            False if contact_diagnostics_enabled else None
        )
        bilateral_target_contact_observed: bool | None = (
            False if contact_diagnostics_enabled else None
        )
        persistent_bilateral_non_target_contact_observed: bool | None = (
            False if contact_diagnostics_enabled else None
        )
        persistent_bilateral_non_target_body_ids: set[int] = set()
        bilateral_non_target_contact_steps: dict[int, int] = {}
        for step in range(1, max_steps + 1):
            self._command(targets)
            pybullet.stepSimulation(physicsClientId=self.client_id)
            if step_observer is not None:
                step_observer()
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
                    termination_reason="target_reached",
                    target_contact_observed=target_contact_observed,
                    bilateral_target_contact_observed=(
                        bilateral_target_contact_observed
                    ),
                    persistent_bilateral_non_target_contact_observed=(
                        persistent_bilateral_non_target_contact_observed
                    ),
                    persistent_bilateral_non_target_body_ids=tuple(
                        sorted(persistent_bilateral_non_target_body_ids)
                    ),
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
            bilateral_target_contact = False
            if contact_diagnostics_enabled:
                contact_links_by_body = _finger_contact_links_by_body(
                    self.client_id,
                    self.robot_id,
                )
                target_contact_links = contact_links_by_body.get(
                    target_body_id,
                    set(),
                )
                target_contact_observed = (
                    target_contact_observed or bool(target_contact_links)
                )
                bilateral_target_contact = target_contact_links == set(
                    PANDA_FINGER_JOINT_INDICES
                )
                bilateral_target_contact_observed = (
                    bilateral_target_contact_observed
                    or bilateral_target_contact
                )
                non_target_body_ids = {
                    body_id
                    for body_id in contact_links_by_body
                    if body_id not in {self.robot_id, target_body_id}
                }
                for body_id in (
                    bilateral_non_target_contact_steps.keys() | non_target_body_ids
                ):
                    bilateral_non_target_contact = (
                        contact_links_by_body.get(body_id, set())
                        == set(PANDA_FINGER_JOINT_INDICES)
                    )
                    bilateral_non_target_contact_steps[body_id] = (
                        bilateral_non_target_contact_steps.get(body_id, 0) + 1
                        if bilateral_non_target_contact
                        else 0
                    )
                    if (
                        bilateral_non_target_contact_steps[body_id]
                        >= CLOSE_CONSECUTIVE_COMPLETION_STEPS
                    ):
                        persistent_bilateral_non_target_body_ids.add(body_id)
                        persistent_bilateral_non_target_contact_observed = True
            if (
                allow_stall
                and target_width < sum(positions)
                and bilateral_target_contact
            ):
                bilateral_target_contact_steps += 1
            else:
                bilateral_target_contact_steps = 0

            # Completion precedence is deterministic: target position first,
            # stable stall second, and sustained target contact third.
            if (
                allow_stall
                and step >= CLOSE_MINIMUM_COMPLETION_STEPS
                and stalled_steps >= CLOSE_CONSECUTIVE_COMPLETION_STEPS
            ):
                return self._result(
                    success=True,
                    steps=step,
                    target_opening_width=target_width,
                    targets=targets,
                    tolerance=tolerance,
                    reached_target=False,
                    stalled=True,
                    failure_reason=None,
                    termination_reason="stable_stall",
                    target_contact_observed=target_contact_observed,
                    bilateral_target_contact_observed=(
                        bilateral_target_contact_observed
                    ),
                    persistent_bilateral_non_target_contact_observed=(
                        persistent_bilateral_non_target_contact_observed
                    ),
                    persistent_bilateral_non_target_body_ids=tuple(
                        sorted(persistent_bilateral_non_target_body_ids)
                    ),
                )
            if (
                allow_stall
                and step >= CLOSE_MINIMUM_COMPLETION_STEPS
                and bilateral_target_contact_steps
                >= CLOSE_CONSECUTIVE_COMPLETION_STEPS
            ):
                return self._result(
                    success=True,
                    steps=step,
                    target_opening_width=target_width,
                    targets=targets,
                    tolerance=tolerance,
                    reached_target=False,
                    stalled=False,
                    failure_reason=None,
                    termination_reason="bilateral_target_contact",
                    target_contact_observed=target_contact_observed,
                    bilateral_target_contact_observed=(
                        bilateral_target_contact_observed
                    ),
                    persistent_bilateral_non_target_contact_observed=(
                        persistent_bilateral_non_target_contact_observed
                    ),
                    persistent_bilateral_non_target_body_ids=tuple(
                        sorted(persistent_bilateral_non_target_body_ids)
                    ),
                )
            previous_width = sum(positions)

        if not contact_diagnostics_enabled:
            timeout_diagnostic = "generic_timeout"
        elif persistent_bilateral_non_target_body_ids:
            timeout_diagnostic = "persistent_bilateral_non_target_contact"
        elif not target_contact_observed:
            timeout_diagnostic = "no_target_contact"
        elif not bilateral_target_contact_observed:
            timeout_diagnostic = "target_contact_not_bilateral"
        else:
            timeout_diagnostic = "bilateral_target_contact_not_sustained"
        return self._result(
            success=False,
            steps=max_steps,
            target_opening_width=target_width,
            targets=targets,
            tolerance=tolerance,
            reached_target=False,
            stalled=False,
            failure_reason=f"Timed out after {max_steps} simulation steps",
            termination_reason="timeout",
            target_contact_observed=target_contact_observed,
            bilateral_target_contact_observed=bilateral_target_contact_observed,
            persistent_bilateral_non_target_contact_observed=(
                persistent_bilateral_non_target_contact_observed
            ),
            persistent_bilateral_non_target_body_ids=tuple(
                sorted(persistent_bilateral_non_target_body_ids)
            ),
            timeout_diagnostic=timeout_diagnostic,
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
        termination_reason: str,
        target_contact_observed: bool | None = None,
        bilateral_target_contact_observed: bool | None = None,
        persistent_bilateral_non_target_contact_observed: bool | None = None,
        persistent_bilateral_non_target_body_ids: tuple[int, ...] = (),
        timeout_diagnostic: str | None = None,
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
            termination_reason=termination_reason,
            target_contact_observed=target_contact_observed,
            bilateral_target_contact_observed=(
                bilateral_target_contact_observed
            ),
            persistent_bilateral_non_target_contact_observed=(
                persistent_bilateral_non_target_contact_observed
            ),
            persistent_bilateral_non_target_body_ids=(
                persistent_bilateral_non_target_body_ids
            ),
            timeout_diagnostic=timeout_diagnostic,
        )


def _finger_contact_links_by_body(
    client_id: int,
    robot_id: int,
) -> dict[int, set[int]]:
    """Group contacting Panda finger links by the other body's ID."""
    contacts = pybullet.getContactPoints(
        bodyA=robot_id,
        physicsClientId=client_id,
    )
    links_by_body: dict[int, set[int]] = {}
    for contact in contacts:
        finger_link = int(contact[3])
        if finger_link in PANDA_FINGER_JOINT_INDICES:
            links_by_body.setdefault(int(contact[2]), set()).add(finger_link)
    return links_by_body


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


def _optional_nonnegative_integer(value: int | None, name: str) -> None:
    if value is not None and (
        not isinstance(value, int) or isinstance(value, bool) or value < 0
    ):
        raise ValueError(f"{name} must be a nonnegative integer or None")
