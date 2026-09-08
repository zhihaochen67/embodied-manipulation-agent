"""Deterministic visual stage verification without simulator object poses."""

from __future__ import annotations

from dataclasses import dataclass
from math import dist, isfinite

from embodied_manipulation.perception import LocalizationResult
from embodied_manipulation.simulation.objects import (
    CUBE_HALF_EXTENT,
    TABLE_SURFACE_Z,
    TRAY_FLOOR_THICKNESS,
    TRAY_INNER_HALF_EXTENTS,
)

Position = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class VerificationConfig:
    """Known geometry and deterministic Phase 9 decision thresholds."""

    cube_half_extent: float = CUBE_HALF_EXTENT
    tray_inner_half_extents: tuple[float, float] = TRAY_INNER_HALF_EXTENTS
    tray_floor_thickness: float = TRAY_FLOOR_THICKNESS
    minimum_lift_clearance: float = 0.08
    maximum_cube_to_ee_distance: float = 0.08
    minimum_visual_displacement: float = 0.08
    horizontal_containment_tolerance: float = 0.006
    vertical_floor_tolerance: float = 0.012
    minimum_released_ee_distance: float = 0.10

    def __post_init__(self) -> None:
        positive = (
            (self.cube_half_extent, "cube_half_extent"),
            (self.tray_floor_thickness, "tray_floor_thickness"),
            (self.minimum_lift_clearance, "minimum_lift_clearance"),
            (
                self.maximum_cube_to_ee_distance,
                "maximum_cube_to_ee_distance",
            ),
            (
                self.minimum_visual_displacement,
                "minimum_visual_displacement",
            ),
            (self.vertical_floor_tolerance, "vertical_floor_tolerance"),
            (
                self.minimum_released_ee_distance,
                "minimum_released_ee_distance",
            ),
        )
        for value, name in positive:
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if (
            len(self.tray_inner_half_extents) != 2
            or not all(
                isfinite(value) and value > 0.0
                for value in self.tray_inner_half_extents
            )
        ):
            raise ValueError(
                "tray_inner_half_extents must contain two positive finite values"
            )
        if (
            not isfinite(self.horizontal_containment_tolerance)
            or self.horizontal_containment_tolerance < 0.0
        ):
            raise ValueError(
                "horizontal_containment_tolerance must be nonnegative and finite"
            )


@dataclass(frozen=True, slots=True)
class GraspVerificationResult:
    """Fresh visual evidence for a lifted, held cube."""

    verified: bool
    cube_detected: bool
    estimated_cube_position: Position | None
    end_effector_position: Position
    cube_height_above_table: float | None
    cube_to_ee_distance: float | None
    visual_displacement_from_initial: float | None
    bilateral_contact: bool | None
    reason: str


@dataclass(frozen=True, slots=True)
class PlacementVerificationResult:
    """Fresh visual evidence for containment and release in the target tray."""

    verified: bool
    cube_detected: bool
    tray_detected: bool
    estimated_cube_position: Position | None
    estimated_tray_position: Position | None
    end_effector_position: Position
    cube_inside_tray: bool
    cube_near_tray_floor: bool
    cube_released: bool
    cube_to_ee_distance: float | None
    reason: str


def verify_grasp(
    cube_detection: LocalizationResult | None,
    *,
    initial_cube_position: Position,
    end_effector_position: Position,
    bilateral_contact: bool | None = None,
    config: VerificationConfig | None = None,
) -> GraspVerificationResult:
    """Verify a grasp from dynamic RGB-D localization plus proprioception."""
    settings = config or VerificationConfig()
    _validate_position(initial_cube_position, "initial_cube_position")
    _validate_position(end_effector_position, "end_effector_position")
    if cube_detection is None:
        return GraspVerificationResult(
            verified=False,
            cube_detected=False,
            estimated_cube_position=None,
            end_effector_position=end_effector_position,
            cube_height_above_table=None,
            cube_to_ee_distance=None,
            visual_displacement_from_initial=None,
            bilateral_contact=bilateral_contact,
            reason="Requested cube was not detected in the post-grasp RGB-D image",
        )

    cube_position = cube_detection.world_position
    _validate_vision_cube(cube_detection)
    bottom_clearance = (
        cube_position[2] - settings.cube_half_extent - TABLE_SURFACE_Z
    )
    ee_distance = dist(cube_position, end_effector_position)
    displacement = dist(cube_position, initial_cube_position)

    if bottom_clearance < settings.minimum_lift_clearance:
        reason = (
            "Visual cube bottom clearance "
            f"{bottom_clearance:.4f} m is below "
            f"{settings.minimum_lift_clearance:.4f} m"
        )
    elif ee_distance > settings.maximum_cube_to_ee_distance:
        reason = (
            f"Visual cube-to-EE distance {ee_distance:.4f} m exceeds "
            f"{settings.maximum_cube_to_ee_distance:.4f} m"
        )
    elif displacement < settings.minimum_visual_displacement:
        reason = (
            f"Visual cube displacement {displacement:.4f} m is below "
            f"{settings.minimum_visual_displacement:.4f} m"
        )
    elif bilateral_contact is False:
        reason = "Visual lift passed but bilateral finger contact was absent"
    else:
        reason = "Fresh RGB-D shows the cube lifted and close to the end effector"

    verified = (
        bottom_clearance >= settings.minimum_lift_clearance
        and ee_distance <= settings.maximum_cube_to_ee_distance
        and displacement >= settings.minimum_visual_displacement
        and bilateral_contact is not False
    )
    return GraspVerificationResult(
        verified=verified,
        cube_detected=True,
        estimated_cube_position=cube_position,
        end_effector_position=end_effector_position,
        cube_height_above_table=bottom_clearance,
        cube_to_ee_distance=ee_distance,
        visual_displacement_from_initial=displacement,
        bilateral_contact=bilateral_contact,
        reason=reason,
    )


def verify_placement(
    cube_detection: LocalizationResult | None,
    tray_detection: LocalizationResult | None,
    *,
    end_effector_position: Position,
    config: VerificationConfig | None = None,
) -> PlacementVerificationResult:
    """Verify placement using only fresh RGB-D geometry and EE proprioception."""
    settings = config or VerificationConfig()
    _validate_position(end_effector_position, "end_effector_position")
    cube_position = (
        None if cube_detection is None else cube_detection.world_position
    )
    tray_position = (
        None if tray_detection is None else tray_detection.world_position
    )
    if cube_detection is not None:
        _validate_vision_cube(cube_detection)
    if tray_detection is not None:
        _validate_vision_tray(tray_detection)

    inside = False
    near_floor = False
    released = False
    ee_distance: float | None = None
    if cube_position is not None and tray_position is not None:
        allowance = settings.horizontal_containment_tolerance
        inside = all(
            abs(cube_position[index] - tray_position[index])
            + settings.cube_half_extent
            <= settings.tray_inner_half_extents[index] + allowance
            for index in (0, 1)
        )
        tray_floor_top = (
            tray_position[2] + settings.tray_floor_thickness / 2.0
        )
        cube_bottom = cube_position[2] - settings.cube_half_extent
        near_floor = (
            abs(cube_bottom - tray_floor_top)
            <= settings.vertical_floor_tolerance
        )
        ee_distance = dist(cube_position, end_effector_position)
        released = ee_distance >= settings.minimum_released_ee_distance

    if cube_position is None:
        reason = "Requested cube was not detected in the post-placement RGB-D image"
    elif tray_position is None:
        reason = "Target tray was not detected in the post-placement RGB-D image"
    elif not inside:
        reason = "Visual cube footprint is outside the target tray interior"
    elif not near_floor:
        reason = "Visual cube height is inconsistent with the target tray floor"
    elif not released:
        reason = "Visual cube remains too close to the end effector"
    else:
        reason = "Fresh RGB-D shows the released cube inside the target tray"

    return PlacementVerificationResult(
        verified=inside and near_floor and released,
        cube_detected=cube_position is not None,
        tray_detected=tray_position is not None,
        estimated_cube_position=cube_position,
        estimated_tray_position=tray_position,
        end_effector_position=end_effector_position,
        cube_inside_tray=inside,
        cube_near_tray_floor=near_floor,
        cube_released=released,
        cube_to_ee_distance=ee_distance,
        reason=reason,
    )


def _validate_vision_cube(detection: LocalizationResult) -> None:
    if detection.source != "vision" or detection.object_ref.object_type != "cube":
        raise ValueError("cube_detection must be a vision cube localization")
    _validate_position(detection.world_position, "cube_detection.world_position")


def _validate_vision_tray(detection: LocalizationResult) -> None:
    if detection.source != "vision" or detection.object_ref.object_type != "tray":
        raise ValueError("tray_detection must be a vision tray localization")
    _validate_position(detection.world_position, "tray_detection.world_position")


def _validate_position(position: Position, name: str) -> None:
    if len(position) != 3 or not all(isfinite(value) for value in position):
        raise ValueError(f"{name} must contain exactly three finite values")
