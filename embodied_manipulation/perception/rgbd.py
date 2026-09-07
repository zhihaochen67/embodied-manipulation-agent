"""Deterministic RGB-D localization for the current colored primitives."""

from __future__ import annotations

from dataclasses import dataclass
from math import radians, tan
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from embodied_manipulation.language import ObjectRef
from embodied_manipulation.simulation.objects import (
    CUBE_HALF_EXTENT,
    TABLE_SURFACE_Z,
    TRAY_FLOOR_THICKNESS,
)

from .base import (
    AmbiguousDetectionError,
    LocalizationResult,
    ObjectNotFoundError,
    PerceptionError,
    UnsupportedObjectError,
)

if TYPE_CHECKING:
    from embodied_manipulation.simulation.camera import FixedCamera


# OpenCV hue is in [0, 179]. The two red intervals intentionally straddle the
# hue wrap. Saturation excludes the brown table and low-saturation robot.
HSV_THRESHOLDS: dict[str, tuple[tuple[int, int], ...]] = {
    "red": ((0, 6), (174, 179)),
    "blue": ((104, 128),),
    "yellow": ((18, 36),),
    "green": ((45, 82),),
}
MIN_SATURATION = 110
MIN_VALUE = 35
MIN_COMPONENT_AREA = 20


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    """Pinhole intrinsics matching ``computeProjectionMatrixFOV``."""

    width: int
    height: int
    vertical_fov_degrees: float
    aspect_ratio: float
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_camera(cls, camera: FixedCamera) -> CameraIntrinsics:
        if camera.width <= 0 or camera.height <= 0:
            raise ValueError("Camera resolution must be positive")
        if not 0.0 < camera.field_of_view < 180.0:
            raise ValueError("Camera vertical field of view must be in (0, 180)")

        aspect = camera.width / camera.height
        tan_half_y = tan(radians(camera.field_of_view) / 2.0)
        fy = camera.height / (2.0 * tan_half_y)
        tan_half_x = aspect * tan_half_y
        fx = camera.width / (2.0 * tan_half_x)
        # Pixel coordinates identify pixel centers. With OpenGL's symmetric
        # viewport, the principal point lies halfway between the middle pixels.
        return cls(
            width=camera.width,
            height=camera.height,
            vertical_fov_degrees=float(camera.field_of_view),
            aspect_ratio=aspect,
            fx=fx,
            fy=fy,
            cx=(camera.width - 1.0) / 2.0,
            cy=(camera.height - 1.0) / 2.0,
        )


@dataclass(frozen=True, slots=True)
class RGBDObservation:
    """One explicit RGB/depth-buffer observation from the fixed camera.

    ``rgb`` has shape ``(height, width, 3)``, RGB channel order, and uint8
    values in [0, 255]. ``depth_buffer`` has shape ``(height, width)`` and
    contains OpenGL's nonlinear values, not metric depth. No renderer object-ID
    image is retained.
    """

    rgb: np.ndarray
    depth_buffer: np.ndarray
    view_matrix: tuple[float, ...]
    projection_matrix: tuple[float, ...]
    intrinsics: CameraIntrinsics
    near_plane: float
    far_plane: float

    @classmethod
    def capture(
        cls,
        camera: FixedCamera,
        client_id: int,
    ) -> RGBDObservation:
        """Render through ``FixedCamera`` without duplicating its settings."""
        rendered = camera.render(client_id)
        return cls.from_render(
            rendered,
            view_matrix=camera.view_matrix(),
            projection_matrix=camera.projection_matrix(),
            intrinsics=CameraIntrinsics.from_camera(camera),
            near_plane=camera.near_plane,
            far_plane=camera.far_plane,
        )

    @classmethod
    def from_render(
        cls,
        rendered: tuple[int, int, Any, Any, Any],
        *,
        view_matrix: tuple[float, ...],
        projection_matrix: tuple[float, ...],
        intrinsics: CameraIntrinsics,
        near_plane: float,
        far_plane: float,
    ) -> RGBDObservation:
        """Normalize PyBullet's five-item camera result; ignore item five."""
        width, height, rgba, raw_depth, _ = rendered
        if (width, height) != (intrinsics.width, intrinsics.height):
            raise PerceptionError("Rendered resolution differs from intrinsics")

        rgba_array = np.asarray(rgba, dtype=np.uint8)
        depth_array = np.asarray(raw_depth, dtype=np.float64)
        if rgba_array.size != width * height * 4:
            raise PerceptionError("Rendered RGBA image has an invalid shape")
        if depth_array.size != width * height:
            raise PerceptionError("Rendered depth image has an invalid shape")
        rgb = np.ascontiguousarray(
            rgba_array.reshape(height, width, 4)[..., :3]
        )
        depth_buffer = np.ascontiguousarray(
            depth_array.reshape(height, width)
        )
        return cls(
            rgb=rgb,
            depth_buffer=depth_buffer,
            view_matrix=tuple(float(value) for value in view_matrix),
            projection_matrix=tuple(
                float(value) for value in projection_matrix
            ),
            intrinsics=intrinsics,
            near_plane=float(near_plane),
            far_plane=float(far_plane),
        )


def depth_buffer_to_metric(
    depth_buffer: np.ndarray | float,
    near_plane: float,
    far_plane: float,
) -> np.ndarray:
    """Convert OpenGL depth-buffer values to positive camera-axis distance.

    For depth ``d`` in [0, 1], OpenGL perspective inversion gives
    ``z = far * near / (far - (far - near) * d)``.
    """
    if not 0.0 < near_plane < far_plane:
        raise ValueError("Expected 0 < near_plane < far_plane")
    values = np.asarray(depth_buffer, dtype=np.float64)
    denominator = far_plane - (far_plane - near_plane) * values
    with np.errstate(divide="ignore", invalid="ignore"):
        return (far_plane * near_plane) / denominator


def segment_color(rgb: np.ndarray, color: str) -> np.ndarray:
    """Return a deterministic boolean mask for one supported semantic color."""
    if color not in HSV_THRESHOLDS:
        raise UnsupportedObjectError(f"Unsupported color: {color}")
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("RGB image must have shape (height, width, 3)")
    if image.dtype != np.uint8:
        raise ValueError("RGB image must use uint8 values in [0, 255]")

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    combined = np.zeros(image.shape[:2], dtype=bool)
    for hue_low, hue_high in HSV_THRESHOLDS[color]:
        combined |= (
            (hsv[..., 0] >= hue_low)
            & (hsv[..., 0] <= hue_high)
            & (hsv[..., 1] >= MIN_SATURATION)
            & (hsv[..., 2] >= MIN_VALUE)
        )
    return combined


def back_project_camera(
    u: np.ndarray | float,
    v: np.ndarray | float,
    metric_depth: np.ndarray | float,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    """Back-project pixels into OpenGL camera coordinates.

    Image ``u`` points right and ``v`` points down. OpenGL camera ``x`` points
    right, ``y`` points up, and points in front of the camera have negative
    ``z``. ``metric_depth`` is the positive magnitude ``-z``.
    """
    u_array, v_array, depth_array = np.broadcast_arrays(
        np.asarray(u, dtype=np.float64),
        np.asarray(v, dtype=np.float64),
        np.asarray(metric_depth, dtype=np.float64),
    )
    x = (u_array - intrinsics.cx) * depth_array / intrinsics.fx
    y = -(v_array - intrinsics.cy) * depth_array / intrinsics.fy
    z = -depth_array
    return np.stack((x, y, z), axis=-1)


def camera_to_world(
    camera_points: np.ndarray,
    view_matrix: tuple[float, ...],
) -> np.ndarray:
    """Transform OpenGL camera points into the PyBullet world frame.

    PyBullet returns the world-to-camera view matrix as a flat column-major
    OpenGL array, hence ``order='F'`` before homogeneous inversion.
    """
    points = np.asarray(camera_points, dtype=np.float64)
    if points.shape[-1] != 3:
        raise ValueError("Camera points must end with three coordinates")
    view = np.asarray(view_matrix, dtype=np.float64)
    if view.size != 16:
        raise ValueError("View matrix must contain 16 values")
    view = view.reshape(4, 4, order="F")
    inverse_view = np.linalg.inv(view)
    flattened = points.reshape(-1, 3)
    homogeneous = np.column_stack(
        (flattened, np.ones(len(flattened), dtype=np.float64))
    )
    world_h = (inverse_view @ homogeneous.T).T
    world = world_h[:, :3] / world_h[:, 3, np.newaxis]
    return world.reshape(points.shape)


def back_project_world(
    u: np.ndarray | float,
    v: np.ndarray | float,
    metric_depth: np.ndarray | float,
    intrinsics: CameraIntrinsics,
    view_matrix: tuple[float, ...],
) -> np.ndarray:
    """Back-project image pixels and metric depth directly to world points."""
    return camera_to_world(
        back_project_camera(u, v, metric_depth, intrinsics),
        view_matrix,
    )


@dataclass(frozen=True, slots=True)
class _Component:
    centroid: tuple[float, float]
    bounding_box: tuple[int, int, int, int]
    area: int
    metric_depth: float
    camera_position: tuple[float, float, float]
    world_points: np.ndarray

    @property
    def xy_spans(self) -> tuple[float, float]:
        spans = np.ptp(self.world_points[:, :2], axis=0)
        return float(spans[0]), float(spans[1])


class VisionPerception:
    """Locate colored cubes and trays using only rendered RGB and depth.

    Semantic object IDs are deliberately ignored for detection. Current cube
    and tray type grounding uses the large gap between their known primitive
    footprints. The returned cube position is its center; the returned tray
    position is the center of its floor body.
    """

    def __init__(
        self,
        observation: RGBDObservation,
        *,
        cube_size: float = 2.0 * CUBE_HALF_EXTENT,
        tray_floor_thickness: float = TRAY_FLOOR_THICKNESS,
    ) -> None:
        if cube_size <= 0.0 or tray_floor_thickness <= 0.0:
            raise ValueError("Known primitive dimensions must be positive")
        self._observation = observation
        self._cube_size = float(cube_size)
        self._tray_floor_thickness = float(tray_floor_thickness)

    def locate(self, object_ref: ObjectRef) -> LocalizationResult:
        if object_ref.object_type not in {"cube", "tray"}:
            raise UnsupportedObjectError(
                f"Unsupported object type: {object_ref.object_type}"
            )
        mask = segment_color(self._observation.rgb, object_ref.color)
        components = self._components(mask)
        matches = [
            component
            for component in components
            if self._matches_type(component, object_ref.object_type)
        ]
        if not matches:
            raise ObjectNotFoundError(
                f"No visible {object_ref.color} {object_ref.object_type}"
            )
        if len(matches) > 1:
            raise AmbiguousDetectionError(
                f"Multiple visible {object_ref.color} "
                f"{object_ref.object_type}s"
            )

        component = matches[0]
        minimum = np.min(component.world_points[:, :2], axis=0)
        maximum = np.max(component.world_points[:, :2], axis=0)
        center_xy = (minimum + maximum) / 2.0
        if object_ref.object_type == "cube":
            center_z = TABLE_SURFACE_Z + self._cube_size / 2.0
            reference = "object_center"
        else:
            center_z = TABLE_SURFACE_Z + self._tray_floor_thickness / 2.0
            reference = "tray_floor_body_center"

        return LocalizationResult(
            object_ref=object_ref,
            world_position=(
                float(center_xy[0]),
                float(center_xy[1]),
                center_z,
            ),
            reference=reference,
            source="vision",
            pixel_centroid=component.centroid,
            bounding_box=component.bounding_box,
            mask_area=component.area,
            metric_depth=component.metric_depth,
            camera_position=component.camera_position,
        )

    def _components(self, mask: np.ndarray) -> list[_Component]:
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8),
            connectivity=8,
        )
        metric_image = depth_buffer_to_metric(
            self._observation.depth_buffer,
            self._observation.near_plane,
            self._observation.far_plane,
        )
        components: list[_Component] = []
        saw_component_without_valid_depth = False
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < MIN_COMPONENT_AREA:
                continue
            rows, columns = np.nonzero(labels == label)
            depths = metric_image[rows, columns]
            valid = (
                np.isfinite(depths)
                & (depths >= self._observation.near_plane)
                & (depths <= self._observation.far_plane)
            )
            if not np.any(valid):
                saw_component_without_valid_depth = True
                continue
            rows = rows[valid]
            columns = columns[valid]
            depths = depths[valid]
            world_points = back_project_world(
                columns,
                rows,
                depths,
                self._observation.intrinsics,
                self._observation.view_matrix,
            )
            if not np.all(np.isfinite(world_points)):
                continue

            centroid = (
                float(centroids[label, 0]),
                float(centroids[label, 1]),
            )
            median_depth = float(np.median(depths))
            representative_camera = back_project_camera(
                centroid[0],
                centroid[1],
                median_depth,
                self._observation.intrinsics,
            )
            left = int(stats[label, cv2.CC_STAT_LEFT])
            top = int(stats[label, cv2.CC_STAT_TOP])
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            components.append(
                _Component(
                    centroid=centroid,
                    bounding_box=(
                        left,
                        top,
                        left + width - 1,
                        top + height - 1,
                    ),
                    area=area,
                    metric_depth=median_depth,
                    camera_position=tuple(
                        float(value) for value in representative_camera
                    ),
                    world_points=world_points,
                )
            )
        if not components and saw_component_without_valid_depth:
            raise PerceptionError(
                "Visible color region has no finite metric depth within "
                "the camera clipping planes"
            )
        return components

    @staticmethod
    def _matches_type(component: _Component, object_type: str) -> bool:
        span_x, span_y = component.xy_spans
        smaller = min(span_x, span_y)
        larger = max(span_x, span_y)
        if object_type == "cube":
            return 0.015 <= smaller and larger <= 0.09
        # Generated trays have a roughly 0.16 x 0.18 m outer footprint. The
        # upper bound prevents a same-hue table-sized region becoming a tray.
        return smaller >= 0.08 and larger <= 0.30
