"""Deterministic oracle and RGB-D perception for colored primitives."""

from .base import (
    AmbiguousDetectionError,
    LocalizationResult,
    ObjectNotFoundError,
    Perception,
    PerceptionError,
    UnsupportedObjectError,
)
from .oracle import OraclePerception
from .rgbd import (
    HSV_THRESHOLDS,
    CameraIntrinsics,
    RGBDObservation,
    VisionPerception,
    back_project_camera,
    back_project_world,
    camera_to_world,
    depth_buffer_to_metric,
    segment_color,
)

__all__ = [
    "HSV_THRESHOLDS",
    "AmbiguousDetectionError",
    "CameraIntrinsics",
    "LocalizationResult",
    "ObjectNotFoundError",
    "OraclePerception",
    "Perception",
    "PerceptionError",
    "RGBDObservation",
    "UnsupportedObjectError",
    "VisionPerception",
    "back_project_camera",
    "back_project_world",
    "camera_to_world",
    "depth_buffer_to_metric",
    "segment_color",
]
