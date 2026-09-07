"""Small shared representations for oracle and image-based perception."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from embodied_manipulation.language import ObjectRef


class PerceptionError(RuntimeError):
    """Base error for a perception or semantic-grounding failure."""


class ObjectNotFoundError(PerceptionError):
    """Raised when no valid detection matches a semantic query."""


class AmbiguousDetectionError(PerceptionError):
    """Raised when more than one valid detection matches a query."""


class UnsupportedObjectError(PerceptionError):
    """Raised when a query is outside the supported Phase 7 vocabulary."""


@dataclass(frozen=True, slots=True)
class LocalizationResult:
    """A geometric object localization shared by both perception backends.

    ``world_position`` always names the reference described by ``reference``.
    Image-specific fields are absent for oracle results.
    """

    object_ref: ObjectRef
    world_position: tuple[float, float, float]
    reference: str
    source: str
    pixel_centroid: tuple[float, float] | None = None
    bounding_box: tuple[int, int, int, int] | None = None
    mask_area: int | None = None
    metric_depth: float | None = None
    camera_position: tuple[float, float, float] | None = None


class Perception(Protocol):
    """The intentionally small interface shared by Phase 7 backends."""

    def locate(self, object_ref: ObjectRef) -> LocalizationResult:
        """Locate one semantic object or raise :class:`PerceptionError`."""
