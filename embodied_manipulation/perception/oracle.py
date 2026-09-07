"""Ground-truth simulator perception baseline."""

from __future__ import annotations

import pybullet

from embodied_manipulation.language import ObjectRef
from embodied_manipulation.simulation.world import World

from .base import (
    AmbiguousDetectionError,
    LocalizationResult,
    ObjectNotFoundError,
    PerceptionError,
    UnsupportedObjectError,
)


class OraclePerception:
    """Locate scenario objects through logical metadata and simulator poses."""

    def __init__(self, world: World) -> None:
        self._world = world

    def locate(self, object_ref: ObjectRef) -> LocalizationResult:
        if object_ref.object_type not in {"cube", "tray"}:
            raise UnsupportedObjectError(
                f"Unsupported object type: {object_ref.object_type}"
            )

        scene = self._world.scene
        if scene is None:
            raise PerceptionError("World must be reset before perception")
        if scene.scenario is None:
            raise PerceptionError(
                "OraclePerception requires a structured scenario"
            )

        matches = [
            item
            for item in scene.scenario.objects
            if item.object_type == object_ref.object_type
            and item.color == object_ref.color
            and (
                object_ref.object_id is None
                or item.object_id == object_ref.object_id
            )
        ]
        if not matches:
            raise ObjectNotFoundError(
                f"No {object_ref.color} {object_ref.object_type} exists"
            )
        if len(matches) > 1:
            raise AmbiguousDetectionError(
                f"Multiple {object_ref.color} {object_ref.object_type}s exist"
            )

        matched = matches[0]
        body_id = self._world.get_object_body_ids(matched.object_id)[0]
        position, _ = pybullet.getBasePositionAndOrientation(
            body_id,
            physicsClientId=self._world.client_id,
        )
        reference = (
            "object_center"
            if matched.object_type == "cube"
            else "tray_floor_body_center"
        )
        return LocalizationResult(
            object_ref=ObjectRef(
                matched.object_id,
                matched.object_type,
                matched.color,
            ),
            world_position=tuple(float(value) for value in position),
            reference=reference,
            source="oracle",
        )
