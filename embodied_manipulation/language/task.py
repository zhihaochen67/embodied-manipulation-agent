"""Structured task semantics for manipulation instructions and scenarios."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ObjectRef:
    """Semantic object reference, optionally grounded to a logical object ID."""

    object_id: str | None
    object_type: str
    color: str

    def __post_init__(self) -> None:
        if self.object_id is not None and (
            not isinstance(self.object_id, str) or not self.object_id
        ):
            raise ValueError("object_id must be None or a nonempty string")
        for value, name in (
            (self.object_type, "object_type"),
            (self.color, "color"),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a nonempty string")

    def to_dict(self) -> dict[str, str | None]:
        return {
            "object_id": self.object_id,
            "type": self.object_type,
            "color": self.color,
        }


@dataclass(frozen=True, slots=True)
class Task:
    """Ground-truth structured semantics and generated instruction text."""

    action: str
    source: ObjectRef
    target: ObjectRef | None
    instruction: str

    def __post_init__(self) -> None:
        if not isinstance(self.action, str) or not self.action:
            raise ValueError("action must be a nonempty string")
        if not isinstance(self.instruction, str) or not self.instruction:
            raise ValueError("instruction must be a nonempty string")
        if (
            self.target is not None
            and self.source.object_id is not None
            and self.source.object_id == self.target.object_id
        ):
            raise ValueError("source and target must have different object IDs")

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.action,
            "source": self.source.to_dict(),
            "target": self.target.to_dict() if self.target is not None else None,
            "instruction": self.instruction,
        }
