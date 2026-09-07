"""Structured task semantics for generated manipulation scenarios."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ObjectRef:
    """Semantic reference to one uniquely identified scenario object."""

    object_id: str
    object_type: str
    color: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.object_id, "object_id"),
            (self.object_type, "object_type"),
            (self.color, "color"),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a nonempty string")

    def to_dict(self) -> dict[str, str]:
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
    target: ObjectRef
    instruction: str

    def __post_init__(self) -> None:
        if not isinstance(self.action, str) or not self.action:
            raise ValueError("action must be a nonempty string")
        if not isinstance(self.instruction, str) or not self.instruction:
            raise ValueError("instruction must be a nonempty string")
        if self.source.object_id == self.target.object_id:
            raise ValueError("source and target must have different object IDs")

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.action,
            "source": self.source.to_dict(),
            "target": self.target.to_dict(),
            "instruction": self.instruction,
        }
