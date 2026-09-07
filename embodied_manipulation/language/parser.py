"""Deterministic parsing for the explicitly supported instruction grammar."""

import re

from .task import ObjectRef, Task

SUPPORTED_COLORS = ("red", "blue", "yellow", "green")

_COLOR_PATTERN = "|".join(SUPPORTED_COLORS)
_PICK_PATTERN = re.compile(
    rf"(?:pick(?: up)?|grab) the (?P<source_color>{_COLOR_PATTERN}) cube"
)
_PUT_OR_PLACE_PATTERN = re.compile(
    rf"(?:put|place) the (?P<source_color>{_COLOR_PATTERN}) cube "
    rf"(?:in|into) the (?P<target_color>{_COLOR_PATTERN}) tray"
)
_MOVE_PATTERN = re.compile(
    rf"move the (?P<source_color>{_COLOR_PATTERN}) cube "
    rf"to the (?P<target_color>{_COLOR_PATTERN}) tray"
)


class InstructionParseError(ValueError):
    """Raised when an instruction is outside the supported grammar."""


def parse_instruction(text: str) -> Task:
    """Parse one supported instruction into ungrounded task semantics.

    Parsing is case-insensitive and tolerates repeated whitespace and trailing
    sentence punctuation. The original instruction is retained in the Task.
    """
    if not isinstance(text, str):
        raise InstructionParseError("instruction must be a string")

    normalized = _normalize(text)
    if not normalized:
        raise InstructionParseError("instruction must not be empty")

    pick_match = _PICK_PATTERN.fullmatch(normalized)
    if pick_match is not None:
        return Task(
            action="pick",
            source=_object_ref("cube", pick_match["source_color"]),
            target=None,
            instruction=text,
        )

    for pattern in (_PUT_OR_PLACE_PATTERN, _MOVE_PATTERN):
        match = pattern.fullmatch(normalized)
        if match is not None:
            return Task(
                action="pick_and_place",
                source=_object_ref("cube", match["source_color"]),
                target=_object_ref("tray", match["target_color"]),
                instruction=text,
            )

    raise InstructionParseError(
        "unsupported or malformed instruction; expected 'pick/pick up/grab "
        "the <color> cube', 'put/place the <color> cube in/into the <color> "
        "tray', or 'move the <color> cube to the <color> tray' using red, "
        "blue, yellow, or green"
    )


def _normalize(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.strip().casefold())
    return re.sub(r"[.!?]+$", "", normalized).rstrip()


def _object_ref(object_type: str, color: str) -> ObjectRef:
    return ObjectRef(object_id=None, object_type=object_type, color=color)
