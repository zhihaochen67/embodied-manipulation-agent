import pytest

from embodied_manipulation.language import (
    InstructionParseError,
    ObjectRef,
    parse_instruction,
)


def test_parse_basic_pick() -> None:
    task = parse_instruction("Pick up the red cube.")

    assert task.action == "pick"
    assert task.source == ObjectRef(None, "cube", "red")
    assert task.target is None
    assert task.instruction == "Pick up the red cube."


@pytest.mark.parametrize(
    ("instruction", "color"),
    (
        ("Pick the green cube.", "green"),
        ("Grab the blue cube.", "blue"),
    ),
)
def test_parse_pick_variants(instruction: str, color: str) -> None:
    task = parse_instruction(instruction)

    assert task.action == "pick"
    assert task.source == ObjectRef(None, "cube", color)
    assert task.target is None


@pytest.mark.parametrize(
    ("instruction", "source_color", "target_color"),
    (
        ("Put the red cube in the blue tray.", "red", "blue"),
        ("Put the blue cube into the green tray.", "blue", "green"),
        ("Place the green cube in the red tray.", "green", "red"),
        ("Place the yellow cube into the blue tray.", "yellow", "blue"),
        ("Move the blue cube to the red tray.", "blue", "red"),
    ),
)
def test_parse_pick_and_place_variants(
    instruction: str,
    source_color: str,
    target_color: str,
) -> None:
    task = parse_instruction(instruction)

    assert task.action == "pick_and_place"
    assert task.source == ObjectRef(None, "cube", source_color)
    assert task.target == ObjectRef(None, "tray", target_color)


def test_parsing_is_case_insensitive_and_normalizes_surface_whitespace() -> None:
    instruction = "  PUT   the RED cube\tinto the BLUE tray!!!  "

    task = parse_instruction(instruction)

    assert task.source == ObjectRef(None, "cube", "red")
    assert task.target == ObjectRef(None, "tray", "blue")
    assert task.instruction == instruction


@pytest.mark.parametrize("color", ("red", "blue", "yellow", "green"))
def test_all_supported_colors_parse(color: str) -> None:
    task = parse_instruction(f"Grab the {color} cube.")

    assert task.source.color == color


@pytest.mark.parametrize(
    "instruction",
    (
        "",
        "   !!!   ",
        "Move something somewhere.",
        "Put the purple cube in the blue tray.",
        "Put the red sphere in the blue tray.",
        "Put the red cube in the box.",
        "Dance around the red cube.",
        "Grab the red cube and put it in the blue tray.",
        "Put the red cube.",
    ),
)
def test_unsupported_or_malformed_instructions_fail(instruction: str) -> None:
    with pytest.raises(InstructionParseError, match="instruction|unsupported"):
        parse_instruction(instruction)


def test_parser_returns_semantics_without_grounding_information() -> None:
    task = parse_instruction("Put the red cube into the blue tray.")

    assert task.source.object_id is None
    assert task.target is not None
    assert task.target.object_id is None
    assert "body_id" not in task.source.to_dict()
    assert "body_id" not in task.target.to_dict()


def test_pick_task_serializes_without_a_target() -> None:
    task = parse_instruction("Pick the red cube.")

    assert task.to_dict()["target"] is None
