import json

import numpy as np
import pybullet

from embodied_manipulation.benchmark.scenarios import (
    INSTRUCTION_TEMPLATES,
    MINIMUM_SOURCE_TARGET_CENTER_DISTANCE,
    REACHABLE_WORKSPACE_BOUNDS,
    CubeSpec,
    Scenario,
    TraySpec,
    generate_scenario,
    validate_scenario,
)
from embodied_manipulation.control.pick_place import oracle_pick_place
from embodied_manipulation.language import ObjectRef, Task, parse_instruction
from embodied_manipulation.simulation import World
from embodied_manipulation.simulation.objects import TABLE_SURFACE_Z


def test_task_contains_structured_semantics() -> None:
    source = ObjectRef("source_cube", "cube", "red")
    target = ObjectRef("target_tray", "tray", "blue")
    task = Task(
        action="pick_and_place",
        source=source,
        target=target,
        instruction="Put the red cube in the blue tray.",
    )

    assert task.action == "pick_and_place"
    assert task.source.object_type == "cube"
    assert task.source.color == "red"
    assert task.target.object_type == "tray"
    assert task.target.color == "blue"
    assert task.to_dict()["source"] == source.to_dict()


def test_same_seed_produces_exactly_the_same_scenario() -> None:
    first = generate_scenario(17, distractor_count=2)
    second = generate_scenario(17, distractor_count=2)

    assert first == second
    assert first.to_dict() == second.to_dict()


def test_different_seeds_produce_diverse_scenarios() -> None:
    scenarios = [generate_scenario(seed) for seed in range(8)]
    configurations = {
        (
            scenario.source.initial_position,
            scenario.target.initial_position,
            scenario.source.color,
            scenario.target.color,
            scenario.task.instruction,
        )
        for scenario in scenarios
    }

    assert len(configurations) == len(scenarios)


def test_generated_geometry_is_valid_by_construction() -> None:
    workspace_min_x, workspace_max_x, workspace_min_y, workspace_max_y = (
        REACHABLE_WORKSPACE_BOUNDS
    )
    for seed in range(10):
        scenario = generate_scenario(seed, distractor_count=seed % 3)
        validate_scenario(scenario)

        assert scenario.source_target_center_distance >= (
            MINIMUM_SOURCE_TARGET_CENTER_DISTANCE
        )
        assert scenario.source_target_clearance > 0.0
        assert scenario.source.size < min(scenario.target.inner_size)
        assert scenario.source.initial_position[2] == (
            TABLE_SURFACE_Z + scenario.source.size / 2.0
        )
        assert scenario.target.initial_position[2] == (
            TABLE_SURFACE_Z + scenario.target.floor_thickness / 2.0
        )
        for item in scenario.objects:
            bounds = item.footprint_bounds
            assert workspace_min_x <= bounds[0] <= bounds[1] <= workspace_max_x
            assert workspace_min_y <= bounds[2] <= bounds[3] <= workspace_max_y


def test_instruction_agrees_with_structured_task() -> None:
    for seed in range(8):
        scenario = generate_scenario(seed)
        task = scenario.task
        expected_instructions = {
            template.format(
                source_color=task.source.color,
                target_color=task.target.color,
            )
            for template in INSTRUCTION_TEMPLATES
        }

        assert task.instruction in expected_instructions
        assert task.source.object_id == scenario.source.object_id
        assert task.target.object_id == scenario.target.object_id


def test_generated_instructions_round_trip_to_matching_semantics() -> None:
    for seed in range(21):
        original = generate_scenario(seed).task
        parsed = parse_instruction(original.instruction)

        assert parsed.action == original.action
        assert parsed.source.object_type == original.source.object_type
        assert parsed.source.color == original.source.color
        assert parsed.target is not None
        assert original.target is not None
        assert parsed.target.object_type == original.target.object_type
        assert parsed.target.color == original.target.color
        assert parsed.source.object_id is None
        assert parsed.target.object_id is None


def test_scenario_serialization_is_deterministic_and_json_compatible() -> None:
    scenario = generate_scenario(5, distractor_count=2)
    first = json.dumps(scenario.to_dict(), sort_keys=True)
    second = json.dumps(scenario.to_dict(), sort_keys=True)

    assert first == second
    assert json.loads(first) == scenario.to_dict()


def test_world_loads_scenario_poses_and_body_mapping() -> None:
    scenario = generate_scenario(2, distractor_count=2)
    with World(gui=False) as world:
        scene = world.reset_from_scenario(scenario)

        assert scene.scenario == scenario
        assert set(scene.object_body_ids) == {
            item.object_id for item in scenario.objects
        }
        assert scene.object_body_ids[scenario.source.object_id] == (scene.cube_id,)
        assert scene.object_body_ids[scenario.target.object_id] == scene.tray.body_ids
        np.testing.assert_allclose(
            world.get_cube_pose()[0],
            scenario.source.initial_position,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            world.get_tray_pose()[0],
            scenario.target.initial_position,
            atol=1e-8,
        )

        for item in scenario.objects:
            body_ids = world.get_object_body_ids(item.object_id)
            assert body_ids
            for body_id in body_ids:
                assert pybullet.getBodyInfo(
                    body_id,
                    physicsClientId=world.client_id,
                )
            if isinstance(item, CubeSpec) and item.role == "distractor":
                position, _ = pybullet.getBasePositionAndOrientation(
                    body_ids[0],
                    physicsClientId=world.client_id,
                )
                np.testing.assert_allclose(
                    position,
                    item.initial_position,
                    atol=1e-8,
                )


def test_generated_scenario_metadata_is_independent_of_body_ids() -> None:
    scenario = generate_scenario(3, distractor_count=1)

    assert isinstance(scenario, Scenario)
    assert isinstance(scenario.source, CubeSpec)
    assert isinstance(scenario.target, TraySpec)
    assert "body_id" not in scenario.source.to_dict()
    assert "body_id" not in scenario.target.to_dict()
    assert all("body_id" not in item.to_dict() for item in scenario.distractors)


def test_development_seed_zero_runs_existing_oracle_pick_place() -> None:
    scenario = generate_scenario(0)
    with World(gui=False) as world:
        world.reset_from_scenario(scenario)
        result = oracle_pick_place(world)

        assert result.success, result.failure_reason
        assert result.pick_success
        assert result.cube_inside_tray
        assert result.cube_released
        assert result.constraint_count_before == result.constraint_count_after == 0
