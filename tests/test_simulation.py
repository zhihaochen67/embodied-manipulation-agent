from collections.abc import Iterator

import numpy as np
import pybullet
import pytest

from embodied_manipulation.simulation.objects import (
    CUBE_HALF_EXTENT,
    TABLE_CENTER,
    TABLE_SURFACE_Z,
    TABLE_TOP_HALF_EXTENTS,
)
from embodied_manipulation.simulation.robot import (
    PANDA_INITIAL_JOINT_POSITIONS,
    get_initial_joint_positions,
)
from embodied_manipulation.simulation.world import World


@pytest.fixture
def world() -> Iterator[World]:
    simulation = World(gui=False)
    try:
        yield simulation
    finally:
        simulation.disconnect()


def test_world_initializes_and_resets_headlessly(world: World) -> None:
    scene = world.reset(seed=0)

    connection = pybullet.getConnectionInfo(physicsClientId=world.client_id)
    assert connection["isConnected"] == 1
    assert connection["connectionMethod"] == pybullet.DIRECT
    assert scene == world.scene


def test_scene_contains_expected_bodies(world: World) -> None:
    scene = world.reset(seed=0)

    body_ids = {scene.plane_id, scene.table_id, scene.robot_id, scene.cube_id}
    assert len(body_ids) == 4
    for body_id in body_ids:
        assert pybullet.getBodyInfo(body_id, physicsClientId=world.client_id)


def test_same_seed_produces_same_cube_position(world: World) -> None:
    first_position = world.reset(seed=17).cube_initial_position
    second_position = world.reset(seed=17).cube_initial_position

    np.testing.assert_array_equal(first_position, second_position)


def test_cube_pose_is_finite_and_plausible(world: World) -> None:
    world.reset(seed=0)
    position, orientation = world.get_cube_pose()

    assert np.all(np.isfinite(position))
    assert np.all(np.isfinite(orientation))
    assert TABLE_CENTER[0] - TABLE_TOP_HALF_EXTENTS[0] < position[0]
    assert position[0] < TABLE_CENTER[0] + TABLE_TOP_HALF_EXTENTS[0]
    assert TABLE_CENTER[1] - TABLE_TOP_HALF_EXTENTS[1] < position[1]
    assert position[1] < TABLE_CENTER[1] + TABLE_TOP_HALF_EXTENTS[1]
    assert position[2] == pytest.approx(TABLE_SURFACE_Z + CUBE_HALF_EXTENT)
    assert np.linalg.norm(orientation) == pytest.approx(1.0)


def test_panda_initial_joint_state_is_deterministic(world: World) -> None:
    first_scene = world.reset(seed=0)
    first_positions = get_initial_joint_positions(world.client_id, first_scene.robot_id)
    second_scene = world.reset(seed=999)
    second_positions = get_initial_joint_positions(world.client_id, second_scene.robot_id)

    assert first_positions == pytest.approx(PANDA_INITIAL_JOINT_POSITIONS)
    assert second_positions == pytest.approx(first_positions)


def test_fixed_camera_renders_headlessly(world: World) -> None:
    world.reset(seed=0)
    width, height, rgba, depth, segmentation = world.render()

    assert (width, height) == (world.camera.width, world.camera.height)
    assert np.asarray(rgba).size == width * height * 4
    assert np.asarray(depth).size == width * height
    assert np.asarray(segmentation).size == width * height
