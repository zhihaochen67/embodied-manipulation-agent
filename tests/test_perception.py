from dataclasses import replace
from math import isfinite, radians, tan

import cv2
import numpy as np
import pybullet
import pytest

from embodied_manipulation.benchmark import generate_scenario
from embodied_manipulation.language import ObjectRef
from embodied_manipulation.perception import (
    AmbiguousDetectionError,
    CameraIntrinsics,
    ObjectNotFoundError,
    OraclePerception,
    PerceptionError,
    RGBDObservation,
    UnsupportedObjectError,
    VisionPerception,
    back_project_world,
    depth_buffer_to_metric,
    segment_color,
)
from embodied_manipulation.simulation import FixedCamera, World
from embodied_manipulation.simulation.objects import (
    COLOR_RGBA,
    TABLE_SURFACE_Z,
    create_cube,
)


# Fixed-seed development results are below 3 mm. Six millimeters leaves room
# for renderer/platform rasterization differences while remaining low-centimeter.
POSITION_TOLERANCE = 0.006


def test_oracle_perception_returns_source_cube_pose() -> None:
    scenario = generate_scenario(0)
    with World(gui=False) as world:
        world.reset_from_scenario(scenario)
        result = OraclePerception(world).locate(scenario.task.source)

        np.testing.assert_allclose(
            result.world_position,
            world.get_cube_pose()[0],
            atol=1e-12,
        )
        assert result.reference == "object_center"
        assert result.source == "oracle"


def test_camera_intrinsics_are_finite_and_match_projection() -> None:
    camera = FixedCamera()
    intrinsics = CameraIntrinsics.from_camera(camera)
    projection = np.asarray(camera.projection_matrix()).reshape(4, 4, order="F")

    assert intrinsics.width == camera.width
    assert intrinsics.height == camera.height
    assert intrinsics.aspect_ratio == pytest.approx(camera.width / camera.height)
    assert all(
        isfinite(value)
        for value in (
            intrinsics.fx,
            intrinsics.fy,
            intrinsics.cx,
            intrinsics.cy,
        )
    )
    assert intrinsics.fy == pytest.approx(
        camera.height / (2.0 * tan(radians(camera.field_of_view) / 2.0))
    )
    assert intrinsics.fx == pytest.approx(camera.width * projection[0, 0] / 2.0)
    assert intrinsics.fy == pytest.approx(camera.height * projection[1, 1] / 2.0)


def test_depth_buffer_conversion_is_metric_and_finite() -> None:
    near, far = 0.1, 4.0
    metric = depth_buffer_to_metric(np.array([0.0, 0.5, 1.0]), near, far)

    assert np.all(np.isfinite(metric))
    assert metric[0] == pytest.approx(near)
    assert metric[-1] == pytest.approx(far)
    assert np.all(np.diff(metric) > 0.0)


def test_back_projection_recovers_a_known_world_point() -> None:
    camera = FixedCamera()
    intrinsics = CameraIntrinsics.from_camera(camera)
    view = np.asarray(camera.view_matrix()).reshape(4, 4, order="F")
    projection = np.asarray(camera.projection_matrix()).reshape(4, 4, order="F")
    expected = np.array([0.52, -0.04, 0.025, 1.0])
    camera_point = view @ expected
    clip = projection @ camera_point
    ndc = clip[:3] / clip[3]
    u = (ndc[0] + 1.0) * camera.width / 2.0 - 0.5
    v = (1.0 - ndc[1]) * camera.height / 2.0 - 0.5

    reconstructed = back_project_world(
        u,
        v,
        -camera_point[2],
        intrinsics,
        camera.view_matrix(),
    )

    np.testing.assert_allclose(reconstructed, expected[:3], atol=1e-7)


def test_rgb_segmentation_detects_red_region() -> None:
    rgb = np.zeros((30, 30, 3), dtype=np.uint8)
    rgb[5:25, 7:27] = (255, 0, 0)

    mask = segment_color(rgb, "red")

    assert mask.dtype == np.bool_
    assert int(mask.sum()) == 400


@pytest.mark.parametrize(
    ("color", "rgb"),
    (
        ("red", (255, 0, 0)),
        ("blue", (0, 0, 255)),
        ("yellow", (255, 255, 0)),
        ("green", (0, 255, 0)),
    ),
)
def test_rgb_segmentation_detects_all_supported_colors(
    color: str,
    rgb: tuple[int, int, int],
) -> None:
    image = np.full((4, 5, 3), rgb, dtype=np.uint8)

    assert np.all(segment_color(image, color))


def test_red_hue_wrap_around_is_supported() -> None:
    hsv = np.array([[[0, 255, 255], [179, 255, 255]]], dtype=np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    np.testing.assert_array_equal(segment_color(rgb, "red"), [[True, True]])


def test_rgbd_observation_has_explicit_shapes_and_types() -> None:
    with World(gui=False) as world:
        world.reset(seed=0)
        observation = RGBDObservation.capture(world.camera, world.client_id)

    assert observation.rgb.shape == (480, 640, 3)
    assert observation.rgb.dtype == np.uint8
    assert observation.depth_buffer.shape == (480, 640)
    assert np.issubdtype(observation.depth_buffer.dtype, np.floating)
    assert len(observation.view_matrix) == len(observation.projection_matrix) == 16


@pytest.mark.parametrize("seed", range(10))
def test_fixed_seed_rgbd_localization_accuracy(seed: int) -> None:
    scenario = generate_scenario(seed, distractor_count=seed % 3)
    with World(gui=False) as world:
        world.reset_from_scenario(scenario)
        observation = RGBDObservation.capture(world.camera, world.client_id)
        vision = VisionPerception(observation)
        oracle = OraclePerception(world)

        cube = vision.locate(scenario.task.source)
        tray = vision.locate(scenario.task.target)
        cube_truth = oracle.locate(scenario.task.source)
        tray_truth = oracle.locate(scenario.task.target)

    assert np.linalg.norm(
        np.subtract(cube.world_position, cube_truth.world_position)
    ) <= POSITION_TOLERANCE
    assert np.linalg.norm(
        np.subtract(tray.world_position, tray_truth.world_position)
    ) <= POSITION_TOLERANCE
    assert cube.reference == "object_center"
    assert tray.reference == "tray_floor_body_center"
    assert cube.mask_area is not None and cube.mask_area >= 20
    assert tray.mask_area is not None and tray.mask_area >= 20
    assert cube.metric_depth is not None
    assert observation.near_plane <= cube.metric_depth <= observation.far_plane


def test_same_observation_produces_identical_localization() -> None:
    scenario = generate_scenario(4, distractor_count=2)
    with World(gui=False) as world:
        world.reset_from_scenario(scenario)
        observation = RGBDObservation.capture(world.camera, world.client_id)
        vision = VisionPerception(observation)

        first = vision.locate(scenario.task.source)
        second = vision.locate(scenario.task.source)

    assert first == second


def test_type_geometry_distinguishes_same_color_cube_and_tray() -> None:
    original = generate_scenario(0)
    source = replace(original.source, color="red")
    target = replace(original.target, color="red")
    task = replace(
        original.task,
        source=ObjectRef(source.object_id, "cube", "red"),
        target=ObjectRef(target.object_id, "tray", "red"),
    )
    scenario = replace(original, source=source, target=target, task=task)
    with World(gui=False) as world:
        world.reset_from_scenario(scenario)
        vision = VisionPerception(
            RGBDObservation.capture(world.camera, world.client_id)
        )

        cube = vision.locate(task.source)
        tray = vision.locate(task.target)

    np.testing.assert_allclose(
        cube.world_position,
        source.initial_position,
        atol=POSITION_TOLERANCE,
    )
    np.testing.assert_allclose(
        tray.world_position,
        target.initial_position,
        atol=POSITION_TOLERANCE,
    )


def test_missing_and_unsupported_targets_fail_cleanly() -> None:
    scenario = generate_scenario(0)
    visible_colors = {scenario.source.color, scenario.target.color}
    missing_color = next(
        color for color in COLOR_RGBA if color not in visible_colors
    )
    with World(gui=False) as world:
        world.reset_from_scenario(scenario)
        vision = VisionPerception(
            RGBDObservation.capture(world.camera, world.client_id)
        )

        with pytest.raises(ObjectNotFoundError):
            vision.locate(ObjectRef(None, "cube", missing_color))
        with pytest.raises(UnsupportedObjectError):
            vision.locate(ObjectRef(None, "sphere", "red"))
        with pytest.raises(UnsupportedObjectError):
            vision.locate(ObjectRef(None, "cube", "purple"))


def test_invalid_object_depth_fails_cleanly() -> None:
    scenario = generate_scenario(0)
    with World(gui=False) as world:
        world.reset_from_scenario(scenario)
        observation = RGBDObservation.capture(world.camera, world.client_id)
    invalid = replace(
        observation,
        depth_buffer=np.full_like(observation.depth_buffer, np.nan),
    )

    with pytest.raises(PerceptionError, match="metric depth"):
        VisionPerception(invalid).locate(scenario.task.source)


def test_multiple_matching_image_components_are_ambiguous() -> None:
    with World(gui=False) as world:
        world.reset(seed=0)
        create_cube(
            world.client_id,
            (0.68, 0.10, TABLE_SURFACE_Z + 0.025),
            color=COLOR_RGBA["red"],
        )
        observation = RGBDObservation.capture(world.camera, world.client_id)

        with pytest.raises(AmbiguousDetectionError):
            VisionPerception(observation).locate(
                ObjectRef(None, "cube", "red")
            )


def test_vision_does_not_consume_id_image_or_pose_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    class PoisonIdImage:
        def __array__(self, *_: object, **__: object) -> np.ndarray:
            raise AssertionError("renderer object-ID image was consumed")

    scenario = generate_scenario(2, distractor_count=2)
    with World(gui=False) as world:
        world.reset_from_scenario(scenario)
        width, height, rgba, depth, _ = world.render()
        observation = RGBDObservation.from_render(
            (width, height, rgba, depth, PoisonIdImage()),
            view_matrix=world.camera.view_matrix(),
            projection_matrix=world.camera.projection_matrix(),
            intrinsics=CameraIntrinsics.from_camera(world.camera),
            near_plane=world.camera.near_plane,
            far_plane=world.camera.far_plane,
        )

        def fail_pose_lookup(*_: object, **__: object) -> None:
            raise AssertionError("ground-truth pose lookup was used")

        monkeypatch.setattr(
            pybullet,
            "getBasePositionAndOrientation",
            fail_pose_lookup,
        )
        result = VisionPerception(observation).locate(scenario.task.source)

    assert result.source == "vision"
    assert result.object_ref == scenario.task.source
