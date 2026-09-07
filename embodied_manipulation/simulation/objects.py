"""Primitive objects for the deterministic tabletop scenes."""

from dataclasses import dataclass

import pybullet

GROUND_Z = -0.65
TABLE_SURFACE_Z = 0.0
TABLE_CENTER = (0.55, 0.0, -0.04)
TABLE_TOP_HALF_EXTENTS = (0.75, 0.5, 0.04)
TABLE_LEG_HALF_EXTENTS = (0.05, 0.05, 0.285)
TABLE_COLOR = (0.55, 0.32, 0.16, 1.0)
CUBE_HALF_EXTENT = 0.025
COLOR_RGBA = {
    "red": (0.9, 0.05, 0.05, 1.0),
    "blue": (0.05, 0.25, 0.9, 1.0),
    "yellow": (0.95, 0.8, 0.05, 1.0),
    "green": (0.05, 0.65, 0.15, 1.0),
}
CUBE_COLOR = COLOR_RGBA["red"]

TRAY_CENTER = (0.50, 0.28)
TRAY_INNER_HALF_EXTENTS = (0.07, 0.08)
TRAY_FLOOR_THICKNESS = 0.008
TRAY_WALL_HEIGHT = 0.045
TRAY_WALL_THICKNESS = 0.01
TRAY_COLOR = COLOR_RGBA["blue"]
TRAY_LATERAL_FRICTION = 0.6
TRAY_RESTITUTION = 0.05


@dataclass(frozen=True, slots=True)
class Tray:
    """Body IDs and explicit usable geometry for the primitive tray."""

    floor_id: int
    wall_ids: tuple[int, int, int, int]
    center: tuple[float, float]
    inner_bounds: tuple[float, float, float, float]
    floor_height: float
    wall_height: float
    wall_thickness: float

    @property
    def body_ids(self) -> tuple[int, ...]:
        return (self.floor_id, *self.wall_ids)

    @property
    def wall_top_height(self) -> float:
        return self.floor_height + self.wall_height


def create_table(client_id: int) -> int:
    """Create a static table from box primitives and return its body ID."""
    top_collision = pybullet.createCollisionShape(
        pybullet.GEOM_BOX,
        halfExtents=TABLE_TOP_HALF_EXTENTS,
        physicsClientId=client_id,
    )
    top_visual = pybullet.createVisualShape(
        pybullet.GEOM_BOX,
        halfExtents=TABLE_TOP_HALF_EXTENTS,
        rgbaColor=TABLE_COLOR,
        physicsClientId=client_id,
    )
    leg_collision = pybullet.createCollisionShape(
        pybullet.GEOM_BOX,
        halfExtents=TABLE_LEG_HALF_EXTENTS,
        physicsClientId=client_id,
    )
    leg_visual = pybullet.createVisualShape(
        pybullet.GEOM_BOX,
        halfExtents=TABLE_LEG_HALF_EXTENTS,
        rgbaColor=TABLE_COLOR,
        physicsClientId=client_id,
    )

    leg_z = -0.325
    leg_positions = [
        (-0.62, -0.38, leg_z),
        (-0.62, 0.38, leg_z),
        (0.62, -0.38, leg_z),
        (0.62, 0.38, leg_z),
    ]
    link_count = len(leg_positions)
    table_id = pybullet.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=top_collision,
        baseVisualShapeIndex=top_visual,
        basePosition=TABLE_CENTER,
        linkMasses=[0.0] * link_count,
        linkCollisionShapeIndices=[leg_collision] * link_count,
        linkVisualShapeIndices=[leg_visual] * link_count,
        linkPositions=leg_positions,
        linkOrientations=[(0.0, 0.0, 0.0, 1.0)] * link_count,
        linkInertialFramePositions=[(0.0, 0.0, 0.0)] * link_count,
        linkInertialFrameOrientations=[(0.0, 0.0, 0.0, 1.0)] * link_count,
        linkParentIndices=[0] * link_count,
        linkJointTypes=[pybullet.JOINT_FIXED] * link_count,
        linkJointAxis=[(0.0, 0.0, 0.0)] * link_count,
        physicsClientId=client_id,
    )
    pybullet.changeDynamics(
        table_id,
        -1,
        lateralFriction=0.8,
        physicsClientId=client_id,
    )
    return table_id


def create_cube(
    client_id: int,
    position: tuple[float, float, float],
    color: tuple[float, float, float, float] = CUBE_COLOR,
    half_extent: float = CUBE_HALF_EXTENT,
) -> int:
    """Create one primitive cube and return its body ID."""
    half_extents = (half_extent,) * 3
    collision_shape = pybullet.createCollisionShape(
        pybullet.GEOM_BOX,
        halfExtents=half_extents,
        physicsClientId=client_id,
    )
    visual_shape = pybullet.createVisualShape(
        pybullet.GEOM_BOX,
        halfExtents=half_extents,
        rgbaColor=color,
        physicsClientId=client_id,
    )
    return pybullet.createMultiBody(
        baseMass=0.05,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=position,
        physicsClientId=client_id,
    )


def create_tray(
    client_id: int,
    *,
    center: tuple[float, float] = TRAY_CENTER,
    inner_half_extents: tuple[float, float] = TRAY_INNER_HALF_EXTENTS,
    floor_thickness: float = TRAY_FLOOR_THICKNESS,
    wall_height: float = TRAY_WALL_HEIGHT,
    wall_thickness: float = TRAY_WALL_THICKNESS,
    color: tuple[float, float, float, float] = TRAY_COLOR,
) -> Tray:
    """Create one colored tray from a floor and four fixed box walls."""
    center_x, center_y = center
    inner_x, inner_y = inner_half_extents
    floor_height = TABLE_SURFACE_Z + floor_thickness
    outer_x = inner_x + wall_thickness
    outer_y = inner_y + wall_thickness

    floor_id = _create_static_box(
        client_id,
        half_extents=(outer_x, outer_y, floor_thickness / 2.0),
        position=(center_x, center_y, TABLE_SURFACE_Z + floor_thickness / 2.0),
        color=color,
    )
    wall_z = floor_height + wall_height / 2.0
    wall_specs = (
        (
            (wall_thickness / 2.0, outer_y, wall_height / 2.0),
            (center_x - inner_x - wall_thickness / 2.0, center_y, wall_z),
        ),
        (
            (wall_thickness / 2.0, outer_y, wall_height / 2.0),
            (center_x + inner_x + wall_thickness / 2.0, center_y, wall_z),
        ),
        (
            (inner_x, wall_thickness / 2.0, wall_height / 2.0),
            (center_x, center_y - inner_y - wall_thickness / 2.0, wall_z),
        ),
        (
            (inner_x, wall_thickness / 2.0, wall_height / 2.0),
            (center_x, center_y + inner_y + wall_thickness / 2.0, wall_z),
        ),
    )
    wall_ids = tuple(
        _create_static_box(
            client_id,
            half_extents=half_extents,
            position=position,
            color=color,
        )
        for half_extents, position in wall_specs
    )
    return Tray(
        floor_id=floor_id,
        wall_ids=wall_ids,
        center=center,
        inner_bounds=(
            center_x - inner_x,
            center_x + inner_x,
            center_y - inner_y,
            center_y + inner_y,
        ),
        floor_height=floor_height,
        wall_height=wall_height,
        wall_thickness=wall_thickness,
    )


def _create_static_box(
    client_id: int,
    *,
    half_extents: tuple[float, float, float],
    position: tuple[float, float, float],
    color: tuple[float, float, float, float],
) -> int:
    collision_shape = pybullet.createCollisionShape(
        pybullet.GEOM_BOX,
        halfExtents=half_extents,
        physicsClientId=client_id,
    )
    visual_shape = pybullet.createVisualShape(
        pybullet.GEOM_BOX,
        halfExtents=half_extents,
        rgbaColor=color,
        physicsClientId=client_id,
    )
    body_id = pybullet.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=position,
        physicsClientId=client_id,
    )
    pybullet.changeDynamics(
        body_id,
        -1,
        lateralFriction=TRAY_LATERAL_FRICTION,
        restitution=TRAY_RESTITUTION,
        physicsClientId=client_id,
    )
    return body_id
