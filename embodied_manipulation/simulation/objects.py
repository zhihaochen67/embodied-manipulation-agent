"""Primitive scene objects for the Phase 1 tabletop."""

import pybullet

GROUND_Z = -0.65
TABLE_SURFACE_Z = 0.0
TABLE_CENTER = (0.55, 0.0, -0.04)
TABLE_TOP_HALF_EXTENTS = (0.75, 0.5, 0.04)
TABLE_LEG_HALF_EXTENTS = (0.05, 0.05, 0.285)
TABLE_COLOR = (0.55, 0.32, 0.16, 1.0)
CUBE_HALF_EXTENT = 0.025
CUBE_COLOR = (0.9, 0.05, 0.05, 1.0)


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
) -> int:
    """Create the single red Phase 1 cube and return its body ID."""
    half_extents = (CUBE_HALF_EXTENT,) * 3
    collision_shape = pybullet.createCollisionShape(
        pybullet.GEOM_BOX,
        halfExtents=half_extents,
        physicsClientId=client_id,
    )
    visual_shape = pybullet.createVisualShape(
        pybullet.GEOM_BOX,
        halfExtents=half_extents,
        rgbaColor=CUBE_COLOR,
        physicsClientId=client_id,
    )
    return pybullet.createMultiBody(
        baseMass=0.05,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=position,
        physicsClientId=client_id,
    )
