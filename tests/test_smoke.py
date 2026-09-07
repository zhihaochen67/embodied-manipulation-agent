import embodied_manipulation
import pybullet
import pybullet_data


def test_package_import() -> None:
    assert embodied_manipulation is not None


def test_pybullet_import() -> None:
    assert pybullet is not None


def test_pybullet_direct_connection() -> None:
    client_id = pybullet.connect(pybullet.DIRECT)
    try:
        assert client_id >= 0
    finally:
        if client_id >= 0:
            pybullet.disconnect(physicsClientId=client_id)


def test_load_franka_panda() -> None:
    client_id = pybullet.connect(pybullet.DIRECT)
    try:
        assert client_id >= 0
        pybullet.setAdditionalSearchPath(
            pybullet_data.getDataPath(),
            physicsClientId=client_id,
        )
        robot_id = pybullet.loadURDF(
            "franka_panda/panda.urdf",
            useFixedBase=True,
            physicsClientId=client_id,
        )
        assert robot_id >= 0
        assert pybullet.getNumJoints(robot_id, physicsClientId=client_id) > 0
    finally:
        if client_id >= 0:
            pybullet.disconnect(physicsClientId=client_id)
