"""Manual GUI demo for the minimal Phase 1 scene."""

import time

from .world import World


def main() -> None:
    try:
        with World(gui=True) as world:
            scene = world.reset(seed=0)
            print(
                f"robot={scene.robot_id} table={scene.table_id} "
                f"cube={scene.cube_id}"
            )
            while world.is_connected:
                world.step()
                time.sleep(world.time_step)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
