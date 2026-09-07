"""Manual GUI demo that reaches to a point above the Phase 1 cube."""

import time

from embodied_manipulation.simulation import World

from .arm_controller import ArmController, target_above_cube

GUI_INITIAL_PAUSE = 1.0
GUI_STEP_DELAY = 1.0 / 60.0


def main() -> None:
    try:
        with World(gui=True) as world:
            scene = world.reset(seed=0)
            controller = ArmController(world.client_id, scene.robot_id)
            cube_position, _ = world.get_cube_pose()
            target = target_above_cube(cube_position)
            initial_position, _ = controller.end_effector_pose()

            time.sleep(GUI_INITIAL_PAUSE)
            result = controller.move_end_effector(
                target,
                step_delay=GUI_STEP_DELAY,
            )
            print(f"cube position: {cube_position}")
            print(f"target position: {result.target_position}")
            print(f"initial position: {initial_position}")
            print(f"final position: {result.final_position}")
            print(f"final position error: {result.position_error:.6f} m")
            print(f"success: {result.success}")
            print(f"simulation steps: {result.steps}")
            if result.failure_reason:
                print(f"failure reason: {result.failure_reason}")

            while world.is_connected:
                world.step()
                time.sleep(world.time_step)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
