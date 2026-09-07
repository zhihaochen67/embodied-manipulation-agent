"""Visible GUI demonstration of the Phase 4 oracle pick-and-place."""

import time

from embodied_manipulation.simulation import World

from .pick_place import oracle_pick_place

GUI_INITIAL_PAUSE = 1.0
GUI_STAGE_PAUSE = 0.35
GUI_STEP_DELAY = 1.0 / 240.0
GUI_PRE_RELEASE_PAUSE = 0.55
GUI_RELEASE_STEP_DELAY = 1.0 / 60.0
GUI_SLOW_RELEASE_SETTLING_STEPS = 30


def main() -> None:
    try:
        with World(gui=True) as world:
            world.reset(seed=0, include_tray=True)
            time.sleep(GUI_INITIAL_PAUSE)
            result = oracle_pick_place(
                world,
                step_delay=GUI_STEP_DELAY,
                stage_pause=GUI_STAGE_PAUSE,
                pre_release_pause=GUI_PRE_RELEASE_PAUSE,
                release_step_delay=GUI_RELEASE_STEP_DELAY,
                slow_release_settling_steps=GUI_SLOW_RELEASE_SETTLING_STEPS,
            )

            linear_speed = sum(
                value * value for value in result.final_cube_linear_velocity
            ) ** 0.5
            angular_speed = sum(
                value * value for value in result.final_cube_angular_velocity
            ) ** 0.5
            print(f"initial cube position: {result.initial_cube_position}")
            print(f"tray center: {result.tray_center}")
            print(f"grasp result: {result.pick_success}")
            print(f"above-tray target: {result.above_tray_target}")
            print(f"transport finger targets: {result.transport_finger_targets}")
            print(
                "maximum transport finger tracking errors: "
                f"{result.maximum_transport_finger_tracking_errors}"
            )
            print(f"release target: {result.release_target}")
            print(
                "actual release clearance above tray floor: "
                f"{result.actual_release_clearance_above_floor}"
            )
            print(
                "cube position before/shortly after release: "
                f"{result.cube_position_before_release} / "
                f"{result.cube_position_shortly_after_release}"
            )
            print(
                "observed vertical fall distance: "
                f"{result.observed_vertical_fall_distance}"
            )
            release_open_steps = (
                0 if result.release_result is None else result.release_result.steps
            )
            print(f"normal GUI step delay: {GUI_STEP_DELAY:.6f} s")
            print(f"release GUI step delay: {GUI_RELEASE_STEP_DELAY:.6f} s")
            print(f"pre-release pause: {GUI_PRE_RELEASE_PAUSE:.2f} s")
            print(
                "slow-motion release steps: "
                f"{release_open_steps + GUI_SLOW_RELEASE_SETTLING_STEPS} "
                f"({release_open_steps} opening + "
                f"{GUI_SLOW_RELEASE_SETTLING_STEPS} settling)"
            )
            print(f"final cube position: {result.final_cube_position}")
            print(
                "cube inside tray: "
                f"{'yes' if result.cube_inside_tray else 'no'}"
            )
            print(f"cube stable: {'yes' if result.stable else 'no'}")
            print(
                "final linear/angular speed: "
                f"{linear_speed:.6f} m/s / {angular_speed:.6f} rad/s"
            )
            print(f"success: {result.success}")
            print(f"failure reason: {result.failure_reason}")
            print(f"total simulation steps: {result.total_steps}")

            while world.is_connected:
                world.step()
                time.sleep(world.time_step)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
