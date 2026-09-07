"""Visible GUI demonstration of the Phase 3 oracle pick."""

import time

from embodied_manipulation.simulation import World

from .pick import oracle_pick

GUI_INITIAL_PAUSE = 1.0
GUI_STAGE_PAUSE = 0.5
GUI_STEP_DELAY = 1.0 / 240.0


def main() -> None:
    try:
        with World(gui=True) as world:
            world.reset(seed=0)
            time.sleep(GUI_INITIAL_PAUSE)
            result = oracle_pick(
                world,
                step_delay=GUI_STEP_DELAY,
                stage_pause=GUI_STAGE_PAUSE,
            )

            print(f"initial cube position: {result.initial_cube_position}")
            print(f"pre-grasp target: {result.pregrasp_target_position}")
            print(f"grasp target: {result.grasp_target_position}")
            print(
                "end-effector position at grasp: "
                f"{result.end_effector_position_at_grasp}"
            )
            if result.open_result is not None:
                print(
                    "gripper open target/state: "
                    f"{result.open_result.target_opening_width:.6f} / "
                    f"{result.open_result.final_opening_width:.6f} m"
                )
            if result.close_result is not None:
                print(
                    "gripper close target/state: "
                    f"{result.close_result.target_opening_width:.6f} / "
                    f"{result.close_result.final_opening_width:.6f} m "
                    f"(contact-stalled={result.close_result.stalled})"
                )
            print(f"grasp contact count: {len(result.grasp_contacts)}")
            for contact in result.grasp_contacts:
                print(
                    f"  finger link {contact.finger_link_index}: "
                    f"normal force={contact.normal_force:.6f} N, "
                    f"distance={contact.contact_distance:.6f} m"
                )
            print(f"final cube position: {result.final_cube_position}")
            print(f"cube lift distance: {result.cube_lift_distance:.6f} m")
            print(
                "final cube-to-end-effector distance: "
                f"{result.final_cube_to_end_effector_distance:.6f} m"
            )
            print(f"success: {result.success}")
            print(f"failure reason: {result.failure_reason}")
            print(
                f"simulation steps: arm={result.arm_steps}, "
                f"gripper={result.gripper_steps}, total={result.total_steps}"
            )
            print(
                "constraints created: "
                f"{result.constraint_count_after - result.constraint_count_before}"
            )

            while world.is_connected:
                world.step()
                time.sleep(world.time_step)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
