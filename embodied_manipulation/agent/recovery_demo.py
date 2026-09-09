"""GUI demonstrations for bounded post- and pre-verification recovery."""

from __future__ import annotations

import argparse
from time import sleep

from embodied_manipulation.benchmark import generate_scenario
from embodied_manipulation.simulation import World

from .recovery import RecoveryFaultInjection, VisionRecoveryAgent

GUI_INITIAL_PAUSE = 1.0
GUI_STAGE_PAUSE = 0.35
GUI_STEP_DELAY = 1.0 / 240.0
DEMO_SEED = 0
DEMO_DISTRACTOR_COUNT = 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--failure",
        choices=("grasp", "placement", "preverification-ik"),
        default="grasp",
        help="first-attempt fault to demonstrate",
    )
    args = parser.parse_args()
    fault = (
        RecoveryFaultInjection.grasp_miss()
        if args.failure in {"grasp", "preverification-ik"}
        else RecoveryFaultInjection.placement_miss()
    )
    seed = 10 if args.failure == "preverification-ik" else DEMO_SEED
    distractor_count = seed % 3 if seed != DEMO_SEED else DEMO_DISTRACTOR_COUNT
    scenario = generate_scenario(seed, distractor_count=distractor_count)
    print(f"instruction: {scenario.task.instruction}")
    print(f"controlled first-attempt failure: {args.failure}")
    try:
        with World(gui=True) as world:
            world.reset_from_scenario(scenario)
            sleep(GUI_INITIAL_PAUSE)
            result = VisionRecoveryAgent().run(
                scenario.task.instruction,
                world,
                step_delay=GUI_STEP_DELAY,
                stage_pause=GUI_STAGE_PAUSE,
                stage_callback=_print_stage,
                fault_injection=fault,
            )
            trace = result.recovery_trace
            print(f"recovery activated: {result.recovery_activated}")
            print(f"recovery attempts: {result.recovery_attempts}")
            print(f"recovered: {result.recovered}")
            print(
                "final objective success: "
                f"{bool(result.final_evaluation and result.final_evaluation.success)}"
            )
            print(f"overall success: {result.success}")
            print(f"observation count: {result.observation_count}")
            print(f"total simulation steps: {result.total_steps}")
            if trace is not None:
                print(f"recovery entry type: {trace.recovery_entry_type}")
                print(
                    "initial execution failure code: "
                    f"{trace.initial_execution_failure_code}"
                )
                print(f"diagnosis: {trace.diagnosis}")
                if trace.recovery_source_detection is not None:
                    print(
                        "re-grounded cube: "
                        f"{trace.recovery_source_detection.world_position}"
                    )
                if trace.recovery_target_detection is not None:
                    print(
                        "re-grounded tray: "
                        f"{trace.recovery_target_detection.world_position}"
                    )
            while world.is_connected:
                world.step()
                sleep(world.time_step)
    except KeyboardInterrupt:
        pass


def _print_stage(stage: str, value: object) -> None:
    messages = {
        "initial_observation_captured": "OBSERVE #1 / GROUND / PLAN",
        "post_grasp_observation_captured": "VERIFY GRASP",
        "post_placement_observation_captured": "VERIFY PLACEMENT",
        "recovery_activated": "RECOVERY ACTIVATED",
        "fresh_recovery_observation_captured": "FRESH RECOVERY RGB-D",
        "failure_diagnosed": "STRUCTURED FAILURE DIAGNOSED",
        "recovery_source_grounded": "RE-GROUNDED CUBE",
        "recovery_target_grounded": "RE-GROUNDED TRAY",
        "recovery_planned": "REPLAN / RETRY",
        "recovery_post_grasp_observation_captured": "VERIFY RECOVERY GRASP",
        "recovery_post_placement_observation_captured": "VERIFY RECOVERY PLACEMENT",
    }
    if stage in messages:
        print(messages[stage])
    elif stage == "grasp_verified" and not value.verified:
        print(f"GRASP VERIFICATION FAILED: {value.reason}")
    elif stage == "placement_verified" and not value.verified:
        print(f"PLACEMENT VERIFICATION FAILED: {value.reason}")
    elif stage == "recovery_grasp_verified":
        print(
            "GRASP RECOVERED"
            if value.verified
            else f"RECOVERY GRASP FAILED: {value.reason}"
        )
    elif stage == "recovery_placement_verified":
        print(
            "PLACEMENT RECOVERED"
            if value.verified
            else f"RECOVERY PLACEMENT FAILED: {value.reason}"
        )


if __name__ == "__main__":
    main()
