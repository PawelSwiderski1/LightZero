import argparse

from run_mountain_car_optimistic_shared_tree import run
from run_mountain_car_optimistic_shared_tree_elite import build_elite_parser, elite_preset_overrides


def build_finite_horizon_parser() -> argparse.ArgumentParser:
    parser = build_elite_parser()
    parser.description = (
        "Run the elite-stratified optimistic shared-tree preset as a finite-horizon MountainCar task."
    )
    parser.set_defaults(
        exp_name="data_ensemble_muzero/mountain_car_optimistic_shared_tree_elite_finite_horizon_seed0",
    )
    return parser


def main() -> None:
    args = build_finite_horizon_parser().parse_args()
    overrides = elite_preset_overrides(args)
    overrides.update(
        include_remaining_time=True,
        bootstrap_at_time_limit=False,
    )
    run(args, preset_overrides=overrides)


if __name__ == "__main__":
    main()
