import argparse

from run_mountain_car_hybrid_vicreg_random_value_init_mixed_sve import (
    build_parser as build_mixed_sve_parser,
    build_preset_overrides as build_mixed_sve_preset_overrides,
)
from run_mountain_car_optimistic_shared_tree import run


def build_parser() -> argparse.ArgumentParser:
    parser = build_mixed_sve_parser()
    parser.description = (
        "MountainCar hybrid VICReg run with td_steps=20 and EfficientZero-style "
        "two-reward-prefix search value targets for stale replay."
    )
    parser.set_defaults(
        exp_name=(
            "data_ensemble_muzero/"
            "mountain_car_hybrid_vicreg_random_value_init_efficientzero_seed0"
        ),
    )
    return parser


def build_preset_overrides() -> dict:
    overrides = build_mixed_sve_preset_overrides()
    overrides.update(
        td_steps=20,
        sve_real_reward_prefix_steps=2,
    )
    return overrides


def main() -> None:
    run(build_parser().parse_args(), preset_overrides=build_preset_overrides())


if __name__ == "__main__":
    main()
