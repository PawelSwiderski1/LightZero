import argparse

from run_mountain_car_hybrid_vicreg_random_value_init_mixed_sve import (
    build_parser as build_mixed_sve_parser,
    build_preset_overrides as build_mixed_sve_preset_overrides,
)
from run_mountain_car_optimistic_shared_tree import run


def build_parser() -> argparse.ArgumentParser:
    parser = build_mixed_sve_parser()
    parser.description = (
        "MountainCar hybrid VICReg mixed-SVE control with td_steps=5."
    )
    parser.set_defaults(
        exp_name=(
            "data_ensemble_muzero/"
            "mountain_car_hybrid_vicreg_random_value_init_mixed_sve_td5_seed0"
        ),
    )
    return parser


def main() -> None:
    overrides = build_mixed_sve_preset_overrides()
    overrides["td_steps"] = 5
    run(build_parser().parse_args(), preset_overrides=overrides)


if __name__ == "__main__":
    main()
