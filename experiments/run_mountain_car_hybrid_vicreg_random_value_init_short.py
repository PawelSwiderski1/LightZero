import argparse

from run_mountain_car_hybrid_vicreg_short import build_parser as build_control_parser, build_preset_overrides
from run_mountain_car_optimistic_shared_tree import run


def build_parser() -> argparse.ArgumentParser:
    parser = build_control_parser()
    parser.description = (
        "Short MountainCar hybrid VICReg ablation with independently randomized ensemble value output heads."
    )
    parser.set_defaults(
        exp_name="data_ensemble_muzero/mountain_car_hybrid_vicreg_random_value_init_short_seed0",
        value_last_linear_layer_init_zero=False,
    )
    return parser


def main() -> None:
    run(build_parser().parse_args(), preset_overrides=build_preset_overrides())


if __name__ == "__main__":
    main()
