import argparse

from run_mountain_car_optimistic_shared_tree import run
from run_mountain_car_rnd_landscape_query import (
    build_parser as build_query_parser,
    build_preset_overrides as build_query_overrides,
)


def build_parser() -> argparse.ArgumentParser:
    parser = build_query_parser()
    parser.description = (
        "RND landscape-query ablation with the query diversity loss disabled. "
        "All prediction targets and other optimizer settings match the baseline."
    )
    parser.set_defaults(
        exp_name="data_ensemble_muzero/mountain_car_rnd_landscape_query_no_diversity_seed0",
        rnd_landscape_query_diversity_weight=0.0,
        rnd_landscape_query_centered_prediction_weight=0.0,
    )
    return parser


def build_preset_overrides(args: argparse.Namespace) -> dict:
    return build_query_overrides(args)


def main() -> None:
    args = build_parser().parse_args()
    run(args, preset_overrides=build_preset_overrides(args))


if __name__ == "__main__":
    main()
