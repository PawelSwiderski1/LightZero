import argparse

from run_mountain_car_optimistic_shared_tree import run
from run_mountain_car_rnd_landscape_query import (
    build_parser as build_query_parser,
    build_preset_overrides as build_query_overrides,
)


def build_parser() -> argparse.ArgumentParser:
    parser = build_query_parser()
    parser.description = (
        "Train MuZero's encoder and dynamics with the latent-only log-RND reader "
        "while retaining the original intrinsic reward head for MCTS edge rewards."
    )
    parser.set_defaults(
        exp_name="data_ensemble_muzero/mountain_car_rnd_query_auxiliary_only_seed0",
        rnd_landscape_query_reader_use_context=False,
        rnd_landscape_query_train_world_model=True,
        rnd_landscape_query_use_tree_rewards=False,
        rnd_landscape_query_diversity_weight=0.0,
    )
    return parser


def build_preset_overrides(args: argparse.Namespace) -> dict:
    overrides = build_query_overrides(args)
    overrides.update(
        rnd_landscape_query_reader_use_context=False,
        rnd_landscape_query_train_world_model=True,
        rnd_landscape_query_use_tree_rewards=False,
        rnd_landscape_query_diversity_weight=0.0,
    )
    return overrides


def main() -> None:
    args = build_parser().parse_args()
    run(args, preset_overrides=build_preset_overrides(args))


if __name__ == "__main__":
    main()
