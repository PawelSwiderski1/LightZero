import argparse

from run_mountain_car_full_intrinsic_rnd import (
    build_parser as build_full_intrinsic_parser,
    build_preset_overrides as build_full_intrinsic_overrides,
)
from run_mountain_car_optimistic_shared_tree import run


def build_parser() -> argparse.ArgumentParser:
    parser = build_full_intrinsic_parser()
    parser.description = (
        "Full-intrinsic MountainCar with an active physical-state query policy. "
        "Each replay minibatch recomputes exact phase-current RND measurements and "
        "jointly trains prospective intrinsic edge rewards from MuZero's own h/g unroll."
    )
    parser.add_argument(
        "--rnd-landscape-query-reader-use-context",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--rnd-landscape-query-train-world-model",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--rnd-landscape-query-member-specific-reader",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--rnd-landscape-query-target-transform",
        choices=("log", "raw", "normalized_raw"),
        default="log",
    )
    parser.add_argument(
        "--rnd-landscape-query-normalized-raw-reference-size", type=int, default=8192
    )
    parser.add_argument(
        "--rnd-landscape-query-normalized-raw-clip", type=float, default=10.0
    )
    parser.add_argument(
        "--rnd-landscape-query-normalized-raw-min-scale", type=float, default=1e-8
    )
    parser.add_argument(
        "--rnd-landscape-query-gradient-diagnostics-interval", type=int, default=0
    )
    parser.add_argument(
        "--rnd-landscape-query-initialization-seed", type=int, default=None
    )
    parser.set_defaults(
        exp_name="data_ensemble_muzero/mountain_car_rnd_landscape_query_seed0",
        rnd_landscape_query=True,
        rnd_landscape_query_count=16,
        rnd_landscape_query_hidden_dim=64,
        rnd_landscape_query_learning_rate=3e-4,
        rnd_landscape_query_loss_weight=1.0,
        rnd_landscape_query_max_horizon=8,
        rnd_landscape_query_min_snapshots=2,
        rnd_landscape_query_diversity_weight=0.01,
        rnd_landscape_query_centered_prediction_weight=0.0,
        rnd_landscape_query_grad_clip=5.0,
        rnd_landscape_query_reader_use_context=True,
        rnd_landscape_query_member_specific_reader=False,
        rnd_landscape_query_target_transform="log",
        rnd_landscape_query_normalized_raw_reference_size=8192,
        rnd_landscape_query_normalized_raw_clip=10.0,
        rnd_landscape_query_normalized_raw_min_scale=1e-8,
        rnd_landscape_query_gradient_diagnostics_interval=0,
        rnd_landscape_query_train_world_model=True,
        rnd_landscape_query_use_tree_rewards=True,
    )
    return parser


def build_preset_overrides(args: argparse.Namespace) -> dict:
    overrides = build_full_intrinsic_overrides(args)
    overrides.update(
        rnd_landscape_query=bool(args.rnd_landscape_query),
        rnd_landscape_query_count=max(1, int(args.rnd_landscape_query_count)),
        rnd_landscape_query_hidden_dim=max(8, int(args.rnd_landscape_query_hidden_dim)),
        rnd_landscape_query_learning_rate=max(
            1e-8, float(args.rnd_landscape_query_learning_rate)
        ),
        rnd_landscape_query_loss_weight=max(
            0.0, float(args.rnd_landscape_query_loss_weight)
        ),
        rnd_landscape_query_max_horizon=max(
            1, int(args.rnd_landscape_query_max_horizon)
        ),
        rnd_landscape_query_min_snapshots=max(
            1, int(args.rnd_landscape_query_min_snapshots)
        ),
        rnd_landscape_query_diversity_weight=max(
            0.0, float(args.rnd_landscape_query_diversity_weight)
        ),
        rnd_landscape_query_centered_prediction_weight=max(
            0.0, float(args.rnd_landscape_query_centered_prediction_weight)
        ),
        rnd_landscape_query_grad_clip=max(0.0, float(args.rnd_landscape_query_grad_clip)),
        rnd_landscape_query_reader_use_context=bool(
            args.rnd_landscape_query_reader_use_context
        ),
        rnd_landscape_query_member_specific_reader=bool(
            args.rnd_landscape_query_member_specific_reader
        ),
        rnd_landscape_query_target_transform=str(
            args.rnd_landscape_query_target_transform
        ),
        rnd_landscape_query_normalized_raw_reference_size=max(
            1, int(args.rnd_landscape_query_normalized_raw_reference_size)
        ),
        rnd_landscape_query_normalized_raw_clip=max(
            1e-6, float(args.rnd_landscape_query_normalized_raw_clip)
        ),
        rnd_landscape_query_normalized_raw_min_scale=max(
            1e-12, float(args.rnd_landscape_query_normalized_raw_min_scale)
        ),
        rnd_landscape_query_gradient_diagnostics_interval=max(
            0, int(args.rnd_landscape_query_gradient_diagnostics_interval)
        ),
        rnd_landscape_query_train_world_model=bool(
            args.rnd_landscape_query_train_world_model
        ),
        rnd_landscape_query_use_tree_rewards=bool(
            args.rnd_landscape_query_use_tree_rewards
        ),
        rnd_landscape_query_initialization_seed=(
            int(args.rnd_landscape_query_initialization_seed)
            if args.rnd_landscape_query_initialization_seed is not None
            else 200000 + int(args.seed)
        ),
    )
    return overrides


def main() -> None:
    args = build_parser().parse_args()
    run(args, preset_overrides=build_preset_overrides(args))


if __name__ == "__main__":
    main()
