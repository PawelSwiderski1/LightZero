import argparse

from run_mountain_car_corrected_rnd_intrinsic_diagnostic import (
    build_parser as build_diagnostic_parser,
    build_preset_overrides as build_diagnostic_preset_overrides,
)
from run_mountain_car_optimistic_shared_tree import run


def build_parser() -> argparse.ArgumentParser:
    parser = build_diagnostic_parser()
    parser.description = (
        "MountainCar with one MuZero tree optimizing extrinsic return plus a "
        "weighted, current-snapshot RND return."
    )
    parser.add_argument("--mixed-reward-beta", type=float, default=50.0)
    parser.set_defaults(
        exp_name=(
            "data_ensemble_muzero/"
            "mountain_car_corrected_rnd_mixed_reward_tree_seed0"
        ),
        max_env_step=45000,
        max_train_iter=15000,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    overrides = build_diagnostic_preset_overrides()
    overrides.update(
        # One classic mean-value tree. Its PUCT Q is jointly normalized after
        # combining the separately backed-up extrinsic and intrinsic returns.
        ensemble_search_mode="mean_shared_tree",
        ensemble_ucb_beta=0.0,
        mixed_reward_tree=True,
        rnd_intrinsic_reward_weight=max(0.0, float(args.mixed_reward_beta)),
        separate_intrinsic_tree=False,
        collection_policy_mode="normal",
        main_tree_policy_prior_mode="learned",
        main_tree_force_root_action_coverage=True,
        # Every trajectory follows the mixed tree, so both component values
        # are policy-evaluated on all replay rather than a pure-episode subset.
        intrinsic_value_train_on_pure_exploration_only=False,
        intrinsic_tree_pure_exploration_episodes=False,
        pure_exploration_episode_fraction=0.0,
        # Intrinsic predictions are queried on recurrent states and therefore
        # continue to shape the representation and dynamics networks.
        intrinsic_head_only=False,
        intrinsic_learning_rate=1e-4,
        intrinsic_tree_immediate_reward_weight=1.0,
        intrinsic_tree_leaf_value_source="default",
        intrinsic_tree_leaf_value_weight=1.0,
        intrinsic_tree_leaf_value_clip=None,
        main_tree_trace_diagnostics=True,
        main_tree_trace_min_train_iter=0,
        main_tree_trace_max_per_train_iter=8,
        main_tree_trace_max_depth=6,
        main_tree_trace_max_nodes=192,
        main_tree_trace_min_q_std=0.0,
    )
    run(args, preset_overrides=overrides)


if __name__ == "__main__":
    main()
