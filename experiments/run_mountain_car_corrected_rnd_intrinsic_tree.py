import argparse

from run_mountain_car_corrected_rnd_intrinsic_diagnostic import (
    build_parser as build_diagnostic_parser,
    build_preset_overrides as build_diagnostic_preset_overrides,
)
from run_mountain_car_optimistic_shared_tree import run


def build_parser() -> argparse.ArgumentParser:
    parser = build_diagnostic_parser()
    parser.description = (
        "MountainCar with the corrected real-observation RND model driving a "
        "separate intrinsic MCTS tree during collection."
    )
    parser.add_argument(
        "--rnd-tree-beta",
        type=float,
        default=1.0,
        help=(
            "Legacy positive enable weight for the RND tree. Episode-separated action selection "
            "does not multiply intrinsic values by this number."
        ),
    )
    parser.add_argument("--intrinsic-tree-num-simulations", type=int, default=25)
    parser.add_argument("--intrinsic-episode-fraction", type=float, default=0.2)
    parser.set_defaults(
        exp_name=(
            "data_ensemble_muzero/"
            "mountain_car_corrected_rnd_intrinsic_pure_episodes_seed0"
        ),
        max_env_step=45000,
        max_train_iter=15000,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    overrides = build_diagnostic_preset_overrides()
    overrides.update(
        rnd_intrinsic_reward_weight=args.rnd_tree_beta,
        # The intrinsic tree queries recurrent latents, so its reward/value
        # losses must also shape the representation and dynamics networks.
        # Keep this auxiliary optimizer conservative relative to the main
        # MuZero optimizer (3e-3).
        intrinsic_head_only=False,
        intrinsic_learning_rate=1e-4,
        # Reward prediction uses all replay, while value prediction learns the
        # continuation policy induced by dedicated intrinsic-tree episodes.
        intrinsic_value_train_on_pure_exploration_only=True,
        separate_intrinsic_tree=True,
        collection_policy_mode="normal",
        intrinsic_tree_num_simulations=args.intrinsic_tree_num_simulations,
        intrinsic_tree_add_exploration_noise=False,
        intrinsic_tree_prior_mode="uniform",
        intrinsic_tree_force_root_action_coverage=True,
        intrinsic_tree_pure_exploration_episodes=True,
        intrinsic_tree_pure_episode_store_intrinsic_policy=False,
        pure_exploration_episode_fraction=max(0.0, min(1.0, args.intrinsic_episode_fraction)),
        pure_exploration_commit_steps=1,
        pure_exploration_store_exploration_policy=False,
        intrinsic_tree_final_score_temperature=1.0,
        intrinsic_tree_counterfactual_betas=(0.0, 1.0, 5.0, 20.0, 50.0),
        intrinsic_tree_store_final_policy=False,
        intrinsic_tree_normalize_q=True,
        intrinsic_tree_immediate_reward_weight=1.0,
        intrinsic_tree_leaf_value_source="default",
        intrinsic_tree_leaf_value_weight=1.0,
        intrinsic_tree_leaf_value_clip=None,
        intrinsic_tree_trace_diagnostics=True,
        intrinsic_tree_trace_best_action=-1,
        intrinsic_tree_trace_min_train_iter=0,
        intrinsic_tree_trace_max_per_train_iter=4,
        intrinsic_tree_trace_max_depth=6,
        intrinsic_tree_trace_max_nodes=192,
    )
    run(args, preset_overrides=overrides)


if __name__ == "__main__":
    main()
