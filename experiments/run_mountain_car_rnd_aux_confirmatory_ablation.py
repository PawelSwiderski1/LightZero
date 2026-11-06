"""Pre-registered controls for the latent RND auxiliary-decoder effect.

All conditions retain private, member-specific raw-RND reward heads and use
those heads in MCTS unless the condition name explicitly says otherwise.

Predeclared comparisons (plan version 1):
    joint_train_only - base:
        total effect of the winning auxiliary branch.
    joint_train_only - detached_train_only:
        effect of auxiliary gradients into representation and dynamics.
    joint_train_only - ordinary_matched_train_only:
        effect of the large landscape-reader architecture.
    ordinary_matched_train_only - ordinary_exact_train_only:
        effect of the ordinary head's target-reference/LR/weight-decay/zero-init bundle.
    joint_train_only - joint_tree:
        effect of deploying the auxiliary prediction as the MCTS edge reward.
    joint_tree - joint_tree_no_private_reward:
        effect of retaining the private raw reward-head training objective.
"""

import argparse
from typing import Dict

from run_mountain_car_optimistic_shared_tree import run
from run_mountain_car_rnd_landscape_query import (
    build_parser as build_query_parser,
    build_preset_overrides as build_query_overrides,
)


PLAN_VERSION = 1
CONDITIONS = (
    "base",
    "joint_train_only",
    "detached_train_only",
    "ordinary_matched_train_only",
    "ordinary_exact_train_only",
    "joint_tree",
    "joint_tree_no_private_reward",
)


def build_parser() -> argparse.ArgumentParser:
    parser = build_query_parser()
    parser.description = (
        "Pre-registered MountainCar controls that isolate why the shared log-RND "
        "auxiliary decoder improved pure intrinsic exploration."
    )
    parser.add_argument(
        "--confirmatory-condition", choices=CONDITIONS, required=True
    )
    parser.set_defaults(
        exp_name="data_ensemble_muzero/mountain_car_rnd_aux_confirmatory_seed0",
        max_env_step=45000,
        max_train_iter=15000,
    )
    return parser


def _canonical_overrides(args: argparse.Namespace) -> Dict[str, object]:
    overrides = build_query_overrides(args)
    overrides.update(
        # Invariants shared by every condition.
        shared_intrinsic_reward_head=False,
        intrinsic_reward_target_transform="raw",
        intrinsic_reward_loss_weight=1.0,
        rnd_landscape_query=False,
        rnd_landscape_query_reader_use_context=False,
        rnd_landscape_query_member_specific_reader=False,
        rnd_landscape_query_reader_architecture="landscape",
        rnd_landscape_query_reader_hidden_dim=32,
        rnd_landscape_query_reader_norm_type="BN",
        rnd_landscape_query_reader_last_linear_layer_init_zero=False,
        rnd_landscape_query_target_transform="log",
        rnd_landscape_query_target_log_reference=1.0,
        rnd_landscape_query_error_epsilon=1e-10,
        rnd_landscape_query_max_raw_error=None,
        rnd_landscape_query_learning_rate=3e-4,
        rnd_landscape_query_weight_decay=0.0,
        rnd_landscape_query_loss_weight=1.0,
        rnd_landscape_query_train_world_model=True,
        rnd_landscape_query_use_tree_rewards=False,
        rnd_landscape_query_diversity_weight=0.0,
        rnd_landscape_query_centered_prediction_weight=0.0,
        rnd_landscape_query_gradient_diagnostics_interval=100,
        # This seed is consumed inside torch.random.fork_rng, so constructing an
        # auxiliary decoder cannot change MuZero or RND initialization.
        rnd_landscape_query_initialization_seed=200000 + int(args.seed),
        rnd_aux_confirmatory_condition=str(args.confirmatory_condition),
        rnd_aux_confirmatory_plan_version=PLAN_VERSION,
    )
    return overrides


def build_preset_overrides(args: argparse.Namespace) -> Dict[str, object]:
    overrides = _canonical_overrides(args)
    condition = str(args.confirmatory_condition)

    if condition == "base":
        return overrides

    overrides["rnd_landscape_query"] = True
    if condition == "detached_train_only":
        overrides["rnd_landscape_query_train_world_model"] = False
    elif condition == "ordinary_matched_train_only":
        # Same log target, LR, and random output initialization as the winning
        # reader; only its decoder architecture is changed.
        overrides["rnd_landscape_query_reader_architecture"] = "ordinary"
    elif condition == "ordinary_exact_train_only":
        # Exact ordinary reward-head architecture and optimization bundle, but
        # auxiliary-only: private raw heads remain operational in MCTS.
        overrides.update(
            rnd_landscape_query_reader_architecture="ordinary",
            rnd_landscape_query_reader_last_linear_layer_init_zero=True,
            rnd_landscape_query_target_log_reference=1e-2,
            rnd_landscape_query_error_epsilon=1e-8,
            rnd_landscape_query_max_raw_error=5.0,
            rnd_landscape_query_learning_rate=1e-4,
            rnd_landscape_query_weight_decay=1e-4,
        )
    elif condition == "joint_tree":
        overrides["rnd_landscape_query_use_tree_rewards"] = True
    elif condition == "joint_tree_no_private_reward":
        overrides["rnd_landscape_query_use_tree_rewards"] = True
        overrides["intrinsic_reward_loss_weight"] = 0.0
    elif condition != "joint_train_only":
        raise ValueError(f"Unsupported confirmatory condition: {condition}")

    return overrides


def main() -> None:
    args = build_parser().parse_args()
    print(
        f"RND auxiliary confirmatory plan v{PLAN_VERSION}: "
        f"condition={args.confirmatory_condition}"
    )
    run(args, preset_overrides=build_preset_overrides(args))


if __name__ == "__main__":
    main()
