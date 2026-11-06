import argparse

from run_mountain_car_full_intrinsic_rnd import (
    build_parser as build_full_intrinsic_parser,
    build_preset_overrides as build_full_intrinsic_overrides,
)
from run_mountain_car_optimistic_shared_tree import run


def build_parser() -> argparse.ArgumentParser:
    parser = build_full_intrinsic_parser()
    parser.description = (
        "Pure-intrinsic MountainCar controls for the auxiliary RND-decoder result."
    )
    parser.add_argument(
        "--reward-head-ablation",
        choices=("double_loss", "log_target", "shared_head", "shared_log"),
        required=True,
    )
    parser.add_argument("--intrinsic-reward-log-reference", type=float, default=1e-2)
    parser.add_argument("--intrinsic-reward-log-epsilon", type=float, default=1e-8)
    parser.add_argument("--intrinsic-reward-log-max-raw", type=float, default=5.0)
    parser.set_defaults(
        exp_name="data_ensemble_muzero/mountain_car_intrinsic_reward_head_ablation_seed0",
        max_env_step=45000,
        max_train_iter=15000,
    )
    return parser


def build_preset_overrides(args: argparse.Namespace) -> dict:
    overrides = build_full_intrinsic_overrides(args)
    overrides.update(
        # None of these controls creates or uses the auxiliary query reader.
        rnd_landscape_query=False,
        intrinsic_reward_loss_weight=1.0,
        intrinsic_reward_target_transform="raw",
        intrinsic_reward_log_reference=max(
            1e-12, float(args.intrinsic_reward_log_reference)
        ),
        intrinsic_reward_log_epsilon=max(
            1e-12, float(args.intrinsic_reward_log_epsilon)
        ),
        intrinsic_reward_log_max_raw=max(
            float(args.intrinsic_reward_log_epsilon),
            float(args.intrinsic_reward_log_max_raw),
        ),
        shared_intrinsic_reward_head=False,
    )

    if args.reward_head_ablation == "double_loss":
        overrides["intrinsic_reward_loss_weight"] = 2.0
    elif args.reward_head_ablation == "log_target":
        overrides["intrinsic_reward_target_transform"] = "log"
    elif args.reward_head_ablation == "shared_head":
        overrides["shared_intrinsic_reward_head"] = True
    elif args.reward_head_ablation == "shared_log":
        overrides["shared_intrinsic_reward_head"] = True
        overrides["intrinsic_reward_target_transform"] = "log"
    else:
        raise ValueError(f"Unsupported reward-head ablation: {args.reward_head_ablation}")

    return overrides


def main() -> None:
    args = build_parser().parse_args()
    run(args, preset_overrides=build_preset_overrides(args))


if __name__ == "__main__":
    main()
