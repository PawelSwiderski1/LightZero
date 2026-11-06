import argparse

from run_mountain_car_ensemble import build_config, build_parser, clean_video_dir, print_run_summary


def build_wrapper_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run MountainCar with only ensemble Q-disagreement exploration."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--exp-name",
        type=str,
        default="data_ensemble_muzero/mountain_car_q_disagreement_fixed_explore_seed0",
    )
    parser.add_argument("--max-env-step", type=int, default=200000)
    parser.add_argument("--max-train-iter", type=int, default=6000)
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--record-collect-video", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--clean-collect-video-dir", action="store_true")
    parser.add_argument("--collect-video-dir", type=str, default=None)
    parser.add_argument("--root-action-diagnostics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--main-tree-trace-diagnostics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--main-tree-trace-min-q-std", type=float, default=5.0)
    parser.add_argument("--main-tree-trace-max-per-train-iter", type=int, default=4)
    parser.add_argument("--main-tree-trace-max-depth", type=int, default=6)
    parser.add_argument("--main-tree-trace-max-nodes", type=int, default=128)
    parser.add_argument("--main-tree-trace-selected-action", type=int, default=-1)
    parser.add_argument("--final-qstd-beta", type=float, default=2.0)
    parser.add_argument("--final-qstd-temperature", type=float, default=None)
    parser.add_argument("--final-qstd-store-final-policy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--separate-exploration-tree", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--separate-exploration-tree-horizon", type=int, default=3)
    parser.add_argument("--separate-exploration-tree-top-k", type=int, default=3)
    parser.add_argument("--separate-exploration-tree-beta", type=float, default=1.0)
    parser.add_argument("--separate-exploration-tree-temperature", type=float, default=None)
    parser.add_argument(
        "--separate-exploration-tree-score-normalization",
        choices=["none", "center", "zscore", "minmax"],
        default="minmax",
    )
    parser.add_argument("--separate-exploration-tree-max-sequences", type=int, default=256)
    parser.add_argument(
        "--separate-exploration-tree-store-final-policy",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--pure-exploration-episode-fraction", type=float, default=0.0)
    parser.add_argument("--pure-exploration-commit-steps", type=int, default=3)
    parser.add_argument("--pure-exploration-sequence-temperature", type=float, default=0.0)
    parser.add_argument(
        "--pure-exploration-store-exploration-policy",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--elite-replay", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--elite-buffer-size", type=int, default=20)
    parser.add_argument("--elite-sample-ratio", type=float, default=0.25)
    parser.add_argument("--elite-min-return", type=float, default=-199)
    parser.add_argument(
        "--value-target-mode",
        choices=["td", "search_bootstrap", "mixed_sve"],
        default="td",
        help="Value target mode for the main replay buffer. Use mixed_sve only when explicitly testing SVE.",
    )
    parser.add_argument("--sve-start-train-iter", type=int, default=1000)
    parser.add_argument("--sve-recent-window", type=int, default=6000)
    return parser


def main() -> None:
    wrapper_args = build_wrapper_parser().parse_args()

    args = build_parser().parse_args([])
    preset = dict(
        exp_name=wrapper_args.exp_name,
        seed=wrapper_args.seed,
        max_env_step=wrapper_args.max_env_step,
        max_train_iter=wrapper_args.max_train_iter,
        cuda=wrapper_args.cuda,
        dry_run=wrapper_args.dry_run,
        ensemble_size=3,
        ensemble_search_mode="mean_shared_tree",
        ensemble_ucb_beta=0.0,
        final_qstd_beta=wrapper_args.final_qstd_beta,
        final_qstd_temperature=wrapper_args.final_qstd_temperature,
        final_qstd_store_final_policy=wrapper_args.final_qstd_store_final_policy,
        separate_exploration_tree=wrapper_args.separate_exploration_tree,
        separate_exploration_tree_horizon=wrapper_args.separate_exploration_tree_horizon,
        separate_exploration_tree_top_k=wrapper_args.separate_exploration_tree_top_k,
        separate_exploration_tree_beta=wrapper_args.separate_exploration_tree_beta,
        separate_exploration_tree_temperature=wrapper_args.separate_exploration_tree_temperature,
        separate_exploration_tree_score_normalization=wrapper_args.separate_exploration_tree_score_normalization,
        separate_exploration_tree_max_sequences=wrapper_args.separate_exploration_tree_max_sequences,
        separate_exploration_tree_store_final_policy=wrapper_args.separate_exploration_tree_store_final_policy,
        pure_exploration_episode_fraction=wrapper_args.pure_exploration_episode_fraction,
        pure_exploration_commit_steps=wrapper_args.pure_exploration_commit_steps,
        pure_exploration_sequence_temperature=wrapper_args.pure_exploration_sequence_temperature,
        pure_exploration_store_exploration_policy=wrapper_args.pure_exploration_store_exploration_policy,
        ensemble_eval_search_mode="pessimistic_shared_tree",
        ensemble_eval_ucb_beta=2.0,
        num_simulations=25,
        num_unroll_steps=5,
        random_prior_scale=0.0,
        rnd_exploration=False,
        rnd_intrinsic_reward_weight=0.0,
        rnd_store_collection_rewards=False,
        rnd_use_stored_collection_rewards=False,
        separate_intrinsic_tree=False,
        intrinsic_episode_replay=False,
        save_intrinsic_episode_trajectories=False,
        collection_policy_mode="normal",
        dynamics_disagreement_exploration=False,
        dynamics_disagreement_intrinsic_reward_weight=0.0,
        dynamics_disagreement_train_intrinsic_value=False,
        latent_novelty_diagnostics=False,
        td_steps=20,
        game_segment_length=200,
        reanalyze_ratio=0.25,
        reanalyze_use_search_value_target=False,
        value_target_mode=wrapper_args.value_target_mode,
        sve_start_train_iter=wrapper_args.sve_start_train_iter,
        sve_recent_window=wrapper_args.sve_recent_window,
        use_priority=True,
        priority_initial_mode="bellman_max",
        priority_new_data_use_max=True,
        elite_replay=wrapper_args.elite_replay,
        elite_buffer_size=wrapper_args.elite_buffer_size,
        elite_sample_ratio=wrapper_args.elite_sample_ratio,
        elite_min_return=wrapper_args.elite_min_return,
        root_action_diagnostics=wrapper_args.root_action_diagnostics,
        main_tree_trace_diagnostics=wrapper_args.main_tree_trace_diagnostics,
        main_tree_trace_min_q_std=wrapper_args.main_tree_trace_min_q_std,
        main_tree_trace_max_per_train_iter=wrapper_args.main_tree_trace_max_per_train_iter,
        main_tree_trace_max_depth=wrapper_args.main_tree_trace_max_depth,
        main_tree_trace_max_nodes=wrapper_args.main_tree_trace_max_nodes,
        main_tree_trace_selected_action=wrapper_args.main_tree_trace_selected_action,
        fixed_sequence_intrinsic_diagnostics=False,
        collection_ssl_diagnostics=False,
        n_episode=15,
        n_evaluator_episode=16,
        stop_value=0,
        record_collect_video=wrapper_args.record_collect_video,
        clean_collect_video_dir=wrapper_args.clean_collect_video_dir,
        collect_video_dir=wrapper_args.collect_video_dir,
    )
    for key, value in preset.items():
        setattr(args, key, value)

    cfg, create_cfg = build_config(args)
    if cfg.policy.record_collect_video and args.clean_collect_video_dir:
        clean_video_dir(cfg.policy.collect_video_dir)
    print_run_summary(cfg, create_cfg, args)

    if args.dry_run:
        return

    from lzero.entry.train_ensemble_muzero import train_ensemble_muzero

    train_ensemble_muzero(
        [cfg, create_cfg],
        seed=args.seed,
        model_path=cfg.policy.model_path,
        max_env_step=args.max_env_step,
        max_train_iter=args.max_train_iter,
    )


if __name__ == "__main__":
    main()
