import argparse

from run_mountain_car_ensemble import build_config, build_parser, clean_video_dir, print_run_summary


def build_wrapper_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run MountainCar with pure intrinsic-tree collection trained from state qstd rewards."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--exp-name",
        type=str,
        default="data_ensemble_muzero/mountain_car_qstd_intrinsic_tree_only_seed0",
    )
    parser.add_argument("--max-env-step", type=int, default=100000)
    parser.add_argument("--max-train-iter", type=int, default=3500)
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--record-collect-video", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--clean-collect-video-dir", action="store_true")
    parser.add_argument("--collect-video-dir", type=str, default=None)
    parser.add_argument("--qstd-intrinsic-reward-weight", type=float, default=1.0)
    parser.add_argument("--qstd-reward-scale", type=float, default=0.1)
    parser.add_argument("--qstd-reward-clip", type=float, default=None)
    parser.add_argument("--qstd-reward-num-simulations", type=int, default=25)
    parser.add_argument(
        "--qstd-reward-aggregation",
        choices=["max", "mean", "mean_topk"],
        default="max",
    )
    parser.add_argument("--qstd-reward-top-k", type=int, default=1)
    parser.add_argument(
        "--qstd-reward-search-mode",
        choices=[
            "mean_shared_tree", "optimistic_shared_tree", "pessimistic_shared_tree",
            "bounded_optimistic_shared_tree", "bounded_pessimistic_shared_tree",
        ],
        default="mean_shared_tree",
    )
    parser.add_argument("--intrinsic-tree-num-simulations", type=int, default=25)
    parser.add_argument("--collect-random-action-epsilon", type=float, default=0.0)
    parser.add_argument("--intrinsic-tree-trace-diagnostics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--intrinsic-tree-trace-best-action", type=int, default=-1)
    parser.add_argument("--intrinsic-tree-trace-min-train-iter", type=int, default=0)
    parser.add_argument("--intrinsic-tree-trace-max-per-train-iter", type=int, default=8)
    parser.add_argument("--intrinsic-tree-trace-max-depth", type=int, default=6)
    parser.add_argument("--intrinsic-tree-trace-max-nodes", type=int, default=192)
    parser.add_argument("--intrinsic-tree-trace-min-position", type=float, default=-0.55)
    parser.add_argument("--intrinsic-tree-trace-max-position", type=float, default=None)
    parser.add_argument("--intrinsic-tree-trace-min-velocity", type=float, default=None)
    parser.add_argument("--intrinsic-tree-trace-max-velocity", type=float, default=None)
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
        ensemble_eval_search_mode="pessimistic_shared_tree",
        ensemble_eval_ucb_beta=2.0,
        num_simulations=25,
        num_unroll_steps=5,
        random_prior_scale=0.0,
        rnd_exploration=False,
        rnd_intrinsic_reward_weight=0.0,
        rnd_store_collection_rewards=False,
        rnd_use_stored_collection_rewards=False,
        rnd_intrinsic_target_mode="monte_carlo",
        rnd_intrinsic_td_steps=200,
        rnd_intrinsic_bootstrap=False,
        rnd_intrinsic_discount_factor=0.99,
        intrinsic_reward_source="qstd",
        qstd_exploration=True,
        qstd_intrinsic_reward_weight=wrapper_args.qstd_intrinsic_reward_weight,
        qstd_store_collection_rewards=True,
        qstd_reward_aggregation=wrapper_args.qstd_reward_aggregation,
        qstd_reward_top_k=wrapper_args.qstd_reward_top_k,
        qstd_reward_scale=wrapper_args.qstd_reward_scale,
        qstd_reward_clip=wrapper_args.qstd_reward_clip,
        qstd_reward_num_simulations=wrapper_args.qstd_reward_num_simulations,
        qstd_reward_search_mode=wrapper_args.qstd_reward_search_mode,
        separate_intrinsic_tree=True,
        collection_policy_mode="intrinsic_tree_only",
        intrinsic_tree_num_simulations=wrapper_args.intrinsic_tree_num_simulations,
        intrinsic_tree_add_exploration_noise=False,
        intrinsic_tree_final_score_temperature=1.0,
        intrinsic_tree_store_final_policy=True,
        intrinsic_tree_normalize_q=True,
        intrinsic_tree_leaf_value_source="default",
        intrinsic_tree_leaf_value_weight=1.0,
        collect_random_action_epsilon=wrapper_args.collect_random_action_epsilon,
        intrinsic_tree_trace_diagnostics=wrapper_args.intrinsic_tree_trace_diagnostics,
        intrinsic_tree_trace_best_action=wrapper_args.intrinsic_tree_trace_best_action,
        intrinsic_tree_trace_min_train_iter=wrapper_args.intrinsic_tree_trace_min_train_iter,
        intrinsic_tree_trace_max_per_train_iter=wrapper_args.intrinsic_tree_trace_max_per_train_iter,
        intrinsic_tree_trace_max_depth=wrapper_args.intrinsic_tree_trace_max_depth,
        intrinsic_tree_trace_max_nodes=wrapper_args.intrinsic_tree_trace_max_nodes,
        intrinsic_tree_trace_min_position=wrapper_args.intrinsic_tree_trace_min_position,
        intrinsic_tree_trace_max_position=wrapper_args.intrinsic_tree_trace_max_position,
        intrinsic_tree_trace_min_velocity=wrapper_args.intrinsic_tree_trace_min_velocity,
        intrinsic_tree_trace_max_velocity=wrapper_args.intrinsic_tree_trace_max_velocity,
        fixed_sequence_intrinsic_diagnostics=True,
        fixed_sequence_intrinsic_max_states=64,
        collection_ssl_diagnostics=True,
        collection_ssl_diagnostics_batch_size=512,
        latent_novelty_diagnostics=False,
        update_per_collect=None,
        replay_ratio=0.25,
        td_steps=20,
        game_segment_length=200,
        reanalyze_ratio=0.25,
        reanalyze_use_search_value_target=False,
        value_target_mode="td",
        use_priority=True,
        priority_initial_mode="bellman_max",
        priority_new_data_use_max=True,
        elite_replay=False,
        intrinsic_episode_replay=False,
        save_intrinsic_episode_trajectories=True,
        intrinsic_episode_save_dir=None,
        root_action_diagnostics=True,
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
