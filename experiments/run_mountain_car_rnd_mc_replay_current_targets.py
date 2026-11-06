import argparse

from run_mountain_car_ensemble import build_config, build_parser, clean_video_dir, print_run_summary


def build_wrapper_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run MountainCar with RND MC intrinsic targets and novelty episode replay."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--exp-name",
        type=str,
        default="data_ensemble_muzero/mountain_car_rnd_mc_replay_current_targets_seed0",
    )
    parser.add_argument("--max-env-step", type=int, default=200000)
    parser.add_argument("--max-train-iter", type=int, default=6000)
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--record-collect-video", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--clean-collect-video-dir", action="store_true")
    parser.add_argument("--collect-video-dir", type=str, default=None)
    parser.add_argument("--root-action-diagnostics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fixed-sequence-intrinsic-diagnostics", action=argparse.BooleanOptionalAction, default=True)
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
        ensemble_search_mode="optimistic_shared_tree",
        ensemble_ucb_beta=2.0,
        ensemble_eval_search_mode="pessimistic_shared_tree",
        ensemble_eval_ucb_beta=2.0,
        num_simulations=25,
        num_unroll_steps=5,
        random_prior_scale=0.0,
        rnd_exploration=True,
        rnd_intrinsic_reward_weight=0.05,
        rnd_intrinsic_td_steps=200,
        rnd_learning_rate=1e-5,
        rnd_predictor_update_proportion=0.25,
        rnd_train_on_new_data_only=True,
        rnd_train_observation_mode="next",
        rnd_bonus_normalization="raw",
        rnd_store_collection_rewards=True,
        rnd_update_stats_on_collection=True,
        rnd_use_stored_collection_rewards=False,
        rnd_intrinsic_target_mode="monte_carlo",
        rnd_intrinsic_bootstrap=False,
        separate_intrinsic_tree=True,
        intrinsic_tree_num_simulations=25,
        intrinsic_tree_add_exploration_noise=False,
        intrinsic_tree_final_score_temperature=1.0,
        intrinsic_tree_store_final_policy=True,
        intrinsic_tree_normalize_q=True,
        td_steps=20,
        game_segment_length=200,
        reanalyze_ratio=0.25,
        reanalyze_use_search_value_target=False,
        value_target_mode="mixed_sve",
        sve_start_train_iter=1000,
        sve_recent_window=6000,
        use_priority=True,
        priority_initial_mode="bellman_max",
        priority_new_data_use_max=True,
        elite_replay=True,
        elite_buffer_size=20,
        elite_sample_ratio=0.25,
        elite_min_return=-199,
        intrinsic_episode_replay=True,
        intrinsic_episode_buffer_size=20,
        intrinsic_episode_sample_ratio=0.25,
        intrinsic_episode_min_score=0.0,
        intrinsic_episode_score_mode="sum",
        intrinsic_episode_peak_window_before=160,
        intrinsic_episode_peak_window_after=5,
        save_intrinsic_episode_trajectories=True,
        root_action_diagnostics=wrapper_args.root_action_diagnostics,
        fixed_sequence_intrinsic_diagnostics=wrapper_args.fixed_sequence_intrinsic_diagnostics,
        fixed_sequence_intrinsic_max_states=64,
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
