import argparse

from run_mountain_car_ensemble import build_config, build_parser, clean_video_dir, print_run_summary


def build_wrapper_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run MountainCar with bounded optimistic ensemble MCTS in normal episodes, exhaustive fixed-horizon "
            "bounded optimistic planning in exploration episodes, and bounded pessimistic evaluation."
        )
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--exp-name",
        type=str,
        default="data_ensemble_muzero/mountain_car_fixed_horizon_value_calibrated_seed0",
    )
    parser.add_argument("--max-env-step", type=int, default=120000)
    parser.add_argument("--max-train-iter", type=int, default=40000)
    parser.add_argument("--num-simulations", type=int, default=25)
    parser.add_argument("--fixed-horizon", type=int, default=5)
    parser.add_argument("--commit-steps", type=int, default=5)
    parser.add_argument("--exploration-episode-fraction", type=float, default=0.2)
    parser.add_argument(
        "--ensemble-search-mode",
        choices=["mean_shared_tree", "optimistic_shared_tree", "bounded_optimistic_shared_tree"],
        default="bounded_optimistic_shared_tree",
    )
    parser.add_argument(
        "--ensemble-eval-search-mode",
        choices=["mean_shared_tree", "pessimistic_shared_tree", "bounded_pessimistic_shared_tree"],
        default="bounded_pessimistic_shared_tree",
    )
    parser.add_argument("--ensemble-ucb-beta", type=float, default=2.0)
    parser.add_argument("--ensemble-eval-ucb-beta", type=float, default=2.0)
    parser.add_argument("--sequence-qstd-beta", type=float, default=2.0)
    parser.add_argument(
        "--sequence-qstd-mode",
        choices=["raw", "member_centered"],
        default="raw",
    )
    parser.add_argument("--sequence-temperature", type=float, default=0.0)
    parser.add_argument("--replay-ratio", type=float, default=0.25)
    parser.add_argument("--root-dirichlet-alpha", type=float, default=1.0)
    parser.add_argument("--root-noise-weight", type=float, default=0.25)
    parser.add_argument("--ensemble-value-calibration", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ensemble-value-calibration-start-phase", type=int, default=10)
    parser.add_argument("--ensemble-value-calibration-sample-size", type=int, default=4096)
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--record-collect-video", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--clean-collect-video-dir", action="store_true")
    parser.add_argument("--collect-video-dir", type=str, default=None)
    parser.add_argument("--main-tree-trace-diagnostics", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    wrapper_args = build_wrapper_parser().parse_args()
    if wrapper_args.fixed_horizon < 1:
        raise ValueError("--fixed-horizon must be at least 1")
    if not 0.0 <= wrapper_args.exploration_episode_fraction <= 1.0:
        raise ValueError("--exploration-episode-fraction must be in [0, 1]")

    args = build_parser().parse_args([])
    preset = dict(
        exp_name=wrapper_args.exp_name,
        seed=wrapper_args.seed,
        max_env_step=wrapper_args.max_env_step,
        max_train_iter=wrapper_args.max_train_iter,
        cuda=wrapper_args.cuda,
        dry_run=wrapper_args.dry_run,
        ensemble_size=3,
        ensemble_search_mode=wrapper_args.ensemble_search_mode,
        ensemble_ucb_beta=wrapper_args.ensemble_ucb_beta,
        root_dirichlet_alpha=wrapper_args.root_dirichlet_alpha,
        root_noise_weight=wrapper_args.root_noise_weight,
        ensemble_normalize_q=True,
        ensemble_uncertainty_decay=False,
        ensemble_value_calibration=wrapper_args.ensemble_value_calibration,
        ensemble_value_calibration_start_phase=wrapper_args.ensemble_value_calibration_start_phase,
        ensemble_value_calibration_sample_size=wrapper_args.ensemble_value_calibration_sample_size,
        ensemble_value_calibration_inference_batch_size=1024,
        ensemble_eval_search_mode=wrapper_args.ensemble_eval_search_mode,
        ensemble_eval_ucb_beta=wrapper_args.ensemble_eval_ucb_beta,
        num_simulations=wrapper_args.num_simulations,
        num_unroll_steps=5,
        random_prior_scale=0.0,
        rnd_exploration=False,
        rnd_intrinsic_reward_weight=0.0,
        rnd_store_collection_rewards=False,
        rnd_use_stored_collection_rewards=False,
        qstd_exploration=False,
        qstd_intrinsic_reward_weight=0.0,
        qstd_store_collection_rewards=False,
        separate_intrinsic_tree=False,
        separate_exploration_tree=True,
        separate_exploration_tree_horizon=wrapper_args.fixed_horizon,
        separate_exploration_tree_top_k=10,
        separate_exploration_tree_beta=0.0,
        separate_exploration_tree_temperature=None,
        separate_exploration_tree_score_normalization="none",
        separate_exploration_tree_max_sequences=0,
        separate_exploration_tree_store_final_policy=False,
        separate_exploration_tree_sequence_score_mode="bounded_mean_q_plus_qstd",
        separate_exploration_tree_sequence_qstd_beta=wrapper_args.sequence_qstd_beta,
        separate_exploration_tree_sequence_qstd_mode=wrapper_args.sequence_qstd_mode,
        pure_exploration_episode_fraction=wrapper_args.exploration_episode_fraction,
        pure_exploration_commit_steps=wrapper_args.commit_steps,
        pure_exploration_sequence_temperature=wrapper_args.sequence_temperature,
        pure_exploration_store_exploration_policy=True,
        collection_policy_mode="normal",
        dynamics_disagreement_exploration=False,
        dynamics_disagreement_intrinsic_reward_weight=0.0,
        dynamics_disagreement_train_intrinsic_value=False,
        latent_novelty_diagnostics=False,
        update_per_collect=None,
        replay_ratio=wrapper_args.replay_ratio,
        td_steps=20,
        game_segment_length=200,
        reanalyze_ratio=0.25,
        reanalyze_use_search_value_target=False,
        value_target_mode="td",
        use_priority=True,
        priority_initial_mode="bellman_max",
        priority_new_data_use_max=True,
        elite_replay=False,
        root_action_diagnostics=True,
        main_tree_trace_diagnostics=wrapper_args.main_tree_trace_diagnostics,
        main_tree_trace_min_q_std=2.0,
        main_tree_trace_max_per_train_iter=8,
        main_tree_trace_max_depth=8,
        main_tree_trace_max_nodes=256,
        main_tree_trace_selected_action=-1,
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
