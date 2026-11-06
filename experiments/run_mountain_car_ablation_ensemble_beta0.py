import argparse

from run_mountain_car_ensemble import build_config, build_parser, clean_video_dir, print_run_summary


def build_wrapper_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the current best MountainCar config with ensemble capacity but no disagreement bonus."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exp-name", type=str, default="data_ensemble_muzero/mountain_car_ablation_ensemble_beta0_seed0")
    parser.add_argument("--max-env-step", type=int, default=120000)
    parser.add_argument("--max-train-iter", type=int, default=6000)
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--record-collect-video", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--clean-collect-video-dir", action="store_true")
    parser.add_argument("--collect-video-dir", type=str, default=None)
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
        ensemble_ucb_beta=0.0,
        ensemble_eval_search_mode="mean_shared_tree",
        ensemble_eval_ucb_beta=0.0,
        num_simulations=25,
        num_unroll_steps=5,
        random_prior_scale=0.0,
        rnd_exploration=True,
        rnd_intrinsic_reward_weight=0.05,
        rnd_intrinsic_td_steps=20,
        rnd_learning_rate=1e-5,
        rnd_predictor_update_proportion=0.25,
        rnd_train_on_new_data_only=True,
        rnd_train_observation_mode="next",
        td_steps=20,
        game_segment_length=200,
        reanalyze_ratio=0.25,
        reanalyze_use_search_value_target=True,
        use_priority=True,
        elite_replay=True,
        elite_buffer_size=20,
        elite_sample_ratio=0.25,
        elite_min_return=-199,
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
