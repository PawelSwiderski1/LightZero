import argparse

from run_mountain_car_ensemble import build_config, build_parser, clean_video_dir, print_run_summary


def build_wrapper_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run MountainCar with optimistic mean-Q plus ensemble-Q-std shared-tree exploration."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--exp-name",
        type=str,
        default="data_ensemble_muzero/mountain_car_optimistic_shared_tree_seed0",
    )
    parser.add_argument("--max-env-step", type=int, default=120000)
    parser.add_argument("--max-train-iter", type=int, default=40000)
    parser.add_argument("--ensemble-ucb-beta", type=float, default=2.0)
    parser.add_argument("--num-simulations", type=int, default=25)
    parser.add_argument("--replay-ratio", type=float, default=0.25)
    parser.add_argument("--root-dirichlet-alpha", type=float, default=1.0)
    parser.add_argument("--root-noise-weight", type=float, default=0.25)
    parser.add_argument("--state-norm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--representation-last-linear-layer-init-zero",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--value-last-linear-layer-init-zero",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--norm-type", choices=["BN", "LN"], default="BN")
    parser.add_argument("--ssl-norm-type", choices=["BN", "LN"], default=None)
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--record-collect-video", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--clean-collect-video-dir", action="store_true")
    parser.add_argument("--collect-video-dir", type=str, default=None)
    parser.add_argument("--main-tree-trace-diagnostics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--main-tree-trace-min-q-std", type=float, default=2.0)
    parser.add_argument("--main-tree-trace-max-per-train-iter", type=int, default=8)
    parser.add_argument("--main-tree-trace-max-depth", type=int, default=8)
    parser.add_argument("--main-tree-trace-max-nodes", type=int, default=256)
    return parser


def run(wrapper_args: argparse.Namespace, preset_overrides=None) -> None:
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
        ensemble_ucb_beta=wrapper_args.ensemble_ucb_beta,
        root_dirichlet_alpha=wrapper_args.root_dirichlet_alpha,
        root_noise_weight=wrapper_args.root_noise_weight,
        ensemble_normalize_q=True,
        ensemble_uncertainty_decay=False,
        ensemble_eval_search_mode="pessimistic_shared_tree",
        ensemble_eval_ucb_beta=2.0,
        num_simulations=wrapper_args.num_simulations,
        num_unroll_steps=5,
        random_prior_scale=0.0,
        state_norm=wrapper_args.state_norm,
        representation_last_linear_layer_init_zero=(
            wrapper_args.representation_last_linear_layer_init_zero
        ),
        value_last_linear_layer_init_zero=wrapper_args.value_last_linear_layer_init_zero,
        norm_type=wrapper_args.norm_type,
        ssl_norm_type=wrapper_args.ssl_norm_type,
        rnd_exploration=False,
        rnd_intrinsic_reward_weight=0.0,
        rnd_store_collection_rewards=False,
        rnd_use_stored_collection_rewards=False,
        qstd_exploration=False,
        qstd_intrinsic_reward_weight=0.0,
        qstd_store_collection_rewards=False,
        separate_intrinsic_tree=False,
        separate_exploration_tree=False,
        pure_exploration_episode_fraction=0.0,
        intrinsic_episode_replay=False,
        save_intrinsic_episode_trajectories=False,
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
        main_tree_trace_min_q_std=wrapper_args.main_tree_trace_min_q_std,
        main_tree_trace_max_per_train_iter=wrapper_args.main_tree_trace_max_per_train_iter,
        main_tree_trace_max_depth=wrapper_args.main_tree_trace_max_depth,
        main_tree_trace_max_nodes=wrapper_args.main_tree_trace_max_nodes,
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
    if preset_overrides is not None:
        preset.update(preset_overrides)
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


def main() -> None:
    run(build_wrapper_parser().parse_args())


if __name__ == "__main__":
    main()
