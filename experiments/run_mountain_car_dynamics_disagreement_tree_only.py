import argparse

from run_mountain_car_ensemble import build_config, build_parser, clean_video_dir, print_run_summary


def build_wrapper_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run MountainCar with collection actions selected only by dynamics-disagreement novelty."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--exp-name",
        type=str,
        default="data_ensemble_muzero/mountain_car_dynamics_disagreement_tree_only_seed0",
    )
    parser.add_argument("--max-env-step", type=int, default=100000)
    parser.add_argument("--max-train-iter", type=int, default=3500)
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--record-collect-video", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--clean-collect-video-dir", action="store_true")
    parser.add_argument("--collect-video-dir", type=str, default=None)
    parser.add_argument("--dynamics-random-prior-scale", type=float, default=0.0)
    parser.add_argument(
        "--dynamics-random-prior-action-mode",
        choices=["one_hot", "per_action"],
        default="per_action",
    )
    parser.add_argument(
        "--dynamics-encoder-latent-normalization",
        choices=["none", "l2"],
        default="l2",
    )
    parser.add_argument("--dynamics-encoder-latent-norm-scale", type=float, default=1.0)
    parser.add_argument(
        "--dynamics-trainable-latent-normalization",
        choices=["none", "l2", "rms", "l2_cap"],
        default="l2_cap",
    )
    parser.add_argument("--dynamics-trainable-latent-norm-scale", type=float, default=1.5)
    parser.add_argument("--dynamics-trainable-latent-norm-eps", type=float, default=1e-3)
    parser.add_argument("--dynamics-prior-latent-norm-cap", type=float, default=0.5)
    parser.add_argument("--dynamics-disagreement-intrinsic-reward-weight", type=float, default=1.0)
    parser.add_argument(
        "--dynamics-disagreement-space",
        choices=["latent", "projection", "prediction_projection"],
        default="projection",
    )
    parser.add_argument(
        "--dynamics-disagreement-distance-metric",
        choices=["mse", "l2"],
        default="l2",
    )
    parser.add_argument("--dynamics-prior-alignment-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--dynamics-prior-alignment-target-normalization",
        choices=["same_as_trainable", "none", "l2", "rms"],
        default="none",
    )
    parser.add_argument("--ssl-loss-weight", type=float, default=2.0)
    parser.add_argument("--rnd-learning-rate", type=float, default=1e-5)
    parser.add_argument("--rnd-predictor-update-proportion", type=float, default=0.25)
    parser.add_argument(
        "--rnd-bonus-normalization",
        choices=["zscore", "raw"],
        default="raw",
    )
    parser.add_argument("--rnd-bonus-clip", type=float, default=5.0)
    parser.add_argument("--rnd-intrinsic-td-steps", type=int, default=20)
    parser.add_argument("--intrinsic-learning-rate", type=float, default=3e-4)
    parser.add_argument("--intrinsic-value-target-clip", type=float, default=50.0)
    parser.add_argument("--intrinsic-tree-immediate-reward-weight", type=float, default=1.0)
    parser.add_argument(
        "--intrinsic-tree-leaf-value-source",
        choices=["none", "rnd", "intrinsic_value"],
        default="intrinsic_value",
    )
    parser.add_argument("--intrinsic-tree-leaf-value-weight", type=float, default=0.25)
    parser.add_argument("--intrinsic-tree-leaf-value-clip", type=float, default=20.0)
    parser.add_argument("--collect-random-action-epsilon", type=float, default=0.0)
    parser.add_argument("--collection-ssl-diagnostics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--collection-ssl-diagnostics-batch-size", type=int, default=512)
    parser.add_argument("--intrinsic-tree-trace-diagnostics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--intrinsic-tree-trace-best-action", type=int, default=-1)
    parser.add_argument("--intrinsic-tree-trace-min-train-iter", type=int, default=0)
    parser.add_argument("--intrinsic-tree-trace-max-per-train-iter", type=int, default=8)
    parser.add_argument("--intrinsic-tree-trace-max-depth", type=int, default=6)
    parser.add_argument("--intrinsic-tree-trace-max-nodes", type=int, default=192)
    parser.add_argument("--intrinsic-tree-trace-min-position", type=float, default=-0.50)
    parser.add_argument("--intrinsic-tree-trace-max-position", type=float, default=None)
    parser.add_argument("--intrinsic-tree-trace-min-velocity", type=float, default=None)
    parser.add_argument("--intrinsic-tree-trace-max-velocity", type=float, default=-0.005)
    parser.add_argument("--fixed-sequence-intrinsic-max-states", type=int, default=128)
    parser.add_argument(
        "--fixed-sequence-intrinsic-sequences",
        type=str,
        default=(
            "left4:0x4;right4:2x4;noop4:1x4;"
            "left8:0x8;right8:2x8;noop8:1x8;"
            "left16:0x16;right16:2x16;noop16:1x16"
        ),
    )
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
        dynamics_random_prior_scale=wrapper_args.dynamics_random_prior_scale,
        dynamics_random_prior_hidden_dim=64,
        dynamics_random_prior_action_mode=wrapper_args.dynamics_random_prior_action_mode,
        dynamics_encoder_latent_normalization=wrapper_args.dynamics_encoder_latent_normalization,
        dynamics_encoder_latent_norm_scale=wrapper_args.dynamics_encoder_latent_norm_scale,
        dynamics_trainable_latent_normalization=wrapper_args.dynamics_trainable_latent_normalization,
        dynamics_trainable_latent_norm_scale=wrapper_args.dynamics_trainable_latent_norm_scale,
        dynamics_trainable_latent_norm_eps=wrapper_args.dynamics_trainable_latent_norm_eps,
        dynamics_prior_latent_norm_cap=wrapper_args.dynamics_prior_latent_norm_cap,
        shared_representation=True,
        shared_prediction=True,
        shared_projection=True,
        rnd_exploration=wrapper_args.intrinsic_tree_leaf_value_source == "rnd",
        rnd_intrinsic_reward_weight=0.0,
        rnd_learning_rate=wrapper_args.rnd_learning_rate,
        rnd_predictor_update_proportion=wrapper_args.rnd_predictor_update_proportion,
        rnd_train_on_new_data_only=True,
        rnd_train_observation_mode="next",
        rnd_bonus_normalization=wrapper_args.rnd_bonus_normalization,
        rnd_bonus_clip=wrapper_args.rnd_bonus_clip,
        rnd_use_learned_intrinsic_model=True,
        rnd_intrinsic_discount_factor=0.99,
        rnd_intrinsic_td_steps=wrapper_args.rnd_intrinsic_td_steps,
        rnd_intrinsic_bootstrap=True,
        rnd_target_snapshot=True,
        intrinsic_learning_rate=wrapper_args.intrinsic_learning_rate,
        intrinsic_reward_loss_weight=0.0,
        intrinsic_value_loss_weight=1.0,
        intrinsic_value_target_clip=wrapper_args.intrinsic_value_target_clip,
        intrinsic_value_bootstrap=True,
        intrinsic_reward_source="dynamics_disagreement",
        dynamics_disagreement_exploration=True,
        dynamics_disagreement_intrinsic_reward_weight=wrapper_args.dynamics_disagreement_intrinsic_reward_weight,
        dynamics_disagreement_shared_root_latent=True,
        dynamics_disagreement_normalize_latent=True,
        dynamics_disagreement_space=wrapper_args.dynamics_disagreement_space,
        dynamics_disagreement_distance_metric=wrapper_args.dynamics_disagreement_distance_metric,
        dynamics_disagreement_reward_clip=None,
        dynamics_disagreement_train_intrinsic_value=wrapper_args.intrinsic_tree_leaf_value_source == "intrinsic_value",
        dynamics_prior_alignment_loss_weight=wrapper_args.dynamics_prior_alignment_loss_weight,
        dynamics_prior_alignment_target_normalization=wrapper_args.dynamics_prior_alignment_target_normalization,
        ssl_loss_weight=wrapper_args.ssl_loss_weight,
        separate_intrinsic_tree=True,
        collection_policy_mode="intrinsic_tree_only",
        intrinsic_tree_num_simulations=25,
        intrinsic_tree_add_exploration_noise=False,
        intrinsic_tree_final_score_temperature=1.0,
        intrinsic_tree_store_final_policy=True,
        intrinsic_tree_normalize_q=True,
        intrinsic_tree_immediate_reward_weight=wrapper_args.intrinsic_tree_immediate_reward_weight,
        intrinsic_tree_leaf_value_source=wrapper_args.intrinsic_tree_leaf_value_source,
        intrinsic_tree_leaf_value_weight=wrapper_args.intrinsic_tree_leaf_value_weight,
        intrinsic_tree_leaf_value_clip=wrapper_args.intrinsic_tree_leaf_value_clip,
        collect_random_action_epsilon=wrapper_args.collect_random_action_epsilon,
        collection_ssl_diagnostics=wrapper_args.collection_ssl_diagnostics,
        collection_ssl_diagnostics_batch_size=wrapper_args.collection_ssl_diagnostics_batch_size,
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
        latent_novelty_diagnostics=False,
        td_steps=20,
        game_segment_length=200,
        reanalyze_ratio=0.25,
        reanalyze_use_search_value_target=False,
        value_target_mode="mixed_sve",
        sve_start_train_iter=1000,
        sve_recent_window=6000,
        use_priority=True,
        elite_replay=True,
        elite_buffer_size=20,
        elite_sample_ratio=0.25,
        elite_min_return=-199,
        root_action_diagnostics=True,
        fixed_sequence_intrinsic_diagnostics=True,
        fixed_sequence_intrinsic_max_states=wrapper_args.fixed_sequence_intrinsic_max_states,
        fixed_sequence_intrinsic_sequences=wrapper_args.fixed_sequence_intrinsic_sequences,
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
