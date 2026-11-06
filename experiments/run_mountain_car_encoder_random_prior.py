import argparse

from run_mountain_car_hybrid_vicreg_short import (
    build_parser as build_hybrid_parser,
    build_preset_overrides as build_hybrid_preset_overrides,
)
from run_mountain_car_optimistic_shared_tree import run


def build_parser() -> argparse.ArgumentParser:
    parser = build_hybrid_parser()
    parser.description = (
        "MountainCar ensemble with fixed-ratio encoder priors, residual value priors, "
        "and optimistic Qstd collection."
    )
    parser.add_argument("--encoder-random-prior-scale", type=float, default=1.0)
    parser.add_argument(
        "--encoder-random-prior-mode",
        choices=["smooth_mlp", "local_grid"],
        default="local_grid",
    )
    parser.add_argument("--encoder-random-prior-hidden-dim", type=int, default=64)
    parser.add_argument("--encoder-random-prior-cell-size", type=float, default=0.25)
    parser.add_argument("--encoder-random-prior-hash-size", type=int, default=16384)
    parser.add_argument("--encoder-random-prior-output-norm", type=float, default=2.0)
    parser.add_argument(
        "--encoder-prior-fixed-norm-composition",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--encoder-prior-learned-norm", type=float, default=8.0)
    parser.add_argument("--encoder-prior-final-norm", type=float, default=8.0)
    parser.add_argument("--imagined-novelty-root-count", type=int, default=32)
    parser.add_argument("--imagined-novelty-bank-size", type=int, default=2048)
    parser.add_argument("--imagined-novelty-depth", type=int, default=5)
    parser.add_argument("--raw-latent-ssl-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--use-qstd-for-collection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use optimistic ensemble Qstd during collection instead of measuring it passively.",
    )
    parser.set_defaults(
        exp_name="data_ensemble_muzero/mountain_car_fixed_norm_local_encoder_value_residual_prior_seed0",
        max_env_step=45000,
        max_train_iter=15000,
        replay_ratio=0.25,
        ensemble_ucb_beta=2.0,
        value_last_linear_layer_init_zero=False,
        main_tree_trace_diagnostics=True,
    )
    return parser


def build_preset_overrides(args: argparse.Namespace) -> dict:
    overrides = build_hybrid_preset_overrides()
    use_qstd = bool(args.use_qstd_for_collection)
    fixed_norm_composition = bool(args.encoder_prior_fixed_norm_composition)
    learned_norm = max(1e-6, float(args.encoder_prior_learned_norm))
    final_norm = max(1e-6, float(args.encoder_prior_final_norm))
    effective_prior_norm = (
        max(0.0, float(args.encoder_random_prior_scale))
        * max(0.0, float(args.encoder_random_prior_output_norm))
    )
    if fixed_norm_composition and learned_norm <= effective_prior_norm:
        raise ValueError(
            "--encoder-prior-learned-norm must strictly exceed the effective encoder-prior norm "
            "(--encoder-random-prior-scale * --encoder-random-prior-output-norm)."
        )
    overrides.update(
        random_prior_scale=5.0,
        value_prior_backprop_to_latent=True,
        encoder_random_prior_scale=max(0.0, float(args.encoder_random_prior_scale)),
        encoder_random_prior_mode=str(args.encoder_random_prior_mode),
        encoder_random_prior_hidden_dim=max(1, int(args.encoder_random_prior_hidden_dim)),
        encoder_random_prior_cell_size=max(1e-6, float(args.encoder_random_prior_cell_size)),
        encoder_random_prior_hash_size=max(1, int(args.encoder_random_prior_hash_size)),
        encoder_random_prior_input_mean=(-0.3, 0.0),
        encoder_random_prior_input_scale=(0.9, 0.07),
        encoder_random_prior_output_norm=max(0.0, float(args.encoder_random_prior_output_norm)),
        encoder_prior_composed_latent_normalization='l2' if fixed_norm_composition else 'none',
        encoder_prior_composed_latent_norm_scale=final_norm,
        dynamics_encoder_latent_normalization='l2' if fixed_norm_composition else 'none',
        dynamics_encoder_latent_norm_scale=learned_norm,
        dynamics_trainable_latent_normalization='l2' if fixed_norm_composition else 'none',
        dynamics_trainable_latent_norm_scale=final_norm,
        ssl_loss_weight=0.0,
        raw_latent_ssl_loss_weight=max(0.0, float(args.raw_latent_ssl_loss_weight)),
        dynamics_random_prior_scale=0.0,
        ensemble_search_mode="optimistic_shared_tree" if use_qstd else "mean_shared_tree",
        ensemble_ucb_beta=float(args.ensemble_ucb_beta) if use_qstd else 0.0,
        ensemble_eval_search_mode="mean_shared_tree",
        ensemble_eval_ucb_beta=0.0,
        ensemble_value_calibration=True,
        ensemble_value_calibration_start_phase=1,
        ensemble_value_calibration_sample_size=4096,
        root_action_diagnostics=True,
        collection_root_action_diagnostics=True,
        encoder_prior_grid_diagnostics=True,
        encoder_prior_grid_position_min=-1.2,
        encoder_prior_grid_position_max=0.6,
        encoder_prior_grid_position_count=37,
        encoder_prior_grid_velocity_min=-0.07,
        encoder_prior_grid_velocity_max=0.07,
        encoder_prior_grid_velocity_count=29,
        encoder_prior_imagined_novelty_diagnostics=True,
        encoder_prior_imagined_novelty_root_count=max(1, int(args.imagined_novelty_root_count)),
        encoder_prior_imagined_novelty_bank_size=max(32, int(args.imagined_novelty_bank_size)),
        encoder_prior_imagined_novelty_history_limit=100000,
        encoder_prior_imagined_novelty_min_bank_size=32,
        encoder_prior_imagined_novelty_depth=max(1, int(args.imagined_novelty_depth)),
        encoder_prior_imagined_novelty_max_sequences=0,
        encoder_prior_imagined_novelty_knn_k=10,
        encoder_prior_imagined_novelty_batch_size=512,
        encoder_prior_imagined_novelty_feature_std_eps=1e-3,
        encoder_prior_imagined_novelty_position_scale=0.05,
        encoder_prior_imagined_novelty_velocity_scale=0.005,
        main_tree_trace_diagnostics=True,
        main_tree_trace_min_q_std=0.0,
        main_tree_trace_max_per_train_iter=8,
        main_tree_trace_max_depth=6,
        main_tree_trace_max_nodes=192,
    )
    return overrides


def main() -> None:
    args = build_parser().parse_args()
    run(args, preset_overrides=build_preset_overrides(args))


if __name__ == "__main__":
    main()
