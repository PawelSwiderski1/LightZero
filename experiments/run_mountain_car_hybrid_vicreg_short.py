import argparse

from run_mountain_car_optimistic_shared_tree import build_wrapper_parser, run


def build_parser() -> argparse.ArgumentParser:
    parser = build_wrapper_parser()
    parser.description = (
        "Short MountainCar anti-collapse ablation: LayerNorm planning, BatchNorm SSL, "
        "no latent state min-max normalization, and VICReg-style planning-latent regularization."
    )
    parser.set_defaults(
        exp_name="data_ensemble_muzero/mountain_car_hybrid_vicreg_short_seed0",
        max_env_step=24000,
        max_train_iter=8000,
        replay_ratio=0.25,
        state_norm=False,
        representation_last_linear_layer_init_zero=False,
        norm_type="LN",
        ssl_norm_type="BN",
        main_tree_trace_diagnostics=True,
    )
    return parser


def build_preset_overrides() -> dict:
    return dict(
        weight_decay=1e-4,
        representation_prediction_weight_decay=0.0,
        latent_variance_loss_weight=25.0,
        latent_covariance_loss_weight=1.0,
        # Planning latents already have per-sample LayerNorm scale. A 0.5
        # per-feature batch std target is strong without forcing every
        # coordinate to consume the full LayerNorm variance budget.
        latent_variance_target=0.5,
        latent_regularization_eps=1e-4,
        rnd_use_learned_intrinsic_model=False,
        intrinsic_model_loss_weight=0.0,
        root_action_diagnostics=True,
        depth_matched_qstd_diagnostics=True,
        depth_matched_qstd_max_depth=5,
        depth_normalized_backup_disagreement=True,
        depth_disagreement_ema_momentum=0.5,
        depth_disagreement_trim_fraction=0.1,
        depth_disagreement_min_groups=100,
        depth_disagreement_min_scale=0.5,
        depth_disagreement_max_depth=0,
        depth_disagreement_conservative_cap=True,
        collection_ssl_diagnostics=False,
        normalization_mode_diagnostics=True,
        normalization_mode_diagnostics_depth=5,
        normalization_mode_diagnostics_sample_size=256,
        prediction_gradient_diagnostics=True,
    )


def main() -> None:
    run(build_parser().parse_args(), preset_overrides=build_preset_overrides())


if __name__ == "__main__":
    main()
