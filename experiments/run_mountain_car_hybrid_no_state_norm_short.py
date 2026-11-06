import argparse

from run_mountain_car_optimistic_shared_tree import build_wrapper_parser, run


def build_parser() -> argparse.ArgumentParser:
    parser = build_wrapper_parser()
    parser.description = (
        "Short MountainCar normalization control with LayerNorm in the planning model, "
        "BatchNorm in the SSL heads, and latent state_norm disabled."
    )
    parser.set_defaults(
        exp_name="data_ensemble_muzero/mountain_car_hybrid_no_state_norm_short_seed0",
        max_env_step=24000,
        max_train_iter=8000,
        replay_ratio=0.25,
        state_norm=False,
        norm_type="LN",
        ssl_norm_type="BN",
        main_tree_trace_diagnostics=True,
    )
    return parser


def main() -> None:
    run(
        build_parser().parse_args(),
        preset_overrides=dict(
            weight_decay=1e-4,
            # Match the earlier successful exploration setup: these unused
            # intrinsic heads must not alter the shared representation.
            rnd_use_learned_intrinsic_model=False,
            intrinsic_model_loss_weight=0.0,
            root_action_diagnostics=True,
            collection_ssl_diagnostics=False,
            normalization_mode_diagnostics=True,
            normalization_mode_diagnostics_depth=5,
            normalization_mode_diagnostics_sample_size=256,
            prediction_gradient_diagnostics=True,
        ),
    )


if __name__ == "__main__":
    main()
