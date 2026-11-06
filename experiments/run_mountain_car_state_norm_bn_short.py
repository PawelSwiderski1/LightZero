import argparse

from run_mountain_car_optimistic_shared_tree import build_wrapper_parser, run


def build_parser() -> argparse.ArgumentParser:
    parser = build_wrapper_parser()
    parser.description = "Short MountainCar normalization diagnostic with MuZero state normalization and BatchNorm."
    parser.set_defaults(
        exp_name="data_ensemble_muzero/mountain_car_state_norm_bn_short_seed0",
        max_env_step=24000,
        max_train_iter=8000,
        replay_ratio=0.25,
        state_norm=True,
        norm_type="BN",
        main_tree_trace_diagnostics=False,
    )
    return parser


def main() -> None:
    run(
        build_parser().parse_args(),
        preset_overrides=dict(
            root_action_diagnostics=False,
            collection_ssl_diagnostics=False,
            normalization_mode_diagnostics=True,
            normalization_mode_diagnostics_depth=5,
            normalization_mode_diagnostics_sample_size=256,
        ),
    )


if __name__ == "__main__":
    main()
