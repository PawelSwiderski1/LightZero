import argparse

from run_mountain_car_hybrid_vicreg_random_value_init_short import build_parser as build_short_parser
from run_mountain_car_hybrid_vicreg_short import build_preset_overrides as build_hybrid_preset_overrides
from run_mountain_car_optimistic_shared_tree import run


def build_parser() -> argparse.ArgumentParser:
    parser = build_short_parser()
    parser.description = (
        "Full MountainCar hybrid VICReg run with independently randomized ensemble value heads "
        "and mixed SVE value targets."
    )
    parser.set_defaults(
        exp_name=(
            "data_ensemble_muzero/"
            "mountain_car_hybrid_vicreg_random_value_init_mixed_sve_seed0"
        ),
        max_env_step=120000,
        max_train_iter=40000,
        replay_ratio=0.25,
    )
    return parser


def build_preset_overrides() -> dict:
    overrides = build_hybrid_preset_overrides()
    overrides.update(
        reanalyze_use_search_value_target=False,
        value_target_mode="mixed_sve",
        # At replay_ratio=0.25, 10k learner updates correspond to roughly
        # 40k newly collected transitions, matching the established warmup.
        sve_start_train_iter=10000,
        sve_recent_window=20000,
        sve_search_noise=False,
    )
    return overrides


def main() -> None:
    overrides = build_preset_overrides()
    run(build_parser().parse_args(), preset_overrides=overrides)


if __name__ == "__main__":
    main()
