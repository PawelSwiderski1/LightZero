import argparse

from run_mountain_car_hybrid_vicreg_random_value_init_mixed_sve import (
    build_parser as build_mixed_sve_parser,
)
from run_mountain_car_hybrid_vicreg_short import build_preset_overrides
from run_mountain_car_optimistic_shared_tree import run


def build_parser() -> argparse.ArgumentParser:
    parser = build_mixed_sve_parser()
    parser.description = (
        "Train diagnostic-only intrinsic reward/value heads from a phase-frozen "
        "density count over real MountainCar position and velocity."
    )
    parser.set_defaults(
        exp_name=(
            "data_ensemble_muzero/"
            "mountain_car_state_density_intrinsic_diagnostic_seed0"
        ),
    )
    return parser


def main() -> None:
    overrides = build_preset_overrides()
    overrides.update(
        reanalyze_use_search_value_target=False,
        value_target_mode="mixed_sve",
        sve_start_train_iter=10000,
        sve_recent_window=20000,
        sve_search_noise=False,
        intrinsic_reward_source="state_density",
        state_density_position_bin_width=0.02,
        state_density_velocity_bin_width=0.002,
        state_density_neighbor_radius=1,
        state_density_snapshot_update_interval=1,
        state_density_reward_scale=1.0,
        state_density_reward_clip=1.0,
        state_density_intrinsic_target_mode="td",
        state_density_train_on_intrinsic_replay_only=False,
        rnd_exploration=False,
        rnd_intrinsic_reward_weight=0.0,
        rnd_use_learned_intrinsic_model=True,
        rnd_intrinsic_discount_factor=0.99,
        # Recompute rewards for old replay observations against this phase's
        # density snapshot, then propagate them with a stable target bootstrap.
        rnd_intrinsic_td_steps=20,
        rnd_intrinsic_bootstrap=True,
        intrinsic_value_bootstrap=True,
        qstd_exploration=False,
        qstd_intrinsic_reward_weight=0.0,
        dynamics_disagreement_exploration=False,
        dynamics_disagreement_intrinsic_reward_weight=0.0,
        dynamics_disagreement_train_intrinsic_value=False,
        separate_intrinsic_tree=False,
        intrinsic_model_loss_weight=1.0,
        intrinsic_reward_loss_weight=1.0,
        intrinsic_value_loss_weight=1.0,
        intrinsic_train_value_head=True,
        intrinsic_learning_rate=3e-4,
        intrinsic_head_only=True,
        intrinsic_loss_type="huber",
        intrinsic_huber_delta=1.0,
        intrinsic_reward_target_clip=1.0,
        intrinsic_value_target_clip=100.0,
        intrinsic_episode_replay=True,
        intrinsic_episode_buffer_size=20,
        intrinsic_episode_sample_ratio=0.25,
        intrinsic_episode_min_score=0.0,
        intrinsic_episode_score_mode="sum",
        intrinsic_episode_peak_window_before=120,
        intrinsic_episode_peak_window_after=5,
        intrinsic_episode_max_age_phases=1,
        save_intrinsic_episode_trajectories=True,
        root_action_diagnostics=True,
        collection_root_action_diagnostics=True,
        fixed_sequence_intrinsic_diagnostics=True,
        fixed_sequence_intrinsic_max_states=64,
    )
    run(build_parser().parse_args(), preset_overrides=overrides)


if __name__ == "__main__":
    main()
