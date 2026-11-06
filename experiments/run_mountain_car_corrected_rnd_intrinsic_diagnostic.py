import argparse

from run_mountain_car_hybrid_vicreg_random_value_init_mixed_sve import (
    build_parser as build_mixed_sve_parser,
)
from run_mountain_car_hybrid_vicreg_short import (
    build_preset_overrides as build_hybrid_preset_overrides,
)
from run_mountain_car_optimistic_shared_tree import run


def build_parser() -> argparse.ArgumentParser:
    parser = build_mixed_sve_parser()
    parser.description = (
        "Diagnostic-only corrected RND run: learn immediate intrinsic rewards and "
        "bootstrapped intrinsic values without using either signal for action selection."
    )
    parser.set_defaults(
        exp_name=(
            "data_ensemble_muzero/"
            "mountain_car_corrected_rnd_intrinsic_diagnostic_seed0"
        ),
    )
    return parser


def build_preset_overrides() -> dict:
    overrides = build_hybrid_preset_overrides()
    overrides.update(
        reanalyze_use_search_value_target=False,
        value_target_mode="mixed_sve",
        sve_start_train_iter=10000,
        sve_recent_window=20000,
        sve_search_noise=False,
        intrinsic_reward_source="rnd",
        # Enable RND target generation and head training, but keep it completely
        # disconnected from both collection and evaluation action selection.
        rnd_exploration=True,
        rnd_intrinsic_reward_weight=0.0,
        rnd_use_learned_intrinsic_model=True,
        separate_intrinsic_tree=False,
        qstd_exploration=False,
        qstd_intrinsic_reward_weight=0.0,
        dynamics_disagreement_exploration=False,
        dynamics_disagreement_intrinsic_reward_weight=0.0,
        dynamics_disagreement_train_intrinsic_value=False,
        # Corrected MountainCar RND signal established by the standalone sweep.
        rnd_observation_dim=2,
        rnd_observation_low="-1.2,-0.07",
        rnd_observation_high="0.6,0.07",
        rnd_learning_rate=1e-3,
        rnd_hidden_dim=128,
        rnd_feature_dim=64,
        rnd_train_batch_size=512,
        rnd_predictor_update_proportion=0.25,
        rnd_train_on_new_data_only=True,
        rnd_train_from_fresh_collection_only=True,
        rnd_fresh_update_batch_size=64,
        rnd_train_observation_mode="next",
        rnd_ignore_elite_replay_for_predictor=True,
        rnd_bonus_normalization="raw",
        rnd_bonus_clip=5.0,
        rnd_store_collection_rewards=True,
        rnd_update_stats_on_collection=False,
        # Recompute replay rewards with one current, phase-frozen novelty model.
        rnd_use_stored_collection_rewards=False,
        rnd_target_snapshot=True,
        rnd_target_snapshot_refresh_per_collect=True,
        rnd_intrinsic_target_mode="td",
        rnd_intrinsic_discount_factor=0.99,
        rnd_intrinsic_td_steps=20,
        rnd_intrinsic_bootstrap=True,
        intrinsic_value_bootstrap=True,
        # Train only the dedicated intrinsic heads; do not perturb the world model.
        intrinsic_model_loss_weight=1.0,
        intrinsic_reward_loss_weight=1.0,
        intrinsic_value_loss_weight=1.0,
        intrinsic_train_value_head=True,
        intrinsic_learning_rate=3e-4,
        intrinsic_head_only=True,
        intrinsic_loss_type="huber",
        intrinsic_huber_delta=1.0,
        intrinsic_reward_target_clip=5.0,
        intrinsic_value_target_clip=100.0,
        # Full replay already supplies older observations. This buffer adds a
        # controlled recent/high-novelty fraction for the intrinsic heads only.
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
    return overrides


def main() -> None:
    run(build_parser().parse_args(), preset_overrides=build_preset_overrides())


if __name__ == "__main__":
    main()
