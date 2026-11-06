import argparse

from run_mountain_car_hybrid_vicreg_random_value_init_mixed_sve import (
    build_parser as build_mixed_sve_parser,
)
from run_mountain_car_hybrid_vicreg_short import build_preset_overrides
from run_mountain_car_optimistic_shared_tree import run


def build_parser() -> argparse.ArgumentParser:
    parser = build_mixed_sve_parser()
    parser.description = (
        "MountainCar mixed-SVE control with diagnostic-only real episodic novelty "
        "reward-head training."
    )
    parser.set_defaults(
        exp_name=(
            "data_ensemble_muzero/"
            "mountain_car_ngu_episodic_novelty_history_diagnostic_seed0"
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
        intrinsic_reward_source="episodic_novelty",
        episodic_novelty_diagnostics=True,
        episodic_novelty_encoder_member=0,
        episodic_novelty_knn_k=10,
        episodic_novelty_kernel_cluster_distance=0.008,
        episodic_novelty_kernel_epsilon=1e-4,
        episodic_novelty_pseudo_count_constant=0.001,
        episodic_novelty_maximum_similarity=8.0,
        episodic_novelty_zero_until_knn_full=True,
        episodic_novelty_reward_scale=1.0,
        episodic_novelty_reward_clip=None,
        episodic_novelty_shadow_predictions=True,
        episodic_novelty_history_shadow=True,
        episodic_novelty_history_dim=64,
        episodic_novelty_history_hidden_dim=64,
        episodic_novelty_history_learning_rate=3e-4,
        episodic_novelty_history_train_epochs=1,
        episodic_novelty_history_train_batch_size=5,
        episodic_novelty_history_updates_per_collect=60,
        episodic_novelty_history_replay_capacity=1000,
        episodic_novelty_history_recent_phase_count=2,
        episodic_novelty_history_recent_sample_ratio=0.5,
        episodic_novelty_history_grad_clip=5.0,
        intrinsic_model_loss_weight=1.0,
        intrinsic_reward_loss_weight=1.0,
        intrinsic_value_loss_weight=0.0,
        intrinsic_train_value_head=False,
        intrinsic_learning_rate=3e-4,
        intrinsic_head_only=True,
        root_action_diagnostics=True,
        collection_root_action_diagnostics=True,
        fixed_sequence_intrinsic_diagnostics=False,
    )
    run(build_parser().parse_args(), preset_overrides=overrides)


if __name__ == "__main__":
    main()
