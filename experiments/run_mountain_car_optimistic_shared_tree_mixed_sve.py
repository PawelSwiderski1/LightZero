import argparse

from run_mountain_car_optimistic_shared_tree import build_wrapper_parser, run


def build_parser() -> argparse.ArgumentParser:
    parser = build_wrapper_parser()
    parser.description = (
        "Run the optimistic shared-tree MountainCar preset with mixed SVE value targets."
    )
    parser.set_defaults(
        exp_name="data_ensemble_muzero/mountain_car_optimistic_shared_tree_mixed_sve_seed0",
        max_train_iter=120000,
        replay_ratio=1.0,
    )
    return parser


def main() -> None:
    run(
        build_parser().parse_args(),
        preset_overrides=dict(
            intrinsic_reward_source="rnd",
            rnd_exploration=False,
            rnd_intrinsic_reward_weight=0.0,
            rnd_use_learned_intrinsic_model=False,
            rnd_store_collection_rewards=False,
            rnd_update_stats_on_collection=False,
            qstd_exploration=False,
            qstd_intrinsic_reward_weight=0.0,
            qstd_store_collection_rewards=False,
            dynamics_disagreement_exploration=False,
            dynamics_disagreement_intrinsic_reward_weight=0.0,
            dynamics_disagreement_train_intrinsic_value=False,
            state_value_disagreement_store_collection_rewards=False,
            state_value_disagreement_use_stored_collection_rewards=False,
            store_selected_intrinsic_rewards=False,
            separate_intrinsic_tree=False,
            intrinsic_model_loss_weight=0.0,
            intrinsic_episode_replay=False,
            save_intrinsic_episode_trajectories=False,
            latent_novelty_diagnostics=False,
            reanalyze_use_search_value_target=False,
            value_target_mode="mixed_sve",
            sve_start_train_iter=40000,
            sve_recent_window=20000,
            sve_search_noise=False,
        ),
    )


if __name__ == "__main__":
    main()
