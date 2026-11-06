import argparse

from run_mountain_car_optimistic_shared_tree import build_wrapper_parser, run


def main() -> None:
    parser: argparse.ArgumentParser = build_wrapper_parser()
    parser.description = (
        "Train state-value-disagreement intrinsic reward/value heads without using them for collection or MCTS."
    )
    parser.set_defaults(
        exp_name="data_ensemble_muzero/mountain_car_state_value_disagreement_diagnostic_seed0",
    )
    args = parser.parse_args()
    run(
        args,
        preset_overrides=dict(
            intrinsic_reward_source="state_value_disagreement",
            state_value_disagreement_reward_scale=1.0,
            state_value_disagreement_reward_clip=None,
            state_value_disagreement_inference_batch_size=1024,
            state_value_disagreement_store_collection_rewards=True,
            state_value_disagreement_use_stored_collection_rewards=True,
            state_value_disagreement_intrinsic_target_mode="td",
            rnd_exploration=False,
            rnd_intrinsic_reward_weight=0.0,
            rnd_use_learned_intrinsic_model=True,
            rnd_intrinsic_discount_factor=0.99,
            rnd_intrinsic_td_steps=20,
            rnd_intrinsic_bootstrap=True,
            qstd_exploration=False,
            qstd_intrinsic_reward_weight=0.0,
            dynamics_disagreement_exploration=False,
            dynamics_disagreement_intrinsic_reward_weight=0.0,
            dynamics_disagreement_train_intrinsic_value=False,
            separate_intrinsic_tree=False,
            intrinsic_model_loss_weight=1.0,
            intrinsic_reward_loss_weight=1.0,
            intrinsic_value_loss_weight=1.0,
            intrinsic_learning_rate=3e-4,
            intrinsic_head_only=True,
            intrinsic_loss_type="huber",
            intrinsic_huber_delta=1.0,
            intrinsic_reward_target_clip=0.0,
            intrinsic_value_target_clip=0.0,
            intrinsic_episode_replay=False,
            save_intrinsic_episode_trajectories=False,
            ensemble_value_calibration=False,
        ),
    )


if __name__ == "__main__":
    main()
