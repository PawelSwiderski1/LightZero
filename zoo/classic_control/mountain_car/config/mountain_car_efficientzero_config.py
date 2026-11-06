from easydict import EasyDict

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 8
n_episode = 15
evaluator_env_num = 8
num_simulations = 25
update_per_collect = 100
batch_size = 256
max_env_step = int(2e5)
reanalyze_ratio = 0.0
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

mountain_car_efficientzero_config = dict(
    exp_name=(
        f'data_ez/mountain_car_efficientzero_ns{num_simulations}_'
        f'upc{update_per_collect}_rer{reanalyze_ratio}_seed0'
    ),
    env=dict(
        env_name='MountainCar-v0',
        stop_value=-110,
        max_episode_steps=200,
        continuous=False,
        manually_discretization=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        use_wandb=False,
        model=dict(
            observation_shape=2,
            action_space_size=3,
            model_type='mlp',
            lstm_hidden_size=128,
            latent_state_dim=64,
            self_supervised_learning_loss=True,
            discrete_action_encoding_type='one_hot',
            norm_type='BN',
        ),
        model_path=None,
        cuda=True,
        env_type='not_board_games',
        game_segment_length=200,
        td_steps=20,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='Adam',
        piecewise_decay_lr_scheduler=False,
        learning_rate=0.003,
        ssl_loss_weight=2,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=int(200),
        replay_buffer_size=int(1e6),
        use_priority=True,
        priority_prob_alpha=0.6,
        priority_prob_beta=0.4,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)

mountain_car_efficientzero_config = EasyDict(mountain_car_efficientzero_config)
main_config = mountain_car_efficientzero_config

mountain_car_efficientzero_create_config = dict(
    env=dict(
        type='mountain_car_lightzero',
        import_names=['zoo.classic_control.mountain_car.envs.mtcar_lightzero_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='efficientzero',
        import_names=['lzero.policy.efficientzero'],
    ),
)
mountain_car_efficientzero_create_config = EasyDict(mountain_car_efficientzero_create_config)
create_config = mountain_car_efficientzero_create_config

if __name__ == "__main__":
    from lzero.entry import train_muzero

    train_muzero(
        [main_config, create_config],
        seed=0,
        model_path=main_config.policy.model_path,
        max_env_step=max_env_step,
    )
