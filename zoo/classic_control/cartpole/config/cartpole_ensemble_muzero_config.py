from easydict import EasyDict

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
num_simulations = 25
update_per_collect = 100
batch_size = 256
max_env_step = int(1e5)
reanalyze_ratio = 0
ensemble_size = 1
thompson_sampling = True
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

cartpole_ensemble_muzero_config = dict(
    exp_name=(
        f'data_ensemble_muzero/cartpole_ensemble_muzero_ns{num_simulations}_upc{update_per_collect}_'
        f'rer{reanalyze_ratio}_ens{ensemble_size}_seed0'
    ),
    env=dict(
        env_id='CartPole-v0',
        stop_value=200,
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
            observation_shape=4,
            action_space_size=2,
            model_type='mlp',
            ensemble_size=ensemble_size,
            random_prior_scale=0.0,
            lstm_hidden_size=128,
            latent_state_dim=128,
            self_supervised_learning_loss=True,
            discrete_action_encoding_type='one_hot',
            norm_type='BN',
        ),
        model_path=None,
        cuda=True,
        env_type='not_board_games',
        game_segment_length=50,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='Adam',
        piecewise_decay_lr_scheduler=False,
        learning_rate=0.003,
        ssl_loss_weight=2,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=int(100),
        replay_buffer_size=int(1e6),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        thompson_sampling=thompson_sampling,
        pessimistic_search=False,
        pessimism_beta=0.0,
        ensemble_search_mode='thompson',
        ensemble_ucb_beta=0.0,
        ensemble_uncertainty_decay=False,
        ensemble_eval_search_mode='member0',
        ensemble_eval_ucb_beta=None,
        ensemble_eval_uncertainty_decay=None,
    ),
)

cartpole_ensemble_muzero_config = EasyDict(cartpole_ensemble_muzero_config)
main_config = cartpole_ensemble_muzero_config

cartpole_ensemble_muzero_create_config = dict(
    env=dict(
        type='cartpole_lightzero',
        import_names=['zoo.classic_control.cartpole.envs.cartpole_lightzero_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='ensemble_muzero',
        import_names=['lzero.policy.ensemble_muzero'],
    ),
)
cartpole_ensemble_muzero_create_config = EasyDict(cartpole_ensemble_muzero_create_config)
create_config = cartpole_ensemble_muzero_create_config

if __name__ == "__main__":
    from lzero.entry.train_ensemble_muzero import train_ensemble_muzero

    train_ensemble_muzero(
        [main_config, create_config],
        seed=0,
        model_path=main_config.policy.model_path,
        max_env_step=max_env_step
    )
