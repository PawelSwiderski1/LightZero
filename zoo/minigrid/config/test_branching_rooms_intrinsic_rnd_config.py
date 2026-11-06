from argparse import Namespace

from experiments.run_branching_rooms_intrinsic_rnd import build_config, build_parser


def _args(
    condition: str,
    intrinsic_target: str = 'bootstrapped_20step',
) -> Namespace:
    return Namespace(
        condition=condition,
        intrinsic_target=intrinsic_target,
        seed=0,
        exp_name=None,
        max_env_step=30000,
        max_train_iter=10000,
        num_simulations=25,
        collector_env_num=4,
        n_episode=4,
        batch_size=128,
        replay_ratio=0.25,
        cuda=True,
        record_collect_video=False,
        dry_run=True,
    )


def test_branching_rooms_conditions_only_toggle_auxiliary_decoder():
    baseline, baseline_create = build_config(_args('baseline'))
    auxiliary, auxiliary_create = build_config(_args('shared_log_auxiliary'))

    baseline.exp_name = 'normalized'
    auxiliary.exp_name = 'normalized'

    assert baseline_create == auxiliary_create
    assert baseline.policy.rnd_auxiliary_decoder is False
    assert auxiliary.policy.rnd_auxiliary_decoder is True
    baseline.policy.rnd_auxiliary_decoder = True
    assert baseline == auxiliary


def test_spatial_shadow_is_diagnostic_only_control():
    auxiliary, auxiliary_create = build_config(_args('shared_log_auxiliary'))
    shadow, shadow_create = build_config(
        _args('shared_log_auxiliary_spatial_shadow')
    )

    assert not auxiliary.policy.spatial_exploration_shadow
    assert shadow.policy.spatial_exploration_shadow
    assert shadow.policy.spatial_exploration_shadow_checkpoint
    assert not shadow.policy.episodic_sequence_novelty
    assert not shadow.policy.rnd_episodic_memory
    assert not shadow.policy.rnd_landscape_query
    assert shadow.policy.rnd_auxiliary_decoder
    assert auxiliary_create == shadow_create

    ignored = {
        'spatial_exploration_shadow',
        'spatial_exploration_shadow_checkpoint',
        'spatial_exploration_shadow_horizon',
        'spatial_exploration_shadow_history_window',
        'spatial_exploration_shadow_roots_per_episode',
        'spatial_exploration_shadow_root_batch_size',
        'spatial_exploration_shadow_export_dataset',
        'spatial_exploration_shadow_export_split_counts',
    }
    auxiliary_policy = {
        key: value for key, value in auxiliary.policy.items() if key not in ignored
    }
    shadow_policy = {
        key: value for key, value in shadow.policy.items() if key not in ignored
    }
    assert auxiliary_policy == shadow_policy


def test_spatial_tree_uses_frozen_checkpoint_and_configured_search_depth():
    args = build_parser().parse_args(
        [
            '--condition', 'shared_log_auxiliary_spatial_tree',
            '--spatial-exploration-checkpoint', 'spatial.pt',
            '--spatial-tree-max-depth', '1',
            '--spatial-tree-reward-mode', 'gate',
            '--spatial-tree-rnd-estimator', 'hybrid',
            '--spatial-tree-rnd-hybrid-weight', '0.75',
            '--spatial-tree-beta', '0.025',
            '--spatial-tree-minimum-reward-factor', '0.1',
        ]
    )
    config, _ = build_config(args)

    assert config.policy.rnd_auxiliary_decoder is True
    assert config.policy.spatial_exploration_tree is True
    assert config.policy.spatial_exploration_tree_checkpoint == 'spatial.pt'
    assert config.policy.spatial_exploration_tree_max_depth == 1
    assert config.policy.spatial_exploration_tree_reward_mode == 'gate'
    assert config.policy.spatial_exploration_tree_rnd_estimator == 'hybrid'
    assert config.policy.spatial_exploration_tree_rnd_hybrid_weight == 0.75
    assert config.policy.spatial_exploration_tree_beta == 0.025
    assert config.policy.spatial_exploration_tree_minimum_reward_factor == 0.1
    assert config.policy.spatial_exploration_tree_adaptive_scale is True
    assert config.policy.spatial_exploration_shadow is True


def test_online_spatial_tree_uses_belief_memory_and_rollout_reconstruction():
    args = build_parser().parse_args(
        [
            '--condition', 'shared_log_auxiliary_spatial_tree_online',
            '--spatial-tree-memory-key-mode', 'belief',
            '--spatial-tree-online-rollout-reconstruction-weight', '1.5',
            '--spatial-tree-online-observation-key-consistency-weight', '0.75',
        ]
    )
    config, _ = build_config(args)

    assert config.policy.spatial_exploration_tree_online_training is True
    assert config.policy.spatial_exploration_tree_memory_key_mode == 'belief'
    assert (
        config.policy.spatial_exploration_tree_online_rollout_reconstruction_weight
        == 1.5
    )
    assert (
        config.policy.spatial_exploration_tree_online_observation_key_consistency_weight
        == 0.75
    )


def test_sequence_continuation_value_has_a_matched_sequence_only_control():
    sequence_only, sequence_only_create = build_config(
        _args('shared_log_auxiliary_sequence_novelty')
    )
    continuation, continuation_create = build_config(
        _args('shared_log_auxiliary_sequence_novelty_value')
    )

    sequence_only.exp_name = 'normalized'
    continuation.exp_name = 'normalized'

    assert sequence_only_create == continuation_create
    assert sequence_only.policy.episodic_sequence_novelty is True
    assert continuation.policy.episodic_sequence_novelty is True
    assert sequence_only.policy.episodic_sequence_continuation_value is False
    assert continuation.policy.episodic_sequence_continuation_value is True
    assert continuation.policy.episodic_sequence_continuation_value_horizon == 20
    continuation.policy.episodic_sequence_continuation_value = False
    assert sequence_only == continuation


def test_predictive_memory_limits_joint_updates_before_predictor_catchup():
    config, _ = build_config(_args('shared_log_auxiliary_predictive_memory'))

    assert config.policy.episodic_sequence_novelty is True
    assert config.policy.episodic_sequence_novelty_model == 'predictive_memory'
    assert config.policy.episodic_predictive_memory_joint_updates_per_collect == 32
    assert config.policy.episodic_predictive_memory_distance_loss_weight == 1.0


def test_branching_rooms_compact_query_context_has_matched_no_context_control():
    query, query_create = build_config(_args('compact_query'))
    no_context, no_context_create = build_config(
        _args('compact_query_no_context')
    )

    query.exp_name = 'normalized'
    no_context.exp_name = 'normalized'

    assert query_create == no_context_create
    assert query.policy.rnd_auxiliary_decoder is True
    assert no_context.policy.rnd_auxiliary_decoder is True
    assert query.policy.rnd_landscape_query is True
    assert no_context.policy.rnd_landscape_query is True
    assert query.policy.rnd_landscape_query_reader_use_context is True
    assert no_context.policy.rnd_landscape_query_reader_use_context is False
    query.policy.rnd_landscape_query_reader_use_context = False
    assert query == no_context


def test_branching_rooms_compact_queries_use_bounded_image_space_and_no_diversity():
    config, _ = build_config(_args('compact_query'))

    assert config.policy.rnd_landscape_query_space == 'compact_image'
    assert config.policy.rnd_landscape_query_count == 8
    assert config.policy.rnd_landscape_query_code_dim == 32
    assert config.policy.rnd_landscape_query_reconstruction_loss_weight == 1.0
    assert config.policy.rnd_landscape_query_diversity_weight == 0.0
    assert config.policy.rnd_landscape_query_target_transform == 'log'
    assert config.policy.rnd_landscape_query_train_world_model is True
    assert config.policy.rnd_landscape_query_use_tree_rewards is True
    assert config.policy.rnd_intrinsic_bootstrap is True


def test_branching_rooms_uses_bootstrapped_intrinsic_td_targets():
    config, _ = build_config(_args('baseline'))

    assert config.policy.environment_coverage_cell_diagnostics is True
    assert config.policy.value_target_mode == 'td'
    assert config.policy.rnd_intrinsic_target_mode == 'td'
    assert config.policy.rnd_intrinsic_bootstrap is True
    assert config.policy.intrinsic_value_bootstrap is True
    assert config.policy.bootstrap_at_time_limit is True
    assert config.policy.intrinsic_terminal_observation_mode == 'time_limit_only'


def test_branching_rooms_supports_finite_horizon_20step_intrinsic_targets():
    config, _ = build_config(
        _args('baseline', intrinsic_target='finite_horizon_20step')
    )

    assert config.policy.value_target_mode == 'td'
    assert config.policy.rnd_intrinsic_target_mode == 'td'
    assert config.policy.rnd_intrinsic_td_steps == 20
    assert config.policy.rnd_intrinsic_bootstrap is False
    assert config.policy.intrinsic_value_bootstrap is False


def test_branching_rooms_inverse_memory_uses_dedicated_embedding_and_real_memory_only():
    config, _ = build_config(_args('compact_query_inverse_memory'))

    assert config.policy.ssl_loss_weight == 2.0
    assert config.policy.raw_latent_ssl_loss_weight == 0.0
    assert config.policy.rnd_episodic_memory is True
    assert config.policy.rnd_episodic_memory_key_space == 'inverse'
    assert config.policy.rnd_episodic_memory_mode == 'ngu_knn'
    assert config.policy.rnd_episodic_memory_include_path is False
    assert config.policy.model.episodic_inverse_embedding_dim == 32
    assert config.policy.episodic_inverse_train_world_model is True


def test_branching_rooms_local_runtime_overrides_reach_the_config():
    args = build_parser().parse_args(
        [
            '--condition', 'compact_query_inverse_memory',
            '--model-path', 'checkpoint.pth.tar',
            '--collection-only',
            '--episode-max-step', '64',
            '--num-simulations', '2',
            '--intrinsic-tree-leaf-value-weight', '0',
            '--collector-env-num', '1',
            '--n-episode', '1',
            '--batch-size', '8',
            '--replay-ratio', '0.5',
            '--rnd-fresh-update-batch-size', '16',
            '--no-cuda',
            '--no-record-collect-video',
        ]
    )

    config, _ = build_config(args)

    assert config.env.max_step == 64
    assert config.env.collector_env_num == 1
    assert config.policy.collector_env_num == 1
    assert config.policy.n_episode == 1
    assert config.policy.batch_size == 8
    assert config.policy.replay_ratio == 0.5
    assert config.policy.model_path == 'checkpoint.pth.tar'
    assert config.policy.diagnostic_collection_only is True
    assert config.policy.rnd_fresh_update_batch_size == 16
    assert config.policy.num_simulations == 2
    assert config.policy.intrinsic_tree_num_simulations == 2
    assert config.policy.intrinsic_tree_leaf_value_weight == 0.0
    assert config.policy.cuda is False
    assert config.policy.record_collect_video is False
