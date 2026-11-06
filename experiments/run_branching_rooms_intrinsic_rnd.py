import argparse
import copy
import json


CONDITIONS = (
    'baseline',
    'shared_log_auxiliary',
    'shared_log_auxiliary_sequence_novelty',
    'shared_log_auxiliary_sequence_novelty_value',
    'shared_log_auxiliary_recent_transformer_novelty',
    'shared_log_auxiliary_predictive_memory',
    'shared_log_auxiliary_reconstructive_memory',
    'shared_log_auxiliary_spatial_shadow',
    'shared_log_auxiliary_spatial_tree',
    'shared_log_auxiliary_spatial_tree_online',
    'compact_query',
    'compact_query_no_context',
    'compact_query_projected_memory',
    'compact_query_raw_ssl',
    'compact_query_raw_memory',
    'compact_query_inverse_memory',
)
INTRINSIC_TARGETS = ('bootstrapped_20step', 'finite_horizon_20step')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Run pure intrinsic pixel-RND MuZero in BranchingRooms, with an '
            'optional shared log-RND auxiliary decoder and compact image-space '
            'RND queries.'
        )
    )
    parser.add_argument('--condition', choices=CONDITIONS, default='baseline')
    parser.add_argument(
        '--intrinsic-target',
        choices=INTRINSIC_TARGETS,
        default='bootstrapped_20step',
        help=(
            'Use 20 discounted RND rewards with either a target-network value '
            'bootstrap or a zero continuation value.'
        ),
    )
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--exp-name', type=str, default=None)
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Load a policy checkpoint before evaluation or training.',
    )
    parser.add_argument(
        '--spatial-exploration-checkpoint',
        type=str,
        default=(
            '.local_runs/branching_rooms_spatial_exploration_world_model_'
            'macro_training_seed0/best.pt'
        ),
        help='Frozen reconstructive exploration-world-model checkpoint.',
    )
    parser.add_argument(
        '--spatial-shadow-roots-per-episode',
        type=int,
        default=128,
        help='Executed roots sampled per completed episode for shadow diagnostics.',
    )
    parser.add_argument(
        '--spatial-shadow-history-window',
        type=int,
        default=128,
        help=(
            'Maximum real-history length used to initialize frozen spatial '
            'dynamics at each root. Episodic novelty still uses the full history.'
        ),
    )
    parser.add_argument(
        '--spatial-shadow-export-dataset',
        type=str,
        default=None,
        help='Write complete collected episodes as a spatial-world-model dataset.',
    )
    parser.add_argument(
        '--spatial-shadow-export-split-counts',
        type=str,
        default=None,
        help='Comma-separated train,validation,test,heldout episode counts.',
    )
    parser.add_argument(
        '--spatial-tree-max-depth',
        type=int,
        choices=(1, 5),
        default=5,
        help='Apply frozen spatial novelty at depth one or at depths one through five.',
    )
    parser.add_argument(
        '--spatial-tree-reward-mode',
        choices=('none', 'bonus', 'gate'),
        default='bonus',
        help=(
            'Apply no episodic correction, add spatial novelty as a new reward, '
            'or use it to gate the estimated RND edge reward.'
        ),
    )
    parser.add_argument(
        '--spatial-tree-rnd-estimator',
        choices=('head', 'direct', 'hybrid'),
        default='head',
        help=(
            'Estimate imagined RND reward with the learned MuZero head, direct '
            'RND evaluation of predicted observations, or their convex hybrid.'
        ),
    )
    parser.add_argument(
        '--spatial-tree-rnd-hybrid-weight',
        type=float,
        default=0.5,
        help='Weight on direct predicted-observation RND in hybrid mode.',
    )
    parser.add_argument(
        '--spatial-tree-beta',
        type=float,
        default=0.05,
        help='Weight multiplying adaptive spatial novelty in additive bonus mode.',
    )
    parser.add_argument(
        '--spatial-tree-minimum-reward-factor',
        type=float,
        default=0.05,
        help='Smallest multiplier applied to predicted RND reward in gate mode.',
    )
    parser.add_argument(
        '--spatial-tree-online-updates-per-collect',
        type=int,
        default=16,
        help='Exploration-world-model gradient updates after each collection phase.',
    )
    parser.add_argument(
        '--spatial-tree-online-activation-updates',
        type=int,
        default=16,
        help='Keep the online gate neutral until this many model updates finish.',
    )
    parser.add_argument(
        '--spatial-tree-online-batch-size',
        type=int,
        default=2,
    )
    parser.add_argument(
        '--spatial-tree-online-roots-per-update',
        type=int,
        default=16,
    )
    parser.add_argument(
        '--spatial-tree-online-evaluation-roots-per-episode',
        type=int,
        default=64,
        help=(
            'Logged roots evaluated at depths one through five before each '
            'new episode is used to update the online exploration model.'
        ),
    )
    parser.add_argument(
        '--spatial-tree-online-replay-capacity',
        type=int,
        default=16,
        help='Number of complete recent episodes retained by the exploration model.',
    )
    parser.add_argument(
        '--spatial-tree-online-history-window',
        type=int,
        default=128,
        help=(
            'Maximum trajectory-window length used when revisit history scope '
            'is window.'
        ),
    )
    parser.add_argument(
        '--spatial-tree-online-revisit-history-scope',
        choices=('episode', 'window'),
        default='episode',
        help=(
            'Use complete episode history for belief-revisit training. The '
            'window option is retained only for explicit ablations.'
        ),
    )
    parser.add_argument(
        '--spatial-tree-online-prioritized-replay-fraction',
        type=float,
        default=0.0,
        help=(
            'Fraction of exploration-model replay windows sampled according '
            'to their latest revisit-prediction error.'
        ),
    )
    parser.add_argument(
        '--spatial-tree-online-prioritized-replay-alpha',
        type=float,
        default=0.6,
        help='Exponent applied to stored revisit-error replay priorities.',
    )
    parser.add_argument(
        '--spatial-tree-online-prioritized-replay-epsilon',
        type=float,
        default=1e-3,
        help='Positive floor added to exploration replay priorities.',
    )
    parser.add_argument(
        '--spatial-tree-online-prioritized-replay-priority-mode',
        choices=('belief_error', 'revisit_error'),
        default='belief_error',
        help=(
            'Choose whether exploration replay priority uses full-belief '
            'cosine error or the endpoint novelty-classification error.'
        ),
    )
    parser.add_argument(
        '--spatial-tree-memory-key-mode',
        choices=(
            'observation',
            'belief',
            'belief_to_observation',
            'belief_revisit',
        ),
        default='belief',
        help=(
            'Use observation keys, belief keys, projected observation keys, or '
            'an exact-RGB-supervised pairwise belief revisit classifier.'
        ),
    )
    parser.add_argument(
        '--spatial-tree-revisit-scoring-mode',
        choices=('model', 'observation_key'),
        default='model',
        help=(
            'Use the memory-key mode default scorer, or compare predicted '
            'observation keys directly with real and branch-local observation '
            'keys.'
        ),
    )
    parser.add_argument(
        '--spatial-tree-online-rollout-reconstruction-weight',
        type=float,
        default=1.0,
        help=(
            'Weight for decoding predicted exploration beliefs into their '
            'real future observations during online world-model training.'
        ),
    )
    parser.add_argument(
        '--spatial-tree-online-rollout-contrastive-weight',
        type=float,
        default=1.0,
        help='Weight for the RGB-identity rollout contrastive loss.',
    )
    parser.add_argument(
        '--spatial-tree-online-observation-key-consistency-weight',
        type=float,
        default=0.0,
        help=(
            'Weight for aligning predicted observation keys with stop-gradient '
            'keys encoded from the corresponding real future observations.'
        ),
    )
    parser.add_argument(
        '--spatial-tree-online-belief-observation-alignment-weight',
        type=float,
        default=1.0,
        help=(
            'Weight for mapping recurrent beliefs into observation-key '
            'coordinates in belief_to_observation mode.'
        ),
    )
    parser.add_argument(
        '--spatial-tree-online-belief-dynamics-consistency-weight',
        type=float,
        default=0.0,
        help=(
            'Weight for aligning predicted beliefs with stop-gradient beliefs '
            'encoded from the corresponding real future trajectory.'
        ),
    )
    parser.add_argument(
        '--spatial-tree-online-pairwise-revisit-weight',
        type=float,
        default=1.0,
        help='Weight for pairwise belief revisit classification.',
    )
    parser.add_argument(
        '--spatial-tree-online-pairwise-revisit-negative-samples',
        type=int,
        default=8,
        help='Maximum hard negative memory pairs retained per imagined query.',
    )
    parser.add_argument(
        '--spatial-tree-online-predictive-equivalence-weight',
        type=float,
        default=0.0,
        help=(
            'Weight for replay-grounded belief-key equivalence from matching '
            'executed action prefixes and their real observation outcomes.'
        ),
    )
    parser.add_argument(
        '--spatial-tree-online-predictive-equivalence-negative-margin',
        type=float,
        default=0.2,
    )
    parser.set_defaults(spatial_tree_adaptive_scale=True)
    parser.add_argument(
        '--spatial-tree-adaptive-scale',
        dest='spatial_tree_adaptive_scale',
        action='store_true',
        help='Normalize spatial novelty by real episodic-memory neighbour spacing.',
    )
    parser.add_argument(
        '--spatial-tree-raw-distance',
        dest='spatial_tree_adaptive_scale',
        action='store_false',
        help='Use raw cosine distance as the spatial novelty bonus.',
    )
    parser.add_argument(
        '--collection-only',
        action='store_true',
        help=(
            'Collect one diagnostic phase and exit before updating RND, the '
            'episodic novelty model, or MuZero.'
        ),
    )
    parser.add_argument('--max-env-step', type=int, default=30000)
    parser.add_argument('--max-train-iter', type=int, default=10000)
    parser.add_argument(
        '--episode-max-step',
        type=int,
        default=None,
        help='Override the 1200-step episode length for short local diagnostics.',
    )
    parser.add_argument('--num-simulations', type=int, default=25)
    parser.add_argument(
        '--intrinsic-tree-leaf-value-weight',
        type=float,
        default=None,
        help=(
            'Override the intrinsic leaf-value bootstrap weight. Use 0 for a '
            'search driven only by episodically gated imagined edge rewards.'
        ),
    )
    parser.add_argument('--collector-env-num', type=int, default=4)
    parser.add_argument('--n-episode', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--replay-ratio', type=float, default=0.25)
    parser.add_argument(
        '--rnd-fresh-update-batch-size',
        type=int,
        default=None,
        help=(
            'Override the fresh-RND update batch size. A smaller value is useful '
            'only for short local diagnostics.'
        ),
    )
    parser.add_argument(
        '--episodic-sequence-updates-per-collect',
        type=int,
        default=None,
        help='Override recurrent episodic-novelty updates per collection phase.',
    )
    parser.add_argument(
        '--episodic-sequence-train-batch-size',
        type=int,
        default=None,
        help='Override recurrent episodic-novelty episode batch size.',
    )
    parser.add_argument(
        '--episodic-sequence-roots-per-update',
        type=int,
        default=None,
        help='Override sampled real roots per episode and sidecar update.',
    )
    parser.add_argument(
        '--episodic-sequence-history-window',
        type=int,
        default=40,
        help='Recent observation/action tokens retained by transformer novelty.',
    )
    parser.add_argument(
        '--episodic-sequence-inference-mode',
        choices=(
            'direct',
            'cached_suffix',
            'incremental_cache',
            'prefix_table',
        ),
        default='cached_suffix',
        help=(
            'Score recent-transformer nodes directly, reuse cached real-history attention '
            'for batched suffixes, reuse branch-local attention caches, '
            'or precompute every short action prefix.'
        ),
    )
    parser.set_defaults(
        cuda=True,
        record_collect_video=False,
        predictive_memory_adaptive_distance=True,
    )
    parser.add_argument('--cuda', dest='cuda', action='store_true')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false')
    parser.add_argument(
        '--record-collect-video', dest='record_collect_video', action='store_true'
    )
    parser.add_argument(
        '--no-record-collect-video', dest='record_collect_video', action='store_false'
    )
    parser.add_argument(
        '--predictive-memory-adaptive-distance',
        dest='predictive_memory_adaptive_distance',
        action='store_true',
        help='Calibrate predictive-memory novelty from real-memory neighbour spacing.',
    )
    parser.add_argument(
        '--predictive-memory-fixed-distance',
        dest='predictive_memory_adaptive_distance',
        action='store_false',
        help='Use the legacy fixed predictive-memory distance threshold.',
    )
    parser.add_argument('--dry-run', action='store_true')
    return parser


def build_config(args: argparse.Namespace):
    from zoo.minigrid.config.branching_rooms_ensemble_muzero_config import (
        create_config,
        main_config,
    )

    cfg = copy.deepcopy(main_config)
    create_cfg = copy.deepcopy(create_config)
    cfg.exp_name = args.exp_name or (
        f'data_ensemble_muzero/branching_rooms_intrinsic_rnd_'
        f'{args.condition}_seed{args.seed}'
    )
    cfg.policy.model_path = getattr(args, 'model_path', None)
    cfg.policy.diagnostic_collection_only = bool(
        getattr(args, 'collection_only', False)
    )
    cfg.env.collector_env_num = int(args.collector_env_num)
    cfg.policy.collector_env_num = int(args.collector_env_num)
    cfg.policy.n_episode = int(args.n_episode)
    cfg.policy.batch_size = int(args.batch_size)
    cfg.policy.replay_ratio = float(args.replay_ratio)
    rnd_fresh_update_batch_size = getattr(
        args, 'rnd_fresh_update_batch_size', None
    )
    if rnd_fresh_update_batch_size is not None:
        cfg.policy.rnd_fresh_update_batch_size = int(rnd_fresh_update_batch_size)
    episode_max_step = getattr(args, 'episode_max_step', None)
    if episode_max_step is not None:
        cfg.env.max_step = int(episode_max_step)
        cfg.policy.game_segment_length = min(
            int(cfg.policy.game_segment_length), int(episode_max_step)
        )
    cfg.policy.num_simulations = int(args.num_simulations)
    cfg.policy.intrinsic_tree_num_simulations = int(args.num_simulations)
    intrinsic_tree_leaf_value_weight = getattr(
        args, 'intrinsic_tree_leaf_value_weight', None
    )
    if intrinsic_tree_leaf_value_weight is not None:
        cfg.policy.intrinsic_tree_leaf_value_weight = float(
            intrinsic_tree_leaf_value_weight
        )
    cfg.policy.cuda = bool(args.cuda)
    cfg.policy.record_collect_video = bool(args.record_collect_video)
    cfg.policy.rnd_initialization_seed = 100000 + int(args.seed)
    cfg.policy.rnd_intrinsic_target_mode = 'td'
    cfg.policy.rnd_intrinsic_td_steps = 20
    use_intrinsic_bootstrap = args.intrinsic_target == 'bootstrapped_20step'
    cfg.policy.rnd_intrinsic_bootstrap = use_intrinsic_bootstrap
    cfg.policy.intrinsic_value_bootstrap = use_intrinsic_bootstrap
    use_auxiliary = args.condition != 'baseline'
    use_compact_query = args.condition.startswith('compact_query')
    use_raw_ssl = args.condition in {
        'compact_query_raw_ssl', 'compact_query_raw_memory'
    }
    use_episodic_memory = args.condition in {
        'compact_query_projected_memory',
        'compact_query_raw_memory',
        'compact_query_inverse_memory',
    }
    use_inverse_memory = args.condition == 'compact_query_inverse_memory'
    use_sequence_novelty = args.condition in {
        'shared_log_auxiliary_sequence_novelty',
        'shared_log_auxiliary_sequence_novelty_value',
        'shared_log_auxiliary_recent_transformer_novelty',
        'shared_log_auxiliary_predictive_memory',
        'shared_log_auxiliary_reconstructive_memory',
    }
    use_sequence_continuation_value = (
        args.condition == 'shared_log_auxiliary_sequence_novelty_value'
    )
    cfg.policy.ssl_loss_weight = 0.0 if use_raw_ssl else 2.0
    cfg.policy.raw_latent_ssl_loss_weight = 1.0 if use_raw_ssl else 0.0
    cfg.policy.rnd_auxiliary_decoder = use_auxiliary
    cfg.policy.rnd_auxiliary_decoder_initialization_seed = (
        200000 + int(args.seed)
    )
    cfg.policy.rnd_auxiliary_decoder_target_transform = 'log'
    cfg.policy.rnd_auxiliary_decoder_train_world_model = True
    cfg.policy.rnd_auxiliary_decoder_loss_weight = 1.0
    cfg.policy.rnd_landscape_query = use_compact_query
    cfg.policy.rnd_landscape_query_space = 'compact_image'
    cfg.policy.rnd_landscape_query_count = 8
    cfg.policy.rnd_landscape_query_hidden_dim = 64
    cfg.policy.rnd_landscape_query_code_dim = 32
    cfg.policy.rnd_landscape_query_reconstruction_loss_weight = 1.0
    cfg.policy.rnd_landscape_query_diagnostic_root_count = 8
    cfg.policy.rnd_landscape_query_learning_rate = 3e-4
    cfg.policy.rnd_landscape_query_weight_decay = 0.0
    cfg.policy.rnd_landscape_query_loss_weight = 1.0
    cfg.policy.rnd_landscape_query_max_horizon = int(cfg.policy.num_unroll_steps)
    cfg.policy.rnd_landscape_query_min_snapshots = 2
    cfg.policy.rnd_landscape_query_diversity_weight = 0.0
    cfg.policy.rnd_landscape_query_centered_prediction_weight = 0.0
    cfg.policy.rnd_landscape_query_reader_use_context = (
        args.condition != 'compact_query_no_context'
    )
    cfg.policy.rnd_landscape_query_member_specific_reader = False
    cfg.policy.rnd_landscape_query_reader_architecture = 'landscape'
    cfg.policy.rnd_landscape_query_target_transform = 'log'
    cfg.policy.rnd_landscape_query_target_log_reference = 1.0
    cfg.policy.rnd_landscape_query_train_world_model = True
    cfg.policy.rnd_landscape_query_use_tree_rewards = True
    cfg.policy.rnd_landscape_query_initialization_seed = (
        300000 + int(args.seed)
    )
    cfg.policy.rnd_episodic_memory = use_episodic_memory
    cfg.policy.rnd_episodic_memory_key_space = (
        'raw' if args.condition == 'compact_query_raw_memory'
        else 'inverse' if use_inverse_memory
        else 'projected'
    )
    cfg.policy.rnd_episodic_memory_max_size = int(cfg.env.max_step)
    cfg.policy.rnd_episodic_memory_kernel_bandwidth = (
        1.0 if use_inverse_memory else 0.25
    )
    cfg.policy.rnd_episodic_memory_min_reward_factor = 0.05
    cfg.policy.rnd_episodic_memory_include_path = not use_inverse_memory
    cfg.policy.rnd_episodic_memory_mode = (
        'ngu_knn' if use_inverse_memory else 'minimum_distance'
    )
    cfg.policy.model.episodic_inverse_embedding_dim = 32 if use_inverse_memory else 0
    cfg.policy.model.episodic_inverse_hidden_dim = 128
    cfg.policy.episodic_inverse_loss_weight = 1.0
    cfg.policy.episodic_inverse_bridge_loss_weight = 1.0
    cfg.policy.episodic_inverse_train_world_model = True
    cfg.policy.episodic_inverse_max_horizon = int(cfg.policy.num_unroll_steps)
    cfg.policy.episodic_sequence_novelty = use_sequence_novelty
    if args.condition == 'shared_log_auxiliary_reconstructive_memory':
        cfg.policy.episodic_sequence_novelty_model = 'reconstructive_memory'
    elif args.condition == 'shared_log_auxiliary_predictive_memory':
        cfg.policy.episodic_sequence_novelty_model = 'predictive_memory'
    elif args.condition == 'shared_log_auxiliary_recent_transformer_novelty':
        cfg.policy.episodic_sequence_novelty_model = 'recent_transformer'
    else:
        cfg.policy.episodic_sequence_novelty_model = 'recurrent'
    cfg.policy.episodic_sequence_novelty_embedding_dim = 64
    cfg.policy.episodic_sequence_novelty_hidden_dim = 128
    cfg.policy.episodic_sequence_novelty_horizon = 5
    cfg.policy.episodic_sequence_novelty_history_window = int(
        getattr(args, 'episodic_sequence_history_window', 40)
    )
    cfg.policy.episodic_sequence_novelty_transformer_layers = 2
    cfg.policy.episodic_sequence_novelty_transformer_heads = 4
    cfg.policy.episodic_sequence_novelty_transformer_dropout = 0.0
    cfg.policy.episodic_sequence_novelty_inference_mode = (
        getattr(args, 'episodic_sequence_inference_mode', 'cached_suffix')
        if args.condition == 'shared_log_auxiliary_recent_transformer_novelty'
        else 'direct'
    )
    cfg.policy.episodic_sequence_novelty_discount_factor = 0.99
    cfg.policy.episodic_sequence_novelty_learning_rate = 3e-4
    cfg.policy.episodic_sequence_novelty_weight_decay = 1e-5
    cfg.policy.episodic_sequence_novelty_sequence_loss_weight = 1.0
    cfg.policy.episodic_sequence_novelty_grad_clip = 5.0
    cfg.policy.episodic_sequence_novelty_updates_per_collect = 64
    cfg.policy.episodic_sequence_novelty_train_batch_size = 1
    cfg.policy.episodic_sequence_novelty_roots_per_update = 32
    cfg.policy.episodic_predictive_memory_joint_updates_per_collect = (
        32
        if args.condition in {
            'shared_log_auxiliary_predictive_memory',
            'shared_log_auxiliary_reconstructive_memory',
        }
        else None
    )
    cfg.policy.episodic_predictive_memory_distance_loss_weight = 1.0
    cfg.policy.episodic_predictive_memory_adaptive_distance_scale = bool(
        getattr(args, 'predictive_memory_adaptive_distance', True)
    )
    cfg.policy.episodic_predictive_memory_adaptive_threshold_ratio = 0.5
    cfg.policy.episodic_predictive_memory_adaptive_temperature_ratio = 0.125
    cfg.policy.episodic_predictive_memory_adaptive_minimum_scale = 0.02
    cfg.policy.episodic_predictive_memory_adaptive_distinct_epsilon = 1e-6
    cfg.policy.episodic_reconstructive_memory_decoder_channels = 64
    cfg.policy.episodic_reconstructive_memory_reconstruction_loss_weight = 1.0
    cfg.policy.episodic_reconstructive_memory_rollout_loss_weight = 1.0
    episodic_sequence_updates_per_collect = getattr(
        args, 'episodic_sequence_updates_per_collect', None
    )
    if episodic_sequence_updates_per_collect is not None:
        cfg.policy.episodic_sequence_novelty_updates_per_collect = int(
            episodic_sequence_updates_per_collect
        )
    episodic_sequence_train_batch_size = getattr(
        args, 'episodic_sequence_train_batch_size', None
    )
    if episodic_sequence_train_batch_size is not None:
        cfg.policy.episodic_sequence_novelty_train_batch_size = int(
            episodic_sequence_train_batch_size
        )
    episodic_sequence_roots_per_update = getattr(
        args, 'episodic_sequence_roots_per_update', None
    )
    if episodic_sequence_roots_per_update is not None:
        cfg.policy.episodic_sequence_novelty_roots_per_update = int(
            episodic_sequence_roots_per_update
        )
    cfg.policy.episodic_sequence_novelty_replay_capacity = 32
    cfg.policy.episodic_sequence_novelty_minimum_reward_factor = 0.05
    cfg.policy.episodic_sequence_novelty_activation_phase = 1
    cfg.policy.episodic_sequence_novelty_observation_scale = (
        1.0 if bool(cfg.env.scale_observation) else 255.0
    )
    cfg.policy.episodic_sequence_novelty_initialization_seed = (
        400000 + int(args.seed)
    )
    cfg.policy.episodic_sequence_continuation_value = (
        use_sequence_continuation_value
    )
    cfg.policy.episodic_sequence_continuation_value_horizon = 20
    cfg.policy.episodic_sequence_continuation_loss_weight = 1.0
    use_spatial_online = (
        args.condition == 'shared_log_auxiliary_spatial_tree_online'
    )
    use_spatial_tree = args.condition in {
        'shared_log_auxiliary_spatial_tree',
        'shared_log_auxiliary_spatial_tree_online',
    }
    use_spatial_shadow = args.condition in {
        'shared_log_auxiliary_spatial_shadow',
        'shared_log_auxiliary_spatial_tree',
    }
    cfg.policy.spatial_exploration_shadow = use_spatial_shadow
    cfg.policy.spatial_exploration_shadow_checkpoint = (
        getattr(
            args,
            'spatial_exploration_checkpoint',
            (
                '.local_runs/branching_rooms_spatial_exploration_world_model_'
                'macro_training_seed0/best.pt'
            ),
        )
        if use_spatial_shadow
        else None
    )
    cfg.policy.spatial_exploration_shadow_horizon = 5
    cfg.policy.spatial_exploration_shadow_history_window = int(
        getattr(args, 'spatial_shadow_history_window', 128)
    )
    cfg.policy.spatial_exploration_shadow_roots_per_episode = int(
        getattr(args, 'spatial_shadow_roots_per_episode', 128)
    )
    cfg.policy.spatial_exploration_shadow_root_batch_size = 128
    cfg.policy.spatial_exploration_shadow_export_dataset = getattr(
        args, 'spatial_shadow_export_dataset', None
    )
    split_counts_text = getattr(
        args, 'spatial_shadow_export_split_counts', None
    )
    cfg.policy.spatial_exploration_shadow_export_split_counts = (
        tuple(int(value) for value in split_counts_text.split(','))
        if split_counts_text
        else None
    )
    if (
        cfg.policy.spatial_exploration_shadow_export_split_counts is not None
        and len(cfg.policy.spatial_exploration_shadow_export_split_counts) != 4
    ):
        raise ValueError(
            '--spatial-shadow-export-split-counts requires exactly four values'
        )
    cfg.policy.spatial_exploration_tree = use_spatial_tree
    cfg.policy.spatial_exploration_tree_checkpoint = (
        cfg.policy.spatial_exploration_shadow_checkpoint
        if use_spatial_tree and not use_spatial_online else None
    )
    cfg.policy.spatial_exploration_tree_online_training = use_spatial_online
    cfg.policy.spatial_exploration_tree_online_initialization_seed = (
        200000 + int(args.seed)
    )
    cfg.policy.spatial_exploration_tree_online_learning_rate = 1e-4
    cfg.policy.spatial_exploration_tree_online_weight_decay = 1e-5
    cfg.policy.spatial_exploration_tree_online_activation_updates = int(
        getattr(args, 'spatial_tree_online_activation_updates', 16)
    )
    cfg.policy.spatial_exploration_tree_online_replay_capacity = int(
        getattr(args, 'spatial_tree_online_replay_capacity', 16)
    )
    cfg.policy.spatial_exploration_tree_online_prioritized_replay_fraction = float(
        getattr(args, 'spatial_tree_online_prioritized_replay_fraction', 0.0)
    )
    if not 0.0 <= (
        cfg.policy.spatial_exploration_tree_online_prioritized_replay_fraction
    ) <= 1.0:
        raise ValueError(
            '--spatial-tree-online-prioritized-replay-fraction must be in [0, 1]'
        )
    cfg.policy.spatial_exploration_tree_online_prioritized_replay_alpha = float(
        getattr(args, 'spatial_tree_online_prioritized_replay_alpha', 0.6)
    )
    if cfg.policy.spatial_exploration_tree_online_prioritized_replay_alpha < 0.0:
        raise ValueError(
            '--spatial-tree-online-prioritized-replay-alpha cannot be negative'
        )
    cfg.policy.spatial_exploration_tree_online_prioritized_replay_epsilon = float(
        getattr(args, 'spatial_tree_online_prioritized_replay_epsilon', 1e-3)
    )
    if cfg.policy.spatial_exploration_tree_online_prioritized_replay_epsilon <= 0.0:
        raise ValueError(
            '--spatial-tree-online-prioritized-replay-epsilon must be positive'
        )
    cfg.policy.spatial_exploration_tree_online_prioritized_replay_priority_mode = str(
        getattr(
            args,
            'spatial_tree_online_prioritized_replay_priority_mode',
            'belief_error',
        )
    )
    cfg.policy.spatial_exploration_tree_online_revisit_history_scope = str(
        getattr(args, 'spatial_tree_online_revisit_history_scope', 'episode')
    )
    cfg.policy.spatial_exploration_tree_online_history_window = min(
        int(getattr(args, 'spatial_tree_online_history_window', 128)),
        int(cfg.env.max_step),
    )
    if cfg.policy.spatial_exploration_tree_online_history_window < 5:
        raise ValueError(
            '--spatial-tree-online-history-window must be at least 5'
        )
    cfg.policy.spatial_exploration_tree_online_batch_size = int(
        getattr(args, 'spatial_tree_online_batch_size', 2)
    )
    cfg.policy.spatial_exploration_tree_online_roots_per_update = int(
        getattr(args, 'spatial_tree_online_roots_per_update', 16)
    )
    cfg.policy.spatial_exploration_tree_online_updates_per_collect = int(
        getattr(args, 'spatial_tree_online_updates_per_collect', 16)
    )
    cfg.policy.spatial_exploration_tree_online_evaluation_roots_per_episode = int(
        getattr(
            args,
            'spatial_tree_online_evaluation_roots_per_episode',
            64,
        )
    )
    cfg.policy.spatial_exploration_tree_online_grad_clip = 5.0
    cfg.policy.spatial_exploration_tree_online_memory_novelty_weight = 10.0
    cfg.policy.spatial_exploration_tree_memory_key_mode = str(
        getattr(args, 'spatial_tree_memory_key_mode', 'belief')
    )
    cfg.policy.spatial_exploration_tree_revisit_scoring_mode = str(
        getattr(args, 'spatial_tree_revisit_scoring_mode', 'model')
    )
    cfg.policy.spatial_exploration_tree_online_rollout_reconstruction_weight = float(
        getattr(
            args,
            'spatial_tree_online_rollout_reconstruction_weight',
            1.0,
        )
    )
    if cfg.policy.spatial_exploration_tree_online_rollout_reconstruction_weight < 0.0:
        raise ValueError(
            '--spatial-tree-online-rollout-reconstruction-weight cannot be negative'
        )
    cfg.policy.spatial_exploration_tree_online_rollout_contrastive_weight = float(
        getattr(args, 'spatial_tree_online_rollout_contrastive_weight', 1.0)
    )
    if cfg.policy.spatial_exploration_tree_online_rollout_contrastive_weight < 0.0:
        raise ValueError(
            '--spatial-tree-online-rollout-contrastive-weight cannot be negative'
        )
    cfg.policy.spatial_exploration_tree_online_observation_key_consistency_weight = float(
        getattr(
            args,
            'spatial_tree_online_observation_key_consistency_weight',
            0.0,
        )
    )
    if (
        cfg.policy.spatial_exploration_tree_online_observation_key_consistency_weight
        < 0.0
    ):
        raise ValueError(
            '--spatial-tree-online-observation-key-consistency-weight cannot be negative'
        )
    cfg.policy.spatial_exploration_tree_online_belief_observation_alignment_weight = float(
        getattr(
            args,
            'spatial_tree_online_belief_observation_alignment_weight',
            1.0,
        )
    )
    if (
        cfg.policy.spatial_exploration_tree_online_belief_observation_alignment_weight
        < 0.0
    ):
        raise ValueError(
            '--spatial-tree-online-belief-observation-alignment-weight cannot '
            'be negative'
        )
    cfg.policy.spatial_exploration_tree_online_belief_dynamics_consistency_weight = float(
        getattr(
            args,
            'spatial_tree_online_belief_dynamics_consistency_weight',
            0.0,
        )
    )
    if (
        cfg.policy.spatial_exploration_tree_online_belief_dynamics_consistency_weight
        < 0.0
    ):
        raise ValueError(
            '--spatial-tree-online-belief-dynamics-consistency-weight cannot '
            'be negative'
        )
    cfg.policy.spatial_exploration_tree_online_pairwise_revisit_weight = float(
        getattr(args, 'spatial_tree_online_pairwise_revisit_weight', 1.0)
    )
    if cfg.policy.spatial_exploration_tree_online_pairwise_revisit_weight < 0.0:
        raise ValueError(
            '--spatial-tree-online-pairwise-revisit-weight cannot be negative'
        )
    cfg.policy.spatial_exploration_tree_online_pairwise_revisit_negative_samples = int(
        getattr(args, 'spatial_tree_online_pairwise_revisit_negative_samples', 8)
    )
    if (
        cfg.policy.spatial_exploration_tree_online_pairwise_revisit_negative_samples
        <= 0
    ):
        raise ValueError(
            '--spatial-tree-online-pairwise-revisit-negative-samples must be positive'
        )
    cfg.policy.spatial_exploration_tree_online_predictive_equivalence_weight = float(
        getattr(args, 'spatial_tree_online_predictive_equivalence_weight', 0.0)
    )
    if cfg.policy.spatial_exploration_tree_online_predictive_equivalence_weight < 0.0:
        raise ValueError(
            '--spatial-tree-online-predictive-equivalence-weight cannot be negative'
        )
    cfg.policy.spatial_exploration_tree_online_predictive_equivalence_negative_margin = float(
        getattr(
            args,
            'spatial_tree_online_predictive_equivalence_negative_margin',
            0.2,
        )
    )
    if not -1.0 <= (
        cfg.policy.spatial_exploration_tree_online_predictive_equivalence_negative_margin
    ) <= 1.0:
        raise ValueError(
            '--spatial-tree-online-predictive-equivalence-negative-margin '
            'must be in [-1, 1]'
        )
    cfg.policy.spatial_exploration_tree_horizon = 5
    cfg.policy.spatial_exploration_tree_max_depth = int(
        getattr(args, 'spatial_tree_max_depth', 5)
    )
    cfg.policy.spatial_exploration_tree_reward_mode = str(
        getattr(args, 'spatial_tree_reward_mode', 'bonus')
    )
    cfg.policy.spatial_exploration_tree_rnd_estimator = str(
        getattr(args, 'spatial_tree_rnd_estimator', 'head')
    )
    cfg.policy.spatial_exploration_tree_rnd_hybrid_weight = float(
        getattr(args, 'spatial_tree_rnd_hybrid_weight', 0.5)
    )
    if not 0.0 <= cfg.policy.spatial_exploration_tree_rnd_hybrid_weight <= 1.0:
        raise ValueError('--spatial-tree-rnd-hybrid-weight must be in [0, 1]')
    cfg.policy.spatial_exploration_tree_beta = float(
        getattr(args, 'spatial_tree_beta', 0.05)
    )
    if cfg.policy.spatial_exploration_tree_beta < 0.0:
        raise ValueError('--spatial-tree-beta cannot be negative')
    cfg.policy.spatial_exploration_tree_minimum_reward_factor = float(
        getattr(args, 'spatial_tree_minimum_reward_factor', 0.05)
    )
    if not 0.0 <= cfg.policy.spatial_exploration_tree_minimum_reward_factor <= 1.0:
        raise ValueError(
            '--spatial-tree-minimum-reward-factor must be in [0, 1]'
        )
    cfg.policy.spatial_exploration_tree_adaptive_scale = bool(
        getattr(args, 'spatial_tree_adaptive_scale', True)
    )
    cfg.policy.spatial_exploration_tree_minimum_scale = 1e-3
    cfg.policy.spatial_exploration_tree_score_clip = 5.0
    return cfg, create_cfg


def print_summary(cfg, args: argparse.Namespace) -> None:
    summary = {
        'condition': args.condition,
        'seed': args.seed,
        'exp_name': cfg.exp_name,
        'model_path': cfg.policy.model_path,
        'collection_only': cfg.policy.diagnostic_collection_only,
        'max_env_step': args.max_env_step,
        'max_train_iter': args.max_train_iter,
        'episode_max_step': cfg.env.max_step,
        'collector_env_num': cfg.env.collector_env_num,
        'n_episode': cfg.policy.n_episode,
        'batch_size': cfg.policy.batch_size,
        'replay_ratio': cfg.policy.replay_ratio,
        'cuda': cfg.policy.cuda,
        'observation_shape': tuple(cfg.policy.model.observation_shape),
        'ensemble_size': cfg.policy.model.ensemble_size,
        'num_simulations': cfg.policy.num_simulations,
        'intrinsic_tree_leaf_value_weight': (
            cfg.policy.intrinsic_tree_leaf_value_weight
        ),
        'rnd_fresh_update_batch_size': cfg.policy.rnd_fresh_update_batch_size,
        'rnd_model_type': cfg.policy.rnd_model_type,
        'target_mode': cfg.policy.rnd_intrinsic_target_mode,
        'intrinsic_target': args.intrinsic_target,
        'intrinsic_td_steps': cfg.policy.rnd_intrinsic_td_steps,
        'rnd_intrinsic_bootstrap': cfg.policy.rnd_intrinsic_bootstrap,
        'intrinsic_value_bootstrap': cfg.policy.intrinsic_value_bootstrap,
        'projected_ssl_loss_weight': cfg.policy.ssl_loss_weight,
        'raw_latent_ssl_loss_weight': cfg.policy.raw_latent_ssl_loss_weight,
        'auxiliary_decoder': cfg.policy.rnd_auxiliary_decoder,
        'auxiliary_used_in_mcts': False,
        'compact_query': cfg.policy.rnd_landscape_query,
        'compact_query_space': cfg.policy.rnd_landscape_query_space,
        'compact_query_count': cfg.policy.rnd_landscape_query_count,
        'compact_query_code_dim': cfg.policy.rnd_landscape_query_code_dim,
        'compact_query_reader_use_context': (
            cfg.policy.rnd_landscape_query_reader_use_context
        ),
        'compact_query_used_in_mcts': (
            cfg.policy.rnd_landscape_query
            and cfg.policy.rnd_landscape_query_use_tree_rewards
        ),
        'episodic_memory': cfg.policy.rnd_episodic_memory,
        'episodic_memory_key_space': cfg.policy.rnd_episodic_memory_key_space,
        'episodic_memory_kernel_bandwidth': (
            cfg.policy.rnd_episodic_memory_kernel_bandwidth
        ),
        'episodic_memory_min_reward_factor': (
            cfg.policy.rnd_episodic_memory_min_reward_factor
        ),
        'episodic_memory_include_path': (
            cfg.policy.rnd_episodic_memory_include_path
        ),
        'episodic_memory_mode': cfg.policy.rnd_episodic_memory_mode,
        'episodic_inverse_embedding_dim': (
            cfg.policy.model.episodic_inverse_embedding_dim
        ),
        'episodic_inverse_loss_weight': cfg.policy.episodic_inverse_loss_weight,
        'episodic_inverse_bridge_loss_weight': (
            cfg.policy.episodic_inverse_bridge_loss_weight
        ),
        'episodic_sequence_novelty': cfg.policy.episodic_sequence_novelty,
        'episodic_sequence_novelty_model': (
            cfg.policy.episodic_sequence_novelty_model
        ),
        'episodic_sequence_novelty_horizon': (
            cfg.policy.episodic_sequence_novelty_horizon
        ),
        'episodic_sequence_novelty_history_window': (
            cfg.policy.episodic_sequence_novelty_history_window
        ),
        'episodic_sequence_novelty_inference_mode': (
            cfg.policy.episodic_sequence_novelty_inference_mode
        ),
        'episodic_sequence_novelty_activation_phase': (
            cfg.policy.episodic_sequence_novelty_activation_phase
        ),
        'episodic_sequence_novelty_minimum_reward_factor': (
            cfg.policy.episodic_sequence_novelty_minimum_reward_factor
        ),
        'episodic_sequence_novelty_updates_per_collect': (
            cfg.policy.episodic_sequence_novelty_updates_per_collect
        ),
        'episodic_predictive_memory_joint_updates_per_collect': (
            cfg.policy.episodic_predictive_memory_joint_updates_per_collect
        ),
        'episodic_predictive_memory_distance_loss_weight': (
            cfg.policy.episodic_predictive_memory_distance_loss_weight
        ),
        'episodic_predictive_memory_adaptive_distance_scale': (
            cfg.policy.episodic_predictive_memory_adaptive_distance_scale
        ),
        'episodic_predictive_memory_adaptive_threshold_ratio': (
            cfg.policy.episodic_predictive_memory_adaptive_threshold_ratio
        ),
        'episodic_predictive_memory_adaptive_temperature_ratio': (
            cfg.policy.episodic_predictive_memory_adaptive_temperature_ratio
        ),
        'episodic_predictive_memory_adaptive_minimum_scale': (
            cfg.policy.episodic_predictive_memory_adaptive_minimum_scale
        ),
        'episodic_reconstructive_memory_reconstruction_loss_weight': (
            cfg.policy.episodic_reconstructive_memory_reconstruction_loss_weight
        ),
        'episodic_reconstructive_memory_rollout_loss_weight': (
            cfg.policy.episodic_reconstructive_memory_rollout_loss_weight
        ),
        'episodic_sequence_novelty_uses_imagined_latents': False,
        'episodic_sequence_novelty_uses_counterfactual_training': False,
        'episodic_sequence_novelty_uses_explicit_memory': (
            cfg.policy.episodic_sequence_novelty_model
            in {'predictive_memory', 'reconstructive_memory'}
        ),
        'episodic_sequence_novelty_target': (
            'embedding_memory_distance'
            if cfg.policy.episodic_sequence_novelty_model
            in {'predictive_memory', 'reconstructive_memory'}
            else 'exact_rgb_recent_window_visit'
            if cfg.policy.episodic_sequence_novelty_model == 'recent_transformer'
            else 'exact_rgb_first_visit'
        ),
        'episodic_sequence_continuation_value': (
            cfg.policy.episodic_sequence_continuation_value
        ),
        'episodic_sequence_continuation_value_horizon': (
            cfg.policy.episodic_sequence_continuation_value_horizon
        ),
        'spatial_exploration_shadow': cfg.policy.spatial_exploration_shadow,
        'spatial_exploration_shadow_checkpoint': (
            cfg.policy.spatial_exploration_shadow_checkpoint
        ),
        'spatial_exploration_shadow_horizon': (
            cfg.policy.spatial_exploration_shadow_horizon
        ),
        'spatial_exploration_shadow_history_window': (
            cfg.policy.spatial_exploration_shadow_history_window
        ),
        'spatial_exploration_shadow_roots_per_episode': (
            cfg.policy.spatial_exploration_shadow_roots_per_episode
        ),
        'spatial_exploration_shadow_export_dataset': (
            cfg.policy.spatial_exploration_shadow_export_dataset
        ),
        'spatial_exploration_shadow_export_split_counts': (
            cfg.policy.spatial_exploration_shadow_export_split_counts
        ),
        'spatial_exploration_shadow_affects_mcts': False,
        'spatial_exploration_tree': cfg.policy.spatial_exploration_tree,
        'spatial_exploration_tree_online_training': (
            cfg.policy.spatial_exploration_tree_online_training
        ),
        'spatial_exploration_tree_online_updates_per_collect': (
            cfg.policy.spatial_exploration_tree_online_updates_per_collect
        ),
        'spatial_exploration_tree_online_activation_updates': (
            cfg.policy.spatial_exploration_tree_online_activation_updates
        ),
        'spatial_exploration_tree_online_batch_size': (
            cfg.policy.spatial_exploration_tree_online_batch_size
        ),
        'spatial_exploration_tree_online_roots_per_update': (
            cfg.policy.spatial_exploration_tree_online_roots_per_update
        ),
        'spatial_exploration_tree_online_replay_capacity': (
            cfg.policy.spatial_exploration_tree_online_replay_capacity
        ),
        'spatial_exploration_tree_online_history_window': (
            cfg.policy.spatial_exploration_tree_online_history_window
        ),
        'spatial_exploration_tree_online_prioritized_replay_fraction': (
            cfg.policy.spatial_exploration_tree_online_prioritized_replay_fraction
        ),
        'spatial_exploration_tree_online_prioritized_replay_alpha': (
            cfg.policy.spatial_exploration_tree_online_prioritized_replay_alpha
        ),
        'spatial_exploration_tree_online_prioritized_replay_priority_mode': (
            cfg.policy.spatial_exploration_tree_online_prioritized_replay_priority_mode
        ),
        'spatial_exploration_tree_online_revisit_history_scope': (
            cfg.policy.spatial_exploration_tree_online_revisit_history_scope
        ),
        'spatial_exploration_tree_online_evaluation_roots_per_episode': (
            cfg.policy.spatial_exploration_tree_online_evaluation_roots_per_episode
        ),
        'spatial_exploration_tree_memory_key_mode': (
            cfg.policy.spatial_exploration_tree_memory_key_mode
        ),
        'spatial_exploration_tree_revisit_scoring_mode': (
            cfg.policy.spatial_exploration_tree_revisit_scoring_mode
        ),
        'spatial_exploration_tree_online_rollout_reconstruction_weight': (
            cfg.policy.spatial_exploration_tree_online_rollout_reconstruction_weight
        ),
        'spatial_exploration_tree_online_rollout_contrastive_weight': (
            cfg.policy.spatial_exploration_tree_online_rollout_contrastive_weight
        ),
        'spatial_exploration_tree_online_observation_key_consistency_weight': (
            cfg.policy.spatial_exploration_tree_online_observation_key_consistency_weight
        ),
        'spatial_exploration_tree_online_belief_dynamics_consistency_weight': (
            cfg.policy.spatial_exploration_tree_online_belief_dynamics_consistency_weight
        ),
        'spatial_exploration_tree_online_belief_observation_alignment_weight': (
            cfg.policy.spatial_exploration_tree_online_belief_observation_alignment_weight
        ),
        'spatial_exploration_tree_online_pairwise_revisit_weight': (
            cfg.policy.spatial_exploration_tree_online_pairwise_revisit_weight
        ),
        'spatial_exploration_tree_online_pairwise_revisit_negative_samples': (
            cfg.policy.spatial_exploration_tree_online_pairwise_revisit_negative_samples
        ),
        'spatial_exploration_tree_online_predictive_equivalence_weight': (
            cfg.policy.spatial_exploration_tree_online_predictive_equivalence_weight
        ),
        'spatial_exploration_tree_checkpoint': (
            cfg.policy.spatial_exploration_tree_checkpoint
        ),
        'spatial_exploration_tree_max_depth': (
            cfg.policy.spatial_exploration_tree_max_depth
        ),
        'spatial_exploration_tree_reward_mode': (
            cfg.policy.spatial_exploration_tree_reward_mode
        ),
        'spatial_exploration_tree_rnd_estimator': (
            cfg.policy.spatial_exploration_tree_rnd_estimator
        ),
        'spatial_exploration_tree_rnd_hybrid_weight': (
            cfg.policy.spatial_exploration_tree_rnd_hybrid_weight
        ),
        'spatial_exploration_tree_beta': cfg.policy.spatial_exploration_tree_beta,
        'spatial_exploration_tree_minimum_reward_factor': (
            cfg.policy.spatial_exploration_tree_minimum_reward_factor
        ),
        'spatial_exploration_tree_adaptive_scale': (
            cfg.policy.spatial_exploration_tree_adaptive_scale
        ),
    }
    print(json.dumps(summary, indent=2))


def main() -> None:
    args = build_parser().parse_args()
    cfg, create_cfg = build_config(args)
    print_summary(cfg, args)
    if args.dry_run:
        return

    from lzero.entry.train_ensemble_muzero import train_ensemble_muzero

    train_ensemble_muzero(
        [cfg, create_cfg],
        seed=args.seed,
        model_path=cfg.policy.model_path,
        max_env_step=args.max_env_step,
        max_train_iter=args.max_train_iter,
    )


if __name__ == '__main__':
    main()
