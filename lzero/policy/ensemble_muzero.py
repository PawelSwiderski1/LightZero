import copy
import itertools
import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ding.utils import POLICY_REGISTRY

from lzero.mcts.ensemble_muzero_mcts import DepthNormalizedBackupDisagreementStats, EnsembleSharedTreeMCTS
from lzero.mcts.utils import prepare_observation
from lzero.policy import prepare_obs, select_action
from lzero.policy.episodic_latent_memory import (
    EpisodicLatentMemoryContext,
    EpisodicLatentRewardGate,
)
from lzero.policy.predictive_episodic_memory import (
    PredictiveEpisodicMemory,
    ReconstructiveEpisodicMemory,
    balanced_memory_distance_loss,
)
from lzero.policy.muzero import MuZeroPolicy
from lzero.policy.rnd_auxiliary_decoder import SharedRNDAuxiliaryDecoder
from lzero.policy.rnd_compact_query import CompactImageRNDLandscapeQuery
from lzero.policy.rnd_landscape_query import MountainCarRNDLandscapeQuery
from lzero.policy.recurrent_sequence_novelty import (
    RecentHistoryTransformerHistoryCache,
    RecentHistoryTransformerNovelty,
    RecurrentSequenceContext,
    RecurrentSequenceNovelty,
    episodic_continuation_factor_targets,
    recent_observation_novelty_targets,
)


def _sync_if_cuda(device: str) -> None:
    if str(device).startswith('cuda') and torch.cuda.is_available():
        torch.cuda.synchronize()


def _profile_start(enabled: bool, device: str):
    if not enabled:
        return None
    _sync_if_cuda(device)
    return time.perf_counter()


def _profile_elapsed(start, device: str) -> float:
    if start is None:
        return 0.0
    _sync_if_cuda(device)
    return time.perf_counter() - start


class _LatentRND(nn.Module):
    """
    Random-network-distillation module over flattened inputs.

    The target network is fixed. In the learned intrinsic model path the
    predictor is trained on real replay observations, and its prediction error
    becomes the target for MuZero's intrinsic reward/value heads. The legacy
    direct-bonus path can still feed MuZero latents through this same module.
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            feature_dim: int,
            input_low: Optional[Sequence[float]] = None,
            input_high: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()
        if (input_low is None) != (input_high is None):
            raise ValueError("RND input_low and input_high must either both be set or both be None.")
        if input_low is None:
            low = torch.empty(0, dtype=torch.float32)
            high = torch.empty(0, dtype=torch.float32)
        else:
            low = torch.as_tensor(input_low, dtype=torch.float32).flatten()
            high = torch.as_tensor(input_high, dtype=torch.float32).flatten()
            if low.numel() != input_dim or high.numel() != input_dim:
                raise ValueError(
                    f"RND input bounds must contain {input_dim} values, got {low.numel()} and {high.numel()}."
                )
            if not bool(torch.all(high > low)):
                raise ValueError("Every RND input_high value must be greater than input_low.")
        # Bounds are configuration, not learned state. Keeping them non-persistent also
        # allows checkpoints created before fixed-bound normalization to load.
        self.register_buffer('_input_low', low, persistent=False)
        self.register_buffer('_input_high', high, persistent=False)
        self.target = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim),
        )
        for param in self.target.parameters():
            param.requires_grad_(False)

    def _flatten(self, latent_state: torch.Tensor) -> torch.Tensor:
        flat = latent_state.reshape(latent_state.shape[0], -1).float()
        if self._input_low.numel() == 0:
            return flat
        normalized = 2.0 * (flat - self._input_low) / (self._input_high - self._input_low) - 1.0
        return normalized.clamp_(-1.0, 1.0)

    def forward(self, latent_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        flat = self._flatten(latent_state)
        prediction = self.predictor(flat)
        with torch.no_grad():
            target = self.target(flat)
        return prediction, target

    def error(self, latent_state: torch.Tensor) -> torch.Tensor:
        prediction, target = self.forward(latent_state)
        return (prediction - target).pow(2).mean(dim=1)

    def error_with_input_grad(self, inputs: torch.Tensor) -> torch.Tensor:
        """Evaluate frozen RND while retaining gradients to physical inputs."""
        flat = self._flatten(inputs)
        prediction = self.predictor(flat)
        target = self.target(flat)
        return (prediction - target).pow(2).mean(dim=1)


class _ImageRND(nn.Module):
    """Convolutional RND that preserves image geometry."""

    def __init__(
            self,
            input_channels: int,
            hidden_dim: int,
            feature_dim: int,
            input_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.input_scale = float(input_scale)
        if self.input_scale <= 0.0:
            raise ValueError(f'input_scale must be positive, got {input_scale}.')

        def make_network() -> nn.Module:
            return nn.Sequential(
                nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
                nn.Linear(64 * 4 * 4, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, feature_dim),
            )

        self.target = make_network()
        self.predictor = make_network()
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)

    def _prepare(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.dim() != 4:
            raise ValueError(
                f'Image RND expects [batch, channels, height, width], got {tuple(inputs.shape)}.'
            )
        return inputs.float() / self.input_scale

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = self._prepare(inputs)
        prediction = self.predictor(inputs)
        with torch.no_grad():
            target = self.target(inputs)
        return prediction, target

    def error(self, inputs: torch.Tensor) -> torch.Tensor:
        prediction, target = self.forward(inputs)
        return (prediction - target).pow(2).mean(dim=1)

    def error_with_input_grad(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self._prepare(inputs)
        prediction = self.predictor(inputs)
        target = self.target(inputs)
        return (prediction - target).pow(2).mean(dim=1)


@POLICY_REGISTRY.register('ensemble_muzero')
class EnsembleMuZeroPolicy(MuZeroPolicy):
    """
    Policy for the ensemble MuZero variant.

    At this stage, collect mode can pin one ensemble member per environment
    episode for Thompson-style exploration, and learn mode trains every member
    on the same sampled replay batch. Future work should add pessimistic
    ensemble aggregation for planning/evaluation.
    """

    config = copy.deepcopy(MuZeroPolicy.config)
    config['model'].update(
        dict(
            ensemble_size=1,
            random_prior_scale=0.0,
            value_prior_backprop_to_latent=False,
            value_prior_latent_normalization='layer_norm',
            value_prior_scalar_loss_weight=1.0,
            encoder_random_prior_scale=0.0,
            encoder_random_prior_mode='smooth_mlp',
            encoder_random_prior_hidden_dim=64,
            encoder_random_prior_cell_size=0.25,
            encoder_random_prior_hash_size=16384,
            encoder_random_prior_input_mean=None,
            encoder_random_prior_input_scale=None,
            encoder_random_prior_output_norm=1.0,
            encoder_prior_composed_latent_normalization='none',
            encoder_prior_composed_latent_norm_scale=1.0,
            latent_novelty_dim=0,
            latent_novelty_hidden_dim=64,
            latent_novelty_prior_scale=0.0,
        )
    )
    config.update(
        dict(
            thompson_sampling=False,
            pessimistic_search=False,
            pessimism_beta=0.0,
            ensemble_search_mode='thompson',
            ensemble_ucb_beta=0.0,
            ensemble_normalize_q=True,
            ensemble_uncertainty_decay=False,
            mixed_reward_tree=False,
            main_tree_policy_prior_mode='learned',
            main_tree_force_root_action_coverage=False,
            depth_normalized_backup_disagreement=False,
            depth_disagreement_ema_momentum=0.5,
            depth_disagreement_trim_fraction=0.1,
            depth_disagreement_min_groups=100,
            depth_disagreement_min_scale=0.5,
            depth_disagreement_max_depth=0,
            depth_disagreement_conservative_cap=True,
            ensemble_value_calibration=False,
            ensemble_value_calibration_start_phase=10,
            ensemble_value_calibration_sample_size=4096,
            ensemble_value_calibration_inference_batch_size=1024,
            final_qstd_beta=0.0,
            final_qstd_temperature=None,
            final_qstd_store_final_policy=True,
            separate_exploration_tree=False,
            separate_exploration_tree_horizon=3,
            separate_exploration_tree_top_k=3,
            separate_exploration_tree_beta=0.0,
            separate_exploration_tree_temperature=None,
            separate_exploration_tree_score_normalization='minmax',
            separate_exploration_tree_max_sequences=256,
            separate_exploration_tree_store_final_policy=True,
            separate_exploration_tree_sequence_score_mode='qstd',
            separate_exploration_tree_sequence_qstd_beta=1.0,
            separate_exploration_tree_sequence_qstd_mode='raw',
            pure_exploration_episode_fraction=0.0,
            pure_exploration_commit_steps=1,
            pure_exploration_sequence_temperature=0.0,
            pure_exploration_store_exploration_policy=True,
            ensemble_eval_search_mode='member0',
            ensemble_eval_ucb_beta=None,
            ensemble_eval_uncertainty_decay=None,
            reanalyze_use_search_value_target=False,
            value_target_mode='auto',
            sve_start_train_iter=0,
            sve_recent_window=0,
            sve_real_reward_prefix_steps=0,
            sve_search_noise=False,
            ensemble_member_bootstrap_targets=True,
            bootstrap_at_time_limit=True,
            rnd_exploration=False,
            rnd_intrinsic_reward_weight=0.0,
            rnd_learning_rate=1e-4,
            rnd_hidden_dim=128,
            rnd_feature_dim=64,
            rnd_model_type='auto',
            rnd_image_input_scale=1.0,
            rnd_train_batch_size=512,
            rnd_predictor_update_proportion=0.25,
            rnd_train_on_new_data_only=True,
            rnd_train_from_fresh_collection_only=False,
            rnd_fresh_update_batch_size=64,
            rnd_train_observation_mode='next',
            rnd_observation_dim=None,
            rnd_observation_indices=None,
            rnd_observation_low=None,
            rnd_observation_high=None,
            rnd_ignore_elite_replay_for_predictor=True,
            rnd_bonus_normalization='zscore',
            rnd_bonus_normalization_momentum=0.99,
            rnd_bonus_clip=5.0,
            rnd_store_collection_rewards=False,
            rnd_update_stats_on_collection=True,
            rnd_use_stored_collection_rewards=False,
            rnd_intrinsic_target_mode='td',
            rnd_use_learned_intrinsic_model=True,
            rnd_intrinsic_discount_factor=None,
            rnd_intrinsic_td_steps=None,
            rnd_intrinsic_bootstrap=True,
            rnd_target_snapshot=True,
            rnd_target_snapshot_update_interval=None,
            rnd_target_snapshot_refresh_per_collect=False,
            rnd_target_snapshot_refresh_after_fresh_update=False,
            rnd_initialization_seed=None,
            rnd_auxiliary_decoder=False,
            rnd_auxiliary_decoder_hidden_dim=64,
            rnd_auxiliary_decoder_learning_rate=3e-4,
            rnd_auxiliary_decoder_weight_decay=0.0,
            rnd_auxiliary_decoder_loss_weight=1.0,
            rnd_auxiliary_decoder_max_horizon=8,
            rnd_auxiliary_decoder_grad_clip=5.0,
            rnd_auxiliary_decoder_target_transform='log',
            rnd_auxiliary_decoder_target_log_reference=1.0,
            rnd_auxiliary_decoder_error_epsilon=1e-10,
            rnd_auxiliary_decoder_max_raw_error=None,
            rnd_auxiliary_decoder_normalize_latent=True,
            rnd_auxiliary_decoder_train_world_model=True,
            rnd_auxiliary_decoder_initialization_seed=None,
            intrinsic_terminal_observation_mode='mountain_car_goal',
            rnd_landscape_query=False,
            rnd_landscape_query_space='physical_2d',
            rnd_landscape_query_count=16,
            rnd_landscape_query_hidden_dim=64,
            rnd_landscape_query_code_dim=32,
            rnd_landscape_query_reconstruction_loss_weight=1.0,
            rnd_landscape_query_diagnostic_root_count=0,
            rnd_landscape_query_learning_rate=3e-4,
            rnd_landscape_query_weight_decay=0.0,
            rnd_landscape_query_loss_weight=1.0,
            rnd_landscape_query_max_horizon=8,
            rnd_landscape_query_min_snapshots=2,
            rnd_landscape_query_diversity_weight=0.01,
            rnd_landscape_query_centered_prediction_weight=0.0,
            rnd_landscape_query_grad_clip=5.0,
            rnd_landscape_query_reader_use_context=True,
            rnd_landscape_query_member_specific_reader=False,
            rnd_landscape_query_reader_architecture='landscape',
            rnd_landscape_query_reader_hidden_channels=(32,),
            rnd_landscape_query_reader_norm_type='BN',
            rnd_landscape_query_reader_last_linear_layer_init_zero=False,
            rnd_landscape_query_target_transform='log',
            rnd_landscape_query_target_log_reference=1.0,
            rnd_landscape_query_error_epsilon=1e-10,
            rnd_landscape_query_max_raw_error=None,
            rnd_landscape_query_normalized_raw_reference_size=8192,
            rnd_landscape_query_normalized_raw_clip=10.0,
            rnd_landscape_query_normalized_raw_min_scale=1e-8,
            rnd_landscape_query_gradient_diagnostics_interval=0,
            rnd_landscape_query_train_world_model=True,
            rnd_landscape_query_use_tree_rewards=True,
            rnd_landscape_query_initialization_seed=None,
            rnd_episodic_memory=False,
            rnd_episodic_memory_key_space='projected',
            rnd_episodic_memory_max_size=1200,
            rnd_episodic_memory_inference_batch_size=128,
            rnd_episodic_memory_kernel_bandwidth=0.25,
            rnd_episodic_memory_min_reward_factor=0.05,
            rnd_episodic_memory_distance_epsilon=1e-8,
            rnd_episodic_memory_include_path=True,
            rnd_episodic_memory_mode='minimum_distance',
            rnd_episodic_memory_knn_k=10,
            rnd_episodic_memory_knn_cluster_distance=0.008,
            rnd_episodic_memory_knn_kernel_epsilon=1e-4,
            rnd_episodic_memory_knn_pseudo_count_constant=0.001,
            rnd_episodic_memory_knn_maximum_similarity=8.0,
            rnd_episodic_memory_knn_warmup_until_full=True,
            episodic_inverse_loss_weight=1.0,
            episodic_inverse_bridge_loss_weight=1.0,
            episodic_inverse_train_world_model=True,
            episodic_inverse_max_horizon=5,
            rnd_aux_confirmatory_condition='none',
            rnd_aux_confirmatory_plan_version=0,
            intrinsic_reward_source='rnd',
            episodic_novelty_diagnostics=False,
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
            episodic_novelty_history_shadow=False,
            episodic_novelty_history_learning_rate=3e-4,
            episodic_novelty_history_train_epochs=1,
            episodic_novelty_history_train_batch_size=5,
            episodic_novelty_history_updates_per_collect=60,
            episodic_novelty_history_replay_capacity=1000,
            episodic_novelty_history_recent_phase_count=2,
            episodic_novelty_history_recent_sample_ratio=0.5,
            episodic_novelty_history_grad_clip=5.0,
            episodic_sequence_novelty=False,
            episodic_sequence_novelty_model='recurrent',
            episodic_sequence_novelty_embedding_dim=64,
            episodic_sequence_novelty_hidden_dim=128,
            episodic_sequence_novelty_horizon=5,
            episodic_sequence_novelty_history_window=40,
            episodic_sequence_novelty_transformer_layers=2,
            episodic_sequence_novelty_transformer_heads=4,
            episodic_sequence_novelty_transformer_dropout=0.0,
            episodic_sequence_novelty_inference_mode='direct',
            episodic_sequence_novelty_discount_factor=0.99,
            episodic_sequence_novelty_learning_rate=3e-4,
            episodic_sequence_novelty_weight_decay=1e-5,
            episodic_sequence_novelty_sequence_loss_weight=1.0,
            episodic_sequence_novelty_grad_clip=5.0,
            episodic_sequence_novelty_updates_per_collect=64,
            episodic_sequence_novelty_train_batch_size=1,
            episodic_sequence_novelty_roots_per_update=32,
            episodic_sequence_novelty_replay_capacity=32,
            episodic_sequence_novelty_minimum_reward_factor=0.05,
            episodic_sequence_novelty_activation_phase=1,
            episodic_sequence_novelty_observation_scale=1.0,
            episodic_sequence_novelty_initialization_seed=None,
            episodic_predictive_memory_prediction_loss_weight=1.0,
            episodic_predictive_memory_distance_loss_weight=1.0,
            episodic_predictive_memory_inverse_loss_weight=1.0,
            episodic_predictive_memory_target_ema=0.99,
            episodic_predictive_memory_joint_updates_per_collect=None,
            episodic_predictive_memory_initial_distance_threshold=0.08,
            episodic_predictive_memory_initial_distance_temperature=0.02,
            episodic_predictive_memory_adaptive_distance_scale=False,
            episodic_predictive_memory_adaptive_threshold_ratio=0.5,
            episodic_predictive_memory_adaptive_temperature_ratio=0.125,
            episodic_predictive_memory_adaptive_minimum_scale=0.02,
            episodic_predictive_memory_adaptive_distinct_epsilon=1e-6,
            episodic_reconstructive_memory_decoder_channels=64,
            episodic_reconstructive_memory_reconstruction_loss_weight=1.0,
            episodic_reconstructive_memory_rollout_loss_weight=1.0,
            episodic_sequence_continuation_value=False,
            episodic_sequence_continuation_value_horizon=20,
            episodic_sequence_continuation_loss_weight=1.0,
            spatial_exploration_tree=False,
            spatial_exploration_tree_checkpoint=None,
            spatial_exploration_tree_online_training=False,
            spatial_exploration_tree_online_initialization_seed=0,
            spatial_exploration_tree_online_learning_rate=1e-4,
            spatial_exploration_tree_online_weight_decay=1e-5,
            spatial_exploration_tree_online_activation_updates=16,
            spatial_exploration_tree_online_replay_capacity=16,
            spatial_exploration_tree_online_prioritized_replay_fraction=0.0,
            spatial_exploration_tree_online_prioritized_replay_alpha=0.6,
            spatial_exploration_tree_online_prioritized_replay_epsilon=1e-3,
            spatial_exploration_tree_online_prioritized_replay_priority_mode='belief_error',
            spatial_exploration_tree_online_revisit_history_scope='episode',
            spatial_exploration_tree_online_history_window=128,
            spatial_exploration_tree_online_batch_size=2,
            spatial_exploration_tree_online_roots_per_update=16,
            spatial_exploration_tree_online_updates_per_collect=16,
            spatial_exploration_tree_online_evaluation_roots_per_episode=64,
            spatial_exploration_tree_online_grad_clip=5.0,
            spatial_exploration_tree_online_memory_novelty_weight=10.0,
            spatial_exploration_tree_memory_key_mode='observation',
            spatial_exploration_tree_revisit_scoring_mode='model',
            spatial_exploration_tree_online_rollout_reconstruction_weight=0.0,
            spatial_exploration_tree_online_rollout_contrastive_weight=1.0,
            spatial_exploration_tree_online_observation_key_consistency_weight=0.0,
            spatial_exploration_tree_online_belief_dynamics_consistency_weight=0.0,
            spatial_exploration_tree_online_belief_observation_alignment_weight=1.0,
            spatial_exploration_tree_online_pairwise_revisit_weight=1.0,
            spatial_exploration_tree_online_pairwise_revisit_negative_samples=8,
            spatial_exploration_tree_online_predictive_equivalence_weight=0.0,
            spatial_exploration_tree_online_predictive_equivalence_negative_margin=0.2,
            spatial_exploration_tree_horizon=5,
            spatial_exploration_tree_max_depth=5,
            spatial_exploration_tree_reward_mode='bonus',
            spatial_exploration_tree_rnd_estimator='head',
            spatial_exploration_tree_rnd_hybrid_weight=0.5,
            spatial_exploration_tree_beta=0.05,
            spatial_exploration_tree_minimum_reward_factor=0.05,
            spatial_exploration_tree_adaptive_scale=True,
            spatial_exploration_tree_minimum_scale=1e-3,
            spatial_exploration_tree_score_clip=5.0,
            state_value_disagreement_reward_scale=1.0,
            state_value_disagreement_reward_clip=None,
            state_value_disagreement_inference_batch_size=1024,
            state_value_disagreement_intrinsic_target_mode='td',
            state_value_disagreement_store_collection_rewards=False,
            state_value_disagreement_use_stored_collection_rewards=False,
            state_density_reward_scale=1.0,
            state_density_reward_clip=1.0,
            state_density_position_min=-1.2,
            state_density_position_max=0.6,
            state_density_velocity_min=-0.07,
            state_density_velocity_max=0.07,
            state_density_position_bin_width=0.02,
            state_density_velocity_bin_width=0.002,
            state_density_neighbor_radius=1,
            state_density_intrinsic_target_mode='td',
            state_density_train_on_intrinsic_replay_only=False,
            qstd_exploration=False,
            qstd_intrinsic_reward_weight=0.0,
            qstd_store_collection_rewards=False,
            qstd_reward_aggregation='max',
            qstd_reward_top_k=1,
            qstd_reward_scale=1.0,
            qstd_reward_clip=None,
            qstd_reward_num_simulations=None,
            qstd_reward_search_mode='mean_shared_tree',
            qstd_recompute_replay_rewards=False,
            qstd_one_step_include_reward=True,
            qstd_one_step_use_value_calibration=True,
            qstd_target_snapshot=True,
            qstd_target_snapshot_refresh_per_collect=True,
            qstd_target_inference_batch_size=1024,
            qstd_intrinsic_target_mode='td',
            dynamics_disagreement_exploration=False,
            dynamics_disagreement_intrinsic_reward_weight=0.0,
            dynamics_disagreement_shared_root_latent=True,
            dynamics_disagreement_normalize_latent=True,
            dynamics_disagreement_space='latent',
            dynamics_disagreement_reward_clip=None,
            dynamics_disagreement_train_intrinsic_value=False,
            dynamics_prior_alignment_loss_weight=0.0,
            dynamics_prior_alignment_target_normalization='none',
            store_selected_intrinsic_rewards=False,
            separate_intrinsic_tree=False,
            collection_policy_mode='normal',
            intrinsic_tree_num_simulations=None,
            intrinsic_tree_add_exploration_noise=False,
            intrinsic_tree_prior_mode='learned',
            intrinsic_tree_force_root_action_coverage=False,
            intrinsic_tree_pure_exploration_episodes=False,
            intrinsic_tree_pure_episode_store_intrinsic_policy=False,
            intrinsic_tree_final_score_temperature=1.0,
            intrinsic_tree_counterfactual_betas=(0.0, 1.0, 5.0, 20.0, 50.0),
            intrinsic_tree_store_final_policy=True,
            intrinsic_tree_eval=False,
            intrinsic_only_training=False,
            intrinsic_tree_normalize_q=True,
            intrinsic_tree_immediate_reward_weight=1.0,
            intrinsic_tree_leaf_value_source='default',
            intrinsic_tree_leaf_value_weight=1.0,
            intrinsic_tree_leaf_value_clip=None,
            intrinsic_value_bootstrap=True,
            collect_random_action_epsilon=0.0,
            main_tree_trace_diagnostics=False,
            main_tree_trace_min_train_iter=0,
            main_tree_trace_max_per_train_iter=4,
            main_tree_trace_max_depth=4,
            main_tree_trace_max_nodes=128,
            main_tree_trace_min_q_std=5.0,
            main_tree_trace_selected_action=-1,
            main_tree_trace_min_position=None,
            main_tree_trace_max_position=None,
            main_tree_trace_min_velocity=None,
            main_tree_trace_max_velocity=None,
            intrinsic_tree_trace_diagnostics=False,
            intrinsic_tree_trace_best_action=1,
            intrinsic_tree_trace_min_train_iter=0,
            intrinsic_tree_trace_max_per_train_iter=4,
            intrinsic_tree_trace_max_depth=4,
            intrinsic_tree_trace_max_nodes=128,
            intrinsic_tree_trace_min_position=None,
            intrinsic_tree_trace_max_position=None,
            intrinsic_tree_trace_min_velocity=None,
            intrinsic_tree_trace_max_velocity=None,
            intrinsic_model_loss_weight=1.0,
            intrinsic_reward_loss_weight=1.0,
            intrinsic_value_loss_weight=1.0,
            intrinsic_train_value_head=True,
            intrinsic_value_train_on_pure_exploration_only=False,
            # LR for intrinsic reward/value head parameter groups. Shared h/g
            # parameters use the primary MuZero LR and receive the joint loss.
            intrinsic_learning_rate=None,
            intrinsic_head_only=True,
            intrinsic_loss_type='huber',
            intrinsic_huber_delta=1.0,
            intrinsic_reward_target_clip=5.0,
            intrinsic_value_target_clip=50.0,
            elite_episode_monte_carlo_value_target=True,
            intrinsic_episode_replay=False,
            intrinsic_episode_buffer_size=20,
            intrinsic_episode_sample_ratio=0.25,
            intrinsic_episode_min_score=0.0,
            intrinsic_episode_score_mode='sum',
            intrinsic_episode_peak_window_before=120,
            intrinsic_episode_peak_window_after=5,
            save_intrinsic_episode_trajectories=True,
            intrinsic_episode_save_dir=None,
            collection_diagnostics=True,
            collection_root_action_diagnostics=False,
            depth_matched_qstd_diagnostics=False,
            depth_matched_qstd_max_depth=5,
            collection_ssl_diagnostics=False,
            collection_ssl_diagnostics_batch_size=512,
            encoder_prior_grid_diagnostics=False,
            encoder_prior_grid_position_min=-1.2,
            encoder_prior_grid_position_max=0.6,
            encoder_prior_grid_position_count=37,
            encoder_prior_grid_velocity_min=-0.07,
            encoder_prior_grid_velocity_max=0.07,
            encoder_prior_grid_velocity_count=29,
            fixed_sequence_intrinsic_diagnostics=False,
            fixed_sequence_intrinsic_max_states=64,
            fixed_sequence_intrinsic_sequences=None,
            diagnostics_dir=None,
            save_elite_episode_trajectories=True,
            elite_episode_save_dir=None,
            profile_learn_timing=False,
            latent_novelty_diagnostics=False,
            latent_novelty_learning_rate=3e-4,
            latent_novelty_train_batch_size=512,
            latent_novelty_temperature=0.1,
            latent_novelty_alignment_loss_weight=0.1,
        )
    )

    def _validate_ensemble_skeleton(self) -> None:
        if self._cfg.pessimistic_search:
            raise NotImplementedError("Pessimistic search is not implemented in the skeleton policy yet.")

    def _validate_intrinsic_only_training(self) -> None:
        if not bool(self._cfg.get('intrinsic_only_training', False)):
            return
        requirements = {
            'collection_policy_mode': self._cfg.get('collection_policy_mode', 'normal') == 'intrinsic_tree_only',
            'separate_intrinsic_tree': bool(self._cfg.get('separate_intrinsic_tree', False)),
            'learned_intrinsic_model': self._use_learned_intrinsic_model(),
            'intrinsic_world_model_training': not bool(self._cfg.get('intrinsic_head_only', True)),
            'learned_intrinsic_policy_prior': self._cfg.get('intrinsic_tree_prior_mode', 'learned') == 'learned',
            'stored_intrinsic_policy_targets': bool(
                self._cfg.get('intrinsic_tree_store_final_policy', True)
            ),
            'main_reward_loss_disabled': float(self._cfg.get('reward_loss_weight', 1.0)) == 0.0,
            'main_value_loss_disabled': float(self._cfg.get('value_loss_weight', 0.25)) == 0.0,
            'policy_training_enabled': float(self._cfg.get('policy_loss_weight', 1.0)) > 0.0,
            'main_tree_reanalysis_disabled': float(self._cfg.get('reanalyze_ratio', 0.0)) == 0.0,
            'extrinsic_priority_disabled': not bool(self._cfg.get('use_priority', False)),
            'extrinsic_elite_replay_disabled': not bool(
                self._cfg.get('elite_episode_replay', False)
            ),
        }
        failed = [name for name, satisfied in requirements.items() if not satisfied]
        if failed:
            raise ValueError(
                "intrinsic_only_training has incompatible configuration: " + ", ".join(failed)
            )

    def _init_rnd_state(self) -> None:
        if not hasattr(self, '_rnd_model'):
            self._rnd_model = None
            self._rnd_input_shape = None
            self._rnd_target_snapshot_model = None
            self._rnd_optimizer = None
            self._episodic_history_optimizer = None
            self._episodic_sequence_novelty_model = None
            self._episodic_sequence_novelty_optimizer = None
            self._episodic_sequence_novelty_replay = []
            self._episodic_sequence_novelty_trained_phase = -1
            self._episodic_sequence_continuation_trained_phase = -1
            self._episodic_sequence_novelty_last_train_stats = {}
            self._episodic_sequence_novelty_factor_stats = {}
            self._episodic_sequence_continuation_factor_stats = {}
            self._episodic_sequence_novelty_inference_stats = self._new_episodic_sequence_inference_stats()
            self._episodic_sequence_novelty_root_action_records = []
            self._episodic_sequence_novelty_selected_stats = {
                'count': 0,
                'prediction_sum': 0.0,
                'target_sum': 0.0,
                'squared_error_sum': 0.0,
                'correct_sum': 0.0,
            }
            self._rnd_bonus_mean = 0.0
            self._rnd_bonus_var = 1.0
            self._rnd_bonus_stats_initialized = False
            self._rnd_trained_observation_keys = set()
            self._rnd_fresh_pending_inputs = None
            self._learn_update_count = 0
            self._rnd_target_snapshot_last_update = -1
            self._rnd_landscape_query_model = None
            self._rnd_landscape_query_snapshot_count = 0
            self._rnd_landscape_query_last_train_stats = {}
            self._rnd_landscape_query_target_scale_phase = -1
            self._rnd_landscape_query_target_scale_sample_count = 0
            self._rnd_compact_query_diagnostic_buffer = []
            self._rnd_episodic_memory_diagnostic_buffer = []
            self._rnd_auxiliary_decoder_model = None
            self._rnd_auxiliary_decoder_optimizer_group_index = None
            self._rnd_auxiliary_decoder_has_backward = False
            self._rnd_auxiliary_decoder_last_train_stats = {}
            self._intrinsic_tree_trace_counts = {}
            self._main_tree_trace_counts = {}
        if not hasattr(self, '_latent_novelty_optimizer'):
            self._latent_novelty_optimizer = None
        if not hasattr(self, '_episodic_history_optimizer'):
            self._episodic_history_optimizer = None
        if not hasattr(self, '_episodic_sequence_novelty_model'):
            self._episodic_sequence_novelty_model = None
            self._episodic_sequence_novelty_optimizer = None
            self._episodic_sequence_novelty_replay = []
            self._episodic_sequence_novelty_trained_phase = -1
            self._episodic_sequence_continuation_trained_phase = -1
            self._episodic_sequence_novelty_last_train_stats = {}
            self._episodic_sequence_novelty_factor_stats = {}
            self._episodic_sequence_continuation_factor_stats = {}
            self._episodic_sequence_novelty_inference_stats = self._new_episodic_sequence_inference_stats()
            self._episodic_sequence_novelty_root_action_records = []
            self._episodic_sequence_novelty_selected_stats = {
                'count': 0,
                'prediction_sum': 0.0,
                'target_sum': 0.0,
                'squared_error_sum': 0.0,
                'correct_sum': 0.0,
            }
        if not hasattr(self, '_spatial_exploration_tree'):
            self._spatial_exploration_tree = None
        if not hasattr(self, '_episodic_sequence_novelty_root_action_records'):
            self._episodic_sequence_novelty_root_action_records = []
        if not hasattr(self, '_episodic_sequence_continuation_trained_phase'):
            self._episodic_sequence_continuation_trained_phase = -1
        if not hasattr(self, '_episodic_sequence_continuation_factor_stats'):
            self._episodic_sequence_continuation_factor_stats = {}
        if not hasattr(self, '_episodic_sequence_novelty_inference_stats'):
            self._episodic_sequence_novelty_inference_stats = self._new_episodic_sequence_inference_stats()
        if not hasattr(self, '_rnd_landscape_query_model'):
            self._rnd_landscape_query_model = None
            self._rnd_landscape_query_snapshot_count = 0
            self._rnd_landscape_query_last_train_stats = {}
        if not hasattr(self, '_rnd_landscape_query_snapshot_count'):
            self._rnd_landscape_query_snapshot_count = 0
        if not hasattr(self, '_rnd_landscape_query_target_scale_phase'):
            self._rnd_landscape_query_target_scale_phase = -1
            self._rnd_landscape_query_target_scale_sample_count = 0
        if not hasattr(self, '_rnd_compact_query_diagnostic_buffer'):
            self._rnd_compact_query_diagnostic_buffer = []
        if not hasattr(self, '_rnd_episodic_memory_diagnostic_buffer'):
            self._rnd_episodic_memory_diagnostic_buffer = []
        if not hasattr(self, '_episodic_history_episode_replay'):
            self._episodic_history_episode_replay = []
            self._episodic_history_replay_next_id = 0
        if not hasattr(self, '_intrinsic_tree_trace_counts'):
            self._intrinsic_tree_trace_counts = {}
        if not hasattr(self, '_main_tree_trace_counts'):
            self._main_tree_trace_counts = {}
        if not hasattr(self, '_state_density_snapshot_counts'):
            self._state_density_snapshot_counts = None
            self._state_density_snapshot_phase = -1
        if not hasattr(self, '_qstd_target_snapshot_model'):
            self._qstd_target_snapshot_model = None
            self._qstd_target_snapshot_value_offsets = None
            self._qstd_target_reward_cache = {}
            self._qstd_target_snapshot_generation = 0
        if not hasattr(self, '_rnd_fresh_pending_inputs'):
            self._rnd_fresh_pending_inputs = None
        if not hasattr(self, '_rnd_input_shape'):
            self._rnd_input_shape = None
        if not hasattr(self, '_rnd_auxiliary_decoder_model'):
            self._rnd_auxiliary_decoder_model = None
            self._rnd_auxiliary_decoder_optimizer_group_index = None
            self._rnd_auxiliary_decoder_has_backward = False
            self._rnd_auxiliary_decoder_last_train_stats = {}

    def _episodic_sequence_novelty_enabled(self) -> bool:
        return bool(self._cfg.get('episodic_sequence_novelty', False))

    def _spatial_exploration_tree_enabled(self) -> bool:
        return bool(self._cfg.get('spatial_exploration_tree', False))

    def _spatial_exploration_tree_reward_mode(self) -> str:
        mode = str(
            self._cfg.get('spatial_exploration_tree_reward_mode', 'bonus')
        ).lower()
        if mode not in {'none', 'bonus', 'gate'}:
            raise ValueError(
                "spatial_exploration_tree_reward_mode must be 'none', 'bonus', "
                "or 'gate'"
            )
        return mode

    def _spatial_exploration_tree_rnd_estimator(self) -> str:
        estimator = str(
            self._cfg.get('spatial_exploration_tree_rnd_estimator', 'head')
        ).lower()
        if estimator not in {'head', 'direct', 'hybrid'}:
            raise ValueError(
                "spatial_exploration_tree_rnd_estimator must be 'head', "
                "'direct', or 'hybrid'"
            )
        return estimator

    def _spatial_exploration_rnd_transform_fn(
            self,
            planner,
            roots,
            maximum_depth: int,
            estimator: str,
    ):
        estimator = str(estimator).lower()
        if estimator == 'head':
            return None
        if self._rnd_model is None or self._rnd_input_shape is None:
            raise RuntimeError(
                'Direct spatial RND estimation requires an initialized RND model.'
            )
        hybrid_weight = float(
            self._cfg.get('spatial_exploration_tree_rnd_hybrid_weight', 0.5)
        )
        if not 0.0 <= hybrid_weight <= 1.0:
            raise ValueError(
                'spatial_exploration_tree_rnd_hybrid_weight must be in [0, 1]'
            )
        maximum_depth = min(planner.horizon, max(1, int(maximum_depth)))
        expected_shape = tuple(int(value) for value in self._rnd_input_shape)
        rnd_model = self._rnd_target_model_for_targets()
        if rnd_model is None:
            raise RuntimeError('No RND target model is available for direct estimation.')
        rnd_model.eval()

        def transform(root_indices, action_paths, _next_latent, base_rewards):
            transformed = np.asarray(base_rewards, dtype=np.float32).reshape(-1).copy()
            valid_rows = [
                row
                for row, path in enumerate(action_paths)
                if 0 < len(path) <= maximum_depth
            ]
            if not valid_rows:
                return transformed
            valid_root_indices = [root_indices[row] for row in valid_rows]
            valid_paths = [action_paths[row] for row in valid_rows]
            predicted_observations = planner.predict_path_observations(
                roots,
                valid_root_indices,
                valid_paths,
                maximum_depth=maximum_depth,
            )
            if predicted_observations.shape[1] != expected_shape[0]:
                raise ValueError(
                    'Predicted observation channels do not match RND input: '
                    f'{predicted_observations.shape[1]} versus {expected_shape[0]}.'
                )
            if tuple(predicted_observations.shape[-2:]) != expected_shape[-2:]:
                predicted_observations = F.interpolate(
                    predicted_observations,
                    size=expected_shape[-2:],
                    mode='bilinear',
                    align_corners=False,
                )
            with torch.inference_mode():
                errors = rnd_model.error(predicted_observations)
                direct_rewards = self._normalize_rnd_errors(errors)
            direct_values = direct_rewards.detach().cpu().numpy().astype(
                np.float32, copy=False
            )
            for direct_row, output_row in enumerate(valid_rows):
                if estimator == 'direct':
                    transformed[output_row] = direct_values[direct_row]
                else:
                    transformed[output_row] = (
                        (1.0 - hybrid_weight) * transformed[output_row]
                        + hybrid_weight * direct_values[direct_row]
                    )
            return transformed

        return transform

    def _ensure_spatial_exploration_tree(self):
        if not self._spatial_exploration_tree_enabled():
            return None
        if self._spatial_exploration_tree is not None:
            return self._spatial_exploration_tree
        from lzero.policy.spatial_exploration_tree import (
            FrozenSpatialExplorationTree,
        )

        checkpoint = self._cfg.get('spatial_exploration_tree_checkpoint', None)
        online_training = bool(
            self._cfg.get('spatial_exploration_tree_online_training', False)
        )
        if not checkpoint and not online_training:
            raise ValueError(
                'spatial_exploration_tree requires spatial_exploration_tree_checkpoint'
            )
        self._spatial_exploration_tree = FrozenSpatialExplorationTree(
            checkpoint_path=None if not checkpoint else str(checkpoint),
            device=str(self._cfg.device),
            horizon=int(self._cfg.get('spatial_exploration_tree_horizon', 5)),
            adaptive_scale=bool(
                self._cfg.get('spatial_exploration_tree_adaptive_scale', True)
            ),
            minimum_scale=float(
                self._cfg.get('spatial_exploration_tree_minimum_scale', 1e-3)
            ),
            score_clip=self._cfg.get('spatial_exploration_tree_score_clip', 5.0),
            online_training=online_training,
            initialization_seed=int(
                self._cfg.get(
                    'spatial_exploration_tree_online_initialization_seed', 0
                )
            ),
            learning_rate=float(
                self._cfg.get(
                    'spatial_exploration_tree_online_learning_rate', 1e-4
                )
            ),
            weight_decay=float(
                self._cfg.get(
                    'spatial_exploration_tree_online_weight_decay', 1e-5
                )
            ),
            activation_updates=int(
                self._cfg.get(
                    'spatial_exploration_tree_online_activation_updates', 16
                )
            ),
            replay_capacity=int(
                self._cfg.get(
                    'spatial_exploration_tree_online_replay_capacity', 16
                )
            ),
            history_window=int(
                self._cfg.get(
                    'spatial_exploration_tree_online_history_window', 128
                )
            ),
            batch_size=int(
                self._cfg.get(
                    'spatial_exploration_tree_online_batch_size', 2
                )
            ),
            roots_per_update=int(
                self._cfg.get(
                    'spatial_exploration_tree_online_roots_per_update', 16
                )
            ),
            updates_per_collect=int(
                self._cfg.get(
                    'spatial_exploration_tree_online_updates_per_collect', 16
                )
            ),
            grad_clip=float(
                self._cfg.get(
                    'spatial_exploration_tree_online_grad_clip', 5.0
                )
            ),
            memory_novelty_weight=float(
                self._cfg.get(
                    'spatial_exploration_tree_online_memory_novelty_weight', 10.0
                )
            ),
            memory_key_mode=str(
                self._cfg.get(
                    'spatial_exploration_tree_memory_key_mode', 'observation'
                )
            ),
            revisit_scoring_mode=str(
                self._cfg.get(
                    'spatial_exploration_tree_revisit_scoring_mode', 'model'
                )
            ),
            rollout_reconstruction_weight=float(
                self._cfg.get(
                    'spatial_exploration_tree_online_rollout_reconstruction_weight',
                    0.0,
                )
            ),
            rollout_contrastive_weight=float(
                self._cfg.get(
                    'spatial_exploration_tree_online_rollout_contrastive_weight',
                    1.0,
                )
            ),
            observation_key_consistency_weight=float(
                self._cfg.get(
                    'spatial_exploration_tree_online_observation_key_consistency_weight',
                    0.0,
                )
            ),
            belief_dynamics_consistency_weight=float(
                self._cfg.get(
                    'spatial_exploration_tree_online_belief_dynamics_consistency_weight',
                    0.0,
                )
            ),
            belief_observation_alignment_weight=float(
                self._cfg.get(
                    'spatial_exploration_tree_online_belief_observation_alignment_weight',
                    1.0,
                )
            ),
            pairwise_revisit_weight=float(
                self._cfg.get(
                    'spatial_exploration_tree_online_pairwise_revisit_weight',
                    1.0,
                )
            ),
            pairwise_revisit_negative_samples=int(
                self._cfg.get(
                    'spatial_exploration_tree_online_pairwise_revisit_negative_samples',
                    8,
                )
            ),
            prioritized_replay_fraction=float(
                self._cfg.get(
                    'spatial_exploration_tree_online_prioritized_replay_fraction',
                    0.0,
                )
            ),
            prioritized_replay_alpha=float(
                self._cfg.get(
                    'spatial_exploration_tree_online_prioritized_replay_alpha',
                    0.6,
                )
            ),
            prioritized_replay_epsilon=float(
                self._cfg.get(
                    'spatial_exploration_tree_online_prioritized_replay_epsilon',
                    1e-3,
                )
            ),
            prioritized_replay_priority_mode=str(
                self._cfg.get(
                    'spatial_exploration_tree_online_prioritized_replay_priority_mode',
                    'belief_error',
                )
            ),
            revisit_history_scope=str(
                self._cfg.get(
                    'spatial_exploration_tree_online_revisit_history_scope',
                    'episode',
                )
            ),
            predictive_equivalence_weight=float(
                self._cfg.get(
                    'spatial_exploration_tree_online_predictive_equivalence_weight',
                    0.0,
                )
            ),
            predictive_equivalence_negative_margin=float(
                self._cfg.get(
                    'spatial_exploration_tree_online_predictive_equivalence_negative_margin',
                    0.2,
                )
            ),
        )
        return self._spatial_exploration_tree

    def consume_spatial_exploration_tree_records(self) -> List[dict]:
        planner = getattr(self, '_spatial_exploration_tree', None)
        return [] if planner is None else planner.consume_records()

    def get_spatial_exploration_tree_model(self):
        planner = self._ensure_spatial_exploration_tree()
        return None if planner is None else planner.model

    def train_spatial_exploration_tree(
            self,
            episodes: Sequence[dict],
            collection_phase: int,
    ) -> dict:
        planner = self._ensure_spatial_exploration_tree()
        if planner is None:
            return {}
        return planner.train_online(episodes, collection_phase)

    def _ensure_episodic_sequence_novelty_model(
            self,
    ) -> Optional[nn.Module]:
        if not self._episodic_sequence_novelty_enabled():
            return None
        if self._episodic_sequence_novelty_model is not None:
            return self._episodic_sequence_novelty_model
        observation_shape = tuple(self._cfg.model.observation_shape)
        if self._cfg.model.model_type != 'conv' or len(observation_shape) != 3:
            raise ValueError(
                'episodic_sequence_novelty currently requires convolutional image observations.'
            )
        initialization_seed = self._cfg.get(
            'episodic_sequence_novelty_initialization_seed', None
        )
        rng_state = torch.random.get_rng_state()
        try:
            if initialization_seed is not None:
                torch.manual_seed(int(initialization_seed))
            model_kind = str(
                self._cfg.get('episodic_sequence_novelty_model', 'recurrent')
            ).lower()
            common_kwargs = dict(
                observation_channels=int(observation_shape[0]),
                action_space_size=int(self._cfg.model.action_space_size),
                embedding_dim=int(
                    self._cfg.get('episodic_sequence_novelty_embedding_dim', 64)
                ),
                hidden_dim=int(
                    self._cfg.get('episodic_sequence_novelty_hidden_dim', 128)
                ),
            )
            if model_kind == 'recurrent':
                model = RecurrentSequenceNovelty(**common_kwargs)
            elif model_kind in {'recent_transformer', 'transformer'}:
                if self._episodic_sequence_continuation_enabled():
                    raise ValueError(
                        'Recent-history transformer novelty does not use the '
                        'recurrent continuation-value head.'
                    )
                model = RecentHistoryTransformerNovelty(
                    **common_kwargs,
                    history_window=int(
                        self._cfg.get(
                            'episodic_sequence_novelty_history_window', 40
                        )
                    ),
                    max_horizon=int(
                        self._cfg.get('episodic_sequence_novelty_horizon', 5)
                    ),
                    transformer_layers=int(
                        self._cfg.get(
                            'episodic_sequence_novelty_transformer_layers', 2
                        )
                    ),
                    transformer_heads=int(
                        self._cfg.get(
                            'episodic_sequence_novelty_transformer_heads', 4
                        )
                    ),
                    transformer_dropout=float(
                        self._cfg.get(
                            'episodic_sequence_novelty_transformer_dropout', 0.0
                        )
                    ),
                )
            elif model_kind in {'predictive_memory', 'reconstructive_memory'}:
                if self._episodic_sequence_continuation_enabled():
                    raise ValueError(
                        'Predictive episodic memory does not use the recurrent '
                        'continuation-value head.'
                    )
                predictive_kwargs = dict(
                    initial_distance_threshold=float(
                        self._cfg.get(
                            'episodic_predictive_memory_initial_distance_threshold',
                            0.08,
                        )
                    ),
                    initial_distance_temperature=float(
                        self._cfg.get(
                            'episodic_predictive_memory_initial_distance_temperature',
                            0.02,
                        )
                    ),
                    adaptive_distance_scale=bool(
                        self._cfg.get(
                            'episodic_predictive_memory_adaptive_distance_scale',
                            False,
                        )
                    ),
                    adaptive_threshold_ratio=float(
                        self._cfg.get(
                            'episodic_predictive_memory_adaptive_threshold_ratio',
                            0.5,
                        )
                    ),
                    adaptive_temperature_ratio=float(
                        self._cfg.get(
                            'episodic_predictive_memory_adaptive_temperature_ratio',
                            0.125,
                        )
                    ),
                    adaptive_minimum_scale=float(
                        self._cfg.get(
                            'episodic_predictive_memory_adaptive_minimum_scale',
                            0.02,
                        )
                    ),
                    adaptive_distinct_epsilon=float(
                        self._cfg.get(
                            'episodic_predictive_memory_adaptive_distinct_epsilon',
                            1e-6,
                        )
                    ),
                )
                if model_kind == 'reconstructive_memory':
                    model = ReconstructiveEpisodicMemory(
                        observation_shape=observation_shape,
                        action_space_size=common_kwargs['action_space_size'],
                        embedding_dim=common_kwargs['embedding_dim'],
                        hidden_dim=common_kwargs['hidden_dim'],
                        decoder_channels=int(
                            self._cfg.get(
                                'episodic_reconstructive_memory_decoder_channels',
                                64,
                            )
                        ),
                        **predictive_kwargs,
                    )
                else:
                    model = PredictiveEpisodicMemory(
                        **common_kwargs,
                        **predictive_kwargs,
                    )
            else:
                raise ValueError(
                    'episodic_sequence_novelty_model must be recurrent, '
                    'recent_transformer, predictive_memory, or '
                    'reconstructive_memory; '
                    f'got {model_kind!r}.'
                )
        finally:
            torch.random.set_rng_state(rng_state)
        self._episodic_sequence_novelty_model = model.to(self._cfg.device)
        self._episodic_sequence_novelty_optimizer = torch.optim.Adam(
            [
                parameter
                for parameter in self._episodic_sequence_novelty_model.parameters()
                if parameter.requires_grad
            ],
            lr=float(
                self._cfg.get('episodic_sequence_novelty_learning_rate', 3e-4)
            ),
            weight_decay=float(
                self._cfg.get('episodic_sequence_novelty_weight_decay', 1e-5)
            ),
        )
        return self._episodic_sequence_novelty_model

    def _episodic_sequence_novelty_tree_active(self) -> bool:
        if not self._episodic_sequence_novelty_enabled():
            return False
        activation_phase = max(
            1,
            int(self._cfg.get('episodic_sequence_novelty_activation_phase', 1)),
        )
        return int(self._episodic_sequence_novelty_trained_phase) >= activation_phase - 1

    def _episodic_sequence_continuation_enabled(self) -> bool:
        return bool(
            self._episodic_sequence_novelty_enabled()
            and self._cfg.get('episodic_sequence_continuation_value', False)
        )

    def _episodic_sequence_continuation_tree_active(self) -> bool:
        if not self._episodic_sequence_continuation_enabled():
            return False
        activation_phase = max(
            1,
            int(self._cfg.get('episodic_sequence_novelty_activation_phase', 1)),
        )
        return (
            int(self._episodic_sequence_continuation_trained_phase)
            >= activation_phase - 1
        )

    def _episodic_sequence_observations(self, observations: torch.Tensor) -> torch.Tensor:
        scale = float(
            self._cfg.get('episodic_sequence_novelty_observation_scale', 1.0)
        )
        if scale <= 0.0:
            raise ValueError('episodic_sequence_novelty_observation_scale must be positive.')
        return observations.float() / scale

    @staticmethod
    def _episodic_sequence_observation_key(observation: torch.Tensor) -> bytes:
        quantized = (
            observation.detach().float().clamp(0.0, 1.0).mul(255.0).round()
            .to(dtype=torch.uint8, device='cpu').contiguous().numpy()
        )
        return quantized.tobytes()

    def _prepare_collection_episodic_sequence_contexts(
            self,
            observations: torch.Tensor,
            ready_env_id: Sequence[int],
    ) -> List[RecurrentSequenceContext]:
        model = self._ensure_episodic_sequence_novelty_model()
        if model is None:
            return []
        model.eval()
        observations = self._episodic_sequence_observations(observations)
        recent_transformer = isinstance(model, RecentHistoryTransformerNovelty)
        contexts = []
        for root_index, env_id in enumerate(ready_env_id):
            env_id = int(env_id)
            observation = observations[root_index:root_index + 1]
            observation_key = self._episodic_sequence_observation_key(observation[0])
            context = self._collect_env_episodic_sequence_context.get(env_id)
            if context is None:
                context = model.start_context(observation).detach()
                self._collect_env_episodic_sequence_seen[env_id] = (
                    [observation_key] if recent_transformer else {observation_key}
                )
            else:
                pending = self._collect_env_episodic_sequence_pending.get(env_id)
                if pending is not None:
                    if recent_transformer:
                        seen = self._collect_env_episodic_sequence_seen.setdefault(
                            env_id, []
                        )
                        episodic_reward = float(observation_key not in seen)
                        seen.append(observation_key)
                        del seen[:-model.history_window]
                    else:
                        seen = self._collect_env_episodic_sequence_seen.setdefault(
                            env_id, set()
                        )
                        episodic_reward = float(observation_key not in seen)
                        seen.add(observation_key)
                    selected_factor = pending.get('predicted_factor', None)
                    if selected_factor is not None:
                        minimum_factor = float(
                            np.clip(
                                self._cfg.get(
                                    'episodic_sequence_novelty_minimum_reward_factor',
                                    0.05,
                                ),
                                0.0,
                                1.0,
                            )
                        )
                        denominator = max(1e-8, 1.0 - minimum_factor)
                        prediction = float(
                            np.clip(
                                (float(selected_factor) - minimum_factor) / denominator,
                                0.0,
                                1.0,
                            )
                        )
                        selected_stats = self._episodic_sequence_novelty_selected_stats
                        selected_stats['count'] += 1
                        selected_stats['prediction_sum'] += prediction
                        selected_stats['target_sum'] += episodic_reward
                        selected_stats['squared_error_sum'] += (
                            prediction - episodic_reward
                        ) ** 2
                        selected_stats['correct_sum'] += float(
                            (prediction >= 0.5) == (episodic_reward >= 0.5)
                        )
                        candidate_probabilities = [
                            float(value)
                            for value in pending.get('candidate_probabilities', [])
                        ]
                        legal_actions = [
                            int(action) for action in pending.get('legal_actions', [])
                        ]
                        selected_rank = int(
                            1 + sum(
                                candidate > prediction + 1e-12
                                for candidate in candidate_probabilities
                            )
                        )
                        candidate_max = float(
                            max(candidate_probabilities, default=prediction)
                        )
                        candidate_min = float(
                            min(candidate_probabilities, default=prediction)
                        )
                        selected_index = (
                            legal_actions.index(int(pending['action']))
                            if int(pending['action']) in legal_actions else -1
                        )
                        argmax_index = (
                            int(np.argmax(candidate_probabilities))
                            if candidate_probabilities else -1
                        )
                        intrinsic_visit_counts = [
                            int(value)
                            for value in pending.get(
                                'intrinsic_visit_counts', []
                            )
                        ]
                        self._episodic_sequence_novelty_root_action_records.append(
                            {
                                'env_id': env_id,
                                'tree_active': float(
                                    pending.get('tree_active', False)
                                ),
                                'trained_phase': int(
                                    pending.get('trained_phase', -1)
                                ),
                                'planned_action': int(
                                    pending.get('planned_action', pending['action'])
                                ),
                                'selected_action': int(pending['action']),
                                'random_action_overridden': float(
                                    pending.get('random_action_overridden', False)
                                ),
                                'legal_actions': legal_actions,
                                'candidate_probabilities': candidate_probabilities,
                                'candidate_factors': [
                                    float(value)
                                    for value in pending.get('candidate_factors', [])
                                ],
                                'intrinsic_visit_counts': intrinsic_visit_counts,
                                'intrinsic_total_scores': [
                                    float(value)
                                    for value in pending.get(
                                        'intrinsic_total_scores', []
                                    )
                                ],
                                'intrinsic_immediate_rewards': [
                                    float(value)
                                    for value in pending.get(
                                        'intrinsic_immediate_rewards', []
                                    )
                                ],
                                'selected_prediction': prediction,
                                'candidate_probability_mean': float(
                                    np.mean(candidate_probabilities)
                                    if candidate_probabilities else prediction
                                ),
                                'candidate_probability_min': candidate_min,
                                'candidate_probability_max': candidate_max,
                                'candidate_probability_spread': (
                                    candidate_max - candidate_min
                                ),
                                'selected_probability_gap_from_max': (
                                    candidate_max - prediction
                                ),
                                'selected_rank': selected_rank,
                                'selected_is_argmax': float(selected_rank == 1),
                                'novelty_argmax_action': int(
                                    legal_actions[argmax_index]
                                    if argmax_index >= 0 else pending['action']
                                ),
                                'selected_intrinsic_visit_count': int(
                                    intrinsic_visit_counts[selected_index]
                                    if 0 <= selected_index < len(
                                        intrinsic_visit_counts
                                    ) else 0
                                ),
                                'novelty_argmax_intrinsic_visit_count': int(
                                    intrinsic_visit_counts[argmax_index]
                                    if 0 <= argmax_index < len(
                                        intrinsic_visit_counts
                                    ) else 0
                                ),
                                'realized_novelty': episodic_reward,
                                'selected_squared_error': (
                                    prediction - episodic_reward
                                ) ** 2,
                                'selected_correct': float(
                                    (prediction >= 0.5)
                                    == (episodic_reward >= 0.5)
                                ),
                            }
                        )
                    context = model.advance_context(
                        context,
                        torch.as_tensor(
                            [int(pending['action'])],
                            dtype=torch.long,
                            device=observation.device,
                        ),
                        observation,
                        torch.as_tensor(
                            [episodic_reward],
                            dtype=torch.float32,
                            device=observation.device,
                        ),
                        torch.as_tensor(
                            [float(pending.get('extrinsic_reward', 0.0))],
                            dtype=torch.float32,
                            device=observation.device,
                        ),
                    ).detach()
                    self._collect_env_episodic_sequence_pending.pop(env_id, None)
            self._collect_env_episodic_sequence_context[env_id] = context
            contexts.append(context)
        return contexts

    def _episodic_sequence_root_action_predictions(
            self,
            root_contexts: Sequence[RecurrentSequenceContext],
            legal_actions: Sequence[Sequence[int]],
            path_probability_tables: Optional[Sequence[Dict[Tuple[int, ...], float]]] = None,
            prefix_cache: Optional[Any] = None,
            history_cache: Optional[RecentHistoryTransformerHistoryCache] = None,
    ) -> List[Dict[int, float]]:
        """Predict one-step novelty for every legal root action."""
        if not root_contexts:
            return [{} for _ in legal_actions]
        model = self._ensure_episodic_sequence_novelty_model()
        if model is None:
            return [{} for _ in legal_actions]
        root_indices = []
        action_paths = []
        flat_actions = []
        for root_index, actions in enumerate(legal_actions):
            for action in actions:
                root_indices.append(root_index)
                action_paths.append([int(action)])
                flat_actions.append(int(action))
        predictions = [{} for _ in legal_actions]
        if not action_paths:
            return predictions
        if history_cache is not None:
            model.eval()
            with torch.no_grad():
                logits = self._score_episodic_sequence_history_cached_paths(
                    model,
                    history_cache,
                    root_indices,
                    action_paths,
                )
                probabilities = torch.sigmoid(logits).detach().float().cpu().tolist()
            for root_index, action, probability in zip(
                    root_indices, flat_actions, probabilities
            ):
                predictions[root_index][action] = float(probability)
            return predictions
        if prefix_cache is not None:
            model.eval()
            with torch.no_grad():
                logits = self._score_episodic_sequence_cached_paths(
                    model,
                    prefix_cache,
                    root_indices,
                    action_paths,
                )
                probabilities = torch.sigmoid(logits).detach().float().cpu().tolist()
            for root_index, action, probability in zip(
                    root_indices, flat_actions, probabilities
            ):
                predictions[root_index][action] = float(probability)
            return predictions
        if path_probability_tables is not None:
            stats = self._episodic_sequence_novelty_inference_stats
            for root_index, action in zip(root_indices, flat_actions):
                probability = path_probability_tables[root_index].get((action,))
                stats['lookup_count'] += 1
                if probability is None:
                    stats['miss_count'] += 1
                    break
                predictions[root_index][action] = float(probability)
            else:
                return predictions
            predictions = [{} for _ in legal_actions]
        model.eval()
        with torch.no_grad():
            probabilities = torch.sigmoid(
                model.score_path_endpoints(
                    root_contexts,
                    root_indices,
                    action_paths,
                )
            ).detach().float().cpu().tolist()
        for root_index, action, probability in zip(
                root_indices, flat_actions, probabilities
        ):
            predictions[root_index][action] = float(probability)
        return predictions

    @staticmethod
    def _new_episodic_sequence_inference_stats() -> dict:
        return {
            'table_build_count': 0,
            'table_build_seconds': 0.0,
            'table_prefix_count': 0,
            'lookup_count': 0,
            'miss_count': 0,
            'cache_build_count': 0,
            'cache_build_seconds': 0.0,
            'cache_request_count': 0,
            'cache_hit_count': 0,
            'cache_path_eval_count': 0,
        }

    def _episodic_sequence_history_cache(
            self,
            root_contexts: Sequence[RecurrentSequenceContext],
    ) -> Optional[RecentHistoryTransformerHistoryCache]:
        mode = str(
            self._cfg.get('episodic_sequence_novelty_inference_mode', 'direct')
        )
        if mode != 'cached_suffix' or not root_contexts:
            return None
        model = self._ensure_episodic_sequence_novelty_model()
        if not isinstance(model, RecentHistoryTransformerNovelty):
            raise ValueError(
                "cached_suffix inference is supported only by recent_transformer."
            )
        model.eval()
        start = time.perf_counter()
        with torch.no_grad():
            history_cache = model.create_history_cache(root_contexts)
        stats = self._episodic_sequence_novelty_inference_stats
        stats['cache_build_count'] += 1
        stats['cache_build_seconds'] += time.perf_counter() - start
        return history_cache

    def _score_episodic_sequence_history_cached_paths(
            self,
            model: RecentHistoryTransformerNovelty,
            history_cache: RecentHistoryTransformerHistoryCache,
            root_indices: Sequence[int],
            action_paths: Sequence[Sequence[int]],
    ) -> torch.Tensor:
        normalized_paths = [tuple(int(action) for action in path) for path in action_paths]
        hit_count = sum(
            path in history_cache.logits[int(root_index)]
            for root_index, path in zip(root_indices, normalized_paths)
        )
        cached_path_count = sum(len(logits) for logits in history_cache.logits)
        logits = model.score_history_cached_path_endpoints(
            history_cache,
            root_indices,
            normalized_paths,
        )
        stats = self._episodic_sequence_novelty_inference_stats
        stats['cache_request_count'] += len(normalized_paths)
        stats['cache_hit_count'] += int(hit_count)
        stats['cache_path_eval_count'] += (
            sum(len(values) for values in history_cache.logits) - cached_path_count
        )
        return logits

    def _episodic_sequence_prefix_cache(
            self,
            root_contexts: Sequence[RecurrentSequenceContext],
    ) -> Optional[Any]:
        mode = str(
            self._cfg.get('episodic_sequence_novelty_inference_mode', 'direct')
        )
        if mode != 'incremental_cache' or not root_contexts:
            return None
        model = self._ensure_episodic_sequence_novelty_model()
        if not isinstance(model, RecentHistoryTransformerNovelty):
            raise ValueError(
                "incremental_cache inference is supported only by recent_transformer."
            )
        model.eval()
        start = time.perf_counter()
        with torch.no_grad():
            prefix_cache = model.create_prefix_cache(root_contexts)
        stats = self._episodic_sequence_novelty_inference_stats
        stats['cache_build_count'] += 1
        stats['cache_build_seconds'] += time.perf_counter() - start
        return prefix_cache

    def _score_episodic_sequence_cached_paths(
            self,
            model: RecentHistoryTransformerNovelty,
            prefix_cache: Any,
            root_indices: Sequence[int],
            action_paths: Sequence[Sequence[int]],
    ) -> torch.Tensor:
        normalized_paths = [tuple(int(action) for action in path) for path in action_paths]
        hit_count = sum(
            path in prefix_cache.logits[int(root_index)]
            for root_index, path in zip(root_indices, normalized_paths)
        )
        cached_path_count = sum(len(logits) for logits in prefix_cache.logits)
        logits = model.score_cached_path_endpoints(
            prefix_cache,
            root_indices,
            normalized_paths,
        )
        stats = self._episodic_sequence_novelty_inference_stats
        stats['cache_request_count'] += len(normalized_paths)
        stats['cache_hit_count'] += int(hit_count)
        stats['cache_path_eval_count'] += (
            sum(len(values) for values in prefix_cache.logits) - cached_path_count
        )
        return logits

    def _episodic_sequence_path_probability_tables(
            self,
            root_contexts: Sequence[RecurrentSequenceContext],
    ) -> Optional[List[Dict[Tuple[int, ...], float]]]:
        mode = str(
            self._cfg.get('episodic_sequence_novelty_inference_mode', 'direct')
        )
        if (
                mode != 'prefix_table'
                or not self._episodic_sequence_novelty_tree_active()
                or not root_contexts
        ):
            return None
        model = self._ensure_episodic_sequence_novelty_model()
        if not isinstance(model, RecentHistoryTransformerNovelty):
            raise ValueError(
                "prefix_table inference is supported only by recent_transformer."
            )
        max_horizon = max(
            1,
            int(self._cfg.get('episodic_sequence_novelty_horizon', 5)),
        )
        model.eval()
        start = time.perf_counter()
        with torch.no_grad():
            tables = model.all_action_prefix_probabilities(
                root_contexts,
                max_horizon=max_horizon,
            )
        stats = self._episodic_sequence_novelty_inference_stats
        stats['table_build_count'] += 1
        stats['table_build_seconds'] += time.perf_counter() - start
        stats['table_prefix_count'] += sum(len(table) for table in tables)
        return tables

    def consume_episodic_sequence_root_action_records(self) -> List[dict]:
        records = list(self._episodic_sequence_novelty_root_action_records)
        self._episodic_sequence_novelty_root_action_records = []
        return records

    def _episodic_sequence_path_reward_modifier_fn(
            self,
            root_contexts: Sequence[RecurrentSequenceContext],
            path_probability_tables: Optional[Sequence[Dict[Tuple[int, ...], float]]] = None,
            prefix_cache: Optional[Any] = None,
            history_cache: Optional[RecentHistoryTransformerHistoryCache] = None,
    ):
        if not self._episodic_sequence_novelty_tree_active() or not root_contexts:
            return None
        model = self._ensure_episodic_sequence_novelty_model()
        max_horizon = max(
            1,
            int(self._cfg.get('episodic_sequence_novelty_horizon', 5)),
        )
        minimum_factor = float(
            np.clip(
                self._cfg.get('episodic_sequence_novelty_minimum_reward_factor', 0.05),
                0.0,
                1.0,
            )
        )

        def modifier(root_indices, action_paths, _next_latent):
            factors = np.ones(len(action_paths), dtype=np.float32)
            valid_rows = [
                row for row, path in enumerate(action_paths)
                if 0 < len(path) <= max_horizon
            ]
            if not valid_rows:
                return factors
            if history_cache is not None:
                model.eval()
                with torch.no_grad():
                    logits = self._score_episodic_sequence_history_cached_paths(
                        model,
                        history_cache,
                        [root_indices[row] for row in valid_rows],
                        [action_paths[row] for row in valid_rows],
                    )
                    probabilities = torch.sigmoid(logits)
                    valid_factors = minimum_factor + (
                        1.0 - minimum_factor
                    ) * probabilities
                valid_factor_array = valid_factors.detach().float().cpu().numpy()
            elif prefix_cache is not None:
                model.eval()
                with torch.no_grad():
                    logits = self._score_episodic_sequence_cached_paths(
                        model,
                        prefix_cache,
                        [root_indices[row] for row in valid_rows],
                        [action_paths[row] for row in valid_rows],
                    )
                    probabilities = torch.sigmoid(logits)
                    valid_factors = minimum_factor + (
                        1.0 - minimum_factor
                    ) * probabilities
                valid_factor_array = valid_factors.detach().float().cpu().numpy()
            elif path_probability_tables is not None:
                inference_stats = self._episodic_sequence_novelty_inference_stats
                probabilities = []
                missing_rows = []
                for row in valid_rows:
                    root_index = int(root_indices[row])
                    path = tuple(int(action) for action in action_paths[row])
                    probability = path_probability_tables[root_index].get(path)
                    inference_stats['lookup_count'] += 1
                    if probability is None:
                        inference_stats['miss_count'] += 1
                        missing_rows.append(row)
                    probabilities.append(probability)
                if missing_rows:
                    raise KeyError(
                        f"Missing {len(missing_rows)} episodic novelty prefix-table entries."
                    )
                valid_factor_array = minimum_factor + (
                    1.0 - minimum_factor
                ) * np.asarray(probabilities, dtype=np.float32)
            else:
                model.eval()
                with torch.no_grad():
                    logits = model.score_path_endpoints(
                        root_contexts,
                        [root_indices[row] for row in valid_rows],
                        [action_paths[row] for row in valid_rows],
                    )
                    probabilities = torch.sigmoid(logits)
                    valid_factors = minimum_factor + (1.0 - minimum_factor) * probabilities
                valid_factor_array = valid_factors.detach().float().cpu().numpy()
            factors[np.asarray(valid_rows, dtype=np.int64)] = valid_factor_array
            for row, factor in zip(valid_rows, valid_factor_array):
                depth = int(len(action_paths[row]))
                stats = self._episodic_sequence_novelty_factor_stats.setdefault(
                    depth,
                    {
                        'count': 0,
                        'sum': 0.0,
                        'sum_sq': 0.0,
                        'min': float('inf'),
                        'max': float('-inf'),
                    },
                )
                factor = float(factor)
                stats['count'] += 1
                stats['sum'] += factor
                stats['sum_sq'] += factor * factor
                stats['min'] = min(stats['min'], factor)
                stats['max'] = max(stats['max'], factor)
            return factors

        return modifier

    def _episodic_sequence_leaf_value_modifier_fn(
            self,
            root_contexts: Sequence[RecurrentSequenceContext],
    ):
        if not self._episodic_sequence_continuation_tree_active() or not root_contexts:
            return None
        model = self._ensure_episodic_sequence_novelty_model()
        max_horizon = max(
            1,
            int(self._cfg.get('episodic_sequence_novelty_horizon', 5)),
        )
        minimum_factor = float(
            np.clip(
                self._cfg.get('episodic_sequence_novelty_minimum_reward_factor', 0.05),
                0.0,
                1.0,
            )
        )

        def modifier(root_indices, action_paths, _next_latent):
            factors = np.ones(len(action_paths), dtype=np.float32)
            valid_rows = [
                row for row, path in enumerate(action_paths)
                if 0 < len(path) <= max_horizon
            ]
            if not valid_rows:
                return factors
            model.eval()
            with torch.no_grad():
                logits = model.score_path_continuation_endpoints(
                    root_contexts,
                    [root_indices[row] for row in valid_rows],
                    [action_paths[row] for row in valid_rows],
                )
                probabilities = torch.sigmoid(logits)
                valid_factors = minimum_factor + (1.0 - minimum_factor) * probabilities
            valid_factor_array = valid_factors.detach().float().cpu().numpy()
            factors[np.asarray(valid_rows, dtype=np.int64)] = valid_factor_array
            for row, factor in zip(valid_rows, valid_factor_array):
                depth = int(len(action_paths[row]))
                stats = self._episodic_sequence_continuation_factor_stats.setdefault(
                    depth,
                    {
                        'count': 0,
                        'sum': 0.0,
                        'sum_sq': 0.0,
                        'min': float('inf'),
                        'max': float('-inf'),
                    },
                )
                factor = float(factor)
                stats['count'] += 1
                stats['sum'] += factor
                stats['sum_sq'] += factor * factor
                stats['min'] = min(stats['min'], factor)
                stats['max'] = max(stats['max'], factor)
            return factors

        return modifier

    def consume_episodic_sequence_novelty_factor_stats(self) -> List[dict]:
        rows = []
        active = float(self._episodic_sequence_novelty_tree_active())
        selected_stats = self._episodic_sequence_novelty_selected_stats
        selected_count = int(selected_stats['count'])
        selected_summary = {
            'selected_count': selected_count,
            'selected_prediction_mean': float(
                selected_stats['prediction_sum'] / max(selected_count, 1)
            ),
            'selected_novel_fraction': float(
                selected_stats['target_sum'] / max(selected_count, 1)
            ),
            'selected_brier': float(
                selected_stats['squared_error_sum'] / max(selected_count, 1)
            ),
            'selected_accuracy': float(
                selected_stats['correct_sum'] / max(selected_count, 1)
            ),
        }
        inference_stats = self._episodic_sequence_novelty_inference_stats
        inference_summary = {
            'inference_mode': str(
                self._cfg.get('episodic_sequence_novelty_inference_mode', 'direct')
            ),
            **inference_stats,
            'table_build_seconds_mean': float(
                inference_stats['table_build_seconds']
                / max(int(inference_stats['table_build_count']), 1)
            ),
            'cache_build_seconds_mean': float(
                inference_stats['cache_build_seconds']
                / max(int(inference_stats['cache_build_count']), 1)
            ),
        }
        all_depths = sorted(
            set(self._episodic_sequence_novelty_factor_stats)
            | set(self._episodic_sequence_continuation_factor_stats)
        )
        for depth in all_depths:
            stats = self._episodic_sequence_novelty_factor_stats.get(depth)
            continuation_stats = self._episodic_sequence_continuation_factor_stats.get(
                depth
            )
            if stats is None:
                stats = {
                    'count': 0,
                    'sum': 0.0,
                    'sum_sq': 0.0,
                    'min': 1.0,
                    'max': 1.0,
                }
            count = int(stats['count'])
            mean = float(stats['sum'] / max(count, 1))
            variance = max(0.0, float(stats['sum_sq'] / max(count, 1)) - mean ** 2)
            continuation_count = int(
                continuation_stats['count'] if continuation_stats is not None else 0
            )
            continuation_mean = float(
                continuation_stats['sum'] / max(continuation_count, 1)
                if continuation_stats is not None else 1.0
            )
            continuation_variance = max(
                0.0,
                float(
                    continuation_stats['sum_sq'] / max(continuation_count, 1)
                    - continuation_mean ** 2
                ) if continuation_stats is not None else 0.0,
            )
            rows.append(
                {
                    'depth': int(depth),
                    'count': count,
                    'factor_mean': mean,
                    'factor_std': math.sqrt(variance),
                    'factor_min': float(stats['min']),
                    'factor_max': float(stats['max']),
                    'tree_active': active,
                    'trained_phase': int(
                        self._episodic_sequence_novelty_trained_phase
                    ),
                    'continuation_count': continuation_count,
                    'continuation_factor_mean': continuation_mean,
                    'continuation_factor_std': math.sqrt(continuation_variance),
                    'continuation_factor_min': float(
                        continuation_stats['min']
                        if continuation_stats is not None else 1.0
                    ),
                    'continuation_factor_max': float(
                        continuation_stats['max']
                        if continuation_stats is not None else 1.0
                    ),
                    'continuation_tree_active': float(
                        self._episodic_sequence_continuation_tree_active()
                    ),
                    'continuation_trained_phase': int(
                        self._episodic_sequence_continuation_trained_phase
                    ),
                    **selected_summary,
                    **inference_summary,
                }
            )
        self._episodic_sequence_novelty_factor_stats = {}
        self._episodic_sequence_continuation_factor_stats = {}
        self._episodic_sequence_novelty_inference_stats = self._new_episodic_sequence_inference_stats()
        self._episodic_sequence_novelty_selected_stats = {
            'count': 0,
            'prediction_sum': 0.0,
            'target_sum': 0.0,
            'squared_error_sum': 0.0,
            'correct_sum': 0.0,
        }
        if not rows:
            rows.append(
                {
                    'depth': 0,
                    'count': 0,
                    'factor_mean': 1.0,
                    'factor_std': 0.0,
                    'factor_min': 1.0,
                    'factor_max': 1.0,
                    'tree_active': active,
                    'trained_phase': int(
                        self._episodic_sequence_novelty_trained_phase
                    ),
                    'continuation_count': 0,
                    'continuation_factor_mean': 1.0,
                    'continuation_factor_std': 0.0,
                    'continuation_factor_min': 1.0,
                    'continuation_factor_max': 1.0,
                    'continuation_tree_active': float(
                        self._episodic_sequence_continuation_tree_active()
                    ),
                    'continuation_trained_phase': int(
                        self._episodic_sequence_continuation_trained_phase
                    ),
                    **selected_summary,
                    **inference_summary,
                }
            )
        return rows

    def set_state_density_snapshot(
            self,
            snapshot_counts: np.ndarray,
            snapshot_phase: int,
    ) -> None:
        """Freeze the real-state density used for all intrinsic targets in one phase."""
        snapshot_counts = np.asarray(snapshot_counts)
        if snapshot_counts.ndim != 2:
            raise ValueError(
                f"State-density snapshot must be a 2-D count table, got {snapshot_counts.shape}."
            )
        self._state_density_snapshot_counts = torch.as_tensor(
            snapshot_counts.copy(), dtype=torch.float32, device=self._cfg.device
        )
        self._state_density_snapshot_phase = int(snapshot_phase)

    def _state_density_rewards_from_observations(
            self,
            observations: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Evaluate current phase density rewards for arbitrary real replay observations."""
        snapshot_counts = getattr(self, '_state_density_snapshot_counts', None)
        if snapshot_counts is None:
            return None, None
        if observations.shape[-1] < 2:
            raise ValueError(
                "State-density novelty requires position and velocity in the first two "
                f"observation components, got shape {tuple(observations.shape)}."
            )

        original_shape = observations.shape[:-1]
        flat_observations = observations.reshape(-1, observations.shape[-1]).float()
        position_min = float(self._cfg.get('state_density_position_min', -1.2))
        velocity_min = float(self._cfg.get('state_density_velocity_min', -0.07))
        position_bin_width = float(self._cfg.get('state_density_position_bin_width', 0.02))
        velocity_bin_width = float(self._cfg.get('state_density_velocity_bin_width', 0.002))
        if position_bin_width <= 0.0 or velocity_bin_width <= 0.0:
            raise ValueError('State-density bin widths must be positive.')

        position_indices = torch.floor(
            (flat_observations[:, 0] - position_min) / position_bin_width
        ).long().clamp(min=0, max=snapshot_counts.shape[0] - 1)
        velocity_indices = torch.floor(
            (flat_observations[:, 1] - velocity_min) / velocity_bin_width
        ).long().clamp(min=0, max=snapshot_counts.shape[1] - 1)

        effective_counts = flat_observations.new_zeros(flat_observations.shape[0])
        neighbor_radius = max(0, int(self._cfg.get('state_density_neighbor_radius', 1)))
        for position_offset in range(-neighbor_radius, neighbor_radius + 1):
            neighbor_position = position_indices + position_offset
            valid_position = (
                (neighbor_position >= 0) & (neighbor_position < snapshot_counts.shape[0])
            )
            for velocity_offset in range(-neighbor_radius, neighbor_radius + 1):
                neighbor_velocity = velocity_indices + velocity_offset
                valid = valid_position & (
                    (neighbor_velocity >= 0) & (neighbor_velocity < snapshot_counts.shape[1])
                )
                if bool(valid.any()):
                    effective_counts[valid] += snapshot_counts[
                        neighbor_position[valid], neighbor_velocity[valid]
                    ]

        rewards = float(self._cfg.get('state_density_reward_scale', 1.0)) / torch.sqrt(
            effective_counts + 1.0
        )
        reward_clip = self._cfg.get('state_density_reward_clip', 1.0)
        if reward_clip is not None and abs(float(reward_clip)) > 0.0:
            rewards = rewards.clamp(min=0.0, max=abs(float(reward_clip)))
        return rewards.reshape(original_shape).detach(), effective_counts.reshape(original_shape).detach()

    def _ensure_ensemble_value_calibration_state(self) -> None:
        ensemble_size = int(self._cfg.model.get('ensemble_size', 1))
        offsets = getattr(self, '_ensemble_value_calibration_offsets', None)
        if offsets is None or np.asarray(offsets).shape != (ensemble_size, ):
            self._ensemble_value_calibration_offsets = np.zeros(ensemble_size, dtype=np.float32)
            self._ensemble_value_calibration_phase = -1
            self._ensemble_value_calibration_sample_count = 0
            self._ensemble_value_calibration_stats = {}

    def _ensure_depth_disagreement_stats(self) -> DepthNormalizedBackupDisagreementStats:
        if not hasattr(self, '_depth_disagreement_stats'):
            self._depth_disagreement_stats = DepthNormalizedBackupDisagreementStats(self._cfg)
        return self._depth_disagreement_stats

    def finalize_depth_disagreement_phase(self, collection_phase: int) -> List[Dict[str, Any]]:
        return self._ensure_depth_disagreement_stats().finalize_phase(collection_phase)

    def _value_calibration_offsets(self, ensemble_size: Optional[int] = None) -> np.ndarray:
        self._ensure_ensemble_value_calibration_state()
        if ensemble_size is not None and int(ensemble_size) != self._ensemble_value_calibration_offsets.size:
            return np.zeros(int(ensemble_size), dtype=np.float32)
        if not bool(self._cfg.get('ensemble_value_calibration', False)):
            return np.zeros_like(self._ensemble_value_calibration_offsets)
        return self._ensemble_value_calibration_offsets.copy()

    def update_ensemble_value_calibration(
            self,
            observations: np.ndarray,
            collection_phase: int,
    ) -> Dict[str, Any]:
        """Freeze replay-anchored per-member value offsets for subsequent search calls."""
        self._ensure_ensemble_value_calibration_state()
        if not bool(self._cfg.get('ensemble_value_calibration', False)):
            self._ensemble_value_calibration_offsets.fill(0.0)
            return {'enabled': 0.0, 'sample_count': 0, 'offsets': self._value_calibration_offsets().tolist()}

        observations = np.asarray(observations)
        if observations.ndim < 2 or observations.shape[0] == 0:
            return {
                'enabled': 1.0,
                'sample_count': 0,
                'offsets': self._ensemble_value_calibration_offsets.astype(np.float32).tolist(),
            }

        model = self._learn_model
        ensemble_size = int(self._cfg.model.ensemble_size)
        inference_batch_size = max(
            1, int(self._cfg.get('ensemble_value_calibration_inference_batch_size', 1024))
        )
        member_value_chunks = []
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                for start in range(0, observations.shape[0], inference_batch_size):
                    obs_chunk, _ = prepare_obs(observations[start:start + inference_batch_size], self._cfg)
                    chunk_size = int(obs_chunk.shape[0])
                    repeated_obs = torch.cat([obs_chunk for _ in range(ensemble_size)], dim=0)
                    member_ids = [
                        member_id
                        for member_id in range(ensemble_size)
                        for _ in range(chunk_size)
                    ]
                    output = model.initial_inference(repeated_obs, member_id=member_ids)
                    values = self.value_inverse_scalar_transform_handle(output.value).detach()
                    values = values.reshape(ensemble_size, chunk_size, -1).squeeze(-1)
                    member_value_chunks.append(values.float().cpu().numpy())
        finally:
            if hasattr(model, 'set_active_member'):
                model.set_active_member(0)
            model.train(was_training)

        member_values = np.concatenate(member_value_chunks, axis=1).astype(np.float32)
        member_means = member_values.mean(axis=1)
        offsets = member_means - member_means.mean()
        calibrated_values = member_values - offsets[:, None]
        raw_state_std = member_values.std(axis=0)
        calibrated_state_std = calibrated_values.std(axis=0)

        self._ensemble_value_calibration_offsets = offsets.astype(np.float32)
        self._ensemble_value_calibration_phase = int(collection_phase)
        self._ensemble_value_calibration_sample_count = int(member_values.shape[1])
        stats = {
            'enabled': 1.0,
            'collection_phase': int(collection_phase),
            'sample_count': int(member_values.shape[1]),
            'offsets': offsets.astype(np.float32).tolist(),
            'member_value_means': member_means.astype(np.float32).tolist(),
            'raw_state_value_std_mean': float(raw_state_std.mean()),
            'raw_state_value_std_p90': float(np.percentile(raw_state_std, 90)),
            'calibrated_state_value_std_mean': float(calibrated_state_std.mean()),
            'calibrated_state_value_std_p90': float(np.percentile(calibrated_state_std, 90)),
            'value_mean': float(member_values.mean()),
            'value_min': float(member_values.min()),
            'value_max': float(member_values.max()),
        }
        self._ensemble_value_calibration_stats = dict(stats)
        return stats

    def _rnd_enabled(self) -> bool:
        return bool(self._cfg.get('rnd_exploration', False))

    def _use_learned_intrinsic_model(self) -> bool:
        return bool(self._cfg.get('rnd_use_learned_intrinsic_model', True))

    def _intrinsic_value_training_enabled(self) -> bool:
        source = self._intrinsic_reward_source()
        if source == 'episodic_novelty':
            return bool(self._cfg.get('episodic_novelty_diagnostics', False))
        if not self._use_learned_intrinsic_model():
            return False
        if source == 'rnd':
            return self._rnd_enabled()
        if source == 'dynamics_disagreement':
            return bool(self._cfg.get('dynamics_disagreement_train_intrinsic_value', False))
        if source == 'qstd':
            return bool(self._cfg.get('qstd_exploration', False))
        if source == 'state_value_disagreement':
            return int(self._cfg.model.get('ensemble_size', 1)) > 1
        if source == 'state_density':
            return True
        return False

    def _has_selected_intrinsic_rewards(self, current_batch) -> bool:
        return bool(self._cfg.get('store_selected_intrinsic_rewards', False))

    def _has_pure_exploration_episode_flags(self, current_batch) -> bool:
        return bool(self._cfg.get('intrinsic_value_train_on_pure_exploration_only', False))

    def _validate_current_batch_layout(self, current_batch) -> None:
        expected_size = (
            6
            + int(self._has_selected_intrinsic_rewards(current_batch))
            + int(self._has_pure_exploration_episode_flags(current_batch))
        )
        if len(current_batch) != expected_size:
            raise ValueError(
                f'Expected current batch with {expected_size} fields, got {len(current_batch)}. '
                'Check store_selected_intrinsic_rewards and '
                'intrinsic_value_train_on_pure_exploration_only.'
            )

    def _current_batch_core(self, current_batch):
        self._validate_current_batch_layout(current_batch)
        return [current_batch[0], current_batch[1], current_batch[2], *current_batch[-3:]]

    def _selected_intrinsic_reward_batch(self, current_batch) -> Optional[np.ndarray]:
        self._validate_current_batch_layout(current_batch)
        if not self._has_selected_intrinsic_rewards(current_batch):
            return None
        return current_batch[3]

    def _pure_exploration_episode_batch(self, current_batch) -> Optional[np.ndarray]:
        self._validate_current_batch_layout(current_batch)
        if not self._has_pure_exploration_episode_flags(current_batch):
            return None
        index = 3 + int(self._has_selected_intrinsic_rewards(current_batch))
        return current_batch[index]

    def _strip_selected_intrinsic_rewards_from_train_data(self, data):
        current_batch, target_batch = data
        if (
                not self._has_selected_intrinsic_rewards(current_batch)
                and not self._has_pure_exploration_episode_flags(current_batch)
        ):
            self._validate_current_batch_layout(current_batch)
            return data
        return [self._current_batch_core(current_batch), target_batch]

    def _latent_novelty_enabled(self, model: Optional[nn.Module] = None) -> bool:
        if not bool(self._cfg.get('latent_novelty_diagnostics', False)):
            return False
        if int(self._cfg.model.get('ensemble_size', 1)) < 2:
            return False
        if model is None:
            model = getattr(self, '_learn_model', None)
        return model is not None and hasattr(model, 'aligned_latent_embedding') and getattr(
            model, 'latent_novelty_heads', None
        ) is not None

    def _ensure_latent_novelty_optimizer(self):
        if self._latent_novelty_optimizer is not None:
            return self._latent_novelty_optimizer
        if not self._latent_novelty_enabled(self._learn_model):
            return None
        params = self._learn_model.latent_novelty_parameters()
        if not params:
            return None
        self._latent_novelty_optimizer = torch.optim.Adam(
            params,
            lr=float(self._cfg.get('latent_novelty_learning_rate', 3e-4)),
            weight_decay=float(self._cfg.get('weight_decay', 0.0)),
        )
        return self._latent_novelty_optimizer

    @staticmethod
    def _latent_novelty_pairwise_disagreement(embeddings: torch.Tensor) -> torch.Tensor:
        ensemble_size = embeddings.shape[0]
        if ensemble_size < 2 or embeddings.shape[-1] == 0:
            return embeddings.new_zeros((embeddings.shape[1], ))
        distances = []
        for member_i in range(ensemble_size):
            for member_j in range(member_i + 1, ensemble_size):
                distances.append((embeddings[member_i] - embeddings[member_j]).pow(2).sum(dim=-1))
        return torch.stack(distances, dim=0).mean(dim=0)

    def _latent_novelty_embeddings_from_obs(
            self,
            model: nn.Module,
            obs_batch: torch.Tensor,
            with_prior: bool = True,
    ) -> Optional[torch.Tensor]:
        if not self._latent_novelty_enabled(model):
            return None
        ensemble_size = int(self._cfg.model.ensemble_size)
        batch_size = int(obs_batch.shape[0])
        if batch_size <= 0:
            return None
        member_ids = [
            member_id
            for member_id in range(ensemble_size)
            for _ in range(batch_size)
        ]
        repeated_obs = self._repeat_tensor_for_members(obs_batch, ensemble_size)
        was_training = model.training
        try:
            model.eval()
            with torch.no_grad():
                latent_state = model.initial_inference(repeated_obs, member_id=member_ids).latent_state.detach()
            embeddings = model.aligned_latent_embedding(
                latent_state,
                member_id=member_ids,
                with_prior=with_prior,
                normalize=True,
                detach_latent=True,
            )
        finally:
            model.train(was_training)
        return embeddings.reshape(ensemble_size, batch_size, -1)

    def _latent_novelty_contrastive_loss(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        ensemble_size, batch_size, _ = embeddings.shape
        disagreement = self._latent_novelty_pairwise_disagreement(embeddings)
        center = embeddings.mean(dim=0, keepdim=True)
        alignment_loss = (embeddings - center).pow(2).sum(dim=-1).mean()

        losses = []
        same_cosines = []
        different_cosines = []
        labels = torch.arange(batch_size, device=embeddings.device)
        temperature = max(1e-6, float(self._cfg.get('latent_novelty_temperature', 0.1)))
        different_mask = ~torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
        for member_i in range(ensemble_size):
            for member_j in range(member_i + 1, ensemble_size):
                cosine = embeddings[member_i] @ embeddings[member_j].transpose(0, 1)
                same_cosines.append(torch.diag(cosine).mean())
                if batch_size > 1:
                    different_cosines.append(cosine[different_mask].mean())
                    logits = cosine / temperature
                    losses.append(F.cross_entropy(logits, labels))
                    losses.append(F.cross_entropy(logits.transpose(0, 1), labels))

        if losses:
            contrastive_loss = torch.stack(losses).mean()
        else:
            contrastive_loss = alignment_loss
        loss = contrastive_loss + float(self._cfg.get('latent_novelty_alignment_loss_weight', 0.1)) * alignment_loss
        same_cosine = torch.stack(same_cosines).mean() if same_cosines else embeddings.new_tensor(0.0)
        different_cosine = torch.stack(different_cosines).mean() if different_cosines else embeddings.new_tensor(0.0)
        log = {
            'latent_novelty/contrastive_loss': float(contrastive_loss.detach().item()),
            'latent_novelty/alignment_loss': float(alignment_loss.detach().item()),
            'latent_novelty/state_disagreement_mean': float(disagreement.detach().mean().item()),
            'latent_novelty/state_disagreement_std': float(disagreement.detach().std(unbiased=False).item()),
            'latent_novelty/state_disagreement_max': float(disagreement.detach().max().item()),
            'latent_novelty/same_obs_cosine_mean': float(same_cosine.detach().item()),
            'latent_novelty/different_obs_cosine_mean': float(different_cosine.detach().item()),
        }
        return loss, log

    def _train_latent_novelty_from_replay_batch(self, data) -> dict:
        if not self._latent_novelty_enabled(self._learn_model):
            return {}
        optimizer = self._ensure_latent_novelty_optimizer()
        if optimizer is None:
            return {}

        current_batch = data[0]
        obs_seq = self._extract_observation_sequence(current_batch[0])
        if obs_seq is None:
            return {}
        mask_batch = current_batch[2]
        valid_obs_mask = torch.ones(obs_seq.shape[:2], dtype=torch.bool, device=obs_seq.device)
        if mask_batch is not None:
            mask_tensor = torch.as_tensor(mask_batch, dtype=torch.bool, device=obs_seq.device)
            for obs_offset in range(1, obs_seq.shape[1]):
                reward_offset = obs_offset - 1
                if reward_offset < mask_tensor.shape[1]:
                    valid_obs_mask[:, obs_offset] = mask_tensor[:, reward_offset]
                else:
                    valid_obs_mask[:, obs_offset] = False
        obs_batch = obs_seq[valid_obs_mask].contiguous()
        if obs_batch.numel() == 0:
            return {}
        train_batch_size = int(self._cfg.get('latent_novelty_train_batch_size', 512))
        if obs_batch.shape[0] > train_batch_size:
            indices = torch.randperm(obs_batch.shape[0], device=obs_batch.device)[:train_batch_size]
            obs_batch = obs_batch.index_select(0, indices)

        embeddings = self._latent_novelty_embeddings_from_obs(self._learn_model, obs_batch, with_prior=True)
        if embeddings is None:
            return {}
        loss, log = self._latent_novelty_contrastive_loss(embeddings)
        optimizer.zero_grad()
        loss.backward()
        total_grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(
            self._learn_model.latent_novelty_parameters(), self._cfg.grad_clip_value
        )
        optimizer.step()

        with torch.no_grad():
            no_prior_embeddings = self._latent_novelty_embeddings_from_obs(
                self._learn_model, obs_batch, with_prior=False
            )
            if no_prior_embeddings is not None:
                no_prior_disagreement = self._latent_novelty_pairwise_disagreement(no_prior_embeddings)
                log['latent_novelty/state_disagreement_no_prior_mean'] = float(
                    no_prior_disagreement.mean().item()
                )
                log['latent_novelty/state_disagreement_no_prior_max'] = float(
                    no_prior_disagreement.max().item()
                )

        log.update(
            {
                'latent_novelty/loss': float(loss.detach().item()),
                'latent_novelty/grad_norm_before_clip': float(total_grad_norm_before_clip.item()),
                'latent_novelty/learning_rate': float(optimizer.param_groups[0]['lr']),
                'latent_novelty/train_sample_count': float(obs_batch.shape[0]),
                'latent_novelty/prior_scale': float(getattr(self._learn_model, 'latent_novelty_prior_scale', 0.0)),
            }
        )
        return log

    def _latent_novelty_root_disagreement(
            self,
            model: nn.Module,
            latent_state_roots: torch.Tensor,
    ) -> Optional[List[float]]:
        if not self._latent_novelty_enabled(model):
            return None
        active_env_num = int(latent_state_roots.shape[0])
        ensemble_size = int(latent_state_roots.shape[1])
        flat_latents = latent_state_roots.reshape(active_env_num * ensemble_size, *latent_state_roots.shape[2:])
        member_ids = [
            member_id
            for _ in range(active_env_num)
            for member_id in range(ensemble_size)
        ]
        embeddings = model.aligned_latent_embedding(
            flat_latents,
            member_id=member_ids,
            with_prior=True,
            normalize=True,
            detach_latent=True,
        )
        embeddings = embeddings.reshape(active_env_num, ensemble_size, -1).transpose(0, 1)
        disagreement = self._latent_novelty_pairwise_disagreement(embeddings)
        return [float(value) for value in disagreement.detach().cpu().tolist()]

    def _ensure_rnd_model(self, input_tensor: torch.Tensor) -> None:
        if not self._rnd_enabled() or self._rnd_model is not None:
            return
        if input_tensor.dim() < 2:
            raise ValueError(
                f'RND inputs require a batch dimension, got {tuple(input_tensor.shape)}.'
            )
        input_shape = tuple(int(size) for size in input_tensor.shape[1:])
        model_type = str(self._cfg.get('rnd_model_type', 'auto')).lower()
        if model_type == 'auto':
            model_type = 'conv' if input_tensor.dim() == 4 else 'mlp'
        if model_type not in {'mlp', 'conv'}:
            raise ValueError(
                f"rnd_model_type must be 'auto', 'mlp', or 'conv', got {model_type!r}."
            )
        if model_type == 'conv' and input_tensor.dim() != 4:
            raise ValueError(
                'Convolutional RND requires [batch, channels, height, width] inputs, '
                f'got {tuple(input_tensor.shape)}.'
            )
        initialization_seed = self._cfg.get('rnd_initialization_seed', None)

        def build_rnd_model() -> nn.Module:
            if model_type == 'conv':
                if (
                        self._cfg.get('rnd_observation_low', None) is not None
                        or self._cfg.get('rnd_observation_high', None) is not None
                ):
                    raise ValueError(
                        'Convolutional RND normalizes image ranges internally; fixed vector '
                        'bounds are not supported.'
                    )
                return _ImageRND(
                    input_channels=input_shape[0],
                    hidden_dim=int(self._cfg.get('rnd_hidden_dim', 128)),
                    feature_dim=int(self._cfg.get('rnd_feature_dim', 64)),
                    input_scale=float(
                        self._cfg.get('rnd_image_input_scale', 1.0)
                    ),
                )
            input_dim = math.prod(input_shape)
            return _LatentRND(
                input_dim=input_dim,
                hidden_dim=int(self._cfg.get('rnd_hidden_dim', 128)),
                feature_dim=int(self._cfg.get('rnd_feature_dim', 64)),
                input_low=self._cfg.get('rnd_observation_low', None),
                input_high=self._cfg.get('rnd_observation_high', None),
            )

        if initialization_seed is None:
            rnd_model = build_rnd_model()
        else:
            # Keep optional modules from changing the RND initialization stream.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(initialization_seed))
                rnd_model = build_rnd_model()
        self._rnd_model = rnd_model.to(self._cfg.device)
        self._rnd_input_shape = input_shape
        self._rnd_optimizer = torch.optim.Adam(
            self._rnd_model.predictor.parameters(),
            lr=float(self._cfg.get('rnd_learning_rate', 1e-4)),
        )
        self._sync_rnd_target_snapshot(force=True)

    def _sync_rnd_target_snapshot(self, force: bool = False) -> None:
        if not self._rnd_enabled() or self._rnd_model is None:
            return
        if not bool(self._cfg.get('rnd_target_snapshot', True)):
            self._rnd_target_snapshot_model = None
            return
        if bool(self._cfg.get('rnd_target_snapshot_refresh_per_collect', False)) and not force:
            return
        update_count = int(getattr(self, '_learn_update_count', 0))
        update_interval = self._cfg.get('rnd_target_snapshot_update_interval', None)
        if update_interval is None:
            update_interval = self._cfg.get('update_per_collect', 1)
        if update_interval is None:
            update_interval = 1
        update_interval = max(1, int(update_interval))
        should_update = (
            force
            or self._rnd_target_snapshot_model is None
            or self._rnd_target_snapshot_last_update < 0
            or update_count - self._rnd_target_snapshot_last_update >= update_interval
        )
        if not should_update:
            return
        self._rnd_target_snapshot_model = copy.deepcopy(self._rnd_model).to(self._cfg.device)
        self._rnd_target_snapshot_model.eval()
        for param in self._rnd_target_snapshot_model.parameters():
            param.requires_grad_(False)
        self._rnd_target_snapshot_last_update = update_count

    def _rnd_landscape_query_enabled(self) -> bool:
        return bool(self._cfg.get('rnd_landscape_query', False))

    def _rnd_landscape_query_space(self) -> str:
        query_space = str(
            self._cfg.get('rnd_landscape_query_space', 'physical_2d')
        ).lower()
        if query_space not in {'physical_2d', 'compact_image'}:
            raise ValueError(
                "rnd_landscape_query_space must be 'physical_2d' or "
                f"'compact_image', got {query_space!r}."
            )
        return query_space

    def _rnd_auxiliary_decoder_enabled(self) -> bool:
        return bool(self._cfg.get('rnd_auxiliary_decoder', False))

    def _ensure_rnd_auxiliary_decoder_model(
            self,
            latent_shape: Sequence[int],
    ) -> Optional[SharedRNDAuxiliaryDecoder]:
        if not self._rnd_auxiliary_decoder_enabled():
            return None
        if self._intrinsic_reward_source() != 'rnd':
            raise ValueError(
                "rnd_auxiliary_decoder requires intrinsic_reward_source='rnd'."
            )
        if bool(self._cfg.get('intrinsic_head_only', True)) and bool(
                self._cfg.get('rnd_auxiliary_decoder_train_world_model', True)
        ):
            raise ValueError(
                'rnd_auxiliary_decoder_train_world_model=True requires '
                'intrinsic_head_only=False.'
            )
        latent_shape = tuple(int(size) for size in latent_shape)
        if self._rnd_auxiliary_decoder_model is not None:
            if self._rnd_auxiliary_decoder_model.latent_shape != latent_shape:
                raise ValueError(
                    'RND auxiliary latent shape changed from '
                    f'{self._rnd_auxiliary_decoder_model.latent_shape} to {latent_shape}.'
                )
            self._attach_rnd_auxiliary_decoder_to_main_optimizer()
            return self._rnd_auxiliary_decoder_model

        initialization_seed = self._cfg.get(
            'rnd_auxiliary_decoder_initialization_seed', None
        )

        def build_model() -> SharedRNDAuxiliaryDecoder:
            return SharedRNDAuxiliaryDecoder(
                latent_shape=latent_shape,
                hidden_dim=int(
                    self._cfg.get('rnd_auxiliary_decoder_hidden_dim', 64)
                ),
                target_transform=str(
                    self._cfg.get('rnd_auxiliary_decoder_target_transform', 'log')
                ),
                target_log_reference=float(
                    self._cfg.get(
                        'rnd_auxiliary_decoder_target_log_reference', 1.0
                    )
                ),
                error_epsilon=float(
                    self._cfg.get('rnd_auxiliary_decoder_error_epsilon', 1e-10)
                ),
                max_raw_error=self._cfg.get(
                    'rnd_auxiliary_decoder_max_raw_error', None
                ),
                normalize_latent=bool(
                    self._cfg.get('rnd_auxiliary_decoder_normalize_latent', True)
                ),
            )

        if initialization_seed is None:
            model = build_model()
        else:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(initialization_seed))
                model = build_model()
        self._rnd_auxiliary_decoder_model = model.to(self._cfg.device)
        self._attach_rnd_auxiliary_decoder_to_main_optimizer()
        return self._rnd_auxiliary_decoder_model

    def _attach_rnd_auxiliary_decoder_to_main_optimizer(self) -> None:
        optimizer = getattr(self, '_optimizer', None)
        model = getattr(self, '_rnd_auxiliary_decoder_model', None)
        if optimizer is None or model is None:
            return
        optimized_parameter_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group['params']
        }
        new_parameters = [
            parameter
            for parameter in model.parameters()
            if id(parameter) not in optimized_parameter_ids
        ]
        if new_parameters:
            optimizer.add_param_group(
                {
                    'params': new_parameters,
                    'lr': float(
                        self._cfg.get(
                            'rnd_auxiliary_decoder_learning_rate', 3e-4
                        )
                    ),
                    'weight_decay': float(
                        self._cfg.get('rnd_auxiliary_decoder_weight_decay', 0.0)
                    ),
                }
            )
            self._register_added_optimizer_group_with_scheduler()
        parameter_ids = {id(parameter) for parameter in model.parameters()}
        self._rnd_auxiliary_decoder_optimizer_group_index = next(
            (
                index
                for index, group in enumerate(optimizer.param_groups)
                if any(id(parameter) in parameter_ids for parameter in group['params'])
            ),
            None,
        )

    def _rnd_landscape_query_tree_rewards_enabled(self) -> bool:
        return (
            self._rnd_landscape_query_enabled()
            and bool(self._cfg.get('rnd_landscape_query_use_tree_rewards', True))
        )

    def _ensure_rnd_landscape_query_model(self) -> Optional[nn.Module]:
        if not self._rnd_landscape_query_enabled():
            return None
        if self._intrinsic_reward_source() != 'rnd':
            raise ValueError("rnd_landscape_query requires intrinsic_reward_source='rnd'.")
        if not bool(self._cfg.get('rnd_target_snapshot', True)):
            raise ValueError("rnd_landscape_query requires rnd_target_snapshot=True.")
        if bool(self._cfg.get('intrinsic_head_only', True)):
            raise ValueError(
                "rnd_landscape_query requires intrinsic_head_only=False for joint h/g training."
            )
        query_space = self._rnd_landscape_query_space()
        if (
                query_space == 'physical_2d'
                and int(self._cfg.get('rnd_observation_dim', 0) or 0) != 2
        ):
            raise ValueError(
                "physical_2d RND landscape queries require rnd_observation_dim=2."
            )
        if (
                query_space == 'compact_image'
                and str(self._cfg.model.get('model_type', '')) not in {'conv', 'conv_context'}
        ):
            raise ValueError(
                "compact_image RND landscape queries require a convolutional model."
            )
        if self._rnd_landscape_query_model is None:
            latent_dim = int(self._cfg.model.get('latent_state_dim', 0) or 0)
            if latent_dim <= 0 and query_space == 'compact_image':
                observation_shape = tuple(
                    int(size) for size in self._cfg.model.observation_shape
                )
                latent_height = int(observation_shape[-2])
                latent_width = int(observation_shape[-1])
                if bool(self._cfg.model.get('downsample', True)):
                    if latent_height in {84, 96}:
                        latent_height = math.ceil(latent_height / 16)
                        latent_width = math.ceil(latent_width / 16)
                    elif latent_height == 64:
                        latent_height = math.ceil(latent_height / 8)
                        latent_width = math.ceil(latent_width / 8)
                latent_dim = (
                    int(self._cfg.model.get('num_channels', 0))
                    * latent_height
                    * latent_width
                )
            if latent_dim <= 0:
                raise ValueError(
                    "rnd_landscape_query requires model.latent_state_dim for its latent reader."
                )
            initialization_seed = self._cfg.get(
                'rnd_landscape_query_initialization_seed', None
            )

            def common_query_kwargs() -> Dict[str, Any]:
                return dict(
                    latent_dim=latent_dim,
                    query_count=int(self._cfg.get('rnd_landscape_query_count', 16)),
                    hidden_dim=int(self._cfg.get('rnd_landscape_query_hidden_dim', 64)),
                    reader_use_context=bool(
                        self._cfg.get('rnd_landscape_query_reader_use_context', True)
                    ),
                    ensemble_size=int(self._cfg.model.get('ensemble_size', 1)),
                    member_specific_reader=bool(
                        self._cfg.get('rnd_landscape_query_member_specific_reader', False)
                    ),
                    reader_architecture=str(
                        self._cfg.get(
                            'rnd_landscape_query_reader_architecture', 'landscape'
                        )
                    ),
                    reader_hidden_channels=tuple(
                        self._cfg.get(
                            'rnd_landscape_query_reader_hidden_channels', (32,)
                        )
                    ),
                    reader_norm_type=self._cfg.get(
                        'rnd_landscape_query_reader_norm_type',
                        self._cfg.model.get('norm_type', 'BN'),
                    ),
                    reader_last_linear_layer_init_zero=bool(
                        self._cfg.get(
                            'rnd_landscape_query_reader_last_linear_layer_init_zero',
                            False,
                        )
                    ),
                    target_transform=str(
                        self._cfg.get('rnd_landscape_query_target_transform', 'log')
                    ),
                    target_log_reference=float(
                        self._cfg.get(
                            'rnd_landscape_query_target_log_reference', 1.0
                        )
                    ),
                    normalized_raw_clip=float(
                        self._cfg.get('rnd_landscape_query_normalized_raw_clip', 10.0)
                    ),
                    max_raw_error=self._cfg.get(
                        'rnd_landscape_query_max_raw_error', None
                    ),
                    position_bounds=(-1.2, 0.6),
                    velocity_bounds=(-0.07, 0.07),
                    error_epsilon=float(
                        self._cfg.get(
                            'rnd_landscape_query_error_epsilon', 1e-10
                        )
                    ),
                )

            def build_query_model() -> nn.Module:
                kwargs = common_query_kwargs()
                if query_space == 'compact_image':
                    return CompactImageRNDLandscapeQuery(
                        observation_shape=tuple(
                            int(size) for size in self._cfg.model.observation_shape
                        ),
                        query_code_dim=int(
                            self._cfg.get('rnd_landscape_query_code_dim', 32)
                        ),
                        reconstruction_loss_weight=float(
                            self._cfg.get(
                                'rnd_landscape_query_reconstruction_loss_weight',
                                1.0,
                            )
                        ),
                        **kwargs,
                    )
                return MountainCarRNDLandscapeQuery(**kwargs)

            if initialization_seed is None:
                query_model = build_query_model()
            else:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(int(initialization_seed))
                    query_model = build_query_model()
            self._rnd_landscape_query_model = query_model.to(self._cfg.device)
        self._attach_rnd_landscape_query_to_main_optimizer()
        return self._rnd_landscape_query_model

    def _attach_rnd_landscape_query_to_main_optimizer(self) -> None:
        """Give the query model one disjoint parameter group in MuZero's optimizer."""
        optimizer = getattr(self, '_optimizer', None)
        model = getattr(self, '_rnd_landscape_query_model', None)
        if optimizer is None or model is None:
            return
        optimized_parameter_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group['params']
        }
        new_parameters = [
            parameter
            for parameter in model.parameters()
            if id(parameter) not in optimized_parameter_ids
        ]
        if new_parameters:
            optimizer.add_param_group(
                {
                    'params': new_parameters,
                    'lr': float(
                        self._cfg.get('rnd_landscape_query_learning_rate', 3e-4)
                    ),
                    'weight_decay': float(
                        self._cfg.get('rnd_landscape_query_weight_decay', 0.0)
                    ),
                }
            )
            self._register_added_optimizer_group_with_scheduler()
        query_parameter_ids = {id(parameter) for parameter in model.parameters()}
        self._rnd_landscape_query_optimizer_group_index = next(
            (
                index
                for index, group in enumerate(optimizer.param_groups)
                if any(id(parameter) in query_parameter_ids for parameter in group['params'])
            ),
            None,
        )

    def _register_added_optimizer_group_with_scheduler(self) -> None:
        scheduler = getattr(self, 'lr_scheduler', None)
        optimizer = getattr(self, '_optimizer', None)
        if scheduler is None or optimizer is None:
            return
        while len(scheduler.base_lrs) < len(optimizer.param_groups):
            group = optimizer.param_groups[len(scheduler.base_lrs)]
            group.setdefault('initial_lr', float(group['lr']))
            scheduler.base_lrs.append(float(group['lr']))
            if hasattr(scheduler, 'lr_lambdas') and scheduler.lr_lambdas:
                scheduler.lr_lambdas.append(scheduler.lr_lambdas[0])
            if hasattr(scheduler, '_last_lr'):
                scheduler._last_lr.append(float(group['lr']))

    def _attach_intrinsic_heads_to_main_optimizer(self) -> None:
        """Keep intrinsic heads at their configured LR without duplicate ownership."""
        optimizer = getattr(self, '_optimizer', None)
        if optimizer is None:
            return
        parameters = self._intrinsic_head_parameters()
        if not parameters:
            return
        parameter_ids = {id(parameter) for parameter in parameters}
        for group in optimizer.param_groups:
            group['params'] = [
                parameter
                for parameter in group['params']
                if id(parameter) not in parameter_ids
            ]
        learning_rate = self._cfg.get('intrinsic_learning_rate', None)
        if learning_rate is None:
            learning_rate = self._cfg.learning_rate
        optimizer.add_param_group(
            {
                'params': parameters,
                'lr': float(learning_rate),
                'weight_decay': float(self._cfg.get('weight_decay', 0.0)),
            }
        )
        self._register_added_optimizer_group_with_scheduler()
        self._intrinsic_head_optimizer_group_index = len(optimizer.param_groups) - 1

    def _rnd_landscape_encode_query_states(
            self,
            world_model: nn.Module,
            query_states: torch.Tensor,
    ) -> torch.Tensor:
        """Encode observation-space queries with every MuZero representation."""
        if query_states.dim() < 3:
            raise ValueError(
                "query_states must have shape [batch, queries, ...], got "
                f"{tuple(query_states.shape)}."
            )
        batch_size, query_count = query_states.shape[:2]
        ensemble_size = int(self._cfg.model.ensemble_size)
        flat_queries = query_states.reshape(
            batch_size * query_count, *query_states.shape[2:]
        )
        observation_batch = torch.cat(
            [flat_queries for _ in range(ensemble_size)], dim=0
        )
        member_ids = [
            member_id
            for member_id in range(ensemble_size)
            for _ in range(batch_size * query_count)
        ]
        restore_training = bool(
            self._rnd_landscape_query_space() == 'compact_image'
            and world_model.training
        )
        if restore_training:
            world_model.eval()
        try:
            encoded = world_model.initial_inference(
                observation_batch, member_id=member_ids
            ).latent_state
        finally:
            if restore_training:
                world_model.train()
        encoded = encoded.reshape(
            ensemble_size, batch_size, query_count, *encoded.shape[1:]
        )
        return encoded.transpose(0, 1).transpose(1, 2).contiguous()

    def refresh_rnd_target_snapshot_for_collection(
            self,
            collection_completed: bool = False,
    ) -> Dict[str, float]:
        """Freeze the current RND predictor for one collection/learning phase."""
        if collection_completed and self._rnd_landscape_query_enabled():
            self._rnd_landscape_query_snapshot_count += 1
        if bool(self._cfg.get('rnd_target_snapshot_refresh_per_collect', False)):
            self._sync_rnd_target_snapshot(force=True)
        return {}

    def refresh_rnd_landscape_query_target_scale(
            self,
            observations,
            collection_phase: int = -1,
    ) -> Dict[str, float]:
        """Freeze one normalized-raw scale for the subsequent learner phase."""
        model = self._ensure_rnd_landscape_query_model()
        if model is None or model.target_transform != 'normalized_raw':
            return {}
        snapshot = self._rnd_target_model_for_targets()
        if snapshot is None:
            return {}

        observation_tensor = torch.as_tensor(
            np.asarray(observations), dtype=torch.float32, device=self._cfg.device
        )
        if observation_tensor.numel() == 0:
            return {}
        selected = self._select_rnd_observation_features(observation_tensor)
        selected = selected.reshape(-1, *selected.shape[-3:]) if (
            self._rnd_landscape_query_space() == 'compact_image'
        ) else selected.reshape(-1, 2)
        inference_batch_size = max(
            1, int(self._cfg.get('rnd_landscape_query_scale_inference_batch_size', 2048))
        )
        error_parts = []
        with torch.no_grad():
            for start in range(0, selected.shape[0], inference_batch_size):
                error_parts.append(
                    snapshot.error(selected[start:start + inference_batch_size]).reshape(-1)
                )
        errors = torch.cat(error_parts).float()
        errors = errors[torch.isfinite(errors)]
        if errors.numel() == 0:
            return {}

        raw_mean = float(errors.mean().item())
        min_scale = max(
            float(self._cfg.get('rnd_landscape_query_normalized_raw_min_scale', 1e-8)),
            float(model.error_epsilon),
        )
        scale = max(raw_mean, min_scale)
        model.set_target_scale(scale)
        self._rnd_landscape_query_target_scale_phase = int(collection_phase)
        self._rnd_landscape_query_target_scale_sample_count = int(errors.numel())
        clip_threshold = scale * float(model.normalized_raw_clip)
        return {
            'rnd_landscape_query/target_scale': scale,
            'rnd_landscape_query/target_scale_raw_mean': raw_mean,
            'rnd_landscape_query/target_scale_raw_std': float(
                errors.std(unbiased=False).item()
            ),
            'rnd_landscape_query/target_scale_raw_p90': float(
                torch.quantile(errors, 0.9).item()
            ),
            'rnd_landscape_query/target_scale_clip_fraction': float(
                (errors >= clip_threshold).float().mean().item()
            ),
            'rnd_landscape_query/target_scale_sample_count': float(errors.numel()),
            'rnd_landscape_query/target_scale_phase': float(collection_phase),
        }

    def _rnd_landscape_query_context(
            self,
            root_observations: torch.Tensor,
            world_model: nn.Module,
            record_diagnostics: bool = False,
    ):
        model = self._ensure_rnd_landscape_query_model()
        if model is None:
            return None
        roots = self._select_rnd_observation_features(root_observations.float())
        if self._rnd_landscape_query_space() == 'compact_image':
            roots = roots.reshape(-1, *roots.shape[-3:])
        else:
            roots = roots.reshape(-1, 2)
        min_snapshots = max(1, int(self._cfg.get('rnd_landscape_query_min_snapshots', 2)))
        if self._rnd_landscape_query_snapshot_count < min_snapshots:
            return None
        snapshot = self._rnd_target_model_for_targets()
        if snapshot is None:
            return None

        def exact_error_fn(query_states: torch.Tensor) -> torch.Tensor:
            return snapshot.error(query_states).reshape(-1)

        model.eval()
        context = model.query(roots, exact_error_fn)
        context.query_latents = self._rnd_landscape_encode_query_states(
            world_model, context.query_states
        ).detach()
        if record_diagnostics:
            self._record_rnd_compact_query_diagnostics(context)
        return context

    def _record_rnd_compact_query_diagnostics(self, context) -> None:
        if self._rnd_landscape_query_space() != 'compact_image':
            return
        limit = max(
            0, int(self._cfg.get('rnd_landscape_query_diagnostic_root_count', 0))
        )
        stored = sum(
            int(item['root_observations'].shape[0])
            for item in self._rnd_compact_query_diagnostic_buffer
        )
        take = min(limit - stored, int(context.root_states.shape[0]))
        if take <= 0:
            return
        self._rnd_compact_query_diagnostic_buffer.append(
            {
                'root_observations': (
                    context.root_states[:take].detach().clamp(0.0, 1.0).mul(255)
                    .round().to(torch.uint8).cpu().numpy()
                ),
                'query_observations': (
                    context.query_states[:take].detach().clamp(0.0, 1.0).mul(255)
                    .round().to(torch.uint8).cpu().numpy()
                ),
                'query_errors': torch.exp(
                    context.query_log_errors[:take, :, 0]
                ).detach().cpu().numpy().astype(np.float32),
                'root_codes': context.root_query_codes[:take].detach().cpu().numpy().astype(
                    np.float16
                ),
                'query_codes': context.query_codes[:take].detach().cpu().numpy().astype(
                    np.float16
                ),
            }
        )

    def pop_rnd_compact_query_diagnostics(self) -> Dict[str, np.ndarray]:
        """Return and clear the bounded collection sample used to inspect queries."""
        if not self._rnd_compact_query_diagnostic_buffer:
            return {}
        keys = tuple(self._rnd_compact_query_diagnostic_buffer[0])
        result = {
            key: np.concatenate(
                [item[key] for item in self._rnd_compact_query_diagnostic_buffer],
                axis=0,
            )
            for key in keys
        }
        self._rnd_compact_query_diagnostic_buffer = []
        return result

    def _rnd_episodic_memory_enabled(self) -> bool:
        return bool(self._cfg.get('rnd_episodic_memory', False))

    def _rnd_episodic_memory_key_space(self) -> str:
        key_space = str(
            self._cfg.get('rnd_episodic_memory_key_space', 'projected')
        ).lower()
        if key_space not in {'projected', 'raw', 'inverse'}:
            raise ValueError(
                "rnd_episodic_memory_key_space must be 'projected', 'raw', or 'inverse', "
                f'got {key_space!r}.'
            )
        return key_space

    def _rnd_episodic_memory_keys(
            self,
            world_model: nn.Module,
            latent_states: torch.Tensor,
            imagined: bool,
            observations: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build member-aligned keys from [batch, member, ...] latents."""
        if latent_states.dim() < 3:
            raise ValueError(
                'Episodic-memory latents must have shape [batch, member, ...], '
                f'got {tuple(latent_states.shape)}.'
            )
        batch_size, ensemble_size = latent_states.shape[:2]
        flat_latents = latent_states.reshape(
            batch_size * ensemble_size, *latent_states.shape[2:]
        )
        key_space = self._rnd_episodic_memory_key_space()
        if key_space == 'inverse':
            if imagined:
                member_ids = [
                    member_id
                    for _ in range(batch_size)
                    for member_id in range(ensemble_size)
                ]
                flat_keys = world_model.episodic_inverse_bridge(
                    flat_latents,
                    member_id=member_ids,
                    detach_latent=True,
                )
                keys = flat_keys.reshape(batch_size, ensemble_size, -1)
            else:
                if observations is None or int(observations.shape[0]) != batch_size:
                    raise ValueError(
                        'Direct inverse-memory keys require one observation per latent batch row.'
                    )
                direct_keys = world_model.episodic_inverse_embedding(observations)
                keys = direct_keys.unsqueeze(1).expand(-1, ensemble_size, -1)
        elif key_space == 'projected':
            member_ids = [
                member_id
                for _ in range(batch_size)
                for member_id in range(ensemble_size)
            ]
            flat_keys = world_model.project(
                flat_latents,
                with_grad=bool(imagined),
                member_id=member_ids,
            )
            keys = F.normalize(
                flat_keys.float().reshape(batch_size, ensemble_size, -1),
                p=2,
                dim=-1,
                eps=1e-8,
            )
        else:
            flat_keys = flat_latents.reshape(batch_size * ensemble_size, -1)
            keys = flat_keys.float().reshape(batch_size, ensemble_size, -1)
        return keys.detach()

    @staticmethod
    def _rnd_episodic_transition_distance(
            first: torch.Tensor,
            second: torch.Tensor,
    ) -> float:
        return float((first - second).pow(2).sum(dim=-1).mean().item())

    def _refresh_rnd_episodic_memory_keys(
            self,
            world_model: nn.Module,
            state: dict,
    ) -> None:
        observations = state['observations']
        if not observations:
            state['keys'] = []
            state['transition_distances'] = []
            state['last_key'] = None
            return
        refresh_observations = list(observations)
        has_last_observation = state.get('last_observation') is not None
        if has_last_observation:
            refresh_observations.append(state['last_observation'])
        batch_size = max(
            1, int(self._cfg.get('rnd_episodic_memory_inference_batch_size', 128))
        )
        ensemble_size = int(self._cfg.model.ensemble_size)
        key_chunks = []
        for start in range(0, len(refresh_observations), batch_size):
            obs_chunk = torch.stack(
                refresh_observations[start:start + batch_size], dim=0
            ).to(device=self._cfg.device, dtype=torch.float32)
            chunk_size = int(obs_chunk.shape[0])
            repeated_obs = torch.cat(
                [obs_chunk for _ in range(ensemble_size)], dim=0
            )
            member_ids = [
                member_id
                for member_id in range(ensemble_size)
                for _ in range(chunk_size)
            ]
            latent_chunk = world_model.initial_inference(
                repeated_obs, member_id=member_ids
            ).latent_state.detach().reshape(
                ensemble_size, chunk_size, -1
            ).transpose(0, 1)
            key_chunks.append(
                self._rnd_episodic_memory_keys(
                    world_model,
                    latent_chunk,
                    imagined=False,
                    observations=obs_chunk,
                )
            )
        refreshed_keys = torch.cat(key_chunks, dim=0)
        keys = refreshed_keys[:len(observations)]
        state['keys'] = [key.detach() for key in keys]
        state['last_key'] = (
            refreshed_keys[-1].detach() if has_last_observation else keys[-1].detach()
        )
        state['transition_distances'] = [
            self._rnd_episodic_transition_distance(keys[index - 1], keys[index])
            for index in range(1, int(keys.shape[0]))
        ]

    def _rnd_episodic_memory_contexts(
            self,
            world_model: nn.Module,
            root_observations: torch.Tensor,
            root_latent_states: torch.Tensor,
            ready_env_ids: Sequence[int],
            memory_by_env: dict,
            model_version: int,
    ) -> Optional[List[EpisodicLatentMemoryContext]]:
        if not self._rnd_episodic_memory_enabled():
            return None
        direct_keys = self._rnd_episodic_memory_keys(
            world_model,
            root_latent_states,
            imagined=False,
            observations=root_observations,
        )
        max_size = max(1, int(self._cfg.get('rnd_episodic_memory_max_size', 1200)))
        distance_epsilon = max(
            1e-12, float(self._cfg.get('rnd_episodic_memory_distance_epsilon', 1e-8))
        )
        retain_repeated_visits = (
            str(self._cfg.get('rnd_episodic_memory_mode', 'minimum_distance')).lower()
            == 'ngu_knn'
        )
        contexts = []
        for row, env_id_value in enumerate(ready_env_ids):
            env_id = int(env_id_value)
            state = memory_by_env.setdefault(
                env_id,
                {
                    'observations': [],
                    'keys': [],
                    'transition_distances': [],
                    'last_key': None,
                    'last_observation': None,
                    'model_version': int(model_version),
                },
            )
            if int(state['model_version']) != int(model_version):
                self._refresh_rnd_episodic_memory_keys(world_model, state)
                state['model_version'] = int(model_version)

            key = direct_keys[row].detach()
            if state['last_key'] is not None:
                state['transition_distances'].append(
                    self._rnd_episodic_transition_distance(state['last_key'], key)
                )
            observation = root_observations[row].detach().to(device='cpu').clone()
            is_duplicate = False
            if not retain_repeated_visits and state['keys']:
                memory_keys = torch.stack(state['keys'], dim=0)
                minimum_distance = (
                    (memory_keys - key.unsqueeze(0)).pow(2).sum(dim=-1).mean(dim=-1).amin()
                )
                is_duplicate = bool(minimum_distance <= distance_epsilon)
            if not is_duplicate:
                state['observations'].append(observation)
                state['keys'].append(key)
            state['last_key'] = key
            state['last_observation'] = observation
            if len(state['keys']) > max_size:
                state['keys'].pop(0)
                state['observations'].pop(0)
            if len(state['transition_distances']) > max_size:
                state['transition_distances'] = state['transition_distances'][-max_size:]

            positive_distances = [
                distance for distance in state['transition_distances']
                if math.isfinite(distance) and distance > distance_epsilon
            ]
            distance_scale = (
                float(np.median(np.asarray(positive_distances, dtype=np.float64)))
                if positive_distances else distance_epsilon
            )
            contexts.append(
                EpisodicLatentMemoryContext(
                    keys=torch.stack(state['keys'], dim=0),
                    transition_distance_scale=distance_scale,
                )
            )
        return contexts

    def _rnd_landscape_path_reward_fn(
            self,
            context,
            world_model: Optional[nn.Module] = None,
            episodic_memories: Optional[Sequence[EpisodicLatentMemoryContext]] = None,
    ):
        if context is None:
            return None
        model = self._rnd_landscape_query_model
        episodic_gate = None
        if episodic_memories is not None:
            if world_model is None:
                raise ValueError('world_model is required with episodic latent memory.')
            episodic_gate = EpisodicLatentRewardGate(
                episodic_memories,
                kernel_bandwidth=max(
                    1e-8,
                    float(self._cfg.get('rnd_episodic_memory_kernel_bandwidth', 0.25)),
                ),
                minimum_reward_factor=float(
                    self._cfg.get('rnd_episodic_memory_min_reward_factor', 0.05)
                ),
                distance_epsilon=max(
                    1e-12,
                    float(self._cfg.get('rnd_episodic_memory_distance_epsilon', 1e-8)),
                ),
                include_path_memory=bool(
                    self._cfg.get('rnd_episodic_memory_include_path', True)
                ),
                mode=str(
                    self._cfg.get('rnd_episodic_memory_mode', 'minimum_distance')
                ),
                knn_k=int(self._cfg.get('rnd_episodic_memory_knn_k', 10)),
                knn_cluster_distance=float(
                    self._cfg.get('rnd_episodic_memory_knn_cluster_distance', 0.008)
                ),
                knn_kernel_epsilon=float(
                    self._cfg.get('rnd_episodic_memory_knn_kernel_epsilon', 1e-4)
                ),
                knn_pseudo_count_constant=float(
                    self._cfg.get(
                        'rnd_episodic_memory_knn_pseudo_count_constant', 0.001
                    )
                ),
                knn_maximum_similarity=float(
                    self._cfg.get('rnd_episodic_memory_knn_maximum_similarity', 8.0)
                ),
                knn_warmup_until_full=bool(
                    self._cfg.get('rnd_episodic_memory_knn_warmup_until_full', True)
                ),
            )

        def reward_fn(root_indices, action_paths, imagined_latents):
            index_tensor = torch.as_tensor(
                root_indices, dtype=torch.long, device=context.root_states.device
            )
            predicted_errors = model.predict_latent_error(
                context,
                imagined_latents,
                root_indices=index_tensor,
            )
            base_bonuses = self._normalize_rnd_errors(predicted_errors)
            bonuses = base_bonuses
            episodic_records = [{} for _ in root_indices]
            if episodic_gate is not None:
                imagined_keys = self._rnd_episodic_memory_keys(
                    world_model, imagined_latents, imagined=True
                )
                bonuses, episodic_records = episodic_gate.apply(
                    base_bonuses,
                    imagined_keys,
                    root_indices,
                    action_paths,
                )
            reward_fn.records.extend(
                {
                    'root_index': int(root_index),
                    'path': tuple(int(action) for action in action_path),
                    'predicted_error': float(predicted_error.item()),
                    'base_predicted_reward': float(base_bonus.item()),
                    'predicted_reward': float(bonus.item()),
                    **episodic_record,
                }
                for root_index, action_path, predicted_error, base_bonus, bonus, episodic_record in zip(
                    root_indices,
                    action_paths,
                    predicted_errors,
                    base_bonuses,
                    bonuses,
                    episodic_records,
                )
            )
            return bonuses.detach().cpu().numpy().astype(np.float32, copy=False)

        reward_fn.records = []
        return reward_fn

    def _annotate_rnd_landscape_query_diagnostics(
            self,
            context,
            legal_actions: List[List[int]],
            root_action_diagnostics: List[List[Dict[str, Any]]],
            path_reward_fn=None,
    ) -> None:
        if context is None:
            return
        model = self._rnd_landscape_query_model
        records = list(getattr(path_reward_fn, 'records', []))
        records_by_root = {root_index: [] for root_index in range(len(legal_actions))}
        for record in records:
            records_by_root[int(record['root_index'])].append(record)
        for root_index, actions in enumerate(legal_actions):
            if not actions:
                continue
            query_errors = torch.exp(context.query_log_errors[root_index, :, 0])
            root_records = records_by_root.get(root_index, [])
            predicted_tree_rewards = np.asarray(
                [record['predicted_reward'] for record in root_records], dtype=np.float32
            )
            base_tree_rewards = np.asarray(
                [
                    record.get('base_predicted_reward', record['predicted_reward'])
                    for record in root_records
                ],
                dtype=np.float32,
            )
            episodic_reward_factors = np.asarray(
                [record.get('episodic_reward_factor', 1.0) for record in root_records],
                dtype=np.float32,
            )
            episodic_matched_sources = [
                record.get('episodic_matched_source', 'none')
                for record in root_records
            ]
            episodic_pseudo_counts = np.asarray(
                [record.get('episodic_pseudo_count', 0.0) for record in root_records],
                dtype=np.float32,
            )
            episodic_knn_warmup = np.asarray(
                [record.get('episodic_knn_warmup', 0.0) for record in root_records],
                dtype=np.float32,
            )
            common = {
                'rnd_landscape_query_enabled': 1.0,
                'rnd_landscape_query_snapshot_count': int(
                    self._rnd_landscape_query_snapshot_count
                ),
                'rnd_landscape_query_count': int(context.query_states.shape[1]),
                'rnd_landscape_query_error_mean': float(query_errors.mean().item()),
                'rnd_landscape_query_error_min': float(query_errors.min().item()),
                'rnd_landscape_query_error_max': float(query_errors.max().item()),
                'rnd_landscape_query_errors': query_errors.detach().cpu().tolist(),
                'rnd_landscape_query_tree_edge_count': int(len(root_records)),
                'rnd_landscape_query_tree_predicted_reward_mean': float(
                    predicted_tree_rewards.mean() if predicted_tree_rewards.size else 0.0
                ),
                'rnd_episodic_memory_enabled': float(
                    self._rnd_episodic_memory_enabled()
                ),
                'rnd_episodic_memory_key_space': (
                    self._rnd_episodic_memory_key_space()
                    if self._rnd_episodic_memory_enabled() else 'none'
                ),
                'rnd_episodic_memory_mode': str(
                    self._cfg.get('rnd_episodic_memory_mode', 'minimum_distance')
                ),
                'rnd_episodic_memory_tree_base_reward_mean': float(
                    base_tree_rewards.mean() if base_tree_rewards.size else 0.0
                ),
                'rnd_episodic_memory_tree_reward_factor_mean': float(
                    episodic_reward_factors.mean()
                    if episodic_reward_factors.size else 1.0
                ),
                'rnd_episodic_memory_tree_suppressed_fraction': float(
                    np.mean(episodic_reward_factors < 0.5)
                    if episodic_reward_factors.size else 0.0
                ),
                'rnd_episodic_memory_tree_real_match_fraction': float(
                    np.mean(np.asarray(episodic_matched_sources) == 'real')
                    if episodic_matched_sources else 0.0
                ),
                'rnd_episodic_memory_tree_path_match_fraction': float(
                    np.mean(np.asarray(episodic_matched_sources) == 'path')
                    if episodic_matched_sources else 0.0
                ),
                'rnd_episodic_memory_tree_pseudo_count_mean': float(
                    episodic_pseudo_counts.mean()
                    if episodic_pseudo_counts.size else 0.0
                ),
                'rnd_episodic_memory_tree_knn_warmup_fraction': float(
                    episodic_knn_warmup.mean()
                    if episodic_knn_warmup.size else 0.0
                ),
            }
            if self._rnd_landscape_query_space() == 'physical_2d':
                common['rnd_landscape_query_states'] = context.query_states[
                    root_index
                ].detach().cpu().tolist()
            elif context.query_codes is not None:
                common['rnd_landscape_query_code_norm_mean'] = float(
                    context.query_codes[root_index].norm(dim=-1).mean().item()
                )
            max_horizon = max(1, int(self._cfg.get('rnd_landscape_query_max_horizon', 8)))
            for depth in range(1, max_horizon + 1):
                depth_records = [
                    record for record in root_records if len(record['path']) == depth
                ]
                common[f'rnd_landscape_query_tree_depth_{depth}_edge_count'] = int(
                    len(depth_records)
                )
                depth_factors = np.asarray(
                    [record.get('episodic_reward_factor', 1.0) for record in depth_records],
                    dtype=np.float32,
                )
                common[f'rnd_episodic_memory_depth_{depth}_reward_factor_mean'] = float(
                    depth_factors.mean() if depth_factors.size else 1.0
                )
                common[f'rnd_episodic_memory_depth_{depth}_suppressed_fraction'] = float(
                    np.mean(depth_factors < 0.5) if depth_factors.size else 0.0
                )
                common[f'rnd_episodic_memory_depth_{depth}_path_match_fraction'] = float(
                    np.mean([
                        record.get('episodic_matched_source', 'none') == 'path'
                        for record in depth_records
                    ]) if depth_records else 0.0
                )
            by_action = {
                int(item.get('action', -1)): item
                for item in root_action_diagnostics[root_index]
            }
            one_step_by_action = {
                int(record['path'][0]): record
                for record in root_records
                if len(record['path']) == 1
            }
            for action in actions:
                item = by_action.get(int(action))
                if item is None:
                    continue
                one_step_record = one_step_by_action.get(int(action))
                item.update(common)
                item.update(
                    {
                        'rnd_landscape_query_predicted_one_step_reward': float(
                            one_step_record['predicted_reward']
                            if one_step_record is not None else 0.0
                        ),
                        'rnd_landscape_query_predicted_one_step_available': float(
                            one_step_record is not None
                        ),
                        'rnd_episodic_memory_one_step_base_reward': float(
                            one_step_record.get(
                                'base_predicted_reward',
                                one_step_record['predicted_reward'],
                            ) if one_step_record is not None else 0.0
                        ),
                        'rnd_episodic_memory_one_step_reward_factor': float(
                            one_step_record.get('episodic_reward_factor', 1.0)
                            if one_step_record is not None else 1.0
                        ),
                        'rnd_episodic_memory_one_step_real_distance': float(
                            one_step_record.get('episodic_real_distance', float('nan'))
                            if one_step_record is not None else float('nan')
                        ),
                        'rnd_episodic_memory_one_step_path_distance': float(
                            one_step_record.get('episodic_path_distance', float('nan'))
                            if one_step_record is not None else float('nan')
                        ),
                        'rnd_episodic_memory_one_step_distance_scale': float(
                            one_step_record.get('episodic_distance_scale', float('nan'))
                            if one_step_record is not None else float('nan')
                        ),
                        'rnd_episodic_memory_one_step_normalized_distance': float(
                            one_step_record.get(
                                'episodic_normalized_distance', float('nan')
                            ) if one_step_record is not None else float('nan')
                        ),
                        'rnd_episodic_memory_one_step_real_memory_size': int(
                            one_step_record.get('episodic_real_memory_size', 0)
                            if one_step_record is not None else 0
                        ),
                        'rnd_episodic_memory_one_step_neighbor_count': int(
                            one_step_record.get('episodic_neighbor_count', 0)
                            if one_step_record is not None else 0
                        ),
                        'rnd_episodic_memory_one_step_knn_distance_mean': float(
                            one_step_record.get(
                                'episodic_knn_distance_mean', float('nan')
                            ) if one_step_record is not None else float('nan')
                        ),
                        'rnd_episodic_memory_one_step_normalized_knn_distance_mean': float(
                            one_step_record.get(
                                'episodic_normalized_knn_distance_mean', float('nan')
                            ) if one_step_record is not None else float('nan')
                        ),
                        'rnd_episodic_memory_one_step_pseudo_count': float(
                            one_step_record.get('episodic_pseudo_count', 0.0)
                            if one_step_record is not None else 0.0
                        ),
                        'rnd_episodic_memory_one_step_knn_warmup': float(
                            one_step_record.get('episodic_knn_warmup', 0.0)
                            if one_step_record is not None else 0.0
                        ),
                    }
                )

    def _record_rnd_episodic_memory_decision(
            self,
            root_action_diagnostics: Sequence[Dict[str, Any]],
            selected_action: int,
            env_id: int,
            train_iter: int,
    ) -> None:
        if not self._rnd_episodic_memory_enabled():
            return
        max_horizon = max(1, int(self._cfg.get('rnd_landscape_query_max_horizon', 8)))
        for action_terms in root_action_diagnostics:
            if float(action_terms.get('rnd_landscape_query_enabled', 0.0)) <= 0.5:
                continue
            action = int(action_terms.get('action', -1))
            row = {
                'train_iter_at_collect': int(train_iter),
                'env_id': int(env_id),
                'action': action,
                'selected': int(action == int(selected_action)),
                'key_space': str(
                    action_terms.get(
                        'rnd_episodic_memory_key_space',
                        self._rnd_episodic_memory_key_space(),
                    )
                ),
                'gate_mode': str(
                    action_terms.get(
                        'rnd_episodic_memory_mode',
                        self._cfg.get('rnd_episodic_memory_mode', 'minimum_distance'),
                    )
                ),
                'tree_edge_count': int(
                    action_terms.get('rnd_landscape_query_tree_edge_count', 0)
                ),
                'tree_base_reward_mean': float(
                    action_terms.get('rnd_episodic_memory_tree_base_reward_mean', 0.0)
                ),
                'tree_final_reward_mean': float(
                    action_terms.get(
                        'rnd_landscape_query_tree_predicted_reward_mean', 0.0
                    )
                ),
                'tree_reward_factor_mean': float(
                    action_terms.get(
                        'rnd_episodic_memory_tree_reward_factor_mean', 1.0
                    )
                ),
                'tree_suppressed_fraction': float(
                    action_terms.get(
                        'rnd_episodic_memory_tree_suppressed_fraction', 0.0
                    )
                ),
                'tree_real_match_fraction': float(
                    action_terms.get(
                        'rnd_episodic_memory_tree_real_match_fraction', 0.0
                    )
                ),
                'tree_path_match_fraction': float(
                    action_terms.get(
                        'rnd_episodic_memory_tree_path_match_fraction', 0.0
                    )
                ),
                'one_step_base_reward': float(
                    action_terms.get('rnd_episodic_memory_one_step_base_reward', 0.0)
                ),
                'one_step_final_reward': float(
                    action_terms.get(
                        'rnd_landscape_query_predicted_one_step_reward', 0.0
                    )
                ),
                'one_step_reward_factor': float(
                    action_terms.get(
                        'rnd_episodic_memory_one_step_reward_factor', 1.0
                    )
                ),
                'one_step_real_distance': float(
                    action_terms.get(
                        'rnd_episodic_memory_one_step_real_distance', float('nan')
                    )
                ),
                'one_step_distance_scale': float(
                    action_terms.get(
                        'rnd_episodic_memory_one_step_distance_scale', float('nan')
                    )
                ),
                'one_step_normalized_distance': float(
                    action_terms.get(
                        'rnd_episodic_memory_one_step_normalized_distance', float('nan')
                    )
                ),
                'real_memory_size': int(
                    action_terms.get(
                        'rnd_episodic_memory_one_step_real_memory_size', 0
                    )
                ),
                'one_step_neighbor_count': int(
                    action_terms.get(
                        'rnd_episodic_memory_one_step_neighbor_count', 0
                    )
                ),
                'one_step_knn_distance_mean': float(
                    action_terms.get(
                        'rnd_episodic_memory_one_step_knn_distance_mean', float('nan')
                    )
                ),
                'one_step_normalized_knn_distance_mean': float(
                    action_terms.get(
                        'rnd_episodic_memory_one_step_normalized_knn_distance_mean',
                        float('nan'),
                    )
                ),
                'one_step_pseudo_count': float(
                    action_terms.get(
                        'rnd_episodic_memory_one_step_pseudo_count', 0.0
                    )
                ),
                'one_step_knn_warmup': float(
                    action_terms.get(
                        'rnd_episodic_memory_one_step_knn_warmup', 0.0
                    )
                ),
            }
            for depth in range(1, max_horizon + 1):
                for metric in (
                    'reward_factor_mean',
                    'suppressed_fraction',
                    'path_match_fraction',
                ):
                    row[f'depth_{depth}_{metric}'] = float(
                        action_terms.get(
                            f'rnd_episodic_memory_depth_{depth}_{metric}',
                            1.0 if metric == 'reward_factor_mean' else 0.0,
                        )
                    )
            self._rnd_episodic_memory_diagnostic_buffer.append(row)

    def pop_rnd_episodic_memory_diagnostics(self) -> List[dict]:
        rows = list(getattr(self, '_rnd_episodic_memory_diagnostic_buffer', []))
        self._rnd_episodic_memory_diagnostic_buffer = []
        return rows

    def _rnd_target_model_for_targets(self) -> Optional[nn.Module]:
        if bool(self._cfg.get('rnd_target_snapshot', True)):
            self._sync_rnd_target_snapshot(force=False)
            if self._rnd_target_snapshot_model is not None:
                return self._rnd_target_snapshot_model
        return self._rnd_model

    def _update_rnd_bonus_stats(self, errors: torch.Tensor) -> None:
        errors = errors.detach().float()
        batch_mean = float(errors.mean().item())
        batch_var = float(errors.var(unbiased=False).item())
        if not self._rnd_bonus_stats_initialized:
            self._rnd_bonus_mean = batch_mean
            self._rnd_bonus_var = max(batch_var, 1e-6)
            self._rnd_bonus_stats_initialized = True
            return
        momentum = float(self._cfg.get('rnd_bonus_normalization_momentum', 0.99))
        self._rnd_bonus_mean = momentum * self._rnd_bonus_mean + (1 - momentum) * batch_mean
        self._rnd_bonus_var = momentum * self._rnd_bonus_var + (1 - momentum) * max(batch_var, 1e-6)

    def _normalize_rnd_errors(self, errors: torch.Tensor) -> torch.Tensor:
        mode = str(self._cfg.get('rnd_bonus_normalization', 'zscore')).lower()
        if mode == 'raw':
            bonus = torch.relu(errors.float())
        elif mode == 'zscore' and self._rnd_bonus_stats_initialized:
            std = max(self._rnd_bonus_var, 1e-6) ** 0.5
            bonus = torch.relu((errors - self._rnd_bonus_mean) / std)
        elif mode == 'zscore':
            bonus = errors
        else:
            raise ValueError(
                f"Unsupported rnd_bonus_normalization={mode!r}; expected 'zscore' or 'raw'."
            )
        bonus_clip = float(self._cfg.get('rnd_bonus_clip', 5.0))
        if bonus_clip > 0:
            bonus = bonus.clamp(max=bonus_clip)
        return bonus.detach()

    def _update_rnd_bonus_stats_for_inputs(
            self,
            errors: torch.Tensor,
            sample_keys: Optional[List[str]] = None,
    ) -> int:
        errors = errors.reshape(-1).detach()
        if errors.numel() == 0:
            return 0

        if sample_keys is not None:
            eligible_positions, _ = self._eligible_rnd_training_positions(int(errors.numel()), sample_keys)
            if not eligible_positions:
                return 0
            indices = torch.as_tensor(eligible_positions, dtype=torch.long, device=errors.device)
            errors = errors.index_select(0, indices)
            sample_keys = [sample_keys[idx] for idx in eligible_positions]

        if bool(self._cfg.get('rnd_train_on_new_data_only', True)) and sample_keys is not None:
            unseen_positions = [
                idx for idx, key in enumerate(sample_keys)
                if key not in self._rnd_trained_observation_keys
            ]
            if not unseen_positions:
                return 0
            indices = torch.as_tensor(unseen_positions, dtype=torch.long, device=errors.device)
            errors = errors.index_select(0, indices)

        self._update_rnd_bonus_stats(errors)
        return int(errors.numel())

    def _rnd_bonus_from_latents(self, latent_state: torch.Tensor) -> torch.Tensor:
        if not self._rnd_enabled() or self._rnd_model is None:
            return torch.zeros(latent_state.shape[0], dtype=torch.float32, device=latent_state.device)
        with torch.no_grad():
            errors = self._rnd_model.error(latent_state)
            bonus = self._normalize_rnd_errors(errors)
        return bonus.detach()

    def _train_rnd_from_latents(self, latent_state: torch.Tensor) -> dict:
        if not self._rnd_enabled():
            return {}
        self._ensure_rnd_model(latent_state)
        if self._rnd_model is None:
            return {}

        latent_state = latent_state.detach()
        train_batch_size = int(self._cfg.get('rnd_train_batch_size', 512))
        if latent_state.shape[0] > train_batch_size:
            indices = torch.randperm(latent_state.shape[0], device=latent_state.device)[:train_batch_size]
            latent_state = latent_state.index_select(0, indices)

        self._rnd_model.train()
        prediction, target = self._rnd_model(latent_state)
        per_sample_error = (prediction - target).pow(2).mean(dim=1)
        loss = per_sample_error.mean()
        self._rnd_optimizer.zero_grad()
        loss.backward()
        self._rnd_optimizer.step()
        self._update_rnd_bonus_stats(per_sample_error.detach())
        return {
            'rnd/loss': float(loss.item()),
            'rnd/error_mean': float(per_sample_error.mean().item()),
            'rnd/error_std': float(per_sample_error.std(unbiased=False).item()),
            'rnd/bonus_mean_running': float(self._rnd_bonus_mean),
            'rnd/bonus_std_running': float(max(self._rnd_bonus_var, 1e-6) ** 0.5),
        }

    def _extract_observation_sequence(self, obs_batch_ori: np.ndarray) -> Optional[torch.Tensor]:
        obs_tensor = torch.from_numpy(obs_batch_ori).to(self._cfg.device).float()
        model_type = str(self._cfg.model.model_type)
        if model_type == 'mlp':
            observation_shape = self._cfg.model.observation_shape
            if isinstance(observation_shape, (list, tuple)):
                obs_dim = int(np.prod(observation_shape))
            else:
                obs_dim = int(observation_shape)
            if obs_dim <= 0 or obs_tensor.dim() != 2 or obs_tensor.shape[1] % obs_dim != 0:
                return None
            return obs_tensor.reshape(obs_tensor.shape[0], -1, obs_dim)

        if model_type not in {'conv', 'conv_context'} or obs_tensor.dim() != 4:
            return None
        image_channels = int(self._cfg.model.get('image_channel', 0))
        frame_stack_num = max(1, int(self._cfg.model.get('frame_stack_num', 1)))
        if image_channels <= 0 or obs_tensor.shape[1] % image_channels != 0:
            return None
        frame_count = int(obs_tensor.shape[1] // image_channels)
        if frame_count < frame_stack_num:
            return None
        frames = obs_tensor.reshape(
            obs_tensor.shape[0],
            frame_count,
            image_channels,
            obs_tensor.shape[-2],
            obs_tensor.shape[-1],
        )
        windows = [
            frames[:, start:start + frame_stack_num].reshape(
                obs_tensor.shape[0],
                frame_stack_num * image_channels,
                obs_tensor.shape[-2],
                obs_tensor.shape[-1],
            )
            for start in range(frame_count - frame_stack_num + 1)
        ]
        return torch.stack(windows, dim=1)

    def _select_rnd_observation_features(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        if str(self._cfg.model.get('model_type', 'mlp')) in {'conv', 'conv_context'}:
            if obs_tensor.dim() < 4:
                raise ValueError(
                    'Image RND observations must have at least four dimensions, got '
                    f'{tuple(obs_tensor.shape)}.'
                )
            channel_axis = obs_tensor.dim() - 3
            image_channels = int(self._cfg.model.get('image_channel', 0))
            if image_channels <= 0 or int(obs_tensor.shape[channel_axis]) < image_channels:
                raise ValueError(
                    f'Invalid image_channel={image_channels} for observation shape '
                    f'{tuple(obs_tensor.shape)}.'
                )
            return obs_tensor.narrow(
                channel_axis,
                int(obs_tensor.shape[channel_axis]) - image_channels,
                image_channels,
            ).contiguous()

        observation_indices = self._cfg.get('rnd_observation_indices', None)
        observation_dim = self._cfg.get('rnd_observation_dim', None)
        if observation_indices is not None:
            indices = [int(idx) for idx in observation_indices]
            if not indices:
                return obs_tensor
            obs_dim = int(obs_tensor.shape[-1])
            if min(indices) < 0 or max(indices) >= obs_dim:
                raise ValueError(
                    f"rnd_observation_indices={indices} are invalid for observation dim {obs_dim}."
                )
            index_tensor = torch.as_tensor(indices, dtype=torch.long, device=obs_tensor.device)
            return torch.index_select(obs_tensor, obs_tensor.dim() - 1, index_tensor).contiguous()

        if observation_dim is None:
            return obs_tensor

        observation_dim = int(observation_dim)
        obs_dim = int(obs_tensor.shape[-1])
        if observation_dim <= 0 or observation_dim > obs_dim:
            raise ValueError(
                f"rnd_observation_dim={observation_dim} is invalid for observation dim {obs_dim}."
            )
        return obs_tensor[..., :observation_dim].contiguous()

    @staticmethod
    def _flatten_observation_time_batch(obs_tensor: torch.Tensor) -> torch.Tensor:
        if obs_tensor.dim() < 3:
            raise ValueError(
                'Expected [batch, time, ...] observations, got '
                f'{tuple(obs_tensor.shape)}.'
            )
        return obs_tensor.reshape(
            int(obs_tensor.shape[0] * obs_tensor.shape[1]),
            *obs_tensor.shape[2:],
        )

    def _get_collection_rnd_reward_fn(self):
        return self._collection_intrinsic_rewards_for_observations

    def _get_collection_episodic_novelty_embedding_fn(self):
        return self._collection_episodic_novelty_embeddings_for_observations

    def _get_collection_episodic_history_reward_feedback_fn(self):
        return self._record_collection_episodic_history_rewards

    def _record_collection_episodic_history_rewards(
            self,
            env_ids: Sequence[int],
            rewards: Sequence[float],
            extrinsic_rewards: Optional[Sequence[float]] = None,
    ) -> None:
        """Attach exact real-transition novelty rewards to pending collector actions."""
        pending_by_env = getattr(self, '_collect_env_episodic_history_pending', None)
        if pending_by_env is None:
            return
        for env_id, reward in zip(env_ids, rewards):
            pending = pending_by_env.get(int(env_id))
            if pending is not None:
                pending['realized_reward'] = float(reward)
        sequence_pending = getattr(
            self, '_collect_env_episodic_sequence_pending', None
        )
        if sequence_pending is not None and extrinsic_rewards is not None:
            for env_id, reward in zip(env_ids, extrinsic_rewards):
                pending = sequence_pending.get(int(env_id))
                if pending is not None:
                    pending['extrinsic_reward'] = float(reward)

    def _collection_episodic_novelty_embeddings_for_observations(self, observations) -> np.ndarray:
        """Encode real collector observations with one explicit ensemble member."""
        observations = np.asarray(observations, dtype=np.float32)
        prepared = prepare_observation(observations, self._cfg.model.model_type)
        obs_tensor = torch.as_tensor(prepared, dtype=torch.float32, device=self._cfg.device)
        member_id = int(self._cfg.get('episodic_novelty_encoder_member', 0))
        ensemble_size = int(self._cfg.model.get('ensemble_size', 1))
        if member_id < 0 or member_id >= ensemble_size:
            raise ValueError(
                "episodic_novelty_encoder_member must be in "
                f"[0, {ensemble_size}), got {member_id}."
            )
        was_training = self._collect_model.training
        try:
            self._collect_model.eval()
            with torch.no_grad():
                latent_state = self._collect_model.initial_inference(
                    obs_tensor,
                    member_id=member_id,
                ).latent_state
        finally:
            self._collect_model.train(was_training)
        return latent_state.detach().float().reshape(latent_state.shape[0], -1).cpu().numpy()

    def _collection_intrinsic_rewards_for_observations(self, observations) -> dict:
        source = self._intrinsic_reward_source()
        if source == 'qstd':
            return self._collection_qstd_rewards_for_observations(observations)
        if source == 'state_value_disagreement':
            return self._collection_state_value_disagreement_rewards_for_observations(observations)
        return self._collection_rnd_rewards_for_observations(observations)

    def _collection_state_value_disagreement_rewards_for_observations(self, observations) -> dict:
        observations = np.asarray(observations, dtype=np.float32)
        if observations.ndim == 1:
            observations = observations.reshape(1, -1)
        batch_size = int(observations.shape[0])
        empty = {
            'bonus': np.zeros(batch_size, dtype=np.float32),
            'raw_error': np.zeros(batch_size, dtype=np.float32),
        }
        if (
                self._intrinsic_reward_source() != 'state_value_disagreement'
                or not bool(self._cfg.get('state_value_disagreement_store_collection_rewards', False))
                or int(self._cfg.model.get('ensemble_size', 1)) < 2
        ):
            return empty

        obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=self._cfg.device)
        raw_rewards, _ = self._state_value_disagreement_from_observations(
            obs_tensor, model=self._collect_model
        )
        bonuses = raw_rewards * float(self._cfg.get('state_value_disagreement_reward_scale', 1.0))
        reward_clip = self._cfg.get('state_value_disagreement_reward_clip', None)
        if reward_clip is not None and abs(float(reward_clip)) > 0.0:
            bonuses = bonuses.clamp(min=0.0, max=abs(float(reward_clip)))
        return {
            'bonus': bonuses.detach().cpu().numpy().astype(np.float32, copy=False),
            'raw_error': raw_rewards.detach().cpu().numpy().astype(np.float32, copy=False),
        }

    def _collection_rnd_rewards_for_observations(self, observations) -> dict:
        observations = np.asarray(observations, dtype=np.float32)
        if observations.ndim == 1:
            observations = observations.reshape(1, -1)
        batch_size = int(observations.shape[0])
        empty = {
            'bonus': np.zeros(batch_size, dtype=np.float32),
            'raw_error': np.zeros(batch_size, dtype=np.float32),
        }
        if (
                not self._rnd_enabled()
                or self._intrinsic_reward_source() != 'rnd'
                or not bool(self._cfg.get('rnd_store_collection_rewards', False))
        ):
            return empty

        self._init_rnd_state()
        obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=self._cfg.device)
        rnd_obs = self._select_rnd_observation_features(obs_tensor)
        rnd_inputs = rnd_obs.reshape(batch_size, *rnd_obs.shape[1:])
        self._ensure_rnd_model(rnd_inputs)
        rnd_target_model = self._rnd_target_model_for_targets()
        if rnd_target_model is None:
            return empty

        rnd_target_model.eval()
        with torch.no_grad():
            raw_errors = rnd_target_model.error(rnd_inputs).reshape(batch_size)
            bonuses = self._normalize_rnd_errors(raw_errors).reshape(batch_size)
            if bool(self._cfg.get('rnd_update_stats_on_collection', True)):
                self._update_rnd_bonus_stats(raw_errors)
        return {
            'bonus': bonuses.detach().cpu().numpy().astype(np.float32, copy=False),
            'raw_error': raw_errors.detach().cpu().numpy().astype(np.float32, copy=False),
        }

    def _qstd_reward_from_action_stds(self, action_stds: np.ndarray) -> float:
        action_stds = np.asarray(action_stds, dtype=np.float32).reshape(-1)
        action_stds = action_stds[np.isfinite(action_stds)]
        if action_stds.size == 0:
            return 0.0
        aggregation = str(self._cfg.get('qstd_reward_aggregation', 'max') or 'max').lower()
        if aggregation == 'max':
            reward = float(action_stds.max())
        elif aggregation == 'mean':
            reward = float(action_stds.mean())
        elif aggregation in {'mean_topk', 'topk_mean'}:
            top_k = max(1, int(self._cfg.get('qstd_reward_top_k', 1)))
            top_k = min(top_k, int(action_stds.size))
            reward = float(np.sort(action_stds)[-top_k:].mean())
        else:
            raise ValueError(
                "qstd_reward_aggregation must be one of 'max', 'mean', or 'mean_topk', "
                f"got {aggregation!r}."
            )
        reward *= float(self._cfg.get('qstd_reward_scale', 1.0))
        clip_value = self._cfg.get('qstd_reward_clip', None)
        if clip_value is not None:
            clip_value = abs(float(clip_value))
            if clip_value > 0.0:
                reward = float(np.clip(reward, 0.0, clip_value))
        return max(0.0, reward)

    def _qstd_recompute_replay_rewards_enabled(self) -> bool:
        return bool(
            self._intrinsic_reward_source() == 'qstd'
            and self._cfg.get('qstd_exploration', False)
            and self._cfg.get('qstd_recompute_replay_rewards', False)
        )

    def _sync_qstd_target_snapshot(self, force: bool = False) -> None:
        """Freeze the extrinsic ensemble used to define one-step Qstd rewards."""
        self._init_rnd_state()
        if not self._qstd_recompute_replay_rewards_enabled():
            return
        if not bool(self._cfg.get('qstd_target_snapshot', True)):
            self._qstd_target_snapshot_model = None
            self._qstd_target_snapshot_value_offsets = None
            self._qstd_target_reward_cache.clear()
            return
        if self._qstd_target_snapshot_model is not None and not force:
            return

        self._qstd_target_snapshot_model = copy.deepcopy(self._learn_model).to(self._cfg.device)
        self._qstd_target_snapshot_model.eval()
        for param in self._qstd_target_snapshot_model.parameters():
            param.requires_grad_(False)
        self._qstd_target_snapshot_value_offsets = self._value_calibration_offsets().astype(
            np.float32, copy=True
        )
        self._qstd_target_reward_cache.clear()
        self._qstd_target_snapshot_generation += 1

    def refresh_qstd_target_snapshot_for_collection(self) -> None:
        """Freeze one Qstd reward definition for the next collection/learning phase."""
        if bool(self._cfg.get('qstd_target_snapshot_refresh_per_collect', False)):
            self._sync_qstd_target_snapshot(force=True)

    def _qstd_target_model_for_targets(self) -> nn.Module:
        if bool(self._cfg.get('qstd_target_snapshot', True)):
            self._sync_qstd_target_snapshot(force=False)
            if self._qstd_target_snapshot_model is not None:
                return self._qstd_target_snapshot_model
        return self._target_model

    def _one_step_action_qstd_rewards(
            self,
            observations: torch.Tensor,
            actions: torch.Tensor,
            valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """Compute std_m[r_m(s,a) + gamma V_m(f_m(s,a))] on real replay edges."""
        if observations.dim() != 3:
            raise ValueError(f'Expected observations [B, T, D], got {tuple(observations.shape)}.')
        batch_size, reward_steps = observations.shape[:2]
        actions = torch.as_tensor(actions, dtype=torch.long, device=observations.device)
        if actions.dim() > 2:
            actions = actions.reshape(actions.shape[0], -1)
        if tuple(actions.shape[:2]) != (batch_size, reward_steps):
            raise ValueError(
                f'Expected actions [{batch_size}, {reward_steps}], got {tuple(actions.shape)}.'
            )

        scaled_rewards = observations.new_zeros((batch_size, reward_steps))
        raw_rewards = observations.new_zeros((batch_size, reward_steps))
        flat_valid = (valid_mask.reshape(-1) > 0.5).nonzero(as_tuple=False).squeeze(-1)
        if flat_valid.numel() == 0:
            return scaled_rewards, raw_rewards, {
                'cache_hit_fraction': 0.0,
                'cache_size': float(len(getattr(self, '_qstd_target_reward_cache', {}))),
            }

        self._init_rnd_state()
        flat_obs = observations.reshape(batch_size * reward_steps, *observations.shape[2:])
        flat_actions = actions.reshape(-1)
        valid_indices = flat_valid.detach().cpu().tolist()
        cache_enabled = bool(self._cfg.get('qstd_target_snapshot', True))
        cache = self._qstd_target_reward_cache if cache_enabled else {}
        key_positions: Dict[Tuple[int, bytes], List[int]] = {}
        representative_indices: Dict[Tuple[int, bytes], int] = {}
        cache_hits = 0

        obs_cpu = flat_obs.index_select(0, flat_valid).detach().float().cpu().numpy()
        action_cpu = flat_actions.index_select(0, flat_valid).detach().cpu().numpy()
        for local_idx, flat_idx in enumerate(valid_indices):
            key = (int(action_cpu[local_idx]), np.ascontiguousarray(obs_cpu[local_idx]).tobytes())
            key_positions.setdefault(key, []).append(flat_idx)
            representative_indices.setdefault(key, flat_idx)
            if key in cache:
                cache_hits += 1

        missing_keys = [key for key in key_positions if key not in cache]
        model = self._qstd_target_model_for_targets()
        ensemble_size = int(self._cfg.model.ensemble_size)
        inference_batch_size = max(1, int(self._cfg.get('qstd_target_inference_batch_size', 1024)))
        if missing_keys:
            was_training = model.training
            try:
                model.eval()
                with torch.no_grad():
                    for start in range(0, len(missing_keys), inference_batch_size):
                        chunk_keys = missing_keys[start:start + inference_batch_size]
                        chunk_indices = torch.as_tensor(
                            [representative_indices[key] for key in chunk_keys],
                            dtype=torch.long,
                            device=observations.device,
                        )
                        obs_chunk = flat_obs.index_select(0, chunk_indices)
                        action_chunk = flat_actions.index_select(0, chunk_indices).view(-1, 1)
                        chunk_size = int(obs_chunk.shape[0])
                        member_ids = [
                            member_id
                            for member_id in range(ensemble_size)
                            for _ in range(chunk_size)
                        ]
                        repeated_obs = self._repeat_tensor_for_members(obs_chunk, ensemble_size)
                        repeated_actions = self._repeat_tensor_for_members(action_chunk, ensemble_size)
                        initial_output = model.initial_inference(repeated_obs, member_id=member_ids)
                        recurrent_output = model.recurrent_inference(
                            initial_output.latent_state, repeated_actions, member_id=member_ids
                        )
                        values = self.value_inverse_scalar_transform_handle(
                            recurrent_output.value
                        ).detach().reshape(ensemble_size, chunk_size, -1).squeeze(-1)
                        if bool(self._cfg.get('qstd_one_step_include_reward', True)):
                            rewards = self.reward_inverse_scalar_transform_handle(
                                recurrent_output.reward
                            ).detach().reshape(ensemble_size, chunk_size, -1).squeeze(-1)
                        else:
                            rewards = torch.zeros_like(values)

                        if bool(self._cfg.get('qstd_one_step_use_value_calibration', True)):
                            offsets_np = self._qstd_target_snapshot_value_offsets
                            if offsets_np is None:
                                offsets_np = self._value_calibration_offsets(ensemble_size)
                            offsets = torch.as_tensor(
                                offsets_np, dtype=values.dtype, device=values.device
                            ).view(ensemble_size, 1)
                            values = values - offsets

                        member_q = rewards + float(self._cfg.discount_factor) * values
                        chunk_qstd = member_q.std(dim=0, unbiased=False).float().cpu().numpy()
                        for key, raw_reward in zip(chunk_keys, chunk_qstd.tolist()):
                            cache[key] = max(0.0, float(raw_reward))
            finally:
                if hasattr(model, 'set_active_member'):
                    model.set_active_member(0)
                model.train(was_training)

        raw_flat = raw_rewards.reshape(-1)
        scaled_flat = scaled_rewards.reshape(-1)
        reward_scale = float(self._cfg.get('qstd_reward_scale', 1.0))
        reward_clip = self._cfg.get('qstd_reward_clip', None)
        reward_clip = abs(float(reward_clip)) if reward_clip is not None else None
        for key, positions in key_positions.items():
            raw_reward = float(cache[key])
            scaled_reward = max(0.0, raw_reward * reward_scale)
            if reward_clip is not None and reward_clip > 0.0:
                scaled_reward = min(scaled_reward, reward_clip)
            position_tensor = torch.as_tensor(positions, dtype=torch.long, device=observations.device)
            raw_flat.index_fill_(0, position_tensor, raw_reward)
            scaled_flat.index_fill_(0, position_tensor, scaled_reward)

        if cache_enabled:
            self._qstd_target_reward_cache = cache
        return scaled_rewards.detach(), raw_rewards.detach(), {
            'cache_hit_fraction': float(cache_hits / max(1, len(valid_indices))),
            'cache_size': float(len(cache)),
        }

    def _collection_qstd_rewards_for_observations(self, observations) -> dict:
        observations = np.asarray(observations, dtype=np.float32)
        if observations.ndim == 1:
            observations = observations.reshape(1, -1)
        batch_size = int(observations.shape[0])
        empty = {
            'bonus': np.zeros(batch_size, dtype=np.float32),
            'raw_error': np.zeros(batch_size, dtype=np.float32),
        }
        if (
                self._intrinsic_reward_source() != 'qstd'
                or not bool(self._cfg.get('qstd_exploration', False))
                or not bool(self._cfg.get('qstd_store_collection_rewards', False))
                or float(self._cfg.get('qstd_intrinsic_reward_weight', 0.0)) <= 0.0
                or int(self._cfg.model.get('ensemble_size', 1)) < 2
        ):
            return empty
        if not hasattr(self, '_ensemble_shared_tree_mcts_collect'):
            return empty

        self._collect_model.eval()
        obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=self._cfg.device)
        ensemble_size = int(self._cfg.model.ensemble_size)
        member_ids = [
            member_id
            for member_id in range(ensemble_size)
            for _ in range(batch_size)
        ]
        repeated_obs = torch.cat([obs_tensor for _ in range(ensemble_size)], dim=0)
        action_space_size = int(self._cfg.model.action_space_size)
        legal_actions = [list(range(action_space_size)) for _ in range(batch_size)]
        num_simulations = self._cfg.get('qstd_reward_num_simulations', None)
        if num_simulations is None:
            num_simulations = int(self._cfg.num_simulations)
        search_mode = str(self._cfg.get('qstd_reward_search_mode', 'mean_shared_tree') or 'mean_shared_tree')
        if search_mode not in [
                'mean_shared_tree', 'optimistic_shared_tree', 'pessimistic_shared_tree',
                'bounded_optimistic_shared_tree', 'bounded_pessimistic_shared_tree'
        ]:
            raise ValueError(
                "qstd_reward_search_mode must be one of 'mean_shared_tree', "
                "'optimistic_shared_tree', 'pessimistic_shared_tree', "
                "'bounded_optimistic_shared_tree', or 'bounded_pessimistic_shared_tree', "
                f"got {search_mode!r}."
            )

        with torch.no_grad():
            network_output = self._collect_model.initial_inference(repeated_obs, member_id=member_ids)
            latent_state_roots = network_output.latent_state.detach()
            policy_logits = network_output.policy_logits.detach()
            pred_values = self.value_inverse_scalar_transform_handle(network_output.value).detach()

            latent_state_roots = latent_state_roots.reshape(
                ensemble_size, batch_size, *latent_state_roots.shape[1:]
            ).transpose(0, 1)
            policy_logits = policy_logits.reshape(ensemble_size, batch_size, -1).transpose(0, 1)
            pred_values = pred_values.reshape(ensemble_size, batch_size, -1).transpose(0, 1).squeeze(-1)

            search_output = self._ensemble_shared_tree_mcts_collect.search(
                model=self._collect_model,
                root_latent_states=latent_state_roots,
                root_policy_logits=policy_logits,
                root_values=pred_values,
                legal_actions=legal_actions,
                add_exploration_noise=False,
                mode=search_mode,
                intrinsic_reward_fn=None,
                use_model_intrinsic=False,
                intrinsic_reward_source='rnd',
                ensemble_ucb_beta=0.0 if search_mode == 'mean_shared_tree' else None,
                ensemble_uncertainty_decay=False,
                num_simulations=int(num_simulations),
                intrinsic_reward_weight=0.0,
                return_tree_diagnostics=False,
                value_calibration_offsets=self._value_calibration_offsets(),
            )

        rewards = []
        raw_rewards = []
        for diagnostics in search_output.root_action_diagnostics:
            action_stds = np.asarray(
                [float(item.get('q_std', 0.0)) for item in diagnostics],
                dtype=np.float32,
            )
            raw_reward = float(action_stds.max()) if action_stds.size > 0 else 0.0
            rewards.append(self._qstd_reward_from_action_stds(action_stds))
            raw_rewards.append(max(0.0, raw_reward))
        return {
            'bonus': np.asarray(rewards, dtype=np.float32),
            'raw_error': np.asarray(raw_rewards, dtype=np.float32),
        }

    def _rnd_observation_keys(self, indices, reward_steps: int) -> List[str]:
        return [
            f"{index}:{offset}"
            for index in indices
            for offset in range(reward_steps)
        ]

    def _eligible_rnd_training_positions(
            self,
            num_inputs: int,
            sample_keys: Optional[List[str]] = None,
    ) -> Tuple[List[int], int]:
        if (
                sample_keys is None
                or not bool(self._cfg.get('rnd_ignore_elite_replay_for_predictor', True))
        ):
            return list(range(num_inputs)), 0
        eligible_positions = [
            idx for idx, key in enumerate(sample_keys)
            if not str(key).startswith('elite:') and not str(key).startswith('intrinsic:')
        ]
        return eligible_positions, num_inputs - len(eligible_positions)

    def _select_rnd_training_indices(
            self,
            num_inputs: int,
            device: torch.device,
            sample_keys: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, List[str], List[str], int, int, int]:
        if num_inputs <= 0:
            return torch.empty(0, dtype=torch.long, device=device), [], [], 0, 0, 0

        eligible_positions, skipped_elite_count = self._eligible_rnd_training_positions(num_inputs, sample_keys)
        if not eligible_positions:
            return torch.empty(0, dtype=torch.long, device=device), [], [], 0, 0, skipped_elite_count

        if bool(self._cfg.get('rnd_train_on_new_data_only', True)) and sample_keys is not None:
            unseen_positions = [
                idx for idx in eligible_positions
                if sample_keys[idx] not in self._rnd_trained_observation_keys
            ]
            skipped_seen_count = len(eligible_positions) - len(unseen_positions)
        else:
            unseen_positions = eligible_positions
            skipped_seen_count = 0

        candidate_count = len(unseen_positions)
        candidate_keys = [sample_keys[idx] for idx in unseen_positions] if sample_keys is not None else []
        if candidate_count == 0:
            return torch.empty(0, dtype=torch.long, device=device), [], [], 0, skipped_seen_count, skipped_elite_count

        update_proportion = float(self._cfg.get('rnd_predictor_update_proportion', 1.0))
        update_proportion = max(0.0, min(1.0, update_proportion))
        if update_proportion < 1.0:
            keep_mask = torch.rand(candidate_count, device=device) < update_proportion
            if not bool(keep_mask.any()) and update_proportion > 0.0:
                keep_mask[torch.randint(candidate_count, (1, ), device=device)] = True
            selected_positions = [
                position for position, keep in zip(unseen_positions, keep_mask.cpu().tolist())
                if keep
            ]
        else:
            selected_positions = unseen_positions

        selected_indices = torch.as_tensor(selected_positions, dtype=torch.long, device=device)
        selected_keys = [sample_keys[idx] for idx in selected_positions] if sample_keys is not None else []
        return selected_indices, selected_keys, candidate_keys, candidate_count, skipped_seen_count, skipped_elite_count

    def _train_rnd_on_inputs(self, input_tensor: torch.Tensor, sample_keys: Optional[List[str]] = None) -> dict:
        if not self._rnd_enabled():
            return {}
        self._ensure_rnd_model(input_tensor)
        if self._rnd_model is None:
            return {}

        input_tensor = input_tensor.detach()
        (
            selected_indices,
            selected_keys,
            processed_keys,
            candidate_count,
            skipped_seen_count,
            skipped_elite_count,
        ) = self._select_rnd_training_indices(
            input_tensor.shape[0], input_tensor.device, sample_keys
        )
        if processed_keys:
            self._rnd_trained_observation_keys.update(processed_keys)
        if selected_indices.numel() == 0:
            return {
                'rnd/loss': 0.0,
                'rnd/error_mean': 0.0,
                'rnd/error_std': 0.0,
                'rnd/bonus_mean_running': float(self._rnd_bonus_mean),
                'rnd/bonus_std_running': float(max(self._rnd_bonus_var, 1e-6) ** 0.5),
                'rnd/train_sample_count': 0.0,
                'rnd/train_candidate_count': float(candidate_count),
                'rnd/train_skipped_seen_count': float(skipped_seen_count),
                'rnd/train_skipped_elite_count': float(skipped_elite_count),
                'rnd/train_seen_cache_size': float(len(self._rnd_trained_observation_keys)),
            }
        input_tensor = input_tensor.index_select(0, selected_indices)
        train_batch_size = int(self._cfg.get('rnd_train_batch_size', 512))
        if input_tensor.shape[0] > train_batch_size:
            indices = torch.randperm(input_tensor.shape[0], device=input_tensor.device)[:train_batch_size]
            input_tensor = input_tensor.index_select(0, indices)
            if selected_keys:
                indices_list = indices.cpu().tolist()
                selected_keys = [selected_keys[idx] for idx in indices_list]

        self._rnd_model.train()
        prediction, target = self._rnd_model(input_tensor)
        per_sample_error = (prediction - target).pow(2).mean(dim=1)
        loss = per_sample_error.mean()
        self._rnd_optimizer.zero_grad()
        loss.backward()
        self._rnd_optimizer.step()
        return {
            'rnd/loss': float(loss.item()),
            'rnd/error_mean': float(per_sample_error.mean().item()),
            'rnd/error_std': float(per_sample_error.std(unbiased=False).item()),
            'rnd/bonus_mean_running': float(self._rnd_bonus_mean),
            'rnd/bonus_std_running': float(max(self._rnd_bonus_var, 1e-6) ** 0.5),
            'rnd/train_sample_count': float(input_tensor.shape[0]),
            'rnd/train_candidate_count': float(candidate_count),
            'rnd/train_skipped_seen_count': float(skipped_seen_count),
            'rnd/train_skipped_elite_count': float(skipped_elite_count),
            'rnd/train_seen_cache_size': float(len(self._rnd_trained_observation_keys)),
        }

    def _empty_rnd_replay_train_log(self) -> dict:
        return {
            'rnd/loss': 0.0,
            'rnd/error_mean': 0.0,
            'rnd/error_std': 0.0,
            'rnd/bonus_mean_running': float(self._rnd_bonus_mean),
            'rnd/bonus_std_running': float(max(self._rnd_bonus_var, 1e-6) ** 0.5),
            'rnd/train_sample_count': 0.0,
            'rnd/train_candidate_count': 0.0,
            'rnd/train_skipped_seen_count': 0.0,
            'rnd/train_skipped_elite_count': 0.0,
            'rnd/train_seen_cache_size': float(len(self._rnd_trained_observation_keys)),
            'rnd/train_from_fresh_collection_only': 1.0,
        }

    @staticmethod
    def _rnd_error_summary(errors: torch.Tensor) -> Tuple[float, float]:
        if errors.numel() == 0:
            return 0.0, 0.0
        return (
            float(errors.mean().item()),
            float(errors.std(unbiased=False).item()),
        )

    def _rnd_errors_in_batches(
            self,
            model: nn.Module,
            inputs: torch.Tensor,
            batch_size: int = 4096,
    ) -> torch.Tensor:
        if inputs.shape[0] == 0:
            return torch.empty(0, dtype=torch.float32)
        error_chunks = []
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                for start in range(0, int(inputs.shape[0]), max(1, int(batch_size))):
                    batch = inputs[start:start + max(1, int(batch_size))].to(self._cfg.device)
                    error_chunks.append(model.error(batch).detach().float().cpu())
        finally:
            model.train(was_training)
        return torch.cat(error_chunks, dim=0)

    def train_rnd_on_fresh_observations(self, observations) -> dict:
        """Train the online RND predictor once per selected fresh transition.

        Selection happens once when a transition is collected. Only complete,
        fixed-size minibatches produce optimizer steps; the remainder is carried
        into the next collection phase. The phase-frozen target snapshot is not
        modified here.
        """
        if (
                not self._rnd_enabled()
                or self._intrinsic_reward_source() != 'rnd'
                or not bool(self._cfg.get('rnd_train_from_fresh_collection_only', False))
        ):
            return {}

        self._init_rnd_state()
        obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=self._cfg.device)
        if obs_tensor.numel() == 0:
            fresh_count = 0
            input_shape = tuple(self._rnd_input_shape or ())
            rnd_inputs = torch.empty(
                (0, *input_shape), dtype=torch.float32, device=self._cfg.device
            )
        else:
            if obs_tensor.dim() == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
            rnd_obs = self._select_rnd_observation_features(obs_tensor)
            fresh_count = int(rnd_obs.shape[0])
            rnd_inputs = rnd_obs.reshape(fresh_count, *rnd_obs.shape[1:]).contiguous()

        if fresh_count > 0:
            self._ensure_rnd_model(rnd_inputs)
        if self._rnd_model is None:
            return {}

        pre_errors = self._rnd_errors_in_batches(self._rnd_model, rnd_inputs)
        pre_error_mean, pre_error_std = self._rnd_error_summary(pre_errors)

        update_proportion = float(self._cfg.get('rnd_predictor_update_proportion', 1.0))
        update_proportion = max(0.0, min(1.0, update_proportion))
        if fresh_count == 0 or update_proportion <= 0.0:
            selected_inputs = torch.empty(
                (0, *rnd_inputs.shape[1:]),
                dtype=torch.float32,
            )
        elif update_proportion >= 1.0:
            selected_inputs = rnd_inputs.detach().cpu()
        else:
            keep_mask = torch.rand(fresh_count, device=rnd_inputs.device) < update_proportion
            selected_inputs = rnd_inputs[keep_mask].detach().cpu()

        pending_inputs = self._rnd_fresh_pending_inputs
        if pending_inputs is None:
            queued_inputs = selected_inputs
        elif selected_inputs.shape[0] == 0:
            queued_inputs = pending_inputs
        else:
            if tuple(pending_inputs.shape[1:]) != tuple(selected_inputs.shape[1:]):
                raise ValueError(
                    "Fresh RND pending input shape changed from "
                    f"{tuple(pending_inputs.shape[1:])} to {tuple(selected_inputs.shape[1:])}."
                )
            queued_inputs = torch.cat([pending_inputs, selected_inputs], dim=0)

        if queued_inputs.shape[0] > 1:
            queued_inputs = queued_inputs[torch.randperm(queued_inputs.shape[0])]
        update_batch_size = max(1, int(self._cfg.get('rnd_fresh_update_batch_size', 64)))
        train_count = (int(queued_inputs.shape[0]) // update_batch_size) * update_batch_size
        train_inputs = queued_inputs[:train_count]
        self._rnd_fresh_pending_inputs = queued_inputs[train_count:].detach().clone()

        batch_losses = []
        for start in range(0, train_count, update_batch_size):
            batch = train_inputs[start:start + update_batch_size].to(self._cfg.device)
            self._rnd_model.train()
            prediction, target = self._rnd_model(batch)
            loss = (prediction - target).pow(2).mean()
            self._rnd_optimizer.zero_grad()
            loss.backward()
            self._rnd_optimizer.step()
            batch_losses.append(float(loss.item()))

        post_errors = self._rnd_errors_in_batches(self._rnd_model, rnd_inputs)
        post_error_mean, post_error_std = self._rnd_error_summary(post_errors)
        optimizer_step_count = len(batch_losses)
        selected_count = int(selected_inputs.shape[0])
        pending_count = int(self._rnd_fresh_pending_inputs.shape[0])
        actual_batch_size = float(update_batch_size) if optimizer_step_count > 0 else 0.0
        return {
            'rnd_fresh/observation_count': float(fresh_count),
            'rnd_fresh/selected_count': float(selected_count),
            'rnd_fresh/selected_fraction': float(selected_count / max(1, fresh_count)),
            'rnd_fresh/optimizer_step_count': float(optimizer_step_count),
            'rnd_fresh/trained_sample_count': float(train_count),
            'rnd_fresh/batch_size_min': actual_batch_size,
            'rnd_fresh/batch_size_mean': actual_batch_size,
            'rnd_fresh/batch_size_max': actual_batch_size,
            'rnd_fresh/pending_count': float(pending_count),
            'rnd_fresh/train_loss_mean': float(np.mean(batch_losses)) if batch_losses else 0.0,
            'rnd_fresh/pre_error_mean': pre_error_mean,
            'rnd_fresh/pre_error_std': pre_error_std,
            'rnd_fresh/post_error_mean': post_error_mean,
            'rnd_fresh/post_error_std': post_error_std,
            'rnd_fresh/post_pre_error_ratio': float(
                post_error_mean / max(pre_error_mean, 1e-12)
            ) if fresh_count > 0 else 0.0,
        }

    def _intrinsic_discount_factor(self) -> float:
        discount_factor = self._cfg.get('rnd_intrinsic_discount_factor', None)
        if discount_factor is None:
            discount_factor = self._cfg.discount_factor
        return float(discount_factor)

    def _intrinsic_td_steps(self, available_reward_steps: int) -> int:
        td_steps = self._cfg.get('rnd_intrinsic_td_steps', None)
        if td_steps is None:
            td_steps = self._cfg.td_steps
        return max(1, min(int(td_steps), int(available_reward_steps)))

    def _make_intrinsic_return_targets(
            self,
            intrinsic_rewards: torch.Tensor,
            bootstrap_values: Optional[torch.Tensor] = None,
            valid_reward_counts: Optional[torch.Tensor] = None,
            time_limit_truncated: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, reward_steps = intrinsic_rewards.shape
        num_unroll_steps = int(self._cfg.num_unroll_steps)
        value_targets = intrinsic_rewards.new_zeros((batch_size, num_unroll_steps + 1))
        bootstrap_discounts = intrinsic_rewards.new_zeros((batch_size, num_unroll_steps + 1))
        if reward_steps == 0:
            return value_targets

        td_steps = self._intrinsic_td_steps(reward_steps)
        discount_factor = self._intrinsic_discount_factor()
        valid_reward_counts, time_limit_truncated = self._intrinsic_target_boundary_info(
            batch_size, reward_steps, intrinsic_rewards.device, valid_reward_counts, time_limit_truncated
        )
        terminal_within_window = valid_reward_counts < reward_steps
        for batch_idx in range(batch_size):
            valid_count = int(valid_reward_counts[batch_idx].item())
            terminal_boundary = bool(terminal_within_window[batch_idx].item())
            timeout_boundary = bool(time_limit_truncated[batch_idx].item())
            for start_idx in range(num_unroll_steps + 1):
                bootstrap_index, horizon = self._intrinsic_bootstrap_index(
                    start_idx, td_steps, valid_count, reward_steps, terminal_boundary, timeout_boundary
                )
                if horizon <= 0:
                    continue
                end_idx = start_idx + horizon
                discounts = intrinsic_rewards.new_tensor(
                    [discount_factor ** power for power in range(horizon)]
                )
                value_targets[batch_idx, start_idx] = (
                    intrinsic_rewards[batch_idx, start_idx:end_idx] * discounts
                ).sum()
                if bootstrap_index is not None:
                    bootstrap_discounts[batch_idx, start_idx] = discount_factor ** horizon

        if bootstrap_values is None:
            return value_targets

        bootstrap_batch_size = bootstrap_values.shape[0]
        if bootstrap_batch_size == batch_size:
            expanded_value_targets = value_targets
        elif bootstrap_batch_size % batch_size == 0:
            repeat_count = bootstrap_batch_size // batch_size
            expanded_value_targets = self._repeat_tensor_for_members(value_targets, repeat_count)
        else:
            raise ValueError(
                f"Invalid intrinsic bootstrap batch size: {bootstrap_batch_size}, expected "
                f"{batch_size} or a multiple of it."
            )
        if bootstrap_batch_size == batch_size:
            expanded_bootstrap_discounts = bootstrap_discounts
        else:
            expanded_bootstrap_discounts = self._repeat_tensor_for_members(
                bootstrap_discounts, bootstrap_batch_size // batch_size
            )
        return expanded_value_targets + expanded_bootstrap_discounts * bootstrap_values

    def _make_intrinsic_monte_carlo_return_targets(
            self,
            intrinsic_rewards: torch.Tensor,
            valid_reward_counts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, reward_steps = intrinsic_rewards.shape
        num_unroll_steps = int(self._cfg.num_unroll_steps)
        value_targets = intrinsic_rewards.new_zeros((batch_size, num_unroll_steps + 1))
        if reward_steps == 0:
            return value_targets

        valid_reward_counts, _ = self._intrinsic_target_boundary_info(
            batch_size,
            reward_steps,
            intrinsic_rewards.device,
            valid_reward_counts,
            None,
        )
        discount_factor = self._intrinsic_discount_factor()
        for batch_idx in range(batch_size):
            valid_count = int(valid_reward_counts[batch_idx].item())
            if valid_count <= 0:
                continue
            for start_idx in range(num_unroll_steps + 1):
                if start_idx >= valid_count:
                    continue
                horizon = valid_count - start_idx
                discounts = intrinsic_rewards.new_tensor(
                    [discount_factor ** power for power in range(horizon)]
                )
                value_targets[batch_idx, start_idx] = (
                    intrinsic_rewards[batch_idx, start_idx:valid_count] * discounts
                ).sum()
        return value_targets

    def _intrinsic_target_boundary_info(
            self,
            batch_size: int,
            reward_steps: int,
            device: torch.device,
            valid_reward_counts: Optional[torch.Tensor],
            time_limit_truncated: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if valid_reward_counts is None:
            valid_reward_counts = torch.full((batch_size, ), reward_steps, dtype=torch.long, device=device)
        else:
            valid_reward_counts = torch.as_tensor(valid_reward_counts, dtype=torch.long, device=device)
            if valid_reward_counts.numel() != batch_size:
                raise ValueError(
                    f"Invalid valid_reward_counts size: {valid_reward_counts.numel()}, expected {batch_size}."
                )
            valid_reward_counts = valid_reward_counts.view(batch_size).clamp(min=0, max=reward_steps)

        if time_limit_truncated is None:
            time_limit_truncated = torch.zeros((batch_size, ), dtype=torch.bool, device=device)
        else:
            time_limit_truncated = torch.as_tensor(time_limit_truncated, dtype=torch.bool, device=device)
            if time_limit_truncated.numel() != batch_size:
                raise ValueError(
                    f"Invalid time_limit_truncated size: {time_limit_truncated.numel()}, expected {batch_size}."
                )
            time_limit_truncated = time_limit_truncated.view(batch_size)
        if not bool(self._cfg.get('bootstrap_at_time_limit', True)):
            time_limit_truncated = torch.zeros_like(time_limit_truncated)
        return valid_reward_counts, time_limit_truncated

    def _intrinsic_bootstrap_index(
            self,
            start_idx: int,
            td_steps: int,
            valid_count: int,
            reward_steps: int,
            terminal_within_window: bool,
            time_limit_truncated: bool,
    ) -> Tuple[Optional[int], int]:
        if start_idx >= valid_count:
            return None, 0
        end_idx = min(valid_count, start_idx + td_steps)
        if end_idx <= start_idx:
            return None, 0
        horizon = end_idx - start_idx
        if terminal_within_window and end_idx == valid_count and not time_limit_truncated:
            return None, horizon
        if end_idx > reward_steps:
            return None, horizon
        return end_idx, horizon

    def _make_intrinsic_bootstrap_values(
            self,
            obs_seq: torch.Tensor,
            reward_steps: int,
            td_steps: int,
            valid_reward_counts: Optional[torch.Tensor] = None,
            time_limit_truncated: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if not bool(self._cfg.get('intrinsic_value_bootstrap', self._cfg.get('rnd_intrinsic_bootstrap', True))):
            return None
        bootstrap_model = self._target_model
        if not hasattr(bootstrap_model, 'intrinsic_value_from_latent'):
            return None

        batch_size = obs_seq.shape[0]
        num_unroll_steps = int(self._cfg.num_unroll_steps)
        ensemble_size = int(self._cfg.model.ensemble_size)
        bootstrap_values = obs_seq.new_zeros((ensemble_size, batch_size, num_unroll_steps + 1))
        bootstrap_obs = []
        bootstrap_target_indices = []
        valid_reward_counts, time_limit_truncated = self._intrinsic_target_boundary_info(
            batch_size, reward_steps, obs_seq.device, valid_reward_counts, time_limit_truncated
        )
        terminal_within_window = valid_reward_counts < reward_steps

        for batch_idx in range(batch_size):
            valid_count = int(valid_reward_counts[batch_idx].item())
            terminal_boundary = bool(terminal_within_window[batch_idx].item())
            timeout_boundary = bool(time_limit_truncated[batch_idx].item())
            for start_idx in range(num_unroll_steps + 1):
                bootstrap_index, _ = self._intrinsic_bootstrap_index(
                    start_idx, td_steps, valid_count, reward_steps, terminal_boundary, timeout_boundary
                )
                if bootstrap_index is None or bootstrap_index >= obs_seq.shape[1]:
                    continue
                bootstrap_obs.append(obs_seq[batch_idx:batch_idx + 1, bootstrap_index])
                bootstrap_target_indices.append((batch_idx, start_idx))

        if not bootstrap_obs:
            return bootstrap_values.reshape(ensemble_size * batch_size, num_unroll_steps + 1)

        flat_bootstrap_obs = torch.cat(bootstrap_obs, dim=0)
        member_batch_size = flat_bootstrap_obs.shape[0]
        member_ids = [
            member_id
            for member_id in range(ensemble_size)
            for _ in range(member_batch_size)
        ]
        repeated_obs = self._repeat_tensor_for_members(flat_bootstrap_obs, ensemble_size)

        was_training = bootstrap_model.training
        try:
            bootstrap_model.eval()
            bootstrap_model.set_active_members(member_ids)
            with torch.no_grad():
                latent_state = bootstrap_model.initial_inference(repeated_obs, member_id=member_ids).latent_state
                value = bootstrap_model.intrinsic_value_from_latent(
                    latent_state, member_id=member_ids
                ).squeeze(-1).detach()
                value = value.reshape(ensemble_size, len(bootstrap_target_indices))
        finally:
            bootstrap_model.set_active_member(0)
            if was_training:
                bootstrap_model.train()

        for local_idx, (batch_idx, start_idx) in enumerate(bootstrap_target_indices):
            bootstrap_values[:, batch_idx, start_idx] = value[:, local_idx]
        return bootstrap_values.reshape(ensemble_size * batch_size, num_unroll_steps + 1).detach()

    def _clip_intrinsic_targets(
            self,
            reward_targets: torch.Tensor,
            value_targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        reward_clip = float(self._cfg.get('intrinsic_reward_target_clip', 0.0))
        value_clip = float(self._cfg.get('intrinsic_value_target_clip', 0.0))
        reward_unclipped_abs_max = float(reward_targets.detach().abs().max().item()) if reward_targets.numel() > 0 else 0.0
        value_unclipped_abs_max = float(value_targets.detach().abs().max().item()) if value_targets.numel() > 0 else 0.0
        if reward_clip > 0:
            reward_targets = reward_targets.clamp(min=-reward_clip, max=reward_clip)
        if value_clip > 0:
            value_targets = value_targets.clamp(min=-value_clip, max=value_clip)
        return reward_targets, value_targets, {
            'intrinsic/reward_target_unclipped_abs_max': reward_unclipped_abs_max,
            'intrinsic/value_target_unclipped_abs_max': value_unclipped_abs_max,
            'intrinsic/reward_target_clip': reward_clip,
            'intrinsic/value_target_clip': value_clip,
        }

    def _intrinsic_valid_reward_mask(
            self,
            current_batch,
            batch_size: int,
            reward_steps: int,
            device: torch.device,
    ) -> torch.Tensor:
        mask_batch = current_batch[2]
        mask_tensor = torch.as_tensor(mask_batch, dtype=torch.float32, device=device)
        base_mask_len = int(self._cfg.num_unroll_steps) + 1
        if mask_tensor.dim() == 2 and mask_tensor.shape[1] >= base_mask_len + reward_steps:
            return mask_tensor[:, base_mask_len:base_mask_len + reward_steps].clamp(min=0.0, max=1.0)
        return torch.ones((batch_size, reward_steps), dtype=torch.float32, device=device)

    def _infer_time_limit_truncated_from_obs_seq(
            self,
            obs_seq: torch.Tensor,
            valid_reward_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        valid_reward_counts = valid_reward_mask.sum(dim=1).long().clamp(min=0, max=valid_reward_mask.shape[1])
        terminal_within_window = valid_reward_counts < valid_reward_mask.shape[1]
        terminal_mode = str(
            self._cfg.get('intrinsic_terminal_observation_mode', 'mountain_car_goal')
        ).lower()
        if terminal_mode == 'time_limit_only':
            return valid_reward_counts, terminal_within_window
        if terminal_mode == 'terminal':
            return valid_reward_counts, torch.zeros_like(terminal_within_window)
        if terminal_mode != 'mountain_car_goal':
            raise ValueError(
                'intrinsic_terminal_observation_mode must be one of '
                "'mountain_car_goal', 'time_limit_only', or 'terminal', got "
                f'{terminal_mode!r}.'
            )
        if obs_seq.shape[-1] < 1:
            return valid_reward_counts, terminal_within_window

        batch_indices = torch.arange(obs_seq.shape[0], device=obs_seq.device)
        final_obs_indices = valid_reward_counts.clamp(min=0, max=obs_seq.shape[1] - 1)
        final_positions = obs_seq[batch_indices, final_obs_indices, 0]
        reached_goal = final_positions >= 0.5
        return valid_reward_counts, terminal_within_window & ~reached_goal

    def _prepare_rnd_targets_and_train(self, data) -> Tuple[dict, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self._rnd_enabled():
            return {}, None, None

        if not self._use_learned_intrinsic_model():
            return self._train_rnd_from_replay_batch_legacy(data), None, None

        current_batch = data[0]
        current_batch_core = self._current_batch_core(current_batch)
        intrinsic_reward_batch = self._selected_intrinsic_reward_batch(current_batch)
        obs_batch_ori = current_batch_core[0]
        obs_seq = self._extract_observation_sequence(obs_batch_ori)
        if obs_seq is None or obs_seq.shape[1] < 2:
            return {}, None, None

        num_unroll_steps = int(self._cfg.num_unroll_steps)
        max_reward_steps = obs_seq.shape[1] - 1
        intrinsic_reward_array = None
        if intrinsic_reward_batch is not None:
            intrinsic_reward_array = np.asarray(intrinsic_reward_batch, dtype=np.float32)
            if intrinsic_reward_array.ndim == 2 and intrinsic_reward_array.shape[1] > 0:
                max_reward_steps = min(max_reward_steps, int(intrinsic_reward_array.shape[1]))
        next_obs = obs_seq[:, 1:max_reward_steps + 1].contiguous()
        rnd_next_obs = self._select_rnd_observation_features(next_obs)
        flat_next_obs = self._flatten_observation_time_batch(rnd_next_obs)
        self._ensure_rnd_model(flat_next_obs)
        if self._rnd_model is None:
            return {}, None, None
        valid_reward_mask = self._intrinsic_valid_reward_mask(
            current_batch_core, obs_seq.shape[0], max_reward_steps, obs_seq.device
        )
        valid_reward_counts, time_limit_truncated = self._infer_time_limit_truncated_from_obs_seq(
            obs_seq, valid_reward_mask
        )
        flat_valid_reward_mask = valid_reward_mask.reshape(-1) > 0.5

        train_observation_mode = self._cfg.get('rnd_train_observation_mode', 'next')
        sample_indices = current_batch_core[3]
        rnd_target_model = self._rnd_target_model_for_targets()
        rnd_target_model.eval()
        with torch.no_grad():
            raw_errors = rnd_target_model.error(flat_next_obs).reshape(obs_seq.shape[0], max_reward_steps)

        if train_observation_mode == 'next':
            next_valid_mask = valid_reward_mask[:, 0] > 0.5
            rnd_train_inputs = rnd_next_obs[:, 0].contiguous()[next_valid_mask]
            first_keys = self._rnd_observation_keys(sample_indices, 1)
            next_valid_list = next_valid_mask.detach().cpu().tolist()
            rnd_sample_keys = [
                key for key, is_valid in zip(first_keys, next_valid_list)
                if is_valid
            ]
            rnd_stats_errors = raw_errors[:, 0][next_valid_mask]
        elif train_observation_mode == 'all':
            rnd_train_inputs = flat_next_obs[flat_valid_reward_mask]
            all_keys = self._rnd_observation_keys(sample_indices, max_reward_steps)
            valid_list = flat_valid_reward_mask.detach().cpu().tolist()
            rnd_sample_keys = [
                key for key, is_valid in zip(all_keys, valid_list)
                if is_valid
            ]
            rnd_stats_errors = raw_errors.reshape(-1)[flat_valid_reward_mask]
        else:
            raise ValueError(
                f"Unsupported rnd_train_observation_mode={train_observation_mode!r}; expected 'next' or 'all'."
            )
        fresh_collection_only = bool(
            self._cfg.get('rnd_train_from_fresh_collection_only', False)
        )
        if fresh_collection_only:
            rnd_stats_sample_count = 0
        else:
            rnd_stats_sample_count = self._update_rnd_bonus_stats_for_inputs(
                rnd_stats_errors, rnd_sample_keys
            )

        with torch.no_grad():
            use_stored_collection_rewards = (
                bool(self._cfg.get('rnd_use_stored_collection_rewards', False))
                and intrinsic_reward_array is not None
                and intrinsic_reward_array.ndim == 2
                and intrinsic_reward_array.shape[1] >= max_reward_steps
            )
            if use_stored_collection_rewards:
                stored_rewards = torch.as_tensor(
                    intrinsic_reward_array[:, :max_reward_steps],
                    dtype=torch.float32,
                    device=obs_seq.device,
                )
                intrinsic_rewards_all = stored_rewards * valid_reward_mask
            else:
                intrinsic_rewards_all = self._normalize_rnd_errors(raw_errors) * valid_reward_mask
            reward_target_steps = min(num_unroll_steps, max_reward_steps)
            intrinsic_reward_targets = intrinsic_rewards_all.new_zeros((intrinsic_rewards_all.shape[0], num_unroll_steps))
            intrinsic_reward_targets[:, :reward_target_steps] = intrinsic_rewards_all[:, :reward_target_steps]
            td_steps = self._intrinsic_td_steps(max_reward_steps)

        intrinsic_target_mode = str(self._cfg.get('rnd_intrinsic_target_mode', 'td') or 'td').lower()
        if intrinsic_target_mode not in ['td', 'monte_carlo']:
            raise ValueError(
                f"Unsupported rnd_intrinsic_target_mode={intrinsic_target_mode!r}; "
                "expected 'td' or 'monte_carlo'."
            )
        bootstrap_values = None
        if intrinsic_target_mode == 'td':
            bootstrap_values = self._make_intrinsic_bootstrap_values(
                obs_seq,
                max_reward_steps,
                td_steps,
                valid_reward_counts=valid_reward_counts,
                time_limit_truncated=time_limit_truncated,
            )
        with torch.no_grad():
            if intrinsic_target_mode == 'monte_carlo':
                intrinsic_values = self._make_intrinsic_monte_carlo_return_targets(
                    intrinsic_rewards_all,
                    valid_reward_counts=valid_reward_counts,
                )
            else:
                intrinsic_values = self._make_intrinsic_return_targets(
                    intrinsic_rewards_all,
                    bootstrap_values,
                    valid_reward_counts=valid_reward_counts,
                    time_limit_truncated=time_limit_truncated,
                )
            intrinsic_reward_targets, intrinsic_values, intrinsic_clip_log = self._clip_intrinsic_targets(
                intrinsic_reward_targets, intrinsic_values
            )

        if fresh_collection_only:
            rnd_log = self._empty_rnd_replay_train_log()
        else:
            rnd_log = self._train_rnd_on_inputs(rnd_train_inputs, sample_keys=rnd_sample_keys)
        rnd_log['rnd/stats_sample_count'] = float(rnd_stats_sample_count)
        if bool(self._cfg.get('rnd_target_snapshot', True)) and self._rnd_target_snapshot_last_update >= 0:
            rnd_log['rnd/target_snapshot_age'] = float(
                max(0, int(getattr(self, '_learn_update_count', 0)) - int(self._rnd_target_snapshot_last_update))
            )
        else:
            rnd_log['rnd/target_snapshot_age'] = -1.0
        if bootstrap_values is None:
            bootstrap_value_mean = 0.0
            bootstrap_value_std = 0.0
        else:
            bootstrap_value_mean = float(bootstrap_values.mean().item())
            bootstrap_value_std = float(bootstrap_values.std(unbiased=False).item())
        valid_intrinsic_rewards = intrinsic_rewards_all[valid_reward_mask > 0.5]
        if valid_intrinsic_rewards.numel() == 0:
            stored_reward_mean = 0.0
            stored_reward_std = 0.0
            stored_reward_max = 0.0
            stored_reward_nonzero_fraction = 0.0
        else:
            stored_reward_mean = float(valid_intrinsic_rewards.mean().item())
            stored_reward_std = float(valid_intrinsic_rewards.std(unbiased=False).item())
            stored_reward_max = float(valid_intrinsic_rewards.max().item())
            stored_reward_nonzero_fraction = float((valid_intrinsic_rewards > 0.0).float().mean().item())
        valid_raw_errors = raw_errors[valid_reward_mask > 0.5]
        raw_error_mean = float(valid_raw_errors.mean().item()) if valid_raw_errors.numel() > 0 else 0.0
        raw_error_max = float(valid_raw_errors.max().item()) if valid_raw_errors.numel() > 0 else 0.0
        rnd_log.update(
            {
                'rnd/target_mode_monte_carlo': float(intrinsic_target_mode == 'monte_carlo'),
                'rnd/using_stored_collection_rewards': float(use_stored_collection_rewards),
                'rnd/raw_error_target_mean': raw_error_mean,
                'rnd/raw_error_target_max': raw_error_max,
                'intrinsic/stored_reward_mean': stored_reward_mean,
                'intrinsic/stored_reward_std': stored_reward_std,
                'intrinsic/stored_reward_max': stored_reward_max,
                'intrinsic/stored_reward_nonzero_fraction': stored_reward_nonzero_fraction,
                'intrinsic/reward_target_mean': float(intrinsic_reward_targets.mean().item()),
                'intrinsic/reward_target_std': float(intrinsic_reward_targets.std(unbiased=False).item()),
                'intrinsic/value_target_mean': float(intrinsic_values.mean().item()),
                'intrinsic/value_target_std': float(intrinsic_values.std(unbiased=False).item()),
                'intrinsic/bootstrap_enabled': float(bootstrap_values is not None),
                'intrinsic/bootstrap_value_mean': bootstrap_value_mean,
                'intrinsic/bootstrap_value_std': bootstrap_value_std,
                'intrinsic/valid_reward_steps_mean': float(valid_reward_counts.float().mean().item()),
                'intrinsic/time_limit_bootstrap_fraction': float(time_limit_truncated.float().mean().item()),
            }
        )
        rnd_log.update(intrinsic_clip_log)
        return rnd_log, intrinsic_reward_targets.detach(), intrinsic_values.detach()

    def _prepare_dynamics_disagreement_targets(
            self, data
    ) -> Tuple[dict, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not bool(self._cfg.get('dynamics_disagreement_train_intrinsic_value', False)):
            return {}, None, None

        current_batch = data[0]
        current_batch_core = self._current_batch_core(current_batch)
        intrinsic_reward_batch = self._selected_intrinsic_reward_batch(current_batch)
        if intrinsic_reward_batch is None:
            return {
                'intrinsic_dd/missing_selected_reward_batch': 1.0,
            }, None, None

        obs_seq = self._extract_observation_sequence(current_batch_core[0])
        if obs_seq is None or obs_seq.shape[1] < 2:
            return {}, None, None

        num_unroll_steps = int(self._cfg.num_unroll_steps)
        max_reward_steps = min(obs_seq.shape[1] - 1, int(np.asarray(intrinsic_reward_batch).shape[1]))
        if max_reward_steps <= 0:
            return {}, None, None

        selected_rewards = torch.as_tensor(
            np.asarray(intrinsic_reward_batch[:, :max_reward_steps], dtype=np.float32),
            dtype=torch.float32,
            device=obs_seq.device,
        )
        valid_reward_mask = self._intrinsic_valid_reward_mask(
            current_batch_core, obs_seq.shape[0], max_reward_steps, obs_seq.device
        )
        selected_rewards = selected_rewards * valid_reward_mask
        valid_reward_counts, time_limit_truncated = self._infer_time_limit_truncated_from_obs_seq(
            obs_seq, valid_reward_mask
        )
        reward_target_steps = min(num_unroll_steps, max_reward_steps)
        intrinsic_reward_targets = selected_rewards.new_zeros((selected_rewards.shape[0], num_unroll_steps))
        intrinsic_reward_targets[:, :reward_target_steps] = selected_rewards[:, :reward_target_steps]
        td_steps = self._intrinsic_td_steps(max_reward_steps)

        bootstrap_values = self._make_intrinsic_bootstrap_values(
            obs_seq,
            max_reward_steps,
            td_steps,
            valid_reward_counts=valid_reward_counts,
            time_limit_truncated=time_limit_truncated,
        )
        with torch.no_grad():
            intrinsic_values = self._make_intrinsic_return_targets(
                selected_rewards,
                bootstrap_values,
                valid_reward_counts=valid_reward_counts,
                time_limit_truncated=time_limit_truncated,
            )
            intrinsic_reward_targets, intrinsic_values, intrinsic_clip_log = self._clip_intrinsic_targets(
                intrinsic_reward_targets, intrinsic_values
            )

        if bootstrap_values is None:
            bootstrap_value_mean = 0.0
            bootstrap_value_std = 0.0
        else:
            bootstrap_value_mean = float(bootstrap_values.mean().item())
            bootstrap_value_std = float(bootstrap_values.std(unbiased=False).item())

        selected_nonzero = selected_rewards[valid_reward_mask > 0.5]
        if selected_nonzero.numel() == 0:
            reward_mean = 0.0
            reward_std = 0.0
            reward_max = 0.0
        else:
            reward_mean = float(selected_nonzero.mean().item())
            reward_std = float(selected_nonzero.std(unbiased=False).item())
            reward_max = float(selected_nonzero.max().item())

        log = {
            'intrinsic_dd/selected_reward_mean': reward_mean,
            'intrinsic_dd/selected_reward_std': reward_std,
            'intrinsic_dd/selected_reward_max': reward_max,
            'intrinsic_dd/value_target_mean': float(intrinsic_values.mean().item()),
            'intrinsic_dd/value_target_std': float(intrinsic_values.std(unbiased=False).item()),
            'intrinsic_dd/bootstrap_value_mean': bootstrap_value_mean,
            'intrinsic_dd/bootstrap_value_std': bootstrap_value_std,
            'intrinsic_dd/valid_reward_steps_mean': float(valid_reward_counts.float().mean().item()),
            'intrinsic_dd/time_limit_bootstrap_fraction': float(time_limit_truncated.float().mean().item()),
            'intrinsic_dd/reward_space_projection': float(
                str(self._cfg.get('dynamics_disagreement_space', 'latent') or 'latent').lower() == 'projection'
            ),
        }
        log.update(intrinsic_clip_log)
        return log, intrinsic_reward_targets.detach(), intrinsic_values.detach()

    def _prepare_qstd_targets(
            self, data
    ) -> Tuple[dict, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not bool(self._cfg.get('qstd_exploration', False)):
            return {}, None, None

        current_batch = data[0]
        current_batch_core = self._current_batch_core(current_batch)
        obs_seq = self._extract_observation_sequence(current_batch_core[0])
        if obs_seq is None or obs_seq.shape[1] < 2:
            return {}, None, None

        num_unroll_steps = int(self._cfg.num_unroll_steps)
        recompute_rewards = self._qstd_recompute_replay_rewards_enabled()
        qstd_target_log = {}
        if recompute_rewards:
            action_batch = np.asarray(current_batch_core[1])
            action_batch = action_batch.reshape(action_batch.shape[0], -1)
            max_reward_steps = min(obs_seq.shape[1] - 1, int(action_batch.shape[1]))
        else:
            intrinsic_reward_batch = self._selected_intrinsic_reward_batch(current_batch)
            if intrinsic_reward_batch is None:
                return {
                    'intrinsic_qstd/missing_selected_reward_batch': 1.0,
                }, None, None
            max_reward_steps = min(obs_seq.shape[1] - 1, int(np.asarray(intrinsic_reward_batch).shape[1]))
        if max_reward_steps <= 0:
            return {}, None, None

        valid_reward_mask = self._intrinsic_valid_reward_mask(
            current_batch_core, obs_seq.shape[0], max_reward_steps, obs_seq.device
        )
        if recompute_rewards:
            selected_rewards, raw_qstd_rewards, cache_stats = self._one_step_action_qstd_rewards(
                obs_seq[:, :max_reward_steps],
                torch.as_tensor(
                    action_batch[:, :max_reward_steps], dtype=torch.long, device=obs_seq.device
                ),
                valid_reward_mask,
            )
            valid_raw_qstd = raw_qstd_rewards[valid_reward_mask > 0.5]
            qstd_target_log = {
                'intrinsic_qstd/recomputed_one_step_action': 1.0,
                'intrinsic_qstd/raw_reward_mean': (
                    float(valid_raw_qstd.mean().item()) if valid_raw_qstd.numel() > 0 else 0.0
                ),
                'intrinsic_qstd/raw_reward_std': (
                    float(valid_raw_qstd.std(unbiased=False).item()) if valid_raw_qstd.numel() > 0 else 0.0
                ),
                'intrinsic_qstd/raw_reward_max': (
                    float(valid_raw_qstd.max().item()) if valid_raw_qstd.numel() > 0 else 0.0
                ),
                'intrinsic_qstd/cache_hit_fraction': cache_stats['cache_hit_fraction'],
                'intrinsic_qstd/cache_size': cache_stats['cache_size'],
                'intrinsic_qstd/target_snapshot_generation': float(
                    getattr(self, '_qstd_target_snapshot_generation', 0)
                ),
            }
        else:
            selected_rewards = torch.as_tensor(
                np.asarray(intrinsic_reward_batch[:, :max_reward_steps], dtype=np.float32),
                dtype=torch.float32,
                device=obs_seq.device,
            )
        selected_rewards = selected_rewards * valid_reward_mask
        valid_reward_counts, time_limit_truncated = self._infer_time_limit_truncated_from_obs_seq(
            obs_seq, valid_reward_mask
        )
        reward_target_steps = min(num_unroll_steps, max_reward_steps)
        intrinsic_reward_targets = selected_rewards.new_zeros((selected_rewards.shape[0], num_unroll_steps))
        intrinsic_reward_targets[:, :reward_target_steps] = selected_rewards[:, :reward_target_steps]
        td_steps = self._intrinsic_td_steps(max_reward_steps)

        intrinsic_target_mode = str(
            self._cfg.get(
                'qstd_intrinsic_target_mode', self._cfg.get('rnd_intrinsic_target_mode', 'td')
            ) or 'td'
        ).lower()
        if intrinsic_target_mode not in ['td', 'monte_carlo']:
            raise ValueError(
                f"Unsupported qstd_intrinsic_target_mode={intrinsic_target_mode!r}; "
                "expected 'td' or 'monte_carlo'."
            )
        bootstrap_values = None
        if intrinsic_target_mode == 'td':
            bootstrap_values = self._make_intrinsic_bootstrap_values(
                obs_seq,
                max_reward_steps,
                td_steps,
                valid_reward_counts=valid_reward_counts,
                time_limit_truncated=time_limit_truncated,
            )
        with torch.no_grad():
            if intrinsic_target_mode == 'monte_carlo':
                intrinsic_values = self._make_intrinsic_monte_carlo_return_targets(
                    selected_rewards,
                    valid_reward_counts=valid_reward_counts,
                )
            else:
                intrinsic_values = self._make_intrinsic_return_targets(
                    selected_rewards,
                    bootstrap_values,
                    valid_reward_counts=valid_reward_counts,
                    time_limit_truncated=time_limit_truncated,
                )
            intrinsic_reward_targets, intrinsic_values, intrinsic_clip_log = self._clip_intrinsic_targets(
                intrinsic_reward_targets, intrinsic_values
            )

        if bootstrap_values is None:
            bootstrap_value_mean = 0.0
            bootstrap_value_std = 0.0
        else:
            bootstrap_value_mean = float(bootstrap_values.mean().item())
            bootstrap_value_std = float(bootstrap_values.std(unbiased=False).item())

        valid_rewards = selected_rewards[valid_reward_mask > 0.5]
        if valid_rewards.numel() == 0:
            reward_mean = 0.0
            reward_std = 0.0
            reward_max = 0.0
            nonzero_fraction = 0.0
        else:
            reward_mean = float(valid_rewards.mean().item())
            reward_std = float(valid_rewards.std(unbiased=False).item())
            reward_max = float(valid_rewards.max().item())
            nonzero_fraction = float((valid_rewards > 0.0).float().mean().item())

        log = {
            'intrinsic_qstd/selected_reward_mean': reward_mean,
            'intrinsic_qstd/selected_reward_std': reward_std,
            'intrinsic_qstd/selected_reward_max': reward_max,
            'intrinsic_qstd/selected_reward_nonzero_fraction': nonzero_fraction,
            'intrinsic_qstd/value_target_mean': float(intrinsic_values.mean().item()),
            'intrinsic_qstd/value_target_std': float(intrinsic_values.std(unbiased=False).item()),
            'intrinsic_qstd/bootstrap_value_mean': bootstrap_value_mean,
            'intrinsic_qstd/bootstrap_value_std': bootstrap_value_std,
            'intrinsic_qstd/valid_reward_steps_mean': float(valid_reward_counts.float().mean().item()),
            'intrinsic_qstd/time_limit_bootstrap_fraction': float(time_limit_truncated.float().mean().item()),
            'intrinsic_qstd/target_mode_monte_carlo': float(intrinsic_target_mode == 'monte_carlo'),
            'intrinsic_qstd/reward_aggregation_max': float(
                str(self._cfg.get('qstd_reward_aggregation', 'max')).lower() == 'max'
            ),
        }
        log.update(qstd_target_log)
        log.update(intrinsic_clip_log)
        return log, intrinsic_reward_targets.detach(), intrinsic_values.detach()

    def _state_value_disagreement_from_observations(
            self,
            observations: torch.Tensor,
            model: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return raw extrinsic value disagreement for real observations."""
        ensemble_size = int(self._cfg.model.ensemble_size)
        flat_observations = observations.reshape(-1, observations.shape[-1]).float()
        if flat_observations.shape[0] == 0:
            empty_values = flat_observations.new_zeros((ensemble_size, 0))
            return flat_observations.new_zeros((0, )), empty_values

        target_model = self._target_model if model is None else model
        inference_batch_size = max(
            1, int(self._cfg.get('state_value_disagreement_inference_batch_size', 1024))
        )
        value_chunks = []
        was_training = target_model.training
        try:
            target_model.eval()
            with torch.no_grad():
                for start in range(0, flat_observations.shape[0], inference_batch_size):
                    obs_chunk = flat_observations[start:start + inference_batch_size]
                    chunk_size = int(obs_chunk.shape[0])
                    member_ids = [
                        member_id
                        for member_id in range(ensemble_size)
                        for _ in range(chunk_size)
                    ]
                    repeated_obs = self._repeat_tensor_for_members(obs_chunk, ensemble_size)
                    target_model.set_active_members(member_ids)
                    output = target_model.initial_inference(repeated_obs, member_id=member_ids)
                    values = self.value_inverse_scalar_transform_handle(output.value).detach()
                    values = values.reshape(ensemble_size, chunk_size, -1)
                    if values.shape[-1] != 1:
                        raise ValueError(
                            "State-value-disagreement targets require scalar decoded values, "
                            f"got shape {tuple(values.shape)}."
                        )
                    value_chunks.append(values.squeeze(-1))
        finally:
            target_model.set_active_member(0)
            if was_training:
                target_model.train()

        member_values = torch.cat(value_chunks, dim=1)
        disagreement = member_values.std(dim=0, unbiased=False)
        return disagreement.detach(), member_values.detach()

    @staticmethod
    def _intrinsic_diagnostic_correlation(x: torch.Tensor, y: torch.Tensor) -> float:
        x = x.detach().float().reshape(-1)
        y = y.detach().float().reshape(-1)
        finite = torch.isfinite(x) & torch.isfinite(y)
        x = x[finite]
        y = y[finite]
        if x.numel() < 2:
            return 0.0
        x = x - x.mean()
        y = y - y.mean()
        denominator = x.square().sum().sqrt() * y.square().sum().sqrt()
        if float(denominator.item()) <= 1e-8:
            return 0.0
        return float((x * y).sum().div(denominator).item())

    def _intrinsic_aux_gradient_diagnostics(
            self,
            private_reward_loss: torch.Tensor,
            auxiliary_loss: torch.Tensor,
    ) -> Dict[str, float]:
        """Compare private-reward and auxiliary gradients on h and g."""
        interval = max(
            0, int(self._cfg.get('rnd_landscape_query_gradient_diagnostics_interval', 0))
        )
        if (
                interval <= 0
                or not self._rnd_landscape_query_enabled()
                or int(getattr(self, '_learn_update_count', 0)) % interval != 0
        ):
            return {}

        parameter_groups = {'representation': [], 'dynamics': []}
        for name, parameter in self._learn_model.named_parameters():
            if not parameter.requires_grad:
                continue
            if 'representation_network' in name:
                parameter_groups['representation'].append(parameter)
            elif 'dynamics_network' in name:
                parameter_groups['dynamics'].append(parameter)

        parameters = parameter_groups['representation'] + parameter_groups['dynamics']
        if not parameters:
            return {}

        def gradients(loss: torch.Tensor) -> List[Optional[torch.Tensor]]:
            if not loss.requires_grad:
                return [None] * len(parameters)
            return list(
                torch.autograd.grad(
                    loss,
                    parameters,
                    retain_graph=True,
                    allow_unused=True,
                )
            )

        private_gradients = gradients(private_reward_loss)
        auxiliary_gradients = gradients(auxiliary_loss)
        log = {}
        offset = 0
        for group_name, group_parameters in parameter_groups.items():
            count = len(group_parameters)
            private_group = private_gradients[offset:offset + count]
            auxiliary_group = auxiliary_gradients[offset:offset + count]
            offset += count
            private_norm_sq = private_reward_loss.new_zeros(())
            auxiliary_norm_sq = private_reward_loss.new_zeros(())
            dot = private_reward_loss.new_zeros(())
            for private_gradient, auxiliary_gradient in zip(
                    private_group, auxiliary_group
            ):
                if private_gradient is not None:
                    private_norm_sq = (
                        private_norm_sq
                        + private_gradient.detach().float().square().sum()
                    )
                if auxiliary_gradient is not None:
                    auxiliary_norm_sq = (
                        auxiliary_norm_sq
                        + auxiliary_gradient.detach().float().square().sum()
                    )
                if private_gradient is not None and auxiliary_gradient is not None:
                    dot = dot + (
                        private_gradient.detach().float()
                        * auxiliary_gradient.detach().float()
                    ).sum()
            private_norm = private_norm_sq.sqrt()
            auxiliary_norm = auxiliary_norm_sq.sqrt()
            denominator = private_norm * auxiliary_norm
            cosine = (
                dot / denominator.clamp_min(1e-12)
                if float(denominator.item()) > 1e-12
                else dot.new_zeros(())
            )
            prefix = f'intrinsic_gradient/{group_name}'
            log[f'{prefix}_private_reward_norm'] = float(private_norm.item())
            log[f'{prefix}_auxiliary_norm'] = float(auxiliary_norm.item())
            log[f'{prefix}_private_auxiliary_cosine'] = float(cosine.item())
        return log

    def _prepare_state_value_disagreement_targets(
            self, data
    ) -> Tuple[dict, Optional[torch.Tensor], Optional[torch.Tensor]]:
        current_batch = data[0]
        current_batch_core = self._current_batch_core(current_batch)
        stored_intrinsic_reward_batch = self._selected_intrinsic_reward_batch(current_batch)
        obs_seq = self._extract_observation_sequence(current_batch_core[0])
        if obs_seq is None or obs_seq.shape[1] < 2:
            return {}, None, None

        ensemble_size = int(self._cfg.model.ensemble_size)
        if ensemble_size < 2:
            return {'intrinsic_state_value_disagreement/ensemble_too_small': 1.0}, None, None

        num_unroll_steps = int(self._cfg.num_unroll_steps)
        available_reward_steps = int(obs_seq.shape[1] - 1)
        intrinsic_target_mode = str(
            self._cfg.get(
                'state_value_disagreement_intrinsic_target_mode',
                self._cfg.get('rnd_intrinsic_target_mode', 'td'),
            ) or 'td'
        ).lower()
        if intrinsic_target_mode not in ['td', 'monte_carlo']:
            raise ValueError(
                "state_value_disagreement_intrinsic_target_mode must be 'td' or 'monte_carlo', "
                f"got {intrinsic_target_mode!r}."
            )
        if intrinsic_target_mode == 'td':
            required_reward_steps = num_unroll_steps + self._intrinsic_td_steps(available_reward_steps)
            max_reward_steps = min(available_reward_steps, required_reward_steps)
        else:
            max_reward_steps = available_reward_steps
        stored_reward_array = None
        if stored_intrinsic_reward_batch is not None:
            stored_reward_array = np.asarray(stored_intrinsic_reward_batch, dtype=np.float32)
            if stored_reward_array.ndim == 2 and stored_reward_array.shape[1] > 0:
                max_reward_steps = min(max_reward_steps, int(stored_reward_array.shape[1]))
        next_obs = obs_seq[:, 1:max_reward_steps + 1].contiguous()
        valid_reward_mask = self._intrinsic_valid_reward_mask(
            current_batch_core, obs_seq.shape[0], max_reward_steps, obs_seq.device
        )
        valid_mask = valid_reward_mask > 0.5
        valid_next_obs = next_obs[valid_mask]

        reward_scale = float(self._cfg.get('state_value_disagreement_reward_scale', 1.0))
        source_reward_clip = self._cfg.get('state_value_disagreement_reward_clip', None)
        use_stored_collection_rewards = bool(
            self._cfg.get('state_value_disagreement_use_stored_collection_rewards', False)
            and stored_reward_array is not None
            and stored_reward_array.ndim == 2
            and stored_reward_array.shape[1] >= max_reward_steps
        )
        raw_rewards = next_obs.new_zeros((obs_seq.shape[0], max_reward_steps))
        member_values = None
        if use_stored_collection_rewards:
            intrinsic_rewards_all = torch.as_tensor(
                stored_reward_array[:, :max_reward_steps],
                dtype=torch.float32,
                device=obs_seq.device,
            )
            # Collection already paid for the ensemble inference and stored the resulting
            # scaled reward. Do not re-evaluate every replay observation merely for logs.
            if abs(reward_scale) > 1e-12:
                raw_rewards = intrinsic_rewards_all / reward_scale
            else:
                raw_rewards = intrinsic_rewards_all.clone()
        else:
            member_values = next_obs.new_zeros(
                (ensemble_size, obs_seq.shape[0], max_reward_steps)
            )
            if valid_next_obs.shape[0] > 0:
                valid_raw_rewards, valid_member_values = self._state_value_disagreement_from_observations(
                    valid_next_obs
                )
                raw_rewards[valid_mask] = valid_raw_rewards
                for member_id in range(ensemble_size):
                    member_values[member_id][valid_mask] = valid_member_values[member_id]
            intrinsic_rewards_all = raw_rewards * reward_scale
            if source_reward_clip is not None and abs(float(source_reward_clip)) > 0.0:
                intrinsic_rewards_all = intrinsic_rewards_all.clamp(
                    min=0.0, max=abs(float(source_reward_clip))
                )
        intrinsic_rewards_all = intrinsic_rewards_all * valid_reward_mask

        valid_reward_counts, time_limit_truncated = self._infer_time_limit_truncated_from_obs_seq(
            obs_seq, valid_reward_mask
        )
        reward_target_steps = min(num_unroll_steps, max_reward_steps)
        intrinsic_reward_targets = intrinsic_rewards_all.new_zeros(
            (intrinsic_rewards_all.shape[0], num_unroll_steps)
        )
        intrinsic_reward_targets[:, :reward_target_steps] = intrinsic_rewards_all[:, :reward_target_steps]

        td_steps = self._intrinsic_td_steps(max_reward_steps)
        bootstrap_values = None
        if intrinsic_target_mode == 'td':
            bootstrap_values = self._make_intrinsic_bootstrap_values(
                obs_seq,
                max_reward_steps,
                td_steps,
                valid_reward_counts=valid_reward_counts,
                time_limit_truncated=time_limit_truncated,
            )
        with torch.no_grad():
            if intrinsic_target_mode == 'monte_carlo':
                intrinsic_values = self._make_intrinsic_monte_carlo_return_targets(
                    intrinsic_rewards_all,
                    valid_reward_counts=valid_reward_counts,
                )
            else:
                intrinsic_values = self._make_intrinsic_return_targets(
                    intrinsic_rewards_all,
                    bootstrap_values,
                    valid_reward_counts=valid_reward_counts,
                    time_limit_truncated=time_limit_truncated,
                )
            intrinsic_reward_targets, intrinsic_values, intrinsic_clip_log = self._clip_intrinsic_targets(
                intrinsic_reward_targets, intrinsic_values
            )

        valid_raw_rewards = raw_rewards[valid_mask]
        valid_scaled_rewards = intrinsic_rewards_all[valid_mask]
        first_step_mask = valid_mask[:, 0]
        first_step_rewards = raw_rewards[:, 0][first_step_mask]
        bootstrap_value_mean = 0.0 if bootstrap_values is None else float(bootstrap_values.mean().item())
        bootstrap_value_std = (
            0.0 if bootstrap_values is None else float(bootstrap_values.std(unbiased=False).item())
        )

        def percentile(values: torch.Tensor, quantile: float) -> float:
            if values.numel() == 0:
                return 0.0
            return float(torch.quantile(values.float(), quantile).item())

        log = {
            'intrinsic_state_value_disagreement/raw_reward_mean': (
                float(valid_raw_rewards.mean().item()) if valid_raw_rewards.numel() > 0 else 0.0
            ),
            'intrinsic_state_value_disagreement/raw_reward_std': (
                float(valid_raw_rewards.std(unbiased=False).item()) if valid_raw_rewards.numel() > 0 else 0.0
            ),
            'intrinsic_state_value_disagreement/raw_reward_min': (
                float(valid_raw_rewards.min().item()) if valid_raw_rewards.numel() > 0 else 0.0
            ),
            'intrinsic_state_value_disagreement/raw_reward_max': (
                float(valid_raw_rewards.max().item()) if valid_raw_rewards.numel() > 0 else 0.0
            ),
            'intrinsic_state_value_disagreement/raw_reward_p50': percentile(valid_raw_rewards, 0.50),
            'intrinsic_state_value_disagreement/raw_reward_p90': percentile(valid_raw_rewards, 0.90),
            'intrinsic_state_value_disagreement/raw_reward_p99': percentile(valid_raw_rewards, 0.99),
            'intrinsic_state_value_disagreement/first_step_reward_mean': (
                float(first_step_rewards.mean().item()) if first_step_rewards.numel() > 0 else 0.0
            ),
            'intrinsic_state_value_disagreement/first_step_reward_p90': percentile(first_step_rewards, 0.90),
            'intrinsic_state_value_disagreement/scaled_reward_mean': (
                float(valid_scaled_rewards.mean().item()) if valid_scaled_rewards.numel() > 0 else 0.0
            ),
            'intrinsic_state_value_disagreement/reward_scale': reward_scale,
            'intrinsic_state_value_disagreement/source_reward_clip': (
                0.0 if source_reward_clip is None else float(source_reward_clip)
            ),
            'intrinsic_state_value_disagreement/using_stored_collection_rewards': float(
                use_stored_collection_rewards
            ),
            'intrinsic_state_value_disagreement/value_target_mean': float(intrinsic_values.mean().item()),
            'intrinsic_state_value_disagreement/value_target_std': float(
                intrinsic_values.std(unbiased=False).item()
            ),
            'intrinsic_state_value_disagreement/value_target_p90': percentile(intrinsic_values, 0.90),
            'intrinsic_state_value_disagreement/bootstrap_value_mean': bootstrap_value_mean,
            'intrinsic_state_value_disagreement/bootstrap_value_std': bootstrap_value_std,
            'intrinsic_state_value_disagreement/valid_reward_steps_mean': float(
                valid_reward_counts.float().mean().item()
            ),
            'intrinsic_state_value_disagreement/time_limit_bootstrap_fraction': float(
                time_limit_truncated.float().mean().item()
            ),
            'intrinsic_state_value_disagreement/target_mode_monte_carlo': float(
                intrinsic_target_mode == 'monte_carlo'
            ),
        }
        if member_values is not None:
            valid_member_values = member_values[:, valid_mask]
            for member_id in range(ensemble_size):
                values = valid_member_values[member_id]
                log[f'intrinsic_state_value_disagreement/member_{member_id}_value_mean'] = (
                    float(values.mean().item()) if values.numel() > 0 else 0.0
                )
                log[f'intrinsic_state_value_disagreement/member_{member_id}_value_std'] = (
                    float(values.std(unbiased=False).item()) if values.numel() > 0 else 0.0
                )

        if valid_next_obs.numel() > 0:
            log['intrinsic_state_value_disagreement/reward_position_correlation'] = (
                self._intrinsic_diagnostic_correlation(valid_raw_rewards, valid_next_obs[:, 0])
            )
            if valid_next_obs.shape[1] > 1:
                log['intrinsic_state_value_disagreement/reward_abs_velocity_correlation'] = (
                    self._intrinsic_diagnostic_correlation(valid_raw_rewards, valid_next_obs[:, 1].abs())
                )
            if valid_raw_rewards.numel() > 0:
                top_threshold = torch.quantile(valid_raw_rewards.float(), 0.90)
                top_mask = valid_raw_rewards >= top_threshold
                log['intrinsic_state_value_disagreement/top10_position_mean'] = float(
                    valid_next_obs[top_mask, 0].mean().item()
                )
                if valid_next_obs.shape[1] > 1:
                    log['intrinsic_state_value_disagreement/top10_abs_velocity_mean'] = float(
                        valid_next_obs[top_mask, 1].abs().mean().item()
                    )
        log.update(intrinsic_clip_log)
        return log, intrinsic_reward_targets.detach(), intrinsic_values.detach()

    def _prepare_state_density_targets(
            self, data
    ) -> Tuple[dict, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Build targets by reevaluating replay states under the current frozen density."""
        current_batch = data[0]
        current_batch_core = self._current_batch_core(current_batch)
        obs_seq = self._extract_observation_sequence(current_batch_core[0])
        if obs_seq is None or obs_seq.shape[1] < 2:
            return {}, None, None
        if getattr(self, '_state_density_snapshot_counts', None) is None:
            return {'intrinsic_state_density/missing_current_snapshot': 1.0}, None, None

        num_unroll_steps = int(self._cfg.num_unroll_steps)
        available_reward_steps = int(obs_seq.shape[1] - 1)
        target_mode = str(
            self._cfg.get('state_density_intrinsic_target_mode', 'td') or 'td'
        ).lower()
        if target_mode not in ['td', 'monte_carlo']:
            raise ValueError(
                "state_density_intrinsic_target_mode must be 'td' or 'monte_carlo', "
                f"got {target_mode!r}."
            )
        if target_mode == 'td':
            required_steps = num_unroll_steps + self._intrinsic_td_steps(available_reward_steps)
            max_reward_steps = min(available_reward_steps, required_steps)
        else:
            max_reward_steps = available_reward_steps

        valid_reward_mask = self._intrinsic_valid_reward_mask(
            current_batch_core, obs_seq.shape[0], max_reward_steps, obs_seq.device
        )
        next_observations = obs_seq[:, 1:max_reward_steps + 1].contiguous()
        intrinsic_rewards_all, effective_counts = self._state_density_rewards_from_observations(
            next_observations
        )
        if intrinsic_rewards_all is None:
            return {'intrinsic_state_density/missing_current_snapshot': 1.0}, None, None
        intrinsic_rewards_all = intrinsic_rewards_all * valid_reward_mask
        valid_reward_counts, time_limit_truncated = self._infer_time_limit_truncated_from_obs_seq(
            obs_seq, valid_reward_mask
        )

        reward_target_steps = min(num_unroll_steps, max_reward_steps)
        intrinsic_reward_targets = intrinsic_rewards_all.new_zeros(
            (intrinsic_rewards_all.shape[0], num_unroll_steps)
        )
        intrinsic_reward_targets[:, :reward_target_steps] = intrinsic_rewards_all[:, :reward_target_steps]

        bootstrap_values = None
        if target_mode == 'td':
            td_steps = self._intrinsic_td_steps(max_reward_steps)
            bootstrap_values = self._make_intrinsic_bootstrap_values(
                obs_seq,
                max_reward_steps,
                td_steps,
                valid_reward_counts=valid_reward_counts,
                time_limit_truncated=time_limit_truncated,
            )
        with torch.no_grad():
            if target_mode == 'monte_carlo':
                intrinsic_values = self._make_intrinsic_monte_carlo_return_targets(
                    intrinsic_rewards_all, valid_reward_counts=valid_reward_counts
                )
            else:
                intrinsic_values = self._make_intrinsic_return_targets(
                    intrinsic_rewards_all,
                    bootstrap_values,
                    valid_reward_counts=valid_reward_counts,
                    time_limit_truncated=time_limit_truncated,
                )
            intrinsic_reward_targets, intrinsic_values, clip_log = self._clip_intrinsic_targets(
                intrinsic_reward_targets, intrinsic_values
            )

        valid_rewards = intrinsic_rewards_all[valid_reward_mask > 0.5]
        valid_effective_counts = effective_counts[valid_reward_mask > 0.5]
        valid_next_obs = next_observations[valid_reward_mask > 0.5]

        def percentile(values: torch.Tensor, quantile: float) -> float:
            return float(torch.quantile(values.float(), quantile).item()) if values.numel() else 0.0

        log = {
            'intrinsic_state_density/reward_mean': (
                float(valid_rewards.mean().item()) if valid_rewards.numel() else 0.0
            ),
            'intrinsic_state_density/reward_std': (
                float(valid_rewards.std(unbiased=False).item()) if valid_rewards.numel() else 0.0
            ),
            'intrinsic_state_density/reward_min': (
                float(valid_rewards.min().item()) if valid_rewards.numel() else 0.0
            ),
            'intrinsic_state_density/reward_max': (
                float(valid_rewards.max().item()) if valid_rewards.numel() else 0.0
            ),
            'intrinsic_state_density/reward_p50': percentile(valid_rewards, 0.50),
            'intrinsic_state_density/reward_p90': percentile(valid_rewards, 0.90),
            'intrinsic_state_density/effective_count_mean': (
                float(valid_effective_counts.mean().item()) if valid_effective_counts.numel() else 0.0
            ),
            'intrinsic_state_density/effective_count_p90': percentile(valid_effective_counts, 0.90),
            'intrinsic_state_density/value_target_mean': float(intrinsic_values.mean().item()),
            'intrinsic_state_density/value_target_std': float(
                intrinsic_values.std(unbiased=False).item()
            ),
            'intrinsic_state_density/value_target_p90': percentile(intrinsic_values, 0.90),
            'intrinsic_state_density/bootstrap_value_mean': (
                0.0 if bootstrap_values is None else float(bootstrap_values.mean().item())
            ),
            'intrinsic_state_density/bootstrap_value_std': (
                0.0 if bootstrap_values is None
                else float(bootstrap_values.std(unbiased=False).item())
            ),
            'intrinsic_state_density/valid_reward_steps_mean': float(
                valid_reward_counts.float().mean().item()
            ),
            'intrinsic_state_density/time_limit_bootstrap_fraction': float(
                time_limit_truncated.float().mean().item()
            ),
            'intrinsic_state_density/target_mode_monte_carlo': float(target_mode == 'monte_carlo'),
            'intrinsic_state_density/using_stored_collection_rewards': 0.0,
            'intrinsic_state_density/recomputed_current_snapshot': 1.0,
            'intrinsic_state_density/snapshot_phase': float(
                getattr(self, '_state_density_snapshot_phase', -1)
            ),
            'intrinsic_state_density/snapshot_transition_count': float(
                self._state_density_snapshot_counts.sum().item()
            ),
        }
        if valid_rewards.numel() and valid_next_obs.numel():
            log['intrinsic_state_density/reward_position_correlation'] = (
                self._intrinsic_diagnostic_correlation(valid_rewards, valid_next_obs[:, 0])
            )
            if valid_next_obs.shape[1] > 1:
                log['intrinsic_state_density/reward_abs_velocity_correlation'] = (
                    self._intrinsic_diagnostic_correlation(valid_rewards, valid_next_obs[:, 1].abs())
                )
            top_threshold = torch.quantile(valid_rewards.float(), 0.90)
            top_mask = valid_rewards >= top_threshold
            log['intrinsic_state_density/top10_position_mean'] = float(
                valid_next_obs[top_mask, 0].mean().item()
            )
            if valid_next_obs.shape[1] > 1:
                log['intrinsic_state_density/top10_abs_velocity_mean'] = float(
                    valid_next_obs[top_mask, 1].abs().mean().item()
                )
        log.update(clip_log)
        return log, intrinsic_reward_targets.detach(), intrinsic_values.detach()

    def _prepare_episodic_novelty_targets(
            self, data
    ) -> Tuple[dict, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not bool(self._cfg.get('episodic_novelty_diagnostics', False)):
            return {}, None, None

        current_batch = data[0]
        current_batch_core = self._current_batch_core(current_batch)
        intrinsic_reward_batch = self._selected_intrinsic_reward_batch(current_batch)
        if intrinsic_reward_batch is None:
            return {'episodic_novelty/missing_stored_reward_batch': 1.0}, None, None

        obs_seq = self._extract_observation_sequence(current_batch_core[0])
        if obs_seq is None or obs_seq.shape[1] < 2:
            return {}, None, None

        num_unroll_steps = int(self._cfg.num_unroll_steps)
        reward_array = np.asarray(intrinsic_reward_batch, dtype=np.float32)
        max_reward_steps = min(obs_seq.shape[1] - 1, int(reward_array.shape[1]))
        if max_reward_steps <= 0:
            return {}, None, None

        rewards = torch.as_tensor(
            reward_array[:, :max_reward_steps], dtype=torch.float32, device=obs_seq.device
        )
        base_mask = torch.as_tensor(
            current_batch_core[2], dtype=torch.float32, device=obs_seq.device
        )
        valid_reward_mask = base_mask[:, :max_reward_steps].clamp(min=0.0, max=1.0)
        rewards = rewards * valid_reward_mask
        reward_target_steps = min(num_unroll_steps, max_reward_steps)
        reward_targets = rewards.new_zeros((rewards.shape[0], num_unroll_steps))
        reward_targets[:, :reward_target_steps] = rewards[:, :reward_target_steps]

        valid_rewards = rewards[valid_reward_mask > 0.5]
        if valid_rewards.numel() == 0:
            reward_mean = reward_std = reward_max = nonzero_fraction = 0.0
        else:
            reward_mean = float(valid_rewards.mean().item())
            reward_std = float(valid_rewards.std(unbiased=False).item())
            reward_max = float(valid_rewards.max().item())
            nonzero_fraction = float((valid_rewards > 0.0).float().mean().item())
        log = {
            'episodic_novelty/stored_reward_mean': reward_mean,
            'episodic_novelty/stored_reward_std': reward_std,
            'episodic_novelty/stored_reward_max': reward_max,
            'episodic_novelty/stored_reward_nonzero_fraction': nonzero_fraction,
            'episodic_novelty/reward_target_mean': float(reward_targets.mean().item()),
            'episodic_novelty/reward_target_std': float(
                reward_targets.std(unbiased=False).item()
            ),
            'episodic_novelty/value_head_trained': 0.0,
        }
        return log, reward_targets.detach(), None

    def _prepare_intrinsic_targets_and_train(
            self, data
    ) -> Tuple[dict, Optional[torch.Tensor], Optional[torch.Tensor]]:
        source = self._intrinsic_reward_source()
        if source == 'episodic_novelty':
            return self._prepare_episodic_novelty_targets(data)
        if source == 'rnd':
            return self._prepare_rnd_targets_and_train(data)
        if source == 'dynamics_disagreement':
            return self._prepare_dynamics_disagreement_targets(data)
        if source == 'qstd':
            return self._prepare_qstd_targets(data)
        if source == 'state_value_disagreement':
            return self._prepare_state_value_disagreement_targets(data)
        if source == 'state_density':
            return self._prepare_state_density_targets(data)
        return {}, None, None

    def _train_rnd_from_replay_batch_legacy(self, data) -> dict:
        if not self._rnd_enabled():
            return {}
        current_batch = data[0]
        obs_batch_ori = current_batch[0]
        obs_batch, _ = prepare_obs(obs_batch_ori, self._cfg)
        ensemble_size = int(self._cfg.model.ensemble_size)
        member_ids = [
            member_id
            for member_id in range(ensemble_size)
            for _ in range(obs_batch.shape[0])
        ]
        obs_batch = torch.cat([obs_batch for _ in range(ensemble_size)], dim=0)

        was_training = self._learn_model.training
        self._learn_model.eval()
        with torch.no_grad():
            latent_state = self._learn_model.initial_inference(obs_batch, member_id=member_ids).latent_state.detach()
        if was_training:
            self._learn_model.train()
        return self._train_rnd_from_latents(latent_state)

    def default_model(self) -> Tuple[str, List[str]]:
        if self._cfg.model.model_type == "conv":
            return 'EnsembleMuZeroModel', ['lzero.model.ensemble_muzero_model']
        elif self._cfg.model.model_type == "mlp":
            return 'EnsembleMuZeroModelMLP', ['lzero.model.ensemble_muzero_model']
        else:
            raise ValueError(
                "model type {} is not supported by ensemble_muzero".format(self._cfg.model.model_type)
            )

    def _init_learn(self) -> None:
        self._validate_ensemble_skeleton()
        self._validate_intrinsic_only_training()
        self._init_rnd_state()
        super()._init_learn()
        self._ensure_episodic_sequence_novelty_model()
        self._attach_intrinsic_heads_to_main_optimizer()
        if self._rnd_landscape_query_enabled():
            self._ensure_rnd_landscape_query_model()
        self._ensure_ensemble_value_calibration_state()
        self._sync_qstd_target_snapshot(force=True)

    def _repeat_batch_for_members(self, data):
        ensemble_size = self._cfg.model.ensemble_size
        if ensemble_size == 1:
            current_batch, target_batch = data
            if len(target_batch) > 3:
                return [current_batch, target_batch[:3]], self._cfg.batch_size, [0 for _ in range(self._cfg.batch_size)]
            return data, self._cfg.batch_size, [0 for _ in range(self._cfg.batch_size)]

        current_batch, target_batch = data
        original_batch_size = self._cfg.batch_size

        def repeat_array(item):
            if hasattr(item, 'shape') and len(item.shape) > 0 and item.shape[0] == original_batch_size:
                return np.concatenate([item.copy() for _ in range(ensemble_size)], axis=0)
            return item

        expanded_current_batch = [repeat_array(item) for item in current_batch]
        expanded_target_batch = [repeat_array(item) for item in target_batch[:3]]
        if len(target_batch) > 3:
            member_target_values = np.asarray(target_batch[3])
            expected_shape = (ensemble_size, original_batch_size)
            if (
                    member_target_values.ndim >= 3
                    and member_target_values.shape[0] == ensemble_size
                    and member_target_values.shape[1] == original_batch_size
            ):
                expanded_target_batch[1] = member_target_values.reshape(
                    ensemble_size * original_batch_size, *member_target_values.shape[2:]
                )
            elif (
                    member_target_values.ndim >= 2
                    and member_target_values.shape[0] == ensemble_size * original_batch_size
            ):
                expanded_target_batch[1] = member_target_values
            else:
                raise ValueError(
                    "Invalid member-specific value target shape: "
                    f"{member_target_values.shape}, expected ({expected_shape[0]}, {expected_shape[1]}, ...)"
                )
        member_ids = [
            member_id
            for member_id in range(ensemble_size)
            for _ in range(original_batch_size)
        ]
        return [expanded_current_batch, expanded_target_batch], original_batch_size, member_ids

    def _member_bootstrap_target_diagnostics(self, data) -> dict:
        current_batch, target_batch = data
        if len(target_batch) <= 3:
            return {}
        member_value_targets = np.asarray(target_batch[3], dtype=np.float32)
        if member_value_targets.ndim < 3 or member_value_targets.shape[0] <= 1:
            return {}
        member_std = float(member_value_targets.std(axis=0).mean())
        root_member_std = float(member_value_targets[:, :, 0].std(axis=0).mean())
        return {
            'ensemble/value_target_member_std': member_std,
            'ensemble/value_target_root_member_std': root_member_std,
            'ensemble/bootstrap_target_member_std': member_std,
            'ensemble/bootstrap_target_root_member_std': root_member_std,
        }

    def _repeat_tensor_for_members(self, tensor: torch.Tensor, ensemble_size: int) -> torch.Tensor:
        if ensemble_size == 1:
            return tensor
        return torch.cat([tensor for _ in range(ensemble_size)], dim=0)

    def _align_intrinsic_targets_for_members(
            self,
            targets: torch.Tensor,
            ensemble_size: int,
            batch_size: int,
    ) -> torch.Tensor:
        if targets.shape[0] == batch_size:
            return self._repeat_tensor_for_members(targets, ensemble_size)
        if targets.shape[0] == ensemble_size * batch_size:
            return targets
        raise ValueError(
            f"Invalid intrinsic target batch size: {targets.shape[0]}, expected "
            f"{batch_size} or {ensemble_size * batch_size}."
        )

    def _intrinsic_model_parameters(self) -> List[nn.Parameter]:
        if not bool(self._cfg.get('intrinsic_head_only', True)):
            return list(self._learn_model.parameters())

        return self._intrinsic_head_parameters()

    def _intrinsic_head_parameters(self) -> List[nn.Parameter]:
        """Return only reward/value heads for their dedicated optimizer group."""

        members = getattr(self._learn_model, 'members', None)
        if members is None:
            members = [self._learn_model]
        params: List[nn.Parameter] = []
        parameter_ids = set()
        head_names = ['intrinsic_reward_head']
        if bool(self._cfg.get('intrinsic_train_value_head', True)):
            head_names.append('intrinsic_value_head')
        for member in members:
            for head_name in head_names:
                head = getattr(member, head_name, None)
                if head is not None:
                    for parameter in head.parameters():
                        if id(parameter) not in parameter_ids:
                            params.append(parameter)
                            parameter_ids.add(id(parameter))
        return params

    def _episodic_history_shadow_enabled(self, model: Optional[nn.Module] = None) -> bool:
        if (
                self._intrinsic_reward_source() != 'episodic_novelty'
                or not bool(self._cfg.get('episodic_novelty_diagnostics', False))
                or not bool(self._cfg.get('episodic_novelty_history_shadow', False))
        ):
            return False
        if model is None:
            model = getattr(self, '_learn_model', None)
        return model is not None and getattr(model, 'episodic_history_models', None) is not None

    def _ensure_episodic_history_optimizer(self):
        if self._episodic_history_optimizer is not None:
            return self._episodic_history_optimizer
        if not self._episodic_history_shadow_enabled(self._learn_model):
            return None
        parameters = self._learn_model.episodic_history_parameters()
        if not parameters:
            return None
        self._episodic_history_optimizer = torch.optim.Adam(
            parameters,
            lr=float(self._cfg.get('episodic_novelty_history_learning_rate', 3e-4)),
            weight_decay=0.0,
        )
        return self._episodic_history_optimizer

    def _set_episodic_history_training(self, training: bool) -> None:
        history_models = getattr(self._learn_model, 'episodic_history_models', None)
        if history_models is None:
            return
        for history_model in history_models:
            history_model.train(training)

    @staticmethod
    def _pad_episodic_history_episode_batch(episodes: Sequence[dict]):
        max_steps = max(int(len(episode['actions'])) for episode in episodes)
        observation_shape = tuple(np.asarray(episodes[0]['observations']).shape[1:])
        observations = np.zeros((len(episodes), max_steps + 1, *observation_shape), dtype=np.float32)
        actions = np.zeros((len(episodes), max_steps), dtype=np.int64)
        targets = np.zeros((len(episodes), max_steps), dtype=np.float32)
        mask = np.zeros((len(episodes), max_steps), dtype=np.float32)
        for batch_index, episode in enumerate(episodes):
            episode_observations = np.asarray(episode['observations'], dtype=np.float32)
            episode_actions = np.asarray(episode['actions'], dtype=np.int64).reshape(-1)
            episode_targets = np.asarray(episode['intrinsic_rewards'], dtype=np.float32).reshape(-1)
            step_count = min(
                int(episode_actions.size),
                int(episode_targets.size),
                max(0, int(episode_observations.shape[0]) - 1),
            )
            if step_count <= 0:
                continue
            observations[batch_index, :step_count + 1] = episode_observations[:step_count + 1]
            observations[batch_index, step_count + 1:] = episode_observations[step_count]
            actions[batch_index, :step_count] = episode_actions[:step_count]
            targets[batch_index, :step_count] = episode_targets[:step_count]
            mask[batch_index, :step_count] = 1.0
        return observations, actions, targets, mask

    def _ensure_episodic_history_replay_state(self) -> None:
        if not hasattr(self, '_episodic_history_episode_replay'):
            self._episodic_history_episode_replay = []
        if not hasattr(self, '_episodic_history_replay_next_id'):
            self._episodic_history_replay_next_id = 0

    @staticmethod
    def _canonical_episodic_history_episode(
            episode: dict,
            collection_phase: Optional[int] = None,
    ) -> Optional[dict]:
        observations = np.asarray(episode.get('observations', []), dtype=np.float32)
        actions = np.asarray(episode.get('actions', []), dtype=np.int64).reshape(-1)
        targets = np.asarray(episode.get('intrinsic_rewards', []), dtype=np.float32).reshape(-1)
        if observations.ndim < 2:
            return None
        step_count = min(
            int(actions.size),
            int(targets.size),
            max(0, int(observations.shape[0]) - 1),
        )
        if step_count <= 0:
            return None
        phase = episode.get('collection_phase', collection_phase)
        if phase is None:
            phase = -1
        return {
            'observations': observations[:step_count + 1].copy(),
            'actions': actions[:step_count].copy(),
            'intrinsic_rewards': targets[:step_count].copy(),
            'collection_phase': int(phase),
            'env_id': int(episode.get('env_id', -1)),
            'collection_episode_index': int(episode.get('collection_episode_index', -1)),
        }

    def _add_episodic_history_replay_episodes(
            self,
            episodes: Sequence[dict],
            collection_phase: Optional[int],
    ) -> Tuple[List[dict], int]:
        self._ensure_episodic_history_replay_state()
        fresh_episodes = []
        for episode in episodes:
            canonical = self._canonical_episodic_history_episode(episode, collection_phase)
            if canonical is None:
                continue
            canonical['replay_id'] = int(self._episodic_history_replay_next_id)
            self._episodic_history_replay_next_id += 1
            self._episodic_history_episode_replay.append(canonical)
            fresh_episodes.append(canonical)

        capacity = max(
            1,
            int(self._cfg.get('episodic_novelty_history_replay_capacity', 1000)),
        )
        evicted_count = max(0, len(self._episodic_history_episode_replay) - capacity)
        if evicted_count > 0:
            del self._episodic_history_episode_replay[:evicted_count]
        return fresh_episodes, evicted_count

    @staticmethod
    def _cyclic_episode_draw(pool: Sequence[dict], draw_count: int) -> List[dict]:
        if not pool or draw_count <= 0:
            return []
        draws = []
        while len(draws) < draw_count:
            order = np.random.permutation(len(pool))
            remaining = draw_count - len(draws)
            draws.extend(pool[int(index)] for index in order[:remaining])
        return draws

    def _sample_episodic_history_replay_batches(
            self,
            collection_phase: Optional[int],
    ) -> Tuple[List[List[dict]], dict]:
        self._ensure_episodic_history_replay_state()
        replay = self._episodic_history_episode_replay
        if not replay:
            return [], {}

        batch_size = max(
            1,
            int(self._cfg.get('episodic_novelty_history_train_batch_size', 5)),
        )
        configured_updates = self._cfg.get('episodic_novelty_history_updates_per_collect', 60)
        if configured_updates is None:
            epochs = max(1, int(self._cfg.get('episodic_novelty_history_train_epochs', 1)))
            configured_updates = epochs * int(np.ceil(len(replay) / batch_size))
        update_count = max(1, int(configured_updates))
        total_slots = update_count * batch_size

        phase_count = max(
            1,
            int(self._cfg.get('episodic_novelty_history_recent_phase_count', 2)),
        )
        replay_phases = sorted({int(episode['collection_phase']) for episode in replay})
        recent_phases = set(replay_phases[-phase_count:])
        recent_pool = [episode for episode in replay if int(episode['collection_phase']) in recent_phases]
        historical_pool = [episode for episode in replay if int(episode['collection_phase']) not in recent_phases]

        if recent_pool and historical_pool:
            recent_ratio = float(
                np.clip(
                    self._cfg.get('episodic_novelty_history_recent_sample_ratio', 0.5),
                    0.0,
                    1.0,
                )
            )
            recent_slot_count = int(round(total_slots * recent_ratio))
        elif recent_pool:
            recent_slot_count = total_slots
        else:
            recent_slot_count = 0
        historical_slot_count = total_slots - recent_slot_count

        recent_draws = self._cyclic_episode_draw(recent_pool, recent_slot_count)
        historical_draws = self._cyclic_episode_draw(historical_pool, historical_slot_count)
        recent_index = 0
        historical_index = 0
        batches = []
        sampled_episodes = []
        sampled_recent_count = 0
        for slot in range(total_slots):
            desired_recent_so_far = int(round((slot + 1) * recent_slot_count / total_slots))
            if sampled_recent_count < desired_recent_so_far:
                episode = recent_draws[recent_index]
                recent_index += 1
                sampled_recent_count += 1
            else:
                episode = historical_draws[historical_index]
                historical_index += 1
            sampled_episodes.append(episode)
            if len(sampled_episodes) % batch_size == 0:
                batch = sampled_episodes[-batch_size:]
                batch_order = np.random.permutation(batch_size)
                batches.append([batch[int(index)] for index in batch_order])

        sample_counts: Dict[int, int] = {}
        for episode in sampled_episodes:
            replay_id = int(episode['replay_id'])
            sample_counts[replay_id] = sample_counts.get(replay_id, 0) + 1
        recent_replay_ids = {int(episode['replay_id']) for episode in recent_pool}
        historical_replay_ids = {int(episode['replay_id']) for episode in historical_pool}
        recent_reuse = [sample_counts[replay_id] for replay_id in recent_replay_ids if replay_id in sample_counts]
        historical_reuse = [
            sample_counts[replay_id]
            for replay_id in historical_replay_ids
            if replay_id in sample_counts
        ]
        current_phase = -1 if collection_phase is None else int(collection_phase)
        current_phase_sample_count = sum(
            int(episode['collection_phase']) == current_phase for episode in sampled_episodes
        )
        sampled_transition_count = sum(len(episode['actions']) for episode in sampled_episodes)
        replay_transition_count = sum(len(episode['actions']) for episode in replay)
        return batches, {
            'episodic_history/replay_episode_count': float(len(replay)),
            'episodic_history/replay_transition_count': float(replay_transition_count),
            'episodic_history/recent_pool_episode_count': float(len(recent_pool)),
            'episodic_history/historical_pool_episode_count': float(len(historical_pool)),
            'episodic_history/recent_pool_phase_count': float(len(recent_phases)),
            'episodic_history/replay_oldest_phase': float(replay_phases[0]),
            'episodic_history/replay_newest_phase': float(replay_phases[-1]),
            'episodic_history/sampled_episode_count': float(len(sampled_episodes)),
            'episodic_history/sampled_unique_episode_count': float(len(sample_counts)),
            'episodic_history/sampled_transition_count': float(sampled_transition_count),
            'episodic_history/recent_sample_count': float(recent_slot_count),
            'episodic_history/historical_sample_count': float(historical_slot_count),
            'episodic_history/recent_sample_fraction': float(recent_slot_count / total_slots),
            'episodic_history/current_phase_sample_count': float(current_phase_sample_count),
            'episodic_history/current_phase_sample_fraction': float(
                current_phase_sample_count / total_slots
            ),
            'episodic_history/sample_reuse_mean': float(np.mean(list(sample_counts.values()))),
            'episodic_history/sample_reuse_max': float(max(sample_counts.values())),
            'episodic_history/recent_sample_reuse_mean': float(np.mean(recent_reuse)) if recent_reuse else 0.0,
            'episodic_history/recent_sample_reuse_min': float(min(recent_reuse)) if recent_reuse else 0.0,
            'episodic_history/recent_sample_reuse_max': float(max(recent_reuse)) if recent_reuse else 0.0,
            'episodic_history/historical_sample_reuse_mean': (
                float(np.mean(historical_reuse)) if historical_reuse else 0.0
            ),
        }

    def _episodic_history_replay_state_dict(self) -> dict:
        self._ensure_episodic_history_replay_state()
        return {
            'episodes': self._episodic_history_episode_replay,
            'next_id': int(self._episodic_history_replay_next_id),
        }

    def _load_episodic_history_replay_state(self, state: dict) -> None:
        self._episodic_history_episode_replay = []
        max_replay_id = -1
        for saved_episode in state.get('episodes', []):
            episode = self._canonical_episodic_history_episode(saved_episode)
            if episode is None:
                continue
            replay_id = int(saved_episode.get('replay_id', max_replay_id + 1))
            episode['replay_id'] = replay_id
            max_replay_id = max(max_replay_id, replay_id)
            self._episodic_history_episode_replay.append(episode)
        capacity = max(
            1,
            int(self._cfg.get('episodic_novelty_history_replay_capacity', 1000)),
        )
        self._episodic_history_episode_replay = self._episodic_history_episode_replay[-capacity:]
        self._episodic_history_replay_next_id = max(
            int(state.get('next_id', max_replay_id + 1)),
            max_replay_id + 1,
        )

    def train_episodic_history_shadow(
            self,
            episodes: Sequence[dict],
            collection_phase: Optional[int] = None,
    ) -> dict:
        """Train recurrent novelty prediction from a stratified complete-episode replay."""
        if not self._episodic_history_shadow_enabled(self._learn_model) or not episodes:
            return {}
        optimizer = self._ensure_episodic_history_optimizer()
        if optimizer is None:
            return {}

        fresh_episodes, evicted_count = self._add_episodic_history_replay_episodes(
            episodes,
            collection_phase,
        )
        if not fresh_episodes:
            return {}
        episode_batches, replay_stats = self._sample_episodic_history_replay_batches(collection_phase)
        if not episode_batches:
            return {}
        ensemble_size = int(self._cfg.model.ensemble_size)
        parameters = self._learn_model.episodic_history_parameters()
        prediction_parts = []
        target_parts = []
        hidden_norm_parts = []
        loss_values = []
        grad_norm_values = []
        update_count = 0
        was_training = self._learn_model.training

        try:
            self._learn_model.eval()
            self._set_episodic_history_training(True)
            for batch_episodes in episode_batches:
                    observations_np, actions_np, targets_np, mask_np = (
                        self._pad_episodic_history_episode_batch(batch_episodes)
                    )
                    observations = torch.as_tensor(
                        observations_np, dtype=torch.float32, device=self._cfg.device
                    )
                    actions = torch.as_tensor(actions_np, dtype=torch.long, device=self._cfg.device)
                    targets = torch.as_tensor(targets_np, dtype=torch.float32, device=self._cfg.device)
                    mask = torch.as_tensor(mask_np, dtype=torch.float32, device=self._cfg.device)
                    episode_batch_size, state_count = observations.shape[:2]
                    flattened_observations = observations.reshape(
                        episode_batch_size * state_count, *observations.shape[2:]
                    )

                    optimizer.zero_grad()
                    loss_sum = torch.zeros((), dtype=torch.float32, device=self._cfg.device)
                    weight_sum = torch.zeros((), dtype=torch.float32, device=self._cfg.device)
                    for member_id in range(ensemble_size):
                        with torch.no_grad():
                            encoded = self._learn_model.initial_inference(
                                flattened_observations, member_id=member_id
                            ).latent_state
                        encoded = encoded.reshape(
                            episode_batch_size, state_count, *encoded.shape[1:]
                        )
                        transition_count = state_count - 1
                        flattened_current_latents = encoded[:, :-1].reshape(
                            episode_batch_size * transition_count, *encoded.shape[2:]
                        )
                        flattened_actions = actions.reshape(-1)
                        with torch.no_grad():
                            imagined_next_latents = self._learn_model.recurrent_inference(
                                flattened_current_latents,
                                flattened_actions.unsqueeze(-1),
                                member_id=member_id,
                            ).latent_state
                        imagined_next_latents = imagined_next_latents.reshape(
                            episode_batch_size,
                            transition_count,
                            *imagined_next_latents.shape[1:],
                        )
                        history = self._learn_model.initial_episodic_history(
                            episode_batch_size,
                            member_id=member_id,
                            device=encoded.device,
                            dtype=encoded.dtype,
                        )
                        histories, reward_predictions = self._learn_model.episodic_history_sequence(
                            encoded[:, :-1],
                            actions,
                            imagined_next_latents,
                            encoded[:, 1:],
                            targets,
                            history,
                            member_id=member_id,
                        )
                        reward_predictions = reward_predictions.squeeze(-1)
                        elementwise_loss = self._intrinsic_elementwise_loss(
                            reward_predictions, targets
                        )
                        loss_sum = loss_sum + (elementwise_loss * mask).sum()
                        weight_sum = weight_sum + mask.sum()
                        valid = mask > 0.5
                        if valid.any().item():
                            prediction_parts.append(reward_predictions.detach()[valid])
                            target_parts.append(targets.detach()[valid])
                            hidden_norm_parts.append(histories.detach()[valid].norm(dim=-1))

                    loss = loss_sum / weight_sum.clamp_min(1.0)
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        parameters,
                        float(self._cfg.get('episodic_novelty_history_grad_clip', 5.0)),
                    )
                    optimizer.step()
                    loss_values.append(float(loss.detach().item()))
                    grad_norm_values.append(float(grad_norm.detach().item()))
                    update_count += 1
        finally:
            self._learn_model.set_active_member(0)
            self._learn_model.train(was_training)

        predictions = torch.cat(prediction_parts) if prediction_parts else torch.empty(0)
        diagnostic_targets = torch.cat(target_parts) if target_parts else torch.empty(0)
        hidden_norms = torch.cat(hidden_norm_parts) if hidden_norm_parts else torch.empty(0)
        if predictions.numel() > 1:
            prediction_np = predictions.float().cpu().numpy()
            target_np = diagnostic_targets.float().cpu().numpy()
            if prediction_np.std() > 1e-8 and target_np.std() > 1e-8:
                correlation = float(np.corrcoef(prediction_np, target_np)[0, 1])
            else:
                correlation = 0.0
        else:
            correlation = 0.0
        stats = {
            'episodic_history/loss': float(np.mean(loss_values)) if loss_values else 0.0,
            'episodic_history/prediction_mean': (
                float(predictions.mean().item()) if predictions.numel() > 0 else 0.0
            ),
            'episodic_history/target_mean': (
                float(diagnostic_targets.mean().item()) if diagnostic_targets.numel() > 0 else 0.0
            ),
            'episodic_history/prediction_mae': (
                float((predictions - diagnostic_targets).abs().mean().item())
                if predictions.numel() > 0 else 0.0
            ),
            'episodic_history/prediction_correlation': correlation,
            'episodic_history/hidden_norm_mean': (
                float(hidden_norms.mean().item()) if hidden_norms.numel() > 0 else 0.0
            ),
            'episodic_history/grad_norm_before_clip': (
                float(np.mean(grad_norm_values)) if grad_norm_values else 0.0
            ),
            'episodic_history/episode_count': float(len(fresh_episodes)),
            'episodic_history/transition_count': float(
                sum(len(episode['actions']) for episode in fresh_episodes)
            ),
            'episodic_history/fresh_episode_count': float(len(fresh_episodes)),
            'episodic_history/fresh_transition_count': float(
                sum(len(episode['actions']) for episode in fresh_episodes)
            ),
            'episodic_history/evicted_episode_count': float(evicted_count),
            'episodic_history/prediction_count': float(predictions.numel()),
            'episodic_history/update_count': float(update_count),
            'episodic_history/teacher_forced_reward_feedback': 1.0,
            'episodic_history/teacher_forced_real_next_latent': 1.0,
            'episodic_history/reward_feedback_log1p': 1.0,
            'episodic_history/replay_target_recomputation': 0.0,
        }
        stats.update(replay_stats)
        return stats

    @staticmethod
    def _canonical_episodic_sequence_episode(
            episode: dict,
            collection_phase: Optional[int] = None,
    ) -> Optional[dict]:
        observations = np.asarray(episode.get('observations', []))
        actions = np.asarray(episode.get('actions', []), dtype=np.int64).reshape(-1)
        episodic_rewards = np.asarray(
            episode.get('episodic_rewards', []), dtype=np.float32
        ).reshape(-1)
        extrinsic_rewards = np.asarray(
            episode.get('extrinsic_rewards', np.zeros_like(episodic_rewards)),
            dtype=np.float32,
        ).reshape(-1)
        rnd_rewards = np.asarray(
            episode.get('rnd_rewards', np.zeros_like(episodic_rewards)),
            dtype=np.float32,
        ).reshape(-1)
        if observations.ndim != 4:
            return None
        step_count = min(
            int(actions.size),
            int(episodic_rewards.size),
            int(extrinsic_rewards.size),
            max(0, int(observations.shape[0]) - 1),
        )
        if step_count <= 0:
            return None
        observations = observations[:step_count + 1]
        if observations.dtype != np.uint8:
            scale = 255.0 if float(np.max(observations, initial=0.0)) <= 1.5 else 1.0
            observations = np.rint(
                np.clip(observations * scale, 0.0, 255.0)
            ).astype(np.uint8)
        phase = episode.get('collection_phase', collection_phase)
        return {
            'observations': observations.copy(),
            'actions': actions[:step_count].copy(),
            'episodic_rewards': episodic_rewards[:step_count].copy(),
            'extrinsic_rewards': extrinsic_rewards[:step_count].copy(),
            'rnd_rewards': rnd_rewards[:step_count].copy(),
            'collection_phase': -1 if phase is None else int(phase),
            'env_id': int(episode.get('env_id', -1)),
            'collection_episode_index': int(
                episode.get('collection_episode_index', -1)
            ),
        }

    def _add_episodic_sequence_replay_episodes(
            self,
            episodes: Sequence[dict],
            collection_phase: Optional[int],
    ) -> List[dict]:
        fresh = []
        for episode in episodes:
            canonical = self._canonical_episodic_sequence_episode(
                episode, collection_phase
            )
            if canonical is not None:
                self._episodic_sequence_novelty_replay.append(canonical)
                fresh.append(canonical)
        capacity = max(
            1,
            int(self._cfg.get('episodic_sequence_novelty_replay_capacity', 32)),
        )
        if len(self._episodic_sequence_novelty_replay) > capacity:
            del self._episodic_sequence_novelty_replay[:-capacity]
        return fresh

    def _refresh_episodic_sequence_rnd_rewards(
            self, episodes: Sequence[dict]
    ) -> int:
        """Recompute sidecar targets with the current frozen RND target snapshot."""
        if not episodes:
            return 0
        observation_parts = [
            np.asarray(episode['observations'][1:], dtype=np.float32) / 255.0
            for episode in episodes
        ]
        if not observation_parts or sum(len(part) for part in observation_parts) <= 0:
            return 0
        observations = np.concatenate(observation_parts, axis=0)
        obs_tensor = torch.as_tensor(
            observations, dtype=torch.float32, device=self._cfg.device
        )
        rnd_inputs = self._select_rnd_observation_features(obs_tensor).contiguous()
        self._ensure_rnd_model(rnd_inputs[:1])
        rnd_target_model = self._rnd_target_model_for_targets()
        if rnd_target_model is None:
            return 0
        inference_batch_size = max(
            1, int(self._cfg.get('rnd_target_inference_batch_size', 1024))
        )
        was_training = rnd_target_model.training
        reward_parts = []
        try:
            rnd_target_model.eval()
            with torch.no_grad():
                for start in range(0, int(rnd_inputs.shape[0]), inference_batch_size):
                    raw_errors = rnd_target_model.error(
                        rnd_inputs[start:start + inference_batch_size]
                    ).reshape(-1)
                    reward_parts.append(
                        self._normalize_rnd_errors(raw_errors).detach().float().cpu()
                    )
        finally:
            rnd_target_model.train(was_training)
        rewards = torch.cat(reward_parts).numpy().astype(np.float32, copy=False)
        cursor = 0
        for episode in episodes:
            step_count = len(episode['actions'])
            episode['rnd_rewards'] = rewards[cursor:cursor + step_count].copy()
            cursor += step_count
        return int(cursor)

    @staticmethod
    def _pad_episodic_sequence_batch(episodes: Sequence[dict]):
        max_steps = max(len(episode['actions']) for episode in episodes)
        observation_shape = tuple(episodes[0]['observations'].shape[1:])
        observations = np.zeros(
            (len(episodes), max_steps + 1, *observation_shape), dtype=np.uint8
        )
        actions = np.zeros((len(episodes), max_steps), dtype=np.int64)
        episodic_rewards = np.zeros((len(episodes), max_steps), dtype=np.float32)
        extrinsic_rewards = np.zeros((len(episodes), max_steps), dtype=np.float32)
        rnd_rewards = np.zeros((len(episodes), max_steps), dtype=np.float32)
        lengths = np.zeros(len(episodes), dtype=np.int64)
        for batch_index, episode in enumerate(episodes):
            step_count = len(episode['actions'])
            lengths[batch_index] = step_count
            observations[batch_index, :step_count + 1] = episode['observations']
            observations[batch_index, step_count + 1:] = episode['observations'][-1]
            actions[batch_index, :step_count] = episode['actions']
            episodic_rewards[batch_index, :step_count] = episode['episodic_rewards']
            extrinsic_rewards[batch_index, :step_count] = episode['extrinsic_rewards']
            rnd_rewards[batch_index, :step_count] = episode['rnd_rewards']
        return (
            observations,
            actions,
            episodic_rewards,
            extrinsic_rewards,
            rnd_rewards,
            lengths,
        )

    def train_episodic_sequence_novelty(
            self,
            episodes: Sequence[dict],
            collection_phase: Optional[int] = None,
    ) -> dict:
        """Train only on action suffixes that occurred in complete real episodes."""
        model = self._ensure_episodic_sequence_novelty_model()
        if model is None or not episodes:
            return {}
        fresh = self._add_episodic_sequence_replay_episodes(
            episodes, collection_phase
        )
        if not fresh:
            return {}
        replay = self._episodic_sequence_novelty_replay
        continuation_enabled = self._episodic_sequence_continuation_enabled()
        refreshed_rnd_count = 0
        if continuation_enabled:
            refreshed_rnd_count = self._refresh_episodic_sequence_rnd_rewards(replay)
            if refreshed_rnd_count <= 0:
                raise RuntimeError(
                    'Episodic continuation-value training requires current RND rewards.'
                )
        horizon = max(
            1,
            int(self._cfg.get('episodic_sequence_novelty_horizon', 5)),
        )
        eligible = [episode for episode in replay if len(episode['actions']) >= horizon]
        if not eligible:
            return {}
        optimizer = self._episodic_sequence_novelty_optimizer
        if optimizer is None:
            raise RuntimeError('Missing episodic sequence novelty optimizer.')

        update_count = max(
            1,
            int(self._cfg.get('episodic_sequence_novelty_updates_per_collect', 64)),
        )
        predictive_memory = isinstance(model, PredictiveEpisodicMemory)
        reconstructive_memory = isinstance(model, ReconstructiveEpisodicMemory)
        recent_transformer = isinstance(model, RecentHistoryTransformerNovelty)
        joint_update_count = update_count
        if predictive_memory:
            configured_joint_updates = self._cfg.get(
                'episodic_predictive_memory_joint_updates_per_collect', None
            )
            if configured_joint_updates is not None:
                joint_update_count = int(
                    np.clip(int(configured_joint_updates), 0, update_count)
                )
        predictor_only_update_count = (
            update_count - joint_update_count if predictive_memory else 0
        )
        batch_size = max(
            1,
            int(self._cfg.get('episodic_sequence_novelty_train_batch_size', 1)),
        )
        roots_per_update = max(
            1,
            int(self._cfg.get('episodic_sequence_novelty_roots_per_update', 32)),
        )
        gamma = float(
            self._cfg.get('episodic_sequence_novelty_discount_factor', 0.99)
        )
        sequence_loss_weight = float(
            self._cfg.get(
                'episodic_sequence_novelty_sequence_loss_weight', 1.0
            )
        )
        continuation_value_horizon = max(
            1,
            int(
                self._cfg.get(
                    'episodic_sequence_continuation_value_horizon', 20
                )
            ),
        )
        continuation_loss_weight = float(
            self._cfg.get('episodic_sequence_continuation_loss_weight', 1.0)
        )
        minimum_factor = float(
            np.clip(
                self._cfg.get(
                    'episodic_sequence_novelty_minimum_reward_factor', 0.05
                ),
                0.0,
                1.0,
            )
        )
        if continuation_enabled and minimum_factor >= 1.0:
            raise ValueError(
                'episodic_sequence_novelty_minimum_reward_factor must be below 1 '
                'when continuation-value training is enabled.'
            )
        grad_clip = float(
            self._cfg.get('episodic_sequence_novelty_grad_clip', 5.0)
        )
        weights = torch.pow(
            torch.tensor(gamma, dtype=torch.float32, device=self._cfg.device),
            torch.arange(horizon, dtype=torch.float32, device=self._cfg.device),
        )
        weight_sum = weights.sum().clamp_min(1e-8)
        loss_values = []
        edge_loss_values = []
        sequence_loss_values = []
        brier_values = []
        prediction_parts = []
        target_parts = []
        depth_brier_parts = [[] for _ in range(horizon)]
        grad_norm_values = []
        continuation_loss_values = []
        continuation_prediction_parts = []
        continuation_target_parts = []
        continuation_raw_return_parts = []
        continuation_gated_return_parts = []
        continuation_depth_mae_parts = [[] for _ in range(horizon)]
        predictive_loss_values = []
        predictive_distance_loss_values = []
        inverse_loss_values = []
        inverse_accuracy_values = []
        predictive_depth_cosine_parts = [[] for _ in range(horizon)]
        predictive_novel_distance_parts = []
        predictive_revisit_distance_parts = []
        predictive_target_novel_distance_parts = []
        predictive_target_revisit_distance_parts = []
        predictive_distance_scale_parts = []
        predictive_distance_threshold_parts = []
        predictive_distance_temperature_parts = []
        predictor_only_predictive_loss_values = []
        predictor_only_distance_loss_values = []
        predictor_only_depth_cosine_parts = [[] for _ in range(horizon)]
        predictive_loss_last_value = 0.0
        predictive_distance_loss_last_value = 0.0
        predictive_depth_cosine_last = [0.0 for _ in range(horizon)]
        reconstruction_loss_values = []
        rollout_reconstruction_loss_values = []
        reconstruction_psnr_values = []
        rollout_reconstruction_psnr_values = []
        rollout_depth_mae_parts = [[] for _ in range(horizon)]
        reconstruction_loss_last_value = 0.0
        rollout_reconstruction_loss_last_value = 0.0
        edge_loss_last_value = 0.0
        brier_last_value = 0.0
        model.train()
        for update_index in range(update_count):
            predictor_only_update = (
                predictive_memory and update_index >= joint_update_count
            )
            sampled = [
                eligible[int(index)]
                for index in np.random.randint(0, len(eligible), size=batch_size)
            ]
            (
                observations_np,
                actions_np,
                episodic_rewards_np,
                extrinsic_rewards_np,
                rnd_rewards_np,
                lengths,
            ) = self._pad_episodic_sequence_batch(sampled)
            root_indices_np = np.stack(
                [
                    np.random.randint(
                        0,
                        int(length) - horizon + 1,
                        size=roots_per_update,
                    )
                    for length in lengths
                ],
                axis=0,
            ).astype(np.int64)
            recent_targets_np = None
            if recent_transformer:
                recent_targets_np = recent_observation_novelty_targets(
                    observations_np,
                    root_indices_np,
                    horizon=horizon,
                    history_window=model.history_window,
                )
            observations = torch.as_tensor(
                observations_np,
                dtype=torch.float32,
                device=self._cfg.device,
            ).div_(255.0)
            actions = torch.as_tensor(
                actions_np, dtype=torch.long, device=self._cfg.device
            )
            episodic_rewards = torch.as_tensor(
                episodic_rewards_np, dtype=torch.float32, device=self._cfg.device
            )
            extrinsic_rewards = torch.as_tensor(
                extrinsic_rewards_np, dtype=torch.float32, device=self._cfg.device
            )
            rnd_rewards = torch.as_tensor(
                rnd_rewards_np, dtype=torch.float32, device=self._cfg.device
            )
            root_indices = torch.as_tensor(
                root_indices_np, dtype=torch.long, device=self._cfg.device
            )
            predictive_outputs = None
            if isinstance(model, PredictiveEpisodicMemory):
                if continuation_enabled:
                    raise RuntimeError(
                        'Predictive episodic memory cannot train the recurrent '
                        'continuation-value head.'
                    )
                predictive_outputs = model.training_outputs(
                    observations,
                    actions,
                    episodic_rewards,
                    extrinsic_rewards,
                    root_indices,
                    horizon,
                    freeze_online_encoder=predictor_only_update,
                )
                logits = predictive_outputs['novelty_logits']
            elif continuation_enabled:
                logits, continuation_logits = (
                    model.score_actual_suffixes_with_continuation(
                        observations,
                        actions,
                        episodic_rewards,
                        extrinsic_rewards,
                        root_indices,
                        horizon,
                    )
                )
            else:
                logits = model.score_actual_suffixes(
                    observations,
                    actions,
                    episodic_rewards,
                    extrinsic_rewards,
                    root_indices,
                    horizon,
                )
            batch_indices = torch.arange(
                batch_size, device=self._cfg.device
            )[:, None]
            if recent_targets_np is not None:
                exact_targets = torch.as_tensor(
                    recent_targets_np,
                    dtype=torch.float32,
                    device=self._cfg.device,
                )
            else:
                exact_targets = torch.stack(
                    [
                        episodic_rewards[
                            batch_indices, root_indices + depth
                        ]
                        for depth in range(horizon)
                    ],
                    dim=-1,
                )
            targets = (
                predictive_outputs['target_novelty_probabilities']
                if predictive_outputs is not None else exact_targets
            )
            edge_loss = F.binary_cross_entropy_with_logits(logits, targets)
            probabilities = torch.sigmoid(logits)
            predicted_return = torch.sum(probabilities * weights, dim=-1) / weight_sum
            target_return = torch.sum(targets * weights, dim=-1) / weight_sum
            sequence_loss = F.smooth_l1_loss(predicted_return, target_return)
            loss = edge_loss + sequence_loss_weight * sequence_loss

            predictive_loss = None
            predictive_distance_loss = None
            inverse_loss = None
            reconstruction_loss = None
            rollout_reconstruction_loss = None
            if predictive_outputs is not None:
                predicted_embeddings = predictive_outputs['predicted_embeddings']
                target_embeddings = predictive_outputs['target_embeddings']
                predictive_loss = torch.mean(
                    1.0 - F.cosine_similarity(
                        predicted_embeddings,
                        target_embeddings,
                        dim=-1,
                    )
                )
                inverse_logits = predictive_outputs['inverse_logits']
                transition_mask = (
                    torch.arange(
                        inverse_logits.shape[1], device=self._cfg.device
                    )[None, :]
                    < torch.as_tensor(
                        lengths, dtype=torch.long, device=self._cfg.device
                    )[:, None]
                )
                inverse_loss = F.cross_entropy(
                    inverse_logits[transition_mask],
                    actions[transition_mask],
                )
                predictive_distance_loss = balanced_memory_distance_loss(
                    predictive_outputs['predicted_distances'],
                    predictive_outputs['target_distances'],
                    torch.maximum(
                        predictive_outputs['distance_scales'],
                        model.distance_threshold.detach(),
                    )[..., None],
                )
                loss = loss + float(
                    self._cfg.get(
                        'episodic_predictive_memory_prediction_loss_weight', 1.0
                    )
                ) * predictive_loss
                loss = loss + float(
                    self._cfg.get(
                        'episodic_predictive_memory_distance_loss_weight', 1.0
                    )
                ) * predictive_distance_loss
                if not predictor_only_update:
                    loss = loss + float(
                        self._cfg.get(
                            'episodic_predictive_memory_inverse_loss_weight', 1.0
                        )
                    ) * inverse_loss
                if reconstructive_memory:
                    reconstruction_loss = F.l1_loss(
                        predictive_outputs['reconstructed_observations'],
                        predictive_outputs['reconstruction_targets'],
                    )
                    rollout_reconstruction_loss = F.l1_loss(
                        predictive_outputs['predicted_future_observations'],
                        predictive_outputs['future_observation_targets'],
                    )
                    loss = loss + float(
                        self._cfg.get(
                            'episodic_reconstructive_memory_reconstruction_loss_weight',
                            1.0,
                        )
                    ) * reconstruction_loss
                    loss = loss + float(
                        self._cfg.get(
                            'episodic_reconstructive_memory_rollout_loss_weight',
                            1.0,
                        )
                    ) * rollout_reconstruction_loss

            if continuation_enabled:
                (
                    raw_continuation_return,
                    gated_continuation_return,
                    continuation_factor_targets,
                    valid_continuation,
                ) = episodic_continuation_factor_targets(
                    rnd_rewards=rnd_rewards,
                    episodic_rewards=episodic_rewards,
                    lengths=torch.as_tensor(
                        lengths, dtype=torch.long, device=self._cfg.device
                    ),
                    root_indices=root_indices,
                    prefix_horizon=horizon,
                    value_horizon=continuation_value_horizon,
                    discount_factor=gamma,
                    minimum_factor=minimum_factor,
                )
                normalized_continuation_targets = (
                    continuation_factor_targets - minimum_factor
                ) / (1.0 - minimum_factor)
                continuation_probabilities = torch.sigmoid(continuation_logits)
                elementwise_continuation_loss = F.smooth_l1_loss(
                    continuation_probabilities,
                    normalized_continuation_targets,
                    reduction='none',
                )
                valid_float = valid_continuation.float()
                valid_raw_mean = (
                    torch.sum(raw_continuation_return * valid_float)
                    / valid_float.sum().clamp_min(1.0)
                ).clamp_min(1e-8)
                importance = (
                    raw_continuation_return.detach() / valid_raw_mean
                ).clamp(max=10.0) * valid_float
                continuation_loss = torch.sum(
                    elementwise_continuation_loss * importance
                ) / importance.sum().clamp_min(1e-8)
                loss = loss + continuation_loss_weight * continuation_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if predictive_memory and not predictor_only_update:
                model.update_target_encoder(
                    float(
                        self._cfg.get(
                            'episodic_predictive_memory_target_ema', 0.99
                        )
                    )
                )

            detached_probabilities = probabilities.detach()
            detached_targets = targets.detach()
            loss_values.append(float(loss.detach().item()))
            edge_loss_values.append(float(edge_loss.detach().item()))
            edge_loss_last_value = float(edge_loss.detach().item())
            sequence_loss_values.append(float(sequence_loss.detach().item()))
            brier_last_value = float(
                torch.mean((detached_probabilities - detached_targets) ** 2).item()
            )
            brier_values.append(brier_last_value)
            prediction_parts.append(detached_probabilities.reshape(-1).cpu())
            target_parts.append(detached_targets.reshape(-1).cpu())
            for depth in range(horizon):
                depth_brier_parts[depth].append(
                    float(
                        torch.mean(
                            (
                                detached_probabilities[..., depth]
                                - detached_targets[..., depth]
                            ) ** 2
                        ).item()
                    )
                )
            grad_norm_values.append(float(grad_norm.detach().item()))
            if predictive_outputs is not None:
                predictive_loss_values.append(float(predictive_loss.detach().item()))
                predictive_loss_last_value = float(predictive_loss.detach().item())
                predictive_distance_loss_values.append(
                    float(predictive_distance_loss.detach().item())
                )
                predictive_distance_loss_last_value = float(
                    predictive_distance_loss.detach().item()
                )
                if predictor_only_update:
                    predictor_only_predictive_loss_values.append(
                        predictive_loss_last_value
                    )
                    predictor_only_distance_loss_values.append(
                        predictive_distance_loss_last_value
                    )
                inverse_loss_values.append(float(inverse_loss.detach().item()))
                detached_inverse = predictive_outputs['inverse_logits'].detach()
                inverse_accuracy_values.append(
                    float(
                        (
                            detached_inverse[transition_mask].argmax(dim=-1)
                            == actions[transition_mask]
                        ).float().mean().item()
                    )
                )
                detached_predicted_embeddings = (
                    predictive_outputs['predicted_embeddings'].detach()
                )
                detached_target_embeddings = (
                    predictive_outputs['target_embeddings'].detach()
                )
                for depth in range(horizon):
                    depth_cosine = float(
                        F.cosine_similarity(
                            detached_predicted_embeddings[..., depth, :],
                            detached_target_embeddings[..., depth, :],
                            dim=-1,
                        ).mean().item()
                    )
                    predictive_depth_cosine_parts[depth].append(depth_cosine)
                    predictive_depth_cosine_last[depth] = depth_cosine
                    if predictor_only_update:
                        predictor_only_depth_cosine_parts[depth].append(
                            depth_cosine
                        )
                detached_predicted_distances = predictive_outputs[
                    'predicted_distances'
                ].detach()
                detached_target_distances = predictive_outputs[
                    'target_distances'
                ].detach()
                predictive_distance_scale_parts.append(
                    predictive_outputs['distance_scales'].detach().reshape(-1).cpu()
                )
                predictive_distance_threshold_parts.append(
                    predictive_outputs['distance_thresholds'].detach().reshape(-1).cpu()
                )
                predictive_distance_temperature_parts.append(
                    predictive_outputs['distance_temperatures'].detach().reshape(-1).cpu()
                )
                novel_mask = exact_targets.detach() > 0.5
                revisit_mask = ~novel_mask
                if bool(novel_mask.any()):
                    predictive_novel_distance_parts.append(
                        detached_predicted_distances[novel_mask].cpu()
                    )
                    predictive_target_novel_distance_parts.append(
                        detached_target_distances[novel_mask].cpu()
                    )
                if bool(revisit_mask.any()):
                    predictive_revisit_distance_parts.append(
                        detached_predicted_distances[revisit_mask].cpu()
                    )
                    predictive_target_revisit_distance_parts.append(
                        detached_target_distances[revisit_mask].cpu()
                    )
                if reconstructive_memory:
                    detached_reconstruction = predictive_outputs[
                        'reconstructed_observations'
                    ].detach()
                    detached_reconstruction_target = predictive_outputs[
                        'reconstruction_targets'
                    ].detach()
                    detached_rollout = predictive_outputs[
                        'predicted_future_observations'
                    ].detach()
                    detached_rollout_target = predictive_outputs[
                        'future_observation_targets'
                    ].detach()
                    reconstruction_loss_last_value = float(
                        reconstruction_loss.detach().item()
                    )
                    rollout_reconstruction_loss_last_value = float(
                        rollout_reconstruction_loss.detach().item()
                    )
                    reconstruction_loss_values.append(
                        reconstruction_loss_last_value
                    )
                    rollout_reconstruction_loss_values.append(
                        rollout_reconstruction_loss_last_value
                    )
                    reconstruction_mse = F.mse_loss(
                        detached_reconstruction,
                        detached_reconstruction_target,
                    ).clamp_min(1e-12)
                    rollout_reconstruction_mse = F.mse_loss(
                        detached_rollout,
                        detached_rollout_target,
                    ).clamp_min(1e-12)
                    reconstruction_psnr_values.append(
                        float((-10.0 * torch.log10(reconstruction_mse)).item())
                    )
                    rollout_reconstruction_psnr_values.append(
                        float(
                            (-10.0 * torch.log10(rollout_reconstruction_mse)).item()
                        )
                    )
                    for depth in range(horizon):
                        rollout_depth_mae_parts[depth].append(
                            float(
                                F.l1_loss(
                                    detached_rollout[..., depth, :, :, :],
                                    detached_rollout_target[..., depth, :, :, :],
                                ).item()
                            )
                        )
            if continuation_enabled:
                detached_continuation_factors = (
                    minimum_factor
                    + (1.0 - minimum_factor)
                    * continuation_probabilities.detach()
                )
                detached_valid = valid_continuation.detach()
                continuation_loss_values.append(
                    float(continuation_loss.detach().item())
                )
                if bool(detached_valid.any()):
                    continuation_prediction_parts.append(
                        detached_continuation_factors[detached_valid].cpu()
                    )
                    continuation_target_parts.append(
                        continuation_factor_targets.detach()[detached_valid].cpu()
                    )
                    continuation_raw_return_parts.append(
                        raw_continuation_return.detach()[detached_valid].cpu()
                    )
                    continuation_gated_return_parts.append(
                        gated_continuation_return.detach()[detached_valid].cpu()
                    )
                for depth in range(horizon):
                    depth_valid = detached_valid[..., depth]
                    if bool(depth_valid.any()):
                        continuation_depth_mae_parts[depth].append(
                            float(
                                torch.mean(
                                    torch.abs(
                                        detached_continuation_factors[..., depth][depth_valid]
                                        - continuation_factor_targets.detach()[..., depth][depth_valid]
                                    )
                                ).item()
                            )
                        )

        predictions = torch.cat(prediction_parts)
        targets = torch.cat(target_parts)
        if predictions.numel() > 1 and predictions.std() > 1e-8 and targets.std() > 1e-8:
            correlation = float(
                np.corrcoef(predictions.numpy(), targets.numpy())[0, 1]
            )
        else:
            correlation = 0.0
        trained_phase = (
            int(collection_phase)
            if collection_phase is not None
            else int(self._episodic_sequence_novelty_trained_phase) + 1
        )
        self._episodic_sequence_novelty_trained_phase = max(
            int(self._episodic_sequence_novelty_trained_phase), trained_phase
        )
        if continuation_enabled and continuation_prediction_parts:
            self._episodic_sequence_continuation_trained_phase = max(
                int(self._episodic_sequence_continuation_trained_phase),
                trained_phase,
            )
        stats = {
            'episodic_sequence/loss': float(np.mean(loss_values)),
            'episodic_sequence/edge_bce': float(np.mean(edge_loss_values)),
            'episodic_sequence/edge_bce_last_update': edge_loss_last_value,
            'episodic_sequence/sequence_loss': float(np.mean(sequence_loss_values)),
            'episodic_sequence/edge_brier': float(np.mean(brier_values)),
            'episodic_sequence/edge_brier_last_update': brier_last_value,
            'episodic_sequence/prediction_mean': float(predictions.mean().item()),
            'episodic_sequence/target_mean': float(targets.mean().item()),
            'episodic_sequence/prediction_correlation': correlation,
            'episodic_sequence/grad_norm_before_clip': float(
                np.mean(grad_norm_values)
            ),
            'episodic_sequence/fresh_episode_count': float(len(fresh)),
            'episodic_sequence/replay_episode_count': float(len(replay)),
            'episodic_sequence/replay_transition_count': float(
                sum(len(episode['actions']) for episode in replay)
            ),
            'episodic_sequence/update_count': float(update_count),
            'episodic_sequence/joint_update_count': float(joint_update_count),
            'episodic_sequence/predictor_only_update_count': float(
                predictor_only_update_count
            ),
            'episodic_sequence/trained_phase': float(
                self._episodic_sequence_novelty_trained_phase
            ),
            'episodic_sequence/tree_active_next_phase': float(
                self._episodic_sequence_novelty_tree_active()
            ),
            'episodic_sequence/uses_imagined_latents': 0.0,
            'episodic_sequence/counterfactual_training_fraction': 0.0,
            'episodic_sequence/uses_explicit_memory_input': float(
                isinstance(model, PredictiveEpisodicMemory)
            ),
            'episodic_sequence/uses_exact_first_visit_targets': float(
                not isinstance(model, PredictiveEpisodicMemory)
                and not recent_transformer
            ),
            'episodic_sequence/uses_exact_recent_window_targets': float(
                recent_transformer
            ),
            'episodic_sequence/recent_history_transformer': float(
                recent_transformer
            ),
            'episodic_sequence/target_history_window': float(
                model.history_window if recent_transformer else 0
            ),
            'episodic_sequence/continuation_value_enabled': float(
                continuation_enabled
            ),
            'episodic_sequence/continuation_rnd_target_count': float(
                refreshed_rnd_count
            ),
            'episodic_sequence/continuation_trained_phase': float(
                self._episodic_sequence_continuation_trained_phase
            ),
            'episodic_sequence/continuation_tree_active_next_phase': float(
                self._episodic_sequence_continuation_tree_active()
            ),
        }
        if isinstance(model, PredictiveEpisodicMemory):
            stats.update(
                {
                    'episodic_predictive_memory/prediction_loss': float(
                        np.mean(predictive_loss_values)
                    ),
                    'episodic_predictive_memory/predictor_only_prediction_loss': float(
                        np.mean(predictor_only_predictive_loss_values)
                        if predictor_only_predictive_loss_values else 0.0
                    ),
                    'episodic_predictive_memory/prediction_loss_last_update': (
                        predictive_loss_last_value
                    ),
                    'episodic_predictive_memory/distance_loss': float(
                        np.mean(predictive_distance_loss_values)
                    ),
                    'episodic_predictive_memory/predictor_only_distance_loss': float(
                        np.mean(predictor_only_distance_loss_values)
                        if predictor_only_distance_loss_values else 0.0
                    ),
                    'episodic_predictive_memory/distance_loss_last_update': (
                        predictive_distance_loss_last_value
                    ),
                    'episodic_predictive_memory/inverse_loss': float(
                        np.mean(inverse_loss_values)
                    ),
                    'episodic_predictive_memory/inverse_accuracy': float(
                        np.mean(inverse_accuracy_values)
                    ),
                    'episodic_predictive_memory/adaptive_distance_scale': float(
                        model.adaptive_distance_scale
                    ),
                    'episodic_predictive_memory/distance_scale': float(
                        torch.cat(predictive_distance_scale_parts).mean().item()
                    ),
                    'episodic_predictive_memory/distance_threshold': float(
                        torch.cat(predictive_distance_threshold_parts).mean().item()
                    ),
                    'episodic_predictive_memory/distance_temperature': float(
                        torch.cat(predictive_distance_temperature_parts).mean().item()
                    ),
                    'episodic_predictive_memory/predicted_novel_distance': float(
                        torch.cat(predictive_novel_distance_parts).mean().item()
                        if predictive_novel_distance_parts else 0.0
                    ),
                    'episodic_predictive_memory/predicted_revisit_distance': float(
                        torch.cat(predictive_revisit_distance_parts).mean().item()
                        if predictive_revisit_distance_parts else 0.0
                    ),
                    'episodic_predictive_memory/target_novel_distance': float(
                        torch.cat(predictive_target_novel_distance_parts).mean().item()
                        if predictive_target_novel_distance_parts else 0.0
                    ),
                    'episodic_predictive_memory/target_revisit_distance': float(
                        torch.cat(predictive_target_revisit_distance_parts).mean().item()
                        if predictive_target_revisit_distance_parts else 0.0
                    ),
                }
            )
            for depth, values in enumerate(
                predictive_depth_cosine_parts, start=1
            ):
                stats[
                    f'episodic_predictive_memory/depth_{depth}_future_cosine'
                ] = float(np.mean(values)) if values else 0.0
                predictor_only_values = predictor_only_depth_cosine_parts[depth - 1]
                stats[
                    f'episodic_predictive_memory/depth_{depth}_predictor_only_future_cosine'
                ] = (
                    float(np.mean(predictor_only_values))
                    if predictor_only_values else 0.0
                )
                stats[
                    f'episodic_predictive_memory/depth_{depth}_future_cosine_last_update'
                ] = predictive_depth_cosine_last[depth - 1]
            if reconstructive_memory:
                stats.update(
                    {
                        'episodic_reconstructive_memory/reconstruction_loss': float(
                            np.mean(reconstruction_loss_values)
                        ),
                        'episodic_reconstructive_memory/reconstruction_loss_last_update': (
                            reconstruction_loss_last_value
                        ),
                        'episodic_reconstructive_memory/rollout_reconstruction_loss': float(
                            np.mean(rollout_reconstruction_loss_values)
                        ),
                        'episodic_reconstructive_memory/rollout_reconstruction_loss_last_update': (
                            rollout_reconstruction_loss_last_value
                        ),
                        'episodic_reconstructive_memory/reconstruction_psnr': float(
                            np.mean(reconstruction_psnr_values)
                        ),
                        'episodic_reconstructive_memory/rollout_reconstruction_psnr': float(
                            np.mean(rollout_reconstruction_psnr_values)
                        ),
                    }
                )
                for depth, values in enumerate(
                    rollout_depth_mae_parts, start=1
                ):
                    stats[
                        f'episodic_reconstructive_memory/depth_{depth}_rollout_mae'
                    ] = float(np.mean(values)) if values else 0.0
        if continuation_enabled and continuation_prediction_parts:
            continuation_predictions = torch.cat(continuation_prediction_parts)
            continuation_targets = torch.cat(continuation_target_parts)
            continuation_raw_returns = torch.cat(continuation_raw_return_parts)
            continuation_gated_returns = torch.cat(
                continuation_gated_return_parts
            )
            if (
                    continuation_predictions.numel() > 1
                    and continuation_predictions.std() > 1e-8
                    and continuation_targets.std() > 1e-8
            ):
                continuation_correlation = float(
                    np.corrcoef(
                        continuation_predictions.numpy(),
                        continuation_targets.numpy(),
                    )[0, 1]
                )
            else:
                continuation_correlation = 0.0
            stats.update(
                {
                    'episodic_sequence/continuation_loss': float(
                        np.mean(continuation_loss_values)
                    ),
                    'episodic_sequence/continuation_prediction_mean': float(
                        continuation_predictions.mean().item()
                    ),
                    'episodic_sequence/continuation_target_mean': float(
                        continuation_targets.mean().item()
                    ),
                    'episodic_sequence/continuation_mae': float(
                        torch.mean(
                            torch.abs(
                                continuation_predictions - continuation_targets
                            )
                        ).item()
                    ),
                    'episodic_sequence/continuation_correlation': (
                        continuation_correlation
                    ),
                    'episodic_sequence/continuation_raw_return_mean': float(
                        continuation_raw_returns.mean().item()
                    ),
                    'episodic_sequence/continuation_gated_return_mean': float(
                        continuation_gated_returns.mean().item()
                    ),
                    'episodic_sequence/continuation_valid_target_count': float(
                        continuation_targets.numel()
                    ),
                }
            )
            for depth, values in enumerate(
                    continuation_depth_mae_parts, start=1
            ):
                stats[
                    f'episodic_sequence/continuation_depth_{depth}_mae'
                ] = float(np.mean(values)) if values else float('nan')
        for depth, values in enumerate(depth_brier_parts, start=1):
            stats[f'episodic_sequence/depth_{depth}_brier'] = float(np.mean(values))
        self._episodic_sequence_novelty_last_train_stats = dict(stats)
        return stats

    def _set_intrinsic_heads_training(self, training: bool) -> None:
        members = getattr(self._learn_model, 'members', None)
        if members is None:
            members = [self._learn_model]
        for member in members:
            for head_name in ['intrinsic_reward_head', 'intrinsic_value_head']:
                head = getattr(member, head_name, None)
                if head is not None:
                    head.train(training)

    def _intrinsic_elementwise_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_type = str(self._cfg.get('intrinsic_loss_type', 'huber')).lower()
        if loss_type == 'mse':
            return F.mse_loss(prediction, target, reduction='none')
        if loss_type in ['huber', 'smooth_l1']:
            beta = max(1e-6, float(self._cfg.get('intrinsic_huber_delta', 1.0)))
            return F.smooth_l1_loss(prediction, target, reduction='none', beta=beta)
        raise ValueError(f"Unsupported intrinsic_loss_type={loss_type!r}; expected 'huber' or 'mse'.")

    def _episodic_inverse_loss_from_replay_unroll(
            self,
            obs_batch_ori: np.ndarray,
            action_batch: np.ndarray,
            mask_batch: torch.Tensor,
            sample_weights: torch.Tensor,
            recurrent_latents: List[torch.Tensor],
            original_batch_size: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Train inverse dynamics and align imagined latents with its embedding."""
        model = self._learn_model
        enabled = (
            getattr(model, 'episodic_inverse_encoder', None) is not None
            and float(self._cfg.get('episodic_inverse_loss_weight', 1.0)) > 0.0
            and recurrent_latents
        )
        zero = (
            recurrent_latents[0].sum() * 0.0
            if recurrent_latents else
            torch.zeros((), dtype=torch.float32, device=self._cfg.device)
        )
        if not enabled:
            return zero, {}

        observation_sequence = self._extract_observation_sequence(obs_batch_ori)
        if observation_sequence is None:
            raise ValueError(
                'Episodic inverse-dynamics training requires a valid replay observation sequence.'
            )
        actions = torch.as_tensor(
            action_batch, dtype=torch.long, device=self._cfg.device
        )
        if actions.dim() == 3 and actions.shape[-1] == 1:
            actions = actions.squeeze(-1)
        max_horizon = min(
            len(recurrent_latents),
            max(1, int(self._cfg.get('episodic_inverse_max_horizon', 5))),
            max(0, int(observation_sequence.shape[1]) - 1),
            int(mask_batch.shape[1]),
            int(actions.shape[1]),
        )
        if max_horizon <= 0:
            return zero, {}

        transition_mask = mask_batch[:original_batch_size, :max_horizon].float()
        transition_weights = (
            transition_mask * sample_weights[:original_batch_size].reshape(-1, 1)
        )
        valid_weight = transition_weights.sum()
        if valid_weight.item() <= 0.0:
            return zero, {}

        observations = observation_sequence[
            :original_batch_size, :max_horizon + 1
        ].contiguous()
        observation_shape = tuple(int(size) for size in observations.shape[2:])
        embeddings = model.episodic_inverse_embedding(
            observations.reshape(-1, *observation_shape)
        ).reshape(original_batch_size, max_horizon + 1, -1)
        current_embeddings = embeddings[:, :-1]
        next_embeddings = embeddings[:, 1:]
        inverse_logits = model.episodic_inverse_action_logits(
            current_embeddings.reshape(-1, current_embeddings.shape[-1]),
            next_embeddings.reshape(-1, next_embeddings.shape[-1]),
        ).reshape(original_batch_size, max_horizon, -1)
        target_actions = actions[:original_batch_size, :max_horizon]
        inverse_elementwise = F.cross_entropy(
            inverse_logits.reshape(-1, inverse_logits.shape[-1]),
            target_actions.reshape(-1),
            reduction='none',
        ).reshape(original_batch_size, max_horizon)
        inverse_loss = (inverse_elementwise * transition_weights).sum() / valid_weight

        ensemble_size = int(self._cfg.model.ensemble_size)
        imagined_latents = torch.stack(recurrent_latents[:max_horizon], dim=1)
        expected_rows = ensemble_size * original_batch_size
        if int(imagined_latents.shape[0]) != expected_rows:
            raise ValueError(
                f'Expected {expected_rows} recurrent latent rows for episodic bridging, '
                f'got {imagined_latents.shape[0]}.'
            )
        imagined_latents = imagined_latents.reshape(
            ensemble_size,
            original_batch_size,
            max_horizon,
            *imagined_latents.shape[2:],
        )
        imagined_latents = imagined_latents.permute(
            1,
            2,
            0,
            *range(3, imagined_latents.dim()),
        ).contiguous()
        flattened_imagined = imagined_latents.reshape(
            original_batch_size * max_horizon * ensemble_size,
            *imagined_latents.shape[3:],
        )
        member_ids = [
            member_id
            for _ in range(original_batch_size * max_horizon)
            for member_id in range(ensemble_size)
        ]
        predicted_embeddings = model.episodic_inverse_bridge(
            flattened_imagined,
            member_id=member_ids,
            detach_latent=not bool(
                self._cfg.get('episodic_inverse_train_world_model', True)
            ),
        ).reshape(original_batch_size, max_horizon, ensemble_size, -1)
        bridge_targets = next_embeddings.detach().unsqueeze(2)
        bridge_elementwise = (
            predicted_embeddings - bridge_targets
        ).pow(2).sum(dim=-1)
        bridge_weights = transition_weights.unsqueeze(-1)
        bridge_weight = valid_weight * ensemble_size
        bridge_loss = (bridge_elementwise * bridge_weights).sum() / bridge_weight

        inverse_weight = float(self._cfg.get('episodic_inverse_loss_weight', 1.0))
        bridge_loss_weight = float(
            self._cfg.get('episodic_inverse_bridge_loss_weight', 1.0)
        )
        total_loss = inverse_weight * inverse_loss + bridge_loss_weight * bridge_loss

        valid = transition_mask > 0.5
        with torch.no_grad():
            inverse_accuracy = (
                (inverse_logits.argmax(dim=-1) == target_actions).float()[valid].mean()
            )
            target_flat = next_embeddings[valid]
            current_flat = current_embeddings[valid]
            predicted_flat = predicted_embeddings[valid]
            bridge_cosine = F.cosine_similarity(
                predicted_flat,
                target_flat.unsqueeze(1).expand_as(predicted_flat),
                dim=-1,
            ).mean()
            member_center = predicted_flat.mean(dim=1, keepdim=True)
            member_disagreement = (
                predicted_flat - member_center
            ).pow(2).sum(dim=-1).mean()
            transition_distance = (
                next_embeddings - current_embeddings
            ).pow(2).sum(dim=-1)[valid].mean()
            embedding_std = target_flat.std(dim=0, unbiased=False).mean()

        log = {
            'episodic_inverse/total_loss': float(total_loss.detach().item()),
            'episodic_inverse/inverse_loss': float(inverse_loss.detach().item()),
            'episodic_inverse/inverse_accuracy': float(inverse_accuracy.item()),
            'episodic_inverse/bridge_loss': float(bridge_loss.detach().item()),
            'episodic_inverse/bridge_cosine': float(bridge_cosine.item()),
            'episodic_inverse/bridge_member_disagreement': float(
                member_disagreement.item()
            ),
            'episodic_inverse/embedding_std_mean': float(embedding_std.item()),
            'episodic_inverse/real_transition_distance_mean': float(
                transition_distance.item()
            ),
            'episodic_inverse/valid_transition_count': float(valid.sum().item()),
            'episodic_inverse/train_world_model': float(
                bool(self._cfg.get('episodic_inverse_train_world_model', True))
            ),
        }
        for depth in range(max_horizon):
            depth_valid = valid[:, depth]
            if not depth_valid.any().item():
                continue
            depth_prediction = predicted_embeddings[:, depth][depth_valid]
            depth_target = next_embeddings[:, depth][depth_valid]
            log[f'episodic_inverse/depth_{depth + 1}_bridge_loss'] = float(
                (
                    depth_prediction - depth_target.unsqueeze(1)
                ).pow(2).sum(dim=-1).mean().item()
            )
            log[f'episodic_inverse/depth_{depth + 1}_bridge_cosine'] = float(
                F.cosine_similarity(
                    depth_prediction,
                    depth_target.unsqueeze(1).expand_as(depth_prediction),
                    dim=-1,
                ).mean().item()
            )
        return total_loss, log

    def _intrinsic_loss_from_primary_unroll(
            self,
            data,
            intrinsic_reward_targets: Optional[torch.Tensor],
            intrinsic_value_targets: Optional[torch.Tensor],
            member_ids: List[int],
            latent_states: List[torch.Tensor],
            mask_batch: torch.Tensor,
            weights: torch.Tensor,
            original_batch_size: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Build intrinsic and query losses on MuZero's primary replay unroll."""
        zero = latent_states[0].sum() * 0.0
        if (
                not self._intrinsic_value_training_enabled()
                or intrinsic_reward_targets is None
                or float(self._cfg.get('intrinsic_model_loss_weight', 1.0)) == 0.0
        ):
            return zero, {}
        num_unroll_steps = min(int(self._cfg.num_unroll_steps), len(latent_states) - 1)
        if num_unroll_steps <= 0:
            return zero, {}

        ensemble_size = int(self._cfg.model.ensemble_size)
        expected_rows = ensemble_size * original_batch_size
        if any(int(latent.shape[0]) != expected_rows for latent in latent_states[:num_unroll_steps + 1]):
            raise ValueError(
                "Primary MuZero unroll does not match the ensemble-expanded intrinsic batch."
            )
        if len(member_ids) != expected_rows:
            raise ValueError(
                f"Expected {expected_rows} member ids, got {len(member_ids)}."
            )

        current_batch = data[0]
        current_batch_core = self._current_batch_core(current_batch)
        indices = current_batch_core[3]
        intrinsic_sample_mask = torch.ones(
            original_batch_size, dtype=torch.float32, device=self._cfg.device
        )
        current_phase_only = (
            self._intrinsic_reward_source() == 'state_density'
            and bool(self._cfg.get('state_density_train_on_intrinsic_replay_only', False))
        )
        if current_phase_only:
            intrinsic_sample_mask = torch.as_tensor(
                [str(index).startswith('intrinsic:') for index in np.asarray(indices).reshape(-1)],
                dtype=torch.float32,
                device=self._cfg.device,
            )
        intrinsic_sample_count = int(intrinsic_sample_mask.sum().item())
        repeated_intrinsic_sample_mask = self._repeat_tensor_for_members(
            intrinsic_sample_mask, ensemble_size
        )
        reward_mask_batch = mask_batch * repeated_intrinsic_sample_mask.unsqueeze(1)

        value_pure_exploration_only = bool(
            self._cfg.get('intrinsic_value_train_on_pure_exploration_only', False)
        )
        pure_exploration_episode_batch = self._pure_exploration_episode_batch(current_batch)
        if value_pure_exploration_only:
            if pure_exploration_episode_batch is None:
                raise ValueError(
                    'intrinsic_value_train_on_pure_exploration_only requires pure-exploration '
                    'episode flags in the replay batch.'
                )
            pure_exploration_sample_mask = torch.as_tensor(
                pure_exploration_episode_batch,
                dtype=torch.float32,
                device=self._cfg.device,
            ).view(-1)
            if pure_exploration_sample_mask.numel() != original_batch_size:
                raise ValueError(
                    'Expected one pure-exploration flag per replay sample, got '
                    f'{pure_exploration_sample_mask.numel()} for batch size {original_batch_size}.'
                )
            pure_exploration_sample_mask = (pure_exploration_sample_mask > 0.5).float()
        else:
            pure_exploration_sample_mask = torch.ones_like(intrinsic_sample_mask)
        value_sample_mask = intrinsic_sample_mask * pure_exploration_sample_mask
        value_sample_count = int(value_sample_mask.sum().item())
        repeated_value_sample_mask = self._repeat_tensor_for_members(
            value_sample_mask, ensemble_size
        )
        value_mask_batch = mask_batch * repeated_value_sample_mask.unsqueeze(1)

        reward_targets = self._align_intrinsic_targets_for_members(
            intrinsic_reward_targets.to(self._cfg.device), ensemble_size, original_batch_size
        )
        transform_reward_target = getattr(
            self._learn_model, 'transform_intrinsic_reward_target', None
        )
        reward_training_targets = (
            transform_reward_target(reward_targets)
            if callable(transform_reward_target) else reward_targets
        )
        train_value_head = (
            bool(self._cfg.get('intrinsic_train_value_head', True))
            and intrinsic_value_targets is not None
            and float(self._cfg.get('intrinsic_value_loss_weight', 1.0)) > 0.0
        )
        value_targets = None
        if train_value_head:
            value_targets = self._align_intrinsic_targets_for_members(
                intrinsic_value_targets.to(self._cfg.device), ensemble_size, original_batch_size
            )

        head_only = bool(self._cfg.get('intrinsic_head_only', True))
        reward_loss_sum = zero
        reward_weight_sum = zero
        value_loss_sum = zero
        value_weight_sum = zero
        reward_prediction_parts = []
        reward_target_parts = []
        reward_member_parts = []
        reward_depth_diagnostics = {}
        value_prediction_parts = []
        value_target_parts = []
        value_member_parts = []
        member_id_tensor = torch.as_tensor(
            member_ids, dtype=torch.long, device=self._cfg.device
        )

        def record_value(latent: torch.Tensor, target_index: int) -> None:
            nonlocal value_loss_sum, value_weight_sum
            prediction_latent = latent.detach() if head_only else latent
            prediction = self._learn_model.intrinsic_value_from_latent(
                prediction_latent, member_id=member_ids
            ).squeeze(-1)
            value_mask = value_mask_batch[:, target_index]
            elementwise = self._intrinsic_elementwise_loss(
                prediction, value_targets[:, target_index]
            )
            value_loss_sum = value_loss_sum + (elementwise * value_mask * weights).sum()
            value_weight_sum = value_weight_sum + (value_mask * weights).sum()
            valid = value_mask > 0.5
            value_prediction_parts.append(prediction.detach()[valid])
            value_target_parts.append(value_targets[:, target_index].detach()[valid])
            value_member_parts.append(member_id_tensor[valid])

        if train_value_head:
            record_value(latent_states[0], 0)

        reward_loss_weight = float(self._cfg.get('intrinsic_reward_loss_weight', 1.0))
        value_loss_weight = float(self._cfg.get('intrinsic_value_loss_weight', 1.0))
        recurrent_latents = latent_states[1:num_unroll_steps + 1]
        for step_k, latent in enumerate(recurrent_latents):
            if reward_loss_weight > 0.0 and step_k < reward_targets.shape[1]:
                prediction_latent = latent.detach() if head_only else latent
                prediction_fn = getattr(
                    self._learn_model,
                    'intrinsic_reward_prediction_from_latent',
                    self._learn_model.intrinsic_reward_from_latent,
                )
                training_prediction = prediction_fn(
                    prediction_latent, member_id=member_ids
                ).squeeze(-1)
                reward_mask = reward_mask_batch[:, step_k]
                elementwise = self._intrinsic_elementwise_loss(
                    training_prediction, reward_training_targets[:, step_k]
                )
                reward_loss_sum = reward_loss_sum + (elementwise * reward_mask * weights).sum()
                reward_weight_sum = reward_weight_sum + (reward_mask * weights).sum()
                valid = reward_mask > 0.5
                inverse_reward_prediction = getattr(
                    self._learn_model,
                    'inverse_transform_intrinsic_reward_prediction',
                    None,
                )
                raw_prediction = (
                    inverse_reward_prediction(training_prediction)
                    if callable(inverse_reward_prediction) else training_prediction
                )
                reward_prediction_parts.append(raw_prediction.detach()[valid])
                reward_target_parts.append(reward_targets[:, step_k].detach()[valid])
                reward_member_parts.append(member_id_tensor[valid])
                if valid.any().item():
                    depth_prediction = raw_prediction.detach()[valid]
                    depth_target = reward_targets[:, step_k].detach()[valid]
                    depth_training_prediction = training_prediction.detach()[valid]
                    depth_training_target = reward_training_targets[:, step_k].detach()[valid]
                    prefix = f'intrinsic/reward_depth_{step_k + 1}'
                    reward_depth_diagnostics.update(
                        {
                            f'{prefix}_prediction_mae_raw': float(
                                (depth_prediction - depth_target).abs().mean().item()
                            ),
                            f'{prefix}_prediction_mae_log': float(
                                (
                                    depth_prediction.clamp_min(1e-10).log()
                                    - depth_target.clamp_min(1e-10).log()
                                ).abs().mean().item()
                            ),
                            f'{prefix}_prediction_mae_target_space': float(
                                (
                                    depth_training_prediction - depth_training_target
                                ).abs().mean().item()
                            ),
                            f'{prefix}_prediction_correlation': (
                                self._intrinsic_diagnostic_correlation(
                                    depth_prediction, depth_target
                                )
                            ),
                            f'{prefix}_valid_count': float(valid.sum().item()),
                        }
                    )
            if train_value_head and step_k + 1 < value_targets.shape[1]:
                record_value(latent, step_k + 1)

        reward_loss_mean = reward_loss_sum / reward_weight_sum.clamp_min(1e-6)
        value_loss_mean = (
            value_loss_sum / value_weight_sum.clamp_min(1e-6)
            if train_value_head else value_loss_sum
        )
        intrinsic_model_loss_weight = float(
            self._cfg.get('intrinsic_model_loss_weight', 1.0)
        )
        private_reward_loss = (
            intrinsic_model_loss_weight * reward_loss_weight * reward_loss_mean
        )
        reward_value_loss = private_reward_loss + (
            intrinsic_model_loss_weight * value_loss_weight * value_loss_mean
        )

        original_mask_batch = torch.as_tensor(
            current_batch_core[2], dtype=torch.float32, device=self._cfg.device
        )
        original_weights = torch.as_tensor(
            current_batch_core[4], dtype=torch.float32, device=self._cfg.device
        ).view(-1)
        query_loss, query_log = self._rnd_landscape_query_loss_from_replay_unroll(
            current_batch_core[0],
            original_mask_batch,
            original_weights,
            recurrent_latents,
            original_batch_size,
        )
        auxiliary_decoder_loss, auxiliary_decoder_log = (
            self._rnd_auxiliary_decoder_loss_from_replay_unroll(
                current_batch_core[0],
                original_mask_batch,
                original_weights,
                recurrent_latents,
                original_batch_size,
            )
        )
        episodic_inverse_loss, episodic_inverse_log = (
            self._episodic_inverse_loss_from_replay_unroll(
                current_batch_core[0],
                current_batch_core[1],
                original_mask_batch,
                original_weights,
                recurrent_latents,
                original_batch_size,
            )
        )
        gradient_interaction_log = self._intrinsic_aux_gradient_diagnostics(
            private_reward_loss, query_loss
        )
        total_loss = (
            reward_value_loss + query_loss + auxiliary_decoder_loss
            + episodic_inverse_loss
        )
        if (
                reward_weight_sum.item() <= 0.0
                and value_weight_sum.item() <= 0.0
                and not query_log
                and not auxiliary_decoder_log
                and not episodic_inverse_log
        ):
            return zero, {
                'intrinsic/state_density_current_phase_only': float(current_phase_only),
                'intrinsic/training_sample_count': 0.0,
                'intrinsic/training_sample_fraction': 0.0,
                'intrinsic/reward_training_sample_count': float(intrinsic_sample_count),
                'intrinsic/value_training_sample_count': float(value_sample_count),
                'intrinsic/value_pure_exploration_only': float(value_pure_exploration_only),
                'intrinsic/skipped_no_eligible_samples': 1.0,
            }

        log = {
            'intrinsic/model_loss': float(total_loss.detach().item()),
            'intrinsic/reward_value_model_loss': float(reward_value_loss.detach().item()),
            'intrinsic/reward_loss': float(reward_loss_mean.detach().item()),
            'intrinsic/value_loss': float(value_loss_mean.detach().item()),
            'intrinsic/train_value_head': float(train_value_head),
            'intrinsic/learning_rate': float(
                self._optimizer.param_groups[
                    getattr(self, '_intrinsic_head_optimizer_group_index', 0)
                ]['lr']
            ),
            'intrinsic/shared_parameter_learning_rate': float(
                self._optimizer.param_groups[0]['lr']
            ),
            'intrinsic/head_only': float(head_only),
            'intrinsic/state_density_current_phase_only': float(current_phase_only),
            'intrinsic/training_sample_count': float(intrinsic_sample_count),
            'intrinsic/training_sample_fraction': float(
                intrinsic_sample_count / max(1, original_batch_size)
            ),
            'intrinsic/reward_training_sample_count': float(intrinsic_sample_count),
            'intrinsic/reward_training_sample_fraction': float(
                intrinsic_sample_count / max(1, original_batch_size)
            ),
            'intrinsic/value_training_sample_count': float(value_sample_count),
            'intrinsic/value_training_sample_fraction': float(
                value_sample_count / max(1, original_batch_size)
            ),
            'intrinsic/value_pure_exploration_only': float(value_pure_exploration_only),
            'intrinsic/skipped_no_pure_value_samples': float(
                train_value_head and value_pure_exploration_only and value_sample_count == 0
            ),
            'intrinsic/shared_muzero_optimizer': 1.0,
            'intrinsic/reward_target_transform_log': float(
                str(
                    getattr(
                        getattr(self._learn_model, 'members', [self._learn_model])[0],
                        'intrinsic_reward_target_transform',
                        'raw',
                    )
                ).lower() == 'log'
            ),
        }
        log.update(reward_depth_diagnostics)
        log.update(query_log)
        log.update(auxiliary_decoder_log)
        log.update(episodic_inverse_log)
        log.update(gradient_interaction_log)

        diagnostic_groups = [
            (
                'reward', reward_prediction_parts, reward_target_parts, reward_member_parts
            ),
            (
                'value', value_prediction_parts, value_target_parts, value_member_parts
            ),
        ]
        for name, prediction_parts, target_parts, member_parts in diagnostic_groups:
            if not prediction_parts:
                continue
            predictions = torch.cat(prediction_parts)
            targets = torch.cat(target_parts)
            members = torch.cat(member_parts)
            log.update(
                {
                    f'intrinsic/{name}_prediction_mean': float(predictions.mean().item()),
                    f'intrinsic/{name}_prediction_std': float(
                        predictions.std(unbiased=False).item()
                    ),
                    f'intrinsic/{name}_training_target_mean': float(targets.mean().item()),
                    f'intrinsic/{name}_prediction_mae': float(
                        (predictions - targets).abs().mean().item()
                    ),
                    f'intrinsic/{name}_prediction_correlation': (
                        self._intrinsic_diagnostic_correlation(predictions, targets)
                    ),
                }
            )
            for member_id in range(ensemble_size):
                member_mask = members == member_id
                if member_mask.any():
                    log[f'intrinsic/member_{member_id}_{name}_prediction_mean'] = float(
                        predictions[member_mask].mean().item()
                    )
                    log[f'intrinsic/member_{member_id}_{name}_prediction_mae'] = float(
                        (predictions[member_mask] - targets[member_mask]).abs().mean().item()
                    )
        return total_loss, log

    def _rnd_auxiliary_decoder_loss_from_replay_unroll(
            self,
            obs_batch_ori: np.ndarray,
            mask_batch: torch.Tensor,
            sample_weights: torch.Tensor,
            recurrent_latents: List[torch.Tensor],
            original_batch_size: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Supervise h/g at every recurrent depth with current real-state RND."""
        if (
                not self._rnd_auxiliary_decoder_enabled()
                or float(
                    self._cfg.get('rnd_auxiliary_decoder_loss_weight', 1.0)
                ) <= 0.0
                or not recurrent_latents
        ):
            zero = (
                recurrent_latents[0].sum() * 0.0
                if recurrent_latents else
                torch.zeros((), dtype=torch.float32, device=self._cfg.device)
            )
            return zero, {}

        snapshot = self._rnd_target_model_for_targets()
        if snapshot is None:
            return recurrent_latents[0].sum() * 0.0, {}
        observation_sequence = self._extract_observation_sequence(obs_batch_ori)
        if observation_sequence is None:
            raise ValueError(
                'RND auxiliary decoder requires a valid replay observation sequence.'
            )
        max_horizon = min(
            len(recurrent_latents),
            max(1, int(self._cfg.get('rnd_auxiliary_decoder_max_horizon', 8))),
            max(0, int(observation_sequence.shape[1]) - 1),
            int(mask_batch.shape[1]),
        )
        if max_horizon <= 0:
            return recurrent_latents[0].sum() * 0.0, {}

        target_mask = mask_batch[:original_batch_size, :max_horizon].float()
        if target_mask.sum().item() <= 0.0:
            return recurrent_latents[0].sum() * 0.0, {}
        target_observations = self._select_rnd_observation_features(
            observation_sequence[
                :original_batch_size, 1:max_horizon + 1
            ]
        )
        flat_target_observations = self._flatten_observation_time_batch(
            target_observations
        )
        snapshot.eval()
        with torch.no_grad():
            target_errors = snapshot.error(flat_target_observations).reshape(
                original_batch_size, max_horizon
            ).detach()

        ensemble_size = int(self._cfg.model.ensemble_size)
        imagined_latents = torch.stack(recurrent_latents[:max_horizon], dim=1)
        expected_rows = ensemble_size * original_batch_size
        if imagined_latents.shape[0] != expected_rows:
            raise ValueError(
                f'Expected {expected_rows} recurrent latent rows, got '
                f'{imagined_latents.shape[0]}.'
            )
        imagined_latents = imagined_latents.reshape(
            ensemble_size,
            original_batch_size,
            max_horizon,
            *imagined_latents.shape[2:],
        )
        imagined_latents = imagined_latents.permute(
            1, 2, 0, *range(3, imagined_latents.dim())
        ).contiguous()
        train_world_model = bool(
            self._cfg.get('rnd_auxiliary_decoder_train_world_model', True)
        )
        if not train_world_model:
            imagined_latents = imagined_latents.detach()

        decoder = self._ensure_rnd_auxiliary_decoder_model(
            imagined_latents.shape[3:]
        )
        decoder.train()
        raw_loss, terms = decoder.loss(
            imagined_latents,
            target_errors,
            target_mask,
            sample_weights[:original_batch_size],
        )
        loss_weight = float(
            self._cfg.get('rnd_auxiliary_decoder_loss_weight', 1.0)
        )
        weighted_loss = loss_weight * raw_loss
        log = {
            'rnd_auxiliary_decoder/train_loss': float(raw_loss.detach().item()),
            'rnd_auxiliary_decoder/weighted_loss': float(
                weighted_loss.detach().item()
            ),
            'rnd_auxiliary_decoder/loss_weight': loss_weight,
            'rnd_auxiliary_decoder/prediction_mae_target_space': float(
                terms['prediction_mae_target_space'].item()
            ),
            'rnd_auxiliary_decoder/prediction_mae_raw': float(
                terms['prediction_mae_raw'].item()
            ),
            'rnd_auxiliary_decoder/prediction_correlation_raw': float(
                terms['prediction_correlation_raw'].item()
            ),
            'rnd_auxiliary_decoder/prediction_target_mean_raw': float(
                terms['prediction_target_mean_raw'].item()
            ),
            'rnd_auxiliary_decoder/prediction_target_std_raw': float(
                terms['prediction_target_std_raw'].item()
            ),
            'rnd_auxiliary_decoder/valid_target_fraction': float(
                terms['valid_target_fraction'].item()
            ),
            'rnd_auxiliary_decoder/replay_root_count': float(original_batch_size),
            'rnd_auxiliary_decoder/replay_horizon': float(max_horizon),
            'rnd_auxiliary_decoder/shared_across_members': 1.0,
            'rnd_auxiliary_decoder/target_transform_log': float(
                decoder.target_transform == 'log'
            ),
            'rnd_auxiliary_decoder/joint_world_model_update': float(
                train_world_model
            ),
            'rnd_auxiliary_decoder/used_in_mcts': 0.0,
        }
        for key, value in terms.items():
            if key.startswith('depth_'):
                log[f'rnd_auxiliary_decoder/{key}'] = float(value.item())
        self._rnd_auxiliary_decoder_last_train_stats = dict(log)
        return weighted_loss, log

    def _rnd_landscape_query_loss_from_replay_unroll(
            self,
            obs_batch_ori: np.ndarray,
            mask_batch: torch.Tensor,
            sample_weights: torch.Tensor,
            recurrent_latents: List[torch.Tensor],
            original_batch_size: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Predict current RND errors at real replay successors from one h/g unroll."""
        if (
                not self._rnd_landscape_query_enabled()
                or float(self._cfg.get('rnd_landscape_query_loss_weight', 1.0)) <= 0.0
                or not recurrent_latents
        ):
            zero = recurrent_latents[0].sum() * 0.0 if recurrent_latents else torch.zeros(
                (), dtype=torch.float32, device=self._cfg.device
            )
            return zero, {}
        if bool(self._cfg.get('intrinsic_head_only', True)):
            raise ValueError(
                "Joint RND landscape-query training requires intrinsic_head_only=False "
                "so its loss can train both h and g."
            )

        query_model = self._ensure_rnd_landscape_query_model()
        snapshot = self._rnd_target_model_for_targets()
        if query_model is None or snapshot is None:
            return recurrent_latents[0].sum() * 0.0, {}
        if any(parameter.requires_grad for parameter in snapshot.parameters()):
            raise RuntimeError(
                "RND landscape-query replay loss requires a frozen RND target snapshot."
            )
        error_with_input_grad = getattr(snapshot, 'error_with_input_grad', None)
        if not callable(error_with_input_grad):
            raise RuntimeError(
                "The frozen RND snapshot must expose error_with_input_grad()."
            )

        observation_sequence = self._extract_observation_sequence(obs_batch_ori)
        if observation_sequence is None:
            raise ValueError(
                "RND landscape-query replay training requires a valid observation sequence."
            )
        frame_stack_num = max(1, int(self._cfg.model.get('frame_stack_num', 1)))
        max_horizon = min(
            len(recurrent_latents),
            max(1, int(self._cfg.get('rnd_landscape_query_max_horizon', 8))),
            max(0, int(observation_sequence.shape[1]) - frame_stack_num),
            int(mask_batch.shape[1]),
        )
        if max_horizon <= 0:
            return recurrent_latents[0].sum() * 0.0, {}

        target_mask = mask_batch[:original_batch_size, :max_horizon].float()
        if target_mask.sum().item() <= 0.0:
            return recurrent_latents[0].sum() * 0.0, {}
        roots = self._select_rnd_observation_features(
            observation_sequence[:original_batch_size, frame_stack_num - 1]
        )
        target_states = self._select_rnd_observation_features(
            observation_sequence[
                :original_batch_size,
                frame_stack_num:frame_stack_num + max_horizon,
            ]
        )
        if self._rnd_landscape_query_space() == 'compact_image':
            observation_shape = tuple(int(size) for size in roots.shape[-3:])
            roots = roots.reshape(original_batch_size, *observation_shape)
            target_states = target_states.reshape(
                original_batch_size, max_horizon, *observation_shape
            )
            flat_target_states = target_states.reshape(-1, *observation_shape)
        else:
            roots = roots.reshape(original_batch_size, 2)
            target_states = target_states.reshape(original_batch_size, max_horizon, 2)
            flat_target_states = target_states.reshape(-1, 2)

        with torch.no_grad():
            target_errors = snapshot.error(
                flat_target_states
            ).reshape(original_batch_size, max_horizon).detach()

        ensemble_size = int(self._cfg.model.ensemble_size)
        imagined_latents = torch.stack(recurrent_latents[:max_horizon], dim=1)
        expected_rows = ensemble_size * original_batch_size
        if imagined_latents.shape[0] != expected_rows:
            raise ValueError(
                "Expected member-major recurrent latents with "
                f"{expected_rows} rows, got {imagined_latents.shape[0]}."
            )
        imagined_latents = imagined_latents.reshape(
            ensemble_size,
            original_batch_size,
            max_horizon,
            *imagined_latents.shape[2:],
        )
        imagined_latents = imagined_latents.permute(
            1, 2, 0, *range(3, imagined_latents.dim())
        ).contiguous()

        train_world_model = bool(
            self._cfg.get('rnd_landscape_query_train_world_model', True)
        )
        if not train_world_model:
            imagined_latents = imagined_latents.detach()

        def encode_query_states(states: torch.Tensor) -> torch.Tensor:
            encoded = self._rnd_landscape_encode_query_states(
                self._learn_model, states
            )
            return encoded if train_world_model else encoded.detach()

        query_model.train()
        raw_loss, terms = query_model.loss_on_observations(
            roots,
            imagined_latents,
            target_errors,
            error_fn=error_with_input_grad,
            query_latent_fn=encode_query_states,
            target_mask=target_mask,
            sample_weights=sample_weights[:original_batch_size],
            diversity_weight=float(
                self._cfg.get('rnd_landscape_query_diversity_weight', 0.01)
            ),
            centered_prediction_weight=float(
                self._cfg.get('rnd_landscape_query_centered_prediction_weight', 0.0)
            ),
        )
        loss_weight = float(self._cfg.get('rnd_landscape_query_loss_weight', 1.0))
        weighted_loss = loss_weight * raw_loss
        gradient_diagnostic_log = {}
        gradient_interval = max(
            0, int(self._cfg.get('rnd_landscape_query_gradient_diagnostics_interval', 0))
        )
        if (
                gradient_interval > 0
                and int(getattr(self, '_learn_update_count', 0)) % gradient_interval == 0
        ):
            representation_parameters = []
            dynamics_parameters = []
            for name, parameter in self._learn_model.named_parameters():
                if not parameter.requires_grad:
                    continue
                if 'representation_network' in name:
                    representation_parameters.append(parameter)
                elif 'dynamics_network' in name:
                    dynamics_parameters.append(parameter)
            reader_parameters = [
                parameter for parameter in query_model.parameters() if parameter.requires_grad
            ]
            diagnostic_parameters = (
                representation_parameters + dynamics_parameters + reader_parameters
            )
            diagnostic_gradients = torch.autograd.grad(
                weighted_loss,
                diagnostic_parameters,
                retain_graph=True,
                allow_unused=True,
            )

            def gradient_norm(start: int, count: int) -> float:
                squared_norm = weighted_loss.new_zeros(())
                for gradient in diagnostic_gradients[start:start + count]:
                    if gradient is not None:
                        squared_norm = squared_norm + gradient.detach().float().square().sum()
                return float(squared_norm.sqrt().item())

            representation_count = len(representation_parameters)
            dynamics_count = len(dynamics_parameters)
            gradient_diagnostic_log = {
                'rnd_landscape_query/aux_representation_grad_norm': gradient_norm(
                    0, representation_count
                ),
                'rnd_landscape_query/aux_dynamics_grad_norm': gradient_norm(
                    representation_count, dynamics_count
                ),
                'rnd_landscape_query/aux_reader_grad_norm': gradient_norm(
                    representation_count + dynamics_count, len(reader_parameters)
                ),
            }
        log = {
            'rnd_landscape_query/train_steps': 1.0,
            'rnd_landscape_query/snapshot_count': float(
                self._rnd_landscape_query_snapshot_count
            ),
            'rnd_landscape_query/train_loss': float(raw_loss.detach().item()),
            'rnd_landscape_query/weighted_loss': float(weighted_loss.detach().item()),
            'rnd_landscape_query/loss_weight': loss_weight,
            'rnd_landscape_query/weight_decay': float(
                self._cfg.get('rnd_landscape_query_weight_decay', 0.0)
            ),
            'rnd_landscape_query/prediction_loss': float(
                terms['prediction_loss'].item()
            ),
            'rnd_landscape_query/centered_prediction_loss': float(
                terms['centered_prediction_loss'].item()
            ),
            'rnd_landscape_query/centered_prediction_correlation': float(
                terms['centered_prediction_correlation'].item()
            ),
            'rnd_landscape_query/prediction_mae_log': float(
                terms['prediction_mae_log'].item()
            ),
            'rnd_landscape_query/prediction_mae_target_space': float(
                terms['prediction_mae_target_space'].item()
            ),
            'rnd_landscape_query/prediction_mae_raw': float(
                terms['prediction_mae_raw'].item()
            ),
            'rnd_landscape_query/prediction_target_mean': float(
                terms['prediction_target_mean'].item()
            ),
            'rnd_landscape_query/prediction_target_std': float(
                terms['prediction_target_std'].item()
            ),
            'rnd_landscape_query/transformed_target_mean': float(
                terms['transformed_target_mean'].item()
            ),
            'rnd_landscape_query/transformed_target_std': float(
                terms['transformed_target_std'].item()
            ),
            'rnd_landscape_query/prediction_target_correlation': float(
                terms['prediction_target_correlation'].item()
            ),
            'rnd_landscape_query/target_scale': float(terms['target_scale'].item()),
            'rnd_landscape_query/target_clip_fraction': float(
                terms['target_clip_fraction'].item()
            ),
            'rnd_landscape_query/diversity_loss': float(
                terms['diversity_loss'].item()
            ),
            'rnd_landscape_query/query_pair_distance_mean': float(
                terms['query_pair_distance_mean'].item()
            ),
            'rnd_landscape_query/query_pair_distance_min': float(
                terms['query_pair_distance_min'].item()
            ),
            'rnd_landscape_query/query_root_distance_mean': float(
                terms['query_root_distance_mean'].item()
            ),
            'rnd_landscape_query/query_error_mean': float(
                terms['query_error_mean'].item()
            ),
            'rnd_landscape_query/target_error_mean': float(
                terms['target_error_mean'].item()
            ),
            'rnd_landscape_query/valid_target_fraction': float(
                terms['valid_target_fraction'].item()
            ),
            'rnd_landscape_query/replay_root_count': float(original_batch_size),
            'rnd_landscape_query/replay_horizon': float(max_horizon),
            'rnd_landscape_query/reader_use_context': float(
                query_model.reader_use_context
            ),
            'rnd_landscape_query/member_specific_reader': float(
                query_model.member_specific_reader
            ),
            'rnd_landscape_query/reader_architecture_landscape': float(
                query_model.reader_architecture == 'landscape'
            ),
            'rnd_landscape_query/reader_architecture_ordinary': float(
                query_model.reader_architecture == 'ordinary'
            ),
            'rnd_landscape_query/reader_last_linear_layer_init_zero': float(
                query_model.reader_last_linear_layer_init_zero
            ),
            'rnd_landscape_query/target_transform_log': float(
                query_model.target_transform == 'log'
            ),
            'rnd_landscape_query/target_log_reference': float(
                query_model.target_log_reference
            ),
            'rnd_landscape_query/error_epsilon': float(
                query_model.error_epsilon
            ),
            'rnd_landscape_query/target_transform_normalized_raw': float(
                query_model.target_transform == 'normalized_raw'
            ),
            'rnd_landscape_query/target_scale_phase': float(
                self._rnd_landscape_query_target_scale_phase
            ),
            'rnd_landscape_query/target_scale_sample_count': float(
                self._rnd_landscape_query_target_scale_sample_count
            ),
            'rnd_landscape_query/joint_world_model_update': float(train_world_model),
            'rnd_landscape_query/use_tree_rewards': float(
                self._rnd_landscape_query_tree_rewards_enabled()
            ),
            'rnd_landscape_query/space_compact_image': float(
                self._rnd_landscape_query_space() == 'compact_image'
            ),
        }
        for key in (
                'prediction_objective',
                'reconstruction_loss',
                'reconstruction_weight',
                'reconstruction_mae',
                'root_code_std',
                'query_code_std',
                'decoded_query_min',
                'decoded_query_max',
        ):
            if key in terms:
                log[f'rnd_landscape_query/{key}'] = float(terms[key].item())
        for key, value in terms.items():
            if key.startswith('depth_'):
                log[f'rnd_landscape_query/{key}'] = float(value.item())
        log.update(gradient_diagnostic_log)
        self._rnd_landscape_query_last_train_stats = dict(log)
        return weighted_loss, log

    def _replay_auxiliary_loss(
            self,
            data,
            obs_batch_ori,
            mask_batch: torch.Tensor,
            weights: torch.Tensor,
            latent_states: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        pending = getattr(self, '_pending_intrinsic_replay_loss', None)
        if pending is None:
            return super()._replay_auxiliary_loss(
                data, obs_batch_ori, mask_batch, weights, latent_states
            )
        loss, log = self._intrinsic_loss_from_primary_unroll(
            pending['data'],
            pending['reward_targets'],
            pending['value_targets'],
            pending['member_ids'],
            latent_states,
            mask_batch,
            weights,
            pending['original_batch_size'],
        )
        self._rnd_landscape_query_has_backward = bool(
            'rnd_landscape_query/train_loss' in log
        )
        self._rnd_auxiliary_decoder_has_backward = bool(
            'rnd_auxiliary_decoder/train_loss' in log
        )
        return loss, log

    def _after_replay_auxiliary_backward(self) -> Dict[str, float]:
        log = {}
        if bool(getattr(self, '_rnd_landscape_query_has_backward', False)):
            query_model = self._rnd_landscape_query_model
            if self._cfg.multi_gpu:
                self.sync_gradients(query_model)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                query_model.parameters(),
                float(self._cfg.get('rnd_landscape_query_grad_clip', 5.0)),
            )
            group_index = getattr(
                self, '_rnd_landscape_query_optimizer_group_index', None
            )
            learning_rate = (
                float(self._optimizer.param_groups[group_index]['lr'])
                if group_index is not None else 0.0
            )
            log.update(
                {
                    'rnd_landscape_query/grad_norm_before_clip': float(
                        grad_norm.item()
                    ),
                    'rnd_landscape_query/learning_rate': learning_rate,
                    'rnd_landscape_query/shared_muzero_optimizer': 1.0,
                }
            )
        if bool(getattr(self, '_rnd_auxiliary_decoder_has_backward', False)):
            decoder = self._rnd_auxiliary_decoder_model
            if self._cfg.multi_gpu:
                self.sync_gradients(decoder)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                decoder.parameters(),
                float(self._cfg.get('rnd_auxiliary_decoder_grad_clip', 5.0)),
            )
            group_index = getattr(
                self, '_rnd_auxiliary_decoder_optimizer_group_index', None
            )
            learning_rate = (
                float(self._optimizer.param_groups[group_index]['lr'])
                if group_index is not None else 0.0
            )
            log.update(
                {
                    'rnd_auxiliary_decoder/grad_norm_before_clip': float(
                        grad_norm.item()
                    ),
                    'rnd_auxiliary_decoder/learning_rate': learning_rate,
                    'rnd_auxiliary_decoder/shared_muzero_optimizer': 1.0,
                }
            )
        return log

    def _normalize_dynamics_prior_alignment_target(self, latent_state: torch.Tensor) -> torch.Tensor:
        mode = str(self._cfg.get('dynamics_prior_alignment_target_normalization', 'none')).lower()
        if mode == 'same_as_trainable':
            if hasattr(self._learn_model, '_normalize_trainable_dynamics_latent'):
                return self._learn_model._normalize_trainable_dynamics_latent(latent_state)
            return latent_state
        if mode == 'none':
            return latent_state
        scale = float(self._cfg.model.get('dynamics_trainable_latent_norm_scale', 1.0))
        eps = float(self._cfg.model.get('dynamics_trainable_latent_norm_eps', 1e-3))
        if latent_state.dim() <= 1:
            return latent_state
        reduce_dims = tuple(range(1, latent_state.dim()))
        latent_float = latent_state.float()
        if mode == 'l2':
            denom = latent_float.pow(2).sum(dim=reduce_dims, keepdim=True).sqrt()
        elif mode == 'rms':
            denom = latent_float.pow(2).mean(dim=reduce_dims, keepdim=True).sqrt()
        else:
            raise ValueError(
                "dynamics_prior_alignment_target_normalization must be one of "
                "'same_as_trainable', 'none', 'l2', or 'rms', "
                f"got {mode!r}."
            )
        denom = denom.clamp_min(eps).to(dtype=latent_state.dtype)
        return latent_state * (scale / denom)

    def _train_dynamics_prior_alignment_from_replay_batch(self, data, member_ids: List[int]) -> dict:
        weight = float(self._cfg.get('dynamics_prior_alignment_loss_weight', 0.0))
        if weight <= 0.0:
            return {}
        if getattr(self._learn_model, 'dynamics_prior_networks', None) is None:
            return {}
        if not bool(self._cfg.model.get('self_supervised_learning_loss', False)):
            return {}

        current_batch = data[0]
        obs_batch_ori, action_batch, mask_batch, _indices, weights, _make_time = self._current_batch_core(current_batch)
        obs_batch, obs_target_batch = prepare_obs(obs_batch_ori, self._cfg)
        if obs_target_batch is None:
            return {}
        obs_batch = obs_batch.float()
        obs_target_batch = obs_target_batch.float()

        if self._cfg.use_augmentation:
            obs_batch = self.image_transforms.transform(obs_batch)
            obs_target_batch = self.image_transforms.transform(obs_target_batch)

        batch_size = int(obs_batch.shape[0])
        if len(member_ids) != batch_size:
            raise ValueError(
                f"Expected {batch_size} member ids for dynamics prior alignment, got {len(member_ids)}."
            )
        action_batch = torch.from_numpy(action_batch).to(self._cfg.device).unsqueeze(-1).long()
        mask_batch = torch.as_tensor(mask_batch, dtype=torch.float32, device=self._cfg.device)
        weights = torch.as_tensor(weights, dtype=torch.float32, device=self._cfg.device).view(-1)

        self._learn_model.train()
        self._learn_model.set_active_members(member_ids)
        if hasattr(self._learn_model, 'set_collect_dynamics_prior_stats'):
            self._learn_model.set_collect_dynamics_prior_stats(True)
        network_output = self._learn_model.initial_inference(obs_batch, member_id=member_ids)
        latent_state = network_output.latent_state

        loss_sum = torch.zeros((), dtype=torch.float32, device=self._cfg.device)
        weight_sum = torch.zeros((), dtype=torch.float32, device=self._cfg.device)
        pred_l2_terms = []
        target_l2_terms = []

        try:
            for step_k in range(int(self._cfg.num_unroll_steps)):
                network_output = self._learn_model.recurrent_inference(
                    latent_state, action_batch[:, step_k], member_id=member_ids
                )
                latent_state = network_output.latent_state

                beg_index, end_index = self._get_target_obs_index_in_step_k(step_k)
                with torch.no_grad():
                    target_member_id = 0 if bool(self._cfg.model.get('shared_representation', False)) else member_ids
                    target_output = self._learn_model.initial_inference(
                        obs_target_batch[:, beg_index:end_index], member_id=target_member_id
                    )
                    target_latent = self._normalize_dynamics_prior_alignment_target(target_output.latent_state.detach())

                elementwise_loss = (latent_state - target_latent).pow(2).reshape(batch_size, -1).mean(dim=1)
                step_mask = mask_batch[:, step_k]
                step_weight = step_mask * weights
                loss_sum = loss_sum + (elementwise_loss * step_weight).sum()
                weight_sum = weight_sum + step_weight.sum()

                pred_l2_terms.append(latent_state.detach().reshape(batch_size, -1).norm(dim=1))
                target_l2_terms.append(target_latent.detach().reshape(batch_size, -1).norm(dim=1))
        finally:
            if hasattr(self._learn_model, 'set_collect_dynamics_prior_stats'):
                self._learn_model.set_collect_dynamics_prior_stats(False)

        if weight_sum.item() <= 0.0:
            return {}

        alignment_loss = loss_sum / weight_sum.clamp_min(1e-6)
        weighted_alignment_loss = alignment_loss * weight

        self._optimizer.zero_grad()
        weighted_alignment_loss.backward()
        if self._cfg.multi_gpu:
            self.sync_gradients(self._learn_model)
        total_grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(
            self._learn_model.parameters(), self._cfg.grad_clip_value
        )
        self._optimizer.step()
        self._target_model.update(self._learn_model.state_dict())
        if self._cfg.use_rnd_model:
            self._target_model_for_intrinsic_reward.update(self._learn_model.state_dict())

        pred_l2 = torch.stack(pred_l2_terms, dim=1)
        target_l2 = torch.stack(target_l2_terms, dim=1)
        log = {
            'dynamics_prior_alignment/loss': float(alignment_loss.item()),
            'dynamics_prior_alignment/weighted_loss': float(weighted_alignment_loss.item()),
            'dynamics_prior_alignment/grad_norm_before_clip': float(total_grad_norm_before_clip.item()),
            'dynamics_prior_alignment/pred_l2_mean': float(pred_l2.mean().item()),
            'dynamics_prior_alignment/target_l2_mean': float(target_l2.mean().item()),
            'dynamics_prior_alignment/valid_weight_sum': float(weight_sum.item()),
        }
        if hasattr(self._learn_model, 'get_dynamics_prior_stats'):
            for key, value in self._learn_model.get_dynamics_prior_stats(reset=True).items():
                log[f'dynamics_prior_alignment/{key}'] = float(value)
        return log

    def _forward_learn(self, data):
        profile_timing = bool(self._cfg.get('profile_learn_timing', False))
        profile_stats = {}
        model_profile_stats = {}
        wrapper_start = _profile_start(profile_timing, self._cfg.device)

        phase_start = _profile_start(profile_timing, self._cfg.device)
        intrinsic_target_log, intrinsic_reward_targets, intrinsic_value_targets = (
            self._prepare_intrinsic_targets_and_train(data)
        )
        profile_stats['profile/learn_intrinsic_targets_s'] = _profile_elapsed(
            phase_start, self._cfg.device
        )
        profile_stats['profile/learn_rnd_s'] = profile_stats[
            'profile/learn_intrinsic_targets_s'
        ]

        phase_start = _profile_start(profile_timing, self._cfg.device)
        expanded_data, original_batch_size, member_ids = self._repeat_batch_for_members(data)
        profile_stats['profile/learn_batch_expand_s'] = _profile_elapsed(phase_start, self._cfg.device)
        profile_stats['profile/learn_original_batch_size'] = original_batch_size
        profile_stats['profile/learn_expanded_batch_size'] = len(member_ids)

        original_cfg_batch_size = self._cfg.batch_size
        phase_start = _profile_start(profile_timing, self._cfg.device)
        self._cfg.batch_size = len(member_ids)
        self._learn_model.set_active_members(member_ids)
        self._learn_model.set_value_prior_learning_mode(True)
        self._pending_intrinsic_replay_loss = {
            'data': data,
            'reward_targets': intrinsic_reward_targets,
            'value_targets': intrinsic_value_targets,
            'member_ids': member_ids,
            'original_batch_size': original_batch_size,
        }
        self._rnd_landscape_query_has_backward = False
        if profile_timing and hasattr(self._learn_model, 'set_profile_timing'):
            self._learn_model.reset_profile_stats()
            self._learn_model.set_profile_timing(True)
        profile_stats['profile/learn_set_active_s'] = _profile_elapsed(phase_start, self._cfg.device)

        restore_s = 0.0
        try:
            phase_start = _profile_start(profile_timing, self._cfg.device)
            super_expanded_data = self._strip_selected_intrinsic_rewards_from_train_data(expanded_data)
            return_log_dict = super()._forward_learn(super_expanded_data)
            profile_stats['profile/learn_super_s'] = _profile_elapsed(phase_start, self._cfg.device)
            if profile_timing and hasattr(self._learn_model, 'get_profile_stats'):
                model_profile_stats = self._learn_model.get_profile_stats(reset=True)
            phase_start = _profile_start(profile_timing, self._cfg.device)
            alignment_log = self._train_dynamics_prior_alignment_from_replay_batch(super_expanded_data, member_ids)
            profile_stats['profile/learn_dynamics_prior_alignment_s'] = _profile_elapsed(
                phase_start, self._cfg.device
            )
            return_log_dict.update(alignment_log)
        finally:
            phase_start = _profile_start(profile_timing, self._cfg.device)
            self._learn_model.set_value_prior_learning_mode(False)
            self._learn_model.set_active_member(0)
            self._pending_intrinsic_replay_loss = None
            self._rnd_landscape_query_has_backward = False
            if getattr(self, '_rnd_landscape_query_model', None) is not None:
                self._rnd_landscape_query_model.eval()
            if profile_timing and hasattr(self._learn_model, 'set_profile_timing'):
                self._learn_model.set_profile_timing(False)
            self._cfg.batch_size = original_cfg_batch_size
            restore_s = _profile_elapsed(phase_start, self._cfg.device)

        phase_start = _profile_start(profile_timing, self._cfg.device)
        if 'value_priority_orig' in return_log_dict and len(member_ids) != original_batch_size:
            value_priority = return_log_dict['value_priority_orig']
            value_priority = np.asarray(value_priority).reshape(self._cfg.model.ensemble_size, original_batch_size)
            value_priority = value_priority.mean(axis=0)
            return_log_dict['value_priority_orig'] = value_priority
            return_log_dict['value_priority'] = value_priority.mean().item()
        if 'root_predicted_value_orig' in return_log_dict and len(member_ids) != original_batch_size:
            root_predicted_value = return_log_dict['root_predicted_value_orig']
            root_predicted_value = np.asarray(root_predicted_value).reshape(
                self._cfg.model.ensemble_size, original_batch_size
            )
            root_predicted_value = root_predicted_value.mean(axis=0)
            return_log_dict['root_predicted_value_orig'] = root_predicted_value
            return_log_dict['root_predicted_value'] = root_predicted_value.mean().item()
        return_log_dict.update(self._member_bootstrap_target_diagnostics(data))
        profile_stats['profile/learn_priority_merge_s'] = _profile_elapsed(phase_start, self._cfg.device)
        return_log_dict['ensemble_size'] = float(self._cfg.model.ensemble_size)
        return_log_dict.update(intrinsic_target_log)
        profile_stats['profile/learn_intrinsic_model_s'] = 0.0
        if 'intrinsic/model_loss' in return_log_dict:
            return_log_dict['intrinsic/grad_norm_before_clip'] = return_log_dict[
                'total_grad_norm_before_clip'
            ]

        phase_start = _profile_start(profile_timing, self._cfg.device)
        latent_novelty_log = self._train_latent_novelty_from_replay_batch(data)
        profile_stats['profile/learn_latent_novelty_s'] = _profile_elapsed(phase_start, self._cfg.device)
        return_log_dict.update(latent_novelty_log)

        if profile_timing:
            profile_stats['profile/learn_restore_s'] = restore_s
            profile_stats['profile/learn_total_wrapper_s'] = _profile_elapsed(wrapper_start, self._cfg.device)
            for key, value in model_profile_stats.items():
                profile_stats[f'profile/model_{key}'] = value
            model_total_s = sum(
                value for key, value in model_profile_stats.items()
                if key.endswith('_total_s')
            )
            profile_stats['profile/model_total_s'] = model_total_s
            profile_stats['profile/learn_non_model_super_s'] = max(
                0.0,
                profile_stats.get('profile/learn_super_s', 0.0) - model_total_s
            )
            return_log_dict.update(profile_stats)
        self._learn_update_count = int(getattr(self, '_learn_update_count', 0)) + 1
        return return_log_dict

    def _init_collect(self) -> None:
        self._validate_ensemble_skeleton()
        self._validate_intrinsic_only_training()
        self._init_rnd_state()
        super()._init_collect()
        self._ensure_episodic_sequence_novelty_model()
        self._ensure_spatial_exploration_tree()
        self._ensure_ensemble_value_calibration_state()
        self._collect_env_member_id = {}
        self._collect_env_pure_exploration_episode = {}
        self._collect_env_pure_exploration_queue = {}
        self._collect_env_pure_exploration_sequence = {}
        self._collect_env_pure_exploration_plan_info = {}
        self._collect_env_episodic_history = {}
        self._collect_env_episodic_history_pending = {}
        self._collect_env_episodic_sequence_context = {}
        self._collect_env_episodic_sequence_pending = {}
        self._collect_env_episodic_sequence_seen = {}
        self._collect_env_rnd_episodic_memory = {}
        depth_disagreement_stats = self._ensure_depth_disagreement_stats()
        self._ensemble_shared_tree_mcts_collect = EnsembleSharedTreeMCTS(
            self._cfg,
            self.value_inverse_scalar_transform_handle,
            self.reward_inverse_scalar_transform_handle,
            depth_disagreement_stats=depth_disagreement_stats,
        )
        self._ensemble_intrinsic_tree_mcts_collect = EnsembleSharedTreeMCTS(
            self._cfg,
            self.value_inverse_scalar_transform_handle,
            self.reward_inverse_scalar_transform_handle,
            depth_disagreement_stats=depth_disagreement_stats,
        )

    def _sample_collect_member_id(self) -> int:
        return int(np.random.randint(0, self._cfg.model.ensemble_size))

    def _get_collect_member_ids(self, ready_env_id: List[int]) -> List[int]:
        member_ids = []
        for env_id in ready_env_id:
            env_id = int(env_id)
            if env_id not in self._collect_env_member_id:
                self._collect_env_member_id[env_id] = self._sample_collect_member_id()
            member_ids.append(self._collect_env_member_id[env_id])
        return member_ids

    def _use_shared_tree_collect(self) -> bool:
        return self._cfg.get('ensemble_search_mode', 'thompson') in [
            'mean_shared_tree', 'mean_tree_final_qstd', 'optimistic_shared_tree', 'pessimistic_shared_tree',
            'bounded_optimistic_shared_tree', 'bounded_pessimistic_shared_tree'
        ]

    def _main_tree_collect_search_mode(self) -> str:
        if bool(self._cfg.get('mixed_reward_tree', False)):
            return 'mixed_reward_shared_tree'
        mode = str(self._cfg.get('ensemble_search_mode', 'thompson') or 'thompson')
        if mode == 'mean_tree_final_qstd':
            return 'mean_shared_tree'
        return mode

    def _use_final_qstd_collect(self) -> bool:
        return str(self._cfg.get('ensemble_search_mode', 'thompson') or 'thompson') == 'mean_tree_final_qstd'

    def _use_separate_exploration_tree_collect(self) -> bool:
        return bool(self._cfg.get('separate_exploration_tree', False))

    def _pure_exploration_episode_fraction(self) -> float:
        return min(1.0, max(0.0, float(self._cfg.get('pure_exploration_episode_fraction', 0.0))))

    def _use_pure_exploration_episodes_collect(self) -> bool:
        has_pure_episode_policy = (
            self._use_separate_exploration_tree_collect()
            or (
                bool(self._cfg.get('intrinsic_tree_pure_exploration_episodes', False))
                and self._use_separate_intrinsic_tree_collect()
            )
        )
        return has_pure_episode_policy and self._pure_exploration_episode_fraction() > 0.0

    def _is_pure_exploration_episode(self, env_id: int) -> bool:
        env_id = int(env_id)
        if not self._use_pure_exploration_episodes_collect():
            return False
        if env_id not in self._collect_env_pure_exploration_episode:
            self._collect_env_pure_exploration_episode[env_id] = (
                bool(np.random.rand() < self._pure_exploration_episode_fraction())
            )
            self._collect_env_pure_exploration_queue[env_id] = []
            self._collect_env_pure_exploration_sequence[env_id] = []
            self._collect_env_pure_exploration_plan_info[env_id] = {}
        return bool(self._collect_env_pure_exploration_episode[env_id])

    def _use_separate_intrinsic_tree_collect(self) -> bool:
        if not bool(self._cfg.get('separate_intrinsic_tree', False)):
            return False
        source = self._intrinsic_reward_source()
        if source == 'dynamics_disagreement':
            return (
                bool(self._cfg.get('dynamics_disagreement_exploration', False))
                and float(self._cfg.get('dynamics_disagreement_intrinsic_reward_weight', 0.0)) > 0.0
            )
        if source == 'qstd':
            return (
                bool(self._cfg.get('qstd_exploration', False))
                and float(self._cfg.get('qstd_intrinsic_reward_weight', 0.0)) > 0.0
            )
        return (
            self._rnd_enabled()
            and float(self._cfg.get('rnd_intrinsic_reward_weight', 0.0)) > 0.0
        )

    def _use_intrinsic_tree_only_collect(self) -> bool:
        return self._cfg.get('collection_policy_mode', 'normal') == 'intrinsic_tree_only'

    def _intrinsic_reward_source(self) -> str:
        source = str(self._cfg.get('intrinsic_reward_source', 'rnd') or 'rnd')
        if source not in [
                'rnd', 'dynamics_disagreement', 'qstd', 'state_value_disagreement',
                'episodic_novelty', 'state_density'
        ]:
            raise ValueError(f"Unsupported intrinsic_reward_source: {source}")
        return source

    def _intrinsic_reward_weight(self) -> float:
        if self._intrinsic_reward_source() == 'dynamics_disagreement':
            return float(self._cfg.get('dynamics_disagreement_intrinsic_reward_weight', 0.0))
        if self._intrinsic_reward_source() == 'qstd':
            return float(self._cfg.get('qstd_intrinsic_reward_weight', 0.0))
        return float(self._cfg.get('rnd_intrinsic_reward_weight', 0.0))

    def _separate_intrinsic_tree_final_weight(self) -> float:
        return self._intrinsic_reward_weight()

    def _episodic_novelty_shadow_root_predictions(
            self,
            model,
            latent_state_roots: torch.Tensor,
            legal_actions: List[List[int]],
    ) -> List[Dict[int, dict]]:
        root_predictions: List[Dict[int, dict]] = [dict() for _ in legal_actions]
        if (
                self._intrinsic_reward_source() != 'episodic_novelty'
                or not bool(self._cfg.get('episodic_novelty_diagnostics', False))
                or not bool(self._cfg.get('episodic_novelty_shadow_predictions', True))
        ):
            return root_predictions

        ensemble_size = int(self._cfg.model.ensemble_size)
        latent_batches = []
        action_batch = []
        member_ids = []
        prediction_keys = []
        for root_index, actions in enumerate(legal_actions):
            for action in actions:
                for member_id in range(ensemble_size):
                    latent_batches.append(latent_state_roots[root_index, member_id:member_id + 1])
                    action_batch.append([int(action)])
                    member_ids.append(member_id)
                    prediction_keys.append((root_index, int(action)))
        if not latent_batches:
            return root_predictions

        latents = torch.cat(latent_batches, dim=0)
        actions = torch.as_tensor(action_batch, dtype=torch.long, device=latents.device)
        was_training = model.training
        try:
            model.eval()
            model.set_active_members(member_ids)
            with torch.no_grad():
                next_latents = model.recurrent_inference(
                    latents, actions, member_id=member_ids
                ).latent_state
                predictions = model.intrinsic_reward_from_latent(
                    next_latents, member_id=member_ids
                ).reshape(-1).detach().cpu().numpy()
        finally:
            model.set_active_member(0)
            model.train(was_training)

        grouped_predictions: Dict[Tuple[int, int], List[float]] = {}
        for key, prediction in zip(prediction_keys, predictions):
            grouped_predictions.setdefault(key, []).append(float(prediction))
        for (root_index, action), values in grouped_predictions.items():
            value_array = np.asarray(values, dtype=np.float32)
            root_predictions[root_index][action] = {
                'episodic_novelty_shadow_reward_mean': float(value_array.mean()),
                'episodic_novelty_shadow_reward_std': float(value_array.std()),
                'episodic_novelty_shadow_reward_min': float(value_array.min()),
                'episodic_novelty_shadow_reward_max': float(value_array.max()),
                'episodic_novelty_shadow_reward_members': value_array.tolist(),
            }
        return root_predictions

    def _episodic_novelty_history_shadow_root_predictions(
            self,
            model,
            latent_state_roots: torch.Tensor,
            legal_actions: List[List[int]],
            ready_env_id: Sequence[int],
    ) -> List[Dict[int, dict]]:
        root_predictions: List[Dict[int, dict]] = [dict() for _ in legal_actions]
        if not self._episodic_history_shadow_enabled(model):
            return root_predictions

        ensemble_size = int(self._cfg.model.ensemble_size)
        grouped_predictions: Dict[Tuple[int, int], List[float]] = {}
        for member_id in range(ensemble_size):
            current_latents = []
            actions = []
            histories = []
            keys = []
            for root_index, (env_id, root_actions) in enumerate(zip(ready_env_id, legal_actions)):
                env_id = int(env_id)
                env_history = self._collect_env_episodic_history.get(env_id)
                if env_history is None:
                    member_history = model.initial_episodic_history(
                        1,
                        member_id=member_id,
                        device=latent_state_roots.device,
                        dtype=latent_state_roots.dtype,
                    )
                else:
                    member_history = env_history[member_id:member_id + 1].to(
                        device=latent_state_roots.device,
                        dtype=latent_state_roots.dtype,
                    )
                for action in root_actions:
                    current_latents.append(
                        latent_state_roots[root_index, member_id:member_id + 1]
                    )
                    actions.append(int(action))
                    histories.append(member_history)
                    keys.append((root_index, int(action)))
            if not current_latents:
                continue
            current_latent_batch = torch.cat(current_latents, dim=0)
            action_batch = torch.as_tensor(
                actions, dtype=torch.long, device=current_latent_batch.device
            )
            history_batch = torch.cat(histories, dim=0)
            with torch.no_grad():
                imagined_next_latents = model.recurrent_inference(
                    current_latent_batch,
                    action_batch.unsqueeze(-1),
                    member_id=member_id,
                ).latent_state
                predictions = model.episodic_history_reward(
                    current_latent_batch,
                    action_batch,
                    imagined_next_latents,
                    history_batch,
                    member_id=member_id,
                )
            for row_index, key in enumerate(keys):
                grouped_predictions.setdefault(key, []).append(
                    float(predictions[row_index].reshape(-1)[0].item())
                )

        for (root_index, action), values in grouped_predictions.items():
            value_array = np.asarray(values, dtype=np.float32)
            current_env_history = self._collect_env_episodic_history.get(
                int(ready_env_id[root_index])
            )
            if current_env_history is None:
                current_history_norm = 0.0
            else:
                current_history_norm = float(
                    current_env_history.detach().float().norm(dim=-1).mean().item()
                )
            root_predictions[root_index][action] = {
                'episodic_novelty_history_shadow_reward_mean': float(value_array.mean()),
                'episodic_novelty_history_shadow_reward_std': float(value_array.std()),
                'episodic_novelty_history_shadow_reward_min': float(value_array.min()),
                'episodic_novelty_history_shadow_reward_max': float(value_array.max()),
                'episodic_novelty_history_shadow_reward_members': value_array.tolist(),
                'episodic_novelty_history_state_norm_mean': current_history_norm,
                'episodic_novelty_history_prediction_uses_one_step_only': 1.0,
            }
        return root_predictions

    def _finalize_collection_episodic_histories(
            self,
            model,
            latent_state_roots: torch.Tensor,
            ready_env_id: Sequence[int],
    ) -> None:
        """Advance persistent histories using completed real transitions only."""
        if not self._episodic_history_shadow_enabled(model):
            return
        pending_by_env = self._collect_env_episodic_history_pending
        ensemble_size = int(self._cfg.model.ensemble_size)
        for root_index, env_id in enumerate(ready_env_id):
            env_id = int(env_id)
            pending = pending_by_env.get(env_id)
            if pending is None or 'realized_reward' not in pending:
                continue
            current_latents = pending['current_latents'].to(
                device=latent_state_roots.device,
                dtype=latent_state_roots.dtype,
            )
            next_latents = latent_state_roots[root_index]
            history = self._collect_env_episodic_history.get(env_id)
            if history is None:
                history = torch.cat(
                    [
                        model.initial_episodic_history(
                            1,
                            member_id=member_id,
                            device=latent_state_roots.device,
                            dtype=latent_state_roots.dtype,
                        )
                        for member_id in range(ensemble_size)
                    ],
                    dim=0,
                )
            else:
                history = history.to(
                    device=latent_state_roots.device,
                    dtype=latent_state_roots.dtype,
                )
            action = torch.as_tensor(
                [int(pending['action'])],
                dtype=torch.long,
                device=latent_state_roots.device,
            )
            realized_reward = torch.as_tensor(
                [float(pending['realized_reward'])],
                dtype=latent_state_roots.dtype,
                device=latent_state_roots.device,
            )
            member_histories = []
            with torch.no_grad():
                for member_id in range(ensemble_size):
                    member_histories.append(
                        model.advance_episodic_history(
                            current_latents[member_id:member_id + 1],
                            action,
                            next_latents[member_id:member_id + 1],
                            realized_reward,
                            history[member_id:member_id + 1],
                            member_id=member_id,
                        )
                    )
            self._collect_env_episodic_history[env_id] = torch.cat(
                member_histories, dim=0
            ).detach()
            pending_by_env.pop(env_id, None)

    def _intrinsic_tree_root_latents(self, latent_state_roots: torch.Tensor) -> torch.Tensor:
        if (
                self._intrinsic_reward_source() == 'dynamics_disagreement'
                and bool(self._cfg.get('dynamics_disagreement_shared_root_latent', True))
        ):
            return latent_state_roots[:, 0:1].expand_as(latent_state_roots).contiguous()
        return latent_state_roots

    def _intrinsic_tree_num_simulations(self) -> int:
        num_simulations = self._cfg.get('intrinsic_tree_num_simulations', None)
        if num_simulations is None:
            return int(self._cfg.num_simulations)
        return int(num_simulations)

    def _main_tree_trace_enabled(self, collect_train_iter: Optional[int] = None) -> bool:
        train_iter = int(collect_train_iter if collect_train_iter is not None else getattr(self, 'train_iter', 0))
        return (
            bool(self._cfg.get('main_tree_trace_diagnostics', False))
            and train_iter >= int(self._cfg.get('main_tree_trace_min_train_iter', 0))
        )

    def _main_tree_trace_remaining(self, collect_train_iter: Optional[int] = None) -> int:
        train_iter = int(collect_train_iter if collect_train_iter is not None else getattr(self, 'train_iter', 0))
        max_count = max(0, int(self._cfg.get('main_tree_trace_max_per_train_iter', 4)))
        used_count = int(self._main_tree_trace_counts.get(train_iter, 0))
        return max(0, max_count - used_count)

    def _main_tree_trace_state_matches(self, root_observation: Optional[Any]) -> bool:
        if root_observation is None:
            return True
        observation = root_observation.detach().cpu().numpy() if torch.is_tensor(root_observation) else root_observation
        observation = np.asarray(observation, dtype=np.float32).reshape(-1)
        has_state_filter = any(
            self._cfg.get(key, None) is not None
            for key in [
                'main_tree_trace_min_position',
                'main_tree_trace_max_position',
                'main_tree_trace_min_velocity',
                'main_tree_trace_max_velocity',
            ]
        )
        if not has_state_filter:
            return True
        if observation.size < 2:
            return False
        position = float(observation[0])
        velocity = float(observation[1])
        min_position = self._cfg.get('main_tree_trace_min_position', None)
        max_position = self._cfg.get('main_tree_trace_max_position', None)
        min_velocity = self._cfg.get('main_tree_trace_min_velocity', None)
        max_velocity = self._cfg.get('main_tree_trace_max_velocity', None)
        if min_position is not None and position < float(min_position):
            return False
        if max_position is not None and position > float(max_position):
            return False
        if min_velocity is not None and velocity < float(min_velocity):
            return False
        if max_velocity is not None and velocity > float(max_velocity):
            return False
        return True

    def _maybe_select_main_tree_trace(
            self,
            root_diagnostics: List[dict],
            tree_diagnostics: Optional[dict],
            collect_train_iter: Optional[int] = None,
            root_observation: Optional[Any] = None,
            selected_action: Optional[int] = None,
    ) -> Optional[dict]:
        if (
                tree_diagnostics is None
                or not root_diagnostics
                or self._main_tree_trace_remaining(collect_train_iter) <= 0
                or not self._main_tree_trace_state_matches(root_observation)
        ):
            return None

        selected_action = None if selected_action is None else int(selected_action)
        target_action = int(self._cfg.get('main_tree_trace_selected_action', -1))
        if target_action >= 0 and selected_action != target_action:
            return None

        selected_terms = None
        if selected_action is not None:
            for action_terms in root_diagnostics:
                if int(action_terms.get('action', -1)) == selected_action:
                    selected_terms = action_terms
                    break
        if selected_terms is None:
            selected_terms = max(root_diagnostics, key=lambda item: float(item.get('q_std', 0.0)))

        min_q_std = float(self._cfg.get('main_tree_trace_min_q_std', 5.0))
        if float(selected_terms.get('q_std', 0.0)) < min_q_std:
            return None

        train_iter = int(collect_train_iter if collect_train_iter is not None else getattr(self, 'train_iter', 0))
        self._main_tree_trace_counts[train_iter] = int(self._main_tree_trace_counts.get(train_iter, 0)) + 1
        return {
            'train_iter': train_iter,
            'selected_action': selected_action,
            'selected_root_terms': selected_terms,
            'main_root_diagnostics': root_diagnostics,
            'tree': tree_diagnostics,
        }

    def _intrinsic_tree_trace_enabled(self, collect_train_iter: Optional[int] = None) -> bool:
        train_iter = int(collect_train_iter if collect_train_iter is not None else getattr(self, 'train_iter', 0))
        return (
            self._use_separate_intrinsic_tree_collect()
            and bool(self._cfg.get('intrinsic_tree_trace_diagnostics', False))
            and train_iter >= int(self._cfg.get('intrinsic_tree_trace_min_train_iter', 0))
        )

    def _intrinsic_tree_trace_remaining(self, collect_train_iter: Optional[int] = None) -> int:
        train_iter = int(collect_train_iter if collect_train_iter is not None else getattr(self, 'train_iter', 0))
        max_count = max(0, int(self._cfg.get('intrinsic_tree_trace_max_per_train_iter', 4)))
        used_count = int(self._intrinsic_tree_trace_counts.get(train_iter, 0))
        return max(0, max_count - used_count)

    def _maybe_select_intrinsic_tree_trace(
            self,
            intrinsic_root_diagnostics: List[dict],
            tree_diagnostics: Optional[dict],
            collect_train_iter: Optional[int] = None,
            root_observation: Optional[Any] = None,
    ) -> Optional[dict]:
        if (
                tree_diagnostics is None
                or not intrinsic_root_diagnostics
                or self._intrinsic_tree_trace_remaining(collect_train_iter) <= 0
        ):
            return None

        best_terms = max(
            intrinsic_root_diagnostics,
            key=lambda item: float(item.get('intrinsic_value', item.get('total_score', 0.0))),
        )
        best_action = int(best_terms.get('action', -1))
        target_action = int(self._cfg.get('intrinsic_tree_trace_best_action', 1))
        if target_action >= 0 and best_action != target_action:
            return None

        if root_observation is not None:
            observation = root_observation.detach().cpu().numpy() if torch.is_tensor(root_observation) else root_observation
            observation = np.asarray(observation, dtype=np.float32).reshape(-1)
            has_state_filter = any(
                self._cfg.get(key, None) is not None
                for key in [
                    'intrinsic_tree_trace_min_position',
                    'intrinsic_tree_trace_max_position',
                    'intrinsic_tree_trace_min_velocity',
                    'intrinsic_tree_trace_max_velocity',
                ]
            )
            if has_state_filter:
                if observation.size < 2:
                    return None
                position = float(observation[0])
                velocity = float(observation[1])
                min_position = self._cfg.get('intrinsic_tree_trace_min_position', None)
                max_position = self._cfg.get('intrinsic_tree_trace_max_position', None)
                min_velocity = self._cfg.get('intrinsic_tree_trace_min_velocity', None)
                max_velocity = self._cfg.get('intrinsic_tree_trace_max_velocity', None)
                if min_position is not None and position < float(min_position):
                    return None
                if max_position is not None and position > float(max_position):
                    return None
                if min_velocity is not None and velocity < float(min_velocity):
                    return None
                if max_velocity is not None and velocity > float(max_velocity):
                    return None

        train_iter = int(collect_train_iter if collect_train_iter is not None else getattr(self, 'train_iter', 0))
        self._intrinsic_tree_trace_counts[train_iter] = int(self._intrinsic_tree_trace_counts.get(train_iter, 0)) + 1
        return {
            'train_iter': train_iter,
            'intrinsic_best_action': best_action,
            'intrinsic_root_diagnostics': intrinsic_root_diagnostics,
            'tree': tree_diagnostics,
        }

    @staticmethod
    def _entropy_from_probs(probs: np.ndarray) -> float:
        probs = np.asarray(probs, dtype=np.float64)
        probs = probs[probs > 0]
        if probs.size == 0:
            return 0.0
        return float(-(probs * np.log2(probs)).sum())

    @staticmethod
    def _tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy()

    @staticmethod
    def _selected_intrinsic_reward_from_diagnostics(
            root_action_diagnostics: List[dict],
            selected_action: int,
    ) -> float:
        selected_action = int(selected_action)
        for action_terms in root_action_diagnostics:
            if int(action_terms.get('action', -1)) != selected_action:
                continue
            return float(
                action_terms.get(
                    'intrinsic_tree_intrinsic_immediate_reward',
                    action_terms.get('intrinsic_immediate_reward', action_terms.get('intrinsic_reward', 0.0))
                )
            )
        return 0.0

    def _visit_count_action_distribution(self, visit_counts: List[float]) -> np.ndarray:
        visit_counts_arr = np.asarray(visit_counts, dtype=np.float64)
        if visit_counts_arr.size == 0:
            return np.asarray([], dtype=np.float32)
        temperature = max(1e-6, float(self._collect_mcts_temperature))
        if float(visit_counts_arr.sum()) <= 0.0:
            probs = np.ones_like(visit_counts_arr) / visit_counts_arr.size
        else:
            probs = np.power(visit_counts_arr, 1.0 / temperature)
            prob_sum = float(probs.sum())
            if prob_sum <= 0.0 or not np.isfinite(prob_sum):
                probs = np.ones_like(visit_counts_arr) / visit_counts_arr.size
            else:
                probs = probs / prob_sum
        return probs.astype(np.float32)

    @staticmethod
    def _augment_visit_policy_with_intrinsic_values(
            base_probs: np.ndarray,
            centered_intrinsic_values: np.ndarray,
            intrinsic_weight: float,
            intrinsic_temperature: float,
    ) -> tuple:
        """Add intrinsic values to visit-policy logits without reviving zero-visit actions."""
        base_probs = np.asarray(base_probs, dtype=np.float64)
        centered_intrinsic_values = np.asarray(centered_intrinsic_values, dtype=np.float64)
        if base_probs.shape != centered_intrinsic_values.shape:
            raise ValueError(
                f"base_probs shape {base_probs.shape} does not match intrinsic values "
                f"shape {centered_intrinsic_values.shape}."
            )

        intrinsic_temperature = max(1e-6, float(intrinsic_temperature))
        intrinsic_logits = (
            float(intrinsic_weight) * centered_intrinsic_values / intrinsic_temperature
        )
        supported = base_probs > 0.0
        if not bool(np.any(supported)):
            final_probs = np.ones_like(base_probs) / max(1, base_probs.size)
            return final_probs.astype(np.float32), intrinsic_logits

        if float(intrinsic_weight) == 0.0 or bool(
                np.allclose(intrinsic_logits[supported], intrinsic_logits[supported][0], rtol=0.0, atol=1e-12)
        ):
            return base_probs.astype(np.float32, copy=True), intrinsic_logits

        # Subtracting a constant leaves the softmax unchanged and prevents overflow.
        supported_max = float(np.max(intrinsic_logits[supported]))
        multipliers = np.zeros_like(base_probs)
        multipliers[supported] = np.exp(
            np.clip(intrinsic_logits[supported] - supported_max, -60.0, 0.0)
        )
        final_probs = base_probs * multipliers
        final_prob_sum = float(final_probs.sum())
        if final_prob_sum <= 0.0 or not np.isfinite(final_prob_sum):
            final_probs = base_probs.copy()
        else:
            final_probs = final_probs / final_prob_sum
        return final_probs.astype(np.float32), intrinsic_logits

    @staticmethod
    def _one_hot_action_distribution(legal_actions: List[int], selected_action: int) -> np.ndarray:
        probs = np.zeros(len(legal_actions), dtype=np.float32)
        for idx, action in enumerate(legal_actions):
            if int(action) == int(selected_action):
                probs[idx] = 1.0
                break
        if float(probs.sum()) <= 0.0 and len(probs) > 0:
            probs[:] = 1.0 / len(probs)
        return probs

    @staticmethod
    def _counts_from_probs(probs: np.ndarray, total_count: float) -> List[float]:
        probs = np.asarray(probs, dtype=np.float32)
        return (probs * max(1.0, float(total_count))).astype(np.float32).tolist()

    def _pure_exploration_sequence_candidates(
            self,
            exploration_diagnostics: List[dict],
            sequence_candidates: Optional[List[dict]] = None,
    ) -> List[dict]:
        if sequence_candidates:
            return [dict(candidate) for candidate in sequence_candidates]
        candidates = []
        for action_terms in exploration_diagnostics:
            for sequence_terms in action_terms.get('separate_exploration_tree_top_sequences', []) or []:
                sequence = sequence_terms.get('sequence', [])
                if not sequence:
                    continue
                candidates.append(
                    {
                        'sequence': [int(action) for action in sequence],
                        'q_std': float(sequence_terms.get('q_std', 0.0)),
                        'raw_q_std': float(sequence_terms.get('raw_q_std', sequence_terms.get('q_std', 0.0))),
                        'member_centered_q_std': float(sequence_terms.get('member_centered_q_std', 0.0)),
                        'score_q_std': float(sequence_terms.get('score_q_std', sequence_terms.get('q_std', 0.0))),
                        'mean_q': float(sequence_terms.get('mean_q', 0.0)),
                        'sequence_score': float(
                            sequence_terms.get('sequence_score', sequence_terms.get('q_std', 0.0))
                        ),
                        'unbounded_sequence_score': float(
                            sequence_terms.get(
                                'unbounded_sequence_score',
                                sequence_terms.get('sequence_score', sequence_terms.get('q_std', 0.0)),
                            )
                        ),
                        'optimistic_member_q': float(sequence_terms.get('optimistic_member_q', 0.0)),
                        'optimism_cap_applied': float(sequence_terms.get('optimism_cap_applied', 0.0)),
                        'optimism_cap_reduction': float(sequence_terms.get('optimism_cap_reduction', 0.0)),
                    }
                )
        return candidates

    def _select_pure_exploration_sequence(
            self,
            legal_actions: List[int],
            exploration_diagnostics: List[dict],
            sequence_candidates: Optional[List[dict]] = None,
    ) -> tuple:
        candidates = self._pure_exploration_sequence_candidates(
            exploration_diagnostics, sequence_candidates=sequence_candidates
        )
        if not candidates:
            fallback_action = int(legal_actions[0]) if legal_actions else 0
            probs = self._one_hot_action_distribution(legal_actions, fallback_action)
            return {
                'sequence': [fallback_action],
                'q_std': 0.0,
                'raw_q_std': 0.0,
                'member_centered_q_std': 0.0,
                'score_q_std': 0.0,
                'mean_q': 0.0,
                'sequence_score': 0.0,
                'unbounded_sequence_score': 0.0,
                'optimistic_member_q': 0.0,
                'optimism_cap_applied': 0.0,
                'optimism_cap_reduction': 0.0,
                'candidate_count': 0,
                'selection_probability': 1.0,
            }, probs

        scores = np.asarray(
            [float(candidate.get('sequence_score', candidate.get('q_std', 0.0))) for candidate in candidates],
            dtype=np.float64,
        )
        temperature = float(self._cfg.get('pure_exploration_sequence_temperature', 0.0))
        if temperature <= 0.0:
            candidate_probs = np.zeros_like(scores, dtype=np.float64)
            selected_index = int(np.argmax(scores))
            candidate_probs[selected_index] = 1.0
        else:
            stable_logits = np.clip((scores - np.max(scores)) / max(1e-6, temperature), -60.0, 60.0)
            candidate_probs = np.exp(stable_logits)
            prob_sum = float(candidate_probs.sum())
            if prob_sum <= 0.0 or not np.isfinite(prob_sum):
                candidate_probs = np.ones_like(scores, dtype=np.float64) / max(1, scores.size)
            else:
                candidate_probs = candidate_probs / prob_sum
            selected_index = int(np.random.choice(len(candidates), p=candidate_probs))

        action_probs = np.zeros(len(legal_actions), dtype=np.float64)
        action_to_index = {int(action): idx for idx, action in enumerate(legal_actions)}
        for candidate, probability in zip(candidates, candidate_probs):
            first_action = int(candidate['sequence'][0])
            if first_action in action_to_index:
                action_probs[action_to_index[first_action]] += float(probability)
        if float(action_probs.sum()) <= 0.0 or not np.isfinite(float(action_probs.sum())):
            action_probs[:] = 1.0 / max(1, action_probs.size)
        else:
            action_probs = action_probs / float(action_probs.sum())

        selected = dict(candidates[selected_index])
        selected['candidate_count'] = int(len(candidates))
        selected['selection_probability'] = float(candidate_probs[selected_index])
        return selected, action_probs.astype(np.float32)

    def _pure_exploration_action_for_env(
            self,
            env_id: int,
            legal_actions: List[int],
            exploration_diagnostics: List[dict],
            total_count: float,
            sequence_candidates: Optional[List[dict]] = None,
    ) -> tuple:
        env_id = int(env_id)
        queue = list(self._collect_env_pure_exploration_queue.get(env_id, []))
        commit_steps = max(1, int(self._cfg.get('pure_exploration_commit_steps', 1)))
        if queue:
            action = int(queue.pop(0))
            self._collect_env_pure_exploration_queue[env_id] = queue
            sequence = list(self._collect_env_pure_exploration_sequence.get(env_id, [action]))
            probs = self._one_hot_action_distribution(legal_actions, action)
            selected = dict(
                self._collect_env_pure_exploration_plan_info.get(
                    env_id,
                    {
                        'sequence': sequence,
                        'q_std': 0.0,
                        'raw_q_std': 0.0,
                        'member_centered_q_std': 0.0,
                        'score_q_std': 0.0,
                        'mean_q': 0.0,
                        'sequence_score': 0.0,
                        'unbounded_sequence_score': 0.0,
                        'optimistic_member_q': 0.0,
                        'optimism_cap_applied': 0.0,
                        'optimism_cap_reduction': 0.0,
                        'candidate_count': 0,
                        'selection_probability': 1.0,
                    }
                )
            )
            source = 'commit'
        else:
            selected, probs = self._select_pure_exploration_sequence(
                legal_actions,
                exploration_diagnostics,
                sequence_candidates=sequence_candidates,
            )
            sequence = [int(action) for action in selected.get('sequence', [])]
            if not sequence:
                sequence = [int(legal_actions[int(np.argmax(probs))])] if legal_actions else [0]
            committed_prefix = sequence[:min(commit_steps, len(sequence))]
            action = int(committed_prefix[0])
            self._collect_env_pure_exploration_queue[env_id] = [int(a) for a in committed_prefix[1:]]
            self._collect_env_pure_exploration_sequence[env_id] = sequence
            self._collect_env_pure_exploration_plan_info[env_id] = dict(selected)
            source = 'new_sequence'

        if action not in [int(a) for a in legal_actions]:
            action = int(legal_actions[0]) if legal_actions else action
            probs = self._one_hot_action_distribution(legal_actions, action)

        counts = self._counts_from_probs(probs, total_count)
        info = {
            'pure_exploration_episode': 1.0,
            'pure_exploration_action_source': source,
            'pure_exploration_commit_steps': int(commit_steps),
            'pure_exploration_commit_remaining': int(len(self._collect_env_pure_exploration_queue.get(env_id, []))),
            'pure_exploration_planned_action': int(action),
            'pure_exploration_selected_sequence': [int(a) for a in selected.get('sequence', [])],
            'pure_exploration_selected_sequence_qstd': float(selected.get('q_std', 0.0)),
            'pure_exploration_selected_sequence_raw_qstd': float(
                selected.get('raw_q_std', selected.get('q_std', 0.0))
            ),
            'pure_exploration_selected_sequence_calibrated_qstd': float(
                selected.get('calibrated_q_std', selected.get('q_std', 0.0))
            ),
            'pure_exploration_selected_sequence_member_centered_qstd': float(
                selected.get('member_centered_q_std', 0.0)
            ),
            'pure_exploration_selected_sequence_score_qstd': float(
                selected.get('score_q_std', selected.get('q_std', 0.0))
            ),
            'pure_exploration_selected_sequence_mean_q': float(selected.get('mean_q', 0.0)),
            'pure_exploration_selected_sequence_score': float(selected.get('sequence_score', 0.0)),
            'pure_exploration_selected_sequence_unbounded_score': float(
                selected.get('unbounded_sequence_score', selected.get('sequence_score', 0.0))
            ),
            'pure_exploration_selected_sequence_optimistic_member_q': float(
                selected.get('optimistic_member_q', 0.0)
            ),
            'pure_exploration_selected_sequence_optimism_cap_applied': float(
                selected.get('optimism_cap_applied', 0.0)
            ),
            'pure_exploration_selected_sequence_optimism_cap_reduction': float(
                selected.get('optimism_cap_reduction', 0.0)
            ),
            'pure_exploration_selected_sequence_q_values': list(selected.get('q_values', [])),
            'pure_exploration_selected_sequence_calibrated_q_values': list(
                selected.get('calibrated_q_values', [])
            ),
            'pure_exploration_selected_sequence_reward_return_values': list(
                selected.get('reward_return_values', [])
            ),
            'pure_exploration_selected_sequence_leaf_value_contribution_values': list(
                selected.get('leaf_value_contribution_values', [])
            ),
            'pure_exploration_selected_sequence_calibrated_leaf_value_contribution_values': list(
                selected.get('calibrated_leaf_value_contribution_values', [])
            ),
            'pure_exploration_selected_sequence_leaf_value_values': list(
                selected.get('leaf_value_values', [])
            ),
            'pure_exploration_selected_sequence_member_q_baselines': list(
                selected.get('member_q_baselines', [])
            ),
            'pure_exploration_sequence_temperature': float(
                self._cfg.get('pure_exploration_sequence_temperature', 0.0)
            ),
            'pure_exploration_sequence_candidate_count': int(selected.get('candidate_count', 0)),
            'pure_exploration_sequence_selection_probability': float(
                selected.get('selection_probability', 1.0)
            ),
        }
        return action, probs, counts, info

    def _separate_exploration_tree_action_sequences(self, legal_actions: List[int]) -> List[tuple]:
        horizon = max(1, int(self._cfg.get('separate_exploration_tree_horizon', 3)))
        actions = [int(action) for action in legal_actions]
        if not actions:
            return []
        sequences = list(itertools.product(actions, repeat=horizon))
        max_sequences = int(self._cfg.get('separate_exploration_tree_max_sequences', 256))
        if max_sequences > 0 and len(sequences) > max_sequences:
            indices = np.random.choice(len(sequences), size=max_sequences, replace=False)
            sequences = [sequences[int(index)] for index in sorted(indices.tolist())]
        return sequences

    def _exploration_tree_sequence_top_k(self) -> int:
        return max(1, int(self._cfg.get('separate_exploration_tree_top_k', 3)))

    @staticmethod
    def _prefix_depth_stats(prefix_records: List[dict], top_k: int) -> dict:
        if not prefix_records:
            return {
                'qstd_mean': 0.0,
                'qstd_std': 0.0,
                'qstd_min': 0.0,
                'qstd_max': 0.0,
                'qstd_p90': 0.0,
                'topk_qstd_mean': 0.0,
                'top_sequence': [],
                'top_sequence_qstd': 0.0,
                'prefix_count': 0,
            }
        qstd_values = np.asarray([float(record['q_std']) for record in prefix_records], dtype=np.float64)
        top_indices = np.argsort(qstd_values)[::-1][:max(1, min(top_k, qstd_values.size))]
        top_record = prefix_records[int(top_indices[0])]
        return {
            'qstd_mean': float(np.mean(qstd_values)),
            'qstd_std': float(np.std(qstd_values)),
            'qstd_min': float(np.min(qstd_values)),
            'qstd_max': float(np.max(qstd_values)),
            'qstd_p90': float(np.percentile(qstd_values, 90)),
            'topk_qstd_mean': float(np.mean(qstd_values[top_indices])),
            'top_sequence': [int(action) for action in top_record['sequence']],
            'top_sequence_qstd': float(top_record['q_std']),
            'prefix_count': int(len(prefix_records)),
        }

    def _separate_exploration_tree_diagnostics_for_root(
            self,
            model: torch.nn.Module,
            root_latents: torch.Tensor,
            legal_actions: List[int],
            return_sequence_candidates: bool = False,
    ):
        sequences = self._separate_exploration_tree_action_sequences(legal_actions)
        if not sequences:
            diagnostics = [
                {'action': int(action), 'separate_exploration_tree_enabled': 1.0}
                for action in legal_actions
            ]
            return (diagnostics, []) if return_sequence_candidates else diagnostics

        horizon = len(sequences[0])
        top_k = self._exploration_tree_sequence_top_k()
        sequence_count = len(sequences)
        ensemble_size = int(root_latents.shape[0])
        device = root_latents.device
        discount_factor = float(self._cfg.discount_factor)
        value_calibration_offsets_np = self._value_calibration_offsets(ensemble_size)
        value_calibration_offsets = torch.as_tensor(
            value_calibration_offsets_np, dtype=torch.float32, device=device
        ).reshape(ensemble_size, 1)
        depth_prefix_records = {
            int(action): {depth: [] for depth in range(1, horizon + 1)} for action in legal_actions
        }
        current_prefixes = [tuple()]
        current_latents = root_latents.unsqueeze(1)
        current_reward_returns = torch.zeros((ensemble_size, 1), dtype=torch.float32, device=device)
        final_records_by_sequence = {}

        for depth_index in range(horizon):
            depth = depth_index + 1
            next_prefixes = sorted({tuple(int(action) for action in sequence[:depth]) for sequence in sequences})
            parent_index = {prefix: idx for idx, prefix in enumerate(current_prefixes)}
            parent_indices = torch.as_tensor(
                [parent_index[prefix[:-1]] for prefix in next_prefixes], dtype=torch.long, device=device
            )
            actions_for_prefixes = torch.as_tensor(
                [int(prefix[-1]) for prefix in next_prefixes], dtype=torch.long, device=device
            )
            prefix_count = len(next_prefixes)
            expanded_latents = current_latents.index_select(1, parent_indices)
            flat_latents = expanded_latents.reshape(
                ensemble_size * prefix_count, *root_latents.shape[1:]
            )
            actions_at_depth = actions_for_prefixes.repeat(ensemble_size)
            member_ids = [member_id for member_id in range(ensemble_size) for _ in range(prefix_count)]
            network_output = model.recurrent_inference(flat_latents, actions_at_depth, member_id=member_ids)
            current_latents = network_output.latent_state.detach().reshape(
                ensemble_size, prefix_count, *root_latents.shape[1:]
            )
            rewards = self.reward_inverse_scalar_transform_handle(network_output.reward).detach().reshape(
                ensemble_size, prefix_count
            ).float()
            values = self.value_inverse_scalar_transform_handle(network_output.value).detach().reshape(
                ensemble_size, prefix_count
            ).float()
            current_reward_returns = current_reward_returns.index_select(1, parent_indices)
            current_reward_returns = current_reward_returns + (discount_factor ** depth_index) * rewards
            leaf_contrib = (discount_factor ** (depth_index + 1)) * values
            backed_up_returns = current_reward_returns + leaf_contrib
            calibrated_values = values - value_calibration_offsets
            calibrated_leaf_contrib = (discount_factor ** (depth_index + 1)) * calibrated_values
            calibrated_backed_up_returns = current_reward_returns + calibrated_leaf_contrib

            returns_np = self._tensor_to_numpy(
                backed_up_returns.transpose(0, 1)
            ).astype(np.float32)
            calibrated_returns_np = self._tensor_to_numpy(
                calibrated_backed_up_returns.transpose(0, 1)
            ).astype(np.float32)
            reward_returns_np = self._tensor_to_numpy(
                current_reward_returns.transpose(0, 1)
            ).astype(np.float32)
            leaf_contrib_np = self._tensor_to_numpy(
                leaf_contrib.transpose(0, 1)
            ).astype(np.float32)
            leaf_values_np = self._tensor_to_numpy(
                values.transpose(0, 1)
            ).astype(np.float32)
            calibrated_leaf_contrib_np = self._tensor_to_numpy(
                calibrated_leaf_contrib.transpose(0, 1)
            ).astype(np.float32)
            calibrated_leaf_values_np = self._tensor_to_numpy(
                calibrated_values.transpose(0, 1)
            ).astype(np.float32)
            raw_qstd_np = np.std(returns_np, axis=1).astype(np.float32)
            calibrated_qstd_np = np.std(calibrated_returns_np, axis=1).astype(np.float32)

            for prefix_index, prefix in enumerate(next_prefixes):
                first_action = int(prefix[0])
                record = {
                    'sequence': [int(action) for action in prefix],
                    'q_std': float(calibrated_qstd_np[prefix_index]),
                    'raw_q_std': float(raw_qstd_np[prefix_index]),
                    'calibrated_q_std': float(calibrated_qstd_np[prefix_index]),
                    'mean_q': float(np.mean(returns_np[prefix_index])),
                    'q_values': returns_np[prefix_index].astype(np.float32).tolist(),
                    'calibrated_q_values': calibrated_returns_np[prefix_index].astype(np.float32).tolist(),
                    'reward_return_values': reward_returns_np[prefix_index].astype(np.float32).tolist(),
                    'leaf_value_contribution_values': leaf_contrib_np[prefix_index].astype(np.float32).tolist(),
                    'calibrated_leaf_value_contribution_values': (
                        calibrated_leaf_contrib_np[prefix_index].astype(np.float32).tolist()
                    ),
                    'leaf_value_values': leaf_values_np[prefix_index].astype(np.float32).tolist(),
                    'calibrated_leaf_value_values': (
                        calibrated_leaf_values_np[prefix_index].astype(np.float32).tolist()
                    ),
                    'value_calibration_offsets': value_calibration_offsets_np.astype(np.float32).tolist(),
                }
                depth_prefix_records[first_action][depth].append(record)
                if depth == horizon:
                    final_records_by_sequence[prefix] = record
            current_prefixes = next_prefixes

        final_records = [final_records_by_sequence[tuple(sequence)] for sequence in sequences]
        final_returns_np = np.asarray(
            [record['calibrated_q_values'] for record in final_records], dtype=np.float32
        )
        member_baselines = np.mean(final_returns_np, axis=0)
        member_centered_qstd = np.std(final_returns_np - member_baselines[None, :], axis=1)
        score_mode = str(
            self._cfg.get('separate_exploration_tree_sequence_score_mode', 'qstd') or 'qstd'
        ).lower()
        if score_mode not in ['qstd', 'mean_q_plus_qstd', 'bounded_mean_q_plus_qstd']:
            raise ValueError(f'Unsupported separate exploration sequence score mode: {score_mode}')
        qstd_mode = str(
            self._cfg.get('separate_exploration_tree_sequence_qstd_mode', 'raw') or 'raw'
        ).lower()
        if qstd_mode not in ['raw', 'member_centered']:
            raise ValueError(f'Unsupported separate exploration sequence qstd mode: {qstd_mode}')
        sequence_qstd_beta = float(
            self._cfg.get('separate_exploration_tree_sequence_qstd_beta', 1.0)
        )
        for sequence_index, record in enumerate(final_records):
            raw_qstd = float(record['raw_q_std'])
            calibrated_qstd = float(record['calibrated_q_std'])
            centered_qstd = float(member_centered_qstd[sequence_index])
            score_qstd = centered_qstd if qstd_mode == 'member_centered' else calibrated_qstd
            mean_q = float(record['mean_q'])
            unbounded_sequence_score = (
                score_qstd
                if score_mode == 'qstd'
                else mean_q + sequence_qstd_beta * score_qstd
            )
            optimistic_member_q = float(np.max(np.asarray(record['q_values'], dtype=np.float32)))
            if score_mode == 'bounded_mean_q_plus_qstd':
                sequence_score = min(unbounded_sequence_score, optimistic_member_q)
            else:
                sequence_score = unbounded_sequence_score
            record.update(
                {
                    'raw_q_std': raw_qstd,
                    'calibrated_q_std': calibrated_qstd,
                    'member_centered_q_std': centered_qstd,
                    'score_q_std': float(score_qstd),
                    'sequence_score': float(sequence_score),
                    'unbounded_sequence_score': float(unbounded_sequence_score),
                    'optimistic_member_q': optimistic_member_q,
                    'optimism_cap_applied': float(sequence_score < unbounded_sequence_score),
                    'optimism_cap_reduction': float(unbounded_sequence_score - sequence_score),
                    'member_q_baselines': member_baselines.astype(np.float32).tolist(),
                }
            )

        diagnostics = []
        for action in legal_actions:
            action = int(action)
            action_records = [record for record in final_records if int(record['sequence'][0]) == action]
            action_qstd = np.asarray([record['raw_q_std'] for record in action_records], dtype=np.float32)
            action_calibrated_qstd = np.asarray(
                [record['calibrated_q_std'] for record in action_records], dtype=np.float32
            )
            action_centered_qstd = np.asarray(
                [record['member_centered_q_std'] for record in action_records], dtype=np.float32
            )
            action_scores = np.asarray([record['sequence_score'] for record in action_records], dtype=np.float32)
            if action_records:
                selected_local_indices = np.argsort(action_scores)[::-1][
                    :max(1, min(top_k, len(action_records)))
                ]
                selected_records = [action_records[int(index)] for index in selected_local_indices]
                top_record = selected_records[0]
                raw_score = float(np.mean([record['sequence_score'] for record in selected_records]))
                top_sequences = [dict(record) for record in selected_records]
                qstd_mean = float(np.mean(action_qstd))
                qstd_std = float(np.std(action_qstd))
                qstd_min = float(np.min(action_qstd))
                qstd_max = float(np.max(action_qstd))
                qstd_p90 = float(np.percentile(action_qstd, 90))
                calibrated_qstd_mean = float(np.mean(action_calibrated_qstd))
                calibrated_qstd_max = float(np.max(action_calibrated_qstd))
                calibrated_qstd_p90 = float(np.percentile(action_calibrated_qstd, 90))
                centered_qstd_mean = float(np.mean(action_centered_qstd))
                centered_qstd_max = float(np.max(action_centered_qstd))
                score_mean = float(np.mean(action_scores))
                score_std = float(np.std(action_scores))
                score_min = float(np.min(action_scores))
                score_max = float(np.max(action_scores))
                score_p90 = float(np.percentile(action_scores, 90))
            else:
                top_record = None
                raw_score = 0.0
                top_sequences = []
                qstd_mean = qstd_std = qstd_min = qstd_max = qstd_p90 = 0.0
                calibrated_qstd_mean = calibrated_qstd_max = calibrated_qstd_p90 = 0.0
                centered_qstd_mean = centered_qstd_max = 0.0
                score_mean = score_std = score_min = score_max = score_p90 = 0.0

            terms = {
                'action': action,
                'separate_exploration_tree_enabled': 1.0,
                'separate_exploration_tree_horizon': int(horizon),
                'separate_exploration_tree_top_k': int(top_k),
                'separate_exploration_tree_sequence_count': int(len(action_records)),
                'separate_exploration_tree_total_sequence_count': int(sequence_count),
                'separate_exploration_tree_raw_score': float(raw_score),
                'separate_exploration_tree_sequence_score_mode': score_mode,
                'separate_exploration_tree_sequence_qstd_beta': float(sequence_qstd_beta),
                'separate_exploration_tree_sequence_qstd_mode': qstd_mode,
                'separate_exploration_tree_member_q_baselines': member_baselines.astype(np.float32).tolist(),
                'separate_exploration_tree_qstd_mean': float(qstd_mean),
                'separate_exploration_tree_qstd_std': float(qstd_std),
                'separate_exploration_tree_qstd_min': float(qstd_min),
                'separate_exploration_tree_qstd_max': float(qstd_max),
                'separate_exploration_tree_qstd_p90': float(qstd_p90),
                'separate_exploration_tree_calibrated_qstd_mean': float(calibrated_qstd_mean),
                'separate_exploration_tree_calibrated_qstd_max': float(calibrated_qstd_max),
                'separate_exploration_tree_calibrated_qstd_p90': float(calibrated_qstd_p90),
                'separate_exploration_tree_member_centered_qstd_mean': float(centered_qstd_mean),
                'separate_exploration_tree_member_centered_qstd_max': float(centered_qstd_max),
                'separate_exploration_tree_sequence_score_mean': float(score_mean),
                'separate_exploration_tree_sequence_score_std': float(score_std),
                'separate_exploration_tree_sequence_score_min': float(score_min),
                'separate_exploration_tree_sequence_score_max': float(score_max),
                'separate_exploration_tree_sequence_score_p90': float(score_p90),
                'separate_exploration_tree_top_sequences': top_sequences,
            }
            if top_record is not None:
                terms.update(
                    {
                        'separate_exploration_tree_top_sequence': list(top_record['sequence']),
                        'separate_exploration_tree_top_sequence_qstd': float(top_record['raw_q_std']),
                        'separate_exploration_tree_top_sequence_calibrated_qstd': float(
                            top_record['calibrated_q_std']
                        ),
                        'separate_exploration_tree_top_sequence_member_centered_qstd': float(
                            top_record['member_centered_q_std']
                        ),
                        'separate_exploration_tree_top_sequence_score_qstd': float(top_record['score_q_std']),
                        'separate_exploration_tree_top_sequence_score': float(top_record['sequence_score']),
                        'separate_exploration_tree_top_sequence_unbounded_score': float(
                            top_record['unbounded_sequence_score']
                        ),
                        'separate_exploration_tree_top_sequence_optimistic_member_q': float(
                            top_record['optimistic_member_q']
                        ),
                        'separate_exploration_tree_top_sequence_optimism_cap_applied': float(
                            top_record['optimism_cap_applied']
                        ),
                        'separate_exploration_tree_top_sequence_optimism_cap_reduction': float(
                            top_record['optimism_cap_reduction']
                        ),
                        'separate_exploration_tree_top_sequence_mean_q': float(top_record['mean_q']),
                        'separate_exploration_tree_top_sequence_q_values': list(top_record['q_values']),
                        'separate_exploration_tree_top_sequence_calibrated_q_values': list(
                            top_record['calibrated_q_values']
                        ),
                        'separate_exploration_tree_top_sequence_reward_return_values': list(
                            top_record['reward_return_values']
                        ),
                        'separate_exploration_tree_top_sequence_leaf_value_contribution_values': list(
                            top_record['leaf_value_contribution_values']
                        ),
                        'separate_exploration_tree_top_sequence_calibrated_leaf_value_contribution_values': list(
                            top_record['calibrated_leaf_value_contribution_values']
                        ),
                        'separate_exploration_tree_top_sequence_leaf_value_values': list(
                            top_record['leaf_value_values']
                        ),
                    }
                )
            else:
                terms.update(
                    {
                        'separate_exploration_tree_top_sequence': [],
                        'separate_exploration_tree_top_sequence_qstd': 0.0,
                        'separate_exploration_tree_top_sequence_calibrated_qstd': 0.0,
                        'separate_exploration_tree_top_sequence_member_centered_qstd': 0.0,
                        'separate_exploration_tree_top_sequence_score_qstd': 0.0,
                        'separate_exploration_tree_top_sequence_score': 0.0,
                        'separate_exploration_tree_top_sequence_unbounded_score': 0.0,
                        'separate_exploration_tree_top_sequence_optimistic_member_q': 0.0,
                        'separate_exploration_tree_top_sequence_optimism_cap_applied': 0.0,
                        'separate_exploration_tree_top_sequence_optimism_cap_reduction': 0.0,
                        'separate_exploration_tree_top_sequence_mean_q': 0.0,
                        'separate_exploration_tree_top_sequence_q_values': [],
                        'separate_exploration_tree_top_sequence_calibrated_q_values': [],
                        'separate_exploration_tree_top_sequence_reward_return_values': [],
                        'separate_exploration_tree_top_sequence_leaf_value_contribution_values': [],
                        'separate_exploration_tree_top_sequence_calibrated_leaf_value_contribution_values': [],
                        'separate_exploration_tree_top_sequence_leaf_value_values': [],
                    }
                )
            for depth in range(1, horizon + 1):
                stats = self._prefix_depth_stats(depth_prefix_records[action][depth], top_k)
                prefix_key = f'separate_exploration_tree_depth{depth}'
                terms.update(
                    {
                        f'{prefix_key}_qstd_mean': float(stats['qstd_mean']),
                        f'{prefix_key}_qstd_std': float(stats['qstd_std']),
                        f'{prefix_key}_qstd_min': float(stats['qstd_min']),
                        f'{prefix_key}_qstd_max': float(stats['qstd_max']),
                        f'{prefix_key}_qstd_p90': float(stats['qstd_p90']),
                        f'{prefix_key}_topk_qstd_mean': float(stats['topk_qstd_mean']),
                        f'{prefix_key}_top_sequence': stats['top_sequence'],
                        f'{prefix_key}_top_sequence_qstd': float(stats['top_sequence_qstd']),
                        f'{prefix_key}_prefix_count': int(stats['prefix_count']),
                    }
                )
            diagnostics.append(terms)
        return (diagnostics, final_records) if return_sequence_candidates else diagnostics

    def _normalize_separate_exploration_scores(self, scores: np.ndarray) -> np.ndarray:
        mode = str(self._cfg.get('separate_exploration_tree_score_normalization', 'minmax') or 'minmax')
        scores = np.asarray(scores, dtype=np.float64)
        if scores.size == 0:
            return scores
        if mode == 'none':
            return scores
        if mode == 'center':
            return scores - float(np.mean(scores))
        if mode == 'zscore':
            std = float(np.std(scores))
            if std <= 1e-8:
                return np.zeros_like(scores)
            return (scores - float(np.mean(scores))) / std
        if mode == 'minmax':
            score_min = float(np.min(scores))
            score_max = float(np.max(scores))
            if score_max - score_min <= 1e-8:
                return np.zeros_like(scores)
            return (scores - score_min) / (score_max - score_min)
        raise ValueError(f"Unsupported separate_exploration_tree_score_normalization: {mode}")

    def _merge_separate_exploration_tree_diagnostics(
            self,
            legal_actions: List[int],
            main_diagnostics: List[dict],
            visit_counts: List[float],
            exploration_diagnostics: List[dict],
    ) -> tuple:
        main_by_action = {int(item.get('action', -1)): dict(item) for item in main_diagnostics}
        exploration_by_action = {int(item.get('action', -1)): dict(item) for item in exploration_diagnostics}
        visit_counts_arr = np.asarray(visit_counts, dtype=np.float64)
        if visit_counts_arr.size == 0:
            return main_diagnostics, np.asarray([], dtype=np.float32), []

        temperature_cfg = self._cfg.get('separate_exploration_tree_temperature', None)
        temperature = (
            float(self._collect_mcts_temperature)
            if temperature_cfg is None else float(temperature_cfg)
        )
        temperature = max(1e-6, temperature)
        beta = float(self._cfg.get('separate_exploration_tree_beta', 0.0))
        eps = 1e-8
        if float(visit_counts_arr.sum()) <= 0.0:
            base_logits = np.zeros_like(visit_counts_arr, dtype=np.float64)
        else:
            base_logits = np.log(np.maximum(visit_counts_arr, eps)) / temperature

        raw_scores = np.asarray(
            [
                float(
                    exploration_by_action.get(int(action), {}).get(
                        'separate_exploration_tree_raw_score',
                        0.0,
                    )
                )
                for action in legal_actions
            ],
            dtype=np.float64,
        )
        normalized_scores = self._normalize_separate_exploration_scores(raw_scores)
        exploration_logits = beta * normalized_scores
        final_logits = base_logits + exploration_logits

        final_stable = np.clip(final_logits - np.max(final_logits), -60.0, 60.0)
        final_probs = np.exp(final_stable)
        final_prob_sum = float(final_probs.sum())
        if final_prob_sum <= 0.0 or not np.isfinite(final_prob_sum):
            final_probs = np.ones_like(final_probs) / max(1, final_probs.size)
        else:
            final_probs = final_probs / final_prob_sum

        base_stable = np.clip(base_logits - np.max(base_logits), -60.0, 60.0)
        base_probs = np.exp(base_stable)
        base_prob_sum = float(base_probs.sum())
        if base_prob_sum <= 0.0 or not np.isfinite(base_prob_sum):
            base_probs = np.ones_like(base_probs) / max(1, base_probs.size)
        else:
            base_probs = base_probs / base_prob_sum

        total_count = float(visit_counts_arr.sum())
        if total_count <= 0.0:
            total_count = float(self._cfg.num_simulations)
        final_counts = (final_probs * max(1.0, total_count)).astype(np.float32).tolist()
        base_best_action = int(legal_actions[int(np.argmax(base_probs))]) if len(legal_actions) > 0 else -1
        final_best_action = int(legal_actions[int(np.argmax(final_probs))]) if len(legal_actions) > 0 else -1
        normalization_mode = str(
            self._cfg.get('separate_exploration_tree_score_normalization', 'minmax') or 'minmax'
        )

        enriched = []
        for idx, action in enumerate(legal_actions):
            action = int(action)
            terms = dict(main_by_action.get(action, {'action': action}))
            exploration_terms = dict(exploration_by_action.get(action, {'action': action}))
            terms.update(exploration_terms)
            terms.update(
                {
                    'separate_exploration_tree_enabled': 1.0,
                    'separate_exploration_tree_beta': float(beta),
                    'separate_exploration_tree_temperature': float(temperature),
                    'separate_exploration_tree_score_normalization': normalization_mode,
                    'separate_exploration_tree_visit_count': float(visit_counts_arr[idx]),
                    'separate_exploration_tree_base_logit': float(base_logits[idx]),
                    'separate_exploration_tree_normalized_score': float(normalized_scores[idx]),
                    'separate_exploration_tree_exploration_logit': float(exploration_logits[idx]),
                    'separate_exploration_tree_final_logit': float(final_logits[idx]),
                    'separate_exploration_tree_base_probability': float(base_probs[idx]),
                    'separate_exploration_tree_probability': float(final_probs[idx]),
                    'separate_exploration_tree_base_best_action': int(base_best_action),
                    'separate_exploration_tree_best_action': int(final_best_action),
                    'separate_exploration_tree_changed_argmax': float(base_best_action != final_best_action),
                    'final_action_probability': float(final_probs[idx]),
                    'final_policy_count': float(final_counts[idx]),
                    'final_total_score': float(final_logits[idx]),
                    'total_score': float(final_logits[idx]),
                    'collection_policy_mode': 'separate_exploration_tree',
                }
            )
            enriched.append(terms)
        return enriched, final_probs.astype(np.float32), final_counts

    def _merge_pure_exploration_episode_diagnostics(
            self,
            legal_actions: List[int],
            main_diagnostics: List[dict],
            visit_counts: List[float],
            exploration_diagnostics: List[dict],
            exploration_probs: np.ndarray,
            exploration_counts: List[float],
            pure_info: Dict[str, Any],
    ) -> List[dict]:
        main_by_action = {int(item.get('action', -1)): dict(item) for item in main_diagnostics}
        exploration_by_action = {int(item.get('action', -1)): dict(item) for item in exploration_diagnostics}
        visit_counts_arr = np.asarray(visit_counts, dtype=np.float64)
        if visit_counts_arr.size == 0:
            visit_counts_arr = np.zeros(len(legal_actions), dtype=np.float64)

        temperature_cfg = self._cfg.get('separate_exploration_tree_temperature', None)
        temperature = (
            float(self._collect_mcts_temperature)
            if temperature_cfg is None else float(temperature_cfg)
        )
        temperature = max(1e-6, temperature)
        eps = 1e-8
        if float(visit_counts_arr.sum()) <= 0.0:
            base_logits = np.zeros_like(visit_counts_arr, dtype=np.float64)
        else:
            base_logits = np.log(np.maximum(visit_counts_arr, eps)) / temperature
        base_probs = self._visit_count_action_distribution(visit_counts_arr.tolist())

        raw_scores = np.asarray(
            [
                float(
                    exploration_by_action.get(int(action), {}).get(
                        'separate_exploration_tree_raw_score',
                        0.0,
                    )
                )
                for action in legal_actions
            ],
            dtype=np.float64,
        )
        normalized_scores = self._normalize_separate_exploration_scores(raw_scores)
        exploration_probs = np.asarray(exploration_probs, dtype=np.float64)
        if exploration_probs.size != len(legal_actions):
            exploration_probs = np.ones(len(legal_actions), dtype=np.float64) / max(1, len(legal_actions))
        if float(exploration_probs.sum()) <= 0.0 or not np.isfinite(float(exploration_probs.sum())):
            exploration_probs = np.ones_like(exploration_probs) / max(1, exploration_probs.size)
        else:
            exploration_probs = exploration_probs / float(exploration_probs.sum())

        selected_action = int(pure_info.get('pure_exploration_planned_action', legal_actions[0]))
        base_best_action = int(legal_actions[int(np.argmax(base_probs))]) if len(legal_actions) > 0 else -1
        exploration_best_action = (
            int(legal_actions[int(np.argmax(exploration_probs))]) if len(legal_actions) > 0 else -1
        )
        store_exploration_policy = bool(self._cfg.get('pure_exploration_store_exploration_policy', True))

        enriched = []
        for idx, action in enumerate(legal_actions):
            action = int(action)
            terms = dict(main_by_action.get(action, {'action': action}))
            exploration_terms = dict(exploration_by_action.get(action, {'action': action}))
            terms.update(exploration_terms)
            action_prob = float(exploration_probs[idx])
            action_count = float(exploration_counts[idx]) if idx < len(exploration_counts) else 0.0
            terms.update(
                {
                    **pure_info,
                    'pure_exploration_action_probability': action_prob,
                    'pure_exploration_policy_count': action_count,
                    'pure_exploration_store_exploration_policy': float(store_exploration_policy),
                    'separate_exploration_tree_enabled': 1.0,
                    'separate_exploration_tree_beta': 0.0,
                    'separate_exploration_tree_temperature': float(temperature),
                    'separate_exploration_tree_score_normalization': str(
                        self._cfg.get('separate_exploration_tree_score_normalization', 'minmax') or 'minmax'
                    ),
                    'separate_exploration_tree_visit_count': float(
                        visit_counts_arr[idx] if idx < visit_counts_arr.size else 0.0
                    ),
                    'separate_exploration_tree_base_logit': float(
                        base_logits[idx] if idx < base_logits.size else 0.0
                    ),
                    'separate_exploration_tree_normalized_score': float(normalized_scores[idx]),
                    'separate_exploration_tree_exploration_logit': float(normalized_scores[idx]),
                    'separate_exploration_tree_final_logit': float(np.log(max(action_prob, eps))),
                    'separate_exploration_tree_base_probability': float(
                        base_probs[idx] if idx < base_probs.size else 0.0
                    ),
                    'separate_exploration_tree_probability': action_prob,
                    'separate_exploration_tree_base_best_action': int(base_best_action),
                    'separate_exploration_tree_best_action': int(exploration_best_action),
                    'separate_exploration_tree_changed_argmax': float(
                        base_best_action != exploration_best_action
                    ),
                    'final_action_probability': action_prob,
                    'final_policy_count': action_count,
                    'final_total_score': float(raw_scores[idx]),
                    'total_score': float(raw_scores[idx]),
                    'collection_policy_mode': 'pure_exploration_episode',
                }
            )
            if action == selected_action:
                terms['pure_exploration_selected_action'] = 1.0
            else:
                terms['pure_exploration_selected_action'] = 0.0
            enriched.append(terms)
        return enriched

    def _final_qstd_action_distribution(
            self,
            legal_actions: List[int],
            main_diagnostics: List[dict],
            visit_counts: List[float],
    ) -> tuple:
        diagnostics_by_action = {int(item.get('action', -1)): dict(item) for item in main_diagnostics}
        visit_counts_arr = np.asarray(visit_counts, dtype=np.float64)
        if visit_counts_arr.size == 0:
            return main_diagnostics, np.asarray([], dtype=np.float32), []

        temperature_cfg = self._cfg.get('final_qstd_temperature', None)
        temperature = (
            float(self._collect_mcts_temperature)
            if temperature_cfg is None else float(temperature_cfg)
        )
        temperature = max(1e-6, temperature)
        beta = float(self._cfg.get('final_qstd_beta', 0.0))
        eps = 1e-8

        if float(visit_counts_arr.sum()) <= 0.0:
            base_logits = np.zeros_like(visit_counts_arr, dtype=np.float64)
        else:
            base_logits = np.log(np.maximum(visit_counts_arr, eps)) / temperature

        qstd_values = np.asarray(
            [float(diagnostics_by_action.get(int(action), {}).get('q_std', 0.0)) for action in legal_actions],
            dtype=np.float64,
        )
        qstd_logits = beta * qstd_values
        final_logits = base_logits + qstd_logits
        stable_logits = np.clip(final_logits - np.max(final_logits), -60.0, 60.0)
        final_probs = np.exp(stable_logits)
        final_prob_sum = float(final_probs.sum())
        if final_prob_sum <= 0.0 or not np.isfinite(final_prob_sum):
            final_probs = np.ones_like(final_probs) / max(1, final_probs.size)
        else:
            final_probs = final_probs / final_prob_sum

        base_stable = np.clip(base_logits - np.max(base_logits), -60.0, 60.0)
        base_probs = np.exp(base_stable)
        base_prob_sum = float(base_probs.sum())
        if base_prob_sum <= 0.0 or not np.isfinite(base_prob_sum):
            base_probs = np.ones_like(base_probs) / max(1, base_probs.size)
        else:
            base_probs = base_probs / base_prob_sum

        total_count = float(visit_counts_arr.sum())
        if total_count <= 0.0:
            total_count = float(self._cfg.num_simulations)
        final_counts = (final_probs * max(1.0, total_count)).astype(np.float32).tolist()
        base_best_action = int(legal_actions[int(np.argmax(base_probs))]) if len(legal_actions) > 0 else -1
        final_best_action = int(legal_actions[int(np.argmax(final_probs))]) if len(legal_actions) > 0 else -1

        enriched = []
        for idx, action in enumerate(legal_actions):
            action = int(action)
            terms = dict(diagnostics_by_action.get(action, {'action': action}))
            terms.update(
                {
                    'final_qstd_enabled': 1.0,
                    'final_qstd_beta': float(beta),
                    'final_qstd_temperature': float(temperature),
                    'final_qstd_visit_count': float(visit_counts_arr[idx]),
                    'final_qstd_base_logit': float(base_logits[idx]),
                    'final_qstd_qstd_logit': float(qstd_logits[idx]),
                    'final_qstd_final_logit': float(final_logits[idx]),
                    'final_qstd_base_probability': float(base_probs[idx]),
                    'final_qstd_probability': float(final_probs[idx]),
                    'final_qstd_base_best_action': int(base_best_action),
                    'final_qstd_best_action': int(final_best_action),
                    'final_qstd_changed_argmax': float(base_best_action != final_best_action),
                    'final_action_probability': float(final_probs[idx]),
                    'final_policy_count': float(final_counts[idx]),
                    'final_total_score': float(final_logits[idx]),
                    'total_score': float(final_logits[idx]),
                    'collection_policy_mode': 'mean_tree_final_qstd',
                }
            )
            enriched.append(terms)
        return enriched, final_probs.astype(np.float32), final_counts

    def _merge_separate_intrinsic_tree_diagnostics(
            self,
            legal_actions: List[int],
            main_diagnostics: List[dict],
            intrinsic_diagnostics: List[dict],
            main_visit_counts: List[float],
        ) -> tuple:
        main_by_action = {int(item.get('action', -1)): dict(item) for item in main_diagnostics}
        intrinsic_by_action = {int(item.get('action', -1)): dict(item) for item in intrinsic_diagnostics}
        visit_counts = np.asarray(main_visit_counts, dtype=np.float64)
        if visit_counts.shape != (len(legal_actions),):
            raise ValueError(
                f"Expected {len(legal_actions)} main-tree visit counts, got shape {visit_counts.shape}."
            )

        intrinsic_weight = self._separate_intrinsic_tree_final_weight()
        intrinsic_temperature = max(
            1e-6, float(self._cfg.get('intrinsic_tree_final_score_temperature', 1.0))
        )
        base_temperature = max(1e-6, float(self._collect_mcts_temperature))
        base_probs = self._visit_count_action_distribution(visit_counts.tolist())
        intrinsic_values = np.asarray(
            [
                float(intrinsic_by_action.get(int(action), {}).get('intrinsic_value', 0.0))
                for action in legal_actions
            ],
            dtype=np.float64,
        )
        intrinsic_values = np.nan_to_num(intrinsic_values, nan=0.0, posinf=0.0, neginf=0.0)
        centered_intrinsic_values = intrinsic_values - float(np.mean(intrinsic_values))
        final_probs, intrinsic_logits = self._augment_visit_policy_with_intrinsic_values(
            base_probs,
            centered_intrinsic_values,
            intrinsic_weight,
            intrinsic_temperature,
        )

        eps = 1e-30
        base_logits = np.log(np.maximum(base_probs.astype(np.float64), eps))
        final_logits = base_logits + intrinsic_logits
        total_count = float(visit_counts.sum())
        if total_count <= 0.0:
            total_count = float(self._cfg.num_simulations)
        final_counts = (final_probs * max(1.0, total_count)).astype(np.float32).tolist()
        base_best_action = int(legal_actions[int(np.argmax(base_probs))]) if legal_actions else -1
        intrinsic_best_action = int(legal_actions[int(np.argmax(intrinsic_values))]) if legal_actions else -1
        final_best_action = int(legal_actions[int(np.argmax(final_probs))]) if legal_actions else -1
        base_entropy = self._entropy_from_probs(base_probs)
        final_entropy = self._entropy_from_probs(final_probs)
        policy_total_variation = 0.5 * float(
            np.abs(final_probs.astype(np.float64) - base_probs.astype(np.float64)).sum()
        )

        counterfactual_probs = {}
        for beta in self._cfg.get('intrinsic_tree_counterfactual_betas', ()) or ():
            beta = float(beta)
            beta_probs, _ = self._augment_visit_policy_with_intrinsic_values(
                base_probs,
                centered_intrinsic_values,
                beta,
                intrinsic_temperature,
            )
            counterfactual_probs[beta] = beta_probs

        merged = []

        for idx, action in enumerate(legal_actions):
            action = int(action)
            main_terms = dict(main_by_action.get(action, {'action': action}))
            intrinsic_terms = intrinsic_by_action.get(action, {})
            main_total_score = float(main_terms.get('total_score', 0.0))
            intrinsic_value = float(intrinsic_values[idx])
            final_intrinsic_score = float(intrinsic_logits[idx])
            final_total_score = float(final_logits[idx])

            main_terms.update(
                {
                    'main_tree_visit_count': int(visit_counts[idx]),
                    'main_tree_total_score': main_total_score,
                    'main_tree_puct_total_score': main_total_score,
                    'main_tree_intrinsic_value': float(main_terms.get('intrinsic_value', 0.0)),
                    'main_tree_intrinsic_score': float(main_terms.get('intrinsic_score', 0.0)),
                    'intrinsic_tree_visit_count': int(intrinsic_terms.get('visit_count', 0)),
                    'intrinsic_tree_prior': float(intrinsic_terms.get('prior', 0.0)),
                    'intrinsic_tree_prior_score': float(intrinsic_terms.get('prior_score', 0.0)),
                    'intrinsic_tree_total_score': float(intrinsic_terms.get('total_score', 0.0)),
                    'intrinsic_tree_intrinsic_value': intrinsic_value,
                    'intrinsic_tree_normalized_intrinsic_value': float(
                        intrinsic_terms.get('normalized_intrinsic_value', intrinsic_value)
                    ),
                    'intrinsic_tree_intrinsic_score': float(intrinsic_terms.get('intrinsic_score', 0.0)),
                    'intrinsic_tree_intrinsic_immediate_reward': float(
                        intrinsic_terms.get(
                            'intrinsic_immediate_reward',
                            intrinsic_terms.get('intrinsic_reward', 0.0)
                        )
                    ),
                    'intrinsic_tree_latent_dd_immediate': float(
                        intrinsic_terms.get('dynamics_disagreement_latent_reward', 0.0)
                    ),
                    'intrinsic_tree_projection_dd_immediate': float(
                        intrinsic_terms.get('dynamics_disagreement_projection_reward', 0.0)
                    ),
                    'intrinsic_tree_prediction_projection_dd_immediate': float(
                        intrinsic_terms.get('dynamics_disagreement_prediction_projection_reward', 0.0)
                    ),
                    'intrinsic_tree_intrinsic_reward_return_mean': float(
                        intrinsic_terms.get('intrinsic_reward_return_mean', 0.0)
                    ),
                    'intrinsic_tree_intrinsic_reward_weighted_return_mean': float(
                        intrinsic_terms.get('intrinsic_reward_weighted_return_mean', 0.0)
                    ),
                    'intrinsic_tree_intrinsic_leaf_value_raw_contribution_mean': float(
                        intrinsic_terms.get('intrinsic_leaf_value_raw_contribution_mean', 0.0)
                    ),
                    'intrinsic_tree_intrinsic_leaf_value_clipped_contribution_mean': float(
                        intrinsic_terms.get('intrinsic_leaf_value_clipped_contribution_mean', 0.0)
                    ),
                    'intrinsic_tree_intrinsic_leaf_value_contribution_mean': float(
                        intrinsic_terms.get('intrinsic_leaf_value_contribution_mean', 0.0)
                    ),
                    'intrinsic_tree_intrinsic_leaf_value_before_modifier_contribution_mean': float(
                        intrinsic_terms.get(
                            'intrinsic_leaf_value_before_modifier_contribution_mean', 0.0
                        )
                    ),
                    'intrinsic_tree_intrinsic_leaf_value_modifier_mean': float(
                        intrinsic_terms.get('intrinsic_leaf_value_modifier_mean', 1.0)
                    ),
                    'intrinsic_tree_immediate_reward_weight': float(
                        intrinsic_terms.get(
                            'intrinsic_immediate_reward_weight',
                            self._cfg.get('intrinsic_tree_immediate_reward_weight', 1.0)
                        )
                    ),
                    'intrinsic_tree_leaf_value_weight': float(
                        intrinsic_terms.get(
                            'intrinsic_leaf_value_weight',
                            self._cfg.get('intrinsic_tree_leaf_value_weight', 1.0)
                        )
                    ),
                    'intrinsic_tree_leaf_value_source': str(
                        intrinsic_terms.get(
                            'intrinsic_tree_leaf_value_source',
                            self._cfg.get('intrinsic_tree_leaf_value_source', 'default')
                        )
                    ),
                    'intrinsic_tree_dynamics_disagreement_space': str(
                        intrinsic_terms.get(
                            'dynamics_disagreement_space',
                            self._cfg.get('dynamics_disagreement_space', 'latent')
                        )
                    ),
                    'intrinsic_tree_mean_q': float(intrinsic_terms.get('mean_q', 0.0)),
                    'intrinsic_tree_q_std': float(intrinsic_terms.get('q_std', 0.0)),
                    'intrinsic_tree_predicted_reward_return_mean': float(
                        intrinsic_terms.get('predicted_reward_return_mean', 0.0)
                    ),
                    'intrinsic_tree_leaf_value_contribution_mean': float(
                        intrinsic_terms.get('leaf_value_contribution_mean', 0.0)
                    ),
                    'separate_intrinsic_tree_enabled': 1.0,
                    'collection_policy_mode': self._cfg.get('collection_policy_mode', 'normal'),
                    'intrinsic_reward_source': self._intrinsic_reward_source(),
                    'intrinsic_value': intrinsic_value,
                    'intrinsic_score': final_intrinsic_score,
                    'final_intrinsic_score': final_intrinsic_score,
                    'final_total_score': final_total_score,
                    'total_score': final_total_score,
                    'intrinsic_tree_final_policy_formula': 'visit_logits_plus_centered_intrinsic_q',
                    'intrinsic_tree_final_weight': float(intrinsic_weight),
                    'intrinsic_tree_base_temperature': float(base_temperature),
                    'intrinsic_tree_intrinsic_temperature': float(intrinsic_temperature),
                    'intrinsic_tree_base_logit': float(base_logits[idx]),
                    'intrinsic_tree_centered_intrinsic_value': float(centered_intrinsic_values[idx]),
                    'intrinsic_tree_final_intrinsic_logit': float(intrinsic_logits[idx]),
                    'intrinsic_tree_final_logit': float(final_logits[idx]),
                    'intrinsic_tree_base_probability': float(base_probs[idx]),
                    'intrinsic_tree_final_probability': float(final_probs[idx]),
                    'intrinsic_tree_base_best_action': int(base_best_action),
                    'intrinsic_tree_intrinsic_best_action': int(intrinsic_best_action),
                    'intrinsic_tree_final_best_action': int(final_best_action),
                    'intrinsic_tree_changed_argmax': float(base_best_action != final_best_action),
                    'intrinsic_tree_base_entropy': float(base_entropy),
                    'intrinsic_tree_final_entropy': float(final_entropy),
                    'intrinsic_tree_policy_total_variation': float(policy_total_variation),
                    'final_action_probability': float(final_probs[idx]),
                    'final_policy_count': float(final_counts[idx]),
                }
            )
            for beta, beta_probs in counterfactual_probs.items():
                beta_label = f'{beta:g}'.replace('-', 'm').replace('.', 'p')
                main_terms[
                    f'intrinsic_tree_counterfactual_probability_beta_{beta_label}'
                ] = float(beta_probs[idx])
            merged.append(main_terms)
        return merged, final_probs, final_counts

    def _intrinsic_tree_episode_policy_diagnostics(
            self,
            legal_actions: List[int],
            main_diagnostics: List[dict],
            intrinsic_diagnostics: List[dict],
            main_visit_counts: List[float],
            intrinsic_visit_counts: List[float],
            pure_intrinsic_episode: bool,
    ) -> tuple:
        """Choose either the main or intrinsic visit policy without blending them."""
        merged, _, _ = self._merge_separate_intrinsic_tree_diagnostics(
            legal_actions,
            main_diagnostics,
            intrinsic_diagnostics,
            main_visit_counts,
        )
        main_probs = self._visit_count_action_distribution(main_visit_counts)
        intrinsic_probs = self._visit_count_action_distribution(intrinsic_visit_counts)
        final_probs = intrinsic_probs if pure_intrinsic_episode else main_probs
        action_visit_counts = intrinsic_visit_counts if pure_intrinsic_episode else main_visit_counts
        final_counts = np.asarray(action_visit_counts, dtype=np.float32).tolist()
        base_best_action = int(legal_actions[int(np.argmax(main_probs))]) if legal_actions else -1
        final_best_action = int(legal_actions[int(np.argmax(final_probs))]) if legal_actions else -1
        policy_total_variation = 0.5 * float(
            np.abs(final_probs.astype(np.float64) - main_probs.astype(np.float64)).sum()
        )
        policy_mode = (
            'intrinsic_tree_pure_exploration_episode'
            if pure_intrinsic_episode else 'normal'
        )
        formula = 'intrinsic_tree_visit_policy' if pure_intrinsic_episode else 'main_tree_visit_policy'
        action_policy_source = 'intrinsic_tree' if pure_intrinsic_episode else 'main_tree'
        policy_target_source = (
            'intrinsic_tree'
            if pure_intrinsic_episode
            and bool(self._cfg.get('intrinsic_tree_pure_episode_store_intrinsic_policy', False))
            else 'main_tree'
        )
        final_entropy = self._entropy_from_probs(final_probs)
        final_logits = np.log(np.maximum(final_probs.astype(np.float64), 1e-30))
        coverage_complete = float(
            bool(intrinsic_visit_counts)
            and all(float(count) > 0.0 for count in intrinsic_visit_counts)
        )

        enriched = []
        for idx, action_terms in enumerate(merged):
            terms = dict(action_terms)
            terms.update(
                {
                    'collection_policy_mode': policy_mode,
                    'pure_exploration_episode': float(pure_intrinsic_episode),
                    'intrinsic_tree_action_policy_source': action_policy_source,
                    'intrinsic_tree_policy_target_source': policy_target_source,
                    'intrinsic_tree_final_policy_formula': formula,
                    'intrinsic_tree_final_weight': 0.0,
                    'intrinsic_tree_prior_mode': str(
                        self._cfg.get('intrinsic_tree_prior_mode', 'learned')
                    ),
                    'intrinsic_tree_force_root_action_coverage': float(
                        bool(self._cfg.get('intrinsic_tree_force_root_action_coverage', False))
                    ),
                    'intrinsic_tree_root_forced_coverage_complete': coverage_complete,
                    'intrinsic_tree_final_logit': float(final_logits[idx]),
                    'intrinsic_tree_final_probability': float(final_probs[idx]),
                    'intrinsic_tree_final_best_action': int(final_best_action),
                    'intrinsic_tree_changed_argmax': float(base_best_action != final_best_action),
                    'intrinsic_tree_final_entropy': float(final_entropy),
                    'intrinsic_tree_policy_total_variation': float(policy_total_variation),
                    'final_intrinsic_score': 0.0,
                    'final_action_probability': float(final_probs[idx]),
                    'final_policy_count': float(final_counts[idx]),
                    'final_total_score': float(final_logits[idx]),
                    'total_score': float(final_logits[idx]),
                }
            )
            enriched.append(terms)
        return enriched, final_probs.astype(np.float32), final_counts

    def _intrinsic_tree_only_diagnostics(
            self,
            legal_actions: List[int],
            intrinsic_diagnostics: List[dict],
            visit_counts: List[float],
    ) -> tuple:
        intrinsic_by_action = {int(item.get('action', -1)): dict(item) for item in intrinsic_diagnostics}
        final_probs = self._visit_count_action_distribution(visit_counts)
        final_counts = np.asarray(visit_counts, dtype=np.float32).tolist()
        merged = []

        for idx, action in enumerate(legal_actions):
            action = int(action)
            intrinsic_terms = dict(intrinsic_by_action.get(action, {'action': action}))
            intrinsic_value = float(intrinsic_terms.get('intrinsic_value', 0.0))
            intrinsic_score = float(intrinsic_terms.get('intrinsic_score', 0.0))
            intrinsic_total_score = float(intrinsic_terms.get('total_score', 0.0))
            intrinsic_terms.update(
                {
                    'main_tree_visit_count': 0,
                    'main_tree_total_score': 0.0,
                    'main_tree_intrinsic_value': 0.0,
                    'main_tree_intrinsic_score': 0.0,
                    'intrinsic_tree_visit_count': int(intrinsic_terms.get('visit_count', 0)),
                    'intrinsic_tree_prior': float(intrinsic_terms.get('prior', 0.0)),
                    'intrinsic_tree_prior_score': float(intrinsic_terms.get('prior_score', 0.0)),
                    'intrinsic_tree_total_score': intrinsic_total_score,
                    'intrinsic_tree_intrinsic_value': intrinsic_value,
                    'intrinsic_tree_normalized_intrinsic_value': float(
                        intrinsic_terms.get('normalized_intrinsic_value', intrinsic_value)
                    ),
                    'intrinsic_tree_intrinsic_score': intrinsic_score,
                    'intrinsic_tree_intrinsic_immediate_reward': float(
                        intrinsic_terms.get(
                            'intrinsic_immediate_reward',
                            intrinsic_terms.get('intrinsic_reward', 0.0)
                        )
                    ),
                    'intrinsic_tree_latent_dd_immediate': float(
                        intrinsic_terms.get('dynamics_disagreement_latent_reward', 0.0)
                    ),
                    'intrinsic_tree_projection_dd_immediate': float(
                        intrinsic_terms.get('dynamics_disagreement_projection_reward', 0.0)
                    ),
                    'intrinsic_tree_prediction_projection_dd_immediate': float(
                        intrinsic_terms.get('dynamics_disagreement_prediction_projection_reward', 0.0)
                    ),
                    'intrinsic_tree_intrinsic_reward_return_mean': float(
                        intrinsic_terms.get('intrinsic_reward_return_mean', 0.0)
                    ),
                    'intrinsic_tree_intrinsic_reward_weighted_return_mean': float(
                        intrinsic_terms.get('intrinsic_reward_weighted_return_mean', 0.0)
                    ),
                    'intrinsic_tree_intrinsic_leaf_value_raw_contribution_mean': float(
                        intrinsic_terms.get('intrinsic_leaf_value_raw_contribution_mean', 0.0)
                    ),
                    'intrinsic_tree_intrinsic_leaf_value_clipped_contribution_mean': float(
                        intrinsic_terms.get('intrinsic_leaf_value_clipped_contribution_mean', 0.0)
                    ),
                    'intrinsic_tree_intrinsic_leaf_value_contribution_mean': float(
                        intrinsic_terms.get('intrinsic_leaf_value_contribution_mean', 0.0)
                    ),
                    'intrinsic_tree_intrinsic_leaf_value_before_modifier_contribution_mean': float(
                        intrinsic_terms.get(
                            'intrinsic_leaf_value_before_modifier_contribution_mean', 0.0
                        )
                    ),
                    'intrinsic_tree_intrinsic_leaf_value_modifier_mean': float(
                        intrinsic_terms.get('intrinsic_leaf_value_modifier_mean', 1.0)
                    ),
                    'intrinsic_tree_immediate_reward_weight': float(
                        intrinsic_terms.get(
                            'intrinsic_immediate_reward_weight',
                            self._cfg.get('intrinsic_tree_immediate_reward_weight', 1.0)
                        )
                    ),
                    'intrinsic_tree_leaf_value_weight': float(
                        intrinsic_terms.get(
                            'intrinsic_leaf_value_weight',
                            self._cfg.get('intrinsic_tree_leaf_value_weight', 1.0)
                        )
                    ),
                    'intrinsic_tree_leaf_value_source': str(
                        intrinsic_terms.get(
                            'intrinsic_tree_leaf_value_source',
                            self._cfg.get('intrinsic_tree_leaf_value_source', 'default')
                        )
                    ),
                    'intrinsic_tree_dynamics_disagreement_space': str(
                        intrinsic_terms.get(
                            'dynamics_disagreement_space',
                            self._cfg.get('dynamics_disagreement_space', 'latent')
                        )
                    ),
                    'intrinsic_tree_mean_q': float(intrinsic_terms.get('mean_q', 0.0)),
                    'intrinsic_tree_q_std': float(intrinsic_terms.get('q_std', 0.0)),
                    'intrinsic_tree_predicted_reward_return_mean': float(
                        intrinsic_terms.get('predicted_reward_return_mean', 0.0)
                    ),
                    'intrinsic_tree_leaf_value_contribution_mean': float(
                        intrinsic_terms.get('leaf_value_contribution_mean', 0.0)
                    ),
                    'separate_intrinsic_tree_enabled': 1.0,
                    'collection_policy_mode': 'intrinsic_tree_only',
                    'intrinsic_reward_source': self._intrinsic_reward_source(),
                    'intrinsic_value': intrinsic_value,
                    'intrinsic_score': intrinsic_score,
                    'final_intrinsic_score': intrinsic_score,
                    'final_total_score': intrinsic_total_score,
                    'total_score': intrinsic_total_score,
                    'final_action_probability': float(final_probs[idx]) if idx < len(final_probs) else 0.0,
                    'final_policy_count': float(final_counts[idx]) if idx < len(final_counts) else 0.0,
                }
            )
            merged.append(intrinsic_terms)
        return merged, final_probs, final_counts

    def _use_shared_tree_eval(self) -> bool:
        return self._cfg.get('ensemble_eval_search_mode', 'member0') in [
            'mean_shared_tree', 'optimistic_shared_tree', 'pessimistic_shared_tree',
            'bounded_optimistic_shared_tree', 'bounded_pessimistic_shared_tree'
        ]

    def _use_intrinsic_tree_eval(self) -> bool:
        return bool(self._cfg.get('intrinsic_tree_eval', False))

    def _resolve_eval_search_options(self) -> Tuple[str, float, bool]:
        eval_mode = self._cfg.get('ensemble_eval_search_mode', 'member0')
        if eval_mode == 'mean_shared_tree':
            mcts_mode = 'mean_shared_tree'
            beta = 0.0
        elif eval_mode in [
                'optimistic_shared_tree', 'pessimistic_shared_tree',
                'bounded_optimistic_shared_tree', 'bounded_pessimistic_shared_tree'
        ]:
            mcts_mode = eval_mode
            beta = self._cfg.get('ensemble_eval_ucb_beta', None)
            if beta is None:
                beta = self._cfg.get('ensemble_ucb_beta', 0.0)
            beta = float(beta)
        else:
            raise ValueError(f"Unsupported ensemble_eval_search_mode: {eval_mode}")

        uncertainty_decay = self._cfg.get('ensemble_eval_uncertainty_decay', None)
        if uncertainty_decay is None:
            uncertainty_decay = self._cfg.get('ensemble_uncertainty_decay', False)
        return mcts_mode, beta, bool(uncertainty_decay)

    def _forward_collect(self, *args, **kwargs):
        if self._use_shared_tree_collect():
            return self._forward_collect_shared_tree(*args, **kwargs)

        data = args[0] if args else kwargs['data']
        ready_env_id = kwargs.get('ready_env_id', None)
        if ready_env_id is None:
            ready_env_id = list(np.arange(data.shape[0]))
        else:
            ready_env_id = list(ready_env_id)
            kwargs['ready_env_id'] = ready_env_id

        member_ids = None
        if self._cfg.thompson_sampling:
            member_ids = self._get_collect_member_ids(ready_env_id)
            self._collect_model.set_active_members(member_ids)
        else:
            self._collect_model.set_active_member(0)

        try:
            output = super()._forward_collect(*args, **kwargs)
        finally:
            self._collect_model.set_active_member(0)

        if member_ids is not None:
            for env_id, member_id in zip(ready_env_id, member_ids):
                output[env_id]['ensemble_member_id'] = member_id
        return output

    def _forward_collect_shared_tree(
            self,
            data: torch.Tensor,
            action_mask: list = None,
            temperature: float = 1,
            to_play: List = [-1],
            epsilon: float = 0.25,
            ready_env_id: np.array = None,
            **kwargs,
    ):
        if self._cfg.model.model_type not in ["conv", "mlp"]:
            raise NotImplementedError("Shared-tree ensemble search currently supports only conv/mlp MuZero models.")

        self._collect_model.eval()
        self._collect_mcts_temperature = temperature
        self.collect_epsilon = epsilon
        active_collect_env_num = data.shape[0]
        if ready_env_id is None:
            ready_env_id = np.arange(active_collect_env_num)
        else:
            ready_env_id = list(ready_env_id)
        output = {i: None for i in ready_env_id}

        with torch.no_grad():
            collect_train_iter = int(kwargs.get('train_iter', getattr(self, 'train_iter', 0)))
            ensemble_size = self._cfg.model.ensemble_size
            member_ids = [
                member_id
                for member_id in range(ensemble_size)
                for _ in range(active_collect_env_num)
            ]
            obs_batch = torch.cat([data for _ in range(ensemble_size)], dim=0)
            network_output = self._collect_model.initial_inference(obs_batch, member_id=member_ids)

            latent_state_roots = network_output.latent_state.detach()
            policy_logits = network_output.policy_logits.detach()
            pred_values = self.value_inverse_scalar_transform_handle(network_output.value).detach()

            latent_state_roots = latent_state_roots.reshape(
                ensemble_size, active_collect_env_num, *latent_state_roots.shape[1:]
            ).transpose(0, 1)
            policy_logits = policy_logits.reshape(ensemble_size, active_collect_env_num, -1).transpose(0, 1)
            pred_values = pred_values.reshape(ensemble_size, active_collect_env_num, -1).transpose(0, 1).squeeze(-1)
            latent_novelty_root_disagreement = self._latent_novelty_root_disagreement(
                self._collect_model, latent_state_roots
            )

            self._finalize_collection_episodic_histories(
                self._collect_model,
                latent_state_roots,
                ready_env_id,
            )
            episodic_sequence_contexts = (
                self._prepare_collection_episodic_sequence_contexts(
                    data,
                    ready_env_id,
                )
                if self._episodic_sequence_novelty_enabled() else []
            )
            spatial_exploration_planner = self._ensure_spatial_exploration_tree()
            spatial_exploration_roots = (
                spatial_exploration_planner.prepare_roots(data, ready_env_id)
                if spatial_exploration_planner is not None else []
            )
            spatial_exploration_reward_mode = (
                self._spatial_exploration_tree_reward_mode()
                if spatial_exploration_planner is not None else 'bonus'
            )
            if (
                    spatial_exploration_planner is not None
                    and not spatial_exploration_planner.active
            ):
                spatial_exploration_reward_mode = 'none'
            spatial_exploration_rnd_estimator = (
                self._spatial_exploration_tree_rnd_estimator()
                if spatial_exploration_planner is not None else 'head'
            )
            spatial_exploration_bonus_fn = (
                spatial_exploration_planner.make_bonus_fn(
                    spatial_exploration_roots,
                    beta=float(
                        self._cfg.get('spatial_exploration_tree_beta', 0.05)
                    ),
                    maximum_depth=int(
                        self._cfg.get('spatial_exploration_tree_max_depth', 5)
                    ),
                )
                if (
                    spatial_exploration_planner is not None
                    and spatial_exploration_reward_mode == 'bonus'
                ) else None
            )
            spatial_exploration_modifier_fn = (
                spatial_exploration_planner.make_modifier_fn(
                    spatial_exploration_roots,
                    minimum_reward_factor=float(
                        self._cfg.get(
                            'spatial_exploration_tree_minimum_reward_factor',
                            0.05,
                        )
                    ),
                    maximum_depth=int(
                        self._cfg.get('spatial_exploration_tree_max_depth', 5)
                    ),
                )
                if (
                    spatial_exploration_planner is not None
                    and spatial_exploration_reward_mode == 'gate'
                ) else None
            )
            rnd_episodic_memories = self._rnd_episodic_memory_contexts(
                self._collect_model,
                data,
                latent_state_roots,
                ready_env_id,
                self._collect_env_rnd_episodic_memory,
                model_version=collect_train_iter,
            )

            legal_actions = [np.nonzero(action_mask[j])[0].tolist() for j in range(active_collect_env_num)]
            episodic_sequence_prefix_cache = (
                self._episodic_sequence_prefix_cache(episodic_sequence_contexts)
                if self._episodic_sequence_novelty_enabled() else None
            )
            episodic_sequence_history_cache = (
                self._episodic_sequence_history_cache(episodic_sequence_contexts)
                if self._episodic_sequence_novelty_enabled() else None
            )
            episodic_sequence_path_probability_tables = (
                self._episodic_sequence_path_probability_tables(
                    episodic_sequence_contexts
                )
                if self._episodic_sequence_novelty_enabled() else None
            )
            episodic_sequence_root_action_predictions = (
                self._episodic_sequence_root_action_predictions(
                    episodic_sequence_contexts,
                    legal_actions,
                    episodic_sequence_path_probability_tables,
                    episodic_sequence_prefix_cache,
                    episodic_sequence_history_cache,
                )
                if self._episodic_sequence_novelty_enabled() else []
            )
            episodic_novelty_shadow_predictions = self._episodic_novelty_shadow_root_predictions(
                self._collect_model, latent_state_roots, legal_actions
            )
            episodic_novelty_history_shadow_predictions = (
                self._episodic_novelty_history_shadow_root_predictions(
                    self._collect_model,
                    latent_state_roots,
                    legal_actions,
                    ready_env_id,
                )
            )
            pure_exploration_episode_flags = [
                self._is_pure_exploration_episode(int(env_id)) for env_id in ready_env_id
            ]
            intrinsic_reward_fn = None
            use_model_intrinsic = False
            use_intrinsic_tree_only = self._use_intrinsic_tree_only_collect()
            use_separate_intrinsic_tree = self._use_separate_intrinsic_tree_collect()
            intrinsic_reward_source = self._intrinsic_reward_source()
            if use_intrinsic_tree_only and not use_separate_intrinsic_tree:
                raise RuntimeError(
                    "collection_policy_mode='intrinsic_tree_only' requires separate_intrinsic_tree=True, "
                    "and an enabled intrinsic source. For RND use rnd_exploration=True and "
                    "rnd_intrinsic_reward_weight > 0; for dynamics disagreement use "
                    "dynamics_disagreement_exploration=True and dynamics_disagreement_intrinsic_reward_weight > 0; "
                    "for qstd use qstd_exploration=True and qstd_intrinsic_reward_weight > 0."
                )
            if (
                    intrinsic_reward_source == 'rnd'
                    and self._rnd_enabled()
                    and float(self._cfg.get('rnd_intrinsic_reward_weight', 0.0)) > 0
                    and not use_separate_intrinsic_tree
            ):
                if self._use_learned_intrinsic_model():
                    use_model_intrinsic = True
                else:
                    self._ensure_rnd_model(
                        latent_state_roots.reshape(active_collect_env_num * ensemble_size, *latent_state_roots.shape[2:])
                    )
                    if self._rnd_model is not None:
                        self._rnd_model.eval()
                        intrinsic_reward_fn = self._rnd_bonus_from_latents
            elif (
                    intrinsic_reward_source == 'qstd'
                    and bool(self._cfg.get('qstd_exploration', False))
                    and float(self._cfg.get('qstd_intrinsic_reward_weight', 0.0)) > 0.0
                    and not use_separate_intrinsic_tree
            ):
                use_model_intrinsic = True
            search_output = None
            return_main_tree_trace = (
                self._main_tree_trace_enabled(collect_train_iter)
                and self._main_tree_trace_remaining(collect_train_iter) > 0
            )
            if not use_intrinsic_tree_only:
                search_output = self._ensemble_shared_tree_mcts_collect.search(
                    model=self._collect_model,
                    root_latent_states=latent_state_roots,
                    root_policy_logits=policy_logits,
                    root_values=pred_values,
                    legal_actions=legal_actions,
                    add_exploration_noise=not self._cfg.collect_with_pure_policy,
                    mode=self._main_tree_collect_search_mode(),
                    intrinsic_reward_fn=intrinsic_reward_fn,
                    use_model_intrinsic=use_model_intrinsic,
                    intrinsic_reward_source=intrinsic_reward_source,
                    intrinsic_reward_weight=self._intrinsic_reward_weight(),
                    return_tree_diagnostics=return_main_tree_trace,
                    tree_diagnostics_max_depth=int(self._cfg.get('main_tree_trace_max_depth', 4)),
                    tree_diagnostics_max_nodes=int(self._cfg.get('main_tree_trace_max_nodes', 128)),
                    value_calibration_offsets=self._value_calibration_offsets(),
                    collect_depth_disagreement_statistics=True,
                    policy_prior_mode=str(
                        self._cfg.get('main_tree_policy_prior_mode', 'learned')
                    ),
                    force_root_action_coverage=bool(
                        self._cfg.get('main_tree_force_root_action_coverage', False)
                    ),
                )
            intrinsic_search_output = None
            if use_separate_intrinsic_tree:
                return_intrinsic_tree_trace = (
                    self._intrinsic_tree_trace_enabled(collect_train_iter)
                    and self._intrinsic_tree_trace_remaining(collect_train_iter) > 0
                )
                intrinsic_reward_fn = None
                use_model_intrinsic = False
                if intrinsic_reward_source == 'qstd':
                    use_model_intrinsic = True
                elif intrinsic_reward_source == 'rnd' and self._use_learned_intrinsic_model():
                    use_model_intrinsic = True
                elif intrinsic_reward_source == 'rnd':
                    self._ensure_rnd_model(
                        latent_state_roots.reshape(active_collect_env_num * ensemble_size, *latent_state_roots.shape[2:])
                    )
                    if self._rnd_model is not None:
                        self._rnd_model.eval()
                        intrinsic_reward_fn = self._rnd_bonus_from_latents
                intrinsic_path_reward_transform_fn = None
                if (
                        spatial_exploration_planner is not None
                        and spatial_exploration_rnd_estimator != 'head'
                ):
                    if intrinsic_reward_source != 'rnd':
                        raise ValueError(
                            'Direct spatial RND estimation requires RND intrinsic rewards.'
                        )
                    root_rnd_inputs = self._select_rnd_observation_features(
                        data.float()
                    )
                    self._ensure_rnd_model(root_rnd_inputs)
                    intrinsic_path_reward_transform_fn = (
                        self._spatial_exploration_rnd_transform_fn(
                            spatial_exploration_planner,
                            spatial_exploration_roots,
                            maximum_depth=int(
                                self._cfg.get(
                                    'spatial_exploration_tree_max_depth', 5
                                )
                            ),
                            estimator=spatial_exploration_rnd_estimator,
                        )
                    )
                rnd_landscape_query_context = None
                intrinsic_path_reward_fn = None
                intrinsic_path_reward_modifier_fn = (
                    self._episodic_sequence_path_reward_modifier_fn(
                        episodic_sequence_contexts,
                        episodic_sequence_path_probability_tables,
                        episodic_sequence_prefix_cache,
                        episodic_sequence_history_cache,
                    )
                )
                if spatial_exploration_modifier_fn is not None:
                    if intrinsic_path_reward_modifier_fn is None:
                        intrinsic_path_reward_modifier_fn = (
                            spatial_exploration_modifier_fn
                        )
                    else:
                        first_modifier_fn = intrinsic_path_reward_modifier_fn
                        second_modifier_fn = spatial_exploration_modifier_fn

                        def combined_intrinsic_path_reward_modifier(
                                root_indices,
                                action_paths,
                                next_latent,
                                first=first_modifier_fn,
                                second=second_modifier_fn,
                        ):
                            return np.asarray(
                                first(root_indices, action_paths, next_latent),
                                dtype=np.float32,
                            ) * np.asarray(
                                second(root_indices, action_paths, next_latent),
                                dtype=np.float32,
                            )

                        intrinsic_path_reward_modifier_fn = (
                            combined_intrinsic_path_reward_modifier
                        )
                intrinsic_leaf_value_modifier_fn = (
                    self._episodic_sequence_leaf_value_modifier_fn(
                        episodic_sequence_contexts
                    )
                )
                if (
                        intrinsic_reward_source == 'rnd'
                        and self._rnd_landscape_query_tree_rewards_enabled()
                ):
                    rnd_landscape_query_context = self._rnd_landscape_query_context(
                        data, self._collect_model, record_diagnostics=True
                    )
                    intrinsic_path_reward_fn = self._rnd_landscape_path_reward_fn(
                        rnd_landscape_query_context,
                        world_model=self._collect_model,
                        episodic_memories=rnd_episodic_memories,
                    )
                intrinsic_search_output = self._ensemble_intrinsic_tree_mcts_collect.search(
                    model=self._collect_model,
                    root_latent_states=self._intrinsic_tree_root_latents(latent_state_roots),
                    root_policy_logits=policy_logits,
                    root_values=pred_values,
                    legal_actions=legal_actions,
                    add_exploration_noise=bool(self._cfg.get('intrinsic_tree_add_exploration_noise', False)),
                    mode='intrinsic_shared_tree',
                    intrinsic_reward_fn=intrinsic_reward_fn,
                    use_model_intrinsic=use_model_intrinsic,
                    intrinsic_reward_source=intrinsic_reward_source,
                    ensemble_ucb_beta=0.0,
                    ensemble_uncertainty_decay=False,
                    num_simulations=self._intrinsic_tree_num_simulations(),
                    intrinsic_reward_weight=1.0,
                    return_tree_diagnostics=return_intrinsic_tree_trace,
                    tree_diagnostics_max_depth=int(self._cfg.get('intrinsic_tree_trace_max_depth', 4)),
                    tree_diagnostics_max_nodes=int(self._cfg.get('intrinsic_tree_trace_max_nodes', 128)),
                    value_calibration_offsets=self._value_calibration_offsets(),
                    policy_prior_mode=str(self._cfg.get('intrinsic_tree_prior_mode', 'learned')),
                    force_root_action_coverage=bool(
                        self._cfg.get('intrinsic_tree_force_root_action_coverage', False)
                    ),
                    intrinsic_path_reward_fn=intrinsic_path_reward_fn,
                    intrinsic_path_reward_transform_fn=(
                        intrinsic_path_reward_transform_fn
                    ),
                    intrinsic_path_reward_modifier_fn=(
                        intrinsic_path_reward_modifier_fn
                    ),
                    intrinsic_path_reward_bonus_fn=spatial_exploration_bonus_fn,
                    intrinsic_leaf_value_modifier_fn=(
                        intrinsic_leaf_value_modifier_fn
                    ),
                )
                self._annotate_rnd_landscape_query_diagnostics(
                    rnd_landscape_query_context,
                    legal_actions,
                    intrinsic_search_output.root_action_diagnostics,
                    path_reward_fn=intrinsic_path_reward_fn,
                )

            separate_exploration_tree_output = None
            separate_exploration_tree_candidates = None
            if (
                    self._use_separate_exploration_tree_collect()
                    and not use_intrinsic_tree_only
                    and not use_separate_intrinsic_tree
            ):
                separate_exploration_tree_output = [None for _ in range(active_collect_env_num)]
                separate_exploration_tree_candidates = [None for _ in range(active_collect_env_num)]
                merge_into_normal_policy = abs(float(self._cfg.get('separate_exploration_tree_beta', 0.0))) > 0.0
                for root_idx, env_id in enumerate(ready_env_id):
                    pure_episode = pure_exploration_episode_flags[root_idx]
                    has_committed_action = bool(
                        self._collect_env_pure_exploration_queue.get(int(env_id), [])
                    )
                    if not merge_into_normal_policy and (not pure_episode or has_committed_action):
                        continue
                    diagnostics, candidates = self._separate_exploration_tree_diagnostics_for_root(
                        self._collect_model,
                        latent_state_roots[root_idx],
                        legal_actions[root_idx],
                        return_sequence_candidates=True,
                    )
                    separate_exploration_tree_output[root_idx] = diagnostics
                    separate_exploration_tree_candidates[root_idx] = candidates

            for i, env_id in enumerate(ready_env_id):
                pure_exploration_episode = pure_exploration_episode_flags[i]
                if use_intrinsic_tree_only:
                    distributions = intrinsic_search_output.distributions[i]
                    value = intrinsic_search_output.values[i]
                    root_action_diagnostics, final_action_probs, final_distributions = (
                        self._intrinsic_tree_only_diagnostics(
                            legal_actions[i],
                            intrinsic_search_output.root_action_diagnostics[i],
                            distributions,
                        )
                    )
                    intrinsic_tree_trace = self._maybe_select_intrinsic_tree_trace(
                        intrinsic_search_output.root_action_diagnostics[i],
                        (
                            intrinsic_search_output.tree_diagnostics[i]
                            if intrinsic_search_output.tree_diagnostics is not None else None
                        ),
                        collect_train_iter,
                        data[i],
                    )
                    intrinsic_root_value = intrinsic_search_output.intrinsic_values[i]
                    output_distributions = final_distributions
                    if self._cfg.eps.eps_greedy_exploration_in_collect:
                        action_index_in_legal_action_set, visit_count_distribution_entropy = select_action(
                            distributions, temperature=self._collect_mcts_temperature, deterministic=True
                        )
                        action = legal_actions[i][action_index_in_legal_action_set]
                        if np.random.rand() < self.collect_epsilon:
                            action = np.random.choice(legal_actions[i])
                    else:
                        action_index_in_legal_action_set, visit_count_distribution_entropy = select_action(
                            distributions, temperature=self._collect_mcts_temperature, deterministic=False
                        )
                        action = legal_actions[i][action_index_in_legal_action_set]
                    action = np.asarray(action, dtype=np.int64)
                    predicted_value = intrinsic_search_output.predicted_values[i]
                    predicted_policy_logits = intrinsic_search_output.shared_policy_logits[i]
                    root_uncertainty = intrinsic_search_output.root_uncertainties[i]
                else:
                    distributions, value = search_output.distributions[i], search_output.values[i]
                    root_action_diagnostics = search_output.root_action_diagnostics[i]
                    intrinsic_root_value = search_output.intrinsic_values[i]
                    predicted_value = search_output.predicted_values[i]
                    predicted_policy_logits = search_output.shared_policy_logits[i]
                    root_uncertainty = search_output.root_uncertainties[i]
                if intrinsic_search_output is not None and not use_intrinsic_tree_only:
                    use_intrinsic_episode_separation = bool(
                        self._cfg.get('intrinsic_tree_pure_exploration_episodes', False)
                    )
                    if use_intrinsic_episode_separation:
                        root_action_diagnostics, final_action_probs, final_distributions = (
                            self._intrinsic_tree_episode_policy_diagnostics(
                                legal_actions[i],
                                root_action_diagnostics,
                                intrinsic_search_output.root_action_diagnostics[i],
                                distributions,
                                intrinsic_search_output.distributions[i],
                                pure_exploration_episode,
                            )
                        )
                    else:
                        root_action_diagnostics, final_action_probs, final_distributions = (
                            self._merge_separate_intrinsic_tree_diagnostics(
                                legal_actions[i],
                                root_action_diagnostics,
                                intrinsic_search_output.root_action_diagnostics[i],
                                distributions,
                            )
                        )
                    intrinsic_tree_trace = self._maybe_select_intrinsic_tree_trace(
                        intrinsic_search_output.root_action_diagnostics[i],
                        (
                            intrinsic_search_output.tree_diagnostics[i]
                            if intrinsic_search_output.tree_diagnostics is not None else None
                        ),
                        collect_train_iter,
                        data[i],
                    )
                    intrinsic_root_value = intrinsic_search_output.intrinsic_values[i]
                    if use_intrinsic_episode_separation:
                        store_intrinsic_policy = (
                            pure_exploration_episode
                            and bool(
                                self._cfg.get(
                                    'intrinsic_tree_pure_episode_store_intrinsic_policy', False
                                )
                            )
                        )
                        output_distributions = final_distributions if store_intrinsic_policy else distributions
                    else:
                        output_distributions = (
                            final_distributions
                            if bool(self._cfg.get('intrinsic_tree_store_final_policy', True)) else distributions
                        )
                    visit_count_distribution_entropy = self._entropy_from_probs(final_action_probs)
                    if self._cfg.eps.eps_greedy_exploration_in_collect:
                        action_index_in_legal_action_set = int(np.argmax(final_action_probs))
                        action = legal_actions[i][action_index_in_legal_action_set]
                        if np.random.rand() < self.collect_epsilon:
                            action = np.random.choice(legal_actions[i])
                    else:
                        action_index_in_legal_action_set = int(
                            np.random.choice(len(legal_actions[i]), p=final_action_probs)
                        )
                        action = legal_actions[i][action_index_in_legal_action_set]
                    action = np.asarray(action, dtype=np.int64)
                elif not use_intrinsic_tree_only:
                    intrinsic_tree_trace = None
                    exploration_diagnostics = (
                        separate_exploration_tree_output[i]
                        if separate_exploration_tree_output is not None
                        and separate_exploration_tree_output[i] is not None else []
                    )
                    exploration_candidates = (
                        separate_exploration_tree_candidates[i]
                        if separate_exploration_tree_candidates is not None
                        and separate_exploration_tree_candidates[i] is not None else []
                    )
                    if pure_exploration_episode or exploration_diagnostics:
                        if pure_exploration_episode:
                            total_count = float(np.sum(np.asarray(distributions, dtype=np.float32)))
                            if total_count <= 0.0:
                                total_count = float(self._cfg.num_simulations)
                            action, final_action_probs, final_distributions, pure_info = (
                                self._pure_exploration_action_for_env(
                                    int(env_id),
                                    legal_actions[i],
                                    exploration_diagnostics,
                                    total_count,
                                    sequence_candidates=exploration_candidates,
                                )
                            )
                            root_action_diagnostics = self._merge_pure_exploration_episode_diagnostics(
                                legal_actions[i],
                                root_action_diagnostics,
                                distributions,
                                exploration_diagnostics,
                                final_action_probs,
                                final_distributions,
                                pure_info,
                            )
                            output_distributions = (
                                final_distributions
                                if bool(self._cfg.get('pure_exploration_store_exploration_policy', True))
                                else distributions
                            )
                            visit_count_distribution_entropy = self._entropy_from_probs(final_action_probs)
                        else:
                            root_action_diagnostics, final_action_probs, final_distributions = (
                                self._merge_separate_exploration_tree_diagnostics(
                                    legal_actions[i],
                                    root_action_diagnostics,
                                    distributions,
                                    exploration_diagnostics,
                                )
                            )
                            output_distributions = (
                                final_distributions
                                if bool(self._cfg.get('separate_exploration_tree_store_final_policy', True))
                                else distributions
                            )
                            visit_count_distribution_entropy = self._entropy_from_probs(final_action_probs)
                            if self._cfg.eps.eps_greedy_exploration_in_collect:
                                action_index_in_legal_action_set = int(np.argmax(final_action_probs))
                                action = legal_actions[i][action_index_in_legal_action_set]
                                if np.random.rand() < self.collect_epsilon:
                                    action = np.random.choice(legal_actions[i])
                            else:
                                action_index_in_legal_action_set = int(
                                    np.random.choice(len(legal_actions[i]), p=final_action_probs)
                                )
                                action = legal_actions[i][action_index_in_legal_action_set]
                    elif self._use_final_qstd_collect():
                        root_action_diagnostics, final_action_probs, final_distributions = (
                            self._final_qstd_action_distribution(
                                legal_actions[i],
                                root_action_diagnostics,
                                distributions,
                            )
                        )
                        output_distributions = (
                            final_distributions
                            if bool(self._cfg.get('final_qstd_store_final_policy', True)) else distributions
                        )
                        visit_count_distribution_entropy = self._entropy_from_probs(final_action_probs)
                        if self._cfg.eps.eps_greedy_exploration_in_collect:
                            action_index_in_legal_action_set = int(np.argmax(final_action_probs))
                            action = legal_actions[i][action_index_in_legal_action_set]
                            if np.random.rand() < self.collect_epsilon:
                                action = np.random.choice(legal_actions[i])
                        else:
                            action_index_in_legal_action_set = int(
                                np.random.choice(len(legal_actions[i]), p=final_action_probs)
                            )
                            action = legal_actions[i][action_index_in_legal_action_set]
                    else:
                        output_distributions = distributions
                        if self._cfg.eps.eps_greedy_exploration_in_collect:
                            action_index_in_legal_action_set, visit_count_distribution_entropy = select_action(
                                distributions, temperature=self._collect_mcts_temperature, deterministic=True
                            )
                            action = np.where(action_mask[i] == 1.0)[0][action_index_in_legal_action_set]
                            if np.random.rand() < self.collect_epsilon:
                                action = np.random.choice(legal_actions[i])
                        else:
                            action_index_in_legal_action_set, visit_count_distribution_entropy = select_action(
                                distributions, temperature=self._collect_mcts_temperature, deterministic=False
                            )
                            action = np.where(action_mask[i] == 1.0)[0][action_index_in_legal_action_set]

                planned_action = int(np.asarray(action).reshape(-1)[0])
                random_action_epsilon = float(self._cfg.get('collect_random_action_epsilon', 0.0))
                random_action_epsilon = min(1.0, max(0.0, random_action_epsilon))
                random_action_overridden = False
                if random_action_epsilon > 0.0 and np.random.rand() < random_action_epsilon:
                    action = np.random.choice(legal_actions[i])
                    random_action_overridden = True
                actual_action = int(np.asarray(action).reshape(-1)[0])
                action = np.asarray(action, dtype=np.int64)

                if spatial_exploration_planner is not None:
                    if intrinsic_search_output is None:
                        raise RuntimeError(
                            'spatial_exploration_tree requires the separate intrinsic tree'
                        )
                    spatial_exploration_planner.commit_actions(
                        [spatial_exploration_roots[i]],
                        [actual_action],
                        [intrinsic_search_output.root_action_diagnostics[i]],
                        planned_actions=[planned_action],
                        beta=float(
                            self._cfg.get('spatial_exploration_tree_beta', 0.05)
                        ),
                        reward_mode=spatial_exploration_reward_mode,
                        minimum_reward_factor=float(
                            self._cfg.get(
                                'spatial_exploration_tree_minimum_reward_factor',
                                0.05,
                            )
                        ),
                    )

                if self._episodic_history_shadow_enabled(self._collect_model):
                    self._collect_env_episodic_history_pending[int(env_id)] = {
                        'current_latents': latent_state_roots[i].detach(),
                        'action': actual_action,
                    }
                if self._episodic_sequence_novelty_enabled():
                    candidate_actions = [int(value) for value in legal_actions[i]]
                    prediction_by_action = (
                        episodic_sequence_root_action_predictions[i]
                    )
                    candidate_probabilities = [
                        float(prediction_by_action[action])
                        for action in candidate_actions
                    ]
                    minimum_factor = float(
                        np.clip(
                            self._cfg.get(
                                'episodic_sequence_novelty_minimum_reward_factor',
                                0.05,
                            ),
                            0.0,
                            1.0,
                        )
                    )
                    candidate_factors = [
                        minimum_factor + (1.0 - minimum_factor) * probability
                        for probability in candidate_probabilities
                    ]
                    intrinsic_terms_by_action = {}
                    if intrinsic_search_output is not None:
                        intrinsic_terms_by_action = {
                            int(terms.get('action', -1)): terms
                            for terms in intrinsic_search_output.root_action_diagnostics[i]
                        }
                    pending_sequence = {
                        'action': actual_action,
                        'planned_action': planned_action,
                        'random_action_overridden': random_action_overridden,
                        'tree_active': self._episodic_sequence_novelty_tree_active(),
                        'trained_phase': int(
                            self._episodic_sequence_novelty_trained_phase
                        ),
                        'legal_actions': candidate_actions,
                        'candidate_probabilities': candidate_probabilities,
                        'candidate_factors': candidate_factors,
                        'intrinsic_visit_counts': [
                            int(
                                intrinsic_terms_by_action.get(action, {}).get(
                                    'visit_count', 0
                                )
                            )
                            for action in candidate_actions
                        ],
                        'intrinsic_total_scores': [
                            float(
                                intrinsic_terms_by_action.get(action, {}).get(
                                    'total_score', float('nan')
                                )
                            )
                            for action in candidate_actions
                        ],
                        'intrinsic_immediate_rewards': [
                            float(
                                intrinsic_terms_by_action.get(action, {}).get(
                                    'intrinsic_immediate_reward', float('nan')
                                )
                            )
                            for action in candidate_actions
                        ],
                    }
                    if actual_action in prediction_by_action:
                        selected_probability = float(
                            prediction_by_action[actual_action]
                        )
                        pending_sequence['predicted_factor'] = float(
                            minimum_factor
                            + (1.0 - minimum_factor) * selected_probability
                        )
                    self._collect_env_episodic_sequence_pending[int(env_id)] = (
                        pending_sequence
                    )

                calibration_offsets = self._value_calibration_offsets()
                root_action_diagnostics = [
                    {
                        **dict(action_terms),
                        **episodic_novelty_shadow_predictions[i].get(
                            int(action_terms.get('action', -1)), {}
                        ),
                        **episodic_novelty_history_shadow_predictions[i].get(
                            int(action_terms.get('action', -1)), {}
                        ),
                        'intrinsic_reward_source': self._intrinsic_reward_source(),
                        'value_calibration_enabled': float(
                            bool(self._cfg.get('ensemble_value_calibration', False))
                        ),
                        'value_calibration_phase': int(
                            getattr(self, '_ensemble_value_calibration_phase', -1)
                        ),
                        'value_calibration_sample_count': int(
                            getattr(self, '_ensemble_value_calibration_sample_count', 0)
                        ),
                        'value_calibration_offsets': calibration_offsets.astype(np.float32).tolist(),
                        'collect_planned_action': planned_action,
                        'collect_actual_action': actual_action,
                        'collect_random_action_epsilon': random_action_epsilon,
                        'collect_random_action_overridden': float(random_action_overridden),
                    }
                    for action_terms in root_action_diagnostics
                ]

                main_tree_trace = None
                if search_output is not None:
                    main_tree_trace = self._maybe_select_main_tree_trace(
                        root_action_diagnostics,
                        (
                            search_output.tree_diagnostics[i]
                            if search_output.tree_diagnostics is not None else None
                        ),
                        collect_train_iter,
                        data[i],
                        actual_action,
                    )

                if latent_novelty_root_disagreement is not None:
                    root_action_diagnostics = [
                        {
                            **dict(action_terms),
                            'latent_novelty_root_disagreement': latent_novelty_root_disagreement[i],
                        }
                        for action_terms in root_action_diagnostics
                    ]

                self._record_rnd_episodic_memory_decision(
                    root_action_diagnostics,
                    selected_action=actual_action,
                    env_id=int(env_id),
                    train_iter=collect_train_iter,
                )
                selected_intrinsic_reward = self._selected_intrinsic_reward_from_diagnostics(
                    root_action_diagnostics,
                    actual_action,
                )

                output[env_id] = {
                    'action': action,
                    'visit_count_distributions': output_distributions,
                    'visit_count_distribution_entropy': visit_count_distribution_entropy,
                    'searched_value': value,
                    'searched_intrinsic_value': intrinsic_root_value,
                    'predicted_value': predicted_value,
                    'predicted_policy_logits': predicted_policy_logits,
                    'ensemble_root_uncertainty': root_uncertainty,
                    'latent_novelty_root_disagreement': (
                        latent_novelty_root_disagreement[i]
                        if latent_novelty_root_disagreement is not None else 0.0
                    ),
                    'ensemble_search_mode': self._cfg.ensemble_search_mode,
                    'separate_intrinsic_tree_enabled': bool(intrinsic_search_output is not None),
                    'separate_exploration_tree_enabled': bool(separate_exploration_tree_output is not None),
                    'collection_policy_mode': (
                        root_action_diagnostics[0].get(
                            'collection_policy_mode',
                            self._cfg.get('collection_policy_mode', 'normal'),
                        )
                        if root_action_diagnostics else self._cfg.get('collection_policy_mode', 'normal')
                    ),
                    'pure_exploration_episode': bool(pure_exploration_episode),
                    'selected_intrinsic_reward': selected_intrinsic_reward,
                    'root_action_diagnostics': root_action_diagnostics,
                    'main_tree_trace': main_tree_trace,
                    'intrinsic_tree_trace': intrinsic_tree_trace,
                }
        return output

    def _init_eval(self) -> None:
        self._validate_ensemble_skeleton()
        self._validate_intrinsic_only_training()
        super()._init_eval()
        self._ensure_ensemble_value_calibration_state()
        self._eval_env_rnd_episodic_memory = {}
        self._ensemble_shared_tree_mcts_eval = EnsembleSharedTreeMCTS(
            self._cfg,
            self.value_inverse_scalar_transform_handle,
            self.reward_inverse_scalar_transform_handle,
            depth_disagreement_stats=self._ensure_depth_disagreement_stats(),
        )
        self._ensemble_intrinsic_tree_mcts_eval = EnsembleSharedTreeMCTS(
            self._cfg,
            self.value_inverse_scalar_transform_handle,
            self.reward_inverse_scalar_transform_handle,
            depth_disagreement_stats=self._ensure_depth_disagreement_stats(),
        )

    def _forward_eval(self, *args, **kwargs):
        if self._use_intrinsic_tree_eval():
            return self._forward_eval_intrinsic_tree(*args, **kwargs)
        if self._use_shared_tree_eval():
            return self._forward_eval_shared_tree(*args, **kwargs)
        self._eval_model.set_active_member(0)
        try:
            return super()._forward_eval(*args, **kwargs)
        finally:
            self._eval_model.set_active_member(0)

    def _forward_eval_intrinsic_tree(
            self,
            data: torch.Tensor,
            action_mask: list,
            to_play: List = [-1],
            ready_env_id: Optional[np.array] = None,
            **kwargs,
    ):
        if not bool(self._cfg.get('separate_intrinsic_tree', False)):
            raise RuntimeError("intrinsic_tree_eval requires separate_intrinsic_tree=True.")
        if self._intrinsic_reward_source() == 'rnd' and not self._use_learned_intrinsic_model():
            raise RuntimeError("Intrinsic-tree evaluation with RND requires the learned intrinsic model.")
        if self._cfg.model.model_type not in ["conv", "mlp"]:
            raise NotImplementedError("Intrinsic-tree eval currently supports only conv/mlp MuZero models.")

        self._eval_model.eval()
        active_eval_env_num = data.shape[0]
        if ready_env_id is None:
            ready_env_id = np.arange(active_eval_env_num)
        else:
            ready_env_id = list(ready_env_id)
        output = {i: None for i in ready_env_id}

        with torch.no_grad():
            ensemble_size = self._cfg.model.ensemble_size
            member_ids = [
                member_id
                for member_id in range(ensemble_size)
                for _ in range(active_eval_env_num)
            ]
            obs_batch = torch.cat([data for _ in range(ensemble_size)], dim=0)
            network_output = self._eval_model.initial_inference(obs_batch, member_id=member_ids)

            latent_state_roots = network_output.latent_state.detach().reshape(
                ensemble_size, active_eval_env_num, *network_output.latent_state.shape[1:]
            ).transpose(0, 1)
            policy_logits = network_output.policy_logits.detach().reshape(
                ensemble_size, active_eval_env_num, -1
            ).transpose(0, 1)
            pred_values = self.value_inverse_scalar_transform_handle(
                network_output.value
            ).detach().reshape(ensemble_size, active_eval_env_num, -1).transpose(0, 1).squeeze(-1)
            legal_actions = [np.nonzero(action_mask[j])[0].tolist() for j in range(active_eval_env_num)]
            if not hasattr(self, '_eval_env_rnd_episodic_memory'):
                self._eval_env_rnd_episodic_memory = {}
            rnd_episodic_memories = self._rnd_episodic_memory_contexts(
                self._eval_model,
                data,
                latent_state_roots,
                ready_env_id,
                self._eval_env_rnd_episodic_memory,
                model_version=int(kwargs.get('train_iter', getattr(self, 'train_iter', 0))),
            )
            rnd_landscape_query_context = None
            intrinsic_path_reward_fn = None
            if (
                    self._intrinsic_reward_source() == 'rnd'
                    and self._rnd_landscape_query_tree_rewards_enabled()
            ):
                rnd_landscape_query_context = self._rnd_landscape_query_context(
                    data, self._eval_model
                )
                intrinsic_path_reward_fn = self._rnd_landscape_path_reward_fn(
                    rnd_landscape_query_context,
                    world_model=self._eval_model,
                    episodic_memories=rnd_episodic_memories,
                )

            search_output = self._ensemble_intrinsic_tree_mcts_eval.search(
                model=self._eval_model,
                root_latent_states=self._intrinsic_tree_root_latents(latent_state_roots),
                root_policy_logits=policy_logits,
                root_values=pred_values,
                legal_actions=legal_actions,
                add_exploration_noise=False,
                mode='intrinsic_shared_tree',
                intrinsic_reward_fn=None,
                use_model_intrinsic=True,
                intrinsic_reward_source=self._intrinsic_reward_source(),
                ensemble_ucb_beta=0.0,
                ensemble_uncertainty_decay=False,
                num_simulations=self._intrinsic_tree_num_simulations(),
                intrinsic_reward_weight=1.0,
                value_calibration_offsets=self._value_calibration_offsets(),
                policy_prior_mode=str(self._cfg.get('intrinsic_tree_prior_mode', 'learned')),
                force_root_action_coverage=bool(
                    self._cfg.get('intrinsic_tree_force_root_action_coverage', False)
                ),
                intrinsic_path_reward_fn=intrinsic_path_reward_fn,
            )
            self._annotate_rnd_landscape_query_diagnostics(
                rnd_landscape_query_context,
                legal_actions,
                search_output.root_action_diagnostics,
                path_reward_fn=intrinsic_path_reward_fn,
            )

            for i, env_id in enumerate(ready_env_id):
                distributions = search_output.distributions[i]
                action_index, visit_count_distribution_entropy = select_action(
                    distributions, temperature=1, deterministic=True
                )
                action = legal_actions[i][action_index]
                output[env_id] = {
                    'action': action,
                    'visit_count_distributions': distributions,
                    'visit_count_distribution_entropy': visit_count_distribution_entropy,
                    'searched_value': search_output.values[i],
                    'searched_intrinsic_value': search_output.intrinsic_values[i],
                    'predicted_value': search_output.predicted_values[i],
                    'predicted_policy_logits': search_output.shared_policy_logits[i],
                    'ensemble_root_uncertainty': search_output.root_uncertainties[i],
                    'ensemble_eval_search_mode': 'intrinsic_tree',
                    'ensemble_eval_ucb_beta': 0.0,
                    'root_action_diagnostics': search_output.root_action_diagnostics[i],
                }
        return output

    def _forward_eval_shared_tree(
            self,
            data: torch.Tensor,
            action_mask: list,
            to_play: List = [-1],
            ready_env_id: Optional[np.array] = None,
            **kwargs,
    ):
        if self._cfg.model.model_type not in ["conv", "mlp"]:
            raise NotImplementedError("Shared-tree ensemble eval currently supports only conv/mlp MuZero models.")

        self._eval_model.eval()
        active_eval_env_num = data.shape[0]
        if ready_env_id is None:
            ready_env_id = np.arange(active_eval_env_num)
        else:
            ready_env_id = list(ready_env_id)
        output = {i: None for i in ready_env_id}

        with torch.no_grad():
            ensemble_size = self._cfg.model.ensemble_size
            member_ids = [
                member_id
                for member_id in range(ensemble_size)
                for _ in range(active_eval_env_num)
            ]
            obs_batch = torch.cat([data for _ in range(ensemble_size)], dim=0)
            network_output = self._eval_model.initial_inference(obs_batch, member_id=member_ids)

            latent_state_roots = network_output.latent_state.detach()
            policy_logits = network_output.policy_logits.detach()
            pred_values = self.value_inverse_scalar_transform_handle(network_output.value).detach()

            latent_state_roots = latent_state_roots.reshape(
                ensemble_size, active_eval_env_num, *latent_state_roots.shape[1:]
            ).transpose(0, 1)
            policy_logits = policy_logits.reshape(ensemble_size, active_eval_env_num, -1).transpose(0, 1)
            pred_values = pred_values.reshape(ensemble_size, active_eval_env_num, -1).transpose(0, 1).squeeze(-1)

            legal_actions = [np.nonzero(action_mask[j])[0].tolist() for j in range(active_eval_env_num)]
            mcts_mode, beta, uncertainty_decay = self._resolve_eval_search_options()
            search_output = self._ensemble_shared_tree_mcts_eval.search(
                model=self._eval_model,
                root_latent_states=latent_state_roots,
                root_policy_logits=policy_logits,
                root_values=pred_values,
                legal_actions=legal_actions,
                add_exploration_noise=False,
                mode=mcts_mode,
                intrinsic_reward_fn=None,
                use_model_intrinsic=False,
                ensemble_ucb_beta=beta,
                ensemble_uncertainty_decay=uncertainty_decay,
                value_calibration_offsets=self._value_calibration_offsets(),
            )

            for i, env_id in enumerate(ready_env_id):
                distributions, value = search_output.distributions[i], search_output.values[i]
                action_index_in_legal_action_set, visit_count_distribution_entropy = select_action(
                    distributions, temperature=1, deterministic=True
                )
                action = np.where(action_mask[i] == 1.0)[0][action_index_in_legal_action_set]

                output[env_id] = {
                    'action': action,
                    'visit_count_distributions': distributions,
                    'visit_count_distribution_entropy': visit_count_distribution_entropy,
                    'searched_value': value,
                    'searched_intrinsic_value': search_output.intrinsic_values[i],
                    'predicted_value': search_output.predicted_values[i],
                    'predicted_policy_logits': search_output.shared_policy_logits[i],
                    'ensemble_root_uncertainty': search_output.root_uncertainties[i],
                    'ensemble_eval_search_mode': self._cfg.ensemble_eval_search_mode,
                    'ensemble_eval_ucb_beta': beta,
                    'root_action_diagnostics': search_output.root_action_diagnostics[i],
                }
        return output

    def _reset_collect(self, data_id=None, task_id: int = None) -> None:
        super()._reset_collect(data_id=data_id, task_id=task_id)
        spatial_tree = getattr(self, '_spatial_exploration_tree', None)
        if spatial_tree is not None:
            spatial_tree.reset(data_id)
        if not hasattr(self, '_collect_env_member_id'):
            return
        exploration_state_names = [
            '_collect_env_pure_exploration_episode',
            '_collect_env_pure_exploration_queue',
            '_collect_env_pure_exploration_sequence',
            '_collect_env_pure_exploration_plan_info',
            '_collect_env_episodic_history',
            '_collect_env_episodic_history_pending',
            '_collect_env_episodic_sequence_context',
            '_collect_env_episodic_sequence_pending',
            '_collect_env_episodic_sequence_seen',
            '_collect_env_rnd_episodic_memory',
        ]
        if data_id is None:
            self._collect_env_member_id.clear()
            for state_name in exploration_state_names:
                state = getattr(self, state_name, None)
                if state is not None:
                    state.clear()
        else:
            for env_id in data_id:
                self._collect_env_member_id.pop(int(env_id), None)
                for state_name in exploration_state_names:
                    state = getattr(self, state_name, None)
                    if state is not None:
                        state.pop(int(env_id), None)

    def _reset_eval(self, data_id=None, task_id: int = None) -> None:
        super()._reset_eval(data_id=data_id, task_id=task_id)
        state = getattr(self, '_eval_env_rnd_episodic_memory', None)
        if state is None:
            return
        if data_id is None:
            state.clear()
        else:
            for env_id in data_id:
                state.pop(int(env_id), None)

    def _state_dict_learn(self):
        state_dict = super()._state_dict_learn()
        self._ensure_ensemble_value_calibration_state()
        state_dict['ensemble_value_calibration'] = self._ensemble_value_calibration_state_dict()
        state_dict['depth_disagreement_stats'] = self._ensure_depth_disagreement_stats().state_dict()
        if self._rnd_enabled() and self._rnd_model is not None:
            state_dict['rnd_model'] = self._rnd_model.state_dict()
            state_dict['rnd_input_shape'] = tuple(self._rnd_input_shape or ())
            state_dict['rnd_model_type'] = (
                'conv' if isinstance(self._rnd_model, _ImageRND) else 'mlp'
            )
            state_dict['rnd_optimizer'] = self._rnd_optimizer.state_dict()
            if self._rnd_target_snapshot_model is not None:
                state_dict['rnd_target_snapshot_model'] = self._rnd_target_snapshot_model.state_dict()
                state_dict['rnd_target_snapshot_last_update'] = self._rnd_target_snapshot_last_update
                state_dict['rnd_learn_update_count'] = self._learn_update_count
            state_dict['rnd_bonus_stats'] = {
                'mean': self._rnd_bonus_mean,
                'var': self._rnd_bonus_var,
                'initialized': self._rnd_bonus_stats_initialized,
            }
            if self._rnd_fresh_pending_inputs is not None:
                state_dict['rnd_fresh_pending_inputs'] = (
                    self._rnd_fresh_pending_inputs.detach().cpu().clone()
                )
        if self._rnd_landscape_query_model is not None:
            state_dict['rnd_landscape_query'] = self._rnd_landscape_query_state_dict()
        if self._rnd_auxiliary_decoder_model is not None:
            state_dict['rnd_auxiliary_decoder'] = {
                'version': 1,
                'latent_shape': self._rnd_auxiliary_decoder_model.latent_shape,
                'model': self._rnd_auxiliary_decoder_model.state_dict(),
                'last_train_stats': dict(
                    self._rnd_auxiliary_decoder_last_train_stats
                ),
            }
        if self._latent_novelty_optimizer is not None:
            state_dict['latent_novelty_optimizer'] = self._latent_novelty_optimizer.state_dict()
        if self._episodic_history_optimizer is not None:
            state_dict['episodic_history_optimizer'] = self._episodic_history_optimizer.state_dict()
        if hasattr(self, '_episodic_history_episode_replay'):
            state_dict['episodic_history_episode_replay'] = self._episodic_history_replay_state_dict()
        episodic_sequence_state = self._episodic_sequence_novelty_state_dict(
            include_optimizer=True
        )
        if episodic_sequence_state:
            state_dict['episodic_sequence_novelty'] = episodic_sequence_state
        spatial_tree_state = self._spatial_exploration_tree_state_dict(
            include_optimizer=True
        )
        if spatial_tree_state:
            state_dict['spatial_exploration_tree_online'] = spatial_tree_state
        return state_dict

    def _spatial_exploration_tree_state_dict(
            self, include_optimizer: bool = False
    ) -> Dict[str, Any]:
        planner = getattr(self, '_spatial_exploration_tree', None)
        if planner is None:
            return {}
        return planner.online_state_dict(include_optimizer=include_optimizer)

    def _load_spatial_exploration_tree_state(
            self, state: Dict[str, Any], load_optimizer: bool = False
    ) -> None:
        if (
                not state
                or not bool(
                    self._cfg.get(
                        'spatial_exploration_tree_online_training', False
                    )
                )
        ):
            return
        planner = self._ensure_spatial_exploration_tree()
        planner.load_online_state_dict(
            state, load_optimizer=load_optimizer
        )

    def _episodic_sequence_novelty_state_dict(
            self, include_optimizer: bool = False
    ) -> Dict[str, Any]:
        model = self._episodic_sequence_novelty_model
        if model is None:
            return {}
        state = {
            'version': 2,
            'model': model.state_dict(),
            'trained_phase': int(self._episodic_sequence_novelty_trained_phase),
            'continuation_trained_phase': int(
                self._episodic_sequence_continuation_trained_phase
            ),
            'last_train_stats': dict(
                self._episodic_sequence_novelty_last_train_stats
            ),
        }
        if include_optimizer and self._episodic_sequence_novelty_optimizer is not None:
            state['optimizer'] = self._episodic_sequence_novelty_optimizer.state_dict()
        return state

    def _load_episodic_sequence_novelty_state(
            self, state: Dict[str, Any], load_optimizer: bool = False
    ) -> None:
        if not state or not self._episodic_sequence_novelty_enabled():
            return
        model = self._ensure_episodic_sequence_novelty_model()
        model.load_state_dict(state.get('model', {}), strict=False)
        self._episodic_sequence_novelty_trained_phase = int(
            state.get('trained_phase', -1)
        )
        self._episodic_sequence_continuation_trained_phase = int(
            state.get('continuation_trained_phase', -1)
        )
        self._episodic_sequence_novelty_last_train_stats = dict(
            state.get('last_train_stats', {})
        )
        if (
                load_optimizer
                and self._episodic_sequence_novelty_optimizer is not None
                and 'optimizer' in state
        ):
            try:
                self._episodic_sequence_novelty_optimizer.load_state_dict(
                    state['optimizer']
                )
            except ValueError:
                pass

    def _rnd_landscape_query_state_dict(self) -> Dict[str, Any]:
        if self._rnd_landscape_query_model is None:
            return {}
        return {
            'version': 6,
            'model': self._rnd_landscape_query_model.state_dict(),
            'snapshot_count': int(self._rnd_landscape_query_snapshot_count),
            'target_scale_phase': int(self._rnd_landscape_query_target_scale_phase),
            'target_scale_sample_count': int(
                self._rnd_landscape_query_target_scale_sample_count
            ),
            'last_train_stats': dict(self._rnd_landscape_query_last_train_stats),
        }

    def _load_rnd_landscape_query_state(self, state: Dict[str, Any]) -> None:
        if not state or not self._rnd_landscape_query_enabled():
            return
        model = self._ensure_rnd_landscape_query_model()
        model.load_state_dict(state.get('model', {}), strict=False)
        replay_version = int(state.get('version', 0))
        self._rnd_landscape_query_snapshot_count = int(
            state.get(
                'snapshot_count',
                len(state.get('snapshots', [])) if replay_version >= 3 else 0,
            )
        )
        self._rnd_landscape_query_target_scale_phase = int(
            state.get('target_scale_phase', -1)
        )
        self._rnd_landscape_query_target_scale_sample_count = int(
            state.get('target_scale_sample_count', 0)
        )
        self._rnd_landscape_query_last_train_stats = dict(
            state.get('last_train_stats', {})
        )

    def _ensemble_value_calibration_state_dict(self) -> Dict[str, Any]:
        self._ensure_ensemble_value_calibration_state()
        return {
            'offsets': self._ensemble_value_calibration_offsets.astype(np.float32).tolist(),
            'collection_phase': int(self._ensemble_value_calibration_phase),
            'sample_count': int(self._ensemble_value_calibration_sample_count),
            'stats': dict(self._ensemble_value_calibration_stats),
        }

    def _load_ensemble_value_calibration_state(self, state_dict: Dict[str, Any]) -> None:
        self._ensure_ensemble_value_calibration_state()
        calibration_state = state_dict.get('ensemble_value_calibration', {})
        saved_offsets = np.asarray(calibration_state.get('offsets', []), dtype=np.float32)
        if saved_offsets.shape != self._ensemble_value_calibration_offsets.shape:
            return
        self._ensemble_value_calibration_offsets = saved_offsets.copy()
        self._ensemble_value_calibration_phase = int(calibration_state.get('collection_phase', -1))
        self._ensemble_value_calibration_sample_count = int(calibration_state.get('sample_count', 0))
        self._ensemble_value_calibration_stats = dict(calibration_state.get('stats', {}))

    def _load_depth_disagreement_stats(self, state_dict: Dict[str, Any]) -> None:
        self._ensure_depth_disagreement_stats().load_state_dict(state_dict.get('depth_disagreement_stats', {}))

    def _state_dict_collect(self):
        state_dict = super()._state_dict_collect()
        state_dict['ensemble_value_calibration'] = self._ensemble_value_calibration_state_dict()
        state_dict['depth_disagreement_stats'] = self._ensure_depth_disagreement_stats().state_dict()
        if self._rnd_landscape_query_model is not None:
            state_dict['rnd_landscape_query'] = self._rnd_landscape_query_state_dict()
        episodic_sequence_state = self._episodic_sequence_novelty_state_dict()
        if episodic_sequence_state:
            state_dict['episodic_sequence_novelty'] = episodic_sequence_state
        spatial_tree_state = self._spatial_exploration_tree_state_dict()
        if spatial_tree_state:
            state_dict['spatial_exploration_tree_online'] = spatial_tree_state
        return state_dict

    def _load_state_dict_collect(self, state_dict):
        super()._load_state_dict_collect(state_dict)
        self._load_ensemble_value_calibration_state(state_dict)
        self._load_depth_disagreement_stats(state_dict)
        self._load_rnd_landscape_query_state(state_dict.get('rnd_landscape_query', {}))
        self._load_episodic_sequence_novelty_state(
            state_dict.get('episodic_sequence_novelty', {})
        )
        self._load_spatial_exploration_tree_state(
            state_dict.get('spatial_exploration_tree_online', {})
        )

    def _state_dict_eval(self):
        state_dict = super()._state_dict_eval()
        state_dict['ensemble_value_calibration'] = self._ensemble_value_calibration_state_dict()
        state_dict['depth_disagreement_stats'] = self._ensure_depth_disagreement_stats().state_dict()
        if self._rnd_landscape_query_model is not None:
            state_dict['rnd_landscape_query'] = self._rnd_landscape_query_state_dict()
        episodic_sequence_state = self._episodic_sequence_novelty_state_dict()
        if episodic_sequence_state:
            state_dict['episodic_sequence_novelty'] = episodic_sequence_state
        spatial_tree_state = self._spatial_exploration_tree_state_dict()
        if spatial_tree_state:
            state_dict['spatial_exploration_tree_online'] = spatial_tree_state
        return state_dict

    def _load_state_dict_eval(self, state_dict):
        super()._load_state_dict_eval(state_dict)
        self._load_ensemble_value_calibration_state(state_dict)
        self._load_depth_disagreement_stats(state_dict)
        self._load_rnd_landscape_query_state(state_dict.get('rnd_landscape_query', {}))
        self._load_episodic_sequence_novelty_state(
            state_dict.get('episodic_sequence_novelty', {})
        )
        self._load_spatial_exploration_tree_state(
            state_dict.get('spatial_exploration_tree_online', {})
        )

    @staticmethod
    def _fill_missing_model_state(saved_state: dict, current_state: dict) -> dict:
        patched_state = dict(saved_state)
        for key, value in current_state.items():
            if key not in patched_state:
                patched_state[key] = value
        return patched_state

    def _load_state_dict_learn(self, state_dict):
        model_state = self._fill_missing_model_state(state_dict['model'], self._learn_model.state_dict())
        target_model_state = self._fill_missing_model_state(
            state_dict['target_model'], self._target_model.state_dict()
        )
        self._learn_model.load_state_dict(model_state, strict=False)
        self._target_model.load_state_dict(target_model_state, strict=False)
        self._load_ensemble_value_calibration_state(state_dict)
        self._load_depth_disagreement_stats(state_dict)
        auxiliary_state = state_dict.get('rnd_auxiliary_decoder', {})
        if auxiliary_state and self._rnd_auxiliary_decoder_enabled():
            latent_shape = tuple(auxiliary_state.get('latent_shape', ()))
            if latent_shape:
                decoder = self._ensure_rnd_auxiliary_decoder_model(latent_shape)
                decoder.load_state_dict(auxiliary_state.get('model', {}), strict=False)
                self._rnd_auxiliary_decoder_last_train_stats = dict(
                    auxiliary_state.get('last_train_stats', {})
                )
        try:
            self._optimizer.load_state_dict(state_dict['optimizer'])
        except ValueError:
            # Older checkpoints do not contain optimizer slots for the newly added intrinsic heads.
            pass
        if self._rnd_enabled() and 'rnd_model' in state_dict:
            rnd_state = state_dict['rnd_model']
            rnd_input_shape = tuple(state_dict.get('rnd_input_shape', ()))
            if not rnd_input_shape:
                first_weight = rnd_state['target.0.weight']
                if first_weight.dim() == 2:
                    rnd_input_shape = (int(first_weight.shape[1]), )
                else:
                    observation_shape = tuple(self._cfg.model.observation_shape)
                    rnd_input_shape = (
                        int(first_weight.shape[1]),
                        int(observation_shape[-2]),
                        int(observation_shape[-1]),
                    )
            configured_model_type = self._cfg.get('rnd_model_type', 'auto')
            if str(configured_model_type).lower() == 'auto':
                self._cfg.rnd_model_type = state_dict.get(
                    'rnd_model_type',
                    'conv' if len(rnd_input_shape) == 3 else 'mlp',
                )
            dummy_input = torch.zeros(
                (1, *rnd_input_shape), device=self._cfg.device
            )
            self._ensure_rnd_model(dummy_input)
            self._rnd_model.load_state_dict(rnd_state)
            if 'rnd_optimizer' in state_dict:
                self._rnd_optimizer.load_state_dict(state_dict['rnd_optimizer'])
            if 'rnd_target_snapshot_model' in state_dict and bool(self._cfg.get('rnd_target_snapshot', True)):
                self._sync_rnd_target_snapshot(force=True)
                self._rnd_target_snapshot_model.load_state_dict(state_dict['rnd_target_snapshot_model'])
                self._rnd_target_snapshot_last_update = int(state_dict.get('rnd_target_snapshot_last_update', 0))
                self._learn_update_count = int(
                    state_dict.get('rnd_learn_update_count', self._rnd_target_snapshot_last_update)
                )
            elif bool(self._cfg.get('rnd_target_snapshot', True)):
                self._sync_rnd_target_snapshot(force=True)
            stats = state_dict.get('rnd_bonus_stats', {})
            self._rnd_bonus_mean = float(stats.get('mean', 0.0))
            self._rnd_bonus_var = float(stats.get('var', 1.0))
            self._rnd_bonus_stats_initialized = bool(stats.get('initialized', False))
            pending_inputs = state_dict.get('rnd_fresh_pending_inputs', None)
            self._rnd_fresh_pending_inputs = (
                None
                if pending_inputs is None
                else torch.as_tensor(pending_inputs, dtype=torch.float32).detach().cpu().clone()
            )
        self._load_rnd_landscape_query_state(state_dict.get('rnd_landscape_query', {}))
        if 'latent_novelty_optimizer' in state_dict:
            latent_novelty_optimizer = self._ensure_latent_novelty_optimizer()
            if latent_novelty_optimizer is not None:
                try:
                    latent_novelty_optimizer.load_state_dict(state_dict['latent_novelty_optimizer'])
                except ValueError:
                    pass
        if 'episodic_history_optimizer' in state_dict:
            episodic_history_optimizer = self._ensure_episodic_history_optimizer()
            if episodic_history_optimizer is not None:
                try:
                    episodic_history_optimizer.load_state_dict(state_dict['episodic_history_optimizer'])
                except ValueError:
                    pass
        if 'episodic_history_episode_replay' in state_dict:
            self._load_episodic_history_replay_state(state_dict['episodic_history_episode_replay'])
        self._load_episodic_sequence_novelty_state(
            state_dict.get('episodic_sequence_novelty', {}),
            load_optimizer=True,
        )
        self._load_spatial_exploration_tree_state(
            state_dict.get('spatial_exploration_tree_online', {}),
            load_optimizer=True,
        )
        self._sync_qstd_target_snapshot(force=True)

    def _monitor_vars_learn(self) -> List[str]:
        monitor_vars = super()._monitor_vars_learn() + [
            'ensemble_size',
            'rnd/loss',
            'rnd/error_mean',
            'rnd/error_std',
            'rnd/bonus_mean_running',
            'rnd/bonus_std_running',
            'rnd/train_sample_count',
            'rnd/train_candidate_count',
            'rnd/train_skipped_seen_count',
            'rnd/train_seen_cache_size',
            'rnd/stats_sample_count',
            'rnd/target_snapshot_age',
            'dynamics_prior_alignment/loss',
            'dynamics_prior_alignment/weighted_loss',
            'dynamics_prior_alignment/grad_norm_before_clip',
            'dynamics_prior_alignment/pred_l2_mean',
            'dynamics_prior_alignment/target_l2_mean',
            'dynamics_prior_alignment/valid_weight_sum',
            'dynamics_prior_alignment/raw_trainable_l2_mean',
            'dynamics_prior_alignment/raw_trainable_l2_max',
            'dynamics_prior_alignment/trainable_l2_mean',
            'dynamics_prior_alignment/trainable_l2_max',
            'dynamics_prior_alignment/raw_prior_l2_mean',
            'dynamics_prior_alignment/raw_prior_l2_max',
            'dynamics_prior_alignment/prior_l2_mean',
            'dynamics_prior_alignment/prior_l2_max',
            'dynamics_prior_alignment/full_l2_mean',
            'dynamics_prior_alignment/full_l2_max',
            'intrinsic/reward_target_mean',
            'intrinsic/reward_target_std',
            'intrinsic/reward_target_unclipped_abs_max',
            'intrinsic/reward_target_clip',
            'intrinsic/value_target_mean',
            'intrinsic/value_target_std',
            'intrinsic/value_target_unclipped_abs_max',
            'intrinsic/value_target_clip',
            'intrinsic/bootstrap_value_mean',
            'intrinsic/bootstrap_value_std',
            'intrinsic_dd/selected_reward_mean',
            'intrinsic_dd/selected_reward_std',
            'intrinsic_dd/selected_reward_max',
            'intrinsic_dd/value_target_mean',
            'intrinsic_dd/value_target_std',
            'intrinsic_dd/bootstrap_value_mean',
            'intrinsic_dd/bootstrap_value_std',
            'intrinsic_dd/valid_reward_steps_mean',
            'intrinsic_dd/time_limit_bootstrap_fraction',
            'intrinsic_dd/reward_space_projection',
            'intrinsic_dd/missing_selected_reward_batch',
            'intrinsic_qstd/selected_reward_mean',
            'intrinsic_qstd/selected_reward_std',
            'intrinsic_qstd/selected_reward_max',
            'intrinsic_qstd/selected_reward_nonzero_fraction',
            'intrinsic_qstd/value_target_mean',
            'intrinsic_qstd/value_target_std',
            'intrinsic_qstd/bootstrap_value_mean',
            'intrinsic_qstd/bootstrap_value_std',
            'intrinsic_qstd/valid_reward_steps_mean',
            'intrinsic_qstd/time_limit_bootstrap_fraction',
            'intrinsic_qstd/target_mode_monte_carlo',
            'intrinsic_qstd/reward_aggregation_max',
            'intrinsic_qstd/missing_selected_reward_batch',
            'intrinsic_qstd/recomputed_one_step_action',
            'intrinsic_qstd/raw_reward_mean',
            'intrinsic_qstd/raw_reward_std',
            'intrinsic_qstd/raw_reward_max',
            'intrinsic_qstd/cache_hit_fraction',
            'intrinsic_qstd/cache_size',
            'intrinsic_qstd/target_snapshot_generation',
            'rnd_landscape_query/train_steps',
            'rnd_landscape_query/snapshot_count',
            'rnd_landscape_query/train_loss',
            'rnd_landscape_query/weighted_loss',
            'rnd_landscape_query/loss_weight',
            'rnd_landscape_query/prediction_loss',
            'rnd_landscape_query/centered_prediction_loss',
            'rnd_landscape_query/centered_prediction_correlation',
            'rnd_landscape_query/prediction_mae_log',
            'rnd_landscape_query/prediction_mae_target_space',
            'rnd_landscape_query/prediction_mae_raw',
            'rnd_landscape_query/prediction_target_mean',
            'rnd_landscape_query/prediction_target_std',
            'rnd_landscape_query/transformed_target_mean',
            'rnd_landscape_query/transformed_target_std',
            'rnd_landscape_query/prediction_target_correlation',
            'rnd_landscape_query/target_scale',
            'rnd_landscape_query/target_clip_fraction',
            'rnd_landscape_query/target_error_mean',
            'rnd_landscape_query/valid_target_fraction',
            'rnd_landscape_query/target_scale_phase',
            'rnd_landscape_query/target_scale_sample_count',
            'rnd_landscape_query/grad_norm_before_clip',
            'rnd_landscape_query/aux_representation_grad_norm',
            'rnd_landscape_query/aux_dynamics_grad_norm',
            'rnd_landscape_query/aux_reader_grad_norm',
            'rnd_landscape_query/learning_rate',
            'rnd_landscape_query/space_compact_image',
            'rnd_landscape_query/reconstruction_loss',
            'rnd_landscape_query/reconstruction_mae',
            'rnd_landscape_query/root_code_std',
            'rnd_landscape_query/query_code_std',
            'rnd_landscape_query/decoded_query_min',
            'rnd_landscape_query/decoded_query_max',
            'rnd_auxiliary_decoder/train_loss',
            'rnd_auxiliary_decoder/weighted_loss',
            'rnd_auxiliary_decoder/loss_weight',
            'rnd_auxiliary_decoder/prediction_mae_target_space',
            'rnd_auxiliary_decoder/prediction_mae_raw',
            'rnd_auxiliary_decoder/prediction_correlation_raw',
            'rnd_auxiliary_decoder/prediction_target_mean_raw',
            'rnd_auxiliary_decoder/prediction_target_std_raw',
            'rnd_auxiliary_decoder/valid_target_fraction',
            'rnd_auxiliary_decoder/replay_root_count',
            'rnd_auxiliary_decoder/replay_horizon',
            'rnd_auxiliary_decoder/shared_across_members',
            'rnd_auxiliary_decoder/target_transform_log',
            'rnd_auxiliary_decoder/joint_world_model_update',
            'rnd_auxiliary_decoder/used_in_mcts',
            'rnd_auxiliary_decoder/grad_norm_before_clip',
            'rnd_auxiliary_decoder/learning_rate',
            'rnd_auxiliary_decoder/shared_muzero_optimizer',
            'episodic_inverse/total_loss',
            'episodic_inverse/inverse_loss',
            'episodic_inverse/inverse_accuracy',
            'episodic_inverse/bridge_loss',
            'episodic_inverse/bridge_cosine',
            'episodic_inverse/bridge_member_disagreement',
            'episodic_inverse/embedding_std_mean',
            'episodic_inverse/real_transition_distance_mean',
            'episodic_inverse/valid_transition_count',
            'episodic_inverse/train_world_model',
            'intrinsic/model_loss',
            'intrinsic/reward_loss',
            'intrinsic/value_loss',
            'intrinsic/grad_norm_before_clip',
            'intrinsic/learning_rate',
            'intrinsic/head_only',
            'intrinsic/state_density_current_phase_only',
            'intrinsic/training_sample_count',
            'intrinsic/training_sample_fraction',
            'intrinsic/skipped_no_eligible_samples',
            'intrinsic/reward_prediction_mean',
            'intrinsic/reward_prediction_std',
            'intrinsic/reward_training_target_mean',
            'intrinsic/reward_prediction_mae',
            'intrinsic/reward_prediction_correlation',
            'intrinsic/value_prediction_mean',
            'intrinsic/value_prediction_std',
            'intrinsic/value_training_target_mean',
            'intrinsic/value_prediction_mae',
            'intrinsic/value_prediction_correlation',
            'intrinsic/member_0_reward_prediction_mean',
            'intrinsic/member_0_reward_prediction_mae',
            'intrinsic/member_0_value_prediction_mean',
            'intrinsic/member_0_value_prediction_mae',
            'intrinsic/member_1_reward_prediction_mean',
            'intrinsic/member_1_reward_prediction_mae',
            'intrinsic/member_1_value_prediction_mean',
            'intrinsic/member_1_value_prediction_mae',
            'intrinsic/member_2_reward_prediction_mean',
            'intrinsic/member_2_reward_prediction_mae',
            'intrinsic/member_2_value_prediction_mean',
            'intrinsic/member_2_value_prediction_mae',
            'intrinsic_state_value_disagreement/raw_reward_mean',
            'intrinsic_state_value_disagreement/raw_reward_std',
            'intrinsic_state_value_disagreement/raw_reward_min',
            'intrinsic_state_value_disagreement/raw_reward_max',
            'intrinsic_state_value_disagreement/raw_reward_p50',
            'intrinsic_state_value_disagreement/raw_reward_p90',
            'intrinsic_state_value_disagreement/raw_reward_p99',
            'intrinsic_state_value_disagreement/first_step_reward_mean',
            'intrinsic_state_value_disagreement/first_step_reward_p90',
            'intrinsic_state_value_disagreement/scaled_reward_mean',
            'intrinsic_state_value_disagreement/reward_scale',
            'intrinsic_state_value_disagreement/source_reward_clip',
            'intrinsic_state_value_disagreement/using_stored_collection_rewards',
            'intrinsic_state_value_disagreement/value_target_mean',
            'intrinsic_state_value_disagreement/value_target_std',
            'intrinsic_state_value_disagreement/value_target_p90',
            'intrinsic_state_value_disagreement/bootstrap_value_mean',
            'intrinsic_state_value_disagreement/bootstrap_value_std',
            'intrinsic_state_value_disagreement/valid_reward_steps_mean',
            'intrinsic_state_value_disagreement/time_limit_bootstrap_fraction',
            'intrinsic_state_value_disagreement/target_mode_monte_carlo',
            'intrinsic_state_value_disagreement/reward_position_correlation',
            'intrinsic_state_value_disagreement/reward_abs_velocity_correlation',
            'intrinsic_state_value_disagreement/top10_position_mean',
            'intrinsic_state_value_disagreement/top10_abs_velocity_mean',
            'intrinsic_state_density/reward_mean',
            'intrinsic_state_density/reward_std',
            'intrinsic_state_density/reward_min',
            'intrinsic_state_density/reward_max',
            'intrinsic_state_density/reward_p50',
            'intrinsic_state_density/reward_p90',
            'intrinsic_state_density/effective_count_mean',
            'intrinsic_state_density/effective_count_p90',
            'intrinsic_state_density/value_target_mean',
            'intrinsic_state_density/value_target_std',
            'intrinsic_state_density/value_target_p90',
            'intrinsic_state_density/bootstrap_value_mean',
            'intrinsic_state_density/bootstrap_value_std',
            'intrinsic_state_density/valid_reward_steps_mean',
            'intrinsic_state_density/time_limit_bootstrap_fraction',
            'intrinsic_state_density/target_mode_monte_carlo',
            'intrinsic_state_density/using_stored_collection_rewards',
            'intrinsic_state_density/recomputed_current_snapshot',
            'intrinsic_state_density/snapshot_phase',
            'intrinsic_state_density/snapshot_transition_count',
            'intrinsic_state_density/missing_current_snapshot',
            'intrinsic_state_density/reward_position_correlation',
            'intrinsic_state_density/reward_abs_velocity_correlation',
            'intrinsic_state_density/top10_position_mean',
            'intrinsic_state_density/top10_abs_velocity_mean',
            'latent_novelty/loss',
            'latent_novelty/contrastive_loss',
            'latent_novelty/alignment_loss',
            'latent_novelty/state_disagreement_mean',
            'latent_novelty/state_disagreement_std',
            'latent_novelty/state_disagreement_max',
            'latent_novelty/state_disagreement_no_prior_mean',
            'latent_novelty/state_disagreement_no_prior_max',
            'latent_novelty/same_obs_cosine_mean',
            'latent_novelty/different_obs_cosine_mean',
            'latent_novelty/grad_norm_before_clip',
            'latent_novelty/learning_rate',
            'latent_novelty/train_sample_count',
            'latent_novelty/prior_scale',
        ]
        auxiliary_horizon = max(
            1, int(self._cfg.get('rnd_auxiliary_decoder_max_horizon', 8))
        )
        monitor_vars.extend(
            f'rnd_auxiliary_decoder/depth_{depth}_{metric}'
            for depth in range(1, auxiliary_horizon + 1)
            for metric in ('prediction_mae_raw', 'prediction_correlation_raw')
        )
        num_unroll_steps = max(1, int(self._cfg.get('num_unroll_steps', 5)))
        monitor_vars.extend(
            f'intrinsic/reward_depth_{depth}_{metric}'
            for depth in range(1, num_unroll_steps + 1)
            for metric in (
                'prediction_mae_raw',
                'prediction_mae_log',
                'prediction_mae_target_space',
                'prediction_correlation',
                'valid_count',
            )
        )
        inverse_horizon = max(
            1, int(self._cfg.get('episodic_inverse_max_horizon', 5))
        )
        monitor_vars.extend(
            f'episodic_inverse/depth_{depth}_{metric}'
            for depth in range(1, inverse_horizon + 1)
            for metric in ('bridge_loss', 'bridge_cosine')
        )
        return monitor_vars
