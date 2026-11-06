import json

import numpy as np
import torch
from easydict import EasyDict

from lzero.entry.train_ensemble_muzero import (
    _fresh_collection_next_observations,
    _maybe_refresh_rnd_landscape_query_target_scale,
    _probe_fixed_sequence_intrinsic_returns,
    _train_rnd_predictor_from_fresh_collection,
)


class _DiagnosticModel:
    def __init__(self):
        self.training = True
        self.active_member_id = 0

    def eval(self):
        self.training = False

    def train(self, mode=True):
        self.training = mode

    def set_active_member(self, member_id):
        self.active_member_id = int(member_id)

    def initial_inference(self, observations, member_id=None):
        return EasyDict(latent_state=observations)

    def recurrent_inference(self, latent_state, actions, member_id=None):
        action_delta = (actions.float() - 1.0).unsqueeze(1) * 0.001
        next_latent = latent_state.clone()
        next_latent[:, :1] += action_delta
        return EasyDict(latent_state=next_latent)

    def intrinsic_reward_from_latent(self, latent_state, member_id=None):
        return latent_state[:, :1].abs()

    def intrinsic_value_from_latent(self, latent_state, member_id=None):
        return torch.zeros((latent_state.shape[0], 1), dtype=latent_state.dtype)


class _DiagnosticPolicy:
    def _collection_rnd_rewards_for_observations(self, observations):
        observations = np.asarray(observations, dtype=np.float32)
        rewards = np.abs(observations[:, 0])
        return {'bonus': rewards, 'raw_error': rewards}


class _FreshObservationSegment:
    def __init__(self):
        self.obs_segment = [
            np.asarray([-0.5, 0.00], dtype=np.float32),
            np.asarray([-0.49, 0.01], dtype=np.float32),
            np.asarray([-0.47, 0.02], dtype=np.float32),
        ]
        self.action_segment = [2, 2]

    def __len__(self):
        return len(self.action_segment)


class _FreshRNDPolicy:
    def __init__(self, optimizer_step_count=1):
        self.optimizer_step_count = optimizer_step_count
        self.refresh_count = 0
        self.collection_completed = False

    def train_rnd_on_fresh_observations(self, observations):
        return {
            'rnd_fresh/observation_count': float(len(observations)),
            'rnd_fresh/optimizer_step_count': float(self.optimizer_step_count),
        }

    def refresh_rnd_target_snapshot_for_collection(
            self, collection_completed=False
    ):
        self.refresh_count += 1
        self.collection_completed = bool(collection_completed)


class _ScaleReplay:
    def __init__(self):
        self.observations = np.arange(20, dtype=np.float32).reshape(10, 2)

    def sample_observations_uniform(self, sample_size):
        indices = np.random.choice(
            len(self.observations), size=min(sample_size, len(self.observations)),
            replace=False,
        )
        return self.observations[indices], indices


class _ScalePolicy:
    def __init__(self):
        self.observations = None
        self.phase = None

    def refresh_rnd_landscape_query_target_scale(
            self, observations, collection_phase=-1
    ):
        self.observations = np.asarray(observations)
        self.phase = int(collection_phase)
        return {
            'rnd_landscape_query/target_scale': 0.25,
            'rnd_landscape_query/target_scale_raw_std': 0.1,
            'rnd_landscape_query/target_scale_raw_p90': 0.4,
            'rnd_landscape_query/target_scale_clip_fraction': 0.0,
            'rnd_landscape_query/target_scale_sample_count': float(
                len(observations)
            ),
        }


def test_fresh_collection_rnd_inputs_are_real_next_observations_once_each():
    cfg = EasyDict(
        policy=EasyDict(
            game_segment_length=200,
            model=EasyDict(frame_stack_num=1),
        )
    )
    segment = _FreshObservationSegment()
    new_data = [[segment], [{'segment_length': 2}]]

    observations = _fresh_collection_next_observations(cfg, new_data)

    assert observations.shape == (2, 2)
    assert np.allclose(observations, np.asarray(segment.obs_segment[1:]))


def test_fresh_rnd_update_refreshes_snapshot_before_replay_learning():
    cfg = EasyDict(
        policy=EasyDict(
            game_segment_length=200,
            model=EasyDict(frame_stack_num=1),
            intrinsic_reward_source='rnd',
            rnd_train_from_fresh_collection_only=True,
            rnd_target_snapshot_refresh_per_collect=True,
            rnd_target_snapshot_refresh_after_fresh_update=True,
        )
    )
    policy = _FreshRNDPolicy(optimizer_step_count=1)
    new_data = [[_FreshObservationSegment()], [{'segment_length': 2}]]

    stats = _train_rnd_predictor_from_fresh_collection(
        cfg, policy, new_data, tb_logger=None, collection_phase=0
    )

    assert policy.refresh_count == 1
    assert stats['rnd_fresh/target_snapshot_refreshed_after_update'] == 1.0


def test_fresh_rnd_without_optimizer_step_keeps_existing_snapshot():
    cfg = EasyDict(
        policy=EasyDict(
            game_segment_length=200,
            model=EasyDict(frame_stack_num=1),
            intrinsic_reward_source='rnd',
            rnd_train_from_fresh_collection_only=True,
            rnd_target_snapshot_refresh_per_collect=True,
            rnd_target_snapshot_refresh_after_fresh_update=True,
        )
    )
    policy = _FreshRNDPolicy(optimizer_step_count=0)
    new_data = [[_FreshObservationSegment()], [{'segment_length': 2}]]

    stats = _train_rnd_predictor_from_fresh_collection(
        cfg, policy, new_data, tb_logger=None, collection_phase=0
    )

    assert policy.refresh_count == 0
    assert stats['rnd_fresh/target_snapshot_refreshed_after_update'] == 0.0


def test_fresh_rnd_query_experiment_marks_collection_complete_before_refresh():
    cfg = EasyDict(
        policy=EasyDict(
            game_segment_length=200,
            model=EasyDict(frame_stack_num=1),
            intrinsic_reward_source='rnd',
            rnd_train_from_fresh_collection_only=True,
            rnd_target_snapshot_refresh_per_collect=True,
            rnd_target_snapshot_refresh_after_fresh_update=True,
            rnd_landscape_query=True,
        )
    )
    policy = _FreshRNDPolicy(optimizer_step_count=1)
    new_data = [[_FreshObservationSegment()], [{'segment_length': 2}]]

    _train_rnd_predictor_from_fresh_collection(
        cfg, policy, new_data, tb_logger=None, collection_phase=0
    )

    assert policy.refresh_count == 1
    assert policy.collection_completed is True


def test_normalized_raw_scale_sampling_does_not_advance_training_numpy_rng():
    cfg = EasyDict(
        policy=EasyDict(
            rnd_landscape_query=True,
            rnd_landscape_query_target_transform='normalized_raw',
            rnd_landscape_query_normalized_raw_reference_size=6,
        )
    )
    replay = _ScaleReplay()
    policy = _ScalePolicy()
    np.random.seed(91)
    expected_next = np.random.random(5)
    np.random.seed(91)

    stats = _maybe_refresh_rnd_landscape_query_target_scale(
        cfg, policy, replay, tb_logger=None, completed_collection_phase=4
    )
    actual_next = np.random.random(5)

    assert np.array_equal(actual_next, expected_next)
    assert policy.observations.shape == (6, 2)
    assert policy.phase == 4
    assert stats['rnd_landscape_query/target_scale'] == 0.25


def test_fixed_sequence_diagnostic_compares_true_rnd_with_imagined_predictions():
    cfg = EasyDict(
        policy=EasyDict(
            device='cpu',
            discount_factor=0.997,
            rnd_intrinsic_discount_factor=0.99,
            intrinsic_reward_source='rnd',
            model=EasyDict(ensemble_size=2),
        )
    )
    state = {
        'obs': np.asarray([-0.5, 0.0], dtype=np.float32),
        'collection_phase': 0,
        'train_iter_at_collect': 0,
        'envstep_before_collect': 0,
        'envstep_after_collect': 1,
        'segment_index_in_collect': 0,
        'env_id': 0,
        'env_episode_index': 0,
        'collection_episode_index': 0,
        'episode_segment_index': 0,
        'episode_return': -200.0,
        'video_path': None,
        'timestep': 0,
        'obs_index': 0,
        'selected_action': 2,
        'selected_action_direction': 'right',
        'obs_position': -0.5,
        'obs_velocity': 0.0,
        'future_max_position_20': -0.5,
        'future_max_position_50': -0.5,
        'future_max_position_100': -0.5,
        'future_reached_goal_100': 0.0,
        'future_reached_goal_remaining': 0.0,
    }

    rows = _probe_fixed_sequence_intrinsic_returns(
        cfg,
        _DiagnosticModel(),
        [state],
        [('right2', [2, 2])],
        policy=_DiagnosticPolicy(),
    )

    assert len(rows) == 1
    assert len(json.loads(rows[0]['sequence_step_intrinsic_rewards'])) == 2
    assert len(json.loads(rows[0]['sequence_step_true_rnd_rewards'])) == 2
    assert np.isfinite(rows[0]['true_rnd_first_reward'])
    assert np.isfinite(rows[0]['true_rnd_discounted_reward_sum'])
    assert rows[0]['direct_true_endpoint_intrinsic_value_mean'] == 0.0
    assert rows[0]['imagined_direct_intrinsic_value_gap_mean'] == 0.0
    assert rows[0]['imagined_direct_intrinsic_value_abs_gap_mean'] == 0.0
    assert len(json.loads(rows[0]['member_direct_true_endpoint_intrinsic_values'])) == 2
    assert len(json.loads(rows[0]['member_imagined_direct_intrinsic_value_gaps'])) == 2
