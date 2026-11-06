from types import SimpleNamespace

import numpy as np
import pytest
from easydict import EasyDict

from lzero.entry.train_ensemble_muzero import (
    _complete_episodic_sequence_training_sequences,
    _episodic_sequence_first_visit_rewards,
    _summarize_episodic_sequence_root_actions,
)


def _frame(value):
    return np.full((3, 4, 4), float(value) / 255.0, dtype=np.float32)


@pytest.mark.unittest
def test_first_visit_rewards_use_exact_observation_history():
    observations = np.stack([_frame(0), _frame(1), _frame(0), _frame(2)])
    assert _episodic_sequence_first_visit_rewards(observations).tolist() == [
        1.0,
        0.0,
        1.0,
    ]


@pytest.mark.unittest
def test_complete_episode_extractor_reassembles_segmented_episode():
    cfg = EasyDict(
        policy=dict(
            episodic_sequence_novelty=True,
            game_segment_length=2,
            model=dict(frame_stack_num=1, model_type='conv'),
        )
    )
    first = SimpleNamespace(
        obs_segment=np.stack([_frame(0), _frame(1), _frame(0)]),
        action_segment=np.asarray([1, 2]),
        reward_segment=np.asarray([[0.0], [0.5]], dtype=np.float32),
    )
    second = SimpleNamespace(
        obs_segment=np.stack([_frame(0), _frame(2), _frame(3)]),
        action_segment=np.asarray([0, 1]),
        reward_segment=np.asarray([[1.0], [1.5]], dtype=np.float32),
    )
    metadata = [
        {
            'env_id': 0,
            'env_episode_index': 4,
            'episode_segment_index': 0,
            'segment_length': 2,
            'done': False,
        },
        {
            'env_id': 0,
            'env_episode_index': 4,
            'collection_episode_index': 7,
            'episode_segment_index': 1,
            'segment_length': 2,
            'done': True,
        },
    ]

    episodes = _complete_episodic_sequence_training_sequences(
        cfg, ([first, second], metadata)
    )

    assert len(episodes) == 1
    episode = episodes[0]
    assert episode['observations'].shape == (5, 3, 4, 4)
    assert episode['actions'].tolist() == [1, 2, 0, 1]
    assert episode['extrinsic_rewards'].tolist() == pytest.approx(
        [0.0, 0.5, 1.0, 1.5]
    )
    assert episode['episodic_rewards'].tolist() == [1.0, 0.0, 1.0, 1.0]
    assert episode['collection_episode_index'] == 7


@pytest.mark.unittest
def test_complete_episode_extractor_ignores_incomplete_segment_groups():
    cfg = EasyDict(
        policy=dict(
            episodic_sequence_novelty=True,
            game_segment_length=2,
            model=dict(frame_stack_num=1, model_type='conv'),
        )
    )
    segment = SimpleNamespace(
        obs_segment=np.stack([_frame(0), _frame(1), _frame(2)]),
        action_segment=np.asarray([1, 2]),
        reward_segment=np.asarray([[0.0], [0.0]], dtype=np.float32),
    )
    metadata = [{
        'env_id': 0,
        'env_episode_index': 0,
        'episode_segment_index': 0,
        'segment_length': 2,
        'done': False,
    }]

    assert _complete_episodic_sequence_training_sequences(
        cfg, ([segment], metadata)
    ) == []


@pytest.mark.unittest
def test_root_action_summary_distinguishes_ranking_and_calibration():
    records = [
        {
            'selected_prediction': 0.8,
            'realized_novelty': 1.0,
            'candidate_probability_mean': 0.5,
            'candidate_probability_max': 0.8,
            'candidate_probability_spread': 0.6,
            'selected_probability_gap_from_max': 0.0,
            'selected_rank': 1,
            'selected_is_argmax': 1.0,
            'selected_intrinsic_visit_count': 3,
            'novelty_argmax_intrinsic_visit_count': 3,
            'tree_active': 1.0,
            'trained_phase': 2,
        },
        {
            'selected_prediction': 0.3,
            'realized_novelty': 0.0,
            'candidate_probability_mean': 0.4,
            'candidate_probability_max': 0.6,
            'candidate_probability_spread': 0.4,
            'selected_probability_gap_from_max': 0.3,
            'selected_rank': 2,
            'selected_is_argmax': 0.0,
            'selected_intrinsic_visit_count': 2,
            'novelty_argmax_intrinsic_visit_count': 0,
            'tree_active': 1.0,
            'trained_phase': 2,
        },
    ]

    summary = _summarize_episodic_sequence_root_actions(records)

    assert summary['root_count'] == 2
    assert summary['selected_auc'] == pytest.approx(1.0)
    assert summary['selected_brier'] == pytest.approx(0.065)
    assert summary['selected_novel_revisit_margin'] == pytest.approx(0.5)
    assert summary['selected_is_argmax_fraction'] == pytest.approx(0.5)
    assert summary['selected_rank_mean'] == pytest.approx(1.5)
    assert summary['selected_probability_gap_from_max_mean'] == pytest.approx(0.15)
    assert summary['candidate_probability_spread_mean'] == pytest.approx(0.5)
    assert summary['novelty_argmax_unvisited_fraction'] == pytest.approx(0.5)
