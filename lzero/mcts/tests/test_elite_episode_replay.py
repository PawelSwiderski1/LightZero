import numpy as np
import pytest
from easydict import EasyDict

from lzero.entry.train_ensemble_muzero import EliteEpisodeReplayBuffer, _concat_train_data


class DummySegment:

    def __init__(self, rewards, start_position=0.0):
        self.reward_segment = list(rewards)
        self.action_segment = [0 for _ in rewards]
        self.obs_segment = [np.asarray([start_position, 0.0], dtype=np.float32)]

    def __len__(self):
        return len(self.action_segment)


class DummyReplayBuffer:

    def __init__(self, num_unroll_steps=0):
        self.orig_data = None
        self.num_unroll_steps = num_unroll_steps

    def sample_from_orig_data(self, orig_data, policy):
        self.orig_data = orig_data
        batch_size = len(orig_data[0])
        target_shape = (batch_size, self.num_unroll_steps + 1)
        return [
            [np.zeros((batch_size, 1), dtype=np.float32)],
            [
                np.zeros(target_shape, dtype=np.float32),
                np.full(target_shape, 999.0, dtype=np.float32),
                np.zeros((batch_size, self.num_unroll_steps + 1, 1), dtype=np.float32),
            ],
        ]


@pytest.mark.unittest
def test_elite_episode_replay_keeps_top_return_segments():
    cfg = EasyDict(
        elite_episode_replay=True,
        elite_episode_buffer_size=2,
        elite_episode_sample_ratio=0.25,
        elite_episode_min_return=None,
        elite_episode_monte_carlo_value_target=True,
        game_segment_length=200,
        num_unroll_steps=2,
        discount_factor=1.0,
    )
    elite = EliteEpisodeReplayBuffer(cfg)
    low = DummySegment([-1] * 200)
    high = DummySegment([-1] * 174)
    mid = DummySegment([-1] * 190)

    added = elite.update([[low, high, mid], [{'done': True}, {'done': True}, {'done': True}]])

    assert added == 3
    stats = elite.stats()
    assert stats['elite_replay/size'] == 2.0
    assert stats['elite_replay/best_return'] == -174.0
    assert stats['elite_replay/worst_return'] == -190.0


@pytest.mark.unittest
def test_elite_episode_replay_samples_explicit_roots():
    cfg = EasyDict(
        elite_episode_replay=True,
        elite_episode_buffer_size=2,
        elite_episode_sample_ratio=0.25,
        elite_episode_min_return=None,
        elite_episode_monte_carlo_value_target=True,
        game_segment_length=200,
        num_unroll_steps=2,
        discount_factor=1.0,
    )
    elite = EliteEpisodeReplayBuffer(cfg)
    segment = DummySegment([-1] * 10)
    elite.update([[segment], [{'done': True}]])
    replay_buffer = DummyReplayBuffer(num_unroll_steps=cfg.num_unroll_steps)

    train_data = elite.sample_train_data(5, replay_buffer, policy=None)

    game_segments, positions, indices, weights, make_time = replay_buffer.orig_data
    assert game_segments == [segment] * 5
    assert all(0 <= pos < 9 for pos in positions)
    assert all(str(index).startswith('elite:') for index in indices)
    assert weights == [1.0] * 5
    assert make_time == [0.0] * 5
    target_values = train_data[1][1]
    assert target_values.shape == (5, cfg.num_unroll_steps + 1)
    for row, pos in zip(target_values, positions):
        expected = [
            -float(max(0, len(segment) - (pos + unroll_offset)))
            for unroll_offset in range(cfg.num_unroll_steps + 1)
        ]
        np.testing.assert_allclose(row, expected)
    assert elite.stats()['elite_replay/last_sample_count'] == 5.0
    assert elite.stats()['elite_replay/mc_value_target_enabled'] == 1.0


@pytest.mark.unittest
def test_elite_episode_replay_stratifies_storage_by_start_observation():
    cfg = EasyDict(
        elite_episode_replay=True,
        elite_episode_buffer_size=4,
        elite_episode_sample_ratio=0.25,
        elite_episode_min_return=None,
        elite_episode_monte_carlo_value_target=True,
        elite_episode_start_stratification=True,
        elite_episode_start_observation_index=0,
        elite_episode_start_bin_edges=[-0.5],
        game_segment_length=200,
        num_unroll_steps=2,
        discount_factor=1.0,
    )
    elite = EliteEpisodeReplayBuffer(cfg)
    segments = [
        DummySegment([-1] * 180, -0.58),
        DummySegment([-1] * 170, -0.57),
        DummySegment([-1] * 160, -0.56),
        DummySegment([-1] * 199, -0.48),
        DummySegment([-1] * 198, -0.47),
    ]

    elite.update([segments, [{'done': True} for _ in segments]])

    stats = elite.stats()
    assert stats['elite_replay/size'] == 4.0
    assert stats['elite_replay/start_bin_0/size'] == 2.0
    assert stats['elite_replay/start_bin_0/best_return'] == -160.0
    assert stats['elite_replay/start_bin_0/worst_return'] == -170.0
    assert stats['elite_replay/start_bin_1/size'] == 2.0
    assert stats['elite_replay/start_bin_1/best_return'] == -198.0
    assert stats['elite_replay/start_bin_1/worst_return'] == -199.0


@pytest.mark.unittest
def test_elite_episode_replay_balances_samples_across_populated_start_bins():
    cfg = EasyDict(
        elite_episode_replay=True,
        elite_episode_buffer_size=4,
        elite_episode_sample_ratio=0.25,
        elite_episode_min_return=None,
        elite_episode_monte_carlo_value_target=True,
        elite_episode_start_stratification=True,
        elite_episode_start_observation_index=0,
        elite_episode_start_bin_edges=[-0.5],
        game_segment_length=200,
        num_unroll_steps=2,
        discount_factor=1.0,
    )
    elite = EliteEpisodeReplayBuffer(cfg)
    left = DummySegment([-1] * 10, -0.58)
    right = DummySegment([-1] * 10, -0.42)
    elite.update([[left, right], [{'done': True}, {'done': True}]])
    replay_buffer = DummyReplayBuffer(num_unroll_steps=cfg.num_unroll_steps)

    elite.sample_train_data(10, replay_buffer, policy=None)

    sampled_segments = replay_buffer.orig_data[0]
    assert sampled_segments.count(left) == 5
    assert sampled_segments.count(right) == 5
    stats = elite.stats()
    assert stats['elite_replay/start_bin_0/last_sample_count'] == 5.0
    assert stats['elite_replay/start_bin_1/last_sample_count'] == 5.0


@pytest.mark.unittest
def test_concat_train_data_preserves_normal_then_elite_order():
    normal = [
        [np.array([[1], [2]]), np.array([10, 20])],
        [np.array([[100], [200]])],
    ]
    elite = [
        [np.array([[3]]), np.array([30])],
        [np.array([[300]])],
    ]

    mixed = _concat_train_data(normal, elite)

    np.testing.assert_array_equal(mixed[0][0], np.array([[1], [2], [3]]))
    np.testing.assert_array_equal(mixed[0][1], np.array([10, 20, 30]))
    np.testing.assert_array_equal(mixed[1][0], np.array([[100], [200], [300]]))


@pytest.mark.unittest
def test_concat_train_data_concats_member_specific_targets_on_batch_axis():
    normal_member_targets = np.arange(18, dtype=np.float32).reshape(3, 2, 3)
    elite_member_targets = np.arange(9, dtype=np.float32).reshape(3, 1, 3) + 100
    normal = [
        [np.array([[1], [2]])],
        [
            np.zeros((2, 3), dtype=np.float32),
            np.zeros((2, 3), dtype=np.float32),
            np.zeros((2, 3, 1), dtype=np.float32),
            normal_member_targets,
        ],
    ]
    elite = [
        [np.array([[3]])],
        [
            np.zeros((1, 3), dtype=np.float32),
            np.zeros((1, 3), dtype=np.float32),
            np.zeros((1, 3, 1), dtype=np.float32),
            elite_member_targets,
        ],
    ]

    mixed = _concat_train_data(normal, elite)

    assert mixed[1][3].shape == (3, 3, 3)
    np.testing.assert_array_equal(mixed[1][3][:, :2], normal_member_targets)
    np.testing.assert_array_equal(mixed[1][3][:, 2:], elite_member_targets)


@pytest.mark.unittest
def test_elite_episode_monte_carlo_overrides_member_specific_targets():
    cfg = EasyDict(
        elite_episode_replay=True,
        elite_episode_buffer_size=2,
        elite_episode_sample_ratio=0.25,
        elite_episode_min_return=None,
        elite_episode_monte_carlo_value_target=True,
        game_segment_length=200,
        num_unroll_steps=2,
        discount_factor=1.0,
    )
    elite = EliteEpisodeReplayBuffer(cfg)
    entries = [
        {'segment': DummySegment([-1] * 5), 'length': 5},
        {'segment': DummySegment([-1] * 5), 'length': 5},
    ]
    positions = [1, 2]
    train_data = [
        [np.zeros((2, 1), dtype=np.float32)],
        [
            np.zeros((2, 3), dtype=np.float32),
            np.full((2, 3), 999.0, dtype=np.float32),
            np.zeros((2, 3, 1), dtype=np.float32),
            np.full((3, 2, 3), 777.0, dtype=np.float32),
        ],
    ]

    updated = elite._override_value_targets_with_monte_carlo(train_data, entries, positions)

    expected = np.asarray([[-4.0, -3.0, -2.0], [-3.0, -2.0, -1.0]], dtype=np.float32)
    np.testing.assert_allclose(updated[1][1], expected)
    for member_id in range(3):
        np.testing.assert_allclose(updated[1][3][member_id], expected)
