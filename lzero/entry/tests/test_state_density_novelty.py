import numpy as np
from easydict import EasyDict

from lzero.entry.state_density_novelty import StateDensityNoveltyTracker


class _Segment:
    def __init__(self, observations):
        self.frame_stack_num = 1
        self.obs_segment = np.asarray(observations, dtype=np.float32)
        self.reward_segment = np.full(len(observations) - 1, -1.0, dtype=np.float32)
        self.action_segment = np.zeros(len(self.reward_segment), dtype=np.int64)
        self.intrinsic_reward_segment = np.zeros(len(self.reward_segment), dtype=np.float32)
        self.intrinsic_reward_raw_segment = np.zeros(len(self.reward_segment), dtype=np.float32)

    def __len__(self):
        return len(self.reward_segment)


def _collection(observations):
    segment = _Segment(observations)
    metadata = {'segment_length': len(segment), 'done': True}
    return [segment], [metadata]


def test_phase_snapshot_is_frozen_and_rewards_decay_after_refresh():
    tracker = StateDensityNoveltyTracker(
        EasyDict(
            state_density_position_bin_width=0.02,
            state_density_velocity_bin_width=0.002,
            state_density_neighbor_radius=0,
            state_density_snapshot_update_interval=1,
            state_density_reward_scale=1.0,
            state_density_reward_clip=1.0,
        )
    )
    observations = [
        [-0.50, 0.000],
        [-0.49, 0.001],
        [-0.49, 0.001],
    ]

    phase_zero = _collection(observations)
    tracker.annotate_collection(phase_zero, collection_phase=0)
    np.testing.assert_allclose(phase_zero[0][0].intrinsic_reward_segment, 1.0)

    # A second collection in the same phase still uses the phase-start empty snapshot.
    same_phase = _collection(observations)
    tracker.annotate_collection(same_phase, collection_phase=0)
    np.testing.assert_allclose(same_phase[0][0].intrinsic_reward_segment, 1.0)

    next_phase = _collection(observations)
    stats = tracker.annotate_collection(next_phase, collection_phase=1)
    assert np.all(next_phase[0][0].intrinsic_reward_segment < 1.0)
    assert stats['snapshot_transition_count'] == 4.0
    assert stats['snapshot_updated'] == 1.0


def test_neighbor_radius_counts_adjacent_bins():
    tracker = StateDensityNoveltyTracker(
        EasyDict(
            state_density_position_bin_width=0.02,
            state_density_velocity_bin_width=0.002,
            state_density_neighbor_radius=1,
            state_density_snapshot_update_interval=1,
        )
    )
    tracker.update_live_counts(np.asarray([[-0.50, 0.0]], dtype=np.float32))
    tracker.begin_phase(0)

    adjacent_position = np.asarray([[-0.48, 0.0]], dtype=np.float32)
    reward, count = tracker.rewards(adjacent_position)
    assert count[0] == 1.0
    np.testing.assert_allclose(reward[0], 1.0 / np.sqrt(2.0), rtol=1e-6)
