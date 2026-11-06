import numpy as np

from experiments.run_branching_rooms_action_novelty import (
    NOVEL,
    ROOT_HISTORY_REVISIT,
    DatasetBundle,
    build_novelty_targets,
    state_ids_from_observations,
)
from experiments.window_branching_rooms_trajectory_dataset import (
    build_windowed_dataset,
    window_starts,
)


def _observation(state: int) -> np.ndarray:
    result = np.zeros((3, 4, 4), dtype=np.uint8)
    result[0, 0, 0] = state
    return result


def _bundle() -> DatasetBundle:
    state_sequences = (
        [0, 1, 2, 3, 1, 4, 5, 6, 7, 8, 9],
        [10, 11, 12, 13, 11, 14, 15, 16, 17, 18, 19],
        [20, 21, 22, 23, 21, 24, 25, 26, 27, 28, 29],
        [30, 31, 32, 33, 31, 34, 35, 36, 37, 38, 39],
    )
    observations = np.stack(
        [[_observation(state) for state in sequence] for sequence in state_sequences]
    )
    poses = np.asarray(
        [[[state, 0, 0] for state in sequence] for sequence in state_sequences],
        dtype=np.int16,
    )
    actions = np.zeros((4, 10), dtype=np.uint8)
    observation_targets = np.stack(
        [
            build_novelty_targets(state_ids_from_observations(episode), 2)
            for episode in observations
        ]
    )
    return DatasetBundle(
        observations=observations,
        actions=actions,
        poses=poses,
        observation_targets=observation_targets,
        pose_targets=observation_targets.copy(),
        split_ids=np.asarray([0, 1, 2, 3], dtype=np.uint8),
        policy_ids=np.asarray([0, 0, 0, 0], dtype=np.uint8),
        episode_seeds=np.asarray([100, 200, 300, 400], dtype=np.int64),
        metadata={
            "horizon": 2,
            "episode_length": 10,
            "split_names": ["train", "validation", "test", "heldout_policy"],
            "policy_names": ["muzero_checkpoint"],
        },
    )


def test_window_starts_includes_final_end_aligned_window():
    assert window_starts(10, 4, 4) == [0, 4, 6]
    assert window_starts(10, 4, 3) == [0, 3, 6]


def test_windowing_preserves_parent_splits_and_recomputes_local_targets():
    result = build_windowed_dataset(
        _bundle(), window_actions=4, stride=3, horizon=2
    )
    bundle = result.bundle

    assert bundle.observations.shape == (12, 5, 3, 4, 4)
    assert bundle.actions.shape == (12, 4)
    assert bundle.observation_targets.shape == (12, 3, 2)
    np.testing.assert_array_equal(
        np.bincount(bundle.split_ids, minlength=4), [3, 3, 3, 3]
    )
    for parent_index in range(4):
        parent_windows = result.parent_episode_indices == parent_index
        assert np.all(bundle.split_ids[parent_windows] == parent_index)

    # At parent time 4, state 1 is a full-trajectory revisit.  In the window
    # beginning at time 3, no earlier state is available, so it is locally novel.
    assert _bundle().observation_targets[0, 3, 0] == ROOT_HISTORY_REVISIT
    window_index = int(
        np.flatnonzero(
            (result.parent_episode_indices == 0) & (result.window_starts == 3)
        )[0]
    )
    assert bundle.observation_targets[window_index, 0, 0] == NOVEL
    assert bundle.metadata["parent_split_preserved"] is True
    assert bundle.metadata["window_novelty_history_scope"] == "window_local"
