import numpy as np

from experiments.diagnose_branching_rooms_raw_latent_geometry import (
    build_state_space,
    exhaustive_path_revisit_rates,
)


def test_exhaustive_diagnostic_state_space_matches_environment():
    state_space = build_state_space()
    pose_to_index = {
        tuple(pose): index for index, pose in enumerate(state_space.poses.tolist())
    }
    start = pose_to_index[(11, 11, 0)]
    turned_left = int(state_space.transition[start, 0])
    turned_back = int(state_space.transition[turned_left, 1])

    assert state_space.poses.shape == (693 * 4, 3)
    assert state_space.observations.shape == (693 * 4, 3, 84, 84)
    assert len(np.unique(state_space.observation_group)) == 2394
    assert turned_back == start
    assert np.any(state_space.transition[:, 2] == np.arange(len(state_space.poses)))


def test_exhaustive_revisit_rates_separate_pose_from_rgb_aliases():
    rates = exhaustive_path_revisit_rates(build_state_space(), horizon=5)

    assert rates[1]["pose_revisit_fraction"] == rates[1]["observation_revisit_fraction"]
    assert rates[5]["pose_revisit_fraction"] == 0.36915599261278276
    assert rates[5]["different_pose_rgb_alias_fraction"] > 0.0
    assert rates[5]["different_pose_rgb_alias_fraction"] < 0.001
