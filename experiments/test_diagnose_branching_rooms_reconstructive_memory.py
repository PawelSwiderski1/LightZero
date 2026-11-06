import numpy as np

from experiments.diagnose_branching_rooms_raw_latent_geometry import build_state_space
from experiments.diagnose_branching_rooms_reconstructive_memory import (
    _binary_auc,
    first_visit_rewards,
    generate_heldout_trajectories,
    image_metrics,
)


def test_first_visit_rewards_use_observation_identity():
    rewards = first_visit_rewards(np.asarray([4, 7, 7, 9, 4, 10]))
    np.testing.assert_array_equal(rewards, np.asarray([1, 0, 1, 0, 1]))


def test_heldout_trajectories_follow_exact_transition_table():
    state_space = build_state_space()
    trajectories = generate_heldout_trajectories(
        state_space, episode_count=3, episode_length=12, seed=17
    )

    assert trajectories.pose_indices.shape == (3, 13)
    assert trajectories.actions.shape == (3, 12)
    assert trajectories.policy_names == ("uniform", "persistent", "macro")
    for episode in range(3):
        expected = state_space.transition[
            trajectories.pose_indices[episode, :-1],
            trajectories.actions[episode],
        ]
        np.testing.assert_array_equal(
            expected, trajectories.pose_indices[episode, 1:]
        )


def test_image_metrics_and_auc_have_expected_direction():
    target = np.zeros((2, 3, 4, 4), dtype=np.float32)
    exact = image_metrics(target, target)
    offset = image_metrics(np.ones_like(target) * 0.5, target)

    assert exact["mae"] == 0.0
    assert exact["psnr"] == 120.0
    assert offset["mae"] == 0.5
    assert offset["psnr"] < exact["psnr"]
    assert _binary_auc(np.asarray([0, 0, 1, 1]), np.asarray([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert _binary_auc(np.asarray([1, 1]), np.asarray([0.1, 0.2])) is None
