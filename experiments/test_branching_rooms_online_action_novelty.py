import numpy as np

from experiments.run_branching_rooms_online_action_novelty import (
    action_suffix_indices,
    binary_root_priorities,
    choose_planned_action,
    discounted_branch_scores,
    enumerate_action_suffixes,
    gather_decision_probabilities,
)


def test_suffix_enumeration_and_indices_match():
    suffixes = enumerate_action_suffixes(action_count=3, horizon=3)
    assert suffixes.shape == (27, 3)
    actions = np.asarray([2, 0, 1, 1], dtype=np.int64)
    indices = action_suffix_indices(actions, action_count=3, horizon=3)
    assert suffixes[indices[0]].tolist() == [2, 0, 1]
    assert suffixes[indices[1]].tolist() == [0, 1, 1]


def test_planner_maximizes_cumulative_novel_probability():
    suffixes = enumerate_action_suffixes(action_count=3, horizon=2)
    probabilities = np.zeros((len(suffixes), 2, 3), dtype=np.float32)
    probabilities[..., 0] = 1.0
    selected = np.flatnonzero(np.all(suffixes == np.asarray([1, 2]), axis=1))[0]
    probabilities[selected, :, 0] = 0.0
    probabilities[selected, :, 2] = 1.0

    scores = discounted_branch_scores(probabilities, discount=0.5)
    assert scores[selected] == 1.5
    action, score, exploratory = choose_planned_action(
        probabilities,
        suffixes,
        discount=0.5,
        epsilon=0.0,
        rng=np.random.default_rng(0),
    )
    assert action == 1
    assert score == 1.5
    assert not exploratory


def test_topk_aggregation_ignores_one_optimistic_outlier():
    suffixes = enumerate_action_suffixes(action_count=3, horizon=2)
    probabilities = np.zeros((len(suffixes), 2, 3), dtype=np.float32)
    probabilities[..., 0] = 1.0
    action_zero = np.flatnonzero(suffixes[:, 0] == 0)
    probabilities[action_zero[0], :, 0] = 0.0
    probabilities[action_zero[0], :, 2] = 1.0
    action_one = np.flatnonzero(suffixes[:, 0] == 1)
    probabilities[action_one, :, 0] = 0.25
    probabilities[action_one, :, 2] = 0.75

    max_action, _, _ = choose_planned_action(
        probabilities,
        suffixes,
        discount=1.0,
        epsilon=0.0,
        rng=np.random.default_rng(0),
        aggregation="max",
    )
    topk_action, score, _ = choose_planned_action(
        probabilities,
        suffixes,
        discount=1.0,
        epsilon=0.0,
        rng=np.random.default_rng(0),
        aggregation="topk",
        top_k=3,
    )
    assert max_action == 0
    assert topk_action == 1
    assert score == 1.5


def test_priorities_emphasize_confident_false_novelty():
    targets = np.asarray(
        [
            [0, 2],
            [0, 2],
        ],
        dtype=np.int64,
    )
    probabilities = np.asarray(
        [
            [[0.01, 0.00, 0.99], [0.00, 0.00, 1.00]],
            [[0.90, 0.00, 0.10], [0.00, 0.00, 1.00]],
        ],
        dtype=np.float32,
    )
    priorities = binary_root_priorities(
        targets, probabilities, false_novel_weight=2.0
    )
    assert priorities[0] > priorities[1]


def test_decision_predictions_are_gathered_for_executed_suffix():
    suffixes = enumerate_action_suffixes(action_count=3, horizon=2)
    planned = []
    for root in range(3):
        probabilities = np.zeros((len(suffixes), 2, 3), dtype=np.float32)
        probabilities[:, :, 0] = 1.0
        probabilities[:, :, 2] = np.arange(len(suffixes))[:, None] / 10.0
        planned.append(probabilities)
    actions = np.asarray([2, 1, 0, 2], dtype=np.int64)
    gathered = gather_decision_probabilities(planned, actions, 3, 2)
    indices = action_suffix_indices(actions, 3, 2)
    assert gathered.shape == (3, 2, 3)
    assert np.allclose(gathered[:, 0, 2], indices / 10.0)
