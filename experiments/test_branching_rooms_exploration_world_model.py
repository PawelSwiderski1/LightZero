import argparse

import numpy as np
import torch

from experiments.run_branching_rooms_exploration_world_model import (
    ExplorationWorldModel,
    binary_metrics,
    cell_targets,
    choose_revisit_threshold,
    direct_key_contrastive_loss,
    gather_rollout_batch,
    multipositive_contrastive_loss,
)


def test_model_encodes_history_and_rolls_multiple_actions():
    model = ExplorationWorldModel(
        encoder_dim=8, key_dim=8, belief_dim=16, action_dim=4
    )
    observations = torch.randint(0, 256, (2, 7, 3, 28, 28), dtype=torch.uint8)
    actions = torch.randint(0, 3, (2, 6))
    encoded = model.encode_trajectory(observations, actions)
    roots = torch.tensor([0, 2])
    root_keys, root_beliefs, suffixes, targets = gather_rollout_batch(
        encoded, actions, roots, horizon=3
    )
    predicted = model.rollout(root_keys, root_beliefs, suffixes)
    assert encoded.keys.shape == (2, 7, 8)
    assert encoded.beliefs.shape == (2, 7, 16)
    assert predicted.shape == targets.shape == (4, 3, 8)
    assert torch.allclose(predicted.norm(dim=-1), torch.ones(4, 3), atol=1e-5)


def test_cell_targets_reduce_rendered_observation_to_logical_grid():
    observations = torch.randint(0, 256, (2, 3, 28, 28), dtype=torch.uint8)
    targets = cell_targets(observations)
    assert targets.shape == (2, 3, 7, 7)
    assert float(targets.min()) >= 0.0
    assert float(targets.max()) <= 1.0


def test_contrastive_loss_accepts_duplicate_positive_observations():
    targets = torch.tensor(
        [[[[1.0, 0.0]], [[1.0, 0.0]], [[0.0, 1.0]]]]
    )
    predicted = targets.clone()
    target_ids = torch.tensor([[[0], [0], [1]]])
    loss = multipositive_contrastive_loss(
        predicted, targets, target_ids, temperature=0.1
    )
    assert torch.isfinite(loss)
    assert float(loss) < 0.01


def test_direct_key_contrastive_loss_accepts_duplicate_observations():
    keys = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
    state_ids = torch.tensor([[0, 0, 1]])
    loss = direct_key_contrastive_loss(keys, state_ids, temperature=0.1)
    assert torch.isfinite(loss)
    assert float(loss) < 0.01


def test_threshold_is_selected_from_revisit_scores():
    labels = np.asarray([0, 0, 1, 1])
    scores = np.asarray([0.1, 0.2, 0.8, 0.9])
    threshold = choose_revisit_threshold(labels, scores)
    metrics = binary_metrics(labels, scores, threshold)
    assert metrics["f1"] == 1.0
    assert 0.2 < threshold <= 0.8
