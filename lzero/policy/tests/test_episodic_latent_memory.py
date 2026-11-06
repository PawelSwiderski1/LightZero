import math

import pytest
import torch

from lzero.policy.episodic_latent_memory import (
    EpisodicLatentMemoryContext,
    EpisodicLatentRewardGate,
)


@pytest.mark.unittest
def test_real_memory_exact_match_receives_minimum_reward_factor():
    memory = EpisodicLatentMemoryContext(
        keys=torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]]),
        transition_distance_scale=2.0,
    )
    gate = EpisodicLatentRewardGate(
        [memory],
        kernel_bandwidth=1.0,
        minimum_reward_factor=0.05,
    )

    rewards, records = gate.apply(
        torch.tensor([4.0, 4.0]),
        torch.tensor([[[0.0, 1.0]], [[-1.0, 0.0]]]),
        root_indices=[0, 0],
        action_paths=[[0], [1]],
    )

    assert rewards[0].item() == pytest.approx(0.2)
    expected_far_factor = 0.05 + 0.95 * (1.0 - math.exp(-1.0))
    assert rewards[1].item() == pytest.approx(4.0 * expected_far_factor)
    assert records[0]['episodic_matched_source'] == 'real'
    assert records[0]['episodic_reward_factor'] == pytest.approx(0.05)


@pytest.mark.unittest
def test_path_memory_suppresses_imagined_cycle():
    memory = EpisodicLatentMemoryContext(
        keys=torch.tensor([[[1.0, 0.0]]]),
        transition_distance_scale=1.0,
    )
    gate = EpisodicLatentRewardGate(
        [memory], minimum_reward_factor=0.0, include_path_memory=True
    )
    first_key = torch.tensor([[[0.0, 1.0]]])

    first_reward, _ = gate.apply(
        torch.ones(1), first_key, root_indices=[0], action_paths=[[2]]
    )
    cycle_reward, records = gate.apply(
        torch.ones(1), first_key, root_indices=[0], action_paths=[[2, 1]]
    )

    assert first_reward.item() > 0.8
    assert cycle_reward.item() == pytest.approx(0.0)
    assert records[0]['episodic_matched_source'] == 'path'
    assert records[0]['episodic_path_distance'] == pytest.approx(0.0)


@pytest.mark.unittest
def test_member_distances_are_matched_by_physical_memory_entry():
    memory = EpisodicLatentMemoryContext(
        keys=torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[-1.0, 0.0], [0.0, -1.0]],
            ]
        ),
        transition_distance_scale=1.0,
    )
    candidate = torch.tensor([[[1.0, 0.0], [0.0, -1.0]]])
    gate = EpisodicLatentRewardGate([memory], minimum_reward_factor=0.0)

    _, records = gate.apply(
        torch.ones(1), candidate, root_indices=[0], action_paths=[[0]]
    )

    # Each member has an exact match, but in a different physical memory entry.
    # The gate must not combine those two unrelated matches into distance zero.
    assert records[0]['episodic_real_distance'] == pytest.approx(2.0)
    assert records[0]['episodic_reward_factor'] > 0.8


@pytest.mark.unittest
def test_ngu_knn_gate_turns_local_repetition_into_a_pseudo_count():
    memory = EpisodicLatentMemoryContext(
        keys=torch.zeros(10, 2, 4),
        transition_distance_scale=1.0,
    )
    gate = EpisodicLatentRewardGate(
        [memory],
        mode='ngu_knn',
        knn_k=10,
        include_path_memory=False,
        minimum_reward_factor=0.05,
    )

    rewards, records = gate.apply(
        torch.ones(2),
        torch.stack([torch.zeros(2, 4), torch.ones(2, 4)], dim=0),
        root_indices=[0, 0],
        action_paths=[[0], [1]],
    )

    expected_repeat_factor = 1.0 / (math.sqrt(10.0) + 0.001)
    assert rewards[0].item() == pytest.approx(expected_repeat_factor)
    assert records[0]['episodic_pseudo_count'] == pytest.approx(10.0)
    assert records[0]['episodic_neighbor_count'] == 10
    assert records[0]['episodic_knn_warmup'] == 0.0
    assert rewards[1].item() == pytest.approx(1.0)


@pytest.mark.unittest
def test_ngu_knn_gate_leaves_rewards_unchanged_during_memory_warmup():
    memory = EpisodicLatentMemoryContext(
        keys=torch.zeros(3, 1, 2),
        transition_distance_scale=1.0,
    )
    gate = EpisodicLatentRewardGate(
        [memory], mode='ngu_knn', knn_k=10, include_path_memory=False
    )

    rewards, records = gate.apply(
        torch.tensor([2.0]),
        torch.zeros(1, 1, 2),
        root_indices=[0],
        action_paths=[[0]],
    )

    assert rewards.item() == pytest.approx(2.0)
    assert records[0]['episodic_knn_warmup'] == 1.0
