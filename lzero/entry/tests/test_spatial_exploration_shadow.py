from types import SimpleNamespace

import numpy as np
import torch

from experiments.run_branching_rooms_action_novelty import load_dataset
from experiments.run_branching_rooms_spatial_exploration_world_model import (
    SpatialExplorationWorldModel,
)
from lzero.entry.spatial_exploration_shadow import (
    SpatialExplorationShadowEvaluator,
    _binary_auc,
    complete_collection_episodes,
)


def _segment(start_value, actions, states):
    observation_count = len(actions) + 1
    observations = [
        np.full((3, 28, 28), start_value + offset, dtype=np.float32)
        for offset in range(observation_count)
    ]
    return SimpleNamespace(
        action_segment=list(actions),
        reward_segment=[0.0] * len(actions),
        obs_segment=observations,
        environment_initial_state=tuple(states[0]),
        environment_state_segment=[tuple(state) for state in states[1:]],
    )


def test_complete_collection_episodes_reassembles_segmented_episode():
    first = _segment(0, [2, 0], [(1, 1, 0), (2, 1, 0), (2, 1, 1)])
    second = _segment(2, [2, 2], [(2, 1, 1), (2, 2, 1), (2, 3, 1)])
    metadata = [
        {
            "env_id": 0,
            "env_episode_index": 7,
            "episode_segment_index": 0,
            "segment_length": 2,
            "done": False,
        },
        {
            "env_id": 0,
            "env_episode_index": 7,
            "episode_segment_index": 1,
            "segment_length": 2,
            "done": True,
            "collection_episode_index": 3,
        },
    ]

    episodes = complete_collection_episodes([[first, second], metadata], 1)

    assert len(episodes) == 1
    assert episodes[0]["observations"].shape == (5, 3, 28, 28)
    assert episodes[0]["actions"].tolist() == [2, 0, 2, 2]
    assert episodes[0]["environment_states"][:, :2].tolist() == [
        [1, 1],
        [2, 1],
        [2, 1],
        [2, 2],
        [2, 3],
    ]


def test_shadow_evaluator_is_frozen_and_scores_only_executed_suffixes(tmp_path):
    model = SpatialExplorationWorldModel(
        spatial_channels=8,
        key_dim=16,
        belief_dim=32,
        action_dim=4,
    )
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": 4,
            "arguments": {
                "horizon": 2,
                "spatial_channels": 8,
                "key_dim": 16,
                "belief_dim": 32,
                "action_dim": 4,
            },
        },
        checkpoint,
    )
    evaluator = SpatialExplorationShadowEvaluator(
        str(checkpoint),
        "cpu",
        horizon=2,
        roots_per_episode=3,
        root_batch_size=2,
    )
    observations = np.random.default_rng(3).integers(
        0, 256, size=(7, 3, 28, 28), dtype=np.uint8
    )
    episode = {
        "observations": observations,
        "actions": np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64),
        "environment_states": np.asarray(
            [[index, 0, 0] for index in range(7)], dtype=np.int64
        ),
        "env_id": 0,
        "env_episode_index": 0,
        "collection_episode_index": 0,
    }

    rows, root_rows = evaluator.evaluate_episodes([episode], collection_phase=2)

    assert len(root_rows) == 3
    assert len(rows) == 6
    assert {row["depth"] for row in rows} == {1, 2}
    assert all(not parameter.requires_grad for parameter in evaluator.model.parameters())
    assert all(0.0 <= row["predicted_novelty"] <= 2.0 for row in rows)
    assert all(len(row["action_prefix"]) > 0 for row in rows)


def test_live_model_evaluator_does_not_freeze_online_parameters():
    model = SpatialExplorationWorldModel(
        spatial_channels=8,
        key_dim=16,
        belief_dim=32,
        action_dim=4,
    )
    evaluator = SpatialExplorationShadowEvaluator.from_live_model(
        model,
        "cpu",
        horizon=2,
        roots_per_episode=3,
        root_batch_size=2,
    )
    observations = np.random.default_rng(17).integers(
        0, 256, size=(7, 3, 28, 28), dtype=np.uint8
    )
    episode = {
        "observations": observations,
        "actions": np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64),
        "environment_states": np.asarray(
            [[index, 0, 0] for index in range(7)], dtype=np.int64
        ),
        "env_id": 0,
        "env_episode_index": 0,
        "collection_episode_index": 0,
    }

    rows, root_rows = evaluator.evaluate_episodes([episode], collection_phase=0)

    assert evaluator.model is model
    assert len(rows) == 6
    assert len(root_rows) == 3
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_shadow_evaluator_uses_belief_keys_and_belief_decoding(tmp_path):
    model = SpatialExplorationWorldModel(
        spatial_channels=8,
        key_dim=16,
        belief_dim=32,
        action_dim=4,
        memory_key_mode="belief",
    )
    checkpoint = tmp_path / "belief_model.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": 0,
            "arguments": {
                "horizon": 2,
                "spatial_channels": 8,
                "key_dim": 16,
                "belief_dim": 32,
                "action_dim": 4,
                "memory_key_mode": "belief",
            },
        },
        checkpoint,
    )
    evaluator = SpatialExplorationShadowEvaluator(
        str(checkpoint),
        "cpu",
        horizon=2,
        roots_per_episode=3,
        root_batch_size=2,
    )
    observations = np.zeros((7, 3, 28, 28), dtype=np.uint8)
    episode = {
        "observations": observations,
        "actions": np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64),
        "environment_states": np.asarray(
            [[index, 0, 0] for index in range(7)], dtype=np.int64
        ),
        "env_id": 0,
        "env_episode_index": 0,
        "collection_episode_index": 0,
    }

    rows, root_rows = evaluator.evaluate_episodes([episode], collection_phase=0)

    assert evaluator.model.memory_key_mode == "belief"
    assert len(rows) == 6
    assert len(root_rows) == 3
    assert all(np.isfinite(row["rollout_rgb_mae"]) for row in rows)


def test_belief_revisit_ablation_substitutes_query_and_memory():
    model = SpatialExplorationWorldModel(
        spatial_channels=8,
        key_dim=16,
        belief_dim=32,
        action_dim=4,
        memory_key_mode="belief_revisit",
    )
    evaluator = SpatialExplorationShadowEvaluator.from_live_model(
        model,
        "cpu",
        horizon=2,
        roots_per_episode=3,
        root_batch_size=2,
        belief_ablation=True,
    )
    observations = np.random.default_rng(29).integers(
        0, 256, size=(7, 3, 28, 28), dtype=np.uint8
    )
    episode = {
        "observations": observations,
        "actions": np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64),
        "environment_states": np.asarray(
            [[index, 0, 0] for index in range(7)], dtype=np.int64
        ),
    }

    rows, _ = evaluator.evaluate_episodes([episode], collection_phase=0)

    assert len(rows) == 6
    assert all(
        np.isfinite(row["real_query_real_memory_novelty"])
        and np.isfinite(row["predicted_query_real_memory_novelty"])
        and np.isfinite(row["real_query_predicted_memory_novelty"])
        for row in rows
    )
    for row in rows:
        if row["depth"] == 1:
            assert row["predicted_novelty"] == row[
                "predicted_query_real_memory_novelty"
            ]
            assert row["real_query_real_memory_novelty"] == row[
                "real_query_predicted_memory_novelty"
            ]


def test_shadow_evaluator_supports_decoder_free_inverse_embedding(tmp_path):
    model = SpatialExplorationWorldModel(
        spatial_channels=8,
        key_dim=16,
        belief_dim=32,
        action_dim=4,
        representation_mode="inverse_dynamics",
    )
    checkpoint = tmp_path / "inverse_model.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": 0,
            "arguments": {
                "horizon": 2,
                "spatial_channels": 8,
                "key_dim": 16,
                "belief_dim": 32,
                "action_dim": 4,
                "representation_mode": "inverse_dynamics",
            },
        },
        checkpoint,
    )
    evaluator = SpatialExplorationShadowEvaluator(
        str(checkpoint),
        "cpu",
        horizon=2,
        roots_per_episode=2,
        root_batch_size=2,
    )
    observations = np.random.default_rng(5).integers(
        0, 256, size=(7, 3, 28, 28), dtype=np.uint8
    )
    episode = {
        "observations": observations,
        "actions": np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64),
        "environment_states": np.asarray(
            [[index, 0, 0] for index in range(7)], dtype=np.int64
        ),
        "env_id": 0,
        "env_episode_index": 0,
        "collection_episode_index": 0,
    }

    rows, _ = evaluator.evaluate_episodes([episode], collection_phase=0)

    assert len(rows) == 4
    assert all(np.isnan(row["rollout_rgb_mae"]) for row in rows)
    assert all(np.isfinite(row["predicted_novelty"]) for row in rows)


def test_bounded_root_beliefs_match_independent_suffix_encoding(tmp_path):
    torch.manual_seed(7)
    model = SpatialExplorationWorldModel(
        spatial_channels=8,
        key_dim=16,
        belief_dim=32,
        action_dim=4,
    )
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": 0,
            "arguments": {
                "horizon": 2,
                "spatial_channels": 8,
                "key_dim": 16,
                "belief_dim": 32,
                "action_dim": 4,
            },
        },
        checkpoint,
    )
    evaluator = SpatialExplorationShadowEvaluator(
        str(checkpoint),
        "cpu",
        horizon=2,
        dynamics_history_window=4,
    )
    observations = torch.rand(9, 3, 28, 28)
    actions = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
    roots = torch.tensor([0, 3, 4, 8])
    with torch.inference_mode():
        latents = evaluator.model.encode_observations(observations)
        keys = evaluator.model.keys_from_latents(latents)
        actual = evaluator._bounded_root_beliefs(keys, actions, roots)
        expected = []
        for root in roots.tolist():
            start = max(0, root - 3)
            encoded = evaluator.model.encode_trajectory(
                observations[start : root + 1].unsqueeze(0),
                actions[start:root].unsqueeze(0),
            )
            expected.append(encoded.beliefs[0, -1])

    assert torch.allclose(actual, torch.stack(expected), atol=1e-6)


def test_shadow_exports_complete_split_trajectory_dataset(tmp_path):
    model = SpatialExplorationWorldModel(
        spatial_channels=8,
        key_dim=16,
        belief_dim=32,
        action_dim=4,
    )
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": 0,
            "arguments": {
                "horizon": 2,
                "spatial_channels": 8,
                "key_dim": 16,
                "belief_dim": 32,
                "action_dim": 4,
            },
        },
        checkpoint,
    )
    dataset_path = tmp_path / "experienced_trajectories.npz"
    evaluator = SpatialExplorationShadowEvaluator(
        str(checkpoint),
        "cpu",
        horizon=2,
        export_dataset_path=str(dataset_path),
        export_split_counts=(1, 1, 1, 1),
    )
    episodes = []
    for episode_index in range(4):
        observations = np.zeros((7, 3, 28, 28), dtype=np.uint8)
        observations[:, :, 0, 0] = (
            np.arange(7)[:, None] + episode_index * 20
        )
        episodes.append(
            {
                "observations": observations,
                "actions": np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64),
                "environment_states": np.asarray(
                    [[step, episode_index, step % 4] for step in range(7)],
                    dtype=np.int64,
                ),
                "env_id": episode_index,
                "env_episode_index": 0,
            }
        )

    result = evaluator.export_episodes(episodes)
    bundle = load_dataset(dataset_path)

    assert result == str(dataset_path)
    assert bundle.observations.shape == (4, 7, 3, 28, 28)
    assert bundle.actions.shape == (4, 6)
    assert bundle.observation_targets.shape == (4, 5, 2)
    assert bundle.pose_targets.shape == (4, 5, 2)
    assert sorted(bundle.split_ids.tolist()) == [0, 1, 2, 3]
    assert bundle.metadata["physical_position_used_for_training"] is False


def test_binary_auc_handles_ties():
    labels = np.asarray([0, 0, 1, 1])
    assert _binary_auc(labels, np.asarray([0.0, 0.5, 0.5, 1.0])) == 0.875
