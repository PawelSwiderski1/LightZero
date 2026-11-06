import numpy as np
import pytest
import torch

from experiments.run_branching_rooms_spatial_exploration_world_model import (
    SpatialExplorationWorldModel,
    SpatialRollout,
)
from lzero.policy.spatial_exploration_tree import FrozenSpatialExplorationTree


@pytest.mark.unittest
def test_frozen_spatial_tree_tracks_real_memory_and_scores_action_paths(tmp_path):
    torch.manual_seed(0)
    model = SpatialExplorationWorldModel(
        observation_shape=(3, 28, 28),
        spatial_channels=8,
        key_dim=16,
        belief_dim=24,
        action_dim=4,
    )
    checkpoint = tmp_path / 'spatial.pt'
    torch.save(
        {
            'model': model.state_dict(),
            'epoch': 7,
            'arguments': {
                'horizon': 5,
                'spatial_channels': 8,
                'key_dim': 16,
                'belief_dim': 24,
                'action_dim': 4,
                'representation_mode': 'spatial_reconstruction',
                'coarse_reconstruction_size': 7,
            },
        },
        checkpoint,
    )
    rng_state = torch.random.get_rng_state()
    planner = FrozenSpatialExplorationTree(
        str(checkpoint),
        device='cpu',
        horizon=5,
        adaptive_scale=False,
    )
    assert torch.equal(torch.random.get_rng_state(), rng_state)

    first = torch.zeros(1, 3, 28, 28)
    roots = planner.prepare_roots(first, [3])
    assert roots[0].memory_keys.shape == (1, 16)
    scores = planner.score_paths(
        roots,
        root_indices=[0, 0],
        action_paths=[[0], [0, 1]],
        maximum_depth=5,
    )
    assert all(np.isfinite(row['novelty']) for row in scores)
    assert all(row['novelty'] >= 0.0 for row in scores)
    assert all(0.0 <= row['normalized_novelty'] <= 1.0 for row in scores)
    predicted_observations = planner.predict_path_observations(
        roots,
        root_indices=[0, 0],
        action_paths=[[0], [0, 1]],
        maximum_depth=5,
    )
    assert predicted_observations.shape == (2, 3, 28, 28)
    assert torch.all((0.0 <= predicted_observations) & (predicted_observations <= 1.0))

    modifier = planner.make_modifier_fn(
        roots,
        minimum_reward_factor=0.05,
        maximum_depth=2,
    )
    factors = modifier(
        root_indices=[0, 0, 0],
        action_paths=[[0], [0, 1], [0, 1, 2]],
        _next_latent=None,
    )
    assert np.all(factors[:2] >= 0.05)
    assert np.all(factors[:2] <= 1.0)
    assert factors[2] == pytest.approx(1.0)
    with pytest.raises(ValueError, match=r'\[0, 1\]'):
        planner.make_modifier_fn(
            roots,
            minimum_reward_factor=-0.01,
            maximum_depth=2,
        )

    planner.commit_actions(
        roots,
        actions=[0],
        planned_actions=[0],
        root_action_diagnostics=[
            [
                {'action': 0, 'intrinsic_path_reward_bonus': 0.2},
                {'action': 1, 'intrinsic_path_reward_bonus': 0.1},
                {'action': 2, 'intrinsic_path_reward_bonus': 0.0},
            ]
        ],
    )
    second = torch.ones(1, 3, 28, 28)
    next_roots = planner.prepare_roots(second, [3])
    assert next_roots[0].memory_keys.shape == (2, 16)
    records = planner.consume_records()
    assert len(records) == 1
    assert records[0]['selected_action'] == 0
    assert 1 <= records[0]['selected_bonus_rank'] <= 3
    assert records[0]['candidate_novelty_spread'] >= 0.0
    assert records[0]['realized_rgb_novel'] == 1.0
    assert records[0]['absolute_novelty_error'] >= 0.0

    planner.reset([3])
    reset_roots = planner.prepare_roots(first, [3])
    assert reset_roots[0].memory_keys.shape == (1, 16)


@pytest.mark.unittest
def test_online_spatial_tree_training_preserves_global_random_streams():
    torch.manual_seed(19)
    np.random.seed(23)
    planner = FrozenSpatialExplorationTree(
        None,
        device='cpu',
        horizon=2,
        online_training=True,
        initialization_seed=7,
        activation_updates=1,
        history_window=6,
        batch_size=1,
        roots_per_update=2,
        updates_per_collect=1,
    )
    observations = np.random.default_rng(5).integers(
        0,
        256,
        size=(7, 3, 28, 28),
        dtype=np.uint8,
    )
    episode = {
        'observations': observations,
        'actions': np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64),
    }
    torch_state = torch.random.get_rng_state()
    numpy_state = np.random.get_state()

    metrics = planner.train_online([episode], collection_phase=0)

    assert metrics['update_count'] == 1.0
    assert planner.active
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    current_numpy_state = np.random.get_state()
    assert current_numpy_state[0] == numpy_state[0]
    assert np.array_equal(current_numpy_state[1], numpy_state[1])
    assert current_numpy_state[2:] == numpy_state[2:]


@pytest.mark.unittest
def test_online_spatial_tree_mixes_uniform_and_error_prioritized_windows():
    planner = FrozenSpatialExplorationTree(
        None,
        device='cpu',
        horizon=2,
        online_training=True,
        initialization_seed=11,
        history_window=6,
        batch_size=4,
        pairwise_revisit_weight=1.0,
        memory_key_mode='belief_revisit',
        prioritized_replay_fraction=0.5,
        prioritized_replay_alpha=1.0,
        prioritized_replay_priority_mode='revisit_error',
    )
    assert planner.prioritized_replay_priority_mode == 'revisit_error'
    observations = np.zeros((9, 3, 28, 28), dtype=np.uint8)
    planner._online_replay = [
        {
            'observations': observations,
            'actions': np.arange(8, dtype=np.int64) % 3,
            'replay_id': 0,
        }
    ]
    hard_key = (0, 0, 8, 3)
    easy_key = (0, 0, 8, 1)
    planner._online_root_priorities = {
        hard_key: 100.0,
        easy_key: 1e-6,
    }

    batch, sampling = planner._sample_online_batch()

    assert batch['actions'].shape == (4, 8)
    assert sampling['full_episode_history']
    assert sampling['window_starts'] == [0, 0, 0, 0]
    assert sum(sampling['prioritized_flags']) == 2
    prioritized_windows_and_roots = [
        (*window_key, root)
        for window_key, root, prioritized in zip(
            sampling['window_keys'],
            sampling['requested_root_offsets'],
            sampling['prioritized_flags'],
        )
        if prioritized
    ]
    assert prioritized_windows_and_roots == [hard_key, hard_key]
    assert [
        priority
        for priority, prioritized in zip(
            sampling['requested_stored_priorities'],
            sampling['prioritized_flags'],
        )
        if prioritized
    ] == [100.0, 100.0]
    roots = planner._sample_online_roots(
        batch['actions'].shape[1], sampling['requested_root_offsets']
    )
    assert 3 in roots.tolist()


@pytest.mark.unittest
def test_belief_revisit_online_batch_uses_episode_level_ground_truth():
    planner = FrozenSpatialExplorationTree(
        None,
        device='cpu',
        horizon=2,
        online_training=True,
        initialization_seed=17,
        history_window=4,
        batch_size=1,
        pairwise_revisit_weight=1.0,
        memory_key_mode='belief_revisit',
    )
    observations = np.zeros((9, 3, 28, 28), dtype=np.uint8)
    for step in range(8):
        observations[step] = step + 1
    observations[8] = observations[0]
    planner._online_replay = [
        {
            'observations': observations,
            'actions': np.arange(8, dtype=np.int64) % 3,
            'replay_id': 0,
        }
    ]

    batch, sampling = planner._sample_online_batch()

    assert batch['actions'].shape == (1, 8)
    assert sampling['window_keys'] == [(0, 0, 8)]
    assert sampling['full_episode_history']
    assert batch['state_ids'][0, 0] == batch['state_ids'][0, 8]


@pytest.mark.unittest
def test_belief_revisit_window_scope_remains_an_explicit_ablation():
    planner = FrozenSpatialExplorationTree(
        None,
        device='cpu',
        horizon=2,
        online_training=True,
        initialization_seed=19,
        history_window=4,
        batch_size=1,
        pairwise_revisit_weight=1.0,
        memory_key_mode='belief_revisit',
        revisit_history_scope='window',
    )
    observations = np.zeros((9, 3, 28, 28), dtype=np.uint8)
    planner._online_replay = [
        {
            'observations': observations,
            'actions': np.arange(8, dtype=np.int64) % 3,
            'replay_id': 0,
        }
    ]

    batch, sampling = planner._sample_online_batch()

    assert batch['actions'].shape == (1, 4)
    assert not sampling['full_episode_history']
    assert sampling['window_keys'][0][2] == 4


@pytest.mark.unittest
def test_belief_spatial_tree_stores_history_conditioned_posterior_keys(tmp_path):
    torch.manual_seed(29)
    model = SpatialExplorationWorldModel(
        observation_shape=(3, 28, 28),
        spatial_channels=8,
        key_dim=16,
        belief_dim=24,
        action_dim=4,
        memory_key_mode='belief',
    )
    checkpoint = tmp_path / 'belief_spatial.pt'
    torch.save(
        {
            'model': model.state_dict(),
            'epoch': 0,
            'arguments': {
                'horizon': 2,
                'spatial_channels': 8,
                'key_dim': 16,
                'belief_dim': 24,
                'action_dim': 4,
                'representation_mode': 'spatial_reconstruction',
                'memory_key_mode': 'belief',
            },
        },
        checkpoint,
    )
    planner = FrozenSpatialExplorationTree(
        str(checkpoint),
        device='cpu',
        horizon=2,
        adaptive_scale=False,
    )
    observation = torch.zeros(1, 3, 28, 28)

    roots = planner.prepare_roots(observation, [0])
    first_memory_key = roots[0].memory_keys[0].clone()
    planner.commit_actions(
        roots,
        actions=[1],
        root_action_diagnostics=[[{'action': 1}]],
    )
    next_roots = planner.prepare_roots(observation, [0])
    second_memory_key = next_roots[0].memory_keys[1]
    records = planner.consume_records()

    assert planner.model.memory_key_mode == 'belief'
    assert next_roots[0].memory_keys.shape == (2, 16)
    assert not torch.allclose(first_memory_key, second_memory_key)
    assert records[0]['realized_rgb_novel'] == 0.0


@pytest.mark.unittest
def test_belief_to_observation_tree_stores_observation_keys(tmp_path):
    torch.manual_seed(31)
    model = SpatialExplorationWorldModel(
        observation_shape=(3, 28, 28),
        spatial_channels=8,
        key_dim=16,
        belief_dim=24,
        action_dim=4,
        memory_key_mode='belief_to_observation',
    )
    checkpoint = tmp_path / 'belief_to_observation_spatial.pt'
    torch.save(
        {
            'model': model.state_dict(),
            'epoch': 0,
            'arguments': {
                'horizon': 2,
                'spatial_channels': 8,
                'key_dim': 16,
                'belief_dim': 24,
                'action_dim': 4,
                'representation_mode': 'spatial_reconstruction',
                'memory_key_mode': 'belief_to_observation',
            },
        },
        checkpoint,
    )
    planner = FrozenSpatialExplorationTree(
        str(checkpoint),
        device='cpu',
        horizon=2,
        adaptive_scale=False,
    )
    observation = torch.zeros(1, 3, 28, 28)

    roots = planner.prepare_roots(observation, [0])
    first_memory_key = roots[0].memory_keys[0].clone()
    planner.commit_actions(
        roots,
        actions=[1],
        root_action_diagnostics=[[{'action': 1}]],
    )
    next_roots = planner.prepare_roots(observation, [0])
    second_memory_key = next_roots[0].memory_keys[1]
    records = planner.consume_records()

    assert planner.model.memory_key_mode == 'belief_to_observation'
    assert torch.allclose(first_memory_key, second_memory_key, atol=1e-6)
    assert records[0]['realized_raw_novelty'] == pytest.approx(0.0, abs=1e-6)
    assert records[0]['realized_rgb_novel'] == 0.0


@pytest.mark.unittest
def test_belief_revisit_tree_stores_beliefs_and_scores_branch_memory(tmp_path):
    torch.manual_seed(37)
    model = SpatialExplorationWorldModel(
        observation_shape=(3, 28, 28),
        spatial_channels=8,
        key_dim=16,
        belief_dim=24,
        action_dim=4,
        memory_key_mode='belief_revisit',
    )
    checkpoint = tmp_path / 'belief_revisit_spatial.pt'
    torch.save(
        {
            'model': model.state_dict(),
            'epoch': 0,
            'arguments': {
                'horizon': 2,
                'spatial_channels': 8,
                'key_dim': 16,
                'belief_dim': 24,
                'action_dim': 4,
                'representation_mode': 'spatial_reconstruction',
                'memory_key_mode': 'belief_revisit',
            },
        },
        checkpoint,
    )
    planner = FrozenSpatialExplorationTree(
        str(checkpoint),
        device='cpu',
        horizon=2,
        adaptive_scale=True,
    )
    observation = torch.zeros(1, 3, 28, 28)
    roots = planner.prepare_roots(observation, [0])

    assert roots[0].memory_beliefs.shape == (1, 24)
    scores = planner.score_paths(
        roots,
        root_indices=[0, 0],
        action_paths=[[0], [0, 1]],
        maximum_depth=2,
    )
    assert all(0.0 <= row['novelty'] <= 1.0 for row in scores)
    assert all(row['novelty'] == row['normalized_novelty'] for row in scores)

    planner.commit_actions(
        roots,
        actions=[0],
        root_action_diagnostics=[[{'action': 0}]],
    )
    next_roots = planner.prepare_roots(observation, [0])
    records = planner.consume_records()

    assert next_roots[0].memory_beliefs.shape == (2, 24)
    assert records[0]['realized_rgb_novel'] == 0.0
    assert 0.0 <= records[0]['realized_raw_novelty'] <= 1.0


@pytest.mark.unittest
def test_observation_key_scoring_uses_real_and_branch_local_key_memory(
        tmp_path, monkeypatch
):
    torch.manual_seed(41)
    model = SpatialExplorationWorldModel(
        observation_shape=(3, 28, 28),
        spatial_channels=8,
        key_dim=16,
        belief_dim=24,
        action_dim=4,
        memory_key_mode='belief_revisit',
    )
    checkpoint = tmp_path / 'observation_key_scoring.pt'
    torch.save(
        {
            'model': model.state_dict(),
            'epoch': 0,
            'arguments': {
                'horizon': 2,
                'spatial_channels': 8,
                'key_dim': 16,
                'belief_dim': 24,
                'action_dim': 4,
                'representation_mode': 'spatial_reconstruction',
                'memory_key_mode': 'belief_revisit',
            },
        },
        checkpoint,
    )
    planner = FrozenSpatialExplorationTree(
        str(checkpoint),
        device='cpu',
        horizon=2,
        adaptive_scale=False,
        revisit_scoring_mode='observation_key',
    )
    observation = torch.zeros(1, 3, 28, 28)
    roots = planner.prepare_roots(observation, [0])
    real_key = roots[0].observation_memory_keys[0].clone()

    def fail_revisit_classifier(*_args, **_kwargs):
        raise AssertionError('belief-pair classifier must not be used')

    def repeated_imagined_key_rollout(latent, belief, actions):
        batch_size, depth = actions.shape
        predicted_key = -real_key
        observation_keys = predicted_key.reshape(1, 1, -1).expand(
            batch_size, depth, -1
        )
        latents = latent[:, None].expand(
            batch_size, depth, *latent.shape[1:]
        )
        beliefs = belief[:, None].expand(batch_size, depth, -1)
        keys = planner.model.keys_from_beliefs(
            beliefs.reshape(-1, beliefs.shape[-1])
        ).reshape(batch_size, depth, -1)
        return SpatialRollout(
            latents=latents,
            observation_keys=observation_keys,
            keys=keys,
            beliefs=beliefs,
        )

    monkeypatch.setattr(
        planner.model, 'revisit_novelty', fail_revisit_classifier
    )
    monkeypatch.setattr(
        planner.model, 'rollout', repeated_imagined_key_rollout
    )
    scores = planner.score_paths(
        roots,
        root_indices=[0, 0],
        action_paths=[[0], [0, 1]],
        maximum_depth=2,
    )

    assert scores[0]['raw_novelty'] == pytest.approx(2.0, abs=1e-6)
    assert scores[1]['raw_novelty'] == pytest.approx(0.0, abs=1e-6)

    planner.commit_actions(
        roots,
        actions=[0],
        root_action_diagnostics=[[{'action': 0}]],
    )
    next_roots = planner.prepare_roots(observation, [0])
    records = planner.consume_records()

    assert next_roots[0].observation_memory_keys.shape == (2, 16)
    assert records[0]['realized_raw_novelty'] == pytest.approx(0.0, abs=1e-6)
