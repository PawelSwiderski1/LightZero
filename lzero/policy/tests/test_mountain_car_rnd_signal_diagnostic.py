from argparse import Namespace

import numpy as np
import pytest
import torch

from experiments.diagnose_mountain_car_rnd_signal import (
    RNDSignalModel,
    build_phase_data,
    load_episodes,
    scripted_trajectory,
)


def _model(mode: str, learning_rate: float = 1e-3) -> RNDSignalModel:
    return RNDSignalModel(
        mode=mode,
        seed=0,
        hidden_dim=32,
        feature_dim=16,
        learning_rate=learning_rate,
        target_first_layer_gain=1.0,
        device=torch.device('cpu'),
    )


@pytest.mark.unittest
def test_legacy_rnd_preprocessing_collapses_distinct_negative_position_states():
    observations = torch.tensor(
        [[-0.5, 0.0], [-0.3, 0.05], [-1.0, -0.05]], dtype=torch.float32
    )
    legacy = _model('legacy_layer_norm').preprocess(observations)
    fixed = _model('fixed_bounds').preprocess(observations)

    assert torch.allclose(legacy[0], legacy[1], atol=1e-3)
    assert torch.allclose(legacy[1], legacy[2], atol=1e-3)
    assert not torch.allclose(fixed[0], fixed[1], atol=1e-3)
    assert not torch.allclose(fixed[1], fixed[2], atol=1e-3)


@pytest.mark.unittest
def test_online_rnd_training_reduces_error_on_fresh_observations():
    model = _model('fixed_bounds', learning_rate=1e-2)
    observations = np.asarray(
        [[-0.55, 0.0], [-0.54, 0.005], [-0.53, 0.01], [-0.52, 0.015]], dtype=np.float32
    )
    before = float(model.errors(observations).mean())

    stats = model.train_fresh(
        observations,
        update_proportion=1.0,
        candidate_batch_size=4,
        train_passes=100,
        random_seed=7,
    )
    after = float(model.errors(observations).mean())

    assert stats['train_selected_count'] == 400
    assert stats['train_update_count'] == 100
    assert after < before * 0.1


@pytest.mark.unittest
def test_saved_episode_replay_uses_previous_phases_as_density_snapshot(tmp_path):
    episode_dir = tmp_path / 'diagnostics' / 'intrinsic_episodes'
    episode_dir.mkdir(parents=True)
    observations = np.asarray(
        [[-0.5, 0.0], [-0.49, 0.01], [-0.48, 0.01]], dtype=np.float32
    )
    actions = np.asarray([2, 2], dtype=np.int64)
    for phase in [0, 1]:
        np.savez(
            episode_dir / f'intrinsic-{phase:04d}_score-0p000_phase-{phase}_env-0_episode-0.npz',
            obs=observations,
            actions=actions,
        )

    args = Namespace(
        position_bin_width=0.02,
        velocity_bin_width=0.002,
        neighbor_radius=1,
    )
    phases = build_phase_data(load_episodes(tmp_path), args)

    assert len(phases) == 2
    assert np.allclose(phases[0].density_rewards, 1.0)
    assert np.all(phases[1].density_rewards < 1.0)
    assert np.all(np.isfinite(phases[1].nearest_prior_distances))
    assert np.allclose(phases[1].nearest_prior_distances, 0.0)


@pytest.mark.unittest
def test_scripted_coordinated_swing_reaches_goal_while_noop_does_not():
    noop = scripted_trajectory(-0.5, 'noop')
    swing = scripted_trajectory(-0.5, 'coordinated_swing')

    assert float(noop[:, 0].max()) < 0.5
    assert float(swing[:, 0].max()) >= 0.5
