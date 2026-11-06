import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest
from easydict import EasyDict

from lzero.entry.train_ensemble_muzero import (
    _write_encoder_prior_grid_diagnostics,
    _write_encoder_prior_imagined_novelty_diagnostics,
)
from lzero.model.ensemble_muzero_model import EnsembleMuZeroModelMLP
from lzero.policy.scaling_transform import DiscreteSupport, InverseScalarTransform


@pytest.mark.unittest
def test_encoder_prior_grid_diagnostic_writes_pre_and_post_rows(tmp_path):
    model = EnsembleMuZeroModelMLP(
        ensemble_size=2,
        observation_shape=2,
        action_space_size=3,
        latent_state_dim=16,
        random_prior_scale=2.0,
        value_prior_backprop_to_latent=True,
        encoder_random_prior_scale=1.0,
        encoder_random_prior_output_norm=0.5,
        encoder_random_prior_input_mean=(-0.3, 0.0),
        encoder_random_prior_input_scale=(0.9, 0.07),
        encoder_prior_composed_latent_normalization='l2',
        encoder_prior_composed_latent_norm_scale=1.0,
        dynamics_encoder_latent_normalization='l2',
        dynamics_encoder_latent_norm_scale=1.0,
        dynamics_trainable_latent_normalization='l2',
        dynamics_trainable_latent_norm_scale=1.0,
        representation_last_linear_layer_init_zero=False,
        value_last_linear_layer_init_zero=False,
        state_norm=False,
        norm_type='LN',
    )
    value_inverse = InverseScalarTransform(DiscreteSupport(-300, 301))
    reward_inverse = InverseScalarTransform(DiscreteSupport(-300, 301))
    policy = SimpleNamespace(
        _collect_model=model,
        value_inverse_scalar_transform_handle=value_inverse,
        reward_inverse_scalar_transform_handle=reward_inverse,
        _value_calibration_offsets=lambda ensemble_size: np.zeros(ensemble_size, dtype=np.float32),
    )
    cfg = EasyDict(
        exp_name=str(tmp_path / 'run'),
        policy=EasyDict(
            device='cpu',
            discount_factor=0.997,
            collection_diagnostics=True,
            diagnostics_dir=str(tmp_path / 'diagnostics'),
            encoder_prior_grid_diagnostics=True,
            encoder_prior_grid_position_min=-0.6,
            encoder_prior_grid_position_max=-0.4,
            encoder_prior_grid_position_count=3,
            encoder_prior_grid_velocity_min=-0.01,
            encoder_prior_grid_velocity_max=0.01,
            encoder_prior_grid_velocity_count=3,
            model=EasyDict(observation_shape=2, ensemble_size=2, frame_stack_num=1),
        ),
    )

    for stage, update_visits in [('pre_learn', True), ('post_learn', False)]:
        _write_encoder_prior_grid_diagnostics(
            cfg,
            policy,
            [[], []],
            tb_logger=None,
            collection_phase=1,
            train_iter=10,
            diagnostic_stage=stage,
            update_visits=update_visits,
        )

    csv_path = tmp_path / 'diagnostics' / 'encoder_prior_state_grid.csv'
    with csv_path.open(newline='') as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 18
    assert {row['diagnostic_stage'] for row in rows} == {'pre_learn', 'post_learn'}
    assert max(float(row['composition_abs_max_error']) for row in rows) < 1e-6
    assert all(float(row['learned_latent_norm_mean']) == pytest.approx(1.0) for row in rows)
    assert all(float(row['prior_contribution_norm_mean']) == pytest.approx(0.5) for row in rows)
    assert all(float(row['full_latent_norm_mean']) == pytest.approx(1.0) for row in rows)
    assert any(float(row['prior_contribution_pair_l2']) > 0.0 for row in rows)
    assert any(float(row['max_action_direct_calibrated_qstd']) > 0.0 for row in rows)
    assert any(float(row['root_value_prior_effect_std']) > 0.0 for row in rows)
    assert any(float(row['max_action_direct_value_prior_effect_std']) > 0.0 for row in rows)
    for row in rows:
        total = np.asarray(json.loads(row['root_member_values']), dtype=np.float64)
        trainable = np.asarray(json.loads(row['root_member_trainable_values']), dtype=np.float64)
        prior_effect = np.asarray(json.loads(row['root_member_value_prior_effects']), dtype=np.float64)
        assert np.allclose(total, trainable + prior_effect, atol=1e-6)
    assert (tmp_path / 'diagnostics' / 'encoder_prior_grid_visit_counts.npz').exists()


@pytest.mark.unittest
def test_encoder_prior_imagined_novelty_diagnostic_writes_sequence_rows(tmp_path):
    model = EnsembleMuZeroModelMLP(
        ensemble_size=2,
        observation_shape=2,
        action_space_size=3,
        latent_state_dim=16,
        encoder_random_prior_scale=1.0,
        encoder_random_prior_input_mean=(-0.3, 0.0),
        encoder_random_prior_input_scale=(0.9, 0.07),
        representation_last_linear_layer_init_zero=False,
        value_last_linear_layer_init_zero=False,
        self_supervised_learning_loss=True,
        state_norm=False,
        norm_type='LN',
    )
    value_inverse = InverseScalarTransform(DiscreteSupport(-300, 301))
    policy = SimpleNamespace(
        _collect_model=model,
        value_inverse_scalar_transform_handle=value_inverse,
        _value_calibration_offsets=lambda ensemble_size: np.zeros(ensemble_size, dtype=np.float32),
    )
    cfg = EasyDict(
        exp_name=str(tmp_path / 'run'),
        policy=EasyDict(
            device='cpu',
            game_segment_length=200,
                collection_diagnostics=True,
                diagnostics_dir=str(tmp_path / 'diagnostics'),
                fixed_sequence_intrinsic_max_states=2,
                ssl_loss_weight=2.0,
                encoder_prior_imagined_novelty_diagnostics=True,
            encoder_prior_imagined_novelty_root_count=2,
            encoder_prior_imagined_novelty_bank_size=4,
            encoder_prior_imagined_novelty_history_limit=32,
            encoder_prior_imagined_novelty_min_bank_size=2,
            encoder_prior_imagined_novelty_depth=2,
            encoder_prior_imagined_novelty_knn_k=2,
            encoder_prior_imagined_novelty_batch_size=4,
            encoder_prior_grid_position_min=-1.2,
            encoder_prior_grid_position_max=0.6,
            encoder_prior_grid_position_count=3,
            encoder_prior_grid_velocity_min=-0.07,
            encoder_prior_grid_velocity_max=0.07,
            encoder_prior_grid_velocity_count=3,
            model=EasyDict(observation_shape=2, ensemble_size=2, frame_stack_num=1),
        ),
    )

    class Segment:
        def __init__(self):
            self.obs_segment = [
                np.asarray([-0.50, 0.000], dtype=np.float32),
                np.asarray([-0.49, 0.010], dtype=np.float32),
                np.asarray([-0.51, -0.010], dtype=np.float32),
                np.asarray([-0.48, 0.015], dtype=np.float32),
            ]
            self.action_segment = [2, 0, 2]
            self.reward_segment = [-1.0, -1.0, -1.0]

        def __len__(self):
            return len(self.action_segment)

    new_data = [
        [Segment()],
        [
            {
                'segment_length': 3,
                'collection_phase': 1,
                'collection_episode_index': 0,
                'timestep': 0,
            }
        ],
    ]
    for stage, update_history in [('pre_learn', True), ('post_learn', False)]:
        _write_encoder_prior_imagined_novelty_diagnostics(
            cfg,
            policy,
            new_data,
            tb_logger=None,
            collection_phase=1,
            train_iter=10,
            diagnostic_stage=stage,
            update_history=update_history,
        )

    csv_path = tmp_path / 'diagnostics' / 'encoder_prior_imagined_novelty.csv'
    with csv_path.open(newline='') as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2 * 2 * 9
    assert {row['diagnostic_stage'] for row in rows} == {'pre_learn', 'post_learn'}
    assert len({row['sequence'] for row in rows}) == 9
    finite_columns = [
        'raw_endpoint_knn_distance',
        'direct_latent_knn_mean',
        'imagined_latent_knn_mean',
        'imagined_direct_latent_l2_mean',
        'direct_projection_knn_mean',
        'imagined_prediction_projection_knn_mean',
        'imagined_direct_projection_cosine_mean',
        'direct_value_std',
        'imagined_value_std',
    ]
    assert all(np.isfinite(float(row[column])) for row in rows for column in finite_columns)
    history_path = tmp_path / 'diagnostics' / 'encoder_prior_observation_history.npz'
    with np.load(history_path, allow_pickle=False) as saved:
        assert saved['observations'].shape == (4, 2)
        assert int(saved['last_phase'][0]) == 1
