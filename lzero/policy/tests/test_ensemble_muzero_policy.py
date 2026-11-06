import pytest
import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict

from lzero.mcts.buffer.game_buffer_muzero import MuZeroGameBuffer
from lzero.model.common import MZNetworkOutput
from lzero.policy.ensemble_muzero import EnsembleMuZeroPolicy
from lzero.policy.episodic_latent_memory import EpisodicLatentMemoryContext
from lzero.policy.rnd_landscape_query import MountainCarRNDLandscapeQuery
from lzero.policy.muzero import (
    _categorical_value_prior_training_terms,
    _latent_variance_covariance_loss,
    _raw_latent_consistency_terms,
    _representation_prediction_parameter_groups,
)
from lzero.policy.scaling_transform import DiscreteSupport
from lzero.policy.utils import prepare_obs
from lzero.model.ensemble_muzero_model import EnsembleMuZeroModelMLP


@pytest.mark.unittest
@pytest.mark.parametrize('scalar_prior', [-2.0, 0.0, 2.0])
def test_categorical_value_prior_training_moves_prediction_toward_target(scalar_prior):
    torch.manual_seed(7)
    model = EnsembleMuZeroModelMLP(
        ensemble_size=2,
        random_prior_scale=5.0,
        observation_shape=4,
        action_space_size=3,
        latent_state_dim=16,
    )
    support = DiscreteSupport(-300.0, 301.0, 1.0)
    target = torch.tensor([-100.0])
    prior = torch.tensor([[scalar_prior]])
    logits = nn.Parameter(torch.randn(1, model.value_support_size) * 0.1)
    optimizer = torch.optim.Adam([logits], lr=3e-3)

    with torch.no_grad():
        initial_error = abs(float(model._value_logits_to_scalar(logits) + prior) - float(target))
    for _ in range(750):
        final_scalar, loss, _ = _categorical_value_prior_training_terms(
            logits,
            prior,
            target,
            support,
            model._value_logits_to_scalar,
            scalar_loss_weight=1.0,
        )
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()

    final_error = abs(float(final_scalar.detach()) - float(target))
    assert final_error < initial_error * 0.2
    assert final_error < 10.0


class _FakeIntrinsicBootstrapModel:

    def __init__(self):
        self.training = True
        self.member_ids = None

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

    def set_active_members(self, member_ids):
        self.member_ids = member_ids

    def set_active_member(self, member_id):
        self.member_id = member_id

    def initial_inference(self, obs, member_id=None):
        return EasyDict(latent_state=obs)

    def intrinsic_value_from_latent(self, latent_state, member_id=None):
        member_offsets = torch.as_tensor(member_id, dtype=torch.float32, device=latent_state.device) * 10.0
        return latent_state[:, :1] + member_offsets.unsqueeze(-1)


class _FakeOneStepQstdModel(nn.Module):

    def initial_inference(self, obs, member_id=None):
        batch_size = obs.shape[0]
        zeros = torch.zeros((batch_size, 1), dtype=obs.dtype, device=obs.device)
        return MZNetworkOutput(zeros, zeros, zeros, obs)

    def recurrent_inference(self, latent_state, action, member_id=None):
        member = torch.as_tensor(member_id, dtype=latent_state.dtype, device=latent_state.device).view(-1, 1)
        action = action.to(dtype=latent_state.dtype).view(-1, 1)
        # Between the two members, reward differs by 2 and value differs by
        # 4 * action. At gamma=0.5, population Qstd is 1 + action.
        reward = 2.0 * member
        value = latent_state[:, :1] + 4.0 * member * action
        policy = torch.zeros((latent_state.shape[0], 3), dtype=latent_state.dtype, device=latent_state.device)
        return MZNetworkOutput(value, reward, policy, latent_state)

    def set_active_member(self, member_id):
        self.active_member = member_id


class _FakeIntrinsicMember(nn.Module):

    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(2, 2)
        self.intrinsic_reward_head = nn.Linear(2, 1)
        self.intrinsic_value_head = nn.Linear(2, 1)


class _FakeIntrinsicEnsemble(nn.Module):

    def __init__(self):
        super().__init__()
        self.members = nn.ModuleList([_FakeIntrinsicMember(), _FakeIntrinsicMember()])


class _FakeTargetModel:

    def __init__(self):
        self.updated = False

    def update(self, state_dict):
        self.updated = True


class _FakeDynamicsPriorAlignmentModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=False)
        nn.init.zeros_(self.linear.weight)
        self.dynamics_prior_networks = nn.ModuleList([nn.Identity()])
        self.member_ids = None

    def set_active_members(self, member_ids):
        self.member_ids = member_ids

    def initial_inference(self, obs, member_id=None):
        return EasyDict(latent_state=obs[:, :2])

    def recurrent_inference(self, latent_state, action, member_id=None):
        return EasyDict(latent_state=self.linear(latent_state))


class _FakeSeparateExplorationTreeModel:

    def __init__(self):
        self.batch_sizes = []

    def eval(self):
        return self

    def recurrent_inference(self, latent_state, action, member_id=None):
        self.batch_sizes.append(int(latent_state.shape[0]))
        action = action.float().view(-1, 1)
        member_id = torch.as_tensor(member_id, dtype=torch.float32, device=latent_state.device).view(-1, 1)
        next_latent_state = latent_state + action + member_id
        reward = action * (member_id + 1.0)
        value = action * (member_id + 1.0)
        policy_logits = torch.zeros(latent_state.shape[0], 2, device=latent_state.device)
        return MZNetworkOutput(value, reward, policy_logits, next_latent_state)


class _FakeValueCalibrationModel:

    def __init__(self):
        self.training = True

    def eval(self):
        self.training = False
        return self

    def train(self, mode=True):
        self.training = bool(mode)
        return self

    def set_active_member(self, member_id):
        self.member_id = int(member_id)

    def initial_inference(self, obs, member_id=None):
        member_id = torch.as_tensor(member_id, dtype=torch.float32, device=obs.device).view(-1, 1)
        value = obs[:, :1].float() + 10.0 * member_id
        return EasyDict(value=value)


class _FakeStateValueDisagreementTargetModel:

    def __init__(self):
        self.training = True

    def eval(self):
        self.training = False
        return self

    def train(self, mode=True):
        self.training = bool(mode)
        return self

    def set_active_members(self, member_ids):
        self.member_ids = member_ids

    def set_active_member(self, member_id):
        self.member_id = int(member_id)

    def initial_inference(self, obs, member_id=None):
        member_id = torch.as_tensor(member_id, dtype=torch.float32, device=obs.device).view(-1, 1)
        value = obs[:, :1].float() * (1.0 + 2.0 * member_id)
        return EasyDict(value=value)


@pytest.mark.unittest
def test_latent_variance_regularization_is_computed_per_ensemble_member():
    # Across the entire batch these two collapsed members appear to have high
    # variance. Per-member grouping must expose that both representations are
    # individually constant.
    latent = torch.tensor(
        [[-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
    )
    ungrouped_variance, _, _ = _latent_variance_covariance_loss(
        latent, variance_target=1.0, eps=1e-4
    )
    grouped_variance, grouped_covariance, stats = _latent_variance_covariance_loss(
        latent,
        group_ids=[0, 0, 0, 1, 1, 1],
        variance_target=1.0,
        eps=1e-4,
    )

    assert ungrouped_variance.item() == pytest.approx(0.0, abs=1e-6)
    assert grouped_variance.item() == pytest.approx(0.99, abs=1e-6)
    assert grouped_covariance.item() == pytest.approx(0.0, abs=1e-6)
    assert stats['group_count'] == 2
    assert stats['sample_count'] == 6


@pytest.mark.unittest
def test_representation_prediction_weight_decay_parameter_split():
    model = EnsembleMuZeroModelMLP(
        ensemble_size=2,
        observation_shape=2,
        action_space_size=3,
        latent_state_dim=16,
        self_supervised_learning_loss=True,
        state_norm=False,
        norm_type='LN',
        ssl_norm_type='BN',
    )
    no_decay, regular_decay = _representation_prediction_parameter_groups(model)
    no_decay_ids = {id(parameter) for parameter in no_decay}
    regular_decay_ids = {id(parameter) for parameter in regular_decay}

    assert no_decay_ids
    assert regular_decay_ids
    assert no_decay_ids.isdisjoint(regular_decay_ids)
    assert no_decay_ids | regular_decay_ids == {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    for member in model.members:
        assert all(id(parameter) in no_decay_ids for parameter in member.representation_network.parameters())
        assert all(id(parameter) in no_decay_ids for parameter in member.prediction_network.parameters())
        assert all(id(parameter) in regular_decay_ids for parameter in member.dynamics_network.parameters())
        assert all(id(parameter) in regular_decay_ids for parameter in member.projection.parameters())
        assert all(id(parameter) in regular_decay_ids for parameter in member.prediction_head.parameters())


@pytest.mark.unittest
def test_raw_latent_consistency_uses_an_exact_detached_target():
    predicted = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
    )
    target = torch.tensor(
        [[2.0, 0.0], [1.0, 5.0]], requires_grad=True
    )

    terms = _raw_latent_consistency_terms(predicted, target)

    assert torch.allclose(terms['loss'], torch.tensor([2.5, 2.5]))
    assert torch.allclose(terms['pred_l2'], predicted.detach().norm(dim=1))
    assert torch.allclose(terms['target_l2'], target.detach().norm(dim=1))

    terms['loss'].sum().backward()
    assert predicted.grad is not None
    assert torch.count_nonzero(predicted.grad).item() > 0
    assert target.grad is None


@pytest.mark.unittest
def test_raw_latent_ssl_prepares_future_observations_without_projector_ssl():
    cfg = EasyDict(
        dict(
            device='cpu',
            raw_latent_ssl_loss_weight=1.0,
            model=dict(
                model_type='mlp',
                observation_shape=2,
                frame_stack_num=1,
                image_channel=1,
                self_supervised_learning_loss=False,
            ),
        )
    )
    observations = np.arange(12, dtype=np.float32).reshape(2, 6)

    root, future = prepare_obs(observations, cfg)

    assert torch.equal(root, torch.tensor([[0.0, 1.0], [6.0, 7.0]]))
    assert torch.equal(future, torch.tensor([[2.0, 3.0, 4.0, 5.0], [8.0, 9.0, 10.0, 11.0]]))


@pytest.mark.unittest
def test_ensemble_raw_latent_ssl_runs_inside_the_main_learn_step():
    cfg = EnsembleMuZeroPolicy.default_config()
    cfg.cuda = False
    cfg.device = 'cpu'
    cfg.batch_size = 2
    cfg.num_unroll_steps = 2
    cfg.optim_type = 'Adam'
    cfg.learning_rate = 3e-4
    cfg.piecewise_decay_lr_scheduler = False
    cfg.use_priority = False
    cfg.monitor_extra_statistics = True
    cfg.ssl_loss_weight = 0.0
    cfg.raw_latent_ssl_loss_weight = 1.0
    cfg.model.model_type = 'mlp'
    cfg.model.observation_shape = 2
    cfg.model.frame_stack_num = 1
    cfg.model.image_channel = 1
    cfg.model.action_space_size = 3
    cfg.model.ensemble_size = 2
    cfg.model.latent_state_dim = 16
    cfg.model.self_supervised_learning_loss = False
    cfg.model.state_norm = False
    cfg.model.norm_type = 'LN'
    cfg.model.encoder_random_prior_scale = 1.0
    cfg.model.encoder_random_prior_input_mean = (-0.3, 0.0)
    cfg.model.encoder_random_prior_input_scale = (0.9, 0.07)
    cfg.model.representation_last_linear_layer_init_zero = False

    model = EnsembleMuZeroModelMLP(
        ensemble_size=2,
        observation_shape=2,
        action_space_size=3,
        latent_state_dim=16,
        self_supervised_learning_loss=False,
        state_norm=False,
        norm_type='LN',
        encoder_random_prior_scale=1.0,
        encoder_random_prior_input_mean=(-0.3, 0.0),
        encoder_random_prior_input_scale=(0.9, 0.07),
        representation_last_linear_layer_init_zero=False,
    )
    policy = EnsembleMuZeroPolicy(cfg, model=model, enable_field=['learn'])
    policy._collect_mcts_temperature = 1.0
    policy.collect_epsilon = 0.0

    observations = np.random.default_rng(0).normal(size=(2, 6)).astype(np.float32)
    actions = np.zeros((2, 2), dtype=np.int64)
    mask = np.ones((2, 3), dtype=np.float32)
    indices = np.arange(2)
    weights = np.ones(2, dtype=np.float32)
    make_time = np.zeros(2, dtype=np.float32)
    target_reward = np.zeros((2, 2), dtype=np.float32)
    target_value = np.zeros((2, 3), dtype=np.float32)
    target_policy = np.full((2, 3, 3), 1 / 3, dtype=np.float32)

    log = policy._forward_learn(
        (
            (observations, actions, mask, indices, weights, make_time),
            (target_reward, target_value, target_policy),
        )
    )

    assert log['raw_latent_ssl/loss'] > 0.0
    assert np.isfinite(log['raw_latent_ssl/cosine'])
    assert log['raw_latent_ssl/depth_1_loss'] > 0.0
    assert log['raw_latent_ssl/depth_2_loss'] > 0.0
    assert not any(key.startswith('dynamics_prior_alignment/') for key in log)


@pytest.mark.unittest
@pytest.mark.parametrize(
    ('member_specific_reader', 'target_transform'),
    [(False, 'log'), (True, 'raw'), (False, 'normalized_raw')],
)
def test_intrinsic_query_loss_runs_inside_primary_muzero_optimizer_step(
        member_specific_reader, target_transform
):
    torch.manual_seed(7)
    cfg = EnsembleMuZeroPolicy.default_config()
    cfg.cuda = False
    cfg.device = 'cpu'
    cfg.batch_size = 2
    cfg.num_unroll_steps = 2
    cfg.optim_type = 'Adam'
    cfg.learning_rate = 3e-4
    cfg.piecewise_decay_lr_scheduler = False
    cfg.use_priority = False
    cfg.monitor_extra_statistics = True
    cfg.intrinsic_head_only = False
    cfg.intrinsic_learning_rate = 1e-4
    cfg.intrinsic_model_loss_weight = 1.0
    cfg.intrinsic_reward_loss_weight = 1.0
    cfg.intrinsic_value_loss_weight = 1.0
    cfg.intrinsic_reward_source = 'rnd'
    cfg.rnd_exploration = True
    cfg.rnd_use_learned_intrinsic_model = True
    cfg.rnd_observation_dim = 2
    cfg.rnd_observation_low = (-1.2, -0.07)
    cfg.rnd_observation_high = (0.6, 0.07)
    cfg.rnd_target_snapshot = True
    cfg.rnd_landscape_query = True
    cfg.rnd_landscape_query_count = 2
    cfg.rnd_landscape_query_hidden_dim = 8
    cfg.rnd_landscape_query_learning_rate = 1e-3
    cfg.rnd_landscape_query_max_horizon = 2
    cfg.rnd_landscape_query_diversity_weight = 0.0
    cfg.rnd_landscape_query_centered_prediction_weight = 0.0
    cfg.rnd_landscape_query_reader_use_context = False
    cfg.rnd_landscape_query_member_specific_reader = member_specific_reader
    cfg.rnd_landscape_query_target_transform = target_transform
    cfg.rnd_landscape_query_gradient_diagnostics_interval = 1
    cfg.rnd_landscape_query_use_tree_rewards = False
    cfg.model.model_type = 'mlp'
    cfg.model.observation_shape = 2
    cfg.model.frame_stack_num = 1
    cfg.model.image_channel = 1
    cfg.model.action_space_size = 3
    cfg.model.ensemble_size = 2
    cfg.model.latent_state_dim = 16
    cfg.model.self_supervised_learning_loss = False
    cfg.model.state_norm = False
    cfg.model.norm_type = 'LN'
    cfg.model.representation_last_linear_layer_init_zero = False

    model = EnsembleMuZeroModelMLP(
        ensemble_size=2,
        observation_shape=2,
        action_space_size=3,
        latent_state_dim=16,
        self_supervised_learning_loss=False,
        state_norm=False,
        norm_type='LN',
        representation_last_linear_layer_init_zero=False,
    )
    policy = EnsembleMuZeroPolicy(cfg, model=model, enable_field=['learn'])
    policy._collect_mcts_temperature = 1.0
    policy.collect_epsilon = 0.0

    query_before = [
        parameter.detach().clone()
        for parameter in policy._rnd_landscape_query_model.parameters()
    ]
    observations = np.asarray(
        [
            [-0.50, 0.00, -0.51, -0.01, -0.52, -0.01],
            [-0.40, 0.00, -0.39, 0.01, -0.37, 0.02],
        ],
        dtype=np.float32,
    )
    actions = np.asarray([[0, 0], [2, 2]], dtype=np.int64)
    mask = np.ones((2, 3), dtype=np.float32)
    indices = np.arange(2)
    weights = np.ones(2, dtype=np.float32)
    make_time = np.zeros(2, dtype=np.float32)
    target_reward = np.zeros((2, 2), dtype=np.float32)
    target_value = np.zeros((2, 3), dtype=np.float32)
    target_policy = np.full((2, 3, 3), 1 / 3, dtype=np.float32)

    log = policy._forward_learn(
        (
            (observations, actions, mask, indices, weights, make_time),
            (target_reward, target_value, target_policy),
        )
    )

    optimized_ids = [
        id(parameter)
        for group in policy._optimizer.param_groups
        for parameter in group['params']
    ]
    assert len(optimized_ids) == len(set(optimized_ids))
    assert not hasattr(policy, '_intrinsic_optimizer')
    assert log['intrinsic/shared_muzero_optimizer'] == 1.0
    assert log['rnd_landscape_query/shared_muzero_optimizer'] == 1.0
    assert log['rnd_landscape_query/joint_world_model_update'] == 1.0
    assert log['rnd_landscape_query/use_tree_rewards'] == 0.0
    assert log['rnd_landscape_query/member_specific_reader'] == float(
        member_specific_reader
    )
    assert log['rnd_landscape_query/target_transform_log'] == float(
        target_transform == 'log'
    )
    assert log['rnd_landscape_query/target_transform_normalized_raw'] == float(
        target_transform == 'normalized_raw'
    )
    for key in [
        'rnd_landscape_query/aux_representation_grad_norm',
        'rnd_landscape_query/aux_dynamics_grad_norm',
        'rnd_landscape_query/aux_reader_grad_norm',
    ]:
        assert np.isfinite(log[key])
        assert log[key] >= 0.0
    representation_parameter = next(
        model.members[0].representation_network.parameters()
    )
    intrinsic_head_parameter = next(
        model.members[0].intrinsic_reward_head.parameters()
    )
    query_parameter = next(policy._rnd_landscape_query_model.parameters())

    def parameter_learning_rate(parameter):
        return next(
            float(group['lr'])
            for group in policy._optimizer.param_groups
            if any(candidate is parameter for candidate in group['params'])
        )

    assert parameter_learning_rate(representation_parameter) == pytest.approx(3e-4)
    assert parameter_learning_rate(intrinsic_head_parameter) == pytest.approx(1e-4)
    assert parameter_learning_rate(query_parameter) == pytest.approx(1e-3)
    assert any(
        not torch.equal(before, after)
        for before, after in zip(
            query_before, policy._rnd_landscape_query_model.parameters()
        )
    )
    assert all(
        parameter.grad is None
        for parameter in policy._rnd_target_snapshot_model.parameters()
    )


@pytest.mark.unittest
def test_rnd_landscape_query_tree_reward_switch_is_independent_of_training_switch():
    policy = object.__new__(EnsembleMuZeroPolicy)
    policy._cfg = EasyDict(
        dict(
            rnd_landscape_query=True,
            rnd_landscape_query_use_tree_rewards=False,
            model=EasyDict(dict(analysis_sim_norm=False)),
        )
    )

    assert policy._rnd_landscape_query_enabled()
    assert not policy._rnd_landscape_query_tree_rewards_enabled()

    policy._cfg.rnd_landscape_query_use_tree_rewards = True
    assert policy._rnd_landscape_query_tree_rewards_enabled()

    policy._cfg.rnd_landscape_query = False
    assert not policy._rnd_landscape_query_tree_rewards_enabled()


@pytest.mark.unittest
def test_ensemble_value_prior_uses_residual_target_in_main_learn_step():
    cfg = EnsembleMuZeroPolicy.default_config()
    cfg.cuda = False
    cfg.device = 'cpu'
    cfg.batch_size = 2
    cfg.num_unroll_steps = 1
    cfg.optim_type = 'Adam'
    cfg.learning_rate = 3e-4
    cfg.piecewise_decay_lr_scheduler = False
    cfg.use_priority = False
    cfg.monitor_extra_statistics = True
    cfg.ssl_loss_weight = 0.0
    cfg.raw_latent_ssl_loss_weight = 0.0
    cfg.model.model_type = 'mlp'
    cfg.model.observation_shape = 2
    cfg.model.frame_stack_num = 1
    cfg.model.image_channel = 1
    cfg.model.action_space_size = 3
    cfg.model.ensemble_size = 2
    cfg.model.latent_state_dim = 16
    cfg.model.self_supervised_learning_loss = False
    cfg.model.state_norm = False
    cfg.model.norm_type = 'LN'
    cfg.model.random_prior_scale = 5.0
    cfg.model.value_prior_backprop_to_latent = True
    cfg.model.value_prior_scalar_loss_weight = 1.0

    model = EnsembleMuZeroModelMLP(
        ensemble_size=2,
        observation_shape=2,
        action_space_size=3,
        latent_state_dim=16,
        self_supervised_learning_loss=False,
        state_norm=False,
        norm_type='LN',
        random_prior_scale=5.0,
        value_prior_backprop_to_latent=True,
        value_prior_scalar_loss_weight=1.0,
    )
    policy = EnsembleMuZeroPolicy(cfg, model=model, enable_field=['learn'])
    policy._collect_mcts_temperature = 1.0
    policy.collect_epsilon = 0.0

    observations = np.random.default_rng(1).normal(size=(2, 4)).astype(np.float32)
    actions = np.zeros((2, 1), dtype=np.int64)
    mask = np.ones((2, 2), dtype=np.float32)
    indices = np.arange(2)
    weights = np.ones(2, dtype=np.float32)
    make_time = np.zeros(2, dtype=np.float32)
    target_reward = np.zeros((2, 1), dtype=np.float32)
    target_value = np.full((2, 2), -40.0, dtype=np.float32)
    target_policy = np.full((2, 2, 3), 1 / 3, dtype=np.float32)

    log = policy._forward_learn(
        (
            (observations, actions, mask, indices, weights, make_time),
            (target_reward, target_value, target_policy),
        )
    )

    assert np.isfinite(log['value_prior/categorical_loss'])
    assert np.isfinite(log['value_prior/scalar_consistency_loss'])
    assert np.isfinite(log['value_prior/final_prediction_mae'])
    assert log['value_prior/scalar_consistency_loss'] >= 0.0
    assert model._value_prior_learning_mode is False
    assert all(
        parameter.grad is None
        for prior in model.value_prior_networks
        for parameter in prior.parameters()
    )


@pytest.mark.unittest
def test_ensemble_muzero_collect_member_assignment_reset():
    policy = object.__new__(EnsembleMuZeroPolicy)
    policy._cfg = EasyDict(dict(model=dict(ensemble_size=3, model_type='mlp', analysis_sim_norm=False)))
    policy._collect_env_member_id = {}

    member_ids_1 = policy._get_collect_member_ids([2, 0])
    member_ids_2 = policy._get_collect_member_ids([2, 0])

    assert member_ids_1 == member_ids_2
    assert set(policy._collect_env_member_id.keys()) == {0, 2}
    assert all(0 <= member_id < 3 for member_id in member_ids_1)

    policy._reset_collect(data_id=[2])
    assert 2 not in policy._collect_env_member_id
    assert 0 in policy._collect_env_member_id

    member_ids_3 = policy._get_collect_member_ids([2])
    assert 0 <= member_ids_3[0] < 3


@pytest.mark.unittest
def test_replay_value_calibration_estimates_centered_member_offsets():
    policy = object.__new__(EnsembleMuZeroPolicy)
    policy._cfg = EasyDict(
        dict(
            device='cpu',
            ensemble_value_calibration=True,
            ensemble_value_calibration_inference_batch_size=2,
            model=EasyDict(
                ensemble_size=3,
                frame_stack_num=1,
                observation_shape=2,
                model_type='mlp',
                self_supervised_learning_loss=False,
                analysis_sim_norm=False,
            ),
        )
    )
    policy._learn_model = _FakeValueCalibrationModel()
    policy.value_inverse_scalar_transform_handle = lambda value: value
    observations = np.asarray(
        [[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32
    )

    stats = policy.update_ensemble_value_calibration(observations, collection_phase=10)

    assert stats['offsets'] == pytest.approx([-10.0, 0.0, 10.0])
    assert stats['member_value_means'] == pytest.approx([0.5, 10.5, 20.5])
    assert stats['raw_state_value_std_mean'] > 8.0
    assert stats['calibrated_state_value_std_mean'] == pytest.approx(0.0, abs=1e-6)
    assert policy._value_calibration_offsets().tolist() == pytest.approx([-10.0, 0.0, 10.0])


@pytest.mark.unittest
def test_state_value_disagreement_uses_raw_member_values():
    policy = object.__new__(EnsembleMuZeroPolicy)
    policy._cfg = EasyDict(
        dict(
            device='cpu',
            state_value_disagreement_inference_batch_size=1,
            model=EasyDict(ensemble_size=2, analysis_sim_norm=False),
        )
    )
    policy._target_model = _FakeStateValueDisagreementTargetModel()
    policy.value_inverse_scalar_transform_handle = lambda value: value

    disagreement, member_values = policy._state_value_disagreement_from_observations(
        torch.tensor([[1.0, 0.0], [3.0, 0.0]])
    )

    np.testing.assert_allclose(member_values.numpy(), [[1.0, 3.0], [3.0, 9.0]])
    np.testing.assert_allclose(disagreement.numpy(), [1.0, 3.0])


@pytest.mark.unittest
def test_collection_state_value_disagreement_is_diagnostic_only_raw_reward():
    policy = object.__new__(EnsembleMuZeroPolicy)
    policy._cfg = EasyDict(
        dict(
            device='cpu',
            intrinsic_reward_source='state_value_disagreement',
            state_value_disagreement_store_collection_rewards=True,
            state_value_disagreement_reward_scale=2.0,
            state_value_disagreement_reward_clip=4.0,
            state_value_disagreement_inference_batch_size=2,
            model=EasyDict(ensemble_size=2, analysis_sim_norm=False),
        )
    )
    policy._collect_model = _FakeStateValueDisagreementTargetModel()
    policy.value_inverse_scalar_transform_handle = lambda value: value

    rewards = policy._collection_state_value_disagreement_rewards_for_observations(
        np.asarray([[1.0, 0.0], [3.0, 0.0]], dtype=np.float32)
    )

    np.testing.assert_allclose(rewards['raw_error'], [1.0, 3.0])
    np.testing.assert_allclose(rewards['bonus'], [2.0, 4.0])


@pytest.mark.unittest
def test_state_value_disagreement_builds_reward_and_value_targets():
    policy = object.__new__(EnsembleMuZeroPolicy)
    policy._cfg = EasyDict(
        dict(
            device='cpu',
            num_unroll_steps=2,
            td_steps=2,
            rnd_intrinsic_td_steps=2,
            rnd_intrinsic_discount_factor=0.5,
            intrinsic_value_bootstrap=False,
            rnd_intrinsic_bootstrap=False,
            bootstrap_at_time_limit=True,
            intrinsic_reward_target_clip=0.0,
            intrinsic_value_target_clip=0.0,
            state_value_disagreement_reward_scale=1.0,
            state_value_disagreement_reward_clip=None,
            state_value_disagreement_inference_batch_size=2,
            state_value_disagreement_intrinsic_target_mode='td',
            model=EasyDict(
                ensemble_size=2,
                model_type='mlp',
                observation_shape=2,
                analysis_sim_norm=False,
            ),
        )
    )
    policy._target_model = _FakeStateValueDisagreementTargetModel()
    policy.value_inverse_scalar_transform_handle = lambda value: value
    observations = np.asarray([[0.0, 0.0, 1.0, 0.0, 3.0, 0.0]], dtype=np.float32)
    actions = np.asarray([[0, 0]], dtype=np.int64)
    masks = np.ones((1, 5), dtype=np.float32)
    current_batch = [
        observations,
        actions,
        masks,
        np.asarray([0], dtype=np.int64),
        np.ones(1, dtype=np.float32),
        np.zeros(1, dtype=np.float32),
    ]

    log, reward_targets, value_targets = policy._prepare_state_value_disagreement_targets(
        [current_batch, []]
    )

    np.testing.assert_allclose(reward_targets.numpy(), [[1.0, 3.0]])
    np.testing.assert_allclose(value_targets.numpy(), [[2.5, 3.0, 0.0]])
    assert log['intrinsic_state_value_disagreement/raw_reward_mean'] == pytest.approx(2.0)


@pytest.mark.unittest
def test_state_value_disagreement_can_train_on_collection_snapshot_rewards():
    policy = object.__new__(EnsembleMuZeroPolicy)
    policy._cfg = EasyDict(
        dict(
            device='cpu',
            num_unroll_steps=2,
            td_steps=2,
            rnd_intrinsic_td_steps=2,
            rnd_intrinsic_discount_factor=0.5,
            intrinsic_value_bootstrap=False,
            rnd_intrinsic_bootstrap=False,
            bootstrap_at_time_limit=True,
            intrinsic_reward_target_clip=0.0,
            intrinsic_value_target_clip=0.0,
            store_selected_intrinsic_rewards=True,
            state_value_disagreement_use_stored_collection_rewards=True,
            state_value_disagreement_reward_scale=1.0,
            state_value_disagreement_reward_clip=None,
            state_value_disagreement_inference_batch_size=2,
            state_value_disagreement_intrinsic_target_mode='td',
            model=EasyDict(
                ensemble_size=2,
                model_type='mlp',
                observation_shape=2,
                analysis_sim_norm=False,
            ),
        )
    )
    policy._target_model = _FakeStateValueDisagreementTargetModel()
    policy.value_inverse_scalar_transform_handle = lambda value: value
    policy._state_value_disagreement_from_observations = lambda observations: pytest.fail(
        "stored collection rewards must not trigger replay-time ensemble inference"
    )
    current_batch = [
        np.asarray([[0.0, 0.0, 1.0, 0.0, 3.0, 0.0]], dtype=np.float32),
        np.asarray([[0, 0]], dtype=np.int64),
        np.ones((1, 5), dtype=np.float32),
        np.asarray([[7.0, 11.0]], dtype=np.float32),
        np.asarray([0], dtype=np.int64),
        np.ones(1, dtype=np.float32),
        np.zeros(1, dtype=np.float32),
    ]

    log, reward_targets, value_targets = policy._prepare_state_value_disagreement_targets(
        [current_batch, []]
    )

    np.testing.assert_allclose(reward_targets.numpy(), [[7.0, 11.0]])
    np.testing.assert_allclose(value_targets.numpy(), [[12.5, 11.0, 0.0]])
    assert log['intrinsic_state_value_disagreement/using_stored_collection_rewards'] == 1.0
    assert log['intrinsic_state_value_disagreement/raw_reward_mean'] == pytest.approx(9.0)
    assert 'intrinsic_state_value_disagreement/stored_current_reward_mae' not in log


@pytest.mark.unittest
def test_state_density_recomputes_targets_from_current_snapshot():
    policy = object.__new__(EnsembleMuZeroPolicy)
    policy._cfg = EasyDict(
        dict(
            device='cpu',
            num_unroll_steps=2,
            td_steps=2,
            rnd_intrinsic_td_steps=2,
            rnd_intrinsic_discount_factor=0.5,
            bootstrap_at_time_limit=True,
            intrinsic_reward_target_clip=0.0,
            intrinsic_value_target_clip=0.0,
            store_selected_intrinsic_rewards=True,
            state_density_intrinsic_target_mode='monte_carlo',
            state_density_position_min=-1.2,
            state_density_velocity_min=-0.07,
            state_density_position_bin_width=0.02,
            state_density_velocity_bin_width=0.002,
            state_density_neighbor_radius=0,
            state_density_reward_scale=1.0,
            state_density_reward_clip=1.0,
            model=EasyDict(
                ensemble_size=2,
                model_type='mlp',
                observation_shape=2,
                analysis_sim_norm=False,
            ),
        )
    )
    observations = np.asarray(
        [[-0.50, 0.000], [-0.49, 0.001], [-0.45, 0.005]], dtype=np.float32
    )
    snapshot_counts = np.zeros((90, 70), dtype=np.int64)
    final_position_bin = int(np.floor((observations[2, 0] + 1.2) / 0.02))
    final_velocity_bin = int(np.floor((observations[2, 1] + 0.07) / 0.002))
    snapshot_counts[final_position_bin, final_velocity_bin] = 3
    policy.set_state_density_snapshot(snapshot_counts, snapshot_phase=7)
    current_batch = [
        observations.reshape(1, -1),
        np.asarray([[0, 0]], dtype=np.int64),
        np.ones((1, 5), dtype=np.float32),
        # Deliberately stale labels: these must not become training targets.
        np.asarray([[0.125, 0.125]], dtype=np.float32),
        np.asarray([0], dtype=np.int64),
        np.ones(1, dtype=np.float32),
        np.zeros(1, dtype=np.float32),
    ]

    log, reward_targets, value_targets = policy._prepare_state_density_targets(
        [current_batch, []]
    )

    np.testing.assert_allclose(reward_targets.numpy(), [[1.0, 0.5]])
    np.testing.assert_allclose(value_targets.numpy(), [[1.25, 0.5, 0.0]])
    assert log['intrinsic_state_density/target_mode_monte_carlo'] == 1.0
    assert log['intrinsic_state_density/using_stored_collection_rewards'] == 0.0
    assert log['intrinsic_state_density/recomputed_current_snapshot'] == 1.0
    assert log['intrinsic_state_density/snapshot_phase'] == 7.0
    assert log['intrinsic_state_density/snapshot_transition_count'] == 3.0


@pytest.mark.unittest
def test_separate_exploration_tree_scores_fixed_depth_sequences():
    policy = object.__new__(EnsembleMuZeroPolicy)
    policy._cfg = EasyDict(
        dict(
            separate_exploration_tree_horizon=2,
            separate_exploration_tree_top_k=1,
            separate_exploration_tree_max_sequences=0,
            discount_factor=1.0,
            model=EasyDict(analysis_sim_norm=False),
        )
    )
    policy.reward_inverse_scalar_transform_handle = lambda value: value
    policy.value_inverse_scalar_transform_handle = lambda value: value

    diagnostics = policy._separate_exploration_tree_diagnostics_for_root(
        _FakeSeparateExplorationTreeModel(),
        torch.zeros(2, 1),
        [0, 1],
    )
    by_action = {int(item['action']): item for item in diagnostics}

    assert by_action[1]['separate_exploration_tree_raw_score'] > by_action[0][
        'separate_exploration_tree_raw_score'
    ]
    assert by_action[1]['separate_exploration_tree_top_sequence'] == [1, 1]
    assert by_action[1]['separate_exploration_tree_depth1_top_sequence'] == [1]
    assert by_action[1]['separate_exploration_tree_depth2_top_sequence'] == [1, 1]


@pytest.mark.unittest
def test_fixed_horizon_exploration_scores_all_sequences_with_mean_and_qstd():
    policy = object.__new__(EnsembleMuZeroPolicy)
    policy._cfg = EasyDict(
        dict(
            separate_exploration_tree_horizon=3,
            separate_exploration_tree_top_k=2,
            separate_exploration_tree_max_sequences=0,
            separate_exploration_tree_sequence_score_mode='mean_q_plus_qstd',
            separate_exploration_tree_sequence_qstd_beta=2.0,
            separate_exploration_tree_sequence_qstd_mode='raw',
            pure_exploration_sequence_temperature=0.0,
            discount_factor=1.0,
            model=EasyDict(analysis_sim_norm=False),
        )
    )
    policy.reward_inverse_scalar_transform_handle = lambda value: value
    policy.value_inverse_scalar_transform_handle = lambda value: value
    model = _FakeSeparateExplorationTreeModel()

    diagnostics, candidates = policy._separate_exploration_tree_diagnostics_for_root(
        model,
        torch.zeros(2, 1),
        [0, 1],
        return_sequence_candidates=True,
    )

    assert len(candidates) == 8
    assert model.batch_sizes == [4, 8, 16]
    member_baselines = np.mean(np.asarray([candidate['q_values'] for candidate in candidates]), axis=0)
    for candidate in candidates:
        assert candidate['sequence_score'] == pytest.approx(
            candidate['mean_q'] + 2.0 * candidate['raw_q_std']
        )
        assert 'member_centered_q_std' in candidate
        assert len(candidate['member_q_baselines']) == 2
        assert candidate['member_centered_q_std'] == pytest.approx(
            np.std(np.asarray(candidate['q_values']) - member_baselines)
        )

    selected, _ = policy._select_pure_exploration_sequence(
        [0, 1], diagnostics, sequence_candidates=candidates
    )
    assert selected['sequence_score'] == max(candidate['sequence_score'] for candidate in candidates)
    assert selected['candidate_count'] == 8


@pytest.mark.unittest
def test_fixed_horizon_bounded_optimism_never_exceeds_best_member_q():
    policy = object.__new__(EnsembleMuZeroPolicy)
    policy._cfg = EasyDict(
        dict(
            separate_exploration_tree_horizon=3,
            separate_exploration_tree_top_k=2,
            separate_exploration_tree_max_sequences=0,
            separate_exploration_tree_sequence_score_mode='bounded_mean_q_plus_qstd',
            separate_exploration_tree_sequence_qstd_beta=2.0,
            separate_exploration_tree_sequence_qstd_mode='raw',
            pure_exploration_sequence_temperature=0.0,
            discount_factor=1.0,
            model=EasyDict(analysis_sim_norm=False),
        )
    )
    policy.reward_inverse_scalar_transform_handle = lambda value: value
    policy.value_inverse_scalar_transform_handle = lambda value: value

    _, candidates = policy._separate_exploration_tree_diagnostics_for_root(
        _FakeSeparateExplorationTreeModel(),
        torch.zeros(2, 1),
        [0, 1],
        return_sequence_candidates=True,
    )

    capped_candidates = 0
    for candidate in candidates:
        best_member_q = max(candidate['q_values'])
        assert candidate['sequence_score'] <= best_member_q + 1e-6
        assert candidate['optimistic_member_q'] == pytest.approx(best_member_q)
        assert candidate['sequence_score'] == pytest.approx(
            min(candidate['unbounded_sequence_score'], best_member_q)
        )
        if candidate['optimism_cap_applied']:
            capped_candidates += 1
            assert candidate['optimism_cap_reduction'] > 0.0
    assert capped_candidates > 0


@pytest.mark.unittest
def test_ensemble_muzero_repeats_learn_batch_for_all_members():
    policy = object.__new__(EnsembleMuZeroPolicy)
    policy._cfg = EasyDict(dict(model=dict(ensemble_size=3, model_type='mlp', analysis_sim_norm=False), batch_size=2))

    current_batch = [
        np.array([[1], [2]]),
        np.array([[3], [4]]),
        np.array([[5], [6]]),
        np.array([7, 8]),
        np.array([1.0, 0.5]),
        np.array([0.0, 0.0]),
    ]
    target_batch = [
        np.array([[1, 2], [3, 4]]),
        np.array([[5, 6], [7, 8]]),
        np.array([[[1.0, 0.0]], [[0.0, 1.0]]]),
    ]

    expanded_data, original_batch_size, member_ids = policy._repeat_batch_for_members([current_batch, target_batch])
    expanded_current_batch, expanded_target_batch = expanded_data

    assert original_batch_size == 2
    assert member_ids == [0, 0, 1, 1, 2, 2]
    assert expanded_current_batch[0].shape[0] == 6
    assert expanded_target_batch[0].shape[0] == 6
    assert np.array_equal(expanded_current_batch[0][:2], current_batch[0])
    assert np.array_equal(expanded_current_batch[0][2:4], current_batch[0])
    assert np.array_equal(expanded_target_batch[2][4:6], target_batch[2])


@pytest.mark.unittest
def test_ensemble_muzero_routes_member_specific_value_targets():
    policy = object.__new__(EnsembleMuZeroPolicy)
    policy._cfg = EasyDict(dict(model=dict(ensemble_size=3, model_type='mlp', analysis_sim_norm=False), batch_size=2))

    current_batch = [
        np.array([[1], [2]]),
        np.array([[3], [4]]),
        np.array([[5], [6]]),
        np.array([7, 8]),
        np.array([1.0, 0.5]),
        np.array([0.0, 0.0]),
    ]
    member_value_targets = np.array(
        [
            [[10, 11], [12, 13]],
            [[20, 21], [22, 23]],
            [[30, 31], [32, 33]],
        ]
    )
    target_batch = [
        np.array([[1, 2], [3, 4]]),
        np.array([[5, 6], [7, 8]]),
        np.array([[[1.0, 0.0]], [[0.0, 1.0]]]),
        member_value_targets,
    ]

    expanded_data, original_batch_size, member_ids = policy._repeat_batch_for_members([current_batch, target_batch])
    _, expanded_target_batch = expanded_data

    assert original_batch_size == 2
    assert member_ids == [0, 0, 1, 1, 2, 2]
    assert len(expanded_target_batch) == 3
    assert np.array_equal(expanded_target_batch[1], member_value_targets.reshape(6, 2))


@pytest.mark.unittest
def test_muzero_game_buffer_member_search_overrides_use_each_member():
    buffer = object.__new__(MuZeroGameBuffer)
    buffer._cfg = EasyDict(dict(sve_search_noise=False))
    seen_member_ids = []
    seen_search_noise = []

    def fake_infer_initial_outputs(obs_list, model, member_id=None):
        seen_member_ids.append(member_id)
        return [member_id]

    def fake_search_values(
            network_output,
            transition_batch_size,
            legal_actions,
            to_play,
            model,
            member_id=None,
            add_root_noise=None,
    ):
        seen_search_noise.append(add_root_noise)
        return np.asarray([100 * int(member_id) + i for i in range(transition_batch_size)], dtype=np.float32)

    buffer._infer_initial_outputs = fake_infer_initial_outputs
    buffer._search_values_from_network_output = fake_search_values

    overrides = MuZeroGameBuffer._member_search_value_overrides(
        buffer,
        obs_list=['o0', 'o1'],
        flat_indices=[3, 7],
        legal_actions=[[0, 1], [1, 2]],
        to_play=[-1, -1],
        model=object(),
        ensemble_member_bootstrap_size=3,
    )

    assert seen_member_ids == [0, 1, 2]
    assert seen_search_noise == [False, False, False]
    assert overrides[0] == {3: 0.0, 7: 1.0}
    assert overrides[1] == {3: 100.0, 7: 101.0}
    assert overrides[2] == {3: 200.0, 7: 201.0}


@pytest.mark.unittest
def test_ensemble_muzero_eval_search_options():
    policy = object.__new__(EnsembleMuZeroPolicy)
    policy._cfg = EasyDict(
        dict(
            model=dict(analysis_sim_norm=False),
            ensemble_eval_search_mode='mean_shared_tree',
            ensemble_eval_ucb_beta=None,
            ensemble_eval_uncertainty_decay=None,
            ensemble_ucb_beta=0.7,
            ensemble_uncertainty_decay=True,
        )
    )

    mode, beta, decay = policy._resolve_eval_search_options()
    assert mode == 'mean_shared_tree'
    assert beta == 0.0
    assert decay is True

    policy._cfg.ensemble_eval_search_mode = 'pessimistic_shared_tree'
    mode, beta, decay = policy._resolve_eval_search_options()
    assert mode == 'pessimistic_shared_tree'
    assert beta == 0.7
    assert decay is True

    policy._cfg.ensemble_eval_ucb_beta = 0.2
    policy._cfg.ensemble_eval_uncertainty_decay = False
    mode, beta, decay = policy._resolve_eval_search_options()
    assert mode == 'pessimistic_shared_tree'
    assert beta == 0.2
    assert decay is False

    policy._cfg.ensemble_eval_search_mode = 'bounded_pessimistic_shared_tree'
    mode, beta, decay = policy._resolve_eval_search_options()
    assert mode == 'bounded_pessimistic_shared_tree'
    assert beta == 0.2
    assert decay is False


@pytest.mark.unittest
def test_intrinsic_only_training_accepts_fully_isolated_configuration():
    policy = object.__new__(EnsembleMuZeroPolicy)
    policy._cfg = EasyDict(
        dict(
            intrinsic_only_training=True,
            collection_policy_mode='intrinsic_tree_only',
            separate_intrinsic_tree=True,
            rnd_use_learned_intrinsic_model=True,
            intrinsic_head_only=False,
            intrinsic_tree_prior_mode='learned',
            intrinsic_tree_store_final_policy=True,
            reward_loss_weight=0.0,
            value_loss_weight=0.0,
            policy_loss_weight=1.0,
            reanalyze_ratio=0.0,
            use_priority=False,
            elite_episode_replay=False,
            model=dict(analysis_sim_norm=False),
        )
    )

    policy._validate_intrinsic_only_training()


@pytest.mark.unittest
def test_intrinsic_only_training_initializes_real_learn_collect_and_eval_modes():
    cfg = EnsembleMuZeroPolicy.default_config()
    cfg.cuda = False
    cfg.device = 'cpu'
    cfg.batch_size = 2
    cfg.optim_type = 'Adam'
    cfg.learning_rate = 3e-4
    cfg.piecewise_decay_lr_scheduler = False
    cfg.intrinsic_only_training = True
    cfg.collection_policy_mode = 'intrinsic_tree_only'
    cfg.separate_intrinsic_tree = True
    cfg.rnd_exploration = True
    cfg.rnd_intrinsic_reward_weight = 1.0
    cfg.rnd_use_learned_intrinsic_model = True
    cfg.intrinsic_head_only = False
    cfg.intrinsic_tree_prior_mode = 'learned'
    cfg.intrinsic_tree_store_final_policy = True
    cfg.intrinsic_tree_eval = True
    cfg.reward_loss_weight = 0.0
    cfg.value_loss_weight = 0.0
    cfg.policy_loss_weight = 1.0
    cfg.reanalyze_ratio = 0.0
    cfg.use_priority = False
    cfg.elite_episode_replay = False
    cfg.model.model_type = 'mlp'
    cfg.model.observation_shape = 2
    cfg.model.frame_stack_num = 1
    cfg.model.image_channel = 1
    cfg.model.action_space_size = 3
    cfg.model.ensemble_size = 2
    cfg.model.latent_state_dim = 16
    cfg.model.self_supervised_learning_loss = False
    cfg.model.state_norm = False
    cfg.model.norm_type = 'LN'

    model = EnsembleMuZeroModelMLP(
        ensemble_size=2,
        observation_shape=2,
        action_space_size=3,
        latent_state_dim=16,
        self_supervised_learning_loss=False,
        state_norm=False,
        norm_type='LN',
    )
    policy = EnsembleMuZeroPolicy(
        cfg,
        model=model,
        enable_field=['learn', 'collect', 'eval'],
    )

    assert policy._ensemble_intrinsic_tree_mcts_collect is not None
    assert policy._ensemble_intrinsic_tree_mcts_eval is not None
    optimized_parameter_ids = {
        id(parameter)
        for group in policy._optimizer.param_groups
        for parameter in group['params']
    }
    representation_parameter = next(model.members[0].representation_network.parameters())
    assert id(representation_parameter) in optimized_parameter_ids
    intrinsic_head_parameter = next(model.members[0].intrinsic_reward_head.parameters())
    assert id(intrinsic_head_parameter) in optimized_parameter_ids
    assert not hasattr(policy, '_intrinsic_optimizer')


@pytest.mark.unittest
@pytest.mark.parametrize(
    ('field', 'value', 'requirement'),
    [
        ('collection_policy_mode', 'normal', 'collection_policy_mode'),
        ('intrinsic_head_only', True, 'intrinsic_world_model_training'),
        ('intrinsic_tree_prior_mode', 'uniform', 'learned_intrinsic_policy_prior'),
        ('reward_loss_weight', 1.0, 'main_reward_loss_disabled'),
        ('value_loss_weight', 0.25, 'main_value_loss_disabled'),
        ('reanalyze_ratio', 0.25, 'main_tree_reanalysis_disabled'),
        ('use_priority', True, 'extrinsic_priority_disabled'),
        ('elite_episode_replay', True, 'extrinsic_elite_replay_disabled'),
    ],
)
def test_intrinsic_only_training_rejects_extrinsic_leakage(field, value, requirement):
    policy = object.__new__(EnsembleMuZeroPolicy)
    policy._cfg = EasyDict(
        dict(
            intrinsic_only_training=True,
            collection_policy_mode='intrinsic_tree_only',
            separate_intrinsic_tree=True,
            rnd_use_learned_intrinsic_model=True,
            intrinsic_head_only=False,
            intrinsic_tree_prior_mode='learned',
            intrinsic_tree_store_final_policy=True,
            reward_loss_weight=0.0,
            value_loss_weight=0.0,
            policy_loss_weight=1.0,
            reanalyze_ratio=0.0,
            use_priority=False,
            elite_episode_replay=False,
            model=dict(analysis_sim_norm=False),
        )
    )
    policy._cfg[field] = value

    with pytest.raises(ValueError, match=requirement):
        policy._validate_intrinsic_only_training()


class _FakeIntrinsicEvalModel:

    def eval(self):
        return self

    def initial_inference(self, observations, member_id=None):
        batch_size = observations.shape[0]
        return EasyDict(
            latent_state=observations,
            policy_logits=torch.zeros(batch_size, 3),
            value=torch.zeros(batch_size, 1),
        )


class _FakeIntrinsicEvalSearch:

    def __init__(self):
        self.kwargs = None

    def search(self, **kwargs):
        self.kwargs = kwargs
        batch_size = len(kwargs['legal_actions'])
        return EasyDict(
            distributions=[[0.1, 0.8, 0.1] for _ in range(batch_size)],
            values=[1.0 for _ in range(batch_size)],
            intrinsic_values=[1.0 for _ in range(batch_size)],
            predicted_values=[0.0 for _ in range(batch_size)],
            shared_policy_logits=[[0.0, 0.0, 0.0] for _ in range(batch_size)],
            root_uncertainties=[0.0 for _ in range(batch_size)],
            root_action_diagnostics=[[] for _ in range(batch_size)],
        )


@pytest.mark.unittest
def test_intrinsic_tree_eval_uses_learned_intrinsic_search_deterministically():
    policy = object.__new__(EnsembleMuZeroPolicy)
    policy._cfg = EasyDict(
        dict(
            separate_intrinsic_tree=True,
            rnd_use_learned_intrinsic_model=True,
            intrinsic_reward_source='rnd',
            intrinsic_tree_num_simulations=25,
            num_simulations=25,
            intrinsic_tree_prior_mode='learned',
            intrinsic_tree_force_root_action_coverage=False,
            model=dict(ensemble_size=2, model_type='mlp', analysis_sim_norm=False),
        )
    )
    policy._eval_model = _FakeIntrinsicEvalModel()
    policy._ensemble_intrinsic_tree_mcts_eval = _FakeIntrinsicEvalSearch()
    policy.value_inverse_scalar_transform_handle = lambda value: value
    policy._value_calibration_offsets = lambda: np.zeros(2, dtype=np.float32)

    output = policy._forward_eval_intrinsic_tree(
        torch.zeros(1, 2),
        action_mask=np.ones((1, 3), dtype=np.float32),
    )

    assert output[0]['action'] == 1
    assert output[0]['ensemble_eval_search_mode'] == 'intrinsic_tree'
    search_kwargs = policy._ensemble_intrinsic_tree_mcts_eval.kwargs
    assert search_kwargs['mode'] == 'intrinsic_shared_tree'
    assert search_kwargs['use_model_intrinsic'] is True
    assert search_kwargs['policy_prior_mode'] == 'learned'
    assert search_kwargs['add_exploration_noise'] is False


@pytest.mark.unittest
def test_ensemble_muzero_intrinsic_return_targets_include_bootstrap():
    policy = object.__new__(EnsembleMuZeroPolicy)
    policy._cfg = EasyDict(
        dict(
            model=dict(analysis_sim_norm=False),
            num_unroll_steps=2,
            rnd_intrinsic_td_steps=2,
            rnd_intrinsic_discount_factor=0.5,
        )
    )

    rewards = torch.ones(1, 5)
    bootstrap_values = torch.tensor([[10.0, 20.0, 30.0]])

    targets = policy._make_intrinsic_return_targets(rewards, bootstrap_values)

    expected = torch.tensor([[4.0, 6.5, 9.0]])
    assert torch.allclose(targets, expected)


@pytest.mark.unittest
def test_ensemble_muzero_intrinsic_return_targets_support_member_specific_bootstrap():
    policy = object.__new__(EnsembleMuZeroPolicy)
    policy._cfg = EasyDict(
        dict(
            model=dict(analysis_sim_norm=False),
            num_unroll_steps=2,
            rnd_intrinsic_td_steps=2,
            rnd_intrinsic_discount_factor=0.5,
        )
    )

    rewards = torch.ones(1, 5)
    bootstrap_values = torch.tensor([[10.0, 20.0, 30.0], [100.0, 200.0, 300.0]])

    targets = policy._make_intrinsic_return_targets(rewards, bootstrap_values)

    expected = torch.tensor([[4.0, 6.5, 9.0], [26.5, 51.5, 76.5]])
    assert torch.allclose(targets, expected)


@pytest.mark.unittest
def test_ensemble_muzero_intrinsic_return_targets_bootstrap_on_time_limit_only():
    policy = object.__new__(EnsembleMuZeroPolicy)
    policy._cfg = EasyDict(
        dict(
            model=dict(analysis_sim_norm=False),
            num_unroll_steps=2,
            rnd_intrinsic_td_steps=3,
            rnd_intrinsic_discount_factor=0.5,
        )
    )

    rewards = torch.ones(2, 5)
    bootstrap_values = torch.tensor([[10.0, 20.0, 30.0], [10.0, 20.0, 30.0]])
    valid_reward_counts = torch.tensor([2, 2])
    time_limit_truncated = torch.tensor([True, False])

    targets = policy._make_intrinsic_return_targets(
        rewards,
        bootstrap_values,
        valid_reward_counts=valid_reward_counts,
        time_limit_truncated=time_limit_truncated,
    )

    expected = torch.tensor([[4.0, 11.0, 0.0], [1.5, 1.0, 0.0]])
    assert torch.allclose(targets, expected)


@pytest.mark.unittest
def test_ensemble_muzero_intrinsic_monte_carlo_return_targets_use_remaining_rewards():
    policy = object.__new__(EnsembleMuZeroPolicy)
    policy._cfg = EasyDict(
        dict(
            model=dict(analysis_sim_norm=False),
            num_unroll_steps=2,
            rnd_intrinsic_discount_factor=0.5,
        )
    )

    rewards = torch.tensor([[1.0, 2.0, 4.0, 8.0]])
    targets = policy._make_intrinsic_monte_carlo_return_targets(
        rewards,
        valid_reward_counts=torch.tensor([4]),
    )

    expected = torch.tensor([[4.0, 6.0, 8.0]])
    assert torch.allclose(targets, expected)


@pytest.mark.unittest
def test_ensemble_muzero_dynamics_prior_alignment_loss_updates_model():
    policy = object.__new__(EnsembleMuZeroPolicy)
    policy._cfg = EasyDict(
        dict(
            device='cpu',
            use_augmentation=False,
            multi_gpu=False,
            grad_clip_value=10.0,
            use_rnd_model=False,
            batch_size=2,
            num_unroll_steps=2,
            dynamics_prior_alignment_loss_weight=0.5,
            dynamics_prior_alignment_target_normalization='none',
            model=dict(
                model_type='mlp',
                observation_shape=2,
                frame_stack_num=1,
                image_channel=1,
                self_supervised_learning_loss=True,
                shared_representation=False,
                analysis_sim_norm=False,
            ),
        )
    )
    policy._learn_model = _FakeDynamicsPriorAlignmentModel()
    policy._target_model = _FakeTargetModel()
    policy._optimizer = torch.optim.SGD(policy._learn_model.parameters(), lr=0.1)

    obs = np.array(
        [
            [1.0, 0.0, 0.5, 0.0, 0.25, 0.0],
            [0.0, 1.0, 0.0, 0.5, 0.0, 0.25],
        ],
        dtype=np.float32,
    )
    actions = np.zeros((2, 2), dtype=np.int64)
    mask = np.ones((2, 3), dtype=np.float32)
    indices = np.arange(2)
    weights = np.ones(2, dtype=np.float32)
    make_time = np.zeros(2, dtype=np.float32)

    before = policy._learn_model.linear.weight.detach().clone()
    log = policy._train_dynamics_prior_alignment_from_replay_batch(
        ((obs, actions, mask, indices, weights, make_time), None),
        [0, 1],
    )
    after = policy._learn_model.linear.weight.detach()

    assert log['dynamics_prior_alignment/loss'] > 0
    assert log['dynamics_prior_alignment/weighted_loss'] == pytest.approx(
        0.5 * log['dynamics_prior_alignment/loss']
    )
    assert log['dynamics_prior_alignment/valid_weight_sum'] == pytest.approx(4.0)
    assert not torch.allclose(before, after)
    assert policy._target_model.updated
    assert policy._learn_model.member_ids == [0, 1]


@pytest.mark.unittest
def test_ensemble_muzero_intrinsic_bootstrap_values_are_member_specific():
    policy = object.__new__(EnsembleMuZeroPolicy)
    policy._cfg = EasyDict(
        dict(
            model=dict(ensemble_size=2, analysis_sim_norm=False),
            num_unroll_steps=2,
            rnd_intrinsic_bootstrap=True,
        )
    )
    policy._learn_model = _FakeIntrinsicBootstrapModel()
    policy._target_model = _FakeIntrinsicBootstrapModel()

    obs_seq = torch.arange(6, dtype=torch.float32).view(1, 6, 1)

    bootstrap_values = policy._make_intrinsic_bootstrap_values(obs_seq, reward_steps=5, td_steps=2)

    assert torch.allclose(bootstrap_values, torch.tensor([[2.0, 3.0, 4.0], [12.0, 13.0, 14.0]]))
    assert policy._learn_model.member_ids is None
    assert policy._target_model.member_ids == [0, 0, 0, 1, 1, 1]
    assert bootstrap_values.requires_grad is False


@pytest.mark.unittest
def test_ensemble_muzero_intrinsic_bootstrap_values_can_be_disabled():
    policy = object.__new__(EnsembleMuZeroPolicy)
    policy._cfg = EasyDict(
        dict(
            model=dict(ensemble_size=2, analysis_sim_norm=False),
            num_unroll_steps=2,
            rnd_intrinsic_bootstrap=False,
            intrinsic_value_bootstrap=False,
        )
    )
    policy._learn_model = _FakeIntrinsicBootstrapModel()
    policy._target_model = _FakeIntrinsicBootstrapModel()

    obs_seq = torch.arange(6, dtype=torch.float32).view(1, 6, 1)

    bootstrap_values = policy._make_intrinsic_bootstrap_values(
        obs_seq, reward_steps=5, td_steps=2
    )

    assert bootstrap_values is None


@pytest.mark.unittest
def test_ensemble_muzero_intrinsic_head_only_parameters_exclude_shared_model():
    policy = object.__new__(EnsembleMuZeroPolicy)
    policy._cfg = EasyDict(dict(intrinsic_head_only=True, model=dict(analysis_sim_norm=False)))
    policy._learn_model = _FakeIntrinsicEnsemble()

    selected_params = set(policy._intrinsic_model_parameters())
    for member in policy._learn_model.members:
        assert member.intrinsic_reward_head.weight in selected_params
        assert member.intrinsic_value_head.weight in selected_params
        assert member.shared.weight not in selected_params

    policy._cfg.intrinsic_head_only = False
    all_params = set(policy._intrinsic_model_parameters())
    for member in policy._learn_model.members:
        assert member.shared.weight in all_params


@pytest.mark.unittest
def test_ensemble_muzero_current_batch_separates_intrinsic_auxiliary_fields():
    policy = object.__new__(EnsembleMuZeroPolicy)
    policy._cfg = EasyDict(
        dict(
            store_selected_intrinsic_rewards=True,
            intrinsic_value_train_on_pure_exploration_only=True,
            model=dict(analysis_sim_norm=False),
        )
    )
    current_batch = [
        np.asarray([[1.0]], dtype=np.float32),
        np.asarray([[0]], dtype=np.int64),
        np.asarray([[1.0]], dtype=np.float32),
        np.asarray([[0.25]], dtype=np.float32),
        np.asarray([1.0], dtype=np.float32),
        np.asarray(['sample-0']),
        np.asarray([1.0], dtype=np.float32),
        np.asarray([0.0], dtype=np.float32),
    ]

    core_batch = policy._current_batch_core(current_batch)

    assert len(core_batch) == 6
    assert policy._selected_intrinsic_reward_batch(current_batch) is current_batch[3]
    assert policy._pure_exploration_episode_batch(current_batch) is current_batch[4]
    assert core_batch[3] is current_batch[5]
    stripped_batch, stripped_targets = policy._strip_selected_intrinsic_rewards_from_train_data(
        [current_batch, ['targets']]
    )
    assert stripped_targets == ['targets']
    assert all(actual is expected for actual, expected in zip(stripped_batch, core_batch))


@pytest.mark.unittest
def test_ensemble_muzero_mixed_reward_tree_selects_explicit_search_mode():
    policy = object.__new__(EnsembleMuZeroPolicy)
    policy._cfg = EasyDict(
        dict(
            mixed_reward_tree=True,
            ensemble_search_mode='mean_shared_tree',
            model=dict(analysis_sim_norm=False),
        )
    )

    assert policy._main_tree_collect_search_mode() == 'mixed_reward_shared_tree'

    policy._cfg.mixed_reward_tree = False
    assert policy._main_tree_collect_search_mode() == 'mean_shared_tree'


@pytest.mark.unittest
def test_ensemble_muzero_joint_intrinsic_query_training_updates_world_model():
    torch.manual_seed(0)
    policy = object.__new__(EnsembleMuZeroPolicy)
    policy._cfg = EasyDict(
        dict(
            device='cpu',
            num_unroll_steps=2,
            batch_size=2,
            multi_gpu=False,
            grad_clip_value=100.0,
            weight_decay=0.0,
            intrinsic_head_only=False,
            intrinsic_learning_rate=1e-3,
            intrinsic_model_loss_weight=1.0,
            intrinsic_reward_loss_weight=1.0,
            intrinsic_value_loss_weight=1.0,
            intrinsic_train_value_head=True,
            intrinsic_loss_type='mse',
            intrinsic_reward_source='rnd',
            rnd_exploration=True,
            rnd_use_learned_intrinsic_model=True,
            rnd_hidden_dim=8,
            rnd_feature_dim=4,
            rnd_learning_rate=1e-3,
            rnd_target_snapshot=True,
            rnd_target_snapshot_refresh_per_collect=True,
            rnd_observation_dim=2,
            rnd_observation_indices=None,
            rnd_observation_low=(-1.2, -0.07),
            rnd_observation_high=(0.6, 0.07),
            rnd_landscape_query=True,
            rnd_landscape_query_count=2,
            rnd_landscape_query_hidden_dim=8,
            rnd_landscape_query_learning_rate=1e-3,
            rnd_landscape_query_loss_weight=1.0,
            rnd_landscape_query_max_horizon=2,
            rnd_landscape_query_diversity_weight=0.0,
            rnd_landscape_query_centered_prediction_weight=0.0,
            rnd_landscape_query_grad_clip=5.0,
            rnd_landscape_query_gradient_diagnostics_interval=1,
            update_per_collect=1,
            store_selected_intrinsic_rewards=False,
            model=EasyDict(
                dict(
                    ensemble_size=1,
                    latent_state_dim=16,
                    model_type='mlp',
                    observation_shape=2,
                    frame_stack_num=1,
                    self_supervised_learning_loss=False,
                    analysis_sim_norm=False,
                )
            ),
        )
    )
    policy._init_rnd_state()
    policy._learn_model = EnsembleMuZeroModelMLP(
        ensemble_size=1,
        observation_shape=2,
        action_space_size=3,
        latent_state_dim=16,
        representation_last_linear_layer_init_zero=False,
        norm_type='LN',
    )
    policy._optimizer = torch.optim.Adam(
        policy._learn_model.parameters(), lr=1e-3
    )
    observations = np.asarray(
        [
            [-0.50, 0.00, -0.51, -0.01, -0.52, -0.01],
            [-0.40, 0.00, -0.39, 0.01, -0.37, 0.02],
        ],
        dtype=np.float32,
    )
    actions = np.asarray([[0, 0], [2, 2]], dtype=np.int64)
    masks = np.ones((2, 3), dtype=np.float32)
    current_batch = (
        observations,
        actions,
        masks,
        np.arange(2),
        np.ones(2, dtype=np.float32),
        np.zeros(2, dtype=np.float32),
    )
    reward_targets = torch.ones((2, 2), dtype=torch.float32)
    value_targets = torch.full((2, 3), 2.0, dtype=torch.float32)
    member = policy._learn_model.members[0]
    representation_parameter = next(member.representation_network.parameters())
    dynamics_parameter = next(member.dynamics_network.parameters())
    representation_before = representation_parameter.detach().clone()
    dynamics_before = dynamics_parameter.detach().clone()
    policy._ensure_rnd_model(torch.from_numpy(observations[:, :2]))
    query_model = policy._ensure_rnd_landscape_query_model()
    query_before = [parameter.detach().clone() for parameter in query_model.parameters()]
    rnd_snapshot_before = [
        parameter.detach().clone()
        for parameter in policy._rnd_target_snapshot_model.parameters()
    ]

    # The intrinsic output layers start at zero, so the first update opens the
    # gradient path and the second update reaches the shared world model.
    for _ in range(2):
        root_observations = torch.from_numpy(observations[:, :2])
        latent_states = [
            policy._learn_model.initial_inference(
                root_observations, member_id=[0, 0]
            ).latent_state
        ]
        for depth in range(actions.shape[1]):
            latent_states.append(
                policy._learn_model.recurrent_inference(
                    latent_states[-1],
                    torch.from_numpy(actions[:, depth]),
                    member_id=[0, 0],
                ).latent_state
            )
        loss, log = policy._intrinsic_loss_from_primary_unroll(
            (current_batch, None),
            reward_targets,
            value_targets,
            [0, 0],
            latent_states,
            torch.ones(2, 3),
            torch.ones(2),
            original_batch_size=2,
        )
        policy._optimizer.zero_grad()
        loss.backward()
        policy._optimizer.step()

    assert log['intrinsic/head_only'] == 0.0
    assert log['rnd_landscape_query/joint_world_model_update'] == 1.0
    assert 'intrinsic_gradient/representation_private_reward_norm' in log
    assert 'intrinsic_gradient/representation_auxiliary_norm' in log
    assert 'intrinsic_gradient/representation_private_auxiliary_cosine' in log
    assert 'intrinsic/reward_depth_1_prediction_mae_raw' in log
    assert 'rnd_landscape_query/depth_1_prediction_mae_raw' in log
    assert not torch.allclose(representation_parameter, representation_before)
    assert not torch.allclose(dynamics_parameter, dynamics_before)
    assert any(
        not torch.equal(before, after)
        for before, after in zip(query_before, query_model.parameters())
    )
    assert all(
        torch.equal(before, after)
        for before, after in zip(
            rnd_snapshot_before, policy._rnd_target_snapshot_model.parameters()
        )
    )
    optimized_ids = [
        id(parameter)
        for group in policy._optimizer.param_groups
        for parameter in group['params']
    ]
    assert len(optimized_ids) == len(set(optimized_ids))
    assert all(id(parameter) in optimized_ids for parameter in query_model.parameters())


@pytest.mark.unittest
def test_intrinsic_reward_loss_uses_log_training_space_but_logs_raw_predictions():
    policy = object.__new__(EnsembleMuZeroPolicy)
    policy._cfg = EasyDict(
        dict(
            device='cpu',
            num_unroll_steps=2,
            batch_size=2,
            intrinsic_head_only=False,
            intrinsic_model_loss_weight=1.0,
            intrinsic_reward_loss_weight=1.0,
            intrinsic_value_loss_weight=0.0,
            intrinsic_train_value_head=False,
            intrinsic_loss_type='huber',
            intrinsic_huber_delta=1.0,
            intrinsic_reward_source='rnd',
            rnd_exploration=True,
            rnd_use_learned_intrinsic_model=True,
            rnd_landscape_query=False,
            store_selected_intrinsic_rewards=False,
            model=EasyDict(dict(ensemble_size=1, analysis_sim_norm=False)),
        )
    )
    policy._learn_model = EnsembleMuZeroModelMLP(
        ensemble_size=1,
        observation_shape=2,
        action_space_size=3,
        latent_state_dim=8,
        intrinsic_reward_target_transform='log',
        intrinsic_reward_log_reference=1e-2,
        norm_type='LN',
    )
    policy._optimizer = torch.optim.Adam(
        policy._learn_model.parameters(), lr=1e-3
    )
    observations = torch.tensor([[-0.5, 0.0], [-0.4, 0.01]])
    actions = torch.tensor([[0, 0], [2, 2]])
    latent_states = [
        policy._learn_model.initial_inference(observations, member_id=[0, 0]).latent_state
    ]
    for step in range(2):
        latent_states.append(
            policy._learn_model.recurrent_inference(
                latent_states[-1], actions[:, step], member_id=[0, 0]
            ).latent_state
        )
    reward_targets = torch.tensor([[1e-3, 1e-2], [1e-1, 1.0]])
    current_batch = (
        np.zeros((2, 6), dtype=np.float32),
        actions.numpy(),
        np.ones((2, 3), dtype=np.float32),
        np.arange(2),
        np.ones(2, dtype=np.float32),
        np.zeros(2, dtype=np.float32),
    )

    loss, log = policy._intrinsic_loss_from_primary_unroll(
        (current_batch, None),
        reward_targets,
        None,
        [0, 0],
        latent_states,
        torch.ones(2, 3),
        torch.ones(2),
        original_batch_size=2,
    )
    transformed_targets = torch.log(reward_targets / 1e-2)
    expected = torch.nn.functional.smooth_l1_loss(
        torch.zeros_like(transformed_targets), transformed_targets
    )
    assert loss.item() == pytest.approx(expected.item())
    assert log['intrinsic/reward_target_transform_log'] == 1.0
    assert log['intrinsic/reward_prediction_mean'] == pytest.approx(1e-2)
    assert log['intrinsic/reward_training_target_mean'] == pytest.approx(
        reward_targets.mean().item()
    )


@pytest.mark.unittest
def test_episodic_inverse_loss_trains_encoder_bridges_and_world_model():
    torch.manual_seed(11)
    batch_size = 2
    horizon = 2
    ensemble_size = 2
    policy = object.__new__(EnsembleMuZeroPolicy)
    policy._cfg = EasyDict(
        dict(
            device='cpu',
            episodic_inverse_loss_weight=1.0,
            episodic_inverse_bridge_loss_weight=1.0,
            episodic_inverse_train_world_model=True,
            episodic_inverse_max_horizon=horizon,
            model=EasyDict(
                dict(
                    ensemble_size=ensemble_size,
                    model_type='mlp',
                    observation_shape=2,
                    frame_stack_num=1,
                    analysis_sim_norm=False,
                )
            ),
        )
    )
    policy._learn_model = EnsembleMuZeroModelMLP(
        ensemble_size=ensemble_size,
        observation_shape=2,
        action_space_size=3,
        latent_state_dim=8,
        episodic_inverse_embedding_dim=6,
        episodic_inverse_hidden_dim=12,
        representation_last_linear_layer_init_zero=False,
        norm_type='LN',
    )
    observations = np.asarray(
        [
            [-0.5, 0.0, -0.51, -0.01, -0.52, -0.01],
            [-0.4, 0.0, -0.39, 0.01, -0.37, 0.02],
        ],
        dtype=np.float32,
    )
    actions = np.asarray([[0, 0], [2, 2]], dtype=np.int64)
    member_ids = [0, 0, 1, 1]
    root_observations = torch.from_numpy(observations[:, :2]).repeat(
        ensemble_size, 1
    )
    latent = policy._learn_model.initial_inference(
        root_observations, member_id=member_ids
    ).latent_state
    recurrent_latents = []
    for depth in range(horizon):
        repeated_actions = torch.from_numpy(actions[:, depth]).repeat(
            ensemble_size
        )
        latent = policy._learn_model.recurrent_inference(
            latent, repeated_actions, member_id=member_ids
        ).latent_state
        recurrent_latents.append(latent)

    loss, log = policy._episodic_inverse_loss_from_replay_unroll(
        observations,
        actions,
        torch.ones(batch_size, horizon),
        torch.ones(batch_size),
        recurrent_latents,
        batch_size,
    )
    loss.backward()

    assert loss.item() > 0.0
    assert 0.0 <= log['episodic_inverse/inverse_accuracy'] <= 1.0
    assert log['episodic_inverse/valid_transition_count'] == batch_size * horizon
    assert log['episodic_inverse/train_world_model'] == 1.0
    assert any(
        parameter.grad is not None
        for parameter in policy._learn_model.episodic_inverse_encoder.parameters()
    )
    assert any(
        parameter.grad is not None
        for parameter in policy._learn_model.episodic_inverse_action_head.parameters()
    )
    assert all(
        any(parameter.grad is not None for parameter in bridge.parameters())
        for bridge in policy._learn_model.episodic_inverse_bridges
    )
    assert all(
        any(
            parameter.grad is not None
            for parameter in member.dynamics_network.parameters()
        )
        for member in policy._learn_model.members
    )


@pytest.mark.unittest
def test_intrinsic_value_training_uses_only_pure_exploration_samples():
    torch.manual_seed(0)
    policy = object.__new__(EnsembleMuZeroPolicy)
    policy._cfg = EasyDict(
        dict(
            device='cpu',
            num_unroll_steps=2,
            batch_size=2,
            multi_gpu=False,
            grad_clip_value=100.0,
            weight_decay=0.0,
            intrinsic_head_only=True,
            intrinsic_learning_rate=1e-3,
            intrinsic_model_loss_weight=1.0,
            intrinsic_reward_loss_weight=1.0,
            intrinsic_value_loss_weight=1.0,
            intrinsic_train_value_head=True,
            intrinsic_value_train_on_pure_exploration_only=True,
            intrinsic_loss_type='mse',
            intrinsic_reward_source='rnd',
            rnd_exploration=True,
            rnd_use_learned_intrinsic_model=True,
            store_selected_intrinsic_rewards=False,
            model=EasyDict(
                dict(
                    ensemble_size=1,
                    model_type='mlp',
                    observation_shape=2,
                    frame_stack_num=1,
                    self_supervised_learning_loss=False,
                    analysis_sim_norm=False,
                )
            ),
        )
    )
    policy._learn_model = EnsembleMuZeroModelMLP(
        ensemble_size=1,
        observation_shape=2,
        action_space_size=3,
        latent_state_dim=16,
        representation_last_linear_layer_init_zero=False,
        norm_type='LN',
    )
    policy._optimizer = torch.optim.Adam(
        policy._learn_model.parameters(), lr=1e-3
    )
    observations = np.asarray(
        [
            [-0.50, 0.00, -0.51, -0.01, -0.52, -0.01],
            [-0.40, 0.00, -0.39, 0.01, -0.37, 0.02],
        ],
        dtype=np.float32,
    )
    current_batch = (
        observations,
        np.asarray([[0, 0], [2, 2]], dtype=np.int64),
        np.ones((2, 3), dtype=np.float32),
        np.asarray([1.0, 0.0], dtype=np.float32),
        np.arange(2),
        np.ones(2, dtype=np.float32),
        np.zeros(2, dtype=np.float32),
    )
    reward_targets = torch.tensor([[1.0, 2.0], [9.0, 10.0]], dtype=torch.float32)
    value_targets = torch.tensor(
        [[2.0, 3.0, 4.0], [100.0, 100.0, 100.0]], dtype=torch.float32
    )

    root_observations = torch.from_numpy(observations[:, :2])
    actions = torch.as_tensor(current_batch[1], dtype=torch.long)
    latent_states = [
        policy._learn_model.initial_inference(
            root_observations, member_id=[0, 0]
        ).latent_state
    ]
    for depth in range(actions.shape[1]):
        latent_states.append(
            policy._learn_model.recurrent_inference(
                latent_states[-1], actions[:, depth], member_id=[0, 0]
            ).latent_state
        )
    _, log = policy._intrinsic_loss_from_primary_unroll(
        (current_batch, None),
        reward_targets,
        value_targets,
        [0, 0],
        latent_states,
        torch.ones(2, 3),
        torch.ones(2),
        original_batch_size=2,
    )

    assert log['intrinsic/reward_training_sample_count'] == 2.0
    assert log['intrinsic/value_training_sample_count'] == 1.0
    assert log['intrinsic/value_training_sample_fraction'] == 0.5
    assert log['intrinsic/value_pure_exploration_only'] == 1.0
    assert log['intrinsic/reward_training_target_mean'] == pytest.approx(5.5)
    assert log['intrinsic/value_training_target_mean'] == pytest.approx(3.0)


@pytest.mark.unittest
def test_ensemble_muzero_reward_only_intrinsic_parameters_exclude_value_head():
    policy = object.__new__(EnsembleMuZeroPolicy)
    policy._cfg = EasyDict(
        dict(
            intrinsic_head_only=True,
            intrinsic_train_value_head=False,
            model=dict(analysis_sim_norm=False),
        )
    )
    policy._learn_model = _FakeIntrinsicEnsemble()

    selected_params = set(policy._intrinsic_model_parameters())
    for member in policy._learn_model.members:
        assert member.intrinsic_reward_head.weight in selected_params
        assert member.intrinsic_value_head.weight not in selected_params
        assert member.shared.weight not in selected_params


@pytest.mark.unittest
def test_collection_episodic_novelty_embeddings_use_configured_real_encoder_member():
    policy = object.__new__(EnsembleMuZeroPolicy)
    policy._cfg = EasyDict(
        dict(
            device='cpu',
            episodic_novelty_encoder_member=1,
            model=dict(model_type='mlp', ensemble_size=2, analysis_sim_norm=False),
        )
    )
    model = EnsembleMuZeroModelMLP(
        ensemble_size=2,
        observation_shape=2,
        action_space_size=3,
        latent_state_dim=16,
        representation_last_linear_layer_init_zero=False,
        state_norm=False,
        norm_type='LN',
    )
    model.eval()
    policy._collect_model = model
    observations = np.asarray(
        [[[-0.5, 0.0]], [[-0.4, 0.01]]],
        dtype=np.float32,
    )

    embeddings = policy._collection_episodic_novelty_embeddings_for_observations(observations)
    expected = model.initial_inference(
        torch.as_tensor(observations.reshape(2, 2)),
        member_id=1,
    ).latent_state.detach().numpy()

    assert embeddings.shape == (2, 16)
    assert np.allclose(embeddings, expected)


@pytest.mark.unittest
def test_episodic_novelty_shadow_predictions_cover_every_root_action():
    policy = object.__new__(EnsembleMuZeroPolicy)
    policy._cfg = EasyDict(
        dict(
            intrinsic_reward_source='episodic_novelty',
            episodic_novelty_diagnostics=True,
            episodic_novelty_shadow_predictions=True,
            model=dict(ensemble_size=2, analysis_sim_norm=False),
        )
    )
    model = EnsembleMuZeroModelMLP(
        ensemble_size=2,
        observation_shape=2,
        action_space_size=3,
        latent_state_dim=16,
        state_norm=False,
        norm_type='LN',
    )
    model.eval()
    root_latents = torch.randn(2, 2, 16)

    predictions = policy._episodic_novelty_shadow_root_predictions(
        model, root_latents, [[0, 1, 2], [0, 2]]
    )

    assert set(predictions[0]) == {0, 1, 2}
    assert set(predictions[1]) == {0, 2}
    for root_predictions in predictions:
        for action_prediction in root_predictions.values():
            assert len(action_prediction['episodic_novelty_shadow_reward_members']) == 2
            assert np.isfinite(action_prediction['episodic_novelty_shadow_reward_mean'])


@pytest.mark.unittest
def test_episodic_history_shadow_predictions_do_not_advance_real_history():
    policy = object.__new__(EnsembleMuZeroPolicy)
    policy._cfg = EasyDict(
        dict(
            intrinsic_reward_source='episodic_novelty',
            episodic_novelty_diagnostics=True,
            episodic_novelty_history_shadow=True,
            model=dict(ensemble_size=2, analysis_sim_norm=False),
        )
    )
    policy._collect_env_episodic_history = {}
    policy._collect_env_episodic_history_pending = {}
    model = EnsembleMuZeroModelMLP(
        ensemble_size=2,
        observation_shape=2,
        action_space_size=3,
        latent_state_dim=16,
        episodic_history_dim=8,
        state_norm=False,
        norm_type='LN',
    )
    model.eval()
    root_latents = torch.randn(2, 2, 16)

    predictions = policy._episodic_novelty_history_shadow_root_predictions(
        model,
        root_latents,
        [[0, 1, 2], [0, 2]],
        [3, 7],
    )

    assert set(predictions[0]) == {0, 1, 2}
    assert set(predictions[1]) == {0, 2}
    assert policy._collect_env_episodic_history == {}
    assert policy._collect_env_episodic_history_pending == {}
    assert predictions[0][0]['episodic_novelty_history_state_norm_mean'] == 0.0
    assert predictions[0][0]['episodic_novelty_history_prediction_uses_one_step_only'] == 1.0


@pytest.mark.unittest
def test_episodic_history_is_committed_only_after_exact_reward_and_real_next_root():
    policy = object.__new__(EnsembleMuZeroPolicy)
    policy._cfg = EasyDict(
        dict(
            intrinsic_reward_source='episodic_novelty',
            episodic_novelty_diagnostics=True,
            episodic_novelty_history_shadow=True,
            model=dict(ensemble_size=2, analysis_sim_norm=False),
        )
    )
    policy._collect_env_episodic_history = {}
    policy._collect_env_episodic_history_pending = {
        3: {
            'current_latents': torch.randn(2, 16),
            'action': 2,
        }
    }
    model = EnsembleMuZeroModelMLP(
        ensemble_size=2,
        observation_shape=2,
        action_space_size=3,
        latent_state_dim=16,
        episodic_history_dim=8,
        state_norm=False,
        norm_type='LN',
    )
    model.eval()
    next_root_latents = torch.randn(1, 2, 16)

    policy._finalize_collection_episodic_histories(model, next_root_latents, [3])
    assert 3 in policy._collect_env_episodic_history_pending
    assert policy._collect_env_episodic_history == {}

    policy._record_collection_episodic_history_rewards([3], [7.5])
    policy._finalize_collection_episodic_histories(model, next_root_latents, [3])

    assert 3 not in policy._collect_env_episodic_history_pending
    assert policy._collect_env_episodic_history[3].shape == torch.Size([2, 8])
    assert torch.count_nonzero(policy._collect_env_episodic_history[3]).item() > 0


@pytest.mark.unittest
def test_episodic_history_shadow_trains_on_complete_sequences():
    torch.manual_seed(0)
    np.random.seed(0)
    policy = object.__new__(EnsembleMuZeroPolicy)
    policy._cfg = EasyDict(
        dict(
            intrinsic_reward_source='episodic_novelty',
            episodic_novelty_diagnostics=True,
            episodic_novelty_history_shadow=True,
            episodic_novelty_history_learning_rate=1e-3,
            episodic_novelty_history_train_epochs=1,
            episodic_novelty_history_train_batch_size=2,
            episodic_novelty_history_updates_per_collect=1,
            episodic_novelty_history_replay_capacity=100,
            episodic_novelty_history_recent_phase_count=2,
            episodic_novelty_history_recent_sample_ratio=0.5,
            episodic_novelty_history_grad_clip=5.0,
            intrinsic_loss_type='huber',
            intrinsic_huber_delta=1.0,
            device='cpu',
            model=dict(ensemble_size=2, analysis_sim_norm=False),
        )
    )
    policy._learn_model = EnsembleMuZeroModelMLP(
        ensemble_size=2,
        observation_shape=2,
        action_space_size=3,
        latent_state_dim=16,
        episodic_history_dim=8,
        representation_last_linear_layer_init_zero=False,
        state_norm=False,
        norm_type='LN',
    )
    policy._episodic_history_optimizer = None
    episodes = [
        {
            'observations': np.asarray(
                [[-0.5, 0.0], [-0.49, 0.01], [-0.48, 0.01], [-0.48, 0.0]],
                dtype=np.float32,
            ),
            'actions': np.asarray([2, 2, 1], dtype=np.int64),
            'intrinsic_rewards': np.asarray([1.0, 0.5, 0.0], dtype=np.float32),
        },
        {
            'observations': np.asarray(
                [[-0.55, 0.0], [-0.56, -0.01], [-0.57, -0.01], [-0.57, 0.0]],
                dtype=np.float32,
            ),
            'actions': np.asarray([0, 0, 1], dtype=np.int64),
            'intrinsic_rewards': np.asarray([1.0, 0.5, 0.0], dtype=np.float32),
        },
    ]
    before = torch.cat(
        [parameter.detach().reshape(-1) for parameter in policy._learn_model.episodic_history_parameters()]
    ).clone()

    stats = policy.train_episodic_history_shadow(episodes, collection_phase=0)

    after = torch.cat(
        [parameter.detach().reshape(-1) for parameter in policy._learn_model.episodic_history_parameters()]
    )
    assert stats['episodic_history/episode_count'] == 2.0
    assert stats['episodic_history/transition_count'] == 6.0
    assert stats['episodic_history/update_count'] == 1.0
    assert stats['episodic_history/replay_episode_count'] == 2.0
    assert stats['episodic_history/recent_sample_fraction'] == 1.0
    assert stats['episodic_history/hidden_norm_mean'] > 0.0
    assert not torch.allclose(before, after)


@pytest.mark.unittest
def test_episodic_history_replay_balances_recent_and_historical_complete_episodes():
    np.random.seed(0)
    policy = object.__new__(EnsembleMuZeroPolicy)
    policy._cfg = EasyDict(
        dict(
            episodic_novelty_history_train_batch_size=5,
            episodic_novelty_history_updates_per_collect=4,
            episodic_novelty_history_replay_capacity=100,
            episodic_novelty_history_recent_phase_count=2,
            episodic_novelty_history_recent_sample_ratio=0.5,
            model=dict(analysis_sim_norm=False),
        )
    )

    def make_episode(phase, episode_index):
        return {
            'observations': np.asarray([[phase, 0.0], [phase, 0.1]], dtype=np.float32),
            'actions': np.asarray([episode_index % 3], dtype=np.int64),
            'intrinsic_rewards': np.asarray([float(episode_index)], dtype=np.float32),
            'collection_phase': phase,
            'collection_episode_index': episode_index,
        }

    for phase in range(3):
        policy._add_episodic_history_replay_episodes(
            [make_episode(phase, phase * 2), make_episode(phase, phase * 2 + 1)],
            collection_phase=phase,
        )

    batches, stats = policy._sample_episodic_history_replay_batches(collection_phase=2)
    sampled = [episode for batch in batches for episode in batch]
    recent = [episode for episode in sampled if episode['collection_phase'] in {1, 2}]
    historical = [episode for episode in sampled if episode['collection_phase'] == 0]

    assert len(batches) == 4
    assert all(len(batch) == 5 for batch in batches)
    assert len(recent) == 10
    assert len(historical) == 10
    assert stats['episodic_history/recent_pool_episode_count'] == 4.0
    assert stats['episodic_history/historical_pool_episode_count'] == 2.0
    assert stats['episodic_history/recent_sample_fraction'] == 0.5
    recent_reuse = {}
    for episode in recent:
        replay_id = episode['replay_id']
        recent_reuse[replay_id] = recent_reuse.get(replay_id, 0) + 1
    assert max(recent_reuse.values()) - min(recent_reuse.values()) <= 1


@pytest.mark.unittest
def test_episodic_history_replay_state_round_trip_and_capacity():
    policy = object.__new__(EnsembleMuZeroPolicy)
    policy._cfg = EasyDict(
        dict(
            episodic_novelty_history_replay_capacity=2,
            model=dict(analysis_sim_norm=False),
        )
    )

    for phase in range(3):
        policy._add_episodic_history_replay_episodes(
            [
                {
                    'observations': np.asarray([[phase, 0.0], [phase, 0.1]], dtype=np.float32),
                    'actions': np.asarray([phase % 3], dtype=np.int64),
                    'intrinsic_rewards': np.asarray([0.5], dtype=np.float32),
                    'collection_phase': phase,
                }
            ],
            collection_phase=phase,
        )

    assert [episode['collection_phase'] for episode in policy._episodic_history_episode_replay] == [1, 2]
    saved_state = policy._episodic_history_replay_state_dict()

    restored = object.__new__(EnsembleMuZeroPolicy)
    restored._cfg = policy._cfg
    restored._load_episodic_history_replay_state(saved_state)

    assert [episode['collection_phase'] for episode in restored._episodic_history_episode_replay] == [1, 2]
    assert restored._episodic_history_replay_next_id == 3


@pytest.mark.unittest
def test_episodic_novelty_targets_are_reward_only_and_mask_padding():
    policy = object.__new__(EnsembleMuZeroPolicy)
    policy._cfg = EasyDict(
        dict(
            episodic_novelty_diagnostics=True,
            store_selected_intrinsic_rewards=True,
            num_unroll_steps=2,
            device='cpu',
            model=dict(model_type='mlp', observation_shape=2, analysis_sim_norm=False),
        )
    )
    current_batch = [
        np.asarray([[-0.5, 0.0, -0.49, 0.001, -0.48, 0.002]], dtype=np.float32),
        np.asarray([[2, 2]], dtype=np.int64),
        np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        np.asarray([[0.4, 0.9]], dtype=np.float32),
        np.asarray(['0']),
        np.asarray([1.0], dtype=np.float32),
        np.asarray([0.0], dtype=np.float32),
    ]

    log, reward_targets, value_targets = policy._prepare_episodic_novelty_targets(
        [current_batch, []]
    )

    assert torch.equal(reward_targets, torch.tensor([[0.4, 0.0]]))
    assert value_targets is None
    assert log['episodic_novelty/value_head_trained'] == 0.0


@pytest.mark.unittest
def test_ensemble_muzero_intrinsic_targets_are_clipped_and_logged():
    policy = object.__new__(EnsembleMuZeroPolicy)
    policy._cfg = EasyDict(
        dict(
            intrinsic_reward_target_clip=2.0,
            intrinsic_value_target_clip=5.0,
            model=dict(analysis_sim_norm=False),
        )
    )

    reward_targets = torch.tensor([[-3.0, 0.5, 4.0]])
    value_targets = torch.tensor([[-10.0, 1.0, 12.0]])
    clipped_reward, clipped_value, log = policy._clip_intrinsic_targets(reward_targets, value_targets)

    assert torch.equal(clipped_reward, torch.tensor([[-2.0, 0.5, 2.0]]))
    assert torch.equal(clipped_value, torch.tensor([[-5.0, 1.0, 5.0]]))
    assert log['intrinsic/reward_target_unclipped_abs_max'] == 4.0
    assert log['intrinsic/value_target_unclipped_abs_max'] == 12.0


@pytest.mark.unittest
def test_ensemble_muzero_rnd_bonus_normalization_modes():
    policy = object.__new__(EnsembleMuZeroPolicy)
    policy._rnd_bonus_stats_initialized = True
    policy._rnd_bonus_mean = 2.0
    policy._rnd_bonus_var = 4.0
    errors = torch.tensor([1.0, 2.0, 5.0, 20.0])

    policy._cfg = EasyDict(
        dict(
            rnd_bonus_normalization='raw',
            rnd_bonus_clip=10.0,
            model=dict(analysis_sim_norm=False),
        )
    )
    raw_bonus = policy._normalize_rnd_errors(errors)
    assert torch.equal(raw_bonus, torch.tensor([1.0, 2.0, 5.0, 10.0]))

    policy._cfg.rnd_bonus_normalization = 'zscore'
    zscore_bonus = policy._normalize_rnd_errors(errors)
    assert torch.equal(zscore_bonus, torch.tensor([0.0, 0.0, 1.5, 9.0]))


@pytest.mark.unittest
def test_ensemble_muzero_intrinsic_loss_supports_huber_and_mse():
    policy = object.__new__(EnsembleMuZeroPolicy)
    policy._cfg = EasyDict(
        dict(
            intrinsic_loss_type='huber',
            intrinsic_huber_delta=1.0,
            model=dict(analysis_sim_norm=False),
        )
    )

    prediction = torch.tensor([0.0, 3.0])
    target = torch.tensor([0.0, 0.0])

    huber_loss = policy._intrinsic_elementwise_loss(prediction, target)
    assert torch.allclose(huber_loss, torch.tensor([0.0, 2.5]))

    policy._cfg.intrinsic_loss_type = 'mse'
    mse_loss = policy._intrinsic_elementwise_loss(prediction, target)
    assert torch.allclose(mse_loss, torch.tensor([0.0, 9.0]))


@pytest.mark.unittest
def test_ensemble_muzero_rnd_target_snapshot_freezes_target_errors():
    policy = object.__new__(EnsembleMuZeroPolicy)
    policy._cfg = EasyDict(
        dict(
            rnd_exploration=True,
            rnd_hidden_dim=8,
            rnd_feature_dim=4,
            rnd_learning_rate=1e-2,
            rnd_train_batch_size=32,
            rnd_train_on_new_data_only=False,
            rnd_predictor_update_proportion=1.0,
            rnd_target_snapshot=True,
            rnd_target_snapshot_update_interval=100,
            device='cpu',
            model=dict(analysis_sim_norm=False),
        )
    )
    policy._init_rnd_state()
    policy._learn_update_count = 0
    inputs = torch.randn(16, 2)

    policy._ensure_rnd_model(inputs)
    snapshot_before = {
        key: value.detach().clone()
        for key, value in policy._rnd_target_snapshot_model.state_dict().items()
    }
    error_before = policy._rnd_target_model_for_targets().error(inputs).detach()

    policy._train_rnd_on_inputs(inputs)
    policy._learn_update_count = 1
    error_after = policy._rnd_target_model_for_targets().error(inputs).detach()

    assert torch.allclose(error_before, error_after)
    for key, value in policy._rnd_target_snapshot_model.state_dict().items():
        assert torch.allclose(value, snapshot_before[key])

    policy._learn_update_count = 101
    policy._sync_rnd_target_snapshot()
    refreshed_error = policy._rnd_target_model_for_targets().error(inputs).detach()
    assert not torch.allclose(error_before, refreshed_error)


@pytest.mark.unittest
def test_ensemble_muzero_rnd_snapshot_refreshes_only_at_collection_boundary():
    policy = object.__new__(EnsembleMuZeroPolicy)
    policy._cfg = EasyDict(
        dict(
            rnd_exploration=True,
            rnd_hidden_dim=8,
            rnd_feature_dim=4,
            rnd_learning_rate=1e-2,
            rnd_train_batch_size=32,
            rnd_train_on_new_data_only=False,
            rnd_predictor_update_proportion=1.0,
            rnd_target_snapshot=True,
            rnd_target_snapshot_update_interval=None,
            rnd_target_snapshot_refresh_per_collect=True,
            update_per_collect=None,
            device='cpu',
            model=dict(analysis_sim_norm=False),
        )
    )
    policy._init_rnd_state()
    policy._learn_update_count = 0
    inputs = torch.randn(16, 2)

    policy._ensure_rnd_model(inputs)
    error_before = policy._rnd_target_model_for_targets().error(inputs).detach()
    policy._train_rnd_on_inputs(inputs)
    policy._learn_update_count = 100

    assert torch.allclose(
        error_before,
        policy._rnd_target_model_for_targets().error(inputs).detach(),
    )

    policy.refresh_rnd_target_snapshot_for_collection()
    refreshed_error = policy._rnd_target_model_for_targets().error(inputs).detach()
    assert not torch.allclose(error_before, refreshed_error)


@pytest.mark.unittest
def test_rnd_landscape_query_counts_completed_collection_when_refreshing_snapshot():
    torch.manual_seed(17)
    policy = object.__new__(EnsembleMuZeroPolicy)
    policy._cfg = EasyDict(
        dict(
            rnd_exploration=True,
            intrinsic_reward_source='rnd',
            rnd_hidden_dim=8,
            rnd_feature_dim=4,
            rnd_learning_rate=1e-2,
            rnd_train_batch_size=32,
            rnd_train_on_new_data_only=False,
            rnd_predictor_update_proportion=1.0,
            rnd_target_snapshot=True,
            rnd_target_snapshot_refresh_per_collect=True,
            rnd_observation_dim=2,
            rnd_observation_indices=None,
            rnd_observation_low=(-1.2, -0.07),
            rnd_observation_high=(0.6, 0.07),
            rnd_landscape_query=True,
            rnd_landscape_query_count=2,
            rnd_landscape_query_hidden_dim=8,
            rnd_landscape_query_learning_rate=1e-3,
            device='cpu',
            model=dict(
                analysis_sim_norm=False,
                ensemble_size=2,
                latent_state_dim=8,
            ),
        )
    )
    policy._init_rnd_state()
    inputs = torch.tensor(
        [[-0.9, -0.03], [-0.5, 0.0], [-0.1, 0.04]], dtype=torch.float32
    )
    policy._ensure_rnd_model(inputs)
    previous_error = policy._rnd_target_snapshot_model.error(inputs).detach()

    for _ in range(10):
        policy._train_rnd_on_inputs(inputs)
    policy.refresh_rnd_target_snapshot_for_collection(collection_completed=True)

    assert policy._rnd_landscape_query_snapshot_count == 1
    refreshed_error = policy._rnd_target_snapshot_model.error(inputs).detach()
    assert not torch.allclose(refreshed_error, previous_error)


@pytest.mark.unittest
def test_rnd_landscape_query_normalized_raw_scale_is_phase_fixed_replay_mean():
    class _PhysicalError(nn.Module):
        def error(self, states):
            return states[:, 0].abs() + 0.5 * states[:, 1].abs() + 0.01

    policy = object.__new__(EnsembleMuZeroPolicy)
    policy._cfg = EasyDict(
        dict(
            rnd_exploration=True,
            intrinsic_reward_source='rnd',
            intrinsic_head_only=False,
            rnd_target_snapshot=True,
            rnd_observation_dim=2,
            rnd_observation_indices=None,
            rnd_landscape_query=True,
            rnd_landscape_query_count=2,
            rnd_landscape_query_hidden_dim=8,
            rnd_landscape_query_target_transform='normalized_raw',
            rnd_landscape_query_normalized_raw_clip=10.0,
            rnd_landscape_query_normalized_raw_min_scale=1e-8,
            device='cpu',
            model=dict(
                analysis_sim_norm=False,
                ensemble_size=2,
                latent_state_dim=8,
            ),
        )
    )
    policy._init_rnd_state()
    policy._rnd_target_snapshot_model = _PhysicalError()
    observations = np.asarray(
        [[-0.50, 0.00], [-0.25, 0.02], [0.10, -0.04]], dtype=np.float32
    )
    expected_errors = policy._rnd_target_snapshot_model.error(
        torch.from_numpy(observations)
    )

    stats = policy.refresh_rnd_landscape_query_target_scale(
        observations, collection_phase=7
    )

    assert stats['rnd_landscape_query/target_scale'] == pytest.approx(
        expected_errors.mean().item()
    )
    assert stats['rnd_landscape_query/target_scale_sample_count'] == 3.0
    assert stats['rnd_landscape_query/target_scale_phase'] == 7.0
    assert policy._rnd_landscape_query_model.target_scale.item() == pytest.approx(
        expected_errors.mean().item()
    )


@pytest.mark.unittest
def test_rnd_landscape_query_scores_imagined_tree_paths():
    class _PhysicalError(nn.Module):
        def error(self, states):
            return 0.01 + (states[:, 0] + 0.2).pow(2) + 4.0 * states[:, 1].pow(2)

    torch.manual_seed(23)
    policy = object.__new__(EnsembleMuZeroPolicy)
    policy._cfg = EasyDict(
        dict(
            rnd_exploration=True,
            intrinsic_reward_source='rnd',
            rnd_target_snapshot=True,
            rnd_target_snapshot_refresh_per_collect=True,
            rnd_bonus_normalization='raw',
            rnd_bonus_clip=5.0,
            rnd_landscape_query=True,
            rnd_landscape_query_max_horizon=4,
            device='cpu',
            model=dict(
                analysis_sim_norm=False,
                ensemble_size=2,
                latent_state_dim=8,
            ),
        )
    )
    policy._init_rnd_state()
    policy._rnd_landscape_query_model = MountainCarRNDLandscapeQuery(
        latent_dim=8, query_count=3, hidden_dim=16
    )
    policy._rnd_target_snapshot_model = _PhysicalError()
    roots = torch.tensor([[-0.5, 0.0]], dtype=torch.float32)
    context = policy._rnd_landscape_query_model.query(
        roots, policy._rnd_target_snapshot_model.error
    )
    query_features = torch.cat(
        [
            context.query_states,
            context.query_states.square(),
            torch.sin(context.query_states),
            torch.cos(context.query_states),
        ],
        dim=-1,
    )
    context.query_latents = query_features.unsqueeze(2).expand(-1, -1, 2, -1)
    path_reward_fn = policy._rnd_landscape_path_reward_fn(context)

    imagined_latents = torch.randn(1, 2, 8)
    rewards = path_reward_fn([0], [[2, 2]], imagined_latents)
    diagnostics = [[{'action': 0}, {'action': 1}, {'action': 2}]]
    policy._annotate_rnd_landscape_query_diagnostics(
        context,
        [[0, 1, 2]],
        diagnostics,
        path_reward_fn=path_reward_fn,
    )

    assert rewards.shape == (1,)
    assert np.isfinite(rewards[0])
    assert len(path_reward_fn.records) == 1
    for action_terms in diagnostics[0]:
        assert action_terms['rnd_landscape_query_tree_edge_count'] == 1
        assert action_terms['rnd_landscape_query_tree_depth_2_edge_count'] == 1
        assert np.isfinite(action_terms['rnd_landscape_query_predicted_one_step_reward'])


@pytest.mark.unittest
def test_rnd_landscape_path_reward_applies_raw_episodic_memory_gate():
    class _FixedReader:
        @staticmethod
        def predict_latent_error(context, imagined_latents, root_indices=None):
            return torch.ones(
                imagined_latents.shape[0], device=imagined_latents.device
            )

    policy = object.__new__(EnsembleMuZeroPolicy)
    policy._cfg = EasyDict(
        dict(
            rnd_bonus_normalization='raw',
            rnd_bonus_clip=5.0,
            rnd_episodic_memory=True,
            rnd_episodic_memory_key_space='raw',
            rnd_episodic_memory_kernel_bandwidth=1.0,
            rnd_episodic_memory_min_reward_factor=0.05,
            rnd_episodic_memory_distance_epsilon=1e-8,
            rnd_episodic_memory_include_path=True,
            model=dict(ensemble_size=2, analysis_sim_norm=False),
        )
    )
    policy._rnd_landscape_query_model = _FixedReader()
    context = EasyDict(root_states=torch.zeros(1, 2))
    imagined_latents = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    memory = EpisodicLatentMemoryContext(
        keys=imagined_latents.clone(), transition_distance_scale=1.0
    )

    reward_fn = policy._rnd_landscape_path_reward_fn(
        context,
        world_model=nn.Identity(),
        episodic_memories=[memory],
    )
    rewards = reward_fn([0], [[2]], imagined_latents)

    assert rewards.tolist() == pytest.approx([0.05])
    assert reward_fn.records[0]['base_predicted_reward'] == pytest.approx(1.0)
    assert reward_fn.records[0]['predicted_reward'] == pytest.approx(0.05)
    assert reward_fn.records[0]['episodic_real_distance'] == pytest.approx(0.0)


@pytest.mark.unittest
@pytest.mark.parametrize(
    'mode, expected_memory_size',
    [('minimum_distance', 1), ('ngu_knn', 12)],
)
def test_rnd_episodic_memory_retains_repeated_visits_for_ngu_counts(
        mode, expected_memory_size
):
    class _InverseEmbeddingModel:
        @staticmethod
        def episodic_inverse_embedding(observations):
            return observations.float().reshape(observations.shape[0], -1)

    policy = object.__new__(EnsembleMuZeroPolicy)
    policy._cfg = EasyDict(
        dict(
            rnd_episodic_memory=True,
            rnd_episodic_memory_key_space='inverse',
            rnd_episodic_memory_mode=mode,
            rnd_episodic_memory_max_size=20,
            rnd_episodic_memory_distance_epsilon=1e-8,
            model=dict(ensemble_size=1, analysis_sim_norm=False),
        )
    )
    memory_by_env = {}
    observation = torch.tensor([[1.0, -1.0]])
    latent = torch.zeros(1, 1, 2)

    for _ in range(12):
        contexts = policy._rnd_episodic_memory_contexts(
            _InverseEmbeddingModel(),
            observation,
            latent,
            ready_env_ids=[0],
            memory_by_env=memory_by_env,
            model_version=0,
        )

    assert contexts[0].keys.shape == (expected_memory_size, 1, 2)
    assert len(memory_by_env[0]['observations']) == expected_memory_size
    assert len(memory_by_env[0]['transition_distances']) == 11


@pytest.mark.unittest
@pytest.mark.parametrize('train_world_model', [True, False])
@pytest.mark.parametrize('reader_architecture', ['landscape', 'ordinary'])
def test_rnd_landscape_replay_loss_respects_world_model_detachment(
        train_world_model, reader_architecture
):
    torch.manual_seed(29)
    policy = object.__new__(EnsembleMuZeroPolicy)
    policy._cfg = EasyDict(
        dict(
            rnd_exploration=True,
            intrinsic_reward_source='rnd',
            rnd_hidden_dim=8,
            rnd_feature_dim=4,
            rnd_learning_rate=1e-3,
            rnd_target_snapshot=True,
            rnd_observation_dim=2,
            rnd_observation_indices=None,
            rnd_observation_low=(-1.2, -0.07),
            rnd_observation_high=(0.6, 0.07),
            rnd_landscape_query=True,
            rnd_landscape_query_count=2,
            rnd_landscape_query_hidden_dim=8,
            rnd_landscape_query_learning_rate=1e-3,
            rnd_landscape_query_loss_weight=1.0,
            rnd_landscape_query_max_horizon=2,
            rnd_landscape_query_diversity_weight=0.0,
            rnd_landscape_query_centered_prediction_weight=0.0,
            rnd_landscape_query_grad_clip=5.0,
            rnd_landscape_query_train_world_model=train_world_model,
            rnd_landscape_query_reader_use_context=False,
            rnd_landscape_query_reader_architecture=reader_architecture,
            rnd_landscape_query_reader_hidden_channels=(32,),
            rnd_landscape_query_reader_norm_type='BN',
            rnd_landscape_query_reader_last_linear_layer_init_zero=False,
            rnd_landscape_query_target_log_reference=1.0,
            rnd_landscape_query_error_epsilon=1e-10,
            rnd_landscape_query_max_raw_error=None,
            intrinsic_head_only=False,
            update_per_collect=1,
            device='cpu',
            model=EasyDict(
                dict(
                    analysis_sim_norm=False,
                    ensemble_size=2,
                    latent_state_dim=8,
                    model_type='mlp',
                    observation_shape=2,
                    frame_stack_num=1,
                )
            ),
        )
    )
    policy._init_rnd_state()
    policy._learn_model = EnsembleMuZeroModelMLP(
        ensemble_size=2,
        observation_shape=2,
        action_space_size=3,
        latent_state_dim=8,
        representation_last_linear_layer_init_zero=False,
        state_norm=False,
        norm_type='LN',
    )
    roots = torch.tensor([[-0.55, 0.0], [-0.35, 0.01]], dtype=torch.float32)
    policy._ensure_rnd_model(roots)
    query_model = policy._ensure_rnd_landscape_query_model()
    future_states = torch.tensor(
        [
            [[-0.56, -0.01], [-0.54, 0.01]],
            [[-0.34, 0.01], [-0.32, 0.02]],
        ],
        dtype=torch.float32,
    )
    replay_observations = torch.cat(
        [roots.unsqueeze(1), future_states], dim=1
    ).reshape(roots.shape[0], -1).numpy()
    actions = torch.tensor([[0, 2], [2, 2]], dtype=torch.long)
    ensemble_size = 2
    member_ids = [0, 0, 1, 1]
    expanded_roots = torch.cat([roots for _ in range(ensemble_size)], dim=0)
    expanded_actions = torch.cat([actions for _ in range(ensemble_size)], dim=0)
    latent = policy._learn_model.initial_inference(
        expanded_roots, member_id=member_ids
    ).latent_state
    recurrent_latents = []
    for depth in range(actions.shape[1]):
        latent = policy._learn_model.recurrent_inference(
            latent, expanded_actions[:, depth], member_id=member_ids
        ).latent_state
        recurrent_latents.append(latent)

    loss, log = policy._rnd_landscape_query_loss_from_replay_unroll(
        replay_observations,
        torch.ones(roots.shape[0], actions.shape[1]),
        torch.ones(roots.shape[0]),
        recurrent_latents,
        original_batch_size=roots.shape[0],
    )
    policy._learn_model.zero_grad()
    query_model.zero_grad()
    loss.backward()

    representation_parameters = [
        parameter
        for member in policy._learn_model.members
        for parameter in member.representation_network.parameters()
    ]
    dynamics_parameters = [
        parameter
        for member in policy._learn_model.members
        for parameter in member.dynamics_network.parameters()
    ]
    assert log['rnd_landscape_query/joint_world_model_update'] == float(
        train_world_model
    )
    assert log[
        f'rnd_landscape_query/reader_architecture_{reader_architecture}'
    ] == 1.0
    assert log['rnd_landscape_query/depth_1_valid_fraction'] == pytest.approx(1.0)
    representation_has_grad = any(
        parameter.grad is not None and parameter.grad.abs().sum().item() > 0
        for parameter in representation_parameters
    )
    dynamics_has_grad = any(
        parameter.grad is not None and parameter.grad.abs().sum().item() > 0
        for parameter in dynamics_parameters
    )
    assert representation_has_grad is train_world_model
    assert dynamics_has_grad is train_world_model
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum().item() > 0
        for parameter in query_model.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in policy._rnd_target_snapshot_model.parameters()
    )


@pytest.mark.unittest
def test_rnd_initialization_seed_is_independent_of_ambient_rng():
    def initialized_rnd_state(ambient_seed):
        policy = object.__new__(EnsembleMuZeroPolicy)
        policy._cfg = EasyDict(
            dict(
                rnd_exploration=True,
                rnd_hidden_dim=8,
                rnd_feature_dim=4,
                rnd_learning_rate=1e-3,
                rnd_target_snapshot=True,
                rnd_initialization_seed=12345,
                rnd_observation_low=(-1.2, -0.07),
                rnd_observation_high=(0.6, 0.07),
                update_per_collect=1,
                device='cpu',
                model=EasyDict(dict(analysis_sim_norm=False)),
            )
        )
        policy._init_rnd_state()
        torch.manual_seed(ambient_seed)
        torch.randn(31)
        policy._ensure_rnd_model(torch.zeros(2, 2))
        return {
            key: value.detach().clone()
            for key, value in policy._rnd_model.state_dict().items()
        }

    first = initialized_rnd_state(1)
    second = initialized_rnd_state(999)
    assert first.keys() == second.keys()
    assert all(torch.equal(first[key], second[key]) for key in first)


@pytest.mark.unittest
def test_rnd_auxiliary_initialization_is_fixed_and_preserves_ambient_rng():
    def initialized_query_state(ambient_seed):
        policy = object.__new__(EnsembleMuZeroPolicy)
        policy._cfg = EasyDict(
            dict(
                device='cpu',
                intrinsic_reward_source='rnd',
                intrinsic_head_only=False,
                rnd_target_snapshot=True,
                rnd_observation_dim=2,
                rnd_landscape_query=True,
                rnd_landscape_query_count=2,
                rnd_landscape_query_hidden_dim=8,
                rnd_landscape_query_reader_use_context=False,
                rnd_landscape_query_initialization_seed=24680,
                model=EasyDict(
                    dict(
                        analysis_sim_norm=False,
                        ensemble_size=2,
                        latent_state_dim=8,
                    )
                ),
            )
        )
        policy._init_rnd_state()
        torch.manual_seed(ambient_seed)
        torch.randn(17)
        rng_before = torch.get_rng_state().clone()
        model = policy._ensure_rnd_landscape_query_model()
        rng_after = torch.get_rng_state().clone()
        return rng_before, rng_after, {
            key: value.detach().clone() for key, value in model.state_dict().items()
        }

    first_before, first_after, first = initialized_query_state(1)
    second_before, second_after, second = initialized_query_state(999)

    assert torch.equal(first_before, first_after)
    assert torch.equal(second_before, second_after)
    assert first.keys() == second.keys()
    assert all(torch.equal(first[key], second[key]) for key in first)


@pytest.mark.unittest
def test_ensemble_muzero_fresh_rnd_training_uses_only_full_batches_and_carries_remainder():
    policy = object.__new__(EnsembleMuZeroPolicy)
    policy._cfg = EasyDict(
        dict(
            rnd_exploration=True,
            intrinsic_reward_source='rnd',
            rnd_hidden_dim=8,
            rnd_feature_dim=4,
            rnd_learning_rate=1e-2,
            rnd_predictor_update_proportion=1.0,
            rnd_train_from_fresh_collection_only=True,
            rnd_fresh_update_batch_size=4,
            rnd_target_snapshot=True,
            rnd_target_snapshot_update_interval=100,
            rnd_target_snapshot_refresh_per_collect=True,
            rnd_observation_dim=2,
            rnd_observation_indices=None,
            rnd_observation_low=(-1.2, -0.07),
            rnd_observation_high=(0.6, 0.07),
            device='cpu',
            model=dict(analysis_sim_norm=False),
        )
    )
    policy._init_rnd_state()
    first_observations = np.stack(
        [
            np.linspace(-1.2, 0.2, 10, dtype=np.float32),
            np.linspace(-0.06, 0.06, 10, dtype=np.float32),
        ],
        axis=1,
    )

    first_log = policy.train_rnd_on_fresh_observations(first_observations)
    snapshot_error = policy._rnd_target_model_for_targets().error(
        torch.as_tensor(first_observations)
    ).detach()
    online_error = policy._rnd_model.error(torch.as_tensor(first_observations)).detach()

    assert first_log['rnd_fresh/selected_count'] == 10.0
    assert first_log['rnd_fresh/optimizer_step_count'] == 2.0
    assert first_log['rnd_fresh/trained_sample_count'] == 8.0
    assert first_log['rnd_fresh/batch_size_min'] == 4.0
    assert first_log['rnd_fresh/batch_size_max'] == 4.0
    assert first_log['rnd_fresh/pending_count'] == 2.0
    assert not torch.allclose(snapshot_error, online_error)

    second_log = policy.train_rnd_on_fresh_observations(first_observations[:2])

    assert second_log['rnd_fresh/selected_count'] == 2.0
    assert second_log['rnd_fresh/optimizer_step_count'] == 1.0
    assert second_log['rnd_fresh/trained_sample_count'] == 4.0
    assert second_log['rnd_fresh/pending_count'] == 0.0
    assert torch.allclose(
        snapshot_error,
        policy._rnd_target_model_for_targets().error(
            torch.as_tensor(first_observations)
        ).detach(),
    )


@pytest.mark.unittest
def test_ensemble_muzero_rnd_observation_keys_and_new_data_filter():
    policy = object.__new__(EnsembleMuZeroPolicy)
    policy._cfg = EasyDict(
        dict(
            model=dict(analysis_sim_norm=False),
            rnd_train_on_new_data_only=True,
            rnd_predictor_update_proportion=1.0,
            rnd_ignore_elite_replay_for_predictor=True,
        )
    )
    policy._rnd_trained_observation_keys = {'7:0'}

    assert policy._rnd_observation_keys([7, 8], 2) == ['7:0', '7:1', '8:0', '8:1']

    (
        indices,
        selected_keys,
        processed_keys,
        candidate_count,
        skipped_seen_count,
        skipped_elite_count,
    ) = policy._select_rnd_training_indices(
        4,
        torch.device('cpu'),
        ['7:0', '7:1', '8:0', '8:1'],
    )

    assert indices.tolist() == [1, 2, 3]
    assert selected_keys == ['7:1', '8:0', '8:1']
    assert processed_keys == ['7:1', '8:0', '8:1']
    assert candidate_count == 3
    assert skipped_seen_count == 1
    assert skipped_elite_count == 0

    policy._cfg.rnd_predictor_update_proportion = 0.0
    (
        indices,
        selected_keys,
        processed_keys,
        candidate_count,
        skipped_seen_count,
        skipped_elite_count,
    ) = policy._select_rnd_training_indices(
        3,
        torch.device('cpu'),
        ['9:0', '9:1', '9:2'],
    )

    assert indices.tolist() == []
    assert selected_keys == []
    assert processed_keys == ['9:0', '9:1', '9:2']
    assert candidate_count == 3
    assert skipped_seen_count == 0
    assert skipped_elite_count == 0

    policy._cfg.rnd_predictor_update_proportion = 1.0
    (
        indices,
        selected_keys,
        processed_keys,
        candidate_count,
        skipped_seen_count,
        skipped_elite_count,
    ) = policy._select_rnd_training_indices(
        4,
        torch.device('cpu'),
        ['elite:0:3:0', '11:0', 'intrinsic:1:4:0', '11:1'],
    )

    assert indices.tolist() == [1, 3]
    assert selected_keys == ['11:0', '11:1']
    assert processed_keys == ['11:0', '11:1']
    assert candidate_count == 2
    assert skipped_seen_count == 0
    assert skipped_elite_count == 2

    policy._rnd_bonus_mean = 0.0
    policy._rnd_bonus_var = 1.0
    policy._rnd_bonus_stats_initialized = False
    policy._rnd_trained_observation_keys = {'10:0'}

    stats_count = policy._update_rnd_bonus_stats_for_inputs(torch.tensor([1.0, 3.0, 5.0]), ['10:0', '10:1', 'elite:2:0:0'])

    assert stats_count == 1
    assert policy._rnd_bonus_mean == 3.0


@pytest.mark.unittest
def test_ensemble_muzero_rnd_observation_projection_supports_feature_selection():
    policy = object.__new__(EnsembleMuZeroPolicy)
    policy._cfg = EasyDict(
        dict(
            rnd_observation_dim=2,
            rnd_observation_indices=None,
            model=dict(analysis_sim_norm=False),
        )
    )

    obs = torch.tensor([[[-0.5, 0.01], [-0.4, 0.02]], [[-0.3, 0.03], [-0.2, 0.04]]])

    projected = policy._select_rnd_observation_features(obs)

    assert projected.shape == (2, 2, 2)
    assert torch.allclose(projected, obs)

    policy._cfg.rnd_observation_indices = [1, 0]
    projected_by_index = policy._select_rnd_observation_features(obs)
    assert torch.allclose(projected_by_index, obs[..., [1, 0]])


@pytest.mark.unittest
def test_ensemble_muzero_rnd_uses_fixed_feature_bounds_without_collapsing_states():
    policy = object.__new__(EnsembleMuZeroPolicy)
    policy._cfg = EasyDict(
        dict(
            rnd_exploration=True,
            rnd_hidden_dim=8,
            rnd_feature_dim=4,
            rnd_learning_rate=1e-3,
            rnd_target_snapshot=True,
            rnd_target_snapshot_update_interval=100,
            rnd_observation_low=(-1.2, -0.07),
            rnd_observation_high=(0.6, 0.07),
            device='cpu',
            model=dict(analysis_sim_norm=False),
        )
    )
    policy._init_rnd_state()

    observations = torch.tensor(
        [
            [-1.2, -0.07],
            [-0.3, 0.0],
            [0.6, 0.07],
        ],
        dtype=torch.float32,
    )
    policy._ensure_rnd_model(observations)

    normalized = policy._rnd_model._flatten(observations)

    assert torch.allclose(
        normalized,
        torch.tensor([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]]),
        atol=1e-6,
    )
    assert not torch.allclose(normalized[0], normalized[1])
    assert not torch.allclose(normalized[1], normalized[2])


@pytest.mark.unittest
def test_ensemble_muzero_rnd_fixed_bounds_clip_out_of_range_observations():
    policy = object.__new__(EnsembleMuZeroPolicy)
    policy._cfg = EasyDict(
        dict(
            rnd_exploration=True,
            rnd_hidden_dim=8,
            rnd_feature_dim=4,
            rnd_learning_rate=1e-3,
            rnd_target_snapshot=False,
            rnd_observation_low=(-1.2, -0.07),
            rnd_observation_high=(0.6, 0.07),
            device='cpu',
            model=dict(analysis_sim_norm=False),
        )
    )
    policy._init_rnd_state()
    observations = torch.tensor([[-2.0, -0.1], [1.0, 0.1]], dtype=torch.float32)
    policy._ensure_rnd_model(observations)

    normalized = policy._rnd_model._flatten(observations)

    assert torch.equal(normalized, torch.tensor([[-1.0, -1.0], [1.0, 1.0]]))


@pytest.mark.unittest
def test_muzero_buffer_extends_obs_window_for_intrinsic_bootstrap_targets():
    buffer = object.__new__(MuZeroGameBuffer)
    buffer._cfg = EasyDict(
        dict(
            num_unroll_steps=5,
            td_steps=10,
            rnd_exploration=True,
            rnd_use_learned_intrinsic_model=True,
            rnd_intrinsic_td_steps=20,
            model=dict(model_type='mlp'),
        )
    )

    assert buffer._current_batch_obs_unroll_steps() == 25

    buffer._cfg.model.model_type = 'conv'
    assert buffer._current_batch_obs_unroll_steps() == 25

    buffer._cfg.rnd_exploration = False
    buffer._cfg.intrinsic_reward_source = 'state_value_disagreement'
    assert buffer._current_batch_obs_unroll_steps() == 25

    buffer._cfg.intrinsic_reward_source = 'state_density'
    assert buffer._current_batch_obs_unroll_steps() == 25


@pytest.mark.unittest
def test_one_step_action_qstd_rewards_are_action_specific_scaled_clipped_and_cached():
    policy = object.__new__(EnsembleMuZeroPolicy)
    policy._cfg = EasyDict(
        dict(
            intrinsic_reward_source='qstd',
            qstd_exploration=True,
            qstd_recompute_replay_rewards=True,
            qstd_target_snapshot=True,
            qstd_one_step_include_reward=True,
            qstd_one_step_use_value_calibration=False,
            qstd_target_inference_batch_size=32,
            qstd_reward_scale=2.0,
            qstd_reward_clip=4.0,
            discount_factor=0.5,
            device='cpu',
            model=dict(ensemble_size=2, analysis_sim_norm=False),
        )
    )
    policy._rnd_model = None
    policy._qstd_target_snapshot_model = _FakeOneStepQstdModel()
    policy._qstd_target_snapshot_value_offsets = np.zeros(2, dtype=np.float32)
    policy._qstd_target_reward_cache = {}
    policy._qstd_target_snapshot_generation = 1
    policy.value_inverse_scalar_transform_handle = lambda value: value
    policy.reward_inverse_scalar_transform_handle = lambda reward: reward

    observations = torch.tensor(
        [[[-0.5, 0.0], [-0.4, 0.01], [-0.3, 0.02]]], dtype=torch.float32
    )
    actions = torch.tensor([[0, 2, 1]], dtype=torch.long)
    valid_mask = torch.tensor([[1.0, 1.0, 0.0]])

    scaled, raw, first_stats = policy._one_step_action_qstd_rewards(
        observations, actions, valid_mask
    )
    second_scaled, second_raw, second_stats = policy._one_step_action_qstd_rewards(
        observations, actions, valid_mask
    )

    assert torch.allclose(raw, torch.tensor([[1.0, 3.0, 0.0]]))
    assert torch.allclose(scaled, torch.tensor([[2.0, 4.0, 0.0]]))
    assert torch.equal(second_raw, raw)
    assert torch.equal(second_scaled, scaled)
    assert first_stats['cache_hit_fraction'] == 0.0
    assert second_stats['cache_hit_fraction'] == 1.0


@pytest.mark.unittest
def test_recomputed_qstd_rewards_train_reward_and_return_targets():
    policy = object.__new__(EnsembleMuZeroPolicy)
    policy._cfg = EasyDict(
        dict(
            intrinsic_reward_source='qstd',
            qstd_exploration=True,
            qstd_recompute_replay_rewards=True,
            qstd_target_snapshot=False,
            qstd_one_step_include_reward=True,
            qstd_one_step_use_value_calibration=False,
            qstd_target_inference_batch_size=32,
            qstd_reward_scale=2.0,
            qstd_reward_clip=4.0,
            qstd_intrinsic_target_mode='monte_carlo',
            rnd_intrinsic_discount_factor=0.5,
            rnd_intrinsic_td_steps=2,
            intrinsic_reward_target_clip=0.0,
            intrinsic_value_target_clip=0.0,
            num_unroll_steps=2,
            td_steps=2,
            discount_factor=0.5,
            device='cpu',
            model=dict(
                ensemble_size=2,
                model_type='mlp',
                observation_shape=2,
                analysis_sim_norm=False,
            ),
            store_selected_intrinsic_rewards=False,
            intrinsic_value_train_on_pure_exploration_only=False,
        )
    )
    policy._target_model = _FakeOneStepQstdModel()
    policy.value_inverse_scalar_transform_handle = lambda value: value
    policy.reward_inverse_scalar_transform_handle = lambda reward: reward
    observations = np.asarray(
        [[-0.5, 0.0, -0.4, 0.01, -0.3, 0.02, -0.2, 0.03]], dtype=np.float32
    )
    actions = np.asarray([[0, 2, 1]], dtype=np.int64)
    # Base mask has num_unroll_steps + 1 entries, followed by the reward mask.
    mask = np.asarray([[1.0, 1.0, 0.0, 1.0, 1.0, 0.0]], dtype=np.float32)
    current_batch = [
        observations,
        actions,
        mask,
        np.asarray(['sample-0']),
        np.asarray([1.0], dtype=np.float32),
        np.asarray([0.0], dtype=np.float32),
    ]

    log, reward_targets, value_targets = policy._prepare_qstd_targets([current_batch, []])

    assert torch.allclose(reward_targets, torch.tensor([[2.0, 4.0]]))
    assert torch.allclose(value_targets, torch.tensor([[4.0, 4.0, 0.0]]))
    assert log['intrinsic_qstd/recomputed_one_step_action'] == 1.0
    assert log['intrinsic_qstd/raw_reward_max'] == pytest.approx(3.0)


@pytest.mark.unittest
def test_qstd_recompute_buffer_keeps_actions_for_full_intrinsic_td_window():
    class _Game:

        def __init__(self):
            self.action_segment = np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64)
            self.action_space_size = 3

        def get_unroll_obs(self, position, num_unroll_steps=0, padding=False):
            return np.zeros((num_unroll_steps + 1, 2), dtype=np.float32)

    buffer = object.__new__(MuZeroGameBuffer)
    buffer._cfg = EasyDict(
        dict(
            num_unroll_steps=2,
            td_steps=2,
            rnd_intrinsic_td_steps=4,
            game_segment_length=10,
            rnd_exploration=False,
            intrinsic_reward_source='qstd',
            qstd_exploration=True,
            qstd_recompute_replay_rewards=True,
            store_selected_intrinsic_rewards=False,
            intrinsic_value_train_on_pure_exploration_only=False,
            model=dict(model_type='mlp'),
        )
    )
    buffer.get_num_of_transitions = lambda: 6
    buffer._prepare_reward_value_context = lambda *args: 'reward-value'
    buffer._prepare_policy_non_reanalyzed_context = lambda *args: 'policy'
    orig_data = (
        [_Game()],
        [0],
        ['sample-0'],
        [1.0],
        [0.0],
    )

    _, _, _, current_batch = buffer._make_batch_from_orig_data(orig_data, reanalyze_ratio=0.0)

    assert current_batch[1].shape == (1, 6)
    np.testing.assert_array_equal(current_batch[1][0], np.asarray([0, 1, 2, 0, 1, 2]))


@pytest.mark.unittest
def test_muzero_buffer_emits_pure_exploration_episode_flags():
    class _Game:

        def __init__(self, pure_exploration_episode):
            self.action_segment = np.asarray([0, 1, 2], dtype=np.int64)
            self.action_space_size = 3
            self.pure_exploration_episode = pure_exploration_episode

        def get_unroll_obs(self, position, num_unroll_steps=0, padding=False):
            return np.zeros((num_unroll_steps + 1, 2), dtype=np.float32)

    buffer = object.__new__(MuZeroGameBuffer)
    buffer._cfg = EasyDict(
        dict(
            num_unroll_steps=2,
            td_steps=2,
            game_segment_length=5,
            rnd_exploration=False,
            store_selected_intrinsic_rewards=False,
            intrinsic_value_train_on_pure_exploration_only=True,
            model=dict(model_type='mlp'),
        )
    )
    buffer.get_num_of_transitions = lambda: 6
    buffer._prepare_reward_value_context = lambda *args: 'reward-value'
    buffer._prepare_policy_non_reanalyzed_context = lambda *args: 'policy'
    orig_data = (
        [_Game(True), _Game(False)],
        [0, 0],
        ['sample-0', 'sample-1'],
        [1.0, 1.0],
        [0.0, 0.0],
    )

    _, _, _, current_batch = buffer._make_batch_from_orig_data(orig_data, reanalyze_ratio=0.0)

    assert len(current_batch) == 7
    np.testing.assert_array_equal(current_batch[3], np.asarray([1.0, 0.0]))


def _make_intrinsic_tree_merge_policy(beta: float) -> EnsembleMuZeroPolicy:
    policy = object.__new__(EnsembleMuZeroPolicy)
    policy._cfg = EasyDict(
        dict(
            num_simulations=25,
            intrinsic_reward_source='rnd',
            rnd_intrinsic_reward_weight=beta,
            intrinsic_tree_final_score_temperature=1.0,
            intrinsic_tree_counterfactual_betas=(0.0, 20.0),
            intrinsic_tree_prior_mode='uniform',
            intrinsic_tree_force_root_action_coverage=True,
            intrinsic_tree_pure_episode_store_intrinsic_policy=False,
            model=dict(analysis_sim_norm=False),
        )
    )
    policy._collect_mcts_temperature = 0.25
    return policy


def _intrinsic_tree_merge_inputs(intrinsic_values):
    main = [
        {'action': 0, 'visit_count': 6, 'total_score': 100.0},
        {'action': 1, 'visit_count': 10, 'total_score': -100.0},
        {'action': 2, 'visit_count': 9, 'total_score': 0.0},
    ]
    intrinsic = [
        {'action': action, 'visit_count': 8, 'intrinsic_value': value}
        for action, value in enumerate(intrinsic_values)
    ]
    return main, intrinsic


@pytest.mark.unittest
def test_separate_intrinsic_tree_equal_values_preserve_main_visit_policy():
    policy = _make_intrinsic_tree_merge_policy(beta=50.0)
    main, intrinsic = _intrinsic_tree_merge_inputs([3.0, 3.0, 3.0])

    diagnostics, final_probs, final_counts = policy._merge_separate_intrinsic_tree_diagnostics(
        [0, 1, 2], main, intrinsic, [6, 10, 9]
    )

    expected = policy._visit_count_action_distribution([6, 10, 9])
    assert np.array_equal(final_probs, expected)
    assert np.asarray(final_counts).sum() == pytest.approx(25.0)
    assert np.argmax(final_probs) == 1
    assert diagnostics[0]['main_tree_puct_total_score'] == 100.0
    assert diagnostics[1]['intrinsic_tree_base_best_action'] == 1
    assert diagnostics[1]['intrinsic_tree_changed_argmax'] == 0.0
    assert diagnostics[1]['intrinsic_tree_policy_total_variation'] == pytest.approx(0.0)


@pytest.mark.unittest
def test_separate_intrinsic_tree_zero_beta_preserves_main_visit_policy():
    policy = _make_intrinsic_tree_merge_policy(beta=0.0)
    main, intrinsic = _intrinsic_tree_merge_inputs([0.0, 10.0, 20.0])

    _, final_probs, _ = policy._merge_separate_intrinsic_tree_diagnostics(
        [0, 1, 2], main, intrinsic, [6, 10, 9]
    )

    assert np.array_equal(final_probs, policy._visit_count_action_distribution([6, 10, 9]))


@pytest.mark.unittest
def test_separate_intrinsic_tree_adds_centered_intrinsic_q_to_visit_logits():
    policy = _make_intrinsic_tree_merge_policy(beta=2.0)
    main, intrinsic = _intrinsic_tree_merge_inputs([0.0, 0.0, 1.0])

    diagnostics, final_probs, _ = policy._merge_separate_intrinsic_tree_diagnostics(
        [0, 1, 2], main, intrinsic, [6, 10, 9]
    )

    base_probs = policy._visit_count_action_distribution([6, 10, 9]).astype(np.float64)
    expected = base_probs * np.exp(2.0 * (np.asarray([0.0, 0.0, 1.0]) - 1.0 / 3.0))
    expected /= expected.sum()
    assert final_probs == pytest.approx(expected, abs=1e-7)
    assert np.argmax(final_probs) == 2
    assert diagnostics[2]['intrinsic_tree_centered_intrinsic_value'] == pytest.approx(2.0 / 3.0)
    assert diagnostics[2]['intrinsic_tree_final_intrinsic_logit'] == pytest.approx(4.0 / 3.0)
    assert diagnostics[2]['intrinsic_tree_changed_argmax'] == 1.0
    assert diagnostics[2]['intrinsic_tree_counterfactual_probability_beta_0'] == pytest.approx(
        base_probs[2]
    )


@pytest.mark.unittest
def test_intrinsic_tree_pure_episode_uses_intrinsic_visits_but_keeps_main_policy_target():
    policy = _make_intrinsic_tree_merge_policy(beta=20.0)
    main, intrinsic = _intrinsic_tree_merge_inputs([1.0, 2.0, 3.0])

    diagnostics, final_probs, final_counts = policy._intrinsic_tree_episode_policy_diagnostics(
        [0, 1, 2],
        main,
        intrinsic,
        [6, 10, 9],
        [20, 3, 2],
        pure_intrinsic_episode=True,
    )

    expected = policy._visit_count_action_distribution([20, 3, 2])
    assert final_probs == pytest.approx(expected)
    assert final_counts == pytest.approx([20, 3, 2])
    assert diagnostics[0]['collection_policy_mode'] == 'intrinsic_tree_pure_exploration_episode'
    assert diagnostics[0]['pure_exploration_episode'] == 1.0
    assert diagnostics[0]['intrinsic_tree_action_policy_source'] == 'intrinsic_tree'
    assert diagnostics[0]['intrinsic_tree_policy_target_source'] == 'main_tree'
    assert diagnostics[0]['intrinsic_tree_root_forced_coverage_complete'] == 1.0


@pytest.mark.unittest
def test_intrinsic_tree_normal_episode_uses_unmodified_main_visit_policy():
    policy = _make_intrinsic_tree_merge_policy(beta=20.0)
    main, intrinsic = _intrinsic_tree_merge_inputs([100.0, 0.0, 0.0])

    diagnostics, final_probs, final_counts = policy._intrinsic_tree_episode_policy_diagnostics(
        [0, 1, 2],
        main,
        intrinsic,
        [6, 10, 9],
        [20, 3, 2],
        pure_intrinsic_episode=False,
    )

    expected = policy._visit_count_action_distribution([6, 10, 9])
    assert np.array_equal(final_probs, expected)
    assert final_counts == pytest.approx([6, 10, 9])
    assert diagnostics[0]['collection_policy_mode'] == 'normal'
    assert diagnostics[0]['pure_exploration_episode'] == 0.0
    assert diagnostics[0]['intrinsic_tree_action_policy_source'] == 'main_tree'
    assert diagnostics[0]['intrinsic_tree_policy_target_source'] == 'main_tree'
