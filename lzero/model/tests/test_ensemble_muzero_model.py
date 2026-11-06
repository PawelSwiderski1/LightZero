import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from lzero.model.common import MZNetworkOutput
from lzero.model.ensemble_muzero_model import EnsembleMuZeroModelMLP
from lzero.policy.ensemble_muzero import EnsembleMuZeroPolicy


def _flatten_parameters(model):
    return torch.cat([param.detach().flatten() for param in model.parameters()])


def _last_linear(module):
    return [child for child in module.modules() if isinstance(child, nn.Linear)][-1]


@pytest.mark.unittest
def test_encoder_random_prior_is_member_specific_scaled_and_frozen():
    torch.manual_seed(0)
    prior_scale = 0.75
    model = EnsembleMuZeroModelMLP(
        ensemble_size=3,
        observation_shape=2,
        action_space_size=3,
        latent_state_dim=16,
        shared_representation=True,
        shared_prediction=True,
        encoder_random_prior_scale=prior_scale,
        encoder_random_prior_hidden_dim=12,
        encoder_random_prior_input_mean=(-0.3, 0.0),
        encoder_random_prior_input_scale=(0.9, 0.07),
        encoder_random_prior_output_norm=1.0,
        representation_last_linear_layer_init_zero=False,
        value_last_linear_layer_init_zero=False,
        state_norm=False,
        norm_type='LN',
    )
    observations = torch.tensor([[-0.5, 0.0], [-0.2, 0.03], [0.1, -0.04]])
    learned_latent = model.members[0]._representation(observations).detach()
    member_outputs = [model.initial_inference(observations, member_id=k) for k in range(3)]
    prior_contributions = torch.stack(
        [output.latent_state.detach() - learned_latent for output in member_outputs], dim=0
    )

    contribution_norms = prior_contributions.norm(p=2, dim=2)
    assert torch.allclose(
        contribution_norms,
        torch.full_like(contribution_norms, prior_scale),
        atol=1e-5,
    )
    assert not torch.allclose(prior_contributions[0], prior_contributions[1])
    assert not torch.allclose(member_outputs[0].value, member_outputs[1].value)
    assert all(
        not parameter.requires_grad
        for prior in model.encoder_prior_networks
        for parameter in prior.parameters()
    )

    model.zero_grad(set_to_none=True)
    sum(output.latent_state.sum() for output in member_outputs).backward()
    assert any(parameter.grad is not None for parameter in model.members[0].representation_network.parameters())
    assert all(
        parameter.grad is None
        for prior in model.encoder_prior_networks
        for parameter in prior.parameters()
    )


@pytest.mark.unittest
def test_episodic_inverse_encoder_and_member_bridges_share_one_embedding_space():
    torch.manual_seed(5)
    model = EnsembleMuZeroModelMLP(
        ensemble_size=2,
        observation_shape=4,
        action_space_size=3,
        latent_state_dim=16,
        episodic_inverse_embedding_dim=8,
        episodic_inverse_hidden_dim=12,
        representation_last_linear_layer_init_zero=False,
        state_norm=False,
        norm_type='LN',
    )
    observations = torch.randn(6, 4)
    embeddings = model.episodic_inverse_embedding(observations)
    logits = model.episodic_inverse_action_logits(
        embeddings[:3], embeddings[3:]
    )
    latent = model.initial_inference(
        observations[:4], member_id=[0, 1, 0, 1]
    ).latent_state
    bridged = model.episodic_inverse_bridge(
        latent, member_id=[0, 1, 0, 1]
    )

    assert embeddings.shape == (6, 8)
    assert logits.shape == (3, 3)
    assert bridged.shape == (4, 8)
    assert torch.allclose(embeddings.norm(dim=-1), torch.ones(6), atol=1e-5)
    assert torch.allclose(bridged.norm(dim=-1), torch.ones(4), atol=1e-5)
    assert model.episodic_inverse_bridges[0] is not model.episodic_inverse_bridges[1]


@pytest.mark.unittest
def test_encoder_random_prior_batched_member_routing_and_deepcopy_are_exact():
    torch.manual_seed(7)
    model = EnsembleMuZeroModelMLP(
        ensemble_size=2,
        observation_shape=2,
        action_space_size=3,
        latent_state_dim=16,
        encoder_random_prior_scale=1.25,
        encoder_random_prior_input_mean=(-0.3, 0.0),
        encoder_random_prior_input_scale=(0.9, 0.07),
        representation_last_linear_layer_init_zero=False,
        state_norm=False,
        norm_type='LN',
    )
    observations = torch.tensor([[-0.6, 0.01], [-0.1, -0.02], [-0.4, 0.04], [0.0, -0.03]])
    member_ids = [1, 0, 1, 0]
    batched = model.initial_inference(observations, member_id=member_ids)
    individual = [
        model.initial_inference(observations[index:index + 1], member_id=member_id)
        for index, member_id in enumerate(member_ids)
    ]

    assert torch.allclose(
        batched.latent_state,
        torch.cat([output.latent_state for output in individual], dim=0),
        atol=1e-7,
    )
    assert torch.allclose(
        batched.value,
        torch.cat([output.value for output in individual], dim=0),
        atol=1e-7,
    )

    copied_model = copy.deepcopy(model)
    copied = copied_model.initial_inference(observations, member_id=member_ids)
    assert torch.equal(batched.latent_state, copied.latent_state)
    assert torch.equal(batched.value, copied.value)


@pytest.mark.unittest
def test_encoder_prior_fixed_norm_composition_preserves_full_directional_reach():
    torch.manual_seed(9)
    model = EnsembleMuZeroModelMLP(
        ensemble_size=2,
        observation_shape=2,
        action_space_size=3,
        latent_state_dim=16,
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
        state_norm=False,
        norm_type='LN',
    )
    observations = torch.tensor([[-0.5, 0.0], [-0.2, 0.03], [0.1, -0.04]])

    raw_learned = model.members[0]._representation(observations)
    learned = model._normalize_encoder_latent(raw_learned)
    prior = model.encoder_prior_networks[0](observations) * model.encoder_random_prior_scale
    output = model.initial_inference(observations, member_id=0)

    assert torch.allclose(learned.norm(dim=1), torch.ones(3), atol=1e-5)
    assert torch.allclose(prior.norm(dim=1), torch.full((3,), 0.5), atol=1e-5)
    assert torch.allclose(output.latent_state.norm(dim=1), torch.ones(3), atol=1e-5)

    learned_opposite = torch.zeros(1, 16)
    learned_opposite[:, 0] = -1.0
    prior_direction = torch.zeros(1, 16)
    prior_direction[:, 0] = 0.5
    composed = model._normalize_composed_encoder_latent(learned_opposite + prior_direction)
    assert torch.allclose(composed, learned_opposite, atol=1e-6)

    recurrent = model.recurrent_inference(
        output.latent_state,
        torch.tensor([[0], [1], [2]], dtype=torch.long),
        member_id=0,
    )
    assert torch.allclose(recurrent.latent_state.norm(dim=1), torch.ones(3), atol=1e-5)


@pytest.mark.unittest
def test_encoder_prior_fixed_norm_composition_rejects_equal_component_norms():
    with pytest.raises(ValueError, match='must strictly exceed'):
        EnsembleMuZeroModelMLP(
            ensemble_size=2,
            observation_shape=2,
            action_space_size=3,
            latent_state_dim=16,
            encoder_random_prior_scale=1.0,
            encoder_random_prior_output_norm=1.0,
            encoder_prior_composed_latent_normalization='l2',
            encoder_prior_composed_latent_norm_scale=1.0,
            dynamics_encoder_latent_normalization='l2',
            dynamics_encoder_latent_norm_scale=1.0,
            dynamics_trainable_latent_normalization='l2',
            dynamics_trainable_latent_norm_scale=1.0,
        )


@pytest.mark.unittest
def test_local_grid_encoder_prior_is_continuous_locally_and_decorrelates_at_distance():
    torch.manual_seed(11)
    model = EnsembleMuZeroModelMLP(
        ensemble_size=2,
        observation_shape=2,
        action_space_size=3,
        latent_state_dim=64,
        encoder_random_prior_scale=1.0,
        encoder_random_prior_mode='local_grid',
        encoder_random_prior_cell_size=0.25,
        encoder_random_prior_hash_size=4096,
        encoder_random_prior_input_mean=(-0.3, 0.0),
        encoder_random_prior_input_scale=(0.9, 0.07),
        encoder_random_prior_output_norm=1.0,
        representation_last_linear_layer_init_zero=False,
        state_norm=False,
        norm_type='LN',
    )
    prior = model.encoder_prior_networks[0]
    normalized_axis = torch.linspace(-2.0, 2.0, 32)
    normalized_x, normalized_y = torch.meshgrid(normalized_axis, normalized_axis, indexing='ij')
    normalized = torch.stack([normalized_x.flatten(), normalized_y.flatten()], dim=1)
    mean = torch.tensor([-0.3, 0.0])
    scale = torch.tensor([0.9, 0.07])

    base = prior(mean + normalized * scale)
    near = prior(mean + (normalized + torch.tensor([0.025, 0.0])) * scale)
    distant = prior(mean + (normalized + torch.tensor([0.75, 0.0])) * scale)
    near_cosine = F.cosine_similarity(base, near, dim=1).mean()
    distant_cosine = F.cosine_similarity(base, distant, dim=1).mean()

    assert torch.allclose(base.norm(p=2, dim=1), torch.ones(base.shape[0]), atol=1e-5)
    assert near_cosine > 0.98
    assert distant_cosine.abs() < 0.15
    assert all(not parameter.requires_grad for parameter in prior.parameters())

    boundary = mean.unsqueeze(0) + torch.tensor([[0.25, 0.0]]) * scale
    left = prior(boundary - torch.tensor([[1e-6, 0.0]]))
    right = prior(boundary + torch.tensor([[1e-6, 0.0]]))
    assert torch.linalg.vector_norm(left - right) < 1e-3


@pytest.mark.unittest
def test_encoder_random_prior_disabled_preserves_standard_initial_inference_path():
    model = EnsembleMuZeroModelMLP(
        ensemble_size=2,
        observation_shape=2,
        action_space_size=3,
        latent_state_dim=16,
        encoder_random_prior_scale=0.0,
        state_norm=False,
        norm_type='LN',
    )
    assert model.encoder_prior_networks is None


@pytest.mark.unittest
def test_value_head_can_be_randomly_initialized_without_changing_policy_or_reward_initialization():
    torch.manual_seed(0)
    model = EnsembleMuZeroModelMLP(
        ensemble_size=3,
        observation_shape=2,
        action_space_size=3,
        latent_state_dim=16,
        last_linear_layer_init_zero=True,
        value_last_linear_layer_init_zero=False,
        representation_last_linear_layer_init_zero=False,
        norm_type='LN',
    )

    value_layers = [_last_linear(member.prediction_network.fc_value_head) for member in model.members]
    policy_layers = [_last_linear(member.prediction_network.fc_policy_head) for member in model.members]
    reward_layers = [_last_linear(member.dynamics_network.fc_reward_head) for member in model.members]

    assert all(torch.count_nonzero(layer.weight).item() > 0 for layer in value_layers)
    assert not torch.allclose(value_layers[0].weight, value_layers[1].weight)
    assert all(torch.count_nonzero(layer.weight).item() == 0 for layer in policy_layers)
    assert all(torch.count_nonzero(layer.bias).item() == 0 for layer in policy_layers)
    assert all(torch.count_nonzero(layer.weight).item() == 0 for layer in reward_layers)
    assert all(torch.count_nonzero(layer.bias).item() == 0 for layer in reward_layers)

    model.eval()
    observations = torch.randn(8, 2)
    member_values = [model.initial_inference(observations, member_id=k).value for k in range(3)]
    assert not torch.allclose(member_values[0], member_values[1])


@pytest.mark.unittest
def test_ensemble_muzero_model_mlp_layer_norm_has_mode_invariant_unrolls():
    torch.manual_seed(0)
    model = EnsembleMuZeroModelMLP(
        ensemble_size=2,
        observation_shape=4,
        action_space_size=2,
        latent_state_dim=16,
        self_supervised_learning_loss=True,
        state_norm=True,
        norm_type='LN',
    )
    assert not any(isinstance(module, nn.modules.batchnorm._BatchNorm) for module in model.modules())
    assert any(isinstance(module, nn.LayerNorm) for module in model.modules())

    obs = torch.randn(8, 4)
    action = torch.randint(0, 2, (8, 1))
    model.train()
    train_root = model.initial_inference(obs, member_id=0)
    train_recurrent = model.recurrent_inference(train_root.latent_state, action, member_id=0)
    train_projection = model.project(train_recurrent.latent_state, with_grad=True, member_id=0)

    model.eval()
    eval_root = model.initial_inference(obs, member_id=0)
    eval_recurrent = model.recurrent_inference(eval_root.latent_state, action, member_id=0)
    eval_projection = model.project(eval_recurrent.latent_state, with_grad=True, member_id=0)

    assert torch.allclose(train_root.latent_state, eval_root.latent_state)
    assert torch.allclose(train_recurrent.latent_state, eval_recurrent.latent_state)
    assert torch.allclose(train_projection, eval_projection)
    for latent in [eval_root.latent_state, eval_recurrent.latent_state]:
        assert torch.all(latent >= 0.0)
        assert torch.all(latent <= 1.0)
        assert torch.allclose(latent.amin(dim=-1), torch.zeros(latent.shape[0]), atol=1e-6)
        assert torch.allclose(latent.amax(dim=-1), torch.ones(latent.shape[0]), atol=1e-6)


@pytest.mark.unittest
def test_episodic_history_model_advances_member_specific_hidden_state():
    torch.manual_seed(0)
    model = EnsembleMuZeroModelMLP(
        ensemble_size=2,
        observation_shape=2,
        action_space_size=3,
        latent_state_dim=16,
        episodic_history_dim=8,
        episodic_history_hidden_dim=12,
        representation_last_linear_layer_init_zero=False,
        state_norm=False,
        norm_type='LN',
    )
    observations = torch.randn(4, 2)
    current_latent = model.initial_inference(observations, member_id=0).latent_state
    actions = torch.tensor([0, 1, 2, 0])
    next_latent = model.recurrent_inference(
        current_latent, actions.unsqueeze(-1), member_id=0
    ).latent_state
    history = model.initial_episodic_history(4, member_id=0, device=current_latent.device)
    realized_rewards = torch.tensor([0.0, 0.5, 1.0, 2.0])
    next_history, reward = model.episodic_history_step(
        current_latent,
        actions,
        next_latent,
        realized_rewards,
        history,
        member_id=0,
    )

    assert next_history.shape == torch.Size([4, 8])
    assert reward.shape == torch.Size([4, 1])
    assert torch.count_nonzero(next_history).item() > 0
    assert not torch.allclose(next_history[0], next_history[1])
    assert len(model.episodic_history_parameters()) > 0


@pytest.mark.unittest
def test_episodic_history_sequence_matches_repeated_single_steps():
    torch.manual_seed(0)
    model = EnsembleMuZeroModelMLP(
        ensemble_size=1,
        observation_shape=2,
        action_space_size=3,
        latent_state_dim=16,
        episodic_history_dim=8,
        episodic_history_hidden_dim=12,
        representation_last_linear_layer_init_zero=False,
        state_norm=False,
        norm_type='LN',
    )
    current_latents = torch.randn(3, 4, 16)
    next_latents = torch.randn(3, 4, 16)
    actions = torch.randint(0, 3, (3, 4))
    realized_rewards = torch.rand(3, 4)
    initial_history = model.initial_episodic_history(3)

    sequence_histories, sequence_rewards = model.episodic_history_sequence(
        current_latents,
        actions,
        next_latents,
        next_latents,
        realized_rewards,
        initial_history,
    )
    history = initial_history
    step_histories = []
    step_rewards = []
    for step in range(4):
        history, reward = model.episodic_history_step(
            current_latents[:, step],
            actions[:, step],
            next_latents[:, step],
            realized_rewards[:, step],
            history,
        )
        step_histories.append(history)
        step_rewards.append(reward)

    assert torch.allclose(sequence_histories, torch.stack(step_histories, dim=1), atol=1e-6)
    assert torch.allclose(sequence_rewards, torch.stack(step_rewards, dim=1), atol=1e-6)


@pytest.mark.unittest
def test_episodic_history_reward_is_predicted_before_real_reward_updates_history():
    torch.manual_seed(0)
    model = EnsembleMuZeroModelMLP(
        ensemble_size=1,
        observation_shape=2,
        action_space_size=3,
        latent_state_dim=16,
        episodic_history_dim=8,
        representation_last_linear_layer_init_zero=False,
        state_norm=False,
        norm_type='LN',
    )
    current_latent = torch.randn(2, 16)
    next_latent = torch.randn(2, 16)
    actions = torch.tensor([0, 2])
    history = model.initial_episodic_history(2)

    history_low, prediction_low = model.episodic_history_step(
        current_latent,
        actions,
        next_latent,
        torch.zeros(2),
        history,
    )
    history_high, prediction_high = model.episodic_history_step(
        current_latent,
        actions,
        next_latent,
        torch.full((2,), 10.0),
        history,
    )

    assert torch.allclose(prediction_low, prediction_high, atol=1e-7)
    assert not torch.allclose(history_low, history_high)


@pytest.mark.unittest
def test_episodic_history_reward_feedback_uses_log1p_without_changing_targets():
    model = EnsembleMuZeroModelMLP(
        ensemble_size=1,
        observation_shape=2,
        action_space_size=3,
        latent_state_dim=16,
        episodic_history_dim=8,
        state_norm=False,
        norm_type='LN',
    )
    history_model = model.episodic_history_models[0]
    reference = torch.zeros(3, 16)
    rewards = torch.tensor([0.0, 3.0, 99.0])

    feedback = history_model._reward_feedback_input(rewards, reference).squeeze(-1)

    assert torch.allclose(feedback, torch.log1p(rewards))


@pytest.mark.unittest
def test_ensemble_muzero_model_mlp_supports_layer_norm_planning_with_batch_norm_ssl():
    torch.manual_seed(0)
    model = EnsembleMuZeroModelMLP(
        ensemble_size=2,
        observation_shape=4,
        action_space_size=2,
        latent_state_dim=16,
        self_supervised_learning_loss=True,
        state_norm=True,
        norm_type='LN',
        ssl_norm_type='BN',
    )

    for member in model.members:
        planning_modules = [
            member.representation_network,
            member.dynamics_network,
            member.prediction_network,
        ]
        assert not any(
            isinstance(module, nn.modules.batchnorm._BatchNorm)
            for planning_module in planning_modules
            for module in planning_module.modules()
        )
        assert any(isinstance(module, nn.LayerNorm) for module in member.representation_network.modules())
        assert any(isinstance(module, nn.modules.batchnorm._BatchNorm) for module in member.projection.modules())
        assert any(isinstance(module, nn.modules.batchnorm._BatchNorm) for module in member.prediction_head.modules())

    obs = torch.randn(8, 4)
    action = torch.randint(0, 2, (8, 1))
    model.train()
    train_root = model.initial_inference(obs, member_id=0)
    train_recurrent = model.recurrent_inference(train_root.latent_state, action, member_id=0)

    model.eval()
    eval_root = model.initial_inference(obs, member_id=0)
    eval_recurrent = model.recurrent_inference(eval_root.latent_state, action, member_id=0)

    assert torch.allclose(train_root.latent_state, eval_root.latent_state)
    assert torch.allclose(train_root.value, eval_root.value)
    assert torch.allclose(train_root.policy_logits, eval_root.policy_logits)
    assert torch.allclose(train_recurrent.latent_state, eval_recurrent.latent_state)
    assert torch.allclose(train_recurrent.reward, eval_recurrent.reward)
    assert torch.allclose(train_recurrent.value, eval_recurrent.value)
    assert torch.allclose(train_recurrent.policy_logits, eval_recurrent.policy_logits)


@pytest.mark.unittest
def test_nonzero_representation_initialization_gives_vicreg_a_noncollapsed_start():
    torch.manual_seed(0)
    model = EnsembleMuZeroModelMLP(
        ensemble_size=2,
        observation_shape=2,
        action_space_size=3,
        latent_state_dim=16,
        self_supervised_learning_loss=True,
        state_norm=False,
        representation_last_linear_layer_init_zero=False,
        norm_type='LN',
        ssl_norm_type='BN',
    )
    observations = torch.randn(32, 2)

    for member_id in range(2):
        latent = model.initial_inference(observations, member_id=member_id).latent_state
        assert latent.var(dim=0, unbiased=True).mean().item() > 1e-4


@pytest.mark.unittest
def test_ensemble_muzero_model_mlp_member_routing():
    batch_size = 4
    model = EnsembleMuZeroModelMLP(
        ensemble_size=3,
        observation_shape=4,
        action_space_size=2,
        latent_state_dim=16,
        self_supervised_learning_loss=True,
    )
    model.eval()

    obs = torch.randn(batch_size, 4)
    output_0 = model.initial_inference(obs, member_id=0)

    assert len(model.members) == 3
    assert model.value_support_size == model.members[0].value_support_size
    assert model.reward_support_size == model.members[0].reward_support_size
    assert output_0.value.shape == torch.Size([batch_size, model.value_support_size])
    assert output_0.policy_logits.shape == torch.Size([batch_size, 2])
    assert output_0.latent_state.shape == torch.Size([batch_size, 16])
    assert not torch.allclose(_flatten_parameters(model.members[0]), _flatten_parameters(model.members[1]))

    action = torch.randint(0, 2, (batch_size, 1))
    recurrent_output = model.recurrent_inference(output_0.latent_state, action, member_id=0)
    assert recurrent_output.value.shape == torch.Size([batch_size, model.value_support_size])
    assert recurrent_output.reward.shape == torch.Size([batch_size, model.reward_support_size])
    assert recurrent_output.policy_logits.shape == torch.Size([batch_size, 2])
    assert recurrent_output.latent_state.shape == torch.Size([batch_size, 16])

    model.set_active_member(1)
    default_output = model.initial_inference(obs)
    explicit_output = model.initial_inference(obs, member_id=1)
    assert torch.allclose(default_output.latent_state, explicit_output.latent_state)
    assert torch.allclose(default_output.value, explicit_output.value)
    assert torch.allclose(default_output.policy_logits, explicit_output.policy_logits)

    with pytest.raises(ValueError):
        model.initial_inference(obs, member_id=3)


@pytest.mark.unittest
def test_ensemble_muzero_model_mlp_batched_member_ids_match_explicit_member_calls():
    batch_size = 5
    model = EnsembleMuZeroModelMLP(
        ensemble_size=3,
        observation_shape=4,
        action_space_size=2,
        latent_state_dim=16,
        self_supervised_learning_loss=True,
    )
    model.eval()

    obs = torch.randn(batch_size, 4)
    member_ids = [0, 1, 2, 1, 0]
    batched_output = model.initial_inference(obs, member_id=member_ids)

    explicit_outputs = [
        model.initial_inference(obs[i:i + 1], member_id=member_id)
        for i, member_id in enumerate(member_ids)
    ]
    assert torch.allclose(batched_output.value, torch.cat([output.value for output in explicit_outputs], dim=0))
    assert torch.allclose(
        batched_output.policy_logits,
        torch.cat([output.policy_logits for output in explicit_outputs], dim=0)
    )
    assert torch.allclose(
        batched_output.latent_state,
        torch.cat([output.latent_state for output in explicit_outputs], dim=0)
    )

    action = torch.randint(0, 2, (batch_size, 1))
    batched_recurrent_output = model.recurrent_inference(batched_output.latent_state, action, member_id=member_ids)
    explicit_recurrent_outputs = [
        model.recurrent_inference(
            batched_output.latent_state[i:i + 1],
            action[i:i + 1],
            member_id=member_id
        )
        for i, member_id in enumerate(member_ids)
    ]
    assert torch.allclose(
        batched_recurrent_output.value,
        torch.cat([output.value for output in explicit_recurrent_outputs], dim=0)
    )
    assert torch.allclose(
        batched_recurrent_output.reward,
        torch.cat([output.reward for output in explicit_recurrent_outputs], dim=0)
    )
    assert torch.allclose(
        batched_recurrent_output.latent_state,
        torch.cat([output.latent_state for output in explicit_recurrent_outputs], dim=0)
    )

    projected = model.project(batched_output.latent_state, with_grad=True, member_id=member_ids)
    explicit_projected = [
        model.project(
            batched_output.latent_state[i:i + 1],
            with_grad=True,
            member_id=member_id
        )
        for i, member_id in enumerate(member_ids)
    ]
    assert torch.allclose(projected, torch.cat(explicit_projected, dim=0))

    intrinsic_rewards = model.intrinsic_reward_from_latent(batched_output.latent_state, member_id=member_ids)
    explicit_intrinsic_rewards = [
        model.intrinsic_reward_from_latent(
            batched_output.latent_state[i:i + 1],
            member_id=member_id
        )
        for i, member_id in enumerate(member_ids)
    ]
    assert intrinsic_rewards.shape == torch.Size([batch_size, 1])
    assert torch.allclose(intrinsic_rewards, torch.cat(explicit_intrinsic_rewards, dim=0))

    intrinsic_values = model.intrinsic_value_from_latent(batched_output.latent_state, member_id=member_ids)
    explicit_intrinsic_values = [
        model.intrinsic_value_from_latent(
            batched_output.latent_state[i:i + 1],
            member_id=member_id
        )
        for i, member_id in enumerate(member_ids)
    ]
    assert intrinsic_values.shape == torch.Size([batch_size, 1])
    assert torch.allclose(intrinsic_values, torch.cat(explicit_intrinsic_values, dim=0))


@pytest.mark.unittest
def test_ensemble_muzero_model_mlp_random_value_prior_adds_member_disagreement():
    torch.manual_seed(0)
    model = EnsembleMuZeroModelMLP(
        ensemble_size=2,
        random_prior_scale=1.0,
        observation_shape=4,
        action_space_size=2,
        latent_state_dim=16,
        self_supervised_learning_loss=True,
    )
    model.eval()
    model.members[1].load_state_dict(model.members[0].state_dict())

    obs = torch.randn(3, 4)
    output_0 = model.initial_inference(obs, member_id=0)
    output_1 = model.initial_inference(obs, member_id=1)

    assert torch.allclose(output_0.latent_state, output_1.latent_state)
    assert torch.allclose(output_0.policy_logits, output_1.policy_logits)
    assert not torch.allclose(output_0.value, output_1.value)

    member_ids = [0, 1, 0]
    batched_output = model.initial_inference(obs, member_id=member_ids)
    explicit_output = torch.cat(
        [model.initial_inference(obs[i:i + 1], member_id=member_id).value for i, member_id in enumerate(member_ids)],
        dim=0
    )
    assert torch.allclose(batched_output.value, explicit_output)
    assert all(not param.requires_grad for prior in model.value_prior_networks for param in prior.parameters())


@pytest.mark.unittest
def test_random_value_prior_input_layer_norm_is_invariant_to_latent_scale_and_shift():
    torch.manual_seed(2)
    model = EnsembleMuZeroModelMLP(
        ensemble_size=2,
        random_prior_scale=3.0,
        value_prior_latent_normalization='layer_norm',
        observation_shape=4,
        action_space_size=2,
        latent_state_dim=16,
    )
    latent = torch.randn(7, 16)
    prior = model.value_prior_networks[0]

    base_output = prior(latent)
    affine_output = prior(3.5 * latent + 8.0)

    assert torch.allclose(base_output, affine_output, atol=1e-5, rtol=1e-5)


@pytest.mark.unittest
@pytest.mark.parametrize('distribution_sigma', [None, 1.0, 12.0])
def test_random_value_prior_is_additive_in_scalar_value_independent_of_categorical_sharpness(
        distribution_sigma,
):
    torch.manual_seed(4)
    prior_scale = 5.0
    prior_fraction = 0.4
    model = EnsembleMuZeroModelMLP(
        ensemble_size=2,
        random_prior_scale=prior_scale,
        observation_shape=4,
        action_space_size=2,
        latent_state_dim=16,
    )
    prior = model.value_prior_networks[0]
    with torch.no_grad():
        for parameter in prior.parameters():
            parameter.zero_()
        prior.net[2].bias.fill_(torch.atanh(torch.tensor(prior_fraction)))

    latent = torch.randn(3, 16)
    support = model._value_prior_support.view(1, -1)
    if distribution_sigma is None:
        base_logits = torch.zeros(3, model.value_support_size)
    else:
        base_logits = (-0.5 * ((support + 11.0) / distribution_sigma).square()).expand(3, -1).clone()
    output = MZNetworkOutput(
        value=base_logits,
        reward=[0.0] * latent.shape[0],
        policy_logits=torch.zeros(latent.shape[0], model.action_space_size),
        latent_state=latent,
    )

    with_prior = model._apply_value_prior(output, member_id=0, device=latent.device)
    scalar_effect = model._value_logits_to_scalar(with_prior.value) - model._value_logits_to_scalar(base_logits)

    assert torch.allclose(
        scalar_effect,
        torch.full_like(scalar_effect, prior_scale * prior_fraction),
        atol=2e-2,
        rtol=2e-2,
    )


@pytest.mark.unittest
@pytest.mark.parametrize(
    'backprop_to_latent, expect_prior_gradient',
    [(False, False), (True, True)],
)
def test_random_value_prior_optionally_backpropagates_only_to_latent(
        backprop_to_latent,
        expect_prior_gradient,
):
    torch.manual_seed(3)
    model = EnsembleMuZeroModelMLP(
        ensemble_size=2,
        random_prior_scale=2.0,
        value_prior_backprop_to_latent=backprop_to_latent,
        observation_shape=4,
        action_space_size=2,
        latent_state_dim=16,
    )
    latent = torch.randn(5, 16, requires_grad=True)
    prior_value = model.value_prior_from_latent(latent, member_id=0)

    if expect_prior_gradient:
        prior_value.sum().backward()
        assert latent.grad is not None
        assert latent.grad.abs().sum().item() > 0.0
    else:
        assert not prior_value.requires_grad
        assert latent.grad is None
    assert all(
        parameter.grad is None
        for prior in model.value_prior_networks
        for parameter in prior.parameters()
    )


@pytest.mark.unittest
def test_random_value_prior_learning_mode_returns_unmodified_categorical_logits():
    torch.manual_seed(5)
    model = EnsembleMuZeroModelMLP(
        ensemble_size=2,
        random_prior_scale=5.0,
        observation_shape=4,
        action_space_size=2,
        latent_state_dim=16,
    )
    latent = torch.randn(3, 16)
    base_logits = torch.randn(3, model.value_support_size)
    output = MZNetworkOutput(
        value=base_logits,
        reward=[0.0] * latent.shape[0],
        policy_logits=torch.zeros(latent.shape[0], model.action_space_size),
        latent_state=latent,
    )

    model.set_value_prior_learning_mode(True)
    learning_output = model._apply_value_prior(output, member_id=0, device=latent.device)
    assert learning_output.value is base_logits

    model.set_value_prior_learning_mode(False)
    inference_output = model._apply_value_prior(output, member_id=0, device=latent.device)
    prior_value = model.value_prior_from_latent(latent, member_id=0)
    scalar_effect = (
        model._value_logits_to_scalar(inference_output.value)
        - model._value_logits_to_scalar(base_logits)
    )
    assert torch.allclose(scalar_effect, prior_value, atol=2e-2, rtol=2e-2)


@pytest.mark.unittest
def test_ensemble_muzero_model_mlp_shared_heads_with_random_dynamics_prior():
    torch.manual_seed(0)
    model = EnsembleMuZeroModelMLP(
        ensemble_size=2,
        dynamics_random_prior_scale=0.5,
        dynamics_random_prior_hidden_dim=32,
        dynamics_random_prior_action_mode='per_action',
        shared_representation=True,
        shared_prediction=True,
        shared_projection=True,
        observation_shape=4,
        action_space_size=2,
        latent_state_dim=16,
        self_supervised_learning_loss=True,
    )
    model.eval()
    model.members[1].dynamics_network.load_state_dict(model.members[0].dynamics_network.state_dict())

    assert model.members[0].representation_network is model.members[1].representation_network
    assert model.members[0].prediction_network is model.members[1].prediction_network
    assert model.members[0].projection is model.members[1].projection
    assert model.members[0].prediction_head is model.members[1].prediction_head
    assert model.dynamics_prior_networks[0].action_mode == 'per_action'
    assert len(model.dynamics_prior_networks[0].action_nets) == model.action_space_size
    assert all(not param.requires_grad for prior in model.dynamics_prior_networks for param in prior.parameters())

    obs = torch.randn(1, 4)
    root = model.initial_inference(obs, member_id=0)
    action_prior_delta = model.dynamics_prior_networks[0](
        root.latent_state.repeat(2, 1),
        torch.tensor([[0], [1]], dtype=torch.long),
    )
    assert not torch.allclose(action_prior_delta[0], action_prior_delta[1])

    latent = root.latent_state.repeat(2, 1)
    action = torch.ones(2, 1, dtype=torch.long)
    output = model.recurrent_inference(latent, action, member_id=[0, 1])

    assert output.latent_state.shape == torch.Size([2, 16])
    assert output.reward.shape == torch.Size([2, model.reward_support_size])
    assert output.value.shape == torch.Size([2, model.value_support_size])
    assert not torch.allclose(output.latent_state[0], output.latent_state[1])


@pytest.mark.unittest
def test_ensemble_muzero_model_mlp_normalizes_trainable_dynamics_before_prior():
    torch.manual_seed(0)
    target_norm = 1.25
    model = EnsembleMuZeroModelMLP(
        ensemble_size=2,
        dynamics_random_prior_scale=0.5,
        dynamics_random_prior_hidden_dim=32,
        dynamics_random_prior_action_mode='per_action',
        dynamics_trainable_latent_normalization='l2',
        dynamics_trainable_latent_norm_scale=target_norm,
        observation_shape=4,
        action_space_size=2,
        latent_state_dim=16,
        self_supervised_learning_loss=True,
    )
    model.eval()

    latent = torch.randn(3, 16)
    action = torch.ones(3, 1, dtype=torch.long)
    trainable_latent, _ = model.members[0]._dynamics(latent, action)
    normalized_latent = model._normalize_trainable_dynamics_latent(trainable_latent)
    assert torch.allclose(
        normalized_latent.norm(dim=-1),
        torch.full((3,), target_norm),
        atol=1e-5,
    )

    with torch.no_grad():
        prior_delta = model.dynamics_prior_networks[0](latent, action) * model.dynamics_random_prior_scale
    output = model.recurrent_inference(latent, action, member_id=0)
    assert torch.allclose(output.latent_state, normalized_latent + prior_delta, atol=1e-6)


@pytest.mark.unittest
def test_ensemble_muzero_model_mlp_caps_dynamics_prior_geometry():
    torch.manual_seed(0)
    model = EnsembleMuZeroModelMLP(
        ensemble_size=2,
        dynamics_random_prior_scale=10.0,
        dynamics_random_prior_hidden_dim=32,
        dynamics_random_prior_action_mode='per_action',
        dynamics_encoder_latent_normalization='l2',
        dynamics_encoder_latent_norm_scale=1.0,
        dynamics_trainable_latent_normalization='l2_cap',
        dynamics_trainable_latent_norm_scale=1.5,
        dynamics_prior_latent_norm_cap=0.25,
        observation_shape=4,
        action_space_size=2,
        latent_state_dim=16,
        self_supervised_learning_loss=True,
    )
    model.eval()

    obs = torch.randn(4, 4)
    root = model.initial_inference(obs, member_id=0)
    assert torch.allclose(root.latent_state.norm(dim=-1), torch.ones(4), atol=1e-5)

    action = torch.ones(4, 1, dtype=torch.long)
    raw_trainable, _ = model.members[0]._dynamics(root.latent_state, action)
    capped_trainable = model._normalize_trainable_dynamics_latent(raw_trainable)
    raw_prior = model.dynamics_prior_networks[0](root.latent_state, action) * model.dynamics_random_prior_scale
    capped_prior = model._bound_dynamics_prior_delta(raw_prior)

    assert torch.all(capped_trainable.norm(dim=-1) <= 1.5 + 1e-5)
    assert torch.all(capped_prior.norm(dim=-1) <= 0.25 + 1e-5)

    output = model.recurrent_inference(root.latent_state, action, member_id=0)
    expected = capped_trainable + capped_prior
    assert torch.allclose(output.latent_state, expected, atol=1e-6)
    assert torch.all(output.latent_state.norm(dim=-1) <= 1.75 + 1e-5)


@pytest.mark.unittest
def test_ensemble_muzero_model_mlp_order_preserved_groups_use_chunk_path():
    batch_size = 6
    model = EnsembleMuZeroModelMLP(
        ensemble_size=3,
        observation_shape=4,
        action_space_size=2,
        latent_state_dim=16,
        self_supervised_learning_loss=True,
    )
    model.eval()
    model.set_profile_timing(True)

    obs = torch.randn(batch_size, 4)
    member_ids = [0, 0, 1, 1, 2, 2]
    batched_output = model.initial_inference(obs, member_id=member_ids)
    explicit_outputs = [
        model.initial_inference(obs[i:i + 1], member_id=member_id)
        for i, member_id in enumerate(member_ids)
    ]

    assert torch.allclose(batched_output.value, torch.cat([output.value for output in explicit_outputs], dim=0))
    assert torch.allclose(
        batched_output.policy_logits,
        torch.cat([output.policy_logits for output in explicit_outputs], dim=0)
    )
    assert torch.allclose(
        batched_output.latent_state,
        torch.cat([output.latent_state for output in explicit_outputs], dim=0)
    )

    stats = model.get_profile_stats(reset=True)
    assert stats['initial_inference_order_preserved_group_count'] == 1
    assert stats.get('initial_inference_stitch_s', 0.0) == 0.0


@pytest.mark.unittest
def test_intrinsic_reward_log_training_space_decodes_to_raw_reward():
    model = EnsembleMuZeroModelMLP(
        ensemble_size=2,
        observation_shape=2,
        action_space_size=3,
        latent_state_dim=8,
        intrinsic_reward_target_transform='log',
        intrinsic_reward_log_reference=1e-2,
        intrinsic_reward_log_epsilon=1e-8,
        intrinsic_reward_log_max_raw=5.0,
        norm_type='LN',
    )
    latent = torch.randn(4, 8)
    member_ids = [0, 0, 1, 1]

    training_prediction = model.intrinsic_reward_prediction_from_latent(
        latent, member_id=member_ids
    )
    raw_prediction = model.intrinsic_reward_from_latent(
        latent, member_id=member_ids
    )
    assert torch.allclose(training_prediction, torch.zeros_like(training_prediction))
    assert torch.allclose(raw_prediction, torch.full_like(raw_prediction, 1e-2))

    raw_targets = torch.tensor([1e-4, 1e-2, 0.2, 5.0])
    transformed = model.transform_intrinsic_reward_target(raw_targets)
    decoded = model.inverse_transform_intrinsic_reward_prediction(transformed)
    assert torch.allclose(decoded, raw_targets, rtol=1e-5, atol=1e-8)


@pytest.mark.unittest
def test_shared_intrinsic_reward_head_is_shared_and_optimizer_parameters_are_unique():
    model = EnsembleMuZeroModelMLP(
        ensemble_size=3,
        observation_shape=2,
        action_space_size=3,
        latent_state_dim=8,
        shared_intrinsic_reward_head=True,
        norm_type='LN',
    )
    reference_head = model.members[0].intrinsic_reward_head
    assert all(
        member.intrinsic_reward_head is reference_head for member in model.members
    )
    assert all(
        member.intrinsic_value_head is not model.members[0].intrinsic_value_head
        for member in model.members[1:]
    )

    policy_state = SimpleNamespace(
        _learn_model=model,
        _cfg={'intrinsic_train_value_head': False},
    )
    parameters = EnsembleMuZeroPolicy._intrinsic_head_parameters(policy_state)
    assert len(parameters) == len({id(parameter) for parameter in parameters})
    assert {id(parameter) for parameter in parameters} == {
        id(parameter) for parameter in reference_head.parameters()
    }
