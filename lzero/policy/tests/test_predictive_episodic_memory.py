import pytest
import torch

from lzero.policy.predictive_episodic_memory import (
    PredictiveEpisodicMemory,
    ReconstructiveEpisodicMemory,
    balanced_memory_distance_loss,
)


def _model() -> PredictiveEpisodicMemory:
    return PredictiveEpisodicMemory(
        observation_channels=3,
        action_space_size=3,
        embedding_dim=12,
        hidden_dim=24,
        initial_distance_threshold=0.2,
        initial_distance_temperature=0.05,
    )


def _adaptive_model() -> PredictiveEpisodicMemory:
    return PredictiveEpisodicMemory(
        observation_channels=3,
        action_space_size=3,
        embedding_dim=12,
        hidden_dim=24,
        initial_distance_threshold=0.2,
        initial_distance_temperature=0.01,
        adaptive_distance_scale=True,
        adaptive_threshold_ratio=0.5,
        adaptive_temperature_ratio=0.125,
        adaptive_minimum_scale=0.02,
    )


def _reconstructive_model() -> ReconstructiveEpisodicMemory:
    return ReconstructiveEpisodicMemory(
        observation_shape=(3, 20, 20),
        action_space_size=3,
        embedding_dim=12,
        hidden_dim=24,
        decoder_channels=16,
        initial_distance_threshold=0.2,
        initial_distance_temperature=0.05,
    )


@pytest.mark.unittest
def test_context_keeps_explicit_real_embedding_memory():
    torch.manual_seed(0)
    model = _model().eval()
    first = torch.rand(1, 3, 20, 20)
    second = torch.rand(1, 3, 20, 20)

    with torch.no_grad():
        context = model.start_context(first)
        assert context.memory_embeddings.shape == (1, 12)
        context = model.advance_context(
            context,
            action=torch.tensor([2]),
            observation=second,
            episodic_reward=torch.tensor([1.0]),
            extrinsic_reward=torch.tensor([0.0]),
        )

    assert context.memory_embeddings.shape == (2, 12)
    torch.testing.assert_close(
        context.memory_embeddings.norm(dim=-1), torch.ones(2), atol=1e-5, rtol=1e-5
    )


@pytest.mark.unittest
def test_context_uses_online_embedding_and_target_memory():
    torch.manual_seed(3)
    model = _model().eval()
    observation = torch.rand(1, 3, 20, 20)
    with torch.no_grad():
        model.observation_encoder[-1].bias[0].add_(1.0)
        online_embedding = model.encode_observations(observation)
        target_embedding = model.encode_memory_observations(observation)
        context = model.start_context(observation)

    assert not torch.allclose(online_embedding, target_embedding)
    torch.testing.assert_close(context.current_embedding, online_embedding)
    torch.testing.assert_close(context.memory_embeddings, target_embedding)


@pytest.mark.unittest
def test_memory_distance_distinguishes_exact_revisit_from_new_direction():
    model = _model().eval()
    memory = torch.tensor([[1.0, 0.0, 0.0]])
    predictions = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32
    )

    logits, distances = model._novelty_logits_for_predictions(predictions, memory)

    assert distances.tolist() == pytest.approx([0.0, 1.0])
    assert torch.sigmoid(logits[0]).item() < 0.1
    assert torch.sigmoid(logits[1]).item() > 0.9


@pytest.mark.unittest
def test_adaptive_calibration_uses_nearest_distinct_memory_spacing():
    model = _adaptive_model().eval()
    memory = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.8, 0.6, 0.0],
        ],
        dtype=torch.float32,
    )

    scale, threshold, temperature = model._memory_distance_calibration(memory)

    assert scale.item() == pytest.approx(0.2)
    assert threshold.item() == pytest.approx(0.1)
    assert temperature.item() == pytest.approx(0.025)


@pytest.mark.unittest
def test_adaptive_calibration_uses_minimum_scale_until_memory_has_distinct_keys():
    model = _adaptive_model().eval()
    repeated_memory = torch.tensor(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32
    )

    scale, threshold, temperature = model._memory_distance_calibration(
        repeated_memory
    )

    assert scale.item() == pytest.approx(0.02)
    assert threshold.item() == pytest.approx(0.01)
    assert temperature.item() == pytest.approx(0.0025)

    revisit_probability = torch.sigmoid(-threshold / temperature).item()
    typical_distinct_probability = torch.sigmoid(
        (scale - threshold) / temperature
    ).item()
    assert revisit_probability == pytest.approx(0.017986, abs=1e-5)
    assert typical_distinct_probability == pytest.approx(0.982014, abs=1e-5)


@pytest.mark.unittest
def test_adaptive_calibration_separates_revisit_and_distant_prediction():
    model = _adaptive_model().eval()
    memory = torch.tensor(
        [[1.0, 0.0, 0.0], [0.8, 0.6, 0.0]], dtype=torch.float32
    )
    predictions = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32
    )

    logits, distances = model._novelty_logits_for_predictions(predictions, memory)

    assert distances.tolist() == pytest.approx([0.0, 1.0])
    assert torch.sigmoid(logits[0]).item() < 0.05
    assert torch.sigmoid(logits[1]).item() > 0.95


@pytest.mark.unittest
def test_balanced_memory_distance_loss_weights_novel_and_revisit_groups_equally():
    predicted = torch.tensor([0.08, 0.04, 0.08], requires_grad=True)
    targets = torch.tensor([0.0, 0.08, 0.08], requires_grad=True)

    loss = balanced_memory_distance_loss(predicted, targets, torch.tensor(0.08))
    revisit_loss = torch.nn.functional.smooth_l1_loss(
        predicted[:1] / 0.08,
        targets[:1].detach() / 0.08,
    )
    novel_loss = torch.nn.functional.smooth_l1_loss(
        predicted[1:] / 0.08,
        targets[1:].detach() / 0.08,
    )

    torch.testing.assert_close(loss, 0.5 * (revisit_loss + novel_loss))
    loss.backward()
    assert predicted.grad is not None
    assert targets.grad is None


@pytest.mark.unittest
def test_balanced_memory_distance_loss_handles_single_available_group():
    predicted = torch.tensor([0.02, 0.04], requires_grad=True)
    targets = torch.zeros(2)

    loss = balanced_memory_distance_loss(predicted, targets, torch.tensor(0.08))

    assert torch.isfinite(loss)
    loss.backward()
    assert torch.count_nonzero(predicted.grad).item() == 2


@pytest.mark.unittest
def test_training_outputs_supervise_future_codes_and_memory_lookup():
    torch.manual_seed(1)
    model = _model().train()
    observations = torch.rand(2, 7, 3, 20, 20)
    actions = torch.randint(0, 3, (2, 6))
    episodic_rewards = torch.randint(0, 2, (2, 6)).float()
    extrinsic_rewards = torch.zeros_like(episodic_rewards)
    roots = torch.tensor([[0, 2], [1, 3]])

    outputs = model.training_outputs(
        observations,
        actions,
        episodic_rewards,
        extrinsic_rewards,
        roots,
        horizon=3,
    )

    assert outputs['novelty_logits'].shape == (2, 2, 3)
    assert outputs['predicted_embeddings'].shape == (2, 2, 3, 12)
    assert outputs['target_embeddings'].shape == (2, 2, 3, 12)
    assert outputs['target_novelty_probabilities'].shape == (2, 2, 3)
    assert outputs['distance_scales'].shape == (2, 2)
    assert outputs['distance_thresholds'].shape == (2, 2)
    assert outputs['distance_temperatures'].shape == (2, 2)
    assert torch.all(outputs['target_novelty_probabilities'] >= 0.0)
    assert torch.all(outputs['target_novelty_probabilities'] <= 1.0)
    assert outputs['inverse_logits'].shape == (2, 6, 3)
    loss = (
        outputs['novelty_logits'].square().mean()
        + (1.0 - torch.nn.functional.cosine_similarity(
            outputs['predicted_embeddings'], outputs['target_embeddings'], dim=-1
        )).mean()
        + outputs['inverse_logits'].square().mean()
    )
    loss.backward()
    assert model.future_embedding_head[-1].weight.grad is not None
    assert any(
        parameter.grad is not None
        for parameter in model.observation_encoder.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in model.target_observation_encoder.parameters()
    )


@pytest.mark.unittest
def test_reconstructive_memory_decodes_real_and_imagined_exploration_states():
    torch.manual_seed(7)
    model = _reconstructive_model().train()
    observations = torch.rand(1, 7, 3, 20, 20)
    actions = torch.randint(0, 3, (1, 6))
    episodic_rewards = torch.randint(0, 2, (1, 6)).float()

    outputs = model.training_outputs(
        observations,
        actions,
        episodic_rewards,
        torch.zeros_like(episodic_rewards),
        torch.tensor([[0, 2]]),
        horizon=3,
    )

    assert outputs['reconstructed_observations'].shape == (1, 2, 4, 3, 20, 20)
    assert outputs['reconstruction_targets'].shape == (1, 2, 4, 3, 20, 20)
    assert outputs['predicted_future_observations'].shape == (
        1, 2, 3, 3, 20, 20
    )
    assert outputs['future_observation_targets'].shape == (
        1, 2, 3, 3, 20, 20
    )
    assert torch.all(outputs['reconstructed_observations'] >= 0.0)
    assert torch.all(outputs['reconstructed_observations'] <= 1.0)

    reconstruction_loss = torch.nn.functional.l1_loss(
        outputs['reconstructed_observations'],
        outputs['reconstruction_targets'],
    )
    rollout_loss = torch.nn.functional.l1_loss(
        outputs['predicted_future_observations'],
        outputs['future_observation_targets'],
    )
    (reconstruction_loss + rollout_loss).backward()

    assert model.decoder_output.weight.grad is not None
    assert model.future_embedding_head[-1].weight.grad is not None
    assert any(
        parameter.grad is not None
        for parameter in model.observation_encoder.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in model.target_observation_encoder.parameters()
    )


@pytest.mark.unittest
def test_reconstructive_predictor_only_update_freezes_exploration_encoder():
    torch.manual_seed(8)
    model = _reconstructive_model().train()
    observations = torch.rand(1, 7, 3, 20, 20)
    actions = torch.randint(0, 3, (1, 6))
    rewards = torch.zeros(1, 6)

    outputs = model.training_outputs(
        observations,
        actions,
        rewards,
        rewards,
        torch.tensor([[1, 2]]),
        horizon=3,
        freeze_online_encoder=True,
    )
    loss = torch.nn.functional.l1_loss(
        outputs['reconstructed_observations'],
        outputs['reconstruction_targets'],
    ) + torch.nn.functional.l1_loss(
        outputs['predicted_future_observations'],
        outputs['future_observation_targets'],
    )
    loss.backward()

    assert model.decoder_output.weight.grad is not None
    assert model.future_embedding_head[-1].weight.grad is not None
    assert all(
        parameter.grad is None
        for parameter in model.observation_encoder.parameters()
    )


@pytest.mark.unittest
def test_future_prediction_losses_train_online_encoder_against_frozen_target():
    torch.manual_seed(4)
    model = _model().train()
    observations = torch.rand(1, 7, 3, 20, 20)
    actions = torch.randint(0, 3, (1, 6))
    episodic_rewards = torch.randint(0, 2, (1, 6)).float()
    roots = torch.tensor([[0, 2]])

    outputs = model.training_outputs(
        observations,
        actions,
        episodic_rewards,
        torch.zeros_like(episodic_rewards),
        roots,
        horizon=3,
    )
    loss = (
        outputs['novelty_logits'].square().mean()
        + (1.0 - torch.nn.functional.cosine_similarity(
            outputs['predicted_embeddings'], outputs['target_embeddings'], dim=-1
        )).mean()
    )
    loss.backward()

    assert model.future_embedding_head[-1].weight.grad is not None
    assert model.history_cell.weight_ih.grad is not None
    assert any(
        parameter.grad is not None for parameter in model.observation_encoder.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in model.target_observation_encoder.parameters()
    )


@pytest.mark.unittest
def test_predictor_only_outputs_freeze_both_observation_encoders():
    torch.manual_seed(5)
    model = _model().train()
    observations = torch.rand(1, 7, 3, 20, 20)
    actions = torch.randint(0, 3, (1, 6))
    episodic_rewards = torch.randint(0, 2, (1, 6)).float()

    outputs = model.training_outputs(
        observations,
        actions,
        episodic_rewards,
        torch.zeros_like(episodic_rewards),
        torch.tensor([[0, 2]]),
        horizon=3,
        freeze_online_encoder=True,
    )
    loss = (
        torch.nn.functional.binary_cross_entropy_with_logits(
            outputs['novelty_logits'], outputs['target_novelty_probabilities']
        )
        + (1.0 - torch.nn.functional.cosine_similarity(
            outputs['predicted_embeddings'], outputs['target_embeddings'], dim=-1
        )).mean()
    )
    loss.backward()

    assert model.future_embedding_head[-1].weight.grad is not None
    assert model.history_cell.weight_ih.grad is not None
    assert all(
        parameter.grad is None for parameter in model.observation_encoder.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in model.target_observation_encoder.parameters()
    )


@pytest.mark.unittest
def test_variable_length_paths_use_each_roots_real_memory():
    torch.manual_seed(2)
    model = _model().eval()
    with torch.no_grad():
        contexts = [
            model.start_context(torch.rand(1, 3, 20, 20)),
            model.start_context(torch.rand(1, 3, 20, 20)),
        ]
        logits = model.score_path_endpoints(
            contexts,
            root_indices=[0, 1, 0],
            action_paths=[[1], [2, 0], [1, 2, 0]],
        )

    assert logits.shape == (3,)
    assert torch.isfinite(logits).all()
