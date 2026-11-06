import numpy as np
import torch

from experiments.run_branching_rooms_recurrent_sequence_novelty import (
    RecurrentSequenceNovelty,
    action_values_from_suffixes,
    enumerate_action_suffixes,
    sequence_prediction_loss,
)


def test_online_history_matches_replay_unroll():
    torch.manual_seed(101)
    model = RecurrentSequenceNovelty(3, 16, 32).eval()
    observations = torch.randint(0, 256, (1, 7, 3, 14, 14), dtype=torch.uint8)
    actions = torch.tensor([[2, 0, 1, 2, 2, 0]])
    intrinsic = torch.tensor([[1.0, 1.0, 0.0, 1.0, 0.0, 1.0]])
    extrinsic = torch.tensor([[0.0, 0.5, 0.0, -0.25, 0.0, 0.0]])

    with torch.no_grad():
        replay_states = model.history_states(
            observations, actions, intrinsic, extrinsic
        )
        context = model.start_online_context(observations[:, 0])
        online_states = [context.hidden]
        for index in range(actions.shape[1]):
            context = model.advance_online_context(
                context,
                actions[:, index],
                observations[:, index + 1],
                intrinsic[:, index],
                extrinsic[:, index],
            )
            online_states.append(context.hidden)
        online_states = torch.stack(online_states, dim=1)

    assert replay_states.shape == (1, 7, 32)
    assert torch.allclose(online_states, replay_states, atol=1e-6, rtol=1e-6)


def test_history_uses_previous_action_and_rewards_without_memory_bank():
    torch.manual_seed(103)
    model = RecurrentSequenceNovelty(3, 16, 32).eval()
    observation = torch.randint(0, 256, (1, 3, 14, 14), dtype=torch.uint8)
    next_observation = torch.randint(0, 256, (1, 3, 14, 14), dtype=torch.uint8)
    with torch.no_grad():
        context = model.start_online_context(observation)
        baseline = model.advance_online_context(
            context,
            torch.tensor([0]),
            next_observation,
            torch.tensor([0.0]),
            torch.tensor([0.0]),
        )
        changed_action = model.advance_online_context(
            context,
            torch.tensor([2]),
            next_observation,
            torch.tensor([0.0]),
            torch.tensor([0.0]),
        )
        changed_intrinsic = model.advance_online_context(
            context,
            torch.tensor([0]),
            next_observation,
            torch.tensor([1.0]),
            torch.tensor([0.0]),
        )
        changed_extrinsic = model.advance_online_context(
            context,
            torch.tensor([0]),
            next_observation,
            torch.tensor([0.0]),
            torch.tensor([1.0]),
        )

    assert set(context.__dataclass_fields__) == {"hidden", "cell"}
    assert not torch.allclose(baseline.hidden, changed_action.hidden)
    assert not torch.allclose(baseline.hidden, changed_intrinsic.hidden)
    assert not torch.allclose(baseline.hidden, changed_extrinsic.hidden)


def test_suffix_predictions_are_causal_by_depth():
    torch.manual_seed(107)
    model = RecurrentSequenceNovelty(3, 16, 32).eval()
    observation = torch.randint(0, 256, (1, 3, 14, 14), dtype=torch.uint8)
    suffixes = torch.tensor([[2, 0, 0], [2, 1, 2], [0, 1, 2]])
    with torch.no_grad():
        context = model.start_online_context(observation)
        logits = model.score_action_suffixes(context, suffixes)

    assert logits.shape == (3, 3)
    assert torch.allclose(logits[0, 0], logits[1, 0], atol=1e-7, rtol=1e-7)
    assert not torch.allclose(logits[0, 0], logits[2, 0])


def test_actual_suffix_forward_matches_online_scoring():
    torch.manual_seed(109)
    model = RecurrentSequenceNovelty(3, 16, 32).eval()
    observations = torch.randint(0, 256, (1, 8, 3, 14, 14), dtype=torch.uint8)
    actions = torch.tensor([[2, 0, 1, 2, 2, 0, 1]])
    rewards = torch.tensor([[1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]])
    root = 3
    horizon = 3
    with torch.no_grad():
        replay_logits = model.forward_actual_suffixes(
            observations,
            actions,
            rewards,
            horizon,
            torch.tensor([root]),
        )[0, 0]
        context = model.start_online_context(observations[:, 0])
        for index in range(root):
            context = model.advance_online_context(
                context,
                actions[:, index],
                observations[:, index + 1],
                rewards[:, index],
            )
        online_logits = model.score_action_suffixes(
            context, actions[:, root : root + horizon]
        )[0]

    assert torch.allclose(replay_logits, online_logits, atol=1e-6, rtol=1e-6)


def test_sequence_loss_is_finite_and_backpropagates():
    torch.manual_seed(113)
    model = RecurrentSequenceNovelty(3, 16, 32)
    observations = torch.randint(0, 256, (2, 8, 3, 14, 14), dtype=torch.uint8)
    actions = torch.randint(0, 3, (2, 7), dtype=torch.long)
    rewards = torch.randint(0, 2, (2, 7), dtype=torch.float32)
    loss, diagnostics = sequence_prediction_loss(
        model,
        observations,
        actions,
        rewards,
        torch.tensor([0, 2, 4]),
        horizon=3,
        gamma=0.99,
        sequence_loss_weight=1.0,
    )
    loss.backward()

    assert bool(torch.isfinite(loss))
    assert diagnostics["edge_bce"] > 0
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_suffix_action_values_use_best_sequence_per_first_action():
    suffixes = enumerate_action_suffixes(3, 2)
    probabilities = np.zeros((9, 2), dtype=np.float32)
    probabilities[(suffixes == np.asarray([1, 2])).all(axis=1)] = [0.9, 0.8]
    values = action_values_from_suffixes(
        probabilities, suffixes, gamma=0.5, top_k=1
    )
    assert np.allclose(values, [0.0, 1.3, 0.0])
