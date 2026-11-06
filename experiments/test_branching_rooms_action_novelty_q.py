import numpy as np
import torch

from experiments.run_branching_rooms_action_novelty import ActionNoveltyPredictor
from experiments.run_branching_rooms_action_novelty_q import (
    PathConditionedEpisodicQ,
    branch_scores,
    build_nstep_sarsa_targets,
    discounted_mc_returns,
    novelty_rewards,
)
from experiments.run_branching_rooms_action_novelty_q_convergence import (
    ranking_metrics,
)
from experiments.run_branching_rooms_online_replay_q import (
    Agent57StyleEpisodicQ,
    dqn_batch_loss,
)


def test_novelty_rewards_reset_per_episode():
    observations = np.zeros((5, 3, 2, 2), dtype=np.uint8)
    observations[1, :, 0, 0] = 1
    observations[2] = observations[0]
    observations[3, :, 0, 1] = 2
    observations[4] = observations[1]
    assert novelty_rewards(observations).tolist() == [1.0, 0.0, 1.0, 0.0]


def test_discounted_mc_returns_are_normalized():
    returns = discounted_mc_returns(np.ones(3, dtype=np.float32), gamma=0.5)
    assert np.allclose(returns, [0.875, 0.75, 0.5])


def test_nstep_targets_include_bootstrap_and_terminal_truncation():
    rewards = torch.tensor([[1.0, 0.0, 1.0, 1.0]])
    next_q = torch.tensor([[0.2, 0.3, 0.4, 0.5]])
    indices = torch.tensor([[0, 2]])
    targets = build_nstep_sarsa_targets(
        rewards, next_q, indices, td_steps=2, gamma=0.5
    )
    assert torch.allclose(targets, torch.tensor([[[0.6, 0.75]]]))


def test_path_q_shapes_for_trajectory_and_hypothetical_suffixes():
    backbone = ActionNoveltyPredictor(3, 16, 32, True)
    model = PathConditionedEpisodicQ(backbone)
    observations = torch.randint(0, 256, (2, 8, 3, 14, 14), dtype=torch.uint8)
    actions = torch.randint(0, 3, (2, 7), dtype=torch.long)
    roots = torch.tensor([0, 2, 4])
    values = model(observations, actions, horizon=3, root_indices=roots)
    assert values.shape == (2, 3, 3)
    assert bool(torch.all((0 <= values) & (values <= 1)))

    with torch.no_grad():
        context = model.start_online_context(observations[:1, 0])
        suffix_values = model.score_action_prefixes(
            context, torch.tensor([[0, 1, 2], [2, 2, 0]])
        )
    assert suffix_values.shape == (2, 3)


def test_online_prefix_scores_match_batched_training_path():
    torch.manual_seed(7)
    backbone = ActionNoveltyPredictor(3, 16, 32, True)
    model = PathConditionedEpisodicQ(backbone).eval()
    observations = torch.randint(0, 256, (1, 8, 3, 14, 14), dtype=torch.uint8)
    actions = torch.randint(0, 3, (1, 7), dtype=torch.long)
    root = 3
    horizon = 3

    with torch.no_grad():
        batched = model(
            observations,
            actions,
            horizon=horizon,
            root_indices=torch.tensor([root]),
        )[0, 0]
        context = model.start_online_context(observations[:, 0])
        for index in range(root):
            context = model.advance_online_context(
                context,
                actions[:, index],
                observations[:, index + 1],
            )
        online = model.score_action_prefixes(
            context, actions[:, root : root + horizon]
        )[0]

    assert torch.allclose(online, batched, atol=1e-6, rtol=1e-6)


def test_all_root_actions_include_the_actual_one_step_value():
    torch.manual_seed(11)
    model = PathConditionedEpisodicQ(
        ActionNoveltyPredictor(3, 16, 32, True)
    ).eval()
    observations = torch.randint(0, 256, (2, 8, 3, 14, 14), dtype=torch.uint8)
    actions = torch.randint(0, 3, (2, 7), dtype=torch.long)
    roots = torch.tensor([1, 3, 5])
    with torch.no_grad():
        actual = model(observations, actions, 1, root_indices=roots).squeeze(-1)
        all_actions = model.root_action_values(observations, actions, roots)
        selected = torch.gather(
            all_actions,
            2,
            actions[:, roots].unsqueeze(-1),
        ).squeeze(-1)
    assert all_actions.shape == (2, 3, 3)
    assert torch.allclose(actual, selected, atol=1e-6, rtol=1e-6)


def test_agent57_online_context_matches_replay_unroll():
    torch.manual_seed(19)
    model = Agent57StyleEpisodicQ(3, 16, 32).eval()
    observations = torch.randint(0, 256, (1, 6, 3, 14, 14), dtype=torch.uint8)
    actions = torch.tensor([[2, 0, 2, 1, 2]])
    rewards = torch.tensor([[1.0, 1.0, 0.0, 1.0, 0.0]])
    with torch.no_grad():
        replay_values = model(observations, actions, rewards)
        context = model.start_online_context(observations[:, 0])
        online_values = [context.q_values]
        for index in range(actions.shape[1]):
            context = model.advance_online_context(
                context,
                actions[:, index],
                observations[:, index + 1],
                rewards[:, index],
            )
            online_values.append(context.q_values)
        online_values = torch.stack(online_values, dim=1)

    assert replay_values.shape == (1, 6, 3)
    assert torch.allclose(online_values, replay_values, atol=1e-6, rtol=1e-6)


def test_agent57_history_uses_previous_intrinsic_reward():
    torch.manual_seed(23)
    model = Agent57StyleEpisodicQ(3, 16, 32).eval()
    observation = torch.randint(0, 256, (1, 3, 14, 14), dtype=torch.uint8)
    next_observation = torch.randint(0, 256, (1, 3, 14, 14), dtype=torch.uint8)
    with torch.no_grad():
        context = model.start_online_context(observation)
        novel = model.advance_online_context(
            context, torch.tensor([2]), next_observation, torch.tensor([1.0])
        )
        revisit = model.advance_online_context(
            context, torch.tensor([2]), next_observation, torch.tensor([0.0])
        )
    assert not torch.allclose(novel.hidden, revisit.hidden)


def test_agent57_dqn_loss_is_finite_and_backpropagates():
    torch.manual_seed(29)
    model = Agent57StyleEpisodicQ(3, 16, 32)
    target_model = Agent57StyleEpisodicQ(3, 16, 32)
    target_model.load_state_dict(model.state_dict())
    observations = torch.randint(0, 256, (2, 8, 3, 14, 14), dtype=torch.uint8)
    actions = torch.randint(0, 3, (2, 7), dtype=torch.long)
    rewards = torch.randint(0, 2, (2, 7), dtype=torch.float32)
    loss, diagnostics = dqn_batch_loss(
        model,
        target_model,
        observations,
        actions,
        rewards,
        roots=torch.tensor([0, 2, 4]),
        horizon=3,
        td_steps=2,
        gamma=0.99,
    )
    loss.backward()
    assert bool(torch.isfinite(loss))
    assert diagnostics["mae"] >= 0
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_branch_score_uses_edges_before_bootstrapped_q():
    probabilities = np.zeros((1, 3, 3), dtype=np.float32)
    probabilities[..., 2] = np.asarray([[1.0, 0.0, 1.0]])
    q_values = np.asarray([[0.2, 0.3, 0.8]], dtype=np.float32)
    score = branch_scores(probabilities, q_values, gamma=0.5)
    # First two edge slots contribute 0.5 * 1 + 0.25 * 0; the depth-3 Q
    # already includes its own immediate reward and contributes 0.25 * 0.8.
    assert np.allclose(score, [0.7])


def test_ranking_metrics_only_counts_avoidable_revisits():
    values = np.asarray(
        [
            [0.9, 0.2, 0.1],
            [0.1, 0.8, 0.2],
            [0.3, 0.2, 0.1],
        ],
        dtype=np.float32,
    )
    labels = np.asarray(
        [
            [False, True, False],
            [False, True, False],
            [False, False, False],
        ]
    )
    metrics = ranking_metrics(values, labels, "q_rgb")
    assert metrics["q_rgb_opportunity_count"] == 2
    assert metrics["q_rgb_correct_count"] == 1
    assert metrics["q_rgb_avoidable_revisit_count"] == 1
    assert metrics["q_rgb_correct_rate"] == 0.5
    assert np.isclose(metrics["q_rgb_novel_margin_mean"], -0.05)
