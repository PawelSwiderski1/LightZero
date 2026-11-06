import numpy as np
import pytest
import torch

from lzero.policy.recurrent_sequence_novelty import (
    RecentHistoryTransformerNovelty,
    RecurrentSequenceNovelty,
    SymbolicGridObservationEncoder,
    TileAlignedRGBObservationEncoder,
    episodic_continuation_factor_targets,
    recent_observation_novelty_targets,
)


@pytest.mark.unittest
@pytest.mark.parametrize("image_size", [28, 84])
def test_tile_aligned_rgb_encoder_accepts_minigrid_render_sizes(image_size):
    encoder = TileAlignedRGBObservationEncoder(
        observation_channels=3,
        embedding_dim=12,
    )

    encoded = encoder(torch.rand(2, 3, image_size, image_size))

    assert encoded.shape == (2, 12)


@pytest.mark.unittest
def test_symbolic_grid_encoder_embeds_native_minigrid_observations():
    encoder = SymbolicGridObservationEncoder(embedding_dim=12)
    observations = torch.zeros(2, 3, 7, 7)
    observations[:, 0] = torch.randint(0, 11, (2, 7, 7))
    observations[:, 1] = torch.randint(0, 6, (2, 7, 7))
    observations[:, 2] = torch.randint(0, 3, (2, 7, 7))

    encoded = encoder(observations)

    assert encoded.shape == (2, 12)


@pytest.mark.unittest
def test_recent_targets_forget_observations_outside_the_window():
    observations = np.arange(47, dtype=np.uint8).reshape(1, 47, 1, 1, 1)
    observations[0, 45] = observations[0, 4]
    observations[0, 46] = observations[0, 6]

    targets = recent_observation_novelty_targets(
        observations,
        root_indices=np.asarray([[44, 45]]),
        horizon=1,
        history_window=40,
    )

    assert targets.tolist() == [[[1.0], [0.0]]]


@pytest.mark.unittest
def test_recent_transformer_context_retains_only_the_configured_window():
    torch.manual_seed(4)
    model = RecentHistoryTransformerNovelty(
        observation_channels=3,
        action_space_size=3,
        embedding_dim=12,
        hidden_dim=24,
        history_window=4,
        max_horizon=3,
        transformer_layers=1,
        transformer_heads=4,
    ).eval()
    observations = torch.rand(1, 7, 3, 20, 20)
    actions = torch.tensor([[0, 1, 2, 0, 1, 2]])

    with torch.no_grad():
        context = model.start_context(observations[:, 0])
        for step in range(6):
            context = model.advance_context(
                context,
                actions[:, step],
                observations[:, step + 1],
                torch.ones(1),
                torch.zeros(1),
            )

    assert context.observation_embeddings.shape == (1, 4, 12)
    assert context.previous_actions.tolist() == [[2, 0, 1, 2]]


@pytest.mark.unittest
def test_recent_transformer_online_and_replay_scoring_match():
    torch.manual_seed(5)
    model = RecentHistoryTransformerNovelty(
        observation_channels=3,
        action_space_size=3,
        embedding_dim=12,
        hidden_dim=24,
        history_window=4,
        max_horizon=3,
        transformer_layers=1,
        transformer_heads=4,
    ).eval()
    observations = torch.rand(1, 8, 3, 20, 20)
    actions = torch.tensor([[0, 1, 2, 0, 1, 2, 1]])
    rewards = torch.zeros(1, 7)

    with torch.no_grad():
        context = model.start_context(observations[:, 0])
        for step in range(5):
            context = model.advance_context(
                context,
                actions[:, step],
                observations[:, step + 1],
                rewards[:, step],
                rewards[:, step],
            )
        online = model.score_path_endpoints(
            [context],
            root_indices=[0],
            action_paths=[actions[0, 5:7].tolist()],
        )
        replay = model.score_actual_suffixes(
            observations,
            actions,
            rewards,
            rewards,
            root_indices=torch.tensor([[5]]),
            horizon=2,
        )

    torch.testing.assert_close(online[0], replay[0, 0, 1])


@pytest.mark.unittest
def test_recent_transformer_prefix_tables_match_direct_endpoint_scoring():
    torch.manual_seed(6)
    model = RecentHistoryTransformerNovelty(
        observation_channels=3,
        action_space_size=3,
        embedding_dim=12,
        hidden_dim=24,
        history_window=4,
        max_horizon=3,
        transformer_layers=1,
        transformer_heads=4,
    ).eval()
    with torch.no_grad():
        contexts = [
            model.start_context(torch.rand(1, 3, 20, 20)),
            model.start_context(torch.rand(1, 3, 20, 20)),
        ]
        tables = model.all_action_prefix_probabilities(contexts)
        paths = [(0,), (2, 1), (1, 0, 2)]
        direct = torch.sigmoid(
            model.score_path_endpoints(
                contexts,
                root_indices=[0, 1, 0],
                action_paths=paths,
            )
        ).cpu()

    assert [len(table) for table in tables] == [39, 39]
    cached = torch.tensor(
        [tables[0][paths[0]], tables[1][paths[1]], tables[0][paths[2]]]
    )
    torch.testing.assert_close(cached, direct)


@pytest.mark.unittest
def test_recent_transformer_incremental_cache_matches_direct_scoring():
    torch.manual_seed(7)
    model = RecentHistoryTransformerNovelty(
        observation_channels=3,
        action_space_size=3,
        embedding_dim=12,
        hidden_dim=24,
        history_window=4,
        max_horizon=3,
        transformer_layers=2,
        transformer_heads=4,
    ).eval()
    with torch.no_grad():
        contexts = []
        for step_count in (3, 1):
            context = model.start_context(torch.rand(1, 3, 20, 20))
            for step in range(step_count):
                context = model.advance_context(
                    context,
                    torch.tensor([step % 3]),
                    torch.rand(1, 3, 20, 20),
                    torch.zeros(1),
                    torch.zeros(1),
                )
            contexts.append(context)
        prefix_cache = model.create_prefix_cache(contexts)
        paths = [(1,), (2, 0), (1, 2, 0), (0, 1, 2)]
        roots = [0, 1, 0, 1]
        cached = model.score_cached_path_endpoints(prefix_cache, roots, paths)
        direct = model.score_path_endpoints(contexts, roots, paths)
        cached_again = model.score_cached_path_endpoints(prefix_cache, roots, paths)

    torch.testing.assert_close(cached, direct)
    torch.testing.assert_close(cached_again, direct)
    assert (1, 2) in prefix_cache.states[0]
    assert (0, 1) in prefix_cache.states[1]


@pytest.mark.unittest
def test_recent_transformer_history_cache_matches_direct_scoring():
    torch.manual_seed(9)
    model = RecentHistoryTransformerNovelty(
        observation_channels=3,
        action_space_size=3,
        embedding_dim=12,
        hidden_dim=24,
        history_window=4,
        max_horizon=3,
        transformer_layers=2,
        transformer_heads=4,
    ).eval()
    with torch.no_grad():
        contexts = []
        for step_count in (0, 3):
            context = model.start_context(torch.rand(1, 3, 20, 20))
            for step in range(step_count):
                context = model.advance_context(
                    context,
                    torch.tensor([step % 3]),
                    torch.rand(1, 3, 20, 20),
                    torch.zeros(1),
                    torch.zeros(1),
                )
            contexts.append(context)
        cache = model.create_history_cache(contexts)
        roots = [0, 1, 0, 1]
        paths = [(2,), (1, 0), (0, 2, 1), (2, 1, 0)]
        cached = model.score_history_cached_path_endpoints(cache, roots, paths)
        direct = model.score_path_endpoints(contexts, roots, paths)
        cached_again = model.score_history_cached_path_endpoints(cache, roots, paths)

    torch.testing.assert_close(cached, direct, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(cached_again, direct, rtol=1e-5, atol=1e-6)


@pytest.mark.unittest
def test_online_context_matches_teacher_forced_replay_history():
    torch.manual_seed(0)
    model = RecurrentSequenceNovelty(
        observation_channels=3,
        action_space_size=3,
        embedding_dim=16,
        hidden_dim=24,
    ).eval()
    observations = torch.rand(2, 6, 3, 20, 20)
    actions = torch.randint(0, 3, (2, 5))
    episodic_rewards = torch.randint(0, 2, (2, 5)).float()
    extrinsic_rewards = torch.randn(2, 5)

    with torch.no_grad():
        replay_states = model.history_states(
            observations,
            actions,
            episodic_rewards,
            extrinsic_rewards,
        )
        context = model.start_context(observations[:, 0])
        torch.testing.assert_close(context.hidden, replay_states[:, 0])
        for step in range(5):
            context = model.advance_context(
                context,
                actions[:, step],
                observations[:, step + 1],
                episodic_rewards[:, step],
                extrinsic_rewards[:, step],
            )
            torch.testing.assert_close(context.hidden, replay_states[:, step + 1])


@pytest.mark.unittest
def test_replay_suffix_scoring_matches_direct_hidden_scoring():
    torch.manual_seed(1)
    model = RecurrentSequenceNovelty(
        observation_channels=3,
        action_space_size=3,
        embedding_dim=12,
        hidden_dim=20,
    ).eval()
    observations = torch.rand(2, 7, 3, 20, 20)
    actions = torch.randint(0, 3, (2, 6))
    rewards = torch.randint(0, 2, (2, 6)).float()
    extrinsic = torch.zeros_like(rewards)
    roots = torch.tensor([[0, 2], [1, 3]])

    with torch.no_grad():
        replay_logits = model.score_actual_suffixes(
            observations,
            actions,
            rewards,
            extrinsic,
            roots,
            horizon=3,
        )
        states = model.history_states(observations, actions, rewards, extrinsic)
        batch = torch.arange(2)[:, None]
        direct_logits = model.score_from_hidden(
            states[batch, roots].reshape(4, 20),
            torch.stack(
                [actions[batch, roots + depth] for depth in range(3)], dim=-1
            ).reshape(4, 3),
        ).reshape(2, 2, 3)

    assert replay_logits == pytest.approx(direct_logits)


@pytest.mark.unittest
def test_variable_length_path_scoring_returns_each_endpoint():
    torch.manual_seed(2)
    model = RecurrentSequenceNovelty(
        observation_channels=3,
        action_space_size=3,
        embedding_dim=12,
        hidden_dim=20,
    ).eval()
    with torch.no_grad():
        contexts = [
            model.start_context(torch.rand(1, 3, 20, 20)),
            model.start_context(torch.rand(1, 3, 20, 20)),
        ]
        endpoints = model.score_path_endpoints(
            contexts,
            root_indices=[0, 1, 0],
            action_paths=[[1], [2, 0], [1, 2, 0]],
        )
        expected = torch.stack(
            [
                model.score_from_hidden(contexts[0].hidden, torch.tensor([[1]]))[0, -1],
                model.score_from_hidden(contexts[1].hidden, torch.tensor([[2, 0]]))[0, -1],
                model.score_from_hidden(contexts[0].hidden, torch.tensor([[1, 2, 0]]))[0, -1],
            ]
        )
    assert endpoints == pytest.approx(expected)


@pytest.mark.unittest
def test_continuation_target_starts_after_the_explicit_leaf_edge():
    raw, gated, factors, valid = episodic_continuation_factor_targets(
        rnd_rewards=torch.tensor([[10.0, 2.0, 4.0, 8.0]]),
        episodic_rewards=torch.tensor([[1.0, 0.0, 1.0, 0.0]]),
        lengths=torch.tensor([4]),
        root_indices=torch.tensor([[0]]),
        prefix_horizon=2,
        value_horizon=2,
        discount_factor=0.5,
        minimum_factor=0.1,
    )

    # Depth one enters transition zero, so its continuation starts at transition one.
    assert raw[0, 0].tolist() == pytest.approx([4.0, 8.0])
    assert gated[0, 0].tolist() == pytest.approx([2.2, 4.4])
    assert factors[0, 0].tolist() == pytest.approx([0.55, 0.55])
    assert valid[0, 0].tolist() == [True, True]


@pytest.mark.unittest
def test_combined_replay_scoring_reuses_the_same_prefix_states():
    torch.manual_seed(3)
    model = RecurrentSequenceNovelty(
        observation_channels=3,
        action_space_size=3,
        embedding_dim=12,
        hidden_dim=20,
    ).eval()
    observations = torch.rand(1, 7, 3, 20, 20)
    actions = torch.randint(0, 3, (1, 6))
    rewards = torch.randint(0, 2, (1, 6)).float()
    extrinsic = torch.zeros_like(rewards)
    roots = torch.tensor([[0, 2]])

    with torch.no_grad():
        novelty, continuation = model.score_actual_suffixes_with_continuation(
            observations,
            actions,
            rewards,
            extrinsic,
            roots,
            horizon=3,
        )
        expected_novelty = model.score_actual_suffixes(
            observations, actions, rewards, extrinsic, roots, horizon=3
        )
        expected_continuation = model.score_actual_continuation_suffixes(
            observations, actions, rewards, extrinsic, roots, horizon=3
        )

    torch.testing.assert_close(novelty, expected_novelty)
    torch.testing.assert_close(continuation, expected_continuation)
