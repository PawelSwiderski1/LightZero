import numpy as np
import torch

from experiments.run_branching_rooms_action_novelty import (
    BRANCH_REVISIT,
    NOVEL,
    ROOT_HISTORY_REVISIT,
    ActionNoveltyPredictor,
    build_novelty_targets,
    state_ids_from_observations,
)


def test_novelty_targets_distinguish_root_and_suffix_revisits():
    # A, B, C, B, D, D: at root A, the second B is a suffix revisit; from
    # root C onward, B is already part of the real root history.
    targets = build_novelty_targets(np.asarray([0, 1, 2, 1, 3, 3]), horizon=3)

    assert targets.tolist() == [
        [NOVEL, NOVEL, BRANCH_REVISIT],
        [NOVEL, ROOT_HISTORY_REVISIT, NOVEL],
        [ROOT_HISTORY_REVISIT, NOVEL, BRANCH_REVISIT],
    ]


def test_exact_rgb_state_ids_merge_only_identical_observations():
    observations = np.zeros((4, 3, 4, 4), dtype=np.uint8)
    observations[1, :, 0, 0] = 1
    observations[2] = observations[0]
    observations[3, :, 1, 1] = 2

    assert state_ids_from_observations(observations).tolist() == [0, 1, 0, 2]


def test_action_novelty_predictor_shapes_and_causal_root_subset():
    observations = torch.randint(0, 256, (2, 9, 3, 14, 14), dtype=torch.uint8)
    actions = torch.randint(0, 3, (2, 8), dtype=torch.long)
    roots = torch.tensor([0, 2, 4])

    for use_history in (False, True):
        model = ActionNoveltyPredictor(
            action_space_size=3,
            embedding_dim=16,
            hidden_dim=32,
            use_history=use_history,
        )
        logits = model(observations, actions, horizon=3, root_indices=roots)
        assert logits.shape == (2, 3, 3, 3)
        assert torch.isfinite(logits).all()

        model.eval()
        root_only = torch.tensor([0])
        with torch.no_grad():
            original = model(
                observations, actions, horizon=3, root_indices=root_only
            )
            changed_future = observations.clone()
            changed_future[:, 1:] = 255 - changed_future[:, 1:]
            after_future_change = model(
                changed_future, actions, horizon=3, root_indices=root_only
            )
        assert torch.allclose(original, after_future_change, atol=1e-6)


def test_online_context_scores_suffixes_without_future_observations():
    model = ActionNoveltyPredictor(
        action_space_size=3,
        embedding_dim=16,
        hidden_dim=32,
        use_history=True,
    )
    model.eval()
    first_observation = torch.randint(0, 256, (1, 3, 14, 14), dtype=torch.uint8)
    next_observation = torch.randint(0, 256, (1, 3, 14, 14), dtype=torch.uint8)
    suffixes = torch.tensor([[0, 2, 1], [2, 2, 0]], dtype=torch.long)

    with torch.no_grad():
        context = model.start_online_context(first_observation)
        first_logits = model.score_action_suffixes(context, suffixes)
        context = model.advance_online_context(
            context, torch.tensor([2]), next_observation
        )
        second_logits = model.score_action_suffixes(context, suffixes)

    assert first_logits.shape == (2, 3, 3)
    assert second_logits.shape == (2, 3, 3)
    assert context.history_states.shape[1] == 2
    assert torch.isfinite(second_logits).all()


def test_online_suffix_scores_match_batched_causal_forward():
    torch.manual_seed(7)
    model = ActionNoveltyPredictor(
        action_space_size=3,
        embedding_dim=16,
        hidden_dim=32,
        use_history=True,
    )
    model.eval()
    observations = torch.randint(0, 256, (1, 5, 3, 14, 14), dtype=torch.uint8)
    past_action = torch.tensor([2], dtype=torch.long)
    suffix = torch.tensor([[1, 0, 2]], dtype=torch.long)
    actions = torch.cat([past_action, suffix.squeeze(0)]).unsqueeze(0)

    with torch.no_grad():
        context = model.start_online_context(observations[:, 0])
        context = model.advance_online_context(
            context, past_action, observations[:, 1]
        )
        online_logits = model.score_action_suffixes(context, suffix)
        batched_logits = model(
            observations,
            actions,
            horizon=3,
            root_indices=torch.tensor([1]),
        ).squeeze(0)

    assert torch.allclose(online_logits, batched_logits, atol=1e-6)
