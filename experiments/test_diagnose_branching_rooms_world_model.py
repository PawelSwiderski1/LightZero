import numpy as np
import torch

from experiments.diagnose_branching_rooms_raw_latent_geometry import (
    BranchingRoomsStateSpace,
)
from experiments.diagnose_branching_rooms_world_model import (
    _retrieval_summary,
    _sample_paths,
    evaluate_representation_space,
)


def _tiny_state_space() -> BranchingRoomsStateSpace:
    return BranchingRoomsStateSpace(
        poses=np.asarray(
            [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)], dtype=np.int64
        ),
        observations=np.empty((4, 0), dtype=np.float32),
        observation_group=np.asarray([0, 1, 1, 2], dtype=np.int64),
        transition=np.asarray(
            [[0, 1, 2], [1, 2, 3], [2, 3, 0], [3, 0, 1]], dtype=np.int64
        ),
        rooms=np.asarray([(0, 0), (0, 0), (0, 1), (0, 1)], dtype=np.int64),
    )


def test_sample_paths_follows_exact_transition_table():
    state_space = _tiny_state_space()
    roots, actions, endpoints = _sample_paths(
        state_space, depth=3, count=20, rng=np.random.default_rng(7)
    )
    expected = roots.copy()
    for step in range(actions.shape[1]):
        expected = state_space.transition[expected, actions[:, step]]
    assert np.array_equal(endpoints, expected)


def test_sample_paths_honors_restricted_root_pool():
    state_space = _tiny_state_space()
    roots, _, _ = _sample_paths(
        state_space,
        depth=2,
        count=50,
        rng=np.random.default_rng(11),
        root_pool=np.asarray([1, 3]),
    )
    assert set(roots.tolist()) == {1, 3}


def test_retrieval_masks_physical_alias_with_same_rgb_group():
    state_space = _tiny_state_space()
    queries = torch.tensor([[1.0, 0.0]])
    targets = torch.tensor([[1.0, 0.0]])
    candidates = torch.tensor(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [-1.0, 0.0],
        ]
    )
    result = _retrieval_summary(
        queries,
        targets,
        candidates,
        target_indices=np.asarray([1]),
        candidate_indices=np.asarray([0, 2, 3]),
        state_space=state_space,
        metric="cosine",
        member=0,
    )
    assert result["top1_tie_aware"] == 1.0
    assert result["retrieved_same_position"] == 1.0


def test_ensemble_pooling_compares_corresponding_member_banks():
    state_space = _tiny_state_space()
    predicted = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    targets = predicted.clone()
    candidates = torch.tensor(
        [
            [[0.0, 1.0], [1.0, 0.0]],
            [[-1.0, 0.0], [0.0, -1.0]],
        ]
    )
    records = evaluate_representation_space(
        predicted,
        targets,
        candidates,
        target_indices=np.asarray([1]),
        candidate_indices=np.asarray([0, 3]),
        state_space=state_space,
        space="raw",
        metrics=("cosine",),
    )
    pooled = next(record for record in records if record["member"] == -1)
    assert pooled["top1_tie_aware"] == 1.0
    assert np.isclose(pooled["alignment_cosine"]["mean"], 1.0)
