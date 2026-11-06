import pytest

from lzero.worker.muzero_collector import EpisodicNoveltyMemory, RunningMeanSquaredDistance


def _memory(k=2, distance_stats=None, **kwargs):
    return EpisodicNoveltyMemory(
        distance_stats=distance_stats or RunningMeanSquaredDistance(epsilon=0.0),
        k=k,
        kernel_cluster_distance=kwargs.pop('kernel_cluster_distance', 0.008),
        kernel_epsilon=kwargs.pop('kernel_epsilon', 1e-4),
        pseudo_count_constant=kwargs.pop('pseudo_count_constant', 0.001),
        maximum_similarity=kwargs.pop('maximum_similarity', 8.0),
        reward_clip=kwargs.pop('reward_clip', None),
        **kwargs,
    )


def test_episodic_novelty_uses_only_prior_episode_embeddings():
    distance_stats = RunningMeanSquaredDistance(epsilon=0.0)
    distance_stats.update([1.0, 1.0])
    memory = _memory(k=2, distance_stats=distance_stats)
    memory.reset([0.0])

    reward, pseudo_count, details = memory.score_and_add([2.0])
    assert reward == 0.0
    assert pseudo_count == 0.0
    assert details['warmup'] == 1.0

    reward, pseudo_count, details = memory.score_and_add([1.0])
    expected_kernel = 1e-4 / ((1.0 - 0.008) + 1e-4)
    expected_pseudo_count = 2.0 * expected_kernel
    expected_reward = 1.0 / (expected_pseudo_count ** 0.5 + 0.001)
    assert pseudo_count == pytest.approx(expected_pseudo_count)
    assert reward == pytest.approx(expected_reward)
    assert details['neighbor_count'] == 2.0
    assert details['warmup'] == 0.0


def test_episodic_novelty_duplicate_embedding_has_large_pseudo_count():
    memory = EpisodicNoveltyMemory(
        distance_stats=RunningMeanSquaredDistance(epsilon=0.0),
        k=2,
        reward_clip=None,
    )
    memory.reset([0.0, 0.0])
    memory.score_and_add([0.0, 0.0])

    reward, pseudo_count, details = memory.score_and_add([0.0, 0.0])
    assert pseudo_count == pytest.approx(2.0)
    assert reward == pytest.approx(1.0 / (2.0 ** 0.5 + 0.001))
    assert details['normalized_knn_distance_mean'] == 0.0


def test_episodic_novelty_reset_keeps_distance_normalizer_but_clears_memory():
    distance_stats = RunningMeanSquaredDistance(epsilon=0.0)
    memory = _memory(k=1, distance_stats=distance_stats)
    memory.reset([0.0])
    memory.score_and_add([1.0])
    prior_distance_mean = distance_stats.mean

    memory.reset([2.0])
    reward, _, details = memory.score_and_add([3.0])
    assert reward > 0.0
    assert details['memory_size'] == 1.0
    assert distance_stats.mean == pytest.approx(prior_distance_mean)


def test_episodic_novelty_reward_scaling_and_clipping():
    memory = _memory(k=1, reward_scale=2.0, reward_clip=0.5)
    memory.reset([0.0])
    reward, pseudo_count, _ = memory.score_and_add([1.0])
    assert pseudo_count > 0.0
    assert reward == 0.5
