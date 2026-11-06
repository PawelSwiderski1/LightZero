from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch


@dataclass(frozen=True)
class EpisodicLatentMemoryContext:
    """Direct-root latent keys and their episode-local distance scale."""

    keys: torch.Tensor
    transition_distance_scale: float


class EpisodicLatentRewardGate:
    """Suppress predicted edge rewards near real or path-local episodic memories."""

    def __init__(
            self,
            memories: Sequence[EpisodicLatentMemoryContext],
            kernel_bandwidth: float = 1.0,
            minimum_reward_factor: float = 0.05,
            distance_epsilon: float = 1e-8,
            include_path_memory: bool = True,
            mode: str = 'minimum_distance',
            knn_k: int = 10,
            knn_cluster_distance: float = 0.008,
            knn_kernel_epsilon: float = 1e-4,
            knn_pseudo_count_constant: float = 0.001,
            knn_maximum_similarity: float = 8.0,
            knn_warmup_until_full: bool = True,
    ) -> None:
        if kernel_bandwidth <= 0.0:
            raise ValueError('kernel_bandwidth must be positive.')
        if not 0.0 <= minimum_reward_factor <= 1.0:
            raise ValueError('minimum_reward_factor must be in [0, 1].')
        if distance_epsilon <= 0.0:
            raise ValueError('distance_epsilon must be positive.')
        self._memories = list(memories)
        self._kernel_bandwidth = float(kernel_bandwidth)
        self._minimum_reward_factor = float(minimum_reward_factor)
        self._distance_epsilon = float(distance_epsilon)
        self._include_path_memory = bool(include_path_memory)
        self._mode = str(mode).lower()
        if self._mode not in {'minimum_distance', 'ngu_knn'}:
            raise ValueError(
                "Episodic reward-gate mode must be 'minimum_distance' or 'ngu_knn', "
                f'got {mode!r}.'
            )
        self._knn_k = max(1, int(knn_k))
        self._knn_cluster_distance = max(0.0, float(knn_cluster_distance))
        self._knn_kernel_epsilon = max(1e-12, float(knn_kernel_epsilon))
        self._knn_pseudo_count_constant = max(
            0.0, float(knn_pseudo_count_constant)
        )
        self._knn_maximum_similarity = max(0.0, float(knn_maximum_similarity))
        self._knn_warmup_until_full = bool(knn_warmup_until_full)
        self._path_keys: Dict[Tuple[int, Tuple[int, ...]], torch.Tensor] = {}

    @staticmethod
    def _mean_member_squared_distances(
            candidate: torch.Tensor,
            memory: torch.Tensor,
    ) -> torch.Tensor:
        if candidate.dim() != 2:
            raise ValueError(
                f'candidate keys must have shape [members, features], got {tuple(candidate.shape)}.'
            )
        if memory.dim() != 3:
            raise ValueError(
                f'memory keys must have shape [states, members, features], got {tuple(memory.shape)}.'
            )
        if tuple(memory.shape[1:]) != tuple(candidate.shape):
            raise ValueError(
                'candidate and memory key shapes do not match: '
                f'{tuple(candidate.shape)} versus {tuple(memory.shape[1:])}.'
            )
        return (memory - candidate.unsqueeze(0)).pow(2).sum(dim=-1).mean(dim=-1)

    def _minimum_distance(
            self,
            root_index: int,
            action_path: Tuple[int, ...],
            candidate: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        memory = self._memories[root_index].keys.to(
            device=candidate.device, dtype=candidate.dtype
        )
        if memory.shape[0] > 0:
            real_distance = self._mean_member_squared_distances(
                candidate, memory
            ).amin()
        else:
            real_distance = candidate.new_tensor(float('inf'))

        path_distance = candidate.new_tensor(float('inf'))
        if self._include_path_memory and len(action_path) > 1:
            ancestor_keys = [
                self._path_keys[(root_index, action_path[:depth])]
                for depth in range(1, len(action_path))
                if (root_index, action_path[:depth]) in self._path_keys
            ]
            if ancestor_keys:
                path_memory = torch.stack(ancestor_keys, dim=0).to(
                    device=candidate.device, dtype=candidate.dtype
                )
                path_distance = self._mean_member_squared_distances(
                    candidate, path_memory
                ).amin()
        return real_distance, path_distance

    def _all_distances(
            self,
            root_index: int,
            action_path: Tuple[int, ...],
            candidate: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        memory = self._memories[root_index].keys.to(
            device=candidate.device, dtype=candidate.dtype
        )
        distances = []
        real_count = int(memory.shape[0])
        if real_count > 0:
            distances.append(self._mean_member_squared_distances(candidate, memory))
        if self._include_path_memory and len(action_path) > 1:
            ancestor_keys = [
                self._path_keys[(root_index, action_path[:depth])]
                for depth in range(1, len(action_path))
                if (root_index, action_path[:depth]) in self._path_keys
            ]
            if ancestor_keys:
                path_memory = torch.stack(ancestor_keys, dim=0).to(
                    device=candidate.device, dtype=candidate.dtype
                )
                distances.append(
                    self._mean_member_squared_distances(candidate, path_memory)
                )
        if not distances:
            return candidate.new_empty((0, )), real_count
        return torch.cat(distances, dim=0), real_count

    def _ngu_knn_factor(
            self,
            root_index: int,
            action_path: Tuple[int, ...],
            candidate: torch.Tensor,
            distance_scale: float,
    ) -> Tuple[torch.Tensor, dict]:
        distances, real_count = self._all_distances(
            root_index, action_path, candidate
        )
        neighbor_count = min(self._knn_k, int(distances.numel()))
        warmup = (
            neighbor_count == 0
            or (self._knn_warmup_until_full and real_count < self._knn_k)
        )
        if warmup:
            factor = candidate.new_tensor(1.0)
            return factor, {
                'episodic_neighbor_count': int(neighbor_count),
                'episodic_knn_distance_mean': float('nan'),
                'episodic_normalized_knn_distance_mean': float('nan'),
                'episodic_pseudo_count': 0.0,
                'episodic_similarity': 0.0,
                'episodic_knn_warmup': 1.0,
            }

        nearest = torch.topk(distances, k=neighbor_count, largest=False).values
        normalized = nearest / (
            max(self._kernel_bandwidth * distance_scale, self._distance_epsilon)
        )
        normalized = (normalized - self._knn_cluster_distance).clamp_min(0.0)
        kernels = self._knn_kernel_epsilon / (
            normalized + self._knn_kernel_epsilon
        )
        pseudo_count = kernels.sum().clamp_min(0.0)
        similarity = pseudo_count.sqrt() + self._knn_pseudo_count_constant
        novelty = similarity.reciprocal()
        if (
                self._knn_maximum_similarity > 0.0
                and bool(similarity > self._knn_maximum_similarity)
        ):
            novelty = novelty.new_zeros(())
        factor = novelty.clamp(min=self._minimum_reward_factor, max=1.0)
        return factor, {
            'episodic_neighbor_count': int(neighbor_count),
            'episodic_knn_distance_mean': float(nearest.mean().item()),
            'episodic_normalized_knn_distance_mean': float(normalized.mean().item()),
            'episodic_pseudo_count': float(pseudo_count.item()),
            'episodic_similarity': float(similarity.item()),
            'episodic_knn_warmup': 0.0,
        }

    def apply(
            self,
            base_rewards: torch.Tensor,
            candidate_keys: torch.Tensor,
            root_indices: Sequence[int],
            action_paths: Sequence[Sequence[int]],
    ) -> Tuple[torch.Tensor, List[dict]]:
        if candidate_keys.dim() != 3:
            raise ValueError(
                'candidate_keys must have shape [paths, members, features], '
                f'got {tuple(candidate_keys.shape)}.'
            )
        base_rewards = base_rewards.reshape(-1)
        batch_size = int(candidate_keys.shape[0])
        if not (
                base_rewards.shape[0] == batch_size
                and len(root_indices) == batch_size
                and len(action_paths) == batch_size
        ):
            raise ValueError('Rewards, candidate keys, roots, and paths must have equal batch sizes.')

        factors = []
        records = []
        for row in range(batch_size):
            root_index = int(root_indices[row])
            if root_index < 0 or root_index >= len(self._memories):
                raise IndexError(f'Invalid episodic-memory root index {root_index}.')
            action_path = tuple(int(action) for action in action_paths[row])
            candidate = candidate_keys[row]
            real_distance, path_distance = self._minimum_distance(
                root_index, action_path, candidate
            )
            minimum_distance = torch.minimum(real_distance, path_distance)
            distance_scale = max(
                float(self._memories[root_index].transition_distance_scale),
                self._distance_epsilon,
            )
            knn_details = {}
            if self._mode == 'ngu_knn':
                factor, knn_details = self._ngu_knn_factor(
                    root_index, action_path, candidate, distance_scale
                )
                normalized_distance = minimum_distance / (
                    self._kernel_bandwidth * distance_scale
                )
            else:
                normalized_distance = minimum_distance / (
                    self._kernel_bandwidth * distance_scale
                )
                novelty = -torch.expm1(-normalized_distance).clamp(max=0.0)
                novelty = novelty.clamp(min=0.0, max=1.0)
                factor = self._minimum_reward_factor + (
                    1.0 - self._minimum_reward_factor
                ) * novelty
            factors.append(factor)

            matched_source = 'path' if bool(path_distance < real_distance) else 'real'
            if not bool(torch.isfinite(minimum_distance)):
                matched_source = 'none'
            records.append(
                {
                    'episodic_real_distance': float(real_distance.item()),
                    'episodic_path_distance': float(path_distance.item()),
                    'episodic_min_distance': float(minimum_distance.item()),
                    'episodic_distance_scale': float(distance_scale),
                    'episodic_normalized_distance': float(normalized_distance.item()),
                    'episodic_reward_factor': float(factor.item()),
                    'episodic_matched_source': matched_source,
                    'episodic_real_memory_size': int(
                        self._memories[root_index].keys.shape[0]
                    ),
                    'episodic_gate_mode': self._mode,
                    **knn_details,
                }
            )
            if self._include_path_memory:
                self._path_keys[(root_index, action_path)] = candidate.detach()

        reward_factors = torch.stack(factors).to(
            device=base_rewards.device, dtype=base_rewards.dtype
        )
        return base_rewards * reward_factors, records
