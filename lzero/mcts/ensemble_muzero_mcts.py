import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch


class EnsembleSearchNode:
    """
    Shared-tree MuZero node whose incoming edge stores vector-valued ensemble statistics.
    """

    def __init__(
            self,
            prior: float,
            legal_actions: Sequence[int],
            ensemble_size: int,
            latent_states: torch.Tensor = None,
            reward: np.ndarray = None,
            intrinsic_reward: float = 0.0,
            q_min_max_stats=None,
    ) -> None:
        self.prior = float(prior)
        self.legal_actions = list(legal_actions)
        self.ensemble_size = int(ensemble_size)
        self.latent_states = latent_states
        self.reward = np.zeros(self.ensemble_size, dtype=np.float32) if reward is None else reward.astype(np.float32)
        self.intrinsic_reward = float(intrinsic_reward)
        self.intrinsic_landscape_query_reward = 0.0
        self.intrinsic_reward_before_path_transform = float(intrinsic_reward)
        self.intrinsic_reward_before_path_modifier = float(intrinsic_reward)
        self.intrinsic_path_reward_modifier = 1.0
        self.intrinsic_reward_before_path_bonus = float(intrinsic_reward)
        self.intrinsic_path_reward_bonus = 0.0
        self.dynamics_disagreement_latent_reward = 0.0
        self.dynamics_disagreement_projection_reward = 0.0
        self.dynamics_disagreement_prediction_projection_reward = 0.0
        self.latent_novelty = 0.0
        self.q_sum = np.zeros(self.ensemble_size, dtype=np.float32)
        self.mixed_q_sum = np.zeros(self.ensemble_size, dtype=np.float32)
        self.calibrated_q_sum = np.zeros(self.ensemble_size, dtype=np.float32)
        self.has_calibrated_q = False
        self.predicted_reward_return_sum = np.zeros(self.ensemble_size, dtype=np.float32)
        self.leaf_value_contribution_sum = np.zeros(self.ensemble_size, dtype=np.float32)
        self.calibrated_leaf_value_contribution_sum = np.zeros(self.ensemble_size, dtype=np.float32)
        self.leaf_latent_novelty_sum = 0.0
        self.leaf_latent_novelty_max = 0.0
        self.leaf_latent_novelty_count = 0
        self.leaf_latent_novelty_depth_sum = 0.0
        self.leaf_latent_novelty_max_depth = 0
        self.leaf_latent_novelty_max_path: List[int] = []
        self.intrinsic_q_sum = 0.0
        self.intrinsic_reward_return_sum = 0.0
        self.intrinsic_reward_weighted_return_sum = 0.0
        self.intrinsic_leaf_value_raw_contribution_sum = 0.0
        self.intrinsic_leaf_value_clipped_contribution_sum = 0.0
        self.intrinsic_leaf_value_contribution_sum = 0.0
        self.intrinsic_leaf_value_before_modifier_contribution_sum = 0.0
        self.intrinsic_leaf_value_modifier_sum = 0.0
        self.q_sum_by_search_depth: Dict[int, np.ndarray] = {}
        self.calibrated_q_sum_by_search_depth: Dict[int, np.ndarray] = {}
        self.visit_count_by_search_depth: Dict[int, int] = {}
        self.raw_backup_disagreement_sum = 0.0
        self.raw_backup_disagreement_sq_sum = 0.0
        self.calibrated_backup_disagreement_sum = 0.0
        self.calibrated_backup_disagreement_sq_sum = 0.0
        self.depth_normalized_backup_disagreement_sum = 0.0
        self.depth_normalized_backup_disagreement_sq_sum = 0.0
        self.depth_disagreement_scale_sum = 0.0
        self.raw_backup_disagreement_sum_by_search_depth: Dict[int, float] = {}
        self.raw_backup_disagreement_sq_sum_by_search_depth: Dict[int, float] = {}
        self.calibrated_backup_disagreement_sum_by_search_depth: Dict[int, float] = {}
        self.calibrated_backup_disagreement_sq_sum_by_search_depth: Dict[int, float] = {}
        self.adjusted_backup_disagreement_sum_by_search_depth: Dict[int, float] = {}
        self.adjusted_backup_disagreement_sq_sum_by_search_depth: Dict[int, float] = {}
        self.depth_disagreement_scale_sum_by_search_depth: Dict[int, float] = {}
        self.visit_count = 0
        self.children: Dict[int, "EnsembleSearchNode"] = {}
        self.q_min_max_stats = q_min_max_stats

    @property
    def expanded(self) -> bool:
        return len(self.children) > 0

    @property
    def q_mean(self) -> np.ndarray:
        if self.visit_count == 0:
            return np.zeros(self.ensemble_size, dtype=np.float32)
        return self.q_sum / self.visit_count

    @property
    def calibrated_q_mean(self) -> np.ndarray:
        if self.visit_count == 0:
            return np.zeros(self.ensemble_size, dtype=np.float32)
        if not self.has_calibrated_q:
            return self.q_mean
        return self.calibrated_q_sum / self.visit_count

    @property
    def mixed_q_mean(self) -> np.ndarray:
        if self.visit_count == 0:
            return np.zeros(self.ensemble_size, dtype=np.float32)
        return self.mixed_q_sum / self.visit_count

    @property
    def predicted_reward_return_mean(self) -> np.ndarray:
        if self.visit_count == 0:
            return np.zeros(self.ensemble_size, dtype=np.float32)
        return self.predicted_reward_return_sum / self.visit_count

    @property
    def leaf_value_contribution_mean(self) -> np.ndarray:
        if self.visit_count == 0:
            return np.zeros(self.ensemble_size, dtype=np.float32)
        return self.leaf_value_contribution_sum / self.visit_count

    @property
    def calibrated_leaf_value_contribution_mean(self) -> np.ndarray:
        if self.visit_count == 0:
            return np.zeros(self.ensemble_size, dtype=np.float32)
        if not self.has_calibrated_q:
            return self.leaf_value_contribution_mean
        return self.calibrated_leaf_value_contribution_sum / self.visit_count

    @property
    def leaf_latent_novelty_mean(self) -> float:
        if self.leaf_latent_novelty_count == 0:
            return 0.0
        return float(self.leaf_latent_novelty_sum / self.leaf_latent_novelty_count)

    @property
    def leaf_latent_novelty_depth_mean(self) -> float:
        if self.leaf_latent_novelty_count == 0:
            return 0.0
        return float(self.leaf_latent_novelty_depth_sum / self.leaf_latent_novelty_count)

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return float(np.mean(self.q_mean))

    @property
    def intrinsic_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return float(self.intrinsic_q_sum / self.visit_count)

    @property
    def mixed_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return float(np.mean(self.mixed_q_mean))

    @property
    def intrinsic_reward_return_mean(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return float(self.intrinsic_reward_return_sum / self.visit_count)

    @property
    def intrinsic_reward_weighted_return_mean(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return float(self.intrinsic_reward_weighted_return_sum / self.visit_count)

    @property
    def intrinsic_leaf_value_raw_contribution_mean(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return float(self.intrinsic_leaf_value_raw_contribution_sum / self.visit_count)

    @property
    def intrinsic_leaf_value_clipped_contribution_mean(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return float(self.intrinsic_leaf_value_clipped_contribution_sum / self.visit_count)

    @property
    def intrinsic_leaf_value_contribution_mean(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return float(self.intrinsic_leaf_value_contribution_sum / self.visit_count)

    @property
    def intrinsic_leaf_value_before_modifier_contribution_mean(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return float(
            self.intrinsic_leaf_value_before_modifier_contribution_sum / self.visit_count
        )

    @property
    def intrinsic_leaf_value_modifier_mean(self) -> float:
        if self.visit_count == 0:
            return 1.0
        return float(self.intrinsic_leaf_value_modifier_sum / self.visit_count)

    @property
    def uncertainty(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return float(np.std(self.calibrated_q_mean))

    @property
    def raw_uncertainty(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return float(np.std(self.q_mean))

    @staticmethod
    def _mean_from_sum(total: float, count: int) -> float:
        return float(total / count) if count > 0 else 0.0

    @staticmethod
    def _rms_from_sq_sum(total: float, count: int) -> float:
        return float(math.sqrt(max(0.0, total / count))) if count > 0 else 0.0

    @property
    def raw_backup_disagreement_mean(self) -> float:
        return self._mean_from_sum(self.raw_backup_disagreement_sum, self.visit_count)

    @property
    def raw_backup_disagreement_rms(self) -> float:
        return self._rms_from_sq_sum(self.raw_backup_disagreement_sq_sum, self.visit_count)

    @property
    def calibrated_backup_disagreement_mean(self) -> float:
        return self._mean_from_sum(self.calibrated_backup_disagreement_sum, self.visit_count)

    @property
    def calibrated_backup_disagreement_rms(self) -> float:
        return self._rms_from_sq_sum(self.calibrated_backup_disagreement_sq_sum, self.visit_count)

    @property
    def depth_normalized_backup_disagreement_mean(self) -> float:
        return self._mean_from_sum(self.depth_normalized_backup_disagreement_sum, self.visit_count)

    @property
    def depth_normalized_backup_disagreement_rms(self) -> float:
        return self._rms_from_sq_sum(self.depth_normalized_backup_disagreement_sq_sum, self.visit_count)

    @property
    def depth_disagreement_scale_mean(self) -> float:
        return self._mean_from_sum(self.depth_disagreement_scale_sum, self.visit_count)

    def record_search_depth_return(
            self,
            search_depth: int,
            returns: np.ndarray,
            calibrated_returns: np.ndarray,
            depth_disagreement_scale: float = 1.0,
    ) -> None:
        search_depth = int(search_depth)
        if search_depth <= 0:
            raise ValueError(f"search_depth must be positive, got {search_depth}.")
        if search_depth not in self.q_sum_by_search_depth:
            self.q_sum_by_search_depth[search_depth] = np.zeros(self.ensemble_size, dtype=np.float32)
            self.calibrated_q_sum_by_search_depth[search_depth] = np.zeros(
                self.ensemble_size, dtype=np.float32
            )
            self.visit_count_by_search_depth[search_depth] = 0
            self.raw_backup_disagreement_sum_by_search_depth[search_depth] = 0.0
            self.raw_backup_disagreement_sq_sum_by_search_depth[search_depth] = 0.0
            self.calibrated_backup_disagreement_sum_by_search_depth[search_depth] = 0.0
            self.calibrated_backup_disagreement_sq_sum_by_search_depth[search_depth] = 0.0
            self.adjusted_backup_disagreement_sum_by_search_depth[search_depth] = 0.0
            self.adjusted_backup_disagreement_sq_sum_by_search_depth[search_depth] = 0.0
            self.depth_disagreement_scale_sum_by_search_depth[search_depth] = 0.0
        raw_disagreement = float(np.std(np.asarray(returns, dtype=np.float32)))
        calibrated_disagreement = float(np.std(np.asarray(calibrated_returns, dtype=np.float32)))
        depth_disagreement_scale = float(depth_disagreement_scale)
        adjusted_disagreement = calibrated_disagreement * depth_disagreement_scale
        self.q_sum_by_search_depth[search_depth] += np.asarray(returns, dtype=np.float32)
        self.calibrated_q_sum_by_search_depth[search_depth] += np.asarray(
            calibrated_returns, dtype=np.float32
        )
        self.visit_count_by_search_depth[search_depth] += 1
        self.raw_backup_disagreement_sum += raw_disagreement
        self.raw_backup_disagreement_sq_sum += raw_disagreement ** 2
        self.calibrated_backup_disagreement_sum += calibrated_disagreement
        self.calibrated_backup_disagreement_sq_sum += calibrated_disagreement ** 2
        self.depth_normalized_backup_disagreement_sum += adjusted_disagreement
        self.depth_normalized_backup_disagreement_sq_sum += adjusted_disagreement ** 2
        self.depth_disagreement_scale_sum += depth_disagreement_scale
        self.raw_backup_disagreement_sum_by_search_depth[search_depth] += raw_disagreement
        self.raw_backup_disagreement_sq_sum_by_search_depth[search_depth] += raw_disagreement ** 2
        self.calibrated_backup_disagreement_sum_by_search_depth[search_depth] += calibrated_disagreement
        self.calibrated_backup_disagreement_sq_sum_by_search_depth[search_depth] += calibrated_disagreement ** 2
        self.adjusted_backup_disagreement_sum_by_search_depth[search_depth] += adjusted_disagreement
        self.adjusted_backup_disagreement_sq_sum_by_search_depth[search_depth] += adjusted_disagreement ** 2
        self.depth_disagreement_scale_sum_by_search_depth[search_depth] += depth_disagreement_scale

    def search_depth_backup_disagreement(self, search_depth: int, kind: str = 'calibrated_mean') -> float:
        search_depth = int(search_depth)
        count = int(self.visit_count_by_search_depth.get(search_depth, 0))
        if count <= 0:
            return 0.0
        sums = {
            'raw_mean': self.raw_backup_disagreement_sum_by_search_depth,
            'calibrated_mean': self.calibrated_backup_disagreement_sum_by_search_depth,
            'adjusted_mean': self.adjusted_backup_disagreement_sum_by_search_depth,
            'scale_mean': self.depth_disagreement_scale_sum_by_search_depth,
        }
        squared_sums = {
            'raw_rms': self.raw_backup_disagreement_sq_sum_by_search_depth,
            'calibrated_rms': self.calibrated_backup_disagreement_sq_sum_by_search_depth,
            'adjusted_rms': self.adjusted_backup_disagreement_sq_sum_by_search_depth,
        }
        if kind in sums:
            return self._mean_from_sum(sums[kind].get(search_depth, 0.0), count)
        if kind in squared_sums:
            return self._rms_from_sq_sum(squared_sums[kind].get(search_depth, 0.0), count)
        raise ValueError(f"Unsupported backup disagreement kind: {kind!r}")

    def search_depth_q_mean(self, search_depth: int, calibrated: bool = False) -> Optional[np.ndarray]:
        search_depth = int(search_depth)
        count = self.visit_count_by_search_depth.get(search_depth, 0)
        if count <= 0:
            return None
        sums = (
            self.calibrated_q_sum_by_search_depth
            if calibrated else self.q_sum_by_search_depth
        )
        return sums[search_depth] / count

    def depth_corrected_q_residual(self, calibrated: bool = True) -> np.ndarray:
        """Return the depth-scaled member residual underlying old-style edge Qstd."""
        total_count = sum(
            int(count) for count in self.visit_count_by_search_depth.values() if count > 0
        )
        if total_count <= 0:
            return np.zeros(self.ensemble_size, dtype=np.float32)

        corrected_residual = np.zeros(self.ensemble_size, dtype=np.float64)
        for depth, count in self.visit_count_by_search_depth.items():
            count = int(count)
            if count <= 0:
                continue
            depth_q = self.search_depth_q_mean(depth, calibrated=calibrated)
            if depth_q is None:
                continue
            depth_q = np.asarray(depth_q, dtype=np.float64)
            depth_residual = depth_q - float(np.mean(depth_q))
            scale = self.search_depth_backup_disagreement(depth, 'scale_mean')
            corrected_residual += (count / total_count) * scale * depth_residual
        return corrected_residual.astype(np.float32)

    def depth_corrected_aggregate_uncertainty(
            self,
            calibrated: bool = True,
            cap_at_aggregate: bool = True,
    ) -> float:
        """
        Apply depth correction to member residuals, then take one final standard deviation.

        Averaging member returns before measuring disagreement preserves the cancellation
        semantics of the original edge Qstd. The optional cap makes depth correction
        suppressive only: changing depth weights cannot create more uncertainty than the
        uncorrected aggregate edge estimate.
        """
        if self.visit_count <= 0:
            return 0.0
        aggregate_q = self.calibrated_q_mean if calibrated else self.q_mean
        aggregate_uncertainty = float(np.std(aggregate_q))
        corrected_uncertainty = float(np.std(self.depth_corrected_q_residual(calibrated)))
        if cap_at_aggregate:
            return min(aggregate_uncertainty, corrected_uncertainty)
        return corrected_uncertainty

    def expand(self, shared_priors: Dict[int, float]) -> None:
        for action in self.legal_actions:
            self.children[int(action)] = EnsembleSearchNode(
                prior=shared_priors[int(action)],
                legal_actions=self.legal_actions,
                ensemble_size=self.ensemble_size,
                q_min_max_stats=self.q_min_max_stats,
            )

    def add_exploration_noise(self, root_noise_weight: float, noises: Sequence[float]) -> None:
        for noise, action in zip(noises, self.legal_actions):
            child = self.children[int(action)]
            child.prior = child.prior * (1 - root_noise_weight) + float(noise) * root_noise_weight

    def get_distribution(self) -> List[int]:
        return [self.children[int(action)].visit_count for action in self.legal_actions]


class IntrinsicMinMaxStats:
    """
    Tracks a per-search value range for MuZero-style normalization of intrinsic Q.
    Starts with a zero baseline so unvisited children have a stable comparison point.
    """

    def __init__(self) -> None:
        self.minimum = 0.0
        self.maximum = 0.0

    def update(self, value: float) -> None:
        value = float(value)
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def normalize(self, value: float) -> float:
        value = float(value)
        value_range = self.maximum - self.minimum
        if value_range <= 1e-8:
            return 0.0
        return float((value - self.minimum) / value_range)


class EnsembleQMinMaxStats:
    """Per-search return range shared by every node in one ensemble tree."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self.minimum = float('inf')
        self.maximum = -float('inf')

    @property
    def value_range(self) -> float:
        if not np.isfinite(self.minimum) or not np.isfinite(self.maximum):
            return 0.0
        return max(0.0, float(self.maximum - self.minimum))

    def update(self, values: np.ndarray) -> None:
        if not self.enabled:
            return
        values = np.asarray(values, dtype=np.float64)
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            return
        self.minimum = min(self.minimum, float(np.min(finite_values)))
        self.maximum = max(self.maximum, float(np.max(finite_values)))

    def normalize_value(self, value: float) -> float:
        value = float(value)
        if not self.enabled:
            return value
        value_range = self.value_range
        if value_range <= 1e-8:
            return value
        return float(np.clip((value - self.minimum) / value_range, 0.0, 1.0))

    def normalize_std(self, value: float) -> float:
        value = float(value)
        if not self.enabled:
            return value
        value_range = self.value_range
        if value_range <= 1e-8:
            return 0.0
        return max(0.0, value / value_range)


class DepthNormalizedBackupDisagreementStats:
    """Phase-frozen relative-depth baselines for aggregate ensemble disagreement."""

    def __init__(self, cfg) -> None:
        self.enabled = bool(cfg.get('depth_normalized_backup_disagreement', False))
        configured_max_depth = int(cfg.get('depth_disagreement_max_depth', 0))
        self.max_depth = max(1, configured_max_depth or int(cfg.get('num_simulations', 1)))
        self.ema_momentum = min(0.999999, max(0.0, float(cfg.get('depth_disagreement_ema_momentum', 0.5))))
        self.trim_fraction = min(0.49, max(0.0, float(cfg.get('depth_disagreement_trim_fraction', 0.1))))
        self.min_groups = max(1, int(cfg.get('depth_disagreement_min_groups', 100)))
        self.min_scale = min(1.0, max(0.0, float(cfg.get('depth_disagreement_min_scale', 0.5))))
        self._raw_baseline: Dict[int, float] = {}
        self._calibrated_baseline: Dict[int, float] = {}
        self._raw_ratio: Dict[int, float] = {}
        self._calibrated_ratio: Dict[int, float] = {}
        self._total_group_count: Dict[int, int] = {}
        self._pending_raw: Dict[int, List[float]] = defaultdict(list)
        self._pending_calibrated: Dict[int, List[float]] = defaultdict(list)
        self._last_completed_phase = -1
        self._last_phase_report: List[Dict[str, Any]] = []

    @staticmethod
    def _trimmed_mean(values: Sequence[float], trim_fraction: float) -> float:
        array = np.asarray(values, dtype=np.float64)
        array = array[np.isfinite(array)]
        if array.size == 0:
            return 0.0
        array.sort()
        trim_count = int(math.floor(array.size * trim_fraction))
        if trim_count > 0 and 2 * trim_count < array.size:
            array = array[trim_count:-trim_count]
        return float(np.mean(array))

    def scale(self, search_depth: int) -> float:
        if not self.enabled or 1 not in self._calibrated_ratio:
            return 1.0
        search_depth = min(self.max_depth, max(1, int(search_depth)))
        effective_ratio = 1.0
        for depth in range(2, search_depth + 1):
            ratio = self._calibrated_ratio.get(depth, effective_ratio)
            effective_ratio = max(effective_ratio, float(ratio))
        return float(np.clip(1.0 / max(1.0, effective_ratio), self.min_scale, 1.0))

    def baseline_snapshot(self, search_depth: int) -> Dict[str, float]:
        search_depth = min(self.max_depth, max(1, int(search_depth)))
        raw_baseline = float(self._raw_baseline.get(search_depth, 0.0))
        calibrated_baseline = float(self._calibrated_baseline.get(search_depth, 0.0))
        raw_ratio = float(self._raw_ratio.get(search_depth, 0.0))
        calibrated_ratio = float(self._calibrated_ratio.get(search_depth, 0.0))
        effective_baseline = 0.0
        effective_ratio = 0.0
        if 1 in self._calibrated_baseline:
            effective_baseline = float(self._calibrated_baseline[1])
            for depth in range(2, search_depth + 1):
                effective_baseline = max(
                    effective_baseline,
                    float(self._calibrated_baseline.get(depth, effective_baseline)),
                )
        if 1 in self._calibrated_ratio:
            effective_ratio = 1.0
            for depth in range(2, search_depth + 1):
                effective_ratio = max(
                    effective_ratio,
                    float(self._calibrated_ratio.get(depth, effective_ratio)),
                )
        return {
            'raw_baseline': raw_baseline,
            'calibrated_baseline': calibrated_baseline,
            'raw_ratio': raw_ratio,
            'calibrated_ratio': calibrated_ratio,
            'effective_calibrated_baseline': effective_baseline,
            'effective_calibrated_ratio': effective_ratio,
            'scale': self.scale(search_depth),
            'total_group_count': int(self._total_group_count.get(search_depth, 0)),
            'initialized': float(search_depth in self._calibrated_ratio),
        }

    def observe_roots(self, roots: Sequence[EnsembleSearchNode]) -> None:
        if not self.enabled:
            return
        stack = list(roots)
        while stack:
            node = stack.pop()
            stack.extend(node.children.values())
            for depth, count in node.visit_count_by_search_depth.items():
                if count <= 0 or depth > self.max_depth:
                    continue
                raw_q = node.search_depth_q_mean(depth, calibrated=False)
                calibrated_q = node.search_depth_q_mean(depth, calibrated=True)
                if raw_q is None or calibrated_q is None:
                    continue
                self._pending_raw[depth].append(float(np.std(raw_q)))
                self._pending_calibrated[depth].append(float(np.std(calibrated_q)))

    def finalize_phase(self, collection_phase: int) -> List[Dict[str, Any]]:
        collection_phase = int(collection_phase)
        report = []
        phase_raw_by_depth: Dict[int, float] = {}
        phase_calibrated_by_depth: Dict[int, float] = {}
        updated_by_depth: Dict[int, bool] = {}
        for depth in range(1, self.max_depth + 1):
            raw_values = self._pending_raw.get(depth, [])
            calibrated_values = self._pending_calibrated.get(depth, [])
            group_count = min(len(raw_values), len(calibrated_values))
            phase_raw = self._trimmed_mean(raw_values, self.trim_fraction)
            phase_calibrated = self._trimmed_mean(calibrated_values, self.trim_fraction)
            updated = group_count >= self.min_groups
            if updated:
                if depth not in self._calibrated_baseline:
                    self._raw_baseline[depth] = phase_raw
                    self._calibrated_baseline[depth] = phase_calibrated
                else:
                    momentum = self.ema_momentum
                    self._raw_baseline[depth] = (
                        momentum * self._raw_baseline[depth] + (1.0 - momentum) * phase_raw
                    )
                    self._calibrated_baseline[depth] = (
                        momentum * self._calibrated_baseline[depth]
                        + (1.0 - momentum) * phase_calibrated
                    )
                self._total_group_count[depth] = self._total_group_count.get(depth, 0) + group_count
            phase_raw_by_depth[depth] = phase_raw
            phase_calibrated_by_depth[depth] = phase_calibrated
            updated_by_depth[depth] = updated

        phase_raw_ratio_by_depth: Dict[int, float] = {}
        phase_calibrated_ratio_by_depth: Dict[int, float] = {}
        ratio_updated_by_depth: Dict[int, bool] = {}
        raw_anchor = phase_raw_by_depth.get(1, 0.0)
        calibrated_anchor = phase_calibrated_by_depth.get(1, 0.0)
        anchor_updated = updated_by_depth.get(1, False) and calibrated_anchor > 1e-8
        for depth in range(1, self.max_depth + 1):
            ratio_updated = bool(anchor_updated and updated_by_depth[depth])
            phase_raw_ratio = (
                phase_raw_by_depth[depth] / raw_anchor
                if ratio_updated and raw_anchor > 1e-8 else 0.0
            )
            phase_calibrated_ratio = (
                phase_calibrated_by_depth[depth] / calibrated_anchor
                if ratio_updated else 0.0
            )
            if ratio_updated:
                if depth not in self._calibrated_ratio:
                    self._raw_ratio[depth] = phase_raw_ratio
                    self._calibrated_ratio[depth] = phase_calibrated_ratio
                else:
                    momentum = self.ema_momentum
                    self._raw_ratio[depth] = (
                        momentum * self._raw_ratio[depth] + (1.0 - momentum) * phase_raw_ratio
                    )
                    self._calibrated_ratio[depth] = (
                        momentum * self._calibrated_ratio[depth]
                        + (1.0 - momentum) * phase_calibrated_ratio
                    )
            phase_raw_ratio_by_depth[depth] = phase_raw_ratio
            phase_calibrated_ratio_by_depth[depth] = phase_calibrated_ratio
            ratio_updated_by_depth[depth] = ratio_updated

        for depth in range(1, self.max_depth + 1):
            snapshot = self.baseline_snapshot(depth)
            report.append(
                {
                    'collection_phase': collection_phase,
                    'depth': depth,
                    'phase_group_count': min(
                        len(self._pending_raw.get(depth, [])),
                        len(self._pending_calibrated.get(depth, [])),
                    ),
                    'phase_raw_trimmed_mean': phase_raw_by_depth[depth],
                    'phase_calibrated_trimmed_mean': phase_calibrated_by_depth[depth],
                    'phase_raw_ratio_to_depth_1': phase_raw_ratio_by_depth[depth],
                    'phase_calibrated_ratio_to_depth_1': phase_calibrated_ratio_by_depth[depth],
                    'ema_raw_baseline': snapshot['raw_baseline'],
                    'ema_calibrated_baseline': snapshot['calibrated_baseline'],
                    'ema_raw_ratio_to_depth_1': snapshot['raw_ratio'],
                    'ema_calibrated_ratio_to_depth_1': snapshot['calibrated_ratio'],
                    'effective_calibrated_baseline': snapshot['effective_calibrated_baseline'],
                    'effective_calibrated_ratio_to_depth_1': snapshot['effective_calibrated_ratio'],
                    'depth_scale_next_phase': snapshot['scale'],
                    'total_group_count': snapshot['total_group_count'],
                    'baseline_initialized': snapshot['initialized'],
                    'baseline_updated': float(updated_by_depth[depth]),
                    'ratio_updated': float(ratio_updated_by_depth[depth]),
                }
            )
        self._pending_raw.clear()
        self._pending_calibrated.clear()
        self._last_completed_phase = collection_phase
        self._last_phase_report = report
        return [dict(row) for row in report]

    def state_dict(self) -> Dict[str, Any]:
        return {
            'version': 2,
            'raw_baseline': dict(self._raw_baseline),
            'calibrated_baseline': dict(self._calibrated_baseline),
            'raw_ratio': dict(self._raw_ratio),
            'calibrated_ratio': dict(self._calibrated_ratio),
            'total_group_count': dict(self._total_group_count),
            'last_completed_phase': int(self._last_completed_phase),
            'last_phase_report': [dict(row) for row in self._last_phase_report],
        }

    def load_state_dict(self, state_dict: Optional[Dict[str, Any]]) -> None:
        if not state_dict:
            return
        self._raw_baseline = {
            int(depth): float(value) for depth, value in state_dict.get('raw_baseline', {}).items()
        }
        self._calibrated_baseline = {
            int(depth): float(value) for depth, value in state_dict.get('calibrated_baseline', {}).items()
        }
        self._raw_ratio = {
            int(depth): float(value) for depth, value in state_dict.get('raw_ratio', {}).items()
        }
        self._calibrated_ratio = {
            int(depth): float(value) for depth, value in state_dict.get('calibrated_ratio', {}).items()
        }
        self._total_group_count = {
            int(depth): int(value) for depth, value in state_dict.get('total_group_count', {}).items()
        }
        self._last_completed_phase = int(state_dict.get('last_completed_phase', -1))
        self._last_phase_report = [dict(row) for row in state_dict.get('last_phase_report', [])]
        self._pending_raw.clear()
        self._pending_calibrated.clear()


@dataclass
class EnsembleSearchOutput:
    distributions: List[List[int]]
    values: List[float]
    intrinsic_values: List[float]
    root_uncertainties: List[float]
    shared_policy_logits: List[List[float]]
    predicted_values: List[float]
    root_action_diagnostics: List[List[Dict[str, Any]]]
    tree_diagnostics: Optional[List[Optional[Dict[str, Any]]]] = None


class EnsembleSharedTreeMCTS:
    """
    Python prototype of a shared-tree ensemble MuZero search.

    The tree has one node per action history, while every node stores one latent
    state per ensemble member. Every edge stores vector-valued Q statistics.
    """

    def __init__(
            self,
            cfg,
            value_inverse_scalar_transform_handle,
            reward_inverse_scalar_transform_handle,
            depth_disagreement_stats: Optional[DepthNormalizedBackupDisagreementStats] = None,
    ) -> None:
        self._cfg = cfg
        self._pb_c_base = float(cfg.get('pb_c_base', 19652))
        self._pb_c_init = float(cfg.get('pb_c_init', 1.25))
        self._root_dirichlet_alpha = float(cfg.get('root_dirichlet_alpha', 0.3))
        self._root_noise_weight = float(cfg.get('root_noise_weight', 0.25))
        self.value_inverse_scalar_transform_handle = value_inverse_scalar_transform_handle
        self.reward_inverse_scalar_transform_handle = reward_inverse_scalar_transform_handle
        self.depth_disagreement_stats = (
            depth_disagreement_stats
            if depth_disagreement_stats is not None
            else DepthNormalizedBackupDisagreementStats(cfg)
        )

    def search(
            self,
            model: torch.nn.Module,
            root_latent_states: torch.Tensor,
            root_policy_logits: torch.Tensor,
            root_values: torch.Tensor,
            legal_actions: List[List[int]],
            add_exploration_noise: bool,
            mode: str,
            intrinsic_reward_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
            use_model_intrinsic: bool = False,
            ensemble_ucb_beta: Optional[float] = None,
            ensemble_uncertainty_decay: Optional[bool] = None,
            num_simulations: Optional[int] = None,
            intrinsic_reward_weight: Optional[float] = None,
            intrinsic_reward_source: str = "rnd",
            return_tree_diagnostics: bool = False,
            tree_diagnostics_max_depth: Optional[int] = None,
            tree_diagnostics_max_nodes: Optional[int] = None,
            value_calibration_offsets: Optional[Sequence[float]] = None,
            collect_depth_disagreement_statistics: bool = False,
            policy_prior_mode: str = "learned",
            force_root_action_coverage: bool = False,
            intrinsic_path_reward_fn: Optional[
                Callable[[Sequence[int], Sequence[Sequence[int]], torch.Tensor], np.ndarray]
            ] = None,
            intrinsic_path_reward_transform_fn: Optional[Callable] = None,
            intrinsic_path_reward_modifier_fn: Optional[
                Callable[[Sequence[int], Sequence[Sequence[int]], torch.Tensor], np.ndarray]
            ] = None,
            intrinsic_path_reward_bonus_fn: Optional[
                Callable[[Sequence[int], Sequence[Sequence[int]], torch.Tensor], np.ndarray]
            ] = None,
            intrinsic_leaf_value_modifier_fn: Optional[
                Callable[[Sequence[int], Sequence[Sequence[int]], torch.Tensor], np.ndarray]
            ] = None,
    ) -> EnsembleSearchOutput:
        assert mode in [
            "mean_shared_tree",
            "mixed_reward_shared_tree",
            "optimistic_shared_tree",
            "pessimistic_shared_tree",
            "bounded_optimistic_shared_tree",
            "bounded_pessimistic_shared_tree",
            "intrinsic_shared_tree",
        ], mode
        intrinsic_reward_source = str(intrinsic_reward_source or "rnd")
        assert intrinsic_reward_source in ["rnd", "dynamics_disagreement", "qstd"], intrinsic_reward_source
        policy_prior_mode = str(policy_prior_mode or "learned").lower()
        if policy_prior_mode not in {"learned", "uniform"}:
            raise ValueError(
                "policy_prior_mode must be either 'learned' or 'uniform', "
                f"got {policy_prior_mode!r}."
            )
        force_root_action_coverage = bool(force_root_action_coverage)
        batch_size, ensemble_size = root_latent_states.shape[0], root_latent_states.shape[1]
        if value_calibration_offsets is None:
            value_calibration_offsets = np.zeros(ensemble_size, dtype=np.float32)
        else:
            value_calibration_offsets = np.asarray(value_calibration_offsets, dtype=np.float32).reshape(-1)
            if value_calibration_offsets.shape != (ensemble_size, ):
                raise ValueError(
                    "value_calibration_offsets must have one value per ensemble member, "
                    f"got {value_calibration_offsets.shape} for ensemble_size={ensemble_size}."
                )
        if ensemble_ucb_beta is None:
            ensemble_ucb_beta = float(self._cfg.get('ensemble_ucb_beta', 0.0))
        else:
            ensemble_ucb_beta = float(ensemble_ucb_beta)
        if ensemble_uncertainty_decay is None:
            ensemble_uncertainty_decay = bool(self._cfg.get('ensemble_uncertainty_decay', False))
        else:
            ensemble_uncertainty_decay = bool(ensemble_uncertainty_decay)
        if num_simulations is None:
            num_simulations = int(self._cfg.num_simulations)
        else:
            num_simulations = int(num_simulations)
        if intrinsic_reward_weight is None:
            intrinsic_reward_weight = 1.0 if mode == "intrinsic_shared_tree" else float(
                self._cfg.get('rnd_intrinsic_reward_weight', 0.0)
            )
        else:
            intrinsic_reward_weight = float(intrinsic_reward_weight)
        roots = self._make_roots(
            root_latent_states=root_latent_states,
            root_policy_logits=root_policy_logits,
            legal_actions=legal_actions,
            add_exploration_noise=add_exploration_noise,
            policy_prior_mode=policy_prior_mode,
            force_root_action_coverage=force_root_action_coverage,
        )
        normalize_intrinsic_q = (
            mode == "intrinsic_shared_tree"
            and bool(self._cfg.get('intrinsic_tree_normalize_q', True))
        )
        intrinsic_min_max_stats: List[Optional[IntrinsicMinMaxStats]] = [
            IntrinsicMinMaxStats() if normalize_intrinsic_q else None for _ in roots
        ]

        model.eval()
        simulation_traces: Optional[List[List[Dict[str, Any]]]] = (
            [[] for _ in range(batch_size)] if return_tree_diagnostics else None
        )
        with torch.no_grad():
            for simulation_idx in range(num_simulations):
                selected = [
                    self._select_leaf(
                        root,
                        mode,
                        ensemble_ucb_beta,
                        ensemble_uncertainty_decay,
                        intrinsic_reward_weight,
                        intrinsic_min_max_stats[root_idx],
                        force_root_action_coverage,
                    ) for root_idx, root in enumerate(roots)
                ]
                for root_idx, selection in enumerate(selected):
                    selection['root_index'] = int(root_idx)
                self._expand_selected(
                    model,
                    selected,
                    ensemble_size,
                    intrinsic_reward_fn,
                    use_model_intrinsic,
                    intrinsic_reward_source,
                    policy_prior_mode,
                    intrinsic_path_reward_fn,
                    intrinsic_path_reward_transform_fn,
                    intrinsic_path_reward_modifier_fn,
                    intrinsic_path_reward_bonus_fn,
                    intrinsic_leaf_value_modifier_fn,
                )
                for root_idx, selection in enumerate(selected):
                    backup_trace = self._backup(
                        selection['root'],
                        selection['path'],
                        selection['leaf_values'],
                        selection.get('leaf_intrinsic_value', 0.0),
                        selection.get('leaf_latent_novelty', 0.0),
                        selection.get('leaf_depth', len(selection['path'])),
                        selection.get('path_actions', []),
                        intrinsic_min_max_stats[root_idx],
                        value_calibration_offsets,
                        mixed_reward_backup=(mode == "mixed_reward_shared_tree"),
                        intrinsic_reward_weight=intrinsic_reward_weight,
                        leaf_intrinsic_value_before_modifier=selection.get(
                            'leaf_intrinsic_value_before_modifier',
                            selection.get('leaf_intrinsic_value', 0.0),
                        ),
                        leaf_intrinsic_value_modifier=selection.get(
                            'leaf_intrinsic_value_modifier', 1.0
                        ),
                    )
                    if simulation_traces is not None:
                        simulation_traces[root_idx].append(
                            {
                                'simulation': int(simulation_idx),
                                'expanded_action': int(selection.get('action', -1)),
                                **backup_trace,
                            }
                        )

        if collect_depth_disagreement_statistics:
            self.depth_disagreement_stats.observe_roots(roots)

        return EnsembleSearchOutput(
            distributions=[root.get_distribution() for root in roots],
            values=[root.value for root in roots],
            intrinsic_values=[root.intrinsic_value for root in roots],
            root_uncertainties=[root.uncertainty for root in roots],
            shared_policy_logits=[
                self._shared_policy_logits(root_policy_logits[i], legal_actions[i]) for i in range(batch_size)
            ],
            predicted_values=self._to_numpy(root_values).reshape(batch_size, ensemble_size).mean(axis=1).tolist(),
            root_action_diagnostics=[
                self._root_action_diagnostics(
                    root,
                    mode,
                    ensemble_ucb_beta,
                    ensemble_uncertainty_decay,
                    intrinsic_reward_weight,
                    intrinsic_min_max_stats[i],
                )
                for i, root in enumerate(roots)
            ],
            tree_diagnostics=[
                self._tree_diagnostics(
                    root,
                    mode,
                    ensemble_ucb_beta,
                    ensemble_uncertainty_decay,
                    intrinsic_reward_weight,
                    intrinsic_min_max_stats[i],
                    simulation_traces[i] if simulation_traces is not None else None,
                    tree_diagnostics_max_depth,
                    tree_diagnostics_max_nodes,
                )
                for i, root in enumerate(roots)
            ] if return_tree_diagnostics else None,
        )

    def _make_roots(
            self,
            root_latent_states: torch.Tensor,
            root_policy_logits: torch.Tensor,
            legal_actions: List[List[int]],
            add_exploration_noise: bool,
            policy_prior_mode: str = "learned",
            force_root_action_coverage: bool = False,
    ) -> List[EnsembleSearchNode]:
        roots = []
        for batch_idx, actions in enumerate(legal_actions):
            shared_priors = self._action_priors(
                root_policy_logits[batch_idx], actions, policy_prior_mode
            )
            q_min_max_stats = EnsembleQMinMaxStats(
                enabled=bool(self._cfg.get('ensemble_normalize_q', True))
            )
            root = EnsembleSearchNode(
                prior=0.0,
                legal_actions=actions,
                ensemble_size=root_latent_states.shape[1],
                latent_states=root_latent_states[batch_idx],
                q_min_max_stats=q_min_max_stats,
            )
            root.policy_prior_mode = str(policy_prior_mode)
            root.force_root_action_coverage = bool(force_root_action_coverage)
            root.expand(shared_priors)
            if add_exploration_noise and len(actions) > 0:
                noises = np.random.dirichlet([self._root_dirichlet_alpha] * len(actions)).astype(np.float32)
                root.add_exploration_noise(self._root_noise_weight, noises)
            roots.append(root)
        return roots

    def _select_leaf(
            self,
            root: EnsembleSearchNode,
            mode: str,
            ensemble_ucb_beta: float,
            ensemble_uncertainty_decay: bool,
            intrinsic_reward_weight: float,
            intrinsic_min_max_stats: Optional[IntrinsicMinMaxStats] = None,
            force_root_action_coverage: bool = False,
    ) -> Dict[str, Any]:
        node = root
        path = []
        path_actions = []
        while node.expanded:
            unvisited_root_actions = []
            if node is root and force_root_action_coverage:
                unvisited_root_actions = [
                    int(action)
                    for action in node.legal_actions
                    if node.children[int(action)].visit_count == 0
                ]
            if unvisited_root_actions:
                action = random.choice(unvisited_root_actions)
            else:
                action = self._select_action(
                    node,
                    mode,
                    ensemble_ucb_beta,
                    ensemble_uncertainty_decay,
                    intrinsic_reward_weight,
                    intrinsic_min_max_stats,
                )
            child = node.children[int(action)]
            path.append(child)
            path_actions.append(int(action))
            if not child.expanded:
                return {
                    'root': root,
                    'parent': node,
                    'action': int(action),
                    'child': child,
                    'path': path,
                    'path_actions': path_actions,
                }
            node = child
        raise RuntimeError("Shared-tree search reached an unexpanded root, which should be expanded before search.")

    def _select_action(
            self,
            node: EnsembleSearchNode,
            mode: str,
            ensemble_ucb_beta: float,
            ensemble_uncertainty_decay: bool,
            intrinsic_reward_weight: float,
            intrinsic_min_max_stats: Optional[IntrinsicMinMaxStats] = None,
    ) -> int:
        max_score = -float('inf')
        best_actions = []
        for action in node.legal_actions:
            child = node.children[int(action)]
            score = self._score_child(
                node,
                child,
                mode,
                ensemble_ucb_beta,
                ensemble_uncertainty_decay,
                intrinsic_reward_weight,
                intrinsic_min_max_stats,
            )
            if score > max_score + 1e-8:
                max_score = score
                best_actions = [int(action)]
            elif score >= max_score - 1e-8:
                best_actions.append(int(action))
        return random.choice(best_actions)

    def _score_child(
            self,
            parent: EnsembleSearchNode,
            child: EnsembleSearchNode,
            mode: str,
            ensemble_ucb_beta: float,
            ensemble_uncertainty_decay: bool,
            intrinsic_reward_weight: float,
            intrinsic_min_max_stats: Optional[IntrinsicMinMaxStats] = None,
    ) -> float:
        return self._score_child_components(
            parent,
            child,
            mode,
            ensemble_ucb_beta,
            ensemble_uncertainty_decay,
            intrinsic_reward_weight,
            intrinsic_min_max_stats,
        )['total_score']

    def _score_child_components(
            self,
            parent: EnsembleSearchNode,
            child: EnsembleSearchNode,
            mode: str,
            ensemble_ucb_beta: float,
            ensemble_uncertainty_decay: bool,
            intrinsic_reward_weight: float,
            intrinsic_min_max_stats: Optional[IntrinsicMinMaxStats] = None,
    ) -> Dict[str, float]:
        pb_c = math.log((max(1, parent.visit_count) + self._pb_c_base + 1) / self._pb_c_base)
        pb_c += self._pb_c_init
        pb_c *= math.sqrt(max(1, parent.visit_count)) / (child.visit_count + 1)
        prior_score = pb_c * child.prior

        mixed_reward_mode = mode == "mixed_reward_shared_tree"
        q_vector = None
        if child.visit_count == 0:
            mean_q = 0.0
            mixed_mean_q = 0.0
            mean_q_for_score = (
                parent.mixed_value if mixed_reward_mode else parent.value
            ) if parent.visit_count > 0 else 0.0
            uncertainty = 0.0
            raw_uncertainty = 0.0
            aggregate_calibrated_uncertainty = 0.0
            depth_corrected_uncertainty_uncapped = 0.0
            depth_corrected_uncertainty = 0.0
            depth_correction_cap_applied = False
        else:
            q_vector = child.q_mean
            mean_q = float(np.mean(q_vector))
            mixed_mean_q = float(np.mean(child.mixed_q_mean))
            mean_q_for_score = mixed_mean_q if mixed_reward_mode else mean_q
            raw_uncertainty = float(np.std(q_vector))
            aggregate_calibrated_uncertainty = float(np.std(child.calibrated_q_mean))
            depth_corrected_uncertainty_uncapped = child.depth_corrected_aggregate_uncertainty(
                calibrated=True,
                cap_at_aggregate=False,
            )
            cap_depth_correction = bool(
                self._cfg.get('depth_disagreement_conservative_cap', True)
            )
            depth_corrected_uncertainty = child.depth_corrected_aggregate_uncertainty(
                calibrated=True,
                cap_at_aggregate=cap_depth_correction,
            )
            depth_correction_cap_applied = bool(
                cap_depth_correction
                and depth_corrected_uncertainty_uncapped > aggregate_calibrated_uncertainty + 1e-8
            )
            if self.depth_disagreement_stats.enabled:
                uncertainty = depth_corrected_uncertainty
                uncertainty_metric = 'depth_corrected_aggregate_q_std'
            else:
                uncertainty = aggregate_calibrated_uncertainty
                uncertainty_metric = 'aggregate_calibrated_q_std'

        q_min_max_stats = parent.q_min_max_stats
        if q_min_max_stats is None:
            normalized_mean_q = mean_q_for_score
            normalized_uncertainty = uncertainty
            q_min = 0.0
            q_max = 0.0
            q_range = 0.0
        else:
            normalized_mean_q = q_min_max_stats.normalize_value(mean_q_for_score)
            normalized_uncertainty = q_min_max_stats.normalize_std(uncertainty)
            q_min = q_min_max_stats.minimum if np.isfinite(q_min_max_stats.minimum) else 0.0
            q_max = q_min_max_stats.maximum if np.isfinite(q_min_max_stats.maximum) else 0.0
            q_range = q_min_max_stats.value_range

        if ensemble_uncertainty_decay:
            normalized_uncertainty = normalized_uncertainty / math.sqrt(1 + child.visit_count)

        raw_uncertainty_for_score = uncertainty
        if ensemble_uncertainty_decay:
            raw_uncertainty_for_score = raw_uncertainty_for_score / math.sqrt(1 + child.visit_count)

        optimistic_mode = mode in ["optimistic_shared_tree", "bounded_optimistic_shared_tree"]
        pessimistic_mode = mode in ["pessimistic_shared_tree", "bounded_pessimistic_shared_tree"]
        bounded_mode = mode in ["bounded_optimistic_shared_tree", "bounded_pessimistic_shared_tree"]
        uncertainty_sign = -1.0 if pessimistic_mode else 1.0
        unbounded_ensemble_q = mean_q_for_score
        bounded_ensemble_q = mean_q_for_score
        ensemble_envelope_q = mean_q_for_score
        ensemble_envelope_member_id = -1
        ensemble_envelope_cap_applied = False
        ensemble_envelope_cap_adjustment = 0.0

        if optimistic_mode or pessimistic_mode:
            unbounded_ensemble_q = (
                mean_q_for_score + uncertainty_sign * ensemble_ucb_beta * raw_uncertainty_for_score
            )
            bounded_ensemble_q = unbounded_ensemble_q
            if q_vector is not None:
                if optimistic_mode:
                    ensemble_envelope_member_id = int(np.argmax(q_vector))
                    ensemble_envelope_q = float(q_vector[ensemble_envelope_member_id])
                    if bounded_mode:
                        bounded_ensemble_q = min(unbounded_ensemble_q, ensemble_envelope_q)
                else:
                    ensemble_envelope_member_id = int(np.argmin(q_vector))
                    ensemble_envelope_q = float(q_vector[ensemble_envelope_member_id])
                    if bounded_mode:
                        bounded_ensemble_q = max(unbounded_ensemble_q, ensemble_envelope_q)
                ensemble_envelope_cap_adjustment = abs(unbounded_ensemble_q - bounded_ensemble_q)
                ensemble_envelope_cap_applied = ensemble_envelope_cap_adjustment > 1e-8

        if q_min_max_stats is None:
            normalized_ensemble_q = bounded_ensemble_q
        else:
            normalized_ensemble_q = q_min_max_stats.normalize_value(bounded_ensemble_q)

        unbounded_uncertainty_score = uncertainty_sign * ensemble_ucb_beta * normalized_uncertainty
        actual_uncertainty_score = normalized_ensemble_q - normalized_mean_q

        if mode == "intrinsic_shared_tree":
            uncertainty_score = 0.0
            normalized_intrinsic_value = (
                intrinsic_min_max_stats.normalize(child.intrinsic_value)
                if intrinsic_min_max_stats is not None else child.intrinsic_value
            )
            intrinsic_score = intrinsic_reward_weight * normalized_intrinsic_value
            total_score = prior_score + intrinsic_score
        elif mode == "mixed_reward_shared_tree":
            uncertainty_score = 0.0
            normalized_intrinsic_value = child.intrinsic_value
            intrinsic_score = intrinsic_reward_weight * child.intrinsic_value
            total_score = normalized_mean_q + prior_score
        elif mode == "mean_shared_tree":
            uncertainty_score = 0.0
            normalized_intrinsic_value = child.intrinsic_value
            intrinsic_score = intrinsic_reward_weight * child.intrinsic_value
            total_score = normalized_mean_q + prior_score + intrinsic_score
        elif bounded_mode:
            uncertainty_score = actual_uncertainty_score
            normalized_intrinsic_value = child.intrinsic_value
            intrinsic_score = intrinsic_reward_weight * child.intrinsic_value
            total_score = normalized_ensemble_q + prior_score + intrinsic_score
        else:
            uncertainty_score = unbounded_uncertainty_score
            normalized_intrinsic_value = child.intrinsic_value
            intrinsic_score = intrinsic_reward_weight * child.intrinsic_value
            total_score = normalized_mean_q + prior_score + uncertainty_score + intrinsic_score
        return {
            'total_score': float(total_score),
            'mean_q': float(mean_q),
            'mixed_mean_q': float(mixed_mean_q),
            'mean_q_for_score': float(mean_q_for_score),
            'normalized_mean_q': float(normalized_mean_q),
            'unbounded_ensemble_q': float(unbounded_ensemble_q),
            'bounded_ensemble_q': float(bounded_ensemble_q),
            'normalized_ensemble_q': float(normalized_ensemble_q),
            'ensemble_envelope_q': float(ensemble_envelope_q),
            'ensemble_envelope_gap_from_mean': float(ensemble_envelope_q - mean_q_for_score),
            'ensemble_envelope_member_id': int(ensemble_envelope_member_id),
            'ensemble_envelope_cap_applied': float(ensemble_envelope_cap_applied),
            'ensemble_envelope_cap_adjustment': float(ensemble_envelope_cap_adjustment),
            'bounded_ensemble_mode': float(bounded_mode),
            'prior_score': float(prior_score),
            'uncertainty_score': float(uncertainty_score),
            'unbounded_uncertainty_score': float(unbounded_uncertainty_score),
            'intrinsic_score': float(intrinsic_score),
            'normalized_intrinsic_value': float(normalized_intrinsic_value),
            'uncertainty': float(uncertainty),
            'raw_uncertainty': float(raw_uncertainty),
            'aggregate_calibrated_uncertainty': float(
                aggregate_calibrated_uncertainty
            ),
            'depth_corrected_uncertainty_uncapped': float(depth_corrected_uncertainty_uncapped),
            'depth_corrected_uncertainty': float(depth_corrected_uncertainty),
            'depth_correction_cap_applied': float(depth_correction_cap_applied),
            'uncertainty_metric': uncertainty_metric if child.visit_count > 0 else (
                'depth_corrected_aggregate_q_std'
                if self.depth_disagreement_stats.enabled else 'aggregate_calibrated_q_std'
            ),
            'normalized_uncertainty': float(normalized_uncertainty),
            'q_min': float(q_min),
            'q_max': float(q_max),
            'q_range': float(q_range),
            'q_normalization_source': 'mixed' if mixed_reward_mode else 'extrinsic',
            'mode': mode,
        }

    def _root_depth_matched_q_diagnostics(
            self,
            root: EnsembleSearchNode,
    ) -> Dict[int, Dict[str, Any]]:
        enabled = bool(self._cfg.get('depth_matched_qstd_diagnostics', False))
        max_depth = max(1, int(self._cfg.get('depth_matched_qstd_max_depth', 5)))
        children = [root.children[int(action)] for action in root.legal_actions]
        common_depths = []
        if enabled and children:
            common_depths = [
                depth for depth in range(1, max_depth + 1)
                if all(child.visit_count_by_search_depth.get(depth, 0) > 0 for child in children)
            ]

        diagnostics: Dict[int, Dict[str, Any]] = {}
        for action in root.legal_actions:
            child = root.children[int(action)]
            action_diagnostics: Dict[str, Any] = {
                'depth_matched_qstd_diagnostics_enabled': float(enabled),
                'depth_matched_qstd_max_depth': int(max_depth),
                'depth_matched_common_depth_count': int(len(common_depths)),
                'depth_matched_common_depths': list(common_depths),
                'depth_matched_available': float(bool(common_depths)),
                'depth_normalized_backup_disagreement_enabled': float(
                    self.depth_disagreement_stats.enabled
                ),
            }

            for depth in range(1, max_depth + 1):
                count = int(child.visit_count_by_search_depth.get(depth, 0))
                raw_values = child.search_depth_q_mean(depth, calibrated=False)
                calibrated_values = child.search_depth_q_mean(depth, calibrated=True)
                if raw_values is None:
                    raw_values = np.zeros(child.ensemble_size, dtype=np.float32)
                if calibrated_values is None:
                    calibrated_values = np.zeros(child.ensemble_size, dtype=np.float32)
                baseline = self.depth_disagreement_stats.baseline_snapshot(depth)
                action_diagnostics.update(
                    {
                        f'depth_{depth}_return_count': count,
                        f'depth_{depth}_mean_q': float(np.mean(raw_values)) if count > 0 else 0.0,
                        f'depth_{depth}_raw_q_std': float(np.std(raw_values)) if count > 0 else 0.0,
                        f'depth_{depth}_calibrated_q_std': (
                            float(np.std(calibrated_values)) if count > 0 else 0.0
                        ),
                        f'depth_{depth}_q_values': raw_values.astype(np.float32).tolist(),
                        f'depth_{depth}_calibrated_q_values': (
                            calibrated_values.astype(np.float32).tolist()
                        ),
                        f'depth_{depth}_raw_backup_disagreement_mean': (
                            child.search_depth_backup_disagreement(depth, 'raw_mean')
                        ),
                        f'depth_{depth}_raw_backup_disagreement_rms': (
                            child.search_depth_backup_disagreement(depth, 'raw_rms')
                        ),
                        f'depth_{depth}_calibrated_backup_disagreement_mean': (
                            child.search_depth_backup_disagreement(depth, 'calibrated_mean')
                        ),
                        f'depth_{depth}_calibrated_backup_disagreement_rms': (
                            child.search_depth_backup_disagreement(depth, 'calibrated_rms')
                        ),
                        f'depth_{depth}_adjusted_backup_disagreement_mean': (
                            child.search_depth_backup_disagreement(depth, 'adjusted_mean')
                        ),
                        f'depth_{depth}_adjusted_backup_disagreement_rms': (
                            child.search_depth_backup_disagreement(depth, 'adjusted_rms')
                        ),
                        f'depth_{depth}_applied_disagreement_scale': (
                            child.search_depth_backup_disagreement(depth, 'scale_mean')
                        ),
                        f'depth_{depth}_running_calibrated_disagreement_baseline': float(
                            baseline['calibrated_baseline']
                        ),
                        f'depth_{depth}_effective_disagreement_baseline': float(
                            baseline['effective_calibrated_baseline']
                        ),
                        f'depth_{depth}_running_disagreement_ratio': float(
                            baseline['calibrated_ratio']
                        ),
                        f'depth_{depth}_effective_disagreement_ratio': float(
                            baseline['effective_calibrated_ratio']
                        ),
                        f'depth_{depth}_next_phase_disagreement_scale': float(baseline['scale']),
                        f'depth_{depth}_baseline_group_count': int(baseline['total_group_count']),
                    }
                )

            if common_depths:
                raw_depth_means = np.stack(
                    [child.search_depth_q_mean(depth, calibrated=False) for depth in common_depths],
                    axis=0,
                )
                calibrated_depth_means = np.stack(
                    [child.search_depth_q_mean(depth, calibrated=True) for depth in common_depths],
                    axis=0,
                )
                raw_matched_values = np.mean(raw_depth_means, axis=0)
                calibrated_matched_values = np.mean(calibrated_depth_means, axis=0)
            else:
                raw_matched_values = np.zeros(child.ensemble_size, dtype=np.float32)
                calibrated_matched_values = np.zeros(child.ensemble_size, dtype=np.float32)

            calibrated_q_std = float(np.std(calibrated_matched_values)) if common_depths else 0.0
            q_min_max_stats = root.q_min_max_stats
            normalized_q_std = (
                q_min_max_stats.normalize_std(calibrated_q_std)
                if q_min_max_stats is not None and common_depths else calibrated_q_std
            )
            action_diagnostics.update(
                {
                    'depth_matched_mean_q': (
                        float(np.mean(raw_matched_values)) if common_depths else 0.0
                    ),
                    'depth_matched_raw_q_std': (
                        float(np.std(raw_matched_values)) if common_depths else 0.0
                    ),
                    'depth_matched_calibrated_q_std': calibrated_q_std,
                    'depth_matched_normalized_q_std': float(normalized_q_std),
                    'depth_matched_q_values': raw_matched_values.astype(np.float32).tolist(),
                    'depth_matched_calibrated_q_values': (
                        calibrated_matched_values.astype(np.float32).tolist()
                    ),
                }
            )
            diagnostics[int(action)] = action_diagnostics
        return diagnostics

    def _root_action_diagnostics(
            self,
            root: EnsembleSearchNode,
            mode: str,
            ensemble_ucb_beta: float,
            ensemble_uncertainty_decay: bool,
            intrinsic_reward_weight: float,
            intrinsic_min_max_stats: Optional[IntrinsicMinMaxStats] = None,
    ) -> List[Dict[str, Any]]:
        diagnostics = []
        depth_matched_diagnostics = self._root_depth_matched_q_diagnostics(root)
        root_forced_coverage_complete = bool(
            root.legal_actions
            and all(root.children[int(action)].visit_count > 0 for action in root.legal_actions)
        )
        for action in root.legal_actions:
            child = root.children[int(action)]
            components = self._score_child_components(
                root,
                child,
                mode,
                ensemble_ucb_beta,
                ensemble_uncertainty_decay,
                intrinsic_reward_weight,
                intrinsic_min_max_stats,
            )
            q_values = child.q_mean.astype(np.float32).tolist() if child.visit_count > 0 else [
                0.0 for _ in range(child.ensemble_size)
            ]
            mixed_q_values = child.mixed_q_mean.astype(np.float32).tolist() if child.visit_count > 0 else [
                0.0 for _ in range(child.ensemble_size)
            ]
            calibrated_q_values = (
                child.calibrated_q_mean.astype(np.float32).tolist() if child.visit_count > 0 else [
                    0.0 for _ in range(child.ensemble_size)
                ]
            )
            predicted_reward_return_values = (
                child.predicted_reward_return_mean.astype(np.float32).tolist() if child.visit_count > 0 else [
                    0.0 for _ in range(child.ensemble_size)
                ]
            )
            leaf_value_contribution_values = (
                child.leaf_value_contribution_mean.astype(np.float32).tolist() if child.visit_count > 0 else [
                    0.0 for _ in range(child.ensemble_size)
                ]
            )
            calibrated_leaf_value_contribution_values = (
                child.calibrated_leaf_value_contribution_mean.astype(np.float32).tolist()
                if child.visit_count > 0 else [0.0 for _ in range(child.ensemble_size)]
            )
            aggregate_calibrated_q_std = float(components['aggregate_calibrated_uncertainty'])
            aggregate_normalized_q_std = (
                root.q_min_max_stats.normalize_std(aggregate_calibrated_q_std)
                if root.q_min_max_stats is not None else aggregate_calibrated_q_std
            )
            calibrated_backup_rms = float(child.calibrated_backup_disagreement_rms)
            cancellation_disagreement = math.sqrt(
                max(0.0, calibrated_backup_rms ** 2 - aggregate_calibrated_q_std ** 2)
            )
            depth_corrected_residual_values = child.depth_corrected_q_residual(
                calibrated=True
            ).astype(np.float32).tolist()
            diagnostics.append(
                {
                    'action': int(action),
                    'search_mode': mode,
                    'policy_prior_mode': str(getattr(root, 'policy_prior_mode', 'learned')),
                    'force_root_action_coverage': float(
                        bool(getattr(root, 'force_root_action_coverage', False))
                    ),
                    'root_forced_coverage_complete': float(root_forced_coverage_complete),
                    'visit_count': int(child.visit_count),
                    'prior': float(child.prior),
                    'mean_q': components['mean_q'],
                    'mixed_mean_q': components['mixed_mean_q'],
                    'score_mean_q': components['mean_q_for_score'],
                    'normalized_mean_q': components['normalized_mean_q'],
                    'unbounded_ensemble_q': components['unbounded_ensemble_q'],
                    'bounded_ensemble_q': components['bounded_ensemble_q'],
                    'normalized_ensemble_q': components['normalized_ensemble_q'],
                    'ensemble_envelope_q': components['ensemble_envelope_q'],
                    'ensemble_envelope_gap_from_mean': components['ensemble_envelope_gap_from_mean'],
                    'ensemble_envelope_member_id': components['ensemble_envelope_member_id'],
                    'ensemble_envelope_cap_applied': components['ensemble_envelope_cap_applied'],
                    'ensemble_envelope_cap_adjustment': components['ensemble_envelope_cap_adjustment'],
                    'bounded_ensemble_mode': components['bounded_ensemble_mode'],
                    'q_std': components['uncertainty'],
                    'raw_q_std': components['raw_uncertainty'],
                    'calibrated_q_std': aggregate_calibrated_q_std,
                    'normalized_q_std': components['normalized_uncertainty'],
                    'uncertainty_metric': components['uncertainty_metric'],
                    'score_q_std': components['uncertainty'],
                    'aggregate_raw_q_std': components['raw_uncertainty'],
                    'aggregate_calibrated_q_std': aggregate_calibrated_q_std,
                    'aggregate_normalized_q_std': float(aggregate_normalized_q_std),
                    'raw_backup_disagreement_mean': float(child.raw_backup_disagreement_mean),
                    'raw_backup_disagreement_rms': float(child.raw_backup_disagreement_rms),
                    'calibrated_backup_disagreement_mean': float(
                        child.calibrated_backup_disagreement_mean
                    ),
                    'calibrated_backup_disagreement_rms': calibrated_backup_rms,
                    'depth_normalized_backup_disagreement_mean': float(
                        child.depth_normalized_backup_disagreement_mean
                    ),
                    'depth_normalized_backup_disagreement_rms': float(
                        child.depth_normalized_backup_disagreement_rms
                    ),
                    'depth_corrected_aggregate_q_std_uncapped': float(
                        components['depth_corrected_uncertainty_uncapped']
                    ),
                    'depth_corrected_aggregate_q_std': float(
                        components['depth_corrected_uncertainty']
                    ),
                    'depth_corrected_q_residual_values': depth_corrected_residual_values,
                    'depth_correction_cap_applied': float(
                        components['depth_correction_cap_applied']
                    ),
                    'depth_correction_ratio': float(
                        components['depth_corrected_uncertainty']
                        / max(1e-8, aggregate_calibrated_q_std)
                        if child.visit_count > 0 else 0.0
                    ),
                    'backup_disagreement_cancellation': float(cancellation_disagreement),
                    'backup_disagreement_cancellation_ratio': float(
                        aggregate_calibrated_q_std / max(1e-8, calibrated_backup_rms)
                        if child.visit_count > 0 else 0.0
                    ),
                    'applied_depth_disagreement_scale_mean': float(
                        child.depth_disagreement_scale_mean
                    ),
                    'q_min': components['q_min'],
                    'q_max': components['q_max'],
                    'q_range': components['q_range'],
                    'q_normalization_source': components['q_normalization_source'],
                    'predicted_reward_return_mean': float(np.mean(predicted_reward_return_values)),
                    'predicted_reward_return_std': float(np.std(predicted_reward_return_values)),
                    'leaf_value_contribution_mean': float(np.mean(leaf_value_contribution_values)),
                    'leaf_value_contribution_std': float(np.std(leaf_value_contribution_values)),
                    'calibrated_leaf_value_contribution_mean': float(
                        np.mean(calibrated_leaf_value_contribution_values)
                    ),
                    'calibrated_leaf_value_contribution_std': float(
                        np.std(calibrated_leaf_value_contribution_values)
                    ),
                    'prior_score': components['prior_score'],
                    'uncertainty_score': components['uncertainty_score'],
                    'unbounded_uncertainty_score': components['unbounded_uncertainty_score'],
                    'intrinsic_score': components['intrinsic_score'],
                    'intrinsic_value': float(child.intrinsic_value),
                    'normalized_intrinsic_value': components['normalized_intrinsic_value'],
                    'intrinsic_immediate_reward': float(child.intrinsic_reward),
                    'intrinsic_landscape_query_reward': float(
                        child.intrinsic_landscape_query_reward
                    ),
                    'intrinsic_reward_before_path_transform': float(
                        child.intrinsic_reward_before_path_transform
                    ),
                    'intrinsic_reward_before_path_modifier': float(
                        child.intrinsic_reward_before_path_modifier
                    ),
                    'intrinsic_path_reward_modifier': float(
                        child.intrinsic_path_reward_modifier
                    ),
                    'intrinsic_reward_before_path_bonus': float(
                        child.intrinsic_reward_before_path_bonus
                    ),
                    'intrinsic_path_reward_bonus': float(
                        child.intrinsic_path_reward_bonus
                    ),
                    'dynamics_disagreement_latent_reward': float(child.dynamics_disagreement_latent_reward),
                    'dynamics_disagreement_projection_reward': float(
                        child.dynamics_disagreement_projection_reward
                    ),
                    'dynamics_disagreement_prediction_projection_reward': float(
                        child.dynamics_disagreement_prediction_projection_reward
                    ),
                    'intrinsic_reward_return_mean': float(child.intrinsic_reward_return_mean),
                    'intrinsic_reward_weighted_return_mean': float(child.intrinsic_reward_weighted_return_mean),
                    'intrinsic_leaf_value_raw_contribution_mean': float(
                        child.intrinsic_leaf_value_raw_contribution_mean
                    ),
                    'intrinsic_leaf_value_clipped_contribution_mean': float(
                        child.intrinsic_leaf_value_clipped_contribution_mean
                    ),
                    'intrinsic_leaf_value_contribution_mean': float(
                        child.intrinsic_leaf_value_contribution_mean
                    ),
                    'intrinsic_leaf_value_before_modifier_contribution_mean': float(
                        child.intrinsic_leaf_value_before_modifier_contribution_mean
                    ),
                    'intrinsic_leaf_value_modifier_mean': float(
                        child.intrinsic_leaf_value_modifier_mean
                    ),
                    'intrinsic_immediate_reward_weight': float(
                        self._cfg.get('intrinsic_tree_immediate_reward_weight', 1.0)
                    ),
                    'intrinsic_leaf_value_weight': float(self._cfg.get('intrinsic_tree_leaf_value_weight', 1.0)),
                    'intrinsic_tree_leaf_value_source': str(
                        self._cfg.get('intrinsic_tree_leaf_value_source', 'default') or 'default'
                    ),
                    'dynamics_disagreement_space': str(
                        self._cfg.get('dynamics_disagreement_space', 'latent') or 'latent'
                    ),
                    'imagined_node_latent_novelty': float(child.latent_novelty),
                    'imagined_leaf_latent_novelty_mean': float(child.leaf_latent_novelty_mean),
                    'imagined_leaf_latent_novelty_max': float(child.leaf_latent_novelty_max),
                    'imagined_leaf_latent_novelty_depth_mean': float(child.leaf_latent_novelty_depth_mean),
                    'imagined_leaf_latent_novelty_max_depth': int(child.leaf_latent_novelty_max_depth),
                    'imagined_leaf_latent_novelty_max_path': child.leaf_latent_novelty_max_path,
                    'total_score': components['total_score'],
                    'mean_only_total_score': (
                        components['normalized_mean_q'] + components['prior_score'] + components['intrinsic_score']
                    ),
                    'q_values': q_values,
                    'mixed_q_values': mixed_q_values,
                    'calibrated_q_values': calibrated_q_values,
                    'predicted_reward_return_values': predicted_reward_return_values,
                    'leaf_value_contribution_values': leaf_value_contribution_values,
                    'calibrated_leaf_value_contribution_values': calibrated_leaf_value_contribution_values,
                    **depth_matched_diagnostics[int(action)],
                }
            )
        if diagnostics:
            bounded_best_action = int(max(diagnostics, key=lambda item: item['total_score'])['action'])
            mean_best_action = int(max(diagnostics, key=lambda item: item['mean_only_total_score'])['action'])
            envelope_member_ids = [
                int(item['ensemble_envelope_member_id'])
                for item in diagnostics
                if int(item['ensemble_envelope_member_id']) >= 0
            ]
            same_envelope_member_all_actions = float(
                len(envelope_member_ids) == len(diagnostics) and len(set(envelope_member_ids)) == 1
            )
            depth_matched_available = all(
                float(item['depth_matched_available']) > 0.5 for item in diagnostics
            )
            depth_matched_scores = []
            aggregate_qstd_scores = []
            for item in diagnostics:
                uncertainty_sign = -1.0 if mode == 'pessimistic_shared_tree' else 1.0
                matched_uncertainty_score = 0.0
                if mode in ['optimistic_shared_tree', 'pessimistic_shared_tree'] and depth_matched_available:
                    matched_uncertainty_score = (
                        uncertainty_sign
                        * ensemble_ucb_beta
                        * float(item['depth_matched_normalized_q_std'])
                    )
                    if ensemble_uncertainty_decay:
                        matched_uncertainty_score /= math.sqrt(1 + int(item['visit_count']))
                item['depth_matched_counterfactual_uncertainty_score'] = float(
                    matched_uncertainty_score
                )
                counterfactual_total_score = float(item['total_score'])
                if depth_matched_available and mode in [
                        'optimistic_shared_tree', 'pessimistic_shared_tree'
                ]:
                    counterfactual_total_score += (
                        matched_uncertainty_score - float(item['uncertainty_score'])
                    )
                item['depth_matched_counterfactual_total_score'] = counterfactual_total_score
                depth_matched_scores.append(counterfactual_total_score)
                aggregate_uncertainty_score = 0.0
                if mode in ['optimistic_shared_tree', 'pessimistic_shared_tree']:
                    aggregate_uncertainty_score = (
                        uncertainty_sign
                        * ensemble_ucb_beta
                        * float(item['aggregate_normalized_q_std'])
                    )
                    if ensemble_uncertainty_decay:
                        aggregate_uncertainty_score /= math.sqrt(1 + int(item['visit_count']))
                aggregate_total_score = float(item['total_score'])
                if mode in ['optimistic_shared_tree', 'pessimistic_shared_tree']:
                    aggregate_total_score += (
                        aggregate_uncertainty_score - float(item['uncertainty_score'])
                    )
                item['aggregate_qstd_counterfactual_uncertainty_score'] = float(
                    aggregate_uncertainty_score
                )
                item['aggregate_qstd_counterfactual_total_score'] = float(aggregate_total_score)
                aggregate_qstd_scores.append(aggregate_total_score)
            current_best_action = int(max(diagnostics, key=lambda item: item['total_score'])['action'])
            depth_matched_best_action = int(
                diagnostics[int(np.argmax(np.asarray(depth_matched_scores, dtype=np.float64)))]['action']
            )
            aggregate_qstd_best_action = int(
                diagnostics[int(np.argmax(np.asarray(aggregate_qstd_scores, dtype=np.float64)))]['action']
            )
            for item in diagnostics:
                item.update(
                    {
                        'bounded_score_best_action': bounded_best_action,
                        'mean_score_best_action': mean_best_action,
                        'bounded_changed_mean_score_argmax': float(bounded_best_action != mean_best_action),
                        'same_envelope_member_all_actions': same_envelope_member_all_actions,
                        'depth_matched_current_score_best_action': current_best_action,
                        'depth_matched_counterfactual_score_best_action': depth_matched_best_action,
                        'depth_matched_counterfactual_changed_argmax': float(
                            depth_matched_available and current_best_action != depth_matched_best_action
                        ),
                        'aggregate_qstd_counterfactual_score_best_action': aggregate_qstd_best_action,
                        'aggregate_qstd_counterfactual_changed_argmax': float(
                            current_best_action != aggregate_qstd_best_action
                        ),
                    }
                )
        return diagnostics

    def _expand_selected(
            self,
            model: torch.nn.Module,
            selected: List[Dict[str, Any]],
            ensemble_size: int,
            intrinsic_reward_fn: Optional[Callable[[torch.Tensor], torch.Tensor]],
            use_model_intrinsic: bool,
            intrinsic_reward_source: str,
            policy_prior_mode: str = "learned",
            intrinsic_path_reward_fn: Optional[
                Callable[[Sequence[int], Sequence[Sequence[int]], torch.Tensor], np.ndarray]
            ] = None,
            intrinsic_path_reward_transform_fn: Optional[Callable] = None,
            intrinsic_path_reward_modifier_fn: Optional[
                Callable[[Sequence[int], Sequence[Sequence[int]], torch.Tensor], np.ndarray]
            ] = None,
            intrinsic_path_reward_bonus_fn: Optional[
                Callable[[Sequence[int], Sequence[Sequence[int]], torch.Tensor], np.ndarray]
            ] = None,
            intrinsic_leaf_value_modifier_fn: Optional[
                Callable[[Sequence[int], Sequence[Sequence[int]], torch.Tensor], np.ndarray]
            ] = None,
    ) -> None:
        device = selected[0]['parent'].latent_states.device
        parent_latents_by_member = []
        actions_by_member = []
        member_ids = []
        for member_id in range(ensemble_size):
            parent_latents_by_member.append(
                torch.stack([selection['parent'].latent_states[member_id] for selection in selected], dim=0)
            )
            actions_by_member.append(
                torch.tensor([selection['action'] for selection in selected], dtype=torch.long, device=device)
            )
            member_ids.extend([member_id for _ in selected])

        latent_batch = torch.cat(parent_latents_by_member, dim=0)
        action_batch = torch.cat(actions_by_member, dim=0)
        network_output = model.recurrent_inference(latent_batch, action_batch, member_id=member_ids)

        next_latent_flat = network_output.latent_state.detach()
        reward = self.reward_inverse_scalar_transform_handle(network_output.reward).detach()
        value = self.value_inverse_scalar_transform_handle(network_output.value).detach()
        policy_logits = network_output.policy_logits.detach()

        batch_size = len(selected)
        next_latent = next_latent_flat.reshape(
            ensemble_size, batch_size, *next_latent_flat.shape[1:]
        ).transpose(0, 1)
        latent_novelty_np = self._latent_novelty_np(
            model=model,
            flat_latent_states=next_latent_flat,
            member_ids=member_ids,
            batch_size=batch_size,
            ensemble_size=ensemble_size,
        )
        intrinsic_reward_np, intrinsic_leaf_value_np = self._intrinsic_reward_value_np(
            model=model,
            flat_latent_states=next_latent_flat,
            member_ids=member_ids,
            batch_size=batch_size,
            ensemble_size=ensemble_size,
            intrinsic_reward_fn=intrinsic_reward_fn,
            use_model_intrinsic=use_model_intrinsic,
            intrinsic_reward_source=intrinsic_reward_source,
        )
        landscape_query_reward_np = None
        if intrinsic_path_reward_fn is not None:
            landscape_query_reward_np = np.asarray(
                intrinsic_path_reward_fn(
                    [int(selection['root_index']) for selection in selected],
                    [selection['path_actions'] for selection in selected],
                    next_latent,
                ),
                dtype=np.float32,
            ).reshape(-1)
            if landscape_query_reward_np.shape != (batch_size, ):
                raise ValueError(
                    "intrinsic_path_reward_fn must return one scalar per selected path; "
                    f"got {landscape_query_reward_np.shape} for batch size {batch_size}."
                )
            if not bool(np.all(np.isfinite(landscape_query_reward_np))):
                raise ValueError("intrinsic_path_reward_fn returned non-finite rewards.")
            intrinsic_reward_np = landscape_query_reward_np
        intrinsic_reward_before_path_transform_np = intrinsic_reward_np.copy()
        if intrinsic_path_reward_transform_fn is not None:
            transformed_reward_np = np.asarray(
                intrinsic_path_reward_transform_fn(
                    [int(selection['root_index']) for selection in selected],
                    [selection['path_actions'] for selection in selected],
                    next_latent,
                    intrinsic_reward_before_path_transform_np.copy(),
                ),
                dtype=np.float32,
            ).reshape(-1)
            if transformed_reward_np.shape != (batch_size, ):
                raise ValueError(
                    "intrinsic_path_reward_transform_fn must return one scalar per "
                    f"selected path; got {transformed_reward_np.shape} for batch "
                    f"size {batch_size}."
                )
            if not bool(np.all(np.isfinite(transformed_reward_np))):
                raise ValueError(
                    "intrinsic_path_reward_transform_fn returned non-finite rewards."
                )
            if bool(np.any(transformed_reward_np < 0.0)):
                raise ValueError(
                    "intrinsic_path_reward_transform_fn returned a negative reward."
                )
            intrinsic_reward_np = transformed_reward_np
        intrinsic_reward_before_path_modifier_np = intrinsic_reward_np.copy()
        intrinsic_path_reward_modifier_np = np.ones(batch_size, dtype=np.float32)
        if intrinsic_path_reward_modifier_fn is not None:
            intrinsic_path_reward_modifier_np = np.asarray(
                intrinsic_path_reward_modifier_fn(
                    [int(selection['root_index']) for selection in selected],
                    [selection['path_actions'] for selection in selected],
                    next_latent,
                ),
                dtype=np.float32,
            ).reshape(-1)
            if intrinsic_path_reward_modifier_np.shape != (batch_size, ):
                raise ValueError(
                    "intrinsic_path_reward_modifier_fn must return one scalar per selected path; "
                    f"got {intrinsic_path_reward_modifier_np.shape} for batch size {batch_size}."
                )
            if not bool(np.all(np.isfinite(intrinsic_path_reward_modifier_np))):
                raise ValueError("intrinsic_path_reward_modifier_fn returned non-finite factors.")
            if bool(np.any(intrinsic_path_reward_modifier_np < 0.0)):
                raise ValueError("intrinsic_path_reward_modifier_fn returned a negative factor.")
            intrinsic_reward_np = (
                intrinsic_reward_np * intrinsic_path_reward_modifier_np
            )
        intrinsic_reward_before_path_bonus_np = intrinsic_reward_np.copy()
        intrinsic_path_reward_bonus_np = np.zeros(batch_size, dtype=np.float32)
        if intrinsic_path_reward_bonus_fn is not None:
            intrinsic_path_reward_bonus_np = np.asarray(
                intrinsic_path_reward_bonus_fn(
                    [int(selection['root_index']) for selection in selected],
                    [selection['path_actions'] for selection in selected],
                    next_latent,
                ),
                dtype=np.float32,
            ).reshape(-1)
            if intrinsic_path_reward_bonus_np.shape != (batch_size, ):
                raise ValueError(
                    "intrinsic_path_reward_bonus_fn must return one scalar per selected path; "
                    f"got {intrinsic_path_reward_bonus_np.shape} for batch size {batch_size}."
                )
            if not bool(np.all(np.isfinite(intrinsic_path_reward_bonus_np))):
                raise ValueError("intrinsic_path_reward_bonus_fn returned non-finite bonuses.")
            if bool(np.any(intrinsic_path_reward_bonus_np < 0.0)):
                raise ValueError("intrinsic_path_reward_bonus_fn returned a negative bonus.")
            intrinsic_reward_np = intrinsic_reward_np + intrinsic_path_reward_bonus_np
        intrinsic_leaf_value_before_modifier_np = intrinsic_leaf_value_np.copy()
        intrinsic_leaf_value_modifier_np = np.ones(batch_size, dtype=np.float32)
        if intrinsic_leaf_value_modifier_fn is not None:
            intrinsic_leaf_value_modifier_np = np.asarray(
                intrinsic_leaf_value_modifier_fn(
                    [int(selection['root_index']) for selection in selected],
                    [selection['path_actions'] for selection in selected],
                    next_latent,
                ),
                dtype=np.float32,
            ).reshape(-1)
            if intrinsic_leaf_value_modifier_np.shape != (batch_size, ):
                raise ValueError(
                    "intrinsic_leaf_value_modifier_fn must return one scalar per selected path; "
                    f"got {intrinsic_leaf_value_modifier_np.shape} for batch size {batch_size}."
                )
            if not bool(np.all(np.isfinite(intrinsic_leaf_value_modifier_np))):
                raise ValueError("intrinsic_leaf_value_modifier_fn returned non-finite factors.")
            if bool(np.any(intrinsic_leaf_value_modifier_np < 0.0)):
                raise ValueError("intrinsic_leaf_value_modifier_fn returned a negative factor.")
            intrinsic_leaf_value_np = (
                intrinsic_leaf_value_np * intrinsic_leaf_value_modifier_np
            )
        dynamics_disagreement_np = self._dynamics_disagreement_diagnostics_np(
            model=model,
            flat_latent_states=next_latent_flat,
            member_ids=member_ids,
            batch_size=batch_size,
            ensemble_size=ensemble_size,
        )

        reward = reward.reshape(ensemble_size, batch_size, -1).transpose(0, 1).squeeze(-1)
        value = value.reshape(ensemble_size, batch_size, -1).transpose(0, 1).squeeze(-1)
        policy_logits = policy_logits.reshape(ensemble_size, batch_size, -1).transpose(0, 1)

        reward_np = self._to_numpy(reward)
        value_np = self._to_numpy(value)
        for batch_idx, selection in enumerate(selected):
            child = selection['child']
            child.latent_states = next_latent[batch_idx]
            child.reward = reward_np[batch_idx].astype(np.float32)
            child.intrinsic_reward = float(intrinsic_reward_np[batch_idx])
            child.intrinsic_landscape_query_reward = (
                float(landscape_query_reward_np[batch_idx])
                if landscape_query_reward_np is not None else 0.0
            )
            child.intrinsic_reward_before_path_transform = float(
                intrinsic_reward_before_path_transform_np[batch_idx]
            )
            child.intrinsic_reward_before_path_modifier = float(
                intrinsic_reward_before_path_modifier_np[batch_idx]
            )
            child.intrinsic_path_reward_modifier = float(
                intrinsic_path_reward_modifier_np[batch_idx]
            )
            child.intrinsic_reward_before_path_bonus = float(
                intrinsic_reward_before_path_bonus_np[batch_idx]
            )
            child.intrinsic_path_reward_bonus = float(
                intrinsic_path_reward_bonus_np[batch_idx]
            )
            child.dynamics_disagreement_latent_reward = float(dynamics_disagreement_np['latent'][batch_idx])
            child.dynamics_disagreement_projection_reward = float(dynamics_disagreement_np['projection'][batch_idx])
            child.dynamics_disagreement_prediction_projection_reward = float(
                dynamics_disagreement_np['prediction_projection'][batch_idx]
            )
            child.latent_novelty = float(latent_novelty_np[batch_idx])
            child.expand(
                self._action_priors(
                    policy_logits[batch_idx], child.legal_actions, policy_prior_mode
                )
            )
            selection['leaf_values'] = value_np[batch_idx].astype(np.float32)
            selection['leaf_intrinsic_value'] = float(intrinsic_leaf_value_np[batch_idx])
            selection['leaf_intrinsic_value_before_modifier'] = float(
                intrinsic_leaf_value_before_modifier_np[batch_idx]
            )
            selection['leaf_intrinsic_value_modifier'] = float(
                intrinsic_leaf_value_modifier_np[batch_idx]
            )
            selection['leaf_latent_novelty'] = float(latent_novelty_np[batch_idx])
            selection['leaf_depth'] = len(selection.get('path', []))

    def _backup(
            self,
            root: EnsembleSearchNode,
            path: List[EnsembleSearchNode],
            leaf_values: np.ndarray,
            leaf_intrinsic_value: float = 0.0,
            leaf_latent_novelty: float = 0.0,
            leaf_depth: int = 0,
            leaf_path_actions: Optional[Sequence[int]] = None,
            intrinsic_min_max_stats: Optional[IntrinsicMinMaxStats] = None,
            value_calibration_offsets: Optional[np.ndarray] = None,
            mixed_reward_backup: bool = False,
            intrinsic_reward_weight: float = 0.0,
            leaf_intrinsic_value_before_modifier: Optional[float] = None,
            leaf_intrinsic_value_modifier: float = 1.0,
    ) -> Dict[str, Any]:
        returns = leaf_values.astype(np.float32)
        if value_calibration_offsets is None:
            value_calibration_offsets = np.zeros_like(returns)
        else:
            value_calibration_offsets = np.asarray(value_calibration_offsets, dtype=np.float32)
            if value_calibration_offsets.shape != returns.shape:
                raise ValueError(
                    f"Value calibration shape {value_calibration_offsets.shape} does not match leaf values "
                    f"shape {returns.shape}."
                )
        calibrated_leaf_values = returns - value_calibration_offsets
        calibrated_returns = calibrated_leaf_values.copy()
        intrinsic_discount_factor = self._cfg.get('rnd_intrinsic_discount_factor', None)
        if intrinsic_discount_factor is None:
            intrinsic_discount_factor = self._cfg.discount_factor
        intrinsic_discount_factor = float(intrinsic_discount_factor)
        discount_factor = float(self._cfg.discount_factor)
        predicted_reward_returns = np.zeros_like(returns)
        leaf_value_contributions = returns.copy()
        calibrated_leaf_value_contributions = calibrated_returns.copy()
        intrinsic_immediate_weight = float(self._cfg.get('intrinsic_tree_immediate_reward_weight', 1.0))
        intrinsic_leaf_value_weight = float(self._cfg.get('intrinsic_tree_leaf_value_weight', 1.0))
        if leaf_intrinsic_value_before_modifier is None:
            leaf_intrinsic_value_before_modifier = float(leaf_intrinsic_value)
        leaf_intrinsic_value_before_modifier = float(
            leaf_intrinsic_value_before_modifier
        )
        leaf_intrinsic_value_modifier = float(leaf_intrinsic_value_modifier)
        leaf_value_clip = self._cfg.get('intrinsic_tree_leaf_value_clip', None)
        intrinsic_reward_returns = 0.0
        intrinsic_reward_weighted_returns = 0.0
        intrinsic_leaf_value_raw_contribution = float(leaf_intrinsic_value)
        intrinsic_leaf_value_before_modifier_contribution = float(
            leaf_intrinsic_value_before_modifier
        )
        intrinsic_leaf_value_clipped_contribution = float(leaf_intrinsic_value)
        if leaf_value_clip is not None:
            clip_value = abs(float(leaf_value_clip))
            intrinsic_leaf_value_clipped_contribution = float(
                np.clip(intrinsic_leaf_value_clipped_contribution, -clip_value, clip_value)
            )
        leaf_intrinsic_value_clipped = float(intrinsic_leaf_value_clipped_contribution)
        intrinsic_leaf_value_contribution = (
            intrinsic_leaf_value_weight * intrinsic_leaf_value_clipped_contribution
        )
        intrinsic_return = intrinsic_leaf_value_contribution
        intrinsic_reward_weight = float(intrinsic_reward_weight)
        mixed_returns = returns + intrinsic_reward_weight * intrinsic_return
        leaf_latent_novelty = float(leaf_latent_novelty)
        leaf_depth = int(leaf_depth)
        if leaf_depth <= 0 and path:
            leaf_depth = len(path)
        leaf_path_actions = [int(action) for action in (leaf_path_actions or [])]
        path_intrinsic_rewards = [float(node.intrinsic_reward) for node in path]
        path_predicted_reward_means = [float(np.mean(node.reward)) for node in path]
        for backed_up_depth, node in enumerate(reversed(path), start=1):
            predicted_reward_returns = node.reward + discount_factor * predicted_reward_returns
            leaf_value_contributions = discount_factor * leaf_value_contributions
            calibrated_leaf_value_contributions = discount_factor * calibrated_leaf_value_contributions
            returns = predicted_reward_returns + leaf_value_contributions
            calibrated_returns = predicted_reward_returns + calibrated_leaf_value_contributions
            intrinsic_reward_returns = node.intrinsic_reward + intrinsic_discount_factor * intrinsic_reward_returns
            intrinsic_reward_weighted_returns = (
                intrinsic_immediate_weight * node.intrinsic_reward
                + intrinsic_discount_factor * intrinsic_reward_weighted_returns
            )
            intrinsic_leaf_value_raw_contribution = (
                intrinsic_discount_factor * intrinsic_leaf_value_raw_contribution
            )
            intrinsic_leaf_value_before_modifier_contribution = (
                intrinsic_discount_factor
                * intrinsic_leaf_value_before_modifier_contribution
            )
            intrinsic_leaf_value_clipped_contribution = (
                intrinsic_discount_factor * intrinsic_leaf_value_clipped_contribution
            )
            intrinsic_leaf_value_contribution = intrinsic_discount_factor * intrinsic_leaf_value_contribution
            intrinsic_return = intrinsic_reward_weighted_returns + intrinsic_leaf_value_contribution
            mixed_returns = returns + intrinsic_reward_weight * intrinsic_return
            node.q_sum += returns
            node.mixed_q_sum += mixed_returns
            node.calibrated_q_sum += calibrated_returns
            node.has_calibrated_q = True
            node.predicted_reward_return_sum += predicted_reward_returns
            node.leaf_value_contribution_sum += leaf_value_contributions
            node.calibrated_leaf_value_contribution_sum += calibrated_leaf_value_contributions
            node.leaf_latent_novelty_sum += leaf_latent_novelty
            node.leaf_latent_novelty_count += 1
            node.leaf_latent_novelty_depth_sum += leaf_depth
            if leaf_latent_novelty > node.leaf_latent_novelty_max:
                node.leaf_latent_novelty_max = leaf_latent_novelty
                node.leaf_latent_novelty_max_depth = leaf_depth
                node.leaf_latent_novelty_max_path = list(leaf_path_actions)
            node.intrinsic_q_sum += intrinsic_return
            node.intrinsic_reward_return_sum += intrinsic_reward_returns
            node.intrinsic_reward_weighted_return_sum += intrinsic_reward_weighted_returns
            node.intrinsic_leaf_value_raw_contribution_sum += intrinsic_leaf_value_raw_contribution
            node.intrinsic_leaf_value_clipped_contribution_sum += intrinsic_leaf_value_clipped_contribution
            node.intrinsic_leaf_value_contribution_sum += intrinsic_leaf_value_contribution
            node.intrinsic_leaf_value_before_modifier_contribution_sum += (
                intrinsic_leaf_value_before_modifier_contribution
            )
            node.intrinsic_leaf_value_modifier_sum += leaf_intrinsic_value_modifier
            node.record_search_depth_return(
                backed_up_depth,
                returns,
                calibrated_returns,
                self.depth_disagreement_stats.scale(backed_up_depth),
            )
            if intrinsic_min_max_stats is not None:
                intrinsic_min_max_stats.update(intrinsic_return)
            node.visit_count += 1
            if node.q_min_max_stats is not None:
                node.q_min_max_stats.update(
                    node.mixed_q_mean if mixed_reward_backup else node.q_mean
                )
        root.q_sum += returns
        root.mixed_q_sum += mixed_returns
        root.calibrated_q_sum += calibrated_returns
        root.has_calibrated_q = True
        root.predicted_reward_return_sum += predicted_reward_returns
        root.leaf_value_contribution_sum += leaf_value_contributions
        root.calibrated_leaf_value_contribution_sum += calibrated_leaf_value_contributions
        root.leaf_latent_novelty_sum += leaf_latent_novelty
        root.leaf_latent_novelty_count += 1
        root.leaf_latent_novelty_depth_sum += leaf_depth
        if leaf_latent_novelty > root.leaf_latent_novelty_max:
            root.leaf_latent_novelty_max = leaf_latent_novelty
            root.leaf_latent_novelty_max_depth = leaf_depth
            root.leaf_latent_novelty_max_path = list(leaf_path_actions)
        root.intrinsic_q_sum += intrinsic_return
        root.intrinsic_reward_return_sum += intrinsic_reward_returns
        root.intrinsic_reward_weighted_return_sum += intrinsic_reward_weighted_returns
        root.intrinsic_leaf_value_raw_contribution_sum += intrinsic_leaf_value_raw_contribution
        root.intrinsic_leaf_value_clipped_contribution_sum += intrinsic_leaf_value_clipped_contribution
        root.intrinsic_leaf_value_contribution_sum += intrinsic_leaf_value_contribution
        root.intrinsic_leaf_value_before_modifier_contribution_sum += (
            intrinsic_leaf_value_before_modifier_contribution
        )
        root.intrinsic_leaf_value_modifier_sum += leaf_intrinsic_value_modifier
        if intrinsic_min_max_stats is not None:
            intrinsic_min_max_stats.update(intrinsic_return)
        root.visit_count += 1
        if root.q_min_max_stats is not None:
            root.q_min_max_stats.update(
                root.mixed_q_mean if mixed_reward_backup else root.q_mean
            )
        return {
            'path_actions': leaf_path_actions,
            'path_depth': leaf_depth,
            'path_intrinsic_rewards': path_intrinsic_rewards,
            'path_predicted_reward_means': path_predicted_reward_means,
            'leaf_value_mean': float(np.mean(leaf_values)),
            'leaf_value_std': float(np.std(leaf_values)),
            'leaf_values': leaf_values.astype(np.float32).tolist(),
            'value_calibration_offsets': value_calibration_offsets.astype(np.float32).tolist(),
            'calibrated_leaf_value_mean': float(np.mean(calibrated_leaf_values)),
            'calibrated_leaf_value_std': float(np.std(calibrated_leaf_values)),
            'calibrated_leaf_values': calibrated_leaf_values.astype(np.float32).tolist(),
            'leaf_intrinsic_value': float(leaf_intrinsic_value),
            'leaf_intrinsic_value_before_modifier': float(
                leaf_intrinsic_value_before_modifier
            ),
            'leaf_intrinsic_value_modifier': float(leaf_intrinsic_value_modifier),
            'leaf_intrinsic_value_clipped': float(leaf_intrinsic_value_clipped),
            'intrinsic_immediate_reward_weight': float(intrinsic_immediate_weight),
            'intrinsic_leaf_value_weight': float(intrinsic_leaf_value_weight),
            'leaf_latent_novelty': leaf_latent_novelty,
            'backed_up_mean_q': float(np.mean(returns)),
            'backed_up_q_std': float(np.std(returns)),
            'backed_up_q_values': returns.astype(np.float32).tolist(),
            'backed_up_calibrated_q_std': float(np.std(calibrated_returns)),
            'backed_up_calibrated_q_values': calibrated_returns.astype(np.float32).tolist(),
            'backed_up_depth_disagreement_scale': float(
                self.depth_disagreement_stats.scale(max(1, len(path)))
            ),
            'backed_up_depth_normalized_disagreement': float(
                np.std(calibrated_returns) * self.depth_disagreement_stats.scale(max(1, len(path)))
            ),
            'backed_up_predicted_reward_return_mean': float(np.mean(predicted_reward_returns)),
            'backed_up_leaf_value_contribution_mean': float(np.mean(leaf_value_contributions)),
            'backed_up_calibrated_leaf_value_contribution_mean': float(
                np.mean(calibrated_leaf_value_contributions)
            ),
            'backed_up_intrinsic_value': float(intrinsic_return),
            'intrinsic_reward_weight': float(intrinsic_reward_weight),
            'backed_up_mixed_q_mean': float(np.mean(mixed_returns)),
            'backed_up_mixed_q_values': mixed_returns.astype(np.float32).tolist(),
            'backed_up_intrinsic_reward_return': float(intrinsic_reward_returns),
            'backed_up_intrinsic_reward_weighted_return': float(intrinsic_reward_weighted_returns),
            'backed_up_intrinsic_leaf_value_raw_contribution': float(intrinsic_leaf_value_raw_contribution),
            'backed_up_intrinsic_leaf_value_clipped_contribution': float(
                intrinsic_leaf_value_clipped_contribution
            ),
            'backed_up_intrinsic_leaf_value_contribution': float(intrinsic_leaf_value_contribution),
        }

    def _tree_diagnostics(
            self,
            root: EnsembleSearchNode,
            mode: str,
            ensemble_ucb_beta: float,
            ensemble_uncertainty_decay: bool,
            intrinsic_reward_weight: float,
            intrinsic_min_max_stats: Optional[IntrinsicMinMaxStats],
            simulation_traces: Optional[List[Dict[str, Any]]],
            max_depth: Optional[int],
            max_nodes: Optional[int],
    ) -> Dict[str, Any]:
        max_depth = int(max_depth) if max_depth is not None else 3
        max_nodes = int(max_nodes) if max_nodes is not None else 128
        nodes: List[Dict[str, Any]] = []
        queue = [(root, [], None, None)]

        while queue and len(nodes) < max_nodes:
            node, path_actions, parent, incoming_action = queue.pop(0)
            node_record = self._node_diagnostic(
                node,
                path_actions,
                parent,
                incoming_action,
                mode,
                ensemble_ucb_beta,
                ensemble_uncertainty_decay,
                intrinsic_reward_weight,
                intrinsic_min_max_stats,
            )
            nodes.append(node_record)
            if len(path_actions) >= max_depth:
                continue
            for action in node.legal_actions:
                child = node.children.get(int(action))
                if child is None or child.visit_count <= 0:
                    continue
                queue.append((child, path_actions + [int(action)], node, int(action)))

        return {
            'mode': mode,
            'num_simulations_recorded': len(simulation_traces or []),
            'max_depth': max_depth,
            'max_nodes': max_nodes,
            'nodes_recorded': len(nodes),
            'intrinsic_q_min': (
                float(intrinsic_min_max_stats.minimum) if intrinsic_min_max_stats is not None else None
            ),
            'intrinsic_q_max': (
                float(intrinsic_min_max_stats.maximum) if intrinsic_min_max_stats is not None else None
            ),
            'external_q_min': (
                float(root.q_min_max_stats.minimum)
                if root.q_min_max_stats is not None and np.isfinite(root.q_min_max_stats.minimum) else None
            ),
            'external_q_max': (
                float(root.q_min_max_stats.maximum)
                if root.q_min_max_stats is not None and np.isfinite(root.q_min_max_stats.maximum) else None
            ),
            'external_q_range': (
                float(root.q_min_max_stats.value_range) if root.q_min_max_stats is not None else None
            ),
            'simulation_traces': simulation_traces or [],
            'nodes': nodes,
        }

    def _node_diagnostic(
            self,
            node: EnsembleSearchNode,
            path_actions: List[int],
            parent: Optional[EnsembleSearchNode],
            incoming_action: Optional[int],
            mode: str,
            ensemble_ucb_beta: float,
            ensemble_uncertainty_decay: bool,
            intrinsic_reward_weight: float,
        intrinsic_min_max_stats: Optional[IntrinsicMinMaxStats],
    ) -> Dict[str, Any]:
        if parent is None:
            mean_q = float(np.mean(node.q_mean)) if node.visit_count > 0 else 0.0
            mixed_mean_q = float(np.mean(node.mixed_q_mean)) if node.visit_count > 0 else 0.0
            mean_q_for_score = (
                mixed_mean_q if mode == 'mixed_reward_shared_tree' else mean_q
            )
            uncertainty = float(node.uncertainty)
            q_min_max_stats = node.q_min_max_stats
            components = {
                'total_score': 0.0,
                'mean_q': mean_q,
                'mixed_mean_q': mixed_mean_q,
                'mean_q_for_score': mean_q_for_score,
                'normalized_mean_q': (
                    q_min_max_stats.normalize_value(mean_q_for_score)
                    if q_min_max_stats is not None else mean_q_for_score
                ),
                'prior_score': 0.0,
                'uncertainty_score': 0.0,
                'normalized_intrinsic_value': (
                    intrinsic_min_max_stats.normalize(node.intrinsic_value)
                    if intrinsic_min_max_stats is not None else node.intrinsic_value
                ),
                'intrinsic_score': intrinsic_reward_weight * (
                    intrinsic_min_max_stats.normalize(node.intrinsic_value)
                    if intrinsic_min_max_stats is not None else node.intrinsic_value
                ),
                'uncertainty': uncertainty,
                'raw_uncertainty': float(node.raw_uncertainty),
                'aggregate_calibrated_uncertainty': uncertainty,
                'depth_corrected_uncertainty_uncapped': uncertainty,
                'depth_corrected_uncertainty': uncertainty,
                'depth_correction_cap_applied': 0.0,
                'uncertainty_metric': 'aggregate_calibrated_q_std',
                'normalized_uncertainty': (
                    q_min_max_stats.normalize_std(uncertainty) if q_min_max_stats is not None else uncertainty
                ),
                'q_min': (
                    float(q_min_max_stats.minimum)
                    if q_min_max_stats is not None and np.isfinite(q_min_max_stats.minimum) else 0.0
                ),
                'q_max': (
                    float(q_min_max_stats.maximum)
                    if q_min_max_stats is not None and np.isfinite(q_min_max_stats.maximum) else 0.0
                ),
                'q_range': float(q_min_max_stats.value_range) if q_min_max_stats is not None else 0.0,
                'q_normalization_source': (
                    'mixed' if mode == 'mixed_reward_shared_tree' else 'extrinsic'
                ),
            }
        else:
            components = self._score_child_components(
                parent,
                node,
                mode,
                ensemble_ucb_beta,
                ensemble_uncertainty_decay,
                intrinsic_reward_weight,
                intrinsic_min_max_stats,
            )

        q_values = node.q_mean.astype(np.float32).tolist() if node.visit_count > 0 else [
            0.0 for _ in range(node.ensemble_size)
        ]
        mixed_q_values = node.mixed_q_mean.astype(np.float32).tolist() if node.visit_count > 0 else [
            0.0 for _ in range(node.ensemble_size)
        ]
        calibrated_q_values = node.calibrated_q_mean.astype(np.float32).tolist() if node.visit_count > 0 else [
            0.0 for _ in range(node.ensemble_size)
        ]
        reward_return_values = node.predicted_reward_return_mean.astype(np.float32).tolist() if node.visit_count > 0 else [
            0.0 for _ in range(node.ensemble_size)
        ]
        leaf_contribution_values = node.leaf_value_contribution_mean.astype(np.float32).tolist() if node.visit_count > 0 else [
            0.0 for _ in range(node.ensemble_size)
        ]
        calibrated_leaf_contribution_values = (
            node.calibrated_leaf_value_contribution_mean.astype(np.float32).tolist()
            if node.visit_count > 0 else [0.0 for _ in range(node.ensemble_size)]
        )
        child_actions = [int(action) for action, child in node.children.items() if child.visit_count > 0]
        child_summaries = []
        for action in node.legal_actions:
            child = node.children.get(int(action))
            if child is None:
                continue
            child_components = self._score_child_components(
                node,
                child,
                mode,
                ensemble_ucb_beta,
                ensemble_uncertainty_decay,
                intrinsic_reward_weight,
                intrinsic_min_max_stats,
            )
            child_summaries.append(
                {
                    'action': int(action),
                    'visit_count': int(child.visit_count),
                    'prior': float(child.prior),
                    'total_score': float(child_components['total_score']),
                    'mean_q': float(child_components['mean_q']),
                    'mixed_mean_q': float(child_components['mixed_mean_q']),
                    'score_mean_q': float(child_components['mean_q_for_score']),
                    'normalized_mean_q': float(child_components['normalized_mean_q']),
                    'unbounded_ensemble_q': float(child_components['unbounded_ensemble_q']),
                    'bounded_ensemble_q': float(child_components['bounded_ensemble_q']),
                    'normalized_ensemble_q': float(child_components['normalized_ensemble_q']),
                    'ensemble_envelope_q': float(child_components['ensemble_envelope_q']),
                    'ensemble_envelope_gap_from_mean': float(
                        child_components['ensemble_envelope_gap_from_mean']
                    ),
                    'ensemble_envelope_member_id': int(child_components['ensemble_envelope_member_id']),
                    'ensemble_envelope_cap_applied': float(
                        child_components['ensemble_envelope_cap_applied']
                    ),
                    'ensemble_envelope_cap_adjustment': float(
                        child_components['ensemble_envelope_cap_adjustment']
                    ),
                    'q_std': float(child_components['uncertainty']),
                    'raw_q_std': float(child_components['raw_uncertainty']),
                    'calibrated_q_std': float(child_components['aggregate_calibrated_uncertainty']),
                    'uncertainty_metric': child_components['uncertainty_metric'],
                    'calibrated_backup_disagreement_mean': float(
                        child.calibrated_backup_disagreement_mean
                    ),
                    'calibrated_backup_disagreement_rms': float(
                        child.calibrated_backup_disagreement_rms
                    ),
                    'depth_normalized_backup_disagreement_mean': float(
                        child.depth_normalized_backup_disagreement_mean
                    ),
                    'depth_normalized_backup_disagreement_rms': float(
                        child.depth_normalized_backup_disagreement_rms
                    ),
                    'depth_corrected_aggregate_q_std_uncapped': float(
                        child_components['depth_corrected_uncertainty_uncapped']
                    ),
                    'depth_corrected_aggregate_q_std': float(
                        child_components['depth_corrected_uncertainty']
                    ),
                    'depth_corrected_q_residual_values': child.depth_corrected_q_residual(
                        calibrated=True
                    ).astype(np.float32).tolist(),
                    'depth_correction_cap_applied': float(
                        child_components['depth_correction_cap_applied']
                    ),
                    'applied_depth_disagreement_scale_mean': float(
                        child.depth_disagreement_scale_mean
                    ),
                    'q_values': child.q_mean.astype(np.float32).tolist() if child.visit_count > 0 else [
                        0.0 for _ in range(child.ensemble_size)
                    ],
                    'mixed_q_values': (
                        child.mixed_q_mean.astype(np.float32).tolist()
                        if child.visit_count > 0 else [0.0 for _ in range(child.ensemble_size)]
                    ),
                    'calibrated_q_values': (
                        child.calibrated_q_mean.astype(np.float32).tolist() if child.visit_count > 0 else [
                            0.0 for _ in range(child.ensemble_size)
                        ]
                    ),
                    'normalized_q_std': float(child_components['normalized_uncertainty']),
                    'q_min': float(child_components['q_min']),
                    'q_max': float(child_components['q_max']),
                    'q_range': float(child_components['q_range']),
                    'q_normalization_source': child_components['q_normalization_source'],
                    'prior_score': float(child_components['prior_score']),
                    'uncertainty_score': float(child_components['uncertainty_score']),
                    'unbounded_uncertainty_score': float(child_components['unbounded_uncertainty_score']),
                    'intrinsic_score': float(child_components['intrinsic_score']),
                    'intrinsic_value': float(child.intrinsic_value),
                    'normalized_intrinsic_value': float(child_components['normalized_intrinsic_value']),
                    'intrinsic_reward_return_mean': float(child.intrinsic_reward_return_mean),
                    'intrinsic_reward_weighted_return_mean': float(child.intrinsic_reward_weighted_return_mean),
                    'intrinsic_leaf_value_raw_contribution_mean': float(
                        child.intrinsic_leaf_value_raw_contribution_mean
                    ),
                    'intrinsic_leaf_value_clipped_contribution_mean': float(
                        child.intrinsic_leaf_value_clipped_contribution_mean
                    ),
                    'intrinsic_leaf_value_contribution_mean': float(
                        child.intrinsic_leaf_value_contribution_mean
                    ),
                    'intrinsic_leaf_value_before_modifier_contribution_mean': float(
                        child.intrinsic_leaf_value_before_modifier_contribution_mean
                    ),
                    'intrinsic_leaf_value_modifier_mean': float(
                        child.intrinsic_leaf_value_modifier_mean
                    ),
                    'intrinsic_reward': float(child.intrinsic_reward),
                    'intrinsic_reward_before_path_transform': float(
                        child.intrinsic_reward_before_path_transform
                    ),
                    'intrinsic_reward_before_path_modifier': float(
                        child.intrinsic_reward_before_path_modifier
                    ),
                    'intrinsic_path_reward_modifier': float(
                        child.intrinsic_path_reward_modifier
                    ),
                }
            )

        return {
            'path_actions': path_actions,
            'depth': int(len(path_actions)),
            'incoming_action': None if incoming_action is None else int(incoming_action),
            'visit_count': int(node.visit_count),
            'prior': float(node.prior),
            'reward_mean': float(np.mean(node.reward)) if node.reward.size > 0 else 0.0,
            'reward_values': node.reward.astype(np.float32).tolist(),
            'intrinsic_reward': float(node.intrinsic_reward),
            'intrinsic_reward_before_path_transform': float(
                node.intrinsic_reward_before_path_transform
            ),
            'intrinsic_reward_before_path_modifier': float(
                node.intrinsic_reward_before_path_modifier
            ),
            'intrinsic_path_reward_modifier': float(node.intrinsic_path_reward_modifier),
            'intrinsic_reward_before_path_bonus': float(
                node.intrinsic_reward_before_path_bonus
            ),
            'intrinsic_path_reward_bonus': float(node.intrinsic_path_reward_bonus),
            'mean_q': float(components['mean_q']),
            'mixed_mean_q': float(components['mixed_mean_q']),
            'score_mean_q': float(components['mean_q_for_score']),
            'normalized_mean_q': float(components['normalized_mean_q']),
            'unbounded_ensemble_q': float(components.get('unbounded_ensemble_q', components['mean_q'])),
            'bounded_ensemble_q': float(components.get('bounded_ensemble_q', components['mean_q'])),
            'normalized_ensemble_q': float(
                components.get('normalized_ensemble_q', components['normalized_mean_q'])
            ),
            'ensemble_envelope_q': float(components.get('ensemble_envelope_q', components['mean_q'])),
            'ensemble_envelope_gap_from_mean': float(
                components.get('ensemble_envelope_gap_from_mean', 0.0)
            ),
            'ensemble_envelope_member_id': int(components.get('ensemble_envelope_member_id', -1)),
            'ensemble_envelope_cap_applied': float(
                components.get('ensemble_envelope_cap_applied', 0.0)
            ),
            'ensemble_envelope_cap_adjustment': float(
                components.get('ensemble_envelope_cap_adjustment', 0.0)
            ),
            'q_std': float(components['uncertainty']),
            'raw_q_std': float(components['raw_uncertainty']),
            'calibrated_q_std': float(
                components.get('aggregate_calibrated_uncertainty', node.uncertainty)
            ),
            'uncertainty_metric': components.get('uncertainty_metric', 'aggregate_calibrated_q_std'),
            'calibrated_backup_disagreement_mean': float(node.calibrated_backup_disagreement_mean),
            'calibrated_backup_disagreement_rms': float(node.calibrated_backup_disagreement_rms),
            'depth_normalized_backup_disagreement_mean': float(
                node.depth_normalized_backup_disagreement_mean
            ),
            'depth_normalized_backup_disagreement_rms': float(
                node.depth_normalized_backup_disagreement_rms
            ),
            'depth_corrected_aggregate_q_std_uncapped': float(
                components['depth_corrected_uncertainty_uncapped']
            ),
            'depth_corrected_aggregate_q_std': float(
                components['depth_corrected_uncertainty']
            ),
            'depth_corrected_q_residual_values': node.depth_corrected_q_residual(
                calibrated=True
            ).astype(np.float32).tolist(),
            'depth_correction_cap_applied': float(components['depth_correction_cap_applied']),
            'applied_depth_disagreement_scale_mean': float(node.depth_disagreement_scale_mean),
            'normalized_q_std': float(components['normalized_uncertainty']),
            'q_min': float(components['q_min']),
            'q_max': float(components['q_max']),
            'q_range': float(components['q_range']),
            'q_normalization_source': components['q_normalization_source'],
            'q_values': q_values,
            'mixed_q_values': mixed_q_values,
            'calibrated_q_values': calibrated_q_values,
            'predicted_reward_return_mean': float(np.mean(reward_return_values)),
            'predicted_reward_return_values': reward_return_values,
            'leaf_value_contribution_mean': float(np.mean(leaf_contribution_values)),
            'leaf_value_contribution_values': leaf_contribution_values,
            'calibrated_leaf_value_contribution_mean': float(np.mean(calibrated_leaf_contribution_values)),
            'calibrated_leaf_value_contribution_values': calibrated_leaf_contribution_values,
            'intrinsic_value': float(node.intrinsic_value),
            'normalized_intrinsic_value': float(components['normalized_intrinsic_value']),
            'intrinsic_reward_return_mean': float(node.intrinsic_reward_return_mean),
            'intrinsic_reward_weighted_return_mean': float(node.intrinsic_reward_weighted_return_mean),
            'intrinsic_leaf_value_raw_contribution_mean': float(
                node.intrinsic_leaf_value_raw_contribution_mean
            ),
            'intrinsic_leaf_value_clipped_contribution_mean': float(
                node.intrinsic_leaf_value_clipped_contribution_mean
            ),
            'intrinsic_leaf_value_contribution_mean': float(node.intrinsic_leaf_value_contribution_mean),
            'intrinsic_immediate_reward_weight': float(self._cfg.get('intrinsic_tree_immediate_reward_weight', 1.0)),
            'intrinsic_leaf_value_weight': float(self._cfg.get('intrinsic_tree_leaf_value_weight', 1.0)),
            'intrinsic_tree_leaf_value_source': str(
                self._cfg.get('intrinsic_tree_leaf_value_source', 'default') or 'default'
            ),
            'dynamics_disagreement_space': str(
                self._cfg.get('dynamics_disagreement_space', 'latent') or 'latent'
            ),
            'latent_novelty': float(node.latent_novelty),
            'leaf_latent_novelty_mean': float(node.leaf_latent_novelty_mean),
            'leaf_latent_novelty_max': float(node.leaf_latent_novelty_max),
            'leaf_latent_novelty_max_depth': int(node.leaf_latent_novelty_max_depth),
            'leaf_latent_novelty_max_path': list(node.leaf_latent_novelty_max_path),
            'prior_score': float(components['prior_score']),
            'uncertainty_score': float(components['uncertainty_score']),
            'unbounded_uncertainty_score': float(components.get('unbounded_uncertainty_score', 0.0)),
            'intrinsic_score': float(components['intrinsic_score']),
            'total_score': float(components['total_score']),
            'expanded': bool(node.expanded),
            'visited_child_actions': child_actions,
            'children': child_summaries,
        }

    def _latent_novelty_np(
            self,
            model: torch.nn.Module,
            flat_latent_states: torch.Tensor,
            member_ids: Sequence[int],
            batch_size: int,
            ensemble_size: int,
    ) -> np.ndarray:
        if (
                not bool(self._cfg.get('latent_novelty_diagnostics', False))
                or not hasattr(model, 'aligned_latent_embedding')
        ):
            return np.zeros(batch_size, dtype=np.float32)

        embeddings = model.aligned_latent_embedding(
            flat_latent_states,
            member_id=member_ids,
            with_prior=True,
            normalize=True,
            detach_latent=True,
        )
        if embeddings.shape[-1] == 0:
            return np.zeros(batch_size, dtype=np.float32)
        embeddings = embeddings.reshape(ensemble_size, batch_size, -1)
        distances = []
        for member_i in range(ensemble_size):
            for member_j in range(member_i + 1, ensemble_size):
                distances.append((embeddings[member_i] - embeddings[member_j]).pow(2).sum(dim=-1))
        if not distances:
            return np.zeros(batch_size, dtype=np.float32)
        disagreement = torch.stack(distances, dim=0).mean(dim=0)
        return self._to_numpy(disagreement).astype(np.float32)

    def _intrinsic_reward_value_np(
            self,
            model: torch.nn.Module,
            flat_latent_states: torch.Tensor,
            member_ids: Sequence[int],
            batch_size: int,
            ensemble_size: int,
            intrinsic_reward_fn: Optional[Callable[[torch.Tensor], torch.Tensor]],
            use_model_intrinsic: bool,
            intrinsic_reward_source: str,
    ) -> tuple:
        if intrinsic_reward_source == "dynamics_disagreement":
            intrinsic_reward = self._dynamics_disagreement_reward_np(
                model=model,
                flat_latent_states=flat_latent_states,
                member_ids=member_ids,
                batch_size=batch_size,
                ensemble_size=ensemble_size,
            )
            intrinsic_value = self._intrinsic_leaf_value_np(
                model=model,
                flat_latent_states=flat_latent_states,
                member_ids=member_ids,
                batch_size=batch_size,
                ensemble_size=ensemble_size,
                intrinsic_reward_source=intrinsic_reward_source,
            )
            return intrinsic_reward, intrinsic_value

        if intrinsic_reward_source == "rnd":
            intrinsic_enabled_weight = float(self._cfg.get('rnd_intrinsic_reward_weight', 0.0))
        elif intrinsic_reward_source == "qstd":
            intrinsic_enabled_weight = float(self._cfg.get('qstd_intrinsic_reward_weight', 0.0))
        else:
            intrinsic_enabled_weight = 0.0
        if intrinsic_enabled_weight == 0.0:
            zeros = np.zeros(batch_size, dtype=np.float32)
            return zeros, zeros

        if use_model_intrinsic and hasattr(model, 'intrinsic_reward_from_latent') and hasattr(
                model, 'intrinsic_value_from_latent'
        ):
            intrinsic_reward = model.intrinsic_reward_from_latent(flat_latent_states, member_id=member_ids).detach()
            intrinsic_value = model.intrinsic_value_from_latent(flat_latent_states, member_id=member_ids).detach()
            intrinsic_reward = intrinsic_reward.reshape(ensemble_size, batch_size, -1).transpose(0, 1).squeeze(-1)
            intrinsic_value = intrinsic_value.reshape(ensemble_size, batch_size, -1).transpose(0, 1).squeeze(-1)
            intrinsic_reward = intrinsic_reward.mean(dim=1)
            intrinsic_value = intrinsic_value.mean(dim=1)
            return (
                self._to_numpy(intrinsic_reward).astype(np.float32),
                self._to_numpy(intrinsic_value).astype(np.float32),
            )

        if intrinsic_reward_fn is None:
            zeros = np.zeros(batch_size, dtype=np.float32)
            return zeros, zeros

        intrinsic_reward = intrinsic_reward_fn(flat_latent_states)
        intrinsic_reward = intrinsic_reward.reshape(ensemble_size, batch_size).transpose(0, 1).mean(dim=1)
        return self._to_numpy(intrinsic_reward).astype(np.float32), np.zeros(batch_size, dtype=np.float32)

    def _intrinsic_leaf_value_source(self, intrinsic_reward_source: str) -> str:
        source = str(self._cfg.get('intrinsic_tree_leaf_value_source', 'default') or 'default').lower()
        if source == 'default':
            if intrinsic_reward_source == 'rnd':
                return 'rnd'
            if intrinsic_reward_source == 'qstd':
                return 'intrinsic_value'
            if bool(self._cfg.get('dynamics_disagreement_train_intrinsic_value', False)):
                return 'intrinsic_value'
            return 'none'
        if source not in {'none', 'rnd', 'intrinsic_value'}:
            raise ValueError(
                "intrinsic_tree_leaf_value_source must be one of "
                "'default', 'none', 'rnd', or 'intrinsic_value', "
                f"got {source!r}."
            )
        return source

    def _intrinsic_leaf_value_np(
            self,
            model: torch.nn.Module,
            flat_latent_states: torch.Tensor,
            member_ids: Sequence[int],
            batch_size: int,
            ensemble_size: int,
            intrinsic_reward_source: str,
    ) -> np.ndarray:
        source = self._intrinsic_leaf_value_source(intrinsic_reward_source)
        if source == 'none':
            return np.zeros(batch_size, dtype=np.float32)
        if source in {'rnd', 'intrinsic_value'}:
            if not hasattr(model, 'intrinsic_value_from_latent'):
                return np.zeros(batch_size, dtype=np.float32)
            if source == 'rnd' and (
                    not bool(self._cfg.get('rnd_exploration', False))
                    or not bool(self._cfg.get('rnd_use_learned_intrinsic_model', True))
            ):
                return np.zeros(batch_size, dtype=np.float32)
            intrinsic_value = model.intrinsic_value_from_latent(
                flat_latent_states, member_id=member_ids
            ).detach()
            intrinsic_value = intrinsic_value.reshape(ensemble_size, batch_size, -1).transpose(0, 1).squeeze(-1)
            intrinsic_value = intrinsic_value.mean(dim=1)
            return self._to_numpy(intrinsic_value).astype(np.float32)
        raise ValueError(f"Unsupported intrinsic leaf value source: {source!r}.")

    def _dynamics_disagreement_reward_np(
            self,
            model: Optional[torch.nn.Module] = None,
            flat_latent_states: Optional[torch.Tensor] = None,
            member_ids: Optional[Sequence[int]] = None,
            batch_size: Optional[int] = None,
            ensemble_size: Optional[int] = None,
            disagreement_space: Optional[str] = None,
            apply_clip: bool = True,
    ) -> np.ndarray:
        if flat_latent_states is None and torch.is_tensor(model):
            flat_latent_states = model
            model = None
        if flat_latent_states is None or batch_size is None or ensemble_size is None:
            raise ValueError("flat_latent_states, batch_size, and ensemble_size are required.")

        disagreement_space = str(
            disagreement_space if disagreement_space is not None
            else self._cfg.get('dynamics_disagreement_space', 'latent') or 'latent'
        ).lower()
        if disagreement_space == 'latent':
            features = flat_latent_states
        elif disagreement_space in {'projection', 'prediction_projection'}:
            if not hasattr(model, 'project'):
                raise ValueError(
                    f"dynamics_disagreement_space={disagreement_space!r} requires model.project(...)."
                )
            if member_ids is None:
                member_ids = [member_id for member_id in range(int(ensemble_size)) for _ in range(int(batch_size))]
            features = model.project(
                flat_latent_states,
                with_grad=disagreement_space == 'prediction_projection',
                member_id=member_ids,
            )
        else:
            raise ValueError(
                "dynamics_disagreement_space must be one of "
                "'latent', 'projection', or 'prediction_projection', "
                f"got {disagreement_space!r}."
            )

        latents = features.reshape(ensemble_size, batch_size, -1).float()
        if bool(self._cfg.get('dynamics_disagreement_normalize_latent', True)):
            latents = torch.nn.functional.normalize(latents, p=2, dim=-1, eps=1e-8)
        metric = str(self._cfg.get('dynamics_disagreement_distance_metric', 'mse')).lower()
        if metric not in {'mse', 'l2'}:
            raise ValueError(
                "dynamics_disagreement_distance_metric must be one of 'mse' or 'l2', "
                f"got {metric!r}."
            )
        distances = []
        for member_i in range(ensemble_size):
            for member_j in range(member_i + 1, ensemble_size):
                pair_delta = latents[member_i] - latents[member_j]
                if metric == 'l2':
                    distances.append(pair_delta.pow(2).sum(dim=-1).sqrt())
                else:
                    distances.append(pair_delta.pow(2).mean(dim=-1))
        if not distances:
            disagreement = torch.zeros(batch_size, dtype=torch.float32, device=flat_latent_states.device)
        else:
            disagreement = torch.stack(distances, dim=0).mean(dim=0)
        clip_value = self._cfg.get('dynamics_disagreement_reward_clip', None)
        if apply_clip and clip_value is not None:
            disagreement = disagreement.clamp(max=float(clip_value))
        return self._to_numpy(disagreement).astype(np.float32)

    def _dynamics_disagreement_diagnostics_np(
            self,
            model: torch.nn.Module,
            flat_latent_states: torch.Tensor,
            member_ids: Sequence[int],
            batch_size: int,
            ensemble_size: int,
    ) -> Dict[str, np.ndarray]:
        zeros = np.zeros(batch_size, dtype=np.float32)
        diagnostics = {
            'latent': zeros,
            'projection': zeros,
            'prediction_projection': zeros,
        }
        for disagreement_space in diagnostics.keys():
            try:
                diagnostics[disagreement_space] = self._dynamics_disagreement_reward_np(
                    model=model,
                    flat_latent_states=flat_latent_states,
                    member_ids=member_ids,
                    batch_size=batch_size,
                    ensemble_size=ensemble_size,
                    disagreement_space=disagreement_space,
                    apply_clip=False,
                )
            except (AttributeError, NotImplementedError, ValueError):
                diagnostics[disagreement_space] = zeros
        return diagnostics

    @staticmethod
    def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy()

    @staticmethod
    def _action_priors(
            policy_logits: torch.Tensor,
            legal_actions: Sequence[int],
            policy_prior_mode: str = "learned",
    ) -> Dict[int, float]:
        legal_action_list = [int(action) for action in legal_actions]
        policy_prior_mode = str(policy_prior_mode or "learned").lower()
        if policy_prior_mode == "uniform":
            probability = 1.0 / len(legal_action_list) if legal_action_list else 0.0
            return {action: probability for action in legal_action_list}
        if policy_prior_mode != "learned":
            raise ValueError(f"Unsupported policy_prior_mode: {policy_prior_mode!r}.")
        return EnsembleSharedTreeMCTS._shared_priors(policy_logits, legal_action_list)

    @staticmethod
    def _shared_priors(policy_logits: torch.Tensor, legal_actions: Sequence[int]) -> Dict[int, float]:
        legal_action_list = [int(action) for action in legal_actions]
        legal_logits = policy_logits[:, legal_action_list]
        member_priors = torch.softmax(legal_logits, dim=-1)
        shared_priors = member_priors.mean(dim=0)
        return {action: float(shared_priors[idx].item()) for idx, action in enumerate(legal_action_list)}

    @staticmethod
    def _shared_policy_logits(policy_logits: torch.Tensor, legal_actions: Sequence[int]) -> List[float]:
        shared_priors = EnsembleSharedTreeMCTS._shared_priors(policy_logits, legal_actions)
        min_prob = 1e-8
        action_space_size = policy_logits.shape[-1]
        logits = np.full(action_space_size, -18.0, dtype=np.float32)
        for action, prob in shared_priors.items():
            logits[int(action)] = math.log(max(min_prob, prob))
        return logits.tolist()
