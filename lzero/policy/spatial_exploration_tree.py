"""Observation-grounded novelty for action paths inside MuZero search."""

from __future__ import annotations

import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class SpatialExplorationRoot:
    env_id: int
    latent: torch.Tensor
    belief: torch.Tensor
    memory_keys: torch.Tensor
    memory_beliefs: torch.Tensor
    observation_memory_keys: torch.Tensor
    memory_scale: float


class FrozenSpatialExplorationTree:
    """Maintain real episodic memory and score hypothetical action prefixes.

    The historical name is retained for checkpoint compatibility.  With
    ``online_training=True`` the model can instead start from random weights
    and update from complete trajectories collected by the current run.
    """

    def __init__(
            self,
            checkpoint_path: str | None,
            device: str,
            horizon: int = 5,
            adaptive_scale: bool = True,
            minimum_scale: float = 1e-3,
            score_clip: float | None = 5.0,
            online_training: bool = False,
            initialization_seed: int = 0,
            learning_rate: float = 1e-4,
            weight_decay: float = 1e-5,
            activation_updates: int = 1,
            replay_capacity: int = 16,
            history_window: int = 128,
            batch_size: int = 2,
            roots_per_update: int = 16,
            updates_per_collect: int = 16,
            grad_clip: float = 5.0,
            memory_novelty_weight: float = 10.0,
            memory_key_mode: str = "observation",
            revisit_scoring_mode: str = "model",
            rollout_reconstruction_weight: float = 0.0,
            rollout_contrastive_weight: float = 1.0,
            observation_key_consistency_weight: float = 0.0,
            belief_dynamics_consistency_weight: float = 0.0,
            belief_observation_alignment_weight: float = 1.0,
            pairwise_revisit_weight: float = 1.0,
            pairwise_revisit_negative_samples: int = 8,
            prioritized_replay_fraction: float = 0.0,
            prioritized_replay_alpha: float = 0.6,
            prioritized_replay_epsilon: float = 1e-3,
            prioritized_replay_priority_mode: str = "belief_error",
            revisit_history_scope: str = "episode",
            predictive_equivalence_weight: float = 0.0,
            predictive_equivalence_negative_margin: float = 0.2,
    ) -> None:
        try:
            from experiments.run_branching_rooms_spatial_exploration_world_model import (
                SpatialExplorationWorldModel,
            )
        except ModuleNotFoundError:
            from run_branching_rooms_spatial_exploration_world_model import (
                SpatialExplorationWorldModel,
            )

        self.online_training = bool(online_training)
        self.initialization_seed = int(initialization_seed)
        self._online_rng = np.random.default_rng(self.initialization_seed)
        if not self.online_training and (
                not checkpoint_path or not os.path.isfile(checkpoint_path)
        ):
            raise FileNotFoundError(
                f"Spatial exploration tree checkpoint does not exist: {checkpoint_path}"
            )
        if checkpoint_path and not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"Spatial exploration tree checkpoint does not exist: {checkpoint_path}"
            )
        self.checkpoint_path = None if not checkpoint_path else str(checkpoint_path)
        self.device = torch.device(device)
        self.horizon = max(1, int(horizon))
        self.adaptive_scale = bool(adaptive_scale)
        self.minimum_scale = max(float(minimum_scale), 1e-8)
        self.score_clip = None if score_clip is None else float(score_clip)
        if self.score_clip is not None and self.score_clip <= 0.0:
            raise ValueError("score_clip must be positive or None")

        checkpoint = (
            torch.load(
                self.checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )
            if self.checkpoint_path is not None else None
        )
        arguments = checkpoint.get("arguments", {}) if checkpoint is not None else {}
        checkpoint_horizon = int(arguments.get("horizon", self.horizon))
        if self.horizon > checkpoint_horizon:
            raise ValueError(
                f"Requested tree horizon {self.horizon} exceeds checkpoint "
                f"horizon {checkpoint_horizon}"
            )
        cpu_rng_state = torch.random.get_rng_state()
        cuda_rng_states = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        try:
            if checkpoint is None:
                torch.manual_seed(self.initialization_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(self.initialization_seed)
            self.model = SpatialExplorationWorldModel(
                observation_shape=(3, 28, 28),
                spatial_channels=int(arguments.get("spatial_channels", 32)),
                key_dim=int(arguments.get("key_dim", 64)),
                belief_dim=int(arguments.get("belief_dim", 128)),
                action_dim=int(arguments.get("action_dim", 16)),
                representation_mode=str(
                    arguments.get("representation_mode", "spatial_reconstruction")
                ),
                coarse_reconstruction_size=int(
                    arguments.get("coarse_reconstruction_size", 7)
                ),
                memory_key_mode=str(
                    arguments.get("memory_key_mode", memory_key_mode)
                ),
            ).to(self.device)
            if checkpoint is not None:
                self.model.load_state_dict(checkpoint["model"], strict=True)
        finally:
            torch.random.set_rng_state(cpu_rng_state)
            if cuda_rng_states is not None:
                torch.cuda.set_rng_state_all(cuda_rng_states)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(self.online_training)
        self.checkpoint_epoch = (
            int(checkpoint.get("epoch", -1)) if checkpoint is not None else -1
        )
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.activation_updates = max(0, int(activation_updates))
        self.replay_capacity = max(1, int(replay_capacity))
        self.history_window = max(self.horizon, int(history_window))
        self.batch_size = max(1, int(batch_size))
        self.roots_per_update = max(1, int(roots_per_update))
        self.updates_per_collect = max(1, int(updates_per_collect))
        self.grad_clip = max(0.0, float(grad_clip))
        self.memory_novelty_weight = max(0.0, float(memory_novelty_weight))
        self.revisit_scoring_mode = str(revisit_scoring_mode).lower()
        if self.revisit_scoring_mode not in {"model", "observation_key"}:
            raise ValueError(
                "revisit_scoring_mode must be 'model' or 'observation_key'"
            )
        self.rollout_reconstruction_weight = max(
            0.0, float(rollout_reconstruction_weight)
        )
        self.rollout_contrastive_weight = max(
            0.0, float(rollout_contrastive_weight)
        )
        self.observation_key_consistency_weight = max(
            0.0, float(observation_key_consistency_weight)
        )
        self.belief_dynamics_consistency_weight = max(
            0.0, float(belief_dynamics_consistency_weight)
        )
        self.belief_observation_alignment_weight = max(
            0.0, float(belief_observation_alignment_weight)
        )
        self.pairwise_revisit_weight = max(
            0.0, float(pairwise_revisit_weight)
        )
        self.pairwise_revisit_negative_samples = max(
            1, int(pairwise_revisit_negative_samples)
        )
        self.prioritized_replay_fraction = float(prioritized_replay_fraction)
        if not 0.0 <= self.prioritized_replay_fraction <= 1.0:
            raise ValueError("prioritized_replay_fraction must be in [0, 1]")
        self.prioritized_replay_alpha = float(prioritized_replay_alpha)
        if self.prioritized_replay_alpha < 0.0:
            raise ValueError("prioritized_replay_alpha cannot be negative")
        self.prioritized_replay_epsilon = float(prioritized_replay_epsilon)
        if self.prioritized_replay_epsilon <= 0.0:
            raise ValueError("prioritized_replay_epsilon must be positive")
        self.prioritized_replay_priority_mode = str(
            prioritized_replay_priority_mode
        ).lower()
        if self.prioritized_replay_priority_mode not in {
            "belief_error",
            "revisit_error",
        }:
            raise ValueError(
                "prioritized_replay_priority_mode must be 'belief_error' or "
                "'revisit_error'"
            )
        self.revisit_history_scope = str(revisit_history_scope).lower()
        if self.revisit_history_scope not in {"episode", "window"}:
            raise ValueError(
                "revisit_history_scope must be 'episode' or 'window'"
            )
        if (
                self.prioritized_replay_fraction > 0.0
                and (
                    self.model.memory_key_mode != "belief_revisit"
                    or self.pairwise_revisit_weight == 0.0
                )
        ):
            raise ValueError(
                "Prioritized replay requires active belief-revisit supervision"
            )
        self.predictive_equivalence_weight = max(
            0.0, float(predictive_equivalence_weight)
        )
        self.predictive_equivalence_negative_margin = float(
            predictive_equivalence_negative_margin
        )
        if not -1.0 <= self.predictive_equivalence_negative_margin <= 1.0:
            raise ValueError(
                "predictive_equivalence_negative_margin must be in [-1, 1]"
            )
        self.optimizer = (
            torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            if self.online_training else None
        )
        self.online_update_count = 0
        self.online_collection_phase = -1
        self._online_replay: List[Dict[str, np.ndarray]] = []
        self._next_online_replay_id = 0
        self._online_root_priorities: Dict[
            tuple[int, int, int, int], float
        ] = {}
        self._states: Dict[int, Dict[str, Any]] = {}
        self._records: List[Dict[str, Any]] = []
        self._path_cache: Dict[tuple[int, tuple[int, ...]], Dict[str, float]] = {}
        self._path_latent_cache: Dict[
            tuple[int, tuple[int, ...]], torch.Tensor
        ] = {}
        self._path_belief_cache: Dict[
            tuple[int, tuple[int, ...]], torch.Tensor
        ] = {}

    @property
    def active(self) -> bool:
        return (
            not self.online_training
            or self.online_update_count >= self.activation_updates
        )

    def _prepare_observations(self, observations: torch.Tensor) -> torch.Tensor:
        tensor = observations.detach().to(self.device).float()
        if float(tensor.detach().amax()) > 1.5:
            tensor = tensor.div(255.0)
        if tuple(tensor.shape[-2:]) != (28, 28):
            tensor = F.interpolate(
                tensor,
                size=(28, 28),
                mode="bilinear",
                align_corners=False,
            )
        return tensor.clamp(0.0, 1.0)

    @staticmethod
    def _observation_key(observation: torch.Tensor) -> bytes:
        return (
            observation.detach().clamp(0.0, 1.0).mul(255.0).round()
            .to(dtype=torch.uint8, device="cpu").contiguous().numpy().tobytes()
        )

    def _memory_scale(self, memory_keys: torch.Tensor) -> float:
        if not self.adaptive_scale or memory_keys.shape[0] < 2:
            return 1.0
        distances = 1.0 - memory_keys @ memory_keys.transpose(0, 1)
        distances = distances.clamp_min(0.0)
        valid = distances > 1e-7
        positive_nearest = torch.where(
            valid,
            distances,
            torch.full_like(distances, float("inf")),
        ).amin(dim=1)
        positive_nearest = positive_nearest[torch.isfinite(positive_nearest)]
        if positive_nearest.numel() == 0:
            return self.minimum_scale
        return max(
            self.minimum_scale,
            float(positive_nearest.median().detach().cpu()),
        )

    def _score(self, raw_distance: float, scale: float) -> float:
        raw_distance = max(0.0, float(raw_distance))
        if self.adaptive_scale:
            score = float(np.log1p(raw_distance / max(scale, self.minimum_scale)))
        else:
            score = raw_distance
        if self.score_clip is not None:
            score = min(score, self.score_clip)
        return score

    @staticmethod
    def _normalized_novelty(score: float) -> float:
        """Map the non-negative log-distance score to a bounded gate value."""
        return float(np.clip(-np.expm1(-max(0.0, float(score))), 0.0, 1.0))

    @staticmethod
    def _raw_novelty(key: torch.Tensor, memory_keys: torch.Tensor) -> float:
        if memory_keys.shape[0] == 0:
            return 1.0
        return float(
            (1.0 - torch.mv(memory_keys, key).amax()).clamp_min(0.0).detach().cpu()
        )

    def _canonical_online_episode(
            self, episode: Mapping[str, Any]
    ) -> Dict[str, np.ndarray] | None:
        observations = np.asarray(episode.get("observations", []))
        actions = np.asarray(
            episode.get("actions", []), dtype=np.int64
        ).reshape(-1)
        if observations.ndim != 4 or observations.shape[1] != 3:
            return None
        step_count = min(actions.size, max(0, observations.shape[0] - 1))
        if step_count < self.horizon:
            return None
        tensor = torch.as_tensor(observations[:step_count + 1]).float()
        if float(tensor.amax()) > 1.5:
            tensor = tensor.div(255.0)
        if tuple(tensor.shape[-2:]) != (28, 28):
            tensor = F.interpolate(
                tensor,
                size=(28, 28),
                mode="bilinear",
                align_corners=False,
            )
        prepared = tensor.clamp(0.0, 1.0).mul(255.0).round().to(
            torch.uint8
        ).cpu().numpy()
        return {
            "observations": prepared,
            "actions": actions[:step_count].copy(),
        }

    def _uniform_online_window(
            self,
            episodes: Sequence[Mapping[str, Any]],
            action_count: int,
    ) -> tuple[Mapping[str, Any], int]:
        window_counts = np.asarray(
            [len(episode["actions"]) - action_count + 1 for episode in episodes],
            dtype=np.float64,
        )
        probabilities = window_counts / window_counts.sum()
        episode_index = int(
            self._online_rng.choice(len(episodes), p=probabilities)
        )
        episode = episodes[episode_index]
        maximum_start = len(episode["actions"]) - action_count
        start = int(self._online_rng.integers(0, maximum_start + 1))
        return episode, start

    def _sample_online_batch(self):
        eligible = [
            episode
            for episode in self._online_replay
            if len(episode["actions"]) >= self.horizon
        ]
        if not eligible:
            return None
        full_episode_history = (
            self.model.memory_key_mode == "belief_revisit"
            and self.revisit_history_scope == "episode"
        )
        if (
            self.prioritized_replay_fraction == 0.0
            and not full_episode_history
        ):
            indices = self._online_rng.integers(
                0, len(eligible), size=self.batch_size
            )
            selected = [eligible[int(index)] for index in indices]
            action_count = min(
                self.history_window,
                min(len(episode["actions"]) for episode in selected),
            )
            selected_windows = []
            for episode in selected:
                maximum_start = len(episode["actions"]) - action_count
                start = (
                    int(self._online_rng.integers(0, maximum_start + 1))
                    if maximum_start > 0 else 0
                )
                selected_windows.append(
                    (episode, start, False, None, None)
                )
        else:
            length_groups: Dict[int, List[Mapping[str, Any]]] = {}
            for episode in eligible:
                action_count = (
                    len(episode["actions"])
                    if full_episode_history
                    else min(
                        self.history_window, len(episode["actions"])
                    )
                )
                length_groups.setdefault(action_count, []).append(episode)
            lengths = np.asarray(list(length_groups), dtype=np.int64)
            group_weights = np.asarray(
                [
                    sum(
                        len(episode["actions"]) - int(length) + 1
                        for episode in length_groups[int(length)]
                    )
                    for length in lengths
                ],
                dtype=np.float64,
            )
            action_count = int(
                self._online_rng.choice(
                    lengths, p=group_weights / group_weights.sum()
                )
            )
            group = length_groups[action_count]
            valid_ids = {int(episode["replay_id"]) for episode in group}
            known = (
                [
                    (key, priority)
                    for key, priority in self._online_root_priorities.items()
                    if (
                        len(key) == 4
                        and key[0] in valid_ids
                        and key[2] == action_count
                        and 0 <= key[3] <= action_count - self.horizon
                    )
                ]
                if self.prioritized_replay_fraction > 0.0
                else []
            )
            prioritized_count = min(
                self.batch_size,
                self.roots_per_update,
                int(round(self.batch_size * self.prioritized_replay_fraction)),
            )
            if not known:
                prioritized_count = 0
            uniform_count = self.batch_size - prioritized_count
            selected_windows = [
                (
                    *self._uniform_online_window(group, action_count),
                    False,
                    None,
                    None,
                )
                for _ in range(uniform_count)
            ]
            if prioritized_count:
                values = np.asarray(
                    [priority for _, priority in known], dtype=np.float64
                )
                weights = np.power(
                    values + self.prioritized_replay_epsilon,
                    self.prioritized_replay_alpha,
                )
                chosen = self._online_rng.choice(
                    len(known),
                    size=prioritized_count,
                    replace=True,
                    p=weights / weights.sum(),
                )
                episodes_by_id = {
                    int(episode["replay_id"]): episode for episode in group
                }
                selected_windows.extend(
                    [
                        (
                            episodes_by_id[known[int(index)][0][0]],
                            known[int(index)][0][1],
                            True,
                            known[int(index)][0][3],
                            known[int(index)][1],
                        )
                        for index in chosen
                    ]
                )
            self._online_rng.shuffle(selected_windows)
        observations = []
        actions = []
        window_keys = []
        prioritized_flags = []
        requested_root_offsets = []
        requested_stored_priorities = []
        episode_action_counts = []
        window_starts = []
        for (
            episode,
            start,
            prioritized,
            requested_root,
            stored_priority,
        ) in selected_windows:
            observations.append(
                episode["observations"][start:start + action_count + 1]
            )
            actions.append(episode["actions"][start:start + action_count])
            window_keys.append(
                (int(episode["replay_id"]), int(start), int(action_count))
            )
            prioritized_flags.append(bool(prioritized))
            requested_root_offsets.append(
                None if requested_root is None else int(requested_root)
            )
            requested_stored_priorities.append(
                None if stored_priority is None else float(stored_priority)
            )
            episode_action_counts.append(len(episode["actions"]))
            window_starts.append(int(start))
        observation_tensor = torch.as_tensor(
            np.stack(observations), dtype=torch.uint8, device=self.device
        )
        action_tensor = torch.as_tensor(
            np.stack(actions), dtype=torch.long, device=self.device
        )
        flattened = observation_tensor.reshape(
            -1, int(np.prod(observation_tensor.shape[-3:]))
        )
        _, inverse = torch.unique(flattened, dim=0, return_inverse=True)
        state_ids = inverse.reshape(observation_tensor.shape[:2])
        batch = {
            "observations": observation_tensor,
            "actions": action_tensor,
            "state_ids": state_ids,
        }
        sampling = {
            "window_keys": window_keys,
            "prioritized_flags": prioritized_flags,
            "requested_root_offsets": requested_root_offsets,
            "requested_stored_priorities": requested_stored_priorities,
            "episode_action_counts": episode_action_counts,
            "window_starts": window_starts,
            "full_episode_history": full_episode_history,
        }
        return batch, sampling

    def _sample_online_roots(
            self,
            action_count: int,
            requested_root_offsets: Sequence[int | None],
    ) -> torch.Tensor:
        available_count = int(action_count) - self.horizon + 1
        if available_count <= 0:
            raise ValueError("The horizon exceeds the trajectory length.")
        root_count = min(self.roots_per_update, available_count)
        requested = sorted(
            {
                int(root)
                for root in requested_root_offsets
                if root is not None and 0 <= int(root) < available_count
            }
        )
        if len(requested) > root_count:
            requested = requested[:root_count]
        requested_set = set(requested)
        remaining = [
            root for root in range(available_count)
            if root not in requested_set
        ]
        fill_count = root_count - len(requested)
        if fill_count:
            order = torch.randperm(len(remaining))[:fill_count].tolist()
            requested.extend(remaining[index] for index in order)
        return torch.tensor(
            sorted(requested), dtype=torch.long, device=self.device
        )

    def train_online(
            self,
            episodes: Sequence[Mapping[str, Any]],
            collection_phase: int,
    ) -> Dict[str, float]:
        """Update the exploration model from executed trajectories only."""
        if not self.online_training or self.optimizer is None:
            return {}
        fresh = []
        for episode in episodes:
            canonical = self._canonical_online_episode(episode)
            if canonical is not None:
                canonical["replay_id"] = self._next_online_replay_id
                self._next_online_replay_id += 1
                self._online_replay.append(canonical)
                fresh.append(canonical)
        if len(self._online_replay) > self.replay_capacity:
            del self._online_replay[:-self.replay_capacity]
        retained_ids = {
            int(episode["replay_id"]) for episode in self._online_replay
        }
        self._online_root_priorities = {
            key: priority
            for key, priority in self._online_root_priorities.items()
            if key[0] in retained_ids
        }
        if not fresh:
            return {}

        try:
            from experiments.run_branching_rooms_spatial_exploration_world_model import (
                sample_roots,
                stage_two_loss,
            )
        except ModuleNotFoundError:
            from run_branching_rooms_spatial_exploration_world_model import (
                sample_roots,
                stage_two_loss,
            )

        loss_args = SimpleNamespace(
            horizon=self.horizon,
            direct_reconstruction_weight=2.0,
            direct_contrastive_weight=1.0,
            belief_observation_alignment_weight=(
                self.belief_observation_alignment_weight
            ),
            pairwise_revisit_weight=self.pairwise_revisit_weight,
            pairwise_revisit_negative_samples=(
                self.pairwise_revisit_negative_samples
            ),
            prioritized_replay_priority_mode=(
                self.prioritized_replay_priority_mode
            ),
            inverse_action_weight=0.1,
            variance_weight=0.1,
            latent_dynamics_weight=0.0,
            key_dynamics_weight=1.0,
            observation_key_consistency_weight=(
                self.observation_key_consistency_weight
            ),
            belief_dynamics_consistency_weight=(
                self.belief_dynamics_consistency_weight
            ),
            rollout_reconstruction_weight=self.rollout_reconstruction_weight,
            rollout_contrastive_weight=self.rollout_contrastive_weight,
            memory_novelty_weight=self.memory_novelty_weight,
            predictive_equivalence_weight=self.predictive_equivalence_weight,
            predictive_equivalence_negative_margin=(
                self.predictive_equivalence_negative_margin
            ),
            joint_representation_dynamics=True,
            temperature=0.1,
        )
        rows: List[Dict[str, float]] = []
        grad_norms = []
        cpu_rng_state = torch.random.get_rng_state()
        cuda_rng_states = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        try:
            local_seed = self.initialization_seed + self.online_update_count
            torch.manual_seed(local_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(local_seed)
            self.model.train()
            for _ in range(self.updates_per_collect):
                sampled = self._sample_online_batch()
                if sampled is None:
                    break
                batch, sampling = sampled
                if self.prioritized_replay_fraction > 0.0:
                    roots = self._sample_online_roots(
                        int(batch["actions"].shape[1]),
                        sampling["requested_root_offsets"],
                    )
                else:
                    roots = sample_roots(
                        int(batch["actions"].shape[1]),
                        self.horizon,
                        self.roots_per_update,
                        self.device,
                    )
                loss, row, sample_priorities = stage_two_loss(
                    self.model,
                    batch,
                    roots,
                    loss_args,
                    return_sample_priorities=True,
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )
                self.optimizer.step()
                priorities = sample_priorities.detach().cpu().numpy()
                if self.prioritized_replay_fraction > 0.0:
                    root_offsets = roots.detach().cpu().tolist()
                    for row_index, window_key in enumerate(
                            sampling["window_keys"]
                    ):
                        for root_index, root_offset in enumerate(root_offsets):
                            priority_key = (*window_key, int(root_offset))
                            self._online_root_priorities[priority_key] = max(
                                self.prioritized_replay_epsilon,
                                float(priorities[row_index, root_index]),
                            )
                prioritized_flags = np.asarray(
                    sampling["prioritized_flags"], dtype=bool
                )
                root_columns = {
                    int(root): index
                    for index, root in enumerate(roots.detach().cpu().tolist())
                }
                requested_priorities = [
                    priorities[row_index, root_columns[int(root_offset)]]
                    for row_index, root_offset in enumerate(
                        sampling["requested_root_offsets"]
                    )
                    if root_offset is not None
                    and int(root_offset) in root_columns
                ]
                requested_stored_priorities = [
                    float(priority)
                    for priority in sampling["requested_stored_priorities"]
                    if priority is not None
                ]
                row.update(
                    {
                        "replay_priority_mean": float(np.mean(priorities)),
                        "replay_priority_uniform_mean": float(
                            np.mean(priorities[~prioritized_flags])
                            if np.any(~prioritized_flags) else 0.0
                        ),
                        "replay_priority_prioritized_mean": float(
                            np.mean(requested_priorities)
                            if requested_priorities else 0.0
                        ),
                        "replay_priority_prioritized_stored_mean": float(
                            np.mean(requested_stored_priorities)
                            if requested_stored_priorities else 0.0
                        ),
                        "replay_prioritized_sample_fraction": float(
                            prioritized_flags.mean()
                        ),
                        "replay_priority_cache_size": float(
                            len(self._online_root_priorities)
                        ),
                        "revisit_full_episode_history": float(
                            sampling["full_episode_history"]
                        ),
                        "revisit_history_fraction": float(
                            np.mean(
                                [
                                    int(batch["actions"].shape[1])
                                    / max(1, int(episode_action_count))
                                    for episode_action_count in sampling[
                                        "episode_action_counts"
                                    ]
                                ]
                            )
                        ),
                        "revisit_window_start_mean": float(
                            np.mean(sampling["window_starts"])
                        ),
                        "revisit_window_start_max": float(
                            np.max(sampling["window_starts"])
                        ),
                        "revisit_history_steps": float(
                            batch["actions"].shape[1]
                        ),
                    }
                )
                rows.append(row)
                grad_norms.append(float(grad_norm.detach().cpu()))
                self.online_update_count += 1
        finally:
            self.model.eval()
            torch.random.set_rng_state(cpu_rng_state)
            if cuda_rng_states is not None:
                torch.cuda.set_rng_state_all(cuda_rng_states)
        self.online_collection_phase = int(collection_phase)
        # Stored keys and recurrent beliefs were produced by the old encoder.
        self.reset()
        if not rows:
            return {}
        mean = {
            key: float(np.mean([row[key] for row in rows]))
            for key in rows[0]
        }
        return {
            "fresh_episode_count": float(len(fresh)),
            "replay_episode_count": float(len(self._online_replay)),
            "update_count": float(len(rows)),
            "total_update_count": float(self.online_update_count),
            "active_next_collection": float(self.active),
            "grad_norm_before_clip": float(np.mean(grad_norms)),
            **mean,
        }

    def online_state_dict(self, include_optimizer: bool = False) -> Dict[str, Any]:
        if not self.online_training:
            return {}
        state = {
            "version": 3,
            "model": self.model.state_dict(),
            "online_update_count": int(self.online_update_count),
            "online_collection_phase": int(self.online_collection_phase),
            "replay": list(self._online_replay),
            "next_online_replay_id": int(self._next_online_replay_id),
            "online_root_priorities": dict(
                self._online_root_priorities
            ),
            "online_rng_state": self._online_rng.bit_generator.state,
        }
        if include_optimizer and self.optimizer is not None:
            state["optimizer"] = self.optimizer.state_dict()
        return state

    def load_online_state_dict(
            self, state: Mapping[str, Any], load_optimizer: bool = False
    ) -> None:
        if not state or not self.online_training:
            return
        self.model.load_state_dict(state.get("model", {}), strict=False)
        self.online_update_count = int(state.get("online_update_count", 0))
        self.online_collection_phase = int(
            state.get("online_collection_phase", -1)
        )
        self._online_replay = []
        for episode in state.get("replay", []):
            canonical = self._canonical_online_episode(episode)
            if canonical is not None:
                replay_id = episode.get("replay_id")
                if replay_id is None:
                    replay_id = self._next_online_replay_id
                    self._next_online_replay_id += 1
                canonical["replay_id"] = int(replay_id)
                self._online_replay.append(canonical)
        self._online_replay = self._online_replay[-self.replay_capacity:]
        if self._online_replay:
            minimum_next_id = 1 + max(
                int(episode["replay_id"])
                for episode in self._online_replay
            )
            self._next_online_replay_id = max(
                minimum_next_id,
                int(state.get("next_online_replay_id", minimum_next_id)),
            )
        retained_ids = {
            int(episode["replay_id"]) for episode in self._online_replay
        }
        stored_priorities = state.get(
            "online_root_priorities",
            state.get("online_window_priorities", {}),
        )
        self._online_root_priorities = {
            tuple(int(value) for value in key): float(priority)
            for key, priority in stored_priorities.items()
            if len(key) == 4 and int(key[0]) in retained_ids
        }
        if "online_rng_state" in state:
            self._online_rng.bit_generator.state = state["online_rng_state"]
        if load_optimizer and self.optimizer is not None and "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        self.model.eval()
        self.reset()

    def prepare_roots(
            self,
            observations: torch.Tensor,
            env_ids: Sequence[int],
    ) -> List[SpatialExplorationRoot]:
        if len(env_ids) != int(observations.shape[0]):
            raise ValueError("env_ids and observations must have equal batch size")
        prepared = self._prepare_observations(observations)
        roots: List[SpatialExplorationRoot] = []
        self._path_cache.clear()
        self._path_latent_cache.clear()
        self._path_belief_cache.clear()
        with torch.inference_mode():
            latents = self.model.encode_observations(prepared)
            observation_keys = self.model.keys_from_latents(latents)
            for row, env_id_value in enumerate(env_ids):
                env_id = int(env_id_value)
                latent = latents[row].detach()
                observation_embedding = observation_keys[row].detach()
                observation_key = self._observation_key(prepared[row])
                state = self._states.get(env_id)
                if state is None:
                    belief = torch.zeros(
                        1,
                        self.model.belief_dim,
                        device=self.device,
                        dtype=observation_embedding.dtype,
                    )
                    previous_action = torch.full(
                        (1,),
                        self.model.action_space_size,
                        device=self.device,
                        dtype=torch.long,
                    )
                    belief = self.model.belief_cell(
                        torch.cat(
                            [
                                observation_embedding[None],
                                self.model.action_embedding(previous_action),
                            ],
                            dim=-1,
                        ),
                        belief,
                    ).squeeze(0)
                    key = self.model.memory_keys(
                        latent[None], belief[None]
                    ).squeeze(0)
                    memory_keys = key[None]
                    state = {
                        "belief": belief.detach(),
                        "memory_keys": memory_keys.detach(),
                        "memory_beliefs": belief[None].detach(),
                        "observation_memory_keys": (
                            observation_embedding[None].detach()
                        ),
                        "observation_keys": {observation_key},
                        "pending": None,
                    }
                    self._states[env_id] = state
                else:
                    pending = state.get("pending")
                    previous_action_value = (
                        self.model.action_space_size
                        if pending is None else int(pending["selected_action"])
                    )
                    previous_action = torch.as_tensor(
                        [previous_action_value],
                        device=self.device,
                        dtype=torch.long,
                    )
                    belief = self.model.belief_cell(
                        torch.cat(
                            [
                                observation_embedding[None],
                                self.model.action_embedding(previous_action),
                            ],
                            dim=-1,
                        ),
                        state["belief"][None],
                    ).squeeze(0)
                    key = self.model.memory_keys(
                        latent[None], belief[None]
                    ).squeeze(0)
                    previous_memory = state["memory_keys"]
                    previous_beliefs = state["memory_beliefs"]
                    previous_observation_memory = state[
                        "observation_memory_keys"
                    ]
                    if self.revisit_scoring_mode == "observation_key":
                        previous_scale = self._memory_scale(
                            previous_observation_memory
                        )
                        realized_raw = self._raw_novelty(
                            observation_embedding,
                            previous_observation_memory,
                        )
                        realized_score = self._score(
                            realized_raw, previous_scale
                        )
                    elif self.model.memory_key_mode == "belief_revisit":
                        realized_raw = float(
                            self.model.revisit_novelty(
                                belief, previous_beliefs
                            ).detach().cpu()
                        )
                        realized_score = realized_raw
                    else:
                        previous_scale = self._memory_scale(previous_memory)
                        realized_raw = self._raw_novelty(key, previous_memory)
                        realized_score = self._score(
                            realized_raw, previous_scale
                        )
                    if pending is not None:
                        self._records.append(
                            {
                                **pending,
                                "realized_raw_novelty": realized_raw,
                                "realized_novelty": realized_score,
                                "realized_rgb_novel": float(
                                    observation_key not in state["observation_keys"]
                                ),
                                "absolute_novelty_error": abs(
                                    float(pending["predicted_novelty"])
                                    - realized_score
                                ),
                            }
                        )
                    state["belief"] = belief.detach()
                    state["memory_keys"] = torch.cat(
                        [previous_memory, key[None]], dim=0
                    ).detach()
                    state["memory_beliefs"] = torch.cat(
                        [previous_beliefs, belief[None]], dim=0
                    ).detach()
                    state["observation_memory_keys"] = torch.cat(
                        [
                            previous_observation_memory,
                            observation_embedding[None],
                        ],
                        dim=0,
                    ).detach()
                    state["observation_keys"].add(observation_key)
                    state["pending"] = None
                memory_keys = state["memory_keys"]
                roots.append(
                    SpatialExplorationRoot(
                        env_id=env_id,
                        latent=latent,
                        belief=state["belief"],
                        memory_keys=memory_keys,
                        memory_beliefs=state["memory_beliefs"],
                        observation_memory_keys=state[
                            "observation_memory_keys"
                        ],
                        memory_scale=(
                            self._memory_scale(
                                state["observation_memory_keys"]
                            )
                            if self.revisit_scoring_mode == "observation_key"
                            else (
                                1.0
                                if self.model.memory_key_mode == "belief_revisit"
                                else self._memory_scale(memory_keys)
                            )
                        ),
                    )
                )
        return roots

    def score_paths(
            self,
            roots: Sequence[SpatialExplorationRoot],
            root_indices: Sequence[int],
            action_paths: Sequence[Sequence[int]],
            maximum_depth: int,
    ) -> List[Dict[str, float]]:
        if len(root_indices) != len(action_paths):
            raise ValueError("root_indices and action_paths must have equal length")
        maximum_depth = min(self.horizon, max(1, int(maximum_depth)))
        results = [
            {
                "raw_novelty": 0.0,
                "novelty": 0.0,
                "normalized_novelty": 0.0,
            }
            for _ in action_paths
        ]
        grouped: Dict[int, List[int]] = {}
        for row, path in enumerate(action_paths):
            depth = len(path)
            if 0 < depth <= maximum_depth:
                root = roots[int(root_indices[row])]
                cache_key = (
                    root.env_id,
                    tuple(int(action) for action in path),
                )
                if cache_key in self._path_cache:
                    results[row] = dict(self._path_cache[cache_key])
                    continue
                grouped.setdefault(depth, []).append(row)
        with torch.inference_mode():
            for depth, rows in grouped.items():
                latent_batch = torch.stack(
                    [roots[int(root_indices[row])].latent for row in rows]
                )
                belief_batch = torch.stack(
                    [roots[int(root_indices[row])].belief for row in rows]
                )
                action_batch = torch.as_tensor(
                    [action_paths[row] for row in rows],
                    device=self.device,
                    dtype=torch.long,
                )
                rollout = self.model.rollout(
                    latent_batch,
                    belief_batch,
                    action_batch,
                )
                for batch_row, output_row in enumerate(rows):
                    root = roots[int(root_indices[output_row])]
                    path = tuple(int(action) for action in action_paths[output_row])
                    if self.revisit_scoring_mode == "observation_key":
                        memory = root.observation_memory_keys
                        if depth > 1:
                            memory = torch.cat(
                                [
                                    memory,
                                    rollout.observation_keys[
                                        batch_row, : depth - 1
                                    ],
                                ],
                                dim=0,
                            )
                        raw = self._raw_novelty(
                            rollout.observation_keys[
                                batch_row, depth - 1
                            ],
                            memory,
                        )
                        novelty = self._score(raw, root.memory_scale)
                        normalized_novelty = self._normalized_novelty(
                            novelty
                        )
                    elif self.model.memory_key_mode == "belief_revisit":
                        memory_beliefs = root.memory_beliefs
                        if depth > 1:
                            memory_beliefs = torch.cat(
                                [
                                    memory_beliefs,
                                    rollout.beliefs[batch_row, : depth - 1],
                                ],
                                dim=0,
                            )
                        raw = float(
                            self.model.revisit_novelty(
                                rollout.beliefs[batch_row, depth - 1],
                                memory_beliefs,
                            ).detach().cpu()
                        )
                        novelty = raw
                        normalized_novelty = raw
                    else:
                        memory = root.memory_keys
                        if depth > 1:
                            memory = torch.cat(
                                [
                                    memory,
                                    rollout.keys[batch_row, : depth - 1],
                                ],
                                dim=0,
                            )
                        raw = self._raw_novelty(
                            rollout.keys[batch_row, depth - 1], memory
                        )
                        novelty = self._score(raw, root.memory_scale)
                        normalized_novelty = self._normalized_novelty(novelty)
                    result = {
                        "raw_novelty": raw,
                        "novelty": novelty,
                        "normalized_novelty": normalized_novelty,
                    }
                    results[output_row] = result
                    cache_key = (root.env_id, path)
                    self._path_cache[cache_key] = dict(result)
                    self._path_latent_cache[cache_key] = (
                        rollout.latents[batch_row, depth - 1].detach()
                    )
                    self._path_belief_cache[cache_key] = (
                        rollout.beliefs[batch_row, depth - 1].detach()
                    )
        return results

    def predict_path_observations(
            self,
            roots: Sequence[SpatialExplorationRoot],
            root_indices: Sequence[int],
            action_paths: Sequence[Sequence[int]],
            maximum_depth: int,
    ) -> torch.Tensor:
        if not action_paths:
            return torch.empty(
                0,
                *self.model.observation_shape,
                device=self.device,
            )
        maximum_depth = min(self.horizon, max(1, int(maximum_depth)))
        if any(not 0 < len(path) <= maximum_depth for path in action_paths):
            raise ValueError(
                "All decoded paths must be nonempty and within maximum_depth"
            )
        self.score_paths(
            roots,
            root_indices,
            action_paths,
            maximum_depth=maximum_depth,
        )
        with torch.inference_mode():
            cache_keys = [
                (
                    roots[int(root_index)].env_id,
                    tuple(int(action) for action in path),
                )
                for root_index, path in zip(root_indices, action_paths)
            ]
            if self.model.memory_key_mode in (
                "belief",
                "belief_to_observation",
                "belief_revisit",
            ):
                beliefs = torch.stack([
                    self._path_belief_cache[cache_key]
                    for cache_key in cache_keys
                ])
                return self.model.decode_beliefs(beliefs).detach()
            latents = torch.stack([
                self._path_latent_cache[cache_key]
                for cache_key in cache_keys
            ])
            return self.model.decode_latents(latents).detach()

    def make_bonus_fn(
            self,
            roots: Sequence[SpatialExplorationRoot],
            beta: float,
            maximum_depth: int,
    ):
        beta = float(beta)
        if beta < 0.0:
            raise ValueError("Spatial exploration beta cannot be negative")

        def bonus(root_indices, action_paths, _next_latent):
            if not self.active:
                return np.zeros(len(action_paths), dtype=np.float32)
            scores = self.score_paths(
                roots,
                root_indices,
                action_paths,
                maximum_depth=maximum_depth,
            )
            return np.asarray(
                [beta * row["novelty"] for row in scores], dtype=np.float32
            )

        return bonus

    def make_modifier_fn(
            self,
            roots: Sequence[SpatialExplorationRoot],
            minimum_reward_factor: float,
            maximum_depth: int,
    ):
        minimum_reward_factor = float(minimum_reward_factor)
        if not 0.0 <= minimum_reward_factor <= 1.0:
            raise ValueError("Spatial exploration minimum reward factor must be in [0, 1]")
        maximum_depth = min(self.horizon, max(1, int(maximum_depth)))

        def modifier(root_indices, action_paths, _next_latent):
            if not self.active:
                return np.ones(len(action_paths), dtype=np.float32)
            scores = self.score_paths(
                roots,
                root_indices,
                action_paths,
                maximum_depth=maximum_depth,
            )
            factors = []
            for path, row in zip(action_paths, scores):
                if not 0 < len(path) <= maximum_depth:
                    factors.append(1.0)
                    continue
                factors.append(
                    minimum_reward_factor
                    + (1.0 - minimum_reward_factor)
                    * float(row["normalized_novelty"])
                )
            return np.asarray(factors, dtype=np.float32)

        return modifier

    def commit_actions(
            self,
            roots: Sequence[SpatialExplorationRoot],
            actions: Sequence[int],
            root_action_diagnostics: Sequence[Sequence[Mapping[str, Any]]],
            planned_actions: Sequence[int] | None = None,
            beta: float = 1.0,
            reward_mode: str = "bonus",
            minimum_reward_factor: float = 0.05,
    ) -> None:
        if planned_actions is None:
            planned_actions = actions
        beta = float(beta)
        reward_mode = str(reward_mode).lower()
        if reward_mode not in {"none", "bonus", "gate"}:
            raise ValueError(
                "Spatial exploration reward mode must be 'none', 'bonus', or 'gate'"
            )
        minimum_reward_factor = float(minimum_reward_factor)
        if not 0.0 <= minimum_reward_factor <= 1.0:
            raise ValueError("Spatial exploration minimum reward factor must be in [0, 1]")
        candidate_root_indices = []
        candidate_paths = []
        candidate_slices = []
        for root_index, diagnostics in enumerate(root_action_diagnostics):
            candidate_actions = [int(row.get("action", -1)) for row in diagnostics]
            start = len(candidate_paths)
            candidate_root_indices.extend([root_index] * len(candidate_actions))
            candidate_paths.extend([[action] for action in candidate_actions])
            candidate_slices.append((start, len(candidate_paths), candidate_actions))
        candidate_scores = self.score_paths(
            roots,
            candidate_root_indices,
            candidate_paths,
            maximum_depth=1,
        )

        for root_index, (root, action, planned_action, candidate_slice) in enumerate(
            zip(roots, actions, planned_actions, candidate_slices)
        ):
            action = int(action)
            selected_search_diagnostics = next(
                (
                    row
                    for row in root_action_diagnostics[root_index]
                    if int(row.get("action", -1)) == action
                ),
                {},
            )
            start, end, candidate_actions = candidate_slice
            candidate_novelty_by_action = {
                candidate_action: float(score["novelty"])
                for candidate_action, score in zip(
                    candidate_actions, candidate_scores[start:end]
                )
            }
            candidate_normalized_novelty_by_action = {
                candidate_action: float(score["normalized_novelty"])
                for candidate_action, score in zip(
                    candidate_actions, candidate_scores[start:end]
                )
            }
            candidate_by_action = {
                candidate_action: beta * novelty
                for candidate_action, novelty in candidate_novelty_by_action.items()
            }
            candidate_factor_by_action = {
                candidate_action: (
                    minimum_reward_factor
                    + (1.0 - minimum_reward_factor) * novelty
                )
                for candidate_action, novelty in (
                    candidate_normalized_novelty_by_action.items()
                )
            }
            candidate_values = list(candidate_by_action.values())
            prediction = self._path_cache[(root.env_id, (action,))]
            selected_bonus = float(
                candidate_by_action.get(action, beta * prediction["novelty"])
            )
            selected_factor = float(
                candidate_factor_by_action.get(
                    action,
                    minimum_reward_factor
                    + (1.0 - minimum_reward_factor)
                    * float(prediction["normalized_novelty"]),
                )
            )
            candidate_factors = list(candidate_factor_by_action.values())
            if reward_mode == "bonus":
                selected_adjustment = selected_bonus
                candidate_adjustments = candidate_values
            elif reward_mode == "gate":
                selected_adjustment = selected_factor
                candidate_adjustments = candidate_factors
            else:
                selected_adjustment = 1.0
                candidate_adjustments = [1.0 for _ in candidate_actions]
            selected_rank = 1 + sum(
                value > selected_adjustment + 1e-12
                for value in candidate_adjustments
            )
            self._states[root.env_id]["pending"] = {
                "env_id": root.env_id,
                "selected_action": action,
                "planned_action": int(planned_action),
                "reward_mode": reward_mode,
                "predicted_raw_novelty": float(prediction["raw_novelty"]),
                "predicted_novelty": float(prediction["novelty"]),
                "predicted_normalized_novelty": float(
                    prediction["normalized_novelty"]
                ),
                "selected_bonus": (
                    selected_bonus if reward_mode == "bonus" else 0.0
                ),
                "selected_reward_factor": (
                    selected_factor if reward_mode == "gate" else 1.0
                ),
                "selected_adjustment": selected_adjustment,
                "candidate_bonus_mean": float(
                    np.mean(candidate_values)
                    if reward_mode == "bonus" and candidate_values else 0.0
                ),
                "candidate_bonus_spread": float(
                    max(candidate_values) - min(candidate_values)
                    if reward_mode == "bonus" and candidate_values else 0.0
                ),
                "candidate_reward_factor_mean": float(
                    np.mean(candidate_factors) if candidate_factors else 1.0
                ),
                "candidate_reward_factor_spread": float(
                    max(candidate_factors) - min(candidate_factors)
                    if candidate_factors else 0.0
                ),
                "candidate_adjustment_mean": float(
                    np.mean(candidate_adjustments)
                    if candidate_adjustments else selected_adjustment
                ),
                "candidate_adjustment_spread": float(
                    max(candidate_adjustments) - min(candidate_adjustments)
                    if candidate_adjustments else 0.0
                ),
                "candidate_novelty_mean": float(
                    np.mean(list(candidate_novelty_by_action.values()))
                    if candidate_novelty_by_action else prediction["novelty"]
                ),
                "candidate_novelty_spread": float(
                    max(candidate_novelty_by_action.values())
                    - min(candidate_novelty_by_action.values())
                    if candidate_novelty_by_action else 0.0
                ),
                "selected_bonus_rank": int(selected_rank),
                "selected_bonus_is_argmax": float(selected_rank == 1),
                "selected_adjustment_rank": int(selected_rank),
                "selected_adjustment_is_argmax": float(selected_rank == 1),
                "search_intrinsic_reward_before_transform": float(
                    selected_search_diagnostics.get(
                        "intrinsic_reward_before_path_transform", float("nan")
                    )
                ),
                "search_intrinsic_reward_before_modifier": float(
                    selected_search_diagnostics.get(
                        "intrinsic_reward_before_path_modifier", float("nan")
                    )
                ),
                "search_intrinsic_path_reward_modifier": float(
                    selected_search_diagnostics.get(
                        "intrinsic_path_reward_modifier", float("nan")
                    )
                ),
                "search_intrinsic_immediate_reward": float(
                    selected_search_diagnostics.get(
                        "intrinsic_immediate_reward", float("nan")
                    )
                ),
                "search_intrinsic_reward_return_mean": float(
                    selected_search_diagnostics.get(
                        "intrinsic_reward_return_mean", float("nan")
                    )
                ),
                "search_intrinsic_leaf_value_contribution_mean": float(
                    selected_search_diagnostics.get(
                        "intrinsic_leaf_value_contribution_mean", float("nan")
                    )
                ),
                "memory_size": int(root.memory_keys.shape[0]),
                "memory_scale": float(root.memory_scale),
            }

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._path_cache.clear()
        self._path_latent_cache.clear()
        self._path_belief_cache.clear()
        if env_ids is None:
            self._states.clear()
            return
        for env_id in env_ids:
            self._states.pop(int(env_id), None)

    def consume_records(self) -> List[Dict[str, Any]]:
        records = list(self._records)
        self._records.clear()
        return records
