"""Frozen exploration-world-model diagnostics for executed MuZero trajectories."""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    positive = int(labels.sum())
    negative = int(labels.size - positive)
    if positive == 0 or negative == 0:
        return float("nan")
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    rank_sum = float(ranks[labels == 1].sum())
    return (rank_sum - positive * (positive + 1) / 2.0) / (positive * negative)


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.size < 2 or float(left.std()) <= 1e-12 or float(right.std()) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _append_csv(path: str, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.isfile(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _exact_state_ids(observations_uint8: np.ndarray) -> np.ndarray:
    mapping: Dict[bytes, int] = {}
    state_ids = np.empty(observations_uint8.shape[0], dtype=np.int64)
    for index, observation in enumerate(observations_uint8):
        key = np.ascontiguousarray(observation).tobytes()
        mapping.setdefault(key, len(mapping))
        state_ids[index] = mapping[key]
    return state_ids


def _first_visit_labels(values: np.ndarray) -> np.ndarray:
    seen = set()
    labels = np.zeros(len(values), dtype=np.int64)
    for index, value in enumerate(values):
        key = tuple(int(item) for item in np.asarray(value).reshape(-1))
        labels[index] = int(key not in seen)
        seen.add(key)
    return labels


def _novelty_targets(state_ids: np.ndarray, horizon: int) -> np.ndarray:
    """Build root-history revisit, branch-revisit, and novel labels."""
    state_ids = np.asarray(state_ids, dtype=np.int64).reshape(-1)
    root_count = state_ids.size - int(horizon)
    if root_count <= 0:
        raise ValueError("Episode is shorter than the requested horizon")
    first_visit: Dict[int, int] = {}
    for index, state_id in enumerate(state_ids.tolist()):
        first_visit.setdefault(int(state_id), index)
    targets = np.empty((root_count, int(horizon)), dtype=np.uint8)
    for root in range(root_count):
        branch_states = set()
        for depth in range(1, int(horizon) + 1):
            endpoint = int(state_ids[root + depth])
            if first_visit[endpoint] <= root:
                label = 0
            elif endpoint in branch_states:
                label = 1
            else:
                label = 2
            targets[root, depth - 1] = label
            branch_states.add(endpoint)
    return targets


def complete_collection_episodes(
    new_data: List[Any],
    frame_stack_num: int,
) -> List[Dict[str, Any]]:
    """Reassemble complete episodes from collector game-segment blocks."""
    segments, metadata = new_data
    grouped: Dict[tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
    for game_segment, meta in zip(segments, metadata):
        valid_steps = min(
            int(meta.get("segment_length", len(game_segment.action_segment))),
            len(game_segment.action_segment),
            len(game_segment.reward_segment),
            len(getattr(game_segment, "environment_state_segment", [])),
        )
        observations = np.asarray(game_segment.obs_segment)
        if valid_steps <= 0 or observations.shape[0] < frame_stack_num + valid_steps:
            continue
        states = []
        for step in range(valid_steps + 1):
            frames = np.asarray(
                observations[step : step + frame_stack_num], dtype=np.float32
            )
            states.append(frames.reshape(-1, *frames.shape[-2:]))
        initial_environment_state = getattr(
            game_segment, "environment_initial_state", None
        )
        transition_states = np.asarray(
            game_segment.environment_state_segment[:valid_steps], dtype=np.int64
        )
        if initial_environment_state is None or transition_states.ndim != 2:
            continue
        environment_states = np.concatenate(
            [
                np.asarray(initial_environment_state, dtype=np.int64).reshape(1, -1),
                transition_states,
            ],
            axis=0,
        )
        key = (int(meta.get("env_id", -1)), int(meta.get("env_episode_index", -1)))
        grouped[key].append(
            {
                "segment_index": int(meta.get("episode_segment_index", 0)),
                "observations": np.stack(states),
                "actions": np.asarray(
                    game_segment.action_segment[:valid_steps], dtype=np.int64
                ).reshape(-1),
                "environment_states": environment_states,
                "done": bool(meta.get("done", False)),
                "collection_episode_index": int(
                    meta.get("collection_episode_index", -1)
                ),
            }
        )

    episodes = []
    for (env_id, env_episode_index), blocks in grouped.items():
        blocks.sort(key=lambda item: item["segment_index"])
        if not blocks[-1]["done"] or [item["segment_index"] for item in blocks] != list(
            range(len(blocks))
        ):
            continue
        observations = [blocks[0]["observations"]]
        environment_states = [blocks[0]["environment_states"]]
        actions = []
        for block_index, block in enumerate(blocks):
            if block_index > 0:
                observations.append(block["observations"][1:])
                environment_states.append(block["environment_states"][1:])
            actions.append(block["actions"])
        observations_array = np.concatenate(observations, axis=0)
        actions_array = np.concatenate(actions, axis=0)
        environment_states_array = np.concatenate(environment_states, axis=0)
        step_count = min(
            len(actions_array),
            len(observations_array) - 1,
            len(environment_states_array) - 1,
        )
        if step_count <= 0:
            continue
        episodes.append(
            {
                "observations": observations_array[: step_count + 1],
                "actions": actions_array[:step_count],
                "environment_states": environment_states_array[: step_count + 1],
                "env_id": env_id,
                "env_episode_index": env_episode_index,
                "collection_episode_index": blocks[-1]["collection_episode_index"],
            }
        )
    return episodes


class SpatialExplorationShadowEvaluator:
    """Evaluate a frozen spatial model without feeding results back to MuZero."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str,
        horizon: int = 5,
        roots_per_episode: int = 128,
        root_batch_size: int = 128,
        dynamics_history_window: int = 128,
        revisit_scoring_mode: str = "model",
        belief_ablation: bool = False,
        export_dataset_path: str | None = None,
        export_split_counts: Sequence[int] | None = None,
    ) -> None:
        try:
            from experiments.run_branching_rooms_spatial_exploration_world_model import (
                SpatialExplorationWorldModel,
            )
        except ModuleNotFoundError:
            from run_branching_rooms_spatial_exploration_world_model import (
                SpatialExplorationWorldModel,
            )

        if not checkpoint_path or not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"Spatial exploration shadow checkpoint does not exist: {checkpoint_path}"
            )
        self.checkpoint_path = str(checkpoint_path)
        self.device = torch.device(device)
        self.horizon = int(horizon)
        self.roots_per_episode = max(1, int(roots_per_episode))
        self.root_batch_size = max(1, int(root_batch_size))
        self.dynamics_history_window = max(1, int(dynamics_history_window))
        self.revisit_scoring_mode = str(revisit_scoring_mode).lower()
        self.belief_ablation = bool(belief_ablation)
        if self.revisit_scoring_mode not in {"model", "observation_key"}:
            raise ValueError(
                "revisit_scoring_mode must be 'model' or 'observation_key'"
            )
        self.export_dataset_path = (
            None if not export_dataset_path else str(export_dataset_path)
        )
        self.export_split_counts = (
            None
            if export_split_counts is None
            else tuple(int(value) for value in export_split_counts)
        )
        if self.export_split_counts is not None:
            if len(self.export_split_counts) != 4:
                raise ValueError("export_split_counts must contain four values")
            if any(value < 0 for value in self.export_split_counts):
                raise ValueError("export_split_counts cannot contain negatives")
        checkpoint = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=False
        )
        arguments = checkpoint.get("arguments", {})
        checkpoint_horizon = int(arguments.get("horizon", self.horizon))
        if self.horizon > checkpoint_horizon:
            raise ValueError(
                f"Requested shadow horizon {self.horizon} exceeds checkpoint "
                f"horizon {checkpoint_horizon}"
            )
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
                arguments.get("memory_key_mode", "observation")
            ),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.checkpoint_epoch = int(checkpoint.get("epoch", -1))

    @classmethod
    def from_live_model(
        cls,
        model,
        device: str,
        horizon: int = 5,
        roots_per_episode: int = 128,
        root_batch_size: int = 128,
        dynamics_history_window: int = 128,
        revisit_scoring_mode: str = "model",
        belief_ablation: bool = False,
    ) -> "SpatialExplorationShadowEvaluator":
        """Build a read-only evaluator around the current online model."""
        evaluator = cls.__new__(cls)
        evaluator.checkpoint_path = None
        evaluator.device = torch.device(device)
        evaluator.horizon = int(horizon)
        evaluator.roots_per_episode = max(1, int(roots_per_episode))
        evaluator.root_batch_size = max(1, int(root_batch_size))
        evaluator.dynamics_history_window = max(
            1, int(dynamics_history_window)
        )
        evaluator.revisit_scoring_mode = str(revisit_scoring_mode).lower()
        evaluator.belief_ablation = bool(belief_ablation)
        if evaluator.revisit_scoring_mode not in {
            "model",
            "observation_key",
        }:
            raise ValueError(
                "revisit_scoring_mode must be 'model' or 'observation_key'"
            )
        evaluator.export_dataset_path = None
        evaluator.export_split_counts = None
        evaluator.model = model
        evaluator.checkpoint_epoch = -1
        return evaluator

    def _prepare_observations(self, observations: np.ndarray) -> torch.Tensor:
        tensor = torch.as_tensor(observations, device=self.device).float()
        if float(tensor.amax()) > 1.5:
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
    def _sample_roots(root_count: int, requested_count: int) -> np.ndarray:
        if root_count <= requested_count:
            return np.arange(root_count, dtype=np.int64)
        return np.unique(
            np.linspace(0, root_count - 1, requested_count, dtype=np.int64)
        )

    def _bounded_root_beliefs(
        self,
        keys: torch.Tensor,
        actions: torch.Tensor,
        roots: torch.Tensor,
    ) -> torch.Tensor:
        """Rebuild root beliefs from a bounded suffix of each real history."""
        starts = torch.clamp(
            roots - self.dynamics_history_window + 1,
            min=0,
        )
        lengths = roots - starts + 1
        belief = torch.zeros(
            roots.shape[0],
            self.model.belief_dim,
            device=keys.device,
            dtype=keys.dtype,
        )
        start_actions = torch.full(
            (roots.shape[0],),
            self.model.action_space_size,
            device=actions.device,
            dtype=torch.long,
        )
        for offset in range(int(lengths.max())):
            active = offset < lengths
            observation_indices = starts + offset
            previous_actions = (
                start_actions
                if offset == 0
                else actions[observation_indices - 1]
            )
            next_belief = self.model.belief_cell(
                torch.cat(
                    [
                        keys[observation_indices],
                        self.model.action_embedding(previous_actions),
                    ],
                    dim=-1,
                ),
                belief,
            )
            belief = torch.where(active[:, None], next_belief, belief)
        return belief

    def export_episodes(self, episodes: Sequence[Mapping[str, Any]]) -> str | None:
        if self.export_dataset_path is None:
            return None
        if self.export_split_counts is None:
            raise ValueError(
                "A trajectory export requires explicit train/validation/test/heldout counts"
            )
        episode_count = len(episodes)
        if sum(self.export_split_counts) != episode_count:
            raise ValueError(
                f"Export split counts sum to {sum(self.export_split_counts)}, "
                f"but collection produced {episode_count} complete episodes"
            )
        lengths = {len(episode["actions"]) for episode in episodes}
        if len(lengths) != 1:
            raise ValueError("Exported episodes must have equal lengths")

        observations = []
        actions = []
        poses = []
        observation_targets = []
        pose_targets = []
        episode_ids = []
        for episode in episodes:
            prepared = self._prepare_observations(episode["observations"])
            observation_uint8 = torch.round(prepared * 255.0).to(
                torch.uint8
            ).cpu().numpy()
            episode_actions = np.asarray(episode["actions"], dtype=np.uint8)
            episode_poses = np.asarray(
                episode["environment_states"], dtype=np.int16
            )[:, :3]
            observations.append(observation_uint8)
            actions.append(episode_actions)
            poses.append(episode_poses)
            observation_targets.append(
                _novelty_targets(
                    _exact_state_ids(observation_uint8), self.horizon
                )
            )
            pose_targets.append(
                _novelty_targets(_exact_state_ids(episode_poses), self.horizon)
            )
            episode_ids.append(
                int(episode.get("env_id", -1)) * 1_000_000
                + int(episode.get("env_episode_index", -1))
            )

        permutation = np.random.default_rng(7102026).permutation(episode_count)
        split_ids = np.empty(episode_count, dtype=np.uint8)
        offset = 0
        for split_id, count in enumerate(self.export_split_counts):
            split_ids[permutation[offset : offset + count]] = split_id
            offset += count
        metadata = {
            "seed": 7102026,
            "episode_length": int(next(iter(lengths))),
            "horizon": int(self.horizon),
            "tile_size": 4,
            "split_names": ["train", "validation", "test", "heldout_policy"],
            "policy_names": ["muzero_checkpoint"],
            "target_source": "executed_muzero_trajectory_recurrence",
            "counterfactual_transition_queries": 0,
            "physical_position_used_for_training": False,
            "split_counts": list(self.export_split_counts),
            "split_assignment_seed": 7102026,
        }
        output_path = Path(self.export_dataset_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp.npz")
        np.savez_compressed(
            temporary_path,
            observations=np.stack(observations),
            actions=np.stack(actions),
            poses=np.stack(poses),
            observation_targets=np.stack(observation_targets),
            pose_targets=np.stack(pose_targets),
            split_ids=split_ids,
            policy_ids=np.zeros(episode_count, dtype=np.uint8),
            episode_seeds=np.asarray(episode_ids, dtype=np.int64),
            metadata_json=np.asarray(json.dumps(metadata)),
        )
        temporary_path.replace(output_path)
        manifest_path = output_path.with_suffix(".manifest.json")
        manifest_path.write_text(
            json.dumps(
                {
                    "dataset": str(output_path),
                    "episode_count": episode_count,
                    "observation_shape": list(np.asarray(observations).shape),
                    "metadata": metadata,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return str(output_path)

    def evaluate_episodes(
        self,
        episodes: Sequence[Mapping[str, Any]],
        collection_phase: int,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        rows: List[Dict[str, Any]] = []
        root_rows: List[Dict[str, Any]] = []
        with torch.inference_mode():
            for episode_index, episode in enumerate(episodes):
                observations = self._prepare_observations(episode["observations"])
                actions = torch.as_tensor(
                    episode["actions"], device=self.device, dtype=torch.long
                )
                step_count = int(actions.shape[0])
                root_count = step_count - self.horizon + 1
                if root_count <= 0:
                    continue
                encoded = self.model.encode_trajectory(
                    observations[None], actions[None]
                )
                latents = encoded.latents[0]
                observation_keys = encoded.observation_keys[0]
                keys = encoded.keys[0]
                beliefs = encoded.beliefs[0]
                roots_np = self._sample_roots(root_count, self.roots_per_episode)
                observations_uint8 = torch.round(observations * 255.0).to(
                    torch.uint8
                ).cpu().numpy()
                state_ids = _exact_state_ids(observations_uint8)
                rgb_first_visit = _first_visit_labels(state_ids[:, None])
                positions = np.asarray(episode["environment_states"], dtype=np.int64)[
                    :, :2
                ]
                position_first_visit = _first_visit_labels(positions)

                for batch_start in range(0, len(roots_np), self.root_batch_size):
                    batch_roots_np = roots_np[
                        batch_start : batch_start + self.root_batch_size
                    ]
                    batch_roots = torch.as_tensor(
                        batch_roots_np, device=self.device, dtype=torch.long
                    )
                    suffix_actions = torch.stack(
                        [actions[batch_roots + depth] for depth in range(self.horizon)],
                        dim=1,
                    )
                    if self.model.memory_key_mode in (
                        "belief",
                        "belief_revisit",
                    ):
                        root_beliefs = beliefs[batch_roots]
                    else:
                        root_beliefs = self._bounded_root_beliefs(
                            observation_keys,
                            actions,
                            batch_roots,
                        )
                    rollout = self.model.rollout(
                        latents[batch_roots],
                        root_beliefs,
                        suffix_actions,
                    )
                    predicted_images = (
                        self.model.decode_rollout(rollout)
                        if self.model.supports_reconstruction
                        else None
                    )

                    for batch_index, root in enumerate(batch_roots_np.tolist()):
                        root_prediction_sum = 0.0
                        root_teacher_sum = 0.0
                        root_actual_novel_count = 0
                        root_actual_position_novel_count = 0
                        for depth_index in range(self.horizon):
                            depth = depth_index + 1
                            endpoint = root + depth
                            predicted_key = rollout.keys[batch_index, depth_index]
                            teacher_key = keys[endpoint]
                            belief_ablation_scores = {}
                            if self.revisit_scoring_mode == "observation_key":
                                observation_query = rollout.observation_keys[
                                    batch_index, depth_index
                                ]
                                real_history = observation_keys[: root + 1]
                                predicted_memory = real_history
                                if depth_index > 0:
                                    predicted_memory = torch.cat(
                                        [
                                            real_history,
                                            rollout.observation_keys[
                                                batch_index, :depth_index
                                            ],
                                        ],
                                        dim=0,
                                    )
                                predicted_novelty = 1.0 - torch.mv(
                                    predicted_memory, observation_query
                                ).amax()
                                teacher_memory = observation_keys[:endpoint]
                                teacher_novelty = 1.0 - torch.mv(
                                    teacher_memory, observation_keys[endpoint]
                                ).amax()
                                all_similarity = torch.mv(
                                    observation_keys, observation_query
                                )
                            elif self.model.memory_key_mode == "belief_revisit":
                                belief_memory = beliefs[: root + 1]
                                if depth_index > 0:
                                    belief_memory = torch.cat(
                                        [
                                            belief_memory,
                                            rollout.beliefs[
                                                batch_index, :depth_index
                                            ],
                                        ],
                                        dim=0,
                                    )
                                predicted_novelty = self.model.revisit_novelty(
                                    rollout.beliefs[batch_index, depth_index],
                                    belief_memory,
                                )
                                if self.belief_ablation:
                                    real_query = beliefs[endpoint]
                                    real_memory = beliefs[:endpoint]
                                    predicted_query = rollout.beliefs[
                                        batch_index, depth_index
                                    ]
                                    belief_ablation_scores = {
                                        "real_query_real_memory_novelty": float(
                                            self.model.revisit_novelty(
                                                real_query,
                                                real_memory,
                                            )
                                        ),
                                        "predicted_query_real_memory_novelty": float(
                                            self.model.revisit_novelty(
                                                predicted_query,
                                                real_memory,
                                            )
                                        ),
                                        "real_query_predicted_memory_novelty": float(
                                            self.model.revisit_novelty(
                                                real_query,
                                                belief_memory,
                                            )
                                        ),
                                    }
                                teacher_novelty = float(
                                    not np.any(
                                        state_ids[:endpoint]
                                        == state_ids[endpoint]
                                    )
                                )
                                all_similarity = torch.sigmoid(
                                    self.model.revisit_logits(
                                        rollout.beliefs[
                                            batch_index, depth_index
                                        ],
                                        beliefs,
                                    )
                                )
                            else:
                                real_history = keys[: root + 1]
                                predicted_memory = real_history
                                if depth_index > 0:
                                    predicted_memory = torch.cat(
                                        [
                                            real_history,
                                            rollout.keys[
                                                batch_index, :depth_index
                                            ],
                                        ],
                                        dim=0,
                                    )
                                predicted_novelty = 1.0 - torch.mv(
                                    predicted_memory, predicted_key
                                ).amax()
                                teacher_memory = keys[:endpoint]
                                teacher_novelty = 1.0 - torch.mv(
                                    teacher_memory, teacher_key
                                ).amax()
                                all_similarity = torch.mv(keys, predicted_key)
                            alignment = F.cosine_similarity(
                                predicted_key[None], teacher_key[None], dim=-1
                            )[0]
                            observation_key_alignment = F.cosine_similarity(
                                rollout.observation_keys[
                                    batch_index, depth_index
                                ][None],
                                observation_keys[endpoint][None],
                                dim=-1,
                            )[0]
                            nearest_index = int(all_similarity.argmax())
                            retrieval_correct = int(
                                state_ids[nearest_index] == state_ids[endpoint]
                            )
                            rollout_mae = float("nan")
                            if predicted_images is not None:
                                rollout_mae = float(
                                    (
                                        predicted_images[batch_index, depth_index]
                                        - self.model.reconstruction_targets(
                                            observations[endpoint]
                                        )
                                    )
                                    .abs()
                                    .mean()
                                )
                            rgb_novel = int(rgb_first_visit[endpoint])
                            position_novel = int(position_first_visit[endpoint])
                            predicted_value = float(predicted_novelty)
                            root_prediction_sum += predicted_value
                            root_teacher_sum += float(teacher_novelty)
                            root_actual_novel_count += rgb_novel
                            root_actual_position_novel_count += position_novel
                            rows.append(
                                {
                                    "collection_phase": int(collection_phase),
                                    "dynamics_history_window": int(
                                        self.dynamics_history_window
                                    ),
                                    "episode_index_in_phase": int(episode_index),
                                    "collection_episode_index": int(
                                        episode.get("collection_episode_index", -1)
                                    ),
                                    "env_id": int(episode.get("env_id", -1)),
                                    "env_episode_index": int(
                                        episode.get("env_episode_index", -1)
                                    ),
                                    "root_timestep": int(root),
                                    "depth": int(depth),
                                    "endpoint_timestep": int(endpoint),
                                    "action_prefix": json.dumps(
                                        suffix_actions[batch_index, :depth].cpu().tolist(),
                                        separators=(",", ":"),
                                    ),
                                    "predicted_novelty": predicted_value,
                                    "teacher_novelty": float(teacher_novelty),
                                    "actual_rgb_novel": rgb_novel,
                                    "actual_position_novel": position_novel,
                                    "predicted_target_cosine": float(alignment),
                                    "predicted_observation_key_cosine": float(
                                        observation_key_alignment
                                    ),
                                    "rgb_retrieval_correct": retrieval_correct,
                                    "rollout_rgb_mae": rollout_mae,
                                    **belief_ablation_scores,
                                }
                            )
                        root_rows.append(
                            {
                                "collection_phase": int(collection_phase),
                                "dynamics_history_window": int(
                                    self.dynamics_history_window
                                ),
                                "episode_index_in_phase": int(episode_index),
                                "collection_episode_index": int(
                                    episode.get("collection_episode_index", -1)
                                ),
                                "env_id": int(episode.get("env_id", -1)),
                                "root_timestep": int(root),
                                "action_suffix": json.dumps(
                                    suffix_actions[batch_index].cpu().tolist(),
                                    separators=(",", ":"),
                                ),
                                "predicted_novelty_sum": root_prediction_sum,
                                "teacher_novelty_sum": root_teacher_sum,
                                "actual_rgb_novel_count": root_actual_novel_count,
                                "actual_position_novel_count": (
                                    root_actual_position_novel_count
                                ),
                            }
                        )
        return rows, root_rows

    @staticmethod
    def summarize(
        rows: Sequence[Mapping[str, Any]],
        collection_phase: int,
        root_rows: Sequence[Mapping[str, Any]] = (),
    ) -> List[Dict[str, Any]]:
        summaries = []
        depth_values = sorted({int(row["depth"]) for row in rows})
        for depth in [0, *depth_values]:
            selected = [
                row for row in rows if depth == 0 or int(row["depth"]) == depth
            ]
            predicted = np.asarray(
                [row["predicted_novelty"] for row in selected], dtype=np.float64
            )
            teacher = np.asarray(
                [row["teacher_novelty"] for row in selected], dtype=np.float64
            )
            rgb_labels = np.asarray(
                [row["actual_rgb_novel"] for row in selected], dtype=np.int64
            )
            position_labels = np.asarray(
                [row["actual_position_novel"] for row in selected], dtype=np.int64
            )
            novel_mask = rgb_labels == 1
            revisit_mask = ~novel_mask
            root_predicted = np.asarray(
                [row["predicted_novelty_sum"] for row in root_rows],
                dtype=np.float64,
            )
            root_teacher = np.asarray(
                [row["teacher_novelty_sum"] for row in root_rows],
                dtype=np.float64,
            )
            root_rgb = np.asarray(
                [row["actual_rgb_novel_count"] for row in root_rows],
                dtype=np.float64,
            )
            root_position = np.asarray(
                [row["actual_position_novel_count"] for row in root_rows],
                dtype=np.float64,
            )
            summary = {
                    "collection_phase": int(collection_phase),
                    "dynamics_history_window": int(
                        rows[0]["dynamics_history_window"]
                    ),
                    "depth": int(depth),
                    "sample_count": len(selected),
                    "rgb_novel_fraction": float(rgb_labels.mean()),
                    "position_novel_fraction": float(position_labels.mean()),
                    "predicted_novelty_mean": float(predicted.mean()),
                    "predicted_novelty_novel_mean": float(
                        predicted[novel_mask].mean() if novel_mask.any() else np.nan
                    ),
                    "predicted_novelty_revisit_mean": float(
                        predicted[revisit_mask].mean() if revisit_mask.any() else np.nan
                    ),
                    "predicted_rgb_novelty_auc": _binary_auc(rgb_labels, predicted),
                    "predicted_position_novelty_auc": _binary_auc(
                        position_labels, predicted
                    ),
                    "teacher_rgb_novelty_auc": _binary_auc(rgb_labels, teacher),
                    "teacher_position_novelty_auc": _binary_auc(
                        position_labels, teacher
                    ),
                    "predicted_teacher_correlation": _correlation(
                        predicted, teacher
                    ),
                    "predicted_target_cosine_mean": float(
                        np.mean([row["predicted_target_cosine"] for row in selected])
                    ),
                    "predicted_observation_key_cosine_mean": float(
                        np.mean(
                            [
                                row["predicted_observation_key_cosine"]
                                for row in selected
                            ]
                        )
                    ),
                    "rgb_retrieval_accuracy": float(
                        np.mean([row["rgb_retrieval_correct"] for row in selected])
                    ),
                    "rollout_rgb_mae": float(
                        np.mean([row["rollout_rgb_mae"] for row in selected])
                    ),
                    "root_predicted_rgb_count_correlation": (
                        _correlation(root_predicted, root_rgb)
                        if depth == 0 and root_rows
                        else float("nan")
                    ),
                    "root_predicted_position_count_correlation": (
                        _correlation(root_predicted, root_position)
                        if depth == 0 and root_rows
                        else float("nan")
                    ),
                    "root_teacher_rgb_count_correlation": (
                        _correlation(root_teacher, root_rgb)
                        if depth == 0 and root_rows
                        else float("nan")
                    ),
                }
            for name in (
                "real_query_real_memory",
                "predicted_query_real_memory",
                "real_query_predicted_memory",
            ):
                field = f"{name}_novelty"
                if field not in selected[0]:
                    continue
                values = np.asarray(
                    [row[field] for row in selected], dtype=np.float64
                )
                summary[f"{name}_novelty_mean"] = float(values.mean())
                summary[f"{name}_novel_mean"] = float(
                    values[novel_mask].mean() if novel_mask.any() else np.nan
                )
                summary[f"{name}_revisit_mean"] = float(
                    values[revisit_mask].mean() if revisit_mask.any() else np.nan
                )
                summary[f"{name}_rgb_auc"] = _binary_auc(rgb_labels, values)
            summaries.append(summary)
        return summaries

    def evaluate_collection(
        self,
        new_data: List[Any],
        frame_stack_num: int,
        collection_phase: int,
        diagnostics_dir: str,
        file_prefix: str = "spatial_exploration_shadow",
    ) -> Dict[str, Any]:
        episodes = complete_collection_episodes(new_data, int(frame_stack_num))
        exported_dataset = self.export_episodes(episodes)
        rows, root_rows = self.evaluate_episodes(episodes, int(collection_phase))
        summaries = (
            self.summarize(rows, int(collection_phase), root_rows)
            if rows
            else []
        )
        _append_csv(
            os.path.join(diagnostics_dir, f"{file_prefix}_steps.csv"),
            rows,
        )
        _append_csv(
            os.path.join(diagnostics_dir, f"{file_prefix}_roots.csv"),
            root_rows,
        )
        _append_csv(
            os.path.join(diagnostics_dir, f"{file_prefix}_summary.csv"),
            summaries,
        )
        return {
            "episode_count": len(episodes),
            "root_count": len(root_rows),
            "step_count": len(rows),
            "summaries": summaries,
            "exported_dataset": exported_dataset,
        }
