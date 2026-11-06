#!/usr/bin/env python3
"""Evaluate reconstructive episodic memory on fresh Branching Rooms paths.

This is a checkpoint-only diagnostic. It does not update any parameters. Fresh
action trajectories are generated from the environment's exact transition
table, while that table is used only to score the learned model's predictions.
The learned model receives observations, action/reward history, and action
suffixes in exactly the same form as during training.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

if __package__:
    from experiments.diagnose_branching_rooms_raw_latent_geometry import (
        ACTION_NAMES,
        BranchingRoomsStateSpace,
        build_state_space,
    )
else:
    from diagnose_branching_rooms_raw_latent_geometry import (
        ACTION_NAMES,
        BranchingRoomsStateSpace,
        build_state_space,
    )
from lzero.policy.predictive_episodic_memory import ReconstructiveEpisodicMemory


@dataclass(frozen=True)
class HeldoutTrajectories:
    pose_indices: np.ndarray
    actions: np.ndarray
    observation_groups: np.ndarray
    episodic_rewards: np.ndarray
    pose_first_visit_rewards: np.ndarray
    position_first_visit_rewards: np.ndarray
    policy_names: tuple[str, ...]


class _BehaviourPolicy:
    def __init__(self, name: str, rng: np.random.Generator) -> None:
        self.name = str(name)
        self.rng = rng
        self.remaining = 0
        self.current_action = 2
        self.macro_actions: deque[int] = deque()

    def action(self) -> int:
        if self.name == "uniform":
            return int(self.rng.integers(0, 3))
        if self.name == "persistent":
            if self.remaining <= 0:
                self.current_action = int(
                    self.rng.choice(np.asarray([0, 1, 2]), p=[0.2, 0.2, 0.6])
                )
                self.remaining = (
                    int(self.rng.integers(2, 9))
                    if self.current_action == 2
                    else 1
                )
            self.remaining -= 1
            return self.current_action
        if self.name != "macro":
            raise ValueError(f"Unknown behaviour policy: {self.name}")
        if not self.macro_actions:
            macros = (
                (2, 2, 2, 0, 2),
                (2, 2, 1, 2, 2),
                (0, 2, 1, 2),
                (1, 2, 0, 2),
                (2, 0, 0, 2, 2),
                (2, 1, 1, 2, 2),
            )
            self.macro_actions.extend(
                macros[int(self.rng.integers(0, len(macros)))]
            )
        return int(self.macro_actions.popleft())


def _load_config(path: Path) -> Dict:
    spec = importlib.util.spec_from_file_location(
        "branching_rooms_reconstructive_config", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import config from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.exp_config


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    return torch.device(name)


def _resolve_checkpoint(run_dir: Path, checkpoint: str | None) -> Path:
    checkpoint_dir = run_dir / "ckpt"
    if checkpoint:
        path = Path(checkpoint)
        if not path.is_absolute() and not path.exists():
            path = checkpoint_dir / checkpoint
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    candidates = []
    for path in checkpoint_dir.glob("iteration_*.pth.tar"):
        try:
            iteration = int(path.name.split("iteration_", 1)[1].split(".", 1)[0])
        except ValueError:
            continue
        candidates.append((iteration, path))
    if candidates:
        return max(candidates)[1]
    best = checkpoint_dir / "ckpt_best.pth.tar"
    if best.exists():
        return best
    raise FileNotFoundError(f"No checkpoint found under {checkpoint_dir}")


def first_visit_rewards(observation_groups: np.ndarray) -> np.ndarray:
    groups = np.asarray(observation_groups, dtype=np.int64)
    if groups.ndim != 1 or groups.size < 1:
        raise ValueError("observation_groups must be a non-empty vector")
    seen = {int(groups[0])}
    rewards = np.zeros(groups.size - 1, dtype=np.float32)
    for index, group in enumerate(groups[1:]):
        group = int(group)
        rewards[index] = float(group not in seen)
        seen.add(group)
    return rewards


def generate_heldout_trajectories(
    state_space: BranchingRoomsStateSpace,
    episode_count: int,
    episode_length: int,
    seed: int,
) -> HeldoutTrajectories:
    pose_to_index = {
        tuple(pose): index for index, pose in enumerate(state_space.poses.tolist())
    }
    start_index = pose_to_index[(11, 11, 0)]
    policies = ("uniform", "persistent", "macro")
    pose_indices = np.empty(
        (int(episode_count), int(episode_length) + 1), dtype=np.int64
    )
    actions = np.empty((int(episode_count), int(episode_length)), dtype=np.int64)
    policy_names: List[str] = []
    for episode in range(int(episode_count)):
        rng = np.random.default_rng(int(seed) + 104729 * episode)
        policy_name = policies[episode % len(policies)]
        policy = _BehaviourPolicy(policy_name, rng)
        pose_indices[episode, 0] = start_index
        for step in range(int(episode_length)):
            action = policy.action()
            actions[episode, step] = action
            pose_indices[episode, step + 1] = state_space.transition[
                pose_indices[episode, step], action
            ]
        policy_names.append(policy_name)
    groups = state_space.observation_group[pose_indices]
    rewards = np.stack([first_visit_rewards(row) for row in groups], axis=0)
    _, position_group_by_pose = np.unique(
        state_space.poses[:, :2], axis=0, return_inverse=True
    )
    pose_rewards = np.stack(
        [first_visit_rewards(row) for row in pose_indices], axis=0
    )
    position_rewards = np.stack(
        [first_visit_rewards(position_group_by_pose[row]) for row in pose_indices],
        axis=0,
    )
    return HeldoutTrajectories(
        pose_indices=pose_indices,
        actions=actions,
        observation_groups=groups,
        episodic_rewards=rewards,
        pose_first_visit_rewards=pose_rewards,
        position_first_visit_rewards=position_rewards,
        policy_names=tuple(policy_names),
    )


def _sample_roots(
    episode_count: int,
    episode_length: int,
    horizon: int,
    roots_per_episode: int,
    seed: int,
) -> np.ndarray:
    valid_count = int(episode_length) - int(horizon) + 1
    if valid_count <= 0:
        raise ValueError("episode_length must be at least horizon")
    root_count = min(int(roots_per_episode), valid_count)
    rows = []
    for episode in range(int(episode_count)):
        rng = np.random.default_rng(int(seed) + 65537 * episode)
        rows.append(np.sort(rng.choice(valid_count, size=root_count, replace=False)))
    return np.asarray(rows, dtype=np.int64)


def _load_model(
    run_dir: Path,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[ReconstructiveEpisodicMemory, Dict]:
    config = _load_config(run_dir / "total_config.py")
    policy = config["policy"]
    model_config = policy["model"]
    if policy.get("episodic_sequence_novelty_model") != "reconstructive_memory":
        raise ValueError("Run does not use episodic_sequence_novelty_model=reconstructive_memory")
    observation_shape = tuple(int(value) for value in model_config["observation_shape"])
    model = ReconstructiveEpisodicMemory(
        observation_shape=observation_shape,
        action_space_size=int(model_config["action_space_size"]),
        embedding_dim=int(policy["episodic_sequence_novelty_embedding_dim"]),
        hidden_dim=int(policy["episodic_sequence_novelty_hidden_dim"]),
        decoder_channels=int(policy["episodic_reconstructive_memory_decoder_channels"]),
        initial_distance_threshold=float(
            policy["episodic_predictive_memory_initial_distance_threshold"]
        ),
        initial_distance_temperature=float(
            policy["episodic_predictive_memory_initial_distance_temperature"]
        ),
        adaptive_distance_scale=bool(
            policy["episodic_predictive_memory_adaptive_distance_scale"]
        ),
        adaptive_threshold_ratio=float(
            policy["episodic_predictive_memory_adaptive_threshold_ratio"]
        ),
        adaptive_temperature_ratio=float(
            policy["episodic_predictive_memory_adaptive_temperature_ratio"]
        ),
        adaptive_minimum_scale=float(
            policy["episodic_predictive_memory_adaptive_minimum_scale"]
        ),
        adaptive_distinct_epsilon=float(
            policy["episodic_predictive_memory_adaptive_distinct_epsilon"]
        ),
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sidecar = checkpoint.get("episodic_sequence_novelty")
    if not isinstance(sidecar, dict) or "model" not in sidecar:
        raise KeyError("Checkpoint has no episodic_sequence_novelty model")
    model.load_state_dict(sidecar["model"], strict=True)
    model.eval()
    return model, checkpoint


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    positive = int(labels.sum())
    negative = int(labels.size - positive)
    if positive == 0 or negative == 0:
        return None
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
    return float(
        (rank_sum - positive * (positive + 1) / 2.0) / (positive * negative)
    )


def image_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if predictions.shape != targets.shape or predictions.size == 0:
        raise ValueError("Predictions and targets must have equal non-empty shapes")
    difference = predictions - targets
    mae = float(np.mean(np.abs(difference)))
    mse = float(np.mean(np.square(difference)))
    psnr = float(-10.0 * math.log10(max(mse, 1e-12)))
    return {"mae": mae, "mse": mse, "psnr": psnr}


def _encode_state_space(
    model: ReconstructiveEpisodicMemory,
    observations: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    parts = []
    with torch.inference_mode():
        for start in range(0, len(observations), int(batch_size)):
            batch = torch.as_tensor(
                observations[start:start + int(batch_size)],
                dtype=torch.float32,
                device=device,
            )
            parts.append(model.encode_memory_observations(batch).cpu())
    return torch.cat(parts, dim=0)


def _retrieval_metrics(
    predicted: torch.Tensor,
    actual_groups: np.ndarray,
    candidate_embeddings: torch.Tensor,
    candidate_groups: np.ndarray,
    batch_size: int = 256,
) -> Dict[str, float]:
    predicted = F.normalize(predicted.float().cpu(), dim=-1)
    candidates = F.normalize(candidate_embeddings.float().cpu(), dim=-1)
    actual_groups = np.asarray(actual_groups, dtype=np.int64).reshape(-1)
    candidate_groups = np.asarray(candidate_groups, dtype=np.int64)
    top1_hits = []
    top5_hits = []
    nearest_cosines = []
    for start in range(0, len(predicted), int(batch_size)):
        similarities = predicted[start:start + int(batch_size)] @ candidates.T
        values, indices = similarities.topk(k=min(5, candidates.shape[0]), dim=1)
        retrieved_groups = candidate_groups[indices.numpy()]
        expected = actual_groups[start:start + len(retrieved_groups), None]
        top1_hits.extend((retrieved_groups[:, :1] == expected).any(axis=1).tolist())
        top5_hits.extend((retrieved_groups == expected).any(axis=1).tolist())
        nearest_cosines.extend(values[:, 0].numpy().tolist())
    return {
        "top1_rgb_group": float(np.mean(top1_hits)),
        "top5_rgb_group": float(np.mean(top5_hits)),
        "nearest_cosine": float(np.mean(nearest_cosines)),
    }


def _append(values: Dict[str, List[np.ndarray]], key: str, value: torch.Tensor) -> None:
    values.setdefault(key, []).append(value.detach().float().cpu().numpy())


def _concatenate(values: Dict[str, List[np.ndarray]], key: str) -> np.ndarray:
    return np.concatenate(values[key], axis=0)


def evaluate_actual_suffixes(
    model: ReconstructiveEpisodicMemory,
    state_space: BranchingRoomsStateSpace,
    trajectories: HeldoutTrajectories,
    roots: np.ndarray,
    horizon: int,
    device: torch.device,
    episode_batch_size: int,
    candidate_embeddings: torch.Tensor,
) -> tuple[List[Dict], Dict, Dict[str, np.ndarray]]:
    values: Dict[str, List[np.ndarray]] = {}
    exact_label_parts: List[np.ndarray] = []
    exact_pose_label_parts: List[np.ndarray] = []
    exact_position_label_parts: List[np.ndarray] = []
    actual_group_parts: List[np.ndarray] = []
    root_parts: List[np.ndarray] = []
    episode_parts: List[np.ndarray] = []
    with torch.inference_mode():
        for batch_start in range(0, len(trajectories.actions), int(episode_batch_size)):
            batch_end = min(
                len(trajectories.actions), batch_start + int(episode_batch_size)
            )
            pose_indices = trajectories.pose_indices[batch_start:batch_end]
            observations_np = state_space.observations[pose_indices]
            observations = torch.as_tensor(
                observations_np, dtype=torch.float32, device=device
            )
            actions = torch.as_tensor(
                trajectories.actions[batch_start:batch_end],
                dtype=torch.long,
                device=device,
            )
            episodic_rewards = torch.as_tensor(
                trajectories.episodic_rewards[batch_start:batch_end],
                dtype=torch.float32,
                device=device,
            )
            extrinsic_rewards = torch.zeros_like(episodic_rewards)
            batch_roots = torch.as_tensor(
                roots[batch_start:batch_end], dtype=torch.long, device=device
            )
            outputs = model.training_outputs(
                observations,
                actions,
                episodic_rewards,
                extrinsic_rewards,
                batch_roots,
                int(horizon),
            )
            _append(values, "predicted_images", outputs["predicted_future_observations"])
            _append(values, "target_images", outputs["future_observation_targets"])
            _append(values, "direct_images", outputs["reconstructed_observations"][:, :, 1:])
            _append(values, "root_reconstructions", outputs["reconstructed_observations"][:, :, :1])
            _append(values, "predicted_embeddings", outputs["predicted_embeddings"])
            _append(values, "target_embeddings", outputs["target_embeddings"])
            _append(values, "probabilities", torch.sigmoid(outputs["novelty_logits"]))
            _append(values, "teacher_probabilities", outputs["target_novelty_probabilities"])
            _append(values, "predicted_distances", outputs["predicted_distances"])
            _append(values, "target_distances", outputs["target_distances"])

            batch_indices = np.arange(batch_end - batch_start)[:, None]
            root_np = roots[batch_start:batch_end]
            root_images = observations_np[batch_indices, root_np]
            copy_images = np.repeat(root_images[:, :, None], int(horizon), axis=2)
            history_means = np.stack(
                [
                    np.stack(
                        [observations_np[row, :int(root) + 1].mean(axis=0) for root in root_np[row]],
                        axis=0,
                    )
                    for row in range(len(root_np))
                ],
                axis=0,
            )
            mean_images = np.repeat(history_means[:, :, None], int(horizon), axis=2)
            values.setdefault("copy_images", []).append(copy_images.astype(np.float32))
            values.setdefault("mean_images", []).append(mean_images.astype(np.float32))
            values.setdefault("root_images", []).append(root_images.astype(np.float32))

            depth_offsets = np.arange(int(horizon), dtype=np.int64)
            exact_label_parts.append(
                trajectories.episodic_rewards[batch_start:batch_end][
                    batch_indices[:, :, None], root_np[:, :, None] + depth_offsets
                ]
            )
            exact_pose_label_parts.append(
                trajectories.pose_first_visit_rewards[batch_start:batch_end][
                    batch_indices[:, :, None], root_np[:, :, None] + depth_offsets
                ]
            )
            exact_position_label_parts.append(
                trajectories.position_first_visit_rewards[batch_start:batch_end][
                    batch_indices[:, :, None], root_np[:, :, None] + depth_offsets
                ]
            )
            future_pose_indices = pose_indices[
                batch_indices[:, :, None],
                root_np[:, :, None] + depth_offsets + 1,
            ]
            actual_group_parts.append(state_space.observation_group[future_pose_indices])
            root_parts.append(root_np.reshape(-1))
            episode_parts.append(
                np.repeat(np.arange(batch_start, batch_end), root_np.shape[1])
            )

    arrays = {key: _concatenate(values, key) for key in values}
    exact_labels = np.concatenate(exact_label_parts, axis=0).astype(np.int64)
    exact_pose_labels = np.concatenate(exact_pose_label_parts, axis=0).astype(np.int64)
    exact_position_labels = np.concatenate(
        exact_position_label_parts, axis=0
    ).astype(np.int64)
    actual_groups = np.concatenate(actual_group_parts, axis=0)
    rows = []
    for depth in range(int(horizon)):
        predicted_images = arrays["predicted_images"][:, :, depth]
        target_images = arrays["target_images"][:, :, depth]
        direct_images = arrays["direct_images"][:, :, depth]
        copy_images = arrays["copy_images"][:, :, depth]
        mean_images = arrays["mean_images"][:, :, depth]
        rollout = image_metrics(predicted_images, target_images)
        direct = image_metrics(direct_images, target_images)
        copy_root = image_metrics(copy_images, target_images)
        history_mean = image_metrics(mean_images, target_images)
        predicted_embeddings = torch.as_tensor(
            arrays["predicted_embeddings"][:, :, depth]
        ).reshape(-1, model.embedding_dim)
        target_embeddings = torch.as_tensor(
            arrays["target_embeddings"][:, :, depth]
        ).reshape(-1, model.embedding_dim)
        cosine = F.cosine_similarity(
            predicted_embeddings, target_embeddings, dim=-1
        ).numpy()
        retrieval = _retrieval_metrics(
            predicted_embeddings,
            actual_groups[:, :, depth].reshape(-1),
            candidate_embeddings,
            state_space.observation_group,
        )
        labels = exact_labels[:, :, depth].reshape(-1)
        pose_labels = exact_pose_labels[:, :, depth].reshape(-1)
        position_labels = exact_position_labels[:, :, depth].reshape(-1)
        probabilities = arrays["probabilities"][:, :, depth].reshape(-1)
        teacher_probabilities = arrays["teacher_probabilities"][:, :, depth].reshape(-1)
        rows.append(
            {
                "depth": depth + 1,
                "sample_count": int(labels.size),
                "novel_fraction": float(labels.mean()),
                "pose_novel_fraction": float(pose_labels.mean()),
                "position_novel_fraction": float(position_labels.mean()),
                "direct_mae": direct["mae"],
                "direct_psnr": direct["psnr"],
                "rollout_mae": rollout["mae"],
                "rollout_psnr": rollout["psnr"],
                "copy_root_mae": copy_root["mae"],
                "copy_root_psnr": copy_root["psnr"],
                "history_mean_mae": history_mean["mae"],
                "history_mean_psnr": history_mean["psnr"],
                "rollout_mae_gain_vs_copy": copy_root["mae"] - rollout["mae"],
                "rollout_mae_gain_vs_history_mean": history_mean["mae"] - rollout["mae"],
                "prediction_target_cosine": float(cosine.mean()),
                **retrieval,
                "novelty_auc_exact": _binary_auc(labels, probabilities),
                "novelty_auc_exact_pose": _binary_auc(pose_labels, probabilities),
                "novelty_auc_exact_position": _binary_auc(
                    position_labels, probabilities
                ),
                "teacher_auc_exact": _binary_auc(labels, teacher_probabilities),
                "teacher_auc_exact_position": _binary_auc(
                    position_labels, teacher_probabilities
                ),
                "novelty_brier_exact": float(np.mean((probabilities - labels) ** 2)),
                "prediction_teacher_mae": float(
                    np.mean(np.abs(probabilities - teacher_probabilities))
                ),
                "predicted_novel_mean": float(
                    probabilities[labels == 1].mean() if np.any(labels == 1) else 0.0
                ),
                "predicted_revisit_mean": float(
                    probabilities[labels == 0].mean() if np.any(labels == 0) else 0.0
                ),
            }
        )

    root_targets = arrays["root_images"]
    root_reconstructions = arrays["root_reconstructions"][:, :, 0]
    direct_root = image_metrics(root_reconstructions, root_targets)
    summary = {
        "direct_root_reconstruction": direct_root,
        "mean_rollout_mae": float(np.mean([row["rollout_mae"] for row in rows])),
        "mean_copy_root_mae": float(np.mean([row["copy_root_mae"] for row in rows])),
        "mean_history_mean_mae": float(
            np.mean([row["history_mean_mae"] for row in rows])
        ),
        "rollout_beats_copy_at_every_depth": bool(
            all(row["rollout_mae_gain_vs_copy"] > 0.0 for row in rows)
        ),
        "rollout_beats_history_mean_at_every_depth": bool(
            all(row["rollout_mae_gain_vs_history_mean"] > 0.0 for row in rows)
        ),
        "mean_novelty_auc_exact_rgb": float(
            np.mean([row["novelty_auc_exact"] for row in rows])
        ),
        "mean_novelty_auc_exact_position": float(
            np.mean([row["novelty_auc_exact_position"] for row in rows])
        ),
        "mean_teacher_auc_exact_rgb": float(
            np.mean([row["teacher_auc_exact"] for row in rows])
        ),
        "mean_teacher_auc_exact_position": float(
            np.mean([row["teacher_auc_exact_position"] for row in rows])
        ),
    }
    arrays["exact_labels"] = exact_labels
    arrays["exact_pose_labels"] = exact_pose_labels
    arrays["exact_position_labels"] = exact_position_labels
    arrays["actual_groups"] = actual_groups
    arrays["roots_flat"] = np.concatenate(root_parts)
    arrays["episodes_flat"] = np.concatenate(episode_parts)
    return rows, summary, arrays


def evaluate_immediate_action_choice(
    model: ReconstructiveEpisodicMemory,
    state_space: BranchingRoomsStateSpace,
    trajectories: HeldoutTrajectories,
    roots: np.ndarray,
    device: torch.device,
) -> Dict[str, float | int | None]:
    predicted_rows = []
    exact_rgb_rows = []
    exact_pose_rows = []
    exact_position_rows = []
    _, position_group_by_pose = np.unique(
        state_space.poses[:, :2], axis=0, return_inverse=True
    )
    with torch.inference_mode():
        for episode in range(len(trajectories.actions)):
            observations = torch.as_tensor(
                state_space.observations[trajectories.pose_indices[episode]],
                dtype=torch.float32,
                device=device,
            )
            context = model.start_context(observations[:1])
            contexts = {0: context}
            requested = set(int(value) for value in roots[episode])
            for step in range(len(trajectories.actions[episode])):
                context = model.advance_context(
                    context,
                    torch.tensor(
                        [trajectories.actions[episode, step]], device=device
                    ),
                    observations[step + 1:step + 2],
                    torch.tensor(
                        [trajectories.episodic_rewards[episode, step]],
                        device=device,
                    ),
                    torch.zeros(1, device=device),
                )
                if step + 1 in requested:
                    contexts[step + 1] = context

            ordered_contexts = [contexts[int(root)] for root in roots[episode]]
            root_indices = []
            paths = []
            for root_offset in range(len(ordered_contexts)):
                for action in range(3):
                    root_indices.append(root_offset)
                    paths.append([action])
            logits = model.score_path_endpoints(
                ordered_contexts, root_indices, paths
            ).reshape(len(ordered_contexts), 3)
            predicted_rows.append(torch.sigmoid(logits).cpu().numpy())

            episode_rgb = []
            episode_pose = []
            episode_position = []
            for root in roots[episode]:
                root = int(root)
                memory_groups = set(
                    int(value)
                    for value in trajectories.observation_groups[episode, :root + 1]
                )
                memory_poses = set(
                    int(value)
                    for value in trajectories.pose_indices[episode, :root + 1]
                )
                memory_positions = set(
                    int(value)
                    for value in position_group_by_pose[
                        trajectories.pose_indices[episode, :root + 1]
                    ]
                )
                pose = int(trajectories.pose_indices[episode, root])
                candidate_poses = state_space.transition[pose]
                candidate_groups = state_space.observation_group[candidate_poses]
                candidate_positions = position_group_by_pose[candidate_poses]
                episode_rgb.append(
                    [float(int(group) not in memory_groups) for group in candidate_groups]
                )
                episode_pose.append(
                    [float(int(value) not in memory_poses) for value in candidate_poses]
                )
                episode_position.append(
                    [
                        float(int(value) not in memory_positions)
                        for value in candidate_positions
                    ]
                )
            exact_rgb_rows.append(np.asarray(episode_rgb, dtype=np.float32))
            exact_pose_rows.append(np.asarray(episode_pose, dtype=np.float32))
            exact_position_rows.append(
                np.asarray(episode_position, dtype=np.float32)
            )

    probabilities = np.concatenate(predicted_rows, axis=0)
    selected = probabilities.argmax(axis=1)
    result: Dict[str, float | int | None] = {"root_count": int(len(probabilities))}

    def add_identity_metrics(name: str, exact: np.ndarray) -> None:
        available = exact.max(axis=1) > 0.5
        selected_novel = exact[np.arange(len(exact)), selected]
        relevant_probabilities = probabilities[available]
        relevant_exact = exact[available]
        relevant_selected = selected_novel[available]
        suffix = "" if name == "rgb" else f"_{name}"
        result[f"roots_with_novel_action{suffix}"] = int(available.sum())
        result[f"roots_with_novel_action_fraction{suffix}"] = float(
            available.mean()
        )
        result[f"selected_novel_fraction_when_available{suffix}"] = float(
            relevant_selected.mean() if relevant_selected.size else 0.0
        )
        result[f"random_novel_fraction_when_available{suffix}"] = float(
            relevant_exact.mean() if relevant_exact.size else 0.0
        )
        result[f"oracle_novel_fraction_when_available{suffix}"] = float(
            relevant_exact.max(axis=1).mean() if relevant_exact.size else 0.0
        )
        result[f"candidate_auc_exact{suffix}"] = (
            _binary_auc(
                relevant_exact.reshape(-1), relevant_probabilities.reshape(-1)
            )
            if relevant_exact.size else None
        )
        result[f"predicted_novel_candidate_mean{suffix}"] = float(
            relevant_probabilities[relevant_exact == 1].mean()
            if np.any(relevant_exact == 1) else 0.0
        )
        result[f"predicted_revisit_candidate_mean{suffix}"] = float(
            relevant_probabilities[relevant_exact == 0].mean()
            if np.any(relevant_exact == 0) else 0.0
        )

    add_identity_metrics("rgb", np.concatenate(exact_rgb_rows, axis=0))
    add_identity_metrics("pose", np.concatenate(exact_pose_rows, axis=0))
    add_identity_metrics("position", np.concatenate(exact_position_rows, axis=0))
    return result


def _chw_to_hwc(image: np.ndarray) -> np.ndarray:
    return np.clip(np.transpose(image, (1, 2, 0)), 0.0, 1.0)


def write_visual_grids(
    output_dir: Path,
    arrays: Dict[str, np.ndarray],
    trajectories: HeldoutTrajectories,
    figure_count: int,
    horizon: int,
) -> List[str]:
    import matplotlib.pyplot as plt

    exact = arrays["exact_labels"].reshape(-1, int(horizon))
    novel_count = exact.sum(axis=1)
    order = np.argsort(-novel_count, kind="stable")
    selected = order[:min(int(figure_count), len(order))]
    paths = []
    root_count_per_episode = arrays["predicted_images"].shape[1]
    for figure_index, flat_index in enumerate(selected):
        episode = int(flat_index // root_count_per_episode)
        root_offset = int(flat_index % root_count_per_episode)
        root = int(arrays["roots_flat"][flat_index])
        actual = np.concatenate(
            [
                arrays["target_images"][episode, root_offset, :1] * 0.0
                + arrays["copy_images"][episode, root_offset, :1],
                arrays["target_images"][episode, root_offset],
            ],
            axis=0,
        )
        direct = np.concatenate(
            [
                arrays["root_reconstructions"][episode, root_offset],
                arrays["direct_images"][episode, root_offset],
            ],
            axis=0,
        )
        rollout = np.concatenate(
            [
                arrays["root_reconstructions"][episode, root_offset],
                arrays["predicted_images"][episode, root_offset],
            ],
            axis=0,
        )
        error = np.abs(rollout - actual)
        fig, axes = plt.subplots(4, int(horizon) + 1, figsize=(3 * (horizon + 1), 11))
        row_names = ("true", "direct decode", "predicted rollout", "absolute error")
        action_suffix = trajectories.actions[episode, root:root + int(horizon)]
        for column in range(int(horizon) + 1):
            title = "root" if column == 0 else (
                f"d{column}: {ACTION_NAMES[int(action_suffix[column - 1])]}\n"
                f"novel={int(exact[flat_index, column - 1])}"
            )
            for row, images in enumerate((actual, direct, rollout, error)):
                axes[row, column].imshow(_chw_to_hwc(images[column]))
                axes[row, column].set_xticks([])
                axes[row, column].set_yticks([])
                if column == 0:
                    axes[row, column].set_ylabel(row_names[row])
                if row == 0:
                    axes[row, column].set_title(title)
        fig.suptitle(
            f"Held-out episode {episode}, root step {root}, "
            f"policy={trajectories.policy_names[episode]}"
        )
        fig.tight_layout()
        path = output_dir / f"reconstruction_rollout_example_{figure_index:02d}.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(str(path))
    return paths


def _write_csv(path: Path, rows: Sequence[Dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _format_number(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.4f}"


def write_markdown_report(path: Path, summary: Dict, depth_rows: Sequence[Dict]) -> None:
    lines = [
        "# Reconstructive Memory Held-out Diagnostic",
        "",
        "This evaluation loads a completed checkpoint and performs no training. "
        "It generates fresh action trajectories from the Branching Rooms start pose.",
        "",
        "## Main Checks",
        "",
        f"- Direct root reconstruction MAE: {_format_number(summary['actual_suffixes']['direct_root_reconstruction']['mae'])}",
        f"- Mean rollout MAE: {_format_number(summary['actual_suffixes']['mean_rollout_mae'])}",
        f"- Mean copy-root MAE: {_format_number(summary['actual_suffixes']['mean_copy_root_mae'])}",
        f"- Rollout beats copy-root at every depth: {_format_number(summary['actual_suffixes']['rollout_beats_copy_at_every_depth'])}",
        f"- Real-state teacher mean RGB-novelty AUC: {_format_number(summary['actual_suffixes']['mean_teacher_auc_exact_rgb'])}",
        f"- Predicted-rollout mean position-novelty AUC: {_format_number(summary['actual_suffixes']['mean_novelty_auc_exact_position'])}",
        f"- Immediate-action novelty selection when a novel action exists: {_format_number(summary['immediate_action_choice']['selected_novel_fraction_when_available'])}",
        f"- Corresponding random-action baseline: {_format_number(summary['immediate_action_choice']['random_novel_fraction_when_available'])}",
        f"- Selected action enters a new position when one is immediately available: {_format_number(summary['immediate_action_choice']['selected_novel_fraction_when_available_position'])}",
        f"- Corresponding random-action position baseline: {_format_number(summary['immediate_action_choice']['random_novel_fraction_when_available_position'])}",
        "",
        "## By Depth",
        "",
        "| depth | direct MAE | rollout MAE | copy-root MAE | rollout gain | cosine | RGB top-1 | RGB AUC | position AUC |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in depth_rows:
        lines.append(
            "| {depth} | {direct_mae:.4f} | {rollout_mae:.4f} | "
            "{copy_root_mae:.4f} | {rollout_mae_gain_vs_copy:.4f} | "
            "{prediction_target_cosine:.4f} | {top1_rgb_group:.4f} | {auc} | "
            "{position_auc} |".format(
                **row,
                auc=_format_number(row["novelty_auc_exact"]),
                position_auc=_format_number(row["novelty_auc_exact_position"]),
            )
        )
    lines.extend(
        [
            "",
            "Positive rollout gain means the learned rollout predicts pixels better "
            "than simply copying the root observation. RGB retrieval treats physically "
            "different poses with identical partial observations as equivalent.",
            "",
            "## Interpretation",
            "",
            "The real-state encoder and decoder retain substantial observation "
            "information, and real encoded states separate first visits from exact "
            "RGB revisits. The action-conditioned rollout does not preserve that "
            "grounding: its decoded future is worse than both trivial image baselines "
            "and its nearest real-state retrieval is almost always the wrong RGB class.",
            "",
            "The model also ranks immediate new physical positions only weakly. Its "
            "stronger RGB/pose score therefore should not be interpreted as equally "
            "strong spatial exploration guidance.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--episode-count", type=int, default=12)
    parser.add_argument("--episode-length", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--roots-per-episode", type=int, default=12)
    parser.add_argument("--episode-batch-size", type=int, default=2)
    parser.add_argument("--encode-batch-size", type=int, default=256)
    parser.add_argument("--figure-count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=90210)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    checkpoint_path = _resolve_checkpoint(run_dir, args.checkpoint)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else run_dir / "diagnostics" / "reconstructive_heldout"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    print(f"device={device} checkpoint={checkpoint_path}", flush=True)
    state_space = build_state_space()
    trajectories = generate_heldout_trajectories(
        state_space,
        args.episode_count,
        args.episode_length,
        args.seed,
    )
    roots = _sample_roots(
        args.episode_count,
        args.episode_length,
        args.horizon,
        args.roots_per_episode,
        args.seed + 1,
    )
    model, checkpoint = _load_model(run_dir, checkpoint_path, device)
    print("encoding complete physical state space", flush=True)
    candidate_embeddings = _encode_state_space(
        model,
        state_space.observations,
        device,
        args.encode_batch_size,
    )
    print("evaluating held-out executed suffixes", flush=True)
    depth_rows, suffix_summary, arrays = evaluate_actual_suffixes(
        model,
        state_space,
        trajectories,
        roots,
        args.horizon,
        device,
        args.episode_batch_size,
        candidate_embeddings,
    )
    print("evaluating immediate action selection", flush=True)
    action_summary = evaluate_immediate_action_choice(
        model,
        state_space,
        trajectories,
        roots,
        device,
    )
    figure_paths = []
    if not args.no_figures:
        figure_paths = write_visual_grids(
            output_dir,
            arrays,
            trajectories,
            args.figure_count,
            args.horizon,
        )
    summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_last_iter": int(checkpoint.get("last_iter", -1)),
        "device": str(device),
        "episode_count": int(args.episode_count),
        "episode_length": int(args.episode_length),
        "roots_per_episode": int(roots.shape[1]),
        "horizon": int(args.horizon),
        "policy_names": list(trajectories.policy_names),
        "fresh_trajectory_note": (
            "New seeded action trajectories; not recovered from training replay. "
            "Individual states may overlap because the environment is fixed."
        ),
        "actual_suffixes": suffix_summary,
        "immediate_action_choice": action_summary,
        "figures": figure_paths,
    }
    _write_csv(output_dir / "metrics_by_depth.csv", depth_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    write_markdown_report(output_dir / "report.md", summary, depth_rows)
    print(json.dumps(summary["actual_suffixes"], indent=2), flush=True)
    print(json.dumps(summary["immediate_action_choice"], indent=2), flush=True)
    print(f"report: {output_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
