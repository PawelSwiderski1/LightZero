#!/usr/bin/env python3
"""Diagnose raw-latent aliasing and path-memory errors in Branching Rooms.

The diagnostic separates three effects that the online episodic-memory logs
cannot distinguish:

1. physical poses that produce exactly the same partial RGB observation;
2. distinct observations that the representation network maps close together;
3. recurrent MuZero latents that contract toward a wrong path ancestor.

It loads a completed run checkpoint and writes a JSON summary, per-depth CSV,
and optional per-path CSV. No training or environment interaction is performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from lzero.model.ensemble_muzero_model import EnsembleMuZeroModel
from zoo.minigrid.envs.branching_rooms_env import BranchingRoomsEnv


ACTION_NAMES = ("left", "right", "forward")


@dataclass(frozen=True)
class BranchingRoomsStateSpace:
    poses: np.ndarray
    observations: np.ndarray
    observation_group: np.ndarray
    transition: np.ndarray
    rooms: np.ndarray


def _load_config(path: Path) -> Dict:
    spec = importlib.util.spec_from_file_location("branching_rooms_geometry_config", path)
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


def _observation_digest(observation: np.ndarray) -> bytes:
    return hashlib.sha1(np.ascontiguousarray(observation).view(np.uint8)).digest()


def build_state_space(
        agent_view_size: int = 7,
        tile_size: int = 12,
) -> BranchingRoomsStateSpace:
    """Enumerate every reachable (x, y, direction) pose and exact transition."""
    env = BranchingRoomsEnv(max_steps=1200, agent_view_size=agent_view_size)
    env.reset(seed=0)
    positions = sorted(env._reachable_positions, key=lambda item: (item[1], item[0]))
    poses = np.asarray(
        [(x, y, direction) for x, y in positions for direction in range(4)],
        dtype=np.int64,
    )
    pose_to_index = {tuple(pose): index for index, pose in enumerate(poses.tolist())}

    observations = []
    digests = []
    rooms = []
    for x, y, direction in poses:
        env.agent_pos = (int(x), int(y))
        env.agent_dir = int(direction)
        image = env.get_frame(tile_size=tile_size, agent_pov=True)
        image = np.transpose(np.asarray(image)[..., :3], (2, 0, 1))
        image = image.astype(np.float32) / 255.0
        observations.append(image)
        digests.append(_observation_digest(image))
        room = env.room_for_position((int(x), int(y)))
        rooms.append((-1, -1) if room is None else room)

    digest_to_group: Dict[bytes, int] = {}
    observation_group = []
    for digest in digests:
        if digest not in digest_to_group:
            digest_to_group[digest] = len(digest_to_group)
        observation_group.append(digest_to_group[digest])

    transition = np.empty((len(poses), len(ACTION_NAMES)), dtype=np.int64)
    direction_vectors = ((1, 0), (0, 1), (-1, 0), (0, -1))
    for pose_index, (x, y, direction) in enumerate(poses):
        transition[pose_index, 0] = pose_to_index[(int(x), int(y), (int(direction) - 1) % 4)]
        transition[pose_index, 1] = pose_to_index[(int(x), int(y), (int(direction) + 1) % 4)]
        dx, dy = direction_vectors[int(direction)]
        next_position = (int(x + dx), int(y + dy))
        cell = env.grid.get(*next_position)
        if cell is None or cell.can_overlap():
            transition[pose_index, 2] = pose_to_index[(*next_position, int(direction))]
        else:
            transition[pose_index, 2] = pose_index
    env.close()

    return BranchingRoomsStateSpace(
        poses=poses,
        observations=np.stack(observations, axis=0),
        observation_group=np.asarray(observation_group, dtype=np.int64),
        transition=transition,
        rooms=np.asarray(rooms, dtype=np.int64),
    )


def _load_model(
        config: Dict,
        checkpoint_path: Path,
        device: torch.device,
) -> Tuple[EnsembleMuZeroModel, Dict]:
    model_config = dict(config["policy"]["model"])
    if str(model_config.get("model_type", "conv")) != "conv":
        raise ValueError("This diagnostic currently supports the convolutional ensemble model only")
    model = EnsembleMuZeroModel(**model_config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, checkpoint


def encode_all_poses(
        model: EnsembleMuZeroModel,
        observations: np.ndarray,
        ensemble_size: int,
        device: torch.device,
        batch_size: int,
) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """Return direct raw latents as CPU tensor [pose, member, flattened_feature]."""
    member_parts: List[List[torch.Tensor]] = [[] for _ in range(ensemble_size)]
    latent_shape: Tuple[int, ...] | None = None
    with torch.inference_mode():
        for start in range(0, len(observations), batch_size):
            batch = torch.as_tensor(
                observations[start:start + batch_size], dtype=torch.float32, device=device
            )
            for member_id in range(ensemble_size):
                latent = model.initial_inference(batch, member_id=member_id).latent_state
                if latent_shape is None:
                    latent_shape = tuple(int(value) for value in latent.shape[1:])
                member_parts[member_id].append(latent.detach().float().flatten(1).cpu())
    if latent_shape is None:
        raise RuntimeError("No observations were encoded")
    members = [torch.cat(parts, dim=0) for parts in member_parts]
    return torch.stack(members, dim=1).contiguous(), latent_shape


def _member_mean_squared_distance(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    return (first.float() - second.float()).pow(2).sum(dim=-1).mean(dim=-1)


def _indexed_distances(
        latents: torch.Tensor,
        left: np.ndarray,
        right: np.ndarray,
        batch_size: int = 4096,
) -> np.ndarray:
    values = []
    for start in range(0, len(left), batch_size):
        left_index = torch.as_tensor(left[start:start + batch_size], dtype=torch.long)
        right_index = torch.as_tensor(right[start:start + batch_size], dtype=torch.long)
        values.append(
            _member_mean_squared_distance(latents[left_index], latents[right_index]).numpy()
        )
    return np.concatenate(values, axis=0) if values else np.empty((0,), dtype=np.float32)


def _distribution(values: np.ndarray) -> Dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "p05": float(np.quantile(values, 0.05)),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "p75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def _sample_pairs(
        state_space: BranchingRoomsStateSpace,
        category: str,
        count: int,
        rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    state_count = len(state_space.poses)
    left: List[int] = []
    right: List[int] = []
    attempt_limit = max(1000, count * 100)
    for _ in range(attempt_limit):
        if len(left) >= count:
            break
        first = int(rng.integers(state_count))
        second = int(rng.integers(state_count))
        if first == second:
            continue
        first_pose = state_space.poses[first]
        second_pose = state_space.poses[second]
        same_position = bool(np.array_equal(first_pose[:2], second_pose[:2]))
        same_room = bool(
            state_space.rooms[first, 0] >= 0
            and np.array_equal(state_space.rooms[first], state_space.rooms[second])
        )
        same_observation = (
            state_space.observation_group[first] == state_space.observation_group[second]
        )
        keep = (
            (category == "same_room_distinct_observation" and same_room and not same_observation)
            or (category == "different_room" and not same_room)
            or (category == "random_distinct_observation" and not same_observation)
            or (category == "different_position_same_observation" and not same_position and same_observation)
        )
        if keep:
            left.append(first)
            right.append(second)
    return np.asarray(left, dtype=np.int64), np.asarray(right, dtype=np.int64)


def _observation_alias_pairs(
        state_space: BranchingRoomsStateSpace,
        count: int,
        rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    groups: Dict[int, List[int]] = defaultdict(list)
    for pose_index, group in enumerate(state_space.observation_group):
        groups[int(group)].append(pose_index)
    pairs = [
        (first, second)
        for indices in groups.values()
        for first, second in itertools.combinations(indices, 2)
        if not np.array_equal(
            state_space.poses[first, :2], state_space.poses[second, :2]
        )
    ]
    if len(pairs) > count:
        selected = rng.choice(len(pairs), size=count, replace=False)
        pairs = [pairs[int(index)] for index in selected]
    if not pairs:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
    left, right = zip(*pairs)
    return np.asarray(left, dtype=np.int64), np.asarray(right, dtype=np.int64)


def direct_geometry(
        state_space: BranchingRoomsStateSpace,
        latents: torch.Tensor,
        pair_count: int,
        rng: np.random.Generator,
) -> Tuple[Dict, float, Dict[str, np.ndarray]]:
    pose_count = len(state_space.poses)
    pose_indices = np.arange(pose_count, dtype=np.int64)

    transition_left = np.repeat(pose_indices, state_space.transition.shape[1])
    transition_right = state_space.transition.reshape(-1)
    transition_distances = _indexed_distances(latents, transition_left, transition_right)
    positive_transition_distances = transition_distances[transition_distances > 1e-8]
    distance_scale = float(np.median(positive_transition_distances))

    same_cell_left = []
    same_cell_right = []
    for position_offset in range(0, pose_count, 4):
        for first_direction, second_direction in itertools.combinations(range(4), 2):
            same_cell_left.append(position_offset + first_direction)
            same_cell_right.append(position_offset + second_direction)

    pair_indices: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
        "same_cell_different_direction": (
            np.asarray(same_cell_left, dtype=np.int64),
            np.asarray(same_cell_right, dtype=np.int64),
        ),
        "one_action_all": (transition_left, transition_right),
        "one_action_changed_pose": (
            transition_left[transition_left != transition_right],
            transition_right[transition_left != transition_right],
        ),
        "different_position_same_observation": _observation_alias_pairs(
            state_space, pair_count, rng
        ),
    }
    for category in (
            "same_room_distinct_observation",
            "different_room",
            "random_distinct_observation",
    ):
        pair_indices[category] = _sample_pairs(state_space, category, pair_count, rng)

    pair_distances = {
        name: _indexed_distances(latents, left, right)
        for name, (left, right) in pair_indices.items()
    }
    group_counts = Counter(state_space.observation_group.tolist())
    ambiguous_mask = np.asarray(
        [group_counts[int(group)] > 1 for group in state_space.observation_group], dtype=bool
    )
    summary = {
        "reachable_positions": int(pose_count // 4),
        "reachable_poses": int(pose_count),
        "unique_rgb_observations": int(len(group_counts)),
        "ambiguous_observation_groups": int(sum(value > 1 for value in group_counts.values())),
        "poses_with_rgb_alias": int(ambiguous_mask.sum()),
        "poses_with_rgb_alias_fraction": float(ambiguous_mask.mean()),
        "largest_rgb_alias_group": int(max(group_counts.values())),
        "global_positive_one_step_distance_scale": distance_scale,
        "pair_distance_squared_l2": {
            name: _distribution(values) for name, values in pair_distances.items()
        },
    }
    return summary, distance_scale, pair_distances


def exhaustive_path_revisit_rates(
        state_space: BranchingRoomsStateSpace,
        horizon: int,
) -> Dict[int, Dict[str, float | int]]:
    """Return exact pose/observation revisit rates over every root and action path."""
    action_count = state_space.transition.shape[1]
    state_count = len(state_space.poses)
    current = np.arange(state_count, dtype=np.int64).reshape(state_count, 1)
    ancestors = [current]
    result = {}
    for depth in range(1, horizon + 1):
        current = state_space.transition[
            current[..., None], np.arange(action_count, dtype=np.int64)
        ].reshape(state_count, -1)
        ancestors = [_expand_sequence_array(ancestor, action_count) for ancestor in ancestors]
        ancestor_stack = np.stack(ancestors, axis=-1)
        pose_revisit = np.any(ancestor_stack == current[..., None], axis=-1)
        observation_revisit = np.any(
            state_space.observation_group[ancestor_stack]
            == state_space.observation_group[current][..., None],
            axis=-1,
        )
        result[depth] = {
            "path_count": int(current.size),
            "pose_revisit_fraction": float(pose_revisit.mean()),
            "observation_revisit_fraction": float(observation_revisit.mean()),
            "different_pose_rgb_alias_fraction": float(
                (observation_revisit & ~pose_revisit).mean()
            ),
        }
        ancestors.append(current)
    return result


def _logged_final_raw_distance_scale(run_dir: Path, checkpoint_iter: int) -> float | None:
    path = run_dir / "diagnostics" / "rnd_episodic_memory.csv"
    if not path.exists():
        return None
    with path.open(newline="") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row.get("key_space") == "raw"
            and int(float(row.get("train_iter_at_collect", -1))) <= checkpoint_iter
        ]
    if not rows:
        return None
    latest_iteration = max(int(float(row["train_iter_at_collect"])) for row in rows)
    scales = np.asarray([
        float(row["one_step_distance_scale"])
        for row in rows
        if int(float(row["train_iter_at_collect"])) == latest_iteration
    ], dtype=np.float64)
    scales = scales[np.isfinite(scales) & (scales > 1e-8)]
    return float(np.median(scales)) if scales.size else None


def nearest_direct_neighbors(
        state_space: BranchingRoomsStateSpace,
        latents: torch.Tensor,
        query_count: int,
        device: torch.device,
        batch_size: int,
        rng: np.random.Generator,
) -> Dict:
    query_count = min(max(0, int(query_count)), len(latents))
    if query_count == 0:
        return {"query_count": 0}
    query_indices = np.sort(rng.choice(len(latents), size=query_count, replace=False))
    feature_bank = latents.flatten(1).to(device=device, dtype=torch.float32)
    ensemble_size = int(latents.shape[1])
    nearest_any = []
    nearest_distinct_observation = []
    any_distances = []
    distinct_observation_distances = []
    for start in range(0, query_count, batch_size):
        indices = query_indices[start:start + batch_size]
        queries = feature_bank[torch.as_tensor(indices, dtype=torch.long, device=device)]
        distances = torch.cdist(queries, feature_bank, p=2).pow(2) / ensemble_size
        local_rows = torch.arange(len(indices), device=device)
        distances[local_rows, torch.as_tensor(indices, dtype=torch.long, device=device)] = float("inf")
        any_distance, any_index = distances.min(dim=1)
        distinct_distances = distances.clone()
        for row, query_index in enumerate(indices):
            same_observation = (
                state_space.observation_group == state_space.observation_group[int(query_index)]
            )
            distinct_distances[row, torch.as_tensor(same_observation, device=device)] = float("inf")
        distinct_distance, distinct_index = distinct_distances.min(dim=1)
        nearest_any.extend(any_index.cpu().numpy().tolist())
        nearest_distinct_observation.extend(distinct_index.cpu().numpy().tolist())
        any_distances.extend(any_distance.cpu().numpy().tolist())
        distinct_observation_distances.extend(distinct_distance.cpu().numpy().tolist())

    nearest_any_array = np.asarray(nearest_any, dtype=np.int64)
    nearest_distinct_array = np.asarray(nearest_distinct_observation, dtype=np.int64)
    same_observation = (
        state_space.observation_group[query_indices]
        == state_space.observation_group[nearest_any_array]
    )
    same_position = np.all(
        state_space.poses[query_indices, :2] == state_space.poses[nearest_any_array, :2], axis=1
    )
    same_room = np.all(
        state_space.rooms[query_indices] == state_space.rooms[nearest_any_array], axis=1
    )
    distinct_same_position = np.all(
        state_space.poses[query_indices, :2]
        == state_space.poses[nearest_distinct_array, :2],
        axis=1,
    )
    return {
        "query_count": int(query_count),
        "nearest_other_same_rgb_fraction": float(same_observation.mean()),
        "nearest_other_same_position_fraction": float(same_position.mean()),
        "nearest_other_same_room_fraction": float(same_room.mean()),
        "nearest_other_distance": _distribution(np.asarray(any_distances)),
        "nearest_distinct_rgb_same_position_fraction": float(distinct_same_position.mean()),
        "nearest_distinct_rgb_distance": _distribution(
            np.asarray(distinct_observation_distances)
        ),
    }


def _expand_recurrent_latents(
        model: EnsembleMuZeroModel,
        current: torch.Tensor,
        ensemble_size: int,
) -> torch.Tensor:
    """Expand [root, sequence, member, ...] through every discrete action."""
    root_count, sequence_count = current.shape[:2]
    action_count = len(ACTION_NAMES)
    latent_shape = current.shape[3:]
    expanded = current.unsqueeze(2).expand(
        root_count, sequence_count, action_count, ensemble_size, *latent_shape
    ).reshape(-1, *latent_shape)
    actions = torch.arange(action_count, device=current.device).view(1, 1, action_count, 1)
    actions = actions.expand(root_count, sequence_count, action_count, ensemble_size).reshape(-1)
    member_ids = torch.arange(ensemble_size, device=current.device).view(1, 1, 1, ensemble_size)
    member_ids = member_ids.expand(root_count, sequence_count, action_count, ensemble_size).reshape(-1)
    output = model.recurrent_inference(expanded, actions, member_id=member_ids)
    return output.latent_state.reshape(
        root_count, sequence_count * action_count, ensemble_size, *latent_shape
    )


def _expand_sequence_array(values: np.ndarray, action_count: int) -> np.ndarray:
    return np.repeat(values, action_count, axis=1)


def _gate_factor(
        distance: torch.Tensor,
        distance_scale: float,
        kernel_bandwidth: float,
        minimum_factor: float,
) -> torch.Tensor:
    normalized = distance / max(kernel_bandwidth * distance_scale, 1e-8)
    novelty = -torch.expm1(-normalized).clamp(max=0.0)
    return minimum_factor + (1.0 - minimum_factor) * novelty.clamp(0.0, 1.0)


def _safe_rate(numerator: np.ndarray, denominator: np.ndarray) -> float:
    denominator_count = int(np.asarray(denominator, dtype=bool).sum())
    if denominator_count == 0:
        return float("nan")
    return float(np.asarray(numerator, dtype=bool)[denominator].mean())


def _depth_summary(rows: List[Dict]) -> Dict:
    pose_revisit = np.asarray([row["pose_revisit"] for row in rows], dtype=bool)
    observation_revisit = np.asarray([row["observation_revisit"] for row in rows], dtype=bool)
    suppressed = np.asarray([row["suppressed"] for row in rows], dtype=bool)
    matched_pose = np.asarray([row["matched_ancestor_same_pose"] for row in rows], dtype=bool)
    matched_observation = np.asarray(
        [row["matched_ancestor_same_observation"] for row in rows], dtype=bool
    )
    physical_new = ~pose_revisit
    observation_new = ~observation_revisit
    return {
        "count": len(rows),
        "pose_revisit_fraction": float(pose_revisit.mean()),
        "observation_revisit_fraction": float(observation_revisit.mean()),
        "mean_reward_factor": float(np.mean([row["reward_factor"] for row in rows])),
        "suppressed_fraction": float(suppressed.mean()),
        "pose_revisit_recall": _safe_rate(suppressed, pose_revisit),
        "pose_revisit_precision": _safe_rate(pose_revisit, suppressed),
        "physical_new_false_suppression_rate": _safe_rate(suppressed, physical_new),
        "observation_revisit_recall": _safe_rate(suppressed, observation_revisit),
        "observation_revisit_precision": _safe_rate(observation_revisit, suppressed),
        "observation_new_false_suppression_rate": _safe_rate(suppressed, observation_new),
        "nearest_ancestor_same_pose_fraction": float(matched_pose.mean()),
        "nearest_ancestor_same_observation_fraction": float(matched_observation.mean()),
        "nearest_ancestor_is_parent_fraction": float(np.mean([
            row["matched_ancestor_depth"] == row["depth"] - 1 for row in rows
        ])),
        "nearest_ancestor_is_root_fraction": float(np.mean([
            row["matched_ancestor_depth"] == 0 for row in rows
        ])),
        "ancestor_distance": _distribution(
            np.asarray([row["minimum_ancestor_distance"] for row in rows])
        ),
        "correct_direct_alignment_distance": _distribution(
            np.asarray([row["correct_direct_distance"] for row in rows])
        ),
        "recurrent_step_distance": _distribution(
            np.asarray([row["recurrent_step_distance"] for row in rows])
        ),
        "wrong_ancestor_closer_than_correct_direct_fraction": float(np.mean([
            row["wrong_ancestor_closer_than_correct_direct"] for row in rows
        ])),
    }


def diagnose_rollouts(
        model: EnsembleMuZeroModel,
        state_space: BranchingRoomsStateSpace,
        direct_latents: torch.Tensor,
        latent_shape: Tuple[int, ...],
        root_indices: np.ndarray,
        horizon: int,
        root_batch_size: int,
        distance_scale: float,
        kernel_bandwidth: float,
        minimum_factor: float,
        suppression_threshold: float,
        device: torch.device,
) -> Tuple[Dict[int, Dict], List[Dict]]:
    ensemble_size = int(direct_latents.shape[1])
    rows_by_depth: Dict[int, List[Dict]] = defaultdict(list)
    path_rows: List[Dict] = []
    direct_flat = direct_latents

    with torch.inference_mode():
        for root_start in range(0, len(root_indices), root_batch_size):
            batch_roots = root_indices[root_start:root_start + root_batch_size]
            batch_size = len(batch_roots)
            root_flat = direct_flat[torch.as_tensor(batch_roots, dtype=torch.long)]
            current = root_flat.reshape(batch_size, 1, ensemble_size, *latent_shape).to(device)
            current_pose = batch_roots.reshape(batch_size, 1)
            sequences = np.empty((batch_size, 1, 0), dtype=np.int64)
            ancestor_latents = [current]
            ancestor_poses = [current_pose]

            for depth in range(1, horizon + 1):
                parent = current
                current = _expand_recurrent_latents(model, parent, ensemble_size)
                action_count = len(ACTION_NAMES)
                parent_pose = current_pose
                current_pose = state_space.transition[parent_pose[..., None], np.arange(action_count)]
                current_pose = current_pose.reshape(batch_size, -1)
                sequences = np.concatenate(
                    [
                        np.repeat(sequences, action_count, axis=1),
                        np.tile(np.arange(action_count), (batch_size, sequences.shape[1]))[..., None],
                    ],
                    axis=2,
                )

                expanded_ancestor_latents = [
                    ancestor.unsqueeze(2).expand(
                        batch_size,
                        ancestor.shape[1],
                        action_count,
                        ensemble_size,
                        *latent_shape,
                    ).reshape(batch_size, -1, ensemble_size, *latent_shape)
                    for ancestor in ancestor_latents
                ]
                expanded_ancestor_poses = [
                    _expand_sequence_array(ancestor, action_count)
                    for ancestor in ancestor_poses
                ]
                current_flat = current.flatten(3)
                ancestor_distances = torch.stack(
                    [
                        _member_mean_squared_distance(
                            current_flat,
                            ancestor.flatten(3),
                        )
                        for ancestor in expanded_ancestor_latents
                    ],
                    dim=-1,
                )
                minimum_distance, matched_depth = ancestor_distances.min(dim=-1)
                factors = _gate_factor(
                    minimum_distance,
                    distance_scale,
                    kernel_bandwidth,
                    minimum_factor,
                )

                endpoint_index = torch.as_tensor(current_pose.reshape(-1), dtype=torch.long)
                direct_target = direct_flat[endpoint_index].reshape(
                    batch_size, -1, ensemble_size, direct_flat.shape[-1]
                ).to(device)
                correct_direct_distance = _member_mean_squared_distance(
                    current_flat, direct_target
                )
                parent_expanded = parent.unsqueeze(2).expand(
                    batch_size, parent.shape[1], action_count, ensemble_size, *latent_shape
                ).reshape(batch_size, -1, ensemble_size, *latent_shape)
                recurrent_step_distance = _member_mean_squared_distance(
                    current_flat, parent_expanded.flatten(3)
                )

                ancestor_pose_stack = np.stack(expanded_ancestor_poses, axis=-1)
                pose_revisit = np.any(ancestor_pose_stack == current_pose[..., None], axis=-1)
                ancestor_observation_groups = state_space.observation_group[ancestor_pose_stack]
                endpoint_observation_group = state_space.observation_group[current_pose]
                observation_revisit = np.any(
                    ancestor_observation_groups == endpoint_observation_group[..., None], axis=-1
                )
                matched_depth_np = matched_depth.cpu().numpy()
                matched_pose = np.take_along_axis(
                    ancestor_pose_stack, matched_depth_np[..., None], axis=-1
                )[..., 0]
                matched_same_pose = matched_pose == current_pose
                matched_same_observation = (
                    state_space.observation_group[matched_pose] == endpoint_observation_group
                )

                minimum_distance_np = minimum_distance.cpu().numpy()
                factor_np = factors.cpu().numpy()
                direct_distance_np = correct_direct_distance.cpu().numpy()
                step_distance_np = recurrent_step_distance.cpu().numpy()
                suppressed = factor_np < suppression_threshold
                wrong_ancestor_closer = (
                    (~matched_same_observation) & (minimum_distance_np < direct_distance_np)
                )

                for root_row in range(batch_size):
                    root_pose = state_space.poses[batch_roots[root_row]]
                    for sequence_row in range(current_pose.shape[1]):
                        endpoint_pose = state_space.poses[current_pose[root_row, sequence_row]]
                        matched_pose_value = state_space.poses[matched_pose[root_row, sequence_row]]
                        row = {
                            "root_index": int(batch_roots[root_row]),
                            "root_pose": ",".join(str(int(value)) for value in root_pose),
                            "depth": int(depth),
                            "action_sequence": "-".join(
                                ACTION_NAMES[int(action)]
                                for action in sequences[root_row, sequence_row]
                            ),
                            "endpoint_pose": ",".join(
                                str(int(value)) for value in endpoint_pose
                            ),
                            "pose_revisit": bool(pose_revisit[root_row, sequence_row]),
                            "observation_revisit": bool(
                                observation_revisit[root_row, sequence_row]
                            ),
                            "matched_ancestor_depth": int(
                                matched_depth_np[root_row, sequence_row]
                            ),
                            "matched_ancestor_pose": ",".join(
                                str(int(value)) for value in matched_pose_value
                            ),
                            "matched_ancestor_same_pose": bool(
                                matched_same_pose[root_row, sequence_row]
                            ),
                            "matched_ancestor_same_observation": bool(
                                matched_same_observation[root_row, sequence_row]
                            ),
                            "minimum_ancestor_distance": float(
                                minimum_distance_np[root_row, sequence_row]
                            ),
                            "correct_direct_distance": float(
                                direct_distance_np[root_row, sequence_row]
                            ),
                            "recurrent_step_distance": float(
                                step_distance_np[root_row, sequence_row]
                            ),
                            "reward_factor": float(factor_np[root_row, sequence_row]),
                            "suppressed": bool(suppressed[root_row, sequence_row]),
                            "wrong_ancestor_closer_than_correct_direct": bool(
                                wrong_ancestor_closer[root_row, sequence_row]
                            ),
                        }
                        rows_by_depth[depth].append(row)
                        path_rows.append(row)

                ancestor_latents = expanded_ancestor_latents + [current]
                ancestor_poses = expanded_ancestor_poses + [current_pose]

    return {
        depth: _depth_summary(rows) for depth, rows in sorted(rows_by_depth.items())
    }, path_rows


def _write_csv(path: Path, rows: Iterable[Dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _json_sanitize(value):
    if isinstance(value, dict):
        return {str(key): _json_sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_sanitize(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint name/path. Defaults to the highest iteration checkpoint.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--encode-batch-size", type=int, default=128)
    parser.add_argument("--root-batch-size", type=int, default=2)
    parser.add_argument("--root-count", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--pair-count", type=int, default=20000)
    parser.add_argument("--nearest-query-count", type=int, default=128)
    parser.add_argument("--nearest-batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--distance-scale", type=float, default=None)
    parser.add_argument("--kernel-bandwidth", type=float, default=None)
    parser.add_argument("--minimum-reward-factor", type=float, default=None)
    parser.add_argument("--suppression-threshold", type=float, default=0.5)
    parser.add_argument("--output-prefix", type=Path, default=None)
    parser.add_argument("--no-path-records", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.horizon <= 0:
        raise ValueError("--horizon must be positive")
    if args.root_count <= 0:
        raise ValueError("--root-count must be positive")
    run_dir = args.run_dir.resolve()
    checkpoint_path = _resolve_checkpoint(run_dir, args.checkpoint)
    config = _load_config(run_dir / "total_config.py")
    device = _resolve_device(args.device)
    rng = np.random.default_rng(args.seed)

    print(f"run: {run_dir}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"device: {device}")
    print("enumerating all reachable physical poses and RGB observations...")
    env_config = config["env"]
    state_space = build_state_space(
        agent_view_size=int(env_config.get("agent_view_size", 7)),
        tile_size=int(env_config.get("rgb_tile_size", 12)),
    )
    model, checkpoint = _load_model(config, checkpoint_path, device)
    model_config = config["policy"]["model"]
    ensemble_size = int(model_config["ensemble_size"])
    print(f"encoding {len(state_space.poses)} poses for {ensemble_size} members...")
    direct_latents, latent_shape = encode_all_poses(
        model,
        state_space.observations,
        ensemble_size,
        device,
        max(1, args.encode_batch_size),
    )

    direct_summary, global_distance_scale, pair_distances = direct_geometry(
        state_space, direct_latents, max(1, args.pair_count), rng
    )
    direct_summary["exhaustive_uniform_path_revisits"] = exhaustive_path_revisit_rates(
        state_space, args.horizon
    )
    checkpoint_iter = int(checkpoint.get("last_iter", -1))
    logged_distance_scale = _logged_final_raw_distance_scale(run_dir, checkpoint_iter)
    if args.distance_scale is not None:
        distance_scale = float(args.distance_scale)
        distance_scale_source = "command_line"
    elif logged_distance_scale is not None:
        distance_scale = logged_distance_scale
        distance_scale_source = "final_logged_raw_memory_median"
    else:
        distance_scale = global_distance_scale
        distance_scale_source = "global_direct_one_step_median"
    policy_config = config["policy"]
    kernel_bandwidth = (
        float(args.kernel_bandwidth)
        if args.kernel_bandwidth is not None
        else float(policy_config.get("rnd_episodic_memory_kernel_bandwidth", 0.25))
    )
    minimum_factor = (
        float(args.minimum_reward_factor)
        if args.minimum_reward_factor is not None
        else float(policy_config.get("rnd_episodic_memory_min_reward_factor", 0.05))
    )
    cutoff_normalized = -math.log(
        max(1e-12, 1.0 - (args.suppression_threshold - minimum_factor) / (1.0 - minimum_factor))
    )
    suppression_distance_cutoff = cutoff_normalized * kernel_bandwidth * distance_scale
    for name, values in pair_distances.items():
        direct_summary["pair_distance_squared_l2"][name][
            "fraction_below_suppression_distance"
        ] = float(np.mean(values < suppression_distance_cutoff)) if len(values) else None

    print("computing sampled direct-latent nearest neighbors...")
    neighbor_summary = nearest_direct_neighbors(
        state_space,
        direct_latents,
        args.nearest_query_count,
        device,
        max(1, args.nearest_batch_size),
        rng,
    )
    root_count = min(args.root_count, len(state_space.poses))
    root_indices = np.sort(rng.choice(len(state_space.poses), size=root_count, replace=False))
    print(
        f"evaluating all action sequences through depth {args.horizon} "
        f"from {root_count} sampled roots..."
    )
    rollout_summary, path_rows = diagnose_rollouts(
        model=model,
        state_space=state_space,
        direct_latents=direct_latents,
        latent_shape=latent_shape,
        root_indices=root_indices,
        horizon=args.horizon,
        root_batch_size=max(1, args.root_batch_size),
        distance_scale=distance_scale,
        kernel_bandwidth=kernel_bandwidth,
        minimum_factor=minimum_factor,
        suppression_threshold=float(args.suppression_threshold),
        device=device,
    )

    output_prefix = args.output_prefix
    if output_prefix is None:
        output_prefix = run_dir / "diagnostics" / (
            "raw_latent_geometry_" + checkpoint_path.name.replace(".pth.tar", "")
        )
    output_prefix = output_prefix.resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_train_iter": checkpoint_iter,
        "checkpoint_env_step": int(checkpoint.get("last_step", -1)),
        "device": str(device),
        "latent_shape": list(latent_shape),
        "ensemble_size": ensemble_size,
        "root_count": root_count,
        "horizon": int(args.horizon),
        "distance_scale": distance_scale,
        "distance_scale_source": distance_scale_source,
        "global_direct_one_step_distance_scale": global_distance_scale,
        "final_logged_raw_memory_distance_scale": logged_distance_scale,
        "kernel_bandwidth": kernel_bandwidth,
        "minimum_reward_factor": minimum_factor,
        "suppression_threshold": float(args.suppression_threshold),
        "suppression_distance_cutoff": suppression_distance_cutoff,
        "direct_geometry": direct_summary,
        "nearest_direct_neighbors": neighbor_summary,
        "rollout_by_depth": rollout_summary,
    }
    json_path = output_prefix.with_suffix(".json")
    json_path.write_text(json.dumps(_json_sanitize(summary), indent=2) + "\n")
    depth_rows = []
    for depth, values in rollout_summary.items():
        flat = {"depth": depth}
        for key, value in values.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    flat[f"{key}_{subkey}"] = subvalue
            else:
                flat[key] = value
        depth_rows.append(flat)
    depth_csv = output_prefix.with_name(output_prefix.name + "_depth.csv")
    _write_csv(depth_csv, depth_rows)
    path_csv = output_prefix.with_name(output_prefix.name + "_paths.csv")
    if not args.no_path_records:
        _write_csv(path_csv, path_rows)

    print("\nDirect observation/encoder geometry")
    print(
        f"  unique RGB observations: {direct_summary['unique_rgb_observations']} / "
        f"{direct_summary['reachable_poses']} poses"
    )
    print(
        "  poses sharing RGB with another physical pose: "
        f"{direct_summary['poses_with_rgb_alias_fraction']:.3f}"
    )
    distinct = direct_summary["pair_distance_squared_l2"]["random_distinct_observation"]
    print(f"  distinct-observation raw distance median: {distinct['median']:.6g}")
    print(f"  one-step distance scale: {distance_scale:.6g}")
    print(f"  factor<{args.suppression_threshold:g} distance cutoff: {suppression_distance_cutoff:.6g}")
    print("\nLabeled recurrent rollout")
    for depth, values in rollout_summary.items():
        print(
            f"  depth {depth}: factor={values['mean_reward_factor']:.3f}, "
            f"suppressed={values['suppressed_fraction']:.3f}, "
            f"pose_revisit={values['pose_revisit_fraction']:.3f}, "
            f"false_suppression_physical={values['physical_new_false_suppression_rate']:.3f}, "
            f"false_suppression_observation={values['observation_new_false_suppression_rate']:.3f}"
        )
    print(f"\nsummary: {json_path}")
    print(f"depth CSV: {depth_csv}")
    if not args.no_path_records:
        print(f"path CSV: {path_csv}")


if __name__ == "__main__":
    main()
