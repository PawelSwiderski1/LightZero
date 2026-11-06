"""Closed-loop episodic-novelty planning in Branching Rooms.

The planner scores every short action suffix from the current real episode
history, executes only the first action, observes the real transition, and
replans.  The online condition trains only on suffixes that were actually
executed.  It never queries the environment transition function for labels of
hypothetical suffixes.
"""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper

if __package__:
    from .run_branching_rooms_action_novelty import (
        LABEL_NAMES,
        NOVEL,
        ActionNoveltyPredictor,
        OnlinePredictorContext,
        _class_weights,
        build_novelty_targets,
        classification_metrics,
        load_dataset,
        state_ids_from_observations,
        state_ids_from_poses,
    )
else:
    from run_branching_rooms_action_novelty import (
        LABEL_NAMES,
        NOVEL,
        ActionNoveltyPredictor,
        OnlinePredictorContext,
        _class_weights,
        build_novelty_targets,
        classification_metrics,
        load_dataset,
        state_ids_from_observations,
        state_ids_from_poses,
    )
from zoo.minigrid.envs.branching_rooms_env import (
    BRANCHING_ROOMS_ENV_ID,
    BranchingRoomsEnv,
)


CONDITIONS = (
    "random",
    "frozen_predictor",
    "online_predictor",
    "frozen_topk",
    "online_prioritized_topk",
)
DEFAULT_CONDITIONS = (
    "random",
    "frozen_predictor",
    "frozen_topk",
    "online_prioritized_topk",
)


@dataclass(frozen=True)
class Trajectory:
    observations: np.ndarray
    actions: np.ndarray
    poses: np.ndarray
    rooms: Tuple[Tuple[int, int] | None, ...]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline-run",
        type=Path,
        default=Path(".local_runs/branching_rooms_action_novelty_seed0"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".local_runs/branching_rooms_online_action_novelty_seed0"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--conditions", nargs="+", choices=CONDITIONS, default=DEFAULT_CONDITIONS
    )
    parser.add_argument("--episodes-per-condition", type=int, default=24)
    parser.add_argument("--episode-length", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--tile-size", type=int, default=None)
    parser.add_argument("--epsilon", type=float, default=0.10)
    parser.add_argument("--novelty-discount", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--online-update-interval", type=int, default=8)
    parser.add_argument("--updates-per-interval", type=int, default=4)
    parser.add_argument("--online-roots-per-update", type=int, default=32)
    parser.add_argument("--offline-roots-per-update", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--offline-loss-weight", type=float, default=0.5)
    parser.add_argument("--priority-false-novel-weight", type=float, default=2.0)
    parser.add_argument("--online-replay-max-episodes", type=int, default=64)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--smoke", action="store_true", help="Run a tiny implementation check."
    )
    return parser


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device(name)


def enumerate_action_suffixes(action_count: int, horizon: int) -> np.ndarray:
    return np.asarray(
        list(itertools.product(range(int(action_count)), repeat=int(horizon))),
        dtype=np.int64,
    )


def discounted_branch_scores(
    probabilities: np.ndarray, discount: float
) -> np.ndarray:
    """Return cumulative predicted novelty for [suffix, depth, class]."""
    if probabilities.ndim != 3 or probabilities.shape[-1] != len(LABEL_NAMES):
        raise ValueError("Expected probabilities shaped [suffix, depth, class].")
    weights = np.power(float(discount), np.arange(probabilities.shape[1]))
    return np.sum(probabilities[..., NOVEL] * weights[None, :], axis=1)


def choose_planned_action(
    probabilities: np.ndarray,
    suffixes: np.ndarray,
    discount: float,
    epsilon: float,
    rng: np.random.Generator,
    aggregation: str = "max",
    top_k: int = 16,
) -> Tuple[int, float, bool]:
    scores = discounted_branch_scores(probabilities, discount)
    action_values = []
    for action in range(int(suffixes[:, 0].max()) + 1):
        action_scores = scores[suffixes[:, 0] == action]
        if aggregation == "max":
            value = float(action_scores.max())
        elif aggregation == "topk":
            selected_count = min(int(top_k), int(action_scores.size))
            if selected_count <= 0:
                raise ValueError("top_k must be positive.")
            value = float(np.partition(action_scores, -selected_count)[-selected_count:].mean())
        else:
            raise ValueError(f"Unknown planner aggregation: {aggregation}")
        action_values.append(value)
    action_values = np.asarray(action_values, dtype=np.float64)
    exploratory = bool(rng.random() < float(epsilon))
    if exploratory:
        action = int(rng.integers(0, action_values.size))
        return action, float(action_values[action]), True

    best_score = float(action_values.max())
    candidates = np.flatnonzero(
        np.isclose(action_values, best_score, rtol=1e-7, atol=1e-9)
    )
    return int(rng.choice(candidates)), best_score, False


def action_suffix_indices(actions: np.ndarray, action_count: int, horizon: int) -> np.ndarray:
    """Indices matching itertools.product's lexicographic suffix order."""
    actions = np.asarray(actions, dtype=np.int64)
    root_count = actions.size - int(horizon) + 1
    if root_count <= 0:
        return np.empty((0,), dtype=np.int64)
    powers = np.power(int(action_count), np.arange(horizon - 1, -1, -1))
    return np.asarray(
        [int(np.dot(actions[root : root + horizon], powers)) for root in range(root_count)],
        dtype=np.int64,
    )


def model_from_checkpoint(
    checkpoint_path: Path, device: torch.device
) -> Tuple[ActionNoveltyPredictor, Mapping[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("model_name") != "full_history":
        raise ValueError("The closed-loop experiment requires the full-history model.")
    model = ActionNoveltyPredictor(
        action_space_size=3,
        embedding_dim=int(checkpoint["embedding_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        use_history=True,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    return model, checkpoint


def make_env(episode_length: int, tile_size: int):
    env = gym.make(
        BRANCHING_ROOMS_ENV_ID,
        max_steps=int(episode_length),
        agent_view_size=7,
    )
    env = RGBImgPartialObsWrapper(env, tile_size=int(tile_size))
    return ImgObsWrapper(env)


def observation_tensor(observation: np.ndarray, device: torch.device) -> torch.Tensor:
    channel_first = np.transpose(np.asarray(observation), (2, 0, 1)).astype(
        np.uint8, copy=False
    )
    return torch.from_numpy(channel_first.copy()).unsqueeze(0).to(device)


def rebuild_context(
    model: ActionNoveltyPredictor,
    observations: Sequence[np.ndarray],
    actions: Sequence[int],
    device: torch.device,
) -> OnlinePredictorContext:
    model.eval()
    with torch.no_grad():
        context = model.start_online_context(observation_tensor(observations[0], device))
        for action, observation in zip(actions, observations[1:]):
            context = model.advance_online_context(
                context,
                torch.tensor([int(action)], device=device),
                observation_tensor(observation, device),
            )
    return context


def trajectory_arrays(
    observations: Sequence[np.ndarray],
    actions: Sequence[int],
    poses: Sequence[Tuple[int, int, int]],
    rooms: Sequence[Tuple[int, int] | None],
) -> Trajectory:
    return Trajectory(
        observations=np.stack(
            [np.transpose(np.asarray(obs), (2, 0, 1)).astype(np.uint8) for obs in observations]
        ),
        actions=np.asarray(actions, dtype=np.int64),
        poses=np.asarray(poses, dtype=np.int16),
        rooms=tuple(rooms),
    )


def predict_trajectory(
    model: ActionNoveltyPredictor,
    trajectory: Trajectory,
    horizon: int,
    device: torch.device,
    root_indices: np.ndarray | None = None,
) -> np.ndarray:
    model.eval()
    observations = torch.from_numpy(trajectory.observations).unsqueeze(0).to(device)
    actions = torch.from_numpy(trajectory.actions).unsqueeze(0).to(device)
    roots = None
    if root_indices is not None:
        roots = torch.from_numpy(np.asarray(root_indices, dtype=np.int64)).to(device)
    with torch.no_grad():
        logits = model(observations, actions, horizon, root_indices=roots)
        return torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()


def trajectory_targets(trajectory: Trajectory, horizon: int) -> np.ndarray:
    return build_novelty_targets(
        state_ids_from_observations(trajectory.observations), horizon
    )


def metrics_by_depth(
    targets: np.ndarray, probabilities: np.ndarray
) -> Mapping[str, object]:
    result: MutableMapping[str, object] = {
        "overall": classification_metrics(targets, probabilities),
        "by_horizon": {},
    }
    for depth in range(targets.shape[1]):
        result["by_horizon"][str(depth + 1)] = classification_metrics(
            targets[:, depth], probabilities[:, depth]
        )
    return result


def gather_decision_probabilities(
    planning_probabilities: Sequence[np.ndarray],
    actions: np.ndarray,
    action_count: int,
    horizon: int,
) -> np.ndarray:
    indices = action_suffix_indices(actions, action_count, horizon)
    if indices.size == 0:
        return np.empty((0, horizon, len(LABEL_NAMES)), dtype=np.float32)
    if len(planning_probabilities) < indices.size:
        raise ValueError("Missing root-time planner predictions.")
    return np.stack(
        [planning_probabilities[root][index] for root, index in enumerate(indices)]
    )


def sampled_roots(root_count: int, count: int, rng: np.random.Generator) -> np.ndarray:
    sample_count = min(int(root_count), int(count))
    if sample_count == root_count:
        return np.arange(root_count, dtype=np.int64)
    return np.sort(rng.choice(root_count, size=sample_count, replace=False)).astype(
        np.int64
    )


def weighted_trajectory_loss(
    model: ActionNoveltyPredictor,
    trajectory: Trajectory,
    horizon: int,
    root_indices: np.ndarray,
    class_weights: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    observations = torch.from_numpy(trajectory.observations).unsqueeze(0).to(device)
    actions = torch.from_numpy(trajectory.actions).unsqueeze(0).to(device)
    targets = torch.from_numpy(trajectory_targets(trajectory, horizon)).to(device)
    roots = torch.from_numpy(root_indices).to(device)
    logits = model(observations, actions, horizon, root_indices=roots)
    return F.cross_entropy(
        logits.reshape(-1, len(LABEL_NAMES)),
        targets[:, :][roots].reshape(-1),
        weight=class_weights,
    )


def binary_logits(logits: torch.Tensor) -> torch.Tensor:
    revisit_logit = torch.logsumexp(logits[..., :NOVEL], dim=-1)
    return torch.stack([revisit_logit, logits[..., NOVEL]], dim=-1)


def binary_targets(targets: torch.Tensor) -> torch.Tensor:
    return (targets == NOVEL).to(dtype=torch.long)


def binary_trajectory_loss(
    model: ActionNoveltyPredictor,
    trajectory: Trajectory,
    horizon: int,
    root_indices: np.ndarray,
    class_weights: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    observations = torch.from_numpy(trajectory.observations).unsqueeze(0).to(device)
    actions = torch.from_numpy(trajectory.actions).unsqueeze(0).to(device)
    targets = torch.from_numpy(trajectory_targets(trajectory, horizon)).to(device)
    roots = torch.from_numpy(root_indices).to(device)
    logits = model(observations, actions, horizon, root_indices=roots)
    selected_targets = targets[roots]
    return F.cross_entropy(
        binary_logits(logits).reshape(-1, 2),
        binary_targets(selected_targets).reshape(-1),
        weight=class_weights,
    )


def binary_root_priorities(
    targets: np.ndarray,
    probabilities: np.ndarray,
    false_novel_weight: float,
) -> np.ndarray:
    """Prioritize roots with high binary error, especially false novelty."""
    target_novel = (np.asarray(targets) == NOVEL).astype(np.float64)
    novel_probability = np.asarray(probabilities)[..., NOVEL].astype(np.float64)
    epsilon = 1e-7
    binary_cross_entropy = -(
        target_novel * np.log(np.clip(novel_probability, epsilon, 1.0))
        + (1.0 - target_novel)
        * np.log(np.clip(1.0 - novel_probability, epsilon, 1.0))
    )
    false_novelty = (1.0 - target_novel) * novel_probability
    return binary_cross_entropy.mean(axis=1) + float(false_novel_weight) * false_novelty.max(axis=1)


def prioritized_roots(
    model: ActionNoveltyPredictor,
    trajectory: Trajectory,
    horizon: int,
    count: int,
    false_novel_weight: float,
    device: torch.device,
    mandatory_roots: np.ndarray | None = None,
) -> np.ndarray:
    root_count = trajectory.actions.size - int(horizon) + 1
    probabilities = predict_trajectory(model, trajectory, horizon, device)
    targets = trajectory_targets(trajectory, horizon)
    priorities = binary_root_priorities(
        targets, probabilities, false_novel_weight
    )
    selected: List[int] = []
    if mandatory_roots is not None:
        selected.extend(
            int(root)
            for root in np.asarray(mandatory_roots, dtype=np.int64)
            if 0 <= int(root) < root_count
        )
    selected = list(dict.fromkeys(selected))
    target_count = min(root_count, max(int(count), len(selected)))
    for root in np.argsort(-priorities, kind="stable"):
        root = int(root)
        if root not in selected:
            selected.append(root)
        if len(selected) >= target_count:
            break
    return np.asarray(sorted(selected), dtype=np.int64)


def binary_class_weights(targets: np.ndarray) -> torch.Tensor:
    binary = (np.asarray(targets) == NOVEL).astype(np.int64)
    counts = np.bincount(binary.reshape(-1), minlength=2).astype(np.float64)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(np.minimum(weights, 5.0), dtype=torch.float32)


def bundle_episode(bundle, episode: int) -> Trajectory:
    rooms = tuple(
        BranchingRoomsEnv.room_for_position((int(pose[0]), int(pose[1])))
        for pose in bundle.poses[episode]
    )
    return Trajectory(
        observations=bundle.observations[episode],
        actions=bundle.actions[episode].astype(np.int64, copy=False),
        poses=bundle.poses[episode],
        rooms=rooms,
    )


def selected_root_metrics(
    model: ActionNoveltyPredictor,
    trajectory: Trajectory,
    horizon: int,
    root_indices: np.ndarray,
    device: torch.device,
) -> Mapping[str, object]:
    probabilities = predict_trajectory(
        model, trajectory, horizon, device, root_indices=root_indices
    )
    targets = trajectory_targets(trajectory, horizon)[root_indices]
    return metrics_by_depth(targets, probabilities)


def online_update(
    model: ActionNoveltyPredictor,
    optimizer: torch.optim.Optimizer,
    current: Trajectory,
    offline_episodes: Sequence[Trajectory],
    class_weights: torch.Tensor,
    args: argparse.Namespace,
    rng: np.random.Generator,
    device: torch.device,
) -> float:
    if optimizer is None:
        raise ValueError("Online updates require an optimizer.")
    current_root_count = current.actions.size - int(args.horizon) + 1
    if current_root_count <= 0:
        return 0.0
    losses = []
    model.train()
    for _ in range(int(args.updates_per_interval)):
        current_roots = sampled_roots(
            current_root_count, args.online_roots_per_update, rng
        )
        current_loss = weighted_trajectory_loss(
            model, current, args.horizon, current_roots, class_weights, device
        )
        offline = offline_episodes[int(rng.integers(0, len(offline_episodes)))]
        offline_root_count = offline.actions.size - int(args.horizon) + 1
        offline_roots = sampled_roots(
            offline_root_count, args.offline_roots_per_update, rng
        )
        offline_loss = weighted_trajectory_loss(
            model, offline, args.horizon, offline_roots, class_weights, device
        )
        loss = current_loss + float(args.offline_loss_weight) * offline_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses))


def prioritized_online_update(
    model: ActionNoveltyPredictor,
    optimizer: torch.optim.Optimizer,
    current: Trajectory,
    new_roots: np.ndarray,
    online_replay: Sequence[Trajectory],
    offline_episodes: Sequence[Trajectory],
    binary_weights: torch.Tensor,
    args: argparse.Namespace,
    rng: np.random.Generator,
    device: torch.device,
) -> float:
    if optimizer is None:
        raise ValueError("Online updates require an optimizer.")
    losses = []
    model.train()
    for update_index in range(int(args.updates_per_interval)):
        use_current = update_index % 2 == 0 or not online_replay
        if use_current:
            online_trajectory = current
            mandatory = new_roots
        else:
            online_trajectory = online_replay[
                int(rng.integers(0, len(online_replay)))
            ]
            mandatory = None
        online_roots = prioritized_roots(
            model,
            online_trajectory,
            args.horizon,
            args.online_roots_per_update,
            args.priority_false_novel_weight,
            device,
            mandatory_roots=mandatory,
        )
        model.train()
        online_loss = binary_trajectory_loss(
            model,
            online_trajectory,
            args.horizon,
            online_roots,
            binary_weights,
            device,
        )

        offline = offline_episodes[int(rng.integers(0, len(offline_episodes)))]
        offline_root_count = offline.actions.size - int(args.horizon) + 1
        offline_roots = sampled_roots(
            offline_root_count, args.offline_roots_per_update, rng
        )
        offline_loss = binary_trajectory_loss(
            model,
            offline,
            args.horizon,
            offline_roots,
            binary_weights,
            device,
        )
        loss = online_loss + float(args.offline_loss_weight) * offline_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses))


def compact_binary_metrics(metrics: Mapping[str, object]) -> Mapping[str, object]:
    overall = metrics["overall"]
    binary = overall["binary_revisit"]
    return {
        "macro_f1": overall["macro_f1"],
        "accuracy": overall["accuracy"],
        "revisit_auroc": binary["auroc"],
        "revisit_brier": binary["brier"],
        "revisit_ece": binary["ece"],
        "revisit_recall": binary["recall"],
        "root_revisit_recall": overall["per_class"][LABEL_NAMES[0]]["recall"],
        "branch_revisit_recall": overall["per_class"][LABEL_NAMES[1]]["recall"],
        "novel_recall": overall["per_class"][LABEL_NAMES[2]]["recall"],
    }


def summarize_update_correction(
    update_records: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    if not update_records:
        return {
            "online_before_revisit_brier": None,
            "online_after_revisit_brier": None,
            "online_revisit_brier_change": None,
            "online_before_revisit_recall": None,
            "online_after_revisit_recall": None,
            "online_revisit_recall_change": None,
        }
    before_brier = finite_mean(
        [record["before"]["revisit_brier"] for record in update_records]
    )
    after_brier = finite_mean(
        [record["after"]["revisit_brier"] for record in update_records]
    )
    before_recall = finite_mean(
        [record["before"]["revisit_recall"] for record in update_records]
    )
    after_recall = finite_mean(
        [record["after"]["revisit_recall"] for record in update_records]
    )
    return {
        "online_before_revisit_brier": before_brier,
        "online_after_revisit_brier": after_brier,
        "online_revisit_brier_change": (
            None if before_brier is None or after_brier is None else after_brier - before_brier
        ),
        "online_before_revisit_recall": before_recall,
        "online_after_revisit_recall": after_recall,
        "online_revisit_recall_change": (
            None
            if before_recall is None or after_recall is None
            else after_recall - before_recall
        ),
    }


def coverage_metrics(trajectory: Trajectory) -> Mapping[str, object]:
    observation_ids = state_ids_from_observations(trajectory.observations)
    position_ids = state_ids_from_poses(trajectory.poses[:, :2])
    pose_ids = state_ids_from_poses(trajectory.poses)
    rooms = {room for room in trajectory.rooms if room is not None}
    dead_ends = BranchingRoomsEnv.dead_end_rooms()
    transition_count = max(trajectory.actions.size, 1)
    return {
        "unique_observations": int(np.unique(observation_ids).size),
        "unique_positions": int(np.unique(position_ids).size),
        "unique_poses": int(np.unique(pose_ids).size),
        "rooms": int(len(rooms)),
        "dead_ends": int(len(rooms & dead_ends)),
        "observation_revisit_fraction": float(
            np.sum([observation_ids[i] in observation_ids[:i] for i in range(1, observation_ids.size)])
            / transition_count
        ),
        "position_revisit_fraction": float(
            np.sum([position_ids[i] in position_ids[:i] for i in range(1, position_ids.size)])
            / transition_count
        ),
        "pose_revisit_fraction": float(
            np.sum([pose_ids[i] in pose_ids[:i] for i in range(1, pose_ids.size)])
            / transition_count
        ),
    }


def run_episode(
    condition: str,
    model: ActionNoveltyPredictor,
    optimizer: torch.optim.Optimizer | None,
    suffixes: np.ndarray,
    suffix_tensor: torch.Tensor,
    offline_episodes: Sequence[Trajectory],
    class_weights: torch.Tensor,
    binary_weights: torch.Tensor,
    online_replay: Sequence[Trajectory],
    args: argparse.Namespace,
    episode_index: int,
    planner_rng: np.random.Generator,
    training_rng: np.random.Generator,
    device: torch.device,
) -> Tuple[
    Trajectory,
    Mapping[str, object],
    List[Mapping[str, object]],
    Mapping[str, np.ndarray],
]:
    env = make_env(args.episode_length, args.tile_size)
    env_seed = int(args.seed) * 1_000_003 + episode_index * 101 + 31
    observation, info = env.reset(seed=env_seed)
    observations: List[np.ndarray] = [np.asarray(observation).copy()]
    actions: List[int] = []
    poses = [
        (
            int(info["agent_position"][0]),
            int(info["agent_position"][1]),
            int(info["agent_direction"]),
        )
    ]
    rooms = [info["room_id"]]
    planning_probabilities: List[np.ndarray] = []
    planned_scores: List[float] = []
    exploratory_flags: List[bool] = []
    exploratory_steps = 0
    update_records: List[Mapping[str, object]] = []
    next_unmeasured_root = 0
    context = rebuild_context(model, observations, actions, device)
    planner_aggregation = (
        "topk" if condition in ("frozen_topk", "online_prioritized_topk") else "max"
    )

    try:
        for step in range(int(args.episode_length)):
            model.eval()
            if condition == "random":
                action = int(planner_rng.integers(0, 3))
            else:
                with torch.no_grad():
                    logits = model.score_action_suffixes(context, suffix_tensor)
                    probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
                planning_probabilities.append(probabilities)
                action, score, exploratory = choose_planned_action(
                    probabilities,
                    suffixes,
                    args.novelty_discount,
                    args.epsilon,
                    planner_rng,
                    aggregation=planner_aggregation,
                    top_k=args.top_k,
                )
                planned_scores.append(score)
                exploratory_steps += int(exploratory)
                exploratory_flags.append(exploratory)

            observation, _, terminated, truncated, info = env.step(action)
            actions.append(action)
            observations.append(np.asarray(observation).copy())
            poses.append(
                (
                    int(info["agent_position"][0]),
                    int(info["agent_position"][1]),
                    int(info["agent_direction"]),
                )
            )
            rooms.append(info["room_id"])
            if condition != "random":
                with torch.no_grad():
                    context = model.advance_online_context(
                        context,
                        torch.tensor([action], device=device),
                        observation_tensor(observation, device),
                    )

            should_update = (
                condition in ("online_predictor", "online_prioritized_topk")
                and len(actions) >= int(args.horizon)
                and (
                    len(actions) % int(args.online_update_interval) == 0
                    or terminated
                    or truncated
                )
            )
            if should_update:
                current = trajectory_arrays(observations, actions, poses, rooms)
                root_count = current.actions.size - int(args.horizon) + 1
                new_roots = np.arange(next_unmeasured_root, root_count, dtype=np.int64)
                if new_roots.size:
                    before = selected_root_metrics(
                        model, current, args.horizon, new_roots, device
                    )
                    if condition == "online_prioritized_topk":
                        training_loss = prioritized_online_update(
                            model,
                            optimizer,
                            current,
                            new_roots,
                            online_replay,
                            offline_episodes,
                            binary_weights,
                            args,
                            training_rng,
                            device,
                        )
                    else:
                        training_loss = online_update(
                            model,
                            optimizer,
                            current,
                            offline_episodes,
                            class_weights,
                            args,
                            training_rng,
                            device,
                        )
                    after = selected_root_metrics(
                        model, current, args.horizon, new_roots, device
                    )
                    update_records.append(
                        {
                            "step": len(actions),
                            "new_root_count": int(new_roots.size),
                            "training_loss": training_loss,
                            "before": compact_binary_metrics(before),
                            "after": compact_binary_metrics(after),
                        }
                    )
                    next_unmeasured_root = root_count
                    context = rebuild_context(model, observations, actions, device)

            if terminated or truncated:
                break
    finally:
        env.close()

    if len(actions) != int(args.episode_length):
        raise RuntimeError(
            f"Expected {args.episode_length} transitions, collected {len(actions)}."
        )
    trajectory = trajectory_arrays(observations, actions, poses, rooms)
    targets = trajectory_targets(trajectory, args.horizon)
    if condition == "random":
        decision_probabilities = predict_trajectory(
            model, trajectory, args.horizon, device
        )
    else:
        decision_probabilities = gather_decision_probabilities(
            planning_probabilities,
            trajectory.actions,
            action_count=3,
            horizon=args.horizon,
        )
    decision_metrics = metrics_by_depth(targets, decision_probabilities)
    result: MutableMapping[str, object] = {
        **coverage_metrics(trajectory),
        **summarize_update_correction(update_records),
        "decision_metrics": decision_metrics,
        "mean_selected_branch_score": (
            float(np.mean(planned_scores)) if planned_scores else None
        ),
        "epsilon_action_fraction": float(exploratory_steps / max(len(actions), 1)),
        "online_update_count": len(update_records),
        "planner_aggregation": planner_aggregation,
        "planner_top_k": int(args.top_k) if planner_aggregation == "topk" else 1,
        "online_update_mode": (
            "prioritized_binary_replay"
            if condition == "online_prioritized_topk"
            else "legacy_three_class" if condition == "online_predictor" else "none"
        ),
    }
    diagnostic_arrays = {
        "targets": targets,
        "decision_probabilities": decision_probabilities,
        "planned_scores": np.asarray(planned_scores, dtype=np.float32),
        "exploratory_flags": np.asarray(exploratory_flags, dtype=np.bool_),
    }
    return trajectory, result, update_records, diagnostic_arrays


def finite_mean(values: Sequence[object]) -> float | None:
    numbers = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(numbers)) if numbers else None


def summarize(rows: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    result: MutableMapping[str, object] = {}
    for condition in CONDITIONS:
        selected = [row for row in rows if row["condition"] == condition]
        if not selected:
            continue
        fields = (
            "unique_observations",
            "unique_positions",
            "unique_poses",
            "rooms",
            "dead_ends",
            "observation_revisit_fraction",
            "position_revisit_fraction",
            "pose_revisit_fraction",
            "decision_macro_f1",
            "decision_revisit_auroc",
            "decision_revisit_brier",
            "decision_revisit_recall",
            "decision_novel_recall",
            "final_horizon_macro_f1",
            "final_horizon_revisit_auroc",
            "mean_selected_branch_score",
            "online_before_revisit_brier",
            "online_after_revisit_brier",
            "online_revisit_brier_change",
            "online_before_revisit_recall",
            "online_after_revisit_recall",
            "online_revisit_recall_change",
        )
        result[condition] = {
            "episode_count": len(selected),
            **{field: finite_mean([row[field] for row in selected]) for field in fields},
            "cumulative_unique_positions": int(selected[-1]["cumulative_unique_positions"]),
            "cumulative_unique_poses": int(selected[-1]["cumulative_unique_poses"]),
            "cumulative_rooms": int(selected[-1]["cumulative_rooms"]),
            "cumulative_dead_ends": int(selected[-1]["cumulative_dead_ends"]),
        }
    return result


def flatten_episode_row(
    condition: str,
    episode: int,
    result: Mapping[str, object],
    cumulative_positions: set,
    cumulative_poses: set,
    cumulative_rooms: set,
) -> Mapping[str, object]:
    decision = result["decision_metrics"]
    compact = compact_binary_metrics(decision)
    final_depth = decision["by_horizon"][str(max(int(key) for key in decision["by_horizon"]))]
    final_binary = final_depth["binary_revisit"]
    return {
        "condition": condition,
        "episode": episode,
        **{key: value for key, value in result.items() if key != "decision_metrics"},
        "decision_macro_f1": compact["macro_f1"],
        "decision_revisit_auroc": compact["revisit_auroc"],
        "decision_revisit_brier": compact["revisit_brier"],
        "decision_revisit_ece": compact["revisit_ece"],
        "decision_revisit_recall": compact["revisit_recall"],
        "decision_novel_recall": compact["novel_recall"],
        "final_horizon_macro_f1": final_depth["macro_f1"],
        "final_horizon_revisit_auroc": final_binary["auroc"],
        "cumulative_unique_positions": len(cumulative_positions),
        "cumulative_unique_poses": len(cumulative_poses),
        "cumulative_rooms": len(cumulative_rooms),
        "cumulative_dead_ends": len(cumulative_rooms & BranchingRoomsEnv.dead_end_rooms()),
    }


def write_progress(
    output_dir: Path,
    flat_rows: Sequence[Mapping[str, object]],
    detailed_results: Mapping[str, object],
) -> None:
    with (output_dir / "episodes.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)
    (output_dir / "episode_details.json").write_text(
        json.dumps(detailed_results, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summarize(flat_rows), indent=2), encoding="utf-8"
    )


def apply_smoke_overrides(args: argparse.Namespace) -> None:
    if not args.smoke:
        return
    args.conditions = DEFAULT_CONDITIONS
    args.episodes_per_condition = 1
    args.episode_length = 12
    args.horizon = min(int(args.horizon or 3), 3)
    args.online_update_interval = 4
    args.updates_per_interval = 1
    args.online_roots_per_update = 4
    args.offline_roots_per_update = 4


def main() -> None:
    args = build_parser().parse_args()
    checkpoint_path = args.offline_run / "full_history" / "best.pt"
    dataset_path = args.offline_run / "experienced_trajectories.npz"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing pretrained predictor: {checkpoint_path}")
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Missing offline trajectories: {dataset_path}")

    device = resolve_device(args.device)
    bundle = load_dataset(dataset_path)
    base_model, checkpoint = model_from_checkpoint(checkpoint_path, device)
    if args.episode_length is None:
        args.episode_length = int(bundle.metadata["episode_length"])
    if args.horizon is None:
        args.horizon = int(checkpoint["horizon"])
    if args.tile_size is None:
        args.tile_size = int(bundle.metadata["tile_size"])
    apply_smoke_overrides(args)
    if not 0.0 <= float(args.epsilon) <= 1.0:
        raise ValueError("epsilon must be between zero and one.")
    if int(args.online_update_interval) <= 0:
        raise ValueError("online-update-interval must be positive.")
    if int(args.top_k) <= 0:
        raise ValueError("top-k must be positive.")
    if int(args.online_replay_max_episodes) <= 0:
        raise ValueError("online-replay-max-episodes must be positive.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_dir = args.output_dir / "trajectories"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    suffixes = enumerate_action_suffixes(3, args.horizon)
    suffix_tensor = torch.from_numpy(suffixes).to(device)
    train_indices = np.flatnonzero(bundle.split_ids == 0)
    offline_episodes = [bundle_episode(bundle, int(index)) for index in train_indices]
    class_weights = _class_weights(
        bundle.observation_targets[bundle.split_ids == 0]
    ).to(device)
    binary_weights = binary_class_weights(
        bundle.observation_targets[bundle.split_ids == 0]
    ).to(device)

    arguments = vars(args).copy()
    arguments["offline_run"] = str(args.offline_run)
    arguments["output_dir"] = str(args.output_dir)
    arguments["device"] = str(device)
    arguments["checkpoint"] = str(checkpoint_path)
    arguments["counterfactual_transition_queries"] = 0
    (args.output_dir / "arguments.json").write_text(
        json.dumps(arguments, indent=2), encoding="utf-8"
    )

    flat_rows: List[Mapping[str, object]] = []
    detailed_results: MutableMapping[str, object] = {}
    for condition_index, condition in enumerate(args.conditions):
        model = copy.deepcopy(base_model).to(device)
        optimizer = None
        if condition in ("online_predictor", "online_prioritized_topk"):
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(args.learning_rate),
                weight_decay=float(args.weight_decay),
            )
        # Frozen and online planning use common random numbers. Optimizer replay
        # sampling has a separate stream, so it cannot shift epsilon decisions.
        planner_stream = 0 if condition == "random" else 1
        planner_rng = np.random.default_rng(
            int(args.seed) * 1009 + planner_stream * 7919 + 43
        )
        training_rng = np.random.default_rng(
            int(args.seed) * 2003 + condition_index * 1237 + 59
        )
        cumulative_positions: set = set()
        cumulative_poses: set = set()
        cumulative_rooms: set = set()
        online_replay: List[Trajectory] = []
        condition_details = []
        for episode in range(int(args.episodes_per_condition)):
            trajectory, result, updates, diagnostic_arrays = run_episode(
                condition,
                model,
                optimizer,
                suffixes,
                suffix_tensor,
                offline_episodes,
                class_weights,
                binary_weights,
                online_replay,
                args,
                episode,
                planner_rng,
                training_rng,
                device,
            )
            cumulative_positions.update(map(tuple, trajectory.poses[:, :2].tolist()))
            cumulative_poses.update(map(tuple, trajectory.poses.tolist()))
            cumulative_rooms.update(room for room in trajectory.rooms if room is not None)
            flat = flatten_episode_row(
                condition,
                episode,
                result,
                cumulative_positions,
                cumulative_poses,
                cumulative_rooms,
            )
            flat_rows.append(flat)
            condition_details.append(
                {
                    "episode": episode,
                    "metrics": result,
                    "updates": updates,
                }
            )
            room_array = np.asarray(
                [(-1, -1) if room is None else room for room in trajectory.rooms],
                dtype=np.int16,
            )
            np.savez_compressed(
                trajectory_dir / f"{condition}_episode_{episode:03d}.npz",
                observations=trajectory.observations,
                actions=trajectory.actions,
                poses=trajectory.poses,
                rooms=room_array,
                **diagnostic_arrays,
            )
            detailed_results[condition] = condition_details
            write_progress(args.output_dir, flat_rows, detailed_results)
            if condition in ("online_predictor", "online_prioritized_topk"):
                online_replay.append(trajectory)
                if len(online_replay) > int(args.online_replay_max_episodes):
                    del online_replay[0]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "source_checkpoint": str(checkpoint_path),
                        "episodes": episode + 1,
                    },
                    args.output_dir / f"{condition}_latest.pt",
                )
            print(
                f"{condition} episode {episode + 1}/{args.episodes_per_condition}: "
                f"positions={flat['unique_positions']} rooms={flat['rooms']} "
                f"dead_ends={flat['dead_ends']} revisit={flat['position_revisit_fraction']:.3f} "
                f"decision_auc={flat['decision_revisit_auroc']} updates={flat['online_update_count']}"
            )
        detailed_results[condition] = condition_details
        if condition in ("online_predictor", "online_prioritized_topk"):
            torch.save(
                {
                    "model": model.state_dict(),
                    "source_checkpoint": str(checkpoint_path),
                    "episodes": int(args.episodes_per_condition),
                },
                args.output_dir / f"{condition}_final.pt",
            )

    summary = summarize(flat_rows)
    write_progress(args.output_dir, flat_rows, detailed_results)
    print(json.dumps(summary, indent=2))
    print(f"wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
