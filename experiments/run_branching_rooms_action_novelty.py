"""Supervised short-horizon episodic-novelty prediction in Branching Rooms.

The experiment deliberately uses only transitions that were actually experienced.
For every trajectory position and horizon, the model receives the real episode
history through the root and the subsequent action prefix.  Its target is whether
the observed endpoint was already present in the root history, was first visited
and then repeated inside the suffix, or was genuinely new.

No environment transition function is queried for counterfactual labels.  The
trained predictor is intended to be evaluated here before it is integrated into
MCTS and asked to generalize to hypothetical action prefixes.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
from torch.utils.data import DataLoader, Dataset

from zoo.minigrid.envs.branching_rooms_env import (
    BRANCHING_ROOMS_ENV_ID,
    BranchingRoomsEnv,
)


ROOT_HISTORY_REVISIT = 0
BRANCH_REVISIT = 1
NOVEL = 2
LABEL_NAMES = (
    "root_history_revisit",
    "within_suffix_revisit",
    "novel",
)
SPLIT_NAMES = ("train", "validation", "test", "heldout_policy")
POLICY_NAMES = ("uniform", "persistent", "macro")


@dataclass(frozen=True)
class DatasetBundle:
    observations: np.ndarray
    actions: np.ndarray
    poses: np.ndarray
    observation_targets: np.ndarray
    pose_targets: np.ndarray
    split_ids: np.ndarray
    policy_ids: np.ndarray
    episode_seeds: np.ndarray
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class OnlinePredictorContext:
    root_embedding: torch.Tensor
    history_states: torch.Tensor | None
    history_hidden: torch.Tensor | None


class EpisodeDataset(Dataset):
    def __init__(self, bundle: DatasetBundle, split_id: int) -> None:
        self.bundle = bundle
        self.indices = np.flatnonzero(bundle.split_ids == int(split_id))

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, index: int):
        episode = int(self.indices[index])
        return {
            "observations": torch.from_numpy(self.bundle.observations[episode]),
            "actions": torch.from_numpy(
                self.bundle.actions[episode].astype(np.int64, copy=False)
            ),
            "targets": torch.from_numpy(
                self.bundle.observation_targets[episode].astype(np.int64, copy=False)
            ),
            "pose_targets": torch.from_numpy(
                self.bundle.pose_targets[episode].astype(np.int64, copy=False)
            ),
            "policy_id": torch.tensor(
                int(self.bundle.policy_ids[episode]), dtype=torch.long
            ),
        }


class ObservationEncoder(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(64 * 2 * 2, int(embedding_dim)),
            nn.LayerNorm(int(embedding_dim)),
            nn.ReLU(inplace=True),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation.float().div(255.0))


class ActionNoveltyPredictor(nn.Module):
    """Predict endpoint novelty from real history and a short action suffix."""

    def __init__(
        self,
        action_space_size: int,
        embedding_dim: int,
        hidden_dim: int,
        use_history: bool,
    ) -> None:
        super().__init__()
        self.action_space_size = int(action_space_size)
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.use_history = bool(use_history)
        action_dim = max(8, self.hidden_dim // 8)

        self.observation_encoder = ObservationEncoder(self.embedding_dim)
        self.action_embedding = nn.Embedding(self.action_space_size + 1, action_dim)
        self.history_gru = nn.GRU(
            self.embedding_dim + action_dim,
            self.hidden_dim,
            batch_first=True,
        )
        root_input_dim = self.embedding_dim + (
            self.hidden_dim if self.use_history else 0
        )
        self.prefix_initial = nn.Sequential(
            nn.Linear(root_input_dim, self.hidden_dim),
            nn.Tanh(),
        )
        self.prefix_cell = nn.GRUCell(action_dim, self.hidden_dim)

        if self.use_history:
            self.memory_key = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
            self.memory_value = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
            classifier_input_dim = (
                2 * self.hidden_dim + self.embedding_dim
            )
        else:
            self.memory_key = None
            self.memory_value = None
            classifier_input_dim = self.hidden_dim + self.embedding_dim

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, len(LABEL_NAMES)),
        )

    def _encode_observations(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size, step_count = observations.shape[:2]
        embeddings = self.observation_encoder(
            observations.reshape(batch_size * step_count, *observations.shape[2:])
        )
        return embeddings.reshape(batch_size, step_count, self.embedding_dim)

    def _history_states(
        self,
        observation_embeddings: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, observation_count = observation_embeddings.shape[:2]
        start_action = torch.full(
            (batch_size, 1),
            self.action_space_size,
            dtype=torch.long,
            device=actions.device,
        )
        previous_actions = torch.cat([start_action, actions], dim=1)
        if previous_actions.shape[1] != observation_count:
            raise ValueError(
                "An episode must contain exactly one more observation than action."
            )
        history_inputs = torch.cat(
            [observation_embeddings, self.action_embedding(previous_actions)], dim=-1
        )
        states, _ = self.history_gru(history_inputs)
        return states

    def _attend_to_root_history(
        self,
        queries: torch.Tensor,
        history_states: torch.Tensor,
        root_indices: torch.Tensor,
    ) -> torch.Tensor:
        keys = self.memory_key(history_states)
        values = self.memory_value(history_states)
        scores = torch.einsum("bqh,bsh->bqs", queries, keys)
        scores = scores / math.sqrt(float(self.hidden_dim))
        memory_positions = torch.arange(
            history_states.shape[1], device=history_states.device
        )
        valid_memory = memory_positions.unsqueeze(0) <= root_indices.unsqueeze(1)
        scores = scores.masked_fill(~valid_memory.unsqueeze(0), -1e9)
        weights = torch.softmax(scores, dim=-1)
        return torch.einsum("bqs,bsh->bqh", weights, values)

    def start_online_context(
        self, observation: torch.Tensor
    ) -> OnlinePredictorContext:
        """Encode the first real observation of an episode."""
        if observation.ndim != 4 or observation.shape[0] != 1:
            raise ValueError("Online context expects one [1, C, H, W] observation.")
        embedding = self.observation_encoder(observation).reshape(1, self.embedding_dim)
        if not self.use_history:
            return OnlinePredictorContext(embedding, None, None)
        start_action = torch.full(
            (1,),
            self.action_space_size,
            dtype=torch.long,
            device=observation.device,
        )
        history_input = torch.cat(
            [embedding, self.action_embedding(start_action)], dim=-1
        ).unsqueeze(1)
        history_state, history_hidden = self.history_gru(history_input)
        return OnlinePredictorContext(
            root_embedding=embedding,
            history_states=history_state,
            history_hidden=history_hidden,
        )

    def advance_online_context(
        self,
        context: OnlinePredictorContext,
        action: torch.Tensor,
        observation: torch.Tensor,
    ) -> OnlinePredictorContext:
        """Advance history using one transition that actually occurred."""
        if observation.ndim != 4 or observation.shape[0] != 1:
            raise ValueError("Online context expects one [1, C, H, W] observation.")
        action = action.to(device=observation.device, dtype=torch.long).reshape(1)
        embedding = self.observation_encoder(observation).reshape(1, self.embedding_dim)
        if not self.use_history:
            return OnlinePredictorContext(embedding, None, None)
        if context.history_states is None or context.history_hidden is None:
            raise ValueError("A full-history model requires recurrent history state.")
        history_input = torch.cat(
            [embedding, self.action_embedding(action)], dim=-1
        ).unsqueeze(1)
        next_state, next_hidden = self.history_gru(
            history_input, context.history_hidden
        )
        return OnlinePredictorContext(
            root_embedding=embedding,
            history_states=torch.cat([context.history_states, next_state], dim=1),
            history_hidden=next_hidden,
        )

    def score_action_suffixes(
        self,
        context: OnlinePredictorContext,
        action_suffixes: torch.Tensor,
    ) -> torch.Tensor:
        """Score hypothetical suffixes without generating future observations."""
        if action_suffixes.ndim != 2:
            raise ValueError("Expected action suffixes shaped [count, horizon].")
        action_suffixes = action_suffixes.to(
            device=context.root_embedding.device, dtype=torch.long
        )
        suffix_count, horizon = action_suffixes.shape
        root_embeddings = context.root_embedding.expand(suffix_count, -1)
        if self.use_history:
            if context.history_states is None:
                raise ValueError("A full-history model requires history states.")
            root_history = context.history_states[:, -1].expand(suffix_count, -1)
            initial_input = torch.cat([root_embeddings, root_history], dim=-1)
            keys = self.memory_key(context.history_states).squeeze(0)
            values = self.memory_value(context.history_states).squeeze(0)
        else:
            initial_input = root_embeddings
            keys = None
            values = None

        prefix_state = self.prefix_initial(initial_input)
        logits = []
        for offset in range(horizon):
            prefix_state = self.prefix_cell(
                self.action_embedding(action_suffixes[:, offset]), prefix_state
            )
            classifier_inputs = [prefix_state]
            if self.use_history:
                scores = torch.matmul(prefix_state, keys.transpose(0, 1))
                scores = scores / math.sqrt(float(self.hidden_dim))
                weights = torch.softmax(scores, dim=-1)
                classifier_inputs.append(torch.matmul(weights, values))
            classifier_inputs.append(root_embeddings)
            logits.append(self.classifier(torch.cat(classifier_inputs, dim=-1)))
        return torch.stack(logits, dim=1)

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        horizon: int,
        root_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if observations.ndim != 5:
            raise ValueError(
                "Expected observations shaped [batch, time, channels, height, width]."
            )
        if actions.ndim != 2:
            raise ValueError("Expected actions shaped [batch, time].")
        horizon = int(horizon)
        root_count = actions.shape[1] - horizon + 1
        if root_count <= 0:
            raise ValueError("The horizon must not exceed the episode length.")
        if root_indices is None:
            root_indices = torch.arange(root_count, device=actions.device)
        else:
            root_indices = root_indices.to(device=actions.device, dtype=torch.long)
        if root_indices.ndim != 1 or bool(
            torch.any((root_indices < 0) | (root_indices >= root_count))
        ):
            raise ValueError("root_indices contains an invalid trajectory position.")

        observation_embeddings = self._encode_observations(observations)
        root_embeddings = observation_embeddings[:, root_indices]
        history_states = None
        if self.use_history:
            history_states = self._history_states(observation_embeddings, actions)
            initial_input = torch.cat(
                [root_embeddings, history_states[:, root_indices]], dim=-1
            )
        else:
            initial_input = root_embeddings

        prefix_state = self.prefix_initial(initial_input)
        logits = []
        for offset in range(horizon):
            prefix_actions = actions[:, root_indices + offset]
            batch_size, query_count = prefix_actions.shape
            prefix_state = self.prefix_cell(
                self.action_embedding(prefix_actions).reshape(
                    batch_size * query_count, -1
                ),
                prefix_state.reshape(batch_size * query_count, -1),
            ).reshape(batch_size, query_count, self.hidden_dim)
            classifier_inputs = [prefix_state]
            if self.use_history:
                classifier_inputs.append(
                    self._attend_to_root_history(
                        prefix_state, history_states, root_indices
                    )
                )
            classifier_inputs.append(root_embeddings)
            logits.append(self.classifier(torch.cat(classifier_inputs, dim=-1)))
        return torch.stack(logits, dim=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".local_runs/branching_rooms_action_novelty"),
    )
    parser.add_argument("--mode", choices=("all", "collect", "train"), default="all")
    parser.add_argument("--force-collect", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode-length", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--tile-size", type=int, default=4)
    parser.add_argument("--train-episodes", type=int, default=192)
    parser.add_argument("--validation-episodes", type=int, default=48)
    parser.add_argument("--test-episodes", type=int, default=48)
    parser.add_argument("--heldout-episodes", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--evaluation-batch-size", type=int, default=4)
    parser.add_argument("--roots-per-episode", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("full_history", "current_only"),
        default=("full_history", "current_only"),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use a tiny dataset and model for a quick implementation check.",
    )
    return parser


def _apply_smoke_overrides(args: argparse.Namespace) -> None:
    if not args.smoke:
        return
    args.episode_length = 16
    args.horizon = min(int(args.horizon), 5)
    args.tile_size = 2
    args.train_episodes = 8
    args.validation_episodes = 4
    args.test_episodes = 4
    args.heldout_episodes = 4
    args.epochs = 2
    args.patience = 2
    args.batch_size = 2
    args.evaluation_batch_size = 2
    args.roots_per_episode = 6
    args.embedding_dim = 16
    args.hidden_dim = 32


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device(name)


class BehaviourPolicy:
    def __init__(self, name: str, rng: np.random.Generator) -> None:
        if name not in POLICY_NAMES:
            raise ValueError(f"Unknown behaviour policy: {name}")
        self.name = name
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

        if not self.macro_actions:
            macros = (
                (2, 2, 2, 0, 2),
                (2, 2, 1, 2, 2),
                (0, 2, 1, 2),
                (1, 2, 0, 2),
                (2, 0, 0, 2, 2),
                (2, 1, 1, 2, 2),
            )
            self.macro_actions.extend(macros[int(self.rng.integers(0, len(macros)))])
        return int(self.macro_actions.popleft())


def _make_rgb_env(episode_length: int, tile_size: int):
    env = gym.make(
        BRANCHING_ROOMS_ENV_ID,
        max_steps=int(episode_length),
        agent_view_size=7,
    )
    env = RGBImgPartialObsWrapper(env, tile_size=int(tile_size))
    return ImgObsWrapper(env)


def collect_episode(
    episode_length: int,
    tile_size: int,
    episode_seed: int,
    policy_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(episode_seed) + 7919)
    policy = BehaviourPolicy(policy_name, rng)
    env = _make_rgb_env(episode_length, tile_size)
    observation, info = env.reset(seed=int(episode_seed))
    observations = [np.transpose(np.asarray(observation), (2, 0, 1)).astype(np.uint8)]
    poses = [
        (
            int(info["agent_position"][0]),
            int(info["agent_position"][1]),
            int(info["agent_direction"]),
        )
    ]
    actions = []
    try:
        for _ in range(int(episode_length)):
            action = policy.action()
            observation, _, terminated, truncated, info = env.step(action)
            actions.append(action)
            observations.append(
                np.transpose(np.asarray(observation), (2, 0, 1)).astype(np.uint8)
            )
            poses.append(
                (
                    int(info["agent_position"][0]),
                    int(info["agent_position"][1]),
                    int(info["agent_direction"]),
                )
            )
            if terminated or truncated:
                break
    finally:
        env.close()
    if len(actions) != int(episode_length):
        raise RuntimeError(
            f"Expected {episode_length} transitions, collected {len(actions)}."
        )
    return (
        np.stack(observations),
        np.asarray(actions, dtype=np.uint8),
        np.asarray(poses, dtype=np.int16),
    )


def state_ids_from_observations(observations: np.ndarray) -> np.ndarray:
    mapping: Dict[bytes, int] = {}
    ids = np.empty(observations.shape[0], dtype=np.int32)
    for index, observation in enumerate(observations):
        key = np.ascontiguousarray(observation).tobytes()
        if key not in mapping:
            mapping[key] = len(mapping)
        ids[index] = mapping[key]
    return ids


def state_ids_from_poses(poses: np.ndarray) -> np.ndarray:
    mapping: Dict[Tuple[int, ...], int] = {}
    ids = np.empty(poses.shape[0], dtype=np.int32)
    for index, pose in enumerate(poses):
        key = tuple(int(value) for value in pose)
        if key not in mapping:
            mapping[key] = len(mapping)
        ids[index] = mapping[key]
    return ids


def build_novelty_targets(state_ids: np.ndarray, horizon: int) -> np.ndarray:
    state_ids = np.asarray(state_ids, dtype=np.int64)
    transition_count = state_ids.size - 1
    root_count = transition_count - int(horizon) + 1
    if root_count <= 0:
        raise ValueError("The trajectory is shorter than the requested horizon.")
    first_visit: Dict[int, int] = {}
    for index, state_id in enumerate(state_ids.tolist()):
        first_visit.setdefault(int(state_id), index)

    targets = np.empty((root_count, int(horizon)), dtype=np.uint8)
    for root in range(root_count):
        branch_states = set()
        for offset in range(1, int(horizon) + 1):
            endpoint = int(state_ids[root + offset])
            if first_visit[endpoint] <= root:
                label = ROOT_HISTORY_REVISIT
            elif endpoint in branch_states:
                label = BRANCH_REVISIT
            else:
                label = NOVEL
            targets[root, offset - 1] = label
            branch_states.add(endpoint)
    return targets


def collect_dataset(args: argparse.Namespace, dataset_path: Path) -> DatasetBundle:
    split_episode_counts = (
        int(args.train_episodes),
        int(args.validation_episodes),
        int(args.test_episodes),
        int(args.heldout_episodes),
    )
    observations = []
    actions = []
    poses = []
    observation_targets = []
    pose_targets = []
    split_ids = []
    policy_ids = []
    episode_seeds = []

    episode_index = 0
    for split_id, episode_count in enumerate(split_episode_counts):
        policy_choices = ("macro",) if split_id == 3 else ("uniform", "persistent")
        for split_episode in range(episode_count):
            policy_name = policy_choices[split_episode % len(policy_choices)]
            episode_seed = int(args.seed) * 1_000_003 + episode_index * 97 + 17
            episode_observations, episode_actions, episode_poses = collect_episode(
                args.episode_length,
                args.tile_size,
                episode_seed,
                policy_name,
            )
            observations.append(episode_observations)
            actions.append(episode_actions)
            poses.append(episode_poses)
            observation_targets.append(
                build_novelty_targets(
                    state_ids_from_observations(episode_observations), args.horizon
                )
            )
            pose_targets.append(
                build_novelty_targets(
                    state_ids_from_poses(episode_poses), args.horizon
                )
            )
            split_ids.append(split_id)
            policy_ids.append(POLICY_NAMES.index(policy_name))
            episode_seeds.append(episode_seed)
            episode_index += 1
        print(
            f"collected {episode_count} {SPLIT_NAMES[split_id]} episodes "
            f"({episode_index} total)"
        )

    metadata = {
        "seed": int(args.seed),
        "episode_length": int(args.episode_length),
        "horizon": int(args.horizon),
        "tile_size": int(args.tile_size),
        "split_names": SPLIT_NAMES,
        "policy_names": POLICY_NAMES,
        "target_source": "exact_rgb_observation_recurrence",
        "counterfactual_transition_queries": 0,
    }
    bundle = DatasetBundle(
        observations=np.stack(observations),
        actions=np.stack(actions),
        poses=np.stack(poses),
        observation_targets=np.stack(observation_targets),
        pose_targets=np.stack(pose_targets),
        split_ids=np.asarray(split_ids, dtype=np.uint8),
        policy_ids=np.asarray(policy_ids, dtype=np.uint8),
        episode_seeds=np.asarray(episode_seeds, dtype=np.int64),
        metadata=metadata,
    )
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = dataset_path.with_suffix(dataset_path.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary_path,
        observations=bundle.observations,
        actions=bundle.actions,
        poses=bundle.poses,
        observation_targets=bundle.observation_targets,
        pose_targets=bundle.pose_targets,
        split_ids=bundle.split_ids,
        policy_ids=bundle.policy_ids,
        episode_seeds=bundle.episode_seeds,
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    temporary_path.replace(dataset_path)
    print(f"wrote dataset: {dataset_path}")
    return bundle


def load_dataset(dataset_path: Path) -> DatasetBundle:
    with np.load(dataset_path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        return DatasetBundle(
            observations=data["observations"],
            actions=data["actions"],
            poses=data["poses"],
            observation_targets=data["observation_targets"],
            pose_targets=data["pose_targets"],
            split_ids=data["split_ids"],
            policy_ids=data["policy_ids"],
            episode_seeds=data["episode_seeds"],
            metadata=metadata,
        )


def _class_weights(targets: np.ndarray) -> torch.Tensor:
    counts = np.bincount(targets.reshape(-1), minlength=len(LABEL_NAMES)).astype(
        np.float64
    )
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    weights = np.minimum(weights, 5.0)
    return torch.tensor(weights, dtype=torch.float32)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = labels.astype(np.int64)
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
    return (rank_sum - positive * (positive + 1) / 2.0) / (positive * negative)


def _expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, bin_count: int = 10
) -> float:
    error = 0.0
    edges = np.linspace(0.0, 1.0, bin_count + 1)
    for index in range(bin_count):
        lower, upper = edges[index], edges[index + 1]
        if index + 1 == bin_count:
            selected = (probabilities >= lower) & (probabilities <= upper)
        else:
            selected = (probabilities >= lower) & (probabilities < upper)
        if not np.any(selected):
            continue
        confidence = float(probabilities[selected].mean())
        frequency = float(labels[selected].mean())
        error += float(selected.mean()) * abs(confidence - frequency)
    return error


def classification_metrics(
    targets: np.ndarray, probabilities: np.ndarray
) -> Mapping[str, object]:
    predictions = probabilities.argmax(axis=-1)
    confusion = np.zeros((len(LABEL_NAMES), len(LABEL_NAMES)), dtype=np.int64)
    for target, prediction in zip(targets.reshape(-1), predictions.reshape(-1)):
        confusion[int(target), int(prediction)] += 1

    per_class = {}
    f1_values = []
    active_f1_values = []
    for class_index, class_name in enumerate(LABEL_NAMES):
        true_positive = int(confusion[class_index, class_index])
        false_positive = int(confusion[:, class_index].sum() - true_positive)
        false_negative = int(confusion[class_index, :].sum() - true_positive)
        precision = _safe_ratio(true_positive, true_positive + false_positive)
        recall = _safe_ratio(true_positive, true_positive + false_negative)
        f1 = _safe_ratio(2 * precision * recall, precision + recall)
        f1_values.append(f1)
        support = int(confusion[class_index, :].sum())
        if support > 0:
            active_f1_values.append(f1)
        per_class[class_name] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    revisit_target = (targets != NOVEL).astype(np.int64)
    revisit_probability = 1.0 - probabilities[..., NOVEL]
    revisit_prediction = (revisit_probability >= 0.5).astype(np.int64)
    true_positive = int(np.sum((revisit_target == 1) & (revisit_prediction == 1)))
    false_positive = int(np.sum((revisit_target == 0) & (revisit_prediction == 1)))
    false_negative = int(np.sum((revisit_target == 1) & (revisit_prediction == 0)))
    precision = _safe_ratio(true_positive, true_positive + false_positive)
    recall = _safe_ratio(true_positive, true_positive + false_negative)
    return {
        "count": int(targets.size),
        "accuracy": float(np.mean(predictions == targets)),
        "macro_f1": float(np.mean(active_f1_values)),
        "macro_f1_all_labels": float(np.mean(f1_values)),
        "confusion": confusion.tolist(),
        "per_class": per_class,
        "binary_revisit": {
            "prevalence": float(revisit_target.mean()),
            "precision": precision,
            "recall": recall,
            "f1": _safe_ratio(2 * precision * recall, precision + recall),
            "auroc": _binary_auc(
                revisit_target.reshape(-1), revisit_probability.reshape(-1)
            ),
            "brier": float(
                np.mean((revisit_probability - revisit_target.astype(np.float64)) ** 2)
            ),
            "ece": _expected_calibration_error(
                revisit_target.reshape(-1), revisit_probability.reshape(-1)
            ),
        },
    }


def evaluate_model(
    model: ActionNoveltyPredictor,
    loader: DataLoader,
    device: torch.device,
    horizon: int,
) -> Tuple[float, Mapping[str, object]]:
    model.eval()
    all_targets = []
    all_probabilities = []
    loss_sum = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            observations = batch["observations"].to(device)
            actions = batch["actions"].to(device)
            targets = batch["targets"].to(device)
            logits = model(observations, actions, horizon)
            loss = F.cross_entropy(
                logits.reshape(-1, len(LABEL_NAMES)), targets.reshape(-1)
            )
            batch_count = int(targets.numel())
            loss_sum += float(loss.item()) * batch_count
            count += batch_count
            all_targets.append(targets.cpu().numpy())
            all_probabilities.append(torch.softmax(logits, dim=-1).cpu().numpy())
    targets = np.concatenate(all_targets, axis=0)
    probabilities = np.concatenate(all_probabilities, axis=0)
    result: MutableMapping[str, object] = {
        "overall": classification_metrics(targets, probabilities),
        "by_horizon": {},
    }
    for depth in range(horizon):
        result["by_horizon"][str(depth + 1)] = classification_metrics(
            targets[:, :, depth], probabilities[:, :, depth]
        )
    return loss_sum / max(count, 1), result


def train_model(
    args: argparse.Namespace,
    bundle: DatasetBundle,
    model_name: str,
    device: torch.device,
) -> Mapping[str, object]:
    use_history = model_name == "full_history"
    model = ActionNoveltyPredictor(
        action_space_size=3,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        use_history=use_history,
    ).to(device)
    train_dataset = EpisodeDataset(bundle, 0)
    validation_dataset = EpisodeDataset(bundle, 1)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.evaluation_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    class_weights = _class_weights(
        bundle.observation_targets[bundle.split_ids == 0]
    ).to(device)
    print(f"{model_name} class weights: {class_weights.tolist()}")

    model_dir = args.output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / "best.pt"
    history_rows = []
    best_macro_f1 = -1.0
    best_epoch = -1
    stale_epochs = 0
    root_count = bundle.actions.shape[1] - int(args.horizon) + 1

    for epoch in range(int(args.epochs)):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch in train_loader:
            observations = batch["observations"].to(device)
            actions = batch["actions"].to(device)
            targets = batch["targets"].to(device)
            sampled_root_count = min(int(args.roots_per_episode), root_count)
            root_indices = torch.randperm(root_count, device=device)[
                :sampled_root_count
            ].sort().values
            selected_targets = targets[:, root_indices]
            logits = model(
                observations,
                actions,
                args.horizon,
                root_indices=root_indices,
            )
            loss = F.cross_entropy(
                logits.reshape(-1, len(LABEL_NAMES)),
                selected_targets.reshape(-1),
                weight=class_weights,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            sample_count = int(selected_targets.numel())
            train_loss_sum += float(loss.item()) * sample_count
            train_count += sample_count

        validation_loss, validation_metrics = evaluate_model(
            model, validation_loader, device, args.horizon
        )
        validation_macro_f1 = float(
            validation_metrics["overall"]["macro_f1"]
        )
        train_loss = train_loss_sum / max(train_count, 1)
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_macro_f1": validation_macro_f1,
            }
        )
        print(
            f"{model_name} epoch {epoch + 1:02d}/{args.epochs}: "
            f"train_loss={train_loss:.4f} val_loss={validation_loss:.4f} "
            f"val_macro_f1={validation_macro_f1:.4f}"
        )
        if validation_macro_f1 > best_macro_f1 + 1e-6:
            best_macro_f1 = validation_macro_f1
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_name": model_name,
                    "embedding_dim": int(args.embedding_dim),
                    "hidden_dim": int(args.hidden_dim),
                    "horizon": int(args.horizon),
                    "epoch": epoch,
                    "validation_macro_f1": validation_macro_f1,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= int(args.patience):
                print(f"{model_name}: early stopping after epoch {epoch + 1}")
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    results: MutableMapping[str, object] = {
        "model": model_name,
        "uses_real_episode_history": use_history,
        "best_epoch": int(best_epoch),
        "best_validation_macro_f1": float(best_macro_f1),
        "training_history": history_rows,
        "splits": {},
    }
    for split_id, split_name in enumerate(SPLIT_NAMES):
        loader = DataLoader(
            EpisodeDataset(bundle, split_id),
            batch_size=args.evaluation_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        loss, metrics = evaluate_model(model, loader, device, args.horizon)
        results["splits"][split_name] = {"loss": loss, **metrics}
    (model_dir / "metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    with (model_dir / "training.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=history_rows[0].keys())
        writer.writeheader()
        writer.writerows(history_rows)
    return results


def target_diagnostics(bundle: DatasetBundle) -> Mapping[str, object]:
    diagnostics: MutableMapping[str, object] = {
        "rgb_vs_pose_disagreement": {},
        "splits": {},
    }
    for split_id, split_name in enumerate(SPLIT_NAMES):
        selected = bundle.split_ids == split_id
        observation_targets = bundle.observation_targets[selected]
        pose_targets = bundle.pose_targets[selected]
        split_result: MutableMapping[str, object] = {"by_horizon": {}}
        for depth in range(observation_targets.shape[2]):
            labels = observation_targets[:, :, depth]
            counts = np.bincount(labels.reshape(-1), minlength=len(LABEL_NAMES))
            split_result["by_horizon"][str(depth + 1)] = {
                LABEL_NAMES[index]: int(counts[index])
                for index in range(len(LABEL_NAMES))
            }
        diagnostics["splits"][split_name] = split_result
        diagnostics["rgb_vs_pose_disagreement"][split_name] = float(
            np.mean(observation_targets != pose_targets)
        )
    return diagnostics


def _observation_digest(observation: np.ndarray) -> bytes:
    return hashlib.blake2b(
        np.ascontiguousarray(observation).tobytes(), digest_size=8
    ).digest()


def current_only_ambiguity(bundle: DatasetBundle) -> Mapping[str, object]:
    """Empirical ambiguity when history is removed from otherwise equal inputs."""
    result: MutableMapping[str, object] = {}
    horizon = bundle.observation_targets.shape[2]
    for split_id, split_name in enumerate(SPLIT_NAMES):
        episode_indices = np.flatnonzero(bundle.split_ids == split_id)
        split_depths: MutableMapping[str, object] = {}
        for depth in range(1, horizon + 1):
            groups: Dict[Tuple[bytes, bytes], np.ndarray] = {}
            mutable_groups: Dict[Tuple[bytes, bytes], np.ndarray] = defaultdict(
                lambda: np.zeros(len(LABEL_NAMES), dtype=np.int64)
            )
            for episode in episode_indices:
                root_count = bundle.observation_targets.shape[1]
                for root in range(root_count):
                    key = (
                        _observation_digest(bundle.observations[episode, root]),
                        bundle.actions[
                            episode, root : root + depth
                        ].tobytes(),
                    )
                    label = int(bundle.observation_targets[episode, root, depth - 1])
                    mutable_groups[key][label] += 1
            groups = dict(mutable_groups)
            total = sum(int(counts.sum()) for counts in groups.values())
            ambiguous_samples = sum(
                int(counts.sum()) for counts in groups.values() if np.count_nonzero(counts) > 1
            )
            majority_correct = sum(int(counts.max()) for counts in groups.values())
            split_depths[str(depth)] = {
                "sample_count": total,
                "unique_current_input_count": len(groups),
                "ambiguous_sample_fraction": _safe_ratio(ambiguous_samples, total),
                "empirical_current_only_accuracy_ceiling": _safe_ratio(
                    majority_correct, total
                ),
            }
        result[split_name] = split_depths
    return result


def _print_result_table(results: Mapping[str, Mapping[str, object]]) -> None:
    print("\nHeld-out test results")
    print(
        "model           depth  macro_f1  revisit_auc  revisit_recall  "
        "new_recall"
    )
    for model_name, model_result in results.items():
        by_horizon = model_result["splits"]["test"]["by_horizon"]
        for depth, metrics in by_horizon.items():
            binary = metrics["binary_revisit"]
            new_metrics = metrics["per_class"]["novel"]
            auc = binary["auroc"]
            auc_text = "n/a" if auc is None else f"{auc:.4f}"
            print(
                f"{model_name:15s} {int(depth):5d}  "
                f"{metrics['macro_f1']:.4f}     {auc_text:>10s}  "
                f"{binary['recall']:.4f}          {new_metrics['recall']:.4f}"
            )
    print("\nHeld-out behaviour-policy results")
    for model_name, model_result in results.items():
        metrics = model_result["splits"]["heldout_policy"]["overall"]
        print(
            f"{model_name}: macro_f1={metrics['macro_f1']:.4f}, "
            f"revisit_auc={metrics['binary_revisit']['auroc']}, "
            f"revisit_recall={metrics['binary_revisit']['recall']:.4f}"
        )


def main() -> None:
    args = build_parser().parse_args()
    _apply_smoke_overrides(args)
    if args.horizon <= 0 or args.horizon > args.episode_length:
        raise ValueError("horizon must be in [1, episode_length].")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = args.output_dir / "experienced_trajectories.npz"
    arguments = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    (args.output_dir / "arguments.json").write_text(
        json.dumps(arguments, indent=2), encoding="utf-8"
    )

    if args.mode in {"all", "collect"} and (
        args.force_collect or not dataset_path.exists()
    ):
        bundle = collect_dataset(args, dataset_path)
    else:
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset does not exist: {dataset_path}. Run with --mode collect first."
            )
        bundle = load_dataset(dataset_path)
        expected = {
            "episode_length": int(args.episode_length),
            "horizon": int(args.horizon),
            "tile_size": int(args.tile_size),
        }
        for key, value in expected.items():
            if int(bundle.metadata[key]) != value:
                raise ValueError(
                    f"Cached dataset {key}={bundle.metadata[key]} does not match {value}. "
                    "Use a new output directory or --force-collect."
                )

    diagnostics = {
        "target": target_diagnostics(bundle),
        "current_only_input_ambiguity": current_only_ambiguity(bundle),
    }
    (args.output_dir / "dataset_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    print(
        f"dataset shape: observations={bundle.observations.shape}, "
        f"actions={bundle.actions.shape}, targets={bundle.observation_targets.shape}"
    )
    print(
        "RGB/pose target disagreement:",
        diagnostics["target"]["rgb_vs_pose_disagreement"],
    )
    if args.mode == "collect":
        return

    device = _resolve_device(args.device)
    print(f"training device: {device}")
    all_results = {}
    for model_index, model_name in enumerate(args.models):
        torch.manual_seed(args.seed + model_index * 10_007)
        all_results[model_name] = train_model(
            args, bundle, model_name, device
        )
    (args.output_dir / "comparison.json").write_text(
        json.dumps(all_results, indent=2), encoding="utf-8"
    )
    _print_result_table(all_results)


if __name__ == "__main__":
    main()
