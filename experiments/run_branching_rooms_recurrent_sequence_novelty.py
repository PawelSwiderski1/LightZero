"""Recurrent-history action-sequence novelty planning in Branching Rooms.

The model compresses the real episode into one LSTM state.  Each recurrent
update receives the current observation, previous action, previous intrinsic
reward, and previous extrinsic reward.  A separate suffix GRU scores candidate
action sequences from that state and predicts the episodic novelty reward at
every depth.

An exact first-visit RGB set constructs intrinsic rewards and training labels,
but that set is never exposed to the model.  Training uses only suffixes that
actually occurred.  Counterfactual one-step transitions are reconstructed only
after collection for diagnostics; they never enter the optimizer.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if __package__:
    from .diagnose_branching_rooms_action_novelty_q_decisions import (
        correction_statistics,
        exact_one_step_novelty,
        planner_statistics,
    )
    from .run_branching_rooms_action_novelty import ObservationEncoder
    from .run_branching_rooms_action_novelty_q import novelty_rewards, resolve_device
    from .run_branching_rooms_online_action_novelty import (
        Trajectory,
        coverage_metrics,
        make_env,
        observation_tensor,
        trajectory_arrays,
    )
else:
    from diagnose_branching_rooms_action_novelty_q_decisions import (
        correction_statistics,
        exact_one_step_novelty,
        planner_statistics,
    )
    from run_branching_rooms_action_novelty import ObservationEncoder
    from run_branching_rooms_action_novelty_q import novelty_rewards, resolve_device
    from run_branching_rooms_online_action_novelty import (
        Trajectory,
        coverage_metrics,
        make_env,
        observation_tensor,
        trajectory_arrays,
    )


@dataclass(frozen=True)
class RecurrentSequenceContext:
    """The complete model-visible summary of the real episode history."""

    hidden: torch.Tensor
    cell: torch.Tensor


class RecurrentSequenceNovelty(nn.Module):
    """Predict finite-horizon episodic novelty from one recurrent state."""

    def __init__(
        self,
        action_space_size: int,
        embedding_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.action_space_size = int(action_space_size)
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        action_dim = max(8, self.hidden_dim // 8)

        self.observation_encoder = ObservationEncoder(self.embedding_dim)
        self.action_embedding = nn.Embedding(self.action_space_size + 1, action_dim)
        self.history_cell = nn.LSTMCell(
            self.embedding_dim + action_dim + 2,
            self.hidden_dim,
        )
        self.prefix_initial = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
        )
        self.prefix_cell = nn.GRUCell(action_dim, self.hidden_dim)
        self.novelty_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, 1),
        )

    def _history_step(
        self,
        observation_embedding: torch.Tensor,
        previous_action: torch.Tensor,
        previous_intrinsic_reward: torch.Tensor,
        previous_extrinsic_reward: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
    ) -> RecurrentSequenceContext:
        recurrent_input = torch.cat(
            [
                observation_embedding,
                self.action_embedding(previous_action),
                previous_intrinsic_reward.reshape(-1, 1),
                previous_extrinsic_reward.reshape(-1, 1),
            ],
            dim=-1,
        )
        hidden, cell = self.history_cell(recurrent_input, (hidden, cell))
        return RecurrentSequenceContext(hidden=hidden, cell=cell)

    def start_online_context(self, observation: torch.Tensor) -> RecurrentSequenceContext:
        if observation.ndim != 4:
            raise ValueError("Expected observations shaped [batch, channels, height, width].")
        batch_size = observation.shape[0]
        hidden = torch.zeros(
            batch_size,
            self.hidden_dim,
            device=observation.device,
            dtype=torch.float32,
        )
        cell = torch.zeros_like(hidden)
        start_action = torch.full(
            (batch_size,),
            self.action_space_size,
            device=observation.device,
            dtype=torch.long,
        )
        zero_reward = torch.zeros(batch_size, device=observation.device)
        return self._history_step(
            self.observation_encoder(observation),
            start_action,
            zero_reward,
            zero_reward,
            hidden,
            cell,
        )

    def advance_online_context(
        self,
        context: RecurrentSequenceContext,
        action: torch.Tensor,
        observation: torch.Tensor,
        intrinsic_reward: torch.Tensor,
        extrinsic_reward: torch.Tensor | None = None,
    ) -> RecurrentSequenceContext:
        action = action.to(device=observation.device, dtype=torch.long).reshape(-1)
        intrinsic_reward = intrinsic_reward.to(
            device=observation.device, dtype=torch.float32
        ).reshape(-1)
        if extrinsic_reward is None:
            extrinsic_reward = torch.zeros_like(intrinsic_reward)
        else:
            extrinsic_reward = extrinsic_reward.to(
                device=observation.device, dtype=torch.float32
            ).reshape(-1)
        return self._history_step(
            self.observation_encoder(observation),
            action,
            intrinsic_reward,
            extrinsic_reward,
            context.hidden,
            context.cell,
        )

    def history_states(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        intrinsic_rewards: torch.Tensor,
        extrinsic_rewards: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Unroll replay and return the one recurrent state at every real root."""
        if observations.ndim != 5 or actions.ndim != 2:
            raise ValueError("Expected batched observation and action sequences.")
        if intrinsic_rewards.shape != actions.shape:
            raise ValueError("Intrinsic rewards must have the same shape as actions.")
        if observations.shape[1] != actions.shape[1] + 1:
            raise ValueError("Expected exactly one more observation than action.")
        if extrinsic_rewards is None:
            extrinsic_rewards = torch.zeros_like(intrinsic_rewards)
        elif extrinsic_rewards.shape != actions.shape:
            raise ValueError("Extrinsic rewards must have the same shape as actions.")

        batch_size, observation_count = observations.shape[:2]
        embeddings = self.observation_encoder(
            observations.reshape(
                batch_size * observation_count, *observations.shape[2:]
            )
        ).reshape(batch_size, observation_count, self.embedding_dim)
        hidden = torch.zeros(
            batch_size,
            self.hidden_dim,
            device=observations.device,
            dtype=torch.float32,
        )
        cell = torch.zeros_like(hidden)
        zero_reward = torch.zeros(batch_size, device=observations.device)
        start_action = torch.full(
            (batch_size,),
            self.action_space_size,
            device=actions.device,
            dtype=torch.long,
        )
        states = []
        for index in range(observation_count):
            if index == 0:
                previous_action = start_action
                previous_intrinsic_reward = zero_reward
                previous_extrinsic_reward = zero_reward
            else:
                previous_action = actions[:, index - 1]
                previous_intrinsic_reward = intrinsic_rewards[:, index - 1]
                previous_extrinsic_reward = extrinsic_rewards[:, index - 1]
            context = self._history_step(
                embeddings[:, index],
                previous_action,
                previous_intrinsic_reward,
                previous_extrinsic_reward,
                hidden,
                cell,
            )
            hidden, cell = context.hidden, context.cell
            states.append(hidden)
        return torch.stack(states, dim=1)

    def _score_from_hidden(
        self,
        root_hidden: torch.Tensor,
        action_suffixes: torch.Tensor,
    ) -> torch.Tensor:
        if root_hidden.ndim != 2 or action_suffixes.ndim != 2:
            raise ValueError("Expected [count, hidden] roots and [count, horizon] suffixes.")
        if root_hidden.shape[0] != action_suffixes.shape[0]:
            raise ValueError("Every action suffix must have one root state.")
        prefix_state = self.prefix_initial(root_hidden)
        logits = []
        for depth in range(action_suffixes.shape[1]):
            prefix_state = self.prefix_cell(
                self.action_embedding(action_suffixes[:, depth]), prefix_state
            )
            logits.append(self.novelty_head(prefix_state).squeeze(-1))
        return torch.stack(logits, dim=1)

    def score_action_suffixes(
        self,
        context: RecurrentSequenceContext,
        action_suffixes: torch.Tensor,
    ) -> torch.Tensor:
        """Score hypothetical suffixes without observations or a memory bank."""
        if action_suffixes.ndim != 2:
            raise ValueError("Expected action suffixes shaped [count, horizon].")
        action_suffixes = action_suffixes.to(
            device=context.hidden.device, dtype=torch.long
        )
        if context.hidden.shape[0] != 1:
            raise ValueError("Online suffix scoring expects one real root context.")
        return self._score_from_hidden(
            context.hidden.expand(action_suffixes.shape[0], -1),
            action_suffixes,
        )

    def forward_actual_suffixes(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        intrinsic_rewards: torch.Tensor,
        horizon: int,
        root_indices: torch.Tensor,
        extrinsic_rewards: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score replay suffixes that actually followed sampled real roots."""
        horizon = int(horizon)
        root_indices = root_indices.to(device=actions.device, dtype=torch.long)
        max_root = actions.shape[1] - horizon
        if root_indices.ndim != 1 or bool(
            torch.any((root_indices < 0) | (root_indices > max_root))
        ):
            raise ValueError("root_indices contains an invalid trajectory position.")
        states = self.history_states(
            observations,
            actions,
            intrinsic_rewards,
            extrinsic_rewards=extrinsic_rewards,
        )
        root_hidden = states[:, root_indices]
        suffix_actions = torch.stack(
            [actions[:, root_indices + depth] for depth in range(horizon)], dim=2
        )
        batch_size, root_count = root_hidden.shape[:2]
        logits = self._score_from_hidden(
            root_hidden.reshape(batch_size * root_count, self.hidden_dim),
            suffix_actions.reshape(batch_size * root_count, horizon),
        )
        return logits.reshape(batch_size, root_count, horizon)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".local_runs/branching_rooms_recurrent_sequence_novelty_seed0"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--episode-length", type=int, default=128)
    parser.add_argument("--tile-size", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--updates-per-episode", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--roots-per-update", type=int, default=32)
    parser.add_argument("--replay-capacity", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--sequence-loss-weight", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.10)
    parser.add_argument("--epsilon-decay-episodes", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--smoke", action="store_true")
    return parser


def enumerate_action_suffixes(action_count: int, horizon: int) -> np.ndarray:
    return np.asarray(
        list(itertools.product(range(int(action_count)), repeat=int(horizon))),
        dtype=np.int64,
    )


def discounted_sequence_values(
    novelty_probabilities: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    weights = torch.pow(
        novelty_probabilities.new_tensor(float(gamma)),
        torch.arange(
            novelty_probabilities.shape[-1],
            device=novelty_probabilities.device,
            dtype=novelty_probabilities.dtype,
        ),
    )
    return torch.sum(novelty_probabilities * weights, dim=-1)


def action_values_from_suffixes(
    probabilities: np.ndarray,
    suffixes: np.ndarray,
    gamma: float,
    top_k: int,
) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    weights = np.power(float(gamma), np.arange(probabilities.shape[1]))
    sequence_values = np.sum(probabilities * weights[None, :], axis=1)
    action_values = []
    for action in range(int(suffixes[:, 0].max()) + 1):
        candidates = sequence_values[suffixes[:, 0] == action]
        count = min(max(int(top_k), 1), int(candidates.size))
        action_values.append(
            float(np.partition(candidates, -count)[-count:].mean())
        )
    return np.asarray(action_values, dtype=np.float32)


def immediate_action_probabilities(
    probabilities: np.ndarray,
    suffixes: np.ndarray,
) -> np.ndarray:
    """Depth-one predictions are identical for suffixes with the same first action."""
    return np.asarray(
        [probabilities[np.flatnonzero(suffixes[:, 0] == action)[0], 0] for action in range(3)],
        dtype=np.float32,
    )


def epsilon_for_episode(args: argparse.Namespace, episode: int) -> float:
    progress = min(float(episode) / max(float(args.epsilon_decay_episodes), 1.0), 1.0)
    return float(args.epsilon_start) + progress * (
        float(args.epsilon_end) - float(args.epsilon_start)
    )


def collect_episode(
    model: RecurrentSequenceNovelty,
    suffixes: np.ndarray,
    suffix_tensor: torch.Tensor,
    args: argparse.Namespace,
    episode: int,
    epsilon: float,
    rng: np.random.Generator,
    device: torch.device,
) -> Tuple[Trajectory, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    env = make_env(args.episode_length, args.tile_size)
    observation, info = env.reset(
        seed=int(args.seed) * 1_000_003 + episode * 101 + 31
    )
    observations = [np.asarray(observation).copy()]
    actions: List[int] = []
    poses = [
        (
            int(info["agent_position"][0]),
            int(info["agent_position"][1]),
            int(info["agent_direction"]),
        )
    ]
    rooms = [info["room_id"]]
    sequence_action_values = []
    immediate_probabilities = []
    exploratory_flags = []
    intrinsic_rewards: List[float] = []
    visited_observations = {np.ascontiguousarray(observation).tobytes()}
    model.eval()
    with torch.no_grad():
        context = model.start_online_context(observation_tensor(observation, device))
    try:
        for _ in range(int(args.episode_length)):
            with torch.no_grad():
                probabilities = torch.sigmoid(
                    model.score_action_suffixes(context, suffix_tensor)
                ).cpu().numpy()
            action_values = action_values_from_suffixes(
                probabilities,
                suffixes,
                args.gamma,
                args.top_k,
            )
            immediate = immediate_action_probabilities(probabilities, suffixes)
            exploratory = bool(rng.random() < float(epsilon))
            if exploratory:
                action = int(rng.integers(0, 3))
            else:
                candidates = np.flatnonzero(
                    np.isclose(
                        action_values,
                        action_values.max(),
                        rtol=1e-7,
                        atol=1e-9,
                    )
                )
                action = int(rng.choice(candidates))

            observation, extrinsic_reward, terminated, truncated, info = env.step(action)
            observation_key = np.ascontiguousarray(observation).tobytes()
            intrinsic_reward = float(observation_key not in visited_observations)
            visited_observations.add(observation_key)
            sequence_action_values.append(action_values)
            immediate_probabilities.append(immediate)
            exploratory_flags.append(exploratory)
            actions.append(action)
            intrinsic_rewards.append(intrinsic_reward)
            observations.append(np.asarray(observation).copy())
            poses.append(
                (
                    int(info["agent_position"][0]),
                    int(info["agent_position"][1]),
                    int(info["agent_direction"]),
                )
            )
            rooms.append(info["room_id"])
            with torch.no_grad():
                context = model.advance_online_context(
                    context,
                    torch.tensor([action], device=device),
                    observation_tensor(observation, device),
                    torch.tensor([intrinsic_reward], device=device),
                    torch.tensor([extrinsic_reward], device=device),
                )
            if terminated or truncated:
                break
    finally:
        env.close()

    trajectory = trajectory_arrays(observations, actions, poses, rooms)
    rewards = np.asarray(intrinsic_rewards, dtype=np.float32)
    if not np.array_equal(rewards, novelty_rewards(trajectory.observations)):
        raise RuntimeError("Online and replay novelty rewards disagree.")
    return (
        trajectory,
        np.asarray(sequence_action_values, dtype=np.float32),
        np.asarray(immediate_probabilities, dtype=np.float32),
        np.asarray(exploratory_flags, dtype=np.bool_),
        rewards,
    )


def replay_arrays(
    replay: Sequence[Trajectory],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    observations = np.stack([trajectory.observations for trajectory in replay])
    actions = np.stack([trajectory.actions for trajectory in replay])
    rewards = np.stack(
        [novelty_rewards(trajectory.observations) for trajectory in replay]
    )
    return observations, actions, rewards


def sequence_prediction_loss(
    model: RecurrentSequenceNovelty,
    observations: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    root_indices: torch.Tensor,
    horizon: int,
    gamma: float,
    sequence_loss_weight: float,
) -> Tuple[torch.Tensor, Mapping[str, float]]:
    logits = model.forward_actual_suffixes(
        observations,
        actions,
        rewards,
        horizon,
        root_indices,
    )
    targets = torch.stack(
        [rewards[:, root_indices + depth] for depth in range(int(horizon))], dim=2
    )
    edge_loss = F.binary_cross_entropy_with_logits(logits, targets)
    probabilities = torch.sigmoid(logits)
    weights = torch.pow(
        probabilities.new_tensor(float(gamma)),
        torch.arange(int(horizon), device=probabilities.device, dtype=probabilities.dtype),
    )
    weight_sum = weights.sum().clamp_min(1e-8)
    predicted_return = torch.sum(probabilities * weights, dim=-1) / weight_sum
    target_return = torch.sum(targets * weights, dim=-1) / weight_sum
    return_loss = F.smooth_l1_loss(predicted_return, target_return)
    loss = edge_loss + float(sequence_loss_weight) * return_loss
    return loss, {
        "edge_bce": float(edge_loss.detach().item()),
        "sequence_loss": float(return_loss.detach().item()),
        "edge_brier": float(torch.mean((probabilities.detach() - targets) ** 2).item()),
        "predicted_novelty": float(probabilities.detach().mean().item()),
        "target_novelty": float(targets.mean().item()),
    }


def train_from_replay(
    model: RecurrentSequenceNovelty,
    optimizer: torch.optim.Optimizer,
    replay: Sequence[Trajectory],
    args: argparse.Namespace,
    rng: np.random.Generator,
    device: torch.device,
) -> Mapping[str, float]:
    observations, actions, rewards = replay_arrays(replay)
    metrics: Dict[str, List[float]] = {
        "train_loss": [],
        "train_edge_bce": [],
        "train_sequence_loss": [],
        "train_edge_brier": [],
    }
    root_count = actions.shape[1] - int(args.horizon) + 1
    for _ in range(int(args.updates_per_episode)):
        episode_indices = rng.integers(
            0, len(replay), size=int(args.batch_size)
        )
        sample_count = min(int(args.roots_per_update), root_count)
        roots = np.sort(
            rng.choice(root_count, size=sample_count, replace=False)
        ).astype(np.int64)
        obs = torch.from_numpy(observations[episode_indices]).to(device)
        act = torch.from_numpy(
            actions[episode_indices].astype(np.int64, copy=False)
        ).to(device)
        rew = torch.from_numpy(rewards[episode_indices]).to(device)
        root_tensor = torch.from_numpy(roots).to(device)
        model.train()
        loss, diagnostics = sequence_prediction_loss(
            model,
            obs,
            act,
            rew,
            root_tensor,
            args.horizon,
            args.gamma,
            args.sequence_loss_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        optimizer.step()
        metrics["train_loss"].append(float(loss.item()))
        metrics["train_edge_bce"].append(diagnostics["edge_bce"])
        metrics["train_sequence_loss"].append(diagnostics["sequence_loss"])
        metrics["train_edge_brier"].append(diagnostics["edge_brier"])
    return {key: float(np.mean(values)) for key, values in metrics.items()}


def score_trajectory_roots(
    model: RecurrentSequenceNovelty,
    trajectory: Trajectory,
    suffixes: np.ndarray,
    suffix_tensor: torch.Tensor,
    gamma: float,
    top_k: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    rewards = novelty_rewards(trajectory.observations)
    observations = torch.from_numpy(trajectory.observations).to(device)
    model.eval()
    with torch.no_grad():
        context = model.start_online_context(observations[0:1])
        sequence_values = []
        immediate_values = []
        for root in range(trajectory.actions.size):
            probabilities = torch.sigmoid(
                model.score_action_suffixes(context, suffix_tensor)
            ).cpu().numpy()
            sequence_values.append(
                action_values_from_suffixes(probabilities, suffixes, gamma, top_k)
            )
            immediate_values.append(
                immediate_action_probabilities(probabilities, suffixes)
            )
            context = model.advance_online_context(
                context,
                torch.tensor([int(trajectory.actions[root])], device=device),
                observations[root + 1 : root + 2],
                torch.tensor([float(rewards[root])], device=device),
            )
    return (
        np.asarray(sequence_values, dtype=np.float32),
        np.asarray(immediate_values, dtype=np.float32),
    )


def selected_prediction_metrics(
    immediate_probabilities: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
) -> Mapping[str, float]:
    selected = immediate_probabilities[np.arange(actions.size), actions]
    return {
        "selected_immediate_probability_mean": float(selected.mean()),
        "selected_immediate_target_mean": float(rewards.mean()),
        "selected_immediate_brier": float(np.mean((selected - rewards) ** 2)),
    }


def prefix_fields(prefix: str, values: Mapping[str, object]) -> Mapping[str, object]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def finite_mean(values: Sequence[object]) -> float | None:
    numbers = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return float(np.mean(numbers)) if numbers else None


def write_outputs(
    output_dir: Path,
    rows: Sequence[Mapping[str, object]],
    details: Mapping[str, object],
) -> None:
    with (output_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "episode_details.json").write_text(
        json.dumps(details, indent=2), encoding="utf-8"
    )
    final_count = min(8, len(rows))
    selected = rows[-final_count:]
    fields = (
        "unique_positions",
        "unique_poses",
        "rooms",
        "dead_ends",
        "observation_revisit_fraction",
        "position_revisit_fraction",
        "rgb_greedy_correct_when_avoidable_rate",
        "position_greedy_correct_when_avoidable_rate",
        "selected_immediate_brier_before",
        "selected_immediate_brier_after",
        "rescored_corrected_rate",
        "rescored_regressed_rate",
    )
    summary = {
        "episode_count": len(rows),
        "final_window_episode_count": final_count,
        "training_uses_counterfactual_transitions": False,
        "model_receives_explicit_episodic_memory": False,
        "history_inputs": [
            "current_observation",
            "previous_action",
            "previous_intrinsic_reward",
            "previous_extrinsic_reward",
        ],
        "final_window": {
            field: finite_mean([row[field] for row in selected]) for field in fields
        },
        "cumulative_unique_positions": int(rows[-1]["cumulative_unique_positions"]),
        "cumulative_unique_poses": int(rows[-1]["cumulative_unique_poses"]),
        "cumulative_rooms": int(rows[-1]["cumulative_rooms"]),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def format_optional(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def apply_smoke(args: argparse.Namespace) -> None:
    if not args.smoke:
        return
    args.episodes = 3
    args.episode_length = 24
    args.horizon = 3
    args.embedding_dim = 16
    args.hidden_dim = 32
    args.updates_per_episode = 2
    args.batch_size = 2
    args.roots_per_update = 4
    args.epsilon_decay_episodes = 2


def main() -> None:
    args = build_parser().parse_args()
    apply_smoke(args)
    if int(args.horizon) <= 0 or int(args.horizon) > int(args.episode_length):
        raise ValueError("horizon must be between one and episode_length.")
    device = resolve_device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model = RecurrentSequenceNovelty(
        action_space_size=3,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    suffixes = enumerate_action_suffixes(3, args.horizon)
    suffix_tensor = torch.from_numpy(suffixes).to(device)
    collection_rng = np.random.default_rng(int(args.seed) * 1009 + 43)
    training_rng = np.random.default_rng(int(args.seed) * 2017 + 61)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_dir = args.output_dir / "trajectories"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    arguments = vars(args).copy()
    arguments.update(
        output_dir=str(args.output_dir),
        device=str(device),
        model_receives_explicit_episodic_memory=False,
        training_counterfactual_transition_queries=0,
    )
    (args.output_dir / "arguments.json").write_text(
        json.dumps(arguments, indent=2), encoding="utf-8"
    )

    replay: List[Trajectory] = []
    rows: List[Mapping[str, object]] = []
    details: MutableMapping[str, object] = {"episodes": []}
    cumulative_positions = set()
    cumulative_poses = set()
    cumulative_rooms = set()

    for episode in range(int(args.episodes)):
        epsilon = epsilon_for_episode(args, episode)
        (
            trajectory,
            before_sequence_values,
            before_immediate_values,
            exploratory,
            rewards,
        ) = collect_episode(
            model,
            suffixes,
            suffix_tensor,
            args,
            episode,
            epsilon,
            collection_rng,
            device,
        )
        labels = exact_one_step_novelty(
            trajectory.observations,
            trajectory.poses,
            trajectory.actions,
            args.episode_length,
            args.tile_size,
        )
        before_rgb = planner_statistics(
            labels["rgb"], trajectory.actions, exploratory, before_sequence_values
        )
        before_position = planner_statistics(
            labels["position"], trajectory.actions, exploratory, before_sequence_values
        )
        before_prediction = selected_prediction_metrics(
            before_immediate_values,
            trajectory.actions,
            rewards,
        )

        replay.append(trajectory)
        if len(replay) > int(args.replay_capacity):
            replay.pop(0)
        update = train_from_replay(
            model,
            optimizer,
            replay,
            args,
            training_rng,
            device,
        )
        after_sequence_values, after_immediate_values = score_trajectory_roots(
            model,
            trajectory,
            suffixes,
            suffix_tensor,
            args.gamma,
            args.top_k,
            device,
        )
        after_prediction = selected_prediction_metrics(
            after_immediate_values,
            trajectory.actions,
            rewards,
        )
        correction = correction_statistics(
            labels["rgb"], before_sequence_values, after_sequence_values
        )

        cumulative_positions.update(map(tuple, trajectory.poses[:, :2].tolist()))
        cumulative_poses.update(map(tuple, trajectory.poses.tolist()))
        cumulative_rooms.update(room for room in trajectory.rooms if room is not None)
        behavior = coverage_metrics(trajectory)
        row = {
            "episode": episode,
            "epsilon": epsilon,
            **behavior,
            "novel_observation_count": int(rewards.sum()),
            "cumulative_unique_positions": len(cumulative_positions),
            "cumulative_unique_poses": len(cumulative_poses),
            "cumulative_rooms": len(cumulative_rooms),
            **update,
            **prefix_fields("rgb", before_rgb),
            **prefix_fields("position", before_position),
            **{f"{key}_before": value for key, value in before_prediction.items()},
            **{f"{key}_after": value for key, value in after_prediction.items()},
            **correction,
        }
        rows.append(row)
        details["episodes"].append(
            {
                "episode": episode,
                "epsilon": epsilon,
                "behavior": behavior,
                "rgb_before_update": before_rgb,
                "position_before_update": before_position,
                "selected_prediction_before": before_prediction,
                "selected_prediction_after": after_prediction,
                "update": update,
                "correction": correction,
            }
        )
        np.savez_compressed(
            trajectory_dir / f"episode_{episode:03d}.npz",
            observations=trajectory.observations,
            actions=trajectory.actions,
            poses=trajectory.poses,
            rooms=np.asarray(
                [(-1, -1) if room is None else room for room in trajectory.rooms],
                dtype=np.int16,
            ),
            novelty_rewards=rewards,
            sequence_values_before=before_sequence_values,
            sequence_values_after=after_sequence_values,
            immediate_probabilities_before=before_immediate_values,
            immediate_probabilities_after=after_immediate_values,
            exploratory_flags=exploratory,
            rgb_action_novelty=labels["rgb"],
            position_action_novelty=labels["position"],
        )
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "episode": episode,
                "arguments": arguments,
            },
            args.output_dir / "latest.pt",
        )
        write_outputs(args.output_dir, rows, details)
        print(
            f"episode={episode:03d} epsilon={epsilon:.3f} "
            f"positions={behavior['unique_positions']:3d} rooms={behavior['rooms']} "
            f"RGB-correct={format_optional(before_rgb['greedy_correct_when_avoidable_rate'])} "
            f"position-correct={format_optional(before_position['greedy_correct_when_avoidable_rate'])} "
            f"brier={before_prediction['selected_immediate_brier']:.4f}"
            f"->{after_prediction['selected_immediate_brier']:.4f} "
            f"corrected={format_optional(correction['rescored_corrected_rate'])} "
            f"regressed={format_optional(correction['rescored_regressed_rate'])} "
            f"loss={update['train_loss']:.5f}",
            flush=True,
        )
    print(f"wrote recurrent sequence-novelty results to {args.output_dir}")


if __name__ == "__main__":
    main()
