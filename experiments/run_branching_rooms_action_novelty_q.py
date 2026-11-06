"""Path-conditioned episodic Q-learning in Branching Rooms.

The Q model receives the real episode history at a search root and an action
prefix.  It is trained only on prefixes that actually occurred, using n-step
SARSA targets and a slowly updated target network.  During planning, frozen
novelty predictions provide short edge rewards and Q supplies the continuation
value.  No counterfactual environment transition is queried for training.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if __package__:
    from .run_branching_rooms_action_novelty import (
        NOVEL,
        ActionNoveltyPredictor,
        OnlinePredictorContext,
        load_dataset,
        state_ids_from_observations,
    )
    from .run_branching_rooms_online_action_novelty import (
        Trajectory,
        coverage_metrics,
        enumerate_action_suffixes,
        make_env,
        observation_tensor,
        trajectory_arrays,
    )
else:
    from run_branching_rooms_action_novelty import (
        NOVEL,
        ActionNoveltyPredictor,
        OnlinePredictorContext,
        load_dataset,
        state_ids_from_observations,
    )
    from run_branching_rooms_online_action_novelty import (
        Trajectory,
        coverage_metrics,
        enumerate_action_suffixes,
        make_env,
        observation_tensor,
        trajectory_arrays,
    )


CONDITIONS = ("random", "frozen_topk", "q_frozen", "q_online")


class PathConditionedEpisodicQ(nn.Module):
    """Q over an actual root history and a real or hypothetical action prefix."""

    def __init__(self, backbone: ActionNoveltyPredictor) -> None:
        super().__init__()
        if not backbone.use_history:
            raise ValueError("Path-conditioned Q requires the full-history backbone.")
        self.backbone = copy.deepcopy(backbone)
        feature_dim = int(self.backbone.classifier[0].in_features)
        self.q_head = nn.Sequential(
            nn.Linear(feature_dim, self.backbone.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.backbone.hidden_dim, 1),
        )
        nn.init.zeros_(self.q_head[-1].weight)
        nn.init.zeros_(self.q_head[-1].bias)

    def start_online_context(self, observation: torch.Tensor) -> OnlinePredictorContext:
        return self.backbone.start_online_context(observation)

    def advance_online_context(
        self,
        context: OnlinePredictorContext,
        action: torch.Tensor,
        observation: torch.Tensor,
    ) -> OnlinePredictorContext:
        return self.backbone.advance_online_context(context, action, observation)

    def score_action_prefixes(
        self,
        context: OnlinePredictorContext,
        action_suffixes: torch.Tensor,
    ) -> torch.Tensor:
        action_suffixes = action_suffixes.to(
            device=context.root_embedding.device, dtype=torch.long
        )
        if action_suffixes.ndim != 2:
            raise ValueError("Expected action suffixes shaped [count, horizon].")
        if context.history_states is None:
            raise ValueError("Path-conditioned Q requires history states.")
        count, horizon = action_suffixes.shape
        root_embeddings = context.root_embedding.expand(count, -1)
        root_history = context.history_states[:, -1].expand(count, -1)
        prefix_state = self.backbone.prefix_initial(
            torch.cat([root_embeddings, root_history], dim=-1)
        )
        keys = self.backbone.memory_key(context.history_states).squeeze(0)
        values = self.backbone.memory_value(context.history_states).squeeze(0)
        outputs = []
        for depth in range(horizon):
            prefix_state = self.backbone.prefix_cell(
                self.backbone.action_embedding(action_suffixes[:, depth]),
                prefix_state,
            )
            scores = torch.matmul(prefix_state, keys.transpose(0, 1))
            scores = scores / math.sqrt(float(self.backbone.hidden_dim))
            memory = torch.matmul(torch.softmax(scores, dim=-1), values)
            features = torch.cat(
                [prefix_state, memory, root_embeddings], dim=-1
            )
            outputs.append(torch.sigmoid(self.q_head(features).squeeze(-1)))
        return torch.stack(outputs, dim=1)

    def root_action_values(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        root_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score every next action from real replay histories."""
        action_count = actions.shape[1]
        if root_indices is None:
            root_indices = torch.arange(action_count, device=actions.device)
        else:
            root_indices = root_indices.to(device=actions.device, dtype=torch.long)
        embeddings = self.backbone._encode_observations(observations)
        history_states = self.backbone._history_states(embeddings, actions)
        root_embeddings = embeddings[:, root_indices]
        initial = self.backbone.prefix_initial(
            torch.cat(
                [root_embeddings, history_states[:, root_indices]], dim=-1
            )
        )
        batch_size, root_count = initial.shape[:2]
        candidate_count = self.backbone.action_space_size
        candidate_actions = torch.arange(
            candidate_count, device=actions.device
        ).reshape(1, 1, candidate_count).expand(batch_size, root_count, -1)
        repeated_initial = initial.unsqueeze(2).expand(-1, -1, candidate_count, -1)
        prefix_state = self.backbone.prefix_cell(
            self.backbone.action_embedding(candidate_actions).reshape(
                batch_size * root_count * candidate_count, -1
            ),
            repeated_initial.reshape(
                batch_size * root_count * candidate_count, -1
            ),
        ).reshape(batch_size, root_count, candidate_count, self.backbone.hidden_dim)
        repeated_roots = root_indices.repeat_interleave(candidate_count)
        memory = self.backbone._attend_to_root_history(
            prefix_state.reshape(
                batch_size, root_count * candidate_count, self.backbone.hidden_dim
            ),
            history_states,
            repeated_roots,
        ).reshape(batch_size, root_count, candidate_count, self.backbone.hidden_dim)
        repeated_embeddings = root_embeddings.unsqueeze(2).expand(
            -1, -1, candidate_count, -1
        )
        features = torch.cat(
            [prefix_state, memory, repeated_embeddings], dim=-1
        )
        return torch.sigmoid(self.q_head(features).squeeze(-1))

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        horizon: int,
        root_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        horizon = int(horizon)
        root_count = actions.shape[1] - horizon + 1
        if root_count <= 0:
            raise ValueError("The horizon must not exceed the episode length.")
        if root_indices is None:
            root_indices = torch.arange(root_count, device=actions.device)
        else:
            root_indices = root_indices.to(device=actions.device, dtype=torch.long)
        embeddings = self.backbone._encode_observations(observations)
        history_states = self.backbone._history_states(embeddings, actions)
        root_embeddings = embeddings[:, root_indices]
        prefix_state = self.backbone.prefix_initial(
            torch.cat(
                [root_embeddings, history_states[:, root_indices]], dim=-1
            )
        )
        outputs = []
        for depth in range(horizon):
            prefix_actions = actions[:, root_indices + depth]
            batch_size, query_count = prefix_actions.shape
            prefix_state = self.backbone.prefix_cell(
                self.backbone.action_embedding(prefix_actions).reshape(
                    batch_size * query_count, -1
                ),
                prefix_state.reshape(batch_size * query_count, -1),
            ).reshape(batch_size, query_count, self.backbone.hidden_dim)
            memory = self.backbone._attend_to_root_history(
                prefix_state, history_states, root_indices
            )
            features = torch.cat(
                [prefix_state, memory, root_embeddings], dim=-1
            )
            outputs.append(torch.sigmoid(self.q_head(features).squeeze(-1)))
        return torch.stack(outputs, dim=2)


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
        default=Path(".local_runs/branching_rooms_action_novelty_q_seed0"),
    )
    parser.add_argument(
        "--q-checkpoint",
        type=Path,
        default=None,
        help="Load a pretrained path-conditioned Q checkpoint and skip offline pretraining.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=CONDITIONS)
    parser.add_argument("--episodes-per-condition", type=int, default=8)
    parser.add_argument("--episode-length", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--tile-size", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--td-steps", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--epsilon", type=float, default=0.10)
    parser.add_argument("--offline-epochs", type=int, default=10)
    parser.add_argument("--offline-batch-size", type=int, default=8)
    parser.add_argument("--roots-per-update", type=int, default=32)
    parser.add_argument("--online-updates-per-episode", type=int, default=64)
    parser.add_argument("--online-batch-size", type=int, default=8)
    parser.add_argument("--online-replay-fraction", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--target-tau", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--smoke", action="store_true")
    return parser


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device(name)


def novelty_rewards(observations: np.ndarray) -> np.ndarray:
    state_ids = state_ids_from_observations(observations)
    rewards = np.zeros(state_ids.size - 1, dtype=np.float32)
    visited = {int(state_ids[0])}
    for action_index, state_id in enumerate(state_ids[1:]):
        state_id = int(state_id)
        rewards[action_index] = float(state_id not in visited)
        visited.add(state_id)
    return rewards


def discounted_mc_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    rewards = np.asarray(rewards, dtype=np.float32)
    returns = np.zeros_like(rewards)
    running = 0.0
    for index in range(rewards.size - 1, -1, -1):
        running = (1.0 - float(gamma)) * float(rewards[index]) + float(gamma) * running
        returns[index] = running
    return returns


def build_nstep_sarsa_targets(
    rewards: torch.Tensor,
    target_first_action_q: torch.Tensor,
    action_indices: torch.Tensor,
    td_steps: int,
    gamma: float,
) -> torch.Tensor:
    """Build normalized n-step returns for action indices [root, depth]."""
    batch_size, action_count = rewards.shape
    indices = action_indices.to(device=rewards.device, dtype=torch.long)
    targets = torch.zeros(
        (batch_size, *indices.shape), dtype=rewards.dtype, device=rewards.device
    )
    for offset in range(int(td_steps)):
        reward_indices = indices + offset
        valid = reward_indices < action_count
        gathered = rewards[:, reward_indices.clamp(max=action_count - 1)]
        targets = targets + (
            (1.0 - float(gamma))
            * (float(gamma) ** offset)
            * gathered
            * valid.unsqueeze(0)
        )
    bootstrap_indices = indices + int(td_steps)
    valid_bootstrap = bootstrap_indices < action_count
    bootstrap = target_first_action_q[
        :, bootstrap_indices.clamp(max=action_count - 1)
    ]
    targets = targets + (
        (float(gamma) ** int(td_steps))
        * bootstrap
        * valid_bootstrap.unsqueeze(0)
    )
    return targets


def q_batch_loss(
    model: PathConditionedEpisodicQ,
    target_model: PathConditionedEpisodicQ,
    observations: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    root_indices: torch.Tensor,
    horizon: int,
    td_steps: int,
    gamma: float,
) -> Tuple[torch.Tensor, Mapping[str, float]]:
    predictions = model(
        observations, actions, horizon, root_indices=root_indices
    )
    with torch.no_grad():
        all_roots = torch.arange(actions.shape[1], device=actions.device)
        target_first_action_q = target_model(
            observations, actions, 1, root_indices=all_roots
        ).squeeze(-1)
        action_indices = root_indices[:, None] + torch.arange(
            horizon, device=actions.device
        )[None, :]
        targets = build_nstep_sarsa_targets(
            rewards,
            target_first_action_q,
            action_indices,
            td_steps,
            gamma,
        )
    loss = F.smooth_l1_loss(predictions, targets)
    return loss, {
        "prediction_mean": float(predictions.detach().mean().item()),
        "target_mean": float(targets.mean().item()),
        "mae": float(torch.mean(torch.abs(predictions.detach() - targets)).item()),
    }


def soft_update(
    target: nn.Module, source: nn.Module, tau: float
) -> None:
    with torch.no_grad():
        for target_parameter, source_parameter in zip(
            target.parameters(), source.parameters()
        ):
            target_parameter.mul_(1.0 - float(tau)).add_(
                source_parameter, alpha=float(tau)
            )


def bundle_rewards(bundle) -> np.ndarray:
    return np.stack(
        [novelty_rewards(observations) for observations in bundle.observations]
    )


def sampled_roots(
    action_count: int,
    horizon: int,
    count: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    root_count = int(action_count) - int(horizon) + 1
    selected_count = min(root_count, int(count))
    roots = rng.choice(root_count, size=selected_count, replace=False)
    return torch.from_numpy(np.sort(roots).astype(np.int64))


def tensor_batch(
    observations: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.from_numpy(observations[indices]).to(device),
        torch.from_numpy(actions[indices].astype(np.int64, copy=False)).to(device),
        torch.from_numpy(rewards[indices]).to(device),
    )


def evaluate_q(
    model: PathConditionedEpisodicQ,
    observations: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    indices: np.ndarray,
    gamma: float,
    device: torch.device,
    batch_size: int = 8,
) -> Mapping[str, float | None]:
    predictions = []
    targets = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), int(batch_size)):
            batch_indices = indices[start : start + int(batch_size)]
            obs, act, _ = tensor_batch(
                observations, actions, rewards, batch_indices, device
            )
            roots = torch.arange(act.shape[1], device=device)
            prediction = model(obs, act, 1, root_indices=roots).squeeze(-1)
            predictions.append(prediction.cpu().numpy().reshape(-1))
            batch_targets = np.stack(
                [discounted_mc_returns(rewards[index], gamma) for index in batch_indices]
            )
            targets.append(batch_targets.reshape(-1))
    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    correlation = None
    if np.std(prediction) > 1e-8 and np.std(target) > 1e-8:
        correlation = float(np.corrcoef(prediction, target)[0, 1])
    return {
        "count": int(target.size),
        "mae": float(np.mean(np.abs(prediction - target))),
        "mse": float(np.mean((prediction - target) ** 2)),
        "correlation": correlation,
        "prediction_mean": float(prediction.mean()),
        "target_mean": float(target.mean()),
    }


def pretrain_q(
    args: argparse.Namespace,
    model: PathConditionedEpisodicQ,
    bundle,
    rewards: np.ndarray,
    device: torch.device,
) -> Tuple[PathConditionedEpisodicQ, Mapping[str, object]]:
    target_model = copy.deepcopy(model).to(device).eval()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    train_indices = np.flatnonzero(bundle.split_ids == 0)
    validation_indices = np.flatnonzero(bundle.split_ids == 1)
    if args.smoke:
        train_indices = train_indices[:8]
        validation_indices = validation_indices[:4]
    rng = np.random.default_rng(int(args.seed) * 3571 + 13)
    history = []
    best_mse = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    for epoch in range(int(args.offline_epochs)):
        shuffled = rng.permutation(train_indices)
        losses = []
        maes = []
        model.train()
        for start in range(0, len(shuffled), int(args.offline_batch_size)):
            batch_indices = shuffled[start : start + int(args.offline_batch_size)]
            observations, actions, batch_rewards = tensor_batch(
                bundle.observations,
                bundle.actions,
                rewards,
                batch_indices,
                device,
            )
            roots = sampled_roots(
                actions.shape[1],
                args.horizon,
                args.roots_per_update,
                rng,
            ).to(device)
            loss, diagnostics = q_batch_loss(
                model,
                target_model,
                observations,
                actions,
                batch_rewards,
                roots,
                args.horizon,
                args.td_steps,
                args.gamma,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()
            soft_update(target_model, model, args.target_tau)
            losses.append(float(loss.item()))
            maes.append(diagnostics["mae"])
        validation = evaluate_q(
            model,
            bundle.observations,
            bundle.actions,
            rewards,
            validation_indices,
            args.gamma,
            device,
            batch_size=args.offline_batch_size,
        )
        row = {
            "epoch": epoch + 1,
            "loss": float(np.mean(losses)),
            "td_mae": float(np.mean(maes)),
            **{f"validation_{key}": value for key, value in validation.items()},
        }
        history.append(row)
        print(
            f"Q pretrain {epoch + 1}/{args.offline_epochs}: "
            f"loss={row['loss']:.5f} td_mae={row['td_mae']:.4f} "
            f"val_mse={validation['mse']:.5f} val_corr={validation['correlation']}"
        )
        if float(validation["mse"]) < best_mse:
            best_mse = float(validation["mse"])
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, {"best_validation_mse": best_mse, "history": history}


def branch_scores(
    novelty_probabilities: np.ndarray,
    q_values: np.ndarray | None,
    gamma: float,
) -> np.ndarray:
    horizon = novelty_probabilities.shape[1]
    if q_values is None:
        weights = np.power(float(gamma), np.arange(horizon))
        return np.sum(novelty_probabilities[..., NOVEL] * weights[None, :], axis=1)
    if q_values.shape != novelty_probabilities.shape[:2]:
        raise ValueError("Q values and novelty predictions must share suffix/depth shape.")
    if horizon == 1:
        return q_values[:, 0]
    edge_weights = (1.0 - float(gamma)) * np.power(
        float(gamma), np.arange(horizon - 1)
    )
    edge_return = np.sum(
        novelty_probabilities[:, :-1, NOVEL] * edge_weights[None, :], axis=1
    )
    return edge_return + (float(gamma) ** (horizon - 1)) * q_values[:, -1]


def choose_action(
    scores: np.ndarray,
    suffixes: np.ndarray,
    top_k: int,
    epsilon: float,
    rng: np.random.Generator,
) -> Tuple[int, float, bool, np.ndarray]:
    values = []
    for action in range(3):
        selected = scores[suffixes[:, 0] == action]
        count = min(int(top_k), int(selected.size))
        values.append(float(np.partition(selected, -count)[-count:].mean()))
    values = np.asarray(values, dtype=np.float32)
    exploratory = bool(rng.random() < float(epsilon))
    if exploratory:
        action = int(rng.integers(0, 3))
    else:
        best = np.flatnonzero(
            np.isclose(values, values.max(), rtol=1e-7, atol=1e-9)
        )
        action = int(rng.choice(best))
    return action, float(values[action]), exploratory, values


def rebuild_context(model, observations, actions, device):
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


def run_episode(
    condition: str,
    novelty_model: ActionNoveltyPredictor,
    q_model: PathConditionedEpisodicQ | None,
    suffixes: np.ndarray,
    suffix_tensor: torch.Tensor,
    args: argparse.Namespace,
    episode: int,
    rng: np.random.Generator,
    device: torch.device,
) -> Tuple[Trajectory, Mapping[str, object], Mapping[str, np.ndarray]]:
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
    selected_scores = []
    action_values = []
    exploratory_flags = []
    novelty_context = rebuild_context(novelty_model, observations, actions, device)
    q_context = (
        rebuild_context(q_model, observations, actions, device)
        if q_model is not None
        else None
    )
    try:
        for _ in range(int(args.episode_length)):
            if condition == "random":
                action = int(rng.integers(0, 3))
            else:
                with torch.no_grad():
                    novelty_logits = novelty_model.score_action_suffixes(
                        novelty_context, suffix_tensor
                    )
                    novelty_probabilities = torch.softmax(
                        novelty_logits, dim=-1
                    ).cpu().numpy()
                    q_values = (
                        q_model.score_action_prefixes(q_context, suffix_tensor)
                        .cpu()
                        .numpy()
                        if q_model is not None
                        else None
                    )
                scores = branch_scores(
                    novelty_probabilities,
                    q_values,
                    args.gamma if q_model is not None else 1.0,
                )
                action, score, exploratory, values = choose_action(
                    scores, suffixes, args.top_k, args.epsilon, rng
                )
                selected_scores.append(score)
                action_values.append(values)
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
                    novelty_context = novelty_model.advance_online_context(
                        novelty_context,
                        torch.tensor([action], device=device),
                        observation_tensor(observation, device),
                    )
                    if q_model is not None:
                        q_context = q_model.advance_online_context(
                            q_context,
                            torch.tensor([action], device=device),
                            observation_tensor(observation, device),
                        )
            if terminated or truncated:
                break
    finally:
        env.close()
    trajectory = trajectory_arrays(observations, actions, poses, rooms)
    rewards = novelty_rewards(trajectory.observations)
    result = {
        **coverage_metrics(trajectory),
        "novel_observation_count": int(rewards.sum()),
        "normalized_discounted_return": float(discounted_mc_returns(rewards, args.gamma)[0]),
        "mean_selected_score": float(np.mean(selected_scores)) if selected_scores else None,
        "mean_action_value_margin": (
            float(np.mean(np.sort(np.stack(action_values), axis=1)[:, -1] - np.sort(np.stack(action_values), axis=1)[:, -2]))
            if action_values
            else None
        ),
        "epsilon_action_fraction": float(np.mean(exploratory_flags)) if exploratory_flags else 0.0,
    }
    arrays = {
        "novelty_rewards": rewards,
        "selected_scores": np.asarray(selected_scores, dtype=np.float32),
        "action_values": np.asarray(action_values, dtype=np.float32),
        "exploratory_flags": np.asarray(exploratory_flags, dtype=np.bool_),
    }
    return trajectory, result, arrays


def trajectories_to_arrays(
    trajectories: Sequence[Trajectory],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.stack([trajectory.observations for trajectory in trajectories]),
        np.stack([trajectory.actions for trajectory in trajectories]),
        np.stack([novelty_rewards(trajectory.observations) for trajectory in trajectories]),
    )


def online_q_updates(
    model: PathConditionedEpisodicQ,
    target_model: PathConditionedEpisodicQ,
    optimizer: torch.optim.Optimizer,
    online_replay: Sequence[Trajectory],
    bundle,
    offline_rewards: np.ndarray,
    args: argparse.Namespace,
    rng: np.random.Generator,
    device: torch.device,
) -> Mapping[str, float]:
    train_indices = np.flatnonzero(bundle.split_ids == 0)
    online_obs, online_actions, online_rewards = trajectories_to_arrays(online_replay)
    losses = []
    maes = []
    online_count = max(
        1,
        min(
            int(args.online_batch_size) - 1,
            int(round(args.online_batch_size * args.online_replay_fraction)),
        ),
    )
    offline_count = int(args.online_batch_size) - online_count
    online_action_count = int(online_actions.shape[1])
    if bundle.actions.shape[1] < online_action_count:
        raise ValueError(
            "Offline replay trajectories are shorter than the online trajectories."
        )
    for _ in range(int(args.online_updates_per_episode)):
        online_indices = rng.integers(0, len(online_replay), size=online_count)
        observations = [online_obs[online_indices]]
        actions = [online_actions[online_indices]]
        rewards = [online_rewards[online_indices]]
        if offline_count:
            selected = rng.choice(train_indices, size=offline_count, replace=True)
            observations.append(
                bundle.observations[selected, : online_action_count + 1]
            )
            actions.append(
                bundle.actions[selected, :online_action_count].astype(
                    np.int64, copy=False
                )
            )
            rewards.append(offline_rewards[selected, :online_action_count])
        obs = torch.from_numpy(np.concatenate(observations)).to(device)
        act = torch.from_numpy(np.concatenate(actions).astype(np.int64, copy=False)).to(device)
        rew = torch.from_numpy(np.concatenate(rewards)).to(device)
        roots = sampled_roots(
            act.shape[1], args.horizon, args.roots_per_update, rng
        ).to(device)
        model.train()
        loss, diagnostics = q_batch_loss(
            model,
            target_model,
            obs,
            act,
            rew,
            roots,
            args.horizon,
            args.td_steps,
            args.gamma,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        optimizer.step()
        soft_update(target_model, model, args.target_tau)
        losses.append(float(loss.item()))
        maes.append(diagnostics["mae"])
    return {"loss": float(np.mean(losses)), "td_mae": float(np.mean(maes))}


def finite_mean(values: Sequence[object]) -> float | None:
    selected = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return float(np.mean(selected)) if selected else None


def write_outputs(
    output_dir: Path,
    rows: Sequence[Mapping[str, object]],
    details: Mapping[str, object],
) -> Mapping[str, object]:
    with (output_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "episode_details.json").write_text(
        json.dumps(details, indent=2), encoding="utf-8"
    )
    summary: MutableMapping[str, object] = {}
    numeric_fields = (
        "unique_observations",
        "unique_positions",
        "unique_poses",
        "rooms",
        "dead_ends",
        "observation_revisit_fraction",
        "position_revisit_fraction",
        "pose_revisit_fraction",
        "novel_observation_count",
        "normalized_discounted_return",
        "mean_selected_score",
        "mean_action_value_margin",
    )
    for condition in CONDITIONS:
        selected = [row for row in rows if row["condition"] == condition]
        if selected:
            summary[condition] = {
                "episode_count": len(selected),
                **{
                    field: finite_mean([row[field] for row in selected])
                    for field in numeric_fields
                },
                "cumulative_unique_positions": selected[-1]["cumulative_unique_positions"],
                "cumulative_unique_poses": selected[-1]["cumulative_unique_poses"],
                "cumulative_rooms": selected[-1]["cumulative_rooms"],
            }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def apply_smoke(args: argparse.Namespace) -> None:
    if not args.smoke:
        return
    args.episodes_per_condition = 1
    args.episode_length = 12
    args.horizon = 3
    args.td_steps = 4
    args.offline_epochs = 1
    args.offline_batch_size = 4
    args.roots_per_update = 4
    args.online_updates_per_episode = 2
    args.online_batch_size = 4


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    checkpoint_path = args.offline_run / "full_history" / "best.pt"
    dataset_path = args.offline_run / "experienced_trajectories.npz"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    bundle = load_dataset(dataset_path)
    if args.episode_length is None:
        args.episode_length = int(bundle.metadata["episode_length"])
    if args.tile_size is None:
        args.tile_size = int(bundle.metadata["tile_size"])
    apply_smoke(args)
    if not 0.0 < float(args.gamma) < 1.0:
        raise ValueError("gamma must be strictly between zero and one.")
    if int(args.td_steps) <= 0 or int(args.top_k) <= 0:
        raise ValueError("td-steps and top-k must be positive.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    novelty_model = ActionNoveltyPredictor(
        action_space_size=3,
        embedding_dim=int(checkpoint["embedding_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        use_history=True,
    ).to(device)
    novelty_model.load_state_dict(checkpoint["model"])
    novelty_model.eval()
    for parameter in novelty_model.parameters():
        parameter.requires_grad_(False)

    q_model = PathConditionedEpisodicQ(novelty_model).to(device)
    offline_rewards = bundle_rewards(bundle)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_dir = args.output_dir / "trajectories"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    if args.q_checkpoint is None:
        q_model, pretraining = pretrain_q(
            args, q_model, bundle, offline_rewards, device
        )
    else:
        q_checkpoint = torch.load(
            args.q_checkpoint, map_location=device, weights_only=False
        )
        if "model" not in q_checkpoint:
            raise KeyError(f"Q checkpoint has no model state: {args.q_checkpoint}")
        q_model.load_state_dict(q_checkpoint["model"])
        pretraining = {
            "skipped": True,
            "loaded_checkpoint": str(args.q_checkpoint),
            "checkpoint_epoch": q_checkpoint.get("epoch"),
        }
    torch.save(
        {
            "model": q_model.state_dict(),
            "source_checkpoint": str(checkpoint_path),
            "gamma": float(args.gamma),
            "td_steps": int(args.td_steps),
            "horizon": int(args.horizon),
        },
        args.output_dir / "q_pretrained.pt",
    )
    (args.output_dir / "q_pretraining.json").write_text(
        json.dumps(pretraining, indent=2), encoding="utf-8"
    )

    arguments = vars(args).copy()
    arguments["offline_run"] = str(args.offline_run)
    arguments["output_dir"] = str(args.output_dir)
    arguments["q_checkpoint"] = (
        str(args.q_checkpoint) if args.q_checkpoint is not None else None
    )
    arguments["device"] = str(device)
    arguments["counterfactual_transition_queries"] = 0
    (args.output_dir / "arguments.json").write_text(
        json.dumps(arguments, indent=2), encoding="utf-8"
    )
    suffixes = enumerate_action_suffixes(3, args.horizon)
    suffix_tensor = torch.from_numpy(suffixes).to(device)
    rows: List[Mapping[str, object]] = []
    details: MutableMapping[str, object] = {}

    for condition_index, condition in enumerate(args.conditions):
        condition_q = (
            copy.deepcopy(q_model).to(device)
            if condition in ("q_frozen", "q_online")
            else None
        )
        target_q = (
            copy.deepcopy(condition_q).to(device).eval()
            if condition == "q_online"
            else None
        )
        optimizer = (
            torch.optim.AdamW(
                condition_q.parameters(),
                lr=float(args.learning_rate),
                weight_decay=float(args.weight_decay),
            )
            if condition == "q_online"
            else None
        )
        planner_stream = 0 if condition == "random" else 1
        planner_rng = np.random.default_rng(
            int(args.seed) * 1009 + planner_stream * 7919 + 43
        )
        training_rng = np.random.default_rng(
            int(args.seed) * 2017 + condition_index * 1237 + 61
        )
        online_replay: List[Trajectory] = []
        cumulative_positions = set()
        cumulative_poses = set()
        cumulative_rooms = set()
        condition_details = []
        for episode in range(int(args.episodes_per_condition)):
            trajectory, result, arrays = run_episode(
                condition,
                novelty_model,
                condition_q,
                suffixes,
                suffix_tensor,
                args,
                episode,
                planner_rng,
                device,
            )
            update = None
            if condition == "q_online":
                before = evaluate_q(
                    condition_q,
                    trajectory.observations[None],
                    trajectory.actions[None],
                    arrays["novelty_rewards"][None],
                    np.asarray([0]),
                    args.gamma,
                    device,
                    batch_size=1,
                )
                online_replay.append(trajectory)
                update = online_q_updates(
                    condition_q,
                    target_q,
                    optimizer,
                    online_replay,
                    bundle,
                    offline_rewards,
                    args,
                    training_rng,
                    device,
                )
                after = evaluate_q(
                    condition_q,
                    trajectory.observations[None],
                    trajectory.actions[None],
                    arrays["novelty_rewards"][None],
                    np.asarray([0]),
                    args.gamma,
                    device,
                    batch_size=1,
                )
                update = {**update, "before": before, "after": after}
                torch.save(
                    {"model": condition_q.state_dict(), "episodes": episode + 1},
                    args.output_dir / "q_online_latest.pt",
                )
                torch.save(
                    {"model": condition_q.state_dict(), "episodes": episode + 1},
                    args.output_dir / f"q_online_episode_{episode:03d}.pt",
                )
            cumulative_positions.update(map(tuple, trajectory.poses[:, :2].tolist()))
            cumulative_poses.update(map(tuple, trajectory.poses.tolist()))
            cumulative_rooms.update(room for room in trajectory.rooms if room is not None)
            row = {
                "condition": condition,
                "episode": episode,
                **result,
                "cumulative_unique_positions": len(cumulative_positions),
                "cumulative_unique_poses": len(cumulative_poses),
                "cumulative_rooms": len(cumulative_rooms),
            }
            rows.append(row)
            condition_details.append(
                {"episode": episode, "metrics": result, "update": update}
            )
            np.savez_compressed(
                trajectory_dir / f"{condition}_episode_{episode:03d}.npz",
                observations=trajectory.observations,
                actions=trajectory.actions,
                poses=trajectory.poses,
                rooms=np.asarray(
                    [(-1, -1) if room is None else room for room in trajectory.rooms],
                    dtype=np.int16,
                ),
                **arrays,
            )
            details[condition] = condition_details
            write_outputs(args.output_dir, rows, details)
            print(
                f"{condition} episode {episode + 1}/{args.episodes_per_condition}: "
                f"positions={result['unique_positions']} rooms={result['rooms']} "
                f"novel={result['novel_observation_count']} "
                f"return={result['normalized_discounted_return']:.4f}"
            )
        if condition == "q_online":
            torch.save(
                {"model": condition_q.state_dict(), "episodes": args.episodes_per_condition},
                args.output_dir / "q_online_final.pt",
            )
        details[condition] = condition_details
    summary = write_outputs(args.output_dir, rows, details)
    print(json.dumps(summary, indent=2))
    print(f"wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
