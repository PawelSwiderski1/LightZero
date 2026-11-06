"""Train episodic Q normally from replay while it controls collection.

The model starts without an offline Q checkpoint.  Each completed episode is
added to replay with exact first-visit RGB novelty rewards.  Q is trained with
an n-step max-Q target, and the updated Q model selects subsequent collection
actions.  Exact alternative-action novelty is used only for diagnostics.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, MutableMapping, Sequence, Tuple

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
    from .run_branching_rooms_action_novelty import (
        ActionNoveltyPredictor,
        ObservationEncoder,
    )
    from .run_branching_rooms_action_novelty_q import (
        PathConditionedEpisodicQ,
        build_nstep_sarsa_targets,
        discounted_mc_returns,
        novelty_rewards,
        resolve_device,
        sampled_roots,
        soft_update,
    )
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
    from run_branching_rooms_action_novelty import (
        ActionNoveltyPredictor,
        ObservationEncoder,
    )
    from run_branching_rooms_action_novelty_q import (
        PathConditionedEpisodicQ,
        build_nstep_sarsa_targets,
        discounted_mc_returns,
        novelty_rewards,
        resolve_device,
        sampled_roots,
        soft_update,
    )
    from run_branching_rooms_online_action_novelty import (
        Trajectory,
        coverage_metrics,
        make_env,
        observation_tensor,
        trajectory_arrays,
    )


ARCHITECTURES = ("path_conditioned", "agent57_lstm")


@dataclass(frozen=True)
class Agent57QContext:
    hidden: torch.Tensor
    cell: torch.Tensor
    q_values: torch.Tensor


class Agent57StyleEpisodicQ(nn.Module):
    """Recurrent Q with Agent57-style action and reward history inputs."""

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
        self.recurrent = nn.LSTMCell(
            self.embedding_dim + action_dim + 2,
            self.hidden_dim,
        )
        self.q_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.action_space_size),
        )
        nn.init.zeros_(self.q_head[-1].weight)
        nn.init.zeros_(self.q_head[-1].bias)

    def _step(
        self,
        observation_embedding: torch.Tensor,
        previous_action: torch.Tensor,
        previous_intrinsic_reward: torch.Tensor,
        previous_extrinsic_reward: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
    ) -> Agent57QContext:
        recurrent_input = torch.cat(
            [
                observation_embedding,
                self.action_embedding(previous_action),
                previous_intrinsic_reward.reshape(-1, 1),
                previous_extrinsic_reward.reshape(-1, 1),
            ],
            dim=-1,
        )
        hidden, cell = self.recurrent(recurrent_input, (hidden, cell))
        q_values = torch.sigmoid(self.q_head(hidden))
        return Agent57QContext(hidden=hidden, cell=cell, q_values=q_values)

    def start_online_context(self, observation: torch.Tensor) -> Agent57QContext:
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
        previous_action = torch.full(
            (batch_size,),
            self.action_space_size,
            device=observation.device,
            dtype=torch.long,
        )
        zero_reward = torch.zeros(batch_size, device=observation.device)
        return self._step(
            self.observation_encoder(observation),
            previous_action,
            zero_reward,
            zero_reward,
            hidden,
            cell,
        )

    def advance_online_context(
        self,
        context: Agent57QContext,
        action: torch.Tensor,
        observation: torch.Tensor,
        intrinsic_reward: torch.Tensor,
        extrinsic_reward: torch.Tensor | None = None,
    ) -> Agent57QContext:
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
        return self._step(
            self.observation_encoder(observation),
            action,
            intrinsic_reward,
            extrinsic_reward,
            context.hidden,
            context.cell,
        )

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        intrinsic_rewards: torch.Tensor,
    ) -> torch.Tensor:
        """Return all-action Q-values at every observation in a replay trace."""
        if observations.ndim != 5 or actions.ndim != 2:
            raise ValueError("Expected batched observation and action sequences.")
        if intrinsic_rewards.shape != actions.shape:
            raise ValueError("Intrinsic rewards must have the same shape as actions.")
        batch_size, observation_count = observations.shape[:2]
        if observation_count != actions.shape[1] + 1:
            raise ValueError("Expected exactly one more observation than action.")
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
        outputs = []
        for index in range(observation_count):
            if index == 0:
                previous_action = start_action
                previous_intrinsic_reward = zero_reward
            else:
                previous_action = actions[:, index - 1]
                previous_intrinsic_reward = intrinsic_rewards[:, index - 1]
            context = self._step(
                embeddings[:, index],
                previous_action,
                previous_intrinsic_reward,
                zero_reward,
                hidden,
                cell,
            )
            hidden, cell = context.hidden, context.cell
            outputs.append(context.q_values)
        return torch.stack(outputs, dim=1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".local_runs/branching_rooms_online_replay_q_seed0"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=48)
    parser.add_argument("--episode-length", type=int, default=128)
    parser.add_argument("--tile-size", type=int, default=4)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument(
        "--architecture",
        choices=ARCHITECTURES,
        default="path_conditioned",
    )
    parser.add_argument("--train-horizon", type=int, default=5)
    parser.add_argument("--td-steps", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--updates-per-episode", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--roots-per-update", type=int, default=32)
    parser.add_argument("--replay-capacity", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--target-tau", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.10)
    parser.add_argument("--epsilon-decay-episodes", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--smoke", action="store_true")
    return parser


def epsilon_for_episode(args: argparse.Namespace, episode: int) -> float:
    progress = min(float(episode) / max(float(args.epsilon_decay_episodes), 1.0), 1.0)
    return float(args.epsilon_start) + progress * (
        float(args.epsilon_end) - float(args.epsilon_start)
    )


def start_context(
    model: nn.Module,
    observation: np.ndarray,
    device: torch.device,
):
    model.eval()
    with torch.no_grad():
        return model.start_online_context(observation_tensor(observation, device))


def collect_episode(
    model: nn.Module,
    args: argparse.Namespace,
    episode: int,
    epsilon: float,
    rng: np.random.Generator,
    device: torch.device,
) -> Tuple[Trajectory, np.ndarray, np.ndarray, np.ndarray]:
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
    values = []
    exploratory = []
    intrinsic_rewards: List[float] = []
    visited_observations = {np.ascontiguousarray(observation).tobytes()}
    context = start_context(model, observation, device)
    candidate_actions = torch.arange(3, device=device).reshape(3, 1)
    try:
        for _ in range(int(args.episode_length)):
            with torch.no_grad():
                if isinstance(model, Agent57StyleEpisodicQ):
                    action_values = context.q_values[0].cpu().numpy()
                else:
                    action_values = model.score_action_prefixes(
                        context, candidate_actions
                    )[:, 0].cpu().numpy()
            is_exploratory = bool(rng.random() < float(epsilon))
            if is_exploratory:
                action = int(rng.integers(0, 3))
            else:
                best = np.flatnonzero(
                    np.isclose(
                        action_values,
                        action_values.max(),
                        rtol=1e-7,
                        atol=1e-9,
                    )
                )
                action = int(rng.choice(best))
            observation, extrinsic_reward, terminated, truncated, info = env.step(
                action
            )
            observation_key = np.ascontiguousarray(observation).tobytes()
            intrinsic_reward = float(observation_key not in visited_observations)
            visited_observations.add(observation_key)
            values.append(action_values)
            exploratory.append(is_exploratory)
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
                if isinstance(model, Agent57StyleEpisodicQ):
                    context = model.advance_online_context(
                        context,
                        torch.tensor([action], device=device),
                        observation_tensor(observation, device),
                        torch.tensor([intrinsic_reward], device=device),
                        torch.tensor([extrinsic_reward], device=device),
                    )
                else:
                    context = model.advance_online_context(
                        context,
                        torch.tensor([action], device=device),
                        observation_tensor(observation, device),
                    )
            if terminated or truncated:
                break
    finally:
        env.close()
    trajectory = trajectory_arrays(observations, actions, poses, rooms)
    rewards = np.asarray(intrinsic_rewards, dtype=np.float32)
    expected_rewards = novelty_rewards(trajectory.observations)
    if not np.array_equal(rewards, expected_rewards):
        raise RuntimeError("Online and replay novelty rewards disagree.")
    return (
        trajectory,
        np.asarray(values, dtype=np.float32),
        np.asarray(exploratory, dtype=np.bool_),
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


def dqn_batch_loss(
    model: nn.Module,
    target_model: nn.Module,
    observations: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    roots: torch.Tensor,
    horizon: int,
    td_steps: int,
    gamma: float,
) -> Tuple[torch.Tensor, Mapping[str, float]]:
    action_indices = roots[:, None] + torch.arange(
        horizon, device=actions.device
    )[None, :]
    if isinstance(model, Agent57StyleEpisodicQ):
        all_q = model(observations, actions, rewards)[:, :-1]
        indexed_q = all_q[:, action_indices]
        indexed_actions = actions[:, action_indices]
        predictions = torch.gather(
            indexed_q,
            dim=-1,
            index=indexed_actions.unsqueeze(-1),
        ).squeeze(-1)
    else:
        predictions = model(observations, actions, horizon, root_indices=roots)
    with torch.no_grad():
        if isinstance(target_model, Agent57StyleEpisodicQ):
            target_max_q = target_model(
                observations, actions, rewards
            )[:, :-1].max(dim=-1).values
        else:
            all_roots = torch.arange(actions.shape[1], device=actions.device)
            target_max_q = target_model.root_action_values(
                observations, actions, all_roots
            ).max(dim=-1).values
        targets = build_nstep_sarsa_targets(
            rewards,
            target_max_q,
            action_indices,
            td_steps,
            gamma,
        )
    loss = F.smooth_l1_loss(predictions, targets)
    return loss, {
        "mae": float(torch.mean(torch.abs(predictions.detach() - targets)).item()),
        "prediction_mean": float(predictions.detach().mean().item()),
        "target_mean": float(targets.mean().item()),
    }


def train_from_replay(
    model: nn.Module,
    target_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    replay: Sequence[Trajectory],
    args: argparse.Namespace,
    rng: np.random.Generator,
    device: torch.device,
) -> Mapping[str, float]:
    observations, actions, rewards = replay_arrays(replay)
    losses = []
    maes = []
    for _ in range(int(args.updates_per_episode)):
        indices = rng.integers(0, len(replay), size=int(args.batch_size))
        obs = torch.from_numpy(observations[indices]).to(device)
        act = torch.from_numpy(actions[indices].astype(np.int64, copy=False)).to(device)
        rew = torch.from_numpy(rewards[indices]).to(device)
        roots = sampled_roots(
            act.shape[1],
            args.train_horizon,
            args.roots_per_update,
            rng,
        ).to(device)
        model.train()
        loss, diagnostics = dqn_batch_loss(
            model,
            target_model,
            obs,
            act,
            rew,
            roots,
            args.train_horizon,
            args.td_steps,
            args.gamma,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        optimizer.step()
        soft_update(target_model, model, args.target_tau)
        losses.append(float(loss.item()))
        maes.append(float(diagnostics["mae"]))
    return {"train_loss": float(np.mean(losses)), "train_td_mae": float(np.mean(maes))}


def score_trajectory_roots(
    model: nn.Module,
    trajectory: Trajectory,
    device: torch.device,
) -> np.ndarray:
    actions = torch.from_numpy(trajectory.actions[None]).to(device)
    observations = torch.from_numpy(trajectory.observations[None]).to(device)
    roots = torch.arange(actions.shape[1], device=device)
    model.eval()
    with torch.no_grad():
        if isinstance(model, Agent57StyleEpisodicQ):
            rewards = torch.from_numpy(
                novelty_rewards(trajectory.observations)[None]
            ).to(device)
            return model(observations, actions, rewards)[0, :-1].cpu().numpy()
        return model.root_action_values(observations, actions, roots)[0].cpu().numpy()


def prefix_fields(prefix: str, values: Mapping[str, object]) -> Mapping[str, object]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def format_optional_rate(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


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


def apply_smoke(args: argparse.Namespace) -> None:
    if not args.smoke:
        return
    args.episodes = 3
    args.episode_length = 24
    args.train_horizon = 3
    args.td_steps = 4
    args.updates_per_episode = 2
    args.batch_size = 2
    args.roots_per_update = 4
    args.epsilon_decay_episodes = 2


def main() -> None:
    args = build_parser().parse_args()
    apply_smoke(args)
    device = resolve_device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.architecture == "agent57_lstm":
        model = Agent57StyleEpisodicQ(
            action_space_size=3,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
        ).to(device)
    else:
        backbone = ActionNoveltyPredictor(
            action_space_size=3,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            use_history=True,
        )
        model = PathConditionedEpisodicQ(backbone).to(device)
    target_model = copy.deepcopy(model).to(device).eval()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    collection_rng = np.random.default_rng(int(args.seed) * 1009 + 43)
    training_rng = np.random.default_rng(int(args.seed) * 2017 + 61)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_dir = args.output_dir / "trajectories"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    arguments = vars(args).copy()
    arguments.update(output_dir=str(args.output_dir), device=str(device))
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
        trajectory, before_values, exploratory, rewards = collect_episode(
            model, args, episode, epsilon, collection_rng, device
        )
        labels = exact_one_step_novelty(
            trajectory.observations,
            trajectory.poses,
            trajectory.actions,
            args.episode_length,
            args.tile_size,
        )
        before_rgb = planner_statistics(
            labels["rgb"], trajectory.actions, exploratory, before_values
        )
        before_position = planner_statistics(
            labels["position"], trajectory.actions, exploratory, before_values
        )
        replay.append(trajectory)
        if len(replay) > int(args.replay_capacity):
            replay.pop(0)
        update = train_from_replay(
            model,
            target_model,
            optimizer,
            replay,
            args,
            training_rng,
            device,
        )
        after_values = score_trajectory_roots(model, trajectory, device)
        correction = correction_statistics(
            labels["rgb"], before_values, after_values
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
            "normalized_discounted_return": float(
                discounted_mc_returns(rewards, args.gamma)[0]
            ),
            "cumulative_unique_positions": len(cumulative_positions),
            "cumulative_unique_poses": len(cumulative_poses),
            "cumulative_rooms": len(cumulative_rooms),
            **update,
            **prefix_fields("rgb", before_rgb),
            **prefix_fields("position", before_position),
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
            q_values_before=before_values,
            q_values_after=after_values,
            exploratory_flags=exploratory,
            rgb_action_novelty=labels["rgb"],
            position_action_novelty=labels["position"],
        )
        torch.save(
            {
                "model": model.state_dict(),
                "target_model": target_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "episode": episode,
            },
            args.output_dir / "latest.pt",
        )
        write_outputs(args.output_dir, rows, details)
        print(
            f"episode={episode:03d} epsilon={epsilon:.3f} "
            f"positions={behavior['unique_positions']:3d} "
            f"rooms={behavior['rooms']} "
            "RGB-correct="
            f"{format_optional_rate(before_rgb['greedy_correct_when_avoidable_rate'])} "
            "position-correct="
            f"{format_optional_rate(before_position['greedy_correct_when_avoidable_rate'])} "
            "corrected="
            f"{format_optional_rate(correction['rescored_corrected_rate'])} "
            f"loss={update['train_loss']:.5f}"
        )
    print(f"wrote online replay-Q results to {args.output_dir}")


if __name__ == "__main__":
    main()
