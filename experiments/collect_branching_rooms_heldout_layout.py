#!/usr/bin/env python3
"""Collect matched fixed-policy trajectories from a Branching Rooms layout."""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper

from experiments.run_branching_rooms_action_novelty import (
    BehaviourPolicy,
    build_novelty_targets,
    state_ids_from_observations,
    state_ids_from_poses,
)
from zoo.minigrid.envs.branching_rooms_env import (
    BRANCHING_ROOMS_ENV_ID,
    BRANCHING_ROOMS_HELDOUT_ENV_ID,
    BranchingRoomsEnv,
)


POLICY_NAMES = ("uniform", "persistent", "macro", "coverage")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            ".local_runs/branching_rooms_heldout_layout/experienced_trajectories.npz"
        ),
    )
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--episode-length", type=int, default=1200)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--tile-size", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument(
        "--layout",
        choices=("training", "heldout"),
        default="heldout",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=POLICY_NAMES,
        default=list(POLICY_NAMES[:3]),
    )
    return parser


def _canonical_observation(observation: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(np.transpose(observation, (2, 0, 1))).float()
    tensor = tensor.div(255.0)[None]
    tensor = F.interpolate(
        tensor,
        size=(28, 28),
        mode="bilinear",
        align_corners=False,
    )
    return torch.round(tensor[0] * 255.0).to(torch.uint8).numpy()


def _shortest_path(
    env: BranchingRoomsEnv,
    start: tuple[int, int],
    target: tuple[int, int],
) -> list[tuple[int, int]]:
    parent = {start: None}
    frontier = deque([start])
    while frontier:
        position = frontier.popleft()
        if position == target:
            break
        x, y = position
        for neighbor in ((x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1)):
            if neighbor in parent:
                continue
            cell = env.grid.get(*neighbor)
            if cell is not None and not cell.can_overlap():
                continue
            parent[neighbor] = position
            frontier.append(neighbor)
    if target not in parent:
        raise RuntimeError(f"No path from {start} to {target}")
    path = []
    cursor = target
    while cursor is not None:
        path.append(cursor)
        cursor = parent[cursor]
    return list(reversed(path))


def _path_actions(
    path: list[tuple[int, int]], start_direction: int
) -> tuple[list[int], int]:
    direction = int(start_direction)
    actions = []
    direction_for_delta = {
        (1, 0): 0,
        (0, 1): 1,
        (-1, 0): 2,
        (0, -1): 3,
    }
    for position, next_position in zip(path, path[1:]):
        delta = (
            next_position[0] - position[0],
            next_position[1] - position[1],
        )
        target_direction = direction_for_delta[delta]
        while direction != target_direction:
            clockwise = (target_direction - direction) % 4
            if clockwise <= 2:
                actions.append(1)
                direction = (direction + 1) % 4
            else:
                actions.append(0)
                direction = (direction - 1) % 4
        actions.append(2)
    return actions, direction


def _coverage_actions(
    env: BranchingRoomsEnv,
    episode_length: int,
    rng: np.random.Generator,
) -> list[int]:
    current = tuple(int(value) for value in env.agent_pos)
    direction = int(env.agent_dir)
    actions = []
    centers = {
        (row, col): tuple(
            int(value)
            for value in env.room_centers[
                row * env.room_count_per_axis + col
            ]
        )
        for row in range(env.room_count_per_axis)
        for col in range(env.room_count_per_axis)
    }
    edges = list(env.room_edges)
    while len(actions) < int(episode_length):
        rng.shuffle(edges)
        for raw_edge in edges:
            edge = raw_edge if rng.random() < 0.5 else tuple(reversed(raw_edge))
            for room in edge:
                target = centers[room]
                path = _shortest_path(env, current, target)
                path_actions, direction = _path_actions(path, direction)
                actions.extend(path_actions)
                current = target
                if len(actions) >= int(episode_length):
                    return actions[: int(episode_length)]
    return actions[: int(episode_length)]


def collect_episode(
    episode_length: int,
    tile_size: int,
    episode_seed: int,
    policy_name: str,
    environment_id: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(episode_seed) + 7919)
    policy = (
        None
        if policy_name == "coverage"
        else BehaviourPolicy(policy_name, rng)
    )
    env = gym.make(
        environment_id,
        max_steps=int(episode_length),
        agent_view_size=7,
    )
    env = RGBImgPartialObsWrapper(env, tile_size=int(tile_size))
    env = ImgObsWrapper(env)
    observation, info = env.reset(seed=int(episode_seed))
    coverage_actions = (
        _coverage_actions(env.unwrapped, episode_length, rng)
        if policy_name == "coverage"
        else None
    )
    observations = [_canonical_observation(np.asarray(observation))]
    poses = [
        (
            int(info["agent_position"][0]),
            int(info["agent_position"][1]),
            int(info["agent_direction"]),
        )
    ]
    actions = []
    try:
        for step in range(int(episode_length)):
            action = (
                int(coverage_actions[step])
                if coverage_actions is not None
                else policy.action()
            )
            observation, _, terminated, truncated, info = env.step(action)
            actions.append(action)
            observations.append(_canonical_observation(np.asarray(observation)))
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
            f"Expected {episode_length} transitions, collected {len(actions)}"
        )
    return (
        np.stack(observations),
        np.asarray(actions, dtype=np.uint8),
        np.asarray(poses, dtype=np.int16),
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.horizon <= 0 or args.horizon > args.episode_length:
        raise ValueError("--horizon must be in [1, episode length]")

    observations = []
    actions = []
    poses = []
    observation_targets = []
    pose_targets = []
    policy_ids = []
    episode_seeds = []
    environment_id = (
        BRANCHING_ROOMS_ENV_ID
        if args.layout == "training"
        else BRANCHING_ROOMS_HELDOUT_ENV_ID
    )
    for episode_index in range(args.episodes):
        policy_name = args.policies[episode_index % len(args.policies)]
        episode_seed = int(args.seed) + episode_index * 97
        episode_observations, episode_actions, episode_poses = collect_episode(
            args.episode_length,
            args.tile_size,
            episode_seed,
            policy_name,
            environment_id,
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
        policy_ids.append(args.policies.index(policy_name))
        episode_seeds.append(episode_seed)
        print(
            f"collected episode {episode_index + 1}/{args.episodes} "
            f"policy={policy_name}",
            flush=True,
        )

    metadata = {
        "seed": int(args.seed),
        "episode_length": int(args.episode_length),
        "horizon": int(args.horizon),
        "tile_size": int(args.tile_size),
        "split_names": ["heldout_layout"],
        "policy_names": list(args.policies),
        "environment_id": environment_id,
        "layout": args.layout,
        "layout_relation": (
            "training graph and doorways"
            if args.layout == "training"
            else "same geometry and colors; held-out graph and doorways"
        ),
        "target_source": "exact_rgb_observation_recurrence",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary,
        observations=np.stack(observations),
        actions=np.stack(actions),
        poses=np.stack(poses),
        observation_targets=np.stack(observation_targets),
        pose_targets=np.stack(pose_targets),
        split_ids=np.zeros(args.episodes, dtype=np.uint8),
        policy_ids=np.asarray(policy_ids, dtype=np.uint8),
        episode_seeds=np.asarray(episode_seeds, dtype=np.int64),
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    temporary.replace(args.output)
    print(f"dataset: {args.output}")


if __name__ == "__main__":
    main()
