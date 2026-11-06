#!/usr/bin/env python3
"""Split long Branching Rooms trajectories into overlapping fixed-size windows.

Every window inherits its parent trajectory's dataset split.  This prevents
overlapping windows from the same trajectory appearing in both training and
validation data.  Novelty labels are recomputed relative to the window because
the recurrent world model is reset at the start of each training example.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

if __package__:
    from .run_branching_rooms_action_novelty import (
        DatasetBundle,
        build_novelty_targets,
        load_dataset,
        state_ids_from_observations,
        state_ids_from_poses,
    )
else:
    from run_branching_rooms_action_novelty import (
        DatasetBundle,
        build_novelty_targets,
        load_dataset,
        state_ids_from_observations,
        state_ids_from_poses,
    )


@dataclass(frozen=True)
class WindowedDataset:
    bundle: DatasetBundle
    parent_episode_indices: np.ndarray
    window_starts: np.ndarray


def window_starts(action_count: int, window_actions: int, stride: int) -> list[int]:
    action_count = int(action_count)
    window_actions = int(window_actions)
    stride = int(stride)
    if window_actions <= 0:
        raise ValueError("window_actions must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if window_actions > action_count:
        raise ValueError(
            f"window_actions={window_actions} exceeds action_count={action_count}"
        )
    final_start = action_count - window_actions
    starts = list(range(0, final_start + 1, stride))
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def build_windowed_dataset(
    source: DatasetBundle,
    window_actions: int,
    stride: int,
    horizon: int | None = None,
) -> WindowedDataset:
    if source.observations.ndim != 5:
        raise ValueError("observations must have shape [episode,time,C,H,W]")
    if source.actions.ndim != 2:
        raise ValueError("actions must have shape [episode,time]")
    if source.poses.ndim != 3:
        raise ValueError("poses must have shape [episode,time,pose_feature]")
    episode_count, observation_count = source.observations.shape[:2]
    action_count = source.actions.shape[1]
    if observation_count != action_count + 1:
        raise ValueError("each trajectory must have one more observation than action")
    if source.poses.shape[:2] != (episode_count, observation_count):
        raise ValueError("poses must align with observations")
    for name, values in (
        ("split_ids", source.split_ids),
        ("policy_ids", source.policy_ids),
        ("episode_seeds", source.episode_seeds),
    ):
        if len(values) != episode_count:
            raise ValueError(f"{name} must have one value per episode")

    if horizon is None:
        horizon = int(source.metadata["horizon"])
    horizon = int(horizon)
    if horizon <= 0 or horizon > int(window_actions):
        raise ValueError("horizon must be in [1, window_actions]")
    starts = window_starts(action_count, window_actions, stride)

    observations = []
    actions = []
    poses = []
    observation_targets = []
    pose_targets = []
    split_ids = []
    policy_ids = []
    episode_seeds = []
    parent_episode_indices = []
    all_window_starts = []

    seed_multiplier = action_count + 1
    for parent_index in range(episode_count):
        for start in starts:
            stop = start + int(window_actions)
            window_observations = source.observations[
                parent_index, start : stop + 1
            ]
            window_poses = source.poses[parent_index, start : stop + 1]
            observations.append(window_observations)
            actions.append(source.actions[parent_index, start:stop])
            poses.append(window_poses)
            observation_targets.append(
                build_novelty_targets(
                    state_ids_from_observations(window_observations), horizon
                )
            )
            pose_targets.append(
                build_novelty_targets(state_ids_from_poses(window_poses), horizon)
            )
            split_ids.append(int(source.split_ids[parent_index]))
            policy_ids.append(int(source.policy_ids[parent_index]))
            episode_seeds.append(
                int(source.episode_seeds[parent_index]) * seed_multiplier + start
            )
            parent_episode_indices.append(parent_index)
            all_window_starts.append(start)

    split_ids_array = np.asarray(split_ids, dtype=source.split_ids.dtype)
    split_names = list(source.metadata.get("split_names", []))
    split_count_length = max(
        len(split_names),
        int(split_ids_array.max()) + 1 if split_ids_array.size else 0,
    )
    split_counts = [
        int(np.sum(split_ids_array == split_id))
        for split_id in range(split_count_length)
    ]
    metadata = dict(source.metadata)
    metadata.update(
        {
            "episode_length": int(window_actions),
            "horizon": horizon,
            "split_counts": split_counts,
            "window_actions": int(window_actions),
            "window_stride": int(stride),
            "window_count_per_parent": len(starts),
            "source_episode_count": int(episode_count),
            "source_episode_length": int(action_count),
            "parent_split_preserved": True,
            "window_novelty_history_scope": "window_local",
        }
    )
    bundle = DatasetBundle(
        observations=np.stack(observations),
        actions=np.stack(actions),
        poses=np.stack(poses),
        observation_targets=np.stack(observation_targets),
        pose_targets=np.stack(pose_targets),
        split_ids=split_ids_array,
        policy_ids=np.asarray(policy_ids, dtype=source.policy_ids.dtype),
        episode_seeds=np.asarray(episode_seeds, dtype=np.int64),
        metadata=metadata,
    )
    return WindowedDataset(
        bundle=bundle,
        parent_episode_indices=np.asarray(parent_episode_indices, dtype=np.int32),
        window_starts=np.asarray(all_window_starts, dtype=np.int32),
    )


def write_windowed_dataset(
    output_path: Path,
    dataset: WindowedDataset,
    source_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(dataset.bundle.metadata)
    metadata["source_dataset"] = str(source_path)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary_path,
        observations=dataset.bundle.observations,
        actions=dataset.bundle.actions,
        poses=dataset.bundle.poses,
        observation_targets=dataset.bundle.observation_targets,
        pose_targets=dataset.bundle.pose_targets,
        split_ids=dataset.bundle.split_ids,
        policy_ids=dataset.bundle.policy_ids,
        episode_seeds=dataset.bundle.episode_seeds,
        parent_episode_indices=dataset.parent_episode_indices,
        window_starts=dataset.window_starts,
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    temporary_path.replace(output_path)


def split_summary(
    split_ids: np.ndarray, split_names: Sequence[str]
) -> Mapping[str, int]:
    return {
        str(name): int(np.sum(split_ids == split_id))
        for split_id, name in enumerate(split_names)
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-actions", type=int, default=128)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Defaults to the source dataset horizon.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source = load_dataset(args.input)
    windowed = build_windowed_dataset(
        source,
        window_actions=args.window_actions,
        stride=args.stride,
        horizon=args.horizon,
    )
    write_windowed_dataset(args.output, windowed, args.input)
    names = list(
        windowed.bundle.metadata.get(
            "split_names", [str(index) for index in range(4)]
        )
    )
    print(
        f"wrote {len(windowed.bundle.actions)} windows to {args.output}",
        flush=True,
    )
    print(
        "split counts: "
        + json.dumps(split_summary(windowed.bundle.split_ids, names), sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
