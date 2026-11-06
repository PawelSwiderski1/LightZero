#!/usr/bin/env python3
"""Audit a spatial-revisit model on the exact replay retained at checkpoint time."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from experiments.evaluate_branching_rooms_spatial_online_checkpoint import (
    _annotate_room_crossings,
    _json_safe,
    _model_from_state,
    _spatial_state,
    _write_csv,
)
from experiments.run_branching_rooms_spatial_exploration_world_model import (
    _pairwise_revisit_scores,
    gather_spatial_rollout_batch,
)
from lzero.entry.spatial_exploration_shadow import (
    SpatialExplorationShadowEvaluator,
)
from zoo.minigrid.envs.branching_rooms_env import BranchingRoomsEnv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--roots-per-episode", type=int, default=0)
    parser.add_argument("--root-batch-size", type=int, default=128)
    parser.add_argument("--history-window", type=int, default=128)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser


def _reconstruct_poses(actions: np.ndarray) -> np.ndarray:
    env = BranchingRoomsEnv()
    _, info = env.reset()
    poses = [
        (
            int(info["agent_position"][0]),
            int(info["agent_position"][1]),
            int(info["agent_direction"]),
        )
    ]
    try:
        for action in np.asarray(actions, dtype=np.int64).reshape(-1):
            _, _, terminated, truncated, info = env.step(int(action))
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
    if len(poses) != len(actions) + 1:
        raise ValueError(
            "A replay episode ended before all stored actions were applied; "
            "its start state cannot be reconstructed safely."
        )
    return np.asarray(poses, dtype=np.int16)


def _load_replay_episodes(
    state: Mapping[str, Any], maximum_episode_count: int | None
) -> list[Dict[str, Any]]:
    stored = list(state.get("replay", []))
    if not stored:
        raise ValueError(
            "Checkpoint has no replay. Extract it again with --include-replay."
        )
    if maximum_episode_count is not None:
        if maximum_episode_count <= 0:
            raise ValueError("--max-episodes must be positive")
        stored = stored[:maximum_episode_count]
    episodes = []
    for episode_index, episode in enumerate(stored):
        observations = np.asarray(episode["observations"], dtype=np.uint8)
        actions = np.asarray(episode["actions"], dtype=np.int64).reshape(-1)
        step_count = min(len(actions), max(0, len(observations) - 1))
        if step_count != len(actions):
            raise ValueError(
                f"Replay episode {episode_index} has inconsistent observations/actions"
            )
        episodes.append(
            {
                "observations": observations,
                "actions": actions,
                "environment_states": _reconstruct_poses(actions),
                "env_id": int(episode["replay_id"]),
                "env_episode_index": int(episode["replay_id"]),
                "collection_episode_index": int(episode["replay_id"]),
                "replay_id": int(episode["replay_id"]),
            }
        )
    return episodes


def _trained_roots(
    state: Mapping[str, Any], retained_ids: set[int]
) -> tuple[Dict[int, set[int]], Dict[tuple[int, int], list[float]]]:
    roots: Dict[int, set[int]] = defaultdict(set)
    priorities: Dict[tuple[int, int], list[float]] = defaultdict(list)
    for raw_key, raw_priority in state.get(
        "online_root_priorities", {}
    ).items():
        key = tuple(int(value) for value in raw_key)
        if len(key) != 4:
            continue
        replay_id, window_start, _, root_offset = key
        if replay_id not in retained_ids:
            continue
        root = window_start + root_offset
        roots[replay_id].add(root)
        priorities[(replay_id, root)].append(float(raw_priority))
    return roots, priorities


def _exact_state_ids(observations: np.ndarray) -> np.ndarray:
    mapping: Dict[bytes, int] = {}
    output = np.empty(len(observations), dtype=np.int64)
    for index, observation in enumerate(observations):
        key = np.ascontiguousarray(observation, dtype=np.uint8).tobytes()
        mapping.setdefault(key, len(mapping))
        output[index] = mapping[key]
    return output


def _first_visit(values: np.ndarray) -> np.ndarray:
    seen = set()
    output = np.empty(len(values), dtype=np.int64)
    for index, value in enumerate(values):
        key = tuple(int(item) for item in np.asarray(value).reshape(-1))
        output[index] = int(key not in seen)
        seen.add(key)
    return output


def _priority_items(
    state: Mapping[str, Any],
    episodes_by_id: Mapping[int, Mapping[str, Any]],
    horizon: int,
) -> list[tuple[tuple[int, int, int, int], float]]:
    items = []
    for raw_key, raw_priority in state.get(
        "online_root_priorities", {}
    ).items():
        key = tuple(int(value) for value in raw_key)
        if len(key) != 4:
            continue
        replay_id, start, action_count, root_offset = key
        episode = episodes_by_id.get(replay_id)
        if episode is None:
            continue
        if start < 0 or action_count < horizon or root_offset < 0:
            continue
        if root_offset > action_count - horizon:
            continue
        if start + action_count > len(episode["actions"]):
            continue
        items.append((key, float(raw_priority)))
    return items


def _evaluate_exact_training_windows(
    model: torch.nn.Module,
    state: Mapping[str, Any],
    episodes: Sequence[Mapping[str, Any]],
    device: torch.device,
    horizon: int,
    batch_size: int,
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    episodes_by_id = {
        int(episode["replay_id"]): episode for episode in episodes
    }
    items = _priority_items(state, episodes_by_id, horizon)
    grouped: Dict[
        tuple[int, int],
        list[tuple[tuple[int, int, int, int], float]],
    ] = defaultdict(list)
    for item in items:
        key, _ = item
        grouped[(key[2], key[3])].append(item)

    full_rgb_novel = {
        replay_id: _first_visit(
            _exact_state_ids(np.asarray(episode["observations"]))[:, None]
        )
        for replay_id, episode in episodes_by_id.items()
    }
    full_position_novel = {
        replay_id: _first_visit(
            np.asarray(episode["environment_states"])[:, :2]
        )
        for replay_id, episode in episodes_by_id.items()
    }
    rows: list[Dict[str, Any]] = []
    root_rows: list[Dict[str, Any]] = []
    model.eval()
    with torch.inference_mode():
        for (action_count, root_offset), group in sorted(grouped.items()):
            root = torch.tensor(
                [root_offset], dtype=torch.long, device=device
            )
            for batch_start in range(0, len(group), batch_size):
                batch_items = group[batch_start:batch_start + batch_size]
                observation_batch = []
                action_batch = []
                for key, _ in batch_items:
                    replay_id, start, _, _ = key
                    episode = episodes_by_id[replay_id]
                    observation_batch.append(
                        episode["observations"][start:start + action_count + 1]
                    )
                    action_batch.append(
                        episode["actions"][start:start + action_count]
                    )
                observations = torch.as_tensor(
                    np.stack(observation_batch),
                    dtype=torch.uint8,
                    device=device,
                )
                actions = torch.as_tensor(
                    np.stack(action_batch),
                    dtype=torch.long,
                    device=device,
                )
                flattened = observations.reshape(observations.shape[0], observations.shape[1], -1)
                _, inverse = torch.unique(
                    flattened.reshape(-1, flattened.shape[-1]),
                    dim=0,
                    return_inverse=True,
                )
                state_ids = inverse.reshape(observations.shape[:2])
                encoded = model.encode_trajectory(observations, actions)
                gathered = gather_spatial_rollout_batch(
                    encoded,
                    observations,
                    actions,
                    state_ids,
                    root,
                    horizon,
                )
                rollout = model.rollout(
                    gathered["root_latents"],
                    gathered["root_beliefs"],
                    gathered["suffix_actions"],
                )
                predicted_beliefs = rollout.beliefs.reshape(
                    len(batch_items), 1, horizon, model.belief_dim
                )
                predicted_scores, teacher_scores = _pairwise_revisit_scores(
                    model,
                    predicted_beliefs,
                    encoded.beliefs,
                    state_ids,
                    gathered["target_ids"],
                    root,
                )
                target_beliefs = gathered["target_beliefs"].reshape_as(
                    predicted_beliefs
                )
                target_observation_keys = gathered[
                    "target_observation_keys"
                ].reshape(
                    len(batch_items), 1, horizon, model.key_dim
                )
                predicted_observation_keys = rollout.observation_keys.reshape(
                    len(batch_items), 1, horizon, model.key_dim
                )

                for item_index, (key, stored_priority) in enumerate(batch_items):
                    replay_id, window_start, _, _ = key
                    episode_index = next(
                        index
                        for index, episode in enumerate(episodes)
                        if int(episode["replay_id"]) == replay_id
                    )
                    global_root = window_start + root_offset
                    root_prediction_sum = 0.0
                    root_teacher_sum = 0.0
                    root_rgb_count = 0
                    root_position_count = 0
                    for depth_index in range(horizon):
                        depth = depth_index + 1
                        global_endpoint = global_root + depth
                        predicted = float(
                            predicted_scores[item_index, 0, depth_index]
                        )
                        teacher = int(
                            round(
                                float(
                                    teacher_scores[
                                        item_index, 0, depth_index
                                    ]
                                )
                            )
                        )
                        position_novel = int(
                            full_position_novel[replay_id][global_endpoint]
                        )
                        root_prediction_sum += predicted
                        root_teacher_sum += teacher
                        root_rgb_count += teacher
                        root_position_count += position_novel
                        rows.append(
                            {
                                "collection_phase": 0,
                                "dynamics_history_window": action_count,
                                "episode_index_in_phase": episode_index,
                                "collection_episode_index": replay_id,
                                "env_id": replay_id,
                                "env_episode_index": replay_id,
                                "replay_id": replay_id,
                                "window_start": window_start,
                                "root_offset": root_offset,
                                "root_timestep": global_root,
                                "depth": depth,
                                "endpoint_timestep": global_endpoint,
                                "predicted_novelty": predicted,
                                "teacher_novelty": float(teacher),
                                "actual_rgb_novel": teacher,
                                "full_episode_rgb_novel": int(
                                    full_rgb_novel[replay_id][global_endpoint]
                                ),
                                "actual_position_novel": position_novel,
                                "predicted_target_cosine": float(
                                    F.cosine_similarity(
                                        predicted_beliefs[
                                            item_index, 0, depth_index
                                        ][None],
                                        target_beliefs[
                                            item_index, 0, depth_index
                                        ][None],
                                        dim=-1,
                                    )[0]
                                ),
                                "predicted_observation_key_cosine": float(
                                    F.cosine_similarity(
                                        predicted_observation_keys[
                                            item_index, 0, depth_index
                                        ][None],
                                        target_observation_keys[
                                            item_index, 0, depth_index
                                        ][None],
                                        dim=-1,
                                    )[0]
                                ),
                                "rgb_retrieval_correct": 0,
                                "rollout_rgb_mae": float("nan"),
                                "stored_training_priority": stored_priority,
                            }
                        )
                    root_rows.append(
                        {
                            "collection_phase": 0,
                            "dynamics_history_window": action_count,
                            "episode_index_in_phase": episode_index,
                            "collection_episode_index": replay_id,
                            "env_id": replay_id,
                            "root_timestep": global_root,
                            "predicted_novelty_sum": root_prediction_sum,
                            "teacher_novelty_sum": root_teacher_sum,
                            "actual_rgb_novel_count": root_rgb_count,
                            "actual_position_novel_count": root_position_count,
                        }
                    )
    return rows, root_rows


def _annotate_training_roots(
    rows: Sequence[Dict[str, Any]],
    episodes: Sequence[Mapping[str, Any]],
    trained_roots: Mapping[int, set[int]],
    priorities: Mapping[tuple[int, int], Sequence[float]],
) -> None:
    for row in rows:
        episode = episodes[int(row["episode_index_in_phase"])]
        replay_id = int(episode["replay_id"])
        root = int(row["root_timestep"])
        values = priorities.get((replay_id, root), ())
        row["replay_id"] = replay_id
        row["exact_root_used_for_training"] = int(
            root in trained_roots.get(replay_id, set())
        )
        row["training_priority_entry_count"] = len(values)
        row["training_priority_mean"] = (
            float(np.mean(values)) if values else float("nan")
        )
        row["training_priority_max"] = (
            float(np.max(values)) if values else float("nan")
        )


def _summaries(
    evaluator: SpatialExplorationShadowEvaluator,
    rows: Sequence[Mapping[str, Any]],
    root_rows: Sequence[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    output = [
        {"subset": "all_retained_replay", **summary}
        for summary in evaluator.summarize(rows, 0, root_rows)
    ]
    for value, label in ((1, "cached_training_root"), (0, "other_root")):
        selected = [
            row
            for row in rows
            if int(row["exact_root_used_for_training"]) == value
        ]
        if selected:
            output.extend(
                {"subset": f"deployed_context:{label}", **summary}
                for summary in evaluator.summarize(selected, 0)
            )
    categories = sorted({str(row["room_crossing_category"]) for row in rows})
    for category in categories:
        selected = [
            row
            for row in rows
            if str(row["room_crossing_category"]) == category
        ]
        output.extend(
            {"subset": f"room_crossing:{category}", **summary}
            for summary in evaluator.summarize(selected, 0)
        )
        trained = [
            row
            for row in selected
            if int(row["exact_root_used_for_training"]) == 1
        ]
        if trained:
            output.extend(
                {
                    "subset": (
                        "deployed_context_cached_root_room_crossing:"
                        f"{category}"
                    ),
                    **summary,
                }
                for summary in evaluator.summarize(trained, 0)
            )
    return output


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(
        args.checkpoint, map_location=device, weights_only=False
    )
    state = _spatial_state(checkpoint)
    episodes = _load_replay_episodes(state, args.max_episodes)
    retained_ids = {int(episode["replay_id"]) for episode in episodes}
    trained_roots, priorities = _trained_roots(state, retained_ids)
    if not any(trained_roots.values()):
        raise ValueError("Checkpoint contains no cached trained roots")

    model = _model_from_state(state["model"], device)
    evaluator = SpatialExplorationShadowEvaluator.from_live_model(
        model=model,
        device=str(device),
        horizon=args.horizon,
        roots_per_episode=(
            2**31 - 1
            if args.roots_per_episode <= 0
            else args.roots_per_episode
        ),
        root_batch_size=args.root_batch_size,
        dynamics_history_window=args.history_window,
        revisit_scoring_mode="model",
    )
    rows, root_rows = evaluator.evaluate_episodes(episodes, collection_phase=0)
    _annotate_training_roots(
        rows, episodes, trained_roots, priorities
    )
    _annotate_room_crossings(rows, episodes)
    summaries = _summaries(evaluator, rows, root_rows)
    exact_rows, exact_root_rows = _evaluate_exact_training_windows(
        model=model,
        state=state,
        episodes=episodes,
        device=device,
        horizon=args.horizon,
        batch_size=args.root_batch_size,
    )
    _annotate_room_crossings(exact_rows, episodes)
    summaries.extend(
        {
            "subset": "exact_training_window",
            **summary,
        }
        for summary in evaluator.summarize(
            exact_rows, 0, exact_root_rows
        )
    )
    exact_full_label_rows = [
        {
            **row,
            "actual_rgb_novel": int(row["full_episode_rgb_novel"]),
        }
        for row in exact_rows
    ]
    summaries.extend(
        {
            "subset": "exact_training_window_with_full_episode_labels",
            **summary,
        }
        for summary in evaluator.summarize(exact_full_label_rows, 0)
    )
    for category in sorted(
        {str(row["room_crossing_category"]) for row in exact_rows}
    ):
        selected = [
            row
            for row in exact_rows
            if str(row["room_crossing_category"]) == category
        ]
        summaries.extend(
            {
                "subset": f"exact_training_window_room_crossing:{category}",
                **summary,
            }
            for summary in evaluator.summarize(selected, 0)
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "steps.csv", rows)
    _write_csv(args.output_dir / "roots.csv", root_rows)
    _write_csv(args.output_dir / "exact_training_window_steps.csv", exact_rows)
    _write_csv(
        args.output_dir / "exact_training_window_roots.csv", exact_root_rows
    )
    _write_csv(args.output_dir / "summary.csv", summaries)
    result = {
        "checkpoint": str(args.checkpoint),
        "online_update_count": int(state.get("online_update_count", -1)),
        "online_collection_phase": int(
            state.get("online_collection_phase", -1)
        ),
        "replay_episode_count": len(episodes),
        "priority_cache_entry_count": len(
            state.get("online_root_priorities", {})
        ),
        "distinct_trained_root_count": sum(
            len(values) for values in trained_roots.values()
        ),
        "prediction_count": len(rows),
        "exact_training_window_prediction_count": len(exact_rows),
        "summary": summaries,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(_json_safe(result), indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"replay episodes={len(episodes)} "
        f"trained roots={result['distinct_trained_root_count']} "
        f"deployed predictions={len(rows)} "
        f"exact-window predictions={len(exact_rows)}"
    )
    for row in summaries:
        if int(row["depth"]) not in (0, args.horizon):
            continue
        print(
            f"{row['subset']} depth={row['depth']} "
            f"n={row['sample_count']} "
            f"rgb_auc={row['predicted_rgb_novelty_auc']:.3f} "
            f"novel={row['predicted_novelty_novel_mean']:.3f} "
            f"revisit={row['predicted_novelty_revisit_mean']:.3f}"
        )
    print(f"metrics: {args.output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
