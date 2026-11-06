#!/usr/bin/env python3
"""Evaluate an online spatial-revisit checkpoint on independent trajectories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import torch

from experiments.run_branching_rooms_spatial_exploration_world_model import (
    SpatialExplorationWorldModel,
)
from lzero.entry.spatial_exploration_shadow import (
    SpatialExplorationShadowEvaluator,
)
from zoo.minigrid.envs.branching_rooms_env import (
    BranchingRoomsEnv,
    BranchingRoomsHeldoutEnv,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the frozen spatial exploration model saved inside a "
            "MuZero checkpoint on an independent trajectory dataset."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--split",
        default="heldout_policy",
        help="Named dataset split to evaluate, or 'all'.",
    )
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--roots-per-episode", type=int, default=128)
    parser.add_argument("--root-batch-size", type=int, default=128)
    parser.add_argument("--history-window", type=int, default=128)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--belief-ablation", action="store_true")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser


def _spatial_state(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    state = checkpoint.get("spatial_exploration_tree_online")
    if isinstance(state, Mapping) and "model" in state:
        return state
    if isinstance(checkpoint.get("model"), Mapping) and any(
        str(key).startswith("encoder.") for key in checkpoint["model"]
    ):
        return checkpoint
    raise KeyError(
        "Checkpoint does not contain an online spatial model"
    )


def _model_from_state(
        state_dict: Mapping[str, torch.Tensor], device: torch.device
) -> SpatialExplorationWorldModel:
    memory_key_mode = (
        "belief_revisit"
        if any(key.startswith("revisit_pair_head.") for key in state_dict)
        else "belief"
        if any(key.startswith("belief_key_projection.") for key in state_dict)
        else "observation"
    )
    representation_mode = (
        "spatial_reconstruction"
        if any(key.startswith("decoder.") for key in state_dict)
        else "compressed_key_reconstruction"
        if any(key.startswith("key_decoder.") for key in state_dict)
        else "inverse_dynamics"
    )
    spatial_channels = int(state_dict["encoder.3.weight"].shape[0])
    key_dim = int(state_dict["key_projection.1.weight"].shape[0])
    belief_dim = int(state_dict["belief_cell.weight_hh"].shape[1])
    action_dim = int(state_dict["action_embedding.weight"].shape[1])
    action_space_size = int(state_dict["inverse_action.2.weight"].shape[0])
    coarse_size = 7
    if representation_mode == "compressed_key_reconstruction":
        output_count = int(state_dict["key_decoder.2.weight"].shape[0])
        coarse_size = int(round(math.sqrt(output_count / 3)))

    model = SpatialExplorationWorldModel(
        observation_shape=(3, 28, 28),
        action_space_size=action_space_size,
        spatial_channels=spatial_channels,
        key_dim=key_dim,
        belief_dim=belief_dim,
        action_dim=action_dim,
        representation_mode=representation_mode,
        coarse_reconstruction_size=coarse_size,
        memory_key_mode=memory_key_mode,
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _load_episodes(
        dataset_path: Path,
        split_name: str,
        maximum_episode_count: int | None,
) -> tuple[list[Dict[str, Any]], Dict[str, Any], np.ndarray]:
    dataset = np.load(dataset_path, allow_pickle=False)
    metadata = json.loads(str(dataset["metadata_json"]))
    split_names = list(metadata.get("split_names", []))
    split_ids = np.asarray(dataset["split_ids"], dtype=np.int64)
    if split_name == "all":
        indices = np.arange(len(split_ids), dtype=np.int64)
    else:
        if split_name not in split_names:
            raise ValueError(
                f"Unknown split {split_name!r}; available splits: {split_names}"
            )
        indices = np.flatnonzero(split_ids == split_names.index(split_name))
    if maximum_episode_count is not None:
        if maximum_episode_count <= 0:
            raise ValueError("--max-episodes must be positive")
        indices = indices[:maximum_episode_count]
    if not len(indices):
        raise ValueError(f"Dataset split {split_name!r} contains no episodes")

    observations = dataset["observations"]
    actions = dataset["actions"]
    poses = dataset["poses"]
    policy_ids = np.asarray(dataset["policy_ids"], dtype=np.int64)
    episode_seeds = np.asarray(dataset["episode_seeds"], dtype=np.int64)
    episodes = []
    for evaluation_index, dataset_index in enumerate(indices.tolist()):
        episodes.append(
            {
                "observations": np.asarray(observations[dataset_index]),
                "actions": np.asarray(actions[dataset_index]),
                "environment_states": np.asarray(poses[dataset_index]),
                "env_id": int(policy_ids[dataset_index]),
                "env_episode_index": int(episode_seeds[dataset_index]),
                "collection_episode_index": int(dataset_index),
                "evaluation_episode_index": int(evaluation_index),
            }
        )
    return episodes, metadata, indices


def _digest(parts: Iterable[bytes]) -> bytes:
    value = hashlib.blake2b(digest_size=16)
    for part in parts:
        value.update(len(part).to_bytes(8, "little"))
        value.update(part)
    return value.digest()


def _observation_digest(observation: np.ndarray) -> bytes:
    return _digest([np.ascontiguousarray(observation, dtype=np.uint8).tobytes()])


def _transition_digest(
        observation: np.ndarray, action: int, next_observation: np.ndarray
) -> bytes:
    return _digest(
        [
            np.ascontiguousarray(observation, dtype=np.uint8).tobytes(),
            int(action).to_bytes(2, "little", signed=False),
            np.ascontiguousarray(next_observation, dtype=np.uint8).tobytes(),
        ]
    )


def _root_action_endpoint_digest(
        root_observation: np.ndarray,
        actions: np.ndarray,
        endpoint_observation: np.ndarray,
) -> bytes:
    return _digest(
        [
            np.ascontiguousarray(root_observation, dtype=np.uint8).tobytes(),
            np.ascontiguousarray(actions, dtype=np.uint8).tobytes(),
            np.ascontiguousarray(endpoint_observation, dtype=np.uint8).tobytes(),
        ]
    )


def _replay_exposure_sets(
        replay: Sequence[Mapping[str, Any]], horizon: int
) -> Dict[str, Any]:
    observations = set()
    transitions = set()
    root_action_endpoint = {depth: set() for depth in range(1, horizon + 1)}
    transition_count = 0
    for episode in replay:
        episode_observations = np.asarray(episode["observations"], dtype=np.uint8)
        episode_actions = np.asarray(episode["actions"], dtype=np.uint8)
        step_count = min(
            len(episode_actions), max(0, len(episode_observations) - 1)
        )
        for observation in episode_observations[:step_count + 1]:
            observations.add(_observation_digest(observation))
        for step in range(step_count):
            transitions.add(
                _transition_digest(
                    episode_observations[step],
                    int(episode_actions[step]),
                    episode_observations[step + 1],
                )
            )
            transition_count += 1
        for depth in range(1, horizon + 1):
            for root in range(max(0, step_count - depth + 1)):
                root_action_endpoint[depth].add(
                    _root_action_endpoint_digest(
                        episode_observations[root],
                        episode_actions[root:root + depth],
                        episode_observations[root + depth],
                    )
                )
    return {
        "observations": observations,
        "transitions": transitions,
        "root_action_endpoint": root_action_endpoint,
        "replay_episode_count": len(replay),
        "replay_transition_count": transition_count,
    }


def _annotate_replay_exposure(
        rows: Sequence[Dict[str, Any]],
        episodes: Sequence[Mapping[str, Any]],
        exposure: Mapping[str, Any],
) -> None:
    for row in rows:
        episode = episodes[int(row["episode_index_in_phase"])]
        observations = np.asarray(episode["observations"], dtype=np.uint8)
        actions = np.asarray(episode["actions"], dtype=np.uint8)
        root = int(row["root_timestep"])
        endpoint = int(row["endpoint_timestep"])
        depth = int(row["depth"])
        row["endpoint_seen_in_final_replay"] = int(
            _observation_digest(observations[endpoint])
            in exposure["observations"]
        )
        row["incoming_transition_seen_in_final_replay"] = int(
            _transition_digest(
                observations[endpoint - 1],
                int(actions[endpoint - 1]),
                observations[endpoint],
            )
            in exposure["transitions"]
        )
        row["root_action_endpoint_seen_in_final_replay"] = int(
            _root_action_endpoint_digest(
                observations[root],
                actions[root:endpoint],
                observations[endpoint],
            )
            in exposure["root_action_endpoint"][depth]
        )


def _annotate_room_crossings(
    rows: Sequence[Dict[str, Any]],
    episodes: Sequence[Mapping[str, Any]],
) -> None:
    training_edges = {
        BranchingRoomsEnv._canonical_edge(*edge)
        for edge in BranchingRoomsEnv.room_edges
    }
    heldout_edges = {
        BranchingRoomsHeldoutEnv._canonical_edge(*edge)
        for edge in BranchingRoomsHeldoutEnv.room_edges
    }
    heldout_only_edges = heldout_edges - training_edges
    for row in rows:
        episode = episodes[int(row["episode_index_in_phase"])]
        poses = np.asarray(episode["environment_states"], dtype=np.int64)
        root = int(row["root_timestep"])
        endpoint = int(row["endpoint_timestep"])
        rooms = []
        for pose in poses[root:endpoint + 1]:
            room = BranchingRoomsEnv.room_for_position(
                (int(pose[0]), int(pose[1]))
            )
            if room is not None and (not rooms or room != rooms[-1]):
                rooms.append(room)
        crossed_edges = {
            BranchingRoomsEnv._canonical_edge(left, right)
            for left, right in zip(rooms, rooms[1:])
        }
        if crossed_edges.intersection(heldout_only_edges):
            category = "heldout_only_edge"
        elif crossed_edges:
            category = "shared_edge_only"
        else:
            category = "no_room_crossing"
        row["room_crossing_category"] = category


def _summaries_by_exposure(
        evaluator: SpatialExplorationShadowEvaluator,
        rows: Sequence[Mapping[str, Any]],
        root_rows: Sequence[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    output = []
    for summary in evaluator.summarize(rows, 0, root_rows):
        output.append({"exposure": "all", **summary})
    for field in (
        "endpoint_seen_in_final_replay",
        "incoming_transition_seen_in_final_replay",
        "root_action_endpoint_seen_in_final_replay",
    ):
        for value, suffix in ((1, "seen"), (0, "unseen")):
            selected = [row for row in rows if int(row[field]) == value]
            if not selected:
                continue
            for summary in evaluator.summarize(selected, 0):
                output.append(
                    {"exposure": f"{field}:{suffix}", **summary}
                )
    categories = sorted({str(row["room_crossing_category"]) for row in rows})
    for category in categories:
        selected = [
            row for row in rows
            if str(row["room_crossing_category"]) == category
        ]
        for summary in evaluator.summarize(selected, 0):
            output.append(
                {"exposure": f"room_crossing:{category}", **summary}
            )
    return output


def _episode_summaries(
        evaluator: SpatialExplorationShadowEvaluator,
        rows: Sequence[Mapping[str, Any]],
        root_rows: Sequence[Mapping[str, Any]],
        dataset_indices: np.ndarray,
) -> list[Dict[str, Any]]:
    output = []
    episode_ids = sorted({int(row["episode_index_in_phase"]) for row in rows})
    for episode_id in episode_ids:
        selected = [
            row for row in rows
            if int(row["episode_index_in_phase"]) == episode_id
        ]
        selected_roots = [
            row for row in root_rows
            if int(row["episode_index_in_phase"]) == episode_id
        ]
        for summary in evaluator.summarize(selected, 0, selected_roots):
            output.append(
                {
                    "evaluation_episode_index": episode_id,
                    "dataset_episode_index": int(dataset_indices[episode_id]),
                    **summary,
                }
            )
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(
        args.checkpoint, map_location=device, weights_only=False
    )
    spatial_state = _spatial_state(checkpoint)
    model = _model_from_state(spatial_state["model"], device)
    episodes, dataset_metadata, dataset_indices = _load_episodes(
        args.dataset, args.split, args.max_episodes
    )
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
        belief_ablation=args.belief_ablation,
    )
    rows, root_rows = evaluator.evaluate_episodes(episodes, collection_phase=0)
    exposure = _replay_exposure_sets(
        spatial_state.get("replay", []), args.horizon
    )
    _annotate_replay_exposure(rows, episodes, exposure)
    _annotate_room_crossings(rows, episodes)
    summaries = _summaries_by_exposure(evaluator, rows, root_rows)
    episode_summaries = _episode_summaries(
        evaluator, rows, root_rows, dataset_indices
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "steps.csv", rows)
    _write_csv(args.output_dir / "roots.csv", root_rows)
    _write_csv(args.output_dir / "summary.csv", summaries)
    _write_csv(args.output_dir / "episode_summary.csv", episode_summaries)
    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_last_iter": int(checkpoint.get("last_iter", -1)),
        "online_update_count": int(spatial_state.get("online_update_count", -1)),
        "online_collection_phase": int(
            spatial_state.get("online_collection_phase", -1)
        ),
        "dataset": str(args.dataset),
        "dataset_metadata": dataset_metadata,
        "split": args.split,
        "dataset_episode_indices": dataset_indices.tolist(),
        "episode_count": len(episodes),
        "root_count": len(root_rows),
        "prediction_count": len(rows),
        "final_replay_episode_count": exposure["replay_episode_count"],
        "final_replay_transition_count": exposure["replay_transition_count"],
        "summary": summaries,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(_json_safe(result), indent=2) + "\n", encoding="utf-8"
    )

    print(
        f"checkpoint updates={result['online_update_count']} "
        f"phase={result['online_collection_phase']}"
    )
    print(
        f"evaluated episodes={len(episodes)} roots={len(root_rows)} "
        f"predictions={len(rows)} split={args.split}"
    )
    for row in summaries:
        if int(row["depth"]) not in (0, args.horizon):
            continue
        print(
            f"{row['exposure']} depth={row['depth']} n={row['sample_count']} "
            f"rgb_auc={row['predicted_rgb_novelty_auc']:.3f} "
            f"novel={row['predicted_novelty_novel_mean']:.3f} "
            f"revisit={row['predicted_novelty_revisit_mean']:.3f}"
        )
        if "real_query_real_memory_rgb_auc" in row:
            print(
                "  belief ablation: "
                f"real/real={row['real_query_real_memory_rgb_auc']:.3f} "
                f"predicted/real={row['predicted_query_real_memory_rgb_auc']:.3f} "
                f"real/predicted={row['real_query_predicted_memory_rgb_auc']:.3f}"
            )
    print(f"metrics: {args.output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
