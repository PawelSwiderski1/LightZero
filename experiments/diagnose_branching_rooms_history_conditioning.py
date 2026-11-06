#!/usr/bin/env python3
"""Measure how much history changes one-step spatial dynamics predictions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from experiments.evaluate_branching_rooms_spatial_online_checkpoint import (
    _json_safe,
    _model_from_state,
    _spatial_state,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-pairs-per-group", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser


def _digest(parts: Iterable[bytes]) -> bytes:
    value = hashlib.blake2b(digest_size=16)
    for part in parts:
        value.update(len(part).to_bytes(8, "little"))
        value.update(part)
    return value.digest()


def _image_bytes(image: np.ndarray) -> bytes:
    return np.ascontiguousarray(image, dtype=np.uint8).tobytes()


def _sample_cross_episode_pairs(
    occurrences: Sequence[int],
    episode_ids: np.ndarray,
    maximum: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    if maximum <= 0:
        return []
    if len(occurrences) <= 256:
        candidates = [
            (left, right)
            for left, right in itertools.combinations(occurrences, 2)
            if episode_ids[left] != episode_ids[right]
        ]
        if len(candidates) <= maximum:
            return candidates
        selected = rng.choice(len(candidates), size=maximum, replace=False)
        return [candidates[int(index)] for index in selected]

    by_episode: Dict[int, list[int]] = defaultdict(list)
    for occurrence in occurrences:
        by_episode[int(episode_ids[occurrence])].append(occurrence)
    episodes = sorted(by_episode)
    if len(episodes) < 2:
        return []
    pairs: set[tuple[int, int]] = set()
    attempts = 0
    while len(pairs) < maximum and attempts < maximum * 100:
        left_episode, right_episode = rng.choice(episodes, size=2, replace=False)
        left = int(rng.choice(by_episode[int(left_episode)]))
        right = int(rng.choice(by_episode[int(right_episode)]))
        pairs.add((min(left, right), max(left, right)))
        attempts += 1
    return sorted(pairs)


def _sample_ambiguous_pairs(
    by_next_observation: Mapping[bytes, Sequence[int]],
    episode_ids: np.ndarray,
    maximum: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    next_groups = [list(values) for values in by_next_observation.values()]
    if len(next_groups) < 2 or maximum <= 0:
        return []
    candidates = []
    if sum(len(group) for group in next_groups) <= 256:
        for left_group, right_group in itertools.combinations(next_groups, 2):
            candidates.extend(
                (left, right)
                for left in left_group
                for right in right_group
                if episode_ids[left] != episode_ids[right]
            )
        if len(candidates) <= maximum:
            return candidates
        selected = rng.choice(len(candidates), size=maximum, replace=False)
        return [candidates[int(index)] for index in selected]

    pairs: set[tuple[int, int]] = set()
    attempts = 0
    while len(pairs) < maximum and attempts < maximum * 100:
        left_group_index, right_group_index = rng.choice(
            len(next_groups), size=2, replace=False
        )
        left = int(rng.choice(next_groups[int(left_group_index)]))
        right = int(rng.choice(next_groups[int(right_group_index)]))
        if episode_ids[left] != episode_ids[right]:
            pairs.add((min(left, right), max(left, right)))
        attempts += 1
    return sorted(pairs)


def _cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return 0.0
    return float(1.0 - np.dot(left, right) / denominator)


def _mae(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.abs(left.astype(np.float32) - right.astype(np.float32)).mean())


def _movement_category(action: int, left_pose: np.ndarray, right_pose: np.ndarray) -> str:
    if action == 0:
        return "turn_left"
    if action == 1:
        return "turn_right"
    if action == 2:
        moved = bool(np.any(left_pose[:2] != right_pose[:2]))
        return "forward_move" if moved else "forward_blocked"
    return f"action_{action}"


def _predict_all(
    model: torch.nn.Module,
    observations: np.ndarray,
    actions: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    belief_rows = []
    predicted_latent_rows = []
    predicted_key_rows = []
    predicted_image_rows = []
    latent_target_errors = []
    key_target_errors = []
    image_target_errors = []

    model.eval()
    with torch.inference_mode():
        for episode_index in range(len(actions)):
            episode_observations = torch.as_tensor(
                observations[episode_index], dtype=torch.uint8, device=device
            ).unsqueeze(0)
            episode_actions = torch.as_tensor(
                actions[episode_index], dtype=torch.long, device=device
            ).unsqueeze(0)
            encoded = model.encode_trajectory(
                episode_observations, episode_actions
            )
            step_count = episode_actions.shape[1]
            for start in range(0, step_count, batch_size):
                end = min(step_count, start + batch_size)
                current_latents = encoded.latents[0, start:end]
                current_beliefs = encoded.beliefs[0, start:end]
                action_batch = episode_actions[0, start:end]
                next_latent, next_key, _, next_belief = model.predict_next(
                    current_latents, current_beliefs, action_batch
                )
                predicted_image = model.decode_beliefs(next_belief)
                target_latent = encoded.latents[0, start + 1:end + 1]
                target_key = encoded.observation_keys[0, start + 1:end + 1]
                target_image = model.reconstruction_targets(
                    episode_observations[0, start + 1:end + 1]
                )

                belief_rows.append(current_beliefs.cpu())
                predicted_latent_rows.append(next_latent.flatten(1).cpu())
                predicted_key_rows.append(next_key.cpu())
                predicted_image_rows.append(predicted_image.cpu())
                latent_target_errors.append(
                    (1.0 - F.cosine_similarity(
                        next_latent.flatten(1), target_latent.flatten(1), dim=-1
                    )).cpu()
                )
                key_target_errors.append(
                    (1.0 - F.cosine_similarity(next_key, target_key, dim=-1)).cpu()
                )
                image_target_errors.append(
                    torch.abs(predicted_image - target_image)
                    .flatten(1)
                    .mean(dim=-1)
                    .cpu()
                )

    return {
        "beliefs": torch.cat(belief_rows).numpy(),
        "predicted_latents": torch.cat(predicted_latent_rows).numpy(),
        "predicted_keys": torch.cat(predicted_key_rows).numpy(),
        "predicted_images": torch.cat(predicted_image_rows).numpy(),
        "latent_target_error": torch.cat(latent_target_errors).numpy(),
        "key_target_error": torch.cat(key_target_errors).numpy(),
        "image_target_error": torch.cat(image_target_errors).numpy(),
    }


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else math.nan


def _percentile(values: Sequence[float], percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values else math.nan


def _correlation(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) < 2 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return math.nan
    return float(np.corrcoef(left, right)[0, 1])


def _summarize(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    belief = [float(row["belief_cosine_distance"]) for row in rows]
    latent = [float(row["predicted_latent_cosine_distance"]) for row in rows]
    key = [float(row["predicted_key_cosine_distance"]) for row in rows]
    image = [float(row["predicted_image_mae"]) for row in rows]
    target_image = [float(row["actual_next_image_mae"]) for row in rows]
    own_error = [float(row["own_target_image_mae"]) for row in rows]
    swapped_error = [float(row["swapped_target_image_mae"]) for row in rows]
    latent_target_error = [
        0.5
        * (
            float(row["left_latent_target_cosine_distance"])
            + float(row["right_latent_target_cosine_distance"])
        )
        for row in rows
    ]
    key_target_error = [
        0.5
        * (
            float(row["left_key_target_cosine_distance"])
            + float(row["right_key_target_cosine_distance"])
        )
        for row in rows
    ]
    return {
        "pair_count": len(rows),
        "group_count": len({str(row["group_id"]) for row in rows}),
        "belief_cosine_distance_mean": _mean(belief),
        "belief_cosine_distance_p95": _percentile(belief, 95),
        "predicted_latent_cosine_distance_mean": _mean(latent),
        "predicted_latent_cosine_distance_p95": _percentile(latent, 95),
        "predicted_key_cosine_distance_mean": _mean(key),
        "predicted_key_cosine_distance_p95": _percentile(key, 95),
        "predicted_image_mae_mean": _mean(image),
        "predicted_image_mae_p95": _percentile(image, 95),
        "actual_next_image_mae_mean": _mean(target_image),
        "latent_target_cosine_distance_mean": _mean(latent_target_error),
        "latent_target_cosine_distance_p95": _percentile(
            latent_target_error, 95
        ),
        "key_target_cosine_distance_mean": _mean(key_target_error),
        "key_target_cosine_distance_p95": _percentile(key_target_error, 95),
        "own_target_image_mae_mean": _mean(own_error),
        "swapped_target_image_mae_mean": _mean(swapped_error),
        "own_better_than_swapped_fraction": _mean(
            [float(own < swapped) for own, swapped in zip(own_error, swapped_error)]
        ),
        "belief_to_latent_distance_correlation": _correlation(belief, latent),
        "belief_to_predicted_image_mae_correlation": _correlation(belief, image),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = build_parser().parse_args()
    if args.max_pairs_per_group <= 0:
        raise ValueError("--max-pairs-per-group must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    device = torch.device(args.device)
    checkpoint = torch.load(
        args.checkpoint, map_location=device, weights_only=False
    )
    spatial_state = _spatial_state(checkpoint)
    model = _model_from_state(spatial_state["model"], device)

    dataset = np.load(args.dataset, allow_pickle=False)
    observations = np.asarray(dataset["observations"], dtype=np.uint8)
    actions = np.asarray(dataset["actions"], dtype=np.int64)
    poses = np.asarray(dataset["poses"], dtype=np.int64)
    if observations.shape[:2] != (actions.shape[0], actions.shape[1] + 1):
        raise ValueError("Dataset observations/actions have inconsistent shapes")
    if poses.shape[:2] != observations.shape[:2]:
        raise ValueError("Dataset poses/observations have inconsistent shapes")

    episode_count, step_count = actions.shape
    flat_count = episode_count * step_count
    episode_ids = np.repeat(np.arange(episode_count), step_count)
    timesteps = np.tile(np.arange(step_count), episode_count)
    exact_groups: Dict[bytes, list[int]] = defaultdict(list)
    current_action_groups: Dict[bytes, Dict[bytes, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for episode_index in range(episode_count):
        for timestep in range(step_count):
            current = _image_bytes(observations[episode_index, timestep])
            next_observation = _image_bytes(
                observations[episode_index, timestep + 1]
            )
            action = int(actions[episode_index, timestep])
            current_action = _digest(
                [current, action.to_bytes(2, "little", signed=False)]
            )
            transition = _digest([current_action, next_observation])
            flat_index = episode_index * step_count + timestep
            exact_groups[transition].append(flat_index)
            current_action_groups[current_action][next_observation].append(
                flat_index
            )

    rng = np.random.default_rng(args.seed)
    pair_specs: list[tuple[str, bytes, int, int]] = []
    eligible_exact_groups = 0
    for group_id, occurrences in exact_groups.items():
        pairs = _sample_cross_episode_pairs(
            occurrences,
            episode_ids,
            args.max_pairs_per_group,
            rng,
        )
        if pairs:
            eligible_exact_groups += 1
            pair_specs.extend(
                ("same_true_transition", group_id, left, right)
                for left, right in pairs
            )

    eligible_ambiguous_groups = 0
    for group_id, by_next_observation in current_action_groups.items():
        pairs = _sample_ambiguous_pairs(
            by_next_observation,
            episode_ids,
            args.max_pairs_per_group,
            rng,
        )
        if pairs:
            eligible_ambiguous_groups += 1
            pair_specs.extend(
                ("different_true_next_view", group_id, left, right)
                for left, right in pairs
            )

    predictions = _predict_all(
        model, observations, actions, device, args.batch_size
    )
    if len(predictions["beliefs"]) != flat_count:
        raise RuntimeError("Prediction count does not match dataset transitions")

    flat_observations = observations[:, :-1].reshape(
        flat_count, *observations.shape[2:]
    )
    flat_next_observations = observations[:, 1:].reshape(
        flat_count, *observations.shape[2:]
    ).astype(np.float32) / 255.0
    flat_actions = actions.reshape(-1)
    flat_poses = poses[:, :-1].reshape(flat_count, poses.shape[-1])
    flat_next_poses = poses[:, 1:].reshape(flat_count, poses.shape[-1])

    rows = []
    for category, raw_group_id, left, right in pair_specs:
        left_target = flat_next_observations[left]
        right_target = flat_next_observations[right]
        left_prediction = predictions["predicted_images"][left]
        right_prediction = predictions["predicted_images"][right]
        own_target_error = 0.5 * (
            _mae(left_prediction, left_target)
            + _mae(right_prediction, right_target)
        )
        swapped_target_error = 0.5 * (
            _mae(left_prediction, right_target)
            + _mae(right_prediction, left_target)
        )
        action = int(flat_actions[left])
        if action != int(flat_actions[right]):
            raise RuntimeError("Grouped pair contains different actions")
        rows.append(
            {
                "category": category,
                "group_id": raw_group_id.hex(),
                "action": action,
                "movement_category_left": _movement_category(
                    action, flat_poses[left], flat_next_poses[left]
                ),
                "movement_category_right": _movement_category(
                    action, flat_poses[right], flat_next_poses[right]
                ),
                "left_episode": int(episode_ids[left]),
                "left_timestep": int(timesteps[left]),
                "right_episode": int(episode_ids[right]),
                "right_timestep": int(timesteps[right]),
                "same_current_rgb": int(
                    np.array_equal(flat_observations[left], flat_observations[right])
                ),
                "same_true_next_rgb": int(
                    np.array_equal(
                        flat_next_observations[left], flat_next_observations[right]
                    )
                ),
                "belief_cosine_distance": _cosine_distance(
                    predictions["beliefs"][left], predictions["beliefs"][right]
                ),
                "predicted_latent_cosine_distance": _cosine_distance(
                    predictions["predicted_latents"][left],
                    predictions["predicted_latents"][right],
                ),
                "predicted_key_cosine_distance": _cosine_distance(
                    predictions["predicted_keys"][left],
                    predictions["predicted_keys"][right],
                ),
                "predicted_image_mae": _mae(
                    left_prediction, right_prediction
                ),
                "actual_next_image_mae": _mae(left_target, right_target),
                "own_target_image_mae": own_target_error,
                "swapped_target_image_mae": swapped_target_error,
                "left_latent_target_cosine_distance": float(
                    predictions["latent_target_error"][left]
                ),
                "right_latent_target_cosine_distance": float(
                    predictions["latent_target_error"][right]
                ),
                "left_key_target_cosine_distance": float(
                    predictions["key_target_error"][left]
                ),
                "right_key_target_cosine_distance": float(
                    predictions["key_target_error"][right]
                ),
                "left_image_target_mae": float(
                    predictions["image_target_error"][left]
                ),
                "right_image_target_mae": float(
                    predictions["image_target_error"][right]
                ),
            }
        )

    summaries = []
    for category in ("same_true_transition", "different_true_next_view"):
        selected = [row for row in rows if row["category"] == category]
        if selected:
            summaries.append(
                {"category": category, "movement_category": "all", **_summarize(selected)}
            )
        movement_categories = sorted(
            {
                str(row["movement_category_left"])
                for row in selected
                if row["movement_category_left"] == row["movement_category_right"]
            }
        )
        for movement_category in movement_categories:
            movement_rows = [
                row
                for row in selected
                if row["movement_category_left"] == movement_category
                and row["movement_category_right"] == movement_category
            ]
            summaries.append(
                {
                    "category": category,
                    "movement_category": movement_category,
                    **_summarize(movement_rows),
                }
            )

    metadata = json.loads(str(dataset["metadata_json"]))
    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_online_update_count": int(
            spatial_state.get("online_update_count", -1)
        ),
        "checkpoint_online_collection_phase": int(
            spatial_state.get("online_collection_phase", -1)
        ),
        "dataset": str(args.dataset),
        "dataset_metadata": metadata,
        "episode_count": episode_count,
        "transition_count": flat_count,
        "eligible_same_transition_group_count": eligible_exact_groups,
        "eligible_ambiguous_current_action_group_count": eligible_ambiguous_groups,
        "pair_count": len(rows),
        "summaries": summaries,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "pairs.csv", rows)
    _write_csv(args.output_dir / "summary.csv", summaries)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(_json_safe(result), indent=2) + "\n", encoding="utf-8"
    )

    print(
        f"checkpoint updates={result['checkpoint_online_update_count']} "
        f"phase={result['checkpoint_online_collection_phase']}"
    )
    print(
        f"episodes={episode_count} transitions={flat_count} "
        f"same-transition groups={eligible_exact_groups} "
        f"ambiguous groups={eligible_ambiguous_groups}"
    )
    for summary in summaries:
        if summary["movement_category"] != "all":
            continue
        print(
            f"{summary['category']}: pairs={summary['pair_count']} "
            f"belief_distance={summary['belief_cosine_distance_mean']:.4f} "
            f"predicted_latent_distance="
            f"{summary['predicted_latent_cosine_distance_mean']:.4f} "
            f"predicted_image_mae={summary['predicted_image_mae_mean']:.4f} "
            f"actual_next_image_mae="
            f"{summary['actual_next_image_mae_mean']:.4f}"
        )
    print(f"metrics: {args.output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
