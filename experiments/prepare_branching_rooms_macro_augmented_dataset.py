#!/usr/bin/env python3
"""Add executed macro-policy trajectories to the Branching Rooms dataset.

The original uniform/persistent train, validation, and test episodes are kept
exactly.  Its old macro-policy held-out split is deliberately discarded.  New,
independently seeded macro episodes are then added to train, validation, and
test, and a larger fresh macro-only held-out split is generated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Mapping

import numpy as np

if __package__:
    from .run_branching_rooms_action_novelty import (
        POLICY_NAMES,
        SPLIT_NAMES,
        DatasetBundle,
        build_novelty_targets,
        collect_episode,
        load_dataset,
        state_ids_from_observations,
        state_ids_from_poses,
    )
else:
    from run_branching_rooms_action_novelty import (
        POLICY_NAMES,
        SPLIT_NAMES,
        DatasetBundle,
        build_novelty_targets,
        collect_episode,
        load_dataset,
        state_ids_from_observations,
        state_ids_from_poses,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dataset",
        type=Path,
        default=Path(
            ".local_runs/branching_rooms_action_novelty_seed0/"
            "experienced_trajectories.npz"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            ".local_runs/branching_rooms_action_novelty_macro_augmented_seed0"
        ),
    )
    parser.add_argument("--macro-seed-base", type=int, default=100_000_019)
    parser.add_argument("--train-macro-episodes", type=int, default=96)
    parser.add_argument("--validation-macro-episodes", type=int, default=24)
    parser.add_argument("--test-macro-episodes", type=int, default=24)
    parser.add_argument("--heldout-macro-episodes", type=int, default=96)
    parser.add_argument("--force", action="store_true")
    return parser


def _trajectory_digest(observations: np.ndarray, actions: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(observations).tobytes())
    digest.update(np.ascontiguousarray(actions).tobytes())
    return digest.hexdigest()


def dataset_split_diagnostics(bundle: DatasetBundle) -> Mapping[str, object]:
    split_rows = {}
    seed_sets = {}
    digest_sets = {}
    for split_id, split_name in enumerate(SPLIT_NAMES):
        indices = np.flatnonzero(bundle.split_ids == split_id)
        policy_counts = Counter(
            POLICY_NAMES[int(policy_id)] for policy_id in bundle.policy_ids[indices]
        )
        seeds = {int(seed) for seed in bundle.episode_seeds[indices]}
        digests = {
            _trajectory_digest(bundle.observations[index], bundle.actions[index])
            for index in indices
        }
        seed_sets[split_name] = seeds
        digest_sets[split_name] = digests
        split_rows[split_name] = {
            "episodes": int(indices.size),
            "policy_counts": dict(sorted(policy_counts.items())),
            "unique_seeds": len(seeds),
            "unique_trajectory_hashes": len(digests),
        }

    seed_overlap = {}
    trajectory_overlap = {}
    for left_index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[left_index + 1 :]:
            pair = f"{left}__{right}"
            seed_overlap[pair] = len(seed_sets[left] & seed_sets[right])
            trajectory_overlap[pair] = len(
                digest_sets[left] & digest_sets[right]
            )
    all_seeds = bundle.episode_seeds.astype(np.int64, copy=False)
    return {
        "splits": split_rows,
        "episode_count": int(all_seeds.size),
        "globally_unique_episode_seeds": bool(
            np.unique(all_seeds).size == all_seeds.size
        ),
        "cross_split_seed_overlap": seed_overlap,
        "cross_split_exact_trajectory_overlap": trajectory_overlap,
    }


def _save_dataset(bundle: DatasetBundle, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp.npz")
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
        metadata_json=np.asarray(json.dumps(bundle.metadata)),
    )
    temporary_path.replace(path)


def build_augmented_dataset(
    base: DatasetBundle,
    macro_episode_counts: tuple[int, int, int, int],
    macro_seed_base: int,
) -> DatasetBundle:
    if tuple(base.metadata["policy_names"]) != POLICY_NAMES:
        raise ValueError("Base dataset policy names do not match this experiment")
    if tuple(base.metadata["split_names"]) != SPLIT_NAMES:
        raise ValueError("Base dataset split names do not match this experiment")
    if any(count < 0 for count in macro_episode_counts):
        raise ValueError("Macro episode counts must be non-negative")

    retained = np.flatnonzero(base.split_ids < 3)
    observations = [base.observations[index] for index in retained]
    actions = [base.actions[index] for index in retained]
    poses = [base.poses[index] for index in retained]
    observation_targets = [base.observation_targets[index] for index in retained]
    pose_targets = [base.pose_targets[index] for index in retained]
    split_ids = [int(base.split_ids[index]) for index in retained]
    policy_ids = [int(base.policy_ids[index]) for index in retained]
    episode_seeds = [int(base.episode_seeds[index]) for index in retained]

    episode_length = int(base.metadata["episode_length"])
    tile_size = int(base.metadata["tile_size"])
    horizon = int(base.metadata["horizon"])
    macro_policy_id = POLICY_NAMES.index("macro")
    retained_seed_set = set(episode_seeds)

    for split_id, episode_count in enumerate(macro_episode_counts):
        for episode_index in range(int(episode_count)):
            episode_seed = (
                int(macro_seed_base)
                + split_id * 1_000_003
                + episode_index * 97
                + 17
            )
            if episode_seed in retained_seed_set:
                raise ValueError(f"Macro seed collides with base data: {episode_seed}")
            episode_observations, episode_actions, episode_poses = collect_episode(
                episode_length,
                tile_size,
                episode_seed,
                "macro",
            )
            observations.append(episode_observations)
            actions.append(episode_actions)
            poses.append(episode_poses)
            observation_targets.append(
                build_novelty_targets(
                    state_ids_from_observations(episode_observations), horizon
                )
            )
            pose_targets.append(
                build_novelty_targets(
                    state_ids_from_poses(episode_poses), horizon
                )
            )
            split_ids.append(split_id)
            policy_ids.append(macro_policy_id)
            episode_seeds.append(episode_seed)
        print(
            f"collected {episode_count} fresh macro episodes for "
            f"{SPLIT_NAMES[split_id]}",
            flush=True,
        )

    metadata = dict(base.metadata)
    metadata.update(
        {
            "dataset_variant": "legacy_nonmacro_plus_fresh_macro",
            "base_heldout_retained": False,
            "retained_base_splits": list(SPLIT_NAMES[:3]),
            "macro_seed_base": int(macro_seed_base),
            "fresh_macro_episode_counts": {
                split_name: int(count)
                for split_name, count in zip(SPLIT_NAMES, macro_episode_counts)
            },
            "heldout_definition": "fresh_independently_seeded_macro_only",
        }
    )
    return DatasetBundle(
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


def main() -> None:
    args = build_parser().parse_args()
    dataset_path = args.output_dir / "experienced_trajectories.npz"
    manifest_path = args.output_dir / "dataset_manifest.json"
    arguments_path = args.output_dir / "arguments.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if dataset_path.exists() and not args.force:
        bundle = load_dataset(dataset_path)
        print(f"using existing dataset: {dataset_path}", flush=True)
    else:
        if not args.base_dataset.is_file():
            raise FileNotFoundError(args.base_dataset)
        base = load_dataset(args.base_dataset)
        bundle = build_augmented_dataset(
            base,
            (
                int(args.train_macro_episodes),
                int(args.validation_macro_episodes),
                int(args.test_macro_episodes),
                int(args.heldout_macro_episodes),
            ),
            int(args.macro_seed_base),
        )
        _save_dataset(bundle, dataset_path)
        print(f"wrote dataset: {dataset_path}", flush=True)

    diagnostics = dataset_split_diagnostics(bundle)
    if not diagnostics["globally_unique_episode_seeds"]:
        raise RuntimeError("Dataset contains duplicate episode seeds")
    if any(diagnostics["cross_split_seed_overlap"].values()):
        raise RuntimeError("Dataset contains episode-seed overlap across splits")
    if any(diagnostics["cross_split_exact_trajectory_overlap"].values()):
        raise RuntimeError("Dataset contains exact trajectory overlap across splits")

    arguments = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    arguments_path.write_text(
        json.dumps(arguments, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "dataset": str(dataset_path),
        "metadata": bundle.metadata,
        **diagnostics,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(diagnostics["splits"], indent=2), flush=True)
    print(f"manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
