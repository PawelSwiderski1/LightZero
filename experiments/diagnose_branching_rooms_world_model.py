#!/usr/bin/env python3
"""Measure whether Branching Rooms MuZero dynamics predict real next states.

For uniformly sampled physical roots and action sequences, this diagnostic
recursively rolls each ensemble member's dynamics and compares the resulting
imagined latent with the same member's direct encoding of the exact physical
endpoint.  It reports alignment and sampled retrieval in both raw latent space
and the projected SSL space used by the default training objective.

Retrieval is scored by exact RGB-observation group rather than pose alone.
This matters because multiple physical poses can produce identical partial
observations and therefore necessarily have identical direct encodings.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from lzero.model.ensemble_muzero_model import EnsembleMuZeroModel

if __package__:
    from .diagnose_branching_rooms_raw_latent_geometry import (
        BranchingRoomsStateSpace,
        _load_config,
        _load_model,
        _resolve_checkpoint,
        _resolve_device,
        build_state_space,
        encode_all_poses,
    )
else:
    from diagnose_branching_rooms_raw_latent_geometry import (
        BranchingRoomsStateSpace,
        _load_config,
        _load_model,
        _resolve_checkpoint,
        _resolve_device,
        build_state_space,
        encode_all_poses,
    )


DISTANCE_METRICS = ("squared_l2", "cosine")


def _distribution(values: np.ndarray) -> Dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p05": float(np.quantile(values, 0.05)),
        "p95": float(np.quantile(values, 0.95)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def _sample_paths(
    state_space: BranchingRoomsStateSpace,
    depth: int,
    count: int,
    rng: np.random.Generator,
    root_pool: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if root_pool is None:
        roots = rng.integers(0, len(state_space.poses), size=count, dtype=np.int64)
    else:
        root_pool = np.asarray(root_pool, dtype=np.int64)
        if root_pool.size == 0:
            raise ValueError("Cannot sample paths from an empty root pool")
        roots = rng.choice(root_pool, size=count, replace=True)
    actions = rng.integers(
        0, state_space.transition.shape[1], size=(count, depth), dtype=np.int64
    )
    endpoints = roots.copy()
    for step in range(depth):
        endpoints = state_space.transition[endpoints, actions[:, step]]
    return roots, actions, endpoints


def _visited_pose_indices(
    run_dir: Path,
    state_space: BranchingRoomsStateSpace,
) -> np.ndarray:
    path = run_dir / "diagnostics" / "environment_coverage_cells.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Root-scope filtering requires environment coverage cells: {path}"
        )
    pose_to_index = {
        tuple(int(value) for value in pose): index
        for index, pose in enumerate(state_space.poses)
    }
    visited = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            x, y = int(row["x"]), int(row["y"])
            for direction in range(4):
                count = int(float(row[f"cumulative_heading_{direction}_visit_count"]))
                if count > 0:
                    pose_index = pose_to_index.get((x, y, direction))
                    if pose_index is not None:
                        visited.add(pose_index)
    return np.asarray(sorted(visited), dtype=np.int64)


def _rollout_paths(
    model: EnsembleMuZeroModel,
    direct_latents: torch.Tensor,
    latent_shape: Tuple[int, ...],
    roots: np.ndarray,
    actions: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """Return imagined raw latents [path, member, flattened_feature] on CPU."""
    ensemble_size = int(direct_latents.shape[1])
    path_count = len(roots)
    current = direct_latents[torch.as_tensor(roots, dtype=torch.long)].reshape(
        path_count, ensemble_size, *latent_shape
    ).to(device)
    member_ids = torch.arange(ensemble_size, device=device).repeat(path_count)
    with torch.inference_mode():
        for step in range(actions.shape[1]):
            step_actions = torch.as_tensor(
                actions[:, step], dtype=torch.long, device=device
            )[:, None].expand(path_count, ensemble_size).reshape(-1)
            output = model.recurrent_inference(
                current.reshape(-1, *latent_shape),
                step_actions,
                member_id=member_ids,
            )
            current = output.latent_state.reshape(
                path_count, ensemble_size, *latent_shape
            )
    return current.detach().float().flatten(2).cpu()


def _project_latents(
    model: EnsembleMuZeroModel,
    latents: torch.Tensor,
    latent_shape: Tuple[int, ...],
    device: torch.device,
    with_grad_path: bool,
    batch_size: int,
) -> torch.Tensor:
    """Project [sample, member, flattened_feature] and preserve both axes."""
    sample_count, ensemble_size = latents.shape[:2]
    flattened = latents.reshape(sample_count * ensemble_size, *latent_shape)
    member_ids = torch.arange(ensemble_size).repeat(sample_count)
    parts = []
    with torch.inference_mode():
        for start in range(0, len(flattened), batch_size):
            end = min(start + batch_size, len(flattened))
            part = model.project(
                flattened[start:end].to(device),
                with_grad=with_grad_path,
                member_id=member_ids[start:end].to(device),
            )
            parts.append(part.detach().float().cpu())
    projected = torch.cat(parts, dim=0)
    return projected.reshape(sample_count, ensemble_size, -1)


def _pair_distance(
    first: torch.Tensor,
    second: torch.Tensor,
    metric: str,
) -> torch.Tensor:
    if metric == "squared_l2":
        return (first - second).pow(2).mean(dim=-1)
    if metric == "cosine":
        return 1.0 - F.cosine_similarity(first, second, dim=-1)
    raise ValueError(metric)


def _candidate_distances(
    queries: torch.Tensor,
    candidates: torch.Tensor,
    metric: str,
) -> torch.Tensor:
    if metric == "squared_l2":
        return torch.cdist(queries.float(), candidates.float(), p=2).pow(2) / max(
            int(queries.shape[-1]), 1
        )
    if metric == "cosine":
        return 1.0 - F.normalize(queries.float(), dim=-1) @ F.normalize(
            candidates.float(), dim=-1
        ).T
    raise ValueError(metric)


def _retrieval_summary(
    queries: torch.Tensor,
    targets: torch.Tensor,
    candidates: torch.Tensor,
    target_indices: np.ndarray,
    candidate_indices: np.ndarray,
    state_space: BranchingRoomsStateSpace,
    metric: str,
    member: int,
    tie_tolerance: float = 1e-7,
) -> Dict[str, object]:
    """Score retrieval of the true observation against fixed real distractors."""
    target_distance = _pair_distance(queries, targets, metric)
    distances = _candidate_distances(queries, candidates, metric)
    target_groups = state_space.observation_group[target_indices]
    candidate_groups = state_space.observation_group[candidate_indices]
    same_group = torch.as_tensor(
        target_groups[:, None] == candidate_groups[None, :], dtype=torch.bool
    )
    wrong_distances = distances.masked_fill(same_group, float("inf"))
    best_wrong_distance, best_wrong_column = wrong_distances.min(dim=1)
    closer_wrong = wrong_distances < (target_distance[:, None] - tie_tolerance)
    rank = closer_wrong.sum(dim=1) + 1
    strict_top1 = target_distance + tie_tolerance < best_wrong_distance
    tie_aware_top1 = target_distance <= best_wrong_distance + tie_tolerance

    best_wrong_indices = candidate_indices[best_wrong_column.cpu().numpy()]
    retrieved_indices = np.where(
        tie_aware_top1.cpu().numpy(), target_indices, best_wrong_indices
    )
    same_position = np.all(
        state_space.poses[retrieved_indices, :2]
        == state_space.poses[target_indices, :2],
        axis=1,
    )
    same_room = np.all(
        state_space.rooms[retrieved_indices] == state_space.rooms[target_indices],
        axis=1,
    )
    target_distance_np = target_distance.cpu().numpy()
    margin_np = (best_wrong_distance - target_distance).cpu().numpy()
    rank_np = rank.cpu().numpy()
    return {
        "member": int(member),
        "sample_count": int(len(queries)),
        "candidate_count": int(len(candidate_indices)),
        "chance_top1": float(1.0 / (len(candidate_indices) + 1)),
        "target_distance": _distribution(target_distance_np),
        "correct_over_best_wrong_margin": _distribution(margin_np),
        "sampled_rank": _distribution(rank_np),
        "top1_strict": float(strict_top1.float().mean().item()),
        "top1_tie_aware": float(tie_aware_top1.float().mean().item()),
        "top5": float((rank <= 5).float().mean().item()),
        "mean_reciprocal_rank": float((1.0 / rank.float()).mean().item()),
        "retrieved_same_position": float(np.mean(same_position)),
        "retrieved_same_room": float(np.mean(same_room)),
    }


def evaluate_representation_space(
    predicted: torch.Tensor,
    targets: torch.Tensor,
    candidates: torch.Tensor,
    target_indices: np.ndarray,
    candidate_indices: np.ndarray,
    state_space: BranchingRoomsStateSpace,
    space: str,
    metrics: Sequence[str],
) -> List[Dict[str, object]]:
    """Return per-member and ensemble-pooled alignment/retrieval records."""
    records: List[Dict[str, object]] = []
    ensemble_size = int(predicted.shape[1])
    for metric in metrics:
        member_records = []
        for member in range(ensemble_size):
            record = _retrieval_summary(
                predicted[:, member],
                targets[:, member],
                candidates[:, member],
                target_indices,
                candidate_indices,
                state_space,
                metric,
                member,
            )
            alignment_mse = (predicted[:, member] - targets[:, member]).pow(2).mean(dim=-1)
            alignment_cosine = F.cosine_similarity(
                predicted[:, member], targets[:, member], dim=-1
            )
            record.update(
                space=space,
                distance_metric=metric,
                alignment_mse=_distribution(alignment_mse.cpu().numpy()),
                alignment_cosine=_distribution(alignment_cosine.cpu().numpy()),
            )
            records.append(record)
            member_records.append(record)

        pooled_predicted = predicted.reshape(predicted.shape[0], -1)
        pooled_targets = targets.reshape(targets.shape[0], -1)
        pooled_candidates = candidates.reshape(candidates.shape[0], -1)
        pooled = _retrieval_summary(
            pooled_predicted,
            pooled_targets,
            pooled_candidates,
            target_indices,
            candidate_indices,
            state_space,
            metric,
            -1,
        )
        pooled_mse = (pooled_predicted - pooled_targets).pow(2).mean(dim=-1)
        pooled_cosine = F.cosine_similarity(
            pooled_predicted, pooled_targets, dim=-1
        )
        pooled.update(
            space=space,
            distance_metric=metric,
            alignment_mse=_distribution(pooled_mse.cpu().numpy()),
            alignment_cosine=_distribution(pooled_cosine.cpu().numpy()),
        )
        records.append(pooled)
    return records


def _flatten_record(record: Dict[str, object]) -> Dict[str, object]:
    flattened: Dict[str, object] = {}
    for key, value in record.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                flattened[f"{key}_{subkey}"] = subvalue
        else:
            flattened[key] = value
    return flattened


def _write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = [_flatten_record(row) for row in rows]
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--samples-per-depth", type=int, default=512)
    parser.add_argument("--candidate-count", type=int, default=256)
    parser.add_argument("--encode-batch-size", type=int, default=128)
    parser.add_argument("--projection-batch-size", type=int, default=256)
    parser.add_argument(
        "--root-scope",
        choices=("all", "visited", "unvisited"),
        default="all",
        help="Restrict sampled roots using the run's cumulative coverage-cell log.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-prefix", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.horizon <= 0 or args.samples_per_depth <= 0:
        raise ValueError("--horizon and --samples-per-depth must be positive")
    run_dir = args.run_dir.resolve()
    checkpoint_path = _resolve_checkpoint(run_dir, args.checkpoint)
    config = _load_config(run_dir / "total_config.py")
    device = _resolve_device(args.device)
    rng = np.random.default_rng(args.seed)
    env_config = config["env"]
    state_space = build_state_space(
        agent_view_size=int(env_config.get("agent_view_size", 7)),
        tile_size=int(env_config.get("rgb_tile_size", 12)),
    )
    model, checkpoint = _load_model(config, checkpoint_path, device)
    ensemble_size = int(config["policy"]["model"]["ensemble_size"])
    visited_indices = _visited_pose_indices(run_dir, state_space)
    if args.root_scope == "visited":
        root_pool = visited_indices
    elif args.root_scope == "unvisited":
        root_pool = np.setdiff1d(
            np.arange(len(state_space.poses), dtype=np.int64),
            visited_indices,
            assume_unique=True,
        )
    else:
        root_pool = None

    print(f"run: {run_dir}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"device: {device}")
    print(
        f"root scope: {args.root_scope} "
        f"({len(state_space.poses) if root_pool is None else len(root_pool)} poses; "
        f"{len(visited_indices)} visited in run)"
    )
    print(f"encoding all {len(state_space.poses)} physical poses...")
    direct_latents, latent_shape = encode_all_poses(
        model,
        state_space.observations,
        ensemble_size,
        device,
        max(1, args.encode_batch_size),
    )

    _, unique_representatives = np.unique(
        state_space.observation_group, return_index=True
    )
    candidate_count = min(int(args.candidate_count), len(unique_representatives))
    candidate_indices = np.sort(
        rng.choice(unique_representatives, size=candidate_count, replace=False)
    )
    raw_candidates = direct_latents[
        torch.as_tensor(candidate_indices, dtype=torch.long)
    ]
    projected_candidates = _project_latents(
        model,
        raw_candidates,
        latent_shape,
        device,
        with_grad_path=False,
        batch_size=max(1, args.projection_batch_size),
    )

    records: List[Dict[str, object]] = []
    path_statistics: List[Dict[str, object]] = []
    for depth in range(1, int(args.horizon) + 1):
        print(f"depth {depth}: rolling {args.samples_per_depth} exact action paths...")
        roots, actions, endpoints = _sample_paths(
            state_space,
            depth,
            int(args.samples_per_depth),
            rng,
            root_pool=root_pool,
        )
        predicted_raw = _rollout_paths(
            model, direct_latents, latent_shape, roots, actions, device
        )
        target_raw = direct_latents[torch.as_tensor(endpoints, dtype=torch.long)]
        predicted_projected = _project_latents(
            model,
            predicted_raw,
            latent_shape,
            device,
            with_grad_path=True,
            batch_size=max(1, args.projection_batch_size),
        )
        target_projected = _project_latents(
            model,
            target_raw,
            latent_shape,
            device,
            with_grad_path=False,
            batch_size=max(1, args.projection_batch_size),
        )
        changed_pose = roots != endpoints
        changed_rgb = (
            state_space.observation_group[roots]
            != state_space.observation_group[endpoints]
        )
        path_statistics.append(
            {
                "depth": depth,
                "sample_count": int(len(roots)),
                "changed_pose_fraction": float(changed_pose.mean()),
                "changed_rgb_fraction": float(changed_rgb.mean()),
            }
        )
        for subset, mask in (
            ("all", np.ones(len(roots), dtype=bool)),
            ("changed_rgb", changed_rgb),
        ):
            if not mask.any():
                continue
            raw_records = evaluate_representation_space(
                predicted_raw[mask],
                target_raw[mask],
                raw_candidates,
                endpoints[mask],
                candidate_indices,
                state_space,
                "raw",
                DISTANCE_METRICS,
            )
            projected_records = evaluate_representation_space(
                predicted_projected[mask],
                target_projected[mask],
                projected_candidates,
                endpoints[mask],
                candidate_indices,
                state_space,
                "projected_ssl",
                ("cosine",),
            )
            for record in raw_records + projected_records:
                record["depth"] = depth
                record["subset"] = subset
                records.append(record)

    output_prefix = args.output_prefix
    if output_prefix is None:
        checkpoint_name = checkpoint_path.name.replace(".pth.tar", "")
        scope_suffix = "" if args.root_scope == "all" else f"_{args.root_scope}_roots"
        output_prefix = run_dir / "diagnostics" / (
            f"world_model_{checkpoint_name}{scope_suffix}"
        )
    output_prefix = output_prefix.resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    pooled_records = [record for record in records if int(record["member"]) == -1]
    summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_train_iter": int(checkpoint.get("last_iter", -1)),
        "checkpoint_env_step": int(checkpoint.get("last_step", -1)),
        "device": str(device),
        "reachable_pose_count": int(len(state_space.poses)),
        "unique_observation_count": int(len(unique_representatives)),
        "ensemble_size": ensemble_size,
        "latent_shape": list(latent_shape),
        "samples_per_depth": int(args.samples_per_depth),
        "candidate_count": candidate_count,
        "root_scope": args.root_scope,
        "root_pool_size": int(
            len(state_space.poses) if root_pool is None else len(root_pool)
        ),
        "visited_pose_count": int(len(visited_indices)),
        "seed": int(args.seed),
        "path_statistics": path_statistics,
        "pooled_results": pooled_records,
        "member_results": [record for record in records if int(record["member"]) >= 0],
    }
    json_path = output_prefix.with_suffix(".json")
    csv_path = output_prefix.with_suffix(".csv")
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_csv(csv_path, records)

    print("\nPooled retrieval against real encoded observations")
    for record in pooled_records:
        if record["subset"] != "changed_rgb":
            continue
        print(
            f"  depth {record['depth']} changed-RGB "
            f"{record['space']}/{record['distance_metric']}: "
            f"cos={record['alignment_cosine']['mean']:.3f}, "
            f"top1={record['top1_tie_aware']:.3f}, "
            f"top5={record['top5']:.3f}, "
            f"same-room={record['retrieved_same_room']:.3f}"
        )
    print(f"\nsummary: {json_path}")
    print(f"rows: {csv_path}")


if __name__ == "__main__":
    main()
