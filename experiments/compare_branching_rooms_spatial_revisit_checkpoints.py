#!/usr/bin/env python3
"""Compare spatial revisit checkpoints on identical recorded trajectories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from run_branching_rooms_action_novelty import load_dataset
from run_branching_rooms_exploration_world_model import ExplorationEpisodeDataset
from run_branching_rooms_spatial_exploration_world_model import (
    SpatialExplorationWorldModel,
    evaluate_model,
    resolve_device,
)


SPLITS = {"train": 0, "validation": 1, "test": 2, "heldout": 3}


def _parse_checkpoint(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError(
            "checkpoint must use LABEL=PATH syntax"
        )
    return label, Path(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=_parse_checkpoint,
        action="append",
        required=True,
        help="Checkpoint in LABEL=PATH form; pass once per model.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=tuple(SPLITS),
        action="append",
        help="Dataset split to evaluate. Defaults to test and heldout.",
    )
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--roots-per-episode",
        type=int,
        default=128,
        help="Evenly spaced roots per episode; 0 uses every eligible root.",
    )
    parser.add_argument("--maximum-episodes", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def _online_state(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    online = checkpoint.get("spatial_exploration_tree_online")
    if isinstance(online, Mapping) and isinstance(online.get("model"), Mapping):
        return online
    model = checkpoint.get("model")
    if isinstance(model, Mapping) and any(
        str(key).startswith("encoder.") for key in model
    ):
        return checkpoint
    raise KeyError("checkpoint does not contain an online spatial model")


def _model_spec(
    state: Mapping[str, torch.Tensor], observation_shape: Sequence[int]
) -> dict[str, Any]:
    if "encoder.3.weight" not in state or "key_projection.1.weight" not in state:
        raise KeyError("unrecognized SpatialExplorationWorldModel state dictionary")
    if any(key.startswith("revisit_pair_head.") for key in state):
        memory_key_mode = "belief_revisit"
    elif any(key.startswith("belief_key_projection.") for key in state):
        memory_key_mode = "belief"
    else:
        memory_key_mode = "observation"

    if any(key.startswith("decoder.") for key in state):
        representation_mode = "spatial_reconstruction"
        coarse_reconstruction_size = 7
    elif any(key.startswith("key_decoder.") for key in state):
        representation_mode = "compressed_key_reconstruction"
        output_dim = int(state["key_decoder.2.weight"].shape[0])
        channels = int(observation_shape[0])
        coarse_reconstruction_size = int(round((output_dim / channels) ** 0.5))
    else:
        representation_mode = "inverse_dynamics"
        coarse_reconstruction_size = 7

    return {
        "observation_shape": tuple(int(value) for value in observation_shape),
        "action_space_size": int(state["action_embedding.weight"].shape[0]) - 1,
        "spatial_channels": int(state["encoder.3.weight"].shape[0]),
        "key_dim": int(state["key_projection.1.weight"].shape[0]),
        "belief_dim": int(state["belief_cell.weight_hh"].shape[1]),
        "action_dim": int(state["action_embedding.weight"].shape[1]),
        "representation_mode": representation_mode,
        "coarse_reconstruction_size": coarse_reconstruction_size,
        "memory_key_mode": memory_key_mode,
    }


def _load_model(
    path: Path, observation_shape: Sequence[int], device: torch.device
) -> tuple[SpatialExplorationWorldModel, dict[str, Any]]:
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(loaded, Mapping):
        raise TypeError(f"Checkpoint is not a mapping: {path}")
    online = _online_state(loaded)
    state = online["model"]
    spec = _model_spec(state, observation_shape)
    model = SpatialExplorationWorldModel(**spec).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    metadata = {
        "path": str(path),
        "online_update_count": int(online.get("online_update_count", -1)),
        "online_collection_phase": int(
            online.get("online_collection_phase", -1)
        ),
        "last_iter": int(loaded.get("last_iter", -1)),
        "last_step": int(loaded.get("last_step", -1)),
        "source_checkpoint": loaded.get("source_checkpoint"),
        "model_spec": spec,
    }
    return model, metadata


def _metric_rows(label: str, split: str, metrics: Mapping[str, Any]) -> list[dict]:
    rows = []
    for depth, values in metrics["by_depth"].items():
        rows.append(
            {
                "model": label,
                "split": split,
                "depth": int(depth),
                "count": int(values["count"]),
                "rgb_novelty_auc": values["rgb_novelty_auc"],
                "position_novelty_auc": values["position_novelty_auc"],
                "teacher_novelty_correlation": values[
                    "novelty_prediction_correlation"
                ],
                "novelty_mae": values["novelty_prediction_mae"],
                "observation_key_cosine": values[
                    "observation_key_alignment_cosine_mean"
                ],
                "belief_key_cosine": values["alignment_cosine_mean"],
                "rgb_retrieval_top1": values["rgb_retrieval_top1"],
                "changed_rgb_retrieval_top1": values[
                    "changed_rgb_retrieval_top1"
                ],
                "rollout_reconstruction_mae": values[
                    "rollout_reconstruction"
                ]["mae"],
                "copy_root_mae": values["copy_root_baseline"]["mae"],
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: list[Any]) -> float | None:
    finite = [
        float(value)
        for value in values
        if value is not None and np.isfinite(value)
    ]
    return float(np.mean(finite)) if finite else None


def _format_metric(value: float | None, digits: int = 3) -> str:
    return "undefined" if value is None else f"{value:.{digits}f}"


def _delta_rows(rows: list[dict], labels: list[str]) -> list[dict]:
    if len(labels) < 2:
        return []
    baseline = labels[0]
    baseline_rows = {
        (row["split"], row["depth"]): row
        for row in rows
        if row["model"] == baseline
    }
    deltas = []
    identity = {"model", "split", "depth", "count"}
    for row in rows:
        if row["model"] == baseline:
            continue
        reference = baseline_rows[(row["split"], row["depth"])]
        delta = {
            "baseline": baseline,
            "comparison": row["model"],
            "split": row["split"],
            "depth": row["depth"],
            "count": row["count"],
        }
        for key, value in row.items():
            if key in identity:
                continue
            reference_value = reference[key]
            delta[f"delta_{key}"] = (
                None
                if value is None or reference_value is None
                else float(value) - float(reference_value)
            )
        deltas.append(delta)
    return deltas


def main() -> None:
    args = build_parser().parse_args()
    if args.horizon <= 0:
        raise ValueError("--horizon must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    labels = [label for label, _ in args.checkpoint]
    if len(set(labels)) != len(labels):
        raise ValueError("checkpoint labels must be unique")
    if not args.dataset.is_file():
        raise FileNotFoundError(args.dataset)
    for _, path in args.checkpoint:
        if not path.is_file():
            raise FileNotFoundError(path)

    device = resolve_device(args.device)
    bundle = load_dataset(args.dataset)
    if int(bundle.metadata["horizon"]) < args.horizon:
        raise ValueError("dataset horizon is shorter than --horizon")
    observation_shape = tuple(int(value) for value in bundle.observations.shape[-3:])
    split_names = args.split or ["test", "heldout"]
    maximum_episodes = args.maximum_episodes or None
    datasets = {
        name: ExplorationEpisodeDataset(bundle, SPLITS[name], maximum_episodes)
        for name in split_names
    }
    if any(len(dataset) == 0 for dataset in datasets.values()):
        raise ValueError("one or more requested dataset splits are empty")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {
        "dataset": str(args.dataset),
        "dataset_metadata": dict(bundle.metadata),
        "horizon": int(args.horizon),
        "roots_per_episode": int(args.roots_per_episode),
        "maximum_episodes": maximum_episodes,
        "device": str(device),
        "models": {},
    }
    rows = []
    for label, path in args.checkpoint:
        model, metadata = _load_model(path, observation_shape, device)
        if metadata["model_spec"]["memory_key_mode"] != "belief_revisit":
            raise ValueError(
                f"{label} is not a belief_revisit model: "
                f"{metadata['model_spec']['memory_key_mode']}"
            )
        model_results = {"metadata": metadata, "splits": {}}
        for split_name, dataset in datasets.items():
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
            )
            metrics = evaluate_model(
                model,
                loader,
                device,
                args.horizon,
                args.roots_per_episode,
            )
            model_results["splits"][split_name] = metrics
            split_rows = _metric_rows(label, split_name, metrics)
            rows.extend(split_rows)
            rgb_auc = _mean([row["rgb_novelty_auc"] for row in split_rows])
            position_auc = _mean(
                [row["position_novelty_auc"] for row in split_rows]
            )
            teacher_correlation = _mean(
                [row["teacher_novelty_correlation"] for row in split_rows]
            )
            novelty_mae = _mean([row["novelty_mae"] for row in split_rows])
            print(
                f"{label} {split_name}: "
                f"rgb_auc={_format_metric(rgb_auc)} "
                f"position_auc={_format_metric(position_auc)} "
                f"teacher_corr={_format_metric(teacher_correlation)} "
                f"novelty_mae={_format_metric(novelty_mae, 4)}",
                flush=True,
            )
        results["models"][label] = model_results
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    (args.output_dir / "comparison_metrics.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(args.output_dir / "comparison_by_depth.csv", rows)
    deltas = _delta_rows(rows, labels)
    if deltas:
        _write_csv(args.output_dir / "comparison_deltas.csv", deltas)
    print(f"metrics: {args.output_dir / 'comparison_metrics.json'}", flush=True)
    print(f"table: {args.output_dir / 'comparison_by_depth.csv'}", flush=True)
    if deltas:
        print(f"deltas: {args.output_dir / 'comparison_deltas.csv'}", flush=True)


if __name__ == "__main__":
    main()
