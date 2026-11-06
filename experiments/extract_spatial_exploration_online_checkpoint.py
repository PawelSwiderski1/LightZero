#!/usr/bin/env python3
"""Extract the latest online spatial model from a LightZero policy checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


def _spatial_state(checkpoint: Mapping[str, Any]) -> Mapping[str, Any] | None:
    state = checkpoint.get("spatial_exploration_tree_online")
    if isinstance(state, Mapping) and isinstance(state.get("model"), Mapping):
        return state
    model = checkpoint.get("model")
    if isinstance(model, Mapping) and any(
        str(key).startswith("encoder.") for key in model
    ):
        return checkpoint
    return None


def _checkpoint_rank(
    path: Path, checkpoint: Mapping[str, Any], state: Mapping[str, Any]
) -> tuple[int, int, int, int]:
    return (
        int(state.get("online_update_count", -1)),
        int(checkpoint.get("last_step", -1)),
        int(checkpoint.get("last_iter", -1)),
        int(path.stat().st_mtime_ns),
    )


def select_checkpoint(
    checkpoint: Path | None, checkpoint_dir: Path | None
) -> tuple[Path, Mapping[str, Any], Mapping[str, Any]]:
    if (checkpoint is None) == (checkpoint_dir is None):
        raise ValueError("Specify exactly one of --checkpoint and --checkpoint-dir")
    candidates = [checkpoint] if checkpoint is not None else sorted(
        checkpoint_dir.glob("*.pth.tar")
    )
    if not candidates:
        raise FileNotFoundError(f"No *.pth.tar files found in {checkpoint_dir}")

    best = None
    failures = []
    for path in candidates:
        try:
            loaded = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(loaded, Mapping):
                raise TypeError("checkpoint is not a mapping")
            state = _spatial_state(loaded)
            if state is None:
                raise KeyError("spatial_exploration_tree_online is absent")
            candidate = (
                _checkpoint_rank(path, loaded, state),
                path,
                loaded,
                state,
            )
            if best is None or candidate[0] > best[0]:
                best = candidate
        except Exception as error:  # Keep scanning periodic checkpoints.
            failures.append(f"{path.name}: {error}")
    if best is None:
        detail = "; ".join(failures)
        raise RuntimeError(f"No usable spatial checkpoint found. {detail}")
    _, path, loaded, state = best
    return path, loaded, state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--include-replay",
        action="store_true",
        help=(
            "Also retain the final online replay episodes and root-priority "
            "cache for exact training-data audits."
        ),
    )
    return parser


def _copy_replay_episode(episode: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "observations": np.asarray(episode["observations"], dtype=np.uint8),
        "actions": np.asarray(episode["actions"], dtype=np.int64),
        "replay_id": int(episode["replay_id"]),
    }


def main() -> None:
    args = build_parser().parse_args()
    source, checkpoint, state = select_checkpoint(
        args.checkpoint, args.checkpoint_dir
    )
    model_state = {
        key: value.detach().cpu() if torch.is_tensor(value) else value
        for key, value in state["model"].items()
    }
    output = {
        "format_version": 1,
        "source_checkpoint": str(source),
        "last_iter": int(checkpoint.get("last_iter", -1)),
        "last_step": int(checkpoint.get("last_step", -1)),
        "online_update_count": int(state.get("online_update_count", -1)),
        "online_collection_phase": int(
            state.get("online_collection_phase", -1)
        ),
        "model": model_state,
    }
    if args.include_replay:
        output["replay"] = [
            _copy_replay_episode(episode)
            for episode in state.get("replay", [])
        ]
        output["online_root_priorities"] = {
            tuple(int(value) for value in key): float(priority)
            for key, priority in state.get(
                "online_root_priorities", {}
            ).items()
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(
        f"extracted {source} -> {args.output} "
        f"(online_updates={output['online_update_count']}, "
        f"collection_phase={output['online_collection_phase']}, "
        f"replay_episodes={len(output.get('replay', []))}, "
        f"priority_entries={len(output.get('online_root_priorities', {}))})",
        flush=True,
    )


if __name__ == "__main__":
    main()
