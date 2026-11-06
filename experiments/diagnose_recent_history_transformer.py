import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym

from lzero.policy.recurrent_sequence_novelty import (
    RecentHistoryTransformerNovelty,
    recent_observation_novelty_targets,
)
from zoo.minigrid.envs.branching_rooms_env import BRANCHING_ROOMS_ENV_ID


ACTION_NAMES = {0: "left", 1: "right", 2: "forward"}
REVISIT_AGE_BINS = (
    ("1", 1, 1),
    ("2_5", 2, 5),
    ("6_10", 6, 10),
    ("11_20", 11, 20),
    ("21_40", 21, 40),
)

OBSERVATION_ENCODERS = (
    "pooled_2x2",
    "rgb_tile_7x7",
    "symbolic_7x7",
)


def replay_symbolic_observations(
    actions: np.ndarray,
    episode_seeds: np.ndarray,
    expected_poses: np.ndarray | None = None,
) -> np.ndarray:
    """Replay stored actions to recover native MiniGrid 7 x 7 observations."""
    actions = np.asarray(actions, dtype=np.int64)
    episode_seeds = np.asarray(episode_seeds, dtype=np.int64)
    if actions.ndim != 2 or episode_seeds.shape != (actions.shape[0],):
        raise ValueError("Actions and episode seeds have incompatible shapes.")
    if expected_poses is not None:
        expected_poses = np.asarray(expected_poses)
        if expected_poses.shape != (actions.shape[0], actions.shape[1] + 1, 3):
            raise ValueError("Expected poses must be shaped [episodes, time+1, 3].")

    env = gym.make(
        BRANCHING_ROOMS_ENV_ID,
        max_steps=int(actions.shape[1]),
        agent_view_size=7,
    )
    episodes = []
    try:
        for episode_index, (episode_actions, seed) in enumerate(
            zip(actions, episode_seeds)
        ):
            observation, info = env.reset(seed=int(seed))
            episode = [
                np.transpose(
                    np.asarray(observation["image"], dtype=np.uint8), (2, 0, 1)
                )
            ]
            replayed_poses = [
                (
                    int(info["agent_position"][0]),
                    int(info["agent_position"][1]),
                    int(info["agent_direction"]),
                )
            ]
            for action in episode_actions:
                observation, _, _, _, info = env.step(int(action))
                episode.append(
                    np.transpose(
                        np.asarray(observation["image"], dtype=np.uint8),
                        (2, 0, 1),
                    )
                )
                replayed_poses.append(
                    (
                        int(info["agent_position"][0]),
                        int(info["agent_position"][1]),
                        int(info["agent_direction"]),
                    )
                )
            if expected_poses is not None and not np.array_equal(
                np.asarray(replayed_poses), expected_poses[episode_index]
            ):
                raise RuntimeError(
                    f"Symbolic replay diverged in episode {episode_index}."
                )
            episodes.append(np.stack(episode))
    finally:
        env.close()
    return np.stack(episodes)


def model_observation_tensor(
    observations: np.ndarray,
    observation_encoder: str,
    device: torch.device,
) -> torch.Tensor:
    tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
    if observation_encoder != "symbolic_7x7":
        tensor = tensor.div(255.0)
    return tensor


def binary_auc(targets: np.ndarray, scores: np.ndarray) -> float:
    targets = np.asarray(targets, dtype=np.bool_).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    positive_count = int(targets.sum())
    negative_count = int((~targets).sum())
    if positive_count == 0 or negative_count == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.arange(1, len(scores) + 1, dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[start:end] = 0.5 * (start + 1 + end)
        start = end
    original_ranks = np.empty_like(ranks)
    original_ranks[order] = ranks
    return float(
        (
            original_ranks[targets].sum()
            - positive_count * (positive_count + 1) / 2.0
        )
        / (positive_count * negative_count)
    )


def prediction_metrics(targets: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    targets = np.asarray(targets, dtype=np.float32).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if targets.size == 0:
        return {
            "count": 0.0,
            "target_mean": float("nan"),
            "prediction_mean": float("nan"),
            "auc": float("nan"),
            "brier": float("nan"),
            "accuracy": float("nan"),
            "novel_prediction_mean": float("nan"),
            "revisit_prediction_mean": float("nan"),
        }
    novel = targets >= 0.5
    revisit = ~novel
    return {
        "count": float(targets.size),
        "target_mean": float(targets.mean()),
        "prediction_mean": float(scores.mean()),
        "auc": binary_auc(targets, scores),
        "brier": float(np.mean((scores - targets) ** 2)),
        "accuracy": float(np.mean((scores >= 0.5) == novel)),
        "novel_prediction_mean": (
            float(scores[novel].mean()) if bool(novel.any()) else float("nan")
        ),
        "revisit_prediction_mean": (
            float(scores[revisit].mean()) if bool(revisit.any()) else float("nan")
        ),
    }


def recent_revisit_ages(
    observations: np.ndarray,
    root_indices: np.ndarray,
    horizon: int,
    history_window: int,
) -> np.ndarray:
    """Return zero for novel endpoints and the last-match distance for revisits."""
    observations = np.asarray(observations)
    if observations.dtype != np.uint8:
        scale = 255.0 if float(np.max(observations, initial=0.0)) <= 1.5 else 1.0
        observations = np.rint(
            np.clip(observations * scale, 0.0, 255.0)
        ).astype(np.uint8)
    ages = np.zeros((*root_indices.shape, int(horizon)), dtype=np.int16)
    for batch_index in range(observations.shape[0]):
        keys = [
            np.ascontiguousarray(observation).tobytes()
            for observation in observations[batch_index]
        ]
        for root_row, root_index in enumerate(root_indices[batch_index]):
            for depth in range(int(horizon)):
                endpoint = int(root_index) + depth + 1
                start = max(0, endpoint - int(history_window))
                endpoint_key = keys[endpoint]
                for previous in range(endpoint - 1, start - 1, -1):
                    if keys[previous] == endpoint_key:
                        ages[batch_index, root_row, depth] = endpoint - previous
                        break
    return ages


def _evaluation_roots(
    action_count: int,
    history_window: int,
    horizon: int,
    stride: int,
) -> np.ndarray:
    first = max(0, int(history_window) - 1)
    last = int(action_count) - int(horizon)
    if last < first:
        return np.zeros(0, dtype=np.int64)
    roots = np.arange(first, last + 1, max(1, int(stride)), dtype=np.int64)
    if roots[-1] != last:
        roots = np.concatenate([roots, np.asarray([last], dtype=np.int64)])
    return roots


def evaluate(
    model: RecentHistoryTransformerNovelty,
    model_observations: np.ndarray,
    target_observations: np.ndarray,
    actions: np.ndarray,
    indices: np.ndarray,
    history_window: int,
    horizon: int,
    device: torch.device,
    episode_batch_size: int,
    root_stride: int,
    max_episodes: int,
) -> dict[str, float]:
    indices = np.asarray(indices, dtype=np.int64)
    if max_episodes > 0:
        indices = indices[:max_episodes]
    roots_1d = _evaluation_roots(
        actions.shape[1], history_window, horizon, root_stride
    )
    if indices.size == 0 or roots_1d.size == 0:
        return prediction_metrics(np.zeros(0), np.zeros(0))

    prediction_parts = []
    target_parts = []
    action_parts = []
    age_parts = []
    blocked_parts = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), episode_batch_size):
            batch_indices = indices[start:start + episode_batch_size]
            batch_model_observations = model_observations[batch_indices]
            batch_target_observations = target_observations[batch_indices]
            batch_actions = actions[batch_indices]
            roots = np.tile(roots_1d[None, :], (len(batch_indices), 1))
            targets = recent_observation_novelty_targets(
                batch_target_observations,
                roots,
                horizon=horizon,
                history_window=history_window,
            )
            ages = recent_revisit_ages(
                batch_target_observations,
                roots,
                horizon=horizon,
                history_window=history_window,
            )
            observation_tensor = model_observation_tensor(
                batch_model_observations,
                model.observation_encoder_type,
                device,
            )
            action_tensor = torch.as_tensor(
                batch_actions, dtype=torch.long, device=device
            )
            root_tensor = torch.as_tensor(roots, dtype=torch.long, device=device)
            zeros = torch.zeros_like(action_tensor, dtype=torch.float32)
            probabilities = torch.sigmoid(
                model.score_actual_suffixes(
                    observation_tensor,
                    action_tensor,
                    zeros,
                    zeros,
                    root_tensor,
                    horizon,
                )
            ).cpu().numpy()

            suffix_indices = roots[:, :, None] + np.arange(horizon)[None, None]
            batch_rows = np.arange(len(batch_indices))[:, None, None]
            edge_actions = batch_actions[batch_rows, suffix_indices]
            before = batch_target_observations[batch_rows, suffix_indices]
            after = batch_target_observations[batch_rows, suffix_indices + 1]
            unchanged = np.all(before == after, axis=tuple(range(3, before.ndim)))

            prediction_parts.append(probabilities)
            target_parts.append(targets)
            action_parts.append(edge_actions)
            age_parts.append(ages)
            blocked_parts.append(unchanged)
    model.train(was_training)

    predictions = np.concatenate(prediction_parts, axis=0)
    targets = np.concatenate(target_parts, axis=0)
    edge_actions = np.concatenate(action_parts, axis=0)
    ages = np.concatenate(age_parts, axis=0)
    unchanged = np.concatenate(blocked_parts, axis=0)
    result = prediction_metrics(targets, predictions)

    def add_group(prefix: str, mask: np.ndarray) -> None:
        metrics = prediction_metrics(targets[mask], predictions[mask])
        for key, value in metrics.items():
            result[f"{prefix}_{key}"] = value

    for depth in range(horizon):
        mask = np.zeros(targets.shape, dtype=np.bool_)
        mask[:, :, depth] = True
        add_group(f"depth_{depth + 1}", mask)
    for action in range(model.action_space_size):
        add_group(f"action_{ACTION_NAMES.get(action, str(action))}", edge_actions == action)
    forward = edge_actions == 2
    add_group("forward_blocked", forward & unchanged)
    add_group("forward_moved", forward & ~unchanged)
    novel = targets >= 0.5
    for name, minimum, maximum in REVISIT_AGE_BINS:
        revisit_in_bin = (ages >= minimum) & (ages <= maximum)
        add_group(f"revisit_age_{name}", novel | revisit_in_bin)
        result[f"revisit_age_{name}_revisit_count"] = float(revisit_in_bin.sum())
        result[f"revisit_age_{name}_revisit_prediction_mean"] = (
            float(predictions[revisit_in_bin].mean())
            if bool(revisit_in_bin.any()) else float("nan")
        )
    result["episode_count"] = float(indices.size)
    result["root_count"] = float(indices.size * roots_1d.size)
    return result


def _format_metric(value: float) -> str:
    return "nan" if not np.isfinite(value) else f"{value:.3f}"


def print_evaluation(budget: int, split: str, metrics: dict[str, float]) -> None:
    depth_auc = "/".join(
        _format_metric(metrics[f"depth_{depth}_auc"])
        for depth in range(1, 6)
        if f"depth_{depth}_auc" in metrics
    )
    action_auc = "/".join(
        f"{name}:{_format_metric(metrics.get(f'action_{name}_auc', float('nan')))}"
        for name in ("left", "right", "forward")
    )
    age_auc = "/".join(
        f"{name}:{_format_metric(metrics.get(f'revisit_age_{name}_auc', float('nan')))}"
        for name, _, _ in REVISIT_AGE_BINS
    )
    print(
        f"budget={budget} split={split} auc={_format_metric(metrics['auc'])} "
        f"brier={_format_metric(metrics['brier'])} "
        f"accuracy={_format_metric(metrics['accuracy'])} "
        f"target={_format_metric(metrics['target_mean'])} "
        f"pred(novel/revisit)="
        f"{_format_metric(metrics['novel_prediction_mean'])}/"
        f"{_format_metric(metrics['revisit_prediction_mean'])}"
    )
    print(f"  depth_auc={depth_auc}")
    print(
        f"  action_auc={action_auc} "
        f"blocked_forward(pred/count)="
        f"{_format_metric(metrics.get('forward_blocked_prediction_mean', float('nan')))}/"
        f"{int(metrics.get('forward_blocked_count', 0.0))} "
        f"moved_forward_auc="
        f"{_format_metric(metrics.get('forward_moved_auc', float('nan')))}"
    )
    print(f"  revisit_age_auc={age_auc}")


def parse_budgets(value: str) -> list[int]:
    budgets = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not budgets or budgets[0] <= 0:
        raise argparse.ArgumentTypeError("budgets must contain positive integers")
    return budgets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default=(
            ".local_runs/branching_rooms_action_novelty_macro_augmented_seed0/"
            "experienced_trajectories.npz"
        ),
    )
    parser.add_argument("--budgets", type=parse_budgets, default=parse_budgets("64,256"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--history-window", type=int, default=40)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--roots-per-episode", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument(
        "--observation-encoder",
        choices=OBSERVATION_ENCODERS,
        default="pooled_2x2",
    )
    parser.add_argument("--symbolic-cache", type=Path, default=None)
    parser.add_argument("--eval-episode-batch-size", type=int, default=4)
    parser.add_argument("--eval-root-stride", type=int, default=8)
    parser.add_argument("--max-eval-episodes", type=int, default=96)
    parser.add_argument("--policy-breakdown", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--checkpoint-out", type=Path, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(8, max(1, torch.get_num_threads())))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = np.load(args.dataset)
    target_observations = dataset["observations"]
    actions = dataset["actions"].astype(np.int64)
    if args.observation_encoder == "symbolic_7x7":
        symbolic_cache = args.symbolic_cache
        if symbolic_cache is None:
            symbolic_cache = Path(args.dataset).with_name(
                "symbolic_observations.npy"
            )
        if symbolic_cache.is_file():
            observations = np.load(symbolic_cache, mmap_mode="r")
        else:
            observations = replay_symbolic_observations(
                actions,
                dataset["episode_seeds"],
                dataset["poses"],
            )
            symbolic_cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(symbolic_cache, observations)
        if observations.shape[:2] != target_observations.shape[:2]:
            raise ValueError("Cached symbolic observations do not match the dataset.")
        print(f"symbolic_observations={symbolic_cache}", flush=True)
    else:
        observations = target_observations
    split_ids = dataset["split_ids"]
    splits = {
        "train": np.flatnonzero(split_ids == 0),
        "validation": np.flatnonzero(split_ids == 1),
        "test": np.flatnonzero(split_ids == 2),
        "heldout_policy": np.flatnonzero(split_ids == 3),
    }
    if args.policy_breakdown and "policy_ids" in dataset.files:
        policy_ids = dataset["policy_ids"]
        metadata = json.loads(str(dataset["metadata_json"].item()))
        policy_names = metadata.get("policy_names", [])
        for split_name, split_indices in list(splits.items()):
            for policy_id in np.unique(policy_ids[split_indices]):
                policy_name = (
                    str(policy_names[int(policy_id)])
                    if int(policy_id) < len(policy_names)
                    else f"policy_{int(policy_id)}"
                )
                matching = split_indices[policy_ids[split_indices] == policy_id]
                if matching.size:
                    splits[f"{split_name}_{policy_name}"] = matching
    train_indices = splits["train"]

    model = RecentHistoryTransformerNovelty(
        observation_channels=int(observations.shape[2]),
        action_space_size=int(actions.max()) + 1,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        history_window=args.history_window,
        max_horizon=args.horizon,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        transformer_dropout=0.0,
        observation_encoder_type=args.observation_encoder,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    weights = torch.pow(
        torch.tensor(0.99, dtype=torch.float32, device=device),
        torch.arange(args.horizon, dtype=torch.float32, device=device),
    )
    weight_sum = weights.sum().clamp_min(1e-8)

    results = {
        "config": {
            **vars(args),
            "budgets": args.budgets,
            "output_json": str(args.output_json) if args.output_json else None,
            "checkpoint_out": (
                str(args.checkpoint_out) if args.checkpoint_out else None
            ),
            "dataset": str(args.dataset),
            "device": str(device),
        },
        "evaluations": {},
    }
    print(
        f"device={device} train_episodes={len(train_indices)} "
        f"model={args.embedding_dim}/{args.hidden_dim}/"
        f"{args.transformer_layers} batch={args.batch_size} "
        f"roots={args.roots_per_episode} budgets={args.budgets}"
    )

    def evaluate_all(budget: int) -> None:
        budget_results = {}
        for name, indices in splits.items():
            metrics = evaluate(
                model,
                observations,
                target_observations,
                actions,
                indices,
                args.history_window,
                args.horizon,
                device,
                args.eval_episode_batch_size,
                args.eval_root_stride,
                args.max_eval_episodes,
            )
            budget_results[name] = metrics
            print_evaluation(budget, name, metrics)
        results["evaluations"][str(budget)] = budget_results
        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(results, indent=2, sort_keys=True))
        if args.checkpoint_out is not None and budget > 0:
            checkpoint_path = args.checkpoint_out.with_name(
                f"{args.checkpoint_out.stem}_update{budget}{args.checkpoint_out.suffix}"
            )
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "budget": budget,
                    "config": results["config"],
                },
                checkpoint_path,
            )

    evaluate_all(0)
    max_root = int(actions.shape[1]) - args.horizon
    report_interval = max(1, max(args.budgets) // 8)
    model.train()
    for update in range(1, max(args.budgets) + 1):
        batch_indices = np.random.choice(
            train_indices, size=args.batch_size, replace=True
        )
        roots = np.random.randint(
            0,
            max_root + 1,
            size=(args.batch_size, args.roots_per_episode),
        ).astype(np.int64)
        batch_observations = observations[batch_indices]
        batch_actions = actions[batch_indices]
        targets = recent_observation_novelty_targets(
            target_observations[batch_indices],
            roots,
            horizon=args.horizon,
            history_window=args.history_window,
        )
        observation_tensor = model_observation_tensor(
            batch_observations,
            args.observation_encoder,
            device,
        )
        action_tensor = torch.as_tensor(
            batch_actions, dtype=torch.long, device=device
        )
        root_tensor = torch.as_tensor(roots, dtype=torch.long, device=device)
        target_tensor = torch.as_tensor(
            targets, dtype=torch.float32, device=device
        )
        zeros = torch.zeros_like(action_tensor, dtype=torch.float32)
        logits = model.score_actual_suffixes(
            observation_tensor,
            action_tensor,
            zeros,
            zeros,
            root_tensor,
            args.horizon,
        )
        edge_loss = F.binary_cross_entropy_with_logits(logits, target_tensor)
        probabilities = torch.sigmoid(logits)
        predicted_return = torch.sum(probabilities * weights, dim=-1) / weight_sum
        target_return = torch.sum(target_tensor * weights, dim=-1) / weight_sum
        sequence_loss = F.smooth_l1_loss(predicted_return, target_return)
        loss = edge_loss + sequence_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if update == 1 or update % report_interval == 0:
            print(
                f"update={update} loss={float(loss.detach()):.4f} "
                f"bce={float(edge_loss.detach()):.4f} "
                f"sequence={float(sequence_loss.detach()):.4f}"
            )
        if update in args.budgets:
            evaluate_all(update)
            model.train()


if __name__ == "__main__":
    main()
