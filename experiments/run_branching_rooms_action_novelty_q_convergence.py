"""Measure convergence of the path-conditioned episodic Q model.

Training is identical to the offline 20-step SARSA stage used by the behavior
experiment.  Evaluation additionally reconstructs all three one-step outcomes
at fixed validation roots.  Environment transitions are used only to construct
retrospective evaluation labels, never as training targets for unselected
actions.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import torch

if __package__:
    from .diagnose_branching_rooms_action_novelty_q_decisions import (
        exact_one_step_novelty,
    )
    from .run_branching_rooms_action_novelty import (
        NOVEL,
        ActionNoveltyPredictor,
        load_dataset,
    )
    from .run_branching_rooms_action_novelty_q import (
        PathConditionedEpisodicQ,
        branch_scores,
        bundle_rewards,
        evaluate_q,
        q_batch_loss,
        resolve_device,
        sampled_roots,
        soft_update,
        tensor_batch,
    )
    from .run_branching_rooms_online_action_novelty import enumerate_action_suffixes
else:
    from diagnose_branching_rooms_action_novelty_q_decisions import (
        exact_one_step_novelty,
    )
    from run_branching_rooms_action_novelty import (
        NOVEL,
        ActionNoveltyPredictor,
        load_dataset,
    )
    from run_branching_rooms_action_novelty_q import (
        PathConditionedEpisodicQ,
        branch_scores,
        bundle_rewards,
        evaluate_q,
        q_batch_loss,
        resolve_device,
        sampled_roots,
        soft_update,
        tensor_batch,
    )
    from run_branching_rooms_online_action_novelty import enumerate_action_suffixes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline-run",
        type=Path,
        default=Path(".local_runs/branching_rooms_action_novelty_seed0"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            ".local_runs/branching_rooms_action_novelty_q_convergence_seed0"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--roots-per-update", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--td-steps", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--target-tau", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--oracle-episode-count", type=int, default=8)
    parser.add_argument("--oracle-roots-per-episode", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser


def aggregate_action_values(
    scores: np.ndarray,
    suffixes: np.ndarray,
    top_k: int,
) -> np.ndarray:
    values = []
    for action in range(3):
        selected = scores[suffixes[:, 0] == action]
        count = min(int(top_k), int(selected.size))
        values.append(float(np.partition(selected, -count)[-count:].mean()))
    return np.asarray(values, dtype=np.float32)


def ranking_metrics(
    values: np.ndarray, labels: np.ndarray, prefix: str
) -> Mapping[str, float | int | None]:
    opportunities = labels.any(axis=1)
    greedy = values.argmax(axis=1)
    correct = labels[np.arange(labels.shape[0]), greedy]
    mixed = opportunities & (~labels).any(axis=1)
    margins = []
    for index in np.flatnonzero(mixed):
        margins.append(
            float(values[index, labels[index]].max())
            - float(values[index, ~labels[index]].max())
        )
    return {
        f"{prefix}_root_count": int(labels.shape[0]),
        f"{prefix}_opportunity_count": int(opportunities.sum()),
        f"{prefix}_correct_count": int((opportunities & correct).sum()),
        f"{prefix}_correct_rate": (
            float(correct[opportunities].mean()) if opportunities.any() else None
        ),
        f"{prefix}_avoidable_revisit_count": int(
            (opportunities & ~correct).sum()
        ),
        f"{prefix}_mixed_root_count": int(mixed.sum()),
        f"{prefix}_novel_margin_mean": (
            float(np.mean(margins)) if margins else None
        ),
        f"{prefix}_novel_margin_positive_rate": (
            float(np.mean(np.asarray(margins) > 0)) if margins else None
        ),
    }


def build_oracle_roots(
    bundle,
    validation_indices: np.ndarray,
    episode_count: int,
    roots_per_episode: int,
    episode_length: int,
    tile_size: int,
    seed: int,
) -> Sequence[Mapping[str, object]]:
    rng = np.random.default_rng(int(seed) * 7919 + 101)
    selected_episodes = validation_indices[: min(int(episode_count), len(validation_indices))]
    records = []
    for episode in selected_episodes:
        actions = bundle.actions[episode].astype(np.int64, copy=False)
        count = min(int(roots_per_episode), actions.size)
        roots = np.sort(rng.choice(actions.size, size=count, replace=False))
        labels = exact_one_step_novelty(
            bundle.observations[episode],
            bundle.poses[episode],
            actions,
            episode_length,
            tile_size,
        )
        records.append(
            {
                "episode": int(episode),
                "roots": roots,
                "labels": labels,
            }
        )
    return records


def evaluate_action_ranking(
    q_model: PathConditionedEpisodicQ,
    novelty_model: ActionNoveltyPredictor,
    bundle,
    oracle_records: Sequence[Mapping[str, object]],
    horizon: int,
    gamma: float,
    top_k: int,
    device: torch.device,
) -> Mapping[str, object]:
    suffixes = enumerate_action_suffixes(3, horizon)
    suffix_tensor = torch.from_numpy(suffixes).to(device)
    one_step_actions = torch.arange(3, device=device).reshape(3, 1)
    q_values_by_label: Dict[str, List[np.ndarray]] = {
        "rgb": [], "pose": [], "position": []
    }
    planner_values_by_label: Dict[str, List[np.ndarray]] = {
        "rgb": [], "pose": [], "position": []
    }
    labels_by_name: Dict[str, List[np.ndarray]] = {
        "rgb": [], "pose": [], "position": []
    }
    q_model.eval()
    novelty_model.eval()
    with torch.no_grad():
        for record in oracle_records:
            episode = int(record["episode"])
            roots = set(int(root) for root in record["roots"])
            actions = bundle.actions[episode].astype(np.int64, copy=False)
            observations = bundle.observations[episode]
            observation = torch.from_numpy(observations[0:1]).to(device)
            novelty_context = novelty_model.start_online_context(observation)
            q_context = q_model.start_online_context(observation)
            for root in range(actions.size):
                if root in roots:
                    one_step_q = q_model.score_action_prefixes(
                        q_context, one_step_actions
                    )[:, 0].cpu().numpy()
                    probabilities = torch.softmax(
                        novelty_model.score_action_suffixes(
                            novelty_context, suffix_tensor
                        ),
                        dim=-1,
                    ).cpu().numpy()
                    branch_q = q_model.score_action_prefixes(
                        q_context, suffix_tensor
                    ).cpu().numpy()
                    planner_scores = branch_scores(probabilities, branch_q, gamma)
                    planner_values = aggregate_action_values(
                        planner_scores, suffixes, top_k
                    )
                    for label_name in labels_by_name:
                        labels = record["labels"][label_name][root]
                        q_values_by_label[label_name].append(one_step_q)
                        planner_values_by_label[label_name].append(planner_values)
                        labels_by_name[label_name].append(labels)
                next_observation = torch.from_numpy(
                    observations[root + 1 : root + 2]
                ).to(device)
                action = torch.tensor([int(actions[root])], device=device)
                novelty_context = novelty_model.advance_online_context(
                    novelty_context, action, next_observation
                )
                q_context = q_model.advance_online_context(
                    q_context, action, next_observation
                )

    metrics: Dict[str, object] = {}
    for label_name in labels_by_name:
        labels = np.stack(labels_by_name[label_name])
        q_values = np.stack(q_values_by_label[label_name])
        planner_values = np.stack(planner_values_by_label[label_name])
        metrics.update(ranking_metrics(q_values, labels, f"q_{label_name}"))
        metrics.update(
            ranking_metrics(planner_values, labels, f"planner_{label_name}")
        )
    return metrics


def write_metrics(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.with_suffix(".json").write_text(
        json.dumps(list(rows), indent=2), encoding="utf-8"
    )
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def apply_smoke(args: argparse.Namespace) -> None:
    if not args.smoke:
        return
    args.epochs = 2
    args.eval_interval = 1
    args.batch_size = 4
    args.roots_per_update = 4
    args.oracle_episode_count = 1
    args.oracle_roots_per_episode = 4


def main() -> None:
    args = build_parser().parse_args()
    apply_smoke(args)
    device = resolve_device(args.device)
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    checkpoint_path = args.offline_run / "full_history" / "best.pt"
    dataset_path = args.offline_run / "experienced_trajectories.npz"
    novelty_checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    bundle = load_dataset(dataset_path)
    episode_length = int(bundle.metadata["episode_length"])
    tile_size = int(bundle.metadata["tile_size"])
    novelty_model = ActionNoveltyPredictor(
        action_space_size=3,
        embedding_dim=int(novelty_checkpoint["embedding_dim"]),
        hidden_dim=int(novelty_checkpoint["hidden_dim"]),
        use_history=True,
    ).to(device)
    novelty_model.load_state_dict(novelty_checkpoint["model"])
    novelty_model.eval()
    for parameter in novelty_model.parameters():
        parameter.requires_grad_(False)

    model = PathConditionedEpisodicQ(novelty_model).to(device)
    target_model = copy.deepcopy(model).to(device).eval()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    rewards = bundle_rewards(bundle)
    train_indices = np.flatnonzero(bundle.split_ids == 0)
    validation_indices = np.flatnonzero(bundle.split_ids == 1)
    if args.smoke:
        train_indices = train_indices[:8]
        validation_indices = validation_indices[:4]
    rng = np.random.default_rng(int(args.seed) * 3571 + 13)
    oracle_records = build_oracle_roots(
        bundle,
        validation_indices,
        args.oracle_episode_count,
        args.oracle_roots_per_episode,
        episode_length,
        tile_size,
        args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arguments = vars(args).copy()
    arguments.update(
        offline_run=str(args.offline_run),
        output_dir=str(args.output_dir),
        device=str(device),
        training_counterfactual_transition_queries=0,
        evaluation_counterfactual_transition_queries=(
            3 * sum(len(record["roots"]) for record in oracle_records)
        ),
    )
    (args.output_dir / "arguments.json").write_text(
        json.dumps(arguments, indent=2), encoding="utf-8"
    )

    start_epoch = 0
    rows: List[Mapping[str, object]] = []
    latest_path = args.output_dir / "latest.pt"
    metrics_path = args.output_dir / "convergence"
    if args.resume:
        if not latest_path.exists():
            raise FileNotFoundError(f"Cannot resume without {latest_path}")
        latest = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(latest["model"])
        target_model.load_state_dict(latest["target_model"])
        optimizer.load_state_dict(latest["optimizer"])
        rng.bit_generator.state = latest["rng_state"]
        start_epoch = int(latest["epoch"])
        if metrics_path.with_suffix(".json").exists():
            rows = json.loads(metrics_path.with_suffix(".json").read_text())

    def evaluate(epoch: int, train_loss=None, train_td_mae=None):
        validation = evaluate_q(
            model,
            bundle.observations,
            bundle.actions,
            rewards,
            validation_indices,
            args.gamma,
            device,
            batch_size=args.batch_size,
        )
        ranking = evaluate_action_ranking(
            model,
            novelty_model,
            bundle,
            oracle_records,
            args.horizon,
            args.gamma,
            args.top_k,
            device,
        )
        row = {
            "epoch": int(epoch),
            "train_loss": train_loss,
            "train_td_mae": train_td_mae,
            **{f"validation_{key}": value for key, value in validation.items()},
            **ranking,
        }
        rows.append(row)
        write_metrics(metrics_path, rows)
        torch.save(
            {
                "model": model.state_dict(),
                "target_model": target_model.state_dict(),
                "epoch": int(epoch),
            },
            args.output_dir / f"epoch_{epoch:03d}.pt",
        )
        print(
            f"epoch={epoch:3d} val_mse={validation['mse']:.5f} "
            f"val_corr={validation['correlation']} "
            f"pred/target={validation['prediction_mean']:.3f}/"
            f"{validation['target_mean']:.3f} "
            f"Q-RGB={ranking['q_rgb_correct_rate']:.3f} "
            f"planner-RGB={ranking['planner_rgb_correct_rate']:.3f} "
            f"planner-position={ranking['planner_position_correct_rate']:.3f}"
        )

    if start_epoch == 0 and not rows:
        evaluate(0)
    for epoch in range(start_epoch + 1, int(args.epochs) + 1):
        losses = []
        td_maes = []
        model.train()
        shuffled = rng.permutation(train_indices)
        for start in range(0, len(shuffled), int(args.batch_size)):
            indices = shuffled[start : start + int(args.batch_size)]
            observations, actions, batch_rewards = tensor_batch(
                bundle.observations,
                bundle.actions,
                rewards,
                indices,
                device,
            )
            roots = sampled_roots(
                actions.shape[1],
                args.horizon,
                args.roots_per_update,
                rng,
            ).to(device)
            loss, diagnostics = q_batch_loss(
                model,
                target_model,
                observations,
                actions,
                batch_rewards,
                roots,
                args.horizon,
                args.td_steps,
                args.gamma,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()
            soft_update(target_model, model, args.target_tau)
            losses.append(float(loss.item()))
            td_maes.append(float(diagnostics["mae"]))
        torch.save(
            {
                "model": model.state_dict(),
                "target_model": target_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng_state": rng.bit_generator.state,
                "epoch": int(epoch),
            },
            latest_path,
        )
        if epoch % int(args.eval_interval) == 0 or epoch == int(args.epochs):
            evaluate(epoch, float(np.mean(losses)), float(np.mean(td_maes)))

    print(f"wrote convergence results to {args.output_dir}")


if __name__ == "__main__":
    main()
