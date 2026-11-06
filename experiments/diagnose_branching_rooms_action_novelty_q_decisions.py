"""Diagnose whether episodic-Q decisions avoid already visited states.

The environment transition function is used only for retrospective evaluation.
It reconstructs all three one-step outcomes at every saved root, verifies the
executed transition, and labels actions by RGB-, pose-, and position-novelty.
For the online-Q condition, earlier planner rankings are also rescored with the
final checkpoint to measure eventual correction of avoidable revisits.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import torch

if __package__:
    from .run_branching_rooms_action_novelty import ActionNoveltyPredictor
    from .run_branching_rooms_action_novelty_q import (
        PathConditionedEpisodicQ,
        branch_scores,
    )
    from .run_branching_rooms_online_action_novelty import (
        enumerate_action_suffixes,
        make_env,
    )
else:
    from run_branching_rooms_action_novelty import ActionNoveltyPredictor
    from run_branching_rooms_action_novelty_q import (
        PathConditionedEpisodicQ,
        branch_scores,
    )
    from run_branching_rooms_online_action_novelty import (
        enumerate_action_suffixes,
        make_env,
    )


CONDITIONS = ("random", "frozen_topk", "q_frozen", "q_online")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(".local_runs/branching_rooms_action_novelty_q_seed0"),
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser


def observation_key(observation: np.ndarray) -> bytes:
    return np.ascontiguousarray(observation).tobytes()


def exact_one_step_novelty(
    observations: np.ndarray,
    poses: np.ndarray,
    actions: np.ndarray,
    episode_length: int,
    tile_size: int,
) -> Mapping[str, np.ndarray]:
    action_count = int(actions.size)
    rgb_novel = np.zeros((action_count, 3), dtype=np.bool_)
    pose_novel = np.zeros((action_count, 3), dtype=np.bool_)
    position_novel = np.zeros((action_count, 3), dtype=np.bool_)
    next_poses = np.zeros((action_count, 3, 3), dtype=np.int16)
    env = make_env(episode_length, tile_size)
    env.reset(seed=0)
    base = env.unwrapped
    visited_rgb = {observation_key(observations[0])}
    visited_poses = {tuple(int(value) for value in poses[0])}
    visited_positions = {tuple(int(value) for value in poses[0, :2])}
    try:
        for root in range(action_count):
            x, y, direction = (int(value) for value in poses[root])
            generated = []
            for action in range(3):
                base.agent_pos = (x, y)
                base.agent_dir = direction
                base.step_count = root
                observation, _, _, _, info = env.step(action)
                channel_first = np.transpose(observation, (2, 0, 1)).astype(
                    np.uint8, copy=False
                )
                next_pose = (
                    int(info["agent_position"][0]),
                    int(info["agent_position"][1]),
                    int(info["agent_direction"]),
                )
                next_poses[root, action] = next_pose
                rgb_novel[root, action] = observation_key(channel_first) not in visited_rgb
                pose_novel[root, action] = next_pose not in visited_poses
                position_novel[root, action] = next_pose[:2] not in visited_positions
                generated.append(channel_first)
            chosen = int(actions[root])
            if not np.array_equal(generated[chosen], observations[root + 1]):
                raise RuntimeError(
                    f"Reconstructed transition differs at root {root}, action {chosen}."
                )
            visited_rgb.add(observation_key(observations[root + 1]))
            visited_poses.add(tuple(int(value) for value in poses[root + 1]))
            visited_positions.add(tuple(int(value) for value in poses[root + 1, :2]))
    finally:
        env.close()
    return {
        "rgb": rgb_novel,
        "pose": pose_novel,
        "position": position_novel,
        "next_poses": next_poses,
    }


def planner_statistics(
    labels: np.ndarray,
    actions: np.ndarray,
    exploratory: np.ndarray,
    values: np.ndarray | None,
) -> Dict[str, float | int | None]:
    opportunities = labels.any(axis=1)
    chosen_novel = labels[np.arange(actions.size), actions]
    policy_mask = ~exploratory
    policy_opportunities = opportunities & policy_mask
    selected_avoidable = opportunities & ~chosen_novel
    result: Dict[str, float | int | None] = {
        "decision_count": int(actions.size),
        "novel_action_opportunity_count": int(opportunities.sum()),
        "selected_novel_count": int(chosen_novel.sum()),
        "selected_novel_rate": float(chosen_novel.mean()),
        "selected_correct_when_avoidable_rate": (
            float(chosen_novel[opportunities].mean()) if opportunities.any() else None
        ),
        "selected_avoidable_revisit_count": int(selected_avoidable.sum()),
        "selected_unavoidable_revisit_count": int((~opportunities & ~chosen_novel).sum()),
        "policy_novel_action_opportunity_count": int(policy_opportunities.sum()),
        "exploration_novel_action_opportunity_count": int(
            (opportunities & exploratory).sum()
        ),
        "policy_avoidable_revisit_count": int(
            (policy_opportunities & ~chosen_novel).sum()
        ),
        "exploration_avoidable_revisit_count": int(
            (opportunities & exploratory & ~chosen_novel).sum()
        ),
    }
    if values is None:
        result.update(
            greedy_correct_when_avoidable_rate=None,
            greedy_avoidable_revisit_count=None,
            mixed_novel_revisit_root_count=None,
            mean_best_novel_minus_best_revisit=None,
            positive_novel_margin_rate=None,
        )
        return result

    greedy_actions = values.argmax(axis=1)
    greedy_novel = labels[np.arange(actions.size), greedy_actions]
    mixed = labels.any(axis=1) & (~labels).any(axis=1)
    margins = np.full(actions.size, np.nan, dtype=np.float32)
    for root in np.flatnonzero(mixed):
        margins[root] = float(values[root, labels[root]].max()) - float(
            values[root, ~labels[root]].max()
        )
    result.update(
        greedy_correct_when_avoidable_rate=(
            float(greedy_novel[opportunities].mean()) if opportunities.any() else None
        ),
        greedy_avoidable_revisit_count=int((opportunities & ~greedy_novel).sum()),
        mixed_novel_revisit_root_count=int(mixed.sum()),
        mean_best_novel_minus_best_revisit=(
            float(np.nanmean(margins)) if mixed.any() else None
        ),
        positive_novel_margin_rate=(
            float(np.mean(margins[mixed] > 0)) if mixed.any() else None
        ),
    )
    return result


def load_final_models(
    run_dir: Path, device: torch.device
) -> Tuple[ActionNoveltyPredictor, PathConditionedEpisodicQ, Mapping[str, object]]:
    arguments = json.loads((run_dir / "arguments.json").read_text(encoding="utf-8"))
    offline_run = Path(arguments["offline_run"])
    novelty_checkpoint = torch.load(
        offline_run / "full_history" / "best.pt",
        map_location=device,
        weights_only=False,
    )
    novelty_model = ActionNoveltyPredictor(
        action_space_size=3,
        embedding_dim=int(novelty_checkpoint["embedding_dim"]),
        hidden_dim=int(novelty_checkpoint["hidden_dim"]),
        use_history=True,
    ).to(device)
    novelty_model.load_state_dict(novelty_checkpoint["model"])
    novelty_model.eval()
    q_model = PathConditionedEpisodicQ(novelty_model).to(device)
    q_checkpoint = torch.load(
        run_dir / "q_online_final.pt", map_location=device, weights_only=False
    )
    q_model.load_state_dict(q_checkpoint["model"])
    q_model.eval()
    return novelty_model, q_model, arguments


def model_action_values(
    observations: np.ndarray,
    actions: np.ndarray,
    novelty_model: ActionNoveltyPredictor,
    q_model: PathConditionedEpisodicQ,
    horizon: int,
    top_k: int,
    gamma: float,
    device: torch.device,
) -> np.ndarray:
    suffixes = enumerate_action_suffixes(3, horizon)
    suffix_tensor = torch.from_numpy(suffixes).to(device)
    first_observation = torch.from_numpy(observations[0:1]).to(device)
    with torch.no_grad():
        novelty_context = novelty_model.start_online_context(first_observation)
        q_context = q_model.start_online_context(first_observation)
        all_values = []
        for root in range(actions.size):
            probabilities = torch.softmax(
                novelty_model.score_action_suffixes(novelty_context, suffix_tensor),
                dim=-1,
            ).cpu().numpy()
            q_values = q_model.score_action_prefixes(
                q_context, suffix_tensor
            ).cpu().numpy()
            scores = branch_scores(probabilities, q_values, gamma)
            values = []
            for action in range(3):
                selected = scores[suffixes[:, 0] == action]
                count = min(int(top_k), int(selected.size))
                values.append(float(np.partition(selected, -count)[-count:].mean()))
            all_values.append(values)
            observation = torch.from_numpy(observations[root + 1 : root + 2]).to(device)
            action = torch.tensor([int(actions[root])], device=device)
            novelty_context = novelty_model.advance_online_context(
                novelty_context, action, observation
            )
            q_context = q_model.advance_online_context(q_context, action, observation)
    return np.asarray(all_values, dtype=np.float32)


def correction_statistics(
    labels: np.ndarray,
    before_values: np.ndarray,
    final_values: np.ndarray,
) -> Mapping[str, float | int | None]:
    opportunities = labels.any(axis=1)
    before_actions = before_values.argmax(axis=1)
    final_actions = final_values.argmax(axis=1)
    before_correct = labels[np.arange(labels.shape[0]), before_actions]
    final_correct = labels[np.arange(labels.shape[0]), final_actions]
    before_wrong = opportunities & ~before_correct
    before_right = opportunities & before_correct
    return {
        "earlier_wrong_ranking_count": int(before_wrong.sum()),
        "rescored_corrected_count": int((before_wrong & final_correct).sum()),
        "rescored_corrected_rate": (
            float(final_correct[before_wrong].mean()) if before_wrong.any() else None
        ),
        "earlier_correct_ranking_count": int(before_right.sum()),
        "rescored_regressed_count": int((before_right & ~final_correct).sum()),
        "rescored_regressed_rate": (
            float((~final_correct[before_right]).mean()) if before_right.any() else None
        ),
    }


def ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator else None


def aggregate_planner_statistics(
    rows: Sequence[Mapping[str, object]], label: str
) -> Mapping[str, object]:
    def total(name: str) -> float:
        return float(sum(float(row[f"{label}_{name}"]) for row in rows))

    decisions = total("decision_count")
    opportunities = total("novel_action_opportunity_count")
    selected_novel = total("selected_novel_count")
    selected_avoidable = total("selected_avoidable_revisit_count")
    policy_opportunities = total("policy_novel_action_opportunity_count")
    exploration_opportunities = total("exploration_novel_action_opportunity_count")
    policy_avoidable = total("policy_avoidable_revisit_count")
    exploration_avoidable = total("exploration_avoidable_revisit_count")
    result: Dict[str, object] = {
        "decision_count": int(decisions),
        "novel_action_opportunity_count": int(opportunities),
        "selected_novel_rate": ratio(selected_novel, decisions),
        "selected_correct_when_avoidable_rate": ratio(
            opportunities - selected_avoidable, opportunities
        ),
        "selected_avoidable_revisit_count": int(selected_avoidable),
        "selected_unavoidable_revisit_count": int(
            total("selected_unavoidable_revisit_count")
        ),
        "policy_avoidable_revisit_count": int(policy_avoidable),
        "policy_correct_when_avoidable_rate": ratio(
            policy_opportunities - policy_avoidable, policy_opportunities
        ),
        "exploration_avoidable_revisit_count": int(exploration_avoidable),
        "exploration_correct_when_avoidable_rate": ratio(
            exploration_opportunities - exploration_avoidable,
            exploration_opportunities,
        ),
    }
    greedy_values = [
        row[f"{label}_greedy_avoidable_revisit_count"] for row in rows
    ]
    if any(value is not None for value in greedy_values):
        greedy_avoidable = float(
            sum(float(value) for value in greedy_values if value is not None)
        )
        mixed_counts = np.asarray(
            [float(row[f"{label}_mixed_novel_revisit_root_count"]) for row in rows]
        )
        margin_means = np.asarray(
            [float(row[f"{label}_mean_best_novel_minus_best_revisit"]) for row in rows]
        )
        positive_rates = np.asarray(
            [float(row[f"{label}_positive_novel_margin_rate"]) for row in rows]
        )
        result.update(
            greedy_avoidable_revisit_count=int(greedy_avoidable),
            greedy_correct_when_avoidable_rate=ratio(
                opportunities - greedy_avoidable, opportunities
            ),
            mixed_novel_revisit_root_count=int(mixed_counts.sum()),
            mean_best_novel_minus_best_revisit=(
                float(np.average(margin_means, weights=mixed_counts))
                if mixed_counts.sum()
                else None
            ),
            positive_novel_margin_rate=(
                float(np.average(positive_rates, weights=mixed_counts))
                if mixed_counts.sum()
                else None
            ),
        )
    return result


def aggregate_corrections(
    rows: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    wrong = sum(int(row["earlier_wrong_ranking_count"]) for row in rows)
    corrected = sum(int(row["rescored_corrected_count"]) for row in rows)
    right = sum(int(row["earlier_correct_ranking_count"]) for row in rows)
    regressed = sum(int(row["rescored_regressed_count"]) for row in rows)
    return {
        "earlier_wrong_ranking_count": wrong,
        "rescored_corrected_count": corrected,
        "rescored_corrected_rate": ratio(corrected, wrong),
        "earlier_correct_ranking_count": right,
        "rescored_regressed_count": regressed,
        "rescored_regressed_rate": ratio(regressed, right),
    }


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)
    arguments = json.loads((args.run_dir / "arguments.json").read_text())
    episode_length = int(arguments["episode_length"])
    tile_size = int(arguments["tile_size"])
    novelty_model, final_q_model, _ = load_final_models(args.run_dir, device)
    rows = []
    correction_rows = []
    trajectory_dir = args.run_dir / "trajectories"
    for condition in CONDITIONS:
        for path in sorted(trajectory_dir.glob(f"{condition}_episode_*.npz")):
            data = np.load(path)
            actions = data["actions"].astype(np.int64, copy=False)
            labels = exact_one_step_novelty(
                data["observations"],
                data["poses"],
                actions,
                episode_length,
                tile_size,
            )
            values = data["action_values"]
            if values.size == 0:
                values = None
            exploratory = data["exploratory_flags"]
            if exploratory.size == 0:
                exploratory = np.zeros(actions.size, dtype=np.bool_)
            episode = int(path.stem.rsplit("_", 1)[1])
            row: Dict[str, object] = {"condition": condition, "episode": episode}
            for label_name in ("rgb", "pose", "position"):
                statistics = planner_statistics(
                    labels[label_name], actions, exploratory, values
                )
                row.update(
                    {f"{label_name}_{key}": value for key, value in statistics.items()}
                )
            rows.append(row)
            if condition == "q_online":
                immediate_checkpoint = (
                    args.run_dir / f"q_online_episode_{episode:03d}.pt"
                )
                checkpoint_path = (
                    immediate_checkpoint
                    if immediate_checkpoint.exists()
                    else args.run_dir / "q_online_final.pt"
                )
                q_checkpoint = torch.load(
                    checkpoint_path, map_location=device, weights_only=False
                )
                final_q_model.load_state_dict(q_checkpoint["model"])
                final_values = model_action_values(
                    data["observations"],
                    actions,
                    novelty_model,
                    final_q_model,
                    int(arguments["horizon"]),
                    int(arguments["top_k"]),
                    float(arguments["gamma"]),
                    device,
                )
                correction = {
                    "condition": condition,
                    "episode": episode,
                    "correction_mode": (
                        "immediate_post_episode"
                        if immediate_checkpoint.exists()
                        else "eventual_final_model"
                    ),
                    **correction_statistics(labels["rgb"], values, final_values),
                }
                correction_rows.append(correction)

    csv_path = args.run_dir / "decision_diagnostics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for condition in CONDITIONS:
        selected = [row for row in rows if row["condition"] == condition]
        summary[condition] = {
            label: aggregate_planner_statistics(selected, label)
            for label in ("rgb", "pose", "position")
        }
    correction_modes = {
        str(row["correction_mode"]) for row in correction_rows
    }
    correction_key = (
        "q_online_immediate_correction"
        if correction_modes == {"immediate_post_episode"}
        else "q_online_final_correction"
    )
    summary[correction_key] = aggregate_corrections(correction_rows)
    output = {
        "definition": (
            "An avoidable revisit occurs when the selected action reaches a state "
            "already in episode memory while at least one other action is novel."
        ),
        "summary": summary,
        "q_online_correction_by_episode": correction_rows,
    }
    output_path = args.run_dir / "decision_diagnostics.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"wrote {csv_path} and {output_path}")


if __name__ == "__main__":
    main()
