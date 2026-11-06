"""Train a dedicated predictive exploration state on executed trajectories.

The model deliberately does not reuse MuZero's task latent.  A deterministic
observation key is used for episodic matching, while a recurrent belief carries
history needed to predict future keys.  Training only uses action/observation
sequences that were actually executed in Branching Rooms.

The primary acceptance tests are future-observation retrieval at depths 1-5
and revisit detection against root history plus the imagined branch prefix.
Physical poses are never training targets and are used only for diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

if __package__:
    from .run_branching_rooms_action_novelty import (
        BRANCH_REVISIT,
        NOVEL,
        ROOT_HISTORY_REVISIT,
        DatasetBundle,
        ObservationEncoder,
        load_dataset,
        state_ids_from_observations,
    )
else:
    from run_branching_rooms_action_novelty import (
        BRANCH_REVISIT,
        NOVEL,
        ROOT_HISTORY_REVISIT,
        DatasetBundle,
        ObservationEncoder,
        load_dataset,
        state_ids_from_observations,
    )


@dataclass(frozen=True)
class EncodedTrajectory:
    keys: torch.Tensor
    beliefs: torch.Tensor


class ExplorationEpisodeDataset(Dataset):
    def __init__(
        self,
        bundle: DatasetBundle,
        split_id: int,
        maximum_episodes: int | None = None,
    ) -> None:
        indices = np.flatnonzero(bundle.split_ids == int(split_id))
        if maximum_episodes is not None:
            indices = indices[: int(maximum_episodes)]
        self.bundle = bundle
        self.indices = indices

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, index: int) -> Mapping[str, torch.Tensor]:
        episode = int(self.indices[index])
        observations = self.bundle.observations[episode]
        return {
            "observations": torch.from_numpy(observations),
            "actions": torch.from_numpy(
                self.bundle.actions[episode].astype(np.int64, copy=False)
            ),
            "poses": torch.from_numpy(
                self.bundle.poses[episode].astype(np.int64, copy=False)
            ),
            "state_ids": torch.from_numpy(
                state_ids_from_observations(observations).astype(
                    np.int64, copy=False
                )
            ),
            "novelty_targets": torch.from_numpy(
                self.bundle.observation_targets[episode].astype(
                    np.int64, copy=False
                )
            ),
        }


class ExplorationWorldModel(nn.Module):
    """Observation-stable memory keys plus action-conditioned recurrent dynamics."""

    def __init__(
        self,
        action_space_size: int = 3,
        encoder_dim: int = 64,
        key_dim: int = 64,
        belief_dim: int = 128,
        action_dim: int = 16,
        cell_grid_size: int = 7,
    ) -> None:
        super().__init__()
        self.action_space_size = int(action_space_size)
        self.key_dim = int(key_dim)
        self.belief_dim = int(belief_dim)
        self.cell_grid_size = int(cell_grid_size)
        self.observation_encoder = ObservationEncoder(int(encoder_dim))
        self.key_projection = nn.Sequential(
            nn.Linear(int(encoder_dim), self.key_dim),
            nn.LayerNorm(self.key_dim),
        )
        self.action_embedding = nn.Embedding(
            self.action_space_size + 1, int(action_dim)
        )
        self.belief_cell = nn.GRUCell(
            self.key_dim + int(action_dim), self.belief_dim
        )
        transition_hidden = max(self.belief_dim, 2 * self.key_dim)
        self.transition = nn.Sequential(
            nn.Linear(
                self.key_dim + self.belief_dim + int(action_dim),
                transition_hidden,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(transition_hidden, transition_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(transition_hidden, self.key_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.key_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 3 * self.cell_grid_size * self.cell_grid_size),
            nn.Sigmoid(),
        )
        self.inverse_action = nn.Sequential(
            nn.Linear(2 * self.key_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.action_space_size),
        )

    def encode_observations(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim < 4:
            raise ValueError("Expected observations ending in [channels, height, width].")
        leading_shape = observations.shape[:-3]
        flattened = observations.reshape(-1, *observations.shape[-3:])
        features = self.observation_encoder(flattened)
        keys = F.normalize(self.key_projection(features), dim=-1)
        return keys.reshape(*leading_shape, self.key_dim)

    def encode_trajectory(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> EncodedTrajectory:
        if observations.ndim != 5 or actions.ndim != 2:
            raise ValueError("Expected batched observation and action sequences.")
        if observations.shape[1] != actions.shape[1] + 1:
            raise ValueError("Expected exactly one more observation than action.")
        keys = self.encode_observations(observations)
        batch_size, observation_count = keys.shape[:2]
        belief = torch.zeros(
            batch_size,
            self.belief_dim,
            device=keys.device,
            dtype=keys.dtype,
        )
        start_action = torch.full(
            (batch_size,),
            self.action_space_size,
            device=actions.device,
            dtype=torch.long,
        )
        beliefs = []
        for step in range(observation_count):
            previous_action = start_action if step == 0 else actions[:, step - 1]
            recurrent_input = torch.cat(
                [keys[:, step], self.action_embedding(previous_action)], dim=-1
            )
            belief = self.belief_cell(recurrent_input, belief)
            beliefs.append(belief)
        return EncodedTrajectory(keys=keys, beliefs=torch.stack(beliefs, dim=1))

    def predict_next(
        self,
        key: torch.Tensor,
        belief: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        action_embedding = self.action_embedding(action.long())
        delta = self.transition(
            torch.cat([key, belief, action_embedding], dim=-1)
        )
        next_key = F.normalize(delta, dim=-1)
        next_belief = self.belief_cell(
            torch.cat([next_key, action_embedding], dim=-1), belief
        )
        return next_key, next_belief

    def rollout(
        self,
        key: torch.Tensor,
        belief: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        if actions.ndim != 2:
            raise ValueError("Expected rollout actions shaped [sample, horizon].")
        predictions = []
        for depth in range(actions.shape[1]):
            key, belief = self.predict_next(key, belief, actions[:, depth])
            predictions.append(key)
        return torch.stack(predictions, dim=1)

    def decode_cells(self, keys: torch.Tensor) -> torch.Tensor:
        leading_shape = keys.shape[:-1]
        decoded = self.decoder(keys.reshape(-1, self.key_dim))
        return decoded.reshape(
            *leading_shape, 3, self.cell_grid_size, self.cell_grid_size
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            ".local_runs/branching_rooms_action_novelty_seed0/"
            "experienced_trajectories.npz"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".local_runs/branching_rooms_exploration_world_model_seed0"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--key-pretrain-epochs", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--evaluation-batch-size", type=int, default=4)
    parser.add_argument("--roots-per-episode", type=int, default=32)
    parser.add_argument("--encoder-dim", type=int, default=64)
    parser.add_argument("--key-dim", type=int, default=64)
    parser.add_argument("--belief-dim", type=int, default=128)
    parser.add_argument("--action-dim", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--dynamics-weight", type=float, default=1.0)
    parser.add_argument("--direct-reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--predicted-reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--contrastive-weight", type=float, default=1.0)
    parser.add_argument("--inverse-action-weight", type=float, default=0.1)
    parser.add_argument("--variance-weight", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--smoke", action="store_true")
    return parser


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device(name)


def _smoke_overrides(args: argparse.Namespace) -> None:
    if not args.smoke:
        return
    args.key_pretrain_epochs = 2
    args.epochs = 2
    args.patience = 2
    args.batch_size = 2
    args.evaluation_batch_size = 2
    args.roots_per_episode = 4
    args.encoder_dim = 16
    args.key_dim = 16
    args.belief_dim = 32
    args.action_dim = 8


def cell_targets(observations: torch.Tensor, grid_size: int = 7) -> torch.Tensor:
    observations = observations.float().div(255.0)
    flattened = observations.reshape(-1, *observations.shape[-3:])
    pooled = F.adaptive_avg_pool2d(flattened, (int(grid_size), int(grid_size)))
    return pooled.reshape(*observations.shape[:-3], *pooled.shape[-3:])


def sample_roots(
    action_count: int,
    horizon: int,
    requested_count: int,
    device: torch.device,
) -> torch.Tensor:
    root_count = int(action_count) - int(horizon) + 1
    if root_count <= 0:
        raise ValueError("The horizon exceeds the trajectory length.")
    count = min(int(requested_count), root_count)
    return torch.randperm(root_count, device=device)[:count].sort().values


def gather_rollout_batch(
    encoded: EncodedTrajectory,
    actions: torch.Tensor,
    roots: torch.Tensor,
    horizon: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = actions.shape[0]
    root_keys = encoded.keys[:, roots].reshape(-1, encoded.keys.shape[-1])
    root_beliefs = encoded.beliefs[:, roots].reshape(
        -1, encoded.beliefs.shape[-1]
    )
    suffix_actions = torch.stack(
        [actions[:, roots + depth] for depth in range(int(horizon))], dim=2
    ).reshape(batch_size * len(roots), int(horizon))
    target_indices = torch.stack(
        [roots + depth + 1 for depth in range(int(horizon))], dim=1
    )
    target_keys = encoded.keys[:, target_indices].reshape(
        batch_size * len(roots), int(horizon), encoded.keys.shape[-1]
    )
    return root_keys, root_beliefs, suffix_actions, target_keys


def multipositive_contrastive_loss(
    predicted: torch.Tensor,
    targets: torch.Tensor,
    target_ids: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Contrast predictions against same-episode targets without false negatives."""
    if predicted.ndim != 4 or targets.shape != predicted.shape:
        raise ValueError("Expected predicted and target keys shaped [B, R, H, K].")
    losses = []
    for batch in range(predicted.shape[0]):
        for depth in range(predicted.shape[2]):
            logits = (
                predicted[batch, :, depth]
                @ targets[batch, :, depth].detach().T
            ) / float(temperature)
            positives = (
                target_ids[batch, :, depth, None]
                == target_ids[batch, None, :, depth]
            )
            positive_logits = logits.masked_fill(~positives, -torch.inf)
            losses.append(
                -(torch.logsumexp(positive_logits, dim=1) - torch.logsumexp(logits, dim=1)).mean()
            )
    return torch.stack(losses).mean()


def direct_key_contrastive_loss(
    keys: torch.Tensor,
    state_ids: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Separate different RGB observations before dynamics can shape the keys."""
    if keys.ndim != 3 or state_ids.shape != keys.shape[:2]:
        raise ValueError("Expected keys [B, T, K] and state IDs [B, T].")
    losses = []
    for batch in range(keys.shape[0]):
        logits = (keys[batch] @ keys[batch].detach().T) / float(temperature)
        positives = state_ids[batch, :, None] == state_ids[batch, None, :]
        positive_logits = logits.masked_fill(~positives, -torch.inf)
        losses.append(
            -(
                torch.logsumexp(positive_logits, dim=1)
                - torch.logsumexp(logits, dim=1)
            ).mean()
        )
    return torch.stack(losses).mean()


def variance_loss(keys: torch.Tensor, target_std: float = 0.08) -> torch.Tensor:
    flattened = keys.reshape(-1, keys.shape[-1])
    standard_deviation = torch.sqrt(flattened.var(dim=0, unbiased=False) + 1e-4)
    return F.relu(float(target_std) - standard_deviation).mean()


def key_pretraining_loss(
    model: ExplorationWorldModel,
    batch: Mapping[str, torch.Tensor],
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Mapping[str, float]]:
    observations = batch["observations"]
    actions = batch["actions"]
    state_ids = batch["state_ids"]
    keys = model.encode_observations(observations)
    reconstruction = F.mse_loss(
        model.decode_cells(keys), cell_targets(observations)
    )
    contrastive = direct_key_contrastive_loss(
        keys, state_ids, args.temperature
    )
    inverse_logits = model.inverse_action(
        torch.cat([keys[:, :-1], keys[:, 1:]], dim=-1)
    )
    inverse_action = F.cross_entropy(
        inverse_logits.reshape(-1, model.action_space_size), actions.reshape(-1)
    )
    key_variance = variance_loss(keys)
    total = (
        float(args.direct_reconstruction_weight) * reconstruction
        + float(args.contrastive_weight) * contrastive
        + float(args.inverse_action_weight) * inverse_action
        + float(args.variance_weight) * key_variance
    )
    return total, {
        "loss": float(total.detach().item()),
        "reconstruction": float(reconstruction.detach().item()),
        "contrastive": float(contrastive.detach().item()),
        "inverse_action": float(inverse_action.detach().item()),
        "variance": float(key_variance.detach().item()),
    }


def key_geometry_metrics(
    model: ExplorationWorldModel,
    loader: DataLoader,
    device: torch.device,
) -> Mapping[str, float]:
    model.eval()
    reconstruction_losses = []
    maximum_different_cosines = []
    with torch.no_grad():
        for batch in loader:
            observations = batch["observations"].to(device)
            state_ids = batch["state_ids"].to(device)
            keys = model.encode_observations(observations)
            reconstruction_losses.append(
                F.mse_loss(
                    model.decode_cells(keys), cell_targets(observations)
                ).item()
            )
            similarities = torch.einsum("bik,bjk->bij", keys, keys)
            same = state_ids[:, :, None] == state_ids[:, None, :]
            similarities = similarities.masked_fill(same, -torch.inf)
            maximum = similarities.max(dim=-1).values
            maximum_different_cosines.append(
                maximum[torch.isfinite(maximum)].cpu().numpy()
            )
    maximum = np.concatenate(maximum_different_cosines)
    return {
        "reconstruction_mse": float(np.mean(reconstruction_losses)),
        "maximum_different_rgb_cosine_mean": float(maximum.mean()),
        "maximum_different_rgb_cosine_p95": float(np.quantile(maximum, 0.95)),
        "minimum_different_rgb_margin_p05": float(
            np.quantile(1.0 - maximum, 0.05)
        ),
    }


def freeze_observation_key(model: ExplorationWorldModel) -> None:
    for module in (
        model.observation_encoder,
        model.key_projection,
        model.decoder,
        model.inverse_action,
    ):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)


def training_loss(
    model: ExplorationWorldModel,
    batch: Mapping[str, torch.Tensor],
    roots: torch.Tensor,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Mapping[str, float]]:
    observations = batch["observations"]
    actions = batch["actions"]
    state_ids = batch["state_ids"]
    encoded = model.encode_trajectory(observations, actions)
    root_keys, root_beliefs, suffix_actions, target_keys = gather_rollout_batch(
        encoded, actions, roots, args.horizon
    )
    predicted_keys = model.rollout(root_keys, root_beliefs, suffix_actions)
    batch_size = observations.shape[0]
    root_count = len(roots)
    target_indices = torch.stack(
        [roots + depth + 1 for depth in range(int(args.horizon))], dim=1
    )
    target_ids = state_ids[:, target_indices]
    selected_observations = observations[:, target_indices]
    selected_cells = cell_targets(selected_observations).reshape(
        batch_size * root_count,
        int(args.horizon),
        3,
        model.cell_grid_size,
        model.cell_grid_size,
    )

    dynamics = (1.0 - F.cosine_similarity(predicted_keys, target_keys.detach(), dim=-1)).mean()
    predicted_reconstruction = F.mse_loss(
        model.decode_cells(predicted_keys), selected_cells
    )
    contrastive = multipositive_contrastive_loss(
        predicted_keys.reshape(
            batch_size, root_count, int(args.horizon), model.key_dim
        ),
        target_keys.reshape(
            batch_size, root_count, int(args.horizon), model.key_dim
        ),
        target_ids,
        args.temperature,
    )
    total = (
        float(args.dynamics_weight) * dynamics
        + float(args.predicted_reconstruction_weight) * predicted_reconstruction
        + float(args.contrastive_weight) * contrastive
    )
    components = {
        "loss": float(total.detach().item()),
        "dynamics": float(dynamics.detach().item()),
        "predicted_reconstruction": float(predicted_reconstruction.detach().item()),
        "contrastive": float(contrastive.detach().item()),
    }
    return total, components


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positive = int(labels.sum())
    negative = int(labels.size - positive)
    if positive == 0 or negative == 0:
        return None
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    rank_sum = float(ranks[labels == 1].sum())
    return (rank_sum - positive * (positive + 1) / 2.0) / (
        positive * negative
    )


def binary_metrics(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> Mapping[str, float | int | None]:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(scores) >= float(threshold)
    true_positive = int(np.sum((labels == 1) & predictions))
    false_positive = int(np.sum((labels == 0) & predictions))
    false_negative = int(np.sum((labels == 1) & ~predictions))
    true_negative = int(np.sum((labels == 0) & ~predictions))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    return {
        "count": int(labels.size),
        "prevalence": float(labels.mean()) if labels.size else 0.0,
        "threshold": float(threshold),
        "accuracy": float(np.mean(predictions == labels)) if labels.size else 0.0,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2 * precision * recall / max(precision + recall, 1e-12)),
        "auroc": _binary_auc(labels, scores),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
    }


def choose_revisit_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.size == 0:
        return 1.0
    quantiles = np.linspace(0.0, 1.0, min(501, labels.size))
    candidates = np.unique(np.quantile(scores, quantiles))
    best_threshold = float(candidates[0])
    best_f1 = -1.0
    for threshold in candidates:
        f1 = float(binary_metrics(labels, scores, float(threshold))["f1"])
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold


@dataclass
class EvaluationArrays:
    alignment: Dict[int, List[np.ndarray]]
    retrieval_top1: Dict[int, List[np.ndarray]]
    retrieval_top5: Dict[int, List[np.ndarray]]
    pose_top1: Dict[int, List[np.ndarray]]
    copy_root_top1: Dict[int, List[np.ndarray]]
    random_candidate_top1: Dict[int, List[np.ndarray]]
    changed_rgb_top1: Dict[int, List[np.ndarray]]
    revisit_scores: Dict[int, List[np.ndarray]]
    revisit_labels: Dict[int, List[np.ndarray]]
    revisit_classes: Dict[int, List[np.ndarray]]


def _empty_evaluation(horizon: int) -> EvaluationArrays:
    return EvaluationArrays(
        alignment={depth: [] for depth in range(1, horizon + 1)},
        retrieval_top1={depth: [] for depth in range(1, horizon + 1)},
        retrieval_top5={depth: [] for depth in range(1, horizon + 1)},
        pose_top1={depth: [] for depth in range(1, horizon + 1)},
        copy_root_top1={depth: [] for depth in range(1, horizon + 1)},
        random_candidate_top1={depth: [] for depth in range(1, horizon + 1)},
        changed_rgb_top1={depth: [] for depth in range(1, horizon + 1)},
        revisit_scores={depth: [] for depth in range(1, horizon + 1)},
        revisit_labels={depth: [] for depth in range(1, horizon + 1)},
        revisit_classes={depth: [] for depth in range(1, horizon + 1)},
    )


def _append_batch_evaluation(
    arrays: EvaluationArrays,
    predicted: torch.Tensor,
    encoded: EncodedTrajectory,
    state_ids: torch.Tensor,
    poses: torch.Tensor,
    novelty_targets: torch.Tensor,
) -> None:
    batch_size, root_count, horizon, _ = predicted.shape
    all_keys = encoded.keys
    similarity = torch.einsum("brhk,btk->brht", predicted, all_keys)
    top_count = min(5, all_keys.shape[1])
    nearest = similarity.argmax(dim=-1)
    top_indices = similarity.topk(top_count, dim=-1).indices

    root_indices = torch.arange(root_count, device=predicted.device)
    for depth_index in range(horizon):
        depth = depth_index + 1
        endpoint_indices = root_indices + depth
        targets = encoded.keys[:, endpoint_indices]
        alignment = F.cosine_similarity(
            predicted[:, :, depth_index], targets, dim=-1
        )
        target_ids = state_ids[:, endpoint_indices]
        root_ids = state_ids[:, :root_count]
        nearest_ids = torch.gather(state_ids, 1, nearest[:, :, depth_index])
        candidate_ids = state_ids[:, None, :].expand(batch_size, root_count, -1)
        top_ids = torch.gather(
            candidate_ids,
            2,
            top_indices[:, :, depth_index],
        )
        top1 = nearest_ids == target_ids
        top5 = (top_ids == target_ids.unsqueeze(-1)).any(dim=-1)
        copy_root = root_ids == target_ids
        random_candidate = (
            state_ids[:, None, :] == target_ids[:, :, None]
        ).float().mean(dim=-1)
        candidate_poses = poses[:, None, :, :].expand(
            batch_size, root_count, poses.shape[1], poses.shape[2]
        )
        nearest_pose = torch.gather(
            candidate_poses,
            2,
            nearest[:, :, depth_index, None, None].expand(
                batch_size, root_count, 1, poses.shape[2]
            ),
        ).squeeze(2)
        target_pose = poses[:, endpoint_indices]
        pose_match = (nearest_pose == target_pose).all(dim=-1)

        revisit_score_parts = []
        for root in range(root_count):
            root_memory = all_keys[:, : root + 1]
            if depth_index > 0:
                branch_memory = predicted[:, root, :depth_index]
                memory = torch.cat([root_memory, branch_memory], dim=1)
            else:
                memory = root_memory
            query = predicted[:, root, depth_index].unsqueeze(1)
            revisit_score_parts.append(
                F.cosine_similarity(query, memory, dim=-1).max(dim=1).values
            )
        revisit_scores = torch.stack(revisit_score_parts, dim=1)
        classes = novelty_targets[:, :root_count, depth_index]

        arrays.alignment[depth].append(alignment.detach().cpu().numpy())
        arrays.retrieval_top1[depth].append(top1.detach().cpu().numpy())
        arrays.retrieval_top5[depth].append(top5.detach().cpu().numpy())
        arrays.pose_top1[depth].append(pose_match.detach().cpu().numpy())
        arrays.copy_root_top1[depth].append(copy_root.detach().cpu().numpy())
        arrays.random_candidate_top1[depth].append(
            random_candidate.detach().cpu().numpy()
        )
        arrays.changed_rgb_top1[depth].append(
            top1[~copy_root].detach().cpu().numpy()
        )
        arrays.revisit_scores[depth].append(revisit_scores.detach().cpu().numpy())
        arrays.revisit_labels[depth].append(
            (classes != NOVEL).detach().cpu().numpy()
        )
        arrays.revisit_classes[depth].append(classes.detach().cpu().numpy())


def evaluate_model(
    model: ExplorationWorldModel,
    loader: DataLoader,
    device: torch.device,
    horizon: int,
    revisit_threshold: float | None = None,
) -> Tuple[Mapping[str, object], np.ndarray, np.ndarray]:
    model.eval()
    arrays = _empty_evaluation(int(horizon))
    with torch.no_grad():
        for batch in loader:
            observations = batch["observations"].to(device)
            actions = batch["actions"].to(device)
            state_ids = batch["state_ids"].to(device)
            poses = batch["poses"].to(device)
            novelty_targets = batch["novelty_targets"].to(device)
            encoded = model.encode_trajectory(observations, actions)
            root_count = actions.shape[1] - int(horizon) + 1
            roots = torch.arange(root_count, device=device)
            root_keys, root_beliefs, suffix_actions, _ = gather_rollout_batch(
                encoded, actions, roots, int(horizon)
            )
            predicted = model.rollout(
                root_keys, root_beliefs, suffix_actions
            ).reshape(
                observations.shape[0], root_count, int(horizon), model.key_dim
            )
            _append_batch_evaluation(
                arrays,
                predicted,
                encoded,
                state_ids,
                poses,
                novelty_targets,
            )

    all_scores = np.concatenate(
        [np.concatenate(arrays.revisit_scores[d]).reshape(-1) for d in arrays.revisit_scores]
    )
    all_labels = np.concatenate(
        [np.concatenate(arrays.revisit_labels[d]).reshape(-1) for d in arrays.revisit_labels]
    ).astype(np.int64)
    if revisit_threshold is None:
        revisit_threshold = choose_revisit_threshold(all_labels, all_scores)

    by_depth: MutableMapping[str, object] = {}
    for depth in range(1, int(horizon) + 1):
        alignment = np.concatenate(arrays.alignment[depth]).reshape(-1)
        top1 = np.concatenate(arrays.retrieval_top1[depth]).reshape(-1)
        top5 = np.concatenate(arrays.retrieval_top5[depth]).reshape(-1)
        pose_top1 = np.concatenate(arrays.pose_top1[depth]).reshape(-1)
        copy_root = np.concatenate(arrays.copy_root_top1[depth]).reshape(-1)
        random_candidate = np.concatenate(
            arrays.random_candidate_top1[depth]
        ).reshape(-1)
        changed_rgb_top1 = np.concatenate(
            arrays.changed_rgb_top1[depth]
        ).reshape(-1)
        scores = np.concatenate(arrays.revisit_scores[depth]).reshape(-1)
        labels = np.concatenate(arrays.revisit_labels[depth]).reshape(-1).astype(
            np.int64
        )
        classes = np.concatenate(arrays.revisit_classes[depth]).reshape(-1)
        revisit = dict(binary_metrics(labels, scores, revisit_threshold))
        predictions = scores >= float(revisit_threshold)
        for class_id, name in (
            (ROOT_HISTORY_REVISIT, "root_history_recall"),
            (BRANCH_REVISIT, "branch_revisit_recall"),
        ):
            selected = classes == class_id
            revisit[name] = (
                float(predictions[selected].mean()) if np.any(selected) else None
            )
            revisit[name.replace("recall", "count")] = int(selected.sum())
        by_depth[str(depth)] = {
            "count": int(alignment.size),
            "alignment_cosine_mean": float(alignment.mean()),
            "alignment_cosine_median": float(np.median(alignment)),
            "rgb_retrieval_top1": float(top1.mean()),
            "rgb_retrieval_top5": float(top5.mean()),
            "copy_root_rgb_top1_baseline": float(copy_root.mean()),
            "random_candidate_rgb_top1_baseline": float(random_candidate.mean()),
            "changed_rgb_count": int(changed_rgb_top1.size),
            "changed_rgb_retrieval_top1": float(changed_rgb_top1.mean()),
            "pose_retrieval_top1_diagnostic": float(pose_top1.mean()),
            "revisit": revisit,
        }
    return {
        "revisit_threshold": float(revisit_threshold),
        "revisit_overall": binary_metrics(
            all_labels, all_scores, float(revisit_threshold)
        ),
        "by_depth": by_depth,
    }, all_labels, all_scores


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean_components(rows: Sequence[Mapping[str, float]]) -> Mapping[str, float]:
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    }


def _loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
    )


def main() -> None:
    args = build_parser().parse_args()
    _smoke_overrides(args)
    if not args.dataset.is_file():
        raise FileNotFoundError(
            f"Missing executed-trajectory dataset: {args.dataset}. "
            "Run experiments/run_branching_rooms_action_novelty.py --mode collect first."
        )
    if args.horizon <= 0:
        raise ValueError("--horizon must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    bundle = load_dataset(args.dataset)
    if int(bundle.metadata["horizon"]) != int(args.horizon):
        raise ValueError(
            "The dataset novelty targets were built for horizon "
            f"{bundle.metadata['horizon']}, not {args.horizon}."
        )
    maximum_episodes = 4 if args.smoke else None
    train_dataset = ExplorationEpisodeDataset(bundle, 0, maximum_episodes)
    validation_dataset = ExplorationEpisodeDataset(bundle, 1, maximum_episodes)
    test_dataset = ExplorationEpisodeDataset(bundle, 2, maximum_episodes)
    heldout_dataset = ExplorationEpisodeDataset(bundle, 3, maximum_episodes)
    train_loader = _loader(
        train_dataset, args.batch_size, True, args.num_workers
    )
    validation_loader = _loader(
        validation_dataset,
        args.evaluation_batch_size,
        False,
        args.num_workers,
    )
    test_loader = _loader(
        test_dataset,
        args.evaluation_batch_size,
        False,
        args.num_workers,
    )
    heldout_loader = _loader(
        heldout_dataset,
        args.evaluation_batch_size,
        False,
        args.num_workers,
    )
    model = ExplorationWorldModel(
        encoder_dim=args.encoder_dim,
        key_dim=args.key_dim,
        belief_dim=args.belief_dim,
        action_dim=args.action_dim,
    ).to(device)
    key_optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best.pt"
    key_history_rows = []
    history_rows = []
    best_depth5_top1 = -1.0
    best_epoch = -1
    stale_epochs = 0

    arguments = vars(args).copy()
    arguments["dataset"] = str(args.dataset)
    arguments["output_dir"] = str(args.output_dir)
    arguments["device_resolved"] = str(device)
    arguments["counterfactual_transition_queries"] = 0
    (args.output_dir / "arguments.json").write_text(
        json.dumps(arguments, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"device={device} train/validation/test/heldout episodes="
        f"{len(train_dataset)}/{len(validation_dataset)}/"
        f"{len(test_dataset)}/{len(heldout_dataset)}"
    )

    print("pretraining observation key")
    for epoch in range(int(args.key_pretrain_epochs)):
        model.train()
        component_rows = []
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            loss, components = key_pretraining_loss(model, batch, args)
            key_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            key_optimizer.step()
            component_rows.append(components)
        components = _mean_components(component_rows)
        geometry = key_geometry_metrics(model, validation_loader, device)
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in components.items()},
            **{f"validation_{key}": value for key, value in geometry.items()},
        }
        key_history_rows.append(row)
        print(
            f"key epoch {epoch:02d} loss={components['loss']:.4f} "
            f"recon={geometry['reconstruction_mse']:.4f} "
            f"different_cos_p95={geometry['maximum_different_rgb_cosine_p95']:.3f}"
        )
    _write_csv(args.output_dir / "key_pretraining.csv", key_history_rows)
    freeze_observation_key(model)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    print("training action-conditioned dynamics with frozen observation keys")

    for epoch in range(int(args.epochs)):
        model.train()
        component_rows = []
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            roots = sample_roots(
                batch["actions"].shape[1],
                args.horizon,
                args.roots_per_episode,
                device,
            )
            loss, components = training_loss(model, batch, roots, args)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            component_rows.append(components)
        train_components = _mean_components(component_rows)
        validation_metrics, _, _ = evaluate_model(
            model, validation_loader, device, args.horizon
        )
        depth5_top1 = float(
            validation_metrics["by_depth"][str(args.horizon)][
                "changed_rgb_retrieval_top1"
            ]
        )
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_components.items()},
            "validation_depth1_top1": validation_metrics["by_depth"]["1"][
                "changed_rgb_retrieval_top1"
            ],
            "validation_depth5_top1": depth5_top1,
            "validation_revisit_auroc": validation_metrics["revisit_overall"][
                "auroc"
            ],
        }
        history_rows.append(row)
        print(
            f"epoch {epoch:02d} loss={train_components['loss']:.4f} "
            f"retrieval d1={row['validation_depth1_top1']:.3f} "
            f"d{args.horizon}={depth5_top1:.3f} "
            f"revisit_auc={row['validation_revisit_auroc']:.3f}"
        )
        if depth5_top1 > best_depth5_top1:
            best_depth5_top1 = depth5_top1
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "arguments": arguments,
                    "validation_metrics": validation_metrics,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= int(args.patience):
                print(f"early stopping after epoch {epoch}")
                break

    _write_csv(args.output_dir / "training.csv", history_rows)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    validation_metrics, validation_labels, validation_scores = evaluate_model(
        model, validation_loader, device, args.horizon
    )
    threshold = choose_revisit_threshold(validation_labels, validation_scores)
    test_metrics, _, _ = evaluate_model(
        model, test_loader, device, args.horizon, threshold
    )
    heldout_metrics, _, _ = evaluate_model(
        model, heldout_loader, device, args.horizon, threshold
    )
    results = {
        "best_epoch": int(best_epoch),
        "selection_metric": "validation_depth5_changed_rgb_retrieval_top1",
        "best_validation_depth5_top1": float(best_depth5_top1),
        "revisit_threshold_selected_on_validation": float(threshold),
        "validation": validation_metrics,
        "test": test_metrics,
        "heldout_policy": heldout_metrics,
        "acceptance_targets": {
            "depth1_rgb_retrieval_top1": 0.90,
            "depth5_rgb_retrieval_top1": 0.70,
            "depth1_changed_rgb_retrieval_top1": 0.85,
            "depth5_changed_rgb_retrieval_top1": 0.65,
            "revisit_auroc": 0.90,
        },
        "training_target_source": "executed_observation_action_trajectories_only",
        "physical_pose_used_for_training": False,
        "counterfactual_transition_queries": 0,
    }
    result_path = args.output_dir / "metrics.json"
    result_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"best epoch: {best_epoch}")
    for split_name, metrics in (("test", test_metrics), ("heldout", heldout_metrics)):
        print(f"{split_name}:")
        for depth, depth_metrics in metrics["by_depth"].items():
            print(
                f"  depth {depth}: cos={depth_metrics['alignment_cosine_mean']:.3f} "
                f"rgb_top1={depth_metrics['rgb_retrieval_top1']:.3f} "
                f"changed_top1={depth_metrics['changed_rgb_retrieval_top1']:.3f} "
                f"copy={depth_metrics['copy_root_rgb_top1_baseline']:.3f} "
                f"rgb_top5={depth_metrics['rgb_retrieval_top5']:.3f} "
                f"revisit_auc={depth_metrics['revisit']['auroc']:.3f} "
                f"revisit_f1={depth_metrics['revisit']['f1']:.3f}"
            )
    print(f"metrics: {result_path}")


if __name__ == "__main__":
    main()
