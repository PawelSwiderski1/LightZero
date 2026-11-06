#!/usr/bin/env python3
"""Train a spatial reconstructive exploration world model in Branching Rooms.

The model is deliberately separate from MuZero. It first learns an
observation-grounded spatial latent, then learns action-conditioned dynamics on
executed trajectories only. The representation can either remain frozen during
dynamics training or continue learning jointly as an ablation. Physical
positions are used exclusively for diagnostics, never as training targets.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

if __package__:
    from .run_branching_rooms_action_novelty import NOVEL, DatasetBundle, load_dataset
    from .run_branching_rooms_exploration_world_model import (
        ExplorationEpisodeDataset,
        direct_key_contrastive_loss,
        multipositive_contrastive_loss,
        sample_roots,
        variance_loss,
    )
else:
    from run_branching_rooms_action_novelty import NOVEL, DatasetBundle, load_dataset
    from run_branching_rooms_exploration_world_model import (
        ExplorationEpisodeDataset,
        direct_key_contrastive_loss,
        multipositive_contrastive_loss,
        sample_roots,
        variance_loss,
    )


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if int(channels) % groups == 0:
            return groups
    return 1


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = _group_count(channels)
        self.network = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.silu(inputs + self.network(inputs))


@dataclass(frozen=True)
class SpatialEncodedTrajectory:
    latents: torch.Tensor
    observation_keys: torch.Tensor
    keys: torch.Tensor
    beliefs: torch.Tensor


@dataclass(frozen=True)
class SpatialRollout:
    latents: torch.Tensor
    observation_keys: torch.Tensor
    keys: torch.Tensor
    beliefs: torch.Tensor


class SpatialExplorationWorldModel(nn.Module):
    """Observation decoder plus action-conditioned convolutional dynamics."""

    def __init__(
        self,
        observation_shape: Sequence[int] = (3, 28, 28),
        action_space_size: int = 3,
        spatial_channels: int = 32,
        key_dim: int = 64,
        belief_dim: int = 128,
        action_dim: int = 16,
        representation_mode: str = "spatial_reconstruction",
        coarse_reconstruction_size: int = 7,
        memory_key_mode: str = "observation",
    ) -> None:
        super().__init__()
        self.observation_shape = tuple(int(value) for value in observation_shape)
        if len(self.observation_shape) != 3:
            raise ValueError("observation_shape must be [channels, height, width]")
        if self.observation_shape[1] % 4 or self.observation_shape[2] % 4:
            raise ValueError("observation height and width must be divisible by four")
        self.action_space_size = int(action_space_size)
        self.spatial_channels = int(spatial_channels)
        self.key_dim = int(key_dim)
        self.belief_dim = int(belief_dim)
        self.action_dim = int(action_dim)
        self.representation_mode = str(representation_mode)
        if self.representation_mode not in (
            "spatial_reconstruction",
            "compressed_key_reconstruction",
            "inverse_dynamics",
        ):
            raise ValueError(
                f"Unknown representation mode: {self.representation_mode}"
            )
        self.coarse_reconstruction_size = int(coarse_reconstruction_size)
        if self.coarse_reconstruction_size <= 0:
            raise ValueError("coarse_reconstruction_size must be positive")
        middle_channels = max(8, self.spatial_channels // 2)
        groups = _group_count(self.spatial_channels)

        self.encoder = nn.Sequential(
            nn.Conv2d(
                self.observation_shape[0],
                middle_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(_group_count(middle_channels), middle_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                middle_channels,
                self.spatial_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(groups, self.spatial_channels),
            nn.SiLU(inplace=True),
            ResidualConvBlock(self.spatial_channels),
            ResidualConvBlock(self.spatial_channels),
        )
        self.latent_height = self.observation_shape[1] // 4
        self.latent_width = self.observation_shape[2] // 4
        if self.representation_mode == "spatial_reconstruction":
            self.decoder = nn.Sequential(
                ResidualConvBlock(self.spatial_channels),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(
                    self.spatial_channels,
                    middle_channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.GroupNorm(_group_count(middle_channels), middle_channels),
                nn.SiLU(inplace=True),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(
                    middle_channels,
                    self.observation_shape[0],
                    kernel_size=3,
                    padding=1,
                ),
                nn.Sigmoid(),
            )
            self.key_decoder = None
        elif self.representation_mode == "compressed_key_reconstruction":
            self.decoder = None
            decoder_hidden_dim = max(128, 4 * self.key_dim)
            self.key_decoder = nn.Sequential(
                nn.Linear(self.key_dim, decoder_hidden_dim),
                nn.SiLU(inplace=True),
                nn.Linear(
                    decoder_hidden_dim,
                    self.observation_shape[0]
                    * self.coarse_reconstruction_size
                    * self.coarse_reconstruction_size,
                ),
                nn.Sigmoid(),
            )
        else:
            self.decoder = None
            self.key_decoder = None
        self.key_projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                self.spatial_channels * self.latent_height * self.latent_width,
                self.key_dim,
            ),
            nn.LayerNorm(self.key_dim),
        )
        self.memory_key_mode = str(memory_key_mode)
        if self.memory_key_mode not in (
            "observation",
            "belief",
            "belief_to_observation",
            "belief_revisit",
        ):
            raise ValueError(
                "memory_key_mode must be 'observation', 'belief', "
                "'belief_to_observation', or 'belief_revisit'"
            )
        self.belief_key_projection = (
            nn.Sequential(
                nn.Linear(self.belief_dim, self.key_dim),
                nn.LayerNorm(self.key_dim),
            )
            if self.memory_key_mode in (
                "belief",
                "belief_to_observation",
                "belief_revisit",
            )
            else None
        )
        self.belief_decoder_projection = (
            nn.Linear(
                self.belief_dim,
                self.spatial_channels * self.latent_height * self.latent_width,
            )
            if self.memory_key_mode in (
                "belief",
                "belief_to_observation",
                "belief_revisit",
            )
            and self.representation_mode == "spatial_reconstruction"
            else None
        )
        revisit_pair_dim = max(32, self.belief_dim // 2)
        self.revisit_query_projection = (
            nn.Sequential(
                nn.Linear(self.belief_dim, revisit_pair_dim),
                nn.LayerNorm(revisit_pair_dim),
                nn.SiLU(inplace=True),
            )
            if self.memory_key_mode == "belief_revisit"
            else None
        )
        self.revisit_memory_projection = (
            nn.Sequential(
                nn.Linear(self.belief_dim, revisit_pair_dim),
                nn.LayerNorm(revisit_pair_dim),
                nn.SiLU(inplace=True),
            )
            if self.memory_key_mode == "belief_revisit"
            else None
        )
        self.revisit_pair_head = (
            nn.Sequential(
                nn.Linear(4 * revisit_pair_dim, revisit_pair_dim),
                nn.SiLU(inplace=True),
                nn.Linear(revisit_pair_dim, 1),
            )
            if self.memory_key_mode == "belief_revisit"
            else None
        )
        self.action_embedding = nn.Embedding(
            self.action_space_size + 1, self.action_dim
        )
        self.belief_cell = nn.GRUCell(
            self.key_dim + self.action_dim, self.belief_dim
        )
        self.belief_to_spatial = nn.Linear(
            self.belief_dim, self.spatial_channels
        )
        transition_channels = (
            2 * self.spatial_channels + self.action_dim
        )
        self.transition = nn.Sequential(
            nn.Conv2d(
                transition_channels,
                self.spatial_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(groups, self.spatial_channels),
            nn.SiLU(inplace=True),
            ResidualConvBlock(self.spatial_channels),
            nn.Conv2d(
                self.spatial_channels,
                self.spatial_channels,
                kernel_size=3,
                padding=1,
            ),
        )
        self.predicted_latent_norm = nn.GroupNorm(
            groups, self.spatial_channels
        )
        self.inverse_action = nn.Sequential(
            nn.Linear(2 * self.key_dim, max(64, self.key_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(max(64, self.key_dim), self.action_space_size),
        )

    def encode_observations(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim < 4:
            raise ValueError("Expected observations ending in [C,H,W]")
        leading = observations.shape[:-3]
        flattened = observations.reshape(-1, *observations.shape[-3:]).float()
        if float(flattened.detach().amax()) > 1.5:
            flattened = flattened.div(255.0)
        latent = self.encoder(flattened)
        return latent.reshape(
            *leading,
            self.spatial_channels,
            self.latent_height,
            self.latent_width,
        )

    def keys_from_latents(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.shape[-3:] != (
            self.spatial_channels,
            self.latent_height,
            self.latent_width,
        ):
            raise ValueError("Unexpected spatial latent shape")
        leading = latents.shape[:-3]
        flattened = latents.reshape(
            -1,
            self.spatial_channels,
            self.latent_height,
            self.latent_width,
        )
        keys = F.normalize(self.key_projection(flattened), dim=-1)
        return keys.reshape(*leading, self.key_dim)

    def keys_from_beliefs(self, beliefs: torch.Tensor) -> torch.Tensor:
        if self.belief_key_projection is None:
            raise RuntimeError("This model uses observation-derived memory keys")
        leading = beliefs.shape[:-1]
        keys = F.normalize(
            self.belief_key_projection(
                beliefs.reshape(-1, self.belief_dim)
            ),
            dim=-1,
        )
        return keys.reshape(*leading, self.key_dim)

    def revisit_logits(
        self,
        query_beliefs: torch.Tensor,
        memory_beliefs: torch.Tensor,
    ) -> torch.Tensor:
        """Predict whether each query belief matches each memory belief."""
        if (
            self.revisit_query_projection is None
            or self.revisit_memory_projection is None
            or self.revisit_pair_head is None
        ):
            raise RuntimeError(
                "Pairwise revisit prediction requires belief_revisit mode"
            )
        squeeze_query = query_beliefs.ndim == 1
        squeeze_memory = memory_beliefs.ndim == 2
        if squeeze_query:
            query_beliefs = query_beliefs.unsqueeze(0)
        if squeeze_memory:
            memory_beliefs = memory_beliefs.unsqueeze(0)
        if query_beliefs.shape[:-1] != memory_beliefs.shape[:-2]:
            raise ValueError(
                "Query leading dimensions must match memory leading dimensions"
            )
        if query_beliefs.shape[-1] != self.belief_dim:
            raise ValueError("Unexpected query belief dimension")
        if memory_beliefs.shape[-1] != self.belief_dim:
            raise ValueError("Unexpected memory belief dimension")
        query = self.revisit_query_projection(query_beliefs).unsqueeze(-2)
        memory = self.revisit_memory_projection(memory_beliefs)
        query = query.expand(*memory.shape[:-1], query.shape[-1])
        features = torch.cat(
            [query, memory, (query - memory).abs(), query * memory], dim=-1
        )
        logits = self.revisit_pair_head(features).squeeze(-1)
        if squeeze_query and squeeze_memory:
            return logits.squeeze(0)
        return logits

    def revisit_novelty(
        self,
        query_belief: torch.Tensor,
        memory_beliefs: torch.Tensor,
    ) -> torch.Tensor:
        """Return one minus the strongest predicted revisit probability."""
        logits = self.revisit_logits(query_belief, memory_beliefs)
        return 1.0 - torch.sigmoid(logits).amax(dim=-1)

    def real_memory_keys(
        self,
        observation_latents: torch.Tensor,
        beliefs: torch.Tensor,
    ) -> torch.Tensor:
        if self.memory_key_mode in ("belief", "belief_revisit"):
            return self.keys_from_beliefs(beliefs)
        return self.keys_from_latents(observation_latents)

    def predicted_memory_keys(
        self,
        predicted_latents: torch.Tensor,
        predicted_beliefs: torch.Tensor,
    ) -> torch.Tensor:
        if self.memory_key_mode in (
            "belief",
            "belief_to_observation",
            "belief_revisit",
        ):
            return self.keys_from_beliefs(predicted_beliefs)
        return self.keys_from_latents(predicted_latents)

    def memory_keys(
        self,
        observation_latents: torch.Tensor,
        beliefs: torch.Tensor,
    ) -> torch.Tensor:
        """Return keys stored for states reached through real observations."""
        return self.real_memory_keys(observation_latents, beliefs)

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        if self.representation_mode == "compressed_key_reconstruction":
            return self.decode_keys(self.keys_from_latents(latents))
        if self.decoder is None:
            raise RuntimeError("This representation has no observation decoder")
        leading = latents.shape[:-3]
        flattened = latents.reshape(
            -1,
            self.spatial_channels,
            self.latent_height,
            self.latent_width,
        )
        decoded = self.decoder(flattened)
        return decoded.reshape(*leading, *self.observation_shape)

    def decode_beliefs(self, beliefs: torch.Tensor) -> torch.Tensor:
        if self.memory_key_mode not in (
            "belief",
            "belief_to_observation",
            "belief_revisit",
        ):
            raise RuntimeError("This model has no belief observation decoder")
        if self.representation_mode == "compressed_key_reconstruction":
            return self.decode_keys(self.keys_from_beliefs(beliefs))
        if self.belief_decoder_projection is None or self.decoder is None:
            raise RuntimeError("This belief representation has no observation decoder")
        leading = beliefs.shape[:-1]
        projected = self.belief_decoder_projection(
            beliefs.reshape(-1, self.belief_dim)
        ).reshape(
            -1,
            self.spatial_channels,
            self.latent_height,
            self.latent_width,
        )
        decoded = self.decoder(projected)
        return decoded.reshape(*leading, *self.observation_shape)

    def decode_rollout(self, rollout: SpatialRollout) -> torch.Tensor:
        if self.memory_key_mode in (
            "belief",
            "belief_to_observation",
            "belief_revisit",
        ):
            return self.decode_beliefs(rollout.beliefs)
        return self.decode_latents(rollout.latents)

    def decode_keys(self, keys: torch.Tensor) -> torch.Tensor:
        if self.key_decoder is None:
            raise RuntimeError("This representation has no key decoder")
        leading = keys.shape[:-1]
        decoded = self.key_decoder(keys.reshape(-1, self.key_dim))
        return decoded.reshape(
            *leading,
            self.observation_shape[0],
            self.coarse_reconstruction_size,
            self.coarse_reconstruction_size,
        )

    def reconstruction_targets(self, observations: torch.Tensor) -> torch.Tensor:
        if not self.supports_reconstruction:
            raise RuntimeError("This representation has no reconstruction target")
        leading = observations.shape[:-3]
        flattened = observations.reshape(-1, *observations.shape[-3:]).float()
        if float(flattened.detach().amax()) > 1.5:
            flattened = flattened.div(255.0)
        if self.representation_mode == "compressed_key_reconstruction":
            flattened = F.adaptive_avg_pool2d(
                flattened,
                (
                    self.coarse_reconstruction_size,
                    self.coarse_reconstruction_size,
                ),
            )
        return flattened.reshape(*leading, *flattened.shape[-3:])

    @property
    def supports_reconstruction(self) -> bool:
        return self.decoder is not None or self.key_decoder is not None

    def representation_state_ids(
        self,
        observations: torch.Tensor,
        full_resolution_state_ids: torch.Tensor,
    ) -> torch.Tensor:
        if self.representation_mode != "compressed_key_reconstruction":
            return full_resolution_state_ids
        targets = self.reconstruction_targets(observations)
        quantized = targets.mul(255.0).round().to(torch.int64)
        flattened = quantized.reshape(quantized.shape[0] * quantized.shape[1], -1)
        _, inverse = torch.unique(flattened, dim=0, return_inverse=True)
        return inverse.reshape(quantized.shape[:2])

    def encode_trajectory(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> SpatialEncodedTrajectory:
        if observations.ndim != 5 or actions.ndim != 2:
            raise ValueError("Expected observations [B,T+1,C,H,W] and actions [B,T]")
        if observations.shape[1] != actions.shape[1] + 1:
            raise ValueError("Expected exactly one more observation than action")
        latents = self.encode_observations(observations)
        observation_keys = self.keys_from_latents(latents)
        batch_size, observation_count = observation_keys.shape[:2]
        belief = torch.zeros(
            batch_size,
            self.belief_dim,
            device=observation_keys.device,
            dtype=observation_keys.dtype,
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
            belief = self.belief_cell(
                torch.cat(
                    [
                        observation_keys[:, step],
                        self.action_embedding(previous_action),
                    ],
                    dim=-1,
                ),
                belief,
            )
            beliefs.append(belief)
        belief_sequence = torch.stack(beliefs, dim=1)
        return SpatialEncodedTrajectory(
            latents=latents,
            observation_keys=observation_keys,
            keys=self.real_memory_keys(latents, belief_sequence),
            beliefs=belief_sequence,
        )

    def predict_next(
        self,
        latent: torch.Tensor,
        belief: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action_vector = self.action_embedding(action.long())
        action_map = action_vector[:, :, None, None].expand(
            -1, -1, self.latent_height, self.latent_width
        )
        belief_map = self.belief_to_spatial(belief)[:, :, None, None].expand(
            -1, -1, self.latent_height, self.latent_width
        )
        delta = self.transition(torch.cat([latent, belief_map, action_map], dim=1))
        next_latent = F.silu(self.predicted_latent_norm(latent + delta))
        next_observation_key = self.keys_from_latents(next_latent)
        next_belief = self.belief_cell(
            torch.cat([next_observation_key, action_vector], dim=-1), belief
        )
        next_memory_key = self.predicted_memory_keys(next_latent, next_belief)
        return next_latent, next_observation_key, next_memory_key, next_belief

    def rollout(
        self,
        latent: torch.Tensor,
        belief: torch.Tensor,
        actions: torch.Tensor,
    ) -> SpatialRollout:
        if actions.ndim != 2:
            raise ValueError("Expected rollout actions [sample,horizon]")
        latent_predictions = []
        observation_key_predictions = []
        key_predictions = []
        belief_predictions = []
        for depth in range(actions.shape[1]):
            latent, observation_key, key, belief = self.predict_next(
                latent, belief, actions[:, depth]
            )
            latent_predictions.append(latent)
            observation_key_predictions.append(observation_key)
            key_predictions.append(key)
            belief_predictions.append(belief)
        return SpatialRollout(
            latents=torch.stack(latent_predictions, dim=1),
            observation_keys=torch.stack(observation_key_predictions, dim=1),
            keys=torch.stack(key_predictions, dim=1),
            beliefs=torch.stack(belief_predictions, dim=1),
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
        default=Path(
            ".local_runs/branching_rooms_spatial_exploration_world_model_seed0"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--autoencoder-epochs", type=int, default=20)
    parser.add_argument("--dynamics-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--evaluation-batch-size", type=int, default=4)
    parser.add_argument("--roots-per-episode", type=int, default=32)
    parser.add_argument(
        "--evaluation-roots-per-episode",
        type=int,
        default=0,
        help="Evenly sample this many evaluation roots; 0 evaluates every root.",
    )
    parser.add_argument("--spatial-channels", type=int, default=32)
    parser.add_argument("--key-dim", type=int, default=64)
    parser.add_argument("--belief-dim", type=int, default=128)
    parser.add_argument("--action-dim", type=int, default=16)
    parser.add_argument(
        "--representation-mode",
        choices=(
            "spatial_reconstruction",
            "compressed_key_reconstruction",
            "inverse_dynamics",
        ),
        default="spatial_reconstruction",
    )
    parser.add_argument(
        "--memory-key-mode",
        choices=(
            "observation",
            "belief",
            "belief_to_observation",
            "belief_revisit",
        ),
        default="observation",
        help=(
            "Use observation keys, belief keys, real observation keys with "
            "imagined beliefs projected into observation-key coordinates, or "
            "a learned pairwise revisit classifier over beliefs."
        ),
    )
    parser.add_argument("--coarse-reconstruction-size", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--direct-reconstruction-weight", type=float, default=2.0)
    parser.add_argument("--direct-contrastive-weight", type=float, default=1.0)
    parser.add_argument(
        "--belief-observation-alignment-weight",
        type=float,
        default=1.0,
        help=(
            "Weight for aligning real recurrent beliefs with the corresponding "
            "observation key in belief_to_observation mode."
        ),
    )
    parser.add_argument(
        "--pairwise-revisit-weight",
        type=float,
        default=1.0,
        help="Weight for exact-RGB-supervised belief-pair revisit prediction.",
    )
    parser.add_argument(
        "--pairwise-revisit-negative-samples",
        type=int,
        default=8,
        help="Maximum hard negative memory pairs retained per imagined query.",
    )
    parser.add_argument("--inverse-action-weight", type=float, default=0.1)
    parser.add_argument("--variance-weight", type=float, default=0.1)
    parser.add_argument("--latent-dynamics-weight", type=float, default=1.0)
    parser.add_argument("--key-dynamics-weight", type=float, default=1.0)
    parser.add_argument(
        "--observation-key-consistency-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for aligning each predicted observation key with the "
            "stop-gradient key encoded from the corresponding real future frame."
        ),
    )
    parser.add_argument(
        "--belief-dynamics-consistency-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for aligning recursively predicted beliefs with the "
            "stop-gradient beliefs encoded from the real future trajectory."
        ),
    )
    parser.add_argument("--rollout-reconstruction-weight", type=float, default=2.0)
    parser.add_argument("--rollout-contrastive-weight", type=float, default=1.0)
    parser.add_argument(
        "--predictive-equivalence-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for shaping belief-memory keys from matching action prefixes "
            "and their real observation outcomes."
        ),
    )
    parser.add_argument(
        "--predictive-equivalence-negative-margin",
        type=float,
        default=0.2,
        help="Maximum cosine similarity allowed for observed divergent outcomes.",
    )
    parser.add_argument(
        "--memory-novelty-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for matching predicted episodic-memory novelty to the novelty "
            "computed from frozen true future keys."
        ),
    )
    parser.add_argument(
        "--joint-representation-dynamics",
        action="store_true",
        help=(
            "Continue training the encoder, key projection, and decoder during "
            "stage-two dynamics training. Real future keys remain stop-gradient "
            "targets, and the stage-one representation losses remain active."
        ),
    )
    parser.add_argument(
        "--selection-metric",
        choices=(
            "rollout_reconstruction",
            "key_novelty",
            "novelty_fidelity",
        ),
        default="rollout_reconstruction",
        help=(
            "Validation objective used for checkpoint selection. key_novelty uses "
            "only RGB-novelty AUC and changed-state key retrieval, without pixel "
            "reconstruction or privileged physical-position labels. "
            "novelty_fidelity rewards predicted/teacher novelty correlation and "
            "penalizes their absolute error."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--figure-count", type=int, default=6)
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help=(
            "Resume dynamics training from a checkpoint. Stage-one representation "
            "training is skipped and --dynamics-epochs is interpreted as the total."
        ),
    )
    parser.add_argument(
        "--fresh-optimizer-on-resume",
        action="store_true",
        help="Do not restore AdamW state when adapting to a new trajectory distribution.",
    )
    parser.add_argument(
        "--reset-selection-on-resume",
        action="store_true",
        help="Select the best checkpoint using the new validation distribution only.",
    )
    parser.add_argument(
        "--evaluate-resume-before-training",
        action="store_true",
        help="Save frozen resume-checkpoint metrics on the new dataset before adaptation.",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def _smoke_overrides(args: argparse.Namespace) -> None:
    if not args.smoke:
        return
    args.autoencoder_epochs = 1
    args.dynamics_epochs = 1
    args.patience = 1
    args.batch_size = 2
    args.evaluation_batch_size = 2
    args.roots_per_episode = 3
    args.spatial_channels = 8
    args.key_dim = 16
    args.belief_dim = 32
    args.action_dim = 4
    args.figure_count = 1


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


def _mean_components(rows: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    }


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> List[Dict[str, object]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
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
    return float(
        (rank_sum - positive * (positive + 1) / 2.0) / (positive * negative)
    )


def position_first_visit_rewards(poses: np.ndarray) -> np.ndarray:
    poses = np.asarray(poses)
    if poses.ndim == 2:
        poses = poses[None]
        squeeze = True
    elif poses.ndim == 3:
        squeeze = False
    else:
        raise ValueError("poses must be [T,3] or [B,T,3]")
    result = np.zeros((poses.shape[0], poses.shape[1] - 1), dtype=np.float32)
    for batch_index, episode in enumerate(poses):
        seen = {tuple(int(value) for value in episode[0, :2])}
        for step in range(1, len(episode)):
            position = tuple(int(value) for value in episode[step, :2])
            result[batch_index, step - 1] = float(position not in seen)
            seen.add(position)
    return result[0] if squeeze else result


def gather_spatial_rollout_batch(
    encoded: SpatialEncodedTrajectory,
    observations: torch.Tensor,
    actions: torch.Tensor,
    state_ids: torch.Tensor,
    roots: torch.Tensor,
    horizon: int,
) -> Dict[str, torch.Tensor]:
    batch_size = actions.shape[0]
    root_count = int(roots.numel())
    target_indices = torch.stack(
        [roots + depth + 1 for depth in range(int(horizon))], dim=1
    )
    suffix_actions = torch.stack(
        [actions[:, roots + depth] for depth in range(int(horizon))], dim=2
    )
    return {
        "root_latents": encoded.latents[:, roots].reshape(
            batch_size * root_count,
            *encoded.latents.shape[-3:],
        ),
        "root_beliefs": encoded.beliefs[:, roots].reshape(
            batch_size * root_count, encoded.beliefs.shape[-1]
        ),
        "root_keys": encoded.keys[:, roots],
        "root_ids": state_ids[:, roots],
        "suffix_actions": suffix_actions.reshape(
            batch_size * root_count, int(horizon)
        ),
        "target_latents": encoded.latents[:, target_indices].reshape(
            batch_size * root_count,
            int(horizon),
            *encoded.latents.shape[-3:],
        ),
        "target_beliefs": encoded.beliefs[:, target_indices].reshape(
            batch_size * root_count,
            int(horizon),
            encoded.beliefs.shape[-1],
        ),
        "target_observation_keys": encoded.observation_keys[:, target_indices].reshape(
            batch_size * root_count,
            int(horizon),
            encoded.observation_keys.shape[-1],
        ),
        "target_keys": encoded.keys[:, target_indices].reshape(
            batch_size * root_count, int(horizon), encoded.keys.shape[-1]
        ),
        "target_observations": observations[:, target_indices].reshape(
            batch_size * root_count,
            int(horizon),
            *observations.shape[-3:],
        ),
        "target_ids": state_ids[:, target_indices],
        "target_indices": target_indices,
    }


def action_conditioned_predictive_equivalence_loss(
    root_keys: torch.Tensor,
    root_ids: torch.Tensor,
    suffix_actions: torch.Tensor,
    target_ids: torch.Tensor,
    negative_margin: float = 0.2,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Shape memory keys using only matched actions and real outcomes.

    A positive pair requires matching current RGB observations, a shared
    recorded action prefix, and matching real outcomes along that prefix. Any
    pair with a shared action prefix becomes negative when a real outcome
    differs. Longer positive prefixes receive more weight because they provide
    stronger evidence of predictive equivalence. No predicted belief is used
    as a target by this loss.
    """
    if root_keys.ndim != 3:
        raise ValueError("root_keys must be [B,R,K]")
    if root_ids.shape != root_keys.shape[:2]:
        raise ValueError("root_ids must match root_keys [B,R]")
    if suffix_actions.ndim != 3 or target_ids.shape != suffix_actions.shape:
        raise ValueError("suffix_actions and target_ids must be [B,R,H]")
    if suffix_actions.shape[:2] != root_keys.shape[:2]:
        raise ValueError("rollout roots must match root_keys")

    margin = float(negative_margin)
    if not -1.0 <= margin <= 1.0:
        raise ValueError("negative_margin must be in [-1, 1]")

    horizon = int(suffix_actions.shape[-1])
    keys = root_keys.reshape(-1, root_keys.shape[-1])
    roots = root_ids.reshape(-1)
    actions = suffix_actions.reshape(-1, horizon)
    outcomes = target_ids.reshape(-1, horizon)
    root_count = int(keys.shape[0])
    upper = torch.triu(
        torch.ones(root_count, root_count, dtype=torch.bool, device=keys.device),
        diagonal=1,
    )
    same_root_observation = roots[:, None].eq(roots[None, :])
    same_actions = actions[:, None, :].eq(actions[None, :, :])
    shared_prefix = same_actions.to(torch.int64).cumprod(dim=-1)
    prefix_length = shared_prefix.sum(dim=-1)
    matched_action = upper & prefix_length.gt(0)
    same_outcomes = outcomes[:, None, :].eq(outcomes[None, :, :])
    divergent = ((~same_outcomes) & shared_prefix.bool()).any(dim=-1)
    positive = matched_action & same_root_observation & ~divergent
    negative = matched_action & divergent
    aliased_negative = negative & same_root_observation
    ordinary_negative = negative & ~same_root_observation
    similarities = keys @ keys.transpose(0, 1)

    zero = root_keys.new_zeros(())
    positive_count = int(positive.sum().item())
    negative_count = int(negative.sum().item())
    positive_loss = zero
    if positive_count:
        weights = prefix_length[positive].to(keys.dtype) / max(1, horizon)
        positive_loss = (
            (1.0 - similarities[positive]) * weights
        ).sum() / weights.sum().clamp_min(1e-8)
    ordinary_negative_count = int(ordinary_negative.sum().item())
    aliased_negative_count = int(aliased_negative.sum().item())
    negative_loss = zero
    if ordinary_negative_count:
        negative_loss = F.relu(
            similarities[ordinary_negative] - margin
        ).mean()
    aliased_negative_loss = zero
    if aliased_negative_count:
        aliased_negative_loss = F.relu(
            similarities[aliased_negative] - margin
        ).mean()

    # Attraction without any contrasting examples can only compress the keys.
    # Keep such pairs in the diagnostics, but wait for a batch containing both
    # kinds of evidence before applying the positive term.
    active = positive_count > 0 and negative_count > 0
    loss_terms = [positive_loss]
    if ordinary_negative_count:
        loss_terms.append(negative_loss)
    if aliased_negative_count:
        loss_terms.append(aliased_negative_loss)
    loss = torch.stack(loss_terms).mean() if active else zero

    pair_count = positive_count + negative_count
    shared_prefix_sum = float(prefix_length[positive | negative].sum().item())
    return loss, {
        "predictive_equivalence_loss": float(loss.detach()),
        "predictive_equivalence_positive_loss": float(positive_loss.detach()),
        "predictive_equivalence_negative_loss": float(negative_loss.detach()),
        "predictive_equivalence_aliased_negative_loss": float(
            aliased_negative_loss.detach()
        ),
        "predictive_equivalence_pair_count": float(pair_count),
        "predictive_equivalence_positive_count": float(positive_count),
        "predictive_equivalence_negative_count": float(negative_count),
        "predictive_equivalence_aliased_negative_count": float(
            aliased_negative_count
        ),
        "predictive_equivalence_active": float(active),
        "predictive_equivalence_mean_prefix": (
            shared_prefix_sum / pair_count if pair_count else 0.0
        ),
    }


def autoencoder_loss(
    model: SpatialExplorationWorldModel,
    batch: Mapping[str, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, Dict[str, float]]:
    observations = batch["observations"]
    actions = batch["actions"]
    encoded = model.encode_trajectory(observations, actions)
    latents = encoded.latents
    keys = encoded.keys
    observation_keys = encoded.observation_keys
    zero = keys.new_zeros(())
    reconstruction = zero
    belief_reconstruction = zero
    belief_observation_alignment = zero
    if float(args.direct_reconstruction_weight) != 0.0:
        if not model.supports_reconstruction:
            raise ValueError(
                "A reconstruction loss was requested for a decoder-free representation"
            )
        targets = model.reconstruction_targets(observations)
        reconstruction = F.l1_loss(model.decode_latents(latents), targets)
        if model.memory_key_mode in (
            "belief",
            "belief_to_observation",
            "belief_revisit",
        ):
            belief_reconstruction = F.l1_loss(
                model.decode_beliefs(encoded.beliefs), targets
            )
            reconstruction = 0.5 * (
                reconstruction + belief_reconstruction
            )
    belief_observation_alignment_weight = float(
        getattr(args, "belief_observation_alignment_weight", 1.0)
    )
    if (
        model.memory_key_mode == "belief_to_observation"
        and belief_observation_alignment_weight != 0.0
    ):
        belief_observation_alignment = torch.mean(
            1.0
            - F.cosine_similarity(
                model.keys_from_beliefs(encoded.beliefs),
                observation_keys.detach(),
                dim=-1,
            )
        )
    contrastive = zero
    if float(args.direct_contrastive_weight) != 0.0:
        state_ids = model.representation_state_ids(
            observations, batch["state_ids"]
        )
        contrastive = direct_key_contrastive_loss(
            observation_keys, state_ids, args.temperature
        )
    inverse = zero
    if float(args.inverse_action_weight) != 0.0:
        inverse_logits = model.inverse_action(
            torch.cat(
                [observation_keys[:, :-1], observation_keys[:, 1:]],
                dim=-1,
            )
        )
        inverse = F.cross_entropy(
            inverse_logits.reshape(-1, model.action_space_size),
            actions.reshape(-1),
        )
    key_variance = zero
    if float(args.variance_weight) != 0.0:
        key_variance = variance_loss(keys)
    total = (
        float(args.direct_reconstruction_weight) * reconstruction
        + float(args.direct_contrastive_weight) * contrastive
        + belief_observation_alignment_weight * belief_observation_alignment
        + float(args.inverse_action_weight) * inverse
        + float(args.variance_weight) * key_variance
    )
    return total, {
        "loss": float(total.detach()),
        "reconstruction_mae": float(reconstruction.detach()),
        "belief_reconstruction_mae": float(
            belief_reconstruction.detach()
        ),
        "belief_observation_alignment": float(
            belief_observation_alignment.detach()
        ),
        "contrastive": float(contrastive.detach()),
        "inverse": float(inverse.detach()),
        "variance": float(key_variance.detach()),
    }


def freeze_observation_model(model: SpatialExplorationWorldModel) -> None:
    modules = [
        model.encoder,
        model.key_projection,
        model.inverse_action,
    ]
    if model.decoder is not None:
        modules.append(model.decoder)
    if model.key_decoder is not None:
        modules.append(model.key_decoder)
    for module in modules:
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)


def pairwise_revisit_prediction_loss(
    model: SpatialExplorationWorldModel,
    predicted_beliefs: torch.Tensor,
    real_beliefs: torch.Tensor,
    state_ids: torch.Tensor,
    target_ids: torch.Tensor,
    roots: torch.Tensor,
    negative_samples: int,
) -> tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
    """Supervise imagined-to-memory matches with exact RGB identity."""
    if model.memory_key_mode != "belief_revisit":
        raise ValueError("Pairwise revisit loss requires belief_revisit mode")
    if predicted_beliefs.ndim != 4:
        raise ValueError("predicted_beliefs must be [B,R,H,D]")
    batch_size, root_count, horizon, _ = predicted_beliefs.shape
    if target_ids.shape != (batch_size, root_count, horizon):
        raise ValueError("target_ids must match predicted rollout dimensions")
    maximum_negatives = max(1, int(negative_samples))
    positive_logits = []
    negative_logits = []
    novelty_errors = [
        [[] for _ in range(root_count)] for _ in range(batch_size)
    ]
    positive_count = 0
    negative_count = 0
    for root_index in range(root_count):
        root_timestep = int(roots[root_index])
        real_memory = real_beliefs[:, : root_timestep + 1]
        real_memory_ids = state_ids[:, : root_timestep + 1]
        for depth in range(horizon):
            query = predicted_beliefs[:, root_index, depth]
            query_ids = target_ids[:, root_index, depth]
            memory = real_memory
            memory_ids = real_memory_ids
            if depth > 0:
                memory = torch.cat(
                    [memory, predicted_beliefs[:, root_index, :depth]], dim=1
                )
                memory_ids = torch.cat(
                    [memory_ids, target_ids[:, root_index, :depth]], dim=1
                )
            logits = model.revisit_logits(query, memory)
            labels = query_ids[:, None].eq(memory_ids)
            for batch_index in range(batch_size):
                predicted_novelty = (
                    1.0 - torch.sigmoid(logits[batch_index]).amax()
                )
                target_novelty = (~labels[batch_index].any()).to(
                    predicted_novelty.dtype
                )
                novelty_errors[batch_index][root_index].append(
                    F.binary_cross_entropy(
                        predicted_novelty.clamp(1e-6, 1.0 - 1e-6),
                        target_novelty,
                    )
                )
                query_positive = logits[batch_index][labels[batch_index]]
                query_negative = logits[batch_index][~labels[batch_index]]
                positive_count += int(query_positive.numel())
                negative_count += int(query_negative.numel())
                if query_positive.numel():
                    positive_logits.append(query_positive)
                if query_negative.numel():
                    keep = min(
                        int(query_negative.numel()),
                        maximum_negatives
                        * max(1, int(query_positive.numel())),
                    )
                    if keep < int(query_negative.numel()):
                        query_negative = torch.topk(
                            query_negative, keep
                        ).values
                    negative_logits.append(query_negative)

    zero = predicted_beliefs.new_zeros(())
    positives = (
        torch.cat(positive_logits) if positive_logits else zero.new_empty(0)
    )
    negatives = (
        torch.cat(negative_logits) if negative_logits else zero.new_empty(0)
    )
    positive_loss = (
        F.softplus(-positives).mean() if positives.numel() else zero
    )
    negative_loss = (
        F.softplus(negatives).mean() if negatives.numel() else zero
    )
    if positives.numel() and negatives.numel():
        loss = 0.5 * (positive_loss + negative_loss)
    else:
        loss = positive_loss + negative_loss
    per_sample_root_novelty_error = torch.stack(
        [
            torch.stack(
                [
                    torch.stack(root_errors).mean()
                    if root_errors else zero
                    for root_errors in sample_errors
                ]
            )
            for sample_errors in novelty_errors
        ]
    )
    return loss, {
        "pairwise_revisit_loss": float(loss.detach()),
        "pairwise_revisit_positive_loss": float(positive_loss.detach()),
        "pairwise_revisit_negative_loss": float(negative_loss.detach()),
        "pairwise_revisit_positive_count": float(positive_count),
        "pairwise_revisit_negative_count": float(negative_count),
        "pairwise_revisit_sampled_negative_count": float(negatives.numel()),
        "pairwise_revisit_positive_probability": float(
            torch.sigmoid(positives).mean().detach()
            if positives.numel() else 0.0
        ),
        "pairwise_revisit_negative_probability": float(
            torch.sigmoid(negatives).mean().detach()
            if negatives.numel() else 0.0
        ),
    }, per_sample_root_novelty_error.detach()


def dynamics_loss(
    model: SpatialExplorationWorldModel,
    batch: Mapping[str, torch.Tensor],
    roots: torch.Tensor,
    args: argparse.Namespace,
    return_sample_priorities: bool = False,
):
    observations = batch["observations"]
    actions = batch["actions"]
    state_ids = model.representation_state_ids(
        observations, batch["state_ids"]
    )
    encoded = model.encode_trajectory(observations, actions)
    gathered = gather_spatial_rollout_batch(
        encoded, observations, actions, state_ids, roots, args.horizon
    )
    rollout = model.rollout(
        gathered["root_latents"],
        gathered["root_beliefs"],
        gathered["suffix_actions"],
    )
    target_keys = gathered["target_keys"].detach()
    zero = rollout.keys.new_zeros(())
    latent = zero
    if float(args.latent_dynamics_weight) != 0.0:
        latent = F.smooth_l1_loss(
            rollout.latents, gathered["target_latents"].detach()
        )
    key = zero
    if (
        model.memory_key_mode != "belief_revisit"
        and float(args.key_dynamics_weight) != 0.0
    ):
        key = torch.mean(
            1.0 - F.cosine_similarity(rollout.keys, target_keys, dim=-1)
        )
    observation_key_consistency = zero
    observation_key_consistency_by_depth = [0.0 for _ in range(int(args.horizon))]
    observation_key_consistency_weight = float(
        getattr(args, "observation_key_consistency_weight", 0.0)
    )
    if observation_key_consistency_weight != 0.0:
        target_observation_keys = gathered["target_observation_keys"].detach()
        observation_key_distances = 1.0 - F.cosine_similarity(
            rollout.observation_keys,
            target_observation_keys,
            dim=-1,
        )
        observation_key_consistency = observation_key_distances.mean()
        observation_key_consistency_by_depth = [
            float(observation_key_distances[:, depth].mean().detach())
            for depth in range(int(args.horizon))
        ]
    reconstruction = zero
    if float(args.rollout_reconstruction_weight) != 0.0:
        target_observations = model.reconstruction_targets(
            gathered["target_observations"]
        )
        reconstruction = F.l1_loss(
            model.decode_rollout(rollout), target_observations
        )
    batch_size = observations.shape[0]
    root_count = len(roots)
    contrastive = zero
    if (
        model.memory_key_mode != "belief_revisit"
        and float(args.rollout_contrastive_weight) != 0.0
    ):
        contrastive = multipositive_contrastive_loss(
            rollout.keys.reshape(
                batch_size, root_count, int(args.horizon), model.key_dim
            ),
            target_keys.reshape(
                batch_size, root_count, int(args.horizon), model.key_dim
            ),
            gathered["target_ids"],
            args.temperature,
        )
    memory_novelty = zero
    if (
        model.memory_key_mode != "belief_revisit"
        and float(args.memory_novelty_weight) != 0.0
    ):
        predicted_keys = rollout.keys.reshape(
            batch_size, root_count, int(args.horizon), model.key_dim
        )
        true_keys = target_keys.reshape_as(predicted_keys)
        predicted_scores, teacher_scores = _novelty_scores(
            predicted_keys,
            true_keys,
            encoded.keys.detach(),
            roots,
        )
        memory_novelty = F.l1_loss(
            predicted_scores,
            teacher_scores.detach(),
        )
    belief_dynamics_consistency = zero
    belief_dynamics_consistency_by_depth = [
        0.0 for _ in range(int(args.horizon))
    ]
    belief_dynamics_consistency_weight = float(
        getattr(args, "belief_dynamics_consistency_weight", 0.0)
    )
    predicted_beliefs = None
    target_beliefs = None
    belief_prediction_errors = None
    if model.memory_key_mode in (
        "belief",
        "belief_to_observation",
        "belief_revisit",
    ):
        predicted_beliefs = rollout.beliefs.reshape(
            batch_size,
            root_count,
            int(args.horizon),
            model.belief_dim,
        )
        target_beliefs = gathered["target_beliefs"].reshape_as(
            predicted_beliefs
        ).detach()
        belief_prediction_errors = 1.0 - F.cosine_similarity(
            predicted_beliefs,
            target_beliefs,
            dim=-1,
        )
        if belief_dynamics_consistency_weight != 0.0:
            belief_dynamics_consistency = belief_prediction_errors.mean()
            belief_dynamics_consistency_by_depth = [
                float(belief_prediction_errors[:, :, depth].mean().detach())
                for depth in range(int(args.horizon))
            ]
    elif belief_dynamics_consistency_weight != 0.0:
        raise ValueError(
            "Belief dynamics consistency requires a belief memory-key mode"
        )

    pairwise_revisit = zero
    sample_priorities = torch.zeros(
        observations.shape[0], root_count, device=observations.device
    )
    pairwise_revisit_components = {
        "pairwise_revisit_loss": 0.0,
        "pairwise_revisit_positive_loss": 0.0,
        "pairwise_revisit_negative_loss": 0.0,
        "pairwise_revisit_positive_count": 0.0,
        "pairwise_revisit_negative_count": 0.0,
        "pairwise_revisit_sampled_negative_count": 0.0,
        "pairwise_revisit_positive_probability": 0.0,
        "pairwise_revisit_negative_probability": 0.0,
    }
    pairwise_revisit_weight = float(
        getattr(args, "pairwise_revisit_weight", 1.0)
    )
    prioritized_replay_priority_mode = str(
        getattr(args, "prioritized_replay_priority_mode", "belief_error")
    ).lower()
    if prioritized_replay_priority_mode not in {
        "belief_error",
        "revisit_error",
    }:
        raise ValueError(
            "prioritized_replay_priority_mode must be 'belief_error' or "
            "'revisit_error'"
        )
    if (
        model.memory_key_mode == "belief_revisit"
        and pairwise_revisit_weight != 0.0
    ):
        assert predicted_beliefs is not None
        assert belief_prediction_errors is not None
        (
            pairwise_revisit,
            pairwise_revisit_components,
            sample_novelty_errors,
        ) = (
            pairwise_revisit_prediction_loss(
                model,
                predicted_beliefs,
                encoded.beliefs,
                state_ids,
                gathered["target_ids"],
                roots,
                int(getattr(args, "pairwise_revisit_negative_samples", 8)),
            )
        )
        belief_error_priorities = belief_prediction_errors.mean(dim=-1).detach()
        sample_priorities = (
            sample_novelty_errors
            if prioritized_replay_priority_mode == "revisit_error"
            else belief_error_priorities
        )
        pairwise_revisit_components.update(
            {
                "pairwise_revisit_novelty_bce": float(
                    sample_novelty_errors.mean()
                ),
                "belief_prediction_priority_mean": float(
                    belief_error_priorities.mean()
                ),
                "revisit_prediction_priority_mean": float(
                    sample_novelty_errors.mean()
                ),
                "replay_priority_mean": float(sample_priorities.mean()),
                "replay_priority_mode_revisit_error": float(
                    prioritized_replay_priority_mode == "revisit_error"
                ),
            }
        )
    predictive_equivalence = zero
    predictive_equivalence_components = {
        "predictive_equivalence_loss": 0.0,
        "predictive_equivalence_positive_loss": 0.0,
        "predictive_equivalence_negative_loss": 0.0,
        "predictive_equivalence_aliased_negative_loss": 0.0,
        "predictive_equivalence_pair_count": 0.0,
        "predictive_equivalence_positive_count": 0.0,
        "predictive_equivalence_negative_count": 0.0,
        "predictive_equivalence_aliased_negative_count": 0.0,
        "predictive_equivalence_active": 0.0,
        "predictive_equivalence_mean_prefix": 0.0,
    }
    predictive_equivalence_weight = float(
        getattr(args, "predictive_equivalence_weight", 0.0)
    )
    if predictive_equivalence_weight != 0.0:
        if model.memory_key_mode != "belief":
            raise ValueError(
                "Predictive equivalence requires belief-state memory keys"
            )
        equivalence_roots = torch.arange(
            int(actions.shape[1]) - int(args.horizon) + 1,
            device=actions.device,
        )
        flattened_observations = observations.reshape(
            -1, int(np.prod(observations.shape[-3:]))
        )
        _, global_observation_ids = torch.unique(
            flattened_observations,
            dim=0,
            return_inverse=True,
        )
        equivalence_state_ids = global_observation_ids.reshape(
            observations.shape[:2]
        )
        equivalence_batch = gather_spatial_rollout_batch(
            encoded,
            observations,
            actions,
            equivalence_state_ids,
            equivalence_roots,
            int(args.horizon),
        )
        predictive_equivalence, predictive_equivalence_components = (
            action_conditioned_predictive_equivalence_loss(
                equivalence_batch["root_keys"],
                equivalence_batch["root_ids"],
                equivalence_batch["suffix_actions"].reshape(
                    batch_size, len(equivalence_roots), int(args.horizon)
                ),
                equivalence_batch["target_ids"],
                negative_margin=float(
                    getattr(
                        args,
                        "predictive_equivalence_negative_margin",
                        0.2,
                    )
                ),
            )
        )
    total = (
        float(args.latent_dynamics_weight) * latent
        + float(args.key_dynamics_weight) * key
        + observation_key_consistency_weight * observation_key_consistency
        + belief_dynamics_consistency_weight * belief_dynamics_consistency
        + float(args.rollout_reconstruction_weight) * reconstruction
        + float(args.rollout_contrastive_weight) * contrastive
        + float(args.memory_novelty_weight) * memory_novelty
        + pairwise_revisit_weight * pairwise_revisit
        + predictive_equivalence_weight * predictive_equivalence
    )
    components = {
        "loss": float(total.detach()),
        "latent": float(latent.detach()),
        "key": float(key.detach()),
        "observation_key_consistency": float(
            observation_key_consistency.detach()
        ),
        "belief_dynamics_consistency": float(
            belief_dynamics_consistency.detach()
        ),
        "rollout_reconstruction_mae": float(reconstruction.detach()),
        "contrastive": float(contrastive.detach()),
        "memory_novelty_mae": float(memory_novelty.detach()),
        **pairwise_revisit_components,
        **predictive_equivalence_components,
        **{
            f"observation_key_consistency_depth_{depth + 1}": value
            for depth, value in enumerate(observation_key_consistency_by_depth)
        },
        **{
            f"belief_dynamics_consistency_depth_{depth + 1}": value
            for depth, value in enumerate(
                belief_dynamics_consistency_by_depth
            )
        },
    }
    if return_sample_priorities:
        return total, components, sample_priorities
    return total, components


def stage_two_loss(
    model: SpatialExplorationWorldModel,
    batch: Mapping[str, torch.Tensor],
    roots: torch.Tensor,
    args: argparse.Namespace,
    return_sample_priorities: bool = False,
):
    """Compute frozen-representation or joint stage-two training loss."""
    dynamics_result = dynamics_loss(
        model,
        batch,
        roots,
        args,
        return_sample_priorities=return_sample_priorities,
    )
    if return_sample_priorities:
        transition_loss, components, sample_priorities = dynamics_result
    else:
        transition_loss, components = dynamics_result
    components = {
        **components,
        "dynamics_loss": float(transition_loss.detach()),
        "representation_loss": 0.0,
    }
    if not bool(args.joint_representation_dynamics):
        if return_sample_priorities:
            return transition_loss, components, sample_priorities
        return transition_loss, components

    representation_loss, representation_components = autoencoder_loss(
        model, batch, args
    )
    total = transition_loss + representation_loss
    components["loss"] = float(total.detach())
    components["representation_loss"] = float(representation_loss.detach())
    components.update(
        {
            f"representation_{name}": value
            for name, value in representation_components.items()
            if name != "loss"
        }
    )
    if return_sample_priorities:
        return total, components, sample_priorities
    return total, components


def _novelty_scores(
    predicted_keys: torch.Tensor,
    target_keys: torch.Tensor,
    all_real_keys: torch.Tensor,
    roots: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, root_count, horizon, _ = predicted_keys.shape
    predicted_scores = torch.empty(
        batch_size, root_count, horizon, device=predicted_keys.device
    )
    teacher_scores = torch.empty_like(predicted_scores)
    for root_index in range(root_count):
        root_timestep = int(roots[root_index])
        real_history = all_real_keys[:, :root_timestep + 1]
        for depth in range(horizon):
            predicted_memory = (
                real_history
                if depth == 0
                else torch.cat(
                    [real_history, predicted_keys[:, root_index, :depth]], dim=1
                )
            )
            teacher_memory = (
                real_history
                if depth == 0
                else torch.cat(
                    [real_history, target_keys[:, root_index, :depth]], dim=1
                )
            )
            predicted_similarity = torch.einsum(
                "bk,bmk->bm",
                predicted_keys[:, root_index, depth],
                predicted_memory,
            ).amax(dim=1)
            teacher_similarity = torch.einsum(
                "bk,bmk->bm",
                target_keys[:, root_index, depth],
                teacher_memory,
            ).amax(dim=1)
            predicted_scores[:, root_index, depth] = 1.0 - predicted_similarity
            teacher_scores[:, root_index, depth] = 1.0 - teacher_similarity
    return predicted_scores, teacher_scores


def _pairwise_revisit_scores(
    model: SpatialExplorationWorldModel,
    predicted_beliefs: torch.Tensor,
    real_beliefs: torch.Tensor,
    state_ids: torch.Tensor,
    target_ids: torch.Tensor,
    roots: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score imagined revisits and exact-RGB teacher novelty by branch."""
    batch_size, root_count, horizon, _ = predicted_beliefs.shape
    predicted_scores = predicted_beliefs.new_empty(
        batch_size, root_count, horizon
    )
    teacher_scores = predicted_scores.new_empty(predicted_scores.shape)
    for root_index in range(root_count):
        root_timestep = int(roots[root_index])
        real_memory = real_beliefs[:, : root_timestep + 1]
        real_memory_ids = state_ids[:, : root_timestep + 1]
        for depth in range(horizon):
            memory = real_memory
            memory_ids = real_memory_ids
            if depth > 0:
                memory = torch.cat(
                    [memory, predicted_beliefs[:, root_index, :depth]], dim=1
                )
                memory_ids = torch.cat(
                    [memory_ids, target_ids[:, root_index, :depth]], dim=1
                )
            query = predicted_beliefs[:, root_index, depth]
            predicted_scores[:, root_index, depth] = model.revisit_novelty(
                query, memory
            )
            is_revisit = target_ids[:, root_index, depth, None].eq(
                memory_ids
            ).any(dim=1)
            teacher_scores[:, root_index, depth] = (~is_revisit).to(
                predicted_scores.dtype
            )
    return predicted_scores, teacher_scores


def evaluate_model(
    model: SpatialExplorationWorldModel,
    loader: DataLoader,
    device: torch.device,
    horizon: int,
    maximum_roots_per_episode: int = 0,
) -> Dict[str, object]:
    model.eval()
    has_reconstruction = model.supports_reconstruction
    fields = (
        "direct_abs",
        "direct_sq",
        "rollout_abs",
        "rollout_sq",
        "copy_abs",
        "copy_sq",
        "history_abs",
        "history_sq",
        "cosine",
        "observation_key_cosine",
        "top1",
        "top5",
        "changed_top1",
        "predicted_novelty",
        "teacher_novelty",
        "rgb_labels",
        "position_labels",
    )
    values: Dict[int, Dict[str, List[np.ndarray]]] = {
        depth: {field: [] for field in fields}
        for depth in range(1, int(horizon) + 1)
    }
    with torch.inference_mode():
        for batch in loader:
            observations = batch["observations"].to(device)
            actions = batch["actions"].to(device)
            state_ids = batch["state_ids"].to(device)
            poses = batch["poses"].to(device)
            novelty_targets = batch["novelty_targets"].to(device)
            encoded = model.encode_trajectory(observations, actions)
            available_root_count = actions.shape[1] - int(horizon) + 1
            if (
                int(maximum_roots_per_episode) > 0
                and available_root_count > int(maximum_roots_per_episode)
            ):
                roots = torch.linspace(
                    0,
                    available_root_count - 1,
                    int(maximum_roots_per_episode),
                    device=device,
                ).long().unique()
            else:
                roots = torch.arange(available_root_count, device=device)
            root_count = int(roots.numel())
            gathered = gather_spatial_rollout_batch(
                encoded, observations, actions, state_ids, roots, int(horizon)
            )
            rollout = model.rollout(
                gathered["root_latents"],
                gathered["root_beliefs"],
                gathered["suffix_actions"],
            )
            batch_size = observations.shape[0]
            predicted_latents = rollout.latents.reshape(
                batch_size,
                root_count,
                int(horizon),
                *encoded.latents.shape[-3:],
            )
            predicted_keys = rollout.keys.reshape(
                batch_size, root_count, int(horizon), model.key_dim
            )
            predicted_observation_keys = rollout.observation_keys.reshape(
                batch_size, root_count, int(horizon), model.key_dim
            )
            predicted_beliefs = rollout.beliefs.reshape(
                batch_size,
                root_count,
                int(horizon),
                model.belief_dim,
            )
            target_latents = gathered["target_latents"].reshape_as(
                predicted_latents
            )
            target_beliefs = gathered["target_beliefs"].reshape(
                batch_size,
                root_count,
                int(horizon),
                model.belief_dim,
            )
            target_keys = gathered["target_keys"].reshape_as(predicted_keys)
            target_observation_keys = gathered["target_observation_keys"].reshape_as(
                predicted_observation_keys
            )
            target_observations = gathered["target_observations"].reshape(
                batch_size,
                root_count,
                int(horizon),
                *observations.shape[-3:],
            )
            if has_reconstruction:
                target_images = model.reconstruction_targets(target_observations)
                rollout_images = model.decode_rollout(rollout).reshape_as(
                    target_images
                )
                direct_images = (
                    model.decode_beliefs(target_beliefs)
                    if model.memory_key_mode in (
                        "belief",
                        "belief_to_observation",
                        "belief_revisit",
                    )
                    else model.decode_latents(target_latents)
                )
                observations_float = model.reconstruction_targets(observations)
                copy_images = observations_float[:, roots, None].expand_as(
                    target_images
                )
                cumulative = observations_float.cumsum(dim=1)
                denominators = (roots + 1).to(observations.dtype).reshape(
                    1, -1, 1, 1, 1
                )
                history_images = (
                    cumulative[:, roots] / denominators
                )[:, :, None].expand_as(target_images)

            if model.memory_key_mode == "belief_revisit":
                similarity = predicted_beliefs.new_empty(
                    batch_size,
                    root_count,
                    int(horizon),
                    encoded.beliefs.shape[1],
                )
                for root_index in range(root_count):
                    for depth_index in range(int(horizon)):
                        similarity[:, root_index, depth_index] = torch.sigmoid(
                            model.revisit_logits(
                                predicted_beliefs[:, root_index, depth_index],
                                encoded.beliefs,
                            )
                        )
            else:
                similarity = torch.einsum(
                    "brhk,btk->brht", predicted_keys, encoded.keys
                )
            nearest = similarity.argmax(dim=-1)
            top_indices = similarity.topk(min(5, encoded.keys.shape[1]), dim=-1).indices
            if model.memory_key_mode == "belief_revisit":
                predicted_novelty, teacher_novelty = _pairwise_revisit_scores(
                    model,
                    predicted_beliefs,
                    encoded.beliefs,
                    state_ids,
                    gathered["target_ids"],
                    roots,
                )
            else:
                predicted_novelty, teacher_novelty = _novelty_scores(
                    predicted_keys, target_keys, encoded.keys, roots
                )
            position_rewards = torch.as_tensor(
                position_first_visit_rewards(poses.cpu().numpy()),
                dtype=torch.float32,
                device=device,
            )
            batch_indices = torch.arange(batch_size, device=device)[:, None]

            for depth_index in range(int(horizon)):
                depth = depth_index + 1
                endpoint_indices = roots + depth
                target_ids = state_ids[:, endpoint_indices]
                nearest_ids = torch.gather(
                    state_ids, 1, nearest[:, :, depth_index]
                )
                candidates = state_ids[:, None, :].expand(
                    batch_size, root_count, -1
                )
                top_ids = torch.gather(
                    candidates, 2, top_indices[:, :, depth_index]
                )
                top1 = nearest_ids == target_ids
                top5 = (top_ids == target_ids.unsqueeze(-1)).any(dim=-1)
                changed = state_ids[:, roots] != target_ids
                rgb_labels = (
                    novelty_targets[:, roots, depth_index] == NOVEL
                )
                position_labels = position_rewards[
                    batch_indices, endpoint_indices[None] - 1
                ] > 0.5
                if has_reconstruction:
                    for prefix, prediction in (
                        ("direct", direct_images[:, :, depth_index]),
                        ("rollout", rollout_images[:, :, depth_index]),
                        ("copy", copy_images[:, :, depth_index]),
                        ("history", history_images[:, :, depth_index]),
                    ):
                        difference = prediction - target_images[:, :, depth_index]
                        values[depth][f"{prefix}_abs"].append(
                            difference.abs().flatten(2).mean(dim=-1).cpu().numpy()
                        )
                        values[depth][f"{prefix}_sq"].append(
                            difference.square().flatten(2).mean(dim=-1).cpu().numpy()
                        )
                values[depth]["cosine"].append(
                    F.cosine_similarity(
                        predicted_keys[:, :, depth_index],
                        target_keys[:, :, depth_index],
                        dim=-1,
                    ).cpu().numpy()
                )
                values[depth]["observation_key_cosine"].append(
                    F.cosine_similarity(
                        predicted_observation_keys[:, :, depth_index],
                        target_observation_keys[:, :, depth_index],
                        dim=-1,
                    ).cpu().numpy()
                )
                values[depth]["top1"].append(top1.cpu().numpy())
                values[depth]["top5"].append(top5.cpu().numpy())
                values[depth]["changed_top1"].append(
                    top1[changed].cpu().numpy()
                )
                values[depth]["predicted_novelty"].append(
                    predicted_novelty[:, :, depth_index].cpu().numpy()
                )
                values[depth]["teacher_novelty"].append(
                    teacher_novelty[:, :, depth_index].cpu().numpy()
                )
                values[depth]["rgb_labels"].append(rgb_labels.cpu().numpy())
                values[depth]["position_labels"].append(
                    position_labels.cpu().numpy()
                )

    by_depth: MutableMapping[str, object] = {}
    for depth in range(1, int(horizon) + 1):
        arrays = {
            key: (
                np.concatenate(parts).reshape(-1)
                if parts
                else np.asarray([], dtype=np.float32)
            )
            for key, parts in values[depth].items()
        }

        def image_summary(prefix: str) -> Dict[str, float | None]:
            if arrays[f"{prefix}_abs"].size == 0:
                return {"mae": None, "mse": None, "psnr": None}
            mae = float(arrays[f"{prefix}_abs"].mean())
            mse = float(arrays[f"{prefix}_sq"].mean())
            return {
                "mae": mae,
                "mse": mse,
                "psnr": float(-10.0 * math.log10(max(mse, 1e-12))),
            }

        rollout_summary = image_summary("rollout")
        copy_summary = image_summary("copy")
        history_summary = image_summary("history")
        by_depth[str(depth)] = {
            "count": int(arrays["cosine"].size),
            "direct_reconstruction": image_summary("direct"),
            "rollout_reconstruction": rollout_summary,
            "copy_root_baseline": copy_summary,
            "history_mean_baseline": history_summary,
            "rollout_mae_gain_vs_copy": (
                None
                if rollout_summary["mae"] is None
                else copy_summary["mae"] - rollout_summary["mae"]
            ),
            "rollout_mae_gain_vs_history_mean": (
                None
                if rollout_summary["mae"] is None
                else history_summary["mae"] - rollout_summary["mae"]
            ),
            "alignment_cosine_mean": float(arrays["cosine"].mean()),
            "observation_key_alignment_cosine_mean": float(
                arrays["observation_key_cosine"].mean()
            ),
            "rgb_retrieval_top1": float(arrays["top1"].mean()),
            "rgb_retrieval_top5": float(arrays["top5"].mean()),
            "changed_rgb_retrieval_top1": float(
                arrays["changed_top1"].mean()
                if arrays["changed_top1"].size else 0.0
            ),
            "rgb_novelty_auc": _binary_auc(
                arrays["rgb_labels"], arrays["predicted_novelty"]
            ),
            "position_novelty_auc": _binary_auc(
                arrays["position_labels"], arrays["predicted_novelty"]
            ),
            "teacher_rgb_novelty_auc": _binary_auc(
                arrays["rgb_labels"], arrays["teacher_novelty"]
            ),
            "teacher_position_novelty_auc": _binary_auc(
                arrays["position_labels"], arrays["teacher_novelty"]
            ),
            "novelty_prediction_mae": float(
                np.mean(
                    np.abs(
                        arrays["predicted_novelty"] - arrays["teacher_novelty"]
                    )
                )
            ),
            "novelty_prediction_correlation": float(
                np.corrcoef(
                    arrays["predicted_novelty"], arrays["teacher_novelty"]
                )[0, 1]
            )
            if (
                arrays["predicted_novelty"].size > 1
                and arrays["predicted_novelty"].std() > 0
                and arrays["teacher_novelty"].std() > 0
            )
            else None,
        }
    depth_values = list(by_depth.values())
    position_auc_values = [
        float(row["position_novelty_auc"])
        for row in depth_values
        if row["position_novelty_auc"] is not None
    ]
    return {
        "by_depth": by_depth,
        "mean_rollout_mae": (
            float(
                np.mean(
                    [
                        row["rollout_reconstruction"]["mae"]
                        for row in depth_values
                    ]
                )
            )
            if has_reconstruction
            else None
        ),
        "mean_copy_root_mae": (
            float(
                np.mean(
                    [row["copy_root_baseline"]["mae"] for row in depth_values]
                )
            )
            if has_reconstruction
            else None
        ),
        "mean_position_novelty_auc": float(
            np.mean(position_auc_values) if position_auc_values else np.nan
        ),
        "rollout_beats_copy_at_every_depth": bool(
            has_reconstruction
            and all(row["rollout_mae_gain_vs_copy"] > 0 for row in depth_values)
        ),
    }


def direct_reconstruction_metrics(
    model: SpatialExplorationWorldModel,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float | None]:
    if not model.supports_reconstruction:
        return {"mae": None, "mse": None, "psnr": None}
    absolute = []
    squared = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            observations = batch["observations"].to(device)
            targets = model.reconstruction_targets(observations)
            decoded = model.decode_latents(model.encode_observations(observations))
            difference = decoded - targets
            absolute.append(float(difference.abs().mean()))
            squared.append(float(difference.square().mean()))
    mse = float(np.mean(squared))
    return {
        "mae": float(np.mean(absolute)),
        "mse": mse,
        "psnr": float(-10.0 * math.log10(max(mse, 1e-12))),
    }


def inverse_action_metrics(
    model: SpatialExplorationWorldModel,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    losses = []
    correct = 0
    count = 0
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            observations = batch["observations"].to(device)
            actions = batch["actions"].to(device)
            latents = model.encode_observations(observations)
            keys = model.keys_from_latents(latents)
            logits = model.inverse_action(
                torch.cat([keys[:, :-1], keys[:, 1:]], dim=-1)
            )
            flat_logits = logits.reshape(-1, model.action_space_size)
            flat_actions = actions.reshape(-1)
            losses.append(
                float(F.cross_entropy(flat_logits, flat_actions, reduction="mean"))
            )
            correct += int((flat_logits.argmax(dim=-1) == flat_actions).sum())
            count += int(flat_actions.numel())
    return {
        "inverse_loss": float(np.mean(losses)),
        "inverse_accuracy": float(correct / max(count, 1)),
    }


def write_visual_grids(
    model: SpatialExplorationWorldModel,
    dataset: Dataset,
    output_dir: Path,
    device: torch.device,
    horizon: int,
    figure_count: int,
) -> List[str]:
    if not model.supports_reconstruction or int(figure_count) <= 0:
        return []
    import matplotlib.pyplot as plt

    paths = []
    model.eval()
    for figure_index in range(min(int(figure_count), len(dataset))):
        item = dataset[figure_index]
        observations = item["observations"].unsqueeze(0).to(device)
        actions = item["actions"].unsqueeze(0).to(device)
        poses = item["poses"].numpy()
        position_rewards = position_first_visit_rewards(poses)
        root_count = actions.shape[1] - int(horizon) + 1
        window_scores = np.asarray(
            [position_rewards[root:root + int(horizon)].sum() for root in range(root_count)]
        )
        root = int(window_scores.argmax())
        encoded = model.encode_trajectory(observations, actions)
        roots = torch.tensor([root], device=device)
        gathered = gather_spatial_rollout_batch(
            encoded,
            observations,
            actions,
            item["state_ids"].unsqueeze(0).to(device),
            roots,
            int(horizon),
        )
        with torch.inference_mode():
            rollout = model.rollout(
                gathered["root_latents"],
                gathered["root_beliefs"],
                gathered["suffix_actions"],
            )
            actual = model.reconstruction_targets(
                observations[0, root:root + int(horizon) + 1]
            )
            direct = (
                model.decode_beliefs(
                    encoded.beliefs[0, root:root + int(horizon) + 1]
                )
                if model.memory_key_mode in (
                    "belief",
                    "belief_to_observation",
                    "belief_revisit",
                )
                else model.decode_latents(
                    encoded.latents[0, root:root + int(horizon) + 1]
                )
            )
            predicted = torch.cat(
                [
                    direct[:1],
                    model.decode_rollout(rollout)[0],
                ],
                dim=0,
            )
            error = (predicted - actual).abs()
        rows = [actual.cpu(), direct.cpu(), predicted.cpu(), error.cpu()]
        fig, axes = plt.subplots(
            4, int(horizon) + 1, figsize=(3 * (int(horizon) + 1), 11)
        )
        row_names = ("true", "direct decode", "predicted rollout", "absolute error")
        suffix = actions[0, root:root + int(horizon)].cpu().numpy()
        for row_index, images in enumerate(rows):
            for column in range(int(horizon) + 1):
                axes[row_index, column].imshow(
                    np.transpose(images[column].numpy(), (1, 2, 0)).clip(0, 1)
                )
                axes[row_index, column].set_xticks([])
                axes[row_index, column].set_yticks([])
                if column == 0:
                    axes[row_index, column].set_ylabel(row_names[row_index])
                if row_index == 0:
                    title = "root" if column == 0 else (
                        f"d{column}: a={int(suffix[column - 1])}\n"
                        f"new-position={int(position_rewards[root + column - 1])}"
                    )
                    axes[row_index, column].set_title(title)
        fig.suptitle(f"Held-out trajectory {figure_index}, root {root}")
        fig.tight_layout()
        path = output_dir / f"spatial_rollout_example_{figure_index:02d}.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(str(path))
    return paths


def _mean_rgb_novelty_auc(metrics: Mapping[str, object]) -> float:
    values = [
        float(row["rgb_novelty_auc"])
        for row in metrics["by_depth"].values()
        if row["rgb_novelty_auc"] is not None
        and np.isfinite(float(row["rgb_novelty_auc"]))
    ]
    if not values:
        raise ValueError("RGB novelty AUC is undefined at every rollout depth")
    return float(np.mean(values))


def _selection_score(
    metrics: Mapping[str, object],
    horizon: int,
    selection_metric: str = "rollout_reconstruction",
) -> float:
    if selection_metric == "key_novelty":
        final_retrieval = float(
            metrics["by_depth"][str(horizon)]["changed_rgb_retrieval_top1"]
        )
        return _mean_rgb_novelty_auc(metrics) + final_retrieval
    if selection_metric == "novelty_fidelity":
        correlations = []
        errors = []
        for row in metrics["by_depth"].values():
            correlation = row["novelty_prediction_correlation"]
            error = row["novelty_prediction_mae"]
            if correlation is not None and np.isfinite(float(correlation)):
                correlations.append(float(correlation))
            if error is not None and np.isfinite(float(error)):
                errors.append(float(error))
        if not correlations or not errors:
            raise ValueError(
                "Novelty fidelity is undefined without finite correlation and MAE"
            )
        return float(np.mean(correlations) - np.mean(errors))
    if selection_metric != "rollout_reconstruction":
        raise ValueError(f"Unknown selection metric: {selection_metric}")
    if metrics["mean_rollout_mae"] is None:
        raise ValueError(
            "rollout_reconstruction selection requires a reconstructive representation"
        )
    depth_rows = list(metrics["by_depth"].values())
    relative_gains = []
    for row in depth_rows:
        copy_mae = max(float(row["copy_root_baseline"]["mae"]), 1e-8)
        relative_gains.append(float(row["rollout_mae_gain_vs_copy"]) / copy_mae)
    final_retrieval = float(
        metrics["by_depth"][str(horizon)]["changed_rgb_retrieval_top1"]
    )
    return float(np.mean(relative_gains) + final_retrieval)


def main() -> None:
    args = build_parser().parse_args()
    _smoke_overrides(args)
    representation_weights = (
        args.direct_reconstruction_weight,
        args.direct_contrastive_weight,
        args.inverse_action_weight,
        args.variance_weight,
    )
    if any(float(weight) < 0.0 for weight in representation_weights):
        raise ValueError("Representation loss weights must be non-negative")
    if args.resume_checkpoint is None and not any(
        float(weight) > 0.0 for weight in representation_weights
    ):
        raise ValueError("At least one representation loss weight must be positive")
    dynamics_weights = (
        args.latent_dynamics_weight,
        args.key_dynamics_weight,
        args.observation_key_consistency_weight,
        args.belief_dynamics_consistency_weight,
        args.rollout_reconstruction_weight,
        args.rollout_contrastive_weight,
        args.memory_novelty_weight,
    )
    if any(float(weight) < 0.0 for weight in dynamics_weights):
        raise ValueError("Dynamics loss weights must be non-negative")
    if not any(float(weight) > 0.0 for weight in dynamics_weights):
        raise ValueError("At least one dynamics loss weight must be positive")
    if (
        float(args.belief_dynamics_consistency_weight) > 0.0
        and args.memory_key_mode
        not in ("belief", "belief_to_observation", "belief_revisit")
    ):
        raise ValueError(
            "Belief dynamics consistency requires a belief memory-key mode"
        )
    if args.representation_mode == "inverse_dynamics":
        if float(args.direct_reconstruction_weight) != 0.0:
            raise ValueError(
                "inverse_dynamics has no decoder; set "
                "--direct-reconstruction-weight 0"
            )
        if float(args.rollout_reconstruction_weight) != 0.0:
            raise ValueError(
                "inverse_dynamics has no decoder; set "
                "--rollout-reconstruction-weight 0"
            )
        if args.selection_metric not in ("key_novelty", "novelty_fidelity"):
            raise ValueError(
                "inverse_dynamics must use a key-based selection metric"
            )
    if not args.dataset.is_file():
        raise FileNotFoundError(
            f"Missing trajectory dataset: {args.dataset}. Run "
            "experiments/run_branching_rooms_action_novelty.py --mode collect first."
        )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    bundle: DatasetBundle = load_dataset(args.dataset)
    if int(bundle.metadata["horizon"]) != int(args.horizon):
        raise ValueError("Dataset horizon does not match --horizon")
    maximum_episodes = 4 if args.smoke else None
    datasets = {
        name: ExplorationEpisodeDataset(bundle, split_id, maximum_episodes)
        for split_id, name in enumerate(("train", "validation", "test", "heldout"))
    }
    loaders = {
        "train": _loader(
            datasets["train"], args.batch_size, True, args.num_workers
        ),
        **{
            name: _loader(
                datasets[name],
                args.evaluation_batch_size,
                False,
                args.num_workers,
            )
            for name in ("validation", "test", "heldout")
        },
    }
    observation_shape = tuple(int(value) for value in bundle.observations.shape[-3:])
    model = SpatialExplorationWorldModel(
        observation_shape=observation_shape,
        spatial_channels=args.spatial_channels,
        key_dim=args.key_dim,
        belief_dim=args.belief_dim,
        action_dim=args.action_dim,
        representation_mode=args.representation_mode,
        coarse_reconstruction_size=args.coarse_reconstruction_size,
        memory_key_mode=args.memory_key_mode,
    ).to(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arguments = vars(args).copy()
    arguments["dataset"] = str(args.dataset)
    arguments["output_dir"] = str(args.output_dir)
    arguments["resume_checkpoint"] = (
        None if args.resume_checkpoint is None else str(args.resume_checkpoint)
    )
    arguments["device_resolved"] = str(device)
    arguments["physical_position_used_for_training"] = False
    arguments["counterfactual_transition_queries"] = 0
    (args.output_dir / "arguments.json").write_text(
        json.dumps(arguments, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"device={device} train/validation/test/heldout episodes="
        f"{len(datasets['train'])}/{len(datasets['validation'])}/"
        f"{len(datasets['test'])}/{len(datasets['heldout'])}",
        flush=True,
    )

    resume_checkpoint = None
    resume_metrics = None
    if args.resume_checkpoint is not None:
        if not args.resume_checkpoint.is_file():
            raise FileNotFoundError(args.resume_checkpoint)
        resume_checkpoint = torch.load(
            args.resume_checkpoint, map_location=device, weights_only=False
        )
        model.load_state_dict(resume_checkpoint["model"], strict=True)
        print(
            f"resuming frozen-representation dynamics from "
            f"{args.resume_checkpoint} at epoch "
            f"{int(resume_checkpoint['epoch']) + 1}",
            flush=True,
        )
    else:
        autoencoder_optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        autoencoder_rows = []
        print(
            f"stage 1: training {args.representation_mode} representation",
            flush=True,
        )
        for epoch in range(int(args.autoencoder_epochs)):
            model.train()
            components = []
            for batch in loaders["train"]:
                batch = {key: value.to(device) for key, value in batch.items()}
                loss, row = autoencoder_loss(model, batch, args)
                autoencoder_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                autoencoder_optimizer.step()
                components.append(row)
            mean = _mean_components(components)
            validation_reconstruction = direct_reconstruction_metrics(
                model, loaders["validation"], device
            )
            validation_inverse = inverse_action_metrics(
                model, loaders["validation"], device
            )
            autoencoder_rows.append(
                {
                    "epoch": epoch,
                    **{f"train_{key}": value for key, value in mean.items()},
                    **{
                        f"validation_{key}": value
                        for key, value in validation_reconstruction.items()
                    },
                    **{
                        f"validation_{key}": value
                        for key, value in validation_inverse.items()
                    },
                }
            )
            if model.supports_reconstruction:
                representation_summary = (
                    f"validation_mae={validation_reconstruction['mae']:.4f} "
                    f"psnr={validation_reconstruction['psnr']:.2f}"
                )
            else:
                representation_summary = "decoder=none"
            print(
                f"representation epoch {epoch:02d} loss={mean['loss']:.4f} "
                f"{representation_summary} "
                f"inverse_loss={validation_inverse['inverse_loss']:.4f} "
                f"inverse_accuracy={validation_inverse['inverse_accuracy']:.3f}",
                flush=True,
            )
        _write_csv(args.output_dir / "autoencoder_training.csv", autoencoder_rows)

    if resume_checkpoint is not None and args.evaluate_resume_before_training:
        print("evaluating resume checkpoint before dynamics adaptation", flush=True)
        resume_metrics = {
            name: evaluate_model(
                model,
                loaders[name],
                device,
                args.horizon,
                args.evaluation_roots_per_episode,
            )
            for name in ("validation", "test", "heldout")
        }
        resume_metrics_path = args.output_dir / "resume_metrics.json"
        resume_metrics_path.write_text(
            json.dumps(resume_metrics, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"resume metrics: {resume_metrics_path}", flush=True)

    if not args.joint_representation_dynamics:
        freeze_observation_model(model)
    dynamics_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    dynamics_optimizer = torch.optim.AdamW(
        dynamics_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    if (
        resume_checkpoint is not None
        and "optimizer" in resume_checkpoint
        and not args.fresh_optimizer_on_resume
    ):
        resumed_joint_training = bool(
            resume_checkpoint.get("arguments", {}).get(
                "joint_representation_dynamics", False
            )
        )
        if resumed_joint_training != bool(args.joint_representation_dynamics):
            raise ValueError(
                "Changing --joint-representation-dynamics while resuming requires "
                "--fresh-optimizer-on-resume because the optimizer parameter set "
                "changes."
            )
        dynamics_optimizer.load_state_dict(resume_checkpoint["optimizer"])
        optimizer_resume = "restored"
    elif resume_checkpoint is not None:
        optimizer_resume = "fresh_for_new_distribution"
        print(
            "using a fresh AdamW optimizer with the trained model weights",
            flush=True,
        )
    else:
        optimizer_resume = "new"
    checkpoint_path = args.output_dir / "best.pt"
    dynamics_history_path = args.output_dir / "dynamics_training.csv"
    dynamics_rows = (
        _read_csv(dynamics_history_path) if resume_checkpoint is not None else []
    )
    source_validation = None
    if resume_checkpoint is not None and not args.reset_selection_on_resume:
        source_validation = (
            resume_metrics["validation"]
            if resume_metrics is not None
            else resume_checkpoint["validation_metrics"]
        )
        source_checkpoint = dict(resume_checkpoint)
        source_checkpoint["source_arguments"] = resume_checkpoint.get("arguments")
        source_checkpoint["arguments"] = arguments
        source_checkpoint["validation_metrics"] = source_validation
        source_checkpoint["optimizer_resume"] = "source_checkpoint"
        source_checkpoint["selection_origin"] = "resume_checkpoint"
        torch.save(source_checkpoint, checkpoint_path)
    best_score = -float("inf")
    if source_validation is not None:
        best_score = _selection_score(
            source_validation, args.horizon, args.selection_metric
        )
    best_epoch = (
        int(resume_checkpoint["epoch"])
        if resume_checkpoint is not None else -1
    )
    start_epoch = best_epoch + 1
    stale_epochs = 0
    print("stage 2: training action-conditioned spatial dynamics", flush=True)
    if start_epoch >= int(args.dynamics_epochs):
        raise ValueError(
            f"Checkpoint is at epoch {best_epoch}, but --dynamics-epochs="
            f"{args.dynamics_epochs} does not request any additional epochs."
        )
    for epoch in range(start_epoch, int(args.dynamics_epochs)):
        model.train()
        components = []
        for batch in loaders["train"]:
            batch = {key: value.to(device) for key, value in batch.items()}
            roots = sample_roots(
                batch["actions"].shape[1],
                args.horizon,
                args.roots_per_episode,
                device,
            )
            loss, row = stage_two_loss(model, batch, roots, args)
            dynamics_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dynamics_parameters, args.grad_clip)
            dynamics_optimizer.step()
            components.append(row)
        mean = _mean_components(components)
        validation = evaluate_model(
            model,
            loaders["validation"],
            device,
            args.horizon,
            args.evaluation_roots_per_episode,
        )
        score = _selection_score(
            validation, args.horizon, args.selection_metric
        )
        depth1 = validation["by_depth"]["1"]
        final = validation["by_depth"][str(args.horizon)]
        dynamics_rows.append(
            {
                "epoch": epoch,
                **{f"train_{key}": value for key, value in mean.items()},
                "selection_score": score,
                "validation_depth1_rollout_mae": depth1[
                    "rollout_reconstruction"
                ]["mae"],
                "validation_depth1_copy_mae": depth1["copy_root_baseline"]["mae"],
                "validation_depth5_rollout_mae": final[
                    "rollout_reconstruction"
                ]["mae"],
                "validation_depth5_copy_mae": final["copy_root_baseline"]["mae"],
                "validation_depth5_changed_top1": final[
                    "changed_rgb_retrieval_top1"
                ],
                "validation_depth1_novelty_mae": depth1[
                    "novelty_prediction_mae"
                ],
                "validation_depth5_novelty_mae": final[
                    "novelty_prediction_mae"
                ],
                "validation_depth1_novelty_correlation": depth1[
                    "novelty_prediction_correlation"
                ],
                "validation_depth5_novelty_correlation": final[
                    "novelty_prediction_correlation"
                ],
                "validation_mean_rgb_novelty_auc": _mean_rgb_novelty_auc(
                    validation
                ),
                "validation_mean_position_auc": validation[
                    "mean_position_novelty_auc"
                ],
            }
        )
        if model.supports_reconstruction:
            reconstruction_summary = (
                f"d1_mae={depth1['rollout_reconstruction']['mae']:.4f}/"
                f"copy={depth1['copy_root_baseline']['mae']:.4f} "
                f"d{args.horizon}_mae="
                f"{final['rollout_reconstruction']['mae']:.4f}/"
                f"copy={final['copy_root_baseline']['mae']:.4f} "
            )
        else:
            reconstruction_summary = "decoder=none "
        representation_summary = (
            f"representation={mean['representation_loss']:.4f} "
            if args.joint_representation_dynamics
            else ""
        )
        print(
            f"dynamics epoch {epoch:02d} loss={mean['loss']:.4f} "
            f"{representation_summary}"
            f"{reconstruction_summary}"
            f"novelty_mae={depth1['novelty_prediction_mae']:.4f}/"
            f"{final['novelty_prediction_mae']:.4f} "
            f"rgb_auc={_mean_rgb_novelty_auc(validation):.3f} "
            f"selection={score:.3f} "
            f"position_auc={validation['mean_position_novelty_auc']:.3f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "arguments": arguments,
                    "validation_metrics": validation,
                    "optimizer": dynamics_optimizer.state_dict(),
                    "optimizer_resume": optimizer_resume,
                    "selection_origin": "adapted_epoch",
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= int(args.patience):
                print(f"early stopping after epoch {epoch}", flush=True)
                break
    _write_csv(dynamics_history_path, dynamics_rows)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    metrics = {
        name: evaluate_model(
            model,
            loaders[name],
            device,
            args.horizon,
            args.evaluation_roots_per_episode,
        )
        for name in ("validation", "test", "heldout")
    }
    heldout = metrics["heldout"]
    heldout_depth1 = heldout["by_depth"]["1"]
    heldout_depth5 = heldout["by_depth"][str(args.horizon)]
    acceptance = {
        "rollout_beats_copy_at_every_depth": bool(
            heldout["rollout_beats_copy_at_every_depth"]
        ),
        "depth1_changed_rgb_retrieval_at_least_0_85": bool(
            heldout_depth1["changed_rgb_retrieval_top1"] >= 0.85
        ),
        "depth5_changed_rgb_retrieval_at_least_0_65": bool(
            heldout_depth5["changed_rgb_retrieval_top1"] >= 0.65
        ),
        "mean_position_novelty_auc_at_least_0_75": bool(
            heldout["mean_position_novelty_auc"] >= 0.75
        ),
    }
    if args.selection_metric in ("key_novelty", "novelty_fidelity"):
        acceptance["passed"] = bool(
            acceptance["depth1_changed_rgb_retrieval_at_least_0_85"]
            and acceptance["depth5_changed_rgb_retrieval_at_least_0_65"]
            and acceptance["mean_position_novelty_auc_at_least_0_75"]
        )
    else:
        acceptance["passed"] = bool(all(acceptance.values()))
    figures = write_visual_grids(
        model,
        datasets["heldout"],
        args.output_dir,
        device,
        args.horizon,
        args.figure_count,
    )
    results = {
        "best_epoch": int(best_epoch),
        "best_validation_selection_score": float(best_score),
        "selection_metric": args.selection_metric,
        "selection_metric_description": {
            "key_novelty": (
                "mean_rgb_novelty_auc_plus_depth5_changed_rgb_retrieval_top1"
            ),
            "novelty_fidelity": (
                "mean_predicted_teacher_novelty_correlation_minus_mean_"
                "novelty_prediction_mae"
            ),
            "rollout_reconstruction": (
                "mean_relative_rollout_mae_gain_plus_"
                "depth5_changed_rgb_retrieval_top1"
            ),
        }[args.selection_metric],
        "best_checkpoint_origin": checkpoint.get(
            "selection_origin", "legacy_checkpoint"
        ),
        "metrics": metrics,
        "acceptance": acceptance,
        "figures": figures,
        "training_target_source": "executed_observation_action_trajectories_only",
        "physical_position_used_for_training": False,
        "counterfactual_transition_queries": 0,
        "reconstruction_target": (
            "full_resolution_observation"
            if args.representation_mode == "spatial_reconstruction"
            else (
                f"pooled_{args.coarse_reconstruction_size}x"
                f"{args.coarse_reconstruction_size}_observation_from_key"
                if args.representation_mode == "compressed_key_reconstruction"
                else "none_inverse_dynamics_embedding"
            )
        ),
    }
    result_path = args.output_dir / "metrics.json"
    result_path.write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    print(f"best epoch: {best_epoch}", flush=True)
    for name in ("test", "heldout"):
        split = metrics[name]
        print(f"{name}:", flush=True)
        for depth, row in split["by_depth"].items():
            position_auc = row["position_novelty_auc"]
            position_auc_text = (
                "undefined" if position_auc is None else f"{position_auc:.3f}"
            )
            if model.supports_reconstruction:
                reconstruction_summary = (
                    f"rollout_mae={row['rollout_reconstruction']['mae']:.4f} "
                    f"copy={row['copy_root_baseline']['mae']:.4f} "
                )
            else:
                reconstruction_summary = "decoder=none "
            print(
                f"  d{depth}: {reconstruction_summary}"
                f"cos={row['alignment_cosine_mean']:.3f} "
                f"rgb_top1={row['rgb_retrieval_top1']:.3f} "
                f"position_auc={position_auc_text}",
                flush=True,
            )
    print(f"acceptance: {json.dumps(acceptance, sort_keys=True)}", flush=True)
    print(f"metrics: {result_path}", flush=True)


if __name__ == "__main__":
    main()
