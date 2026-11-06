"""Predictive exploration embeddings with an explicit episodic memory."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class PredictiveMemoryContext:
    """Real root state and all real exploration embeddings seen this episode."""

    hidden: torch.Tensor
    cell: torch.Tensor
    current_embedding: torch.Tensor
    memory_embeddings: torch.Tensor
    memory_distance_samples: torch.Tensor
    distance_scale: torch.Tensor
    distance_threshold: torch.Tensor
    distance_temperature: torch.Tensor

    def detach(self) -> "PredictiveMemoryContext":
        return PredictiveMemoryContext(
            hidden=self.hidden.detach(),
            cell=self.cell.detach(),
            current_embedding=self.current_embedding.detach(),
            memory_embeddings=self.memory_embeddings.detach(),
            memory_distance_samples=self.memory_distance_samples.detach(),
            distance_scale=self.distance_scale.detach(),
            distance_threshold=self.distance_threshold.detach(),
            distance_temperature=self.distance_temperature.detach(),
        )


def _inverse_softplus(value: float) -> float:
    value = max(float(value), 1e-6)
    return math.log(math.expm1(value))


def balanced_memory_distance_loss(
    predicted_distances: torch.Tensor,
    target_distances: torch.Tensor,
    distance_scale: torch.Tensor,
    zero_distance_epsilon: float = 1e-6,
) -> torch.Tensor:
    """Match raw memory distances while weighting revisits and novel states equally."""
    if predicted_distances.shape != target_distances.shape:
        raise ValueError(
            "Predicted and target memory distances must have equal shapes, "
            f"got {tuple(predicted_distances.shape)} and "
            f"{tuple(target_distances.shape)}."
        )
    scale = torch.as_tensor(
        distance_scale,
        device=predicted_distances.device,
        dtype=predicted_distances.dtype,
    ).detach().clamp_min(1e-6)
    detached_targets = target_distances.detach()
    elementwise_loss = F.smooth_l1_loss(
        predicted_distances / scale,
        detached_targets / scale,
        reduction='none',
    )
    novel_mask = detached_targets > float(zero_distance_epsilon)
    revisit_mask = ~novel_mask
    group_losses = []
    if bool(novel_mask.any()):
        group_losses.append(elementwise_loss[novel_mask].mean())
    if bool(revisit_mask.any()):
        group_losses.append(elementwise_loss[revisit_mask].mean())
    if not group_losses:
        return elementwise_loss.mean()
    return torch.stack(group_losses).mean()


class PredictiveEpisodicMemory(nn.Module):
    """Predict future exploration codes and compare them with real memory.

    The model never receives MuZero's imagined latents. A dedicated observation
    encoder supplies memory keys, while an action-conditioned dynamics model
    predicts keys for short search paths. Novelty is then a differentiable
    nearest-neighbour lookup rather than a free scalar output head.
    """

    def __init__(
        self,
        observation_channels: int,
        action_space_size: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        initial_distance_threshold: float = 0.08,
        initial_distance_temperature: float = 0.02,
        adaptive_distance_scale: bool = False,
        adaptive_threshold_ratio: float = 0.5,
        adaptive_temperature_ratio: float = 0.125,
        adaptive_minimum_scale: float = 1e-3,
        adaptive_distinct_epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        self.observation_channels = int(observation_channels)
        self.action_space_size = int(action_space_size)
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.adaptive_distance_scale = bool(adaptive_distance_scale)
        self.adaptive_threshold_ratio = float(adaptive_threshold_ratio)
        self.adaptive_temperature_ratio = float(adaptive_temperature_ratio)
        self.adaptive_minimum_scale = float(adaptive_minimum_scale)
        self.adaptive_distinct_epsilon = float(adaptive_distinct_epsilon)
        if self.observation_channels <= 0:
            raise ValueError("observation_channels must be positive.")
        if self.action_space_size <= 0:
            raise ValueError("action_space_size must be positive.")
        if self.adaptive_threshold_ratio <= 0.0:
            raise ValueError("adaptive_threshold_ratio must be positive.")
        if self.adaptive_temperature_ratio <= 0.0:
            raise ValueError("adaptive_temperature_ratio must be positive.")
        if self.adaptive_minimum_scale <= 0.0:
            raise ValueError("adaptive_minimum_scale must be positive.")
        if self.adaptive_distinct_epsilon < 0.0:
            raise ValueError("adaptive_distinct_epsilon must be non-negative.")

        action_dim = max(8, self.hidden_dim // 8)
        self.observation_encoder = nn.Sequential(
            nn.Conv2d(self.observation_channels, 24, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(64 * 2 * 2, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
        )
        self.target_observation_encoder = copy.deepcopy(self.observation_encoder)
        self.target_observation_encoder.requires_grad_(False)

        self.action_embedding = nn.Embedding(self.action_space_size + 1, action_dim)
        self.history_cell = nn.LSTMCell(
            self.embedding_dim + action_dim + 2,
            self.hidden_dim,
        )
        self.dynamics_initial = nn.Sequential(
            nn.Linear(self.hidden_dim + self.embedding_dim, self.hidden_dim),
            nn.Tanh(),
        )
        self.dynamics_cell = nn.GRUCell(action_dim, self.hidden_dim)
        self.future_embedding_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.embedding_dim),
        )
        self.inverse_action_head = nn.Sequential(
            nn.Linear(2 * self.embedding_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.action_space_size),
        )

        self.distance_threshold_unconstrained = nn.Parameter(
            torch.tensor(_inverse_softplus(initial_distance_threshold)),
            requires_grad=False,
        )
        self.distance_temperature_unconstrained = nn.Parameter(
            torch.tensor(_inverse_softplus(initial_distance_temperature)),
            requires_grad=False,
        )

    @property
    def distance_threshold(self) -> torch.Tensor:
        return F.softplus(self.distance_threshold_unconstrained).clamp_min(1e-4)

    @property
    def distance_temperature(self) -> torch.Tensor:
        return F.softplus(self.distance_temperature_unconstrained).clamp_min(1e-4)

    def _memory_distance_calibration(
        self,
        real_memory: torch.Tensor,
        distance_samples: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return scale, threshold, and temperature for one real memory bank."""
        if real_memory.ndim != 2:
            raise ValueError("Expected real memory shaped [count, embedding_dim].")
        fallback_threshold = self.distance_threshold.to(real_memory.device)
        fallback_temperature = self.distance_temperature.to(real_memory.device)
        if not self.adaptive_distance_scale:
            return fallback_threshold, fallback_threshold, fallback_temperature

        with torch.no_grad():
            if distance_samples is None:
                distance_samples = self._causal_nearest_memory_distances(real_memory)
            nearest_distinct = distance_samples[
                distance_samples > self.adaptive_distinct_epsilon
            ]
            if int(nearest_distinct.numel()) == 0:
                scale = real_memory.new_tensor(self.adaptive_minimum_scale)
            else:
                scale = nearest_distinct.median().clamp(
                    min=self.adaptive_minimum_scale,
                    max=2.0,
                )
            threshold = (scale * self.adaptive_threshold_ratio).clamp_min(1e-4)
            temperature = (scale * self.adaptive_temperature_ratio).clamp_min(1e-4)
        return scale, threshold, temperature

    @staticmethod
    def _causal_nearest_memory_distances(real_memory: torch.Tensor) -> torch.Tensor:
        """Distance from each real key after the first to its closest earlier key."""
        if real_memory.ndim != 2:
            raise ValueError("Expected real memory shaped [count, embedding_dim].")
        count = int(real_memory.shape[0])
        if count < 2:
            return real_memory.new_empty(0)
        similarities = torch.matmul(real_memory, real_memory.transpose(0, 1))
        prior_mask = torch.tril(
            torch.ones(
                count,
                count,
                dtype=torch.bool,
                device=real_memory.device,
            ),
            diagonal=-1,
        )
        nearest_prior_similarity = similarities.masked_fill(
            ~prior_mask,
            -torch.inf,
        ).amax(dim=1)[1:]
        return (1.0 - nearest_prior_similarity).clamp(min=0.0, max=2.0)

    @staticmethod
    def _normalize(embedding: torch.Tensor) -> torch.Tensor:
        return F.normalize(embedding, p=2, dim=-1, eps=1e-8)

    def _check_observations(self, observations: torch.Tensor) -> None:
        if observations.ndim != 4:
            raise ValueError(
                "Expected observations shaped [batch, channels, height, width], "
                f"got {tuple(observations.shape)}."
            )
        if int(observations.shape[1]) != self.observation_channels:
            raise ValueError(
                f"Expected {self.observation_channels} channels, "
                f"got {int(observations.shape[1])}."
            )

    def encode_observations(self, observations: torch.Tensor) -> torch.Tensor:
        self._check_observations(observations)
        return self._normalize(self.observation_encoder(observations.float()))

    def encode_memory_observations(self, observations: torch.Tensor) -> torch.Tensor:
        self._check_observations(observations)
        with torch.no_grad():
            return self._normalize(
                self.target_observation_encoder(observations.float())
            )

    def update_target_encoder(self, decay: float) -> None:
        decay = float(decay)
        if not 0.0 <= decay <= 1.0:
            raise ValueError("Target encoder decay must be in [0, 1].")
        with torch.no_grad():
            for target, online in zip(
                self.target_observation_encoder.parameters(),
                self.observation_encoder.parameters(),
            ):
                target.mul_(decay).add_(online, alpha=1.0 - decay)
            for target, online in zip(
                self.target_observation_encoder.buffers(),
                self.observation_encoder.buffers(),
            ):
                target.copy_(online)

    def _history_step(
        self,
        observation_embedding: torch.Tensor,
        previous_action: torch.Tensor,
        previous_episodic_reward: torch.Tensor,
        previous_extrinsic_reward: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        recurrent_input = torch.cat(
            [
                observation_embedding,
                self.action_embedding(previous_action),
                previous_episodic_reward.reshape(-1, 1),
                previous_extrinsic_reward.reshape(-1, 1),
            ],
            dim=-1,
        )
        return self.history_cell(recurrent_input, (hidden, cell))

    def start_context(self, observation: torch.Tensor) -> PredictiveMemoryContext:
        online_embedding = self.encode_observations(observation)
        memory_embedding = self.encode_memory_observations(observation)
        batch_size = int(observation.shape[0])
        hidden = torch.zeros(
            batch_size,
            self.hidden_dim,
            device=observation.device,
            dtype=torch.float32,
        )
        cell = torch.zeros_like(hidden)
        start_action = torch.full(
            (batch_size,),
            self.action_space_size,
            device=observation.device,
            dtype=torch.long,
        )
        zero_reward = torch.zeros(batch_size, device=observation.device)
        hidden, cell = self._history_step(
            online_embedding,
            start_action,
            zero_reward,
            zero_reward,
            hidden,
            cell,
        )
        memory_distance_samples = memory_embedding.new_empty(0)
        distance_calibration = self._memory_distance_calibration(
            memory_embedding,
            memory_distance_samples,
        )
        return PredictiveMemoryContext(
            hidden=hidden,
            cell=cell,
            current_embedding=online_embedding,
            memory_embeddings=memory_embedding,
            memory_distance_samples=memory_distance_samples,
            distance_scale=distance_calibration[0],
            distance_threshold=distance_calibration[1],
            distance_temperature=distance_calibration[2],
        )

    def advance_context(
        self,
        context: PredictiveMemoryContext,
        action: torch.Tensor,
        observation: torch.Tensor,
        episodic_reward: torch.Tensor,
        extrinsic_reward: torch.Tensor,
    ) -> PredictiveMemoryContext:
        online_embedding = self.encode_observations(observation)
        memory_embedding = self.encode_memory_observations(observation)
        hidden, cell = self._history_step(
            online_embedding,
            action.to(device=observation.device, dtype=torch.long).reshape(-1),
            episodic_reward.to(device=observation.device, dtype=torch.float32).reshape(-1),
            extrinsic_reward.to(device=observation.device, dtype=torch.float32).reshape(-1),
            context.hidden.to(observation.device),
            context.cell.to(observation.device),
        )
        previous_memory = context.memory_embeddings.to(observation.device)
        memory_embeddings = torch.cat(
            [previous_memory, memory_embedding],
            dim=0,
        )
        new_distance_sample = (
            1.0 - torch.matmul(previous_memory, memory_embedding[0]).amax()
        ).clamp(min=0.0, max=2.0)
        memory_distance_samples = torch.cat(
            [
                context.memory_distance_samples.to(observation.device),
                new_distance_sample.reshape(1),
            ]
        )
        distance_calibration = self._memory_distance_calibration(
            memory_embeddings,
            memory_distance_samples,
        )
        return PredictiveMemoryContext(
            hidden=hidden,
            cell=cell,
            current_embedding=online_embedding,
            memory_embeddings=memory_embeddings,
            memory_distance_samples=memory_distance_samples,
            distance_scale=distance_calibration[0],
            distance_threshold=distance_calibration[1],
            distance_temperature=distance_calibration[2],
        )

    def _history_states_from_embeddings(
        self,
        embeddings: torch.Tensor,
        actions: torch.Tensor,
        episodic_rewards: torch.Tensor,
        extrinsic_rewards: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, observation_count = embeddings.shape[:2]
        hidden = torch.zeros(
            batch_size,
            self.hidden_dim,
            device=embeddings.device,
            dtype=torch.float32,
        )
        cell = torch.zeros_like(hidden)
        start_action = torch.full(
            (batch_size,),
            self.action_space_size,
            device=embeddings.device,
            dtype=torch.long,
        )
        zero_reward = torch.zeros(batch_size, device=embeddings.device)
        states = []
        for index in range(observation_count):
            if index == 0:
                previous_action = start_action
                previous_episodic_reward = zero_reward
                previous_extrinsic_reward = zero_reward
            else:
                previous_action = actions[:, index - 1]
                previous_episodic_reward = episodic_rewards[:, index - 1]
                previous_extrinsic_reward = extrinsic_rewards[:, index - 1]
            hidden, cell = self._history_step(
                embeddings[:, index],
                previous_action,
                previous_episodic_reward,
                previous_extrinsic_reward,
                hidden,
                cell,
            )
            states.append(hidden)
        return torch.stack(states, dim=1)

    def _predict_from_roots(
        self,
        root_hidden: torch.Tensor,
        root_embedding: torch.Tensor,
        action_suffixes: torch.Tensor,
    ) -> torch.Tensor:
        if root_hidden.ndim != 2 or action_suffixes.ndim != 2:
            raise ValueError("Expected roots [N,H] and action suffixes [N,K].")
        dynamics_state = self.dynamics_initial(
            torch.cat([root_hidden, root_embedding], dim=-1)
        )
        predictions = []
        for depth in range(int(action_suffixes.shape[1])):
            dynamics_state = self.dynamics_cell(
                self.action_embedding(action_suffixes[:, depth]),
                dynamics_state,
            )
            predictions.append(
                self._normalize(self.future_embedding_head(dynamics_state))
            )
        return torch.stack(predictions, dim=1)

    def _novelty_logits_for_predictions(
        self,
        predicted_embeddings: torch.Tensor,
        real_memory: torch.Tensor,
        distance_calibration: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if predicted_embeddings.ndim != 2 or real_memory.ndim != 2:
            raise ValueError("Predictions and memory must be rank-two tensors.")
        if distance_calibration is None:
            distance_calibration = self._memory_distance_calibration(real_memory)
        _, distance_threshold, distance_temperature = distance_calibration
        keys = real_memory
        logits = []
        distances = []
        for depth in range(int(predicted_embeddings.shape[0])):
            query = predicted_embeddings[depth]
            if int(keys.shape[0]) == 0:
                distance = query.new_tensor(2.0)
            else:
                maximum_similarity = torch.matmul(keys, query).amax()
                distance = (1.0 - maximum_similarity).clamp(min=0.0, max=2.0)
            logits.append(
                (distance - distance_threshold) / distance_temperature
            )
            distances.append(distance)
            keys = torch.cat([keys, query.reshape(1, -1)], dim=0)
        return torch.stack(logits), torch.stack(distances)

    def score_path_endpoints(
        self,
        root_contexts: Sequence[PredictiveMemoryContext],
        root_indices: Sequence[int],
        action_paths: Sequence[Sequence[int]],
    ) -> torch.Tensor:
        if len(root_indices) != len(action_paths):
            raise ValueError("root_indices and action_paths must have equal lengths.")
        if not action_paths:
            return torch.empty(0, device=next(self.parameters()).device)
        result = torch.empty(len(action_paths), device=next(self.parameters()).device)
        paths_by_length = {}
        for row, path in enumerate(action_paths):
            if len(path) <= 0:
                raise ValueError("MCTS action paths cannot be empty.")
            paths_by_length.setdefault(len(path), []).append(row)
        for length, rows in paths_by_length.items():
            contexts = [root_contexts[int(root_indices[row])] for row in rows]
            root_hidden = torch.cat([context.hidden for context in contexts], dim=0)
            root_embedding = torch.cat(
                [context.current_embedding for context in contexts], dim=0
            )
            actions = torch.as_tensor(
                [action_paths[row] for row in rows],
                dtype=torch.long,
                device=root_hidden.device,
            )
            predicted = self._predict_from_roots(
                root_hidden, root_embedding, actions
            )
            for local_row, output_row in enumerate(rows):
                path_logits, _ = self._novelty_logits_for_predictions(
                    predicted[local_row],
                    contexts[local_row].memory_embeddings,
                    (
                        contexts[local_row].distance_scale,
                        contexts[local_row].distance_threshold,
                        contexts[local_row].distance_temperature,
                    ),
                )
                result[output_row] = path_logits[int(length) - 1]
        return result

    def training_outputs(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        episodic_rewards: torch.Tensor,
        extrinsic_rewards: torch.Tensor,
        root_indices: torch.Tensor,
        horizon: int,
        freeze_online_encoder: bool = False,
    ) -> dict:
        if observations.ndim != 5 or actions.ndim != 2:
            raise ValueError("Expected observations [B,T+1,C,H,W] and actions [B,T].")
        batch_size, observation_count = observations.shape[:2]
        flattened = observations.reshape(
            batch_size * observation_count, *observations.shape[2:]
        )
        if freeze_online_encoder:
            with torch.no_grad():
                online_embeddings = self.encode_observations(flattened).reshape(
                    batch_size, observation_count, self.embedding_dim
                )
        else:
            online_embeddings = self.encode_observations(flattened).reshape(
                batch_size, observation_count, self.embedding_dim
            )
        target_embeddings = self.encode_memory_observations(flattened).reshape(
            batch_size, observation_count, self.embedding_dim
        )
        history_states = self._history_states_from_embeddings(
            online_embeddings, actions, episodic_rewards, extrinsic_rewards
        )
        batch_indices = torch.arange(batch_size, device=actions.device)[:, None]
        root_hidden = history_states[batch_indices, root_indices]
        root_embeddings = online_embeddings[batch_indices, root_indices]
        suffix_actions = torch.stack(
            [actions[batch_indices, root_indices + depth] for depth in range(horizon)],
            dim=-1,
        )
        root_count = int(root_indices.shape[1])
        predicted = self._predict_from_roots(
            root_hidden.reshape(batch_size * root_count, self.hidden_dim),
            root_embeddings.reshape(batch_size * root_count, self.embedding_dim),
            suffix_actions.reshape(batch_size * root_count, horizon),
        ).reshape(batch_size, root_count, horizon, self.embedding_dim)
        target_future = torch.stack(
            [
                target_embeddings[batch_indices, root_indices + depth + 1]
                for depth in range(horizon)
            ],
            dim=2,
        )

        logits = torch.empty(
            batch_size,
            root_count,
            horizon,
            device=actions.device,
            dtype=predicted.dtype,
        )
        predicted_distances = torch.empty_like(logits)
        target_distances = torch.empty_like(logits)
        target_novelty_logits = torch.empty_like(logits)
        distance_scales = torch.empty(
            batch_size,
            root_count,
            device=actions.device,
            dtype=predicted.dtype,
        )
        distance_thresholds = torch.empty_like(distance_scales)
        distance_temperatures = torch.empty_like(distance_scales)
        causal_memory_distance_samples = [
            self._causal_nearest_memory_distances(target_embeddings[batch_index])
            for batch_index in range(batch_size)
        ]
        for batch_index in range(batch_size):
            for root_offset in range(root_count):
                root = int(root_indices[batch_index, root_offset].item())
                memory = target_embeddings[batch_index, :root + 1]
                distance_calibration = self._memory_distance_calibration(
                    memory,
                    causal_memory_distance_samples[batch_index][:root],
                )
                path_logits, path_distances = self._novelty_logits_for_predictions(
                    predicted[batch_index, root_offset],
                    memory,
                    distance_calibration,
                )
                real_logits, real_distances = self._novelty_logits_for_predictions(
                    target_future[batch_index, root_offset],
                    memory,
                    distance_calibration,
                )
                logits[batch_index, root_offset] = path_logits
                predicted_distances[batch_index, root_offset] = path_distances
                target_distances[batch_index, root_offset] = real_distances
                target_novelty_logits[batch_index, root_offset] = real_logits
                distance_scales[batch_index, root_offset] = distance_calibration[0]
                distance_thresholds[batch_index, root_offset] = distance_calibration[1]
                distance_temperatures[batch_index, root_offset] = distance_calibration[2]

        inverse_logits = self.inverse_action_head(
            torch.cat(
                [online_embeddings[:, :-1], online_embeddings[:, 1:]], dim=-1
            )
        )
        return {
            'novelty_logits': logits,
            'predicted_embeddings': predicted,
            'target_embeddings': target_future,
            'target_novelty_probabilities': torch.sigmoid(
                target_novelty_logits.detach()
            ),
            'predicted_distances': predicted_distances,
            'target_distances': target_distances,
            'distance_scales': distance_scales,
            'distance_thresholds': distance_thresholds,
            'distance_temperatures': distance_temperatures,
            'inverse_logits': inverse_logits,
            '_online_embeddings': online_embeddings,
        }

    def score_actual_suffixes(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        episodic_rewards: torch.Tensor,
        extrinsic_rewards: torch.Tensor,
        root_indices: torch.Tensor,
        horizon: int,
    ) -> torch.Tensor:
        return self.training_outputs(
            observations,
            actions,
            episodic_rewards,
            extrinsic_rewards,
            root_indices,
            int(horizon),
        )['novelty_logits']


class ReconstructiveEpisodicMemory(PredictiveEpisodicMemory):
    """Predictive episodic memory with an observation-grounded latent space.

    The parent model already keeps its exploration representation separate from
    MuZero. This variant adds the missing representational contract: real
    exploration embeddings must reconstruct their observations, and imagined
    embeddings must reconstruct the real future observations reached by the
    corresponding action suffixes.
    """

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_space_size: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        decoder_channels: int = 64,
        **kwargs,
    ) -> None:
        observation_shape = tuple(int(value) for value in observation_shape)
        if len(observation_shape) != 3:
            raise ValueError(
                "observation_shape must contain channels, height, and width."
            )
        if min(observation_shape) <= 0:
            raise ValueError("All observation dimensions must be positive.")
        decoder_channels = int(decoder_channels)
        if decoder_channels <= 0:
            raise ValueError("decoder_channels must be positive.")
        super().__init__(
            observation_channels=observation_shape[0],
            action_space_size=action_space_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            **kwargs,
        )
        self.observation_shape = observation_shape
        self.decoder_channels = decoder_channels
        seed_height = min(7, observation_shape[1])
        seed_width = min(7, observation_shape[2])
        self._decoder_seed_shape = (seed_height, seed_width)
        middle_channels = max(16, decoder_channels // 2)
        output_channels = max(12, decoder_channels // 4)
        self.decoder_input = nn.Sequential(
            nn.Linear(
                self.embedding_dim,
                decoder_channels * seed_height * seed_width,
            ),
            nn.ReLU(inplace=True),
        )
        self.decoder_upsampler = nn.Sequential(
            nn.ConvTranspose2d(
                decoder_channels,
                decoder_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                decoder_channels,
                middle_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                middle_channels,
                output_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )
        self.decoder_output = nn.Conv2d(
            output_channels,
            observation_shape[0],
            kernel_size=3,
            padding=1,
        )

    def decode_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dimension {self.embedding_dim}, "
                f"got {int(embeddings.shape[-1])}."
            )
        leading_shape = embeddings.shape[:-1]
        flattened = embeddings.reshape(-1, self.embedding_dim)
        seed_height, seed_width = self._decoder_seed_shape
        decoded = self.decoder_input(flattened).reshape(
            -1,
            self.decoder_channels,
            seed_height,
            seed_width,
        )
        decoded = self.decoder_upsampler(decoded)
        decoded = F.interpolate(
            decoded,
            size=self.observation_shape[1:],
            mode='bilinear',
            align_corners=False,
        )
        decoded = torch.sigmoid(self.decoder_output(decoded))
        return decoded.reshape(*leading_shape, *self.observation_shape)

    def training_outputs(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        episodic_rewards: torch.Tensor,
        extrinsic_rewards: torch.Tensor,
        root_indices: torch.Tensor,
        horizon: int,
        freeze_online_encoder: bool = False,
    ) -> dict:
        outputs = super().training_outputs(
            observations,
            actions,
            episodic_rewards,
            extrinsic_rewards,
            root_indices,
            horizon,
            freeze_online_encoder=freeze_online_encoder,
        )
        batch_size = int(observations.shape[0])
        online_embeddings = outputs.pop('_online_embeddings')

        time_offsets = torch.arange(
            int(horizon) + 1,
            device=root_indices.device,
            dtype=torch.long,
        )
        sampled_indices = root_indices[..., None] + time_offsets
        batch_indices = torch.arange(
            batch_size,
            device=root_indices.device,
            dtype=torch.long,
        )[:, None, None]
        sampled_embeddings = online_embeddings[batch_indices, sampled_indices]
        sampled_observations = observations[batch_indices, sampled_indices]

        outputs.update(
            {
                'reconstructed_observations': self.decode_embeddings(
                    sampled_embeddings
                ),
                'reconstruction_targets': sampled_observations,
                'predicted_future_observations': self.decode_embeddings(
                    outputs['predicted_embeddings']
                ),
                'future_observation_targets': sampled_observations[:, :, 1:],
            }
        )
        return outputs
