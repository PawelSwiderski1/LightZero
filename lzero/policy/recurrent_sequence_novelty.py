"""Recurrent episodic-novelty prediction from real history and action prefixes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def recent_observation_novelty_targets(
    observations: np.ndarray,
    root_indices: np.ndarray,
    horizon: int,
    history_window: int,
) -> np.ndarray:
    """Label actual suffix endpoints against the preceding observation window."""
    observations = np.asarray(observations)
    root_indices = np.asarray(root_indices, dtype=np.int64)
    horizon = int(horizon)
    history_window = int(history_window)
    if observations.ndim < 3:
        raise ValueError("observations must be shaped [batch, time, ...].")
    if root_indices.ndim != 2 or root_indices.shape[0] != observations.shape[0]:
        raise ValueError("root_indices must be shaped [batch, roots].")
    if horizon <= 0 or history_window <= 0:
        raise ValueError("horizon and history_window must be positive.")

    if observations.dtype != np.uint8:
        scale = 255.0 if float(np.max(observations, initial=0.0)) <= 1.5 else 1.0
        observations = np.rint(
            np.clip(observations * scale, 0.0, 255.0)
        ).astype(np.uint8)
    observations = np.ascontiguousarray(observations)

    batch_size, observation_count = observations.shape[:2]
    targets = np.empty(
        (*root_indices.shape, horizon), dtype=np.float32
    )
    for batch_index in range(batch_size):
        identity_by_bytes = {}
        identities = []
        for observation in observations[batch_index]:
            key = observation.tobytes()
            identity = identity_by_bytes.setdefault(key, len(identity_by_bytes))
            identities.append(identity)
        for root_row, root_index in enumerate(root_indices[batch_index]):
            for depth in range(horizon):
                endpoint = int(root_index) + depth + 1
                if endpoint >= observation_count:
                    raise ValueError("A sampled suffix extends past the observations.")
                memory_start = max(0, endpoint - history_window)
                targets[batch_index, root_row, depth] = float(
                    identities[endpoint]
                    not in identities[memory_start:endpoint]
                )
    return targets


def episodic_continuation_factor_targets(
    rnd_rewards: torch.Tensor,
    episodic_rewards: torch.Tensor,
    lengths: torch.Tensor,
    root_indices: torch.Tensor,
    prefix_horizon: int,
    value_horizon: int,
    discount_factor: float,
    minimum_factor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build RND-weighted continuation factors after every actual prefix leaf."""
    if rnd_rewards.ndim != 2 or episodic_rewards.shape != rnd_rewards.shape:
        raise ValueError("Reward tensors must share shape [batch, time].")
    if root_indices.ndim != 2 or root_indices.shape[0] != rnd_rewards.shape[0]:
        raise ValueError("root_indices must be shaped [batch, roots].")
    prefix_horizon = int(prefix_horizon)
    value_horizon = int(value_horizon)
    if prefix_horizon <= 0 or value_horizon <= 0:
        raise ValueError("Both horizons must be positive.")
    minimum_factor = float(minimum_factor)
    if not 0.0 <= minimum_factor < 1.0:
        raise ValueError("minimum_factor must be in [0, 1).")

    batch_size, max_steps = rnd_rewards.shape
    depths = torch.arange(
        1, prefix_horizon + 1, dtype=torch.long, device=rnd_rewards.device
    ).view(1, 1, prefix_horizon, 1)
    continuation_offsets = torch.arange(
        value_horizon, dtype=torch.long, device=rnd_rewards.device
    ).view(1, 1, 1, value_horizon)
    continuation_indices = root_indices[:, :, None, None] + depths + continuation_offsets
    length_tensor = lengths.to(device=rnd_rewards.device, dtype=torch.long).view(
        batch_size, 1, 1, 1
    )
    continuation_mask = continuation_indices < length_tensor
    safe_indices = continuation_indices.clamp(min=0, max=max(0, max_steps - 1))
    batch_indices = torch.arange(
        batch_size, dtype=torch.long, device=rnd_rewards.device
    ).view(batch_size, 1, 1, 1)
    future_rnd = rnd_rewards[batch_indices, safe_indices]
    future_novelty = episodic_rewards[batch_indices, safe_indices]
    weights = torch.pow(
        torch.tensor(
            float(discount_factor), dtype=torch.float32, device=rnd_rewards.device
        ),
        continuation_offsets.float(),
    )
    weighted_mask = weights * continuation_mask.float()
    raw_return = torch.sum(future_rnd * weighted_mask, dim=-1)
    future_factors = minimum_factor + (1.0 - minimum_factor) * future_novelty
    gated_return = torch.sum(future_rnd * future_factors * weighted_mask, dim=-1)
    valid = raw_return > 1e-8
    factor_target = (gated_return / raw_return.clamp_min(1e-8)).clamp(
        min=minimum_factor, max=1.0
    )
    return raw_return, gated_return, factor_target, valid


@dataclass(frozen=True)
class RecurrentSequenceContext:
    """The model-visible summary of one real episode history."""

    hidden: torch.Tensor
    cell: torch.Tensor

    def detach(self) -> "RecurrentSequenceContext":
        return RecurrentSequenceContext(
            hidden=self.hidden.detach(),
            cell=self.cell.detach(),
        )


@dataclass(frozen=True)
class RecentHistoryTransformerContext:
    """The observation/action tokens retained from one recent real history."""

    observation_embeddings: torch.Tensor
    previous_actions: torch.Tensor

    def detach(self) -> "RecentHistoryTransformerContext":
        return RecentHistoryTransformerContext(
            observation_embeddings=self.observation_embeddings.detach(),
            previous_actions=self.previous_actions.detach(),
        )


@dataclass
class RecentHistoryTransformerPrefixCache:
    """Branch-local transformer states and endpoint logits for one MCTS root set."""

    states: list[
        dict[
            tuple[int, ...],
            tuple[list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor],
        ]
    ]
    logits: list[dict[tuple[int, ...], torch.Tensor]]


@dataclass
class RecentHistoryTransformerHistoryCache:
    """Per-layer real-history keys/values plus lazily cached path logits."""

    layer_caches: list[tuple[torch.Tensor, torch.Tensor]]
    padding_mask: torch.Tensor
    logits: list[dict[tuple[int, ...], torch.Tensor]]


class TileAlignedRGBObservationEncoder(nn.Module):
    """Encode each rendered MiniGrid tile before mixing the 7 x 7 grid."""

    def __init__(
        self,
        observation_channels: int,
        embedding_dim: int,
        grid_size: int = 7,
    ) -> None:
        super().__init__()
        self.observation_channels = int(observation_channels)
        self.embedding_dim = int(embedding_dim)
        self.grid_size = int(grid_size)
        self.tile_encoder = nn.Sequential(
            nn.Conv2d(
                self.observation_channels, 24, kernel_size=3, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                64 * self.grid_size * self.grid_size, self.embedding_dim
            ),
            nn.LayerNorm(self.embedding_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = observations.shape
        if channels != self.observation_channels:
            raise ValueError(
                f"Expected {self.observation_channels} RGB channels, got {channels}."
            )
        if height % self.grid_size != 0 or width % self.grid_size != 0:
            raise ValueError(
                "Tile-aligned RGB observations must have spatial dimensions "
                f"divisible by {self.grid_size}, got {(height, width)}."
            )
        tile_height = height // self.grid_size
        tile_width = width // self.grid_size
        tiles = observations.reshape(
            batch_size,
            channels,
            self.grid_size,
            tile_height,
            self.grid_size,
            tile_width,
        ).permute(0, 2, 4, 1, 3, 5)
        tiles = tiles.reshape(
            batch_size * self.grid_size * self.grid_size,
            channels,
            tile_height,
            tile_width,
        )
        tile_features = self.tile_encoder(tiles).reshape(
            batch_size,
            self.grid_size,
            self.grid_size,
            32,
        ).permute(0, 3, 1, 2)
        return self.projection(self.spatial_encoder(tile_features))


class SymbolicGridObservationEncoder(nn.Module):
    """Embed MiniGrid object, color, and state IDs on their native grid."""

    def __init__(
        self,
        embedding_dim: int,
        grid_size: int = 7,
        category_counts: tuple[int, int, int] = (11, 6, 3),
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.grid_size = int(grid_size)
        self.category_counts = tuple(int(count) for count in category_counts)
        category_dims = (16, 8, 4)
        self.category_embeddings = nn.ModuleList(
            nn.Embedding(count, dimension)
            for count, dimension in zip(self.category_counts, category_dims)
        )
        input_channels = sum(category_dims)
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                64 * self.grid_size * self.grid_size, self.embedding_dim
            ),
            nn.LayerNorm(self.embedding_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.shape[1:] != (3, self.grid_size, self.grid_size):
            raise ValueError(
                "Expected symbolic observations shaped "
                f"[batch, 3, {self.grid_size}, {self.grid_size}], got "
                f"{tuple(observations.shape)}."
            )
        categories = observations.to(dtype=torch.long)
        embedded_channels = [
            embedding(categories[:, channel])
            for channel, embedding in enumerate(self.category_embeddings)
        ]
        grid = torch.cat(embedded_channels, dim=-1).permute(0, 3, 1, 2)
        return self.projection(self.spatial_encoder(grid))


class RecentHistoryTransformerNovelty(nn.Module):
    """Predict recent-window revisits directly from history and action prefixes.

    Real observations are available only in the history tokens. Candidate tokens
    contain actions, so the transformer must infer where each candidate prefix
    leads and whether that endpoint occurred in the recent history.
    """

    def __init__(
        self,
        observation_channels: int,
        action_space_size: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        history_window: int = 40,
        max_horizon: int = 5,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        transformer_dropout: float = 0.0,
        observation_encoder_type: str = "pooled_2x2",
    ) -> None:
        super().__init__()
        self.observation_channels = int(observation_channels)
        self.action_space_size = int(action_space_size)
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.history_window = int(history_window)
        self.max_horizon = int(max_horizon)
        self.observation_encoder_type = str(observation_encoder_type).lower()
        if self.observation_channels <= 0 or self.action_space_size <= 0:
            raise ValueError("Observation channels and action space must be positive.")
        if self.history_window <= 0 or self.max_horizon <= 0:
            raise ValueError("history_window and max_horizon must be positive.")
        if self.hidden_dim % int(transformer_heads) != 0:
            raise ValueError("hidden_dim must be divisible by transformer_heads.")

        if self.observation_encoder_type == "pooled_2x2":
            self.observation_encoder = nn.Sequential(
                nn.Conv2d(
                    self.observation_channels,
                    24,
                    kernel_size=5,
                    stride=2,
                    padding=2,
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((2, 2)),
                nn.Flatten(),
                nn.Linear(64 * 2 * 2, self.embedding_dim),
                nn.LayerNorm(self.embedding_dim),
                nn.ReLU(inplace=True),
            )
        elif self.observation_encoder_type == "rgb_tile_7x7":
            self.observation_encoder = TileAlignedRGBObservationEncoder(
                self.observation_channels,
                self.embedding_dim,
            )
        elif self.observation_encoder_type == "symbolic_7x7":
            if self.observation_channels != 3:
                raise ValueError(
                    "The symbolic MiniGrid encoder requires three input channels."
                )
            self.observation_encoder = SymbolicGridObservationEncoder(
                self.embedding_dim
            )
        else:
            raise ValueError(
                "Unsupported observation_encoder_type: "
                f"{self.observation_encoder_type}"
            )
        self.observation_projection = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.action_embedding = nn.Embedding(
            self.action_space_size + 1, self.hidden_dim
        )
        self.token_type_embedding = nn.Embedding(2, self.hidden_dim)
        self.candidate_depth_embedding = nn.Parameter(
            torch.zeros(self.max_horizon, self.hidden_dim)
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(
                self.history_window + self.max_horizon,
                self.hidden_dim,
            )
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=int(transformer_heads),
            dim_feedforward=2 * self.hidden_dim,
            dropout=float(transformer_dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(transformer_layers),
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.novelty_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.normal_(self.position_embedding, std=0.02)
        nn.init.normal_(self.candidate_depth_embedding, std=0.02)

    def _encode_observations(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim != 4:
            raise ValueError(
                "Expected observations shaped [batch, channels, height, width], "
                f"got {tuple(observations.shape)}."
            )
        if int(observations.shape[1]) != self.observation_channels:
            raise ValueError(
                f"Expected {self.observation_channels} observation channels, "
                f"got {int(observations.shape[1])}."
            )
        return self.observation_encoder(observations.float())

    def start_context(
        self, observation: torch.Tensor
    ) -> RecentHistoryTransformerContext:
        batch_size = int(observation.shape[0])
        return RecentHistoryTransformerContext(
            observation_embeddings=self._encode_observations(observation)[:, None, :],
            previous_actions=torch.full(
                (batch_size, 1),
                self.action_space_size,
                dtype=torch.long,
                device=observation.device,
            ),
        )

    def advance_context(
        self,
        context: RecentHistoryTransformerContext,
        action: torch.Tensor,
        observation: torch.Tensor,
        episodic_reward: torch.Tensor,
        extrinsic_reward: torch.Tensor,
    ) -> RecentHistoryTransformerContext:
        del episodic_reward, extrinsic_reward
        embeddings = torch.cat(
            [
                context.observation_embeddings.to(observation.device),
                self._encode_observations(observation)[:, None, :],
            ],
            dim=1,
        )[:, -self.history_window:]
        previous_actions = torch.cat(
            [
                context.previous_actions.to(observation.device),
                action.to(device=observation.device, dtype=torch.long).reshape(-1, 1),
            ],
            dim=1,
        )[:, -self.history_window:]
        return RecentHistoryTransformerContext(
            observation_embeddings=embeddings,
            previous_actions=previous_actions,
        )

    def _score_histories(
        self,
        history_embeddings: torch.Tensor,
        previous_actions: torch.Tensor,
        history_valid: torch.Tensor,
        action_suffixes: torch.Tensor,
    ) -> torch.Tensor:
        if history_embeddings.ndim != 3 or action_suffixes.ndim != 2:
            raise ValueError("Expected histories [N,W,E] and suffixes [N,H].")
        batch_size, history_length = history_embeddings.shape[:2]
        horizon = int(action_suffixes.shape[1])
        if history_length != self.history_window:
            raise ValueError("Histories must be padded to history_window.")
        if horizon <= 0 or horizon > self.max_horizon:
            raise ValueError("Action suffix horizon is outside the configured range.")
        if previous_actions.shape != (batch_size, history_length):
            raise ValueError("previous_actions must match the history tokens.")
        if history_valid.shape != (batch_size, history_length):
            raise ValueError("history_valid must match the history tokens.")

        history_tokens = (
            self.observation_projection(history_embeddings)
            + self.action_embedding(previous_actions)
            + self.token_type_embedding.weight[0]
        )
        candidate_tokens = (
            self.action_embedding(action_suffixes)
            + self.token_type_embedding.weight[1]
            + self.candidate_depth_embedding[:horizon]
        )
        tokens = torch.cat([history_tokens, candidate_tokens], dim=1)
        tokens = tokens + self.position_embedding[:history_length + horizon]
        padding_mask = torch.cat(
            [
                ~history_valid,
                torch.zeros(
                    batch_size,
                    horizon,
                    dtype=torch.bool,
                    device=history_valid.device,
                ),
            ],
            dim=1,
        )
        sequence_length = history_length + horizon
        causal_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=tokens.device,
            ),
            diagonal=1,
        )
        encoded = self.transformer(
            tokens,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )
        candidate_states = self.output_norm(encoded[:, history_length:])
        return self.novelty_head(candidate_states).squeeze(-1)

    def _stack_contexts(
        self, contexts: Sequence[RecentHistoryTransformerContext]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        count = len(contexts)
        embeddings = torch.zeros(
            count,
            self.history_window,
            self.embedding_dim,
            dtype=torch.float32,
            device=device,
        )
        actions = torch.full(
            (count, self.history_window),
            self.action_space_size,
            dtype=torch.long,
            device=device,
        )
        valid = torch.zeros(
            count,
            self.history_window,
            dtype=torch.bool,
            device=device,
        )
        for row, context in enumerate(contexts):
            if int(context.observation_embeddings.shape[0]) != 1:
                raise ValueError("Path scoring expects one environment per context.")
            length = min(
                self.history_window,
                int(context.observation_embeddings.shape[1]),
            )
            embeddings[row, :length] = context.observation_embeddings[0, -length:].to(
                device
            )
            actions[row, :length] = context.previous_actions[0, -length:].to(device)
            valid[row, :length] = True
        return embeddings, actions, valid

    def score_path_endpoints(
        self,
        root_contexts: Sequence[RecentHistoryTransformerContext],
        root_indices: Sequence[int],
        action_paths: Sequence[Sequence[int]],
    ) -> torch.Tensor:
        if len(root_indices) != len(action_paths):
            raise ValueError("root_indices and action_paths must have equal lengths.")
        device = next(self.parameters()).device
        if not action_paths:
            return torch.empty(0, device=device)
        result = torch.empty(len(action_paths), device=device)
        paths_by_length = {}
        for row, path in enumerate(action_paths):
            if len(path) <= 0:
                raise ValueError("MCTS action paths cannot be empty.")
            paths_by_length.setdefault(len(path), []).append(row)
        for length, rows in paths_by_length.items():
            selected_contexts = [
                root_contexts[int(root_indices[row])] for row in rows
            ]
            embeddings, previous_actions, valid = self._stack_contexts(
                selected_contexts
            )
            suffixes = torch.as_tensor(
                [action_paths[row] for row in rows],
                dtype=torch.long,
                device=device,
            )
            row_tensor = torch.as_tensor(rows, dtype=torch.long, device=device)
            result[row_tensor] = self._score_histories(
                embeddings,
                previous_actions,
                valid,
                suffixes,
            )[:, int(length) - 1]
        return result

    def create_history_cache(
        self,
        root_contexts: Sequence[RecentHistoryTransformerContext],
    ) -> RecentHistoryTransformerHistoryCache:
        """Encode the shared real histories once while retaining batched suffixes."""
        if self.training:
            raise RuntimeError("Cached history inference requires evaluation mode.")
        embeddings, previous_actions, history_valid = self._stack_contexts(
            root_contexts
        )
        layer_caches, padding_mask = self._cached_history_state(
            embeddings,
            previous_actions,
            history_valid,
        )
        return RecentHistoryTransformerHistoryCache(
            layer_caches=layer_caches,
            padding_mask=padding_mask,
            logits=[{} for _ in root_contexts],
        )

    def _score_history_cached_suffixes(
        self,
        history_cache: RecentHistoryTransformerHistoryCache,
        root_indices: torch.Tensor,
        action_suffixes: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, horizon = action_suffixes.shape
        hidden = (
            self.action_embedding(action_suffixes)
            + self.token_type_embedding.weight[1]
            + self.candidate_depth_embedding[:horizon]
            + self.position_embedding[
                self.history_window:self.history_window + horizon
            ]
        )
        history_padding = history_cache.padding_mask[root_indices]
        candidate_causal = torch.ones(
            horizon,
            horizon,
            dtype=torch.bool,
            device=hidden.device,
        ).tril()
        attention_mask = torch.cat(
            [
                (~history_padding)[:, None, None, :].expand(
                    batch_size, 1, horizon, self.history_window
                ),
                candidate_causal[None, None].expand(batch_size, 1, horizon, horizon),
            ],
            dim=-1,
        )
        for layer_index, layer in enumerate(self.transformer.layers):
            normalized = layer.norm1(hidden)
            queries, candidate_keys, candidate_values = self._project_attention_qkv(
                layer.self_attn, normalized
            )
            history_keys, history_values = history_cache.layer_caches[layer_index]
            keys = torch.cat([history_keys[root_indices], candidate_keys], dim=2)
            values = torch.cat([history_values[root_indices], candidate_values], dim=2)
            attended = F.scaled_dot_product_attention(
                queries,
                keys,
                values,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            )
            hidden = hidden + layer.dropout1(
                self._merge_attention_heads(layer.self_attn, attended)
            )
            hidden = hidden + self._feed_forward(layer, hidden)
        return self.novelty_head(self.output_norm(hidden)).squeeze(-1)

    def score_history_cached_path_endpoints(
        self,
        history_cache: RecentHistoryTransformerHistoryCache,
        root_indices: Sequence[int],
        action_paths: Sequence[Sequence[int]],
    ) -> torch.Tensor:
        """Score complete suffixes in batches while reusing real-history KV state."""
        if len(root_indices) != len(action_paths):
            raise ValueError("root_indices and action_paths must have equal lengths.")
        device = next(self.parameters()).device
        if not action_paths:
            return torch.empty(0, device=device)
        normalized_paths = [tuple(int(action) for action in path) for path in action_paths]
        missing_by_length: dict[int, list[tuple[int, tuple[int, ...]]]] = {}
        scheduled = set()
        for root_index, path in zip(root_indices, normalized_paths):
            root_index = int(root_index)
            if root_index < 0 or root_index >= len(history_cache.logits):
                raise ValueError("A history-cache root index is out of range.")
            if len(path) <= 0 or len(path) > self.max_horizon:
                raise ValueError("A history-cache action path has an invalid length.")
            if any(action < 0 or action >= self.action_space_size for action in path):
                raise ValueError("A history-cache action path contains an invalid action.")
            key = (root_index, path)
            if path not in history_cache.logits[root_index] and key not in scheduled:
                missing_by_length.setdefault(len(path), []).append(key)
                scheduled.add(key)

        for length, missing in missing_by_length.items():
            selected_roots = torch.as_tensor(
                [root_index for root_index, _ in missing],
                dtype=torch.long,
                device=device,
            )
            suffixes = torch.as_tensor(
                [path for _, path in missing],
                dtype=torch.long,
                device=device,
            )
            logits = self._score_history_cached_suffixes(
                history_cache,
                selected_roots,
                suffixes,
            )[:, length - 1]
            for (root_index, path), logit in zip(missing, logits):
                history_cache.logits[root_index][path] = logit

        return torch.stack(
            [
                history_cache.logits[int(root_index)][path]
                for root_index, path in zip(root_indices, normalized_paths)
            ]
        )

    @staticmethod
    def _project_attention_qkv(
        attention: nn.MultiheadAttention,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if attention.in_proj_weight is None:
            raise ValueError("Cached inference requires shared QKV projection weights.")
        queries, keys, values = F.linear(
            inputs,
            attention.in_proj_weight,
            attention.in_proj_bias,
        ).chunk(3, dim=-1)
        batch_size, sequence_length, embedding_dim = queries.shape
        head_count = int(attention.num_heads)
        head_dim = embedding_dim // head_count

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(
                batch_size, sequence_length, head_count, head_dim
            ).transpose(1, 2)

        return split_heads(queries), split_heads(keys), split_heads(values)

    @staticmethod
    def _merge_attention_heads(
        attention: nn.MultiheadAttention,
        attended: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _, sequence_length, _ = attended.shape
        merged = attended.transpose(1, 2).contiguous().reshape(
            batch_size, sequence_length, int(attention.embed_dim)
        )
        return attention.out_proj(merged)

    @staticmethod
    def _feed_forward(
        layer: nn.TransformerEncoderLayer,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        return layer.dropout2(
            layer.linear2(
                layer.dropout(layer.activation(layer.linear1(layer.norm2(inputs))))
            )
        )

    def _cached_history_state(
        self,
        history_embeddings: torch.Tensor,
        previous_actions: torch.Tensor,
        history_valid: torch.Tensor,
    ) -> tuple[
        list[tuple[torch.Tensor, torch.Tensor]],
        torch.Tensor,
    ]:
        hidden = (
            self.observation_projection(history_embeddings)
            + self.action_embedding(previous_actions)
            + self.token_type_embedding.weight[0]
            + self.position_embedding[:self.history_window]
        )
        sequence_length = int(hidden.shape[1])
        causal = torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool,
            device=hidden.device,
        ).tril()
        attention_mask = causal[None, None] & history_valid[:, None, None, :]
        layer_caches = []
        for layer in self.transformer.layers:
            normalized = layer.norm1(hidden)
            queries, keys, values = self._project_attention_qkv(
                layer.self_attn, normalized
            )
            attended = F.scaled_dot_product_attention(
                queries,
                keys,
                values,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            )
            hidden = hidden + layer.dropout1(
                self._merge_attention_heads(layer.self_attn, attended)
            )
            hidden = hidden + self._feed_forward(layer, hidden)
            layer_caches.append((keys, values))
        return layer_caches, ~history_valid

    def _advance_cached_prefixes(
        self,
        layer_caches: Sequence[tuple[torch.Tensor, torch.Tensor]],
        padding_mask: torch.Tensor,
        actions: torch.Tensor,
        depth: int,
    ) -> tuple[
        torch.Tensor,
        list[tuple[torch.Tensor, torch.Tensor]],
        torch.Tensor,
    ]:
        hidden = (
            self.action_embedding(actions)[:, None, :]
            + self.token_type_embedding.weight[1]
            + self.candidate_depth_embedding[depth - 1]
            + self.position_embedding[self.history_window + depth - 1]
        )
        next_padding_mask = torch.cat(
            [
                padding_mask,
                torch.zeros(
                    int(padding_mask.shape[0]),
                    1,
                    dtype=torch.bool,
                    device=padding_mask.device,
                ),
            ],
            dim=1,
        )
        attention_mask = (~next_padding_mask)[:, None, None, :]
        next_layer_caches = []
        for layer, (cached_keys, cached_values) in zip(
            self.transformer.layers, layer_caches
        ):
            normalized = layer.norm1(hidden)
            queries, new_keys, new_values = self._project_attention_qkv(
                layer.self_attn, normalized
            )
            keys = torch.cat([cached_keys, new_keys], dim=2)
            values = torch.cat([cached_values, new_values], dim=2)
            attended = F.scaled_dot_product_attention(
                queries,
                keys,
                values,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            )
            hidden = hidden + layer.dropout1(
                self._merge_attention_heads(layer.self_attn, attended)
            )
            hidden = hidden + self._feed_forward(layer, hidden)
            next_layer_caches.append((keys, values))
        logits = self.novelty_head(self.output_norm(hidden[:, 0])).squeeze(-1)
        return logits, next_layer_caches, next_padding_mask

    def create_prefix_cache(
        self,
        root_contexts: Sequence[RecentHistoryTransformerContext],
    ) -> RecentHistoryTransformerPrefixCache:
        """Encode each real history once for incremental branch inference."""
        if self.training:
            raise RuntimeError("Cached prefix inference requires evaluation mode.")
        embeddings, previous_actions, history_valid = self._stack_contexts(
            root_contexts
        )
        layer_caches, padding_mask = self._cached_history_state(
            embeddings,
            previous_actions,
            history_valid,
        )
        states = []
        for root_index in range(len(root_contexts)):
            states.append(
                {
                    tuple(): (
                        [
                            (
                                keys[root_index:root_index + 1],
                                values[root_index:root_index + 1],
                            )
                            for keys, values in layer_caches
                        ],
                        padding_mask[root_index:root_index + 1],
                    )
                }
            )
        return RecentHistoryTransformerPrefixCache(
            states=states,
            logits=[{} for _ in root_contexts],
        )

    def score_cached_path_endpoints(
        self,
        prefix_cache: RecentHistoryTransformerPrefixCache,
        root_indices: Sequence[int],
        action_paths: Sequence[Sequence[int]],
    ) -> torch.Tensor:
        """Incrementally score endpoints, reusing cached ancestors and histories."""
        if len(root_indices) != len(action_paths):
            raise ValueError("root_indices and action_paths must have equal lengths.")
        device = next(self.parameters()).device
        if not action_paths:
            return torch.empty(0, device=device)
        normalized_paths = [tuple(int(action) for action in path) for path in action_paths]
        for path in normalized_paths:
            if len(path) <= 0 or len(path) > self.max_horizon:
                raise ValueError("A cached action path has an invalid length.")
            if any(action < 0 or action >= self.action_space_size for action in path):
                raise ValueError("A cached action path contains an invalid action.")

        maximum_depth = max(len(path) for path in normalized_paths)
        for depth in range(1, maximum_depth + 1):
            missing = []
            seen = set()
            for root_index, path in zip(root_indices, normalized_paths):
                prefix = path[:depth]
                key = (int(root_index), prefix)
                if (
                    depth <= len(path)
                    and prefix not in prefix_cache.states[int(root_index)]
                    and key not in seen
                ):
                    missing.append(key)
                    seen.add(key)
            if not missing:
                continue

            parent_states = [
                prefix_cache.states[root_index][path[:-1]]
                for root_index, path in missing
            ]
            batched_layer_caches = []
            for layer_index in range(len(self.transformer.layers)):
                batched_layer_caches.append(
                    (
                        torch.cat(
                            [state[0][layer_index][0] for state in parent_states],
                            dim=0,
                        ),
                        torch.cat(
                            [state[0][layer_index][1] for state in parent_states],
                            dim=0,
                        ),
                    )
                )
            batched_padding_mask = torch.cat(
                [state[1] for state in parent_states], dim=0
            )
            actions = torch.as_tensor(
                [path[-1] for _, path in missing],
                dtype=torch.long,
                device=device,
            )
            logits, next_layer_caches, next_padding_mask = self._advance_cached_prefixes(
                batched_layer_caches,
                batched_padding_mask,
                actions,
                depth,
            )
            for row, (root_index, path) in enumerate(missing):
                prefix_cache.states[root_index][path] = (
                    [
                        (
                            keys[row:row + 1],
                            values[row:row + 1],
                        )
                        for keys, values in next_layer_caches
                    ],
                    next_padding_mask[row:row + 1],
                )
                prefix_cache.logits[root_index][path] = logits[row]

        return torch.stack(
            [
                prefix_cache.logits[int(root_index)][path]
                for root_index, path in zip(root_indices, normalized_paths)
            ]
        )

    def all_action_prefix_probabilities(
        self,
        root_contexts: Sequence[RecentHistoryTransformerContext],
        max_horizon: int | None = None,
    ) -> list[dict[tuple[int, ...], float]]:
        """Score every action prefix once and return CPU lookup tables per root."""
        if not root_contexts:
            return []
        horizon = self.max_horizon if max_horizon is None else int(max_horizon)
        if horizon <= 0 or horizon > self.max_horizon:
            raise ValueError("max_horizon is outside the configured range.")
        if self.training:
            raise RuntimeError("Cached prefix inference requires evaluation mode.")

        embeddings, previous_actions, history_valid = self._stack_contexts(
            root_contexts
        )
        layer_caches, padding_mask = self._cached_history_state(
            embeddings,
            previous_actions,
            history_valid,
        )
        root_rows = torch.arange(
            len(root_contexts), dtype=torch.long, device=embeddings.device
        )
        prefix_roots = list(range(len(root_contexts)))
        prefixes = [tuple() for _ in root_contexts]
        probability_parts = []
        table_keys = []
        for depth in range(1, horizon + 1):
            parent_count = len(prefixes)
            parent_rows = torch.arange(
                parent_count, dtype=torch.long, device=embeddings.device
            ).repeat_interleave(self.action_space_size)
            actions = torch.arange(
                self.action_space_size, dtype=torch.long, device=embeddings.device
            ).repeat(parent_count)
            root_rows = root_rows[parent_rows]
            layer_caches = [
                (keys[parent_rows], values[parent_rows])
                for keys, values in layer_caches
            ]
            padding_mask = padding_mask[parent_rows]
            logits, layer_caches, padding_mask = self._advance_cached_prefixes(
                layer_caches,
                padding_mask,
                actions,
                depth,
            )
            prefixes = [
                prefix + (action,)
                for prefix in prefixes
                for action in range(self.action_space_size)
            ]
            prefix_roots = [
                root_index
                for root_index in prefix_roots
                for _ in range(self.action_space_size)
            ]
            probability_parts.append(torch.sigmoid(logits))
            table_keys.extend(zip(prefix_roots, prefixes))

        probabilities = torch.cat(probability_parts).detach().float().cpu().tolist()
        tables: list[dict[tuple[int, ...], float]] = [
            {} for _ in root_contexts
        ]
        for (root_index, path), probability in zip(table_keys, probabilities):
            tables[root_index][path] = float(probability)
        return tables

    def score_actual_suffixes(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        episodic_rewards: torch.Tensor,
        extrinsic_rewards: torch.Tensor,
        root_indices: torch.Tensor,
        horizon: int,
    ) -> torch.Tensor:
        del episodic_rewards, extrinsic_rewards
        horizon = int(horizon)
        if observations.ndim != 5 or actions.ndim != 2:
            raise ValueError("Expected observations [B,T+1,C,H,W] and actions [B,T].")
        if root_indices.ndim != 2 or root_indices.shape[0] != actions.shape[0]:
            raise ValueError("root_indices must be shaped [batch, roots].")
        if observations.shape[1] != actions.shape[1] + 1:
            raise ValueError("Expected exactly one more observation than action.")
        if horizon <= 0 or horizon > self.max_horizon:
            raise ValueError("horizon is outside the configured range.")
        if bool((root_indices < 0).any()) or bool(
            (root_indices + horizon > actions.shape[1]).any()
        ):
            raise ValueError("A sampled suffix extends past the action sequence.")

        batch_size, root_count = root_indices.shape
        history_offsets = torch.arange(
            self.history_window,
            dtype=torch.long,
            device=actions.device,
        )
        history_lengths = (root_indices + 1).clamp(max=self.history_window)
        history_starts = root_indices - history_lengths + 1
        history_indices = history_starts[:, :, None] + history_offsets
        history_valid = history_offsets < history_lengths[:, :, None]
        safe_history_indices = torch.where(
            history_valid,
            history_indices,
            torch.zeros_like(history_indices),
        )
        batch_indices = torch.arange(
            batch_size, dtype=torch.long, device=actions.device
        )[:, None, None].expand(batch_size, root_count, self.history_window)
        observation_count = int(observations.shape[1])
        episode_embeddings = self._encode_observations(
            observations.reshape(
                batch_size * observation_count,
                *observations.shape[2:],
            )
        ).reshape(batch_size, observation_count, self.embedding_dim)
        history_embeddings = episode_embeddings[
            batch_indices, safe_history_indices
        ].reshape(
            batch_size * root_count,
            self.history_window,
            self.embedding_dim,
        )
        previous_action_indices = (safe_history_indices - 1).clamp(min=0)
        previous_actions = actions[batch_indices, previous_action_indices]
        previous_actions = torch.where(
            history_valid & (safe_history_indices > 0),
            previous_actions,
            torch.full_like(previous_actions, self.action_space_size),
        )
        suffix_offsets = torch.arange(
            horizon, dtype=torch.long, device=actions.device
        )
        suffix_indices = root_indices[:, :, None] + suffix_offsets
        suffix_batch_indices = torch.arange(
            batch_size, dtype=torch.long, device=actions.device
        )[:, None, None].expand(batch_size, root_count, horizon)
        suffix_actions = actions[suffix_batch_indices, suffix_indices]
        return self._score_histories(
            history_embeddings,
            previous_actions.reshape(
                batch_size * root_count, self.history_window
            ),
            history_valid.reshape(
                batch_size * root_count, self.history_window
            ),
            suffix_actions.reshape(batch_size * root_count, horizon),
        ).reshape(batch_size, root_count, horizon)


class RecurrentSequenceNovelty(nn.Module):
    """Predict first-visit probability for every edge in an action suffix.

    The observation encoder and history LSTM are independent of MuZero. Imagined
    MuZero latents are deliberately not accepted by this module.
    """

    def __init__(
        self,
        observation_channels: int,
        action_space_size: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.observation_channels = int(observation_channels)
        self.action_space_size = int(action_space_size)
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        if self.observation_channels <= 0:
            raise ValueError("observation_channels must be positive.")
        if self.action_space_size <= 0:
            raise ValueError("action_space_size must be positive.")

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
            nn.ReLU(inplace=True),
        )
        self.action_embedding = nn.Embedding(self.action_space_size + 1, action_dim)
        self.history_cell = nn.LSTMCell(
            self.embedding_dim + action_dim + 2,
            self.hidden_dim,
        )
        self.prefix_initial = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
        )
        self.prefix_cell = nn.GRUCell(action_dim, self.hidden_dim)
        self.novelty_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, 1),
        )
        self.continuation_factor_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, 1),
        )

    def _encode_observations(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim != 4:
            raise ValueError(
                "Expected observations shaped [batch, channels, height, width], "
                f"got {tuple(observations.shape)}."
            )
        if int(observations.shape[1]) != self.observation_channels:
            raise ValueError(
                f"Expected {self.observation_channels} observation channels, "
                f"got {int(observations.shape[1])}."
            )
        return self.observation_encoder(observations.float())

    def _history_step(
        self,
        observation_embedding: torch.Tensor,
        previous_action: torch.Tensor,
        previous_episodic_reward: torch.Tensor,
        previous_extrinsic_reward: torch.Tensor,
        context: RecurrentSequenceContext,
    ) -> RecurrentSequenceContext:
        recurrent_input = torch.cat(
            [
                observation_embedding,
                self.action_embedding(previous_action),
                previous_episodic_reward.reshape(-1, 1),
                previous_extrinsic_reward.reshape(-1, 1),
            ],
            dim=-1,
        )
        hidden, cell = self.history_cell(
            recurrent_input,
            (context.hidden, context.cell),
        )
        return RecurrentSequenceContext(hidden=hidden, cell=cell)

    def start_context(self, observation: torch.Tensor) -> RecurrentSequenceContext:
        batch_size = int(observation.shape[0])
        hidden = torch.zeros(
            batch_size,
            self.hidden_dim,
            device=observation.device,
            dtype=torch.float32,
        )
        context = RecurrentSequenceContext(hidden=hidden, cell=torch.zeros_like(hidden))
        start_action = torch.full(
            (batch_size,),
            self.action_space_size,
            device=observation.device,
            dtype=torch.long,
        )
        zero_reward = torch.zeros(batch_size, device=observation.device)
        return self._history_step(
            self._encode_observations(observation),
            start_action,
            zero_reward,
            zero_reward,
            context,
        )

    def advance_context(
        self,
        context: RecurrentSequenceContext,
        action: torch.Tensor,
        observation: torch.Tensor,
        episodic_reward: torch.Tensor,
        extrinsic_reward: torch.Tensor,
    ) -> RecurrentSequenceContext:
        return self._history_step(
            self._encode_observations(observation),
            action.to(device=observation.device, dtype=torch.long).reshape(-1),
            episodic_reward.to(device=observation.device, dtype=torch.float32).reshape(-1),
            extrinsic_reward.to(device=observation.device, dtype=torch.float32).reshape(-1),
            RecurrentSequenceContext(
                hidden=context.hidden.to(observation.device),
                cell=context.cell.to(observation.device),
            ),
        )

    def history_states(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        episodic_rewards: torch.Tensor,
        extrinsic_rewards: torch.Tensor,
    ) -> torch.Tensor:
        """Return the recurrent history state at every real trajectory root."""
        if observations.ndim != 5 or actions.ndim != 2:
            raise ValueError("Expected observations [B,T+1,C,H,W] and actions [B,T].")
        if observations.shape[1] != actions.shape[1] + 1:
            raise ValueError("Expected exactly one more observation than action.")
        if episodic_rewards.shape != actions.shape or extrinsic_rewards.shape != actions.shape:
            raise ValueError("Reward tensors must have the same shape as actions.")

        batch_size, observation_count = observations.shape[:2]
        embeddings = self._encode_observations(
            observations.reshape(
                batch_size * observation_count,
                *observations.shape[2:],
            )
        ).reshape(batch_size, observation_count, self.embedding_dim)
        hidden = torch.zeros(
            batch_size,
            self.hidden_dim,
            device=observations.device,
            dtype=torch.float32,
        )
        context = RecurrentSequenceContext(hidden=hidden, cell=torch.zeros_like(hidden))
        start_action = torch.full(
            (batch_size,),
            self.action_space_size,
            device=observations.device,
            dtype=torch.long,
        )
        zero_reward = torch.zeros(batch_size, device=observations.device)
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
            context = self._history_step(
                embeddings[:, index],
                previous_action,
                previous_episodic_reward,
                previous_extrinsic_reward,
                context,
            )
            states.append(context.hidden)
        return torch.stack(states, dim=1)

    def _prefix_states_from_hidden(
        self,
        root_hidden: torch.Tensor,
        action_suffixes: torch.Tensor,
    ) -> torch.Tensor:
        """Return the recurrent state after every edge of each suffix."""
        if root_hidden.ndim != 2 or action_suffixes.ndim != 2:
            raise ValueError("Expected roots [N,H] and action suffixes [N,K].")
        if root_hidden.shape[0] != action_suffixes.shape[0]:
            raise ValueError("Every action suffix must have one root history state.")
        if action_suffixes.shape[1] <= 0:
            raise ValueError("Action suffixes cannot be empty.")
        prefix_state = self.prefix_initial(root_hidden)
        states = []
        for depth in range(int(action_suffixes.shape[1])):
            prefix_state = self.prefix_cell(
                self.action_embedding(action_suffixes[:, depth]),
                prefix_state,
            )
            states.append(prefix_state)
        return torch.stack(states, dim=1)

    def score_from_hidden(
        self,
        root_hidden: torch.Tensor,
        action_suffixes: torch.Tensor,
    ) -> torch.Tensor:
        """Return one immediate-novelty logit per edge of each suffix."""
        prefix_states = self._prefix_states_from_hidden(root_hidden, action_suffixes)
        return self.novelty_head(prefix_states).squeeze(-1)

    def score_continuation_from_hidden(
        self,
        root_hidden: torch.Tensor,
        action_suffixes: torch.Tensor,
    ) -> torch.Tensor:
        """Return the episodic continuation-factor logit at every suffix endpoint."""
        prefix_states = self._prefix_states_from_hidden(root_hidden, action_suffixes)
        return self.continuation_factor_head(prefix_states).squeeze(-1)

    def score_path_endpoints(
        self,
        root_contexts: Sequence[RecurrentSequenceContext],
        root_indices: Sequence[int],
        action_paths: Sequence[Sequence[int]],
    ) -> torch.Tensor:
        """Score variable-length MCTS paths and return their endpoint logits."""
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
            hidden = torch.cat(
                [root_contexts[int(root_indices[row])].hidden for row in rows],
                dim=0,
            )
            actions = torch.as_tensor(
                [action_paths[row] for row in rows],
                dtype=torch.long,
                device=hidden.device,
            )
            result[torch.as_tensor(rows, dtype=torch.long, device=result.device)] = (
                self.score_from_hidden(hidden, actions)[:, int(length) - 1]
            )
        return result

    def score_path_continuation_endpoints(
        self,
        root_contexts: Sequence[RecurrentSequenceContext],
        root_indices: Sequence[int],
        action_paths: Sequence[Sequence[int]],
    ) -> torch.Tensor:
        """Score the continuation factor at variable-length MCTS leaves."""
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
            hidden = torch.cat(
                [root_contexts[int(root_indices[row])].hidden for row in rows],
                dim=0,
            )
            actions = torch.as_tensor(
                [action_paths[row] for row in rows],
                dtype=torch.long,
                device=hidden.device,
            )
            result[torch.as_tensor(rows, dtype=torch.long, device=result.device)] = (
                self.score_continuation_from_hidden(hidden, actions)[:, int(length) - 1]
            )
        return result

    def score_actual_suffixes(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        episodic_rewards: torch.Tensor,
        extrinsic_rewards: torch.Tensor,
        root_indices: torch.Tensor,
        horizon: int,
    ) -> torch.Tensor:
        """Score actual replay suffixes selected independently per episode."""
        horizon = int(horizon)
        if root_indices.ndim != 2:
            raise ValueError("root_indices must be shaped [batch, roots].")
        states = self.history_states(
            observations,
            actions,
            episodic_rewards,
            extrinsic_rewards,
        )
        batch_size, root_count = root_indices.shape
        batch_indices = torch.arange(batch_size, device=actions.device)[:, None]
        root_hidden = states[batch_indices, root_indices]
        suffix_actions = torch.stack(
            [actions[batch_indices, root_indices + depth] for depth in range(horizon)],
            dim=-1,
        )
        return self.score_from_hidden(
            root_hidden.reshape(batch_size * root_count, self.hidden_dim),
            suffix_actions.reshape(batch_size * root_count, horizon),
        ).reshape(batch_size, root_count, horizon)

    def score_actual_continuation_suffixes(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        episodic_rewards: torch.Tensor,
        extrinsic_rewards: torch.Tensor,
        root_indices: torch.Tensor,
        horizon: int,
    ) -> torch.Tensor:
        """Return continuation-factor logits at actual replay suffix endpoints."""
        horizon = int(horizon)
        if root_indices.ndim != 2:
            raise ValueError("root_indices must be shaped [batch, roots].")
        states = self.history_states(
            observations,
            actions,
            episodic_rewards,
            extrinsic_rewards,
        )
        batch_size, root_count = root_indices.shape
        batch_indices = torch.arange(batch_size, device=actions.device)[:, None]
        root_hidden = states[batch_indices, root_indices]
        suffix_actions = torch.stack(
            [actions[batch_indices, root_indices + depth] for depth in range(horizon)],
            dim=-1,
        )
        return self.score_continuation_from_hidden(
            root_hidden.reshape(batch_size * root_count, self.hidden_dim),
            suffix_actions.reshape(batch_size * root_count, horizon),
        ).reshape(batch_size, root_count, horizon)

    def score_actual_suffixes_with_continuation(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        episodic_rewards: torch.Tensor,
        extrinsic_rewards: torch.Tensor,
        root_indices: torch.Tensor,
        horizon: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score immediate novelty and continuation with one history encoding."""
        horizon = int(horizon)
        if root_indices.ndim != 2:
            raise ValueError("root_indices must be shaped [batch, roots].")
        states = self.history_states(
            observations,
            actions,
            episodic_rewards,
            extrinsic_rewards,
        )
        batch_size, root_count = root_indices.shape
        batch_indices = torch.arange(batch_size, device=actions.device)[:, None]
        root_hidden = states[batch_indices, root_indices]
        suffix_actions = torch.stack(
            [actions[batch_indices, root_indices + depth] for depth in range(horizon)],
            dim=-1,
        )
        prefix_states = self._prefix_states_from_hidden(
            root_hidden.reshape(batch_size * root_count, self.hidden_dim),
            suffix_actions.reshape(batch_size * root_count, horizon),
        )
        output_shape = (batch_size, root_count, horizon)
        return (
            self.novelty_head(prefix_states).squeeze(-1).reshape(output_shape),
            self.continuation_factor_head(prefix_states).squeeze(-1).reshape(output_shape),
        )
