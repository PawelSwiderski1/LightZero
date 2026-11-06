import time
from typing import List, Optional, Sequence, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from ding.utils import MODEL_REGISTRY

from .common import MZNetworkOutput
from .muzero_model import MuZeroModel
from .muzero_model_mlp import MuZeroModelMLP


class _RandomValuePrior(nn.Module):
    """
    Fixed random prior function for ensemble value predictions.

    The network maps a normalized latent state to a bounded scalar. The
    ensemble wrapper interprets that scalar in original value units, keeping
    the prior's effect independent of categorical value-prediction sharpness.
    """

    def __init__(
            self,
            input_dim: int,
            hidden_size: int = 64,
            latent_normalization: str = 'layer_norm',
    ) -> None:
        super().__init__()
        self.latent_normalization = str(latent_normalization).lower()
        if self.latent_normalization not in {'none', 'layer_norm', 'l2'}:
            raise ValueError(
                "value_prior_latent_normalization must be one of 'none', 'layer_norm', or 'l2', "
                f"got {latent_normalization}"
            )
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Tanh(),
        )
        for param in self.parameters():
            param.requires_grad_(False)

    def forward(self, latent_state: torch.Tensor) -> torch.Tensor:
        if latent_state.dim() > 2:
            prior_input = latent_state.mean(dim=tuple(range(2, latent_state.dim())))
        else:
            prior_input = latent_state.view(latent_state.shape[0], -1)
        prior_input = prior_input.float()
        if self.latent_normalization == 'layer_norm':
            prior_input = F.layer_norm(prior_input, (prior_input.shape[-1],))
        elif self.latent_normalization == 'l2':
            target_norm = float(prior_input.shape[-1]) ** 0.5
            norm = prior_input.pow(2).sum(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
            prior_input = prior_input * (target_norm / norm)
        scalar_prior = self.net(prior_input)
        return scalar_prior.to(dtype=latent_state.dtype, device=latent_state.device)


class _RandomEncoderPrior(nn.Module):
    """Frozen smooth random feature map from vector observations to latent space."""

    def __init__(
            self,
            observation_dim: int,
            latent_dim: int,
            hidden_size: int = 64,
            input_mean: Optional[Sequence[float]] = None,
            input_scale: Optional[Sequence[float]] = None,
            output_norm: float = 1.0,
    ) -> None:
        super().__init__()
        if observation_dim <= 0 or latent_dim <= 0:
            raise ValueError(
                f"Random encoder prior dimensions must be positive, got {observation_dim} -> {latent_dim}."
            )
        if hidden_size <= 0:
            raise ValueError(f"encoder_random_prior_hidden_dim must be positive, got {hidden_size}")
        if output_norm < 0:
            raise ValueError(f"encoder_random_prior_output_norm must be non-negative, got {output_norm}")

        mean = torch.zeros(observation_dim, dtype=torch.float32)
        scale = torch.ones(observation_dim, dtype=torch.float32)
        if input_mean is not None:
            mean = torch.as_tensor(input_mean, dtype=torch.float32).reshape(-1)
        if input_scale is not None:
            scale = torch.as_tensor(input_scale, dtype=torch.float32).reshape(-1)
        if mean.numel() != observation_dim or scale.numel() != observation_dim:
            raise ValueError(
                "Random encoder prior input normalization must have one value per observation feature: "
                f"expected {observation_dim}, got mean={mean.numel()} and scale={scale.numel()}."
            )
        if torch.any(scale <= 0).item():
            raise ValueError("encoder_random_prior_input_scale entries must all be positive")

        self.output_norm = float(output_norm)
        self.register_buffer('input_mean', mean)
        self.register_buffer('input_scale', scale)
        self.net = nn.Sequential(
            nn.Linear(observation_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, latent_dim),
            nn.Tanh(),
        )
        for param in self.parameters():
            param.requires_grad_(False)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        flat = observation.reshape(observation.shape[0], -1).to(
            dtype=self.input_mean.dtype, device=self.input_mean.device
        )
        normalized = (flat - self.input_mean) / self.input_scale
        prior = self.net(normalized)
        if self.output_norm > 0:
            norm = prior.pow(2).sum(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
            prior = prior * (self.output_norm / norm)
        return prior


class _LocalGridEncoderPrior(nn.Module):
    """Frozen locally interpolated random field over normalized vector observations."""

    _HASH_MULTIPLIERS = (
        73856093,
        19349663,
        83492791,
        2654435761,
        805459861,
        3674653429,
        2097192037,
        1434869437,
    )

    def __init__(
            self,
            observation_dim: int,
            latent_dim: int,
            cell_size: float = 0.25,
            hash_size: int = 16384,
            input_mean: Optional[Sequence[float]] = None,
            input_scale: Optional[Sequence[float]] = None,
            output_norm: float = 1.0,
    ) -> None:
        super().__init__()
        if observation_dim <= 0 or latent_dim <= 0:
            raise ValueError(
                f"Local encoder prior dimensions must be positive, got {observation_dim} -> {latent_dim}."
            )
        if observation_dim > len(self._HASH_MULTIPLIERS):
            raise NotImplementedError(
                "Local encoder priors support at most "
                f"{len(self._HASH_MULTIPLIERS)} observation dimensions, got {observation_dim}."
            )
        if cell_size <= 0:
            raise ValueError(f"encoder_random_prior_cell_size must be positive, got {cell_size}")
        if hash_size <= 0:
            raise ValueError(f"encoder_random_prior_hash_size must be positive, got {hash_size}")
        if output_norm < 0:
            raise ValueError(f"encoder_random_prior_output_norm must be non-negative, got {output_norm}")

        mean = torch.zeros(observation_dim, dtype=torch.float32)
        scale = torch.ones(observation_dim, dtype=torch.float32)
        if input_mean is not None:
            mean = torch.as_tensor(input_mean, dtype=torch.float32).reshape(-1)
        if input_scale is not None:
            scale = torch.as_tensor(input_scale, dtype=torch.float32).reshape(-1)
        if mean.numel() != observation_dim or scale.numel() != observation_dim:
            raise ValueError(
                "Local encoder prior input normalization must have one value per observation feature: "
                f"expected {observation_dim}, got mean={mean.numel()} and scale={scale.numel()}."
            )
        if torch.any(scale <= 0).item():
            raise ValueError("encoder_random_prior_input_scale entries must all be positive")

        self.observation_dim = int(observation_dim)
        self.cell_size = float(cell_size)
        self.hash_size = int(hash_size)
        self.output_norm = float(output_norm)
        self.register_buffer('input_mean', mean)
        self.register_buffer('input_scale', scale)
        self.register_buffer(
            'hash_multipliers',
            torch.as_tensor(self._HASH_MULTIPLIERS[:observation_dim], dtype=torch.long),
        )
        corner_count = 1 << observation_dim
        corner_ids = torch.arange(corner_count, dtype=torch.long).unsqueeze(1)
        dimensions = torch.arange(observation_dim, dtype=torch.long).unsqueeze(0)
        self.register_buffer('corner_offsets', ((corner_ids >> dimensions) & 1))

        self.embedding = nn.Embedding(self.hash_size, latent_dim)
        nn.init.normal_(self.embedding.weight)
        with torch.no_grad():
            self.embedding.weight.copy_(F.normalize(self.embedding.weight, p=2, dim=1))
        self.embedding.weight.requires_grad_(False)

    def _hash(self, coordinates: torch.Tensor) -> torch.Tensor:
        hashed = torch.zeros(coordinates.shape[:-1], dtype=torch.long, device=coordinates.device)
        multipliers = self.hash_multipliers.to(device=coordinates.device)
        for dimension in range(self.observation_dim):
            hashed = torch.bitwise_xor(hashed, coordinates[..., dimension] * multipliers[dimension])
        return torch.remainder(hashed, self.hash_size)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        flat = observation.reshape(observation.shape[0], -1).to(
            dtype=self.input_mean.dtype, device=self.input_mean.device
        )
        normalized = (flat - self.input_mean) / self.input_scale
        grid_position = normalized / self.cell_size
        lower = torch.floor(grid_position)
        fraction = grid_position - lower
        lower = lower.to(dtype=torch.long)

        offsets = self.corner_offsets.to(device=flat.device)
        corners = lower.unsqueeze(1) + offsets.unsqueeze(0)
        interpolation = torch.where(
            offsets.unsqueeze(0).bool(),
            fraction.unsqueeze(1),
            1.0 - fraction.unsqueeze(1),
        ).prod(dim=-1)
        prior = (self.embedding(self._hash(corners)) * interpolation.unsqueeze(-1)).sum(dim=1)
        if self.output_norm > 0:
            prior = F.normalize(prior, p=2, dim=1) * self.output_norm
        return prior.to(dtype=observation.dtype, device=observation.device)


class _LatentProjectionMLP(nn.Module):
    """
    Small projection head over a member latent.

    These heads are used only for auxiliary latent-space diagnostics. They do
    not feed MuZero policy/value/reward/dynamics outputs.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_size: int = 64) -> None:
        super().__init__()
        if hidden_size <= 0:
            self.net = nn.Linear(input_dim, output_dim)
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size, output_dim),
            )

    @staticmethod
    def _flatten_latent(latent_state: torch.Tensor) -> torch.Tensor:
        if latent_state.dim() > 2:
            latent_state = latent_state.mean(dim=tuple(range(2, latent_state.dim())))
        return latent_state.reshape(latent_state.shape[0], -1).float()

    def forward(self, latent_state: torch.Tensor) -> torch.Tensor:
        return self.net(self._flatten_latent(latent_state))


class _ControllableStateEncoder(nn.Module):
    """Observation encoder trained only through an inverse-dynamics objective."""

    def __init__(
            self,
            observation_shape,
            embedding_dim: int,
            hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        hidden_dim = int(hidden_dim)
        if self.embedding_dim <= 0 or hidden_dim <= 0:
            raise ValueError('Controllable-state embedding dimensions must be positive.')

        if isinstance(observation_shape, int):
            self.image_encoder = None
            self.vector_encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(int(observation_shape), hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, self.embedding_dim),
            )
        else:
            observation_shape = tuple(int(size) for size in observation_shape)
            if len(observation_shape) != 3:
                raise ValueError(
                    'Image controllable-state encoders require [channels, height, width], '
                    f'got {observation_shape}.'
                )
            channels = max(16, hidden_dim // 4)
            self.image_encoder = nn.Sequential(
                nn.Conv2d(observation_shape[0], channels, kernel_size=5, stride=2, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, 2 * channels, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(2 * channels, 2 * channels, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((3, 3)),
            )
            self.vector_encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(2 * channels * 3 * 3, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, self.embedding_dim),
            )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        features = observation.float()
        if self.image_encoder is not None:
            features = self.image_encoder(features)
        embedding = self.vector_encoder(features)
        return F.normalize(embedding, p=2, dim=-1, eps=1e-8)


class _ControllableStateBridge(nn.Module):
    """Map one member's imagined MuZero latent into the shared episodic space."""

    def __init__(
            self,
            latent_dim: int,
            embedding_dim: int,
            hidden_dim: int = 128,
            spatial: bool = False,
    ) -> None:
        super().__init__()
        latent_dim = int(latent_dim)
        embedding_dim = int(embedding_dim)
        hidden_dim = int(hidden_dim)
        self.spatial = bool(spatial)
        if self.spatial:
            channels = max(16, hidden_dim // 2)
            self.spatial_encoder = nn.Sequential(
                nn.Conv2d(latent_dim, channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((3, 3)),
            )
            input_dim = channels * 3 * 3
        else:
            self.spatial_encoder = None
            input_dim = latent_dim
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, latent_state: torch.Tensor) -> torch.Tensor:
        features = latent_state.float()
        if self.spatial_encoder is not None:
            if features.dim() != 4:
                raise ValueError(
                    'Spatial episodic bridges require [batch, channels, height, width], '
                    f'got {tuple(features.shape)}.'
                )
            features = self.spatial_encoder(features)
        embedding = self.head(features)
        return F.normalize(embedding, p=2, dim=-1, eps=1e-8)


class _RandomLatentPrior(_LatentProjectionMLP):
    """
    Frozen per-member random feature over the member's own latent.

    It can be added to the trainable aligned embedding to keep member-specific
    diversity available for future novelty diagnostics over imagined states.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_size: int = 64) -> None:
        super().__init__(input_dim, output_dim, hidden_size)
        for param in self.parameters():
            param.requires_grad_(False)


class _EpisodicHistoryRewardModel(nn.Module):
    """One-step novelty predictor with a real-transition episodic state."""

    def __init__(
            self,
            latent_dim: int,
            action_space_size: int,
            history_dim: int,
            hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.action_space_size = int(action_space_size)
        self.history_dim = int(history_dim)
        self.history_gru = nn.GRU(
            2 * self.latent_dim + self.action_space_size + 1,
            self.history_dim,
            batch_first=True,
        )
        self.reward_head = nn.Sequential(
            nn.Linear(
                2 * self.latent_dim + self.action_space_size + self.history_dim,
                int(hidden_dim),
            ),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), 1),
        )
        nn.init.zeros_(self.reward_head[-1].weight)
        nn.init.zeros_(self.reward_head[-1].bias)

    def _flatten_latent(self, latent_state: torch.Tensor) -> torch.Tensor:
        if latent_state.dim() > 2:
            latent_state = latent_state.mean(dim=tuple(range(2, latent_state.dim())))
        latent_state = latent_state.reshape(latent_state.shape[0], -1).float()
        if latent_state.shape[1] != self.latent_dim:
            raise ValueError(
                f"Expected episodic-history latent dimension {self.latent_dim}, "
                f"got {latent_state.shape[1]}"
            )
        return latent_state

    def initial_history(
            self,
            batch_size: int,
            device: torch.device,
            dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        return torch.zeros(int(batch_size), self.history_dim, device=device, dtype=dtype)

    def _action_one_hot(
            self,
            action: torch.Tensor,
            reference: torch.Tensor,
    ) -> torch.Tensor:
        action = action.long().reshape(-1)
        if action.shape[0] != reference.shape[0]:
            raise ValueError(
                f"Expected {reference.shape[0]} episodic-history actions, got {action.shape[0]}"
            )
        return F.one_hot(action, num_classes=self.action_space_size).to(
            device=reference.device,
            dtype=reference.dtype,
        )

    @staticmethod
    def _reward_feedback_input(reward: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        reward = reward.to(device=reference.device, dtype=reference.dtype).reshape(-1, 1)
        if reward.shape[0] != reference.shape[0]:
            raise ValueError(
                f"Expected {reference.shape[0]} episodic-history rewards, got {reward.shape[0]}"
            )
        return torch.log1p(reward.clamp_min(0.0))

    def predict_reward(
            self,
            current_latent: torch.Tensor,
            action: torch.Tensor,
            next_latent: torch.Tensor,
            history: torch.Tensor,
    ) -> torch.Tensor:
        current_latent = self._flatten_latent(current_latent)
        next_latent = self._flatten_latent(next_latent)
        action_one_hot = self._action_one_hot(action, current_latent)
        history = history.to(device=current_latent.device, dtype=current_latent.dtype)
        return self.reward_head(
            torch.cat([current_latent, action_one_hot, next_latent, history], dim=-1)
        )

    def advance_history(
            self,
            current_latent: torch.Tensor,
            action: torch.Tensor,
            next_latent: torch.Tensor,
            realized_reward: torch.Tensor,
            history: torch.Tensor,
    ) -> torch.Tensor:
        current_latent = self._flatten_latent(current_latent)
        next_latent = self._flatten_latent(next_latent)
        action_one_hot = self._action_one_hot(action, current_latent)
        reward_input = self._reward_feedback_input(realized_reward, current_latent)
        history = history.to(device=current_latent.device, dtype=current_latent.dtype)
        _, final_history = self.history_gru(
            torch.cat(
                [current_latent, action_one_hot, next_latent, reward_input], dim=-1
            ).unsqueeze(1),
            history.unsqueeze(0),
        )
        return final_history.squeeze(0)

    def forward(
            self,
            current_latent: torch.Tensor,
            action: torch.Tensor,
            next_latent: torch.Tensor,
            realized_reward: torch.Tensor,
            history: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        reward_prediction = self.predict_reward(
            current_latent, action, next_latent, history
        )
        next_history = self.advance_history(
            current_latent,
            action,
            next_latent,
            realized_reward,
            history,
        )
        return next_history, reward_prediction

    def forward_sequence(
            self,
            current_latents: torch.Tensor,
            actions: torch.Tensor,
            predicted_next_latents: torch.Tensor,
            realized_next_latents: torch.Tensor,
            realized_rewards: torch.Tensor,
            initial_history: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, step_count = current_latents.shape[:2]
        current_latents = self._flatten_latent(current_latents.reshape(
            batch_size * step_count, *current_latents.shape[2:]
        )).reshape(batch_size, step_count, self.latent_dim)
        predicted_next_latents = self._flatten_latent(predicted_next_latents.reshape(
            batch_size * step_count, *predicted_next_latents.shape[2:]
        )).reshape(batch_size, step_count, self.latent_dim)
        realized_next_latents = self._flatten_latent(realized_next_latents.reshape(
            batch_size * step_count, *realized_next_latents.shape[2:]
        )).reshape(batch_size, step_count, self.latent_dim)
        actions = actions.long().reshape(batch_size, step_count)
        action_one_hot = F.one_hot(actions, num_classes=self.action_space_size).to(
            device=current_latents.device,
            dtype=current_latents.dtype,
        )
        reward_inputs = self._reward_feedback_input(
            realized_rewards.reshape(-1), current_latents.reshape(-1, self.latent_dim)
        ).reshape(batch_size, step_count, 1)
        history_inputs = torch.cat(
            [current_latents, action_one_hot, realized_next_latents, reward_inputs], dim=-1
        )
        histories, _ = self.history_gru(
            history_inputs,
            initial_history.to(
                device=current_latents.device,
                dtype=current_latents.dtype,
            ).unsqueeze(0),
        )
        pre_transition_histories = torch.cat(
            [initial_history.unsqueeze(1), histories[:, :-1]], dim=1
        )
        rewards = self.reward_head(
            torch.cat(
                [current_latents, action_one_hot, predicted_next_latents, pre_transition_histories],
                dim=-1,
            )
        )
        return histories, rewards


class _RandomDynamicsPrior(nn.Module):
    """
    Fixed per-member prior over state-action latents.

    It predicts a latent delta, not reward. The wrapper adds this delta to the
    trainable dynamics latent before running the shared/member prediction head.
    ``one_hot`` preserves the original single prior network over
    ``[latent, one_hot(action)]``. ``per_action`` uses a separate frozen prior
    network for each discrete action, which keeps unvisited actions from sharing
    most of the same random prior geometry as visited actions at the same state.
    """

    def __init__(
            self,
            latent_dim: int,
            action_space_size: int,
            hidden_size: int = 64,
            action_mode: str = 'one_hot',
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.action_space_size = int(action_space_size)
        self.action_mode = str(action_mode)
        if self.latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        if self.action_space_size <= 0:
            raise ValueError(f"action_space_size must be positive, got {action_space_size}")
        if self.action_mode not in {'one_hot', 'per_action'}:
            raise ValueError(
                "dynamics random prior action_mode must be 'one_hot' or 'per_action', "
                f"got {action_mode}"
            )
        if self.action_mode == 'one_hot':
            self.net = self._build_net(self.latent_dim + self.action_space_size, hidden_size)
            self.action_nets = None
        else:
            self.net = None
            self.action_nets = nn.ModuleList(
                [self._build_net(self.latent_dim, hidden_size) for _ in range(self.action_space_size)]
            )
        for param in self.parameters():
            param.requires_grad_(False)

    def _build_net(self, input_dim: int, hidden_size: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, self.latent_dim),
            nn.Tanh(),
        )

    def forward(self, latent_state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if latent_state.dim() != 2:
            raise NotImplementedError(
                "Random dynamics prior currently supports vector latents only. "
                f"Got latent shape {tuple(latent_state.shape)}."
            )
        latent_input = latent_state.detach().reshape(latent_state.shape[0], -1).float()
        action = action.detach().long().reshape(-1)
        if action.numel() != latent_input.shape[0]:
            raise ValueError(
                f"Expected {latent_input.shape[0]} actions for random dynamics prior, got {action.numel()}"
            )
        invalid_action = (action < 0) | (action >= self.action_space_size)
        if invalid_action.any().item():
            raise ValueError(
                f"Action ids must be in [0, {self.action_space_size}), got {action[invalid_action].tolist()}"
            )
        if self.action_mode == 'one_hot':
            action_one_hot = F.one_hot(action, num_classes=self.action_space_size).float()
            action_one_hot = action_one_hot.to(device=latent_state.device)
            prior_input = torch.cat([latent_input, action_one_hot], dim=-1)
            prior_delta = self.net(prior_input)
        else:
            prior_delta = latent_input.new_empty((latent_input.shape[0], self.latent_dim))
            for action_id, action_net in enumerate(self.action_nets):
                indices = (action == action_id).nonzero(as_tuple=False).view(-1)
                if indices.numel() == 0:
                    continue
                prior_delta.index_copy_(0, indices, action_net(latent_input.index_select(0, indices)))
        return prior_delta.to(dtype=latent_state.dtype, device=latent_state.device)


class _EnsembleMuZeroBase(nn.Module):
    """
    Whole-model ensemble wrapper for MuZero-style models.

    Inference defaults to member 0 for compatibility with vanilla MuZero call
    sites. Ensemble-aware code can pass either one member id for the whole
    batch or one member id per batch row.
    """

    member_cls: Type[nn.Module] = None

    def __init__(
            self,
            ensemble_size: int = 1,
            active_member_id: int = 0,
            random_prior_scale: float = 0.0,
            value_prior_backprop_to_latent: bool = False,
            value_prior_latent_normalization: str = 'layer_norm',
            value_prior_scalar_loss_weight: float = 1.0,
            encoder_random_prior_scale: float = 0.0,
            encoder_random_prior_mode: str = 'smooth_mlp',
            encoder_random_prior_hidden_dim: int = 64,
            encoder_random_prior_cell_size: float = 0.25,
            encoder_random_prior_hash_size: int = 16384,
            encoder_random_prior_input_mean: Optional[Sequence[float]] = None,
            encoder_random_prior_input_scale: Optional[Sequence[float]] = None,
            encoder_random_prior_output_norm: float = 1.0,
            encoder_prior_composed_latent_normalization: str = 'none',
            encoder_prior_composed_latent_norm_scale: float = 1.0,
            dynamics_random_prior_scale: float = 0.0,
            dynamics_random_prior_hidden_dim: int = 64,
            dynamics_random_prior_action_mode: str = 'one_hot',
            dynamics_encoder_latent_normalization: str = 'none',
            dynamics_encoder_latent_norm_scale: float = 1.0,
            dynamics_trainable_latent_normalization: str = 'none',
            dynamics_trainable_latent_norm_scale: float = 1.0,
            dynamics_trainable_latent_norm_eps: float = 1e-3,
            dynamics_prior_latent_norm_cap: float = 0.0,
            shared_representation: bool = False,
            shared_prediction: bool = False,
            shared_projection: bool = False,
            shared_intrinsic_reward_head: bool = False,
            latent_novelty_dim: int = 0,
            latent_novelty_hidden_dim: int = 64,
            latent_novelty_prior_scale: float = 0.0,
            episodic_history_dim: int = 0,
            episodic_history_hidden_dim: int = 64,
            episodic_inverse_embedding_dim: int = 0,
            episodic_inverse_hidden_dim: int = 128,
            *args,
            **kwargs
    ) -> None:
        super().__init__()
        assert ensemble_size >= 1, ensemble_size
        self.ensemble_size = ensemble_size
        self.active_member_id: Union[int, List[int]] = active_member_id
        self.random_prior_scale = float(random_prior_scale)
        self.value_prior_backprop_to_latent = bool(value_prior_backprop_to_latent)
        self.value_prior_latent_normalization = str(value_prior_latent_normalization).lower()
        self.value_prior_scalar_loss_weight = float(value_prior_scalar_loss_weight)
        if self.value_prior_latent_normalization not in {'none', 'layer_norm', 'l2'}:
            raise ValueError(
                "value_prior_latent_normalization must be one of 'none', 'layer_norm', or 'l2', "
                f"got {value_prior_latent_normalization}"
            )
        self.encoder_random_prior_scale = float(encoder_random_prior_scale)
        self.encoder_random_prior_mode = str(encoder_random_prior_mode).lower()
        self.encoder_random_prior_output_norm = float(encoder_random_prior_output_norm)
        self.encoder_prior_composed_latent_normalization = str(
            encoder_prior_composed_latent_normalization
        ).lower()
        self.encoder_prior_composed_latent_norm_scale = float(
            encoder_prior_composed_latent_norm_scale
        )
        self.dynamics_random_prior_scale = float(dynamics_random_prior_scale)
        self.dynamics_random_prior_action_mode = str(dynamics_random_prior_action_mode)
        self.dynamics_encoder_latent_normalization = str(dynamics_encoder_latent_normalization).lower()
        self.dynamics_encoder_latent_norm_scale = float(dynamics_encoder_latent_norm_scale)
        self.dynamics_trainable_latent_normalization = str(dynamics_trainable_latent_normalization).lower()
        self.dynamics_trainable_latent_norm_scale = float(dynamics_trainable_latent_norm_scale)
        self.dynamics_trainable_latent_norm_eps = float(dynamics_trainable_latent_norm_eps)
        self.dynamics_prior_latent_norm_cap = float(dynamics_prior_latent_norm_cap)
        self.shared_representation = bool(shared_representation)
        self.shared_prediction = bool(shared_prediction)
        self.shared_projection = bool(shared_projection)
        self.shared_intrinsic_reward_head = bool(shared_intrinsic_reward_head)
        self.latent_novelty_dim = int(latent_novelty_dim)
        self.latent_novelty_prior_scale = float(latent_novelty_prior_scale)
        self.episodic_history_dim = int(episodic_history_dim)
        self.episodic_inverse_embedding_dim = int(episodic_inverse_embedding_dim)
        if self.random_prior_scale < 0:
            raise ValueError(f"random_prior_scale must be non-negative, got {random_prior_scale}")
        if self.value_prior_scalar_loss_weight < 0:
            raise ValueError(
                "value_prior_scalar_loss_weight must be non-negative, "
                f"got {value_prior_scalar_loss_weight}"
            )
        if self.encoder_random_prior_scale < 0:
            raise ValueError(
                f"encoder_random_prior_scale must be non-negative, got {encoder_random_prior_scale}"
            )
        if self.encoder_random_prior_mode not in {'smooth_mlp', 'local_grid'}:
            raise ValueError(
                "encoder_random_prior_mode must be one of 'smooth_mlp' or 'local_grid', "
                f"got {encoder_random_prior_mode}"
            )
        if encoder_random_prior_cell_size <= 0:
            raise ValueError(
                "encoder_random_prior_cell_size must be positive, "
                f"got {encoder_random_prior_cell_size}"
            )
        if encoder_random_prior_hash_size <= 0:
            raise ValueError(
                "encoder_random_prior_hash_size must be positive, "
                f"got {encoder_random_prior_hash_size}"
            )
        if self.encoder_random_prior_output_norm < 0:
            raise ValueError(
                "encoder_random_prior_output_norm must be non-negative, "
                f"got {encoder_random_prior_output_norm}"
            )
        if self.encoder_prior_composed_latent_normalization not in {'none', 'l2'}:
            raise ValueError(
                "encoder_prior_composed_latent_normalization must be one of 'none' or 'l2', "
                f"got {encoder_prior_composed_latent_normalization}"
            )
        if self.encoder_prior_composed_latent_norm_scale <= 0:
            raise ValueError(
                "encoder_prior_composed_latent_norm_scale must be positive, "
                f"got {encoder_prior_composed_latent_norm_scale}"
            )
        if self.dynamics_random_prior_scale < 0:
            raise ValueError(
                f"dynamics_random_prior_scale must be non-negative, got {dynamics_random_prior_scale}"
            )
        if self.dynamics_encoder_latent_normalization not in {'none', 'l2'}:
            raise ValueError(
                "dynamics_encoder_latent_normalization must be one of 'none' or 'l2', "
                f"got {dynamics_encoder_latent_normalization}"
            )
        if self.dynamics_encoder_latent_norm_scale <= 0:
            raise ValueError(
                "dynamics_encoder_latent_norm_scale must be positive, "
                f"got {dynamics_encoder_latent_norm_scale}"
            )
        if self.dynamics_trainable_latent_normalization not in {'none', 'l2', 'rms', 'l2_cap'}:
            raise ValueError(
                "dynamics_trainable_latent_normalization must be one of 'none', 'l2', 'rms', or 'l2_cap', "
                f"got {dynamics_trainable_latent_normalization}"
            )
        if self.dynamics_trainable_latent_norm_scale <= 0:
            raise ValueError(
                "dynamics_trainable_latent_norm_scale must be positive, "
                f"got {dynamics_trainable_latent_norm_scale}"
            )
        if self.dynamics_trainable_latent_norm_eps <= 0:
            raise ValueError(
                "dynamics_trainable_latent_norm_eps must be positive, "
                f"got {dynamics_trainable_latent_norm_eps}"
            )
        if self.dynamics_prior_latent_norm_cap < 0:
            raise ValueError(
                "dynamics_prior_latent_norm_cap must be non-negative, "
                f"got {dynamics_prior_latent_norm_cap}"
            )
        if self.encoder_prior_composed_latent_normalization == 'l2':
            if self.encoder_random_prior_scale <= 0 or self.encoder_random_prior_output_norm <= 0:
                raise ValueError(
                    "L2-normalized encoder-prior composition requires a non-zero encoder random prior."
                )
            if self.dynamics_encoder_latent_normalization != 'l2':
                raise ValueError(
                    "L2-normalized encoder-prior composition requires "
                    "dynamics_encoder_latent_normalization='l2'."
                )
            effective_prior_norm = (
                self.encoder_random_prior_scale * self.encoder_random_prior_output_norm
            )
            if self.dynamics_encoder_latent_norm_scale <= effective_prior_norm:
                raise ValueError(
                    "The learned encoder latent norm must strictly exceed the effective encoder-prior norm "
                    "so the composed latent can represent every direction: "
                    f"learned={self.dynamics_encoder_latent_norm_scale}, prior={effective_prior_norm}."
                )
            if self.dynamics_trainable_latent_normalization != 'l2':
                raise ValueError(
                    "L2-normalized encoder-prior composition requires "
                    "dynamics_trainable_latent_normalization='l2'."
                )
            if not torch.isclose(
                    torch.tensor(self.dynamics_trainable_latent_norm_scale),
                    torch.tensor(self.encoder_prior_composed_latent_norm_scale),
                    rtol=1e-6,
                    atol=1e-8,
            ).item():
                raise ValueError(
                    "Direct and recurrent final latent norms must match when using L2-normalized "
                    "encoder-prior composition: "
                    f"direct={self.encoder_prior_composed_latent_norm_scale}, "
                    f"recurrent={self.dynamics_trainable_latent_norm_scale}."
                )
        if self.latent_novelty_dim < 0:
            raise ValueError(f"latent_novelty_dim must be non-negative, got {latent_novelty_dim}")
        if self.latent_novelty_prior_scale < 0:
            raise ValueError(f"latent_novelty_prior_scale must be non-negative, got {latent_novelty_prior_scale}")
        if self.episodic_history_dim < 0:
            raise ValueError(f"episodic_history_dim must be non-negative, got {episodic_history_dim}")
        if self.episodic_inverse_embedding_dim < 0:
            raise ValueError(
                'episodic_inverse_embedding_dim must be non-negative, '
                f'got {episodic_inverse_embedding_dim}'
            )
        self.profile_timing = False
        self._profile_stats = {}
        self._value_prior_learning_mode = False
        self._collect_dynamics_prior_stats = False
        self._dynamics_prior_stats = {}
        self.members = nn.ModuleList([self.member_cls(*args, **kwargs) for _ in range(ensemble_size)])

        reference_member = self.members[0]
        self._share_member_modules(reference_member)
        self.value_support_size = reference_member.value_support_size
        self.reward_support_size = reference_member.reward_support_size
        self.action_space_size = reference_member.action_space_size
        self.continuous_action_space = getattr(reference_member, 'continuous_action_space', False)
        self.categorical_distribution = bool(getattr(reference_member, 'categorical_distribution', True))
        if self.categorical_distribution:
            value_support_range = kwargs.get('value_support_range', (-300.0, 301.0, 1.0))
            value_support = torch.arange(*value_support_range, dtype=torch.float32)
            if value_support.numel() != self.value_support_size:
                raise ValueError(
                    "value_support_range does not match the member value output size: "
                    f"range creates {value_support.numel()} atoms, member has {self.value_support_size}"
                )
            if value_support.numel() < 2:
                raise ValueError("Categorical value priors require at least two value-support atoms")
            value_support_step = value_support[1] - value_support[0]
            if not torch.allclose(
                    value_support[1:] - value_support[:-1],
                    torch.full_like(value_support[1:], value_support_step),
            ):
                raise ValueError("Categorical value priors require an evenly spaced value support")
        else:
            value_support = torch.zeros(1, dtype=torch.float32)
            value_support_step = torch.tensor(1.0, dtype=torch.float32)
        self.register_buffer('_value_prior_support', value_support, persistent=False)
        self.register_buffer('_value_prior_support_step', value_support_step.reshape(()), persistent=False)
        self.register_buffer(
            '_value_prior_support_indices',
            torch.arange(value_support.numel(), dtype=torch.float32),
            persistent=False,
        )
        if self.encoder_random_prior_scale > 0:
            observation_shape = kwargs.get('observation_shape', None)
            if not isinstance(observation_shape, int):
                raise NotImplementedError(
                    "Random encoder priors currently support vector observations with integer observation_shape."
                )
            latent_dim = self._infer_value_prior_input_dim(reference_member)
            if self.encoder_random_prior_mode == 'local_grid':
                prior_factory = lambda: _LocalGridEncoderPrior(
                    int(observation_shape),
                    latent_dim,
                    cell_size=float(encoder_random_prior_cell_size),
                    hash_size=int(encoder_random_prior_hash_size),
                    input_mean=encoder_random_prior_input_mean,
                    input_scale=encoder_random_prior_input_scale,
                    output_norm=self.encoder_random_prior_output_norm,
                )
            else:
                prior_factory = lambda: _RandomEncoderPrior(
                    int(observation_shape),
                    latent_dim,
                    hidden_size=int(encoder_random_prior_hidden_dim),
                    input_mean=encoder_random_prior_input_mean,
                    input_scale=encoder_random_prior_input_scale,
                    output_norm=self.encoder_random_prior_output_norm,
                )
            self.encoder_prior_networks = nn.ModuleList([prior_factory() for _ in range(ensemble_size)])
        else:
            self.encoder_prior_networks = None
        if self.random_prior_scale > 0:
            prior_input_dim = self._infer_value_prior_input_dim(reference_member)
            self.value_prior_networks = nn.ModuleList(
                [
                    _RandomValuePrior(
                        prior_input_dim,
                        latent_normalization=self.value_prior_latent_normalization,
                    )
                    for _ in range(ensemble_size)
                ]
            )
        else:
            self.value_prior_networks = None
        if self.dynamics_random_prior_scale > 0:
            if self.continuous_action_space:
                raise NotImplementedError("Random dynamics prior currently supports discrete action spaces only.")
            latent_dim = self._infer_value_prior_input_dim(reference_member)
            self.dynamics_prior_networks = nn.ModuleList(
                [
                    _RandomDynamicsPrior(
                        latent_dim,
                        self.action_space_size,
                        hidden_size=int(dynamics_random_prior_hidden_dim),
                        action_mode=self.dynamics_random_prior_action_mode,
                    )
                    for _ in range(ensemble_size)
                ]
            )
        else:
            self.dynamics_prior_networks = None
        if self.latent_novelty_dim > 0:
            latent_input_dim = self._infer_value_prior_input_dim(reference_member)
            self.latent_novelty_heads = nn.ModuleList(
                [
                    _LatentProjectionMLP(
                        latent_input_dim,
                        self.latent_novelty_dim,
                        hidden_size=int(latent_novelty_hidden_dim),
                    )
                    for _ in range(ensemble_size)
                ]
            )
            self.latent_novelty_prior_networks = nn.ModuleList(
                [
                    _RandomLatentPrior(
                        latent_input_dim,
                        self.latent_novelty_dim,
                        hidden_size=int(latent_novelty_hidden_dim),
                    )
                    for _ in range(ensemble_size)
                ]
            )
        else:
            self.latent_novelty_heads = None
            self.latent_novelty_prior_networks = None
        if self.episodic_history_dim > 0:
            latent_input_dim = self._infer_value_prior_input_dim(reference_member)
            self.episodic_history_models = nn.ModuleList(
                [
                    _EpisodicHistoryRewardModel(
                        latent_input_dim,
                        self.action_space_size,
                        self.episodic_history_dim,
                        hidden_dim=int(episodic_history_hidden_dim),
                    )
                    for _ in range(ensemble_size)
                ]
            )
        else:
            self.episodic_history_models = None
        if self.episodic_inverse_embedding_dim > 0:
            observation_shape = kwargs.get('observation_shape', None)
            if observation_shape is None:
                raise ValueError(
                    'episodic_inverse_embedding_dim requires an observation_shape.'
                )
            latent_input_dim = self._infer_value_prior_input_dim(reference_member)
            spatial_latent = not hasattr(reference_member, 'latent_state_dim')
            self.episodic_inverse_encoder = _ControllableStateEncoder(
                observation_shape,
                self.episodic_inverse_embedding_dim,
                hidden_dim=int(episodic_inverse_hidden_dim),
            )
            self.episodic_inverse_action_head = nn.Sequential(
                nn.Linear(2 * self.episodic_inverse_embedding_dim, int(episodic_inverse_hidden_dim)),
                nn.ReLU(inplace=True),
                nn.Linear(int(episodic_inverse_hidden_dim), self.action_space_size),
            )
            self.episodic_inverse_bridges = nn.ModuleList(
                [
                    _ControllableStateBridge(
                        latent_input_dim,
                        self.episodic_inverse_embedding_dim,
                        hidden_dim=int(episodic_inverse_hidden_dim),
                        spatial=spatial_latent,
                    )
                    for _ in range(ensemble_size)
                ]
            )
        else:
            self.episodic_inverse_encoder = None
            self.episodic_inverse_action_head = None
            self.episodic_inverse_bridges = None

    def _share_member_modules(self, reference_member: nn.Module) -> None:
        for member in list(self.members)[1:]:
            if self.shared_representation:
                member.representation_network = reference_member.representation_network
            if self.shared_prediction:
                member.prediction_network = reference_member.prediction_network
            if self.shared_projection and hasattr(reference_member, 'projection'):
                member.projection = reference_member.projection
            if self.shared_projection and hasattr(reference_member, 'prediction_head'):
                member.prediction_head = reference_member.prediction_head
            if self.shared_intrinsic_reward_head and hasattr(
                    reference_member, 'intrinsic_reward_head'
            ):
                member.intrinsic_reward_head = reference_member.intrinsic_reward_head

    @staticmethod
    def _infer_value_prior_input_dim(reference_member: nn.Module) -> int:
        if hasattr(reference_member, 'latent_state_dim'):
            return int(reference_member.latent_state_dim)
        dynamics_network = getattr(reference_member, 'dynamics_network', None)
        dynamics_channels = getattr(dynamics_network, 'num_channels', None)
        action_encoding_dim = getattr(reference_member, 'action_encoding_dim', 0)
        if dynamics_channels is not None:
            return int(dynamics_channels) - int(action_encoding_dim)
        raise ValueError(
            "Unable to infer latent feature dimension for random value prior. "
            "Use an MLP/conv MuZero member with latent_state_dim or dynamics_network.num_channels."
        )

    def _resolve_member_id(self, member_id: Optional[int] = None) -> int:
        if member_id is None:
            member_id = self.active_member_id
        if isinstance(member_id, (list, tuple)):
            if len(member_id) != 1:
                raise ValueError(f"Expected a scalar member_id, got {member_id}")
            member_id = member_id[0]
        if torch.is_tensor(member_id):
            if member_id.numel() != 1:
                raise ValueError(f"Expected a scalar member_id tensor, got shape {tuple(member_id.shape)}")
            member_id = member_id.item()
        if not 0 <= int(member_id) < self.ensemble_size:
            raise ValueError(f"member_id must be in [0, {self.ensemble_size}), got {member_id}")
        return int(member_id)

    def _resolve_member_ids(self, batch_size: int, member_id: Optional[Union[int, Sequence[int], torch.Tensor]]) -> List[int]:
        if member_id is None:
            member_id = self.active_member_id
        if torch.is_tensor(member_id):
            member_id = member_id.detach().cpu().view(-1).tolist()
        if isinstance(member_id, (list, tuple)):
            if len(member_id) != batch_size:
                raise ValueError(f"Expected {batch_size} member ids, got {len(member_id)}")
            member_ids = [self._resolve_member_id(int(mid)) for mid in member_id]
        else:
            member_ids = [self._resolve_member_id(member_id) for _ in range(batch_size)]
        return member_ids

    def set_active_member(self, member_id: int) -> None:
        self.active_member_id = self._resolve_member_id(member_id)

    def set_active_members(self, member_ids: Sequence[int]) -> None:
        self.active_member_id = [self._resolve_member_id(member_id) for member_id in member_ids]

    def set_value_prior_learning_mode(self, enabled: bool) -> None:
        """Return base categorical logits while the policy builds the prior-aware value loss."""
        self._value_prior_learning_mode = bool(enabled)

    def set_profile_timing(self, enabled: bool) -> None:
        self.profile_timing = bool(enabled)

    def reset_profile_stats(self) -> None:
        self._profile_stats = {}

    def get_profile_stats(self, reset: bool = False) -> dict:
        stats = dict(self._profile_stats)
        if reset:
            self.reset_profile_stats()
        return stats

    def _profile_sync(self, device: torch.device) -> None:
        if self.profile_timing and device.type == 'cuda' and torch.cuda.is_available():
            torch.cuda.synchronize(device)

    def _profile_start(self, device: torch.device):
        if not self.profile_timing:
            return None
        self._profile_sync(device)
        return time.perf_counter()

    def _profile_elapsed(self, start, device: torch.device) -> float:
        if start is None:
            return 0.0
        self._profile_sync(device)
        return time.perf_counter() - start

    def _profile_add(self, name: str, value: float) -> None:
        if self.profile_timing:
            self._profile_stats[name] = self._profile_stats.get(name, 0.0) + float(value)

    @staticmethod
    def _group_member_indices(member_ids: Sequence[int]):
        groups = []
        group_by_member = {}
        for idx, member_id in enumerate(member_ids):
            if member_id not in group_by_member:
                group_by_member[member_id] = []
                groups.append((member_id, group_by_member[member_id]))
            group_by_member[member_id].append(idx)
        return groups

    @staticmethod
    def _groups_preserve_batch_order(groups, batch_size: int) -> bool:
        grouped_indices = [idx for _, indices in groups for idx in indices]
        return grouped_indices == list(range(batch_size))

    @staticmethod
    def _indices_are_slice(indices: Sequence[int]) -> bool:
        if len(indices) == 0:
            return False
        return indices == list(range(indices[0], indices[-1] + 1))

    @staticmethod
    def _select_rows(tensor: torch.Tensor, indices: Sequence[int], batch_size: int) -> torch.Tensor:
        if tensor.shape[0] != batch_size:
            return tensor
        if _EnsembleMuZeroBase._indices_are_slice(indices):
            return tensor[indices[0]:indices[-1] + 1]
        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=tensor.device)
        return tensor.index_select(0, index_tensor)

    @staticmethod
    def _extend_reward_chunks(reward_chunks: list, reward) -> None:
        if torch.is_tensor(reward):
            reward_chunks.append(reward)
        elif isinstance(reward, list):
            reward_chunks.extend(reward)
        elif isinstance(reward, tuple):
            reward_chunks.extend(list(reward))
        else:
            reward_chunks.append(reward)

    def get_member(self, member_id: Optional[int] = None) -> nn.Module:
        return self.members[self._resolve_member_id(member_id)]

    @staticmethod
    def _scalar_transform_value(value: torch.Tensor, epsilon: float = 0.001) -> torch.Tensor:
        return torch.sign(value) * (torch.sqrt(torch.abs(value) + 1.0) - 1.0) + epsilon * value

    @staticmethod
    def _inverse_scalar_transform_value(value: torch.Tensor, epsilon: float = 0.001) -> torch.Tensor:
        root = (torch.sqrt(1.0 + 4.0 * epsilon * (torch.abs(value) + 1.0 + epsilon)) - 1.0) / (
            2.0 * epsilon
        )
        return torch.sign(value) * (root.square() - 1.0)

    def _value_logits_to_scalar(self, value: torch.Tensor) -> torch.Tensor:
        if not self.categorical_distribution:
            return value
        support = self._value_prior_support.to(dtype=value.dtype, device=value.device).view(1, -1)
        transformed_value = (torch.softmax(value, dim=1) * support).sum(dim=1, keepdim=True)
        return self._inverse_scalar_transform_value(transformed_value)

    def value_prior_from_latent(
            self,
            latent_state: torch.Tensor,
            member_id: Optional[Union[int, Sequence[int], torch.Tensor]] = None,
    ) -> Optional[torch.Tensor]:
        """Evaluate the frozen scalar prior for each latent using its assigned ensemble member."""
        if self.value_prior_networks is None:
            return None
        batch_size = int(latent_state.shape[0])
        member_ids = self._resolve_member_ids(batch_size, member_id)
        groups = self._group_member_indices(member_ids)
        batch_order_preserved = self._groups_preserve_batch_order(groups, batch_size)
        prior_chunks = [] if batch_order_preserved else [None] * batch_size

        for current_member_id, indices in groups:
            member_latent = self._select_rows(latent_state, indices, batch_size)
            if self.value_prior_backprop_to_latent:
                member_prior = self.value_prior_networks[current_member_id](member_latent)
            else:
                with torch.no_grad():
                    member_prior = self.value_prior_networks[current_member_id](member_latent.detach())
            member_prior = float(self.random_prior_scale) * member_prior
            if batch_order_preserved:
                prior_chunks.append(member_prior)
            else:
                for local_idx, global_idx in enumerate(indices):
                    prior_chunks[global_idx] = member_prior[local_idx:local_idx + 1]

        return torch.cat(prior_chunks, dim=0).to(
            dtype=latent_state.dtype,
            device=latent_state.device,
        )

    def _add_scalar_prior_to_categorical_logits(
            self,
            base_logits: torch.Tensor,
            scalar_prior: torch.Tensor,
    ) -> torch.Tensor:
        """Translate a categorical value distribution by an additive scalar value prior."""
        if base_logits.dim() != 2 or base_logits.shape[1] != self.value_support_size:
            raise ValueError(
                "Expected categorical value logits with shape [batch, value_support_size], "
                f"got {tuple(base_logits.shape)}"
            )
        calculation_dtype = torch.float32
        logits = base_logits.to(dtype=calculation_dtype)
        support = self._value_prior_support.to(dtype=calculation_dtype, device=logits.device).view(1, -1)
        step = self._value_prior_support_step.to(dtype=calculation_dtype, device=logits.device)
        base_probs = torch.softmax(logits, dim=1)
        base_transformed_value = (base_probs * support).sum(dim=1, keepdim=True)
        base_scalar_value = self._inverse_scalar_transform_value(base_transformed_value)
        shifted_scalar_value = base_scalar_value + scalar_prior.to(dtype=calculation_dtype)
        shifted_transformed_value = torch.maximum(
            torch.minimum(self._scalar_transform_value(shifted_scalar_value), support[:, -1:]),
            support[:, :1],
        )
        shift_atoms = (shifted_transformed_value - base_transformed_value) / step

        output_indices = self._value_prior_support_indices.to(
            dtype=calculation_dtype, device=logits.device
        ).view(1, -1)
        source_positions = output_indices - shift_atoms
        source_low_float = torch.floor(source_positions)
        source_low = source_low_float.long()
        source_high = source_low + 1
        high_weight = source_positions - source_low_float
        low_weight = 1.0 - high_weight

        low_valid = (source_low >= 0) & (source_low < self.value_support_size)
        high_valid = (source_high >= 0) & (source_high < self.value_support_size)
        low_probs = base_probs.gather(1, source_low.clamp(0, self.value_support_size - 1))
        high_probs = base_probs.gather(1, source_high.clamp(0, self.value_support_size - 1))
        translated_probs = (
            low_weight * low_probs * low_valid.to(dtype=calculation_dtype)
            + high_weight * high_probs * high_valid.to(dtype=calculation_dtype)
        )
        translated_probs = translated_probs / translated_probs.sum(dim=1, keepdim=True).clamp_min(1e-12)

        # Translation is exact away from finite-support boundaries. Correct
        # any truncated boundary mass with the smallest convex adjustment that
        # restores the requested expected support exactly.
        translated_mean = (translated_probs * support).sum(dim=1, keepdim=True)
        use_upper_anchor = shifted_transformed_value >= translated_mean
        anchor_value = torch.where(use_upper_anchor, support[:, -1:], support[:, :1])
        correction_denominator = anchor_value - translated_mean
        correction_weight = torch.where(
            correction_denominator.abs() > 1e-8,
            (shifted_transformed_value - translated_mean) / correction_denominator,
            torch.ones_like(correction_denominator),
        ).clamp(0.0, 1.0)
        batch_size = int(base_logits.shape[0])
        lower_anchor = F.one_hot(
            torch.zeros(batch_size, dtype=torch.long, device=logits.device),
            num_classes=self.value_support_size,
        ).to(dtype=calculation_dtype)
        upper_anchor = F.one_hot(
            torch.full(
                (batch_size,), self.value_support_size - 1, dtype=torch.long, device=logits.device
            ),
            num_classes=self.value_support_size,
        ).to(dtype=calculation_dtype)
        anchor_probs = torch.where(use_upper_anchor, upper_anchor, lower_anchor)
        corrected_probs = (
            (1.0 - correction_weight) * translated_probs
            + correction_weight * anchor_probs
        )
        return corrected_probs.clamp_min(1e-30).log().to(dtype=base_logits.dtype)

    def _apply_value_prior(self, output: MZNetworkOutput, member_id: int, device: torch.device) -> MZNetworkOutput:
        if self.value_prior_networks is None:
            return output
        if self._value_prior_learning_mode and self.categorical_distribution:
            # The policy trains these base logits against target - prior and
            # adds a scalar consistency term for base_value + prior.
            return output
        prior_start = self._profile_start(device)
        prior_value = self.value_prior_from_latent(output.latent_state, member_id=member_id)
        prior_value = prior_value.to(dtype=output.value.dtype, device=output.value.device)
        if self.categorical_distribution:
            value = self._add_scalar_prior_to_categorical_logits(output.value, prior_value)
        else:
            value = output.value + prior_value
        self._profile_add('value_prior_s', self._profile_elapsed(prior_start, device))
        self._profile_add('value_prior_call_count', 1)
        return MZNetworkOutput(value, output.reward, output.policy_logits, output.latent_state)

    def _latent_reduce_dims(self, latent_state: torch.Tensor):
        if latent_state.dim() <= 1:
            return None
        return tuple(range(1, latent_state.dim()))

    def _latent_l2_norm(self, latent_state: torch.Tensor) -> torch.Tensor:
        reduce_dims = self._latent_reduce_dims(latent_state)
        if reduce_dims is None:
            return latent_state.abs()
        return latent_state.float().pow(2).sum(dim=reduce_dims, keepdim=True).sqrt()

    def _l2_normalize_latent(self, latent_state: torch.Tensor, scale: float) -> torch.Tensor:
        reduce_dims = self._latent_reduce_dims(latent_state)
        if reduce_dims is None:
            return latent_state
        norm = self._latent_l2_norm(latent_state)
        zero_mask = norm < self.dynamics_trainable_latent_norm_eps
        if zero_mask.any().item():
            fallback = torch.zeros_like(latent_state)
            fallback_view = fallback.reshape(fallback.shape[0], -1)
            fallback_view[:, 0] = self.dynamics_trainable_latent_norm_eps
            latent_state = torch.where(zero_mask.expand_as(latent_state), latent_state + fallback, latent_state)
            norm = self._latent_l2_norm(latent_state)
        denom = norm.clamp_min(self.dynamics_trainable_latent_norm_eps)
        denom = denom.to(dtype=latent_state.dtype)
        scale_tensor = torch.as_tensor(scale, dtype=latent_state.dtype, device=latent_state.device)
        return latent_state * (scale_tensor / denom)

    def _l2_cap_latent(self, latent_state: torch.Tensor, cap: float) -> torch.Tensor:
        reduce_dims = self._latent_reduce_dims(latent_state)
        if reduce_dims is None or cap <= 0:
            return latent_state
        norm = self._latent_l2_norm(latent_state)
        cap_tensor = torch.as_tensor(cap, dtype=latent_state.dtype, device=latent_state.device)
        norm_for_scale = norm.clamp_min(self.dynamics_trainable_latent_norm_eps).to(dtype=latent_state.dtype)
        # Smooth radial cap: small vectors are nearly unchanged, large vectors approach ``cap``.
        factor = cap_tensor * torch.tanh(norm.to(dtype=latent_state.dtype) / cap_tensor) / norm_for_scale
        return latent_state * factor

    def _normalize_encoder_latent(self, latent_state: torch.Tensor) -> torch.Tensor:
        mode = self.dynamics_encoder_latent_normalization
        if mode == 'none':
            return latent_state
        if mode == 'l2':
            return self._l2_normalize_latent(latent_state, self.dynamics_encoder_latent_norm_scale)
        raise ValueError(f"Unsupported encoder latent normalization mode: {mode}")

    def _normalize_composed_encoder_latent(self, latent_state: torch.Tensor) -> torch.Tensor:
        mode = self.encoder_prior_composed_latent_normalization
        if mode == 'none':
            return latent_state
        if mode == 'l2':
            return self._l2_normalize_latent(
                latent_state, self.encoder_prior_composed_latent_norm_scale
            )
        raise ValueError(f"Unsupported composed encoder latent normalization mode: {mode}")

    def _normalize_trainable_dynamics_latent(self, latent_state: torch.Tensor) -> torch.Tensor:
        mode = self.dynamics_trainable_latent_normalization
        if mode == 'none':
            return latent_state
        if latent_state.dim() <= 1:
            return latent_state
        reduce_dims = tuple(range(1, latent_state.dim()))
        latent_float = latent_state.float()
        if mode == 'l2':
            denom = latent_float.pow(2).sum(dim=reduce_dims, keepdim=True).sqrt()
        elif mode == 'rms':
            denom = latent_float.pow(2).mean(dim=reduce_dims, keepdim=True).sqrt()
        elif mode == 'l2_cap':
            return self._l2_cap_latent(latent_state, self.dynamics_trainable_latent_norm_scale)
        else:
            raise ValueError(f"Unsupported trainable dynamics latent normalization mode: {mode}")
        scale = torch.as_tensor(
            self.dynamics_trainable_latent_norm_scale,
            dtype=latent_state.dtype,
            device=latent_state.device,
        )
        denom = denom.clamp_min(self.dynamics_trainable_latent_norm_eps).to(dtype=latent_state.dtype)
        return latent_state * (scale / denom)

    def _bound_dynamics_prior_delta(self, prior_delta: torch.Tensor) -> torch.Tensor:
        if self.dynamics_prior_latent_norm_cap <= 0:
            return prior_delta
        return self._l2_cap_latent(prior_delta, self.dynamics_prior_latent_norm_cap)

    def reset_dynamics_prior_stats(self) -> None:
        self._dynamics_prior_stats = {}

    def set_collect_dynamics_prior_stats(self, enabled: bool) -> None:
        self._collect_dynamics_prior_stats = bool(enabled)
        if enabled:
            self.reset_dynamics_prior_stats()

    def _add_dynamics_prior_stat_tensor(self, name: str, tensor: torch.Tensor) -> None:
        norm = self._latent_l2_norm(tensor.detach()).reshape(-1).float()
        if norm.numel() == 0:
            return
        entry = self._dynamics_prior_stats.setdefault(
            name,
            {'sum': 0.0, 'max': 0.0, 'count': 0},
        )
        entry['sum'] += float(norm.sum().item())
        entry['max'] = max(entry['max'], float(norm.max().item()))
        entry['count'] += int(norm.numel())

    def _record_dynamics_prior_stats(
            self,
            raw_trainable_latent: torch.Tensor,
            trainable_latent: torch.Tensor,
            raw_prior_delta: torch.Tensor,
            prior_delta: torch.Tensor,
            full_latent: torch.Tensor,
    ) -> None:
        if not self._collect_dynamics_prior_stats:
            return
        self._add_dynamics_prior_stat_tensor('raw_trainable_l2', raw_trainable_latent)
        self._add_dynamics_prior_stat_tensor('trainable_l2', trainable_latent)
        self._add_dynamics_prior_stat_tensor('raw_prior_l2', raw_prior_delta)
        self._add_dynamics_prior_stat_tensor('prior_l2', prior_delta)
        self._add_dynamics_prior_stat_tensor('full_l2', full_latent)

    def get_dynamics_prior_stats(self, reset: bool = False) -> dict:
        result = {}
        for name, entry in self._dynamics_prior_stats.items():
            count = max(int(entry.get('count', 0)), 1)
            result[f'{name}_mean'] = float(entry.get('sum', 0.0)) / count
            result[f'{name}_max'] = float(entry.get('max', 0.0))
        if reset:
            self.reset_dynamics_prior_stats()
        return result

    @property
    def representation_network(self) -> nn.Module:
        return self.get_member(0).representation_network

    @property
    def dynamics_network(self) -> nn.Module:
        return self.get_member(0).dynamics_network

    @property
    def prediction_network(self) -> nn.Module:
        return self.get_member(0).prediction_network

    @property
    def encoder_hook(self):
        return self.get_member(0).encoder_hook

    def _member_initial_with_encoder_normalization(
            self,
            member: nn.Module,
            member_id: int,
            obs: torch.Tensor,
    ) -> MZNetworkOutput:
        if not hasattr(member, '_representation') or not hasattr(member, '_prediction'):
            raise NotImplementedError(
                "Encoder latent transforms require MuZero members exposing _representation and _prediction."
            )
        batch_size = obs.size(0)
        latent_state = member._representation(obs)
        latent_state = self._normalize_encoder_latent(latent_state)
        if self.encoder_prior_networks is not None:
            with torch.no_grad():
                prior = self.encoder_prior_networks[member_id](obs.detach())
                prior = prior.to(dtype=latent_state.dtype, device=latent_state.device)
            if prior.shape != latent_state.shape:
                raise ValueError(
                    f"Encoder prior shape {tuple(prior.shape)} does not match latent shape {tuple(latent_state.shape)}."
                )
            latent_state = latent_state + self.encoder_random_prior_scale * prior
        latent_state = self._normalize_composed_encoder_latent(latent_state)
        policy_logits, value = member._prediction(latent_state)
        return MZNetworkOutput(value, [0. for _ in range(batch_size)], policy_logits, latent_state)

    def _run_batched_member_initial_with_encoder_normalization(
            self,
            obs: torch.Tensor,
            member_id: Optional[Union[int, Sequence[int], torch.Tensor]] = None,
    ) -> MZNetworkOutput:
        device = obs.device
        total_start = self._profile_start(device)
        resolve_start = self._profile_start(device)
        batch_size = obs.shape[0]
        member_ids = self._resolve_member_ids(batch_size, member_id)
        self._profile_add('initial_inference_resolve_s', self._profile_elapsed(resolve_start, device))
        self._profile_add('initial_inference_call_count', 1)
        self._profile_add('initial_inference_batch_rows', batch_size)

        groups = self._group_member_indices(member_ids)
        batch_order_preserved = self._groups_preserve_batch_order(groups, batch_size)
        if len(groups) == 1:
            self._profile_add('initial_inference_fast_path_count', 1)
        else:
            self._profile_add('initial_inference_grouped_path_count', 1)

        if batch_order_preserved:
            value_chunks, reward_chunks, policy_chunks, latent_chunks = [], [], [], []
        else:
            values, rewards, policy_logits, latent_states = [None] * batch_size, [None] * batch_size, [None] * batch_size, [None] * batch_size

        for current_member_id, indices in groups:
            index_start = self._profile_start(device)
            member_obs = self._select_rows(obs, indices, batch_size)
            self._profile_add('initial_inference_index_s', self._profile_elapsed(index_start, device))
            member_start = self._profile_start(device)
            output = self._member_initial_with_encoder_normalization(
                self.members[current_member_id], current_member_id, member_obs
            )
            self._profile_add('initial_inference_member_forward_s', self._profile_elapsed(member_start, device))
            self._profile_add('initial_inference_member_forward_count', 1)
            output = self._apply_value_prior(output, current_member_id, device)

            if batch_order_preserved:
                value_chunks.append(output.value)
                policy_chunks.append(output.policy_logits)
                latent_chunks.append(output.latent_state)
                self._extend_reward_chunks(reward_chunks, output.reward)
            else:
                stitch_start = self._profile_start(device)
                for local_idx, global_idx in enumerate(indices):
                    values[global_idx] = output.value[local_idx:local_idx + 1]
                    policy_logits[global_idx] = output.policy_logits[local_idx:local_idx + 1]
                    latent_states[global_idx] = output.latent_state[local_idx:local_idx + 1]
                    if torch.is_tensor(output.reward):
                        rewards[global_idx] = output.reward[local_idx:local_idx + 1]
                    else:
                        rewards[global_idx] = output.reward[local_idx]
                self._profile_add('initial_inference_stitch_s', self._profile_elapsed(stitch_start, device))

        concat_start = self._profile_start(device)
        if batch_order_preserved:
            value = torch.cat(value_chunks, dim=0)
            policy_logit = torch.cat(policy_chunks, dim=0)
            latent_state = torch.cat(latent_chunks, dim=0)
            if torch.is_tensor(reward_chunks[0]):
                reward = torch.cat(reward_chunks, dim=0)
            else:
                reward = reward_chunks
        else:
            value = torch.cat(values, dim=0)
            policy_logit = torch.cat(policy_logits, dim=0)
            latent_state = torch.cat(latent_states, dim=0)
            if torch.is_tensor(rewards[0]):
                reward = torch.cat(rewards, dim=0)
            else:
                reward = rewards
        self._profile_add('initial_inference_concat_s', self._profile_elapsed(concat_start, device))
        self._profile_add('initial_inference_total_s', self._profile_elapsed(total_start, device))
        return MZNetworkOutput(value, reward, policy_logit, latent_state)

    def _run_batched_member_inference(self, fn_name: str, batch: torch.Tensor, *args, member_id=None) -> MZNetworkOutput:
        device = batch.device
        total_start = self._profile_start(device)
        resolve_start = self._profile_start(device)
        batch_size = batch.shape[0]
        member_ids = self._resolve_member_ids(batch_size, member_id)
        self._profile_add(f'{fn_name}_resolve_s', self._profile_elapsed(resolve_start, device))
        self._profile_add(f'{fn_name}_call_count', 1)
        self._profile_add(f'{fn_name}_batch_rows', batch_size)

        groups = self._group_member_indices(member_ids)
        if len(groups) == 1:
            self._profile_add(f'{fn_name}_fast_path_count', 1)
            member_start = self._profile_start(device)
            output = getattr(self.members[groups[0][0]], fn_name)(batch, *args)
            self._profile_add(f'{fn_name}_member_forward_s', self._profile_elapsed(member_start, device))
            self._profile_add(f'{fn_name}_member_forward_count', 1)
            output = self._apply_value_prior(output, groups[0][0], device)
            self._profile_add(f'{fn_name}_total_s', self._profile_elapsed(total_start, device))
            return output

        batch_order_preserved = self._groups_preserve_batch_order(groups, batch_size)
        self._profile_add(f'{fn_name}_grouped_path_count', 1)
        if batch_order_preserved:
            self._profile_add(f'{fn_name}_order_preserved_group_count', 1)
            value_chunks, reward_chunks, policy_chunks, latent_chunks = [], [], [], []
        else:
            values, rewards, policy_logits, latent_states = [None] * batch_size, [None] * batch_size, [None] * batch_size, [None] * batch_size

        for current_member_id, indices in groups:
            index_start = self._profile_start(device)
            member_batch = self._select_rows(batch, indices, batch_size)
            member_args = [
                self._select_rows(arg, indices, batch_size)
                if torch.is_tensor(arg) and len(arg.shape) > 0 and arg.shape[0] == batch_size else arg
                for arg in args
            ]
            self._profile_add(f'{fn_name}_index_s', self._profile_elapsed(index_start, device))
            member_start = self._profile_start(device)
            output = getattr(self.members[current_member_id], fn_name)(member_batch, *member_args)
            self._profile_add(f'{fn_name}_member_forward_s', self._profile_elapsed(member_start, device))
            self._profile_add(f'{fn_name}_member_forward_count', 1)
            output = self._apply_value_prior(output, current_member_id, device)

            if batch_order_preserved:
                value_chunks.append(output.value)
                policy_chunks.append(output.policy_logits)
                latent_chunks.append(output.latent_state)
                self._extend_reward_chunks(reward_chunks, output.reward)
            else:
                stitch_start = self._profile_start(device)
                for local_idx, global_idx in enumerate(indices):
                    values[global_idx] = output.value[local_idx:local_idx + 1]
                    policy_logits[global_idx] = output.policy_logits[local_idx:local_idx + 1]
                    latent_states[global_idx] = output.latent_state[local_idx:local_idx + 1]
                    if torch.is_tensor(output.reward):
                        rewards[global_idx] = output.reward[local_idx:local_idx + 1]
                    else:
                        rewards[global_idx] = output.reward[local_idx]
                self._profile_add(f'{fn_name}_stitch_s', self._profile_elapsed(stitch_start, device))

        concat_start = self._profile_start(device)
        if batch_order_preserved:
            value = torch.cat(value_chunks, dim=0)
            policy_logit = torch.cat(policy_chunks, dim=0)
            latent_state = torch.cat(latent_chunks, dim=0)
            if torch.is_tensor(reward_chunks[0]):
                reward = torch.cat(reward_chunks, dim=0)
            else:
                reward = reward_chunks
        else:
            value = torch.cat(values, dim=0)
            policy_logit = torch.cat(policy_logits, dim=0)
            latent_state = torch.cat(latent_states, dim=0)
            if torch.is_tensor(rewards[0]):
                reward = torch.cat(rewards, dim=0)
            else:
                reward = rewards
        self._profile_add(f'{fn_name}_concat_s', self._profile_elapsed(concat_start, device))
        self._profile_add(f'{fn_name}_total_s', self._profile_elapsed(total_start, device))
        return MZNetworkOutput(value, reward, policy_logit, latent_state)

    def _member_recurrent_with_dynamics_prior(
            self,
            member: nn.Module,
            member_id: int,
            latent_state: torch.Tensor,
            action: torch.Tensor,
    ) -> MZNetworkOutput:
        if not hasattr(member, '_dynamics') or not hasattr(member, '_prediction'):
            raise NotImplementedError(
                "Random dynamics prior requires MuZero members exposing _dynamics and _prediction."
            )
        raw_next_latent_state, reward = member._dynamics(latent_state, action)
        next_latent_state = self._normalize_trainable_dynamics_latent(raw_next_latent_state)
        with torch.no_grad():
            raw_prior_delta = self.dynamics_prior_networks[member_id](latent_state, action)
            raw_prior_delta = raw_prior_delta * float(self.dynamics_random_prior_scale)
            prior_delta = self._bound_dynamics_prior_delta(raw_prior_delta)
        full_latent_state = next_latent_state + prior_delta
        if self.encoder_prior_composed_latent_normalization == 'l2':
            full_latent_state = self._l2_normalize_latent(
                full_latent_state, self.encoder_prior_composed_latent_norm_scale
            )
        self._record_dynamics_prior_stats(
            raw_next_latent_state,
            next_latent_state,
            raw_prior_delta,
            prior_delta,
            full_latent_state,
        )
        policy_logits, value = member._prediction(full_latent_state)
        return MZNetworkOutput(value, reward, policy_logits, full_latent_state)

    def _member_recurrent_with_trainable_dynamics_normalization(
            self,
            member: nn.Module,
            latent_state: torch.Tensor,
            action: torch.Tensor,
    ) -> MZNetworkOutput:
        if not hasattr(member, '_dynamics') or not hasattr(member, '_prediction'):
            raise NotImplementedError(
                "Trainable dynamics latent normalization requires MuZero members exposing _dynamics and _prediction."
            )
        raw_next_latent_state, reward = member._dynamics(latent_state, action)
        next_latent_state = self._normalize_trainable_dynamics_latent(raw_next_latent_state)
        policy_logits, value = member._prediction(next_latent_state)
        return MZNetworkOutput(value, reward, policy_logits, next_latent_state)

    def _run_batched_member_recurrent_with_dynamics_prior(
            self,
            latent_state: torch.Tensor,
            action: torch.Tensor,
            member_id: Optional[Union[int, Sequence[int], torch.Tensor]] = None,
    ) -> MZNetworkOutput:
        device = latent_state.device
        total_start = self._profile_start(device)
        resolve_start = self._profile_start(device)
        batch_size = latent_state.shape[0]
        member_ids = self._resolve_member_ids(batch_size, member_id)
        self._profile_add('recurrent_inference_resolve_s', self._profile_elapsed(resolve_start, device))
        self._profile_add('recurrent_inference_call_count', 1)
        self._profile_add('recurrent_inference_batch_rows', batch_size)

        groups = self._group_member_indices(member_ids)
        if len(groups) == 1:
            self._profile_add('recurrent_inference_fast_path_count', 1)
            member_start = self._profile_start(device)
            current_member_id = groups[0][0]
            output = self._member_recurrent_with_dynamics_prior(
                self.members[current_member_id],
                current_member_id,
                latent_state,
                action,
            )
            self._profile_add('recurrent_inference_member_forward_s', self._profile_elapsed(member_start, device))
            self._profile_add('recurrent_inference_member_forward_count', 1)
            output = self._apply_value_prior(output, current_member_id, device)
            self._profile_add('dynamics_prior_call_count', 1)
            self._profile_add('recurrent_inference_total_s', self._profile_elapsed(total_start, device))
            return output

        batch_order_preserved = self._groups_preserve_batch_order(groups, batch_size)
        self._profile_add('recurrent_inference_grouped_path_count', 1)
        if batch_order_preserved:
            self._profile_add('recurrent_inference_order_preserved_group_count', 1)
            value_chunks, reward_chunks, policy_chunks, latent_chunks = [], [], [], []
        else:
            values, rewards, policy_logits, latent_states = (
                [None] * batch_size,
                [None] * batch_size,
                [None] * batch_size,
                [None] * batch_size,
            )

        for current_member_id, indices in groups:
            index_start = self._profile_start(device)
            member_latent_state = self._select_rows(latent_state, indices, batch_size)
            member_action = self._select_rows(action, indices, batch_size)
            self._profile_add('recurrent_inference_index_s', self._profile_elapsed(index_start, device))

            member_start = self._profile_start(device)
            output = self._member_recurrent_with_dynamics_prior(
                self.members[current_member_id],
                current_member_id,
                member_latent_state,
                member_action,
            )
            self._profile_add('recurrent_inference_member_forward_s', self._profile_elapsed(member_start, device))
            self._profile_add('recurrent_inference_member_forward_count', 1)
            self._profile_add('dynamics_prior_call_count', 1)
            output = self._apply_value_prior(output, current_member_id, device)

            if batch_order_preserved:
                value_chunks.append(output.value)
                policy_chunks.append(output.policy_logits)
                latent_chunks.append(output.latent_state)
                self._extend_reward_chunks(reward_chunks, output.reward)
            else:
                stitch_start = self._profile_start(device)
                for local_idx, global_idx in enumerate(indices):
                    values[global_idx] = output.value[local_idx:local_idx + 1]
                    policy_logits[global_idx] = output.policy_logits[local_idx:local_idx + 1]
                    latent_states[global_idx] = output.latent_state[local_idx:local_idx + 1]
                    if torch.is_tensor(output.reward):
                        rewards[global_idx] = output.reward[local_idx:local_idx + 1]
                    else:
                        rewards[global_idx] = output.reward[local_idx]
                self._profile_add('recurrent_inference_stitch_s', self._profile_elapsed(stitch_start, device))

        concat_start = self._profile_start(device)
        if batch_order_preserved:
            value = torch.cat(value_chunks, dim=0)
            policy_logit = torch.cat(policy_chunks, dim=0)
            next_latent_state = torch.cat(latent_chunks, dim=0)
            if torch.is_tensor(reward_chunks[0]):
                reward = torch.cat(reward_chunks, dim=0)
            else:
                reward = reward_chunks
        else:
            value = torch.cat(values, dim=0)
            policy_logit = torch.cat(policy_logits, dim=0)
            next_latent_state = torch.cat(latent_states, dim=0)
            if torch.is_tensor(rewards[0]):
                reward = torch.cat(rewards, dim=0)
            else:
                reward = rewards
        self._profile_add('recurrent_inference_concat_s', self._profile_elapsed(concat_start, device))
        self._profile_add('recurrent_inference_total_s', self._profile_elapsed(total_start, device))
        return MZNetworkOutput(value, reward, policy_logit, next_latent_state)

    def _run_batched_member_recurrent_with_trainable_dynamics_normalization(
            self,
            latent_state: torch.Tensor,
            action: torch.Tensor,
            member_id: Optional[Union[int, Sequence[int], torch.Tensor]] = None,
    ) -> MZNetworkOutput:
        device = latent_state.device
        total_start = self._profile_start(device)
        resolve_start = self._profile_start(device)
        batch_size = latent_state.shape[0]
        member_ids = self._resolve_member_ids(batch_size, member_id)
        self._profile_add('recurrent_inference_resolve_s', self._profile_elapsed(resolve_start, device))
        self._profile_add('recurrent_inference_call_count', 1)
        self._profile_add('recurrent_inference_batch_rows', batch_size)

        groups = self._group_member_indices(member_ids)
        if len(groups) == 1:
            self._profile_add('recurrent_inference_fast_path_count', 1)
            member_start = self._profile_start(device)
            current_member_id = groups[0][0]
            output = self._member_recurrent_with_trainable_dynamics_normalization(
                self.members[current_member_id],
                latent_state,
                action,
            )
            self._profile_add('recurrent_inference_member_forward_s', self._profile_elapsed(member_start, device))
            self._profile_add('recurrent_inference_member_forward_count', 1)
            output = self._apply_value_prior(output, current_member_id, device)
            self._profile_add('recurrent_inference_total_s', self._profile_elapsed(total_start, device))
            return output

        batch_order_preserved = self._groups_preserve_batch_order(groups, batch_size)
        self._profile_add('recurrent_inference_grouped_path_count', 1)
        if batch_order_preserved:
            self._profile_add('recurrent_inference_order_preserved_group_count', 1)
            value_chunks, reward_chunks, policy_chunks, latent_chunks = [], [], [], []
        else:
            values, rewards, policy_logits, latent_states = (
                [None] * batch_size,
                [None] * batch_size,
                [None] * batch_size,
                [None] * batch_size,
            )

        for current_member_id, indices in groups:
            index_start = self._profile_start(device)
            member_latent_state = self._select_rows(latent_state, indices, batch_size)
            member_action = self._select_rows(action, indices, batch_size)
            self._profile_add('recurrent_inference_index_s', self._profile_elapsed(index_start, device))

            member_start = self._profile_start(device)
            output = self._member_recurrent_with_trainable_dynamics_normalization(
                self.members[current_member_id],
                member_latent_state,
                member_action,
            )
            self._profile_add('recurrent_inference_member_forward_s', self._profile_elapsed(member_start, device))
            self._profile_add('recurrent_inference_member_forward_count', 1)
            output = self._apply_value_prior(output, current_member_id, device)

            if batch_order_preserved:
                value_chunks.append(output.value)
                policy_chunks.append(output.policy_logits)
                latent_chunks.append(output.latent_state)
                self._extend_reward_chunks(reward_chunks, output.reward)
            else:
                stitch_start = self._profile_start(device)
                for local_idx, global_idx in enumerate(indices):
                    values[global_idx] = output.value[local_idx:local_idx + 1]
                    policy_logits[global_idx] = output.policy_logits[local_idx:local_idx + 1]
                    latent_states[global_idx] = output.latent_state[local_idx:local_idx + 1]
                    if torch.is_tensor(output.reward):
                        rewards[global_idx] = output.reward[local_idx:local_idx + 1]
                    else:
                        rewards[global_idx] = output.reward[local_idx]
                self._profile_add('recurrent_inference_stitch_s', self._profile_elapsed(stitch_start, device))

        concat_start = self._profile_start(device)
        if batch_order_preserved:
            value = torch.cat(value_chunks, dim=0)
            policy_logit = torch.cat(policy_chunks, dim=0)
            next_latent_state = torch.cat(latent_chunks, dim=0)
            if torch.is_tensor(reward_chunks[0]):
                reward = torch.cat(reward_chunks, dim=0)
            else:
                reward = reward_chunks
        else:
            value = torch.cat(values, dim=0)
            policy_logit = torch.cat(policy_logits, dim=0)
            next_latent_state = torch.cat(latent_states, dim=0)
            if torch.is_tensor(rewards[0]):
                reward = torch.cat(rewards, dim=0)
            else:
                reward = rewards
        self._profile_add('recurrent_inference_concat_s', self._profile_elapsed(concat_start, device))
        self._profile_add('recurrent_inference_total_s', self._profile_elapsed(total_start, device))
        return MZNetworkOutput(value, reward, policy_logit, next_latent_state)

    def initial_inference(
            self, obs: torch.Tensor, member_id: Optional[Union[int, Sequence[int], torch.Tensor]] = None
    ) -> MZNetworkOutput:
        if self.dynamics_encoder_latent_normalization != 'none' or self.encoder_prior_networks is not None:
            return self._run_batched_member_initial_with_encoder_normalization(obs, member_id=member_id)
        return self._run_batched_member_inference('initial_inference', obs, member_id=member_id)

    def recurrent_inference(
            self,
            latent_state: torch.Tensor,
            action: torch.Tensor,
            member_id: Optional[Union[int, Sequence[int], torch.Tensor]] = None
    ) -> MZNetworkOutput:
        if self.dynamics_prior_networks is not None:
            return self._run_batched_member_recurrent_with_dynamics_prior(latent_state, action, member_id=member_id)
        if self.dynamics_trainable_latent_normalization != 'none':
            return self._run_batched_member_recurrent_with_trainable_dynamics_normalization(
                latent_state, action, member_id=member_id
            )
        return self._run_batched_member_inference('recurrent_inference', latent_state, action, member_id=member_id)

    def _run_batched_member_tensor_method(
            self,
            fn_name: str,
            batch: torch.Tensor,
            member_id: Optional[Union[int, Sequence[int], torch.Tensor]] = None,
            fallback_dim: int = 1,
    ) -> torch.Tensor:
        device = batch.device
        batch_size = batch.shape[0]
        member_ids = self._resolve_member_ids(batch_size, member_id)
        groups = self._group_member_indices(member_ids)

        if len(groups) == 1:
            current_member_id = groups[0][0]
            member = self.members[current_member_id]
            if hasattr(member, fn_name):
                output = getattr(member, fn_name)(batch)
            else:
                output = batch.new_zeros((batch_size, fallback_dim))
            if output.dim() == 1:
                output = output.unsqueeze(-1)
            return output

        batch_order_preserved = self._groups_preserve_batch_order(groups, batch_size)
        if batch_order_preserved:
            output_chunks = []
        else:
            outputs = [None] * batch_size

        for current_member_id, indices in groups:
            member_batch = self._select_rows(batch, indices, batch_size)
            member = self.members[current_member_id]
            if hasattr(member, fn_name):
                output = getattr(member, fn_name)(member_batch)
            else:
                output = member_batch.new_zeros((len(indices), fallback_dim))
            if output.dim() == 1:
                output = output.unsqueeze(-1)

            if batch_order_preserved:
                output_chunks.append(output)
            else:
                for local_idx, global_idx in enumerate(indices):
                    outputs[global_idx] = output[local_idx:local_idx + 1]

        return torch.cat(output_chunks if batch_order_preserved else outputs, dim=0)

    def intrinsic_reward_from_latent(
            self,
            latent_state: torch.Tensor,
            member_id: Optional[Union[int, Sequence[int], torch.Tensor]] = None
    ) -> torch.Tensor:
        return self._run_batched_member_tensor_method(
            'intrinsic_reward_from_latent', latent_state, member_id=member_id
        )

    def intrinsic_reward_prediction_from_latent(
            self,
            latent_state: torch.Tensor,
            member_id: Optional[Union[int, Sequence[int], torch.Tensor]] = None
    ) -> torch.Tensor:
        return self._run_batched_member_tensor_method(
            'intrinsic_reward_prediction_from_latent', latent_state, member_id=member_id
        )

    def transform_intrinsic_reward_target(self, reward: torch.Tensor) -> torch.Tensor:
        transform = getattr(self.members[0], 'transform_intrinsic_reward_target', None)
        return transform(reward) if callable(transform) else reward

    def inverse_transform_intrinsic_reward_prediction(
            self, prediction: torch.Tensor
    ) -> torch.Tensor:
        inverse = getattr(
            self.members[0], 'inverse_transform_intrinsic_reward_prediction', None
        )
        return inverse(prediction) if callable(inverse) else prediction

    def intrinsic_value_from_latent(
            self,
            latent_state: torch.Tensor,
            member_id: Optional[Union[int, Sequence[int], torch.Tensor]] = None
    ) -> torch.Tensor:
        return self._run_batched_member_tensor_method(
            'intrinsic_value_from_latent', latent_state, member_id=member_id
        )

    def episodic_history_parameters(self) -> List[nn.Parameter]:
        if self.episodic_history_models is None:
            return []
        return [
            parameter
            for history_model in self.episodic_history_models
            for parameter in history_model.parameters()
            if parameter.requires_grad
        ]

    def initial_episodic_history(
            self,
            batch_size: int,
            member_id: int = 0,
            device: Optional[torch.device] = None,
            dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if self.episodic_history_models is None:
            raise RuntimeError("The episodic-history model is disabled.")
        history_model = self.episodic_history_models[self._resolve_member_id(member_id)]
        if device is None:
            device = next(history_model.parameters()).device
        return history_model.initial_history(batch_size, device=device, dtype=dtype)

    def episodic_history_step(
            self,
            current_latent: torch.Tensor,
            action: torch.Tensor,
            next_latent: torch.Tensor,
            realized_reward: torch.Tensor,
            history: torch.Tensor,
            member_id: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.episodic_history_models is None:
            raise RuntimeError("The episodic-history model is disabled.")
        history_model = self.episodic_history_models[self._resolve_member_id(member_id)]
        return history_model(
            current_latent,
            action,
            next_latent,
            realized_reward,
            history,
        )

    def episodic_history_reward(
            self,
            current_latent: torch.Tensor,
            action: torch.Tensor,
            next_latent: torch.Tensor,
            history: torch.Tensor,
            member_id: int = 0,
    ) -> torch.Tensor:
        if self.episodic_history_models is None:
            raise RuntimeError("The episodic-history model is disabled.")
        history_model = self.episodic_history_models[self._resolve_member_id(member_id)]
        return history_model.predict_reward(current_latent, action, next_latent, history)

    def advance_episodic_history(
            self,
            current_latent: torch.Tensor,
            action: torch.Tensor,
            next_latent: torch.Tensor,
            realized_reward: torch.Tensor,
            history: torch.Tensor,
            member_id: int = 0,
    ) -> torch.Tensor:
        if self.episodic_history_models is None:
            raise RuntimeError("The episodic-history model is disabled.")
        history_model = self.episodic_history_models[self._resolve_member_id(member_id)]
        return history_model.advance_history(
            current_latent,
            action,
            next_latent,
            realized_reward,
            history,
        )

    def episodic_history_sequence(
            self,
            current_latents: torch.Tensor,
            actions: torch.Tensor,
            predicted_next_latents: torch.Tensor,
            realized_next_latents: torch.Tensor,
            realized_rewards: torch.Tensor,
            initial_history: torch.Tensor,
            member_id: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.episodic_history_models is None:
            raise RuntimeError("The episodic-history model is disabled.")
        history_model = self.episodic_history_models[self._resolve_member_id(member_id)]
        return history_model.forward_sequence(
            current_latents,
            actions,
            predicted_next_latents,
            realized_next_latents,
            realized_rewards,
            initial_history,
        )

    def latent_novelty_parameters(self) -> List[nn.Parameter]:
        if self.latent_novelty_heads is None:
            return []
        return [
            param
            for head in self.latent_novelty_heads
            for param in head.parameters()
            if param.requires_grad
        ]

    def _aligned_latent_embedding_single_member(
            self,
            latent_state: torch.Tensor,
            member_id: int,
            with_prior: bool = True,
            normalize: bool = True,
            detach_latent: bool = True,
    ) -> torch.Tensor:
        if self.latent_novelty_heads is None:
            return latent_state.new_zeros((latent_state.shape[0], 0))
        latent_input = latent_state.detach() if detach_latent else latent_state
        embedding = self.latent_novelty_heads[member_id](latent_input)
        if (
                with_prior
                and self.latent_novelty_prior_networks is not None
                and self.latent_novelty_prior_scale > 0
        ):
            prior_embedding = self.latent_novelty_prior_networks[member_id](latent_input)
            embedding = embedding + float(self.latent_novelty_prior_scale) * prior_embedding
        if normalize and embedding.shape[-1] > 0:
            embedding = F.normalize(embedding, dim=-1)
        return embedding

    def aligned_latent_embedding(
            self,
            latent_state: torch.Tensor,
            member_id: Optional[Union[int, Sequence[int], torch.Tensor]] = None,
            with_prior: bool = True,
            normalize: bool = True,
            detach_latent: bool = True,
    ) -> torch.Tensor:
        """
        Project member-specific latents into a shared diagnostic feature space.

        ``detach_latent=True`` is the intended diagnostic mode: the auxiliary
        alignment heads learn, but MuZero's representation/dynamics do not get
        gradients from this loss.
        """
        batch_size = latent_state.shape[0]
        member_ids = self._resolve_member_ids(batch_size, member_id)
        groups = self._group_member_indices(member_ids)

        if len(groups) == 1:
            return self._aligned_latent_embedding_single_member(
                latent_state, groups[0][0], with_prior=with_prior, normalize=normalize, detach_latent=detach_latent
            )

        batch_order_preserved = self._groups_preserve_batch_order(groups, batch_size)
        if batch_order_preserved:
            embedding_chunks = []
        else:
            embeddings = [None] * batch_size

        for current_member_id, indices in groups:
            member_latent_state = self._select_rows(latent_state, indices, batch_size)
            output = self._aligned_latent_embedding_single_member(
                member_latent_state,
                current_member_id,
                with_prior=with_prior,
                normalize=normalize,
                detach_latent=detach_latent,
            )
            if batch_order_preserved:
                embedding_chunks.append(output)
            else:
                for local_idx, global_idx in enumerate(indices):
                    embeddings[global_idx] = output[local_idx:local_idx + 1]

        return torch.cat(embedding_chunks if batch_order_preserved else embeddings, dim=0)

    def episodic_inverse_embedding(self, observation: torch.Tensor) -> torch.Tensor:
        """Encode a real observation in the shared controllable-state space."""
        if self.episodic_inverse_encoder is None:
            raise RuntimeError('The episodic inverse-dynamics encoder is disabled.')
        return self.episodic_inverse_encoder(observation)

    def episodic_inverse_action_logits(
            self,
            current_embedding: torch.Tensor,
            next_embedding: torch.Tensor,
    ) -> torch.Tensor:
        if self.episodic_inverse_action_head is None:
            raise RuntimeError('The episodic inverse-dynamics action head is disabled.')
        if current_embedding.shape != next_embedding.shape:
            raise ValueError(
                'Current and next episodic embeddings must have equal shapes, got '
                f'{tuple(current_embedding.shape)} and {tuple(next_embedding.shape)}.'
            )
        return self.episodic_inverse_action_head(
            torch.cat([current_embedding, next_embedding], dim=-1)
        )

    def episodic_inverse_bridge(
            self,
            latent_state: torch.Tensor,
            member_id: Optional[Union[int, Sequence[int], torch.Tensor]] = None,
            detach_latent: bool = False,
    ) -> torch.Tensor:
        """Predict the shared controllable-state embedding from imagined latents."""
        if self.episodic_inverse_bridges is None:
            raise RuntimeError('The episodic inverse-dynamics bridges are disabled.')
        batch_size = int(latent_state.shape[0])
        member_ids = self._resolve_member_ids(batch_size, member_id)
        groups = self._group_member_indices(member_ids)
        bridge_input = latent_state.detach() if detach_latent else latent_state

        if len(groups) == 1:
            return self.episodic_inverse_bridges[groups[0][0]](bridge_input)

        batch_order_preserved = self._groups_preserve_batch_order(groups, batch_size)
        if batch_order_preserved:
            embedding_chunks = []
        else:
            embeddings = [None] * batch_size
        for current_member_id, indices in groups:
            member_latent = self._select_rows(bridge_input, indices, batch_size)
            output = self.episodic_inverse_bridges[current_member_id](member_latent)
            if batch_order_preserved:
                embedding_chunks.append(output)
            else:
                for local_idx, global_idx in enumerate(indices):
                    embeddings[global_idx] = output[local_idx:local_idx + 1]
        return torch.cat(embedding_chunks if batch_order_preserved else embeddings, dim=0)

    def project(
            self,
            latent_state: torch.Tensor,
            with_grad: bool = True,
            member_id: Optional[Union[int, Sequence[int], torch.Tensor]] = None
    ) -> torch.Tensor:
        device = latent_state.device
        total_start = self._profile_start(device)
        resolve_start = self._profile_start(device)
        batch_size = latent_state.shape[0]
        member_ids = self._resolve_member_ids(batch_size, member_id)
        self._profile_add('project_resolve_s', self._profile_elapsed(resolve_start, device))
        self._profile_add('project_call_count', 1)
        self._profile_add('project_batch_rows', batch_size)
        if with_grad:
            self._profile_add('project_with_grad_call_count', 1)
        else:
            self._profile_add('project_no_grad_call_count', 1)

        groups = self._group_member_indices(member_ids)
        if len(groups) == 1:
            self._profile_add('project_fast_path_count', 1)
            member_start = self._profile_start(device)
            output = self.members[groups[0][0]].project(latent_state, with_grad=with_grad)
            self._profile_add('project_member_forward_s', self._profile_elapsed(member_start, device))
            self._profile_add('project_member_forward_count', 1)
            self._profile_add('project_total_s', self._profile_elapsed(total_start, device))
            return output

        batch_order_preserved = self._groups_preserve_batch_order(groups, batch_size)
        self._profile_add('project_grouped_path_count', 1)
        if batch_order_preserved:
            self._profile_add('project_order_preserved_group_count', 1)
            projection_chunks = []
        else:
            projections = [None] * batch_size

        for current_member_id, indices in groups:
            index_start = self._profile_start(device)
            member_latent_state = self._select_rows(latent_state, indices, batch_size)
            self._profile_add('project_index_s', self._profile_elapsed(index_start, device))
            member_start = self._profile_start(device)
            output = self.members[current_member_id].project(member_latent_state, with_grad=with_grad)
            self._profile_add('project_member_forward_s', self._profile_elapsed(member_start, device))
            self._profile_add('project_member_forward_count', 1)
            if batch_order_preserved:
                projection_chunks.append(output)
            else:
                stitch_start = self._profile_start(device)
                for local_idx, global_idx in enumerate(indices):
                    projections[global_idx] = output[local_idx:local_idx + 1]
                self._profile_add('project_stitch_s', self._profile_elapsed(stitch_start, device))
        concat_start = self._profile_start(device)
        output = torch.cat(projection_chunks if batch_order_preserved else projections, dim=0)
        self._profile_add('project_concat_s', self._profile_elapsed(concat_start, device))
        self._profile_add('project_total_s', self._profile_elapsed(total_start, device))
        return output

    def get_params_mean(self) -> float:
        params = [p.detach().mean() for p in self.parameters() if p.numel() > 0]
        if len(params) == 0:
            return 0.0
        return torch.stack(params).mean().item()


@MODEL_REGISTRY.register('EnsembleMuZeroModel')
class EnsembleMuZeroModel(_EnsembleMuZeroBase):
    """
    MuZero convolutional whole-model ensemble.

    With no explicit member id, inference routes to member 0 so this remains
    compatible with vanilla MuZero policy logic.
    """

    member_cls = MuZeroModel


@MODEL_REGISTRY.register('EnsembleMuZeroModelMLP')
class EnsembleMuZeroModelMLP(_EnsembleMuZeroBase):
    """
    MuZero MLP whole-model ensemble.

    With no explicit member id, inference routes to member 0 so this remains
    compatible with vanilla MuZero policy logic.
    """

    member_cls = MuZeroModelMLP
