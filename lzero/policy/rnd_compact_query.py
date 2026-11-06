from typing import Callable, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from lzero.policy.rnd_landscape_query import (
    MountainCarRNDLandscapeQuery,
    RNDLandscapeQueryContext,
)


class CompactImageRNDLandscapeQuery(MountainCarRNDLandscapeQuery):
    """Query image RND through a learned, bounded observation autoencoder space."""

    def __init__(
            self,
            observation_shape: Sequence[int],
            latent_dim: int,
            query_code_dim: int = 32,
            reconstruction_loss_weight: float = 1.0,
            **kwargs,
    ) -> None:
        observation_shape = tuple(int(size) for size in observation_shape)
        if len(observation_shape) != 3 or any(size <= 0 for size in observation_shape):
            raise ValueError(
                "observation_shape must be positive [channels, height, width], got "
                f"{observation_shape}."
            )
        if observation_shape[1:] != (84, 84):
            raise ValueError(
                "Compact image queries currently support 84x84 observations, got "
                f"{observation_shape[1:]}."
            )
        self.observation_shape = observation_shape
        self.query_code_dim = int(query_code_dim)
        if self.query_code_dim <= 0:
            raise ValueError(f"query_code_dim must be positive, got {query_code_dim}.")
        self.reconstruction_loss_weight = float(reconstruction_loss_weight)
        if self.reconstruction_loss_weight < 0.0:
            raise ValueError(
                "reconstruction_loss_weight must be non-negative, got "
                f"{reconstruction_loss_weight}."
            )

        super().__init__(latent_dim=latent_dim, **kwargs)

        channels = self.observation_shape[0]
        self.observation_encoder_conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
        )
        self.observation_encoder_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 6 * 6, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.query_code_dim),
            nn.Tanh(),
        )
        self.observation_decoder_input = nn.Sequential(
            nn.Linear(self.query_code_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 64 * 6 * 6),
            nn.SiLU(),
        )
        self.observation_decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(32, channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

        self.query_pair_encoder = nn.Sequential(
            nn.Linear(self.query_code_dim + 1, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.query_policy = nn.Sequential(
            nn.Linear(self.query_code_dim + 2 * self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.query_code_dim),
            nn.Tanh(),
        )

    def encode_observations(self, observations: torch.Tensor) -> torch.Tensor:
        observations = observations.float()
        if tuple(observations.shape[-3:]) != self.observation_shape:
            raise ValueError(
                f"Expected observations ending in {self.observation_shape}, got "
                f"{tuple(observations.shape)}."
            )
        leading_shape = observations.shape[:-3]
        flat = observations.reshape(-1, *self.observation_shape)
        codes = self.observation_encoder_head(self.observation_encoder_conv(flat))
        return codes.reshape(*leading_shape, self.query_code_dim)

    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        if codes.shape[-1] != self.query_code_dim:
            raise ValueError(
                f"Expected code dimension {self.query_code_dim}, got {codes.shape[-1]}."
            )
        leading_shape = codes.shape[:-1]
        decoded = self.observation_decoder_input(codes.float()).reshape(-1, 64, 6, 6)
        decoded = self.observation_decoder_conv(decoded)
        return decoded.reshape(*leading_shape, *self.observation_shape)

    def query(
            self,
            root_states: torch.Tensor,
            error_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> RNDLandscapeQueryContext:
        root_states = root_states.reshape(-1, *self.observation_shape).float()
        root_codes = self.encode_observations(root_states)
        summary = root_codes.new_zeros((root_codes.shape[0], self.hidden_dim))
        query_codes = []
        query_states = []
        query_log_errors = []
        query_tokens = []

        for step in range(self.query_count):
            step_ids = torch.full(
                (root_codes.shape[0],), step, dtype=torch.long, device=root_codes.device
            )
            step_embedding = self.query_step_embedding(step_ids)
            code = self.query_policy(
                torch.cat([root_codes, summary, step_embedding], dim=-1)
            )
            decoded = self.decode_codes(code)
            exact_error = error_fn(decoded).reshape(-1, 1)
            log_error = self._log_errors(exact_error)
            token = self.query_pair_encoder(torch.cat([code, log_error], dim=-1))

            query_codes.append(code)
            query_states.append(decoded)
            query_log_errors.append(log_error)
            query_tokens.append(token)
            summary = torch.stack(query_tokens, dim=1).mean(dim=1)

        return RNDLandscapeQueryContext(
            root_states=root_states,
            query_states=torch.stack(query_states, dim=1),
            query_log_errors=torch.stack(query_log_errors, dim=1),
            root_query_codes=root_codes,
            query_codes=torch.stack(query_codes, dim=1),
        )

    def _query_geometry(
            self,
            context: RNDLandscapeQueryContext,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if context.root_query_codes is None or context.query_codes is None:
            raise ValueError("Compact image query context is missing compact codes.")
        scale = max(float(self.query_code_dim) ** 0.5, 1.0)
        return context.root_query_codes / scale, context.query_codes / scale

    def loss_on_observations(
            self,
            root_states: torch.Tensor,
            imagined_latents: torch.Tensor,
            target_errors: torch.Tensor,
            error_fn: Callable[[torch.Tensor], torch.Tensor],
            query_latent_fn: Callable[[torch.Tensor], torch.Tensor],
            target_mask: Optional[torch.Tensor] = None,
            sample_weights: Optional[torch.Tensor] = None,
            diversity_weight: float = 0.0,
            centered_prediction_weight: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        context = self.query(root_states, error_fn)
        context.query_latents = query_latent_fn(context.query_states)
        predicted_target = self.predict_latent_member_target(context, imagined_latents)
        with torch.no_grad():
            target_error = target_errors.float().unsqueeze(-1).expand_as(
                predicted_target
            )
        prediction_loss, terms = self._loss_from_predictions(
            context,
            predicted_target,
            target_error,
            diversity_weight,
            centered_prediction_weight,
            target_mask=target_mask,
            sample_weights=sample_weights,
        )

        reconstructed = self.decode_codes(context.root_query_codes)
        per_root_reconstruction = F.smooth_l1_loss(
            reconstructed, root_states.float(), reduction="none"
        ).flatten(1).mean(dim=1)
        if sample_weights is None:
            reconstruction_loss = per_root_reconstruction.mean()
        else:
            weights = sample_weights.to(
                device=per_root_reconstruction.device,
                dtype=per_root_reconstruction.dtype,
            ).reshape(-1)
            reconstruction_loss = (
                per_root_reconstruction * weights
            ).sum() / weights.sum().clamp_min(1e-6)
        total_loss = (
            prediction_loss
            + self.reconstruction_loss_weight * reconstruction_loss
        )
        terms.update(
            {
                "prediction_objective": prediction_loss.detach(),
                "reconstruction_loss": reconstruction_loss.detach(),
                "reconstruction_weight": reconstruction_loss.new_tensor(
                    self.reconstruction_loss_weight
                ),
                "reconstruction_mae": (
                    reconstructed - root_states.float()
                ).abs().mean().detach(),
                "root_code_std": context.root_query_codes.std(
                    dim=0, unbiased=False
                ).mean().detach(),
                "query_code_std": context.query_codes.std(
                    dim=(0, 1), unbiased=False
                ).mean().detach(),
                "decoded_query_min": context.query_states.amin().detach(),
                "decoded_query_max": context.query_states.amax().detach(),
            }
        )
        return total_loss, terms
