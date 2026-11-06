import math
from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn


class SharedRNDAuxiliaryDecoder(nn.Module):
    """Decode current RND error from every ensemble member's imagined latent."""

    def __init__(
            self,
            latent_shape: Sequence[int],
            hidden_dim: int = 64,
            target_transform: str = 'log',
            target_log_reference: float = 1.0,
            error_epsilon: float = 1e-10,
            max_raw_error: float | None = None,
            normalize_latent: bool = True,
    ) -> None:
        super().__init__()
        self.latent_shape = tuple(int(size) for size in latent_shape)
        if not self.latent_shape or any(size <= 0 for size in self.latent_shape):
            raise ValueError(f'latent_shape must be positive, got {latent_shape!r}.')
        self.latent_dim = math.prod(self.latent_shape)
        self.hidden_dim = max(8, int(hidden_dim))
        self.target_transform = str(target_transform).lower()
        if self.target_transform not in {'log', 'raw'}:
            raise ValueError(
                "target_transform must be 'log' or 'raw', got "
                f'{target_transform!r}.'
            )
        self.target_log_reference = float(target_log_reference)
        self.error_epsilon = float(error_epsilon)
        self.max_raw_error = (
            None if max_raw_error is None else float(max_raw_error)
        )
        self.normalize_latent = bool(normalize_latent)
        if self.target_log_reference <= 0.0:
            raise ValueError('target_log_reference must be positive.')
        if self.error_epsilon <= 0.0:
            raise ValueError('error_epsilon must be positive.')
        if self.max_raw_error is not None and self.max_raw_error <= 0.0:
            raise ValueError('max_raw_error must be positive when set.')

        self.latent_encoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def transform_target(self, errors: torch.Tensor) -> torch.Tensor:
        errors = errors.float().clamp_min(self.error_epsilon)
        if self.target_transform == 'log':
            return torch.log(errors / self.target_log_reference)
        return errors

    def inverse_prediction(self, predictions: torch.Tensor) -> torch.Tensor:
        if self.target_transform == 'raw':
            return predictions.clamp_min(0.0)
        minimum = math.log(self.error_epsilon / self.target_log_reference)
        maximum = (
            math.log(self.max_raw_error / self.target_log_reference)
            if self.max_raw_error is not None else None
        )
        predictions = predictions.clamp(min=minimum, max=maximum)
        return self.target_log_reference * torch.exp(predictions)

    def forward(self, imagined_latents: torch.Tensor) -> torch.Tensor:
        """Return target-space predictions for ``[..., *latent_shape]`` inputs."""
        latent_rank = len(self.latent_shape)
        if tuple(imagined_latents.shape[-latent_rank:]) != self.latent_shape:
            raise ValueError(
                f'Expected trailing latent shape {self.latent_shape}, got '
                f'{tuple(imagined_latents.shape)}.'
            )
        leading_shape = imagined_latents.shape[:-latent_rank]
        flat = imagined_latents.reshape(-1, self.latent_dim).float()
        if self.normalize_latent:
            flat = torch.nn.functional.layer_norm(flat, (self.latent_dim,))
        prediction = self.decoder(self.latent_encoder(flat)).squeeze(-1)
        return prediction.reshape(*leading_shape)

    @staticmethod
    def _weighted_correlation(
            prediction: torch.Tensor,
            target: torch.Tensor,
            weights: torch.Tensor,
    ) -> torch.Tensor:
        weight_sum = weights.sum().clamp_min(1e-6)
        prediction_mean = (prediction * weights).sum() / weight_sum
        target_mean = (target * weights).sum() / weight_sum
        prediction_centered = prediction - prediction_mean
        target_centered = target - target_mean
        covariance = (
            prediction_centered * target_centered * weights
        ).sum() / weight_sum
        prediction_std = (
            (prediction_centered.square() * weights).sum() / weight_sum
        ).sqrt()
        target_std = (
            (target_centered.square() * weights).sum() / weight_sum
        ).sqrt()
        return covariance / (prediction_std * target_std).clamp_min(1e-12)

    def loss(
            self,
            imagined_latents: torch.Tensor,
            target_errors: torch.Tensor,
            target_mask: torch.Tensor,
            sample_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Train one shared decoder on ``[batch, depth, member, ...]`` latents."""
        predictions = self(imagined_latents)
        if predictions.dim() != 3:
            raise ValueError(
                'Expected [batch, depth, member] predictions, got '
                f'{tuple(predictions.shape)}.'
            )
        expanded_targets = target_errors.float().unsqueeze(-1).expand_as(predictions)
        transformed_targets = self.transform_target(expanded_targets)
        weights = (
            target_mask.float().unsqueeze(-1)
            * sample_weights.float().view(-1, 1, 1)
        ).expand_as(predictions)
        weight_sum = weights.sum().clamp_min(1e-6)
        elementwise = torch.nn.functional.smooth_l1_loss(
            predictions, transformed_targets, reduction='none'
        )
        loss = (elementwise * weights).sum() / weight_sum

        with torch.no_grad():
            raw_predictions = self.inverse_prediction(predictions)
            valid = weights > 0.0
            prediction_values = predictions[valid]
            transformed_values = transformed_targets[valid]
            raw_prediction_values = raw_predictions[valid]
            raw_target_values = expanded_targets[valid]
            valid_weights = weights[valid]
            terms = {
                'prediction_loss': loss.detach(),
                'prediction_mae_target_space': (
                    (prediction_values - transformed_values).abs() * valid_weights
                ).sum().div(valid_weights.sum().clamp_min(1e-6)),
                'prediction_mae_raw': (
                    (raw_prediction_values - raw_target_values).abs() * valid_weights
                ).sum().div(valid_weights.sum().clamp_min(1e-6)),
                'prediction_correlation_raw': self._weighted_correlation(
                    raw_prediction_values, raw_target_values, valid_weights
                ),
                'prediction_target_mean_raw': (
                    raw_target_values * valid_weights
                ).sum().div(valid_weights.sum().clamp_min(1e-6)),
                'prediction_target_std_raw': raw_target_values.std(unbiased=False),
                'valid_target_fraction': target_mask.float().mean(),
            }
            for depth in range(predictions.shape[1]):
                depth_weights = weights[:, depth]
                depth_weight_sum = depth_weights.sum().clamp_min(1e-6)
                depth_raw_prediction = raw_predictions[:, depth]
                depth_raw_target = expanded_targets[:, depth]
                terms[f'depth_{depth + 1}_prediction_mae_raw'] = (
                    (depth_raw_prediction - depth_raw_target).abs() * depth_weights
                ).sum().div(depth_weight_sum)
                terms[f'depth_{depth + 1}_prediction_correlation_raw'] = (
                    self._weighted_correlation(
                        depth_raw_prediction.reshape(-1),
                        depth_raw_target.reshape(-1),
                        depth_weights.reshape(-1),
                    )
                )
        return loss, terms
