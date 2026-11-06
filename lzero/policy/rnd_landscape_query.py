import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from lzero.model.common import MLP_V2


@dataclass
class RNDLandscapeQueryContext:
    """Query results and their MuZero encodings for one batch of roots."""

    root_states: torch.Tensor
    query_states: torch.Tensor
    query_log_errors: torch.Tensor
    query_latents: Optional[torch.Tensor] = None
    root_query_codes: Optional[torch.Tensor] = None
    query_codes: Optional[torch.Tensor] = None

    def select(self, indices: torch.Tensor) -> "RNDLandscapeQueryContext":
        return RNDLandscapeQueryContext(
            root_states=self.root_states.index_select(0, indices),
            query_states=self.query_states.index_select(0, indices),
            query_log_errors=self.query_log_errors.index_select(0, indices),
            query_latents=(
                self.query_latents.index_select(0, indices)
                if self.query_latents is not None else None
            ),
            root_query_codes=(
                self.root_query_codes.index_select(0, indices)
                if self.root_query_codes is not None else None
            ),
            query_codes=(
                self.query_codes.index_select(0, indices)
                if self.query_codes is not None else None
            ),
        )


class MountainCarRNDLandscapeQuery(nn.Module):
    """Query RND in observation space and read it from imagined MuZero latents.

    Query locations are selected sequentially in physical observation space.
    The reader receives MuZero encodings of those locations, their measured RND
    errors, and an imagined child latent. Environment transition equations are
    never used.
    """

    def __init__(
            self,
            latent_dim: int,
            query_count: int = 16,
            hidden_dim: int = 64,
            reader_use_context: bool = True,
            ensemble_size: int = 1,
            member_specific_reader: bool = False,
            reader_architecture: str = "landscape",
            reader_hidden_channels: Sequence[int] = (32,),
            reader_norm_type: Optional[str] = "BN",
            reader_last_linear_layer_init_zero: bool = False,
            target_transform: str = "log",
            target_log_reference: float = 1.0,
            normalized_raw_clip: float = 10.0,
            max_raw_error: Optional[float] = None,
            position_bounds: Tuple[float, float] = (-1.2, 0.6),
            velocity_bounds: Tuple[float, float] = (-0.07, 0.07),
            error_epsilon: float = 1e-10,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        if self.latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}.")
        self.query_count = max(1, int(query_count))
        self.hidden_dim = max(8, int(hidden_dim))
        self.reader_use_context = bool(reader_use_context)
        self.ensemble_size = max(1, int(ensemble_size))
        self.member_specific_reader = bool(member_specific_reader)
        self.reader_architecture = str(reader_architecture).lower()
        if self.reader_architecture not in {'landscape', 'ordinary'}:
            raise ValueError(
                "reader_architecture must be 'landscape' or 'ordinary', got "
                f"{reader_architecture!r}."
            )
        if self.reader_architecture == 'ordinary' and self.reader_use_context:
            raise ValueError(
                "The ordinary-shaped auxiliary decoder is latent-only; set "
                "reader_use_context=False."
            )
        self.reader_hidden_channels = tuple(int(width) for width in reader_hidden_channels)
        if not self.reader_hidden_channels or any(width <= 0 for width in self.reader_hidden_channels):
            raise ValueError(
                "reader_hidden_channels must contain positive widths, got "
                f"{reader_hidden_channels!r}."
            )
        self.reader_norm_type = reader_norm_type
        self.reader_last_linear_layer_init_zero = bool(
            reader_last_linear_layer_init_zero
        )
        self.target_transform = str(target_transform).lower()
        if self.target_transform not in {'log', 'raw', 'normalized_raw'}:
            raise ValueError(
                "target_transform must be 'log', 'raw', or 'normalized_raw', got "
                f"{target_transform!r}."
            )
        self.normalized_raw_clip = float(normalized_raw_clip)
        if self.normalized_raw_clip <= 0.0:
            raise ValueError(
                "normalized_raw_clip must be positive, got "
                f"{normalized_raw_clip}."
            )
        self.target_log_reference = float(target_log_reference)
        if not math.isfinite(self.target_log_reference) or self.target_log_reference <= 0.0:
            raise ValueError(
                "target_log_reference must be finite and positive, got "
                f"{target_log_reference}."
            )
        self.max_raw_error = (
            None if max_raw_error is None else float(max_raw_error)
        )
        if self.max_raw_error is not None and (
                not math.isfinite(self.max_raw_error) or self.max_raw_error <= 0.0
        ):
            raise ValueError(
                f"max_raw_error must be finite and positive, got {max_raw_error}."
            )
        self.error_epsilon = float(error_epsilon)
        if not math.isfinite(self.error_epsilon) or self.error_epsilon <= 0.0:
            raise ValueError(
                f"error_epsilon must be finite and positive, got {error_epsilon}."
            )

        low = torch.tensor([position_bounds[0], velocity_bounds[0]], dtype=torch.float32)
        high = torch.tensor([position_bounds[1], velocity_bounds[1]], dtype=torch.float32)
        if not bool(torch.all(high > low)):
            raise ValueError("RND landscape query bounds must have high > low.")
        self.register_buffer("state_low", low)
        self.register_buffer("state_high", high)
        self.register_buffer("target_scale", torch.ones((), dtype=torch.float32))

        self.query_pair_encoder = nn.Sequential(
            nn.Linear(3, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.query_step_embedding = nn.Embedding(self.query_count, self.hidden_dim)
        self.query_policy = nn.Sequential(
            nn.Linear(2 + 2 * self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 2),
            nn.Tanh(),
        )

        def make_latent_encoder() -> nn.Module:
            return nn.Sequential(
                nn.Linear(self.latent_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
            )

        def make_latent_pair_encoder() -> nn.Module:
            return nn.Sequential(
                nn.Linear(3 * self.hidden_dim + 1, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
            )

        def make_latent_reader() -> nn.Module:
            if self.reader_architecture == 'ordinary':
                return MLP_V2(
                    in_channels=self.latent_dim,
                    hidden_channels=list(self.reader_hidden_channels),
                    out_channels=1,
                    activation=nn.ReLU(inplace=True),
                    norm_type=self.reader_norm_type,
                    output_activation=False,
                    output_norm=False,
                    last_linear_layer_init_zero=(
                        self.reader_last_linear_layer_init_zero
                    ),
                )
            return nn.Sequential(
                nn.Linear(2 * self.hidden_dim + 1, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, 1),
            )

        if self.member_specific_reader:
            self.latent_encoders = nn.ModuleList(
                [make_latent_encoder() for _ in range(self.ensemble_size)]
            )
            self.latent_pair_encoders = nn.ModuleList(
                [make_latent_pair_encoder() for _ in range(self.ensemble_size)]
            )
            self.latent_pair_scores = nn.ModuleList(
                [nn.Linear(self.hidden_dim, 1) for _ in range(self.ensemble_size)]
            )
            self.latent_readers = nn.ModuleList(
                [make_latent_reader() for _ in range(self.ensemble_size)]
            )
            self.latent_encoder = None
            self.latent_pair_encoder = None
            self.latent_pair_score = None
            self.latent_reader = None
        else:
            # Preserve the original module names and initialization order for old checkpoints.
            self.latent_encoder = make_latent_encoder()
            self.latent_pair_encoder = make_latent_pair_encoder()
            self.latent_pair_score = nn.Linear(self.hidden_dim, 1)
            self.latent_reader = make_latent_reader()
            self.latent_encoders = None
            self.latent_pair_encoders = None
            self.latent_pair_scores = None
            self.latent_readers = None

    def normalize_states(self, states: torch.Tensor) -> torch.Tensor:
        states = states.float()
        return 2.0 * (states - self.state_low) / (self.state_high - self.state_low) - 1.0

    def denormalize_states(self, normalized_states: torch.Tensor) -> torch.Tensor:
        return self.state_low + 0.5 * (normalized_states + 1.0) * (
            self.state_high - self.state_low
        )

    def _log_errors(self, errors: torch.Tensor) -> torch.Tensor:
        return torch.log(errors.float().clamp_min(self.error_epsilon))

    def set_target_scale(self, scale: float) -> None:
        scale = float(scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"target scale must be finite and positive, got {scale}.")
        self.target_scale.fill_(scale)

    def _transform_errors(self, errors: torch.Tensor) -> torch.Tensor:
        if self.target_transform == 'log':
            return torch.log(
                errors.float().clamp_min(self.error_epsilon)
                / self.target_log_reference
            )
        errors = errors.float().clamp_min(0.0)
        if self.target_transform == 'normalized_raw':
            return (errors / self.target_scale.clamp_min(self.error_epsilon)).clamp_max(
                self.normalized_raw_clip
            )
        return errors

    def _inverse_transform_errors(self, predictions: torch.Tensor) -> torch.Tensor:
        if self.target_transform == 'log':
            min_prediction = math.log(
                self.error_epsilon / self.target_log_reference
            )
            max_prediction = (
                math.log(self.max_raw_error / self.target_log_reference)
                if self.max_raw_error is not None else None
            )
            predictions = predictions.clamp(
                min=min_prediction, max=max_prediction
            )
            return self.target_log_reference * torch.exp(predictions)
        predictions = predictions.clamp_min(0.0)
        if self.target_transform == 'normalized_raw':
            return predictions.clamp_max(self.normalized_raw_clip) * self.target_scale
        return predictions

    def query(
            self,
            root_states: torch.Tensor,
            error_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> RNDLandscapeQueryContext:
        root_states = root_states.reshape(-1, 2).float()
        root_normalized = self.normalize_states(root_states).clamp(-1.0, 1.0)
        summary = root_states.new_zeros((root_states.shape[0], self.hidden_dim))
        query_states = []
        query_log_errors = []
        query_tokens = []

        for step in range(self.query_count):
            step_ids = torch.full(
                (root_states.shape[0],), step, dtype=torch.long, device=root_states.device
            )
            step_embedding = self.query_step_embedding(step_ids)
            query_normalized = self.query_policy(
                torch.cat([root_normalized, summary, step_embedding], dim=-1)
            )
            physical_query = self.denormalize_states(query_normalized)
            exact_error = error_fn(physical_query).reshape(-1, 1)
            log_error = self._log_errors(exact_error)
            token = self.query_pair_encoder(
                torch.cat([query_normalized, log_error], dim=-1)
            )

            query_states.append(physical_query)
            query_log_errors.append(log_error)
            query_tokens.append(token)
            summary = torch.stack(query_tokens, dim=1).mean(dim=1)

        return RNDLandscapeQueryContext(
            root_states=root_states,
            query_states=torch.stack(query_states, dim=1),
            query_log_errors=torch.stack(query_log_errors, dim=1),
        )

    def _flatten_query_latents(self, query_latents: torch.Tensor) -> torch.Tensor:
        if query_latents.dim() < 4:
            raise ValueError(
                "query_latents must have shape [batch, queries, ensemble, ...], "
                f"got {tuple(query_latents.shape)}."
            )
        flat = query_latents.reshape(*query_latents.shape[:3], -1).float()
        if flat.shape[-1] != self.latent_dim:
            raise ValueError(
                f"Expected query latent dimension {self.latent_dim}, got {flat.shape[-1]}."
            )
        return flat

    def _flatten_imagined_latents(
            self,
            context: RNDLandscapeQueryContext,
            imagined_latents: torch.Tensor,
    ) -> Tuple[torch.Tensor, bool]:
        if context.query_latents is None:
            raise ValueError("RND landscape query context has no encoded query latents.")
        query_rank = context.query_latents.dim()
        squeeze_target = imagined_latents.dim() == query_rank - 1
        if squeeze_target:
            imagined_latents = imagined_latents.unsqueeze(1)
        if imagined_latents.dim() != query_rank:
            raise ValueError(
                "imagined_latents must have shape [batch, ensemble, ...] or "
                "[batch, targets, ensemble, ...], got "
                f"{tuple(imagined_latents.shape)}."
            )
        flat = imagined_latents.reshape(*imagined_latents.shape[:3], -1).float()
        if flat.shape[-1] != self.latent_dim:
            raise ValueError(
                f"Expected imagined latent dimension {self.latent_dim}, got {flat.shape[-1]}."
            )
        return flat, squeeze_target

    def predict_latent_member_target(
            self,
            context: RNDLandscapeQueryContext,
            imagined_latents: torch.Tensor,
            root_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if root_indices is not None:
            context = context.select(root_indices.long())
        query_latents = self._flatten_query_latents(context.query_latents)
        imagined_latents, squeeze_target = self._flatten_imagined_latents(
            context, imagined_latents
        )
        if imagined_latents.shape[0] != query_latents.shape[0]:
            raise ValueError("Every imagined latent row must have one matching query context.")
        if imagined_latents.shape[2] != query_latents.shape[2]:
            raise ValueError("Imagined and queried latents must use the same ensemble size.")

        if self.reader_architecture == 'landscape':
            query_latents = F.layer_norm(query_latents, (self.latent_dim,))
            imagined_latents = F.layer_norm(imagined_latents, (self.latent_dim,))

        def apply_ordinary_reader(
                latent_reader: nn.Module,
                member_imagined_latents: torch.Tensor,
        ) -> torch.Tensor:
            if member_imagined_latents.dim() == 3:
                depth_predictions = [
                    latent_reader(member_imagined_latents[:, depth]).squeeze(-1)
                    for depth in range(member_imagined_latents.shape[1])
                ]
                return torch.stack(depth_predictions, dim=1)
            if member_imagined_latents.dim() == 4:
                depth_predictions = []
                for depth in range(member_imagined_latents.shape[1]):
                    member_predictions = [
                        latent_reader(
                            member_imagined_latents[:, depth, member_id]
                        ).squeeze(-1)
                        for member_id in range(member_imagined_latents.shape[2])
                    ]
                    depth_predictions.append(
                        torch.stack(member_predictions, dim=1)
                    )
                return torch.stack(depth_predictions, dim=1)
            raise ValueError(
                "Ordinary auxiliary reader expected member latents with rank 3 "
                "or shared-reader latents with rank 4, got "
                f"{tuple(member_imagined_latents.shape)}."
            )

        def predict_member(
                member_query_latents: torch.Tensor,
                member_imagined_latents: torch.Tensor,
                latent_encoder: nn.Module,
                latent_pair_encoder: nn.Module,
                latent_pair_score: nn.Module,
                latent_reader: nn.Module,
        ) -> torch.Tensor:
            if self.reader_architecture == 'ordinary':
                return apply_ordinary_reader(
                    latent_reader, member_imagined_latents
                )
            query_features = latent_encoder(member_query_latents)
            target_features = latent_encoder(member_imagined_latents)
            if self.reader_use_context:
                target_count = target_features.shape[1]
                query_count = query_features.shape[1]
                queries_for_targets = query_features.unsqueeze(1).expand(
                    -1, target_count, -1, -1
                )
                targets_for_queries = target_features.unsqueeze(2).expand(
                    -1, -1, query_count, -1
                )
                errors_for_targets = context.query_log_errors.unsqueeze(1).expand(
                    -1, target_count, -1, -1
                )
                pair_features = latent_pair_encoder(
                    torch.cat(
                        [
                            queries_for_targets,
                            targets_for_queries,
                            targets_for_queries - queries_for_targets,
                            errors_for_targets,
                        ],
                        dim=-1,
                    )
                )
                attention = torch.softmax(
                    latent_pair_score(pair_features).squeeze(-1), dim=2
                )
                context_features = (attention.unsqueeze(-1) * pair_features).sum(dim=2)
                weighted_log_error = (
                    attention.unsqueeze(-1) * errors_for_targets
                ).sum(dim=2)
            else:
                context_features = torch.zeros_like(target_features)
                weighted_log_error = target_features.new_zeros(
                    *target_features.shape[:-1], 1
                )
            return latent_reader(
                torch.cat([target_features, context_features, weighted_log_error], dim=-1)
            ).squeeze(-1)

        if self.member_specific_reader:
            if imagined_latents.shape[2] != self.ensemble_size:
                raise ValueError(
                    "Member-specific reader expected "
                    f"{self.ensemble_size} members, got {imagined_latents.shape[2]}."
                )
            member_predictions = []
            for member_id in range(self.ensemble_size):
                member_predictions.append(
                    predict_member(
                        query_latents[:, :, member_id],
                        imagined_latents[:, :, member_id],
                        self.latent_encoders[member_id],
                        self.latent_pair_encoders[member_id],
                        self.latent_pair_scores[member_id],
                        self.latent_readers[member_id],
                    )
                )
            prediction = torch.stack(member_predictions, dim=2)
        else:
            if self.reader_architecture == 'ordinary':
                prediction = apply_ordinary_reader(
                    self.latent_reader, imagined_latents
                )
                return prediction.squeeze(1) if squeeze_target else prediction
            query_features = self.latent_encoder(query_latents)
            target_features = self.latent_encoder(imagined_latents)
            if self.reader_use_context:
                target_count = target_features.shape[1]
                query_count = query_features.shape[1]
                ensemble_size = target_features.shape[2]
                queries_for_targets = query_features.unsqueeze(1).expand(
                    -1, target_count, -1, -1, -1
                )
                targets_for_queries = target_features.unsqueeze(2).expand(
                    -1, -1, query_count, -1, -1
                )
                errors_for_targets = context.query_log_errors.unsqueeze(1).unsqueeze(3).expand(
                    -1, target_count, -1, ensemble_size, -1
                )
                pair_features = self.latent_pair_encoder(
                    torch.cat(
                        [
                            queries_for_targets,
                            targets_for_queries,
                            targets_for_queries - queries_for_targets,
                            errors_for_targets,
                        ],
                        dim=-1,
                    )
                )
                attention = torch.softmax(
                    self.latent_pair_score(pair_features).squeeze(-1), dim=2
                )
                context_features = (attention.unsqueeze(-1) * pair_features).sum(dim=2)
                weighted_log_error = (
                    attention.unsqueeze(-1) * errors_for_targets
                ).sum(dim=2)
            else:
                context_features = torch.zeros_like(target_features)
                weighted_log_error = target_features.new_zeros(
                    *target_features.shape[:-1], 1
                )
            prediction = self.latent_reader(
                torch.cat([target_features, context_features, weighted_log_error], dim=-1)
            ).squeeze(-1)
        return prediction.squeeze(1) if squeeze_target else prediction

    def predict_latent_member_log_error(
            self,
            context: RNDLandscapeQueryContext,
            imagined_latents: torch.Tensor,
            root_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.target_transform != 'log':
            raise RuntimeError(
                "predict_latent_member_log_error is only valid for target_transform='log'."
            )
        prediction = self.predict_latent_member_target(
            context, imagined_latents, root_indices=root_indices
        )
        return prediction + math.log(self.target_log_reference)

    def predict_latent_error(
            self,
            context: RNDLandscapeQueryContext,
            imagined_latents: torch.Tensor,
            root_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        member_predictions = self.predict_latent_member_target(
            context, imagined_latents, root_indices=root_indices
        )
        return self._inverse_transform_errors(member_predictions).mean(dim=-1).clamp_min(0.0)

    def _sample_grid(
            self,
            grid_errors: torch.Tensor,
            physical_states: torch.Tensor,
    ) -> torch.Tensor:
        """Bilinearly sample [batch, position, velocity] error grids."""
        if physical_states.dim() == 2:
            physical_states = physical_states.unsqueeze(1)
        normalized = self.normalize_states(physical_states).clamp(-1.0, 1.0)
        sampling_grid = normalized.unsqueeze(2)
        image = grid_errors.transpose(1, 2).unsqueeze(1)
        sampled = F.grid_sample(
            image,
            sampling_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return sampled[:, 0, :, 0]

    def _query_geometry(
            self,
            context: RNDLandscapeQueryContext,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return comparable root/query coordinates for diagnostics and regularization."""
        return (
            self.normalize_states(context.root_states),
            self.normalize_states(context.query_states),
        )

    def _loss_from_predictions(
            self,
            context: RNDLandscapeQueryContext,
            predicted_target: torch.Tensor,
            target_error: torch.Tensor,
            diversity_weight: float,
            centered_prediction_weight: float,
            target_mask: Optional[torch.Tensor] = None,
            sample_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        transformed_target = self._transform_errors(target_error)
        if target_mask is None:
            target_mask = torch.ones(
                predicted_target.shape[:2],
                dtype=predicted_target.dtype,
                device=predicted_target.device,
            )
        else:
            target_mask = target_mask.to(
                device=predicted_target.device,
                dtype=predicted_target.dtype,
            )
        if target_mask.shape != predicted_target.shape[:2]:
            raise ValueError(
                "target_mask must have shape [batch, targets], got "
                f"{tuple(target_mask.shape)} for predictions "
                f"{tuple(predicted_target.shape)}."
            )
        if sample_weights is None:
            sample_weights = torch.ones(
                predicted_target.shape[0],
                dtype=predicted_target.dtype,
                device=predicted_target.device,
            )
        else:
            sample_weights = sample_weights.to(
                device=predicted_target.device,
                dtype=predicted_target.dtype,
            ).reshape(-1)
        if sample_weights.shape[0] != predicted_target.shape[0]:
            raise ValueError(
                "sample_weights must contain one value per batch row, got "
                f"{sample_weights.shape[0]} for batch size "
                f"{predicted_target.shape[0]}."
            )

        element_weights = (
            target_mask.unsqueeze(-1) * sample_weights[:, None, None]
        ).expand_as(predicted_target)
        weight_sum = element_weights.sum().clamp_min(1.0)

        prediction_terms = F.smooth_l1_loss(
            predicted_target, transformed_target, reduction="none"
        )
        prediction_loss = (prediction_terms * element_weights).sum() / weight_sum

        target_count = target_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        prediction_mean = (
            predicted_target * target_mask.unsqueeze(-1)
        ).sum(dim=1, keepdim=True) / target_count.unsqueeze(-1)
        target_mean = (
            transformed_target * target_mask.unsqueeze(-1)
        ).sum(dim=1, keepdim=True) / target_count.unsqueeze(-1)
        centered_prediction = predicted_target - prediction_mean
        centered_target = transformed_target - target_mean
        centered_terms = F.smooth_l1_loss(
            centered_prediction, centered_target, reduction="none"
        )
        centered_prediction_loss = (centered_terms * element_weights).sum() / weight_sum

        valid = element_weights > 0
        centered_prediction_flat = centered_prediction[valid]
        centered_target_flat = centered_target[valid]
        if centered_prediction_flat.numel() > 0:
            centered_prediction_correlation = (
                (centered_prediction_flat * centered_target_flat).mean()
                / (
                    centered_prediction_flat.square().mean().sqrt()
                    * centered_target_flat.square().mean().sqrt()
                ).clamp_min(1e-12)
            )
        else:
            centered_prediction_correlation = prediction_loss.new_zeros(())

        root_normalized, query_normalized = self._query_geometry(context)
        if self.query_count > 1:
            pairwise_distance_sq = (
                query_normalized.unsqueeze(2) - query_normalized.unsqueeze(1)
            ).pow(2).sum(dim=-1)
            off_diagonal = ~torch.eye(
                self.query_count, dtype=torch.bool, device=pairwise_distance_sq.device
            ).unsqueeze(0)
            diversity_loss = torch.exp(-pairwise_distance_sq / 0.25)[
                off_diagonal.expand_as(pairwise_distance_sq)
            ].mean()
            pairwise_distance = pairwise_distance_sq.clamp_min(0.0).sqrt()
            query_pair_distance_mean = pairwise_distance[
                off_diagonal.expand_as(pairwise_distance)
            ].mean()
            query_pair_distance_min = pairwise_distance.masked_fill(
                ~off_diagonal, float('inf')
            ).amin(dim=(1, 2)).mean()
        else:
            diversity_loss = prediction_loss.new_zeros(())
            query_pair_distance_mean = prediction_loss.new_zeros(())
            query_pair_distance_min = prediction_loss.new_zeros(())
        query_root_distance_mean = (
            query_normalized - root_normalized.unsqueeze(1)
        ).norm(dim=-1).mean()
        total_loss = (
            prediction_loss
            + float(centered_prediction_weight) * centered_prediction_loss
            + float(diversity_weight) * diversity_loss
        )
        predicted_error = self._inverse_transform_errors(predicted_target)
        prediction_mae_target_space = (
            (predicted_target - transformed_target).abs() * element_weights
        ).sum().div(weight_sum).detach()
        prediction_target_mean = (
            predicted_target * element_weights
        ).sum().div(weight_sum)
        transformed_target_mean = (
            transformed_target * element_weights
        ).sum().div(weight_sum)
        prediction_target_centered = predicted_target - prediction_target_mean
        transformed_target_centered = transformed_target - transformed_target_mean
        prediction_target_std = (
            prediction_target_centered.square() * element_weights
        ).sum().div(weight_sum).sqrt()
        transformed_target_std = (
            transformed_target_centered.square() * element_weights
        ).sum().div(weight_sum).sqrt()
        prediction_target_correlation = (
            (
                prediction_target_centered
                * transformed_target_centered
                * element_weights
            ).sum().div(weight_sum)
            / (prediction_target_std * transformed_target_std).clamp_min(1e-12)
        )
        if self.target_transform == 'normalized_raw':
            target_clip_fraction = (
                (
                    target_error >= self.target_scale * self.normalized_raw_clip
                ).to(element_weights.dtype) * element_weights
            ).sum().div(weight_sum)
        else:
            target_clip_fraction = prediction_loss.new_zeros(())
        terms = {
            "prediction_loss": prediction_loss.detach(),
            "centered_prediction_loss": centered_prediction_loss.detach(),
            "centered_prediction_correlation": centered_prediction_correlation.detach(),
            "diversity_loss": diversity_loss.detach(),
            "prediction_mae_log": (
                (
                    self._log_errors(predicted_error)
                    - self._log_errors(target_error)
                ).abs() * element_weights
            ).sum().div(weight_sum).detach(),
            "prediction_mae_raw": (
                (predicted_error - target_error).abs() * element_weights
            ).sum().div(weight_sum).detach(),
            "prediction_mae_target_space": prediction_mae_target_space,
            "prediction_target_mean": prediction_target_mean.detach(),
            "prediction_target_std": prediction_target_std.detach(),
            "transformed_target_mean": transformed_target_mean.detach(),
            "transformed_target_std": transformed_target_std.detach(),
            "prediction_target_correlation": prediction_target_correlation.detach(),
            "target_scale": self.target_scale.detach().clone(),
            "target_clip_fraction": target_clip_fraction.detach(),
            "query_pair_distance_mean": query_pair_distance_mean.detach(),
            "query_pair_distance_min": query_pair_distance_min.detach(),
            "query_root_distance_mean": query_root_distance_mean.detach(),
            "query_error_mean": torch.exp(context.query_log_errors).mean().detach(),
            "target_error_mean": (
                target_error * element_weights
            ).sum().div(weight_sum).detach(),
            "valid_target_fraction": target_mask.mean().detach(),
        }
        for target_index in range(predicted_target.shape[1]):
            depth_weights = element_weights[:, target_index]
            depth_weight_sum = depth_weights.sum().clamp_min(1.0)
            depth_prediction = predicted_target[:, target_index]
            depth_target = transformed_target[:, target_index]
            depth_predicted_error = predicted_error[:, target_index]
            depth_target_error = target_error[:, target_index]
            depth_prediction_mean = (
                depth_prediction * depth_weights
            ).sum().div(depth_weight_sum)
            depth_target_mean = (
                depth_target * depth_weights
            ).sum().div(depth_weight_sum)
            depth_prediction_centered = depth_prediction - depth_prediction_mean
            depth_target_centered = depth_target - depth_target_mean
            depth_prediction_std = (
                depth_prediction_centered.square() * depth_weights
            ).sum().div(depth_weight_sum).sqrt()
            depth_target_std = (
                depth_target_centered.square() * depth_weights
            ).sum().div(depth_weight_sum).sqrt()
            prefix = f"depth_{target_index + 1}"
            terms[f"{prefix}_prediction_mae_target_space"] = (
                (depth_prediction - depth_target).abs() * depth_weights
            ).sum().div(depth_weight_sum).detach()
            terms[f"{prefix}_prediction_mae_log"] = (
                (
                    self._log_errors(depth_predicted_error)
                    - self._log_errors(depth_target_error)
                ).abs() * depth_weights
            ).sum().div(depth_weight_sum).detach()
            terms[f"{prefix}_prediction_mae_raw"] = (
                (depth_predicted_error - depth_target_error).abs() * depth_weights
            ).sum().div(depth_weight_sum).detach()
            terms[f"{prefix}_prediction_target_correlation"] = (
                (
                    depth_prediction_centered
                    * depth_target_centered
                    * depth_weights
                ).sum().div(depth_weight_sum)
                / (depth_prediction_std * depth_target_std).clamp_min(1e-12)
            ).detach()
            terms[f"{prefix}_valid_fraction"] = target_mask[
                :, target_index
            ].mean().detach()
        return total_loss, terms

    def loss_on_observations(
            self,
            root_states: torch.Tensor,
            imagined_latents: torch.Tensor,
            target_errors: torch.Tensor,
            error_fn: Callable[[torch.Tensor], torch.Tensor],
            query_latent_fn: Callable[[torch.Tensor], torch.Tensor],
            target_mask: Optional[torch.Tensor] = None,
            sample_weights: Optional[torch.Tensor] = None,
            diversity_weight: float = 0.01,
            centered_prediction_weight: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Train from current RND queries and real replay successor targets."""
        context = self.query(root_states, error_fn)
        context.query_latents = query_latent_fn(context.query_states)
        predicted_target = self.predict_latent_member_target(
            context, imagined_latents
        )
        with torch.no_grad():
            target_error = target_errors.float().unsqueeze(-1).expand_as(
                predicted_target
            )
        return self._loss_from_predictions(
            context,
            predicted_target,
            target_error,
            diversity_weight,
            centered_prediction_weight,
            target_mask=target_mask,
            sample_weights=sample_weights,
        )

    def loss_on_latents(
            self,
            grid_errors: torch.Tensor,
            root_states: torch.Tensor,
            imagined_latents: torch.Tensor,
            target_errors: torch.Tensor,
            query_latent_fn: Callable[[torch.Tensor], torch.Tensor],
            diversity_weight: float = 0.01,
            centered_prediction_weight: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        def grid_error_fn(query_states: torch.Tensor) -> torch.Tensor:
            return self._sample_grid(grid_errors, query_states).squeeze(1)

        context = self.query(root_states, grid_error_fn)
        context.query_latents = query_latent_fn(context.query_states)
        predicted_target = self.predict_latent_member_target(
            context, imagined_latents
        )
        with torch.no_grad():
            target_error = target_errors.float().unsqueeze(-1).expand_as(
                predicted_target
            )
        return self._loss_from_predictions(
            context,
            predicted_target,
            target_error,
            diversity_weight,
            centered_prediction_weight,
        )
