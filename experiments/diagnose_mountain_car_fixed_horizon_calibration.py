import argparse
import csv
import importlib.util
import itertools
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from lzero.model.ensemble_muzero_model import EnsembleMuZeroModelMLP
from lzero.policy import DiscreteSupport, InverseScalarTransform


POSITION_BIN_WIDTH = 0.005
VELOCITY_BIN_WIDTH = 0.001


@dataclass(frozen=True)
class RootState:
    collection_phase: int
    train_iter: int
    envstep_after_collect: int
    env_id: int
    episode_index: int
    timestep: int
    position: float
    velocity: float

    @property
    def observation(self) -> Tuple[float, float]:
        return self.position, self.velocity


class OccupancyIndex:
    def __init__(self, observations: Iterable[Tuple[float, float]]) -> None:
        self.counts = Counter(observation_bin_key(position, velocity) for position, velocity in observations)
        self._position_min_bin = int(round(-1.2 / POSITION_BIN_WIDTH))
        self._position_max_bin = int(round(0.6 / POSITION_BIN_WIDTH))
        self._velocity_min_bin = int(round(-0.07 / VELOCITY_BIN_WIDTH))
        self._velocity_max_bin = int(round(0.07 / VELOCITY_BIN_WIDTH))
        self._dense_counts = np.zeros(
            (
                self._position_max_bin - self._position_min_bin + 1,
                self._velocity_max_bin - self._velocity_min_bin + 1,
            ),
            dtype=np.int64,
        )
        for (position_bin, velocity_bin), count in self.counts.items():
            position_index = position_bin - self._position_min_bin
            velocity_index = velocity_bin - self._velocity_min_bin
            if (
                    0 <= position_index < self._dense_counts.shape[0]
                    and 0 <= velocity_index < self._dense_counts.shape[1]
            ):
                self._dense_counts[position_index, velocity_index] = int(count)

    def count(self, position: float, velocity: float, radius_bins: int = 0) -> int:
        position_bin, velocity_bin = observation_bin_key(position, velocity)
        radius_bins = max(0, int(radius_bins))
        return int(
            sum(
                self.counts[(position_bin + position_delta, velocity_bin + velocity_delta)]
                for position_delta in range(-radius_bins, radius_bins + 1)
                for velocity_delta in range(-radius_bins, radius_bins + 1)
            )
        )

    def count_many(
            self,
            positions: np.ndarray,
            velocities: np.ndarray,
            radius_bins: int = 0,
    ) -> np.ndarray:
        positions = np.asarray(positions)
        velocities = np.asarray(velocities)
        position_bins = np.rint(positions / POSITION_BIN_WIDTH).astype(np.int64)
        velocity_bins = np.rint(velocities / VELOCITY_BIN_WIDTH).astype(np.int64)
        result = np.zeros(np.broadcast_shapes(position_bins.shape, velocity_bins.shape), dtype=np.int64)
        position_bins, velocity_bins = np.broadcast_arrays(position_bins, velocity_bins)
        radius_bins = max(0, int(radius_bins))
        for position_delta in range(-radius_bins, radius_bins + 1):
            position_indices = position_bins + position_delta - self._position_min_bin
            for velocity_delta in range(-radius_bins, radius_bins + 1):
                velocity_indices = velocity_bins + velocity_delta - self._velocity_min_bin
                valid = (
                    (position_indices >= 0)
                    & (position_indices < self._dense_counts.shape[0])
                    & (velocity_indices >= 0)
                    & (velocity_indices < self._dense_counts.shape[1])
                )
                result[valid] += self._dense_counts[
                    position_indices[valid], velocity_indices[valid]
                ]
        return result


def observation_bin_key(position: float, velocity: float) -> Tuple[int, int]:
    return (
        int(round(float(position) / POSITION_BIN_WIDTH)),
        int(round(float(velocity) / VELOCITY_BIN_WIDTH)),
    )


def mountain_car_step(position: float, velocity: float, action: int) -> Tuple[float, float, float, bool]:
    next_velocity = float(velocity) + (int(action) - 1) * 0.001 - 0.0025 * math.cos(3.0 * float(position))
    next_velocity = float(np.clip(next_velocity, -0.07, 0.07))
    next_position = float(np.clip(float(position) + next_velocity, -1.2, 0.6))
    if next_position <= -1.2 and next_velocity < 0.0:
        next_velocity = 0.0
    terminated = next_position >= 0.5
    return next_position, next_velocity, -1.0, terminated


def simulate_action_sequences(
        observations: np.ndarray,
        sequences: np.ndarray,
        discount: float,
) -> Dict[str, np.ndarray]:
    observations = np.asarray(observations, dtype=np.float32)
    sequences = np.asarray(sequences, dtype=np.int64)
    batch_size = observations.shape[0]
    sequence_count, horizon = sequences.shape
    positions = np.broadcast_to(observations[:, None, 0], (batch_size, sequence_count)).copy()
    velocities = np.broadcast_to(observations[:, None, 1], (batch_size, sequence_count)).copy()
    active = np.ones((batch_size, sequence_count), dtype=bool)
    terminated = np.zeros_like(active)
    executed_steps = np.zeros((batch_size, sequence_count), dtype=np.int16)
    reward_returns = np.zeros((batch_size, sequence_count), dtype=np.float32)

    for depth in range(horizon):
        actions = np.broadcast_to(sequences[None, :, depth], (batch_size, sequence_count))
        next_velocities = velocities + (actions - 1) * 0.001 - 0.0025 * np.cos(3.0 * positions)
        next_velocities = np.clip(next_velocities, -0.07, 0.07)
        next_positions = np.clip(positions + next_velocities, -1.2, 0.6)
        next_velocities = np.where((next_positions <= -1.2) & (next_velocities < 0.0), 0.0, next_velocities)

        reward_returns += active.astype(np.float32) * (float(discount) ** depth) * -1.0
        executed_steps += active.astype(np.int16)
        positions = np.where(active, next_positions, positions)
        velocities = np.where(active, next_velocities, velocities)
        just_terminated = active & (positions >= 0.5)
        terminated |= just_terminated
        active &= ~just_terminated

    return {
        "leaf_observations": np.stack([positions, velocities], axis=-1).astype(np.float32),
        "reward_returns": reward_returns,
        "terminated": terminated,
        "executed_steps": executed_steps,
    }


def evaluate_reachable_novelty(
        observations: np.ndarray,
        continuation_sequences: np.ndarray,
        occupancy: OccupancyIndex,
        density_radius_bins: int,
        novelty_discount: float = 1.0,
        terminal_mask: np.ndarray = None,
        batch_size: int = 512,
) -> Dict[str, np.ndarray]:
    """Evaluate future replay novelty using exact MountainCar transitions."""
    observations = np.asarray(observations, dtype=np.float32).reshape(-1, 2)
    continuation_sequences = np.asarray(continuation_sequences, dtype=np.int64)
    if continuation_sequences.ndim != 2 or continuation_sequences.shape[1] <= 0:
        raise ValueError("continuation_sequences must have shape [sequence_count, positive_horizon]")
    terminal_mask = (
        np.zeros(len(observations), dtype=bool)
        if terminal_mask is None
        else np.asarray(terminal_mask, dtype=bool).reshape(-1)
    )
    if len(terminal_mask) != len(observations):
        raise ValueError("terminal_mask must have one entry per observation")

    sequence_count, horizon = continuation_sequences.shape
    best_returns = np.zeros(len(observations), dtype=np.float32)
    mean_returns = np.zeros(len(observations), dtype=np.float32)
    best_sequence_indices = np.full(len(observations), -1, dtype=np.int32)
    best_path_min_densities = np.full(len(observations), -1, dtype=np.int64)
    reachable_min_densities = np.full(len(observations), -1, dtype=np.int64)
    best_endpoint_positions = observations[:, 0].astype(np.float32).copy()
    best_endpoint_velocities = observations[:, 1].astype(np.float32).copy()
    best_endpoint_densities = np.full(len(observations), -1, dtype=np.int64)

    for start in range(0, len(observations), max(1, int(batch_size))):
        end = min(len(observations), start + max(1, int(batch_size)))
        batch_observations = observations[start:end]
        batch_terminal = terminal_mask[start:end]
        local_batch_size = len(batch_observations)
        positions = np.broadcast_to(
            batch_observations[:, None, 0], (local_batch_size, sequence_count)
        ).copy()
        velocities = np.broadcast_to(
            batch_observations[:, None, 1], (local_batch_size, sequence_count)
        ).copy()
        active = np.broadcast_to(~batch_terminal[:, None], positions.shape).copy()
        novelty_returns = np.zeros_like(positions, dtype=np.float32)
        path_min_densities = np.full(positions.shape, np.iinfo(np.int64).max, dtype=np.int64)

        for depth in range(horizon):
            actions = np.broadcast_to(
                continuation_sequences[None, :, depth], positions.shape
            )
            next_velocities = velocities + (actions - 1) * 0.001 - 0.0025 * np.cos(3.0 * positions)
            next_velocities = np.clip(next_velocities, -0.07, 0.07)
            next_positions = np.clip(positions + next_velocities, -1.2, 0.6)
            next_velocities = np.where(
                (next_positions <= -1.2) & (next_velocities < 0.0), 0.0, next_velocities
            )
            positions = np.where(active, next_positions, positions)
            velocities = np.where(active, next_velocities, velocities)
            densities = occupancy.count_many(
                positions, velocities, radius_bins=density_radius_bins
            )
            novelty = 1.0 / np.sqrt(densities.astype(np.float32) + 1.0)
            novelty_returns += active.astype(np.float32) * (float(novelty_discount) ** depth) * novelty
            path_min_densities = np.where(
                active, np.minimum(path_min_densities, densities), path_min_densities
            )
            active &= positions < 0.5

        best_indices = np.argmax(novelty_returns, axis=1)
        row_indices = np.arange(local_batch_size)
        nonterminal_rows = ~batch_terminal
        best_returns[start:end] = novelty_returns[row_indices, best_indices]
        mean_returns[start:end] = novelty_returns.mean(axis=1)
        best_sequence_indices[start:end] = np.where(nonterminal_rows, best_indices, -1)
        best_path_min = path_min_densities[row_indices, best_indices]
        reachable_min = path_min_densities.min(axis=1)
        best_path_min_densities[start:end] = np.where(nonterminal_rows, best_path_min, -1)
        reachable_min_densities[start:end] = np.where(nonterminal_rows, reachable_min, -1)
        best_endpoint_positions[start:end] = positions[row_indices, best_indices]
        best_endpoint_velocities[start:end] = velocities[row_indices, best_indices]
        best_endpoint_densities[start:end] = np.where(
            nonterminal_rows,
            occupancy.count_many(
                positions[row_indices, best_indices],
                velocities[row_indices, best_indices],
                radius_bins=density_radius_bins,
            ),
            -1,
        )

    return {
        "best_return": best_returns,
        "mean_return": mean_returns,
        "best_sequence_index": best_sequence_indices,
        "best_path_min_density": best_path_min_densities,
        "reachable_min_density": reachable_min_densities,
        "best_endpoint_position": best_endpoint_positions,
        "best_endpoint_velocity": best_endpoint_velocities,
        "best_endpoint_density": best_endpoint_densities,
    }


def density_bucket(count: int) -> str:
    if count <= 0:
        return "unseen"
    if count <= 5:
        return "sparse_1_5"
    if count <= 50:
        return "medium_6_50"
    return "dense_gt_50"


def _finite_float(value, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _finite_int(value, default: int = -1) -> int:
    numeric = _finite_float(value, float(default))
    return int(numeric) if np.isfinite(numeric) else int(default)


def _truthy(value) -> bool:
    return str(value).strip().lower() in {"1", "1.0", "true", "yes"}


def load_real_root_states(
        csv_path: Path,
        max_train_iter: int,
        max_envstep: int,
) -> List[RootState]:
    roots: List[RootState] = []
    seen = set()
    with csv_path.open(newline="") as file:
        for row in csv.DictReader(file):
            if not _truthy(row.get("selected", "0")):
                continue
            train_iter = _finite_int(row.get("train_iter_at_collect"))
            envstep_after_collect = _finite_int(row.get("envstep_after_collect"))
            if max_train_iter >= 0 and train_iter > max_train_iter:
                continue
            if max_envstep >= 0 and envstep_after_collect > max_envstep:
                continue
            position = _finite_float(row.get("obs_position"))
            velocity = _finite_float(row.get("obs_velocity"))
            if not np.isfinite(position) or not np.isfinite(velocity):
                continue
            root = RootState(
                collection_phase=_finite_int(row.get("collection_phase")),
                train_iter=train_iter,
                envstep_after_collect=envstep_after_collect,
                env_id=_finite_int(row.get("env_id")),
                episode_index=_finite_int(row.get("collection_episode_index")),
                timestep=_finite_int(row.get("segment_timestep"), _finite_int(row.get("timestep"))),
                position=position,
                velocity=velocity,
            )
            key = (
                root.collection_phase,
                root.env_id,
                root.episode_index,
                root.timestep,
            )
            if key not in seen:
                seen.add(key)
                roots.append(root)
    if not roots:
        raise RuntimeError(f"No selected real root states found in {csv_path}")
    return roots


def stratified_sample_roots(
        roots: Sequence[RootState],
        occupancy: OccupancyIndex,
        sample_count: int,
        radius_bins: int,
        seed: int,
) -> List[RootState]:
    if sample_count <= 0 or sample_count >= len(roots):
        return list(roots)
    grouped: Dict[str, List[RootState]] = {bucket: [] for bucket in density_bucket_order()}
    for root in roots:
        count = occupancy.count(root.position, root.velocity, radius_bins=radius_bins)
        grouped[density_bucket(count)].append(root)

    rng = np.random.default_rng(seed)
    selected: List[RootState] = []
    nonempty_groups = [values for values in grouped.values() if values]
    per_group = max(1, sample_count // max(1, len(nonempty_groups)))
    for values in nonempty_groups:
        take = min(per_group, len(values), sample_count - len(selected))
        indices = rng.choice(len(values), size=take, replace=False)
        selected.extend(values[int(index)] for index in np.atleast_1d(indices))
    if len(selected) < sample_count:
        selected_keys = set(selected)
        remaining = [root for root in roots if root not in selected_keys]
        take = min(sample_count - len(selected), len(remaining))
        indices = rng.choice(len(remaining), size=take, replace=False)
        selected.extend(remaining[int(index)] for index in np.atleast_1d(indices))
    return selected


def density_bucket_order() -> List[str]:
    return ["unseen", "sparse_1_5", "medium_6_50", "dense_gt_50"]


def _load_config(path: Path) -> Dict:
    spec = importlib.util.spec_from_file_location("fixed_horizon_diagnostic_config", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import config from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.exp_config


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    return torch.device(device_arg)


def _decode_handle(model_config: Dict, support_key: str, device: torch.device) -> InverseScalarTransform:
    support_range = model_config[support_key]
    support = DiscreteSupport(
        float(support_range[0]),
        float(support_range[1]),
        float(support_range[2]),
        device=device,
    )
    return InverseScalarTransform(support, bool(model_config.get("categorical_distribution", True)))


def _member_ids(ensemble_size: int, rows_per_member: int) -> List[int]:
    return [member_id for member_id in range(ensemble_size) for _ in range(rows_per_member)]


def _initial_member_outputs(
        model: EnsembleMuZeroModelMLP,
        observations: torch.Tensor,
        ensemble_size: int,
):
    repeated = torch.cat([observations for _ in range(ensemble_size)], dim=0)
    return model.initial_inference(
        repeated,
        member_id=_member_ids(ensemble_size, int(observations.shape[0])),
    )


def replay_value_offsets(
        model: EnsembleMuZeroModelMLP,
        observations: np.ndarray,
        ensemble_size: int,
        value_decoder: InverseScalarTransform,
        device: torch.device,
        inference_batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    member_values = []
    with torch.no_grad():
        for start in range(0, len(observations), inference_batch_size):
            obs = torch.as_tensor(
                observations[start:start + inference_batch_size], dtype=torch.float32, device=device
            )
            output = _initial_member_outputs(model, obs, ensemble_size)
            values = value_decoder(output.value).reshape(ensemble_size, len(obs)).cpu().numpy()
            member_values.append(values)
    values = np.concatenate(member_values, axis=1)
    member_means = values.mean(axis=1)
    return (member_means - member_means.mean()).astype(np.float32), member_means.astype(np.float32)


def _safe_cosine(imagined: torch.Tensor, direct: torch.Tensor) -> torch.Tensor:
    imagined = imagined.reshape(*imagined.shape[:3], -1).float()
    direct = direct.reshape(*direct.shape[:3], -1).float()
    numerator = (imagined * direct).sum(dim=-1)
    denominator = imagined.norm(dim=-1) * direct.norm(dim=-1)
    return numerator / denominator.clamp_min(1e-8)


def evaluate_fixed_horizon(
        model: EnsembleMuZeroModelMLP,
        roots: Sequence[RootState],
        sequences: np.ndarray,
        occupancy: OccupancyIndex,
        value_decoder: InverseScalarTransform,
        reward_decoder: InverseScalarTransform,
        value_offsets: np.ndarray,
        discount: float,
        density_radius_bins: int,
        state_batch_size: int,
        sequence_batch_size: int,
        device: torch.device,
) -> List[Dict]:
    ensemble_size = len(model.members)
    horizon = int(sequences.shape[1])
    records: List[Dict] = []
    offset_tensor = torch.as_tensor(value_offsets, dtype=torch.float32, device=device).reshape(
        ensemble_size, 1, 1
    )

    with torch.no_grad():
        for state_start in range(0, len(roots), state_batch_size):
            state_roots = roots[state_start:state_start + state_batch_size]
            observations_np = np.asarray([root.observation for root in state_roots], dtype=np.float32)
            observations = torch.as_tensor(observations_np, dtype=torch.float32, device=device)
            root_output = _initial_member_outputs(model, observations, ensemble_size)
            root_latents = root_output.latent_state.reshape(
                ensemble_size, len(state_roots), *root_output.latent_state.shape[1:]
            )

            for sequence_start in range(0, len(sequences), sequence_batch_size):
                sequence_chunk = sequences[sequence_start:sequence_start + sequence_batch_size]
                sequence_count = len(sequence_chunk)
                true_rollout = simulate_action_sequences(observations_np, sequence_chunk, discount)

                latents = root_latents.unsqueeze(2).expand(
                    ensemble_size,
                    len(state_roots),
                    sequence_count,
                    *root_latents.shape[2:],
                ).reshape(
                    ensemble_size * len(state_roots) * sequence_count,
                    *root_latents.shape[2:],
                )
                predicted_reward_returns = torch.zeros(
                    (ensemble_size, len(state_roots), sequence_count),
                    dtype=torch.float32,
                    device=device,
                )
                imagined_values = None
                for depth in range(horizon):
                    actions = torch.as_tensor(sequence_chunk[:, depth], dtype=torch.long, device=device)
                    actions = actions.view(1, 1, sequence_count).expand(
                        ensemble_size, len(state_roots), sequence_count
                    ).reshape(-1)
                    output = model.recurrent_inference(
                        latents,
                        actions,
                        member_id=_member_ids(ensemble_size, len(state_roots) * sequence_count),
                    )
                    latents = output.latent_state
                    rewards = reward_decoder(output.reward).reshape(
                        ensemble_size, len(state_roots), sequence_count
                    )
                    predicted_reward_returns += (float(discount) ** depth) * rewards
                    imagined_values = value_decoder(output.value).reshape(
                        ensemble_size, len(state_roots), sequence_count
                    )

                imagined_latents = latents.reshape(
                    ensemble_size,
                    len(state_roots),
                    sequence_count,
                    *latents.shape[1:],
                )
                imagined_returns = predicted_reward_returns + (float(discount) ** horizon) * imagined_values

                leaf_obs_np = true_rollout["leaf_observations"].reshape(-1, observations_np.shape[1])
                leaf_obs = torch.as_tensor(leaf_obs_np, dtype=torch.float32, device=device)
                direct_output = _initial_member_outputs(model, leaf_obs, ensemble_size)
                direct_values = value_decoder(direct_output.value).reshape(
                    ensemble_size, len(state_roots), sequence_count
                )
                direct_latents = direct_output.latent_state.reshape(
                    ensemble_size,
                    len(state_roots),
                    sequence_count,
                    *direct_output.latent_state.shape[1:],
                )
                reward_returns = torch.as_tensor(
                    true_rollout["reward_returns"], dtype=torch.float32, device=device
                )
                terminal_mask = torch.as_tensor(
                    true_rollout["terminated"], dtype=torch.bool, device=device
                )
                direct_returns = reward_returns.unsqueeze(0) + (
                    (~terminal_mask).float().unsqueeze(0)
                    * (float(discount) ** horizon)
                    * direct_values
                )

                calibrated_imagined_returns = imagined_returns - (float(discount) ** horizon) * offset_tensor
                calibrated_direct_returns = direct_returns - (
                    (~terminal_mask).float().unsqueeze(0)
                    * (float(discount) ** horizon)
                    * offset_tensor
                )
                latent_cosine = _safe_cosine(imagined_latents, direct_latents)
                latent_rms = (
                    imagined_latents.reshape(ensemble_size, len(state_roots), sequence_count, -1).float()
                    - direct_latents.reshape(ensemble_size, len(state_roots), sequence_count, -1).float()
                ).square().mean(dim=-1).sqrt()

                arrays = {
                    "imagined_returns": imagined_returns.cpu().numpy(),
                    "direct_returns": direct_returns.cpu().numpy(),
                    "calibrated_imagined_returns": calibrated_imagined_returns.cpu().numpy(),
                    "calibrated_direct_returns": calibrated_direct_returns.cpu().numpy(),
                    "imagined_values": imagined_values.cpu().numpy(),
                    "direct_values": direct_values.cpu().numpy(),
                    "predicted_reward_returns": predicted_reward_returns.cpu().numpy(),
                    "latent_cosine": latent_cosine.cpu().numpy(),
                    "latent_rms": latent_rms.cpu().numpy(),
                }

                for root_index, root in enumerate(state_roots):
                    root_density = occupancy.count(
                        root.position, root.velocity, radius_bins=density_radius_bins
                    )
                    for local_sequence_index, sequence in enumerate(sequence_chunk):
                        leaf_position, leaf_velocity = true_rollout["leaf_observations"][
                            root_index, local_sequence_index
                        ]
                        leaf_density_exact = occupancy.count(leaf_position, leaf_velocity, radius_bins=0)
                        leaf_density = occupancy.count(
                            leaf_position, leaf_velocity, radius_bins=density_radius_bins
                        )
                        imagined_q = arrays["imagined_returns"][:, root_index, local_sequence_index]
                        direct_q = arrays["direct_returns"][:, root_index, local_sequence_index]
                        calibrated_imagined_q = arrays["calibrated_imagined_returns"][
                            :, root_index, local_sequence_index
                        ]
                        calibrated_direct_q = arrays["calibrated_direct_returns"][
                            :, root_index, local_sequence_index
                        ]
                        imagined_leaf_values = arrays["imagined_values"][:, root_index, local_sequence_index]
                        direct_leaf_values = arrays["direct_values"][:, root_index, local_sequence_index]
                        predicted_rewards = arrays["predicted_reward_returns"][
                            :, root_index, local_sequence_index
                        ]
                        raw_imagined_qstd = float(np.std(imagined_q))
                        raw_direct_qstd = float(np.std(direct_q))
                        calibrated_imagined_qstd = float(np.std(calibrated_imagined_q))
                        calibrated_direct_qstd = float(np.std(calibrated_direct_q))
                        record = {
                            "root_sample_index": state_start + root_index,
                            "collection_phase": root.collection_phase,
                            "train_iter_at_collect": root.train_iter,
                            "envstep_after_collect": root.envstep_after_collect,
                            "env_id": root.env_id,
                            "collection_episode_index": root.episode_index,
                            "timestep": root.timestep,
                            "root_position": root.position,
                            "root_velocity": root.velocity,
                            "root_density": root_density,
                            "sequence": "".join(str(int(action)) for action in sequence),
                            "first_action": int(sequence[0]),
                            "horizon": horizon,
                            "true_leaf_position": float(leaf_position),
                            "true_leaf_velocity": float(leaf_velocity),
                            "true_leaf_density_exact": leaf_density_exact,
                            "true_leaf_density": leaf_density,
                            "true_leaf_density_bucket": density_bucket(leaf_density),
                            "true_terminated": int(true_rollout["terminated"][root_index, local_sequence_index]),
                            "true_executed_steps": int(
                                true_rollout["executed_steps"][root_index, local_sequence_index]
                            ),
                            "true_reward_return": float(
                                true_rollout["reward_returns"][root_index, local_sequence_index]
                            ),
                            "predicted_reward_return_mean": float(np.mean(predicted_rewards)),
                            "predicted_reward_return_mae": float(
                                np.mean(
                                    np.abs(
                                        predicted_rewards
                                        - true_rollout["reward_returns"][root_index, local_sequence_index]
                                    )
                                )
                            ),
                            "imagined_q_mean": float(np.mean(imagined_q)),
                            "direct_q_mean": float(np.mean(direct_q)),
                            "imagined_direct_mean_gap": float(np.mean(imagined_q) - np.mean(direct_q)),
                            "imagined_direct_member_mae": float(np.mean(np.abs(imagined_q - direct_q))),
                            "raw_imagined_qstd": raw_imagined_qstd,
                            "raw_direct_qstd": raw_direct_qstd,
                            "raw_qstd_excess": raw_imagined_qstd - raw_direct_qstd,
                            "calibrated_imagined_qstd": calibrated_imagined_qstd,
                            "calibrated_direct_qstd": calibrated_direct_qstd,
                            "calibrated_qstd_excess": calibrated_imagined_qstd - calibrated_direct_qstd,
                            "imagined_leaf_value_std": float(np.std(imagined_leaf_values)),
                            "direct_leaf_value_std": float(np.std(direct_leaf_values)),
                            "imagined_direct_leaf_value_mae": float(
                                np.mean(np.abs(imagined_leaf_values - direct_leaf_values))
                            ),
                            "imagined_direct_latent_cosine_mean": float(
                                np.mean(arrays["latent_cosine"][:, root_index, local_sequence_index])
                            ),
                            "imagined_direct_latent_rms_mean": float(
                                np.mean(arrays["latent_rms"][:, root_index, local_sequence_index])
                            ),
                            "imagined_q_members": json.dumps(imagined_q.astype(float).tolist()),
                            "direct_q_members": json.dumps(direct_q.astype(float).tolist()),
                            "calibrated_imagined_q_members": json.dumps(
                                calibrated_imagined_q.astype(float).tolist()
                            ),
                            "calibrated_direct_q_members": json.dumps(
                                calibrated_direct_q.astype(float).tolist()
                            ),
                            "imagined_leaf_value_members": json.dumps(
                                imagined_leaf_values.astype(float).tolist()
                            ),
                            "direct_leaf_value_members": json.dumps(direct_leaf_values.astype(float).tolist()),
                        }
                        records.append(record)
    return records


def attach_reachable_novelty(
        records: Sequence[Dict],
        occupancy: OccupancyIndex,
        continuation_sequences: np.ndarray,
        density_radius_bins: int,
        novelty_discount: float,
        batch_size: int,
) -> None:
    leaf_observations = np.asarray(
        [
            [record["true_leaf_position"], record["true_leaf_velocity"]]
            for record in records
        ],
        dtype=np.float32,
    )
    terminal_mask = np.asarray(
        [bool(record["true_terminated"]) for record in records], dtype=bool
    )
    reachable = evaluate_reachable_novelty(
        leaf_observations,
        continuation_sequences,
        occupancy,
        density_radius_bins=density_radius_bins,
        novelty_discount=novelty_discount,
        terminal_mask=terminal_mask,
        batch_size=batch_size,
    )
    for index, record in enumerate(records):
        density = int(record["true_leaf_density"])
        sequence_index = int(reachable["best_sequence_index"][index])
        record["real_leaf_novelty"] = float(1.0 / math.sqrt(density + 1.0))
        record["reachable_novelty_return_max"] = float(reachable["best_return"][index])
        record["reachable_novelty_return_mean"] = float(reachable["mean_return"][index])
        record["reachable_novelty_best_sequence"] = (
            ""
            if sequence_index < 0
            else "".join(str(int(action)) for action in continuation_sequences[sequence_index])
        )
        record["reachable_novelty_best_path_min_density"] = int(
            reachable["best_path_min_density"][index]
        )
        record["reachable_novelty_min_density"] = int(
            reachable["reachable_min_density"][index]
        )
        record["reachable_novelty_best_endpoint_position"] = float(
            reachable["best_endpoint_position"][index]
        )
        record["reachable_novelty_best_endpoint_velocity"] = float(
            reachable["best_endpoint_velocity"][index]
        )
        record["reachable_novelty_best_endpoint_density"] = int(
            reachable["best_endpoint_density"][index]
        )


def add_tree_local_path_metrics(records: Sequence[Dict]) -> None:
    by_root: Dict[int, List[int]] = {}
    for index, record in enumerate(records):
        by_root.setdefault(int(record["root_sample_index"]), []).append(index)

    for root_indices in by_root.values():
        imagined_q = np.stack(
            [np.asarray(json.loads(records[index]["imagined_q_members"]), dtype=np.float64) for index in root_indices]
        )
        direct_q = np.stack(
            [np.asarray(json.loads(records[index]["direct_q_members"]), dtype=np.float64) for index in root_indices]
        )
        nonterminal = np.asarray(
            [not bool(records[index]["true_terminated"]) for index in root_indices], dtype=bool
        )
        offset_mask = nonterminal if nonterminal.any() else np.ones(len(root_indices), dtype=bool)
        imagined_member_means = imagined_q[offset_mask].mean(axis=0)
        direct_member_means = direct_q[offset_mask].mean(axis=0)
        imagined_offsets = imagined_member_means - imagined_member_means.mean()
        direct_offsets = direct_member_means - direct_member_means.mean()
        imagined_centered = imagined_q - imagined_offsets
        direct_centered = direct_q - direct_offsets

        reference_candidates = np.flatnonzero(offset_mask)
        reference_local_index = int(
            reference_candidates[np.argmax(imagined_q[offset_mask].mean(axis=1))]
        )
        imagined_reference_differences = imagined_q - imagined_q[reference_local_index]
        direct_reference_differences = direct_q - direct_q[reference_local_index]
        imagined_pairwise_std = np.std(
            imagined_q[:, None, :] - imagined_q[None, :, :], axis=2
        )
        direct_pairwise_std = np.std(
            direct_q[:, None, :] - direct_q[None, :, :], axis=2
        )
        reference_sequence = str(records[root_indices[reference_local_index]]["sequence"])

        for local_index, record_index in enumerate(root_indices):
            record = records[record_index]
            record["tree_centered_imagined_qstd"] = float(
                np.std(imagined_centered[local_index])
            )
            record["tree_centered_direct_qstd"] = float(
                np.std(direct_centered[local_index])
            )
            record["best_mean_path_relative_imagined_qstd"] = float(
                np.std(imagined_reference_differences[local_index])
            )
            record["best_mean_path_relative_direct_qstd"] = float(
                np.std(direct_reference_differences[local_index])
            )
            record["mean_pairwise_imagined_qstd"] = float(
                np.mean(imagined_pairwise_std[local_index, offset_mask])
            )
            record["mean_pairwise_direct_qstd"] = float(
                np.mean(direct_pairwise_std[local_index, offset_mask])
            )
            record["tree_imagined_member_offsets"] = json.dumps(
                imagined_offsets.astype(float).tolist()
            )
            record["tree_direct_member_offsets"] = json.dumps(
                direct_offsets.astype(float).tolist()
            )
            record["tree_centered_imagined_q_members"] = json.dumps(
                imagined_centered[local_index].astype(float).tolist()
            )
            record["tree_centered_direct_q_members"] = json.dumps(
                direct_centered[local_index].astype(float).tolist()
            )
            record["tree_best_mean_reference_sequence"] = reference_sequence


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray, rank: bool = False) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    left = left[valid]
    right = right[valid]
    if len(left) < 2 or float(np.std(left)) < 1e-12 or float(np.std(right)) < 1e-12:
        return float("nan")
    if rank:
        left = _rankdata(left)
        right = _rankdata(right)
    return float(np.corrcoef(left, right)[0, 1])


def _metric_summary(records: Sequence[Dict], key: str) -> Dict[str, float]:
    values = np.asarray([float(record[key]) for record in records], dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def _example_record(
        record: Dict,
        score_beta: float,
        qstd_key: str = "calibrated_imagined_qstd",
) -> Dict:
    return {
        "root_sample_index": int(record["root_sample_index"]),
        "root_position": float(record["root_position"]),
        "root_velocity": float(record["root_velocity"]),
        "sequence": str(record["sequence"]),
        "true_leaf_position": float(record["true_leaf_position"]),
        "true_leaf_velocity": float(record["true_leaf_velocity"]),
        "true_leaf_density": int(record["true_leaf_density"]),
        "raw_imagined_qstd": float(record["raw_imagined_qstd"]),
        "raw_direct_qstd": float(record["raw_direct_qstd"]),
        "raw_qstd_excess": float(record["raw_qstd_excess"]),
        "calibrated_imagined_qstd": float(record["calibrated_imagined_qstd"]),
        "calibrated_direct_qstd": float(record["calibrated_direct_qstd"]),
        "calibrated_qstd_excess": float(record["calibrated_qstd_excess"]),
        "imagined_q_mean": float(record["imagined_q_mean"]),
        "optimistic_score": float(
            record["imagined_q_mean"] + score_beta * record[qstd_key]
        ),
        "imagined_direct_member_mae": float(record["imagined_direct_member_mae"]),
        "imagined_direct_latent_cosine_mean": float(record["imagined_direct_latent_cosine_mean"]),
        "imagined_q_members": json.loads(record["imagined_q_members"]),
        "direct_q_members": json.loads(record["direct_q_members"]),
    }


def _winner_summary(records: Sequence[Dict], score_beta: float, qstd_key: str) -> Dict:
    by_root: Dict[int, List[Dict]] = {}
    for record in records:
        by_root.setdefault(int(record["root_sample_index"]), []).append(record)

    qstd_winners = []
    optimistic_winners = []
    all_bucket_counts = Counter(record["true_leaf_density_bucket"] for record in records)
    all_count = max(1, len(records))
    all_bucket_fractions = {
        bucket: float(all_bucket_counts.get(bucket, 0) / all_count) for bucket in density_bucket_order()
    }
    for root_records in by_root.values():
        qstd_winners.append(max(root_records, key=lambda record: record[qstd_key]))
        optimistic_winners.append(
            max(
                root_records,
                key=lambda record: (
                    record["imagined_q_mean"]
                    + score_beta * record[qstd_key]
                ),
            )
        )

    def summarize_winners(winners: Sequence[Dict]) -> Dict:
        bucket_counts = Counter(record["true_leaf_density_bucket"] for record in winners)
        count = max(1, len(winners))
        bucket_fractions = {
            bucket: float(bucket_counts.get(bucket, 0) / count) for bucket in density_bucket_order()
        }
        return {
            "count": len(winners),
            "leaf_density_bucket_counts": dict(bucket_counts),
            "leaf_density_bucket_fractions": bucket_fractions,
            "leaf_density_fraction_lift_vs_all_sequences": {
                bucket: float(bucket_fractions[bucket] - all_bucket_fractions[bucket])
                for bucket in density_bucket_order()
            },
            "mean_leaf_density": float(np.mean([record["true_leaf_density"] for record in winners])),
            "mean_calibrated_imagined_qstd": float(
                np.mean([record["calibrated_imagined_qstd"] for record in winners])
            ),
            "mean_calibrated_direct_qstd": float(
                np.mean([record["calibrated_direct_qstd"] for record in winners])
            ),
            "mean_raw_imagined_qstd": float(
                np.mean([record["raw_imagined_qstd"] for record in winners])
            ),
            "mean_raw_direct_qstd": float(
                np.mean([record["raw_direct_qstd"] for record in winners])
            ),
            "mean_rollout_error": float(
                np.mean([record["imagined_direct_member_mae"] for record in winners])
            ),
            "examples": [_example_record(record, score_beta, qstd_key=qstd_key) for record in winners[:5]],
        }

    optimistic_winners = sorted(
        optimistic_winners,
        key=lambda record: record["imagined_q_mean"] + score_beta * record[qstd_key],
        reverse=True,
    )
    qstd_winners = sorted(
        qstd_winners, key=lambda record: record[qstd_key], reverse=True
    )
    return {
        "score_beta": float(score_beta),
        "qstd_key": qstd_key,
        "all_sequence_leaf_density_bucket_fractions": all_bucket_fractions,
        "max_qstd_sequence_per_root": summarize_winners(qstd_winners),
        "max_optimistic_score_sequence_per_root": summarize_winners(optimistic_winners),
    }


def _finite_distribution(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {"count": 0, "mean": float("nan"), "median": float("nan"), "p10": float("nan"), "p90": float("nan")}
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
    }


def summarize_tree_directional_novelty(records: Sequence[Dict]) -> Dict:
    score_keys = [
        "tree_centered_imagined_qstd",
        "best_mean_path_relative_imagined_qstd",
        "mean_pairwise_imagined_qstd",
        "tree_centered_direct_qstd",
        "best_mean_path_relative_direct_qstd",
        "mean_pairwise_direct_qstd",
    ]
    target_keys = ["real_leaf_novelty", "reachable_novelty_return_max"]
    by_root: Dict[int, List[Dict]] = {}
    for record in records:
        if not bool(record["true_terminated"]):
            by_root.setdefault(int(record["root_sample_index"]), []).append(record)

    summary = {}
    for target_key in target_keys:
        target_summary = {}
        for score_key in score_keys:
            correlations = []
            captures = []
            selected_targets = []
            top_decile_targets = []
            all_targets = []
            oracle_targets = []
            first_action_matches = []
            random_action_match_probabilities = []
            misses = []
            for root_sample_index, root_records in by_root.items():
                scores = np.asarray([record[score_key] for record in root_records], dtype=np.float64)
                targets = np.asarray([record[target_key] for record in root_records], dtype=np.float64)
                correlation = _correlation(scores, targets, rank=True)
                correlations.append(correlation)
                selected_index = int(np.argmax(scores))
                oracle_index = int(np.argmax(targets))
                top_count = max(1, int(math.ceil(0.1 * len(root_records))))
                top_indices = np.argsort(scores)[-top_count:]
                all_target = float(np.mean(targets))
                selected_target = float(targets[selected_index])
                oracle_target = float(targets[oracle_index])
                all_targets.append(all_target)
                selected_targets.append(selected_target)
                top_decile_targets.append(float(np.mean(targets[top_indices])))
                oracle_targets.append(oracle_target)
                denominator = oracle_target - all_target
                captures.append(
                    float((selected_target - all_target) / denominator)
                    if abs(denominator) > 1e-12
                    else float("nan")
                )

                score_by_action = []
                oracle_by_action = []
                for action in range(3):
                    action_mask = np.asarray(
                        [int(record["first_action"]) == action for record in root_records], dtype=bool
                    )
                    if action_mask.any():
                        score_by_action.append(float(np.max(scores[action_mask])))
                        oracle_by_action.append(float(np.max(targets[action_mask])))
                    else:
                        score_by_action.append(float("-inf"))
                        oracle_by_action.append(float("-inf"))
                selected_action = int(np.argmax(score_by_action))
                oracle_action_values = np.asarray(oracle_by_action, dtype=np.float64)
                available_actions = np.isfinite(oracle_action_values)
                oracle_best_value = float(np.max(oracle_action_values[available_actions]))
                oracle_best_actions = available_actions & np.isclose(
                    oracle_action_values, oracle_best_value, rtol=1e-7, atol=1e-9
                )
                first_action_matches.append(float(oracle_best_actions[selected_action]))
                random_action_match_probabilities.append(
                    float(np.sum(oracle_best_actions) / np.sum(available_actions))
                )
                misses.append(
                    {
                        "root_sample_index": int(root_sample_index),
                        "root_position": float(root_records[0]["root_position"]),
                        "root_velocity": float(root_records[0]["root_velocity"]),
                        "selected_sequence": str(root_records[selected_index]["sequence"]),
                        "selected_score": float(scores[selected_index]),
                        "selected_target": selected_target,
                        "selected_leaf_density": int(root_records[selected_index]["true_leaf_density"]),
                        "selected_reachable_sequence": str(
                            root_records[selected_index]["reachable_novelty_best_sequence"]
                        ),
                        "oracle_sequence": str(root_records[oracle_index]["sequence"]),
                        "oracle_score": float(scores[oracle_index]),
                        "oracle_target": oracle_target,
                        "oracle_leaf_density": int(root_records[oracle_index]["true_leaf_density"]),
                        "oracle_reachable_sequence": str(
                            root_records[oracle_index]["reachable_novelty_best_sequence"]
                        ),
                        "target_regret": float(oracle_target - selected_target),
                    }
                )

            target_summary[score_key] = {
                "within_root_spearman": _finite_distribution(correlations),
                "normalized_novelty_capture": _finite_distribution(captures),
                "mean_all_path_target": float(np.mean(all_targets)),
                "mean_top_decile_target": float(np.mean(top_decile_targets)),
                "mean_selected_path_target": float(np.mean(selected_targets)),
                "mean_oracle_path_target": float(np.mean(oracle_targets)),
                "first_action_match_fraction": float(np.mean(first_action_matches)),
                "random_action_match_fraction": float(
                    np.mean(random_action_match_probabilities)
                ),
                "largest_target_misses": sorted(
                    misses, key=lambda item: item["target_regret"], reverse=True
                )[:5],
            }
        summary[target_key] = target_summary
    return summary


def summarize_records(records: Sequence[Dict], score_beta: float = 2.0) -> Dict:
    metric_keys = [
        "raw_imagined_qstd",
        "raw_direct_qstd",
        "raw_qstd_excess",
        "calibrated_imagined_qstd",
        "calibrated_direct_qstd",
        "calibrated_qstd_excess",
        "imagined_direct_member_mae",
        "imagined_direct_leaf_value_mae",
        "imagined_direct_latent_cosine_mean",
        "imagined_direct_latent_rms_mean",
        "predicted_reward_return_mae",
        "tree_centered_imagined_qstd",
        "tree_centered_direct_qstd",
        "best_mean_path_relative_imagined_qstd",
        "mean_pairwise_imagined_qstd",
        "real_leaf_novelty",
        "reachable_novelty_return_max",
    ]
    density = np.asarray([record["true_leaf_density"] for record in records], dtype=np.float64)
    calibrated_qstd = np.asarray(
        [record["calibrated_imagined_qstd"] for record in records], dtype=np.float64
    )
    raw_qstd = np.asarray([record["raw_imagined_qstd"] for record in records], dtype=np.float64)
    raw_direct_qstd = np.asarray([record["raw_direct_qstd"] for record in records], dtype=np.float64)
    direct_qstd = np.asarray([record["calibrated_direct_qstd"] for record in records], dtype=np.float64)
    rollout_error = np.asarray(
        [record["imagined_direct_member_mae"] for record in records], dtype=np.float64
    )
    qstd_threshold = float(np.percentile(calibrated_qstd, 90))
    direct_low_threshold = float(np.median(direct_qstd))
    high_qstd = calibrated_qstd >= qstd_threshold
    rollout_false_uncertainty = high_qstd & (direct_qstd <= direct_low_threshold)

    groups = {}
    for bucket in density_bucket_order():
        bucket_records = [record for record in records if record["true_leaf_density_bucket"] == bucket]
        if bucket_records:
            groups[bucket] = {
                "count": len(bucket_records),
                "metrics": {key: _metric_summary(bucket_records, key) for key in metric_keys},
            }

    dense_records = [record for record in records if record["true_leaf_density"] > 50]
    sparse_records = [record for record in records if record["true_leaf_density"] <= 5]
    rollout_inflated = sorted(records, key=lambda record: record["calibrated_qstd_excess"], reverse=True)
    dense_high_qstd = sorted(
        dense_records, key=lambda record: record["calibrated_imagined_qstd"], reverse=True
    )
    sparse_low_qstd = sorted(
        sparse_records, key=lambda record: record["calibrated_imagined_qstd"]
    )

    return {
        "sequence_count": len(records),
        "aggregate_metrics": {key: _metric_summary(records, key) for key in metric_keys},
        "density_groups": groups,
        "correlations": {
            "pearson_log1p_density_vs_raw_imagined_qstd": _correlation(
                np.log1p(density), raw_qstd
            ),
            "spearman_density_vs_raw_imagined_qstd": _correlation(
                density, raw_qstd, rank=True
            ),
            "pearson_raw_imagined_vs_direct_qstd": _correlation(raw_qstd, raw_direct_qstd),
            "pearson_raw_imagined_qstd_vs_rollout_error": _correlation(raw_qstd, rollout_error),
            "pearson_log1p_density_vs_calibrated_imagined_qstd": _correlation(
                np.log1p(density), calibrated_qstd
            ),
            "spearman_density_vs_calibrated_imagined_qstd": _correlation(
                density, calibrated_qstd, rank=True
            ),
            "pearson_calibrated_imagined_vs_direct_qstd": _correlation(
                calibrated_qstd, direct_qstd
            ),
            "pearson_calibrated_imagined_qstd_vs_rollout_error": _correlation(
                calibrated_qstd, rollout_error
            ),
        },
        "high_qstd_diagnostics": {
            "calibrated_qstd_p90_threshold": qstd_threshold,
            "direct_qstd_median_threshold": direct_low_threshold,
            "top_decile_count": int(high_qstd.sum()),
            "top_decile_dense_fraction": float(
                np.mean(density[high_qstd] > 50) if high_qstd.any() else 0.0
            ),
            "top_decile_unseen_or_sparse_fraction": float(
                np.mean(density[high_qstd] <= 5) if high_qstd.any() else 0.0
            ),
            "top_decile_low_direct_qstd_fraction": float(
                np.mean(rollout_false_uncertainty[high_qstd]) if high_qstd.any() else 0.0
            ),
        },
        "root_sequence_winners": {
            "raw": _winner_summary(records, score_beta, "raw_imagined_qstd"),
            "replay_offset_calibrated": _winner_summary(
                records, score_beta, "calibrated_imagined_qstd"
            ),
        },
        "examples": {
            "largest_rollout_qstd_inflation": [
                _example_record(record, score_beta) for record in rollout_inflated[:5]
            ],
            "highest_qstd_dense_real_leaves": [
                _example_record(record, score_beta) for record in dense_high_qstd[:5]
            ],
            "lowest_qstd_unseen_or_sparse_real_leaves": [
                _example_record(record, score_beta) for record in sparse_low_qstd[:5]
            ],
        },
        "tree_directional_novelty": summarize_tree_directional_novelty(records),
    }


def _write_csv(path: Path, records: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def _print_summary(summary: Dict) -> None:
    aggregate = summary["aggregate_metrics"]
    print("\nAggregate fixed-horizon calibration")
    print("metric                                      mean       median        p90")
    for key in [
        "raw_imagined_qstd",
        "raw_direct_qstd",
        "calibrated_imagined_qstd",
        "calibrated_direct_qstd",
        "calibrated_qstd_excess",
        "imagined_direct_member_mae",
        "imagined_direct_latent_cosine_mean",
    ]:
        stats = aggregate[key]
        print(f"{key:40s} {stats['mean']:10.3f} {stats['median']:12.3f} {stats['p90']:10.3f}")

    print("\nBy exact real-leaf replay density")
    print("bucket              count   imagined Qstd   direct Qstd   rollout MAE")
    for bucket in density_bucket_order():
        group = summary["density_groups"].get(bucket)
        if group is None:
            continue
        metrics = group["metrics"]
        print(
            f"{bucket:18s} {group['count']:7d} "
            f"{metrics['calibrated_imagined_qstd']['mean']:15.3f} "
            f"{metrics['calibrated_direct_qstd']['mean']:13.3f} "
            f"{metrics['imagined_direct_member_mae']['mean']:13.3f}"
        )

    print("\nCorrelations")
    for key, value in summary["correlations"].items():
        print(f"  {key}: {value:+.4f}")
    print("\nHigh-Qstd top decile")
    for key, value in summary["high_qstd_diagnostics"].items():
        print(f"  {key}: {value}")

    print("\nPer-root winning sequences")
    for mode, winner_summary in summary["root_sequence_winners"].items():
        print(
            f"  {mode} all-sequence density fractions: "
            f"{winner_summary['all_sequence_leaf_density_bucket_fractions']}"
        )
        for key in ["max_qstd_sequence_per_root", "max_optimistic_score_sequence_per_root"]:
            values = winner_summary[key]
            metric_prefix = "raw" if mode == "raw" else "calibrated"
            print(
                f"  {mode}/{key}: mean density={values['mean_leaf_density']:.1f}, "
                f"imagined/direct Qstd={values[f'mean_{metric_prefix}_imagined_qstd']:.2f}/"
                f"{values[f'mean_{metric_prefix}_direct_qstd']:.2f}, "
                f"density fractions={values['leaf_density_bucket_fractions']}, "
                f"lift={values['leaf_density_fraction_lift_vs_all_sequences']}"
            )

    print("\nWithin-tree directional novelty")
    directional = summary["tree_directional_novelty"]
    for target_key in ["real_leaf_novelty", "reachable_novelty_return_max"]:
        print(f"  target={target_key}")
        for score_key in [
            "tree_centered_imagined_qstd",
            "best_mean_path_relative_imagined_qstd",
            "mean_pairwise_imagined_qstd",
        ]:
            values = directional[target_key][score_key]
            spearman = values["within_root_spearman"]
            capture = values["normalized_novelty_capture"]
            print(
                f"    {score_key}: spearman mean/median={spearman['mean']:+.3f}/"
                f"{spearman['median']:+.3f}, selected/all/oracle="
                f"{values['mean_selected_path_target']:.3f}/"
                f"{values['mean_all_path_target']:.3f}/"
                f"{values['mean_oracle_path_target']:.3f}, "
                f"capture={capture['mean']:+.3f}, "
                f"first_action_match={values['first_action_match_fraction']:.3f} "
                f"(random={values['random_action_match_fraction']:.3f})"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare fixed-horizon ensemble returns from imagined MuZero leaves with values obtained by "
            "encoding the exact MountainCar leaves."
        )
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--checkpoint", default="ckpt_best.pth.tar")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--sample-states", type=int, default=64)
    parser.add_argument("--calibration-samples", type=int, default=4096)
    parser.add_argument("--state-batch-size", type=int, default=8)
    parser.add_argument("--sequence-batch-size", type=int, default=243)
    parser.add_argument("--density-radius-bins", type=int, default=1)
    parser.add_argument("--reachable-horizon", type=int, default=5)
    parser.add_argument("--reachable-novelty-discount", type=float, default=1.0)
    parser.add_argument("--reachable-batch-size", type=int, default=512)
    parser.add_argument(
        "--score-beta",
        type=float,
        default=2.0,
        help="Beta used to audit the per-root winner of mean_Q + beta * calibrated_Qstd.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-prefix", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.horizon <= 0:
        raise ValueError("--horizon must be positive")
    if args.reachable_horizon <= 0:
        raise ValueError("--reachable-horizon must be positive")
    device = _resolve_device(args.device)
    config = _load_config(args.run_dir / "total_config.py")
    model_config = dict(config["policy"]["model"])
    discount = float(config["policy"]["discount_factor"])
    checkpoint_path = args.run_dir / "ckpt" / args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    max_train_iter = int(checkpoint.get("last_iter", -1))
    max_envstep = int(checkpoint.get("last_step", -1))

    root_csv = args.run_dir / "diagnostics" / "root_action_terms.csv"
    roots = load_real_root_states(root_csv, max_train_iter=max_train_iter, max_envstep=max_envstep)
    occupancy = OccupancyIndex(root.observation for root in roots)
    sampled_roots = stratified_sample_roots(
        roots,
        occupancy,
        sample_count=args.sample_states,
        radius_bins=args.density_radius_bins,
        seed=args.seed,
    )

    model = EnsembleMuZeroModelMLP(**model_config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    ensemble_size = int(model_config["ensemble_size"])
    value_decoder = _decode_handle(model_config, "value_support_range", device)
    reward_decoder = _decode_handle(model_config, "reward_support_range", device)

    rng = np.random.default_rng(args.seed)
    calibration_count = min(max(1, args.calibration_samples), len(roots))
    calibration_indices = rng.choice(len(roots), size=calibration_count, replace=False)
    calibration_observations = np.asarray(
        [roots[int(index)].observation for index in calibration_indices], dtype=np.float32
    )
    value_offsets, member_value_means = replay_value_offsets(
        model,
        calibration_observations,
        ensemble_size,
        value_decoder,
        device,
        inference_batch_size=max(1, args.state_batch_size * 32),
    )

    sequences = np.asarray(
        list(itertools.product(range(int(model_config["action_space_size"])), repeat=args.horizon)),
        dtype=np.int64,
    )
    reachable_sequences = np.asarray(
        list(
            itertools.product(
                range(int(model_config["action_space_size"])), repeat=args.reachable_horizon
            )
        ),
        dtype=np.int64,
    )
    print(f"run: {args.run_dir}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"checkpoint train_iter/envstep: {max_train_iter}/{max_envstep}")
    print(f"device: {device}")
    print(f"real logged states at checkpoint: {len(roots)}")
    print(f"sampled roots: {len(sampled_roots)}")
    print(f"horizon/sequences: {args.horizon}/{len(sequences)}")
    print(
        f"reachable horizon/sequences: {args.reachable_horizon}/{len(reachable_sequences)}"
    )
    print(f"replay value member means: {member_value_means.tolist()}")
    print(f"replay value offsets: {value_offsets.tolist()}")

    records = evaluate_fixed_horizon(
        model=model,
        roots=sampled_roots,
        sequences=sequences,
        occupancy=occupancy,
        value_decoder=value_decoder,
        reward_decoder=reward_decoder,
        value_offsets=value_offsets,
        discount=discount,
        density_radius_bins=args.density_radius_bins,
        state_batch_size=max(1, args.state_batch_size),
        sequence_batch_size=max(1, args.sequence_batch_size),
        device=device,
    )
    add_tree_local_path_metrics(records)
    attach_reachable_novelty(
        records,
        occupancy,
        reachable_sequences,
        density_radius_bins=args.density_radius_bins,
        novelty_discount=args.reachable_novelty_discount,
        batch_size=max(1, args.reachable_batch_size),
    )
    summary = summarize_records(records, score_beta=args.score_beta)
    summary["metadata"] = {
        "run_dir": str(args.run_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_train_iter": max_train_iter,
        "checkpoint_envstep": max_envstep,
        "device": str(device),
        "discount": discount,
        "horizon": args.horizon,
        "reachable_horizon": args.reachable_horizon,
        "reachable_novelty_discount": args.reachable_novelty_discount,
        "logged_state_count": len(roots),
        "sampled_root_count": len(sampled_roots),
        "sequences_per_root": len(sequences),
        "density_radius_bins": args.density_radius_bins,
        "score_beta": args.score_beta,
        "position_bin_width": POSITION_BIN_WIDTH,
        "velocity_bin_width": VELOCITY_BIN_WIDTH,
        "calibration_sample_count": calibration_count,
        "replay_value_member_means": member_value_means.astype(float).tolist(),
        "replay_value_offsets": value_offsets.astype(float).tolist(),
    }

    checkpoint_stem = Path(args.checkpoint).name.replace(".pth.tar", "")
    output_prefix = args.output_prefix or (
        args.run_dir / "diagnostics" /
        f"fixed_horizon_calibration_{checkpoint_stem}_h{args.horizon}_r{args.reachable_horizon}"
    )
    csv_path = Path(f"{output_prefix}.csv")
    json_path = Path(f"{output_prefix}.json")
    _write_csv(csv_path, records)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _print_summary(summary)
    print(f"\nPer-sequence records: {csv_path}")
    print(f"Summary: {json_path}")


if __name__ == "__main__":
    main()
