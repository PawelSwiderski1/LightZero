"""Standalone MountainCar RND signal diagnostic.

This script replays saved real collection episodes through only an RND target
and predictor. It deliberately does not construct or train MuZero. The replay
order preserves collection phases so RND can be compared against a
phase-frozen, explicit state-density reward.
"""

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict
from scipy.spatial import cKDTree

from lzero.entry.state_density_novelty import StateDensityNoveltyTracker
from lzero.policy.ensemble_muzero import _LatentRND


POSITION_LOW = -1.2
POSITION_HIGH = 0.6
VELOCITY_LOW = -0.07
VELOCITY_HIGH = 0.07
EPISODE_PATTERN = re.compile(
    r"intrinsic-(?P<collection_id>\d+)_.*_phase-(?P<phase>\d+)_env-(?P<env>\d+)_episode-(?P<episode>\d+)\.npz$"
)


@dataclass
class EpisodeData:
    path: Path
    collection_id: int
    phase: int
    env_id: int
    episode_id: int
    observations: np.ndarray
    actions: np.ndarray
    density_rewards: Optional[np.ndarray] = None
    effective_counts: Optional[np.ndarray] = None
    nearest_prior_distances: Optional[np.ndarray] = None

    @property
    def next_observations(self) -> np.ndarray:
        transition_count = min(len(self.actions), max(0, len(self.observations) - 1))
        return self.observations[1:transition_count + 1, :2]


@dataclass
class PhaseData:
    phase: int
    episodes: List[EpisodeData]
    observations: np.ndarray
    density_rewards: np.ndarray
    effective_counts: np.ndarray
    nearest_prior_distances: np.ndarray
    density_snapshot_counts: np.ndarray


def parse_number_list(value: str, value_type=float) -> List:
    return [value_type(item.strip()) for item in value.split(',') if item.strip()]


def resolve_episode_dir(input_path: Path) -> Path:
    candidates = [
        input_path,
        input_path / 'intrinsic_episodes',
        input_path / 'diagnostics' / 'intrinsic_episodes',
    ]
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob('*.npz')):
            return candidate
    raise FileNotFoundError(
        f"Could not find saved .npz episodes under {input_path}. Expected diagnostics/intrinsic_episodes."
    )


def load_episodes(input_path: Path, max_phase: Optional[int] = None) -> List[EpisodeData]:
    episode_dir = resolve_episode_dir(input_path)
    episodes = []
    for path in episode_dir.glob('*.npz'):
        match = EPISODE_PATTERN.match(path.name)
        if match is None:
            continue
        phase = int(match.group('phase'))
        if max_phase is not None and phase > max_phase:
            continue
        with np.load(path, allow_pickle=False) as data:
            observations = np.asarray(data['obs'], dtype=np.float32)
            actions = np.asarray(data['actions'], dtype=np.int64).reshape(-1)
        if observations.ndim != 2 or observations.shape[1] < 2:
            raise ValueError(f"Episode {path} has invalid observation shape {observations.shape}.")
        episodes.append(
            EpisodeData(
                path=path,
                collection_id=int(match.group('collection_id')),
                phase=phase,
                env_id=int(match.group('env')),
                episode_id=int(match.group('episode')),
                observations=observations,
                actions=actions,
            )
        )
    episodes.sort(key=lambda episode: (episode.phase, episode.collection_id))
    if not episodes:
        raise FileNotFoundError(f"No matching intrinsic episode files found in {episode_dir}.")
    return episodes


def normalize_physical_observations(observations: np.ndarray) -> np.ndarray:
    observations = np.asarray(observations, dtype=np.float32)[..., :2]
    low = np.asarray([POSITION_LOW, VELOCITY_LOW], dtype=np.float32)
    high = np.asarray([POSITION_HIGH, VELOCITY_HIGH], dtype=np.float32)
    return np.clip(2.0 * (observations - low) / (high - low) - 1.0, -1.0, 1.0)


def _density_config(args: argparse.Namespace) -> EasyDict:
    return EasyDict(
        dict(
            state_density_position_min=POSITION_LOW,
            state_density_position_max=POSITION_HIGH,
            state_density_velocity_min=VELOCITY_LOW,
            state_density_velocity_max=VELOCITY_HIGH,
            state_density_position_bin_width=args.position_bin_width,
            state_density_velocity_bin_width=args.velocity_bin_width,
            state_density_neighbor_radius=args.neighbor_radius,
            state_density_snapshot_update_interval=1,
            state_density_reward_scale=1.0,
            state_density_reward_clip=1.0,
        )
    )


def build_phase_data(episodes: Sequence[EpisodeData], args: argparse.Namespace) -> List[PhaseData]:
    tracker = StateDensityNoveltyTracker(_density_config(args))
    episodes_by_phase: Dict[int, List[EpisodeData]] = {}
    for episode in episodes:
        episodes_by_phase.setdefault(episode.phase, []).append(episode)

    phase_data = []
    prior_observations: List[np.ndarray] = []
    for phase in sorted(episodes_by_phase):
        phase_episodes = episodes_by_phase[phase]
        tracker.begin_phase(phase)
        density_snapshot_counts = tracker.snapshot_counts.copy()
        nearest_tree = None
        if prior_observations:
            nearest_tree = cKDTree(normalize_physical_observations(np.concatenate(prior_observations, axis=0)))

        observation_parts = []
        reward_parts = []
        count_parts = []
        distance_parts = []
        for episode in phase_episodes:
            observations = episode.next_observations
            rewards, counts = tracker.rewards(observations)
            if nearest_tree is None:
                distances = np.full(len(observations), np.nan, dtype=np.float32)
            else:
                distances = nearest_tree.query(normalize_physical_observations(observations), k=1)[0].astype(np.float32)
            episode.density_rewards = rewards
            episode.effective_counts = counts
            episode.nearest_prior_distances = distances
            observation_parts.append(observations)
            reward_parts.append(rewards)
            count_parts.append(counts)
            distance_parts.append(distances)

        observations = np.concatenate(observation_parts, axis=0)
        density_rewards = np.concatenate(reward_parts, axis=0)
        effective_counts = np.concatenate(count_parts, axis=0)
        nearest_distances = np.concatenate(distance_parts, axis=0)
        phase_data.append(
            PhaseData(
                phase=phase,
                episodes=phase_episodes,
                observations=observations,
                density_rewards=density_rewards,
                effective_counts=effective_counts,
                nearest_prior_distances=nearest_distances,
                density_snapshot_counts=density_snapshot_counts,
            )
        )
        tracker.update_live_counts(observations)
        prior_observations.append(observations)
    return phase_data


class RNDSignalModel:
    def __init__(
            self,
            mode: str,
            seed: int,
            hidden_dim: int,
            feature_dim: int,
            learning_rate: float,
            target_first_layer_gain: float,
            device: torch.device,
    ) -> None:
        if mode not in {'fixed_bounds', 'legacy_layer_norm'}:
            raise ValueError(f"Unknown RND preprocessing mode: {mode}")
        self.mode = mode
        self.seed = int(seed)
        self.device = device
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        input_low = (POSITION_LOW, VELOCITY_LOW) if mode == 'fixed_bounds' else None
        input_high = (POSITION_HIGH, VELOCITY_HIGH) if mode == 'fixed_bounds' else None
        self.model = _LatentRND(
            input_dim=2,
            hidden_dim=hidden_dim,
            feature_dim=feature_dim,
            input_low=input_low,
            input_high=input_high,
        ).to(device)
        with torch.no_grad():
            self.model.target[0].weight.mul_(float(target_first_layer_gain))
        self.optimizer = torch.optim.Adam(self.model.predictor.parameters(), lr=learning_rate)

    def preprocess(self, observations: torch.Tensor) -> torch.Tensor:
        flat = observations.reshape(observations.shape[0], -1).float()
        if self.mode == 'legacy_layer_norm':
            return F.layer_norm(flat, flat.shape[1:])
        return self.model._flatten(flat)

    def _forward(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        prepared = self.preprocess(observations)
        prediction = self.model.predictor(prepared)
        with torch.no_grad():
            target = self.model.target(prepared)
        return prediction, target

    def errors(self, observations: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        outputs = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(observations), batch_size):
                batch = torch.as_tensor(
                    observations[start:start + batch_size], dtype=torch.float32, device=self.device
                )
                prediction, target = self._forward(batch)
                outputs.append((prediction - target).pow(2).mean(dim=1).cpu().numpy())
        if not outputs:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(outputs).astype(np.float32)

    def target_features(self, observations: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        outputs = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(observations), batch_size):
                batch = torch.as_tensor(
                    observations[start:start + batch_size], dtype=torch.float32, device=self.device
                )
                prepared = self.preprocess(batch)
                outputs.append(self.model.target(prepared).cpu().numpy())
        return np.concatenate(outputs) if outputs else np.zeros((0, 0), dtype=np.float32)

    def train_fresh(
            self,
            observations: np.ndarray,
            update_proportion: float,
            candidate_batch_size: int,
            train_passes: int,
            random_seed: int,
    ) -> Dict[str, float]:
        generator = torch.Generator(device='cpu')
        generator.manual_seed(int(random_seed))
        observations_tensor = torch.as_tensor(observations, dtype=torch.float32)
        losses = []
        selected_total = 0
        update_count = 0
        self.model.train()
        for _ in range(max(1, int(train_passes))):
            permutation = torch.randperm(len(observations_tensor), generator=generator)
            for start in range(0, len(permutation), candidate_batch_size):
                candidate_indices = permutation[start:start + candidate_batch_size]
                keep = torch.rand(len(candidate_indices), generator=generator) < update_proportion
                if update_proportion > 0.0 and not bool(keep.any()):
                    keep[torch.randint(len(keep), (1,), generator=generator)] = True
                selected_indices = candidate_indices[keep]
                if selected_indices.numel() == 0:
                    continue
                batch = observations_tensor.index_select(0, selected_indices).to(self.device)
                prediction, target = self._forward(batch)
                loss = (prediction - target).pow(2).mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(float(loss.detach().cpu().item()))
                selected_total += int(selected_indices.numel())
                update_count += 1
        return {
            'train_selected_count': float(selected_total),
            'train_update_count': float(update_count),
            'train_loss_mean': float(np.mean(losses)) if losses else 0.0,
        }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind='mergesort')
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def pearson_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    left = left[valid]
    right = right[valid]
    if len(left) < 2 or float(left.std()) <= 1e-12 or float(right.std()) <= 1e-12:
        return float('nan')
    return float(np.corrcoef(left, right)[0, 1])


def spearman_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 2:
        return float('nan')
    return pearson_correlation(_average_ranks(left[valid]), _average_ranks(right[valid]))


def top_fraction_overlap(left: np.ndarray, right: np.ndarray, fraction: float = 0.1) -> float:
    left = np.asarray(left)
    right = np.asarray(right)
    if len(left) == 0 or len(left) != len(right) or left.std() <= 1e-12 or right.std() <= 1e-12:
        return float('nan')
    count = max(1, int(math.ceil(len(left) * fraction)))
    left_top = set(np.argpartition(left, -count)[-count:].tolist())
    right_top = set(np.argpartition(right, -count)[-count:].tolist())
    return float(len(left_top & right_top) / count)


def safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) <= 1e-12:
        return float('nan')
    return float(numerator / denominator)


def summarize_phase(
        model: RNDSignalModel,
        phase_data: PhaseData,
        errors: np.ndarray,
        gain: float,
) -> Tuple[Dict[str, float], List[Dict], List[Dict]]:
    counts = phase_data.effective_counts
    density_rewards = phase_data.density_rewards
    distances = phase_data.nearest_prior_distances
    unseen = counts <= 0.0
    dense = counts >= 100.0
    unseen_mean = float(errors[unseen].mean()) if bool(unseen.any()) else float('nan')
    dense_mean = float(errors[dense].mean()) if bool(dense.any()) else float('nan')

    episode_rows = []
    offset = 0
    for episode in phase_data.episodes:
        length = len(episode.next_observations)
        episode_errors = errors[offset:offset + length]
        offset += length
        observations = episode.next_observations
        episode_rows.append(
            {
                'mode': model.mode,
                'target_first_layer_gain': gain,
                'seed': model.seed,
                'phase': phase_data.phase,
                'collection_id': episode.collection_id,
                'env_id': episode.env_id,
                'episode_id': episode.episode_id,
                'transition_count': length,
                'rnd_error_sum': float(episode_errors.sum()),
                'rnd_error_mean': float(episode_errors.mean()),
                'rnd_error_max': float(episode_errors.max()),
                'density_reward_sum': float(episode.density_rewards.sum()),
                'density_reward_mean': float(episode.density_rewards.mean()),
                'effective_count_mean': float(episode.effective_counts.mean()),
                'max_position': float(observations[:, 0].max()),
                'min_position': float(observations[:, 0].min()),
                'max_abs_velocity': float(np.abs(observations[:, 1]).max()),
                'reached_goal': float(bool(np.any(observations[:, 0] >= 0.5))),
            }
        )

    episode_rnd_sums = np.asarray([row['rnd_error_sum'] for row in episode_rows])
    episode_density_sums = np.asarray([row['density_reward_sum'] for row in episode_rows])
    episode_max_positions = np.asarray([row['max_position'] for row in episode_rows])
    phase_row = {
        'mode': model.mode,
        'target_first_layer_gain': gain,
        'seed': model.seed,
        'phase': phase_data.phase,
        'transition_count': len(errors),
        'rnd_error_mean': float(errors.mean()),
        'rnd_error_std': float(errors.std()),
        'rnd_error_p50': float(np.percentile(errors, 50)),
        'rnd_error_p90': float(np.percentile(errors, 90)),
        'density_reward_mean': float(density_rewards.mean()),
        'effective_count_mean': float(counts.mean()),
        'rnd_density_pearson': pearson_correlation(errors, density_rewards),
        'rnd_density_spearman': spearman_correlation(errors, density_rewards),
        'rnd_count_spearman': spearman_correlation(errors, counts),
        'rnd_nearest_distance_spearman': spearman_correlation(errors, distances),
        'top10_density_overlap': top_fraction_overlap(errors, density_rewards, 0.1),
        'unseen_fraction': float(unseen.mean()),
        'dense_fraction': float(dense.mean()),
        'unseen_rnd_error_mean': unseen_mean,
        'dense_rnd_error_mean': dense_mean,
        'unseen_dense_error_ratio': safe_ratio(unseen_mean, dense_mean),
        'episode_rnd_density_sum_spearman': spearman_correlation(episode_rnd_sums, episode_density_sums),
        'episode_rnd_max_position_spearman': spearman_correlation(episode_rnd_sums, episode_max_positions),
    }

    bucket_rows = []
    bucket_specs = [
        ('count_0', counts <= 0),
        ('count_1_9', (counts >= 1) & (counts < 10)),
        ('count_10_99', (counts >= 10) & (counts < 100)),
        ('count_100_999', (counts >= 100) & (counts < 1000)),
        ('count_1000_plus', counts >= 1000),
    ]
    for bucket_name, mask in bucket_specs:
        if not bool(mask.any()):
            continue
        bucket_rows.append(
            {
                'mode': model.mode,
                'target_first_layer_gain': gain,
                'seed': model.seed,
                'phase': phase_data.phase,
                'bucket': bucket_name,
                'sample_count': int(mask.sum()),
                'rnd_error_mean': float(errors[mask].mean()),
                'rnd_error_p90': float(np.percentile(errors[mask], 90)),
                'density_reward_mean': float(density_rewards[mask].mean()),
                'nearest_prior_distance_mean': float(np.nanmean(distances[mask]))
                if bool(np.isfinite(distances[mask]).any()) else float('nan'),
            }
        )
    return phase_row, episode_rows, bucket_rows


def mountain_car_step(state: np.ndarray, action: int) -> np.ndarray:
    position, velocity = float(state[0]), float(state[1])
    velocity += (int(action) - 1) * 0.001 - math.cos(3.0 * position) * 0.0025
    velocity = float(np.clip(velocity, VELOCITY_LOW, VELOCITY_HIGH))
    position = float(np.clip(position + velocity, POSITION_LOW, POSITION_HIGH))
    if position <= POSITION_LOW and velocity < 0.0:
        velocity = 0.0
    return np.asarray([position, velocity], dtype=np.float32)


def scripted_trajectory(start_position: float, policy_name: str, max_steps: int = 200) -> np.ndarray:
    state = np.asarray([start_position, 0.0], dtype=np.float32)
    observations = []
    for timestep in range(max_steps):
        if policy_name == 'noop':
            action = 1
        elif policy_name == 'alternating':
            action = 0 if timestep % 2 == 0 else 2
        elif policy_name == 'all_left':
            action = 0
        elif policy_name == 'all_right':
            action = 2
        elif policy_name == 'coordinated_swing':
            action = 2 if state[1] >= 0.0 else 0
        else:
            raise ValueError(f"Unknown scripted policy: {policy_name}")
        state = mountain_car_step(state, action)
        observations.append(state.copy())
        if state[0] >= 0.5:
            break
    return np.asarray(observations, dtype=np.float32)


def density_rewards_for_snapshot(
        observations: np.ndarray,
        snapshot_counts: np.ndarray,
        args: argparse.Namespace,
) -> np.ndarray:
    tracker = StateDensityNoveltyTracker(_density_config(args))
    tracker.snapshot_counts = snapshot_counts
    return tracker.rewards(observations)[0]


def trajectory_rows(
        model: RNDSignalModel,
        phase_data: PhaseData,
        gain: float,
        starts: Sequence[float],
        args: argparse.Namespace,
) -> List[Dict]:
    rows = []
    policies = ['noop', 'alternating', 'all_left', 'all_right', 'coordinated_swing']
    for start_position in starts:
        for policy_name in policies:
            observations = scripted_trajectory(start_position, policy_name)
            errors = model.errors(observations)
            density_rewards = density_rewards_for_snapshot(
                observations, phase_data.density_snapshot_counts, args
            )
            rows.append(
                {
                    'mode': model.mode,
                    'target_first_layer_gain': gain,
                    'seed': model.seed,
                    'phase': phase_data.phase,
                    'start_position': start_position,
                    'policy': policy_name,
                    'steps': len(observations),
                    'reached_goal': float(bool(np.any(observations[:, 0] >= 0.5))),
                    'max_position': float(observations[:, 0].max()),
                    'min_position': float(observations[:, 0].min()),
                    'max_abs_velocity': float(np.abs(observations[:, 1]).max()),
                    'rnd_error_sum': float(errors.sum()),
                    'rnd_error_mean': float(errors.mean()),
                    'rnd_error_max': float(errors.max()),
                    'density_reward_sum': float(density_rewards.sum()),
                    'density_reward_mean': float(density_rewards.mean()),
                }
            )
    return rows


def make_grid(position_points: int, velocity_points: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.linspace(POSITION_LOW, POSITION_HIGH, position_points, dtype=np.float32)
    velocities = np.linspace(VELOCITY_LOW, VELOCITY_HIGH, velocity_points, dtype=np.float32)
    position_grid, velocity_grid = np.meshgrid(positions, velocities)
    observations = np.stack([position_grid.reshape(-1), velocity_grid.reshape(-1)], axis=1)
    return observations, positions, velocities


def target_smoothness_row(
        model: RNDSignalModel,
        gain: float,
        grid_observations: np.ndarray,
        pair_count: int,
) -> Dict:
    features = model.target_features(grid_observations)
    rng = np.random.default_rng(model.seed + 9173)
    left_indices = rng.integers(0, len(grid_observations), size=pair_count)
    right_indices = rng.integers(0, len(grid_observations), size=pair_count)
    normalized = normalize_physical_observations(grid_observations)
    physical_distances = np.linalg.norm(normalized[left_indices] - normalized[right_indices], axis=1)
    feature_distances = np.linalg.norm(features[left_indices] - features[right_indices], axis=1)
    near = physical_distances <= 0.1
    far = physical_distances >= 1.0
    near_mean = float(feature_distances[near].mean()) if bool(near.any()) else float('nan')
    far_mean = float(feature_distances[far].mean()) if bool(far.any()) else float('nan')
    prepared = model.preprocess(torch.as_tensor(grid_observations, device=model.device)).detach().cpu().numpy()
    rounded_unique = len(np.unique(np.round(prepared, decimals=4), axis=0))
    return {
        'mode': model.mode,
        'target_first_layer_gain': gain,
        'seed': model.seed,
        'grid_state_count': len(grid_observations),
        'rounded_preprocessed_unique_count': rounded_unique,
        'target_feature_std_mean': float(features.std(axis=0).mean()),
        'physical_target_distance_pearson': pearson_correlation(physical_distances, feature_distances),
        'physical_target_distance_spearman': spearman_correlation(physical_distances, feature_distances),
        'near_target_distance_mean': near_mean,
        'far_target_distance_mean': far_mean,
        'far_near_target_distance_ratio': safe_ratio(far_mean, near_mean),
    }


def write_csv(path: Path, rows: Sequence[Dict]) -> None:
    if not rows:
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_phase_metrics(rows: Sequence[Dict]) -> List[Dict]:
    metric_names = [
        'rnd_error_mean',
        'post_train_rnd_error_mean',
        'post_pre_error_ratio',
        'rnd_density_pearson',
        'rnd_density_spearman',
        'rnd_count_spearman',
        'rnd_nearest_distance_spearman',
        'top10_density_overlap',
        'unseen_rnd_error_mean',
        'dense_rnd_error_mean',
        'unseen_dense_error_ratio',
        'episode_rnd_density_sum_spearman',
        'episode_rnd_max_position_spearman',
    ]
    grouped: Dict[Tuple, List[Dict]] = {}
    for row in rows:
        grouped.setdefault(
            (row['mode'], row['target_first_layer_gain'], row['phase']), []
        ).append(row)
    aggregates = []
    for (mode, gain, phase), group in sorted(grouped.items()):
        aggregate = {
            'mode': mode,
            'target_first_layer_gain': gain,
            'phase': phase,
            'seed_count': len(group),
        }
        for metric in metric_names:
            values = np.asarray([item.get(metric, np.nan) for item in group], dtype=np.float64)
            finite = values[np.isfinite(values)]
            aggregate[f'{metric}_mean'] = float(finite.mean()) if finite.size else float('nan')
            aggregate[f'{metric}_std'] = float(finite.std()) if finite.size else float('nan')
        aggregates.append(aggregate)
    return aggregates


def _plot_results(
        output_dir: Path,
        aggregate_rows: Sequence[Dict],
        grid_snapshots: Dict[Tuple[str, float, int], np.ndarray],
        density_grid_snapshots: Dict[int, np.ndarray],
        positions: np.ndarray,
        velocities: np.ndarray,
        gain: float,
) -> None:
    import matplotlib.pyplot as plt

    modes = ['legacy_layer_norm', 'fixed_bounds']
    labels = {'legacy_layer_norm': 'legacy LayerNorm', 'fixed_bounds': 'fixed bounds'}
    colors = {'legacy_layer_norm': '#c44e52', 'fixed_bounds': '#4c72b0'}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metrics = [
        ('rnd_density_spearman_mean', 'RND vs density Spearman'),
        ('top10_density_overlap_mean', 'Top 10% novelty overlap'),
        ('unseen_dense_error_ratio_mean', 'Unseen / dense RND error'),
    ]
    for axis, (metric, title) in zip(axes, metrics):
        for mode in modes:
            selected = [
                row for row in aggregate_rows
                if row['mode'] == mode and float(row['target_first_layer_gain']) == float(gain)
            ]
            selected.sort(key=lambda row: int(row['phase']))
            axis.plot(
                [row['phase'] for row in selected],
                [row[metric] for row in selected],
                marker='o', label=labels[mode], color=colors[mode],
            )
        axis.set_title(title)
        axis.set_xlabel('collection phase')
        axis.grid(alpha=0.25)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output_dir / 'phase_summary.png', dpi=160)
    plt.close(fig)

    phases = sorted(density_grid_snapshots)
    for phase in phases:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharex=True, sharey=True)
        panels = [
            (grid_snapshots.get(('legacy_layer_norm', gain, phase)), 'Legacy RND error'),
            (grid_snapshots.get(('fixed_bounds', gain, phase)), 'Fixed-bound RND error'),
            (density_grid_snapshots[phase], 'Explicit density reward'),
        ]
        for axis, (values, title) in zip(axes, panels):
            if values is None:
                axis.set_visible(False)
                continue
            image = axis.imshow(
                values.reshape(len(velocities), len(positions)),
                origin='lower',
                aspect='auto',
                extent=[positions[0], positions[-1], velocities[0], velocities[-1]],
                cmap='viridis',
            )
            axis.set_title(title)
            axis.set_xlabel('position')
            fig.colorbar(image, ax=axis, shrink=0.8)
        axes[0].set_ylabel('velocity')
        fig.suptitle(f'Before RND update on collection phase {phase}')
        fig.tight_layout()
        fig.savefig(output_dir / f'heatmap_phase_{phase:02d}.png', dpi=160)
        plt.close(fig)


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def run_experiment(args: argparse.Namespace) -> Path:
    episodes = load_episodes(args.input, args.max_phase)
    phase_data = build_phase_data(episodes, args)
    seeds = parse_number_list(args.seeds, int)
    gains = parse_number_list(args.target_first_layer_gains, float)
    starts = parse_number_list(args.trajectory_start_positions, float)
    modes = ['legacy_layer_norm', 'fixed_bounds'] if args.compare_legacy else ['fixed_bounds']
    device = torch.device(args.device)
    output_dir = args.output_dir or (args.input / 'rnd_signal_diagnostic')
    output_dir.mkdir(parents=True, exist_ok=True)

    grid_observations, positions, velocities = make_grid(args.grid_position_points, args.grid_velocity_points)
    phase_rows: List[Dict] = []
    episode_rows: List[Dict] = []
    bucket_rows: List[Dict] = []
    all_trajectory_rows: List[Dict] = []
    smoothness_rows: List[Dict] = []
    grid_snapshots: Dict[Tuple[str, float, int], np.ndarray] = {}
    density_grid_snapshots: Dict[int, np.ndarray] = {}

    for phase in phase_data:
        density_grid_snapshots[phase.phase] = density_rewards_for_snapshot(
            grid_observations, phase.density_snapshot_counts, args
        )

    for gain in gains:
        for seed in seeds:
            for mode in modes:
                model = RNDSignalModel(
                    mode=mode,
                    seed=seed,
                    hidden_dim=args.hidden_dim,
                    feature_dim=args.feature_dim,
                    learning_rate=args.learning_rate,
                    target_first_layer_gain=gain,
                    device=device,
                )
                smoothness_rows.append(
                    target_smoothness_row(model, gain, grid_observations, args.target_pair_count)
                )
                for phase in phase_data:
                    pre_errors = model.errors(phase.observations, args.inference_batch_size)
                    phase_row, current_episode_rows, current_bucket_rows = summarize_phase(
                        model, phase, pre_errors, gain
                    )
                    all_trajectory_rows.extend(trajectory_rows(model, phase, gain, starts, args))
                    if seed == seeds[0] and float(gain) == float(gains[0]):
                        grid_snapshots[(mode, gain, phase.phase)] = model.errors(
                            grid_observations, args.inference_batch_size
                        )

                    train_stats = model.train_fresh(
                        phase.observations,
                        update_proportion=args.update_proportion,
                        candidate_batch_size=args.candidate_batch_size,
                        train_passes=args.train_passes,
                        random_seed=seed * 100003 + phase.phase * 997 + int(round(gain * 100)),
                    )
                    post_errors = model.errors(phase.observations, args.inference_batch_size)
                    phase_row.update(train_stats)
                    phase_row['post_train_rnd_error_mean'] = float(post_errors.mean())
                    phase_row['post_pre_error_ratio'] = safe_ratio(
                        float(post_errors.mean()), float(pre_errors.mean())
                    )
                    phase_rows.append(phase_row)
                    episode_rows.extend(current_episode_rows)
                    bucket_rows.extend(current_bucket_rows)

    aggregate_rows = aggregate_phase_metrics(phase_rows)
    write_csv(output_dir / 'phase_metrics.csv', phase_rows)
    write_csv(output_dir / 'aggregate_phase_metrics.csv', aggregate_rows)
    write_csv(output_dir / 'episode_metrics.csv', episode_rows)
    write_csv(output_dir / 'density_bucket_metrics.csv', bucket_rows)
    write_csv(output_dir / 'trajectory_metrics.csv', all_trajectory_rows)
    write_csv(output_dir / 'target_smoothness.csv', smoothness_rows)

    summary = {
        'input': str(args.input),
        'episode_count': len(episodes),
        'transition_count': int(sum(len(episode.next_observations) for episode in episodes)),
        'phases': [phase.phase for phase in phase_data],
        'seeds': seeds,
        'modes': modes,
        'target_first_layer_gains': gains,
        'learning_rate': args.learning_rate,
        'update_proportion': args.update_proportion,
        'candidate_batch_size': args.candidate_batch_size,
        'train_passes': args.train_passes,
        'aggregate_phase_metrics': aggregate_rows,
        'target_smoothness': smoothness_rows,
    }
    with (output_dir / 'summary.json').open('w') as file:
        json.dump(_json_safe(summary), file, indent=2)

    if not args.no_plots:
        _plot_results(
            output_dir,
            aggregate_rows,
            grid_snapshots,
            density_grid_snapshots,
            positions,
            velocities,
            gains[0],
        )

    print(f"Loaded {len(episodes)} episodes and {summary['transition_count']} transitions.")
    print(f"Results written to: {output_dir}")
    print("Aggregate phase metrics:")
    for row in aggregate_rows:
        if float(row['target_first_layer_gain']) != float(gains[0]):
            continue
        print(
            f"  {row['mode']:<18} phase {int(row['phase']):>2}: "
            f"density rho={row['rnd_density_spearman_mean']:+.3f}, "
            f"nearest rho={row['rnd_nearest_distance_spearman_mean']:+.3f}, "
            f"top10={row['top10_density_overlap_mean']:.3f}, "
            f"unseen/dense={row['unseen_dense_error_ratio_mean']:.3f}, "
            f"post/pre={row['post_pre_error_ratio_mean']:.3f}"
        )
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay saved MountainCar observations through standalone RND and compare with explicit density."
    )
    parser.add_argument('input', type=Path, help="Run directory or intrinsic_episodes directory.")
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--max-phase', type=int, default=None)
    parser.add_argument('--seeds', type=str, default='0,1,2,3,4')
    parser.add_argument('--target-first-layer-gains', type=str, default='1.0')
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--feature-dim', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--update-proportion', type=float, default=0.25)
    parser.add_argument('--candidate-batch-size', type=int, default=256)
    parser.add_argument('--train-passes', type=int, default=1)
    parser.add_argument('--inference-batch-size', type=int, default=4096)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--compare-legacy', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--position-bin-width', type=float, default=0.02)
    parser.add_argument('--velocity-bin-width', type=float, default=0.002)
    parser.add_argument('--neighbor-radius', type=int, default=1)
    parser.add_argument('--trajectory-start-positions', type=str, default='-0.55,-0.45')
    parser.add_argument('--grid-position-points', type=int, default=91)
    parser.add_argument('--grid-velocity-points', type=int, default=71)
    parser.add_argument('--target-pair-count', type=int, default=20000)
    parser.add_argument('--no-plots', action='store_true')
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 <= args.update_proportion <= 1.0:
        raise ValueError('--update-proportion must be in [0, 1].')
    if args.candidate_batch_size <= 0 or args.train_passes <= 0:
        raise ValueError('Batch size and train passes must be positive.')
    run_experiment(args)


if __name__ == '__main__':
    main()
