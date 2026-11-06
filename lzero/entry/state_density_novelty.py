from typing import Any, Dict, List, Tuple

import numpy as np


class StateDensityNoveltyTracker:
    """Phase-frozen visitation density over MountainCar's real physical state."""

    def __init__(self, cfg) -> None:
        self.position_min = float(cfg.get('state_density_position_min', -1.2))
        self.position_max = float(cfg.get('state_density_position_max', 0.6))
        self.velocity_min = float(cfg.get('state_density_velocity_min', -0.07))
        self.velocity_max = float(cfg.get('state_density_velocity_max', 0.07))
        self.position_bin_width = float(cfg.get('state_density_position_bin_width', 0.02))
        self.velocity_bin_width = float(cfg.get('state_density_velocity_bin_width', 0.002))
        self.neighbor_radius = max(0, int(cfg.get('state_density_neighbor_radius', 1)))
        self.snapshot_update_interval = max(1, int(cfg.get('state_density_snapshot_update_interval', 1)))
        self.reward_scale = float(cfg.get('state_density_reward_scale', 1.0))
        reward_clip = cfg.get('state_density_reward_clip', 1.0)
        self.reward_clip = None if reward_clip is None else abs(float(reward_clip))

        if self.position_max <= self.position_min or self.velocity_max <= self.velocity_min:
            raise ValueError('State-density bounds must have max > min.')
        if self.position_bin_width <= 0.0 or self.velocity_bin_width <= 0.0:
            raise ValueError('State-density bin widths must be positive.')

        self.position_bin_count = int(
            np.ceil((self.position_max - self.position_min) / self.position_bin_width)
        )
        self.velocity_bin_count = int(
            np.ceil((self.velocity_max - self.velocity_min) / self.velocity_bin_width)
        )
        shape = (self.position_bin_count, self.velocity_bin_count)
        self.live_counts = np.zeros(shape, dtype=np.int64)
        self.snapshot_counts = np.zeros(shape, dtype=np.int64)
        self.snapshot_phase = -1

    def begin_phase(self, collection_phase: int) -> bool:
        collection_phase = int(collection_phase)
        should_update = (
            self.snapshot_phase < 0
            or collection_phase - self.snapshot_phase >= self.snapshot_update_interval
        )
        if should_update:
            self.snapshot_counts = self.live_counts.copy()
            self.snapshot_phase = collection_phase
        return should_update

    def _bin_indices(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        observations = np.asarray(observations, dtype=np.float32)
        if observations.ndim == 1:
            observations = observations.reshape(1, -1)
        if observations.ndim != 2 or observations.shape[1] < 2:
            raise ValueError(
                'State-density novelty requires observations with position and velocity; '
                f'got shape {observations.shape}.'
            )
        position_indices = np.floor(
            (observations[:, 0] - self.position_min) / self.position_bin_width
        ).astype(np.int64)
        velocity_indices = np.floor(
            (observations[:, 1] - self.velocity_min) / self.velocity_bin_width
        ).astype(np.int64)
        position_indices = np.clip(position_indices, 0, self.position_bin_count - 1)
        velocity_indices = np.clip(velocity_indices, 0, self.velocity_bin_count - 1)
        return position_indices, velocity_indices

    def effective_counts(self, observations: np.ndarray) -> np.ndarray:
        position_indices, velocity_indices = self._bin_indices(observations)
        counts = np.zeros(position_indices.shape[0], dtype=np.float64)
        for position_offset in range(-self.neighbor_radius, self.neighbor_radius + 1):
            neighbor_position = position_indices + position_offset
            valid_position = (
                (neighbor_position >= 0) & (neighbor_position < self.position_bin_count)
            )
            for velocity_offset in range(-self.neighbor_radius, self.neighbor_radius + 1):
                neighbor_velocity = velocity_indices + velocity_offset
                valid = valid_position & (
                    (neighbor_velocity >= 0) & (neighbor_velocity < self.velocity_bin_count)
                )
                if np.any(valid):
                    counts[valid] += self.snapshot_counts[
                        neighbor_position[valid], neighbor_velocity[valid]
                    ]
        return counts.astype(np.float32)

    def rewards(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        counts = self.effective_counts(observations)
        rewards = self.reward_scale / np.sqrt(counts + 1.0)
        if self.reward_clip is not None and self.reward_clip > 0.0:
            rewards = np.clip(rewards, 0.0, self.reward_clip)
        return rewards.astype(np.float32), counts

    def update_live_counts(self, observations: np.ndarray) -> None:
        position_indices, velocity_indices = self._bin_indices(observations)
        np.add.at(self.live_counts, (position_indices, velocity_indices), 1)

    @staticmethod
    def _segment_next_observations(game_segment: Any) -> np.ndarray:
        frame_stack_num = max(1, int(getattr(game_segment, 'frame_stack_num', 1)))
        rewards = np.asarray(getattr(game_segment, 'reward_segment', []))
        observations = np.asarray(getattr(game_segment, 'obs_segment', []), dtype=np.float32)
        reward_count = min(len(rewards), max(0, len(observations) - frame_stack_num))
        return observations[frame_stack_num:frame_stack_num + reward_count]

    def annotate_collection(
            self,
            new_data: List[Any],
            collection_phase: int,
    ) -> Dict[str, float]:
        snapshot_updated = self.begin_phase(collection_phase)
        segments, metadata = new_data
        all_rewards = []
        all_counts = []
        real_next_observations = []
        phase_bins = set()

        for game_segment, meta in zip(segments, metadata):
            next_observations = self._segment_next_observations(game_segment)
            rewards, counts = self.rewards(next_observations)
            game_segment.intrinsic_reward_segment = rewards.copy()
            game_segment.intrinsic_reward_raw_segment = counts.copy()

            valid_len = min(
                int(meta.get('segment_length', len(game_segment))),
                int(next_observations.shape[0]),
            )
            valid_observations = next_observations[:valid_len]
            valid_rewards = rewards[:valid_len]
            valid_counts = counts[:valid_len]
            if valid_len > 0:
                real_next_observations.append(valid_observations)
                all_rewards.append(valid_rewards)
                all_counts.append(valid_counts)
                position_indices, velocity_indices = self._bin_indices(valid_observations)
                phase_bins.update(zip(position_indices.tolist(), velocity_indices.tolist()))
                meta['state_density_reward_sum'] = float(valid_rewards.sum())
                meta['state_density_reward_max'] = float(valid_rewards.max())
                meta['state_density_effective_count_mean'] = float(valid_counts.mean())
                meta['state_density_effective_count_min'] = float(valid_counts.min())

        # The active snapshot is not changed by any transition from this phase.
        if real_next_observations:
            self.update_live_counts(np.concatenate(real_next_observations, axis=0))

        rewards_array = np.concatenate(all_rewards) if all_rewards else np.zeros(0, dtype=np.float32)
        counts_array = np.concatenate(all_counts) if all_counts else np.zeros(0, dtype=np.float32)
        occupied_live = int(np.count_nonzero(self.live_counts))
        occupied_snapshot = int(np.count_nonzero(self.snapshot_counts))
        return {
            'collection_phase': float(collection_phase),
            'snapshot_phase': float(self.snapshot_phase),
            'snapshot_updated': float(snapshot_updated),
            'snapshot_transition_count': float(self.snapshot_counts.sum()),
            'live_transition_count': float(self.live_counts.sum()),
            'snapshot_occupied_bin_count': float(occupied_snapshot),
            'live_occupied_bin_count': float(occupied_live),
            'phase_unique_bin_count': float(len(phase_bins)),
            'reward_mean': float(rewards_array.mean()) if rewards_array.size else 0.0,
            'reward_std': float(rewards_array.std()) if rewards_array.size else 0.0,
            'reward_min': float(rewards_array.min()) if rewards_array.size else 0.0,
            'reward_max': float(rewards_array.max()) if rewards_array.size else 0.0,
            'effective_count_mean': float(counts_array.mean()) if counts_array.size else 0.0,
            'effective_count_p50': float(np.percentile(counts_array, 50)) if counts_array.size else 0.0,
            'effective_count_p90': float(np.percentile(counts_array, 90)) if counts_array.size else 0.0,
            'position_bin_count': float(self.position_bin_count),
            'velocity_bin_count': float(self.velocity_bin_count),
            'neighbor_radius': float(self.neighbor_radius),
        }
