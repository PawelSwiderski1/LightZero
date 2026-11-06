import csv
import json
import os
import time
from collections import Counter, defaultdict
from functools import partial
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import wandb
from ding.config import compile_config
from ding.envs import create_env_manager, get_vec_env_setting
from ding.policy import create_policy
from ding.rl_utils import get_epsilon_greedy_fn
from ding.utils import get_rank, set_pkg_seed
from ding.worker import BaseLearner
from ditk import logging
from lzero.mcts.utils import prepare_observation
from lzero.entry.utils import log_buffer_memory_usage, log_buffer_run_time
from lzero.policy import visit_count_temperature
from lzero.policy.utils import prepare_obs
from lzero.policy.random_policy import LightZeroRandomPolicy
from lzero.worker import MuZeroCollector as Collector
from lzero.worker import MuZeroEvaluator as Evaluator
from tensorboardX import SummaryWriter

from .utils import calculate_update_per_collect, random_collect
from .state_density_novelty import StateDensityNoveltyTracker


def _sync_if_cuda(device: str) -> None:
    if str(device).startswith('cuda') and torch.cuda.is_available():
        torch.cuda.synchronize()


def _timer_start(device: str) -> float:
    _sync_if_cuda(device)
    return time.perf_counter()


def _timer_elapsed(start: float, device: str) -> float:
    _sync_if_cuda(device)
    return time.perf_counter() - start


def _maybe_update_ensemble_value_calibration(
        cfg,
        policy,
        replay_buffer,
        tb_logger,
        completed_collection_phase: int,
) -> Optional[Dict[str, Any]]:
    if not bool(cfg.policy.get('ensemble_value_calibration', False)):
        return None

    effective_collection_phase = int(completed_collection_phase) + 1
    start_phase = max(0, int(cfg.policy.get('ensemble_value_calibration_start_phase', 10)))
    if effective_collection_phase < start_phase:
        return None

    sample_size = max(1, int(cfg.policy.get('ensemble_value_calibration_sample_size', 4096)))
    observations, replay_indices = replay_buffer.sample_observations_uniform(sample_size)
    if observations.shape[0] == 0:
        logging.warning(
            '[VALUE_CALIBRATION] phase %s skipped: replay contains no observations',
            effective_collection_phase,
        )
        return None

    stats = policy.update_ensemble_value_calibration(
        observations,
        collection_phase=effective_collection_phase,
    )
    stats['completed_collection_phase'] = int(completed_collection_phase)
    stats['effective_collection_phase'] = effective_collection_phase
    stats['sampled_replay_index_min'] = int(np.min(replay_indices))
    stats['sampled_replay_index_max'] = int(np.max(replay_indices))

    logging.info(
        '[VALUE_CALIBRATION] effective phase %s from %s uniform replay states | '
        'member means %s, centered offsets %s | state std raw %.4f -> calibrated %.4f',
        effective_collection_phase,
        int(stats['sample_count']),
        np.round(np.asarray(stats['member_value_means']), 4).tolist(),
        np.round(np.asarray(stats['offsets']), 4).tolist(),
        float(stats['raw_state_value_std_mean']),
        float(stats['calibrated_state_value_std_mean']),
    )
    tb_logger.add_scalar(
        'ensemble_value_calibration/raw_state_value_std_mean',
        float(stats['raw_state_value_std_mean']),
        effective_collection_phase,
    )
    tb_logger.add_scalar(
        'ensemble_value_calibration/calibrated_state_value_std_mean',
        float(stats['calibrated_state_value_std_mean']),
        effective_collection_phase,
    )
    for member_id, offset in enumerate(stats['offsets']):
        tb_logger.add_scalar(
            f'ensemble_value_calibration/member_{member_id}_offset',
            float(offset),
            effective_collection_phase,
        )
    return stats


def _maybe_refresh_rnd_landscape_query_target_scale(
        cfg,
        policy,
        replay_buffer,
        tb_logger,
        completed_collection_phase: int,
) -> Optional[Dict[str, float]]:
    if (
            not bool(cfg.policy.get('rnd_landscape_query', False))
            or str(cfg.policy.get('rnd_landscape_query_target_transform', 'log')).lower()
            != 'normalized_raw'
    ):
        return None
    refresh_fn = getattr(policy, 'refresh_rnd_landscape_query_target_scale', None)
    if not callable(refresh_fn):
        raise AttributeError(
            "normalized_raw RND landscape targets require "
            "policy.refresh_rnd_landscape_query_target_scale()."
        )

    sample_size = max(
        1,
        int(cfg.policy.get('rnd_landscape_query_normalized_raw_reference_size', 8192)),
    )
    numpy_rng_state = np.random.get_state()
    try:
        observations, replay_indices = replay_buffer.sample_observations_uniform(sample_size)
    finally:
        np.random.set_state(numpy_rng_state)
    if observations.shape[0] == 0:
        logging.warning(
            '[RND_LANDSCAPE_SCALE] phase %s skipped: replay contains no observations',
            completed_collection_phase,
        )
        return None

    stats = refresh_fn(
        observations,
        collection_phase=completed_collection_phase,
    )
    if not stats:
        return None
    stats = dict(stats)
    stats['rnd_landscape_query/reference_replay_index_min'] = float(
        np.min(replay_indices)
    )
    stats['rnd_landscape_query/reference_replay_index_max'] = float(
        np.max(replay_indices)
    )
    logging.info(
        '[RND_LANDSCAPE_SCALE] phase %s replay_samples=%d scale=%.6g '
        'raw_std=%.6g raw_p90=%.6g clipped=%.4f',
        completed_collection_phase,
        int(stats['rnd_landscape_query/target_scale_sample_count']),
        float(stats['rnd_landscape_query/target_scale']),
        float(stats['rnd_landscape_query/target_scale_raw_std']),
        float(stats['rnd_landscape_query/target_scale_raw_p90']),
        float(stats['rnd_landscape_query/target_scale_clip_fraction']),
    )
    if tb_logger is not None:
        for key, value in stats.items():
            tb_logger.add_scalar(
                f'collection_phase/{key}',
                float(value),
                int(completed_collection_phase),
            )
    return stats


def _safe_video_dir_name(exp_name: str) -> str:
    return exp_name.strip('./').replace(os.sep, '_').replace('/', '_')


def _maybe_enable_collect_video(cfg, collector_env) -> None:
    if not cfg.policy.get('record_collect_video', False):
        return
    video_dir = cfg.policy.get('collect_video_dir', None)
    if video_dir is None:
        video_dir = os.path.join('./video', f'{_safe_video_dir_name(cfg.exp_name)}_collect')
    os.makedirs(video_dir, exist_ok=True)
    collector_env.enable_save_replay(video_dir)
    logging.info(f'[COLLECT_VIDEO] Recording collector episodes to {video_dir}')


def _set_collect_video_phase(cfg, collector_env, collection_phase: int) -> None:
    if not cfg.policy.get('record_collect_video', False):
        return
    for env in getattr(collector_env, '_envs', []):
        set_phase = getattr(env, 'set_collect_video_phase', None)
        if callable(set_phase):
            set_phase(collection_phase)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _scalar_reward(reward: Any) -> float:
    if isinstance(reward, (list, tuple, np.ndarray)):
        reward = np.asarray(reward).reshape(-1)[0]
    return float(reward)


def _segment_return(game_segment: Any, game_segment_length: int) -> float:
    valid_len = min(len(game_segment), int(game_segment_length))
    rewards = game_segment.reward_segment[:valid_len]
    return float(sum(_scalar_reward(reward) for reward in rewards))


def _diagnostics_enabled(cfg) -> bool:
    return bool(cfg.policy.get('collection_diagnostics', True))


def _diagnostics_dir(cfg) -> str:
    diagnostics_dir = cfg.policy.get('diagnostics_dir', None)
    if diagnostics_dir is None:
        diagnostics_dir = os.path.join('./', cfg.exp_name, 'diagnostics')
    os.makedirs(diagnostics_dir, exist_ok=True)
    return diagnostics_dir


def _create_spatial_exploration_shadow(policy_config):
    if not bool(policy_config.get('spatial_exploration_shadow', False)):
        return None
    from .spatial_exploration_shadow import SpatialExplorationShadowEvaluator

    evaluator = SpatialExplorationShadowEvaluator(
        checkpoint_path=str(
            policy_config.get('spatial_exploration_shadow_checkpoint', '')
        ),
        device=str(policy_config.device),
        horizon=int(policy_config.get('spatial_exploration_shadow_horizon', 5)),
        roots_per_episode=int(
            policy_config.get('spatial_exploration_shadow_roots_per_episode', 128)
        ),
        root_batch_size=int(
            policy_config.get('spatial_exploration_shadow_root_batch_size', 128)
        ),
        dynamics_history_window=int(
            policy_config.get('spatial_exploration_shadow_history_window', 128)
        ),
        export_dataset_path=policy_config.get(
            'spatial_exploration_shadow_export_dataset', None
        ),
        export_split_counts=policy_config.get(
            'spatial_exploration_shadow_export_split_counts', None
        ),
    )
    logging.info(
        '[SPATIAL_EXPLORATION_SHADOW] loaded frozen checkpoint %s at epoch %s; '
        'dynamics history window=%s; '
        'diagnostic outputs do not affect MCTS or learning',
        evaluator.checkpoint_path,
        evaluator.checkpoint_epoch,
        evaluator.dynamics_history_window,
    )
    return evaluator


def _write_spatial_exploration_shadow_diagnostics(
        cfg,
        evaluator,
        new_data: List[Any],
        tb_logger,
        collection_phase: int,
) -> None:
    if evaluator is None:
        return
    result = evaluator.evaluate_collection(
        new_data,
        frame_stack_num=int(cfg.policy.model.get('frame_stack_num', 1)),
        collection_phase=int(collection_phase),
        diagnostics_dir=_diagnostics_dir(cfg),
    )
    summaries = result.get('summaries', [])
    if result.get('exported_dataset'):
        logging.info(
            '[SPATIAL_EXPLORATION_SHADOW] exported complete trajectory dataset to %s',
            result['exported_dataset'],
        )
    if not summaries:
        logging.warning(
            '[SPATIAL_EXPLORATION_SHADOW] phase %s produced no complete episodes',
            collection_phase,
        )
        return
    overall = next(row for row in summaries if int(row['depth']) == 0)
    final = max(summaries, key=lambda row: int(row['depth']))
    logging.info(
        '[SPATIAL_EXPLORATION_SHADOW] phase %s episodes=%s roots=%s predictions=%s | '
        'all-depth rgb_auc=%.3f position_auc=%.3f cosine=%.3f obskey=%.3f retrieval=%.3f mae=%.4f | '
        'root rgb_count_corr=%.3f position_count_corr=%.3f | '
        'depth%s rgb_auc=%.3f position_auc=%.3f cosine=%.3f obskey=%.3f retrieval=%.3f mae=%.4f',
        collection_phase,
        int(result['episode_count']),
        int(result['root_count']),
        int(result['step_count']),
        float(overall['predicted_rgb_novelty_auc']),
        float(overall['predicted_position_novelty_auc']),
        float(overall['predicted_target_cosine_mean']),
        float(overall['predicted_observation_key_cosine_mean']),
        float(overall['rgb_retrieval_accuracy']),
        float(overall['rollout_rgb_mae']),
        float(overall['root_predicted_rgb_count_correlation']),
        float(overall['root_predicted_position_count_correlation']),
        int(final['depth']),
        float(final['predicted_rgb_novelty_auc']),
        float(final['predicted_position_novelty_auc']),
        float(final['predicted_target_cosine_mean']),
        float(final['predicted_observation_key_cosine_mean']),
        float(final['rgb_retrieval_accuracy']),
        float(final['rollout_rgb_mae']),
    )
    if tb_logger is not None:
        for row in summaries:
            depth_name = 'all' if int(row['depth']) == 0 else f"depth_{int(row['depth'])}"
            for key, value in row.items():
                if key in {
                    'collection_phase', 'depth', 'sample_count',
                    'dynamics_history_window'
                }:
                    continue
                if np.isfinite(float(value)):
                    tb_logger.add_scalar(
                        f'spatial_exploration_shadow/{depth_name}/{key}',
                        float(value),
                        collection_phase,
                    )


def _write_spatial_exploration_online_preupdate_diagnostics(
        cfg,
        policy,
        new_data: List[Any],
        tb_logger,
        collection_phase: int,
) -> None:
    if not bool(
            cfg.policy.get('spatial_exploration_tree_online_training', False)
    ):
        return
    model_fn = getattr(policy, 'get_spatial_exploration_tree_model', None)
    if not callable(model_fn):
        return
    model = model_fn()
    if model is None:
        return
    from .spatial_exploration_shadow import SpatialExplorationShadowEvaluator

    evaluator = SpatialExplorationShadowEvaluator.from_live_model(
        model=model,
        device=str(cfg.policy.device),
        horizon=int(cfg.policy.get('spatial_exploration_tree_horizon', 5)),
        roots_per_episode=int(
            cfg.policy.get(
                'spatial_exploration_tree_online_evaluation_roots_per_episode',
                64,
            )
        ),
        root_batch_size=int(
            cfg.policy.get('spatial_exploration_shadow_root_batch_size', 128)
        ),
        dynamics_history_window=int(
            cfg.policy.get(
                'spatial_exploration_tree_online_history_window', 128
            )
        ),
        revisit_scoring_mode=str(
            cfg.policy.get(
                'spatial_exploration_tree_revisit_scoring_mode', 'model'
            )
        ),
    )
    result = evaluator.evaluate_collection(
        new_data,
        frame_stack_num=int(cfg.policy.model.get('frame_stack_num', 1)),
        collection_phase=int(collection_phase),
        diagnostics_dir=_diagnostics_dir(cfg),
        file_prefix='spatial_exploration_online_preupdate',
    )
    summaries = result.get('summaries', [])
    if not summaries:
        logging.warning(
            '[SPATIAL_EXPLORATION_ONLINE_EVAL] phase %s produced no complete episodes',
            collection_phase,
        )
        return
    overall = next(row for row in summaries if int(row['depth']) == 0)
    final = max(summaries, key=lambda row: int(row['depth']))
    logging.info(
        '[SPATIAL_EXPLORATION_ONLINE_EVAL] phase %s pre_update=1 episodes=%s '
        'roots=%s predictions=%s | all-depth rgb_auc=%.3f position_auc=%.3f '
        'teacher_corr=%.3f cosine=%.3f obskey=%.3f retrieval=%.3f mae=%.4f | '
        'depth%s rgb_auc=%.3f position_auc=%.3f teacher_corr=%.3f '
        'cosine=%.3f obskey=%.3f retrieval=%.3f mae=%.4f',
        collection_phase,
        int(result['episode_count']),
        int(result['root_count']),
        int(result['step_count']),
        float(overall['predicted_rgb_novelty_auc']),
        float(overall['predicted_position_novelty_auc']),
        float(overall['predicted_teacher_correlation']),
        float(overall['predicted_target_cosine_mean']),
        float(overall['predicted_observation_key_cosine_mean']),
        float(overall['rgb_retrieval_accuracy']),
        float(overall['rollout_rgb_mae']),
        int(final['depth']),
        float(final['predicted_rgb_novelty_auc']),
        float(final['predicted_position_novelty_auc']),
        float(final['predicted_teacher_correlation']),
        float(final['predicted_target_cosine_mean']),
        float(final['predicted_observation_key_cosine_mean']),
        float(final['rgb_retrieval_accuracy']),
        float(final['rollout_rgb_mae']),
    )
    if tb_logger is not None:
        for row in summaries:
            depth_name = (
                'all' if int(row['depth']) == 0
                else f"depth_{int(row['depth'])}"
            )
            for key, value in row.items():
                if key in {
                    'collection_phase', 'depth', 'sample_count',
                    'dynamics_history_window'
                }:
                    continue
                if np.isfinite(float(value)):
                    tb_logger.add_scalar(
                        f'spatial_exploration_online_preupdate/{depth_name}/{key}',
                        float(value),
                        collection_phase,
                    )


class EnvironmentCoverageTracker:
    """Accumulate diagnostic-only physical coverage across collection phases."""

    def __init__(self, policy_config) -> None:
        self.enabled = bool(
            policy_config.get('environment_coverage_diagnostics', False)
        )
        self.cell_diagnostics_enabled = self.enabled and bool(
            policy_config.get('environment_coverage_cell_diagnostics', False)
        )
        self.total_positions = max(
            0, int(policy_config.get('environment_reachable_position_count', 0))
        )
        self.total_rooms = max(
            0, int(policy_config.get('environment_room_count', 0))
        )
        configured_dead_ends = policy_config.get(
            'environment_dead_end_room_ids', []
        )
        self.dead_end_rooms = {
            (int(room[0]), int(room[1])) for room in configured_dead_ends
        }
        self.positions = set()
        self.poses = set()
        self.rooms = set()
        self.position_visit_counts = Counter()
        self.pose_visit_counts = Counter()
        self.position_rooms = {}
        self.position_first_visit_phase = {}
        self._latest_cell_rows = []

    @staticmethod
    def _heading_mask(poses) -> int:
        mask = 0
        for _, _, direction in poses:
            if 0 <= int(direction) < 4:
                mask |= 1 << int(direction)
        return mask

    def latest_cell_rows(self) -> List[Dict[str, int]]:
        return list(self._latest_cell_rows)

    def annotate_collection(
            self,
            new_data: List[Any],
            collection_phase: int,
    ) -> Optional[Dict[str, float]]:
        if not self.enabled:
            return None
        phase_positions = set()
        phase_poses = set()
        phase_rooms = set()
        phase_position_visit_counts = Counter()
        phase_pose_visit_counts = Counter()
        phase_position_rooms = {}

        def record_state(state) -> None:
            if state is None or len(state) < 5:
                return
            x, y, direction, room_row, room_col = state[:5]
            position = (int(x), int(y))
            direction = int(direction)
            room = (int(room_row), int(room_col))
            phase_positions.add(position)
            phase_position_visit_counts[position] += 1
            if direction >= 0:
                pose = (*position, direction)
                phase_poses.add(pose)
                phase_pose_visit_counts[pose] += 1
            if room[0] >= 0 and room[1] >= 0:
                phase_rooms.add(room)
                phase_position_rooms[position] = room

        for game_segment, meta in zip(new_data[0], new_data[1]):
            initial_state = getattr(game_segment, 'environment_initial_state', None)
            record_state(initial_state)
            states = np.asarray(
                getattr(game_segment, 'environment_state_segment', [])
            )
            if states.ndim != 2 or states.shape[1] < 5:
                continue
            valid_step_count = min(
                int(meta.get('segment_length', len(game_segment))),
                int(states.shape[0]),
            )
            for state in states[:valid_step_count, :5]:
                record_state(state)

        previous_positions = set(self.positions)
        previous_poses = set(self.poses)
        previous_rooms = set(self.rooms)
        new_positions = phase_positions - previous_positions
        new_poses = phase_poses - previous_poses
        self.positions.update(phase_positions)
        self.poses.update(phase_poses)
        self.rooms.update(phase_rooms)
        self.position_visit_counts.update(phase_position_visit_counts)
        self.pose_visit_counts.update(phase_pose_visit_counts)
        self.position_rooms.update(phase_position_rooms)
        for position in new_positions:
            self.position_first_visit_phase[position] = int(collection_phase)

        if self.cell_diagnostics_enabled:
            phase_poses_by_position = defaultdict(set)
            cumulative_poses_by_position = defaultdict(set)
            new_poses_by_position = defaultdict(set)
            for pose in phase_poses:
                phase_poses_by_position[pose[:2]].add(pose)
            for pose in self.poses:
                cumulative_poses_by_position[pose[:2]].add(pose)
            for pose in new_poses:
                new_poses_by_position[pose[:2]].add(pose)
            rows = []
            for x, y in sorted(phase_positions, key=lambda position: (position[1], position[0])):
                position = (x, y)
                room_row, room_col = self.position_rooms.get(position, (-1, -1))
                phase_position_poses = phase_poses_by_position[position]
                cumulative_position_poses = cumulative_poses_by_position[position]
                new_position_poses = new_poses_by_position[position]
                rows.append({
                    'collection_phase': int(collection_phase),
                    'x': int(x),
                    'y': int(y),
                    'room_row': int(room_row),
                    'room_col': int(room_col),
                    'phase_visit_count': int(phase_position_visit_counts[position]),
                    'cumulative_visit_count': int(self.position_visit_counts[position]),
                    'new_global_position': int(position in new_positions),
                    'first_visit_phase': int(self.position_first_visit_phase[position]),
                    'phase_heading_mask': self._heading_mask(phase_position_poses),
                    'new_global_heading_mask': self._heading_mask(new_position_poses),
                    'cumulative_heading_mask': self._heading_mask(cumulative_position_poses),
                    'phase_pose_count': int(len(phase_position_poses)),
                    'cumulative_pose_count': int(len(cumulative_position_poses)),
                    **{
                        f'phase_heading_{direction}_visit_count': int(
                            phase_pose_visit_counts[(x, y, direction)]
                        )
                        for direction in range(4)
                    },
                    **{
                        f'cumulative_heading_{direction}_visit_count': int(
                            self.pose_visit_counts[(x, y, direction)]
                        )
                        for direction in range(4)
                    },
                })
            self._latest_cell_rows = rows
        else:
            self._latest_cell_rows = []

        visited_dead_ends = self.rooms & self.dead_end_rooms
        return {
            'phase_position_count': float(len(phase_positions)),
            'phase_pose_count': float(len(phase_poses)),
            'phase_room_count': float(len(phase_rooms)),
            'new_global_position_count': float(len(new_positions)),
            'new_global_pose_count': float(len(new_poses)),
            'new_global_room_count': float(len(phase_rooms - previous_rooms)),
            'global_position_count': float(len(self.positions)),
            'global_pose_count': float(len(self.poses)),
            'global_room_count': float(len(self.rooms)),
            'global_dead_end_room_count': float(len(visited_dead_ends)),
            'global_position_coverage': float(
                len(self.positions) / self.total_positions
                if self.total_positions > 0 else 0.0
            ),
            'global_pose_coverage': float(
                len(self.poses) / (4 * self.total_positions)
                if self.total_positions > 0 else 0.0
            ),
            'global_room_coverage': float(
                len(self.rooms) / self.total_rooms
                if self.total_rooms > 0 else 0.0
            ),
            'global_dead_end_room_coverage': float(
                len(visited_dead_ends) / len(self.dead_end_rooms)
                if self.dead_end_rooms else 0.0
            ),
        }


def _write_environment_coverage_cell_diagnostics(
        cfg,
        rows: List[Dict[str, int]],
) -> None:
    if not rows:
        return
    csv_path = os.path.join(
        _diagnostics_dir(cfg), 'environment_coverage_cells.csv'
    )
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _write_compact_query_diagnostics(
        cfg,
        policy,
        collection_phase: int,
) -> None:
    if not bool(cfg.policy.get('rnd_landscape_query', False)):
        return
    if str(cfg.policy.get('rnd_landscape_query_space', 'physical_2d')) != 'compact_image':
        return
    pop_diagnostics = getattr(policy, 'pop_rnd_compact_query_diagnostics', None)
    if not callable(pop_diagnostics):
        logging.warning(
            '[COMPACT_QUERY] policy does not expose compact-query diagnostics'
        )
        return
    payload = pop_diagnostics()
    if not payload:
        return
    output_path = os.path.join(
        _diagnostics_dir(cfg),
        f'compact_query_phase_{int(collection_phase):03d}.npz',
    )
    np.savez_compressed(
        output_path,
        collection_phase=np.asarray([collection_phase], dtype=np.int32),
        **payload,
    )
    logging.info(
        '[COMPACT_QUERY] phase %s wrote %s roots x %s queries to %s',
        collection_phase,
        payload['root_observations'].shape[0],
        payload['query_observations'].shape[1],
        output_path,
    )


def _write_rnd_episodic_memory_diagnostics(
        cfg,
        policy,
        tb_logger,
        collection_phase: int,
) -> None:
    if not bool(cfg.policy.get('rnd_episodic_memory', False)):
        return
    pop_diagnostics = getattr(policy, 'pop_rnd_episodic_memory_diagnostics', None)
    if not callable(pop_diagnostics):
        logging.warning(
            '[RND_EPISODIC_MEMORY] policy does not expose episodic-memory diagnostics'
        )
        return
    rows = pop_diagnostics()
    if not rows:
        return
    output_rows = [
        {'collection_phase': int(collection_phase), **row} for row in rows
    ]
    output_path = os.path.join(
        _diagnostics_dir(cfg), 'rnd_episodic_memory.csv'
    )
    write_header = not os.path.exists(output_path)
    with open(output_path, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(output_rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(output_rows)

    factors = np.asarray(
        [row['one_step_reward_factor'] for row in rows], dtype=np.float32
    )
    selected_rows = [row for row in rows if int(row['selected']) == 1]
    selected_factors = np.asarray(
        [row['one_step_reward_factor'] for row in selected_rows],
        dtype=np.float32,
    )
    suppressed_fraction = float((factors < 1.0 - 1e-6).mean())
    selected_suppressed_fraction = (
        float((selected_factors < 1.0 - 1e-6).mean())
        if selected_factors.size else 0.0
    )
    tree_path_match = np.asarray(
        [row['tree_path_match_fraction'] for row in rows], dtype=np.float32
    )
    pseudo_counts = np.asarray(
        [row.get('one_step_pseudo_count', 0.0) for row in selected_rows],
        dtype=np.float32,
    )
    normalized_knn_distances = np.asarray(
        [
            row.get('one_step_normalized_knn_distance_mean', np.nan)
            for row in selected_rows
        ],
        dtype=np.float32,
    )
    knn_warmup = np.asarray(
        [row.get('one_step_knn_warmup', 0.0) for row in selected_rows],
        dtype=np.float32,
    )
    tb_logger.add_scalar(
        'rnd_episodic_memory/one_step_reward_factor_mean',
        float(factors.mean()),
        collection_phase,
    )
    tb_logger.add_scalar(
        'rnd_episodic_memory/selected_one_step_reward_factor_mean',
        float(selected_factors.mean()) if selected_factors.size else 1.0,
        collection_phase,
    )
    tb_logger.add_scalar(
        'rnd_episodic_memory/one_step_reward_factor_min',
        float(factors.min()),
        collection_phase,
    )
    tb_logger.add_scalar(
        'rnd_episodic_memory/one_step_suppressed_fraction',
        suppressed_fraction,
        collection_phase,
    )
    tb_logger.add_scalar(
        'rnd_episodic_memory/selected_one_step_suppressed_fraction',
        selected_suppressed_fraction,
        collection_phase,
    )
    tb_logger.add_scalar(
        'rnd_episodic_memory/tree_path_match_fraction_mean',
        float(tree_path_match.mean()),
        collection_phase,
    )
    tb_logger.add_scalar(
        'rnd_episodic_memory/one_step_pseudo_count_mean',
        float(pseudo_counts.mean()) if pseudo_counts.size else 0.0,
        collection_phase,
    )
    finite_knn_distances = normalized_knn_distances[
        np.isfinite(normalized_knn_distances)
    ]
    tb_logger.add_scalar(
        'rnd_episodic_memory/one_step_normalized_knn_distance_mean',
        (
            float(finite_knn_distances.mean())
            if finite_knn_distances.size else 0.0
        ),
        collection_phase,
    )
    tb_logger.add_scalar(
        'rnd_episodic_memory/one_step_knn_warmup_fraction',
        float(knn_warmup.mean()) if knn_warmup.size else 0.0,
        collection_phase,
    )
    logging.info(
        '[RND_EPISODIC_MEMORY] phase %s rows %s | one-step factor mean %.4f '
        'min %.4f selected %.4f | suppressed %.3f selected %.3f | '
        'pseudo-count %.4f | normalized kNN distance %.4f | warmup %.3f | '
        'tree path-match fraction %.4f',
        collection_phase,
        len(rows),
        float(factors.mean()),
        float(factors.min()),
        float(selected_factors.mean()) if selected_factors.size else 1.0,
        suppressed_fraction,
        selected_suppressed_fraction,
        float(pseudo_counts.mean()) if pseudo_counts.size else 0.0,
        float(finite_knn_distances.mean()) if finite_knn_distances.size else 0.0,
        float(knn_warmup.mean()) if knn_warmup.size else 0.0,
        float(tree_path_match.mean()),
    )


def _write_environment_coverage_diagnostics(
        cfg,
        stats: Optional[Dict[str, float]],
        tb_logger,
        collection_phase: int,
) -> None:
    if not stats:
        return
    row = {'collection_phase': int(collection_phase), **stats}
    csv_path = os.path.join(
        _diagnostics_dir(cfg), 'environment_coverage.csv'
    )
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    logging.info(
        '[ENV_COVERAGE] phase %s positions %.0f/%.0f (%.3f), rooms %.0f/%.0f '
        '(%.3f), dead_ends %.0f/%.0f (%.3f), new_positions %.0f',
        collection_phase,
        stats['global_position_count'],
        float(cfg.policy.get('environment_reachable_position_count', 0)),
        stats['global_position_coverage'],
        stats['global_room_count'],
        float(cfg.policy.get('environment_room_count', 0)),
        stats['global_room_coverage'],
        stats['global_dead_end_room_count'],
        float(len(cfg.policy.get('environment_dead_end_room_ids', []))),
        stats['global_dead_end_room_coverage'],
        stats['new_global_position_count'],
    )
    if tb_logger is not None:
        for key, value in stats.items():
            tb_logger.add_scalar(
                f'environment_coverage/{key}', value, collection_phase
            )


def _collect_video_path(cfg, meta: dict, collection_phase: int) -> Optional[str]:
    if not cfg.policy.get('record_collect_video', False):
        return None
    if 'env_id' not in meta or 'env_episode_index' not in meta:
        return None
    video_dir = cfg.policy.get('collect_video_dir', None)
    if video_dir is None:
        video_dir = os.path.join('./video', f'{_safe_video_dir_name(cfg.exp_name)}_collect')
    filename = (
        f"env-{int(meta['env_id'])}_collection_phase-{int(collection_phase)}-"
        f"episode-{int(meta['env_episode_index'])}.mp4"
    )
    return os.path.join(video_dir, filename)


def _annotate_collection_metadata(
        cfg,
        new_data: List[Any],
        collection_phase: int,
        train_iter: int,
        envstep_before_collect: int,
        envstep_after_collect: int,
) -> None:
    data, meta_list = new_data
    for segment_index, (game_segment, meta) in enumerate(zip(data, meta_list)):
        meta['collection_phase'] = int(collection_phase)
        meta['train_iter_at_collect'] = int(train_iter)
        meta['envstep_before_collect'] = int(envstep_before_collect)
        meta['envstep_after_collect'] = int(envstep_after_collect)
        meta['segment_index_in_collect'] = int(segment_index)
        meta['segment_length'] = int(min(len(game_segment), int(cfg.policy.game_segment_length)))
        meta['segment_return'] = float(_segment_return(game_segment, cfg.policy.game_segment_length))
        if bool(meta.get('done', False)) and 'episode_return' not in meta:
            meta['episode_return'] = float(meta['segment_return'])
        video_path = _collect_video_path(cfg, meta, collection_phase)
        if video_path is not None:
            meta['video_path'] = video_path


def _write_collection_episode_diagnostics(cfg, new_data: List[Any], tb_logger, collection_phase: int) -> None:
    if not _diagnostics_enabled(cfg):
        return

    episode_rows = []
    for game_segment, meta in zip(new_data[0], new_data[1]):
        if not bool(meta.get('done', False)):
            continue
        valid_step_count = int(meta.get('segment_length', min(len(game_segment), int(cfg.policy.game_segment_length))))
        state_summary = _segment_state_summary(cfg, game_segment, valid_step_count)
        intrinsic_rewards = np.asarray(
            getattr(game_segment, 'intrinsic_reward_segment', []), dtype=np.float32
        ).reshape(-1)[:valid_step_count]
        intrinsic_reward_raws = np.asarray(
            getattr(game_segment, 'intrinsic_reward_raw_segment', []), dtype=np.float32
        ).reshape(-1)[:valid_step_count]
        intrinsic_summary = {
            'intrinsic_reward_sum': float(intrinsic_rewards.sum()) if intrinsic_rewards.size > 0 else 0.0,
            'intrinsic_reward_max': float(intrinsic_rewards.max()) if intrinsic_rewards.size > 0 else 0.0,
            'intrinsic_reward_nonzero_fraction': (
                float((intrinsic_rewards > 0.0).mean()) if intrinsic_rewards.size > 0 else 0.0
            ),
            'intrinsic_reward_raw_sum': float(intrinsic_reward_raws.sum()) if intrinsic_reward_raws.size > 0 else 0.0,
            'intrinsic_reward_raw_max': float(intrinsic_reward_raws.max()) if intrinsic_reward_raws.size > 0 else 0.0,
        }
        policy_mode_summary = _segment_policy_mode_summary(game_segment, valid_step_count)
        episode_rows.append(
            {
                'collection_phase': meta.get('collection_phase'),
                'train_iter_at_collect': meta.get('train_iter_at_collect'),
                'envstep_before_collect': meta.get('envstep_before_collect'),
                'envstep_after_collect': meta.get('envstep_after_collect'),
                'segment_index_in_collect': meta.get('segment_index_in_collect'),
                'env_id': meta.get('env_id'),
                'env_episode_index': meta.get('env_episode_index'),
                'collection_episode_index': meta.get('collection_episode_index'),
                'episode_segment_index': meta.get('episode_segment_index'),
                'episode_return': meta.get('episode_return'),
                'segment_return': meta.get('segment_return'),
                'segment_length': meta.get('segment_length'),
                'episode_visited_position_count': meta.get('visited_position_count'),
                'episode_visited_pose_count': meta.get('visited_pose_count'),
                'episode_visited_room_count': meta.get('visited_room_count'),
                'intrinsic_reward_source': cfg.policy.get('intrinsic_reward_source', 'rnd'),
                **state_summary,
                **intrinsic_summary,
                **policy_mode_summary,
                'video_path': meta.get('video_path'),
            }
        )
    if not episode_rows:
        return

    diagnostics_dir = _diagnostics_dir(cfg)
    csv_path = os.path.join(diagnostics_dir, 'collection_episodes.csv')
    fieldnames = list(episode_rows[0].keys())
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in episode_rows:
            writer.writerow(row)
            logging.info(
                "[COLLECT_EPISODE] phase %s env %s episode %s return %.3f length %s "
                "intrinsic_sum %.3f intrinsic_max %.3f video %s",
                row['collection_phase'],
                row['env_id'],
                row['env_episode_index'],
                float(row['episode_return']),
                row['segment_length'],
                float(row['intrinsic_reward_sum']),
                float(row['intrinsic_reward_max']),
                row['video_path'],
            )

    returns = np.asarray([float(row['episode_return']) for row in episode_rows], dtype=np.float32)
    logging.info(
        "[COLLECT_PHASE_DIAG] phase %s episodes %s reward_mean %.3f reward_max %.3f reward_min %.3f",
        collection_phase,
        len(episode_rows),
        float(returns.mean()),
        float(returns.max()),
        float(returns.min()),
    )
    if cfg.policy.get('intrinsic_reward_source', 'rnd') == 'episodic_novelty':
        intrinsic_sums = np.asarray(
            [float(row['intrinsic_reward_sum']) for row in episode_rows], dtype=np.float32
        )
        max_positions = np.asarray(
            [float(row['segment_max_position']) for row in episode_rows], dtype=np.float32
        )
        max_velocities = np.asarray(
            [float(row['segment_max_abs_velocity']) for row in episode_rows], dtype=np.float32
        )
        position_correlation = (
            float(np.corrcoef(intrinsic_sums, max_positions)[0, 1])
            if intrinsic_sums.size > 1 and intrinsic_sums.std() > 1e-8 and max_positions.std() > 1e-8
            else 0.0
        )
        velocity_correlation = (
            float(np.corrcoef(intrinsic_sums, max_velocities)[0, 1])
            if intrinsic_sums.size > 1 and intrinsic_sums.std() > 1e-8 and max_velocities.std() > 1e-8
            else 0.0
        )
        logging.info(
            "[EPISODIC_NOVELTY_EPISODES] phase %s | sum mean %.4f std %.4f, "
            "corr(max_position) %.4f, corr(max_abs_velocity) %.4f",
            collection_phase,
            float(intrinsic_sums.mean()),
            float(intrinsic_sums.std()),
            position_correlation,
            velocity_correlation,
        )
    if tb_logger is not None:
        tb_logger.add_scalar('collection_phase/episode_count', float(len(episode_rows)), collection_phase)
        tb_logger.add_scalar('collection_phase/reward_mean', float(returns.mean()), collection_phase)
        tb_logger.add_scalar('collection_phase/reward_max', float(returns.max()), collection_phase)
        tb_logger.add_scalar('collection_phase/reward_min', float(returns.min()), collection_phase)
        max_positions = np.asarray([float(row['segment_max_position']) for row in episode_rows], dtype=np.float32)
        max_abs_velocities = np.asarray(
            [float(row['segment_max_abs_velocity']) for row in episode_rows], dtype=np.float32
        )
        tb_logger.add_scalar('collection_phase/segment_max_position_mean', float(max_positions.mean()), collection_phase)
        tb_logger.add_scalar('collection_phase/segment_max_position_max', float(max_positions.max()), collection_phase)
        tb_logger.add_scalar(
            'collection_phase/segment_max_abs_velocity_mean', float(max_abs_velocities.mean()), collection_phase
        )
        intrinsic_sums = np.asarray([float(row['intrinsic_reward_sum']) for row in episode_rows], dtype=np.float32)
        intrinsic_maxes = np.asarray([float(row['intrinsic_reward_max']) for row in episode_rows], dtype=np.float32)
        tb_logger.add_scalar('collection_phase/intrinsic_reward_sum_mean', float(intrinsic_sums.mean()), collection_phase)
        tb_logger.add_scalar('collection_phase/intrinsic_reward_sum_max', float(intrinsic_sums.max()), collection_phase)
        tb_logger.add_scalar('collection_phase/intrinsic_reward_max_mean', float(intrinsic_maxes.mean()), collection_phase)
        tb_logger.add_scalar('collection_phase/intrinsic_reward_max_max', float(intrinsic_maxes.max()), collection_phase)
        pure_fractions = np.asarray(
            [float(row['pure_exploration_step_fraction']) for row in episode_rows], dtype=np.float32
        )
        tb_logger.add_scalar(
            'collection_phase/pure_exploration_episode_fraction',
            float((pure_fractions > 0.5).mean()),
            collection_phase,
        )
        tb_logger.add_scalar(
            'collection_phase/pure_exploration_step_fraction_mean',
            float(pure_fractions.mean()),
            collection_phase,
        )


def _write_state_density_phase_diagnostics(
        cfg,
        stats: Optional[Dict[str, float]],
        tb_logger,
        collection_phase: int,
) -> None:
    if not stats:
        return
    diagnostics_dir = _diagnostics_dir(cfg)
    csv_path = os.path.join(diagnostics_dir, 'state_density_phases.csv')
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(stats.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(stats)
    logging.info(
        '[STATE_DENSITY] phase %s snapshot phase %s (%s transitions, %s bins) | '
        'phase bins %s | reward mean %.4f [%.4f, %.4f] | effective count p50 %.1f p90 %.1f',
        collection_phase,
        int(stats['snapshot_phase']),
        int(stats['snapshot_transition_count']),
        int(stats['snapshot_occupied_bin_count']),
        int(stats['phase_unique_bin_count']),
        float(stats['reward_mean']),
        float(stats['reward_min']),
        float(stats['reward_max']),
        float(stats['effective_count_p50']),
        float(stats['effective_count_p90']),
    )
    if tb_logger is not None:
        for key, value in stats.items():
            if key == 'collection_phase':
                continue
            tb_logger.add_scalar(f'state_density/{key}', float(value), collection_phase)


def _discounted_return_to_go(rewards: np.ndarray, discount_factor: float) -> np.ndarray:
    rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
    returns = np.zeros_like(rewards)
    running_return = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running_return = float(rewards[index]) + float(discount_factor) * running_return
        returns[index] = running_return
    return returns


def _write_state_density_value_propagation_diagnostics(
        cfg,
        policy,
        new_data: List[Any],
        tb_logger,
        collection_phase: int,
) -> None:
    if str(cfg.policy.get('intrinsic_reward_source', 'rnd')) != 'state_density':
        return
    model = getattr(policy, '_learn_model', None)
    if model is None:
        model = getattr(policy, '_collect_model', None)
    if model is None or not hasattr(model, 'intrinsic_value_from_latent'):
        return

    discount_factor = float(cfg.policy.get('rnd_intrinsic_discount_factor', 0.99))
    samples = []
    episode_count = 0
    for segment_index, (game_segment, meta) in enumerate(zip(new_data[0], new_data[1])):
        valid_len = min(
            int(meta.get('segment_length', len(game_segment))),
            len(getattr(game_segment, 'action_segment', [])),
            len(getattr(game_segment, 'intrinsic_reward_segment', [])),
        )
        if valid_len <= 0:
            continue
        rewards = np.asarray(
            game_segment.intrinsic_reward_segment[:valid_len], dtype=np.float32
        )
        returns = _discounted_return_to_go(rewards, discount_factor)
        peak_index = int(np.argmax(rewards))
        episode_count += int(bool(meta.get('done', False)))
        for timestep in range(valid_len):
            try:
                root_observation = game_segment.get_unroll_obs(
                    timestep, num_unroll_steps=0, padding=False
                )
            except Exception:
                continue
            samples.append(
                {
                    'segment_index': segment_index,
                    'timestep': timestep,
                    'root_observation': root_observation,
                    'target': float(returns[timestep]),
                    'reward': float(rewards[timestep]),
                    'peak_index': peak_index,
                    'distance_before_peak': peak_index - timestep,
                }
            )
    if not samples:
        return

    ensemble_size = int(cfg.policy.model.get('ensemble_size', 1))
    inference_batch_size = max(
        1, int(cfg.policy.get('state_density_inference_batch_size', 1024))
    )
    prediction_chunks = []
    member_std_chunks = []
    was_training = model.training
    active_member_state = getattr(model, 'active_member_id', None)
    model.eval()
    try:
        with torch.no_grad():
            for batch_start in range(0, len(samples), inference_batch_size):
                batch = samples[batch_start:batch_start + inference_batch_size]
                prepared = prepare_observation(
                    [sample['root_observation'] for sample in batch],
                    cfg.policy.model.model_type,
                )
                observations = torch.as_tensor(
                    prepared, dtype=torch.float32, device=cfg.policy.device
                )
                batch_size = int(observations.shape[0])
                member_ids = [
                    member_id
                    for member_id in range(ensemble_size)
                    for _ in range(batch_size)
                ]
                repeated_observations = torch.cat(
                    [observations for _ in range(ensemble_size)], dim=0
                )
                model.set_active_members(member_ids)
                latent_state = model.initial_inference(
                    repeated_observations, member_id=member_ids
                ).latent_state
                values = model.intrinsic_value_from_latent(
                    latent_state, member_id=member_ids
                ).reshape(ensemble_size, batch_size)
                prediction_chunks.append(values.mean(dim=0).cpu().numpy())
                member_std_chunks.append(values.std(dim=0, unbiased=False).cpu().numpy())
    finally:
        _restore_model_member_state(model, active_member_state)
        model.train(was_training)

    predictions = np.concatenate(prediction_chunks).astype(np.float32)
    member_stds = np.concatenate(member_std_chunks).astype(np.float32)
    targets = np.asarray([sample['target'] for sample in samples], dtype=np.float32)
    immediate_rewards = np.asarray([sample['reward'] for sample in samples], dtype=np.float32)
    correlation = (
        float(np.corrcoef(predictions, targets)[0, 1])
        if predictions.std() > 1e-8 and targets.std() > 1e-8 else 0.0
    )

    row = {
        'collection_phase': int(collection_phase),
        'sample_count': int(len(samples)),
        'episode_count': int(episode_count),
        'reward_mean': float(immediate_rewards.mean()),
        'reward_max': float(immediate_rewards.max()),
        'mc_target_mean': float(targets.mean()),
        'mc_target_std': float(targets.std()),
        'prediction_mean': float(predictions.mean()),
        'prediction_std': float(predictions.std()),
        'prediction_mae': float(np.abs(predictions - targets).mean()),
        'prediction_correlation': correlation,
        'member_prediction_std_mean': float(member_stds.mean()),
    }

    def add_subset(prefix: str, mask: np.ndarray) -> None:
        if not np.any(mask):
            row[f'{prefix}_count'] = 0
            row[f'{prefix}_target_mean'] = 0.0
            row[f'{prefix}_prediction_mean'] = 0.0
            row[f'{prefix}_mae'] = 0.0
            return
        row[f'{prefix}_count'] = int(mask.sum())
        row[f'{prefix}_target_mean'] = float(targets[mask].mean())
        row[f'{prefix}_prediction_mean'] = float(predictions[mask].mean())
        row[f'{prefix}_mae'] = float(np.abs(predictions[mask] - targets[mask]).mean())

    timesteps = np.asarray([sample['timestep'] for sample in samples], dtype=np.int64)
    distances = np.asarray([sample['distance_before_peak'] for sample in samples], dtype=np.int64)
    add_subset('start', timesteps == 0)
    for offset in [5, 20, 50]:
        add_subset(f'pre_peak_{offset}', distances == offset)
    top_threshold = np.percentile(targets, 90)
    add_subset('top10_target', targets >= top_threshold)

    diagnostics_dir = _diagnostics_dir(cfg)
    csv_path = os.path.join(diagnostics_dir, 'state_density_value_propagation.csv')
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    logging.info(
        '[STATE_DENSITY_VALUE] phase %s post-learn | target %.3f prediction %.3f | '
        'MAE %.3f corr %.3f member_std %.3f | start target/pred %.3f/%.3f | '
        '50-before-peak target/pred %.3f/%.3f',
        collection_phase,
        row['mc_target_mean'],
        row['prediction_mean'],
        row['prediction_mae'],
        row['prediction_correlation'],
        row['member_prediction_std_mean'],
        row['start_target_mean'],
        row['start_prediction_mean'],
        row['pre_peak_50_target_mean'],
        row['pre_peak_50_prediction_mean'],
    )
    if tb_logger is not None:
        for key, value in row.items():
            if key == 'collection_phase':
                continue
            tb_logger.add_scalar(f'state_density_value/{key}', float(value), collection_phase)


def _segment_policy_mode_summary(game_segment, valid_step_count: int) -> Dict[str, Any]:
    diagnostics_segment = getattr(game_segment, 'root_action_diagnostics_segment', [])
    action_segment = getattr(game_segment, 'action_segment', [])
    selected_terms = []
    for timestep in range(min(valid_step_count, len(diagnostics_segment), len(action_segment))):
        selected_action = _scalar_action(action_segment[timestep])
        timestep_terms = diagnostics_segment[timestep] or []
        selected = None
        for action_terms in timestep_terms:
            if int(action_terms.get('action', -1)) == selected_action:
                selected = action_terms
                break
        if selected is not None:
            selected_terms.append(selected)

    if not selected_terms:
        return {
            'collection_policy_mode_majority': 'unknown',
            'pure_exploration_step_fraction': 0.0,
            'pure_exploration_new_sequence_fraction': 0.0,
            'pure_exploration_commit_fraction': 0.0,
            'pure_exploration_selected_sequence_qstd_mean': 0.0,
            'pure_exploration_selected_sequence_qstd_max': 0.0,
            'pure_exploration_selected_sequence_score_mean': 0.0,
            'pure_exploration_selected_sequence_member_centered_qstd_mean': 0.0,
        }

    modes = [str(item.get('collection_policy_mode', 'normal')) for item in selected_terms]
    mode_counts = Counter(modes)
    pure_terms = [
        item for item in selected_terms
        if float(item.get('pure_exploration_episode', 0.0)) > 0.5
        or str(item.get('collection_policy_mode', '')) == 'pure_exploration_episode'
    ]
    pure_count = len(pure_terms)
    new_sequence_count = sum(
        1 for item in pure_terms if str(item.get('pure_exploration_action_source', '')) == 'new_sequence'
    )
    commit_count = sum(
        1 for item in pure_terms if str(item.get('pure_exploration_action_source', '')) == 'commit'
    )
    qstd_values = np.asarray(
        [float(item.get('pure_exploration_selected_sequence_qstd', 0.0)) for item in pure_terms],
        dtype=np.float32,
    )
    score_values = np.asarray(
        [float(item.get('pure_exploration_selected_sequence_score', 0.0)) for item in pure_terms],
        dtype=np.float32,
    )
    centered_qstd_values = np.asarray(
        [
            float(item.get('pure_exploration_selected_sequence_member_centered_qstd', 0.0))
            for item in pure_terms
        ],
        dtype=np.float32,
    )
    return {
        'collection_policy_mode_majority': mode_counts.most_common(1)[0][0],
        'pure_exploration_step_fraction': float(pure_count / max(1, len(selected_terms))),
        'pure_exploration_new_sequence_fraction': float(new_sequence_count / max(1, len(selected_terms))),
        'pure_exploration_commit_fraction': float(commit_count / max(1, len(selected_terms))),
        'pure_exploration_selected_sequence_qstd_mean': (
            float(qstd_values.mean()) if qstd_values.size > 0 else 0.0
        ),
        'pure_exploration_selected_sequence_qstd_max': (
            float(qstd_values.max()) if qstd_values.size > 0 else 0.0
        ),
        'pure_exploration_selected_sequence_score_mean': (
            float(score_values.mean()) if score_values.size > 0 else 0.0
        ),
        'pure_exploration_selected_sequence_member_centered_qstd_mean': (
            float(centered_qstd_values.mean()) if centered_qstd_values.size > 0 else 0.0
        ),
    }


def _scalar_action(action: Any) -> int:
    if isinstance(action, (list, tuple, np.ndarray)):
        action = np.asarray(action).reshape(-1)[0]
    return int(action)


def _state_components(obs: Any) -> Tuple[float, float]:
    obs_arr = np.asarray(obs, dtype=np.float32).reshape(-1)
    if obs_arr.shape[0] < 2:
        return float('nan'), float('nan')
    return float(obs_arr[0]), float(obs_arr[1])


def _action_direction(action: int) -> float:
    if action == 0:
        return -1.0
    if action == 2:
        return 1.0
    if action == 1:
        return 0.0
    return float('nan')


def _action_matches_velocity(action: int, velocity: float) -> float:
    direction = _action_direction(action)
    if np.isnan(direction) or np.isnan(velocity) or direction == 0.0 or abs(velocity) < 1e-8:
        return 0.0
    return float(np.sign(velocity) == direction)


def _finite_float(value: Any, default: float = float('nan')) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _mean_float(rows: List[Dict[str, Any]], key: str, default: float = 0.0) -> float:
    values = [_finite_float(row.get(key, default)) for row in rows]
    values = [value for value in values if np.isfinite(value)]
    if not values:
        return default
    return float(np.mean(values))


def _mountain_car_next_state(position: float, velocity: float, action: int) -> Tuple[float, float]:
    """
    Deterministic Gym MountainCar-v0 one-step transition. Used only for diagnostics.
    """
    if not np.isfinite(position) or not np.isfinite(velocity) or int(action) not in [0, 1, 2]:
        return float('nan'), float('nan')
    next_velocity = float(velocity) + (int(action) - 1) * 0.001 - 0.0025 * np.cos(3.0 * float(position))
    next_velocity = float(np.clip(next_velocity, -0.07, 0.07))
    next_position = float(position) + next_velocity
    next_position = float(np.clip(next_position, -1.2, 0.6))
    if next_position <= -1.2 and next_velocity < 0:
        next_velocity = 0.0
    return next_position, next_velocity


def _simulate_mountain_car_sequence(position: float, velocity: float, actions: List[int]) -> Dict[str, float]:
    if not np.isfinite(position) or not np.isfinite(velocity):
        return {
            'true_sequence_final_position': float('nan'),
            'true_sequence_final_velocity': float('nan'),
            'true_sequence_max_position': float('nan'),
            'true_sequence_min_position': float('nan'),
            'true_sequence_max_abs_velocity': float('nan'),
            'true_sequence_reached_goal': 0.0,
        }
    positions = [float(position)]
    velocities = [float(velocity)]
    current_position = float(position)
    current_velocity = float(velocity)
    for action in actions:
        current_position, current_velocity = _mountain_car_next_state(current_position, current_velocity, int(action))
        positions.append(current_position)
        velocities.append(current_velocity)
    return {
        'true_sequence_final_position': float(current_position),
        'true_sequence_final_velocity': float(current_velocity),
        'true_sequence_max_position': float(max(positions)),
        'true_sequence_min_position': float(min(positions)),
        'true_sequence_max_abs_velocity': float(max(abs(velocity_value) for velocity_value in velocities)),
        'true_sequence_reached_goal': float(max(positions) >= 0.5),
    }


def _mountain_car_sequence_next_observations(
        position: float,
        velocity: float,
        actions: List[int],
) -> np.ndarray:
    """Return the real next observation after every action in a diagnostic sequence."""
    observations = []
    current_position = float(position)
    current_velocity = float(velocity)
    for action in actions:
        current_position, current_velocity = _mountain_car_next_state(
            current_position, current_velocity, int(action)
        )
        observations.append([current_position, current_velocity])
    return np.asarray(observations, dtype=np.float32)


def _observation_bin_key(position: float, velocity: float) -> Optional[Tuple[int, int]]:
    if not np.isfinite(position) or not np.isfinite(velocity):
        return None
    # Matches the diagnostic scale used in our ad-hoc MountainCar analyses.
    return int(round(float(position) / 0.005)), int(round(float(velocity) / 0.001))


def _load_prior_mountain_car_counts(csv_path: str) -> Tuple[Counter, Counter]:
    obs_counts: Counter = Counter()
    obs_action_counts: Counter = Counter()
    if not os.path.exists(csv_path):
        return obs_counts, obs_action_counts
    try:
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if str(row.get('selected', '0')) not in ['1', 'True', 'true']:
                    continue
                obs_key = _observation_bin_key(
                    _finite_float(row.get('obs_position')),
                    _finite_float(row.get('obs_velocity')),
                )
                if obs_key is None:
                    continue
                action = int(_finite_float(row.get('action'), -1))
                obs_counts[obs_key] += 1
                obs_action_counts[(obs_key[0], obs_key[1], action)] += 1
    except (OSError, ValueError, KeyError):
        return Counter(), Counter()
    return obs_counts, obs_action_counts


def _root_row_key(row: Dict[str, Any]) -> Tuple[int, int, int, int, int, int]:
    return (
        int(_finite_float(row.get('collection_episode_index'), -1)),
        int(_finite_float(row.get('segment_timestep'), row.get('timestep', -1))),
        int(_finite_float(row.get('segment_index_in_collect'), -1)),
        int(_finite_float(row.get('env_id'), -1)),
        int(_finite_float(row.get('env_episode_index'), -1)),
        int(_finite_float(row.get('episode_segment_index'), -1)),
    )


def _annotate_mountain_car_novelty_counts(rows: List[Dict[str, Any]], csv_path: str) -> None:
    if not rows:
        return
    obs_counts, obs_action_counts = _load_prior_mountain_car_counts(csv_path)
    root_groups: Dict[Tuple[int, int, int, int, int, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        row.update(
            {
                'obs_bin_prior_count': 0,
                'obs_action_bin_prior_count': 0,
                'true_next_position': float('nan'),
                'true_next_velocity': float('nan'),
                'true_next_obs_bin_prior_count': 0,
                'true_next_obs_bin_unseen': 0.0,
                'true_next_obs_bin_rare2': 0.0,
            }
        )
        root_groups[_root_row_key(row)].append(row)

    for _, root_rows in sorted(root_groups.items(), key=lambda item: item[0]):
        selected_rows = [row for row in root_rows if int(row.get('selected', 0)) == 1]
        reference_row = selected_rows[0] if selected_rows else root_rows[0]
        position = _finite_float(reference_row.get('obs_position'))
        velocity = _finite_float(reference_row.get('obs_velocity'))
        obs_key = _observation_bin_key(position, velocity)
        obs_prior_count = int(obs_counts.get(obs_key, 0)) if obs_key is not None else 0

        for row in root_rows:
            action = int(_finite_float(row.get('action'), -1))
            action_prior_count = (
                int(obs_action_counts.get((obs_key[0], obs_key[1], action), 0)) if obs_key is not None else 0
            )
            next_position, next_velocity = _mountain_car_next_state(position, velocity, action)
            next_key = _observation_bin_key(next_position, next_velocity)
            next_prior_count = int(obs_counts.get(next_key, 0)) if next_key is not None else 0
            row.update(
                {
                    'obs_bin_prior_count': obs_prior_count,
                    'obs_action_bin_prior_count': action_prior_count,
                    'true_next_position': next_position,
                    'true_next_velocity': next_velocity,
                    'true_next_obs_bin_prior_count': next_prior_count,
                    'true_next_obs_bin_unseen': float(next_prior_count == 0),
                    'true_next_obs_bin_rare2': float(next_prior_count <= 2),
                }
            )

        if selected_rows and obs_key is not None:
            selected_action = int(_finite_float(selected_rows[0].get('action'), -1))
            obs_counts[obs_key] += 1
            obs_action_counts[(obs_key[0], obs_key[1], selected_action)] += 1


def _prob_entropy(rows: List[Dict[str, Any]], key: str) -> float:
    probs = np.asarray([_finite_float(row.get(key), 0.0) for row in rows], dtype=np.float64)
    probs = probs[probs > 0.0]
    if probs.size == 0:
        return 0.0
    return float(-(probs * np.log(np.maximum(probs, 1e-12))).sum())


def _log_intrinsic_tree_failure_diagnostics(
        cfg,
        rows: List[Dict[str, Any]],
        selected_rows: List[Dict[str, Any]],
        tb_logger,
        collection_phase: int,
) -> None:
    if not rows or not selected_rows:
        return
    if not any(float(row.get('separate_intrinsic_tree_enabled', 0.0)) > 0.0 for row in selected_rows):
        return

    root_groups: Dict[Tuple[int, int, int, int, int, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        root_groups[_root_row_key(row)].append(row)

    root_count = max(1, len(root_groups))
    one_action_expanded_count = 0
    right_only_expanded_count = 0
    high_final_prob_count = 0
    final_entropies = []
    visit_entropies = []
    max_intrinsic_priors = []
    selected_intrinsic_priors = []
    selected_actions = []
    num_simulations = max(1, int(cfg.policy.get('intrinsic_tree_num_simulations', cfg.policy.num_simulations)))
    near_full_visit_threshold = max(1, num_simulations - 1)

    for root_rows in root_groups.values():
        expanded = [
            row for row in root_rows
            if int(_finite_float(row.get('intrinsic_tree_visit_count'), 0)) > 0
        ]
        max_visits = max(
            [int(_finite_float(row.get('intrinsic_tree_visit_count'), 0)) for row in root_rows] or [0]
        )
        if len(expanded) <= 1 and max_visits >= near_full_visit_threshold:
            one_action_expanded_count += 1
            if expanded and int(_finite_float(expanded[0].get('action'), -1)) == 2:
                right_only_expanded_count += 1
        if max([_finite_float(row.get('final_action_probability'), 0.0) for row in root_rows] or [0.0]) >= 0.95:
            high_final_prob_count += 1
        final_entropies.append(_prob_entropy(root_rows, 'final_action_probability'))
        visit_total = sum(max(0.0, _finite_float(row.get('intrinsic_tree_visit_count'), 0.0)) for row in root_rows)
        if visit_total > 0:
            visit_rows = [
                dict(row, intrinsic_tree_visit_probability=max(
                    0.0, _finite_float(row.get('intrinsic_tree_visit_count'), 0.0)
                ) / visit_total)
                for row in root_rows
            ]
            visit_entropies.append(_prob_entropy(visit_rows, 'intrinsic_tree_visit_probability'))
        max_intrinsic_priors.append(
            max([_finite_float(row.get('intrinsic_tree_prior'), 0.0) for row in root_rows] or [0.0])
        )
        selected = [row for row in root_rows if int(row.get('selected', 0)) == 1]
        if selected:
            selected_intrinsic_priors.append(_finite_float(selected[0].get('intrinsic_tree_prior'), 0.0))
            selected_actions.append(int(_finite_float(selected[0].get('action'), -1)))

    action_counts = Counter(selected_actions)
    repeated_threshold = int(cfg.policy.get('collection_failure_diag_repeated_action_count_threshold', 50))
    high_dd_threshold = float(cfg.policy.get('collection_failure_diag_high_dd_threshold', 0.25))
    low_dd_threshold = float(cfg.policy.get('collection_failure_diag_low_dd_threshold', 0.05))

    repeated_rows = [
        row for row in selected_rows
        if int(row.get('obs_action_bin_prior_count', 0)) >= repeated_threshold
    ]
    repeated_high_pred_dd_rows = [
        row for row in repeated_rows
        if _finite_float(row.get('intrinsic_tree_prediction_projection_dd_immediate'), 0.0) >= high_dd_threshold
    ]
    repeated_high_proj_dd_rows = [
        row for row in repeated_rows
        if _finite_float(row.get('intrinsic_tree_projection_dd_immediate'), 0.0) >= high_dd_threshold
    ]
    expanded_unseen_rows = [
        row for row in rows
        if float(row.get('true_next_obs_bin_unseen', 0.0)) > 0.5
        and int(_finite_float(row.get('intrinsic_tree_visit_count'), 0)) > 0
    ]
    unseen_rows = [row for row in rows if float(row.get('true_next_obs_bin_unseen', 0.0)) > 0.5]
    unseen_unexpanded_rows = [
        row for row in unseen_rows
        if int(_finite_float(row.get('intrinsic_tree_visit_count'), 0)) <= 0
    ]
    unseen_low_pred_dd_rows = [
        row for row in expanded_unseen_rows
        if _finite_float(row.get('intrinsic_tree_prediction_projection_dd_immediate'), 0.0) < low_dd_threshold
    ]
    unseen_low_proj_dd_rows = [
        row for row in expanded_unseen_rows
        if _finite_float(row.get('intrinsic_tree_projection_dd_immediate'), 0.0) < low_dd_threshold
    ]
    unseen_low_intrinsic_rows = [
        row for row in expanded_unseen_rows
        if _finite_float(row.get('intrinsic_tree_intrinsic_immediate_reward'), 0.0) < low_dd_threshold
    ]

    logging.info(
        "[INTRINSIC_TREE_FAILURE_DIAG] phase %s selected_actions L/N/R=%s/%s/%s | "
        "one_action_expanded %.3f, right_only_expanded %.3f, final_prob_ge_0.95 %.3f, "
        "final_entropy %.3f, visit_entropy %.3f, max_prior %.3f, selected_prior %.3f",
        collection_phase,
        int(action_counts.get(0, 0)),
        int(action_counts.get(1, 0)),
        int(action_counts.get(2, 0)),
        float(one_action_expanded_count / root_count),
        float(right_only_expanded_count / root_count),
        float(high_final_prob_count / root_count),
        float(np.mean(final_entropies)) if final_entropies else 0.0,
        float(np.mean(visit_entropies)) if visit_entropies else 0.0,
        float(np.mean(max_intrinsic_priors)) if max_intrinsic_priors else 0.0,
        float(np.mean(selected_intrinsic_priors)) if selected_intrinsic_priors else 0.0,
    )
    logging.info(
        "[INTRINSIC_TREE_NOVELTY_FAILURE_DIAG] phase %s repeated_count>=%s n=%s | "
        "high_pred_dd n=%s pred_dd %.3f ssl_pred %.3f ssl_proj %.3f | "
        "high_proj_dd n=%s proj_dd %.3f | unseen_next n=%s unexpanded %.3f | "
        "expanded_unseen n=%s low_pred_dd %.3f low_proj_dd %.3f low_immediate %.3f",
        collection_phase,
        repeated_threshold,
        len(repeated_rows),
        len(repeated_high_pred_dd_rows),
        _mean_float(repeated_high_pred_dd_rows, 'intrinsic_tree_prediction_projection_dd_immediate'),
        _mean_float(repeated_high_pred_dd_rows, 'real_transition_ssl_prediction_error_mean'),
        _mean_float(repeated_high_pred_dd_rows, 'real_transition_ssl_projection_error_mean'),
        len(repeated_high_proj_dd_rows),
        _mean_float(repeated_high_proj_dd_rows, 'intrinsic_tree_projection_dd_immediate'),
        len(unseen_rows),
        float(len(unseen_unexpanded_rows) / max(1, len(unseen_rows))),
        len(expanded_unseen_rows),
        float(len(unseen_low_pred_dd_rows) / max(1, len(expanded_unseen_rows))),
        float(len(unseen_low_proj_dd_rows) / max(1, len(expanded_unseen_rows))),
        float(len(unseen_low_intrinsic_rows) / max(1, len(expanded_unseen_rows))),
    )

    if tb_logger is None:
        return
    tb_logger.add_scalar(
        'collection_phase/intrinsic_tree_one_action_expanded_fraction',
        float(one_action_expanded_count / root_count),
        collection_phase,
    )
    tb_logger.add_scalar(
        'collection_phase/intrinsic_tree_right_only_expanded_fraction',
        float(right_only_expanded_count / root_count),
        collection_phase,
    )
    tb_logger.add_scalar(
        'collection_phase/intrinsic_tree_final_prob_ge_0_95_fraction',
        float(high_final_prob_count / root_count),
        collection_phase,
    )
    tb_logger.add_scalar(
        'collection_phase/intrinsic_tree_final_policy_entropy',
        float(np.mean(final_entropies)) if final_entropies else 0.0,
        collection_phase,
    )
    tb_logger.add_scalar(
        'collection_phase/intrinsic_tree_visit_entropy',
        float(np.mean(visit_entropies)) if visit_entropies else 0.0,
        collection_phase,
    )
    tb_logger.add_scalar(
        'collection_phase/intrinsic_tree_repeated_high_pred_dd_count',
        float(len(repeated_high_pred_dd_rows)),
        collection_phase,
    )
    tb_logger.add_scalar(
        'collection_phase/intrinsic_tree_repeated_high_pred_dd_ssl_pred_error',
        _mean_float(repeated_high_pred_dd_rows, 'real_transition_ssl_prediction_error_mean'),
        collection_phase,
    )
    tb_logger.add_scalar(
        'collection_phase/intrinsic_tree_unseen_next_unexpanded_fraction',
        float(len(unseen_unexpanded_rows) / max(1, len(unseen_rows))),
        collection_phase,
    )
    tb_logger.add_scalar(
        'collection_phase/intrinsic_tree_expanded_unseen_low_pred_dd_fraction',
        float(len(unseen_low_pred_dd_rows) / max(1, len(expanded_unseen_rows))),
        collection_phase,
    )
    tb_logger.add_scalar(
        'collection_phase/intrinsic_tree_expanded_unseen_low_immediate_fraction',
        float(len(unseen_low_intrinsic_rows) / max(1, len(expanded_unseen_rows))),
        collection_phase,
    )


def _real_observation_count(cfg, valid_step_count: int, obs_count: int) -> int:
    frame_stack_num = int(cfg.policy.model.get('frame_stack_num', 1))
    return min(int(obs_count), frame_stack_num + int(valid_step_count))


def _decision_obs_index(cfg, timestep: int, obs_count: int) -> int:
    frame_stack_num = int(cfg.policy.model.get('frame_stack_num', 1))
    return min(max(0, int(timestep) + frame_stack_num - 1), max(0, int(obs_count) - 1))


def _fresh_collection_next_observations(cfg, new_data: List[Any]) -> np.ndarray:
    """Return each real next observation from this collection phase exactly once."""
    observations = []
    segments, metadata = new_data
    for game_segment, meta in zip(segments, metadata):
        valid_step_count = min(
            int(meta.get('segment_length', len(game_segment))),
            len(getattr(game_segment, 'action_segment', [])),
        )
        if valid_step_count <= 0:
            continue
        obs_segment = np.asarray(game_segment.obs_segment)
        obs_count = _real_observation_count(cfg, valid_step_count, len(obs_segment))
        for timestep in range(valid_step_count):
            next_obs_index = _decision_obs_index(cfg, timestep + 1, obs_count)
            if next_obs_index < len(obs_segment):
                observations.append(
                    np.asarray(obs_segment[next_obs_index], dtype=np.float32)
                )
    if not observations:
        return np.empty((0, 0), dtype=np.float32)
    return np.stack(observations, axis=0)


def _train_rnd_predictor_from_fresh_collection(
        cfg,
        policy,
        new_data: List[Any],
        tb_logger,
        collection_phase: int,
) -> Dict[str, float]:
    if (
            not bool(cfg.policy.get('rnd_train_from_fresh_collection_only', False))
            or str(cfg.policy.get('intrinsic_reward_source', 'rnd')) != 'rnd'
    ):
        return {}
    train_fn = getattr(policy, 'train_rnd_on_fresh_observations', None)
    if not callable(train_fn):
        raise AttributeError(
            "rnd_train_from_fresh_collection_only requires "
            "policy.train_rnd_on_fresh_observations()."
        )

    observations = _fresh_collection_next_observations(cfg, new_data)
    stats = train_fn(observations)
    if not stats:
        return {}
    stats = dict(stats)

    refresh_after_update = bool(
        cfg.policy.get('rnd_target_snapshot_refresh_after_fresh_update', False)
    )
    optimizer_step_count = int(stats.get('rnd_fresh/optimizer_step_count', 0.0))
    snapshot_refreshed = False
    if refresh_after_update and optimizer_step_count > 0:
        if not bool(cfg.policy.get('rnd_target_snapshot', True)):
            raise ValueError(
                "rnd_target_snapshot_refresh_after_fresh_update requires "
                "rnd_target_snapshot=True."
            )
        if not bool(cfg.policy.get('rnd_target_snapshot_refresh_per_collect', False)):
            raise ValueError(
                "rnd_target_snapshot_refresh_after_fresh_update requires "
                "rnd_target_snapshot_refresh_per_collect=True."
            )
        refresh_fn = getattr(policy, 'refresh_rnd_target_snapshot_for_collection', None)
        if not callable(refresh_fn):
            raise AttributeError(
                "rnd_target_snapshot_refresh_after_fresh_update requires "
                "policy.refresh_rnd_target_snapshot_for_collection()."
            )
        refresh_fn(collection_completed=True)
        snapshot_refreshed = True
    stats['rnd_fresh/target_snapshot_refreshed_after_update'] = float(snapshot_refreshed)

    logging.info(
        "[RND_FRESH] phase %s observations=%d selected=%d steps=%d "
        "batch(min/mean/max)=%.0f/%.1f/%.0f pending=%d "
        "error=%.6g->%.6g ratio=%.4f target_snapshot_refreshed=%d",
        collection_phase,
        int(stats.get('rnd_fresh/observation_count', 0.0)),
        int(stats.get('rnd_fresh/selected_count', 0.0)),
        int(stats.get('rnd_fresh/optimizer_step_count', 0.0)),
        float(stats.get('rnd_fresh/batch_size_min', 0.0)),
        float(stats.get('rnd_fresh/batch_size_mean', 0.0)),
        float(stats.get('rnd_fresh/batch_size_max', 0.0)),
        int(stats.get('rnd_fresh/pending_count', 0.0)),
        float(stats.get('rnd_fresh/pre_error_mean', 0.0)),
        float(stats.get('rnd_fresh/post_error_mean', 0.0)),
        float(stats.get('rnd_fresh/post_pre_error_ratio', 0.0)),
        int(snapshot_refreshed),
    )
    if tb_logger is not None:
        for key, value in stats.items():
            tb_logger.add_scalar(key, float(value), collection_phase)
    return stats


def _future_trajectory_metrics(cfg, game_segment: Any, timestep: int, valid_step_count: int) -> Dict[str, float]:
    obs_segment = np.asarray(game_segment.obs_segment)
    reward_segment = game_segment.reward_segment
    obs_count = _real_observation_count(cfg, valid_step_count, len(obs_segment))
    obs_index = _decision_obs_index(cfg, timestep, obs_count)
    position, velocity = _state_components(obs_segment[obs_index])
    metrics = {
        'obs_position': position,
        'obs_velocity': velocity,
    }

    horizons = [20, 50, 100]
    for horizon in horizons:
        obs_end = min(obs_count, obs_index + int(horizon) + 1)
        future_obs = obs_segment[obs_index:obs_end]
        positions = []
        abs_velocities = []
        for obs in future_obs:
            pos, vel = _state_components(obs)
            if not np.isnan(pos):
                positions.append(pos)
            if not np.isnan(vel):
                abs_velocities.append(abs(vel))
        if positions:
            max_position = float(max(positions))
            min_position = float(min(positions))
        else:
            max_position = float('nan')
            min_position = float('nan')
        max_abs_velocity = float(max(abs_velocities)) if abs_velocities else float('nan')
        reward_end = min(int(valid_step_count), int(timestep) + int(horizon))
        future_return = float(sum(_scalar_reward(reward) for reward in reward_segment[int(timestep):reward_end]))
        metrics.update(
            {
                f'future_return_{horizon}': future_return,
                f'future_max_position_{horizon}': max_position,
                f'future_min_position_{horizon}': min_position,
                f'future_position_gain_{horizon}': max_position - position if not np.isnan(max_position) else float('nan'),
                f'future_max_abs_velocity_{horizon}': max_abs_velocity,
                f'future_reached_goal_{horizon}': float(max_position >= 0.5) if not np.isnan(max_position) else 0.0,
            }
        )

    remaining_obs = obs_segment[obs_index:obs_count]
    remaining_positions = []
    remaining_abs_velocities = []
    for obs in remaining_obs:
        pos, vel = _state_components(obs)
        if not np.isnan(pos):
            remaining_positions.append(pos)
        if not np.isnan(vel):
            remaining_abs_velocities.append(abs(vel))
    remaining_max_position = float(max(remaining_positions)) if remaining_positions else float('nan')
    remaining_min_position = float(min(remaining_positions)) if remaining_positions else float('nan')
    remaining_max_abs_velocity = float(max(remaining_abs_velocities)) if remaining_abs_velocities else float('nan')
    remaining_return = float(
        sum(_scalar_reward(reward) for reward in reward_segment[int(timestep):int(valid_step_count)])
    )
    metrics.update(
        {
            'future_return_remaining': remaining_return,
            'future_max_position_remaining': remaining_max_position,
            'future_min_position_remaining': remaining_min_position,
            'future_position_gain_remaining': (
                remaining_max_position - position if not np.isnan(remaining_max_position) else float('nan')
            ),
            'future_max_abs_velocity_remaining': remaining_max_abs_velocity,
            'future_reached_goal_remaining': (
                float(remaining_max_position >= 0.5) if not np.isnan(remaining_max_position) else 0.0
            ),
        }
    )
    return metrics


def _segment_state_summary(cfg, game_segment: Any, valid_step_count: int) -> Dict[str, float]:
    obs_segment = np.asarray(game_segment.obs_segment)
    obs_count = _real_observation_count(cfg, valid_step_count, len(obs_segment))
    positions = []
    abs_velocities = []
    for obs in obs_segment[:obs_count]:
        position, velocity = _state_components(obs)
        if not np.isnan(position):
            positions.append(position)
        if not np.isnan(velocity):
            abs_velocities.append(abs(velocity))
    if not positions:
        return {
            'segment_max_position': float('nan'),
            'segment_min_position': float('nan'),
            'segment_max_abs_velocity': float('nan'),
            'segment_reached_goal': 0.0,
        }
    max_position = float(max(positions))
    return {
        'segment_max_position': max_position,
        'segment_min_position': float(min(positions)),
        'segment_max_abs_velocity': float(max(abs_velocities)) if abs_velocities else float('nan'),
        'segment_reached_goal': float(max_position >= 0.5),
    }


def _pairwise_feature_disagreement(
        features: torch.Tensor,
        ensemble_size: int,
        batch_size: int,
        normalize: bool,
        metric: str,
) -> torch.Tensor:
    features = features.reshape(int(ensemble_size), int(batch_size), -1).float()
    if normalize:
        features = torch.nn.functional.normalize(features, p=2, dim=-1, eps=1e-8)
    metric = str(metric).lower()
    distances = []
    for member_i in range(int(ensemble_size)):
        for member_j in range(member_i + 1, int(ensemble_size)):
            delta = features[member_i] - features[member_j]
            if metric == 'l2':
                distances.append(delta.pow(2).sum(dim=-1).sqrt())
            else:
                distances.append(delta.pow(2).mean(dim=-1))
    if not distances:
        return torch.zeros(int(batch_size), dtype=torch.float32, device=features.device)
    return torch.stack(distances, dim=0).mean(dim=0)


def _compute_collection_selected_transition_ssl_diagnostics(
        cfg,
        policy,
        new_data: List[Any],
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    if not bool(cfg.policy.get('collection_ssl_diagnostics', False)):
        return {}
    if not bool(cfg.policy.model.get('self_supervised_learning_loss', False)):
        return {}

    model = getattr(policy, '_collect_model', None)
    if model is None:
        model = getattr(policy, '_learn_model', None)
    if model is None or not all(hasattr(model, name) for name in ['initial_inference', 'recurrent_inference', 'project']):
        return {}

    samples = []
    data, meta_list = new_data
    frame_stack_num = int(cfg.policy.model.get('frame_stack_num', 1))
    for segment_index, (game_segment, meta) in enumerate(zip(data, meta_list)):
        diagnostics_segment = getattr(game_segment, 'root_action_diagnostics_segment', [])
        if not diagnostics_segment:
            continue
        valid_step_count = min(
            int(meta.get('segment_length', len(game_segment))),
            len(diagnostics_segment),
            len(game_segment.action_segment),
        )
        obs_count = len(game_segment.obs_segment)
        for timestep in range(valid_step_count):
            if int(timestep) + frame_stack_num >= obs_count:
                continue
            try:
                obs_unroll = game_segment.get_unroll_obs(timestep, num_unroll_steps=1, padding=False)
            except Exception:
                continue
            if len(obs_unroll) < frame_stack_num + 1:
                continue
            samples.append(
                {
                    'key': (int(segment_index), int(timestep)),
                    'obs_unroll': obs_unroll,
                    'action': _scalar_action(game_segment.action_segment[timestep]),
                }
            )
    if not samples:
        return {}

    device = torch.device(cfg.policy.device)
    ensemble_size = int(cfg.policy.model.get('ensemble_size', 1))
    batch_size_limit = max(1, int(cfg.policy.get('collection_ssl_diagnostics_batch_size', 512)))
    normalize_for_dd = bool(cfg.policy.get('dynamics_disagreement_normalize_latent', True))
    dd_metric = str(cfg.policy.get('dynamics_disagreement_distance_metric', 'mse') or 'mse').lower()
    if dd_metric not in {'mse', 'l2'}:
        dd_metric = 'mse'

    result: Dict[Tuple[int, int], Dict[str, Any]] = {}
    was_training = model.training
    active_member_state = getattr(model, 'active_member_id', None)
    model.eval()
    try:
        with torch.no_grad():
            for batch_start in range(0, len(samples), batch_size_limit):
                batch = samples[batch_start:batch_start + batch_size_limit]
                obs_batch_ori = prepare_observation(
                    [sample['obs_unroll'] for sample in batch],
                    cfg.policy.model.model_type,
                )
                obs_batch, obs_target_batch = prepare_obs(obs_batch_ori, cfg.policy)
                if obs_target_batch is None:
                    continue
                obs_batch = obs_batch.float()
                obs_target_batch = obs_target_batch.float()
                target_obs_batch = obs_target_batch[:, :obs_batch.shape[1]]
                batch_size = int(obs_batch.shape[0])
                member_ids = [
                    member_id
                    for member_id in range(ensemble_size)
                    for _ in range(batch_size)
                ]
                repeated_obs = torch.cat([obs_batch for _ in range(ensemble_size)], dim=0)
                repeated_target_obs = torch.cat([target_obs_batch for _ in range(ensemble_size)], dim=0)
                action_batch = torch.as_tensor(
                    [sample['action'] for sample in batch],
                    dtype=torch.long,
                    device=device,
                ).view(batch_size, 1)
                repeated_action = torch.cat([action_batch for _ in range(ensemble_size)], dim=0)

                root_output = model.initial_inference(repeated_obs, member_id=member_ids)
                pred_output = model.recurrent_inference(
                    root_output.latent_state.detach(),
                    repeated_action,
                    member_id=member_ids,
                )
                target_output = model.initial_inference(repeated_target_obs, member_id=member_ids)

                pred_latent = pred_output.latent_state.detach()
                target_latent = target_output.latent_state.detach()
                pred_projection = model.project(pred_latent, with_grad=False, member_id=member_ids).detach()
                pred_prediction_projection = model.project(pred_latent, with_grad=True, member_id=member_ids).detach()
                target_projection = model.project(target_latent, with_grad=False, member_id=member_ids).detach()

                pred_projection_flat = pred_projection.reshape(ensemble_size, batch_size, -1).float()
                pred_prediction_flat = pred_prediction_projection.reshape(ensemble_size, batch_size, -1).float()
                target_projection_flat = target_projection.reshape(ensemble_size, batch_size, -1).float()

                projection_cos = torch.nn.functional.cosine_similarity(
                    pred_projection_flat,
                    target_projection_flat,
                    dim=-1,
                    eps=1e-8,
                )
                prediction_cos = torch.nn.functional.cosine_similarity(
                    pred_prediction_flat,
                    target_projection_flat,
                    dim=-1,
                    eps=1e-8,
                )
                projection_error = 1.0 - projection_cos
                prediction_error = 1.0 - prediction_cos

                projection_dd = _pairwise_feature_disagreement(
                    pred_projection,
                    ensemble_size,
                    batch_size,
                    normalize_for_dd,
                    dd_metric,
                )
                prediction_dd = _pairwise_feature_disagreement(
                    pred_prediction_projection,
                    ensemble_size,
                    batch_size,
                    normalize_for_dd,
                    dd_metric,
                )
                target_projection_dd = _pairwise_feature_disagreement(
                    target_projection,
                    ensemble_size,
                    batch_size,
                    normalize_for_dd,
                    dd_metric,
                )
                target_latent_dd = _pairwise_feature_disagreement(
                    target_latent,
                    ensemble_size,
                    batch_size,
                    normalize_for_dd,
                    dd_metric,
                )

                for local_index, sample in enumerate(batch):
                    member_projection_errors = projection_error[:, local_index].detach().cpu().numpy().astype(
                        np.float32
                    )
                    member_prediction_errors = prediction_error[:, local_index].detach().cpu().numpy().astype(
                        np.float32
                    )
                    result[sample['key']] = {
                        'real_transition_ssl_valid': 1.0,
                        'real_transition_ssl_projection_cos_mean': float(
                            projection_cos[:, local_index].mean().item()
                        ),
                        'real_transition_ssl_projection_error_mean': float(member_projection_errors.mean()),
                        'real_transition_ssl_projection_error_std': float(member_projection_errors.std()),
                        'real_transition_ssl_projection_error_max': float(member_projection_errors.max()),
                        'real_transition_ssl_prediction_cos_mean': float(
                            prediction_cos[:, local_index].mean().item()
                        ),
                        'real_transition_ssl_prediction_error_mean': float(member_prediction_errors.mean()),
                        'real_transition_ssl_prediction_error_std': float(member_prediction_errors.std()),
                        'real_transition_ssl_prediction_error_max': float(member_prediction_errors.max()),
                        'real_transition_projection_pairwise_dd': float(projection_dd[local_index].item()),
                        'real_transition_prediction_projection_pairwise_dd': float(
                            prediction_dd[local_index].item()
                        ),
                        'real_transition_target_projection_pairwise_dd': float(
                            target_projection_dd[local_index].item()
                        ),
                        'real_transition_target_latent_pairwise_dd': float(target_latent_dd[local_index].item()),
                        'real_transition_member_projection_ssl_errors': member_projection_errors.tolist(),
                        'real_transition_member_prediction_ssl_errors': member_prediction_errors.tolist(),
                    }
    finally:
        _restore_model_member_state(model, active_member_state)
        model.train(was_training)
    return result


def _nan_ssl_diagnostics() -> Dict[str, Any]:
    return {
        'real_transition_ssl_valid': 0.0,
        'real_transition_ssl_projection_cos_mean': float('nan'),
        'real_transition_ssl_projection_error_mean': float('nan'),
        'real_transition_ssl_projection_error_std': float('nan'),
        'real_transition_ssl_projection_error_max': float('nan'),
        'real_transition_ssl_prediction_cos_mean': float('nan'),
        'real_transition_ssl_prediction_error_mean': float('nan'),
        'real_transition_ssl_prediction_error_std': float('nan'),
        'real_transition_ssl_prediction_error_max': float('nan'),
        'real_transition_projection_pairwise_dd': float('nan'),
        'real_transition_prediction_projection_pairwise_dd': float('nan'),
        'real_transition_target_projection_pairwise_dd': float('nan'),
        'real_transition_target_latent_pairwise_dd': float('nan'),
        'real_transition_member_projection_ssl_errors': [],
        'real_transition_member_prediction_ssl_errors': [],
    }


def _restore_model_buffers(model: torch.nn.Module, buffer_snapshot: Dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, buffer in model.named_buffers():
            if name in buffer_snapshot:
                buffer.copy_(buffer_snapshot[name])


def _latent_diversity_statistics(latent: torch.Tensor) -> Dict[str, float]:
    latent = latent.detach().reshape(latent.shape[0], -1).float()
    feature_std = latent.std(dim=0, unbiased=False)
    normalized = torch.nn.functional.normalize(latent, dim=-1, eps=1e-8)
    sample_count = int(latent.shape[0])
    if sample_count > 1:
        cosine_sum = normalized.sum(dim=0).square().sum() - sample_count
        pairwise_cosine = cosine_sum / (sample_count * (sample_count - 1))
    else:
        pairwise_cosine = latent.new_tensor(float('nan'))

    singular_values = torch.linalg.svdvals(latent - latent.mean(dim=0, keepdim=True))
    spectrum = singular_values.square()
    spectrum_sum = spectrum.sum()
    if float(spectrum_sum.item()) > 1e-12:
        probabilities = spectrum / spectrum_sum
        effective_rank = torch.exp(
            -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
        )
    else:
        effective_rank = latent.new_tensor(0.0)

    return {
        'latent_feature_std_mean': float(feature_std.mean().item()),
        'latent_feature_std_min': float(feature_std.min().item()),
        'latent_feature_std_fraction_above_1e3': float((feature_std > 1e-3).float().mean().item()),
        'latent_pairwise_cosine_mean': float(pairwise_cosine.item()),
        'latent_effective_rank': float(effective_rank.item()),
    }


def _tensor_correlation(first: torch.Tensor, second: torch.Tensor) -> float:
    first = first.detach().reshape(-1).float()
    second = second.detach().reshape(-1).float()
    finite = torch.isfinite(first) & torch.isfinite(second)
    if int(finite.sum().item()) < 2:
        return float('nan')
    first = first[finite] - first[finite].mean()
    second = second[finite] - second[finite].mean()
    denominator = first.square().sum().sqrt() * second.square().sum().sqrt()
    if float(denominator.item()) <= 1e-12:
        return 0.0
    return float((first * second).sum().div(denominator).item())


def _write_normalization_mode_diagnostics(
        cfg,
        policy,
        new_data: List[Any],
        tb_logger,
        collection_phase: int,
        train_iter: int,
) -> None:
    if not bool(cfg.policy.get('normalization_mode_diagnostics', False)):
        return
    if not bool(cfg.policy.model.get('self_supervised_learning_loss', False)):
        logging.warning('[NORM_DIAGNOSTICS] skipped: self-supervised projection heads are disabled')
        return

    model = getattr(policy, '_collect_model', None)
    if model is None:
        model = getattr(policy, '_learn_model', None)
    required_methods = ['initial_inference', 'recurrent_inference', 'project']
    if model is None or not all(hasattr(model, method) for method in required_methods):
        logging.warning('[NORM_DIAGNOSTICS] skipped: policy model does not expose recurrent SSL inference')
        return

    max_depth = max(1, int(cfg.policy.get('normalization_mode_diagnostics_depth', 5)))
    sample_size = max(2, int(cfg.policy.get('normalization_mode_diagnostics_sample_size', 256)))
    frame_stack_num = int(cfg.policy.model.get('frame_stack_num', 1))
    candidates = []
    data, meta_list = new_data
    for segment_index, (game_segment, meta) in enumerate(zip(data, meta_list)):
        valid_step_count = min(
            int(meta.get('segment_length', len(game_segment))),
            len(game_segment.action_segment),
        )
        final_start = valid_step_count - max_depth
        if final_start < 0:
            continue
        obs_count = len(game_segment.obs_segment)
        for timestep in range(final_start + 1):
            if timestep + frame_stack_num + max_depth > obs_count:
                continue
            candidates.append((segment_index, timestep))
    if len(candidates) < 2:
        logging.warning(
            '[NORM_DIAGNOSTICS] phase %s skipped: only %s complete depth-%s windows',
            collection_phase,
            len(candidates),
            max_depth,
        )
        return

    if len(candidates) > sample_size:
        selected_indices = np.linspace(0, len(candidates) - 1, num=sample_size, dtype=np.int64)
        candidates = [candidates[int(index)] for index in selected_indices]

    obs_unrolls = []
    action_sequences = []
    for segment_index, timestep in candidates:
        game_segment = data[segment_index]
        try:
            obs_unroll = game_segment.get_unroll_obs(
                timestep,
                num_unroll_steps=max_depth,
                padding=False,
            )
        except Exception:
            continue
        if len(obs_unroll) < frame_stack_num + max_depth:
            continue
        actions = [
            _scalar_action(game_segment.action_segment[timestep + offset])
            for offset in range(max_depth)
        ]
        obs_unrolls.append(obs_unroll)
        action_sequences.append(actions)
    if len(obs_unrolls) < 2:
        logging.warning('[NORM_DIAGNOSTICS] phase %s skipped: no valid observation windows', collection_phase)
        return

    obs_batch_ori = prepare_observation(obs_unrolls, cfg.policy.model.model_type)
    obs_batch, obs_target_batch = prepare_obs(obs_batch_ori, cfg.policy)
    if obs_target_batch is None:
        logging.warning('[NORM_DIAGNOSTICS] skipped: target observations were not prepared')
        return
    obs_batch = obs_batch.float()
    obs_target_batch = obs_target_batch.float()
    action_batch = torch.as_tensor(
        action_sequences,
        dtype=torch.long,
        device=obs_batch.device,
    )

    batch_size = int(obs_batch.shape[0])
    ensemble_size = int(cfg.policy.model.get('ensemble_size', 1))
    member_ids = [
        member_id
        for member_id in range(ensemble_size)
        for _ in range(batch_size)
    ]
    repeated_obs = torch.cat([obs_batch for _ in range(ensemble_size)], dim=0)
    observation_width = int(cfg.policy.model.observation_shape)
    target_stack_width = observation_width * frame_stack_num
    inverse_value = policy.value_inverse_scalar_transform_handle

    was_training = model.training
    active_member_state = getattr(model, 'active_member_id', None)
    buffer_snapshot = {
        name: buffer.detach().clone()
        for name, buffer in model.named_buffers()
    }
    rows = []
    mode_outputs = {}
    try:
        for mode_name, training in [('train', True), ('eval', False)]:
            _restore_model_buffers(model, buffer_snapshot)
            model.train(training)
            with torch.no_grad():
                root_output = model.initial_inference(repeated_obs, member_id=member_ids)
                imagined_latent = root_output.latent_state.detach()
                for depth in range(1, max_depth + 1):
                    repeated_action = torch.cat(
                        [action_batch[:, depth - 1:depth] for _ in range(ensemble_size)],
                        dim=0,
                    )
                    imagined_output = model.recurrent_inference(
                        imagined_latent,
                        repeated_action,
                        member_id=member_ids,
                    )
                    imagined_latent = imagined_output.latent_state.detach()

                    target_begin = observation_width * (depth - 1)
                    target_end = target_begin + target_stack_width
                    direct_obs = obs_target_batch[:, target_begin:target_end]
                    repeated_direct_obs = torch.cat([direct_obs for _ in range(ensemble_size)], dim=0)
                    direct_output = model.initial_inference(repeated_direct_obs, member_id=member_ids)
                    direct_latent = direct_output.latent_state.detach()

                    imagined_projection = model.project(
                        imagined_latent,
                        with_grad=True,
                        member_id=member_ids,
                    ).detach()
                    direct_projection = model.project(
                        direct_latent,
                        with_grad=False,
                        member_id=member_ids,
                    ).detach()
                    latent_cosine = torch.nn.functional.cosine_similarity(
                        imagined_latent.reshape(imagined_latent.shape[0], -1).float(),
                        direct_latent.reshape(direct_latent.shape[0], -1).float(),
                        dim=-1,
                        eps=1e-8,
                    ).reshape(ensemble_size, batch_size)
                    ssl_cosine = torch.nn.functional.cosine_similarity(
                        imagined_projection.reshape(imagined_projection.shape[0], -1).float(),
                        direct_projection.reshape(direct_projection.shape[0], -1).float(),
                        dim=-1,
                        eps=1e-8,
                    ).reshape(ensemble_size, batch_size)
                    imagined_value = inverse_value(imagined_output.value).detach().reshape(
                        ensemble_size, batch_size
                    ).float()
                    direct_value = inverse_value(direct_output.value).detach().reshape(
                        ensemble_size, batch_size
                    ).float()
                    value_gap = imagined_value - direct_value
                    imagined_intrinsic_value = model.intrinsic_value_from_latent(
                        imagined_latent, member_id=member_ids
                    ).detach().reshape(ensemble_size, batch_size).float()
                    direct_intrinsic_value = model.intrinsic_value_from_latent(
                        direct_latent, member_id=member_ids
                    ).detach().reshape(ensemble_size, batch_size).float()
                    intrinsic_value_gap = imagined_intrinsic_value - direct_intrinsic_value
                    imagined_policy = torch.softmax(imagined_output.policy_logits.detach(), dim=-1).reshape(
                        ensemble_size, batch_size, -1
                    ).float()
                    direct_policy = torch.softmax(direct_output.policy_logits.detach(), dim=-1).reshape(
                        ensemble_size, batch_size, -1
                    ).float()
                    direct_latent_by_member = direct_latent.reshape(ensemble_size, batch_size, -1)
                    member_diversity = [
                        _latent_diversity_statistics(direct_latent_by_member[member_id])
                        for member_id in range(ensemble_size)
                    ]
                    mode_outputs[(mode_name, depth)] = {
                        'imagined_value': imagined_value.detach().cpu(),
                        'direct_value': direct_value.detach().cpu(),
                        'imagined_policy': imagined_policy.detach().cpu(),
                        'direct_policy': direct_policy.detach().cpu(),
                    }

                    for member_id in [-1, *range(ensemble_size)]:
                        member_slice = slice(None) if member_id < 0 else member_id
                        member_latent_cosine = latent_cosine[member_slice].reshape(-1)
                        member_ssl_cosine = ssl_cosine[member_slice].reshape(-1)
                        member_imagined_value = imagined_value[member_slice].reshape(-1)
                        member_direct_value = direct_value[member_slice].reshape(-1)
                        member_value_gap = value_gap[member_slice].reshape(-1)
                        member_imagined_intrinsic_value = imagined_intrinsic_value[member_slice].reshape(-1)
                        member_direct_intrinsic_value = direct_intrinsic_value[member_slice].reshape(-1)
                        member_intrinsic_value_gap = intrinsic_value_gap[member_slice].reshape(-1)
                        if member_id < 0:
                            diversity = {
                                key: float(np.mean([stats[key] for stats in member_diversity]))
                                for key in member_diversity[0]
                            }
                        else:
                            diversity = member_diversity[member_id]
                        rows.append(
                            {
                                'collection_phase': int(collection_phase),
                                'train_iter_before_update': int(train_iter),
                                'sample_count': int(member_value_gap.numel()),
                                'mode': mode_name,
                                'depth': int(depth),
                                'member_id': int(member_id),
                                'latent_cosine_mean': float(member_latent_cosine.mean().item()),
                                'ssl_cosine_mean': float(member_ssl_cosine.mean().item()),
                                'imagined_value_mean': float(member_imagined_value.mean().item()),
                                'direct_value_mean': float(member_direct_value.mean().item()),
                                'value_gap_mean': float(member_value_gap.mean().item()),
                                'value_gap_abs_mean': float(member_value_gap.abs().mean().item()),
                                'value_gap_abs_p90': float(
                                    torch.quantile(member_value_gap.abs(), 0.9).item()
                                ),
                                'imagined_intrinsic_value_mean': float(
                                    member_imagined_intrinsic_value.mean().item()
                                ),
                                'direct_intrinsic_value_mean': float(
                                    member_direct_intrinsic_value.mean().item()
                                ),
                                'intrinsic_value_gap_mean': float(
                                    member_intrinsic_value_gap.mean().item()
                                ),
                                'intrinsic_value_gap_abs_mean': float(
                                    member_intrinsic_value_gap.abs().mean().item()
                                ),
                                'intrinsic_value_gap_abs_p90': float(
                                    torch.quantile(member_intrinsic_value_gap.abs(), 0.9).item()
                                ),
                                'intrinsic_value_direct_imagined_correlation': _tensor_correlation(
                                    member_imagined_intrinsic_value,
                                    member_direct_intrinsic_value,
                                ),
                                **diversity,
                            }
                        )
    finally:
        _restore_model_buffers(model, buffer_snapshot)
        _restore_model_member_state(model, active_member_state)
        model.train(was_training)

    row_lookup = {
        (row['mode'], row['depth'], row['member_id']): row
        for row in rows
    }
    for depth in range(1, max_depth + 1):
        train_outputs = mode_outputs[('train', depth)]
        eval_outputs = mode_outputs[('eval', depth)]
        for member_id in [-1, *range(ensemble_size)]:
            member_slice = slice(None) if member_id < 0 else member_id
            imagined_value_difference = (
                train_outputs['imagined_value'][member_slice]
                - eval_outputs['imagined_value'][member_slice]
            ).abs().reshape(-1)
            direct_value_difference = (
                train_outputs['direct_value'][member_slice]
                - eval_outputs['direct_value'][member_slice]
            ).abs().reshape(-1)
            imagined_policy_tv = 0.5 * (
                train_outputs['imagined_policy'][member_slice]
                - eval_outputs['imagined_policy'][member_slice]
            ).abs().sum(dim=-1).reshape(-1)
            direct_policy_tv = 0.5 * (
                train_outputs['direct_policy'][member_slice]
                - eval_outputs['direct_policy'][member_slice]
            ).abs().sum(dim=-1).reshape(-1)
            cross_mode = {
                'imagined_value_train_eval_abs_mean': float(imagined_value_difference.mean().item()),
                'imagined_value_train_eval_abs_p90': float(torch.quantile(imagined_value_difference, 0.9).item()),
                'direct_value_train_eval_abs_mean': float(direct_value_difference.mean().item()),
                'direct_value_train_eval_abs_p90': float(torch.quantile(direct_value_difference, 0.9).item()),
                'imagined_policy_train_eval_tv_mean': float(imagined_policy_tv.mean().item()),
                'imagined_policy_train_eval_tv_p90': float(torch.quantile(imagined_policy_tv, 0.9).item()),
                'direct_policy_train_eval_tv_mean': float(direct_policy_tv.mean().item()),
                'direct_policy_train_eval_tv_p90': float(torch.quantile(direct_policy_tv, 0.9).item()),
            }
            row_lookup[('train', depth, member_id)].update(cross_mode)
            row_lookup[('eval', depth, member_id)].update(cross_mode)

    aggregate_rows = [row for row in rows if row['member_id'] == -1]
    by_mode_depth = {(row['mode'], row['depth']): row for row in aggregate_rows}
    for depth in range(1, max_depth + 1):
        train_row = by_mode_depth[('train', depth)]
        eval_row = by_mode_depth[('eval', depth)]
        logging.info(
            '[NORM_DIAGNOSTICS] phase %s iter %s depth %s samples %s | '
            'SSL cosine train %.4f eval %.4f | latent cosine train %.4f eval %.4f | '
            'value gap train %+.2f eval %+.2f (eval abs-p90 %.2f)',
            collection_phase,
            train_iter,
            depth,
            batch_size,
            train_row['ssl_cosine_mean'],
            eval_row['ssl_cosine_mean'],
            train_row['latent_cosine_mean'],
            eval_row['latent_cosine_mean'],
            train_row['value_gap_mean'],
            eval_row['value_gap_mean'],
            eval_row['value_gap_abs_p90'],
        )
        logging.info(
            '[NORM_DIAGNOSTICS] phase %s iter %s depth %s | '
            'cross-mode value abs-mean direct %.3f imagined %.3f | '
            'policy TV direct %.4f imagined %.4f | '
            'eval latent feature-std %.5f active %.3f effective-rank %.2f pair-cos %.4f',
            collection_phase,
            train_iter,
            depth,
            train_row['direct_value_train_eval_abs_mean'],
            train_row['imagined_value_train_eval_abs_mean'],
            train_row['direct_policy_train_eval_tv_mean'],
            train_row['imagined_policy_train_eval_tv_mean'],
            eval_row['latent_feature_std_mean'],
            eval_row['latent_feature_std_fraction_above_1e3'],
            eval_row['latent_effective_rank'],
            eval_row['latent_pairwise_cosine_mean'],
        )
        logging.info(
            '[INTRINSIC_VALUE_CONSISTENCY] phase %s iter %s depth %s | '
            'imagined %.5f direct %.5f gap %+.5f abs-mean %.5f abs-p90 %.5f corr %.4f',
            collection_phase,
            train_iter,
            depth,
            eval_row['imagined_intrinsic_value_mean'],
            eval_row['direct_intrinsic_value_mean'],
            eval_row['intrinsic_value_gap_mean'],
            eval_row['intrinsic_value_gap_abs_mean'],
            eval_row['intrinsic_value_gap_abs_p90'],
            eval_row['intrinsic_value_direct_imagined_correlation'],
        )
        if tb_logger is not None:
            for mode_name, row in [('train', train_row), ('eval', eval_row)]:
                prefix = f'normalization_mode/{mode_name}/depth_{depth}'
                tb_logger.add_scalar(f'{prefix}/ssl_cosine_mean', row['ssl_cosine_mean'], collection_phase)
                tb_logger.add_scalar(
                    f'{prefix}/latent_cosine_mean', row['latent_cosine_mean'], collection_phase
                )
                tb_logger.add_scalar(f'{prefix}/value_gap_mean', row['value_gap_mean'], collection_phase)
                tb_logger.add_scalar(
                    f'{prefix}/value_gap_abs_p90', row['value_gap_abs_p90'], collection_phase
                )
                tb_logger.add_scalar(
                    f'{prefix}/latent_feature_std_mean', row['latent_feature_std_mean'], collection_phase
                )
                tb_logger.add_scalar(
                    f'{prefix}/latent_effective_rank', row['latent_effective_rank'], collection_phase
                )
                tb_logger.add_scalar(
                    f'{prefix}/latent_pairwise_cosine_mean', row['latent_pairwise_cosine_mean'], collection_phase
                )
                for metric_name in [
                    'imagined_intrinsic_value_mean',
                    'direct_intrinsic_value_mean',
                    'intrinsic_value_gap_mean',
                    'intrinsic_value_gap_abs_mean',
                    'intrinsic_value_gap_abs_p90',
                    'intrinsic_value_direct_imagined_correlation',
                ]:
                    tb_logger.add_scalar(
                        f'{prefix}/{metric_name}', row[metric_name], collection_phase
                    )
            cross_prefix = f'normalization_mode/cross_mode/depth_{depth}'
            for metric_name in [
                'imagined_value_train_eval_abs_mean',
                'imagined_value_train_eval_abs_p90',
                'direct_value_train_eval_abs_mean',
                'direct_value_train_eval_abs_p90',
                'imagined_policy_train_eval_tv_mean',
                'imagined_policy_train_eval_tv_p90',
                'direct_policy_train_eval_tv_mean',
                'direct_policy_train_eval_tv_p90',
            ]:
                tb_logger.add_scalar(
                    f'{cross_prefix}/{metric_name}', train_row[metric_name], collection_phase
                )

    csv_path = os.path.join(_diagnostics_dir(cfg), 'normalization_mode_diagnostics.csv')
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _finalize_and_write_depth_disagreement_stats(cfg, policy, tb_logger, collection_phase: int) -> None:
    if not bool(cfg.policy.get('depth_normalized_backup_disagreement', False)):
        return
    finalize = getattr(policy, 'finalize_depth_disagreement_phase', None)
    if not callable(finalize):
        logging.warning('[DEPTH_DISAGREEMENT_STATS] policy does not expose phase finalization')
        return
    rows = finalize(collection_phase)
    if not rows:
        return

    csv_path = os.path.join(_diagnostics_dir(cfg), 'depth_disagreement_stats.csv')
    fieldnames = list(rows[0].keys())
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    log_depth_limit = max(1, int(cfg.policy.get('depth_matched_qstd_max_depth', 5)))
    logged_rows = [row for row in rows if int(row['depth']) <= log_depth_limit]
    logging.info(
        '[DEPTH_DISAGREEMENT_STATS] phase %s | phase_calibrated %s | phase_ratio %s | '
        'ema_ratio %s | scale_next %s | groups %s',
        collection_phase,
        {int(row['depth']): round(float(row['phase_calibrated_trimmed_mean']), 4) for row in logged_rows},
        {
            int(row['depth']): round(float(row['phase_calibrated_ratio_to_depth_1']), 4)
            for row in logged_rows
        },
        {
            int(row['depth']): round(float(row['ema_calibrated_ratio_to_depth_1']), 4)
            for row in logged_rows
        },
        {int(row['depth']): round(float(row['depth_scale_next_phase']), 4) for row in logged_rows},
        {int(row['depth']): int(row['phase_group_count']) for row in logged_rows},
    )
    if tb_logger is not None:
        for row in rows:
            depth = int(row['depth'])
            prefix = f'depth_disagreement/depth_{depth}'
            tb_logger.add_scalar(
                f'{prefix}/phase_calibrated_trimmed_mean',
                float(row['phase_calibrated_trimmed_mean']),
                collection_phase,
            )
            tb_logger.add_scalar(
                f'{prefix}/ema_calibrated_baseline',
                float(row['ema_calibrated_baseline']),
                collection_phase,
            )
            tb_logger.add_scalar(
                f'{prefix}/phase_calibrated_ratio_to_depth_1',
                float(row['phase_calibrated_ratio_to_depth_1']),
                collection_phase,
            )
            tb_logger.add_scalar(
                f'{prefix}/ema_calibrated_ratio_to_depth_1',
                float(row['ema_calibrated_ratio_to_depth_1']),
                collection_phase,
            )
            tb_logger.add_scalar(
                f'{prefix}/effective_calibrated_ratio_to_depth_1',
                float(row['effective_calibrated_ratio_to_depth_1']),
                collection_phase,
            )
            tb_logger.add_scalar(
                f'{prefix}/effective_calibrated_baseline',
                float(row['effective_calibrated_baseline']),
                collection_phase,
            )
            tb_logger.add_scalar(
                f'{prefix}/scale_next_phase',
                float(row['depth_scale_next_phase']),
                collection_phase,
            )
            tb_logger.add_scalar(
                f'{prefix}/phase_group_count',
                float(row['phase_group_count']),
                collection_phase,
            )


def _write_collection_root_action_diagnostics(cfg, policy, new_data: List[Any], tb_logger, collection_phase: int) -> None:
    if not _diagnostics_enabled(cfg) or not bool(cfg.policy.get('collection_root_action_diagnostics', False)):
        return

    rows = []
    depth_matched_qstd_max_depth = max(
        1, int(cfg.policy.get('depth_matched_qstd_max_depth', 5))
    )
    rnd_landscape_query_max_horizon = max(
        1, int(cfg.policy.get('rnd_landscape_query_max_horizon', 8))
    )
    transition_ssl_diagnostics = _compute_collection_selected_transition_ssl_diagnostics(cfg, policy, new_data)
    data, meta_list = new_data
    for segment_index, (game_segment, meta) in enumerate(zip(data, meta_list)):
        diagnostics_segment = getattr(game_segment, 'root_action_diagnostics_segment', [])
        if not diagnostics_segment:
            continue
        real_step_count = min(
            int(meta.get('segment_length', len(game_segment))),
            len(diagnostics_segment),
            len(game_segment.action_segment),
        )
        for timestep in range(real_step_count):
            selected_action = _scalar_action(game_segment.action_segment[timestep])
            intrinsic_rewards = getattr(game_segment, 'intrinsic_reward_segment', [])
            intrinsic_reward_raws = getattr(game_segment, 'intrinsic_reward_raw_segment', [])
            episodic_novelty_target = (
                float(intrinsic_rewards[timestep]) if timestep < len(intrinsic_rewards) else float('nan')
            )
            episodic_novelty_raw_target = (
                float(intrinsic_reward_raws[timestep])
                if timestep < len(intrinsic_reward_raws) else float('nan')
            )
            future_metrics = _future_trajectory_metrics(cfg, game_segment, timestep, real_step_count)
            obs_velocity = float(future_metrics.get('obs_velocity', float('nan')))
            timestep_value = None
            if timestep < len(game_segment.timestep_segment):
                timestep_value = _scalar_action(game_segment.timestep_segment[timestep])
            for action_terms in diagnostics_segment[timestep]:
                action = int(action_terms.get('action', -1))
                planned_action = int(action_terms.get('collect_planned_action', selected_action))
                actual_action = int(action_terms.get('collect_actual_action', selected_action))
                ssl_diagnostics = _nan_ssl_diagnostics()
                if action == selected_action:
                    ssl_diagnostics.update(transition_ssl_diagnostics.get((int(segment_index), int(timestep)), {}))
                rows.append(
                    {
                        'collection_phase': meta.get('collection_phase'),
                        'train_iter_at_collect': meta.get('train_iter_at_collect'),
                        'envstep_before_collect': meta.get('envstep_before_collect'),
                        'envstep_after_collect': meta.get('envstep_after_collect'),
                        'segment_index_in_collect': meta.get('segment_index_in_collect', segment_index),
                        'env_id': meta.get('env_id'),
                        'env_episode_index': meta.get('env_episode_index'),
                        'collection_episode_index': meta.get('collection_episode_index'),
                        'episode_segment_index': meta.get('episode_segment_index'),
                        'episode_return': meta.get('episode_return'),
                        'video_path': meta.get('video_path'),
                        'timestep': timestep_value if timestep_value is not None else timestep,
                        'segment_timestep': timestep,
                        'selected_action': selected_action,
                        'collect_planned_action': planned_action,
                        'collect_actual_action': actual_action,
                        'collect_random_action_epsilon': float(
                            action_terms.get('collect_random_action_epsilon', 0.0)
                        ),
                        'collect_random_action_overridden': float(
                            action_terms.get('collect_random_action_overridden', 0.0)
                        ),
                        'action': action,
                        'selected': int(action == selected_action),
                        'planned': int(action == planned_action),
                        'action_direction': _action_direction(action),
                        'selected_action_direction': _action_direction(selected_action),
                        'action_matches_velocity': _action_matches_velocity(action, obs_velocity),
                        'selected_action_matches_velocity': _action_matches_velocity(selected_action, obs_velocity),
                        'collection_policy_mode': action_terms.get('collection_policy_mode', 'normal'),
                        'pure_exploration_episode': float(
                            action_terms.get('pure_exploration_episode', 0.0)
                        ),
                        'pure_exploration_action_source': action_terms.get(
                            'pure_exploration_action_source', ''
                        ),
                        'pure_exploration_planned_action': int(
                            action_terms.get('pure_exploration_planned_action', -1)
                        ),
                        'pure_exploration_selected_action': float(
                            action_terms.get('pure_exploration_selected_action', 0.0)
                        ),
                        'pure_exploration_commit_steps': int(
                            action_terms.get('pure_exploration_commit_steps', 0)
                        ),
                        'pure_exploration_commit_remaining': int(
                            action_terms.get('pure_exploration_commit_remaining', 0)
                        ),
                        'pure_exploration_selected_sequence': json.dumps(
                            _jsonable(action_terms.get('pure_exploration_selected_sequence', []))
                        ),
                        'pure_exploration_selected_sequence_qstd': float(
                            action_terms.get('pure_exploration_selected_sequence_qstd', 0.0)
                        ),
                        'pure_exploration_selected_sequence_raw_qstd': float(
                            action_terms.get('pure_exploration_selected_sequence_raw_qstd', 0.0)
                        ),
                        'pure_exploration_selected_sequence_calibrated_qstd': float(
                            action_terms.get('pure_exploration_selected_sequence_calibrated_qstd', 0.0)
                        ),
                        'pure_exploration_selected_sequence_member_centered_qstd': float(
                            action_terms.get('pure_exploration_selected_sequence_member_centered_qstd', 0.0)
                        ),
                        'pure_exploration_selected_sequence_score_qstd': float(
                            action_terms.get('pure_exploration_selected_sequence_score_qstd', 0.0)
                        ),
                        'pure_exploration_selected_sequence_mean_q': float(
                            action_terms.get('pure_exploration_selected_sequence_mean_q', 0.0)
                        ),
                        'pure_exploration_selected_sequence_score': float(
                            action_terms.get('pure_exploration_selected_sequence_score', 0.0)
                        ),
                        'pure_exploration_selected_sequence_unbounded_score': float(
                            action_terms.get('pure_exploration_selected_sequence_unbounded_score', 0.0)
                        ),
                        'pure_exploration_selected_sequence_optimistic_member_q': float(
                            action_terms.get('pure_exploration_selected_sequence_optimistic_member_q', 0.0)
                        ),
                        'pure_exploration_selected_sequence_optimism_cap_applied': float(
                            action_terms.get('pure_exploration_selected_sequence_optimism_cap_applied', 0.0)
                        ),
                        'pure_exploration_selected_sequence_optimism_cap_reduction': float(
                            action_terms.get('pure_exploration_selected_sequence_optimism_cap_reduction', 0.0)
                        ),
                        'pure_exploration_selected_sequence_q_values': json.dumps(
                            _jsonable(action_terms.get('pure_exploration_selected_sequence_q_values', []))
                        ),
                        'pure_exploration_selected_sequence_calibrated_q_values': json.dumps(
                            _jsonable(
                                action_terms.get('pure_exploration_selected_sequence_calibrated_q_values', [])
                            )
                        ),
                        'pure_exploration_selected_sequence_reward_return_values': json.dumps(
                            _jsonable(
                                action_terms.get('pure_exploration_selected_sequence_reward_return_values', [])
                            )
                        ),
                        'pure_exploration_selected_sequence_leaf_value_contribution_values': json.dumps(
                            _jsonable(
                                action_terms.get(
                                    'pure_exploration_selected_sequence_leaf_value_contribution_values', []
                                )
                            )
                        ),
                        'pure_exploration_selected_sequence_calibrated_leaf_value_contribution_values': json.dumps(
                            _jsonable(
                                action_terms.get(
                                    'pure_exploration_selected_sequence_calibrated_leaf_value_contribution_values', []
                                )
                            )
                        ),
                        'pure_exploration_selected_sequence_leaf_value_values': json.dumps(
                            _jsonable(action_terms.get('pure_exploration_selected_sequence_leaf_value_values', []))
                        ),
                        'pure_exploration_selected_sequence_member_q_baselines': json.dumps(
                            _jsonable(
                                action_terms.get('pure_exploration_selected_sequence_member_q_baselines', [])
                            )
                        ),
                        'pure_exploration_sequence_temperature': float(
                            action_terms.get('pure_exploration_sequence_temperature', 0.0)
                        ),
                        'pure_exploration_sequence_candidate_count': int(
                            action_terms.get('pure_exploration_sequence_candidate_count', 0)
                        ),
                        'pure_exploration_sequence_selection_probability': float(
                            action_terms.get('pure_exploration_sequence_selection_probability', 0.0)
                        ),
                        'pure_exploration_action_probability': float(
                            action_terms.get('pure_exploration_action_probability', 0.0)
                        ),
                        'pure_exploration_policy_count': float(
                            action_terms.get('pure_exploration_policy_count', 0.0)
                        ),
                        'pure_exploration_store_exploration_policy': float(
                            action_terms.get('pure_exploration_store_exploration_policy', 0.0)
                        ),
                        'intrinsic_reward_source': action_terms.get('intrinsic_reward_source', 'rnd'),
                        'episodic_novelty_real_target': (
                            episodic_novelty_target if action == selected_action else float('nan')
                        ),
                        'episodic_novelty_real_raw_target': (
                            episodic_novelty_raw_target if action == selected_action else float('nan')
                        ),
                        'episodic_novelty_real_pseudo_count': (
                            episodic_novelty_raw_target if action == selected_action else float('nan')
                        ),
                        'episodic_novelty_shadow_reward_mean': float(
                            action_terms.get('episodic_novelty_shadow_reward_mean', float('nan'))
                        ),
                        'episodic_novelty_shadow_reward_std': float(
                            action_terms.get('episodic_novelty_shadow_reward_std', float('nan'))
                        ),
                        'episodic_novelty_shadow_reward_min': float(
                            action_terms.get('episodic_novelty_shadow_reward_min', float('nan'))
                        ),
                        'episodic_novelty_shadow_reward_max': float(
                            action_terms.get('episodic_novelty_shadow_reward_max', float('nan'))
                        ),
                        'episodic_novelty_shadow_reward_members': json.dumps(
                            _jsonable(action_terms.get('episodic_novelty_shadow_reward_members', []))
                        ),
                        'episodic_novelty_selected_prediction_error': (
                            float(action_terms.get('episodic_novelty_shadow_reward_mean', float('nan')))
                            - episodic_novelty_target
                            if action == selected_action else float('nan')
                        ),
                        'episodic_novelty_history_shadow_reward_mean': float(
                            action_terms.get(
                                'episodic_novelty_history_shadow_reward_mean', float('nan')
                            )
                        ),
                        'episodic_novelty_history_shadow_reward_std': float(
                            action_terms.get(
                                'episodic_novelty_history_shadow_reward_std', float('nan')
                            )
                        ),
                        'episodic_novelty_history_shadow_reward_min': float(
                            action_terms.get(
                                'episodic_novelty_history_shadow_reward_min', float('nan')
                            )
                        ),
                        'episodic_novelty_history_shadow_reward_max': float(
                            action_terms.get(
                                'episodic_novelty_history_shadow_reward_max', float('nan')
                            )
                        ),
                        'episodic_novelty_history_shadow_reward_members': json.dumps(
                            _jsonable(
                                action_terms.get(
                                    'episodic_novelty_history_shadow_reward_members', []
                                )
                            )
                        ),
                        'episodic_novelty_history_state_norm_mean': float(
                            action_terms.get(
                                'episodic_novelty_history_state_norm_mean', float('nan')
                            )
                        ),
                        'episodic_novelty_history_prediction_uses_one_step_only': float(
                            action_terms.get(
                                'episodic_novelty_history_prediction_uses_one_step_only', 0.0
                            )
                        ),
                        'episodic_novelty_history_next_state_norm_mean': float(
                            action_terms.get(
                                'episodic_novelty_history_next_state_norm_mean', float('nan')
                            )
                        ),
                        'episodic_novelty_history_selected_prediction_error': (
                            float(
                                action_terms.get(
                                    'episodic_novelty_history_shadow_reward_mean', float('nan')
                                )
                            ) - episodic_novelty_target
                            if action == selected_action else float('nan')
                        ),
                        **future_metrics,
                        'visit_count': int(action_terms.get('visit_count', 0)),
                        'search_mode': action_terms.get('search_mode', ''),
                        'value_calibration_enabled': float(
                            action_terms.get('value_calibration_enabled', 0.0)
                        ),
                        'value_calibration_phase': int(action_terms.get('value_calibration_phase', -1)),
                        'value_calibration_sample_count': int(
                            action_terms.get('value_calibration_sample_count', 0)
                        ),
                        'value_calibration_offsets': json.dumps(
                            _jsonable(action_terms.get('value_calibration_offsets', []))
                        ),
                        'prior': float(action_terms.get('prior', 0.0)),
                        'mean_q': float(action_terms.get('mean_q', 0.0)),
                        'mixed_mean_q': float(action_terms.get('mixed_mean_q', 0.0)),
                        'score_mean_q': float(
                            action_terms.get('score_mean_q', action_terms.get('mean_q', 0.0))
                        ),
                        'normalized_mean_q': float(action_terms.get('normalized_mean_q', 0.0)),
                        'unbounded_ensemble_q': float(action_terms.get('unbounded_ensemble_q', 0.0)),
                        'bounded_ensemble_q': float(action_terms.get('bounded_ensemble_q', 0.0)),
                        'normalized_ensemble_q': float(action_terms.get('normalized_ensemble_q', 0.0)),
                        'ensemble_envelope_q': float(action_terms.get('ensemble_envelope_q', 0.0)),
                        'ensemble_envelope_gap_from_mean': float(
                            action_terms.get('ensemble_envelope_gap_from_mean', 0.0)
                        ),
                        'ensemble_envelope_member_id': int(
                            action_terms.get('ensemble_envelope_member_id', -1)
                        ),
                        'ensemble_envelope_cap_applied': float(
                            action_terms.get('ensemble_envelope_cap_applied', 0.0)
                        ),
                        'ensemble_envelope_cap_adjustment': float(
                            action_terms.get('ensemble_envelope_cap_adjustment', 0.0)
                        ),
                        'bounded_ensemble_mode': float(action_terms.get('bounded_ensemble_mode', 0.0)),
                        'bounded_score_best_action': int(action_terms.get('bounded_score_best_action', -1)),
                        'mean_score_best_action': int(action_terms.get('mean_score_best_action', -1)),
                        'bounded_changed_mean_score_argmax': float(
                            action_terms.get('bounded_changed_mean_score_argmax', 0.0)
                        ),
                        'same_envelope_member_all_actions': float(
                            action_terms.get('same_envelope_member_all_actions', 0.0)
                        ),
                        'q_std': float(action_terms.get('q_std', 0.0)),
                        'raw_q_std': float(action_terms.get('raw_q_std', action_terms.get('q_std', 0.0))),
                        'calibrated_q_std': float(
                            action_terms.get('calibrated_q_std', action_terms.get('q_std', 0.0))
                        ),
                        'normalized_q_std': float(action_terms.get('normalized_q_std', 0.0)),
                        'uncertainty_metric': str(action_terms.get('uncertainty_metric', '')),
                        'score_q_std': float(action_terms.get('score_q_std', action_terms.get('q_std', 0.0))),
                        'aggregate_raw_q_std': float(
                            action_terms.get('aggregate_raw_q_std', action_terms.get('raw_q_std', 0.0))
                        ),
                        'aggregate_calibrated_q_std': float(
                            action_terms.get(
                                'aggregate_calibrated_q_std', action_terms.get('calibrated_q_std', 0.0)
                            )
                        ),
                        'aggregate_normalized_q_std': float(
                            action_terms.get('aggregate_normalized_q_std', 0.0)
                        ),
                        'raw_backup_disagreement_mean': float(
                            action_terms.get('raw_backup_disagreement_mean', 0.0)
                        ),
                        'raw_backup_disagreement_rms': float(
                            action_terms.get('raw_backup_disagreement_rms', 0.0)
                        ),
                        'calibrated_backup_disagreement_mean': float(
                            action_terms.get('calibrated_backup_disagreement_mean', 0.0)
                        ),
                        'calibrated_backup_disagreement_rms': float(
                            action_terms.get('calibrated_backup_disagreement_rms', 0.0)
                        ),
                        'depth_normalized_backup_disagreement_mean': float(
                            action_terms.get('depth_normalized_backup_disagreement_mean', 0.0)
                        ),
                        'depth_normalized_backup_disagreement_rms': float(
                            action_terms.get('depth_normalized_backup_disagreement_rms', 0.0)
                        ),
                        'depth_corrected_aggregate_q_std_uncapped': float(
                            action_terms.get('depth_corrected_aggregate_q_std_uncapped', 0.0)
                        ),
                        'depth_corrected_aggregate_q_std': float(
                            action_terms.get('depth_corrected_aggregate_q_std', 0.0)
                        ),
                        'depth_corrected_q_residual_values': json.dumps(
                            _jsonable(action_terms.get('depth_corrected_q_residual_values', []))
                        ),
                        'depth_correction_cap_applied': float(
                            action_terms.get('depth_correction_cap_applied', 0.0)
                        ),
                        'depth_correction_ratio': float(
                            action_terms.get('depth_correction_ratio', 0.0)
                        ),
                        'backup_disagreement_cancellation': float(
                            action_terms.get('backup_disagreement_cancellation', 0.0)
                        ),
                        'backup_disagreement_cancellation_ratio': float(
                            action_terms.get('backup_disagreement_cancellation_ratio', 0.0)
                        ),
                        'applied_depth_disagreement_scale_mean': float(
                            action_terms.get('applied_depth_disagreement_scale_mean', 1.0)
                        ),
                        'aggregate_qstd_counterfactual_uncertainty_score': float(
                            action_terms.get('aggregate_qstd_counterfactual_uncertainty_score', 0.0)
                        ),
                        'aggregate_qstd_counterfactual_total_score': float(
                            action_terms.get('aggregate_qstd_counterfactual_total_score', 0.0)
                        ),
                        'aggregate_qstd_counterfactual_score_best_action': int(
                            action_terms.get('aggregate_qstd_counterfactual_score_best_action', -1)
                        ),
                        'aggregate_qstd_counterfactual_changed_argmax': float(
                            action_terms.get('aggregate_qstd_counterfactual_changed_argmax', 0.0)
                        ),
                        'depth_matched_qstd_diagnostics_enabled': float(
                            action_terms.get('depth_matched_qstd_diagnostics_enabled', 0.0)
                        ),
                        'depth_normalized_backup_disagreement_enabled': float(
                            action_terms.get('depth_normalized_backup_disagreement_enabled', 0.0)
                        ),
                        'depth_matched_qstd_max_depth': int(
                            action_terms.get('depth_matched_qstd_max_depth', depth_matched_qstd_max_depth)
                        ),
                        'depth_matched_common_depth_count': int(
                            action_terms.get('depth_matched_common_depth_count', 0)
                        ),
                        'depth_matched_common_depths': json.dumps(
                            _jsonable(action_terms.get('depth_matched_common_depths', []))
                        ),
                        'depth_matched_available': float(
                            action_terms.get('depth_matched_available', 0.0)
                        ),
                        'depth_matched_mean_q': float(action_terms.get('depth_matched_mean_q', 0.0)),
                        'depth_matched_raw_q_std': float(
                            action_terms.get('depth_matched_raw_q_std', 0.0)
                        ),
                        'depth_matched_calibrated_q_std': float(
                            action_terms.get('depth_matched_calibrated_q_std', 0.0)
                        ),
                        'depth_matched_normalized_q_std': float(
                            action_terms.get('depth_matched_normalized_q_std', 0.0)
                        ),
                        'depth_matched_q_values': json.dumps(
                            _jsonable(action_terms.get('depth_matched_q_values', []))
                        ),
                        'depth_matched_calibrated_q_values': json.dumps(
                            _jsonable(action_terms.get('depth_matched_calibrated_q_values', []))
                        ),
                        'depth_matched_counterfactual_uncertainty_score': float(
                            action_terms.get('depth_matched_counterfactual_uncertainty_score', 0.0)
                        ),
                        'depth_matched_counterfactual_total_score': float(
                            action_terms.get('depth_matched_counterfactual_total_score', 0.0)
                        ),
                        'depth_matched_current_score_best_action': int(
                            action_terms.get('depth_matched_current_score_best_action', -1)
                        ),
                        'depth_matched_counterfactual_score_best_action': int(
                            action_terms.get('depth_matched_counterfactual_score_best_action', -1)
                        ),
                        'depth_matched_counterfactual_changed_argmax': float(
                            action_terms.get('depth_matched_counterfactual_changed_argmax', 0.0)
                        ),
                        **{
                            f'depth_{depth}_return_count': int(
                                action_terms.get(f'depth_{depth}_return_count', 0)
                            )
                            for depth in range(1, depth_matched_qstd_max_depth + 1)
                        },
                        **{
                            f'depth_{depth}_mean_q': float(
                                action_terms.get(f'depth_{depth}_mean_q', 0.0)
                            )
                            for depth in range(1, depth_matched_qstd_max_depth + 1)
                        },
                        **{
                            f'depth_{depth}_raw_q_std': float(
                                action_terms.get(f'depth_{depth}_raw_q_std', 0.0)
                            )
                            for depth in range(1, depth_matched_qstd_max_depth + 1)
                        },
                        **{
                            f'depth_{depth}_calibrated_q_std': float(
                                action_terms.get(f'depth_{depth}_calibrated_q_std', 0.0)
                            )
                            for depth in range(1, depth_matched_qstd_max_depth + 1)
                        },
                        **{
                            f'depth_{depth}_q_values': json.dumps(
                                _jsonable(action_terms.get(f'depth_{depth}_q_values', []))
                            )
                            for depth in range(1, depth_matched_qstd_max_depth + 1)
                        },
                        **{
                            f'depth_{depth}_calibrated_q_values': json.dumps(
                                _jsonable(action_terms.get(f'depth_{depth}_calibrated_q_values', []))
                            )
                            for depth in range(1, depth_matched_qstd_max_depth + 1)
                        },
                        **{
                            f'depth_{depth}_{metric}': float(
                                action_terms.get(f'depth_{depth}_{metric}', default)
                            )
                            for depth in range(1, depth_matched_qstd_max_depth + 1)
                            for metric, default in [
                                ('raw_backup_disagreement_mean', 0.0),
                                ('raw_backup_disagreement_rms', 0.0),
                                ('calibrated_backup_disagreement_mean', 0.0),
                                ('calibrated_backup_disagreement_rms', 0.0),
                                ('adjusted_backup_disagreement_mean', 0.0),
                                ('adjusted_backup_disagreement_rms', 0.0),
                                ('applied_disagreement_scale', 1.0),
                                ('running_calibrated_disagreement_baseline', 0.0),
                                ('effective_disagreement_baseline', 0.0),
                                ('running_disagreement_ratio', 0.0),
                                ('effective_disagreement_ratio', 0.0),
                                ('next_phase_disagreement_scale', 1.0),
                            ]
                        },
                        **{
                            f'depth_{depth}_baseline_group_count': int(
                                action_terms.get(f'depth_{depth}_baseline_group_count', 0)
                            )
                            for depth in range(1, depth_matched_qstd_max_depth + 1)
                        },
                        'q_min': float(action_terms.get('q_min', 0.0)),
                        'q_max': float(action_terms.get('q_max', 0.0)),
                        'q_range': float(action_terms.get('q_range', 0.0)),
                        'q_normalization_source': str(
                            action_terms.get('q_normalization_source', 'extrinsic')
                        ),
                        'predicted_reward_return_mean': float(
                            action_terms.get('predicted_reward_return_mean', 0.0)
                        ),
                        'predicted_reward_return_std': float(
                            action_terms.get('predicted_reward_return_std', 0.0)
                        ),
                        'leaf_value_contribution_mean': float(
                            action_terms.get('leaf_value_contribution_mean', 0.0)
                        ),
                        'leaf_value_contribution_std': float(
                            action_terms.get('leaf_value_contribution_std', 0.0)
                        ),
                        'prior_score': float(action_terms.get('prior_score', 0.0)),
                        'uncertainty_score': float(action_terms.get('uncertainty_score', 0.0)),
                        'unbounded_uncertainty_score': float(
                            action_terms.get('unbounded_uncertainty_score', 0.0)
                        ),
                        'intrinsic_score': float(action_terms.get('intrinsic_score', 0.0)),
                        'intrinsic_value': float(action_terms.get('intrinsic_value', 0.0)),
                        'intrinsic_immediate_reward': float(
                            action_terms.get('intrinsic_immediate_reward', action_terms.get('intrinsic_reward', 0.0))
                        ),
                        'rnd_landscape_query_enabled': float(
                            action_terms.get('rnd_landscape_query_enabled', 0.0)
                        ),
                        'rnd_landscape_query_snapshot_count': int(
                            action_terms.get('rnd_landscape_query_snapshot_count', 0)
                        ),
                        'rnd_landscape_query_count': int(
                            action_terms.get('rnd_landscape_query_count', 0)
                        ),
                        'rnd_landscape_query_error_mean': float(
                            action_terms.get('rnd_landscape_query_error_mean', 0.0)
                        ),
                        'rnd_landscape_query_error_min': float(
                            action_terms.get('rnd_landscape_query_error_min', 0.0)
                        ),
                        'rnd_landscape_query_error_max': float(
                            action_terms.get('rnd_landscape_query_error_max', 0.0)
                        ),
                        'rnd_landscape_query_states': json.dumps(
                            _jsonable(action_terms.get('rnd_landscape_query_states', []))
                        ),
                        'rnd_landscape_query_errors': json.dumps(
                            _jsonable(action_terms.get('rnd_landscape_query_errors', []))
                        ),
                        'rnd_landscape_query_predicted_one_step_reward': float(
                            action_terms.get('rnd_landscape_query_predicted_one_step_reward', 0.0)
                        ),
                        'rnd_landscape_query_tree_edge_count': int(
                            action_terms.get('rnd_landscape_query_tree_edge_count', 0)
                        ),
                        'rnd_landscape_query_tree_predicted_reward_mean': float(
                            action_terms.get('rnd_landscape_query_tree_predicted_reward_mean', 0.0)
                        ),
                        'rnd_episodic_memory_enabled': float(
                            action_terms.get('rnd_episodic_memory_enabled', 0.0)
                        ),
                        'rnd_episodic_memory_key_space': str(
                            action_terms.get('rnd_episodic_memory_key_space', 'none')
                        ),
                        'rnd_episodic_memory_tree_base_reward_mean': float(
                            action_terms.get(
                                'rnd_episodic_memory_tree_base_reward_mean', 0.0
                            )
                        ),
                        'rnd_episodic_memory_tree_reward_factor_mean': float(
                            action_terms.get(
                                'rnd_episodic_memory_tree_reward_factor_mean', 1.0
                            )
                        ),
                        'rnd_episodic_memory_tree_suppressed_fraction': float(
                            action_terms.get(
                                'rnd_episodic_memory_tree_suppressed_fraction', 0.0
                            )
                        ),
                        'rnd_episodic_memory_tree_real_match_fraction': float(
                            action_terms.get(
                                'rnd_episodic_memory_tree_real_match_fraction', 0.0
                            )
                        ),
                        'rnd_episodic_memory_tree_path_match_fraction': float(
                            action_terms.get(
                                'rnd_episodic_memory_tree_path_match_fraction', 0.0
                            )
                        ),
                        'rnd_episodic_memory_one_step_base_reward': float(
                            action_terms.get(
                                'rnd_episodic_memory_one_step_base_reward', 0.0
                            )
                        ),
                        'rnd_episodic_memory_one_step_reward_factor': float(
                            action_terms.get(
                                'rnd_episodic_memory_one_step_reward_factor', 1.0
                            )
                        ),
                        'rnd_episodic_memory_one_step_real_distance': float(
                            action_terms.get(
                                'rnd_episodic_memory_one_step_real_distance', float('nan')
                            )
                        ),
                        'rnd_episodic_memory_one_step_path_distance': float(
                            action_terms.get(
                                'rnd_episodic_memory_one_step_path_distance', float('nan')
                            )
                        ),
                        'rnd_episodic_memory_one_step_distance_scale': float(
                            action_terms.get(
                                'rnd_episodic_memory_one_step_distance_scale', float('nan')
                            )
                        ),
                        'rnd_episodic_memory_one_step_normalized_distance': float(
                            action_terms.get(
                                'rnd_episodic_memory_one_step_normalized_distance', float('nan')
                            )
                        ),
                        'rnd_episodic_memory_one_step_real_memory_size': int(
                            action_terms.get(
                                'rnd_episodic_memory_one_step_real_memory_size', 0
                            )
                        ),
                        **{
                            f'rnd_landscape_query_tree_depth_{depth}_edge_count': int(
                                action_terms.get(
                                    f'rnd_landscape_query_tree_depth_{depth}_edge_count', 0
                                )
                            )
                            for depth in range(1, rnd_landscape_query_max_horizon + 1)
                        },
                        **{
                            f'rnd_episodic_memory_depth_{depth}_{metric}': float(
                                action_terms.get(
                                    f'rnd_episodic_memory_depth_{depth}_{metric}',
                                    1.0 if metric == 'reward_factor_mean' else 0.0,
                                )
                            )
                            for depth in range(1, rnd_landscape_query_max_horizon + 1)
                            for metric in (
                                'reward_factor_mean',
                                'suppressed_fraction',
                                'path_match_fraction',
                            )
                        },
                        'intrinsic_reward_return_mean': float(
                            action_terms.get('intrinsic_reward_return_mean', 0.0)
                        ),
                        'intrinsic_reward_weighted_return_mean': float(
                            action_terms.get('intrinsic_reward_weighted_return_mean', 0.0)
                        ),
                        'intrinsic_leaf_value_raw_contribution_mean': float(
                            action_terms.get('intrinsic_leaf_value_raw_contribution_mean', 0.0)
                        ),
                        'intrinsic_leaf_value_clipped_contribution_mean': float(
                            action_terms.get('intrinsic_leaf_value_clipped_contribution_mean', 0.0)
                        ),
                        'intrinsic_leaf_value_contribution_mean': float(
                            action_terms.get('intrinsic_leaf_value_contribution_mean', 0.0)
                        ),
                        'separate_intrinsic_tree_enabled': float(
                            action_terms.get('separate_intrinsic_tree_enabled', 0.0)
                        ),
                        'main_tree_visit_count': int(
                            action_terms.get('main_tree_visit_count', action_terms.get('visit_count', 0))
                        ),
                        'main_tree_total_score': float(
                            action_terms.get('main_tree_total_score', action_terms.get('total_score', 0.0))
                        ),
                        'main_tree_puct_total_score': float(
                            action_terms.get(
                                'main_tree_puct_total_score',
                                action_terms.get('main_tree_total_score', action_terms.get('total_score', 0.0))
                            )
                        ),
                        'main_tree_intrinsic_value': float(
                            action_terms.get('main_tree_intrinsic_value', action_terms.get('intrinsic_value', 0.0))
                        ),
                        'main_tree_intrinsic_score': float(
                            action_terms.get('main_tree_intrinsic_score', action_terms.get('intrinsic_score', 0.0))
                        ),
                        'intrinsic_tree_visit_count': int(action_terms.get('intrinsic_tree_visit_count', 0)),
                        'intrinsic_tree_prior': float(action_terms.get('intrinsic_tree_prior', 0.0)),
                        'intrinsic_tree_prior_mode': action_terms.get(
                            'intrinsic_tree_prior_mode',
                            action_terms.get('policy_prior_mode', 'learned'),
                        ),
                        'intrinsic_tree_force_root_action_coverage': float(
                            action_terms.get(
                                'intrinsic_tree_force_root_action_coverage',
                                action_terms.get('force_root_action_coverage', 0.0),
                            )
                        ),
                        'intrinsic_tree_root_forced_coverage_complete': float(
                            action_terms.get(
                                'intrinsic_tree_root_forced_coverage_complete',
                                action_terms.get('root_forced_coverage_complete', 0.0),
                            )
                        ),
                        'intrinsic_tree_action_policy_source': action_terms.get(
                            'intrinsic_tree_action_policy_source', ''
                        ),
                        'intrinsic_tree_policy_target_source': action_terms.get(
                            'intrinsic_tree_policy_target_source', ''
                        ),
                        'intrinsic_tree_prior_score': float(action_terms.get('intrinsic_tree_prior_score', 0.0)),
                        'intrinsic_tree_total_score': float(action_terms.get('intrinsic_tree_total_score', 0.0)),
                        'intrinsic_tree_intrinsic_value': float(
                            action_terms.get('intrinsic_tree_intrinsic_value', 0.0)
                        ),
                        'intrinsic_tree_normalized_intrinsic_value': float(
                            action_terms.get(
                                'intrinsic_tree_normalized_intrinsic_value',
                                action_terms.get('intrinsic_tree_intrinsic_value', 0.0)
                            )
                        ),
                        'intrinsic_tree_intrinsic_score': float(
                            action_terms.get('intrinsic_tree_intrinsic_score', 0.0)
                        ),
                        'intrinsic_tree_intrinsic_immediate_reward': float(
                            action_terms.get(
                                'intrinsic_tree_intrinsic_immediate_reward',
                                action_terms.get('intrinsic_immediate_reward', 0.0)
                            )
                        ),
                        'intrinsic_tree_latent_dd_immediate': float(
                            action_terms.get('intrinsic_tree_latent_dd_immediate', 0.0)
                        ),
                        'intrinsic_tree_projection_dd_immediate': float(
                            action_terms.get('intrinsic_tree_projection_dd_immediate', 0.0)
                        ),
                        'intrinsic_tree_prediction_projection_dd_immediate': float(
                            action_terms.get('intrinsic_tree_prediction_projection_dd_immediate', 0.0)
                        ),
                        'real_transition_ssl_valid': float(ssl_diagnostics['real_transition_ssl_valid']),
                        'real_transition_ssl_projection_cos_mean': float(
                            ssl_diagnostics['real_transition_ssl_projection_cos_mean']
                        ),
                        'real_transition_ssl_projection_error_mean': float(
                            ssl_diagnostics['real_transition_ssl_projection_error_mean']
                        ),
                        'real_transition_ssl_projection_error_std': float(
                            ssl_diagnostics['real_transition_ssl_projection_error_std']
                        ),
                        'real_transition_ssl_projection_error_max': float(
                            ssl_diagnostics['real_transition_ssl_projection_error_max']
                        ),
                        'real_transition_ssl_prediction_cos_mean': float(
                            ssl_diagnostics['real_transition_ssl_prediction_cos_mean']
                        ),
                        'real_transition_ssl_prediction_error_mean': float(
                            ssl_diagnostics['real_transition_ssl_prediction_error_mean']
                        ),
                        'real_transition_ssl_prediction_error_std': float(
                            ssl_diagnostics['real_transition_ssl_prediction_error_std']
                        ),
                        'real_transition_ssl_prediction_error_max': float(
                            ssl_diagnostics['real_transition_ssl_prediction_error_max']
                        ),
                        'real_transition_projection_pairwise_dd': float(
                            ssl_diagnostics['real_transition_projection_pairwise_dd']
                        ),
                        'real_transition_prediction_projection_pairwise_dd': float(
                            ssl_diagnostics['real_transition_prediction_projection_pairwise_dd']
                        ),
                        'real_transition_target_projection_pairwise_dd': float(
                            ssl_diagnostics['real_transition_target_projection_pairwise_dd']
                        ),
                        'real_transition_target_latent_pairwise_dd': float(
                            ssl_diagnostics['real_transition_target_latent_pairwise_dd']
                        ),
                        'intrinsic_tree_intrinsic_reward_return_mean': float(
                            action_terms.get('intrinsic_tree_intrinsic_reward_return_mean', 0.0)
                        ),
                        'intrinsic_tree_intrinsic_reward_weighted_return_mean': float(
                            action_terms.get('intrinsic_tree_intrinsic_reward_weighted_return_mean', 0.0)
                        ),
                        'intrinsic_tree_intrinsic_leaf_value_raw_contribution_mean': float(
                            action_terms.get('intrinsic_tree_intrinsic_leaf_value_raw_contribution_mean', 0.0)
                        ),
                        'intrinsic_tree_intrinsic_leaf_value_clipped_contribution_mean': float(
                            action_terms.get('intrinsic_tree_intrinsic_leaf_value_clipped_contribution_mean', 0.0)
                        ),
                        'intrinsic_tree_intrinsic_leaf_value_contribution_mean': float(
                            action_terms.get('intrinsic_tree_intrinsic_leaf_value_contribution_mean', 0.0)
                        ),
                        'intrinsic_tree_intrinsic_leaf_value_before_modifier_contribution_mean': float(
                            action_terms.get(
                                'intrinsic_tree_intrinsic_leaf_value_before_modifier_contribution_mean',
                                0.0,
                            )
                        ),
                        'intrinsic_tree_intrinsic_leaf_value_modifier_mean': float(
                            action_terms.get(
                                'intrinsic_tree_intrinsic_leaf_value_modifier_mean', 1.0
                            )
                        ),
                        'intrinsic_tree_immediate_reward_weight': float(
                            action_terms.get('intrinsic_tree_immediate_reward_weight', 1.0)
                        ),
                        'intrinsic_tree_leaf_value_weight': float(
                            action_terms.get('intrinsic_tree_leaf_value_weight', 1.0)
                        ),
                        'intrinsic_tree_leaf_value_source': action_terms.get(
                            'intrinsic_tree_leaf_value_source', 'default'
                        ),
                        'intrinsic_tree_dynamics_disagreement_space': action_terms.get(
                            'intrinsic_tree_dynamics_disagreement_space',
                            action_terms.get('dynamics_disagreement_space', 'latent')
                        ),
                        'intrinsic_tree_mean_q': float(action_terms.get('intrinsic_tree_mean_q', 0.0)),
                        'intrinsic_tree_q_std': float(action_terms.get('intrinsic_tree_q_std', 0.0)),
                        'intrinsic_tree_predicted_reward_return_mean': float(
                            action_terms.get('intrinsic_tree_predicted_reward_return_mean', 0.0)
                        ),
                        'intrinsic_tree_leaf_value_contribution_mean': float(
                            action_terms.get('intrinsic_tree_leaf_value_contribution_mean', 0.0)
                        ),
                        'intrinsic_tree_final_policy_formula': action_terms.get(
                            'intrinsic_tree_final_policy_formula', ''
                        ),
                        'intrinsic_tree_final_weight': float(
                            action_terms.get('intrinsic_tree_final_weight', 0.0)
                        ),
                        'intrinsic_tree_base_temperature': float(
                            action_terms.get('intrinsic_tree_base_temperature', 0.0)
                        ),
                        'intrinsic_tree_intrinsic_temperature': float(
                            action_terms.get('intrinsic_tree_intrinsic_temperature', 0.0)
                        ),
                        'intrinsic_tree_base_logit': float(
                            action_terms.get('intrinsic_tree_base_logit', 0.0)
                        ),
                        'intrinsic_tree_centered_intrinsic_value': float(
                            action_terms.get('intrinsic_tree_centered_intrinsic_value', 0.0)
                        ),
                        'intrinsic_tree_final_intrinsic_logit': float(
                            action_terms.get('intrinsic_tree_final_intrinsic_logit', 0.0)
                        ),
                        'intrinsic_tree_final_logit': float(
                            action_terms.get('intrinsic_tree_final_logit', 0.0)
                        ),
                        'intrinsic_tree_base_probability': float(
                            action_terms.get('intrinsic_tree_base_probability', 0.0)
                        ),
                        'intrinsic_tree_final_probability': float(
                            action_terms.get('intrinsic_tree_final_probability', 0.0)
                        ),
                        'intrinsic_tree_base_best_action': int(
                            action_terms.get('intrinsic_tree_base_best_action', -1)
                        ),
                        'intrinsic_tree_intrinsic_best_action': int(
                            action_terms.get('intrinsic_tree_intrinsic_best_action', -1)
                        ),
                        'intrinsic_tree_final_best_action': int(
                            action_terms.get('intrinsic_tree_final_best_action', -1)
                        ),
                        'intrinsic_tree_changed_argmax': float(
                            action_terms.get('intrinsic_tree_changed_argmax', 0.0)
                        ),
                        'intrinsic_tree_base_entropy': float(
                            action_terms.get('intrinsic_tree_base_entropy', 0.0)
                        ),
                        'intrinsic_tree_final_entropy': float(
                            action_terms.get('intrinsic_tree_final_entropy', 0.0)
                        ),
                        'intrinsic_tree_policy_total_variation': float(
                            action_terms.get('intrinsic_tree_policy_total_variation', 0.0)
                        ),
                        **{
                            (
                                'intrinsic_tree_counterfactual_probability_beta_'
                                f"{float(beta):g}".replace('-', 'm').replace('.', 'p')
                            ): float(
                                action_terms.get(
                                    (
                                        'intrinsic_tree_counterfactual_probability_beta_'
                                        f"{float(beta):g}".replace('-', 'm').replace('.', 'p')
                                    ),
                                    0.0,
                                )
                            )
                            for beta in cfg.policy.get('intrinsic_tree_counterfactual_betas', ()) or ()
                        },
                        'final_qstd_enabled': float(action_terms.get('final_qstd_enabled', 0.0)),
                        'final_qstd_beta': float(action_terms.get('final_qstd_beta', 0.0)),
                        'final_qstd_temperature': float(action_terms.get('final_qstd_temperature', 0.0)),
                        'final_qstd_visit_count': float(
                            action_terms.get('final_qstd_visit_count', action_terms.get('visit_count', 0.0))
                        ),
                        'final_qstd_base_logit': float(action_terms.get('final_qstd_base_logit', 0.0)),
                        'final_qstd_qstd_logit': float(action_terms.get('final_qstd_qstd_logit', 0.0)),
                        'final_qstd_final_logit': float(action_terms.get('final_qstd_final_logit', 0.0)),
                        'final_qstd_base_probability': float(
                            action_terms.get('final_qstd_base_probability', 0.0)
                        ),
                        'final_qstd_probability': float(action_terms.get('final_qstd_probability', 0.0)),
                        'final_qstd_base_best_action': int(
                            action_terms.get('final_qstd_base_best_action', -1)
                        ),
                        'final_qstd_best_action': int(action_terms.get('final_qstd_best_action', -1)),
                        'final_qstd_changed_argmax': float(action_terms.get('final_qstd_changed_argmax', 0.0)),
                        'separate_exploration_tree_enabled': float(
                            action_terms.get('separate_exploration_tree_enabled', 0.0)
                        ),
                        'separate_exploration_tree_horizon': int(
                            action_terms.get('separate_exploration_tree_horizon', 0)
                        ),
                        'separate_exploration_tree_top_k': int(
                            action_terms.get('separate_exploration_tree_top_k', 0)
                        ),
                        'separate_exploration_tree_beta': float(
                            action_terms.get('separate_exploration_tree_beta', 0.0)
                        ),
                        'separate_exploration_tree_temperature': float(
                            action_terms.get('separate_exploration_tree_temperature', 0.0)
                        ),
                        'separate_exploration_tree_score_normalization': action_terms.get(
                            'separate_exploration_tree_score_normalization', ''
                        ),
                        'separate_exploration_tree_sequence_count': int(
                            action_terms.get('separate_exploration_tree_sequence_count', 0)
                        ),
                        'separate_exploration_tree_total_sequence_count': int(
                            action_terms.get('separate_exploration_tree_total_sequence_count', 0)
                        ),
                        'separate_exploration_tree_visit_count': float(
                            action_terms.get('separate_exploration_tree_visit_count', 0.0)
                        ),
                        'separate_exploration_tree_base_logit': float(
                            action_terms.get('separate_exploration_tree_base_logit', 0.0)
                        ),
                        'separate_exploration_tree_raw_score': float(
                            action_terms.get('separate_exploration_tree_raw_score', 0.0)
                        ),
                        'separate_exploration_tree_normalized_score': float(
                            action_terms.get('separate_exploration_tree_normalized_score', 0.0)
                        ),
                        'separate_exploration_tree_exploration_logit': float(
                            action_terms.get('separate_exploration_tree_exploration_logit', 0.0)
                        ),
                        'separate_exploration_tree_final_logit': float(
                            action_terms.get('separate_exploration_tree_final_logit', 0.0)
                        ),
                        'separate_exploration_tree_base_probability': float(
                            action_terms.get('separate_exploration_tree_base_probability', 0.0)
                        ),
                        'separate_exploration_tree_probability': float(
                            action_terms.get('separate_exploration_tree_probability', 0.0)
                        ),
                        'separate_exploration_tree_base_best_action': int(
                            action_terms.get('separate_exploration_tree_base_best_action', -1)
                        ),
                        'separate_exploration_tree_best_action': int(
                            action_terms.get('separate_exploration_tree_best_action', -1)
                        ),
                        'separate_exploration_tree_changed_argmax': float(
                            action_terms.get('separate_exploration_tree_changed_argmax', 0.0)
                        ),
                        'separate_exploration_tree_qstd_mean': float(
                            action_terms.get('separate_exploration_tree_qstd_mean', 0.0)
                        ),
                        'separate_exploration_tree_qstd_std': float(
                            action_terms.get('separate_exploration_tree_qstd_std', 0.0)
                        ),
                        'separate_exploration_tree_qstd_min': float(
                            action_terms.get('separate_exploration_tree_qstd_min', 0.0)
                        ),
                        'separate_exploration_tree_qstd_max': float(
                            action_terms.get('separate_exploration_tree_qstd_max', 0.0)
                        ),
                        'separate_exploration_tree_qstd_p90': float(
                            action_terms.get('separate_exploration_tree_qstd_p90', 0.0)
                        ),
                        'separate_exploration_tree_calibrated_qstd_mean': float(
                            action_terms.get('separate_exploration_tree_calibrated_qstd_mean', 0.0)
                        ),
                        'separate_exploration_tree_calibrated_qstd_max': float(
                            action_terms.get('separate_exploration_tree_calibrated_qstd_max', 0.0)
                        ),
                        'separate_exploration_tree_calibrated_qstd_p90': float(
                            action_terms.get('separate_exploration_tree_calibrated_qstd_p90', 0.0)
                        ),
                        'separate_exploration_tree_sequence_score_mode': action_terms.get(
                            'separate_exploration_tree_sequence_score_mode', ''
                        ),
                        'separate_exploration_tree_sequence_qstd_beta': float(
                            action_terms.get('separate_exploration_tree_sequence_qstd_beta', 0.0)
                        ),
                        'separate_exploration_tree_sequence_qstd_mode': action_terms.get(
                            'separate_exploration_tree_sequence_qstd_mode', ''
                        ),
                        'separate_exploration_tree_member_q_baselines': json.dumps(
                            _jsonable(action_terms.get('separate_exploration_tree_member_q_baselines', []))
                        ),
                        'separate_exploration_tree_member_centered_qstd_mean': float(
                            action_terms.get('separate_exploration_tree_member_centered_qstd_mean', 0.0)
                        ),
                        'separate_exploration_tree_member_centered_qstd_max': float(
                            action_terms.get('separate_exploration_tree_member_centered_qstd_max', 0.0)
                        ),
                        'separate_exploration_tree_sequence_score_mean': float(
                            action_terms.get('separate_exploration_tree_sequence_score_mean', 0.0)
                        ),
                        'separate_exploration_tree_sequence_score_std': float(
                            action_terms.get('separate_exploration_tree_sequence_score_std', 0.0)
                        ),
                        'separate_exploration_tree_sequence_score_min': float(
                            action_terms.get('separate_exploration_tree_sequence_score_min', 0.0)
                        ),
                        'separate_exploration_tree_sequence_score_max': float(
                            action_terms.get('separate_exploration_tree_sequence_score_max', 0.0)
                        ),
                        'separate_exploration_tree_sequence_score_p90': float(
                            action_terms.get('separate_exploration_tree_sequence_score_p90', 0.0)
                        ),
                        'separate_exploration_tree_top_sequence': json.dumps(
                            _jsonable(action_terms.get('separate_exploration_tree_top_sequence', []))
                        ),
                        'separate_exploration_tree_top_sequence_qstd': float(
                            action_terms.get('separate_exploration_tree_top_sequence_qstd', 0.0)
                        ),
                        'separate_exploration_tree_top_sequence_calibrated_qstd': float(
                            action_terms.get('separate_exploration_tree_top_sequence_calibrated_qstd', 0.0)
                        ),
                        'separate_exploration_tree_top_sequence_member_centered_qstd': float(
                            action_terms.get('separate_exploration_tree_top_sequence_member_centered_qstd', 0.0)
                        ),
                        'separate_exploration_tree_top_sequence_score_qstd': float(
                            action_terms.get('separate_exploration_tree_top_sequence_score_qstd', 0.0)
                        ),
                        'separate_exploration_tree_top_sequence_score': float(
                            action_terms.get('separate_exploration_tree_top_sequence_score', 0.0)
                        ),
                        'separate_exploration_tree_top_sequence_unbounded_score': float(
                            action_terms.get('separate_exploration_tree_top_sequence_unbounded_score', 0.0)
                        ),
                        'separate_exploration_tree_top_sequence_optimistic_member_q': float(
                            action_terms.get(
                                'separate_exploration_tree_top_sequence_optimistic_member_q', 0.0
                            )
                        ),
                        'separate_exploration_tree_top_sequence_optimism_cap_applied': float(
                            action_terms.get(
                                'separate_exploration_tree_top_sequence_optimism_cap_applied', 0.0
                            )
                        ),
                        'separate_exploration_tree_top_sequence_optimism_cap_reduction': float(
                            action_terms.get(
                                'separate_exploration_tree_top_sequence_optimism_cap_reduction', 0.0
                            )
                        ),
                        'separate_exploration_tree_top_sequence_mean_q': float(
                            action_terms.get('separate_exploration_tree_top_sequence_mean_q', 0.0)
                        ),
                        'separate_exploration_tree_top_sequence_q_values': json.dumps(
                            _jsonable(action_terms.get('separate_exploration_tree_top_sequence_q_values', []))
                        ),
                        'separate_exploration_tree_top_sequence_calibrated_q_values': json.dumps(
                            _jsonable(
                                action_terms.get('separate_exploration_tree_top_sequence_calibrated_q_values', [])
                            )
                        ),
                        'separate_exploration_tree_top_sequence_reward_return_values': json.dumps(
                            _jsonable(
                                action_terms.get(
                                    'separate_exploration_tree_top_sequence_reward_return_values', []
                                )
                            )
                        ),
                        'separate_exploration_tree_top_sequence_leaf_value_contribution_values': json.dumps(
                            _jsonable(
                                action_terms.get(
                                    'separate_exploration_tree_top_sequence_leaf_value_contribution_values', []
                                )
                            )
                        ),
                        'separate_exploration_tree_top_sequence_calibrated_leaf_value_contribution_values': json.dumps(
                            _jsonable(
                                action_terms.get(
                                    'separate_exploration_tree_top_sequence_calibrated_leaf_value_contribution_values', []
                                )
                            )
                        ),
                        'separate_exploration_tree_top_sequence_leaf_value_values': json.dumps(
                            _jsonable(action_terms.get('separate_exploration_tree_top_sequence_leaf_value_values', []))
                        ),
                        'separate_exploration_tree_top_sequences': json.dumps(
                            _jsonable(action_terms.get('separate_exploration_tree_top_sequences', []))
                        ),
                        **{
                            f'separate_exploration_tree_depth{depth}_{metric}': (
                                json.dumps(
                                    _jsonable(
                                        action_terms.get(
                                            f'separate_exploration_tree_depth{depth}_{metric}', []
                                        )
                                    )
                                )
                                if metric == 'top_sequence'
                                else float(
                                    action_terms.get(
                                        f'separate_exploration_tree_depth{depth}_{metric}', 0.0
                                    )
                                )
                            )
                            for depth in range(1, 9)
                            for metric in [
                                'qstd_mean',
                                'qstd_std',
                                'qstd_min',
                                'qstd_max',
                                'qstd_p90',
                                'topk_qstd_mean',
                                'top_sequence',
                                'top_sequence_qstd',
                                'prefix_count',
                            ]
                        },
                        'final_intrinsic_score': float(action_terms.get('final_intrinsic_score', 0.0)),
                        'final_total_score': float(action_terms.get('final_total_score', action_terms.get('total_score', 0.0))),
                        'final_action_probability': float(action_terms.get('final_action_probability', 0.0)),
                        'final_policy_count': float(action_terms.get('final_policy_count', 0.0)),
                        'latent_novelty_root_disagreement': float(
                            action_terms.get('latent_novelty_root_disagreement', 0.0)
                        ),
                        'imagined_node_latent_novelty': float(
                            action_terms.get('imagined_node_latent_novelty', 0.0)
                        ),
                        'imagined_leaf_latent_novelty_mean': float(
                            action_terms.get('imagined_leaf_latent_novelty_mean', 0.0)
                        ),
                        'imagined_leaf_latent_novelty_max': float(
                            action_terms.get('imagined_leaf_latent_novelty_max', 0.0)
                        ),
                        'imagined_leaf_latent_novelty_depth_mean': float(
                            action_terms.get('imagined_leaf_latent_novelty_depth_mean', 0.0)
                        ),
                        'imagined_leaf_latent_novelty_max_depth': int(
                            action_terms.get('imagined_leaf_latent_novelty_max_depth', 0)
                        ),
                        'imagined_leaf_latent_novelty_max_path': json.dumps(
                            _jsonable(action_terms.get('imagined_leaf_latent_novelty_max_path', []))
                        ),
                        'real_transition_member_projection_ssl_errors': json.dumps(
                            _jsonable(ssl_diagnostics['real_transition_member_projection_ssl_errors'])
                        ),
                        'real_transition_member_prediction_ssl_errors': json.dumps(
                            _jsonable(ssl_diagnostics['real_transition_member_prediction_ssl_errors'])
                        ),
                        'total_score': float(action_terms.get('total_score', 0.0)),
                        'q_values': json.dumps(_jsonable(action_terms.get('q_values', []))),
                        'mixed_q_values': json.dumps(
                            _jsonable(action_terms.get('mixed_q_values', []))
                        ),
                        'calibrated_q_values': json.dumps(
                            _jsonable(action_terms.get('calibrated_q_values', action_terms.get('q_values', [])))
                        ),
                        'predicted_reward_return_values': json.dumps(
                            _jsonable(action_terms.get('predicted_reward_return_values', []))
                        ),
                        'leaf_value_contribution_values': json.dumps(
                            _jsonable(action_terms.get('leaf_value_contribution_values', []))
                        ),
                        'calibrated_leaf_value_contribution_values': json.dumps(
                            _jsonable(
                                action_terms.get(
                                    'calibrated_leaf_value_contribution_values',
                                    action_terms.get('leaf_value_contribution_values', []),
                                )
                            )
                        ),
                    }
                )
    if not rows:
        return

    diagnostics_dir = _diagnostics_dir(cfg)
    csv_path = os.path.join(diagnostics_dir, 'root_action_terms.csv')
    _annotate_mountain_car_novelty_counts(rows, csv_path)
    fieldnames = list(rows[0].keys())
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    selected_rows = [row for row in rows if int(row['selected']) == 1]
    logging.info(
        "[ROOT_ACTION_DIAG] phase %s wrote %s action rows for %s selected root decisions to %s",
        collection_phase,
        len(rows),
        len(selected_rows),
        csv_path,
    )
    episodic_rows = [
        row for row in selected_rows
        if row.get('intrinsic_reward_source') == 'episodic_novelty'
        and np.isfinite(float(row.get('episodic_novelty_real_target', float('nan'))))
        and np.isfinite(float(row.get('episodic_novelty_shadow_reward_mean', float('nan'))))
    ]
    if episodic_rows:
        real_targets = np.asarray(
            [float(row['episodic_novelty_real_target']) for row in episodic_rows], dtype=np.float32
        )
        shadow_predictions = np.asarray(
            [float(row['episodic_novelty_shadow_reward_mean']) for row in episodic_rows], dtype=np.float32
        )
        if real_targets.size > 1 and real_targets.std() > 1e-8 and shadow_predictions.std() > 1e-8:
            target_prediction_correlation = float(np.corrcoef(real_targets, shadow_predictions)[0, 1])
        else:
            target_prediction_correlation = 0.0
        stationary_rows = [
            row for row in episodic_rows
            if _finite_float(row.get('episode_return'), 0.0) <= -199.0
            and (
                _finite_float(row.get('future_max_position_remaining'), float('inf'))
                - _finite_float(row.get('future_min_position_remaining'), float('-inf'))
            ) <= 0.05
            and _finite_float(row.get('future_max_abs_velocity_remaining'), float('inf')) <= 0.005
        ]
        action_shadow_means = {
            int(action): float(np.mean([
                float(row['episodic_novelty_shadow_reward_mean'])
                for row in rows
                if int(row['action']) == int(action)
                and np.isfinite(float(row.get('episodic_novelty_shadow_reward_mean', float('nan'))))
            ]))
            for action in sorted({int(row['action']) for row in rows})
        }
        logging.info(
            "[EPISODIC_NOVELTY_SHADOW] phase %s selected %s | target mean %.4f std %.4f, "
            "prediction mean %.4f std %.4f, MAE %.4f, corr %.4f, action predictions %s",
            collection_phase,
            len(episodic_rows),
            float(real_targets.mean()),
            float(real_targets.std()),
            float(shadow_predictions.mean()),
            float(shadow_predictions.std()),
            float(np.abs(shadow_predictions - real_targets).mean()),
            target_prediction_correlation,
            {action: round(value, 4) for action, value in action_shadow_means.items()},
        )
        if stationary_rows:
            stationary_targets = np.asarray(
                [float(row['episodic_novelty_real_target']) for row in stationary_rows], dtype=np.float32
            )
            stationary_predictions = np.asarray(
                [float(row['episodic_novelty_shadow_reward_mean']) for row in stationary_rows], dtype=np.float32
            )
            logging.info(
                "[EPISODIC_NOVELTY_STATIONARY] phase %s decisions %s | target mean %.4f max %.4f, "
                "prediction mean %.4f max %.4f",
                collection_phase,
                len(stationary_rows),
                float(stationary_targets.mean()),
                float(stationary_targets.max()),
                float(stationary_predictions.mean()),
                float(stationary_predictions.max()),
            )
        if tb_logger is not None:
            tb_logger.add_scalar(
                'episodic_novelty_shadow/real_target_mean', float(real_targets.mean()), collection_phase
            )
            tb_logger.add_scalar(
                'episodic_novelty_shadow/prediction_mean', float(shadow_predictions.mean()), collection_phase
            )
            tb_logger.add_scalar(
                'episodic_novelty_shadow/prediction_mae',
                float(np.abs(shadow_predictions - real_targets).mean()),
                collection_phase,
            )
            tb_logger.add_scalar(
                'episodic_novelty_shadow/target_prediction_correlation',
                target_prediction_correlation,
                collection_phase,
            )
            if stationary_rows:
                tb_logger.add_scalar(
                    'episodic_novelty_shadow/stationary_real_target_mean',
                    float(stationary_targets.mean()),
                    collection_phase,
                )
                tb_logger.add_scalar(
                    'episodic_novelty_shadow/stationary_prediction_mean',
                    float(stationary_predictions.mean()),
                    collection_phase,
                )
    episodic_history_rows = [
        row for row in selected_rows
        if row.get('intrinsic_reward_source') == 'episodic_novelty'
        and np.isfinite(float(row.get('episodic_novelty_real_target', float('nan'))))
        and np.isfinite(float(row.get('episodic_novelty_history_shadow_reward_mean', float('nan'))))
    ]
    if episodic_history_rows:
        history_targets = np.asarray(
            [float(row['episodic_novelty_real_target']) for row in episodic_history_rows],
            dtype=np.float32,
        )
        history_predictions = np.asarray(
            [
                float(row['episodic_novelty_history_shadow_reward_mean'])
                for row in episodic_history_rows
            ],
            dtype=np.float32,
        )
        history_mae = float(np.abs(history_predictions - history_targets).mean())
        if history_targets.std() > 1e-8 and history_predictions.std() > 1e-8:
            history_correlation = float(np.corrcoef(history_targets, history_predictions)[0, 1])
        else:
            history_correlation = 0.0
        feedforward_predictions = np.asarray(
            [float(row.get('episodic_novelty_shadow_reward_mean', float('nan'))) for row in episodic_history_rows],
            dtype=np.float32,
        )
        finite_feedforward = np.isfinite(feedforward_predictions)
        feedforward_mae = (
            float(np.abs(feedforward_predictions[finite_feedforward] - history_targets[finite_feedforward]).mean())
            if finite_feedforward.any() else float('nan')
        )
        early_rows = [row for row in episodic_history_rows if int(row['segment_timestep']) < 50]
        late_rows = [row for row in episodic_history_rows if int(row['segment_timestep']) >= 150]

        def _history_row_mean(source_rows, key):
            if not source_rows:
                return float('nan')
            return float(np.mean([float(row[key]) for row in source_rows]))

        history_stationary_rows = [
            row for row in episodic_history_rows
            if _finite_float(row.get('episode_return'), 0.0) <= -199.0
            and (
                _finite_float(row.get('future_max_position_remaining'), float('inf'))
                - _finite_float(row.get('future_min_position_remaining'), float('-inf'))
            ) <= 0.05
            and _finite_float(row.get('future_max_abs_velocity_remaining'), float('inf')) <= 0.005
        ]
        history_action_means = {
            int(action): float(np.mean([
                float(row['episodic_novelty_history_shadow_reward_mean'])
                for row in rows
                if int(row['action']) == int(action)
                and np.isfinite(float(row.get('episodic_novelty_history_shadow_reward_mean', float('nan'))))
            ]))
            for action in sorted({int(row['action']) for row in rows})
        }
        logging.info(
            "[EPISODIC_NOVELTY_HISTORY_SHADOW] phase %s selected %s | target %.4f, "
            "prediction %.4f, MAE %.4f vs feedforward %.4f, corr %.4f, "
            "early target/pred %.4f/%.4f, late target/pred %.4f/%.4f, actions %s",
            collection_phase,
            len(episodic_history_rows),
            float(history_targets.mean()),
            float(history_predictions.mean()),
            history_mae,
            feedforward_mae,
            history_correlation,
            _history_row_mean(early_rows, 'episodic_novelty_real_target'),
            _history_row_mean(early_rows, 'episodic_novelty_history_shadow_reward_mean'),
            _history_row_mean(late_rows, 'episodic_novelty_real_target'),
            _history_row_mean(late_rows, 'episodic_novelty_history_shadow_reward_mean'),
            {action: round(value, 4) for action, value in history_action_means.items()},
        )
        if history_stationary_rows:
            logging.info(
                "[EPISODIC_NOVELTY_HISTORY_STATIONARY] phase %s decisions %s | target %.4f, "
                "history prediction %.4f, feedforward prediction %.4f",
                collection_phase,
                len(history_stationary_rows),
                _history_row_mean(history_stationary_rows, 'episodic_novelty_real_target'),
                _history_row_mean(
                    history_stationary_rows, 'episodic_novelty_history_shadow_reward_mean'
                ),
                _history_row_mean(
                    history_stationary_rows, 'episodic_novelty_shadow_reward_mean'
                ),
            )
        if tb_logger is not None:
            tb_logger.add_scalar(
                'episodic_novelty_history_shadow/prediction_mae', history_mae, collection_phase
            )
            tb_logger.add_scalar(
                'episodic_novelty_history_shadow/feedforward_prediction_mae',
                feedforward_mae,
                collection_phase,
            )
            tb_logger.add_scalar(
                'episodic_novelty_history_shadow/target_prediction_correlation',
                history_correlation,
                collection_phase,
            )
            tb_logger.add_scalar(
                'episodic_novelty_history_shadow/late_prediction_mean',
                _history_row_mean(late_rows, 'episodic_novelty_history_shadow_reward_mean'),
                collection_phase,
            )
    if selected_rows:
        logging.info(
            "[ROOT_ACTION_Q_NORM] phase %s selected avg | raw_mean_q %.4f, normalized_mean_q %.4f, "
            "raw_qstd %.4f, calibrated_qstd %.4f, normalized_qstd %.4f, tree_q_range %.4f",
            collection_phase,
            float(np.mean([float(row['mean_q']) for row in selected_rows])),
            float(np.mean([float(row['normalized_mean_q']) for row in selected_rows])),
            float(np.mean([float(row['raw_q_std']) for row in selected_rows])),
            float(np.mean([float(row['calibrated_q_std']) for row in selected_rows])),
            float(np.mean([float(row['normalized_q_std']) for row in selected_rows])),
            float(np.mean([float(row['q_range']) for row in selected_rows])),
        )

        rnd_landscape_rows = [
            row for row in selected_rows
            if float(row.get('rnd_landscape_query_enabled', 0.0)) > 0.5
        ]
        if rnd_landscape_rows:
            all_query_action_rows = [
                row for row in rows
                if float(row.get('rnd_landscape_query_enabled', 0.0)) > 0.5
            ]
            one_step_predicted = np.asarray([
                float(row['rnd_landscape_query_predicted_one_step_reward'])
                for row in all_query_action_rows
            ], dtype=np.float32)
            depth_counts = {}
            for depth in range(1, rnd_landscape_query_max_horizon + 1):
                count_key = f'rnd_landscape_query_tree_depth_{depth}_edge_count'
                edge_count = int(sum(int(row.get(count_key, 0)) for row in rnd_landscape_rows))
                if edge_count > 0:
                    depth_counts[depth] = edge_count
            logging.info(
                "[RND_LANDSCAPE_QUERY] phase %s roots=%d snapshots=%.1f queries=%.1f | "
                "query_error mean %.6g min %.6g max %.6g | one_step predicted %.6g | "
                "tree_edges %.1f predicted %.6g depth_counts %s",
                collection_phase,
                len(rnd_landscape_rows),
                float(np.mean([
                    float(row['rnd_landscape_query_snapshot_count'])
                    for row in rnd_landscape_rows
                ])),
                float(np.mean([
                    float(row['rnd_landscape_query_count']) for row in rnd_landscape_rows
                ])),
                float(np.mean([
                    float(row['rnd_landscape_query_error_mean']) for row in rnd_landscape_rows
                ])),
                float(np.mean([
                    float(row['rnd_landscape_query_error_min']) for row in rnd_landscape_rows
                ])),
                float(np.mean([
                    float(row['rnd_landscape_query_error_max']) for row in rnd_landscape_rows
                ])),
                float(one_step_predicted.mean()),
                float(np.mean([
                    float(row['rnd_landscape_query_tree_edge_count']) for row in rnd_landscape_rows
                ])),
                float(np.mean([
                    float(row['rnd_landscape_query_tree_predicted_reward_mean'])
                    for row in rnd_landscape_rows
                ])),
                depth_counts,
            )

        if any(float(row.get('depth_normalized_backup_disagreement_enabled', 0.0)) > 0.5 for row in selected_rows):
            logging.info(
                '[DEPTH_CORRECTED_AGGREGATE_QSTD] phase %s selected avg | aggregate_calibrated %.4f, '
                'corrected_uncapped %.4f, corrected_used %.4f, correction_ratio %.4f, cap_rate %.3f, '
                'legacy_backup_mean %.4f, legacy_backup_rms %.4f, legacy_depth_corrected_mean %.4f, '
                'cancellation %.4f, aggregate_over_rms %.4f, applied_scale %.4f, '
                'changed_vs_aggregate_argmax %.3f',
                collection_phase,
                float(np.mean([float(row['aggregate_calibrated_q_std']) for row in selected_rows])),
                float(np.mean([
                    float(row['depth_corrected_aggregate_q_std_uncapped']) for row in selected_rows
                ])),
                float(np.mean([
                    float(row['depth_corrected_aggregate_q_std']) for row in selected_rows
                ])),
                float(np.mean([float(row['depth_correction_ratio']) for row in selected_rows])),
                float(np.mean([float(row['depth_correction_cap_applied']) for row in selected_rows])),
                float(np.mean([float(row['calibrated_backup_disagreement_mean']) for row in selected_rows])),
                float(np.mean([float(row['calibrated_backup_disagreement_rms']) for row in selected_rows])),
                float(np.mean([
                    float(row['depth_normalized_backup_disagreement_mean']) for row in selected_rows
                ])),
                float(np.mean([float(row['backup_disagreement_cancellation']) for row in selected_rows])),
                float(np.mean([
                    float(row['backup_disagreement_cancellation_ratio']) for row in selected_rows
                ])),
                float(np.mean([
                    float(row['applied_depth_disagreement_scale_mean']) for row in selected_rows
                ])),
                float(np.mean([
                    float(row['aggregate_qstd_counterfactual_changed_argmax']) for row in selected_rows
                ])),
            )
    depth_matched_selected_rows = [
        row for row in selected_rows
        if float(row.get('depth_matched_available', 0.0)) > 0.5
    ]
    if selected_rows and any(
            float(row.get('depth_matched_qstd_diagnostics_enabled', 0.0)) > 0.5
            for row in selected_rows
    ):
        available_fraction = len(depth_matched_selected_rows) / max(1, len(selected_rows))
        if depth_matched_selected_rows:
            logging.info(
                "[DEPTH_MATCHED_QSTD] phase %s selected avg | available %.3f, common_depths %.2f, "
                "current_raw %.3f, matched_raw %.3f, current_calibrated %.3f, matched_calibrated %.3f, "
                "counterfactual_changed_argmax %.3f",
                collection_phase,
                available_fraction,
                float(np.mean([
                    int(row['depth_matched_common_depth_count']) for row in depth_matched_selected_rows
                ])),
                float(np.mean([float(row['raw_q_std']) for row in depth_matched_selected_rows])),
                float(np.mean([
                    float(row['depth_matched_raw_q_std']) for row in depth_matched_selected_rows
                ])),
                float(np.mean([float(row['calibrated_q_std']) for row in depth_matched_selected_rows])),
                float(np.mean([
                    float(row['depth_matched_calibrated_q_std'])
                    for row in depth_matched_selected_rows
                ])),
                float(np.mean([
                    float(row['depth_matched_counterfactual_changed_argmax'])
                    for row in depth_matched_selected_rows
                ])),
            )
            for depth in range(1, depth_matched_qstd_max_depth + 1):
                depth_rows = [
                    row for row in selected_rows
                    if int(row.get(f'depth_{depth}_return_count', 0)) > 0
                ]
                if not depth_rows:
                    continue
                logging.info(
                    "[DEPTH_MATCHED_QSTD_DEPTH] phase %s depth %s selected avg | present %.3f, "
                    "returns %.2f, raw_qstd %.3f, calibrated_qstd %.3f",
                    collection_phase,
                    depth,
                    len(depth_rows) / max(1, len(selected_rows)),
                    float(np.mean([
                        int(row[f'depth_{depth}_return_count']) for row in depth_rows
                    ])),
                    float(np.mean([float(row[f'depth_{depth}_raw_q_std']) for row in depth_rows])),
                    float(np.mean([
                        float(row[f'depth_{depth}_calibrated_q_std']) for row in depth_rows
                    ])),
                )
            for action in sorted({int(row['action']) for row in rows}):
                action_rows = [
                    row for row in rows
                    if int(row['action']) == action
                    and float(row.get('depth_matched_available', 0.0)) > 0.5
                ]
                if not action_rows:
                    continue
                exact_depth_raw_qstd = {
                    depth: round(
                        float(np.mean([
                            float(row[f'depth_{depth}_raw_q_std'])
                            for row in action_rows
                            if int(row.get(f'depth_{depth}_return_count', 0)) > 0
                        ])),
                        3,
                    )
                    for depth in range(1, depth_matched_qstd_max_depth + 1)
                    if any(
                        int(row.get(f'depth_{depth}_return_count', 0)) > 0
                        for row in action_rows
                    )
                }
                logging.info(
                    "[DEPTH_MATCHED_QSTD_ACTION] phase %s action %s roots %s | "
                    "current_raw %.3f, matched_raw %.3f, delta %+.3f, exact_depth_raw %s",
                    collection_phase,
                    action,
                    len(action_rows),
                    float(np.mean([float(row['raw_q_std']) for row in action_rows])),
                    float(np.mean([float(row['depth_matched_raw_q_std']) for row in action_rows])),
                    float(np.mean([
                        float(row['depth_matched_raw_q_std']) - float(row['raw_q_std'])
                        for row in action_rows
                    ])),
                    exact_depth_raw_qstd,
                )
        else:
            logging.info(
                "[DEPTH_MATCHED_QSTD] phase %s selected avg | available %.3f (no common root-action depth)",
                collection_phase,
                available_fraction,
            )
    bounded_normal_rows = [
        row for row in rows
        if float(row.get('bounded_ensemble_mode', 0.0)) > 0.5
        and float(row.get('pure_exploration_episode', 0.0)) < 0.5
    ]
    mixed_reward_selected_rows = [
        row for row in selected_rows
        if str(row.get('search_mode', '')) == 'mixed_reward_shared_tree'
    ]
    if mixed_reward_selected_rows:
        logging.info(
            "[MIXED_REWARD_TREE_DIAG] phase %s selected avg | extrinsic_q %.3f, "
            "intrinsic_q %.3f, beta_intrinsic_q %.3f, mixed_q %.3f, "
            "identity_error %.6f, normalized_mixed_q %.3f, q_range %.3f, "
            "prior_score %.3f, total_score %.3f",
            collection_phase,
            float(np.mean([float(row['mean_q']) for row in mixed_reward_selected_rows])),
            float(np.mean([float(row['intrinsic_value']) for row in mixed_reward_selected_rows])),
            float(np.mean([float(row['intrinsic_score']) for row in mixed_reward_selected_rows])),
            float(np.mean([float(row['mixed_mean_q']) for row in mixed_reward_selected_rows])),
            float(np.mean([
                abs(
                    float(row['mixed_mean_q'])
                    - float(row['mean_q'])
                    - float(row['intrinsic_score'])
                )
                for row in mixed_reward_selected_rows
            ])),
            float(np.mean([
                float(row['normalized_mean_q']) for row in mixed_reward_selected_rows
            ])),
            float(np.mean([float(row['q_range']) for row in mixed_reward_selected_rows])),
            float(np.mean([float(row['prior_score']) for row in mixed_reward_selected_rows])),
            float(np.mean([float(row['total_score']) for row in mixed_reward_selected_rows])),
        )
    bounded_normal_selected_rows = [row for row in bounded_normal_rows if int(row['selected']) == 1]
    if bounded_normal_selected_rows:
        winner_counts = {}
        for row in bounded_normal_selected_rows:
            member_id = int(row.get('ensemble_envelope_member_id', -1))
            winner_counts[member_id] = winner_counts.get(member_id, 0) + 1
        logging.info(
            "[BOUNDED_ENSEMBLE_DIAG] phase %s normal selected avg | bounded_q %.3f, envelope_gap %.3f, "
            "selected_cap_fraction %.3f, all_edge_cap_fraction %.3f, changed_mean_argmax %.3f, "
            "same_member_all_actions %.3f, selected_winner_counts %s",
            collection_phase,
            float(np.mean([float(row['bounded_ensemble_q']) for row in bounded_normal_selected_rows])),
            float(
                np.mean([float(row['ensemble_envelope_gap_from_mean']) for row in bounded_normal_selected_rows])
            ),
            float(
                np.mean([float(row['ensemble_envelope_cap_applied']) for row in bounded_normal_selected_rows])
            ),
            float(np.mean([float(row['ensemble_envelope_cap_applied']) for row in bounded_normal_rows])),
            float(
                np.mean([float(row['bounded_changed_mean_score_argmax']) for row in bounded_normal_selected_rows])
            ),
            float(
                np.mean([float(row['same_envelope_member_all_actions']) for row in bounded_normal_selected_rows])
            ),
            winner_counts,
        )
    if selected_rows and any(float(row['separate_intrinsic_tree_enabled']) > 0.0 for row in selected_rows):
        logging.info(
            "[SEPARATE_INTRINSIC_TREE_DIAG] phase %s selected avg | intrinsic_tree_value %.3f, "
            "immediate %.3f, latent_dd %.3f, projection_dd %.3f, prediction_dd %.3f, "
            "leaf_raw %.3f, leaf_weighted %.3f, final_intrinsic_score %.3f, "
            "main_puct %.3f, final_logit %.3f, base_prob %.3f, final_prob %.3f, "
            "policy_tv %.3f, changed_argmax %.3f, entropy %.3f -> %.3f",
            collection_phase,
            float(np.mean([float(row['intrinsic_tree_intrinsic_value']) for row in selected_rows])),
            float(np.mean([float(row['intrinsic_tree_intrinsic_immediate_reward']) for row in selected_rows])),
            float(np.mean([float(row['intrinsic_tree_latent_dd_immediate']) for row in selected_rows])),
            float(np.mean([float(row['intrinsic_tree_projection_dd_immediate']) for row in selected_rows])),
            float(
                np.mean([float(row['intrinsic_tree_prediction_projection_dd_immediate']) for row in selected_rows])
            ),
            float(np.mean([float(row['intrinsic_tree_intrinsic_leaf_value_raw_contribution_mean']) for row in selected_rows])),
            float(np.mean([float(row['intrinsic_tree_intrinsic_leaf_value_contribution_mean']) for row in selected_rows])),
            float(np.mean([float(row['final_intrinsic_score']) for row in selected_rows])),
            float(np.mean([float(row['main_tree_puct_total_score']) for row in selected_rows])),
            float(np.mean([float(row['intrinsic_tree_final_logit']) for row in selected_rows])),
            float(np.mean([float(row['intrinsic_tree_base_probability']) for row in selected_rows])),
            float(np.mean([float(row['final_action_probability']) for row in selected_rows])),
            float(np.mean([float(row['intrinsic_tree_policy_total_variation']) for row in selected_rows])),
            float(np.mean([float(row['intrinsic_tree_changed_argmax']) for row in selected_rows])),
            float(np.mean([float(row['intrinsic_tree_base_entropy']) for row in selected_rows])),
            float(np.mean([float(row['intrinsic_tree_final_entropy']) for row in selected_rows])),
        )
        _log_intrinsic_tree_failure_diagnostics(cfg, rows, selected_rows, tb_logger, collection_phase)
        intrinsic_episode_rows = [
            row for row in selected_rows
            if str(row.get('intrinsic_tree_action_policy_source', '')) == 'intrinsic_tree'
        ]
        if intrinsic_episode_rows:
            logging.info(
                "[INTRINSIC_TREE_PURE_EPISODE_DIAG] phase %s decisions %s/%s | "
                "coverage_complete %.3f, selected_visits %.2f, selected_QI %.4f, "
                "selected_prob %.3f, main_to_intrinsic_tv %.3f, policy_target_main %.3f",
                collection_phase,
                len(intrinsic_episode_rows),
                len(selected_rows),
                float(np.mean([
                    float(row.get('intrinsic_tree_root_forced_coverage_complete', 0.0))
                    for row in intrinsic_episode_rows
                ])),
                float(np.mean([
                    float(row.get('intrinsic_tree_visit_count', 0.0))
                    for row in intrinsic_episode_rows
                ])),
                float(np.mean([
                    float(row.get('intrinsic_tree_intrinsic_value', 0.0))
                    for row in intrinsic_episode_rows
                ])),
                float(np.mean([
                    float(row.get('intrinsic_tree_final_probability', 0.0))
                    for row in intrinsic_episode_rows
                ])),
                float(np.mean([
                    float(row.get('intrinsic_tree_policy_total_variation', 0.0))
                    for row in intrinsic_episode_rows
                ])),
                float(np.mean([
                    str(row.get('intrinsic_tree_policy_target_source', '')) == 'main_tree'
                    for row in intrinsic_episode_rows
                ])),
            )
    if selected_rows and any(float(row.get('final_qstd_enabled', 0.0)) > 0.0 for row in selected_rows):
        changed_count = sum(1 for row in selected_rows if float(row.get('final_qstd_changed_argmax', 0.0)) > 0.5)
        logging.info(
            "[FINAL_QSTD_DIAG] phase %s selected avg | base_prob %.3f, final_prob %.3f, "
            "base_logit %.3f, qstd_logit %.3f, final_logit %.3f, changed_argmax %.3f",
            collection_phase,
            float(np.mean([float(row['final_qstd_base_probability']) for row in selected_rows])),
            float(np.mean([float(row['final_qstd_probability']) for row in selected_rows])),
            float(np.mean([float(row['final_qstd_base_logit']) for row in selected_rows])),
            float(np.mean([float(row['final_qstd_qstd_logit']) for row in selected_rows])),
            float(np.mean([float(row['final_qstd_final_logit']) for row in selected_rows])),
            float(changed_count / max(1, len(selected_rows))),
        )
    if selected_rows and any(float(row.get('separate_exploration_tree_enabled', 0.0)) > 0.0 for row in selected_rows):
        changed_count = sum(
            1 for row in selected_rows
            if float(row.get('separate_exploration_tree_changed_argmax', 0.0)) > 0.5
        )
        logging.info(
            "[SEPARATE_EXPLORATION_TREE_DIAG] phase %s selected avg | raw_score %.3f, norm_score %.3f, "
            "explore_logit %.3f, base_prob %.3f, final_prob %.3f, changed_argmax %.3f",
            collection_phase,
            float(np.mean([float(row['separate_exploration_tree_raw_score']) for row in selected_rows])),
            float(np.mean([float(row['separate_exploration_tree_normalized_score']) for row in selected_rows])),
            float(np.mean([float(row['separate_exploration_tree_exploration_logit']) for row in selected_rows])),
            float(np.mean([float(row['separate_exploration_tree_base_probability']) for row in selected_rows])),
                float(np.mean([float(row['separate_exploration_tree_probability']) for row in selected_rows])),
                float(changed_count / max(1, len(selected_rows))),
            )
        for depth in range(1, 9):
            key = f'separate_exploration_tree_depth{depth}_topk_qstd_mean'
            if key not in selected_rows[0]:
                continue
            values = [float(row.get(key, 0.0)) for row in selected_rows]
            if not any(abs(value) > 1e-12 for value in values):
                continue
            logging.info(
                "[SEPARATE_EXPLORATION_TREE_DEPTH_DIAG] phase %s depth %s selected avg | "
                "topk_qstd %.3f, max_qstd %.3f, mean_qstd %.3f",
                collection_phase,
                depth,
                float(np.mean(values)),
                float(
                    np.mean(
                        [
                            float(row.get(f'separate_exploration_tree_depth{depth}_qstd_max', 0.0))
                            for row in selected_rows
                        ]
                    )
                ),
                float(
                    np.mean(
                        [
                            float(row.get(f'separate_exploration_tree_depth{depth}_qstd_mean', 0.0))
                            for row in selected_rows
                        ]
                    )
                ),
            )
    pure_rows = [
        row for row in selected_rows
        if float(row.get('pure_exploration_episode', 0.0)) > 0.5
    ]
    if pure_rows:
        new_sequence_fraction = float(
            np.mean([str(row.get('pure_exploration_action_source', '')) == 'new_sequence' for row in pure_rows])
        )
        commit_fraction = float(
            np.mean([str(row.get('pure_exploration_action_source', '')) == 'commit' for row in pure_rows])
        )
        logging.info(
            "[PURE_EXPLORATION_EPISODE_DIAG] phase %s selected decisions %s/%s | "
            "new_sequence %.3f, commit %.3f, seq_score %.3f, mean_q %.3f, "
            "raw_qstd %.3f, centered_qstd %.3f, action_prob %.3f",
            collection_phase,
            len(pure_rows),
            len(selected_rows),
            new_sequence_fraction,
            commit_fraction,
            float(np.mean([float(row['pure_exploration_selected_sequence_score']) for row in pure_rows])),
            float(np.mean([float(row['pure_exploration_selected_sequence_mean_q']) for row in pure_rows])),
            float(np.mean([float(row['pure_exploration_selected_sequence_raw_qstd']) for row in pure_rows])),
            float(
                np.mean(
                    [float(row['pure_exploration_selected_sequence_member_centered_qstd']) for row in pure_rows]
                )
            ),
            float(np.mean([float(row['pure_exploration_action_probability']) for row in pure_rows])),
        )
    ssl_rows = [
        row for row in selected_rows
        if float(row.get('real_transition_ssl_valid', 0.0)) > 0.5
    ]
    if ssl_rows:
        logging.info(
            "[REAL_TRANSITION_SSL_DIAG] phase %s selected avg | pred_err %.4f, proj_err %.4f, "
            "pred_pair_dd %.4f, proj_pair_dd %.4f, target_proj_dd %.4f, target_latent_dd %.4f",
            collection_phase,
            float(np.mean([float(row['real_transition_ssl_prediction_error_mean']) for row in ssl_rows])),
            float(np.mean([float(row['real_transition_ssl_projection_error_mean']) for row in ssl_rows])),
            float(np.mean([float(row['real_transition_prediction_projection_pairwise_dd']) for row in ssl_rows])),
            float(np.mean([float(row['real_transition_projection_pairwise_dd']) for row in ssl_rows])),
            float(np.mean([float(row['real_transition_target_projection_pairwise_dd']) for row in ssl_rows])),
            float(np.mean([float(row['real_transition_target_latent_pairwise_dd']) for row in ssl_rows])),
        )
    if tb_logger is not None:
        tb_logger.add_scalar('collection_phase/root_action_term_rows', float(len(rows)), collection_phase)
        if selected_rows:
            tb_logger.add_scalar(
                'collection_phase/selected_action_mean_q',
                float(np.mean([float(row['mean_q']) for row in selected_rows])),
                collection_phase,
            )
            tb_logger.add_scalar(
                'collection_phase/selected_action_normalized_mean_q',
                float(np.mean([float(row['normalized_mean_q']) for row in selected_rows])),
                collection_phase,
            )
            tb_logger.add_scalar(
                'collection_phase/selected_action_normalized_q_std',
                float(np.mean([float(row['normalized_q_std']) for row in selected_rows])),
                collection_phase,
            )
            for metric in [
                    'aggregate_calibrated_q_std',
                    'calibrated_backup_disagreement_mean',
                    'calibrated_backup_disagreement_rms',
                    'depth_normalized_backup_disagreement_mean',
                    'depth_normalized_backup_disagreement_rms',
                    'depth_corrected_aggregate_q_std_uncapped',
                    'depth_corrected_aggregate_q_std',
                    'depth_correction_cap_applied',
                    'depth_correction_ratio',
                    'backup_disagreement_cancellation',
                    'backup_disagreement_cancellation_ratio',
                    'applied_depth_disagreement_scale_mean',
                    'aggregate_qstd_counterfactual_changed_argmax',
            ]:
                tb_logger.add_scalar(
                    f'collection_phase/selected_action_{metric}',
                    float(np.mean([float(row[metric]) for row in selected_rows])),
                    collection_phase,
                )
            tb_logger.add_scalar(
                'collection_phase/root_action_q_range',
                float(np.mean([float(row['q_range']) for row in selected_rows])),
                collection_phase,
            )
            if any(
                    float(row.get('depth_matched_qstd_diagnostics_enabled', 0.0)) > 0.5
                    for row in selected_rows
            ):
                tb_logger.add_scalar(
                    'collection_phase/depth_matched_qstd_available_fraction',
                    float(len(depth_matched_selected_rows) / max(1, len(selected_rows))),
                    collection_phase,
                )
                if depth_matched_selected_rows:
                    tb_logger.add_scalar(
                        'collection_phase/selected_action_depth_matched_raw_q_std',
                        float(np.mean([
                            float(row['depth_matched_raw_q_std'])
                            for row in depth_matched_selected_rows
                        ])),
                        collection_phase,
                    )
                    tb_logger.add_scalar(
                        'collection_phase/depth_matched_counterfactual_changed_argmax_fraction',
                        float(np.mean([
                            float(row['depth_matched_counterfactual_changed_argmax'])
                            for row in depth_matched_selected_rows
                        ])),
                        collection_phase,
                    )
            if bounded_normal_selected_rows:
                tb_logger.add_scalar(
                    'collection_phase/bounded_ensemble_selected_cap_fraction',
                    float(
                        np.mean(
                            [float(row['ensemble_envelope_cap_applied']) for row in bounded_normal_selected_rows]
                        )
                    ),
                    collection_phase,
                )
                tb_logger.add_scalar(
                    'collection_phase/bounded_ensemble_all_edge_cap_fraction',
                    float(np.mean([float(row['ensemble_envelope_cap_applied']) for row in bounded_normal_rows])),
                    collection_phase,
                )
                tb_logger.add_scalar(
                    'collection_phase/bounded_ensemble_changed_mean_argmax_fraction',
                    float(
                        np.mean(
                            [float(row['bounded_changed_mean_score_argmax']) for row in bounded_normal_selected_rows]
                        )
                    ),
                    collection_phase,
                )
                tb_logger.add_scalar(
                    'collection_phase/bounded_ensemble_same_member_all_actions_fraction',
                    float(
                        np.mean(
                            [float(row['same_envelope_member_all_actions']) for row in bounded_normal_selected_rows]
                        )
                    ),
                    collection_phase,
                )
                tb_logger.add_scalar(
                    'collection_phase/bounded_ensemble_selected_envelope_gap',
                    float(
                        np.mean(
                            [float(row['ensemble_envelope_gap_from_mean']) for row in bounded_normal_selected_rows]
                        )
                    ),
                    collection_phase,
                )
                for member_id in range(int(cfg.policy.model.ensemble_size)):
                    tb_logger.add_scalar(
                        f'collection_phase/bounded_ensemble_selected_winner_member_{member_id}_fraction',
                        float(
                            np.mean(
                                [
                                    int(row['ensemble_envelope_member_id']) == member_id
                                    for row in bounded_normal_selected_rows
                                ]
                            )
                        ),
                        collection_phase,
                    )
            tb_logger.add_scalar(
                'collection_phase/selected_action_total_score',
                float(np.mean([float(row['total_score']) for row in selected_rows])),
                collection_phase,
            )
            tb_logger.add_scalar(
                'collection_phase/selected_action_main_tree_total_score',
                float(np.mean([float(row['main_tree_total_score']) for row in selected_rows])),
                collection_phase,
            )
            tb_logger.add_scalar(
                'collection_phase/selected_action_intrinsic_tree_value',
                float(np.mean([float(row['intrinsic_tree_intrinsic_value']) for row in selected_rows])),
                collection_phase,
            )
            tb_logger.add_scalar(
                'collection_phase/selected_action_intrinsic_tree_reward_return',
                float(
                    np.mean([float(row['intrinsic_tree_intrinsic_reward_return_mean']) for row in selected_rows])
                ),
                collection_phase,
            )
            tb_logger.add_scalar(
                'collection_phase/selected_action_intrinsic_tree_immediate_reward',
                float(
                    np.mean([float(row['intrinsic_tree_intrinsic_immediate_reward']) for row in selected_rows])
                ),
                collection_phase,
            )
            tb_logger.add_scalar(
                'collection_phase/selected_action_intrinsic_tree_latent_dd_immediate',
                float(np.mean([float(row['intrinsic_tree_latent_dd_immediate']) for row in selected_rows])),
                collection_phase,
            )
            tb_logger.add_scalar(
                'collection_phase/selected_action_intrinsic_tree_projection_dd_immediate',
                float(np.mean([float(row['intrinsic_tree_projection_dd_immediate']) for row in selected_rows])),
                collection_phase,
            )
            tb_logger.add_scalar(
                'collection_phase/selected_action_intrinsic_tree_prediction_projection_dd_immediate',
                float(
                    np.mean(
                        [float(row['intrinsic_tree_prediction_projection_dd_immediate']) for row in selected_rows]
                    )
                ),
                collection_phase,
            )
            if ssl_rows:
                tb_logger.add_scalar(
                    'collection_phase/selected_action_real_transition_ssl_prediction_error',
                    float(np.mean([float(row['real_transition_ssl_prediction_error_mean']) for row in ssl_rows])),
                    collection_phase,
                )
                tb_logger.add_scalar(
                    'collection_phase/selected_action_real_transition_ssl_projection_error',
                    float(np.mean([float(row['real_transition_ssl_projection_error_mean']) for row in ssl_rows])),
                    collection_phase,
                )
                tb_logger.add_scalar(
                    'collection_phase/selected_action_real_transition_prediction_pairwise_dd',
                    float(np.mean([float(row['real_transition_prediction_projection_pairwise_dd']) for row in ssl_rows])),
                    collection_phase,
                )
                tb_logger.add_scalar(
                    'collection_phase/selected_action_real_transition_projection_pairwise_dd',
                    float(np.mean([float(row['real_transition_projection_pairwise_dd']) for row in ssl_rows])),
                    collection_phase,
                )
            tb_logger.add_scalar(
                'collection_phase/selected_action_intrinsic_tree_leaf_value_contribution',
                float(
                    np.mean(
                        [float(row['intrinsic_tree_intrinsic_leaf_value_contribution_mean']) for row in selected_rows]
                    )
                ),
                collection_phase,
            )
            tb_logger.add_scalar(
                'collection_phase/selected_action_intrinsic_tree_immediate_weighted_return',
                float(
                    np.mean(
                        [float(row['intrinsic_tree_intrinsic_reward_weighted_return_mean']) for row in selected_rows]
                    )
                ),
                collection_phase,
            )
            tb_logger.add_scalar(
                'collection_phase/selected_action_intrinsic_tree_leaf_value_raw_contribution',
                float(
                    np.mean(
                        [
                            float(row['intrinsic_tree_intrinsic_leaf_value_raw_contribution_mean'])
                            for row in selected_rows
                        ]
                    )
                ),
                collection_phase,
            )
            tb_logger.add_scalar(
                'collection_phase/selected_action_final_intrinsic_score',
                float(np.mean([float(row['final_intrinsic_score']) for row in selected_rows])),
                collection_phase,
            )
            tb_logger.add_scalar(
                'collection_phase/selected_action_final_action_probability',
                float(np.mean([float(row['final_action_probability']) for row in selected_rows])),
                collection_phase,
            )
            if any(float(row.get('final_qstd_enabled', 0.0)) > 0.0 for row in selected_rows):
                tb_logger.add_scalar(
                    'collection_phase/final_qstd_changed_argmax_fraction',
                    float(
                        np.mean(
                            [
                                float(row.get('final_qstd_changed_argmax', 0.0))
                                for row in selected_rows
                            ]
                        )
                    ),
                    collection_phase,
                )
                tb_logger.add_scalar(
                    'collection_phase/selected_action_final_qstd_base_probability',
                    float(np.mean([float(row['final_qstd_base_probability']) for row in selected_rows])),
                    collection_phase,
                )
                tb_logger.add_scalar(
                    'collection_phase/selected_action_final_qstd_probability',
                    float(np.mean([float(row['final_qstd_probability']) for row in selected_rows])),
                    collection_phase,
                )
                tb_logger.add_scalar(
                    'collection_phase/selected_action_final_qstd_base_logit',
                    float(np.mean([float(row['final_qstd_base_logit']) for row in selected_rows])),
                    collection_phase,
                )
                tb_logger.add_scalar(
                    'collection_phase/selected_action_final_qstd_qstd_logit',
                    float(np.mean([float(row['final_qstd_qstd_logit']) for row in selected_rows])),
                    collection_phase,
                )
                tb_logger.add_scalar(
                    'collection_phase/selected_action_final_qstd_final_logit',
                    float(np.mean([float(row['final_qstd_final_logit']) for row in selected_rows])),
                    collection_phase,
                )
            if any(float(row.get('separate_exploration_tree_enabled', 0.0)) > 0.0 for row in selected_rows):
                tb_logger.add_scalar(
                    'collection_phase/separate_exploration_tree_changed_argmax_fraction',
                    float(
                        np.mean(
                            [
                                float(row.get('separate_exploration_tree_changed_argmax', 0.0))
                                for row in selected_rows
                            ]
                        )
                    ),
                    collection_phase,
                )
                tb_logger.add_scalar(
                    'collection_phase/selected_action_separate_exploration_tree_raw_score',
                    float(np.mean([float(row['separate_exploration_tree_raw_score']) for row in selected_rows])),
                    collection_phase,
                )
                tb_logger.add_scalar(
                    'collection_phase/selected_action_separate_exploration_tree_normalized_score',
                    float(
                        np.mean([float(row['separate_exploration_tree_normalized_score']) for row in selected_rows])
                    ),
                    collection_phase,
                )
                tb_logger.add_scalar(
                    'collection_phase/selected_action_separate_exploration_tree_exploration_logit',
                    float(
                        np.mean([float(row['separate_exploration_tree_exploration_logit']) for row in selected_rows])
                    ),
                    collection_phase,
                )
                tb_logger.add_scalar(
                    'collection_phase/selected_action_separate_exploration_tree_probability',
                    float(np.mean([float(row['separate_exploration_tree_probability']) for row in selected_rows])),
                    collection_phase,
                )
                for depth in range(1, 9):
                    key = f'separate_exploration_tree_depth{depth}_topk_qstd_mean'
                    values = [float(row.get(key, 0.0)) for row in selected_rows]
                    if not any(abs(value) > 1e-12 for value in values):
                        continue
                    tb_logger.add_scalar(
                        f'collection_phase/separate_exploration_tree_depth{depth}_selected_topk_qstd',
                        float(np.mean(values)),
                        collection_phase,
                    )
            if pure_rows:
                tb_logger.add_scalar(
                    'collection_phase/pure_exploration_selected_decision_fraction',
                    float(len(pure_rows) / max(1, len(selected_rows))),
                    collection_phase,
                )
                tb_logger.add_scalar(
                    'collection_phase/pure_exploration_new_sequence_fraction',
                    float(
                        np.mean(
                            [
                                str(row.get('pure_exploration_action_source', '')) == 'new_sequence'
                                for row in pure_rows
                            ]
                        )
                    ),
                    collection_phase,
                )
                tb_logger.add_scalar(
                    'collection_phase/pure_exploration_selected_sequence_qstd',
                    float(np.mean([float(row['pure_exploration_selected_sequence_qstd']) for row in pure_rows])),
                    collection_phase,
                )
                tb_logger.add_scalar(
                    'collection_phase/pure_exploration_selected_sequence_score',
                    float(np.mean([float(row['pure_exploration_selected_sequence_score']) for row in pure_rows])),
                    collection_phase,
                )
                tb_logger.add_scalar(
                    'collection_phase/pure_exploration_selected_sequence_mean_q',
                    float(np.mean([float(row['pure_exploration_selected_sequence_mean_q']) for row in pure_rows])),
                    collection_phase,
                )
                tb_logger.add_scalar(
                    'collection_phase/pure_exploration_selected_sequence_member_centered_qstd',
                    float(
                        np.mean(
                            [
                                float(row['pure_exploration_selected_sequence_member_centered_qstd'])
                                for row in pure_rows
                            ]
                        )
                    ),
                    collection_phase,
                )
            tb_logger.add_scalar(
                'collection_phase/selected_action_predicted_reward_return_mean',
                float(np.mean([float(row['predicted_reward_return_mean']) for row in selected_rows])),
                collection_phase,
            )
            tb_logger.add_scalar(
                'collection_phase/selected_action_predicted_reward_return_std',
                float(np.mean([float(row['predicted_reward_return_std']) for row in selected_rows])),
                collection_phase,
            )
            tb_logger.add_scalar(
                'collection_phase/selected_action_leaf_value_contribution_mean',
                float(np.mean([float(row['leaf_value_contribution_mean']) for row in selected_rows])),
                collection_phase,
            )
            tb_logger.add_scalar(
                'collection_phase/selected_action_leaf_value_contribution_std',
                float(np.mean([float(row['leaf_value_contribution_std']) for row in selected_rows])),
                collection_phase,
            )
            tb_logger.add_scalar(
                'collection_phase/selected_action_obs_position',
                float(np.mean([float(row['obs_position']) for row in selected_rows])),
                collection_phase,
            )
            tb_logger.add_scalar(
                'collection_phase/selected_action_obs_velocity',
                float(np.mean([float(row['obs_velocity']) for row in selected_rows])),
                collection_phase,
            )
            tb_logger.add_scalar(
                'collection_phase/selected_action_future_max_position_100',
                float(np.mean([float(row['future_max_position_100']) for row in selected_rows])),
                collection_phase,
            )
            tb_logger.add_scalar(
                'collection_phase/selected_action_future_position_gain_100',
                float(np.mean([float(row['future_position_gain_100']) for row in selected_rows])),
                collection_phase,
            )
            tb_logger.add_scalar(
                'collection_phase/selected_action_future_max_abs_velocity_100',
                float(np.mean([float(row['future_max_abs_velocity_100']) for row in selected_rows])),
                collection_phase,
            )
            tb_logger.add_scalar(
                'collection_phase/selected_action_future_reached_goal_remaining',
                float(np.mean([float(row['future_reached_goal_remaining']) for row in selected_rows])),
                collection_phase,
            )
            tb_logger.add_scalar(
                'collection_phase/selected_action_latent_novelty_root_disagreement',
                float(np.mean([float(row['latent_novelty_root_disagreement']) for row in selected_rows])),
                collection_phase,
            )
            tb_logger.add_scalar(
                'collection_phase/selected_action_imagined_node_latent_novelty',
                float(np.mean([float(row['imagined_node_latent_novelty']) for row in selected_rows])),
                collection_phase,
            )
            tb_logger.add_scalar(
                'collection_phase/selected_action_imagined_leaf_latent_novelty_mean',
                float(np.mean([float(row['imagined_leaf_latent_novelty_mean']) for row in selected_rows])),
                collection_phase,
            )
            tb_logger.add_scalar(
                'collection_phase/selected_action_imagined_leaf_latent_novelty_max',
                float(np.mean([float(row['imagined_leaf_latent_novelty_max']) for row in selected_rows])),
                collection_phase,
            )
            tb_logger.add_scalar(
                'collection_phase/selected_action_imagined_leaf_latent_novelty_depth_mean',
                float(np.mean([float(row['imagined_leaf_latent_novelty_depth_mean']) for row in selected_rows])),
                collection_phase,
            )


def _write_tree_trace_diagnostics(
        cfg,
        new_data: List[Any],
        tb_logger,
        collection_phase: int,
        *,
        enabled_key: str,
        segment_attr: str,
        output_filename: str,
        log_prefix: str,
        tb_key: str,
) -> None:
    if not _diagnostics_enabled(cfg) or not bool(cfg.policy.get(enabled_key, False)):
        return

    rows = []
    data, meta_list = new_data
    for segment_index, (game_segment, meta) in enumerate(zip(data, meta_list)):
        trace_segment = getattr(game_segment, segment_attr, [])
        if not trace_segment:
            continue
        real_step_count = min(
            int(meta.get('segment_length', len(game_segment))),
            len(trace_segment),
            len(game_segment.action_segment),
        )
        for timestep in range(real_step_count):
            trace = trace_segment[timestep]
            if trace is None:
                continue
            selected_action = _scalar_action(game_segment.action_segment[timestep])
            future_metrics = _future_trajectory_metrics(cfg, game_segment, timestep, real_step_count)
            timestep_value = None
            if timestep < len(game_segment.timestep_segment):
                timestep_value = _scalar_action(game_segment.timestep_segment[timestep])
            rows.append(
                {
                    'collection_phase': meta.get('collection_phase'),
                    'train_iter_at_collect': meta.get('train_iter_at_collect'),
                    'envstep_before_collect': meta.get('envstep_before_collect'),
                    'envstep_after_collect': meta.get('envstep_after_collect'),
                    'segment_index_in_collect': meta.get('segment_index_in_collect', segment_index),
                    'env_id': meta.get('env_id'),
                    'env_episode_index': meta.get('env_episode_index'),
                    'collection_episode_index': meta.get('collection_episode_index'),
                    'episode_segment_index': meta.get('episode_segment_index'),
                    'episode_return': meta.get('episode_return'),
                    'video_path': meta.get('video_path'),
                    'timestep': timestep_value if timestep_value is not None else timestep,
                    'segment_timestep': timestep,
                    'selected_action': selected_action,
                    'selected_action_direction': _action_direction(selected_action),
                    **future_metrics,
                    'trace': trace,
                }
            )

    if not rows:
        return

    diagnostics_dir = _diagnostics_dir(cfg)
    jsonl_path = os.path.join(diagnostics_dir, output_filename)
    with open(jsonl_path, 'a') as f:
        for row in rows:
            f.write(json.dumps(_jsonable(row), sort_keys=True) + '\n')

    logging.info(
        "[%s] phase %s wrote %s tree trace(s) to %s",
        log_prefix,
        collection_phase,
        len(rows),
        jsonl_path,
    )
    if tb_logger is not None:
        tb_logger.add_scalar(tb_key, float(len(rows)), collection_phase)


def _complete_episodic_history_training_sequences(cfg, new_data: List[Any]) -> List[dict]:
    if (
            not bool(cfg.policy.get('episodic_novelty_history_shadow', False))
            or cfg.policy.model.model_type != 'mlp'
    ):
        return []
    frame_stack_num = int(cfg.policy.model.get('frame_stack_num', 1))
    segments, metadata = new_data
    episodes = []
    for game_segment, meta in zip(segments, metadata):
        if not bool(meta.get('done', False)) or int(meta.get('episode_segment_index', 0)) != 0:
            continue
        valid_steps = min(
            int(meta.get('segment_length', len(game_segment))),
            int(cfg.policy.game_segment_length),
            len(game_segment.action_segment),
            len(getattr(game_segment, 'intrinsic_reward_segment', [])),
        )
        observations = np.asarray(game_segment.obs_segment)
        required_observations = frame_stack_num + valid_steps
        if valid_steps <= 0 or observations.shape[0] < required_observations:
            continue
        stacked_states = np.stack(
            [
                np.asarray(observations[step:step + frame_stack_num], dtype=np.float32).reshape(-1)
                for step in range(valid_steps + 1)
            ],
            axis=0,
        )
        episodes.append(
            {
                'observations': stacked_states,
                'actions': np.asarray(game_segment.action_segment[:valid_steps], dtype=np.int64).reshape(-1),
                'intrinsic_rewards': np.asarray(
                    game_segment.intrinsic_reward_segment[:valid_steps], dtype=np.float32
                ).reshape(-1),
                'env_id': int(meta.get('env_id', -1)),
                'collection_episode_index': int(meta.get('collection_episode_index', -1)),
            }
        )
    return episodes


def _episodic_sequence_first_visit_rewards(observations: np.ndarray) -> np.ndarray:
    """Exact RGB first-visit labels used only as supervision and reward feedback."""
    observations = np.asarray(observations)
    if observations.shape[0] < 2:
        return np.zeros(0, dtype=np.float32)
    if observations.dtype != np.uint8:
        scale = 255.0 if float(np.max(observations, initial=0.0)) <= 1.5 else 1.0
        observations = np.rint(
            np.clip(observations * scale, 0.0, 255.0)
        ).astype(np.uint8)
    seen = {np.ascontiguousarray(observations[0]).tobytes()}
    rewards = np.zeros(observations.shape[0] - 1, dtype=np.float32)
    for index in range(1, observations.shape[0]):
        key = np.ascontiguousarray(observations[index]).tobytes()
        rewards[index - 1] = float(key not in seen)
        seen.add(key)
    return rewards


def _complete_episodic_sequence_training_sequences(
        cfg, new_data: List[Any]
) -> List[dict]:
    """Reassemble complete real episodes, including episodes split into segments."""
    if not (
            bool(cfg.policy.get('episodic_sequence_novelty', False))
            or bool(
                cfg.policy.get(
                    'spatial_exploration_tree_online_training', False
                )
            )
    ):
        return []
    frame_stack_num = int(cfg.policy.model.get('frame_stack_num', 1))
    model_type = str(cfg.policy.model.model_type)
    segments, metadata = new_data
    grouped = {}
    for game_segment, meta in zip(segments, metadata):
        metadata_segment_length = meta.get('segment_length', None)
        if metadata_segment_length is None:
            metadata_segment_length = len(game_segment.action_segment)
        valid_steps = min(
            int(metadata_segment_length),
            int(cfg.policy.game_segment_length),
            len(game_segment.action_segment),
            len(game_segment.reward_segment),
        )
        observations = np.asarray(game_segment.obs_segment)
        if valid_steps <= 0 or observations.shape[0] < frame_stack_num + valid_steps:
            continue
        states = []
        for step in range(valid_steps + 1):
            frames = np.asarray(
                observations[step:step + frame_stack_num], dtype=np.float32
            )
            if model_type == 'conv':
                states.append(frames.reshape(-1, *frames.shape[-2:]))
            else:
                states.append(frames.reshape(-1))
        actions = np.asarray(
            game_segment.action_segment[:valid_steps], dtype=np.int64
        ).reshape(-1)
        extrinsic_rewards = np.asarray(
            game_segment.reward_segment[:valid_steps], dtype=np.float32
        ).reshape(valid_steps, -1)[:, 0]
        key = (
            int(meta.get('env_id', -1)),
            int(meta.get('env_episode_index', -1)),
        )
        grouped.setdefault(key, []).append(
            {
                'segment_index': int(meta.get('episode_segment_index', 0)),
                'states': np.stack(states, axis=0),
                'actions': actions,
                'extrinsic_rewards': extrinsic_rewards,
                'done': bool(meta.get('done', False)),
                'collection_episode_index': int(
                    meta.get('collection_episode_index', -1)
                ),
            }
        )

    episodes = []
    for (env_id, _env_episode_index), episode_segments in grouped.items():
        episode_segments.sort(key=lambda item: item['segment_index'])
        segment_indices = [item['segment_index'] for item in episode_segments]
        if (
                not episode_segments[-1]['done']
                or segment_indices != list(range(len(episode_segments)))
        ):
            continue
        states = [episode_segments[0]['states']]
        actions = []
        extrinsic_rewards = []
        for segment_index, segment in enumerate(episode_segments):
            if segment_index > 0:
                states.append(segment['states'][1:])
            actions.append(segment['actions'])
            extrinsic_rewards.append(segment['extrinsic_rewards'])
        observations = np.concatenate(states, axis=0)
        actions_array = np.concatenate(actions, axis=0)
        extrinsic_array = np.concatenate(extrinsic_rewards, axis=0)
        step_count = min(
            len(actions_array),
            len(extrinsic_array),
            max(0, len(observations) - 1),
        )
        if step_count <= 0:
            continue
        observations = observations[:step_count + 1]
        episodes.append(
            {
                'observations': observations,
                'actions': actions_array[:step_count],
                'episodic_rewards': _episodic_sequence_first_visit_rewards(
                    observations
                ),
                'extrinsic_rewards': extrinsic_array[:step_count],
                'env_id': env_id,
                'collection_episode_index': episode_segments[-1][
                    'collection_episode_index'
                ],
            }
        )
    return episodes


def _train_spatial_exploration_tree_from_collection(
        cfg,
        policy,
        new_data: List[Any],
        tb_logger,
        collection_phase: int,
) -> Dict[str, float]:
    if not bool(
            cfg.policy.get('spatial_exploration_tree_online_training', False)
    ):
        return {}
    episodes = _complete_episodic_sequence_training_sequences(cfg, new_data)
    if not episodes:
        return {}
    train_fn = getattr(policy, 'train_spatial_exploration_tree', None)
    if not callable(train_fn):
        return {}
    stats = train_fn(episodes, collection_phase=collection_phase)
    if not stats:
        return {}
    logging.info(
        '[SPATIAL_EXPLORATION_ONLINE_TRAIN] phase %s fresh=%s replay=%s '
        'updates=%s total_updates=%s active_next=%s loss=%.5f '
        'representation=%.5f key=%.5f contrastive=%.5f novelty=%.5f '
        'predictive_equivalence=%.5f pairs=%.1f positive=%.1f negative=%.1f '
        'aliased_negative=%.1f aliased_negative_loss=%.5f '
        'equivalence_active=%.2f '
        'observation_reconstruction=%.5f belief_reconstruction=%.5f '
        'belief_observation_alignment=%.5f '
        'belief_dynamics_consistency=%.5f '
        'pairwise_revisit=%.5f pair_positive_probability=%.3f '
        'pair_negative_probability=%.3f pair_positive=%.1f pair_negative=%.1f '
        'replay_priority=%.5f priority_uniform=%.5f '
        'priority_selected=%.5f priority_stored=%.5f '
        'prioritized_fraction=%.3f priority_cache=%.1f '
        'episode_history=%.2f history_fraction=%.3f history_steps=%.1f '
        'window_start_mean=%.1f window_start_max=%.1f '
        'observation_key_consistency=%.5f key_depth1=%.5f key_depth5=%.5f '
        'rollout_reconstruction=%.5f inverse=%.5f grad_norm=%.4f',
        collection_phase,
        int(stats.get('fresh_episode_count', 0.0)),
        int(stats.get('replay_episode_count', 0.0)),
        int(stats.get('update_count', 0.0)),
        int(stats.get('total_update_count', 0.0)),
        bool(stats.get('active_next_collection', 0.0)),
        float(stats.get('loss', 0.0)),
        float(stats.get('representation_loss', 0.0)),
        float(stats.get('key', 0.0)),
        float(stats.get('contrastive', 0.0)),
        float(stats.get('memory_novelty_mae', 0.0)),
        float(stats.get('predictive_equivalence_loss', 0.0)),
        float(stats.get('predictive_equivalence_pair_count', 0.0)),
        float(stats.get('predictive_equivalence_positive_count', 0.0)),
        float(stats.get('predictive_equivalence_negative_count', 0.0)),
        float(
            stats.get('predictive_equivalence_aliased_negative_count', 0.0)
        ),
        float(
            stats.get('predictive_equivalence_aliased_negative_loss', 0.0)
        ),
        float(stats.get('predictive_equivalence_active', 0.0)),
        float(stats.get('representation_reconstruction_mae', 0.0)),
        float(stats.get('representation_belief_reconstruction_mae', 0.0)),
        float(stats.get('representation_belief_observation_alignment', 0.0)),
        float(stats.get('belief_dynamics_consistency', 0.0)),
        float(stats.get('pairwise_revisit_loss', 0.0)),
        float(stats.get('pairwise_revisit_positive_probability', 0.0)),
        float(stats.get('pairwise_revisit_negative_probability', 0.0)),
        float(stats.get('pairwise_revisit_positive_count', 0.0)),
        float(stats.get('pairwise_revisit_negative_count', 0.0)),
        float(stats.get('replay_priority_mean', 0.0)),
        float(stats.get('replay_priority_uniform_mean', 0.0)),
        float(stats.get('replay_priority_prioritized_mean', 0.0)),
        float(stats.get('replay_priority_prioritized_stored_mean', 0.0)),
        float(stats.get('replay_prioritized_sample_fraction', 0.0)),
        float(stats.get('replay_priority_cache_size', 0.0)),
        float(stats.get('revisit_full_episode_history', 0.0)),
        float(stats.get('revisit_history_fraction', 0.0)),
        float(stats.get('revisit_history_steps', 0.0)),
        float(stats.get('revisit_window_start_mean', 0.0)),
        float(stats.get('revisit_window_start_max', 0.0)),
        float(stats.get('observation_key_consistency', 0.0)),
        float(stats.get('observation_key_consistency_depth_1', 0.0)),
        float(stats.get('observation_key_consistency_depth_5', 0.0)),
        float(stats.get('rollout_reconstruction_mae', 0.0)),
        float(stats.get('representation_inverse', 0.0)),
        float(stats.get('grad_norm_before_clip', 0.0)),
    )
    if tb_logger is not None:
        for key, value in stats.items():
            tb_logger.add_scalar(
                f'spatial_exploration_online/{key}',
                float(value),
                collection_phase,
            )
    return stats


def _train_episodic_sequence_novelty_from_collection(
        cfg,
        policy,
        new_data: List[Any],
        tb_logger,
        collection_phase: int,
) -> Dict[str, float]:
    episodes = _complete_episodic_sequence_training_sequences(cfg, new_data)
    if not episodes:
        return {}
    train_fn = getattr(policy, 'train_episodic_sequence_novelty', None)
    if not callable(train_fn):
        return {}
    for episode in episodes:
        episode['collection_phase'] = int(collection_phase)
    stats = train_fn(episodes, collection_phase=collection_phase)
    if not stats:
        return {}
    logging.info(
        '[EPISODIC_SEQUENCE_TRAIN] phase %s fresh=%s replay=%s transitions=%s '
        'updates=%s joint=%s catchup=%s loss=%.5f bce=%.5f final_bce=%.5f sequence=%.5f brier=%.5f corr=%.4f '
        'distance_loss=%.5f final_distance_loss=%.5f '
        'continuation_loss=%.5f continuation_mae=%.4f continuation_corr=%.4f '
        'future_cos_d1=%.4f future_cos_d5=%.4f final_cos_d1=%.4f final_cos_d5=%.4f inverse_acc=%.4f '
        'pred_distance(novel/revisit)=%.4f/%.4f '
        'memory_scale=%.5f threshold=%.5f temperature=%.5f '
        'reconstruction=%.5f rollout_reconstruction=%.5f '
        'reconstruction_psnr=%.2f rollout_psnr=%.2f '
        'tree_active_next=%s continuation_tree_active_next=%s',
        collection_phase,
        int(stats.get('episodic_sequence/fresh_episode_count', 0.0)),
        int(stats.get('episodic_sequence/replay_episode_count', 0.0)),
        int(stats.get('episodic_sequence/replay_transition_count', 0.0)),
        int(stats.get('episodic_sequence/update_count', 0.0)),
        int(stats.get('episodic_sequence/joint_update_count', 0.0)),
        int(stats.get('episodic_sequence/predictor_only_update_count', 0.0)),
        float(stats.get('episodic_sequence/loss', 0.0)),
        float(stats.get('episodic_sequence/edge_bce', 0.0)),
        float(stats.get('episodic_sequence/edge_bce_last_update', 0.0)),
        float(stats.get('episodic_sequence/sequence_loss', 0.0)),
        float(stats.get('episodic_sequence/edge_brier', 0.0)),
        float(stats.get('episodic_sequence/prediction_correlation', 0.0)),
        float(stats.get('episodic_predictive_memory/distance_loss', 0.0)),
        float(
            stats.get(
                'episodic_predictive_memory/distance_loss_last_update', 0.0
            )
        ),
        float(stats.get('episodic_sequence/continuation_loss', 0.0)),
        float(stats.get('episodic_sequence/continuation_mae', 0.0)),
        float(stats.get('episodic_sequence/continuation_correlation', 0.0)),
        float(stats.get('episodic_predictive_memory/depth_1_future_cosine', 0.0)),
        float(stats.get('episodic_predictive_memory/depth_5_future_cosine', 0.0)),
        float(
            stats.get(
                'episodic_predictive_memory/depth_1_future_cosine_last_update',
                0.0,
            )
        ),
        float(
            stats.get(
                'episodic_predictive_memory/depth_5_future_cosine_last_update',
                0.0,
            )
        ),
        float(stats.get('episodic_predictive_memory/inverse_accuracy', 0.0)),
        float(stats.get('episodic_predictive_memory/predicted_novel_distance', 0.0)),
        float(stats.get('episodic_predictive_memory/predicted_revisit_distance', 0.0)),
        float(stats.get('episodic_predictive_memory/distance_scale', 0.0)),
        float(stats.get('episodic_predictive_memory/distance_threshold', 0.0)),
        float(stats.get('episodic_predictive_memory/distance_temperature', 0.0)),
        float(
            stats.get(
                'episodic_reconstructive_memory/reconstruction_loss', 0.0
            )
        ),
        float(
            stats.get(
                'episodic_reconstructive_memory/rollout_reconstruction_loss',
                0.0,
            )
        ),
        float(
            stats.get('episodic_reconstructive_memory/reconstruction_psnr', 0.0)
        ),
        float(
            stats.get(
                'episodic_reconstructive_memory/rollout_reconstruction_psnr',
                0.0,
            )
        ),
        int(stats.get('episodic_sequence/tree_active_next_phase', 0.0)),
        int(
            stats.get(
                'episodic_sequence/continuation_tree_active_next_phase', 0.0
            )
        ),
    )
    diagnostics_path = os.path.join(
        _diagnostics_dir(cfg), 'episodic_sequence_training.csv'
    )
    diagnostics_row = {'collection_phase': int(collection_phase), **stats}
    write_header = not os.path.exists(diagnostics_path)
    with open(diagnostics_path, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(diagnostics_row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(diagnostics_row)
    if tb_logger is not None:
        for key, value in stats.items():
            tb_logger.add_scalar(key, float(value), collection_phase)
    return stats


def _write_episodic_sequence_novelty_factor_diagnostics(
        cfg,
        policy,
        tb_logger,
        collection_phase: int,
) -> None:
    if not bool(cfg.policy.get('episodic_sequence_novelty', False)):
        return
    consume_fn = getattr(
        policy, 'consume_episodic_sequence_novelty_factor_stats', None
    )
    if not callable(consume_fn):
        return
    rows = [
        {'collection_phase': int(collection_phase), **row}
        for row in consume_fn()
    ]
    csv_path = os.path.join(
        _diagnostics_dir(cfg), 'episodic_sequence_novelty_factors.csv'
    )
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    if tb_logger is not None:
        for row in rows:
            depth = int(row['depth'])
            tb_logger.add_scalar(
                f'episodic_sequence/tree_factor_depth_{depth}_mean',
                float(row['factor_mean']),
                collection_phase,
            )
            tb_logger.add_scalar(
                f'episodic_sequence/tree_factor_depth_{depth}_count',
                float(row['count']),
                collection_phase,
            )
            tb_logger.add_scalar(
                f'episodic_sequence/leaf_factor_depth_{depth}_mean',
                float(row['continuation_factor_mean']),
                collection_phase,
            )
            tb_logger.add_scalar(
                f'episodic_sequence/leaf_factor_depth_{depth}_count',
                float(row['continuation_count']),
                collection_phase,
            )
        tb_logger.add_scalar(
            'episodic_sequence/selected_prediction_mean',
            float(rows[0]['selected_prediction_mean']),
            collection_phase,
        )
        tb_logger.add_scalar(
            'episodic_sequence/selected_novel_fraction',
            float(rows[0]['selected_novel_fraction']),
            collection_phase,
        )
        tb_logger.add_scalar(
            'episodic_sequence/selected_brier',
            float(rows[0]['selected_brier']),
            collection_phase,
        )
        tb_logger.add_scalar(
            'episodic_sequence/selected_accuracy',
            float(rows[0]['selected_accuracy']),
            collection_phase,
        )
    logging.info(
        '[EPISODIC_SEQUENCE_INFERENCE] phase %s mode=%s table_builds=%s '
        'prefixes=%s lookups=%s misses=%s table_seconds=%.3f '
        'cache_builds=%s requests=%s hits=%s path_evals=%s cache_seconds=%.3f',
        collection_phase,
        rows[0]['inference_mode'],
        int(rows[0]['table_build_count']),
        int(rows[0]['table_prefix_count']),
        int(rows[0]['lookup_count']),
        int(rows[0]['miss_count']),
        float(rows[0]['table_build_seconds']),
        int(rows[0]['cache_build_count']),
        int(rows[0]['cache_request_count']),
        int(rows[0]['cache_hit_count']),
        int(rows[0]['cache_path_eval_count']),
        float(rows[0]['cache_build_seconds']),
    )


def _binary_prediction_auc(
        predictions: np.ndarray,
        targets: np.ndarray,
) -> float:
    """Pairwise binary AUROC with half credit for ties."""
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    positive = predictions[targets >= 0.5]
    negative = predictions[targets < 0.5]
    if positive.size == 0 or negative.size == 0:
        return float('nan')
    comparisons = positive[:, None] - negative[None, :]
    return float(
        np.mean((comparisons > 0.0) + 0.5 * (comparisons == 0.0))
    )


def _summarize_episodic_sequence_root_actions(records: List[dict]) -> dict:
    predictions = np.asarray(
        [record['selected_prediction'] for record in records],
        dtype=np.float64,
    )
    targets = np.asarray(
        [record['realized_novelty'] for record in records],
        dtype=np.float64,
    )
    candidate_means = np.asarray(
        [record['candidate_probability_mean'] for record in records],
        dtype=np.float64,
    )
    candidate_maxima = np.asarray(
        [record['candidate_probability_max'] for record in records],
        dtype=np.float64,
    )
    candidate_spreads = np.asarray(
        [record['candidate_probability_spread'] for record in records],
        dtype=np.float64,
    )
    gaps = np.asarray(
        [record['selected_probability_gap_from_max'] for record in records],
        dtype=np.float64,
    )
    ranks = np.asarray(
        [record['selected_rank'] for record in records],
        dtype=np.float64,
    )
    is_argmax = np.asarray(
        [record['selected_is_argmax'] for record in records],
        dtype=np.float64,
    )
    novel_predictions = predictions[targets >= 0.5]
    revisit_predictions = predictions[targets < 0.5]
    novel_mean = float(
        np.mean(novel_predictions) if novel_predictions.size else float('nan')
    )
    revisit_mean = float(
        np.mean(revisit_predictions) if revisit_predictions.size else float('nan')
    )
    selected_visits = np.asarray(
        [record['selected_intrinsic_visit_count'] for record in records],
        dtype=np.float64,
    )
    argmax_visits = np.asarray(
        [record['novelty_argmax_intrinsic_visit_count'] for record in records],
        dtype=np.float64,
    )
    return {
        'root_count': int(len(records)),
        'tree_active': float(np.mean([record['tree_active'] for record in records])),
        'trained_phase': int(max(record['trained_phase'] for record in records)),
        'selected_prediction_mean': float(np.mean(predictions)),
        'selected_novel_fraction': float(np.mean(targets)),
        'selected_brier': float(np.mean((predictions - targets) ** 2)),
        'selected_accuracy': float(
            np.mean((predictions >= 0.5) == (targets >= 0.5))
        ),
        'selected_auc': _binary_prediction_auc(predictions, targets),
        'selected_novel_prediction_mean': novel_mean,
        'selected_revisit_prediction_mean': revisit_mean,
        'selected_novel_revisit_margin': novel_mean - revisit_mean,
        'candidate_probability_mean': float(np.mean(candidate_means)),
        'candidate_probability_max_mean': float(np.mean(candidate_maxima)),
        'candidate_probability_spread_mean': float(np.mean(candidate_spreads)),
        'selected_probability_gap_from_max_mean': float(np.mean(gaps)),
        'selected_rank_mean': float(np.mean(ranks)),
        'selected_is_argmax_fraction': float(np.mean(is_argmax)),
        'selected_rank_2plus_fraction': float(np.mean(ranks >= 2.0)),
        'selected_rank_3plus_fraction': float(np.mean(ranks >= 3.0)),
        'all_candidates_below_half_fraction': float(
            np.mean(candidate_maxima < 0.5)
        ),
        'all_candidates_below_0p2_fraction': float(
            np.mean(candidate_maxima < 0.2)
        ),
        'selected_intrinsic_visit_count_mean': float(np.mean(selected_visits)),
        'novelty_argmax_intrinsic_visit_count_mean': float(
            np.mean(argmax_visits)
        ),
        'novelty_argmax_unvisited_fraction': float(np.mean(argmax_visits <= 0.0)),
    }


def _write_episodic_sequence_root_action_diagnostics(
        cfg,
        policy,
        tb_logger,
        collection_phase: int,
) -> None:
    if not bool(cfg.policy.get('episodic_sequence_novelty', False)):
        return
    consume_fn = getattr(
        policy, 'consume_episodic_sequence_root_action_records', None
    )
    if not callable(consume_fn):
        return
    records = consume_fn()
    if not records:
        return

    detailed_rows = []
    list_fields = {
        'legal_actions',
        'candidate_probabilities',
        'candidate_factors',
        'intrinsic_visit_counts',
        'intrinsic_total_scores',
        'intrinsic_immediate_rewards',
    }
    for decision_index, record in enumerate(records):
        row = {
            'collection_phase': int(collection_phase),
            'decision_index': int(decision_index),
        }
        row.update(
            {
                key: json.dumps(value) if key in list_fields else value
                for key, value in record.items()
            }
        )
        detailed_rows.append(row)

    diagnostics_dir = _diagnostics_dir(cfg)
    detailed_path = os.path.join(
        diagnostics_dir, 'episodic_sequence_root_actions.csv'
    )
    write_header = not os.path.exists(detailed_path)
    with open(detailed_path, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(detailed_rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(detailed_rows)

    summary = {
        'collection_phase': int(collection_phase),
        **_summarize_episodic_sequence_root_actions(records),
    }
    summary_path = os.path.join(
        diagnostics_dir, 'episodic_sequence_root_action_summary.csv'
    )
    write_header = not os.path.exists(summary_path)
    with open(summary_path, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(summary.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(summary)

    logging.info(
        '[EPISODIC_SEQUENCE_ROOT_ACTIONS] phase %s roots=%s selected_argmax=%.3f '
        'rank=%.3f spread=%.4f gap=%.4f selected_auc=%.3f '
        'all_below_half=%.3f',
        collection_phase,
        int(summary['root_count']),
        float(summary['selected_is_argmax_fraction']),
        float(summary['selected_rank_mean']),
        float(summary['candidate_probability_spread_mean']),
        float(summary['selected_probability_gap_from_max_mean']),
        float(summary['selected_auc']),
        float(summary['all_candidates_below_half_fraction']),
    )
    if tb_logger is not None:
        for key, value in summary.items():
            if key == 'collection_phase' or not np.isfinite(float(value)):
                continue
            tb_logger.add_scalar(
                f'episodic_sequence_root_actions/{key}',
                float(value),
                collection_phase,
            )


def _write_spatial_exploration_tree_diagnostics(
        cfg,
        policy,
        tb_logger,
        collection_phase: int,
) -> None:
    if not bool(cfg.policy.get('spatial_exploration_tree', False)):
        return
    consume_fn = getattr(policy, 'consume_spatial_exploration_tree_records', None)
    if not callable(consume_fn):
        return
    records = consume_fn()
    if not records:
        return
    diagnostics_dir = _diagnostics_dir(cfg)
    detailed_rows = [
        {
            'collection_phase': int(collection_phase),
            'decision_index': int(index),
            **record,
        }
        for index, record in enumerate(records)
    ]
    detailed_path = os.path.join(
        diagnostics_dir, 'spatial_exploration_tree_actions.csv'
    )
    write_header = not os.path.exists(detailed_path)
    with open(detailed_path, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(detailed_rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(detailed_rows)

    predicted = np.asarray(
        [record['predicted_novelty'] for record in records], dtype=np.float64
    )
    realized = np.asarray(
        [record['realized_novelty'] for record in records], dtype=np.float64
    )
    correlation = float('nan')
    if (
            predicted.size > 1
            and float(predicted.std()) > 1e-12
            and float(realized.std()) > 1e-12
    ):
        correlation = float(np.corrcoef(predicted, realized)[0, 1])
    summary = {
        'collection_phase': int(collection_phase),
        'reward_mode': str(records[0].get('reward_mode', 'bonus')),
        'decision_count': int(len(records)),
        'predicted_novelty_mean': float(predicted.mean()),
        'realized_novelty_mean': float(realized.mean()),
        'novelty_mae': float(np.mean(np.abs(predicted - realized))),
        'novelty_correlation': correlation,
        'realized_rgb_novel_fraction': float(
            np.mean([record['realized_rgb_novel'] for record in records])
        ),
        'selected_bonus_mean': float(
            np.mean([record['selected_bonus'] for record in records])
        ),
        'candidate_bonus_spread_mean': float(
            np.mean([record['candidate_bonus_spread'] for record in records])
        ),
        'selected_reward_factor_mean': float(
            np.mean([
                record.get('selected_reward_factor', 1.0)
                for record in records
            ])
        ),
        'candidate_reward_factor_spread_mean': float(
            np.mean([
                record.get('candidate_reward_factor_spread', 0.0)
                for record in records
            ])
        ),
        'selected_adjustment_mean': float(
            np.mean([
                record.get('selected_adjustment', record['selected_bonus'])
                for record in records
            ])
        ),
        'candidate_adjustment_spread_mean': float(
            np.mean([
                record.get(
                    'candidate_adjustment_spread',
                    record['candidate_bonus_spread'],
                )
                for record in records
            ])
        ),
        'candidate_novelty_spread_mean': float(
            np.mean([record['candidate_novelty_spread'] for record in records])
        ),
        'selected_bonus_rank_mean': float(
            np.mean([record['selected_bonus_rank'] for record in records])
        ),
        'selected_bonus_is_argmax_fraction': float(
            np.mean([record['selected_bonus_is_argmax'] for record in records])
        ),
        'selected_adjustment_rank_mean': float(
            np.mean([
                record.get(
                    'selected_adjustment_rank', record['selected_bonus_rank']
                )
                for record in records
            ])
        ),
        'selected_adjustment_is_argmax_fraction': float(
            np.mean([
                record.get(
                    'selected_adjustment_is_argmax',
                    record['selected_bonus_is_argmax'],
                )
                for record in records
            ])
        ),
        'search_intrinsic_reward_before_transform_mean': float(
            np.mean([
                record.get(
                    'search_intrinsic_reward_before_transform', float('nan')
                )
                for record in records
            ])
        ),
        'search_intrinsic_reward_before_modifier_mean': float(
            np.mean([
                record.get(
                    'search_intrinsic_reward_before_modifier', float('nan')
                )
                for record in records
            ])
        ),
        'search_intrinsic_immediate_reward_mean': float(
            np.mean([
                record.get('search_intrinsic_immediate_reward', float('nan'))
                for record in records
            ])
        ),
        'search_intrinsic_reward_return_mean': float(
            np.mean([
                record.get('search_intrinsic_reward_return_mean', float('nan'))
                for record in records
            ])
        ),
        'search_intrinsic_leaf_value_contribution_mean': float(
            np.mean([
                record.get(
                    'search_intrinsic_leaf_value_contribution_mean', float('nan')
                )
                for record in records
            ])
        ),
        'memory_scale_mean': float(
            np.mean([record['memory_scale'] for record in records])
        ),
    }
    summary_path = os.path.join(
        diagnostics_dir, 'spatial_exploration_tree_summary.csv'
    )
    write_header = not os.path.exists(summary_path)
    with open(summary_path, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(summary.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(summary)
    logging.info(
        '[SPATIAL_EXPLORATION_TREE] phase %s decisions=%s predicted=%.4f '
        'realized=%.4f mae=%.4f corr=%.3f rgb_novel=%.3f mode=%s '
        'adjustment=%.4f reward_factor=%.4f selected_argmax=%.3f '
        'head_edge=%.4f estimated_edge=%.4f edge_after=%.4f '
        'path_return=%.4f leaf=%.4f',
        collection_phase,
        int(summary['decision_count']),
        float(summary['predicted_novelty_mean']),
        float(summary['realized_novelty_mean']),
        float(summary['novelty_mae']),
        float(summary['novelty_correlation']),
        float(summary['realized_rgb_novel_fraction']),
        summary['reward_mode'],
        float(summary['selected_adjustment_mean']),
        float(summary['selected_reward_factor_mean']),
        float(summary['selected_adjustment_is_argmax_fraction']),
        float(summary['search_intrinsic_reward_before_transform_mean']),
        float(summary['search_intrinsic_reward_before_modifier_mean']),
        float(summary['search_intrinsic_immediate_reward_mean']),
        float(summary['search_intrinsic_reward_return_mean']),
        float(summary['search_intrinsic_leaf_value_contribution_mean']),
    )
    if tb_logger is not None:
        for key, value in summary.items():
            if key in {'collection_phase', 'reward_mode'}:
                continue
            if not np.isfinite(float(value)):
                continue
            tb_logger.add_scalar(
                f'spatial_exploration_tree/{key}',
                float(value),
                collection_phase,
            )


def _train_episodic_history_shadow_from_collection(
        cfg,
        policy,
        new_data: List[Any],
        tb_logger,
        collection_phase: int,
) -> Dict[str, float]:
    episodes = _complete_episodic_history_training_sequences(cfg, new_data)
    if not episodes:
        return {}
    train_fn = getattr(policy, 'train_episodic_history_shadow', None)
    if not callable(train_fn):
        return {}
    for episode in episodes:
        episode['collection_phase'] = int(collection_phase)
    stats = train_fn(episodes, collection_phase=collection_phase)
    if not stats:
        return {}
    logging.info(
        "[EPISODIC_NOVELTY_HISTORY_TRAIN] phase %s | fresh episodes %s transitions %s, "
        "replay %s episodes (%s recent/%s historical), updates %s, sampled %s episodes "
        "(recent %.2f, current phase %.2f, unique %s, recent reuse %.2f [%s,%s]), "
        "loss %.5f, prediction %.4f target %.4f, "
        "MAE %.4f corr %.4f, hidden_norm %.4f, grad_norm %.4f",
        collection_phase,
        int(stats.get('episodic_history/episode_count', 0.0)),
        int(stats.get('episodic_history/transition_count', 0.0)),
        int(stats.get('episodic_history/replay_episode_count', 0.0)),
        int(stats.get('episodic_history/recent_pool_episode_count', 0.0)),
        int(stats.get('episodic_history/historical_pool_episode_count', 0.0)),
        int(stats.get('episodic_history/update_count', 0.0)),
        int(stats.get('episodic_history/sampled_episode_count', 0.0)),
        float(stats.get('episodic_history/recent_sample_fraction', 0.0)),
        float(stats.get('episodic_history/current_phase_sample_fraction', 0.0)),
        int(stats.get('episodic_history/sampled_unique_episode_count', 0.0)),
        float(stats.get('episodic_history/recent_sample_reuse_mean', 0.0)),
        int(stats.get('episodic_history/recent_sample_reuse_min', 0.0)),
        int(stats.get('episodic_history/recent_sample_reuse_max', 0.0)),
        float(stats.get('episodic_history/loss', 0.0)),
        float(stats.get('episodic_history/prediction_mean', 0.0)),
        float(stats.get('episodic_history/target_mean', 0.0)),
        float(stats.get('episodic_history/prediction_mae', 0.0)),
        float(stats.get('episodic_history/prediction_correlation', 0.0)),
        float(stats.get('episodic_history/hidden_norm_mean', 0.0)),
        float(stats.get('episodic_history/grad_norm_before_clip', 0.0)),
    )
    if tb_logger is not None:
        for key, value in stats.items():
            tb_logger.add_scalar(key, float(value), collection_phase)
    return stats


def _write_main_tree_trace_diagnostics(cfg, new_data: List[Any], tb_logger, collection_phase: int) -> None:
    _write_tree_trace_diagnostics(
        cfg,
        new_data,
        tb_logger,
        collection_phase,
        enabled_key='main_tree_trace_diagnostics',
        segment_attr='main_tree_trace_segment',
        output_filename='main_tree_traces.jsonl',
        log_prefix='MAIN_TREE_TRACE_DIAG',
        tb_key='collection_phase/main_tree_trace_count',
    )


def _write_intrinsic_tree_trace_diagnostics(cfg, new_data: List[Any], tb_logger, collection_phase: int) -> None:
    _write_tree_trace_diagnostics(
        cfg,
        new_data,
        tb_logger,
        collection_phase,
        enabled_key='intrinsic_tree_trace_diagnostics',
        segment_attr='intrinsic_tree_trace_segment',
        output_filename='intrinsic_tree_traces.jsonl',
        log_prefix='INTRINSIC_TREE_TRACE_DIAG',
        tb_key='collection_phase/intrinsic_tree_trace_count',
    )


def _parse_fixed_sequence_tokens(sequence_spec: str) -> List[int]:
    actions = []
    for raw_token in sequence_spec.split(','):
        token = raw_token.strip()
        if not token:
            continue
        if 'x' in token:
            action_token, repeat_token = token.split('x', 1)
            actions.extend([int(action_token.strip())] * int(repeat_token.strip()))
        else:
            actions.append(int(token))
    return actions


def _default_fixed_intrinsic_sequences() -> List[Tuple[str, List[int]]]:
    return [
        ('left5', [0] * 5),
        ('right5', [2] * 5),
        ('noop5', [1] * 5),
        ('dither5', [1, 1, 0, 2, 1]),
        ('left10', [0] * 10),
        ('right10', [2] * 10),
        ('left10_right10', [0] * 10 + [2] * 10),
        ('right10_left10', [2] * 10 + [0] * 10),
    ]


def _fixed_intrinsic_sequences(cfg) -> List[Tuple[str, List[int]]]:
    sequence_config = cfg.policy.get('fixed_sequence_intrinsic_sequences', None)
    if not sequence_config:
        return _default_fixed_intrinsic_sequences()
    if isinstance(sequence_config, str):
        sequences = []
        for raw_entry in sequence_config.split(';'):
            entry = raw_entry.strip()
            if not entry:
                continue
            if ':' not in entry:
                raise ValueError(
                    "fixed_sequence_intrinsic_sequences string entries must look like "
                    "'name:0,0,2' or 'name:0x5,2x5'."
                )
            name, sequence_spec = entry.split(':', 1)
            sequences.append((name.strip(), _parse_fixed_sequence_tokens(sequence_spec)))
        return sequences
    sequences = []
    for item in sequence_config:
        if isinstance(item, dict):
            name = str(item['name'])
            actions = [int(action) for action in item['actions']]
        else:
            name, actions = item
            name = str(name)
            actions = [int(action) for action in actions]
        sequences.append((name, actions))
    return sequences


def _sample_fixed_intrinsic_states(cfg, new_data: List[Any]) -> List[Dict[str, Any]]:
    candidates = []
    data, meta_list = new_data
    for segment_index, (game_segment, meta) in enumerate(zip(data, meta_list)):
        valid_step_count = int(meta.get('segment_length', min(len(game_segment), int(cfg.policy.game_segment_length))))
        real_step_count = min(valid_step_count, len(game_segment.action_segment))
        if real_step_count <= 0:
            continue
        obs_segment = np.asarray(game_segment.obs_segment)
        obs_count = _real_observation_count(cfg, valid_step_count, len(obs_segment))
        for timestep in range(real_step_count):
            obs_index = _decision_obs_index(cfg, timestep, obs_count)
            if obs_index >= len(obs_segment):
                continue
            obs = np.asarray(obs_segment[obs_index], dtype=np.float32)
            position, velocity = _state_components(obs)
            selected_action = int(game_segment.action_segment[timestep]) if timestep < len(
                game_segment.action_segment
            ) else -1
            candidates.append(
                {
                    'obs': obs,
                    'obs_index': int(obs_index),
                    'collection_phase': meta.get('collection_phase'),
                    'train_iter_at_collect': meta.get('train_iter_at_collect'),
                    'envstep_before_collect': meta.get('envstep_before_collect'),
                    'envstep_after_collect': meta.get('envstep_after_collect'),
                    'segment_index_in_collect': meta.get('segment_index_in_collect', segment_index),
                    'env_id': meta.get('env_id'),
                    'env_episode_index': meta.get('env_episode_index'),
                    'collection_episode_index': meta.get('collection_episode_index'),
                    'episode_segment_index': meta.get('episode_segment_index'),
                    'episode_return': meta.get('episode_return'),
                    'video_path': meta.get('video_path'),
                    'timestep': timestep,
                    'selected_action': selected_action,
                    'selected_action_direction': _action_direction(selected_action),
                    'obs_position': position,
                    'obs_velocity': velocity,
                    **_future_trajectory_metrics(cfg, game_segment, timestep, real_step_count),
                }
            )

    max_states = int(cfg.policy.get('fixed_sequence_intrinsic_max_states', 64))
    if max_states <= 0 or len(candidates) <= max_states:
        return candidates
    sampled_indices = np.linspace(0, len(candidates) - 1, num=max_states, dtype=np.int64)
    return [candidates[int(index)] for index in sampled_indices]


def _intrinsic_reward_source(cfg) -> str:
    source = str(cfg.policy.get('intrinsic_reward_source', 'rnd') or 'rnd')
    if source not in [
            'rnd', 'dynamics_disagreement', 'qstd', 'state_value_disagreement',
            'episodic_novelty', 'state_density'
    ]:
        raise ValueError(f"Unsupported intrinsic_reward_source: {source}")
    return source


def _restore_model_member_state(model: torch.nn.Module, active_member_state: Any) -> None:
    if active_member_state is None:
        return
    if isinstance(active_member_state, list):
        if hasattr(model, 'set_active_members'):
            model.set_active_members(active_member_state)
    elif hasattr(model, 'set_active_member'):
        model.set_active_member(int(active_member_state))


def _mean_pairwise_latent_l2(member_latents: torch.Tensor) -> torch.Tensor:
    """Per-state mean L2 distance over all member pairs."""
    pair_distances = []
    for first in range(int(member_latents.shape[0])):
        for second in range(first + 1, int(member_latents.shape[0])):
            pair_distances.append(
                (member_latents[first] - member_latents[second]).reshape(member_latents.shape[1], -1).norm(
                    p=2, dim=1
                )
            )
    if not pair_distances:
        return member_latents.new_zeros(member_latents.shape[1])
    return torch.stack(pair_distances, dim=0).mean(dim=0)


def _encoder_prior_grid(cfg) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    position_count = int(cfg.policy.get('encoder_prior_grid_position_count', 37))
    velocity_count = int(cfg.policy.get('encoder_prior_grid_velocity_count', 29))
    if position_count < 2 or velocity_count < 2:
        raise ValueError("Encoder-prior grid axis counts must both be at least two.")
    positions = np.linspace(
        float(cfg.policy.get('encoder_prior_grid_position_min', -1.2)),
        float(cfg.policy.get('encoder_prior_grid_position_max', 0.6)),
        num=position_count,
        dtype=np.float32,
    )
    velocities = np.linspace(
        float(cfg.policy.get('encoder_prior_grid_velocity_min', -0.07)),
        float(cfg.policy.get('encoder_prior_grid_velocity_max', 0.07)),
        num=velocity_count,
        dtype=np.float32,
    )
    position_grid, velocity_grid = np.meshgrid(positions, velocities, indexing='ij')
    observations = np.stack([position_grid.reshape(-1), velocity_grid.reshape(-1)], axis=1)
    return positions, velocities, observations


def _encoder_prior_grid_visit_counts(
        cfg,
        new_data: List[Any],
        collection_phase: int,
        positions: np.ndarray,
        velocities: np.ndarray,
        update: bool,
) -> np.ndarray:
    diagnostics_dir = _diagnostics_dir(cfg)
    counts_path = os.path.join(diagnostics_dir, 'encoder_prior_grid_visit_counts.npz')
    expected_shape = (positions.size, velocities.size)
    counts = np.zeros(expected_shape, dtype=np.int64)
    last_phase = -1
    if os.path.exists(counts_path):
        try:
            with np.load(counts_path, allow_pickle=False) as saved:
                saved_counts = np.asarray(saved['counts'], dtype=np.int64)
                saved_positions = np.asarray(saved['positions'], dtype=np.float32)
                saved_velocities = np.asarray(saved['velocities'], dtype=np.float32)
                if (
                        saved_counts.shape == expected_shape
                        and np.allclose(saved_positions, positions)
                        and np.allclose(saved_velocities, velocities)
                ):
                    counts = saved_counts
                    last_phase = int(np.asarray(saved['last_phase']).reshape(-1)[0])
        except (OSError, ValueError, KeyError):
            logging.warning('[ENCODER_PRIOR_GRID] could not read %s; rebuilding visit counts', counts_path)

    if update and int(collection_phase) > last_phase:
        observations = _fresh_collection_next_observations(cfg, new_data)
        if observations.ndim == 2 and observations.shape[0] > 0 and observations.shape[1] >= 2:
            position_indices = np.rint(
                (observations[:, 0] - positions[0])
                / max(float(positions[-1] - positions[0]), 1e-12)
                * (positions.size - 1)
            ).astype(np.int64)
            velocity_indices = np.rint(
                (observations[:, 1] - velocities[0])
                / max(float(velocities[-1] - velocities[0]), 1e-12)
                * (velocities.size - 1)
            ).astype(np.int64)
            valid = (
                np.isfinite(observations[:, 0])
                & np.isfinite(observations[:, 1])
                & (position_indices >= 0)
                & (position_indices < positions.size)
                & (velocity_indices >= 0)
                & (velocity_indices < velocities.size)
            )
            np.add.at(counts, (position_indices[valid], velocity_indices[valid]), 1)
        np.savez_compressed(
            counts_path,
            counts=counts,
            positions=positions,
            velocities=velocities,
            last_phase=np.asarray([collection_phase], dtype=np.int64),
        )
    return counts.reshape(-1)


def _write_encoder_prior_grid_diagnostics(
        cfg,
        policy,
        new_data: List[Any],
        tb_logger,
        collection_phase: int,
        train_iter: int,
        diagnostic_stage: str,
        update_visits: bool,
) -> None:
    if not _diagnostics_enabled(cfg) or not bool(cfg.policy.get('encoder_prior_grid_diagnostics', False)):
        return
    if int(cfg.policy.model.get('observation_shape', 0)) != 2:
        logging.warning('[ENCODER_PRIOR_GRID] skipped: the fixed MountainCar probe requires observation_shape=2')
        return

    model = getattr(policy, '_collect_model', None)
    if model is None:
        model = getattr(policy, '_learn_model', None)
    prior_networks = getattr(model, 'encoder_prior_networks', None) if model is not None else None
    if model is None or prior_networks is None:
        logging.warning('[ENCODER_PRIOR_GRID] skipped: encoder random priors are not enabled')
        return

    positions, velocities, observations = _encoder_prior_grid(cfg)
    visit_counts = _encoder_prior_grid_visit_counts(
        cfg, new_data, collection_phase, positions, velocities, update=update_visits
    )
    device = torch.device(cfg.policy.device)
    obs_batch = torch.as_tensor(observations, dtype=torch.float32, device=device)
    state_count = int(obs_batch.shape[0])
    ensemble_size = int(cfg.policy.model.get('ensemble_size', 1))
    member_ids = [member_id for member_id in range(ensemble_size) for _ in range(state_count)]
    repeated_obs = torch.cat([obs_batch for _ in range(ensemble_size)], dim=0)
    value_offsets = np.zeros(ensemble_size, dtype=np.float32)
    calibration_fn = getattr(policy, '_value_calibration_offsets', None)
    if callable(calibration_fn):
        value_offsets = np.asarray(calibration_fn(ensemble_size), dtype=np.float32)
    offset_tensor = torch.as_tensor(value_offsets, dtype=torch.float32, device=device).view(ensemble_size, 1)
    discount_factor = float(cfg.policy.discount_factor)

    was_training = model.training
    active_member_state = getattr(model, 'active_member_id', None)
    model.eval()
    rows = []
    try:
        with torch.no_grad():
            learned_latents = []
            prior_latents = []
            full_latents = []
            for member_id in range(ensemble_size):
                learned = model.members[member_id]._representation(obs_batch)
                learned = model._normalize_encoder_latent(learned)
                prior = prior_networks[member_id](obs_batch).to(dtype=learned.dtype, device=learned.device)
                prior = float(model.encoder_random_prior_scale) * prior
                learned_latents.append(learned)
                prior_latents.append(prior)
                full_latents.append(model._normalize_composed_encoder_latent(learned + prior))
            learned_latents = torch.stack(learned_latents, dim=0)
            prior_latents = torch.stack(prior_latents, dim=0)
            full_latents = torch.stack(full_latents, dim=0)

            root_output = model.initial_inference(repeated_obs, member_id=member_ids)
            actual_full_latents = root_output.latent_state.reshape(ensemble_size, state_count, -1)
            composition_error = (actual_full_latents - full_latents).abs().amax().item()
            root_values = policy.value_inverse_scalar_transform_handle(root_output.value).detach()
            root_values = root_values.reshape(ensemble_size, state_count, -1).squeeze(-1).float()
            root_trainable_values = []
            for member_id in range(ensemble_size):
                _, trainable_value_logits = model.members[member_id]._prediction(
                    actual_full_latents[member_id]
                )
                root_trainable_values.append(
                    policy.value_inverse_scalar_transform_handle(trainable_value_logits).detach()
                )
            root_trainable_values = torch.stack(root_trainable_values, dim=0).squeeze(-1).float()
            root_value_prior_effects = root_values - root_trainable_values
            calibrated_root_values = root_values - offset_tensor

            learned_norm = learned_latents.reshape(ensemble_size, state_count, -1).norm(p=2, dim=2).mean(dim=0)
            prior_norm = prior_latents.reshape(ensemble_size, state_count, -1).norm(p=2, dim=2).mean(dim=0)
            full_norm = full_latents.reshape(ensemble_size, state_count, -1).norm(p=2, dim=2).mean(dim=0)
            learned_pair_l2 = _mean_pairwise_latent_l2(learned_latents)
            prior_pair_l2 = _mean_pairwise_latent_l2(prior_latents)
            full_pair_l2 = _mean_pairwise_latent_l2(full_latents)

            action_metrics = {}
            for action in range(int(model.action_space_size)):
                action_batch = torch.full(
                    (ensemble_size * state_count, 1), action, dtype=torch.long, device=device
                )
                recurrent_output = model.recurrent_inference(
                    actual_full_latents.reshape(ensemble_size * state_count, -1),
                    action_batch,
                    member_id=member_ids,
                )
                recurrent_values = policy.value_inverse_scalar_transform_handle(recurrent_output.value).detach()
                recurrent_values = recurrent_values.reshape(ensemble_size, state_count, -1).squeeze(-1).float()
                recurrent_rewards = policy.reward_inverse_scalar_transform_handle(recurrent_output.reward).detach()
                recurrent_rewards = recurrent_rewards.reshape(ensemble_size, state_count, -1).squeeze(-1).float()
                recurrent_latents = recurrent_output.latent_state.reshape(ensemble_size, state_count, -1)
                recurrent_trainable_values = []
                for member_id in range(ensemble_size):
                    _, trainable_value_logits = model.members[member_id]._prediction(
                        recurrent_latents[member_id]
                    )
                    recurrent_trainable_values.append(
                        policy.value_inverse_scalar_transform_handle(trainable_value_logits).detach()
                    )
                recurrent_trainable_values = torch.stack(
                    recurrent_trainable_values, dim=0
                ).squeeze(-1).float()
                imagined_q = recurrent_rewards + discount_factor * recurrent_values
                imagined_trainable_q = recurrent_rewards + discount_factor * recurrent_trainable_values
                imagined_value_prior_effect = imagined_q - imagined_trainable_q
                imagined_calibrated_q = recurrent_rewards + discount_factor * (recurrent_values - offset_tensor)

                true_next_obs = np.stack(
                    [
                        _mountain_car_sequence_next_observations(float(position), float(velocity), [action])[-1]
                        for position, velocity in observations
                    ],
                    axis=0,
                ).astype(np.float32)
                true_next_batch = torch.as_tensor(true_next_obs, dtype=torch.float32, device=device)
                repeated_true_next = torch.cat([true_next_batch for _ in range(ensemble_size)], dim=0)
                direct_output = model.initial_inference(repeated_true_next, member_id=member_ids)
                direct_latents = direct_output.latent_state.reshape(ensemble_size, state_count, -1)
                direct_values = policy.value_inverse_scalar_transform_handle(direct_output.value).detach()
                direct_values = direct_values.reshape(ensemble_size, state_count, -1).squeeze(-1).float()
                direct_trainable_values = []
                for member_id in range(ensemble_size):
                    _, trainable_value_logits = model.members[member_id]._prediction(
                        direct_latents[member_id]
                    )
                    direct_trainable_values.append(
                        policy.value_inverse_scalar_transform_handle(trainable_value_logits).detach()
                    )
                direct_trainable_values = torch.stack(
                    direct_trainable_values, dim=0
                ).squeeze(-1).float()
                terminal = torch.as_tensor(true_next_obs[:, 0] >= 0.5, device=device).view(1, state_count)
                direct_q = -torch.ones_like(direct_values) + discount_factor * direct_values.masked_fill(terminal, 0.0)
                direct_trainable_q = -torch.ones_like(direct_trainable_values) + discount_factor * (
                    direct_trainable_values.masked_fill(terminal, 0.0)
                )
                direct_value_prior_effect = direct_q - direct_trainable_q
                direct_calibrated_q = -torch.ones_like(direct_values) + discount_factor * (
                    direct_values - offset_tensor
                ).masked_fill(terminal, 0.0)
                latent_error = (recurrent_latents - direct_latents).norm(p=2, dim=2).mean(dim=0)

                action_metrics[action] = {
                    'imagined_q': imagined_q,
                    'imagined_qstd': imagined_q.std(dim=0, unbiased=False),
                    'imagined_trainable_qstd': imagined_trainable_q.std(dim=0, unbiased=False),
                    'imagined_value_prior_effect_std': imagined_value_prior_effect.std(dim=0, unbiased=False),
                    'imagined_calibrated_qstd': imagined_calibrated_q.std(dim=0, unbiased=False),
                    'direct_q': direct_q,
                    'direct_qstd': direct_q.std(dim=0, unbiased=False),
                    'direct_trainable_qstd': direct_trainable_q.std(dim=0, unbiased=False),
                    'direct_value_prior_effect_std': direct_value_prior_effect.std(dim=0, unbiased=False),
                    'direct_calibrated_qstd': direct_calibrated_q.std(dim=0, unbiased=False),
                    'imagined_direct_q_mae': (imagined_q - direct_q).abs().mean(dim=0),
                    'latent_error': latent_error,
                }

            for state_index, (position, velocity) in enumerate(observations):
                row = {
                    'collection_phase': int(collection_phase),
                    'diagnostic_stage': str(diagnostic_stage),
                    'train_iter': int(train_iter),
                    'position': float(position),
                    'velocity': float(velocity),
                    'visit_count': int(visit_counts[state_index]),
                    'learned_latent_norm_mean': float(learned_norm[state_index].item()),
                    'prior_contribution_norm_mean': float(prior_norm[state_index].item()),
                    'full_latent_norm_mean': float(full_norm[state_index].item()),
                    'learned_latent_pair_l2': float(learned_pair_l2[state_index].item()),
                    'prior_contribution_pair_l2': float(prior_pair_l2[state_index].item()),
                    'full_latent_pair_l2': float(full_pair_l2[state_index].item()),
                    'root_member_values': json.dumps(root_values[:, state_index].cpu().tolist()),
                    'root_value_std': float(root_values[:, state_index].std(unbiased=False).item()),
                    'root_member_trainable_values': json.dumps(
                        root_trainable_values[:, state_index].cpu().tolist()
                    ),
                    'root_trainable_value_std': float(
                        root_trainable_values[:, state_index].std(unbiased=False).item()
                    ),
                    'root_member_value_prior_effects': json.dumps(
                        root_value_prior_effects[:, state_index].cpu().tolist()
                    ),
                    'root_value_prior_effect_std': float(
                        root_value_prior_effects[:, state_index].std(unbiased=False).item()
                    ),
                    'root_calibrated_value_std': float(
                        calibrated_root_values[:, state_index].std(unbiased=False).item()
                    ),
                    'composition_abs_max_error': float(composition_error),
                }
                imagined_qstds = []
                direct_qstds = []
                imagined_prior_effect_stds = []
                direct_prior_effect_stds = []
                for action, metrics in action_metrics.items():
                    row[f'action_{action}_imagined_member_q'] = json.dumps(
                        metrics['imagined_q'][:, state_index].cpu().tolist()
                    )
                    row[f'action_{action}_direct_member_q'] = json.dumps(
                        metrics['direct_q'][:, state_index].cpu().tolist()
                    )
                    row[f'action_{action}_imagined_qstd'] = float(metrics['imagined_qstd'][state_index].item())
                    row[f'action_{action}_imagined_trainable_qstd'] = float(
                        metrics['imagined_trainable_qstd'][state_index].item()
                    )
                    row[f'action_{action}_imagined_value_prior_effect_std'] = float(
                        metrics['imagined_value_prior_effect_std'][state_index].item()
                    )
                    row[f'action_{action}_imagined_calibrated_qstd'] = float(
                        metrics['imagined_calibrated_qstd'][state_index].item()
                    )
                    row[f'action_{action}_direct_qstd'] = float(metrics['direct_qstd'][state_index].item())
                    row[f'action_{action}_direct_trainable_qstd'] = float(
                        metrics['direct_trainable_qstd'][state_index].item()
                    )
                    row[f'action_{action}_direct_value_prior_effect_std'] = float(
                        metrics['direct_value_prior_effect_std'][state_index].item()
                    )
                    row[f'action_{action}_direct_calibrated_qstd'] = float(
                        metrics['direct_calibrated_qstd'][state_index].item()
                    )
                    row[f'action_{action}_imagined_direct_q_mae'] = float(
                        metrics['imagined_direct_q_mae'][state_index].item()
                    )
                    row[f'action_{action}_recurrent_direct_latent_l2'] = float(
                        metrics['latent_error'][state_index].item()
                    )
                    imagined_qstds.append(row[f'action_{action}_imagined_calibrated_qstd'])
                    direct_qstds.append(row[f'action_{action}_direct_calibrated_qstd'])
                    imagined_prior_effect_stds.append(
                        row[f'action_{action}_imagined_value_prior_effect_std']
                    )
                    direct_prior_effect_stds.append(
                        row[f'action_{action}_direct_value_prior_effect_std']
                    )
                row['max_action_imagined_calibrated_qstd'] = float(max(imagined_qstds))
                row['max_action_direct_calibrated_qstd'] = float(max(direct_qstds))
                row['max_action_imagined_value_prior_effect_std'] = float(
                    max(imagined_prior_effect_stds)
                )
                row['max_action_direct_value_prior_effect_std'] = float(max(direct_prior_effect_stds))
                rows.append(row)
    finally:
        _restore_model_member_state(model, active_member_state)
        model.train(was_training)

    diagnostics_dir = _diagnostics_dir(cfg)
    csv_path = os.path.join(diagnostics_dir, 'encoder_prior_state_grid.csv')
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    visited = np.asarray([row['visit_count'] > 0 for row in rows], dtype=bool)
    full_l2 = np.asarray([row['full_latent_pair_l2'] for row in rows], dtype=np.float64)
    imagined_qstd = np.asarray(
        [row['max_action_imagined_calibrated_qstd'] for row in rows], dtype=np.float64
    )
    direct_qstd = np.asarray([row['max_action_direct_calibrated_qstd'] for row in rows], dtype=np.float64)
    direct_prior_effect_std = np.asarray(
        [row['max_action_direct_value_prior_effect_std'] for row in rows], dtype=np.float64
    )
    logging.info(
        '[ENCODER_PRIOR_GRID] phase %s %s states %s visited %s | full_pair_l2 visited %.4f unseen %.4f | '
        'max_qstd imagined visited %.3f unseen %.3f, direct visited %.3f unseen %.3f | '
        'direct_prior_effect_std visited %.3f unseen %.3f | composition_error %.2e',
        collection_phase,
        diagnostic_stage,
        len(rows),
        int(visited.sum()),
        float(full_l2[visited].mean()) if visited.any() else 0.0,
        float(full_l2[~visited].mean()) if (~visited).any() else 0.0,
        float(imagined_qstd[visited].mean()) if visited.any() else 0.0,
        float(imagined_qstd[~visited].mean()) if (~visited).any() else 0.0,
        float(direct_qstd[visited].mean()) if visited.any() else 0.0,
        float(direct_qstd[~visited].mean()) if (~visited).any() else 0.0,
        float(direct_prior_effect_std[visited].mean()) if visited.any() else 0.0,
        float(direct_prior_effect_std[~visited].mean()) if (~visited).any() else 0.0,
        float(rows[0]['composition_abs_max_error']),
    )
    if tb_logger is not None:
        stage = str(diagnostic_stage).replace(' ', '_').replace('/', '_')
        tb_logger.add_scalar(
            f'encoder_prior_grid/{stage}_full_pair_l2_unseen',
            float(full_l2[~visited].mean()) if (~visited).any() else 0.0,
            collection_phase,
        )
        tb_logger.add_scalar(
            f'encoder_prior_grid/{stage}_direct_qstd_unseen',
            float(direct_qstd[~visited].mean()) if (~visited).any() else 0.0,
            collection_phase,
        )
        tb_logger.add_scalar(
            f'encoder_prior_grid/{stage}_direct_value_prior_effect_std_unseen',
            float(direct_prior_effect_std[~visited].mean()) if (~visited).any() else 0.0,
            collection_phase,
        )


def _fresh_collection_observations(cfg, new_data: List[Any]) -> np.ndarray:
    """Return every real MountainCar observation in the fresh collection once per segment."""
    observations = []
    segments, metadata = new_data
    for game_segment, meta in zip(segments, metadata):
        valid_step_count = min(
            int(meta.get('segment_length', len(game_segment))),
            len(getattr(game_segment, 'action_segment', [])),
        )
        obs_segment = np.asarray(game_segment.obs_segment)
        obs_count = _real_observation_count(cfg, valid_step_count, len(obs_segment))
        for timestep in range(valid_step_count + 1):
            obs_index = _decision_obs_index(cfg, timestep, obs_count)
            if obs_index >= len(obs_segment):
                continue
            obs = np.asarray(obs_segment[obs_index], dtype=np.float32).reshape(-1)
            if obs.size >= 2 and np.isfinite(obs[:2]).all():
                observations.append(obs[:2])
    if not observations:
        return np.empty((0, 2), dtype=np.float32)
    return np.stack(observations, axis=0).astype(np.float32, copy=False)


def _encoder_prior_probe_observation_history(
        cfg,
        new_data: List[Any],
        collection_phase: int,
        update: bool,
) -> np.ndarray:
    diagnostics_dir = _diagnostics_dir(cfg)
    history_path = os.path.join(diagnostics_dir, 'encoder_prior_observation_history.npz')
    observations = np.empty((0, 2), dtype=np.float32)
    last_phase = -1
    if os.path.exists(history_path):
        try:
            with np.load(history_path, allow_pickle=False) as saved:
                saved_observations = np.asarray(saved['observations'], dtype=np.float32)
                if saved_observations.ndim == 2 and saved_observations.shape[1] == 2:
                    observations = saved_observations
                    last_phase = int(np.asarray(saved['last_phase']).reshape(-1)[0])
        except (OSError, ValueError, KeyError):
            logging.warning('[ENCODER_PRIOR_IMAGINED_NOVELTY] could not read %s; rebuilding history', history_path)

    if update and int(collection_phase) > last_phase:
        fresh = _fresh_collection_observations(cfg, new_data)
        if fresh.shape[0] > 0:
            observations = np.concatenate([observations, fresh], axis=0)
        history_limit = max(1, int(cfg.policy.get('encoder_prior_imagined_novelty_history_limit', 100000)))
        if observations.shape[0] > history_limit:
            indices = np.linspace(0, observations.shape[0] - 1, num=history_limit, dtype=np.int64)
            observations = observations[indices]
        np.savez_compressed(
            history_path,
            observations=observations,
            last_phase=np.asarray([collection_phase], dtype=np.int64),
        )
    return observations


def _batched_knn_distances(
        query: torch.Tensor,
        bank: torch.Tensor,
        k: int,
        batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return mean k-nearest distance and nearest-bank index for every query."""
    if query.dim() != 2 or bank.dim() != 2:
        raise ValueError('KNN inputs must both be rank-two tensors.')
    if bank.shape[0] <= 0:
        raise ValueError('KNN reference bank must not be empty.')
    neighbor_count = min(max(1, int(k)), int(bank.shape[0]))
    chunk_size = max(1, int(batch_size))
    mean_distances = []
    nearest_indices = []
    for start in range(0, int(query.shape[0]), chunk_size):
        distances = torch.cdist(query[start:start + chunk_size].float(), bank.float(), p=2)
        nearest_distances, indices = torch.topk(
            distances, k=neighbor_count, dim=1, largest=False, sorted=True
        )
        mean_distances.append(nearest_distances.mean(dim=1))
        nearest_indices.append(indices[:, 0])
    return torch.cat(mean_distances, dim=0), torch.cat(nearest_indices, dim=0)


def _finite_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    valid = np.isfinite(first) & np.isfinite(second)
    if valid.sum() < 2 or first[valid].std() <= 1e-12 or second[valid].std() <= 1e-12:
        return 0.0
    return float(np.corrcoef(first[valid], second[valid])[0, 1])


def _same_root_novelty_ranking_accuracy(
        rows: List[Dict[str, Any]],
        score_key: str,
) -> float:
    outcomes = []
    rows_by_root = defaultdict(list)
    for row in rows:
        rows_by_root[int(row['root_index'])].append(row)
    for root_rows in rows_by_root.values():
        valid_rows = [
            row for row in root_rows
            if np.isfinite(float(row['raw_endpoint_knn_distance']))
            and np.isfinite(float(row[score_key]))
        ]
        raw_scores = np.asarray(
            [row['raw_endpoint_knn_distance'] for row in valid_rows], dtype=np.float64
        )
        if raw_scores.size < 2 or np.ptp(raw_scores) <= 1e-12:
            continue
        ordered_indices = np.argsort(raw_scores)
        quartile_size = max(1, len(valid_rows) // 4)
        familiar_score = float(np.mean([
            float(valid_rows[int(index)][score_key]) for index in ordered_indices[:quartile_size]
        ]))
        novel_score = float(np.mean([
            float(valid_rows[int(index)][score_key]) for index in ordered_indices[-quartile_size:]
        ]))
        outcomes.append(float(novel_score > familiar_score))
    return float(np.mean(outcomes)) if outcomes else 0.0


def _write_encoder_prior_imagined_novelty_diagnostics(
        cfg,
        policy,
        new_data: List[Any],
        tb_logger,
        collection_phase: int,
        train_iter: int,
        diagnostic_stage: str,
        update_history: bool,
) -> None:
    """Measure whether direct-state latent novelty transfers to depth-H imagined latents."""
    if (
            not _diagnostics_enabled(cfg)
            or not bool(cfg.policy.get('encoder_prior_imagined_novelty_diagnostics', False))
    ):
        return
    if (
            int(cfg.policy.model.get('observation_shape', 0)) != 2
            or int(cfg.policy.model.get('frame_stack_num', 1)) != 1
    ):
        logging.warning(
            '[ENCODER_PRIOR_IMAGINED_NOVELTY] skipped: probe requires unstacked two-dimensional observations'
        )
        return

    model = getattr(policy, '_collect_model', None)
    if model is None:
        model = getattr(policy, '_learn_model', None)
    if model is None or not hasattr(model, 'members'):
        logging.warning('[ENCODER_PRIOR_IMAGINED_NOVELTY] skipped: ensemble model is unavailable')
        return

    history = _encoder_prior_probe_observation_history(
        cfg, new_data, collection_phase, update=update_history
    )
    min_bank_size = max(2, int(cfg.policy.get('encoder_prior_imagined_novelty_min_bank_size', 32)))
    if history.shape[0] < min_bank_size:
        logging.warning(
            '[ENCODER_PRIOR_IMAGINED_NOVELTY] phase %s %s skipped: observation history has %s states',
            collection_phase,
            diagnostic_stage,
            history.shape[0],
        )
        return

    bank_size = max(2, int(cfg.policy.get('encoder_prior_imagined_novelty_bank_size', 2048)))
    if history.shape[0] > bank_size:
        bank_indices = np.linspace(0, history.shape[0] - 1, num=bank_size, dtype=np.int64)
        bank_observations = history[bank_indices]
    else:
        bank_observations = history

    roots = _sample_fixed_intrinsic_states(cfg, new_data)
    max_roots = max(1, int(cfg.policy.get('encoder_prior_imagined_novelty_root_count', 32)))
    if len(roots) > max_roots:
        root_indices = np.linspace(0, len(roots) - 1, num=max_roots, dtype=np.int64)
        roots = [roots[int(index)] for index in root_indices]
    if not roots:
        return

    depth = max(1, int(cfg.policy.get('encoder_prior_imagined_novelty_depth', 5)))
    action_space_size = int(getattr(model, 'action_space_size', cfg.policy.model.get('action_space_size', 3)))
    sequences = np.asarray(list(product(range(action_space_size), repeat=depth)), dtype=np.int64)
    max_sequences = int(cfg.policy.get('encoder_prior_imagined_novelty_max_sequences', 0))
    if max_sequences > 0 and sequences.shape[0] > max_sequences:
        sequence_indices = np.linspace(0, sequences.shape[0] - 1, num=max_sequences, dtype=np.int64)
        sequences = sequences[sequence_indices]

    root_observations = np.stack(
        [np.asarray(root['obs'], dtype=np.float32).reshape(-1)[:2] for root in roots], axis=0
    )
    query_root_indices = np.repeat(np.arange(len(roots), dtype=np.int64), sequences.shape[0])
    query_sequences = np.tile(sequences, (len(roots), 1))
    endpoint_observations = np.empty((query_sequences.shape[0], 2), dtype=np.float32)
    for query_index, (root_index, sequence) in enumerate(zip(query_root_indices, query_sequences)):
        endpoint_observations[query_index] = _mountain_car_sequence_next_observations(
            float(root_observations[root_index, 0]),
            float(root_observations[root_index, 1]),
            sequence.tolist(),
        )[-1]

    positions, velocities, _ = _encoder_prior_grid(cfg)
    flat_visit_counts = _encoder_prior_grid_visit_counts(
        cfg, new_data, collection_phase, positions, velocities, update=False
    )
    grid_visit_counts = flat_visit_counts.reshape(positions.size, velocities.size)
    position_indices = np.rint(
        (endpoint_observations[:, 0] - positions[0])
        / max(float(positions[-1] - positions[0]), 1e-12)
        * (positions.size - 1)
    ).astype(np.int64)
    velocity_indices = np.rint(
        (endpoint_observations[:, 1] - velocities[0])
        / max(float(velocities[-1] - velocities[0]), 1e-12)
        * (velocities.size - 1)
    ).astype(np.int64)
    position_indices = np.clip(position_indices, 0, positions.size - 1)
    velocity_indices = np.clip(velocity_indices, 0, velocities.size - 1)
    endpoint_visit_counts = grid_visit_counts[position_indices, velocity_indices]

    device = torch.device(cfg.policy.device)
    ensemble_size = int(cfg.policy.model.get('ensemble_size', 1))
    knn_k = max(1, int(cfg.policy.get('encoder_prior_imagined_novelty_knn_k', 10)))
    query_batch_size = max(1, int(cfg.policy.get('encoder_prior_imagined_novelty_batch_size', 256)))
    feature_std_eps = max(
        1e-8, float(cfg.policy.get('encoder_prior_imagined_novelty_feature_std_eps', 1e-3))
    )
    position_scale = max(
        1e-8, float(cfg.policy.get('encoder_prior_imagined_novelty_position_scale', 0.05))
    )
    velocity_scale = max(
        1e-8, float(cfg.policy.get('encoder_prior_imagined_novelty_velocity_scale', 0.005))
    )
    observation_scale = torch.as_tensor(
        [position_scale, velocity_scale], dtype=torch.float32, device=device
    )
    value_offsets = np.zeros(ensemble_size, dtype=np.float32)
    calibration_fn = getattr(policy, '_value_calibration_offsets', None)
    if callable(calibration_fn):
        value_offsets = np.asarray(calibration_fn(ensemble_size), dtype=np.float32)
    value_offset_tensor = torch.as_tensor(value_offsets, dtype=torch.float32, device=device).view(
        ensemble_size, 1
    )

    was_training = model.training
    active_member_state = getattr(model, 'active_member_id', None)
    model.eval()
    rows = []
    try:
        with torch.no_grad():
            bank_obs_tensor = torch.as_tensor(bank_observations, dtype=torch.float32, device=device)
            bank_count = int(bank_obs_tensor.shape[0])
            bank_member_ids = [
                member_id for member_id in range(ensemble_size) for _ in range(bank_count)
            ]
            repeated_bank_obs = torch.cat([bank_obs_tensor for _ in range(ensemble_size)], dim=0)
            bank_output = model.initial_inference(repeated_bank_obs, member_id=bank_member_ids)
            bank_latents = bank_output.latent_state.reshape(ensemble_size, bank_count, -1).float()
            bank_latent_mean = bank_latents.mean(dim=1, keepdim=True)
            bank_latent_std = bank_latents.std(dim=1, unbiased=False, keepdim=True).clamp_min(feature_std_eps)
            standardized_bank_latents = (bank_latents - bank_latent_mean) / bank_latent_std

            projection_available = (
                float(cfg.policy.get('ssl_loss_weight', 0.0)) > 0.0
                and all(hasattr(member, 'project') for member in model.members)
            )
            normalized_bank_projections = None
            if projection_available:
                bank_projection = model.project(
                    bank_output.latent_state, with_grad=False, member_id=bank_member_ids
                ).reshape(ensemble_size, bank_count, -1).float()
                normalized_bank_projections = bank_projection / bank_projection.norm(
                    p=2, dim=2, keepdim=True
                ).clamp_min(1e-6)

            root_obs_tensor = torch.as_tensor(root_observations, dtype=torch.float32, device=device)
            root_count = int(root_obs_tensor.shape[0])
            root_member_ids = [
                member_id for member_id in range(ensemble_size) for _ in range(root_count)
            ]
            repeated_root_obs = torch.cat([root_obs_tensor for _ in range(ensemble_size)], dim=0)
            root_output = model.initial_inference(repeated_root_obs, member_id=root_member_ids)
            root_latents = root_output.latent_state.reshape(ensemble_size, root_count, -1).float()

            normalized_bank_observations = bank_obs_tensor / observation_scale
            for query_start in range(0, query_sequences.shape[0], query_batch_size):
                query_end = min(query_sequences.shape[0], query_start + query_batch_size)
                chunk_sequences = query_sequences[query_start:query_end]
                chunk_root_indices = query_root_indices[query_start:query_end]
                chunk_endpoints_np = endpoint_observations[query_start:query_end]
                chunk_size = int(query_end - query_start)
                chunk_member_ids = [
                    member_id for member_id in range(ensemble_size) for _ in range(chunk_size)
                ]

                chunk_endpoints = torch.as_tensor(chunk_endpoints_np, dtype=torch.float32, device=device)
                repeated_endpoints = torch.cat([chunk_endpoints for _ in range(ensemble_size)], dim=0)
                direct_output = model.initial_inference(repeated_endpoints, member_id=chunk_member_ids)
                direct_latents = direct_output.latent_state.reshape(ensemble_size, chunk_size, -1).float()

                recurrent_latents = root_latents[:, chunk_root_indices, :].reshape(
                    ensemble_size * chunk_size, -1
                )
                recurrent_output = None
                for step in range(depth):
                    step_actions = torch.as_tensor(
                        np.tile(chunk_sequences[:, step], ensemble_size),
                        dtype=torch.long,
                        device=device,
                    ).view(-1, 1)
                    recurrent_output = model.recurrent_inference(
                        recurrent_latents, step_actions, member_id=chunk_member_ids
                    )
                    recurrent_latents = recurrent_output.latent_state
                imagined_latents = recurrent_latents.reshape(ensemble_size, chunk_size, -1).float()

                direct_values = policy.value_inverse_scalar_transform_handle(direct_output.value).detach()
                direct_values = direct_values.reshape(ensemble_size, chunk_size, -1).squeeze(-1).float()
                imagined_values = policy.value_inverse_scalar_transform_handle(recurrent_output.value).detach()
                imagined_values = imagined_values.reshape(ensemble_size, chunk_size, -1).squeeze(-1).float()
                direct_calibrated_values = direct_values - value_offset_tensor
                imagined_calibrated_values = imagined_values - value_offset_tensor

                direct_latent_knn = []
                imagined_latent_knn = []
                direct_nearest_indices = []
                imagined_nearest_indices = []
                imagined_direct_latent_l2 = []
                for member_id in range(ensemble_size):
                    direct_standardized = (
                        direct_latents[member_id:member_id + 1]
                        - bank_latent_mean[member_id:member_id + 1]
                    ) / bank_latent_std[member_id:member_id + 1]
                    imagined_standardized = (
                        imagined_latents[member_id:member_id + 1]
                        - bank_latent_mean[member_id:member_id + 1]
                    ) / bank_latent_std[member_id:member_id + 1]
                    direct_standardized = direct_standardized.squeeze(0)
                    imagined_standardized = imagined_standardized.squeeze(0)
                    direct_knn, direct_nearest = _batched_knn_distances(
                        direct_standardized,
                        standardized_bank_latents[member_id],
                        knn_k,
                        query_batch_size,
                    )
                    imagined_knn, imagined_nearest = _batched_knn_distances(
                        imagined_standardized,
                        standardized_bank_latents[member_id],
                        knn_k,
                        query_batch_size,
                    )
                    direct_latent_knn.append(direct_knn)
                    imagined_latent_knn.append(imagined_knn)
                    direct_nearest_indices.append(direct_nearest)
                    imagined_nearest_indices.append(imagined_nearest)
                    imagined_direct_latent_l2.append(
                        (imagined_standardized - direct_standardized).norm(p=2, dim=1)
                    )
                direct_latent_knn = torch.stack(direct_latent_knn, dim=0)
                imagined_latent_knn = torch.stack(imagined_latent_knn, dim=0)
                direct_nearest_indices = torch.stack(direct_nearest_indices, dim=0)
                imagined_nearest_indices = torch.stack(imagined_nearest_indices, dim=0)
                imagined_direct_latent_l2 = torch.stack(imagined_direct_latent_l2, dim=0)

                endpoint_normalized = chunk_endpoints / observation_scale
                raw_endpoint_knn, _ = _batched_knn_distances(
                    endpoint_normalized,
                    normalized_bank_observations,
                    knn_k,
                    query_batch_size,
                )
                direct_nearest_state_distance = []
                imagined_nearest_state_distance = []
                for member_id in range(ensemble_size):
                    direct_neighbor_obs = bank_obs_tensor.index_select(
                        0, direct_nearest_indices[member_id]
                    )
                    imagined_neighbor_obs = bank_obs_tensor.index_select(
                        0, imagined_nearest_indices[member_id]
                    )
                    direct_nearest_state_distance.append(
                        ((direct_neighbor_obs - chunk_endpoints) / observation_scale).norm(p=2, dim=1)
                    )
                    imagined_nearest_state_distance.append(
                        ((imagined_neighbor_obs - chunk_endpoints) / observation_scale).norm(p=2, dim=1)
                    )
                direct_nearest_state_distance = torch.stack(direct_nearest_state_distance, dim=0)
                imagined_nearest_state_distance = torch.stack(imagined_nearest_state_distance, dim=0)

                direct_projection_knn = torch.full_like(direct_latent_knn, float('nan'))
                imagined_projection_knn = torch.full_like(imagined_latent_knn, float('nan'))
                imagined_direct_projection_cosine = torch.full_like(
                    imagined_direct_latent_l2, float('nan')
                )
                if projection_available:
                    direct_projection = model.project(
                        direct_output.latent_state, with_grad=False, member_id=chunk_member_ids
                    ).reshape(ensemble_size, chunk_size, -1).float()
                    imagined_prediction_projection = model.project(
                        recurrent_latents, with_grad=True, member_id=chunk_member_ids
                    ).reshape(ensemble_size, chunk_size, -1).float()
                    direct_projection = direct_projection / direct_projection.norm(
                        p=2, dim=2, keepdim=True
                    ).clamp_min(1e-6)
                    imagined_prediction_projection = imagined_prediction_projection / imagined_prediction_projection.norm(
                        p=2, dim=2, keepdim=True
                    ).clamp_min(1e-6)
                    imagined_direct_projection_cosine = (
                        imagined_prediction_projection * direct_projection
                    ).sum(dim=2)
                    for member_id in range(ensemble_size):
                        direct_knn, _ = _batched_knn_distances(
                            direct_projection[member_id],
                            normalized_bank_projections[member_id],
                            knn_k,
                            query_batch_size,
                        )
                        imagined_knn, _ = _batched_knn_distances(
                            imagined_prediction_projection[member_id],
                            normalized_bank_projections[member_id],
                            knn_k,
                            query_batch_size,
                        )
                        direct_projection_knn[member_id] = direct_knn
                        imagined_projection_knn[member_id] = imagined_knn

                for chunk_index in range(chunk_size):
                    query_index = query_start + chunk_index
                    root_index = int(chunk_root_indices[chunk_index])
                    root = roots[root_index]
                    sequence = chunk_sequences[chunk_index]
                    direct_knn_values = direct_latent_knn[:, chunk_index]
                    imagined_knn_values = imagined_latent_knn[:, chunk_index]
                    latent_alignment_values = imagined_direct_latent_l2[:, chunk_index]
                    direct_projection_values = direct_projection_knn[:, chunk_index]
                    imagined_projection_values = imagined_projection_knn[:, chunk_index]
                    projection_cosine_values = imagined_direct_projection_cosine[:, chunk_index]
                    alignment_ratio = latent_alignment_values / direct_knn_values.clamp_min(1e-6)
                    row = {
                        'collection_phase': int(collection_phase),
                        'diagnostic_stage': str(diagnostic_stage),
                        'train_iter': int(train_iter),
                        'root_index': root_index,
                        'root_position': float(root_observations[root_index, 0]),
                        'root_velocity': float(root_observations[root_index, 1]),
                        'root_collection_episode_index': root.get('collection_episode_index'),
                        'root_timestep': root.get('timestep'),
                        'sequence_index': int(query_index % sequences.shape[0]),
                        'sequence': ','.join(str(int(action)) for action in sequence),
                        'sequence_depth': int(depth),
                        'endpoint_position': float(chunk_endpoints_np[chunk_index, 0]),
                        'endpoint_velocity': float(chunk_endpoints_np[chunk_index, 1]),
                        'endpoint_reached_goal': float(chunk_endpoints_np[chunk_index, 0] >= 0.5),
                        'endpoint_grid_visit_count': int(endpoint_visit_counts[query_index]),
                        'raw_endpoint_knn_distance': float(raw_endpoint_knn[chunk_index].item()),
                        'direct_latent_knn_mean': float(direct_knn_values.mean().item()),
                        'imagined_latent_knn_mean': float(imagined_knn_values.mean().item()),
                        'direct_latent_knn_members': json.dumps(direct_knn_values.cpu().tolist()),
                        'imagined_latent_knn_members': json.dumps(imagined_knn_values.cpu().tolist()),
                        'imagined_direct_latent_l2_mean': float(latent_alignment_values.mean().item()),
                        'imagined_direct_latent_l2_members': json.dumps(
                            latent_alignment_values.cpu().tolist()
                        ),
                        'imagined_direct_l2_over_direct_knn_mean': float(alignment_ratio.mean().item()),
                        'direct_nearest_bank_state_distance_mean': float(
                            direct_nearest_state_distance[:, chunk_index].mean().item()
                        ),
                        'imagined_nearest_bank_state_distance_mean': float(
                            imagined_nearest_state_distance[:, chunk_index].mean().item()
                        ),
                        'direct_projection_knn_mean': float(direct_projection_values.mean().item()),
                        'imagined_prediction_projection_knn_mean': float(
                            imagined_projection_values.mean().item()
                        ),
                        'imagined_direct_projection_cosine_mean': float(
                            projection_cosine_values.mean().item()
                        ),
                        'direct_value_std': float(direct_values[:, chunk_index].std(unbiased=False).item()),
                        'imagined_value_std': float(imagined_values[:, chunk_index].std(unbiased=False).item()),
                        'direct_calibrated_value_std': float(
                            direct_calibrated_values[:, chunk_index].std(unbiased=False).item()
                        ),
                        'imagined_calibrated_value_std': float(
                            imagined_calibrated_values[:, chunk_index].std(unbiased=False).item()
                        ),
                    }
                    rows.append(row)
    finally:
        _restore_model_member_state(model, active_member_state)
        model.train(was_training)

    if not rows:
        return
    diagnostics_dir = _diagnostics_dir(cfg)
    csv_path = os.path.join(diagnostics_dir, 'encoder_prior_imagined_novelty.csv')
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    raw_novelty = np.asarray([row['raw_endpoint_knn_distance'] for row in rows], dtype=np.float64)
    direct_novelty = np.asarray([row['direct_latent_knn_mean'] for row in rows], dtype=np.float64)
    imagined_novelty = np.asarray([row['imagined_latent_knn_mean'] for row in rows], dtype=np.float64)
    direct_projection_novelty = np.asarray(
        [row['direct_projection_knn_mean'] for row in rows], dtype=np.float64
    )
    imagined_projection_novelty = np.asarray(
        [row['imagined_prediction_projection_knn_mean'] for row in rows], dtype=np.float64
    )
    raw_to_direct = _finite_correlation(raw_novelty, direct_novelty)
    raw_to_imagined = _finite_correlation(raw_novelty, imagined_novelty)
    direct_to_imagined = _finite_correlation(direct_novelty, imagined_novelty)
    projection_direct_to_imagined = _finite_correlation(
        direct_projection_novelty, imagined_projection_novelty
    )
    direct_rank_accuracy = _same_root_novelty_ranking_accuracy(rows, 'direct_latent_knn_mean')
    imagined_rank_accuracy = _same_root_novelty_ranking_accuracy(rows, 'imagined_latent_knn_mean')
    imagined_projection_rank_accuracy = _same_root_novelty_ranking_accuracy(
        rows, 'imagined_prediction_projection_knn_mean'
    )
    alignment_ratios = np.asarray(
        [row['imagined_direct_l2_over_direct_knn_mean'] for row in rows], dtype=np.float64
    )
    retrieval_errors = np.asarray(
        [row['imagined_nearest_bank_state_distance_mean'] for row in rows], dtype=np.float64
    )
    logging.info(
        '[ENCODER_PRIOR_IMAGINED_NOVELTY] phase %s %s roots %s sequences/root %s bank %s | '
        'corr raw->direct %.3f raw->imagined %.3f direct->imagined %.3f projection %.3f | '
        'same-root novel>familiar direct %.3f imagined %.3f projection %.3f | '
        'alignment/direct-density median %.3f retrieval-state-error median %.3f',
        collection_phase,
        diagnostic_stage,
        len(roots),
        sequences.shape[0],
        bank_observations.shape[0],
        raw_to_direct,
        raw_to_imagined,
        direct_to_imagined,
        projection_direct_to_imagined,
        direct_rank_accuracy,
        imagined_rank_accuracy,
        imagined_projection_rank_accuracy,
        float(np.nanmedian(alignment_ratios)),
        float(np.nanmedian(retrieval_errors)),
    )
    if tb_logger is not None:
        stage = str(diagnostic_stage).replace(' ', '_').replace('/', '_')
        metrics = {
            'raw_to_direct_correlation': raw_to_direct,
            'raw_to_imagined_correlation': raw_to_imagined,
            'direct_to_imagined_correlation': direct_to_imagined,
            'projection_direct_to_imagined_correlation': projection_direct_to_imagined,
            'direct_same_root_ranking_accuracy': direct_rank_accuracy,
            'imagined_same_root_ranking_accuracy': imagined_rank_accuracy,
            'imagined_projection_same_root_ranking_accuracy': imagined_projection_rank_accuracy,
            'alignment_over_direct_density_median': float(np.nanmedian(alignment_ratios)),
            'retrieval_state_error_median': float(np.nanmedian(retrieval_errors)),
        }
        for name, value in metrics.items():
            tb_logger.add_scalar(
                f'encoder_prior_imagined_novelty/{stage}_{name}', float(value), collection_phase
            )


def _probe_fixed_sequence_intrinsic_returns(
        cfg,
        model: torch.nn.Module,
        states: List[Dict[str, Any]],
        sequences: List[Tuple[str, List[int]]],
        policy=None,
) -> List[Dict[str, Any]]:
    if not states or not sequences:
        return []
    if not hasattr(model, 'intrinsic_reward_from_latent') or not hasattr(model, 'intrinsic_value_from_latent'):
        return []

    device = torch.device(cfg.policy.device)
    ensemble_size = int(cfg.policy.model.get('ensemble_size', 1))
    discount_factor = cfg.policy.get('rnd_intrinsic_discount_factor', None)
    if discount_factor is None:
        discount_factor = cfg.policy.discount_factor
    discount_factor = float(discount_factor)

    obs_batch = torch.as_tensor(
        np.stack([np.asarray(state['obs'], dtype=np.float32) for state in states], axis=0),
        dtype=torch.float32,
        device=device,
    )
    batch_size = obs_batch.shape[0]
    member_ids = [member_id for member_id in range(ensemble_size) for _ in range(batch_size)]
    repeated_obs = torch.cat([obs_batch for _ in range(ensemble_size)], dim=0)

    rows = []
    was_training = model.training
    active_member_state = getattr(model, 'active_member_id', None)
    model.eval()
    try:
        with torch.no_grad():
            root_output = model.initial_inference(repeated_obs, member_id=member_ids)
            root_latent = root_output.latent_state.detach()
            for sequence_name, actions in sequences:
                if not actions:
                    continue
                true_next_observations = np.stack(
                    [
                        _mountain_car_sequence_next_observations(
                            _finite_float(state.get('obs_position')),
                            _finite_float(state.get('obs_velocity')),
                            actions,
                        )
                        for state in states
                    ],
                    axis=0,
                )
                true_rnd_rewards = None
                if _intrinsic_reward_source(cfg) == 'rnd' and policy is not None:
                    rnd_reward_fn = getattr(policy, '_collection_rnd_rewards_for_observations', None)
                    if callable(rnd_reward_fn):
                        reward_info = rnd_reward_fn(true_next_observations.reshape(-1, 2))
                        true_rnd_rewards = np.asarray(
                            reward_info['bonus'], dtype=np.float32
                        ).reshape(batch_size, len(actions))
                latent_state = root_latent
                reward_terms = []
                for action in actions:
                    action_batch = torch.full(
                        (ensemble_size * batch_size,),
                        int(action),
                        dtype=torch.long,
                        device=device,
                    )
                    recurrent_output = model.recurrent_inference(
                        latent_state, action_batch, member_id=member_ids
                    )
                    latent_state = recurrent_output.latent_state.detach()
                    intrinsic_reward = model.intrinsic_reward_from_latent(
                        latent_state, member_id=member_ids
                    ).detach().reshape(ensemble_size, batch_size)
                    reward_terms.append(intrinsic_reward.transpose(0, 1))

                reward_tensor = torch.stack(reward_terms, dim=1)
                discounts = torch.as_tensor(
                    [discount_factor ** step for step in range(len(actions))],
                    dtype=reward_tensor.dtype,
                    device=device,
                ).view(1, len(actions), 1)
                reward_sum = (reward_tensor * discounts).sum(dim=1)

                bootstrap_value = model.intrinsic_value_from_latent(
                    latent_state, member_id=member_ids
                ).detach().reshape(ensemble_size, batch_size).transpose(0, 1)
                true_endpoint_model_observations = np.stack(
                    [np.asarray(state['obs'], dtype=np.float32).reshape(-1) for state in states],
                    axis=0,
                )
                true_endpoint_model_observations[:, :2] = true_next_observations[:, -1, :]
                true_endpoint_obs = torch.as_tensor(
                    true_endpoint_model_observations,
                    dtype=torch.float32,
                    device=device,
                )
                repeated_true_endpoint_obs = torch.cat(
                    [true_endpoint_obs for _ in range(ensemble_size)], dim=0
                )
                direct_endpoint_output = model.initial_inference(
                    repeated_true_endpoint_obs, member_id=member_ids
                )
                direct_bootstrap_value = model.intrinsic_value_from_latent(
                    direct_endpoint_output.latent_state, member_id=member_ids
                ).detach().reshape(ensemble_size, batch_size).transpose(0, 1)
                bootstrap_gap = bootstrap_value - direct_bootstrap_value
                bootstrap_discount = discount_factor ** len(actions)
                returns = reward_sum + bootstrap_discount * bootstrap_value

                reward_sum_np = reward_sum.detach().cpu().numpy()
                bootstrap_np = bootstrap_value.detach().cpu().numpy()
                direct_bootstrap_np = direct_bootstrap_value.detach().cpu().numpy()
                bootstrap_gap_np = bootstrap_gap.detach().cpu().numpy()
                returns_np = returns.detach().cpu().numpy()
                immediate_reward_np = reward_tensor[:, 0, :].detach().cpu().numpy()
                predicted_step_rewards_np = reward_tensor.mean(dim=2).detach().cpu().numpy()

                for state_index, state in enumerate(states):
                    member_returns = returns_np[state_index].astype(np.float32)
                    member_reward_sums = reward_sum_np[state_index].astype(np.float32)
                    member_bootstrap = bootstrap_np[state_index].astype(np.float32)
                    member_direct_bootstrap = direct_bootstrap_np[state_index].astype(np.float32)
                    member_bootstrap_gap = bootstrap_gap_np[state_index].astype(np.float32)
                    true_sequence_metrics = _simulate_mountain_car_sequence(
                        _finite_float(state.get('obs_position')),
                        _finite_float(state.get('obs_velocity')),
                        actions,
                    )
                    true_step_rewards = (
                        true_rnd_rewards[state_index]
                        if true_rnd_rewards is not None
                        else np.full(len(actions), np.nan, dtype=np.float32)
                    )
                    true_discounted_reward_sum = float(
                        np.sum(true_step_rewards * discounts.reshape(-1).detach().cpu().numpy())
                    )
                    rows.append(
                        {
                            'collection_phase': state['collection_phase'],
                            'train_iter_at_collect': state['train_iter_at_collect'],
                            'envstep_before_collect': state['envstep_before_collect'],
                            'envstep_after_collect': state['envstep_after_collect'],
                            'segment_index_in_collect': state['segment_index_in_collect'],
                            'env_id': state['env_id'],
                            'env_episode_index': state['env_episode_index'],
                            'collection_episode_index': state['collection_episode_index'],
                            'episode_segment_index': state['episode_segment_index'],
                            'episode_return': state['episode_return'],
                            'video_path': state['video_path'],
                            'timestep': state['timestep'],
                            'obs_index': state['obs_index'],
                            'selected_action': state['selected_action'],
                            'selected_action_direction': state['selected_action_direction'],
                            'obs_position': state['obs_position'],
                            'obs_velocity': state['obs_velocity'],
                            'intrinsic_reward_source': str(cfg.policy.get('intrinsic_reward_source', 'rnd') or 'rnd'),
                            'sequence_name': sequence_name,
                            'sequence_actions': json.dumps(actions),
                            'sequence_length': len(actions),
                            'discount_factor': discount_factor,
                            'pred_intrinsic_first_reward_mean': float(immediate_reward_np[state_index].mean()),
                            'pred_intrinsic_reward_sum_mean': float(member_reward_sums.mean()),
                            'pred_intrinsic_reward_sum_std': float(member_reward_sums.std()),
                            'pred_intrinsic_bootstrap_mean': float(member_bootstrap.mean()),
                            'pred_intrinsic_bootstrap_std': float(member_bootstrap.std()),
                            'direct_true_endpoint_intrinsic_value_mean': float(
                                member_direct_bootstrap.mean()
                            ),
                            'direct_true_endpoint_intrinsic_value_std': float(
                                member_direct_bootstrap.std()
                            ),
                            'imagined_direct_intrinsic_value_gap_mean': float(
                                member_bootstrap_gap.mean()
                            ),
                            'imagined_direct_intrinsic_value_abs_gap_mean': float(
                                np.abs(member_bootstrap_gap).mean()
                            ),
                            'pred_intrinsic_bootstrap_discounted_mean': float(
                                (bootstrap_discount * member_bootstrap).mean()
                            ),
                            'pred_intrinsic_return_mean': float(member_returns.mean()),
                            'pred_intrinsic_return_std': float(member_returns.std()),
                            'member_intrinsic_returns': json.dumps(_jsonable(member_returns)),
                            'member_intrinsic_reward_sums': json.dumps(_jsonable(member_reward_sums)),
                            'member_intrinsic_bootstrap_values': json.dumps(_jsonable(member_bootstrap)),
                            'member_direct_true_endpoint_intrinsic_values': json.dumps(
                                _jsonable(member_direct_bootstrap)
                            ),
                            'member_imagined_direct_intrinsic_value_gaps': json.dumps(
                                _jsonable(member_bootstrap_gap)
                            ),
                            'sequence_step_intrinsic_rewards': json.dumps(
                                _jsonable(predicted_step_rewards_np[state_index])
                            ),
                            'true_rnd_first_reward': float(true_step_rewards[0]),
                            'true_rnd_discounted_reward_sum': true_discounted_reward_sum,
                            'sequence_step_true_rnd_rewards': json.dumps(_jsonable(true_step_rewards)),
                            'sequence_step_pair_std': json.dumps(_jsonable(np.zeros(len(actions), dtype=np.float32))),
                            'future_max_position_20': state['future_max_position_20'],
                            'future_max_position_50': state['future_max_position_50'],
                            'future_max_position_100': state['future_max_position_100'],
                            'future_reached_goal_100': state['future_reached_goal_100'],
                            'future_reached_goal_remaining': state['future_reached_goal_remaining'],
                            **true_sequence_metrics,
                        }
                    )
    finally:
        _restore_model_member_state(model, active_member_state)
        model.train(was_training)
    return rows


def _sequence_dynamics_disagreement(
        cfg,
        flat_latent_states: torch.Tensor,
        batch_size: int,
        ensemble_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    disagreement, disagreement_by_pair = _sequence_pairwise_disagreement(
        cfg,
        flat_latent_states,
        batch_size=batch_size,
        ensemble_size=ensemble_size,
        normalize=bool(cfg.policy.get('dynamics_disagreement_normalize_latent', True)),
        apply_clip=True,
    )
    return disagreement, disagreement_by_pair.std(dim=0, unbiased=False)


def _sequence_pairwise_disagreement(
        cfg,
        flat_latent_states: torch.Tensor,
        batch_size: int,
        ensemble_size: int,
        normalize: bool,
        apply_clip: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    latents = flat_latent_states.reshape(ensemble_size, batch_size, -1).float()
    if normalize:
        latents = torch.nn.functional.normalize(latents, p=2, dim=-1, eps=1e-8)
    metric = str(cfg.policy.get('dynamics_disagreement_distance_metric', 'mse')).lower()
    if metric not in {'mse', 'l2'}:
        raise ValueError(
            "dynamics_disagreement_distance_metric must be one of 'mse' or 'l2', "
            f"got {metric!r}."
        )
    distances = []
    for member_i in range(ensemble_size):
        for member_j in range(member_i + 1, ensemble_size):
            pair_delta = latents[member_i] - latents[member_j]
            if metric == 'l2':
                distances.append(pair_delta.pow(2).sum(dim=-1).sqrt())
            else:
                distances.append(pair_delta.pow(2).mean(dim=-1))
    if not distances:
        disagreement_by_pair = latents.new_zeros((1, batch_size))
    else:
        disagreement_by_pair = torch.stack(distances, dim=0)
    disagreement = disagreement_by_pair.mean(dim=0)
    clip_value = cfg.policy.get('dynamics_disagreement_reward_clip', None)
    if apply_clip and clip_value is not None:
        disagreement = disagreement.clamp(max=float(clip_value))
    return disagreement, disagreement_by_pair


def _sequence_pairwise_cosine(
        first_flat_states: torch.Tensor,
        second_flat_states: torch.Tensor,
        batch_size: int,
        ensemble_size: int,
) -> torch.Tensor:
    first = first_flat_states.reshape(ensemble_size, batch_size, -1).float()
    second = second_flat_states.reshape(ensemble_size, batch_size, -1).float()
    cosines = []
    for member_i in range(ensemble_size):
        for member_j in range(member_i + 1, ensemble_size):
            first_delta = first[member_i] - first[member_j]
            second_delta = second[member_i] - second[member_j]
            cosines.append(torch.nn.functional.cosine_similarity(first_delta, second_delta, dim=-1, eps=1e-8))
    if not cosines:
        return first.new_zeros(batch_size)
    return torch.stack(cosines, dim=0).mean(dim=0)


def _sequence_member_output_norm(
        flat_outputs: torch.Tensor,
        batch_size: int,
        ensemble_size: int,
) -> torch.Tensor:
    outputs = flat_outputs.reshape(ensemble_size, batch_size, -1).float()
    return outputs.norm(dim=-1).mean(dim=0)


def _sequence_projected_disagreement(
        cfg,
        model: torch.nn.Module,
        latent_state: torch.Tensor,
        batch_size: int,
        ensemble_size: int,
) -> Optional[Dict[str, torch.Tensor]]:
    if not bool(cfg.policy.model.get('self_supervised_learning_loss', False)):
        return None
    if not hasattr(model, 'project'):
        return None

    member_ids = [member_id for member_id in range(ensemble_size) for _ in range(batch_size)]
    try:
        projection = model.project(latent_state, with_grad=False, member_id=member_ids)
        prediction_projection = model.project(latent_state, with_grad=True, member_id=member_ids)
    except (AttributeError, NotImplementedError):
        return None

    projection_disagreement, _ = _sequence_pairwise_disagreement(
        cfg,
        projection,
        batch_size=batch_size,
        ensemble_size=ensemble_size,
        normalize=True,
        apply_clip=False,
    )
    prediction_projection_disagreement, _ = _sequence_pairwise_disagreement(
        cfg,
        prediction_projection,
        batch_size=batch_size,
        ensemble_size=ensemble_size,
        normalize=True,
        apply_clip=False,
    )
    projection_raw_disagreement, _ = _sequence_pairwise_disagreement(
        cfg,
        projection,
        batch_size=batch_size,
        ensemble_size=ensemble_size,
        normalize=False,
        apply_clip=False,
    )
    prediction_projection_raw_disagreement, _ = _sequence_pairwise_disagreement(
        cfg,
        prediction_projection,
        batch_size=batch_size,
        ensemble_size=ensemble_size,
        normalize=False,
        apply_clip=False,
    )

    return {
        'projection_disagreement': projection_disagreement.detach(),
        'prediction_projection_disagreement': prediction_projection_disagreement.detach(),
        'projection_raw_disagreement': projection_raw_disagreement.detach(),
        'prediction_projection_raw_disagreement': prediction_projection_raw_disagreement.detach(),
        'projection_norm': _sequence_member_output_norm(projection, batch_size, ensemble_size).detach(),
        'prediction_projection_norm': _sequence_member_output_norm(
            prediction_projection, batch_size, ensemble_size
        ).detach(),
    }


def _sequence_dynamics_prior_decomposition(
        cfg,
        model: torch.nn.Module,
        latent_state: torch.Tensor,
        action_batch: torch.Tensor,
        batch_size: int,
        ensemble_size: int,
) -> Optional[Dict[str, torch.Tensor]]:
    if not hasattr(model, 'members'):
        return None

    trainable_chunks = []
    prior_chunks = []
    raw_trainable_norm_chunks = []
    trainable_norm_chunks = []
    raw_prior_norm_chunks = []
    prior_norm_chunks = []
    full_norm_chunks = []
    scale = float(getattr(model, 'dynamics_random_prior_scale', cfg.policy.model.get('dynamics_random_prior_scale', 0.0)))
    prior_networks = getattr(model, 'dynamics_prior_networks', None)
    for member_id in range(ensemble_size):
        start = member_id * batch_size
        end = start + batch_size
        member = model.members[member_id]
        if not hasattr(member, '_dynamics'):
            return None
        member_latent = latent_state[start:end]
        member_action = action_batch[start:end]
        raw_trainable_latent, _ = member._dynamics(member_latent, member_action)
        trainable_latent = raw_trainable_latent
        if hasattr(model, '_normalize_trainable_dynamics_latent'):
            trainable_latent = model._normalize_trainable_dynamics_latent(trainable_latent)
        if prior_networks is None:
            raw_prior_delta = torch.zeros_like(trainable_latent)
            prior_delta = raw_prior_delta
        else:
            raw_prior_delta = prior_networks[member_id](member_latent, member_action) * scale
            prior_delta = raw_prior_delta
            if hasattr(model, '_bound_dynamics_prior_delta'):
                prior_delta = model._bound_dynamics_prior_delta(prior_delta)
        full_latent = trainable_latent + prior_delta
        trainable_chunks.append(trainable_latent.detach())
        prior_chunks.append(prior_delta.detach())
        raw_trainable_norm_chunks.append(raw_trainable_latent.detach().reshape(batch_size, -1).norm(dim=1))
        trainable_norm_chunks.append(trainable_latent.detach().reshape(batch_size, -1).norm(dim=1))
        raw_prior_norm_chunks.append(raw_prior_delta.detach().reshape(batch_size, -1).norm(dim=1))
        prior_norm_chunks.append(prior_delta.detach().reshape(batch_size, -1).norm(dim=1))
        full_norm_chunks.append(full_latent.detach().reshape(batch_size, -1).norm(dim=1))

    trainable_flat = torch.cat(trainable_chunks, dim=0)
    prior_flat = torch.cat(prior_chunks, dim=0)
    full_flat = trainable_flat + prior_flat
    normalize = bool(cfg.policy.get('dynamics_disagreement_normalize_latent', True))
    full_disagreement, _ = _sequence_pairwise_disagreement(
        cfg, full_flat, batch_size=batch_size, ensemble_size=ensemble_size, normalize=normalize, apply_clip=True
    )
    trainable_disagreement, _ = _sequence_pairwise_disagreement(
        cfg, trainable_flat, batch_size=batch_size, ensemble_size=ensemble_size, normalize=normalize, apply_clip=False
    )
    prior_disagreement, _ = _sequence_pairwise_disagreement(
        cfg, prior_flat, batch_size=batch_size, ensemble_size=ensemble_size, normalize=normalize, apply_clip=False
    )
    full_raw_disagreement, _ = _sequence_pairwise_disagreement(
        cfg, full_flat, batch_size=batch_size, ensemble_size=ensemble_size, normalize=False, apply_clip=False
    )
    trainable_raw_disagreement, _ = _sequence_pairwise_disagreement(
        cfg, trainable_flat, batch_size=batch_size, ensemble_size=ensemble_size, normalize=False, apply_clip=False
    )
    prior_raw_disagreement, _ = _sequence_pairwise_disagreement(
        cfg, prior_flat, batch_size=batch_size, ensemble_size=ensemble_size, normalize=False, apply_clip=False
    )
    train_prior_pair_cosine = _sequence_pairwise_cosine(
        trainable_flat, prior_flat, batch_size=batch_size, ensemble_size=ensemble_size
    )
    return {
        'full_latent': full_flat.detach(),
        'full_disagreement': full_disagreement.detach(),
        'trainable_disagreement': trainable_disagreement.detach(),
        'prior_disagreement': prior_disagreement.detach(),
        'full_raw_disagreement': full_raw_disagreement.detach(),
        'trainable_raw_disagreement': trainable_raw_disagreement.detach(),
        'prior_raw_disagreement': prior_raw_disagreement.detach(),
        'train_prior_pair_cosine': train_prior_pair_cosine.detach(),
        'raw_trainable_norm': torch.stack(raw_trainable_norm_chunks, dim=0).mean(dim=0).detach(),
        'trainable_norm': torch.stack(trainable_norm_chunks, dim=0).mean(dim=0).detach(),
        'raw_prior_norm': torch.stack(raw_prior_norm_chunks, dim=0).mean(dim=0).detach(),
        'prior_norm': torch.stack(prior_norm_chunks, dim=0).mean(dim=0).detach(),
        'full_norm': torch.stack(full_norm_chunks, dim=0).mean(dim=0).detach(),
    }


def _probe_fixed_sequence_dynamics_disagreement_returns(
        cfg,
        model: torch.nn.Module,
        states: List[Dict[str, Any]],
        sequences: List[Tuple[str, List[int]]],
) -> List[Dict[str, Any]]:
    if not states or not sequences:
        return []

    device = torch.device(cfg.policy.device)
    ensemble_size = int(cfg.policy.model.get('ensemble_size', 1))
    discount_factor = cfg.policy.get('rnd_intrinsic_discount_factor', None)
    if discount_factor is None:
        discount_factor = cfg.policy.discount_factor
    discount_factor = float(discount_factor)

    obs_batch = torch.as_tensor(
        np.stack([np.asarray(state['obs'], dtype=np.float32) for state in states], axis=0),
        dtype=torch.float32,
        device=device,
    )
    batch_size = obs_batch.shape[0]
    member_ids = [member_id for member_id in range(ensemble_size) for _ in range(batch_size)]
    repeated_obs = torch.cat([obs_batch for _ in range(ensemble_size)], dim=0)

    rows = []
    was_training = model.training
    active_member_state = getattr(model, 'active_member_id', None)
    model.eval()
    try:
        with torch.no_grad():
            root_output = model.initial_inference(repeated_obs, member_id=member_ids)
            root_latent = root_output.latent_state.detach()
            if bool(cfg.policy.get('dynamics_disagreement_shared_root_latent', True)) and ensemble_size > 1:
                root_latent = root_latent.reshape(ensemble_size, batch_size, *root_latent.shape[1:])
                root_latent = root_latent[0:1].expand_as(root_latent).contiguous()
                root_latent = root_latent.reshape(ensemble_size * batch_size, *root_latent.shape[2:])

            for sequence_name, actions in sequences:
                if not actions:
                    continue
                latent_state = root_latent
                reward_terms = []
                reward_pair_std_terms = []
                full_disagreement_terms = []
                trainable_disagreement_terms = []
                prior_disagreement_terms = []
                full_raw_disagreement_terms = []
                trainable_raw_disagreement_terms = []
                prior_raw_disagreement_terms = []
                train_prior_pair_cosine_terms = []
                raw_trainable_norm_terms = []
                trainable_norm_terms = []
                raw_prior_norm_terms = []
                prior_norm_terms = []
                full_norm_terms = []
                projection_disagreement_terms = []
                prediction_projection_disagreement_terms = []
                projection_raw_disagreement_terms = []
                prediction_projection_raw_disagreement_terms = []
                projection_norm_terms = []
                prediction_projection_norm_terms = []
                for action in actions:
                    action_batch = torch.full(
                        (ensemble_size * batch_size,),
                        int(action),
                        dtype=torch.long,
                        device=device,
                    )
                    decomposition = _sequence_dynamics_prior_decomposition(
                        cfg,
                        model,
                        latent_state,
                        action_batch,
                        batch_size=batch_size,
                        ensemble_size=ensemble_size,
                    )
                    recurrent_output = model.recurrent_inference(
                        latent_state, action_batch, member_id=member_ids
                    )
                    latent_state = recurrent_output.latent_state.detach()
                    disagreement, pair_std = _sequence_dynamics_disagreement(
                        cfg,
                        latent_state,
                        batch_size=batch_size,
                        ensemble_size=ensemble_size,
                    )
                    reward_terms.append(disagreement)
                    reward_pair_std_terms.append(pair_std)
                    if decomposition is not None:
                        full_disagreement_terms.append(decomposition['full_disagreement'])
                        trainable_disagreement_terms.append(decomposition['trainable_disagreement'])
                        prior_disagreement_terms.append(decomposition['prior_disagreement'])
                        full_raw_disagreement_terms.append(decomposition['full_raw_disagreement'])
                        trainable_raw_disagreement_terms.append(decomposition['trainable_raw_disagreement'])
                        prior_raw_disagreement_terms.append(decomposition['prior_raw_disagreement'])
                        train_prior_pair_cosine_terms.append(decomposition['train_prior_pair_cosine'])
                        raw_trainable_norm_terms.append(decomposition['raw_trainable_norm'])
                        trainable_norm_terms.append(decomposition['trainable_norm'])
                        raw_prior_norm_terms.append(decomposition['raw_prior_norm'])
                        prior_norm_terms.append(decomposition['prior_norm'])
                        full_norm_terms.append(decomposition['full_norm'])
                    projection_decomposition = _sequence_projected_disagreement(
                        cfg,
                        model,
                        latent_state,
                        batch_size=batch_size,
                        ensemble_size=ensemble_size,
                    )
                    if projection_decomposition is not None:
                        projection_disagreement_terms.append(projection_decomposition['projection_disagreement'])
                        prediction_projection_disagreement_terms.append(
                            projection_decomposition['prediction_projection_disagreement']
                        )
                        projection_raw_disagreement_terms.append(
                            projection_decomposition['projection_raw_disagreement']
                        )
                        prediction_projection_raw_disagreement_terms.append(
                            projection_decomposition['prediction_projection_raw_disagreement']
                        )
                        projection_norm_terms.append(projection_decomposition['projection_norm'])
                        prediction_projection_norm_terms.append(
                            projection_decomposition['prediction_projection_norm']
                        )

                disagreement_space = str(cfg.policy.get('dynamics_disagreement_space', 'latent') or 'latent').lower()
                reward_terms_for_return = reward_terms
                if disagreement_space == 'projection' and projection_disagreement_terms:
                    reward_terms_for_return = projection_disagreement_terms
                elif disagreement_space == 'prediction_projection' and prediction_projection_disagreement_terms:
                    reward_terms_for_return = prediction_projection_disagreement_terms
                reward_tensor = torch.stack(reward_terms_for_return, dim=1)
                reward_pair_std_tensor = torch.stack(reward_pair_std_terms, dim=1)
                discounts = torch.as_tensor(
                    [discount_factor ** step for step in range(len(actions))],
                    dtype=reward_tensor.dtype,
                    device=device,
                ).view(1, len(actions))
                reward_sum = (reward_tensor * discounts).sum(dim=1)
                immediate_weight = float(cfg.policy.get('intrinsic_tree_immediate_reward_weight', 1.0))
                leaf_value_weight = float(cfg.policy.get('intrinsic_tree_leaf_value_weight', 1.0))
                leaf_value_source = str(
                    cfg.policy.get('intrinsic_tree_leaf_value_source', 'default') or 'default'
                ).lower()
                if leaf_value_source == 'default':
                    leaf_value_source = (
                        'intrinsic_value'
                        if bool(cfg.policy.get('dynamics_disagreement_train_intrinsic_value', False)) else 'none'
                    )
                if leaf_value_source not in {'none', 'rnd', 'intrinsic_value'}:
                    raise ValueError(
                        "intrinsic_tree_leaf_value_source must be one of "
                        "'default', 'none', 'rnd', or 'intrinsic_value', "
                        f"got {leaf_value_source!r}."
                    )
                leaf_bootstrap = reward_sum.new_zeros((batch_size,))
                if (
                        (
                            leaf_value_source == 'intrinsic_value'
                            or (
                                leaf_value_source == 'rnd'
                                and bool(cfg.policy.get('rnd_exploration', False))
                                and bool(cfg.policy.get('rnd_use_learned_intrinsic_model', True))
                            )
                        )
                        and hasattr(model, 'intrinsic_value_from_latent')
                ):
                    leaf_values = model.intrinsic_value_from_latent(
                        latent_state, member_id=member_ids
                    ).squeeze(-1)
                    leaf_values = leaf_values.reshape(ensemble_size, batch_size).mean(dim=0)
                    leaf_bootstrap = leaf_values.detach()
                leaf_bootstrap_clipped = leaf_bootstrap
                leaf_clip = cfg.policy.get('intrinsic_tree_leaf_value_clip', None)
                if leaf_clip is not None:
                    clip_value = abs(float(leaf_clip))
                    leaf_bootstrap_clipped = leaf_bootstrap_clipped.clamp(min=-clip_value, max=clip_value)
                leaf_bootstrap_discount = discount_factor ** len(actions)
                reward_weighted_sum = immediate_weight * reward_sum
                leaf_bootstrap_weighted_discounted = (
                    leaf_bootstrap_discount * leaf_value_weight * leaf_bootstrap_clipped
                )
                intrinsic_return = reward_weighted_sum + leaf_bootstrap_weighted_discounted

                reward_tensor_np = reward_tensor.detach().cpu().numpy()
                reward_pair_std_np = reward_pair_std_tensor.detach().cpu().numpy()
                reward_sum_np = reward_sum.detach().cpu().numpy()
                reward_weighted_sum_np = reward_weighted_sum.detach().cpu().numpy()
                leaf_bootstrap_np = leaf_bootstrap.detach().cpu().numpy()
                leaf_bootstrap_clipped_np = leaf_bootstrap_clipped.detach().cpu().numpy()
                leaf_bootstrap_weighted_discounted_np = (
                    leaf_bootstrap_weighted_discounted.detach().cpu().numpy()
                )
                intrinsic_return_np = intrinsic_return.detach().cpu().numpy()
                decomposition_np = {}
                if full_disagreement_terms:
                    for key, terms in [
                            ('full_disagreement', full_disagreement_terms),
                            ('trainable_disagreement', trainable_disagreement_terms),
                            ('prior_disagreement', prior_disagreement_terms),
                            ('full_raw_disagreement', full_raw_disagreement_terms),
                            ('trainable_raw_disagreement', trainable_raw_disagreement_terms),
                            ('prior_raw_disagreement', prior_raw_disagreement_terms),
                            ('train_prior_pair_cosine', train_prior_pair_cosine_terms),
                            ('raw_trainable_norm', raw_trainable_norm_terms),
                            ('trainable_norm', trainable_norm_terms),
                            ('raw_prior_norm', raw_prior_norm_terms),
                            ('prior_norm', prior_norm_terms),
                            ('full_norm', full_norm_terms),
                    ]:
                        tensor = torch.stack(terms, dim=1)
                        decomposition_np[key] = tensor.detach().cpu().numpy()
                        if key not in {'train_prior_pair_cosine', 'raw_trainable_norm', 'trainable_norm',
                                       'raw_prior_norm', 'prior_norm', 'full_norm'}:
                            decomposition_np[f'{key}_sum'] = (tensor * discounts).sum(dim=1).detach().cpu().numpy()
                projection_np = {}
                if projection_disagreement_terms:
                    for key, terms in [
                            ('projection_disagreement', projection_disagreement_terms),
                            ('prediction_projection_disagreement', prediction_projection_disagreement_terms),
                            ('projection_raw_disagreement', projection_raw_disagreement_terms),
                            ('prediction_projection_raw_disagreement', prediction_projection_raw_disagreement_terms),
                            ('projection_norm', projection_norm_terms),
                            ('prediction_projection_norm', prediction_projection_norm_terms),
                    ]:
                        tensor = torch.stack(terms, dim=1)
                        projection_np[key] = tensor.detach().cpu().numpy()
                        if key not in {'projection_norm', 'prediction_projection_norm'}:
                            projection_np[f'{key}_sum'] = (tensor * discounts).sum(dim=1).detach().cpu().numpy()
                for state_index, state in enumerate(states):
                    step_rewards = reward_tensor_np[state_index].astype(np.float32)
                    step_pair_stds = reward_pair_std_np[state_index].astype(np.float32)
                    discounted_return = float(reward_sum_np[state_index])
                    weighted_reward_return = float(reward_weighted_sum_np[state_index])
                    bootstrap_raw = float(leaf_bootstrap_np[state_index])
                    bootstrap_clipped = float(leaf_bootstrap_clipped_np[state_index])
                    bootstrap_weighted_discounted = float(leaf_bootstrap_weighted_discounted_np[state_index])
                    combined_intrinsic_return = float(intrinsic_return_np[state_index])
                    true_sequence_metrics = _simulate_mountain_car_sequence(
                        _finite_float(state.get('obs_position')),
                        _finite_float(state.get('obs_velocity')),
                        actions,
                    )
                    row = {
                        'collection_phase': state['collection_phase'],
                        'train_iter_at_collect': state['train_iter_at_collect'],
                        'envstep_before_collect': state['envstep_before_collect'],
                        'envstep_after_collect': state['envstep_after_collect'],
                        'segment_index_in_collect': state['segment_index_in_collect'],
                        'env_id': state['env_id'],
                        'env_episode_index': state['env_episode_index'],
                        'collection_episode_index': state['collection_episode_index'],
                        'episode_segment_index': state['episode_segment_index'],
                        'episode_return': state['episode_return'],
                        'video_path': state['video_path'],
                        'timestep': state['timestep'],
                        'obs_index': state['obs_index'],
                        'selected_action': state['selected_action'],
                        'selected_action_direction': state['selected_action_direction'],
                        'obs_position': state['obs_position'],
                        'obs_velocity': state['obs_velocity'],
                        'intrinsic_reward_source': 'dynamics_disagreement',
                        'dynamics_disagreement_space': disagreement_space,
                        'dynamics_disagreement_distance_metric': str(
                            cfg.policy.get('dynamics_disagreement_distance_metric', 'mse')
                        ),
                        'sequence_name': sequence_name,
                        'sequence_actions': json.dumps(actions),
                        'sequence_length': len(actions),
                        'discount_factor': discount_factor,
                        'intrinsic_tree_immediate_reward_weight': immediate_weight,
                        'intrinsic_tree_leaf_value_source': leaf_value_source,
                        'intrinsic_tree_leaf_value_weight': leaf_value_weight,
                        'intrinsic_tree_leaf_value_clip': cfg.policy.get('intrinsic_tree_leaf_value_clip', None),
                        'pred_intrinsic_first_reward_mean': float(step_rewards[0]),
                        'pred_intrinsic_reward_sum_mean': discounted_return,
                        'pred_intrinsic_reward_weighted_sum_mean': weighted_reward_return,
                        'pred_intrinsic_reward_sum_std': float(step_rewards.std()),
                        'pred_intrinsic_bootstrap_mean': bootstrap_raw,
                        'pred_intrinsic_bootstrap_clipped_mean': bootstrap_clipped,
                        'pred_intrinsic_bootstrap_std': 0.0,
                        'pred_intrinsic_bootstrap_discounted_mean': bootstrap_weighted_discounted,
                        'pred_intrinsic_return_mean': combined_intrinsic_return,
                        'pred_intrinsic_return_std': 0.0,
                        'member_intrinsic_returns': json.dumps(_jsonable([combined_intrinsic_return])),
                        'member_intrinsic_reward_sums': json.dumps(_jsonable([discounted_return])),
                        'member_intrinsic_bootstrap_values': json.dumps(_jsonable([bootstrap_raw])),
                        'sequence_step_intrinsic_rewards': json.dumps(_jsonable(step_rewards)),
                        'sequence_step_pair_std': json.dumps(_jsonable(step_pair_stds)),
                        'future_max_position_20': state['future_max_position_20'],
                        'future_max_position_50': state['future_max_position_50'],
                        'future_max_position_100': state['future_max_position_100'],
                        'future_reached_goal_100': state['future_reached_goal_100'],
                        'future_reached_goal_remaining': state['future_reached_goal_remaining'],
                        **true_sequence_metrics,
                    }
                    if decomposition_np:
                        row.update(
                            {
                                'pred_full_dynamics_disagreement_sum_mean': float(
                                    decomposition_np['full_disagreement_sum'][state_index]
                                ),
                                'pred_trainable_dynamics_disagreement_sum_mean': float(
                                    decomposition_np['trainable_disagreement_sum'][state_index]
                                ),
                                'pred_prior_dynamics_disagreement_sum_mean': float(
                                    decomposition_np['prior_disagreement_sum'][state_index]
                                ),
                                'pred_full_dynamics_raw_disagreement_sum_mean': float(
                                    decomposition_np['full_raw_disagreement_sum'][state_index]
                                ),
                                'pred_trainable_dynamics_raw_disagreement_sum_mean': float(
                                    decomposition_np['trainable_raw_disagreement_sum'][state_index]
                                ),
                                'pred_prior_dynamics_raw_disagreement_sum_mean': float(
                                    decomposition_np['prior_raw_disagreement_sum'][state_index]
                                ),
                                'sequence_step_full_dynamics_disagreement': json.dumps(
                                    _jsonable(decomposition_np['full_disagreement'][state_index].astype(np.float32))
                                ),
                                'sequence_step_trainable_dynamics_disagreement': json.dumps(
                                    _jsonable(
                                        decomposition_np['trainable_disagreement'][state_index].astype(np.float32)
                                    )
                                ),
                                'sequence_step_prior_dynamics_disagreement': json.dumps(
                                    _jsonable(decomposition_np['prior_disagreement'][state_index].astype(np.float32))
                                ),
                                'sequence_step_full_dynamics_raw_disagreement': json.dumps(
                                    _jsonable(
                                        decomposition_np['full_raw_disagreement'][state_index].astype(np.float32)
                                    )
                                ),
                                'sequence_step_trainable_dynamics_raw_disagreement': json.dumps(
                                    _jsonable(
                                        decomposition_np['trainable_raw_disagreement'][state_index].astype(np.float32)
                                    )
                                ),
                                'sequence_step_prior_dynamics_raw_disagreement': json.dumps(
                                    _jsonable(
                                        decomposition_np['prior_raw_disagreement'][state_index].astype(np.float32)
                                    )
                                ),
                                'sequence_step_train_prior_pair_cosine': json.dumps(
                                    _jsonable(
                                        decomposition_np['train_prior_pair_cosine'][state_index].astype(np.float32)
                                    )
                                ),
                                'sequence_train_prior_pair_cosine_mean': float(
                                    decomposition_np['train_prior_pair_cosine'][state_index].mean()
                                ),
                                'sequence_step_raw_trainable_l2_norm': json.dumps(
                                    _jsonable(decomposition_np['raw_trainable_norm'][state_index].astype(np.float32))
                                ),
                                'sequence_step_trainable_l2_norm': json.dumps(
                                    _jsonable(decomposition_np['trainable_norm'][state_index].astype(np.float32))
                                ),
                                'sequence_step_raw_prior_l2_norm': json.dumps(
                                    _jsonable(decomposition_np['raw_prior_norm'][state_index].astype(np.float32))
                                ),
                                'sequence_step_prior_l2_norm': json.dumps(
                                    _jsonable(decomposition_np['prior_norm'][state_index].astype(np.float32))
                                ),
                                'sequence_step_full_l2_norm': json.dumps(
                                    _jsonable(decomposition_np['full_norm'][state_index].astype(np.float32))
                                ),
                                'sequence_raw_trainable_l2_norm_mean': float(
                                    decomposition_np['raw_trainable_norm'][state_index].mean()
                                ),
                                'sequence_trainable_l2_norm_mean': float(
                                    decomposition_np['trainable_norm'][state_index].mean()
                                ),
                                'sequence_raw_prior_l2_norm_mean': float(
                                    decomposition_np['raw_prior_norm'][state_index].mean()
                                ),
                                'sequence_prior_l2_norm_mean': float(
                                    decomposition_np['prior_norm'][state_index].mean()
                                ),
                                'sequence_full_l2_norm_mean': float(
                                    decomposition_np['full_norm'][state_index].mean()
                                ),
                            }
                        )
                    if projection_np:
                        row.update(
                            {
                                'pred_projection_disagreement_sum_mean': float(
                                    projection_np['projection_disagreement_sum'][state_index]
                                ),
                                'pred_prediction_projection_disagreement_sum_mean': float(
                                    projection_np['prediction_projection_disagreement_sum'][state_index]
                                ),
                                'pred_projection_raw_disagreement_sum_mean': float(
                                    projection_np['projection_raw_disagreement_sum'][state_index]
                                ),
                                'pred_prediction_projection_raw_disagreement_sum_mean': float(
                                    projection_np['prediction_projection_raw_disagreement_sum'][state_index]
                                ),
                                'sequence_step_projection_disagreement': json.dumps(
                                    _jsonable(projection_np['projection_disagreement'][state_index].astype(np.float32))
                                ),
                                'sequence_step_prediction_projection_disagreement': json.dumps(
                                    _jsonable(
                                        projection_np['prediction_projection_disagreement'][state_index].astype(
                                            np.float32
                                        )
                                    )
                                ),
                                'sequence_step_projection_raw_disagreement': json.dumps(
                                    _jsonable(
                                        projection_np['projection_raw_disagreement'][state_index].astype(np.float32)
                                    )
                                ),
                                'sequence_step_prediction_projection_raw_disagreement': json.dumps(
                                    _jsonable(
                                        projection_np['prediction_projection_raw_disagreement'][state_index].astype(
                                            np.float32
                                        )
                                    )
                                ),
                                'sequence_step_projection_l2_norm': json.dumps(
                                    _jsonable(projection_np['projection_norm'][state_index].astype(np.float32))
                                ),
                                'sequence_step_prediction_projection_l2_norm': json.dumps(
                                    _jsonable(
                                        projection_np['prediction_projection_norm'][state_index].astype(np.float32)
                                    )
                                ),
                                'sequence_projection_l2_norm_mean': float(
                                    projection_np['projection_norm'][state_index].mean()
                                ),
                                'sequence_prediction_projection_l2_norm_mean': float(
                                    projection_np['prediction_projection_norm'][state_index].mean()
                                ),
                            }
                        )
                    rows.append(row)
    finally:
        _restore_model_member_state(model, active_member_state)
        model.train(was_training)
    return rows


def _write_fixed_sequence_intrinsic_diagnostics(
        cfg,
        policy,
        new_data: List[Any],
        tb_logger,
        collection_phase: int,
        diagnostic_stage: str = 'pre_learn',
) -> None:
    if not _diagnostics_enabled(cfg) or not bool(cfg.policy.get('fixed_sequence_intrinsic_diagnostics', False)):
        return
    intrinsic_reward_source = _intrinsic_reward_source(cfg)
    if intrinsic_reward_source == 'rnd' and (
            not bool(cfg.policy.get('rnd_exploration', False))
            or not bool(cfg.policy.get('rnd_use_learned_intrinsic_model', True))
    ):
        return
    if intrinsic_reward_source == 'dynamics_disagreement' and (
            not bool(cfg.policy.get('dynamics_disagreement_exploration', False))
    ):
        return
    if intrinsic_reward_source == 'qstd' and (
            not bool(cfg.policy.get('qstd_exploration', False))
    ):
        return

    model = getattr(policy, '_collect_model', None)
    if model is None:
        model = getattr(policy, '_learn_model', None)
    if model is None:
        return

    states = _sample_fixed_intrinsic_states(cfg, new_data)
    sequences = _fixed_intrinsic_sequences(cfg)
    if intrinsic_reward_source == 'dynamics_disagreement':
        rows = _probe_fixed_sequence_dynamics_disagreement_returns(cfg, model, states, sequences)
    else:
        rows = _probe_fixed_sequence_intrinsic_returns(cfg, model, states, sequences, policy=policy)
    if not rows:
        return
    for row in rows:
        row['diagnostic_stage'] = str(diagnostic_stage)

    diagnostics_dir = _diagnostics_dir(cfg)
    csv_path = os.path.join(diagnostics_dir, 'fixed_sequence_intrinsic_returns.csv')
    fieldnames = list(rows[0].keys())
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    sequence_means = {}
    sequence_true_max_positions = {}
    sequence_true_rnd_means = {}
    for sequence_name, _ in sequences:
        sequence_rows = [row for row in rows if row['sequence_name'] == sequence_name]
        if sequence_rows:
            sequence_means[sequence_name] = float(
                np.mean([float(row['pred_intrinsic_return_mean']) for row in sequence_rows])
            )
            if 'true_sequence_max_position' in sequence_rows[0]:
                sequence_true_max_positions[sequence_name] = float(
                    np.mean([float(row['true_sequence_max_position']) for row in sequence_rows])
                )
            if 'true_rnd_discounted_reward_sum' in sequence_rows[0]:
                sequence_true_rnd_means[sequence_name] = float(
                    np.mean([float(row['true_rnd_discounted_reward_sum']) for row in sequence_rows])
                )

    predicted_step_rewards = []
    true_step_rewards = []
    for row in rows:
        if 'sequence_step_true_rnd_rewards' not in row:
            continue
        predicted_step_rewards.extend(json.loads(row['sequence_step_intrinsic_rewards']))
        true_step_rewards.extend(json.loads(row['sequence_step_true_rnd_rewards']))
    predicted_step_rewards = np.asarray(predicted_step_rewards, dtype=np.float64)
    true_step_rewards = np.asarray(true_step_rewards, dtype=np.float64)
    finite_mask = np.isfinite(predicted_step_rewards) & np.isfinite(true_step_rewards)
    if finite_mask.sum() >= 2:
        predicted_step_rewards = predicted_step_rewards[finite_mask]
        true_step_rewards = true_step_rewards[finite_mask]
        step_reward_mae = float(np.mean(np.abs(predicted_step_rewards - true_step_rewards)))
        if np.std(predicted_step_rewards) > 1e-12 and np.std(true_step_rewards) > 1e-12:
            step_reward_correlation = float(np.corrcoef(predicted_step_rewards, true_step_rewards)[0, 1])
        else:
            step_reward_correlation = 0.0
    else:
        step_reward_mae = float('nan')
        step_reward_correlation = float('nan')

    imagined_endpoint_values = np.asarray(
        [row['pred_intrinsic_bootstrap_mean'] for row in rows], dtype=np.float64
    )
    direct_endpoint_values = np.asarray(
        [row['direct_true_endpoint_intrinsic_value_mean'] for row in rows], dtype=np.float64
    )
    finite_value_mask = np.isfinite(imagined_endpoint_values) & np.isfinite(direct_endpoint_values)
    if finite_value_mask.any():
        finite_imagined_values = imagined_endpoint_values[finite_value_mask]
        finite_direct_values = direct_endpoint_values[finite_value_mask]
        intrinsic_value_consistency_mae = float(
            np.mean(np.abs(finite_imagined_values - finite_direct_values))
        )
        if (
                finite_value_mask.sum() >= 2
                and np.std(finite_imagined_values) > 1e-12
                and np.std(finite_direct_values) > 1e-12
        ):
            intrinsic_value_consistency_correlation = float(
                np.corrcoef(finite_imagined_values, finite_direct_values)[0, 1]
            )
        else:
            intrinsic_value_consistency_correlation = 0.0
    else:
        intrinsic_value_consistency_mae = float('nan')
        intrinsic_value_consistency_correlation = float('nan')
    logging.info(
        "[FIXED_INTRINSIC_DIAG] phase %s %s wrote %s rows from %s states to %s | "
        "returns %s | true_rnd %s | reward_mae %.4f reward_corr %.4f | "
        "value_consistency_mae %.5f value_consistency_corr %.4f | true_max_pos %s",
        collection_phase,
        diagnostic_stage,
        len(rows),
        len(states),
        csv_path,
        ", ".join(f"{name}={value:.3f}" for name, value in sequence_means.items()),
        ", ".join(f"{name}={value:.3f}" for name, value in sequence_true_rnd_means.items()),
        step_reward_mae,
        step_reward_correlation,
        intrinsic_value_consistency_mae,
        intrinsic_value_consistency_correlation,
        ", ".join(f"{name}={value:.3f}" for name, value in sequence_true_max_positions.items()),
    )
    if tb_logger is not None:
        tb_logger.add_scalar('collection_phase/fixed_intrinsic_state_count', float(len(states)), collection_phase)
        safe_stage = str(diagnostic_stage).replace('/', '_').replace(' ', '_')
        if np.isfinite(step_reward_mae):
            tb_logger.add_scalar(
                f'fixed_intrinsic_sequence/{safe_stage}_rnd_reward_mae',
                step_reward_mae,
                collection_phase,
            )
            tb_logger.add_scalar(
                f'fixed_intrinsic_sequence/{safe_stage}_rnd_reward_correlation',
                step_reward_correlation,
                collection_phase,
            )
        if np.isfinite(intrinsic_value_consistency_mae):
            tb_logger.add_scalar(
                f'fixed_intrinsic_sequence/{safe_stage}_value_consistency_mae',
                intrinsic_value_consistency_mae,
                collection_phase,
            )
            tb_logger.add_scalar(
                f'fixed_intrinsic_sequence/{safe_stage}_value_consistency_correlation',
                intrinsic_value_consistency_correlation,
                collection_phase,
            )
        for sequence_name, mean_return in sequence_means.items():
            safe_name = sequence_name.replace('/', '_').replace(' ', '_')
            tb_logger.add_scalar(
                f'fixed_intrinsic_sequence/{safe_name}_return_mean',
                float(mean_return),
                collection_phase,
            )
            if sequence_name in sequence_true_max_positions:
                tb_logger.add_scalar(
                    f'fixed_intrinsic_sequence/{safe_name}_true_max_position_mean',
                    float(sequence_true_max_positions[sequence_name]),
                    collection_phase,
                )


class EliteEpisodeReplayBuffer:
    """
    Keep the top-return completed game segments and sample roots from them.

    For MountainCar with ``game_segment_length >= episode_limit`` one segment is
    one full episode, which is the use case this lightweight helper targets.
    """

    def __init__(self, cfg, exp_name: Optional[str] = None) -> None:
        self.enabled = bool(cfg.get('elite_episode_replay', False))
        self.capacity = max(0, int(cfg.get('elite_episode_buffer_size', 20)))
        self.sample_ratio = max(0.0, min(1.0, float(cfg.get('elite_episode_sample_ratio', 0.25))))
        self.min_return = cfg.get('elite_episode_min_return', None)
        self.monte_carlo_value_target = bool(cfg.get('elite_episode_monte_carlo_value_target', True))
        self.start_stratification = bool(cfg.get('elite_episode_start_stratification', False))
        self.start_observation_index = int(cfg.get('elite_episode_start_observation_index', 0))
        start_bin_edges = cfg.get('elite_episode_start_bin_edges', None)
        if isinstance(start_bin_edges, str):
            start_bin_edges = [float(value.strip()) for value in start_bin_edges.split(',') if value.strip()]
        self.start_bin_edges = tuple(float(value) for value in (start_bin_edges or []))
        if any(not np.isfinite(value) for value in self.start_bin_edges):
            raise ValueError("Elite replay start-bin edges must be finite")
        if any(left >= right for left, right in zip(self.start_bin_edges, self.start_bin_edges[1:])):
            raise ValueError("Elite replay start-bin edges must be strictly increasing")
        self.num_start_bins = len(self.start_bin_edges) + 1 if self.start_stratification else 0
        if self.start_stratification and not self.start_bin_edges:
            raise ValueError("Elite replay start stratification requires at least one bin edge")
        if self.start_stratification and self.capacity < self.num_start_bins:
            raise ValueError(
                "Elite replay capacity must be at least the number of start bins: "
                f"capacity={self.capacity}, bins={self.num_start_bins}"
            )
        self.save_trajectories = bool(cfg.get('save_elite_episode_trajectories', True))
        self.save_dir = cfg.get('elite_episode_save_dir', None)
        if self.save_dir is None and exp_name is not None:
            self.save_dir = os.path.join('./', exp_name, 'diagnostics', 'elite_episodes')
        self._cfg = cfg
        self._entries: List[dict] = []
        self._seen_segment_ids = set()
        self._next_entry_id = 0
        self.last_sample_count = 0
        self.last_mc_value_target_mean = 0.0
        self.last_mc_value_target_min = 0.0
        self.last_mc_value_target_max = 0.0
        self.last_sample_bin_counts = {bin_id: 0 for bin_id in range(self.num_start_bins)}

    def __len__(self) -> int:
        return len(self._entries)

    @staticmethod
    def _scalar_reward(reward: Any) -> float:
        if isinstance(reward, (list, tuple, np.ndarray)):
            reward = np.asarray(reward).reshape(-1)[0]
        return float(reward)

    def _segment_return(self, game_segment: Any) -> float:
        valid_len = min(len(game_segment), int(self._cfg.game_segment_length))
        rewards = game_segment.reward_segment[:valid_len]
        return float(sum(self._scalar_reward(reward) for reward in rewards))

    def _start_value_and_bin(self, game_segment: Any) -> Tuple[Optional[float], Optional[int]]:
        if not self.start_stratification:
            return None, None
        observations = getattr(game_segment, 'obs_segment', None)
        if observations is None or len(observations) == 0:
            raise ValueError("Stratified elite replay requires a non-empty game_segment.obs_segment")
        start_observation = np.asarray(observations[0]).reshape(-1)
        if not 0 <= self.start_observation_index < start_observation.size:
            raise ValueError(
                "Elite replay start observation index is out of range: "
                f"index={self.start_observation_index}, observation_shape={np.asarray(observations[0]).shape}"
            )
        start_value = float(start_observation[self.start_observation_index])
        if not np.isfinite(start_value):
            raise ValueError(f"Elite replay start observation value must be finite, got {start_value}")
        return start_value, int(np.searchsorted(self.start_bin_edges, start_value, side='right'))

    def _start_bin_capacity(self, bin_id: int) -> int:
        base_capacity, remainder = divmod(self.capacity, self.num_start_bins)
        return base_capacity + int(bin_id < remainder)

    def _entries_in_start_bin(self, bin_id: int) -> List[dict]:
        return [entry for entry in self._entries if entry.get('start_bin') == bin_id]

    def _discounted_return_from(self, rewards: List[Any], start_index: int, valid_len: int) -> float:
        if start_index >= valid_len:
            return 0.0
        discount = 1.0
        total_return = 0.0
        discount_factor = float(self._cfg.discount_factor)
        for reward in rewards[start_index:valid_len]:
            total_return += discount * self._scalar_reward(reward)
            discount *= discount_factor
        return float(total_return)

    def _make_monte_carlo_value_targets(self, entries: List[dict], positions: List[int]) -> np.ndarray:
        batch_targets = []
        num_unroll_steps = int(self._cfg.num_unroll_steps)
        for entry, pos_in_game_segment in zip(entries, positions):
            valid_len = int(entry['length'])
            rewards = entry['segment'].reward_segment
            target_values = [
                self._discounted_return_from(rewards, pos_in_game_segment + unroll_offset, valid_len)
                for unroll_offset in range(num_unroll_steps + 1)
            ]
            batch_targets.append(target_values)
        return np.asarray(batch_targets, dtype=np.float32)

    def _override_value_targets_with_monte_carlo(
            self,
            train_data: List[Any],
            entries: List[dict],
            positions: List[int],
    ) -> List[Any]:
        if not self.monte_carlo_value_target:
            return train_data
        mc_target_values = self._make_monte_carlo_value_targets(entries, positions)
        target_values = np.asarray(train_data[1][1])
        if target_values.shape != mc_target_values.shape:
            raise RuntimeError(
                "Elite Monte Carlo value target shape mismatch: "
                f"target_values={target_values.shape}, mc_target_values={mc_target_values.shape}"
            )
        train_data[1][1] = mc_target_values.astype(target_values.dtype, copy=False)
        if len(train_data[1]) > 3:
            member_target_values = np.asarray(train_data[1][3])
            expected_shape = (member_target_values.shape[0],) + mc_target_values.shape
            if member_target_values.ndim < 3 or member_target_values.shape != expected_shape:
                raise RuntimeError(
                    "Elite Monte Carlo member value target shape mismatch: "
                    f"member_target_values={member_target_values.shape}, expected={expected_shape}"
                )
            train_data[1][3] = np.broadcast_to(
                mc_target_values[None, :, :], member_target_values.shape
            ).astype(member_target_values.dtype, copy=True)
        self.last_mc_value_target_mean = float(mc_target_values.mean())
        self.last_mc_value_target_min = float(mc_target_values.min())
        self.last_mc_value_target_max = float(mc_target_values.max())
        return train_data

    def _safe_return_for_filename(self, episode_return: float) -> str:
        return f"{episode_return:.1f}".replace('-', 'm').replace('.', 'p')

    def _save_entry(self, entry: dict, meta: dict) -> None:
        if not self.save_trajectories or self.save_dir is None:
            return
        os.makedirs(self.save_dir, exist_ok=True)
        safe_return = self._safe_return_for_filename(float(entry['return']))
        start_bin_label = (
            f"start-bin-{entry['start_bin']}_" if self.start_stratification else ""
        )
        filename_stem = (
            f"elite-{int(entry['id']):04d}_return-{safe_return}_"
            f"{start_bin_label}"
            f"phase-{meta.get('collection_phase', 'unknown')}_"
            f"env-{meta.get('env_id', 'unknown')}_"
            f"episode-{meta.get('env_episode_index', 'unknown')}"
        )
        npz_path = os.path.join(self.save_dir, f'{filename_stem}.npz')
        json_path = os.path.join(self.save_dir, f'{filename_stem}.json')
        segment = entry['segment']
        priorities = meta.get('priorities', None)
        priorities = np.asarray(priorities) if priorities is not None else np.asarray([])
        np.savez_compressed(
            npz_path,
            obs=np.asarray(segment.obs_segment),
            actions=np.asarray(segment.action_segment),
            rewards=np.asarray(segment.reward_segment),
            intrinsic_rewards=np.asarray(getattr(segment, 'intrinsic_reward_segment', []), dtype=np.float32),
            intrinsic_reward_raws=np.asarray(getattr(segment, 'intrinsic_reward_raw_segment', []), dtype=np.float32),
            child_visits=np.asarray(segment.child_visit_segment),
            root_values=np.asarray(segment.root_value_segment),
            action_masks=np.asarray(segment.action_mask_segment),
            timesteps=np.asarray(segment.timestep_segment),
            priorities=priorities,
        )
        json_metadata = dict(meta)
        json_metadata.pop('priorities', None)
        with open(json_path, 'w') as f:
            json.dump(
                _jsonable(
                    {
                        'elite_id': entry['id'],
                        'episode_return': entry['return'],
                        'length': entry['length'],
                        'metadata': json_metadata,
                        'npz_path': npz_path,
                    }
                ),
                f,
                indent=2,
                sort_keys=True,
            )
        logging.info("[ELITE_REPLAY_SAVE] saved elite episode %s to %s", entry['id'], npz_path)

    def _maybe_add_segment(self, game_segment: Any, meta: dict) -> bool:
        if not self.enabled or self.capacity <= 0:
            return False
        if not meta.get('done', False):
            return False
        segment_id = id(game_segment)
        if segment_id in self._seen_segment_ids:
            return False

        episode_return = self._segment_return(game_segment)
        if self.min_return is not None and episode_return < float(self.min_return):
            return False
        start_value, start_bin = self._start_value_and_bin(game_segment)
        if self.start_stratification:
            bin_entries = self._entries_in_start_bin(start_bin)
            bin_capacity = self._start_bin_capacity(start_bin)
            if len(bin_entries) >= bin_capacity:
                worst_bin_entry = min(bin_entries, key=lambda item: (item['return'], -item['id']))
                if episode_return <= worst_bin_entry['return']:
                    return False
        elif len(self._entries) >= self.capacity and episode_return <= self._entries[-1]['return']:
            return False

        self._seen_segment_ids.add(segment_id)
        entry_meta = dict(meta)
        if self.start_stratification:
            entry_meta['elite_start_value'] = start_value
            entry_meta['elite_start_bin'] = start_bin
        entry = {
            'id': self._next_entry_id,
            'segment': game_segment,
            'return': episode_return,
            'length': min(len(game_segment), int(self._cfg.game_segment_length)),
            'metadata': entry_meta,
            'start_value': start_value,
            'start_bin': start_bin,
        }
        self._entries.append(entry)
        self._next_entry_id += 1
        self._entries.sort(key=lambda entry: (entry['return'], entry['id']), reverse=True)
        if self.start_stratification:
            bin_entries = self._entries_in_start_bin(start_bin)
            bin_capacity = self._start_bin_capacity(start_bin)
            if len(bin_entries) > bin_capacity:
                worst_bin_entry = min(bin_entries, key=lambda item: (item['return'], -item['id']))
                self._entries.remove(worst_bin_entry)
        else:
            del self._entries[self.capacity:]
        self._seen_segment_ids = {id(entry['segment']) for entry in self._entries}
        self._save_entry(entry, entry_meta)
        return True

    def update(self, data_and_meta: Any) -> int:
        if not self.enabled:
            return 0
        data, meta = data_and_meta
        added = 0
        for game_segment, meta_game in zip(data, meta):
            if self._maybe_add_segment(game_segment, meta_game):
                added += 1
        return added

    def sample_train_data(self, batch_size: int, replay_buffer, policy):
        self.last_sample_count = 0
        self.last_sample_bin_counts = {bin_id: 0 for bin_id in range(self.num_start_bins)}
        if not self.enabled or batch_size <= 0 or not self._entries:
            return None

        game_segment_list = []
        pos_in_game_segment_list = []
        batch_index_list = []
        weights_list = []
        make_time_list = []
        sampled_entries = []

        if self.start_stratification:
            populated_bins = [
                bin_id for bin_id in range(self.num_start_bins) if self._entries_in_start_bin(bin_id)
            ]
            sampled_bin_ids = [populated_bins[index % len(populated_bins)] for index in range(batch_size)]
            np.random.shuffle(sampled_bin_ids)
            sampled_entry_pool = [
                self._entries_in_start_bin(bin_id)[
                    np.random.randint(0, len(self._entries_in_start_bin(bin_id)))
                ]
                for bin_id in sampled_bin_ids
            ]
        else:
            sampled_entry_pool = [
                self._entries[np.random.randint(0, len(self._entries))] for _ in range(batch_size)
            ]

        for entry in sampled_entry_pool:
            if self.start_stratification:
                self.last_sample_bin_counts[int(entry['start_bin'])] += 1
            segment_len = max(1, int(entry['length']))
            if segment_len > 1:
                pos_in_game_segment = int(np.random.randint(0, segment_len - 1))
            else:
                pos_in_game_segment = 0

            game_segment_list.append(entry['segment'])
            pos_in_game_segment_list.append(pos_in_game_segment)
            batch_index_list.append(f"elite:{entry['id']}:{pos_in_game_segment}")
            weights_list.append(1.0)
            make_time_list.append(0.0)
            sampled_entries.append(entry)

        self.last_sample_count = len(batch_index_list)
        orig_data = (
            game_segment_list,
            pos_in_game_segment_list,
            batch_index_list,
            weights_list,
            make_time_list,
        )
        train_data = replay_buffer.sample_from_orig_data(orig_data, policy)
        return self._override_value_targets_with_monte_carlo(
            train_data, sampled_entries, pos_in_game_segment_list
        )

    def stats(self) -> dict:
        if not self._entries:
            stats = {
                'elite_replay/size': 0.0,
                'elite_replay/best_return': 0.0,
                'elite_replay/worst_return': 0.0,
                'elite_replay/mean_return': 0.0,
                'elite_replay/last_sample_count': float(self.last_sample_count),
                'elite_replay/mc_value_target_enabled': float(self.monte_carlo_value_target),
                'elite_replay/last_mc_value_target_mean': float(self.last_mc_value_target_mean),
                'elite_replay/last_mc_value_target_min': float(self.last_mc_value_target_min),
                'elite_replay/last_mc_value_target_max': float(self.last_mc_value_target_max),
            }
        else:
            returns = np.asarray([entry['return'] for entry in self._entries], dtype=np.float32)
            stats = {
                'elite_replay/size': float(len(self._entries)),
                'elite_replay/best_return': float(returns.max()),
                'elite_replay/worst_return': float(returns.min()),
                'elite_replay/mean_return': float(returns.mean()),
                'elite_replay/last_sample_count': float(self.last_sample_count),
                'elite_replay/mc_value_target_enabled': float(self.monte_carlo_value_target),
                'elite_replay/last_mc_value_target_mean': float(self.last_mc_value_target_mean),
                'elite_replay/last_mc_value_target_min': float(self.last_mc_value_target_min),
                'elite_replay/last_mc_value_target_max': float(self.last_mc_value_target_max),
            }
        stats['elite_replay/start_stratification_enabled'] = float(self.start_stratification)
        stats['elite_replay/start_bin_count'] = float(self.num_start_bins)
        for bin_id in range(self.num_start_bins):
            bin_entries = self._entries_in_start_bin(bin_id)
            bin_returns = np.asarray([entry['return'] for entry in bin_entries], dtype=np.float32)
            bin_start_values = np.asarray([entry['start_value'] for entry in bin_entries], dtype=np.float32)
            prefix = f'elite_replay/start_bin_{bin_id}'
            stats[f'{prefix}/capacity'] = float(self._start_bin_capacity(bin_id))
            stats[f'{prefix}/size'] = float(len(bin_entries))
            stats[f'{prefix}/best_return'] = float(bin_returns.max()) if len(bin_entries) else 0.0
            stats[f'{prefix}/worst_return'] = float(bin_returns.min()) if len(bin_entries) else 0.0
            stats[f'{prefix}/mean_return'] = float(bin_returns.mean()) if len(bin_entries) else 0.0
            stats[f'{prefix}/start_value_min'] = float(bin_start_values.min()) if len(bin_entries) else 0.0
            stats[f'{prefix}/start_value_max'] = float(bin_start_values.max()) if len(bin_entries) else 0.0
            stats[f'{prefix}/last_sample_count'] = float(self.last_sample_bin_counts.get(bin_id, 0))
        return stats


class IntrinsicEpisodeReplayBuffer:
    """
    Keep completed episodes with the largest collection-time intrinsic reward.

    This is intentionally separate from return-based elite replay: it preserves
    rare novelty spikes even when the extrinsic episode return is still poor.
    """

    def __init__(self, cfg, exp_name: Optional[str] = None) -> None:
        self.enabled = bool(cfg.get('intrinsic_episode_replay', False))
        self.capacity = max(0, int(cfg.get('intrinsic_episode_buffer_size', 20)))
        self.sample_ratio = max(0.0, min(1.0, float(cfg.get('intrinsic_episode_sample_ratio', 0.25))))
        self.min_score = float(cfg.get('intrinsic_episode_min_score', 0.0))
        self.score_mode = str(cfg.get('intrinsic_episode_score_mode', 'sum') or 'sum')
        if self.score_mode not in ['sum', 'max']:
            raise ValueError("intrinsic_episode_score_mode must be 'sum' or 'max'.")
        self.peak_window_before = max(0, int(cfg.get('intrinsic_episode_peak_window_before', 120)))
        self.peak_window_after = max(0, int(cfg.get('intrinsic_episode_peak_window_after', 5)))
        self.max_age_phases = max(0, int(cfg.get('intrinsic_episode_max_age_phases', 0)))
        self.save_trajectories = bool(cfg.get('save_intrinsic_episode_trajectories', True))
        self.save_dir = cfg.get('intrinsic_episode_save_dir', None)
        if self.save_dir is None and exp_name is not None:
            self.save_dir = os.path.join('./', exp_name, 'diagnostics', 'intrinsic_episodes')
        self._cfg = cfg
        self._entries: List[dict] = []
        self._seen_segment_ids = set()
        self._next_entry_id = 0
        self.last_sample_count = 0
        self.last_sample_peak_distance_mean = 0.0

    def __len__(self) -> int:
        return len(self._entries)

    def _segment_intrinsic_rewards(self, game_segment: Any) -> np.ndarray:
        rewards = np.asarray(getattr(game_segment, 'intrinsic_reward_segment', []), dtype=np.float32).reshape(-1)
        valid_len = min(len(game_segment), int(self._cfg.game_segment_length), int(rewards.shape[0]))
        return rewards[:valid_len]

    def _segment_intrinsic_raw_rewards(self, game_segment: Any) -> np.ndarray:
        rewards = np.asarray(getattr(game_segment, 'intrinsic_reward_raw_segment', []), dtype=np.float32).reshape(-1)
        valid_len = min(len(game_segment), int(self._cfg.game_segment_length), int(rewards.shape[0]))
        return rewards[:valid_len]

    def _segment_score_info(self, game_segment: Any) -> Optional[dict]:
        rewards = self._segment_intrinsic_rewards(game_segment)
        if rewards.size == 0:
            return None
        reward_sum = float(np.sum(rewards))
        reward_max = float(np.max(rewards))
        peak_index = int(np.argmax(rewards))
        score = reward_sum if self.score_mode == 'sum' else reward_max
        return {
            'score': float(score),
            'reward_sum': reward_sum,
            'reward_max': reward_max,
            'peak_index': peak_index,
            'length': int(rewards.shape[0]),
        }

    def _safe_score_for_filename(self, score: float) -> str:
        return f"{score:.3f}".replace('-', 'm').replace('.', 'p')

    def _save_entry(self, entry: dict, meta: dict) -> None:
        if not self.save_trajectories or self.save_dir is None:
            return
        os.makedirs(self.save_dir, exist_ok=True)
        safe_score = self._safe_score_for_filename(float(entry['score']))
        filename_stem = (
            f"intrinsic-{int(entry['id']):04d}_score-{safe_score}_"
            f"phase-{meta.get('collection_phase', 'unknown')}_"
            f"env-{meta.get('env_id', 'unknown')}_"
            f"episode-{meta.get('env_episode_index', 'unknown')}"
        )
        npz_path = os.path.join(self.save_dir, f'{filename_stem}.npz')
        json_path = os.path.join(self.save_dir, f'{filename_stem}.json')
        segment = entry['segment']
        priorities = meta.get('priorities', None)
        priorities = np.asarray(priorities) if priorities is not None else np.asarray([])
        np.savez_compressed(
            npz_path,
            obs=np.asarray(segment.obs_segment),
            actions=np.asarray(segment.action_segment),
            rewards=np.asarray(segment.reward_segment),
            intrinsic_rewards=np.asarray(getattr(segment, 'intrinsic_reward_segment', []), dtype=np.float32),
            intrinsic_reward_raws=np.asarray(getattr(segment, 'intrinsic_reward_raw_segment', []), dtype=np.float32),
            child_visits=np.asarray(segment.child_visit_segment),
            root_values=np.asarray(segment.root_value_segment),
            action_masks=np.asarray(segment.action_mask_segment),
            timesteps=np.asarray(segment.timestep_segment),
            priorities=priorities,
        )
        json_metadata = dict(meta)
        json_metadata.pop('priorities', None)
        with open(json_path, 'w') as f:
            json.dump(
                _jsonable(
                    {
                        'intrinsic_id': entry['id'],
                        'score': entry['score'],
                        'score_mode': self.score_mode,
                        'intrinsic_reward_sum': entry['reward_sum'],
                        'intrinsic_reward_max': entry['reward_max'],
                        'peak_index': entry['peak_index'],
                        'length': entry['length'],
                        'metadata': json_metadata,
                        'npz_path': npz_path,
                    }
                ),
                f,
                indent=2,
                sort_keys=True,
            )
        logging.info("[INTRINSIC_REPLAY_SAVE] saved intrinsic episode %s to %s", entry['id'], npz_path)

    def _maybe_add_segment(self, game_segment: Any, meta: dict) -> bool:
        if not self.enabled or self.capacity <= 0:
            return False
        if not meta.get('done', False):
            return False
        segment_id = id(game_segment)
        if segment_id in self._seen_segment_ids:
            return False

        score_info = self._segment_score_info(game_segment)
        if score_info is None or score_info['score'] <= self.min_score:
            return False
        if len(self._entries) >= self.capacity and score_info['score'] <= self._entries[-1]['score']:
            return False

        self._seen_segment_ids.add(segment_id)
        entry = {
            'id': self._next_entry_id,
            'segment': game_segment,
            'metadata': dict(meta),
            **score_info,
        }
        self._entries.append(entry)
        self._next_entry_id += 1
        self._entries.sort(key=lambda item: (item['score'], item['id']), reverse=True)
        del self._entries[self.capacity:]
        self._seen_segment_ids = {id(item['segment']) for item in self._entries}
        self._save_entry(entry, meta)
        return True

    def update(self, data_and_meta: Any) -> int:
        if not self.enabled:
            return 0
        data, meta = data_and_meta
        if self.max_age_phases > 0 and meta:
            current_phase = max(
                int(meta_game.get('collection_phase', 0)) for meta_game in meta
            )
            minimum_phase = current_phase - self.max_age_phases + 1
            self._entries = [
                entry for entry in self._entries
                if int(entry['metadata'].get('collection_phase', 0)) >= minimum_phase
            ]
            self._seen_segment_ids = {id(entry['segment']) for entry in self._entries}
        added = 0
        for game_segment, meta_game in zip(data, meta):
            if self._maybe_add_segment(game_segment, meta_game):
                added += 1
        return added

    def _sample_position(self, entry: dict) -> int:
        segment_len = max(1, int(entry['length']))
        if segment_len <= 1:
            return 0
        peak_index = int(entry['peak_index'])
        low = max(0, peak_index - self.peak_window_before)
        high = min(segment_len - 1, peak_index + self.peak_window_after)
        if high < low:
            low, high = 0, segment_len - 1
        return int(np.random.randint(low, high + 1))

    def sample_train_data(self, batch_size: int, replay_buffer, policy):
        self.last_sample_count = 0
        self.last_sample_peak_distance_mean = 0.0
        if not self.enabled or batch_size <= 0 or not self._entries:
            return None

        game_segment_list = []
        pos_in_game_segment_list = []
        batch_index_list = []
        weights_list = []
        make_time_list = []
        peak_distances = []

        for _ in range(batch_size):
            entry = self._entries[np.random.randint(0, len(self._entries))]
            pos_in_game_segment = self._sample_position(entry)
            game_segment_list.append(entry['segment'])
            pos_in_game_segment_list.append(pos_in_game_segment)
            batch_index_list.append(f"intrinsic:{entry['id']}:{pos_in_game_segment}")
            weights_list.append(1.0)
            make_time_list.append(0.0)
            peak_distances.append(abs(pos_in_game_segment - int(entry['peak_index'])))

        self.last_sample_count = len(batch_index_list)
        if peak_distances:
            self.last_sample_peak_distance_mean = float(np.mean(peak_distances))
        orig_data = (
            game_segment_list,
            pos_in_game_segment_list,
            batch_index_list,
            weights_list,
            make_time_list,
        )
        return replay_buffer.sample_from_orig_data(orig_data, policy)

    def stats(self) -> dict:
        if not self._entries:
            return {
                'intrinsic_replay/size': 0.0,
                'intrinsic_replay/best_score': 0.0,
                'intrinsic_replay/worst_score': 0.0,
                'intrinsic_replay/mean_score': 0.0,
                'intrinsic_replay/best_reward_sum': 0.0,
                'intrinsic_replay/best_reward_max': 0.0,
                'intrinsic_replay/last_sample_count': float(self.last_sample_count),
                'intrinsic_replay/last_sample_peak_distance_mean': float(self.last_sample_peak_distance_mean),
                'intrinsic_replay/max_age_phases': float(self.max_age_phases),
            }
        scores = np.asarray([entry['score'] for entry in self._entries], dtype=np.float32)
        reward_sums = np.asarray([entry['reward_sum'] for entry in self._entries], dtype=np.float32)
        reward_maxes = np.asarray([entry['reward_max'] for entry in self._entries], dtype=np.float32)
        return {
            'intrinsic_replay/size': float(len(self._entries)),
            'intrinsic_replay/best_score': float(scores.max()),
            'intrinsic_replay/worst_score': float(scores.min()),
            'intrinsic_replay/mean_score': float(scores.mean()),
            'intrinsic_replay/best_reward_sum': float(reward_sums.max()),
            'intrinsic_replay/best_reward_max': float(reward_maxes.max()),
            'intrinsic_replay/last_sample_count': float(self.last_sample_count),
            'intrinsic_replay/last_sample_peak_distance_mean': float(self.last_sample_peak_distance_mean),
            'intrinsic_replay/max_age_phases': float(self.max_age_phases),
        }


def _concat_train_data_many(train_data_list: List[List[Any]]) -> List[Any]:
    train_data_list = [train_data for train_data in train_data_list if train_data is not None]
    if not train_data_list:
        return None
    current_batch = [
        np.concatenate([np.asarray(train_data[0][idx]) for train_data in train_data_list], axis=0)
        for idx in range(len(train_data_list[0][0]))
    ]
    target_batch = []
    for idx in range(len(train_data_list[0][1])):
        arrays = [np.asarray(train_data[1][idx]) for train_data in train_data_list]
        is_member_specific_target = (
            idx >= 3
            and arrays[0].ndim >= 3
            and all(array.ndim >= 3 and array.shape[0] == arrays[0].shape[0] for array in arrays)
        )
        concat_axis = 1 if is_member_specific_target else 0
        target_batch.append(np.concatenate(arrays, axis=concat_axis))
    return [current_batch, target_batch]


def _concat_train_data(normal_train_data, elite_train_data):
    return _concat_train_data_many([normal_train_data, elite_train_data])


def _maybe_mix_elite_train_data(
        replay_buffer,
        elite_replay_buffer: EliteEpisodeReplayBuffer,
        intrinsic_replay_buffer: Optional[IntrinsicEpisodeReplayBuffer],
        batch_size: int,
        policy,
) -> Tuple[Optional[List[Any]], Optional[List[Any]], int, int, int]:
    elite_available = (
        elite_replay_buffer.enabled
        and len(elite_replay_buffer) > 0
        and elite_replay_buffer.sample_ratio > 0.0
    )
    intrinsic_available = (
        intrinsic_replay_buffer is not None
        and intrinsic_replay_buffer.enabled
        and len(intrinsic_replay_buffer) > 0
        and intrinsic_replay_buffer.sample_ratio > 0.0
    )
    if (not elite_available and not intrinsic_available) or batch_size <= 1:
        train_data = replay_buffer.sample(batch_size, policy)
        return train_data, train_data, batch_size, 0, 0

    elite_batch_size = int(round(batch_size * elite_replay_buffer.sample_ratio)) if elite_available else 0
    intrinsic_batch_size = (
        int(round(batch_size * intrinsic_replay_buffer.sample_ratio)) if intrinsic_available else 0
    )
    if elite_available:
        elite_batch_size = max(1, elite_batch_size)
    if intrinsic_available:
        intrinsic_batch_size = max(1, intrinsic_batch_size)
    while elite_batch_size + intrinsic_batch_size > batch_size - 1:
        if intrinsic_batch_size >= elite_batch_size and intrinsic_batch_size > 0:
            intrinsic_batch_size -= 1
        elif elite_batch_size > 0:
            elite_batch_size -= 1
        else:
            break
    normal_batch_size = batch_size - elite_batch_size - intrinsic_batch_size

    normal_train_data = replay_buffer.sample(normal_batch_size, policy)
    mixed_parts = [normal_train_data]

    if elite_batch_size > 0:
        elite_train_data = elite_replay_buffer.sample_train_data(elite_batch_size, replay_buffer, policy)
        if elite_train_data is None:
            elite_batch_size = 0
        else:
            mixed_parts.append(elite_train_data)

    if intrinsic_batch_size > 0:
        intrinsic_train_data = intrinsic_replay_buffer.sample_train_data(intrinsic_batch_size, replay_buffer, policy)
        if intrinsic_train_data is None:
            intrinsic_batch_size = 0
        else:
            mixed_parts.append(intrinsic_train_data)

    mixed_train_data = _concat_train_data_many(mixed_parts)
    return mixed_train_data, normal_train_data, normal_batch_size, elite_batch_size, intrinsic_batch_size


def _log_elite_replay_stats(
        elite_replay_buffer: EliteEpisodeReplayBuffer,
        tb_logger,
        train_iter: int,
        collection_phase: Optional[int] = None,
        phase_sample_count: Optional[int] = None,
        phase_update_count: Optional[int] = None,
        batch_size: Optional[int] = None,
) -> None:
    if not elite_replay_buffer.enabled:
        return
    stats = elite_replay_buffer.stats()
    if phase_sample_count is not None:
        stats['elite_replay/phase_sample_count'] = float(phase_sample_count)
    if phase_update_count is not None:
        stats['elite_replay/phase_update_count'] = float(phase_update_count)
    if phase_sample_count is not None and phase_update_count is not None and batch_size:
        denominator = max(1, int(phase_update_count) * int(batch_size))
        stats['elite_replay/phase_sample_ratio_actual'] = float(phase_sample_count) / float(denominator)
    logging.info(
        "[ELITE_REPLAY] phase %s, size %s, best_return %.3f, worst_return %.3f, mean_return %.3f, "
        "last_sample_count %s, phase_sample_count %s, phase_update_count %s%s",
        collection_phase,
        int(stats['elite_replay/size']),
        stats['elite_replay/best_return'],
        stats['elite_replay/worst_return'],
        stats['elite_replay/mean_return'],
        int(stats['elite_replay/last_sample_count']),
        int(stats.get('elite_replay/phase_sample_count', 0.0)),
        int(stats.get('elite_replay/phase_update_count', 0.0)),
        (
            ", start_bins [" + ", ".join(
                f"b{bin_id}={int(stats[f'elite_replay/start_bin_{bin_id}/size'])}/"
                f"{int(stats[f'elite_replay/start_bin_{bin_id}/capacity'])} "
                f"mean={stats[f'elite_replay/start_bin_{bin_id}/mean_return']:.1f}"
                for bin_id in range(int(stats.get('elite_replay/start_bin_count', 0.0)))
            ) + "]"
            if stats.get('elite_replay/start_stratification_enabled', 0.0) else ""
        ),
    )
    if tb_logger is not None:
        for key, value in stats.items():
            tb_logger.add_scalar(key, value, train_iter)


def _log_intrinsic_replay_stats(
        intrinsic_replay_buffer: IntrinsicEpisodeReplayBuffer,
        tb_logger,
        train_iter: int,
        collection_phase: Optional[int] = None,
        phase_sample_count: Optional[int] = None,
        phase_update_count: Optional[int] = None,
        batch_size: Optional[int] = None,
) -> None:
    if intrinsic_replay_buffer is None or not intrinsic_replay_buffer.enabled:
        return
    stats = intrinsic_replay_buffer.stats()
    if phase_sample_count is not None:
        stats['intrinsic_replay/phase_sample_count'] = float(phase_sample_count)
    if phase_update_count is not None:
        stats['intrinsic_replay/phase_update_count'] = float(phase_update_count)
    if phase_sample_count is not None and phase_update_count is not None and batch_size:
        denominator = max(1, int(phase_update_count) * int(batch_size))
        stats['intrinsic_replay/phase_sample_ratio_actual'] = float(phase_sample_count) / float(denominator)
    logging.info(
        "[INTRINSIC_REPLAY] phase %s, size %s, best_score %.3f, worst_score %.3f, mean_score %.3f, "
        "best_sum %.3f, best_max %.3f, last_sample_count %s, phase_sample_count %s, phase_update_count %s",
        collection_phase,
        int(stats['intrinsic_replay/size']),
        stats['intrinsic_replay/best_score'],
        stats['intrinsic_replay/worst_score'],
        stats['intrinsic_replay/mean_score'],
        stats['intrinsic_replay/best_reward_sum'],
        stats['intrinsic_replay/best_reward_max'],
        int(stats['intrinsic_replay/last_sample_count']),
        int(stats.get('intrinsic_replay/phase_sample_count', 0.0)),
        int(stats.get('intrinsic_replay/phase_update_count', 0.0)),
    )
    if tb_logger is not None:
        for key, value in stats.items():
            tb_logger.add_scalar(key, value, train_iter)


def _new_learn_split_phase_stats() -> Dict[str, Dict[str, float]]:
    return {
        'normal': {'samples': 0.0},
        'elite': {'samples': 0.0},
        'intrinsic': {'samples': 0.0},
    }


def _new_value_target_phase_stats() -> Dict[str, float]:
    return {'samples': 0.0}


def _accumulate_value_target_phase_stats(
        phase_stats: Dict[str, float],
        value_target_stats: Optional[Dict[str, float]],
) -> None:
    if not value_target_stats:
        return
    count = float(value_target_stats.get('value_target/count', 0.0))
    if count <= 0:
        return
    phase_stats['samples'] = phase_stats.get('samples', 0.0) + count
    for key, value in value_target_stats.items():
        if key in {
                'value_target/count',
                'value_target/current_train_iter',
                'value_target/current_envstep',
                'value_target/replay_transitions',
        }:
            continue
        if not np.isfinite(float(value)):
            continue
        phase_stats[f'{key}_sum'] = phase_stats.get(f'{key}_sum', 0.0) + float(value) * count


def _log_value_target_phase_stats(
        phase_stats: Dict[str, float],
        tb_logger,
        train_iter: int,
        collection_phase: int,
) -> None:
    sample_count = phase_stats.get('samples', 0.0)
    if sample_count <= 0:
        return
    metrics = {
        key[:-4]: value / sample_count
        for key, value in phase_stats.items()
        if key.endswith('_sum')
    }
    logging.info(
        "[VALUE_TARGETS] phase %s, samples %s, sve_fraction %.3f, final_mean %.2f, "
        "td_candidate_mean %.2f, sve_final_mean %.2f, sve_td_candidate_mean %.2f, td_final_mean %.2f, "
        "sve_prefix_steps %.2f, sve_prefix_return %.2f, sve_search_value %.2f, "
        "sve_search_contribution %.2f",
        collection_phase,
        int(sample_count),
        metrics.get('value_target/sve_fraction', float('nan')),
        metrics.get('value_target/final_mean', float('nan')),
        metrics.get('value_target/td_candidate_mean', float('nan')),
        metrics.get('value_target/sve_final_mean', float('nan')),
        metrics.get('value_target/sve_td_candidate_mean', float('nan')),
        metrics.get('value_target/td_final_mean', float('nan')),
        metrics.get('value_target/sve_prefix_steps_mean', float('nan')),
        metrics.get('value_target/sve_prefix_return_mean', float('nan')),
        metrics.get('value_target/sve_search_value_mean', float('nan')),
        metrics.get('value_target/sve_search_contribution_mean', float('nan')),
    )
    if tb_logger is not None:
        tb_logger.add_scalar('value_target_phase/sample_count', float(sample_count), train_iter)
        for name, value in metrics.items():
            tb_logger.add_scalar(f'value_target_phase/{name.split("/", 1)[-1]}', float(value), train_iter)


def _accumulate_split_metric(stats: Dict[str, Dict[str, float]], split: str, name: str, value: float, count: int) -> None:
    if count <= 0 or not np.isfinite(value):
        return
    split_stats = stats.setdefault(split, {'samples': 0.0})
    split_stats[f'{name}_sum'] = split_stats.get(f'{name}_sum', 0.0) + float(value) * float(count)


def _split_mean(values: Optional[np.ndarray], start: int, end: int) -> Optional[float]:
    if values is None or end <= start:
        return None
    sliced = np.asarray(values)[start:end]
    if sliced.size == 0:
        return None
    return float(np.mean(sliced))


def _accumulate_learn_split_diagnostics(
        phase_stats: Dict[str, Dict[str, float]],
        tb_logger,
        train_iter: int,
        train_data: List[Any],
        log_vars: dict,
        normal_batch_size: int,
        elite_batch_size: int,
        intrinsic_batch_size: int = 0,
) -> None:
    if train_data is None or len(train_data) < 2:
        return
    target_batch = train_data[1]
    if len(target_batch) < 2:
        return

    root_target = np.asarray(target_batch[1]).reshape(len(target_batch[1]), -1)[:, 0]
    batch_size = root_target.shape[0]
    normal_count = min(int(normal_batch_size), batch_size)
    elite_count = max(0, min(int(elite_batch_size), batch_size - normal_count))
    intrinsic_count = max(0, min(int(intrinsic_batch_size), batch_size - normal_count - elite_count))
    root_pred = log_vars.get('root_predicted_value_orig', None)
    root_pred = np.asarray(root_pred).reshape(-1) if root_pred is not None else None
    value_priority = log_vars.get('value_priority_orig', None)
    value_priority = np.asarray(value_priority).reshape(-1) if value_priority is not None else None

    split_ranges = {
        'normal': (0, normal_count),
        'elite': (normal_count, normal_count + elite_count),
        'intrinsic': (normal_count + elite_count, normal_count + elite_count + intrinsic_count),
    }
    for split, (start, end) in split_ranges.items():
        count = end - start
        if count <= 0:
            continue
        split_stats = phase_stats.setdefault(split, {'samples': 0.0})
        split_stats['samples'] = split_stats.get('samples', 0.0) + float(count)
        metrics = {
            'root_target_value': _split_mean(root_target, start, end),
            'root_predicted_value': _split_mean(root_pred, start, end),
            'root_value_error': _split_mean(value_priority, start, end),
        }
        for name, value in metrics.items():
            if value is None:
                continue
            _accumulate_split_metric(phase_stats, split, name, value, count)
            if tb_logger is not None:
                tb_logger.add_scalar(f'learn_split_update/{split}/{name}', float(value), train_iter)
        if tb_logger is not None:
            tb_logger.add_scalar(f'learn_split_update/{split}/sample_count', float(count), train_iter)


def _log_learn_split_phase_stats(
        phase_stats: Dict[str, Dict[str, float]],
        tb_logger,
        train_iter: int,
        collection_phase: int,
) -> None:
    for split, split_stats in phase_stats.items():
        sample_count = split_stats.get('samples', 0.0)
        if sample_count <= 0:
            if tb_logger is not None:
                tb_logger.add_scalar(f'learn_split_phase/{split}/sample_count', 0.0, train_iter)
            continue
        metrics = {}
        for key, value in split_stats.items():
            if not key.endswith('_sum'):
                continue
            metric_name = key[:-4]
            metrics[metric_name] = value / sample_count
        logging.info(
            "[LEARN_SPLIT_PHASE] phase %s split %s samples %s root_target %.3f root_pred %.3f root_error %.3f",
            collection_phase,
            split,
            int(sample_count),
            metrics.get('root_target_value', float('nan')),
            metrics.get('root_predicted_value', float('nan')),
            metrics.get('root_value_error', float('nan')),
        )
        if tb_logger is not None:
            tb_logger.add_scalar(f'learn_split_phase/{split}/sample_count', float(sample_count), train_iter)
            for name, value in metrics.items():
                tb_logger.add_scalar(f'learn_split_phase/{split}/{name}', float(value), train_iter)


def _record_timing_scalars(tb_logger, train_iter: int, timing: dict) -> None:
    if tb_logger is None:
        return
    for name, value in timing.items():
        if isinstance(value, (int, float)):
            tb_logger.add_scalar(f'timing/train_loop/{name}', value, train_iter)


def _accumulate_learn_profile(timing: dict, log_dict: dict) -> None:
    profile_sum = timing.setdefault('learn_profile_sum', {})
    for key, value in log_dict.items():
        if not key.startswith('profile/'):
            continue
        if isinstance(value, (int, float)):
            profile_sum[key] = profile_sum.get(key, 0.0) + float(value)


def _learn_profile_avg_ms(timing: dict, key: str, update_count: int) -> float:
    if update_count <= 0:
        return 0.0
    return timing.get('learn_profile_sum', {}).get(key, 0.0) * 1000 / update_count


def _learn_profile_sum_avg_ms(timing: dict, keys: Tuple[str, ...], update_count: int) -> float:
    if update_count <= 0:
        return 0.0
    profile_sum = timing.get('learn_profile_sum', {})
    return sum(profile_sum.get(key, 0.0) for key in keys) * 1000 / update_count


def _log_timing_summary(
        timing: dict,
        tb_logger,
        start_train_iter: int,
        end_train_iter: int,
        start_envstep: int,
        end_envstep: int,
) -> None:
    update_count = timing.get('learn_update_count', 0)
    timing['sample_ms_per_update'] = timing['sample_s'] * 1000 / update_count if update_count > 0 else 0.0
    timing['learn_ms_per_update'] = timing['learn_s'] * 1000 / update_count if update_count > 0 else 0.0
    timing['total_ms_per_update'] = timing['loop_s'] * 1000 / update_count if update_count > 0 else 0.0
    accounted_s = sum(
        timing[key] for key in [
            'bookkeeping_s', 'eval_s', 'collect_s', 'update_calc_s', 'replay_push_s',
            'sample_s', 'learn_s', 'priority_s'
        ]
    )
    timing['other_s'] = max(0.0, timing['loop_s'] - accounted_s)
    profile_sum = timing.get('learn_profile_sum', {})
    if update_count > 0 and profile_sum:
        for key, value in profile_sum.items():
            timing[f'{key}_ms_per_update'] = value * 1000 / update_count
    logging.info(
        "[TIMING] train_iter %s->%s, envstep %s->%s, updates %s, skipped_updates %s, "
        "loop %.3fs | bookkeeping %.3fs, eval %.3fs, collect %.3fs, update_calc %.3fs, "
        "replay_push %.3fs, sample %.3fs (%.2f ms/update), learn %.3fs (%.2f ms/update), "
        "priority %.3fs, other %.3fs, total %.2f ms/update",
        start_train_iter,
        end_train_iter,
        start_envstep,
        end_envstep,
        update_count,
        timing.get('skipped_update_count', 0),
        timing['loop_s'],
        timing['bookkeeping_s'],
        timing['eval_s'],
        timing['collect_s'],
        timing['update_calc_s'],
        timing['replay_push_s'],
        timing['sample_s'],
        timing['sample_ms_per_update'],
        timing['learn_s'],
        timing['learn_ms_per_update'],
        timing['priority_s'],
        timing['other_s'],
        timing['total_ms_per_update'],
    )
    if update_count > 0 and profile_sum:
        member_forward_ms = _learn_profile_sum_avg_ms(
            timing,
            (
                'profile/model_initial_inference_member_forward_s',
                'profile/model_recurrent_inference_member_forward_s',
                'profile/model_project_member_forward_s',
            ),
            update_count,
        )
        index_ms = _learn_profile_sum_avg_ms(
            timing,
            (
                'profile/model_initial_inference_index_s',
                'profile/model_recurrent_inference_index_s',
                'profile/model_project_index_s',
            ),
            update_count,
        )
        stitch_concat_ms = _learn_profile_sum_avg_ms(
            timing,
            (
                'profile/model_initial_inference_stitch_s',
                'profile/model_initial_inference_concat_s',
                'profile/model_recurrent_inference_stitch_s',
                'profile/model_recurrent_inference_concat_s',
                'profile/model_project_stitch_s',
                'profile/model_project_concat_s',
            ),
            update_count,
        )
        logging.info(
            "[TIMING_LEARN] train_iter %s->%s, avg/update | wrapper %.2f ms, super %.2f ms, "
            "non_model_super %.2f ms, model_total %.2f ms | initial %.2f ms, recurrent %.2f ms, "
            "project %.2f ms | member_forward %.2f ms, index %.2f ms, stitch_concat %.2f ms",
            start_train_iter,
            end_train_iter,
            _learn_profile_avg_ms(timing, 'profile/learn_total_wrapper_s', update_count),
            _learn_profile_avg_ms(timing, 'profile/learn_super_s', update_count),
            _learn_profile_avg_ms(timing, 'profile/learn_non_model_super_s', update_count),
            _learn_profile_avg_ms(timing, 'profile/model_total_s', update_count),
            _learn_profile_avg_ms(timing, 'profile/model_initial_inference_total_s', update_count),
            _learn_profile_avg_ms(timing, 'profile/model_recurrent_inference_total_s', update_count),
            _learn_profile_avg_ms(timing, 'profile/model_project_total_s', update_count),
            member_forward_ms,
            index_ms,
            stitch_concat_ms,
        )
    _record_timing_scalars(tb_logger, end_train_iter, timing)


def train_ensemble_muzero(
        input_cfg: Tuple[dict, dict],
        seed: int = 0,
        model: Optional[torch.nn.Module] = None,
        model_path: Optional[str] = None,
        max_train_iter: Optional[int] = int(1e10),
        max_env_step: Optional[int] = int(1e10),
) -> 'Policy':  # noqa
    """
    Training entry for the ensemble MuZero variant.

    This skeleton intentionally mirrors the vanilla MuZero training loop while
    accepting ``create_cfg.policy.type == 'ensemble_muzero'``. With
    ``model.ensemble_size == 1`` it should behave like normal MuZero.
    """
    cfg, create_cfg = input_cfg
    assert create_cfg.policy.type == 'ensemble_muzero', \
        "train_ensemble_muzero only supports create_cfg.policy.type == 'ensemble_muzero'"

    from lzero.mcts import MuZeroGameBuffer as GameBuffer

    if cfg.policy.cuda and torch.cuda.is_available():
        cfg.policy.device = 'cuda'
    else:
        cfg.policy.device = 'cpu'

    cfg = compile_config(cfg, seed=seed, env=None, auto=True, create_cfg=create_cfg, save_cfg=True)
    env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
    _maybe_enable_collect_video(cfg, collector_env)

    collector_env.seed(cfg.seed)
    evaluator_env.seed(cfg.seed, dynamic_seed=False)
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

    if cfg.policy.eval_offline:
        cfg.policy.learn.learner.hook.save_ckpt_after_iter = cfg.policy.eval_freq

    if cfg.policy.use_wandb:
        wandb.init(
            project="LightZero",
            config=cfg,
            sync_tensorboard=False,
            monitor_gym=False,
            save_code=True,
        )

    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval'])

    if model_path is not None:
        policy.learn_mode.load_state_dict(torch.load(model_path, map_location=cfg.policy.device))

    tb_logger = SummaryWriter(
        os.path.join('./{}/log/'.format(cfg.exp_name), 'serial')
    ) if get_rank() == 0 else None
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)

    policy_config = cfg.policy
    if (
            bool(policy_config.get('rnd_store_collection_rewards', False))
            or bool(policy_config.get('rnd_use_stored_collection_rewards', False))
            or bool(policy_config.get('qstd_store_collection_rewards', False))
            or bool(policy_config.get('state_value_disagreement_store_collection_rewards', False))
            or str(policy_config.get('intrinsic_reward_source', 'rnd')) == 'state_density'
            or bool(policy_config.get('intrinsic_episode_replay', False))
    ):
        policy_config.store_selected_intrinsic_rewards = True
    batch_size = policy_config.batch_size
    profile_timing = bool(policy_config.get('profile_train_loop_timing', False))
    profile_timing_freq = int(policy_config.get('profile_train_loop_timing_freq', 1))
    profile_loop_count = 0
    replay_buffer = GameBuffer(policy_config)
    elite_replay_buffer = EliteEpisodeReplayBuffer(policy_config, exp_name=cfg.exp_name)
    intrinsic_replay_buffer = IntrinsicEpisodeReplayBuffer(policy_config, exp_name=cfg.exp_name)
    state_density_tracker = (
        StateDensityNoveltyTracker(policy_config)
        if str(policy_config.get('intrinsic_reward_source', 'rnd')) == 'state_density'
        else None
    )
    environment_coverage_tracker = EnvironmentCoverageTracker(policy_config)
    spatial_exploration_shadow = _create_spatial_exploration_shadow(policy_config)
    collector = Collector(
        env=collector_env,
        policy=policy.collect_mode,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name,
        policy_config=policy_config,
    )
    evaluator = Evaluator(
        eval_freq=cfg.policy.eval_freq,
        n_evaluator_episode=cfg.env.n_evaluator_episode,
        stop_value=cfg.env.stop_value,
        env=evaluator_env,
        policy=policy.eval_mode,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name,
        policy_config=policy_config
    )

    learner.call_hook('before_run')
    if policy_config.use_wandb:
        policy.set_train_iter_env_step(learner.train_iter, collector.envstep)

    if cfg.policy.update_per_collect is not None:
        update_per_collect = cfg.policy.update_per_collect

    diagnostic_collection_only = bool(
        policy_config.get('diagnostic_collection_only', False)
    )

    if cfg.policy.random_collect_episode_num > 0:
        random_collect(cfg.policy, policy, LightZeroRandomPolicy, collector, collector_env, replay_buffer)
    if cfg.policy.eval_offline:
        eval_train_iter_list = []
        eval_train_envstep_list = []

    if diagnostic_collection_only:
        stop, reward = False, None
        logging.info(
            '[DIAGNOSTIC_COLLECTION_ONLY] Skipping initial evaluation; the '
            'loaded policy will collect one phase without parameter updates.'
        )
    else:
        stop, reward = evaluator.eval(
            learner.save_checkpoint, learner.train_iter, collector.envstep
        )
    collection_phase = 0

    while True:
        profile_loop_count += 1
        start_train_iter = learner.train_iter
        start_envstep = collector.envstep
        timing = None
        loop_start = None
        if profile_timing:
            timing = {
                'bookkeeping_s': 0.0,
                'eval_s': 0.0,
                'collect_s': 0.0,
                'update_calc_s': 0.0,
                'replay_push_s': 0.0,
                'sample_s': 0.0,
                'learn_s': 0.0,
                'priority_s': 0.0,
                'learn_update_count': 0,
                'skipped_update_count': 0,
            }
            loop_start = _timer_start(policy_config.device)

        phase_start = _timer_start(policy_config.device) if profile_timing else None
        log_buffer_memory_usage(learner.train_iter, replay_buffer, tb_logger)
        log_buffer_run_time(learner.train_iter, replay_buffer, tb_logger)
        collect_kwargs = {}
        collect_kwargs['temperature'] = visit_count_temperature(
            policy_config.manual_temperature_decay,
            policy_config.fixed_temperature_value,
            policy_config.threshold_training_steps_for_final_temperature,
            trained_steps=learner.train_iter
        )

        if policy_config.eps.eps_greedy_exploration_in_collect:
            epsilon_greedy_fn = get_epsilon_greedy_fn(
                start=policy_config.eps.start,
                end=policy_config.eps.end,
                decay=policy_config.eps.decay,
                type_=policy_config.eps.type
            )
            collect_kwargs['epsilon'] = epsilon_greedy_fn(collector.envstep)
        else:
            collect_kwargs['epsilon'] = 0.0
        if profile_timing:
            timing['bookkeeping_s'] += _timer_elapsed(phase_start, policy_config.device)

        if not diagnostic_collection_only and evaluator.should_eval(learner.train_iter):
            if cfg.policy.eval_offline:
                eval_train_iter_list.append(learner.train_iter)
                eval_train_envstep_list.append(collector.envstep)
            else:
                phase_start = _timer_start(policy_config.device) if profile_timing else None
                stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
                if profile_timing:
                    timing['eval_s'] += _timer_elapsed(phase_start, policy_config.device)
                if stop:
                    if profile_timing:
                        timing['loop_s'] = _timer_elapsed(loop_start, policy_config.device)
                        _log_timing_summary(
                            timing, tb_logger, start_train_iter, learner.train_iter, start_envstep, collector.envstep
                        )
                    break

        phase_start = _timer_start(policy_config.device) if profile_timing else None
        current_collection_phase = collection_phase
        refresh_rnd_snapshot = getattr(policy, 'refresh_rnd_target_snapshot_for_collection', None)
        if callable(refresh_rnd_snapshot):
            refresh_rnd_snapshot()
        refresh_qstd_snapshot = getattr(policy, 'refresh_qstd_target_snapshot_for_collection', None)
        if callable(refresh_qstd_snapshot):
            refresh_qstd_snapshot()
        _set_collect_video_phase(cfg, collector_env, current_collection_phase)
        new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)
        collection_phase += 1
        _annotate_collection_metadata(
            cfg,
            new_data,
            current_collection_phase,
            learner.train_iter,
            start_envstep,
            collector.envstep,
        )
        state_density_stats = None
        if state_density_tracker is not None:
            state_density_stats = state_density_tracker.annotate_collection(
                new_data, current_collection_phase
            )
            policy.set_state_density_snapshot(
                state_density_tracker.snapshot_counts,
                state_density_tracker.snapshot_phase,
            )
            _write_state_density_phase_diagnostics(
                cfg, state_density_stats, tb_logger, current_collection_phase
            )
        _write_collection_episode_diagnostics(cfg, new_data, tb_logger, current_collection_phase)
        environment_coverage_stats = environment_coverage_tracker.annotate_collection(
            new_data, current_collection_phase
        )
        _write_environment_coverage_diagnostics(
            cfg,
            environment_coverage_stats,
            tb_logger,
            current_collection_phase,
        )
        _write_environment_coverage_cell_diagnostics(
            cfg,
            environment_coverage_tracker.latest_cell_rows(),
        )
        _write_spatial_exploration_shadow_diagnostics(
            cfg,
            spatial_exploration_shadow,
            new_data,
            tb_logger,
            current_collection_phase,
        )
        _write_spatial_exploration_online_preupdate_diagnostics(
            cfg,
            policy,
            new_data,
            tb_logger,
            current_collection_phase,
        )
        _write_compact_query_diagnostics(
            cfg, policy, current_collection_phase
        )
        _write_rnd_episodic_memory_diagnostics(
            cfg, policy, tb_logger, current_collection_phase
        )
        _write_episodic_sequence_novelty_factor_diagnostics(
            cfg, policy, tb_logger, current_collection_phase
        )
        _write_episodic_sequence_root_action_diagnostics(
            cfg, policy, tb_logger, current_collection_phase
        )
        _write_spatial_exploration_tree_diagnostics(
            cfg, policy, tb_logger, current_collection_phase
        )
        _write_collection_root_action_diagnostics(cfg, policy, new_data, tb_logger, current_collection_phase)
        _finalize_and_write_depth_disagreement_stats(cfg, policy, tb_logger, current_collection_phase)
        _write_normalization_mode_diagnostics(
            cfg,
            policy,
            new_data,
            tb_logger,
            current_collection_phase,
            learner.train_iter,
        )
        _write_main_tree_trace_diagnostics(cfg, new_data, tb_logger, current_collection_phase)
        _write_intrinsic_tree_trace_diagnostics(cfg, new_data, tb_logger, current_collection_phase)
        _write_fixed_sequence_intrinsic_diagnostics(cfg, policy, new_data, tb_logger, current_collection_phase)
        _write_encoder_prior_grid_diagnostics(
            cfg,
            policy,
            new_data,
            tb_logger,
            current_collection_phase,
            learner.train_iter,
            diagnostic_stage='pre_learn',
            update_visits=True,
        )
        _write_encoder_prior_imagined_novelty_diagnostics(
            cfg,
            policy,
            new_data,
            tb_logger,
            current_collection_phase,
            learner.train_iter,
            diagnostic_stage='pre_learn',
            update_history=True,
        )
        if diagnostic_collection_only:
            logging.info(
                '[DIAGNOSTIC_COLLECTION_ONLY] Completed phase %s at envstep %s; '
                'exiting before all parameter updates.',
                current_collection_phase,
                collector.envstep,
            )
            break
        _train_rnd_predictor_from_fresh_collection(
            cfg,
            policy,
            new_data,
            tb_logger,
            current_collection_phase,
        )
        _train_spatial_exploration_tree_from_collection(
            cfg,
            policy,
            new_data,
            tb_logger,
            current_collection_phase,
        )
        _train_episodic_history_shadow_from_collection(
            cfg,
            policy,
            new_data,
            tb_logger,
            current_collection_phase,
        )
        _train_episodic_sequence_novelty_from_collection(
            cfg,
            policy,
            new_data,
            tb_logger,
            current_collection_phase,
        )
        if profile_timing:
            timing['collect_s'] += _timer_elapsed(phase_start, policy_config.device)

        phase_start = _timer_start(policy_config.device) if profile_timing else None
        update_per_collect = calculate_update_per_collect(cfg, new_data)
        if profile_timing:
            timing['update_calc_s'] += _timer_elapsed(phase_start, policy_config.device)

        phase_start = _timer_start(policy_config.device) if profile_timing else None
        replay_buffer.push_game_segments(new_data)
        replay_buffer.remove_oldest_data_to_fit()
        _maybe_refresh_rnd_landscape_query_target_scale(
            cfg,
            policy,
            replay_buffer,
            tb_logger,
            current_collection_phase,
        )
        elite_added = elite_replay_buffer.update(new_data)
        if elite_replay_buffer.enabled and elite_added > 0:
            logging.info("[ELITE_REPLAY] phase %s added %s new elite episode(s)", current_collection_phase, elite_added)
        intrinsic_added = intrinsic_replay_buffer.update(new_data)
        if intrinsic_replay_buffer.enabled and intrinsic_added > 0:
            logging.info(
                "[INTRINSIC_REPLAY] phase %s added %s high-novelty episode(s)",
                current_collection_phase,
                intrinsic_added,
            )
        if profile_timing:
            timing['replay_push_s'] += _timer_elapsed(phase_start, policy_config.device)

        elite_phase_sample_count = 0
        intrinsic_phase_sample_count = 0
        learn_phase_update_count = 0
        learn_split_phase_stats = _new_learn_split_phase_stats()
        value_target_phase_stats = _new_value_target_phase_stats()
        for i in range(update_per_collect):
            if replay_buffer.get_num_of_transitions() > batch_size:
                set_replay_progress = getattr(replay_buffer, 'set_train_iter_env_step', None)
                if callable(set_replay_progress):
                    set_replay_progress(learner.train_iter, collector.envstep)
                phase_start = _timer_start(policy_config.device) if profile_timing else None
                (
                    train_data,
                    priority_train_data,
                    priority_batch_size,
                    elite_batch_size,
                    intrinsic_batch_size,
                ) = _maybe_mix_elite_train_data(
                    replay_buffer, elite_replay_buffer, intrinsic_replay_buffer, batch_size, policy
                )
                elite_phase_sample_count += int(elite_batch_size)
                intrinsic_phase_sample_count += int(intrinsic_batch_size)
                _accumulate_value_target_phase_stats(
                    value_target_phase_stats,
                    getattr(replay_buffer, 'last_value_target_stats', None),
                )
                if profile_timing:
                    timing['sample_s'] += _timer_elapsed(phase_start, policy_config.device)
            else:
                if profile_timing:
                    timing['skipped_update_count'] += update_per_collect - i
                logging.warning(
                    f'The data in replay_buffer is not sufficient to sample a mini-batch: '
                    f'batch_size: {batch_size}, '
                    f'{replay_buffer} '
                    f'continue to collect now ....'
                )
                break

            if policy_config.use_wandb:
                policy.set_train_iter_env_step(learner.train_iter, collector.envstep)

            phase_start = _timer_start(policy_config.device) if profile_timing else None
            log_vars = learner.train(train_data, collector.envstep)
            learn_phase_update_count += 1
            if len(log_vars) > 0:
                _accumulate_learn_split_diagnostics(
                    learn_split_phase_stats,
                    tb_logger,
                    learner.train_iter,
                    train_data,
                    log_vars[0],
                    priority_batch_size,
                    elite_batch_size,
                    intrinsic_batch_size,
                )
            if profile_timing:
                timing['learn_s'] += _timer_elapsed(phase_start, policy_config.device)
                timing['learn_update_count'] += 1
                if len(log_vars) > 0:
                    _accumulate_learn_profile(timing, log_vars[0])

            if cfg.policy.use_priority:
                phase_start = _timer_start(policy_config.device) if profile_timing else None
                priority_values = log_vars[0]['value_priority_orig']
                if priority_batch_size < len(priority_values):
                    priority_values = priority_values[:priority_batch_size]
                replay_buffer.update_priority(priority_train_data, priority_values)
                if profile_timing:
                    timing['priority_s'] += _timer_elapsed(phase_start, policy_config.device)

        _log_elite_replay_stats(
            elite_replay_buffer,
            tb_logger,
            learner.train_iter,
            collection_phase=current_collection_phase,
            phase_sample_count=elite_phase_sample_count,
            phase_update_count=learn_phase_update_count,
            batch_size=batch_size,
        )
        _log_intrinsic_replay_stats(
            intrinsic_replay_buffer,
            tb_logger,
            learner.train_iter,
            collection_phase=current_collection_phase,
            phase_sample_count=intrinsic_phase_sample_count,
            phase_update_count=learn_phase_update_count,
            batch_size=batch_size,
        )
        _log_value_target_phase_stats(
            value_target_phase_stats,
            tb_logger,
            learner.train_iter,
            current_collection_phase,
        )
        _log_learn_split_phase_stats(
            learn_split_phase_stats,
            tb_logger,
            learner.train_iter,
            current_collection_phase,
        )
        _maybe_update_ensemble_value_calibration(
            cfg,
            policy,
            replay_buffer,
            tb_logger,
            current_collection_phase,
        )
        _write_state_density_value_propagation_diagnostics(
            cfg,
            policy,
            new_data,
            tb_logger,
            current_collection_phase,
        )
        _write_fixed_sequence_intrinsic_diagnostics(
            cfg,
            policy,
            new_data,
            tb_logger,
            current_collection_phase,
            diagnostic_stage='post_learn',
        )
        _write_encoder_prior_grid_diagnostics(
            cfg,
            policy,
            new_data,
            tb_logger,
            current_collection_phase,
            learner.train_iter,
            diagnostic_stage='post_learn',
            update_visits=False,
        )
        _write_encoder_prior_imagined_novelty_diagnostics(
            cfg,
            policy,
            new_data,
            tb_logger,
            current_collection_phase,
            learner.train_iter,
            diagnostic_stage='post_learn',
            update_history=False,
        )

        if profile_timing:
            timing['loop_s'] = _timer_elapsed(loop_start, policy_config.device)
            if profile_loop_count % profile_timing_freq == 0:
                _log_timing_summary(
                    timing, tb_logger, start_train_iter, learner.train_iter, start_envstep, collector.envstep
                )

        if collector.envstep >= max_env_step or learner.train_iter >= max_train_iter:
            if cfg.policy.eval_offline:
                logging.info(f'eval offline beginning...')
                ckpt_dirname = './{}/ckpt'.format(learner.exp_name)
                for train_iter, collector_envstep in zip(eval_train_iter_list, eval_train_envstep_list):
                    ckpt_name = 'iteration_{}.pth.tar'.format(train_iter)
                    ckpt_path = os.path.join(ckpt_dirname, ckpt_name)
                    policy.learn_mode.load_state_dict(torch.load(ckpt_path, map_location=cfg.policy.device))
                    stop, reward = evaluator.eval(learner.save_checkpoint, train_iter, collector_envstep)
                    logging.info(
                        f'eval offline at train_iter: {train_iter}, '
                        f'collector_envstep: {collector_envstep}, reward: {reward}'
                    )
                logging.info(f'eval offline finished!')
            break

    if diagnostic_collection_only:
        logging.info(
            '[DIAGNOSTIC_COLLECTION_ONLY] Skipping after-run checkpoint save.'
        )
    else:
        learner.call_hook('after_run')
    collector.close()
    evaluator.close()
    learner.close()
    if cfg.policy.use_wandb:
        wandb.finish()
    return policy
