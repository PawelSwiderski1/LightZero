import time
from collections import deque, namedtuple
from typing import Optional, Any, List, Dict, Set, Sequence, Tuple

import numpy as np
import torch
import wandb
from ding.envs import BaseEnvManager
from ding.torch_utils import to_ndarray
from ding.utils import build_logger, EasyTimer, SERIAL_COLLECTOR_REGISTRY, get_rank, get_world_size, \
    allreduce_data
from ding.worker.collector.base_serial_collector import ISerialCollector
from torch.nn import L1Loss
import torch.distributed as dist

from lzero.mcts.buffer.game_segment import GameSegment, should_bootstrap_at_trajectory_boundary
from lzero.mcts.utils import prepare_observation


class RunningMeanSquaredDistance:
    """Running mean used to normalize NGU's squared kNN distances."""

    def __init__(self, epsilon: float = 1e-4) -> None:
        self._count = max(0.0, float(epsilon))
        self._mean = 0.0

    @property
    def count(self) -> float:
        return float(self._count)

    @property
    def mean(self) -> float:
        return float(self._mean)

    def update(self, squared_distances: np.ndarray) -> None:
        values = np.asarray(squared_distances, dtype=np.float64).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return
        batch_count = float(values.size)
        total_count = self._count + batch_count
        self._mean = (
            self._mean * self._count + float(values.sum())
        ) / max(total_count, 1e-12)
        self._count = total_count


class EpisodicNoveltyMemory:
    """NGU episodic pseudo-count reward over real-observation encoder latents."""

    def __init__(
            self,
            distance_stats: RunningMeanSquaredDistance,
            k: int = 10,
            kernel_cluster_distance: float = 0.008,
            kernel_epsilon: float = 1e-4,
            pseudo_count_constant: float = 0.001,
            maximum_similarity: float = 8.0,
            reward_scale: float = 1.0,
            reward_clip: Optional[float] = None,
            zero_until_knn_full: bool = True,
    ) -> None:
        self._distance_stats = distance_stats
        self._k = max(1, int(k))
        self._kernel_cluster_distance = max(0.0, float(kernel_cluster_distance))
        self._kernel_epsilon = max(1e-12, float(kernel_epsilon))
        self._pseudo_count_constant = max(0.0, float(pseudo_count_constant))
        self._maximum_similarity = max(0.0, float(maximum_similarity))
        self._reward_scale = float(reward_scale)
        self._reward_clip = None if reward_clip is None else max(0.0, float(reward_clip))
        self._zero_until_knn_full = bool(zero_until_knn_full)
        self._embeddings: List[np.ndarray] = []

    @staticmethod
    def _embedding(value: Any) -> np.ndarray:
        embedding = np.asarray(value, dtype=np.float32).reshape(-1)
        if embedding.size == 0 or not np.all(np.isfinite(embedding)):
            raise ValueError("episodic novelty encoder embedding must be non-empty and finite")
        return embedding

    def reset(self, initial_embedding: Optional[Any] = None) -> None:
        self._embeddings = []
        if initial_embedding is not None:
            self._embeddings.append(self._embedding(initial_embedding))

    def score_and_add(self, value: Any) -> Tuple[float, float, Dict[str, float]]:
        embedding = self._embedding(value)
        memory_size = len(self._embeddings)
        details = {
            'memory_size': float(memory_size),
            'neighbor_count': 0.0,
            'embedding_l2_norm': float(np.linalg.norm(embedding)),
            'knn_squared_distance_mean': 0.0,
            'normalized_knn_distance_mean': 0.0,
            'distance_scale': float(self._distance_stats.mean),
            'pseudo_count': 0.0,
            'similarity': 0.0,
            'unscaled_reward': 0.0,
            'warmup': 1.0,
        }
        if not self._embeddings or (
                self._zero_until_knn_full and memory_size < self._k
        ):
            self._embeddings.append(embedding)
            return 0.0, 0.0, details

        memory = np.stack(self._embeddings, axis=0)
        squared_distances = np.square(memory - embedding[None, :]).sum(axis=1)
        nearest_count = min(self._k, squared_distances.size)
        nearest = np.partition(squared_distances, nearest_count - 1)[:nearest_count]
        self._distance_stats.update(nearest)

        distance_scale = max(self._distance_stats.mean, 1e-12)
        normalized_distances = nearest / distance_scale
        normalized_distances = np.maximum(
            normalized_distances - self._kernel_cluster_distance,
            0.0,
        )
        kernels = self._kernel_epsilon / (
            normalized_distances + self._kernel_epsilon
        )
        pseudo_count = max(0.0, float(kernels.sum()))
        similarity = float(np.sqrt(pseudo_count) + self._pseudo_count_constant)
        if self._maximum_similarity > 0.0 and similarity > self._maximum_similarity:
            unscaled_reward = 0.0
        else:
            unscaled_reward = 1.0 / max(similarity, 1e-12)

        reward = max(0.0, unscaled_reward * self._reward_scale)
        if self._reward_clip is not None:
            reward = min(reward, self._reward_clip)
        self._embeddings.append(embedding)
        details.update(
            neighbor_count=float(nearest_count),
            knn_squared_distance_mean=float(nearest.mean()),
            normalized_knn_distance_mean=float(normalized_distances.mean()),
            distance_scale=float(distance_scale),
            pseudo_count=float(pseudo_count),
            similarity=float(similarity),
            unscaled_reward=float(unscaled_reward),
            warmup=0.0,
        )
        return float(reward), float(pseudo_count), details


@SERIAL_COLLECTOR_REGISTRY.register('episode_muzero')
class MuZeroCollector(ISerialCollector):
    """
    Overview:
        The episode-based collector for MCTS-based reinforcement learning algorithms, 
        including MuZero, EfficientZero, Sampled EfficientZero, and Gumbel MuZero.
        It orchestrates the data collection process in a serial manner, managing interactions 
        between the policy and the environment to generate game segments for training.
    Interfaces:
        ``__init__``, ``reset``, ``reset_env``, ``reset_policy``, ``_reset_stat``, ``collect``, 
        ``_compute_priorities``, ``pad_and_save_last_trajectory``, ``_output_log``, ``close``, ``__del__``.
    Properties:
        ``envstep``.
    """

    # Default configuration for the collector. To be compatible with ISerialCollector.
    config = dict()

    def __init__(
            self,
            collect_print_freq: int = 100,
            env: Optional[BaseEnvManager] = None,
            policy: Optional[namedtuple] = None,
            tb_logger: 'SummaryWriter' = None,  # noqa
            exp_name: str = 'default_experiment',
            instance_name: str = 'collector',
            policy_config: 'policy_config' = None,  # noqa
            task_id: Optional[int] = None,
    ) -> None:
        """
        Overview:
            Initializes the MuZeroCollector with the given configuration.
        Arguments:
            - collect_print_freq (:obj:`int`): The frequency (in training iterations) at which to print collection statistics.
            - env (:obj:`Optional[BaseEnvManager]`): An instance of a vectorized environment manager.
            - policy (:obj:`Optional[namedtuple]`): A namedtuple containing the policy's forward pass and other methods.
            - tb_logger (:obj:`Optional[SummaryWriter]`): A TensorBoard logger instance for logging metrics.
            - exp_name (:obj:`str`): The name of the experiment, used for organizing logs.
            - instance_name (:obj:`str`): A unique name for this collector instance.
            - policy_config (:obj:`'policy_config'`): The configuration object for the policy.
            - task_id (:obj:`Optional[int]`): The identifier for the current task in a multi-task setting. If None, operates in single-task mode.
        """
        self.task_id = task_id
        self._exp_name = exp_name
        self._instance_name = instance_name
        self._collect_print_freq = collect_print_freq
        self._timer = EasyTimer()
        self._end_flag = False

        # Get distributed training info
        self._rank = get_rank()
        self._world_size = get_world_size()

        # Logger setup: only rank 0 creates the main logger and TensorBoard logger.
        if self._rank == 0:
            if tb_logger is not None:
                self._logger, _ = build_logger(
                    path=f'./{self._exp_name}/log/{self._instance_name}',
                    name=self._instance_name,
                    need_tb=False
                )
                self._tb_logger = tb_logger
            else:
                self._logger, self._tb_logger = build_logger(
                    path=f'./{self._exp_name}/log/{self._instance_name}', name=self._instance_name
                )
        else:
            self._logger, _ = build_logger(
                path=f'./{self._exp_name}/log/{self._instance_name}', name=self._instance_name, need_tb=False
            )
            self._tb_logger = None

        self.policy_config = policy_config
        self.collect_with_pure_policy = self.policy_config.collect_with_pure_policy

        self.reset(policy, env)

    def reset_env(self, _env: Optional[BaseEnvManager] = None) -> None:
        """
        Overview:
            Resets or replaces the environment managed by the collector.
            If `_env` is None, it resets the existing environment. Otherwise, it replaces the old
            environment with the new one and launches it.
        Arguments:
            - _env (:obj:`Optional[BaseEnvManager]`): The new environment to be used. If None, resets the current environment.
        """
        if _env is not None:
            self._env = _env
            self._env.launch()
            self._env_num = self._env.env_num
        else:
            self._env.reset()

    def reset_policy(self, _policy: Optional[namedtuple] = None) -> None:
        """
        Overview:
            Resets or replaces the policy used by the collector.
            If `_policy` is None, it resets the existing policy. Otherwise, it replaces the old
            policy with the new one.
        Arguments:
            - _policy (:obj:`Optional[namedtuple]`): The new policy to be used.
        """
        assert hasattr(self, '_env'), "Please set env first before resetting policy."
        if _policy is not None:
            self._policy = _policy
            self._default_n_episode = _policy.get_attribute('cfg').get('n_episode', None)
            self._logger.debug(
                f"Set default n_episode mode(n_episode({self._default_n_episode}), env_num({self._env_num}))"
            )
        self._policy.reset()

    def reset(self, _policy: Optional[namedtuple] = None, _env: Optional[BaseEnvManager] = None) -> None:
        """
        Overview:
            Resets the collector, including the environment and policy. Also re-initializes
            internal state variables for tracking collection progress.
        Arguments:
            - _policy (:obj:`Optional[namedtuple]`): The new policy to use.
            - _env (:obj:`Optional[BaseEnvManager]`): The new environment to use.
        """
        if _env is not None:
            self.reset_env(_env)
        if _policy is not None:
            self.reset_policy(_policy)

        # Initialize per-environment tracking info
        self._env_info = {env_id: {'time': 0., 'step': 0} for env_id in range(self._env_num)}

        # Reset overall statistics
        self._episode_info = []
        self._total_envstep_count = 0
        self._total_episode_count = 0
        self._total_duration = 0
        self._last_train_iter = 0
        self._end_flag = False
        self._episodic_novelty_distance_stats = [
            RunningMeanSquaredDistance() for _ in range(self._env_num)
        ]

        # A pool to store completed game segments, implemented using a deque.
        self.game_segment_pool = deque(maxlen=int(1e6))
        self.unroll_plus_td_steps = self.policy_config.num_unroll_steps + self.policy_config.td_steps

    def _reset_stat(self, env_id: int) -> None:
        """
        Overview:
            Resets the statistics for a specific environment, identified by `env_id`.
            This is typically called when an episode in that environment ends.
        Arguments:
            - env_id (:obj:`int`): The ID of the environment to reset statistics for.
        """
        self._env_info[env_id] = {'time': 0., 'step': 0}

    @property
    def envstep(self) -> int:
        """
        Overview:
            Returns the total number of environment steps collected since the last reset.
        Returns:
            - envstep (:obj:`int`): The total environment step count.
        """
        return self._total_envstep_count

    def close(self) -> None:
        """
        Overview:
            Closes the collector, including the environment and any loggers.
            Ensures that all resources are properly released.
        """
        if self._end_flag:
            return
        self._end_flag = True
        self._env.close()
        if self._tb_logger:
            self._tb_logger.flush()
            self._tb_logger.close()

    def __del__(self) -> None:
        """
        Overview:
            Destructor for the collector instance, ensuring that `close` is called
            to clean up resources.
        """
        self.close()

    # ==============================================================
    # MCTS+RL Core Collection Logic
    # ==============================================================
    @staticmethod
    def _scalar_float(value: Any) -> float:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        if isinstance(value, (list, tuple, np.ndarray)):
            value = np.asarray(value).reshape(-1)[0]
        return float(value)

    def _priority_initial_mode(self) -> str:
        return str(getattr(self.policy_config, 'priority_initial_mode', 'bellman_max')).lower()

    def _compute_search_value_priorities(
            self, i: int, pred_values_lst: List[float], search_values_lst: List[float]
    ) -> Optional[np.ndarray]:
        if not self.policy_config.use_priority:
            return None
        pred_values = torch.from_numpy(np.array(pred_values_lst[i])).to(self.policy_config.device).float().view(-1)
        search_values = torch.from_numpy(np.array(search_values_lst[i])).to(self.policy_config.device).float().view(-1)
        return L1Loss(reduction='none')(pred_values, search_values).detach().cpu().numpy() + 1e-6

    def _compute_bellman_priorities(self, game_segment: GameSegment) -> Optional[np.ndarray]:
        if not self.policy_config.use_priority:
            return None

        data_length = min(len(game_segment.action_segment), self.policy_config.game_segment_length)
        if data_length <= 0:
            return np.asarray([], dtype=np.float32)

        pred_values_raw = getattr(game_segment, 'pred_value_segment', [])
        if len(pred_values_raw) == 0:
            return None

        rewards = [self._scalar_float(reward) for reward in game_segment.reward_segment]
        pred_values = [self._scalar_float(value) for value in pred_values_raw]
        trajectory_len = len(game_segment.action_segment)
        td_steps = max(1, int(self.policy_config.td_steps))
        discount = float(self.policy_config.discount_factor)
        time_limit_truncated = bool(getattr(game_segment, 'time_limit_truncated', False))
        bootstrap_at_time_limit = bool(getattr(self.policy_config, 'bootstrap_at_time_limit', True))
        priorities = np.zeros(data_length, dtype=np.float32)

        for state_index in range(data_length):
            horizon = min(td_steps, max(0, trajectory_len - state_index))
            target_value = 0.0
            for reward_offset in range(horizon):
                reward_index = state_index + reward_offset
                if reward_index >= len(rewards):
                    break
                target_value += (discount ** reward_offset) * rewards[reward_index]

            bootstrap_index = state_index + horizon
            can_bootstrap = horizon > 0 and should_bootstrap_at_trajectory_boundary(
                bootstrap_index,
                trajectory_len,
                time_limit_truncated,
                bootstrap_at_time_limit,
            )
            if can_bootstrap and bootstrap_index < len(pred_values):
                target_value += (discount ** horizon) * pred_values[bootstrap_index]

            if state_index < len(pred_values):
                priorities[state_index] = abs(pred_values[state_index] - target_value) + 1e-6

        return priorities

    def _compute_priorities(
            self,
            i: int,
            pred_values_lst: List[float],
            search_values_lst: List[float],
            game_segment: Optional[GameSegment] = None,
    ) -> Optional[np.ndarray]:
        """
        Overview:
            Computes priorities for experience replay based on the discrepancy between
            collection-time value predictions and n-step Bellman targets.
        Arguments:
            - i (:obj:`int`): The index of the environment's data in the lists.
            - pred_values_lst (:obj:`List[float]`): A list containing lists of predicted values for each environment.
            - search_values_lst (:obj:`List[float]`): A list containing lists of search values from MCTS for each environment.
        Returns:
            - priorities (:obj:`Optional[np.ndarray]`): An array of priorities for the transitions. Returns None if priority is not used.
        """
        if not self.policy_config.use_priority:
            return None
        if self._priority_initial_mode() in ['pred_vs_search', 'search', 'search_value']:
            return self._compute_search_value_priorities(i, pred_values_lst, search_values_lst)
        if game_segment is None:
            return None
        return self._compute_bellman_priorities(game_segment)

    def pad_and_save_last_trajectory(
            self, i: int, last_game_segments: List[Optional[GameSegment]],
            last_game_priorities: List[Optional[np.ndarray]],
            game_segments: List[GameSegment], done: np.ndarray,
            last_game_metadata: Optional[List[Optional[dict]]] = None,
            segment_metadata: Optional[dict] = None,
    ) -> None:
        """
        Overview:
            Pads the end of the `last_game_segment` with data from the start of the current `game_segment`.
            This is necessary to compute target values for the final transitions of a segment. After padding,
            the completed segment is stored in the `game_segment_pool`.
        Arguments:
            - i (:obj:`int`): The index of the environment being processed.
            - last_game_segments (:obj:`List[Optional[GameSegment]]`): List of game segments from the previous collection chunk.
            - last_game_priorities (:obj:`List[Optional[np.ndarray]]`): List of priorities corresponding to the last game segments.
            - game_segments (:obj:`List[GameSegment]`): List of game segments from the current collection chunk.
            - done (:obj:`np.ndarray`): Array indicating if the episode has terminated for each environment.
        Note:
            An implicit assumption is that the start of the new segment's observation history overlaps with the
            end of the last segment's, e.g., `(last_game_segments[i].obs_segment[-4:][j] == game_segments[i].obs_segment[:4][j]).all()` is True.
        """
        # --- Prepare padding data from the current game segment ---
        # Observations for padding are taken from the start of the new segment.
        beg_index_obs = self.policy_config.model.frame_stack_num
        end_index_obs = beg_index_obs + self.policy_config.num_unroll_steps + self.policy_config.td_steps
        pad_obs_lst = game_segments[i].obs_segment[beg_index_obs:end_index_obs]

        # Actions for padding.
        beg_index_ac = 0
        end_index_ac = beg_index_ac + self.policy_config.num_unroll_steps + self.policy_config.td_steps
        pad_action_lst = game_segments[i].action_segment[beg_index_ac:end_index_ac]
        pad_action_mask_lst = game_segments[i].action_mask_segment[beg_index_ac:end_index_ac]
        pad_to_play_lst = game_segments[i].to_play_segment[beg_index_ac:end_index_ac]
        pad_timestep_lst = game_segments[i].timestep_segment[beg_index_ac:end_index_ac]

        # Child visits for padding.
        pad_child_visits_lst = game_segments[i].child_visit_segment[:self.policy_config.num_unroll_steps + self.policy_config.td_steps]

        # Rewards for padding.
        beg_index_rew = 0
        end_index_rew = beg_index_rew + self.unroll_plus_td_steps - 1
        pad_reward_lst = game_segments[i].reward_segment[beg_index_rew:end_index_rew]
        pad_intrinsic_reward_lst = getattr(
            game_segments[i], 'intrinsic_reward_segment', []
        )[beg_index_rew:end_index_rew]
        pad_intrinsic_reward_raw_lst = getattr(
            game_segments[i], 'intrinsic_reward_raw_segment', []
        )[beg_index_rew:end_index_rew]
        
        # Root values for padding.
        beg_index_val = 0
        end_index_val = beg_index_val + self.unroll_plus_td_steps
        pad_root_values_lst = game_segments[i].root_value_segment[beg_index_val:end_index_val]
        pad_pred_values_lst = getattr(game_segments[i], 'pred_value_segment', [])[beg_index_val:end_index_val]

        if self.policy_config.use_ture_chance_label_in_chance_encoder:
            chance_lst = game_segments[i].chance_segment[beg_index_rew:end_index_rew]
        
        if self.policy_config.gumbel_algo:
            pad_improved_policy_prob = game_segments[i].improved_policy_probs[beg_index_val:end_index_val]

        # --- Pad the last game segment and save it ---
        if self.policy_config.gumbel_algo:
                last_game_segments[i].pad_over(
                    pad_obs_lst, pad_reward_lst, pad_action_lst, pad_root_values_lst,
                    pad_child_visits_lst, next_segment_improved_policy=pad_improved_policy_prob,
                    next_segment_intrinsic_rewards=pad_intrinsic_reward_lst,
                    next_segment_intrinsic_reward_raws=pad_intrinsic_reward_raw_lst,
                    next_segment_pred_values=pad_pred_values_lst,
                    next_segment_action_masks=pad_action_mask_lst,
                    next_segment_to_plays=pad_to_play_lst,
                    next_segment_timesteps=pad_timestep_lst,
                )
        else:
            if self.policy_config.use_ture_chance_label_in_chance_encoder:
                last_game_segments[i].pad_over(
                    pad_obs_lst, pad_reward_lst, pad_action_lst, pad_root_values_lst,
                    pad_child_visits_lst, next_chances=chance_lst,
                    next_segment_intrinsic_rewards=pad_intrinsic_reward_lst,
                    next_segment_intrinsic_reward_raws=pad_intrinsic_reward_raw_lst,
                    next_segment_pred_values=pad_pred_values_lst,
                    next_segment_action_masks=pad_action_mask_lst,
                    next_segment_to_plays=pad_to_play_lst,
                    next_segment_timesteps=pad_timestep_lst,
                )
            else:
                last_game_segments[i].pad_over(
                    pad_obs_lst, pad_reward_lst, pad_action_lst, pad_root_values_lst, pad_child_visits_lst,
                    next_segment_intrinsic_rewards=pad_intrinsic_reward_lst,
                    next_segment_intrinsic_reward_raws=pad_intrinsic_reward_raw_lst,
                    next_segment_pred_values=pad_pred_values_lst,
                    next_segment_action_masks=pad_action_mask_lst,
                    next_segment_to_plays=pad_to_play_lst,
                    next_segment_timesteps=pad_timestep_lst,
                )
        
        metadata = {}
        if last_game_metadata is not None and last_game_metadata[i] is not None:
            metadata.update(last_game_metadata[i])
        if segment_metadata is not None:
            metadata.update(segment_metadata)
        metadata.setdefault('env_id', int(i))
        metadata.setdefault('done', bool(done[i]))
        metadata.setdefault(
            'pure_exploration_episode',
            bool(getattr(last_game_segments[i], 'pure_exploration_episode', False)),
        )
        last_game_segments[i].pure_exploration_episode = bool(
            metadata['pure_exploration_episode']
        )
        metadata['segment_length'] = int(len(last_game_segments[i]))
        last_game_segments[i].set_terminal_metadata(
            done=metadata.get('done', False),
            time_limit_truncated=metadata.get('time_limit_truncated', False),
            reached_goal=metadata.get('reached_goal', False),
            episode_return=metadata.get('episode_return', None),
        )

        # Convert the segment's lists to NumPy arrays for efficient storage.
        if self._priority_initial_mode() not in ['pred_vs_search', 'search', 'search_value']:
            last_game_priorities[i] = self._compute_bellman_priorities(last_game_segments[i])
        last_game_segments[i].game_segment_to_array()

        # Add the completed game segment and its associated data to the pool.
        self.game_segment_pool.append((last_game_segments[i], last_game_priorities[i], done[i], metadata))

        # Reset the placeholder for the last game segment.
        last_game_segments[i] = None
        last_game_priorities[i] = None
        if last_game_metadata is not None:
            last_game_metadata[i] = None

    def collect(
            self,
            n_episode: Optional[int] = None,
            train_iter: int = 0,
            policy_kwargs: Optional[Dict] = None,
            collect_with_pure_policy: bool = False
    ) -> List[Any]:
        """
        Overview:
            Collects `n_episode` episodes of data. It manages the entire lifecycle of an episode,
            from getting actions from the policy, stepping the environment, storing transitions,
            and saving completed game segments.
        Arguments:
            - n_episode (:obj:`Optional[int]`): The number of episodes to collect. If None, uses the default from the policy config.
            - train_iter (:obj:`int`): The current training iteration, used for logging.
            - policy_kwargs (:obj:`Optional[Dict]`): Additional keyword arguments to pass to the policy's forward method, like temperature for exploration.
            - collect_with_pure_policy (:obj:`bool`): If True, collects data using a pure policy (e.g., greedy action) without MCTS.
        Returns:
            - return_data (:obj:`List[Any]`): A list containing the collected game segments and metadata.
        """
        # TODO(author): Consider implementing `collect_with_pure_policy` as a separate, more streamlined collector for clarity and modularity.
        if n_episode is None:
            if self._default_n_episode is None:
                raise RuntimeError("Please specify `n_episode` for collection.")
            else:
                n_episode = self._default_n_episode
        assert n_episode >= self._env_num, f"Please ensure n_episode ({n_episode}) >= env_num ({self._env_num})."
        
        if policy_kwargs is None:
            policy_kwargs = {}
        temperature = policy_kwargs.get('temperature', 1.0)
        epsilon = policy_kwargs.get('epsilon', 0.0)

        # --- Initializations ---
        collected_episode = 0
        collected_step = 0
        env_nums = self._env_num
        retry_waiting_time = 0.05

        # Wait for all environments to be ready and get initial observations.
        init_obs = self._env.ready_obs
        while len(init_obs.keys()) != self._env_num:
            self._logger.warning(f"Waiting for all environments to reset. Ready envs: {list(init_obs.keys())}")
            time.sleep(retry_waiting_time)
            init_obs = self._env.ready_obs

        # Prepare initial state dictionaries from observations.
        action_mask_dict = {i: to_ndarray(init_obs[i]['action_mask']) for i in range(env_nums)}
        to_play_dict = {i: to_ndarray(init_obs[i]['to_play']) for i in range(env_nums)}
        timestep_dict = {i: to_ndarray(init_obs[i].get('timestep', -1)) for i in range(env_nums)}
        if self.policy_config.use_ture_chance_label_in_chance_encoder:
            chance_dict = {i: to_ndarray(init_obs[i]['chance']) for i in range(env_nums)}

        # Initialize game segments and observation stacks for each environment.
        game_segments = [GameSegment(self._env.action_space, game_segment_length=self.policy_config.game_segment_length, config=self.policy_config) for _ in range(env_nums)]
        observation_window_stack = [deque(maxlen=self.policy_config.model.frame_stack_num) for _ in range(env_nums)]
        for env_id in range(env_nums):
            for _ in range(self.policy_config.model.frame_stack_num):
                observation_window_stack[env_id].append(to_ndarray(init_obs[env_id]['observation']))
            game_segments[env_id].reset(observation_window_stack[env_id])
            initial_environment_state = init_obs[env_id].get('environment_state', None)
            if initial_environment_state is not None:
                game_segments[env_id].environment_initial_state = tuple(
                    int(value) for value in np.asarray(initial_environment_state).reshape(-1)
                )

        # State tracking variables for the collection loop.
        dones = np.array([False for _ in range(env_nums)])
        last_game_segments: List[Optional[GameSegment]] = [None for _ in range(env_nums)]
        last_game_priorities: List[Optional[np.ndarray]] = [None for _ in range(env_nums)]
        last_game_metadata: List[Optional[dict]] = [None for _ in range(env_nums)]
        env_episode_indices = np.zeros(env_nums, dtype=np.int64)
        env_episode_segment_indices = np.zeros(env_nums, dtype=np.int64)
        
        # Buffers for priority calculation.
        search_values_lst = [[] for _ in range(env_nums)]
        pred_values_lst = [[] for _ in range(env_nums)]
        if self.policy_config.gumbel_algo:
            improved_policy_lst = [[] for _ in range(env_nums)]

        # Logging variables.
        eps_steps_lst = np.zeros(env_nums)
        visit_entropies_lst = np.zeros(env_nums)
        eps_intrinsic_reward_sum_lst = np.zeros(env_nums)
        eps_intrinsic_reward_max_lst = np.zeros(env_nums)
        eps_intrinsic_reward_raw_sum_lst = np.zeros(env_nums)
        eps_intrinsic_reward_raw_max_lst = np.zeros(env_nums)
        eps_episodic_novelty_count_lst = np.zeros(env_nums)
        eps_episodic_novelty_embedding_norm_sum_lst = np.zeros(env_nums)
        eps_episodic_novelty_knn_squared_distance_sum_lst = np.zeros(env_nums)
        eps_episodic_novelty_normalized_distance_sum_lst = np.zeros(env_nums)
        eps_episodic_novelty_similarity_sum_lst = np.zeros(env_nums)
        eps_episodic_novelty_unscaled_reward_sum_lst = np.zeros(env_nums)
        eps_episodic_novelty_warmup_sum_lst = np.zeros(env_nums)
        eps_episodic_novelty_distance_scale_sum_lst = np.zeros(env_nums)
        if self.policy_config.gumbel_algo:
            completed_value_lst = np.zeros(env_nums)

        ready_env_id: Set[int] = set()
        remain_episode = n_episode
        if collect_with_pure_policy:
            # Dummy visit counts for pure policy collection.
            temp_visit_list = [0.0 for _ in range(self._env.action_space.n)]
        store_collection_intrinsic_rewards = (
            bool(getattr(self.policy_config, 'rnd_store_collection_rewards', False))
            or bool(getattr(self.policy_config, 'qstd_store_collection_rewards', False))
            or bool(getattr(self.policy_config, 'state_value_disagreement_store_collection_rewards', False))
            or bool(getattr(self.policy_config, 'episodic_novelty_diagnostics', False))
        )
        episodic_novelty_enabled = bool(
            getattr(self.policy_config, 'episodic_novelty_diagnostics', False)
        )
        episodic_novelty_memories = None
        episodic_novelty_embedding_fn = None
        episodic_history_reward_feedback_fn = None
        if episodic_novelty_enabled:
            try:
                episodic_novelty_embedding_fn = self._policy.get_attribute(
                    'collection_episodic_novelty_embedding_fn'
                )
            except NotImplementedError as error:
                raise RuntimeError(
                    "episodic_novelty_diagnostics requires a policy real-observation "
                    "encoder callback"
                ) from error
            episodic_novelty_memories = [
                EpisodicNoveltyMemory(
                    distance_stats=self._episodic_novelty_distance_stats[env_id],
                    k=int(getattr(self.policy_config, 'episodic_novelty_knn_k', 10)),
                    kernel_cluster_distance=float(
                        getattr(self.policy_config, 'episodic_novelty_kernel_cluster_distance', 0.008)
                    ),
                    kernel_epsilon=float(
                        getattr(self.policy_config, 'episodic_novelty_kernel_epsilon', 1e-4)
                    ),
                    pseudo_count_constant=float(
                        getattr(self.policy_config, 'episodic_novelty_pseudo_count_constant', 0.001)
                    ),
                    maximum_similarity=float(
                        getattr(self.policy_config, 'episodic_novelty_maximum_similarity', 8.0)
                    ),
                    reward_scale=float(
                        getattr(self.policy_config, 'episodic_novelty_reward_scale', 1.0)
                    ),
                    reward_clip=getattr(
                        self.policy_config, 'episodic_novelty_reward_clip', None
                    ),
                    zero_until_knn_full=bool(
                        getattr(self.policy_config, 'episodic_novelty_zero_until_knn_full', True)
                    ),
                )
                for env_id in range(env_nums)
            ]
            if bool(getattr(self.policy_config, 'episodic_novelty_history_shadow', False)):
                episodic_history_reward_feedback_fn = self._policy.get_attribute(
                    'collection_episodic_history_reward_feedback_fn'
                )
        if bool(getattr(self.policy_config, 'episodic_sequence_novelty', False)):
            episodic_history_reward_feedback_fn = self._policy.get_attribute(
                'collection_episodic_history_reward_feedback_fn'
            )
        collection_rnd_reward_fn = None
        if store_collection_intrinsic_rewards and not episodic_novelty_enabled:
            try:
                collection_rnd_reward_fn = self._policy.get_attribute('collection_rnd_reward_fn')
            except NotImplementedError:
                collection_rnd_reward_fn = None

        # --- Main Collection Loop ---
        while True:
            with self._timer:
                # Get observations from ready environments.
                obs = self._env.ready_obs
                new_available_env_id = set(obs.keys()).difference(ready_env_id)
                if episodic_novelty_memories is not None:
                    new_env_ids = sorted(int(env_id) for env_id in new_available_env_id)
                    if new_env_ids:
                        initial_stacks = [game_segments[env_id].get_obs() for env_id in new_env_ids]
                        initial_embeddings = episodic_novelty_embedding_fn(initial_stacks)
                        if len(initial_embeddings) != len(new_env_ids):
                            raise RuntimeError(
                                "episodic novelty encoder returned an unexpected batch size: "
                                f"expected {len(new_env_ids)}, got {len(initial_embeddings)}"
                            )
                        for env_id, embedding in zip(new_env_ids, initial_embeddings):
                            episodic_novelty_memories[env_id].reset(embedding)
                ready_env_id.update(list(new_available_env_id)[:remain_episode])
                remain_episode -= min(len(new_available_env_id), remain_episode)
                
                # Prepare policy inputs.
                stack_obs_list = [game_segments[env_id].get_obs() for env_id in ready_env_id]
                action_mask = [action_mask_dict[env_id] for env_id in ready_env_id]
                to_play = [to_play_dict[env_id] for env_id in ready_env_id]
                timestep = [timestep_dict[env_id] for env_id in ready_env_id]
                
                stack_obs_array = to_ndarray(stack_obs_list)
                stack_obs_tensor = prepare_observation(stack_obs_array, self.policy_config.model.model_type)
                stack_obs_tensor = torch.from_numpy(stack_obs_tensor).to(self.policy_config.device)

                # ==============================================================
                # Policy Forward Pass
                # ==============================================================
                policy_input = {
                    'data': stack_obs_tensor,
                    'action_mask': action_mask,
                    'temperature': temperature,
                    'to_play': to_play,
                    'epsilon': epsilon,
                    'ready_env_id': ready_env_id,
                    'timestep': timestep,
                    'train_iter': int(train_iter),
                }
                if self.task_id is not None:
                    policy_input['task_id'] = self.task_id
                
                policy_output = self._policy.forward(**policy_input)
                
                # --- Unpack policy outputs ---
                actions, value_dict, pred_value_dict = {}, {}, {}
                distributions_dict, visit_entropy_dict = {}, {}
                root_action_diagnostics_dict = {}
                main_tree_trace_dict = {}
                intrinsic_tree_trace_dict = {}
                selected_intrinsic_reward_dict = {}
                pure_exploration_episode_dict = {}
                if self.policy_config.sampled_algo:
                    root_sampled_actions_dict = {}
                if self.policy_config.gumbel_algo:
                    improved_policy_dict, completed_value_dict = {}, {}

                for env_id in ready_env_id:
                    output = policy_output[env_id]
                    actions[env_id] = output['action']
                    value_dict[env_id] = output['searched_value']
                    pred_value_dict[env_id] = output['predicted_value']
                    pure_exploration_episode_dict[env_id] = bool(
                        output.get('pure_exploration_episode', False)
                    )
                    
                    if not collect_with_pure_policy:
                        distributions_dict[env_id] = output['visit_count_distributions']
                        visit_entropy_dict[env_id] = output['visit_count_distribution_entropy']
                        if getattr(self.policy_config, 'collection_root_action_diagnostics', False):
                            root_action_diagnostics_dict[env_id] = output.get('root_action_diagnostics', [])
                        if getattr(self.policy_config, 'main_tree_trace_diagnostics', False):
                            main_tree_trace_dict[env_id] = output.get('main_tree_trace', None)
                        if getattr(self.policy_config, 'intrinsic_tree_trace_diagnostics', False):
                            intrinsic_tree_trace_dict[env_id] = output.get('intrinsic_tree_trace', None)
                        selected_intrinsic_reward_dict[env_id] = float(output.get('selected_intrinsic_reward', 0.0))
                        if self.policy_config.sampled_algo:
                            root_sampled_actions_dict[env_id] = output['root_sampled_actions']
                        if self.policy_config.gumbel_algo:
                            improved_policy_dict[env_id] = output['improved_policy_probs']
                            completed_value_dict[env_id] = output['roots_completed_value']

                # ==============================================================
                # Environment Interaction
                # ==============================================================
                timesteps = self._env.step(actions)

                episodic_next_embeddings = {}
                if episodic_novelty_memories is not None and timesteps:
                    next_env_ids = sorted(int(env_id) for env_id in timesteps)
                    next_stacks = []
                    frame_stack_num = int(self.policy_config.model.frame_stack_num)
                    for env_id in next_env_ids:
                        next_frames = list(observation_window_stack[env_id])
                        next_frames.append(to_ndarray(timesteps[env_id].obs['observation']))
                        next_stacks.append(next_frames[-frame_stack_num:])
                    next_embeddings = episodic_novelty_embedding_fn(next_stacks)
                    if len(next_embeddings) != len(next_env_ids):
                        raise RuntimeError(
                            "episodic novelty encoder returned an unexpected next-state batch size: "
                            f"expected {len(next_env_ids)}, got {len(next_embeddings)}"
                        )
                    episodic_next_embeddings = dict(zip(next_env_ids, next_embeddings))

                collection_intrinsic_rewards = {}
                if (
                        episodic_novelty_memories is None
                        and collection_rnd_reward_fn is not None
                        and timesteps
                ):
                    intrinsic_env_ids = sorted(int(env_id) for env_id in timesteps)
                    intrinsic_observations = [
                        to_ndarray(timesteps[env_id].obs['observation'])
                        for env_id in intrinsic_env_ids
                    ]
                    reward_info = collection_rnd_reward_fn(intrinsic_observations)
                    bonuses = np.asarray(
                        reward_info.get('bonus', []), dtype=np.float32
                    ).reshape(-1)
                    raw_errors = np.asarray(
                        reward_info.get('raw_error', []), dtype=np.float32
                    ).reshape(-1)
                    if (
                            bonuses.size != len(intrinsic_env_ids)
                            or raw_errors.size != len(intrinsic_env_ids)
                    ):
                        raise RuntimeError(
                            'Collection intrinsic reward function returned an unexpected '
                            f'batch size: bonus={bonuses.size}, raw={raw_errors.size}, '
                            f'expected={len(intrinsic_env_ids)}.'
                        )
                    collection_intrinsic_rewards = {
                        env_id: (float(bonus), float(raw_error))
                        for env_id, bonus, raw_error in zip(
                            intrinsic_env_ids, bonuses, raw_errors
                        )
                    }

            interaction_duration = self._timer.value / len(timesteps) if timesteps else 0

            for env_id, episode_timestep in timesteps.items():
                with self._timer:
                    # Handle abnormal timesteps by resetting the environment and policy state.
                    if episode_timestep.info.get('abnormal', False):
                        self._env.reset({env_id: None})
                        self._policy.reset([env_id])
                        self._reset_stat(env_id)
                        if episodic_novelty_memories is not None:
                            episodic_novelty_memories[env_id].reset()
                        self._logger.info(f"Environment {env_id} returned an abnormal step, info: {episode_timestep.info}")
                        continue

                    obs, reward, done, info = episode_timestep.obs, episode_timestep.reward, episode_timestep.done, episode_timestep.info
                    stored_intrinsic_reward = selected_intrinsic_reward_dict.get(env_id, 0.0)
                    stored_intrinsic_reward_raw = 0.0
                    episodic_novelty_details = None
                    if episodic_novelty_memories is not None:
                        stored_intrinsic_reward, stored_intrinsic_reward_raw, episodic_novelty_details = (
                            episodic_novelty_memories[env_id].score_and_add(
                                episodic_next_embeddings[env_id]
                            )
                        )
                    elif collection_rnd_reward_fn is not None:
                        (
                            stored_intrinsic_reward,
                            stored_intrinsic_reward_raw,
                        ) = collection_intrinsic_rewards.get(int(env_id), (0.0, 0.0))

                    if episodic_history_reward_feedback_fn is not None:
                        episodic_history_reward_feedback_fn(
                            [int(env_id)],
                            [float(stored_intrinsic_reward)],
                            [self._scalar_float(reward)],
                        )

                    # Store MCTS search statistics.
                    if collect_with_pure_policy:
                        game_segments[env_id].store_search_stats(temp_visit_list, 0, pred_value=0)
                    else:
                        if self.policy_config.sampled_algo:
                            game_segments[env_id].store_search_stats(
                                distributions_dict[env_id], value_dict[env_id],
                                root_sampled_actions_dict[env_id], pred_value=pred_value_dict[env_id]
                            )
                        elif self.policy_config.gumbel_algo:
                            game_segments[env_id].store_search_stats(
                                distributions_dict[env_id], value_dict[env_id],
                                improved_policy=improved_policy_dict[env_id], pred_value=pred_value_dict[env_id]
                            )
                        else:
                            game_segments[env_id].store_search_stats(
                                distributions_dict[env_id], value_dict[env_id], pred_value=pred_value_dict[env_id]
                            )
                        if getattr(self.policy_config, 'collection_root_action_diagnostics', False):
                            game_segments[env_id].store_root_action_diagnostics(
                                root_action_diagnostics_dict.get(env_id, [])
                            )
                        if getattr(self.policy_config, 'main_tree_trace_diagnostics', False):
                            game_segments[env_id].store_main_tree_trace(
                                main_tree_trace_dict.get(env_id, None)
                            )
                        if getattr(self.policy_config, 'intrinsic_tree_trace_diagnostics', False):
                            game_segments[env_id].store_intrinsic_tree_trace(
                                intrinsic_tree_trace_dict.get(env_id, None)
                            )

                    # Append the current transition to the game segment.
                    game_segments[env_id].pure_exploration_episode = bool(
                        pure_exploration_episode_dict.get(env_id, False)
                    )
                    append_kwargs = {
                        'timestep': timestep_dict[env_id],
                        'intrinsic_reward': stored_intrinsic_reward,
                        'intrinsic_reward_raw': stored_intrinsic_reward_raw,
                    }
                    agent_position = info.get('agent_position', None)
                    if agent_position is not None:
                        room_id = info.get('room_id', None)
                        room_row, room_col = (
                            (-1, -1)
                            if room_id is None else
                            (int(room_id[0]), int(room_id[1]))
                        )
                        append_kwargs['environment_state'] = (
                            int(agent_position[0]),
                            int(agent_position[1]),
                            int(info.get('agent_direction', -1)),
                            room_row,
                            room_col,
                        )
                    if self.policy_config.use_ture_chance_label_in_chance_encoder:
                        append_kwargs['chance'] = chance_dict[env_id]
                    game_segments[env_id].append(
                        actions[env_id],
                        to_ndarray(obs['observation']),
                        reward,
                        action_mask_dict[env_id],
                        to_play_dict[env_id],
                        **append_kwargs,
                    )

                    # Update state dictionaries for the next step.
                    action_mask_dict[env_id] = to_ndarray(obs['action_mask'])
                    to_play_dict[env_id] = to_ndarray(obs['to_play'])
                    timestep_dict[env_id] = to_ndarray(obs.get('timestep', -1))
                    if self.policy_config.use_ture_chance_label_in_chance_encoder:
                        chance_dict[env_id] = to_ndarray(obs['chance'])

                    dones[env_id] = done if not self.policy_config.ignore_done else False
                    
                    # Update logging and priority data.
                    if not collect_with_pure_policy:
                        visit_entropies_lst[env_id] += visit_entropy_dict[env_id]
                        if self.policy_config.gumbel_algo:
                            completed_value_lst[env_id] += np.mean(np.array(completed_value_dict[env_id]))
                    
                    eps_steps_lst[env_id] += 1
                    eps_intrinsic_reward_sum_lst[env_id] += stored_intrinsic_reward
                    eps_intrinsic_reward_max_lst[env_id] = max(
                        eps_intrinsic_reward_max_lst[env_id], stored_intrinsic_reward
                    )
                    eps_intrinsic_reward_raw_sum_lst[env_id] += stored_intrinsic_reward_raw
                    eps_intrinsic_reward_raw_max_lst[env_id] = max(
                        eps_intrinsic_reward_raw_max_lst[env_id], stored_intrinsic_reward_raw
                    )
                    if episodic_novelty_details is not None:
                        eps_episodic_novelty_count_lst[env_id] += 1
                        eps_episodic_novelty_embedding_norm_sum_lst[env_id] += float(
                            episodic_novelty_details['embedding_l2_norm']
                        )
                        eps_episodic_novelty_knn_squared_distance_sum_lst[env_id] += float(
                            episodic_novelty_details['knn_squared_distance_mean']
                        )
                        eps_episodic_novelty_normalized_distance_sum_lst[env_id] += float(
                            episodic_novelty_details['normalized_knn_distance_mean']
                        )
                        eps_episodic_novelty_similarity_sum_lst[env_id] += float(
                            episodic_novelty_details['similarity']
                        )
                        eps_episodic_novelty_unscaled_reward_sum_lst[env_id] += float(
                            episodic_novelty_details['unscaled_reward']
                        )
                        eps_episodic_novelty_warmup_sum_lst[env_id] += float(
                            episodic_novelty_details['warmup']
                        )
                        eps_episodic_novelty_distance_scale_sum_lst[env_id] += float(
                            episodic_novelty_details['distance_scale']
                        )
                    if self.policy_config.use_priority:
                        pred_values_lst[env_id].append(pred_value_dict[env_id])
                        search_values_lst[env_id].append(value_dict[env_id])

                    # Update the observation window with the new observation.
                    observation_window_stack[env_id].append(to_ndarray(obs['observation']))

                    # ==============================================================
                    # Game Segment Saving Logic
                    # ==============================================================
                    # If a segment is full, pad and save the previous segment.
                    if game_segments[env_id].is_full():
                        if last_game_segments[env_id] is not None:
                            self.pad_and_save_last_trajectory(
                                env_id, last_game_segments, last_game_priorities, game_segments, dones,
                                last_game_metadata=last_game_metadata
                            )

                        # Calculate priorities for the now-completed `last_game_segment`.
                        priorities = self._compute_priorities(
                            env_id, pred_values_lst, search_values_lst, game_segments[env_id]
                        )
                        pred_values_lst[env_id], search_values_lst[env_id] = [], []

                        # The current segment becomes the `last_game_segment`.
                        last_game_segments[env_id] = game_segments[env_id]
                        last_game_priorities[env_id] = priorities
                        last_game_metadata[env_id] = {
                            'env_id': int(env_id),
                            'env_episode_index': int(env_episode_indices[env_id]),
                            'episode_segment_index': int(env_episode_segment_indices[env_id]),
                            'pure_exploration_episode': bool(
                                game_segments[env_id].pure_exploration_episode
                            ),
                            'done': False,
                        }
                        env_episode_segment_indices[env_id] += 1

                        # Start a new game segment.
                        game_segments[env_id] = GameSegment(self._env.action_space, game_segment_length=self.policy_config.game_segment_length, config=self.policy_config)
                        game_segments[env_id].reset(observation_window_stack[env_id])
                        initial_environment_state = append_kwargs.get(
                            'environment_state', None
                        )
                        if initial_environment_state is not None:
                            game_segments[env_id].environment_initial_state = tuple(
                                int(value)
                                for value in np.asarray(initial_environment_state).reshape(-1)
                            )

                    self._env_info[env_id]['step'] += 1
                    collected_step += 1

                self._env_info[env_id]['time'] += self._timer.value + interaction_duration
                
                # --- Episode Termination Handling ---
                if done:
                    collection_episode_index = collected_episode
                    collected_episode += 1
                    reward = info['eval_episode_return']
                    terminal_metadata = {
                        'env_id': int(env_id),
                        'env_episode_index': int(env_episode_indices[env_id]),
                        'collection_episode_index': int(collection_episode_index),
                        'episode_return': float(reward),
                        'done': True,
                        'time_limit_truncated': bool(
                            info.get('time_limit_truncated', info.get('TimeLimit.truncated', False))
                        ),
                        'reached_goal': bool(info.get('reached_goal', False)),
                        'visited_position_count': info.get('visited_position_count'),
                        'visited_pose_count': info.get('visited_pose_count'),
                        'visited_room_count': info.get('visited_room_count'),
                        'reachable_position_count': info.get('reachable_position_count'),
                        'room_count': info.get('room_count'),
                        'pure_exploration_episode': bool(
                            pure_exploration_episode_dict.get(env_id, False)
                        ),
                    }
                    log_info = {'reward': reward, 'time': self._env_info[env_id]['time'], 'step': self._env_info[env_id]['step']}
                    if store_collection_intrinsic_rewards:
                        log_info['intrinsic_reward_sum'] = float(eps_intrinsic_reward_sum_lst[env_id])
                        log_info['intrinsic_reward_max'] = float(eps_intrinsic_reward_max_lst[env_id])
                        log_info['intrinsic_reward_raw_sum'] = float(eps_intrinsic_reward_raw_sum_lst[env_id])
                        log_info['intrinsic_reward_raw_max'] = float(eps_intrinsic_reward_raw_max_lst[env_id])
                    if episodic_novelty_memories is not None:
                        novelty_count = max(1.0, float(eps_episodic_novelty_count_lst[env_id]))
                        log_info['episodic_novelty_pseudo_count_mean'] = float(
                            eps_intrinsic_reward_raw_sum_lst[env_id] / novelty_count
                        )
                        log_info['episodic_novelty_embedding_norm_mean'] = float(
                            eps_episodic_novelty_embedding_norm_sum_lst[env_id] / novelty_count
                        )
                        log_info['episodic_novelty_knn_squared_distance_mean'] = float(
                            eps_episodic_novelty_knn_squared_distance_sum_lst[env_id] / novelty_count
                        )
                        log_info['episodic_novelty_normalized_knn_distance_mean'] = float(
                            eps_episodic_novelty_normalized_distance_sum_lst[env_id] / novelty_count
                        )
                        log_info['episodic_novelty_similarity_mean'] = float(
                            eps_episodic_novelty_similarity_sum_lst[env_id] / novelty_count
                        )
                        log_info['episodic_novelty_unscaled_reward_mean'] = float(
                            eps_episodic_novelty_unscaled_reward_sum_lst[env_id] / novelty_count
                        )
                        log_info['episodic_novelty_warmup_fraction'] = float(
                            eps_episodic_novelty_warmup_sum_lst[env_id] / novelty_count
                        )
                        log_info['episodic_novelty_distance_scale_mean'] = float(
                            eps_episodic_novelty_distance_scale_sum_lst[env_id] / novelty_count
                        )
                    if not collect_with_pure_policy:
                        log_info['visit_entropy'] = visit_entropies_lst[env_id] / eps_steps_lst[env_id] if eps_steps_lst[env_id] > 0 else 0
                        if self.policy_config.gumbel_algo:
                            log_info['completed_value'] = completed_value_lst[env_id] / eps_steps_lst[env_id] if eps_steps_lst[env_id] > 0 else 0
                    self._episode_info.append(log_info)

                    # Pad and save the segment before the final one.
                    if last_game_segments[env_id] is not None:
                        # If the terminal transition exactly filled a segment, the current segment is empty
                        # and the held segment is the terminal episode segment. Otherwise, it is an older
                        # nonterminal segment being padded by the current final segment.
                        terminal_last_segment = len(game_segments[env_id]) == 0
                        self.pad_and_save_last_trajectory(
                            env_id, last_game_segments, last_game_priorities, game_segments, dones,
                            last_game_metadata=last_game_metadata,
                            segment_metadata=terminal_metadata if terminal_last_segment else None
                        )
                    
                    # Process and save the final segment of the episode.
                    if len(game_segments[env_id].reward_segment) > 0:
                        final_segment_metadata = {
                            **terminal_metadata,
                            'episode_segment_index': int(env_episode_segment_indices[env_id]),
                            'segment_length': int(len(game_segments[env_id])),
                        }
                        game_segments[env_id].set_terminal_metadata(
                            done=final_segment_metadata.get('done', False),
                            time_limit_truncated=final_segment_metadata.get('time_limit_truncated', False),
                            reached_goal=final_segment_metadata.get('reached_goal', False),
                            episode_return=final_segment_metadata.get('episode_return', None),
                        )
                        priorities = self._compute_priorities(
                            env_id, pred_values_lst, search_values_lst, game_segments[env_id]
                        )
                        game_segments[env_id].game_segment_to_array()
                        self.game_segment_pool.append((game_segments[env_id], priorities, dones[env_id], final_segment_metadata))

                    # Reset environment-specific states for a new episode.
                    if n_episode > self._env_num:
                        # Re-initialize the state for this env_id.
                        init_obs = self._env.ready_obs
                        while env_id not in init_obs:
                            self._logger.warning(f"Waiting for env {env_id} to reset...")
                            time.sleep(retry_waiting_time)
                            init_obs = self._env.ready_obs
                        
                        action_mask_dict[env_id] = to_ndarray(init_obs[env_id]['action_mask'])
                        to_play_dict[env_id] = to_ndarray(init_obs[env_id]['to_play'])
                        timestep_dict[env_id] = to_ndarray(init_obs[env_id].get('timestep', -1))
                        if self.policy_config.use_ture_chance_label_in_chance_encoder:
                           chance_dict[env_id] = to_ndarray(init_obs[env_id]['chance'])

                        # Reset game segment and observation stack.
                        game_segments[env_id] = GameSegment(self._env.action_space, game_segment_length=self.policy_config.game_segment_length, config=self.policy_config)
                        observation_window_stack[env_id].clear()
                        for _ in range(self.policy_config.model.frame_stack_num):
                            observation_window_stack[env_id].append(init_obs[env_id]['observation'])
                        game_segments[env_id].reset(observation_window_stack[env_id])
                        initial_environment_state = init_obs[env_id].get(
                            'environment_state', None
                        )
                        if initial_environment_state is not None:
                            game_segments[env_id].environment_initial_state = tuple(
                                int(value)
                                for value in np.asarray(initial_environment_state).reshape(-1)
                            )
                        last_game_segments[env_id] = None
                        last_game_priorities[env_id] = None
                        last_game_metadata[env_id] = None
                        if episodic_novelty_memories is not None:
                            episodic_novelty_memories[env_id].reset()
                    elif episodic_novelty_memories is not None:
                        episodic_novelty_memories[env_id].reset()

                    # Reset tracking and logging variables.
                    pred_values_lst[env_id], search_values_lst[env_id] = [], []
                    eps_steps_lst[env_id], visit_entropies_lst[env_id] = 0, 0
                    eps_intrinsic_reward_sum_lst[env_id] = 0
                    eps_intrinsic_reward_max_lst[env_id] = 0
                    eps_intrinsic_reward_raw_sum_lst[env_id] = 0
                    eps_intrinsic_reward_raw_max_lst[env_id] = 0
                    eps_episodic_novelty_count_lst[env_id] = 0
                    eps_episodic_novelty_embedding_norm_sum_lst[env_id] = 0
                    eps_episodic_novelty_knn_squared_distance_sum_lst[env_id] = 0
                    eps_episodic_novelty_normalized_distance_sum_lst[env_id] = 0
                    eps_episodic_novelty_similarity_sum_lst[env_id] = 0
                    eps_episodic_novelty_unscaled_reward_sum_lst[env_id] = 0
                    eps_episodic_novelty_warmup_sum_lst[env_id] = 0
                    eps_episodic_novelty_distance_scale_sum_lst[env_id] = 0
                    if self.policy_config.gumbel_algo:
                        completed_value_lst[env_id] = 0
                    env_episode_indices[env_id] += 1
                    env_episode_segment_indices[env_id] = 0

                    # Reset policy and collector stats for the finished environment.
                    self._policy.reset([env_id])
                    self._reset_stat(env_id)
                    ready_env_id.remove(env_id)

            # --- Check for Collection Completion ---
            if collected_episode >= n_episode:
                # Prepare data for returning.
                return_data = [
                    [item[0] for item in self.game_segment_pool],
                    [{
                        'priorities': item[1],
                        'done': item[2],
                        'unroll_plus_td_steps': self.unroll_plus_td_steps,
                        **(item[3] if len(item) > 3 and item[3] is not None else {}),
                    } for item in self.game_segment_pool]
                ]
                self.game_segment_pool.clear()
                break
        
        # --- Finalize and Log ---
        collected_duration = sum([d['time'] for d in self._episode_info])

        # NOTE: Only for usual DDP not for unizero_multitask pipeline.
        # In DDP, aggregate statistics across all processes.
        # if self._world_size > 1:
        #     collected_step = allreduce_data(collected_step, 'sum')
        #     collected_episode = allreduce_data(collected_episode, 'sum')
        #     collected_duration = allreduce_data(collected_duration, 'sum')

        self._total_envstep_count += collected_step
        self._total_episode_count += collected_episode
        self._total_duration += collected_duration

        self._output_log(train_iter)
        return return_data

    def _output_log(self, train_iter: int) -> None:
        """
        Overview:
            Aggregates and logs collection statistics to the console, TensorBoard, and WandB.
            This method is only executed by the rank 0 process in a distributed setup.
        Arguments:
            - train_iter (:obj:`int`): The current training iteration number, used as the logging step.
        """
        if self._rank != 0:
            return
        
        if (train_iter - self._last_train_iter) >= self._collect_print_freq and len(self._episode_info) > 0:
            self._last_train_iter = train_iter
            episode_count = len(self._episode_info)
            envstep_count = sum([d['step'] for d in self._episode_info])
            duration = sum([d['time'] for d in self._episode_info])
            episode_reward = [d['reward'] for d in self._episode_info]
            
            info = {
                'episode_count': episode_count,
                'envstep_count': envstep_count,
                'avg_envstep_per_episode': envstep_count / episode_count,
                'avg_envstep_per_sec': envstep_count / duration if duration > 0 else 0,
                'avg_episode_per_sec': episode_count / duration if duration > 0 else 0,
                'collect_time': duration,
                'reward_mean': np.mean(episode_reward),
                'reward_std': np.std(episode_reward),
                'reward_max': np.max(episode_reward),
                'reward_min': np.min(episode_reward),
                'total_envstep_count': self._total_envstep_count,
                'total_episode_count': self._total_episode_count,
                'total_duration': self._total_duration,
            }
            
            if not self.collect_with_pure_policy:
                visit_entropy = [d['visit_entropy'] for d in self._episode_info]
                info['visit_entropy_mean'] = np.mean(visit_entropy)
            store_collection_intrinsic_rewards = (
                bool(getattr(self.policy_config, 'rnd_store_collection_rewards', False))
                or bool(getattr(self.policy_config, 'qstd_store_collection_rewards', False))
                or bool(getattr(self.policy_config, 'state_value_disagreement_store_collection_rewards', False))
                or bool(getattr(self.policy_config, 'episodic_novelty_diagnostics', False))
            )
            if store_collection_intrinsic_rewards:
                for source_key in [
                    'intrinsic_reward_sum',
                    'intrinsic_reward_max',
                    'intrinsic_reward_raw_sum',
                    'intrinsic_reward_raw_max',
                ]:
                    values = [d[source_key] for d in self._episode_info if source_key in d]
                    if not values:
                        continue
                    info[f'{source_key}_mean'] = float(np.mean(values))
                    info[f'{source_key}_max'] = float(np.max(values))
            if bool(getattr(self.policy_config, 'episodic_novelty_diagnostics', False)):
                for source_key in [
                    'episodic_novelty_pseudo_count_mean',
                    'episodic_novelty_embedding_norm_mean',
                    'episodic_novelty_knn_squared_distance_mean',
                    'episodic_novelty_normalized_knn_distance_mean',
                    'episodic_novelty_similarity_mean',
                    'episodic_novelty_unscaled_reward_mean',
                    'episodic_novelty_warmup_fraction',
                    'episodic_novelty_distance_scale_mean',
                ]:
                    values = [d[source_key] for d in self._episode_info if source_key in d]
                    if values:
                        info[source_key] = float(np.mean(values))
            if self.policy_config.gumbel_algo:
                completed_value = [d['completed_value'] for d in self._episode_info]
                info['completed_value_mean'] = np.mean(completed_value)

            self._episode_info.clear()
            
            # Log to console
            self._logger.info("Collector Training Summary:\n{}".format('\n'.join([f'  {k}: {v}' for k, v in info.items()])))
            
            # Log to TensorBoard and WandB
            for k, v in info.items():
                if self.task_id is None:
                    tb_prefix_iter = f'{self._instance_name}_iter/'
                    tb_prefix_step = f'{self._instance_name}_step/'
                else:
                    tb_prefix_iter = f'{self._instance_name}_iter_task{self.task_id}/'
                    tb_prefix_step = f'{self._instance_name}_step_task{self.task_id}/'
                
                self._tb_logger.add_scalar(tb_prefix_iter + k, v, train_iter)
                self._tb_logger.add_scalar(tb_prefix_step + k, v, self._total_envstep_count)
            
            if self.policy_config.use_wandb:
                wandb_log_data = {tb_prefix_step + k: v for k, v in info.items()}
                wandb.log(wandb_log_data, step=self._total_envstep_count)
