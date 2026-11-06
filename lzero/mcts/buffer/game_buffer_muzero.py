from typing import Any, List, Tuple, Union, TYPE_CHECKING, Optional

import numpy as np
import torch
from ding.utils import BUFFER_REGISTRY, EasyTimer

from lzero.mcts.tree_search.mcts_ctree import MuZeroMCTSCtree as MCTSCtree
from lzero.mcts.tree_search.mcts_ptree import MuZeroMCTSPtree as MCTSPtree
from lzero.mcts.utils import prepare_observation
from lzero.policy import DiscreteSupport, to_detach_cpu_numpy, concat_output, concat_output_value, inverse_scalar_transform
from .game_buffer import GameBuffer
from .game_segment import should_bootstrap_at_trajectory_boundary

if TYPE_CHECKING:
    from lzero.policy import MuZeroPolicy, EfficientZeroPolicy, SampledEfficientZeroPolicy


@BUFFER_REGISTRY.register('game_buffer_muzero')
class MuZeroGameBuffer(GameBuffer):
    """
    Overview:
        The specific game buffer for MuZero policy.
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        """
        Overview:
            Use the default configuration mechanism. If a user passes in a cfg with a key that matches an existing key
            in the default configuration, the user-provided value will override the default configuration. Otherwise,
            the default configuration will be used.
        """
        default_config = self.default_config()
        default_config.update(cfg)
        self._cfg = default_config
        assert self._cfg.env_type in ['not_board_games', 'board_games']
        assert self._cfg.action_type in ['fixed_action_space', 'varied_action_space']
        self.replay_buffer_size = self._cfg.replay_buffer_size
        self.batch_size = self._cfg.batch_size
        self._alpha = self._cfg.priority_prob_alpha
        self._beta = self._cfg.priority_prob_beta

        self.keep_ratio = 1
        self.model_update_interval = 10
        self.num_of_collected_episodes = 0
        self.base_idx = 0
        self.clear_time = 0

        self.game_segment_buffer = []
        self.game_pos_priorities = []
        self.game_segment_game_pos_look_up = []

        self._compute_target_timer = EasyTimer()
        self._reuse_search_timer = EasyTimer()
        self._origin_search_timer = EasyTimer()
        self.buffer_reanalyze = False

        self.compute_target_re_time = 0
        self.reuse_search_time = 0
        self.origin_search_time = 0
        self.sample_times = 0
        self.active_root_num = 0
        self.current_train_iter = 0
        self.current_envstep = 0
        self.last_value_target_stats = {}

        if hasattr(self._cfg, 'task_id'):
            self.task_id = self._cfg.task_id
            print(f"Task ID is set to {self.task_id}.")
            try:
                self.action_space_size = self._cfg.model.action_space_size_list[self.task_id]
            except Exception as e:
                self.action_space_size = self._cfg.model.action_space_size

        else:
            self.task_id = None
            print("No task_id found in configuration. Task ID is set to None.")
            self.action_space_size = self._cfg.model.action_space_size
        self.value_support = DiscreteSupport(*self._cfg.model.value_support_range)
        self.reward_support = DiscreteSupport(*self._cfg.model.reward_support_range)

    def set_train_iter_env_step(self, train_iter: int, env_step: int) -> None:
        """
        Store learner progress so target construction can decide whether replay
        samples are old enough for Search-Based Value Estimation.
        """
        self.current_train_iter = int(train_iter)
        self.current_envstep = int(env_step)

    def _assign_target_policy_for_legal_actions(
            self, policy_tmp: List[float], policy: List[float], legal_actions: List[int]
    ) -> None:
        """
        Assign visit-count policy values into the target policy vector.

        Non-sampled algorithms use a full action-space policy vector, so each legal action is the destination index.
        Sampled algorithms override this method because their target policy vector is indexed by sampled-action slot.
        """
        for index, legal_action in enumerate(legal_actions):
            policy_tmp[legal_action] = policy[index]

    def _value_target_mode(self) -> str:
        mode = str(getattr(self._cfg, 'value_target_mode', 'auto')).lower()
        if mode == 'auto':
            if getattr(self._cfg, 'use_root_value', False) or getattr(
                    self._cfg, 'reanalyze_use_search_value_target', False
            ):
                return 'search_bootstrap'
            return 'td'
        if mode not in ['td', 'search_bootstrap', 'mixed_sve']:
            raise ValueError(
                "value_target_mode must be one of ['auto', 'td', 'search_bootstrap', 'mixed_sve'], "
                f"got {mode!r}"
            )
        return mode

    @staticmethod
    def _coerce_transition_index(index: Any) -> Optional[int]:
        if isinstance(index, (int, np.integer)):
            return int(index)
        if isinstance(index, str) and index.isdigit():
            return int(index)
        return None

    def _transition_index_with_offset(self, index: Any, offset: int) -> Optional[int]:
        base_index = self._coerce_transition_index(index)
        if base_index is None:
            return None
        return base_index + int(offset)

    def _use_sve_for_transition(self, transition_index: Optional[int], total_transitions: int) -> bool:
        if self._value_target_mode() != 'mixed_sve':
            return False
        if int(self.current_train_iter) < int(getattr(self._cfg, 'sve_start_train_iter', 0)):
            return False
        if transition_index is None:
            return False
        recent_window = int(getattr(self._cfg, 'sve_recent_window', 0))
        if recent_window > 0 and transition_index >= max(0, int(total_transitions) - recent_window):
            return False
        return True

    def reset_runtime_metrics(self):
        """
        Overview:
            Reset the runtime metrics of the buffer.
        """
        self.compute_target_re_time = 0
        self.reuse_search_time = 0
        self.origin_search_time = 0
        self.sample_times = 0
        self.active_root_num = 0

    def reanalyze_buffer(
            self, batch_size: int, policy: Union["MuZeroPolicy", "EfficientZeroPolicy", "SampledEfficientZeroPolicy"]
    ) -> List[Any]:
        """
        Overview:
            sample data from ``GameBuffer`` and prepare the current and target batch for training.
        Arguments:
            - batch_size (:obj:`int`): batch size.
            - policy (:obj:`Union["MuZeroPolicy", "EfficientZeroPolicy", "SampledEfficientZeroPolicy"]`): policy.
        Returns:
            - train_data (:obj:`List`): List of train data, including current_batch and target_batch.
        """
        policy._target_model.to(self._cfg.device)
        policy._target_model.eval()
        self.policy = policy
        # obtain the current_batch and prepare target context
        policy_re_context = self._make_batch_for_reanalyze(batch_size)
        # target policy
        self._compute_target_policy_reanalyzed(policy_re_context, policy._target_model)

    def _make_batch_for_reanalyze(self, batch_size: int) -> Tuple[Any]:
        """
        Overview:
            first sample orig_data through ``_sample_orig_data()``,
            then prepare the context of a batch:
                reward_value_context:        the context of reanalyzed value targets
                policy_re_context:           the context of reanalyzed policy targets
                policy_non_re_context:       the context of non-reanalyzed policy targets
                current_batch:                the inputs of batch
        Arguments:
            - batch_size (:obj:`int`): the batch size of orig_data from replay buffer.
        Returns:
            - context (:obj:`Tuple`): reward_value_context, policy_re_context, policy_non_re_context, current_batch
        """
        # obtain the batch context from replay buffer
        orig_data = self._sample_orig_reanalyze_batch(batch_size)
        game_segment_list, pos_in_game_segment_list, batch_index_list, weights_list, make_time_list = orig_data
        batch_size = len(batch_index_list)
        # obtain the context of reanalyzed policy targets
        policy_re_context = self._prepare_policy_reanalyzed_context(
            batch_index_list, game_segment_list,
            pos_in_game_segment_list
        )
        self.reanalyze_num = batch_size
        return policy_re_context

    def sample(
            self, batch_size: int, policy: Union["MuZeroPolicy", "EfficientZeroPolicy", "SampledEfficientZeroPolicy"]
    ) -> List[Any]:
        """
        Overview:
            sample data from ``GameBuffer`` and prepare the current and target batch for training.
        Arguments:
            - batch_size (:obj:`int`): batch size.
            - policy (:obj:`Union["MuZeroPolicy", "EfficientZeroPolicy", "SampledEfficientZeroPolicy"]`): policy.
        Returns:
            - train_data (:obj:`List`): List of train data, including current_batch and target_batch.
        """
        policy._target_model.to(self._cfg.device)
        policy._target_model.eval()

        # obtain the current_batch and prepare target context
        reward_value_context, policy_re_context, policy_non_re_context, current_batch = self._make_batch(
            batch_size, self._cfg.reanalyze_ratio
        )
        return self._build_train_data_from_context(
            reward_value_context, policy_re_context, policy_non_re_context, current_batch, policy
        )

    def sample_from_orig_data(
            self,
            orig_data: Tuple[Any],
            policy: Union["MuZeroPolicy", "EfficientZeroPolicy", "SampledEfficientZeroPolicy"],
    ) -> List[Any]:
        """
        Build a training batch from explicitly selected game segments/positions.

        This is used by auxiliary samplers, such as elite-episode replay, that
        choose roots outside the main replay buffer's PER distribution while
        still reusing the standard MuZero target construction.
        """
        policy._target_model.to(self._cfg.device)
        policy._target_model.eval()

        reward_value_context, policy_re_context, policy_non_re_context, current_batch = self._make_batch_from_orig_data(
            orig_data, self._cfg.reanalyze_ratio
        )
        return self._build_train_data_from_context(
            reward_value_context, policy_re_context, policy_non_re_context, current_batch, policy
        )

    def _build_train_data_from_context(
            self,
            reward_value_context: List[Any],
            policy_re_context: List[Any],
            policy_non_re_context: List[Any],
            current_batch: List[Any],
            policy: Union["MuZeroPolicy", "EfficientZeroPolicy", "SampledEfficientZeroPolicy"],
    ) -> List[Any]:
        """
        Convert prepared batch contexts into the train_data consumed by policy learning.
        """
        # target reward, target value
        target_reward_value = self._compute_target_reward_value(
            reward_value_context, policy._target_model
        )
        batch_member_target_values = None
        if len(target_reward_value) == 2:
            batch_rewards, batch_target_values = target_reward_value
        elif len(target_reward_value) == 3:
            batch_rewards, batch_target_values, batch_member_target_values = target_reward_value
        else:
            raise ValueError(
                f"Unexpected reward/value target output length: {len(target_reward_value)}"
            )
        with self._compute_target_timer:
            batch_target_policies_re = self._compute_target_policy_reanalyzed(policy_re_context, policy._target_model)
        self.compute_target_re_time += self._compute_target_timer.value

        batch_target_policies_non_re = self._compute_target_policy_non_reanalyzed(
            policy_non_re_context, self.action_space_size
        )

        # fusion of batch_target_policies_re and batch_target_policies_non_re to batch_target_policies
        if 0 < self._cfg.reanalyze_ratio < 1:
            batch_target_policies = np.concatenate([batch_target_policies_re, batch_target_policies_non_re])
        elif self._cfg.reanalyze_ratio == 1:
            batch_target_policies = batch_target_policies_re
        elif self._cfg.reanalyze_ratio == 0:
            batch_target_policies = batch_target_policies_non_re

        target_batch = [batch_rewards, batch_target_values, batch_target_policies]
        if batch_member_target_values is not None:
            target_batch.append(batch_member_target_values)

        # a batch contains the current_batch and the target_batch
        train_data = [current_batch, target_batch]
        if not self.buffer_reanalyze:
            self.sample_times += 1
        return train_data

    def _make_batch(self, batch_size: int, reanalyze_ratio: float) -> Tuple[Any]:
        """
        Overview:
            first sample orig_data through ``_sample_orig_data()``,
            then prepare the context of a batch:
                reward_value_context:        the context of reanalyzed value targets
                policy_re_context:           the context of reanalyzed policy targets
                policy_non_re_context:       the context of non-reanalyzed policy targets
                current_batch:                the inputs of batch
        Arguments:
            - batch_size (:obj:`int`): the batch size of orig_data from replay buffer.
            - reanalyze_ratio (:obj:`float`): ratio of reanalyzed policy (value is 100% reanalyzed)
        Returns:
            - context (:obj:`Tuple`): reward_value_context, policy_re_context, policy_non_re_context, current_batch
        """
        # obtain the batch context from replay buffer
        orig_data = self._sample_orig_data(batch_size)
        return self._make_batch_from_orig_data(orig_data, reanalyze_ratio)

    def _make_batch_from_orig_data(self, orig_data: Tuple[Any], reanalyze_ratio: float) -> Tuple[Any]:
        """
        Prepare contexts from explicit sampled roots.
        """
        game_segment_list, pos_in_game_segment_list, batch_index_list, weights_list, make_time_list = orig_data
        batch_size = len(batch_index_list)
        obs_list, action_list, mask_list = [], [], []
        intrinsic_reward_list = []
        pure_exploration_episode_list = []
        include_intrinsic_rewards = bool(getattr(self._cfg, 'store_selected_intrinsic_rewards', False))
        include_pure_exploration_episode_flags = bool(
            getattr(self._cfg, 'intrinsic_value_train_on_pure_exploration_only', False)
        )
        obs_unroll_steps = self._current_batch_obs_unroll_steps()
        qstd_recompute_actions = bool(
            getattr(self._cfg, 'intrinsic_reward_source', None) == 'qstd'
            and getattr(self._cfg, 'qstd_exploration', False)
            and getattr(self._cfg, 'qstd_recompute_replay_rewards', False)
        )
        action_unroll_steps = obs_unroll_steps if qstd_recompute_actions else self._cfg.num_unroll_steps
        # prepare the inputs of a batch
        for i in range(batch_size):
            game = game_segment_list[i]
            pos_in_game_segment = pos_in_game_segment_list[i]

            actions_tmp = game.action_segment[
                pos_in_game_segment:pos_in_game_segment + action_unroll_steps
            ].tolist()

            # add mask for invalid actions (out of trajectory), 1 for valid, 0 for invalid
            # mask_tmp = [1. for i in range(len(actions_tmp))]
            # mask_tmp += [0. for _ in range(self._cfg.num_unroll_steps + 1 - len(mask_tmp))]

            # TODO: the child_visits after position <self._cfg.game_segment_length> in the segment (with padded part) may not be updated
            # So the corresponding position should not be used in the training
            base_mask_len = self._cfg.num_unroll_steps + 1
            valid_base_steps = min(
                len(actions_tmp),
                self._cfg.num_unroll_steps,
                max(0, self._cfg.game_segment_length - pos_in_game_segment),
            )
            mask_tmp = [1. for _ in range(valid_base_steps)]
            mask_tmp += [0. for _ in range(base_mask_len - len(mask_tmp))]

            if obs_unroll_steps > base_mask_len:
                future_mask_len = obs_unroll_steps
                valid_future_steps = min(
                    future_mask_len,
                    max(0, len(game.action_segment) - pos_in_game_segment),
                )
                future_mask = [1. for _ in range(valid_future_steps)]
                future_mask += [0. for _ in range(future_mask_len - len(future_mask))]
                mask_tmp += future_mask

            # pad random action
            actions_tmp += [
                np.random.randint(0, game.action_space_size)
                for _ in range(action_unroll_steps - len(actions_tmp))
            ]
            if include_intrinsic_rewards:
                reward_steps = obs_unroll_steps
                intrinsic_rewards_tmp = []
                if hasattr(game, 'intrinsic_reward_segment'):
                    intrinsic_rewards_tmp = list(
                        game.intrinsic_reward_segment[pos_in_game_segment:pos_in_game_segment + reward_steps]
                    )
                intrinsic_rewards_tmp += [0.0 for _ in range(reward_steps - len(intrinsic_rewards_tmp))]
                intrinsic_reward_list.append(intrinsic_rewards_tmp[:reward_steps])
            if include_pure_exploration_episode_flags:
                pure_exploration_episode_list.append(
                    float(bool(getattr(game, 'pure_exploration_episode', False)))
                )

            # obtain the input observations
            # pad if length of obs in game_segment is less than stack+num_unroll_steps
            # e.g. stack+num_unroll_steps = 4+5
            obs_list.append(
                game_segment_list[i].get_unroll_obs(
                    pos_in_game_segment_list[i], num_unroll_steps=obs_unroll_steps, padding=True
                )
            )
            action_list.append(actions_tmp)
            mask_list.append(mask_tmp)

        # formalize the input observations
        obs_list = prepare_observation(obs_list, self._cfg.model.model_type)

        # formalize the inputs of a batch
        current_batch = [obs_list, action_list, mask_list]
        if include_intrinsic_rewards:
            current_batch.append(intrinsic_reward_list)
        if include_pure_exploration_episode_flags:
            current_batch.append(pure_exploration_episode_list)
        current_batch += [batch_index_list, weights_list, make_time_list]
        for i in range(len(current_batch)):
            current_batch[i] = np.asarray(current_batch[i])

        total_transitions = self.get_num_of_transitions()

        # obtain the context of value targets
        reward_value_context = self._prepare_reward_value_context(
            batch_index_list, game_segment_list, pos_in_game_segment_list, total_transitions
        )
        """
        only reanalyze recent reanalyze_ratio (e.g. 50%) data
        if self._cfg.reanalyze_outdated is True, batch_index_list is sorted according to its generated env_steps
        0: reanalyze_num -> reanalyzed policy, reanalyze_num:end -> non reanalyzed policy
        """
        reanalyze_num = int(batch_size * reanalyze_ratio)
        # reanalyzed policy
        if reanalyze_num > 0:
            # obtain the context of reanalyzed policy targets
            policy_re_context = self._prepare_policy_reanalyzed_context(
                batch_index_list[:reanalyze_num], game_segment_list[:reanalyze_num],
                pos_in_game_segment_list[:reanalyze_num]
            )
        else:
            policy_re_context = None

        # non reanalyzed policy
        if reanalyze_num < batch_size:
            # obtain the context of non-reanalyzed policy targets
            policy_non_re_context = self._prepare_policy_non_reanalyzed_context(
                batch_index_list[reanalyze_num:], game_segment_list[reanalyze_num:],
                pos_in_game_segment_list[reanalyze_num:]
            )
        else:
            policy_non_re_context = None

        context = reward_value_context, policy_re_context, policy_non_re_context, current_batch
        return context

    def _current_batch_obs_unroll_steps(self) -> int:
        obs_unroll_steps = int(self._cfg.num_unroll_steps)
        if (
                (
                    getattr(self._cfg, 'rnd_exploration', False)
                    and getattr(self._cfg, 'rnd_use_learned_intrinsic_model', True)
                )
                or (
                    getattr(self._cfg, 'intrinsic_reward_source', None) == 'dynamics_disagreement'
                    and getattr(self._cfg, 'dynamics_disagreement_train_intrinsic_value', False)
                )
                or (
                    getattr(self._cfg, 'intrinsic_reward_source', None) == 'qstd'
                    and getattr(self._cfg, 'qstd_exploration', False)
                )
                or getattr(self._cfg, 'intrinsic_reward_source', None) == 'state_value_disagreement'
                or getattr(self._cfg, 'intrinsic_reward_source', None) == 'state_density'
        ):
            intrinsic_td_steps = getattr(self._cfg, 'rnd_intrinsic_td_steps', None)
            if intrinsic_td_steps is None:
                intrinsic_td_steps = getattr(self._cfg, 'td_steps', obs_unroll_steps)
            obs_unroll_steps = obs_unroll_steps + max(1, int(intrinsic_td_steps))
        return obs_unroll_steps

    def _prepare_reward_value_context(
            self, batch_index_list: List[str], game_segment_list: List[Any], pos_in_game_segment_list: List[Any],
            total_transitions: int
    ) -> List[Any]:
        """
        Overview:
            prepare the context of rewards and values for calculating TD value target in reanalyzing part.
        Arguments:
            - batch_index_list (:obj:`list`): the index of start transition of sampled minibatch in replay buffer
            - game_segment_list (:obj:`list`): list of game segments
            - pos_in_game_segment_list (:obj:`list`): list of transition index in game_segment
            - total_transitions (:obj:`int`): number of collected transitions
        Returns:
            - reward_value_context (:obj:`list`): value_obs_list, value_mask, pos_in_game_segment_list, rewards_list, game_segment_lens,
              td_steps_list, action_mask_segment, to_play_segment
        """
        zero_obs = game_segment_list[0].zero_obs()
        value_obs_list = []
        current_obs_list = []
        sve_obs_list = []
        # the value is valid or not (out of game_segment)
        value_mask = []
        current_value_mask = []
        sve_value_mask = []
        transition_index_list = []
        rewards_list = []
        game_segment_lens = []
        # for board games
        action_mask_segment, to_play_segment = [], []

        root_values = []

        td_steps_list = []
        sve_prefix_steps_list = []
        sve_action_mask_list = []
        sve_to_play_list = []
        for batch_index, game_segment, state_index in zip(batch_index_list, game_segment_list, pos_in_game_segment_list):
            game_segment_len = len(game_segment)
            game_segment_lens.append(game_segment_len)
            current_game_obs = game_segment.get_unroll_obs(state_index, self._cfg.num_unroll_steps, padding=True)

            rewards_list.append(game_segment.reward_segment)
            
            # for board games
            action_mask_segment.append(game_segment.action_mask_segment)
            to_play_segment.append(game_segment.to_play_segment)

            truncation_length = game_segment_len
            time_limit_truncated = bool(getattr(game_segment, 'time_limit_truncated', False))
            bootstrap_at_time_limit = bool(getattr(self._cfg, 'bootstrap_at_time_limit', True))

            for current_index in range(state_index, state_index + self._cfg.num_unroll_steps + 1):
                offset = current_index - state_index
                transition_index_list.append(self._transition_index_with_offset(batch_index, offset))

                remaining_steps = max(0, truncation_length - current_index)
                effective_td_steps = min(int(self._cfg.td_steps), remaining_steps) if remaining_steps > 0 else 0
                td_steps_list.append(np.int32(effective_td_steps))
                bootstrap_index = current_index + effective_td_steps

                can_bootstrap = should_bootstrap_at_trajectory_boundary(
                    bootstrap_index,
                    truncation_length,
                    time_limit_truncated,
                    bootstrap_at_time_limit,
                )
                if remaining_steps > 0 and can_bootstrap:
                    value_mask.append(1)
                    obs = game_segment.get_unroll_obs(bootstrap_index, 0, padding=True)
                else:
                    value_mask.append(0)
                    obs = zero_obs

                value_obs_list.append(obs)

                if current_index < truncation_length:
                    current_value_mask.append(1)
                    beg_index = offset
                    end_index = beg_index + self._cfg.model.frame_stack_num
                    current_obs = current_game_obs[beg_index:end_index]
                else:
                    current_value_mask.append(0)
                    current_obs = zero_obs

                current_obs_list.append(current_obs)

                requested_sve_prefix = max(
                    0, int(getattr(self._cfg, 'sve_real_reward_prefix_steps', 0))
                )
                effective_sve_prefix = min(requested_sve_prefix, remaining_steps)
                sve_prefix_steps_list.append(np.int32(effective_sve_prefix))
                sve_bootstrap_index = current_index + effective_sve_prefix
                can_sve_bootstrap = remaining_steps > 0 and should_bootstrap_at_trajectory_boundary(
                    sve_bootstrap_index,
                    truncation_length,
                    time_limit_truncated,
                    bootstrap_at_time_limit,
                )
                if can_sve_bootstrap:
                    sve_value_mask.append(1)
                    sve_obs = game_segment.get_unroll_obs(sve_bootstrap_index, 0, padding=True)
                else:
                    sve_value_mask.append(0)
                    sve_obs = zero_obs
                sve_obs_list.append(sve_obs)

                metadata_length = len(game_segment.to_play_segment)
                if self._cfg.model.continuous_action_space is not True:
                    metadata_length = min(
                        metadata_length, len(game_segment.action_mask_segment)
                    )
                if sve_bootstrap_index < metadata_length:
                    sve_to_play_list.append(game_segment.to_play_segment[sve_bootstrap_index])
                    if self._cfg.model.continuous_action_space is not True:
                        sve_action_mask_list.append(game_segment.action_mask_segment[sve_bootstrap_index])
                else:
                    sve_to_play_list.append(-1)
                    if self._cfg.model.continuous_action_space is not True:
                        sve_action_mask_list.append(
                            np.ones(self._cfg.model.action_space_size, dtype=np.int8)
                        )

        reward_value_context = [
            value_obs_list, value_mask, pos_in_game_segment_list, rewards_list, root_values, game_segment_lens, td_steps_list,
            action_mask_segment, to_play_segment, current_obs_list, current_value_mask, transition_index_list,
            total_transitions, sve_obs_list, sve_value_mask, sve_prefix_steps_list, sve_action_mask_list,
            sve_to_play_list
        ]
        return reward_value_context

    def _prepare_policy_non_reanalyzed_context(
            self, batch_index_list: List[int], game_segment_list: List[Any], pos_in_game_segment_list: List[int]
    ) -> List[Any]:
        """
        Overview:
            prepare the context of policies for calculating policy target in non-reanalyzing part, just return the policy in self-play
        Arguments:
            - batch_index_list (:obj:`list`): the index of start transition of sampled minibatch in replay buffer
            - game_segment_list (:obj:`list`): list of game segments
            - pos_in_game_segment_list (:obj:`list`): list transition index in game
        Returns:
            - policy_non_re_context (:obj:`list`): pos_in_game_segment_list, child_visits, game_segment_lens, action_mask_segment, to_play_segment
        """
        child_visits = []
        game_segment_lens = []
        # for board games
        action_mask_segment, to_play_segment = [], []

        for game_segment, state_index, idx in zip(game_segment_list, pos_in_game_segment_list, batch_index_list):
            game_segment_len = len(game_segment)
            game_segment_lens.append(game_segment_len)
            # for board games
            action_mask_segment.append(game_segment.action_mask_segment)
            to_play_segment.append(game_segment.to_play_segment)

            child_visits.append(game_segment.child_visit_segment)

        policy_non_re_context = [
            pos_in_game_segment_list, child_visits, game_segment_lens, action_mask_segment, to_play_segment
        ]
        return policy_non_re_context

    def _prepare_policy_reanalyzed_context(
            self, batch_index_list: List[str], game_segment_list: List[Any], pos_in_game_segment_list: List[str]
    ) -> List[Any]:
        """
        Overview:
            prepare the context of policies for calculating policy target in reanalyzing part.
        Arguments:
            - batch_index_list (:obj:'list'): start transition index in the replay buffer
            - game_segment_list (:obj:'list'): list of game segments
            - pos_in_game_segment_list (:obj:'list'): position of transition index in one game history
        Returns:
            - policy_re_context (:obj:`list`): policy_obs_list, policy_mask, pos_in_game_segment_list, indices,
              child_visits, game_segment_lens, action_mask_segment, to_play_segment
        """
        zero_obs = game_segment_list[0].zero_obs()
        with torch.no_grad():
            # for policy
            policy_obs_list = []
            policy_mask = []
            # 0 -> Invalid target policy for padding outside of game segments,
            # 1 -> Previous target policy for game segments.
            rewards, child_visits, game_segment_lens, root_values = [], [], [], []
            # for board games
            action_mask_segment, to_play_segment = [], []
            for game_segment, state_index in zip(game_segment_list, pos_in_game_segment_list):
                game_segment_len = len(game_segment)
                game_segment_lens.append(game_segment_len)
                rewards.append(game_segment.reward_segment)
                # for board games
                action_mask_segment.append(game_segment.action_mask_segment)
                to_play_segment.append(game_segment.to_play_segment)

                child_visits.append(game_segment.child_visit_segment)
                root_values.append(game_segment.root_value_segment)
                # prepare the corresponding observations
                game_obs = game_segment.get_unroll_obs(state_index, self._cfg.num_unroll_steps)

                for current_index in range(state_index, state_index + self._cfg.num_unroll_steps + 1):

                    if current_index < game_segment_len: # original
                        policy_mask.append(1)
                        beg_index = current_index - state_index
                        end_index = beg_index + self._cfg.model.frame_stack_num
                        obs = game_obs[beg_index:end_index]
                    else:
                        policy_mask.append(0)
                        obs = zero_obs
                    policy_obs_list.append(obs)

        policy_re_context = [
            policy_obs_list, policy_mask, pos_in_game_segment_list, batch_index_list, child_visits, root_values, game_segment_lens,
            action_mask_segment, to_play_segment
        ]
        return policy_re_context

    def _scalar_reward(self, r: Any) -> float:
        """
        Overview:
            Convert a reward input of various types into a scalar float value.
        Arguments:
            - r (Any): The reward input, which can be a numpy array, list, tuple, or a scalar.
                    If it is a numpy array, list, or tuple, the function uses the first element.
        Returns:
            - float: The scalar representation of the input reward.
        """
        # If the reward is in the form of a list, tuple, or numpy array,
        # convert it to a numpy array, reshape it into a flat array, and take the first element.
        if isinstance(r, (list, tuple, np.ndarray)):
            r = np.asarray(r).reshape(-1)[0]
        
        # Return the float value of the reward.
        return float(r)

    def _use_search_value_target(self) -> bool:
        """
        Use MCTS searched values as the bootstrap term in normal MuZero value targets.

        ``use_root_value`` is the existing LightZero/EfficientZero knob.  The
        more explicit ``reanalyze_use_search_value_target`` is kept separate so
        experiment runners can enable this MuZero-specific comparison without
        changing older configs.

        In ``mixed_sve`` mode, normal/recent samples should stay anchored to
        real-reward TD targets with a raw network bootstrap.  Stale samples get
        MCTS search values through the explicit SVE override path instead.
        """
        mode = self._value_target_mode()
        if mode == 'search_bootstrap':
            return True
        return False

    def _search_values_from_network_output(
            self,
            network_output: List[Any],
            transition_batch_size: int,
            legal_actions: List[Any],
            to_play: List[int],
            model: Any,
            member_id: Optional[int] = None,
            add_root_noise: Optional[bool] = None,
    ) -> np.ndarray:
        _, reward_pool, policy_logits_pool, latent_state_roots = concat_output(network_output, data_type='muzero')
        reward_pool = np.asarray(reward_pool).reshape(-1).tolist()
        policy_logits_pool = np.asarray(policy_logits_pool).tolist()
        if add_root_noise is None:
            add_root_noise = bool(self._cfg.reanalyze_noise)
        noises = None
        if add_root_noise:
            noises = [
                np.random.dirichlet([self._cfg.root_dirichlet_alpha] * self.action_space_size
                                    ).astype(np.float32).tolist() for _ in range(transition_batch_size)
            ]
        active_member_state = getattr(model, 'active_member_id', None)
        if member_id is not None and hasattr(model, 'set_active_member'):
            model.set_active_member(member_id)
        try:
            if self._cfg.mcts_ctree:
                roots = MCTSCtree.roots(transition_batch_size, legal_actions)
                if add_root_noise:
                    roots.prepare(self._cfg.root_noise_weight, noises, reward_pool, policy_logits_pool, to_play)
                else:
                    roots.prepare_no_noise(reward_pool, policy_logits_pool, to_play)
                with self._origin_search_timer:
                    if self.task_id is not None:
                        MCTSCtree(self._cfg).search(roots, model, latent_state_roots, to_play, task_id=self.task_id)
                    else:
                        MCTSCtree(self._cfg).search(roots, model, latent_state_roots, to_play)
                self.origin_search_time += self._origin_search_timer.value
            else:
                roots = MCTSPtree.roots(transition_batch_size, legal_actions)
                if add_root_noise:
                    roots.prepare(self._cfg.root_noise_weight, noises, reward_pool, policy_logits_pool, to_play)
                else:
                    roots.prepare_no_noise(reward_pool, policy_logits_pool, to_play)
                with self._origin_search_timer:
                    if self.task_id is not None:
                        MCTSPtree(self._cfg).search(roots, model, latent_state_roots, to_play, task_id=self.task_id)
                    else:
                        MCTSPtree(self._cfg).search(roots, model, latent_state_roots, to_play)
                self.origin_search_time += self._origin_search_timer.value
        finally:
            if member_id is not None and active_member_state is not None:
                if isinstance(active_member_state, list) and hasattr(model, 'set_active_members'):
                    model.set_active_members(active_member_state)
                elif hasattr(model, 'set_active_member'):
                    model.set_active_member(int(active_member_state))
        return np.asarray(roots.get_values())

    def _infer_initial_outputs(self, obs_list: List[Any], model: Any, member_id: Optional[int] = None) -> List[Any]:
        obs_array = prepare_observation(obs_list, self._cfg.model.model_type)
        transition_batch_size = len(obs_array)
        network_output = []
        slices = int(np.ceil(transition_batch_size / self._cfg.mini_infer_size))
        for i in range(slices):
            beg_index = self._cfg.mini_infer_size * i
            end_index = self._cfg.mini_infer_size * (i + 1)
            m_obs = torch.from_numpy(obs_array[beg_index:end_index]).to(self._cfg.device)
            if member_id is not None and self.task_id is not None:
                m_output = model.initial_inference(m_obs, task_id=self.task_id, member_id=member_id)
            elif member_id is not None:
                m_output = model.initial_inference(m_obs, member_id=member_id)
            elif self.task_id is not None:
                m_output = model.initial_inference(m_obs, task_id=self.task_id)
            else:
                m_output = model.initial_inference(m_obs)
            [m_output.latent_state, m_output.value, m_output.policy_logits] = to_detach_cpu_numpy(
                [
                    m_output.latent_state,
                    inverse_scalar_transform(m_output.value, self.value_support),
                    m_output.policy_logits
                ]
            )
            network_output.append(m_output)
        return network_output

    def _ensemble_member_bootstrap_size(self) -> int:
        if not bool(getattr(self._cfg, 'ensemble_member_bootstrap_targets', False)):
            return 1
        model_cfg = getattr(self._cfg, 'model', {})
        return max(1, int(getattr(model_cfg, 'ensemble_size', 1)))

    def _discount_bootstrap_values(
            self,
            values: Any,
            value_mask: List[Any],
            td_steps_list: List[Any],
            transition_batch_size: int,
            to_play_segment: List[Any],
    ) -> np.ndarray:
        value_array = np.asarray(values).reshape(-1)
        if self._cfg.env_type == 'board_games' and to_play_segment[0][0] in [1, 2]:
            value_array = value_array * np.array(
                [
                    self._cfg.discount_factor ** td_steps_list[i]
                    if int(td_steps_list[i]) % 2 == 0
                    else -self._cfg.discount_factor ** td_steps_list[i]
                    for i in range(transition_batch_size)
                ]
            )
        else:
            value_array = value_array * (
                np.array([self._cfg.discount_factor for _ in range(transition_batch_size)]) ** td_steps_list
            )
        return value_array * np.array(value_mask)

    def _bootstrap_value_array_for_member(
            self,
            obs_list: List[Any],
            model: Any,
            value_mask: List[Any],
            td_steps_list: List[Any],
            transition_batch_size: int,
            to_play_segment: List[Any],
            use_search_value_target: bool,
            legal_actions: Optional[List[Any]] = None,
            to_play: Optional[List[int]] = None,
            member_id: Optional[int] = None,
    ) -> np.ndarray:
        network_output = self._infer_initial_outputs(obs_list, model, member_id=member_id)
        if use_search_value_target:
            value_list = self._search_values_from_network_output(
                network_output, transition_batch_size, legal_actions, to_play, model, member_id=member_id
            )
        else:
            value_list = concat_output_value(network_output)
        return self._discount_bootstrap_values(
            value_list, value_mask, td_steps_list, transition_batch_size, to_play_segment
        )

    def _member_search_value_overrides(
            self,
            obs_list: List[Any],
            flat_indices: List[int],
            legal_actions: List[Any],
            to_play: List[int],
            model: Any,
            ensemble_member_bootstrap_size: int,
    ) -> List[dict]:
        member_overrides = [{} for _ in range(ensemble_member_bootstrap_size)]
        for member_id in range(ensemble_member_bootstrap_size):
            search_network_output = self._infer_initial_outputs(obs_list, model, member_id=member_id)
            search_values = self._search_values_from_network_output(
                search_network_output,
                len(flat_indices),
                legal_actions,
                to_play,
                model,
                member_id=member_id,
                add_root_noise=bool(getattr(self._cfg, 'sve_search_noise', False)),
            )
            member_overrides[member_id] = {
                flat_index: float(value)
                for flat_index, value in zip(flat_indices, np.asarray(search_values).reshape(-1))
            }
        return member_overrides

    def _compute_target_reward_value(self, reward_value_context: List[Any], model: Any) -> Tuple[Any, ...]:
        """
        Overview:
            prepare reward and value targets from the context of rewards and values.
        Arguments:
            - reward_value_context (:obj:'list'): the reward value context
            - model (:obj:'torch.tensor'):model of the target model
        Returns:
            - batch_value_prefixs (:obj:'np.ndarray): batch of value prefix
            - batch_target_values (:obj:'np.ndarray): batch of value estimation
        """
        value_obs_list, value_mask, pos_in_game_segment_list, rewards_list, root_values, game_segment_lens, td_steps_list, action_mask_segment, \
        to_play_segment, current_obs_list, current_value_mask, transition_index_list, total_transitions, sve_obs_list, \
        sve_value_mask, sve_prefix_steps_list, sve_action_mask_list, sve_to_play_list = reward_value_context  # noqa
        # transition_batch_size = game_segment_batch_size * (num_unroll_steps+1)
        transition_batch_size = len(value_obs_list)
        use_search_value_target = self._use_search_value_target()
        use_mixed_sve = self._value_target_mode() == 'mixed_sve'
        if use_search_value_target:
            game_segment_batch_size = len(pos_in_game_segment_list)
            to_play, action_mask = self._preprocess_to_play_and_action_mask(
                game_segment_batch_size, to_play_segment, action_mask_segment, pos_in_game_segment_list
            )
            if self._cfg.model.continuous_action_space is True:
                action_mask = [
                    list(np.ones(self._cfg.model.num_of_sampled_actions, dtype=np.int8))
                    for _ in range(transition_batch_size)
                ]
                legal_actions = [
                    [-1 for _ in range(self._cfg.model.num_of_sampled_actions)]
                    for _ in range(transition_batch_size)
                ]
            else:
                legal_actions = [np.nonzero(action_mask[j])[0].tolist() for j in range(transition_batch_size)]

        batch_target_values, batch_rewards = [], []
        with torch.no_grad():
            ensemble_member_bootstrap_size = self._ensemble_member_bootstrap_size()
            member_value_array = None
            if ensemble_member_bootstrap_size > 1:
                member_value_array = np.stack(
                    [
                        self._bootstrap_value_array_for_member(
                            value_obs_list,
                            model,
                            value_mask,
                            td_steps_list,
                            transition_batch_size,
                            to_play_segment,
                            use_search_value_target,
                            legal_actions if use_search_value_target else None,
                            to_play if use_search_value_target else None,
                            member_id,
                        )
                        for member_id in range(ensemble_member_bootstrap_size)
                    ],
                    axis=0,
                )
                value_array = member_value_array[0].copy()
            else:
                value_array = self._bootstrap_value_array_for_member(
                    value_obs_list,
                    model,
                    value_mask,
                    td_steps_list,
                    transition_batch_size,
                    to_play_segment,
                    use_search_value_target,
                    legal_actions if use_search_value_target else None,
                    to_play if use_search_value_target else None,
                    None,
                )

            sve_value_overrides = {}
            member_sve_value_overrides = None
            if use_mixed_sve:
                if self._cfg.model.continuous_action_space is True:
                    sve_legal_actions = [
                        [-1 for _ in range(self._cfg.model.num_of_sampled_actions)]
                        for _ in range(transition_batch_size)
                    ]
                else:
                    sve_legal_actions = [
                        np.nonzero(sve_action_mask_list[j])[0].tolist()
                        for j in range(transition_batch_size)
                    ]

                sve_indices = [
                    index for index, (is_valid, transition_index) in enumerate(
                        zip(sve_value_mask, transition_index_list)
                    )
                    if bool(is_valid) and self._use_sve_for_transition(transition_index, total_transitions)
                ]
                if len(sve_indices) > 0:
                    sve_search_obs_list = [sve_obs_list[index] for index in sve_indices]
                    sve_legal_actions_subset = [sve_legal_actions[index] for index in sve_indices]
                    sve_to_play_subset = [sve_to_play_list[index] for index in sve_indices]
                    if ensemble_member_bootstrap_size > 1:
                        member_sve_value_overrides = self._member_search_value_overrides(
                            sve_search_obs_list,
                            sve_indices,
                            sve_legal_actions_subset,
                            sve_to_play_subset,
                            model,
                            ensemble_member_bootstrap_size,
                        )
                        sve_value_overrides = member_sve_value_overrides[0]
                    else:
                        member_sve_value_overrides = self._member_search_value_overrides(
                            sve_search_obs_list,
                            sve_indices,
                            sve_legal_actions_subset,
                            sve_to_play_subset,
                            model,
                            1,
                        )
                        sve_value_overrides = member_sve_value_overrides[0]

            value_list = value_array.tolist()
            batch_member_target_values = (
                [[] for _ in range(ensemble_member_bootstrap_size)]
                if member_value_array is not None
                else None
            )
            value_index = 0
            final_target_values_for_stats = []
            td_target_values_for_stats = []
            sve_final_target_values_for_stats = []
            sve_td_target_values_for_stats = []
            td_final_target_values_for_stats = []
            sve_prefix_steps_for_stats = []
            sve_prefix_returns_for_stats = []
            sve_search_values_for_stats = []
            sve_search_contributions_for_stats = []

            for game_segment_len_non_re, reward_list, state_index, to_play_list in zip(game_segment_lens, rewards_list,
                                                                                       pos_in_game_segment_list,
                                                                                       to_play_segment):
                target_values = []
                target_rewards = []
                member_target_values = (
                    [[] for _ in range(ensemble_member_bootstrap_size)]
                    if member_value_array is not None
                    else None
                )
                base_index = state_index

                truncation_length = game_segment_len_non_re

                for current_index in range(state_index, state_index + self._cfg.num_unroll_steps + 1):
                    bootstrap_index = current_index + td_steps_list[value_index]
                    for i, reward in enumerate(reward_list[current_index:bootstrap_index]):
                        if self._cfg.env_type == 'board_games' and to_play_segment[0][0] in [1, 2]:
                            # TODO(pu): for board_games, very important, to check
                            if to_play_list[base_index] == to_play_list[i]:
                                value_list[value_index] += reward * self._cfg.discount_factor ** i
                                if member_value_array is not None:
                                    member_value_array[:, value_index] += reward * self._cfg.discount_factor ** i
                            else:
                                value_list[value_index] += -reward * self._cfg.discount_factor ** i
                                if member_value_array is not None:
                                    member_value_array[:, value_index] += -reward * self._cfg.discount_factor ** i
                        else:
                            value_list[value_index] += reward * self._cfg.discount_factor ** i
                            if member_value_array is not None:
                                member_value_array[:, value_index] += reward * self._cfg.discount_factor ** i

                    # TODO: check the boundary condition
                    td_target_value = self._scalar_reward(value_list[value_index])
                    if value_index in sve_value_overrides:
                        sve_prefix_steps = int(sve_prefix_steps_list[value_index])
                        sve_prefix_return = 0.0
                        for i, reward in enumerate(
                                reward_list[current_index:current_index + sve_prefix_steps]
                        ):
                            scalar_reward = self._scalar_reward(reward)
                            if self._cfg.env_type == 'board_games' and to_play_segment[0][0] in [1, 2]:
                                if to_play_list[base_index] == to_play_list[i]:
                                    sve_prefix_return += scalar_reward * self._cfg.discount_factor ** i
                                else:
                                    sve_prefix_return += -scalar_reward * self._cfg.discount_factor ** i
                            else:
                                sve_prefix_return += scalar_reward * self._cfg.discount_factor ** i

                        sve_search_value = self._scalar_reward(sve_value_overrides[value_index])
                        sve_search_discount = self._cfg.discount_factor ** sve_prefix_steps
                        if (
                                self._cfg.env_type == 'board_games'
                                and to_play_segment[0][0] in [1, 2]
                                and sve_prefix_steps % 2 != 0
                        ):
                            sve_search_discount = -sve_search_discount
                        sve_search_contribution = sve_search_discount * sve_search_value
                        target_value = sve_prefix_return + sve_search_contribution
                    else:
                        target_value = value_list[value_index]
                    final_target_value = self._scalar_reward(target_value)
                    target_values.append(final_target_value)
                    td_target_values_for_stats.append(td_target_value)
                    final_target_values_for_stats.append(final_target_value)
                    if value_index in sve_value_overrides:
                        sve_final_target_values_for_stats.append(final_target_value)
                        sve_td_target_values_for_stats.append(td_target_value)
                        sve_prefix_steps_for_stats.append(float(sve_prefix_steps))
                        sve_prefix_returns_for_stats.append(float(sve_prefix_return))
                        sve_search_values_for_stats.append(float(sve_search_value))
                        sve_search_contributions_for_stats.append(float(sve_search_contribution))
                    else:
                        td_final_target_values_for_stats.append(final_target_value)
                    if member_value_array is not None:
                        for member_id in range(ensemble_member_bootstrap_size):
                            member_override = (
                                member_sve_value_overrides[member_id]
                                if member_sve_value_overrides is not None
                                else {}
                            )
                            if value_index in member_override:
                                member_search_value = self._scalar_reward(member_override[value_index])
                                member_target_value = sve_prefix_return + sve_search_discount * member_search_value
                            else:
                                member_target_value = member_value_array[member_id, value_index]
                            member_target_values[member_id].append(self._scalar_reward(member_target_value))
                    if current_index < len(reward_list):
                        target_rewards.append(self._scalar_reward(reward_list[current_index]))
                    else:
                        target_rewards.append(0.)

                    value_index += 1

                batch_rewards.append(target_rewards)
                batch_target_values.append(target_values)
                if batch_member_target_values is not None:
                    for member_id in range(ensemble_member_bootstrap_size):
                        batch_member_target_values[member_id].append(member_target_values[member_id])

        batch_rewards = np.asarray(batch_rewards)
        batch_target_values = np.asarray(batch_target_values)
        target_count = len(final_target_values_for_stats)
        sve_count = len(sve_final_target_values_for_stats)

        def _safe_mean(values: List[float]) -> float:
            return float(np.mean(values)) if len(values) > 0 else 0.0

        self.last_value_target_stats = {
            'value_target/count': float(target_count),
            'value_target/sve_count': float(sve_count),
            'value_target/sve_fraction': float(sve_count / target_count) if target_count > 0 else 0.0,
            'value_target/td_fraction': float(1.0 - sve_count / target_count) if target_count > 0 else 0.0,
            'value_target/final_mean': _safe_mean(final_target_values_for_stats),
            'value_target/td_candidate_mean': _safe_mean(td_target_values_for_stats),
            'value_target/td_final_mean': _safe_mean(td_final_target_values_for_stats),
            'value_target/sve_final_mean': _safe_mean(sve_final_target_values_for_stats),
            'value_target/sve_td_candidate_mean': _safe_mean(sve_td_target_values_for_stats),
            'value_target/sve_prefix_steps_mean': _safe_mean(sve_prefix_steps_for_stats),
            'value_target/sve_prefix_return_mean': _safe_mean(sve_prefix_returns_for_stats),
            'value_target/sve_search_value_mean': _safe_mean(sve_search_values_for_stats),
            'value_target/sve_search_contribution_mean': _safe_mean(sve_search_contributions_for_stats),
            'value_target/bootstrap_mask_fraction': float(np.mean(value_mask)) if len(value_mask) > 0 else 0.0,
            'value_target/bootstrap_at_time_limit': float(
                bool(getattr(self._cfg, 'bootstrap_at_time_limit', True))
            ),
            'value_target/current_train_iter': float(self.current_train_iter),
            'value_target/current_envstep': float(self.current_envstep),
            'value_target/replay_transitions': float(total_transitions),
        }
        if batch_member_target_values is not None:
            return batch_rewards, batch_target_values, np.asarray(batch_member_target_values)

        return batch_rewards, batch_target_values

    # @profile
    def _compute_target_policy_reanalyzed(self, policy_re_context: List[Any], model: Any) -> np.ndarray:
        """
        Overview:
            prepare policy targets from the reanalyzed context of policies
        Arguments:
            - policy_re_context (:obj:`List`): List of policy context to reanalyzed
        Returns:
            - batch_target_policies_re
        """
        if policy_re_context is None:
            return []
        batch_target_policies_re = []

        # for board games
        policy_obs_list, policy_mask, pos_in_game_segment_list, batch_index_list, child_visits, root_values, game_segment_lens, action_mask_segment, \
        to_play_segment = policy_re_context
        # transition_batch_size = game_segment_batch_size * (self._cfg.num_unroll_steps + 1)
        transition_batch_size = len(policy_obs_list)
        game_segment_batch_size = len(pos_in_game_segment_list)

        to_play, action_mask = self._preprocess_to_play_and_action_mask(
            game_segment_batch_size, to_play_segment, action_mask_segment, pos_in_game_segment_list
        )

        if self._cfg.model.continuous_action_space is True:
            # when the action space of the environment is continuous, action_mask[:] is None.
            action_mask = [
                list(np.ones(self._cfg.model.num_of_sampled_actions, dtype=np.int8)) for _ in range(transition_batch_size)
            ]
            # NOTE: in continuous action space env: we set all legal_actions as -1
            legal_actions = [
                [-1 for _ in range(self._cfg.model.num_of_sampled_actions)] for _ in range(transition_batch_size)
            ]
        else:
            legal_actions = [np.nonzero(action_mask[j])[0].tolist() for j in range(transition_batch_size)]

        with torch.no_grad():
            policy_obs_list = prepare_observation(policy_obs_list, self._cfg.model.model_type)
            # split a full batch into slices of mini_infer_size: to save the GPU memory for more GPU actors
            slices = int(np.ceil(transition_batch_size / self._cfg.mini_infer_size))
            network_output = []
            for i in range(slices):
                beg_index = self._cfg.mini_infer_size * i
                end_index = self._cfg.mini_infer_size * (i + 1)
                m_obs = torch.from_numpy(policy_obs_list[beg_index:end_index]).to(self._cfg.device)
                if self.task_id is not None:
                    m_output = model.initial_inference(m_obs, task_id=self.task_id)
                else:
                    m_output = model.initial_inference(m_obs)

                # if not in training, obtain the scalars of the value/reward
                [m_output.latent_state, m_output.value, m_output.policy_logits] = to_detach_cpu_numpy(
                    [
                        m_output.latent_state,
                        inverse_scalar_transform(m_output.value, self.value_support),
                        m_output.policy_logits
                    ]
                )

                network_output.append(m_output)

            _, reward_pool, policy_logits_pool, latent_state_roots = concat_output(network_output, data_type='muzero')
            reward_pool = reward_pool.squeeze().tolist()
            policy_logits_pool = policy_logits_pool.tolist()
            noises = [
                np.random.dirichlet([self._cfg.root_dirichlet_alpha] * self.action_space_size
                                    ).astype(np.float32).tolist() for _ in range(transition_batch_size)
            ]
            if self._cfg.mcts_ctree:
                # cpp mcts_tree
                roots = MCTSCtree.roots(transition_batch_size, legal_actions)
                if self._cfg.reanalyze_noise:
                    roots.prepare(self._cfg.root_noise_weight, noises, reward_pool, policy_logits_pool, to_play)
                else:
                    roots.prepare_no_noise(reward_pool, policy_logits_pool, to_play)
                # do MCTS for a new policy with the recent target model
                with self._origin_search_timer:
                    if self.task_id is not None:
                        MCTSCtree(self._cfg).search(roots, model, latent_state_roots, to_play, task_id=self.task_id)
                    else:
                        MCTSCtree(self._cfg).search(roots, model, latent_state_roots, to_play)
                
                self.origin_search_time += self._origin_search_timer.value
            else:
                # python mcts_tree
                roots = MCTSPtree.roots(transition_batch_size, legal_actions)
                if self._cfg.reanalyze_noise:
                    roots.prepare(self._cfg.root_noise_weight, noises, reward_pool, policy_logits_pool, to_play)
                else:
                    roots.prepare_no_noise(reward_pool, policy_logits_pool, to_play)
                # do MCTS for a new policy with the recent target model
                if self.task_id is not None:
                    MCTSPtree(self._cfg).search(roots, model, latent_state_roots, to_play, task_id=self.task_id)
                else:
                    MCTSPtree(self._cfg).search(roots, model, latent_state_roots, to_play)


            roots_legal_actions_list = legal_actions
            roots_distributions = roots.get_distributions()
            roots_values = roots.get_values()
            policy_index = 0
            # NOTE: It is very important to use the latest MCTS visit count distribution.
            for state_index, child_visit, root_value, game_index in zip(
                    pos_in_game_segment_list, child_visits, root_values, batch_index_list
            ):
                target_policies = []

                for current_index in range(state_index, state_index + self._cfg.num_unroll_steps + 1):
                    distributions = roots_distributions[policy_index]
                    searched_value = roots_values[policy_index]

                    if policy_mask[policy_index] == 0:
                        # NOTE: the invalid padding target policy, O is to make sure the corresponding cross_entropy_loss=0
                        target_policies.append([0 for _ in range(self.action_space_size)])
                    else:
                        # NOTE: It is very important to use the latest MCTS visit count distribution.
                        sum_visits = sum(distributions)
                        child_visit[current_index] = [visit_count / sum_visits for visit_count in distributions]

                        if distributions is None:
                            # if at some obs, the legal_action is None, add the fake target_policy
                            target_policies.append(
                                list(np.ones(self.action_space_size) / self.action_space_size)
                            )
                        else:
                            # Update the data in game segment:
                            # after the reanalyze search, new target policies and root values are obtained
                            # the target policies and root values are stored in the gamesegment, specifically, ``child_visit_segment`` and ``root_value_segment``
                            # we replace the data at the corresponding location with the latest search results to keep the most up-to-date targets
                            sim_num = sum(distributions)
                            child_visit[current_index] = [visit_count/sim_num for visit_count in distributions]
                            root_value[current_index] = searched_value
                            if self._cfg.action_type == 'fixed_action_space':
                                # for atari/classic_control/box2d environments that only have one player.
                                sum_visits = sum(distributions)
                                policy = [visit_count / sum_visits for visit_count in distributions]
                                target_policies.append(policy)
                            else:
                                # for board games that have two players and legal_actions is dy
                                policy_tmp = [0 for _ in range(self.action_space_size)]
                                # to make sure target_policies have the same dimension
                                sum_visits = sum(distributions)
                                policy = [visit_count / sum_visits for visit_count in distributions]
                                self._assign_target_policy_for_legal_actions(
                                    policy_tmp, policy, roots_legal_actions_list[policy_index]
                                )
                                target_policies.append(policy_tmp)

                    policy_index += 1

                batch_target_policies_re.append(target_policies)

        batch_target_policies_re = np.array(batch_target_policies_re)

        return batch_target_policies_re

    def _compute_target_policy_non_reanalyzed(
            self, policy_non_re_context: List[Any], policy_shape: Optional[int]
    ) -> np.ndarray:
        """
        Overview:
            prepare policy targets from the non-reanalyzed context of policies
        Arguments:
            - policy_non_re_context (:obj:`List`): List containing:
                - pos_in_game_segment_list
                - child_visits
                - game_segment_lens
                - action_mask_segment
                - to_play_segment
            - policy_shape: self.action_space_size
        Returns:
            - batch_target_policies_non_re
        """
        batch_target_policies_non_re = []
        if policy_non_re_context is None:
            return batch_target_policies_non_re

        pos_in_game_segment_list, child_visits, game_segment_lens, action_mask_segment, to_play_segment = policy_non_re_context
        game_segment_batch_size = len(pos_in_game_segment_list)
        transition_batch_size = game_segment_batch_size * (self._cfg.num_unroll_steps + 1)

        to_play, action_mask = self._preprocess_to_play_and_action_mask(
            game_segment_batch_size, to_play_segment, action_mask_segment, pos_in_game_segment_list
        )

        if self._cfg.model.continuous_action_space is True:
            # when the action space of the environment is continuous, action_mask[:] is None.
            action_mask = [
                list(np.ones(self._cfg.model.num_of_sampled_actions, dtype=np.int8)) for _ in range(transition_batch_size)
            ]
            # NOTE: in continuous action space env: we set all legal_actions as -1
            legal_actions = [
                [-1 for _ in range(self.action_space_size)] for _ in range(transition_batch_size)
            ]
        else:
            legal_actions = [np.nonzero(action_mask[j])[0].tolist() for j in range(transition_batch_size)]
            
        with torch.no_grad():
            policy_index = 0
            # 0 -> Invalid target policy for padding outside of game segments,
            # 1 -> Previous target policy for game segments.
            policy_mask = []
            for game_segment_len, child_visit, state_index in zip(game_segment_lens, child_visits,
                                                                  pos_in_game_segment_list):
                target_policies = []

                for current_index in range(state_index, state_index + self._cfg.num_unroll_steps + 1):
                    if current_index < game_segment_len:
                        policy_mask.append(1)
                        # NOTE: child_visit is already a distribution
                        distributions = child_visit[current_index]
                        if self._cfg.action_type == 'fixed_action_space':
                            # for atari/classic_control/box2d environments that only have one player.
                            target_policies.append(distributions)
                        else:
                            # for board games that have two players or envs that have varied action space.
                            policy_tmp = [0 for _ in range(policy_shape)]
                            # only the action in ``legal_action`` the policy logits is nonzero
                            self._assign_target_policy_for_legal_actions(
                                policy_tmp, distributions, legal_actions[policy_index]
                            )
                            target_policies.append(policy_tmp)
                    else:
                        # NOTE: the invalid padding target policy, O is to make sure the corresponding cross_entropy_loss=0
                        policy_mask.append(0)
                        target_policies.append([0 for _ in range(policy_shape)])

                    policy_index += 1

                batch_target_policies_non_re.append(target_policies)
        batch_target_policies_non_re = np.asarray(batch_target_policies_non_re)
        return batch_target_policies_non_re

    def update_priority(self, train_data: List[np.ndarray], batch_priorities: Any) -> None:
        """
        Overview:
            Update the priority of training data.
        Arguments:
            - train_data (:obj:`List[np.ndarray]`): training data to be updated priority.
            - batch_priorities (:obj:`batch_priorities`): priorities to update to.
        NOTE:
            train_data = [current_batch, target_batch]
            current_batch = [obs_list, action_list, improved_policy_list(only in Gumbel MuZero), mask_list, batch_index_list, weights, make_time_list]
            target_batch = [batch_rewards, batch_target_values, batch_target_policies]
        """
        indices = train_data[0][-3]
        metas = {'make_time': train_data[0][-1], 'batch_priorities': batch_priorities}
        # only update the priorities for data still in replay buffer
        for i in range(len(indices)):
            if metas['make_time'][i] > self.clear_time:
                idx, prio = indices[i], metas['batch_priorities'][i]
                self.game_pos_priorities[idx] = prio
