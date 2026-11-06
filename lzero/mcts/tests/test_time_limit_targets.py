import numpy as np
import pytest
from easydict import EasyDict

from lzero.mcts.buffer.game_buffer_muzero import MuZeroGameBuffer


class _TimeoutSegment:

    def __init__(self):
        self.reward_segment = [
            np.asarray([-1.0], dtype=np.float32),
            np.asarray([-1.0], dtype=np.float32),
        ]
        self.action_mask_segment = [np.ones(3, dtype=np.int8) for _ in range(2)]
        self.to_play_segment = [-1, -1]
        self.time_limit_truncated = True

    def __len__(self):
        return 2

    def zero_obs(self):
        return [np.zeros(3, dtype=np.float32)]

    def get_unroll_obs(self, timestep, num_unroll_steps, padding):
        return [np.asarray([float(timestep), 0.0, 0.0], dtype=np.float32)]


def _timeout_value_mask(bootstrap_at_time_limit: bool):
    buffer = object.__new__(MuZeroGameBuffer)
    buffer._cfg = EasyDict(
        num_unroll_steps=0,
        td_steps=20,
        bootstrap_at_time_limit=bootstrap_at_time_limit,
        sve_real_reward_prefix_steps=2,
        model=EasyDict(
            frame_stack_num=1,
            continuous_action_space=False,
            action_space_size=3,
        ),
    )
    context = buffer._prepare_reward_value_context(
        batch_index_list=['0'],
        game_segment_list=[_TimeoutSegment()],
        pos_in_game_segment_list=[0],
        total_transitions=2,
    )
    return context[1], context[6]


@pytest.mark.unittest
def test_finite_horizon_timeout_disables_final_value_bootstrap():
    value_mask, td_steps = _timeout_value_mask(bootstrap_at_time_limit=False)
    assert value_mask == [0]
    assert td_steps == [2]


@pytest.mark.unittest
def test_continuing_timeout_preserves_final_value_bootstrap():
    value_mask, td_steps = _timeout_value_mask(bootstrap_at_time_limit=True)
    assert value_mask == [1]
    assert td_steps == [2]


@pytest.mark.unittest
def test_mixed_sve_retains_real_reward_prefix_before_search_value():
    buffer = object.__new__(MuZeroGameBuffer)
    buffer._cfg = EasyDict(
        value_target_mode='mixed_sve',
        sve_start_train_iter=0,
        sve_recent_window=0,
        reanalyze_use_search_value_target=False,
        use_root_value=False,
        ensemble_member_bootstrap_targets=False,
        num_unroll_steps=0,
        discount_factor=0.9,
        env_type='not_board_games',
        model=EasyDict(
            continuous_action_space=False,
            ensemble_size=1,
            action_space_size=3,
        ),
    )
    buffer.current_train_iter = 1
    buffer.current_envstep = 3
    buffer.last_value_target_stats = {}
    buffer._bootstrap_value_array_for_member = lambda *args, **kwargs: np.asarray([0.0])
    buffer._member_search_value_overrides = lambda *args, **kwargs: [{0: 10.0}]

    obs = [np.zeros(3, dtype=np.float32)]
    action_mask = np.ones(3, dtype=np.int8)
    reward_value_context = [
        [obs],                         # 20-step bootstrap observations
        [1],                           # 20-step bootstrap mask
        [0],                           # sampled positions
        [[-1.0, -1.0, -1.0]],         # real rewards
        [],                            # unused root values
        [3],                           # segment lengths
        [3],                           # effective TD steps
        [[action_mask] * 3],           # action masks
        [[-1, -1, -1]],                # players
        [obs],                         # current observations
        [1],                           # current masks
        [0],                           # replay transition indices
        3,                             # replay size
        [obs],                         # SVE endpoint observations
        [1],                           # SVE endpoint masks
        [2],                           # real-reward prefix lengths
        [action_mask],                 # SVE endpoint action masks
        [-1],                          # SVE endpoint players
    ]

    batch_rewards, batch_values = buffer._compute_target_reward_value(reward_value_context, model=None)

    # -1 + 0.9 * -1 + 0.9^2 * 10 = 6.2
    np.testing.assert_allclose(batch_rewards, [[-1.0]])
    np.testing.assert_allclose(batch_values, [[6.2]])
    assert buffer.last_value_target_stats['value_target/sve_prefix_steps_mean'] == pytest.approx(2.0)
    assert buffer.last_value_target_stats['value_target/sve_prefix_return_mean'] == pytest.approx(-1.9)
    assert buffer.last_value_target_stats['value_target/sve_search_value_mean'] == pytest.approx(10.0)
    assert buffer.last_value_target_stats['value_target/sve_search_contribution_mean'] == pytest.approx(8.1)


@pytest.mark.unittest
def test_mixed_sve_uses_each_ensemble_members_search_bootstrap():
    buffer = object.__new__(MuZeroGameBuffer)
    buffer._cfg = EasyDict(
        value_target_mode='mixed_sve',
        sve_start_train_iter=0,
        sve_recent_window=0,
        reanalyze_use_search_value_target=False,
        use_root_value=False,
        ensemble_member_bootstrap_targets=True,
        num_unroll_steps=0,
        discount_factor=0.9,
        env_type='not_board_games',
        model=EasyDict(
            continuous_action_space=False,
            ensemble_size=3,
            action_space_size=3,
        ),
    )
    buffer.current_train_iter = 1
    buffer.current_envstep = 3
    buffer.last_value_target_stats = {}
    buffer._bootstrap_value_array_for_member = lambda *args, **kwargs: np.asarray([0.0])
    buffer._member_search_value_overrides = lambda *args, **kwargs: [
        {0: 10.0},
        {0: 20.0},
        {0: 30.0},
    ]

    obs = [np.zeros(3, dtype=np.float32)]
    action_mask = np.ones(3, dtype=np.int8)
    reward_value_context = [
        [obs], [1], [0], [[-1.0, -1.0, -1.0]], [], [3], [3],
        [[action_mask] * 3], [[-1, -1, -1]], [obs], [1], [0], 3,
        [obs], [1], [2], [action_mask], [-1],
    ]

    _, batch_values, member_values = buffer._compute_target_reward_value(
        reward_value_context, model=None
    )

    np.testing.assert_allclose(batch_values, [[6.2]])
    np.testing.assert_allclose(member_values[:, 0, 0], [6.2, 14.3, 22.4])
