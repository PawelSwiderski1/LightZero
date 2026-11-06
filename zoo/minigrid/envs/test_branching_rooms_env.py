import gymnasium as gym
import numpy as np
import pytest
import torch
from easydict import EasyDict

from lzero.model.muzero_model import MuZeroModel
from zoo.minigrid.envs.branching_rooms_env import (
    BRANCHING_ROOMS_ENV_ID,
    BRANCHING_ROOMS_HELDOUT_ENV_ID,
    BranchingRoomsEnv,
    BranchingRoomsHeldoutEnv,
)
from zoo.minigrid.envs.minigrid_lightzero_env import MiniGridEnvLightZero


@pytest.mark.envtest
class TestBranchingRoomsEnv:

    def test_topology_has_loops_and_five_dead_ends(self):
        graph = BranchingRoomsEnv.room_graph()
        discovered = {BranchingRoomsEnv.start_room}
        frontier = [BranchingRoomsEnv.start_room]
        while frontier:
            room = frontier.pop()
            for neighbor in graph[room]:
                if neighbor not in discovered:
                    discovered.add(neighbor)
                    frontier.append(neighbor)

        assert len(graph) == 16
        assert discovered == set(graph)
        assert BranchingRoomsEnv.dead_end_rooms() == {
            (0, 0),
            (0, 3),
            (3, 0),
            (3, 2),
            (3, 3),
        }
        assert len(BranchingRoomsEnv.room_edges) > len(graph) - 1

    def test_fixed_map_and_coverage_diagnostics(self):
        env = gym.make(BRANCHING_ROOMS_ENV_ID, max_steps=4)
        first_obs, first_info = env.reset(seed=3)
        first_grid = env.unwrapped.grid.encode().copy()
        second_obs, second_info = env.reset(seed=99)
        second_grid = env.unwrapped.grid.encode().copy()

        assert np.array_equal(first_grid, second_grid)
        assert np.array_equal(first_obs['image'], second_obs['image'])
        assert env.action_space.n == 3
        assert first_info['agent_position'] == BranchingRoomsEnv.start_position
        assert first_info['room_id'] == BranchingRoomsEnv.start_room
        assert first_info['reachable_position_count'] == 693
        assert first_info['visited_position_count'] == 1
        assert first_info['visited_room_count'] == 1
        assert first_info['position_coverage'] == pytest.approx(1 / 693)
        assert first_info['room_coverage'] == pytest.approx(1 / 16)

        for step in range(4):
            _, reward, terminated, truncated, info = env.step(env.unwrapped.actions.forward)
            assert reward == 0.0
            assert not terminated
            assert truncated == (step == 3)
        assert info['visited_position_count'] > 1
        assert info['position_coverage'] > first_info['position_coverage']
        env.close()

    def test_heldout_layout_matches_difficulty_but_changes_topology(self):
        graph = BranchingRoomsHeldoutEnv.room_graph()
        discovered = {BranchingRoomsHeldoutEnv.start_room}
        frontier = [BranchingRoomsHeldoutEnv.start_room]
        while frontier:
            room = frontier.pop()
            for neighbor in graph[room]:
                if neighbor not in discovered:
                    discovered.add(neighbor)
                    frontier.append(neighbor)

        assert discovered == set(graph)
        assert len(BranchingRoomsHeldoutEnv.room_edges) == len(
            BranchingRoomsEnv.room_edges
        )
        assert len(BranchingRoomsHeldoutEnv.dead_end_rooms()) == 5
        assert {
            BranchingRoomsHeldoutEnv._canonical_edge(*edge)
            for edge in BranchingRoomsHeldoutEnv.room_edges
        } != {
            BranchingRoomsEnv._canonical_edge(*edge)
            for edge in BranchingRoomsEnv.room_edges
        }

        training_env = gym.make(BRANCHING_ROOMS_ENV_ID)
        heldout_env = gym.make(BRANCHING_ROOMS_HELDOUT_ENV_ID)
        _, training_info = training_env.reset(seed=0)
        _, heldout_info = heldout_env.reset(seed=0)
        assert heldout_info['reachable_position_count'] == training_info[
            'reachable_position_count'
        ]
        assert not np.array_equal(
            training_env.unwrapped.grid.encode(),
            heldout_env.unwrapped.grid.encode(),
        )
        training_env.close()
        heldout_env.close()

    def test_lightzero_partial_rgb_mode(self):
        env = MiniGridEnvLightZero(
            EasyDict(
                env_id=BRANCHING_ROOMS_ENV_ID,
                flat_obs=False,
                observation_mode='rgb_partial',
                rgb_tile_size=12,
                scale_observation=True,
                agent_view_size=7,
                max_step=3,
                save_replay_gif=False,
                replay_path_gif=None,
            )
        )
        env.seed(7, dynamic_seed=False)
        obs = env.reset()

        assert obs['observation'].shape == (3, 84, 84)
        assert obs['observation'].dtype == np.float32
        assert 0.0 <= float(obs['observation'].min())
        assert float(obs['observation'].max()) <= 1.0
        assert obs['action_mask'].shape == (3, )
        assert tuple(obs['environment_state']) == (11, 11, 0, 1, 1)
        assert env.observation_space.shape == (3, 84, 84)
        assert env.action_space.n == 3
        assert env.reward_space.low.item() == 0.0
        assert env.reward_space.high.item() == 0.0
        assert env._last_reset_info['reachable_position_count'] == 693

        env.enable_save_replay('/tmp/branching_rooms_replay_test')
        assert env._save_replay_gif is True
        assert env._replay_path_gif == '/tmp/branching_rooms_replay_test'

        model = MuZeroModel(
            observation_shape=(3, 84, 84),
            action_space_size=3,
            num_channels=16,
            categorical_distribution=False,
            downsample=True,
        )
        model.eval()
        with torch.no_grad():
            initial_output = model.initial_inference(torch.from_numpy(obs['observation']).unsqueeze(0))
            recurrent_output = model.recurrent_inference(initial_output.latent_state, torch.tensor([2]))

        assert initial_output.latent_state.shape == (1, 16, 6, 6)
        assert initial_output.policy_logits.shape == (1, 3)
        assert recurrent_output.latent_state.shape == (1, 16, 6, 6)
        assert recurrent_output.reward.shape == (1, 1)

        for step in range(3):
            timestep = env.step(np.asarray([2], dtype=np.int64))
            assert timestep.obs['observation'].shape == (3, 84, 84)
            assert timestep.reward.item() == 0.0
            assert timestep.done == (step == 2)
            assert timestep.info['time_limit_truncated'] == (step == 2)
            assert 'position_coverage' in timestep.info
            assert 'room_coverage' in timestep.info
        env.close()
