import pytest
import numpy as np
from easydict import EasyDict
from zoo.classic_control.mountain_car.envs import MountainCarEnv


@pytest.mark.envtest
class TestMountainCarEnv:
    def test_naive(self):
        env = MountainCarEnv(cfg={})
        env.seed(314, dynamic_seed=False)
        assert env._seed == 314
        obs = env.reset()
        assert obs['observation'].shape == (2, )
        for _ in range(5):
            env.reset()
            np.random.seed(314)
            print('=' * 60)
            for i in range(10):
                # Both ``env.random_action()``, and utilizing ``np.random`` as well as action space,
                # can generate legal random action.
                if i < 5:
                    random_action = np.array([env.action_space.sample()])
                else:
                    random_action = env.random_action()
                timestep = env.step(random_action)
                print("Action taken : ", random_action)
                print(timestep, "\n")
                # assert isinstance(timestep.obs, np.ndarray)
                assert isinstance(timestep.done, bool)
                assert timestep.obs['observation'].shape == (2, )
                assert timestep.reward.shape == (1, )
                assert timestep.reward >= env.reward_space.low
                assert timestep.reward <= env.reward_space.high
        print(env.observation_space, env.action_space, env.reward_space)
        env.close()

    def test_remaining_time_observation(self):
        env = MountainCarEnv(
            cfg=EasyDict(
                max_episode_steps=200,
                include_remaining_time=True,
            )
        )
        env.seed(314, dynamic_seed=False)
        obs = env.reset()

        assert env.observation_space.shape == (3, )
        assert obs['observation'].shape == (3, )
        assert obs['observation'][2] == pytest.approx(1.0)

        env._timestep = 200
        final_obs = env._format_observation(np.asarray([-0.5, 0.0], dtype=np.float32))
        assert final_obs.shape == (3, )
        assert final_obs[2] == pytest.approx(0.0)
        env.close()
