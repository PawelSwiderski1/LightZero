import copy
import os
from typing import Any, List, Union, Optional
import gym
import numpy as np
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.torch_utils import to_ndarray, to_list
from ding.utils import ENV_REGISTRY


@ENV_REGISTRY.register('mountain_car_lightzero')
class MountainCarEnv(BaseEnv):
    """
    Implementation of LightZero's version of the Mountain Car deterministic MDP. 

    Important references that contributed to the creation of this env:
    > Source code of OpenAI's mountain car gym : https://is.gd/y1FkMT
    > Gym documentation of mountain car : https://is.gd/29S0dt
    > Based off LightZero existing implementation of cartpole_env.py
    > LightZero's env creation conventions : https://is.gd/ZHLISj

    Only __init__ , step, seed and reset are mandatory & impt.
    The other methods are generally for convenience.
    """

    def __init__(self, cfg) -> None:
        self._cfg = cfg
        self._init_flag = False
        self._replay_path = None
        self._record_video = False
        self._env_id = int(getattr(cfg, 'env_id', 0))
        self._collect_video_phase = 0
        self._max_episode_steps = int(getattr(cfg, 'max_episode_steps', 200))
        if self._max_episode_steps <= 0:
            raise ValueError(f"max_episode_steps must be positive, got {self._max_episode_steps}")
        self._include_remaining_time = bool(getattr(cfg, 'include_remaining_time', False))

        # Following specifications from https://is.gd/29S0dt
        observation_low = [-1.2, -0.07]
        observation_high = [0.6, 0.07]
        if self._include_remaining_time:
            observation_low.append(0.0)
            observation_high.append(1.0)
        self._observation_space = gym.spaces.Box(
            low=np.array(observation_low, dtype=np.float32),
            high=np.array(observation_high, dtype=np.float32),
            dtype=np.float32
        )
        self._action_space = gym.spaces.Discrete(3, start=0)
        self._reward_space = gym.spaces.Box(low=-1, high=0.0, shape=(1, ), dtype=np.float32)
        self._timestep = 0

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def reset(self) -> np.ndarray:
        # Instantiate environment if not already done so
        if not self._init_flag:
            self._env = gym.make('MountainCar-v0')
            self._init_flag = True

        # Check if we have a valid replay path and save replay video accordingly
        if self._replay_path is not None and not self._record_video:
            self._env = gym.wrappers.RecordVideo(
                self._env,
                video_folder=self._replay_path,
                episode_trigger=lambda episode_id: True,
                name_prefix=self._collect_video_prefix(),
            )
            self._record_video = True

        # Set the seeds for randomization.
        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._env.seed(self._seed + np_seed)
            self._action_space.seed(self._seed + np_seed)
        elif hasattr(self, '_seed'):
            self._env.seed(self._seed)
            self._action_space.seed(self._seed)

        # Get first observation from original environment
        obs = self._env.reset()

        # Init final reward : cumulative sum of the real rewards obtained by a whole episode,
        # used to evaluate the agent Performance on this environment, not used for training.
        self._eval_episode_return = 0.

        action_mask = np.ones(self.action_space.n, 'int8')
        self._timestep = 0
        obs = self._format_observation(obs)

        obs = {'observation': obs, 'action_mask': action_mask, 'to_play': -1, 'timestep': self._timestep}
        return obs

    def step(self, action: np.ndarray) -> BaseEnvTimestep:
        # Making sure that input action is of numpy ndarray
        # assert isinstance(action, np.ndarray), type(action)

        # Accept either the usual 0-dim/1-dim ndarray action or a scalar int
        # from custom collection policies.
        action = int(np.asarray(action).squeeze())
        # Take a step of faith into the unknown!
        obs, rew, done, info = self._env.step(action)
        reached_goal = bool(np.asarray(obs).reshape(-1)[0] >= self._env.unwrapped.goal_position)

        # Cummulate reward
        self._eval_episode_return += rew

        # Save final cummulative reward when done.
        if done:
            info['eval_episode_return'] = self._eval_episode_return
            info['reached_goal'] = reached_goal
            info['time_limit_truncated'] = bool(not reached_goal)

        # Making sure we conform to di-engine conventions
        rew = to_ndarray([rew]).astype(np.float32)
        action_mask = np.ones(self.action_space.n, 'int8')
        self._timestep += 1
        obs = self._format_observation(obs)

        obs = {'observation': obs, 'action_mask': action_mask, 'to_play': -1, 'timestep': self._timestep}

        return BaseEnvTimestep(obs, rew, done, info)

    def _format_observation(self, obs: Any) -> np.ndarray:
        obs = to_ndarray(obs).astype(np.float32).reshape(-1)
        if not self._include_remaining_time:
            return obs
        remaining_time = 1.0 - min(self._timestep, self._max_episode_steps) / self._max_episode_steps
        return np.concatenate([obs, np.asarray([remaining_time], dtype=np.float32)])

    def close(self) -> None:
        # If init flag is False, then reset() was never run, no point closing.
        if self._init_flag:
            self._close_video_recorder(delete_tiny=True)
            self._env.close()
        self._init_flag = False
        self._record_video = False

    def enable_save_replay(self, replay_path: Optional[str] = None) -> None:
        if replay_path is None:
            replay_path = './video'
        self._replay_path = replay_path

    def set_collect_video_phase(self, phase: int) -> None:
        if int(phase) == self._collect_video_phase:
            return
        self._collect_video_phase = int(phase)
        if self._record_video and hasattr(self._env, 'close_video_recorder'):
            self._close_video_recorder(delete_tiny=True)
            self._env.name_prefix = self._collect_video_prefix()
            self._env.episode_id = 0
            self._env.step_id = 0
            self._env.terminated = False
            self._env.truncated = False

    def _collect_video_prefix(self) -> str:
        return f'env-{self._env_id}_collection_phase-{self._collect_video_phase}'

    def _close_video_recorder(self, delete_tiny: bool = False) -> None:
        if not hasattr(self._env, 'close_video_recorder'):
            return
        video_recorder = getattr(self._env, 'video_recorder', None)
        video_path = getattr(video_recorder, 'path', None)
        try:
            self._env.close_video_recorder()
        except (FileNotFoundError, NameError):
            if video_recorder is not None:
                video_recorder._closed = True
        finally:
            if hasattr(self._env, 'video_recorder'):
                self._env.video_recorder = None
            if hasattr(self._env, 'recording'):
                self._env.recording = False
        if not delete_tiny or not video_path or not os.path.exists(video_path):
            return
        if os.path.getsize(video_path) > 5000:
            return
        os.remove(video_path)
        meta_path = os.path.splitext(video_path)[0] + '.meta.json'
        if os.path.exists(meta_path):
            os.remove(meta_path)

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        collector_env_num = cfg.pop('collector_env_num')
        env_cfg_list = []
        for env_id in range(collector_env_num):
            env_cfg = copy.deepcopy(cfg)
            env_cfg.env_id = env_id
            env_cfg_list.append(env_cfg)
        return env_cfg_list

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_env_num = cfg.pop('evaluator_env_num')
        env_cfg_list = []
        for env_id in range(evaluator_env_num):
            env_cfg = copy.deepcopy(cfg)
            env_cfg.env_id = env_id
            env_cfg_list.append(env_cfg)
        return env_cfg_list

    def random_action(self) -> np.ndarray:
        random_action = self.action_space.sample()
        random_action = to_ndarray([random_action], dtype=np.int64)
        return random_action

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_space

    def __repr__(self) -> str:
        return "LightZero Mountain Car Env"
