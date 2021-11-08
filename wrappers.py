import time
import gym
import numpy as np
import copy

from gym.spaces import Box, Dict
from typing import Tuple

TimeStep = Tuple[np.ndarray, float, bool, dict]


class EpisodeMonitor(gym.ActionWrapper):
    """A class that computes episode returns and lengths."""
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action: np.ndarray) -> TimeStep:
        observation, reward, done, info = self.env.step(action)

        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info['total'] = {'timesteps': self.total_timesteps}

        if done:
            info['episode'] = {}
            info['episode']['return'] = self.reward_sum
            info['episode']['length'] = self.episode_length
            info['episode']['duration'] = time.time() - self.start_time

            if hasattr(self, 'get_normalized_score'):
                info['episode']['return'] = self.get_normalized_score(
                    info['episode']['return']) * 100.0

        return observation, reward, done, info

    def reset(self) -> np.ndarray:
        self._reset_stats()
        return self.env.reset()


class SinglePrecision(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        if isinstance(self.observation_space, Box):
            obs_space = self.observation_space
            self.observation_space = Box(obs_space.low, obs_space.high,
                                         obs_space.shape)
        elif isinstance(self.observation_space, Dict):
            obs_spaces = copy.copy(self.observation_space.spaces)
            for k, v in obs_spaces.items():
                obs_spaces[k] = Box(v.low, v.high, v.shape)
            self.observation_space = Dict(obs_spaces)
        else:
            raise NotImplementedError

    def observation(self, observation: np.ndarray) -> np.ndarray:
        if isinstance(observation, np.ndarray):
            return observation.astype(np.float32)
        elif isinstance(observation, dict):
            observation = copy.copy(observation)
            for k, v in observation.items():
                observation[k] = v.astype(np.float32)
            return observation
