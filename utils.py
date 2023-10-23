"""Implements common utilities for training."""

from typing import Tuple, Union

import collections
import numpy as np
import gym
import d4rl

from tqdm import tqdm

Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations'])


def normalize(dataset: np.ndarray):
    """Splits and normalizes the dataset."""

    trajs = split_into_trajectories(dataset.observations, dataset.actions,
                                    dataset.rewards, dataset.masks,
                                    dataset.dones_float,
                                    dataset.next_observations)

    def compute_returns(traj):
        episode_return = 0
        for _, _, rew, _, _, _ in traj:
            episode_return += rew

        return episode_return

    trajs.sort(key=compute_returns)

    dataset.rewards /= compute_returns(trajs[-1]) - compute_returns(trajs[0])
    dataset.rewards *= 1000.0


def split_into_trajectories(observations: np.ndarray,
                            actions: np.ndarray, rewards: np.ndarray,
                            masks: np.ndarray, dones_float: np.ndarray,
                            next_observations: np.ndarray) -> np.ndarray:
    """Splits the samples into trajectories."""
    trajs = [[]]

    for i in tqdm(range(len(observations))):
        trajs[-1].append((observations[i], actions[i], rewards[i], masks[i],
                          dones_float[i], next_observations[i]))
        if dones_float[i] == 1.0 and i + 1 < len(observations):
            trajs.append([])

    return trajs


def merge_trajectories(trajs: np.ndarray):
    """Merges multiple agent trajectories."""
    observations = []
    actions = []
    rewards = []
    masks = []
    dones_float = []
    next_observations = []

    for traj in trajs:
        for (obs, act, rew, mask, done, next_obs) in traj:
            observations.append(obs)
            actions.append(act)
            rewards.append(rew)
            masks.append(mask)
            dones_float.append(done)
            next_observations.append(next_obs)

    return np.stack(observations), np.stack(actions), np.stack(
        rewards), np.stack(masks), np.stack(dones_float), np.stack(
            next_observations)


class Dataset(object):
    """Implements a dataset object for storing samples.
    
    Attributes:
        observations: current states of the agent.
        actions: current actions of the agent.
        rewards: current rewards of the agent.
        dones: current termination flags of the agent.
        masks: current dones of the agent.
        next_observations: next states of the agent.
        size: size of the dataset.
    """
    def __init__(self, observations: np.ndarray, actions: np.ndarray,
                 rewards: np.ndarray, masks: np.ndarray,
                 dones_float: np.ndarray, next_observations: np.ndarray,
                 size: int):
        """Initializes the class object."""
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.masks = masks
        self.dones_float = dones_float
        self.next_observations = next_observations
        self.size = size

    def sample(self, batch_size: int) -> Batch:
        """Samples a batch of data from the dataset."""
        indx = np.random.randint(self.size, size=batch_size)
        return Batch(observations=self.observations[indx],
                     actions=self.actions[indx],
                     rewards=self.rewards[indx],
                     masks=self.masks[indx],
                     next_observations=self.next_observations[indx])

    def get_initial_states(
        self,
        and_action: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Fetches the initial state for the agent."""
        states = []
        if and_action:
            actions = []
        trajs = split_into_trajectories(self.observations, self.actions,
                                        self.rewards, self.masks,
                                        self.dones_float,
                                        self.next_observations)

        def compute_returns(traj):
            """Computes returns by summing up rewards."""
            episode_return = 0
            for _, _, rew, _, _, _ in traj:
                episode_return += rew

            return episode_return

        trajs.sort(key=compute_returns)

        for traj in trajs:
            states.append(traj[0][0])
            if and_action:
                actions.append(traj[0][1])

        states = np.stack(states, 0)
        if and_action:
            actions = np.stack(actions, 0)
            return states, actions
        else:
            return states

    def get_monte_carlo_returns(self, discount) -> np.ndarray:
        """Computes monte carlo returns."""
        trajs = split_into_trajectories(self.observations, self.actions,
                                        self.rewards, self.masks,
                                        self.dones_float,
                                        self.next_observations)
        mc_returns = []
        for traj in trajs:
            mc_return = 0.0
            for i, (_, _, reward, _, _, _) in enumerate(traj):
                mc_return += reward * (discount**i)
            mc_returns.append(mc_return)

        return np.asarray(mc_returns)

    def take_top(self, percentile: float = 100.0):
        """Fteches the top data samples as per returns."""
        assert percentile > 0.0 and percentile <= 100.0

        trajs = split_into_trajectories(self.observations, self.actions,
                                        self.rewards, self.masks,
                                        self.dones_float,
                                        self.next_observations)

        def compute_returns(traj):
            episode_return = 0
            for _, _, rew, _, _, _ in traj:
                episode_return += rew

            return episode_return

        trajs.sort(key=compute_returns)

        N = int(len(trajs) * percentile / 100)
        N = max(1, N)

        trajs = trajs[-N:]

        (self.observations, self.actions, self.rewards, self.masks,
         self.dones_float, self.next_observations) = merge_trajectories(trajs)

        self.size = len(self.observations)

    def take_random(self, percentage: float = 100.0):
        """Fetches random data samples."""
        assert percentage > 0.0 and percentage <= 100.0

        trajs = split_into_trajectories(self.observations, self.actions,
                                        self.rewards, self.masks,
                                        self.dones_float,
                                        self.next_observations)
        np.random.shuffle(trajs)

        N = int(len(trajs) * percentage / 100)
        N = max(1, N)

        trajs = trajs[-N:]

        (self.observations, self.actions, self.rewards, self.masks,
         self.dones_float, self.next_observations) = merge_trajectories(trajs)

        self.size = len(self.observations)

    def train_validation_split(self,
                               train_fraction: float = 0.8
                               ) -> Tuple['Dataset', 'Dataset']:
        """Splits the dataset into train and validation splits."""
        trajs = split_into_trajectories(self.observations, self.actions,
                                        self.rewards, self.masks,
                                        self.dones_float,
                                        self.next_observations)
        train_size = int(train_fraction * len(trajs))

        np.random.shuffle(trajs)

        (train_observations, train_actions, train_rewards, train_masks,
         train_dones_float,
         train_next_observations) = merge_trajectories(trajs[:train_size])

        (valid_observations, valid_actions, valid_rewards, valid_masks,
         valid_dones_float,
         valid_next_observations) = merge_trajectories(trajs[train_size:])

        train_dataset = Dataset(train_observations,
                                train_actions,
                                train_rewards,
                                train_masks,
                                train_dones_float,
                                train_next_observations,
                                size=len(train_observations))
        valid_dataset = Dataset(valid_observations,
                                valid_actions,
                                valid_rewards,
                                valid_masks,
                                valid_dones_float,
                                valid_next_observations,
                                size=len(valid_observations))

        return train_dataset, valid_dataset


class D4RLDataset(Dataset):
    """Implements a dataset for D4RL control environments."""
    def __init__(self,
                 env: gym.Env,
                 clip_to_eps: bool = True,
                 eps: float = 1e-5):
        """Initializes the dataset object."""
        dataset = d4rl.qlearning_dataset(env)

        if clip_to_eps:
            lim = 1 - eps
            dataset['actions'] = np.clip(dataset['actions'], -lim, lim)

        dones_float = np.zeros_like(dataset['rewards'])

        for i in range(len(dones_float) - 1):
            if np.linalg.norm(dataset['observations'][i + 1] -
                              dataset['next_observations'][i]
                              ) > 1e-6 or dataset['terminals'][i] == 1.0:
                dones_float[i] = 1
            else:
                dones_float[i] = 0

        dones_float[-1] = 1

        super().__init__(dataset['observations'].astype(np.float32),
                         actions=dataset['actions'].astype(np.float32),
                         rewards=dataset['rewards'].astype(np.float32),
                         masks=1.0 - dataset['terminals'].astype(np.float32),
                         dones_float=dones_float.astype(np.float32),
                         next_observations=dataset['next_observations'].astype(
                             np.float32),
                         size=len(dataset['observations']))
