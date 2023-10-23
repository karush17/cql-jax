"""Implements the main training protocol."""

from typing import Tuple

import os
import numpy as np
import tqdm
import gym
import wrappers

from absl import app, flags
from ml_collections import config_flags
from tensorboardX import SummaryWriter
from utils import D4RLDataset, normalize
from cql_learner import CQLLearner
from evaluation import evaluate

FLAGS = flags.FLAGS
flags.DEFINE_string('env_name', 'hopper-expert-v0', 'Environment name.')
flags.DEFINE_float('min_q_weight', 5, 'Min Q Weight.')
flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 5000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 512, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_float(
    'percentile', 100.0,
    'Dataset percentile (see https://arxiv.org/abs/2106.01345).')
flags.DEFINE_float('percentage', 100.0,
                   'Pencentage of the dataset to use for training.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
config_flags.DEFINE_config_file(
    'config',
    'cql_config.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


def make_env_and_dataset(env_name: str, seed: int) -> Tuple[gym.Env,
                                                            D4RLDataset]:
    """Creates the environment and offline dataset."""
    env = gym.make(env_name)

    env = wrappers.EpisodeMonitor(env)
    env = wrappers.SinglePrecision(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    dataset = D4RLDataset(env)

    if 'antmaze' in FLAGS.env_name:
        dataset.rewards -= 1.0
    elif ('halfcheetah' in FLAGS.env_name or 'walker2d' in FLAGS.env_name
          or 'hopper' in FLAGS.env_name):
        normalize(dataset)

    return env, dataset


def main(_):
    """Implements the main training scheme."""
    summary_writer = SummaryWriter(
        os.path.join(FLAGS.save_dir, 'tb', str(FLAGS.seed)))

    env, dataset = make_env_and_dataset(FLAGS.env_name, FLAGS.seed)

    if FLAGS.percentage < 100.0:
        dataset.take_random(FLAGS.percentage)

    if FLAGS.percentile < 100.0:
        dataset.take_top(FLAGS.percentile)

    kwargs = dict(FLAGS.config)
    kwargs['min_q_weight'] = FLAGS.min_q_weight
    agent = CQLLearner(FLAGS.seed,
                    env.observation_space.sample()[np.newaxis],
                    env.action_space.sample()[np.newaxis], **kwargs)

    eval_returns = []
    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        batch = dataset.sample(FLAGS.batch_size)

        update_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            for k, v in update_info.items():
                summary_writer.add_scalar(f'training/{k}', v, i)
            summary_writer.flush()

        if i % FLAGS.eval_interval == 0:
            eval_stats = evaluate(agent, env, FLAGS.eval_episodes)

            for k, v in eval_stats.items():
                summary_writer.add_scalar(f'evaluation/average_{k}s', v, i)
            summary_writer.flush()

            eval_returns.append((i, eval_stats['return']))
            np.savetxt(os.path.join(FLAGS.save_dir, f'{FLAGS.seed}.txt'),
                       eval_returns,
                       fmt=['%d', '%.1f'])


if __name__ == '__main__':
    app.run(main)
