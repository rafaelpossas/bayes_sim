import os
import sys

py_path = os.path.split(os.getcwd())[0]
if py_path not in sys.path:
    sys.path.append(py_path)

import click
import numpy as np
import pickle

from src.utils import logger
from src.utils.util import set_global_seeds
from src.data.episode_rollout import EpisodeRollout
from src import collect_data_config as sim_param
import src.config as config
from carbongym import gymapi

def save(data, file):
    """Saves data to a file."""

    f = open(file, 'wb')
    pickle.dump(data, f)
    f.close()


def load(file):
    """Loads data from file."""

    f = open(file, 'rb')
    data = pickle.load(f)
    f.close()
    return data
@click.command()
@click.argument('policy_file', type=str, default="src\models\controllers\DDPG\policy_best.pkl")
@click.option('--seed', type=int, default=10)
@click.option('--n_test_rollouts', type=int, default=5000)
@click.option('--render', type=int, default=1)
@click.option('--randomize_every_step', type=int, default=0)

def main(policy_file, seed, n_test_rollouts, render, randomize_every_step):
    set_global_seeds(seed)

    # Load policy.
    with open(policy_file, 'rb') as f:
        policy = pickle.load(f)
    env_name = policy.info['env_name']

    # Prepare params.
    params = config.DEFAULT_PARAMS
    if env_name in config.DEFAULT_ENV_PARAMS:
        params.update(config.DEFAULT_ENV_PARAMS[env_name])  # merge env-specific parameters in
    params['env_name'] = env_name
    params = config.prepare_params(params)
    config.log_params(params, logger=logger)

    dims = config.configure_dims(params)

    eval_params = {
        'exploit': True,
        'use_target_net': params['test_with_polyak'],
        'compute_Q': True,
        'render': bool(render),
    }

    for name in ['max_episode_steps', 'gamma', 'noise_eps', 'random_eps', 'rollout_batch_size']:
        eval_params[name] = params[name]

    evaluator = EpisodeRollout(params['make_env'], policy, dims, logger, **eval_params)
    evaluator.seed(seed)
    gym = evaluator.envs._gym
    sim = evaluator.envs._sim
    if render:
        viewer = gym.create_viewer(sim, gymapi.DEFAULT_VIEWER_WIDTH, gymapi.DEFAULT_VIEWER_HEIGHT)
        evaluator.viewer = viewer
        evaluator.viewer = viewer
    # Run evaluation.
    evaluator.clear_history()
    all_episodes = []
    for _ in range(n_test_rollouts):
        print("Rollout: {} of {}".format(_, n_test_rollouts))
        if randomize_every_step == 0 or _ % randomize_every_step == 0:
            params = evaluator.set_physics(sim_param.BODIES, sim_param.RANDOM_PARAMS)
        # if friction_idx == len(friction_arr):
        #     friction_idx = 0
        # if _ % 5 == 0:
        #     evaluator.seed(seed)
        #     friction = friction_arr[friction_idx]
        #     friction_idx += 1
        #     evaluator.set_physics(param=friction)
        episode = evaluator.generate_rollouts()
        all_episodes.append((params, episode))


    # record logs
    for key, val in evaluator.logs('test'):
        logger.record_tabular(key, np.mean(val))
    logger.dump_tabular()
    save(all_episodes, 'data_friction_mass_5k.pkl')


if __name__ == '__main__':
    #test_data = load('data_friction_object_table.pkl')
    main()
    #print(len(test_data))