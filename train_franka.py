import os
import sys

py_path = os.path.split(os.getcwd())[0]
if py_path not in sys.path:
    sys.path.append(py_path)
    
import src.config as config
import click
import numpy as np
import json
import datetime
from carbongym import gymapi

from src.utils import logger
from src.utils.util import set_global_seeds
from src.utils.util import mpi_average
from src.utils import helper

from src.data.episode_rollout import EpisodeRollout

from src.data.her import HindisghtExperienceReplay

from src.models.ddpg import DDPG

from src.config import simple_goal_subtract
import torch

def train(policy, rollout_worker, evaluator,
          n_epochs, n_test_rollouts, n_cycles, n_batches, policy_save_interval,
          save_policies, mdn_prior=None, **kwargs):

    if mdn_prior is not None:
        mdn = helper.load(mdn_prior)
        x_obs = torch.from_numpy(np.float32(np.array([0.57546])))
        friction_arr = []
        posterior = mdn.get_mog(x_obs)
        for ix in range(1000):
            friction = posterior.gen()
            friction_arr.append(friction)
        print("Mean Friction from prior: {}"+np.mean(friction))


    latest_policy_path = os.path.join(logger.get_dir(), 'policy_latest.pkl')
    best_policy_path = os.path.join(logger.get_dir(), 'policy_best.pkl')
    periodic_policy_path = os.path.join(logger.get_dir(), 'policy_{}.pkl')

    logger.info("Training...")
    best_success_rate = -1

    if rollout_worker.render or evaluator.render:
        gym = rollout_worker.envs._gym
        sim = rollout_worker.envs._sim
        viewer = gym.create_viewer(sim, gymapi.DEFAULT_VIEWER_WIDTH, gymapi.DEFAULT_VIEWER_HEIGHT)
        rollout_worker.viewer = viewer
        evaluator.viewer = viewer

    for epoch in range(n_epochs):
        # train
        rollout_worker.clear_history()
        # rollout_worker.set_physics(epoch, prior=posterior.gen)
        rollout_worker.set_physics(bodies={}, random_params={})
        for _ in range(n_cycles):
            episode = rollout_worker.generate_rollouts()
            policy.store_episode(episode)
            for _ in range(n_batches):
                policy.train()
            policy.update_target_net()

        # test
        evaluator.clear_history()
        evaluator.set_physics(bodies={}, random_params={})
        for _ in range(n_test_rollouts):
            evaluator.generate_rollouts()

        # record logs
        logger.record_tabular('epoch', epoch)
        for key, val in evaluator.logs('test'):
            logger.record_tabular(key, mpi_average(val))
        for key, val in rollout_worker.logs('train'):
            logger.record_tabular(key, mpi_average(val))
        for key, val in policy.logs():
            logger.record_tabular(key, mpi_average(val))

        logger.dump_tabular()
        print("Buffer Current Size: " + str(policy.buffer.current_size))

        # save the policy if it's better than the previous ones
        success_rate = evaluator.current_success_rate()

        if success_rate >= best_success_rate and save_policies:
            best_success_rate = success_rate
            logger.info(
                'New best success rate: {}. Saving policy to {} ...'.format(best_success_rate, best_policy_path))
            evaluator.save_policy(best_policy_path)
            evaluator.save_policy(latest_policy_path)

        if policy_save_interval > 0 and epoch % policy_save_interval == 0 and save_policies:
            policy_path = periodic_policy_path.format(epoch)
            logger.info('Saving periodic policy to {} ...'.format(policy_path))
            evaluator.save_policy(policy_path)

        # make sure that different threads have different seeds
        local_uniform = np.random.uniform(size=(1,))
        root_uniform = local_uniform.copy()


def set_params(env, replay_strategy, override_params):
    # Prepare params.
    params = config.DEFAULT_PARAMS
    params['env_name'] = env
    params['replay_strategy'] = replay_strategy
    if env in config.DEFAULT_ENV_PARAMS:
        params.update(config.DEFAULT_ENV_PARAMS[env])  # merge env-specific parameters in

    with open(os.path.join(logger.get_dir(), 'params.json'), 'w') as f:
        json.dump(params, f)

    params.update(**override_params)  # makes it possible to override any parameter

    params = config.prepare_params(params)

    return params


def launch(env, logdir, n_epochs, seed,
           replay_strategy, policy_save_interval, clip_return, random_physics,
           lower_bound, upper_bound, randomise_every_n_epoch,
           override_params={}, save_policies=True, mdn_prior=None):

    now = datetime.datetime.now()
    logdir += "/" + env + "/" + str(now.strftime("%Y-%m-%d-%H:%M"))

    # Configure logging
    if logdir or logger.get_dir() is None:
        logger.configure(dir=logdir)


    logdir = logger.get_dir()

    assert logdir is not None
    os.makedirs(logdir, exist_ok=True)

    params = set_params(env, replay_strategy, override_params)

    config.log_params(params, logger=logger)

    rollout_params = {
        'exploit': False,
        'use_target_net': False,
        'use_demo_states': True,
        'compute_Q': False,
        'render': params['render'],
        'max_episode_steps': params['max_episode_steps'],
        'random_physics': random_physics,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'randomise_every_n_epoch': randomise_every_n_epoch
    }

    eval_params = {
        'exploit': True,
        'use_target_net': params['test_with_polyak'],
        'use_demo_states': False,
        'compute_Q': True,
        'render': params['render'],
        'max_episode_steps': params['max_episode_steps'],
        'random_physics': False,
        'rnd_phys_lower_bound': lower_bound,
        'rnd_phys_upper_bound': upper_bound,
        'randomise_every_n_epoch': randomise_every_n_epoch
    }

    dims = config.configure_dims(params)

    her = HindisghtExperienceReplay(params['make_env'], replay_strategy, params['replay_k'])

    sample_her_transitions = her.make_sample_her_transitions()

    # Seed everything.
    rank_seed = seed + 1000000
    set_global_seeds(rank_seed)

    # DDPG agent
    ddpg_params = params['ddpg_params']

    ddpg_params.update({'input_dims': dims.copy(),  # agent takes an input observations
                        'max_episode_steps': params['max_episode_steps'],
                        'clip_pos_returns': True,  # clip positive returns
                        'clip_return': (1. / (1. - params['gamma'])) if clip_return else np.inf,  # max abs of return
                        'rollout_batch_size': params['rollout_batch_size'],
                        'subtract_goals': simple_goal_subtract,
                        'sample_transitions': sample_her_transitions,
                        'gamma': params['gamma'],
                        })


    ddpg_params['info'] = {
        'env_name': params['env_name'],
    }

    policy = DDPG(reuse=False, **ddpg_params, use_mpi=True)

    for name in ['max_episode_steps', 'rollout_batch_size', 'gamma', 'noise_eps', 'random_eps']:
        rollout_params[name] = params[name]
        eval_params[name] = params[name]

    rollout_worker = EpisodeRollout(params['make_env'], policy, dims, logger, **rollout_params)
    rollout_worker.seed(rank_seed)

    evaluator = EpisodeRollout(params['make_env'], policy, dims, logger, **eval_params)
    evaluator.seed(rank_seed)

    train(logdir=logdir, policy=policy, rollout_worker=rollout_worker,
          evaluator=evaluator, n_epochs=n_epochs, n_test_rollouts=params['n_test_rollouts'],
          n_cycles=params['n_cycles'], n_batches=params['n_batches'],
          policy_save_interval=policy_save_interval, save_policies=save_policies,
          mdn_prior=mdn_prior)


@click.command()
@click.option('--random_physics', type=bool, default=False)
@click.option('--lower_bound', type=float, default=0.1)
@click.option('--upper_bound', type=float, default=1.0)
@click.option('--randomise_every_n_epoch', type=int, default=10)

@click.option('--env', type=str, default='FrankaPush-v0',
              help='the name of the OpenAI Gym environment that you want to train on')

@click.option('--logdir', type=str, default="logs",
              help='the path to where logs and policy pickles should go. If not specified, creates a folder in /tmp/')

@click.option('--n_epochs', type=int, default=200, help='the number of training epochs to run')
@click.option('--seed', type=int, default=0,
              help='the random seed used to seed both the environment and the training code')

@click.option('--policy_save_interval', type=int, default=10,
              help='the interval with which policy pickles are saved. If set to 0, only the best and latest policy will be pickled.')

@click.option('--replay_strategy', type=click.Choice(['future', 'none']), default='future',
              help='the HER replay strategy to be used. "future" uses HER, "none" disables HER.')

@click.option('--clip_return', type=int, default=1, help='whether or not returns should be clipped')
@click.option('--mdn_prior', type=str, default=None, help='Friction Prior')

def main(**kwargs):
    launch(**kwargs)


if __name__ == '__main__':
    main()
