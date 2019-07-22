import numpy as np
import gym

from src.utils import logger
from src.sim import rlbase
from src.sim.franka import FrankaEnv
from src.sim.franka_push import FrankaPush
from carbongym import gymapi
DEFAULT_ENV_PARAMS = {
    'FetchReach-v1': {
        'n_cycles': 10,
    },
}


DEFAULT_PARAMS = {
    # env
    'max_u': 1.,  # max absolute value of actions on different coordinates
    # ddpg
    'layers': 3,  # number of layers in the critic/actor networks
    'hidden': 256,  # number of neurons in each hidden layers
    'network_class': 'src.models.actor_critic:ActorCritic',
    'Q_lr': 0.001,  # critic learning rate
    'pi_lr': 0.001,  # actor learning rate
    'buffer_size': int(1E6),  # for experience replay
    'polyak': 0.95,  # polyak averaging coefficient
    'action_l2': 1.0,  # quadratic penalty on actions (before rescaling by max_u)
    'clip_obs': 200.,
    'scope': 'ddpg',  # can be tweaked for testing
    'relative_goals': False,
    # training,
    'steps': 50,
    'render': False,
    'n_cycles': 40,  # per epoch,
    'rollout_batch_size': 1,  # per mpi thread
    'n_batches': 40,  # training batches per cycle
    'batch_size': 256,  # per mpi thread, measured in transitions and reduced to even multiple of chunk_length.
    'n_test_rollouts': 10,  # number of test rollouts per epoch, each consists of rollout_batch_size rollouts
    'test_with_polyak': False,  # run test episodes with the target network
    # exploration
    'random_eps': 0.30,  # percentage of time a random action is taken
    'noise_eps': 0.2,  # std of gaussian noise added to not-completely-random actions as a percentage of max_u
    # HER
    'replay_strategy': 'future',  # supported modes: future, none
    'replay_k': 4,  # number of additional goals used for replay, only used if off_policy_data=future
    # normalization
    'norm_eps': 0.01,  # epsilon used for observation normalization
    'norm_clip': 5,  # normalized observations are cropped to this values
}

CACHED_ENVS = {}

def cached_make_env(make_env, env_name):
    """
    Only creates a new environment from the provided function if one has not yet already been
    created. This is useful here because we need to infer certain properties of the env, e.g.
    its observation and action spaces, without any intend of actually using it.
    """
    if env_name not in CACHED_ENVS:
        env = make_env()
        CACHED_ENVS[env_name] = env
    return CACHED_ENVS[env_name]




def prepare_params(kwargs):
    # DDPG params
    ddpg_params = dict()

    env_name = kwargs['env_name']

    if env_name == "FrankaPush-v0":
        carbgym = gymapi.acquire_gym()
        if not carbgym.initialize():
            print("*** Failed to initialize gym")
            quit()

        sim = carbgym.create_sim(0, 0)

        # simulation params
        sim_params = gymapi.SimParams()

        sim_params.solver_type = 5
        sim_params.num_outer_iterations = 5
        sim_params.num_inner_iterations = 60
        sim_params.relaxation = 0.75
        sim_params.warm_start = 0.5

        sim_params.shape_collision_margin = 0.003
        sim_params.contact_regularization = 1e-7

        sim_params.deterministic_mode = True

        sim_params.gravity = gymapi.Vec3(0.0, -9.8, 0.0)

        # create experiment
        num_envs = DEFAULT_PARAMS['rollout_batch_size']
        env_spacing = 2.5

    def make_env():
        if env_name == "FrankaPush-v0":
            env = rlbase.Experiment(carbgym, sim, FrankaEnv, FrankaPush, num_envs, env_spacing, kwargs['steps'], sim_params)
            return env
        else:
            return gym.make(env_name)

    kwargs['make_env'] = lambda: cached_make_env(make_env, env_name)
    tmp_env = kwargs['make_env']()
    assert hasattr(tmp_env, '_max_episode_steps')
    kwargs['max_episode_steps'] = tmp_env._max_episode_steps
    tmp_env.reset()
    kwargs['max_u'] = np.array(kwargs['max_u']) if isinstance(kwargs['max_u'], list) else kwargs['max_u']
    kwargs['gamma'] = 1. - 1. / kwargs['max_episode_steps']
    if 'lr' in kwargs:
        kwargs['pi_lr'] = kwargs['lr']
        kwargs['Q_lr'] = kwargs['lr']
        del kwargs['lr']
    for name in ['buffer_size', 'hidden', 'layers',
                 'network_class',
                 'polyak',
                 'batch_size', 'Q_lr', 'pi_lr',
                 'norm_eps', 'norm_clip', 'max_u',
                 'action_l2', 'clip_obs', 'scope', 'relative_goals']:
        ddpg_params[name] = kwargs[name]
        kwargs['_' + name] = kwargs[name]
        del kwargs[name]
    kwargs['ddpg_params'] = ddpg_params

    return kwargs


def log_params(params, logger=logger):
    for key in sorted(params.keys()):
        logger.info('{}: {}'.format(key, params[key]))


def simple_goal_subtract(a, b):
    assert a.shape == b.shape
    return a - b


def configure_dims(params):
    env = params['make_env']()
    env.reset()
    obs, _, _, info = env.step(env.action_space.sample())

    # If this is multi environment

    obs = obs[0]
    info = info[0]

    dims = {
        'observation': obs['observation'].shape[0],
        'action': env.action_space.shape[0],
        'goal': obs['desired_goal'].shape[0],
    }
    for key, value in info.items():
        value = np.array(value)
        if value.ndim == 0:
            value = value.reshape(1)
        dims['info_{}'.format(key)] = value.shape[0]
    return dims
