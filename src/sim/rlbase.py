from __future__ import print_function, division, absolute_import

import abc
from math import sqrt
import numpy as np

from carbongym import gymapi
import types
class Environment(abc.ABC):
    """ Base class for environment instances """

    def __init__(self, **kwargs):
        self._gym = kwargs["gym"]
        self._env = kwargs["env"]
        self._env_index = kwargs["env_index"]


    @abc.abstractmethod
    def step(self, actions):
        """ Callback for stepping the environment """

    @staticmethod
    @abc.abstractmethod
    def create_shared_data(gym):
        """ Create or load assets to be shared by all environment instances """

    # convenience wrappers for native env instance API

    def create_actor(self, asset, pose, name, group=-1, filter=-1):
        return self._gym.create_actor(self._env, asset, pose, name, group, filter)

    def get_rigid_handle(self, instance_name, body_name):
        return self._gym.get_rigid_handle(self._env, instance_name, body_name)

    def get_joint_handle(self, instance_name, joint_name):
        return self._gym.get_joint_handle(self._env, instance_name, joint_name)

    def get_sensor_handle(self, instance_name, sensor_name):
        return self._gym.get_sensor_handle(self._env, instance_name, sensor_name)

    def get_rigid_transform(self, handle):
        return self._gym.get_rigid_transform(self._env, handle)
    
    def set_rigid_transform(self, handle, transform):
        self._gym.set_rigid_transform(self._env, handle, transform)

    def get_rigid_linear_velocity(self, handle):
        return self._gym.get_rigid_linear_velocity(self._env, handle)

    def set_rigid_linear_velocity(self, handle, linear_velocity):
        self._gym.set_rigid_linear_velocity(self._env, handle, linear_velocity)

    def get_rigid_angular_velocity(self, handle):
        return self._gym.get_rigid_angular_velocity(self._env, handle)

    def set_rigid_angular_velocity(self, handle, angular_velocity):
        self._gym.set_rigid_angular_velocity(self._env, handle, angular_velocity)

    def get_joint_position(self, handle):
        return self._gym.get_joint_position(self._env, handle)
    
    def get_joint_velocity(self, handle):
        return self._gym.get_joint_velocity(self._env, handle)
    
    def apply_body_force(self, handle, force, pos):
        self._gym.apply_body_force(self._env, handle, force, pos)
    
    def apply_joint_effort(self, handle, effort):
        self._gym.apply_joint_effort(self._env, handle, effort)

    def set_joint_target_position(self, handle, pos):
        self._gym.set_joint_target_position(self._env, handle, pos)

    def set_joint_target_velocity(self, handle, vel):
        self._gym.set_joint_target_velocity(self._env, handle, vel)


class Task(abc.ABC):
    """ A base class for RL tasks to be performed in an Environment """

    def __init__(self, **kwargs):
        pass

    @staticmethod
    @abc.abstractmethod
    def num_observations():
        """ Number of observations that this task generates. """

    @abc.abstractmethod
    def post_step(self):
        """ Called after stepping the environment.  Get observations, reward, and termination. """

    @abc.abstractmethod
    def reset(self):
        """ Callback to re-initialize the episode """


class Experiment:
    def __init__(self, gym, sim, EnvClass, TaskClass, num_envs, spacing, max_episode_steps, sim_params):
        self._num_envs = num_envs
        self._num_actions = TaskClass.num_actions()
        self._num_obs = TaskClass.num_observations()

        self._gym = gym
        self._sim = sim
        self._gym.set_sim_params(self._sim, sim_params)
        self._max_episode_steps = max_episode_steps
        # acquire data to be shared by all environment instances
        shared_data = EnvClass.create_shared_data(self._gym, self._sim)
        #self.viewer = gym.create_viewer(sim, gymapi.DEFAULT_VIEWER_WIDTH, gymapi.DEFAULT_VIEWER_HEIGHT)

        # env bounds
        lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # create environment and task instances
        self.envs = []
        self.tasks = []

        num_per_row = int(sqrt(num_envs))
        for i in range(num_envs):
            # create env instance
            env_ptr = self._gym.create_env(self._sim, lower, upper, num_per_row)
            env_base_args = {
                "gym": self._gym,
                "env": env_ptr,
                "env_index": i,
            }
            env = EnvClass(shared_data, **env_base_args)
            self.envs.append(env)

            # create task instance
            task_base_args = {
                "gym": self._gym,
                "sim": self._sim
            }
            task = TaskClass(env, **task_base_args)
            self.tasks.append(task)

        # exposing the compute reward to the training algorithm
        self.compute_reward = task.compute_reward

        # allocate buffers shared by all envs
        self.observations = np.zeros((self._num_envs, self._num_obs), dtype=np.float32)
        self.actions = np.zeros((self._num_envs, self.num_actions()), dtype=np.float32)
        self.rewards = np.zeros((self._num_envs,), dtype=np.float32)
        self.kills = np.zeros((self._num_envs,), dtype=np.bool_)
        self.info = np.zeros((self._num_envs,), dtype=np.bool_)

        # Workaround to work with original GYM interface
        self.action_space = types.SimpleNamespace()
        self.action_space.sample = self.sample_action
        self.action_space.shape = [self.num_actions()]

    def sample_action(self):
        actions = (np.random.rand(self._num_envs, self.num_actions()))
        return actions

    def num_actions(self):
        return self._num_actions

    def num_observations(self):
        return self._num_obs

    def reset(self):
        all_obs = []
        for task in self.tasks:
            all_obs.append(task.reset())

        return np.array(all_obs)

    def step(self, actions, dt=0.5):
        # apply the actions
        for env in self.envs:
            env.step(actions, dt=dt)

        # simulate
        self._gym.simulate(self._sim, self.envs[0].dt, 2)
        self._gym.fetch_results(self._sim, True)
        # update observations and stuff

        # delegating data type to the task itself
        all_obs, all_rewards, all_info = [], [], []
        import copy
        for task in self.tasks:
            obs, rewards, info = task.post_step()
            all_obs.append(copy.deepcopy(obs))
            all_rewards.append(copy.deepcopy(rewards))
            all_info.append(copy.deepcopy(info))

        return all_obs, all_rewards, None, all_info

    def release(self):
        # TODO
        pass
