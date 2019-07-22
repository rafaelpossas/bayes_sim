"""Tests for isaac_gym.suite domains."""

# Authors: Rafael Possas, Ankur Handa

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import copy
from tqdm import tqdm
root_dir = os.path.dirname(os.getcwd())
sys.path.append(root_dir)

# Internal dependencies.
from absl.testing import absltest
from absl import flags
from carbongym import gymapi
from src.sim.franka import FrankaEnv, rlbase
from src.sim.franka_push import FrankaPush

FLAGS = flags.FLAGS
FLAGS(sys.argv)

flags.DEFINE_integer('num_envs', 2, 'Number of environments to be loaded', lower_bound=1)
flags.DEFINE_integer('max_episode_steps', 150, 'Number of steps per episode', lower_bound=1)
flags.DEFINE_integer('env_spacing', 2, 'Space between environments in simulation', lower_bound=1)
flags.DEFINE_boolean('enable_viewer', False, 'Show sim viewer')


class FrankaTest(absltest.TestCase):

    def setUp(self):
        try:
            # initialize gym
            # initialize gym
            self.gym = gymapi.acquire_gym()

            if not self.gym.initialize():
                self.assertTrue(False, "Failed to initialize GYM")

            self.sim = self.gym.create_sim(0, 0)

            # simulation params
            sim_params = gymapi.SimParams()

            sim_params.solver_type = 5
            sim_params.num_outer_iterations = 4
            sim_params.num_inner_iterations = 10

            sim_params.relaxation = 0.75
            sim_params.warm_start = 0.4

            sim_params.gravity = gymapi.Vec3(0.0, -9.8, 0.0)

            self.gym.set_sim_params(self.sim, sim_params)

            # create experiment
            self.num_envs = FLAGS.num_envs
            env_spacing = FLAGS.env_spacing
            self.max_episode_steps = FLAGS.max_episode_steps

            self.push_exp = rlbase.Experiment(self.gym, self.sim, FrankaEnv, FrankaPush, self.num_envs, env_spacing,
                                              self.max_episode_steps,
                                              sim_params)

            if FLAGS.enable_viewer:
                # create viewer
                self.viewer = self.gym.create_viewer(self.sim, gymapi.DEFAULT_VIEWER_WIDTH, gymapi.DEFAULT_VIEWER_HEIGHT)

                if self.viewer is None:
                    self.assertTrue(False, "Failed to create viewer")

                # set viewer viewpoint0
                self.gym.viewer_camera_look_at(self.viewer, self.push_exp.envs[0]._env,
                                          gymapi.Vec3(0.5, 1.3, -2.0), gymapi.Vec3(.8, 0, 2))

        except Exception as e:
            self.assertTrue(False, "Error Initializing Simulation: {}".format(e))

    """
    Tests if the arm moves randomly between episodes after different random actions being applied
    """
    def test_randomised_actions_all_env(self):
        print("Running randomised actions test")
        step_counts = 0
        last_obs = []

        last_dist = None
        mean = None
        for _ in tqdm(range(self.max_episode_steps*100)):

            if step_counts == self.max_episode_steps:
                cur_obs = []
                cur_dist = []
                for i in range(self.num_envs):
                    cur_obs.append(self.push_exp.envs[i].get_franka_joint_states()['pos'])

                    if last_obs is not None and len(last_obs) == self.num_envs:
                        cur_dist.append(np.linalg.norm(np.array(cur_obs[i]) - np.array(last_obs[i])))
                        last_obs[i] = copy.deepcopy(cur_obs[i])
                    else:
                        last_obs.append(copy.deepcopy(cur_obs[i]))

                step_counts = 0

                if len(cur_dist) > 0:
                    # Calculating the running mean of the L2 Norm for all environments
                    mean = (0.99 * np.array(mean)) + (0.01 * np.array(cur_dist)) if last_dist is not None else cur_dist
                    last_dist = copy.deepcopy(cur_dist)

                self.push_exp.reset()

            step_counts += 1
            action = np.random.uniform(-1, 1, size=(self.num_envs, 7))
            # take the action
            obs, rews, _, info = self.push_exp.step(action)

            # draw some debug visualization
            if FLAGS.enable_viewer:
                self.gym.clear_lines(self.viewer)
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)

        self.gym.shutdown()

        #Testing if the average distance between environemnts is above the threshold
        self.assertTrue(np.all(mean > 0.5), "Distances between episodes are below the specified threshold")


if __name__ == '__main__':
  absltest.main()
