from collections import deque

import numpy as np
import pickle
from carbongym import gymapi

from src.utils.util import convert_episode_to_batch_major, store_args


class EpisodeRollout(object):

    @store_args
    def __init__(self, make_env, policy, dims, logger, max_episode_steps, rollout_batch_size=1,
                 exploit=False, use_target_net=False, compute_Q=False, noise_eps=0,
                 random_eps=0, history_len=100, render=True, random_physics=False,
                 lower_bound=0.1, upper_bound=1.0, randomise_every_n_epoch=25,
                 **kwargs):
        """Rollout worker generates experience by interacting with one or many environments.

        Args:
            make_env (function): a factory function that creates a new instance of the environment
                when called
            policy (object): the policy that is used to act
            dims (dict of ints): the dimensions for observations (o), goals (g), and actions (u)
            logger (object): the logger that is used by the rollout worker
            num_envs (int): the number of parallel rollouts that should be used
            exploit (boolean): whether or not to exploit, i.e. to act optimally according to the
                current policy without any exploration
            use_target_net (boolean): whether or not to use the target net for rollouts
            compute_Q (boolean): whether or not to compute the Q values alongside the actions
            noise_eps (float): scale of the additive Gaussian noise
            random_eps (float): probability of selecting a completely random action
            history_len (int): length of history for statistics smoothing
            render (boolean): whether or not to render the rollouts
        """
        self.envs = make_env()
        assert self.max_episode_steps > 0

        self.info_keys = [key.replace('info_', '') for key in dims.keys() if key.startswith('info_')]

        self.success_history = deque(maxlen=history_len)
        self.Q_history = deque(maxlen=history_len)
        self.reward_history = deque(maxlen=history_len)
        self.friction_coefficients = []
        self.n_episodes = 0
        self.g = np.empty((self.rollout_batch_size, self.dims['goal']), np.float32)  # goals
        self.initial_o = np.empty((self.rollout_batch_size, self.dims['observation']), np.float32)  # observations
        self.initial_ag = np.empty((self.rollout_batch_size, self.dims['goal']), np.float32)  # achieved goals
        self.reset_all_rollouts()
        self.clear_history()

    def set_physics(self, bodies, random_params):
        # TODO: Implement Physics on Isaac Gym
        params = {}
        for body in bodies:
            for prop in random_params[body]:
                if random_params[body][prop]['active']:
                    if body not in params:
                        params[body] = {}
                    if prop not in params[body]:
                        params[body][prop] = {}

                    value = np.random.uniform(random_params[body][prop]['min_value'],
                                              random_params[body][prop]['max_value'])

                    params[body][prop] = value

                    print("Setting new physics - {} - {}: {}".format(body, prop, value))

                    for task in self.envs.tasks:
                        task.set_physics(body, prop, value)
        return params


    def reset_all_rollouts(self):
        """Resets all `num_env` rollout workers.
        """
        obs = self.envs.reset()
        for i in range(self.rollout_batch_size):
            self.initial_o[i] = obs[i]['observation']
            self.initial_ag[i] = obs[i]['achieved_goal']
            self.g[i] = obs[i]['desired_goal']

    def generate_rollouts(self):
        """Performs `num_envs` rollouts in parallel for time horizon `T` with the current
        policy acting on it accordingly.
        """
        self.reset_all_rollouts()

        # compute observations
        o = np.empty((self.rollout_batch_size, self.dims['observation']), np.float32)  # observations
        ag = np.empty((self.rollout_batch_size, self.dims['goal']), np.float32)  # achieved goals
        o[:] = self.initial_o
        ag[:] = self.initial_ag
        # generate episodes
        obs, achieved_goals, acts, goals, successes, rewards = [], [], [], [], [], []
        info_values = [np.empty((self.max_episode_steps, self.rollout_batch_size, self.dims['info_' + key]), np.float32)
                       for key in self.info_keys]
        Qs = []

        def gaussian_eps_greedy_noise(action, noise_eps, random_eps, max_u):

            def random_action(n):
                return np.random.uniform(low=-max_u, high=max_u, size=(n, action.shape[1]))

            noise = noise_eps * max_u * np.random.randn(*action.shape)  # gaussian noise
            action += noise
            action = np.clip(action, -max_u, max_u)
            action += np.random.binomial(1, random_eps, action.shape[0]).reshape(-1, 1) \
                      * (random_action(action.shape[0]) - action)  # eps-greedy
            return action

        for t in range(self.max_episode_steps):

            policy_output = self.policy.get_actions(o, ag, self.g,
                                                    compute_Q=self.compute_Q,
                                                    noise_eps=self.noise_eps if not self.exploit else 0.,
                                                    random_eps=self.random_eps if not self.exploit else 0.,
                                                    use_target_net=self.use_target_net,
                                                    noise_fn=gaussian_eps_greedy_noise)

            if self.compute_Q:
                u, Q = policy_output
                Qs.append(Q)
            else:
                u = policy_output

            if u.ndim == 1:
                # The non-batched case should still have a reasonable shape.
                u = u.reshape(1, -1)

            o_new = np.empty((self.rollout_batch_size, self.dims['observation']))
            ag_new = np.empty((self.rollout_batch_size, self.dims['goal']))
            success = np.zeros(self.rollout_batch_size)
            r = np.zeros(self.rollout_batch_size)
            # compute new states and observations

            # Step all envionments at once
            curr_o_new_arr, reward_arr, kills_arr, info_arr = self.envs.step(u)

            for i in range(self.rollout_batch_size):
                try:
                    # We fully ignore the reward here because it will have to be re-computed
                    # for HER.
                    curr_o_new, reward, _, info = curr_o_new_arr[i], reward_arr[i], None, info_arr[i]

                    if 'is_success' in info:
                        success[i] = info['is_success']
                    r[i] = reward
                    o_new[i] = curr_o_new['observation']
                    ag_new[i] = curr_o_new['achieved_goal']

                    for idx, key in enumerate(self.info_keys):
                        info_values[idx][t, i] = info[key]

                    if self.render:
                        self.envs._gym.step_graphics(self.envs._sim)
                        self.envs._gym.draw_viewer(self.viewer, self.envs._sim, False)

                except Exception as e:
                    return self.generate_rollouts()

            if np.isnan(o_new).any():
                self.logger.warn('NaN caught during rollout generation. Trying again...')
                self.reset_all_rollouts()
                return self.generate_rollouts()

            obs.append(o.copy())
            achieved_goals.append(ag.copy())
            successes.append(success.copy())
            acts.append(u.copy())
            goals.append(self.g.copy())
            rewards.append(r.copy())
            o[...] = o_new
            ag[...] = ag_new

        obs.append(o.copy())
        achieved_goals.append(ag.copy())
        self.initial_o[:] = o

        episode = dict(observation=obs,
                       action=acts,
                       goal=goals,
                       achieved_goal=achieved_goals)

        for key, value in zip(self.info_keys, info_values):
            episode['info_{}'.format(key)] = value

        # stats
        successful = np.array(successes)[-1, :]
        total_rewards = 0
        for i in range(self.rollout_batch_size):
            for j in range(len(rewards)):
                total_rewards += rewards[j][i]

        mean_reward = total_rewards / self.rollout_batch_size

        # mean_reward = sum([rewards[i][0]+rewards[i][1] for i, j in zip(range(len(rewards), range(self.rollout_batch_size)))])\
        #               /self.rollout_batch_size
        assert successful.shape == (self.rollout_batch_size,)
        success_rate = np.mean(successful)
        self.success_history.append(success_rate)
        self.reward_history.append(mean_reward)
        if self.compute_Q:
            self.Q_history.append(np.mean(Qs))
        self.n_episodes += self.rollout_batch_size

        return convert_episode_to_batch_major(episode)

    def clear_history(self):
        """Clears all histories that are used for statistics
        """
        self.success_history.clear()
        self.reward_history.clear()
        self.Q_history.clear()

    def current_success_rate(self):
        return np.mean(self.success_history)

    def current_mean_Q(self):
        return np.mean(self.Q_history)

    def current_mean_friction(self):
        return np.mean(self.friction_coefficients)

    def save_policy(self, path):
        """Pickles the current policy for later inspection.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.policy, f)

    def logs(self, prefix='worker'):
        """Generates a dictionary that contains all collected statistics.
        """
        logs = []
        logs += [('success_rate', np.mean(self.success_history))]
        logs += [('reward', np.mean(self.reward_history))]
        if self.compute_Q:
            logs += [('mean_Q', np.mean(self.Q_history))]
        logs += [('episode', self.n_episodes)]
        logs += [('friction', self.current_mean_friction())]

        if prefix is not '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

    def seed(self, seed):
        """Seeds each environment with a distinct seed derived from the passed in global seed.
        """
        # TODO
        # for idx, env in enumerate(self.envs):
        #     env.seed(seed + 1000 * idx)
        return None
