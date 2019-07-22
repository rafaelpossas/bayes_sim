from src.sim.cartpole import *
from sklearn import preprocessing
from tqdm import tqdm
from stable_baselines.ppo2 import PPO2
import tensorflow as tf

class CartPoleDataGenerator():

    def __init__(self, policy_file="../models/controllers/PPO/CartPole-v1.pkl",
                 mass_prior=None, length_prior=None, episodes_per_params=1, seed=1995,
                 params=["length", "masspole"], steps_per_episode=50, sufficient_stats="Cross-Correlation"):

        self.env = CartPoleEnv()
        self.seed = seed

        self.cached_data = None
        self.params_scaler = None
        self.params = params
        self.steps_per_episode = steps_per_episode
        self.sufficient_stats = sufficient_stats

        self.policy = PPO2.load(policy_file)
        if mass_prior is None:
            self.m_low = 0.1
            self.m_high = 2.0
            self.m_prior = self.sample_mass_from_uniform_prior

        if length_prior is None:
            self.l_low = 0.1
            self.l_high = 2.0
            self.l_prior = self.sample_length_from_uniform_prior

    def sample_mass_from_uniform_prior(self):
        return np.random.uniform(self.m_low, self.m_high, 1)[0]

    def sample_length_from_uniform_prior(self):
        return np.random.uniform(self.l_low, self.l_high, 1)[0]

    def scale_params(self, params):
        params_scaler = preprocessing.MinMaxScaler()
        params = params_scaler.fit_transform(np.array(params))
        cur_params = []
        # for body in params:
        #     for par in params[body]:
        #         cur_params.append(params[body][par])
        return params, params_scaler

    def calculate_cross_correlation(self, episode):
        n_steps = len(episode['action'])

        cur_state = episode['observation']
        #next_state = obs['observation'][idx][1:]
        cur_action = episode['action']
        sdim = cur_state.shape[1]
        adim = cur_action.shape[1]
        #state_difference = np.array(list(next_state - cur_state))
        state_difference = np.array(cur_state)
        actions = np.array(cur_action)
        sample = np.zeros((sdim, adim))
        for i in range(sdim):
            for j in range(adim):
                sample[i, j] = np.dot(state_difference[:, i], actions[:, j]) / (n_steps-1)
                # Add mean of absolut states changes and std to the summary statistics

        sample = sample.reshape(-1)
        sample = np.append(sample, np.mean(state_difference, axis=0))
        sample = np.append(sample, np.std(state_difference.astype(np.float64), axis=0))

        stats = np.array(sample)

        return stats

    def rollout(self):

        t = 0
        total_reward = 0
        ep_tuple = ({"cartpole": {"masspole": self.env.masspole, "length": self.env.length}},
                    {"observation": [], "action": [], "total_reward": 0})

        self.env.seed(1995)

        state = self.env.reset()

        while True:
            t += 1
            action = self.policy.predict(state, deterministic=True)[0].ravel()

            if isinstance(self.env.action_space, gym.spaces.Box):
                action = np.clip(action, self.env.action_space.low, self.env.action_space.high)

            next_state, reward, done, _ = self.env.step(action)

            ep_tuple[1]['action'].append(action)
            ep_tuple[1]['observation'].append(next_state.ravel())

            total_reward += reward
            if t >= self.steps_per_episode or done:
                # print('episode: {} '.format(ep), 'reward: {} '.format(total_reward))
                action = np.array(ep_tuple[1]['action'])
                observation = np.array(ep_tuple[1]['observation'])
                ep_tuple[1]['action'] = action
                ep_tuple[1]['observation'] = observation
                ep_tuple[1]['total_reward'] = total_reward
                break

            else:
                state = next_state

        return ep_tuple

    def gen_single(self, param):
        for i in range(len(['length', 'masspole'])):
            setattr(self.env,
                    ['length', 'masspole'][i],
                    param[i])

        all_data = {"data": None, "params": None}
        ep_tuple = self.rollout()

        if self.sufficient_stats == "Cross-Correlation":
            all_data["data"] = self.calculate_cross_correlation(ep_tuple[1])

        if self.sufficient_stats == "State-Action":
            all_data["data"] = np.concatenate((ep_tuple[1]['observation'], ep_tuple[1]['action']), axis=1)

        all_data["params"] = ep_tuple[0]

        return all_data

    def gen(self, n_samples):

        if self.cached_data is not None and n_samples == len(self.cached_data["params"]):
            print('Data has already been generated, loading from memory...')
            return self.cached_data["params"], self.cached_data["data"]
        else:
            self.cached_data = {"data": None, "params": None}

        all_params = []
        all_data = []
        print("\nDrawing Parameters and Running simulation...")
        for ep in tqdm(range(n_samples)):

            # if ep > 0 and ep % self.episodes_per_params == 0:
            #     mass = self.m_prior()
            #     length = self.l_prior()
            #     self.env.set_dynamics(mass=mass, length=length)
            masspole = self.m_prior()
            length = self.l_prior()
            cur_params = []

            for p in self.params:
                if p == "masspole":
                    cur_params.append(masspole)
                elif p == "length":
                    cur_params.append(length)

            all_data.append(self.gen_single(cur_params)["data"])
            all_params.append(cur_params)

        all_params = np.array(all_params)
        all_data = np.array(all_data)

        self.cached_data["data"] = all_data
        self.cached_data["params"] = all_params

        return all_params, all_data

    def run_forward_model(self, true_obs, ntest=10):
        true_obs = np.ravel(true_obs)
        dt = []

        for _ in range(ntest):
            dt.append(self.gen_single(true_obs)["data"])

        # data = np.mean(data, axis=1)
        # params = np.mean(params, axis=1)

        return true_obs, np.mean(dt, axis=0)


if __name__ == "__main__":
    g = CartPoleDataGenerator()
    dt = g.gen(100)

