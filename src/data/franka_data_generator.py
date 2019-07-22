import numpy as np
import pickle
from sklearn import preprocessing

class FrankaDataGenerator(object):

    def __init__(self, data_file="data/data_friction_object_50k.pkl",
                 params_dim=1, data_dim=154, load_from_disk=True,
                 env_params=None, scale_params=True):

        self.params_dim = params_dim
        self.data_dim = data_dim
        self.load_from_disk = load_from_disk
        self.env_params = env_params

        if load_from_disk:
            self.data_file = data_file

            with open(data_file, 'rb') as f:
                self.data = pickle.load(f)
        if not scale_params:
            self.params_scaler = None
        else:
            self.params_scaler = preprocessing.MinMaxScaler()
        self.cached_data = None

    def _get_closest_param_idx(self, param, param_arr):
        diff_arr = np.array(list(map(lambda x: np.abs(x - param), param_arr))).squeeze()
        if len(diff_arr.shape) > 1:
            return np.random.choice(np.where(np.sum(diff_arr, axis=1) == min(np.sum(diff_arr, axis=1)))[0])
        else:
            return np.random.choice(np.where(diff_arr == min(diff_arr))[0])

    def gen(self, n_samples):
        all_data = []
        all_params = []
        env_params = []

        self.cached_data = {"data": None, "params": None}

        for cur_dt in self.data:
            params = cur_dt[0]
            episode = cur_dt[1]

            cur_params = []
            stats = self.calculate_cross_correlation(episode)

            all_data.append(stats)

            for body in params:
                for par in params[body]:
                    cur_params.append(params[body][par])
                    env_params.append(body + "_" + par)

            if self.env_params is None:
                self.env_params = env_params

            all_params.append(cur_params)

        all_params = np.array(all_params) if self.params_scaler is None \
            else self.params_scaler.fit_transform(np.array(all_params))

        all_data = np.array(all_data)

        if n_samples != len(all_data):
            indexes = np.random.choice(range(len(all_data)), n_samples)
            all_params = all_params[indexes]
            all_data = all_data[indexes]

        self.cached_data['data'] = all_data
        self.cached_data['params'] = all_params

        return all_params, all_data

    def calculate_cross_correlation(self, episode):
        n_steps = len(episode['action'].squeeze())

        cur_state = episode['observation'].squeeze()
        #next_state = obs['observation'][idx][1:]
        cur_action = episode['action'].squeeze()
        sdim = cur_state.shape[1]
        adim = cur_action.shape[1]
        #state_difference = np.array(list(next_state - cur_state))
        state_difference = np.array(cur_state[1:])
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

    def run_forward_model(self, true_obs, ntest=10):
        # Number of testing samples
        data = np.zeros((self.data_dim, ntest))
        params = np.zeros((self.params_dim, ntest))

        if self.load_from_disk:

            for i in range(ntest):
                idx = self._get_closest_param_idx(true_obs, self.cached_data['params'])
                data[:, i] = self.cached_data["data"][idx]
                params[:, i] = self.cached_data["params"][idx]

            data = np.mean(data, axis=1)
            params = np.mean(params, axis=1)

        else: # TODO: Run Forward model directly from the simulator
            raise NotImplementedError

        return params, data
    #
    # def gen(self, n_samples):
    #
    #     if self.load_from_disk:
    #         indexes = np.random.choice(range(self.total_size), n_samples)
    #         return self.sufficient_stats_data["params"][indexes], self.sufficient_stats_data["data"][indexes]
    #
    #     else: # TODO: Implement samples from simulator in real time
    #         raise NotImplementedError
