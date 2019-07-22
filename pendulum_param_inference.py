#%% Imports
from src.data.pendulum_data_generator import PendulumDataGenerator
from src.utils.param_inference import *
import os

cur_root_dir = os.getcwd()
print("Current Directory: {}".format(cur_root_dir))
#%% Load Policy

policy_file = os.path.join(cur_root_dir, "src", "models", "controllers", "PPO", "Pendulum-v0.pkl")
g = PendulumDataGenerator(policy_file=policy_file, load_from_file=False)


#%% Load data shape
params, stats = g.gen(1)
shapes = {"params": params.shape[1], "data": stats.shape[1]}
print("Total data size: {}".format(params.shape[0]))

#%% Train model
log_mdn, inf_mdn = train(epochs=1000, batch_size=100, params_dim=2, stats_dim=9, num_sampled_points=1000,
                         generator=g, model="MDRFF", n_components=10)

#%% Plot Results for mass and length specific params
true_obs = np.array([[0.5, 0.5]])

get_results_from_true_obs(env_params=["length", "mass"], true_obs=true_obs, generator=g, inf=inf_mdn, shapes=shapes,
                          p_lower=[0.1, 0.1], p_upper=[2.0, 2.0])




