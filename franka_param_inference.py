#%%
import os
import sys
cur_root_dir = os.getcwd()
print("Current root dir is {}".format(cur_root_dir))
if cur_root_dir not in sys.path:
    sys.path.append(cur_root_dir)
#%%
from src.utils.param_inference import *
import numpy as np

#%%
data_file = os.path.join(os.path.join(cur_root_dir, "assets/data/data_stiffness_5k.pkl"))

#%%
g = FrankaDataGenerator(data_file=data_file, load_from_disk=True, params_dim=1, data_dim=198)
params, stats = g.gen(1)
shapes = {"params": params.shape[1], "data": stats.shape[1]}

#%%
log_mdn, inf_mdn = train(epochs=1000, batch_size=150, generator=g, model="MDN", stats_dim=198, params_dim=1,
                         num_sampled_points=5000)

#%%
log_rff, inf_rff = train(epochs=1000, batch_size=150, generator=g, model="MDRFF", stats_dim=198, params_dim=1,
                         num_sampled_points=5000)


#%%
true_obs = np.array([[6000]])
get_results_from_true_obs(env_params=["stiffness"], true_obs=true_obs, generator=g, inf=inf_mdn, shapes=shapes,
                          p_lower=[0.], p_upper=[1.0])

#%%
true_obs = np.array([[40000]])
get_results_from_true_obs(env_params=["stiffness"], true_obs=true_obs, generator=g, inf=inf_mdn, shapes=shapes,
                          p_lower=[0.], p_upper=[1.0])

#%%
true_obs = np.array([[6000]])
get_results_from_true_obs(env_params=["stiffness"], true_obs=true_obs, generator=g, inf=inf_rff, shapes=shapes,
                          p_lower=[0.], p_upper=[1.0])

#%%
true_obs = np.array([[40000]])
get_results_from_true_obs(env_params=["stiffness"], true_obs=true_obs, generator=g, inf=inf_rff, shapes=shapes,
                          p_lower=[0.], p_upper=[1.0])

#%%
data_file = os.path.join(os.path.join(cur_root_dir, "assets/data/data_friction_stiffness_5k.pkl"))

#%%
g = FrankaDataGenerator(data_file=data_file, load_from_disk=True, params_dim=2, data_dim=198)
params, stats = g.gen(1)
shapes = {"params": params.shape[1], "data": stats.shape[1]}

#%%
log_mdn, inf_mdn = train(epochs=3000, batch_size=100, generator=g, model="MDN", stats_dim=198, params_dim=2,
                         num_sampled_points=5000)

#%%
log_rff, inf_rff = train(epochs=3000, batch_size=100, generator=g, model="MDRFF",stats_dim=198, params_dim=2,
                         num_sampled_points=5000)

#%%
true_obs = np.array([[0.6, 6000]])

#%%
get_results_from_true_obs(env_params=["friction", "stiffness"], true_obs=true_obs, generator=g, inf=inf_mdn, shapes=shapes,
                          p_lower=[0., 0.], p_upper=[1.0, 1.0])

#%%
get_results_from_true_obs(env_params=["friction", "stiffness"], true_obs=true_obs, generator=g, inf=inf_rff, shapes=shapes,
                          p_lower=[0., 0.], p_upper=[1.0, 1.0])


####### Mass, Friction and Stiffness

#%%
data_file = os.path.join(os.path.join(cur_root_dir, "assets/data/data_friction_mass_5k.pkl"))

#%%
g = FrankaDataGenerator(data_file=data_file, load_from_disk=True, params_dim=2, data_dim=198)
params, stats = g.gen(1)
shapes = {"params": params.shape[1], "data": stats.shape[1]}

#%%
log_mdn, inf_mdn = train(epochs=1000, batch_size=150, generator=g, model="MDN", stats_dim=198, params_dim=2,
                         num_sampled_points=5000)

#%%
log_rff, inf_rff = train(epochs=1000, batch_size=150, generator=g, model="MDRFF", stats_dim=198, params_dim=2,
                         num_sampled_points=5000)


#%%
true_obs = np.array([[0.1, 0.2]])
get_results_from_true_obs(env_params=["friction", "mass"], true_obs=true_obs, generator=g, inf=inf_mdn, shapes=shapes,
                          p_lower=[0., 0.], p_upper=[1.0, 1.0])

#%%
true_obs = np.array([[1.0, 1.0]])
get_results_from_true_obs(env_params=["friction", "mass"], true_obs=true_obs, generator=g, inf=inf_rff, shapes=shapes,
                          p_lower=[0., 0.], p_upper=[1.0, 1.0])

####### Mass

#%%
data_file = os.path.join(os.path.join(cur_root_dir, "assets/data/data_mass_5k.pkl"))

#%%
g = FrankaDataGenerator(data_file=data_file, load_from_disk=True, params_dim=1, data_dim=198, scale_params=True)
params, stats = g.gen(1)
shapes = {"params": params.shape[1], "data": stats.shape[1]}

#%%
log_mdn, inf_mdn = train(epochs=500, batch_size=150, generator=g, model="MDN", stats_dim=198, params_dim=1,
                         num_sampled_points=5000)

#%%
log_rff, inf_rff = train(epochs=1000, batch_size=150, generator=g, model="MDRFF", stats_dim=198, params_dim=1,
                         num_sampled_points=5000)


#%%
true_obs = np.array([[0.3]])
get_results_from_true_obs(env_params=["mass"], true_obs=true_obs, generator=g, inf=inf_mdn, shapes=shapes,
                          p_lower=[0.], p_upper=[1.0])

#%%
true_obs = np.array([[0.5]])
get_results_from_true_obs(env_params=["mass"], true_obs=true_obs, generator=g, inf=inf_rff, shapes=shapes,
                          p_lower=[0.], p_upper=[1.0])