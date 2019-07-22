"""
MPPI on the Franka Cabinet environment with the opening task.
Authors: Jan Czarnowski, Ankur Handa, Ian Abraham
"""
from src.sim.franka import FrankaEnv, rlbase
from src.sim.franka_push import FrankaPush

from carbongym import gymapi
import numpy as np

def main():

    # initialize gym
    gym = gymapi.acquire_gym()
    if not gym.initialize():
        print("*** Failed to initialize gym")
        exit()
    sim = gym.create_sim(0, 0)# gymapi.SIM_PHYSX)

    # create viewer
    viewer = gym.create_viewer(sim, gymapi.DEFAULT_VIEWER_WIDTH, gymapi.DEFAULT_VIEWER_HEIGHT)
    if viewer is None:
        print("*** Failed to create viewer")
        exit()

    # simulation params
    sim_params = gymapi.SimParams()
    sim_params.solver_type = 5
    sim_params.num_outer_iterations = 5
    sim_params.num_inner_iterations = 60
    sim_params.relaxation = 0.75
    sim_params.warm_start = 0.75
    sim_params.shape_collision_margin = 0.01
    sim_params.contact_regularization = 1e-5
    sim_params.deterministic_mode = True
    sim_params.gravity = gymapi.Vec3(0.0, -9.81, 0.0)
    gym.set_sim_params(sim, sim_params)

    dt = 0.01

    # create experiment
    num_envs = 2
    env_spacing = 1.5

    exp = rlbase.Experiment(gym, sim, FrankaEnv, FrankaPush, num_envs,
                            env_spacing, 50, sim_params)

    # set viewer viewpoint0
    gym.viewer_camera_look_at(viewer, exp.envs[1]._env, gymapi.Vec3(0.4, 1.3, -1.3), gymapi.Vec3(.8, 0, 2))

    # create and configure MPPI
    def set_state(s):
        for i in range(0, num_envs):
            exp.envs[i].set_state(*s)

    def get_state():
        return exp.envs[0].get_state()

    def step(action):
        obs, rews, _, info = exp.step(action)
        return obs, rews
    step_counts = 0
    all_obs = []
    while not gym.query_viewer_has_closed(viewer):
        # calculate action with mppi

        if step_counts == 500:
            last_obs = []

            for i in range(num_envs):
                last_obs.append(exp.envs[i].get_franka_joint_states()['pos'])

            print(last_obs[0])
            all_obs.append(last_obs)
            step_counts = 0
            exp.reset()

        step_counts += 1
        action = np.random.uniform(-1, 1, size=(num_envs, 7))
        # take the action
        obs, rews = step(action)

        # draw some debug visualization
        gym.clear_lines(viewer)
        for i in range(len(exp.envs)):
            env = exp.envs[i]
            task = exp.tasks[i]

        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)

    gym.shutdown()


if __name__ == '__main__':
    main()
