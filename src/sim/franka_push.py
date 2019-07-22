from src.sim import rlbase
from carbongym import gymapi
import numpy as np
import random
import os

class FrankaPush(rlbase.Task):

    def __init__(self, env, **base_args):
        super(FrankaPush, self).__init__(**base_args)

        self.prev_pos = None
        self.env = env
        self._gym = env._gym
        self.distance_threshold = 0.05
        self.franka_handle = self._gym.find_actor_handle(self.env._env, "franka")

        asset_options = gymapi.AssetImportOptions()
        asset_options.fix_base_link = False
        # asset_options.flip_visual_attachments = True
        asset_options.thickness = 0.004
        asset_options.armature = 0.002

        pose = gymapi.Transform()
        # add cabinet - 1 Index rigid body
        pose.p = gymapi.Vec3(0.58, 0.17, 0.0)
        pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)
        self.env.create_actor(env.shared_data.table_asset, pose, "table")

        # add object - 1 Index Rigid body
        pose, randx, randz = self.get_random_xz_pos()
        self.env.create_actor(env.shared_data.object_asset, pose, "object", self.env._env_index, 1)

        # add target
        # acceptable range x: -0.35 - 0.35
        # acceptable range y: -0.35 - 0.035

        pose, randx, randz = self.get_random_xz_pos()
        pose.p.y = 0.3

        self.env.create_actor(env.shared_data.target_asset, pose, "target", self.env._env_index, 1)
        self.goal = np.array([randx, pose.p.y, randz])

        self.table_handle = self._gym.find_actor_handle(self.env._env, "table")
        self.object_handle = self._gym.find_actor_handle(self.env._env, "object")
        self.target_handle = self._gym.find_actor_handle(self.env._env, "target")

        self.cube_rigid_shape_prop = self._gym.get_actor_rigid_shape_properties(self.env._env, self.object_handle)
        self.cube_rigid_body_prop = self._gym.get_actor_rigid_body_properties(self.env._env, self.object_handle)

        self.table_rigid_shape_prop = self._gym.get_actor_rigid_shape_properties(self.env._env, self.table_handle)
        self.table_rigid_body_prop = self._gym.get_actor_rigid_body_properties(self.env._env, self.table_handle)

        self.set_physics("object", "all_friction", 0.3)
        self.set_physics("table", "all_friction", 0.3)
        self.set_physics("object", "mass", 1.0)

        #self.cube_rigid_shape_prop[0].friction = 0.1
        #self.cube_rigid_shape_prop[0].rolling_friction = 0.05
        #self.cube_rigid_shape_prop[0].torsion_friction = 0.05

        self.object_dof_props = self._gym.get_actor_dof_properties(self.env._env, self.object_handle)
        self.table_dof_props = self._gym.get_actor_dof_properties(self.env._env, self.table_handle)

        # self.object_dof_props['stiffness'].fill(6000)
        # self.object_dof_props['damping'].fill(1000)

        c = 0.5 + 0.5 * np.random.random(3)
        color = gymapi.Vec3(c[0], c[1], c[2])
        self._gym.set_rigid_body_color(self.env._env, self.target_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

        color = gymapi.Vec3(0.1, 0.1, 0.5)
        self._gym.set_rigid_body_color(self.env._env, self.table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

        color = gymapi.Vec3(1.0, 0.5, 0.1)
        self._gym.set_rigid_body_color(self.env._env, self.object_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

        self._gym.set_actor_dof_properties(self.env._env, self.object_handle, self.object_dof_props)

        self._gym.set_actor_dof_properties(self.env._env, self.table_handle, self.table_dof_props)

    def set_physics(self, target, prop, value):
        if target == "object":
            if prop == "friction" or prop == "all_friction":
                self.cube_rigid_shape_prop[0].friction = value
            if prop == "rolling_friction" or prop == "all_friction":
                self.cube_rigid_shape_prop[0].rolling_friction = value
            if prop == "torsion_friction" or prop == "all_friction":
                self.cube_rigid_shape_prop[0].torsion_friction = value
            if prop == "mass":
                old_value = self.cube_rigid_body_prop[0].mass
                self.cube_rigid_body_prop[0].mass = value
                self.cube_rigid_body_prop[0].inertia.x.x *= (value / old_value)
                self.cube_rigid_body_prop[0].inertia.y.y *= (value / old_value)
                self.cube_rigid_body_prop[0].inertia.z.z *= (value / old_value)

                if value > 0.0:
                    self.cube_rigid_body_prop[0].invMass = 1.0 / value
                    self.cube_rigid_body_prop[0].invInertia.x.x *= (old_value / value)
                    self.cube_rigid_body_prop[0].invInertia.y.y *= (old_value / value)
                    self.cube_rigid_body_prop[0].invInertia.z.z *= (old_value / value)

            self._gym.set_actor_rigid_shape_properties(self.env._env, self.object_handle, self.cube_rigid_shape_prop)
            self._gym.set_actor_rigid_body_properties(self.env._env, self.object_handle, self.cube_rigid_body_prop)

        elif target == "franka":
            self.env.franka_dof_props[prop].fill(value)
            self._gym.set_actor_dof_properties(self.env._env, self.env.franka_handle, self.env.franka_dof_props)

        elif target == "table":
            if prop == "friction" or "all_friction":
                self.table_rigid_shape_prop[0].friction = value
            if prop == "rolling_friction" or "all_friction":
                self.table_rigid_shape_prop[0].rolling_friction = value
            if prop == "torsion_friction" or "all_friction":
                self.table_rigid_shape_prop[0].torsion_friction = value

            self._gym.set_actor_rigid_shape_properties(self.env._env, self.table_handle, self.table_rigid_shape_prop)
            self._gym.set_actor_rigid_body_properties(self.env._env, self.table_handle, self.table_rigid_body_prop)


    @staticmethod
    def num_observations():
        return 23

    @staticmethod
    def num_actions():
        return 7

    def get_random_xz_pos(self):
        pose = gymapi.Transform()
        randx = random.uniform(0.5, 0.6)
        randz = random.uniform(-0.15, 0.15)
        pose.p = gymapi.Vec3(randx, 0.35, randz)
        pose.r = gymapi.Quat(0.0, 0.0, 0.0, 0.0)
        return pose, randx, randz

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = self.goal_distance(achieved_goal, goal)
        return -(d > self.distance_threshold).astype(np.float32)

    def goal_distance(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def _is_success(self, achieved_goal, desired_goal):
        d = self.goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def get_rigid_body_pose(self, name, handle):
        body_dict = self._gym.get_actor_rigid_body_dict(self.env._env, handle)
        rigid_body_states = self._gym.get_actor_rigid_body_states(self.env._env, handle, gymapi.STATE_ALL)
        idx = body_dict[name]
        pos = rigid_body_states[idx]['pose']['p']
        rotation = rigid_body_states[idx]['pose']['r']
        vel_linear = rigid_body_states[idx]['vel']['linear']
        vel_angular = rigid_body_states[idx]['vel']['angular']

        return {"p": np.array([pos['x'], pos['y'], pos['z']]),
                "r": np.array([rotation['x'], rotation['y'], rotation['z'], rotation['w']]),
                "lin_vel": np.array([vel_linear['x'], vel_linear['y'], vel_linear['z']]),
                "ang_vel": np.array([vel_angular['x'], vel_angular['y'], vel_angular['z']])}

    def get_observations(self):

        object_rigid_body_state = self.get_rigid_body_pose("object", self.object_handle)
        grip_rigid_body_state = self.get_rigid_body_pose("panda_hand", self.env.franka_handle)
        # # grip pose
        grip_pos = grip_rigid_body_state['p']
        grip_velp = grip_rigid_body_state['lin_vel']
        #
        # #print("Gripper Velp: {}".format(grip_velp))
        #
        # # object pose
        object_pos = object_rigid_body_state['p']
        object_rot = object_rigid_body_state['r']
        #
        # # velocities
        object_velp = object_rigid_body_state['lin_vel']
        object_velr = object_rigid_body_state['ang_vel']

        object_rel_pos = object_pos - grip_pos
        object_velp -= grip_velp

        achieved_goal = np.squeeze(object_pos.copy())

        obs = np.concatenate([
            grip_pos, object_pos.ravel(), object_rel_pos.ravel(), object_rot.ravel(),
            object_velp.ravel(), object_velr.ravel(), grip_velp.ravel(),
        ])

        result_dict = {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal,
        }
        return result_dict

    def post_step(self):
        """ Called after stepping the environment.  Get observations, reward, and termination. """
        idx = self.env._env_index

        #        # write observations
        obs_dict = self.get_observations()
        achieved_goal = obs_dict['achieved_goal']

        info = {
            'is_success': self._is_success(achieved_goal, self.goal),
        }

        # self.cube_rigid_body_prop = self._gym.get_actor_rigid_body_properties(self.env._env, self.object_handle)
        # print("Mass: {}".format(self.cube_rigid_body_prop[0].mass))

        reward = self.compute_reward(achieved_goal, self.goal, info)

        return obs_dict, reward, info

    def reset(self, obs=None):
        """ Callback to re-initialize the episode """
        self.env.reset()
        self.prev_pos = None
        # New random object pose
        object = self._gym.get_actor_rigid_body_states(self.env._env, self.object_handle, gymapi.STATE_POS)
        target = self._gym.get_actor_rigid_body_states(self.env._env, self.target_handle, gymapi.STATE_POS)

        pose_object, randx_object, randz_object = self.get_random_xz_pos()
        pose_target, randx_target, randz_target = self.get_random_xz_pos()

        while self.compute_reward(np.array([randx_object, pose_object.p.y, randz_object]),
                                  np.array([randx_target, pose_target.p.y - 0.01, randz_target]), info=None) == 0:
            pose_target, randx_target, randz_target = self.get_random_xz_pos()
            pose_object, randx_object, randz_object = self.get_random_xz_pos()


        object[0]['pose']['p']['x'] = randx_object
        object[0]['pose']['p']['y'] = pose_object.p.y
        object[0]['pose']['p']['z'] = randz_object
        object[0]['pose']['r']['x'] = 0
        object[0]['pose']['r']['y'] = 0
        object[0]['pose']['r']['z'] = 0
        object[0]['vel']['linear']['x'] = 0
        object[0]['vel']['linear']['y'] = 0
        object[0]['vel']['linear']['z'] = 0
        object[0]['vel']['angular']['x'] = 0
        object[0]['vel']['angular']['y'] = 0
        object[0]['vel']['angular']['z'] = 0

        target[0]['pose']['p']['x'] = randx_target
        target[0]['pose']['p']['y'] = pose_target.p.y - 0.01
        target[0]['pose']['p']['z'] = randz_target

        self._gym.set_actor_rigid_body_states(self.env._env, self.object_handle, object, gymapi.STATE_ALL)
        self._gym.set_actor_rigid_body_states(self.env._env, self.target_handle, target, gymapi.STATE_ALL)

        self.goal = np.array([target[0]['pose']['p']['x'], target[0]['pose']['p']['y'], target[0]['pose']['p']['z']])

        return self.get_observations()

