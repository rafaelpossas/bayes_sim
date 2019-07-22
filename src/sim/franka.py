import sys
from os.path import dirname

sys.path.append(dirname(__file__))

from src.sim import rlbase
from carbongym import gymapi
import numpy as np
import os


class FrankaSharedData:

    def __init__(self, gym, sim):
        asset_options = gymapi.AssetImportOptions()
        asset_options.fix_base_link = False
        asset_options.flip_visual_attachments = True
        asset_options.thickness = 0.005
        asset_options.armature = 0.001

        carbgym_idx = [idx for idx, path in enumerate(sys.path) if "carbgym" in path]
        if len(carbgym_idx) != 1:
            raise Exception("Couldn't find carbgym path")

        self.asset_root = os.sep.join(sys.path[carbgym_idx[0]].split(os.sep)[:-1]+['assets', 'urdf'])

        franka_asset_file = "franka_description/robots/franka_panda.urdf"
        table_asset_file = "square_table.urdf"
        object_asset_file = "cube.urdf"
        target_asset_file = "small_ball.urdf"

        asset_options.thickness = 0.003
        asset_options.armature = 0.002

        self.object_asset = gym.load_asset(sim, self.asset_root, object_asset_file, asset_options)

        asset_options.fix_base_link = True
        self.table_asset = gym.load_asset(sim, self.asset_root, table_asset_file, asset_options)

        asset_options.thickness = 0.01
        asset_options.armature = 0.001

        self.franka_asset = gym.load_asset(sim, self.asset_root, franka_asset_file, asset_options)
        self.target_asset = gym.load_asset(sim, self.asset_root, target_asset_file, asset_options)

        print("Franka Assets loaded...")


class FrankaEnv(rlbase.Environment):
    # Motor limits taken from the URDF from the franka_description ROS package
    # scale motor limits to account for fact we don't have real masses / joint damping / armature / etc
    effort_scale = 50
    motor_limits = effort_scale * np.array((87, 87, 87, 87, 12, 12, 12, 20, 20))

    def __init__(self, shared_data,
                 join_init=None,
                 rev_stiffness=4.0,
                 rev_damping=3.0,
                 prism_stiffness=5.0,
                 prism_damping=0.0,
                 kv=1.0,
                 **base_args):

        super(FrankaEnv, self).__init__(**base_args)
        self.dt = 0.01
        self.kv = kv
        self.initial_dof_pos = join_init or [0.0, 0.0, 0.0, -1.75, 0.0, 2.0, 0.8, 0., 0.]

        self.franka_pose = gymapi.Transform()

        self.shared_data = shared_data

        # add franka - 12 Indexes for Rigid Body
        self.franka_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        self.franka_pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)
        self.create_actor(self.shared_data.franka_asset, self.franka_pose, "franka", self._env_index, 0)

        # for retrieving Franka
        self.franka_handle = self._gym.find_actor_handle(self._env, "franka")

        # for retrieving finger positions
        self.f1_handle = self._gym.find_actor_rigid_body_handle(self._env, self.franka_handle, 'panda_leftfinger')
        self.f2_handle = self._gym.find_actor_rigid_body_handle(self._env, self.franka_handle, 'panda_rightfinger')

        self.franka_joint_names = self._gym.get_actor_dof_names(self._env, self.franka_handle)

        self.body_map = {}
        self.joint_map = {}
        self.initial_transforms_map = {}
        self.prev_joint_positions = {}
        self.joint_velocities = {}

        # print("franka joints:")

        # for jn in self.franka_joint_names:

        #     self.joint_map[jn] = self.get_joint_handle("franka", jn)
        #     self.prev_joint_positions[jn] = self.get_joint_position(self.joint_map[jn])
        #     self.joint_velocities[jn] = 0.0
        #     #print(jn, " = ", self.prev_joint_positions[jn])

        # override default stiffness and damping values for PD control
        # setup joint stiffness and damping
        self.franka_dof_props = self._setup_joint_props(rev_stiffness, rev_damping, prism_stiffness, prism_damping)

        self.step_counter = 0
        self.reset()

    def _setup_joint_props(self, rev_stiffness, rev_damping, prism_stiffness, prism_damping):
        self.franka_dof_props = self._gym.get_actor_dof_properties(self._env, self.franka_handle)

        self.franka_lower_limits = self.franka_dof_props['lower']
        self.franka_upper_limits = self.franka_dof_props['upper']
        self.franka_ranges = self.franka_upper_limits - self.franka_lower_limits
        self.franka_mids = 0.5 * (self.franka_upper_limits + self.franka_lower_limits)
        self.franka_num_dofs = len(self.franka_dof_props)

        self.franka_dof_props['stiffness'].fill(6000)
        self.franka_dof_props['damping'].fill(1000)

        for i in range(len(self.franka_joint_names)):
            if i < 7:
                self.franka_dof_props['stiffness'][i] = 10 ** rev_stiffness
                self.franka_dof_props['damping'][i] = 10 ** rev_damping
            else:
                self.franka_dof_props['stiffness'][i] = 10 ** prism_stiffness
                self.franka_dof_props['damping'][i] = 10 ** prism_damping

            self.franka_dof_props['effort'][i] = self.motor_limits[i]

        self._gym.set_actor_dof_properties(self._env, self.franka_handle, self.franka_dof_props)

        return self.franka_dof_props

    def get_current_joint_positions(self):
        joint_positions = []
        for jn in self.franka_joint_names:
            self.joint_map[jn] = self.get_joint_handle("franka", jn)
            joint_positions.append(self.get_joint_position(self.joint_map[jn]))

        return joint_positions

    # def get_gripper_current_cart_pos(self):
    #     joint_positions = self.get_current_joint_positions()
    #     cartesian_pos = self.franka_kdl.set_joint_init(joint_positions[:-2])
    #     # q_rw = self.franka_pose.r.inverse()
    #     world_pos = self.franka_pose.transform_point(gymapi.Vec3(cartesian_pos.p[0], cartesian_pos.p[1],
    #                                                              cartesian_pos.p[2]))
    #     return np.array([world_pos.x, world_pos.y, world_pos.z])

    def set_pose(self, franka_dof_positions):
        # set the pose
        franka_dof_states = self._gym.get_actor_dof_states(self._env, self.franka_handle, gymapi.STATE_NONE)

        for j in range(self.franka_num_dofs):
            franka_dof_states['pos'][j] = franka_dof_positions[j]
            # franka_dof_states['vel'][j] = 0.0
            # print("Joint {} start value: {}".format(j, franka_dof_positions[j]))

        self._gym.set_actor_dof_states(self._env, self.franka_handle, franka_dof_states, gymapi.STATE_POS)

    def set_initial_pose(self):
        self.set_pose(self.initial_dof_pos)
        self.cur_dof_pos = self.initial_dof_pos

    def reset(self):
        self.set_initial_pose()
        self.step_counter = 0

    @staticmethod
    def num_joints():
        return 9

    @staticmethod
    def num_actions():
        return 7

    def scale_ctrl(self, ctrl):
        ctrl = (self.franka_upper_limits - self.franka_lower_limits) * ctrl
        return ctrl

    def set_franka_joint_targets(self, angles):
        self._gym.set_actor_dof_position_targets(self._env, self.franka_handle, angles)

    def get_franka_joint_targets(self):
        return self._gym.get_actor_dof_position_targets(self._env, self.franka_handle)

    def set_franka_joint_positions(self, angles):
        franka_dof_states = self._gym.get_actor_dof_states(self._env, self.franka_handle, gymapi.STATE_NONE)
        for j in range(self.franka_num_dofs):
            franka_dof_states['pos'][j] = angles[j]
        self._gym.set_actor_dof_states(self._env, self.franka_handle, franka_dof_states, gymapi.STATE_POS)

    def get_franka_joint_states(self):
        return self._gym.get_actor_dof_states(self._env, self.franka_handle, gymapi.STATE_ALL)

    def set_franka_joint_states(self, states):
        self._gym.set_actor_dof_states(self._env, self.franka_handle, states, gymapi.STATE_ALL)

    def step(self, actions, dt=0.3):
        limit_margins = [.05] * 7 + [0.0] * 2

        states = self.get_franka_joint_states()
        angles = [f['pos'] for f in states]

        # treat finger action separately
        act = actions[self._env_index][:7]
        #act_open = actions[-1]

        angles[:7] += act * self.kv * dt

        # apply velocity to gripper open
        # target_open = np.sum(angles[-2:])
        # target_open += self.dt * act_open * self.kv
        # angles[-2:] = [target_open / 2.] * 2

        # clamp to joint limits
        targets = [max(min(ang, up - margin), low + margin)
                   for ang, low, up, margin
                   in zip(angles, self.franka_lower_limits, self.franka_upper_limits, limit_margins)]

        # set new targets
        self.set_franka_joint_targets(targets)
        self.step_counter += 1

        return True

    @staticmethod
    def create_shared_data(gym, sim):
        return FrankaSharedData(gym, sim)
