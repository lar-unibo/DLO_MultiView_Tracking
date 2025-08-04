import elastica as ea
from scipy.spatial.transform import Rotation, Slerp
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from pipy.utils import rpy_to_rot, rot_to_rpy


def plot_frame(ax, origin, rotation_matrix, scale=0.02):
    x_dir = rotation_matrix[:, 0] * scale
    y_dir = rotation_matrix[:, 1] * scale
    z_dir = rotation_matrix[:, 2] * scale

    ax.quiver(origin[0], origin[1], origin[2], x_dir[0], x_dir[1], x_dir[2], color="r", linewidth=4)
    ax.quiver(origin[0], origin[1], origin[2], y_dir[0], y_dir[1], y_dir[2], color="g", linewidth=4)
    ax.quiver(origin[0], origin[1], origin[2], z_dir[0], z_dir[1], z_dir[2], color="b", linewidth=4)


@dataclass
class Action:
    idx: int
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float


def convert_action_to_global_frame(action, dlo_shape, dlo_directors):
    """
    action contains the following fields:
    - idx: the index of the node where the action is applied (idx, idx+1)
    - x, y, z: the displacement of the node in the local frame
    - roll, pitch, yaw: the rotation of the node in the local frame
    dlo_shape is the shape of the rod in the global frame
    dlo_directors is the orientations of the rod in the global frame
    """

    # node orientation wrt the global frame
    Q = dlo_directors[..., action.idx].T

    # node position wrt the global frame
    x_world = dlo_shape[..., action.idx]

    # node position wrt the local frame
    x = Q.T @ x_world

    # add the displacements
    new_frame_world = np.matmul(Q, rpy_to_rot([action.roll, action.pitch, action.yaw]))
    new_x = Q @ (x + np.array([action.x, action.y, action.z]))

    return new_x, new_frame_world


def convert_action_to_local_frame(action_idx, target, dlo_shape, dlo_directors):

    Q = dlo_directors[..., action_idx].T  # node orientation wrt the global frame
    x_world = dlo_shape[..., action_idx]  # node position wrt the global frame

    target_pos = target[:3]
    target_rot = rpy_to_rot(target[3:])

    disp_local = Q.T @ (target_pos - x_world)

    rot = Q.T @ target_rot
    rpy = rot_to_rpy(rot)

    if False:
        print("x_world", x_world)
        print("target_pos", target_pos)

        x = Q.T @ x_world
        new_frame_world = np.matmul(Q, rpy_to_rot(rpy))
        new_x = Q @ (x + np.array(disp_local))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(dlo_shape[0], dlo_shape[1], dlo_shape[2], c="b")
        plot_frame(ax, x_world, Q)
        plot_frame(ax, target_pos, target_rot)
        plot_frame(ax, new_x, new_frame_world)
        plt.axis("equal")
        plt.tight_layout()
        plt.show()

    return Action(action_idx, disp_local[0], disp_local[1], disp_local[2], rpy[0], rpy[1], rpy[2])


def interpolate_waypoints(initial_pose, target_pose, num_waypoints):
    """
    Generates waypoints between initial and target poses using linear interpolation for position
    and SLERP for orientation.

    :param initial_pose: Initial pose as (x, y, z, roll, pitch, yaw)
    :param target_pose: Target pose as (x, y, z, roll, pitch, yaw)
    :param num_waypoints: Number of waypoints to generate
    :return: Array of waypoints
    """
    initial_position = np.array(initial_pose[:3])
    target_position = np.array(target_pose[:3])
    initial_rpy = np.array(initial_pose[3:])
    target_rpy = np.array(target_pose[3:])

    waypoints = []
    for i in range(num_waypoints):
        t = i / (num_waypoints - 1)
        waypoint = initial_position * (1 - t) + target_position * t
        waypoints.append(waypoint)

    # slerp
    key_rots = Rotation.from_matrix([rpy_to_rot(initial_rpy), rpy_to_rot(target_rpy)])
    slerp = Slerp([0, 1], key_rots)
    slerp_quats = slerp(np.linspace(0, 1, num_waypoints))
    slerp_rpy = np.array([rot_to_rpy(quat.as_matrix()) for quat in slerp_quats])
    waypoints = np.concatenate([waypoints, slerp_rpy], axis=1)
    return waypoints


class MoveAction(ea.NoForces):

    def __init__(self, action, dt, velocity):
        """
        action: list of floats [idx, x, y, z, r, p, y]
        dt: float, time step
        velocity: float, m/s
        """

        self.action_idx = action.idx
        self.action_disp = np.array([action.x, action.y, action.z])
        self.action_theta = np.array([action.roll, action.pitch, action.yaw])

        #
        self.dt = dt
        self.vel = velocity
        self.init_pos_0 = None
        self.init_pos_1 = None
        self.disp0_inc = None
        self.disp1_inc = None

        ##########################################
        disp_norm = np.linalg.norm(self.action_disp)
        if disp_norm == 0:
            self.steps = 100000
        else:
            self.steps = disp_norm / (self.vel * self.dt)

    def apply_forces(self, system, time: float = 0.0):
        curr_step = int(time / self.dt)
        if curr_step == 0:
            self.init_pos_0 = system.position_collection[..., self.action_idx].copy()
            self.init_dir_0 = system.director_collection[..., self.action_idx].copy()
            self.init_pos_1 = system.position_collection[..., self.action_idx + 1].copy()
            self.disp0_inc, self.disp1_inc = self.compute_edge_disps_increments()

            self.rpy_inc = self.action_theta / self.steps

        # Move the position
        system.position_collection[..., self.action_idx] = self.init_pos_0 + self.disp0_inc * curr_step
        system.position_collection[..., self.action_idx + 1] = self.init_pos_1 + self.disp1_inc * curr_step

        # Move the director
        rot_step = rpy_to_rot(self.rpy_inc * curr_step)
        system.director_collection[..., self.action_idx] = np.dot(self.init_dir_0.T, rot_step).T

    def compute_edge_disps_increments(self):

        edge_pos_world = (self.init_pos_0 + self.init_pos_1) / 2
        edge_dir_world = self.init_pos_1 - self.init_pos_0
        edge_len = np.linalg.norm(edge_dir_world)
        edge_dir_world = edge_dir_world / edge_len

        ##############################
        # LOCAL
        ##############################
        rot = self.init_dir_0.T
        edge_pos_local = rot.T @ edge_pos_world
        edge_dir_local = rot.T @ edge_dir_world

        # new pos
        new_edge_pos_local = edge_pos_local + self.action_disp

        # new edge dir
        new_edge_dir_local = np.dot(rpy_to_rot(self.action_theta), edge_dir_local)

        ##############################
        # WORLD
        ##############################
        new_edge_pos = rot @ new_edge_pos_local
        new_edge_dir = rot @ new_edge_dir_local

        # new pos node 0
        pos0_tgt = new_edge_pos - new_edge_dir * edge_len / 2

        # new pos node 1
        pos1_tgt = new_edge_pos + new_edge_dir * edge_len / 2

        # displacement (total)
        disp0 = pos0_tgt - self.init_pos_0
        disp1 = pos1_tgt - self.init_pos_1

        # displacement (increment)
        disp0_inc = disp0 / self.steps
        disp1_inc = disp1 / self.steps

        return disp0_inc, disp1_inc


class EndsMoveAction(ea.NoForces):

    def __init__(self, list_actions, dt, velocity):
        """
        list_actions: list of list of floats [idx, x, y, z, r, p, y] for the ends
        dt: float, time step
        velocity: float, m/s
        """
        self.dt = dt
        self.vel = velocity
        self.list_actions = list_actions

        max_disp_norm = 0
        for action in list_actions:
            disp_norm = np.linalg.norm([action.x, action.y, action.z])
            if disp_norm > max_disp_norm:
                max_disp_norm = disp_norm

        if max_disp_norm == 0:
            self.steps = 50000
        else:
            self.steps = max_disp_norm / (self.vel * self.dt)
        ##########################################

        self.action_disp = np.array([action.x, action.y, action.z])
        self.action_theta = np.array([action.roll, action.pitch, action.yaw])

        self.list_disp_inc = []
        self.list_init_pos = []
        self.list_rpy_inc = []
        self.list_init_dir = []

    def apply_forces(self, system, time: float = 0.0):
        curr_step = int(time / self.dt)
        if curr_step == 0:
            self.num_nodes = system.position_collection.shape[1]

            for it, action in enumerate(self.list_actions):
                if it == 0:
                    init_pos_0 = system.position_collection[..., 0].copy()
                    init_pos_1 = system.position_collection[..., 1].copy()
                    init_dir_0 = system.director_collection[..., 0].copy()
                else:
                    init_pos_0 = system.position_collection[..., -1].copy()
                    init_pos_1 = system.position_collection[..., -2].copy()
                    init_dir_0 = system.director_collection[..., -1].copy()

                disp = np.array([action.x, action.y, action.z])
                theta = np.array([action.roll, action.pitch, action.yaw])

                disp0_inc, disp1_inc = self.compute_edge_disps_increments(
                    init_pos_0, init_dir_0, init_pos_1, disp, theta
                )

                rpy_inc = theta / self.steps

                self.list_disp_inc.append((disp0_inc, disp1_inc))
                self.list_init_pos.append((init_pos_0, init_pos_1))
                self.list_init_dir.append(init_dir_0)
                self.list_rpy_inc.append(rpy_inc)

        for i, action in enumerate(self.list_actions):
            disp0_inc, disp1_inc = self.list_disp_inc[i]
            init_pos_0, init_pos_1 = self.list_init_pos[i]
            init_dir_0 = self.list_init_dir[i]
            rpy_inc = self.list_rpy_inc[i]

            # Move the director
            rot_step = rpy_to_rot(rpy_inc * curr_step)

            # Move the position
            if i == 0:
                system.position_collection[..., 0] = init_pos_0 + disp0_inc * curr_step
                system.position_collection[..., 1] = init_pos_1 + disp1_inc * curr_step
                system.director_collection[..., 0] = np.dot(init_dir_0.T, rot_step).T
            else:
                system.position_collection[..., -1] = init_pos_0 + disp0_inc * curr_step
                system.position_collection[..., -2] = init_pos_1 + disp1_inc * curr_step
                system.director_collection[..., -1] = np.dot(init_dir_0.T, rot_step).T

    def compute_edge_disps_increments(self, init_pos_0, init_dir_0, init_pos_1, action_disp, action_theta):

        edge_pos_world = (init_pos_0 + init_pos_1) / 2
        edge_dir_world = init_pos_1 - init_pos_0
        edge_len = np.linalg.norm(edge_dir_world)
        edge_dir_world = edge_dir_world / edge_len

        ##############################
        # LOCAL
        ##############################
        rot = init_dir_0.T
        edge_pos_local = rot.T @ edge_pos_world
        edge_dir_local = rot.T @ edge_dir_world

        # new pos
        new_edge_pos_local = edge_pos_local + action_disp

        # new edge dir
        new_edge_dir_local = np.dot(rpy_to_rot(action_theta), edge_dir_local)

        ##############################
        # WORLD
        ##############################
        new_edge_pos = rot @ new_edge_pos_local
        new_edge_dir = rot @ new_edge_dir_local

        # new pos node 0
        pos0_tgt = new_edge_pos - new_edge_dir * edge_len / 2

        # new pos node 1
        pos1_tgt = new_edge_pos + new_edge_dir * edge_len / 2

        # displacement (total)
        disp0 = pos0_tgt - init_pos_0
        disp1 = pos1_tgt - init_pos_1

        # displacement (increment)
        disp0_inc = disp0 / self.steps
        disp1_inc = disp1 / self.steps

        return disp0_inc, disp1_inc
