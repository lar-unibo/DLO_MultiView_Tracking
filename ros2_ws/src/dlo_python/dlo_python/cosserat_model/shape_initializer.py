import numpy as np
import torch
import matplotlib.pyplot as plt
from pipy.tf import Rotation
from pipy.utils import quat_to_rpy
from dlo_python.cosserat_model.dlo_model import DloModel, DloModelParams
from dlo_python.cosserat_model.action import convert_action_to_local_frame
from dlo_python.cosserat_model.plot import plot_interactive_3d
import pickle
CONFIG = {
    "dt": 1e-5,  # time step
    "nu": 1,  # damping coefficient for the simulation
    "n_elem": 50,  # number of elements
    "length": 0.50,  # length of the rod (m)
    "radius": 0.005,  # radius of the rod (m)
    "density": 1e3,  # density of the rod (kg/m^3)
    "youngs_modulus": 1e6,  # young's modulus of the rod (Pa)
    "action_velocity": 0.2,  # m/s (velocity of the action which influences the number of steps during simulation)
}


class DloInitShapeFromModel:

    def __init__(self):

        self.dlo_params = DloModelParams(
            dt=CONFIG["dt"],
            n_elem=CONFIG["n_elem"],
            length=CONFIG["length"],
            radius=CONFIG["radius"],
            density=CONFIG["density"],
            youngs_modulus=CONFIG["youngs_modulus"],
            nu=CONFIG["nu"],
            action_velocity=CONFIG["action_velocity"],
        )

        pos = torch.load('/docker_camere/ros2_ws/src/dlo_python/dlo_python/cosserat_model/positions.pth', weights_only=False)
        dir = torch.load('/docker_camere/ros2_ws/src/dlo_python/dlo_python/cosserat_model/directors.pth', weights_only=False)


        # data = pickle.load(open(path, "rb"))

        # self.init_pos = data["final_shape"]
        # self.init_directors = data["final_directors"]

        self.init_pos = pos.numpy()
        self.init_directors = dir.numpy()

    def run_model(self, action1, action2, debug=False, dict_out=None):
        dlo = DloModel(self.dlo_params, position=self.init_pos, directors=self.init_directors)
        dlo.build_model(action_multi_move=[action1, action2], gravity=True, damping=True, plane=False)
        dict_out = dlo.run_simulation()
        return dict_out
    
    def run(self, target_1, target_2, debug=False):
        
        init_pos = self.init_pos.T
        mean_pos = np.mean(init_pos, axis=0)
        init_pos_centered = init_pos - mean_pos

        targets_mid_point = (target_1[:3] + target_2[:3]) / 2
        init_pos_centered = init_pos_centered + targets_mid_point
         
        #SISTEMA 
        action1 = convert_action_to_local_frame(
            action_idx=0,
            target=np.concatenate([target_1[:3], quat_to_rpy(target_1[3:])]),
            dlo_shape=init_pos_centered.T,
            dlo_directors=self.init_directors,
        )

        action2 = convert_action_to_local_frame(
            action_idx=-1,
            target=np.concatenate([target_2[:3], quat_to_rpy(target_2[3:])]),
            dlo_shape=init_pos_centered.T,
            dlo_directors=self.init_directors,
        )

        dict_out = self.run_model(action1, action2, debug=debug)

        final_shape = dict_out["final_shape"].T
        final_shape = final_shape - mean_pos
        final_shape = final_shape + targets_mid_point
        

        if False:
            target_1_rot = Rotation.quaternion(target_1[3], target_1[4], target_1[5], target_1[6]).to_numpy()
            target_2_rot = Rotation.quaternion(target_2[3], target_2[4], target_2[5], target_2[6]).to_numpy()
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(
                init_pos_centered[:, 0],
                init_pos_centered[:, 1],
                init_pos_centered[:, 2],
                s=10,
                label="Initial shape",
            )
            ax.scatter(final_shape[:, 0], final_shape[:, 1], final_shape[:, 2], s=20, label="Final shape")
            plot_frame(ax, [0, 0, 0], np.eye(3), scale=0.1, text="Base")
            plot_frame(ax, target_1[:3], target_1_rot, scale=0.1, text="Target 1")
            plot_frame(ax, target_2[:3], target_2_rot, scale=0.1, text="Target 2")

            plt.axis("equal")
            plt.legend()
            plt.tight_layout()
            plt.savefig("/docker_camere/ros2_ws/output.png")

        return final_shape


def plot_frame(ax, origin, rotation_matrix, scale=0.02, text=None):
    x_dir = rotation_matrix[:, 0] * scale
    y_dir = rotation_matrix[:, 1] * scale
    z_dir = rotation_matrix[:, 2] * scale

    ax.quiver(origin[0], origin[1], origin[2], x_dir[0], x_dir[1], x_dir[2], color="r", linewidth=4)
    ax.quiver(origin[0], origin[1], origin[2], y_dir[0], y_dir[1], y_dir[2], color="g", linewidth=4)
    ax.quiver(origin[0], origin[1], origin[2], z_dir[0], z_dir[1], z_dir[2], color="b", linewidth=4)

    if text is not None:
        ax.text(origin[0], origin[1], origin[2], text, color="black", fontsize=12)