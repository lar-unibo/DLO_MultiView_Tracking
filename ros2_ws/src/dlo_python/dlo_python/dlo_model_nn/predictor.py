import torch
import numpy as np
from pipy.utils import rotation_from1to2, list_quat_to_rot
from dlo_python.dlo_model_nn.model import Model


class DloNN:

    def __init__(self, checkpoint_path, device="cpu") -> None:
        self.device = device
        print("DloNN device:", self.device)

        state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.model = Model(state)
        self.model = self.model.to(self.device)
        self.model.load_state_dict(state["model"])
        self.model.eval()

    def predict(self, dlo0, act0, act1):
        dlo0 = torch.tensor(dlo0, dtype=torch.float32).to(self.device).unsqueeze(0)
        act0 = torch.tensor(act0, dtype=torch.float32).to(self.device).unsqueeze(0)
        act1 = torch.tensor(act1, dtype=torch.float32).to(self.device).unsqueeze(0)
        pred = self.model(dlo0, act0, act1).squeeze()
        return pred.detach().cpu().numpy()

    def compute_actions(self, curr_1, curr_2, target_1, target_2):

        # compute position displacement
        disp_pos1 = target_1[:3] - curr_1[:3]
        disp_pos2 = target_2[:3] - curr_2[:3]

        # compute rotation displacement
  
        disp_rot1 = rotation_from1to2(list_quat_to_rot(curr_1[3:]), list_quat_to_rot(target_1[3:])).get_quaternion()
        disp_rot2 = rotation_from1to2(list_quat_to_rot(curr_2[3:]), list_quat_to_rot(target_2[3:])).get_quaternion()

        ##########################
        target_1_ok = np.concatenate([disp_pos1, disp_rot1])
        target_2_ok = np.concatenate([disp_pos2, disp_rot2])

        return target_1_ok, target_2_ok

    def run(self, dlo_state, curr_1, curr_2, target_1, target_2):
        """
        dlo_state: initial shape of the dlo (np.array)
        curr_1: curr pose of target1 (Frame)
        curr_2: curr pose of target2 (Frame)
        target_1: target pose of target1 (Frame)
        target_2: target pose of target2 (Frame)

        """

        # compute actions
        act0, act1 = self.compute_actions(curr_1, curr_2, target_1, target_2)

        # normalize init shape
        mean_pos = np.mean(dlo_state, axis=0)
        dlo_state_copy = dlo_state.copy()
        dlo_state_copy[:, 0] -= mean_pos[0]
        dlo_state_copy[:, 1] -= mean_pos[1]
        dlo_state_copy[:, 2] -= mean_pos[2]

        # predict
        final_shape = self.predict(dlo_state_copy, act0, act1)

        # denormalize
        final_shape[:, 0] += mean_pos[0]
        final_shape[:, 1] += mean_pos[1]
        final_shape[:, 2] += mean_pos[2]

        return final_shape

    def plot_frame(self, ax, origin, rotation_matrix, scale=0.02, text=None):
        x_dir = rotation_matrix[:, 0] * scale
        y_dir = rotation_matrix[:, 1] * scale
        z_dir = rotation_matrix[:, 2] * scale

        ax.quiver(origin[0], origin[1], origin[2], x_dir[0], x_dir[1], x_dir[2], color="r", linewidth=4)
        ax.quiver(origin[0], origin[1], origin[2], y_dir[0], y_dir[1], y_dir[2], color="g", linewidth=4)
        ax.quiver(origin[0], origin[1], origin[2], z_dir[0], z_dir[1], z_dir[2], color="b", linewidth=4)

        # text
        if text is not None:
            ax.text(origin[0], origin[1], origin[2], text, color="black", fontsize=12)
