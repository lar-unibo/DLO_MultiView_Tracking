import os, pickle, glob
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from pipy.utils import rotation_from1to2

from pipy.tf import Frame, Vector, Rotation


class MlpDloDataset(Dataset):
    def __init__(self, dataset_path, additive_noise=False):
        super().__init__()
        self.additive_noise = additive_noise

        data_files = glob.glob(os.path.join(dataset_path, "*.pkl"))
        data_files = sorted(data_files)
        print(f"Found {len(data_files)} files")
        self.data_samples = []
        for it, file in enumerate(data_files):
            x = self.process_sample(file)
            if x is not None:
                self.data_samples.append(x)

    def process_sample(self, file):
        data = self.load_sample(file)

        if data is None:
            return None

        x0, x1, act0, ac1 = data

        x0_t = torch.from_numpy(x0).float()
        x1_t = torch.from_numpy(x1).float()
        act0_t = torch.from_numpy(act0).float()
        ac1_t = torch.from_numpy(ac1).float()

        return x0_t, x1_t, act0_t, ac1_t

    def load_sample(self, file):
        data = pickle.load(open(file, "rb"))

        x0 = data["init_shape"].T
        x1 = data["final_shape"].T

        # check if nan
        if np.isnan(x0).any() or np.isnan(x1).any():
            return None

        dir_0 = data["init_directors"].T
        dir_1 = data["final_directors"].T

        curr1_pos = x0[0]
        curr1_rot = dir_0[0]
        curr1_frame = Frame(Rotation(curr1_rot), Vector(curr1_pos[0], curr1_pos[1], curr1_pos[2]))

        curr2_pos = x0[-1]
        curr2_rot = dir_0[-1]
        curr2_frame = Frame(Rotation(curr2_rot), Vector(curr2_pos[0], curr2_pos[1], curr2_pos[2]))

        target1_pos = x1[0]
        target1_rot = dir_1[0]
        target1_frame = Frame(Rotation(target1_rot), Vector(target1_pos[0], target1_pos[1], target1_pos[2]))

        target2_pos = x1[-1]
        target2_rot = dir_1[-1]
        target2_frame = Frame(Rotation(target2_rot), Vector(target2_pos[0], target2_pos[1], target2_pos[2]))

        act0, act1 = self.compute_actions(curr1_frame, curr2_frame, target1_frame, target2_frame)

        # normalize x0 and x1
        mean_pos = x0.mean(axis=0)
        x0[:, 0] -= mean_pos[0]
        x0[:, 1] -= mean_pos[1]
        x0[:, 2] -= mean_pos[2]
        x1[:, 0] -= mean_pos[0]
        x1[:, 1] -= mean_pos[1]
        x1[:, 2] -= mean_pos[2]

        if self.additive_noise:
            random_x0_noise = np.random.normal(0, 0.0005, x0.shape)
            x0 = x0 + random_x0_noise

        return x0, x1, act0, act1

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        return self.data_samples[idx]

    def compute_actions(self, curr_1, curr_2, target_1, target_2):

        # compute position displacement
        disp_pos1 = target_1.p.to_numpy() - curr_1.p.to_numpy()
        disp_pos2 = target_2.p.to_numpy() - curr_2.p.to_numpy()

        # compute rotation displacement
        disp_rot1 = rotation_from1to2(curr_1.M, target_1.M).get_quaternion()
        disp_rot2 = rotation_from1to2(curr_2.M, target_2.M).get_quaternion()

        ##########################
        target_1_ok = np.concatenate([disp_pos1, disp_rot1])
        target_2_ok = np.concatenate([disp_pos2, disp_rot2])

        return target_1_ok, target_2_ok


if __name__ == "__main__":

    path = "/home/lar/dev24/DLO_FUSION/DATASETS_NN/dataset_new_2/val"
    dataset = MlpDloDataset(path, additive_noise=False)

    for i in range(len(dataset)):
        x0, x1, act0, act1 = dataset[i]

        if True:
            import matplotlib.pyplot as plt

            np.set_printoptions(precision=5, suppress=True)

            x0 = x0.detach().cpu().numpy()
            x1 = x1.detach().cpu().numpy()

            print(act0)
            print(act1)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(x0[:, 0], x0[:, 1], x0[:, 2], label="x0")
            ax.scatter(x1[:, 0], x1[:, 1], x1[:, 2], label="pred_x1")
            ax.scatter(x0[0, 0], x0[0, 1], x0[0, 2], label="x0_0", color="r", s=100)
            ax.scatter(x1[0, 0], x1[0, 1], x1[0, 2], label="x1_0", color="g", s=100)
            ax.legend()
            plt.axis("equal")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.tight_layout()
            plt.show()
