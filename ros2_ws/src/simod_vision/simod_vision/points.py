import copy
from std_msgs.msg import Float32MultiArray
import numpy as np
import cv2

from dlo_python.dlo_model_nn.predictor import DloNN
from pipy.tf import Frame, Vector, Rotation
from dlo_python.dlo_fusion.utils import compute_spline
from dlo_python.dlo_fusion.estimate3D import RayTracing


class Points3DTriangulation:

    def __init__(
        self,
        K_bassa,
        D_bassa,
        K_alta,
        D_alta,
        frame_world2bassa_optical,
        frame_world2alta_optical,
        frame_bassa2world,
    ):
        self.K_bassa = K_bassa
        self.D_bassa = D_bassa
        self.K_alta = K_alta
        self.D_alta = D_alta

        self.K_bassa_inv = np.linalg.inv(self.K_bassa)
        self.K_alta_inv = np.linalg.inv(self.K_alta)

        self.T_bassa = frame_world2bassa_optical.to_numpy()
        self.T_alta = frame_world2alta_optical.to_numpy()
        self.T_bassa2world = frame_bassa2world.to_numpy()

    def triangulate_dlo_shape(self, points_bassa, points_alta):
        # T_bassa: world -> bassa optical frame
        # T_alta: world -> alta optical frame
        # T_world2bassa_base: world -> bassa base frame

        # triangulation
        centers_bassa, dirs_bassa = RayTracing.compute_3d_rays_batched(points_bassa, self.K_bassa_inv, self.T_bassa)
        centers_alta, dirs_alta = RayTracing.compute_3d_rays_batched(points_alta, self.K_alta_inv, self.T_alta)

        centers = np.concatenate((centers_bassa.reshape(-1, 1, 3), centers_alta.reshape(-1, 1, 3)), axis=1)
        dirs = np.concatenate((dirs_bassa.reshape(-1, 1, 3), dirs_alta.reshape(-1, 1, 3)), axis=1)

        new_points = RayTracing.triangulate_rays_batched(centers, dirs)  # world frame

        # convert to camera bassa base frame
        new_points = new_points @ self.T_bassa2world[:3, :3].T + self.T_bassa2world[:3, 3]
        return new_points

    def check_and_add_tip_points(self, points, curr1, curr2):
        # check if points are consistent with grippers

        curr_1_T = Frame(
            Rotation.quaternion(curr1[3], curr1[4], curr1[5], curr1[6]), Vector(curr1[0], curr1[1], curr1[2])
        ).to_numpy()
        curr_2_T = Frame(
            Rotation.quaternion(curr2[3], curr2[4], curr2[5], curr2[6]), Vector(curr2[0], curr2[1], curr2[2])
        ).to_numpy()

        gap_points = np.linalg.norm(np.diff(points, axis=0), axis=1).mean()

        # RIGHT
        closer_point_1 = np.argmin(np.linalg.norm(points - curr_1_T[:3, 3], axis=1))
        point_1 = points[closer_point_1]

        next_point_1 = point_1 + curr_1_T[:3, 2] * gap_points
        closer_point_1_1 = np.argmin(np.linalg.norm(points - next_point_1, axis=1))
        point_1_1 = points[closer_point_1_1]

        # LEFT
        closer_point_2 = np.argmin(np.linalg.norm(points - curr_2_T[:3, 3], axis=1))
        point_2 = points[closer_point_2]

        next_point_2 = point_2 - curr_2_T[:3, 2] * gap_points
        closer_point_2_2 = np.argmin(np.linalg.norm(points - next_point_2, axis=1))
        point_2_2 = points[closer_point_2_2]

        mid_points = points[closer_point_1_1 + 1 : closer_point_2_2, :]
        start_points = np.array([curr_1_T[:3, 3], point_1_1]).reshape(2, -1)
        end_points = np.array([point_2_2, curr_2_T[:3, 3]]).reshape(2, -1)

        return np.concatenate((start_points, mid_points, end_points), axis=0)

    def check_and_add_tip_points_smooth(self, points, curr1, curr2, score_th=0.9):
        def consine_similarity(v1, v2):
            return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

        curr_1_T = Frame(
            Rotation.quaternion(curr1[3], curr1[4], curr1[5], curr1[6]), Vector(curr1[0], curr1[1], curr1[2])
        ).to_numpy()
        curr_2_T = Frame(
            Rotation.quaternion(curr2[3], curr2[4], curr2[5], curr2[6]), Vector(curr2[0], curr2[1], curr2[2])
        ).to_numpy()

        # subset of points
        points_s = points[:5]
        points_e = points[-5:]

        # compute scores
        scores_start = np.array([consine_similarity(point - curr_1_T[:3, 3], curr_1_T[:3, 2]) for point in points_s])
        scores_end = np.array([consine_similarity(point - curr_2_T[:3, 3], curr_2_T[:3, 2]) for point in points_e])

        # Find indices for filtering points based on scores
        idx_start = np.argmax(scores_start > score_th) if np.any(scores_start > score_th) else np.argmax(scores_start)
        idx_end = np.argmin(scores_end < -score_th) if np.any(scores_end < -score_th) else np.argmin(scores_end)

        # filter points
        points_start = points_s[idx_start:]
        points_end = points_e[:idx_end]

        new_points = np.concatenate(
            (
                curr_1_T[:3, 3].reshape(1, -1),
                points_start,
                points[5:-5],
                points_end,
                curr_2_T[:3, 3].reshape(1, -1),
            ),
            axis=0,
        )

        return new_points

    def smooth(self, points, num_points=50, s=0):
        return compute_spline(points, num_points=num_points, s=s)


class Points2DFromModel:
    def __init__(
        self, checkpoint_path, num_points, frame_bassa2alta, frame_base2optical, K_bassa, D_bassa, K_alta, D_alta
    ):

        self.num_points = num_points
        self.dlo_nn = DloNN(checkpoint_path, device="cpu")

        self.prev_tcp_link_right = None
        self.prev_tcp_link_left = None
        self.prev_prediction = None

        self.T_bassa2alta = frame_bassa2alta.to_numpy()
        self.T_base2optical = frame_base2optical.to_numpy()
        self.K_bassa = K_bassa
        self.D_bassa = D_bassa
        self.K_alta = K_alta
        self.D_alta = D_alta

        self.K_bassa_inv = np.linalg.inv(self.K_bassa)
        self.K_alta_inv = np.linalg.inv(self.K_alta)

    def update_prev_prediction(self, prev_prediction, prev_tcp_link_right, prev_tcp_link_left):
        self.prev_prediction = copy.deepcopy(prev_prediction)
        self.prev_tcp_link_right = copy.deepcopy(prev_tcp_link_right)
        self.prev_tcp_link_left = copy.deepcopy(prev_tcp_link_left)

    def get_curr_frames(self):
        return self.curr_frame_1, self.curr_frame_2

    def set_curr_frames(self, curr_frame_1, curr_frame_2):
        self.curr_frame_1 = copy.deepcopy(curr_frame_1)
        self.curr_frame_2 = copy.deepcopy(curr_frame_2)

    def compute_points2d(self):

        # predict new dlo shape
        points3d_bassa = self.dlo_nn.run(
            dlo_state=self.prev_prediction,
            curr_1=self.prev_tcp_link_right,
            curr_2=self.prev_tcp_link_left,
            target_1=self.curr_frame_1,
            target_2=self.curr_frame_2,
        )

        # projection
        points3d_alta = points3d_bassa @ self.T_bassa2alta[:3, :3].T + self.T_bassa2alta[:3, 3]

        points2d_bassa = self.project_points(points3d_bassa, self.T_base2optical, self.K_bassa, self.D_bassa)
        points2d_alta = self.project_points(points3d_alta, self.T_base2optical, self.K_alta, self.D_alta)

        points_tcps_bassa = np.array([self.curr_frame_1[:3], self.curr_frame_2[:3]])
        points_tcps_alta = points_tcps_bassa @ self.T_bassa2alta[:3, :3].T + self.T_bassa2alta[:3, 3]

        points2d_tcps_bassa = self.project_points(points_tcps_bassa, self.T_base2optical, self.K_bassa, self.D_bassa)
        points2d_tcps_alta = self.project_points(points_tcps_alta, self.T_base2optical, self.K_alta, self.D_alta)

        msg_bassa = Float32MultiArray(data=points2d_bassa.flatten())
        msg_alta = Float32MultiArray(data=points2d_alta.flatten())
        msg_tcps_bassa = Float32MultiArray(data=points2d_tcps_bassa.flatten())
        msg_tcps_alta = Float32MultiArray(data=points2d_tcps_alta.flatten())
        return msg_bassa, msg_alta, msg_tcps_bassa, msg_tcps_alta

    def project_points(self, points, pose, camera_k, camera_d):
        T = np.linalg.inv(pose)
        rvec = cv2.Rodrigues(T[:3, :3])[0]
        tvec = T[:3, 3]
        points = np.array(points).reshape(-1, 3)
        points2d = cv2.projectPoints(points, rvec, tvec, camera_k, camera_d)[0].squeeze()
        return points2d.astype(int)
