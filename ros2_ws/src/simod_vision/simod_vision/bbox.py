import numpy as np
from std_msgs.msg import Float32MultiArray
import cv2
import numpy as np


class BboxTcp:

    def __init__(self, frame_base2optical, frame_bassa2alta, K_bassa, D_bassa, K_alta, D_alta):

        self.T_base2optical = frame_base2optical.to_numpy()
        self.frame_bassa2alta_np = frame_bassa2alta.to_numpy()
        self.K_bassa = K_bassa
        self.D_bassa = D_bassa
        self.K_alta = K_alta
        self.D_alta = D_alta

    def compute_bbox(self, tcp_link_right, tcp_link_left, dlo_length=0.5):

        # crop bassa
        slack = dlo_length - np.linalg.norm(tcp_link_right[:3] - tcp_link_left[:3])
        point_slack_left = tcp_link_left[:3] + np.array([0.0, 0.0, -slack])
        point_slack_right = tcp_link_right[:3] + np.array([0.0, 0.0, -slack])

        points = np.array([point_slack_left, point_slack_right, tcp_link_left[:3], tcp_link_right[:3]])
        points2d_bassa = self.project_points(points, self.T_base2optical, self.K_bassa, self.D_bassa).astype(
            np.float32
        )
        msg_bassa = Float32MultiArray(
            data=[
                min(points2d_bassa[:, 0]),
                min(points2d_bassa[:, 1]),
                max(points2d_bassa[:, 0]),
                max(points2d_bassa[:, 1]),
            ]
        )

        # crop alta
        points_alta = points @ self.frame_bassa2alta_np[:3, :3].T + self.frame_bassa2alta_np[:3, 3]
        points2d_alta = self.project_points(points_alta, self.T_base2optical, self.K_alta, self.D_alta).astype(
            np.float32
        )
        msg_alta = Float32MultiArray(
            data=[
                min(points2d_alta[:, 0]),
                min(points2d_alta[:, 1]),
                max(points2d_alta[:, 0]),
                max(points2d_alta[:, 1]),
            ]
        )

        return msg_bassa, msg_alta

    def project_points(self, points, pose, camera_k, camera_d):
        T = np.linalg.inv(pose)
        rvec = cv2.Rodrigues(T[:3, :3])[0]
        tvec = T[:3, 3]
        points = np.array(points).reshape(-1, 3)
        points2d = cv2.projectPoints(points, rvec, tvec, camera_k, camera_d)[0].squeeze()
        return points2d.astype(int)
