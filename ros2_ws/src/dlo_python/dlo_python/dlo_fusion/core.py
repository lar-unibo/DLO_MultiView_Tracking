import cv2, os
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import arrow

from dlo_fusion.utils import project_points, compute_confidence_mask, compute_spline, points_association_and_correction
from dlo_fusion.graph_gen import GraphGeneration
from dlo_fusion.estimate3D import RayTracing


@dataclass
class CamParams:
    pose: np.ndarray
    intrinsic: np.ndarray
    img_w: int
    img_h: int


class DloFusion:
    def __init__(self, cam1: CamParams, cam2: CamParams):

        self.graph_gen = GraphGeneration(n_knn=8, th_edges_similarity=0.25, th_mask=127, wsize=15, sampling_ratio=0.1)

        self.cam1 = cam1
        self.cam2 = cam2

    def run(self, dlo_points_3d, mask1, mask2):

        t0 = arrow.utcnow()
        pred1, pred2 = self.project_points_to_cameras(dlo_points_3d)

        confidence_mask1, confidence_mask2 = self.confidence_masks(dlo_points_3d)

        spline1_path, spline2_path = self.splines_from_graphs(
            mask1, mask2, start1=pred1[0], end1=pred1[-1], start2=pred2[0], end2=pred2[-1]
        )

        values = self.triangulate(pred1, pred2, spline1_path, spline2_path)

        t1 = arrow.utcnow()
        time = (t1 - t0).total_seconds() * 1000
        print("Time: ", time)

        return values

    def project_points_to_cameras(self, dlo_points_3d):
        cam1_pose_inv = np.linalg.inv(self.cam1.pose)
        cam2_pose_inv = np.linalg.inv(self.cam2.pose)
        points_1 = project_points(dlo_points_3d, cam1_pose_inv, self.cam1.intrinsic, self.cam1.img_w, self.cam1.img_h)
        points_2 = project_points(dlo_points_3d, cam2_pose_inv, self.cam2.intrinsic, self.cam2.img_w, self.cam2.img_h)
        print("Projected 3d points to image planes: ", points_1.shape, points_2.shape)
        return points_1, points_2

    def confidence_masks(self, dlo_points_3d, volume_size=0.05):
        confidence_mask1 = compute_confidence_mask(
            dlo_points_3d,
            self.cam1.pose,
            self.cam1.img_w,
            self.cam1.img_h,
            self.cam1.intrinsic,
            volume_size=volume_size,
        )
        confidence_mask2 = compute_confidence_mask(
            dlo_points_3d,
            self.cam2.pose,
            self.cam2.img_w,
            self.cam2.img_h,
            self.cam2.intrinsic,
            volume_size=volume_size,
        )
        return confidence_mask1, confidence_mask2

    def splines_from_graphs(self, mask1, mask2, start1, end1, start2, end2):

        ###################
        # Graph Generation
        nodes1, edges1, path1 = self.graph_gen.exec(mask1, path_start=start1, path_end=end1)
        nodes2, edges2, path2 = self.graph_gen.exec(mask2, path_start=start2, path_end=end2)
        print("Graph Generation 1: ", nodes1.shape, edges1.shape, path1.shape)
        print("Graph Generation 2: ", nodes2.shape, edges2.shape, path2.shape)

        spline1_path = compute_spline(path1, num_points=500)
        spline2_path = compute_spline(path2, num_points=500)

        return spline1_path, spline2_path

    def triangulate(self, pred1, pred2, spline1_path, spline2_path):

        points1_corrected = points_association_and_correction(pred1, spline1_path)
        points2_corrected = points_association_and_correction(pred2, spline2_path)

        ###############################################################
        # triangulation
        intrinsic1_inv = np.linalg.inv(self.cam1.intrinsic)
        intrinsic2_inv = np.linalg.inv(self.cam2.intrinsic)
        centers1, dirs1 = RayTracing.compute_3d_rays_batched(points1_corrected, intrinsic1_inv, self.cam1.pose)
        centers2, dirs2 = RayTracing.compute_3d_rays_batched(points2_corrected, intrinsic2_inv, self.cam2.pose)
        centers = np.concatenate((centers1.reshape(-1, 1, 3), centers2.reshape(-1, 1, 3)), axis=1)
        dirs = np.concatenate((dirs1.reshape(-1, 1, 3), dirs2.reshape(-1, 1, 3)), axis=1)
        values = RayTracing.triangulate_rays_batched(centers, dirs)

        return values
