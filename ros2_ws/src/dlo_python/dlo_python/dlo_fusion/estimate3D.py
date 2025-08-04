import numpy as np
import cv2


def get_3d_point(pixel, distance, camera_K):
    # pixel_0 = column
    # pixel_1 = row

    x = (pixel[0] - camera_K[0][2]) * distance / camera_K[0][0]
    y = (pixel[1] - camera_K[1][2]) * distance / camera_K[1][1]
    return x, y, distance


def project_to_3d(points, points3d, T, K):
    points3d_proj = []
    for it, p in enumerate(points):
        points3d_proj.append(get_3d_point([p[0], p[1]], points3d[it, 1], K))
    points3d_proj = np.array(points3d_proj)
    points3d_proj = np.dot(T[:3, :3], points3d_proj.T).T + T[:3, 3]
    return points3d_proj


def interpolate_between_shapes(points3d, new_points3d, side_safe, top_safe, max_distance=5):
    ids_side_not_safe = np.where(side_safe == 0)[0]
    ids_top_not_safe = np.where(top_safe == 0)[0]

    close_ids_side_not_safe = []
    if len(ids_side_not_safe) > 0:
        for it in range(len(new_points3d)):
            id_below = it - max(ids_side_not_safe)
            id_above = min(ids_side_not_safe) - it

            if id_above > 0 and id_above < max_distance:
                close_ids_side_not_safe.append(it)
            if id_below > 0 and id_below < max_distance:
                close_ids_side_not_safe.append(it)

    close_ids_top_not_safe = []
    if len(ids_top_not_safe) > 0:
        for it in range(len(new_points3d)):
            id_below = it - max(ids_top_not_safe)
            id_above = min(ids_top_not_safe) - it

            if id_above > 0 and id_above < max_distance:
                close_ids_top_not_safe.append(it)
            if id_below > 0 and id_below < max_distance:
                close_ids_top_not_safe.append(it)

    # Iterate through the new points and update them based on safety flags
    for it in ids_side_not_safe:
        new_points3d[it] = points3d[it]

    for it in ids_top_not_safe:
        new_points3d[it] = points3d[it]

    # interpolate for close ids
    for it in close_ids_side_not_safe:
        min_distance = min(abs(it - ids_side_not_safe))
        w = min_distance / max_distance
        new_points3d[it] = (1 - w) * points3d[it] + w * new_points3d[it]
    for it in close_ids_top_not_safe:
        min_distance = min(abs(it - ids_top_not_safe))
        w = min_distance / max_distance
        new_points3d[it] = (1 - w) * points3d[it] + w * new_points3d[it]

    return new_points3d


class RayTracing:

    @staticmethod
    def triangulate_rays_batched(rays_center, rays_dir):
        """
        Computes 3D intersection of rays

        Parameters
        ------------
        rays_center: np.array of shape (B, N, 3)
                An array containing the 3D coordinates of the rays center

        rays_dir: np.array of shape (B, N, 3)
                An array containing the 3D coordinates of the rays direction

        Returns
        ------------
        np.array
                Vector Bx3 containing the 3D coordinates of the intersection points

        """
        # Normalize the direction vectors
        rays_dir /= np.linalg.norm(rays_dir, axis=2, keepdims=True)

        # B: number of batches, N: number of rays per batch
        B, N, _ = rays_center.shape
        v_l = np.zeros((B, 3, 3))
        v_r = np.zeros((B, 3, 1))

        for i in range(B):
            # Compute the v_matrices for each direction in the batch
            v_matrices = np.eye(3) - np.einsum("ij,ik->ijk", rays_dir[i], rays_dir[i])

            # Sum the v_matrices to get v_l for the batch
            v_l[i] = np.sum(v_matrices, axis=0)

            # Compute v_r for the batch
            v_r[i] = np.einsum("ijk,ik->ij", v_matrices, rays_center[i]).sum(axis=0).reshape(3, 1)

        # Compute the intersection points for each batch
        values = np.matmul(np.linalg.pinv(v_l), v_r).reshape(B, 3)

        return values

    @staticmethod
    def compute_3d_rays_batched(points_2d, camera_matrix_inv, camera_pose):
        """
        Computes 3D Rays originated from the camera origin frame and passing by each point in points_2d

        Parameters
        ------------
        points_2d: np.array of shape (N, 2)
                An array containing the 2D pixel coordinates of the target points where the rays should pass
        camera_matrix_inv: np.array(3,3)
                A matrix defined as the inverse of the camera intrinsics matrix K
        camera_pose: np.array(4,4)
                A matrix containing the pose of the camera wrt world frame

        Returns
        ------------
        centers, dirs: np.array
        """

        num_points = points_2d.shape[0]

        # Append 1 to each 2D point to make them homogeneous coordinates
        points_2d_hom = np.hstack((points_2d, np.ones((num_points, 1)))).T  # Shape (3, N)

        # Compute rays in camera frame
        rays = np.dot(camera_matrix_inv, points_2d_hom)  # Shape (3, N)
        rays /= np.linalg.norm(rays, axis=0)  # Normalize each ray

        # Convert rays to homogeneous coordinates for transformation
        rays_hom = np.vstack((rays, np.ones((1, num_points))))  # Shape (4, N)

        # Transform rays to world frame
        rays_world_hom = np.dot(camera_pose, rays_hom)  # Shape (4, N)
        rays_world = rays_world_hom[:3, :]  # Only take the first three rows, shape (3, N)

        # Camera center in world coordinates
        center = camera_pose[:3, 3].reshape(3, 1)  # Shape (3, 1)

        # Calculate ray directions
        ray_dirs = rays_world - center  # Shape (3, N)
        ray_dirs /= np.linalg.norm(ray_dirs, axis=0)  # Normalize each direction vector

        centers = np.tile(center, (1, num_points)).T  # Shape (N, 3)
        ray_dirs = ray_dirs.T  # Shape (N, 3)
        return centers, ray_dirs
