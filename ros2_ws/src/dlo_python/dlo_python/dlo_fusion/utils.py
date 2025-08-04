import cv2
import numpy as np
from scipy.interpolate import splprep, splev


def points_association_and_correction(points1, points2):
    # Calculate the distance between each point in points1 and all points in points2
    distances = np.linalg.norm(points1[:, np.newaxis, :] - points2, axis=2)

    # Find the indices of the minimum distances
    min_indices = distances.argmin(axis=1)

    # Use these indices to get the corresponding points from points2
    return points2[min_indices]


def compute_spline(points, num_points=10, k=3, s=0):
    points = np.array(points).squeeze()
    tck, u = splprep(points.T, u=None, k=k, s=s, per=0)
    u_new = np.linspace(u.min(), u.max(), num_points)

    if points.shape[1] == 2:
        x_, y_ = splev(u_new, tck, der=0)
        return np.vstack([x_, y_]).T
    else:
        x_, y_, z_ = splev(u_new, tck, der=0)
        return np.vstack([x_, y_, z_]).T


def project_points(points, pose, intrinsic, img_w, img_h):
    rvec = cv2.Rodrigues(pose[:3, :3])[0]
    tvec = pose[:3, 3]
    points2d = cv2.projectPoints(points, rvec, tvec, intrinsic, None)[0]
    points2d = points2d.squeeze()

    # check if the points are in the image
    mask = (points2d[:, 0] > 0) & (points2d[:, 0] < img_w) & (points2d[:, 1] > 0) & (points2d[:, 1] < img_h)
    points2d = points2d[mask]
    return points2d


def convert_points_camera_frame(points, pose):
    points = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
    return np.dot(pose, points.T).T[:, :3]


def draw_filled_polyline_to_mask(mask, points):
    cv2.fillPoly(mask, [points.astype(int)], color=255)


def compute_confidence_mask(dlo_points_3d, camera_pose, mask_w, mask_h, intrinsics, volume_size=0.05):
    # transform the points to the camera frame to compute correct volume size according to cartesian coordinates
    world_to_camera1 = np.linalg.inv(camera_pose)
    dlo_cam = convert_points_camera_frame(dlo_points_3d, world_to_camera1)
    dlo_cam_distance = np.mean(dlo_cam[:, 2])
    volume_size_pixel = intrinsics[0, 0] * volume_size / dlo_cam_distance

    # project 3d points to 2d
    points1 = project_points(dlo_points_3d, np.linalg.inv(camera_pose), intrinsics, mask_w, mask_h)

    # compute confidence mask around the projected points
    confidence_mask = np.zeros((mask_h, mask_w))
    points_left = []
    points_right = []

    # circle top
    circle_top = []
    for x in np.linspace(0, 2 * np.pi, 20):
        point = np.array([np.cos(x), np.sin(x)]) * volume_size_pixel + points1[0]
        circle_top.append(point)
    circle_top = np.array(circle_top)

    # circle bottom
    circle_bottom = []
    for x in np.linspace(0, 2 * np.pi, 20):
        point = np.array([np.cos(x), np.sin(x)]) * volume_size_pixel + points1[-1]
        circle_bottom.append(point)
    circle_bottom = np.array(circle_bottom)

    for i in range(1, points1.shape[0] - 1):
        point0 = points1[i - 1]
        point1 = points1[i]
        dir = point1 - point0
        dir = dir / np.linalg.norm(dir)

        # compute the perpendicular direction
        perp_dir = np.array([-dir[1], dir[0]])
        perp_dir = perp_dir / np.linalg.norm(perp_dir)

        # compute the point on the left and right
        point_left = point1 + perp_dir * volume_size_pixel
        point_right = point1 - perp_dir * volume_size_pixel

        points_left.append(point_left)
        points_right.append(point_right)

    points_left = np.array(points_left)
    points_right = np.array(points_right)
    points = np.concatenate((points_left, points_right[::-1]), axis=0)

    # draw the confidence mask
    draw_filled_polyline_to_mask(confidence_mask, points)
    draw_filled_polyline_to_mask(confidence_mask, circle_top)
    draw_filled_polyline_to_mask(confidence_mask, circle_bottom)

    return confidence_mask
