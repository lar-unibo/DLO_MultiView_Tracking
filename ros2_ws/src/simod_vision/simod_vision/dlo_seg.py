import numpy as np
import cv2
from dlo_python.dlo_segmentation.main import DloSegmentationNetwork


def bounding_box(points):
    x_coordinates, y_coordinates = zip(*points)
    return [(min(x_coordinates), min(y_coordinates)), (max(x_coordinates), max(y_coordinates))]


def crop_around_bbox(image, bbox, target_aspect_ratio, offset):
    x_min, y_min, x_max, y_max = bbox

    # Expand the bounding box by the given offset
    x_min = max(0, x_min - offset)
    y_min = max(0, y_min - offset)
    x_max = min(image.shape[1], x_max + offset)
    y_max = min(image.shape[0], y_max + offset)

    crop_width = x_max - x_min
    crop_height = y_max - y_min
    current_aspect_ratio = crop_width / crop_height

    if current_aspect_ratio > target_aspect_ratio:
        # The bbox is wider than the target aspect ratio, so we need to adjust height
        new_height = int(crop_width / target_aspect_ratio)
        y_min = max(0, y_min - (new_height - crop_height) // 2)
        y_max = y_min + new_height
    else:
        # The bbox is taller than the target aspect ratio, so we need to adjust width
        new_width = int(crop_height * target_aspect_ratio)
        x_min = max(0, x_min - (new_width - crop_width) // 2)
        x_max = x_min + new_width

    # Crop the image
    y_min = int(y_min)
    y_max = int(y_max)
    x_min = int(x_min)
    x_max = int(x_max)
    cropped_image = image[y_min:y_max, x_min:x_max]

    return cropped_image, (x_min, y_min)


class DloSeg:

    def __init__(self, checkpoint_path, show=False) -> None:
        self.seg = DloSegmentationNetwork(checkpoint_path)

        self.show = show
        if self.show:
            cv2.namedWindow("crop_top", cv2.WINDOW_NORMAL)
            cv2.namedWindow("crop_side", cv2.WINDOW_NORMAL)
            cv2.namedWindow("mask_top", cv2.WINDOW_NORMAL)
            cv2.namedWindow("mask_side", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("crop_top", 800, 500)
            cv2.resizeWindow("crop_side", 800, 500)
            cv2.resizeWindow("mask_top", 800, 500)
            cv2.resizeWindow("mask_side", 800, 500)

    def predict(self, img_side, img_top):
        mask_side, mask_top = self.seg.predict_batch([img_side, img_top])
        return (mask_side * 255).astype(np.uint8), (mask_top * 255).astype(np.uint8)

    def masks_from_model_points(self, points_side, points_top, img_shape):
        mask_model_side = np.zeros(img_shape, dtype=np.uint8)
        mask_model_top = np.zeros(img_shape, dtype=np.uint8)
        for point in points_side:
            mask_model_side = cv2.circle(mask_model_side, (int(point[0]), int(point[1])), 30, 255, -1)
        mask_model_side = cv2.dilate(mask_model_side, np.ones((5, 5), np.uint8), iterations=3)
        for point in points_top:
            mask_model_top = cv2.circle(mask_model_top, (int(point[0]), int(point[1])), 30, 255, -1)
        mask_model_top = cv2.dilate(mask_model_top, np.ones((5, 5), np.uint8), iterations=3)

        return mask_model_side, mask_model_top

    def crop_predict_and_filter_from_points(self, img_side, img_top, points_side, points_top, bbox_side, bbox_top):

        # crop around the bounding box
        img_crop_side, side_offset = crop_around_bbox(img_side, bbox_side, img_side.shape[1] / img_side.shape[0], 50)
        img_crop_top, top_offset = crop_around_bbox(img_top, bbox_top, img_top.shape[1] / img_top.shape[0], 50)

        # predict the masks by the segmentation model
        mask_crop_side, mask_crop_top = self.predict(img_crop_side, img_crop_top)

        # restore the masks to the original size from the cropped size
        mask_crop_side = cv2.resize(mask_crop_side, (img_crop_side.shape[1], img_crop_side.shape[0]))
        mask_crop_top = cv2.resize(mask_crop_top, (img_crop_top.shape[1], img_crop_top.shape[0]))

        mask_side = np.zeros(img_side.shape[:2], dtype=np.uint8)
        mask_top = np.zeros(img_top.shape[:2], dtype=np.uint8)

        mask_side[
            side_offset[1] : side_offset[1] + mask_crop_side.shape[0],
            side_offset[0] : side_offset[0] + mask_crop_side.shape[1],
        ] = mask_crop_side
        mask_top[
            top_offset[1] : top_offset[1] + mask_crop_top.shape[0],
            top_offset[0] : top_offset[0] + mask_crop_top.shape[1],
        ] = mask_crop_top

        # generate the masks from the dlo model points
        mask_model_side, mask_model_top = self.masks_from_model_points(points_side, points_top, img_side.shape[:2])

        if self.show:
            mask_side = cv2.cvtColor(mask_side, cv2.COLOR_GRAY2BGR)
            mask_top = cv2.cvtColor(mask_top, cv2.COLOR_GRAY2BGR)

            mask_side = cv2.addWeighted(mask_side, 1.0, cv2.cvtColor(mask_model_side, cv2.COLOR_GRAY2BGR), 0.5, 0)
            mask_top = cv2.addWeighted(mask_top, 1.0, cv2.cvtColor(mask_model_top, cv2.COLOR_GRAY2BGR), 0.5, 0)

            for point in points_side:
                mask_side = cv2.circle(mask_side, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)
            for point in points_top:
                mask_top = cv2.circle(mask_top, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)

            cv2.imshow("crop_top", img_top)
            cv2.imshow("crop_side", img_side)
            cv2.imshow("mask_top", mask_top)
            cv2.imshow("mask_side", mask_side)
            if cv2.waitKey(30) == ord("q"):
                quit()

        return mask_side, mask_top, mask_model_side, mask_model_top

    def single_crop_predict(self, img, bbox):

        # crop around the bounding box
        img_crop, offset = crop_around_bbox(img, bbox, img.shape[1] / img.shape[0], 50)

        # predict the masks by the segmentation model
        mask_crop = self.seg.predict(img_crop)
        mask_crop = (mask_crop * 255).astype(np.uint8)

        mask_crop_proc = self.threshold_and_small_blob_removal(mask_crop, th=127, min_size=100)

        # restore the masks to the original size from the cropped size
        mask_crop_proc = cv2.resize(mask_crop_proc, (img_crop.shape[1], img_crop.shape[0]))

        mask_out = np.zeros(img.shape[:2], dtype=np.uint8)
        x1, y1 = offset
        mask_out[y1 : y1 + mask_crop_proc.shape[0], x1 : x1 + mask_crop_proc.shape[1]] = mask_crop_proc
        return mask_out, img_crop

    def single_mask_points(self, points, img_shape):
        mask_model = np.zeros(img_shape, dtype=np.uint8)
        for point in points:
            mask_model = cv2.circle(mask_model, (int(point[0]), int(point[1])), 30, 255, -1)
        mask_model = cv2.dilate(mask_model, np.ones((5, 5), np.uint8), iterations=3)

        return mask_model

    def threshold_and_small_blob_removal(self, mask, th=127, min_size=100):
        mask2 = mask.copy()
        mask2[mask > th] = 255
        mask2[mask <= th] = 0

        # remove small blobs
        nb_components, output, stats, _ = cv2.connectedComponentsWithStats(mask2, connectivity=4)
        sizes = stats[1:, -1]
        nb_components = nb_components - 1
        mask2 = np.zeros((output.shape), dtype=np.uint8)
        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                mask2[output == i + 1] = 255

        return mask2
