
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
