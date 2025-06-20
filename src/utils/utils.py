import cv2
import sys
import numpy as np

sys.path.append('./')

def resize_image_to_height(image: np.ndarray, target_height: int) -> np.ndarray:
    """
    Resize an image to a specific height while maintaining aspect ratio.

    Parameters:
        image (np.ndarray): Input image.
        target_height (int): Desired height.

    Returns:
        np.ndarray: Resized image.
    """
    scale_factor = target_height / image.shape[0]
    target_width = int(image.shape[1] * scale_factor)
    return cv2.resize(image, (target_width, target_height))


def calculate_hu_moments(contour: np.ndarray) -> np.ndarray:
    """
    Compute the log-transformed Hu Moments of a contour.

    Parameters:
        contour (np.ndarray): Contour points.

    Returns:
        np.ndarray: Array of Hu Moments.
    """
    moments = cv2.moments(contour)
    hu_moments = cv2.HuMoments(moments).flatten()
    return -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)


def perform_non_maximum_suppression(boxes: list[list[int]], overlap_threshold: float = 0.3) -> list[list[int]]:
    """
    Apply non-maximum suppression to bounding boxes.

    Parameters:
        boxes (list[list[int]]): List of bounding boxes [x1, y1, x2, y2].
        overlap_threshold (float): Threshold for overlapping boxes.

    Returns:
        list[list[int]]: Filtered bounding boxes after suppression.
    """
    if not boxes:
        return []

    boxes_array = np.array(boxes)
    x1, y1, x2, y2 = boxes_array[:, 0], boxes_array[:, 1], boxes_array[:, 2], boxes_array[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    sorted_indices = np.argsort(y2)
    selected_indices = []

    while sorted_indices.size > 0:
        last_index = sorted_indices[-1]
        selected_indices.append(last_index)
        sorted_indices = sorted_indices[:-1]

        xx1 = np.maximum(x1[last_index], x1[sorted_indices])
        yy1 = np.maximum(y1[last_index], y1[sorted_indices])
        xx2 = np.minimum(x2[last_index], x2[sorted_indices])
        yy2 = np.minimum(y2[last_index], y2[sorted_indices])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / areas[sorted_indices]

        sorted_indices = sorted_indices[overlap <= overlap_threshold]

    return boxes_array[selected_indices].astype(int).tolist()
