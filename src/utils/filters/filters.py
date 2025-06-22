import cv2
import sys
import numpy as np

sys.path.append('./')

from src.config import max_solidity
from src.utils.filters.detection_base import BaseFilter, DetectionObject

class AreaFilter(BaseFilter):
    """Filter objects based on minimum area threshold."""
    def __init__(self, min_area_pixels):
        self.min_area_pixels = min_area_pixels
    
    def check(self, obj: DetectionObject, context: dict) -> bool:
        return obj.area >= self.min_area_pixels

class IrregularityFilter(BaseFilter):
    """Filter objects by shape solidity to detect irregular contours."""
    def __init__(self, max_solidity=max_solidity):
        self.max_solidity = max_solidity
        
    def check(self, obj: DetectionObject, context: dict) -> bool:
        hull = cv2.convexHull(obj.contour)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            return False
        solidity = obj.area / float(hull_area)
        return solidity < self.max_solidity

class ShapeUnstabilityFilter(BaseFilter):
    """Filter based on shape changes between frames to detect unstable objects."""
    def __init__(self, shape_change_threshold=0.2, max_distance=50):
        self.shape_change_threshold = shape_change_threshold
        self.max_distance = max_distance
        self.previous_objects = {}
        self.next_obj_id = 0

    def check(self, obj: DetectionObject, context: dict) -> bool:
        is_unstable = True
        matched_id = None
        min_dist = self.max_distance

        for obj_id, prev_contour in self.previous_objects.items():
            if cv2.moments(prev_contour)['m00'] == 0: continue
            prev_centroid_x = int(cv2.moments(prev_contour)['m10'] / cv2.moments(prev_contour)['m00'])
            prev_centroid_y = int(cv2.moments(prev_contour)['m01'] / cv2.moments(prev_contour)['m00'])
            dist = np.sqrt((prev_centroid_x - obj.centroid[0])**2 + (prev_centroid_y - obj.centroid[1])**2)
            
            if dist < min_dist:
                min_dist = dist
                matched_id = obj_id

        if matched_id is not None:
            prev_contour = self.previous_objects[matched_id]
            score = cv2.matchShapes(prev_contour, obj.contour, cv2.CONTOURS_MATCH_I1, 0.0)
            if score < self.shape_change_threshold:
                is_unstable = False

        context['current_matches'][obj.centroid] = is_unstable
        return is_unstable

    def update_previous_frame(self, current_contours):
        self.previous_objects = {i: cnt for i, cnt in enumerate(current_contours)}