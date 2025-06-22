import cv2
import sys

sys.path.append('./')

from src.utils.filters.detection_base import DetectionObject
from src.utils.filters.filters import *

class FireDetector:
    def __init__(self, filters: list):
        self.filters = filters

    def detect(self, frame):
        context = {'frame': frame, 'current_matches': {}}
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        thresh_cleaned = cv2.dilate(thresh, kernel, iterations=2)
        
        contours, _ = cv2.findContours(thresh_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_objects = []
        for c in contours:
            obj = DetectionObject(c)
            
            for f in self.filters:
                if not f.check(obj, context):
                    obj.passed_filters = False
                    if isinstance(f, ShapeUnstabilityFilter):
                        obj.label = "Hot"
                    break
            
            if obj.passed_filters:
                obj.is_fire = True
                obj.label = "Fire"
            
            detected_objects.append(obj)
        
        valid_contours_for_update = [d.contour for d in detected_objects if d.area > 0]
        for f in self.filters:
            if hasattr(f, 'update_previous_frame'):
                f.update_previous_frame(valid_contours_for_update)

        return detected_objects