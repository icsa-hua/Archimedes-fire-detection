import cv2
import sys

sys.path.append('./')

class DetectionObject:
    """Holds properties and status of a detected contour object."""
    def __init__(self, contour):
        self.contour = contour
        self.area = cv2.contourArea(contour)
        self.x, self.y, self.w, self.h = cv2.boundingRect(contour)
        self.centroid = (self.x + self.w // 2, self.y + self.h // 2)
        
        self.passed_filters = True
        self.is_fire = False
        self.label = ""

class BaseFilter:
    """Abstract base class for all detection filters."""
    def check(self, obj: DetectionObject, context: dict) -> bool:
        self.obj= obj
        self.context = context
        raise NotImplementedError("Each filter must implement the 'check' method.")