import cv2
import sys

sys.path.append('./')

from src.config import BGR_RED, BGR_ORANGE

class Visualizer:
    def __init__(self, scale_percent=50, alpha=0.7):
        self.scale_percent = scale_percent
        self.alpha = alpha

    def annotate_frame(self, frame, detections):
        overlay = frame.copy()
        fire_detected = False

        for obj in detections:
            color = None
            if obj.is_fire:
                fire_detected = True
                color = BGR_RED
            elif obj.label == "Hot":
                color = BGR_ORANGE

            if color:
                cv2.drawContours(overlay, [obj.contour], -1, color, -1)

        annotated = cv2.addWeighted(overlay, self.alpha, frame, 1 - self.alpha, 0)
        return annotated, fire_detected

    def display_frames(self, original, annotated):
        width = int(original.shape[1] * self.scale_percent / 100)
        height = int(original.shape[0] * self.scale_percent / 100)
        dim = (width, height)

        resized_original = cv2.resize(original, dim)
        resized_annotated = cv2.resize(annotated, dim)

        combined = cv2.hconcat([resized_original, resized_annotated])
        cv2.imshow('Original | Output', combined)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            return False
        return True
