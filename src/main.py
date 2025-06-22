import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 1. THE MODULAR FILTERING CLASSES (No changes in this section)
# ---------------------------------------------------------------------------

class DetectionObject:
    """A class to hold information about a detected object."""
    def __init__(self, contour):
        self.contour = contour
        self.area = cv2.contourArea(contour)
        self.x, self.y, self.w, self.h = cv2.boundingRect(contour)
        self.centroid = (self.x + self.w // 2, self.y + self.h // 2)
        
        # Status flags for each filter
        self.passed_filters = True
        self.is_fire = False
        self.label = ""

class BaseFilter:
    """Abstract base class for all detection filters."""
    def check(self, obj: DetectionObject, context: dict) -> bool:
        self.obj= obj
        self.context = context
        raise NotImplementedError("Each filter must implement the 'check' method.")

class AreaFilter(BaseFilter):
    def __init__(self, min_area_pixels):
        self.min_area_pixels = min_area_pixels
    
    def check(self, obj: DetectionObject, context: dict) -> bool:
        return obj.area >= self.min_area_pixels

class IrregularityFilter(BaseFilter):
    def __init__(self, max_solidity=0.85):
        self.max_solidity = max_solidity
        
    def check(self, obj: DetectionObject, context: dict) -> bool:
        hull = cv2.convexHull(obj.contour)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            return False
        solidity = obj.area / float(hull_area)
        return solidity < self.max_solidity

class ShapeUnstabilityFilter(BaseFilter):
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

# ---------------------------------------------------------------------------
# 2. THE MAIN DETECTOR CLASS
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# 3. MAIN EXECUTION LOOP
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    video_path = './data/1.MP4'
    cap = cv2.VideoCapture(video_path)
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    min_area_percentage = 0.85
    min_pixel_area = (frame_width * frame_height * min_area_percentage) / 100

    filters_to_use = [
        AreaFilter(min_area_pixels=min_pixel_area),
        IrregularityFilter(max_solidity=0.85),
        ShapeUnstabilityFilter(shape_change_threshold=0.85),
    ]
    
    detector = FireDetector(filters=filters_to_use)
    
    print("Starting analysis. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        original_frame = frame.copy()
        overlay = frame.copy()
        results = detector.detect(frame)
        
        for obj in results:
            color = None
            if obj.is_fire:
                color = (0, 0, 255)
                print("fire")
            elif obj.label == "Hot":
                color = (0, 255, 255)
                print("hot")

            if color is not None:
                cv2.drawContours(overlay, [obj.contour], -1, color, -1)
                cv2.putText(frame, obj.label, (obj.x, obj.y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        alpha = 0.8
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Resize frames to make them smaller (e.g., 50% of original size)
        scale_percent = 50
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)

        resized_original = cv2.resize(original_frame, dim)
        resized_output = cv2.resize(frame, dim)

        combined = cv2.hconcat([resized_original, resized_output])
        cv2.imshow('Original | Output', combined)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()