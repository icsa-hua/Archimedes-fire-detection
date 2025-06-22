import cv2
import sys

sys.path.append('./')

from src.config import *
from src.utils.filters.filters import *
from src.utils.rest.exporter import PDFOutput
from src.utils.rest.visualizer import Visualizer
from src.utils.core.detector import FireDetector
from src.utils.llm.llm_controler import LLMController

class Client:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        min_pixel_area = (width * height * min_area) / 100

        filters = [
            AreaFilter(min_pixel_area),
            IrregularityFilter(),
            ShapeUnstabilityFilter(),
        ]

        self.detector = FireDetector(filters=filters)
        self.visualizer = Visualizer()
        self.llm_controller = LLMController(PDFOutput())

    def run(self, llm=False, visualize=False):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            detections = self.detector.detect(frame)
            annotated_frame, fire_detected = self.visualizer.annotate_frame(frame, detections)

            if llm and fire_detected and not self.llm_controller.llm_busy:
                self.llm_controller.process_frame(annotated_frame)

            if visualize:
                if not self.visualizer.display_frames(frame, annotated_frame):
                    break

        self.cap.release()
        if visualize:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    video_path = './data/1.MP4'
    client = Client(video_path)
    client.run(llm=True, visualize=True)
