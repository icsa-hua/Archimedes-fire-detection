import cv2
import sys

sys.path.append('./')

from src.utils.filters.filters import AreaFilter, IrregularityFilter, ShapeUnstabilityFilter
from src.utils.llm.llm_handler import LLMHandler
from src.utils.core.detector import FireDetector
from src.utils.rest.pdf_exporter import PDFOutput
from src.utils.llm.llm_prompt import prompt

from src.config import *

class Client:
    def __init__(self, video_path):
        """ Main Client """
        self.cap = cv2.VideoCapture(video_path)

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.min_pixel_area = (self.frame_width * self.frame_height * min_area) / 100

        self.filters = [
            AreaFilter(min_area_pixels=self.min_pixel_area),
            IrregularityFilter(max_solidity),
            ShapeUnstabilityFilter(shape_change_threshold),
        ]
        self.detector = FireDetector(filters=self.filters)

        self.llm = LLMHandler()
        self.llm_busy = False  
        self.pdf = PDFOutput()

    def llm_callback(self, answer):
        self.pdf.save_frame_and_text(self.last_frame, answer)

    def run(self, llm=False, visualize=False):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            original_frame = frame.copy()
            overlay = frame.copy()

            results = self.detector.detect(frame)

            fire_detected = False
            for obj in results:
                color = None
                if obj.is_fire:
                    fire_detected = True
                    color = BGR_RED
                elif obj.label == "Hot":
                    color = BGR_ORANGE

                if color is not None:
                    cv2.drawContours(overlay, [obj.contour], -1, color, -1)

            alpha = 0.7
            annotated_frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            if fire_detected and llm and not self.llm_busy:
                question = prompt
                self.llm_busy = True
                self.last_frame = annotated_frame.copy()
                self.llm.run_async(frame, question, callback=self.llm_callback)

            if visualize:
                scale_percent = 50
                width = int(frame.shape[1] * scale_percent / 100)
                height = int(frame.shape[0] * scale_percent / 100)
                dim = (width, height)

                resized_original = cv2.resize(original_frame, dim)
                resized_output = cv2.resize(annotated_frame, dim)

                combined = cv2.hconcat([resized_original, resized_output])
                cv2.imshow('Original | Output', combined)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

        self.cap.release()
        if visualize:
            cv2.destroyAllWindows()


if __name__ == '__main__':

    video_path = './data/1.MP4'
    client = Client(video_path)
    client.run(llm=True, visualize=True)
