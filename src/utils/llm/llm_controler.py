import sys
sys.path.append('./')

from src.utils.llm.llm_handler import LLMHandler
from src.utils.llm.llm_prompt import prompt

class LLMController:
    def __init__(self, pdf_exporter):
        self.llm = LLMHandler()
        self.pdf = pdf_exporter
        self.llm_busy = False
        self.last_frame = None

    def process_frame(self, frame):
        """Store frame and send async LLM request"""
        self.last_frame = frame.copy()
        self.llm_busy = True
        self.llm.run_async(frame, prompt, callback=self.llm_callback)

    def llm_callback(self, answer):
        self.pdf.save_frame_and_text(self.last_frame, answer)
        self.llm_busy = False
