import cv2
import sys
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import io
from reportlab.platypus import Paragraph, Frame
from reportlab.lib.styles import getSampleStyleSheet
sys.path.append('./')

from src.config import *

class PDFOutput:
    def __init__(self, output_pdf_path="./outputs/llm_output.pdf"):
        self.output_pdf_path = output_pdf_path
        self.saved = False  

    def save_frame_and_text(self, frame, text):
        if self.saved:
            return  

        is_success, buffer = cv2.imencode(".jpg", frame)
        if not is_success:
            print("Failed to encode frame for PDF.")
            return
        image_bytes = io.BytesIO(buffer.tobytes())

        c = canvas.Canvas(self.output_pdf_path, pagesize=letter)
        width, height = letter

        c.setFont("Helvetica-Bold", 16)
        c.drawString(72, height - 72, "LLM Fire Detection Output")

        img = ImageReader(image_bytes)
        img_width = width - 144  
        img_height = img_width * frame.shape[0] / frame.shape[1]
        img_y = height - 72 - 16 - img_height - 10
        c.drawImage(img, 72, img_y, width=img_width, height=img_height)

        styles = getSampleStyleSheet()
        styleN = styles['Normal']

        para = Paragraph(text.replace('\n', '<br/>'), styleN)

        text_frame_y = img_y - 100 
        frame_height = text_frame_y - 72
        frame = Frame(72, 72, width - 144, frame_height, showBoundary=0)

        frame.addFromList([para], c)

        c.save()
        print(f"Saved LLM output and frame to PDF: {self.output_pdf_path}")
        self.saved = True