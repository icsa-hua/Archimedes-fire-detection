import sys
import gc
import cv2
import torch
import base64
import numpy as np
import threading

from langchain_community.chat_models import ChatOllama
from langchain.schema.messages import HumanMessage

sys.path.append('./')

from src.config import model, temperature, max_tokens

class LLMHandler:
    def __init__(self, 
                 model_name: str = model, 
                 temperature: float = temperature, 
                 max_tokens: int = max_tokens):
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm = None

    def _load_model(self):
        if not self.llm:
            self.llm = ChatOllama(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
        return self.llm

    def get_model(self) -> ChatOllama:
        return self._load_model()

    def clear_cache(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

    def run(self, frame: np.ndarray, question: str):
        llm = self.get_model()

        success, encoded_image = cv2.imencode('.png', frame)
        if not success:
            raise RuntimeError("Failed to encode frame to PNG")

        img_bytes = encoded_image.tobytes()
        b64_img = base64.b64encode(img_bytes).decode("utf-8")

        message = HumanMessage(
            content=[
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}},
                {"type": "text", "text": question}
            ]
        )

        response = llm.invoke([message])
        return response.content

    def run_async(self, frame: np.ndarray, question: str, callback=None):
        """
        Run the LLM inference in a separate thread.
        If a callback is provided, it will be called with the result when ready.
        """
        def worker():
            try:
                result = self.run(frame, question)
                if callback:
                    callback(result)
            except Exception as e:
                if callback:
                    callback(f"LLM error: {e}")

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()


if __name__ == "__main__":
    
    handler = LLMHandler()
 
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    question = "What's in the image, blah blah"

    try:
        result = handler.run(dummy_frame, question)
        print("Model answer:", result)
    except Exception as e:
        print("Error:", e)

    def print_result(res):
        print("Async LLM insight:", res)

    handler.run_async(dummy_frame, question, callback=print_result)
