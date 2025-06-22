import sys
import gc
import torch
import base64

sys.path.append('./')

from langchain_community.chat_models import ChatOllama
from langchain.schema.messages import HumanMessage

class LLMHandler:
    def __init__(self, model_name: str = 'gemma3:12b', temperature: float = 1.0, max_tokens = None):
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

    def ask_with_image(self, image_path: str, question: str):
        llm = self.get_model()

        with open(image_path, "rb") as f:
            img_bytes = f.read()
        b64_img = base64.b64encode(img_bytes).decode("utf-8")

        message = HumanMessage(
            content=[
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}},
                {"type": "text", "text": question}
            ]
        )

        response = llm.invoke([message])
        return response.content

if __name__ == "__main__":

    handler = LLMHandler()
    
    try:
        result = handler.ask_with_image("a.png", "What is happening in this image?")
        print("Model answer:", result)
    except Exception as e:
        print("Error:", e)