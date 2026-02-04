import sys

sys.path.append('./')

import gc
import torch
from langchain_community.chat_models import ChatOllama
from langchain.schema.messages import HumanMessage

class LLMHandler:
    def __init__(self, model_name: str = 'ollama3:12b', temperature: float = 0.7, max_tokens=None):
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

    def get_model(self):
        return self._load_model()

    def clear_cache(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

    def ask(self, prompt: str):
        llm = self.get_model()
        message = HumanMessage(content=prompt)
        response = llm.invoke([message])
        return response.content
