import sys

sys.path.append('./')

from langchain_community.chat_models import ChatOllama

class LLMHandler:
    """
    Handles loading and interacting with an Ollama LLM model.
    """

    def __init__(self, model_name: str = 'ollama3:12b', temperature: float = 0.7, max_tokens=None):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm = None

    def _load_model(self):
        """
        Load the Ollama model if not already loaded.
        """
        if not self.llm:
            self.llm = ChatOllama(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
        return self.llm

    def get_model(self):
        """
        Return the loaded model.
        """
        return self._load_model()

    def ask(self, prompt: str):
        """
        Send a prompt to the LLM and return its text response.
        """
        llm = self.get_model()
        response = llm.invoke(prompt)
        return response.content
