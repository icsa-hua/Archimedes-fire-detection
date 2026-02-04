from llm_handler import LLMHandler

class HumanComparator:
    """
    Compares two humans based on ontology labels using an LLM.
    """

    def __init__(self, model_name: str = 'ollama3:12b', temperature: float = 0.7, max_tokens=None):
        self.llm = LLMHandler(model_name=model_name, temperature=temperature, max_tokens=max_tokens)

    def compare(self, real_labels: dict, generated_labels: dict):
        """
        Compare real and generated human labels and return LLM analysis.
        """
        prompt = f"""
You are an AI that compares two humans based on ontology labels.
Real human labels: {real_labels}
Generated human labels: {generated_labels}

Compare the labels and output a JSON with:
{{
    "label_matches": {{}},
    "similarity_score": 0,
    "comment": ""
}}
"""
        return self.llm.ask(prompt)
