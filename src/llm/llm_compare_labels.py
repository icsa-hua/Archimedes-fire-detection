# compare_labels.py
from llm_handler import LLMHandler

class HumanComparator:
    def __init__(self):
        self.llm = LLMHandler()

    def compare_humans(self, real_labels: dict, generated_labels: dict):
        """
        real_labels: dict of ontology labels for the real human
        generated_labels: dict of ontology labels for generated human
        """
        prompt = f"""
        You are an AI that compares two humans based on their ontology labels.
        Real human labels: {real_labels}
        Generated human labels: {generated_labels}

        Compare the two sets of labels and answer the following:

        1. For each label, mention if they match or not.
        2. Provide a **similarity score** between 0 and 100, where 100 means they look identical.
        3. Consider "intuition": if they are similar overall even if some labels differ, explain why.

        Output should be a clear JSON with:
        {{
            "label_matches": {{}},
            "similarity_score": 0,
            "comment": ""
        }}
        """
        response = self.llm.ask(prompt)
        return response

if __name__ == "__main__":

    real = {"skin_color": "black", "glasses": True, "hair_length": "short", "hat": False}
    generated = {"skin_color": "black", "glasses": True, "hair_length": "medium", "hat": False}

    comparator = HumanComparator()
    result = comparator.compare_humans(real, generated)
    print("LLM Comparison Result:", result)
