import sys
sys.path.append('/')

from src.llm.llm_comparator import HumanComparator

class Pipeline:
    """
    Example pipeline to compare real and generated humans using ontology labels.
    """

    def __init__(self):
        self.comparator = HumanComparator()

    def run(self, real_labels: dict, generated_labels: dict):
        """
        Run the comparison and return the LLM result.
        """
        return self.comparator.compare(real_labels, generated_labels)


if __name__ == "__main__":
    real_human_labels = {
        "skin_color": "black",
        "glasses": True,
        "hair_length": "short",
        "hat": False
    }

    generated_human_labels = {
        "skin_color": "black",
        "glasses": True,
        "hair_length": "medium",
        "hat": False
    }

    pipeline = Pipeline()
    result = pipeline.run(real_human_labels, generated_human_labels)
    print(result)
