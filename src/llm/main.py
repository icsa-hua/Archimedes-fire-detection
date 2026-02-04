import sys
sys.path.append('./')

from src.llm.llm_compare_labels import HumanComparator

def main():
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

    comparator = HumanComparator()
    comparison_result = comparator.compare_humans(real_human_labels, generated_human_labels)

    print("Comparison Result:")
    print(comparison_result)

if __name__ == "__main__":
    main()
