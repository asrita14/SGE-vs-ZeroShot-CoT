import json
from typing import List, Dict, Any

from datasets import load_dataset
from src.compute_metrics import (
    compute_gsm8k_metrics,
    compute_boolq_metrics,
    compute_csqa_metrics,
)


def tag_difficulty_gsm8k(questions: List[str]) -> List[str]:
    tags = []
    for q in questions:
        length = len(q.split())
        # heuristic: short & simple wording = easy, else hard
        if length <= 40:
            tags.append("easy")
        else:
            tags.append("hard")
    return tags


def tag_difficulty_boolq(passages: List[str]) -> List[str]:
    tags = []
    for p in passages:
        length = len(p.split())
        if length <= 80:
            tags.append("easy")
        else:
            tags.append("hard")
    return tags


def tag_difficulty_csqa(questions: List[str]) -> List[str]:
    tags = []
    for q in questions:
        length = len(q.split())
        if length <= 10:
            tags.append("easy")
        else:
            tags.append("hard")
    return tags


def analyze_gsm8k(run_name: str, num_examples: int = 200):
    ds = load_dataset("gsm8k", "main")["test"].select(range(num_examples))
    questions = [ex["question"] for ex in ds]
    diffs = tag_difficulty_gsm8k(questions)

    with open(f"results/metrics/gsm8k_{run_name}.json") as f:
        data = json.load(f)
    results = data["results"]

    easy_results = []
    hard_results = []
    for diff, r in zip(diffs, results):
        if diff == "easy":
            easy_results.append(r)
        else:
            hard_results.append(r)

    print("=== GSM8K Difficulty Analysis ===")
    print(f"Run: {run_name}")
    print(f"Easy examples: {len(easy_results)}, Hard examples: {len(hard_results)}")

    print("\nEasy subset metrics:")
    easy_metrics = compute_gsm8k_metrics(easy_results)
    print(json.dumps(easy_metrics, indent=2))

    print("\nHard subset metrics:")
    hard_metrics = compute_gsm8k_metrics(hard_results)
    print(json.dumps(hard_metrics, indent=2))


def analyze_boolq(run_name: str, num_examples: int = 200):
    ds = load_dataset("boolq")["validation"].select(range(num_examples))
    passages = [ex["passage"] for ex in ds]
    diffs = tag_difficulty_boolq(passages)

    with open(f"results/metrics/boolq_{run_name}.json") as f:
        data = json.load(f)
    results = data["results"]

    easy_results = []
    hard_results = []
    for diff, r in zip(diffs, results):
        if diff == "easy":
            easy_results.append(r)
        else:
            hard_results.append(r)

    print("=== BoolQ Difficulty Analysis ===")
    print(f"Run: {run_name}")
    print(f"Easy examples: {len(easy_results)}, Hard examples: {len(hard_results)}")

    print("\nEasy subset metrics:")
    easy_metrics = compute_boolq_metrics(easy_results)
    print(json.dumps(easy_metrics, indent=2))

    print("\nHard subset metrics:")
    hard_metrics = compute_boolq_metrics(hard_results)
    print(json.dumps(hard_metrics, indent=2))


def analyze_csqa(run_name: str, num_examples: int = 200):
    ds = load_dataset("commonsense_qa")["validation"].select(range(num_examples))
    questions = [ex["question"] for ex in ds]
    diffs = tag_difficulty_csqa(questions)

    with open(f"results/metrics/csqa_{run_name}.json") as f:
        data = json.load(f)
    results = data["results"]

    easy_results = []
    hard_results = []
    for diff, r in zip(diffs, results):
        if diff == "easy":
            easy_results.append(r)
        else:
            hard_results.append(r)

    print("=== CSQA Difficulty Analysis ===")
    print(f"Run: {run_name}")
    print(f"Easy examples: {len(easy_results)}, Hard examples: {len(hard_results)}")

    print("\nEasy subset metrics:")
    easy_metrics = compute_csqa_metrics(easy_results)
    print(json.dumps(easy_metrics, indent=2))

    print("\nHard subset metrics:")
    hard_metrics = compute_csqa_metrics(hard_results)
    print(json.dumps(hard_metrics, indent=2))


if __name__ == "__main__":
    analyze_gsm8k(run_name="gsm200_filtered_sc", num_examples=200)
    analyze_boolq(run_name="bool200_filtered_sc", num_examples=200)
    analyze_csqa(run_name="csqa200_filtered_sc", num_examples=200)


