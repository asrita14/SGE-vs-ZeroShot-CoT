"""
Evaluate Zero-Shot, Few-Shot, CoT, and SGE on GSM8K, BoolQ, CSQA.
"""

import json
from datasets import load_dataset
from prompting.baselines import PromptingBaselines
from prompting.sge_pipeline import SelfGeneratedExamples
from tqdm import tqdm

def evaluate():
    model_name = "google/flan-t5-base"
    baseline = PromptingBaselines(model_name)
    sge = SelfGeneratedExamples(model_name)

    gsm = load_dataset("gsm8k", "main")["test"].select(range(30))  # small subset

    results = []

    for sample in tqdm(gsm, desc="Evaluating GSM8K"):
        q = sample["question"]
        a = sample["answer"]

        zs = baseline.zero_shot(q)
        cot = baseline.cot(q)
        sge_ans = sge.infer_with_sge(q, "Solve grade-school math word problems.")

        results.append({
            "question": q,
            "gold": a,
            "zero_shot": zs,
            "cot": cot,
            "sge": sge_ans
        })

    with open("results/metrics/gsm8k_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Saved results → results/metrics/gsm8k_results.json")


if __name__ == "__main__":
    evaluate()
