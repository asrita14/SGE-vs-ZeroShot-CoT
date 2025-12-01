"""
Evaluate Zero-Shot, Few-Shot, CoT, and SGE on:
- GSM8K (math word problems)
- BoolQ (yes/no QA)
- CommonsenseQA (multiple choice)
"""

import json
import argparse
from datasets import load_dataset
from tqdm import tqdm

from src.prompting.baselines import PromptingBaselines
from src.prompting.sge_pipeline import SelfGeneratedExamples


def evaluate_gsm8k(baseline, sge, num_examples=30):
    """Evaluate all prompting strategies on GSM8K."""
    gsm = load_dataset("gsm8k", "main")["test"].select(range(num_examples))

    # Few-shot examples for GSM8K
    few_shot_examples = [
        ("Tom has 8 apples and eats 3. How many left?", "5"),
        ("What is 12 + 7?", "19"),
        ("A box has 10 candies and 4 more are added. Total?", "14"),
    ]

    results = []

    for sample in tqdm(gsm, desc="Evaluating GSM8K"):
        q = sample["question"]
        a = sample["answer"]

        zs = baseline.zero_shot(q)
        fs = baseline.few_shot(q, few_shot_examples)
        cot = baseline.cot(q)
        sge_ans = sge.infer_with_sge(q, "Solve grade-school math word problems.")

        results.append({
            "dataset": "gsm8k",
            "question": q,
            "gold": a,
            "zero_shot": zs,
            "few_shot": fs,
            "cot": cot,
            "sge": sge_ans,
        })

    out_path = "results/metrics/gsm8k_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved GSM8K results → {out_path}")


def evaluate_boolq(baseline, sge, num_examples=30):
    """Evaluate all prompting strategies on BoolQ (yes/no QA)."""
    boolq = load_dataset("boolq")["validation"].select(range(num_examples))

    # Few-shot examples for BoolQ
    # We structure as (passage + question, answer)
    few_shot_examples = [
        (
            "Passage: The Earth orbits the Sun.\nQuestion: Is the Earth a planet?\nAnswer yes or no.",
            "yes",
        ),
        (
            "Passage: Penguins are birds that cannot fly but can swim very well.\n"
            "Question: Can penguins fly?\nAnswer yes or no.",
            "no",
        ),
        (
            "Passage: Water freezes at 0 degrees Celsius.\nQuestion: Does water freeze at 0°C?\nAnswer yes or no.",
            "yes",
        ),
    ]

    results = []

    for sample in tqdm(boolq, desc="Evaluating BoolQ"):
        question = sample["question"]
        passage = sample["passage"]
        gold_bool = sample["answer"]  # True/False
        gold = "yes" if gold_bool else "no"

        # We include passage + question in the prompt
        q_text = (
            f"Passage: {passage}\n"
            f"Question: {question}\n"
            "Answer yes or no."
        )

        zs = baseline.zero_shot(q_text)
        fs = baseline.few_shot(q_text, few_shot_examples)
        cot = baseline.cot(q_text)
        sge_ans = sge.infer_with_sge(
            q_text,
            "Read the passage and answer the yes/no question."
        )

        results.append({
            "dataset": "boolq",
            "passage": passage,
            "question": question,
            "gold": gold,
            "zero_shot": zs,
            "few_shot": fs,
            "cot": cot,
            "sge": sge_ans,
        })

    out_path = "results/metrics/boolq_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved BoolQ results → {out_path}")


def evaluate_csqa(baseline, sge, num_examples=30):
    """Evaluate all prompting strategies on CommonsenseQA (multiple choice)."""
    csqa = load_dataset("commonsense_qa")["validation"].select(range(num_examples))

    # Few-shot examples for CSQA
    # We use letter options A/B/C/D/E
    few_shot_examples = [
        (
            "Question: Where would you store milk to keep it cold?\n"
            "A) in a drawer\nB) in the refrigerator\nC) on the table\n"
            "D) in the oven\nE) in your pocket\n"
            "Answer with the correct option letter.",
            "B",
        ),
        (
            "Question: What do people use to call someone far away?\n"
            "A) telephone\nB) spoon\nC) pillow\nD) plate\nE) blanket\n"
            "Answer with the correct option letter.",
            "A",
        ),
        (
            "Question: Where do you usually see clouds?\n"
            "A) in the basement\nB) under the bed\nC) in the sky\n"
            "D) in the fridge\nE) in your shoes\n"
            "Answer with the correct option letter.",
            "C",
        ),
    ]

    results = []

    for sample in tqdm(csqa, desc="Evaluating CommonsenseQA"):
        question = sample["question"]
        choices = sample["choices"]["text"]   # list of answer texts
        labels = sample["choices"]["label"]   # list of labels like ["A", "B", ...]
        gold_label = sample["answerKey"]      # correct label like "C"

        # Build multiple-choice prompt
        options_str_lines = []
        for label, choice_text in zip(labels, choices):
            options_str_lines.append(f"{label}) {choice_text}")
        options_str = "\n".join(options_str_lines)

        q_text = (
            f"Question: {question}\n"
            f"{options_str}\n"
            "Answer with the correct option letter."
        )

        zs = baseline.zero_shot(q_text)
        fs = baseline.few_shot(q_text, few_shot_examples)
        cot = baseline.cot(q_text)
        sge_ans = sge.infer_with_sge(
            q_text,
            "Answer commonsense multiple choice questions by selecting the correct option letter."
        )

        results.append({
            "dataset": "commonsense_qa",
            "question": question,
            "choices": options_str,
            "gold": gold_label,
            "zero_shot": zs,
            "few_shot": fs,
            "cot": cot,
            "sge": sge_ans,
        })

    out_path = "results/metrics/csqa_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved CommonsenseQA results → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="gsm8k",
        choices=["gsm8k", "boolq", "csqa"],
        help="Which dataset to evaluate on.",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=30,
        help="How many examples to evaluate (per dataset).",
    )
    args = parser.parse_args()

    model_name = "google/flan-t5-base"
    print(f"Loading model: {model_name}")
    baseline = PromptingBaselines(model_name)
    sge = SelfGeneratedExamples(model_name)

    if args.task == "gsm8k":
        evaluate_gsm8k(baseline, sge, num_examples=args.num_examples)
    elif args.task == "boolq":
        evaluate_boolq(baseline, sge, num_examples=args.num_examples)
    elif args.task == "csqa":
        evaluate_csqa(baseline, sge, num_examples=args.num_examples)
    else:
        raise ValueError(f"Unknown task: {args.task}")


if __name__ == "__main__":
    main()
