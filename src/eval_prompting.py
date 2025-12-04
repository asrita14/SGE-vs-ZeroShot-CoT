"""
Evaluate Zero-Shot, Few-Shot, CoT, and SGE on:
- GSM8K (math word problems)
- BoolQ (yes/no QA)
- CommonsenseQA (multiple choice)

This script supports multiple runs via --run_name, so you can store
multiple result files per dataset for analysis.
"""

import json
import argparse
from datasets import load_dataset
from tqdm import tqdm

from src.prompting.baselines import PromptingBaselines
from src.prompting.sge_pipeline import SelfGeneratedExamples


def evaluate_gsm8k(baseline, sge, num_examples=30, sge_k=3, run_name="default"):
    """Evaluate all prompting strategies on GSM8K."""
    gsm = load_dataset("gsm8k", "main")["test"].select(range(num_examples))

    few_shot_examples = [
        ("Tom has 8 apples and eats 3. How many left?", "5"),
        ("What is 12 + 7?", "19"),
        ("A box has 10 candies and 4 more are added. Total?", "14"),
    ]

    results = []

    for sample in tqdm(gsm, desc=f"Evaluating GSM8K ({run_name})"):
        q = sample["question"]
        a = sample["answer"]

        zs = baseline.zero_shot(q)
        fs = baseline.few_shot(q, few_shot_examples)
        cot = baseline.cot(q)
        cot_sc = baseline.cot_self_consistency(q, n=5, temperature=0.7)
        sge_ans = sge.infer_with_sge(
            q,
            "Solve grade-school math word problems."
        )

        results.append({
            "dataset": "gsm8k",
            "question": q,
            "gold": a,
            "zero_shot": zs,
            "few_shot": fs,
            "cot": cot,
            "cot_sc": cot_sc,   # NEW
            "sge": sge_ans,
        })


    payload = {
        "config": {
            "task": "gsm8k",
            "run_name": run_name,
            "num_examples": num_examples,
            "sge_k": sge_k,
            "model_name": baseline.model_name if hasattr(baseline, "model_name") else "unknown",
        },
        "results": results,
    }

    out_path = f"results/metrics/gsm8k_{run_name}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved GSM8K results → {out_path}")


def evaluate_boolq(baseline, sge, num_examples=30, sge_k=3, run_name="default"):
    """Evaluate all prompting strategies on BoolQ (yes/no QA)."""
    boolq = load_dataset("boolq")["validation"].select(range(num_examples))

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

    for sample in tqdm(boolq, desc=f"Evaluating BoolQ ({run_name})"):
        question = sample["question"]
        passage = sample["passage"]
        gold_bool = sample["answer"]
        gold = "yes" if gold_bool else "no"

        q_text = (
            f"Passage: {passage}\n"
            f"Question: {question}\n"
            "Answer yes or no."
        )

        zs = baseline.zero_shot(q_text)
        fs = baseline.few_shot(q_text, few_shot_examples)
        cot = baseline.cot(q_text)
        cot_sc = baseline.cot_self_consistency(q_text, n=5, temperature=0.7)
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
            "cot_sc": cot_sc,   # NEW
            "sge": sge_ans,
        })


    payload = {
        "config": {
            "task": "boolq",
            "run_name": run_name,
            "num_examples": num_examples,
            "sge_k": sge_k,
            "model_name": baseline.model_name if hasattr(baseline, "model_name") else "unknown",
        },
        "results": results,
    }

    out_path = f"results/metrics/boolq_{run_name}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved BoolQ results → {out_path}")


def evaluate_csqa(baseline, sge, num_examples=30, sge_k=3, run_name="default"):
    """Evaluate all prompting strategies on CommonsenseQA (multiple choice)."""
    csqa = load_dataset("commonsense_qa")["validation"].select(range(num_examples))

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

    for sample in tqdm(csqa, desc=f"Evaluating CommonsenseQA ({run_name})"):
        question = sample["question"]
        choices = sample["choices"]["text"]
        labels = sample["choices"]["label"]
        gold_label = sample["answerKey"]

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
        cot_sc = baseline.cot_self_consistency(q_text, n=5, temperature=0.7)
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
            "cot_sc": cot_sc,   # NEW
            "sge": sge_ans,
        })



    payload = {
        "config": {
            "task": "commonsense_qa",
            "run_name": run_name,
            "num_examples": num_examples,
            "sge_k": sge_k,
            "model_name": baseline.model_name if hasattr(baseline, "model_name") else "unknown",
        },
        "results": results,
    }

    out_path = f"results/metrics/csqa_{run_name}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

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
    parser.add_argument(
        "--run_name",
        type=str,
        default="debug",
        help="Name suffix for this run (used in output filename).",
    )
    parser.add_argument(
        "--sge_k",
        type=int,
        default=3,
        help="Number of self-generated examples to use.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/flan-t5-base",
        help="HuggingFace model name to use (e.g., google/flan-t5-base).",
    )
    args = parser.parse_args()

    print(f"Loading model: {args.model_name}")
    baseline = PromptingBaselines(args.model_name)
    baseline.model_name = args.model_name  # for logging
    sge = SelfGeneratedExamples(args.model_name, k=args.sge_k)

    if args.task == "gsm8k":
        evaluate_gsm8k(baseline, sge, num_examples=args.num_examples,
                       sge_k=args.sge_k, run_name=args.run_name)
    elif args.task == "boolq":
        evaluate_boolq(baseline, sge, num_examples=args.num_examples,
                       sge_k=args.sge_k, run_name=args.run_name)
    elif args.task == "csqa":
        evaluate_csqa(baseline, sge, num_examples=args.num_examples,
                      sge_k=args.sge_k, run_name=args.run_name)
    else:
        raise ValueError(f"Unknown task: {args.task}")


if __name__ == "__main__":
    main()
