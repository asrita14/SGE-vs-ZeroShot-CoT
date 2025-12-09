import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


# --------------------------
# 1. Answer extraction helpers
# --------------------------

def extract_gsm8k_answer(text: str) -> str:
    """
    Extract the final numeric answer from a GSM8K-style solution.
    Heuristic: take the last integer in the string.
    """
    if text is None:
        return ""
    # Find all integers (possibly negative)
    nums = re.findall(r"-?\d+", text)
    if not nums:
        return text.strip()
    return nums[-1].lstrip("+").strip()


def extract_boolq_answer(text: str) -> str:
    """
    Normalize yes/no answers from text.
    """
    if text is None:
        return ""
    t = text.lower()
    if "yes" in t and "no" not in t:
        return "yes"
    if "no" in t and "yes" not in t:
        return "no"
    # fallback: first word
    return t.strip().split()[0]


def extract_csqa_answer(text: str) -> str:
    """
    Extract option letter A/B/C/D/E from the model output.
    """
    if text is None:
        return ""
    t = text.strip().upper()

    # Look for standalone letters A-E
    m = re.search(r"\b([A-E])\b", t)
    if m:
        return m.group(1)

    # Sometimes models say 'Option B' or 'Answer: C'
    m = re.search(r"OPTION\s+([A-E])", t)
    if m:
        return m.group(1)

    m = re.search(r"ANSWER[:\s]+([A-E])", t)
    if m:
        return m.group(1)

    # fallback: first capital letter
    for ch in t:
        if ch in "ABCDE":
            return ch

    return t[:1]  # terrible fallback, but better than empty


def get_extract_fn(task: str):
    if task == "gsm8k":
        return extract_gsm8k_answer
    elif task == "boolq":
        return extract_boolq_answer
    elif task == "commonsense_qa":
        return extract_csqa_answer
    else:
        # identity
        return lambda x: ("" if x is None else str(x).strip())


# --------------------------
# 2. Load one JSON result file
# --------------------------

def load_result(path: Path) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


# --------------------------
# 3. Compute accuracy & avg time per method
# --------------------------

def compute_metrics_for_file(path: Path) -> Dict:
    data = load_result(path)
    cfg = data.get("config", {})
    results = data.get("results", [])

    task = cfg.get("task", "unknown")
    extract_fn = get_extract_fn(task)

    # Fields we expect per-example
    methods = [
        "zero_shot",
        "few_shot",
        "cot",
        "cot_sc",
        "cot_sir",
        "sge",
    ]
    time_fields = {
        "zero_shot": "zero_shot_time",
        "few_shot": "few_shot_time",
        "cot": "cot_time",
        "cot_sc": "cot_sc_time",
        "cot_sir": "cot_sir_time",
        "sge": "sge_time",
    }

    metrics = {
        m: {
            "correct": 0,
            "total": 0,
            "times": [],
        }
        for m in methods
    }

    for ex in results:
        gold = ex.get("gold", None)
        if gold is None:
            continue

        # Normalize gold answer
        if task == "gsm8k":
            gold_norm = extract_gsm8k_answer(gold)
        elif task == "boolq":
            gold_norm = extract_boolq_answer(gold)
        elif task == "commonsense_qa":
            gold_norm = extract_csqa_answer(gold)
        else:
            gold_norm = str(gold).strip()

        for m in methods:
            pred = ex.get(m, None)
            if pred is None:
                continue

            # normalize prediction
            if task == "gsm8k":
                pred_norm = extract_gsm8k_answer(pred)
            elif task == "boolq":
                pred_norm = extract_boolq_answer(pred)
            elif task == "commonsense_qa":
                pred_norm = extract_csqa_answer(pred)
            else:
                pred_norm = str(pred).strip()

            metrics[m]["total"] += 1
            if pred_norm == gold_norm:
                metrics[m]["correct"] += 1

            t_field = time_fields[m]
            t_val = ex.get(t_field, None)
            if isinstance(t_val, (int, float)):
                metrics[m]["times"].append(float(t_val))

    # finalize
    summary = {}
    for m in methods:
        total = metrics[m]["total"]
        correct = metrics[m]["correct"]
        times = metrics[m]["times"]
        acc = correct / total if total > 0 else 0.0
        avg_time = float(np.mean(times)) if times else 0.0

        summary[m] = {
            "accuracy": acc,
            "avg_time": avg_time,
            "n": total,
        }

    return {
        "config": cfg,
        "metrics": summary,
        "path": str(path),
    }


# --------------------------
# 4. Plot functions
# --------------------------

def plot_accuracy_bar(summary: Dict, title_suffix: str = ""):
    metrics = summary["metrics"]
    methods = list(metrics.keys())
    accuracies = [metrics[m]["accuracy"] for m in methods]

    plt.figure()
    plt.bar(methods, accuracies)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.0)
    plt.title(f"Accuracy by method{title_suffix}")
    plt.tight_layout()


def plot_time_bar(summary: Dict, title_suffix: str = ""):
    metrics = summary["metrics"]
    methods = list(metrics.keys())
    avg_times = [metrics[m]["avg_time"] for m in methods]

    plt.figure()
    plt.bar(methods, avg_times)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Avg inference time per example (s)")
    plt.title(f"Inference time by method{title_suffix}")
    plt.tight_layout()


def plot_cost_vs_accuracy(summary: Dict, title_suffix: str = ""):
    metrics = summary["metrics"]
    methods = list(metrics.keys())

    accuracies = [metrics[m]["accuracy"] for m in methods]
    avg_times = [metrics[m]["avg_time"] for m in methods]

    plt.figure()
    plt.scatter(avg_times, accuracies)

    for m, x, y in zip(methods, avg_times, accuracies):
        plt.text(x, y, m, fontsize=8)

    plt.xlabel("Avg inference time per example (s)")
    plt.ylabel("Accuracy")
    plt.title(f"Cost vs Accuracy{title_suffix}")
    plt.tight_layout()


# --------------------------
# 5. Main entrypoint
# --------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_file",
        type=str,
        required=True,
        help="Path to a single JSON results file (e.g., results/metrics/gsm8k_1k.json)",
    )
    parser.add_argument(
        "--save_prefix",
        type=str,
        default=None,
        help="If set, save the plots to files with this prefix instead of showing.",
    )
    args = parser.parse_args()

    path = Path(args.results_file)
    summary = compute_metrics_for_file(path)

    cfg = summary["config"]
    task = cfg.get("task", "unknown")
    run_name = cfg.get("run_name", "run")

    title_suffix = f" ({task}, {run_name})"

    # Plot accuracy bar
    plot_accuracy_bar(summary, title_suffix=title_suffix)

    # Plot time bar
    plot_time_bar(summary, title_suffix=title_suffix)

    # Plot cost vs accuracy
    plot_cost_vs_accuracy(summary, title_suffix=title_suffix)

    if args.save_prefix:
        # save three plots as PNG
        base = args.save_prefix
        plt.figure(1)
        plt.savefig(f"{base}_accuracy.png", dpi=200)
        plt.figure(2)
        plt.savefig(f"{base}_time.png", dpi=200)
        plt.figure(3)
        plt.savefig(f"{base}_cost_vs_accuracy.png", dpi=200)
    else:
        plt.show()


if __name__ == "__main__":
    main()