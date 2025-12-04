"""
compute_metrics.py

Given a results JSON produced by eval_prompting.py, compute metrics for:
- GSM8K: numeric exact match (EM) and MAE
- BoolQ: accuracy (yes/no)
- CommonsenseQA: accuracy (multiple choice A-E)

Usage examples:

  python -m src.compute_metrics --task gsm8k --run_name tinytest
  python -m src.compute_metrics --task boolq --run_name base100
  python -m src.compute_metrics --task csqa --run_name full50

This assumes files are named:
  results/metrics/{task}_{run_name}.json
"""

import os
import re
import json
import argparse
from typing import List, Dict, Any, Optional


# =========================
# Utility functions
# =========================

def load_results(task: str, run_name: str, metrics_dir: str = "results/metrics") -> Dict[str, Any]:
    """
    Load a results JSON with structure:
    {
      "config": {...},
      "results": [ {...}, {...}, ... ]
    }
    """
    filename = f"{task}_{run_name}.json"
    path = os.path.join(metrics_dir, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find results file at: {path}")

    with open(path, "r") as f:
        data = json.load(f)
    return data


def extract_first_number(text: str) -> Optional[float]:
    """
    Extract the first integer or decimal number from a string.
    Returns None if no number is found.
    """
    if text is None:
        return None
    # Look for patterns like -12, 3, 4.5 etc.
    match = re.search(r"-?\d+(\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def normalize_yes_no(pred: str) -> Optional[str]:
    """
    Normalize a free-form prediction to 'yes' / 'no' if possible.
    Returns None if it can't be interpreted.
    """
    if pred is None:
        return None
    t = pred.strip().lower()

    # Look at just the first word to avoid confusion from extra text.
    first_token = t.split()[0] if t.split() else ""

    if first_token.startswith("yes") or first_token.startswith("y"):
        return "yes"
    if first_token.startswith("no") or first_token.startswith("n"):
        return "no"
    return None


def extract_choice_label(pred: str) -> Optional[str]:
    """
    Try to extract a single-choice label (A/B/C/D/E) from the prediction.
    Returns the letter as uppercase, or None if not found.
    """
    if pred is None:
        return None
    # Look for a standalone A-E
    match = re.search(r"\b([A-E])\b", pred)
    if match:
        return match.group(1).upper()

    # Fallback: first character if it's in A-E
    first_char = pred.strip()[:1].upper()
    if first_char in ["A", "B", "C", "D", "E"]:
        return first_char

    return None

def get_strategies(results: List[Dict[str, Any]]) -> list:
    """
    Determine which strategies are present in the results.
    This lets us support new strategies (cot_sc, few_shot_data, etc)
    without hard-coding them everywhere.
    """
    # candidates = superset of all strategies we might log
    candidates = [
        "zero_shot",
        "few_shot",
        "few_shot_data",
        "cot",
        "cot_sc",
        "sge",
    ]
    present = set()
    for r in results:
        for c in candidates:
            if c in r:
                present.add(c)
    # preserve order of candidates, keep only those actually present
    return [c for c in candidates if c in present]

# =========================
# GSM8K metrics
# =========================
def compute_gsm8k_metrics(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Compute numeric EM and MAE for each prompting strategy on GSM8K.

    We treat a prediction as correct (EM) if the extracted number equals
    the number extracted from the gold answer.
    """
    strategies = get_strategies(results)
    metrics = {}

    for strat in strategies:
        total = 0
        em_count = 0
        abs_errors = []

        for item in results:
            gold_text = item.get("gold", "")
            pred_text = item.get(strat, "")

            gold_num = extract_first_number(gold_text)
            pred_num = extract_first_number(pred_text)

            # Only consider examples where both numbers can be parsed
            if gold_num is None or pred_num is None:
                continue

            total += 1
            if abs(gold_num - pred_num) < 1e-6:
                em_count += 1
            abs_errors.append(abs(gold_num - pred_num))

        if total == 0:
            em = 0.0
            mae = 0.0
        else:
            em = em_count / total
            mae = sum(abs_errors) / len(abs_errors)

        metrics[strat] = {
            "num_evaluated": total,
            "numeric_em": em,
            "mae": mae,
        }

    return metrics


# =========================
# BoolQ metrics
# =========================

def compute_boolq_metrics(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Compute accuracy for each prompting strategy on BoolQ.
    Gold labels are 'yes'/'no', predictions are free-form.
    """
    strategies = get_strategies(results)
    metrics = {}

    for strat in strategies:
        total = 0
        correct = 0
        covered = 0  # number of predictions we could interpret as yes/no

        for item in results:
            gold = item.get("gold", "").strip().lower()
            pred_text = item.get(strat, "")

            pred_norm = normalize_yes_no(pred_text)
            total += 1

            if pred_norm is not None:
                covered += 1
                if pred_norm == gold:
                    correct += 1

        acc = (correct / covered) if covered > 0 else 0.0
        coverage = covered / total if total > 0 else 0.0

        metrics[strat] = {
            "num_examples": total,
            "num_covered": covered,
            "coverage": coverage,
            "accuracy": acc,
        }

    return metrics


# =========================
# CommonsenseQA metrics
# =========================

def compute_csqa_metrics(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Compute multiple-choice accuracy for each prompting strategy on CommonsenseQA.
    Gold labels are 'A'-'E', predictions are free-form.
    """
    strategies = get_strategies(results)
    metrics = {}

    for strat in strategies:
        total = 0
        correct = 0
        covered = 0

        for item in results:
            gold_label = item.get("gold", "").strip().upper()
            pred_text = item.get(strat, "")

            pred_label = extract_choice_label(pred_text)
            total += 1

            if pred_label is not None:
                covered += 1
                if pred_label == gold_label:
                    correct += 1

        acc = (correct / covered) if covered > 0 else 0.0
        coverage = covered / total if total > 0 else 0.0

        metrics[strat] = {
            "num_examples": total,
            "num_covered": covered,
            "coverage": coverage,
            "accuracy": acc,
        }

    return metrics



# =========================
# Main entry point
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["gsm8k", "boolq", "csqa"],
        help="Which dataset's results to score.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="Run name used when generating results (part of filename).",
    )
    parser.add_argument(
        "--metrics_dir",
        type=str,
        default="results/metrics",
        help="Directory where result JSONs are stored.",
    )
    args = parser.parse_args()

    # Map task name strings to metric functions
    task_to_metric_fn = {
        "gsm8k": compute_gsm8k_metrics,
        "boolq": compute_boolq_metrics,
        "csqa": compute_csqa_metrics,
    }

    data = load_results(args.task, args.run_name, metrics_dir=args.metrics_dir)
    config = data.get("config", {})
    results = data.get("results", [])

    print(f"Loaded {len(results)} examples for task={args.task}, run_name={args.run_name}")
    print(f"Config: {config}")

    metric_fn = task_to_metric_fn[args.task]
    metrics = metric_fn(results)

    # Print metrics nicely
    print("\n=== METRICS ===")
    for strat, vals in metrics.items():
        print(f"\nStrategy: {strat}")
        for k, v in vals.items():
            # print floats with 4 decimal places
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

    # Save metrics to a JSON file for later plotting/reporting
    out_name = f"{args.task}_{args.run_name}_metrics.json"
    out_path = os.path.join(args.metrics_dir, out_name)
    out_payload = {
        "task": args.task,
        "run_name": args.run_name,
        "config": config,
        "metrics": metrics,
    }
    with open(out_path, "w") as f:
        json.dump(out_payload, f, indent=2)

    print(f"\nSaved metric summary → {out_path}")


if __name__ == "__main__":
    main()
