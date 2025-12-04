#!/usr/bin/env python

"""
make_plots.py

Generate figures for the poster from the metrics JSON files.

Assumes files like:
  results/metrics/boolq_bool200_filtered_sc_metrics.json
  results/metrics/csqa_csqa200_filtered_sc_metrics.json
  results/metrics/gsm8k_gsm200_filtered_sc_metrics.json

And the corresponding raw results:
  results/metrics/boolq_bool200_filtered_sc.json
  results/metrics/csqa_csqa200_filtered_sc.json
  results/metrics/gsm8k_gsm200_filtered_sc.json
"""

import os
import json

import matplotlib.pyplot as plt
from datasets import load_dataset

from src.compute_metrics import (
    compute_gsm8k_metrics,
    compute_boolq_metrics,
    compute_csqa_metrics,
)
from src.difficulty_analysis import (
    tag_difficulty_gsm8k,
    tag_difficulty_boolq,
)


METRICS_DIR = "results/metrics"


# -----------------------
# Helpers
# -----------------------

def load_metrics_json(task: str, run_name: str):
    """
    Load aggregated metrics JSON:
      results/metrics/{task}_{run_name}_metrics.json
    """
    path = os.path.join(METRICS_DIR, f"{task}_{run_name}_metrics.json")
    with open(path, "r") as f:
        data = json.load(f)
    return data["metrics"]  # dict[strategy -> metrics]


def load_results_json(task: str, run_name: str):
    """
    Load raw per-example results JSON:
      results/metrics/{task}_{run_name}.json
    """
    path = os.path.join(METRICS_DIR, f"{task}_{run_name}.json")
    with open(path, "r") as f:
        data = json.load(f)
    return data["results"]


# -----------------------
# Main results plots
# -----------------------

def plot_boolq_main(run_name="bool200_filtered_sc"):
    """
    Bar chart of BoolQ accuracy for each strategy (200 examples).
    """
    metrics = load_metrics_json("boolq", run_name)

    strategies = ["zero_shot", "few_shot", "cot", "cot_sc", "sge"]
    labels = ["Zero-shot", "Few-shot", "CoT", "CoT-SC", "SGE-filtered"]

    values = [metrics[s]["accuracy"] for s in strategies]

    plt.figure()
    x = range(len(strategies))
    plt.bar(x, values)
    plt.xticks(x, labels, rotation=20)
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.0)
    plt.title("BoolQ (200 ex) – Accuracy by Prompting Strategy")
    plt.tight_layout()
    plt.savefig("fig_boolq_main.png", dpi=300)
    plt.close()


def plot_csqa_main(run_name="csqa200_filtered_sc"):
    """
    Bar chart of CSQA accuracy for each strategy (200 examples).
    """
    metrics = load_metrics_json("csqa", run_name)

    strategies = ["zero_shot", "few_shot", "cot", "cot_sc", "sge"]
    labels = ["Zero-shot", "Few-shot", "CoT", "CoT-SC", "SGE-filtered"]

    values = [metrics[s]["accuracy"] for s in strategies]

    plt.figure()
    x = range(len(strategies))
    plt.bar(x, values)
    plt.xticks(x, labels, rotation=20)
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.0)
    plt.title("CommonsenseQA (200 ex) – Accuracy by Prompting Strategy")
    plt.tight_layout()
    plt.savefig("fig_csqa_main.png", dpi=300)
    plt.close()


def plot_gsm8k_main(run_name="gsm200_filtered_sc"):
    """
    Two plots:
      - Exact match
      - MAE
    for GSM8K (200 examples).
    """
    metrics = load_metrics_json("gsm8k", run_name)

    strategies = ["zero_shot", "few_shot", "cot", "cot_sc", "sge"]
    labels = ["Zero-shot", "Few-shot", "CoT", "CoT-SC", "SGE-filtered"]

    em_values = [metrics[s]["numeric_em"] for s in strategies]
    mae_values = [metrics[s]["mae"] for s in strategies]

    # Exact match
    plt.figure()
    x = range(len(strategies))
    plt.bar(x, em_values)
    plt.xticks(x, labels, rotation=20)
    plt.ylabel("Exact Match")
    plt.ylim(0.0, 1.0)
    plt.title("GSM8K (200 ex) – Exact Match by Prompting Strategy")
    plt.tight_layout()
    plt.savefig("fig_gsm8k_em.png", dpi=300)
    plt.close()

    # MAE
    plt.figure()
    x = range(len(strategies))
    plt.bar(x, mae_values)
    plt.xticks(x, labels, rotation=20)
    plt.ylabel("Mean Absolute Error")
    plt.title("GSM8K (200 ex) – MAE by Prompting Strategy")
    plt.tight_layout()
    plt.savefig("fig_gsm8k_mae.png", dpi=300)
    plt.close()


# -----------------------
# Difficulty plots
# -----------------------

def plot_boolq_difficulty(run_name="bool200_filtered_sc", num_examples=200):
    """
    Grouped bar chart: BoolQ easy vs hard accuracy by strategy.
    Uses the same heuristic difficulty split as difficulty_analysis.py.
    """
    # Load raw results
    results = load_results_json("boolq", run_name)

    # Load original dataset and compute difficulty tags
    ds = load_dataset("boolq")["validation"].select(range(num_examples))
    passages = [ex["passage"] for ex in ds]
    diffs = tag_difficulty_boolq(passages)  # "easy" / "hard"

    # Split results into easy / hard
    easy_results = []
    hard_results = []
    for d, r in zip(diffs, results):
        if d == "easy":
            easy_results.append(r)
        else:
            hard_results.append(r)

    easy_metrics = compute_boolq_metrics(easy_results)
    hard_metrics = compute_boolq_metrics(hard_results)

    strategies = ["zero_shot", "few_shot", "cot", "cot_sc", "sge"]
    labels = ["Zero-shot", "Few-shot", "CoT", "CoT-SC", "SGE-filtered"]

    easy_vals = [easy_metrics[s]["accuracy"] for s in strategies]
    hard_vals = [hard_metrics[s]["accuracy"] for s in strategies]

    x = range(len(strategies))
    width = 0.35

    plt.figure()
    plt.bar([i - width / 2 for i in x], easy_vals, width=width, label="Easy")
    plt.bar([i + width / 2 for i in x], hard_vals, width=width, label="Hard")
    plt.xticks(x, labels, rotation=20)
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.0)
    plt.title("BoolQ – Easy vs Hard Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig_boolq_difficulty.png", dpi=300)
    plt.close()


def plot_gsm8k_difficulty(run_name="gsm200_filtered_sc", num_examples=200):
    """
    Grouped bar chart: GSM8K easy vs hard exact match by strategy.
    Uses the same heuristic difficulty split as difficulty_analysis.py.
    """
    # Load raw results
    results = load_results_json("gsm8k", run_name)

    # Load original dataset and compute difficulty tags
    ds = load_dataset("gsm8k", "main")["test"].select(range(num_examples))
    questions = [ex["question"] for ex in ds]
    diffs = tag_difficulty_gsm8k(questions)  # "easy" / "hard"

    easy_results = []
    hard_results = []
    for d, r in zip(diffs, results):
        if d == "easy":
            easy_results.append(r)
        else:
            hard_results.append(r)

    easy_metrics = compute_gsm8k_metrics(easy_results)
    hard_metrics = compute_gsm8k_metrics(hard_results)

    strategies = ["zero_shot", "few_shot", "cot", "cot_sc", "sge"]
    labels = ["Zero-shot", "Few-shot", "CoT", "CoT-SC", "SGE-filtered"]

    easy_vals = [easy_metrics[s]["numeric_em"] for s in strategies]
    hard_vals = [hard_metrics[s]["numeric_em"] for s in strategies]

    x = range(len(strategies))
    width = 0.35

    plt.figure()
    plt.bar([i - width / 2 for i in x], easy_vals, width=width, label="Easy")
    plt.bar([i + width / 2 for i in x], hard_vals, width=width, label="Hard")
    plt.xticks(x, labels, rotation=20)
    plt.ylabel("Exact Match")
    plt.ylim(0.0, 1.0)
    plt.title("GSM8K – Easy vs Hard Exact Match")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig_gsm8k_difficulty.png", dpi=300)
    plt.close()


# -----------------------
# Main entry
# -----------------------

if __name__ == "__main__":
    # Main results
    plot_boolq_main(run_name="bool200_filtered_sc")
    plot_csqa_main(run_name="csqa200_filtered_sc")
    plot_gsm8k_main(run_name="gsm200_filtered_sc")

    # Difficulty plots
    plot_boolq_difficulty(run_name="bool200_filtered_sc", num_examples=200)
    plot_gsm8k_difficulty(run_name="gsm200_filtered_sc", num_examples=200)

    print("Saved figures:")
    print("  fig_boolq_main.png")
    print("  fig_csqa_main.png")
    print("  fig_gsm8k_em.png")
    print("  fig_gsm8k_mae.png")
    print("  fig_boolq_difficulty.png")
    print("  fig_gsm8k_difficulty.png")
