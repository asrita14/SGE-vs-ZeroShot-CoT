#!/usr/bin/env python

"""
fancy_plots.py

Generate nicer, more "researchy" figures for the poster:

1. Radar chart: strategy performance across tasks (BoolQ acc, CSQA acc, GSM8K EM)
2. Heatmap: BoolQ accuracy for Easy vs Hard subsets by strategy
3. Delta bar plot: BoolQ accuracy difference (filtered vs unfiltered) per strategy
4. Relative cost bar plot: approximate inference cost per strategy

Assumes the following metric files exist:

  results/metrics/boolq_bool200_filtered_sc_metrics.json
  results/metrics/boolq_bool200_unfiltered_metrics.json
  results/metrics/csqa_csqa200_filtered_sc_metrics.json
  results/metrics/gsm8k_gsm200_filtered_sc_metrics.json

And raw results files:

  results/metrics/boolq_bool200_filtered_sc.json
  results/metrics/gsm8k_gsm200_filtered_sc.json
"""

import os
import json
import math

import matplotlib.pyplot as plt
from datasets import load_dataset

from src.compute_metrics import (
    compute_gsm8k_metrics,
    compute_boolq_metrics,
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


# ============================================================
# 1) Radar chart: strategy performance across tasks
# ============================================================

def plot_radar_across_tasks():
    """
    Radar chart where each axis is a task-metric:
      - BoolQ accuracy
      - CSQA accuracy
      - GSM8K EM

    We plot three strategies as polygons:
      - Zero-shot
      - CoT
      - SGE-filtered
    """

    boolq_metrics = load_metrics_json("boolq", "bool200_filtered_sc")
    csqa_metrics = load_metrics_json("csqa", "csqa200_filtered_sc")
    gsm8k_metrics = load_metrics_json("gsm8k", "gsm200_filtered_sc")

    # Axes labels (three dimensions)
    labels = ["BoolQ Acc", "CSQA Acc", "GSM8K EM"]

    # Strategies to show
    strategies = {
        "Zero-shot": {
            "boolq": boolq_metrics["zero_shot"]["accuracy"],
            "csqa": csqa_metrics["zero_shot"]["accuracy"],
            "gsm8k": gsm8k_metrics["zero_shot"]["numeric_em"],
        },
        "CoT": {
            "boolq": boolq_metrics["cot"]["accuracy"],
            "csqa": csqa_metrics["cot"]["accuracy"],
            "gsm8k": gsm8k_metrics["cot"]["numeric_em"],
        },
        "SGE-filtered": {
            "boolq": boolq_metrics["sge"]["accuracy"],
            "csqa": csqa_metrics["sge"]["accuracy"],
            "gsm8k": gsm8k_metrics["sge"]["numeric_em"],
        },
    }

    # Normalize GSM8K EM to [0,1] just for visual comparability (optional)
    # Here we just keep raw EM since it's already in [0,1].

    # Prepare angle positions
    num_axes = len(labels)
    angles = [n / float(num_axes) * 2.0 * math.pi for n in range(num_axes)]
    angles += angles[:1]  # close the loop

    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)

    for strat_name, vals in strategies.items():
        stats = [
            vals["boolq"],
            vals["csqa"],
            vals["gsm8k"],
        ]
        stats += stats[:1]  # close polygon

        ax.plot(angles, stats, linewidth=1, linestyle="-", label=strat_name)
        ax.fill(angles, stats, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"])
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Strategy Performance Across Tasks", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1))

    plt.tight_layout()
    plt.savefig("fig_radar_across_tasks.png", dpi=300)
    plt.close()


# ============================================================
# 2) Heatmap: BoolQ Easy vs Hard accuracy by strategy
# ============================================================

def plot_boolq_heatmap_difficulty(run_name="bool200_filtered_sc", num_examples=200):
    """
    Heatmap:
        rows = [Easy, Hard]
        cols = [Zero-shot, Few-shot, CoT, CoT-SC, SGE-filtered]
        values = accuracy

    Uses the same difficulty split as difficulty_analysis.py.
    """
    results = load_results_json("boolq", run_name)

    ds = load_dataset("boolq")["validation"].select(range(num_examples))
    passages = [ex["passage"] for ex in ds]
    diffs = tag_difficulty_boolq(passages)  # "easy" / "hard"

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
    col_labels = ["Zero-shot", "Few-shot", "CoT", "CoT-SC", "SGE-filtered"]

    data = [
        [easy_metrics[s]["accuracy"] for s in strategies],  # easy row
        [hard_metrics[s]["accuracy"] for s in strategies],  # hard row
    ]

    fig, ax = plt.subplots()
    im = ax.imshow(data, aspect="auto")

    # Set ticks
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=20)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Easy", "Hard"])

    # Annotate cells with values
    for i in range(2):
        for j in range(len(col_labels)):
            val = data[i][j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white")

    ax.set_title("BoolQ – Easy vs Hard Accuracy (Heatmap)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Accuracy")

    plt.tight_layout()
    plt.savefig("fig_boolq_difficulty_heatmap.png", dpi=300)
    plt.close()


# ============================================================
# 3) Delta bar plot: BoolQ filtered vs unfiltered metrics
# ============================================================

def plot_boolq_delta_filtered_vs_unfiltered(
    filtered_run="bool200_filtered_sc",
    unfiltered_run="bool200_unfiltered",
):
    """
    Bar plot of (filtered - unfiltered) accuracy per strategy on BoolQ.

    Positive values mean filtering / CoT-SC setup improved accuracy.
    """

    filtered = load_metrics_json("boolq", filtered_run)
    unfiltered = load_metrics_json("boolq", unfiltered_run)

    strategies = ["zero_shot", "few_shot", "cot", "cot_sc", "sge"]
    labels = ["Zero-shot", "Few-shot", "CoT", "CoT-SC", "SGE"]

    deltas = [
        filtered[s]["accuracy"] - unfiltered[s]["accuracy"] for s in strategies
    ]

    x = range(len(strategies))

    plt.figure()
    plt.axhline(0.0, linestyle="--")
    plt.bar(x, deltas)
    plt.xticks(x, labels, rotation=20)
    plt.ylabel("Δ Accuracy (Filtered − Unfiltered)")
    plt.title("BoolQ – Effect of Filtering / SC Setup")
    plt.tight_layout()
    plt.savefig("fig_boolq_delta_filtered.png", dpi=300)
    plt.close()


# ============================================================
# 4) Relative cost bar plot
# ============================================================

def plot_relative_cost():
    """
    Approximate relative inference cost (number of model calls per question)
    for each strategy. This is not exact timing, but a conceptual plot.

    Assumptions:
      Zero-shot:       1 call
      Few-shot:        1 call
      CoT:             1 call
      CoT-SC (n=5):    5 calls
      SGE-unfiltered:  1 (examples) + 1 (answer) = 2
      SGE-filtered:    same as SGE (~2)
    """

    strategies = [
        "Zero-shot",
        "Few-shot",
        "CoT",
        "CoT-SC",
        "SGE",
        "SGE-filtered",
    ]
    relative_calls = [1, 1, 1, 5, 2, 2]

    x = range(len(strategies))

    plt.figure()
    plt.bar(x, relative_calls)
    plt.xticks(x, strategies, rotation=20)
    plt.ylabel("Relative Model Calls per Question")
    plt.title("Relative Inference Cost by Strategy")
    plt.tight_layout()
    plt.savefig("fig_relative_cost.png", dpi=300)
    plt.close()


# ============================================================
# Main entry
# ============================================================

if __name__ == "__main__":
    # 1) Radar chart across tasks
    plot_radar_across_tasks()

    # 2) BoolQ difficulty heatmap
    plot_boolq_heatmap_difficulty(run_name="bool200_filtered_sc", num_examples=200)

    # 3) BoolQ filtered vs unfiltered delta
    plot_boolq_delta_filtered_vs_unfiltered(
        filtered_run="bool200_filtered_sc",
        unfiltered_run="bool200_unfiltered",
    )

    # 4) Relative cost plot
    plot_relative_cost()

    print("Saved figures:")
    print("  fig_radar_across_tasks.png")
    print("  fig_boolq_difficulty_heatmap.png")
    print("  fig_boolq_delta_filtered.png")
    print("  fig_relative_cost.png")
