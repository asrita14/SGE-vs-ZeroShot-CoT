import os
import json

import matplotlib.pyplot as plt
from datasets import load_dataset

from src.compute_metrics import (
    compute_gsm8k_metrics,
    compute_boolq_metrics,
    compute_csqa_metrics,
)


# ========= Difficulty tagging (copied from difficulty_analysis.py) =========

def tag_difficulty_gsm8k(questions):
    tags = []
    for q in questions:
        length = len(q.split())
        if length <= 40:
            tags.append("easy")
        else:
            tags.append("hard")
    return tags


def tag_difficulty_boolq(passages):
    tags = []
    for p in passages:
        length = len(p.split())
        if length <= 80:
            tags.append("easy")
        else:
            tags.append("hard")
    return tags


def tag_difficulty_csqa(questions):
    tags = []
    for q in questions:
        length = len(q.split())
        if length <= 10:
            tags.append("easy")
        else:
            tags.append("hard")
    return tags


# ========= Helper to load results JSON =========

def load_results(task_id, run_name, metrics_dir="results/metrics"):
    filename = f"{task_id}_{run_name}.json"
    path = os.path.join(metrics_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find {path}")
    with open(path, "r") as f:
        data = json.load(f)
    return data["results"]


# ========= Build easy/hard matrices for each dataset =========

def build_gsm8k_matrix(run_name="gsm200_filtered_sc", num_examples=200):
    ds = load_dataset("gsm8k", "main")["test"].select(range(num_examples))
    questions = [ex["question"] for ex in ds]
    diffs = tag_difficulty_gsm8k(questions)

    results = load_results("gsm8k", run_name)
    assert len(results) == num_examples, "Mismatch in GSM8K example count."

    easy_results = []
    hard_results = []
    for diff, r in zip(diffs, results):
        (easy_results if diff == "easy" else hard_results).append(r)

    # Use your existing metric computation
    easy_metrics = compute_gsm8k_metrics(easy_results)
    hard_metrics = compute_gsm8k_metrics(hard_results)

    strategies = ["zero_shot", "few_shot", "cot", "cot_sc", "sge"]
    # For GSM8K we plot numeric_em
    metric_key = "numeric_em"

    matrix = []
    for subset_metrics in (easy_metrics, hard_metrics):
        row = [subset_metrics[s][metric_key] for s in strategies]
        matrix.append(row)

    return strategies, matrix


def build_boolq_matrix(run_name="bool200_filtered_sc", num_examples=200):
    ds = load_dataset("boolq")["validation"].select(range(num_examples))
    passages = [ex["passage"] for ex in ds]
    diffs = tag_difficulty_boolq(passages)

    results = load_results("boolq", run_name)
    assert len(results) == num_examples, "Mismatch in BoolQ example count."

    easy_results = []
    hard_results = []
    for diff, r in zip(diffs, results):
        (easy_results if diff == "easy" else hard_results).append(r)

    easy_metrics = compute_boolq_metrics(easy_results)
    hard_metrics = compute_boolq_metrics(hard_results)

    strategies = ["zero_shot", "few_shot", "cot", "cot_sc", "sge"]
    metric_key = "accuracy"

    matrix = []
    for subset_metrics in (easy_metrics, hard_metrics):
        row = [subset_metrics[s][metric_key] for s in strategies]
        matrix.append(row)

    return strategies, matrix


def build_csqa_matrix(run_name="csqa200_filtered_sc", num_examples=200):
    ds = load_dataset("commonsense_qa")["validation"].select(range(num_examples))
    questions = [ex["question"] for ex in ds]
    diffs = tag_difficulty_csqa(questions)

    results = load_results("csqa", run_name)
    assert len(results) == num_examples, "Mismatch in CSQA example count."

    easy_results = []
    hard_results = []
    for diff, r in zip(diffs, results):
        (easy_results if diff == "easy" else hard_results).append(r)

    easy_metrics = compute_csqa_metrics(easy_results)
    hard_metrics = compute_csqa_metrics(hard_results)

    strategies = ["zero_shot", "few_shot", "cot", "cot_sc", "sge"]
    metric_key = "accuracy"

    matrix = []
    for subset_metrics in (easy_metrics, hard_metrics):
        row = [subset_metrics[s][metric_key] for s in strategies]
        matrix.append(row)

    return strategies, matrix


# ========= Plotting all three as one figure =========

def plot_all_difficulty_heatmaps(
    gsm_run="gsm200_filtered_sc",
    bool_run="bool200_filtered_sc",
    csqa_run="csqa200_filtered_sc",
    out_dir="results/figs",
):
    os.makedirs(out_dir, exist_ok=True)

    # Build matrices
    gsm_strats, gsm_matrix = build_gsm8k_matrix(gsm_run)
    bool_strats, bool_matrix = build_boolq_matrix(bool_run)
    csqa_strats, csqa_matrix = build_csqa_matrix(csqa_run)

    # Sanity check: strategies should match across tasks
    assert gsm_strats == bool_strats == csqa_strats
    strategies = gsm_strats

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True)

    tasks = [
        ("GSM8K (numeric EM)", gsm_matrix),
        ("BoolQ (accuracy)", bool_matrix),
        ("CommonsenseQA (accuracy)", csqa_matrix),
    ]

    ylabels = ["Easy", "Hard"]

    for ax, (title, matrix) in zip(axes, tasks):
        im = ax.imshow(matrix, aspect="auto")
        ax.set_title(title)
        ax.set_xticks(range(len(strategies)))
        ax.set_xticklabels(strategies, rotation=45, ha="right")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(ylabels)

        # Add colorbar for each subplot
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Score", rotation=270, labelpad=15)

    fig.suptitle("Easy vs Hard Performance by Prompting Strategy", fontsize=16)
    out_path = os.path.join(out_dir, "difficulty_heatmaps.png")
    plt.savefig(out_path, dpi=300)
    print(f"Saved heatmap figure → {out_path}")


if __name__ == "__main__":
    plot_all_difficulty_heatmaps()
