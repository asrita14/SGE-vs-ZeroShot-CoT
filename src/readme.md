

# src/ — Experiment Execution Guide

This folder contains all experiment scripts.  
Use these instructions to run evaluations, compute metrics, and generate analyses.

---

# For Running Evaluations

Run all prompting methods (Zero-shot, Few-shot, CoT, CoT-SC, SGE, Filtered SGE):

```bash
python -m src.eval_prompting \
  --task {gsm8k|boolq|csqa} \
  --num_examples N \
  --run_name NAME \
  --sge_k 3 \
  --model_name google/flan-t5-base
```
Example:
```bash
python -m src.eval_prompting --task boolq --num_examples 200 --run_name bool200_filtered_sc
```
Outputs are saved to:
results/metrics/<task>_<run_name>.json

# For Computing Metrics:
```bash
python -m src.compute_metrics \
  --task boolq \
  --run_name bool200_filtered_sc
```
It should Produce something like:
results/metrics/<task>_<run_name>_metrics.json

# Difficulty Analysis (Easy vs Hard)
```bash
python -m src.difficulty_analysis
```
Tags examples as “easy” or “hard” using dataset-specific heuristics and computes metrics per subset.

# Generating Plots
```bash
python make_plots.py
```
Figures appear in:
results/figs/
^I'm not to sure if it appears there or not(check other folders aswell if you don't see it

