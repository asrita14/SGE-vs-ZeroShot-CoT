<!-- PROJECT HEADER -->
<p align="center">
  <img src="https://img.shields.io/badge/NLP-Project-9146FF?style=for-the-badge">
  <img src="https://img.shields.io/badge/Transformers-FLAN--T5-blue?style=for-the-badge">
  <img src="https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge">
  <br><br>
  <img src="https://img.shields.io/badge/Python-3.10-yellow?style=flat-square">
  <img src="https://img.shields.io/badge/License-MIT-purple?style=flat-square">
</p>

<h1 align="center">🤖 Self-Generated Examples vs Zero-shot and Chain-of-Thought Prompting</h1>

<p align="center">
A comparative study of prompting strategies across GSM8K, BoolQ, and CommonsenseQA using FLAN-T5.<br>
This repo contains the **full codebase**, **evaluation pipeline**, **SGE filtering method**, and **all plots & results** used in the project.
</p>

---

# Overview

This project systematically evaluates five prompting paradigms:

### **Prompting Strategies**
- **Zero-shot**
- **Few-shot (manual)**
- **Chain-of-Thought (CoT)**
- **CoT Self-Consistency (CoT-SC)**
- **Self-Generated Examples (SGE)**
- **Filtered SGE (our contribution)** ✔️

### **Benchmarks**
- **GSM8K** – math word problems  
- **BoolQ** – yes/no reading comprehension  
- **CommonsenseQA** – 5-way multiple-choice  

---

#  What We Contribute
- A complete prompting evaluation framework  
- Implementation of **CoT-Self-Consistency**  
- A novel **SGE-Filtering function** improving stability  
- **Difficulty-aware analysis** (easy vs hard subsets)  
- A reproducibility-focused, simple CLI pipeline  
- Fully-generated plots for poster/report  

---

# 🔧 Installation

```bash
git clone https://github.com/asrita14/SGE-vs-ZeroShot-CoT.git
cd SGE-vs-ZeroShot-CoT
conda create -n prompting-env python=3.10 -y
conda activate prompting-env
pip install -r requirements.txt
If missing, install manually:

pip install transformers datasets torch tqdm matplotlib

Running Experiments
Run both baselines + SGE on any dataset
python -m src.eval_prompting \
  --task boolq \
  --num_examples 200 \
  --run_name bool200_filtered_sc \
  --sge_k 3


Outputs:

results/metrics/boolq_bool200_filtered_sc.json

Compute metrics
python -m src.compute_metrics --task boolq --run_name bool200_filtered_sc


Outputs:

results/metrics/boolq_bool200_filtered_sc_metrics.json

Difficulty analysis (easy vs hard)
python -m src.difficulty_analysis

Generate all plots for poster/report
python make_plots.py


Plots saved to:

results/figs/

🧱 Repository Structure
SGE-vs-ZeroShot-CoT/
│
├── README.md                # Main project README
├── make_plots.py            # All plots used in the poster
├── make_difficulty_heatmaps.py
│
├── src/
│   ├── README.md
│   ├── eval_prompting.py              # Run all prompting strategies
│   ├── compute_metrics.py             # Compute dataset metrics
│   ├── difficulty_analysis.py         # Easy vs Hard evaluation
│   └── prompting/
│       ├── README.md
│       ├── baselines.py               # Zero-shot, Few-shot, CoT, CoT-SC
│       └── sge_pipeline.py            # SGE & Filtered SGE pipeline
│
└── results/
    ├── README.md
    ├── metrics/                       # Raw outputs + metric summaries
    └── figs/                          # All charts

Method Diagram
<p align="center"> <img width="600" src="https://github.com/asrita14/SGE-vs-ZeroShot-CoT/assets/diagram-placeholder.png" alt="SGE Diagram (replace this with your final diagram)"> </p>

SGE-Filtered Pipeline

Task Description → Generate Synthetic Q/A → Score Examples → Keep Top-k → Build Few-shot Prompt → Predict Answer

Key Results (Summary)
Dataset	Best Method	Observation
BoolQ	SGE-Filtered	Slightly > Zero-shot; CoT hurts performance.
CSQA	Zero-shot	Strong priors already inside FLAN-T5.
GSM8K	CoT-SC	Best stability; reduces error by 50%+.

Reproducibility Checklist

✔ Uses only HF datasets
✔ Single model class (FLAN-T5)
✔ All outputs saved as JSON
✔ Metrics reproducible by one command
✔ Figures reproducible by one command

Acknowledgements:

NYU DS-GA 1011 — Natural Language Processing

HuggingFace Transformers

FLAN-T5 Instruction-Tuned Model

📫 Contact

For questions or issues, open a GitHub Issue.
