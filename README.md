<p align="center">
  <img src="https://img.shields.io/badge/NLP-Project-9146FF?style=for-the-badge">
  <img src="https://img.shields.io/badge/Transformers-FLAN--T5-blue?style=for-the-badge">
  <img src="https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge">
</p>

<h1 align="center"> Self-Generated Examples vs Zero-shot and CoT Prompting</h1>

<p align="center">
A comparative study of prompting strategies across several reasoning benchmarks using FLAN-T5.
</p>

---

#  Overview

This project evaluates how different prompting strategies behave across reasoning tasks:

### **Prompting Methods**
- **Zero-shot**  
- **Few-shot**, manually written  
- **Chain-of-Thought (CoT)**  
- **CoT Self-Consistency (CoT-SC)**  
- **Self-Generated Examples (SGE)**  
- **Filtered SGE (our new method)** ✔️

### **Datasets**
- **GSM8K** — math word problems  
- **BoolQ** — yes/no QA  
- **CommonsenseQA** — 5-way multiple choice  

We measure performance, stability, and behavior across:
- **task types**
- **difficulty levels (easy vs hard)**  
- **SGE quality (filtered vs unfiltered)**

---

#  Key Contributions

✔️ Introduce **Filtered SGE**, a scoring-based filtering mechanism for synthetic examples  
✔️ Add **CoT-Self-Consistency** for more stable math reasoning  
✔️ Design a unified evaluation pipeline  
✔️ Provide difficulty-aware analysis across datasets  
✔️ Produce reproducible metrics and plots  

---

# Installation

```bash
git clone https://github.com/asrita14/SGE-vs-ZeroShot-CoT.git
cd SGE-vs-ZeroShot-CoT
conda create -n prompting-env python=3.10 -y
conda activate prompting-env
pip install -r requirements.txt
```
# Repository Structure
```graphql
SGE-vs-ZeroShot-CoT/
│
├── README.md                 # High-level overview (this file)
├── make_plots.py             # Generates all project figures
│
├── src/
│   ├── README.md             # How to run experiments, metrics, analysis
│   ├── eval_prompting.py     # Main evaluation driver
│   ├── compute_metrics.py    # Metrics computation
│   ├── difficulty_analysis.py# Easy/Hard tagging + analysis
│   └── prompting/
│       ├── README.md         # Explanation of prompting methods
│       ├── baselines.py      # Zero-shot, Few-shot, CoT, CoT-SC
│       └── sge_pipeline.py   # SGE + SGE filtering implementation
│
└── results/
    ├── README.md             # What the JSONs / figs mean
    ├── metrics/              # Raw & summarized JSON outputs
    └── figs/                 # Plots used in report + poster
```
# High-Level Results

| Dataset   | Best Method  | Observation                                          |
| --------- | ------------ | ---------------------------------------------------- |
| **BoolQ** | SGE-Filtered | Slightly improves over Zero-shot; CoT harms accuracy |
| **CSQA**  | Zero-shot    | Strong commonsense priors already in FLAN-T5         |
| **GSM8K** | CoT-SC       | Best stability; large MAE reduction                  |

# Reproducibility Checklist

- **Uses only HF datasets**
- **Single model class (FLAN-T5)**
- **All outputs saved as JSON**
- **Metrics reproducible by one command**
- **Figures reproducible by one command**

# Acknowledgements

- **NYU DS-GA 1011 — Natural Language Processing**
- **HuggingFace Transformers**
- **FLAN-T5 Instruction-Tuned Model**

# Contact

For issues or reproducibility questions, open a GitHub issue.
