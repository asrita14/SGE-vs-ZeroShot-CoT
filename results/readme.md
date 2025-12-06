
# results/ — Evaluation Outputs

This directory contains all experiment outputs: raw predictions, metric summaries, and visualizations.

---

# 📁 metrics/

Contains two types of JSON files:

### 1️⃣ Raw evaluation outputs  

<task>_<run_name>.json

Each entry includes:
- input example  
- gold label  
- predictions from:  
  - zero_shot  
  - few_shot  
  - cot  
  - cot_sc  
  - sge  

### 2️⃣ Metric summaries  
<task>_<run_name>_metrics.json

Metrics:
- GSM8K → EM & MAE  
- BoolQ → accuracy & coverage  
- CSQA → accuracy & coverage  

---

# 📁 figs/

Plots generated using:
```bash
python make_plots.py
```
Includes:

- main bar charts

- difficulty (easy/hard) breakdowns

- radar plots

- heatmaps
