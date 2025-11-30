# SGE vs Zero-Shot vs CoT

This project evaluates how prompting strategies affect reasoning in instruction-tuned language models (FLAN-T5).

## Prompting Strategies
- **Zero-Shot**
- **Few-Shot**
- **Chain-of-Thought (CoT)**
- **Self-Generated Examples (SGE)** ← main focus

## Datasets
- GSM8K (math word problems)
- BoolQ
- CommonsenseQA

## Run Locally
```bash
conda env create -f environment.yml
conda activate prompting-env
python -m src.eval_prompting
