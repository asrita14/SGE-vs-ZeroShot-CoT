"""
sir_only_accuracy.py

Run Self-Improving Reasoning (SIR) on GSM8K, BoolQ, and CommonsenseQA
and print ONLY accuracy for each dataset.

Usage:
    python sir_only_accuracy.py --model_name google/flan-t5-large --k_eval 200
"""

import argparse
import random
import re
import time
from dataclasses import dataclass

import torch
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


DATASETS = ["gsm8k", "boolq", "commonsense_qa"]


@dataclass
class EvalResult:
    num_examples: int
    num_correct: int

    @property
    def accuracy(self) -> float:
        return self.num_correct / self.num_examples if self.num_examples > 0 else 0.0


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------- dataset loaders ----------

def load_gsm8k(split: str = "test"):
    return load_dataset("gsm8k", "main", split=split)


def load_boolq(split: str = "validation"):
    return load_dataset("boolq", split=split)


def load_csqa(split: str = "validation"):
    return load_dataset("commonsense_qa", split=split)


# ---------- normalization helpers ----------

NUM_RE = re.compile(r"[-+]?\d*\.?\d+")


def extract_first_number(text: str) -> str:
    m = NUM_RE.search(text)
    return m.group(0) if m else text.strip()


def normalize_yes_no(text: str) -> str:
    text = text.strip().lower()
    if "yes" in text:
        return "yes"
    if "no" in text:
        return "no"
    return text


def normalize_choice_letter(text: str) -> str:
    text = text.strip().upper()
    m = re.search(r"[A-E]", text)
    return m.group(0) if m else text


def gold_answer(example, dataset_name: str) -> str:
    if dataset_name == "gsm8k":
        return extract_first_number(example["answer"])
    elif dataset_name == "boolq":
        return "yes" if example["answer"] else "no"
    elif dataset_name == "commonsense_qa":
        return example["answerKey"].strip().upper()
    else:
        raise ValueError(dataset_name)


def extract_final_answer(text: str, dataset_name: str) -> str:
    # look for "Final answer:"
    m = re.search(r"Final answer\s*[:\-]\s*(.+)", text, flags=re.IGNORECASE)
    if m:
        candidate = m.group(1).strip()
    else:
        candidate = text.strip()

    if dataset_name == "gsm8k":
        return extract_first_number(candidate)
    elif dataset_name == "boolq":
        return normalize_yes_no(candidate)
    elif dataset_name == "commonsense_qa":
        return normalize_choice_letter(candidate)
    else:
        return candidate


# ---------- SGE context for SIR ----------

def generate_sge_examples(
    dataset_name: str,
    model,
    tokenizer,
    device,
    num_examples: int = 3,
    max_new_tokens: int = 128,
) -> str:
    if dataset_name == "gsm8k":
        meta_prompt = (
            "Generate {k} examples of math word problems with their numeric answers. "
            "Format each as:\nQuestion: <question>\nAnswer: <answer>\n\n"
        )
    elif dataset_name == "boolq":
        meta_prompt = (
            "Generate {k} examples of reading comprehension yes/no questions. "
            "Format each as:\nPassage: <short passage>\nQuestion: <question>\nAnswer: <yes or no>\n\n"
        )
    elif dataset_name == "commonsense_qa":
        meta_prompt = (
            "Generate {k} examples of commonsense multiple choice questions. "
            "Use options A to E. Format each as:\n"
            "Question: <question>\n"
            "Choices: (A) <optA> (B) <optB> (C) <optC> (D) <optD> (E) <optE>\n"
            "Answer: <letter>\n\n"
        )
    else:
        return ""

    prompt = meta_prompt.format(k=num_examples)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text + "\n\n"


# ---------- SIR prompts ----------

def build_prompt_gsm8k(example, sge_examples: str = "") -> str:
    question = example["question"].strip()
    base_instruction = (
        "You are a helpful math tutor. Solve the following math word problem. "
        "Show your reasoning, then give the final numeric answer clearly."
    )
    ctx = sge_examples or ""
    reasoning_suffix = " Let's think step by step."
    prompt = f"{base_instruction}\n\n{ctx}Question: {question}\nAnswer:{reasoning_suffix}"
    return prompt


def build_prompt_boolq(example, sge_examples: str = "") -> str:
    passage = example["passage"].strip()
    question = example["question"].strip()
    base_instruction = (
        "You will be given a passage and a yes/no question about it. "
        "Explain your reasoning, then answer with 'yes' or 'no'."
    )
    ctx = sge_examples or ""
    reasoning_suffix = " Let's think step by step."
    prompt = (
        f"{base_instruction}\n\n{ctx}Passage: {passage}\n"
        f"Question: {question}\nAnswer:{reasoning_suffix}"
    )
    return prompt


def build_prompt_csqa(example, sge_examples: str = "") -> str:
    question = example["question"].strip()
    choices = example["choices"]["text"]
    labels = example["choices"]["label"]

    options_str = " ".join(
        [f"({lab}) {txt}" for lab, txt in zip(labels, choices)]
    )

    base_instruction = (
        "You will be given a question with multiple choice answers. "
        "Explain your reasoning and then answer with only the letter of the correct option (A, B, C, D, or E)."
    )

    ctx = sge_examples or ""
    reasoning_suffix = " Let's think step by step."
    prompt = (
        f"{base_instruction}\n\n{ctx}Question: {question}\n"
        f"Choices: {options_str}\nAnswer:{reasoning_suffix}"
    )
    return prompt


def build_prompt(dataset_name: str, example, sge_examples: str = "") -> str:
    if dataset_name == "gsm8k":
        return build_prompt_gsm8k(example, sge_examples)
    elif dataset_name == "boolq":
        return build_prompt_boolq(example, sge_examples)
    elif dataset_name == "commonsense_qa":
        return build_prompt_csqa(example, sge_examples)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


# ---------- SIR core ----------

def run_sir_pass(
    base_prompt: str,
    model,
    tokenizer,
    device,
    max_new_tokens: int = 128,
) -> str:
    # 1st pass
    inputs = tokenizer(base_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out1 = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    answer1 = tokenizer.decode(out1[0], skip_special_tokens=True)

    # 2nd pass: critique + refine
    critique_prompt = (
        f"{base_prompt}\n\nProposed solution:\n{answer1}\n\n"
        "Check carefully if the proposed solution is correct. "
        "If it is wrong, explain the mistake and provide the correct final answer. "
        "End your response with 'Final answer: <answer>'."
    )

    inputs2 = tokenizer(critique_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out2 = model.generate(
            **inputs2,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    full = tokenizer.decode(out2[0], skip_special_tokens=True)
    return full


# ---------- evaluation ----------

def evaluate_sir_on_dataset(
    dataset_name: str,
    dataset,
    model,
    tokenizer,
    device,
    max_new_tokens: int = 128,
    k_eval: int = 200,
) -> EvalResult:
    print(f"Generating SGE context for SIR on {dataset_name}...")
    sge_examples = generate_sge_examples(
        dataset_name, model, tokenizer, device
    )

    num_correct = 0
    n = min(k_eval, len(dataset))

    for i in range(n):
        ex = dataset[i]
        prompt = build_prompt(dataset_name, ex, sge_examples)

        _ = time.perf_counter()
        text = run_sir_pass(
            prompt, model, tokenizer, device, max_new_tokens=max_new_tokens
        )
        _ = time.perf_counter()

        pred = extract_final_answer(text, dataset_name)
        gold = gold_answer(ex, dataset_name)

        if dataset_name == "gsm8k":
            try:
                if float(pred) == float(gold):
                    num_correct += 1
            except Exception:
                if pred.strip() == gold.strip():
                    num_correct += 1
        else:
            if pred == gold:
                num_correct += 1

        if (i + 1) % 10 == 0:
            print(f"[{dataset_name} / SIR] {i + 1}/{n} examples processed...")

    return EvalResult(num_examples=n, num_correct=num_correct)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/flan-t5-base",
        help="HF model name (e.g., google/flan-t5-large)",
    )
    parser.add_argument(
        "--k_eval",
        type=int,
        default=200,
        help="Number of examples per dataset for evaluation.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Max new tokens for generation.",
    )
    args = parser.parse_args()

    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model {args.model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    model.to(device)
    model.eval()

    print("Loading datasets...")
    gsm8k_ds = load_gsm8k("test")
    boolq_ds = load_boolq("validation")
    csqa_ds = load_csqa("validation")

    dataset_map = {
        "gsm8k": gsm8k_ds,
        "boolq": boolq_ds,
        "commonsense_qa": csqa_ds,
    }

    for ds_name in DATASETS:
        ds = dataset_map[ds_name]
        print(f"\n=== Evaluating {ds_name} with SIR (accuracy only) ===")
        res = evaluate_sir_on_dataset(
            ds_name,
            ds,
            model,
            tokenizer,
            device,
            max_new_tokens=args.max_new_tokens,
            k_eval=args.k_eval,
        )
        print(
            f"{ds_name} | SIR accuracy: {res.accuracy:.4f} "
            f"({res.num_correct}/{res.num_examples})"
        )


if __name__ == "__main__":
    main()