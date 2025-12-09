import json
import re
from pathlib import Path
from typing import Dict, List, Any, Callable, Tuple

import argparse


# --------------------------
# 1. Answer extraction helpers (same logic as in analyze_results)
# --------------------------

def extract_gsm8k_answer(text: str) -> str:
    """Extract final numeric answer from GSM8K-style solution."""
    if text is None:
        return ""
    nums = re.findall(r"-?\d+", text)
    if not nums:
        return text.strip()
    return nums[-1].lstrip("+").strip()


def extract_boolq_answer(text: str) -> str:
    """Normalize yes/no answers from text."""
    if text is None:
        return ""
    t = text.lower()
    if "yes" in t and "no" not in t:
        return "yes"
    if "no" in t and "yes" not in t:
        return "no"
    return t.strip().split()[0]


def extract_csqa_answer(text: str) -> str:
    """Extract option letter A-E from CSQA-style answer text."""
    if text is None:
        return ""
    t = text.strip().upper()

    m = re.search(r"\b([A-E])\b", t)
    if m:
        return m.group(1)

    m = re.search(r"OPTION\s+([A-E])", t)
    if m:
        return m.group(1)

    m = re.search(r"ANSWER[:\s]+([A-E])", t)
    if m:
        return m.group(1)

    for ch in t:
        if ch in "ABCDE":
            return ch

    return t[:1]  # fallback


def get_extract_fn(task: str) -> Callable[[str], str]:
    if task == "gsm8k":
        return extract_gsm8k_answer
    elif task == "boolq":
        return extract_boolq_answer
    elif task == "commonsense_qa":
        return extract_csqa_answer
    else:
        return lambda x: ("" if x is None else str(x).strip())


# --------------------------
# 2. Load JSON
# --------------------------

def load_results(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


# --------------------------
# 3. Find interesting error patterns
# --------------------------

def find_error_cases(
    data: Dict[str, Any],
    max_per_pattern: int = 5,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Returns a dict mapping pattern name -> list of examples.

    Patterns:
      - zs_wrong_sir_right
      - cot_wrong_sir_right
      - zs_wrong_sge_right
    """
    cfg = data.get("config", {})
    results = data.get("results", [])

    task = cfg.get("task", "unknown")
    extract_fn = get_extract_fn(task)

    patterns: Dict[str, List[Dict[str, Any]]] = {
        "zs_wrong_sir_right": [],
        "cot_wrong_sir_right": [],
        "zs_wrong_sge_right": [],
    }

    for ex in results:
        gold_raw = ex.get("gold", None)
        if gold_raw is None:
            continue
        gold = extract_fn(gold_raw)

        zs = ex.get("zero_shot", None)
        fs = ex.get("few_shot", None)
        cot = ex.get("cot", None)
        cot_sc = ex.get("cot_sc", None)
        cot_sir = ex.get("cot_sir", None)
        sge = ex.get("sge", None)

        zs_n = extract_fn(zs) if zs is not None else ""
        cot_n = extract_fn(cot) if cot is not None else ""
        cot_sir_n = extract_fn(cot_sir) if cot_sir is not None else ""
        sge_n = extract_fn(sge) if sge is not None else ""

        # Pattern 1: zero-shot wrong, CoT-SIR right
        if (
            zs is not None
            and cot_sir is not None
            and zs_n != gold
            and cot_sir_n == gold
            and len(patterns["zs_wrong_sir_right"]) < max_per_pattern
        ):
            patterns["zs_wrong_sir_right"].append(ex)

        # Pattern 2: CoT wrong, CoT-SIR right
        if (
            cot is not None
            and cot_sir is not None
            and cot_n != gold
            and cot_sir_n == gold
            and len(patterns["cot_wrong_sir_right"]) < max_per_pattern
        ):
            patterns["cot_wrong_sir_right"].append(ex)

        # Pattern 3: zero-shot wrong, SGE right
        if (
            zs is not None
            and sge is not None
            and zs_n != gold
            and sge_n == gold
            and len(patterns["zs_wrong_sge_right"]) < max_per_pattern
        ):
            patterns["zs_wrong_sge_right"].append(ex)

        # Early exit if we filled all
        if all(len(v) >= max_per_pattern for v in patterns.values()):
            break

    return patterns


# --------------------------
# 4. Pretty printing
# --------------------------

def short_text(x: str, max_len: int = 200) -> str:
    if x is None:
        return ""
    x = str(x).strip().replace("\n", " ")
    if len(x) > max_len:
        return x[: max_len - 3] + "..."
    return x


def print_examples_for_pattern(
    pattern_name: str,
    examples: List[Dict[str, Any]],
    task: str,
):
    print("\n" + "=" * 80)
    print(f"PATTERN: {pattern_name} (n={len(examples)})")
    print("=" * 80)

    for i, ex in enumerate(examples):
        print(f"\n--- Example {i+1} ---")

        if task == "gsm8k":
            q = ex.get("question", "")
            print(f"QUESTION: {q}\n")

        elif task == "boolq":
            q = ex.get("question", "")
            passage = ex.get("passage", "")
            print(f"PASSAGE: {short_text(passage)}")
            print(f"QUESTION: {q}\n")

        elif task == "commonsense_qa":
            q = ex.get("question", "")
            choices = ex.get("choices", "")
            print(f"QUESTION: {q}")
            print(f"CHOICES:\n{choices}\n")

        else:
            # generic
            q = ex.get("question", ex.get("input", ""))
            print(f"INPUT: {q}\n")

        print(f"GOLD:       {ex.get('gold', '')}\n")
        print(f"ZERO-SHOT:  {short_text(ex.get('zero_shot', ''))}\n")
        print(f"FEW-SHOT:   {short_text(ex.get('few_shot', ''))}\n")
        print(f"COT:        {short_text(ex.get('cot', ''))}\n")
        print(f"COT-SC:     {short_text(ex.get('cot_sc', ''))}\n")
        print(f"COT-SIR:    {short_text(ex.get('cot_sir', ''))}\n")
        print(f"SGE:        {short_text(ex.get('sge', ''))}\n")


# --------------------------
# 5. Optional: save to TSV
# --------------------------

def save_patterns_to_tsv(
    path: Path,
    patterns: Dict[str, List[Dict[str, Any]]],
    task: str,
):
    """Flatten patterns to a TSV file so you can inspect in Excel / Sheets."""
    import csv

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "pattern",
                "dataset",
                "question_or_input",
                "passage",
                "choices",
                "gold",
                "zero_shot",
                "few_shot",
                "cot",
                "cot_sc",
                "cot_sir",
                "sge",
            ]
        )
        for pattern_name, examples in patterns.items():
            for ex in examples:
                if task == "boolq":
                    question = ex.get("question", "")
                    passage = ex.get("passage", "")
                    choices = ""
                elif task == "commonsense_qa":
                    question = ex.get("question", "")
                    passage = ""
                    choices = ex.get("choices", "")
                else:
                    question = ex.get("question", ex.get("input", ""))
                    passage = ""
                    choices = ""

                writer.writerow(
                    [
                        pattern_name,
                        ex.get("dataset", task),
                        question,
                        passage,
                        choices,
                        ex.get("gold", ""),
                        ex.get("zero_shot", ""),
                        ex.get("few_shot", ""),
                        ex.get("cot", ""),
                        ex.get("cot_sc", ""),
                        ex.get("cot_sir", ""),
                        ex.get("sge", ""),
                    ]
                )


# --------------------------
# 6. Main
# --------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results_file",
        type=str,
        required=True,
        help="Path to JSON results (e.g., results/metrics/gsm8k_sir100.json)",
    )
    ap.add_argument(
        "--max_per_pattern",
        type=int,
        default=5,
        help="Max examples to show per error pattern.",
    )
    ap.add_argument(
        "--save_tsv",
        type=str,
        default=None,
        help="If set, save all selected examples to this TSV path.",
    )
    args = ap.parse_args()

    path = Path(args.results_file)
    data = load_results(path)
    cfg = data.get("config", {})
    task = cfg.get("task", "unknown")

    patterns = find_error_cases(data, max_per_pattern=args.max_per_pattern)

    # Print nicely to stdout
    for pattern_name, exs in patterns.items():
        if not exs:
            continue
        print_examples_for_pattern(pattern_name, exs, task)

    # Optionally save to TSV
    if args.save_tsv is not None:
        out_path = Path(args.save_tsv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_patterns_to_tsv(out_path, patterns, task)
        print(f"\nSaved TSV with error cases → {out_path}")


if __name__ == "__main__":
    main()