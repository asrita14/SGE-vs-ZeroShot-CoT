"""
Self-Generated Examples (SGE):
1. Ask model to generate examples for the task
2. Use those examples as context for final inference
"""

import re
from typing import List, Tuple

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class SelfGeneratedExamples:
    def __init__(self, model_name: str = "google/flan-t5-base", k: int = 3, max_examples: int = 5):
        """
        k: how many examples to USE as few-shot context
        max_examples: how many examples to GENERATE before filtering
        """
        self.model_name = model_name
        self.k = k
        self.max_examples = max_examples
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # -----------------------
    # Low-level generation
    # -----------------------
    def _generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    # -----------------------
    # Parsing Q/A blocks
    # -----------------------
    def _parse_examples(self, text: str) -> List[Tuple[str, str]]:
        """
        Parse a block of:
          Q: ...
          A: ...

        into a list of (q, a) pairs.
        """
        examples: List[Tuple[str, str]] = []

        pattern = re.compile(r"Q:\s*(.*?)\nA:\s*(.*?)(?=\nQ:|\Z)", re.DOTALL)
        for match in pattern.finditer(text):
            q = match.group(1).strip()
            a = match.group(2).strip()
            if q and a:
                examples.append((q, a))

        return examples

    # -----------------------
    # SGE generation
    # -----------------------
    def generate_examples(self, task_description: str) -> List[Tuple[str, str]]:
        """
        Ask the model to generate self.max_examples Q/A pairs for the task,
        then parse them into a list of (question, answer).
        """
        meta_prompt = (
            f"Generate {self.max_examples} valid input-output examples for the task:\n"
            f"{task_description}\n"
            "Return them in the format:\n"
            "Q: ...\nA: ...\n\n"
        )
        raw = self._generate(meta_prompt)
        return self._parse_examples(raw)

    # -----------------------
    # SGE quality scoring & filtering
    # -----------------------
    def _score_example(self, original_question: str, q: str, a: str, task_description: str) -> float:
        """
        Heuristic quality score for a synthetic example.
        Higher is better.
        """
        if not q or not a:
            return 0.0

        score = 0.0

        # 1. Q and A non-trivial
        if len(q.split()) >= 4:
            score += 0.5
        if 1 <= len(a.split()) <= 15:
            score += 0.5

        # 2. Question-like structure
        if q.strip().endswith("?"):
            score += 0.5

        # 3. On-topic wrt original question / task description (shared tokens)
        orig_tokens = set(re.findall(r"\w+", original_question.lower()))
        task_tokens = set(re.findall(r"\w+", task_description.lower()))
        q_tokens = set(re.findall(r"\w+", q.lower()))

        overlap_orig = len(orig_tokens & q_tokens)
        overlap_task = len(task_tokens & q_tokens)

        if overlap_orig > 0:
            score += min(overlap_orig, 5) * 0.1   # up to +0.5
        if overlap_task > 0:
            score += min(overlap_task, 5) * 0.1   # up to +0.5

        return score

    def _filter_examples(self, original_question: str, examples: List[Tuple[str, str]], task_description: str) -> List[Tuple[str, str]]:
        """
        examples: list of (q, a) pairs
        Returns: top self.k examples by heuristic quality.
        """
        scored = []
        for (q, a) in examples:
            s = self._score_example(original_question, q, a, task_description)
            scored.append((s, (q, a)))

        # keep only examples with positive score
        scored = [pair for pair in scored if pair[0] > 0.0]
        scored.sort(key=lambda x: x[0], reverse=True)

        # return only the example payloads (up to self.k)
        return [ex for _, ex in scored[: self.k]]

    # -----------------------
    # Final inference with SGE
    # -----------------------
    def infer_with_sge(self, question: str, task_description: str) -> str:
        """
        1) Generate self.max_examples synthetic examples
        2) Filter them
        3) Use the best self.k as few-shot context
        4) Answer the target question
        """
        raw_examples = self.generate_examples(task_description)

        filtered_examples = self._filter_examples(
            original_question=question,
            examples=raw_examples,
            task_description=task_description,
        )

        # Fallback if filtering was too strict
        if not filtered_examples:
            filtered_examples = raw_examples

        # If still empty, just do zero-shot-style prompt
        if not filtered_examples:
            final_prompt = f"{task_description}\n\nQ: {question}\nA:"
            return self._generate(final_prompt)

        demo = ""
        for q_ex, a_ex in filtered_examples:
            demo += f"Q: {q_ex}\nA: {a_ex}\n\n"

        final_prompt = demo + f"Q: {question}\nA:"
        return self._generate(final_prompt)
