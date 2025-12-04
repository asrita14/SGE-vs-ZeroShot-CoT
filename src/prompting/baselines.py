"""
Baselines: Zero-Shot, Few-Shot, Chain-of-Thought (CoT) and CoT Self-Consistency (CoT-SC)
"""

from collections import Counter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class PromptingBaselines:
    def __init__(self, model_name: str = "google/flan-t5-base"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def zero_shot(self, question: str) -> str:
        prompt = f"Answer the question:\n{question}"
        return self._generate(prompt)

    def few_shot(self, question: str, examples):
        """
        examples = [
            ("input1", "output1"),
            ("input2", "output2")
        ]
        """
        demo = ""
        for q, a in examples:
            demo += f"Q: {q}\nA: {a}\n\n"
        prompt = demo + f"Q: {question}\nA:"
        return self._generate(prompt)

    def cot(self, question: str) -> str:
        prompt = f"{question}\nLet's think step by step."
        return self._generate(prompt)

    def cot_self_consistency(self, question: str, n: int = 5, temperature: float = 0.7) -> str:
        """
        Chain-of-Thought with self-consistency:
        - Generate n CoT answers with sampling
        - Return the majority-vote full answer string
        """
        prompt = f"{question}\nLet's think step by step."
        answers = []

        for _ in range(n):
            ans = self._generate(
                prompt,
                max_new_tokens=128,
                do_sample=True,
                temperature=temperature,
            )
            answers.append(ans)

        if not answers:
            return ""

        counts = Counter(answers)
        best_answer, _ = counts.most_common(1)[0]
        return best_answer

    def _generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        do_sample: bool = False,
        temperature: float = 1.0,
    ) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
