"""
Baselines: Zero-Shot, Few-Shot, Chain-of-Thought (CoT)
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class PromptingBaselines:
    def __init__(self, model_name="google/flan-t5-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def zero_shot(self, question):
        prompt = f"Answer the question:\n{question}"
        return self._generate(prompt)

    def few_shot(self, question, examples):
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

    def cot(self, question):
        prompt = f"{question}\nLet's think step by step."
        return self._generate(prompt)

    def _generate(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=128)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
