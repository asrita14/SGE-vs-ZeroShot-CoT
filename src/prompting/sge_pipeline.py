"""
Self-Generated Examples (SGE):
1. Ask model to generate examples for the task
2. Use those examples as context for final inference
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class SelfGeneratedExamples:
    def __init__(self, model_name="google/flan-t5-base", k=3):
        self.k = k
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def generate_examples(self, task_description):
        meta_prompt = (
            f"Generate {self.k} valid input-output examples for the task:\n"
            f"{task_description}\n"
            "Return them in the format:\n"
            "Q: ...\nA: ...\n\n"
        )
        return self._generate(meta_prompt)

    def infer_with_sge(self, question, task_description):
        examples = self.generate_examples(task_description)
        final_prompt = f"{examples}\nQ: {question}\nA:"
        return self._generate(final_prompt)

    def _generate(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=256)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
