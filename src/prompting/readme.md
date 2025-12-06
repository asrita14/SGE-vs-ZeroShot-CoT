# prompting/ — Prompting Method Implementations

This module defines all prompting strategies used in the project.

---

## baselines.py

Implements:

### **Zero-shot**
Direct question → answer.

### **Few-shot**
Manual 3-shot demonstration examples.

### **Chain-of-Thought (CoT)**
Appends:
"Let's think step by step."

### **CoT Self-Consistency**
- Samples multiple CoT outputs  
- Extracts answers  
- Returns the majority vote  
Improves stability on GSM8K.

---

## sge_pipeline.py

Implements Self-Generated Examples (SGE) + our new **Filtered SGE**.

Pipeline:

1. Generate >k synthetic examples  
2. Score each example using heuristics:
   - valid format (“Q: … A: …”)  
   - non-trivial question length  
   - short, clean answer  
   - ends with “?”  
   - keyword overlap with original question  
   - keyword overlap with task description  
3. Keep only the top-k examples  
4. Build a few-shot prompt  
5. Predict final answer  

Filtered SGE improves accuracy and consistency, especially on BoolQ and GSM8K-hard subsets.
