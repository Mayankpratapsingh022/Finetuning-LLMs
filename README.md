# Finetuning-LLMs
A collection of code and workflows for fine-tuning various Large Language Models (LLMs) on task-specific datasets

![Finetuning_main](https://github.com/user-attachments/assets/299eab30-235b-4c3b-ab17-786dfa6e0eb4)

# Fine-Tuning Large Language Models (LLMs)

This repository provides a practical guide to fine-tuning various open-source LLMs such as LLaMA 2, Mistral, etc., using efficient techniques like LoRA and Quantization. We use tools like **Unsloth** to make training faster and more memory-efficient.

---

## Why Fine-Tuning?

Fine-tuning allows you to adapt a base LLM to your specific domain or task. Instead of relying solely on generic prompts or retrieval systems, you bake in behavior and knowledge directly into the model.

---

## RAG vs Fine-Tuning

| Category         | RAG (Retrieval-Augmented Generation) | Fine-Tuning                  |
|------------------|--------------------------------------|------------------------------|
| Setup            | Easy and dynamic                     | Requires training            |
| Knowledge Type   | Real-time, updatable                 | Static (baked into weights)  |
| Use Case         | General, flexible                    | Task-specific, specialized   |
| Cost             | Depends on context length            | Cheaper for repeated queries |
| Performance      | Limited by prompt size               | More accurate for niche tasks|

Use **RAG** when you want flexibility and up-to-date knowledge. Use **Fine-Tuning** when you need:
- Better performance for domain-specific tasks
- Specialized behavior baked in
- Smaller and faster models at inference time

---

## Benefits of Using Unsloth

- 2x faster training
- 70% less memory usage
- Open-source and beginner-friendly

---

## Finetuning Pipeline Overview

1. **Prepare the Training Data**
2. **Choose a Base Model and Finetuning Method**
3. **Evaluate & Iterate**
4. **Deploy the Finetuned Model**

Note: Fine-tuning is not always a linear process. Proper evaluation and iteration are important to get good results.

---

## Preparing the Dataset

There are 3 ways to get training data:

1. **Existing Datasets** – from HuggingFace, Kaggle, etc.
2. **Manual Curation** – creating examples by hand
3. **Synthetic Generation** – use another LLM to create examples

### Recommended Format

Many models expect data in `JSONL` format like this:

```json
{"messages": [
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "What is the capital of France?"},
  {"role": "assistant", "content": "Paris."}
]}

```
## Choosing a Model

Factors to consider:

- Inference cost and speed
- Task requirements (chat, coding, reasoning, etc.)
- Available hardware (VRAM)

Popular open-source models:
- LLaMA 2
- Mistral
- Phi
- Falcon
- Gemma

---

## Finetuning Techniques

### 1. Full Finetuning

- Updates all weights
- Requires a lot of GPU memory
- Higher compute cost

> Analogy: Rewriting the entire book.

### 2. LoRA (Low-Rank Adaptation)

- Freezes original weights
- Adds small adapter layers
- Much faster and memory-efficient

> Analogy: Adding sticky notes to a book for extra info.

### 3. QLoRA

- Combines quantization + LoRA
- Enables training on low-VRAM setups
- Ideal for 12GB to 24GB GPUs

---

## LoRA Weight Formula

Original:  
  `X = W * Y`  
LoRA:  
  `X = (W + A * B) * Y`

Where:
- `W` = frozen base weight
- `A`, `B` = trainable low-rank matrices

---

## Quantization

Reduces model weight precision to save memory:

| Type      | Bit | Memory |
|-----------|-----|--------|
| float32   | 32  | 4 bytes |
| float16   | 16  | 2 bytes |
| int8      | 8   | 1 byte |
| 4-bit     | 4   | 0.5 byte |

Techniques:
1. **Post-Training Quantization (PTQ)** – Quantize after training
2. **Quantization Aware Training (QAT)** – Train with quantization effects

---

## Finetuning Tools & Platforms

| Tool/Platform    | Type         | Notes                                     |
|------------------|--------------|-------------------------------------------|
| Unsloth          | Open Source  | Fast, memory-efficient LoRA + QLoRA       |
| LLaMA Factory    | Open Source  | Supports full and PEFT finetuning         |
| Together AI      | Hosted       | Provides APIs and finetuning platform     |
| Fireworks AI     | Hosted       | LoRA-based finetuning service             |
| RunPod, Modal    | Self-hosted  | Full control over training & deployment   |

---


---

## Summary

Fine-tuning is an essential skill to make LLMs work better for your domain. With tools like Unsloth and LoRA, you can fine-tune even large models on consumer hardware.

> Fine-tuning is most effective when RAG fails to provide depth or when inference costs need to be optimized.

---


