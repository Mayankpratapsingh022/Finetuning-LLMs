# Instruction Fine-Tuning of GPT using OpenAI API

Fine-tuning GPT models for **instruction-based tasks** can significantly improve performance on **customized domains**, such as customer support, domain-specific Q&A, or chatbot behavior shaping.

This repository contains a guide and implementation for **instruction fine-tuning** of GPT models using the **OpenAI API**.

---

## What is Instruction Fine-tuning?

Instruction fine-tuning involves training an LLM (like GPT) using **instructional prompts** that include a structured interaction between a system, user, and assistant.

This is particularly useful when:

- You want the model to follow **specific behavior or tone**
- You need it to answer based on **proprietary/internal data**
- You want to replicate **domain-specific patterns**

---

##  Data Format: JSONL

Each prompt/response pair must be formatted as a `.jsonl` file:

```json
{"messages":[
  {"role":"system", "content":"You are a helpful assistant for answering product queries."},
  {"role":"user", "content":"Can you tell me about the warranty for product X?"},
  {"role":"assistant", "content":"Sure! Product X comes with a 2-year replacement warranty."}
]}
```

###  Requirements

- File extension: `.jsonl` (JSON Lines)
- At least **10 examples** (rows), else OpenAI throws an error.
- Each entry must include **system**, **user**, and **assistant** messages.
- You can use this format for **custom support** systems by including past inquiries and responses.

---


##  Steps to Fine-Tune GPT

1. **Prepare JSONL file**
2. **Upload File to OpenAI**
   ```bash
   openai file upload --file data.jsonl --purpose fine-tune
   ```
3. **Start Fine-Tuning Job**
   ```bash
   openai fine_tunes.create -t <file-id> -m gpt-3.5-turbo
   ```
4. **Monitor the Training**
   ```bash
   openai fine_tunes.follow -i <fine-tune-id>
   ```

---

You can refer the ipynb file for the code
