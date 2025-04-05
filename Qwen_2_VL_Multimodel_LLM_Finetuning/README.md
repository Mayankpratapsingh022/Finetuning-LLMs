

# Qwen2-VL: Equation Image → LaTeX with LoRA + Unsloth

[Final Fine-tuned Model on Hugging Face ](https://huggingface.co/Mayank022/qwen2-vl-finetuned-Image-to-LaTeX)


Fine-tune [Qwen2-VL](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct), a Vision-Language model, to convert equation images into LaTeX code using the [Unsloth](https://github.com/unslothai/unsloth) framework and LoRA adapters.

## Project Objective

Train an Equation-to-LaTeX transcriber using a pre-trained multimodal model. The model learns to read rendered math equations and generate corresponding LaTeX.

![qwen](https://github.com/user-attachments/assets/102ce3ec-cd88-435d-b1af-e1ecbcc1fb1c)


---


## Dataset

- [`unsloth/LaTeX_OCR`](https://huggingface.co/datasets/unsloth/LaTeX_OCR) – Image-LaTeX pairs of printed mathematical expressions.
- ~68K train / 7K test samples.
- Example:
  - Image: ![image](https://github.com/user-attachments/assets/e0d87582-7ba4-4e59-8f00-fd8f6c0f862d)
  - Target: `R - { \frac { 1 } { 2 } } ( \nabla \Phi ) ^ { 2 } - { \frac { 1 } { 2 } } \nabla ^ { 2 } \Phi = 0 .`

---

## Tech Stack

| Component | Description |
|----------|-------------|
| Qwen2-VL | Multimodal vision-language model (7B) by Alibaba |
| Unsloth | Fast & memory-efficient training |
| LoRA (via PEFT) | Parameter-efficient fine-tuning |
| 4-bit Quantization | Enabled by `bitsandbytes` |
| Datasets, HF Hub | For loading/saving models & datasets |

---

## Setup

```bash
pip install unsloth unsloth_zoo peft trl datasets accelerate bitsandbytes xformers==0.0.29.post3 sentencepiece protobuf hf_transfer triton
```

---

## Training (Jupyter Notebook)

Refer to: `Qwen2__VL_image_to_latext.ipynb`

Steps:
1. Load Qwen2-VL (`load_in_4bit=True`)
2. Load dataset via `datasets.load_dataset("unsloth/LaTeX_OCR")`
3. Apply LoRA adapters
4. Use `SFTTrainer` from Unsloth to fine-tune
5. Save adapters or merged model

LoRA rank used: `r=16`  
LoRA alpha: `16`

---

## Inference

```python
from PIL import Image
image = Image.open("equation.png")
prompt = "Write the LaTeX representation for this image."
inputs = tokenizer(image, tokenizer.apply_chat_template([("user", prompt)], add_generation_prompt=True), return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

---

## Evaluation

- Exact Match Accuracy: ~90%+
- Strong generalization to complex equations and symbols

---

## Results

| Metric           | Value         |
|------------------|---------------|
| Exact Match      | ~90–92%       |
| LoRA Params      | ~<1% of model |
| Training Time    | ~20–40 mins on A100 |
| Model Size       | 7B (4-bit)    |

---

## Future Work

- Extend to handwritten formulas (e.g., CROHME dataset)
- Add LaTeX syntax validation or auto-correction
- Build a lightweight Gradio/Streamlit interface for demo

---

## Folder Structure

```
.
├── Qwen2__VL_image_to_latext.ipynb   # Training Notebook
├── output/                           # Saved fine-tuned model
└── README.md
```

---


