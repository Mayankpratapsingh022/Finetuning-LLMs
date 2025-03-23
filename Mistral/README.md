# Finetuning Mistral-7B with qLoRA & PEFT on SAMSum Dataset

This project demonstrates how to fine-tune the powerful [Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) model for text summarization using **Parameter-Efficient Fine-Tuning (PEFT)** and **Quantization** via **qLoRA**.

**Check out the fine-tuned Mistral model on Hugging Face:**  
[Mayank022/mistral-finetuned-samsum](https://huggingface.co/Mayank022/mistral-finetuned-samsum)


## 📌 Project Highlights

- **Model**: [`Mistral-7B-Instruct-v0.1`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)
- **Technique**: PEFT + LoRA + Quantization → `qLoRA`
-  **Dataset**: [SAMSum Dialogue Summarization Dataset](https://huggingface.co/datasets/Samsung/samsum)
-  **Libraries**: `transformers`, `peft`, `bitsandbytes`, `trl`
-  **Hardware**: Fine-tuned on NVIDIA A100 (80GB) 

## 🛠️ Setup

```bash
pip install accelerate peft bitsandbytes git+https://github.com/huggingface/transformers trl py7zr auto-gptq optimum
```

Login to Hugging Face:
```python
from huggingface_hub import notebook_login
notebook_login()
```

---

##  Finetuning Script Summary

### 1. Load Tokenizer & Model with 4-bit Quantization

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.1",
    quantization_config=quant_config,
    device_map="auto"
)
```

---

### 2. Apply LoRA using PEFT

```python
from peft import LoraConfig, get_peft_model

peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"]
)

model = get_peft_model(model, peft_config)
```

---

### 3. Define Training Arguments

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="mistral-finetuned-samsum",
    per_device_train_batch_size=8,
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_steps=100,
    fp16=True,
    save_strategy="epoch",
    push_to_hub=True
)
```

---

### 4. Fine-tune with `SFTTrainer`

```python
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    train_dataset=data,
    tokenizer=tokenizer,
    peft_config=peft_config,
    args=training_args,
)

trainer.train()
```

---

## 💾 Save & Reload the Fine-Tuned Model

```python
# Save
!cp -r mistral-finetuned-samsum /content/drive/MyDrive/model

# Reload
from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained(
    "/content/drive/MyDrive/model/mistral-finetuned-samsum",
    torch_dtype=torch.float16,
    device_map="cuda"
)
```

---

## 🧠 PEFT Recap

- Train only a **subset** of model parameters (e.g., Q and V projections).
- Dramatically reduces training time and GPU memory.
- Ideal for **domain adaptation**, **instruction tuning**, and **low-resource training**.

---

## 💡 Useful Terms

| Term        | Description                          |
|-------------|--------------------------------------|
| **PEFT**     | Parameter-Efficient Fine-Tuning      |
| **LoRA**     | Low-Rank Adaptation for Transformers |
| **Quantization** | Reducing precision to 8/4/2-bit  |
| **qLoRA**    | Quantized LoRA                       |
| **SFT**      | Supervised Fine-Tuning               |

---

## 📎 References

- [Mistral-7B on HuggingFace](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [PEFT Library](https://github.com/huggingface/peft)
- [GGML, GPTQ, AutoGPTQ](https://github.com/qwopqwop200/AutoGPTQ)
