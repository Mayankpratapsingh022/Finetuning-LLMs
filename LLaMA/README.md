
# 🦙 Fine-Tuning LLaMA 2 7B on Custom Dataset using Unsloth + LoRA + Quantization

This project demonstrates how to fine-tune the **LLaMA 2 7B** model on a **custom raw text dataset** using **Unsloth**, **LoRA**, and **4-bit Quantization** (QLoRA) techniques. We use a modular approach that can be adapted to any other dataset.

---
![Llama](https://github.com/user-attachments/assets/a0c2c9f3-5541-4bef-883f-b352eecb3add)


##   Dataset

We use two datasets in this tutorial:

1. **[FineTome-100k](https://huggingface.co/datasets/mlabonne/FineTome-100k)** — for GPT-like behavior.
2. **Hawaiian Wildfire** — a plain text dataset, demonstrating how to use your own data.

---

## Libraries Used

Install the required libraries:

```bash
pip install peft accelerate bitsandbytes transformers datasets GPUtil
```

Key libraries:

- `transformers` (HuggingFace)
- `datasets` (HuggingFace)
- `bitsandbytes` (for quantization)
- `peft` (Parameter-Efficient Fine-Tuning)
- `GPUtil` (to monitor GPU usage)

---

##  Finetuning Setup

### ✅ GPU Check

We check GPU availability and set CUDA configurations:

```python
import torch, GPUtil, os
GPUtil.showUtilization()

if torch.cuda.is_available():
    print("✅ GPU Available")
else:
    print("❌ Using CPU")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

---

##  Load Base Model with Quantization

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    "unsloth/llama-2-7b",
    quantization_config=bnb_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("unsloth/llama-2-7b")
```

---

## Apply LoRA using PEFT

```python
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
```

---

##  Load Your Custom Dataset

Here we show how to use a simple text file (`hawaiian_wildfire.txt`) as your dataset.

```python
from datasets import Dataset

with open("hawaiian_wildfire.txt", "r") as f:
    data = f.read()

dataset = Dataset.from_dict({"text": [data]})
```

---

## Tokenization and Formatting

```python
def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize)
```

---

## Training

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="llama-custom-lora",
    per_device_train_batch_size=1,
    num_train_epochs=2,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    fp16=True,
    optim="paged_adamw_8bit",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)
trainer.train()
```

---

## Inference

```python
model.eval()
input_text = "The wildfire in Hawaii caused"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

---

## Theory: LoRA, QLoRA & PEFT

- **LoRA**: Freeze original weights and train low-rank matrices.
- **QLoRA**: Quantize model (4-bit) + apply LoRA.
- **PEFT**: Framework for efficient fine-tuning (e.g., LoRA, Prompt Tuning).
- **Quantization**:
  - Reduces precision (float32 → int8/float4)
  - Saves memory and speeds up training.

---

## Tools Like Unsloth & LLaMA-Factory

Instead of manual setup, you can use:

-  **[Unsloth](https://github.com/unslothai/unsloth)** — Simplifies QLoRA finetuning
-  **[LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory)** — Another plug-and-play trainer

---

