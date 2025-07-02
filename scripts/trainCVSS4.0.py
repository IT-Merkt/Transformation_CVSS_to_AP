from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling, BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from datasets import load_dataset
import torch
from pathlib import Path

SYSTEM_PROMPT="""You are a cybersecurity expert specialized in automotive risk assessment following ISO/SAE 21434 and TARA methodology.

Your task is to analyze a CVSS vector and a short vulnerability description, and estimate both:
1. The Attack Potential (AP) according to ISO/SAE 21434.
2. The resulting risk classification according to TARA (Threat Analysis & Risk Assessment).

You must:

A. Derive all five AP factors:
   - elapsed time
   - specialist expertise
   - knowledge of the item or component
   - window of opportunity
   - equipment

B. Provide justifications based on the CVSS vector and description.

C. Map CVSS metrics to AP decisions (CVSS → AP mapping).

D. Estimate the **overall Attack Potential** (Low, Medium, High, Very High).

E. Assess the **Impact Rating (TARA)**:
   - Safety (S0–S3)
   - Financial (F0–F3)
   - Operational (O0–O3)
   - Privacy (P0–P3)

F. Conclude with the **overall Risk Level**: Low / Medium / High / Critical

💡 Output your answer in a clean, structured format so that another engineer or tool can parse the results easily.
Start directly with the attack potential estimation. No extra commentary.
"""

# ---------- UNIVERSAL 4-BIT LOADER ----------------------------------
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def load_model_and_tokenizer(model_id: str, train_mode: bool = False):
    """Returns (model, tokenizer) with 4-bit NF4 quant; Mixtral mit Offload."""

    qcfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if "gemma" in model_id.lower() else torch.float16,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    max_mem = {0: "14GiB", "cpu": "20GiB"} if "gemma" in model_id.lower() else None
    kwargs = dict(device_map="auto",
                  quantization_config=qcfg,
                  torch_dtype=torch.float16,
                  max_memory=max_mem,
                  trust_remote_code=True)

    if "mixtral" in model_id.lower():                     # MoE braucht Offload
        kwargs.update(
            offload_folder="/mnt/e/llm_offload/mixtral",   # ➜ existierendes SSD-Verzeichnis!
            offload_state_dict=True,
            max_memory={                          # GPU-Budget hart begrenzen
                0: "14GiB",      # alles darüber wandert in den Offload-Ordner
                "cpu": "20GiB"   # reservierter RAM-Puffer
            },
        )

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

    if train_mode:                                        # LoRA/Fine-Tuning
        model.config.use_cache = False
        model.enable_input_require_grads()

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return model, tok
# --------------------------------------------------------------------

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--base",   default="01-ai/Yi-1.5-9B-Chat")
parser.add_argument("--epochs_cvss40", type=int, default=15)
parser.add_argument("--output", default=None)        # let run_matrix decide
parser.add_argument("--resume", default=None)
args = parser.parse_args()

base_model  = args.base
num_epochs  = args.epochs_cvss40
output_path = args.output or f"../models/lora-cvss40/{Path(base_model).name}"
resume_from = args.resume or f"../models/lora-cvss31/{Path(base_model).name}"


# === Modell & Tokenizer ===
#base_model = "01-ai/Yi-1.5-9B-Chat"
#resume_from = "../models/lora-cvss31/Yi1.5_9B"  # Pfad zum vorher trainierten Modell
#output_path = "../models/lora-cvss40/Yi1.5_9B"

# === Tokenizer vorbereiten ===

#tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
#tokenizer.pad_token = tokenizer.eos_token

base, tokenizer = load_model_and_tokenizer(base_model, train_mode=True)

def lora_targets(ckpt_name):
    if "mistral" in ckpt_name or "yi-" in ckpt_name or "mixtral" in ckpt_name:
        return ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj"]
    if "gemma" in ckpt_name:
        return ["q_proj","k_proj","v_proj","o_proj"]
    # default = llama-style
    return ["q_proj","k_proj","v_proj","o_proj","down_proj"]


# === LoRA-Konfiguration laden und übernehmen ===
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules= lora_targets(base_model),
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False
)

#model = get_peft_model(base, peft_config)
model = PeftModel.from_pretrained(base, resume_from, is_trainable=True) # Übernehme vorheriges Training
model.config.use_cache = False
model.enable_input_require_grads()
#model = torch.compile(model, mode="reduce-overhead")  # ← ADD THIS LINE

# === Daten vorbereiten ===
data = load_dataset("json", data_files="../data/train_cvss40.jsonl")["train"]

def tokenize(example):
    text = f"{SYSTEM_PROMPT}{example['input']}\n{example['output']}"
    return tokenizer(text, truncation=True, max_length=1024)

tokenized = data.map(tokenize, batched=False)

def pick_batch(model_name: str) -> int:
    if "nemo" in model_name.lower():
        return 2          # passt sicher in 16 GB
    if "small" in model_name.lower():
        return 1
    if "gemma" in model_name.lower():
        return 2          # gemma braucht etwas weniger
    return 4              # Standard

local_batch = pick_batch(base_model)

# === Trainingsargumente ===
args = TrainingArguments(
    output_dir=output_path,
    per_device_train_batch_size=local_batch,
    gradient_accumulation_steps=2,
    num_train_epochs=num_epochs,
    learning_rate=1e-5,
    optim="paged_adamw_8bit",
    warmup_ratio=0.05,
    lr_scheduler_type="cosine_with_restarts",
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    report_to="none",
    max_grad_norm=0.3,
    gradient_checkpointing=True,
    torch_compile={
    "fullgraph": False,
    "dynamic": True,
    "backend": "inductor",
    "mode": "reduce-overhead"
    },
    remove_unused_columns=True
)

# === Daten-Collator ===
collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8
)

# === Trainer ===
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized,
    data_collator=collator
)

# === Training starten ===
print("🚀 Starte CVSS 4.0 Finetuning...")
trainer.train()

# === Modell speichern ===
model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)
print(f"✅ Training abgeschlossen. Modell gespeichert unter: {output_path}")
