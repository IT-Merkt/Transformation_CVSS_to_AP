import gradio as gr
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig
)
from peft import PeftModel
import torch

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

# 1. Quantization Config (MUST match training)
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

# 2. Model Loading
base_model = "mistralai/Mistral-7B-Instruct-v0.2"
lora_path = "../models/lora-test"

tokenizer = AutoTokenizer.from_pretrained(lora_path)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    device_map="auto"
)
model = PeftModel.from_pretrained(model, lora_path)
model.eval()

# 3. Generation Config
generation_config = GenerationConfig(
    temperature=0.3,
    top_p=0.85,
    repetition_penalty=1.1,
    max_new_tokens=512,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

# 4. Prompt Formatting (MUST match training)
def format_prompt(cvss_vector, description=""):
    return f"{SYSTEM_PROMPT}{cvss_vector}\nDescription: {description}"

# 5. Inference Function
def generate_ap(cvss_vector, description=""):
    prompt = format_prompt(cvss_vector, description)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            generation_config=generation_config
        )
    
    # Extract only the new text
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full_output#[len(prompt):].strip()

# 6. Gradio Interface
iface = gr.Interface(
    fn=generate_ap,
    inputs=[
        gr.Textbox(label="CVSS Vector", placeholder="CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"),
        gr.Textbox(label="Description (optional)", lines=3)
    ],
    outputs=gr.Textbox(label="Attack Potential Assessment", lines=10),
    title="CVSS → Attack Potential (LoRA)",
    examples=[
        ["CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H", "Remote code execution"],
        ["CVSS:3.1/AV:P/AC:H/PR:L/UI:N/S:C/C:L/I:L/A:N", "Physical access required"]
    ]
)

iface.launch()
