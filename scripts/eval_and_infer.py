#!/usr/bin/env python
# ---------------------------------------------------------------
#  Evaluate LoRA model:  ISO 21434 Attack-Potential + Risk Matrix
# ---------------------------------------------------------------
import re, json, argparse, numpy as np, pandas as pd, torch
from pathlib import Path
from datetime import datetime
from difflib import SequenceMatcher
from tqdm import tqdm
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer, util
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          BitsAndBytesConfig)
from peft import PeftModel
from risk_matrix import adjust_risk       # ◀︎ Matrix-Funktion nutzen

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
    )
    max_mem = {0: "12GiB", "cpu": "20GiB"} if "gemma" in model_id.lower() else None
    kwargs = dict(device_map="auto",
                  quantization_config=qcfg,
                  torch_dtype=torch.float16,
                  max_memory=max_mem,
                  trust_remote_code=True)

    if "mixtral" in model_id.lower():                     # MoE braucht Offload
        kwargs.update(offload_folder="./offload_mixtral",
                      offload_state_dict=True)

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

    if train_mode:                                        # LoRA/Fine-Tuning
        model.config.use_cache = False
        model.enable_input_require_grads()

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return model, tok
# --------------------------------------------------------------------

# ------------------------ Argumente -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--base", default="meta-llama/Llama-2-7b-chat-hf")
parser.add_argument("--model_path")               # LoRA-Pfad
args = parser.parse_args()

BASE_CKPT = args.base
LORA_DIR  = args.model_path or f"../models/lora-cvss40/{Path(BASE_CKPT).name}"

# ------------------------ Konstanten ----------------------------
SYSTEM_PROMPT = """You are a cybersecurity expert specialized in automotive risk assessment following ISO/SAE 21434 and TARA methodology.

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
EVAL_FILE   = "../data/eval_data.jsonl"
OUT_DIR     = Path("../outputs") / Path(LORA_DIR).name           # z.B. outputs/Yi-1.5-9B-Chat
OUT_DIR.mkdir(parents=True, exist_ok=True)
GENERATED   = OUT_DIR / "generated.jsonl"
SCORES_CSV  = "../outputs/scores.csv"
SUMMARY_CSV = "../outputs/summary_log.csv"

# ------------------------ Tokenizer & Modell --------------------

base, tok = load_model_and_tokenizer(BASE_CKPT)
model = PeftModel.from_pretrained(base, LORA_DIR)
model.eval()


tok.pad_token  = tok.eos_token
tok.padding_side = "left"                         # ◀︎ wichtig bei Decoder-Only

# ------------------------ Eval-Daten ----------------------------
def load_jsonl(fp): return [json.loads(l) for l in Path(fp).read_text().splitlines()]
examples = load_jsonl(EVAL_FILE)

# ------------------------ Inferenz ------------------------------
Path("./outputs").mkdir(exist_ok=True, parents=True)
generated = []

print("💡  Generiere Antworten …")
BS = 4
for i in tqdm(range(0, len(examples), BS)):
    chunk = examples[i:i+BS]
    prompts = [f"{SYSTEM_PROMPT}{ex['input'].strip()}\n" for ex in chunk]
    inputs  = tok(prompts, return_tensors="pt", padding=True,
                  truncation=True, max_length=1024).to("cuda")

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=512,           # niedrigere Werte währen schneller
            do_sample=False,
            temperature=None,
            top_p=None,
            repetition_penalty=1.1,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id
        )
    decoded = tok.batch_decode(out_ids, skip_special_tokens=True)
    for ex, pred in zip(chunk, decoded):
        generated.append({"input": ex["input"],
                          "expected_output": ex["expected_output"],
                          "output": pred.strip()})

Path(GENERATED).write_text("\n".join(json.dumps(e, ensure_ascii=False)
                                    for e in generated), encoding="utf-8")
print(f"✅  Antworten gespeichert unter {GENERATED}")

# ------------------------ Regex-Tools ---------------------------
FACTOR_RX = {
    "Elapsed Time"         : r"Elapsed Time\s*:\s*(\w+)",
    "Specialist Expertise" : r"Specialist Expertise\s*:\s*(\w+)",
    "Knowledge"            : r"Knowledge.*?:\s*(\w+)",
    "Window of Opportunity": r"Window of Opportunity\s*:\s*(\w+)",
    "Equipment"            : r"Equipment\s*:\s*(\w+)",
}
def parse_factors(txt:str):
    out={}
    for k,rx in FACTOR_RX.items():
        m=re.search(rx,txt,re.I)
        if m: out[k]=m.group(1).title()
    return out

def parse_impacts(txt:str):
    # sucht S3 F1 … → {'S':3,'F':1,…}
    m=re.findall(r"\b([SFOP])(\d)\b",txt)
    return {q:int(v) for q,v in m}

AP_ABBR   = {"VL":"Very Low","L":"Low","M":"Medium","H":"High","VH":"Very High"}
RISK_ABBR = {"L":"Low","M":"Medium","H":"High","C":"Critical"}
CVSS_AP_RX = re.compile(r"/VA:([HML])(/|$)", re.I)           #  /VA:M/ → Medium

def parse_answer(txt: str):
    """liefert (ap_class, risk_class) oder (None, None)"""
    txt = txt or ""
    ap = rk = None

    # 1)  AP im CVSS-Vektor
    m = CVSS_AP_RX.search(txt)
    if m:
        ap = {"H":"High", "M":"Medium", "L":"Low"}[m.group(1).upper()]

    # 2)  Verbale / abgekürzte AP-Zeile
    if not ap:
        m = re.search(r"(Attack\s*Potential|AP|VA)[^A-Za-z0-9]{0,4}"
                      r"(Very\s+Low|Low|Medium|High|Very\s+High|VL|VH|L|M|H)",
                      txt, re.I)
        if m:
            t = m.group(2).strip()
            ap = AP_ABBR.get(t.upper(), t.title())

    # 3)  Risk-Zeile
    m = re.search(r"(?:Resulting\s*)?Risk\s*(?:Level|Rating|Value)?"
                  r"[^A-Za-z]*(Low|Medium|High|Critical|L|M|H|C)",
                  txt, re.I)
    if m:
        t = m.group(1).strip()
        rk = RISK_ABBR.get(t.upper(), t.title())

    return ap, rk

# ------------------------ Ähnlichkeit ---------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
pred_texts = [e["output"]           for e in generated]
exp_texts  = [e["expected_output"]  for e in generated]

exp_emb = embedder.encode(exp_texts, batch_size=16, convert_to_tensor=True, normalize_embeddings=True)
pred_emb = embedder.encode(pred_texts, batch_size=16, convert_to_tensor=True, normalize_embeddings=True)
sem_sims = util.cos_sim(exp_emb, pred_emb).diagonal()

# ------------------------ Metriken ------------------------------
gold_ap, pred_ap = [], []
gold_risk, pred_risk = [], []
factor_matches, risk_consistency, tara_valid = [], [], []

for g in generated:
    exp_ap, exp_risk = parse_answer(g["expected_output"])
    prd_ap, prd_risk = parse_answer(g["output"])

    if exp_ap and prd_ap:   gold_ap.append(exp_ap);   pred_ap.append(prd_ap)
    if exp_risk and prd_risk: gold_risk.append(exp_risk); pred_risk.append(prd_risk)

    # 5-Faktor-Treffer
    exp_fac = parse_factors(g["expected_output"])
    prd_fac = parse_factors(g["output"])
    factor_matches.append(sum(exp_fac.get(k)==prd_fac.get(k) for k in FACTOR_RX))

    # Risiko-Matrix prüfen
    prd_imp  = parse_impacts(g["output"])
    if prd_ap and prd_imp and prd_risk:
        calc = adjust_risk(prd_ap, {k:f"{k}{v}" for k,v in prd_imp.items()})
        risk_consistency.append(calc == prd_risk)
    else:
        risk_consistency.append(False)

    # TARA-Quadranten plausibel?
    tara_valid.append(all(0<=v<=3 for v in prd_imp.values()))

# Klassische F1
F1_LABELS_AP   = ["Very Low","Low","Medium","High","Very High"]
F1_LABELS_RISK = ["Low","Medium","High","Critical"]

f1_ap = (f1_score(gold_ap,  pred_ap,
                  labels=F1_LABELS_AP, average="macro", zero_division=0)
         if gold_ap else float("nan"))
f1_risk = (f1_score(gold_risk, pred_risk,
                    labels=F1_LABELS_RISK, average="macro", zero_division=0)
           if gold_risk else float("nan"))

# ------------------------ Zeilen für CSV ------------------------
rows=[]
for i,g in enumerate(generated):
    token_sim = SequenceMatcher(None, exp_texts[i], pred_texts[i]).ratio()
    rows.append({
        "id": i,
        "input": g["input"],
        "expected_output": exp_texts[i],
        "model_output": pred_texts[i],
        "exact_match": exp_texts[i]==pred_texts[i],
        "token_similarity": round(token_sim,4),
        "semantic_similarity": round(sem_sims[i].item(),4),
        "ap_gold": gold_ap[i] if i<len(gold_ap) else "",
        "ap_pred": pred_ap[i] if i<len(pred_ap) else "",
        "risk_gold": gold_risk[i] if i<len(gold_risk) else "",
        "risk_pred": pred_risk[i] if i<len(pred_risk) else ""
    })
pd.DataFrame(rows).to_csv(SCORES_CSV, index=False)
print(f"✅  Detail-Scores gespeichert → {SCORES_CSV}")

# ------------------------ Zusammenfassung -----------------------
summary = {
    # 0-5
    "model": Path(LORA_DIR).name,
    "date":  datetime.now().strftime("%Y-%m-%d %H:%M"),
    "epochs": "",
    "batch_size": "",
    "accumulation_steps": "",
    "learning_rate": "",

    # 6-8
    "exact_match": np.mean([r["exact_match"] for r in rows]),
    "token_similarity": np.mean([r["token_similarity"] for r in rows]),
    "semantic_similarity": np.mean([r["semantic_similarity"] for r in rows]),

    # 9-14
    "f1_ap": f1_ap,
    "f1_risk": f1_risk,
    "ap_factor_acc": np.mean([m/5 for m in factor_matches]),
    "risk_consistency": np.mean(risk_consistency),
    "tara_validity": np.mean(tara_valid),

    # 15-18
    "gpt_score": "",
    "deepseek_score": "",
    "worked_best": "",
    "worked_worst": ""
}
pd.DataFrame([summary]).to_csv(SUMMARY_CSV,
    mode="a", header=not Path(SUMMARY_CSV).exists(), index=False)
print(f"📈  Zusammenfassung anhängt → {SUMMARY_CSV}")
