#!/usr/bin/env python3
# ------------------------------------------------------------
#  GPT/DeepSeek-Judge-Prompt-Generator
#  Erstellt pro Modellordner unter ./outputs/ ein Prompt-File
# ------------------------------------------------------------
import json, re
from pathlib import Path

# ------------------------------------------------------------
# 1) Parameter
# ------------------------------------------------------------
SELECTED_LINES = {0, 2, 4, 11, 20, 22, 23, 31}        # ⇠ max. 8 Beispiele
EVAL_FILE      = Path("../data/eval_data.jsonl")      # Gold-Referenz
OUTPUTS_DIR    = Path("/mnt/e/llm_offload/outputs_8Rework")                   #/mnt/e/llm_offload/outputs Modell-Ordner

HEADER = """You are an expert in vulnerability analysis and ISO/SAE 21434 compliance.

The model was trained to transform CVSS (Common Vulnerability Scoring System) vectors
into Attack-Potential (AP) assessments and TARA (Threat Assessment and Risk Analysis)
outputs.

Below are multiple examples including:
- CVSS Input
- Expected Output (gold standard)
- Model Output (generated result)

Your task:
1. Compare the *expected* vs. *model* output.
2. For each example, fill the table below using this scale: 1 = poor, 10 = excellent.
3. Keep the rating consistent across all examples.

Rating criteria:
- Clarity
- Accuracy of AP factors (must follow  ISO 21434)
- Correct VA→Risk mapping  (see VA-to-Risk table)
- Reasoning quality
- Use of CVSS data
- Completeness (e.g. TARA part present and well integrated)

--- Begin Evaluation ---

Evaluation:
| Criterion           | Score (1–10) | Justification |
|--------------------|--------------|----------------|
| Clarity            |              |                |
| Accuracy of factors|              |                |
| Reasoning quality  |              |                |
| CVSS usage         |              |                |
| Completeness       |              |                |
| **Overall Score**  |              |                |

"""

def make_block(idx: int, ref: dict, pred: dict) -> str:
    """Formatiert ein Beispiel – beide Outputs bereits als String."""
    return f"""
📌 Example {idx}

CVSS Input:
{ref['input']}

Expected Output:
{ref['expected_output']}

Model Output:
{pred['output']}

"""

# ------------------------------------------------------------
# 2) Gold-Daten einmal laden
# ------------------------------------------------------------
with EVAL_FILE.open(encoding="utf-8") as f:
    REF_ROWS = [json.loads(l) for l in f]

# ------------------------------------------------------------
# 3) Alle Unterordner in ./outputs durchlaufen
# ------------------------------------------------------------
for sub in OUTPUTS_DIR.iterdir():
    pred_path = sub / "generated.jsonl"
    if not (sub.is_dir() and pred_path.exists()):
        continue                                     # kein Modell-Run → überspringen

    with pred_path.open(encoding="utf-8") as pred_f:
        PRED_ROWS = [json.loads(l) for l in pred_f]

    # 3a) Prompt bauen
    parts = [HEADER]
    for idx in sorted(SELECTED_LINES):
        parts.append(make_block(idx, REF_ROWS[idx], PRED_ROWS[idx]))

    prompt_text = "\n".join(parts)

    # 3b) Prompt speichern
    prompt_file = sub / "gpt_judge_prompt.txt"
    prompt_file.write_text(prompt_text, encoding="utf-8")
    print(f"✅ Prompt für {sub.name} gespeichert → {prompt_file}")

print("🏁 Alle fertig.")
