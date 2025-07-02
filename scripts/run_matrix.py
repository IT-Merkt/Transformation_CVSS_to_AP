import yaml, subprocess, pathlib, datetime as dt
from pathlib import Path

CONF = yaml.safe_load(open("experiments.yaml"))
ROOT = pathlib.Path("../models")

for run in CONF["runs"]:
    # 0️⃣  paths
    cvss31_dir = ROOT / "lora-cvss31" / Path(run["base"]).name
    cvss40_dir = ROOT / "lora-cvss40" / Path(run["base"]).name
    
    # 1️⃣  pre-train on CVSS 3.1  (unless already done)
    if not cvss31_dir.exists():
        subprocess.run([
            "python", "train.py",
            "--base", run["base"],
            "--epochs_cvss31", str(run["epochs_cvss31"]),
            "--output", str(cvss31_dir)
        ], check=True)
    
    # 2️⃣  fine-tune on CVSS 4.0
    subprocess.run([
        "python", "trainCVSS4.0.py",
        "--base",   run["base"],
        "--epochs_cvss40", str(run["epochs_cvss40"]),
        "--output", str(cvss40_dir),
        "--resume", str(cvss31_dir)
    ], check=True)
    
    # 3️⃣  evaluate
    subprocess.run([
        "python", "eval_and_infer.py",
        "--base", run["base"],
        "--model_path", str(cvss40_dir)
    ], check=True)

