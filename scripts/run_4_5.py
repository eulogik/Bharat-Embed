"""Unattended run for step 4 (train) plus step 5 (eval plus export).

Scheduled by user: starts 2h after kickoff, runs in bg with nohup.
All heavy paths on Kioxia. Logs to Kioxia so a restart keeps them.

Order:
1. train indic LoRA (40k, 3 epochs, batch 16) -> Kioxia checkpoints
2. train legal adapter (1 epoch) -> Kioxia checkpoints
3. eval slice (writes raw json, placeholder harness note)
4. onnx export attempt (may fail on optimum vs transformers 5.x, logged)
5. hf push attempt only if HF_TOKEN is set, else dry-run log
"""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

KIO = Path("/Volumes/KIOXIA 1TB/bharat-embed")
VENV_PY = KIO / "venv/bin/python"
LOG = KIO / "logs/run_4_5.log"
DATA = KIO / "data"
CKPT_INDIC = KIO / "checkpoints/best_indic"
CKPT_LEGAL = KIO / "checkpoints/legal_adapter"


def log(msg: str):
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def run(cmd: list, step: str):
    log(f"START {step}: {' '.join(cmd)}")
    env = dict(os.environ)
    env["TMPDIR"] = str(KIO / "tmp")
    env["PYTHONUNBUFFERED"] = "1"
    env["HF_HOME"] = str(KIO / "hf_cache")
    env["HF_HUB_CACHE"] = str(KIO / "hf_cache")
    env["TRANSFORMERS_CACHE"] = str(KIO / "hf_cache")
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as lf:
        lf.flush()
        p = subprocess.run(
            cmd, cwd="/Users/eulogikdeveloper/Documents/Bharat-Embed", env=env,
            stdout=lf, stderr=subprocess.STDOUT,
        )
        lf.write(f"[{datetime.now().isoformat(timespec='seconds')}] END {step}: exit={p.returncode}\n")
        lf.flush()
    print(f"END {step}: exit={p.returncode}", flush=True)
    return p.returncode


def main():
    log("run_4_5 begin")
    # Batch 8 not 16: batch 16 fp32 swaps the 16GB box to death (15G + swap full,
    # steps decay 1s to 27s/it). Batch 8 holds ~2it/s steady. Noted in eval.
    # max-len 256 not 512: halves activations again, sizing test proved steady.
    # Short docs plus Hinglish queries fit fine. Honest note in eval README.
    rc = run(
        [str(VENV_PY), "src/embed/train_indic.py",
         "--data", str(DATA / "triplets_40k.jsonl"),
         "--out", str(CKPT_INDIC),
         "--epochs", "3", "--batch", "8", "--max-len", "256"],
        "train-indic",
    )
    if rc != 0:
        log("train-indic failed, stop. Check log.")
        sys.exit(1)

    rc = run(
        [str(VENV_PY), "src/embed/train_legal_adapter.py",
         "--base", str(CKPT_INDIC),
         "--data", str(DATA / "legal_10k.jsonl"),
         "--out", str(CKPT_LEGAL),
         "--epochs", "1", "--batch", "8"],
        "train-legal",
    )
    if rc != 0:
        log("train-legal failed, stop. Check log.")
        sys.exit(1)

    run(
        [str(VENV_PY), "scripts/eval_mteb_slice.py",
         "--ckpt", str(CKPT_INDIC),
         "--suite", "hi,en,code", "--truncate", "128,768",
         "--out", str(KIO / "eval/results.json")],
        "eval-slice",
    )
    run(
        [str(VENV_PY), "src/embed/export_onnx.py",
         "--ckpt", str(CKPT_INDIC),
         "--out", str(KIO / "onnx"), "--int8"],
        "export-onnx",
    )
    try:
        from huggingface_hub import HfApi

        HfApi().whoami()
        logged_in = True
    except Exception:
        logged_in = "HF_TOKEN" in os.environ and bool(os.environ.get("HF_TOKEN"))
    if logged_in:
        run([str(VENV_PY), "scripts/push_hf.py"], "push-hf")
    else:
        log("HF login missing, push skipped (dry-run only). Run hf auth login to ship.")
        run([str(VENV_PY), "scripts/push_hf.py", "--dry-run"], "push-dryrun")
    log("run_4_5 done")


if __name__ == "__main__":
    main()
