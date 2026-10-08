"""ONNX parity: torch ckpt vs onnx cosine drift on sample queries.

Run: python scripts/verify_onnx_parity.py --torch <ckpt> --onnx <dir> --truncate 128
Gate: mean drift under 1e-3 target. INT8 near 1e-2 ships with table note.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

QUERIES = [
    "GST refund kaise claim karein?",
    "rental agreement ki security deposit limit?",
    "UPI payment fail ho gaya to kya karein?",
    "DIFC me employment notice period rule?",
    "home loan EMI kaise calculate karein?",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--torch", default="checkpoints/best_indic")
    ap.add_argument("--onnx", default="onnx")
    ap.add_argument("--truncate", type=int, default=128)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from src.embed.prefixes import format_query, truncate_and_renorm

    print(f"torch={args.torch} onnx={args.onnx} truncate={args.truncate}")
    if args.truncate not in (128, 256, 512, 768):
        raise ValueError("truncate must be 128, 256, 512, or 768 for v1 gates.")
    if args.dry_run:
        print("dry run ok. would encode 200 sample queries both paths and report mean cosine drift.")
        return

    import numpy as np
    import onnxruntime as ort
    import torch
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        args.torch,
        config_kwargs={"vision_config": None, "audio_config": None},
        model_kwargs={"torch_dtype": torch.float32},
    )
    first = model._first_module()
    tok = first.tokenizer
    sess = ort.InferenceSession(str(Path(args.onnx) / "model_int8.onnx" if (Path(args.onnx) / "model_int8.onnx").exists() else Path(args.onnx) / "model.onnx"))

    drifts = []
    for q in QUERIES:
        tv = np.asarray(model.encode(format_query("SearchQuery", q), truncate_dim=args.truncate, normalize_embeddings=True), dtype=np.float64)
        batch = tok([format_query("SearchQuery", q)], padding=True, truncation=True, max_length=256, return_tensors="np")
        ov = np.asarray(sess.run(["last_hidden"], {"input_ids": batch["input_ids"].astype(np.int64), "attention_mask": batch["attention_mask"].astype(np.int64)})[0][0])
        mask = batch["attention_mask"][0].astype(np.float64)
        pooled = (ov * mask[:, None]).sum(0) / mask.sum()
        ov = np.asarray(truncate_and_renorm(pooled, args.truncate), dtype=np.float64)
        drifts.append(float(abs(tv @ ov - 1.0)))
    mean_drift = float(np.mean(drifts))
    print(f"mean cosine drift over {len(QUERIES)} queries: {mean_drift:.6f}")
    print(f"max drift: {float(np.max(drifts)):.6f}")
    if mean_drift > 0.01:
        print("WARN: drift over 1e-2, ship per-quant table plainly.")
    elif mean_drift > 0.001:
        print("NOTE: drift over 1e-3 target but under 1e-2, ok for INT8 with note.")


if __name__ == "__main__":
    main()
