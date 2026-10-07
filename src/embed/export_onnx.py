"""Export ONNX INT8 for CPU serve. Verifies parity separately.

Run: python src/embed/export_onnx.py --ckpt checkpoints/best_indic --out onnx --int8 --dry-run
Real run uses optimum onnxruntime export plus dynamic quant. No torch needed at serve time.
"""

import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/best_indic")
    ap.add_argument("--out", default="onnx")
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    print(f"ckpt={args.ckpt} out={out} int8={args.int8}")
    if args.dry_run:
        print("dry run ok. would export onnx graph with mean pooling, 768d, truncate at runtime.")
        return

    try:
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        from transformers import AutoTokenizer
    except ImportError as e:
        raise RuntimeError(f"need optimum plus onnxruntime: {e}")

    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.ckpt)
    ort = ORTModelForFeatureExtraction.from_pretrained(args.ckpt, export=True)
    ort.save_pretrained(str(out))
    tok.save_pretrained(str(out))
    print(f"saved fp onnx to {out}")
    if args.int8:
        print("next: run onnxruntime quantize_dynamic on model.onnx to INT8. Then run scripts/verify_onnx_parity.py")


if __name__ == "__main__":
    main()
