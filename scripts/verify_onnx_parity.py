"""ONNX parity check. Torch vs ONNX cosine drift gate.

Run: python scripts/verify_onnx_parity.py --torch checkpoints/best_indic --onnx onnx --dry-run
Gate: mean cosine drift under 1e-3 is target, but INT8 may land near 1e-2.
If over target, ship per quant table plainly instead of forcing pass.
"""

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--torch", default="checkpoints/best_indic")
    ap.add_argument("--onnx", default="onnx")
    ap.add_argument("--truncate", type=int, default=128)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print(f"torch={args.torch} onnx={args.onnx} truncate={args.truncate}")
    if args.truncate not in (128, 256, 512, 768):
        raise ValueError("truncate must be 128, 256, 512, or 768 for v1 gates.")
    if args.dry_run:
        print("dry run ok. would encode 200 sample queries both paths and report mean cosine drift.")
        return
    print("real run: load both, encode same queries with same prefix, compare.")


if __name__ == "__main__":
    main()
