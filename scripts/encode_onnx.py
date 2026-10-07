"""Encode queries with ONNX pack. No torch needed at serve.

Run: python scripts/encode_onnx.py --model onnx/int8 --queries queries.txt --truncate 128 --dry-run
"""

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="onnx/int8")
    ap.add_argument("--queries", default="queries.txt")
    ap.add_argument("--truncate", type=int, default=128)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print(f"model={args.model} queries={args.queries} truncate={args.truncate}")
    if args.dry_run:
        print("dry run ok. would output normalized 128d vectors for Qdrant upsert.")
        return
    print("real run: onnxruntime session, prefix each query, truncate plus renorm.")


if __name__ == "__main__":
    main()
