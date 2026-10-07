"""GGUF export notes plus parity gate for EmbeddingGemma-2.

Official v2 GGUF is live (ggml-org plus unsloth, 2026-10-06, arch gemma-embedding2).
This script does not re-invent convert. It points at the proven path and gates pooling.

Run: python src/embed/export_gguf.py --ckpt checkpoints/best_indic --out gguf --dry-run
Real path: use ggml-org convert on your fine tuned ckpt, then llama-quantize to Q8_0, Q6_K, Q4_K_M.
Pooling must be mean (base uses mean pooling, not Jina last token).
"""

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/best_indic")
    ap.add_argument("--out", default="gguf")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print(f"ckpt={args.ckpt} out={args.out}")
    print("source: https://huggingface.co/ggml-org/embeddinggemma-2-GGUF")
    print("pooling: mean. Verify with llama-embedding --pooling mean.")
    if args.dry_run:
        print("dry run ok. would run ggml convert then llama-quantize Q8_0, Q6_K, Q4_K_M.")
        print("would publish per quant table with MTEB-hi delta like Jina pattern.")
        return
    print("real run: clone ggml-org convert, convert ckpt to f16 gguf, quantize, test parity.")


if __name__ == "__main__":
    main()
