"""Eval slice runner. Pre-registered gates live in eval/README.md.

Run: python scripts/eval_mteb_slice.py --ckpt checkpoints/best_indic --suite hi,en,code --truncate 128,768 --dry-run
Real run: MTEB-hi slice, hi-en cross lingual, Banking77 pairs, legal recall at 5, truncation sweep.
Writes eval JSON plus truncation chart data. Raw scores only, no cherry pick.
"""

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/best_indic")
    ap.add_argument("--suite", default="hi,en,code")
    ap.add_argument("--truncate", default="128,768")
    ap.add_argument("--out", default="eval/results.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    suites = [s.strip() for s in args.suite.split(",")]
    dims = [int(x) for x in args.truncate.split(",")]
    print(f"ckpt={args.ckpt} suites={suites} dims={dims}")
    for d in dims:
        if d not in (32, 128, 256, 512, 768):
            raise ValueError(f"bad dim {d}")
    if 32 in dims:
        print("note: 32d is lab only. Report with warning.")

    if args.dry_run:
        print("dry run ok. would write raw eval json with per suite plus per dim plus per quant delta.")
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"ckpt": args.ckpt, "suites": suites, "dims": dims, "status": "wire MTEB harness here"}
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}. Now fill real MTEB calls.")


if __name__ == "__main__":
    main()
