"""Prepare frozen triplets. License safe by default.

FLORES is eval only (CC-BY-SA). Never put FLORES text in Apache train pack.
Train pack: Banking77 pairs plus MS MARCO hard subset plus synthetic Hinglish plus synthetic legal (ours).
Writes data/frozen/triplets_40k.jsonl plus data/frozen/legal_10k.jsonl plus SHA256SUMS.

Run: python scripts/prepare_triplets.py --dry-run
"""

import argparse
import hashlib
from pathlib import Path

TRAIN_POLICY = """
Train pack license policy:
- Banking77: CC-BY-4.0, needs attribution. Keep source note in dataset card.
- MS MARCO hard negatives subset: pin exact revision plus license before publish.
- Synthetic Hinglish and legal queries: ours, Apache-2.0. Human spot check 500 each.
- FLORES and AI4Bharat Indic pairs: eval only unless license is cleared. Do not ship their text in train pack.
"""


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-dir", default="data/frozen")
    args = ap.parse_args()

    print(TRAIN_POLICY.strip())
    out = Path(args.out_dir)
    if args.dry_run:
        print(f"dry run ok. would write {out}/triplets_40k.jsonl and {out}/legal_10k.jsonl plus SHA256SUMS")
        print("target mix: 15k Hinglish QA synth plus 5k Banking77 plus 10k legal synth plus 10k MS MARCO hard")
        return
    out.mkdir(parents=True, exist_ok=True)
    print(f"real run not wired to source downloads yet. Wire Banking77 plus MARCO pulls here, then freeze hashes.")


if __name__ == "__main__":
    main()
