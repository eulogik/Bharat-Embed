"""Prepare frozen triplets. License safe by default.

FLORES is eval only (CC-BY-SA). Never put FLORES text in Apache train pack.
Train pack: Banking77 pairs plus MS MARCO hard subset plus synthetic Hinglish plus synthetic legal (ours).
Writes triplets_40k.jsonl plus legal_10k.jsonl plus SHA256SUMS.

Real run pulls from HF datasets into Kioxia cache, writes outputs to --out-dir.
Default out dir is Kioxia so restarts do not wipe anything.

Run: python scripts/prepare_triplets.py --dry-run
Real: python scripts/prepare_triplets.py --out-dir "/Volumes/KIOXIA 1TB/bharat-embed/data"
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

TRAIN_POLICY = """
Train pack license policy:
- Banking77: CC-BY-4.0, needs attribution. Keep source note in dataset card.
- MS MARCO hard negatives subset: pin exact revision plus license before publish.
- Synthetic Hinglish and legal queries: ours, Apache-2.0. Human spot check 500 each.
- FLORES and AI4Bharat Indic pairs: eval only unless license is cleared. Do not ship their text in train pack.
"""

HINGLISH_TEMPLATES = [
    ("{x} kaise karein?", "{x} kaise karen"),
    ("{x} kya hai?", "{x} kya hota hai"),
    ("{x} ke liye apply kaise kare?", "{x} ke liye aavedan kaise karein"),
]

HINGLISH_TOPICS = [
    "GST refund", "PAN card update", "bank account KYC", "UPI payment fail",
    "train ticket refund", "passport renew", "Aadhaar address change",
    "fixed deposit rate", "home loan EMI", "rental agreement deposit",
]

LEGAL_TOPICS = [
    ("DIFC", "employment contract notice period"),
    ("ADGM", "company registration steps"),
    ("GST", "input tax credit claim"),
    ("VAT", "VAT refund for tourists"),
    ("DIFC", "tenancy renewal rules"),
    ("GST", "late fee on GSTR filing"),
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def synth_hinglish(n: int):
    rows = []
    i = 0
    while len(rows) < n:
        topic = HINGLISH_TOPICS[i % len(HINGLISH_TOPICS)]
        tmpl = HINGLISH_TEMPLATES[(i // len(HINGLISH_TOPICS)) % len(HINGLISH_TEMPLATES)]
        q_roman = tmpl[0].format(x=topic)
        q_deva = tmpl[1].format(x=topic)
        q = q_roman if i % 2 == 0 else q_deva
        rows.append({
            "query": f"task: search result | query: {q}",
            "pos": f"title: {topic} | text: {topic} help steps in Hindi and English with portal link and docs needed",
            "negs": [f"title: none | text: unrelated banking note {i}"],
            "src": "synth-hinglish-ours",
        })
        i += 1
    return rows


def synth_legal(n: int, prefix: str = "synth-legal-ours"):
    rows = []
    i = 0
    while len(rows) < n:
        law, topic = LEGAL_TOPICS[i % len(LEGAL_TOPICS)]
        q = f"{law} me {topic} ka rule kya hai?"
        rows.append({
            "query": f"task: search result | query: {q}",
            "pos": f"title: {law} {topic} | text: {law} statute snippet for {topic}, section and steps",
            "negs": [f"title: none | text: unrelated statute note {i}"],
            "src": prefix,
        })
        i += 1
    return rows


def pull_banking77(n_pairs: int):
    from datasets import load_dataset

    last_err = None
    ds = None
    for name in ("PolyAI/banking77", "banking77"):
        try:
            ds = load_dataset(name, split="train")
            print(f"banking source: {name}")
            break
        except Exception as e:
            last_err = e
    if ds is None:
        raise RuntimeError(f"banking77 pull failed: {last_err}")
    rows = []
    by_label: dict = {}
    for r in ds:
        by_label.setdefault(r["label"], []).append(r["text"])
    labels = list(by_label)
    i = 0
    while len(rows) < n_pairs:
        lab = labels[i % len(labels)]
        texts = by_label[lab]
        pos = texts[(i // len(labels)) % len(texts)]
        other_lab = labels[(i + 1) % len(labels)]
        neg = by_label[other_lab][0]
        rows.append({
            "query": f"task: classification | query: {pos}",
            "pos": f"title: none | text: {texts[(i + 1) % len(texts)]}",
            "negs": [f"title: none | text: {neg}"],
            "src": "banking77-CC-BY-4.0",
        })
        i += 1
    return rows


def pull_msmarco_hard(n: int):
    from datasets import load_dataset

    ds = load_dataset("sentence-transformers/msmarco-hard-negatives", split="train")
    rows = []
    for r in ds:
        if len(rows) >= n:
            break
        q = r.get("query") or r.get("anchor")
        pos = r.get("positive") or (r.get("docs") or [None])[0]
        neg = (r.get("negatives") or r.get("docs") or [None, None])[1]
        if not q or not pos:
            continue
        rows.append({
            "query": f"task: search result | query: {q}",
            "pos": f"title: none | text: {pos}",
            "negs": [f"title: none | text: {neg or 'unrelated passage'}"],
            "src": "msmarco-hard-negatives-pin-revision",
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-dir", default="/Volumes/KIOXIA 1TB/bharat-embed/data")
    ap.add_argument("--n-hinglish", type=int, default=15000)
    ap.add_argument("--n-banking", type=int, default=5000)
    ap.add_argument("--n-legal-train", type=int, default=10000)
    ap.add_argument("--n-marco", type=int, default=10000)
    ap.add_argument("--n-legal-extra", type=int, default=10000)
    args = ap.parse_args()

    print(TRAIN_POLICY.strip())
    out = Path(args.out_dir)

    total = args.n_hinglish + args.n_banking + args.n_legal_train + args.n_marco
    print(f"target: {total} train rows ({args.n_hinglish} hinglish, {args.n_banking} banking, {args.n_legal_train} legal, {args.n_marco} marco)")
    if args.dry_run:
        print(f"dry run ok. would write {out}/triplets_40k.jsonl and {out}/legal_10k.jsonl plus SHA256SUMS")
        return

    out.mkdir(parents=True, exist_ok=True)
    print("building synth slices (fast, no download)...")
    hinglish = synth_hinglish(args.n_hinglish)
    legal_train = synth_legal(args.n_legal_train)
    legal_extra = synth_legal(args.n_legal_extra)
    print("pulling banking77 (CC-BY-4.0)...")
    banking = pull_banking77(args.n_banking)
    print("pulling msmarco hard negatives (pin revision)...")
    marco = pull_msmarco_hard(args.n_marco)

    train_rows = hinglish + banking + legal_train + marco
    assert len(train_rows) == total, f"got {len(train_rows)}, want {total}"

    trip = out / "triplets_40k.jsonl"
    with open(trip, "w") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    legal_path = out / "legal_10k.jsonl"
    with open(legal_path, "w") as f:
        for r in legal_extra:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    sums = out / "SHA256SUMS"
    sums.write_text(f"{sha256_file(trip)}  triplets_40k.jsonl\n{sha256_file(legal_path)}  legal_10k.jsonl\n")
    print(f"wrote {trip} ({len(train_rows)} rows)")
    print(f"wrote {legal_path} ({len(legal_extra)} rows)")
    print(f"wrote {sums}")
    print(sums.read_text().strip())


if __name__ == "__main__":
    main()
