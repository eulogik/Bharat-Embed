"""Freeze v1.1 triplets: MIRACL Hindi real queries plus v1 slices.

Mix: ~10k MIRACL hi (real queries, judged positives, Apache-2.0) plus
15k Hinglish synth plus 5k Banking77 plus 10k legal synth = ~40k.
Writes triplets_40k_v11.jsonl plus SHA256SUMS. Stays local until verified.

Run: python scripts/prepare_v11.py
"""

import gzip
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_triplets import synth_hinglish, synth_legal, pull_banking77

RAW = Path("/Volumes/KIOXIA 1TB/bharat-embed/data/_raw/miracl-hi")
OUT_DIR = Path("/Volumes/KIOXIA 1TB/bharat-embed/data")


def load_corpus() -> dict:
    docs = {}
    for gz in sorted((RAW / "corpus").rglob("*.jsonl.gz")):
        with gzip.open(gz, "rt") as f:
            for line in f:
                r = json.loads(line)
                docs[str(r["docid"])] = f"{r.get('title', '')} {r.get('text', '')}".strip()[:2000]
    print(f"corpus docs: {len(docs)}")
    return docs


def load_topics_qrels():
    topics = {}
    with open(RAW / "miracl-v1.0-hi/topics/topics.miracl-v1.0-hi-train.tsv") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                topics[parts[0]] = parts[1]
    pos = {}
    with open(RAW / "miracl-v1.0-hi/qrels/qrels.miracl-v1.0-hi-train.tsv") as f:
        for line in f:
            qid, _, docid, rel = line.rstrip("\n").split("\t")[:4]
            if int(rel) > 0:
                pos.setdefault(qid, []).append(docid)
    print(f"topics: {len(topics)}, judged queries: {len(pos)}")
    return topics, pos


def main():
    random.seed(7)
    docs = load_corpus()
    topics, pos = load_topics_qrels()
    docids = list(docs)
    rows = []
    for qid, qtext in topics.items():
        for docid in pos.get(qid, [])[:3]:
            if docid not in docs:
                continue
            neg = random.choice(docids)
            rows.append({
                "query": f"task: search result | query: {qtext}",
                "pos": f"title: none | text: {docs[docid]}",
                "negs": [f"title: none | text: {docs[neg]}"],
                "src": "miracl-hi-train-apache2",
            })
    print(f"miracl rows: {len(rows)}")
    rows += synth_hinglish(15000)
    rows += pull_banking77(5000)
    rows += synth_legal(10000)
    random.shuffle(rows)
    print(f"total: {len(rows)}")

    trip = OUT_DIR / "triplets_40k_v11.jsonl"
    with open(trip, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    h = hashlib.sha256()
    with open(trip, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    sums = OUT_DIR / "SHA256SUMS_V11"
    sums.write_text(f"{h.hexdigest()}  triplets_40k_v11.jsonl\n")
    print(f"wrote {trip} ({len(rows)} rows)")
    print(sums.read_text().strip())


if __name__ == "__main__":
    main()
