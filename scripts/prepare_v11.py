"""Freeze v1.2 triplets: MIRACL Hindi plus the full v1 mix.

The v1.1 attempt dropped the 10k near-miss hard slice and Hindi retrieval
fell 0.732 to 0.581. v1.2 puts that slice back and keeps MIRACL on top.

Mix: 2k MIRACL hi (real queries, judged positives, Apache-2.0), 15k Hinglish
synth, 5k Banking77, 10k legal synth, 10k near-miss hard.
Writes triplets_40k_v12.jsonl plus SHA256SUMS_V12. Stays local until verified.

Run: python scripts/prepare_v11.py
"""

import gzip
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_triplets import build_near_miss_hard, pull_banking77, synth_hinglish, synth_legal

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
    rows += build_near_miss_hard(10000)
    rows += synth_hinglish(15000)
    rows += pull_banking77(5000)
    rows += synth_legal(10000)
    random.shuffle(rows)
    print(f"total: {len(rows)}")

    trip = OUT_DIR / "triplets_40k_v12.jsonl"
    with open(trip, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    h = hashlib.sha256()
    with open(trip, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    sums = OUT_DIR / "SHA256SUMS_V12"
    sums.write_text(f"{h.hexdigest()}  triplets_40k_v12.jsonl\n")
    print(f"wrote {trip} ({len(rows)} rows)")
    print(sums.read_text().strip())


if __name__ == "__main__":
    main()
