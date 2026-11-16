"""Build the published benchmark files from local sources.

Outputs (all in benchmarks/data/):
  indic_cross_script.jsonl   100 Roman query / Devanagari passage pairs, ours
  adgm_legal_queries.jsonl   290 professional queries with target passage id
  adgm_legal_corpus.jsonl    304 passage corpus, source plus target

The ADGM portion is derived from RegNLP/xrefrag-adgm-dpel (Apache-2.0) with
attribution in the benchmark README.

Run: python scripts/build_benchmark.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks/data"
LEGAL_RAW = Path("/Volumes/KIOXIA 1TB/bharat-embed/data/_raw/legal/xrefrag-dpel/data")


def build_indic():
    src = ROOT / "eval/cross_script_pairs.jsonl"
    rows = [json.loads(l) for l in open(src) if l.strip()]
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / "indic_cross_script.jsonl"
    with open(dest, "w") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({
                "id": f"indic-{i:03d}",
                "topic": r["topic"],
                "query": r["q_roman"],
                "query_devanagari": r["q_deva"],
                "passage": r["passage"],
            }, ensure_ascii=False) + "\n")
    print(f"wrote {dest} ({len(rows)} rows)")
    return len(rows)


def build_legal():
    rows = []
    for split in ("train", "dev", "test"):
        p = LEGAL_RAW / f"{split}.jsonl"
        rows += [json.loads(l) for l in open(p) if l.strip()]
    corpus = {}
    for r in rows:
        corpus[str(r["source_passage_id"])] = r["source_text"]
        corpus[str(r["target_passage_id"])] = r["target_text"]

    OUT.mkdir(parents=True, exist_ok=True)
    qdest = OUT / "adgm_legal_queries.jsonl"
    with open(qdest, "w") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({
                "id": f"adgm-{i:03d}",
                "query": r["question"],
                "target_passage_id": str(r["target_passage_id"]),
                "source_passage_id": str(r["source_passage_id"]),
                "gold_answer": r.get("gold_answer", ""),
            }, ensure_ascii=False) + "\n")

    cdest = OUT / "adgm_legal_corpus.jsonl"
    with open(cdest, "w") as f:
        for pid, text in corpus.items():
            f.write(json.dumps({"passage_id": pid, "text": text}, ensure_ascii=False) + "\n")
    print(f"wrote {qdest} ({len(rows)} rows)")
    print(f"wrote {cdest} ({len(corpus)} passages)")
    return len(rows), len(corpus)


if __name__ == "__main__":
    a = build_indic()
    b, c = build_legal()
    print(f"total: {a} indic pairs, {b} legal queries, {c} legal passages")
