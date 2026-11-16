#!/usr/bin/env python
"""Run the Bharat retrieval benchmark on any sentence-transformers model.

Two tasks, both with real hand checked or professionally authored queries:

  indic  100 Roman Hinglish queries, each with the Devanagari passage that
         answers it. Corpus is all 100 passages. Catches the case where a model
         handles Devanagari but fails when the user types in Roman script.

  legal  290 professional questions over ADGM data protection law, 304 passage
         corpus. Both the source and target passage of each cross reference sit
         in the corpus, so the distractor is topically adjacent by design.

Usage:
  python run_bench.py --model google/embeddinggemma-2
  python run_bench.py --model ./my-ckpt --task legal
  python run_bench.py --model google/embeddinggemma-2 --json out.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def load_jsonl(name):
    p = DATA / name
    if not p.exists():
        sys.exit(f"missing {p}. Files ship with this benchmark.")
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def rank_metrics(ranks, corpus_size):
    n = len(ranks)
    return {
        "queries": n,
        "corpus": corpus_size,
        "recall_at_1": round(sum(1 for i in ranks if i < 1) / n, 4),
        "recall_at_5": round(sum(1 for i in ranks if i < 5) / n, 4),
        "recall_at_10": round(sum(1 for i in ranks if i < 10) / n, 4),
        "mrr_at_10": round(sum(1.0 / (i + 1) for i in ranks if i < 10) / n, 4),
    }


def encode(model, texts):
    return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=16)


def run_indic(model):
    rows = load_jsonl("indic_cross_script.jsonl")
    docs = [f"title: none | text: {r['passage']}" for r in rows]
    dv = encode(model, docs)
    out = {}
    for mode, key in (("roman_query", "query"), ("devanagari_query", "query_devanagari")):
        qs = [f"task: search result | query: {r[key]}" for r in rows]
        qv = encode(model, qs)
        sims = qv @ dv.T
        ranks = []
        for i in range(len(rows)):
            order = np.argsort(-sims[i])
            ranks.append(int(np.where(order == i)[0][0]))
        out[mode] = rank_metrics(ranks, len(rows))
    return out


def run_legal(model):
    queries = load_jsonl("adgm_legal_queries.jsonl")
    corpus = load_jsonl("adgm_legal_corpus.jsonl")
    pids = [c["passage_id"] for c in corpus]
    idx = {p: i for i, p in enumerate(pids)}

    docs = [f"title: none | text: {c['text']}" for c in corpus]
    dv = encode(model, docs)
    qs = [f"task: search result | query: {r['query']}" for r in queries]
    qv = encode(model, qs)
    sims = qv @ dv.T

    ranks = []
    for i, r in enumerate(queries):
        want = idx[str(r["target_passage_id"])]
        order = np.argsort(-sims[i])
        ranks.append(int(np.where(order == want)[0][0]))
    return {"legal": rank_metrics(ranks, len(pids))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id or local path")
    ap.add_argument("--task", default="all", choices=["all", "indic", "legal"])
    ap.add_argument("--json", default="", help="write results here")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        args.model,
        config_kwargs={"vision_config": None, "audio_config": None},
        device=args.device,
    )

    results = {"model": args.model}
    if args.task in ("all", "indic"):
        results["indic"] = run_indic(model)
        print("\nIndic cross script (100 pairs, larger is better)")
        for mode, m in results["indic"].items():
            print(f"  {mode:<20} R@1 {m['recall_at_1']:.3f}  R@5 {m['recall_at_5']:.3f}  MRR {m['mrr_at_10']:.3f}")

    if args.task in ("all", "legal"):
        results["legal"] = run_legal(model)
        m = results["legal"]["legal"]
        print("\nADGM legal (290 queries, 304 passages, larger is better)")
        print(f"  {'retrieval':<20} R@1 {m['recall_at_1']:.3f}  R@5 {m['recall_at_5']:.3f}  MRR {m['mrr_at_10']:.3f}")

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
