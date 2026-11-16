"""Legal stack eval: embed to retrieve top K, rerank with a cross-encoder.

The question this answers: base embeddings already return the right statute in
the top 5 for 98 percent of real ADGM questions. Is the remaining gap
(rank 1 precision) closed by reranking, and by how much?

Pipeline: bi-encoder top K, then cross-encoder score on query plus candidate.

Run: <venv>/bin/python scripts/eval_legal_rerank.py \
       --embedder google/embeddinggemma-2 \
       --reranker eulogik/flashrank-pro-base \
       --tag stack_base_flashrank --topk 10
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
RAW = Path("/Volumes/KIOXIA 1TB/bharat-embed/data/_raw/legal/xrefrag-dpel/data")


def load_rows():
    rows = []
    for split in ("train", "dev", "test"):
        p = RAW / f"{split}.jsonl"
        rows += [json.loads(l) for l in open(p) if l.strip()]
    return rows


def build_corpus(rows):
    corpus = {}
    for r in rows:
        corpus[str(r["source_passage_id"])] = r["source_text"]
        corpus[str(r["target_passage_id"])] = r["target_text"]
    return corpus


def metrics(ranks: list, corpus_size: int) -> dict:
    n = len(ranks)
    return {
        "n": n,
        "corpus": corpus_size,
        "recall_at_1": round(sum(1 for i in ranks if i < 1) / n, 4),
        "recall_at_5": round(sum(1 for i in ranks if i < 5) / n, 4),
        "mrr_at_10": round(sum(1.0 / (i + 1) for i in ranks if i < 10) / n, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", required=True)
    ap.add_argument("--reranker", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--maxlen", type=int, default=512)
    args = ap.parse_args()

    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    rows = load_rows()
    corpus = build_corpus(rows)
    pids = list(corpus)
    pid_index = {p: i for i, p in enumerate(pids)}
    print(f"queries {len(rows)} | corpus {len(pids)} | topk {args.topk}", flush=True)

    emb = SentenceTransformer(
        args.embedder,
        config_kwargs={"vision_config": None, "audio_config": None},
        device="cpu",
    )
    docs = [f"title: none | text: {corpus[p]}" for p in pids]
    dv = emb.encode(docs, normalize_embeddings=True, convert_to_numpy=True, batch_size=16)
    queries = [f"task: search result | query: {r['question']}" for r in rows]
    qv = emb.encode(queries, normalize_embeddings=True, convert_to_numpy=True, batch_size=16)
    sims = qv @ dv.T
    print("retrieval done, reranking", flush=True)

    tok = AutoTokenizer.from_pretrained(args.reranker)
    rk = AutoModelForSequenceClassification.from_pretrained(args.reranker, dtype=torch.float32)
    rk.eval()

    ranks = []
    for i, r in enumerate(rows):
        want = pid_index[str(r["target_passage_id"])]
        cand = np.argsort(-sims[i])[: args.topk].tolist()
        pairs = [(r["question"], corpus[pids[c]]) for c in cand]
        with torch.no_grad():
            batch = tok([q + " </s> " + d for q, d in pairs], truncation=True,
                        max_length=args.maxlen, padding=True, return_tensors="pt")
            logits = rk(**batch).logits.view(-1).float()
        order = torch.argsort(logits, descending=True).tolist()
        reranked = [cand[j] for j in order]
        rank = reranked.index(want) if want in reranked else args.topk + int(np.where(np.argsort(-sims[i]) == want)[0][0])
        ranks.append(rank)

    out = {"tag": args.tag, "embedder": args.embedder, "reranker": args.reranker, "topk": args.topk}
    out.update(metrics(ranks, len(pids)))
    print(json.dumps(out, indent=2))
    dest = ROOT / "eval" / f"legal_{args.tag}.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
