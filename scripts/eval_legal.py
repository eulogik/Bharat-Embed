"""Legal retrieval eval: ADGM data protection questions over the real statute corpus.

Source: RegNLP/xrefrag-adgm-dpel, Apache-2.0. 290 professional questions, each
with a source passage and the target passage that answers it. Both go into the
corpus, so the source acts as a genuinely hard distractor: it is topically
adjacent by construction.

Random baselines over about 317 passages: R@1 0.3 percent, R@5 1.6 percent.

Metrics: Recall@1, Recall@5, Recall@10, MRR@10.

Run: <venv>/bin/python scripts/eval_legal.py --model <path_or_hf_id> --tag base
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
        if not p.exists():
            raise FileNotFoundError(f"missing {p}. Pull with hf_hub_download first.")
        rows += [json.loads(l) for l in open(p) if l.strip()]
    return rows


def build_corpus(rows):
    corpus = {}
    for r in rows:
        corpus[str(r["source_passage_id"])] = r["source_text"]
        corpus[str(r["target_passage_id"])] = r["target_text"]
    return corpus


def metrics(ranks: list) -> dict:
    n = len(ranks)
    return {
        "n": n,
        "corpus": None,
        "recall_at_1": round(sum(1 for i in ranks if i < 1) / n, 4),
        "recall_at_5": round(sum(1 for i in ranks if i < 5) / n, 4),
        "recall_at_10": round(sum(1 for i in ranks if i < 10) / n, 4),
        "mrr_at_10": round(sum(1.0 / (i + 1) for i in ranks if i < 10) / n, 4),
    }


def run(model_path: str, tag: str):
    import numpy as np
    from sentence_transformers import SentenceTransformer

    rows = load_rows()
    corpus = build_corpus(rows)
    pids = list(corpus)
    pid_index = {p: i for i, p in enumerate(pids)}
    print(f"queries {len(rows)} | corpus {len(pids)}")

    model = SentenceTransformer(
        model_path,
        config_kwargs={"vision_config": None, "audio_config": None},
        device="cpu",
    )

    docs = [f"title: none | text: {corpus[p]}" for p in pids]
    dv = model.encode(docs, normalize_embeddings=True, convert_to_numpy=True, batch_size=16)

    queries = [f"task: search result | query: {r['question']}" for r in rows]
    qv = model.encode(queries, normalize_embeddings=True, convert_to_numpy=True, batch_size=16)

    sims = qv @ dv.T
    ranks = []
    for i, r in enumerate(rows):
        want = pid_index[str(r["target_passage_id"])]
        order = np.argsort(-sims[i])
        ranks.append(int(np.where(order == want)[0][0]))

    out = {"tag": tag, "model": model_path, "pairs": len(rows), "corpus": len(pids)}
    out.update(metrics(ranks))
    print(json.dumps(out, indent=2))

    dest = ROOT / "eval" / f"legal_{tag}.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"wrote {dest}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    run(a.model, a.tag)
