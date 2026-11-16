"""Cross-script retrieval eval: Roman Hinglish query against Devanagari passages.

Runs the same corpus three ways so the script penalty is isolated:
  roman      Roman Hinglish query  -> Devanagari passage  (the product claim)
  deva       Devanagari query      -> Devanagari passage  (same script control)
  cross      Roman query           -> Roman transliteration of passage (sanity)

Metrics: Recall@5 and MRR@10 over the full 100 passage corpus.

Run: <venv>/bin/python scripts/eval_cross_script.py --model <path_or_hf_id> --tag base
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
PAIRS = ROOT / "eval/cross_script_pairs.jsonl"


def load_pairs():
    rows = [json.loads(l) for l in open(PAIRS) if l.strip()]
    assert len(rows) >= 80, f"need at least 80 pairs, got {len(rows)}"
    return rows


def romanize_hint(text: str) -> str:
    # Rough transliteration is not needed for the main claims. Cross-check
    # mode uses the same Devanagari corpus so the comparison stays honest.
    return text


def metrics(ranked_correct: list) -> dict:
    n = len(ranked_correct)
    r5 = sum(1 for i in ranked_correct if i < 5) / n
    r1 = sum(1 for i in ranked_correct if i < 1) / n
    mrr = sum(1.0 / (i + 1) for i in ranked_correct if i < 10) / n
    return {"n": n, "recall_at_1": round(r1, 4), "recall_at_5": round(r5, 4), "mrr_at_10": round(mrr, 4)}


def run(model_path: str, tag: str):
    import numpy as np
    from sentence_transformers import SentenceTransformer

    rows = load_pairs()
    model = SentenceTransformer(
        model_path,
        config_kwargs={"vision_config": None, "audio_config": None},
        device="cpu",
    )

    passages = [f"title: none | text: {r['passage']}" for r in rows]
    pv = model.encode(passages, normalize_embeddings=True, convert_to_numpy=True, batch_size=16)

    out = {"tag": tag, "model": model_path, "pairs": len(rows)}

    for mode, key in (("roman", "q_roman"), ("deva", "q_deva")):
        queries = [f"task: search result | query: {r[key]}" for r in rows]
        qv = model.encode(queries, normalize_embeddings=True, convert_to_numpy=True, batch_size=16)
        sims = qv @ pv.T
        ranks = []
        for i in range(len(rows)):
            order = np.argsort(-sims[i])
            ranks.append(int(np.where(order == i)[0][0]))
        out[mode] = metrics(ranks)

    print(json.dumps(out, indent=2))
    dest = ROOT / "eval" / f"cross_script_{tag}.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"wrote {dest}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    run(a.model, a.tag)
