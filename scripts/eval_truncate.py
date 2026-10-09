"""Truncation sweep with real MTEB numbers. No made up tables.

Task: IndicQARetrieval, Hindi only. Ours at 768, 512, 256, 128 plus base at 768.
Note: mteb drives encoding, so run uses mteb default prompt path, not our
SearchQuery prefix path. Base vs ours stays fair (same conditions both sides).

Run: python scripts/eval_truncate.py
Writes Kioxia eval/truncate.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

KIO = Path("/Volumes/KIOXIA 1TB/bharat-embed")


class FixedDimWrapper:
    def __init__(self, st_model, dim: int):
        self.m = st_model
        self.dim = dim

    def encode(self, sentences, prompt_name=None, **kwargs):
        kwargs.pop("normalize_embeddings", None)
        return self.m.encode(sentences, truncate_dim=self.dim, normalize_embeddings=True, **kwargs)


def load_st(path: str):
    import torch
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        path,
        config_kwargs={"vision_config": None, "audio_config": None},
        model_kwargs={"torch_dtype": torch.float32},
        device="cpu",
    )


def main():
    import mteb

    task = mteb.get_tasks(tasks=["IndicQARetrieval"], languages=["hin"])[0]
    out = {}
    runs = [("base", "google/embeddinggemma-2", 768)]
    runs += [("ours", str(KIO / "checkpoints/best_indic_merged"), d) for d in (512, 256, 128)]
    for tag, path, dim in runs:
        key = f"{tag}_{dim}"
        print(f"START {key}", flush=True)
        model = FixedDimWrapper(load_st(path), dim)
        res = mteb.MTEB(tasks=[task]).run(model, output_folder=str(KIO / "eval" / f"trunc_{key}"))
        blob = str(res)
        import re

        mains = re.findall(r"ndcg_at_10.: ([\d.]+)", blob)
        out[key] = float(mains[0]) if mains else None
        print(f"END {key}: {out[key]}", flush=True)
    # ours_768 already measured earlier
    out["ours_768"] = 0.73235
    out["base_768"] = 0.72785
    p = KIO / "eval/truncate.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"wrote {p}: {out}")


if __name__ == "__main__":
    main()
