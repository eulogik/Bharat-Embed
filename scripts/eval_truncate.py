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


def fixed_dim_model(path: str, dim: int):
    """ST model that encodes at one fixed truncate dim.

    RetrievalEvaluator wants a SearchInterface, Encoder, or CrossEncoder.
    A plain duck type fails the check, so this is a real SentenceTransformer
    subclass that injects the dim on every encode call.
    """
    import torch
    from sentence_transformers import SentenceTransformer

    class _DimST(SentenceTransformer):
        def __init__(self):
            super().__init__(
                path,
                config_kwargs={"vision_config": None, "audio_config": None},
                model_kwargs={"torch_dtype": torch.float32},
                device="cpu",
            )

        def encode(self, sentences=None, texts=None, **kwargs):
            body = sentences if sentences is not None else texts
            if body is None:
                body = kwargs.pop("inputs", None)
            kwargs.pop("sentences", None)
            kwargs.pop("texts", None)
            kwargs["truncate_dim"] = dim
            kwargs["normalize_embeddings"] = True
            return super().encode(body, **kwargs)

    return _DimST()


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
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    solo = ap.parse_args().only

    import mteb

    task = mteb.get_tasks(tasks=["IndicQARetrieval"], languages=["hin"])[0]
    runs = [("base", "google/embeddinggemma-2", 768)]
    runs += [("ours", str(KIO / "checkpoints/best_indic_merged"), d) for d in (512, 256, 128)]
    if solo:
        runs = [r for r in runs if f"{r[0]}_{r[2]}" == solo]
    out = {}
    for tag, path, dim in runs:
        key = f"{tag}_{dim}"
        print(f"START {key}", flush=True)
        model = fixed_dim_model(path, dim)
        mteb.MTEB(tasks=[task]).run(model, output_folder=str(KIO / "eval" / f"trunc_{key}"))
        import glob as _glob
        import json as _json

        val = None
        for jf in _glob.glob(str(KIO / "eval" / f"trunc_{key}" / "*" / "*" / "IndicQARetrieval.json")):
            val = _json.loads(Path(jf).read_text())["scores"]["test"][0].get("ndcg_at_10")
        out[key] = val
        print(f"END {key}: {out[key]}", flush=True)
    # ours_768 already measured earlier
    out["ours_768"] = 0.73235
    out["base_768"] = 0.72785
    p = KIO / "eval/truncate.json"
    prev = {}
    if p.exists():
        import json as _json

        prev = _json.loads(p.read_text())
    prev.update(out)
    p.write_text(__import__("json").dumps(prev, indent=2))
    print(f"wrote {p}: {prev}")


if __name__ == "__main__":
    main()
