"""Step 2 smoke: pull base to Kioxia cache, load text only, verify dim plus prefix plus truncate.

Writes: /Volumes/KIOXIA 1TB/bharat-embed/smoke_result.json
Run with venv python. Heavy download about 3GB lives in Kioxia hf_cache.
"""

import json
import os
import sys
from pathlib import Path

KIO = Path("/Volumes/KIOXIA 1TB/bharat-embed")
os.environ["HF_HOME"] = str(KIO / "hf_cache")
os.environ["TRANSFORMERS_CACHE"] = str(KIO / "hf_cache")
os.environ["HF_HUB_CACHE"] = str(KIO / "hf_cache")
os.environ["TMPDIR"] = str(KIO / "tmp")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.embed.prefixes import format_query, format_doc, check_same_dim


def main():
    import torch
    from sentence_transformers import SentenceTransformer

    print(f"mps built={torch.backends.mps.is_built()} avail={torch.backends.mps.is_available()}")
    print(f"torch={torch.__version__}")

    # fp32 default on Mac per plan (bf16 only if proven). Smoke uses fp32.
    model = SentenceTransformer(
        "google/embeddinggemma-2",
        config_kwargs={"vision_config": None, "audio_config": None},
        model_kwargs={"torch_dtype": torch.float32},
    )
    dim = model.get_sentence_embedding_dimension()
    print(f"dim={dim}")
    assert dim == 768, f"want 768, got {dim}"

    q_raw = "GST refund kaise claim karein?"
    d_raw = "apply through portal with invoice proof"
    q = format_query("SearchQuery", q_raw)
    d = format_doc("GST refund", d_raw)
    qv = model.encode(q, truncate_dim=128, normalize_embeddings=True)
    dv = model.encode(d, truncate_dim=128, normalize_embeddings=True)
    check_same_dim(len(qv), len(dv))
    assert len(qv) == 128

    import numpy as np

    sim = float(np.dot(qv, dv))
    print(f"sim128={sim:.4f}")
    assert -1.0 <= sim <= 1.0

    out = KIO / "smoke_result.json"
    out.write_text(json.dumps({"dim": dim, "q_len": len(qv), "sim128": sim, "status": "pass"}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
