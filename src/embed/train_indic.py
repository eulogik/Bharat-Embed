"""Train Indic retrieval LoRA on text-only EmbeddingGemma-2.

Run: python src/embed/train_indic.py --data data/frozen/triplets_40k.jsonl --epochs 3 --batch 16 --dry-run
Real run needs: sentence-transformers, transformers, peft, datasets, torch with mps or cuda.
Text only. vision_config None and audio_config None. fp16 is blocked.
Default MRL: 128,256,512,768. 32 is lab only and off by default.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.embed.prefixes import DEFAULT_MRL_TRAIN, LAB_MRL_32, pick_dtype


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_triplets(path: Path, limit: int | None = None):
    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="google/embeddinggemma-2")
    ap.add_argument("--data", default="data/frozen/triplets_40k.jsonl")
    ap.add_argument("--out", default="checkpoints/best_indic")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--mrl", default=",".join(map(str, DEFAULT_MRL_TRAIN)))
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    mrl_dims = [int(x) for x in args.mrl.split(",") if x.strip()]
    if LAB_MRL_32 in mrl_dims:
        print("WARN: 32d is lab only. Quality at 32d is not guaranteed. 128d stays default.")
    assert set(mrl_dims) <= set(DEFAULT_MRL_TRAIN + [LAB_MRL_32]), f"bad mrl list: {mrl_dims}"

    dtype = pick_dtype()
    print(f"base={args.base} dtype={dtype} mrl={mrl_dims} seed={args.seed}")

    data_path = Path(args.data)
    if args.dry_run:
        print(f"dry run ok. would train {args.epochs} epochs, batch {args.batch}, lr {args.lr}, lora r{args.lora_r}, max_len {args.max_len}")
        print(f"data path: {data_path} (missing is fine in dry run)")
        if data_path.exists():
            print(f"data sha: {sha256_file(data_path)[:16]} rows check skipped in dry run")
        return

    if not data_path.exists():
        raise FileNotFoundError(f"missing frozen data: {data_path}. Freeze triplets plus SHA256SUMS first.")
    print(f"data sha: {sha256_file(data_path)}")

    # Heavy imports only for real runs so dry run stays light.
    import torch
    from sentence_transformers import SentenceTransformer, InputExample
    try:
        from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
    except ImportError:
        from sentence_transformers.losses import MultipleNegativesRankingLoss
    from torch.utils.data import DataLoader
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(args.seed)
    model = SentenceTransformer(
        args.base,
        config_kwargs={"vision_config": None, "audio_config": None},
        model_kwargs={"torch_dtype": torch.bfloat16 if dtype == "bfloat16" else torch.float32},
    )
    assert model.get_embedding_dimension() == 768
    model.max_seq_length = args.max_len
    print(f"max_seq_length set to {model.max_seq_length}")

    rows = load_triplets(data_path, limit=args.limit or None)
    examples = [InputExample(texts=[r["query"], r["pos"]] + r.get("negs", [])[:1]) for r in rows]
    loader = DataLoader(examples, batch_size=args.batch, shuffle=True)

    # LoRA on text backbone only.
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="FEATURE_EXTRACTION",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    try:
        model._first_module().auto_model = get_peft_model(model._first_module().auto_model, lora_cfg)
    except Exception as e:
        raise RuntimeError(f"LoRA wrap failed: {e}")

    loss = MultipleNegativesRankingLoss(model)
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=args.epochs,
        warmup_steps=min(500, max(50, len(loader) // 10)),
        optimizer_params={"lr": args.lr},
        show_progress_bar=True,
        output_path=args.out,
    )
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
