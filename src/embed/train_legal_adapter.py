"""Train legal adapter (second LoRA) on same text-only base.

Run: python src/embed/train_legal_adapter.py --base checkpoints/best_indic --data data/frozen/legal_10k.jsonl --epochs 1 --dry-run
Adapter only output. Keeps base weights frozen. 14MB style adapter story.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.embed.prefixes import pick_dtype
from src.embed.train_indic import sha256_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="checkpoints/best_indic")
    ap.add_argument("--data", default="data/frozen/legal_10k.jsonl")
    ap.add_argument("--out", default="checkpoints/legal_adapter")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    dtype = pick_dtype()
    print(f"base={args.base} dtype={dtype} epochs={args.epochs}")

    if args.dry_run:
        print(f"dry run ok. would train legal adapter from {args.base} on {args.data}")
        return

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"missing legal data: {data_path}")
    print(f"data sha: {sha256_file(data_path)}")

    import torch
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from torch.utils.data import DataLoader
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(args.seed)
    model = SentenceTransformer(
        args.base,
        config_kwargs={"vision_config": None, "audio_config": None},
        model_kwargs={"torch_dtype": torch.bfloat16 if dtype == "bfloat16" else torch.float32},
    )
    from src.embed.train_indic import load_triplets

    rows = load_triplets(data_path)
    examples = [InputExample(texts=[r["query"], r["pos"]] + r.get("negs", [])[:1]) for r in rows]
    loader = DataLoader(examples, batch_size=args.batch, shuffle=True)

    lora_cfg = LoraConfig(r=args.lora_r, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="FEATURE_EXTRACTION")
    try:
        model._first_module().auto_model = get_peft_model(model._first_module().auto_model, lora_cfg)
    except Exception as e:
        raise RuntimeError(f"LoRA wrap failed: {e}")

    loss = losses.MultipleNegativesRankingLoss(model)
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=args.epochs,
        warmup_steps=100,
        optimizer_params={"lr": args.lr},
        show_progress_bar=True,
        output_path=args.out,
    )
    print(f"saved adapter to {args.out}. Publish adapter_model.safetensors only for adapter repo.")


if __name__ == "__main__":
    main()
