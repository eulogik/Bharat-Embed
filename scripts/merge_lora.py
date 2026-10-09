"""Merge PEFT LoRA weights into base weights with standard key names.

ST saved full tensors in PEFT naming (base_layer plus lora_A/B, no adapter
config). Plain reloads then miss attention keys and random init them.
This folds each adapter into its base and writes clean standard keys.

Run: python scripts/merge_lora.py --src <ckpt> --dst <merged> --r 16 --alpha 32
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    args = ap.parse_args()

    import torch
    from safetensors.torch import load_file, save_file

    src, dst = Path(args.src), Path(args.dst)
    sd = load_file(str(src / "model.safetensors"))
    scaling = args.alpha / args.r
    out = {}
    merged = 0
    for k, v in sd.items():
        if k.endswith(".base_layer.weight"):
            stem = k[: -len(".base_layer.weight")]
            a = sd.get(stem + ".lora_A.default.weight", sd.get(stem + ".lora_A.weight"))
            b = sd.get(stem + ".lora_B.default.weight", sd.get(stem + ".lora_B.weight"))
            if a is not None and b is not None:
                out[stem + ".weight"] = v + (b @ a) * scaling
                merged += 1
            else:
                out[stem + ".weight"] = v
        elif ".lora_" in k:
            continue
        else:
            out[k] = v
    print(f"merged {merged} adapters, {len(out)} clean tensors")

    dst.mkdir(parents=True, exist_ok=True)
    save_file(out, str(dst / "model.safetensors"))
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                 "modules.json", "config_sentence_transformers.json", "sentence_bert_config.json",
                 "chat_template.jinja", "processor_config.json", "preprocessor_config.json",
                 "tokenizer.model", "sentencepiece.model"):
        if (src / name).exists():
            shutil.copy(src / name, dst / name)
    for sub in ("1_Pooling", "2_Normalize"):
        if (src / sub).exists():
            shutil.copytree(src / sub, dst / sub, dirs_exist_ok=True)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
