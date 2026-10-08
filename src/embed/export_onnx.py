"""Export ONNX INT8 with plain torch, no optimum needed.

optimum-onnx pins transformers under 4.58, but v2 needs transformers 5.x.
So this uses torch.onnx directly on the text encoder. Mean pooling stays
in code at serve time (same as base card: truncate plus renorm in caller).

Run: python src/embed/export_onnx.py --ckpt <dir> --out <dir> --int8 --dry-run
Writes out/model.onnx plus tokenizer files, then out/model_int8.onnx.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/best_indic")
    ap.add_argument("--out", default="onnx")
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    out = Path(args.out)
    print(f"ckpt={args.ckpt} out={out} int8={args.int8}")
    if args.dry_run:
        print("dry run ok. would export text encoder to onnx with dynamic axes.")
        return

    import torch
    from sentence_transformers import SentenceTransformer

    out.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(
        args.ckpt,
        config_kwargs={"vision_config": None, "audio_config": None},
        model_kwargs={"torch_dtype": torch.float32},
    )
    first = model._first_module()
    enc = first.auto_model
    try:
        enc = enc.merge_and_unload()
    except Exception:
        pass
    enc = enc.cpu().float()
    enc.eval()
    tok = first.tokenizer
    tok.save_pretrained(str(out))

    sample = tok(["test query for export"], padding="max_length", max_length=16, truncation=True, return_tensors="pt")
    input_ids = sample["input_ids"]
    attention_mask = sample["attention_mask"]

    class Enc(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):
            return self.m(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    torch.onnx.export(
        Enc(enc), (input_ids, attention_mask), str(out / "model.onnx"),
        input_names=["input_ids", "attention_mask"], output_names=["last_hidden"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}, "last_hidden": {0: "batch", 1: "seq"}},
        opset_version=args.opset,
    )
    print(f"saved {out / 'model.onnx'}")
    if args.int8:
        from onnxruntime.quantization import quantize_dynamic, QuantType

        quantize_dynamic(str(out / "model.onnx"), str(out / "model_int8.onnx"), weight_type=QuantType.QInt8)
        print(f"saved {out / 'model_int8.onnx'}")


if __name__ == "__main__":
    main()
