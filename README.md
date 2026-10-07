# Bharat-Embed 270M

Text only EmbeddingGemma-2 fork for Hinglish and Hindi retrieval RAG. Built on Mac mini M4. Apache-2.0.

## What is here

- Hero: Indic retrieval LoRA (270M text only, MRL 128 default)
- Edge pack: ONNX INT8 plus GGUF plus WebGPU Space
- Legal adapter: DIFC ADGM GST VAT, adapter only
- Eval gates locked before run one in eval/README.md

## Quickstart

See USAGE.md for copy paste snippets.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "eulogik/bharat-embed-270m-gemma2",
    config_kwargs={"vision_config": None, "audio_config": None},
    model_kwargs={"torch_dtype": "bfloat16"},
)
q = model.encode("GST refund kaise claim karein?", prompt_name="SearchQuery", truncate_dim=128, normalize_embeddings=True)
```

## Repo layout

- src/embed: train, export, shared prefix helpers
- scripts: triplet prep, eval slice, parity, onnx encode, hf push
- tests: prefix plus truncation gates
- eval/README.md: pre-registered gates
- data/frozen/README.md: freeze policy plus hashes
- cards: model, adapter, dataset drafts
- spaces: search plus webgpu demos
- blog: build log draft

## Train (needs heavy run, ask before starting on M4)

```bash
pip install -r requirements.txt
python scripts/prepare_triplets.py
python src/embed/train_indic.py --data data/frozen/triplets_40k.jsonl --epochs 3 --batch 16
python src/embed/train_legal_adapter.py --data data/frozen/legal_10k.jsonl --epochs 1
python scripts/eval_mteb_slice.py --ckpt checkpoints/best_indic
```

## Status

Scaffold plus gates plus docs are done and tested. Weights, ONNX, GGUF, and Spaces go live after the M4 train run plus eval.

License: Apache-2.0. Base: google/embeddinggemma-2.
