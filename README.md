# Bharat-Embed

What started as a Hindi/Hinglish embedding fine-tune, and ended as a measured
negative result plus a benchmark that is worth keeping.

**Start with the benchmark if you are here to decide whether to fine-tune an
embedding model for Indic or legal retrieval:**
[Bharat Retrieval Benchmark](https://huggingface.co/datasets/eulogik/bharat-retrieval-benchmark)

## The finding

We fine-tuned EmbeddingGemma-2 twice, for Hindi/Hinglish and for legal, then
measured both against the base model with real queries.

| Model | Indic Roman R@1 | Indic Roman R@5 | ADGM legal R@1 | ADGM legal R@5 |
|---|---:|---:|---:|---:|
| `google/embeddinggemma-2` | 0.920 | 0.970 | 0.397 | 0.979 |
| Bharat-Embed v1 (Hindi LoRA) | 0.930 | 1.000 | 0.428 | 0.966 |
| Bharat legal adapter | not run | not run | 0.007 | 0.031 |

Three conclusions, all measured:

1. The base model already handles Roman Hinglish queries against Devanagari
   passages at 92 percent Recall@1. The cross-script gap we set out to fill
   does not exist.
2. The base model already puts the right statute in the top 5 for 98 percent of
   real ADGM legal questions. No legal retrieval gap either.
3. The legal adapter is a catastrophic failure. R@5 of 3.1 percent, barely
   above random. It is private now, with the failure printed on its card.

The Hindi LoRA is parity, not improvement. It is published because the
measurements are real, not because it beats the base model. If you want a
Hindi-capable 270M embedding model, use `google/embeddinggemma-2`.

## What is actually useful here

- [Bharat Retrieval Benchmark](benchmarks/README.md): 100 hand-built Indic
  cross-script pairs plus 290 real ADGM regulatory queries, with a runner that
  works on any sentence-transformers model in about two minutes on CPU.
- ONNX INT8 export with a parity gate, drift 0.0066 on the shipped weights.
- GGUF quants at BF16, Q8_0, Q6_K, Q4_K_M, built with the official ggml path.
- The full evaluation trail, including the failures, in
  [eval/V1X_POSTMORTEM.md](eval/V1X_POSTMORTEM.md).

## Run the benchmark

```bash
pip install sentence-transformers
python benchmarks/run_bench.py --model google/embeddinggemma-2
```

```
Indic cross script (100 pairs, larger is better)
  roman_query          R@1 0.920  R@5 0.970  MRR 0.946
  devanagari_query     R@1 1.000  R@5 1.000  MRR 1.000

ADGM legal (290 queries, 304 passages, larger is better)
  retrieval            R@1 0.397  R@5 0.979  MRR 0.650
```

## If you still want the weights

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "eulogik/bharat-embed-270m-gemma2",
    config_kwargs={"vision_config": None, "audio_config": None},
    model_kwargs={"torch_dtype": "bfloat16"},
)
q = model.encode("GST refund kaise claim karein?", prompt_name="SearchQuery",
                 truncate_dim=256, normalize_embeddings=True)
```

Remember what the table above says: this is a text-only 270M pack at parity
with base. Measured Hindi NDCG is 0.7324 against base 0.7279. Use it for the
small footprint and the ONNX/GGUF packaging, not for accuracy.

## Repo layout

- `benchmarks` the published benchmark and runner
- `src/embed` train, export, shared prefix helpers
- `scripts` data freeze, eval, parity, charts, HF push
- `eval` measured results and the post-mortem
- `tests` prefix and ship gates, 12 green
- `cards` model and dataset cards

## Honest limits

- Text only. No vision, video, or audio.
- Hindi and Hinglish measured. Tamil, Telugu, Bengali unmeasured.
- Trained at 128 tokens. Long document behavior unmeasured.
- The 100 and 290 query tasks catch regressions, they do not rank models within
  a point.

Built by [Eulogik](https://github.com/eulogik). Apache-2.0. Base model
`google/embeddinggemma-2`.
