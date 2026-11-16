# Bharat Retrieval Benchmark

Two small retrieval benchmarks for the cases people actually get wrong when
they ship embedding models to India and the Gulf.

Both are built from real queries with known correct passages. Both are cheap to
run, CPU only, no GPU needed, about two minutes per model.

- **Indic cross script**: 100 Roman Hinglish queries, each paired with the
  Devanagari passage that answers it. Tests whether a model can match
  `gst refund kaise karein` to a passage written in Devanagari.
- **ADGM legal**: 290 professional questions over ADGM data protection law with
  a 304 passage corpus. Tests regulatory retrieval where every query has a
  topically adjacent decoy in the corpus.

## Why this exists

We fine-tuned EmbeddingGemma-2 into a Hindi/Hinglish model and a legal adapter,
shipped them, then measured them properly. Both fine-tunes were worse than doing
nothing, and the base model already solved both jobs.

The evidence:

| Model | Indic Roman R@1 | Indic Roman R@5 | Legal R@1 | Legal R@5 |
|---|---:|---:|---:|---:|
| `google/embeddinggemma-2` | 0.920 | 0.970 | 0.397 | 0.979 |
| Bharat-Embed v1 (our Hindi LoRA) | 0.930 | 1.000 | 0.428 | 0.966 |
| our legal adapter | not run | not run | 0.007 | 0.031 |

That third row is a real failure, not a typo. One epoch on synthetic legal
pairs destroyed the model. It is now private with the measurement on its card.

We publish the benchmark instead of the weights, because the benchmark is the
part that was missing. If you are about to fine-tune an embedding model for a
language or a domain, run this first. If the base model already scores well,
you just saved three weeks.

## Run it

```bash
pip install sentence-transformers
python run_bench.py --model google/embeddinggemma-2
python run_bench.py --model ./my-finetune --task legal
python run_bench.py --model google/embeddinggemma-2 --json results.json
```

Sample output:

```
Indic cross script (100 pairs, larger is better)
  roman_query          R@1 0.920  R@5 0.970  MRR 0.946
  devanagari_query     R@1 1.000  R@5 1.000  MRR 1.000

ADGM legal (290 queries, 304 passages, larger is better)
  retrieval            R@1 0.397  R@5 0.979  MRR 0.650
```

The `devanagari_query` row is the control. If roman and devanagari scores are
close, the model handles script mixing and you do not have a problem to fix.

## Reference points

Random baselines, so you can tell signal from noise:

| Task | Corpus | Random R@1 | Random R@5 |
|---|---:|---:|---:|
| Indic cross script | 100 passages | 0.010 | 0.050 |
| ADGM legal | 304 passages | 0.003 | 0.016 |

## What the tasks are good for

- Catching a model that looks fine on paper but breaks when users type Roman
  script, which is most Hinglish users.
- Catching catastrophic forgetting after a domain fine-tune. That is exactly
  what the legal adapter row above shows.
- A fast go or no-go gate before you spend money on training.

## What they are not

- Not a leaderboard. 100 and 290 queries are enough to catch a regression, not
  enough to rank models within a point of each other.
- Not MTEB. These are narrow, deliberately adversarial slices.
- The ADGM legal R@1 ceiling of about 0.40 is the task, not a defect. The
  corpus contains the passage that cross-references the answer as well as the
  answer itself. Normal RAG feeds the top 5 and lets the model read both.

## Data

| File | Rows | Source | License |
|---|---:|---|---|
| `data/indic_cross_script.jsonl` | 100 | written by Eulogik | Apache-2.0 |
| `data/adgm_legal_queries.jsonl` | 290 | RegNLP/xrefrag-adgm-dpel | Apache-2.0 |
| `data/adgm_legal_corpus.jsonl` | 304 | RegNLP/xrefrag-adgm-dpel | Apache-2.0 |

The legal portion is derived from `RegNLP/xrefrag-adgm-dpel` on the Hugging
Face Hub, which is Apache-2.0. Attribution kept here and in the file metadata.

The Indic pairs are ours. Each pair is a Roman query, the Devanagari form of the
same question, and a Devanagari passage that answers it. One distinct topic per
pair. If you find a wrong pair, open an issue and it gets fixed.

## Citation

If this saves you a training run, link the repo. That is enough.

Built by [Eulogik](https://github.com/eulogik). Apache-2.0.
