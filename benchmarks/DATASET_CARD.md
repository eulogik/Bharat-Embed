---
license: apache-2.0
task_categories:
- text-retrieval
- feature-extraction
language:
- hi
- en
tags:
- benchmark
- retrieval
- hinglish
- indic
- legal
- embedding
- embeddinggemma
pretty_name: Bharat Retrieval Benchmark
---

# Bharat Retrieval Benchmark

Two small retrieval benchmarks for the cases people get wrong when shipping
embedding models to India and the Gulf. CPU only, about two minutes per model.

- **Indic cross script**: 100 Roman Hinglish queries paired with the Devanagari
  passage that answers them. Catches models that work in Devanagari but break
  when the user types Roman script.
- **ADGM legal**: 290 professional questions over ADGM data protection law,
  304 passage corpus, with the cross-referencing source passage sitting in the
  corpus as a hard decoy.

## Reference results

| Model | Indic Roman R@1 | Indic Roman R@5 | Legal R@1 | Legal R@5 |
|---|---:|---:|---:|---:|
| `google/embeddinggemma-2` | 0.920 | 0.970 | 0.397 | 0.979 |
| Bharat-Embed v1 (Hindi LoRA) | 0.930 | 1.000 | 0.428 | 0.966 |
| our legal adapter | not run | not run | 0.007 | 0.031 |

The legal adapter row is a real measured failure. One epoch on synthetic legal
pairs destroyed the model. It is published here as a cautionary data point.

## Files

| File | Rows | What |
|---|---:|---|
| `data/indic_cross_script.jsonl` | 100 | query, query_devanagari, passage |
| `data/adgm_legal_queries.jsonl` | 290 | query, target_passage_id |
| `data/adgm_legal_corpus.jsonl` | 304 | passage_id, text |

## Run

```bash
pip install sentence-transformers
python run_bench.py --model google/embeddinggemma-2
```

The harness lives at `github.com/eulogik/Bharat-Embed` under `benchmarks/`.

## Method

Indic task: corpus is all 100 passages. Encode with `task: search result |
query: {q}` and `title: none | text: {p}`, rank by cosine, score R@1, R@5, MRR.

Legal task: corpus is all 304 passages. Same prefix scheme. Each query is scored
against whether the judged target passage lands in the top K.

## Limits

100 and 290 queries catch regressions. They do not rank models within a point.
The legal R@1 of about 0.40 is the task being adversarial by design, since the
corpus holds both the answer and the passage that cross-references it.

## Data provenance

The Indic pairs are written by Eulogik. The legal portion is derived from
`RegNLP/xrefrag-adgm-dpel` (Apache-2.0), attribution kept.

Built by [Eulogik](https://github.com/eulogik). Apache-2.0.
