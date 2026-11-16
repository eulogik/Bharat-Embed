---
library_name: sentence-transformers
pipeline_tag: sentence-similarity
license: apache-2.0
base_model: eulogik/bharat-embed-270m-gemma2
language:
- en
- hi
- ar
tags:
- legal
- difc
- gst
---

# Bharat legal adapter: WITHDRAWN, failed measurement

This adapter is withdrawn. Do not use it. It was measured on real regulatory
data and it is broken.

Evaluated on RegNLP/xrefrag-adgm-dpel, 290 professional ADGM data protection
questions over a 304 passage corpus:

| Model | Recall@1 | Recall@5 | MRR@10 |
|---|---:|---:|---:|
| base text only | 0.3966 | 0.9793 | 0.6503 |
| this legal adapter | 0.0069 | 0.0310 | 0.0171 |

Recall@5 of 3.1 percent is barely above the 1.6 percent random baseline. One
epoch on templated synthetic GST pairs destroyed the model's general ability.
The base model already retrieves the right statute in the top 5 for 98 percent
of these questions, so there was nothing to improve in the first place.

Card kept for the record, not for use. The evaluation is reproducible with
scripts/eval_legal.py in github.com/eulogik/Bharat-Embed.

Same base as Bharat-Embed 270M plus legal LoRA.

## Use

- DIFC, ADGM, GST, VAT statute retrieval. EN, AR, HI queries.
- Swap adapter on same base. No extra base weights.
- Train: 1 epoch on legal_10k only.

## Gate

- Statute recall at 5 plus 0.10 over generic on held out DIFC and GST queries.
- Human spot check before publish. Synthetic queries alone are not proof.

## License

Apache-2.0 for our synthetic slice. Cite Banking and MARCO sources where reused.
