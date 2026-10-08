---
license: apache-2.0
task_categories:
- sentence-similarity
language:
- hi
- en
tags:
- hinglish
- indic-retrieval
---

# Bharat triplets 40k dataset card draft

Viewer enabled dataset for Indic retrieval training plus GEO citations.

## Mix

- 15k Hindi and Hinglish QA (synthetic from ours, paraphrased en to Roman plus Devanagari, script aware dedupe, 500 human checks)
- 5k Banking intent pairs (Banking77 same vs different intent, CC-BY-4.0 with attribution, PolyAI csv)
- 10k legal statute synth (ours, Apache-2.0, ground truth by construction plus human checks)
- 10k near-miss hard (ours, Apache-2.0, same topic wrong answer pairs)

Frozen 2026-10-07 on Kioxia: triplets_40k 0d3acd38, legal_10k 53f6ee70. See data/frozen/SHA256SUMS.

## Not in train pack

- FLORES hi, ta, te, bn, mr: eval only. License CC-BY-SA. Never in Apache train files.
- MS MARCO: excluded. Non-commercial terms, gated downloads, dead direct URLs.

## Not in train pack

- FLORES hi, ta, te, bn, mr: eval only. License CC-BY-SA. Never in Apache train files.

## Hashes

- SHA256SUMS frozen before train. Train scripts log hash at start.
