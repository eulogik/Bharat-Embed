# Bharat triplets 40k dataset card draft

Viewer enabled dataset for Indic retrieval training plus GEO citations.

## Mix

- 15k Hindi and Hinglish QA (synthetic from ours, paraphrased en to Roman plus Devanagari, script aware dedupe, 500 human checks)
- 5k Banking intent pairs (Banking77 same vs different intent, CC-BY-4.0 with attribution)
- 10k legal statute synth (ours, Apache-2.0, ground truth by construction plus human checks)
- 10k MS MARCO hard negatives subset (pin exact revision plus license before publish)

## Not in train pack

- FLORES hi, ta, te, bn, mr: eval only. License CC-BY-SA. Never in Apache train files.

## Hashes

- SHA256SUMS frozen before train. Train scripts log hash at start.
