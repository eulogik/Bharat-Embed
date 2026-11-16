# Bharat-Embed v1.x post-mortem (2026-10-14)

Goal: beat v1's Hindi IndicQA NDCG@10 of 0.7323 with real Hindi training data.
Result: two attempts, both failed. v1 stays the shipped model.

## Attempts

| Run | Rows | Config | Steps | Hindi NDCG@10 | STS | Verdict |
|---|---|---|---:|---:|---:|---|
| v1 (shipped) | 40k synth | batch 8, 3ep, len 128 | 15000 | 0.7324 | 0.6423 | baseline |
| v1.1 | 32k (MIRACL, no hard negs) | batch 4, 3ep | 24000 | 0.5807 | 0.6461 | fail |
| v1.2 | 42k (MIRACL + hard negs) | batch 4, 1ep | 10525 | 0.5749 | 0.6459 | fail |

base text-only reference: 0.7279 NDCG, 0.6430 STS.

## What ruled the theories out

Theory 1, dropped hard negatives. v1.2 restored the full 10k near-miss hard
slice and still scored 0.5749. Not the cause.

Theory 2, overfitting from too many steps. v1.1 ran 3 epochs (0.5807),
v1.2 ran 1 epoch (0.5749). Epoch count moved nothing. Not the cause.

Theory 3, MIRACL data quality. Confirmed by inspection.

## Confirmed root cause: MIRACL positives are topical, not answers

MIRACL qrels label the best available passage, not a passage that actually
answers the query. Example from the frozen v1.2 file:

 query: महाभारत का युद्ध कितने दिन तक चला था?  (how long did the war last)
 pos:   कहानियाँ हमारे महाभारत की (टीवी धारावाहिक) ...  (a TV serial about Mahabharata)

Related subject, wrong answer. MultipleNegativesRankingLoss then pulls the
query vector toward loosely related docs, which teaches topical match instead
of answer match. 2097 such rows out of 42097 (5 percent) cost 0.16 NDCG.

Rows were otherwise clean: zero pos == neg, zero tiny positives, median 97
words. This is a label quality problem, not a data hygiene problem.

## What would be needed to actually beat v1

Real Hindi question to answer-passage pairs, verified at the pair level.
MIRACL alone does not qualify. Candidates not yet tried: AI4Bharat
IndicQA-style sets (needs license check), or hand built QA pairs from
Hinglish SFT logs where the positive is known to answer the question.

No third retrain until verified pair-level data exists. Retraining on the
same flawed premise a third time would burn another 3 hours to learn nothing.

## 2026-11-16 addendum: the product premise itself is false

The v1.x retrains failed, so the last open question was whether the original
thesis held at all: Roman Hinglish query against Devanagari documents. That
was never measured until now.

Built a 100 pair eval, hand written Roman query plus the Devanagari passage
that answers it, one distinct topic per pair, full 100 doc corpus per query.
Same script control included so the script penalty isolates cleanly.

| Mode | base R@1 | base R@5 | v1 R@1 | v1 R@5 | v1 MRR |
|---|---:|---:|---:|---:|---:|
| Roman query to Devanagari passage | 0.92 | 0.97 | 0.93 | 1.00 | 0.9617 |
| Devanagari query to Devanagari passage | 1.00 | 1.00 | 1.00 | 1.00 | 1.0000 |

Base already scores 92 percent Recall@1 across scripts. The gap we planned
to fill does not exist. EmbeddingGemma-2 was trained on 100 plus languages
and handles Roman to Devanagari matching on its own.

v1 adds plus 0.01 R@1 and plus 0.03 R@5 against a 0.97 base ceiling. That is
noise, and it is not a product.

Caveat, stated honestly: this task is easy. 100 distinct topics, no near
duplicate distractors. Absolute numbers would drop in the wild for both
models. But the comparison is the point, and the comparison says parity.

Files: eval/cross_script_pairs.jsonl, scripts/eval_cross_script.py,
eval/cross_script_base.json, eval/cross_script_v1.json.

## Verdict

The fine-tune line is closed. Not because the training was bad, but because
the base model already solves the problem we set out to solve. There is no
honest marketing claim left for these weights.

What still has value: the eval harness, the ONNX parity work, the GGUF pack,
and the discipline of shipping measured numbers.

## Infrastructure fixes that came out of this

- Periodic checkpoints in train_indic.py so a restart does not lose a run.
- max_seq_length flag is now actually applied (it was dead in v1).
- MPS memory spiral documented: batch 8 at 42k rows swaps the 16G box to
  death within 10 minutes. batch 4 at len 128 is the stable envelope.
