---
library_name: sentence-transformers
pipeline_tag: sentence-similarity
license: apache-2.0
base_model: google/embeddinggemma-2
language:
- hi
- en
tags:
- hinglish
- indic-rag
- on-device
- mrl
---

# Bharat-Embed 270M model card draft

## Direct answer

Bharat-Embed 270M is a 270M text only EmbeddingGemma-2 fork for Hinglish and Hindi retrieval RAG. Loads without vision or audio. Runs on CPU, ONNX, and GGUF. MRL to 128d. Apache-2.0. Use after BM25 or dense top 100 with flashrank-pro-base reranker.

## Base

- base_model: google/embeddinggemma-2
- license: apache-2.0
- library_name: sentence-transformers
- pipeline_tag: sentence-similarity
- text only: config_kwargs vision None audio None, 270M effective
- dims: 768 native, MRL 128, 256, 512. 32 is lab only.

## Quickstart

See USAGE.md. Query prefix SearchQuery. Doc form title plus text. Same truncate dim both sides. Normalize True.

## Eval (fill after run)

- MTEB-hi slice delta vs base
- MTEB-en delta (gate over minus 0.01)
- Truncation table 768 down to 128 plus 32 lab
- bf16 vs INT8 vs Q4_K_M delta
- Legal recall at 5 for adapter repo

## Limits

- Text only. No image, video, or audio.
- Hindi and Hinglish first. Tamil and Telugu weaker until measured.
- Trained at 512. 8K ctx kept in arch but long doc unmeasured.
- Scores are ranking scores, not calibrated probs.

## Keywords

embeddinggemma2 hindi embedding, hinglish embedding, indic rag embedding, on-device embedding, onnx embedding, gguf embedding ollama, qdrant hindi search, mrl matryoshka embedding, uae law embedding, difc rag
