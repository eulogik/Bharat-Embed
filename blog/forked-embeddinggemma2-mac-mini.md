# I forked EmbeddingGemma-2 on a Mac Mini (draft)

Goal: 270M text only Indic retrieval fork, built for Hinglish RAG, $0 on M4.

## Why text only

Full 740M plus 8K ctx OOMs on 16GB. Text only selective load gives 270M. Fits with LoRA r16 plus grad accum.

## What I shipped

- Indic LoRA hero with MRL 128 default
- ONNX INT8 plus GGUF edge pack
- Legal adapter for DIFC and GST
- Eval gates locked before run one, raw JSON published

## Numbers (fill after run)

- MTEB-hi delta vs base
- MTEB-en delta
- Truncation table
- Latency on CPU and Mac

## Honest bits

- FLORES kept eval only for license reasons
- 32d stayed lab only
- fp16 never used (base warns NaN)
- Touch grace: long doc unmeasured, Tamil weaker until proven
