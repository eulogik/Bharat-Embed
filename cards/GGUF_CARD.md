---
license: apache-2.0
pipeline_tag: feature-extraction
tags:
- gguf
- llama-cpp
- ollama
- embedding
- hindi
- hinglish
base_model:
- eulogik/bharat-embed-270m-gemma2
language:
- hi
- en
---

# Bharat-Embed GGUF

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
![Q4 182M](https://img.shields.io/badge/Q4_K_M-182M-green)
![Built by Eulogik](https://img.shields.io/badge/Built%20by-Eulogik-7c3aed)

GGUF quants of [eulogik/bharat-embed-270m-gemma2](https://huggingface.co/eulogik/bharat-embed-270m-gemma2) for llama.cpp, Ollama, and LM Studio. Converted with the official ggml-org script from merged clean weights.

## Files

| File | Size | Use |
|---|---|---|
| `bharat-embed-BF16.gguf` | 558M | best quality |
| `bharat-embed-Q8_0.gguf` | 310M | near lossless |
| `bharat-embed-Q6_K.gguf` | 246M | balanced |
| `bharat-embed-Q4_K_M.gguf` | 182M | smallest, cosine 0.75 on Hindi spot check |

## Use

```bash
ollama run hf.co/eulogik/bharat-embed-270m-gemma2-GGUF:Q4_K_M
./build/bin/llama-embedding -m bharat-embed-Q4_K_M.gguf --pooling mean -p "rental agreement ki security deposit limit?"
```

Pooling is mean. Q4 spot check (Hindi query vs doc): cosine 0.75, sane ordering.

Built by [Eulogik](https://github.com/eulogik). Main model card has measured tables.
