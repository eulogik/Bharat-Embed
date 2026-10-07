# EMBEDDINGGEMMA-2 VARIANT PLAN — Bharat-Embed Family
## Quick, viral, Mac-mini-buildable fork of `google/embeddinggemma-2`

**Status:** approved for build | **Owner:** Eulogik | **Hardware:** Mac mini M4 10-core, 16GB LPDDR5 | **License:** Apache-2.0 (base is Apache-2.0) | **Date:** 2026-10-07

> TL;DR: don't fine-tune 740M multimodal. Ship text-only 270M family: **Indic-retrieval LoRA (hero) + ONNX/GGUF/LiteRT/WebGPU edge pack (download hack) + legal adapter (enterprise moat)**. 4-5 days, $0, all on M4.

---

## 1. Deep research: what actually gets downloaded in embeddings

### 1.1 Base facts (verified 2026-10-07)

`google/embeddinggemma-2`: 740M total = 270M text (130M transformer + 140M embedder) + 170M vision + 300M audio. Unified 768d, MRL truncation 128/256/512/768, 8k ctx, 100+ langs, task prefixes (`SearchQuery`, `Document`, `Classification`, ...), selective loading (`config_kwargs={"vision_config": None, "audio_config": None}` -> 270M), bf16 (never fp16), sentence-transformers compatible. Apache-2.0. 669 likes in 16h.

### 1.2 What ranks by downloads (feature-extraction, sort=downloads)

| Model | Params | Pulls/mo | Why it wins |
|---|---|---|---|
| `BAAI/bge-small-en-v1.5` | 33M | 62.3M | tiny + fast + permissive + sentence-transformers + ONNX-friendly |
| `sentence-transformers/all-MiniLM-L6-v2` | 22.7M | 234M (sentence-similarity) | smallest viable, default in every tutorial |
| `Xenova/all-MiniLM-L6-v2` | 22M ONNX/Web | 3.87M | **distribution = downloads**: ONNX + WebGPU + Transformers.js |
| `Qwen/Qwen3-Embedding-0.6B` | 0.6B | 9.25M | small variant of big family wins over 4B/8B siblings |
| `intfloat/multilingual-e5-large` | 0.6B | 7.16M | multilingual + instruct prefixes |
| `BAAI/bge-reranker-large` | 0.6B | 2.36M | paired reranker doubles ecosystem |
| `jinaai/jina-embeddings-v5-text-nano-retrieval-GGUF` | 239M GGUF | 6.5k fast-growing | 239M nano + MRL 32-768 + task-targeted (retrieval vs matching) + `ollama run` / `llama-embedding`. Weakness: CC-BY-NC-4.0 (not commercial) |
| `litert-community/embeddinggemma-300m` | 300M LiteRT | 1.4k in 5h | **edge pack alone trends**: NPU 7.8ms, CPU XNNPACK, Android/iOS/WebGPU Spaces |
| `Qwen/Qwen3-VL-Embedding-2B/8B` | 2-8B multimodal | 1.1-1.5M | vision adds pull but needs GPU, downloads 6x smaller than 0.6B text |

### 1.3 Laws (embeddings edition)

1. **Tiny beats large 10-25x on pulls.** 33M > 8B. Ship 270M text-only, not 740M full.
2. **Distribution = downloads.** Xenova (ONNX/Web), Jina (GGUF + Ollama + LM Studio), LiteRT (Android/NPU + WebGPU Spaces). All three on day-1.
3. **Task-targeted beats generic.** Jina v5 splits retrieval/matching/clustering/classification. Single `retrieval` nano beats one-size-fits-all.
4. **Apache-2.0 beats CC-BY-NC.** Jina v5's biggest weakness is non-commercial license. Open alternative steals commercial RAG.
5. **Small + reranker = stack.** bge-embed + bge-reranker, Jina-embed + reranker. Eulogik already has `flashrank-pro-base` 149M ModernBERT reranker (0.3314 NDCG, Apache-2.0) - pair it.
6. **MRL + truncation = cost story.** 768d -> 128d = 6x Qdrant storage cut. Publish truncation table like base does.
7. **Multilingual + prefixes = AEO.** Task instruction prefixes (`task: search result | query:`) are the prompt-format.json of embeddings. Byte-exact docs get cited by agents.

---

## 2. Options scored (beyond Indic) — Mac-mini feasible only

| # | Variant | Params/train | Data | Days | Downloads | Moat | Eulogik reuse | Score |
|---|---|---|---|---|---|---|---|---|
| A | **Indic/Hinglish retrieval LoRA** (Roman+Devanagari code-switch) | 270M LoRA r16, MPS hours | 40k triplets AI4Bharat/FLORES/Banking77 | 4 | medium-high (Hindi RAG niche) | high (only Hinglish task-prefix model) | Bharat-Tiny-LLM, PolyWhisper ortho norm | 8/10 |
| B | **Zero-train edge pack** (ONNX INT8 + GGUF + LiteRT + WebGPU) | 0 train, export only | - | 1 | **highest** (Xenova pattern) | low alone (Google/LiteRT will ship v2 pack) | - | 7/10 alone, 10/10 as companion |
| C | **Legal adapter** (UAE/India: DIFC/ADGM/GST/VAT, EN/AR/HI) | 270M LoRA, same cost as A | 20k statute triplets synthetic+LexRAG | +2 | low (niche) but **revenue high** | very high (no legal embeddinggemma fork) | LexRAG, TinyDoc-VLM, Signizy | 9/10 business, 6/10 pulls |
| D | **Nano-retrieval Apache-2.0** (Jina-v5 killer, MRL to 32d) | 270M + teacher distill | 100k MS MARCO hard negs (reuse flashrank pipeline) | 5-6 | high (retrieval is #1 job, 62M bge-small proof) | high (open vs CC-BY-NC) | flashrank-pro training + eval harness | 8/10 |
| E | VisDoc (vision 170M + text, contracts/receipts) | 440M joint, heavy | DocVQA + TinyDoc real-data | 10+ | medium | medium | TinyDoc-VLM | 5/10 quick (not quick) |
| F | Audio/voice (300M audio, Indic voice search) | 300M + audio data | IndicVoices-ST | 10+ | low-medium | high but slow | PolyWhisper | 4/10 quick |
| G | Guard/tool-call embed (Suraksha-Embed) | 270M | MCP traces + OpenTrustBench | 3 | low (tiny TAM) | medium | Suraksha + OpenTrustBench | 5/10 |

**Winner: A + B + C as one family, D merged into A.** Hero = A with D's retrieval-only + MRL-32 training (so it *is* the Apache-2.0 nano-retrieval for Indic). B ships same day for download hack. C ships as second PEFT adapter on same base (+2 days, no extra base weights).

Why combo beats pure-Indic: pure-Indic misses (a) edge-pack funnel (`onnx/gguf/litert embeddinggemma` searches), (b) Jina CC-BY-NC commercial-steal angle, (c) LexRAG enterprise funnel (`uae law embedding`, `difc rag`). Combo captures all three with one training run + two exports.

**Not doing:** E/F/G now (heavy or tiny TAM). Doc/audio become v1.1 after traction.

---

## 3. Product spec

### 3.1 Repos

- `eulogik/bharat-embed-270m-gemma2` — text-only 270M + Indic-retrieval LoRA (merged fp16 + adapter), ONNX INT8, eval JSONs. Tags: `embeddinggemma2 hindi hinglish indic-rag on-device onnx mrl qdrant apache-2.0`.
- `eulogik/bharat-embed-270m-gemma2-GGUF` — Q8_0 / Q6_K / Q4_K_M + KL + MTEB-hi per-quant table (copy humanizer GGUF page pattern).
- `eulogik/bharat-embed-triplets-40k` — Viewer-enabled dataset (SEO + GEO citations).
- `eulogik/bharat-legal-embed-270m` (adapter-only repo, v1.1) — same base + legal LoRA, `adapter_model.safetensors` 14MB pattern (copy PolyWhisper per-language adapter story).
- Spaces: `eulogik/bharat-embed-search` (Hinglish query -> Qdrant hits + 128d vs 768d toggle), `eulogik/bharat-embed-webgpu` (Transformers.js in-browser, copy `webml-community/embeddinggemma-2-webgpu`).

### 3.2 Direct answer (card first lines, AEO)

> **Bharat-Embed 270M is a 270M text-only EmbeddingGemma-2 fork for Hinglish/Hindi retrieval RAG.** Loads without vision/audio, runs CPU/ONNX/GGUF, MRL to 128d/32d, Apache-2.0. Use after BM25 or dense top-100 with `flashrank-pro-base` reranker.

Find with: `embeddinggemma2 hindi hinglish on-device onnx gguf qdrant mrl indic rag apache 2.0`.

### 3.3 Quickstart (must work day-1)

```python
from sentence_transformers import SentenceTransformer
# text-only: 270M, no vision/audio RAM
model = SentenceTransformer("eulogik/bharat-embed-270m-gemma2",
  config_kwargs={"vision_config": None, "audio_config": None},
  model_kwargs={"torch_dtype": "bfloat16"})
q = model.encode("GST refund kaise claim karein?", prompt_name="SearchQuery",
  truncate_dim=128, normalize_embeddings=True)
d = model.encode("title: GST refund | text: ...", truncate_dim=128, normalize_embeddings=True)
print(model.similarity(q, d))
```

```bash
# ONNX CPU (no torch needed at serve)
pip install onnxruntime qdrant-client
python scripts/encode_onnx.py --model onnx/int8 --queries queries.txt --truncate 128
# GGUF / Ollama / llama.cpp
ollama run hf.co/eulogik/bharat-embed-270m-gemma2-GGUF:Q4_K_M
./build/bin/llama-embedding -m bharat-embed-Q4_K_M.gguf --pooling mean -p "rental agreement ki security deposit limit?"
```

Pairing: retrieve 100 with Bharat-Embed 128d -> rerank top-20 with `eulogik/flashrank-pro-base` (149M, CPU 50ms) -> generate with `Bharat-Tiny-LLM-v3`.

---

## 4. Training recipe (M4 16GB, $0)

### 4.1 Why text-only fits

Full 740M + 8k ctx OOMs on 16GB. Text-only 270M bf16 ~0.6GB weights + LoRA r16 (~8M trainable) + Adam8bit + batch 16 x 512 tokens fits unified memory. Vision/audio towers never load. MRL head trains jointly (keep base MRL loss).

```bash
pip install -U sentence-transformers transformers peft datasets onnx onnxruntime huggingface_hub
python -c "from sentence_transformers import SentenceTransformer; m=SentenceTransformer('google/embeddinggemma-2', config_kwargs={'vision_config': None, 'audio_config': None}); print(m.get_sentence_embedding_dimension())"  # 768
```

### 4.2 Data (40k triplets, hash-frozen)

| Slice | N | Source | License |
|---|---|---|---|
| Hindi/Hinglish QA | 15k | AI4Bharat Indic pairs + FLORES hi/ta/te/bn/mr + Bharat-Tiny SFT queries | open, filtered commercial-safe |
| Banking intent pairs | 5k | Banking77 same/different intent (reuse nirnay template) | CC-BY-4.0 |
| Legal statutes (train for C adapter, eval for A) | 10k | LexRAG DIFC/ADGM/GST snippets, synthetic query->statute (ground truth by construction) | ours Apache-2.0 |
| MS MARCO hard (for D-style retrieval hardness) | 10k | `sentence-transformers/msmarco-hard-negatives` subset (reuse flashrank `prepare_msmarco_data.py`) | open |

Hinglish generation: `Bharat-Tiny-LLM-v3` paraphrase en->Roman+Devanagari, script-aware dedupe via PolyWhisper `normalize_ortho.py`. Human spot-check 500. `SHA256SUMS` frozen before train (nirnay convention).

Prefixes (byte-exact, like base): `task: search result | query: {q}` / `title: {t} | text: {c}` / `task: classification | query:` for symmetry evals.

### 4.3 Schedule

- Loss: MultipleNegativesRankingLoss + MRL (768/512/256/128/32) + optional distillation from `Qwen3-Embedding-0.6B` teacher on 5k hard queries (copy flashrank KD insight: distill > CL).
- 3 epochs, batch 16, AdamW 2e-5, cosine, fp32 (base warns fp16 NaN; bf16 on MPS if supported else fp32), seed 7.
- Legal adapter: same base, second LoRA, 1 epoch on legal slice only (adapter swap story like PolyWhisper per-lang 14MB).
- Wall: ~3-5h M4 (flashrank did 12.5k steps Colab T4; this is 7.5k steps smaller encoder).

```bash
python src/embed/train_indic.py --base google/embeddinggemma-2 --text-only \
  --data data/frozen/triplets_40k.jsonl --epochs 3 --batch 16 --mrl 32,128,256,512,768 --seed 7 --device mps
python src/embed/train_legal_adapter.py --base checkpoints/best_indic --data data/frozen/legal_10k.jsonl --epochs 1
python scripts/eval_mteb_slice.py --ckpt checkpoints/best_indic --suite hi,en,code --truncate 32,128,768
```

### 4.4 Targets (pre-registered)

- MTEB-hi / Indic slice +0.03-0.05 over base text-only; no regression on MTEB-en (>=-0.01).
- 128d within 0.03 of 768d on text (base claims near-lossless to 256d; we extend to 32d for Qdrant nano story, report honestly).
- Legal adapter: statute recall@5 +0.10 over generic on held-out DIFC/GST queries.
- Latency: ONNX INT8 CPU <60ms/q (512 ctx), GGUF Q4 <80ms Mac, RAM <1GB text-only INT8.

---

## 5. Eval + honesty

- Suites: MTEB-hi slice, cross-lingual hi-en, Banking77 pair accuracy, legal statute recall, truncation sweep (768/512/256/128/32, re-normalized, queries+docs same dim), bf16 vs INT8 vs Q4_K_M delta.
- Publish `eval/*.json` raw + `assets/truncation.png` + `assets/benchmark_indic.png`. State omitted suites (vision/video/audio not evaluated - text-only fork).
- Limits: text-only (no image/video/audio), English+Hinglish+Hindi first (Tamil/Telugu weaker, measure per-lang like PolyWhisper v9 table), 8k ctx kept but trained at 512 (long-doc behavior unmeasured), sigmoid/cosine scores are ranking scores not calibrated probs (pair with Suraksha for decisions, flashrank for rerank).

---

## 6. Export + SEO/AEO/GEO/AIO

```bash
# ONNX INT8 (CPU serve, no torch)
python src/embed/export_onnx.py --ckpt checkpoints/best_indic --out onnx/ --int8
python scripts/verify_onnx_parity.py --torch checkpoints/best_indic --onnx onnx/  # cosine drift <1e-3
# GGUF (llama.cpp embedding, last-token/mean pooling flag documented)
python src/embed/export_gguf.py --ckpt checkpoints/best_indic --out gguf/f16.gguf
llama-quantize gguf/f16.gguf gguf/bharat-embed-Q4_K_M.gguf Q4_K_M
# LiteRT / WebGPU pointers: link litert-community pattern + Transformers.js Space (no custom TFLite compile in v1)
```

- HF metadata: `library_name: sentence-transformers`, `pipeline_tag: sentence-similarity`, `eval_results` (MTEB-hi, truncation), `datasets: eulogik/bharat-embed-triplets-40k`, `base_model: google/embeddinggemma-2`, `license: apache-2.0`.
- Card keyword block (bottom, like nirnay): `embeddinggemma2 hindi embedding, hinglish embedding, indic rag embedding, on-device embedding, onnx embedding, gguf embedding ollama, qdrant hindi search, mrl matryoshka embedding, uae law embedding, difc rag`.
- `AGENTS.md` + `USAGE.md` with byte-exact prefix + truncate+normalize snippet (agents copy-paste = citations).
- Collection `eulogik/bharat-embed` (base + GGUF + dataset + legal adapter + Spaces). Submit to MTEB discussion + `tardellirs/model-pulse`.
- Reranker cross-link: flashrank card links Bharat-Embed as retriever, Bharat-Embed links flashrank as reranker (mutual SEO).

---

## 7. Timeline + gates + risks

| Day | Milestone | Gate |
|---|---|---|
| 1 | Freeze triplets + hashes, baseline base text-only scores | SHA256SUMS |
| 2-3 | Train indic LoRA on M4, best ckpt | hi +0.03, en no-regress |
| 4 | ONNX/GGUF/Ollama + Spaces live | parity green, demo live |
| 5 | Card + dataset + collection + blog `I forked EmbeddingGemma-2 on a Mac Mini` | push HF |

Gates: no fp16 (bf16/fp32 only per base warning), queries+docs same dim enforced in code, vision/audio code paths assert-disabled in text-only serve, license file Apache-2.0 present.

Risks: Google ships official v2 edge pack (mitigate: Indic+legal differentiation, not pure pack); Hindi gains <0.02 (ship edge-pack + legal, publish negative Indic honestly, like nirnay hard-tier); 32d collapses on multimodal (state text-only scope, keep 128d recommended, 32d experimental).

---

## Appendix: reuse map

- Base: https://huggingface.co/google/embeddinggemma-2
- Patterns: https://huggingface.co/litert-community/embeddinggemma-300m (LiteRT table), https://huggingface.co/jinaai/jina-embeddings-v5-text-nano-retrieval-GGUF (GGUF + MRL + task-targeted), https://huggingface.co/Xenova/all-MiniLM-L6-v2 (ONNX/Web distribution)
- Eulogik: https://github.com/eulogik/flashrank-pro + https://huggingface.co/eulogik/flashrank-pro-base (retriever->reranker stack), https://github.com/eulogik/LexRAG (legal data), PolyWhisper `normalize_ortho.py` (Indic norm), `eulogik/Bharat-Tiny-LLM-v3` (Hinglish synth)
