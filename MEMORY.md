# MEMORY.md - Bharat-Embed Family (living doc)

> Living memory for Bharat-Embed work. Update on every real finding.
> Source plan: EMBEDDINGGEMMA2_VARIANT_PLAN.md (2026-10-07, Owner: Eulogik, HW: Mac mini M4 10c 16GB, License: Apache-2.0).
> Last updated: 2026-10-07 - second validation pass with HF API numbers.

## 1. What this project is (locked)

- Fork google/embeddinggemma-2 (740M multimodal) as text-only 270M family.
- Hero: A - Indic and Hinglish retrieval LoRA (Roman plus Devanagari code switch), with D merged in (retrieval only plus MRL training, Apache-2.0 alt to Jina nano for Indic).
- Companion day one: B - zero train edge pack (ONNX INT8 plus GGUF plus LiteRT pointer plus WebGPU Space). This is the download hack.
- Second adapter: C - legal adapter (UAE and India DIFC ADGM GST VAT, EN AR HI), plus 2 days, same base, adapter only repo.
- Not doing now: E (VisDoc 440M), F (audio 300M), G (guard and tool call). Push to v1.1 after traction.
- Stack story: retrieve 100 with Bharat-Embed 128d -> rerank with eulogik/flashrank-pro-base (149M, 0.3314 five-set NDCG, Apache-2.0, card verified 2026-10-07, note it is BEIR style five-set vs run-init, not official BEIR) -> generate with Bharat-Tiny-LLM-v3 (still unverified).

## 2. Verified base facts (HF API, fetched 2026-10-07)

Checked via https://huggingface.co/api/models/google/embeddinggemma-2 plus model card fetch.

- ID google/embeddinggemma-2, license apache-2.0, arch embedding_gemma2.
- Params BF16 744371512. Matches 740M claim. Split from card: 270M text (130M transformer plus 140M embedder) plus 170M vision plus 300M audio. Verified.
- Unified 768d, MRL 128 plus 256 plus 512 plus 768 only, 8K ctx, 100 plus langs (train data 140 plus), task prefixes, selective loading, bf16 or fp32 (never fp16), sentence-transformers compatible. Verified.
- Selective loading exact: text only {"vision_config": None, "audio_config": None} -> 270M. Text plus image 440M. Text plus audio 570M. Full 740M. Verified.
- Prefix format exact: task: search result | query: {q} and title: {t} | text: {c} and title: none fallback. prompt_name SearchQuery, Document, Classification and rest. Verified.
- Truncation rules: must re-normalize after slice. Queries and docs must share dim. Use truncate_dim plus normalize_embeddings True. Verified.
- Quality vs dim from base table: 768d 61.36 -> 512d 61.17 -> 256d 60.41 -> 128d 57.89 (MTEB multilingual v2 mean). Base says near lossless to 256d. 128d is text only, weak on multimodal (MMEB overall 59.01 -> 45.65). Verified.
- Likes 718 at second fetch (was 713, plan said 669 in 16h, growth looks normal). Downloads 364 (model is days old, low count is expected). Verified.

## 3. Download table - rechecked with API

All numbers below are HF API downloads field, fetched 2026-10-07.

| Model | API downloads | Status | Note |
|---|---|---|---|
| BAAI/bge-small-en-v1.5 | 62278766 | verified | Plan said 62.3M. Holds. MIT. ONNX included. |
| sentence-transformers/all-MiniLM-L6-v2 | 234015516 | verified but sort flag | Plan said 234M. Holds. But pipeline_tag is sentence-similarity, not feature-extraction. Keep footnote. Params 22713216. Holds 22.7M. |
| Xenova/all-MiniLM-L6-v2 | 3869766 | verified | Plan said 3.87M. Exact. |
| Qwen/Qwen3-Embedding-0.6B | 9110331 | verified | Plan said 9.25M. Close. Apache-2.0. MRL 32-1024. |
| intfloat/multilingual-e5-large | 7160371 | verified | Plan said 7.16M. Exact. MIT. |
| BAAI/bge-reranker-large | 2357413 | verified | Plan said 2.36M. Holds. MIT. |
| jinaai/jina-embeddings-v5-text-nano-retrieval-GGUF | 6556 | verified | Plan said 6.5k. Holds. License cc-by-nc-4.0. GGUF total in API 211766016, card says 239M. Small gap, note it. MRL 32-768 and last token pooling from card. NC tag means Apache alt angle is real. |
| litert-community/embeddinggemma-300m | 1393 | verified but version flag | Plan said 1.4k. Holds. But license gemma, created 2025-09-03. This is v1, not v2. No v2 LiteRT pack yet. Do not cite v1 as v2 traction. |
| Qwen/Qwen3-VL-Embedding-2B | 1142619 | verified | Plan said 1.1-1.5M. Holds. Apache-2.0. |
| google/embeddinggemma-2 | 364 | verified | New model. Low count is normal. |
| tardellirs/model-pulse, webml-community/embeddinggemma-2-webgpu | n/a | verified exist | Both in base Spaces list. Good for SEO and AEO. |

Prior gap closed: Xenova, e5, reranker, Qwen VL are now verified. No more unverified rows in this table.

## 4. Fixes and risks

1. MRL-32 is not native. Base supports 128, 256, 512, 768 only. Plan trains mrl 32,128,256,512,768. Jina and Qwen do support 32d, so it is trainable, but it is extra work. 128d already drops 61.36 -> 57.89. Keep 32d as experimental. Ship 128d as default. Update plan 4.3 and 4.4.
2. LiteRT v1 vs v2 mixup. v1 pack is gemma licensed 300M. v2 Apache pack does not exist yet. That is the opening. Keep plan line of no custom TFLite compile in v1, just pointer plus Space.
3. GGUF path is now proven (correction to prior risk). Fresh HF search shows ggml-org/embeddinggemma-2-GGUF official from 2026-10-06, arch gemma-embedding2, Q8_0 310MB, BF16 558MB, license apache-2.0. Plus unsloth/embeddinggemma-2-GGUF same day. So no day one blocker. Still need pooling parity check since base is mean pooling and Jina is last token. Gate on llama-embedding pooling mean test.
4. ONNX INT8 drift under 1e-3 is hopeful. INT8 drifts more in practice. Keep gate but plan to relax to 1e-2 or ship per quant table like Jina pattern.
5. Data licenses. FLORES-200 is CC-BY-SA 4.0 (checked facebookresearch/flores GitHub, Licenses section). Not clean for Apache-2.0 redistribution. Banking77 CC-BY-4.0 needs attribution. MS MARCO subset needs exact license pin. Freeze SHA256SUMS plus per slice license file before train.
6. Synthetic legal data. Ground truth by construction is too strong. Needs held out human check. Plan has 500 spot checks for Hinglish, extend same to legal.
7. Targets are bold. hi plus 0.03-0.05 on 40k LoRA triplets is possible but not sure. en no regress over minus 0.01 is the hard gate. Publish misses plainly.
8. M4 16GB fit looks ok on paper but not run yet. 270M bf16 is about 0.54GB plus LoRA r16 (about 8M trainable). MPS bf16 has no clear guarantee in PyTorch MPS docs I fetched (page shows mps flow, no dtype matrix). So keep fp32 fallback. Run selective load smoke test plus batch finder with grad accum.
9. Eulogik reuse partly done. flashrank-pro-base verified live 2026-10-07: API shows 149605633 params, Apache-2.0, ModernBERT base, downloads 22. Card shows 0.3314 five-set avg vs 0.2703 run-init (plus 22.6 percent). Card notes BEIR style only, not official BEIR, touche skipped for RAM. Cite with that footnote. Still unverified: Tiny-LLM, PolyWhisper norm script, LexRAG, TinyDoc-VLM, Signizy. Do not cross link those until cards exist.
10. 8K ctx kept but trained at 512. Plan already flags long doc as unmeasured. Keep that note. Do not claim 8K RAG without eval.

## 5. Why the plan still works

- Laws 1-7 hold: tiny beats large on pulls, distribution drives downloads, task targeted beats generic, Apache beats NC, embed plus reranker stack wins, MRL cost story lands, prefix docs get cited. All backed by API rows above.
- A plus B plus C is the right mix. Pure Indic alone would miss onnx gguf litert search flow and Jina NC steal angle. B pulls downloads, A gives edge, C gives paid use.
- 270M text only (not 740M) is the only sane M4 target. Full 740M plus 8K will OOM. Selective load to 270M is officially supported.

## 6. Locked spec snapshot (pending fixes above)

- Repos: eulogik/bharat-embed-270m-gemma2, GGUF split, triplets-40k dataset, bharat-legal-embed-270m (adapter only, 14MB pattern) plus Spaces bharat-embed-search and bharat-embed-webgpu.
- Quickstart: SentenceTransformer with config_kwargs vision None audio None, model_kwargs torch_dtype bfloat16, prompt_name SearchQuery, truncate_dim 128 plus normalize True. ONNX via onnxruntime plus qdrant-client. GGUF via ollama run hf.co slash GGUF tag or llama-embedding with pooling mean.
- Train: MNRLoss plus MRL (see risk 1), 3 epochs, batch 16, AdamW 2e-5 cosine, fp32 or bf16 only, seed 7, about 3-5h on M4 (still estimate, needs run). Legal adapter 1 epoch on legal slice.
- Eval: MTEB-hi slice, hi-en cross lingual, Banking77 pairs, legal recall at 5, truncation sweep 768 down to 32 (re-normed, same dim both sides), bf16 vs INT8 vs Q4 delta. Ship eval json plus truncation chart. Mark vision video audio as not evaled.
- Card SEO: library sentence-transformers, pipeline sentence-similarity, base_model google/embeddinggemma-2, license apache-2.0, keyword block, AGENTS.md plus USAGE.md with byte exact snippets, collection plus MTEB thread plus model-pulse submit, flashrank cross link once cards are live.

## 7. Open work

- [x] Re-pull download table under one pass with timestamp. Done above. MiniLM footnote still needed in plan card.
- [x] Confirm llama.cpp and GGUF path for v2. Done. Official ggml-org GGUF live 2026-10-06.
- [ ] MPS smoke test: selective load 270M plus bf16 vs fp32 plus batch finder with accum steps for eff batch 16. Local check 2026-10-07: python 3.14.7, torch 2.12.1, mps built True, mps avail True, no ml libs installed yet. Needs pip install plus 3GB model pull before real test.
- [ ] Freeze triplet slice licenses (FLORES SA handling first) plus SHA256SUMS before train.
- [ ] Lock MRL list: keep 32d experimental or drop to 128d min for v1.
- [x] Verify flashrank card. Done 2026-10-07, see section 4 item 9. Still open: Tiny-LLM, LexRAG and rest.
- [x] Scaffold repo to shippable shape. Done 2026-10-07: prefixes plus 7 unit tests pass, train_indic plus legal adapter with dry run, onnx plus gguf export, triplet prep with license policy, eval slice plus parity plus encode scripts, eval README gates, USAGE, frozen README, 3 card drafts. All dry runs pass. Style check clean.
- [x] Pre-register eval gates as eval README. Done 2026-10-07. Gates locked before run one.

## 8. Update log

- 2026-10-07: first pass. Base card plus 4 rows checked. Flagged MRL-32, LiteRT v1 vs v2, sort mix, license, GGUF and ONNX risks.
- 2026-10-07: second pass. HF API recheck for all 10 download rows. Closed Xenova, e5, reranker, Qwen VL gaps. Proved FLORES CC-BY-SA via GitHub. Corrected GGUF risk, official v2 GGUF is live. Cleaned file to match AGENTS.md style (no em dashes, plain tone).
- 2026-10-07: third pass. Verified flashrank-pro-base card (149M, 0.3314 five-set, Apache-2.0, BEIR style footnote). Local env check (torch 2.12.1, mps True).
- 2026-10-07: scaffold ship. Prefix helpers plus 7 tests green. All 8 dry runs pass. Eval gates locked. Cards plus USAGE plus frozen policy written. No em dash in new files.
