# eval - pre-registered gates

Do not move goalposts after training. These gates were set before run one.

## Gates

- MTEB-hi and Indic slice: plus 0.03 to 0.05 over base text only.
- MTEB-en: no regress worse than minus 0.01.
- 128d within 0.03 of 768d on text. 32d is lab only, report with warning.
  - Measured 2026-10-09: 128d missed (0.6294 vs 0.7324, delta -0.103). 256d delta -0.030, 512d delta -0.008. Default recommendation moved to 256d. Gate recorded as missed.
- Legal adapter: statute recall at 5 plus 0.10 over generic on held out DIFC and GST queries.
- Latency: ONNX INT8 CPU under 60ms per query at 512 ctx. GGUF Q4 under 80ms on Mac. RAM under 1GB text only INT8.
- Parity: torch vs ONNX mean cosine drift target under 1e-3. If INT8 lands near 1e-2, publish table as is.

## Suites

- MTEB-hi slice, hi-en cross lingual, Banking77 pair accuracy, legal statute recall.
- Truncation sweep: 768, 512, 256, 128, plus 32 lab. Re-normalized. Queries and docs same dim.
- Quant delta: bf16 vs INT8 vs Q4_K_M.

## Honesty

- Publish eval JSON raw plus truncation chart.
- Omitted: vision, video, audio. Text only fork.
- Tamil and Telugu likely weaker than Hindi and Hinglish. Report per lang.
- Long doc 8K kept in arch but trained at 256 (512 planned, 16GB box swaps above 256). Mark long doc as unmeasured.
