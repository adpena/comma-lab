---
name: QAT from epoch 0 is WORSE than float+post-hoc — Quantizr trains float
description: QAT costs 0.34 proxy for 0.044 rate savings. ASYM wins 8x. Quantizr trains float then quantizes post-hoc. Don't fight quantization noise during training.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## The QAT Mistake (2026-04-23)

QAT from epoch 0 proxy: 0.665
Float (no QAT) proxy: 0.326 (previous run, same architecture)
Gap: 0.34 proxy points

FP4 saves: 130KB ASYM → 75KB FP4 = 55KB saved = 0.037 rate
QAT costs: 0.34 distortion (at minimum)
**ASYM wins by 9x.**

## What Quantizr Actually Does
1. Trains FLOAT with diff_round (eval roundtrip STE)
2. Exports to FP4 with custom codebook + Brotli POST-HOC
3. His model is 88K params — small enough that post-hoc FP4 degradation is minimal

## CONFIRMED BY HEAD-TO-HEAD (2026-04-23)

Same architecture (103K, DSConv, CRF50 masks), same 4090, same settings.
Only difference: --qat vs no --qat.

| Run | Epoch 225 best | Epoch 1950 best |
|-----|---------------|-----------------|
| QAT from epoch 0 | 1.056 | 0.644 |
| Float (no QAT) | **0.644** | (still training) |

Float at epoch 225 already MATCHES QAT at epoch 1950.
Float converges 8.7x faster. QAT wastes gradient capacity fighting noise.

## How to Apply — NON-NEGOTIABLE
- ALWAYS train FLOAT with CRF50 masks + eval_roundtrip (no --qat)
- Export ASYM for best quality
- Post-hoc FP4 only if rate savings > distortion cost (test both)
- NEVER use --qat from epoch 0 for models under 200K params
- Quantizr trains float then quantizes. Do the same.
