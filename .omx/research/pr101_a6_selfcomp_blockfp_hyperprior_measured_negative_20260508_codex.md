---
title: PR101 A6 Selfcomp Block-FP x Hyperprior Proxy Measured Negative
date: 2026-05-08
author: Codex
score_claim: false
evidence_grade: byte-roundtrip proxy
ready_for_exact_eval_dispatch: false
---

# PR101 A6 Selfcomp Block-FP x Hyperprior Proxy Measured Negative

## Result

Command:

```bash
.venv/bin/python tools/pr101_a6_blockfp_hyperprior_anchor.py \
  --output-dir experiments/results/pr101_a6_blockfp_hyperprior_codex_20260508Tlocal \
  --no-evidence-append
```

Best measured proxy cell:

| Cell | Estimated archive bytes | Delta vs PR101 brotli |
|---|---:|---:|
| block_size=64, scale_quant=uint8 | 214,035 B | +35,891 B |

This is not a score claim. No byte-closed `archive.zip -> inflate.sh` packet
consumes A6 bytes, and no exact CUDA auth eval has run for this lane.

## Classification

Status: `measured-config negative`.

Scope: current max-abs-scale conditional Gaussian range-coder over the PR101
int8 stream. This does not kill real Selfcomp per-channel block-FP, learned
ChARM, tensor-aware PMFs, joint AC over side-info, PR106-specific substrates,
or byte-map-preserving arithmetic rewrites.

The proxy beats two weak internal baselines (`blockfp_only` and
`hyperprior_only_global_sigma`) but loses badly to the relevant baseline:
PR101 split-brotli at 178,144 B.

## Fixes Landed With The Classification

- A6BF v1 now rejects non-default `alpha` and `sigma_floor` until a future wire
  version serializes them.
- Decode now rejects trailing bytes, tampered `block_size=0`, and malformed
  empty-stream chunk sections.
- Scale side-info is little-endian and handles `int8(-128)` without overflow.
- Baseline helpers now return actual byte strings whose lengths equal their
  ledgers.
- The A6 evidence row keeps `score_affecting_payload_changed=false` and
  `charged_bits_changed=false` until a runtime-consumed packet exists.

## Reactivation Criteria

- Implement true Selfcomp per-channel block-FP invariants from
  `src/tac/block_fp_codec.py`, not max-abs scales over an already quantized
  stream.
- Use tensor-aware or learned PMFs, not a fixed linear sigma map.
- Build a byte-closed runtime packet and prove changed bytes are consumed by
  inflate.
- Re-run paired `[contest-CUDA]` and `[contest-CPU]` eval only after strict
  packet custody.
