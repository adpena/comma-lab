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

## 2026-05-08 Adversarial Review Addendum

Scope: correctness, byte accounting, side-info accounting, decode roundtrip,
and stack semantics for `src/tac/codec/a6_selfcomp_blockfp_hyperprior_compose.py`.

Findings fixed in the follow-up hardening patch:

- `uint8` scale side-info now uses the top code (`255`) for the `int8(-128)`
  endpoint, representing `127.5` instead of clipping to `127.0`; ledgers report
  `scale_quantization_saturated_blocks`,
  `scale_quantization_max_abs_error`, and `scale_side_info_exact`.
- Empty streams now run the same decode-of-encode verification path as non-empty
  streams when `verify_roundtrip=True`.
- Non-integer symbol arrays are rejected instead of silently truncating floats
  into int8 symbols.
- Wire emitters reject `block_size > 65535` with an explicit A6BF uint16-header
  error instead of relying on a late `struct.pack` failure.
- `tools/phase_a_pareto_summary.py` now points at this tracked ledger, not a
  stale `feedback_pr101_a6_...` path.

These are guard and accounting fixes only. They do not change A6's evidence
grade, do not promote a candidate, and do not create a score claim.

## Ranked Reactivation Criteria

1. Test the A6 compose on the PR106/HNeRV substrate, where monolithic
   heteroscedastic streams are a closer match than PR101's near-iid int8
   stream.
2. Add a learned hyper-decoder or tensor-aware PMF map instead of the fixed
   linear `sigma = sigma_floor + alpha * scale` rule.
3. Use cross-tensor grouping so side-info and PMF context are amortized across
   the payload actually being coded.
4. Implement true Selfcomp per-channel block-FP invariants from
  `src/tac/block_fp_codec.py`, not max-abs scales over an already quantized
  stream.
5. Compose after lossy coarsening only if the old/new charged-byte boundary and
   runtime decoder path are explicit.

Any reactivation still requires a byte-closed runtime packet, proof that changed
bytes are consumed by `inflate.sh`, local inflate parity, and paired
`[contest-CUDA]` / `[contest-CPU]` evaluation only after strict packet custody.
