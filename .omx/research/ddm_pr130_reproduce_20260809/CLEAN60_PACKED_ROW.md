# clean60: the A1+A2 axis beaten by −1,871 PACKED bytes, model half EXACT

First conversion of the HPAC training work from trainer ESTIMATES into REAL packed bytes with a
bit-exact round-trip proof. Scorer-free, local Metal + CPU. `score_claim=false`.

## 1. The run

`clean60` (pid 34461, 2,432 s, rc=0): 60 epochs from the pinned init
`hpac_p64_exact_from_archive.pt`, original `T_max=60` cosine, `qat_fraction 0.5`, seed 20260716,
`--device mps`. NO resume, NO optimizer-moment reset. Report:
`/Volumes/VertigoDataTier/pact/ddm_hb1_20260806/reports/gt_hpac_clean60.json`.

`best_epoch 52` · bpp 0.0077998 · top1_error 0.0019275.

### It beat both resume arms — the moment-reset diagnosis held

| arm | joint (trainer est.) | vs PR130 137,159 |
|---|---:|---:|
| v1 naive resume (full-LR restart) | ASCENDED (137,210 by ep2) | worse |
| v2 LR-matched resume (moments still reset) | 135,289 | −1,870 |
| **clean60 from scratch** | **135,165** | **−1,994** |

### The QAT phase OSCILLATES — correcting my own earlier claim

15 QAT evals ep32..60: min 135,165 · max 137,361 · last 135,578 · **spread 2,196 B**.
Tail: ep50 135,435 · ep52 **135,165** · ep54 135,208 · ep56 135,319 · ep58 136,126 · ep60 135,578.

I earlier wrote that the original run "died at the steepest part of its descent with ~28 QAT epochs
unrun." **That overstated it.** ep32 was a local dip in an oscillating phase, not a point on a
monotone descent. What IS confirmed: clean60 reproduced ep32 = 135,828 EXACTLY, so the trajectory
is deterministic under fixed seed+config. And `best`-tracking was load-bearing — the final epoch
(60) is 413 B WORSE than the banked best (52).

## 2. The PACKED row (the honest one)

`pack_hpac_self_compress.py` on the clean60 best checkpoint, which carries a fail-closed bit-exact
logit round-trip (`if max_diff != 0.0: raise`):

```
raw_model_bytes 20,157 · compressed_model_bytes 15,188 · metadata_bytes 259
verified_exact: true · max_logit_diff: 0.0
```

Against PR130's measured leave-one-out PACKED marginals (`.omx/state/main_hot_state.md`):

| section | PR130 packed | ours | Δ |
|---|---:|---:|---:|
| HPAC model | 15,092 | **15,188** (EXACT) | **+96** |
| tokens | 116,980 | 115,013 (coder rate estimate) | **−1,967** |
| **A1+A2** | **132,072** | **130,201** | **−1,871** |

**ΔS = 25 · (−1,871) / 37,545,489 = −0.0012457.**

**The trainer estimate said −1,994; the packed truth is −1,871 — a 123 B (6.2%) overstatement.**
The rr1 audit's "estimates are not archive bytes" warning is now quantified.

## 3. First data point on d(tokens)/d(model) — SCOPED

rr1 rank-4 recorded the model↔token exchange derivative as **ABSENT**. This row gives a first
secant: **+96 model bytes bought −1,967 token bytes ⇒ ≈ −20.5 token B per model B**, i.e. ~20×
past the −1 break-even where growing the prior becomes a net win.

**SCOPE, binding:** this is a two-point secant between DIFFERENTLY-TRAINED models, NOT a capacity
ladder. Both arms ran `--channels 64 --patch 64 --delta 2 --frame-dim 8`; the 96 B difference is a
training artifact, not a deliberate capacity change. It is SUGGESTIVE that the exchange is steeply
favorable. It is **NOT** the derivative. The controlled ladder (rr1 fire-queue rank 4) remains owed
and this makes it look more valuable, not less.

## 4. What is NOT yet measured

- **The token half is still the arithmetic coder's own rate estimate** (`bpp × 117,964,800 / 8`),
  not a byte count from an emitted stream. Arithmetic-coder rate estimates are tight but not exact.
  A real token encode is owed before this is an archive row.
- **No archive was built.** This is an A1+A2 section comparison, not a packed `archive.zip`, and not
  an `evaluate.py` row. B (semantic renderer) and C (pose carrier) are PR130's, untouched.
- **d_seg/d_pose UNCHANGED by construction** — HPAC is a lossless coder over the exact GT-argmax
  tokens, so this is a pure rate move on the zero-coupling axis. Not independently re-verified here
  (would need CUDA for the receiver, `inflate.py:665`).
- No score claim. `score_claim=false`.
