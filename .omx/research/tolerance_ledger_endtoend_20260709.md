# END-TO-END TOLERANCE LEDGER — the dual-waterfill (P3, SHARED dual-chain owed-gate) — 2026-07-09

**Task #393 D2. `$0` · no GPU · no launch. `[macOS-MLX/CPU advisory · NON-PROMOTABLE]` — a ledger moves
no pointer. Pointer contest-CPU 0.19110 UNMOVED — MEANS.**

**Why this file exists.** BOTH twelve-philosophy passes (`philosophy_pass_v752_20260709.md` §P3 →
F-P3-1; `philosophy_pass_v8_20260709.md` §P3 rank-2) render the SAME finding: the pipeline
`fit → quantize → R → uint8 → scorer-parity` has only PIECEWISE margins, **never an end-to-end ledger**
that (a) sums per-stage numerical tolerances and (b) allocates the d_seg distortion budget per stage.
Both memos route it as a launch-gate row and BOTH explicitly say it is **owed on BOTH chains** — so it
is one SHARED gate, not two. This ledger collects every EXISTING MEASURED stage tolerance into ONE
table so the #385 which-to-run brief carries a single P3 column. **DERIVED-NOT-INVENTED: every number
cites its artifact; every UNMEASURED cell is marked owed.**

**STORES CONSULTED:** `feedback_mlx_pytorch_render_parity_crux_landed_20260601.md` (+ anchor
`pr95_hnerv_mlx_pytorch_render_parity_crux_anchor.json`) · `src/tac/inc1a_harness/decoupling_screen.py`
(L43 δ_mask, L48 δ_R RETIRED) · `src/tac/boundary_math/{aa_sdf_observation_render.py:50,
analytic_lane_render_band.py:76, island_protection.py:62}` (parity ≥0.9997) · CLAUDE.md
deterministic-reproducibility principle (numpy-fp32 = bit-identical authority; byte-close bit-exact
decode) · memory L70 (`--fused-r-kernel` cross-process determinism) · `philosophy_pass_{v752,v8}_20260709.md`.

---

## 0. THE DUAL WATERFILL (what "budget" means here)

Two orthogonal budgets share the same 5 stages:
- **Waterfill (1) — NUMERICAL tolerance:** the fp / quant / rounding error each stage introduces vs the
  numpy-fp32 authority. Almost all of it is SUB-ARGMAX (below the SegNet argmax flip threshold) → ~ZERO
  d_seg. This waterfill is **MOSTLY MEASURED** (table §1).
- **Waterfill (2) — DISTORTION budget:** the d_seg each stage is ALLOWED to spend (fit-residual +
  quantize-δ + R-downsample-δ + uint8-STE-δ + scorer-parity-δ), summing to a target. This waterfill is
  **the OWED end-to-end ledger** — the cells that matter (quantize-realized, R-realized-vs-direct,
  composed fit→scorer) are UNMEASURED-until-byte-close. **That is the P3 gate.**

The measurement RESOLUTION FLOOR (below which any per-stage Δ is un-trustworthy) is δ_mask = **3.46e-6**
(n600, R7-MEASURED; `decoupling_screen.py:43 DELTA_MASK_FRAME_SAMPLING_FLOOR`). Any waterfill-(2)
allocation smaller than this is below the floor and non-resolvable.

---

## 1. THE STAGE TABLE (measured numerical tolerance + d_seg allocation, per stage)

| # | stage | numerical tolerance (MEASURED) | artifact | d_seg impact | waterfill-(2) allocation |
|---|---|---|---|---|---|
| 1 | **fit** (MLX decoder forward vs numpy-fp32 authority) | mean-abs **1.18e-3**, max **1 LSB**, **46 / 1,179,648 px = 0.0039%** differ (at rounding boundaries) | render-parity memo (MLX-opt vs PyTorch-fp32 row) + anchor `e976acd5…` (600 pairs) | **ZERO** ("sub-quantization for the SegNet argmax", memo §"EXACTLY ZERO impact on d_seg") | 0 (MEASURED sub-argmax) |
| 2 | **quantize** (fp16 / quantized `code` round-trip) | **BIT-EXACT** at byte-close (deterministic decode → bit-identical inflate every run/host) | CLAUDE.md deterministic-repro (bit-exact decode) | bit-exact → 0 for the SHIPPED code; **realized d_seg of any code TRUNCATION is OWED** | **OWED** (D18 k90-truncate byte-close A/B: real Δbytes vs Δd_seg — needs the FINAL ckpt, which does not exist yet) |
| 3 | **R** (bicubic↑384→874 → uint8 → bilinear↓512×384) | R OP itself **BIT-EXACT** cross-process under `--fused-r-kernel` (fixed-order VJP); the through-R distortion PROXY δ_R = **0.019590** is **RETIRED** | `decoupling_screen.py:48 DELTA_R_PROXY_RETIRED` (comment: never a default, ~5600× category error); memory L70 (fused-R determinism) | δ_R was a **d_seg PROXY, RETIRED** — NOT a live budget rung; the R op is bit-exact | **the LARGE owed cell**: direct-partition d_seg ≠ realized-through-R d_seg (the F-P9-1 basis gap = **170–350×** off; auditor-A C1). Realized-through-R d_seg is the authority; direct-partition is proxy |
| 4 | **uint8** (float→uint8 frame buffer) | **1 LSB** at x.5 rounding boundaries (= the SAME event as stage-1 fit residual; the two do not independently sum) | render-parity memo §"irreducible float→uint8 rounding-boundary floor" | **ZERO** (sub-argmax) | 0 (MEASURED; = stage-1 residual, not additive) |
| 5 | **scorer-parity** (MLX/torch argmax vs numpy-fp32 authority) | **parity ≥ 0.9997** (≤ 3e-4 argmax-disagreement fraction, backend portability) | `aa_sdf_observation_render.py:50` · `analytic_lane_render_band.py:76` · `island_protection.py:62` · MLX portability contract; numpy-fp32 = bit-identical verdict authority (CLAUDE.md) | portability bound only — the d_seg VERDICT is computed on numpy-fp32/CPU-torch authority, NEVER on a ≥0.9997 backend | ≤ 3e-4 (portability tolerance, NOT a spent d_seg — authority is numpy-fp32) |

---

## 2. COMPOSITION — the sum, and the owed remainder

**Waterfill (1) numerical, MEASURED composition:** stages 1+4 are the SAME uint8-rounding-boundary event
(46 px / 0.0039%, sub-argmax) → **compose to ZERO d_seg**; stage 3's R-op is bit-exact under fused-R;
stage 5 bounds backend transfer at ≤ 3e-4 argmax-disagreement (and is not spent because the verdict runs
on the numpy-fp32 authority). **Measured numerical sum ⇒ ~0 d_seg** (all sub-argmax / bit-exact / on-
authority). This is the good news: the numerical spine is clean.

**Waterfill (2) distortion, the OWED remainder (= the P3 gate):**
- **quantize-realized-δ** (stage 2): the realized d_seg of code truncation — OWED (D18 byte-close A/B).
- **R-realized-vs-direct-δ** (stage 3): the DOMINANT unmeasured cell — direct-partition d_seg is **170–350×**
  off realized-through-R (auditor-A C1; the F-P9-1 basis −48% is a direct-partition proxy). Until the
  basis-reality n600 A/B (owed-16, v7.5.2 addendum §A) runs clean through the CORRECTED R, this cell is a
  proxy, not a budget.
- **composed fit→scorer d_seg tolerance**: no end-to-end sum exists — it requires a byte-closed
  `upstream/evaluate.py` n600 row, which neither chain has produced.

**Verdict:** the numerical waterfill is MEASURED-and-clean (~0 d_seg); the distortion waterfill is OWED
at byte-close on the FINAL ckpt. This is exactly why P3 is a **launch-gate, not doc-closable** — a doc
cannot manufacture the byte-closed row. The gate is: at each chain's byte-close, fill stages 2 + 3 +
composed with realized-through-R d_seg allocations and confirm the sum ≤ the term's target.

---

## 3. ROUTED (→ #385 brief, ONE shared P3 column)

| chain | P3 status | owed at |
|---|---|---|
| v7.5.2 | numerical spine MEASURED-clean; distortion cells OWED (esp. R-realized-vs-direct = owed-16 basis) | v752 byte-close (FINAL ckpt) |
| v8 | identical gate (`philosophy_pass_v8` §P3 rank-2); de-share `dilate=2` realized-cover feeds stage-2/3 allocation | SPEC_v8.1 / v8 byte-close |

**SHARED — carry as ONE P3 row on the #385 which-to-run brief, not two** (both memos' explicit
dual-chain note). Sister owed-gates on both chains: clause-A archive dedup / pairwise-non-derivability ·
P12 composition-sign matrix.

**Pointer 0.19110 UNMOVED.** This ledger renders NO research kill — it is a collection of EXISTING
measured tolerances with the owed cells named; narrowest-supported OBSERVATION.
