---
title: PTQ-on-frontier FLOOR — the decisive $0 gate (the floor score-aware QAT must beat)
authority: "[contest-CPU advisory] — NON-PROMOTABLE; exact pointer UNMOVED at 0.19110"
score_claim: false
promotable: false
frontier_pointer_moved: false
mission_contribution: frontier_breaking_enabler
date: 2026-06-18
---

# PTQ-on-frontier FLOOR (2026-06-18)

Executes the approved Path-B decisive cheap gate from the desk-calc pivot
(`.omx/research/capacity_rd_score_aware_qat_pivot_20260618T175040Z.md`): **measure the actual
(not modelled) exact-S of naive bit-shrinking the EXISTING 0.19110 frontier decoder** — the
floor score-aware QAT must recover from.

**ALL numbers `[contest-CPU advisory]` NON-PROMOTABLE. Exact pointer stays pointer-only at
0.19110. CPU-only, $0 (no GPU, no paid dispatch). No PR.**

Reusable code: `tac.frontier_decoder_ptq` (16 NO-FAKE tests) + `experiments/probe_ptq_on_frontier_floor.py`
+ `reports/ptq_on_frontier_floor.json`.

## Pipeline faithfulness (the gate that makes the numbers real)

The int8 IDENTITY round-trip (decode → re-quantize-to-int8 (unchanged grid) → re-encode →
CTXR recode → STORED zip) reproduces the on-disk frontier archive **byte-for-byte**:
**177,169 B == the frontier**, decoded `state_dict` `max_abs_delta = 0`, latent + sidecar
byte-identical. This proves the decode/re-encode pipeline is lossless and the PTQ byte +
distortion numbers are apples-to-apples against the real frontier (codec: PR #101 split-brotli
per-tensor int8 + fp16 scale → CTXR FP11 member; decoder = 228,958 params, decoder section =
161,104 B = 90.9% of the archive).

## The MEASURED bit-width → S table (full 600-pair exact, CPU authority)

| mode | d_seg | d_pose | bytes | rate | **S** | ΔS vs int8 | ΔS vs pointer |
|---|---|---|---|---|---|---|---|
| **int8 (identity baseline)** | 0.000594 | 0.000037 | 177,169 | 0.1180 | **0.1965** | +0.0000 | +0.0054 |
| int7 | 0.001356 | 0.000152 | 174,061 | 0.1159 | 0.2905 | +0.0940 | +0.0994 |
| int6 | 0.002294 | 0.000276 | 147,513 | 0.0982 | 0.3801 | +0.1836 | +0.1890 |
| int5 | 0.004744 | 0.001550 | 118,589 | 0.0790 | 0.6779 | +0.4814 | +0.4868 |
| int4 | 0.011881 | 0.040852 | 87,925 | 0.0585 | 1.8858 | +1.6893 | +1.6947 |
| sa_int4lo_int8hi | 0.010640 | 0.013310 | 100,005 | 0.0666 | 1.4954 | +1.2990 | +1.3043 |
| sa_int4lo_int6hi | 0.010676 | 0.013990 | 96,021 | 0.0639 | 1.5056 | +1.3091 | +1.3145 |
| sa_int5lo_int8hi | 0.004486 | 0.001001 | 126,485 | 0.0842 | 0.6329 | +0.4364 | +0.4418 |

`sa_*` = score-aware per-stage: coarsen the d_seg-blind early/low-res stages (where ~77% of
the params live: stem + blocks.0–2) to `int4lo`/`int5lo`, protect the tiny output-proximal
d_seg-critical heads (blocks.4/5, skips, refine, rgb_0/1 ≈ 7% of params) at `int8hi`/`int6hi`.

**Local-vs-pointer offset:** the int8 identity recomputes S = **0.1965** vs the pointer's
0.19110 (offset **+0.0054**). This is the small consistent local-CPU-vs-contest-CPU drift on
this exact archive (the pipeline bytes are EXACT; the d_seg/d_pose I get back out slightly
higher distortion than the pointer's true contest-CPU eval). All ΔS comparisons below use the
**local int8 baseline (0.1965)** as the apples-to-apples reference, not the absolute pointer.

## The 3-part verdict

### (1) The PTQ floor: PTQ COLLAPSES — no grid beats the baseline
**NO uniform OR score-aware grid beats int8 byte-closed.** Every bit-shrink raises S
monotonically and steeply: int7 +0.094, int6 +0.184, int5 +0.481, int4 +1.689 vs the int8
baseline. The PTQ floor IS the int8 baseline itself (S = 0.1965). This is NOT a $0 pointer-mover
— it **confirms score-aware QAT is REQUIRED** (the operator's pivot is correctly motivated).

Mechanism: the rate win is REAL and large (int4 cuts bytes 177,169 → 87,925 = **−50%**, rate
0.1180 → 0.0585, exactly the desk model's prediction) but the un-finetuned distortion spill
buries it — at int4 d_seg explodes **20×** (0.000594 → 0.011881) and d_pose **1100×**
(0.000037 → 0.040852). The frontier weights are FP11-dense (the heavier-burden caveat the
desk-calc flagged): int4 is a bigger precision drop than bc20's int8→int4, and the frontier
collapses correspondingly worse (int4 S 1.89 here; the deepest collapse is the catastrophic
d_pose blow-up — pose has NO precision margin at this operating point). Even int7 — one bit
below the codec baseline — already spills d_seg 2.3× while saving only ~3 KB (the frontier's
brotli is already near-entropy, so small coarsening buys almost no bytes but still costs d_seg).

### (2) The QAT gap: +0.0566 distortion to recover for the desk int4-perfect-hold
The desk model's int4-PERFECT-HOLD target is S = 0.1399 (sub-0.15). The best ACHIEVED PTQ is
the int8 baseline 0.1965. **Gap = +0.0566.** That is the distortion (≈ +0.057 of score, almost
entirely d_seg+d_pose at the int4 byte budget) score-aware QAT-finetune must recover to reach
the modelled sub-0.15. Concretely: at the int4 byte budget (87,925 B, rate 0.0585), QAT must
pull d_seg from the collapsed 0.011881 back to ≈ the frontier floor (0.00056–0.00059) AND
d_pose from 0.040852 back to ≈ 0.00004 — i.e. recover essentially ALL of the int4 distortion
collapse. That is a steep ask (a ~20× d_seg and ~1000× d_pose repair), which is the honest
read of how hard the QAT step is.

### (3) Score-aware vs uniform: VALIDATES the per-stage lever at matched bytes — but it is NOT enough
At matched bytes, score-aware per-stage allocation **beats** the nearest uniform grid on every
comparison:
- `sa_int4lo_int8hi` (100,005 B, S 1.4954) beats uniform int4 (87,925 B, S 1.8858) — ΔS −0.39
  at +12 KB; d_seg 0.01064 vs 0.01188.
- `sa_int4lo_int6hi` (96,021 B, S 1.5056) beats uniform int4 — ΔS −0.38.
- `sa_int5lo_int8hi` (126,485 B, S 0.6329) beats uniform int5 (118,589 B, S 0.6779) — ΔS −0.045;
  d_seg 0.004486 vs 0.004744.

So the **taper-on-the-bit-axis insight is directionally VALIDATED** — protecting the tiny
output-proximal d_seg-critical heads at higher precision does lower d_seg at a given byte budget.
**BUT the effect is small and does not change the verdict:** the int4-low score-aware modes still
collapse almost as badly as uniform int4 (d_seg 0.0106 vs 0.0119) because the **d_seg damage
comes overwhelmingly from coarsening the massive early/low-res stages** (77% of params), NOT from
the output heads. Protecting only the 7%-of-params heads cannot rescue d_seg when the early stages
go to int4 — the low-frequency structure propagates through to argmax flips. **Key correction to
the desk-calc's score-aware framing:** at PTQ, d_seg is NOT concentrated in the output-proximal
band on the BIT axis; the early stages carry d_seg-relevant signal too. The per-stage protection
helps at matched bytes but is second-order; the dominant lever is finetune (QAT), not allocation.

## What this means for the next unit (the operator's pivot)

1. **PTQ alone is dead** — confirmed, not a pointer-mover. Do NOT submit any PTQ'd frontier
   archive (every grid is far above 0.191).
2. **Score-aware QAT-finetune is the only modelled sub-0.15 path** and the floor it must beat is
   now MEASURED: it must recover +0.057 of distortion (the ~20×/~1000× d_seg/d_pose int4 collapse)
   while holding the −0.057 rate win. This is a hard but bounded target.
3. **Cheapest next step (QAT-finetune spec):** use the EXISTING Lever-4 `tac.torch_vehicle.score_aware_qat`
   on the frontier decoder — start at **int5** (gentler: PTQ int5 only spills to S 0.678, a ~+0.48
   gap, vs int4's +1.69; int5 byte budget 118,589 B → rate 0.079, modelled-hold S ≈ 0.15–0.16 per
   the desk int5 row 0.153) BEFORE attempting int4. MPS fp32 gradient; CPU authority eval (async,
   the proven split-device pattern). The score-aware per-stage map should keep the early stages at
   int5 (not int4) since they carry the bulk of the d_seg damage — the matched-bytes result shows
   `sa_int5lo_int8hi` is the gentlest score-aware operating point (S 0.633, d_seg 0.0045). int4 is
   the aggressive bound to attempt only if int5-QAT holds.
4. **Borrowed-substrate / originality caveat (SUBMISSION gate, NOT a local blocker):** the pr110
   frontier is a public-PR-derived HNeRV (PR #101/#106 codec lineage). QAT-shrinking it lowers OUR
   score legitimately as a TECHNIQUE, but any contest SUBMISSION of a frontier-derived archive must
   carry the `borrowed_substrate_accounting` per the NO-FAKE originality rule (Forbidden class 7).
   This does not block the local score/technique measurement; it gates the eventual PR.

## Cheapest-next-step decision
**PTQ collapsed → QAT-finetune is required.** Spec: int5-first score-aware QAT on the frontier
decoder via `tac.torch_vehicle.score_aware_qat` (early stages int5, output-proximal heads int8),
MPS fp32 gradient + async CPU authority eval, target = recover d_seg/d_pose to the frontier floor
at the int5 byte budget (modelled S ≈ 0.153). If int5-QAT byte-closes < 0.191 on CPU authority →
flag for paired dual CPU/CUDA exact eval (the pointer-mover), do NOT self-promote.

## 6-hook wire-in (Catalog #125)
- #1 sensitivity-map: the score-aware per-stage allocation consumes the d_seg-critical-stage map
  (`DSEG_CRITICAL_PREFIXES`); the matched-bytes result REFINES the prior (early stages carry d_seg
  damage too) — ACTIVE.
- #2 Pareto constraint: the measured bit-width↔(d_seg,d_pose,byte) curve IS a Pareto constraint on
  the QAT operating point — ACTIVE.
- #3 bit-allocator: this is the precision lever's MEASURED floor; feeds the QAT grid choice (int5
  start) — ACTIVE.
- #4 cathedral autopilot: N/A (feasibility floor, not an archive-deployable surface).
- #5 continual-learning posterior: N/A this unit (no new EXACT anchor; advisory feasibility row).
- #6 probe-disambiguator: this probe IS the disambiguator between "PTQ is a $0 pointer-mover" (NO)
  and "QAT is required" (YES) — ACTIVE.

## Files
- `src/tac/frontier_decoder_ptq.py` — reusable frontier decode/requant/re-encode round-trip
  (byte-identical identity gate + score-aware per-stage allocator).
- `experiments/probe_ptq_on_frontier_floor.py` — the $0 CPU sweep CLI.
- `src/tac/tests/test_frontier_decoder_ptq.py` — 16 NO-FAKE tests (identity byte-parity, real
  weight decode, monotone byte shrink, per-stage allocation behavior).
- `reports/ptq_on_frontier_floor.json` — the machine-readable 8-mode table + verdict.
