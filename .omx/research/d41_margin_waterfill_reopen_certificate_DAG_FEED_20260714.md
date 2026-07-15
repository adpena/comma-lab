# DAG FEED — D41 margin-adaptive mixed-precision SegNet forward, reopen certificate (2026-07-14)

**Merge target:** `sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md` (standalone; main merges).
**Ledger row owed to main-merge:** `D41` in `.omx/state/deferral_ledger.md` (`.omx/state` outside this
worktree's frontier — verdict routed here; ledger note update owed on cherry-pick).
**Pointer:** submittable **0.19108** / bank **0.18804** — UNMOVED (throughput apparatus). MEANS.
**Axis:** `[macOS-CPU-torch 1-thread advisory / NumPy-fp32]` NON-PROMOTABLE, no score claim.
**Probe (runnable, $0):** `experiments/probe_d41_margin_waterfill_from_cached_fixedpoint.py` — reads only
the cached n600 receipts in `experiments/results/throughput_authority_ladder_20260714/`. No forward run.

## Node FEED-d41 — reopen of `NO_ADMITTED_PRECISION_IN_LADDER` under per-channel + margin-waterfill

### The reopenable
Naive verdict (`fixedpoint_qdq_rung2_nogo_certificate_DAG_FEED_20260714.md`): a SINGLE GLOBAL uniform
fixed-scale WnAn QDQ flips boundary argmaxes at every 8..24-bit width → `NO_ADMITTED_PRECISION_IN_LADDER`,
scoped **INSTANCE** (global-uniform fixed-scale). The audit `naive_nogo_rescoping_audit_498_20260715.md`
ranked D41 the top-1 reopenable: does **per-channel scales + a margin-waterfilled bit schedule** (high
bits on the boundary annulus per Fisher=margin 0.978, low bits on the flat interior) admit a LOWER
average bit-width that is STILL argmax-bit-identical?

### What the cache holds (measured, n600, pixel-exact flips vs the verified fp32 SegNet argmax)
fp32_control self-argmax-exact on every receipt → the fp32 argmax IS the reference authority.

| formulation | arm | tot_flips /117.96M px | pairs_flip | px_preserve | argmax-EXACT (==1.000000)? |
|---|---|---:|---:|---:|:--:|
| fixed-scale fp32-accum | w8a8 | 1,200,717 | 600 | 0.989821 | no |
| fixed-scale fp32-accum | w8a8_head_fp32 | 1,215,408 | 600 | 0.989697 | no (mix HURTS vs uniform w8a8) |
| fixed-scale fp32-accum | w16a16 | 13,197 | 600 | 0.999888 | no |
| fixed-scale fp32-accum | w20a20 | 9,066 | 517 | 0.999923 | no |
| fixed-scale fp32-accum | w22a22 | 8,965 | 471 | 0.999924 | no |
| fixed-scale fp32-accum | w24a24 | 8,960 | 457 | 0.999924 | no (**plateau ~8960** — scale-clip artifact) |
| dynamic-exact-absmax int64 | w25a25 | 13 | 13 | 0.99999989 | no |
| dynamic-exact-absmax int64 | w26a26 | **3** | 3 | 0.99999997 | no |
| exact_int64 | w26a26 | 4 | 4 | 0.99999997 | no |
| mixed_int64 (per-LAYER geometry mix) | mixed_w26_w30_geometry_safe | **1** | 1 | 0.99999999 | no |
| weight_l1_int64 (per-LAYER mix) | weight_l1_safe_w26_w31 | **1** | 1 | 0.99999999 | no |

**Minimum argmax-EXACT arm = NONE.** No cached QDQ arm — uniform (8..26 bit, fixed-scale OR
dynamic-exact-absmax int64) NOR the two hand-designed per-LAYER geometry/weight-L1 mixes (w26..w31) —
reaches argmax-preservation **1.000000** on n600. The best cached leaves **≥1 residual flip**.

### The binding wall = fp32 argmax TIES (bit-ALLOCATION-invariant)
The residual flips at the CEILING arms sit at the min-margin boundary pixels, and their fp32 margins are
**at or below the fp32 reduction-order noise floor**:
- dynamic w26 (3 flips): fp32 min-margins `4.77e-7`, `4.05e-6`, `1.43e-6`.
- exact_int64 w26 (4 flips): `4.77e-7`, `4.77e-6`, `4.05e-6`, `7.15e-6`.
- per-layer geometry mix / weight-L1 mix (1 flip, pair 11): fp32 min-margin **0.000e+00** — an EXACT
  fp32 argmax tie (two classes equal to fp32 precision).

These are not quantization error — they are pixels where the fp32 argmax itself is knife-edge. The
certificate's own custody note (1-pixel argmax mismatch between 1-thread and ambient-thread fp32,
`max_margin_abs_delta=3.6e-5`) independently proves fp32 is non-deterministic at this exact pixel class.
A margin-waterfill sheds bits on flat-interior channels while keeping bits on annulus channels — but the
residual lives at the ties AT the ceiling, and no finite bit ALLOCATION (per-channel included) can
preserve an argmax whose fp32 margin is 0.0..~5e-7. **Argmax-exactness requires ≥ fp32 precision at the
tie pixels**, so a "cheaper argmax-identical forward" is self-defeating for this scorer.

Separately: the fixed-scale plateau (w20→w24 stuck at ~8960 flips) is a **fixed-calibration-scale
representation artifact** (a global scale cannot cover the dynamic range) — dynamic per-TENSOR-exact-absmax
already collapses it to 3. So the scale-artifact residual is fully explained by per-tensor scaling; only
the tie residual survives at the ceiling, and it is the true wall.

### Verdict (D41 reopen)
**NO** — per-channel + margin-waterfill does NOT reopen a free argmax-identical cheaper forward.
Two independent legs:
1. **MEASURED (cached, $0):** no QDQ arm reaches argmax-preservation 1.000000 at n600 at any cached
   precision, uniform OR per-layer-mixed, fixed-scale OR int64-exact-accum. The residual is fp32 argmax
   ties (margin 0.0..~5e-7), which is bit-ALLOCATION-invariant. Per-tensor-exact-absmax already sits at
   this wall at 26 bits, so per-channel scales (the D41 tightening) have no headroom to reach exactness
   at a LOWER average bit-width — the trend w25→w26 (13→3 flips) shows exactness demands MORE bits, not
   fewer, i.e. toward fp32.
2. **BLOCKER (the exact per-channel-waterfill measurement):** the cached arms are whole-network uniform
   bit-widths + 2 hand-designed per-LAYER mixes. ABSENT and required for a genuine waterfill:
   (a) per-CHANNEL quantization scales, (b) per-layer/per-channel argmax-flip ATTRIBUTION (each arm
   reports only the whole-network flip count — which channel/layer caused a residual flip is
   unidentifiable). A real margin-waterfilled per-channel schedule needs per-channel ablation forwards
   (channel × bit sweep at n600) — a SWEEP, not $0-cached / one-forward. No admitted bit-width fabricated.

### Scope tightening (the real finding)
The certificate advances **INSTANCE → FORMULATION**: not merely "global-uniform fixed-scale fails," but
"**no fixed-point QDQ bit-ALLOCATION reaches argmax-exact on n600**, because the binding residual is fp32
argmax ties, not quantization error." Per-channel-scale specifically remains formally UNMEASURED (missing
artifact) but is ruled out on headroom. The lossless EXACT-int64 reorder-invariant path (a DIFFERENT
formulation — determinism, not low-bit speed) is untouched by this and stays open (D51/L70).

**Net for the #456/#449 throughput P0:** the "make the frozen-SegNet forward cheaper AND argmax-exact"
route via fixed-point QDQ is now FORMULATION-dead — argmax-exactness pins precision to ≥ fp32 at the tie
pixels. Cheaper-forward EV must move to a formulation that does NOT require pointwise argmax-exactness
(e.g. a certified interval bound that tolerates the known tie set, or the lossless int64-determinism
path), not to a finer bit allocation of the same QDQ scheme.
