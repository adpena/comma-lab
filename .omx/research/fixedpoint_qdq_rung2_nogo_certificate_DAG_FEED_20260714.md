# DAG FEED — fixed-point QDQ rung-2 certificate (2026-07-14), verdict-scoped

**Merge target:** `sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md` (standalone; main merges).
**Pointer:** submittable **0.19108** / bank **0.18804** — UNMOVED (throughput apparatus). MEANS.

## Node FEED-fpqdq — MEASURED n600: uniform fixed-scale QDQ does NOT preserve the SegNet argmax.

**Measurement:** `tools/probe_fixedpoint_scorer_forward_n600.py`, gt_n600 (real 0.mkv), bits {8,10,12,14,16,18,20,22,24},
calibrated fixed-scale WnAn QDQ + fp32 accumulation, frozen 1-thread CPU-Torch SegNet control.
Artifact: `experiments/results/throughput_authority_ladder_20260714/fixedpoint_scorer_forward_n600_fresh_89b970ff60.json`.
Axis: `[macOS-CPU-torch 1-thread advisory]` NON-PROMOTABLE, no score claim.

**Verdict = `NO_ADMITTED_PRECISION_IN_LADDER`.** `minimum_argmax_exact_arm = null`,
`minimum_training_tolerance_arm = null` — NO bit-width in 8..24 holds the argmax exactly (or within
training tolerance) at n600. Narrowest level = **n600 INSTANCE** of the **calibrated FIXED-SCALE
uniform WnAn QDQ FORMULATION** (per the probe's own verdict_scope). NOT a family kill.

**Custody note (separate axis):** the one-thread control vs legacy ambient-thread GT cache differs by
`argmax_mismatch_pixels=1`, `max_margin_abs_delta=3.6e-5` — an explicit thread/reduction-geometry custody
delta, audited separately, NOT the QDQ verdict (not silently coerced).

**What this KILLS (formulation):** the cheap-low-bit fast SegNet forward as authority via a SINGLE
GLOBAL/UNIFORM fixed scale — a global scale cannot cover the dynamic range without flipping boundary
argmaxes. This is the negative that MOTIVATES the reformulation, not a throughput dead end.

**Reformulation queue (untested formulations — the required alternatives):**
1. **Margin-adaptive MIXED-precision** (the frontier-math margin-waterfill): high bits on the boundary
   annulus / low margin, low bits on the flat interior. Uniform failing is the reason the margin field
   (Fisher=margin 0.978) must drive the per-region/per-layer bit allocation — the interval-arithmetic
   argmax certificate `L_top1 > max(U_other)` gives the per-layer error budget.
2. **Per-tensor / per-channel scales** (not a single global scale) — the classic low-bit fix the uniform
   fixed-scale omitted (sister of the #147 int5 LSQ/per-channel lesson).
3. **EXACT int64 reorder-invariant accumulation (LOSSLESS)** — a DIFFERENT path, untouched by this LOSSY
   QDQ probe: it gives cross-process bit-identity (the L70 determinism unlock) WITHOUT quantization loss,
   so it preserves argmax by construction. It buys GPU/ANE *determinism* authority, not low-bit *speed*.
   The speed-via-low-bit end-state depends on reformulation 1/2 passing; the determinism end-state does not.

**Net for the throughput P0:** the fast-low-bit-forward-as-authority is NOT admitted at uniform
fixed-scale; the live path is margin-adaptive mixed precision (needs the per-region certificate at n600).
The lossless int64 determinism path is independent and still open. Certified-forward remains rank-1 in
the queue but must be MIXED-precision, not uniform.
