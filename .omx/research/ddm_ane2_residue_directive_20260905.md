# ddm_ane2 residue — owed items after "Drift can be engineered to fix" landed (directive, 2026-09-05)

Tokens: `[no-triality] [p0-ledger-ok]`. Source: ane2's final report (memo `.omx/research/ddm_ane2_engineer_the_precision_drift_20260905.md`).
Owner until claimed: MAIN. Priority: MEANS (screening throughput), below every exact-row unit; fire when an Opus slot is idle and no Metal cell contends.

## ITEM 1 — Re-measure the g13 / g13stem finalists through the FIXED conversion path
ane2 found its own instrument defect: `FP16ComputePrecision(op_selector=…)` makes coremltools declare the model OUTPUT fp16 even when it
transforms zero ops (MIL dtype 65552 vs 65568). Every SegNet flip rate in the split-point ladder and the selective-fp32 table is therefore an
UPPER BOUND (the pass is conservative). Owed: fix the conversion (force the fp32 output dtype), re-run n600 generated-decode flip rates for
`all fp16`, `g13`, `g13stem`, and the tail k=64 rung, and update the anchors on `scorer_fp16_drift_by_axis_v1` + `BACKEND_AXIS_VERDICTS`.
Acceptance: the fixed-path g13stem rate ≤ 3.3e-05 at ≥ 75% ANE with the n120→n600 agreement recorded; if it fails, the pass is withdrawn in the memo.

## ITEM 2 — Explain the tail-split worsening (MEASURED, UNEXPLAINED)
PoseNet's dim-0 error is pinned at 0.150–0.154 for every fp32-TAIL split and the tail split gets monotonically worse past k=32; both of ane2's
candidate mechanisms died to its own instruments (cast census 1–8 casts; exactly 1 ANE segment). Owed: the per-op activation diff (fp16 vs
fp32 at each boundary op 0:18) that localizes the error to a named op and a named cause. Acceptance: one named op whose fp16 activation error
explains ≥ 80% of the dim-0 delta, or a scoped negative.

## ITEM 3 — Wire the g13stem ANE SegNet as a screening backend only after ITEM 1 passes
`coreml_cpu_fp32` (bit-exact, 1.94×) is the screening backend of record today (38/39 argmin agreement on fs1/pr1's sweep). The ANE
`g13stem` backend (75.5% ANE, 4× less latency than the k=64 tail split) enters `tac.ane_screening` as an admissible SEG-ONLY screening axis
only with ITEM 1's fixed-path measurement attached; pose stays refused on the ANE (fp16 head cure hd128 = 7.02e-5 self-MSE is still 11× d_pose).
