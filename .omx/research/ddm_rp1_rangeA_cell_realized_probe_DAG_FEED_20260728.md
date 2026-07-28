# DAG FEED — ddm_rp1 range(A)-cell realized probe [no-triality] [p0-ledger-ok]

**2026-07-28 · `[macOS-CPU frozen-scorer advisory]` · score_claim=false · pointer 0.1910828242 UNMOVED**

## Node: rp1 — range(A)-cell realized probe → CELLS HOLD (verdict routes family-d GN)

**Question (gc5 M2-Q2 fork):** does range(A)-projected + generic-ker-fill + uint8 of solved frames stay
in the same SegNet argmax cells through the real decode? HOLD → sc1-far = engine capacity → family-d GN;
BREAK → re-derive cell-index formulation.

**Custody correction (matches r6cal 07-27):** the 1.52e-4 q1 object is a MEASURED SCORER CONTROL, ZERO
frame records on disk (only q4/q8 box-solve chunks exist). Charter C0-on-1.52e-4 BLOCKED. Probe re-scoped
to GT frames (`gt_n600.npz`), the highest-margin / OPTIMISTIC-bound operating point.

**MEASURED (n600, real upstream SegNet+PoseNet, C0 custody = 0 flips):**
- C1 range-carrier zero-ker uint8 lift `Y=round(clip(project_range(X)))` (decoder-derivable from A(X)):
  **d_seg = 3.6296e-4 = 2.39× q1(1.52e-4)** ∈ pre-registered ~2–3× HOLD band;
  **d_pose = 3.965e-4, contribution 0.063** ∈ R1 tube (0.127).
- C2 = round(project_range+project_kernel) = X ≡ C0 (identity verified = 0.000e+00) → degenerate by
  construction (uint8 solve) → ALL realization damage lives in C1 (charter C1/C2-split correction).
- Margin telemetry: flips @ pre-round margin 0.034 vs held 5.61 (**166× gap**) → slack absorbs it.
- Per-class: Road 46% / Undriv 30% mass; Lane 4.45e-3 + Movable 3.91e-3 top rates; MyCar 4.78e-5 (robust)
  → matches natural d_seg residual, no new failure mode.
- #532 reproduced: A-space uint8 break = **63.82** (their Δ=62.74); float range A-exact 1.9e-11.

**VERDICT: CELLS HOLD.** The range-carrier+uint8 formulation reaches the cells at ~q1 precision (existence
proof) → sc1-far seed (d_seg 0.070519 = 464× q1) is ENGINE CAPACITY, not formulation break →
**family-d GN-in-description-coordinates build = named next arm.**

**verdict_scope:** HOLD on GT (optimistic bound) + existence proof; NOT the box-solve smaller-margin
operating point. NEXT MEASUREMENT: inflate box-solve 277.7 MB → identical C1 probe on box-solve frame_1's
(tool `--substrate boxsolve`, inflate stub owed).

**Artifacts:** memo `.omx/research/ddm_rp1_rangeA_cell_realized_probe_20260728.md`; tool
`tools/measure_ddm_rp1_rangeA_cell_probe.py`; SSD `/Volumes/VertigoDataTier/pact/ddm_rp1_20260728/`.
