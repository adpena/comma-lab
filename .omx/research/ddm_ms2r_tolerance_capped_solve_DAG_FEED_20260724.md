# FEED-603-ms2r — tolerance-capped solve preflight

Date: 2026-07-24
Lane: `lane_ddm_ms2r_tolerance_capped_solve_20260724`
Evidence axis: `[macOS-CPU frozen-scorer advisory]`
`score_claim=false`; pointer `0.1910828242 [contest-CPU]` UNMOVED.
MAIN landing review is required.

## Outcome first

The requested solve did not cross its mandatory scientific admission edge:

```text
RG3 exact pair/bucket assignment
  9/24 hard pairs fully joined; 25 blocks missing
        |
        v
MS4 metric production
  Pose COMPLETE n600/batch32
  Seg / composite-R / dual PARTIAL
        |
        v
MS3 load_metric_custody_bundle(require_complete=True)
  REFUSE: PF2_BUCKET_INPUT_ASSIGNMENT_ABSENT
        X
        |
        +--> MS2R waterfilled tolerance homotopy NOT LAUNCHED
             rungs = 0
             real-coder objects = 0
             RD1 measured dual cells = 0/162
```

The refusal is a durable outcome, not a workaround target. The exact receipt is
`.omx/research/ddm_ms2r_tolerance_capped_solve_20260724T152730Z/00_preflight_receipt.json`
(SHA-256 `a253f844bd0a79ca548335e5abaddcdbbf3c831ff781bf6e566aecee54c93b72`).

## Re-derived objective and admitted successor

For one parse-back-exact object \(x\), with global error count
\(e(x)=\sum_b e_b(x)\), the successor solve is:

\[
x^\star =
\arg\min_{\substack{x\in\mathbb Z_{256}^n\\e(x)\le 136839}}
\left[
100\frac{e(x)}{117964800}
+\sqrt{10D_{\rm pose}(x)}
+\frac{25}{37545489}\min_{c\in\mathcal C}B_c(x)
\right].
\]

The error allowance is waterfilled over
`stratum × scorer_visibility × g4_temporal_class` using measured
bytes-per-error KKT duals. Same-pool tolerances compete and never add.
Uniform-tolerance rungs are controls only.

The coder set \(\mathcal C\) must retain separate raw-compact and best-coded
columns and race:

- raw compact representation;
- zlib-9;
- raw LZMA1;
- order-1 context arithmetic;
- the E4 Brotli-Q11 path;
- the G4 free decoder-derived spatial context model.

Every rung must report global allowance, per-block allocated allowance,
per-block realized errors, per-block measured dual, achieved d_seg,
d_pose through the active tube, raw bytes, best-coded bytes, winning coder,
and joint S. Those fields must all refer to one exact parse-back object.

## Visibility partition

The admitted waterfill operates only after scorer-recursive partitioning:

1. **both-blind / GAUGE** — exact resize-null and #401 blind coordinates cost
   zero bytes and zero tolerance, but every proposed gauge move must be
   reverified through uint8/R/parse-back because #532 falsified real-valued
   range(A) exactness after quantization;
2. **seg-only** — fine chroma below PoseNet's 2×2 chroma support;
3. **pose-only** — frame 0, structurally Seg-free, receives zero Seg tolerance
   and zero Seg description bytes;
4. **joint** — scorer-visible coordinates governed by rank-4 margin-Fisher,
   Pose6, exact composite-R, and corrected inner-Jacobian custody.

Per-type mass and byte spend remain NULL in this lane because no admitted
candidate was measured.

## Exact arithmetic correction

The C1 source row has measured
`d_seg=0.0001519690619574653`. Multiplying by 117,964,800 scored pixels gives
exactly **17,927** errors. The charter's **17,931** is the rounded
`1.52e-4 × N` statement, four errors higher, and is not exact custody. The box
allowance is 136,839 errors, or `7.633123221955709×` the measured C1 count.

## RD1 and knee edge

The source RD1 cube was rehashed and validated as the exact
`3 edges × 6 strata × 3 visibilities × 3 G4 classes = 162` Cartesian product.
The output supplement preserves 162/162 lambda cells as NULL; it does not
transfer pooled proposal-channel prices.

RD1's proposal-channel knee is quoted, not recomposed:
138,801 bytes, d_seg `0.07051923116048177`, d_pose `36.6181847780574`,
joint S `26.28022355199344`, one measurement-harness object. The MS2R knee and
channel-suboptimality price remain NULL because there is no MS2R rung.

## Triality and routing

- DSL/apparatus:
  `tools/preflight_ddm_ms2r_tolerance_capped_solve.py` plus the existing
  `tac.optimization.ddm_typed_quotient_solve` and strict MS3 loader.
- DAG: this FEED.
- Equation:
  registered `ddm_tolerance_capped_min_score_waterfill_v1`, callable at
  `tac.canonical_equations.ddm_ms2r_tolerance_capped_solve_20260724:tolerance_capped_rung_score`.

Downstream `c1/ic1` composition and the train-decision SOLVE column remain
blocked. Exact reactivation is: close the 25 RG3 assignment obligations, rerun
MS4 to a loader-accepted `BUNDLE-COMPLETE`, then resume from the same MS3 gate.
