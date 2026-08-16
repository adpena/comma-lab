# ddm_wd3 F64 verdict + the fresh-family boundary read — W96 does NOT fire

Date: 2026-08-15/16 boundary · Owner: MAIN (#1070 fire chain) · Axis: [macOS-MPS train / n60 subset
advisory — NEVER a score] · Instance: `F64` (factorized_d4_w64_r19, scorer-free birth, 65 epochs)

STORES CONSULTED: F64/TRAIN_RESULT.json + 14 epoch eval JSONs (APDataStore custody) · D56 verdict
memo `ddm_wd3_d56_fresh_dense_verdict_20260815.md` · canonical equation
`wd3_fresh_topology_pose_carry_v1` (2 anchors) · sealed wd3 charter ARM_ORDER + spec law · memory
`[[same_defect_negatives_masquerade_as_family_convergence_20260805]]` +
`[[wd3-fresh-topology-pose-carry-and-seg-asymptote-read]]`.

## The run

- Governed watched launch 65, attempt 1 (the D56 MPS-env lesson applied at compose time). rc=0 in
  **6,851 s (~105 s/epoch)**, peak RSS 7,511 MiB, run.log clean, no watcher alerts, done receipt
  clean.
- `TRAIN_RESULT.json`: `complete: true`, `instance_status:
  TRAINED_PENDING_N120_IF_NEGATIVE_AND_N600_SAME_INSTRUMENT`, `n120_negative_confirmation_run:
  false`, `n600_same_instrument_run: false`. All payloads + stage checkpoints retained.

## Measured endpoint (n60 fixed subset, same instrument as W0/D56)

| epoch | hard_d_seg | d_pose |
|---|---|---|
| 40 | 0.0036293 | 0.4997 |
| 50 | 0.0030519 | 0.5892 |
| 60 | **0.0029242 (best)** | 0.3714 |
| 65 | 0.0029793 | **0.2325** |

Read precisely: F64 is a DECELERATING DESCENT (ep40→60 −19%), not the flat asymptote D56 showed
from ep40. Its last-10-epoch band is 0.0029–0.0032. Pose oscillates 0.23–0.59, no convergence —
the registered fact-1 handicap, as predicted.

## The four-arm table at matched total budget

| arm | hard_d_seg | ratio vs warm floor | d_pose |
|---|---|---|---|
| W0_warm | 0.0010857 | 1.02× | 0.02294 |
| W0_reset | 0.0010628 | 1.00× (floor) | 0.03408 |
| D56 fresh dense w56 | 0.002682 | 2.52× | 0.3105 |
| F64 fresh factorized w64 r19 | 0.0029242 | 2.75× | 0.2325 |

## Adjudication

1. **F64 instance verdict: NEGATIVE-LEANING.** verdict_scope: instance (INSTANCE(F64, n60) — one
   config, one subset, one seed). Same standing as D56: per the sealed spec law "a negative cannot
   be emitted from n60," NO family word is emitted from this row; the n120 seeded confirmation is
   the owed discharge for BOTH fresh arms.
2. **The dense-vs-factorized comparison is ONE finding, not two.** Two DIFFERENT capacity forms
   (dense w56 vs factorized w64 rank-19) converged to endpoints 9–11% apart while both sit
   150–180% above the warm floor. Per the same-defect-negatives law
   ([[same_defect_negatives_masquerade_as_family_convergence_20260805]]), this is one instance of
   one defect measured twice: the FRESH-INIT OPTIMIZATION REGIME at this budget (no wd2 curriculum
   inheritance + the registered ~3× optimizer-state pose handicap bleeding into the joint
   objective), not the capacity form.
3. **W96 does NOT fire.** The sealed gate requires `capacity_pressure_confirmed: true`. The
   fresh-arm evidence CANNOT confirm capacity pressure: both capacity forms land near-identical
   gaps, and the two live explanations (capacity vs regime/budget) predict exactly this
   observation equally. A wider fresh arm (W96, ~2 h Metal) would measure the same confound a
   third time. The disambiguating experiments are DIFFERENT axes: longer-budget fresh (regime
   test) or warm-lineage-at-w56 (capacity test at matched form) — neither is chartered; both are
   recorded as reactivation criteria, not fired.
4. **Family disposition (pending n120):** the fresh-init distillation family stays
   NEGATIVE-LEANING at instance scope ×2. If the n120 confirmation upholds both instance
   negatives, the family parks with a measured reactivation ladder (warm-at-w56 · longer budget ·
   curriculum-inherited fresh birth); admission-n600 prospects then rest on the W0 warm arms,
   whose endpoints sit AT the teacher floor — i.e. byte savings, not seg, would have to carry any
   admission (the wd2 refusal already measured that trade 8.2× over the bar at w48-equivalent).
5. **n120 mechanism (measured from source):** no eval-only subcommand exists (actions: compile ·
   verify-build · prepare-arm-birth · prepare-teacher-scorer-cache · train · inventory). The
   verdict compiler `compile_n120_negative_confirmation` (:2551) demands two RETAINED
   receiver-closed `ddm_wd3_retained_subset_evaluation.v1` rows at exactly the seeded
   nonprefix n120 (`pair_ids` match + same `cache_surface_sha256`), candidate vs matched
   baseline. The cache-prep baseline row is n60 (controller subset), so BOTH legs must be
   produced fresh. Discharge path: a small eval-only harness reusing
   `evaluate_subset_and_retain` on the retained endpoint checkpoints (D56 ep65, F64 ep65,
   W0_warm endpoint as matched baseline) over `config["subsets"]["negative_n120"]` — built as
   the next scorer-free unit, fired when the Metal/memory window admits.

## Side receipts this boundary

- wc1 optimized n600 r3 FIRED (launch pid 50857, watched, governed): admission re-check passed
  post-F64 (available 108.8 GiB, pressure 1) — the r2 refusal adjudication (real transient
  file-cache pressure) held: no code change, single retry, admitted.
- gb1 D5 launch-tree aggregation remained correct across the F64+wc1 overlap window (no phantom
  double-charge in the admission decision).
