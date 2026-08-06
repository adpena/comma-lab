# ddm_dk1 - Lattice-Native Pose-Null Realizer Receipt

## Leakage Ladder

Axis: `[macOS-CPU frozen-PoseNet advisory] small-n`; `score_claim=false`; `promotion_eligible=false`;
`n600_run=false`. Inputs are real tq1c parent frames and real tq1c block16 phase offsets from the
SSD tier. Selection is the first four nonzero phase-field blocks, centered to even 2x2 scorer
blocks. The target is a local shifted scorer-plane RGB delta projected to `ker(A)`.

| Arm | mean pose leakage sq in A(Dx) | median pose leakage sq | mean real PoseNet dpose vs parent pair | mean scorer-delta discrepancy |
|---|---:|---:|---:|---:|
| naive uniform scorer-round | 0.072179250000 | 0.043272000000 | 4.913175500720e-09 | 0.710732227913 |
| Dykstra round/project, k=8 | 0.039971942048 | 0.049681201777 | 2.123022215385e-10 | 0.405178292895 |
| CVP/Babai kept-set enum | 0.007107145151 | 0.005004955490 | 6.435523410295e-13 | 0.0652860850619 |

Result: CVP is the local best on the aggregate ladder, with lower pose leakage and better local
scorer-delta fidelity than the naive and Dykstra arms on the measured blocks. This is not a score,
not a SegNet dseg verdict, and not a population claim.

## Build

Added `src/tac/optimization/lattice_native_pose_null_realizer.py`:

- Uses the exact #580 `DisjointResizeOperator` private supports and integer-derived D weights.
  `assumes_uniform_025=false` is recorded in the geometry receipt.
- Exposes a callable block interface for successor c-space/phase solvers:
  `private_block_geometry`, `project_scorer_delta_to_pose_null`,
  `realize_lattice_native_block`, `dykstra_integer_realize`, and `cvp_integer_realize`.
- Races the required arms:
  - naive baseline: round scorer deltas, broadcast to private taps;
  - Dykstra: continuous projection to `ker(A @ D_private)` plus bounded integer round/clip
    with error-feedback;
  - CVP/Babai: bounded private-window integer enumeration around the continuous preimage,
    minimizing `(pose_leakage_sq, seg_discrepancy, L1_delta)`.

Added `tools/measure_ddm_dk1_lattice_realizer.py` to reproduce the small-n receipt and write
`.omx/research/ddm_dk1_20260806/lattice_realizer_measurement.json`.

Added `src/tac/optimization/tests/test_lattice_native_pose_null_realizer.py` with coverage for:
pose-null rank/kernel, exact nonuniform D weights, operator parity, odd-block refusal,
Dykstra/CVP nontriviality, pruned-scope diagnostics, and the three-arm interface.

## Solver Scope

Measured command:

```bash
.venv/bin/python tools/measure_ddm_dk1_lattice_realizer.py \
  --n-blocks 4 \
  --threads 6 \
  --out .omx/research/ddm_dk1_20260806/lattice_realizer_measurement.json
```

Inputs:

| Input | SHA-256 |
|---|---|
| `/Volumes/VertigoDataTier/pact/ddm_et2_20260806/parent_tq1c_decode/submission/inflated/0.raw` | `82de098f5b97e6c61c7a53b4180f425117ea2e3c89e6ab435e7aea423f81291a` |
| `/Volumes/VertigoDataTier/pact/ddm_et2_20260806/phase_field/tq1c_block16_offsets.npy` | `71365c74d49c2c0f611b4a3e01cbfe735177398c0510751bdf0f4642fad5af0d` |

Selected blocks:

| pair | phase block | scorer row | scorer col | dy,dx |
|---:|---:|---:|---:|---|
| 0 | 335 | 168 | 248 | `[-1, 3]` |
| 0 | 337 | 168 | 280 | `[2, 0]` |
| 0 | 348 | 168 | 456 | `[2, 0]` |
| 0 | 350 | 168 | 488 | `[2, 0]` |

Dykstra convergence: two measured blocks stopped at the iteration cap (`8/8`), and two detected a
cycle after three iterations. The receipt keeps `cap_stop=true` where the cap bound.

CVP scope: `tap_radius=1`, `max_channel_candidates=9`, `max_pixel_candidates=16`,
`max_combinations=250000`. Each measured block evaluated 65,536 kept combinations. Diagnostics label
the scope as `BABAI_PRUNED_TOP_K_WITH_EXACT_ENUMERATION_OF_KEPT_SET`,
`exact_declared_scope=false`, and `global_integer_optimum_claim=false`. This is exact over the kept
set only; it is not a global MIQP/integer optimum claim.

## Seam Sweep

Counts over the searched live/sibling surfaces:

| Classification | Count |
|---|---:|
| `LATTICE-NATIVE-SOLVABLE` | 5 |
| `PRICED` | 2 |
| `NEGLIGIBLE` | 1 |

| Site | Classification | Owner / fire order |
|---|---|---|
| `experiments/ddm_q3x_q3_convergence_measurement.py:182-184` float `project_null` then scorer-lattice `round` then camera realization | `LATTICE-NATIVE-SOLVABLE` | Replace the local realization with DK1 `realize_lattice_native_block` before any promoted Q3 convergence row. |
| `experiments/ddm_sq1_pose_null_constrained_paint.py:108-115` float null field optimized then rounded to scorer paint | `LATTICE-NATIVE-SOLVABLE` | SQ1 successor should use DK1 per-block integer realizer; SQ1's integer caveat remains formulation-scoped evidence. |
| `experiments/ddm_q31_q3_constrained_solve.py:267-304,566` Q3 float constrained solve snapped to uint8 scorer paint | `LATTICE-NATIVE-SOLVABLE` | Q31 successor should reopen only with box/lattice-native realization; q31 itself remains not-cleared formulation scope. |
| `experiments/ddm_lr2_realization_ladder.py:538-579` solve0 null field projected then rounded through private support paint | `LATTICE-NATIVE-SOLVABLE` | LR2/SQ1 successor can call DK1 for the 2x2 block realizer. |
| `experiments/ddm_lr2_realization_ladder.py:819-857` FO1 AC/Q3 field projected then rounded through private support paint | `LATTICE-NATIVE-SOLVABLE` | LR2 FO1 successor can call DK1 for in-null block realization. |
| `tools/measure_ddm_rp1_rangeA_cell_probe.py:124-130,169` `project_range` then `round(clip(...))` | `PRICED` | RP1 priced the range(A) uint8 break: A-space break `63.82`, cells held at 2.39x q1, pose within tube. No unpriced exactness reliance remains for this route. |
| `src/tac/optimization/uint8_lattice_feasibility.py:468-650` real preimage then exact uint8 solve path | `PRICED` | Already lattice-native for integer scorer-plane targets; keep as the general D-lattice solver, not a DK1 blocker. |
| `experiments/train_tr1_partition_renderer_mlx.py:687-722,1474-1476,1587,4507` token/weight/target quantization | `NEGLIGIBLE` for DK1 scope | Token/weight quantization is description-level lattice with STE and pricing; target materialization is cache construction, not a live `ker(A)`/`range(A)` exactness claim. |

Live reliance answer: yes, the Q3/project-null sibling surfaces still contain float-first
`ker(A)` exactness followed by post-hoc rounding. DK1 now provides the callable lattice-native block
realizer, but those sibling experiment scripts are not rewired in this commit. The range(A) seam is
not unpriced: RP1 reproduced the #532 break and bounded the formulation outcome.

## Recall Evidence

Sources searched beyond the charter seeds:

- Memory registry: `uint8`, `PF3`, `realized uint8`, `sq1`, and `receiver` hits. This changed the plan
  by forcing a realized-uint8-quanta receipt and avoiding any proxy-only float-null claim.
- `.omx/research/ddm_rp1_rangeA_cell_realized_probe_20260728.md`: confirmed the known range(A)
  seam is already priced rather than an unresolved DK1 blocker.
- `.omx/research/ddm_et2_metric_amendment_20260806.md`: confirmed pose-nullity is `A delta = 0`
  while the metric chooses placement; DK1 therefore measures uint8 leakage per arm instead of
  treating a Euclidean projector as a verdict.
- `.omx/research/ddm_sq1_eta_seg_and_hinge_ab_20260803.md`: confirmed D-support privacy and the
  integer-actuator caveat, so DK1 used exact private D weights and did not re-open the settled
  same-value-paint parity facts.
- `.omx/research/ddm_q31_20260804/Q31_Q3_CONSTRAINED_SOLVE_RECEIPT_20260804.md`: confirmed Q31 is a
  formulation-scoped not-cleared result with whole-2x2 Q3 snapping and pre-uint8 clipping; DK1
  classifies it as a successor seam, not as killed family evidence.
- `.omx/research/ddm_ms1_min_description_lattice_solve_DAG_FEED_20260724.md` and
  `.omx/research/ddm_ms2_typed_quotient_solve_DAG_FEED_20260724.md`: confirmed the lattice-first
  doctrine and typed CVP/effective quantum route.
- Canonical-equation registry search for `pose_null`, `uint8`, `quantum`, and `lattice`: confirmed the
  standing AC-only pose-null subspace and realized-quantum discipline; no broader machinery was
  spawned because DK1 has a named exact-row consumer path only through ET/Q3 successors.

## Boundary

No archive was built. No n600 scorer slot was used. No exact score, dseg, or population result is
claimed. This commit only lands the local lattice-native block realizer, its small-n PoseNet
leakage receipt, and the seam ledger.

Own-vehicle frontier unchanged: `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`.
