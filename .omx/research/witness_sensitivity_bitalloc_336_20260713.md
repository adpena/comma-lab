# Witness sensitivity bit allocation #336 — n600 byte-closed APPLY

**Status:** `COMPLETE_ADVISORY_REJECT`  
**Axis:** `[macOS-CPU advisory; exact shipped LVLS1 NumPy receiver; frozen CPU scorers; n600]`  
**score_claim:** `false`  
**pointer_moved:** `false`  
**research_only:** `true`  
**verdict_scope:** `INSTANCE x FORMULATION` — frozen canonical #406 V9 ep150 EMA-best
checkpoint plus post-hoc independent scalar precision transforms. The negative joint result does not
kill sensitivity allocation, entropy co-design, jointly measured search, or retrained witness
families.

## One-line outcome

**MEASURED n600:** all 18/18 tensor curves completed. The reverse-waterfill KKT proposal produced a
36,975-byte receiver-closed archive with `d_seg=0.11998607211642796` and
`d_pose=152.6197926660713`, for `Delta S=+8.72053480699927` versus the 63,664-byte int8 baseline:
**REJECT**. The lossless #461 chart composed exactly but saved only another 77 bytes on this changed
grid (36,898 bytes, same decoded tensor state and distortions, `Delta S=+8.720483535859879`), so the
final admissible assignment remains the all-int8 baseline.

## Authority and target custody

- Target checkpoint:
  `experiments/results/v9_cgauge_432_coherent_arm_20260711/levelset_witness_ema_BEST.npz`.
- Target SHA-256: `2599ad8b396af2af220a3bdbeee2ade92f194771ae6ef01a6faa15d39333484c`
  (**MEASURED** before the run and checked on resume).
- Real GT: `experiments/results/mlx_fleet_gt_cache/gt_n600.npz`, 600 pairs, SHA-256
  `cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6`.
- Result root: `experiments/results/witness_sensitivity_bitalloc_336_20260713T042157Z/`.
- Response schema: `witness_section_precision_response_curves.v1`, lineage-generalizing
  `jrd_pr110_section_response_curves.v1` from PR110 sections to LVLS1 witness tensors.
- No MPS, MLX scorer, cloud, paid dispatch, or `upstream/evaluate.py` was used. MPS is never score
  authority.

## Reused settled apparatus

- **#157 reused, not rebuilt:**
  `tac.losses.variable_level_waterfill_allocator.solve_waterfill_allocation` lower-convex-hull
  discrete KKT/reverse-waterfill, plus `verify_kkt_marginal_equalization`.
- **#153 reused:** per-tensor coarsening/deletion classification. Strict deletion admission uses the
  MEASURED repeat floor; the historical `Delta S_task <= 0.005` label remains separate.
- **#406 extended:** the frozen checkpoint was copied, never modified. Every candidate used
  `levelset_byte_close_and_eval.build_levelset_blob` and the actual shipped `_INFLATE_PY` receiver.

## Allocation law

For tensor `t` and rung `q`, the probe MEASURES archive bytes `B_tq`, SegNet distortion
`d_seg,tq`, and PoseNet distortion `d_pose,tq` on the same 600 real-GT pairs. Its task-term change is

```text
D_tq = 100 (d_seg,tq - d_seg,0)
       + sqrt(10 d_pose,tq) - sqrt(10 d_pose,0).
```

The #157 solver convexifies each tensor curve and admits a marginal coarsening segment while

```text
Delta D / Delta B_saved < 25 / 37,545,489.
```

This allocation is **DERIVED** from MEASURED single-tensor rows. Brotli bytes and scorer effects are
non-additive, so the prediction is not evidence; only the joint n600 receiver replay is the
advisory allocation verdict.

## Complete measurement surface

The durable resume state contains **146 pre-allocation MEASURED units**: two independent baseline
replays plus 144 tensor perturbations (18 tensors times int7..int2, zero, and mean). The response
artifact exposes 162 rows because the shared int8 baseline is represented once per tensor. All
18/18 tensors have the complete nine-rung int8..int2/zero/mean response curve.

| Quantity | Baseline | KKT proposal | Delta vs baseline | Classification |
|---|---:|---:|---:|---|
| archive bytes | 63,664 | 36,975 | -26,689 | **MEASURED**, actual ZIP |
| `d_seg` | 0.03365824381510417 | 0.11998607211642796 | +0.08632782830132379 | **MEASURED n600** |
| `d_pose` | 151.79642088984443 | 152.6197926660713 | +0.82337177622687 | **MEASURED n600** |
| `Delta S` | 0 | +8.72053480699927 | +8.72053480699927 | **MEASURED n600 advisory** |

The baseline repeat had exactly zero archive, `d_seg`, and `d_pose` deltas. The response
classification found 97 Pareto-dominated rows, 43 zero-invariant rows at the exact repeat floor,
and 78 rows passing the separate historical P-sufficiency tolerance; these labels do not override
the combined replay.

## Reverse-waterfill result

The separable solver returned `kkt_holds=true` with 10/18 tensors coarsened:

| Tensor | Proposed bits | Tensor | Proposed bits |
|---|---:|---|---:|
| `code` | 3 | `film.bias` | 4 |
| `film.weight` | 8 | `hidden.0.bias` | 3 |
| `hidden.0.weight` | 4 | `hidden.1.bias` | 8 |
| `hidden.1.weight` | 4 | `hidden.2.bias` | 8 |
| `hidden.2.weight` | 6 | `hidden.3.bias` | 8 |
| `hidden.3.weight` | 8 | `in_proj.bias` | 8 |
| `in_proj.weight` | 5 | `out_sdf.bias` | 8 |
| `out_sdf.weight` | 4 | `out_tex.bias` | 5 |
| `out_tex.weight` | 3 | `palette` | 8 |

- **DERIVED separable prediction:** 26,247 bytes saved, distortion-term change
  `-2.7729930961813825`, and total `Delta S=-2.79046989612398`.
- **MEASURED joint result:** 26,689 bytes saved but `Delta S=+8.72053480699927`, driven by the
  non-additive joint SegNet failure.
- **Admissible allocation verdict:** reject the KKT proposal and retain all tensors at int8 for this
  frozen checkpoint/formulation. This is a scoped failure of independent-response composition, not
  a negative verdict on measured bit allocation as a family.

The receiver-closed allocation archive is SHA-256
`eb91719e01dece09340302a348e44c91f8aa56463870b922f40c952a5b0679be`; NumPy-oracle versus actual
inflate deltas are exactly zero for both distortions.

## Composition with cross-tensor codec #461

The allocation **does change #461's input**: #461 operates on the emitted quantized grid, and the
mixed 3..8-bit assignment is not the historical all-int8 grid. The historical `-417 B` therefore
does not transfer.

- **MEASURED rate:** the selected joint chart (base permutation indices `[0,8,10,14]`, raw-int8 code
  transform `c=0`) changed 36,975 bytes to 36,898 bytes: **-77 bytes**.
- **MEASURED lossless state check:** identity and joint decoded quantized-state SHA-256 are both
  `f87b2ea27a24654ee131d3222e81772111d9ba69958ad15a00a14b70e6cce363`, with zero base and code
  symbol mismatches.
- **DERIVED distortion reuse:** exact full decoded-state equality preserves
  `d_seg=0.11998607211642796` and `d_pose=152.6197926660713`; composed
  `Delta S=+8.720483535859879`, still **REJECT**.
- **Fail-closed diagnostic:** the landed #461 measurement wrapper exited 1 only because its optional
  pair-only chart forced code delta `c=1` although this grid selected raw `c=0`. The actual selected
  joint chart is unaffected and independently lossless-checked. The sibling-sealed #461 tool was not
  edited.

## Resumability, postprocessing lineage, and disk hygiene

- `resume_state.json` checkpointed every fixed scorer batch and binds checkpoint SHA, GT SHA, n600,
  precision grid, scorer/worker axes, and original producer/allocator source hashes.
- A completed-surface postprocessor fixed only the `bits=null` sort for zero/mean rows and added a
  guarded finalization path. Before allocation it re-derived all 146 candidate archive/blob custody
  records byte-for-byte. `postprocess_lineage.json` preserves both original measurement and fixed
  postprocess fingerprints; no measurement row was relabeled or rerun under changed source.
- Each candidate raw was atomic, scored in fixed batches, SHA-certified in a machine-readable
  cleanup manifest, and deleted success-only. No operator-facing conclusion cites temporary storage.

## Triality

- DSL: **N/A-with-reason** — offline frozen-payload measurement; no trainer/curriculum knob added.
- Canonical equation: `witness_measured_reverse_waterfill_v1`, registered append-only from
  `src/tac/canonical_equations/witness_measured_reverse_waterfill_20260713.py`.
- DAG FEED: `.omx/research/sub015_DAG_witness_sensitivity_bitalloc_336_20260713.md`.

## Durable outputs

- `section_precision_response_curves.json` — complete 18/18 n600 response matrix.
- `allocated_bit_budget.json` — KKT assignment and separable prediction.
- `allocated_archive.zip`, `allocated_witness.npz`, and `allocated_artifact_custody.json` — joint
  proposed allocation.
- `byte_closed_advisory_row.json` — receiver-closed measured rejection.
- `crosstensor_composed/crosstensor_compose_receipt_336.json` — #461 exact-state composition receipt.
- `postprocess_lineage.json` — original-measurement to fixed-finalizer custody.

## Pointer-delta honesty and caveats

- This is witness-LINE rate evidence, not the `0.19108` frontier pointer.
- It is `[macOS-CPU advisory; NumPy-fp32 authority]`, NON-PROMOTABLE, `score_claim=false`.
- No exact contest-CPU/CUDA evaluation was dispatched; `upstream/evaluate.py` remains operator-GO.
- The canonical frontier pointer is unchanged. The measured negative verdict is scoped to this
  checkpoint and independent single-tensor composition formulation.

## STORES CONSULTED

`CLAUDE.md`; `AGENTS.md`; `docs/operating_manual_craft_handoff.md`; v7.5/v8 canonical specs;
`reports/latest.md`; `.omx/state/lane_registry.json`; `.omx/state/subagent_progress.jsonl`;
the #153/#157/#406/#461 memos and source surfaces; the JRD PR110 completion memo and response
artifact; the #336 resume/custody/result artifacts named above.
