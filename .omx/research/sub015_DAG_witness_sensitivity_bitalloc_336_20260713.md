# FEED-witness-bitalloc-336-20260713 — byte-closed witness sensitivity allocation DAG

`research_only=true` · `score_claim=false` · `pointer_moved=false` · `$0 LOCAL` · `n600`

## Executed dependency graph

```text
#406 canonical first-good witness checkpoint (frozen SHA custody)
  └─> exact LVLS1 per-candidate inflate (NumPy receiver, real GT n600)
       ├─> COMPLETE: 18/18 int8..int2 response curves
       ├─> COMPLETE: #153 zero/mean rows + exact-repeat zero-invariance gate
       └─> COMPLETE: lower convex hull per tensor
            └─> #157 KKT reverse-waterfill at byte value 25/37,545,489
                 └─> DERIVED proposal: mixed 3..8-bit grid; predicted Delta S=-2.79046990
                      └─> MEASURED joint archive at n600
                           ├─> 36,975 B; d_seg=0.1199860721; d_pose=152.6197926661
                           ├─> Delta S=+8.7205348070 => REJECT INSTANCE/FORMULATION
                           └─> #461 lossless chart on changed grid
                                ├─> 36,898 B (-77 B, not historical -417 B)
                                ├─> exact decoded quantized-state equality
                                └─> Delta S=+8.7204835359 => STILL REJECT
```

The admissible output for this instance is the original all-int8 baseline: 63,664 bytes,
`d_seg=0.03365824381510417`, `d_pose=151.79642088984443`. Exact contest evaluation is neither
recommended by this advisory result nor authorized without operator GO.

## Triality

- DSL: **N/A-with-reason**. This is an offline frozen-payload codec measurement and adds no
  trainer/curriculum knob. Probe controls are argparse-validated; adding a training DSL lever would
  create a stray configuration surface.
- Canonical equation: `witness_measured_reverse_waterfill_v1` in
  `tac.canonical_equations.witness_measured_reverse_waterfill_20260713`.
- DAG: this FEED. The shared main DAG was sibling-dirty, so this isolated append-only FEED preserves
  executable dependency edges without serializer collision.

## Final canonical task rows

| Node | State | Producer | Consumer | Hard gate / verdict scope |
|---|---|---|---|---|
| `witness_336_response_matrix_n600` | `COMPLETE_MEASURED` | `tools/probe_witness_sensitivity_bitalloc.py` | KKT allocator | 18/18 tensors; 146 measured pre-allocation units; all rows n600 and actual archive bytes. |
| `witness_336_dominated_rungs` | `COMPLETE_MEASURED` | #153 classifier | allocation memo | 97 Pareto-dominated, 43 exact-repeat zero-invariant, 78 historical-P-sufficiency rows; labels stay distinct. |
| `witness_336_kkt_allocation` | `COMPLETE_DERIVED_REJECTED_BY_COMPOSITION` | existing #157 solver | combined replay | KKT holds on separable curves; joint non-additivity falsifies its favorable prediction. |
| `witness_336_combined_byte_close` | `COMPLETE_MEASURED_REJECT` | shipped LVLS1 runtime | operator decision | 36,975 B, `d_seg=0.1199860721`, `d_pose=152.6197926661`, `Delta S=+8.7205348070`. |
| `witness_336_crosstensor_461_compose` | `COMPLETE_LOSSLESS_REJECT` | #461 codec + #336 receipt | operator decision | Changed grid saves 77 B; exact state equality preserves rejected distortion. |
| `witness_336_promotion` | `NOT_AUTHORIZED_NOT_RECOMMENDED` | `upstream/evaluate.py` | frontier pointer | Advisory result is worse; exact CPU/CUDA requires operator GO. |

## KKT-proposed assignment

```text
code=3             film.bias=4       film.weight=8
hidden.0.bias=3    hidden.0.weight=4 hidden.1.bias=8
hidden.1.weight=4  hidden.2.bias=8   hidden.2.weight=6
hidden.3.bias=8    hidden.3.weight=8 in_proj.bias=8
in_proj.weight=5   out_sdf.bias=8    out_sdf.weight=4
out_tex.bias=5     out_tex.weight=3  palette=8
```

This assignment is solver output, not the admissible final assignment. The final advisory
recommendation retains all tensors at int8.

## Custody and resumability

- Result root: `experiments/results/witness_sensitivity_bitalloc_336_20260713T042157Z/`.
- Each candidate raw was atomic, scored in fixed batches, checkpointed in `resume_state.json`,
  hashed, certified in a cleanup manifest, and deleted success-only.
- The resume fingerprint binds checkpoint SHA, GT SHA, n600, precision grid, scorer axis, worker
  geometry, and original producer/allocator source hashes.
- `postprocess_lineage.json` records the guarded complete-surface finalization and byte-for-byte
  custody re-derivation after the null-bit classifier fix.
- `crosstensor_composed/crosstensor_compose_receipt_336.json` records the selected #461 chart,
  exact decoded-state equality, and the optional pair-only diagnostic failure without editing #461.

## Pointer-delta honesty and verdict scope

The canonical frontier pointer is unchanged. This is witness-LINE rate evidence, not the `0.19108`
pointer; `[macOS-CPU advisory; NumPy-fp32 authority]` is NON-PROMOTABLE and `score_claim=false`.
No MPS score or contest exact evaluation was used. The negative verdict applies only to the frozen
V9 ep150 checkpoint plus independent single-tensor response composition; it does not kill jointly
measured allocation, entropy co-design, or retrained witness families.
