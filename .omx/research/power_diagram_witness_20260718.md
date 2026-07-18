# Power-diagram witness: channel-target build contract and $0 proof plan

`research_only=true` · task `#539` · lane `power_diagram_witness_20260718` · `$0` · `NO LAUNCH` · `NO SCORE CLAIM`

Pointer `0.1910828242 [contest-CPU Linux x86_64]` is **UNMOVED**. Sacred c2 and every live run are read-only.
This artifact specifies an offline target representation; it does not authorize a render, trainer, provider,
GPU, evaluator, archive mutation, or pointer change.

## Stores consulted

- `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`, and `docs/operating_manual_craft_handoff.md`.
- `SPEC_v75_optimal_single_trunk_20260708.md` section 8 and
  `SPEC_v8_perclass_decomposition_20260708.md`.
- `SPEC_v10_capstone_cold_start_seeded_20260717.md` sections 14.7 and 14.11 from its canonical branch.
- `campaign-is-a-constrained-mdl-inverse-solve`,
  `solve-the-right-problem-is-right-coordinates-at-every-level-kolmogorov`, and
  `intrinsic-complexity-design-philosophy` project memories.
- `v8_laguerre_generator_feasibility_and_perclass_hybrid_20260710.md`,
  `necessity_solver_inverse_factorization_20260715.md`,
  `collateral_coupling_geometry_and_film_flicker_sidecar_20260718.md`, and
  `campaign_meta_adversarial_review_v9c2_to_v10_20260718.md`.
- `src/tac/boundary_math/partition_collapse.py`, `laguerre_logit_offset.py`, and the frozen
  `upstream/modules.py` / `models/segnet.safetensors` head surface.
- Canonical frontier, lane, checkpoint, equation, and probe ledgers. The new lane was registered at L0,
  phase 1, `research_only=true`. Registry-wide validation remains red on 110 older missing-evidence paths;
  the new row itself is structurally valid.

## Premise boundary: exact nucleus versus unproved pullback

**[MEASURED]** The frozen segmentation head is an affine `5 x 144` map: the safetensors keys are
`segmentation_head.0.weight` with shape `(5,16,3,3)` and `segmentation_head.0.bias` with shape `(5,)`.
Its centered row space has rank 4 in settled custody.

**[DERIVED, exact]** For centered head rows `a_c` and biases `beta_c`, choose a deterministic orthonormal
basis `Q` for their row-difference space and put `z=Q^T f`, `A_c=a_c Q`. Define weighted sites

```
s_c = A_c / 2
omega_c = beta_c + ||s_c||^2 - common_gauge.
```

Then

```
argmax_c (A_c z + beta_c)
  = argmax_c (2 s_c^T z + omega_c - ||s_c||^2)
  = argmin_c (||z-s_c||^2 - omega_c).
```

The class-pair tie hyperplane for canonical orientation `i<j` is

```
2(s_j-s_i)^T z + ||s_i||^2 - ||s_j||^2 - omega_i + omega_j = 0.
```

It is an **active scored facet only under the co-maximum clause**: classes `i,j` must tie and dominate
every other class. A formal equality between two losing logits is not a scored boundary.

**[DERIVED correction / NO-FAKE]** The sites describe the frozen head's cells in its rank-4 channel
quotient. They do **not** describe the video-specific spatial partition without the nonlinear feature-field
pullback `F_theta(x)`. Sites plus a region-adjacency graph plus full channel tie hyperplanes do not locate a
single spatial curve. They are therefore a byte-close **channel-target certificate**, not a standalone
partition codec and not a closed-form inverse through the convolutional renderer.

The cached `gt_n600.npz` has `lstars`, margins, source frames, and poses, but no penultimate `16 x 3 x 3`
feature patches. Labels-only fitting in actual SegNet channel space is underdetermined and must fail closed.
Video-fed initialization may legitimately use `L*` to identify active classes, observed adjacencies, counts,
and a partition digest while placing the exact sites from the frozen head. A true cells-to-generators fit
requires paired channel features and reports its residual; it may not manufacture a canonical simplex lift
and call that lift the scorer inverse.

## Frozen implementation contract

The owned module is `src/tac/boundary_math/power_diagram_witness.py`; tests live beside boundary-math tests.

1. **Exact forward and head conversion.** Implement classical weighted-site power distance, deterministic
   row-difference quotient construction, affine-head-to-sites conversion including bias and gauge, pair
   tie-loci, and the co-maximum test. Random and boundary fixtures must reproduce affine-head argmax exactly.
2. **Well-posed inverse fit.** Given paired channel features and target labels, fit a unique strictly
   regularized multiclass affine target, convert it to weighted sites, and return sample agreement, true-label
   margin, objective, and `exact_on_samples`. This is a fit, not a theorem that every spatial partition is one
   global power diagram. Missing feature samples is a hard error.
3. **Video-fed cached-GT initialization.** Memory-map the `ZIP_STORED` `lstars.npy` member without materializing
   the 4.7 GB cache. Read the frozen head tensors without importing the scorer. Derive active classes,
   class counts, observed 4-neighbour adjacency, selected-partition SHA-256, head SHA-256, and the exact target.
   No scorer or source-derived table enters the encoded blob.
4. **Strict byte-close.** Define one deterministic little-endian `PDW1` payload with canonical class/site order,
   float32 sites/weights, sorted unique adjacency, and explicit float32 tie normals/offsets. Decode must reject
   malformed counts, non-finite values, noncanonical edges, inconsistent tie-loci, truncation, and every
   trailing byte. Encode -> fresh decode -> encode must be byte-identical.
5. **Description-length receipt.** Compare exact target blob bytes with explicitly named local checkpoint/blob
   files, with hashes. The comparison must say `TARGET_ONLY_VS_REALIZATION_NON_EQUIVALENT`; no ratio is a score
   gain, archive saving, or spatial Kolmogorov bound.

## $0 cached-GT proof matrix

The proof will read, never mutate:

- `/Users/adpena/Projects/pact/experiments/results/mlx_fleet_gt_cache/gt_n600.npz`;
- `/Users/adpena/Projects/pact/upstream/models/segnet.safetensors`;
- the defensive-bank coordinate-witness checkpoint and, where present, its packed carrier/archive.

Required receipt rows:

| row | required verdict |
|---|---|
| frozen-head rank and affine/power pair-margin parity | exact or fail |
| all-n600 label-cache class counts, adjacency, and digest | measured advisory |
| `PDW1` byte count and encode/decode/re-encode identity | measured byte-close target |
| coordinate-INR checkpoint/blob file bytes and SHA-256 | measured comparator custody |
| spatial K lower bound | `NO_VERDICT` until a legal spatial pullback/feature-field generator is encoded |
| renderer / through-R / Seg / Pose / score | `NOT RUN`, `NO AUTHORITY` |

The settled global **image-space** Laguerre result remains intact: a few spatial generators saturate above
d_seg-relevant fidelity, while heterogeneous per-class carriers remain open. This lane neither reruns nor
relabels that negative; it builds the distinct frozen-head **channel-space target**.

## Triality disposition

- **DAG FEED candidate `FEED-power-target`:** `cached L* + frozen affine head -> rank-4 quotient -> weighted
  sites -> active adjacency/tie certificate -> conv-realization residual`. This memo is the isolated FEED;
  MAIN must review before folding it into any shared DAG.
- **Canonical-equation candidate `affine_head_power_diagram_generator_duality_v1`:** the head/sites identity
  and co-maximum condition above. Candidate only; do not register until implementation parity and byte-close
  receipts land.
- **DSL leg:** N/A with rationale. This is an advisory target/codec primitive, not a trainer or launcher lever.
  Any future realization consumer must enter through the typed witness DSL and the full resume registry.

## Results and verdict

Implementation and measurements are pending in this build-contract revision. No result should be inferred from
the existence of the specification.

