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

**[DERIVED, exact in real arithmetic before serialization]** For centered head rows `a_c` and biases
`beta_c`, choose a deterministic orthonormal basis `Q` for their row-difference space and put
`z=Q^T f`, `A_c=a_c Q`. Define weighted sites

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

**[DERIVED precision boundary / NO-FAKE]** The identity above is algebraically exact over the reals. The
counted `PDW1` target deliberately stores float32 sites, weights, and tie loci, so finite-precision
quantization can change an arbitrarily close tie. Byte-close parse-back is exact; universal boundary-label
parity is not. The implementation must report sampled float32 parity and the maximum sampled pair-score
error, and retain `NO_GENERAL_VERDICT_WITHIN_F32_TIE_UNCERTAINTY` for unseen or constructed near ties.

**[DERIVED correction / NO-FAKE]** The sites describe the frozen head's cells in its rank-4 channel
quotient. They do **not** describe the video-specific spatial partition without the nonlinear feature-field
pullback `F_theta(x)`. Sites plus a region-adjacency graph plus full channel tie hyperplanes do not locate a
single spatial curve. They are therefore a byte-close **channel-target certificate**, not a standalone
partition codec and not a closed-form inverse through the convolutional renderer.

The cached `gt_n600.npz` has `lstars`, margins, source frames, and poses, but no penultimate `16 x 3 x 3`
feature patches. Labels-only fitting in actual SegNet channel space is underdetermined and must fail closed.
Video-fed initialization may legitimately use `L*` to identify active classes, observed adjacencies, counts,
and a partition digest while placing transformed sites from the frozen head. A true cells-to-generators fit
requires paired channel features and reports its residual; it may not manufacture a canonical simplex lift
and call that lift the scorer inverse. The counted target contains transformed head coefficients and aggregate
cached-`L*` adjacency, but no full scorer, scorer tensor, or per-pixel GT table. This remains research-only
pending contest-compliance review.

## Frozen implementation contract

The owned module is `src/tac/boundary_math/power_diagram_witness.py`; tests live beside boundary-math tests.

1. **Real-arithmetic forward plus measured float32 parity.** Implement classical weighted-site power
   distance, deterministic row-difference quotient construction, affine-head-to-sites conversion including
   bias and gauge, pair tie-loci, and the co-maximum test. Ordinary sampled features must emit a numerical
   parity receipt; boundary fixtures must expose the float32 tie-uncertainty limit rather than claim exactness.
2. **Well-posed inverse fit.** Given paired channel features and target labels, fit a unique strictly
   regularized multiclass affine target, convert it to weighted sites, and return sample agreement, true-label
   margin, objective, and `exact_on_samples`. This is a fit, not a theorem that every spatial partition is one
   global power diagram. Missing feature samples is a hard error.
3. **Video-fed cached-GT initialization.** Memory-map the `ZIP_STORED` `lstars.npy` member without materializing
   the 4.7 GB cache. Read the frozen head tensors without importing the scorer. Derive active classes,
   class counts, observed 4-neighbour adjacency, selected-partition SHA-256, head SHA-256, and the transformed
   target. No full scorer, scorer tensor, or per-pixel GT table enters the encoded blob.
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
| frozen-head rank and real-arithmetic identity | derived exact before float32 serialization |
| sampled float32 affine/power parity | measured, with near-tie uncertainty explicit |
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

### Cached custody and target bytes

- **[MEASURED]** Read-only cache
  `/Users/adpena/Projects/pact/experiments/results/mlx_fleet_gt_cache/gt_n600.npz` is `5,078,017,610`
  bytes, SHA-256 `cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6`.
  `lstars.npy` was memory-mapped read-only as `(600,384,512) <i8`; its selected-array digest is
  `bf1d0e5c7e2ef1b3c38ce6cd51ec827169ac13a02c675638b7bd97344a089ec4`.
- **[MEASURED]** Across `117,964,800` cached labels, class counts in canonical order are
  `(27,407,046, 690,639, 58,413,281, 1,460,325, 29,993,509)`. All five classes are active. The observed
  four-neighbour class edges are `(0,1) (0,2) (0,3) (0,4) (1,2) (1,3) (1,4) (2,3) (3,4)`; only `(2,4)`
  is absent.
- **[MEASURED]** Frozen head file SHA-256 is
  `68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6`; weight/bias shapes are
  `(5,16,3,3)` / `(5,)`. The deterministic centered quotient rank is `4`; singular values are
  `3.1283763256, 2.1542713873, 2.0247078699, 1.7962638357, 3.7304e-16`.
- **[MEASURED, byte-close target]** The n600-adjacency `PDW1` is exactly `338` bytes, SHA-256
  `84a49d802dc5bd9c416013fd71bc6f08655a2f3c23c249374469a4dc4d8ee275`; strict
  encode -> fresh decode -> encode is byte-identical. Decoder tests reject malformed counts, non-finites,
  negative zero, lossy IDs/dtypes, inconsistent or noncanonical edges/ties, truncation, and every trailing byte.

### Numerical parity and description length

- **[MEASURED synthetic channel probe, not real penultimate-feature custody]** With seed `17`, `200,000`
  ordinary Gaussian 144-D feature samples had `0` affine/power label mismatches after float32 target
  serialization. Maximum sampled class-pair score error was `4.949247811580904e-07`; minimum sampled affine
  winner margin was `2.92061269768773e-06`; sampled tie-uncertain count was `0`. Verdict remains
  `NO_GENERAL_VERDICT_WITHIN_F32_TIE_UNCERTAINTY`: this finite sample is not a boundary theorem.
- **[MEASURED comparator custody]** Coordinate-witness checkpoint
  `/Users/adpena/Projects/pact/experiments/results/banks/v9c2_defensive_bank_20260718/levelset_witness_ema_BEST.npz`
  is `460,448` bytes, SHA-256
  `b0a431e9259cd3c54ae53b677076823f36e096b27eb0d9ba74ed7c54c9113cef`. Packed realization
  `/Users/adpena/Projects/pact/experiments/results/witness_crosstensor_structure_rate_20260713/identity_packet/archive.zip`
  is `63,659` bytes, SHA-256
  `1056a39427133ee3d160f3612455f191d32496f8039ab41188c52896465c8de1`.
- **[DERIVED verdict scope]** `338` target bytes versus either realization file is
  `TARGET_ONLY_VS_REALIZATION_NON_EQUIVALENT`. It is not an archive saving, score gain, or spatial
  Kolmogorov bound. The target omits the feature-field/render inverse that those realization bytes must carry.

### Verification and disposition

- **[MEASURED local CPU]** `56` targeted tests passed with warnings-as-errors, including the adjacent
  Laguerre-logit regression suite; Ruff check, Ruff format check, and `git diff --check` are clean.
- **[CONFIRMED within verdict scope]** The module supplies a strict byte-close **channel-target certificate**,
  a well-posed strictly regularized inverse only when paired channel features are provided, and a labels-only
  hard refusal. It does **not** establish the video pullback, renderer/preimage, through-R survival, Seg/Pose
  score, archive compliance, or a spatial generator count.
- `spatial K lower bound = NO_VERDICT`; `renderer / through-R / Seg / Pose / score = NOT RUN, NO AUTHORITY`.
  Pointer `0.1910828242 [contest-CPU Linux x86_64]` remains **UNMOVED**. MAIN landing review is required before
  any shared-DAG or canonical-equation registration.
