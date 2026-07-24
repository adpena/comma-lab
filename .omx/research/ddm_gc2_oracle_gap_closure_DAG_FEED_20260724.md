# DDM GC2 scorer-value oracle gap closure: triality and feed

Captured: 2026-07-24T21:45:59Z  
`research_only=true` · `execution_allowed=false` · `score_claim=false` ·
`main_landing_review_required=true`

## DSL / code leg

No launch flag, typed configuration, runtime, archive, or frontier pointer
changed. `src/tac/scorer_value_oracle.py` now binds the seven formerly missing
DDM-366 rows to immutable JSON producers. Every read rechecks bytes, SHA-256,
schema, and selector before exposing a value. The seven public accessors are:

- `normalization_affine`
- `r_frequency_passband`
- `yuv6_luma_phases`
- `chroma_pose_null`
- `null_gauge_energy`
- `pose_excluded_dimensions`
- `score_functional`

`DEFAULT_GAPS` is empty only because every row has a sealed producer. It does
not widen any producer's `authority_scope`.

## DAG leg

```text
upstream/modules.py + posenet.safetensors
                    -> normalization_affine.json

exact bicubic-up / bilinear-down matrices + #580 + scoped 3.2x receipt
                    -> frequency_r_passband.json

upstream/frame_utils.py + upstream/modules.py
                    -> yuv6_luma_phases.json
                    -> chroma_pose_null.json

#580 ker(A) + #519 n32 advisory gauge/null measurements
                    -> null_gauge_energy.json

upstream/modules.py  -> pose_dims_exclusion.json

upstream/evaluate.py + tac.contest_score + upstream/modules.py
                    -> score_functional.json

seven producer bytes
                    -> exact SHA/schema/selector bindings
                    -> ScorerValueOracle FAIL_CLOSED read
                    -> 21 WRAPPED / 0 TYPED-GAP / 0 stale
                    -> DDM costate source custody
```

The chroma edge deliberately stops before camera-coordinate consumers:

```text
post-resize 2x2 RGB kernel + scoped scorer-grid uint8 readback
                    -X-> j8f / pa2 / F2 camera authority
                         (camera preimage and receiver closure are NULL)
```

That refusal prevents a post-resize preprocessing kernel from being promoted
into a camera-grid or receiver-closed actuator certificate.

## Equation leg

The exact composite resize operator is the matrix composition

```text
R = D_bilinear(384x512 <- 874x1164) o U_bicubic(874x1164 <- 384x512).
```

Its deterministic Fourier-mode gains are evaluated from the finite matrices,
not from a shift-invariant approximation. The sampled Nyquist horizontal versus
vertical ratio is `1.00055394`; the measured `3.125` (~3.2x) along-tangent
representation deficit is therefore retained as a separate scoped measurement,
not mislabeled as resize attenuation.

For one clamp-inactive post-resize `2x2` RGB block, Pose preprocessing has the
six-dimensional kernel

```text
Y_i = 0 for i in {00,10,01,11},
sum_i R_i = 0,
sum_i B_i = 0.
```

One exact integer RGB primitive is `(-15, 9, -7)` because

```text
299(-15) + 587(9) + 114(-7) = 0.
```

It is RGB-input-visible to SegNet. No SegNet argmax change, camera preimage, or
receiver-closed realization is inferred.

Pose distortion excludes outputs 7-12 structurally:

```text
d_pose = mean((pose_src[..., :6] - pose_dst[..., :6])^2).
```

The exact score functional remains

```text
S = 100 d_seg + sqrt(10 d_pose) + 25 archive_bytes / 37,545,489.
```

Only `upstream/evaluate.py` on exact archive bytes and the required contest
hardware axes can issue an authoritative score.

## Canonical feed delta

- DDM-366 scorer-value coverage moves from historical `14/21` to `21/21`.
- The seven new rows become hash-fresh, typed producer inputs.
- j8f/pa2/F2 learn a strict camera-preimage blocker, not a chroma actuator.
- The null/gauge row preserves two different measurements:
  `0.52356` is a gauge fraction of norm (`0.27412` of energy), while
  `0.52425` is separately measured rendered output energy in `ker(A)`.
  Their cross-space intersection remains NULL.
- Pointer delta is zero; no dispatch, replay, score, or promotion occurred.

STORES CONSULTED: delegated authority; CLAUDE.md; AGENTS.md; operating manual;
DDM-366 contract; OF1 facade memo; upstream evaluate/modules/frame_utils;
PoseNet safetensors; canonical contest-score helper; exact R-chain matrices;
#580 receipt; #519 receipt and memo; measured 3.2x source receipt and scoped
follow-up; lane registry; subagent checkpoint ledger; both directive inboxes.
