# Boundary inverse custody — finite genuine-frame solve and n600 measurement

- Captured: 2026-07-21 UTC
- Lane: `boundary_inverse_custody_20260721`
- Axis: **[macOS-CPU advisory]**
- Score claim: **false**
- Promotion eligible: **false**
- Pointer delta: **none**; `0.19108 [contest-CPU]` is unchanged
- Verdict: **GENUINE_RESIDUAL_INVERSE_MEASURED_MASK_ONLY_RATE_ADOPTION_PENDING**
- Verdict scope: the real n600 decoded 41,303-byte coherent Lane chart, this finite mixed-frame
  dictionary, shared-coefficient generic arm, and eight-bin dash-arc-phase arm. Mask fidelity
  only; no through-R `d_seg`, PoseNet, score, archive, or family-wide negative.

## Outcome first

`BLOCKED_TARGET_BOUNDARY_INVERSE_CUSTODY` is closed at the representation-fidelity layer. The
true Lane-mask XOR residual is now mapped by a deterministic finite solve into selected atoms
from the genuine #502 literal-polar-curvelet and compact-shearlet implementations. Learned atom
selection and signed-int8 coefficients are carried by a strict #557 context-arithmetic sidecar;
every measured sidecar parses back exactly and re-encodes byte-identically.

The best measured composition is `generic_2d_k4`:

| row | chart B | sidecar B | total B | curvelet/shearlet nonzeros | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| settled decoded dash chart | 41,303 | 0 | 41,303 | 0/0 | 0.606196 | 0.543964 | 0.573397 |
| finite generic inverse k4 | 41,303 | **586** | **41,889** | **1/3** | **0.613652** | **0.546404** | **0.578079** |

This is a measured mask-F1 gain of `+0.0046822634384159`, not a score gain. The sidecar's exact
rate price is `0.00039019334652959245 S`; because realized through-R recovery is absent,
`realization_breakeven_bytes_v1` remains **FORMALIZATION_PENDING** rather than being fed mask F1.

The preregistered threshold-0.25 curve was genuinely negative. A bounded threshold-only
disambiguator found that threshold 0.5 separates the positive finite arm from over-actuation;
the full n600 prefix curve was then rerun from source at 0.5. The earlier receipt is retained as
the control, not rewritten.

## D1 — deterministic finite inverse and exact coefficient custody

The dictionary has 296 scalar columns:

- 80 immutable `literal_polar_curvelet` columns, atom-spec SHA-256
  `48df53b84660396adc522fe966cb8e7c631c108332a3529eefe17ee9aaa44f6e`;
- 216 real/imaginary compact-shearlet columns from a fixed two-scale, two-cone, three-shear,
  3x3-translation generic frame; config SHA-256
  `3a44a466f5209560795f6c854a6337997802178a35bfb1ebc12c6077e46e34cb`.

The solver performs deterministic normalized-correlation screening, forces the best genuine atom
from each family into every nonempty prefix, refits each finite prefix with ridge-stable least
squares, and quantizes at `1/64` to signed int8. Explicit NumPy reductions replace the host BLAS
matrix path; a finite-value guard refuses any non-finite feature, coefficient, or rendered sum.

The final threshold-0.5 curve is:

| treatment | sidecar B | total B | nonzero atoms | curvelet/shearlet | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| generic k2 | 583 | 41,886 | 2 | 1/1 | 0.615488 | 0.543964 | 0.577520 |
| generic k4 | 586 | 41,889 | 4 | 1/3 | 0.613652 | 0.546404 | **0.578079** |
| generic k8 | 602 | 41,905 | 8 | 1/7 | 0.576911 | 0.570023 | 0.573446 |
| generic k16 | 777 | 42,080 | 15 | 1/14 | 0.591379 | 0.562857 | 0.576765 |
| phase k2 | 871 | 42,174 | 16 | 8/8 | 0.364891 | 0.493198 | 0.419452 |
| phase k4 | 1,199 | 42,502 | 32 | 22/10 | 0.277794 | 0.536994 | 0.366166 |
| phase k8 | 1,485 | 42,788 | 62 | 32/30 | 0.286827 | 0.522759 | 0.370415 |
| phase k16 | 2,366 | 43,669 | 123 | 34/89 | 0.290237 | 0.495616 | 0.366089 |

Best-sidecar SHA-256:
`7017c7bd131f4a4d63f0180812492f748d019d391c2747ba12a40b4c6ab6eab1`.

## D2 — real dash-arc-phase treatment, scoped negative

The phase arm is not a generic 2-D control with a phase label. For every decoded dashed line it:

1. evaluates the decoded polynomial centerline;
2. integrates centerline arc length by fixed eight-node Gauss-Legendre quadrature;
3. anchors phase at the decoded `dash_phase_m` and reduces modulo decoded `dash_period_m`;
4. selects one of eight counted coefficient rows; and
5. evaluates genuine atoms at `(normalized cross-line coordinate, local phase)`.

That treatment is structurally real and decodes without truth, but this formulation loses to the
generic control: best phase F1 `0.419452` versus generic `0.578079` (delta `-0.158627`). At the
threshold-only probe, phase k2 remains below baseline at 0.5, 0.75, and 1.0; 1.5 is a no-op.
Verdict scope is this shared eight-bin qint arm and its current coordinate/solver, not all
phase-conditioned curvelet/shearlet inverses.

## D3 — composed bytes, break-even, and eat-the-flip remainder

The selected 41,889-byte composition changes 24,310 pixels:

- 16,232 beneficial changes;
- 8,078 harmful changes;
- 313,271 false negatives remain;
- 237,586 false positives remain.

The exact stratum remainder explains the win and its remaining debt:

| stratum | beneficial | harmful | remaining FN | remaining FP |
|---|---:|---:|---:|---:|
| Lane boundary | 715 | 0 | 246,092 | 0 |
| Lane interior | 970 | 0 | 67,179 | 0 |
| Road negative | 0 | 4,565 | 0 | 165,840 |
| other negative | 14,547 | 3,513 | 0 | 71,746 |
| upper half | 0 | 0 | 22,975 | 32,730 |
| lower half | 16,232 | 8,078 | 290,296 | 204,856 |

The finite arm mostly removes non-Road false positives and adds a small number of genuine Lane
pixels, while still paying 4,565 Road false positives. It is therefore a real but incomplete mask
closure, not a blanket fix.

`realization_breakeven_bytes_v1` round-trips the required recovery exactly:
`breakeven_bytes(586 * 25 / 37,545,489) = 586`. The law cannot decide adoption because its input
must be realized hard-oracle score recovery. Mask F1 is deliberately not substituted. No new
canonical law is registered; the missing integration is a #597 receiver parse-back with exact
`d_seg`, `d_pose`, and wall-clock, followed by the #596/#597 joint waterfill.

The post-run M1 decomposition directive was consumed from
`rep_mine_solved_binary_20260721T045500Z.json` (SHA-256
`265302908fd7c4789891ab0d3b0f8aacaf9f178ea8e40f8737ed5f4fcd55b368`). Its 222,447-byte ideal
constraint price versus the 216,222-byte box does not authorize adding this 586-byte row: the M1
price excludes the exact receiver/pose closure this arm also lacks. This sidecar is instead a
small component candidate for the binding parse-back/joint-marginal next node.

The later 05:34/05:37 UTC operator sell-back directives sharpen that next node: a realized
SegNet flip-pixel has value `100 / (600*512*384) = 8.48e-7 S`, or about `1.27 B/flip-pixel` at
the current water level, and the solved-event carrier must be iteratively re-priced under the
#557 context model until its kept/eaten set is a stable fixed point. Those numbers do **not**
authorize pricing this row's 24,310 Lane-mask changes. They are source-label pixels, not G3
through-R scorer flip identities, and a BIC1 coefficient is a spatially overlapping atom rather
than one independently deletable scorer-flip event. The #597 receiver must therefore emit the
exact G3 identity set plus per-coefficient removal/quantization influence, re-code each proposed
tranche, and apply the individual-flip fixed-point rule before #596 can keep any part of this
sidecar. No mask pixel is assigned the `8.48e-7 S` value in this receipt.

## D4 — rule-118 split

COUNTED state is exactly the coordinate-mode/phase-row header, qstep, correction threshold, and
dense int8 coefficient tensor (zeros/nonzeros encode atom selection), plus every #557 context
model/header/payload byte. The container refuses truncated, trailing, wrong-shape, wrong-codec,
or non-canonical encodings.

FREE generic interpreter state is frame construction, decoded-chart corridor construction,
arc-length phase construction, sparse-solve replay, and correction rendering. A regression test
decodes the same selected atoms against a different Lane chart and obtains a different chart-bound
mask without truth access. No GT mask, scorer weight, RGB/YUV payload, or per-video atom parameter
is embedded in interpreter code.

## Reproducibility and custody

- Final receipt: `boundary_inverse_custody_20260721T052100Z.json`, SHA-256
  `2c7c091c61d1676c80b5db1772a29d3b2f73934398966c8566b6175abc4021e3`.
- Threshold-0.25 control receipt: `boundary_inverse_custody_20260721T050700Z.json`, SHA-256
  `96b883cab958ce4485d759b6ae706ac0dbdace20754d7f587b8cffa8404229af`.
- Parent Lane receipt: SHA-256
  `273d7ef28b9312973831403c57552274a8fef57f53a1eb4517c1e3551d76ef94`.
- GT cache: 5,078,017,610 B, SHA-256
  `cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6`, read-only ZIP_STORED
  memmap.
- Genuine #502 proof: SHA-256
  `677a2252c43c1272ec0e2e83d65ce1b82d23b8ddb089d73a111a5f0b26d46d25`.
- SSD stage:
  `/Volumes/VertigoDataTier/pact/evidence/boundary_inverse_20260721/run_20260721T052100Z_threshold0p5/`.
- Chart/sample/program manifest SHA-256 values:
  `5ed7e187b417cba0e6b8b56f35498b58ca23bc34a946d10fb3a2b3377f88cf6b`,
  `707dbf04c9c957c44592faf4e64324db37a0146c0d60ed8dbd042d45fda464f1`, and
  `62b7ea4507a8ef46222127632970beaea50a0054e47d78d81f8608bc15966dfb`.
- The fresh full run took 304.841 seconds; all 25-pair evaluation checkpoints are preserved.
- After the final input/resume guards changed the source hash, the eight evaluation states were
  recomputed from the preserved programs in 53.352 seconds under renderer SHA-256
  `21518fa8cbbb949babdb6394717f044050d250fb5b7eef1ac65e6fcb691c838b` and matched the original
  sweep exactly. A strict-manifest `--resume` replay then completed in 3.781 seconds and again
  reproduced the chart reconstruction, semantic detection, and all eight sweep rows exactly.
  Its SSD-only receipt SHA-256 is
  `ced0ec13d935965ebf5907846acc7d1b1d0054822dccec792e3791d2dba99fb9`.
- A host matrix-contraction warning was caught before authority. Suspect states were moved intact
  to `evaluation_invalid_accelerate_matmul_20260721T051800Z`; authoritative rows were rerun through
  explicit deterministic reductions and source-hash-bound evaluation manifests.
- Focused verification: 17 tests pass with warnings as errors; Ruff, py_compile, and diff checks
  pass. Two clean `review_tracker` passes cover all 57 entities in the four changed Python files.

## Cathedral triality and MAIN landing review

- DSL/wire leg: strict `BIC1` counted sidecar and exact #557 context-arithmetic parse-back.
- DAG leg: `boundary_inverse_custody_DAG_FEED_20260721T052100Z.md` replaces the old missing-inverse
  edge with the finite measured edge while refusing score/rate admission.
- Equation leg: genuine #502 frame laws and `realization_breakeven_bytes_v1` are consumed by ID;
  no mask-F1-to-score law is invented.

MAIN landing review is mandatory. It must recheck literal-family custody, exact sidecar parse-back,
the generic-versus-phase negative scope, the non-authorizing mask-F1 boundary, and the #596/#597
receiver handoff before merge or any promotion claim.

## STORES CONSULTED

- `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`, `docs/operating_manual_craft_handoff.md`
- v7.5 and v8 canonical SPECs
- `reports/latest.md`
- `.omx/state/lane_registry.json`
- `.omx/state/subagent_progress.jsonl`
- `.omx/state/master_gradient_anchors.jsonl`
- `.omx/state/modal_call_id_ledger.jsonl`
- `.omx/state/cost_band_posterior.jsonl`
- `.omx/state/continual_learning_posterior.jsonl`
- `.omx/state/canonical_task_status.jsonl`
- `.omx/research/s2_lane_true_mask_curve_20260721T042500Z.json`
- `.omx/research/s2_compose_full_partition_20260721T041640Z.{md,json}`
- `.omx/research/genuine_curvelet_shearlet_structural_proof_v2_polar_frequency_wedge_20260714.json`
- `/Users/adpena/Projects/pact/.omx/research/rep_mine_solved_binary_20260721T045500Z.json`
- delegated per-arm and broadcast inboxes through 2026-07-21T05:17:23Z
- operator per-flip sell-back directives at 2026-07-21T05:34:22Z and 05:37:49Z
