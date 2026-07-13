# Inverse-depth Riemannian compander — implementation specification

Date: 2026-07-13  
Lane: `compander_build`  
Authority: BUILD-only, `$0` local, no trainer/evaluator/provider launch, no commit  
Parent receipt: `.omx/research/manifold_geometry_slots_probe_s1_s2_20260713.json`

## Outcome contract

Build the default-OFF DSL lever `MarginCompandedGroundChart`. It composes a softened
inverse-depth row compander with the existing `GroundFrameChart` projective/xi chart; it does not
replace that chart and cannot be enabled without it. The composition order is

```text
x_frame -> GroundFrameChart(x_frame) -> row_compander(y_ground)
```

so the trained coordinate field receives the canonical ground-frame coordinate followed by the
measured row-density reparameterization. The lever changes where the fixed feature/parameter
capacity lands, never the feature width, parameter count, step count, or archive budget.

## Value provenance and math

The S1 n600 receipt measured the softened inverse-depth density

```text
rho(v) proportional to (v - v_h + delta)^-2
```

on rows below the horizon, with `v_h = 174.0` rows and
`delta = 32.5257801441824` rows. The measured JS divergence is
`0.06994264689610602` versus `0.16044044809089875` for unshifted log-depth and
`0.24818817978749463` for uniform. Do not refit or round these constants.

For `b = height - 1`, `D = b - v_h`, and
`Z = 1/delta - 1/(D + delta)`, use the normalized cumulative map on
`v_h <= v <= b`:

```text
C(v) = v_h + D * ((1/delta - 1/(v-v_h+delta)) / Z).
```

Keep `C(v)=v` above the horizon. Extend linearly below the image bottom using the exact endpoint
derivative so the map remains continuous, strictly monotone, and analytically invertible for
projective coordinates outside the raster. Implement the analytic inverse. Convert normalized
`[-1,1]` y to/from rows with the existing endpoint-inclusive grid convention.

The transform is analytic and consumes no RNG, but carry and persist a deterministic seed field
(`0`, DERIVED conventional analytic-family seed) so a future empirical-CDF extension cannot add an
untracked random stream. No linalg belongs in the compander. The existing projective homography
linalg remains the pinned NumPy CPU construction.

## Source ownership

Create:

- `src/tac/boundary_math/inverse_depth_compander.py`
- `src/tac/boundary_math/tests/test_inverse_depth_compander.py`
- `tools/build_compander_ground_class_pair_ledger.py`
- `tools/probe_compander_receiver_close_ab.py`
- focused new tests for the two tools under `src/tac/tests/`

Modify additively only:

- `src/tac/witness_dsl/curriculum_dsl.py`
- `experiments/train_levelset_witness_realized_through_R_mlx.py`
- `src/tac/witness_control/resume_registry.py`

Do not edit launch-ticket directories, memory-envelope configs, `reports/latest.md`, live run dirs,
or sibling-owned research artifacts. Do not commit.

## Coordinate module requirements

Expose a frozen validated profile object, NumPy forward/inverse row and coordinate transforms, MLX
forward/inverse twins, and a wrapper that holds an existing `GroundFrameChart`. The wrapper's methods
must call the base chart first and the compander second. NumPy uses explicit fp32 operation order;
MLX mirrors that order and must be bit-identical on `mx.cpu`. Tests must cover:

- exact horizon and bottom endpoints;
- strict monotonicity and positive density;
- forward/inverse fp32 round trip;
- x-coordinate preservation;
- analytic density allocation (rows 175-210 denser than the lower tail);
- wrapper composition order versus direct `C(G(x))`;
- NumPy/MLX CPU bit parity and close default-device parity;
- invalid domain/config refusal;
- resume-state round trip and mismatch refusal.

The profile/resume state must persist the enabled version, seed, grid height, horizon, and exact
softening value under a unique `__mcc_` prefix.

## DSL, trainer, resume, and visibility requirements

Add the nilary/defaulted factory `MarginCompandedGroundChart` returning exactly one `Lever`. It emits
only canonical trainer flags, including `--ground-frame-chart`,
`--margin-companded-ground-chart`, the exact S1 horizon/softening values, and the seed. The lever is
structural from epoch zero, default OFF because no program composes it. It must appear automatically
in `lever_registry.lever_factories`, `name_composable_levers`, and the activation ledger's
`never-fired` queue.

Trainer wiring must:

- add BooleanOptionalAction/default-off plus typed horizon/softening/seed args near the existing
  ground-frame chart parser region;
- fail closed if the compander is requested without `--ground-frame-chart`;
- build the base `GroundFrameChart`, then wrap it;
- route the existing per-pair curvelet-feature closure through the wrapper only when ON;
- leave OFF and GroundFrameChart-only paths unchanged;
- keep feature/cache shapes identical (no projected-memory change);
- persist config in deploy and resume arrays and include it in the resume divergence guard;
- register the compander state through the run-scoped `ResumeRegistry` under the literal name
  `margin_compander`; add that name to `DIRECT_CONTROLLER_NAMES` so static coverage remains closed.

Do not hand-add a launch config or trainer command.

## Ground-class-pair n600 ledger

Build a reproducible read-only tool over:

- `experiments/results/mlx_fleet_gt_cache/gt_n600.npz` member `lstars`;
- all six `experiments/results/residual_inr_adversarial_overturn_20260630T235910Z/n600/chunk_*.npz`
  cached witness argmax chunks.

No scorer, renderer, evaluator, or training call is allowed. Verify exactly 600 unique pair indices
with no gap/duplicate and shape `(600,384,512)`. Record SHA-256 custody for every source. Emit an
atomic, fingerprint-resumable small JSON ledger at
`.omx/research/compander_ground_class_pair_ledger_n600_20260713.json` containing every directed
off-diagonal source->witness class pair and every undirected pair, their total flip count/share, and
the normalized 384-row density/count arrays. Mark strict planar ground pairs conservatively as
Road<->Lane; keep every other pair visible instead of asserting that mixed `Undrivable` (contains
sky) or `Movable` is globally planar. For each nonempty pair, compare uniform, unshifted log-depth,
and the fixed S1 softened-inverse-depth density on `v>174`; do not refit delta. Include explicit
false-authority fields and provenance. The output is a cached-witness mechanism ledger, not the
future chart-arm verdict.

## Receiver-close counted A/B harness stub

Build but do not run a pure receipt comparator. It accepts future control and treatment arm JSON
receipts and refuses unless:

- both are n600 and have identical optimizer steps and total archive bytes;
- each archive parses back and its decoded hash equals its pre-archive reference hash;
- the treatment identifies `MarginCompandedGroundChart`;
- the video-derived chart payload is explicitly counted inside treatment archive bytes with byte
  count and SHA-256 custody;
- every canonical per-class d_seg value exists, including Lane;
- axis, config, archive, and receiver custody are present.

The output reports treatment-control per-class d_seg deltas, with Lane primary, at matched bytes and
steps. It makes no score/promotion claim. Unit tests use synthetic receipts only.

## Triality and final artifacts

Reuse canonical equation `flip_density_chart_metric_v1`; do not register a duplicate law. After code
review, the main agent will write:

- `.omx/research/compander_build_DAG_FEED_20260713.md`
- `.omx/research/compander_build_20260713.md`
- the n600 ledger receipt above
- the pool transition via locked `record_candidate`: `margin_companded_ground_chart` from
  `needs-build` to `built-never-fired` (hyphenated), DSL leg `MarginCompandedGroundChart`.

The DAG/memo must state: receiver-close counted matched-byte/matched-step n600 A/B remains owed;
Lane per-class d_seg is the primary effect; pose/rate cannot worsen; pointer delta is NONE; no launch
or commit occurred; projected memory is unchanged, so the sibling launch-ticket recompile needs no
memory-envelope adjustment for this lever.

## Verification required before handoff

Run focused pytest suites, existing `test_ground_frame_chart.py`, resume-registry and lever-registry
tests, trainer parse/compile behavior tests, `ruff check` on every touched Python file, and a direct
NumPy/MLX CPU parity measurement. Do not run the trainer, scorer, evaluator, byte-close, or A/B tool.
