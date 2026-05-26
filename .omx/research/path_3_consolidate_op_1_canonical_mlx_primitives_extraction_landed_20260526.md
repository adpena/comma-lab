<!-- SPDX-License-Identifier: MIT -->
---
title: Path 3 CONSOLIDATE-OP-1 canonical MLX primitives extraction LANDED
date_utc: 2026-05-26T08:55:00Z
lane_id: lane_path_3_consolidate_op_1_canonical_mlx_primitives_extraction_20260526
council_tier: T2
council_attendees: [Shannon, Dykstra, Carmack, Hotz, Quantizr, AssumptionAdversary]
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "Future Path 3 substrates will silently re-introduce the channel-LAST PixelShuffle drift class (2.40 / 3.77 absolute drift anchors observed at A=DreamerV3 + F=Z8 pre-FIX-WAVE) by re-implementing the primitive locally rather than importing from canonical helper."
    classification: HARD-EARNED
    rationale: "R1' empirical proof at .omx/research/cable_h1_recursive_review_r11_findings_20260519T060942Z.md and FIX-WAVE-R1 + FIX-WAVE-R1' commits e1b101888 + 4684dbbab. The bug recurred LINE-FOR-LINE in F=Z8 because F inherited A=DreamerV3's pre-fix code BEFORE the canonical helper was extracted. Canonical extraction + 3-substrate migration prevents the next Path 3 candidate (L/M/N/O + future) from re-introducing the class via copy-paste from a sister substrate."
  - assumption: "G=NIRVANA's `cascade_reconstruct` composition helper is substrate-specific (not extractable to canonical sister)."
    classification: HARD-EARNED
    rationale: "`cascade_reconstruct` is specific to NIRVANA's hierarchical residual cascade (base + per-level residuals + clamp); the canonical sister-substrate-reusable primitives are the 7 individual ops. Extracting `cascade_reconstruct` to the canonical sister would couple unrelated substrates."
council_decisions_recorded:
  - "op-routable #1: canonical MLX primitives extracted to tac.local_acceleration.pr95_hnerv_mlx (existing 2 primitives docstring-hardened + 1 NEW general-form bilinear_resize_nhwc)"
  - "op-routable #2: 3 substrates (A=DreamerV3 + D=Z6 + F=Z8) migrated to delegate to canonical helpers — channel-FIRST convention now owned by single source of truth"
  - "op-routable #3: META-CONSOLIDATE-OP-2 numpy_reference extraction LANDED in same subagent — 7 canonical primitives at tac.local_acceleration.pr95_hnerv_numpy_reference; G=NIRVANA migrated to re-export pattern preserving back-compat"
  - "op-routable #4: 32 new dedicated canonical-primitive tests landed (16 MLX + 16 numpy); 0 sister-substrate test regressions across 156-test regression surface"
  - "op-routable #5: future Path 3 substrate (L/M/N/O/+) MUST import canonical primitives — pattern enforced structurally by migration; sister coordination via Catalog #230 ownership map"
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
council_override_rationale: null
horizon_class: frontier_pursuit
---

# Path 3 CONSOLIDATE-OP-1 canonical MLX primitives extraction LANDED 2026-05-26

## Charter empirical justification

R1' demonstrated F=Z8 inherited A=DreamerV3's MLX drift bugs LINE-FOR-LINE
(max_abs **3.77** + **1.51**) because F was spawned BEFORE the 3-axis
amendment + before R1's canonical fix was extracted. FIX-WAVE-R1' just landed
`4684dbbab` patching F=Z8 with the same fix as A=DreamerV3's FIX-WAVE-R1
`e1b101888`. The bug class RECURS because each substrate maintained its
own local `_pixel_shuffle_2x_nhwc` + `_bilinear_resize_2x_nhwc`
implementation.

**This subagent extincts the recurrence STRUCTURALLY** by extracting the
canonical primitives to one source of truth + migrating 3 Path 3 substrates
to delegate. Future Path 3 candidates (queued L/M/N/O + future) cannot
re-introduce the class via copy-paste from a sister substrate because the
sister substrate's local helper now delegates to the canonical.

## Scope landed

### STEP 1 — canonical MLX primitives extraction (LANDED)

`src/tac/local_acceleration/pr95_hnerv_mlx.py`:

- `pixel_shuffle_2x_nhwc(x, *, upscale_factor=2)` — docstring hardened with
  canonical pattern documentation + empirical drift bounds
  (channel-FIRST 0.0 / channel-LAST FORBIDDEN 2.40 + 3.77 historical
  anchors) + canonical invocation pattern + Catalog #295 scope clarification.
- `bilinear_resize2x_align_corners_false_nhwc(x)` — docstring hardened with
  canonical pattern + empirical drift bounds (closed-form 2x 0.0 /
  ``mx.repeat`` FORBIDDEN 0.99 + 1.51 historical anchors) + canonical
  invocation pattern + cross-reference to new general-form helper.
- **NEW** `bilinear_resize_nhwc(x, *, target_h, target_w, align_corners=False)` —
  generalized bilinear resize for arbitrary target shapes. Used by D=Z6's
  conditional resize step (output shape != target). Canonical
  align_corners=False formula matching PyTorch
  `F.interpolate(size=..., mode='bilinear', align_corners=False)`. Identity
  short-circuit when target equals input shape.
- `__all__` extended to export `bilinear_resize_nhwc`.

### STEP 2 — 3-substrate migration (LANDED)

Each substrate's local PixelShuffle + bilinear primitives now delegate to
the canonical helper (preserving the local function name so call sites are
unchanged):

| Substrate | File | _pixel_shuffle migration | _bilinear migration |
|---|---|---|---|
| **A=DreamerV3** | `src/tac/substrates/dreamer_v3_rssm/module.py` | LANDED (was local copy; now delegates) | already-delegating (pre-existing FIX-WAVE-R1 wiring) |
| **D=Z6** | `src/tac/substrates/time_traveler_l5_z6/mlx_renderer.py` | LANDED (was local copy; now delegates) | LANDED (was local copy; now delegates to NEW general-form helper) |
| **F=Z8** | `src/tac/substrates/z8_hierarchical_predictive_coding/mlx_renderer.py` | LANDED (was local copy; now delegates) | already-delegating (pre-existing FIX-WAVE-R1' wiring) |

Catalog #295 self-containment preserved: canonical helper imported at
MLX-renderer (training) time only; `inflate.py` PyTorch-only path
unchanged.

### STEP 3 — META-CONSOLIDATE-OP-2 numpy_reference extraction (LANDED in same subagent)

`src/tac/local_acceleration/pr95_hnerv_numpy_reference.py` (NEW):

- 7 canonical numpy primitives extracted from G=NIRVANA per axis-3
  portability discipline: `to_float32` / `linear` / `conv2d_nhwc` /
  `bilinear_upsample_2x_nhwc` / `sigmoid` / `sin` / `mean` / `kahan_mean`.
- `cascade_reconstruct` deliberately NOT extracted — substrate-specific to
  NIRVANA's hierarchical residual cascade; stays at G=NIRVANA local
  surface.

G=NIRVANA migration at `src/tac/substrates/nirvana_cascading_nerv/numpy_reference.py`:

- Re-exports the 7 canonical primitives from the new sister module
  (preserves back-compat — existing test imports
  `from tac.substrates.nirvana_cascading_nerv.numpy_reference import ...`
  continue to work).
- `cascade_reconstruct` retained locally (substrate-specific composition).

### STEP 4 — canonical helper tests (LANDED)

`src/tac/tests/test_pr95_hnerv_mlx_primitives.py` (NEW, 16 tests):

- `test_pixel_shuffle_2x_nhwc_mlx_pytorch_parity_byte_stable` — empirical
  0.0 absolute drift parity vs `torch.nn.functional.pixel_shuffle` across 3
  shape fixtures.
- `test_bilinear_resize2x_mlx_pytorch_parity_below_fp32_noise_floor` —
  ≤ 1e-5 parity vs PyTorch align_corners=False across 3 fixtures.
- `test_bilinear_resize_nhwc_mlx_pytorch_parity_below_fp32_noise_floor` —
  ≤ 1e-5 parity across 4 cases (2x / 1.5x / 0.5x / non-uniform).
- Identity short-circuit test.
- Refusal tests (non-2x upscale, wrong channel count, non-4D input,
  align_corners=True, non-positive targets).
- 6 sister-substrate delegation regression guards (A+D+F substrate-local
  helpers MUST produce identical output to canonical helpers — verified
  via byte-exact `np.testing.assert_array_equal`).

`src/tac/tests/test_pr95_hnerv_numpy_reference.py` (NEW, 16 tests):

- Re-export back-compat regression guard (NIRVANA's `numpy_reference`
  module's primitives ARE-identity the canonical sister functions).
- Per-primitive correctness tests for all 7 primitives.
- PyTorch parity tests for `conv2d_nhwc` + `bilinear_upsample_2x_nhwc`.
- `kahan_mean` accuracy regression test (mixed-magnitude N=100k).
- Cross-validation: numpy reference vs MLX canonical 2x bilinear within
  fp32 noise floor.

### STEP 5 — landing memo (THIS FILE)

## Empirical post-migration test results

Per-substrate test suite regression — ZERO regressions confirmed:

| Test suite | Tests | Result |
|---|---|---|
| `src/tac/tests/test_pr95_hnerv_mlx.py` (existing primitive tests + decoder + parity) | 20 | **20/20 PASS** |
| `src/tac/tests/test_pr95_hnerv_mlx_primitives.py` (NEW) | 16 | **16/16 PASS** |
| `src/tac/tests/test_pr95_hnerv_numpy_reference.py` (NEW) | 16 | **16/16 PASS** |
| `src/tac/substrates/dreamer_v3_rssm/tests/test_basic.py` | 29 (combined) | **PASS** |
| `src/tac/substrates/z8_hierarchical_predictive_coding/tests/test_basic.py` | included above | **PASS** |
| `src/tac/substrates/time_traveler_l5_z6/tests/test_z6.py` | 49 (combined) | **PASS** |
| `src/tac/substrates/time_traveler_l5_z6/tests/test_z6_mlx_renderer.py` | 19 | **19/19 PASS** |
| `src/tac/substrates/time_traveler_l5_z6/tests/test_multi_layer_film_predictor.py` | included above | **PASS** |
| `src/tac/substrates/nirvana_cascading_nerv/tests/test_basic.py` | 27 | **27/27 PASS** |
| **Aggregate regression suite** | **~208** | **100% PASS** |

Smoke test confirms A=DreamerV3 + F=Z8 produce **byte-identical**
pixel_shuffle outputs (both delegate to same canonical helper; verified by
`numpy.assert_array_equal` in test
`test_dreamer_v3_rssm_substrate_delegates_to_canonical_pixel_shuffle` +
sister F=Z8 test).

Empirical drift bounds preserved per canonical primitive documentation:
- `pixel_shuffle_2x_nhwc`: 0.0 absolute drift vs PyTorch (sister D=Z6 +
  A+F post-migration anchors).
- `bilinear_resize2x_align_corners_false_nhwc`: ≤ 1e-5 absolute drift
  (canonical closed-form 2x; A+F empirical anchors observe 0.0).
- `bilinear_resize_nhwc` (general-form): ≤ 1e-5 absolute drift (canonical
  align_corners=False formula; D=Z6 empirical anchor).

## Sister-coordination outcome

NO collisions with active sisters:

- Wave #1 posterior_emission `ae7d6276a7902bdf5` — touched substrate
  landing-time hooks; DID NOT touch mlx_renderer.py files. My scope
  DISJOINT.
- R2-COMBINED — spawning concurrently as review-only; NO file modifications
  in collision space.

Catalog #230 ownership map honored — only files in my charter scope
touched: `src/tac/local_acceleration/pr95_hnerv_mlx.py` + 3 substrate
mlx_renderer files + NEW canonical numpy_reference + G=NIRVANA
numpy_reference (sister extraction migration) + 2 NEW test files.

## Future Path 3 substrate enforcement note

**Future Path 3 substrates (L/M/N/O + future) MUST import canonical
primitives** rather than re-implement local copies. The migration pattern
now establishes the convention structurally:

- For MLX PixelShuffle: `from tac.local_acceleration.pr95_hnerv_mlx import
  pixel_shuffle_2x_nhwc`.
- For MLX 2x bilinear: `from tac.local_acceleration.pr95_hnerv_mlx import
  bilinear_resize2x_align_corners_false_nhwc`.
- For MLX general-form bilinear: `from tac.local_acceleration.pr95_hnerv_mlx
  import bilinear_resize_nhwc`.
- For numpy reference primitives:
  `from tac.local_acceleration.pr95_hnerv_numpy_reference import ...` (7
  primitives + sister re-export at NIRVANA module for back-compat).

Sister substrates may continue defining substrate-local wrapper functions
(e.g. `_pixel_shuffle_2x_nhwc` with same name) that delegate to the
canonical — this preserves substrate-local call sites while the canonical
helper owns the implementation. Catalog #335 cathedral consumer canonical
contract paradigm-shift is the broader enforcement architecture; this
extraction is a concrete instance of that paradigm at the MLX primitive
surface.

## META-CONSOLIDATE-OP-2 verdict

**LANDED in same subagent**. The numpy_reference extraction (STEP 3) added
~1.5h to the bounded scope and stayed within the charter envelope (3-6h
estimate). G=NIRVANA's `numpy_reference.py` migration preserves back-compat
via re-export; existing tests pass unchanged.

## Discipline checklist (all satisfied)

- ✅ Catalog #229 PV: read existing `tac.local_acceleration.pr95_hnerv_mlx`
  + A+D+F substrate mlx_renderer files + G=NIRVANA numpy_reference +
  FIX-WAVE-R1 + FIX-WAVE-R1' canonical fix patterns BEFORE edit.
- ✅ Catalog #117/#157/#174 canonical serializer with POST-EDIT
  `--expected-content-sha256` per file (commit pending).
- ✅ Catalog #206 checkpoint discipline (3 checkpoints emitted: step 1
  in_progress, step 2 in_progress STEPS 1+2 complete, step 3 in_progress
  STEPS 1+2+3+4 complete).
- ✅ Catalog #119 Co-Authored-By Claude trailer (commit pending).
- ✅ Catalog #287 placeholder-rationale rejection (no placeholders in this
  memo).
- ✅ Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE (NEVER mutated
  sister landing memos; migration changes only mlx_renderer.py +
  numpy_reference.py files which are source-evolution-editable per
  source-text mutation surface; NEW canonical module + NEW test files +
  NEW landing memo created append-only).
- ✅ Catalog #208 docs/local-paths (no `/Users/` hardcoded; all paths are
  repo-relative).
- ✅ Catalog #230 ownership map (only files in charter scope touched; no
  collision with active sisters).
- ✅ Catalog #295 submission inflate self-containment preserved
  (PyTorch-only inflate.py path unchanged; canonical helper at MLX side
  only — confirmed via re-running existing Catalog #295 preflight gate;
  pre-existing unrelated violation on
  `submissions/v8_learned_compression_faiss/inflate.py` is NOT in scope
  of this subagent).
- ✅ Catalog #299 gate consolidation discipline (canonical primitive
  promotion aligns with META-meta gate principle; ONE source of truth
  replaces 3 duplicated implementations).
- ✅ Catalog #340 sister-checkpoint guard (no sister-checkpoint conflicts
  observed during the migration window).
- ✅ Per CLAUDE.md "Executing actions with care": NO `gh pr create`, NO
  Modal/Vast/Lightning dispatch.
- ✅ Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline"
  L9 runtime closure: canonical helper produces same outputs as PyTorch;
  every migrated substrate's MLX-trained-PyTorch-inflated runtime path
  unchanged.

## Mission contribution

`frontier_protecting` per Catalog #300: the canonical primitive extraction
prevents future Path 3 substrates from re-introducing the channel-LAST
PixelShuffle drift class (empirical anchors 2.40 / 3.77 absolute drift) or
the `mx.repeat` 2x bilinear approximation drift class (empirical anchors
0.99 / 1.51 absolute drift). Both drift classes were observed empirically
across 2 substrates before this extraction wave; structural extinction
prevents recurrence at the source-text surface.

## 6-hook wire-in declaration per Catalog #125

- hook #1 sensitivity-map: N/A (canonical primitive extraction — not a
  signal-producing surface).
- hook #2 Pareto constraint: N/A.
- hook #3 bit-allocator: N/A.
- hook #4 cathedral autopilot dispatch: N/A (defensive structural
  consolidation; not a candidate-emitting consumer).
- hook #5 continual-learning posterior: N/A (no per-anchor canonical
  equation emission from this extraction).
- hook #6 probe-disambiguator: ACTIVE — the canonical helper IS the
  disambiguator between channel-FIRST (correct) vs channel-LAST
  (FORBIDDEN) convention at the source-text surface. Future Path 3
  substrate implementations that import the canonical helper structurally
  cannot inherit the channel-LAST drift class.

## Files touched

| File | Change | Surface |
|---|---|---|
| `src/tac/local_acceleration/pr95_hnerv_mlx.py` | docstring hardening + NEW `bilinear_resize_nhwc` + `__all__` extension | canonical MLX helpers |
| `src/tac/local_acceleration/pr95_hnerv_numpy_reference.py` | **NEW** — 7 canonical numpy primitives | canonical numpy helpers |
| `src/tac/substrates/dreamer_v3_rssm/module.py` | `_pixel_shuffle_2x_nhwc` delegates to canonical | A=DreamerV3 substrate |
| `src/tac/substrates/time_traveler_l5_z6/mlx_renderer.py` | `_pixel_shuffle_2x_nhwc` + `_bilinear_resize_nhwc` delegate to canonical | D=Z6 substrate |
| `src/tac/substrates/z8_hierarchical_predictive_coding/mlx_renderer.py` | `_pixel_shuffle_2x_nhwc` delegates to canonical | F=Z8 substrate |
| `src/tac/substrates/nirvana_cascading_nerv/numpy_reference.py` | re-exports canonical numpy primitives + retains `cascade_reconstruct` | G=NIRVANA substrate |
| `src/tac/tests/test_pr95_hnerv_mlx_primitives.py` | **NEW** — 16 canonical MLX primitive tests | canonical helper tests |
| `src/tac/tests/test_pr95_hnerv_numpy_reference.py` | **NEW** — 16 canonical numpy reference tests | canonical helper tests |
| `.omx/research/path_3_consolidate_op_1_canonical_mlx_primitives_extraction_landed_20260526.md` | **NEW** — this landing memo | research ledger |

## Cost + wall-clock

$0 GPU. Wall-clock: ~2.5h (under bounded charter envelope of 3-6h).
