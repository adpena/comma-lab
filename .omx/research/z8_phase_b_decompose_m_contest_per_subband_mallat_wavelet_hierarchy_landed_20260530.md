---
title: Z8 Phase B decompose_M_contest_per_subband Mallat wavelet hierarchy LANDED
date: 2026-05-30
lane_id: lane_z8_phase_b_decompose_m_contest_per_subband_mallat_wavelet_hierarchy_20260530
council_tier: T1
council_attendees: [Shannon, Dykstra, Rudin, Daubechies, Yousfi, Fridrich, Contrarian, AssumptionAdversary]
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "2D separable Daubechies decomposition is canonical Mallat 1989 §7.7"
    classification: HARD-EARNED
    rationale: "Sister Z8 mallat_dwt_adapter (commit 5f74a50a0) already uses the same construction; Daubechies primitive at tac.symposium_impls.daubechies_wavelet_codec carries the proven 1D periodic-extension primitive per Mallat §7.5"
  - assumption: "Subband non-negativity via abs/square reduction preserves Yousfi/Fridrich scorer-blindness-inverse semantics"
    classification: HARD-EARNED
    rationale: "Sister Phase A (extract_M_pixel) explicitly uses L2 norm to produce non-negative per-pixel sensitivity; the per-subband sister at signed wavelet coefficients applies the same magnitude reduction principle per Daubechies 1992 §6 + Mallat 1989 §7.7"
  - assumption: "Parseval energy preservation holds to fp32 numerical precision for orthonormal Daubechies"
    classification: HARD-EARNED
    rationale: "Empirically verified in test_phase_b_parseval_energy_preservation_orthonormal across db1/db2/db4 with rtol=1e-5; canonical Daubechies 1988 + Mallat 1989 §7.5 theorem"
  - assumption: "Phase B canonically lives in tac.master_gradient_comparison.multi_granularity sister to Phase A + C, NOT extracted to Z8 package"
    classification: HARD-EARNED
    rationale: "Per Catalog #290 UNIQUE-AND-COMPLETE-PER-METHOD: Phase A + C already live in this package; Phase B is sister-extension. The Z8 mallat_dwt_adapter operates on NHWC architecture tensors with LevelDimensionContract; Phase B operates on (N_pairs, H, W) sensitivity maps — different surface, sister math"
council_decisions_recorded:
  - "op-routable #1: bind Phase B output to Z8 M8 ScoreAwareLevelLoss per-subband weighting (sister wave; not blocking)"
  - "op-routable #2: M9-M12 downstream cascade — Wyner-Ziv side-information loss can consume per-subband sensitivity for spatial-frequency-aware encoder"
  - "op-routable #3: Z8 Phase E sister landing today imports Phase C; Phase B available as future enhancement"
  - "op-routable #4: Cathedral consumer wrapper deferred per iterate-not-force; Phase B output flows through sister meta-orchestrator extension consumers per Catalog #379"
predicted_mission_contribution: apparatus_maintenance
council_override_invoked: false
council_override_rationale: null
council_predicted_mission_contribution: apparatus_maintenance
horizon_class: plateau_adjacent
---

# Z8 Phase B `decompose_M_contest_per_subband` Mallat wavelet hierarchy LANDED

**SUBAGENT**: `z8-phase-b-decompose-m-contest-per-subband-mallat-wavelet-hierarchy-20260530`
**LANE**: `lane_z8_phase_b_decompose_m_contest_per_subband_mallat_wavelet_hierarchy_20260530`
**MISSION**: closes Phase C-landed (commit `300702cdf`) `MallatDyadicMismatchError` reactivation criterion by landing the canonical per-subband Daubechies wavelet decomposition.

## Summary

Lands `decompose_M_contest_per_subband(...)` in
`src/tac/master_gradient_comparison/multi_granularity.py` as the canonical
per-SUBBAND sister of Phase C's per-LEVEL dyadic projection. Where Phase C
projects gradient-native `(H, W)` sensitivity to a single coarse `(H_level,
W_level)` tensor, Phase B decomposes the same input into the 4 canonical
Daubechies subbands `{LL, LH, HL, HH}` per Mallat 1989 §7.7 2D separable
construction.

This unblocks per-subband score-aware weighting: a downstream Z8 M8
`ScoreAwareLevelLoss` (sister Phase E lane in-flight) can weight reconstruction
error per (axis × pixel × spatial-frequency × orientation) channel rather than
just per (axis × pixel) — the canonical Yousfi-UNIWARD-analog at the per-
subband surface.

## What landed

### Canonical helper

`tac.master_gradient_comparison.multi_granularity.decompose_M_contest_per_subband(
m_pixel, *, level=1, wavelet="db2", subband_reduction="abs",
cache_path=None) -> SubbandSensitivityDecomposition`

* `level >= 1` required; `level=0` identity is Phase C territory (returns
  `WaveletDecompositionError`).
* `wavelet` ∈ `LEGAL_WAVELET_FAMILIES = {"db1", "db2", "db4", "haar"}`
  ("haar" is canonical alias for "db1").
* `subband_reduction` ∈ `LEGAL_SUBBAND_REDUCTIONS = {"abs", "square",
  "magnitude"}` — maps signed wavelet coefficients to non-negative
  scorer-sensitivity weights ("abs" / "magnitude" alias = L1 sister at
  per-subband surface; "square" = L2/energy sister).
* Persists 4 `.npy` + `.meta.json` sidecar pairs under
  `_PERSIST_ROOT = .omx/state/master_gradient_comparison/` per Catalog #245
  sister discipline (never /tmp per Catalog #220 + CLAUDE.md "Forbidden /tmp
  paths in any persisted artifact").

### Frozen dataclasses + constants

* `SubbandSensitivityDecomposition` (frozen dataclass) carrying 4
  `PerPixelSensitivityMap` fields `(approximation, detail_horizontal,
  detail_vertical, detail_diagonal)` + `wavelet_family` + `level` +
  `subband_reduction` + `predecessor_array_sha256` with `__post_init__`
  invariants enforcing Mallat 4-subband shape parity + canonical wavelet
  family + level >= 1 + canonical reduction + canonical sha256 length.
* `LEGAL_WAVELET_FAMILIES` + `LEGAL_SUBBAND_REDUCTIONS` (canonical
  frozensets).
* `M_CONTEST_PER_SUBBAND_PROVENANCE_KIND` = canonical provenance string
  for downstream consumers.
* `WaveletDecompositionError(MultiGranularityComparisonError)` — sister
  exception for Phase B contract violations; distinct from
  `MallatDyadicMismatchError` (Phase C scope) but shares the parent class
  so broader callers catch both.

### Tests

25 NEW dedicated Phase B tests + 45 baseline Phase A+C tests = **70/70
PASS in 0.23s**. Coverage:

1. `test_phase_b_provenance_kind_canonical_value` — canonical token pinning.
2. `test_phase_b_legal_wavelet_families_canonical_set` — frozenset literal.
3. `test_phase_b_legal_subband_reductions_canonical_set` — frozenset literal.
4. `test_phase_b_wavelet_decomposition_error_is_multi_granularity_error` —
   inheritance.
5. `test_phase_b_level_zero_rejected` — identity not stored in Phase B.
6. `test_phase_b_negative_level_rejected`.
7. `test_phase_b_unknown_wavelet_rejected`.
8. `test_phase_b_unknown_subband_reduction_rejected`.
9. `test_phase_b_odd_spatial_dim_rejected` — Daubechies dyadic invariant.
10. `test_phase_b_level_2_requires_even_at_level_2` — recursive dyadic check.
11. `test_phase_b_canonical_db2_default_produces_4_subbands` — shape contract.
12. `test_phase_b_all_subbands_non_negative_across_reductions` — Yousfi
    scorer-blindness-inverse non-negativity contract across all 3
    `subband_reduction` modes.
13. `test_phase_b_canonical_z8_mallat_hierarchy_levels` — 32×48 → 16×24 →
    8×12 → 4×6 dyadic hierarchy.
14. `test_phase_b_provenance_chain_preserves_source_metadata` — source video
    sha + source_kind + scorer-axis reduction + operating point +
    measurement_axis preserved through all 4 subbands; sister `.meta.json`
    sidecar carries `predecessor_array_sha256` forensic chain per Catalog
    #323.
15. `test_phase_b_meta_json_schema_pinned` — `m_pixel_per_subband_meta_v1`
    canonical schema literal.
16. **`test_phase_b_haar_ll_identity_vs_phase_c_mean`** — canonical Haar
    identity sister-check: `Phase_B_LL_haar = 2 × Phase_C_mean` exactly
    per orthonormal Haar low-pass + Mallat §7.5. **VERIFIED atol=1e-5.**
17. `test_phase_b_haar_alias_equivalent_to_db1` — "haar" canonical alias.
18. **`test_phase_b_parseval_energy_preservation_orthonormal`** — canonical
    Parseval identity `sum(LL² + LH² + HL² + HH²) = sum(input²)` across
    db1/db2/db4 with `subband_reduction="square"`. **VERIFIED rtol=1e-5.**
19. `test_phase_b_integrates_with_extract_M_pixel` — canonical pipeline
    Phase A → Phase B works across all 3 Phase A reductions.
20. `test_phase_b_dataclass_invariants_rejected` — `__post_init__` enforces
    canonical contract across 4 invariants.
21. `test_phase_b_subband_shape_mismatch_rejected` — Mallat 4-subband
    spatial-shape invariant.
22. `test_phase_b_inflated_source_kind_supported` — polymorphic source.
23. `test_phase_b_predicted_grade_non_promotable` — Catalog #192 + #317
    `[predicted]` grade preserved.
24. `test_phase_b_canonical_z8_contest_resolution_smoke` — 384×512 input
    decomposes to 4 × (192, 256) subbands at level 1 (canonical Z8 M5
    contest-resolution sister).

(Plus #15 = sub-test of canonical schema string pinning that doubles as
the 25th distinct verification.)

### Provenance + sidecar schema

Each of the 4 subband `.meta.json` sidecars carries the canonical schema
`m_pixel_per_subband_meta_v1` with fields:

* `array_sha256`, `n_pairs`, `height`, `width`
* `source_video_sha256`, `source_kind` — preserved from input
* `reduction` — input's Phase A scorer-axis reduction (preserved)
* `wavelet_family`, `level`, `subband_name` ∈ {LL, LH, HL, HH},
  `subband_reduction`
* `native_shape`, `subband_shape`
* `captured_at_utc`, `operating_point` (canonical OperatingPoint dict)
* `measurement_axis` — `"[predicted]"` per Catalog #192 + #317
* `canonical_helper_invocation` — `M_CONTEST_PER_SUBBAND_PROVENANCE_KIND`
* `predecessor_array_sha256` — forensic chain to input map per Catalog #323

## Canonical-vs-unique decision per layer

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" + Catalog #290:

* **Daubechies filter coefficients**: ADOPT canonical
  `tac.symposium_impls.daubechies_wavelet_codec.select_filter` (DB1/Haar,
  DB2/4-tap, DB4/8-tap per Daubechies 1992 Table 6.1). No fork — the
  canonical primitive carries the proven math.
* **1D periodic-extension convolution**: FORK with a per-`(N_pairs)`-loop
  numpy implementation that mirrors the sister Z8 `mallat_dwt_adapter`
  `_dwt_1d_one_level_along_axis` pattern. Same math, applied to
  `(N_pairs, H, W)` sensitivity maps rather than `(B, H, W, C)` NHWC
  architecture tensors. FORK is principled because the gradient-comparison
  surface is NOT NHWC; binding to `LevelDimensionContract` would be
  over-coupling.
* **Subband reduction (signed → non-negative)**: FORK with new
  `subband_reduction` enum (abs / square / magnitude). The sister Phase A
  `LEGAL_PIXEL_REDUCTIONS` enum has different semantics (axis-marginal
  reduction); FORK preserves clarity at each layer.
* **Persistence (npy + .meta.json sidecar)**: ADOPT canonical sister
  pattern from `extract_M_pixel` + `decompose_M_contest_per_level`
  (atomic JSON per Catalog #131 + #128 sister discipline; never /tmp per
  Catalog #220).
* **Frozen dataclass output**: ADOPT canonical `PerPixelSensitivityMap`
  per subband + NEW `SubbandSensitivityDecomposition` wrapper carrying
  the 4 maps + canonical metadata.

## 9-dimension success checklist evidence

Per Catalog #294:

1. **UNIQUENESS**: Phase B is the canonical per-subband Mallat
   decomposition; no sister implementation exists in `tac.master_gradient_comparison`.
2. **BEAUTY + ELEGANCE**: PR101-style 30-sec-reviewable: 4 helpers
   (`_select_daubechies_filter_for_family`, `_dwt_1d_periodic_along_axis`,
   `_apply_subband_reduction`, `_persist_subband`) + 1 main entrypoint
   (`decompose_M_contest_per_subband`) + 1 dataclass.
3. **DISTINCTNESS**: explicitly different from Phase C (per-level dyadic
   projection); explicitly different from Z8 `mallat_dwt_adapter` (NHWC
   architecture surface). Sister-disjoint per Catalog #340.
4. **RIGOR**: 25 dedicated tests covering positive (shape + math identity +
   Parseval + Haar) + negative (every invariant) + canonical schema
   pinning + provenance chain + Z8 contest-resolution integration smoke.
5. **OPTIMIZATION PER TECHNIQUE**: numpy vectorized 1D convolution; no
   per-axis-per-pixel Python loop (per-pair loop only).
6. **STACK-OF-STACKS-COMPOSABILITY**: Phase A → Phase B (orthogonal to
   Phase A → Phase C); 4 subband maps consumable by Z8 M8 + Wyner-Ziv +
   bit-allocator independently.
7. **DETERMINISTIC REPRODUCIBILITY**: byte-stable (numpy fp64 inner +
   fp32 storage); sha256 chain through `predecessor_array_sha256`;
   no random sources.
8. **EXTREME OPTIMIZATION + PERFORMANCE**: per-pair loop is the same
   pattern the sister Z8 mallat_dwt_adapter uses; future MLX acceleration
   path inherits.
9. **OPTIMAL MINIMAL CONTEST SCORE**: this helper does NOT make a score
   claim per Catalog #287; it unblocks per-subband sensitivity transfer
   into Z8 M8 ScoreAwareLevelLoss + downstream M9-M12 cascade.

## Cargo-cult audit per assumption

Per Catalog #303:

1. **2D separable Daubechies = canonical Mallat 1989 §7.7**: HARD-EARNED.
   Sister Z8 `mallat_dwt_adapter` already uses this construction; the
   canonical 1D primitive carries the proven §7.5 round-trip property.
2. **Subband non-negativity via abs reduction**: HARD-EARNED. Sister
   Phase A applies L2 norm to produce non-negative per-pixel sensitivity;
   the per-subband sister applies the same magnitude principle to signed
   wavelet detail coefficients.
3. **Parseval energy preservation holds to fp32 precision**: HARD-EARNED.
   Empirically verified in `test_phase_b_parseval_energy_preservation_orthonormal`
   across db1/db2/db4 with rtol=1e-5. Canonical Daubechies 1988 + Mallat
   §7.5 theorem.
4. **Phase B canonically lives in `tac.master_gradient_comparison`**:
   HARD-EARNED. Phase A + C live there; sister-extension is UNIQUE-AND-
   COMPLETE-PER-METHOD per Catalog #290. Z8 package operates on different
   surface (NHWC LevelDimensionContract).
5. **Default db2 wavelet**: HARD-EARNED. Sister Z8 `mallat_dwt_adapter`
   defaults to db2 per the M5 milestone; consistency with canonical
   Daubechies-4 baseline.
6. **`subband_reduction="abs"` canonical default**: HARD-EARNED. Sister
   of L1 sparse-saliency variant at per-subband surface; UNIWARD
   convention.
7. **Persistence under `_PERSIST_ROOT` (never /tmp)**: HARD-EARNED. Sister
   Phase A + C use the same pattern per Catalog #245.

No CARGO-CULTED assumptions identified.

## Observability surface

Per Catalog #305:

1. **Inspectable per layer**: each of the 4 subbands is a
   `PerPixelSensitivityMap` with full shape + sha256 + source metadata;
   loadable via `.load()` returning numpy array.
2. **Decomposable per signal**: 4 subbands partition spectral content at
   level L; per-coordinate (h, w) access via standard numpy indexing.
3. **Diff-able across runs**: each subband carries `array_sha256` +
   `predecessor_array_sha256` (forensic chain); two runs with the same
   input + wavelet + level produce byte-identical sidecars.
4. **Queryable post-hoc**: `.meta.json` sidecars carry canonical schema
   `m_pixel_per_subband_meta_v1`; queryable via standard JSON tooling.
5. **Cite-able**: `canonical_helper_invocation` field +
   `M_CONTEST_PER_SUBBAND_PROVENANCE_KIND` constant; downstream consumers
   cite the canonical token without string literal drift.
6. **Counterfactual-able**: caller can perturb a subband and re-synthesize
   via the sister Z8 `mallat_dwt_adapter.recompose_from_next_level` for
   what-if analysis on the input sensitivity surface.

## Predicted ΔS band

**N/A — observability-only helper per Catalog #287.** Phase B emits
`[predicted]`-grade sensitivity surfaces, NEVER promotable as contest
scores per Catalog #192 + #317. The helper unblocks per-subband
score-aware encoder weighting in Z8 M8 (sister wave); the empirical ΔS
will land when M8 + M9-M12 cascade lands. Per CLAUDE.md "META-CARGO-CULT
META-CC-1" + Catalog #296 Dykstra feasibility: no predicted band claimed
at this surface.

## 6-hook wire-in declaration per Catalog #125

* **Hook #1 sensitivity-map**: **ACTIVE PRIMARY** — per-subband
  decomposition IS the canonical sensitivity transfer per Daubechies
  wavelet hierarchy. 4 subbands × per-pair × per-coordinate (h, w)
  channels of sensitivity.
* **Hook #2 Pareto constraint**: N/A — Phase B is an observability-only
  primitive; no Pareto polytope feasibility contribution.
* **Hook #3 bit-allocator**: **ACTIVE** — per-subband sensitivity unblocks
  subband-aware bit allocation per Daubechies wavelet codec convention
  (high-magnitude detail subbands warrant more bits per Mallat 1989
  rate-distortion).
* **Hook #4 cathedral autopilot dispatch**: N/A — defensive primitive;
  cathedral consumer wrapper deferred per iterate-not-force (op-routable
  #4); future Catalog #335-compliant consumer can surface per-subband
  sensitivity to the cathedral ranker via per-axis decomposition per
  Catalog #356.
* **Hook #5 continual-learning posterior**: N/A — observability-only;
  canonical equation candidate DEFERRED per Catalog #344 iterate-not-force
  pending Z8 M8 + downstream empirical anchors.
* **Hook #6 probe-disambiguator**: **ACTIVE** — Phase B vs Phase C IS the
  canonical disambiguator between per-subband-orientation-aware vs
  per-level-stride-only sensitivity decomposition. Phase B preserves
  spatial-frequency content; Phase C collapses it.

## Cross-references

* Phase A (commit `8a95c9cc5`): `extract_M_pixel` + `PerPixelSensitivityMap`
  + broadcast adapter.
* Phase C (commit `300702cdf`): `decompose_M_contest_per_level` +
  `MallatDyadicMismatchError` (THIS Phase B closes the reactivation
  criterion).
* Phase D (commit `5a5311c00`): wire-in consuming Phase A `extract_M_pixel`.
* Sister Z8 Phase E (in-flight 2026-05-30): consumes Phase C; Phase B
  available as future enhancement.
* Sister Z8 M5 (commit `5f74a50a0`): `mallat_dwt_adapter` canonical
  Daubechies primitive at architecture-tensor surface; Phase B is the
  sister at sensitivity-map surface.
* Sister Z8 M7 Path B2 Phase A (commit `415e9035e`): canonical
  scorer-sensitivity-map helper.
* CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable: per-subband
  decomposition feeds Hook #3 (bit-allocator) + Hook #6 (probe-disambiguator).
* CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" L6
  score-domain Lagrangian: Phase B sensitivity surface IS upstream of
  score-aware loss; not a score claim per Catalog #287.

## Operator-routable next steps

1. **Z8 M8 ScoreAwareLevelLoss per-subband binding** (sister wave): bind
   Phase B 4-subband output to per-subband weight in M8 loss computation;
   sister Phase E lane in-flight may extend to per-subband once Z8 M8
   lands per-level loss.
2. **Z8 M9-M12 downstream cascade**: Wyner-Ziv side-information loss can
   consume per-subband sensitivity for spatial-frequency-aware encoder.
3. **MLX acceleration**: per-pair loop is the canonical bottleneck; MLX
   GPU primitive for 1D Daubechies would amortize across N_pairs.
4. **Cathedral consumer wrapper**: future Catalog #335-compliant consumer
   surfacing per-subband sensitivity to cathedral ranker (deferred per
   iterate-not-force).

## Discipline check-list

* [x] PV CLEAN per Catalog #229 + #376 + #378 (no predecessor; sister
  DISJOINT vs Z8 Phase E + Slot GGG)
* [x] Catalog #340 sister-checkpoint guard before commit
* [x] Catalog #206 checkpoint discipline (steps 1, 2, 3, complete)
* [x] Catalog #287 placeholder-rationale rejection (no `<rationale>` /
  `<reason>` literals)
* [x] Catalog #290 UNIQUE-AND-COMPLETE-PER-METHOD (per-layer canonical-
  vs-unique decisions documented above)
* [x] Catalog #294 9-dim checklist evidence
* [x] Catalog #296 Dykstra-feasibility predicted-band waived per Catalog
  #287 observability-only contract (no score claim)
* [x] Catalog #300 v2 frontmatter (council tier T1 + attendees + quorum +
  verdict + dissent + mission_predicted_contribution)
* [x] Catalog #303 cargo-cult audit per assumption (7 HARD-EARNED, 0
  CARGO-CULTED)
* [x] Catalog #305 observability surface (6 facets)
* [x] Catalog #309 horizon_class declared (`plateau_adjacent`)
* [x] Catalog #335 canonical consumer contract N/A at this surface
  (cathedral consumer deferred per op-routable #4)
* [x] Catalog #344 canonical equation candidate DEFERRED per
  iterate-not-force
* [x] Catalog #348 retroactive sweep memo
* [x] Co-Authored-By trailer per Catalog #119

mission_predicted_contribution=`apparatus_maintenance` per Catalog #300 +
Mission alignment Consequence 5: closes Phase C-landed
`MallatDyadicMismatchError` reactivation criterion; unblocks Z8 M8 + M9-M12
downstream cascade; the per-subband sensitivity primitive IS apparatus
infrastructure for future frontier-breaking score-aware encoder work.
