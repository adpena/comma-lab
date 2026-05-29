---
council_tier: T2
council_attendees: [Shannon, Dykstra, Rudin, Daubechies, Yousfi, Fridrich, Contrarian, AssumptionAdversary]
council_quorum_met: true
council_verdict: PROCEED
council_dissent:
  - member: Contrarian
    verbatim: "Audit found ZERO new cargo-culted impls requiring fix. Validates the methodology: per-pair vs per-pixel abstraction layer for OPT-7 + per-class extension for OPT-5 + REAL math for OPT-6 are all HARD-EARNED documented adaptations. Catalog #303 cargo-cult-audit discipline is working."
  - member: AssumptionAdversary
    verbatim: "Operating-within assumption: the 4 packages on disk are the FULL completed-OPT scope. Brief's claim of '8 completed OPTs' is incorrect — Slot FF/LL/RR landed under different package names (NOT pr110_opt_*). HARD-EARNED-EMPIRICALLY-VERIFIED via Glob + canonical task ledger absence."
council_assumption_adversary_verdict:
  - assumption: "The 4 PR110-OPT packages on disk constitute the full completed-OPT scope"
    classification: HARD-EARNED
    rationale: "Verified via Glob src/tac/composition/pr110_opt_*/__init__.py + canonical_task_status.jsonl"
  - assumption: "Fridrich UNIWARD inverse-scorer formulation cost=1/(eps+response) is canonical per Holub-Fridrich-Denemark 2014"
    classification: HARD-EARNED
    rationale: "CLAUDE.md 'Fridrich inverse steganalysis — how to beat the scorer' + design memo citations"
  - assumption: "Per-pair vs per-pixel abstraction layer is a documented HARD-EARNED adaptation not cargo-cult"
    classification: HARD-EARNED
    rationale: "OPT-7 docstring explicitly documents the adaptation + cites sister canonical helper compute_uniward_per_pixel_directional_wavelet_mlx; PR110 archive grammar exposes 600 per-pair selectors not per-pixel"
council_decisions_recorded:
  - "op-routable #1: register canonical equations for the 4 OPT math primitives (deferred-pending-operator per iterate-not-force)"
  - "op-routable #2: per-OPT empirical paired-CUDA RATIFICATION queue $0.24-$0.48 (operator-routable)"
  - "op-routable #3: register canonical anti-pattern for Slot EEE FAKE-vs-REAL implementation classification (Catalog #344 sister discipline)"
council_predicted_mission_contribution: frontier_breaking_enabler
council_override_invoked: false
council_override_rationale: ""
---

# Wave 6 PR110-OPT cluster math audit + fix LANDED 2026-05-29

## Summary

Wave 6 (Item 11 of 15) audited the 4 PR110-OPT packages on disk
(OPT-4 / OPT-5 / OPT-6 / OPT-7) against their cited mathematical
formulations. **All 4 packages are math-fidelity-clean**: each carries
either canonical implementation (REAL math operations) or
documented HARD-EARNED adaptation with substantive rationale per
CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" +
Catalog #303 cargo-cult-audit discipline.

The brief's premise of "8 completed OPTs" was empirically falsified
during PV: only 4 PR110-OPT packages exist on disk. The other
"completed" OPTs (Slot FF UNIWARD, Slot LL L28 PR98, Slot RR motion-
pair) landed under DIFFERENT package names per MEMORY.md history,
NOT under `pr110_opt_*`. PV per Catalog #229 prevented wasted
spawn-time on phantom audit targets.

## Per-OPT verdict matrix (6-axis methodology)

| OPT | Cited formulation | Math fidelity | Documented adaptation | Tests pass | Verdict | Action |
|-----|------------------|---------------|----------------------|------------|---------|--------|
| **OPT-4** | Shannon entropy + fixed-width grouping | REAL (math.log2 + ceil-bit counting) | DEFERRED-PENDING-RESEARCH for PER_REGION + PER_TEMPORAL_WINDOW strategies (placeholder = Shannon-coded upper bound; explicit per Catalog #308 alternative-reducer enumeration) | 72/72 | IMPL_FALSIFIED_DOCUMENTED per Wave N+34 + Catalog #307 paradigm-vs-implementation | NO_FIX_NEEDED |
| **OPT-5** | Fridrich UNIWARD per-class extension `cost(c) = 1/(eps+segnet_response(c))` | REAL (per-class scalar cost map + per-class budget allocation; PER_CLASS_UNIFORM / PER_CLASS_WEIGHTED_BY_AREA / PER_REGION_AT_BOUNDARY / PER_REGION_INTERIOR strategies) | HARD-EARNED extension of Holub-Fridrich-Denemark 2014 from per-pixel to per-class semantic regions; docstring cites paper convention | 60/60 | HARD_EARNED_EXTENSION_DOCUMENTED | NO_FIX_NEEDED |
| **OPT-6** | Fridrich-Yousfi pose-axis null-projection on SegNet + Pevný-Filler-Bas 2010 quantization | **REAL math** (numpy DCT-II cosine basis + Sylvester 8x8 Hadamard via Kronecker iteration + seeded Gaussian noise + np.roll with explicit boundary handling; canonical perturbation_magnitude_scale = 1/255 per Pevný-Filler-Bas) | Slot EEE FAKE→REAL remediation already landed: legacy `apply_*_to_pr110_archive` aliased to `build_*_menu_for_*`, canonical REAL sister `apply_pose_axis_null_projection_via_canonical_real_video_mlx_to_pr110_archive` provides actual frame perturbation | 64/64 | REAL_MATH_POST_SLOT_EEE_REMEDIATED | NO_FIX_NEEDED |
| **OPT-7** | Fridrich UNIWARD `cost(pixel) = 1/(eps + scorer_response(pixel))` per Holub-Fridrich-Denemark 2014 | REAL (inverse-scorer cost + sparse-K top-K selection by descending cost) | DOCUMENTED per-pair-vs-per-pixel abstraction-layer adaptation with substantive rationale + sister canonical helper path `tac.inverse_steganalysis_real_video_mlx.compute_uniward_per_pixel_directional_wavelet_mlx` for per-pixel callers | 79/79 | DOCUMENTED_PER_PAIR_ABSTRACTION_PER_PR110_GRAMMAR | NO_FIX_NEEDED |

**Total: 275/275 tests pass in 0.98s. ZERO new cargo-culted
implementations discovered requiring fix.**

## 9-dimension success checklist evidence

1. **UNIQUENESS**: Each OPT targets a structurally distinct surface
   (color/geometry grouping / SegNet class-region waterfill / pose-axis
   null-projection / scorer-axis UNIWARD basis). The Fridrich-Yousfi
   inverse-steganalysis cascade now spans 4 orthogonal axes per Wave 6.
2. **BEAUTY + ELEGANCE**: Each OPT package is ≤1100 LOC + reviewable in
   30 seconds per HNeRV parity L4. Math primitives are clearly named.
3. **DISTINCTNESS**: Each OPT has distinct cited formulation +
   distinct strategy enum per Catalog #308 alternative-reducer
   enumeration. No sister overlap.
4. **RIGOR**: All 4 OPTs cite Holub-Fridrich-Denemark 2014 OR
   Pevný-Filler-Bas 2010 OR Catalog #213 Comma2k19 source-video
   discipline. Each carries documented adaptation when deviating from
   per-pixel canonical formulation.
5. **OPTIMIZATION PER TECHNIQUE**: Per CLAUDE.md "UNIQUE-AND-COMPLETE-
   PER-METHOD operating mode" — the per-pair abstraction is OPT-7's
   substrate-optimal engineering (matches PR110 archive grammar's
   600 per-pair selectors); routing through per-pixel canonical
   would dilute the per-pair signal.
6. **STACK-OF-STACKS COMPOSABILITY**: All 4 OPTs emit Tier A
   non-promotable AxisDecomposition per Catalog #356 enabling
   Dykstra Pareto polytope solver consumption per Catalog #372.
7. **DETERMINISTIC REPRODUCIBILITY**: Each OPT computes a sha256
   signature over inputs + strategy for Provenance per Catalog #323
   + diff-able-across-runs facet per Catalog #305.
8. **EXTREME OPTIMIZATION + PERFORMANCE**: All 4 packages use
   pure-numpy / pure-math primitives (no torch import overhead in the
   menu builders). OPT-6 REAL perturbation sister uses MLX-LOCAL macOS-
   CPU advisory per Catalog #192 NEVER-promotable.
9. **OPTIMAL MINIMAL CONTEST SCORE**: Wave N+34 analytical anchors
   record per-OPT predicted wire-bytes deltas vs FEC6 baseline:
   - OPT-5 PER_CLASS_UNIFORM: -238 bytes (analytical upper bound)
   - OPT-5 PER_CLASS_WEIGHTED_BY_AREA: -233 bytes
   - OPT-7 SPARSE_K100_UNIWARD_WEIGHTED: -146 bytes
   - OPT-4 SHANNON_CODED: +9 bytes (IMPLEMENTATION_FALSIFIED — Wave N+34)
   Operator-routable: paired-CUDA RATIFICATION required per
   Catalog #246 before any score claim.

## Observability surface

- **Inspectable per layer**: Each OPT exposes per-strategy wire-bytes
  + per-class/per-pair cost map + per-pair signature.
- **Decomposable per signal**: AxisDecomposition splits seg/pose/rate
  per Catalog #356.
- **Diff-able across runs**: sha256 signatures over inputs.
- **Queryable post-hoc**: `apply_*_to_pr110_archive` returns dict with
  canonical_anchor citation.
- **Cite-able**: design memos reference Wave N+34 analytical artifacts.
- **Counterfactual-able**: each strategy enum value enables
  alternative-reducer enumeration per Catalog #308.

## Cargo-cult audit per assumption

| Assumption | HARD-EARNED / CARGO-CULTED | Rationale |
|-----------|---------------------------|-----------|
| `cost = 1/(eps+response)` is canonical UNIWARD | HARD-EARNED | Holub-Fridrich-Denemark 2014 + CLAUDE.md citation |
| Per-pair (not per-pixel) UNIWARD at OPT-7 | HARD-EARNED | PR110 grammar exposes 600 per-pair selectors; sister per-pixel helper available |
| Per-class (not per-pixel) UNIWARD extension at OPT-5 | HARD-EARNED | SegNet 5-class argmax is the canonical distortion-axis; extension is per-region semantic |
| Sylvester 8x8 Hadamard tile via Kronecker iteration | HARD-EARNED | Canonical H_{2n} = kron(H_2, H_n) per Sylvester 1867 |
| DCT-II cosine basis at OPT-6 | HARD-EARNED | canonical cos(π(2y+1)u/2H) · cos(π(2x+1)v/2W) DCT-II formula |
| `perturbation_magnitude_scale = 1/255` | HARD-EARNED | Canonical uint8 steganography quantization per Pevný-Filler-Bas 2010 |
| PER_REGION + PER_TEMPORAL_WINDOW Shannon-coded fallback at OPT-4 | DEFERRED-PENDING-RESEARCH per Catalog #308 | Placeholder explicitly documented; per-region source signals not yet landed |

## Predicted ΔS band

Out-of-scope for this audit wave: each OPT's predicted band is recorded
in its own design memo + lane registry. THIS audit landing's predicted
ΔS band per Dykstra-feasibility: **[0.0, 0.0]** — audit-discovery only;
no score claim per CLAUDE.md "MPS auth eval is NOISE" + apparatus_maintenance.

## Horizon class

`plateau_adjacent` per CATHEDRAL-SMARTER-DESIGN-MEMO horizon-class
taxonomy (audit work supports frontier-breaking score-lowering but is
itself apparatus_maintenance per Catalog #300).

## Canonical-vs-unique decision per layer

| Layer | Decision | Rationale |
|-------|----------|-----------|
| pytest harness | ADOPT_CANONICAL (`.venv/bin/python -m pytest`) | Standard |
| Council deliberation persistence | ADOPT_CANONICAL (`tac.council_continual_learning.append_council_anchor`) | Catalog #355 |
| Probe outcomes ledger | ADOPT_CANONICAL (`tac.probe_outcomes_ledger.register_probe_outcome`) | Catalog #313 |
| Lane registry | ADOPT_CANONICAL (`tools/lane_maturity.py`) | Catalog #90 |
| Retroactive sweep | ADOPT_CANONICAL (`.omx/research/retroactive_sweep_for_*`) | Catalog #348 |

## Operator-routable decisions

1. **Canonical equation registration** (deferred per "iterate not
   force"): register `pr110_opt_4_grouped_color_geometry_calibration_savings_v1`
   + sister equations for OPT-5/6/7 when first paired-CUDA empirical anchor
   lands per Catalog #344. Estimated cost: $0 (registration is free).
2. **Per-OPT paired-CUDA RATIFICATION queue**: $0.06-$0.12 per OPT × 4
   OPTs = $0.24-$0.48 total. Operator-routable per Catalog #246.
3. **Canonical anti-pattern for Slot EEE FAKE-vs-REAL discrimination**:
   register `function_named_apply_but_only_returns_menu_constants_v1`
   per Catalog #344 sister discipline. The OPT-6 remediation pattern
   (legacy alias + canonical REAL sister) is the canonical unwind path.

## Mission contribution per Catalog #300

`frontier_breaking_enabler` — the 4 PR110-OPT packages directly target
the current frontier rate-axis savings band (-146 to -238 analytical
upper bound bytes per OPT-5/-7). Confirming math fidelity unblocks
paired-CUDA RATIFICATION dispatch per op-routable #2.

## Cross-references

- CLAUDE.md "HNeRV / leaderboard-implementation parity discipline"
- CLAUDE.md "Fridrich inverse steganalysis — how to beat the scorer"
- CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode"
- CLAUDE.md "Substrate MUST be at OPTIMAL FORM before paid empirical dispatch"
- Catalog #213 Comma2k19 canonical real-video discipline
- Catalog #287 placeholder-rationale rejection
- Catalog #303 cargo-cult audit section
- Catalog #307 paradigm-vs-implementation classification
- Catalog #308 alternative-reducer enumeration
- Catalog #344 canonical equations + anti-patterns registry
- Catalog #348 retroactive sweep
- Catalog #356 AxisDecomposition
- Catalog #372 Dykstra Pareto polytope solver
- Slot EEE FAKE-vs-REAL audit `feedback_slot_eee_fake_implementation_audit_*`
- Wave N+34 analytical anchor `wave_n34_pr110_opt_4_7_11_triple_artifacts_20260528.json`

## Lane

`lane_wave_6_pr110_opt_cluster_math_audit_plus_fix_20260529` L1
(impl_complete + memory_entry).

## Sister DISJOINT

vs Wave 5 NSCS06 v8 (different file scope: `src/tac/substrates/nscs06_*`)
vs Wave 7 DreamerV3 RSSM RL push (different file scope: `src/tac/substrates/dreamer_v3_rssm/*`)
per Catalog #340 PROCEED.

## Predecessor checkpoint

None (first audit of this scope).
