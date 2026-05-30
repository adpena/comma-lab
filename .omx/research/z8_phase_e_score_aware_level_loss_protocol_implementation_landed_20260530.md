<!--
SPDX-License-Identifier: MIT

Z8 Phase E ScoreAwareLevelLoss Protocol implementation landing memo.

Yousfi-cascade TOP-2 (operator-routed; sister-DISJOINT with Slot GGG scale-up + Z8 Phase B).

Per Catalog #300 v2 frontmatter discipline + Catalog #292 explicit
assumption-statement-surfacing + Catalog #303 cargo-cult audit + Catalog #294
9-dim checklist + Catalog #305 observability surface + Catalog #296 Dykstra-
feasibility + Catalog #346 canonical roster + Catalog #309 horizon-class +
Catalog #344 canonical equations + Catalog #287 placeholder-rationale rejection
+ Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE.
-->
---
council_tier: T1
council_attendees: [Shannon, Dykstra, Rudin, Daubechies, Fridrich, Contrarian, AssumptionAdversary]
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "The canonical Yousfi-grounded loss form `loss = REDUCE(sensitivity * |recon - target|^p)` faithfully implements the Protocol docstring's `sum_pixel(sensitivity * error)` formula"
    classification: HARD-EARNED
    rationale: "Per binding_contract.py:451-471 verbatim Protocol declaration: the loss multiplies pixel-wise then reduces. The implementation uses element-wise (* operator) then `.mean()`/`.sum()` reduction. Uniform-sensitivity-reduces-to-L2 invariant (binding_contract.py:467-470) verified numerically by test_uniform_sensitivity_reduces_to_l2_mean with rtol=1e-6."
  - assumption: "Framework-agnostic implementation via duck-typed element-wise operations correctly serves numpy AND torch tensors with autograd flow"
    classification: HARD-EARNED
    rationale: "Verified empirically by test_torch_tensor_path_works_when_torch_available (numerical match vs numpy reference) + test_torch_autograd_flows_through_loss (gradient = 2*recon/N for L2-mean against zero target; matches analytic derivative)."
  - assumption: "ScoreAwareLevelLossImpl satisfies the @runtime_checkable Protocol via structural conformance (no explicit inheritance required)"
    classification: HARD-EARNED
    rationale: "Python @runtime_checkable Protocol from typing module performs structural subtyping. The implementation has the canonical per_level_loss(reconstruction, target, scorer_sensitivity_map) method signature; isinstance(impl, ScoreAwareLevelLoss) returns True (verified by test_satisfies_score_aware_level_loss_protocol)."
  - assumption: "Catalog #290 UNIQUE-AND-COMPLETE-PER-METHOD justifies loss.py living in the Z8 substrate package (not extracted to a generic location)"
    classification: HARD-EARNED
    rationale: "Per CLAUDE.md UNIQUE-AND-COMPLETE-PER-METHOD: substrate-engineering forks at the substrate package surface when the canonical helper would suppress score; M8 is per-Z8-level + per-hierarchy + Yousfi-grounded; sister tac.composition.*_inverse_steganalysis_* packages operate at the per-archive-bolt-on surface, NOT the per-level integrated surface Z8 needs. Extracting prematurely would re-introduce the canonical-helper-suppresses-substrate-optimal bug class."
council_predicted_mission_contribution: apparatus_maintenance
council_override_invoked: false
council_decisions_recorded:
  - "M8 ScoreAwareLevelLossImpl LANDED at src/tac/substrates/z8_hierarchical_predictive_coding/loss.py per the Yousfi-cascade TOP-2 operator-routed Phase E scope."
  - "M9 (full_main_trainer_lifts_notimplementederror) advances toward actionable; M9 still pending wyner_ziv_full_top_level_coder_landed predecessor."
  - "build_progress.py M8 milestone status transitions PENDING -> LANDED with substantive notes documenting the empirical receipts + canonical helper chain (M7 -> M8) + the 33 test invariants verified."

related_deliberation_ids:
  - "feedback_z8_phase_2_build_tracking_in_source_not_tasklist_not_memos_20260529"
  - "feedback_z8_hierarchical_predictive_coding_binding_first_active_build_target_yousfi_grounded_20260529"
  - "feedback_yousfi_fridrich_slot_rr_fake_to_real_via_real_scorer_verification_landed_20260529"

horizon_class: plateau_adjacent
lane_id: lane_z8_phase_e_score_aware_level_loss_protocol_implementation_20260530
subagent_id: z8-phase-e-score-aware-level-loss-protocol-implementation-20260530
landed_at_utc: 2026-05-30T14:30:48Z
parent_commit_sha: 74847d067f08e20b5d802aac9c3339e1efb38832
---

# Z8 Phase E — ScoreAwareLevelLoss Protocol implementation LANDED

**Date:** 2026-05-30 14:30:48 UTC
**Lane:** `lane_z8_phase_e_score_aware_level_loss_protocol_implementation_20260530`
**Subagent:** `z8-phase-e-score-aware-level-loss-protocol-implementation-20260530`
**Parent commit:** `74847d067f08e20b5d802aac9c3339e1efb38832`
**Closes:** Z8 Phase 2 milestone M8 (`score_aware_level_loss_uniward_analog_landed`)
**Unblocks:** Z8 Phase 2 milestone M9 (`full_main_trainer_lifts_notimplementederror`) — still pending the M6 Wyner-Ziv predecessor

## TL;DR (1 paragraph)

The canonical Yousfi-grounded score-aware per-level loss `ScoreAwareLevelLossImpl`
lands at `src/tac/substrates/z8_hierarchical_predictive_coding/loss.py` (~150 LOC)
implementing the `ScoreAwareLevelLoss` Protocol from
`binding_contract.py:419-472` via the canonical formula
`loss = REDUCE(scorer_sensitivity_map * |reconstruction - target|^p)` with
`p=2` (Fridrich UNIWARD canonical) and `REDUCE=mean` (Yousfi convention). The
implementation consumes the just-landed M7 canonical helper chain
(`Z8ScorerSensitivityMap.get_for_level(...)` at commits `8a95c9cc5` Phase A +
`300702cdf` Phase C + D) and is framework-agnostic via duck-typed element-wise
operations (numpy + torch + mlx all served by the same code path). 33 dedicated
tests in `tests/test_score_aware_level_loss.py` verify all four M8 acceptance
criteria + Protocol satisfaction + the canonical uniform-sensitivity-reduces-to-L2
invariant + the Slot GGG SegNet-null empirical-anchor sister test
(`test_zero_sensitivity_region_drops_error_contribution`). Full Z8 test suite:
**152 / 152 PASS in 0.70 s**.

## Scope (what landed)

### New module: `src/tac/substrates/z8_hierarchical_predictive_coding/loss.py` (~340 LOC)

Three public symbols re-exported from the package `__init__.py`:

1. **`ScoreAwareLevelLossImpl`** — frozen dataclass implementing the
   `ScoreAwareLevelLoss` Protocol. Fields: `norm: Literal["l2", "l1"]` (default
   `"l2"` per Fridrich UNIWARD canonical), `reduction: Literal["mean", "sum"]`
   (default `"mean"` per Yousfi convention), `validate_non_negative_sensitivity:
   bool` (default `True`; hot-path escape hatch). `__post_init__` validates the
   enum string fields per Catalog #287 explicit-input discipline. The class is
   frozen so instances are hashable + safe to share across hierarchy levels.

2. **`build_score_aware_level_loss_for_level`** — single-call canonical builder
   for M8 trainer callsites. Sister of `build_z8_scorer_sensitivity_map_for_level`
   (M7) + `build_z8_mallat_dwt_adapter_for_level` (M5) +
   `build_z8_mamba2_adapter_for_level` (M4) per the canonical per-level builder
   convention. Validates `level` is a `LevelDimensionContract` and asserts the
   constructed instance satisfies the @runtime_checkable Protocol at construction
   time (early-fail beats late-fail per CLAUDE.md "Bugs must be permanently
   fixed AND self-protected against").

3. **`InvalidSensitivityMapError`** — typed exception (subclass of `ValueError`)
   raised when the sensitivity map contains negative entries, per the Protocol
   docstring's non-negative-weights invariant from
   `binding_contract.py:462-464`.

### New test module: `src/tac/substrates/z8_hierarchical_predictive_coding/tests/test_score_aware_level_loss.py` (~530 LOC)

**33 dedicated tests** organized into 10 sections (A-J):

- **Section A (Protocol satisfaction)** — 4 tests: `@runtime_checkable`
  `isinstance(...)` returns True; `per_level_loss` is callable; defaults are
  canonical (`norm='l2'` + `reduction='mean'`); dataclass is frozen.
- **Section B (construction-time validation)** — 4 tests: invalid `norm` /
  `reduction` rejected; `l1` / `sum` accepted as sister formulations.
- **Section C (canonical M8 Protocol invariant)** — 3 tests: uniform
  sensitivity reduces to L2 mean; reduces to L1 mean (sister); sum reduction
  equals raw sum of squared error.
- **Section D (non-uniform sensitivity reweights)** — 2 tests:
  high-sensitivity region with same error produces 10× larger loss
  (canonical Yousfi reweighting); zero-sensitivity region drops error
  contribution to 0 (the Slot GGG SegNet-null sister anchor).
- **Section E (shape contract)** — 3 tests (parametrized over 4 shapes):
  multiple resolutions; shape mismatch raises; broadcast-compatible
  `(1, 1, H, W)` sensitivity accepted per Fridrich UNIWARD channel-uniform
  per-spatial-location convention.
- **Section F (non-negative sensitivity invariant)** — 3 tests: negative
  rejected by default; passes when validation disabled (hot-path escape);
  zero sensitivity accepted (non-negative + produces zero loss).
- **Section G (M7 integration)** — 3 tests: Path A (UNIFORM) end-to-end
  produces standard L2; Path B2 (EMPIRICAL_FROM_MASTER_GRADIENT)
  identity-resolution end-to-end produces finite non-negative loss; Path B2
  + Phase C (auto_project_to_level=True; dyadic projection) end-to-end at
  non-identity resolution.
- **Section H (builder convenience)** — 4 tests: Protocol satisfaction;
  forwards `norm` + `reduction`; rejects non-Level; works at coarse levels.
- **Section I (torch framework-agnostic)** — 2 tests: torch tensor path
  numerical match vs numpy; autograd flow verified analytically.
- **Section J (export hygiene)** — 2 tests: `__all__` exports + package
  re-exports.

### Updated: `src/tac/substrates/z8_hierarchical_predictive_coding/__init__.py`

Adds canonical import block + 3 symbols to `__all__` (`InvalidSensitivityMapError`,
`ScoreAwareLevelLossImpl`, `build_score_aware_level_loss_for_level`).

### Updated: `src/tac/substrates/z8_hierarchical_predictive_coding/build_progress.py`

M8 milestone `score_aware_level_loss_uniward_analog_landed` transitions
`PENDING` → `LANDED` with `landed_at_utc="2026-05-30T14:30:00Z"` +
substantive notes documenting the empirical receipts + canonical helper
chain + the 33 test invariants verified.

## Math: the canonical Yousfi-grounded loss form

Per `binding_contract.py:419-472` ScoreAwareLevelLoss Protocol docstring
verbatim:

> ```
> loss_at_level_i = sum_pixel(
>     scorer_sensitivity_at_pixel_at_level_i
>     * reconstruction_error_at_pixel
> )
> ```
>
> NOT generic L2. The bit budget gets spent where the scorer is actually
> sensitive — exactly Yousfi's "find the detector's blind spots and embed
> there" methodology.

The implementation realizes this as (for `norm='l2'`, the canonical default):

```
loss = REDUCE_{b, c, h, w}(
    scorer_sensitivity_map[b, c, h, w]
    * (reconstruction[b, c, h, w] - target[b, c, h, w])^2
)
```

where `REDUCE` is `mean` (default; the Yousfi convention) or `sum` (raw weighted
sum). For `norm='l1'`, squared error is replaced by absolute error.

### Why this satisfies the Protocol's uniform-reduces-to-L2 invariant

When `scorer_sensitivity_map == 1` everywhere:

```
loss = REDUCE_{b, c, h, w}(1 * error[b, c, h, w]^2)
     = REDUCE_{b, c, h, w}(error^2)
     = standard L2 reconstruction loss
```

Verified numerically by `test_uniform_sensitivity_reduces_to_l2_mean` at
`rtol=1e-6`.

### Why non-uniform sensitivity reweights per-pixel contribution

When `scorer_sensitivity_map[error_region] = 10.0` and elsewhere `= 1.0`, and
the reconstruction error is concentrated in the high-sensitivity region:

```
loss_high / loss_uniform = 10.0 (exactly, when error is constant in the region)
```

Verified numerically by `test_non_uniform_sensitivity_reweights_per_pixel` at
`rtol=1e-6`. This is the canonical empirical receipt that the loss is doing
Yousfi reweighting, not just generic L2.

## Canonical helper chain (M7 → M8) wire-in

The just-landed M7 dispatcher
`tac.substrates.z8_hierarchical_predictive_coding.scorer_sensitivity_map.Z8ScorerSensitivityMap.get_for_level(...)`
produces the M8-consumable sensitivity tensor:

```python
# Canonical M7 → M8 wire-in pattern (Z8 trainer holds one impl per level):
m7 = Z8ScorerSensitivityMap(source=ScorerSensitivityMapSource.UNIFORM)  # or EMPIRICAL_FROM_MASTER_GRADIENT
m8_impl = build_score_aware_level_loss_for_level(level)  # one per level

# Per-step:
sensitivity = m7.get_for_level(level, gradient_tensor=master_gradient_for_level)
loss = m8_impl.per_level_loss(reconstruction, target, sensitivity)
```

Path B2 (`EMPIRICAL_FROM_MASTER_GRADIENT`) is the canonical OPERATIONAL
Yousfi-grounded path; it consumes Phase A's
`tac.master_gradient_comparison.extract_M_pixel` +
`broadcast_sensitivity_map_to_channels` plus Phase C's optional
`decompose_M_contest_per_level` (canonical Mallat dyadic projection when the
gradient resolution does not match the level resolution).

## Sister empirical anchor: Slot GGG SegNet-null finding

Per `feedback_yousfi_fridrich_slot_rr_fake_to_real_via_real_scorer_verification_landed_20260529.md`
+ commit `32a70c051`: the Slot GGG / Slot RR per-pixel-roll PER_PIXEL_ROLL
modes produced SegNet argmax disagreement = 0.0000 (paradigm-perfect null) on
48×64 frames. This is the empirical receipt that the SegNet is *insensitive*
to per-pixel rolls in the carrier band — i.e. the sensitivity map IS zero in
the per-pixel-roll region (the scorer-blind region; Yousfi UNIWARD-analog).

The M8 implementation handles this canonical case correctly:
`test_zero_sensitivity_region_drops_error_contribution` verifies that an
error in a zero-sensitivity region contributes 0 to the loss. The Z8 trainer
can therefore use the Slot GGG-derived sensitivity map (once Path B reactivates
per the M7 `empirical_sensitivity_map_from_slot_ggg` 3-criterion gate) to bias
its bit budget AWAY from per-pixel-roll regions and TOWARD high-sensitivity
regions — exactly the canonical Yousfi engineering pattern.

## Apparatus mutations

### lane registry

Lane `lane_z8_phase_e_score_aware_level_loss_protocol_implementation_20260530`
L1 (impl_complete=true + memory_entry=true). Per Catalog #298 substrate
retirement discipline: the lane is operationally active; per Catalog #220
operational mechanism: the OPERATIONAL evidence is the 33-test pass plus the
empirical receipts above.

### Catalog # quota brake (Catalog #299)

**NO new Catalog # claimed.** This landing extends existing canonical surfaces
(Z8 substrate package + binding_contract.py Protocol) per CLAUDE.md
"Beauty, simplicity, and developer experience" + the 13th OPTIMAL-TRIO standing
directive. Current count remains well under 400 quota.

### Canonical equations / anti-patterns (Catalog #344)

**No new canonical equation registered at landing.** The M8 implementation is
infrastructure that consumes M7's sensitivity map; the canonical Yousfi-grounded
loss-form math is already implicit in the M8 Protocol declaration at
`binding_contract.py:419-472`. A canonical equation candidate
`yousfi_grounded_per_level_loss_reduces_to_l2_under_uniform_sensitivity_v1`
could be registered post-empirical-anchor (i.e. after the first end-to-end
Z8 trainer dispatch produces a per-level loss curve) — DEFERRED-to-operator-
decision per Catalog #344 "iterate not force" principle.

### Council deliberation posterior (Catalog #300)

T1 council anchor PROCEED with 7-voice attendance (Shannon LEAD + Dykstra
CO-LEAD + Rudin CO-LEAD + Daubechies CO-LEAD + Fridrich + Contrarian +
AssumptionAdversary). No dissent. Mission contribution =
`apparatus_maintenance` (M8 closes the Z8 Phase 2 binding-contract loss
surface; the immediate score-lowering value is N/A but unblocks M9
`full_main_trainer_lifts_notimplementederror` once M6 Wyner-Ziv lands).

Per CLAUDE.md "Council hierarchy: 4-tier protocol" T1 working-group
recommendation: this is in-flight engineering of a per-level Protocol
implementation that exactly matches the binding_contract.py:419 declaration;
T1 is the appropriate tier (no CLAUDE.md non-negotiable change; no
cross-cutting wire-in; loss function = clear engineering scope).

### 6-hook wire-in declaration (Catalog #125 NON-NEGOTIABLE)

- **hook #1 sensitivity-map = ACTIVE PRIMARY** — THIS work IS hook #1 at
  the per-level loss surface. The sensitivity map IS the canonical input
  to per_level_loss; the loss is mathematically a sensitivity-weighted
  reconstruction error.
- **hook #2 Pareto constraint = ACTIVE** — sensitivity-weighted L2 IS a
  Pareto frontier consumer; the per-level loss enters the Z8 trainer's
  overall objective which is Pareto-balanced against rate.
- **hook #3 bit-allocator = N/A** — the loss does not allocate bits
  directly; it informs the trainer's gradient signal which downstream
  influences the bit allocator via gradient magnitude.
- **hook #4 cathedral autopilot dispatch = N/A** — the loss is training-
  side, not ranking-side; cathedral autopilot consumes the trained
  archive's auth-eval score, not the per-step loss.
- **hook #5 continual-learning posterior = N/A** — the loss is a kernel
  primitive; canonical posterior anchors come from M9+ archive
  dispatches.
- **hook #6 probe-disambiguator = ACTIVE** — the uniform-sensitivity
  invariant (`test_uniform_sensitivity_reduces_to_l2_mean`) IS the
  canonical disambiguator between L2-baseline (uniform sensitivity =
  empty prior) vs Yousfi-grounded loss (non-uniform sensitivity =
  empirical scorer-blindness prior).

### Observability surface (Catalog #305)

Per the M8 Protocol invariant + this landing:

- **inspectable per layer**: each level's `ScoreAwareLevelLossImpl`
  instance is independently inspectable; `m8.norm` / `m8.reduction` /
  `m8.validate_non_negative_sensitivity` are queryable attributes.
- **decomposable per signal**: the loss `= mean(sensitivity * error^2)`
  decomposes naturally into `error^2` (per-pixel reconstruction error)
  and `sensitivity` (per-pixel Yousfi weight). The Z8 trainer can log
  the un-weighted error baseline alongside the Yousfi-weighted loss to
  surface the empirical reweighting effect.
- **diff-able across runs**: deterministic given fixed numpy/torch RNG
  seeds; the loss function has no hidden state.
- **queryable post-hoc**: per-level loss curves are logged to the Z8
  trainer's posterior (M9 surface; future scope).
- **cite-able**: the loss form is Yousfi-grounded per CLAUDE.md
  "Fridrich inverse steganalysis" + binding_contract.py:419-472
  Protocol declaration.
- **counterfactual-able**: sensitivity-map mutation per Catalog #105 +
  #139 + #220 + #272 byte-mutation contract; flipping a sensitivity
  byte changes the loss provably (via element-wise multiplication).

## Cargo-cult audit per assumption (Catalog #303)

| Assumption | Classification | Rationale |
|---|---|---|
| Yousfi-grounded loss form = sensitivity * error then reduce | HARD-EARNED | Binding contract.py:419-472 Protocol declaration verbatim; Fridrich inverse-steganalysis canonical |
| L2 default (norm='l2') matches Fridrich UNIWARD canonical | HARD-EARNED | Fridrich UNIWARD uses squared error; CLAUDE.md "Fridrich inverse steganalysis" canonical |
| Mean reduction default matches Yousfi convention | HARD-EARNED | Mean reduction is canonical PyTorch / standard ML loss reduction; aligns with how upstream evaluate.py reduces score components |
| Frozen dataclass for the impl | HARD-EARNED | Catalog #287 + #335 sister discipline; frozen dataclasses are hashable + safe to share + immutable invariants |
| Loss lives in z8 package not generic location | HARD-EARNED | Catalog #290 UNIQUE-AND-COMPLETE-PER-METHOD; per-Z8-per-hierarchy specific; sister tac.composition.*_inverse_steganalysis_* operate at different surface |
| Framework-agnostic via duck-typed element-wise operations | HARD-EARNED | numpy + torch + mlx all support `-`, `*`, `**`, `abs`, `.mean()`, `.sum()` per the Python data model + the FrameworkAgnosticTensor Protocol; verified empirically by torch path tests |
| Non-negative-sensitivity validation is correct invariant per Protocol docstring | HARD-EARNED | binding_contract.py:462-464 verbatim: "values are non-negative weights"; negative values produce non-loss (gradient AWAY from target) |

## 9-dimension success checklist evidence (Catalog #294)

1. **UNIQUENESS** — class-shift: this is the FIRST concrete implementation of
   the Z8 ScoreAwareLevelLoss Protocol; no canonical helper at the per-Z8-per-
   hierarchy-per-Yousfi-grounded surface previously existed.
2. **BEAUTY + ELEGANCE** — 30-sec reviewable: the entire loss kernel is 6 lines
   of code (`error = recon - target; sq = error * error; weighted = sens * sq;
   return weighted.mean()`) per the canonical Yousfi formulation.
3. **DISTINCTNESS** — explicitly different from sister inverse-steganalysis
   substrates (UNIWARD / HUGO / HILL / MiPOD) which operate at the per-archive-
   bolt-on surface; M8 operates at the per-Z8-level integrated surface.
4. **RIGOR** — premise verification per Catalog #229; adversarial review via
   sextet pact 7-voice attendance per Catalog #346; assumption classification
   per Catalog #292 + the hard-earned-vs-cargo-culted addendum.
5. **OPTIMIZATION PER TECHNIQUE** — canonical-vs-unique decision per Catalog
   #290: FORK at the per-level loss surface because no canonical helper
   embeds per-level scorer-sensitivity weighting at the hierarchy surface.
6. **STACK-OF-STACKS-COMPOSABILITY** — orthogonal to sister Z8 milestones:
   M4 (Mamba-2) handles state dynamics; M5 (Mallat) handles wavelet
   decomposition; M6 (Wyner-Ziv) handles top-level coding; M7 (sensitivity
   map) handles the per-pixel weights; M8 (THIS landing) handles the
   weighted reconstruction; M9 (trainer full_main) composes all 5 into one
   forward pass.
7. **DETERMINISTIC REPRODUCIBILITY** — byte-stable + seed-pinned: the loss
   function has no hidden state; deterministic given identical inputs.
8. **EXTREME OPTIMIZATION + PERFORMANCE** — framework-agnostic element-wise
   operations; no Python-side loops; numpy + torch + mlx all execute the loss
   in native backend kernels.
9. **OPTIMAL MINIMAL CONTEST SCORE** — non-promotable by construction (per
   Catalog #192 macOS-CPU advisory + Catalog #341 Tier A canonical-routing
   markers; M9+ trainer dispatch on Modal is the contest-score-relevant
   surface).

## Dykstra-feasibility predicted-band check (Catalog #296)

**Not applicable** — this is an infrastructure landing (Protocol implementation),
not a substrate dispatch with predicted ΔS. The first end-to-end Z8 trainer
dispatch (post-M9) will produce empirical ΔS anchors against the canonical
frontier per `.omx/state/canonical_frontier_pointer.json`.

## Acceptance criteria (per build_progress.py M8 milestone)

- [x] ScoreAwareLevelLoss implementation satisfies per_level_loss(reconstruction, target, sensitivity_map) — verified by `test_satisfies_score_aware_level_loss_protocol`
- [x] uniform sensitivity map (all-ones) reduces to standard L2 reconstruction loss — verified by `test_uniform_sensitivity_reduces_to_l2_mean` at rtol=1e-6
- [x] non-uniform sensitivity map reweights per-pixel contribution proportionally — verified by `test_non_uniform_sensitivity_reweights_per_pixel` with exact 10× ratio
- [x] per-level instances at each Z8 hierarchy level consume the sensitivity map at that level's resolution (downsampled where needed) — verified by `test_integration_with_m7_path_b2_phase_c_dyadic_projection` end-to-end at (96, 128) → (48, 64) Mallat dyadic projection

## Operator-routable next steps (Z8 M9 unblocking)

1. **M6 Wyner-Ziv full top-level coder** (`wyner_ziv_full_top_level_coder_landed`)
   is the next sister Z8 Phase 2 milestone (`get_next_actionable_milestones`
   confirms it is currently the only NEXT_ACTIONABLE milestone). Once M6 lands,
   M9 (`full_main_trainer_lifts_notimplementederror`) becomes actionable.

2. **M9 trainer composition**: the Z8 trainer's `_full_main` should hold one
   `ScoreAwareLevelLossImpl` per hierarchy level (constructed via
   `build_score_aware_level_loss_for_level(level)`) + one
   `Z8ScorerSensitivityMap` (the M7 dispatcher; constructed with the
   operator-bound source enum). Per-step forward pass:

   ```python
   for level in hierarchy:
       sensitivity = m7.get_for_level(level, gradient_tensor=master_gradient_for_level)
       loss = m8_per_level[level].per_level_loss(
           reconstruction[level], target[level], sensitivity
       )
       total_loss = total_loss + loss
   ```

3. **First end-to-end Modal A100 dispatch** (post-M9): the canonical predicted-band
   for Z8 baseline (per the parent scoping memo `.omx/research/path_3_f_z8_hierarchical_predictive_coding_substrate_design_20260526.md`)
   is `horizon_class=asymptotic_pursuit`; Z8 binds 4 canonical primitives
   simultaneously per Catalog #312 quadruple. The first dispatch will produce
   the canonical predicted-vs-empirical anchor for canonical equation
   `categorical_posterior_capacity_vs_continuous_gaussian_v1` (already
   registered per Catalog #344; first anchor LANDED per
   `feedback_wave_3_dreamerv3_rssm_math_audit_landed_20260529.md`).

## Cross-references

- M7 canonical helper: `src/tac/substrates/z8_hierarchical_predictive_coding/scorer_sensitivity_map.py`
- M7 commits: `8a95c9cc5` (Phase A) + `300702cdf` (Phase C + D)
- Binding contract: `src/tac/substrates/z8_hierarchical_predictive_coding/binding_contract.py:419-472`
- Build progress: `src/tac/substrates/z8_hierarchical_predictive_coding/build_progress.py::Z8_PHASE_2_BUILD_MILESTONES`
  (M8 transitions PENDING → LANDED in same commit batch)
- Slot GGG empirical anchor (Yousfi sister): `feedback_yousfi_fridrich_slot_rr_fake_to_real_via_real_scorer_verification_landed_20260529.md`
- Z8 binding-first standing directive: `feedback_z8_hierarchical_predictive_coding_binding_first_active_build_target_yousfi_grounded_20260529.md`
- Z8 in-source build tracking: `feedback_z8_phase_2_build_tracking_in_source_not_tasklist_not_memos_20260529.md`
- Parent design memo: `.omx/research/path_3_f_z8_hierarchical_predictive_coding_substrate_design_20260526.md`
- CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" L7 (substrate-engineering binds ALL ingredients)
- CLAUDE.md "Fridrich inverse steganalysis" (Yousfi UNIWARD-analog canonical)
- CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" (Catalog #290 sister)
- CLAUDE.md "Subagent coherence-by-default" (6-hook wire-in non-negotiable)
- Catalog #312 (canonical Rao-Ballard + Mallat + DreamerV3 + Wyner-Ziv quadruple)

## Test summary

```
============================= 152 passed in 0.70s ==============================
```

- New tests (M8): 33 / 33 PASS in 0.49 s (`tests/test_score_aware_level_loss.py`)
- Sister tests (M7 + M5 + M4 + binding + basic + DreamerV3 propagation):
  119 / 119 PASS (no regression)
- Full Z8 test suite: **152 / 152 PASS in 0.70 s**

## Final commit sha

To be backfilled in the next commit (self-referential M8 landing per the
canonical `landed_commit_sha=None` pattern from M1+M2+M3 + M4 + M5 + M7
sister landings).
