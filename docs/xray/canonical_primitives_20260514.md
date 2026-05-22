# Canonical xray primitives (2026-05-14)

Lane: `lane_xray_canon_math_findings_wire_in_20260514`

The `tac.xray.*` package landed 13 canonical primitives during XRAY-CANON-WIRE-IN
(commits `bef68c270` + `fc266c43e` + `ca21c497c` + `b62f3b872` + `caefbe1b3` +
`3b86b4e00`). The DQS1 pairset component-marginal follow-up adds the 14th
primitive, `pairset_component_marginal`, so exact auth-eval component deltas
are discoverable by the same xray wire-in surface. This document is the
canonical operator/subagent reference
for the inventory, the 6-hook wire-in surface, and the cross-primitive
composition contract.

## What an xray primitive is

Per `tac.xray.base`:

> Xray primitives ANALYZE an existing archive, substrate, or scorer to
> extract a sensitivity / bound / coverage / margin report that the
> solver stack consumes (sensitivity-map, Pareto, bit-allocator,
> autopilot, continual-learning, probe-disambiguator).

They are DISTINCT from `tac.composition.registry` PACKET-COMPILER primitives
(PR101 GOLD, sign-encoding, Cheng-2020, etc.), which produce ARCHIVE BYTES.
Xray primitives only PRODUCE EVIDENCE — they never emit score claims and
they never write archive bytes.

Every xray primitive returns an `XRayPrimitiveResult` carrying a typed
`primitive_value` (dataclass), an `evidence_grade` (one of
`mathematical-derivation` / `first-principles-bound` / `empirical-anchor` /
`proxy` / `council-deliberation` / `structural-code-contract`), and a
`wire_in_hooks_engaged` tuple declaring which of the 6 canonical hooks the
result feeds.

## The 6 canonical wire-in hooks

Per CLAUDE.md "Subagent coherence-by-default" NON-NEGOTIABLE, every xray
primitive declares which of these 6 hooks it engages. Silent omission is
the orphan-work failure mode.

1. **sensitivity_map** (`tac.sensitivity_map.*`) — per-tensor importance / per-bit sensitivity. 12 primitives engage this hook.
2. **pareto_constraint** (`tac.pareto_*` / `tac.optimization.field_equation_planner`) — rate-distortion / feasibility constraints. 4 primitives engage.
3. **bit_allocator** (`tac.optimization.bit_allocator_end_to_end`) — per-tensor bit budget. 11 primitives engage.
4. **cathedral_autopilot** (`tac.optimization.autopilot_dispatch_ranking`) — top-K candidate ranking for paid GPU dispatch. 4 primitives engage.
5. **continual_learning** (`tac.continual_learning`) — posterior update with empirical anchors. 2 primitives engage.
6. **probe_disambiguator** — competing-interpretation arbitration (`tools/probe_<track>_disambiguator.py`). 10 primitives engage.

Total hook-primitive engagements across the 14-primitive inventory: **43**.

## The 14 canonical primitives

The inventory is loaded from `tac.xray.registry.canonical_xray_primitive_inventory()`.

| F# | name | category | evidence | hooks | composes_with |
|---:|------|----------|----------|-------|---------------|
| F1 | `mdl_scorer_conditional` | information-geometric | mathematical-derivation | continual_learning, probe_disambiguator, cathedral_autopilot | shannon_vector_r_d, per_pair_score_decomposition |
| F2 | `shannon_vector_r_d` | information-theoretic | mathematical-derivation | pareto_constraint, sensitivity_map, probe_disambiguator | mdl_scorer_conditional, score_lipschitz |
| F3 | `bilinear_resize_nullspace` | scorer-preprocessing | mathematical-derivation | sensitivity_map, bit_allocator, probe_disambiguator | yuv6_sublattice_geometry |
| F4 | `score_lipschitz` | continuity-bound | mathematical-derivation | pareto_constraint, bit_allocator, probe_disambiguator | shannon_vector_r_d, segnet_margin_polytope |
| F5 | `vq_codebook_coverage` | codec-axis | mathematical-derivation | sensitivity_map, bit_allocator, probe_disambiguator | wavelet_hf_energy, yuv6_sublattice_geometry |
| F6 | `wavelet_hf_energy` | codec-axis | mathematical-derivation | sensitivity_map, bit_allocator | vq_codebook_coverage |
| F7 | `segnet_margin_polytope` | scorer-internal | mathematical-derivation | sensitivity_map, bit_allocator, probe_disambiguator | score_lipschitz, posenet_se3_lie_algebra |
| F8 | `posenet_se3_lie_algebra` | scorer-internal | mathematical-derivation | sensitivity_map, probe_disambiguator | segnet_margin_polytope |
| F9 | `per_pair_score_decomposition` | task-decomposition | mathematical-derivation | sensitivity_map, bit_allocator, cathedral_autopilot | mdl_scorer_conditional, unified_action_principle |
| F10 | `yuv6_sublattice_geometry` | scorer-preprocessing | mathematical-derivation | sensitivity_map, bit_allocator | bilinear_resize_nullspace |
| F11 | `unified_action_principle` | variational-principle | mathematical-derivation | pareto_constraint, sensitivity_map, bit_allocator, cathedral_autopilot | shannon_vector_r_d, per_pair_score_decomposition |
| F12 | `predictive_coding_hierarchy` | temporal-axis | mathematical-derivation | sensitivity_map, bit_allocator, probe_disambiguator | foveation_ego_motion |
| F13 | `foveation_ego_motion` | temporal-axis | mathematical-derivation | sensitivity_map, bit_allocator, probe_disambiguator | predictive_coding_hierarchy |
| F14 | `pairset_component_marginal` | scorer-internal | empirical-anchor | sensitivity_map, pareto_constraint, bit_allocator, cathedral_autopilot, continual_learning, probe_disambiguator | per_pair_score_decomposition, segnet_margin_polytope, posenet_se3_lie_algebra, score_lipschitz |

## Cross-primitive composition pairs (verified by integration tests)

Per `src/tac/xray/tests/test_compositional_integration.py`:

- **F1 × F2 (MDL × Shannon)** — `r_min_bytes` (F2) is the information-theoretic
  lower bound; `total_archive_bytes` (F1) MUST exceed it. Verified on A1.
- **F3 × F10 (resize-nullspace × YUV6-sublattice)** — Together they bound
  the TOTAL nullspace of the scorer preprocessing pipeline (resize then YUV6).
  Both feed `sensitivity_map` + `bit_allocator`.
- **F7 × F8 (SegNet polytope × PoseNet Lie-algebra)** — Per-scorer margin
  geometry. Both feed `sensitivity_map` + `probe_disambiguator`.
- **F5 × F6 (VQ codebook × Wavelet HF)** — Codec-axis pair informing the
  bit-allocator on WHERE in the archive structure to route bits.
- **F4 × F2 (Lipschitz × Shannon R(D))** — Pareto-constraint pair; both
  carry `confidence_band` for trust-region construction.
- **F9 × F11 (per-pair × unified action)** — Cathedral autopilot pair; F9
  yields top-K pairs by marginal contribution, F11 yields the unified action
  scalar.
- **F12 × F13 (predictive coding × foveation)** — Temporal-axis pair; both
  feed `sensitivity_map` + `bit_allocator` for temporal redundancy.
- **F14 × F9 × F7 × F8 (component marginal × per-pair × scorer geometry)** —
  Exact auth-eval component deltas explain when one-byte rate credit is
  dominated by SegNet/PoseNet penalty, and route safe/protected pairs into the
  master-gradient consumers and candidate planner.

## How to consume the wire-in surface

```python
from tac.xray import wire_in_for_hook, discover_primitives_by_hook

# Discover which primitives feed your hook.
discovered = discover_primitives_by_hook()
sensitivity_primitives = discovered["sensitivity_map"]  # 12 names

# Run all primitives engaging a hook with per-primitive targets.
bundle = wire_in_for_hook(
    "pareto_constraint",
    targets={
        "shannon_vector_r_d": {
            "target": None,
            "d_seg_target": 0.067,
            "d_pose_target": 0.018,
        },
        "score_lipschitz": {"target": b"\x00" * 16},
    },
)
# bundle.results: tuple of XRayPrimitiveResult, one per primitive that ran.
# bundle.skipped_primitives: list of (name, reason) for primitives whose
#   target was missing or whose compute raised.
```

## Solver-stack adapters

The 4 major solver-stack consumer surfaces:

- `tac.sensitivity_map.*` — 12 primitives wire in.
- `tac.optimization.bit_allocator_end_to_end` — 11 primitives wire in.
- `tac.optimization.autopilot_dispatch_ranking` — 4 primitives wire in.
- `tac.optimization.field_equation_planner` — 4 primitives wire in.
- `tac.continual_learning` — 2 primitives wire in.

Each consumer's adapter iterates `wire_in_for_hook(hook, targets)` and
consumes `XRayPrimitiveResult` rows in the canonical format — no consumer-side
primitive-specific code required.

## Cross-references

- **Master math memo**: `.omx/research/deep_math_geometry_manifolds_synthesis_20260514.md`
- **Zen-floor council**: `.omx/research/zen_floor_field_medal_grade_council_20260514.md`
- **Subagent coherence non-negotiable**: `CLAUDE.md` § "Mandatory wire-in for every landing (no orphaned signals)"
- **HNeRV parity discipline**: `CLAUDE.md` § "HNeRV / leaderboard-implementation parity discipline"
- **Unified Lagrangian action principle**: `feedback_unified_lagrangian_action_principle_GR_style_20260509.md`

## Test coverage

- Per-primitive: 14 test files (one per primitive)
- Solver-stack integration: `test_integration_with_solver_stack.py` (16 tests)
- Compositional integration: `test_compositional_integration.py` (14 tests, this landing)
- Registry: `test_registry.py` (12 tests)
- Wire-in: `test_wire_in.py` (8 tests)

Current focused verification for this surface includes the registry,
solver-stack integration, compositional integration, and
`test_pairset_component_marginal.py` primitive tests. Re-run those tests after
any registry cardinality change.

## Canonical contract

Every xray primitive class:

1. Implements `XRayPrimitive` protocol from `tac.xray.base`.
2. Returns `XRayPrimitiveResult` from `compute(target, **kwargs)`.
3. Declares its `wire_in_hooks_engaged` tuple non-empty.
4. Cross-references its sister primitives via `composes_with` in the registry.
5. Carries an `upstream_memo` path pointing to its derivation source.
6. Never emits score claims (the contest scorer is the only authority).
7. Never reads/writes transient-only paths (no `/`tmp`/` references in persisted evidence per CLAUDE.md "Forbidden /tmp paths in any persisted artifact").

## Adding a new xray primitive

1. Implement the primitive class in `src/tac/xray/<name>.py`.
2. Add an `XRayPrimitiveSpec` to `canonical_xray_primitive_inventory()` in
   `src/tac/xray/registry.py` with non-empty `wire_in_hooks`.
3. Add a per-primitive test file in `src/tac/xray/tests/test_<name>.py`.
4. Verify all 6 hooks still have ≥1 engaging primitive via
   `test_compositional_integration::test_all_six_hooks_engaged_by_at_least_one_primitive`.
5. Update this document's primitive table.
6. Pre-register the lane in `.omx/state/lane_registry.json` per CLAUDE.md
   Catalog #126 before any work starts.
