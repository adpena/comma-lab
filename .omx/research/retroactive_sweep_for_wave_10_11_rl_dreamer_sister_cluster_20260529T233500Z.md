# Retroactive sweep — Wave 10/11 RL + DreamerV3 sister cluster — 2026-05-29T23:35:00Z

Per Catalog #348 (`check_new_gate_landing_includes_retroactive_sweep_evidence`)
the 4-field contract requires every new gate landing OR new bug-class fix to
ship evidence that historical KILL / DEFER / FALSIFY verdicts whose evidence
basis was invalidated by the new fix have been re-evaluated.

## 1. Bug-class symptom signature

**The bug class**: a substrate's docstring claims canonical cross-substrate
reuse per Catalog #290 ADOPT_CANONICAL_BECAUSE_SERVES, but the implementation
is a local duplicate that diverges from the cited canonical helper. Future
canonical-helper fixes silently fail to propagate.

**Empirical anchor** (Wave 10/11 audit 2026-05-29):
- `src/tac/substrates/z8_hierarchical_predictive_coding/__init__.py:24-25`
  + design memo `path_3_f_z8_hierarchical_predictive_coding_substrate_design_20260526.md`
  both claim sister DreamerV3 categorical-posterior reuse per Catalog #290.
- `src/tac/substrates/z8_hierarchical_predictive_coding/mlx_renderer.py:195-227`
  (pre-fix) contained a local duplicate of `gumbel_softmax_sample` with
  `mx.random.uniform` + Gumbel-perturbation math directly inline, lacking
  the `unimix_alpha` parameter entirely.
- Sister DreamerV3 Wave 3 fix 2026-05-29 added Hafner 2023 §3 1% unimix
  robustness mixture to `tac.substrates.dreamer_v3_rssm.gumbel_softmax_sample`
  — but the fix did NOT propagate to Z8 because Z8 was a duplicate.

## 2. Pre-fix window

- Z8 L0 scaffold landed via Catalog #312 canonical quadruple wave (sister A
  DreamerV3 reuse claim explicit in design memo + module docstring).
- Sister DreamerV3 Wave 3 unimix fix landed 2026-05-29 commit `5e18c0bbf`.
- Pre-Wave-10/11 window: 2026-05-26 (Z8 L0 scaffold landing) through
  2026-05-29 (THIS landing closes the gap).

## 3. Historical KILL / DEFER / FALSIFY search results

Searched `.omx/research/` + canonical posterior + canonical anti-patterns for
verdicts whose evidence basis was the pre-fix Z8 categorical-posterior path:

- **NO** historical KILL verdict on Z8 categorical-posterior surface (Z8 is
  L0 SCAFFOLD with `research_only=true`; never reached paid-GPU eval per
  Catalog #240).
- **NO** historical DEFER on Z8 unimix omission specifically (the gap
  surfaced only via Wave 10/11 audit; prior cargo-cult audits at
  `path_3_b_z7_mamba_2_cargo_cult_audit_of_existing_scaffold_20260526.md`
  enumerated Z7-Mamba-2 cargo-cults, NOT Z8 unimix omission).
- **NO** historical FALSIFY on Z8 (no empirical anchors landed pre-Wave-10/11
  because the substrate has `_full_main raises NotImplementedError`).

**Sister context**: DreamerV3 Wave 3 fix DID invalidate the prior 2 anchors
on canonical equation `categorical_posterior_capacity_vs_continuous_gaussian_v1`
by adding the unimix mixture (the 2 anchors predate the fix). The Wave 3
landing already documented this; Wave 10/11 inherits the same status for
the Z8 propagation surface.

**Conclusion**: ZERO historical verdicts invalidated by this fix. The Wave
10/11 closure is a forward-only structural extinction that prevents future
cargo-cult-duplicate-helper recurrences at the Z8 surface and creates a
delegation pattern other sister substrates can follow.

## 4. RE-EVAL-priority assignment per affected historical finding

N/A — no historical findings invalidated.

**Forward-looking RE-EVAL queue** (operator-routable, not Wave 10/11 scope):

1. **Z8 Phase 2 lift** (operator-attended): when the Z8 substrate council
   approves lifting `_full_main raises NotImplementedError` per Catalog #240,
   the per-level categorical posterior MUST be paired-CUDA verified against
   sister DreamerV3 single-level reference at α=0.01 vs α=0.0 ablation.
   Priority: queued behind sister Compound C/D heterogeneous bit-allocation
   landings.
2. **Other sister substrates citing Catalog #290 ADOPT_CANONICAL_BECAUSE_SERVES**:
   audit them for the same docstring-vs-implementation divergence pattern.
   Sister Z6, Z7-Mamba-2 (variants), and Path 3 candidate G=NIRVANA all
   reference sister canonical primitives in their design memos. Priority:
   schedule as a future sister-audit wave (NOT Wave 10/11 scope; would
   collide with in-flight Wave 9 NSCS06 v8 surface per Catalog #340).

## 5. Canonical structural-protection summary

- **Catalog #287** placeholder-rationale rejection — discipline already
  active; no extension needed.
- **Catalog #290** UNIQUE-AND-COMPLETE-PER-METHOD operating mode — the
  delegation pattern IS the canonical implementation of "canonical helpers
  used WHEN they serve" per the falling-rule list.
- **Catalog #335** auto-discovery + canonical contract — already protects
  the cathedral-consumer surface; the delegation pattern preserves the
  contract.
- **Catalog #344** canonical equations registry — the 3rd anchor on
  `categorical_posterior_capacity_vs_continuous_gaussian_v1` crosses
  the 3+ trigger threshold for Catalog #371 auto-recalibration.
- **Catalog #371** auto-recalibrator — will fire next preflight scan and
  re-derive `predicted_vs_empirical_residual` for the canonical equation
  from all 3 anchors.

## 6. mission_predicted_contribution per Catalog #300

`apparatus_maintenance` — closes a docstring-vs-implementation divergence
structurally so future sister-canonical fixes propagate via delegation
rather than requiring per-sibling copy-paste maintenance.

## 7. Cross-references

- Wave 10/11 landing memo (THIS commit batch):
  `.omx/research/wave_10_11_rl_dreamer_sister_cluster_landed_20260529.md`
- Wave 3 DreamerV3 audit (sister fix that didn't propagate):
  `.omx/research/wave_3_dreamerv3_rssm_math_fidelity_audit_landed_20260529.md`
- Wave 4 Z7-Mamba-2 audit (RL-cluster sister):
  `.omx/research/wave_4_z7_mamba_2_dao_gu_fidelity_audit_landed_20260529.md`
- Z7-Mamba-2 L2 stability hardening:
  `.omx/research/z7_mamba_2_v2_l2_stability_hardening_landed_20260526.md`
