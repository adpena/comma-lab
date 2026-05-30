# Retroactive sweep for Z8 M9 `_full_main` lift canonical quadruple binding-integration

Per Catalog #348 `check_new_gate_landing_includes_retroactive_sweep_evidence` 4-field contract.

## Bug-class symptom signature

**Bug class**: substrate trainer `_full_main` raises NotImplementedError
when canonical quadruple Protocol implementations (M4 Mamba-2 + M5 Mallat
full DWT + M6 Wyner-Ziv + M8 ScoreAwareLevelLoss) are all LANDED in sister
modules but never composed into the end-to-end forward pass — the
"all primitives built but never bound" failure mode per HNeRV parity
discipline L7 sub-class.

**Symptom signature**: trainer's `_full_main` raises NotImplementedError
OR routes through L0 SCAFFOLD that does NOT use the canonical Protocol
implementations from sister modules. The canonical compose pattern
(`m5.decompose -> m6.encode -> m8.per_level_loss`) is not structurally
executable end-to-end despite all 4 Protocol implementations existing.

**Empirical anchor**: pre-fix, `_full_main` in
`experiments/train_substrate_z8_hierarchical_predictive_coding_mlx.py`
routed through the canonical MLX harness using `Z8HierarchicalPredictiveCoderMLX`
(L0 SCAFFOLD renderer) but did NOT consume any of the canonical Protocol
implementations at `mamba2_adapter.py` / `mallat_dwt_adapter.py` /
`wyner_ziv_coder.py` / `loss.py` / `scorer_sensitivity_map.py`. The M9
milestone in `build_progress.py` was PENDING with all 4 predecessors
LANDED — structural readiness with no consumer.

## Pre-fix window

2026-05-29 (Z8 Phase 2 binding contract landed `506f57c13` defining
the per-level Protocols) through 2026-05-30T15:21 (this landing).

## Historical-KILL/DEFER/FALSIFY search

Searched `.omx/research/`, `~/.claude/projects/-Users-adpena-Projects-pact/memory/`,
and `.omx/state/probe_outcomes.jsonl` for prior verdicts that this landing
might retroactively invalidate.

**Verdict**: NO historical KILL / DEFER / FALSIFY verdicts found against
the Z8 M9 milestone or the canonical compose pattern. The Z8 substrate is
itself a Path 3 candidate F (Catalog #312 canonical quadruple) with no
prior false-positive kills. Sister Path 3 candidates A (DreamerV3) / D (Z6)
/ E (BoostNeRV) all LANDED separately; none had M9 binding-integration
verdicts that this landing changes.

**Sister Z8 commits scanned** (last 30 days):
- `5d5634dd3` M6 Wyner-Ziv LANDED — predecessor of M9 (no retroactive impact)
- `95b8c6336` M8 ScoreAwareLevelLoss LANDED — predecessor (no retroactive impact)
- `300702cdf` + `5a5311c00` + `8a95c9cc5` + `415e9035e` M7 cascade — predecessor (no retroactive impact)
- `5f74a50a0` M5 Mallat LANDED — predecessor (no retroactive impact)
- `4d567bf0b` M4 Mamba-2 LANDED — predecessor (no retroactive impact)
- `506f57c13` Phase 2 binding contract + tracking surface LANDED — root (no retroactive impact)

## Per-finding RE-EVAL-priority assignment

NONE. This landing is a binding-integration milestone that COMPOSES
sister-landed primitives; it does NOT introduce any new bug class or
invalidate any prior verdict. The canonical compose pattern is the
canonical M9 milestone definition per `build_progress.py`. Per Catalog
#307 paradigm-vs-implementation classification: this landing is
IMPLEMENTATION-LEVEL forward-progress (M9 LANDED), NOT paradigm-level
revision; no prior paradigm-level KILL/DEFER/FALSIFY needs re-evaluation.

## Cross-references

- Landing memo: `feedback_z8_m9_full_main_notimplementederror_lift_canonical_quadruple_binding_integration_landed_20260530.md`
- Build progress milestone: `src/tac/substrates/z8_hierarchical_predictive_coding/build_progress.py::Z8_PHASE_2_BUILD_MILESTONES['full_main_trainer_lifts_notimplementederror']`
- Canonical helper: `src/tac/substrates/z8_hierarchical_predictive_coding/canonical_quadruple_binding.py`
- Tests: `src/tac/tests/test_train_substrate_z8_canonical_quadruple_binding.py` (25 tests, all PASS)
- Empirical anchor: `experiments/results/z8_m9_full_main_macos_cpu_advisory_smoke_20260530T152144Z/m9_canonical_quadruple_artifact.json`

## Lane

`lane_z8_m9_full_main_notimplementederror_lift_canonical_quadruple_binding_integration_20260530` L1
(impl_complete + memory_entry per Catalog #298 substrate retirement
discipline + Catalog #294 9-dim checklist + Catalog #303 cargo-cult audit
+ Catalog #305 observability surface + Catalog #300 v2 frontmatter).
