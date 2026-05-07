# Field Acquisition Ranking Tranche - 2026-05-07

## Scope

This tranche adds a planning-only acquisition layer to the unified
field-equation planner. It does not create a candidate archive, run exact CUDA
auth eval, dispatch remote work, or claim a score.

## Implementation

- `tac.optimization.field_equation_planner.field_acquisition_ranking()` now
  emits per-atom acquisition rows.
- The acquisition score combines expected improvement, expected information
  gain, and negative variational-action pressure.
- Every row carries explicit dispatch blockers, score-claim state, Pareto/KKT
  readiness, byte-closure readiness, and custody readiness.

## Artifacts

Local ignored planning artifacts were generated under:

- `.omx/research/artifacts/field_equation_tranche_20260507_codex/field_equation_plan.json`
- `.omx/research/artifacts/field_equation_tranche_20260507_codex/field_equation_plan_from_prior_artifact.json`

Summary:

- `reports/cross_paradigm_atom_ledger_v3_20260506.json`: 39 atoms, 0 Pareto
  eligible, 0 KKT-ready, 0 design-ready. High-ranking LA-pose rows remain
  blocked by rankability, Pareto/KKT readiness, and missing byte-closed archive
  manifests.
- Prior curated field artifact: 39 atoms, 1 Pareto eligible, 1 KKT-ready,
  1 design-ready. The design-ready row is
  `wr01_wavelet_apply:pr106x:latents_and_sidecar_brotli`.

## Next Actions

1. Promote WR01 selected atoms from plan artifact to a reviewed runtime-apply
   and decode-validation manifest.
2. Give LA-pose atoms a byte-closed archive consumer and KKT-ready marginal
   records before ranking them as dispatchable.
3. Keep JCSP AQ/rawvideo work fail-closed until runtime consumption, parity,
   and exact-eval custody are complete.
