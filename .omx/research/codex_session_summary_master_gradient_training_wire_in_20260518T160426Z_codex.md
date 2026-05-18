# Codex Session Summary — master-gradient training and pipeline wire-in — 2026-05-18T16:04:26Z

## Scope

- Lane: `lane_codex_master_gradient_training_pipeline_wire_in_20260518`.
- Inputs: xhigh adversarial subagents Volta and Popper, plus the new upstream procedural-generation compliance memo.
- Objective: convert per-pair master-gradient guidance from planning text into training/compress/inflate policy surfaces while preserving contest-axis custody.

## Landed

- Hardened `DeliverabilityProof.contest_compliance_rationale` so Tier-2 Wyner-Ziv/procedural-generation proofs carry the upstream loophole boundary: `upstream/evaluate.py:63`, `PR #68 loophole_v2`, `Catalog #213`, `Comma2k19LocalCache`, `INSIDE archive.zip`, `OUTSIDE archive.zip`, `score_claim=False`, and `promotion_eligible=False`.
- Added verifier enforcement: Tier-2 proofs with blank or weak rationale now fail `verify_deliverability_proof_contest_compliance`.
- Added `tac.training_curriculum.master_gradient_pair_weights` to derive bounded, mean-normalized pair weights from canonical per-pair master-gradient difficulty atlases.
- Added `tac.training_curriculum.per_pair_master_gradient_wire_in` and extended the canonical namespace wire-in set to include `training_curriculum`.
- Added opt-in `auto_per_pair_wire_in` policy threading to `ComposableCompressPipeline.run` and `ComposableInflatePipeline.run`, preserving default behavior and keeping inflate-time scorer-free.

## Verification

- `435 passed` across affected pipeline/package suites:
  - `src/tac/tests/test_tac_compress_time_optimization.py`
  - `src/tac/tests/test_tac_inflate_time_post_processing.py`
  - `src/tac/tests/test_training_curriculum.py`
  - `src/tac/training_curriculum/tests/test_training_curriculum.py`
  - `src/tac/tests/test_wyner_ziv_deliverability_proof_builder.py`
  - `src/tac/tests/test_low_gap_closure_widened_namespace_wire_ins.py`
  - `src/tac/tests/test_wave_package_import_surfaces.py::test_training_curriculum_import_surface_is_not_dangling`
- `py_compile` passed on edited implementation and test surfaces.
- `git diff --check` passed.
- `tools/lane_maturity.py validate` passed with 886 lanes.

## Remaining Risk

- The full trainer-level callsites still need substrate-specific adoption of `MasterGradientPairWeights.as_policy()` in their sampling/loss loops.
- Cathedral autopilot already consumes optimal-plan and per-pair sidecars; the next higher-EV bridge is to expose optimal-plan candidate rows as a direct candidate source in the ranker.
