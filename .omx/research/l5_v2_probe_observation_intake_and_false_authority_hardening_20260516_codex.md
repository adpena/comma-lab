# L5 v2 Probe Observation Intake + False-Authority Hardening

Date: 2026-05-16

Status: landed for L5 v2 / TT5L control plane. This memo records no score
claim, no promotion eligibility, and no ready-for-exact-eval-dispatch claim.

## What changed

- Added `tools/audit_l5_v2_probe_observations.py` and
  `tac.optimization.l5_v2_probe_intake`.
- Generated `.omx/research/l5_v2_probe_observation_intake_20260516_codex.json`
  and `.omx/research/l5_v2_probe_observation_intake_20260516_codex.md`.
- Rebuilt `.omx/research/l5_v2_probe_gate_artifact_20260516_codex.json` from
  the intake surface.
- Changed TT5L remote provenance to `predicted_band: null` plus
  `retired_predicted_band: [0.150, 0.170]` and
  `prediction_band_rank_reward_suppressed: true`.
- Backfilled the TT5L lane-registry row to remove the naked `predicted
  0.150-0.170` name.
- Replaced Z4/Z5 archive-meta naked `predicted_band_lo/hi` fields with a
  structured `prediction_band_verdict` object carrying `score_claim: false`,
  `promotion_eligible: false`, `ready_for_exact_eval_dispatch: false`, and
  `valid_for_rank_reward: false`.
- Added `verdict_authority_scope:
  score_axis_consistent_only_no_move_level_or_score_authority` to TT5L
  Dykstra artifacts and readiness validation.

## Current intake result

The intake found one TT5L `[contest-CUDA]` source cluster and no accepted
`[contest-CPU]` row for TT5L. It found C1 probe JSONs, but they are not exact
axis evidence. It found no Z5 exact probe artifact.

The rebuilt probe gate stays blocked:

- `architecture_lock_allowed: false`
- `score_claim: false`
- `promotion_eligible: false`
- `ready_for_exact_eval_dispatch: false`

Primary blockers remain: paired CPU/CUDA observations for C1, Z5, and TT5L;
side-info/usefulness predicate binding per candidate; per-axis score deltas;
and exact artifact/log/device custody.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_l5_v2_probe_intake.py
  src/tac/tests/test_l5_v2_probe_disambiguator.py
  src/tac/tests/test_l5_staircase_v2.py
  src/tac/tests/test_check_substrate_dykstra_feasibility.py
  src/tac/substrates/z4_cooperative_receiver_loss/tests/test_z4_substrate.py
  src/tac/substrates/z5_predictive_coding_world_model/tests/test_z5_substrate.py
  -q`
  - Result: 184 passed, 1 pre-existing warning.
- `.venv/bin/python -m ruff check
  src/tac/optimization/l5_v2_probe_intake.py
  tools/audit_l5_v2_probe_observations.py
  tools/check_substrate_dykstra_feasibility.py
  src/tac/tests/test_l5_v2_probe_intake.py
  src/tac/tests/test_check_substrate_dykstra_feasibility.py`
  - Result: all checks passed.

Full `ruff check` over legacy `src/tac/preflight.py` and older substrate test
files is not clean in the current worktree because unrelated Catalog #305 and
legacy lint surfaces are in flight. This landing did not stage those partner
changes.

## Next action

Populate real paired C1/Z5/TT5L observations or dispatch the smallest compliant
paired timing/probe jobs with lane claims. Do not grant architecture lock or
rank reward from the current intake.
