# L5 v2 asymptotic candidate surface refresh

Date: 2026-05-16

Scope: L5-v2 asymptotic candidate tracking for Z6/Rudin/Tishby.

This is a no-score-claim durability fix.

## Finding

The executable L5-v2 readiness code already recognized the actual Z6 L1
scaffold artifacts:

- `src/tac/substrates/time_traveler_l5_z6/`
- `experiments/train_substrate_time_traveler_l5_z6.py`
- `.omx/operator_authorize_recipes/substrate_time_traveler_l5_z6_modal_t4_dispatch.yaml`

The tracked `.omx/research/l5_v2_asymptotic_candidate_surface_20260516_codex.json`
was stale and still pointed at the older provisional names:

- `src/tac/substrates/z6_predictive_coding_world_model/`
- `experiments/train_substrate_z6_predictive_coding_world_model.py`
- `.omx/operator_authorize_recipes/substrate_z6_predictive_coding_modal_t4_dispatch.yaml`

That stale artifact made the operator-facing surface report
`expected_first_artifacts_all_present=false` and `ready_for_l1_build=true`,
even though the real Z6 L1 scaffold already exists. This was not a model
result, but it was a no-signal-loss failure in the control plane: completed
frontier work could be hidden behind a stale planning snapshot.

## Change

- Added `tools/build_l5_v2_asymptotic_candidate_surface.py`.
- Added canonical JSON/Markdown render helpers in
  `src/tac/optimization/l5_staircase_v2.py`.
- Regenerated the tracked asymptotic candidate surface JSON/Markdown from the
  live `l5_v2_asymptotic_pursuit_candidates()` payload.
- Added a regression test that compares the tracked `.omx` JSON artifact
  against the live payload for first-artifact paths, L1 scaffold presence, next
  action status, readiness booleans, and blockers.

## Current Authority

The refreshed surface now reports Z6, Rudin, and Tishby first artifacts as
present and their original first-build recommendations as
`completed_or_superseded`. It still does not authorize rank, kill, promotion,
paid dispatch, or exact-eval dispatch:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `ready_for_paid_dispatch=false`

Z6 remains blocked on the mechanism-level next steps:

- identity-predictor disambiguator before paradigm claim;
- paired CPU/CUDA anchor before score or rank authority;
- Phase-2/council path before lifting the research-only full-main gate.

## Verification Target

Run:

```bash
.venv/bin/ruff check \
  src/tac/optimization/l5_staircase_v2.py \
  src/tac/tests/test_l5_staircase_v2.py \
  tools/build_l5_v2_asymptotic_candidate_surface.py

.venv/bin/pytest \
  src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_asymptotic_pursuit_candidates_are_source_backed \
  src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_asymptotic_candidate_surface_artifact_tracks_live_payload \
  src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_asymptotic_candidate_surface_markdown_reports_current_status \
  src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_asymptotic_candidate_surface_cli_writes_json_and_markdown \
  -q
```

## Next Action

Do not rebuild the Z6 L1 scaffold again. Move to the next mechanism-bearing
step: identity-predictor disambiguator and paired CPU/CUDA evidence planning
for the existing Z6 scaffold, while preserving its research-only dispatch gate
until the Phase-2/council criteria are explicitly satisfied.
