# DDM CE1 Receipt - censorship-sources remaining legs

## Answer first

CE1 cured ten top triaged censoring sites without scorer runs, launches, archive mutations, or scored-path behavior changes.

Subset-default cures:

- `experiments/ddm_p3v2_optimal_form_pose_resolve.py` now emits `pair_selection` for its default `--n-pairs=24` video-order prefix receipt.
- `experiments/ddm_pz1_dpose_window_solve_paired.py` now emits strided `pair_selection` plus the executed m88 `governing_ratio`.
- `experiments/ddm_pz1_partial_refit_bound.py` now emits strided `pair_selection`.
- `experiments/ddm_pz1_scorer_plane_pose_delta.py` now emits strided `pair_selection`.
- `experiments/diag_custom_vs_reference_trajectory_n8.py` now emits prefix `pair_selection` for the N8 diagnostic.

Silent-cap cure:

- `experiments/multi_pass_inflate_optimizer.py` now emits `cap_stop_receipt` through `tac.optimization.trajectory_stopping.build_cap_stop_receipt`.

Current post-patch denominators:

- Subset scan: 6,234 files, 595 subset-default sites, 21 `scope_reported`, 558 silent, 16 dormant, parse errors 0. Artifact: `.omx/research/ddm_ce1_20260805/ce1_subset_scope_inventory_20260805.json`.
- Cap scan: 6,234 files, 91 cap-default sites, 7 stop-reporting, 84 silent, parse errors 0. Artifact: `.omx/research/ddm_ce1_20260805/ce1_cap_inventory_20260805.json`.
- Vacuity family: not re-swept live in CE1; carried as scoped recall from VC1/SI1 only. Known recalled denominators include 18 `tools/check_gate*` shared-main rows, 12 prose-only `_finish(ok_detail=...)` emitters, 462 subprocess sites with 70 no-rc-check and 41 no fail-return/raise rows, and about 50 empty-on-failure collectors.

Tail ledger: `.omx/research/ddm_ce1_20260805/ce1_tail_ledger_20260805.json`.

## Regression evidence

- `.venv/bin/python tools/check_subset_default_scope_fields.py experiments/ddm_p3v2_optimal_form_pose_resolve.py experiments/ddm_pz1_dpose_window_solve_paired.py experiments/ddm_pz1_partial_refit_bound.py experiments/ddm_pz1_scorer_plane_pose_delta.py experiments/diag_custom_vs_reference_trajectory_n8.py --json`
  - Result: 5 files scanned, 9 subset-default sites, 9 `scope_reported`, 0 silent, 0 dormant.
- `.venv/bin/python tools/check_no_silent_cap_defaults.py experiments/multi_pass_inflate_optimizer.py --json`
  - Result: 1 file scanned, 1 cap-default site, 0 silent.
- `.venv/bin/python -m pytest tools/tests/test_check_subset_default_scope_fields.py tools/tests/test_check_no_silent_cap_defaults.py`
  - Result: 14 passed.
- `.venv/bin/python -m py_compile experiments/ddm_p3v2_optimal_form_pose_resolve.py experiments/ddm_pz1_dpose_window_solve_paired.py experiments/ddm_pz1_partial_refit_bound.py experiments/ddm_pz1_scorer_plane_pose_delta.py experiments/diag_custom_vs_reference_trajectory_n8.py experiments/multi_pass_inflate_optimizer.py`
  - Result: passed.
- `.venv/bin/python -m ruff check --isolated --select F821,F841 experiments/ddm_p3v2_optimal_form_pose_resolve.py experiments/ddm_pz1_dpose_window_solve_paired.py experiments/ddm_pz1_partial_refit_bound.py experiments/ddm_pz1_scorer_plane_pose_delta.py experiments/diag_custom_vs_reference_trajectory_n8.py experiments/multi_pass_inflate_optimizer.py`
  - Result: all checks passed.

## Recall evidence

- Governing files read: `AGENTS.md`, `PROGRAM.md`, `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md`; `CLAUDE.md` and `AGENTS.md` are byte-identical at SHA-256 `65da6dd8dcf6b11c0ecdd352938570fd5589c5e5e014d97acd63297f82a8c47c`.
- `6d3cbbb68465dbb64eb08154a10723b42ff8ef8a` recalled as the prior censorship-sources landing: tail-slope adjudicator plus CA1 scanner stop-report marker update.
- CA1 recall: `.omx/research/ddm_ca1_20260805/CA1_RECEIPT.md` and `ca1_classified_inventory_20260805.json` supplied live cap rows and scorer/owner deferrals.
- SS1 recall: `.omx/research/ddm_ss1_20260805/SS1_RECEIPT.md` supplied subset-default denominators and the ranked P3/PZ1/R10/N8/GR1/frontier-adapter rows.
- VC1/SI1 recall: `.omx/research/ddm_vc1_vacuity_denominator_cure_and_census_20260801.md` and `.omx/research/ddm_si1_vacuity_equals_pass_authority_path_20260803.md` supplied remaining vacuity family rows.
- NA4 recall: `.omx/research/ddm_na4_20260805/NA4_RECEIPT.md` supplied the current warning that prefix/subset evidence is selection-mode dependent and must not be read as population evidence.

## Boundaries

- No scorer runs, exact eval, CUDA/CPU contest eval, paid dispatch, launches, banked submissions, or live run dirs were touched.
- Protected paths were not edited: `upstream/`, `src/tac/optimization/direct_description_carrier_compose.py`, `tools/promote_frontier.py`, `.omx/state/frontier/candidate.json`.
- The untracked `tools/run_taskspace_r10_feature_texture_relay.py` was ledgered, not edited or committed.
- Contest pointer and own-vehicle frontier are unchanged: `S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]`; contest pointer borrowed/unmoved.

