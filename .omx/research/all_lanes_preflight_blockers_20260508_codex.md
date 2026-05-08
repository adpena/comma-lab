# All-Lanes Preflight Blockers - Codex - 2026-05-08

generated_at_utc: 2026-05-08T06:34:00Z
command: `.venv/bin/python tools/all_lanes_preflight.py --jobs 4 --timings`
exit_code: `3`
scope: dispatch-surface audit after half-frame scanner and exact-negative
evidence-row hardening

## Passes

The preflight entrypoint imports and runs. The following dispatch-relevant
gates passed:

- dispatch CLI/shell hazards
- reverse-engineering tree curation
- hidden-gem registry/readiness
- HNeRV frontier scorecard
- HNeRV low-level repack proof
- cross-paradigm frontier inventory
- PR91/HPM1 fail-closed custody
- frontier monolithic archive layout
- Omega-OPT anchor discipline
- apogee intN self-protection
- Omega-W-V3 local-smoke self-protection
- PR106 sidechannel readiness

Focused strict checks also pass on the live tree:

- `check_evidence_row_has_falsification_scope_when_negative(strict=False)`:
  `0` violations across `40` evidence rows
- `check_137531_candidate_decoder_path_wired(strict=False)`: `0` violations
  for the one `137531` cross-paradigm row

## Blockers

### Gate 10 - Untracked Source Inventory

Failure class: source-like untracked files need disposition. This is not a
score-method failure; it is a no-signal-loss/canonicalization blocker.

Current classes observed:

- untracked `.omx/research/*20260508*` ledgers
- untracked `docs/paper/phase4_paper_harness_blueprint_20260508.md`
- untracked reusable `src/tac/*` modules and tests
- untracked `tools/*` implementation scripts
- untracked derived reports

Do not delete these. They must be tracked, dispositioned, moved to an explicit
recovery queue, or ignored only if rebuildable/private.

### Gate 11 - Orphan Recovery Canonicalization

Failure class: many source-like deletions are staged outside
`reverse_engineering/orphan_pyc_recovery_20260505_codex`.

Examples include:

- `tools/pr101_omega_opt_*.py`
- `tools/pr101_tiny_nn_*.py`
- `tools/pr101_lossy_int4_{gptq,awq}.py`
- `tools/build_cross_paradigm_admm_x_op1_finalizer.py`
- `src/tac/tests/test_pr101_*`
- `src/tac/tests/test_preflight_implementation_model_match.py`

Do not accept these deletions until the matching canonical replacements are
explicitly tracked or the operator intentionally discards them.

### Gate 15 - Release Index / Worktree Split

Failure class: staged rollback shadows. These are dangerous because local tests
read the working tree but a commit would publish the staged rollback/deletion.

Observed blocker paths include:

- `.omx/research/findings.md`
- `.ralph/run_log.md`
- `experiments/lossy_coarsening_lightning_cuda_test.py`
- `reports/latest.md`
- `src/tac/experiments/train_renderer.py`
- `src/tac/joint_scorer_aware_training.py`
- `src/tac/learnable_entropy_model.py`
- `src/tac/self_compress_full_renderer.py`
- associated tests for joint scorer-aware training, entropy model, and
  self-compression

This is an index-custody issue. Do not resolve by reverting the working tree.
The next safe action is to review staged-vs-worktree intent and either unstage
stale rollback entries or intentionally stage the desired current content.

## Current Conclusion

The score/dispatch gates that reason about archive compliance, exact evidence,
monolithic layout, and proxy discipline are operational again. The remaining
all-lanes failures are release/canonicalization blockers caused by the dirty
index and untracked source artifacts. They block dispatch/release hygiene, not
the mathematical status of any codec family.

## Resolution Update - 2026-05-08T06:41Z

Follow-up actions:

- Removed stale staged deletion/rollback entries from the index without
  changing working-tree file contents.
- Added explicit `track` dispositions for 73 source-like untracked artifacts in
  `.omx/research/untracked_source_dispositions_20260505_codex.json`.
- Added same-line `[contest-CUDA] [A-negative]` tags to the lossy-coarsening
  score lines in `.ralph/run_log.md` and `.omx/research/findings.md`.

Verification:

- `tools/audit_untracked_source_artifacts.py --disposition-manifest
  .omx/research/untracked_source_dispositions_20260505_codex.json --format
  json`: `ready_for_no_signal_loss_canonicalization=true`, blockers `[]`.
- `tools/audit_orphan_recovery_canonicalization.py --format json`: blockers
  `[]`.
- `tools/audit_release_index_split.py --format json`: blockers `[]`.
- `.venv/bin/python tools/all_lanes_preflight.py --jobs 4 --timings`: all 25
  checks passed.

Remaining notes:

- All-lanes still reports intended non-dispatch warnings: apogee intN is
  forensic-only, and Omega-W-V3 is local-smoke-only until stricter CUDA
  sensitivity readiness exists.
- The optional profile-level command
  `.venv/bin/python -m tac.preflight --profile q_faithful_dilated_88k` was
  stopped after producing no output for over two minutes. The operator-facing
  all-lanes preflight passed, so this is not a dispatch blocker, but the
  profile command's long runtime should be profiled separately.

## Index Custody Follow-Up - 2026-05-08T06:50Z

The release/orphan audits drifted again after further parallel work: two
`admm_x_lossy_coarsening_path_b_step6_no_dead_k` files existed on disk but the
index held staged deletions, and several paths had staged rollback shadows.

Resolution was index-only:

- unstaged the two `no_dead_k` deletions while preserving the working-tree
  files;
- unstaged release-index blocker paths where the preserved working tree and
  staged index disagreed.

Verification:

- `.venv/bin/python tools/audit_orphan_recovery_canonicalization.py`: PASS
- `.venv/bin/python tools/audit_release_index_split.py`: no blockers, custody
  warnings only for raw/generated experiment snapshots.
- `git diff --cached --name-status --diff-filter=D`: empty after cleanup.

## Strict Disposition Follow-Up - 2026-05-08T13:33Z

Further parallel work introduced new source-like untracked artifacts:

- `src/tac/tests/preflight/*`
- `.omx/research/preflight_profile_no_output_diagnosis_20260508_worker_p2.md`
- `src/tac/tests/test_preflight_codebase_drift_scan_scope.py`
- `tools/prove_monolithic_runtime_consumption.py`
- `src/tac/tests/test_prove_monolithic_runtime_consumption.py`

Resolution:

- Added explicit `track` dispositions for each artifact in
  `.omx/research/untracked_source_dispositions_20260505_codex.json`.
- Verified the new preflight/proof tests locally.

Verification:

- `.venv/bin/python -m pytest src/tac/tests/test_prove_monolithic_runtime_consumption.py src/tac/tests/test_preflight_codebase_drift_scan_scope.py src/tac/tests/preflight -q`:
  `47 passed`.
- `.venv/bin/python tools/audit_untracked_source_artifacts.py --disposition-manifest .omx/research/untracked_source_dispositions_20260505_codex.json --format json`:
  blockers `[]`, `ready_for_no_signal_loss_canonicalization=true`.
- `.venv/bin/python tools/all_lanes_preflight.py --jobs 4 --timings`: all 25
  checks passed.

## Swarm Integration Follow-Up - 2026-05-08T06:59Z

Worker R2 and P2 artifacts were integrated:

- Added `track` disposition for
  `.omx/research/monolithic_runtime_consumption_proof_20260508_worker_r2.md`.
- Verified `tools/prove_monolithic_runtime_consumption.py` and
  `src/tac/tests/test_prove_monolithic_runtime_consumption.py` as tracked
  source artifacts.
- Runtime proof is now present for
  `pr106x_lgblock16_monolithic_section_candidate_from_manifest`, but the
  candidate remains blocked by the active PR103-on-PR106 `185578` byte floor
  because it is rate-only at `186079` bytes.
- The profile-preflight zero-output issue is narrowed: the command now emits
  immediate progress and the fast `--no-codebase` path passes; the remaining
  full-command runtime is dirty-tree scan cost, not a silent hang.

Verification:

- `.venv/bin/python -m pytest -q src/tac/tests/test_prove_monolithic_runtime_consumption.py src/tac/tests/test_build_monolithic_runtime_consumption_proof.py src/tac/tests/test_monolithic_packet_closure_gate.py src/tac/tests/test_run_monolithic_candidate_preflight.py src/tac/tests/test_preflight_codebase_drift_scan_scope.py src/tac/tests/preflight`:
  `62 passed`.
- `.venv/bin/python -m py_compile src/tac/preflight.py src/tac/tests/test_preflight_codebase_drift_scan_scope.py tools/prove_monolithic_runtime_consumption.py src/tac/tests/test_prove_monolithic_runtime_consumption.py`:
  passed.
- `.venv/bin/python -m ruff check tools/prove_monolithic_runtime_consumption.py src/tac/tests/test_prove_monolithic_runtime_consumption.py src/tac/tests/test_preflight_codebase_drift_scan_scope.py`:
  passed.
- `.venv/bin/python tools/all_lanes_preflight.py --jobs 4 --timings`: all 25
  checks passed.

Note: whole-file Ruff on `src/tac/preflight.py` is not a clean gate yet because
that legacy file currently has broad unrelated lint debt. Use focused tests,
`py_compile`, and all-lanes preflight as the current verification surface for
the scan-scope patch.

## Planner Validation Queue And Swarm Integration - 2026-05-08T07:16Z

This tranche integrated the F3/B3/L3 swarm outputs and hardened the frontier
planner without creating a score claim:

- F3 ranked the next exact-evaluable work: harvest/adjudicate the active
  `arch_shrink_x0.4_lightning` job first, then build a real runtime packet for
  corrected ADMM-K + Op1, then requalify the no-dead-K ADMM/coarsening packet,
  then close categorical/HPM1 parity, then WR01 PR106x-half component-response
  gating.
- B3 found no existing materialized PR106/monolithic candidate below the active
  `185578` byte floor. The runtime-proven PR106x monolithic control remains
  `186079` bytes and floor-blocked.
- L3 confirmed the active arch-shrink Lightning job should keep running, but
  the observed epoch speed makes it more likely to be checkpoint/loss-curve
  signal than a terminal score producer under the current 18h cap.

`tools/cathedral_autopilot.py` now emits a `validation_queue` for blocked,
proxy, MPS/CPU, unknown-catalog, and cross-paradigm evidence rows. These rows
preserve signal and potential rate upside for operator review but explicitly
remain `score_claim=false`, `rank_or_kill_eligible=false`, and
`ready_for_exact_eval_dispatch=false` until their blockers close. A zero-byte
or missing-finalizer row is now classified as
`unknown_invalid_or_missing_archive_bytes` and receives zero potential score
delta instead of an inflated rate-win priority.

Generated planning artifacts:

- `reports/dispatch_advice_pr103_pr106_current_20260508.json`
- `reports/cathedral_autopilot_plan_pr103_pr106_to_019_20260508.json`
- `reports/cathedral_autopilot_plan_pr103_pr106_to_0155_20260508.json`
- `reports/cathedral_autopilot_catalog_updated_20260508.json`

Durable operator guidance was also added to `AGENTS.md` and `CLAUDE.md`:
beauty, simplicity, developer experience, composable contracts, typed public
APIs, human-readable/machine-checkable artifacts, and cross-language
conformance are explicit engineering constraints.

Verification:

- `.venv/bin/python -m pytest src/tac/tests/test_cathedral_autopilot.py src/tac/tests/test_cathedral_autopilot_proxy_guards.py -q`:
  `26 passed`.
- `.venv/bin/python -m py_compile tools/cathedral_autopilot.py src/tac/tests/test_cathedral_autopilot.py`:
  passed.
- `.venv/bin/python -m ruff check tools/cathedral_autopilot.py src/tac/tests/test_cathedral_autopilot.py`:
  passed.
- `.venv/bin/python tools/audit_untracked_source_artifacts.py --disposition-manifest .omx/research/untracked_source_dispositions_20260505_codex.json --format json`:
  blockers `[]`, `ready_for_no_signal_loss_canonicalization=true`.
- First all-lanes preflight after the patch failed only on staged rollback
  shadows in the git index. Those were fixed with `git restore --staged -- ...`
  on the affected paths, preserving working-tree content.
- Final `.venv/bin/python tools/all_lanes_preflight.py --jobs 4 --timings`:
  `ALL 25 PREFLIGHT CHECKS PASSED`.
