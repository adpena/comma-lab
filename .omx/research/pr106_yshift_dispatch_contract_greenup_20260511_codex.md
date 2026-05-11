# PR106 y-shift dispatch contract greenup — codex

Date: 2026-05-11

Scope:
- `src/tac/deploy/pr106_yshift.py`
- `tools/lightning_dispatch_pr106_yshift_score_table.py`
- `src/tac/tests/test_pr106_yshift_deploy_contract.py`

Objective: keep the PR106 y-shift score-table lane score-lowering path
provider-portable while preserving dispatch claims, scorer-free inflate,
CUDA/CPU/MPS axis separation, and exact-eval custody gates.

## Pass 1 — interface and custody review

Verdict: CLEAN.

Checks:
- The shared module contains only the provider-neutral lane contract:
  constants, dispatch-shaped dataclass, environment builder, and claim metadata.
- Provider concerns remain outside `tac.deploy.pr106_yshift`; Lightning staging,
  SSH checks, Batch submission, and log harvest remain in the Lightning adapter.
- `pr106_archive` is required to be repo-relative, preventing local absolute
  path leakage into Kaggle/Modal/Lightning commands.
- Dispatch-shaping numeric fields fail closed for nonpositive values.
- The canonical environment preserves the exact remote-script contract:
  `PR106_YSHIFT_MODE=score_table`, score-table lane ID, instance/job ID,
  candidate radius, score step, pair count, batch pairs, and candidate batch
  size.
- The claim helper delegates to `tac.deploy.claims.DispatchClaimSpec`, avoiding
  a second hand-rolled dispatch-claim command shape.

Risks considered:
- Accidental scorer load at inflate time: not introduced; this patch only
  changes dispatch metadata and env construction.
- MPS promotion leakage: not introduced; remote score-table producer and exact
  eval path remain CUDA-gated by the existing remote script.
- Provider drift: reduced; Kaggle and Modal can now import the same contract
  instead of copying Lightning-specific constants.

## Pass 2 — implementation and test review

Verdict: CLEAN.

Checks:
- Lightning adapter preserves prior public functions used by tests:
  `build_claim_command`, `score_table_env`, `build_dispatch_command`, and
  `build_batch_spec`.
- The adapter still stages `.omx/state/active_lane_dispatch_claims.md` and the
  PR106 archive before dispatch.
- Batch metadata continues to declare `score_claim=false` and promotion through
  adjudicated `contest_auth_eval.json`, not through launcher output alone.
- New tests cover non-Lightning provider claims, environment parity,
  fail-closed invalid parameter handling, and log-dir behavior.
- Existing Lightning dispatch tests still pass without changing their expected
  lane ID, remote script, environment keys, or role.

Verification:
- `.venv/bin/python -m pytest -q src/tac/tests/test_pr106_yshift_deploy_contract.py src/tac/tests/test_lightning_dispatch_pr106_yshift_score_table.py`
  - Result: 22 passed in 0.19s.
- `.venv/bin/python tools/all_lanes_preflight.py`
  - Result: all 29 checks passed in 2.27s.

Conclusion: provider-neutral contract extraction is greenup-cleared. This does
not by itself create a score claim or dispatch; it removes a provider lock-in
blocker for the PR106 y-shift score-table candidate generation lane.
