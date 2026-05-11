# Kaggle PR106 y-shift score-table bundle greenup — codex

Date: 2026-05-11

Scope:
- `src/tac/deploy/claims.py`
- `src/tac/deploy/kaggle/pr106_yshift_score_table.py`
- `src/tac/sidechannel_score_table.py`
- `tools/kaggle_build_pr106_yshift_score_table.py`
- `scripts/kaggle_check.py`
- `src/tac/tests/test_deploy_claims_active_row.py`
- `src/tac/tests/test_kaggle_pr106_yshift_score_table.py`

Objective: make the PR106 y-shift score-table lane usable on Kaggle without
duplicating Lightning dispatch logic, weakening lane claims, or promoting any
proxy/MPS/CPU artifact as exact CUDA evidence.

## Pass 1 — custody and provider-boundary review

Verdict: CLEAN.

Checks:
- The score-lowering contract remains in `tac.deploy.pr106_yshift`; the Kaggle
  module only writes a private script-kernel bundle.
- `tac.deploy.claims.active_claim_row()` centralizes active-claim parsing so
  Kaggle, Lightning, Modal, and score-table producers share terminal-status
  semantics.
- `tac.sidechannel_score_table.verify_active_lane_claim()` now delegates to the
  deploy claim parser rather than carrying a second parser.
- The Kaggle bundle writer refuses to write a CUDA-capable score-table bundle
  unless `.omx/state/active_lane_dispatch_claims.md` contains the matching
  active lane/job row.
- The generated Kaggle launcher sets `PR106_YSHIFT_MODE=score_table` and the
  same candidate/job/env fields used by Lightning.
- The generated launcher sets `WORKSPACE`, `PYBIN`, `PYTHONPATH`, and
  `TAC_UPSTREAM_DIR` explicitly; it does not load scorers at inflate time.

Risks considered:
- Phantom dispatch claim: blocked by active-claim verification before bundle
  emission.
- Provider drift: reduced by sharing `DispatchClaimSpec`, `active_claim_row`,
  and `Pr106YshiftScoreTableSpec`.
- Stale wheel on Kaggle: mitigated by bundling `src/tac` and prepending
  `workspace/src` before bootstrap verification.
- DALI/NVDEC mismatch: left in the canonical remote script via
  `scripts/probe_nvdec.sh --ensure-dali`; the launcher does not install an
  unpinned DALI wheel first.

## Pass 2 — implementation and test review

Verdict: CLEAN.

Checks:
- Bundle metadata stays private and GPU-enabled, with `score_claim=false` and
  `promotion_requires=contest_auth_eval_json_adjudication`.
- Bundle copies only required source/runtime surfaces plus the PR106 archive
  and active claim ledger.
- Bundle output is generated under `experiments/kaggle_kernels/` by the CLI;
  generated bundles are not committed as source of truth.
- `scripts/kaggle_check.py` now watches the PR106 y-shift kernel slug.
- The CLI can print the exact Kaggle claim command without mutating state.
- Attempting to build without a matching active claim fails closed.

Verification:
- `.venv/bin/python -m pytest -q src/tac/tests/test_deploy_claims_active_row.py src/tac/tests/test_kaggle_pr106_yshift_score_table.py src/tac/tests/test_pr106_yshift_deploy_contract.py src/tac/tests/test_pr106_yshift_score_table.py src/tac/tests/test_pr106_latent_score_table.py`
  - Result: 35 passed in 1.76s.
- `.venv/bin/ruff check --select F821 --no-cache src/tac/deploy/claims.py src/tac/sidechannel_score_table.py src/tac/deploy/kaggle/pr106_yshift_score_table.py tools/kaggle_build_pr106_yshift_score_table.py scripts/kaggle_check.py src/tac/tests/test_deploy_claims_active_row.py src/tac/tests/test_kaggle_pr106_yshift_score_table.py`
  - Result: all checks passed.
- `.venv/bin/python tools/all_lanes_preflight.py`
  - Result: all 29 checks passed in 2.33s.

Conclusion: Kaggle PR106 y-shift score-table bundle support is greenup-cleared.
It does not dispatch or claim a score by itself. The next score-lowering action
is to create a Kaggle lane claim, build the bundle from that claimed ledger,
push the private kernel, then harvest/adjudicate `contest_auth_eval.json`.
