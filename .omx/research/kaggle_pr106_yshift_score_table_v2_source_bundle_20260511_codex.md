# Kaggle PR106 y-shift score-table v2 source-bundle fix (2026-05-11)

<!-- generated_at: 2026-05-11T08:37:52Z, agent: codex:gpt-5.5 -->

## Context

Kaggle v1 for `lane_pr106_yshift_score_table` failed before scorer work:

- kernel: `adpena/comma-lab-pr106-yshift-score-table`
- status: `KernelWorkerStatus.ERROR`
- failure class: provider packaging / runtime source unavailable
- log root cause: Kaggle executed only the script kernel's `run_kernel.py`; copied
  repo paths were not visible, and the mounted `comma-lab-private-assets` wheel
  `tac-1.0.5-py3-none-any.whl` lacked `tac.deploy.pr106_yshift`.

This is not a PR106 y-shift method result and not a score claim.

## Fix

The Kaggle launcher is now intentionally small. Required source/runtime/archive
state is transported through a separate deterministic dataset tarball:

- source bundle name: `pact_pr106_yshift_source_bundle.tar.gz`
- default dataset ref: `adpena/comma-lab-pr106-yshift-source`
- kernel ref: `adpena/comma-lab-pr106-yshift-score-table`
- source bundle includes:
  - `src/tac`
  - PR106 y-shift build/eval scripts
  - remote launch helpers
  - PR106 archive at `inputs/pr106_archive.zip`
  - `.omx/state/active_lane_dispatch_claims.md`

The launcher extracts the tarball into
`/kaggle/working/pact_pr106_yshift_workspace`, imports `tac` from that extracted
workspace, installs only non-DALI Python runtime dependencies, clones upstream,
and delegates to the canonical `scripts/remote_lane_pr106_yshift_sidechannel.sh`.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_deploy_claims_active_row.py src/tac/tests/test_kaggle_pr106_yshift_score_table.py src/tac/tests/test_pr106_yshift_deploy_contract.py src/tac/tests/test_pr106_yshift_score_table.py src/tac/tests/test_pr106_latent_score_table.py -q`
  - 36 passed in 3.22s
- `.venv/bin/ruff check --select F821 src/tac/deploy/kaggle/pr106_yshift_score_table.py tools/kaggle_build_pr106_yshift_score_table.py src/tac/tests/test_kaggle_pr106_yshift_score_table.py`
  - passed
- `.venv/bin/python tools/all_lanes_preflight.py`
  - all 29 checks passed

## Next dispatch contract

Before v2 push:

1. create an active dispatch claim for `lane_pr106_yshift_score_table`;
2. run `tools/kaggle_build_pr106_yshift_score_table.py --write-source-bundle`;
3. create/version the private source dataset;
4. push the kernel bundle;
5. harvest logs/output and classify as exact artifact, infrastructure failure,
   or method result only after contest-auth JSON exists.
