# Kaggle PR106 latent score-table bundle

Generated: 2026-05-11T10:02:00Z
Owner: codex

## Summary

Added a provider-neutral and Kaggle-specific launch surface for the PR106
latent-sidecar score-table lane. This prepares the next free-GPU score-lowering
branch without dispatching over the active y-shift Kaggle claim or the active
T1 Modal claim.

## Implementation

- `src/tac/deploy/pr106_latent.py`
  - canonical environment and lane-claim contract for
    `lane_pr106_latent_sidecar`.
- `src/tac/deploy/kaggle/source_bundle.py`
  - shared deterministic source-bundle helpers for Kaggle score-table kernels.
- `src/tac/deploy/kaggle/pr106_latent_score_table.py`
  - private Kaggle script-kernel bundle writer;
  - deterministic source dataset tarball writer;
  - P100 PyTorch fallback and `PYTORCH_CUDA_ALLOC_CONF` launcher policy;
  - source-tree and expanded-archive reconstruction matching the y-shift lane.
- `tools/kaggle_build_pr106_latent_score_table.py`
  - operator CLI for claim command, source bundle, and kernel bundle creation.
- `scripts/kaggle_check.py`
  - adds the latent kernel to the default status watchlist.
- `tools/operator_briefing.py`
  - surfaces the Kaggle bundle tool and kernel slug alongside the existing
    PR106 latent sidecar lane.

## Local materialization proof

The bundle was materialized against a temporary local claim ledger under:

`/tmp/pact_latent_kaggle_contract_20260511`

The output contained:

- `kaggle_pr106_latent_score_table_bundle_v1`
- `kaggle_pr106_latent_source_bundle_v1`
- required source paths:
  - `src/tac`
  - `experiments/build_pr106_latent_score_table.py`
  - `experiments/build_pr106_latent_sidecar.py`
  - `scripts/remote_lane_pr106_latent_sidecar.sh`
  - `submissions/pr106_latent_sidecar`
- `score_claim=false`

No live provider dispatch was started by this proof.

## Evidence

- `.venv/bin/python -m pytest src/tac/tests/test_pr106_latent_deploy_contract.py src/tac/tests/test_kaggle_pr106_latent_score_table.py src/tac/tests/test_pr106_latent_score_table.py tests/test_kaggle_check.py src/tac/tests/test_operator_briefing.py -q`
  - 34 passed
- `.venv/bin/python -m pytest src/tac/tests/test_kaggle_pr106_yshift_score_table.py src/tac/tests/test_kaggle_pr106_latent_score_table.py src/tac/tests/test_pr106_latent_deploy_contract.py src/tac/tests/test_pr106_latent_score_table.py tests/test_kaggle_check.py src/tac/tests/test_operator_briefing.py -q`
  - 40 passed
- `.venv/bin/ruff check --select F821 src/tac/deploy/kaggle/source_bundle.py src/tac/deploy/kaggle/pr106_yshift_score_table.py src/tac/deploy/kaggle/pr106_latent_score_table.py src/tac/deploy/pr106_latent.py tools/kaggle_build_pr106_latent_score_table.py scripts/kaggle_check.py tools/operator_briefing.py`
  - passed

## Claim discipline

This is not a score claim and not a dispatch. Before real Kaggle launch:

1. wait for or explicitly classify the active y-shift Kaggle claim;
2. insert a fresh `lane_pr106_latent_sidecar` claim with the final Kaggle job id;
3. build/version the source dataset;
4. push the private Kaggle kernel;
5. harvest output and adjudicate the charged sidecar archive through exact CUDA
   auth eval before any score promotion.

## Current live state

At implementation time:

- `lane_pr106_yshift_score_table` / `kaggle_pr106_yshift_score_table_v6`:
  active, Kaggle status `KernelWorkerStatus.RUNNING`.
- `t1_balle_128k_endtoend` / `t1_balle_modal_phase1_ab2d0f6_20260510T1437Z`:
  active, Modal recover still reports queued or running.
