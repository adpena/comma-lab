# PR106 format0C latent score-table Kaggle dispatch - 2026-05-15

## Dispatch

- lane_id: `lane_pr106_latent_sidecar`
- provider: `kaggle`
- instance_job_id: `kaggle_pr106_format0c_latent_score_table_20260515T133639Z`
- repo_commit: `3c2a34c3b` (`deploy: harden PR106 format0C score-table path`)
- kernel: `https://www.kaggle.com/code/adpena/comma-lab-pr106-latent-score-table`
- kernel_version: `3`
- source_dataset: `https://www.kaggle.com/datasets/adpena/comma-lab-pr106-latent-source`
- claim_status: recorded in `.omx/state/active_lane_dispatch_claims.md`

## Source Contract

- source_archive: `experiments/results/pr106_format0c_exact_radix_candidate_20260515_codex/candidates/pr101_hdm9_hlm3_magicless_exact_radix_dim_fixed_meta_noop_rank_elided_sidecar_format_0x0c.archive.zip`
- source_archive_bytes: `186327`
- source_archive_sha256: `56cdd10bdc43708f2021458d0877b6c5e5a065a482a61280e727078462aed8e7`
- source_archive_member: `x`
- source_archive_member_bytes: `186227`
- source_archive_member_sha256: `852a4cb1231413cf1a8fc867e2a808de9ec78511d2ebf283df2c5b608cb4a749`
- runtime_dir: `submissions/pr106_latent_sidecar_r2_pr101_grammar`
- delta_radius: `2`
- latent_dim: `28`
- n_pairs: `600`
- candidate_grid_shape_expected: `[600, 113]`

## Evidence And Guardrails

- `score_claim=false` for this Kaggle producer. Any score emitted by the kernel is provider-CUDA advisory until paired contest-axis adjudication.
- Promotion still requires byte-closed archive/runtime custody and paired `[contest-CUDA]` and `[contest-CPU]` eval artifacts.
- The remote script now fails closed before CUDA if the source archive, explicit member `x`, or runtime files are missing.
- Local preflight smoke wrote `/tmp/pr106_latent_preflight_smoke/source_preflight.json` with `ok=true`, then failed cleanly at the expected no-local-CUDA gate.

## Verification

- `.venv/bin/python -m pytest -q src/tac/tests/test_local_pre_deploy_check.py src/tac/tests/test_stack_of_stacks_dispatch_blocked.py src/tac/composition/tests/test_stack_of_stacks_inflate_runtime.py src/tac/tests/test_sub_0192_viability_guard.py src/tac/tests/test_pr106_latent_sidecar_recode.py src/tac/tests/test_pr106_latent_score_table.py src/tac/tests/test_pr106_latent_deploy_contract.py src/tac/tests/test_kaggle_pr106_latent_score_table.py src/tac/tests/test_harvest_kaggle_pr106_latent_score_table.py src/tac/tests/test_check_244_remote_lane_canonical_nvml_block.py src/tac/tests/test_substrate_contract.py src/tac/tests/test_substrate_registry.py src/tac/tests/test_check_241_242_meta_layer_gates.py src/tac/tests/test_meta_layer_adversarial_review_fixes.py`
  - result: `185 passed in 12.23s`
- `.venv/bin/ruff check src/tac/deploy/pr106_latent.py src/tac/deploy/kaggle/pr106_latent_score_table.py tools/kaggle_build_pr106_latent_score_table.py src/tac/tests/test_pr106_latent_deploy_contract.py src/tac/tests/test_kaggle_pr106_latent_score_table.py src/tac/tests/test_pr106_latent_score_table.py`
  - result: `All checks passed`
- `git diff --cached --check`
  - result: clean before commit
- `bash -n scripts/remote_lane_pr106_latent_sidecar.sh`
  - result: clean

## Harvest Plan

1. Poll `kaggle kernels status adpena/comma-lab-pr106-latent-score-table`.
2. On completion, download kernel output and preserve `kaggle_pr106_latent_score_table_summary.json`, `latent_run/source_preflight.json`, `latent_run/score_table/score_table_manifest.json`, score table, built archive, runtime-consumption proof, and any `contest_auth_eval.json`.
3. Classify output as `advisory_provider_cuda` unless it is rerun through canonical paired contest hardware.
4. If the score-table materializer yields a nonnegative or positive CUDA advisory delta, dispatch paired exact eval through the canonical Modal/Lightning T4 and CPU paths with a fresh lane claim.
5. Close the lane claim with a terminal `completed_...` or `failed_...` row after every harvest.

## Harvest Result - kernel v3

- harvested_at: `2026-05-15T13:42Z`
- evidence_dir: `reports/raw/kaggle_ingested/kaggle_pr106_format0c_latent_score_table_20260515T133639Z`
- status: `failed_kaggle_missing_constriction`
- failure: `ModuleNotFoundError: No module named 'constriction'` before score-table generation.
- score_table: none
- score_claim: false
- terminal_claim: recorded as `failed_kaggle_missing_constriction`

The source preflight did run before the failure and verified the intended source
archive/member hashes. This is an infrastructure dependency closure failure, not
a method negative.

## Hardening Delta - retry-ready

- Added exact source archive and member SHA-256 defaults to the PR106 score-table deploy contract.
- The remote runner now fails closed if the source archive or explicit member `x` hashes differ from the dispatch contract.
- Kaggle provider output is labeled `[provider-CUDA:kaggle advisory]`; `[contest-CUDA]` and public-frontier language are reserved for canonical contest-axis adjudication.
- The Kaggle launcher now installs pinned score-table dependencies, including `constriction==0.4.2`.
- The Kaggle launcher checks out pinned upstream commit `11ad728f563d8970929e8947a1cf6124ee6303e4` before pulling LFS assets and records it in run summary/provenance.

## Verification - hardening delta

- `.venv/bin/python -m pytest -q src/tac/tests/test_pr106_latent_deploy_contract.py src/tac/tests/test_kaggle_pr106_latent_score_table.py src/tac/tests/test_pr106_latent_score_table.py src/tac/tests/test_harvest_kaggle_pr106_latent_score_table.py`
  - result: `31 passed in 8.65s`
- `.venv/bin/python -m pytest -q src/tac/tests/test_local_pre_deploy_check.py src/tac/tests/test_stack_of_stacks_dispatch_blocked.py src/tac/composition/tests/test_stack_of_stacks_inflate_runtime.py src/tac/tests/test_sub_0192_viability_guard.py src/tac/tests/test_pr106_latent_sidecar_recode.py src/tac/tests/test_pr106_latent_score_table.py src/tac/tests/test_pr106_latent_deploy_contract.py src/tac/tests/test_kaggle_pr106_latent_score_table.py src/tac/tests/test_harvest_kaggle_pr106_latent_score_table.py src/tac/tests/test_check_244_remote_lane_canonical_nvml_block.py src/tac/tests/test_substrate_contract.py src/tac/tests/test_substrate_registry.py src/tac/tests/test_check_241_242_meta_layer_gates.py src/tac/tests/test_meta_layer_adversarial_review_fixes.py`
  - result: `188 passed in 12.35s`
- `.venv/bin/ruff check src/tac/deploy/pr106_latent.py src/tac/deploy/kaggle/pr106_latent_score_table.py tools/kaggle_build_pr106_latent_score_table.py src/tac/tests/test_pr106_latent_deploy_contract.py src/tac/tests/test_kaggle_pr106_latent_score_table.py src/tac/tests/test_pr106_latent_score_table.py`
  - result: `All checks passed`
- `bash -n scripts/remote_lane_pr106_latent_sidecar.sh`
  - result: clean
- Bad-SHA local smoke:
  - command sets `PR106_EXPECTED_ARCHIVE_SHA256=000...000`; result `rc=3` before CUDA.
- Good-source local smoke:
  - command uses default expected hashes; `source_preflight.json` records `ok=true`, then local host stops at the expected no-CUDA guard.
