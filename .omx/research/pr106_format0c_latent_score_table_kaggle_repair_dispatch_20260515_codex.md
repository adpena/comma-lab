# PR106 format0C latent score-table Kaggle repair dispatch - 2026-05-15

## Classification

- lane_id: `lane_pr106_latent_sidecar`
- provider: `kaggle`
- instance_job_id: `kaggle_pr106_format0c_latent_score_table_repair_20260515T203529Z`
- kernel: `https://www.kaggle.com/code/adpena/comma-lab-pr106-latent-score-table`
- kernel_version: `5`
- source_dataset: `https://www.kaggle.com/datasets/adpena/comma-lab-pr106-latent-source`
- score_claim: `false`
- promotion_eligible: `false`
- classification: repaired provider producer, not a score result

## Root Cause Closed

The previous retry
`kaggle_pr106_format0c_latent_score_table_retry_20260515T135531Z` failed with:

`ValueError: no active lane claim found for lane_id=lane_pr106_latent_sidecar instance_job_id=kaggle_pr106_format0c_latent_score_table_retry_20260515T135531Z`

Local audit showed the checked-in reusable launcher had already been hardened,
but the ignored generated Kaggle directory on disk was stale. It still allowed
expanded-source fallback and did not set
`PR106_LATENT_ALLOW_PROVIDER_CLAIM_MIRROR=1`. Kernel version 5 was rebuilt from
`src/tac/deploy/kaggle/pr106_latent_score_table.py` after the source-bundle and
claim-mirror fixes, then pushed from the fresh generated directory.

## Source Contract

- source_archive: `experiments/results/pr106_format0c_exact_radix_candidate_20260515_codex/candidates/pr101_hdm9_hlm3_magicless_exact_radix_dim_fixed_meta_noop_rank_elided_sidecar_format_0x0c.archive.zip`
- source_archive_sha256: `56cdd10bdc43708f2021458d0877b6c5e5a065a482a61280e727078462aed8e7`
- source_archive_member: `x`
- source_archive_member_sha256: `852a4cb1231413cf1a8fc867e2a808de9ec78511d2ebf283df2c5b608cb4a749`
- runtime_dir: `submissions/pr106_latent_sidecar_r2_pr101_grammar`
- upstream_commit: `11ad728f563d8970929e8947a1cf6124ee6303e4`
- delta_radius: `2`
- latent_dim: `28`
- n_pairs: `600`

## Commands

```bash
.venv/bin/python tools/claim_lane_dispatch.py claim \
  --lane-id lane_pr106_latent_sidecar \
  --platform kaggle \
  --instance-job-id kaggle_pr106_format0c_latent_score_table_repair_20260515T203529Z \
  --agent codex:gpt-5.5 \
  --predicted-eta-utc 2026-05-15T23:35:29Z \
  --status active_dispatching \
  --notes 'PR106 format0C latent score-table repaired source-bundle/claim-mirror Kaggle CUDA producer; score_claim=false; promotion requires byte-closed paired contest-axis adjudication'

.venv/bin/python tools/kaggle_build_pr106_latent_score_table.py \
  --username adpena \
  --job-name kaggle_pr106_format0c_latent_score_table_repair_20260515T203529Z \
  --write-source-bundle

uv run --with kaggle kaggle datasets version \
  -p experiments/kaggle_datasets/comma-lab-pr106-latent-source \
  -m 'PR106 format0C latent source bundle repair 20260515T203529Z'

uv run --with kaggle kaggle kernels push \
  -p experiments/kaggle_kernels/comma-lab-pr106-latent-score-table
```

## Verification

- Generated `run_kernel.py` contains:
  - `PR106_LATENT_ALLOW_PROVIDER_CLAIM_MIRROR=1`
  - `PR106_LATENT_ALLOW_EXPANDED_SOURCE_TREE` fail-closed fallback gate
  - `refusing expanded-source fallback`
- Generated source bundle contains:
  - `src/tac/sidechannel_score_table.py`
  - `scripts/remote_lane_pr106_latent_sidecar.sh`
  - `.omx/state/active_lane_dispatch_claims.md`
  - `inputs/pr106_archive.zip`
- `.venv/bin/python -m pytest -q src/tac/tests/test_kaggle_pr106_latent_score_table.py src/tac/tests/test_pr106_latent_deploy_contract.py src/tac/tests/test_harvest_kaggle_pr106_latent_score_table.py`
  - result: `21 passed in 7.50s`
- `uv run --with kaggle kaggle kernels status adpena/comma-lab-pr106-latent-score-table`
  - result after push: `KernelWorkerStatus.RUNNING`

## Harvest Result - kernel v5

- instance_job_id: `kaggle_pr106_format0c_latent_score_table_repair_20260515T203529Z`
- evidence_dir: `reports/raw/kaggle_ingested/kaggle_pr106_format0c_latent_score_table_repair_20260515T203529Z`
- status: `failed_kaggle_kernel_error`
- failure: `FileNotFoundError: required source bundle 'pact_pr106_latent_source_bundle.tar.gz' not found under ['/kaggle/src', '/kaggle/input']; refusing expanded-source fallback for Kaggle custody.`
- score_table: none
- score_claim: false
- terminal_claim: recorded as `failed_kaggle_kernel_error`

Classification: this was a custody-shape failure, not a method negative. Kaggle
expanded the uploaded `pact_pr106_latent_source_bundle.tar.gz` dataset into a
directory named `pact_pr106_latent_source_bundle/...` instead of mounting the
tarball itself. The broad expanded-source fallback correctly stayed disabled,
but the verified expanded source-bundle directory was not yet accepted.

## Hardening Delta - verified expanded source-bundle tree

- Added `source_bundle_manifest.json` inside deterministic Kaggle source
  tarballs for latent and y-shift score-table producers.
- The generated launchers now accept Kaggle-expanded source-bundle directories
  only when the tree name, manifest schema, job id, lane id, marker file types,
  runtime directory, remote script, builder script, archive member, and expected
  SHA custody match.
- Arbitrary expanded `src/tac` trees still require explicit local-dev override:
  `PR106_LATENT_ALLOW_EXPANDED_SOURCE_TREE=1` or
  `PR106_YSHIFT_ALLOW_EXPANDED_SOURCE_TREE=1`.
- Added `tools/materialize_pr106_latent_score_table_candidate.py` to the latent
  source bundle closure because the remote script invokes it after score-table
  generation.
- Fixed the direct latent score-table CLI default lane id to
  `lane_pr106_latent_sidecar` so direct CUDA runs use the same claim contract as
  provider adapters.

## Verification - verified expanded source-bundle tree

- `.venv/bin/python -m pytest -q src/tac/tests/test_kaggle_pr106_latent_score_table.py src/tac/tests/test_kaggle_pr106_yshift_score_table.py src/tac/tests/test_pr106_latent_score_table.py src/tac/tests/test_pr106_latent_deploy_contract.py src/tac/tests/test_harvest_kaggle_pr106_latent_score_table.py`
  - result: `46 passed in 19.04s`
- `.venv/bin/ruff check src/tac/deploy/kaggle/source_bundle.py src/tac/deploy/kaggle/pr106_latent_score_table.py src/tac/deploy/kaggle/pr106_yshift_score_table.py experiments/build_pr106_latent_score_table.py src/tac/tests/test_kaggle_pr106_latent_score_table.py src/tac/tests/test_kaggle_pr106_yshift_score_table.py src/tac/tests/test_pr106_latent_score_table.py`
  - result: `All checks passed`
- `git diff --check -- src/tac/deploy/kaggle/source_bundle.py src/tac/deploy/kaggle/pr106_latent_score_table.py src/tac/deploy/kaggle/pr106_yshift_score_table.py experiments/build_pr106_latent_score_table.py src/tac/tests/test_kaggle_pr106_latent_score_table.py src/tac/tests/test_kaggle_pr106_yshift_score_table.py src/tac/tests/test_pr106_latent_score_table.py`
  - result: clean

## Dispatch - kernel v6

- instance_job_id: `kaggle_pr106_format0c_latent_score_table_repair2_20260515T204229Z`
- kernel_version: `6`
- source_dataset: versioned with message `PR106 format0C latent source bundle repair2 20260515T204229Z`
- source bundle contains:
  - `source_bundle_manifest.json`
  - `tools/materialize_pr106_latent_score_table_candidate.py`
  - `.omx/state/active_lane_dispatch_claims.md`
  - `inputs/pr106_archive.zip`
- `uv run --with kaggle kaggle kernels status adpena/comma-lab-pr106-latent-score-table`
  - result after push: `KernelWorkerStatus.RUNNING`

## Harvest Plan

1. Poll `kaggle kernels status adpena/comma-lab-pr106-latent-score-table`.
2. On completion, run `tools/harvest_kaggle_pr106_latent_score_table.py --run-id kaggle_pr106_format0c_latent_score_table_repair2_20260515T204229Z --instance-job-id kaggle_pr106_format0c_latent_score_table_repair2_20260515T204229Z --close-claim`.
3. Treat any Kaggle score as `[provider-CUDA:kaggle advisory]`, not `[contest-CUDA]`.
4. If the score table materializes a candidate archive, dispatch paired exact eval with `tools/dispatch_modal_paired_auth_eval.py` or the canonical contest CPU/CUDA path before any promotion language.
