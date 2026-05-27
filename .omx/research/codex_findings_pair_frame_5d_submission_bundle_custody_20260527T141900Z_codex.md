# Pair-Frame 5D Submission Bundle Custody Hardening

Generated: 2026-05-27T14:19:00Z
Author: Codex

## Finding

The pair-frame 5D exact follow-up surfaces could bind a
`submission_bundle_result.json` contract without proving the referenced
submission bundle was live and byte-closed. A JSON-only result could advance
readiness if the schema and archive SHA matched, even when `submission_dir`,
`archive.zip`, `inflate.sh`, and manifest files were missing.

## Fix

- Refuse symlinked bundle result paths.
- Require a live submission directory.
- Require `archive.zip` to exist and match declared bytes/SHA-256.
- Require canonical `inflate.sh`, `inflate.py`, `README.md`, `report.txt`, and
  `archive_manifest.json` files inside the submission directory.
- Revalidate `archive_manifest.json` against the same archive bytes/SHA.
- Preserve exact follow-up readiness as fail-closed when auto-discovered bundle
  paths are symlinks or contract-only JSON.

## Regression Coverage

- `test_followup_readiness_refuses_contract_only_missing_bundle_files`
- `test_followup_input_binding_refuses_contract_only_missing_bundle_files`
- `test_followup_readiness_refuses_symlinked_submission_bundle_result`
- `test_followup_input_binding_refuses_autodiscovered_symlinked_bundle_result`

## Verification

```
.venv/bin/python -m pytest src/tac/tests/test_pair_frame_5d_coverage_acquisition_queue.py -q
# 28 passed in 17.34s

.venv/bin/ruff check src/comma_lab/scheduler/pair_frame_5d_coverage_acquisition_queue.py src/tac/tests/test_pair_frame_5d_coverage_acquisition_queue.py
# All checks passed
```
