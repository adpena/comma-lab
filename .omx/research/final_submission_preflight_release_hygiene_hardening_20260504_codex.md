# Final Submission Preflight And Release Hygiene Hardening - 2026-05-04

Owner: Codex worker, final-submission/preflight/report helper slice.

Scope:

- `scripts/pre_submission_compliance_check.py`
- `scripts/build_contest_submission_packet.py`
- `src/tac/preflight.py`
- focused tests under `src/tac/tests/`

No score-critical runtime, archive builder, inflate path, scorer, or remote GPU
dispatch was changed.

## Bug Classes Hardened

1. Stale or missing exact archive manifest in final packets.
   - `--contest-final` now requires an archive manifest, defaulting to
     `submission_dir/archive_manifest.json`.
   - The manifest must agree with the actual archive SHA-256 and byte size.
   - If member rows are present, they must be unique and match the exact ZIP
     members by name, uncompressed bytes, and SHA-256 when provided.

2. Multiple packed payload containers.
   - Final pre-submission checks and packet building now fail closed when an
     archive contains more than one of `p`, `renderer_payload.bin`, or
     `renderer_payload.bin.br`.
   - This protects the packed-payload singleton rule before any report or
     packet is promoted.

3. ZIP custody edge cases in packet automation.
   - `build_contest_submission_packet.py` now inspects archive CRCs, duplicate
     member names, unsafe/hidden/zip-slip member names, local-header versus
     central-directory name mismatches, and local-header flag mismatches.
   - These checks run before packet manifests/checklists are emitted.

4. Public report not tied to exact archive bytes.
   - `--contest-final` now requires `report.txt` to include the exact archive
     SHA-256 and byte size.
   - Optional auth-score, lane-id, and job-id linkage checks are available for
     private custody packets when those identifiers are intentionally supplied.

5. Dispatch-claim terminal linkage.
   - `scripts/pre_submission_compliance_check.py` can now verify that a given
     lane/job has a matching terminal row in `active_lane_dispatch_claims.md`.
   - This is explicit/opt-in because raw provider and claim state remains
     private custody rather than public report material.

6. Modal provider leakage in public surfaces.
   - `check_public_release_hygiene()` now flags raw Modal call IDs (`fc-...`)
     and app IDs (`ap-...`) alongside existing local-path, token, Vast,
     Lightning, and Cloudflare leak checks.

## Verification

```bash
.venv/bin/python -m py_compile \
  src/tac/preflight.py \
  scripts/pre_submission_compliance_check.py \
  scripts/build_contest_submission_packet.py \
  src/tac/tests/test_pre_submission_compliance_check.py \
  src/tac/tests/test_build_contest_submission_packet.py

.venv/bin/python -m pytest \
  src/tac/tests/test_pre_submission_compliance_check.py \
  src/tac/tests/test_build_contest_submission_packet.py -q
```

Result: `38 passed, 1 warning` (the warning is the intentional duplicate-ZIP
test fixture from Python `zipfile`).

Evidence grade: engineering guardrail; no score claim.
