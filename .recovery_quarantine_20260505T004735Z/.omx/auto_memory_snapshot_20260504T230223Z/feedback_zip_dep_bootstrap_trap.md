---
name: zip dep + set -e cascade trap (LANE-B post-mortem)
description: PyTorch container has no `zip` binary; without `set -e` the failed zip silently leaves ARCHIVE_BYTES empty and crashes auth_eval at the very end of a 6.5h job
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
LANE-B (2026-04-26) burned 6.5h + ~$2 because three silent-failure cascades stacked:

1. The standard PyTorch CUDA container (`pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel`) does NOT ship `zip`. `apt list --installed` confirms.
2. `scripts/remote_pose_tto_only_bootstrap.sh` had `set -uo pipefail` (no `-e`), so the failed `zip` command did NOT abort the script — every other bootstrap script uses `set -euo pipefail`.
3. `ARCHIVE_BYTES=$(stat -c '%s' archive.zip)` returned an empty string (no file). The empty value flowed into `--archive-size-bytes ""` which crashed argparse — but only AFTER the heartbeat said "STAGE 4 done".

**Why:** The `/tmp/lane_b_launcher.sh` ad-hoc script that this bootstrap was lifted from didn't have `-e` or zip-presence guards, and that sloppiness rode along through the canonicalization.

**How to apply:**
- Every bootstrap MUST start with `set -euo pipefail` — `-e` is non-negotiable. Test enforces this (`test_set_e_present`).
- Build archives via Python `zipfile` not the shell `zip` binary — Python is guaranteed available, `zip` is not. Test enforces this (`test_no_shell_zip_binary` + `test_uses_python_zipfile`).
- Validate any captured size BEFORE passing to a downstream argparse — empty string crashes loudly but only after the time-expensive stage. Hard-fail before that. Test enforces this (`test_archive_size_validated_before_auth_eval`).
- Validate the final auth eval log contains `RESULT_JSON:` before exiting 0 — "ran to completion" ≠ "produced a score". Test enforces this (`test_auth_eval_log_validated`).

Fix shipped: bash bootstrap commit + `src/tac/tests/test_bootstrap_pose_tto_only.py` (8 tests, all 4 fixes asserted).
