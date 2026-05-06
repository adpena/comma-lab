# Recovered Remote Lanes

These scripts were recovered during the 2026-05-05 no-signal-loss sweep. They
are intentionally tracked as source candidates rather than hidden quarantine
fragments.

## Current Classification

| Script | Status | Dispatch Rule |
| --- | --- | --- |
| `scripts/remote_lane_sjkl_c067.sh` | canonical recovered executable | Claim `lane_sjkl_c067` first; exact score evidence comes only from delegated `remote_archive_only_eval.sh` custody JSON. |
| `scripts/remote_lane_pr79_segaction_search.sh` | proxy-search recovered executable | Do not rank or promote from its output. It writes `score_claim=false`; run exact CUDA eval on chosen archive bytes before any score claim. |
| `scripts/remote_lane_q_faithful_jointgen.sh` | legacy recovered executable | Not the default next-run path. If reused, preserve `--keep-work-dir --work-dir "$LOG_DIR/eval_work"` and adjudicate exact archive bytes. |

## Guardrails

- `tools/audit_recovered_remote_lanes.py` is the canonical machine-readable
  audit for this surface. It checks script presence, `bash -n`, custody/proxy
  markers, and emits `score_claim=false` / `dispatch_attempted=false`.
- `tools/all_lanes_preflight.py` runs that audit as Gate #7, so recovered-lane
  drift is visible in the normal operator preflight.
- `src/tac/tests/test_recovered_remote_lane_scripts.py` protects the above
  classifications and exact-eval custody markers.
- `src/tac/tests/test_remote_auth_eval_hardening.py` scans live remote lane
  scripts for fragile `contest_auth_eval` handling.
- `bash -n scripts/remote_lane_pr79_segaction_search.sh
  scripts/remote_lane_q_faithful_jointgen.sh
  scripts/remote_lane_sjkl_c067.sh` must pass after edits.

Do not delete these scripts as quarantine cleanup unless a tracked replacement
exists or a dated `.omx/research/` ledger retires the lane with evidence.
