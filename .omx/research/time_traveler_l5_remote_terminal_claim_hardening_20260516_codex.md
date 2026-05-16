# Time-Traveler L5 Remote Terminal-Claim Hardening

- date: `2026-05-16`
- agent: `codex`
- lane_id: `lane_time_traveler_l5_autonomy_substrate_20260513`
- trigger: remote substrate driver custody-pattern scan after ATW v2 hardening
- score_claim: `false`
- promotion_eligible: `false`

## Finding

`scripts/remote_lane_substrate_time_traveler_l5_autonomy.sh` already verified an
active lane/job dispatch claim before bootstrap and did not print a direct
`[contest-CUDA]` marker. It still used an EXIT trap only to kill the heartbeat,
so successful, failed, and claim-verification-refused runs could leave a live
dispatch row without a terminal status.

Classification: L5 staircase dispatch-custody gap / no-score-change hardening.

## Fix

1. Added `append_terminal_claim()` with status split:
   `completed_tt5l_remote_driver`,
   `failed_tt5l_claim_verification_rc_<rc>`, and
   `failed_tt5l_remote_driver_rc_<rc>`.
2. Replaced the heartbeat-only EXIT trap with a `cleanup` trap that kills the
   heartbeat and appends the terminal claim row.
3. Added focused regression tests for shell syntax, active-claim verification,
   terminal claim closure, and trap replacement.

## Verification

- `bash -n scripts/remote_lane_substrate_time_traveler_l5_autonomy.sh`
- `.venv/bin/python -m ruff check src/tac/tests/test_remote_lane_time_traveler_l5_script.py`
- `.venv/bin/python -m pytest src/tac/tests/test_remote_lane_time_traveler_l5_script.py -q` -> `2 passed`
- `.venv/bin/python -m pytest src/tac/tests/test_remote_lane_time_traveler_l5_script.py src/tac/substrates/atw_codec_v2/tests/test_atw_codec_v2.py -q` -> `28 passed`
- `.venv/bin/python tools/lane_maturity.py validate` -> `767 lane(s) validated cleanly`
- `git diff --check`
