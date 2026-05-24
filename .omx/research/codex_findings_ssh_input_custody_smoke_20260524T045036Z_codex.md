# Codex Findings: SSH Input Custody Smoke

- UTC: 2026-05-24T04:50:36Z
- Lane: `codex_ssh_input_custody_smoke_20260524`
- Scope: non-authoritative staircase SSH executor custody proof for directory input artifacts.

## Landing

Added `tools/smoke_staircase_ssh_input_custody.py` as the operator-facing
no-network smoke for the SSH executor artifact-mobility contract. The smoke
builds a tiny `experiment_queue.v1`, emits a staircase dispatch plan, runs the
production `run_staircase_ssh_executor(...)` path with a fake transport, records
recursive directory input manifests, verifies the directory push uses
`rsync --delete`, pulls back a local output artifact, and forces the report
through the proxy false-authority boundary.

Added regression coverage in `src/tac/tests/test_ssh_input_custody_smoke.py`
and documented the command in `docs/experiment_scheduler_design.md`.

## Smoke Artifact

- Report: `.omx/research/staircase_ssh_input_custody_smoke_20260524T000000Z/smoke_report.json`
- Report SHA-256: `913025f16a8294ff7ebaebf91a3ecb56541870f1e259263e91fc248c0bf969c5`
- Queue SHA-256: `2e42816621d7fbc9d5ab7cf50e24e534f91ed9feef9bdec0ed752328144f9388`
- Dispatch plan SHA-256: `0dca53afab0062799be0e41bd1d101502491d1b014b3d2057ac94cb59fd329e5`
- Output artifact SHA-256: `23e8be33e7d1f6c437df0097c33c93d26facc9000ef8ac96270e88b1c4ec4840`

Observed:

- `success_count=1`
- `failure_count=0`
- `directory_push_used_delete=true`
- `output_false_authority=true`
- `recursive_entry_count=3`
- input manifest SHA-256: `0ff0b3f54fbe6d1f9f7475000aed80179559eea8f00b54baf1c883f142c888ae`
- recursive input SHA-256: `f517fa0540e9339d84dba1cc9e409438dab85ee2c747d1baf349d51ece0d180e`

This is not SSH reachability evidence, paid dispatch evidence, or score
authority. The report carries `score_claim=false`, `promotion_eligible=false`,
`rank_or_kill_eligible=false`, `ready_for_exact_eval_dispatch=false`, and
dispatch blockers including `local_fake_transport_no_network`,
`non_authoritative_custody_smoke_only`, and `exact_eval_not_attempted`.

## Verification

- `.venv/bin/python tools/smoke_staircase_ssh_input_custody.py --run-dir .omx/research/staircase_ssh_input_custody_smoke_20260524T000000Z --queue-id staircase_ssh_input_custody_smoke_codex_20260524`
- `.venv/bin/python -m pytest src/tac/tests/test_ssh_input_custody_smoke.py src/tac/tests/test_ssh_experiment_queue_executor.py -q`
- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_materializer_exact_eval_consumer.py src/tac/tests/test_ssh_input_custody_smoke.py -q`
- `.venv/bin/python -m py_compile tools/smoke_staircase_ssh_input_custody.py src/tac/tests/test_ssh_input_custody_smoke.py`
- `.venv/bin/python -m ruff check tools/smoke_staircase_ssh_input_custody.py src/tac/tests/test_ssh_input_custody_smoke.py`

## Remaining Gaps

The next tranche step is a real bounded SSH smoke on a reachable lab host with
the same queue and artifact-map contract. That should stay non-authoritative
until the remote git preflight, input push, remote command, pullback, local
postconditions, and terminal queue-state writeback all pass on the actual
transport. After that, the six first-per-lane exact-eval dry-run dispatch rows
can be reconsidered only after a fresh frontier scan and explicit lane-claim
lifecycle.
