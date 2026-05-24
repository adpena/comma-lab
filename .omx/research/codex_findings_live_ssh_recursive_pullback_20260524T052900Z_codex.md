---
schema: codex_findings_v1
created_at_utc: 2026-05-24T05:29:00Z
agent: codex
topic: live_ssh_recursive_pullback
score_authority: false
promotion_eligible: false
rank_or_kill_eligible: false
ready_for_exact_eval_dispatch: false
---

# Live SSH Recursive Pullback Proof

## Finding

The committed no-network SSH custody smoke initially hid a real remote
portability bug: it used `python`, while the clean `tertiary` checkout reliably
exposes `.venv/bin/python`. A live SSH run on the pushed commit failed with
return code `127` after remote git preflight and recursive input push had both
passed.

The smoke now emits `.venv/bin/python` commands. A second live run then proved
the actual recursive output pullback contract on `tertiary`.

## Evidence

Failed live probe before the venv fix:

- Result: `.omx/research/live_ssh_recursive_pullback_smoke_20260524T052200Z/ssh_executor_result.json`
- SHA-256: `ba79d831bc25f9ef37b95cb7bb9741800b168bfd5ed848505b17e3d2d9dc5e1d`
- `remote_preflight.passed=true`
- `input_mobility.succeeded=true`
- `returncode=127`
- failure class: remote command path used `python`.

Successful live recursive output pullback:

- Result: `.omx/research/live_ssh_recursive_output_pullback_smoke_20260524T052700Z/ssh_executor_result.json`
- SHA-256: `f48b126d74ced19ef96fca22ca3af9e09969164b5e3667dd0f004e03d8095d0f`
- Output: `.omx/research/live_ssh_recursive_output_pullback_smoke_20260524T052700Z/outputs/remote_tree/result.json`
- Output SHA-256: `5c744daac88958ac4d1f6e16920dd34959a53edd18f42c7b048b15eb7bc9b695`
- `success_count=1`
- `failure_count=0`
- recursive output pullback used `rsync -a --delete` with trailing slash
  source and destination;
- pre-pull manifest contained stale local-only file
  `stale_local_only.txt`;
- post-pull manifest removed that stale file, recorded
  `recursive_entry_count=3`, `recursive_truncated=false`, and SHA-256
  `84f68f6c43dca2e3114fcb49501529652f1ef937c28967e921276a3ab65eee31`.

This is not score authority. All result artifacts carry
`score_claim=false`, `promotion_eligible=false`,
`rank_or_kill_eligible=false`, and
`ready_for_exact_eval_dispatch=false`.

## Verification

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_ssh_input_custody_smoke.py \
  src/tac/tests/test_ssh_experiment_queue_executor.py -q

.venv/bin/ruff check \
  tools/smoke_staircase_ssh_input_custody.py \
  src/tac/tests/test_ssh_input_custody_smoke.py
```

Both passed after the venv command fix.
