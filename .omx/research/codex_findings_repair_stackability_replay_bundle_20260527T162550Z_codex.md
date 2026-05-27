# Codex Findings: Repair Stackability Replay Bundle

Date: 2026-05-27T16:25:50Z

## Verdict

Repair stackability probes now have a deterministic replay bundle surface that
hash-binds the score report, local MLX custody artifacts, replay argv, safe
environment capture, Python/platform context, and git context. The bundle is a
local MLX research artifact only: it preserves stackability signal for continual
learning and rerun-diff analysis, but it cannot claim score, spend budget, rank,
kill, promote, or dispatch exact eval.

## Landed Integration

- `repair_campaign_stackability_queue` emits a second executable child step for
  ready rows: build the replay bundle after the local stackability probe.
- `tac.optimization.repair_campaign_replay_bundle` validates the score report
  and probe schemas, requires local MLX custody records, records source-file
  SHA-256s, and fails closed on any truthy authority field.
- Replay identity is separated from execution context. `hash_manifest_sha256`
  captures stable probe identity; `execution_context_sha256` captures argv,
  Python, environment, and git drift.
- `tools/build_repair_campaign_stackability_replay_bundle.py` writes bundle
  artifacts from queue-owned probe rows.
- `tools/diff_repair_campaign_stackability_replay_bundle.py` compares two
  bundles without granting authority and reports stable identity versus context
  drift separately.

## Safeguards

- Credential-like environment keys are redacted before hashing or artifact
  persistence.
- Replay bundle and diff artifacts include false authority fields and are
  checked by `require_no_truthy_authority_fields`.
- Queue postconditions require the replay bundle schema, false authority, and
  `ready_for_exact_eval_dispatch=false`.
- A local MLX replay bundle is explicitly not a calibration target for contest
  CPU/CUDA score authority.

## Verification

- `.venv/bin/python -m ruff check src/tac/optimization/repair_campaign_replay_bundle.py tools/build_repair_campaign_stackability_replay_bundle.py tools/diff_repair_campaign_stackability_replay_bundle.py src/comma_lab/scheduler/repair_campaign_stackability_queue.py src/tac/tests/test_repair_campaign_stackability_queue.py`
- `.venv/bin/python -m py_compile src/tac/optimization/repair_campaign_replay_bundle.py tools/build_repair_campaign_stackability_replay_bundle.py tools/diff_repair_campaign_stackability_replay_bundle.py src/comma_lab/scheduler/repair_campaign_stackability_queue.py`
- `.venv/bin/python -m pytest src/tac/tests/test_repair_campaign_stackability_queue.py -q`
  - `5 passed`
- `git diff --check -- src/tac/optimization/repair_campaign_replay_bundle.py tools/build_repair_campaign_stackability_replay_bundle.py tools/diff_repair_campaign_stackability_replay_bundle.py src/comma_lab/scheduler/repair_campaign_stackability_queue.py src/tac/tests/test_repair_campaign_stackability_queue.py`
- `.venv/bin/python tools/lane_maturity.py validate`
  - `OK - 1439 lane(s) validated cleanly.`
- `.venv/bin/python tools/review_tracker.py policy-check src/tac/optimization/repair_campaign_replay_bundle.py tools/build_repair_campaign_stackability_replay_bundle.py tools/diff_repair_campaign_stackability_replay_bundle.py src/comma_lab/scheduler/repair_campaign_stackability_queue.py src/tac/tests/test_repair_campaign_stackability_queue.py`
  - `0 violations`

## Remaining Work

The replay bundle makes local stackability probes reproducible and diffable, but
the repair scorer still needs to consume the same typed ledger as its default
optimization object. The next implementation step is to promote operator
families and interaction terms into executable campaign rows, then route every
probe result into acquisition/posterior updates instead of leaving it as a local
artifact.
