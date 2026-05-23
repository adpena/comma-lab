# Codex Findings: Optimizer Queue Schema Ranking Hardening

UTC: 2026-05-23T00:14:03Z

## Axis

This is an infrastructure hardening finding. It does not claim score movement.
All affected rows remain planning/proxy evidence until exact auth-eval gates
promote a byte-closed candidate on a contest axis.

## Finding

The optimizer candidate queue still had two residual false-authority surfaces:

1. Unknown JSON payloads with a top-level `candidates[]` array were adapted as
   `codec_op_param_sweep_manifest` rows even when the source schema had no
   explicit adapter.
2. The queue sorter could rank bare `predicted_score`, `proxy_score`, or
   `macos_cpu_score` fields even when a known adapter had not converted them
   into an intentional `rank_score`.

That made unknown future optimizer outputs too easy to promote into the
planning top-k with incomparable numeric fields.

## Fix Landed

Commit: `90ceae352` (`Harden optimizer queue schema ranking`)

Changes:

- `src/tac/optimizer/candidate_queue.py`
  - replaced the loose extraction tuple with `SourceExtraction`;
  - removed generic `candidates[]` adaptation for unknown schemas;
  - records unsupported sources via `unsupported_sources` and per-source
    diagnostics instead of dropping the signal silently;
  - limits sorting to rank fields emitted by known adapters.
- `src/tac/tests/test_optimizer_candidate_queue.py`
  - added a regression test proving unknown `candidates[]` rows with attractive
    proxy/local numbers do not enter `top_k`;
  - added a regression test proving raw proxy/local score fields do not affect
    sort order unless an adapter emits `rank_score`.

Verification:

- `.venv/bin/python -m ruff check src/tac/optimizer/candidate_queue.py src/tac/tests/test_optimizer_candidate_queue.py`
- `.venv/bin/python -m pytest -q src/tac/tests/test_optimizer_candidate_queue.py`
- `.venv/bin/python -m pytest -q src/tac/tests/test_cathedral_consumer_payload_passthrough.py src/tac/tests/test_mlx_dynamic_learned_sweep.py src/tac/tests/test_optimizer_candidate_queue.py src/tac/tests/test_representation_training_manifest_writer.py src/tac/tests/test_representation_training_probe_integration.py src/tac/tests/test_pr95_muon_local_training_integration.py src/tac/tests/test_run_pr95_local_training_probe.py src/tac/tests/test_proxy_candidate_contract.py src/tac/tests/test_dqs1_local_first_queue_builder.py src/tac/tests/test_local_cpu_contest_drift.py`

## Follow-Up

Remaining adjacent work:

- Canonicalize DQS1 post-advisory eureka generation as a queue step or strict
  queue-builder output rather than a manual side action.
- Harden local CPU drift eureka validation so all authority fields are required
  and exactly false, not merely absent or optional.
- When rank007 completes, harvest only as `[macOS-CPU advisory]`, build the
  drift eureka, reroute the local-first queue, and reserve exact CPU/CUDA auth
  spend for explicit eureka/dispatch gates.
