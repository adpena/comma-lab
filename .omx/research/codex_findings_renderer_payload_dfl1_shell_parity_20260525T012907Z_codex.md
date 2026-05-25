# Codex Findings: Renderer Payload DFL1 Shell Parity Bridge

UTC: 2026-05-25T01:29:07Z

## Summary

The native DFL1 materializer now has a bounded same-runtime shell parity bridge.
The canonical proof shape is `shell_inflate_parity_proof_v2` from
`tools/prove_shell_inflate_parity.py`, not a second DFL1-only runner. It runs
the real `inflate.sh archive_dir output_dir file_list` contract for both source
and candidate archives, hashes the emitted raw outputs, records submission tree
and `inflate.sh` hashes, and deletes scratch/raw output trees by default.

## Authority Boundary

This remains local parity evidence only:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- exact CPU/CUDA auth eval is still required before any score or rank claim

Exact-readiness no longer trusts row-level DFL1 parity flags. It reopens the
proof artifact, verifies the proof SHA, binds source and candidate archive
SHA-256 values, checks full-frame file-list scope, and only then clears the DFL1
full-frame/runtime blockers.

## Wire-In

- `tools/prove_shell_inflate_parity.py` supports multi-entry full-frame file
  lists and writes `shell_inflate_parity_proof_v2`.
- `tac.optimization.family_agnostic_materializers` accepts
  `full_frame_inflate_parity_proof` for `renderer_payload_dfl1_v1` and promotes
  the runtime-consumption proof from parser-smoke to satisfied only when native
  unpack reconstruction and shell parity both pass.
- `tools/run_family_agnostic_materializer.py` exposes
  `--full-frame-inflate-parity-proof`.
- `tac.optimizer.materializer_chain_harvest` carries DFL1 shell parity proof
  path/SHA and verification facts into source queue rows.
- `tac.optimizer.exact_readiness` re-verifies DFL1 shell parity proof artifacts
  before clearing DFL1 source blockers or treating the family proof as backed by
  full-frame parity.

## Verification

- `.venv/bin/python -m py_compile` on touched implementation, tool, and tests
- `.venv/bin/python -m ruff check` on touched implementation, tool, and tests
- `PYTHONPATH=. .venv/bin/pytest src/tac/tests/test_materializer_chain_harvest_scheduler.py src/tac/tests/test_family_agnostic_materializers.py src/tac/tests/test_optimizer_exact_readiness.py -q`
  - `126 passed`
- `PYTHONPATH=. .venv/bin/pytest src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_final_byte_operation_contexts.py src/tac/tests/test_renderer_packed_payload.py src/tac/tests/test_unpack_renderer_payload_stub.py -q`
  - `164 passed`
- `.venv/bin/python tools/lane_maturity.py validate`
- `git diff --check`

## Next Required Work

1. Run the shell parity bridge on the real robust DFL1 candidate with the
   canonical full contest file list, keeping scratch on external storage only if
   inspection is needed.
2. Use the parity proof in the queue-owned materializer campaign path so DFL1
   candidates can move from source queue rows to exact-ready packets without
   manual row edits.
3. Extend the same shell parity proof pattern to section entropy recode,
   packet-member recompress, tensor factorize, and grouped materializer
   composition candidates.
