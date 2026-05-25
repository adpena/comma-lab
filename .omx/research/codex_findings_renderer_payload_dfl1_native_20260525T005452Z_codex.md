# Codex Findings: Native Renderer Payload DFL1 Materializer

UTC: 2026-05-25T00:54:52Z

## Summary

Landed `renderer_payload_dfl1_v1` as a source-runtime-native family-agnostic
materializer. The candidate converts the canonical robust archive's three
score-affecting renderer members into one short `p` payload:

- `renderer.bin`
- `masks.mkv`
- `optimized_poses.pt`

The payload format is `DFL1` plus fixed-order raw ZIP deflate streams. The
contest-facing unpacker remains standalone and now parses this native DFL1
container directly, so the proof binds to the actual robust source unpacker
instead of a TAC-only parser.

## Real Archive Smoke

Smoke artifact:
`experiments/results/renderer_payload_dfl1_native_smoke_20260525T005941Z/`

- Source archive: `submissions/robust_current/archive_correct.zip`
- Source bytes: `345802`
- Candidate bytes: `345422`
- Saved bytes: `380`
- Candidate member: `p`
- Runtime proof:
  `experiments/results/renderer_payload_dfl1_native_smoke_20260525T005941Z/runtime_consumption_proof.json`
- Native unpack proof: `submissions/robust_current/unpack_renderer_payload.py`
  unpacked all three logical members and SHA-256 matched the source archive;
  `runtime_consumption_probe.native_unpacker_probe.member_sha256s` matched the
  reconstructed renderer/mask/pose members.
- `tac.submission_archive.validate_archive(..., strict=True)` accepted the
  candidate.

## Wire-In

- `tac.optimization.family_agnostic_materializers` owns the executable
  materializer, DFL1 parser, source-runtime-native unpacker probe, and
  target-bound runtime-proof verification.
- `submissions/robust_current/unpack_renderer_payload.py` consumes `p` payloads
  that start with `DFL1` without requiring Brotli.
- `tools/run_family_agnostic_materializer.py` can materialize DFL1 candidates
  and auto-write the runtime-consumption proof.
- `comma_lab.scheduler.byte_shaving_materializer_registry`,
  `byte_shaving_campaign_queue`, and `final_byte_operation_contexts` expose
  DFL1 as executable `packet_member/native_renderer_payload` queue work.
- `tools/run_byte_shaving_materializer_campaign.py` can now generate packet
  artifact-map contexts for non-default packet-member targets, including DFL1.
- `tac.packet_compiler.cooperative_receiver_grammars` registers the DFL1 magic
  so the receiver/compiler packet surface can recognize the native renderer
  payload family.
- `tac.optimizer.materializer_chain_harvest` carries family-agnostic
  `target_kind`, `materializer_id`, and `receiver_contract_kind` through to
  candidate rows, and preserves DFL1 payload anatomy as non-authoritative
  planning signal.
- `tac.optimizer.exact_readiness` rejects family-agnostic runtime proofs whose
  target/materializer/receiver metadata do not match the candidate row.
- `tac.submission_archive` now validates packed renderer payloads by requiring
  logical renderer, mask, and pose groups rather than accepting a parser-only
  payload.

## Authority Boundary

This is not a score claim and not an exact-eval promotion. The candidate remains
blocked by:

- `renderer_payload_dfl1_full_frame_inflate_parity_missing`
- exact auth eval on contest CPU/CUDA before any score, rank, kill, or
  promotion authority

The useful signal is now durable and queue-consumable: this is a byte-closed
real archive materializer with native unpack proof, but the final full-frame
inflate parity and exact-auth gates remain fail-closed.

## Verification

- `.venv/bin/python -m py_compile ...` on touched implementation and test files
- `.venv/bin/python -m ruff check ...`
- `PYTHONPATH=. .venv/bin/pytest src/tac/tests/test_family_agnostic_materializers.py src/tac/tests/test_renderer_packed_payload.py -q`
  - `57 passed`
- `PYTHONPATH=. .venv/bin/pytest src/tac/tests/test_unpack_renderer_payload_stub.py src/tac/tests/test_byte_shaving_campaign.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_final_byte_operation_contexts.py -q`
  - `110 passed`
- `PYTHONPATH=. .venv/bin/pytest src/tac/tests/test_materializer_chain_harvest_scheduler.py src/tac/tests/test_optimizer_exact_readiness.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py -q`
  - `139 passed`
- `PYTHONPATH=. .venv/bin/pytest src/tac/tests/test_family_agnostic_materializers.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_final_byte_operation_contexts.py src/tac/tests/test_byte_shaving_campaign.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_renderer_packed_payload.py src/tac/tests/test_unpack_renderer_payload_stub.py src/tac/tests/test_materializer_chain_harvest_scheduler.py src/tac/tests/test_optimizer_exact_readiness.py -q`
  - `308 passed`
- `PYTHONPATH=. .venv/bin/pytest src/tac/tests/test_cooperative_receiver_integration.py src/tac/tests/test_cooperative_receiver_packet_grammars.py src/tac/tests/test_xray_substrate_classifier.py -q`
  - `56 passed`
- `.venv/bin/python tools/lane_maturity.py validate`

## Next Required Work

1. Add full-frame same-runtime inflate parity for the DFL1 candidate without
   materializing unbounded raw outputs on local disk.
2. Add queue-owned grouped acquisition candidates that compose DFL1 with
   section entropy recode, packet member recompress, tensor factorization, and
   future HNeRV/NeRV-family materializers.
3. Run exact CPU/CUDA auth eval only after parity proof and dispatch-claim
   hygiene are satisfied.
