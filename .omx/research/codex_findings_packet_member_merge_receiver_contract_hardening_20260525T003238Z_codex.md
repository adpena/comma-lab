# Codex Findings: Packet Member Merge And DFL1 Receiver Contract Hardening

UTC: 2026-05-25T00:32:38Z

## Summary

Hardened `packet_member_merge_v1` so parser-only reconstruction proof cannot
stand in for a byte-closed receiver/runtime consumption proof. Also adopted
and wired the source-runtime-native `renderer_payload_dfl1_v1` materializer so
the robust renderer payload path is part of the queue/DAG/final-byte stack
instead of remaining a side-channel script.

The materializer still records the useful internal reconstruction signal, but
the original-member reconstruction proof now reports:

- `runtime_consumption_probe.passed=false`
- `receiver_contract_satisfied=false`
- `runtime_consumption_proof_passed=false`
- `passed=false`
- `runtime_adapter_ready=false`

This keeps the no-loss signal (`internal_reconstruction_passed=true`) without
granting receiver authority. A compiled receiver/runtime proof remains the
only path that can satisfy the packet-member merge receiver contract.

## Code Wiring

- `tac.optimization.family_agnostic_materializers.materialize_packet_member_merge_candidate`
  now preserves internal reconstruction as `reconstruction_proof_satisfied`
  while refusing receiver authority unless the verification proof has a ready
  runtime adapter.
- The exact-readiness runtime-adapter blocker is appended only when the runtime
  adapter is actually missing.
- Regression coverage now exercises the second-pass path where a real
  `packet_member_merge_runtime_adapter_consumption_proof.v1` is fed back into
  the materializer and clears the runtime-adapter blocker.
- `tac.optimization.family_agnostic_materializers.materialize_renderer_payload_dfl1_candidate`
  packs `renderer.bin`, `masks.mkv`, and `optimized_poses.pt` into short member
  `p` as fixed-order raw ZIP deflate streams with DFL1 magic.
- `submissions/robust_current/unpack_renderer_payload.py` consumes the DFL1
  payload natively, and the materializer proof imports that source unpacker to
  verify the receiver/parser contract.
- `comma_lab.scheduler.byte_shaving_materializer_registry`,
  `byte_shaving_campaign_queue`, and `final_byte_operation_contexts` expose the
  DFL1 target as an executable `packet_member/native_renderer_payload` unit.
- `tools/run_family_agnostic_materializer.py` and the campaign artifact-map
  path can now drive DFL1 via normal materializer queue commands.
- Chain harvest preserves family-agnostic target metadata into candidate rows,
  and exact readiness rejects runtime proofs whose `target_kind`,
  `materializer_id`, or `receiver_contract_kind` do not match the candidate row.
- Sidecar adversarial review found that `packet_member_merge_v1` verification
  still accepted any passing family-agnostic proof with matching archive/member
  SHA. The materializer and CLI runtime path now require
  `packet_member_merge_runtime_adapter_consumption_proof.v1`,
  `packet_member_merge_v1`, `packet_member_merge_adapter`, and
  `family_agnostic_packet_member_merge` before receiver authority can clear.
- DFL1 harvest rows now preserve non-authoritative payload anatomy:
  selected member names, payload member name, selected payload stats, payload
  table, reconstructed member SHA-256s, and native unpacker member SHA-256s.
- `tac.packet_compiler.cooperative_receiver_grammars` registers the DFL1 magic
  as a cooperative receiver packet grammar entry.
- The exact-readiness bridge has a regression proving harvested DFL1 rows with
  native parser proof but `renderer_payload_dfl1_full_frame_inflate_parity_missing`
  do not produce an exact-ready queue.

## Authority Boundary

This change does not create any score, promotion, rank/kill, or exact-eval
authority. DFL1 still carries full-frame inflate parity and exact auth eval
blockers. The packet-member merge change only prevents parser/local
reconstruction proof from being misread as a consumed contest-runtime
transform.
