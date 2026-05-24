# Codex Findings: Packet Member Recompress Runtime Proof

Date: 2026-05-24T19:00:10Z

## Verdict

`packet_member_recompress_v1` now has a file-backed, deterministic
payload-identity proof path. The materializer can prove that ZIP recompression
changed charged archive bytes while preserving the selected member payload, and
the proof is carried into harvest/exact-readiness as member-bound custody
evidence rather than as an unbound pass bit.

This is still not score authority, promotion authority, rank/kill authority, or
paid-dispatch authority.

## Landed Integration

- `materialize_packet_member_recompress_candidate(...)` can emit
  `family_agnostic_runtime_consumption_proof_v1` via
  `runtime_consumption_proof_out`.
- `tools/run_family_agnostic_materializer.py` auto-emits a sibling
  `*.runtime_consumption_proof.json` for packet-member recompress when no
  external proof is supplied.
- `byte_shaving_campaign_queue` declares the proof as an output/pullback
  artifact for packet-member recompress rows.
- `materializer_chain_harvest` validates candidate/source member SHA custody
  when family-agnostic materializer manifests include member records.
- `exact_readiness` rejects family-agnostic proofs whose candidate-member SHA
  does not match the harvested candidate member.
- Zero-byte ZIP members are accepted when the declared and observed member
  sizes both match zero, so valid empty payloads are not rejected by the custody
  verifier.

## Safeguards

- Materializer-generated proof fields explicitly keep score/promotion/rank/kill
  and dispatch authority false.
- The proof verifier still rejects truthy authority fields.
- Exact-readiness checks archive SHA and, when present, candidate-member SHA.
- Exact-ready queues remain no-score dispatch packets and still require lane
  claim plus contest auth eval before any score claim.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_family_agnostic_materializers.py src/tac/tests/test_materializer_chain_harvest_scheduler.py::test_harvest_family_agnostic_packet_recompress_payload_identity_proof src/tac/tests/test_materializer_chain_harvest_scheduler.py::test_harvest_work_queue_family_agnostic_candidate_manifest src/tac/tests/test_optimizer_exact_readiness.py::test_promotes_family_agnostic_packet_member_proof_with_member_binding src/tac/tests/test_optimizer_exact_readiness.py::test_family_agnostic_packet_member_proof_rejects_member_sha_mismatch src/tac/tests/test_optimizer_exact_readiness.py::test_promotes_family_agnostic_candidate_with_receiver_proof src/tac/tests/test_optimizer_exact_readiness.py::test_family_agnostic_runtime_proof_fails_closed_on_invalid_evidence -q`
  - `22 passed`
- `.venv/bin/python -m pytest src/tac/tests/test_staircase_dag.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_family_agnostic_materializers.py src/tac/tests/test_materializer_chain_harvest_scheduler.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_optimizer_exact_readiness.py -q`
  - `217 passed`
- `.venv/bin/python -m ruff check src/tac/optimization/family_agnostic_materializers.py src/tac/optimizer/materializer_chain_harvest.py src/tac/optimizer/exact_readiness.py tools/run_family_agnostic_materializer.py src/comma_lab/scheduler/byte_shaving_campaign_queue.py src/tac/tests/test_family_agnostic_materializers.py src/tac/tests/test_materializer_chain_harvest_scheduler.py src/tac/tests/test_optimizer_exact_readiness.py src/tac/tests/test_byte_shaving_campaign_queue.py`

## Remaining Work

This closes the narrow packet-member recompress proof gap. Archive-section
recode and tensor-factorize still need stronger receiver proofs before they can
clear family-agnostic receiver contracts; tensor-factorize in particular still
requires a cooperative receiver and must stay blocked without one.
