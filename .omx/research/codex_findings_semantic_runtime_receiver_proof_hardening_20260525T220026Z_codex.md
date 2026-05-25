<!-- SPDX-License-Identifier: MIT -->
---
schema: codex_findings_v1
topic: semantic_runtime_receiver_proof_hardening
created_at_utc: 2026-05-25T22:00:26Z
author: codex
lane_id: lane_codex_semantic_runtime_receiver_proof_hardening_20260525
score_claim: false
promotion_eligible: false
rank_or_kill_eligible: false
ready_for_exact_eval_dispatch: false
dispatch_attempted: false
research_only: false
---

# Semantic Runtime Receiver Proof Hardening

## Finding

The family-agnostic runtime-consumption verifier accepted generic pass flags
for operations that are semantic receiver/runtime rewrites. That is too broad
for `packet_member_merge_v1`, `renderer_payload_dfl1_v1`, and
`tensor_factorize_v1`: parser success or a generic boolean proof must not imply
that the contest runtime actually consumes the transformed representation.

## Landing

Runtime proof verification now adds target-specific blockers for semantic
rewrites:

- `packet_member_merge_v1` requires a packet-member merge runtime adapter probe
  and shadow archive reconstruction proof.
- `renderer_payload_dfl1_v1` requires the DFL1 reconstruction probe, full-frame
  inflate parity, and a passing native unpacker probe.
- `tensor_factorize_v1` requires a tensor-factorize runtime adapter probe and
  shadow archive reconstruction proof.

Generic `passed=true` payloads remain useful as diagnostic evidence, but they no
longer satisfy receiver contract readiness for semantic rewrites.

## Authority Boundary

This hardening is fail-closed. It does not grant score, promotion, rank/kill, or
exact-eval dispatch authority. It only prevents false readiness when a semantic
materializer lacks the target-specific receiver/runtime proof required by its
own contract.

## Verification

The focused regression covers all three target kinds and asserts that generic
boolean pass probes leave `receiver_contract_satisfied=false` with the expected
target-specific blocker.
