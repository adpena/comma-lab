# Codex Findings - Frontier Final Rate Attack Autoresolve

UTC: 2026-05-27T14:55:00Z
Agent: codex
Scope: queue-owned final rate attack against current canonical contest-CPU frontier

## Canonical Frontier Input

- Pointer: `.omx/state/canonical_frontier_pointer.json`
- Axis: `[contest-CPU] linux_x86_64_cpu`
- Score: `0.19202062679074616`
- Archive: `experiments/results/v14_v2_dqs1_plus_fec10_substituted_20260526T023000Z/submission_dir/archive.zip`
- Archive sha256: `0a3abfe645c4fac0df9ea89237f25dd9bfc6b2471b897c36d7437795d27d1403`
- Archive bytes: `178546`

## What Landed As System Intelligence

`resolve_current_frontier_archive(...)` now has a bounded default local
submission-packet search. It first consumes auth request records as before, then
checks only canonical source archive locations such as
`experiments/results/*/submission_dir/archive.zip`. This avoids broad recursive
scans over thousands of generated queue archives while allowing
`tools/build_frontier_final_rate_attack_queue.py` to run against the current
frontier without a manual `--archive` override.

Runtime-adapter identity false authority was already re-hardened by the sibling
landing `839615851`; the final queue revalidated that boundary end to end.
Identity/source-runtime materializers now carry `receiver_contract_satisfied`
without claiming `runtime_adapter_ready` unless a concrete adapter dir/file
identity exists.

## Queue Runs

- Historical guard-failure run:
  `.omx/research/frontier_final_rate_attack_20260527T143625Z`
  - Queue id: `frontier_final_rate_attack_runtime_identity_20260527T1444Z`
  - Outcome: worker commands succeeded, observer unhealthy
  - Signal preserved: optimizer source queues claimed runtime adapter readiness
    without runtime tree identity.

- Intermediate autoresolve repair attempts:
  `.omx/research/frontier_final_rate_attack_20260527T144649Z`
  `.omx/research/frontier_final_rate_attack_20260527T144824Z`

- Final healthy run:
  `.omx/research/frontier_final_rate_attack_20260527T144924Z`
  - Queue id: `frontier_final_rate_attack_autoresolve_20260527T1510Z`
  - State DB: `.omx/state/experiment_queue_frontier_final_rate_attack_autoresolve_20260527T1510Z.sqlite`
  - Observer verdict: healthy, zero blockers
  - Worker result: 10/10 steps succeeded, no command failures
  - Local CPU elapsed sum reported by observer: about 5.99 seconds

## Empirical Rate Result

Executable materializers for the current single-member frontier archive:

- `packet_member_zip_header_elide_v1`: `saved_bytes=0`, `rate_positive=false`
- `packet_member_recompress_v1`: `saved_bytes=0`, `rate_positive=false`

Both exact-readiness bridges skipped dispatch with
`materializer_candidate_not_rate_positive_for_exact_readiness`. This is the
correct fail-closed result: the current `x` member archive has no ZIP header
or stdlib recompress savings left at this layer.

## Blocked Materializer Classes

The queue recorded typed omissions rather than silently dropping work:

- `packet_member_merge_v1`: blocked because current archive has one ZIP member.
- `renderer_payload_dfl1_v1`: blocked because current archive lacks
  `renderer.bin`, `masks.mkv`, and `optimized_poses.pt`.
- `archive_section_entropy_recode_v1`: blocked because no valid section
  manifest was derivable for this packet.
- `tensor_factorize_v1`: blocked because no tensor manifest and factorization
  contract/rank were supplied.

## Next Engineering Implication

For the current PR110/PR111 single-member selector-stream frontier, the next
rate attack must move upstream of generic ZIP/header surfaces:

1. Selector-stream/internal payload coders and PacketIR-level transforms.
2. Archive/member grammar materializers that understand the FEC10/DQS1 packet.
3. Distortion-budget attacks using freed rate only after a positive rate move.
4. Substrate-family materializers for HNeRV/BoostNeRV/NeRV/non-NeRV archives
   where DFL1, tensor, codebook, and receiver-runtime classes are structurally
   applicable.

This run should demote generic `packet_member_recompress_v1` and
`packet_member_zip_header_elide_v1` for matching single-member stored archives,
not globally retire them.

