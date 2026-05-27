# Codex Findings: Final-Rate Feedback State And PR103 Guard

UTC: 2026-05-27T15:15:47Z
Agent: Codex
Scope: final-rate materializer feedback loop, auxiliary queue execution, byte-range chain context binding

## Finding

The queue-owned final-rate feedback loop now has a concrete fail-closed boundary for stale byte-range contexts. The current contest-CPU frontier archive regenerated two local materializer sweeps:

- `packet_member_zip_header_elide_v1`
- `packet_member_recompress_v1`

Both were receiver-valid but saved `0` bytes on archive SHA `0a3abfe645c4fac0df9ea89237f25dd9bfc6b2471b897c36d7437795d27d1403`, so the bridge demoted both with `receiver_contract_satisfied_but_no_archive_delta`. This is reusable planner signal, not a score claim.

## Bug Class Closed

The operation-chain byte-range stage previously inherited old PR103 arithmetic-code fixtures when the current chain had no payload-grammar schema/runtime/archive binding. That was false authority for PR110/current-frontier work and would also be false authority for other videos or corpora.

The byte-range stage now disables default PR103 context whenever the chain is unbound and missing `payload_grammar_schema_manifest`. The regenerated feedback cycle emits:

- `default_pr103_context_disabled=true`
- `default_pr103_context_disable_reason=unbound_chain_missing_payload_grammar_contract`
- `byte_range_stage_default_pr103_context_disabled_for_unbound_chain`
- missing schema/beam/runtime/archive/member blockers

This keeps contest-video overfitting explicit: the active corpus can be `upstream/videos/0.mkv`, but every materializer and chain must carry its own archive/runtime/payload-grammar proof instead of silently borrowing another PR or corpus.

## Automation Fix

Bounded auxiliary queue execution now rewrites embedded `--state .omx/state/experiment_queue_<queue>.sqlite` command arguments to the same artifact-local SQLite state used by the worker. This prevents stale canonical state and prevents harvest steps from missing materializer steps that succeeded in the isolated queue-owned run.

Proof artifact:

- `.omx/research/frontier_rate_attack_feedback_cycle_final_rate_regen_20260527T1625Z/initial_refresh/operation_materializer_execution_queue.artifact_state.json`
- rewrite count: `2`
- failed auxiliary queues: `0`

## Current Artifacts

- Regenerated final-rate sweeps: `.omx/research/frontier_final_rate_attack_regen_20260527T1620Z/`
- Feedback cycle: `.omx/research/frontier_rate_attack_feedback_cycle_final_rate_regen_20260527T1625Z/`
- Byte-range stage inputs: `.omx/research/frontier_rate_attack_feedback_cycle_final_rate_regen_20260527T1625Z/results/frontier_operation_chain_compiler/frontier_rate_attack_feedback_final_rate_regen_20260527t1625z_chain_compiler/chain_registered_multisurface_materializer_program/byte_range_stage_inputs.json`

## Next Integration

The next executable bridge is not more ZIP/member recompression on this archive class. It is a current-corpus payload grammar and receiver binding for upstream/internal entropy positions: selector streams, archive sections, tensor/latent payloads, or HNeRV/NeRV substrate archives. The same machinery should accept contest-video-specific contexts and corpus-level contexts such as comma10k19 by explicit manifest, not by hardcoded defaults.
