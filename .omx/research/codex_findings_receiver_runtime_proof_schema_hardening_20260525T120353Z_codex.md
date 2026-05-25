# Codex Findings: Receiver Runtime Proof Schema Hardening

UTC: 2026-05-25T12:03:53Z
Status: implemented; local proof-axis evidence only

## Summary

The receiver path now fails closed on runtime-consumption proof schema drift and
on materializer/proof-family cross-use. This closes a bug class where a proof
with the right payload hashes but the wrong family metadata could be consumed as
receiver evidence by a different materializer.

Code changes:

- `src/tac/optimization/family_agnostic_materializers.py` defines one canonical
  runtime-proof schema constant and uses it across emitted proof payloads.
- `verify_runtime_consumption_proof(...)` now rejects proof schema mismatch
  with `runtime_consumption_proof_schema_mismatch`.
- `packet_member_zip_header_elide_v1` now binds proof kind, receiver contract
  kind, target kind, and materializer id during verification instead of relying
  only on payload identity.
- Tests cover wrong-schema rejection and wrong proof-family metadata rejection.

This is not score authority and does not promote any archive. It strengthens the
receiver proof gate consumed by materializer sweeps, queue feedback, and dynamic
sparse feedback hints.

## Current CPU Frontier Rate-Attack Smoke

I ran a bounded receiver-aware materializer sweep against the current scanner
`[contest-CPU Linux x86_64]` frontier archive:

- Archive:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/pairset_drop_one_rank021_pair0371_materialization_20260522T180446Z/submission_dir/archive.zip`
- Archive SHA:
  `7a0da5d0fc327cba3f7d1387a544fd5ce5f05bc56ecc8e12cd5097141672f4fe`
- Frontier score being attacked:
  `0.19202828295713675 [contest-CPU Linux x86_64]`
- Output root:
  `experiments/results/codex_current_cpu_frontier_rate_attack_receiver_smoke_20260525T120353Z/`

Results:

- `packet_member_zip_header_elide_v1`: 0 saved bytes, receiver proof satisfied,
  planner feedback demotes matching archive class for this target.
- `packet_member_recompress_v1`: 0 saved bytes, receiver proof satisfied,
  planner feedback demotes matching archive class for this target.

Interpretation: the already-current CPU frontier ZIP has no generic header or
member-recompression slack left for these two family-agnostic materializers.
That is useful negative signal, not a lane failure. The next comprehensive run
should route toward DQS1-aware pairset/selector operations, archive-section
entropy recoding where a valid section manifest exists, and receiver-aware
compositions rather than repeating generic ZIP/member transforms on this exact
archive class.

## Next Wiring

The immediate next queue-owned attack should use this negative feedback as
training signal:

1. Feed these sweep outputs into the feedback/replan surface as demotion rows.
2. Use the receiver-gated dynamic sparse compiler hint only when a receiver
   positive rate-saving row exists.
3. Expand from generic ZIP/member transforms to DQS1-aware operations and
   composition candidates, then run local CPU/MLX advisory before exact auth
   anchoring.

