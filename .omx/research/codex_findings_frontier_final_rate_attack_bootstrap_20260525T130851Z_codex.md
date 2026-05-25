# Codex Findings - Frontier Final-Rate Attack Bootstrap

UTC: 2026-05-25T13:08:51Z

## Summary

Landed and exercised a queue-owned bootstrap for the PR110/current-frontier final
rate attack. The new surface resolves the canonical current `[contest-CPU]`
frontier archive from `.omx/state/canonical_frontier_pointer.json`, accepts
explicit comparison archives such as the PR110/FEC6 packet, inspects ZIP
members, and compiles existing family-agnostic materializer sweeps into
`experiment_queue.v1`.

This is local materializer signal only. It does not claim score, promotion,
rank/kill authority, or exact-readiness.

## Artifacts

- Bootstrap/control artifacts:
  `.omx/research/pr110_current_frontier_final_rate_attack_20260525T130851Z/`
- Queue artifact:
  `.omx/research/pr110_current_frontier_final_rate_attack_20260525T130851Z/experiment_queue.json`
- Queue state:
  `.omx/state/experiment_queue_pr110_current_frontier_final_rate_attack_20260525T130851Z.sqlite`
- Materializer outputs:
  `/Volumes/VertigoDataTier/experiments/results/frontier_final_rate_attack/pr110_current_frontier_final_rate_attack_20260525T130851Z/`

## Empirical Result

Executed two queue steps against two archives:

- `current_contest_cpu_frontier`
  - score axis in pointer: `[contest-CPU]`
  - score in pointer: `0.19202828295713675`
  - archive bytes: `178559`
  - archive SHA-256: `7a0da5d0fc327cba3f7d1387a544fd5ce5f05bc56ecc8e12cd5097141672f4fe`
- `pr110_fec6`
  - archive bytes: `178517`
  - archive SHA-256: `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`

Both currently executable packet-member materializer sweeps completed:

- `packet_member_zip_header_elide_v1`: `rate_positive_count=0`,
  `max_saved_bytes=0`
- `packet_member_recompress_v1`: `rate_positive_count=0`,
  `max_saved_bytes=0`

The queue worker reported `succeeded=2`, `failed_command_count=0`.

## Canonical Signal

The current packet-member-only surface appears saturated for both the current
CPU frontier archive and the PR110/FEC6 base under this local materializer
family. The next materializer expansion must move above ZIP member metadata:

- `archive_section_entropy_recode_v1` is blocked by missing section manifest.
- `tensor_factorize_v1` is blocked by missing tensor manifest plus
  factorization rank/contract.

These blockers are encoded as typed `frontier_rate_attack_target_omission.v1`
rows in the bootstrap artifact rather than being left as chat-only analysis.

## Next Implementation Hooks

1. Add a current-frontier section/tensor manifest builder that can derive
   candidate sections/tensors from HNeRV-style archives, BoostNeRV bolt-ons,
   and non-NeRV packets.
2. Feed the emitted `family_agnostic_materializer_empirical_observation.v1`
   JSONL files into the existing acquisition/water-bucket feedback path so
   saturated packet-member operations are demoted automatically for matching
   archive classes.
3. Extend the same queue bootstrap with grouped operation sets once section and
   tensor manifests exist, so the next run tests combinations rather than
   isolated packet-member leaves.
4. Keep outputs on `VertigoDataTier` first. The committed `.omx/research`
   control artifacts are small; materialized candidate payloads stay off the
   local repo disk.
