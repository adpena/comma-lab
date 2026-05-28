# Codex Findings - Archive Family Fingerprint Coverage

UTC: 2026-05-28T07:00Z

## Result

The repair/final-rate path was too FEC6-centered. I added a reusable archive
family fingerprint surface and wired it into the repair byte-transform executor
so unsupported archive families become explicit fail-closed adapter gaps instead
of implicit filename lore.

## Implemented

- `tac.optimization.archive_family_fingerprint` fingerprints ZIP archives by
  payload grammar and member layout.
- `tools/inspect_repair_archive_family_coverage.py` emits an operator-visible
  coverage report for arbitrary archive sets.
- `repair_family_byte_transform_executor` now embeds the richer fingerprint in
  its archive-family probe while preserving false authority.

Recognized families now include FEC3, FEC5, FEC6, FEC8, FES1, FP11 selector
wrappers, PSV3/PSV4 PACT-NeRV packets, HDM latent sidecars, DFL1/RPK1/ASYM
renderer payloads, raw HNeRV payloads, single-file payload archives,
multi-member runtime archives, and generic ZIP payloads.

## Live Evidence

Coverage artifact:
`.omx/research/repair_archive_family_coverage_20260528T065957Z.json`

Representative live archive census:

- archives inspected: 10
- implemented score-affecting adapter families: `fec6_fixed_huffman_k16_selector`
- unsupported score-affecting adapter families observed: `fec3_compact_selector`,
  `fec5_fixed_huffman_k8_selector`, `fec8_static_second_order_k16_selector`,
  `fes1_all_none_selector`, `pact_nerv_selector_v4_packet`,
  `hnerv_latent_sidecar_hdm`, `renderer_dfl1_payload`, `renderer_rpk1_payload`,
  `renderer_asym_payload`

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_archive_family_fingerprint.py src/tac/tests/test_repair_family_materializers.py::test_byte_transform_executor_mutates_fec6_selector_payload_when_detected -q`
  - 4 passed
- `.venv/bin/python -m ruff check src/tac/optimization/archive_family_fingerprint.py src/tac/optimization/repair_family_byte_transform_executor.py tools/inspect_repair_archive_family_coverage.py src/tac/tests/test_archive_family_fingerprint.py src/tac/tests/test_repair_family_materializers.py`
  - clean
- live CLI run wrote the coverage artifact above and preserved
  `score_claim=false`, `promotion_eligible=false`,
  `ready_for_exact_eval_dispatch=false`

## Next adapter priorities

1. FEC3/FEC5/FEC8/FES1 selector semantic mutators.
2. PSV3/PSV4 packet parser/mutator for PACT-NeRV repair.
3. HDM sidecar adapter for HNeRV-family latent repair.
4. DFL1/RPK1/ASYM renderer payload adapters.
