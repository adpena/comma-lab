# Codex Findings: Final-Rate Target Coverage Guard

UTC: 2026-05-27T21:35:55Z

## Finding

The final-rate attack bootstrap had a false-coverage risk: executable materializer
coverage was split between the registry, hard-coded default targets, optional
target blockers, and scorer/DQS1 follow-up loops. That made it too easy for a
future executable materializer to land in the registry without being included in
the archive-rate campaign or explicitly deferred to the correct higher-level
loop.

## Fix Landed

- Added `frontier_final_rate_attack_target_coverage.v1` as a first-class
  bootstrap and CLI artifact.
- Classified archive-rate supported targets:
  `archive_zip_repack_v1`, `packet_member_zip_header_elide_v1`,
  `packet_member_recompress_v1`, `packet_member_merge_v1`,
  `renderer_payload_dfl1_v1`, `archive_section_entropy_recode_v1`, and
  `tensor_factorize_v1`.
- Explicitly deferred `dqs1_pairset_drop_pair` to the DQS1 local-first feedback
  cycle and `inverse_scorer_cell_candidate_v1` to the inverse-steganalysis
  acquisition chain.
- Made unclassified executable candidate materializers fail closed in bootstrap.
- Wrote `frontier_rate_attack_target_coverage.json` from the canonical queue
  builder so normal operator flows expose the coverage contract.

## Verification

- `ruff check` passed for the changed bootstrap, queue-builder, and tests.
- `pytest src/tac/tests/test_frontier_rate_attack_bootstrap.py -q` passed:
  33 tests.
- Queue-builder smoke produced
  `/tmp/pact_frontier_rate_coverage_smoke_20260527_codex/frontier_rate_attack_target_coverage.json`
  with `coverage_complete=true`, no unclassified executable candidate targets,
  false score/dispatch authority, and a valid 3-experiment local queue.

## Next Integration Edge

The archive-rate bootstrap is now self-auditing. The next high-EV bridge is to
make the DQS1 local-first feedback cycle and inverse-steganalysis acquisition
chain emit comparable coverage artifacts so "all executable" has one portfolio
view across archive-rate, DQS1, and scorer-inverse loops.
