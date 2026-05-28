# Codex Findings: Range/ANS Entropy Probe Materializers

Timestamp: 2026-05-28T18:31:33Z

## Verdict

Range and ANS are no longer only missing rows in archive entropy coverage. The
byte-transform executor now emits deterministic archive-bound entropy probes for
both coder families, computes zero-order byte entropy lower bounds on the
selected payload member, and writes hashable probe artifacts into the replay
bundle surface.

## What Changed

- Added `range_coder_entropy_probe` and `ans_coder_entropy_probe` archive-native
  variants.
- Each probe records selected ZIP member custody, source archive SHA/bytes,
  zero-order entropy, theoretical lower-bound bytes, and estimated redundancy.
- Coverage now distinguishes `probe_only_materializer_missing` from plain
  `not_materialized`, preserving the difference between "measured opportunity"
  and "not investigated."
- Stack-search and floor-loop summaries expose probed entropy substrates and
  probe counts so automation can route range/ANS materializer work directly.

## Authority Contract

The probes are not coders and do not emit a candidate archive. They remain
fail-closed with `range_coder_materializer_missing`,
`ans_coder_materializer_missing`, and runtime-adapter blockers until real
encoder/decoder materializers exist. No score, promotion, rank/kill, or exact
dispatch authority is granted.

## Verification

- `.venv/bin/ruff check --fix src/tac/optimization/repair_family_byte_transform_executor.py src/tac/optimization/repair_archive_entropy_substrate_coverage.py src/tac/optimization/repair_family_stack_search.py tools/run_repair_campaign_autonomous_floor_loop.py src/tac/tests/test_repair_family_materializers.py src/tac/tests/test_repair_campaign_materialization_queue.py`
- `.venv/bin/python -m py_compile src/tac/optimization/repair_family_byte_transform_executor.py src/tac/optimization/repair_archive_entropy_substrate_coverage.py src/tac/optimization/repair_family_stack_search.py tools/run_repair_campaign_autonomous_floor_loop.py`
- `.venv/bin/pytest src/tac/tests/test_repair_family_materializers.py -q`
- `.venv/bin/pytest src/tac/tests/test_repair_campaign_materialization_queue.py -q`
