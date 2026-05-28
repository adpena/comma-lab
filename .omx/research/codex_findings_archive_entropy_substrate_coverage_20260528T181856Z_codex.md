# Archive entropy substrate coverage closure

Codex landing: repair byte transforms and composed chains now expose a typed
archive/entropy substrate coverage matrix.

## What changed

- Added `repair_archive_entropy_substrate_coverage.v1`.
- Byte-transform reports now classify FEC variants, ZIP/member repack, header
  rewrite, selector streams, range coding, ANS coding, Huffman coding,
  pre-coder shaping, coder-boundary recoding, and post-coder legal repack.
- Chain stage reports and chain bundles carry the substrate coverage rows
  forward, and the autonomous floor loop exposes the coverage count and rows.

## Current coverage truth

- Materialized today: FEC selector variants, selector streams, Huffman selector
  families, PSV4 header rewrite, packet/member entropy-boundary recompress, and
  ZIP legal repack when the archive supports them.
- Explicitly not materialized yet: range coder and ANS coder materializers.
  They are now visible as typed blockers rather than lost as prose.

## Verification

- `ruff check --fix` on touched files: passed
- `py_compile` on touched Python modules/tools: passed
- selector/FEC/header/chain/floor-loop sentinels: passed
- `pytest src/tac/tests/test_repair_family_materializers.py -q`: 25 passed
- `pytest src/tac/tests/test_repair_campaign_materialization_queue.py -q`: 10 passed

## Review notes

This makes the before-coder / coder-boundary / after-coder entropy principle
operator-visible and planner-consumable. The next concrete closure is to turn
the range and ANS blockers into executable materializer variants under the same
coverage schema.
