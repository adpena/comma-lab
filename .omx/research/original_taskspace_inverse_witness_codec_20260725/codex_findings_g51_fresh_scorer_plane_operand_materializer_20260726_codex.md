# G51 Fresh Scorer-Plane Operand Materializer — Closure

## Result

The fresh source-derived n600 operand compiler is complete. It independently
maps only sealed `gt_f0` and `gt_f1` through the exact integer
`DisjointResizeOperator` and half-up uint8 rounding. It never reads an older
C1/V15 archive, plane, receipt, payload, or candidate target.

- Aggregate:
  `/Volumes/VertigoDataTier/pact/taskspace_fresh_scorer_planes_n600_20260726/aggregate_receipt.json`
- Aggregate file SHA-256:
  `ae9048dfc24947a6268315590b65da02b56549379e347cbaced25e2e6f67d915`
- Aggregate sealed self-hash:
  `4363827c2aeb613916029d8bacde8aeb4ded961c4d1ca297310a1e53e204619c`
- Coverage: five immutable 120-pair stages, chronological 600/600.
- Loader:
  `FreshScorerPlaneOperandLoaderV1.open(path, expected_sha256=...)`
  followed by `iter_stages(max_pairs=120)` or `iter_chunks(max_pairs)`.
- Validation: Ruff clean; focused pytest `7 passed`; recursive aggregate
  dependency/file validation and read-only mmap reopen passed.

## Governed execution

Attempt 01 correctly failed before any stage: 2,267 MiB peak exceeded the
2,048 MiB process-group cap. Its immutable log was preserved. Attempt 02 used
the identical sealed config with an 8 GiB governed reservation/cap and
completed in 50.256 seconds at 3,862 MiB peak RSS.

The sealed attempts receipt is:
`/Volumes/VertigoDataTier/pact/taskspace_fresh_scorer_planes_n600_20260726/governed_launch_attempts_receipt.json`
(file SHA-256
`fc02692e0ca674c3a0cc2b67d09ad13611b2f4e4fa67191706cee41d343a14bd`;
self-hash
`5db71ff1a379156868f0068da6ed9547a083dfc87dc26151c15fc78a6c1cb8c3`).

## Truth boundary and next consumer

This is `DIRECT_TASK_LAYERED` source/scorer operand custody. The current
batch-16 target-label bank is bound. `gt_poses` is explicitly sealed
source-cache advisory evidence, not a fresh batch-16 pose target and not pose
authority. No fresh V15 semantic predictor/base stream is included, so
`PROGRAM_RESIDUAL_LAYERED` remains owed to G53/G52 composition.

This artifact is not a candidate, is not a score, and did not move the
frontier pointer. Its purpose is immediate injection into the G52 whole-n600
lossy plane-codec race, followed by public receiver closure and exact
authoritative evaluation.
