# Task #543 production receiver/archive implementation spec

Date: 2026-07-19
Lane: `production_receiver_543_20260719`
Authority: delegated operator GO to build only; no launch, paid dispatch, score,
submission, or frontier-pointer authority. `launch_ready=false` is invariant.

## Objective

Land a production, scorer-free archive builder and inflate path that consumes a
compact scorer-plane description, realizes deterministic camera-resolution
uint8 frame pairs through the factor-2 lattice construction, writes the contest
`<video>.raw` layout atomically, and proves exact parse-back on at least twelve
real pairs. Every video-derived payload byte must be inside `archive.zip`.

## Archive and arithmetic contract

- One monolithic `0.bin` member uses a fixed magic/version prefix, a canonical
  JSON header, and an exact ordered sequence of length-prefixed sections.
- Required sections: `y_description`, `frame0_policy`; optional
  `quotient_residual`. The y codec id is exactly one of `raw-uint8-y`,
  `brotli-y`, or the fail-closed `witness-y-stub`.
- The header binds geometry, pair count, section byte lengths and SHA-256s,
  counted byte totals, and receiver/tie-policy ids. Parsing refuses truncation,
  duplicate/unknown sections, hash drift, size-cap breaches, and trailing data.
- Decode contains no scorer, scorer weight, GT argmax table, source frame, or
  hidden video lookup. Encode may select y under native CPU-Torch float32
  semantics with first-max class-index tie policy; decode validates the declared
  policy id but never recomputes logits.
- For an integer uint8 scorer plane, factor-2 realization uses the exact
  disjoint half-pixel support operator. The deterministic canonical feasible
  point fills every owned support tap with the target byte; all unowned camera
  coordinates are zero. This is integer-only and gives exact numerator equality
  because each support's coefficients sum to the common denominator. Overlap or
  geometry drift refuses. Residual addition is ordered, signed-int16, saturating
  to uint8.
- Frame0 policy initially supports `repeat-frame1`, a generic zero-byte policy.
  Output order is pair-index ascending, frame0 then frame1, C-contiguous RGB.
- Per-pair stage files and state are atomically write-once/resumable; final raw is
  assembled to a `.partial`, size/hash checked, and atomically promoted. Storage
  preflight covers final output plus preserved stages. Manifest writes are
  atomic and refuse overwrite.

## Owned implementation surfaces

- `src/tac/witness_dsl/v10_production_receiver.py` — grammar, builder, parser,
  deterministic receiver, storage/resume/tree-hash helpers, CLI-compatible API.
- `tools/build_v10_production_archive.py` — encode-side packet/archive builder.
- `tools/inflate_v10_production_archive.py` — scorer-free decode CLI matching
  `inflate.sh` arguments (`archive_dir`, `output_dir`, `video_names_file`).
- `src/tac/optimization/uint8_lattice_feasibility.py` — additive exact integer
  scorer-plane factor-2 specialization and verifier only; preserve existing
  solver semantics.
- `src/tac/witness_dsl/v10_compiler_receiver.py` — additive factor-2 route,
  handler, receipt/contract id, and completeness surface; factor 10 remains
  missing and `launch_ready` remains false.
- focused behavioral tests under `src/tac/tests/`.
- `.omx/research/production_receiver_543_byteclose_receipt_20260719.json` and
  `.omx/research/production_receiver_543_20260719_codex.md`.

## Acceptance criteria

1. `python3 -m pytest -q src/tac/tests/test_v10_production_receiver.py src/tac/tests/test_v10_compiler_receiver.py src/tac/tests/test_uint8_lattice_feasibility.py`
   passes.
2. Builder round-trip proves each section hash/length and exact stream
   consumption; malformed length/hash/trailing bytes and witness stub refuse.
3. Double decode into distinct roots produces identical raw SHA-256 and full
   tree hash, with exact expected bytes and no scorer import/load path.
4. Every decoded frame passes `apply_numerators(frame) == y * denominator` for
   every block and channel. Tests include at least twelve real cache-derived y
   planes in the durable measurement, plus bounded synthetic regressions.
5. Resume after an interrupted prefix reopens and revalidates preserved pair
   stages before continuing; edited stage/state/archive bytes refuse.
6. Factor 2 has one counted compiler route and authoritative consumption receipt,
   is no longer listed missing, factor 10 remains missing, and compile output
   still reports `launch_ready=false`, `score_claim=false`,
   `promotion_eligible=false`.
7. Receipt records source/cache/archive/runtime hashes, exact command, host,
   dependencies, pair ids, MEASURED subset runtime, DERIVED n600 projection,
   30-minute comparison, exact numerator proof totals, both double-decode tree
   hashes, arithmetic/tie declaration, storage preflight, and pointer unmoved.

## Explicit non-targets

No edit to `upstream/`, `submissions/exact_current/*`, the sacred result tree,
or existing solver behavior. No scorer at inflate, hard-oracle repair at
inflate, training, GPU/provider dispatch, exact evaluator, score claim, pointer
move, or `launch_ready=true`. `witness-y-stub` is a typed refusal surface, not a
fake implementation.

MAIN must independently review archive byte accounting, parser bounds,
resumability, scorer exclusion, deterministic arithmetic, the factor-2 compiler
receipt, real-pair custody, and every MEASURED/DERIVED label before landing.
