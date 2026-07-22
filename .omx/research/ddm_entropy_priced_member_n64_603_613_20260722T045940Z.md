---
title: DDM entropy-priced member solve v3 at n64
date_utc: 2026-07-22T04:59:40Z
task: 603
master_task: 578
feeds_task: 613
lane_id: lane_ddm_mdl_member_solve_v3_entropy_603_613_20260722
research_only: true
execution_allowed: false
score_claim: false
verdict: MEASURED_ENTROPY_RATE_GRADIENT_GREEN_ABSOLUTE_MEMBER_CONSTRAINT_INFEASIBLE_N64
verdict_scope: Lossless six-stream entropy recoding and the exhaustive n64 power set of three maximal safe-zero residual collapses; wider direct-description and member families remain open
main_landing_review_required: true
---

# Outcome

The n64 solve replaced v2's fixed-record rate wall with an exact variable-length archive grammar.
Lossless recoding of the identical six semantic payloads reduced the receiver-consumed archive from
`274,664` to `45,369` bytes (`-229,295`, `-83.482%`) without changing frozen-SegNet membership
`0.493605613708` or Pose6 completeness `1.000000000000`. Exact candidate archive sizes then varied
over eight exhaustive safe-zero subsets:

`45,369, 34,657, 31,431, 20,719, 31,203, 20,491, 17,265, 6,553` bytes.

That is a real MDL gradient, not an estimate. It does not produce an admissible knee: no candidate
satisfies the absolute per-stratum membership constraint at any of the five tolerance rungs. The
published rung bytes therefore remain the 45,369-byte infeasible diagnostic baseline at every rung,
not candidate archives.

# Per-stream measured tournament

The full historical coder menu requested by the operator was measured per stream with no recency
floor: Brotli-Q11, LZMA preset-9-extreme, existing AQc1 arithmetic, PR101 rank-Huffman when its
alphabet bound applied, and split metadata plus existing Rice/Golomb or zlib value coding. Temporal
delta, canonical sparse records, dense temporal bitmap, and dense temporal colex transforms competed.
No new entropy coder was invented.

| Semantic stream | Selected transform | Selected coder | Semantic bytes | Coded bytes | Exact ZIP-home bytes |
|---|---|---|---:|---:|---:|
| global anchors | pair temporal delta | Brotli-Q11 | 768 | 118 | 338 |
| axial gradients | pair temporal delta | split metadata + Rice/Golomb | 1,920 | 309 | 531 |
| low residual | canonical sparse records | LZMA | 90,112 | 11,896 | 12,134 |
| mid residual | canonical sparse records | LZMA | 90,112 | 15,924 | 16,162 |
| high residual | canonical sparse records | LZMA | 90,112 | 15,316 | 15,556 |
| Pose6 codes | pair temporal delta | split metadata + Rice/Golomb | 512 | 414 | 626 |

The six unique homes sum with 22 container-only bytes to exactly `45,369`. AQc1 and rank-Huffman
were measured candidates but never won a stream; that is not a negative verdict on either family.

# Candidate byte gradient

| Mask | Collapsed residual streams | Exact bytes | Overall membership | Pose completeness |
|---:|---|---:|---:|---:|
| 0 | none | 45,369 | 0.493605613708 | 1.000000000000 |
| 1 | low | 34,657 | 0.489909092585 | 1.000000000000 |
| 2 | mid | 31,431 | 0.494335810343 | 1.000000000000 |
| 3 | low + mid | 20,719 | 0.337850968043 | 1.000000000000 |
| 4 | high | 31,203 | 0.476289113363 | 1.000000000000 |
| 5 | low + high | 20,491 | 0.493612130483 | 1.000000000000 |
| 6 | mid + high | 17,265 | 0.482733805974 | 1.000000000000 |
| 7 | low + mid + high | 6,553 | 0.493612130483 | 1.000000000000 |

The apparent aggregate improvement in masks 5 and 7 is not efficacy. At mask 7, Road, Lane, MyCar,
and Movable each have escape fraction `1.000000000000`; Undrivable alone has escape `0`. Boundary
and overall escape are `0.881302230633` and `0.506387869517`. Absolute strata correctly refuse it.

# Tolerance-versus-bytes curve

| Rung | Allowed escape | Feasible candidates | Selected diagnostic bytes | Per-stream ZIP-home bytes | Membership | Pose |
|---|---:|---:|---:|---|---:|---:|
| exact | 0.000000 | 0 | 45,369 | 338/531/12,134/16,162/15,556/626 | 0.493605613708 | 1.000000000000 |
| 1 | 0.000152 | 0 | 45,369 | 338/531/12,134/16,162/15,556/626 | 0.493605613708 | 1.000000000000 |
| 2 | 0.000300 | 0 | 45,369 | 338/531/12,134/16,162/15,556/626 | 0.493605613708 | 1.000000000000 |
| 3 | 0.000500 | 0 | 45,369 | 338/531/12,134/16,162/15,556/626 | 0.493605613708 | 1.000000000000 |
| 4 | 0.000800 | 0 | 45,369 | 338/531/12,134/16,162/15,556/626 | 0.493605613708 | 1.000000000000 |

Task #613 receives a measured value-responsive byte curve, but no feasible knee under this absolute
constraint and candidate family.

# Receiver, resume, and custody closure

- Final receipt: `446,613` bytes, SHA-256
  `9ab2251e4826d56fd0498ffa861a48f9cf699dfde082f9f55338e5b5ed59ca04`.
- Lossless entropy baseline archive: `45,369` bytes, SHA-256
  `cef61eb5aed49842fa41eba2092fa930a72f56b7506168ab983e42bc2a91abbb`.
- Compiler determinism x2, parse/re-encode identity, streaming decode determinism x2, all six
  semantic payload roundtrips, exact coded-section consumption, unique byte homes, 18 semantic
  no-op samples, and 39 archive-home fail-closed samples are green.
- The run stopped after candidate mask 3 and rung 1, resumed from disk, and preserved all eight
  candidate plus five rung checkpoints. Independent post-run replay validated every envelope and all
  five published archive parse/receive paths.
- The first candidate checkpoint to final-receipt artifact-write span was 294 seconds, below the
  delegated ten-minute n64 bound. This span is not presented as a benchmark.
- Lane maturity marks for implementation and real-artifact evidence are green (L2). The repository-
  wide lane validator separately reports 110 inherited missing historical evidence paths; this arm
  did not claim or repair those unrelated gates.
- SSD target/cache inputs remained read-only. No raw source, scorer weights, candidate claim, paid
  dispatch, GPU work, contest evaluation, deletion, movement, or pointer change occurred.

# Blocker delta

1. `COUNTED_ARCHIVE_MDL_INSIDE_SOLVE`: green; exact value-responsive entropy ZIP length is the rate.
2. `VARIABLE_LENGTH_RECEIVER_GRAMMAR`: red to green for all six n64 streams end to end.
3. `PER_STREAM_WATERFILL_BYTES`: red to green; every rung records exact stream home and coded bytes.
4. `ABSOLUTE_PER_STRATUM_TOLERANCE_FEASIBILITY`: remains red for the exhaustive safe-zero subset formulation.
5. `PRE_UINT8_MEMBER_STATE`: remains red structurally.
6. `N600_MEMBER_SOLVE_COVERAGE`: partial green at the required n64 minimum; n600 remains owed.
7. `POSE_STREAM_IN_MEMBER_PAYLOAD`: green; all 384 n64 Pose6 coordinates are exact.

The inherited PRIMARY launch register remains unchanged and blocked. Nothing here authorizes PRIMARY
execution, contest replay, promotion, or pointer movement.

# Bounded re-derivation argv

```bash
.venv/bin/python tools/run_direct_description_entropy_priced_member.py \
  --config .omx/research/ddm_entropy_priced_member_n64_603_613_20260722T044916Z.config.json \
  --output-dir .omx/research/ddm_entropy_priced_member_n64_603_613_REDERIVE_artifacts \
  --execution-allowed false
```

The output path must be fresh because receipts are immutable. The observed n64 run completed within
the delegated ten-minute bound.

# Stores consulted

- `CLAUDE.md`; `AGENTS.md`; `docs/operating_manual_craft_handoff.md`.
- `.omx/research/direct_description_minimizer_PRIMARY_SPEC_20260721T214800Z.md` and Task #603 target,
  membership, receiver, v2 priced-solve, and carrier blocker artifacts.
- Historical coder implementations and receipts with no recency floor, including arithmetic/rank-
  Huffman/colex, AQc1, temporal delta/Rice-Golomb, Brotli, and LZMA surfaces.
- `.omx/state/lane_registry.json`, `.omx/state/subagent_progress.jsonl`, and both delegated inboxes.

0.1910828242 [contest-CPU] — unchanged.
