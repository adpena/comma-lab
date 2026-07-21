# Task #578 G2d contextual predict-base realization — implementation specification

`lane_id=lane_realization_g2d_predict_base_578_20260721` · `BUILD+MEASURE` ·
`[macOS-CPU advisory]` · `score_claim=false` · `promotion_eligible=false` ·
`pointer=0.1910828242 [contest-CPU] UNMOVED` · `MAIN_REVIEW_REQUIRED=true`

## Objective and narrow intervention

Measure the untested contextual realization child without reopening the three
settled context-free negatives.  Decode one continuous 1,200-frame RGB sequence:
the first scorer-plane RGB frame is an exact counted bootstrap, every later base
is the canonical G1 xi warp of the previously decoded RGB plane, and PROJECT
changes only currently violated declared seed cells.  Each changed RGB value is
selected from deterministic L-infinity bands directed toward the settled
class-specific max-margin prototypes; the smallest band attaining the best
positive-margin survival tuple wins.

The receiver remains scorer-free.  The encoder may invoke the frozen native
CPU-Torch scorer to choose a band, but the decoder consumes only the bootstrap,
the existing motion/seed custody, and a strict parse-backed exception stream.
No current-frame source RGB value is consulted after frame zero.

## Owned edits and collision boundary

- Extend `tools/measure_realization_g2_lattice.py` with the contextual runner,
  resumable pair stages, prefix reducers, strict exception codec, sequential
  replay, and CLI switch.
- Extend `tools/tests/test_measure_realization_g2_lattice.py` with deterministic
  warp/projection, codec, prefix, resume-refusal, and admission fixtures.
- Own dated receipt, findings, DAG FEED, and reuse manifest for this lane.
- Do not edit sibling-owned predictor/coder/schema modules.  Import settled G1,
  #557 adaptive-range-coder, receiver, lattice, and equation surfaces as-is.
- Do not edit upstream, live runs, provider surfaces, frontier pointers, or any
  other worktree.

## Measurement contract

1. Verify the exact seed, frozen GT cache, G1 receipt/LawRefs, scorer source and
   weights, receiver/lattice sources, and tool source in the config hash.
2. Storage-preflight `/Volumes/VertigoDataTier`, write atomic immutable pair
   stages and per-chunk checkpoints, and refuse any source/config drift.
3. Run cumulative `n16 -> n64 -> n600`; every pair records exact factor-2
   custody, double-decode identity, declared-cell target margins by class,
   stratum and bucket, PoseNet output/tube debt, exception bytes, and separate
   encoder versus decoder timings.
4. Preserve the single bootstrap byte stream and every exception sidecar.  The
   bootstrap is Brotli-11 over exact frame-zero scorer RGB, explicitly the
   settled #557 classical complete-coder choice, not a free keyframe.  Sparse
   exception ordinals and RGB deltas use the existing #557 adaptive range coder
   and must parse back and re-encode byte-identically.
5. Replay all 1,200 frames sequentially from durable bytes, with zero decoder
   scorer calls, and report wall time as engineering telemetry only.
6. Evaluate `predict_project_realization_admissibility_v1` on the n600 row.  Do
   not coerce the bootstrap or exceptions to zero bytes, do not redefine whole
   description equality, and do not promote declared-write survival into a
   score.  A failed conjunct is a measured contextual-formulation result only.

## Acceptance and stop rules

- Focused tests, Ruff, format check, `py_compile`, JSON parse, and `git diff
  --check` pass; changed Python receives two clean review-tracker passes.
- D1 reports whole-description semantic exactness and declared-write survival
  separately.  Partial survival is decomposed by class, stratum, and achieved
  margin bucket.
- D2 reports measured realized `d_pose`, tube debt, and tube pair count.
- D3 reports bootstrap, exception, seed baseline 78,969 B, and 216,222 B box.
- D4 reports the actual sequential replay time over exactly 1,200 frames.
- D5 is the unmodified canonical certificate.  If it remains false, the memo
  names every failed predicate and the narrow measured formulation scope.

## STORES CONSULTED

Delegated authority SHA `8bcc0bbc9534197d14b6b514fc447d3b334ce4f8a2a1eef6bb100eb9c4d8c1fc`;
`CLAUDE.md`; `AGENTS.md`; craft handoff; v7.5 section 8; Task #597
PREDICT-to-PROJECT interfaces; G1/G3 receipt; predictor round-1/round-2 code and
findings; G2/G2b/G2c receipts, code, tests, equations, DAGs and reuse manifests;
r1b6 receiver-bound negative; frozen seed/cache/scorer custody; lane/progress
state; and both live inboxes through `2026-07-19T19:48:01Z`.
