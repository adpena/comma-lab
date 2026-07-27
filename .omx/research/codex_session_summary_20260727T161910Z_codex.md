# Codex TIER-0 capstone session anchor — 2026-07-27 16:19 UTC

## Pointer-delta honesty

- Canonical competitive target: `0.172`, official upstream leaderboard PR130.
- Pointer moved: **false**.
- Candidate archive emitted by this session: **false**.
- The conditional `0.1362947950400364` score at `133941` bytes is a
  rate/distortion composition envelope only. It is not a same-object archive
  score.
- The exact strict byte ceiling at the measured low-distortion anchor is
  `187563` bytes; `187564` bytes misses strict `< 0.172`.

## Completed and never to be reopened as unfinished

Canonical task lifecycle is committed in
`.omx/state/canonical_task_status.jsonl` at `e1b948a389`.

- Fresh-process DSL custody: `9ee4044853`.
- Governed typed production resume: `2aa4d264f7`.
- Physical GREEN release evidence hardening: `3642caf83e`.
- Complete cold-root state ordering: `318c5bf698`.
- Terminal-safe live G121 harvest handoff: `5039432cb1`.
- Clean public stdlib-only XIP2 receiver path: `9a63e5bc23`.
- Atomic G119-row to G110 release materializer: `2efce90722`.
- G110 release-path adversarial verdict: `3bf067fb51`.
- Governed G120 two-process batch-resume gate: `acc51db6c2`.

The older G111/G117/G119/G120/G121/G110 implementation tasks listed as
`completed` by `canonical_task_status.py` are also terminal. Future sessions
must query the canonical ledger and must not rediscover or recreate them.

## Structural defects closed in this capstone pass

The composition failures were not marginal score tuning:

1. typed DSL compilation depended on ambient in-process registry state;
2. production resume was not a complete typed launch operand;
3. mutable marker files could forge a GREEN dry-start;
4. the cold-root checkpoint ran before complete controller-state
   initialization;
5. G121 could publish an unsafe live-snapshot notion of exhaustive coverage;
6. G120 wrote immutable batch artifacts but recomputed them after restart;
7. the public receiver depended on undeclared Brotli for the fresh path and
   mishandled exact output reuse;
8. compiled archive bytes were discarded without an atomic public release
   transducer.

Each defect now has code, regression tests, review-policy evidence, and a
terminal canonical task row.

## Active physical gate

Only `pact-g111-current-typed-clean-dry-start` is in progress.

- Run directory:
  `/Volumes/VertigoDataTier/pact/g111_batch16_v9_semantic_base_dry_start_v6_20260727`
- Exact launch DSL compile:
  `583d746ce65f40a65772ed346ec0cf6da48f2b5c664562a841514627a964f827`
- Pass budget: `2400` seconds per process.
- The run is full `n=600`, batch `16`, fresh lineage, governed, SSD-backed,
  and memory-admitted.
- G120 gate receipt in that directory is already GREEN:
  `178907cd677d2d6a2e2b6a3a394e110778a2d3c524365e5db38db786882ffc65`.
- That G120 receipt proves two distinct PIDs, completed-batch reopen, both
  storage preflights, and `scorer_calls=0`; it is not a score receipt.

Do not mark the typed clean dry-start complete until v6 produces its own
physical cold-boot plus exact crash-resume proof.

## Exact next execution edge

If v6 is GREEN:

1. append the terminal row for
   `pact-g111-current-typed-clean-dry-start`;
2. recompile the exact current typed G111 manifest and prove its launch
   blocker set is empty;
3. change `pact-g111-first-real-n600-capstone-run` to `in_progress` only when
   the governed resumable producer has actually spawned in a fresh permanent
   SSD directory;
4. start the G121 live monitor against the exact producer manifest SHA;
5. harvest immutable preserved stages and run G119 pose refit/global coder
   arbitration for each admitted stage;
6. explicitly select a nondominated G119 row and run the G110 atomic
   materializer with runtime tree
   `21e8288e7ea9bf46527e8c68db7b08886f396edae6b3d510b2ba43127a9ec686`;
7. prove clean-root public double decode and bit-identical output;
8. run exact upstream CPU and CUDA evaluation on the same archive bytes;
9. move the pointer only for a score strictly below the then-live effective
   frontier.

If v6 fails, preserve the directory and receipt, register one narrowly named
blocker task, land the smallest real fix, and use a new directory for the next
physical attempt.

## Closeness assessment

The receiver/compiler/release architecture is now one connected chain rather
than disconnected components. The remaining uncertainty is empirical:
whether fresh G111 stages preserve the measured low-distortion point while
the selected semantic and conditional-pose operands remain below the live
byte ceiling.

The full typed projection is about `7.2` days for `3000` epochs. Early
immutable stage rows can be harvested before terminal completion, so the
first exact same-object archive decision does not need to wait for the final
epoch. No schedule estimate is score authority.

## Triality

- DSL: exact G111 compile hash, typed resume operands, G120/G121/G119/G110
  archive contracts.
- DAG: v6 gate -> real producer -> live stage harvest -> pose refit -> atomic
  materialization -> double decode -> CPU/CUDA evaluation -> pointer.
- Equations:
  `S = 100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37545489`,
  with strict target `S < effective_frontier`.

## STORES CONSULTED

- `CLAUDE.md`
- `AGENTS.md`
- `.omx/state/canonical_frontier_pointer.json`
- `.omx/state/canonical_task_status.jsonl`
- `.omx/state/lane_registry.json`
- `.omx/state/subagent_progress.jsonl`
- G111 v6 launch manifests and physical G120 receipt
- G110 macro findings memo and exact materializer runtime identity
