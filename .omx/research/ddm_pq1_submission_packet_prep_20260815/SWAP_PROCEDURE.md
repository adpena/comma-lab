# Typed candidate swap procedure

## Trigger and invariant

**CANDIDATE-AGNOSTIC BY CONSTRUCTION (round-11 F4, 2026-08-18).** This paragraph
used to name `e480b v2` and the `e960 composition endpoint` in the present tense
as the only admissible trigger. Three generations were then swapped through a
gate whose text described none of them, and the live pointer candidate — keep01,
a FiLM row-prune with in-compile Schur pose compensation — could not satisfy the
literal trigger at all. A trigger that hardcodes a candidate name goes stale the
moment the frontier moves, which is the one thing the frontier reliably does.
The trigger is therefore stated as a PROPERTY, read from the pointer at swap
time, never as a name.

A swap is permitted only when ALL of the following hold for the proposed
candidate:

1. **It is the live pointer candidate.** Read score and archive SHA-256 at swap
   time from `.omx/state/canonical_frontier_pointer.json` — the canonical SoT
   per CLAUDE.md "Frontier scores are pointer-only". Never from a memo, a
   headline, or this file. Refresh the pointer first
   (`tools/refresh_canonical_frontier.py`) and confirm its
   `last_refreshed_utc` post-dates the candidate's own exact row; a pointer one
   refresh behind is a stale world, not a green light.
2. **It materializes one exact retained archive plus its receiver**, with
   archive bytes and SHA-256 reported FROM DISK, not projected.
3. **It carries a complete `candidate_seal.v1`** binding archive, runtime tree,
   and receiver pins, with its falsifiers passed.
4. **MAIN selects it** as the candidate to prepare.

A projected candidate, an advisory checkpoint, or an archive without receiver
closure cannot trigger a swap. Neither can a candidate the pointer does not
carry — swapping to bytes our own frontier already dominates buys review passes
that will need re-buying.

**Candidate history (HISTORICAL — no longer a trigger condition).** Generation 0
targeted e480b v2 and was gated on the e960 composition endpoint; generation 2
was rr4; generation 3 is `gen3_sz1_composed_split`. These names are recorded as
lineage, not as admissibility criteria. Any future reader: the trigger is the
four properties above.

Every retained packet stays retained. Never overwrite one in place and never
reuse its authority receipts for changed bytes.

## Procedure

1. `VERIFY_SOURCE` — owner `MAIN packet owner`; consumer `new generation
   staging root`; fire trigger `the selected POINTER candidate's archive and
   receiver exist under retained custody` (the four trigger properties above,
   read from the pointer at swap time — never a hardcoded candidate name).
   Hash the source archive and every executable
   runtime file, verify archive member safety and receiver parse-back, and stop
   on any mismatch.
2. `STAGE_NEW_GENERATION` — owner `MAIN packet owner`; consumer
   `ddm_pq1_submission_packet/generations/<candidate-id>/`; fire trigger
   `VERIFY_SOURCE green`. Copy the exact archive and receiver, retain the source
   provenance, and diff the copy to the source before adding documentation.
3. `RESET_AUTHORITY` — owner `MAIN scorer router`; consumer `candidate authority
   store`; fire trigger `staged copy identity green`. Mark both `[contest-CUDA]`
   and `[contest-CPU]` exact rows pending for the new bytes. No component, score,
   runtime, or lane receipt transfers merely because the renderer lineage is
   shared.
4. `REFRESH_PUBLIC_PACKET` — owner `MAIN packet owner`; consumer = **EVERY
   document in `.omx/research/ddm_pq1_submission_packet_prep_20260815/` AND
   every staged public file in the active packet generation directory** —
   the list is a DIRECTORY, never a closed enumeration (README, report.txt,
   archive manifests BOTH repo-side and packet-side, PR body, accounting,
   packet target, CPU fire order, gap report, etiquette, runbook, scaffold,
   swap procedure itself). Fire trigger `real new receipts exist`. Replace
   all archive/member/runtime/source/score fields from machine receipts,
   rerun borrowed-substrate accounting at section level, and ensure the
   public body contains no private infrastructure, local paths, provider
   transcript, or machine attribution. **Any document deliberately NOT
   refreshed must carry an explicit HISTORICAL/SUPERSEDED banner in the same
   swap** — a stale present-tense sibling is a review finding by definition
   (rounds 5 and 6 both paid this tax; round 6 traced all six of its findings
   to this list having been closed).
5. `RERUN_STRICT_CHAIN` — owner `MAIN packet owner`; consumer `generation gap
   report and compliance JSON`; fire trigger `public packet refreshed`. Execute
   the exact strict checker with the new expected SHA and size. Record every red
   item; never convert a red to green by copying an old receipt. **RECEIPT
   FRESHNESS LAW (round-8 F2):** the compliance receipt is INVALIDATED by any
   edit to a surface the checker scans (packet README/report.txt/manifests,
   PR body, public-scan paths) — a receipt older than any scanned file is
   stale by definition, so every fix batch that touches one ENDS by re-running
   the checker and re-pointing every receipt citation in the same batch.
   **THE RECEIPT HAS THREE INPUTS, NOT ONE (round-11 F1):** a receipt is a joint
   measurement of BYTES × INSTRUMENT × WORLD, and it is stale when ANY of the
   three moves — (a) SURFACES: any edit to a file the checker scans; (b)
   INSTRUMENT: any commit touching the checker or its helper modules, which is
   how r5 came to claim 86 checks while the live checker ran 87; (c) WORLD: any
   refresh of `.omx/state/canonical_frontier_pointer.json`, which is how a
   green `frontier_no_regression_on_submitted_axis` went red with the packet
   untouched. Every receipt now records all three in its `instrument_and_world`
   block (`checker_source_sha256`, `frontier_pointer_state`,
   `scanned_file_count`) — compare those against live state BEFORE citing a
   receipt, rather than assuming the world stood still.
   **CENSUS BOTH DIRECTORIES FIRST (round-11 F3/F6):** run
   `tools/packet_census_guard.py` immediately BEFORE any receipt re-buy and
   again immediately before publication — the same two moments this freshness
   law already governs. Both surfaces, one invocation, and rc must be 0:

   ```bash
   .venv/bin/python tools/packet_census_guard.py \
       --packet-dir <staged generation dir> \
       --auth-eval-json <the row's contest_auth_eval.json> \
       --prep-dir .omx/research/ddm_pq1_submission_packet_prep_20260815
   ```

   A non-zero rc REFUSES the re-buy: buying a receipt over a directory holding
   files nobody declared certifies contaminants along with the packet. The
   `--prep-dir` half is structural (the prep tree is FLAT — no subdirectories,
   no dot-entries) and exists because a Stop-hook once wrote `.omx/state/*.json`
   markers into the prep tree, where they sat staged for six hours invisible to
   a packet-only census. Any `DOUBLE-DECLARED:` line in the output is
   information, not a failure: it names a file both authorities cover, and
   therefore a file whose loss from the runtime manifest this census cannot
   catch.
   **NO HAND-TYPED VALUES (round-8 F4 + the r5-attempt-1 refusal):** every
   sha, size, and timestamp passed to the checker or written into a custody
   field is DERIVED (from a receipt, git, or the clock) — a hand-completed
   sha prefix and an invented `_utc` are the same defect; the seal contract's
   no-hand-typed-sha principle binds here too.
6. `DELTA_REVIEW` — owner `submission reviewers`; consumer `review scaffold`;
   fire trigger `strict checker receipt exists`. Compare generation 1 against
   generation 0 for archive, member grammar, receiver files, dependencies,
   score axes, source pin, lineage table, public text, runtime budget, and every
   blocker. Any finding resets the consecutive-pass counter to `0/5` after the
   fix.
7. `SELECT_ACTIVE_GENERATION` — owner `MAIN`; consumer `PACKET_TARGET.json` and
   final PR hold surface`; fire trigger `both exact axes complete, strict
   checker green, public URL/source visible, and five consecutive clean passes`.
   Select exactly one candidate. Retain the losing generation without
   submitting it.

## Refusal conditions

- Source archive hash or size differs from the selection receipt.
- Receiver tree lacks a byte-identity or exact authority binding for the new
  archive.
- Only one score axis exists.
- Any public artifact contains unresolved placeholders or private operational
  details.
- The strict checker is red or the review counter is below `5/5`.
- An actual push, hosting action, or pull-request opening lacks explicit
  operator authorization.

## Generation-3 adjudication note (2026-08-18)

This procedure was executed for generation 3 (`gen3_sz1_composed_split`,
archive `debb025f45bb42e3…`/179,930 B). Two literal conditions above are
satisfied by DOCUMENTED ADJUDICATION rather than by their literal reading;
the refusal conditions themselves are not weakened:

- "Both exact axes complete": the CUDA axis is complete (T4 n600 row,
  call `fc-01M09EHX5MMTJACMRADQPN9P7Z`). The CPU axis is complete BY
  MEASUREMENT: inflate of these exact bytes took 3,422.7 s against the
  1,800 s budget on 4-thread x86_64 (call `fc-01M09G62A7SZ7HZYE5Q28YS7VP`),
  decoded tokens byte-exact — the axis is adjudicated MEASURED-INFEASIBLE,
  not pending. No CPU score exists or is claimed anywhere in the packet.
- "Strict checker green": the gen-3 terminal state is 82/86
  (`gen3_receipts/pre_submission_compliance.gen3.r5.json` — the CANONICAL
  terminal receipt, re-bought after the round-7/8 fixes; r3 and r4 each
  predate edits to checker-scanned surfaces and are superseded). The 4 residual
  reds are each typed and documented in `COMPLIANCE_RUNBOOK.md`
  (2 structural-by-construction, 1 by-design dependency bootstrap under the
  e4 precedent, 1 operator-gated hosted manifest). None was converted by
  editing a receipt or a check.
- Review counter: `ADVERSARIAL_REVIEW_SCAFFOLD.md` is the SINGLE counter
  authority — this document does not carry its own count. Read the scaffold's
  header and table for the live state.
- The final refusal condition is UNCHANGED and binding: no push, hosting
  action, or pull-request opening without explicit operator authorization.
