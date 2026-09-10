# ddm_ffi6 — implement pr16's literal completion-custody requirements and the typed consumer-fix amendment row (charter, MAIN 2026-09-10)

pr16 (`.omx/research/ddm_pr16_ratify_amendment3_content_only_first_measurement_digest_20260910.md`, sha cb678491cc6859f7…, landed)
ratified amendment[3]'s content-only comparison as byte-strong but ordered, BEFORE any `candidate_seal.v3` completion:
1. **Completion custody (AMEND, literal):** `tools/make_candidate_seal.py --complete-first-fire-intent` /
   `tac.candidate_seal.complete_first_fire_intent` must (a) JOIN the independently recomputed content digest across the
   FIRST_MEASUREMENT_CONTEXT, the local request, the exact argv, the harvested result, and the worker provenance (all must
   agree, else refuse); (b) PROJECT the local dependency manifest onto the worker's RECORDED retained root (from provenance
   `runtime_root`) and VALIDATE the actual worker tree hash (so the path-coupled digest is checked against the real root,
   not the normal upload root); (c) write IDENTICAL runtime-custody objects (content digest, files digest, retained-root tree
   digest, root path, file count, definitions named) into `candidate_seal.v3` and into its `t4_direct` leg — without
   changing the legacy `t4_direct` validators' requirements (add fields; break nothing).
2. **Typed consumer-fix rows (AMEND-PROSPECTIVE):** introduce `prefire_contract_consumer_fix.v1` rows in the freeze, chained
   to the unchanged definition amendment (parent = pr14's row) and the exact latest implementation commit/manifest; the
   validator's latest-row refusal, definition-parent identity, ancestry, and manifest identity must be preserved with typed
   tests; migrate amendments[1..3] semantics by ADDING the typed rows (never rewriting the existing rows).
3. Run3's reconciliation receipt is MAIN's (`.omx/research/ddm_rlc5_20260910/RUN3_RECONCILIATION.json`, committed) — read it;
   your completion path must refuse to complete against run3's receipt (non-replayable) and accept run4's when it lands.

## Deliverable
Code + tests in ONE serializer commit (two visible review passes per .py; ruff; suites `test_candidate_prefire_intent.py`,
`test_candidate_seal.py`, `test_decode_wall_clock_t4_direct.py`, `test_decode_wall_clock.py` green on the host); a
synthetic-receipt completion test AND a test that replays run3's RETAINED provenance (read-only) through the projection and
refuses on the tree hash while passing content — plus, if run4's harvest `/Volumes/VertigoDataTier/pact/ddm_rlc5_first_measurement_run4/MODAL_REMOTE_RESULT.json`
exists when you reach that step, a DRY validation of the joined digests on it (no completion, no seal written: MAIN completes).
Regenerate the implementation manifest at your commit (draft `.omx/research/ddm_ffi6_20260910/PREFIRE_IMPLEMENTATION_MANIFEST.json`
+ typed freeze-row DRAFT); memo `.omx/research/ddm_ffi6_completion_custody_and_consumer_fix_rows_20260910.md` with
requirement → code → test rows. Commit LAST, once; rc 17/19 is NOT a stop (SSDs may be under the fallback reserve; leave the files
in the tree). Checkpoint as `ddm_ffi6`.

## Boundaries
No Modal, no fires, no authorization, no completion (MAIN's), no timing windows, no n600 runs; never edit `upstream/`, the PR
tree, sealed trees, the frozen receipt (MAIN appends), or any `/Volumes/...` path; do not touch gdc2's or sr4's directories.
Do not change any contract RULE beyond pr16's literal requirements.

## OPTIMAL FORM
- Reference form: pr16's literal amendment requirements; ffi3/ffi4/ffi5 landings as the code-and-tests-in-one-commit pattern;
  the normal seal path's custody objects as the shape to mirror.
- Provenance pins (sha256 prefixes): pr16 memo cb678491cc6859f7…; freeze receipt 9557af75c031b06e… (amendments[0..3]); ffi5 landing 0919b62e; run3
  provenance `/Volumes/VertigoDataTier/pact/ddm_rlc5_first_measurement_run3/provenance.json` (record sha); run4 FIRE_MANIFEST
  (record sha); pointer move 43 commit 48109233e / archive 7beb6a5fc7c2bf477d04a107ab0cf4113d3bd74b2c5a6b94ff77f7ff61971c1e.

## Prior negatives accounted (operator 2026-08-15)
- Eight pass-path defects were found only by real runs (dwc1 genus): your run3-provenance replay test is the real control
  for the projection; a fixture-only test does not close it.
- r9m: the joined digest must be content-only on every side; the retained-root projection is custody, not identity.
- pr10: consumer changes only; pr17 ratifies post hoc.

Final message: requirement → code → test rows, test counts, the run3 replay result, the run4 dry validation (if reached), the
serializer rc, and the frontier line quoted from `.omx/state/canonical_frontier_pointer.json`.

<!-- # FORMALIZATION_PENDING: implementation charter; consumer fixes, no measured row -->
