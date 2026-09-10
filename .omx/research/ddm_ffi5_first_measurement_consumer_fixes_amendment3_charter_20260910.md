# ddm_ffi5 — land the three first-measurement CONSUMER fixes as freeze amendment[3] (charter, MAIN 2026-09-10)

The pre-fire contract's first real dispatch (rlc5 run3, Modal call fc-01M26G7JYMY39TVN1ET3JJV2YD) reached the T4 worker and
failed in 5.7 s: `RuntimeError: inflate runtime tree hash mismatch: expected=e3d2371917920ce3dad002761203af48cfa72d98b07f82f077ef6dc0115b1891
actual=baeb53afc8d1bb0c43bb2ce91fe1e09e845809b76fc3047bc8ed3397254155af` (retained: `/Volumes/VertigoDataTier/pact/ddm_rlc5_first_measurement_run3/`,
read-only). pr15 (`.omx/research/ddm_pr15_ratify_fire_tool_fix_amendment1_and_first_measurement_findings_20260910.md`, sha 39efb49c2e0f34b9…)
ratified amendments[1–2] and ordered amendment[3] for the authorize import path. Three consumer defects remain; land them
TOGETHER as amendment[3], verbatim from the findings below, nothing else.

## The three fixes (all consumer code; the contract rules are unchanged)
1. **Expected runtime digest definition (pr15 finding 8 / r9m genus).** `tools/fire_modal_auth_eval.py::_first_measurement_main`
   passes the INTENT's `candidate.runtime.sha256` (`tac.candidate_seal.measure_runtime_digest`) as
   `expected_runtime_tree_sha256`; the worker (`experiments/contest_auth_eval.py::_validate_expected_runtime_tree`) checks the
   dependency-manifest definition that the NORMAL seal path supplies via `tac.decode_wall_clock.measure_t4_runtime_digest`
   (sj1's fire: seal digest 68fae56a… vs expected_runtime_tree a726739a…, and it MATCHED on the worker). Make the
   first-measurement path compute and pass the worker's definition exactly as the normal path does, and RECORD BOTH digests
   (name each definition) in the context and manifest. Prove with a test that the argv's expected digest equals
   `measure_t4_runtime_digest(runtime)` and that the intent's digest is still validated separately.
2. **Ledger written into the source snapshot (pr15 finding 7; the exact hazard `verify_dispatch_paths` documents).** The
   first-measurement dispatch ran from the snapshot cwd; `register_dispatched_call_id` wrote the 'dispatched' row into
   `.omx/tmp/modal_fire_snapshots/<label>/.omx/state/modal_call_id_ledger.jsonl` (retained; read-only), so
   `candidate_seal`'s one-call T4 registration check refused after a REAL spawn. Fix: the local half must register into
   the REPO ledger (pass the repo root explicitly, as the normal path does) — never the snapshot; add a test with a foreign cwd.
3. **Authorize tool import path (pr15 AMEND, verbatim):** `tools/authorize_candidate_first_measurement.py` adds
   `sys.path.insert(1, str(REPO))` (the repo root) so `tac.decode_wall_clock` can import `experiments.contest_auth_eval`
   from a bare invocation; add the foreign-cwd regression pr15 names.

## Deliverable
Code + tests in ONE serializer commit (two visible review passes per .py; ruff clean; the existing suites
`test_candidate_prefire_intent.py`, `test_candidate_seal.py`, `test_decode_wall_clock_t4_direct.py` green on the host);
regenerate the implementation manifest over `PREFIRE_IMPLEMENTATION_PATHS` at your commit (draft
`.omx/research/ddm_ffi5_20260910/PREFIRE_IMPLEMENTATION_MANIFEST_AMENDMENT3.json` + FREEZE_APPEND_DRAFT with the commit
placeholder; MAIN appends amendment[3] after landing); memo
`.omx/research/ddm_ffi5_first_measurement_consumer_fixes_amendment3_20260910.md` with finding → code → test rows and
the two runtime digests of rlc5's tree under each definition. Commit LAST, once; rc 17/19 is NOT a stop (both SSDs are under
the serializer's 40 GiB fallback reserve — if the fallback bundle is refused, leave the files in the tree and say so; MAIN
lands from the tree). Checkpoint as `ddm_ffi5`.

## Boundaries
No Modal, no fires, no authorization, no timing windows, no n600 runs; never edit `upstream/`, the PR tree, sealed trees,
`src/tac/decode_wall_clock.py`'s legacy `t4_direct` requirements, the frozen receipt (MAIN appends), or any `/Volumes/...`
path; do not touch gdc2's live directory. Do not change any contract RULE (schemas, gates, digests' scoped definition).

## OPTIMAL FORM
- Reference form: the NORMAL seal fire path in `tools/fire_modal_auth_eval.py` (digest computation, repo-root ledger
  registration) as the pattern the first-measurement path must match; ffi3/ffi4 landings as the code-and-tests-in-one-commit form.
- Provenance pins (sha256 prefixes): pr15 memo 39efb49c2e0f34b9…; freeze receipt (record its sha; amendments[0..2] present); run3
  spawn record `/Volumes/VertigoDataTier/pact/ddm_rlc5_first_measurement_run3/modal_auth_eval_spawn.json` (record sha);
  snapshot ledger row path above (record sha); fire-tool commits 808da43c / 7a837a9de; pointer move 43 commit 48109233e /
  archive 7beb6a5fc7c2bf477d04a107ab0cf4113d3bd74b2c5a6b94ff77f7ff61971c1e.

## Prior negatives accounted (operator 2026-08-15)
- r9m (two validators disagree ⇒ env-coupled digest): fix 1 is that genus — pass the worker's definition, record both.
- verify_dispatch_paths' own docstring (ps2 rc 5): fix 2 is the ledger half of that hazard, now measured for real.
- dwc1 (gate with no door): each of the eight pass-path defects so far was a door with no producer; your tests must exercise
  the REAL argv/paths (foreign cwd, real digest functions), not fixtures that assume the pass.
- pr10: consumer fixes only; the second family's rules are untouched; pr16 ratifies post hoc.

Final message: three finding→code→test rows, both digests of rlc5's tree, test counts, the serializer rc, and the frontier line
`composition S 0.1372848557085275 @ 180,466 B [contest-CUDA T4 n600] (move 43)` unchanged.

<!-- # FORMALIZATION_PENDING: implementation charter; consumer fixes, no measured row -->
