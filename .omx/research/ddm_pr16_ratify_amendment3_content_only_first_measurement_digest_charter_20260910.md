# ddm_pr16 — second-family ratification of freeze amendment[3] (ffi5: content-only first-measurement runtime digest, repo-root ledger registration, authorize import path) and of the run3→run4 chain (charter, MAIN 2026-09-10)

You are the pr-family reviewer (pr8→pr15 lineage; second family; read-only). Since pr15 (`.omx/research/ddm_pr15_ratify_fire_tool_fix_amendment1_and_first_measurement_findings_20260910.md`,
sha 39efb49c2e0f34b9…) the chain ran twice more:
- **run3** (call fc-01M26G7JYMY39TVN1ET3JJV2YD): passed every local guard, reserved its nonce, dispatched, and the T4 worker
  failed in 5.7 s: `inflate runtime tree hash mismatch expected e3d23719… actual baeb53af…`. MEASURED (retained
  `/Volumes/VertigoDataTier/pact/ddm_rlc5_first_measurement_run3/provenance.json`): the first-measurement worker extracts
  the runtime under a Modal volume root keyed by the authorization sha; the local predictor rewrites to the normal upload
  root; `runtime_content_tree_sha256` (e1e6d125…) and `runtime_files_sha256` (fb1f6295…) are IDENTICAL on both sides;
  only the root-path-coupled tree digest differs (r9m genus). Also: the local half registered the 'dispatched' row into the
  SNAPSHOT's ledger (dispatch cwd = snapshot), so `candidate_seal`'s one-call registration check refused after a real spawn.
  MAIN registered the call in the canonical ledger (dispatched + failed) and closed the claim.
- **ffi5** (memo `.omx/research/ddm_ffi5_first_measurement_consumer_fixes_amendment3_20260910.md`, sha 2482e7eeee94b832…; landed
  0919b62e) implemented: (1) content-only runtime digest validation for first-measurement runs through dispatcher → worker →
  evaluator, path-coupled tree hashes RETAINED for custody (relocation passes; one-byte change refuses; rlc5's real worker
  provenance passes); (2) repo-root ledger registration with the matching lock; (3) `sys.path.insert(1, REPO)` in the
  authorize tool with a foreign-cwd regression. Freeze amendment[3] appended (receipt sha 9557af75c031b06e…; commit 974dda54c).
- **run4** (call fc-01M26JVVPJX57J7YTYENQZXW6A, intent v4 digest ad149549…, authorization v7): first CLEAN dispatch —
  canonical ledger row with first-measurement fields, manifest carrying the expected content digest e1e6d125…; harvest pending
  at your start; MAIN completes and packets only if the exact row qualifies.

## What you adjudicate (RATIFY / AMEND with literal text / REFUSE, each with pr12 sentence → code → test)
1. Is content-only comparison for first-measurement runs a LOOSENING relative to pr12's runtime-identity requirement? The
   normal seal path compares the path-coupled tree hash (which matches because roots coincide). Show that the content-only
   digest is at least as strong on every byte that executes, and rule whether the retained tree digest must ALSO be recorded
   in the completed `candidate_seal.v3` and the t4_direct leg.
2. Repo-root ledger registration: does the lock/ordering match the normal path exactly (no double rows, no snapshot writes)?
3. Authorize import path: ratify pr15's AMEND as implemented.
4. The amendment mechanism itself: three amendments in one day, each a consumer re-pin under the same definition. Rule whether
   the freeze schema needs a "consumer-fix" amendment class distinct from a "definition" amendment (pr14), so ratification
   can be batched without weakening the latest-row check.
5. Run3's consumed nonce/job and run4's fresh pairing: confirm custody (ledger rows, claims, retained receipts) is complete and
   immutable; name anything missing.

## Method
Read pr12 (sha 50d00e3956dc7ae5…), pr13, pr14, pr15, ffi5's memo + tests, the freeze (amendments[0..3]),
`experiments/contest_auth_eval.py` + `experiments/modal_auth_eval.py` (diff 0919b62e^..0919b62e), `tools/fire_modal_auth_eval.py`,
`tools/authorize_candidate_first_measurement.py`, `src/tac/candidate_seal.py`; run the suites on the host. Read-only on code and
every `/Volumes/...` path; no Modal, no fires. Memo `.omx/research/ddm_pr16_ratify_amendment3_content_only_first_measurement_digest_20260910.md`;
serializer commit LAST (`REVIEW_GATE_OVERRIDE=1` ok for .md; add `# FORMALIZATION_PENDING:<rationale>` as a trailing comment — the hook
requires it for review memos); rc 17/19 is NOT a stop. Checkpoint as `ddm_pr16`.

## OPTIMAL FORM
- Reference form: pr12's contract + pr13–pr15 rulings as normative text; the landed code at 0919b62e as the object; the pr13
  clause-table form. No delta.
- Provenance pins (sha256 prefixes): pr15 memo 39efb49c2e0f34b9…; ffi5 memo 2482e7eeee94b832…; freeze receipt 9557af75c031b06e…; ffi5 landing 0919b62e; freeze
  append 974dda54c; run3 provenance path above (record sha); run4 FIRE_MANIFEST
  `/Volumes/VertigoDataTier/pact/ddm_rlc5_first_measurement_run4/FIRE_MANIFEST.json` (record sha); pointer move 43 commit
  48109233e / archive 7beb6a5fc7c2bf477d04a107ab0cf4113d3bd74b2c5a6b94ff77f7ff61971c1e.

## Prior negatives accounted (operator 2026-08-15)
- r9m (env-coupled digest): item 1 is the cure applied; verify it is content-only on BOTH sides, not a relaxed check on one.
- pr10 (rule tuned after data): consumer fixes only; if you find a RULE changed, that is a REFUSE.
- dwc1 (gate with no door): eight pass-path defects were found only by real runs; say which of your checks are runnable
  without a dispatch and which are not.

Final message: five verdicts (one line each), any AMEND text, custody gaps, serializer rc, and the frontier line quoted from
`.omx/state/canonical_frontier_pointer.json` at the time you finish (move 43 or 44).

<!-- # FORMALIZATION_PENDING: review charter; no measured row -->
