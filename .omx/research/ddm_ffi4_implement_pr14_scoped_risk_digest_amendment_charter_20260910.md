# ddm_ffi4 — implement pr14's scoped timing-risk receiver digest amendment (charter, MAIN 2026-09-10)

pr14 (second family; memo `.omx/research/ddm_pr14_adjudicate_manifest_in_normalized_receiver_digest_20260910.md`, sha 6e1732baea1912731fef6a53abca0cb94faf2e9440accc304dd9b34e593bade9, landed) AMENDED the frozen
pre-fire contract after its first REAL refusal (rlc4): `MANIFEST.sha256` is excluded ONLY from a NEW versioned
timing-risk receiver digest; every full runtime identity / custody / dependency-manifest check keeps the manifest.
Under the amended digest the RLC1 timed reference and the RLC4 candidate both recompute to
`9f6e71680a13d8598974ee13f78a1a72759a758681e6d86b7b288cc105442890` over 49 byte-identical normalized rows.
The legacy `t4_direct` leg (`tac.decode_wall_clock`, receiver 6726fd77…) is UNTOUCHED.

## Your deliverable — pr14 §"Literal implementation patch contract" (memo lines ~230–434), implemented VERBATIM
1. `src/tac/candidate_seal.py`: exactly the functions pr14 specifies (`_materialize_prefire_receiver_rows`,
   `_prefire_receiver_rows_digest`, `prefire_receiver_rows`, `prefire_risk_receiver_rows`,
   `measure_prefire_risk_receiver_digest`, the versioned schema/field names it names, and the
   `validate_prefire_risk` endpoint comparison moved onto the scoped digest with the version pinned in the receipt).
   No other behavior change; `measure_runtime_digest` and every seal/custody check keep the manifest.
2. Exactly the six tests pr14 names, added to `src/tac/tests/test_candidate_prefire_intent.py`
   (`test_prefire_risk_accepts_only_regenerated_manifest_indirection`,
   `test_prefire_risk_refuses_nonpin_inflate_byte_change`,
   `test_prefire_risk_manifest_exclusion_does_not_bypass_dependency_manifest_validation`,
   `test_prefire_amendment_keeps_legacy_t4_direct_manifest_inclusive`,
   `test_prefire_amendment_allows_pinned_old_evidence_but_refuses_pre_amendment_intent`,
   `test_prefire_amendment_requires_latest_frozen_row_and_live_committed_sources`). The non-pin `inflate.py`
   one-byte falsifier MUST refuse.
3. Real-tree check (read-only): run the new `measure_prefire_risk_receiver_digest` over
   `/Volumes/VertigoDataTier/pact/ddm_rlc1_rule118_cure/candidate_runtime` and
   `/Volumes/VertigoDataTier/pact/ddm_rlc2_cure_on_move42/candidate_runtime`; both MUST equal
   9f6e71680a13d859…; record both with the 49-row listing sha in your memo. A mismatch is a STOP with the differing row.
4. Validation on the host paths: `.venv/bin/python -m pytest src/tac/tests/test_candidate_prefire_intent.py
   src/tac/tests/test_candidate_seal.py src/tac/tests/test_decode_wall_clock_t4_direct.py -q` and
   `.venv/bin/python -m ruff check src/tac/candidate_seal.py src/tac/tests/test_candidate_prefire_intent.py`.
5. Memo `.omx/research/ddm_ffi4_implement_pr14_scoped_risk_digest_amendment_20260910.md`: clause → code → test table
   for every pr14 patch clause, the two real-tree digests, test counts, and the freeze-append row DRAFT (pr14 §"Freeze
   append" schema) with placeholders ONLY for the landed commit sha and the implementation manifest sha, which MAIN fills
   at landing (the freeze is appended AFTER the landing, never rewritten).
6. Two visible review passes per changed .py (`tools/review_tracker.py mark-file … --status reviewed`); serializer
   commit LAST, once, with post-edit shas, tags `[no-triality] [p0-ledger-ok]`, no co-author trailer. A Git-object write
   denial (rc 17) is NOT a stop: leave the verified bundle, report the rc, MAIN lands. Checkpoint as `ddm_ffi4`.

## Boundaries
No Modal, no fires, no timing windows, no n600 runs; never edit `upstream/`, the PR tree, sealed trees, any
`/Volumes/...` content, `src/tac/decode_wall_clock.py`, or the frozen receipt itself (MAIN appends). Do not touch the
live sj1 fire directories (`ddm_sj1_pass6`). Do not widen the exclusion beyond the scoped risk digest.

## OPTIMAL FORM
- Reference form: pr14's literal patch contract, verbatim; the a47543199 landing (ffi3) as the code-and-tests-in-one-commit
  pattern. No delta, no scope reduction.
- Provenance pins: pr14 memo sha 6e1732baea1912731fef6a53abca0cb94faf2e9440accc304dd9b34e593bade9; pr13 memo sha
  3156992449ea4f71…; pr12 memo sha 50d00e3956dc7ae5…; frozen receipt sha 59158b8fce89e12c…; contract commit a47543199;
  rlc4 conflict record `.omx/research/ddm_rlc4_20260910/RECEIVER_MANIFEST_CONFLICT.json`; pointer move 42 d2803c214 /
  archive f111ab4259c757409e791247d33978a714ceb1cd66e50c149e2e65fbf208756f.

## Prior negatives accounted (operator 2026-08-15)
- rlc4's real refusal (PREFIRE_RISK_EVIDENCE_REFUSED) — the instance this cures; the cure is SCOPED, every other check
  stays manifest-inclusive (pr14 dead-end: excluding the manifest from custody is closed).
- pr10 (rule tuned after data) — the amendment is the second family's, landed BEFORE any producer re-runs; your tests
  must include the pre-amendment-intent refusal.
- ffi1/ffi2 STOPs on clauses — pr14's contract is literal; if a clause cannot be implemented as written, STOP with the
  exact sentence, do not improvise.
- r9m (env-coupled digest) — the scoped digest must be content-only and identical across actors (the two real trees).

Final message: commit rc, test counts, the two real-tree digests, the freeze-row draft, every boundary, and the frontier
line `composition S 0.1374765052591843 @ 180,238 B [contest-CUDA T4 n600] (move 42)` unchanged.

<!-- # FORMALIZATION_PENDING: implementation charter; the amended validator definition is a contract rule, not a score law -->
