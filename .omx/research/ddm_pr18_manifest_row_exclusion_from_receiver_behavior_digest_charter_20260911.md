# ddm_pr18 — amend the receiver behavior digest to exclude the derived MANIFEST.sha256 listing, through the contract's own amendment path (charter, MAIN 2026-09-11; operator full-authority GO)

## The measured defect (two candidates blocked today)
`tac.decode_wall_clock.measure_receiver_digest` hashes every shipped file raw except `archive.zip`, normalizing only the literal
archive-pin assignments inside `inflate.py`. `MANIFEST.sha256` is a DERIVED listing: it records inflate.py's RAW hash, which changes
whenever the archive pins change. So the manifest row of the digest differs for every candidate whose archive differs, and
`inherit_decode_wall_clock` refuses with "inherited receiver code differs" even when every behavioral row is identical.
MEASURED today: pc3's candidate (`/Volumes/VertigoDataTier/pact/ddm_pc3_pose_carrier_curve/seal_inputs/RECEIVER_DIGEST_ROW_DIFF.json`,
sha 0160ddfff979c72d…): of 50 rows exactly one differs — MANIFEST.sha256 (4,648 B vs the pointer's 4,570 B); inflate.py identical
after pin normalization. ntb2's candidate (`.omx/research/ddm_ntb2_20260911/`) has the same shape. Earlier moves passed inherit only
by carrying a STALE manifest (pr9 condition 1, a defect); honest regeneration since rlc4 closed the path. pr14 named this amendment
(memory `derived_hash_listing_inside_an_identity_digest_couples_identity_to_raw_pins_scope_the_exclusion_20260910`): scope the
exclusion to the BEHAVIOR digest only; the raw manifest stays in every custody check. The ffi4 scoped risk digest
(`src/tac/candidate_seal.py: measure_prefire_risk_receiver_digest` / `prefire_risk_receiver_rows`, 49 rows) already does this for
the intent path — the normal seal/inherit path must agree with it.

## Deliverable (one serializer commit for code + tests; one for the frozen-contract amendment row)
1. `src/tac/decode_wall_clock.py`: a VERSIONED receiver behavior digest that excludes `MANIFEST.sha256` (and any other derived
   listing you can name with a measurement — do not guess) from the identity rows, plus an INDEPENDENT manifest validation that the
   custody path runs on every tree it touches: every listed file exists, every listed hash matches the file's raw bytes, inflate.py's
   listed hash matches its raw bytes, no file outside the listing except archive.zip. Stored receipts carry the old digest: the
   inherit/complete validators must accept a leg whose stored digest was computed under the old definition when the trees agree under
   the new one, and record which definition each comparison used (typed field, not a comment). Do not change the completed
   `t4_direct` leg's other requirements. Prove on the host: move 44's tree (`/Volumes/VertigoDataTier/pact/ddm_rlc5_cure_on_move43/
   candidate_runtime`) vs pc3's tree → EQUAL under the new digest, DIFFERENT under the old (the measured defect); move 44 vs the
   move-43 tree (`/Volumes/VertigoDataTier/pact/ddm_sj1_pass6/candidate/candidate_runtime`) → DIFFERENT under both (a real receiver
   change must still refuse); the new digest over move 44's tree equals the ffi4 49-row risk digest's row set (state exactly how they
   relate; if they cannot be made identical say why in the memo).
2. `src/tac/candidate_seal.py` consumers: wherever the seal/inherit/complete paths compare receiver digests, route through the
   versioned function; add typed consumer-fix rows in the frozen contract's own format (see amendments[0..3] and the
   `prefire_contract_consumer_fix.v1` rows in `.omx/research/ddm_ffi1_20260910/PREFIRE_CONTRACT_FROZEN.json`; validated by
   `cs._pf_freeze_history`); `definition_change: true` for the digest definition, with its parent digest; append the amendment row
   (`prefire_contract_amendment.v1`) — appending invalidates in-flight intents (pr17 NO-GRACE): say so in the memo and in your final
   message, because pc3's intent may be in flight.
3. Tests for every refusal and pass above (real trees on the host where the paths exist; fixtures only for the synthetic refusals);
   `ruff` clean; existing suites green (`test_candidate_seal.py`, `test_decode_wall_clock.py`, `test_decode_wall_clock_t4_direct.py`,
   `test_candidate_prefire_intent.py`, `test_decode_timing_concurrency.py`).
4. Memo `.omx/research/ddm_pr18_manifest_row_exclusion_from_receiver_behavior_digest_20260911.md` with the clause table
   (defect → definition → code location → test → measured proof on the three tree pairs) and `# FORMALIZATION_PENDING:<rationale>`
   or a canonical-equation cite.

## Boundaries
No Modal, no fires, no scorer runs, no candidate archives; never edit `upstream/`, the PR tree, sealed trees, or any arm's directory
(ddm_pc3_*, ddm_ntb2_*, ddm_gpp1, ddm_mxo2/3, ddm_obx2_*); never weaken a refusal that guards a REAL receiver change; never touch
`.omx/research/ddm_rlc5_20260910/` receipts. Commits via `tools/subagent_commit_serializer.py` with post-edit shas, two visible
review passes per .py, no co-author trailer, tags `[no-triality] [p0-ledger-ok]`. Checkpoint as `ddm_pr18`.

## OPTIMAL FORM
- Reference form: the ffi4 scoped risk digest (landed) + the ffi5/ffi6 typed consumer-fix rows + pr17's NO-GRACE freshness rule —
  the "amend through the contract, never around it" pattern; real trees on the host, no ledger sums.
- Provenance pins (record shas in the memo): `src/tac/decode_wall_clock.py`, `src/tac/candidate_seal.py`,
  `.omx/research/ddm_ffi1_20260910/PREFIRE_CONTRACT_FROZEN.json`, pc3's RECEIVER_DIGEST_ROW_DIFF.json (0160ddff…), move 44 tree,
  move 43 tree, pointer move 44 (commit 99625f32f, archive 04758c0dfb8d94ebe801602aac96d93a4f260aad24f58b6b9cdbbe2270ad460e).
- Every delta vs the reference is SCOPE (one digest definition) — no mechanism reduction; no fixtures on the pass path.

## Prior negatives accounted (operator 2026-08-15)
- pr9 condition 1: stale manifests passed inherit — the cure must not reintroduce staleness (independent manifest validation).
- rlc4/pr14: a derived listing inside an identity digest couples identity to raw pins — scope the exclusion to behavior only.
- dwc1: a gate no producer can satisfy is a forever refusal — prove the pass path on the two real candidate trees.
- r9m ×5: two validators disagreeing ⇒ env-coupled digest — the new digest must be content-only and computed identically by both sides.
- pr17 NO-GRACE: an appended amendment row invalidates in-flight intents — announce it.
