# ddm_pr18 — the derived listing leaves the receiver BEHAVIOR digest, and is validated instead

`# FORMALIZATION_PENDING: this is an identity-definition amendment to a custody contract, not a
score/rate/distortion law. It produces no equation of the S-arithmetic; the canonical equation it
must not disturb is the rate term itself (archive.zip bytes), which no digest definition touches.
Formalize if a later unit generalizes "derived restatements leave identity digests" across custody
surfaces.`

**Axis: `[exact digest arithmetic]`. No score claim, no promotion claim, no fire. Pointer unmoved:
composition S 0.1372449041713402 @ 180,406 B [contest-CUDA T4 n600] (move 44).**

## Answer first

The receiver behavior digest now excludes `MANIFEST.sha256` and validates it independently instead.
MEASURED on the host: move 44's tree and pc3's candidate tree are **EQUAL** under the new definition
(`9f6e71680a13d8598974ee13f78a1a72759a758681e6d86b7b288cc105442890`) and **DIFFERENT** under the old
one (`7e6da183f3b3…` vs `fad971b11b79…`); move 44 and the move-43 tree stay **DIFFERENT under both**.
pc3's candidate inherits move 44's `t4_direct` leg today, on the real trees, with the definition it
used recorded in a typed field. Two candidates were blocked by the old definition (pc3 −160 B,
ntb2 −245 B); the block is gone, and nothing that guards a real receiver change was relaxed.

## The defect (measured, not argued)

`MANIFEST.sha256` is a **derived** file: it restates the raw sha256 of every other shipped file,
including `inflate.py`'s. `inflate.py` carries the two archive pins, so its raw hash moves with every
archive — and the receiver digest normalizes those pins away in `inflate.py` only to re-admit them,
one layer down, through the manifest's restatement of them. The identity therefore changed for every
archive-only successor, and `inherit_decode_wall_clock` refused a receiver it had already measured.

pc3's row diff (`RECEIVER_DIGEST_ROW_DIFF.json`, sha `0160ddff…`): of 50 rows exactly one differs.

I also measured a second, unrecorded difference the row diff did not name: **pc3's listing has 50
entries, move 44's has 49** — pc3's producer lists `archive.zip` as well. The manifest row differs
for two reasons at once (a moved `inflate.py` hash and an extra listed line), and both are properties
of a restatement, not of receiver behavior. The validation below accepts either listing shape and
holds both to the same hash equality.

## Clause table — defect → definition → code → test → measured proof

| # | Defect / requirement | Definition | Code | Test | Measured proof |
|---|---|---|---|---|---|
| 1 | Derived listing couples identity to archive pins | behavior digest = receiver rows minus `MANIFEST.sha256`; legacy digest unchanged | `decode_wall_clock.receiver_identity` / `measure_receiver_behavior_digest` (`…v2`) | `test_the_listing_leaves_the_behavior_identity_but_not_the_raw_one` | move 44 vs pc3 EQUAL `9f6e7168…` under v2; DIFFERENT under v1 |
| 2 | pr9 condition 1: a stale listing must never ride through | independent validation: every listed file exists, every listed hash re-derives from raw bytes, `inflate.py` listed and matching, nothing shipped is unlisted (`archive.zip` optional, `MANIFEST.sha256` itself excluded) | `decode_wall_clock.validate_receiver_manifest` | `test_every_broken_listing_refuses_before_it_can_be_excluded` (9 mutations), `test_a_listing_must_name_the_receiver_entry_point`, `test_a_listed_archive_is_allowed_but_still_held_to_its_bytes` | the move-43 tree on disk today carries a STALE `inflate.py` row (listed `1e14a68a…`, actual `598dbd7a…`) and REFUSES |
| 3 | Validation must run on every tree the custody path touches | the behavior digest cannot be obtained without it; every leg validation computes it for the candidate tree and (recursively) the source tree; the intent path validates its own tree | `validate_decode_wall_clock`, `candidate_seal._pf_identity` | `test_every_broken_listing_refuses_before_it_can_be_excluded` (via `_validate`), `test_prefire_risk_manifest_exclusion_does_not_bypass_dependency_manifest_validation` | a broken listing refuses at `PREFIRE_IDENTITY_DRIFT_REFUSED: derived listing invalid` |
| 4 | Stored receipts carry the OLD digest | a stored digest is accepted under either definition, and WHICH one matched is recorded | `decode_wall_clock.compare_receiver_identity` → `stored_digest_definition` | `test_a_stored_digest_may_be_written_under_either_definition` | move 44's leg stores `7e6da183…` (legacy) and still names its own tree |
| 5 | Every comparison records its definition | typed `receiver_identity_comparison.v1` on the emitted inherited leg and in `observed`; re-derived and refused on disagreement | `decode_wall_clock._receiver_comparison`, `inherit_decode_wall_clock` | `test_an_archive_only_successor_inherits_and_records_the_definition`, `test_a_recorded_comparison_cannot_disagree_with_the_trees` | pc3's live inherited leg carries `comparison_digest_definition: …v2`, `behavior_digests_equal: true`, `legacy_digests_equal: false` |
| 6 | A real receiver change must still refuse | cross-tree comparison is over the behavior rows, which cover every executable file | `validate_decode_wall_clock` inherit branch | `test_a_real_receiver_change_still_refuses_inheritance` | move 44 vs move 43 DIFFERENT under both definitions |
| 7 | Amend through the contract, never around it | a SUPERSEDING amendment row may follow consumer fixes only by pinning the definition it replaces and the row it follows | `candidate_seal._pf_freeze_history`, `PREFIRE_AMENDMENT_DEFINITIONS` | `test_a_superseding_amendment_may_follow_consumer_fixes_and_becomes_the_parent`, `test_a_superseding_amendment_cannot_reset_the_chain` (5 refusals) | the appended `prefire_contract_amendment.v1` + `prefire_contract_consumer_fix.v1` rows validate through `_pf_freeze_history` |

## The three tree pairs (MEASURED on the host, read-only)

Retained: `.omx/research/ddm_pr18_20260911/RECEIVER_BEHAVIOR_DIGEST_PROOF.json`.

| Pair | legacy `measure_receiver_digest` | behavior `…measure_receiver_behavior_digest.v2` |
|---|---|---|
| move 44 (`ddm_rlc5_cure_on_move43/candidate_runtime`) | `7e6da183f3b318a4de4fb4dafcdb2f768aa04b02c65f5a4507a821a3cb2c868b` | `9f6e71680a13d8598974ee13f78a1a72759a758681e6d86b7b288cc105442890` (49 rows) |
| pc3 (`ddm_pc3_pose_carrier_curve/candidate/candidate_runtime`) | `fad971b11b799cedc82f07bbcbf2e6a07a66a7867bb353754fe5771bc1fc3256` | `9f6e71680a13d8598974ee13f78a1a72759a758681e6d86b7b288cc105442890` (49 rows) |
| move 43 (`ddm_sj1_pass6/candidate/candidate_runtime`) | `6726fd77a7c4fa80b70ce37accb30595c1004cdd91b9ef990bf9c12f21c296bf` | REFUSED — `manifest hash differs from the file's raw bytes: inflate.py`; its unvalidated row digest is `b37ac3f122188a6b9d990b072b0f4f45be08ffb5e9048f84877c63f5f82c58ab` ≠ `9f6e7168…` |

- **move 44 vs pc3** — EQUAL under the new definition, DIFFERENT under the old. The measured defect,
  and its cure.
- **move 44 vs move 43** — DIFFERENT under both. A real receiver change still refuses; and the
  move-43 tree refuses one step earlier at the listing validation, which is a *stronger* refusal
  than the charter asked for, not a weaker one.

## How this relates to the ffi4 49-row risk digest

Row-for-row **identical**, MEASURED on all three trees: same skip rules
(`runtime_digest_skip_reason`), same `archive.zip` exclusion, same `<ARCHIVE_PIN>` normalization,
same `(relative_path, normalized_bytes, sha256)` tuple, same `json.dumps(separators=(",", ":"))`
digest, same single exclusion. On move 44 and pc3 both produce `9f6e7168…`, which is exactly the
`amended_sha256` the pr14 amendment recorded for the rlc1/rlc2 reference receivers — four trees, one
receiver, one behavior identity.

They differ in two ways, both deliberate:

1. `measure_prefire_risk_receiver_digest` **requires both archive pins** to be present in
   `inflate.py`; the decode_wall_clock definition does not (it normalizes whichever pins it finds).
   The prefire path wants that extra refusal at intent time.
2. The v2 behavior digest **cannot be computed without validating the listing**; the ffi4 risk digest
   excludes the listing unconditionally. I deliberately did NOT add validation to the frozen ffi4
   definition — it is frozen, and an in-flight arm's tree is not mine to newly refuse. The pr18
   amendment declares the validation for its own definition and for the intent identity path.

So: the same 49 rows, reached by two functions that keep their own entry conditions. They cannot be
collapsed into one call without re-opening the frozen ffi4 definition, which is out of pr18's scope.

## What was NOT changed (and why)

- `measure_receiver_digest` (v1) is byte-identical in behavior. Every raw custody statement — the
  seal's `measure_runtime_digest`, the intent's `normalized_receiver`, the receiver file pins — still
  pins the manifest's exact bytes. The exclusion is scoped to ONE question: "is this the same
  receiver?"
- The completed `t4_direct` leg's other requirements are untouched, and its document shape is
  unchanged (only the *inherited* leg gains the typed comparison field), so a previously stored
  `t4_direct` leg still re-validates byte-identically.
- The seal's `receiver_sha256` for a candidate's own tree remains the legacy digest: a single-tree
  self-check has no cross-tree coupling to remove, and the raw statement is stronger.

## Honest cost of the exclusion

The behavior digest no longer pins the manifest's exact BYTES; it pins the manifest's
CORRECTNESS. Two different byte-strings can both validate (line order, an optional `archive.zip`
line, trailing whitespace). What the exclusion gives up is therefore real, and it is recovered in
full by `measure_runtime_digest`, which every seal and every intent still carries. What it buys is
that identity stops moving for a reason that has nothing to do with the receiver.

## Attacking my own conclusion

- *Could a candidate delete `MANIFEST.sha256` to dodge validation?* Then nothing is excluded, the two
  digests coincide, and the raw runtime digest plus the intent's dependency-manifest evidence still
  see the deletion. The exclusion is only available to a tree that ships a listing that validates.
- *Does excluding the listing hide a receiver change?* `MANIFEST.sha256` is not read by the receiver:
  MEASURED — `grep MANIFEST inflate.sh inflate.py README.md` in move 44's tree returns nothing. Every
  executable file remains in the behavior rows.
- *Is the new comparison looser than the old one?* Yes, by exactly one row, and only for trees whose
  listings both validated. Equal legacy digests imply equal behavior digests (the behavior rows are a
  subset), so no tree pair that passed before can fail now, and no pair that differs in any
  behavior-bearing byte can pass.
- *Did I make a gate no producer can satisfy (dwc1)?* No: both live candidate trees pass, and the
  pass path was exercised on the real trees, not on fixtures.

## NO-GRACE consequence (pr17)

Appending rows to `PREFIRE_CONTRACT_FROZEN.json` **invalidates every in-flight prefire intent**: an
intent pins `amendment` to the frozen file's last row, and that row is no longer last. pc3's intent,
if it is already written, must be re-emitted against the new latest row (`build_prefire_intent` reads
the frozen file, so a re-run picks it up with no edit). This is the contract working, not a defect.

## Boundaries honored

No Modal, no fires, no scorer runs, no candidate archives, no `/tmp` evidence path. `upstream/`, the
PR tree, sealed trees and every arm directory (`ddm_pc3_*`, `ddm_ntb2_*`, `ddm_rlc5_20260910/`,
`ddm_gpp1`, `ddm_mxo2/3`, `ddm_obx2_*`) were READ ONLY — the pc3 and move-43 trees were hashed, never
written. No refusal that guards a real receiver change was weakened.
