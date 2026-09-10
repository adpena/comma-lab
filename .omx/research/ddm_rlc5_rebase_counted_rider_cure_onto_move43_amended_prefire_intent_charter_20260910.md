# ddm_rlc5 — re-base the counted-rider cure onto MOVE 43 and emit the real pre-fire intent under the AMENDED contract (charter, MAIN 2026-09-10)

## What changed since rlc4
1. **The pointer moved.** Move 43: S 0.1372848557085275 @ 180,466 B [contest-CUDA T4 n600], archive sha
   7beb6a5fc7c2bf477d04a107ab0cf4113d3bd74b2c5a6b94ff77f7ff61971c1e, tree (read-only; copy)
   `/Volumes/VertigoDataTier/pact/ddm_sj1_pass6/candidate/candidate_runtime`, retained cold n600 public raw at
   `/Volumes/VertigoDataTier/pact/ddm_sj1_pass6/` (find the parse-back `0.raw`, record path/bytes/sha). rlc4's candidate
   (180,178 B) was on move 42's bytes and is now a superseded base: every base-dependent artifact REGENERATES on move 43.
2. **The contract was amended (pr14 → ffi4 → freeze append).** `MANIFEST.sha256` is excluded ONLY from the new versioned
   timing-risk receiver digest (`tac.candidate_seal.measure_prefire_risk_receiver_digest.v1`); every custody check keeps
   it. Implementation commit d05c2fbd27c7c2eeddd7cd681b4bd045cac30800; freeze append commit a6125f2a79e1a35c8ee22a641fecfdf3614d63bd
   (`.omx/research/ddm_ffi1_20260910/PREFIRE_CONTRACT_FROZEN.json`, now sha a3358944…; `amendments[0]`). The timed RLC1
   reference recomputes to `9f6e71680a13d8598974ee13f78a1a72759a758681e6d86b7b288cc105442890` under the scoped digest;
   your candidate's scoped digest MUST equal it (the only permitted differences are the archive pins in `inflate.py`,
   which normalize away, and the manifest, which is excluded from this one digest).
3. **rlc4's scripts are landed** (`experiments/ddm_rlc4_rebase.py`, `ddm_rlc4_evidence.py`, `ddm_rlc4_public.py`,
   `ddm_rlc4_native_cache.py`, `ddm_rlc3_move42_trace.py`; memo `.omx/research/ddm_rlc4_resume_rebase_cure_onto_move42_prefire_intent_20260910.md`).
   Reuse them parameterized to move 43; do not rewrite them. Work root: `/Volumes/VertigoDataTier/pact/ddm_rlc5_cure_on_move43/`.

## Deliverable (rlc3/rlc4's chain on the new base; commit LAST)
1. Real move-43 re-base of the counted-rider cure: exact seven-path receiver delta; rider rebuilt against move 43's
   archive; archive = move 43's bytes + rider by the REAL coder; retain BOTH full encodes (twins identical).
   Gate: `archive_bytes ≤ 180,466 − 21 = 180,445` is the trivial bar; the real bar is the admit bar at move 43:
   net_dS = 25·(B − 180,466)/37,545,489 must be < −2e-5, i.e. **B ≤ 180,435**. Expect ≈ 180,406 (−60 B) — prove it.
2. Full cold n600 public parse-back proving raw byte identity with move 43's retained raw; `MANIFEST.sha256` regenerated
   from OUTSIDE the tree and validated independently; pr9's literal census; candidate AND frontier public smokes.
3. Timing-risk receipt (`candidate_prefire_timing_risk.v1`) under the amended definition: scoped digest of your tree =
   9f6e7168… (record the 49 normalized rows' listing sha), lineage to move 40's completed `t4_direct` leg (990.054 s) and
   move 43's own leg (`/Volumes/VertigoDataTier/pact/ddm_sj1_pass6/` t4_direct 978.12 s); risk ceiling ≤ 1,260 s;
   exactly the fields `validate_prefire_risk` reads.
4. Emit + self-validate the intent with `tools/make_candidate_seal.py --first-fire-intent … --timing-risk-evidence …`
   (pointer = move 43, sha 7beb6a5f…; admit bar as above; both smoke groups; retained payload paths; falsifiers
   pre-registered; `score_claim=false`; the `contract` block must name the freeze's latest amendment exactly). Then the
   two refusal controls: normal `validate_seal` must refuse with exactly `PREFIRE_INTENT_SCHEMA_REFUSED`; normal
   `tools/fire_modal_auth_eval.py --seal <intent>` must refuse BEFORE any subprocess. Record both receipts.
5. Memo `.omx/research/ddm_rlc5_rebase_counted_rider_cure_onto_move43_amended_prefire_intent_20260910.md`: bytes + sha,
   raw identity, census, smokes, risk receipt, the intent's path / file_sha256 / bytes / canonical digest
   (`prefire_digest(intent,"intent_sha256")`), both refusal receipts, the conditional row, every boundary.
   Serializer commit LAST, once (two review passes per .py; `[no-triality] [p0-ledger-ok]`; no co-author trailer). A
   Git-object write denial (rc 17) is NOT a stop: leave the verified bundle + format-patch, report the rc; MAIN commits
   the exact intent bytes and then authorizes. Checkpoint as `ddm_rlc5`.

## Boundaries (binding)
NO Modal, NO `authorize_candidate_first_measurement.py`, NO fire, NO completion (MAIN's). No timing windows. Never edit
`upstream/`, the PR tree, sealed trees (`ddm_sj1_pass6`, `ddm_rp1_round2`, `ddm_rlc2_cure_on_move42`), `src/tac/decode_wall_clock.py`,
or the contract code; if the amended contract refuses your REAL intent for a reason you believe is a defect, STOP with the
exact refusal (that is a real-control result), do not patch around it. Keep every payload on the SSD tier (hardlink
byte-identical raws with certificates); heavy steps through `tools/launch_detached_process.py --done-receipt …`. Do not
touch swp2's staging or gdc1's directories. Rule 118: the rider is the only counted content; quote the literal census.

## OPTIMAL FORM
- Reference form: rlc4's landed chain (d598ef3b6 trace; rlc4 bundle landed) on move 43's tree + the amended contract as
  frozen (d05c2fbd / a6125f2a); real coder, real n600 parse-back, real smokes — no ledger sums, no fixtures.
- Provenance pins: pointer move 43 commit 48109233e / archive 7beb6a5fc7c2bf477d04a107ab0cf4113d3bd74b2c5a6b94ff77f7ff61971c1e;
  contract a47543199 + amendment d05c2fbd27c7c2eeddd7cd681b4bd045cac30800 + freeze append a6125f2a79e1a35c8ee22a641fecfdf3614d63bd;
  pr14 memo sha 6e1732baea1912731fef6a53abca0cb94faf2e9440accc304dd9b34e593bade9; rlc4 memo (record its sha); timed RLC1
  reference tree `/Volumes/VertigoDataTier/pact/ddm_rlc1_rule118_cure/candidate_runtime` (scoped digest 9f6e7168…).

## Prior negatives accounted (operator 2026-08-15)
- rlc4: refused on the manifest-in-digest defect — cured by the amendment; your scoped digest must equal 9f6e7168… or STOP.
- rlc3: stopped at its first commit — commit LAST, continue on rc 17.
- rlc2 / ffi1 / ffi2 STOPs (clauses) — resolved; a NEW refusal from the real producer is reported exactly, never patched.
- pr9 condition 1 (stale manifest) — regenerate from outside the tree; rp1 r2 (flag vs constant) — bind base archive/tree
  by sha in every receipt; sj1's silent revert — archive must be move 43's bytes + rider, proven by raw identity.
- dwc1 (gate with no door) — this is the second real door test; only the real intent + MAIN's real harvest open it.

Final message: candidate bytes + sha, raw identity, census, the intent path with file_sha256/bytes/digest, both refusal
receipts, the conditional row, the serializer rc, every boundary, and the frontier line
`composition S 0.1372848557085275 @ 180,466 B [contest-CUDA T4 n600] (move 43)` unchanged (MAIN fires).

<!-- # FORMALIZATION_PENDING: producer charter; the row it enables is scored only by the exact T4 harvest -->
