# ddm_pr14 — adjudicate the MANIFEST.sha256-in-normalized-receiver-digest conflict that refused rlc4's real intent (charter, MAIN 2026-09-10)

You are the pr-family reviewer (pr8→pr13 lineage; second family; you own the admission rule). The frozen
pre-fire contract (a47543199; pr13 RATIFIED ×4) met its first REAL producer and refused it. This is the
real-control result pr12 asked for; adjudicate it prospectively, before any evidence is re-timestamped.

## The instance (rlc4; read `.omx/research/ddm_rlc4_resume_rebase_cure_onto_move42_prefire_intent_20260910.md`
and `.omx/research/ddm_rlc4_20260910/RECEIVER_MANIFEST_CONFLICT.json` first)
- Real candidate: archive 180,178 B sha eaf17a7038a4671bf39c3071192b15eec7d7513d471d520b6d13bc9acf94a60b
  (move 42's 180,238 B − 60 B rider cure); twin encodes identical; full cold n600 public raw byte-IDENTICAL to
  move 42's retained raw (1db04341…); 49-row manifest regenerated from outside the tree and verified; 51-file census;
  four smokes PASS. Conditional net −3.9951537e-5 → S 0.13743655372199698.
- Refusal: `tools/make_candidate_seal.py --first-fire-intent …` rc 3, `PREFIRE_RISK_EVIDENCE_REFUSED: receiver risk
  endpoints differ`. Reference normalized receiver digest (the timed rlc1 tree) b06e59a67b60f577…; candidate
  27948d3d5ac0c32b…. The ONLY normalized-path difference is `MANIFEST.sha256` (4,570 B both; reference sha
  98993a00…, candidate 7c2b977a…). Cause: the normalized receiver includes `MANIFEST.sha256` VERBATIM, and the
  manifest lists the RAW hash of `inflate.py`, which carries the counted-archive pin that legitimately changed; after
  normalization no executable byte differs. pr9 REQUIRES the manifest to be regenerated (a stale manifest was pr9
  condition 1). So two second-family requirements collide on one derived file: pr9 "regenerate the manifest" vs pr12
  "normalized receiver digest equals the timed reference for risk inheritance".

## What you adjudicate (verdict per item: RATIFY-AS-IS / AMEND with exact patch text / REFUSE)
1. **Is `MANIFEST.sha256` receiver content?** It is a derived listing of the other files' hashes; it executes nothing.
   Read `src/tac/candidate_seal.py` (`measure_runtime_digest`, the normalization that strips pins, and
   `validate_prefire_risk`'s "receiver risk endpoints" comparison) and `src/tac/decode_wall_clock.py`'s receiver-identity
   fields. Decide whether the normalized receiver digest must (a) EXCLUDE `MANIFEST.sha256`, (b) include a NORMALIZED
   manifest (the manifest recomputed over the normalized files), or (c) keep verbatim inclusion (then rlc4's
   refusal is correct and the cure needs a fresh timed reference). Pick one with the sentence of pr12/pr9 it rests on.
   Never loosen: an amendment may not let any executable byte differ between endpoints.
2. **Reference re-derivation.** Under your chosen definition, the timed reference digest b06e59a6… is no longer the
   value to compare against. Specify exactly how the new reference value is derived from the RETAINED rlc1 timed
   tree (`/Volumes/VertigoDataTier/pact/ddm_rlc1_rule118_cure/candidate_runtime`, read-only) — a recomputation over
   retained bytes, never a re-timestamp — and require that the same function over rlc4's tree
   (`/Volumes/VertigoDataTier/pact/ddm_rlc2_cure_on_move42/candidate_runtime`, read-only) equals it. Compute both
   with a read-only script if you can (record path/bytes/sha of every input); if they differ on anything but the
   manifest, that is a REFUSE with the differing path.
3. **Evidence reuse without re-timestamping.** rlc4's retained evidence (twin encodes, raw identity, manifest, census,
   smokes) was produced against the frozen contract. Decide which receipts a resumed producer may REUSE by
   `{path,bytes,sha256}` and which it must regenerate under the amended rule, and say why each. The intent itself must
   be re-emitted by the producer after the amendment lands (it never existed).
4. **Move 40's legacy leg.** Does the amendment touch the completed `t4_direct` leg validators (`tac.decode_wall_clock`,
   frozen, receiver 6726fd77…)? pr12 preserves them unchanged; confirm the amendment is confined to the NEW objects'
   normalized-digest comparison, or state the exact minimal touch and why it cannot be avoided.
5. **Freeze update.** Specify the exact edit to `.omx/research/ddm_ffi1_20260910/PREFIRE_CONTRACT_FROZEN.json` (append
   an amendment row: your memo sha, the definition change, the recomputed reference values) — an APPEND, never a rewrite.

## Method (binding)
- Read pr12 (`.omx/research/ddm_pr12_adjudicate_first_fire_intent_contract_20260910.md`, sha 50d00e3956dc7ae5…), pr9
  (`.omx/research/ddm_pr9_*`), pr13 (`.omx/research/ddm_pr13_ratify_prefire_contract_clarifications_20260910.md`), the
  frozen receipt, `src/tac/candidate_seal.py`, `tools/make_candidate_seal.py`, `src/tac/decode_wall_clock.py`, and
  the tests `src/tac/tests/test_candidate_prefire_intent.py` + `test_candidate_seal.py`. Read-only on code, sealed and
  live trees; no edits to `src/`, `tools/`, `upstream/`, the PR tree, or any `/Volumes/...` path. No Modal, no fires.
- Every claim `clause → code location → test`. AMEND text must be a literal patch MAIN can land (code + test names +
  the freeze append), and must state the falsifier that proves the amended validator STILL refuses a real executable
  difference (e.g. one byte in `inflate.py` outside the pins → refusal).

## Deliverable
Memo `.omx/research/ddm_pr14_adjudicate_manifest_in_normalized_receiver_digest_20260910.md`: the five verdicts, the
recomputed reference values (item 2) with input pins, the reuse table (item 3), the patch text, the freeze append,
every boundary. Serializer commit of the memo LAST, once (`REVIEW_GATE_OVERRIDE=1` allowed for the .md); rc 17 is NOT
a stop — leave the bundle, report the rc, MAIN lands. Checkpoint as `ddm_pr14`.

## OPTIMAL FORM
- Reference form: pr12's contract + pr13's ratifications as the normative text; the landed code at a47543199 as the
  object; rlc4's real refusal as the instance. No delta.
- Provenance pins: pr12 memo sha 50d00e3956dc7ae5…; frozen receipt sha 59158b8fce89e12c…; pr13 memo sha
  3156992449ea4f71…; rlc4 conflict record `.omx/research/ddm_rlc4_20260910/RECEIVER_MANIFEST_CONFLICT.json` (record
  its sha); contract commit a47543199; pointer move 42 d2803c214 / archive f111ab4259c757409e791247d33978a714ceb1cd66e50c149e2e65fbf208756f.

## Prior negatives accounted (operator 2026-08-15)
- pr10 (rule tuned after data): you are amending a rule AFTER a real refusal — the amendment must be justified from the
  contract's own purpose (receiver-behavior identity for risk inheritance), tighten or hold every executable check, and
  land BEFORE any producer re-runs; say explicitly why this is not a loosening.
- pr9 condition 1 (stale manifest) — the regeneration requirement stands; your definition must make regeneration and
  identity compatible, not drop either.
- r9m (two validators disagree ⇒ env-coupled digest) — a manifest-inclusive digest is that genus (a derived file
  coupled to raw pins); the cure is a content-only definition both sides compute.
- dwc1 (gate with no door) — rlc4 is the first real producer to reach the emitter; the door stays shut until a real
  intent passes; fixtures prove nothing.

Final message: the five verdicts in one line each, the recomputed reference values, the patch summary, the serializer
rc, and the frontier line `composition S 0.1374765052591843 @ 180,238 B [contest-CUDA T4 n600] (move 42)` unchanged.

<!-- # FORMALIZATION_PENDING: review/process charter with no measured row of its own; the contract amendment it produces is a validator definition (receiver identity), not a score law -->
