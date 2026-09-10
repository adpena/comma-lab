# ddm_pr13 — second-family ratification of MAIN's two pre-fire-contract clarifications and the arm-commit ordering (charter, MAIN 2026-09-10)

You are the pr-family reviewer (pr8→pr12 lineage; second family, adversarial, read-only). pr12's contract was
implemented by ffi3 and landed by MAIN (commit a47543199; frozen receipt
`.omx/research/ddm_ffi1_20260910/PREFIRE_CONTRACT_FROZEN.json` sha 59158b8fce89e12c…). Two clauses of pr12's text
were resolved by MAIN, not by your family. pr12's real-control clause returns them to you. Ratify, amend, or refuse
each — with the exact sentence of pr12 you rely on.

## What you adjudicate
1. **Clarification 1 (ffi2 charter §MAIN's clarification):** pr12 line ~308 `authorization.intent.sha256 = "<intent
   digest>"` vs line ~85 (every referenced JSON validated by exact byte count + SHA-256). MAIN split the single field
   into THREE bindings: `authorization.intent.file_sha256` + `authorization.intent.file_bytes` (bytes on disk) AND
   `authorization.intent.digest` (the canonical self-omitting digest). Question: does binding both strengthen or
   silently change pr12's contract? Is there a case where the two representations can be satisfied by DIFFERENT
   intent files (a digest collision by construction — e.g. two files with identical canonical content but different
   whitespace, one committed, one fired)? Find the code (`src/tac/candidate_seal.py` `build_first_measurement_authorization`,
   `validate_first_measurement_authorization`; `tools/fire_modal_auth_eval.py` first-measurement mode) and say
   whether the fire path pins the FILE it read or only the digest.
2. **Clarification 2 (ffi3 charter §Clarification 2):** `{path, bytes, sha256}` applies to references INTRODUCED by
   the new objects only; the embedded legacy `t4_direct` leg keeps its `{path, sha256}` interior under the unchanged
   `tac.decode_wall_clock` validator. Question: does a new object that embeds a legacy leg by FILE reference
   (`{path, bytes, sha256}` of the leg file) plus the leg's own interior validation leave any reference un-pinned by
   bytes? Enumerate every reference in `candidate_prefire_intent.v1`, `candidate_first_measurement_authorization.v1`,
   `candidate_prefire_timing_risk.v1`, and the v3 completion fields; mark each pinned-by-bytes / pinned-by-sha-only /
   unpinned. Any "unpinned" is a refusal of the clarification.
3. **Arm-commit ordering (rlc4 charter §rule):** pr12 requires a COMMITTED intent before authorization. In this sandbox
   codex arms cannot write Git objects (memory law), so rlc4 produces the intent bytes + serializer bundle and MAIN
   commits them BEFORE running `tools/authorize_candidate_first_measurement.py`. Question: does the ordering
   "arm produces → MAIN commits → MAIN authorizes" satisfy pr12's intent (the commit exists before the
   authorization and the authorization binds the committed file's bytes), or does pr12 require that the PRODUCER's
   commit and the intent be the same actor's act? Quote the sentence. If MAIN's commit is acceptable, state the
   exact custody the authorization must record (commit sha of the intent file) and verify `authorize_candidate_first_measurement.py`
   records it (read the tool; if it does not, that is a finding, not a patch).
4. **Real positive control status:** pr12 §"Prospective freeze and real positive control" says fixtures do not close
   the pass path. rlc4 is producing the first REAL intent now. State what evidence from rlc4's output you require to
   declare the door open (the typed refusal by the normal validator + the intent's file_sha256/bytes/digest + the
   emitter's acceptance) and what would falsify it.

## Method (binding)
- Read pr12 in full (`.omx/research/ddm_pr12_adjudicate_first_fire_intent_contract_20260910.md`, sha 50d00e3956dc7ae5…),
  the ffi2 + ffi3 charters, ffi3's report (`.omx/research/ddm_ffi3_implement_prefire_intent_contract_20260910.md`), the
  frozen receipt, `src/tac/candidate_seal.py` (intent / authorization / completion paths), `tools/make_candidate_seal.py`,
  `tools/authorize_candidate_first_measurement.py`, `tools/fire_modal_auth_eval.py`, and the tests in
  `src/tac/tests/test_candidate_seal.py`. Run the existing suites on the host paths you can (`.venv/bin/python -m pytest
  src/tac/tests/test_candidate_seal.py -q`); a sandbox failure is not evidence.
- Verdict per item: RATIFY / AMEND (exact replacement text) / REFUSE (exact sentence violated). Never loosen: an
  AMEND may only add a binding. Cite `clause → code location → test` for every claim.
- Read-only on code and on all sealed/live trees. Do NOT edit `src/tac/`, `tools/`, `upstream/`, the PR tree, or any
  `/Volumes/...` directory. Do not touch rlc4's or sj1's live directories. No Modal, no fires, no timing windows.

## Deliverable
Memo `.omx/research/ddm_pr13_ratify_prefire_contract_clarifications_20260910.md`: the four verdicts, the reference
pin table (item 2), the ordering ruling with the custody field it requires (item 3), the positive-control evidence
list (item 4), every boundary. If any verdict is AMEND/REFUSE, include the exact patch text MAIN must land (no code
edits by you). Serializer commit of the memo LAST, once (`REVIEW_GATE_OVERRIDE=1` is allowed for the .md); a Git-object
write denial (rc 17) is NOT a stop condition — leave the bundle, report the rc, MAIN lands. Checkpoint as `ddm_pr13`.

## OPTIMAL FORM
- Reference form: pr12's exact contract as the sole normative text; the landed code at a47543199 as the object under
  review; the pr8→pr12 review pattern (clause table with verdicts) as the form. No delta.
- Provenance pins: pr12 memo sha 50d00e3956dc7ae5…; frozen receipt sha 59158b8fce89e12c…; ffi3 charter sha
  dd0de8cd84b59c79…; rlc4 charter sha 096f0db8f4e91be0…; contract commit a47543199; pointer move 42 d2803c214 /
  archive f111ab4259c757409e791247d33978a714ceb1cd66e50c149e2e65fbf208756f.

## Prior negatives accounted (operator 2026-08-15)
- pr10 (rule tuned after data) — the second family owns the freeze; MAIN's clarifications were made BEFORE rlc4's
  real intent existed, and this review lands before MAIN authorizes anything; say whether that ordering held.
- ffi1/ffi2 STOPs — the two clauses they found are exactly items 1–2; do not re-find them, adjudicate them.
- dwc1 (gate with no door) — item 4 exists so the real producer, not fixtures, opens the door.
- r9m (two validators disagree ⇒ env-coupled digest) — item 1's digest-vs-file question is that genus; check whether
  the canonical digest is content-only on both the emitter and the fire side.

Final message: the four verdicts in one line each, any AMEND text, the serializer rc, and the frontier line
`composition S 0.1374765052591843 @ 180,238 B [contest-CUDA T4 n600] (move 42)` unchanged.
