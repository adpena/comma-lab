# ddm_ffi2 — resume ddm_ffi1: implement ddm_pr12's pre-fire intent contract with the digest clarification (charter, MAIN 2026-09-10)

ddm_ffi1 (FINISHED rc=0; report `.omx/research/ddm_ffi1_implement_prefire_intent_contract_20260910.md`,
final message under `arm_final_messages/ddm_ffi1_*`) correctly STOPPED on one ambiguity in pr12's text:
line ~85 requires every referenced JSON to be validated by "exact byte count and SHA-256", while line
~308 sets `authorization.intent.sha256` to "<intent digest>" — and the canonical intent digest omits its
own self field, so the file-byte SHA-256 and the canonical digest differ.

## MAIN's clarification (binding for this implementation; recorded for the second family to ratify)
Bind BOTH, and refuse on either mismatch:
- `authorization.intent.file_sha256` = the exact file-byte SHA-256 of the committed intent JSON and
  `authorization.intent.file_bytes` = its exact byte count (the general custody rule, pr12 line ~85);
- `authorization.intent.digest` = the canonical self-omitting intent digest exactly as pr12 defines it
  (the "intent digest" the authorization names, line ~308), which the intent's own self field carries.
The dispatch custody record stores both. Nothing is loosened: the authorization now pins the bytes on
disk AND the canonical content. Where pr12 says `authorization.intent.sha256`, implement the two fields
above and record in the memo that the single field was split into a two-field binding by MAIN's
clarification pending pr12's family ratification (the real-control clause brings it back to them).

## Everything else
The ffi1 charter binds unchanged:
`.omx/research/ddm_ffi1_implement_prefire_intent_contract_charter_20260910.md` (surfaces, deliverables —
code + tests in ONE serializer commit with post-edit shas and two review passes per .py; the frozen
contract receipt `.omx/research/ddm_ffi1_20260910/PREFIRE_CONTRACT_FROZEN.json` (keep that path; add the
clarification text and its sha to the receipt); the per-clause compliance table; boundaries: no Modal,
no fires, no windows, no n600, no edits to `src/tac/decode_wall_clock.py`'s completed `t4_direct`
requirements, sealed trees, `upstream/`, the PR tree). Land ffi1's retained report bundle content only
if it is byte-identical to the report file already in the working tree (it is — MAIN committed the
report); do not re-land it.

## OPTIMAL FORM
- Reference form: pr12's exact contract (sha 50d00e3956dc7ae5…) + this clarification; no other delta.
- Provenance pins: pr12 memo sha, ffi1 report sha (record), 0524522f0 (t4_direct), pointer move 42 d2803c214.

## Prior negatives accounted (operator 2026-08-15)
- ffi1/rlc2/mv2/mv3: STOP on exact clauses — this charter resolves the one clause found; if another
  appears, STOP the same way with the exact sentence.
- dwc1's gate with no door: fixtures cannot close the pass path; say so in the memo (pr12 §freeze).

Checkpoint as `ddm_ffi2`. Final message: commit sha, frozen receipt sha, test counts, the per-clause table
summary incl. the split-field note, every boundary, and the frontier line
`composition S 0.1374765052591843 @ 180,238 B [contest-CUDA T4 n600] (move 42)` unchanged.
