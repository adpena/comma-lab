# ddm_sr1 — SUBMISSION REVIEW GAUNTLET (deadline: tonight)

**Operator binding 2026-08-17, verbatim: "We want to submit before the end of the evening."**
This arm runs the pre-submission adversarial review passes on the FROZEN packet per the CLAUDE.md
submission gate, under deadline compression (race-mode rigor: passes are real, cadence is fast).

Standing laws: NO submission from this arm (MAIN publishes after operator's final confirm) · NO
Modal/paid · NO scorer runs · serializer commits · upstream/ READ-ONLY · findings reset the
counter; fabricated clean passes are the NO-FAKE cardinal sin — a found defect tonight is CHEAPER
than a found defect in the public PR.

## The packet under review

pq2's gen-2 packet (commits a411f612aa + 7b00461687), staged at
/Volumes/APDataStore/pact/ddm_pq2/submission_staging/, targeted at the rr4 candidate
(archive sha 35ac2b9beb…, 181,161 B, S 0.15853325034789678 [contest-CUDA T4 n600], recovered
contest_auth_eval.json + report.txt in experiments/results/ddm_rr4_cuda_exact_contest_cuda_20260817_r1/returned_artifacts/).
A HOT-SWAP candidate may replace it before freeze (fx1's 180,601 B / S 0.158160 if its T4 row
lands) — review the CANDIDATE-AGNOSTIC surfaces first so a swap invalidates minimal passes.

MAIN's blocker adjudications (binding, review WITH them not against them): dep-bootstrap ACCEPTED
(e4/PR100-101 precedent) · contest-CPU row OPTIONAL (PR135/138 GPU-eval field norm; declare
"requires GPU for evaluation, same as #135") · schema naming drift ships as-is (receipt not
edited) · promotion-policy circularity waived-with-rationale (internal gate, invisible to contest).

## The passes (each = one full adversarial sweep; findings reset; report per pass)

Order candidate-agnostic-first: (1) COMPLIANCE — the pre_submission_compliance_check surface,
archive custody chain, rule-118 accounting (no video-derived data in code), payload cleanliness,
dependency closure (bare-venv smoke receipt), inflate determinism claims. (2) PR BODY + HONESTY —
every number in the draft traced to a receipt; NO-FAKE #7 borrowed-substrate accounting complete
and honest (inherited PR130/135-class HPAC substrate itemized vs OUR mechanisms: the recode, the
selection, the micro-edits, the mixer if it ships); claimed-score language matches author-asserted
norms (no eval-bot claim); GPU-eval declaration present; competitive/innovative statement
UNQUESTIONABLE per the Innovation Gate. (3) COMPRESSION SCRIPT — runnable-by-a-judge read-through;
sanitization sweep (NO fleet IPs, private URLs, local absolute paths, provider logs — grep the
whole staged tree); the sha assertion logic correct. (4) MECHANICS DRY-RUN — the release+PR
command sequence MAIN prepared (verify flags against gh's real interface; the archive-URL flow
matches the PR135/138 body pattern). (5) FINAL SWEEP — fresh top-to-bottom read of the frozen
tree as a hostile maintainer would.

If the fx1 swap fires mid-gauntlet: re-run ONLY the candidate-bound passes (1 partially, 2's
numbers, 4's sha pins) — record which passes survive the swap and which reset.

## Deliverables

1. Per-pass findings (fix-or-flag each; trivial fixes applied via serializer, judgment calls
   surfaced to MAIN immediately — do not sit on a blocker).
2. `.omx/research/ddm_sr1_submission_gauntlet_20260817.md` — pass ledger with counter state,
   the final GREEN/RED verdict, and the exact residual risk list for the operator's final confirm.
3. Final message: counter state + verdict + anything MAIN must do before publishing. End with the
   own-vehicle frontier line + whether your unit moved it (no — say so).
