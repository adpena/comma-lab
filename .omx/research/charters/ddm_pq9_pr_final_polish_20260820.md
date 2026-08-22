# ddm_pq9 — PR final polish: hosting wired, body finished-to-operator-line, gated figures held

## MANDATE

Operator directive 2026-08-20: *"we can wait hours for the composed candidate and use that time
to polish and finish the PR."* Hosting is RESOLVED and VERIFIED by MAIN this turn: the jg5
archive is committed to the public repo and the commit-pinned raw URL serves the EXACT bytes —

    https://raw.githubusercontent.com/adpena/comma-lab/2d61b51988799ec3561d5f8a6f659aeb88cc99d9/submissions/robust_current/jg5_sub015_runtime/runtime/archive.zip
    → HTTP 200, sha256 f3bce5d259a081839c48d8089c2b43a57cc7cc96cf5b8f787ff85089be8acb7e (verified by download)

This arm brings the packet to FINAL-MINUS-OPERATOR-LINE. It does NOT publish, host anything new,
open a PR, or fire any eval.

## WORK ITEMS

1. **Hosting section into `PR_BODY_DRAFT.md`** — the pinned-URL mechanism above, stated as
   mechanics-verified. CARRY THE SWAP NOTE: if the rc2 composed candidate (rider archive
   `df7fd266…` @ 180,456 B) confirms on T4, the shipping archive CHANGES — it must be committed,
   pushed, and the URL RE-PINNED to that commit before publish. The pq1 swap procedure is the
   instrument; extend it with the URL-re-pin step.
2. **Fold pq5 (review personas) + pq6 (headroom/methods) findings** into the body draft — each
   fold cited to its source memo row; no invented content.
3. **LLM-disclosure section: DRAFT ONLY.** Prepare an honest draft (the setup, the NO-FAKE #7
   borrowed-substrate accounting pointer, the "most of the code" answer scaffolded from the
   measured accounting) — clearly marked OPERATOR MUST REWRITE IN THEIR OWN WORDS. The policy
   answer itself remains operator-owned; the existing notice at the top of the draft stays.
4. **Runtime figures: placeholders, not transfers.** The declared runtime + the 4,369.6 s
   contest-CPU figure need re-derivation against the composed tree (rr8 fire-order item 4).
   Mark each figure GATED-ON-RC2 with the measured source it will come from. Do NOT transfer
   464.559 s (instrumented tree) or 1,283 s (local CPU) — the cross-regime constant-transfer
   genus has cost three corrections this week.
5. **Reviewer-executable verification appendix** — the exact command walk a contest reviewer
   runs: `shasum -a 256 -c MANIFEST.sha256` (33/33), archive sha vs the hosted URL, the T4
   receipt recompute. Every claim executable, none asserted.
6. **One genuinely-fresh review pass** over the three packet documents (this arm assembled
   nothing in the packet, so it qualifies as fresh eyes). Record it against the 5-consecutive-
   clean-pass counter HONESTLY: a finding resets to 0; state the counter value at exit.

## HARD CONSTRAINTS

- Packet edits ONLY through the canonical stager (pq8's tool, `d678b60c24` fixed derivation);
  re-verify MANIFEST 33/33 + `runtime_tree_sha256` 2103073d… UNCHANGED after every doc edit
  (docs are outside the runtime tree — if that hash moves, STOP: something touched the runtime).
- NO publish, NO push to the contest repo, NO new hosting actions, NO Modal fire.
- NO edits under `submissions/robust_current/jg5_sub015_runtime/` or `upstream/`.
- Serializer commits w/ post-edit shas; any `.py` touched = 2 genuine review passes.
- File ownership: this arm owns `generations/gen5*` docs + PR body; ddm_sw1 (parallel) owns the
  repo-wide path scrub and must NOT touch the packet — if sw1's census flags a packet file,
  it routes here as a row, not an edit.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- **rv15 F1** — the packet carried "15.3% slower" with the WRONG denominator on three surfaces
  (token-stage ratio paired with inflate walls). Cure encoded here as item 4's no-transfer rule.
- **rv15 F2** — the staging tool's tree re-derivation was a TAUTOLOGY (hashed its own manifest
  rows); fixed at mechanism `d678b60c24`. This charter re-verifies from MEASURED bytes only.
- **#877 rounded-display trap** — the rr8 receipt's `final_score` field literally reads `0.15`;
  every number in the body must be recomputed from components, never read from display fields.
- **The 5-pass review counter sits at 0** because every round this week found real defects —
  the PRIOR-LAW PREDICTION below is calibrated on that streak, not on optimism.

## OPTIMAL FORM

- Family reference: pq8's freeze-assembly discipline (measured-bytes derivation, executable
  identity claims) — landed form `.omx/research/ddm_pq8_packet_freeze_assembly_20260820.md`,
  stager derivation fix commit `d678b60c24`. Provenance pins: archive custody commit
  `2d61b51988` (hosted-URL pin); runtime tree sha256
  `2103073d739fc3f27d329ea0785ea3010307360c2380af0476e16d0f5b57cb9b` (must be UNCHANGED after
  every doc edit); pq5/pq6 source memos `.omx/research/ddm_pq5_fresh_eyes_pr_review_20260820`
  + `ddm_pq6_headroom_methods_sections_20260820` at HEAD.
- SCOPE reductions declared per row. MECHANISM reductions FORBIDDEN: no asserted-but-not-
  executable claim in the body; no figure carried across trees/axes; no counted review pass
  that found-and-ignored.
- **PRIOR-LAW PREDICTION (falsifiable):** the fresh review pass finds ≥1 real defect in the
  packet docs (every review round this week has; counter sits at 0 for cause). FALSIFIER: a
  genuinely clean pass — count it and say so plainly.

## DELIVERABLE

`.omx/research/ddm_pq9_pr_final_polish_20260820.md` — per-item rows {what changed · receipt ·
verification}, the review-pass verdict + counter state, the remaining operator-only list
(unchanged or shortened, never grown silently). End with the own-vehicle frontier line.
