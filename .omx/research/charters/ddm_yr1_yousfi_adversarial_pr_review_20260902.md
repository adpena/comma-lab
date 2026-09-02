# ddm_yr1_yousfi_adversarial_pr_review — Yousfi-persona adversarial review of the gen-7 AFR1 packet AS AN OFFICIAL PR: ship-set vs internal-custody adjudication + maintainer red-flag pass (review round 2, memo ddm_pq14_drive_pr_review_round1_20260902.md lineage)

## MANDATE

Operator 20260902 ×2, verbatim: *"You are creating a lot of files and doing stuff that I don't
think is best practices perhaps for an official PR"* + *"Useful to have but we need adversarial
review from yousfi pwrspective [perspective]"*.
This arm is review round 2 of the frozen generation-7 AFR1 packet, re-aimed per the operator's
steer: review the packet and the PR body drafts THROUGH THE MAINTAINER'S EYES — Yassine Yousfi,
contest designer, steganalysis expert, author of the coding-agents policy, the person who will
actually receive, run, and judge this PR. The packet has accumulated a large internal doc
surface (README.md 5,292 B · COMPRESS.md 6,974 B · report.txt 4,309 B ·
BORROWED_SUBSTRATE_ACCOUNTING.md 51,202 B · MANIFEST.sha256 · archive_manifest.json · review
scaffolds — ~90 files total) that the intake corpus shows no winning PR shipping (VERIFY against
the intake memos — this is the prior-law prediction below, to be tested not assumed). The
maintainer-seat adjudication of WHICH files belong in the PR attachment set vs the repository vs
internal custody has not been run: round-1 (`ddm_pq14_drive_pr_review_round1_20260902.md`)
reviewed content correctness, not ship-set membership — cite it as the baseline, do not repeat it. MAIN is the fixer of this packet and cannot self-review (fresh-eyes law);
this arm supplies the independent adversarial pass NOW, before any publish confirm.

## SCOPE

1. **Reconstruct what the maintainer actually receives and expects.** Authority: the live PR
   template + coding-agents policy receipts in
   `.omx/research/ddm_pq7_*` (census) and `.omx/research/ddm_pq14_drive_pr_review_round1_20260902.md`
   (round-1 + addenda, incl. the F13 verbatim-report and F14 policy-slot rows), plus what the
   winning/relevant PRs ACTUALLY attached: PR #130/#133/#135/#136/#138 intake memos
   (`ddm_pi135_pr135_intake_20260810.md`, `pr86_pr130_fullstack_intake_20260728.md`,
   `ddm_hx1_pr_wave_harvest_20260817.md`). Produce a comparison table: their attachment set vs
   our packet's file census.
2. **Ship-set adjudication — the operator's core question.** For every file in
   `/Volumes/APDataStore/pact/ddm_pq12/generation_7_afr1/` (read-only), classify
   {SHIP-IN-PR (attached/inlined) · REPO-SIDE (lives in the public source repo, linked from the
   PR body) · INTERNAL-ONLY (custody, never surfaced)} with a one-line reason each. The PR body
   drafts (`PR_BODY_FINAL_DRAFT.md` + `_TIGHT.md` under
   `.omx/research/ddm_pq1_submission_packet_prep_20260815/`) currently reference internal docs —
   flag every reference that would dangle or confuse a maintainer who has only the PR.
3. **Adversarial Yousfi pass on the PR body drafts + shipped runtime.** What raises flags for
   THIS maintainer: the network Brotli self-install at inflate time · the `cc` compile at decode
   (unguarded dep) · CUDA-only declaration with no CPU row · the 0.15-display vs recomputed
   0.14797617125559104 framing · the coding-agents-policy disclosure posture · the 51 KB
   self-audit accounting file's tone and whether attaching it helps or hurts · anything reading
   as machine-generated bulk vs a human submission. Would he run it? Would he trust it? What
   single question would he ask first?
4. **Practical compliance from the maintainer's seat:** does the SHIP-set alone (without our
   repo checkout) inflate and evaluate? Template question coverage complete? 30-min budget claim
   legible? Every claim in the PR body verifiable from what he holds?
5. **Ranked findings F1..Fn** (severity + exact fix + owner: MAIN-executable vs OPERATOR-slot
   per the policy that final public text is operator-authored). This is a finding round of the
   5-pass counter (currently 0/5): any finding resets it, and that is the desired honest outcome.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO Modal fire from the arm (MAIN owns dispatch + single-flight).
- The local SCORER LANE belongs to MAIN, always. Do NOT write who currently holds it into a
  charter: an occupancy claim goes stale the moment that holder exits, and the arm has no way
  to learn it did (the #1210 stale-precondition genus — MEASURED 2026-08-29, when
  `ddm_bz2_bornsmall_capacity_ceiling` correctly refused to claim a capacity ceiling because
  a charter told it a since-released lane was taken). If this arm's work needs a scorer run,
  emit a typed fire order naming its trigger and let MAIN fire it; landing an honest partial
  plus a fire order is the CORRECT outcome, never a failure.
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes.
- ALWAYS KEEP THE PAYLOAD; bulky receipts to `/Volumes/APDataStore/pact/ddm_yr1_yousfi_adversarial_pr_review/`.
- DETACHED >30-MIN COMPUTE: any single compute step projected to exceed 30 minutes MUST
  launch outside the arm session with `nohup` + `disown`, a pidfile, crash-resumable stage
  checkpoints, and a durable done-receipt. The arm MONITORS that process; a successor or
  MAIN harvests the done-receipt. An in-session multi-hour compute loop is FORBIDDEN.
- CLOSED-FORM-FIRST (operator 2026-08-31 "All upstream can be closed form"): the scoring
  chain is frozen piecewise-analytic math with every non-analytic locus exactly known —
  derive/solve against the EXACT upstream operators (atlas:
  ddm_cfa1_closed_form_atlas_20260831.md) before any fit, surrogate, or sampled estimate;
  a fitted stage owes a one-line reason the closed form was not usable.
- **The frozen packet is READ-ONLY for this arm.** `/Volumes/APDataStore/pact/ddm_pq12/` is
  sealed custody — review only, zero mutations, zero new files inside it. Recommendations land
  in the memo; MAIN executes fixes after adjudication.
- **NO publication actions of any kind** (no gh, no hosting, no Drive, no email). The #1111
  one-line operator confirm (p0_1111 ledger row; publication-state section of
  `ddm_pq14_drive_pr_review_round1_20260902.md`) is the only publish gate and it is not this
  arm's to touch.
- **Persona discipline:** the Yousfi persona is a REVIEW LENS grounded in his real public
  record (contest design, PR review comments in the census, the coding-agents policy, the
  steganalysis background per the corpus). Do not fabricate quotes or attribute invented
  positions to the real person; findings say "a maintainer in his position would likely…"
  with the receipt that grounds the inference.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- `ddm_pq14_drive_pr_review_round1_20260902.md` (round 1 + 3 addenda): F13 — my report.txt fence
  was NOT verbatim (fixed against the retained T4 report); F14 — the LLM-setup disclosure bullet
  is an OPERATOR slot, not draftable by MAIN; F9 — a misattributed author citation. Lesson this
  arm inherits: packet claims drift from their receipts under iteration; re-derive, never trust
  the draft.
- `ddm_pq10` finding round (commit `562e8a287b`): the packet review found the selected-object
  swap MISSING — a whole-candidate-level defect survived four earlier doc passes. Same-lens
  reviews saturate; the maintainer lens has never been applied.
- The #1357/#1195 reading-semantics genus: four same-day slips reading numbers/labels off our
  own artifacts — every figure this review cites must be re-read from the primary file, not
  from a summary.

## OPTIMAL FORM

- Family exemplar: the packet-review-round family's landed reference form is
  `ddm_pq14_drive_pr_review_round1_20260902.md` at commit `4852b1eae7` (11 claims re-derived
  clean, findings ranked F1..Fn with per-finding owner + fix; receipt-cited throughout) — this
  charter's deliverable follows that reference structure with the NEW maintainer lens.
- SCOPE reductions declared per row. MECHANISM reductions FORBIDDEN — every file in the packet
  census gets a classification row (no sampling); every PR-body claim checked against what the
  SHIP-set alone contains.
- **PRIOR-LAW PREDICTION (falsifiable):** the lean-PR law (every winning/relevant PR in the
  intake corpus shipped only {archive + runtime + template answers + optional compress script},
  never analysis documents) predicts the SHIP-set excludes BORROWED_SUBSTRATE_ACCOUNTING.md,
  COMPRESS.md, and the review scaffolds as ATTACHMENTS (they live repo-side, linked). FALSIFIER:
  the live PR template or coding-agents policy text explicitly requesting attached documentation
  of that kind — if found, quote it and count the prediction refuted plainly.

## DELIVERABLE

`.omx/research/ddm_yr1_yousfi_adversarial_pr_review_20260902.md` — typed rows contract:
(a) §SHIP-SET table: every packet file → {SHIP-IN-PR · REPO-SIDE · INTERNAL-ONLY} + reason;
(b) §COMPARISON: winning-PR attachment sets vs ours, receipt-cited;
(c) §FINDINGS: ranked F1..Fn, each {severity · exact observation w/ file:line · concrete fix ·
owner = MAIN | OPERATOR}; (d) §MAINTAINER-WALKTHROUGH: one page of "what Yousfi sees, in order,
and where he stops"; (e) §VERDICT: finding-round outcome for the 5-pass counter + the single
highest-leverage change. Commit via the serializer. End with the own-vehicle frontier line.
