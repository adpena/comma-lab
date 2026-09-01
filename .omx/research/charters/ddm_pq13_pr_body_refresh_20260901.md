# ddm_pq13_pr_body_refresh — PR watch + refresh OUR PR body against the LATEST contest requirements + Yousfi's feedback/policy (operator 2026-09-01 "Check for any new PRs as well and let's refresh and update our PR body against the latest requirements and comments and feedback yousfi has provided"): PR #139 + closed-PR intake · live-repo requirements/policy diff vs our pinned snapshot · Yousfi comment census · PR_BODY_DRAFT refresh for OPERATOR sign-off · compliance-delta rows (tasks #1111 / #1363 / #1156)

## MANDATE

The operator asked for a PR check and a PR-body refresh. MAIN's fresh census (2026-09-01):
ONE new open PR — **#139 "Reproduce rhnerv_comma 0.19 score on Windows" (DarkPsionics808,
opened 09-01)**, a reproduction PR at 0.19 (no threat to the 0.162 PR135 bar) — plus #137
(metric_shift_av1) and #138 (opal_v1) CLOSED 08-31, unskimmed. Separately, #1363 records
that Yousfi added AND enforced a coding-agents/LLM policy (~08-31) that bears directly on
the #1111 submission; our frozen gen-7 packet's PR body draft
(`.omx/research/ddm_pq1_submission_packet_prep_20260815/PR_BODY_DRAFT.md`) predates BOTH
that policy AND the afr1 re-swap. This arm brings the whole public-interface surface
current and hands the operator a sign-off-ready draft. NOTHING IS PUBLISHED.

## SCOPE

1. **PR watch intake (light, typed).** #139: verify it is a reproduction PR (score, method,
   any technique signal owed to us — likely none; a Windows-reproduction row may still carry
   runtime-portability signal for our own inflate path). #137/#138 closed skim: what were
   they, why closed, any technique/feedback signal. Full comment census on ALL PRs updated
   since 2026-08-10 (gh api, paginated) — capture EVERY comment authored by Yousfi (and any
   maintainer) verbatim into a typed feedback ledger {pr · date · verbatim · what it demands
   of a submitter · does it bind OUR packet}.
2. **Requirements/policy diff AT SOURCE.** Fetch the LIVE contest repo's README + any
   policy/rules files (gh api / raw.githubusercontent) and diff against our pinned
   `upstream/` snapshot (READ-ONLY — report drift, NEVER edit the snapshot). The
   coding-agents/LLM policy (#1363): find its authoritative text (README section, discussion,
   pinned issue, or PR comment where Yousfi enforced it), quote it verbatim, and state
   plainly what it requires of an agent-assisted submission (disclosure? authorship
   statement? restrictions?). This is the single highest-stakes deliverable — the #1111
   submission decision consumes it.
3. **PR body refresh (PREPARE-ONLY, operator signs).** Rewrite PR_BODY_DRAFT.md v2 against:
   the gen-7 afr1 packet facts (S 0.14797617125559104 @ 180,002 B [contest-CUDA T4 n600],
   archive sha cbb8d928…d405bf25; CPU row 0.20513189 on predecessor bytes — carry the axis
   honestly) · the NO-FAKE #7 borrowed-substrate accounting brought current to the afr1
   lineage · whatever the new policy demands · the patterns in Yousfi's comments on OTHER
   PRs (what he asks submitters for: reproduction commands, hosted archive, honest axis
   labels, decode budget). Keep the operator-authorship contract: the draft is a PROPOSAL;
   final text authorship + the publish act stay with the operator (#1111/#1363).
4. **Compliance delta rows.** Does the new policy (or any README drift) change the packet's
   80-GREEN/7-RED compliance table? Typed rows {requirement · source quote · packet status ·
   action needed}; route any new RED to a named owner.

## HARD CONSTRAINTS

- **NOTHING PUBLISHED**: no PR opened/edited on GitHub, no comments posted, no reactions —
  read-only API access. The #1111 operator one-line confirm remains the only publish gate.
- `upstream/` READ-ONLY — requirements drift is REPORTED, never applied to the snapshot.
- **NO Claude/AI attribution anywhere in the draft**; public hygiene binding (no fleet IPs,
  no local absolute paths, no provider ledger contents in any drafted text).
- Verbatim quotes for every requirement/feedback row — the #1357 reading-semantics genus
  forbids paraphrase-derived compliance claims.
- Axis honesty in the draft: scores from components, [contest-CUDA]/[contest-CPU] labels
  inline, no rounded-display citation (#877).
- Serializer commits w/ post-edit `--expected-content-sha256`; bundle-fallback (#1293).
- NO-OVERLAP: lv3/dd1/ccs1/sp978 scopes untouched; packet custody frozen (draft file + memo
  edits only — the archive and runtime tree are NOT touched).

## PRIOR NEGATIVE SIGNAL

- pq1's GitHub comment-census retry was owed (API unreachable that day) — this arm pays it.
- The #1156 pq5/pq6 review-persona work exists — consume its findings, do not re-derive.
- sw1/sw2 scrub lessons: drafted public text has leaked private surfaces before — run the
  public-hygiene scan on the NEW draft (the #1112 vacuous-scanner lesson: scan the DRAFT
  file itself, not just README).
- m07 SHIP LAW honesty-half: borrowed-lineage accounting must be plain in the body.

## OPTIMAL FORM

- Family exemplars / receipts: the pq-family packet-polish reference form — **pq12's afr1
  re-swap receipt** (memo `.omx/research/ddm_pq12_afr1_reswap_20260831.md`, SHA
  `0bfbb0042fcf5a1364fefb64977b7fbe9ca157918625323489b151a506872b32`) + the pq1 packet-prep
  receipt bundle's existing draft
  (`.omx/research/ddm_pq1_submission_packet_prep_20260815/PR_BODY_DRAFT.md`, SHA
  `90627a67c353949329c03ff1c11c86e86287d53de5bf07d469eda1e7f67f5a40`) — the object under
  refresh. Provenance pins: afr1 archive SHA
  `cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25` · task rows #1111/#1363.
- SCOPE reductions legal: the comment census may cap at PRs updated since 08-10 (state the
  cap). MECHANISM reductions FORBIDDEN: verbatim policy text · verbatim Yousfi quotes ·
  the draft's public-hygiene scan · compliance-delta typed rows.
- **PRIOR-LAW PREDICTION (falsifiable):** the coding-agents policy will require explicit
  DISCLOSURE of agent involvement rather than prohibit it (contests generally regulate
  disclosure; Yousfi enforced it on someone else's PR, which implies a compliance path
  exists). FALSIFIER: the policy text prohibits agent-assisted submissions outright — then
  the #1111 decision changes character entirely; surface to the operator VERBATIM as the
  memo's first line either way.

## DELIVERABLE

`.omx/research/ddm_pq13_pr_body_refresh_verdict_20260901.md` — the PR intake rows (#139 +
#137/#138) · the Yousfi feedback ledger (verbatim) · the policy text VERBATIM + what it
demands · the requirements drift table (live repo vs pinned snapshot) · PR_BODY_DRAFT v2
(new file, old draft untouched) + a one-screen operator sign-off summary of what changed
and why · the compliance-delta rows · DEAD-ENDS + denominators (PRs examined / comments
captured). Commit via serializer. End with the own-vehicle frontier line
(S 0.14797617125559104 @ 180,002 B [contest-CUDA T4 n600], afr1 sha cbb8d928…d405bf25 —
UNMOVED).
