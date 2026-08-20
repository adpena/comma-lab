# ddm_pq1_submission_packet_prep — prepare (NOT submit) the frontier PR packet, candidate-agnostic

## OPERATOR CONTEXT (2026-08-15, verbatim): "In the meantime, should we prepare a pull request for the current frontier?"
MAIN's adjudication: YES prepare / HOLD fire. Build the complete submission packet against the
CURRENT frontier archive as the ready fallback; the e960 burn endpoint (~5h) likely produces a
better composed candidate — the packet must be SWAPPABLE (candidate-agnostic machinery + a typed
swap procedure). ACTUAL PR SUBMISSION IS OPERATOR-GO ONLY. This arm never submits, never pushes
to any public remote, never hosts anything externally.

## THE OBJECT (locate by SHA, then pin the path in your memo)
Frontier archive: e480b v2, S 0.1600920261571558 @ 183,502 B [contest-CUDA T4 n600],
archive sha256 e3e6f440b45bbb92f2eeb58c7a56d74b3cd0a62bbcff01a26adcd008391c19d3, components
seg 0.029611 + pose 0.0082946 + rate 0.1221864. Custody: locate under the retained tiers
(/Volumes/VertigoDataTier/pact/ + /Volumes/APDataStore/pact/ — start from the mz1 memo
.omx/research/ddm_mz1_model_section_rate_race_20260815.md FINAL_RESULT custody and the
rx2/mc36 identity-race receipts, experiments/ddm_rx2_mc36_identity_race.py conventions).
VERIFY the sha before any packet step; refuse on mismatch.

## RECALL FIRST (charter violation to rebuild these)
- CLAUDE.md gates: "Submission auth eval — BOTH CPU AND CUDA" · "Submission PR gate" (5
  consecutive clean-pass adversarial review) · "Public Disclosure Hygiene" (contest CLOSED,
  IP open-source; sanitize infra/paths; PR user-attributed, NO Claude attribution) ·
  docs/submission_template.md · scripts/pre_submission_compliance_check.py (--contest-final
  --strict contract) · the 2026-05-19 DEFER blocker report (hosted-URL + report.txt gaps).
- tools/create_fork_pr_for_submission.py is for self-eval fork PRs, NOT contest submission.
- Lineage receipts for the accounting: pi1 (PR130 intake) · fd135 (PR135 decomposition) ·
  mz1 (section shas: semantic b0d41ec904aca82f… / carrier 065fce08…, byte-identical to the
  PR130 reproduce) · hb1/rx2 (HPAC retrained on OUR MC36 labels) · micro-edit campaign rows
  (MC36 promotion #1049, e480b v2) · f26p (F26 CPU-unlock port) · #1054 (CPU row 0.20513189
  on the MC36 predecessor bytes) · rr3/#1008 (constriction dep declared, bootstrap smoke).
- Dependency-closure law: fail-closed self-install (e4 brotli precedent), bare-venv proof.

## DELIVERABLES (all $0 local; each with receipts)
(1) COMPLIANCE-CHAIN RUN on the e480b v2 packet: assemble the submission_dir (archive.zip +
    inflate.sh + inflate.py + README per docs/submission_template.md) from retained custody;
    run scripts/pre_submission_compliance_check.py --contest-final --strict with expected
    sha+size; emit a typed GAP REPORT for every red item (hosted URL, report.txt, auth-eval
    JSON linkage, runtime-tree custody) — close what is closable locally, name the rest.
(2) BORROWED-SUBSTRATE ACCOUNTING (NO-FAKE #7, the honesty half): itemized table — per
    section/mechanism: {ours-original | PR130-lineage-retrained-on-our-labels |
    PR130/135-byte-identical}, with sha receipts. This table goes IN the PR body draft.
(3) PR BODY DRAFT: sanitized (no Tailscale IPs, no local absolute paths, no provider
    transcripts), operator-attributed, NO Claude/AI attribution anywhere; states both axes
    honestly (CUDA 0.1600920 T4 n600; CPU row pending on exact bytes — the f26p port makes
    our packet CPU-runnable, which the PR130/135 lineage is not); includes the accounting
    table + decode-budget receipts (CPU decode 831.5 s on predecessor bytes, 2.17× headroom).
(4) CPU-AXIS SEALED FIRE-ORDER (do NOT dispatch): Modal contest-CPU n600 eval of the EXACT
    e480b v2 bytes through the f26p-ported runtime — sealed config + hash manifest + cost
    estimate (~$0.15), MAIN fires. Mirror the proven r4/js1c dispatch conventions + the
    canonical tools/modal_harvest_poller.py close.
(5) SWAP PROCEDURE + REVIEW SCAFFOLD: a typed checklist that re-targets the packet to the
    e960-composed candidate at the endpoint (re-run (1), re-sha, delta-review) + the 5-pass
    adversarial-review scaffold (round log, counter resets on any finding) initialized at 0/5.

## OPERATOR ADDENDUM (2026-08-15, mid-flight, verbatim): "When preparing the PR we need to ensure
## we follow all appropriate conventions and best practices and include all required statements
## and include our compression scripts in a friendly manner also look at the closed PRs and those
## with yousfi comments for feedback about how to contribute correctly and respectfully"
This adds deliverable (6) and expands (3). READ-ONLY access to the public contest repo
(commaai/comma_video_compression_challenge) via gh/API is GRANTED for this — reading is not
submission; the no-push/no-submit constraint is unchanged.
(6) CONTRIBUTION-ETIQUETTE HARVEST: mine the contest repo's actual contribution surface —
    README submission instructions, any PULL_REQUEST_TEMPLATE/CONTRIBUTING, and the PR corpus:
    (a) CLOSED/rejected PRs — why were they closed (non-compliance, missing report.txt, wrong
    format, hosting problems, spammy resubmission)? (b) every PR carrying Yousfi (maintainer)
    comments — harvest his feedback verbatim as the authoritative how-to-contribute signal;
    (c) the accepted-winner convention corpus (PR95/100/101/130/135 body structure, score-claim
    format, report.txt, archive hosting norm). OUTPUT: a DO/DON'T etiquette table with PR-number
    + quote receipts, folded INTO the PR body draft and the submission runbook.
(3-EXPANDED) The PR body draft must additionally: carry ALL required statements the repo's
    instructions/template demand (compliance, reproducibility, dependency declarations, license
    posture — derive the list from the repo, don't guess); include our COMPRESSION-SIDE scripts
    in the friendly community norm (PR95 published its full training stack — that is the bar):
    sanitized, self-contained, seeded/reproducible, documented with a readable README, no fleet/
    infra references; credit prior work courteously (PR130/135 attribution is both NO-FAKE #7
    accounting AND community courtesy); respectful tone matching the repo's culture; ONE PR,
    never serial spam.

## HARD CONSTRAINTS
- NO submission, NO public push, NO external hosting, NO Modal dispatch (sealed fire-orders
  only; MAIN fires). The LIVE e960 burn (pid 47772) + watchers are SACRED — read-only.
- ALWAYS KEEP THE PAYLOAD: assembled packet + every intermediate to
  /Volumes/VertigoDataTier/pact/ddm_pq1_submission_packet/ with sha256 manifest.
- Serializer commits w/ post-edit SHAs (tools/commit_autosha.sh), [no-triality]
  [p0-ledger-ok]; .py = 2 review passes; memo
  .omx/research/ddm_pq1_submission_packet_prep_20260815.md w/ DEAD-ENDS + LIVE-HYPOTHESES;
  end with the own-vehicle frontier line.

## OPTIMAL FORM
- Family reference: the 2026-05-19 submission-DEFER packet work (the named prerequisite list
  IS this charter's gap list) + ic1/e4/e5a byte-close→export conventions + t1r1 container
  rehearsal. This is the packet member of that family at landed form — not a sketch.
- SCOPE reduction (legal): ONE candidate (e480b v2) + a swap procedure; no speculative
  multi-candidate packets. MECHANISM reductions: NONE — a compliance "pass" without actually
  executing the checker, or a hand-asserted sha, is inadmissible.
- Provenance pins: archive sha e3e6f440b45bbb92… (verify at source) · mz1 memo + section
  shas above · #1054 CPU-row receipt · f26p landing · docs/submission_template.md ·
  scripts/pre_submission_compliance_check.py.
