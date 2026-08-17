# ddm_pq2 — FINISH + POLISH OUR PR PACKET, incl. the END-TO-END COMPRESSION SCRIPT

**Operator binding 2026-08-17, verbatim: "Let's check for any new PRs and finish and polish ours
including the end to end compression script."** MAIN did the PR sweep (below); this arm brings the
pq1 submission packet to READY-except-OPERATOR-GO on the CURRENT frontier candidate, builds the
sanitized e2e compression script, and intakes the two new public PRs. URGENT: PR 138 claims exact
0.1591495384 on our own axis — our unsubmitted 0.15853325034789678 leads by only 6.2e-4.

Standing laws: NO submission (OPERATOR-GO only, pq1 HOLD stands) · NO Modal/paid from this arm
(prepare fire-orders; MAIN fires via tools/fire_modal_auth_eval.py) · ALWAYS KEEP THE PAYLOAD ·
disclosure hygiene (NO fleet IPs, private infra URLs, local absolute paths, provider logs in any
packet file) · serializer commits w/ POST-EDIT sha · .py = 2 review passes · upstream/ READ-ONLY ·
payloads → /Volumes/APDataStore/pact/ddm_pq2/.

## The state (recall; verify at source)

- **Packet**: pq1 bundle dec5402577, 90/90 files sha-verified on Vertigo (manifest df001d743b42…),
  compliance 78/86, PR draft + NO-FAKE #7 borrowed-substrate accounting + etiquette + SWAP
  PROCEDURE + 0/5 review scaffold. Built pre-rr4 — re-target via the swap procedure.
- **The candidate**: the rr4 frontier archive — S 0.15853325034789678 @ 181,161 B [contest-CUDA T4
  n600], sha 35ac2b9beb7e6fa81075c7d84b5247d8d24c056fe49ce1cbd22a334bc9618956, runtime tree at
  /Volumes/APDataStore/pact/ddm_rr4_cuda_prob_reencode/candidate_runtime/ (GENERATION_RECEIPT
  chain; rr2 receipts prove deterministic re-encode byte-identity + CPU-inflate identity e5539653).
- **Field intake (MAIN sweep 2026-08-17 ~19:4xZ)**: PR 138 "opal_v1" (claims 0.1591495384;
  182,040 B, sha bd9a4714…; online-adaptive prior on PR135's frozen tokens; GPU-eval declared;
  compression script included-for-merge) · PR 137 "metric_shift_av1" (pose 0.0788 → S ≫ 0.16,
  non-competitive) · PR 136 hnerv_rc 0.19258 CPU. Leaderboard best still PR135 0.162 (neither new
  PR evaluated yet).

## Deliverables

1. **Packet re-target**: execute pq1's swap procedure onto the rr4 candidate (archive + shas +
   report numbers + PR-draft body updated to S 0.1585…/181,161 B; the NO-FAKE #7 accounting
   updated for the rr4 recode lineage — inherited-PR130/135-class HPAC substrate vs OUR original
   mechanisms itemized honestly). GPU-eval declaration modeled on PR135/138 precedent ("requires
   GPU for evaluation") — the field-norm answer to the CUDA-locked F26 decode.
2. **THE E2E COMPRESSION SCRIPT** (the operator-named deliverable; PR 138 sets the field norm of
   including it for merge): ONE sanitized entry point that reproduces the archive:
   stage A (documented, optional to run: training configs + exact commands that produced the
   checkpoint — days of compute, honestly labeled); stage B (EXACT + verifiable: checkpoint →
   token emission → HPAC/F26 re-encode → archive.zip asserting sha 35ac2b9b… byte-identity —
   deterministic per the rr2/rr4 receipts); stage C (decode = the shipped inflate path).
   Seeded, deterministic, no private paths, runnable by a judge; smoke it end-to-end from the
   retained checkpoint and record the sha assertion passing.
3. **Dep-closure verification**: the rr4 runtime already carries the rr3 constriction declared-dep
   + fail-closed self-install (e4 precedent = the MAIN-adjudicated policy). VERIFY at source on
   the candidate tree + one bare-venv bootstrap smoke receipt.
4. **CPU-axis fire-order PREP** (do not fire): exact argparse-true invocation for a contest-CPU
   row on the rr4 bytes (f26p CPU-unlock receipts govern feasibility; note the #1054 precedent
   CPU row cost ~$0.40). MAIN adjudicates whether the field-norm GPU-eval declaration makes this
   optional for submission; prepare it regardless.
5. **PR 137/138 intake**: download both archives + PR source per the standing public-intake
   discipline (PR138: bd9a4714…, 182,040 B; record shas, bit-level anatomy vs PR135's sections,
   opal token-section delta confirmation 114,706→110,022 B). The opal MECHANISM deep-read routes
   to ddm_me1 (already relayed the body; your anatomy adds the byte-level receipts). Detached
   clones only; never checkout into the shared worktree.
6. **Red burn-down**: the 3 historical lane-ledger binding defects (hygiene) + the GitHub
   comment-census retry (API was unreachable at pq1 time — retry now) + hosted-manifest PREP
   (exact artifact list + shas for operator hosting authorization; do NOT publish anything).
7. **Review scaffold pass 1**: after the re-target freeze, run the first of the 5 adversarial
   passes yourself (fresh-eyes rules; findings reset the counter).
8. Memo `.omx/research/ddm_pq2_packet_polish_20260817.md` + arm final message: packet state
   (READY-except-GO or the exact blockers), e2e-script smoke receipt, intake headlines, the
   prepared CPU fire-order, NEXT_IF_RESUMED. STORES CONSULTED per section. End with the
   own-vehicle frontier line and whether your unit moved it (it will not — say so).

## Falsifier honesty

If the swap procedure surfaces a custody gap on the rr4 candidate (e.g. the stale
RECEIVER_PARSEBACK.json the rv2 review is adjudicating), STOP the re-target, report the gap, and
fall back to preparing both candidates (rr4 + hv1 ep0634) so the packet can freeze on whichever
the review clears.
