# CHARTER — ddm_pr135ps_truncated_search_resume (2026-08-13)

RESUME PR135'S TRUNCATED COMPENSATION SEARCH on the cp135 base — the named pose-axis
opening. Recalled fact (fd135/pi135 receipts; verify at source, pin path+sha): PR135's
pose mechanism is the PR133 quantize-then-COMPENSATE int12 Jacobian re-solve, and **their
8-pass search TRUNCATED while STILL ACCEPTING** — the authors stopped their own descent
early. The off-the-shelf grant (operator 08-10) covers their code and full corpus.
Waterfall: pose contributes 0.0082946 S of the cp135 floor (d_pose 6.885642960696714e-6)
— the single largest bar-beat term; pose→0 ceiling −0.0083. Scorer-free local solve; NO
Modal fire (MAIN fires); sealed fire-order output.

**OPERATOR DOCTRINES BINDING:** "no naive or toy or generic basis ever" (resume THEIR
exact algorithm at THEIR truncation point — never a reinvented search; extensions beyond
their form are declared as such) · byte-closed-row cadence (output = candidate archive(s)
+ sealed fire-order) · "as much as possible locally" (the solve is encode-side compute on
archive contents; only the CUDA-locked decode + scorer verification leaves the Mac).

## OPTIMAL FORM
Solve-resume arm. Reference form = PR135/PR133's OWN compensation algorithm (their code,
provenance-pinned from the fd135 decomposition store). SCOPE extensions allowed: more
passes, better acceptance bookkeeping. MECHANISM changes (different objective, different
Jacobian) = declared TOY-BRACKET, separate rows, never conflated with the resume.

## THE WORK
1. **Recall + pin**: locate the fd135/pi135 receipts documenting the truncation
   ("8-pass search TRUNCATED still-accepting") + PR135's compensation code + the exact
   int12 sections in the cp135 base archive (186,252 B — pin the LIVE base sha from the
   cp135 receipt, do not assume). Verify the still-accepting claim AT SOURCE (their
   logs/code, not our memo).
2. **Resume the search locally**: continue their acceptance loop past pass 8 with an
   extended budget until convergence (acceptance rate → 0) or a derived stopping rule
   (cite #344/trajectory laws; no arbitrary cap — the cap-censoring lesson, #874).
   Track the accepted-compensation trajectory: per pass, projected Δd_pose (their own
   surrogate) + Δbytes (compensation is value-replacement in existing int12 sections —
   verify Δbytes=0 claim honestly; if the coded size shifts, count it).
3. **THE FAMILY LAW BINDS IN REVERSE** (`.omx/research/ddm_re1_round1_full_auth_row_
   20260813.md`): pose edits leak into seg through the SAME shared D. NO single-axis
   admission. The verification gate must measure BOTH axes.
4. **Output**: candidate archive(s) at the convergence point (+ intermediate checkpoints
   per the payload law — retain every accepted-pass archive) + adapted runtime (po1 pin
   lesson) + sealed DUAL-AXIS fire-order (the js6b-extended worker: seg field + PoseNet
   6-vectors in one dispatch; if js6b's extension has not landed yet, seal against the
   re1t worker + a pose-vector follow-up and note the dependency). BATCHING clause: MAIN
   may batch this gate with js6b's compile gate — write the fire-order so candidates
   compose into one dispatch where the worker supports it. Fresh run-id, ~$0.16, #381
   (~$2.7 spent of $20).

## OUTPUT
`.omx/research/ddm_pr135ps_truncated_search_resume_20260813.md` + code/tests + retained
store + the sealed fire-order. Commit via `tools/subagent_commit_serializer.py`
(post-edit shas, `[no-triality] [p0-ledger-ok]`). End with NEXT_IF_RESUMED +
LIVE-HYPOTHESES + DEAD-ENDS.
