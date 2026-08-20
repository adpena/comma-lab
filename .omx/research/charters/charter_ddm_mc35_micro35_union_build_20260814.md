# CHARTER — ddm_mc35_micro35_union_build (2026-08-14, rfo1 QUEUED route 2 — trigger MET)

CONTEXT (recall, do not re-derive). rfo1 (memo ddm_rfo1_fresh_hybrid_compose_20260814.md,
commit 6fab4cd3fc) specified `RFO1-MICRO35`, the smallest bank composition that
can clear the 1e-5 naming bar on cp135. Its fire trigger — "a distinct lane is
claimed for byte-only build and the EC2 scorer slot is released" — is MET: EC2
REFUSED (−40,779 net flips) and its lane closed terminal. The bank: qs2
(−32 flips, +34 B, archive lineage in the qs2 verdict receipts) + re1 (+2
flips-class 0 B, archive 7be3eb94) + HP4's receiver-identical five-byte repack
(−5 B) + ONE additional sign-verified neighboring exact-lattice edit + FRESH
in-compile Schur compensation (the qs5-proven mechanism — never carry a stale
compensation across objects, the qs4 lesson).

## THE WORK

1. **Recall first**: rfo1 memo §"Composed candidate: RFO1-MICRO35" (the spec IS
   there — execute it, do not redesign) + qs2/qs5/re1 verdict memos + HP4
   receipts + GT_ATTRIBUTED_DECOMPOSITION.json (B/H model).
2. **Build the ACTUAL union object**: apply all pieces to ONE archive; overlap
   between qs2 and re1 supports must be MEASURED by building, never summed
   (rfo1's stated unknown). Re-solve the Schur compensation IN-COMPILE against
   the FINAL composed edits, asserted in code.
3. **Local recount (advisory, labeled)**: decode the composed archive, recount
   flips vs the retained base field, recompute bytes. HARD GATES before any
   fire order: `F_union >= 35` net flips · `ΔB <= +29` · projected
   `Δd_pose <= 5.9739759814e-10` (rfo1's exact pose cap). If any gate fails,
   report the measured shortfall honestly — no fire order, no re-spec creep.
4. **Seal**: if gates pass, a sealed dual-axis T4 fire order for MAIN through
   the proven re1t/js1b worker chain (~$0.16), request + payload SHAs pinned,
   dispatcher owns claims (do NOT pre-claim — the claim-ordering genus).

## OPTIMAL FORM

Family reference PINS (receipts): rfo1 memo commit 6fab4cd3fc · cp135 base
sha 6eb1a3b79cb167e03372339e07e93cae13b6ba3114a9eb917288bb038622edb6 · base
instrument (34,970 flips · d_pose 6.885642960696714e-6 · 186,252 B) · exact
rate term 0.124017561736910658 · re1 admitted archive sha 7be3eb94… · qs5
in-compile compensation proof (ddm_qs5_verdict memo). MECHANISM reductions =
TOY-BRACKET: summed-not-built unions · stale compensation transfer · local
scorer as admission authority · prefix recounts. Payload law DEF CON 1000:
persist the composed archive + repeat + every intermediate with sha256+bytes.
Arms cannot reach Metal or fire Modal.

## OUTPUT

`.omx/research/ddm_mc35_micro35_union_build_20260814.md` + composed artifacts
+ sealed fire order (or the honest gate-failure report). Commit via
`tools/subagent_commit_serializer.py` (post-edit shas, `[no-triality]
[p0-ledger-ok]`, no co-author trailer). End with NEXT_IF_RESUMED +
LIVE-HYPOTHESES + DEAD-ENDS.
