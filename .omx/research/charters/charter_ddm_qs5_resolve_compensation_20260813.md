# CHARTER — ddm_qs5_resolve_compensation (2026-08-13, from the qs4 REFUSED verdict)

THE QS4 SUPER-BAND CANDIDATE REFUSED AT +2.437870e-4 AND THE MECHANISM IS EXACT:
seg −1.441e-5 (−17 flips, real win) · rate +1.864e-5 (+28 B, as designed) · **pose
+2.396e-4 — the Schur frame-0 compensation was NOT re-solved for the TRIMMED frame-1
edits** (d_pose 6.885643e-6 → 7.288944e-6, deterministic repeat identical; receipt
`/Volumes/VertigoDataTier/pact/ddm_qs4_20260813/dispatch/ddm_qs4_dual_axis_20260813_r1/QS4_T4_REMOTE_RESULT.json`).
qs2 proved the compensation works when solved FOR ITS OBJECT (+1.126e-7 S leakage);
qs4 carried qs2-lineage compensation onto a changed perturbation — the cross-regime
constant-transfer genus. The cure is narrow and local.

**PRIOR-LAW PREDICTION (m38):** with the compensation RE-SOLVED against the trimmed
dx1 (the qs1 Schur machinery, J_pose,0·dc0 ≈ −J_pose,1·dx1), pose leakage returns to
the ~1e-7 S class (qs2 evidence) ⇒ net ≈ −1.44e-5 + 1.9e-5 + 1e-7 ≈ **+4.5e-6 — STILL
REFUSED at current seg realization.** So compensation repair ALONE is insufficient:
the trimmed object under-realized seg (−17 from 100 changed vs −57 modeled). BOTH legs
must move: (a) re-solved compensation AND (b) seg value recovered — either partial
de-trimming (keep neighbor-cell trims, restore same-cell benefit the trim destroyed)
or adding re-screened bank pairs under the corrected B/H model. Target: net ≤ −1e-5
verified. If the B/H decomposition of the qs4 field shows benefit/collateral
INSEPARABLE below the needed ratio, report that measured ceiling plainly — that closes
the strict-support formulation honestly.

## THE WORK
1. **$0 decomposition of the qs4 candidate field** (retained on volume
   `comma-ddm-js1b-argmax-retained/ddm_qs4_dual_axis_20260813_r1`; MAIN can pull if
   codex cannot reach Modal — emit the exact recovery command as rung 0 and CONTINUE
   with local work; the qs1/qs3 GT+base fields are already local): B/H/W for the 100
   changed pixels → where did the modeled −57 lose 40 flips? (trim removed benefit vs
   collateral appeared elsewhere).
2. **Re-solve the Schur compensation** for the exact trimmed dx1 (qs1 machinery,
   provenance-pinned; per-pair dc0; verify cancellation locally via the js1 local pose
   model to the extent it is calibrated — label advisory).
3. **Recover seg value**: compose the compensation-repaired strict-support object with
   either (a) restored same-cell edits whose collateral is priced affordable under the
   corrected model, or (b) top re-screened bank pairs (the 57.1% B-rate prior + qs4's
   full_bank_screen.jsonl). Waterfill to projected net ≤ −1.5e-5 with pose budgeted at
   2× the qs2 leakage class (not zero — derived margin, not optimism).
4. **Byte-close through HP3/RC64** (count every byte; retain everything) + **sealed
   dual-axis fire-order** (worker unchanged, self-claims; MAIN fires ~$0.16). Admission
   at harvest: net realized ΔS < 0 matched-instruments; super-band gate |ΔS| ≥ 1e-5
   for the canonical-row naming.
5. Fold the LESSON into the engine: the compensation solver runs INSIDE the compile for
   ANY edit-object change — never carried across objects (assert in code, not memo).

## OPTIMAL FORM
Narrow repair + recompose on PROVEN machinery (qs1 solver, qs2 coder at 4.0 B/active
pair, qs4 collateral map + bank screen — all provenance-pinned in their stores).
Instrument unchanged (mechanism change = TOY-BRACKET). Payload law binds.

## OUTPUT
`.omx/research/ddm_qs5_resolve_compensation_20260813.md` + code/tests + retained store
(`/Volumes/VertigoDataTier/pact/ddm_qs5_20260813/`) + sealed (no-)fire-order. Commit
via `tools/subagent_commit_serializer.py` (post-edit shas, `[no-triality]
[p0-ledger-ok]`, no co-author trailer). End with NEXT_IF_RESUMED + LIVE-HYPOTHESES +
DEAD-ENDS.
