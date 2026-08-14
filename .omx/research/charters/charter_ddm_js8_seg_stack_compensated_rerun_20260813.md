# CHARTER — ddm_js8_seg_stack_compensated_rerun (2026-08-13, THE LOAD-BEARING SEG LEG)

OPERATOR: "Seems like we're missing something or overlooking." FOUND — the GOAL-BACKWARDS
arithmetic: sub-0.15 from cp135 (0.16195513827824176) needs **−0.011955**; pose→ZERO buys
only **−0.008298 (69.4%)**; **seg MUST supply −0.003657 ≈ 4,314 net realized flips (at
zero byte cost) even at PERFECT pose**. Seg is LOAD-BEARING and has had NO live owner
since js7 landed 08-12. This arm revives it at the corrected form its own verdict named.

## THE STATE (recall, do not re-derive)
- **js7** (`ddm_js7_exact_row_verdict_20260812.md`): the FIRST end-to-end seg chain to a
  real T4 row — ec1 event alphabet → js6/js7 realized acceptance → joint compose →
  byte-close (+323 B, decode byte-identical, determinism repeat exact) → S 0.16342603740620176
  = +0.00147 vs cp135 (projected −0.00058). TWO exact killers, both now curable:
  (K1) POSE STACK +2.18e-6 d_pose = +0.00122 S — the 2e-6 gate was ~10× too loose
  (marginal 603 S/unit at base 6.885642960696714e-6 ⇒ stack budget ~1.3e-7); (K2) seg
  flip projection SIGN ERROR (−1,133 projected vs ≈+63 realized) — the 32-pair stratified
  panel cannot see n600 per-event receiver interactions (m96/m94). "MECHANISM survives —
  this 44-event stack is dead" (verdict_scope: instance — that stack at those budgets).
- **The qs-family cure for K1 (MEASURED, qs5 r2)**: in-compile exact-object Schur
  compensation landed d_pose BELOW base (6.88501e-6 vs 6.88564e-6, deterministic repeat)
  — frame-1 edits carry ~ZERO pose tax when compensation is solved per-object in-compile
  with the fail-closed anti-stale guard (qs1 compiler :963 pattern). Per-event
  compensation converts K1 from a budget-juggling problem into a solved mechanism.
- **The realization laws for K2 (MEASURED, qs3/qs4/qs5)**: collateral-not-washout (97.4%
  of edits realize; loss = H flips of previously-correct pixels, H ≈ 0.7×B under strict
  support); benefit model B is EXACT (qs4 B=57 model-exact); the 17-flip ceiling is
  formulation-scoped to the 3-pair micro support — js7's events are a DIFFERENT, larger
  class. Breakeven 0.785 flips/B.

## THE WORK (optimal form — mechanism floors, not aspirations)
1. **Per-event realized calibration at n600 advisory (LOCAL Metal, $0)**: re-screen the
   ec1/js6 event alphabet ON the cp135 base with per-event REALIZED n600-advisory flips
   (through the real receiver + R + uint8 + frozen CPU scorers) — never the 32-pair
   panel as composition authority (js7 K2). Retain every per-event field (payload law).
2. **Per-event in-compile compensation (the qs5 mechanism, generalized)**: each accepted
   event's frame-0 compensation solved against ITS OWN realized perturbation, content-
   bound, fail-closed against stale reuse. Pose stack budget target: ≤1.3e-7 d_pose
   TOTAL (the derived bar), with the qs5 evidence that per-object solves can land ~0.
3. **Composition under JOINT n600 remeasure**: greedy-under-joint-remeasure (the v19b
   pattern) over the calibrated events — accept only realized joint ΔS<0 at each step;
   NO projected-sum acceptance (js7 K2 extinct structurally).
4. **Byte-close + sealed dual-axis fire-order** for any composed candidate with
   n600-advisory net ≤ −5e-4 class (the seg leg's scale — NOT the 1e-5 micro class);
   worker self-claims; pose-placeholder emitted structurally (0.0 hardcoded in the
   request builder); MAIN fires (~$0.16). Below that scale: report the measured
   per-event reach table — the curve is the deliverable.
5. **Honest ceiling**: if the calibrated alphabet's total realized reach < ~1,000 net
   flips, say so with the per-event table — that measured ceiling re-routes the seg leg
   to implicit-conditioning/trained-receiver (fd135 OPEN row · #982 · #978), not to more
   event iterations.

## OPTIMAL FORM
Family reference = js7's chain (provenance-pin the verdict memo + its byte-close
receipts) + qs5's compiler (pin path+sha). SCOPE reductions legal + declared; MECHANISM
reductions = TOY-BRACKET (cannot produce family verdicts; the fire gate refuses
toy-bound stores). Model-derived candidates declare `model_derived` +
`generalization_gate{passed}` in store JSONs. No naive or toy ever.

## OUTPUT
`.omx/research/ddm_js8_seg_stack_compensated_rerun_20260813.md` + code/tests + retained
store `/Volumes/VertigoDataTier/pact/ddm_js8_20260813/` + sealed (no-)fire-order.
Commit via `tools/subagent_commit_serializer.py` (post-edit shas, `[no-triality]
[p0-ledger-ok]`, no co-author trailer; git-blocked → declare memo SHA for MAIN handoff).
End with NEXT_IF_RESUMED + LIVE-HYPOTHESES + DEAD-ENDS.
