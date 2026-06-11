# Capstone adversarial-review synthesis + HONEST CORRECTIONS (2026-06-11)

**Source:** the operator-requested adversarial review (subagent a463f21d) + the optimizer/training audit
(subagent acc80882, captured in `per_step_optimizer_training_poison_audit_20260611T014320Z.md`). This memo
records the corrections both forced and the REFINED plan. Per NO-FAKE: the agent (me) OVERSTATED several
positive claims this session; this memo retracts them honestly.

## HONEST CORRECTIONS (claims I overstated — retracted)
1. **"Pose crux fix EMPIRICALLY CONFIRMED" → RETRACTED to "plausible higher-gain lever, NOT confirmed."**
   The "old shared-FiLM stuck at 0.437" was ONE bounce of an oscillating series (`capstone_real_recipe.log`
   d_pose: 0.09→0.34→0.43→0.76→0.09→0.44 — it hit 0.09 at ep70, did NOT stall at 0.437). The new per-frame
   "~0.1" ALSO oscillates 0.06–0.34. There is NO clean A/B (different pair-count + decoder-size confound).
   AND my mechanism story — "shared FiLM has ~0 Jacobian in the pose direction" — is **WRONG**: frame0/frame1
   pass through SEPARATE `rgb_0`/`rgb_1` heads, so a shared FiLM's pose differential IS learnable through the
   differing heads. Per-frame FiLM may be higher-gain, but "J≈0 for shared" is fiction. **Needs a controlled
   A/B at IDENTICAL config (same pairs/base_ch/epochs/seed), reporting mean d_pose over the last 10 evals.**
2. **"d_seg INCONCLUSIVE / could keep grinding down" → CORRECTED to "CE-recipe plateaus ~0.008."**
   The LONG run's d_seg deltas decay geometrically (−0.00204 → −0.00099 → −0.00032; ratios 0.49, 0.33) →
   asymptote ~0.008 under CE-only. It CANNOT cross the 15× to 5.6e-4 with the fixed CE recipe. The honest
   claim: "CE plateaus ~0.008; the OPEN question is whether the loss-form curriculum re-accelerates." (Still
   correctly NOT a paradigm KILL per Catalog #307 — the curriculum is untested.)
3. **Capstone status must LEAD with the recomputed S, not per-axis deltas.** Best capstone state (LONG ep40:
   d_seg 0.00838, d_pose 0.0724, 97,025 B) recomputes to **S ≈ 1.75 — 9× WORSE than the 0.191 frontier**.
   "d_pose ~0.1 = crux confirmed" hid that its pose term √(10·0.1)=1.0 is ALONE 5× the whole frontier score.
   Sub-0.15 needs d_seg≤~1e-3 AND d_pose≤~1e-4 SIMULTANEOUSLY; the capstone is 2–3 orders from both at once.
   d_pose has never touched 0.01.
4. **PROCESS FAILURE: the LONG daemon had NO marker-on-exit waiter (the session-watcher trap, GOAL_v3
   violation); 5 daemons launched in one session, first 3 produced zero epoch rows (killed on each config
   change). Substantial churn.** FIX: future long daemons write a done-marker on exit; don't relaunch 4×.

## STRATEGIC CORRECTION (the big one): the "smaller-than-frontier basis" thesis FIGHTS the measured physics
The lab's OWN floor memo (`grand_council_fields_medal_theoretical_floor_20260509.md`) states: *"To break
d_seg: ADD params — 88K → 256K → 512K… At 256K params shrink d_seg by 2× to 2.8e-4."* The frontier reached
d_seg 5.6e-4 with **162K–229K params**. The capstone is **85K (base_ch=20) — 2× SMALLER than the frontier
decoder, 3× smaller than the recommended-for-d_seg 256K.** Hoping 85K reaches the SAME d_seg the frontier
needed 2–3× more params for is motivated optimism CONTRADICTED by our own param↔d_seg curve.
**KEY REALIZATION: the byte budget does NOT require going below frontier params.** Sub-0.15 rate budget is
< 0.077 = 115,640 B. base_ch≈24 (115K params, 126,690 B) is just over; **base_ch≈22–24 (~100–115K params)
sits at the budget edge with FRONTIER-CLASS capacity.** The right capstone is NOT "smaller than frontier" —
it's "frontier-CLASS params (≈100–160K), byte-budget-fit, with the proper curriculum + per-step fixes."

## THE REFINED PLAN (supersedes "port the full 8-stage curriculum + smaller basis")
1. **Fix the 3 per-step poisons** (B's audit, integrate ON TOP of subagent A's curriculum): (#1) cosine LR
   schedule, (#2) build+export the EMA shadow @0.997, (#3) FiLM→AdamW (the Quantizr-pose auditor owns #3 + the
   pose mechanism). + #4 d_pose roundtrip measurement.
2. **Port the SEG-LOSS-FORM SCHEDULER (CE→softplus→smooth→L7), NOT the full C1a/sigma/QAT stack.** The
   reviewer's sharpest technical correction: C1a (`cat_lambda`) + sigma (`cat_sigma`) are BYTE-COST /
   quant-robustness regularizers — they reduce *bytes*, NOT d_seg. The d_seg lever past the CE plateau is the
   loss-form schedule + epochs. The 4 loss fns already exist in MLX; only the scheduler is missing (small).
   Defer C1a/sigma/QAT to the byte-close phase. (Subagent A is building the full curriculum; USE its
   loss-scheduler; defer its C1a/sigma/QAT to byte-close — do not block the d_seg probe on them.)
3. **Re-scope the basis to FRONTIER-CLASS params (base_ch≈22–24) + the factorized carrier.** The realistic
   sub-0.15 learned path (reviewer's independent read): `[seg-argmax blob] ⊕ [explicit 6-scalar GT-pose store
   + FiLM]` at ~100–160K params (NOT below frontier), each scored quantity in its own minimal representation.
4. **Process: ONE marker-on-exit daemon at base_ch≈24 with curriculum + per-step fixes + the controlled
   shared-vs-per-frame-FiLM A/B.** Only if d_seg crosses ~1e-3 AND d_pose holds ≤~1e-3 → byte-close + paired
   exact eval. Until the curriculum demonstrably re-accelerates d_seg below ~0.003, a funded CUDA port is
   spending to chase a recipe the local evidence predicts plateaus high.

## WHAT SURVIVED SCRUTINY (the reviewer confirmed — trust these)
- #79 packaging CLOSED (sound, honest negative, correct reactivation). · Lever G DEFER (sound, NOT a kill).
- Pointer honestly UNMOVED 0.19109982; all capstone numbers correctly tagged `[advisory]` `score_claim=false`;
  authority hygiene clean. · The 8 post-hoc no-moves are NOT premature kills (evidence-backed DEFERs). · The
  numpy-inflate MATH parity (d_seg |Δ|=0.0) is a legit portability gate (but "exact-eval path validated" is
  overstated — only toy/random-weight inputs, never `inflate.sh→evaluate.py` on 600 real samples).
- B's audit: Muon/AdamW partition faithful, Muon-throughout defensible, eval_roundtrip faithful in the loss
  path, ce_seg_loss bit-exact.

## NET
The capstone's apparatus is real + honestly tagged, but the headline POSITIVES were overstated (crux
"confirmed", d_seg "could grind down") and the "smaller basis → sub-0.15" thesis is contradicted by our own
measurements. The honest state: **best capstone S ≈ 1.75 (9× from frontier); the d_seg floor is a poisoned
CE-plateau (3 fixable per-step bugs + the loss-form curriculum, untested); the realistic learned path is
frontier-CLASS params + the factorized carrier, not a smaller basis.** Pointer UNMOVED; goal UNSATISFIED.
