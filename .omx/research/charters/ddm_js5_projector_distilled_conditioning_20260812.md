# ddm_js5 — projector-DISTILLED conditioning (js4 FOLLOW_ON action-2 fire; trigger MET)

Successor of ddm_js4 per its FOLLOW_ON.json action-2 route (fire trigger MET:
F1 not fired — 64 robust beneficial flips, projected n600 robust −305 under
full linear projection). js4's TWO walls define this arm's whole mission:

- **F3 (receiver wall, FIRED):** decode-time projection needs the 452,988,928 B
  basis (or PoseNet at inflate — FORBIDDEN by the strict scorer rule). Cure:
  the projector becomes a TRAINING-TIME-ONLY operator; ship the module alone
  (744 B class). The receiver runs the module bare — so the MODULE ITSELF must
  emit (approximately) pose-null corrections. That is what "distilled" means.
- **F2-realized (nonlinear leakage, measured):** continuous leakage 8.836e-4 =
  442× the 2e-6 gate with first-order exactly nulled; quantization SHRANK it
  (−2.6e-5) — CVP is not the cure, NONLINEARITY is the enemy. #532's quantum
  bound is subsumed.

## THE DESIGN (race the cures inside ONE trainer, cheapest measurement first)

Extend the js3 trainer (eb450d1281) + js4 projector (measured source sha
8989a846…) with, in order:

1. **Leakage-vs-amplitude curve ($0, first):** scale the js4 step-25 correction
   by α ∈ {1, 1/2, 1/4, 1/8, 1/16} and measure realized pose leakage + robust
   flips at each α on the stratified n32. Pre-registered law: first-order-nulled
   leakage ~ α² (quadratic); robust flips ~ α-ish (locally). This ONE curve
   decides whether amplitude control alone can buy gate-passing corrections and
   at what seg cost. PERSIST the curve (payload law).
2. **Realized-acceptance training (the v19/j5 law, our proven mechanism):**
   projected proposal step → REALIZED pose check through the real chain
   (receiver→R→uint8→custody PoseNet planes) → accept only if pose delta stays
   under a per-step budget derived from the 2e-6 endpoint gate; reject-and-
   shrink on violation. The projector makes most proposals cheap to accept
   (first-order nulled); realized acceptance prices the second order EXACTLY —
   no linearization trust anywhere in the accept path.
3. **Periodic re-linearization (v16/v17 validity-radius law):** recompute J_p
   at the current corrected point every K accepted steps (K from the measured
   α-curve validity radius); the projector tracks the operating point.
4. **Distillation closure:** the SHIPPED module is trained under 1–3; final
   admission check runs the module BARE (no projection at eval) on held-out
   stratified n32 — the module's own outputs must pass the realized pose gate.
   If bare-module leakage exceeds gate while projected-training passed, add the
   distillation loss term ‖c_theta − P c_theta‖² (teach the module the
   projection) and re-measure. Bare-module pass is the ONLY admission that
   clears F3.

## BINDING LAWS (inherited js3 verbatim + specific)

- δ-hinge objective, δ=0.08036041259765625; robust flips only; relative gauge
  baseline 50,389 @ batch16/8-threads; [macOS-CPU advisory, instrument floor
  0.0131 S]; pose on custody planes, endpoint gate 2e-6.
- ECONOMICS BAR (sharpened from js4 F3 arithmetic): break-even for the full
  seg debt is 6,007 B for −4,700 n600 robust flips ⇒ ≤1.28 B/robust-flip.
  js4 measured 2.44 B/flip at hidden-4/25-steps. The capacity ladder
  (hidden 4/8/12) + longer projected training must show the RATE OF IMPROVEMENT
  toward 1.28; report B/flip per rung per step-count. T4 admission row stays:
  n600-projected robust ≤ −2,000 at ≤ +1,500 B (module bare).
- Trainer resumable, per-stage ckpts P0; sealed MAIN burn recipe; DO NOT launch
  the long burn. Bounded smokes ≤ ~40 min.
- Payload law P0: every checkpoint/curve/module persisted sha256+bytes to
  /Volumes/VertigoDataTier/pact/ddm_js5_20260812/.
- Serializer --no-co-author, post-edit shas, tags [no-triality] [p0-ledger-ok];
  2 review passes per .py; blocked-git → commit_intent; skeleton annexes queued
  to run dir (MAIN-owned).

## OPTIMAL FORM

- Reference: js3 trainer + js4 projector measured source (both landed; REUSE).
  Provenance pins: js4 FINAL_RESULT sha 21bb694d…, projector measurement sha
  a805fc00…, δ at js2b FINAL_RESULT, cp135 archive sha 6eb1a3b7….
- SCOPE reductions legal: n32 stratified screens, bounded steps, ladder tiny-first.
- MECHANISM reductions TOY-BRACKETED: skipping realized-acceptance (linearized
  accept), skipping the bare-module admission check, entropy-estimate bytes —
  none can produce a family verdict.

## FALSIFIERS

- F1: α-curve shows no α with (leakage ≤ per-step budget ∧ robust movement > 0)
  AND realized-acceptance acceptance-rate < 5% over ≥200 proposals → the
  pose-null seg-reachable overlap is unusable at realizable amplitudes on THIS
  family → seg leg routes to representation-level conditioning (grammar-v2 /
  tk1 semantic-hybrid), not module-correction.
- F2: bare-module leakage cannot be distilled under gate at any rung while
  projected-training passes → decode-time projection is essential → family
  blocked on F3 economics (453 MB basis) — honest wall, report.
- F3: B/flip curve flat or rising across rungs/steps (never trends toward
  1.28) → rate-dominated; report the measured exchange-rate ceiling.
