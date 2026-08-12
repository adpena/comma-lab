# ddm_js4 — pose-null PROJECTED learned conditioning (fires if the js3 burn endpoint fails the pose gate)

Successor of ddm_js3, PRE-STAGED at mid-burn. FIRES ONLY IF the 300-step
pose-guarded burn endpoint selects NO module passing pose < 2e-6 with robust
movement (the mid-burn trajectory at step 100: robust beneficial 100 GROWING,
pose Δ 0.0157 ERODING despite λ_pose=1000 — the seg hinge is winning the
gradient fight). If the endpoint PASSES, this charter is void; do not spawn.

## THE STRUCTURAL CURE (recalled, not invented)

Stop FIGHTING pose with a penalty; make pose damage IMPOSSIBLE by construction:
- #889 (bo1, SHARPENED): seg training spends pose legibility ONLY when it
  spends the pose-visible subspace; a Q3-constrained correction CANNOT create
  pose damage (exact kernel, pre-quantization).
- #837 (Q3 measurement): the exactly-pose-null frame_1 subspace IS
  seg-reachable — the seg signal survives projection.
- j11 (#714): the pose-null/seg-null proposal split projector exists as
  landed machinery — recall it, do not rebuild.
- CAVEAT (#532, measured): uint8 rounding breaks exact-projection nullity
  (Δ=62.74 vs 1.7e-13) — post-uint8 pose leakage is bounded by the quantum
  but NOT zero. The pose guard therefore STAYS as a verification (cheap,
  pose-every-25 suffices — the projector does the heavy lifting; the guard
  only checks quantization leakage).

## THE CHANGE (minimal delta on the landed js3 trainer)

One composable stage: after computing c_theta (the learned correction, pre-R),
project onto the pose-null subspace of the custody PoseNet Jacobian:

  c_proj = (I − J_p^T (J_p J_p^T)^{-1} J_p) c_theta

with J_p the per-pair 6×(input-dim) PoseNet Jacobian on the custody planes
(computed ONCE per pair from the frozen net, cached — analysis-time cost, not
per-step; the projector is FIXED during training, per the anti-predictive
validity law it is a constraint not a model). λ_pose drops to a small
verification weight (or 0 — derive: with the projector, the pose term's only
job is quantization leakage, which the hinge form already prices).

Everything else inherits js3 verbatim: δ-hinge objective (δ=0.08036041259765625),
hidden-4 int8 module (751-819 B measured), batch 16 / 8 threads instrument,
stratified n32 sample seed 20260812, checkpoints every 25, stage rules, T4
admission gate (n600 robust ≤ −2,000 at ≤ 1,500 B via complete receiver).

## OPTIMAL FORM

- Reference form: js3 trainer (eb450d1281) + j11 projector class; the
  projector is the ONLY mechanism change. Provenance pins: js3 charter +
  FINAL_HANDOFF.json; #837/#889/#532 receipts by task id.
- SCOPE reductions legal (n32, bounded steps); MECHANISM reductions
  TOY-BRACKETED: an approximate/diagonal projector or skipping the uint8
  leakage guard cannot produce a family verdict.

## FALSIFIERS

- F1: projected gradient kills robust seg movement (the seg-reachable overlap
  #837 measured does not survive THIS module family) → report the measured
  overlap; route to per-pair projector relaxation (rank-k partial projection).
- F2: quantization leakage alone exceeds 2e-6 → the quantum bounds bind;
  report leakage distribution; route to leakage-aware rounding (CVP class).
- F3: same T4-gate economics failure as js2b F2 (robust flips too expensive).

## OPERATIONAL

Same laws as js3 (serializer --no-co-author, payload law, resumable, skeleton
annex queue, burn fires from MAIN). Run dir
/Volumes/VertigoDataTier/pact/ddm_js4_20260812/.
