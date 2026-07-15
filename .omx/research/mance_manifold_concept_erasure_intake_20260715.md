# Intake: MANCE — Manifold Aware Concept Erasure (Avitan/Goldberg/Elazar, arXiv 2607.03973) + github.com/MatanAvitan/mance

**Operator-routed 2026-07-15.** Per `PAPER_WARM_START_FROM_DIVERGENCE`: trace the fork, import what
survives OUR premises (single frozen instance, deterministic scorers, exact gradients, byte-counted
payload).

## What it is

Concept erasure with the **Manifold Constraint Hypothesis**: natural representations concentrate on a
low-dim manifold, so edits constrained to the LOCAL TANGENT of that manifold preserve unrelated
information far better than unconstrained edits. Mechanics per edit step: nonlinear concept probe →
local-PCA tangent basis `B_i` on k-NN of the UNEDITED representations → spectrally weighted tangent
direction `d_i = B_i diag(σ_i^α) B_iᵀ u_i` → per-row trust region `‖x̃_i−x_i‖ ≤ ε·r_i`. Variants
prepend LEACE (closed-form linear erasure) and CovMatch. SOTA nonlinear erasure over 119 settings;
the evaluation object is the **leakage-vs-control-degradation curve at every budget ΔY**. Clean small
codebase (erasure.py / tangent.py / intrinsic_dim.py / preprocess.py).

## The assumption fork

They erase a STATISTICAL concept over a dataset distribution using fitted probes; we edit a SINGLE
frozen instance against DETERMINISTIC scorers with exact gradients. We do not need their probe (we
have the true scorer and its Jacobian), and we run the SIGN-FLIPPED problem: they REMOVE a concept
while preserving controls; our terminal finisher INJECTS corrections (fix argmax flips) while
preserving controls (d_pose, non-target pixels). Same algebra, opposite sign.

## What SURVIVES (3 imports, all landing on the terminal-band finisher line)

1. **THE reframe for #400/#396 (MC-finisher pair-local DIAGONAL mode): realizability = the manifold
   constraint.** For us the "natural manifold" is the set of WITNESS-DECODABLE frames — the image of
   the decoder map. A pixel-space flip-fix that leaves that manifold cannot be expressed by ANY
   payload edit. MANCE's prescription, translated: compute the correction INSIDE the decoder's
   tangent space — project the desired flip-fix onto the RANGE of ∂frames/∂payload (which we already
   possess exactly via the deterministic-differentiable decode, #350), spectrally weight by scorer
   sensitivity (our margin/Fisher surrogate plays their diag(σ^α)), and bound each edit by a per-site
   trust region (our #500 reachable-decision-geometry G_dec plays their ε·r_i). This upgrades the
   diagonal click-polish from coordinate-axis edits (#399's PR128-style clicks) to
   manifold-tangent edits — strictly more surgical per unit payload change.
2. **The measurement discipline:** their leakage-vs-ΔY curve is exactly the finisher's honest scoring
   object — flips-fixed vs (Δd_pose + collateral flips) at every edit budget, a CURVE not a point.
   Adopt as the #400 A/B deliverable shape (prevents a single-point "polish worked" claim).
3. **Nonlinear-leakage warning (bounded):** their measured core lesson — linear erasure leaks under
   nonlinear probes — cautions any of OUR null-space claims certified only against linearizations.
   Our #401 blind-coordinate exploit is EXEMPT (bit-identity through both scorers at n600, stronger
   than statistical erasure), but future "safe-plane" style claims (#453-class JRD planes) should
   state whether the certificate is exact-through-scorer or linearized. verdict_scope of the caution:
   family (linearized-certificate claims), not our exact-certified rows.

## Explicitly NOT imported

Probe fitting / LEACE / CovMatch as components (we hold the true scorer; statistical erasure
machinery is dominated by exact gradients on a frozen instance); dataset-level k-NN manifold
estimation (our manifold is the decoder image, known analytically through #350, not estimated).

**Routing:** design input to #400 (fires post-launch per the compose arm's sequencing) + #396; the
leakage-curve deliverable shape folds into the #400 charter at fire time. Their intrinsic_dim.py is a
cheap cross-check of our measured ~8-dim island manifold (L17) if ever re-derived — noted, not owed.
**Pointer honesty:** intake/means; pointer 0.19108 UNMOVED. papers-checked: mance_2026 →
IMPORT-3 (decoder-tangent realizability reframe · leakage-curve discipline · linearized-certificate
caution), machinery NOT-imported (probes/LEACE dominated by exact gradients).
