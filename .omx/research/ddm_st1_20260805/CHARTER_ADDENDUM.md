# ST1 CHARTER ADDENDUM (operator mid-flight, 2026-08-05) — SCORER-NATIVE TRAINING SIGNAL

Operator verbatim: "This student should be trained based on our upstream GT and gradients and
all, including hyperplanes and channels and bases."

Leg A amendments (BINDING):
1. SUPERVISION = the frozen scorer's OWN signals, not labels: distill the MARGIN FIELD /
   logit-hyperplane distances (#141 margin-saliency, ∂margin/∂input; #63's measured lesson —
   margin/logit distillation sidesteps the argmax-CE wall). Upstream GT (cached lstars +
   the flip set) = the target; the scorer's GRADIENTS = the loss geometry.
2. OUTPUT PARAMETRIZATION in the HEAD'S OWN BASIS: predict margin/logit-space quantities in
   the head-hyperplane metric (lg1's coordinates; the argmax = the head hyperplanes' power
   diagram — predict distances-to-bisectors, not class masks).
3. CHANNELS: condition on / exploit the BN-derived per-channel capacity structure (#725
   FISHER_MARGIN per-stratum codebooks from the scorer's own BN buffers) + upstream
   channel/basis structure (YUV6 reads, D-nullity where relevant).
4. REALIZATION IN-LOOP (from TODAY'S n600 batch measurement — direct paint is DEAD: all 3
   carrier candidates S 1.51–6.11 vs live 0.754, pose damage ∝ painted area, the #934 leg-B
   mechanism at scale): the student's output must be trained THROUGH paint→R→uint8→SegNet
   (uint8-STE, the fd2 lesson) OR feed the solve-from-frozen-head path (m95: solve η +0.7895
   vs paint −3.764 on identical bytes). NO naive-paint evaluation — it measures the dead
   formulation, not the student.
Leg B unchanged. Comparison bar unchanged at the DESCRIPTION level; realized value judged
only through in-loop/solve realization.
