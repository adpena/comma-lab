# DAG FEED — arXiv 2607.10169 (RIPO) + 2607.04751 (TOP-D) → route to #500 optimal-metric + costate-organ distillation

**Leg:** DAG (routed intake). Operator shares 2026-07-14. Both re-pointed vs the full inventory. Both are
RL/distillation papers ⇒ they route to the METRIC (#500/#504) and COSTATE-ORGAN/SCORER-DISTILLATION
threads — NOT the witness codec directly (the witness trains AdamW/Muon on a reconstruction loss, not a
PPO policy). Disposition: genuine adjacency + conditional levers, NO fresh dispatch (refinements of live
threads). Stronger than the MorphoHDL/log-Sobolev conceptual-only; weaker than the ERM direct-fit.

## 2607.10169 — RIPO (Riemannian Isometric Policy Optimization) — Cai et al.
**Core:** PPO-Clip's failure = it implicitly measures policy discrepancy with a EUCLIDEAN metric, which is
inconsistent with the intrinsic RIEMANNIAN (Fisher/information) geometry of the policy manifold ⇒
systematically over-conservative in low-probability regions, over-aggressive in high-probability regions ⇒
exploration collapse. RIPO does ISOMETRIC updates on the Riemannian manifold (up to 60% over GRPO on AIME24).

**The fit — STRONG cross-validation of #500 (optimal metric) + #504 (Bregman/Fisher):** this is the RL
instantiation of OUR EXACT thesis — "a Euclidean/cosine surrogate metric where the natural geometry is
Fisher causes systematically biased updates; the proper metric is the flat Mahalanobis/Fisher pullback"
(we MEASURED "cosine is the wrong metric"; Fisher=margin 0.978). RIPO's region-dependent bias
(low-density vs high-density) maps to #500's **curriculum-varying metric** (different metric across
epochs). CONDITIONAL lever: IF any costate-organ / witness sub-optimization uses a PPO-Clip/GRPO-style
policy-gradient update (the #433 "DEEPEN RL/post-training" dig + #499 organ), RIPO's Fisher-isometric
correction is a direct drop-in. Route → #500 (grounding/citation) + #433/#499 (conditional RL lever).

## 2607.04751 — TOP-D (Trust Region Policy Distillation) — Xie et al.
**Core:** on-policy distillation (OPD) is unstable/high-variance; TOP-D dynamically constructs a PROXIMAL
teacher (a trust-region teacher kept NEAR the student) ⇒ stable, variance-reduced, monotonic-improvement,
NO extra compute vs OPD.

**The fit — costate-organ SCORER-DISTILLATION thread (#428 survey · #431 distilled surrogate · #455
forward-surrogate · #485 JEPA-latent):** we distill the frozen SegNet/PoseNet into a cheap surrogate (the
95%-kill throughput program). Our distillation is mostly OFFLINE (fixed frozen teacher), so TOP-D's
on-policy-instability fix does NOT directly apply — EXCEPT the real risk that a surrogate trained early
goes STALE as the witness output distribution shifts mid-training (a mild on-policy/distribution-shift
problem). IF #455/#485 measure surrogate staleness under witness-distribution drift, TOP-D's proximal-teacher
(keep the surrogate near the CURRENT distribution) is the stability technique. Route → #455/#485 as a
conditional stability lever; measurable only if surrogate-drift is observed.

## Honest limits (over-credit guard)
- Both are LLM-reasoning-RL papers (AIME/math benchmarks); the TRANSFER is the ALGORITHM/principle, not the
  results. The witness is not a policy; the organ is a small regressor (ridge/persistence per L-organ, n=1),
  not a PPO policy — so neither is a witness-training lever. RIPO's value is CROSS-VALIDATION of the metric
  thesis; TOP-D's is a conditional surrogate-stability technique.
- No pointer implication until a metric/organ/surrogate change is measured through R at n600, byte-closed.

**Pointer:** 0.19108 / 0.18804 UNMOVED. Routed grounding + conditional levers for the metric + distillation
threads; the pointer moves only via a byte-closed exact row.
