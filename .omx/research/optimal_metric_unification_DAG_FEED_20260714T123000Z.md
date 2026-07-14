# DAG FEED — the OPTIMAL METRIC as ONE unifying principle (operator P0, 2026-07-14)

**Leg:** DAG (campaign synthesis). Equation leg deferred to the arms (register the metric law only when MEASURED).

## The principle (from arm surrogate_vjp_fidelity_metric, MEASURED)
The argmax lives in the **reachable decision-geometry**, not the ambient RGB metric:
`<g_T,g_S>_(R,M) = g_T^T J_R M J_R^T g_S` — renderer-pullback `J_R` (~19 reachable dims vs 589,824 ambient)
+ Fisher/preconditioner `M` + the 4-dim centered-logit decision quotient `q=P·H·z`. Ambient→this metric
lifted alignment **12.5×–51×**. Sisters already in-tree: Fisher=margin 0.978, distortion-on-separatrix,
anisotropic-annulus/dark-interior.

## The operator's leap: it is ONE principle across FOUR surfaces (not a surrogate fact)
1. **Fidelity** (surrogate/organ agreement) — replace raw-cosine with `<·,·>_(R,M)`. [arm: optimal_metric_p0_build_surrogate_followons]
2. **Training loss** (operator: "perhaps that applies to our training loss term too") — a natural-gradient /
   Fisher-Rao / mirror-descent(Bregman) loss in the decision-quotient metric vs raw CE/soft-cosine; likely
   the unifying generalization of MarginBandSatisficing(#360/#459) + per-pair σ(#382) + margin-saliency(#141/#274).
   [arm: optimal_metric_training_loss_curriculum]
3. **Curriculum-varying metric** (operator: "we can use different throughout the curriculum too") — METRIC-ANNEAL
   dual to τ-anneal/curvelet-scale: coarse ambient early (interior dark → Fisher degenerate) → Fisher/margin-weighted
   mid (annulus localizes) → functional-flip-preservation terminal (flicker band, GT-oracle floor 0.005318). DSL
   lever = metric-as-stage-parameter. [same arm]
4. **RGB cargo-cult replacement** (operator: "we have cargo culted RGB optimal stuff... find optimal replacement")
   — every surface that optimizes/represents/MEASURES in RGB when only the argmax/task-manifold matters. HARD-EARNED
   boundary: the FINAL render MUST be RGB (SegNet reads RGB / PoseNet YUV6); the cargo-cult is everything
   before/around it (RGB fidelity losses, RGB-native bases, full-RGB recon objectives, RGB-derived filters). This
   is the §NON-RGB TASK-SPACE WITNESS CAPSTONE thesis made an audit. [arm: rgb_cargocult_scrutiny_optimal_replacement]

## Why this is coherent (the double one-object)
Same coarse-to-fine as the unified level-set flow, seen through the METRIC instead of the basis. The metric IS
the geometry the separatrix lives in; measuring/optimizing/representing in RGB is measuring in the wrong chart.
The four surfaces are four readings of "use the decision-geometry, not the ambient RGB."

## Discipline
All arms: MEASURE on n600-real (a metric win is a lower d_seg/d_pose through R OR faster same-argmax convergence,
never a proxy-loss claim); NO-FAKE (RGB IS required at the boundary — do not "fix" a real requirement);
verdict-scope; measure/backtest-gated, NO launch (operator-GO CONTAINMENT); ONE shared canonical metric helper
across the three #500 arms (agreed via broadcast, not duplicated).

**Pointer:** 0.19108 / 0.18804 UNMOVED — MEANS (apparatus). Task #500.
