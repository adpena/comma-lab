---
schema: ddm_v13_worldsheet_event_predictor_equations.v1
date_utc: 2026-07-22
equation_id: ddm_describe_line_rate_distortion_bracket_v1
axis: "[macOS-CPU frozen-scorer advisory]"
score_claim: false
main_landing_review_required: true
---

# V13 measured equation row

For each receiver-closed rung `r`, the advisory action is

`S_r = 100 D_seg,r + sqrt(10 D_pose,r) + 25 B_r / 37,545,489`.

The exact G1 Movable derivation has

`L_G1 = 29,810 B`,
`e_mask = 33,378`, and
`D_mask,clean-rest = 33,378 / (600*384*512) = 0.00028294881184895833`.

Its selected receiver-visible realization is

`(B,D_seg,D_pose,D_Movable) = (132,606, 0.029592759874, 163.016398660918, 0.481331895297)`.

Relative to the 102,105 B inherited base:

`Delta S_islands = 43.422862186555 - 43.896380982393 = -0.473518795838`,

`Delta S_islands / Delta B = -1.552469741445e-05 per byte`.

The mask clean-rest value and Movable-conditional through-R value have different denominators and
must not be subtracted as one metric. Their joint evidence is categorical: an exact, below-box
semantic grammar exists, while the receiver-visible realization remains far above the target.
Therefore the registered binder is `receiver_projection`, not `shape_expressiveness` or
`track_fidelity`.

For Lane:

`Delta B_lane = 2,894 B`,
`Delta S_lane = +0.366949051781`,
so the measured family instance is rejected. Its derivation-field price is
`284 + 1,040 + 308 + 420 = 2,052 B`; manifest/ZIP unique-home overhead is `842 B`.
This Lane equation is a pre-19:16 measured baseline only. It is not the later BEV-curvature,
range-gated dash-comb, anisotropic-volatility AR(1) innovation, or Road-polytope equation.

The G2 receiver-closed phase-only ablation adds a measured split law. For n600,

`Delta B_phase = 2,128 B`,

`Delta D_seg,total = +0.000280736287`,

`Delta D_seg,Lane = -0.029228004790`, and

`Delta S_phase = +0.028907928562`.

Thus the raw q8 phase-symbol instance is Lane-helping but total-objective-harming and is rejected.
This does not close the distinct anisotropic AR(1)-whitened BEV formulation. The law is SHA-bound
to n64 phase receipt `b39b86b23517d1696cdc24316f05bd11066ecb3fe72d7acf7156bb76e9f7bf93`
and n600 phase receipt `622e6eda5b06e3fbe16ec494648dd85f6ba73545716b65fcdb07f797539f62ad`.

G3 adds the allocation constraint

`C_top10,joint = 0.019785252279635502`,
`C_top100,joint = 0.1870385612422957`, and
`C_top100,Seg = 0.2685436608566537`.

These measured concentrations imply `debt_shape = BROAD_NOT_HEAVY_TAILED`. Consequently,
`per_pair_topk_correction_allowed = false`; admissible spend must be amortized shared
grammar/templates/process priors, and top24 (`r=0.5953065905385343`) is screening-only with a
mandatory full-n600 verdict. This policy binds G3 receipt SHA
`6c4157092a7bdf7ba44b458cd470725cc470d84a8fc77ed7d3dedb59160734f5`.

The V13 empirical law row is INSTANCE-scoped:

`G1 exact payload AND receiver closed AND B<=200,000 AND D_Movable<=0.5`
`AND D_seg>0.01 => ADVISORY_V13_INSTANCE_FALSIFIER_TRIGGERED_FORMULATION_ONLY`.

It is SHA-bound to final-source n64 receipt
`0c51bb33e42dabeed1f434c838cad4b0ec51cce02ebb61b89e2fc9cbb2b3da4a` and n600 receipt
`31fc55b3bfeec4dbc1b5a40155438cb71190b814dadc05ad73f678bcd9c46bf5`.

This is not a contest score, promotion law, successor-grammar measurement, or family-level
negative. Pointer unchanged.
