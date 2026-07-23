# DDM SN1 error-source tensor — canonical equations

Date: 2026-07-23  
Lane: `ddm_sn1_segnet_telemetry_asymmetry`  
Axis: `[macOS-CPU frozen-SegNet+PoseNet advisory]`  
Scope: `research_only=true`, `score_claim=false`, pointer unmoved

## Exact residual partition

For pair \(t\), site \(p\), target cell \(y_{tp}\), exact v19c receiver
prediction \(\hat y_{tp}\), current semantic request \(s^0_{tp}\), and the
SHA-pinned DV1 `spline_plus_events` request \(s^1_{tp}\), define the measured
target-stratum residual

\[
E_{tp}=[\hat y_{tp}\ne y_{tp}]
       [y_{tp}\in\{\mathrm{Road},\mathrm{Undrivable},\mathrm{MyCar}\}].
\]

Each \(E_{tp}=1\) is assigned exactly once, in precedence order:

\[
q_{tp} =
\begin{cases}
\mathrm{DESCRIBED\_BUT\_REALIZATION\_LOST}, & s^0_{tp}=y_{tp},\\
\mathrm{NEVER\_DESCRIBED}, & s^0_{tp}\ne y_{tp}\land s^1_{tp}=y_{tp},\\
\mathrm{STRUCTURALLY\_HARD\_IRREDUCIBLE}, & \text{otherwise}.
\end{cases}
\]

The last name is formulation-scoped: it means irreducible only under the
current semantic program plus the one tested DV1 extension. It is not a
family-impossibility claim.

## Sided decision distance

For the current predicted winner \(c=\hat y_{tp}\), target rival
\(c'=y_{tp}\), frozen final-head logits \(z\), and final-head normal
\(\Delta w_{cc'}=w_c-w_{c'}\),

\[
D_2(t,p;c\to c') =
\frac{z_c(t,p)-z_{c'}(t,p)}{\lVert\Delta w_{cc'}\rVert_2}.
\]

The band is `BELOW_SIDED_Q10`, `Q10_TO_Q90`, or `ABOVE_SIDED_Q90` using the
measured full-n600 SDWL1 quantiles for the ordered orientation. An orientation
with no measured boundary support is explicitly
`NO_SIDED_REFERENCE_SUPPORT`; no zero threshold is invented.

## Geometry and temporal proxies

For the target-cell four-neighbour support,

\[
\kappa_4(t,p)=4-\sum_{q\in N_4(p)}[y_{tq}=y_{tp}].
\]

`INTERIOR` is \(\kappa_4=0\), `CODIM1_FLAT` is \(\kappa_4=1\), and
`CODIM2_OR_CORNER` is \(\kappa_4\ge2\). Four-connected components report
component count, min/median/p90/max size, and pixel mass at point
(\(\le4\)), boundary-segment (5–64), and region (\(>64\)) scale.

Historical G4 same-transition count \(r_{tp}\) is only a recurrence covariate:
\(r=0\) has no history, \(r=1\) is advected-or-singleton historical, and
\(r\ge2\) is static-in-image historical. A G3 scene-event label replaces this
with `EVENT_ADJACENT_G3_PROXY`. None of these labels asserts record-level
constancy or current causality.

## Budget and solve-first allocation

For source \(q\) and target stratum \(k\),

\[
N_{qk}=\sum_{t,p} E_{tp}[q_{tp}=q][y_{tp}=k],\qquad
d^{\mathrm{global}}_{qk}=\frac{N_{qk}}{600\cdot384\cdot512}.
\]

The exact closure conditions are

\[
\sum_{q,k}N_{qk}=2{,}265{,}811
\quad\text{and}\quad
\sum_{q,k}d^{\mathrm{global}}_{qk}=0.019207517835829.
\]

The measured source counts and fractions are

\[
\begin{array}{r|r|r}
q & N_q & N_q / 2{,}265{,}811\\ \hline
\mathrm{DESCRIBED\_BUT\_REALIZATION\_LOST} & 892{,}710 & 0.393991\\
\mathrm{NEVER\_DESCRIBED} & 738{,}090 & 0.325751\\
\mathrm{STRUCTURALLY\_HARD\_IRREDUCIBLE} & 635{,}011 & 0.280258
\end{array}
\]

Menu rank is lexicographic:

\[
\mathrm{vocabulary}
\succ \mathrm{chart/parameter}
\succ \mathrm{point\ correction},
\]

then descending measured error mass. The only measured cross-clip
error-mass/byte ratio in this pass is
\(738{,}090/1{,}610=458.4409937888\) errors per shared DV1 byte. It is a
semantic-reach ratio; receiver realization, Pose, exact archive bytes, and
score remain owed.

## Paint-floor mechanism and #149 wall

For an error already assigned to `DESCRIBED_BUT_REALIZATION_LOST`, the
observable mechanism classifier uses target-boundary distance, availability
of a continuous Lane curve, and sided \(D_2\). It emits one of
`COARSE_DESCRIPTION`, `PAINT_FUNCTION`, or `TEXTURE_PRIOR_REGION_ERF`.
The measured counts close exactly:

\[
587{,}913 + 208{,}623 + 96{,}174 = 892{,}710.
\]

This is a deterministic partition of measured axes, not identification of a
hidden causal state.

For target-boundary band \(B\), the current survival-wall statistic is

\[
f_{\mathrm{wall}} =
\frac{\sum_{t,p}[p\in B_t][\hat y_{tp}\ne y_{tp}]}
     {\sum_{t,p}[p\in B_t]}
= \frac{1{,}613{,}214}{4{,}684{,}236}
=0.34439212712596035.
\]

Against the historical mp128 three-frame reference
\(f_{\mathrm{mp128}}=0.1605960279317129\), the contextual ratio is
\(2.1444622981111303\). The formula is common, but the receiver, sample scope,
and authority axis differ, so this is not a pooled estimator.

## Scorer-native relay product

For relay \(\ell\), channel \(c\), native pooled cell \(u\), pair \(t\), and
group \(g\), the bounded product retains moments and contrast energy of
activation \(a_{\ell c u t}^{(g)}\). The principal measured gap is

\[
\delta_{\ell c u t} =
a_{\ell c u t}^{(\mathrm{painted})}
-a_{\ell c u t}^{(\mathrm{GT})}.
\]

The geometry residual fraction separates a channelwise uniform shift:

\[
\delta^{\mathrm{uniform}}_{\ell c t}
=\frac{1}{|U|}\sum_u\delta_{\ell c u t},\qquad
r^{\mathrm{geom}}_\ell
=1-\frac{\sum_{c,t,u}
  (\delta^{\mathrm{uniform}}_{\ell c t})^2}
 {\sum_{c,t,u}\delta_{\ell c u t}^2}.
\]

Directional expansion is a trajectory secant, not a Jacobian condition
number:

\[
\gamma_\ell =
\frac{\operatorname{RMS}(\delta_\ell)}
     {\operatorname{RMS}(\delta_{\ell^-})+\epsilon}.
\]

For centered channel trajectory matrix \(A_\ell\), the deterministic
stable-rank summary is

\[
\operatorname{srank}(A_\ell)
=\frac{\lVert A_\ell\rVert_F^2}
       {\sigma_1(A_\ell)^2},
\]

where \(\sigma_1\) is estimated with fixed float64 power steps and
normalization at each half-step. Any non-finite intermediate is a hard
refusal.

The advisory relay score is

\[
\operatorname{relay\_score}_\ell =
\frac{\prod_{j>\ell}\gamma_j}
     {\max(\operatorname{RMS}(\delta_\ell),\epsilon)
      \max(\operatorname{srank}(A_\ell),1)}.
\]

It ranks low-rank, lower-gap relays with measured downstream leverage; it does
not authorize intervention.

For the first six pairs only, PoseNet output sanity reports

\[
\operatorname{MSE}_{\mathrm{pose}} =
\frac{1}{6d}\sum_{t=1}^{6}
\lVert P(x_t^{\mathrm{painted}})
       -P(x_t^{\mathrm{GT}})\rVert_2^2
=163.06120732858346.
\]

This is an advisory microbatch diagnostic, not official \(d_{\rm pose}\).

## Frozen-weight amplitude and spectral factors

For frozen convolution kernel \(K_{\ell,o,i}\), the exact periodic-grid
response retained by the analytic artifact is

\[
H_{\ell,o,i}(\omega_x,\omega_y)
=\sum_{m,n}K_{\ell,o,i}[m,n]
 e^{-\,\mathrm{i}(\omega_xm+\omega_yn)}.
\]

BatchNorm inference contributes affine gain
\(\alpha_c=\gamma_c/\sqrt{\sigma_c^2+\varepsilon}\);
`LayerScale2d` contributes its learned channel multiplier; squeeze-excite
contributes the measured gate; GELU contributes its local derivative. PoseNet
has exactly 24 `LayerScale2d`, 8 `BatchNorm1d`, 19 `GELUTanh`, and one
`SEModule`.

The camera-to-scorer resize artifact records exact phase-indexed bicubic-up and
bilinear-down kernels. Their composition across uint8 quantization has no
single global transfer scalar: resize phase and borders vary spatially, and
uint8 makes the map piecewise affine. No such scalar is inferred.

## Historical receipt custody

G2 contributes only a class-level measured energy/byte marginal and every
tensor/menu row declares `NO_JOINT_PER_CELL_CUSTODY`. G3 contributes pair
rank/tail/event covariates, G4 contributes historical recurrence, v14 is a
named realization-leak cross-check, and e1 is an exporter/receiver-survival
cross-check. None is used as current per-cell causal truth.
