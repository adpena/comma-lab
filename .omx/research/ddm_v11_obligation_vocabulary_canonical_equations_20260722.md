# Canonical equations draft — DDM V11 scorer-obligation vocabulary

**Status:** DERIVED laws plus MEASURED local advisory anchors. `score_claim=false`; MAIN review is
required before registration or downstream promotion.

## E1 — exact action and marginal admission

For source bytes `N=37,545,489`, exact receiver-closed archive bytes `B`, Seg disagreement `D_seg`,
and official YUV6 Pose error `D_pose`,

`S = 100 D_seg + sqrt(10 D_pose) + 25 B/N`.

For a measured candidate with proposed archive increment `Delta B`, full-window error counts
`E -> E'`, and `P=pair_count*384*512`,

`Delta S_seg = 100(E'-E)/P`,

`Delta S_pose = sqrt(10 D'_pose) - sqrt(10 D_pose)`,

`Delta S_rate = 25 Delta B/N`,

`admit iff Delta S_seg + Delta S_pose + Delta S_rate < 0`

and `D'_pose <= D_pose,base + 1.0`. The tube is a safety rail only. Unlike V10, positive
`Delta S_pose` is legal when the complete measured delta remains negative.

## E2 — rank-4 Fisher/margin acquisition

For erroneous cell target `c`, predicted class `c'`, target top-two margin `m`, and exact frozen-head
normal `n_cc'=||w_c-w_c'||_2`,

`d_flip = |m|/n_cc'`,

`kappa_F(m) = 1/2 sech^2(m/2)`,

`A = 1[c != c'] kappa_F(m) w_band(|m|) / max(d_flip, 10^-3)`,

where `w_band` is `4,2,1,.25` on `[0,.1),[.1,.5),[.5,1),[1,inf)`. `A` orders encode-side
obligations only; it is not an inner-Jacobian or admission verdict.

## E3 — Lane chart obligations

For stored Lane line vector `a_tj`, V11 can address `q in {0,1,2,3,4,5,7}` and applies

`a'_(t,j,q) = fp32(a_(t,j,q) + delta_(t,j,q))`.

`q=0..3` are centerline coefficients, `q=4..5` width coefficients, and `q=7` dash phase. The
encoder derives phase against the forward chart and stored Pose6 key
`xi_key=p_t[0]-p_t[1]`; the generic LBND2 receiver rerasterizes the counted chart. No lane pixels
or GT labels enter the packet.

## E4 — compact parabolic boundary shearlets

For atom center `(x0,y0)`, scales `(s_x,s_y)` with `s_x >= 2 s_y`, shear `h/16`, and amplitude
`a/16`,

`u=(x-x0)/s_x`,

`v=(y-y0-(h/16)(x-x0))/s_y`,

`psi(u,v)=[max(1-u^2,0)]^2 [max(1-v^2,0)]^2`,

`Delta y(x,y)=round((a/16) psi(u,v))`,

`M'(y,x)=M(y-Delta y(x,y),x)`.

This finite compact parabolic atom is the governed Fourier-free Road/Undrivable displacement
surface. The result is receiver-synthesized from parameters, not a residual image.

## E5 — Movable moment/curvelet island

Pose6 transports the center from birth pair `t0`:

`x_t=x0+round((p_t[0]-p_t0[0])g_x/16)`,

`y_t=y0+round((p_t[1]-p_t0[1])g_y/16)`.

After rotation and radius normalization to `(u,v)`, define skew `k`, taper `tau`, curvelet
coefficient `gamma`,

`r(v)=clip(1+tau v,.25,2)`,

`u_s=(u-k v max(1-v^2,0))/r(v)`,

`lobe=max(1-4(u-.5)^2,0) max(1-v^2,0)`,

`T=max(.25,1+gamma lobe)`,

`M_island = 1[u_s^2+v^2 <= T]`.

Birth unions and death subtracts this generic shape from the Movable semantic layer.

## E6 — surgical measured bundles and ladder

Atomic obligations are partitioned by canonical scorer batch and by one of six families: Lane
center, Lane width, Lane phase, Road boundary, Undrivable boundary, Movable shape. Each measured
bundle contains at most 16 nonconflicting atoms. Up to 32 bundles are chosen with at least one from
every available family, then ordered by Fisher priority. Exact sequential replay, not additive proxy
composition, determines admission.

For requested additions `b in {0,16384,49152,98304,147456}`,

`b_eff=min(b,200000-B0)`,

and each rung selects the longest admitted prefix whose exact receiver-closed bytes satisfy
`B-B0 <= b_eff`. Unspent bytes are explicit and never filled by an unmeasured symbol.

## E7 — scoped plateau falsifier

Let `P200` mean the typed ladder made `B0+b_eff >= 199,000` available, `I` mean every bounded atomic
obligation entered a measured bundle, `F` mean the last admission preceded at least four measured
rejects, and `T` mean `D_seg > .00116`. The preregistered n600 formulation signal is

`Phi = 1[n600] * P200 * I * F * T`.

At n600, `P200=1`, `F=1`, `T=1`, but `I=0`: 4,096 atoms were retained, 492 measured, and 3,604
unmeasured. Hence `Phi=0`. The result identifies bounded measurement coverage as the current blocker;
it does not identify the v6 worldsheet as the binding cause and does not close any family.

## Measured anchors

- n64: one Lane-phase bundle, +319 B, `Delta S=-5.688893671509e-3`.
- n256: three Lane-width bundles, +575 B total, objective `43.789395685794 -> 43.787940257364`.
- n600: 32 bundles / 492 atoms measured, zero admitted; exact base/final archive SHA
  `7a544e1d8a33f19f9045054d4ec2ab391d119e27e948b0bf33ab20f223df622a`.

## STORES CONSULTED

`direct_description_minimizer_PRIMARY_SPEC_20260721T214800Z.md`; V10 equations/finding/receipts;
V5/V6 bound predictor receipts; frozen target receipt/cache; 2026-07-19 Fisher/EV/curvelet/xi
directives; `docs/operating_manual_craft_handoff.md`; v7.5 and v8 canonical specs.
