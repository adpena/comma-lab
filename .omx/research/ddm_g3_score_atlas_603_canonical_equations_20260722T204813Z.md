# DDM G3 Score Atlas — Canonical Equations

Date: 2026-07-22  
Scope: n600, `[macOS-CPU frozen-scorer advisory]`, `score_claim=false`

## Score-currency law

For pair `i`, let `e_i` be the measured SegNet argmax disagreement count over
the scored last frame, and let `q_i` be the measured squared error over the six
frozen PoseNet coordinates. With `N=600`, `H=384`, and `W=512`:

`D_seg = sum_i(e_i)/(N H W)`

`D_pose = sum_i(q_i)/(6 N)`

`M_seg,i = 100 e_i/(N H W)`

`M_pose,i = sqrt(10 D_pose) q_i/sum_j(q_j)`

`M_i = M_seg,i + M_pose,i`

`M_i` is the only pair-rank currency. The Pose term is an additive attribution
of the observed nonlinear global term, not an exact leave-one-pair-out
counterfactual. L2, pixel energy, response-cone volume, and byte allocation are
diagnostics and MUST NOT be rank keys.

For class `c`, target-margin band `b`, and topology stratum `t`:

`D_seg(c,b,t) = e(c,b,t)/(N H W)`

`S_seg(c,b,t) = 100 D_seg(c,b,t)`

The cells in one pair sum to its exact Seg score mass. The pair rows preserve
both this global mass and the conditional error rate `e(c,b,t)/sites(c,b,t)`.

The rank-four head-flip distance attached to an erroneous target/predicted class
pair is:

`d_flip = |margin_target| / ||w_target - w_predicted||_2`

It is geometry for costate consumers, not score currency.

## Concentration anchor

For pair masses sorted descending, the diagnostic top-k concentration is:

`H_k = sum_{i=1..k} M_(i) / sum_{i=1..N} M_i`

Measured anchors are joint `H_10=0.0197853`, `H_50=0.0954760`,
`H_100=0.187039`; Seg-only `0.0353794`, `0.154902`, `0.268544`; Pose-only
`0.0202390`, `0.0975207`, `0.189224`. Thus the joint debt is broad rather than
strongly heavy-tailed. “Heavy-tail” here names the concentration test, not a
positive conclusion.

## Hard-subset correlation anchor

For every exact measured v12 proposal `j`, `Delta_full,j` is its full-n600 joint
objective gain including exact marginal rate. For subset `A`,
`Delta_A,j` retains only distortion changes from pairs in `A` and retains the
same exact marginal rate. The measured Pearson anchor is:

`r_A = cov(Delta_A, Delta_full)/(sigma_A sigma_full)`

Measured over 338 proposals: `r_top24=0.595307`, `r_top64=0.575063`, and
`r_control24=0.234175`. These correlations license measure-first triage, never a
subset-only rank/kill verdict. Every future subset result must be paired with a
contemporaneous full-n600 delta that refreshes `r_A`.

## Authority and pointer

All values above are derived from SHA-bound frozen-scorer caches. They are not a
contest score. The canonical pointer remains `0.1910828242 [contest-CPU]`.

