# DDM DV1 description vocabulary — canonical equations

Date: 2026-07-23
Axis: `[macOS-CPU frozen-scorer advisory]`
Authority: semantic-cell description measurement only; `score_claim=false`

## 1. Vocabulary-reach law

Let \(p_i\) be the preserved predictor argmax class, \(t_i\) the target class,
and \(D_G(i)\) the class proposed by derivation grammar \(G\).  The reach of
\(G\) on target stratum \(c\) is

\[
\mathcal R_c(G)=
\frac{\sum_i \mathbf 1[p_i\ne t_i,\ t_i=c,\ D_G(i)=c]}
{\sum_i \mathbf 1[p_i\ne t_i,\ t_i=c]}.
\]

This is description reach, not a score.  The receipt separately records
collateral and net semantic-cell closure:

\[
\Delta E(G)=
\sum_i \mathbf 1[p_i\ne t_i,\ D_G(i)=t_i]
-
\sum_i \mathbf 1[p_i=t_i,\ D_G(i)\ne t_i].
\]

A high \(\mathcal R_c\) with negative \(\Delta E\) means the vocabulary can name
the target sites but its current arbitration rule is too broad.  That negative
is arbitration-formulation scoped; it is not evidence against the primitive
family.

## 2. Three coordinates of the ground worldsheet

The persistent phase field is

\[
c_\star(x)=\arg\max_c \sum_t \mathbf 1[c_t(x)=c].
\]

The upper Road separatrix is a bilinear worldsheet spline

\[
\Gamma(t,x)=(t,x,\gamma(t,x)),\qquad
\mathrm{Road}(t,x,y)=\mathbf 1[y\ge\gamma(t,x)].
\]

The curvature/arc-length curve uses tangent-angle coordinates

\[
\frac{\partial \Gamma}{\partial s}
=(\cos\theta(s),\sin\theta(s)),\qquad
\kappa(s)=\frac{\partial\theta}{\partial s}.
\]

These expose the persistent phase, temporal separatrix, and local turning
coordinates demanded by a level-set energy of the form

\[
E(\phi)=\int
\alpha\lVert\nabla\phi\rVert
+\beta\kappa(\phi)^2
+\lambda\,\mathbf 1[c(\phi)\ne t]\,dx\,dt.
\]

## 3. Non-additive composition law

Individual reach and individual coded length are diagnostics.  Selection uses
one decoded composition and one actually coded joint section:

\[
L_{\mathrm{joint}}
=
\left|\mathcal C(G_{\mathrm{static}},
G_{\mathrm{boundary}},
G_{\mathrm{curve}})\right|,
\]

not \(\sum_j |\mathcal C(G_j)|\), and

\[
D_{\mathrm{joint}}
=D_{\mathrm{events}}\circ D_{\mathrm{ground}},
\]

where existing Lane, MyCar, and Movable event masks are applied after the
ground partition.  Error closure is recomputed from
\(D_{\mathrm{joint}}\); individual error pools are never summed.

## 4. Triality

- DSL: typed `persistent_level_set`, `boundary_worldsheet_spline`,
  `turning_angle_curve`, and `joint_ground_vocabulary` envelopes with strict
  real-coder parse-back.
- DAG: bound n600 target labels + reconstructed final v12 argmax cells + G4
  recurrence counts → three fits → existing event-mask composition → joint
  semantic-cell measurement → compact ledger/receipt.
- Equations: \(\mathcal R_c\), \(\Delta E\), \(c_\star\), \(\Gamma(t,x)\),
  \(\kappa(s)\), and \(L_{\mathrm{joint}}\) above.

## Scope

The equations govern the exact semantic-cell audit.  They do not establish RGB
receiver survival, Pose safety, exact final container bytes, or contest score.
The independently observed physical-BEV registration required to promote
\(\gamma\) from image-worldsheet coordinates remains owed.
