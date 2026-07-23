# DDM J5 #366 realized-acceptance opening equations

Date: 2026-07-23  
Lane: `ddm_j5_366_realized_acceptance_warmstart`  
Evidence: `[macOS-CPU frozen-scorer advisory]`; `score_claim=false`.

## Settled input and reused laws

J4's beta2 transient cure remains valid:

\[
T_{\rm rw}
=\left\lceil {2\over 1-0.999}\right\rceil
=2000,
\qquad
a_t=0.1+0.9\min(t/2000,1).
\]

J5 removes only the fixed quarter-quantum cap. It does not remove the rewarmup,
Pose gate, deterministic receiver, or rollback.

The exact realized objective is the existing contest action:

\[
S(A)=100d_{\rm seg}(A)
  +\sqrt{10d_{\rm pose}(A)}
  +{25\,|A|\over 37{,}545{,}489},
\]

where \(A\) is the exact candidate archive and both distortions are measured
after receiver parse-back, paint, final uint8, evaluator \(R\), and frozen
scorers.

## Q8 proposal staging

Let \(\theta_t\) be the continuous receiver-coordinate shadow. Before exact
integer realization, J5 banks the proposal on camera Q8:

\[
\theta^{Q8}_{t+1}
={1\over256}\operatorname{round}(256\widetilde{\theta}_{t+1}),
\qquad
q_{t+1}=\operatorname{round}(\theta^{Q8}_{t+1}).
\]

The opening step has base learning rate \(0.25\), rewarm factor \(0.1\), and
first multiplier \(32\):

\[
\eta_0=0.25\cdot0.1\cdot32=0.8,
\qquad
\theta^{Q8}=205/256,
\qquad
q=1.
\]

Thus contributions accumulate on Q8 but the archive changes only at an exact
receiver quantum.

## V19-active lifecycle-feasible proposal

For the sealed active pair set

\[
P=(447,53,416,296,547,278,501,346),
\]

a track is eligible on axis \(a\) only if one lifecycle knot touches \(P\) and
the proposed integer shift keeps every polygon vertex in the scorer grid:

\[
\mathcal F_a(P,s)=
\left\{j:
\exists k\in{\rm knots}(j),\ p_k\in P,\quad
-m_{j,a}\le s\le (L_a-1)-M_{j,a}
\right\},
\]

with \(L_x=512,L_y=384\), and \(m,M\) the lifecycle-wide vertex extrema.
For `worldsheet_joint_active_x_+1`, \(|\mathcal F_x|=18\). The compiled archive
must equal v19's measured byte stream:

\[
\operatorname{SHA256}(A_{x+1})
=\texttt{d4eb1450f461437e714d08a9349cc735fe79b53a1739a2de92ef4850287dfd0d}.
\]

## Realized admission, shrink, and rollback

Against the last admitted exact archive \(A_t\):

\[
\Delta S=S(A')-S(A_t),
\qquad
\operatorname{admit}(A')\iff\Delta S<0.
\]

If \(\Delta S\ge0\), the candidate is not applied and its multiplier descends
the sealed ladder

\[
(32,16,8,4,2,1,\tfrac12,\tfrac14).
\]

If no candidate admits, the complete Adam/EMA/cursor state remains \(X_t\), an
atomic rollback checkpoint is written, immediately reloaded, and compared
bit-for-bit. An ascending receiver state therefore cannot enter campaign state.

## Component and residual fire gate

Pure-price admission is necessary but not sufficient for fire readiness. Let
\(\Delta E_R\) be the exact error delta on C1 residual target classes
Road+Undrivable+MyCar. J5 clears the fire gate iff

\[
\Delta S<0,\quad
\Delta d_{\rm seg}\le0,\quad
\Delta d_{\rm pose}\le0,\quad
(\Delta d_{\rm seg}<0\lor\Delta d_{\rm pose}<0),\quad
\Delta E_R<0.
\]

Lane+Movable errors form the role/correction bucket and are reported
separately; they cannot substitute for residual-trunk descent.

## Measured bounded row

\[
\begin{aligned}
\Delta d_{\rm seg}&=-2.8203328450521203\times10^{-5},\\
\Delta d_{\rm pose}&=-1.6296301816964842\times10^{-4},\\
\Delta B&=-5,\\
\Delta S_{\rm seg}&=-0.0028203328450521203,\\
\Delta S_{\rm pose}&=-0.00002017825870126444,\\
\Delta S_{\rm rate}&=-0.000003329294765610857,\\
\Delta S&=-0.002843840398518996,\\
\Delta E_R&=-2013,\\
\Delta E_{\rm role}&=-1314.
\end{aligned}
\]

All fire inequalities hold, producing
`READY_TO_FIRE_UNDER_STANDING_GO`. This equation note confers no execution
authority before MAIN review.
