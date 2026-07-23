# DDM J4 warm-start reform equations

Date: 2026-07-23
Lane: `ddm_j4_366_warm_start_reform`
Evidence axis: `[macOS-CPU frozen-scorer advisory]`
`score_claim=false`; pointer `0.1910828242 [contest-CPU]` unmoved.

## Settled inputs

- The V15 archive has receiver bytes but no loadable optimizer moments.
- J3 used Adam with implicit \(\beta_2=0.999\). On its first step, every nonzero
  gradient produced an update of magnitude \(0.25\), demonstrating the
  fresh-second-moment sign-normalization transient.
- J3 first realized at step 4. Its five island quanta regressed both exact n600
  components; its nine global-template quanta regressed Seg while improving
  Pose. EMA remained below one receiver quantum and did not trigger the failure.
- A local four-pair CE-plus-margin improvement is not an exact receiver
  admission. A smaller trust-region step is also not an admission.

## Reused canonical law

No new equation is registered. J4 consumes
`adam_v_variance_warmup_length_v1`:

\[
T_{\mathrm{rw}}
= \left\lceil {c \over 1-\beta_2} \right\rceil
= \left\lceil {2 \over 1-0.999} \right\rceil
= 2000\ \text{optimizer steps}.
\]

The opening linear factor after \(t\) completed steps is

\[
a_t = 0.1 + 0.9\min\left({t\over 2000},1\right),
\qquad
\eta_t = 0.25\,a_t.
\]

The optimizer applies the independent continuous-coordinate cap

\[
\left|\Delta\theta_i\right| \le {1\over4}
\quad\text{receiver quantum}.
\]

The cap limits the transient. It never authorizes an integer receiver change.

## Opening force schedule

Before the first strict exact admission,

\[
g_{\text{template}}=0,\qquad
w_{\text{pose}}=0,
\]

so the active opening force is island Seg CE-plus-margin only. The global
template group and Pose force release only after a strict exact n600 island
admission.

Let \(R_q(\theta)\) be the integer receiver wire realization. The first boundary

\[
R_q(\theta_{t+1}) \ne R_q(\theta_t)
\]

immediately invokes exact archive compile, parse-back, uint8 \(R\), frozen
SegNet, and frozen PoseNet over n600.

## Component-safe admission and rollback

With exact deltas measured against the last admitted receiver state,

\[
\operatorname{admit}
\iff
\left(\Delta d_{\rm seg}<0 \land \Delta d_{\rm pose}\le0\right)
\lor
\left(\Delta d_{\rm seg}=0 \land \Delta d_{\rm pose}<0\right).
\]

Every other case aborts and rolls back the candidate. In particular,

\[
\Delta d_{\rm seg}=0,\quad\Delta d_{\rm pose}=0
\Longrightarrow
\texttt{BLOCKED\_REALIZED\_NO\_COMPONENT\_DESCENT}.
\]

## Bounded J4 remeasurement

Four opening steps remained in the original receiver cell:

\[
\Delta d_{\rm seg}=0,\qquad
\Delta d_{\rm pose}=0,\qquad
\Delta B=0.
\]

This removes the J3 regression but does not satisfy admission. The scoped
instance remains blocked; no campaign launch is authorized.
