---
task: 603
feeds_task: 613
research_only: true
triality_leg: equations
formalization_status: FORMALIZATION_PENDING
main_landing_review_required: true
---

# DDM v6 evaluator-bridge and temporal-amortization law draft

`FORMALIZATION_PENDING`: MAIN must review and register this draft.  This branch does not mutate the
canonical equation registry.

Let `A(z)` be the deterministic composed archive, `R(A(z))_p=(y0,y1)` the exact uint8 receiver
output for pair `p`, `C` the frozen SegNet last-frame argmax, `Q` the official PoseNet YUV6 map and
first-six output, `L*_p` and `q*_p` the SHA-bound `gt_n600` targets, and `N=384*512`.

## Proper evaluator for structured carriers

```text
D_seg(F,z) = (1 / (|F| N)) sum_{p in F} sum_i 1[C(R(A(z))_p)_i != L*_{p,i}]

D_pose(F,z) = (1 / (6 |F|)) sum_{p in F} ||Q(R(A(z))_p)[:6] - q*_p||_2^2

B(F,z) = |A(z)|
```

These are measured through receiver bytes and frozen CPU-Torch scorers.  Pose6 payload exactness is
an independent custody predicate; it is not `D_pose`.

## Honest membership bridge

Let `T_p=C(C1_p)` and `epsilon=Pr[T_p != L*_p]=0.000126361847` from the settled
`0.999873638153` C1-to-GT match.  For composed membership `M=Pr[C(R(A(z)))=T]`:

```text
1 - D_seg - epsilon <= M <= 1 - D_seg + epsilon
```

The v5 exact controls retain their directly MEASURED `M`.  Amortized rows use only this DERIVED
interval until a fresh C1-target scorer pass exists.

## Unit-root AR(1) key hold

For semantic stream state `x_p`, key set `K`, and latest-key map `k(p)=max{k in K:k<=p}`:

```text
x_hat_p = x_{k(p)}
x_hat_p = 1*x_hat_{p-1} + 0,  p not in K
x_hat_p = x_p,              p in K
```

Thus the named AR(1) formulation is specifically a unit-root zero-innovation hold, not a fitted
general-phi predictor.  Pose6 stays exact and is never held.

Fixed mode uses `K={0,24,48,...}`.  Xi mode derives `K` from counted Pose6 motion
`m_p=||pose6_p-pose6_{p-1}||_1`, budget
`b=ceil(sum_p m_p / ceil(P/24))`, and emits a key when accumulated motion reaches `b` or the gap
reaches 24.  No unmeasured motion threshold is invented.

## Marginal rate law

For the two measured windows `P0=64`, `P1=256`:

```text
beta(z) = (B(P1,z) - B(P0,z)) / (P1-P0)
```

Measured anchors:

```text
beta(v5_exact) = 204925 / 192 = 1067.317708333333 B/pair
beta(fixed24)  =  20728 / 192 =  107.958333333333 B/pair
beta(xi24)     =  20724 / 192 =  107.937500000000 B/pair
beta(zero)     =  21167 / 192 =  110.244791666667 B/pair
```

Static-once homes have `beta=0`.  Pose6 remains `5.75 B/pair`; fixed24 chart residuals fall from
`664.9166667` to `37.7916667 B/pair` in aggregate.

## Admission law

```text
local_rate_pass(z) := beta(z) <= 300 B/pair and B(z) <= 216207
efficacy_pass(z)   := D_seg(z) <= 0.00116
admit(z)           := local_rate_pass(z) and efficacy_pass(z) and receiver_closed(z)
                      and contest_axis_verified(z)
```

All amortized n64/n256 rows pass the marginal-rate test.  None passes `D_seg<=0.016`, hence none
passes `efficacy_pass` and none is admissible.

## Registry draft

```text
law_id: structured_carriers_evaluator_bridge_temporal_amortization_v1
status: FORMALIZATION_PENDING
evaluator:
  cross_window_receipt_sha256: 6b697a6d0355aad525f144d6853dda8362c2b9876694326139d27f85499932e6
  n64_receipt_sha256: d531585f7a46d23d3cec31249e49ee9508892b757f8ab86365554ea30b32a1f0
  n256_receipt_sha256: 862f0e9eb83a60775e42e5ec20770efccc15f13c4954b7e9434dd9dfd10be7bf
authority: macOS CPU frozen SegNet/PoseNet advisory only
score_claim: false
```
