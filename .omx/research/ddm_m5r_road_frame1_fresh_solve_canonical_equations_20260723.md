# DDM m5r Road Frame-1 Reach-Curve Canonical Equations

Date: 2026-07-23  
Implementation:
`src/tac/canonical_equations/ddm_road_frame1_reach_curve_20260723.py`  
`research_only=true`, `score_claim=false`.

Let `theta` be an integer receiver-wire state and `A(theta)` the exact compiled
archive. Let `R(A(theta), p)` be the decoded frame pair after the receiver's
uint8/resize chain, and let `C(R)` be frozen SegNet argmax cells.

For the Road class id self-derived from the frozen class order,

```text
E_R(theta; P) =
  sum over p in P, pixels x:
    1[label(p,x) = Road] * 1[C(R(A(theta),p),x) != label(p,x)]

Delta_R(theta; P) = E_R(0; P) - E_R(theta; P).
```

The contest-weighted realized objective is

```text
J(theta; P) =
  100 * d_seg(theta; P)
  + sqrt(10 * d_pose(theta; P))
  + 25 * bytes(A(theta)) / 37,545,489.
```

No proxy or Road-only improvement admits a state. At byte box `B`, the exact
admission predicate is

```text
Admit_B(theta; P) =
  [bytes(A(theta)) <= B] and [J(theta; P) < J(0; P)].
```

For a finite exact-replayed set `Theta`, the receiver-closed byte reach curve is
the strictly improving Pareto envelope

```text
Reach_Theta(B; P) =
  max({Delta_R(theta; P) :
       theta in Theta,
       bytes(A(theta)) <= B} union {0}).
```

The stored curve keeps only points whose Road candidate-error count is strictly
lower than every point at an equal or smaller exact archive size. The knee is
the interior point maximizing normalized vertical distance above the chord
between the first and last envelope points. Degenerate spans return no knee.

For `E0` control errors and measured admitted reach `m`, the certified
infeasible-residual interval is

```text
I_residual =
  [E0 - m, E0 - m]  if the reachable set was exhaustive,
  [0, E0 - m]       otherwise.
```

The lower bound remains zero in a non-exhaustive search. Search debt may never
be converted into an infeasibility point estimate.

In this run, the top-24 subset winner had `m=2,017`, but failed full-n600
admission. Consequently full-run credited `m=0`, the Road residual interval is
`[0, 2,210,770]`, and Catalog #366 receives no numeric residual.

Verdict scope:
`INSTANCE:V15_368_RECEIVER_EFFECTIVE_INTEGER_DOF_X_TOP24_PROXY_SCREEN_X_EXACT_RESTRICTED_MASTER_ORDER4_X_C1_200000_BYTE_BOX`.

