---
task: 603
feeds_task: 613
research_only: true
triality_leg: equations
main_landing_review_required: true
---

# DDM v4 structured-member canonical equations

Let `s` denote a configured member role, `k` an evaluator class, `R` the exact receiver chain,
`A_s` the deterministic archive compiler, `x` the bound S4 source state, and `T` the bound target.

## Exact rate and receiver closure

```text
B_s = |A_s(x)|
R_s = R(A_s(x))
A_s(parse(A_s(x))) = A_s(x)
```

The final equality is enforced byte-for-byte.  It forbids appended ZIP payloads and ambiguous byte
homes even when a permissive ZIP reader would otherwise accept them.

## Scoped membership matrix

For frozen local SegNet argmax `C(.)`, class mask `1_k`, and measured frame set `F_64`:

```text
M[s,k] = sum_{f in F_64} |1_k(C(T_f)) intersect 1_k(C(R_s,f))|
         / sum_{f in F_64} |1_k(C(T_f)) union 1_k(C(R_s,f))|
```

`M[s,k]` is a representation-membership instrument, not d_seg and not a contest score.  The
measured positive rung is `M[Lane, Lane] = 0.686570162333` with `B_Lane = 84,918`.

## Pose completeness

```text
P_s = (# required two-frame Pose inputs materialized by R_s)
      / (# required two-frame Pose inputs in F_64)
```

All six measured roles have `P_s = 1`.  Completeness is necessary but says nothing about d_pose.

## Admission and blocker logic

```text
full_admit(s,k) := B_s <= B_box and M[s,k] > 0 and P_s = 1
representation_escape(s,k) := B_s <= B_box and M[s,k] > 0
```

The Lane role satisfies both local predicates.  Neither predicate authorizes promotion: exact
evaluator d_seg/d_pose, archive custody on contest hardware, and the governed promotion gates are
still owed.

## Event subset price

For class-filtered event prefix `E_{k,n}`:

```text
Q(k,n) = |encode(E_{k,n})|
Delta_Q(k,n_i,n_j) = Q(k,n_j) - Q(k,n_i)
```

The measured `Q` curve is the bit-allocator input.  `Q(Movable,n)=62` for all measured prefixes
because the first 64 frames contain zero Movable PCE3 records; this is a scoped empty-support
result, not a dead-family verdict.

