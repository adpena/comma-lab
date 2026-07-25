# DDM RS1 feature-relay equation surface

Date: 2026-07-25  
Status: `DERIVED_IMPLEMENTED; EMPIRICAL RELAY PARAMETERS BLOCKED`  
Score claim: `false`

This file records the implemented equations without registering an empirical
relay-radius law. Such a law is forbidden until the missing station bundle is
measured and the end-to-end ladder is realized.

## Station chain

Let `u` be the #580 `range(A)` input actuator and let `z_i` be independent
shooting variables at block2 PRE-SE, block3 PRE-SE, and the rank-4 head.
For measured local segment Jacobians `J_i`, exact linear continuity is

```text
z_1 - J_1 u       = 0
z_2 - J_2 z_1     = 0
z_3 - J_3 z_2     = 0.
```

Two-station chains use the same construction with the final row omitted.

## Fisher-primary objective

For measured target deltas `t_i`, categorical margin-Fisher Grams `F_i`, and
input metric `F_u`, solve

```text
min  1/2 u^T F_u u + sum_i 1/2 (z_i - t_i)^T F_i (z_i - t_i)
s.t. C [u,z_1,z_2,z_3]^T = 0.
```

Euclidean norms are emitted only as labeled controls. They never replace
`F_i`. The KKT system is

```text
[ H  C^T ] [x] = [-g]
[ C   0  ] [λ]   [ 0],
```

solved by the fp64 minimum-norm least-squares solution at the
machine-epsilon-derived rank cutoff. No damping, learning rate, or shrink
constant is introduced.

## Direct comparator

The direct one-shot final-head map is

```text
J_direct = J_3 J_2 J_1
u_direct = pinv(F_u + J_direct^T F_3 J_direct)
           J_direct^T F_3 t_3.
```

It and the relay result are predictions only. Neither can admit a candidate.

## Realized end acceptance

For exact n600 reference and candidate rows after receiver parse-back,
uint8/R, and frozen scorers,

```text
delta_S =
  100 (d_seg_candidate - d_seg_reference)
  + sqrt(10 d_pose_candidate) - sqrt(10 d_pose_reference)
  + 25 (bytes_candidate - bytes_reference) / 37545489.
```

Admit iff `delta_S < 0`. Intermediate station gains do not enter this
decision.

## Equal-budget validity radius

Direct and relay ladders must contain the same unique positive radius quanta
and the same number of realized end verdicts. A method's validity radius is
the largest radius in its contiguous accepted prefix, stopping at the first
rejection. An isolated later acceptance cannot enlarge the radius.

The publish fork is:

```text
relay_radius > direct_radius
  => publish measured ms2r-consumable relay targets
relay_radius <= direct_radius
  => formulation-scoped negative with per-station decomposition.
```

Neither branch is currently reachable because the SHA-bound internal station
targets, Fisher Grams, Jacobians, and continuity secants are absent.
