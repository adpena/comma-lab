# Canonical equations draft — DDM V9 carrier composition

**Status:** DERIVED equation draft from measured receiver contracts; no new score law is promoted here.

## E1 — exact contest action, advisory instantiation

For exact final archive bytes `B`, frozen evaluator distances `D_seg` and `D_pose`, and source bytes `N=37,545,489`,

`S = 100 D_seg + sqrt(10 D_pose) + (25/N) B`.

The local rows instantiate this equation but remain `[macOS-CPU frozen-scorer advisory]`, `score_claim=false`.

## E2 — semantic carrier merge

Let `M_k(x,t; z_k, ξ)` be the decoder-rasterized mask for canonical class ID `k∈{0,…,4}`, and let `π=(Undrivable,Road,Lane,MyCar,Movable)` be the governed overwrite order. The final class field is

`C(x,t) = fold_overwrite_π(C_base(x,t; ξ), M_k(x,t; z_k))`.

No luma sort or inferred class permutation is allowed. The archive binds the fixed IDs `Road=0, Lane=1, Undrivable=2, Movable=3, MyCar=4`; receiver validation fails closed on drift.

## E3 — chart-symbol correction, not pixel paste

For a decoded Lane chart coefficient vector `a_{t,j}`, a counted symbol `(t,j,q,δ)` produces

`a'_{t,j,q} = a_{t,j,q} + fp32(δ)`,

followed by the generic Lane rasterizer

`M'_Lane(x,t) = 1[ AA-SDF(x; a'_{t,*}, ξ_t) ≥ 1/2 ]`.

The payload cannot name an `(x,y)` site or an RGB value. Thus one coefficient delta moves a region-coherent band and does not recreate the v8 sparse-pixel ERF failure.

## E4 — birth/death event exception alphabet

For transported carrier state `z_{t+1}^- = T_{ξ_t}(z_t)`, the only authorized sparse temporal exception family is

`z_{t+1} = z_{t+1}^- ⊕ E_{t+1}`,

where `E` contains semantic birth/death/split/merge or phase-reset symbols. It does not contain per-frame RGB residual sites. This landing reuses existing nested event streams; a new jointly optimized xi-event stream remains owed.

## E5 — exact rate and unique homes

For canonical outer ZIP homes `H_i`,

`B = Σ_i |H_i|`, with `H_i ∩ H_j = ∅` for `i≠j`.

Nested attribution reports exact homes inside `predictor.zip` but does not add them again to `B`. Pose6 has one nested home only. The outer archive's parser proves `Σ_i |H_i| = len(archive)`.

## E6 — surgical reverse-waterfill admission

For candidate chart/event DOF `u`, rank by a custodied sensitivity surrogate

`EV(u) ∝ flip_distance(u) × margin_band(u) × curvature(u)`,

with realization predicted by the corrected inner Jacobian/secant law. Admit in descending measured value only while

`-ΔS(u) / ΔB(u) ≥ 25/37,545,489`,

and only if hard semantic-cell and Pose-tube predicates remain green through the exact receiver. An unmeasured arbitrary nonzero symbol is refused.

## E7 — scoped falsifier

The exact instance falsifier is

`F_instance = [B≤154,600] ∧ [D_seg≤0.00116] ∧ [Pose tube green]`.

The measured n64/n256 archives satisfy the byte term but fail the semantic and Pose terms. Therefore the verdict is an **INSTANCE/FORMULATION** negative, not a carrier-family negative. The primary remaining degree of freedom is the joint Fisher-ranked G2CS1 + xi-event solve.
