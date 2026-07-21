# FEED-costate-organ-v2-exact-anchor — DAG feed (2026-07-21T01:59Z)

**Lane:** `p0_costate_organ_factorization_grounded_ABC`  
**Status:** `BUILT`, `RETROSPECTIVE-DEVELOPMENT-BACKTEST-PASS`, `research_only=true`  
**Authority:** advisory-only, `actuation=NONE`, `_dev`, pointer unchanged, MAIN review required.

```text
realized-through-R current d_seg/d_pose
  + #547 n600 exact anchor
  + fp32-exact canonical support fill
    -> exact_gap [S]

pair + frame + channel + site-space
  + segnet_head_rank4_linear_flipdist_v1
  + exact four-tap resize adjoint
  + separable_resize_full_kernel_direct_sum_v1 (80.6742% nullity)
  + frame0 Seg zero / Pose chroma-2x2 scope
    -> visibility [0,1]

route + uint8/resize/parse-back counts
  + r1b6/r1b7 gate
  + M1 11,453/38,077 design anchor
  + formulation/apparatus validity
    -> realizability [0,1]

realized recovery S + charged bytes
  + realization_breakeven_bytes_v1 (latest=domain_refined)
    -> byte_price [0,1]

exact_gap * visibility * realizability * byte_price
    -> lambda(pair,site)
    -> same-pool cap/order via witness_measured_reverse_waterfill_v1
    -> dual Euclidean + Fisher readback (never blended)
    -> optional xi transport, REFUSE on sparse topology event
    -> maturity _dev/_prod + apparatus readback
    -> shadow report + digest, beside v1
    -> actuation NONE

historical #205 + C2 rows
    -> source SHA before/after
    -> old/v2/realized Spearman + four ablations
    -> PASS 0.569739 -> 0.697826
    -> costate_organ_exact_anchor_product_v2 registry anchor

fixed synthetic scalar LQR (#593 row 2)
    -> state forward / costate backward / bounded control projection
    -> finite-horizon Riccati analytic control and costate
    -> central-difference Hamiltonian dH/dx and dH/du
    -> monotone relaxed-sweep residual
    -> reject non-stabilizing algebraic Riccati root
    -> CONFORMANCE ONLY; no curriculum/live-control authority
```

## Triality legs

- **DSL:** no invented flags; readback consumes typed state/factor inputs only.
- **DAG:** this FEED is the producer→validation→consumer chain.
- **Equations:** `costate_organ_exact_anchor_product_v2` composes
  `segnet_head_rank4_linear_flipdist_v1`,
  `separable_resize_full_kernel_direct_sum_v1`,
  `realization_breakeven_bytes_v1`, and
  `witness_measured_reverse_waterfill_v1`. Dedicated opportunity-pool law remains
  `FORMALIZATION_PENDING_NOT_REGISTERED`.

## Mandatory six-hook wire-in

1. **Sensitivity map:** pair/site lambda is a typed marginal producer; absent site custody refuses.
2. **Pareto constraint:** exact score debt and rate rent remain separate factors; same-pool gains do not add.
3. **Bit allocator:** consumes the domain-refined break-even threshold and KKT marginal ordering.
4. **Cathedral/autopilot:** advisory row only; `_prod` necessary, operator/governor authority unchanged.
5. **Continual learning:** 24-row backtest plus ablation table are the durable calibration anchor.
6. **Probe disambiguator:** Euclidean/Fisher, xi/no-xi, and camera/scorer visibility are separately emitted.

The Pontryagin/LQR fixture is an additional offline conformance guard around the
adjoint sign/projection convention. It is not a seventh actuation hook and cannot
formalize the nonlinear training plant.

## Apparatus and scope

- `--ckpt-every 1` rows are flagged/excluded.
- EMA de-lag is applied only with explicit reset/lag custody; otherwise `unknown_not_applied`.
- Fisher bank full SHA/schema are checked; aggregate rows do not fabricate a site-level Fisher rank.
- C2 flat-amplitude negative is formulation-scoped; curvelet/shearlet/joint-trained forms remain open.
- No run, checkpoint, pointer, process, config, scorer, or archive was mutated.
