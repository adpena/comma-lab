# FEED-603 — DDM V16 coupled joint scorer-margin solve

**research_only=true · score_claim=false · MAIN review required**

```text
v15 counted receiver control
  -> canonical class contract (Lane=1, Movable=3)
  -> shared 2x2 template DOF + measured sparse collateral supports
  -> counted receiver -> exact R -> frozen SegNet/PoseNet VJP
  -> local M (151 x 141)
  -> conditional level-set KKT {first-order, Gauss-Newton}
  -> Hessian-metric Babai uint8 projection
  -> counted receiver + frozen-scorer remeasurement
  -> hold control selected in rounds 1 and 2
  -> eight-island -> n64 -> n600 ladder
  -> fork C: M/costate signal for #366; no dispatch or promotion
```

## Triality and solver-stack feed

1. **DSL/config:** strict local-only config SHA-binds v15 n64/n600 receipts and archives, target
   cache, canonical representative islands, trust boxes, scorer batch geometry, false-authority
   flags, and immutable output stages.
2. **DAG/receiver:** the v16 receiver extends v15 with counted pair-template phases and counted
   sparse signed RGB compensation. Parse/re-encode and member-home accounting close; no scorer,
   logits, gradients, or GT argmax table enters decode.
3. **Equations:** the adjacent JSON/Markdown equation artifacts and executable law distinguish
   exact conditional equations from modeled GN curvature, bounded Babai projection, searched phase
   switches, and real-receiver admission.
4. **Sensitivity map:** M is the local signed-margin response of all active target/protected/pose
   rows to 141 counted DOF. Its NPZ bytes and activation-pattern hash are preserved.
5. **Pareto/bit allocator:** every proposal is measured on Seg, Pose, and exact bytes. Twelve sparse
   compensation records added 120 program bytes but still worsened the receiver result; zero bytes
   were admitted.
6. **Continual learning/probe:** the negative realized correlations and unclean KKT residuals route
   M to #366 only as an instance-scoped costate/preconditioner signal. They do not close nonlinear
   joint training or the representation family.

## Corrected measured ladder

- Eight islands: `135,328 B`, `d_seg=0.025053024292`, Movable `0.133435387167`, Lane
  `0.515966914459`.
- n64: `59,875 B`, `d_seg=0.041460116704`, Movable `0.667507693326`, Lane
  `0.496831048153`.
- n600: `135,328 B`, `d_seg=0.027470296224`, `d_pose=163.061327281443`, Movable
  `0.291615222639`, Lane `0.435195521828`; all 600 camera outputs are byte-identical to v15.

## Invalidation and fork

The recovered first receipt mapped Lane to class 4 (MyCar). It is preserved but machine-readably
invalidated and must not feed M, Lane conditionals, the fork, or #366. The canonical Lane=1
supersession fires fork C: `INSTANCE_VALIDITY_RADII_COSTATE_PRECONDITIONER_FOR_CATALOG_366`.

STORES CONSULTED: `CLAUDE.md`; `AGENTS.md`; `PROGRAM.md`;
`docs/operating_manual_craft_handoff.md`; v7.5/v8 specs; v14/v15 receipts and equations; #549/#391
joint-inverse and exact-R surfaces; #341/#423 GN/Hessian surfaces; #532/#586 uint8 lattice surfaces;
`reports/latest.md`; lane/progress registries; operator directives through 2026-07-23T00:31:59Z.

Pointer `0.1910828242 [contest-CPU]` unchanged.
