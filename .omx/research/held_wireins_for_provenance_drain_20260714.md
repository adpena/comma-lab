# HELD wire-ins — drained equations; one honest QAT lever remains OWED (2026-07-14)

The prior provenance ownership hold has drained. The two independent findings were landed as separate
coherent commits. This ledger remains because `TieredCodeQATLever` is still honestly **OWED_NOT_BUILT**;
its two proposed trainer flags do not exist and were not invented. Trajectory legs remain in `sub015_DAG_*`
(`FEED-bregman-review`, `FEED-fable-AMC`).

## 1. Bregman squared-Hessian dual-metric correction (from bregman_v9_all_surfaces; #500/#501/#504)
- **canonical equation (DRAINED, commit `3dd3ccbf25`):**
  `bregman_dual_metric_squared_hessian_v1`. The dual metric is
  `‖Δη‖² = Δθᵀ H² Δθ` (squared-Hessian), NOT the ordinary
  Hessian metric `Δθᵀ H Δθ = Δηᵀ H⁻¹ Δη`. Fisher-natural dual REQUIRES the typed `H⁻¹` solve. Anchor: the
  measured 600/600 false-equality; max raw-dual/squared-Hessian error
  `9.094947017729282e-13` on `MEASURED_LOCAL_CPU_SYNTHETIC_MATH_FIXTURE_NOT_SCORE`.
- **DSL guard (DRAINED, same commit `3dd3ccbf25`):** `argmax_native_vjp_fidelity_v1` cannot be rebound to a
  no-solve `‖Δη‖` shortcut (name-preserving fake). Fisher-natural cotangent adoption requires a typed
  `H⁻¹` solve; raw dual Euclidean remains scoped to `squared_hessian_H_squared_only`.
- Memo: `.omx/research/codex_premise_falsification_bregman_dual_euclidean_20260714_codex.md`. Memory:
  `dual_metric_no_solve_is_squared_hessian_not_fisher_natural_20260714`.

## 2. Fable AMC per-row tiered code bit-allocation (from Apple warm-start; #406/#336)
- **canonical equation (DRAINED, commit `7f134776c8`):** `amc_perrow_tiered_code_bitalloc_v1` — law:
  pair-local code rows ⇒ per-pair
  d_seg composition is EXACTLY ADDITIVE ⇒ measured-response (per-pair-KKT) allocation DOMINATES proxy-saliency
  tiers. Anchors: the Fable measurement artifact + the 07-13 n600 custody (byte-identical 6/6 sha). Advisory
  axis `[macOS-CPU advisory; NumPy-fp32 receiver; CPU frozen scorers]`; no score or promotion claim until a
  fresh joint n600 row and exact contest-CPU transfer exist. The already-committed measurement tool is the
  producer, so a measurement-tool DSL Lever is N/A (FEED-07l precedent).
- **SOLE REMAINING OWED ITEM — `TieredCodeQATLever` (`OWED_NOT_BUILT`):** the proposed train-time flags
  `--code-row-bits-map` and `--code-qat-tiered` do not exist. Do not invent them. Reactivate only when both
  conditions hold: (1) a competitive witness checkpoint exists; and (2) the operator gives GO to add both
  train-time flags through the typed DSL.
- Full spec: `.omx/research/fable_amc_saliency_codex.md` §8.

## DAG FEED — 2026-07-15 provenance drain

`FEED-held-wireins-drain-v2`: `bregman_dual_metric_squared_hessian_v1` + its no-solve-fake DSL guard
**DRAINED** in `3dd3ccbf25`; `amc_perrow_tiered_code_bitalloc_v1` **DRAINED** in `7f134776c8` on the advisory
axis with the existing measurement tool as producer and DSL Lever N/A. `TieredCodeQATLever` remains the sole
**OWED_NOT_BUILT** item under the competitive-checkpoint + operator-GO typed-DSL reactivation criterion.
Frontier pointer unchanged; no heavy or paid launch occurred.
