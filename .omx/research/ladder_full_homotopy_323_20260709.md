# #323 — FULL LADDER island-birth lever: the per-class-λ-gated homotopy [no-triality]

**Date:** 2026-07-08. **Task:** #323 (build-class → Opus). **Axis:** `[macOS-MLX advisory]` —
NON-PROMOTABLE. Pointer contest-CPU **0.19110 UNMOVED** (MEANS, not a score row; only a
byte-closed `evaluate.py` n600 exact row moves it). This lever is DEFAULT-OFF and is registered
in the duty-to-measure queue — its ΔS is OWED, not claimed.

## What was built
The FULL LADDER island-birth lever the amplify/nucleus machinery only PARTIALLY realized: a
per-epoch, per-class SUPPORT SCHEDULE over the eased-island radius, GATED by the measured
per-class costate λ_c. Goes into the RESTART config (run-1 at
`experiments/results/levelset_n600_crucible_v6_run1_*` was READ-ONLY; not touched).

- **Pure control law** `src/tac/witness_curriculum/ladder_homotopy.py` (`LadderHomotopy` +
  `LadderArmSpec` + `homotopy_from_flags` + `perclass_lambda_proxy`). Deterministic numpy/float,
  holds NO training state.
- **Movable arm — dilation-GO**: SDF forward-Euler dilation (proven transfer, 1-Lipschitz),
  support ceiling'd by the critical-nucleus RELEASE law r*(t) = coeff·σ_eff (LawRef
  `critical_nucleus_release_v1`, consumed via `tac.canonical_equations.evaluators`; MEASURED
  dilation knee native 44.6% → +1px 90.0% → +2px 98.3%). λ-gate defaults OPEN (dilation-GO is
  sound independent of lane-share, per the T3 symposium split).
- **Lane arm — curve-prior**: grows support ALONG the openpilot VP-tangent (`oriented_width_eased`;
  stays on the ~8-dim lane manifold — isotropic dilation of a curve is the measured NO-GO) with a
  dash-phase window; NO isotropic nucleus ceiling (lane's barrier is area/margin, released by the
  schedule + λ-gate).
- **Per-class-λ gate**: support flows to a class ONLY while its MEASURED costate
  λ_c = flip_share_c·d_seg_by_class_c (`tac.witness_control.perclass_verdict`, the #315 sensor)
  exceeds `lambda_gate`; the soft-gate band (`gate_softness`) fades support out CONTINUOUSLY as the
  class's residual is won. **UNIFORM always-on amplification is the MEASURED net-negative
  anti-pattern (Δd_seg ∝ n_big3 − n_isl) and is NEVER emitted** — the gate withdraws support the
  moment a class is won.
- **Homotopy schedule**: eased-FIRST (support r0 = winnable variant), held, then annealed r0→0
  (transfer to the true argmax), per the CT-2 §6 / DRAFT r(t) continuation form. The physical
  0→r0 ease-in and the gate-close ease-out are the 1-Lipschitz stepper's job
  (`step_radius`, pseudo-arclength Δr ≤ max_step_px) — NO hard switch (continuation, not a binary).
- **Trainer wiring** (`experiments/train_levelset_witness_realized_through_R_mlx.py`): 16 new
  `--ladder-*` flags (default-OFF master `--ladder-island-homotopy`); modulates the AMPLIFY
  island-birth masks only (`--amplify-weight` > 0; auto-forces class-aware eased masks). Per-epoch
  refresh recomputes per-class rungs from the schedule + last verdict's λ_c and rebuilds the amplify
  masks IN PLACE only on an integer rung CHANGE (bounded cost). The SEED keeps its own
  `--seed-anneal-epochs` transfer schedule → **no double-application**.
- **Compose surface** `eased_island_masks(..., lane_px=, movable_px=)`: optional per-class radii
  (default None → shared `dilate_px` → byte-identical single-radius call).
- **DSL** `LadderIslandHomotopy` Lever factory (`tac.witness_dsl.curriculum_dsl`) — 18 flag
  overrides, LawRef-anchored constants; `lever_registry.completeness()` maps all `--ladder-*` flags
  (0 unmapped); auto-surfaced never-fired / duty-to-measure via `known_levers()`.

## Default-OFF byte-identity (the safety guarantee)
When `--ladder-island-homotopy` is unset (default): `_ladder_state` is None; the setup `if _ladder_on`
block, the verdict-row λ-capture, and the epoch-top refresh are ALL guarded and skipped; the amplify
masks use the fixed `--island-dilate-px` exactly as the pre-#323 trainer. PROVEN at the shared
surface: `eased_island_masks` with `lane_px=movable_px=None` is `array_equal` to the single-radius
call (test). The only compiled graph is R (`maybe_enable_mx_compile_r`), which does not capture the
island weights, so the mid-run in-place mask rebuild is correctly seen by the loss.

## Tests (22, all green) + regressions (168 sibling tests green)
`src/tac/witness_curriculum/tests/test_ladder_homotopy.py`: gate-closed⇒zero-support · soft-gate
fractional+continuous+monotone · ungated at λ_gate=0 · eased-first schedule · schedule 1-Lipschitz ·
movable release ceiling consumes LawRef + shrinks with σ_eff · lane no-isotropic-ceiling · lane
dash-gate window · stepper up/down cap · rung rounding · λ proxy · param guards · eased-mask
per-class byte-identity + independent growth · DSL factory spelling + never-invent-flags +
registry mapping. Sibling suites (curriculum_dsl / lever_registry / eased_targets /
island_protection / activation_ledger) — 168 pass, no regressions.

## Verdict scope on negatives consumed (req R)
- "UNIFORM amplification net-negative" (full-stack 0.121 / paint-seed 0.026) = **FORMULATION-level**
  (uniform, un-gated support) — this lever is the reformulation (margin/λ-GATED) the T3 symposium
  designed as net-positive by construction. Not a family kill.
- "isotropic dilation of a lane curve = NO-GO" = **FORMULATION-level** (isotropic operator); the lane
  arm uses the along-VP-tangent operator, which is NOT under that scope.

## STORES CONSULTED
`corpus_query` "LADDER island birth per-class lambda homotopy" + "islands treatment arm" ·
`.omx/research/council_t3_symposium_islands_treatment_arm_20260706.md` (REVISE + gating conditions:
net-positive iff n_isl > n_big3; uniform is measured net-negative) ·
`.omx/research/tufa_duck_harness_ladder_costate_synthesis_20260706.md` (LADDER ⊂ our costate;
per-class λ IS LADDER's verifier signal) ·
`.omx/research/t5_crucible/ct_deepresearch_2_pde_geometric_topological_control_20260707.md` §5–§6
(critical-nucleus release r*=0.95·σ_eff; fold-advance/continuation step law) ·
`ORCHESTRATION_LEDGER.md` requirement C · existing trainer machinery (`seed_islands`/`amplify`/
`seed-island-eased`/`eased_island_masks`) · `lawref_builtins.critical_nucleus_release_v1` +
`canonical_equations.evaluators.eval_critical_nucleus_release_r_star` ·
`witness_control.perclass_verdict` (#315 per-class λ sensor) · `lever_registry` / `activation_ledger`.

## Owed (duty-to-measure — NOT claimed)
The lever's d_seg ΔS is OWED via a byte-closed n600 exact row in the RESTART config; run-1 records
the trace. Run-2 refinements (noted, not built): live σ_eff(t) from the τ schedule (currently the
DERIVED-AT-CONFIG flag constant, req-T) feeding the movable release ceiling; and the CT-2 fold-advance
birth-scheduler db_c/dw_c fit from the recorded per-class dμ_c/dt trace.
