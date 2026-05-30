# Retroactive sweep for Z7-Mamba-2 Wave N+11 stabilizer anchor 2026-05-30T23:45:00Z

Per Catalog #348 STRICT preflight gate `check_new_gate_landing_includes_retroactive_sweep_evidence` + canonical 2-landing pattern per CLAUDE.md "Bugs must be permanently fixed AND self-protected against": every canonical equation EmpiricalAnchor + sister adapter wire-in landing MUST ship a retroactive-sweep memo evaluating whether prior verdicts on the same bug class need RE-EVAL.

## Bug-class symptom signature

NaN-at-ep-~20 on Z7-Mamba-2 MLX-LOCAL 600pair × 50ep score-aware training due to gradient explosion in the Mamba-2 SSM selective recurrence under AdamW with no clip + no warmup + default weight_decay. The signature: `ema_drift_l2` exploding from 0.0078 (ep0) → 0.895 (ep15) → 6.86 (ep18) → 9.92 (ep19) → NaN crash by ep ~20.

## Pre-fix window

2026-05-28 (Wave N+10 first 50ep attempt at lr=1e-3 with no stabilizer) → 2026-05-30 (Wave N+11 stabilizer landing).

## Historical KILL / DEFER / FALSIFY verdicts searched

Searched canonical posterior + memo registry for verdicts mentioning Z7-Mamba-2 stability / NaN / gradient-explosion / lr-stability classes:

1. **Z7-Mamba-2 anchor #2 (2026-05-28T19:07:14Z)** — `mlx_local_canonical_harness_long_training_real_teacher_lr_stability_falsification_anchor` — verdict `IMPLEMENTATION_LEVEL_FALSIFICATION_lr_alone_insufficient_stabilizer_per_Catalog_307_paradigm_intact`; residual=1.0. **RE-EVAL: NO** — verdict is correct (lr-alone is insufficient; this Wave N+11 anchor confirms the IMPLEMENTATION-LEVEL classification by demonstrating that the FULL stabilizer recipe IS sufficient; paradigm INTACT preserved per Catalog #307). The verdict's "lr alone insufficient stabilizer" claim is empirically reinforced — lr=3e-4 alone failed at ep38 (anchor #2); lr=3e-4 + full stabilizer succeeds at 50/50 (this anchor).

2. **Z7-Mamba-2 anchor #3 (2026-05-28T19:54:00Z)** — `mlx_local_canonical_harness_long_training_real_teacher_lr_reduced_stabilizer_anchor` — verdict `nan_cleared_at_lr_1e-4`; residual=0.0; pose_axis_reduction_percent=19.2. **RE-EVAL: NO** — verdict is correct (reduced-lr ALSO works as a stabilizer; this Wave N+11 anchor shows that the FULL stabilizer recipe lets the operator keep the HIGHER lr=3e-4 by adding canonical primitives instead of having to drop lr). The Wave N+10 RESUME finding (anchor #3 at lr=1e-4) and this Wave N+11 finding at lr=3e-4 are COMPATIBLE — both clear NaN; Wave N+11 keeps the more aggressive lr.

3. **Z7-Mamba-2 anchor #4 (2026-05-29T00:14:43+00:00Z)** — `mlx_local_canonical_harness_identity_predictor_disambiguator_catalog_308_anchor_4_4_wave_n32` — verdict `CONVERGENCE_PARITY` (identity-predictor and Mamba-2 same pose reduction at 600pair × 50ep); residual=0.309. **RE-EVAL: PARTIAL** — verdict's claim "Mamba-2 distinguishing primitive NULL at 600pair × 50ep" was made WITHIN THE UNDERCONVERGED CONFIG (anchor #3's lr=1e-4 + no Wave N+11 stabilizer). The Wave N+11 stabilizer + real-teacher anchor at extended horizon (Operator-routable #1 in landing memo) MAY reveal Mamba-2 distinguishing primitive non-null at higher epoch budgets + higher lr — but ONLY a NEW anchor proves it. The CONVERGENCE_PARITY verdict at the SPECIFIC UNDERCONVERGED CONFIG remains valid; the GENERAL claim is open. Per CLAUDE.md "Forbidden premature KILL": this is a DEFER-pending-research-with-Wave-N+11-stabilizer-extended-horizon-anchor classification, NOT a kill.

4. **Z7-Mamba-2 anchor #5 (2026-05-29T20:47:44Z)** — `audit_source_inspection_plus_webfetch_canonical_paper_reference_plus_test_pinning` — verdict `IMPLEMENTATION_LEVEL_S6_form_not_SSD_form` classification of Z7-Mamba-2 architecture as Mamba-1 S6 (not Mamba-2 SSD); paradigm INTACT. **RE-EVAL: NO** — verdict is correct (Z7-Mamba-2 substrate uses S6 form per the documented adaptation per Catalog #303 cargo-cult audit; the canonical Mamba-2 SSD form lives in `tac.substrates._shared.mamba2_adapter` per task #1539 wire-in). The Wave N+11 stabilizer recipe applies to BOTH S6 and SSD forms (grad-clip + warmup + weight_decay + EMA + smaller dims are SSM-form-agnostic).

5. **Sister Z3 v2 IMPLEMENTATION-LEVEL falsification (slot recurrence)** — searched canonical posterior for sister state-space NaN signatures. **NO sister verdicts** require RE-EVAL; Z3 v2 is a different paradigm (latent-replacement diagnostic, not selective state-space recurrence).

## Per-finding RE-EVAL priority

| Verdict | RE-EVAL | Priority | Action |
|---|---|---|---|
| anchor #2 lr-stability falsification | NO | n/a | preserved as canonical IMPLEMENTATION-LEVEL falsification anchor |
| anchor #3 reduced-lr stabilizer | NO | n/a | preserved as canonical reduced-lr stabilizer anchor (compatible alternative) |
| anchor #4 identity-predictor CONVERGENCE_PARITY | PARTIAL | LOW | DEFER pending Wave N+11 stabilizer + extended horizon + real-teacher anchor (Operator-routable #1 in landing memo); the SPECIFIC UNDERCONVERGED CONFIG verdict remains valid |
| anchor #5 S6-vs-SSD form audit | NO | n/a | preserved as canonical IMPLEMENTATION-LEVEL documented-adaptation audit |
| sister Z3 v2 NaN | NO | n/a | different paradigm; not in scope |

## Net structural protection

- Canonical equation `z7_mamba2_state_space_predictive_coding_pose_axis_savings_v1` now carries 6 anchors with the Wave N+11 stabilizer EMPIRICAL anchor at residual=0.0 + Catalog #371 auto-recalibration triggered.
- MlxScoreAwareAdapter Wave N+11 stabilizer wire-in extincts the "flag accepted but not wired" IMPLEMENTATION-LEVEL falsification pattern at the canonical sister surface (sister substrates that adopt the stabilizer kwargs inherit the same protection automatically).
- No historical verdict is retracted; one verdict (anchor #4) carries a PARTIAL RE-EVAL marker noting the specific underconverged config remains valid AND the general claim requires extended-horizon anchor for closure.

## Cross-references

- `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_z7_mamba2_wave_n11_stabilizer_re_fire_landed_20260530.md` (landing memo)
- `.omx/research/z7_mamba2_wave_n11_stabilizer_600pair_50ep_20260530T233603Z/` (empirical anchor artifact)
- `.omx/state/canonical_equations_registry.jsonl` (canonical equation registry with new anchor + recalibration event)
- Catalog #307 paradigm-vs-implementation classification
- Catalog #344 canonical equations registry
- Catalog #371 auto-recalibrator
- Catalog #348 (THIS gate's parent META-meta gate)
