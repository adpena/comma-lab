# BASIS-INTEGRATION v7 — DirectionalBasisRebalance(lane_offloaded) IN-v7 (2026-07-08)

[no-triality]  (this memo is the record; the triality legs are DSL + equations + DAG, touched in the same commit)

STORES CONSULTED: `.omx/research/t5_crucible/SYNTHESIS_seal_v7_round1_20260708.md` R-1 disposition ·
`src/tac/witness_dsl/curriculum_dsl.py:2427` (`DirectionalBasisRebalance` factory + its two-regime
docstring) · `src/tac/canonical_equations/anisotropic_basis_two_regime_allocation_20260707.py`
(`freq_along_for_regime` law) · `tools/memory_waterfill_config.py` + `tools/witness_memory_preflight.py`
(#294 waterfill / `project_peak_rss_gib` / `derive_in_feat_from_flags`) ·
`experiments/train_levelset_witness_realized_through_R_mlx.py:2846` (`dir_w = 4 * n_dir_freqs`) ·
`src/tac/boundary_math/lever_b_levelset_generator.py` (curvelet bank / in_feat) · run-1 v6 launch.sh
(`levelset_n600_crucible_v6_run1_20260708T095730Z`, basis ground truth: n_dir_freqs 2 / freq_across 32
/ freq_along 4 / max_bank_freq 64) · L25/L65 memory (−48% directional MEASURED; 3.2× along deficit) ·
`src/tac/witness_autoconfig.py` `_build_crucible_v7` / `_CRUCIBLE_V7_DSL_LEVERS` / `CrucibleV7LaunchConfig`.
review_status: builder self-review + 2 review-tracker passes; fresh-eyes on the waterfill derivation.

## THE APPROVED DECISION (operator, 2026-07-08)
The R-1 basis recommendation goes IN-v7. Evidence chain: measured 3.2× along-tangent deficit (L65) +
the blind derivation's independent √32≈6 minimum (seal_v7_r1_structure_blind §PHASE-1/2 R-1) +
basis-before-capacity law (−48% directional MEASURED) + the crucible's Arm-A mission lever built-never-
fired. The operator approved the REC; the waterfill picks the NUMBER.

## THE GATING MATH — WATERFILL FIRST
The cf-feature bank memory scales with `in_feat`. The MEASURED mechanism (trainer + preflight, not
assumed): the directional (self-orient) feature WIDTH is `dir_w = 4 · n_dir_freqs` — it depends on the
COUNT `n_dir_freqs`, NOT on the `freq_across` / `freq_along` VALUES (those set the frequency content, not
the column count). So:
- `in_feat = 2·curvelet_cols + 4·n_dir_freqs` (self-orient). Curvelet cols track the bank_* flags +
  max_bank_freq (unchanged by this lever).
- A minimal along-only rebalance (freq_along 4→6 or 4→8, n_dir_freqs held at 2) is **MEMORY-NEUTRAL**
  (in_feat 88 either way).
- The DSL lever lane_offloaded (n_dir_freqs 2→4, freq_along=6) grows in_feat **88→96** (+8 = 4·(4−2)).

WATERFILL TABLE (all peaks from the REAL `project_peak_rss_gib`, n600, render 384×512, verdict_batch 32,
render_aa ipe, 128 GiB machine; envelope = safe_frac × 128; control-plane floor 10 GiB):

| candidate                                   | n_dir | freq_along | in_feat | cf_cache | peak GiB | env@0.70 | env@0.85 | verdict |
|---------------------------------------------|------:|-----------:|--------:|---------:|---------:|---------:|---------:|---------|
| run-1 / v6 baseline                         |     2 |          4 |      88 |   43.20  |  67.61   | SAFE     | SAFE     | baseline |
| minimal along-only (4→6 / 4→8)              |     2 |        6/8 |      88 |   43.20  |  67.61   | SAFE     | SAFE     | memory-neutral |
| **DSL lever lane_offloaded (CHOSEN)**       |     4 |          6 |      96 |   47.13  |  **71.54** | **SAFE (margin 18.1)** | **SAFE (margin 37.3)** | **ADMITTED** |

Deltas vs baseline: cf_mx_cache **+3.93 GiB**, peak **+3.93 GiB**. Envelopes: 0.70→89.6 GiB, 0.85→108.8
GiB; floor OK (128−108.8=19.2 ≥ 10). The lane_offloaded allocation FITS with the standard margin under
BOTH the conservative concurrent (0.70, 18.1 GiB margin) and the sole-workload (0.85, 37.3 GiB margin)
fractions — so per the priority the DERIVED DSL lever is preferred as-designed. No fall-through to a
minimal rebalance; the derived form (Candès–Donoho parabolic along=√across=6 in the lane_offloaded
regime) is exactly what the equation prescribes.

## THE CHOSEN ALLOCATION + DERIVATION
`DirectionalBasisRebalance(regime="lane_offloaded")` → emits `--self-orient` (already True in base →
no-op override), `--n-dir-freqs 4`, `--freq-across 32.0`, `--freq-along 6.0`. `freq_along = 6` is
`max(4, round(√32))` from `freq_along_for_regime(32, "lane_offloaded")` (the equations-leg law) — NOT a
hand number. NO bare constants: every value carries its waterfill/equation derivation via
`witness_autoconfig.crucible_v7_basis_allocation_provenance()` (re-derives peaks from the real preflight).

Note — lane_offloaded (along=6), NOT lane_carried (along=26): the lane rides the FREE rule-118 analytic
band (MEASURED lane d_seg 0.00087), so the boundaries the witness still carries are C²-cartoon edges →
parabolic scaling. The 3.2× dash-comb deficit is the lane_CARRIED regime; it justifies the DIRECTION of
the rebalance (raise along), and its magnitude sets the ceiling, but the derived lane_offloaded optimum
is √32≈6 (over-allocating to 26 would spend the bank on dash frequencies the analytic band already
carries).

## PROJECTED STEP-TIME / WALL-CLOCK
in_feat 88→96 (ratio 1.091) scales only the in_proj first-layer matmul + the per-pair cf-feature forward
— a small fraction of the total step (SDF hidden layers + verdict dominate), so total step cost ≲ few %.
Even the full 9.1% in_feat cost is inside the 15% wall-clock slack (`derive_wall_clock_budget_days`
anchor × 1.15), so the DERIVED wall-clock budget need NOT change; window=0 → no epoch delta. The rc=8
launcher gate verifies at admission with the REAL bench (the projection is advisory).

## ACTIVATION-LEDGER EVENT
The lever is now on `CrucibleV7LaunchConfig.dsl_levers` (4 levers: the 3 v6-inherited spine levers +
`FEED_07a_directional_basis_rebalance`). At the NEXT real launch, `tools/launch_witness_run.py:1462`
records the lever's FIRST `fired` event — Arm-A (built-never-fired) finally fires. NO fake fire recorded
now (no launch; the ledger's `fired` = "a run launched with this lever").

FINDING (not fixed — out of basis-block scope, flagged for the launch-path/ledger owner): the launcher
records the Lever.name (`FEED_07a_directional_basis_rebalance`) while `activation_ledger.never_fired()`
keys on the FACTORY name (`DirectionalBasisRebalance`). This Lever.name↔factory-name mismatch affects ALL
crucible DSL levers equally (pre-existing, systemic), so the costate digest's never-fired list may not
clear on launch. Reconciling the two naming surfaces is a shared ledger/registry fix, not a basis-block
hunk — recommend the launch-path fixer or a follow-up unify them.

## TRIALITY
- DSL: `DirectionalBasisRebalance` factory + gauge component `ALONG_TANGENT_FREQ` already exist (this is
  a CONSUMPTION, not a lever change); `_CRUCIBLE_V7_DSL_LEVERS` + the config levers tuple updated.
- equations: `anisotropic_basis_two_regime_allocation_v1` gains the applied-at-config consumer
  `tac.witness_autoconfig` + an `applied_at_config` note on the parabolic anchor. Status STAYS
  `ASSUMED_AWAITING_VERIFICATION` — the A/B verdict is the run itself.
- DAG: FEED-08b row appended.

## VERIFICATION
36/36 `test_crucible_v7_config.py`; sisters green (v7_compute_exploitation, revisions_b, wallclock,
memory_waterfill, witness_memory_preflight, feed07_dsl_wirein, domain_priors); ruff F clean.

means != ends: this wires a MEANS. Only a byte-closed n600 exact row < 0.19110 from
`upstream/evaluate.py` (contest-CPU/CUDA, NEVER MPS) moves the pointer 0.19110 (UNMOVED).
