# EN1 Receipt - Engineer The Inert

Date: 2026-08-05. Axis: `[macOS-CPU scorer-free byte-only]` unless stated otherwise. Score claim: false. Scorer forwards: 0. `upstream/evaluate.py`: not run.

## Answer First

EN1 built the missing tau consumer for `tr1_seg_margin_weight`, priced XO1 as a context coder against the live Brotli-Q11/IX2 token stream, calibrated the live entropy rate surrogate locally, and swept the 08-01 to 08-05 inert-dropped queue into build/fire-order rows.

| leg | denominator | result | disposition |
|---|---:|---|---|
| 1 tau margin weighting | 4 argparse start forms + structural tau branch proof | `tau_softplus` now consumes `_live_margin_weight(...)`; only `l7_softplus` remains inert by design | BUILD LANDED; race at next clean window boundary |
| 2 XO1 context coder | 600 pairs, token lattice `(600,24,32,4)`, 10 context rows | best optimistic context split: `340,735 B`, `-560 B` vs live forced Brotli-Q11/IX2 `341,295 B` | BELOW WEAK-GO and not receiver-causal as implemented |
| 3 rate surrogate calibration | 20 deterministic perturbations of b4s/window_02 token field | entropy vs forced IX2/Brotli Spearman `0.9981`, Pearson `0.9926`, sign `18/20`; entropy vs SMEVR Spearman `1.0`, sign `19/20` | local calibration OK; does not overturn rsf1 trajectory warning |
| 4 inert sweep | 9 rows | each row has engineering gap, build/exclusion, and fire order | queued/folded below |

Own vehicle frontier unchanged: `S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.`

## Leg 1 - Tau Consumer

Built in:

| file | change |
|---|---|
| `experiments/train_witness_realized_through_R_mlx.py` | `tau_softplus` now multiplies its softplus penalty by `_live_margin_weight(seg_logits, margin_weight_fn, temp_use)` under `if apply_mw:`. |
| `experiments/train_tr1_partition_renderer_mlx.py` | honoring set now includes `tau_softplus`; guard accepts `ce -> tau_softplus`, `tau_softplus`, `unify_tau`, and `margin_hinge`; still refuses `l7_softplus`. |
| `src/tac/witness_dsl/pt2_ported_levers_20260803.py` | retired-trainer port notes no longer cite `--margin-weighted-loss` as tau-inert. |
| `src/tac/tests/test_ddm_tp2_tr1_form_predicate_and_margin_gate.py` | regression proves `tau_softplus` has exactly one `apply_mw` block and that block calls `_live_margin_weight`. |

No new constant was introduced. The existing DSL lever is `spec_tr1_renderer_20260728.lever_seg_margin_weight(temp=1.0)`, with trust rung `RACED-NOT-ASSERTED`; EN1 only makes its declared path real on the live tau form. RT2 recall changed the implementation plan: do not invent a floor or generic weight; use the existing hinge-weight function and existing temp path.

Fire order: one-variable A/B at the next clean window boundary, `tr1_seg_margin_weight` ON vs OFF, same seed/schedule/checkpoint custody, matched-epoch read, no mutation of the burning run, and no retroactive claim on b4s or r1c.

## Leg 2 - XO1 Context Coder

Baseline is the live `sub_auto_pairbit` token stream as IX2/forced Brotli-Q11, not SMEVR.

| baseline | bytes | notes |
|---|---:|---|
| IX2 auto token frame | 341,295 | identical sha to forced Brotli-Q11 in this stream |
| forced IX2 Brotli-Q11 token frame | 341,295 | residual 339,970 B + base 1,297 B + 28 B header/magic |
| R7 SMEVR comparison | 346,478 | comparison only, not the bar |
| R7 Brotli11 comparison | 396,442 | comparison only |

XO1 input packet: `201 B`, sha256 `e2a48e785e05615ddad913eb2d8673452c1b748eccf7c2be66f04ea0b0f5a450`. Context score source: dequantized XO1 additive named-feature head over cached `cx1_argmax_n600` and token activity; no scorer forwards.

| context mode | contexts | total bytes | delta vs forced Brotli-Q11 | residual Brotli bytes |
|---|---:|---:|---:|---:|
| cell score | 2 | 340,735 | -560 | 339,197 |
| cell score | 4 | 341,362 | +67 | 339,808 |
| cell score | 8 | 341,685 | +390 | 340,099 |
| cell score | 16 | 344,223 | +2,928 | 342,573 |
| cell score | 32 | 348,742 | +7,447 | 346,964 |
| pair score | 2 | 347,048 | +5,753 | 345,510 |
| pair score | 4 | 354,161 | +12,866 | 352,607 |
| pair score | 8 | 365,475 | +24,180 | 363,889 |
| pair score | 16 | 372,036 | +30,741 | 370,386 |
| pair score | 32 | 379,780 | +38,485 | 378,002 |

Verdict scope: FORMULATION, optimistic context split by XO1 score. Best row is positive by 560 B, but it is below the EU2 weak-go bar of `15,000 B` and below PA2's projected `19,543-28,179 B` context floor. It is also not receiver-causal as implemented: the context map uses cached rendered argmax/token-derived XO1 scores that are not available before token decode. A shippable version would need a decoder-legal context feature path and would need to beat 15 KB after counting its map/features.

Persisted evidence:

| artifact | bytes | sha256 |
|---|---:|---|
| `.omx/research/ddm_en1_20260805/en1_xo1_context_coder_measurement.json` | 9,034 | `73dba20e42a78523c27686691b99e4ae5f4c329761bbcd5e027a6f8c8f01edfd` |

## Leg 3 - Rate Surrogate Calibration

Live b4s/window_02 config fields:

| field | value |
|---|---|
| `w_rate` | `0.05` |
| `rate_model` | `entropy` |
| `byte_ledger_coder` | `smevr` |
| `token_temporal_mode` | `shared_base` |
| `token_quant_levels` | `16` |
| `token_cell_mask` | `/Volumes/VertigoDataTier/pact/ddm_sg1_20260731/qa24_grid_keep_mask_50.npy` |
| `token_rowband_spec` | `None` |

The in-loss surrogate being priced is trainer `rate_model=entropy`: marginal soft-histogram entropy over kept token values. Perturbations were deterministic token-field edits around `/Volumes/VertigoDataTier/pact/ddm_b4s_20260731/window_02/checkpoints/stage_seg_trunk_tau_final.npz`.

| baseline | value |
|---|---:|
| entropy bits | 2.9879976830050925 |
| forced IX2/Brotli-Q11 bytes | 274,943 |
| SMEVR bytes | 272,578 |

| agreement | value |
|---|---:|
| entropy vs forced IX2/Brotli Spearman | 0.9981196644178428 |
| entropy vs forced IX2/Brotli Pearson | 0.9926421004078082 |
| entropy vs forced IX2/Brotli sign agreement | 18/20 |
| entropy vs SMEVR Spearman | 1.0 |
| entropy vs SMEVR Pearson | 0.9987497692154191 |
| entropy vs SMEVR sign agreement | 19/20 |

Representative rows:

| row | changed scalar tokens | delta entropy | delta IX2/Brotli B | delta SMEVR B |
|---|---:|---:|---:|---:|
| mode_snap_01 | 1,843 | -0.0021074754 | -182 | -250 |
| mode_snap_10 | 18,432 | -0.0218620906 | -3,015 | -2,825 |
| jitter_pm1_10 | 7,568 | +0.0111991663 | +2,016 | +1,864 |

Verdict scope: local quantized-token perturbation neighborhood. This says the live entropy surrogate is directionally sensible for small local token-value edits. It does not overturn `.omx/research/ddm_rsf1_rate_surrogate_fidelity_20260801.md`, which measured trajectory/regime anti-correlation under training dynamics. Cure route if the next window shows trajectory mismatch again: use rg5-style affine refit or swap the rate model at a clean window boundary, never mutate the burning run.

Persisted evidence:

| artifact | bytes | sha256 |
|---|---:|---|
| `.omx/research/ddm_en1_20260805/en1_rate_surrogate_perturbations.json` | 12,488 | `281324c102bc285b59df4b0a157982866a1c22f73c02b9d0c15784d8f4c4ea90` |

## Leg 4 - Inert-Dropped Sweep

| item | engineering gap | build or derived exclusion | fire order |
|---|---|---|---|
| `tr1_seg_margin_weight` / #925 / #888 | Flag existed, but `tau_softplus` did not consume it. | BUILT in EN1; existing DSL lever now has a real tau consumer. | Single-variable A/B at next clean window boundary. |
| fh1 R1+R4 tie-locus edge weighting | `TieLocusEdgeWeighted` was designed-stub text; no trainer loss/provider consumer. | Build `make_loss_fn` provider and per-class-pair edge field, or exclude only after a matched consumer A/B. | Post-knee placement-pool A/B after TP1 boundary, with R13 placement-vs-area telemetry. |
| fh1 R2 xi-advected token base | Live `shared_base` is xi=0 approximation; no token-grid warp/receiver decode path. | Build token-grid advection and parse-back, or exclude if QA90 says token coherence is absent. | Rate-axis A/B only after decoded-token parity. |
| fh1 R3 margin-satisfice cap | Satisfice cap allocator was not a branch in `_live_margin_weight`. | Build allocator branch sharing the margin-weight consumer; no separate constant without RT2/hinge recall. | Race against inverse-margin weighting after EN1 consumer lands. |
| fh1 R5 renderer-weight rate force | Real missing rate term, but measured pool was small, about `0.0022 S`. | Derived low-priority exclusion from immediate score-moving path; not a family death. | Queue only if renderer bytes become binding after token/address work. |
| fh1 R6 class-weight-lane | Was never-fired in fh1; b4s later fired it. | Not inert now. b4s W02 supported `R6_PAYS` with attribution caveats. | Fold into TP1/birth reads; do not relabel as inert. |
| lg1 lane-guard lambda dual | Wired, but b4s kept `lambda=0` because the level constraint was slack. | This is correct KKT behavior, not lever death. Gap is topology false-positive and rollback policy. | No lambda raise unless dual violation `g > 0`; tune/HT-test topology guard before using it as terminal stop evidence. |
| lg1 dual persistence/inertness | Original lg1 deferred checkpoint persistence and did not surface 64/64 slack gates loudly. | BS2/OP2 lineage built persistence/inertness surfaces; require them in future windows. | Keep persistence ON in successors; report slack state each gate. |
| #824 reset / bias correction | Old B/Bprime result was retrain/diagonal-scoped and not a grounded solve; optimizer moment persistence was initially unbuilt. | Arm C persistence is now built; `--persist-optimizer-state on` is the live build result. | Use persistence for future multi-window runs; reopen B/Bprime only with moment-vs-decay separation and excursion-magnitude controls. |
| b4s "minus inert" set | Dropping inert margin-weight preserved byte fidelity but left build debt. | Do not rerun b4s as-is. Carry W02 evidence, fix guard/process gaps, and race newly built single-variable levers. | TP1/BI1 boundary read first; then EN1 margin-weight A/B if a clean window remains. |

## Recall Evidence

| source searched | query / scope | finding beyond charter seeds | plan impact |
|---|---|---|---|
| `.omx/research/ddm_audit_naive_binary_20260805.md` | amendment F3/F6 | XO1 ordering alone was dead; context model was owed. Inert means un-engineered, not dead. | Ran Leg 2 as context-coder pricing and Leg 1 as build, not a drop. |
| `.omx/state/main_hot_state.md` | live fleet and operator-corrected narrative | Live bar is own-vehicle `0.7539807296911207`; SMEVR is not token-bulk bar; EN1 owns this exact scorer-free work. | Kept scorer-free and used IX2/Brotli baseline. |
| `src/tac/witness_dsl/spec_tr1_renderer_20260728.py` | `lever_seg_margin_weight`, `lever_reset_operator` | Margin-weight DSL lever already existed with RACED-NOT-ASSERTED temp; #824 arm C persistence superseded the old unbuilt state. | Did not invent a new lever or constants; classified #824 as persistence/fire-order, not dead lever. |
| `src/tac/witness_dsl/guarded_constant_registry.py`, `src/tac/witness_dsl/guarded_constant.py` | `RT2`, margin floor | RT2 withdrew the wrong-floor finding and says the shipped floor is the guarded value. | Avoided new margin constants; used existing `_live_margin_weight`. |
| `.omx/research/ddm_xo1_20260805/XO1_RECEIPT.md` | XO1 order/control | Ordering worsened IX2 by 11,561 B and oracle order worsened it too; packet is 201 B and parsed. | Did not rerace ordering; reused the packet for context pricing only. |
| `src/tac/optimization/ddm_ix2_archive_container.py` | IX2 token frame | Current token frame baseline is `341,295 B`; forced Brotli-Q11 equals IX2 auto on this stream. | Set the honest Leg 2 bar to Brotli-Q11/IX2, with SMEVR comparison only. |
| `.omx/research/ddm_fh1_forces_harvest_20260731.md`, `.omx/research/ddm_burn4_charter_skeleton_20260731.md` | fh1 force rows R1-R7 | Several fh1 entries were designed stubs or never-fired levers, not negative evidence. | Produced build/fire-order rows instead of folding the family. |
| `.omx/research/ddm_lg1_lane_guard_20260731.md`, `.omx/research/ddm_b4s_guard_audit_20260801.md`, `.omx/research/ddm_bs2_lane_guard_schedule_and_binary_occupancy_sweep_20260801.md` | lane guard, dual, topology | λ dual was wired; b4s stayed slack; topology guard had false positives and rollback raise was not KKT-licensed under `g < 0`. | Classified lane guard as wired/HOLD with rollback precondition, not inert/dead. |
| `tools/list_canonical_equations.py --json` | token/context/rate/margin/lane/reset filters | Found adjacent rate/context laws but no current-vehicle decoder-causal XO1 context law overriding the measured byte race. | Kept Leg 2 empirical and non-causal-scoped. |

I did not find in those searched scopes a measured, decoder-causal XO1 context coder result on the current `sub_auto_pairbit` token stream.

## Verification

Focused tests:

```sh
.venv/bin/python -m pytest src/tac/tests/test_ddm_tp2_tr1_form_predicate_and_margin_gate.py src/tac/witness_dsl/tests/test_ddm_pt2_lever_port.py
```

Result: `60 passed in 15.45s`.

Headless runtime note: an attempted direct MLX loss-value smoke could not allocate Metal in this environment even after selecting CPU, so the regression proof is AST/source-structural plus the existing import/DSL tests.

## NEXT_IF_RESUMED

```json
{
  "run_id": "ddm_en1_20260805",
  "axis": "[macOS-CPU scorer-free byte-only]",
  "score_claim": false,
  "scorer_forwards": 0,
  "upstream_evaluate_py": false,
  "leg1": {
    "status": "BUILT",
    "lever": "tr1_seg_margin_weight",
    "next_fire": "single-variable A/B at next clean window boundary; do not mutate burning run"
  },
  "leg2": {
    "status": "BELOW_WEAK_GO_AND_NON_CAUSAL",
    "best_context_total_bytes": 340735,
    "baseline_forced_ix2_brotli_q11_bytes": 341295,
    "delta_bytes": -560,
    "weak_go_bar_saved_bytes": 15000,
    "required_before_shipping": "decoder-legal context feature path plus >=15000 B saved after counting map/features"
  },
  "leg3": {
    "status": "LOCAL_CALIBRATION_OK_SCOPE_LIMITED",
    "entropy_vs_ix2_brotli_spearman": 0.9981196644178428,
    "entropy_vs_ix2_brotli_sign_agreement": "18/20",
    "do_not_claim": "does not overturn rsf1 trajectory/regime evidence"
  },
  "leg4": {
    "status": "ROWS_CLASSIFIED",
    "highest_priority_fire_order": [
      "complete TP1/BI1 matched boundary read",
      "race EN1 margin-weight consumer at a clean boundary",
      "only then revisit fh1 R1+R4/R3 build arms"
    ]
  },
  "do_not_do": [
    "Do not rerace XO1 ordering; it is already folded.",
    "Do not use SMEVR-only as the token-bulk baseline.",
    "Do not call old inert flags dead without a consumer/build audit.",
    "Do not raise lane lambda when its own dual constraint is slack."
  ]
}
```

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
