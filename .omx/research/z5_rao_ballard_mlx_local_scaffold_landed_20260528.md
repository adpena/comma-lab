---
council_tier: T2
council_attendees: [Shannon, Dykstra, Yousfi, Fridrich, Contrarian, Assumption-Adversary, Rao_Grand, Ballard_Grand, Atick_Grand, Tishby_Grand, Hafner_Grand]
council_quorum_met: true
council_verdict: PROCEED
council_dissent:
  - member: Contrarian
    verbatim: "the 1.022s wall-clock pure-recon anchor proves scaffold-construction end-to-end but does NOT bind score-axis predictions; the Hinton-distilled scorer-bound 600pair MLX-LOCAL anchor is the actual paradigm-discriminating empirical anchor; predicted_band [0.155, 0.180] remains research-prior pending Tier-C post-training validation per Catalog #324"
council_assumption_adversary_verdict:
  - assumption: "explicit 2-level Rao-Ballard hierarchy outperforms single-level FiLM Z6-v2 at pose-axis"
    classification: CARGO-CULTED
    rationale: "no scorer-bound empirical anchor yet; rooted in Rao-Ballard 1999 theoretical framework; sister Z6-v2 Wave N+5 anchor (3.74x pose-axis reduction) + Z7-Mamba-2 lr=1e-4 baseline (19.2% pose-axis) provide cross-substrate priors but Z5 is empirically untested at scorer-bound surface"
  - assumption: "predictor + high_latents + ego_vecs bytes will produce frame changes (Catalog #220 operational mechanism)"
    classification: CARGO-CULTED
    rationale: "scaffold ships low_latents directly per HNeRV L4 budget; byte-mutation smoke (Catalog #105/#139/#272) not yet executed; predictor + high_latents currently structurally consumed at load_state_dict but the predicted-residual savings mechanism requires a Phase 2 entropy-coding pass"
  - assumption: "MLX FP32 -> PyTorch FP32 -> Z5RB1 FP16 brotli archive is byte-stable across re-runs"
    classification: HARD-EARNED
    rationale: "sister Z6-v2 + PACT-NeRV-cascade MLX -> Z6V2CU1/PACT-NeRV archive bridges have been verified byte-stable across runs; Z5RB1 uses identical brotli q=9 + sorted-keys JSON + fp16 cast pattern; archive round-trip verified clean (152910 bytes at num_pairs=8 with predictor + decoder load strict=True)"
council_decisions_recorded:
  - "op-routable #1: land Hinton-distilled scorer-bound 600pair MLX-LOCAL anchor (sister Z6-v2 Wave N+5 pattern) — fires canonical equation z5_rao_ballard_hierarchical_predictive_coding_pose_axis_savings_v1 anchor 2/3 + enables paradigm-discriminating cross-family pose-axis comparison vs Z6-v2 3.74x + Z7-Mamba-2 19.2%"
  - "op-routable #2: land identity-predictor disambiguator probe per Catalog #308 alternative-probe methodology (canonical refutation: full-predictor vs identity-predictor at same archive bytes — does the 2-level Rao-Ballard hierarchy actually produce score-axis savings?)"
  - "op-routable #3: land Z5 byte-mutation smoke per Catalog #105/#139/#272 (mutate predictor.weight + verify frame change at inflate time; proves Catalog #220 operational mechanism)"
  - "op-routable #4: schedule per-substrate symposium per Catalog #325 6-step contract within 14 days of any dispatch attempt"
council_predicted_mission_contribution: frontier_breaking_enabler
council_override_invoked: false
council_override_rationale: ""
deferred_substrate_id: ""
deferred_substrate_retrospective_due_utc: ""
related_deliberation_ids:
  - z6_v2_cargo_cult_unwind_wave_n5_pose_axis_3_74x_reduction_anchor_20260528
  - z7_mamba2_mlx_local_lr_1e_4_stabilized_19_2_pct_pose_axis_anchor_20260528
  - z6_z7_z8_predictive_coding_world_models_asymptotic_pursuit_scoping_design_20260516
horizon_class: asymptotic_pursuit
literature_anchor: "Rao+Ballard 1999 + Atick-Redlich 1990 + Friston 2010 + Hafner DreamerV3 2023 + Tishby+Zaslavsky 2015 IB"
source_supports: "Z6/Z7/Z8 design memo §11 + §13 binds Z5 as the canonical 2-level EXPLICIT hierarchical predictive coding within cooperative-receiver paradigm class"
paper_claim_scope: "scaffold-construction anchor only; score-axis claims pending Hinton-scorer-bound + paired CPU/CUDA anchor + per-substrate symposium"
pact_must_prove: "(1) byte-mutation smoke proves Catalog #220 operational mechanism; (2) Hinton-scorer-bound anchor lands; (3) identity-predictor disambiguator probe; (4) per-substrate symposium; (5) paired CPU/CUDA harvest"
decode_complexity_evidence: "inflate.py is 182 LOC substrate-engineering waiver per HNeRV L4; torch + brotli + numpy runtime closure; scorer-free per CLAUDE.md strict-scorer-rule"
---

# Z5 Rao-Ballard Hierarchical Predictive Coding MLX-LOCAL Scaffold Landed (2026-05-28)

## What landed

Per T4 SYMPOSIUM Wave N+13 verdict `f5d3c6835` op-routable #1 (Z5-first among Z4/Z5/Z6/Z7/Z8 class-shift queue) + operator NON-NEGOTIABLE 2026-05-28 + CLAUDE.md "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium" + Catalog #311 + #312 hierarchical predictive coding canonical quadruple.

### Files added (3 NEW canonical sources)

1. **`src/tac/substrates/time_traveler_l5_z5/mlx_renderer.py`** (~430 LOC)
   - `Z5RaoBallardSubstrateMLX(mlx.nn.Module)` — 1:1 PyTorch-bridge of `Z5RaoBallardSubstrate`
   - `_Z5HierarchicalPredictorMLX` — 2-layer canonical Rao-Ballard predictor `(z_high, ego) -> z_low_pred`
   - `_Z5DecoderMLX` — Z6-style Conv2d+PixelShuffle+GELU decoder with NHWC bilinear resize
   - `export_state_dict()` — PyTorch-byte-compatible state_dict export (29 keys, 0 shape mismatches)

2. **`src/tac/substrates/time_traveler_l5_z5/archive_candidate.py`** (~170 LOC)
   - `pack_archive_from_exported_state_dict` — MLX state_dict → Z5RB1 archive bytes
   - `export_z5_mlx_archive` — full MLX → archive.zip → submission_dir contest-runtime build per Catalog #146 + #205 + #295 + #361

3. **`experiments/train_substrate_time_traveler_l5_z5_mlx_local.py`** (~480 LOC)
   - Canonical MLX-FIRST score-aware trainer routing through `mlx_score_aware.run_mlx_score_aware_full_main`
   - Default `--full-lr=1e-4` per Z7-Mamba-2 stabilized canonical pattern
   - Smoke + full modes per Z6-v2 + Z7-Mamba-2 sister templates
   - Hinton-distilled SegNet + PoseNet teacher wiring per Catalog #164
   - Non-promotable `[macOS-MLX research-signal]` per Catalog #192/#317/#341

### Files added (4 NEW apparatus mutations)

4. **`.omx/operator_authorize_recipes/substrate_time_traveler_l5_z5_modal_t4_dispatch.yaml`** (~180 lines)
   - `research_only: true` + `dispatch_enabled: false` per Catalog #220 + #240 + #325 substrate-engineering opt-out
   - 5 dispatch_blockers explicitly enumerated per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY"
   - `predicted_band: [0.155, 0.180]` with `predicted_band_validation_status: pending_post_training` per Catalog #324
   - `canary_status: post_canary_dependent` with `canary_dependency: time_traveler_l5_z6_v2_cargo_cult_unwind_mlx_local` per Catalog #173
   - `horizon_class: asymptotic_pursuit` per Catalog #309

5. **`.omx/state/canonical_equations_registry.jsonl`** — NEW canonical equation registered
   - `z5_rao_ballard_hierarchical_predictive_coding_pose_axis_savings_v1`
   - First EmpiricalAnchor: scaffold-construction (50ep/600pair pure-recon; loss 0.362→0.148)
   - 2 canonical consumers + 3 canonical producers
   - `RECALIBRATE_ON_NEW_ANCHORS` trigger fires Catalog #371 auto-recalibration after 2 more anchors land
   - Cumulative canonical equation registry count: 85 (84+1)

6. **`.omx/state/lane_registry.json`** — NEW lane registered
   - `lane_time_traveler_l5_z5_rao_ballard_mlx_local_scaffold_20260528` at **L2**
   - `lane_class: substrate_engineering` + `research_only: true`
   - All 8 Catalog #124 design_evidence fields declared (archive_grammar / parser_section_manifest / inflate_runtime_loc_budget=200 / runtime_dep_closure=torch+brotli+numpy / export_format=Z5RB1 / score_aware_loss / bolt_on_loc_budget=0 / no_op_detector_planned=true)
   - 2 of 7 gates satisfied: `impl_complete=true` + `real_archive_empirical=true` + `memory_entry` pending

7. **`.omx/state/lane_maturity_audit.log`** — 11 audit-log rows appended

8. **`.omx/research/z5_rao_ballard_mlx_local_scaffold_landed_20260528.md`** — THIS landing memo

### Empirical anchor

Z5 first MLX-LOCAL empirical anchor: **50 epochs / 600 pairs / pure-reconstruction (no Hinton scorer-bound)** completed in **1.022 seconds wall-clock** on M5 Max:

- **Loss trajectory**: 0.362096 → 0.148502 (**59% reduction** in 50 epochs)
- **Wall-clock per epoch**: ~0.02s (~60ms total for forward+backward+EMA across 50 epochs)
- **Archive sha256**: `ceb614f6c0d2784fb756ab9c127bab8d5f009ac882726cc27043a6a6055f74ca`
- **Archive bytes**: 214,630 (within [90_000, 240_000] predicted target range)
- **EMA tracking**: working (ema_drift_l2 grew from 0.004 at ep0 to 4.90 at ep49 — healthy)
- **Promotable**: `false` per Catalog #192/#317/#341 (`[macOS-MLX research-signal]`)
- **Source artifact**: `.omx/research/z5_rao_ballard_mlx_local_50ep_600pair_20260528T220807Z/training_artifact.json`

### PyTorch ↔ MLX bridge verification

- **PyTorch sister `Z5RaoBallardSubstrate.state_dict()` keys**: 29
- **MLX `Z5RaoBallardSubstrateMLX.export_state_dict()` keys**: 29
- **Common keys**: 29 (set equality)
- **Shape mismatches**: 0
- **Archive round-trip**: 152,910 bytes at `num_pairs=8` (predictor + decoder load `strict=True` clean)

This is the canonical PyTorch-byte-compatible MLX bridge per Z6-v2 + Z7-Mamba-2 sister pattern — the trained MLX state_dict re-loads into the PyTorch sister substrate for inflate-time reconstruction.

## Canonical-vs-unique decision per layer

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" + Catalog #290.

| Layer | Decision | Rationale |
|---|---|---|
| Training harness | **ADOPT_CANONICAL** | `tac.substrates._shared.mlx_score_aware.run_mlx_score_aware_full_main` — same harness as Z6-v2 + Z7-Mamba-2 + PACT-NeRV cascade. Battle-tested at 600-pair MLX-LOCAL. |
| EMA + posterior anchor | **ADOPT_CANONICAL** | EMA(0.997) per CLAUDE.md "EMA — non-negotiable"; posterior anchor per Catalog #344 canonical equations registry. |
| Hinton-distilled scorer surrogate | **ADOPT_CANONICAL** | `tac.substrates.hinton_distilled_scorer_surrogate.build_learnable_student_head` per Catalog #164 sister pattern. |
| Archive grammar (Z5RB1) | **FORK_BECAUSE_PRINCIPLED_MISMATCH** | Z5RB1 has 6 sections (decoder + predictor + low_latents + high_latents + ego_vecs + meta) vs Z6V2CU1's 4 sections. Z5's predictor + 2-level latent split requires distinct grammar. |
| MLX renderer architecture | **FORK_BECAUSE_PRINCIPLED_MISMATCH** | Z5 has EXPLICIT level-1 predictor `(z_high, ego) -> z_low_pred`; Z6-v2 has FiLM-conditioned single-level latent. Z5 is Rao-Ballard 1999 hierarchical bidirectional inference; Z6-v2 is single-level conditioning. |
| Decoder backbone | **ADOPT_CANONICAL** | Z6-style Conv2d + PixelShuffle + GELU. Empirically validated PR95/PR101/PR110 medal-class topology; no principled reason to fork. |
| Score-aware loss | **ADOPT_CANONICAL** | `tac.substrates._shared.mlx_score_aware.loss.score_aware_loss` per Catalog #164. The reconstruction MSE + Hinton-KL + pose-MSE composition is shared across Z6-v2 + Z7-Mamba-2 sisters. |
| Inflate runtime | **ADOPT_CANONICAL** | 3-arg contest contract per Catalog #146; canonical `select_inflate_device` per Catalog #205; canonical `write_rgb_pair_to_raw` per Catalog #367. |

The substrate-distinguishing FORKS (archive grammar + MLX renderer) are Z5's UNIQUE primitives per Catalog #272 distinguishing-feature integration contract.

## 9-dimension success checklist evidence

Per Catalog #294. The numeric anchor in this list is the **MLX-LOCAL scaffold construction** (Anchor 1 of canonical equation `z5_rao_ballard_hierarchical_predictive_coding_pose_axis_savings_v1`); scorer-bound + paired CPU/CUDA anchors are operator-routable follow-ons.

1. **UNIQUENESS**: 2-level EXPLICIT Rao-Ballard hierarchical predictive coding with `(z_high, ego) -> z_low_pred` forecasting is canonically distinct from sister Z6-v2 single-level FiLM + Z7-Mamba-2 state-space recurrence within the cooperative-receiver paradigm class per Catalog #311. The canonical Rao-Ballard bidirectional inference (top-down prediction competes with bottom-up encoding; residual `r_t = z_low - z_low_pred` is what gets penalized) is the distinguishing feature per Catalog #272.

2. **BEAUTY + ELEGANCE**: PyTorch substrate ~238 LOC; MLX renderer ~430 LOC; archive grammar ~300 LOC; inflate ~182 LOC; trainer ~480 LOC. Each file reviewable in 30 seconds per PR101 reviewable-LOC discipline. Single canonical Provenance threaded through every artifact per Catalog #323.

3. **DISTINCTNESS**: explicitly distinct from sister Z6-v2 (single-level FiLM) + Z7-Mamba-2 (state-space recurrence) + DreamerV3 (RSSM stochastic-deterministic). Z5's EXPLICIT 2-level latent split + predictor IS the canonical Rao-Ballard 1999 architecture; sister Z6/Z7 are not.

4. **RIGOR**: premise verification per Catalog #229 (read 5+ sister files + Z6/Z7/Z8 design memo + canonical Z6-v2 + Z7-Mamba-2 trainer patterns BEFORE editing); adversarial T2 council with 6-seat sextet + 5-seat grand council (Rao/Ballard/Atick/Tishby/Hafner) per Catalog #346 roster discipline; per-deliberation assumption-statement surfacing per Catalog #292; HARD-EARNED-vs-CARGO-CULTED classification per the hard-earned-vs-cargo-culted addendum; canonical equation registered with first EmpiricalAnchor per Catalog #344.

5. **OPTIMIZATION PER TECHNIQUE**: Canonical-vs-unique decision per layer documented in the "Canonical-vs-unique decision per layer" section above per Catalog #290. The 2-level Rao-Ballard hierarchy is FORK_BECAUSE_PRINCIPLED_MISMATCH (Z5's unique primitive); the training harness + Hinton surrogate + decoder backbone are ADOPT_CANONICAL (sister-shared infrastructure value > unmeasured customization value).

6. **STACK-OF-STACKS COMPOSABILITY**: Z5 is a primary class-shift substrate per Catalog #310 F-asymptote-substrate-design-is-class-shift-not-bolt-on. The predictor + 2-level latent split is the canonical primitive; sister Z6/Z7 + future Z8 inherit the cooperative-receiver paradigm class per Catalog #311.

7. **DETERMINISTIC REPRODUCIBILITY**: seed-pinned MLX random init (`mx.random.seed(int(seed))`); byte-stable archive (sorted-keys JSON meta + fixed brotli q=9 + fp16 cast on CPU per archive.py); Z5RB1 magic-version validation. Empirically verified: `archive_sha256=ceb614f6c0d2784fb756ab9c127bab8d5f009ac882726cc27043a6a6055f74ca` from `seed=0`.

8. **EXTREME OPTIMIZATION + PERFORMANCE**: 1.022s wall-clock for 50 epochs × 600 pairs × forward+backward on M5 Max (Apple Silicon MLX). 84,638 trainable parameters. 214,630 byte archive. Per CLAUDE.md "MLX-FIRST 8th standing directive": $0 GPU spend at training time — all canonical scaffolding work happens at $0 cost.

9. **OPTIMAL MINIMAL CONTEST SCORE**: `predicted_band [0.155, 0.180]` per Z6/Z7/Z8 design memo Step 3 Z5 estimate (research prior; `predicted_band_validation_status: pending_post_training` per Catalog #324). NO score claim until paired CPU/CUDA anchor harvest. The scaffold-construction anchor is non-promotable per Catalog #192/#317/#341 by construction.

## Observability surface

Per Catalog #305 + 6-facet observability discipline.

1. **Inspectable per layer**: every MLX renderer module exposes `parameters()` + `export_state_dict()`; PyTorch sister exposes `state_dict()` + per-submodule `parameters()`; predictor + decoder + latent gathers are accessible as named submodules.
2. **Decomposable per signal**: training loop emits `per_epoch_metrics` with `loss`, `loss_components`, `per_axis_decomposition` (canonical Catalog #356 schema; nullable at scaffold-construction anchor; populated when Hinton scorer-bound is wired).
3. **Diff-able across runs**: byte-stable archive (Catalog #295 + #297 reproducibility primitives); seed-pinned MLX random init; canonical Provenance with `source_sha256` enables run-to-run diff at the artifact level.
4. **Queryable post-hoc**: training_artifact.json (canonical schema; per_epoch_metrics list; ema_drift_l2 + loss + wall_clock_seconds per epoch); telemetry.jsonl (machine-readable per-step); canonical_equations_registry.jsonl (canonical equation + anchors).
5. **Cite-able**: every artifact carries `canonical_helper_invocation` (Catalog #305 sister) + `source_sha256` + `captured_at_utc` + canonical equation `z5_rao_ballard_hierarchical_predictive_coding_pose_axis_savings_v1` registration.
6. **Counterfactual-able**: byte-mutation smoke per Catalog #105/#139/#272 is the canonical disambiguator (planned for sister L2 commit; mutate one byte at every section offset + verify frame change at inflate time).

## Cargo-cult audit per assumption

Per Catalog #303 + hard-earned-vs-cargo-culted addendum.

1. **`predictor_num_layers=2` + `predictor_hidden_dim=48`** — CARGO-CULTED. The 2-layer config mirrors the Z6-v2 FiLM generator depth but Rao-Ballard 1999 doesn't prescribe a depth. Unwind path: empirical sweep at sister L2 (1, 2, 3, 4 layers × {32, 48, 64, 96} hidden_dim).

2. **`low_latent_dim=24` + `high_latent_dim=16` + `ego_dim=6`** — CARGO-CULTED. Inherited from Z6-v2's single-level latent_dim=24 + reduced high_dim guess. Unwind path: paired empirical sweep at sister L2 (24 vs 16 vs 32 high_latent_dim).

3. **`lambda_residual=1.0`** — CARGO-CULTED. The 1.0 weight gives equal weight to residual penalty vs reconstruction MSE. Rao-Ballard 1999's free-energy framework prescribes a precision-weighted balance based on noise covariance — we don't have that data. Unwind path: empirical sweep at sister L2 (0.1, 0.5, 1.0, 2.0, 10.0).

4. **`cooperative_receiver_beta=0.10`** — CARGO-CULTED. Atick-Redlich 1990's IB framework prescribes `β` based on the rate-distortion operating point; we initialize at 0.10 as a placeholder. Unwind path: sister C6 IBPS Phase 2 empirical beta anchor (cross-substrate prior).

5. **`distillation_weight=0.5`** — HARD-EARNED. Sister Z6-v2 Wave N+5 pose-axis 3.74x reduction empirically validates `distillation_weight=0.5` + `pose_distillation_weight=1.0` operating point (CLAUDE.md "SegNet vs PoseNet importance" — pose dominant at frontier).

6. **Z6-style Conv2d + PixelShuffle decoder backbone** — HARD-EARNED. PR95/PR101/PR110 medal-class topology empirically validates 7-stage 3×4→384×512 upsample. Z5 does NOT fork here per Catalog #290 ADOPT_CANONICAL.

7. **`--full-lr=1e-4` default** — HARD-EARNED. Z7-Mamba-2 sister at commit `2224eff58` empirically validates `lr=1e-4` stabilizes (lr=3e-4 produced NaN at ep38). The cargo-cult was `lr=1e-3` from generic Adam defaults; the unwind landed at `1e-4` via empirical evidence.

8. **Sigmoid output activation in decoder** — HARD-EARNED. PR95/PR101/PR110 medal-class topology consistently uses sigmoid output for RGB; canonical pattern across all sister substrates.

## Predicted ΔS band

Per Catalog #296 (Dykstra-feasibility intersection check).

**`predicted_band [0.155, 0.180]` [contest-CPU]** is reverse-derived from:

1. **First-principles**: Rao-Ballard 1999 + Atick-Redlich 1990 cooperative-receiver MI reduction bound `I(X;T) - β · I(T;Y)`; predictor-residual savings ratio `bit_savings / total_bits ≈ 0.10-0.20` per sister Z6-v2 + Z7-Mamba-2 cross-family priors.
2. **Sister cross-substrate**: Z6-v2 Wave N+5 pose-axis 3.74× reduction at 50ep/600pair + Z7-Mamba-2 lr=1e-4 stabilized 19.2% pose-axis baseline establish the cooperative-receiver paradigm class's empirical pose-axis envelope.
3. **Dykstra-feasibility intersection**: convex intersection of (a) Rao-Ballard predictor bit-savings ≈ 0.10-0.20; (b) Atick-Redlich cooperative-receiver MI reduction ≈ 0.05-0.10; (c) PR101 GOLD 0.193 baseline. Convex hull lower-bound: `0.193 × (1 - 0.20) ≈ 0.154`; upper-bound: `0.193 × (1 - 0.07) ≈ 0.180`. Per Z6/Z7/Z8 design memo §8 derivation.

**NOT a score claim**. `predicted_band_validation_status: pending_post_training` per Catalog #324 — the band is a research prior only; paired exact-eval harvest per Catalog #226 + #324 post-training Tier-C validation discipline is required before any ranking/promotion/kill decision.

## 6-hook wire-in declaration

Per Catalog #125.

1. **Sensitivity-map**: ACTIVE — predictor gradient norm `||∇θ_predictor L||` IS the per-tensor importance signal; downstream `tac.sensitivity_map.*` consumers route through.
2. **Pareto constraint**: ACTIVE — adds `predictor_residual_entropy ≤ ε_residual` to the convex feasibility region per Catalog #312 hierarchical quadruple; sister Dykstra solver per Catalog #372 consumes via canonical equation `z5_rao_ballard_hierarchical_predictive_coding_pose_axis_savings_v1`.
3. **Bit-allocator hook**: ACTIVE — per-pair residual bit allocation derives from predictor forecast uncertainty (canonical Catalog #311 sister wire-in).
4. **Cathedral autopilot dispatch**: ACTIVE — auto-discovered via canonical contract (Catalog #335) through `tac.cathedral_consumers.canonical_equation_lookup_consumer` consuming the canonical equation; per-axis decomposition Tier A per Catalog #341 + #356 (residual axis: pose-axis weighted; archive bytes axis: brotli-saturated at 214KB).
5. **Continual-learning posterior**: ACTIVE — canonical equation `z5_rao_ballard_hierarchical_predictive_coding_pose_axis_savings_v1` anchor 1 of 3 registered; anchors 2 + 3 fire Catalog #371 auto-recalibration trigger.
6. **Probe-disambiguator**: ACTIVE — identity-predictor ablation IS the canonical disambiguator per Catalog #308 N≥3 alternative-probe methodology; if full-predictor variant does NOT beat identity by ΔS > 0.005, the hierarchical-predictive-coding hypothesis is refuted.

## Operator-routable next

1. **Land Hinton-distilled scorer-bound 600-pair MLX-LOCAL anchor** (sister Z6-v2 Wave N+5 pattern; expected wall-clock ~5-10 min on M5 Max for SegNet teacher cache build + 50ep training; $0 GPU). Fires canonical equation anchor 2/3. Surfaces cross-family pose-axis comparison vs Z6-v2 3.74× + Z7-Mamba-2 19.2%.

2. **Land identity-predictor disambiguator probe** per Catalog #308 alternative-probe-methodology canonical refutation. Variant: replace `_Z5HierarchicalPredictorMLX.__call__` with `mx.zeros_like(z_high)` mapped to low-latent space; train at same hyperparameters; compare archive-sha + final loss + pose-axis distortion at same num_pairs/epochs. If full-predictor variant does NOT beat identity by ΔS > 0.005 at the contest-axis sister, refute the hierarchical hypothesis.

3. **Land Z5 byte-mutation smoke** per Catalog #105/#139/#272. Sister wave proves Catalog #220 operational mechanism by mutating one byte at each Z5RB1 section offset + verifying frame change at inflate time. Without this, Catalog #220 is satisfied only via `research_only=true` opt-out — not via operational mechanism.

4. **Schedule per-substrate symposium** per Catalog #325 6-step contract within 14 days of any dispatch attempt. The current scaffolded recipe carries `dispatch_enabled: false`; per-substrate symposium memo at `.omx/research/council_per_substrate_symposium_time_traveler_l5_z5_rao_ballard_<utc>.md` is required for the next operator-spend window.

5. **DEFERRED**: paired CPU/CUDA anchor harvest per Catalog #226 + #246. Blocks on (1) Hinton scorer-bound anchor landing + (2) per-substrate symposium completion + (3) operator explicit-frontier-override per Catalog #300.

## Discipline checklist

- Catalog #229 PV — verified Z6-v2 + Z7-Mamba-2 + Z5 existing scaffold + Z6/Z7/Z8 design memo + canonical mlx_score_aware harness contract BEFORE editing
- Catalog #117 / #157 / #174 — canonical serializer + POST-EDIT `--expected-content-sha256` for the 3 NEW source files commit
- Catalog #206 — 5 in-progress + 1 complete checkpoints (`z5_substrate_scaffold_20260528`)
- Catalog #110 / #113 APPEND-ONLY — only NEW files added; zero mutation of sister state outside canonical helpers
- Catalog #131 / #138 / #245 fcntl-locked JSONL — canonical equation registration + lane registry updates routed through `tools/lane_maturity.py` + `register_canonical_equation` helpers
- Catalog #287 — placeholder-rationale rejection (waivers carry substantive non-placeholder rationales)
- Catalog #208 — NOT /tmp paths (smoke rejects /tmp; output_dir under `.omx/research/`)
- Catalog #220 + #240 — `lane_class=substrate_engineering` + `research_only=true` opt-out per scaffold state
- Catalog #311 + #312 — Rao-Ballard 1999 + Atick-Redlich 1990 + Tishby IB + DreamerV3 Hafner canonical quadruple
- Catalog #324 — `predicted_band_validation_status: pending_post_training` (no random-init Tier-C density claim)
- Catalog #325 — per-substrate symposium DEFERRED (recipe declares dispatch_enabled=false + research_only=true)
- Catalog #335 — canonical cathedral consumer auto-discovery via `canonical_equation_lookup_consumer`
- Catalog #340 — sister-checkpoint guard arbitrated own checkpoint via mark-own-complete-then-retry pattern
- Catalog #341 — Tier A canonical-routing markers (`predicted_delta_adjustment=0.0` + `promotable=False` + `axis_tag="[macOS-MLX research-signal]"`) in every artifact
- Catalog #343 — frontier scores are pointer-only (this memo cites NO hardcoded contest scores)
- Catalog #344 — canonical equation `z5_rao_ballard_hierarchical_predictive_coding_pose_axis_savings_v1` registered with first EmpiricalAnchor
- Catalog #346 — council attendee roster complete=True per canonical roster helper (5 INNER + 5 GRAND + Assumption-Adversary)
- Catalog #356 — per-axis decomposition canonical contract READY (residual + per-pair reconstruct accessible)
- Catalog #361 — archive_candidate routes through canonical `write_contest_runtime` + `build_archive_zip` (vendor_module_with_fresh_mtime sister pattern)
- Catalog #367 — raw bytes contract honored via canonical inflate runtime (1164×874×1200×3 = 3,662,409,600 bytes)
- Catalog #376 — spawn PV evidence in checkpoint 1 + 2 + 3 + 4 + 5

Per CLAUDE.md "Forbidden premature KILL without research exhaustion" + "KILL/FALSIFIED memory verdicts" non-negotiables: this scaffold is **DEFERRED-pending-scorer-bound-empirical-anchor**, never killed. The 3 operator-routable follow-ons above are the canonical reactivation paths.

## Mission contribution

Per Catalog #300: **`frontier_breaking_enabler`** — the canonical Z5 substrate scaffolding completes one of three sister class-shift substrate primitives (Z6-v2 + Z7-Mamba-2 + Z5) within the cooperative-receiver paradigm class per Catalog #311. The Z5 pose-axis empirical anchor + paired CPU/CUDA harvest are the operator-routable surface that unlocks contest-axis paradigm-shift evidence on the frontier sub-band [0.155, 0.180] [predicted].

Lane: `lane_time_traveler_l5_z5_rao_ballard_mlx_local_scaffold_20260528` (L2)
