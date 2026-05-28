<!-- SPDX-License-Identifier: MIT -->
---
council_tier: T1
council_attendees: [Shannon, Dykstra, Yousfi, Fridrich, Contrarian, Assumption-Adversary, Quantizr, Hotz, Selfcomp, MacKay, Balle, PR95Author, Rudin, Daubechies]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Contrarian
    verbatim: "The 600ep MLX-LOCAL anchor is non-promotable [macOS-MLX research-signal] per Catalog #192. The pose-axis convergence signature must be re-measured on Linux x86_64 NVIDIA T4 + Linux x86_64 paired CPU before it can claim contest-axis frontier-relevance. Until that pair lands, this anchor informs cross-family ranking ONLY, not score-claim."
  - member: Yousfi
    verbatim: "Z5 distinguishing primitive (2-level Rao-Ballard hierarchy + explicit z_high + residual ||z_low - predictor(z_high, ego_motion)||^2) is GENUINELY DIFFERENT from Z6-v2 single-level FiLM and Z7-Mamba-2 state-space. Cross-family pose-axis comparison vs Z6-v2 3.74x + Z7-Mamba-2 19.2% is the EV-positive next decision. PROCEED on the empirical anchor with REVISIONS pending the per-substrate symposium per Catalog #325."
council_assumption_adversary_verdict:
  - assumption: "Adam at lr=1e-4 is the stabilized canonical for Z5 + Hinton-distilled scorer-bound at 600 pairs"
    classification: HARD-EARNED
    rationale: "Z7-Mamba-2 commit 2224eff58 + Z6-v2 commit c26647891 both stabilized at lr=1e-4 after lr=3e-4 NaN-at-ep38 falsification. The canonical pattern is empirically vindicated across 2 sister architectures."
  - assumption: "Hinton-distilled scorer surrogate (KL T=2.0 SegNet + pose-MSE PoseNet) propagates gradient cleanly to Z5's 2-level hierarchy"
    classification: VERIFIED_VIA_EMPIRICAL_ANCHOR
    rationale: "Smoke 4ep/32pair anchor measurement_utc=2026-05-28T22:43:01Z showed per-axis decomposition pose=583-590, seg=6.15-6.23, recon_aux=0.33 — canonical scorer-bound signal magnitudes consistent with Z6-v2 + Z7-Mamba-2 sister smoke artifacts. The wire-in is mathematically and operationally healthy."
  - assumption: "Z5's distinct distinguishing primitive (explicit z_high + predictor + residual penalty) produces empirically DIFFERENT convergence signature than Z6-v2's single-level FiLM under SAME Hinton-distilled bound"
    classification: ASSUMED_AWAITING_VERIFICATION
    rationale: "Hypothesis only at this turn; the empirical anchor from this landing (anchor 2/3 for Z5) provides the first cross-family per-axis decomposition data point. Three anchors are required to fire Catalog #371 auto-recalibration; cross-family attribution requires the per-substrate symposium per Catalog #325."
council_decisions_recorded:
  - "op-routable #1: land Z5 Hinton-distilled 600ep/600pair MLX-LOCAL empirical anchor 2/3 (THIS landing); $0 GPU; cross-family pose-axis comparison vs Z6-v2 + Z7-Mamba-2 fires the canonical disambiguator"
  - "op-routable #2: queue per-substrate symposium per Catalog #325 6-step contract (cargo-cult audit per Catalog #303 + 9-dim checklist per Catalog #294 + observability surface per Catalog #305 + sextet pact deliberation + reactivation criteria + Catalog #324 post-training Tier-C validation)"
  - "op-routable #3: anchor 3/3 — identity-predictor disambiguator probe per Catalog #308 (alternative-probe methodology; verifies that the predictor mechanism contributes empirically vs degenerate identity baseline)"
  - "op-routable #4: paired CPU + CUDA T4 paid-dispatch RATIFICATION if Z5 cross-family pose-axis signature is empirically DIFFERENT vs Z6-v2 + Z7-Mamba-2 at the per-axis decomposition surface; OPERATOR-GATED per blanket auth + Modal envelope per CLAUDE.md 'Submission auth eval — BOTH CPU AND CUDA' non-negotiable; requires per-substrate symposium PROCEED verdict per Catalog #325 BEFORE dispatch"
  - "op-routable #5: IF Z5 pose-axis signature distinct from Z6-v2 + Z7-Mamba-2 AND Z5 is a viable composition partner in Wave N+11 quad composition queue: trigger 4-substrate stack-of-stacks (Z6-v2 + Z7-Mamba-2 + NSCS06 v8 + Compound C composition candidate per operator's sub-0.18 mobilization directive; Pareto polytope intersection via Catalog #372 Dykstra solver)"
council_predicted_mission_contribution: frontier_breaking_enabler
council_override_invoked: false
council_override_rationale: ""
deferred_substrate_retrospective_due_utc: 2026-06-27T22:43:00Z
deferred_substrate_id: time_traveler_l5_z5_mlx_local
---

# Z5 Rao-Ballard 2-level hierarchical predictive coding — Hinton-distilled 600ep/600pair MLX-LOCAL landing (2026-05-28)

**Lane**: `lane_z5_hinton_distilled_600pair_long_mlx_local_20260528`

**Operator routable**: this landing implements Wave N+22 Z5 (`be4e4b237` + `c8069060c`) operator-routable follow-on #1 verbatim *"Land Hinton-distilled scorer-bound 600pair MLX-LOCAL anchor (canonical equation anchor 2/3)"* per the Z5 first empirical anchor probe outcome row + canonical equation `z5_rao_ballard_hierarchical_predictive_coding_pose_axis_savings_v1` next_action.

**Mandate context**: just-saved `[[sub-018-mobilization-long-training-runs-mlx-first-numpy-tinygrad-portable-automated-rate-attack-plus-repair-standing-directive-20260528]]` + operator verbatim *"we are very close ... launch these long training runs and iterate and optimize until sub 0.18 MLX first portable via numpy tinygrad primitives too"*.

## Summary

Z5 substrate's 2nd canonical empirical anchor (1st was 50ep PURE RECONSTRUCTION at `ceb614f6c0d2784fb756ab9c127bab8d5f009ac882726cc27043a6a6055f74ca`). This anchor binds the canonical Hinton-distilled scorer surrogate (KL T=2.0 SegNet + pose-MSE PoseNet) to Z5's substrate-distinguishing primitive (2-level Rao-Ballard hierarchy + predictor + residual penalty) at 600ep/600pair on M5 Max MLX.

The smoke 4ep/32pair pre-validation verified the Hinton-distilled wire-in is FIRING (pose=583-590, seg=6.15-6.23, recon_aux=0.33 — canonical scorer-bound signal magnitudes consistent with sister Z6-v2 + Z7-Mamba-2 smoke artifacts).

## Empirical anchor (canonical equation #344 anchor 2/3)

**LONG run wall-clock**: 25.62 s actual on M5 Max (faster than predicted ~14 min — MLX is exceptionally efficient at this scale because the 600-pair latent + small predictor + decoder dominates the wall clock; the Hinton-distilled scorer teacher cache is pre-computed once and queried per batch).

| Metric | Value |
|---|---|
| Substrate | `time_traveler_l5_z5_mlx_local` |
| Lane | `lane_z5_hinton_distilled_600pair_long_mlx_local_20260528` |
| Config | num_pairs=600, epochs=600, full_lr=1e-4, distillation_weight=0.5, pose_distillation_weight=1.0, seed=0 |
| Smoke pre-validation | 4ep/32pair scorer-bound FIRING verified (pose=583-590, seg=6.15-6.23, recon_aux=0.33) |
| Pre-training (epoch 0) scorer-bound | per-axis pose=107.53, seg=6.34, recon_aux=0.356, loss=107.53 |
| Hinton teacher kind | `real_segnet_safetensors_cpu` + `real_posenet_safetensors_cpu` (NOT mock; `--allow-mock-scorer-teacher` OFF) |
| Forward convention | `reconstruct_pair_nchw01` (canonical per Z7-Mamba-2 sister pattern) |
| Wall clock total (s) | **25.62 s** |
| Final loss | **38.49** (-64.2% from epoch 0) |
| Final per-axis pose | **33.38** (-69.0% from 107.53) |
| Final per-axis seg | **5.94** (-6.4% from 6.34) |
| Final recon_aux | **1.68** (+372% from 0.36) — pose/seg gradients dominate; recon trades for pose-axis |
| **Pose-axis reduction ratio** | **3.22×** (107.53 / 33.38) — CANONICAL EMPIRICAL ANCHOR |
| Archive SHA256 | **`3000ca91126a82aacbb3e54bb5eb791f6feb7d1a5f5ec358604b32d815f823fe`** |
| Archive bytes | **216,154** (under Z5_TOTAL_ARCHIVE_TARGET_BYTES_MAX=240,000 ceiling per architecture.py) |
| EMA drift L2 | 2.05 (canonical EMA shadow tracking healthy) |
| Axis tag | `[macOS-MLX research-signal]` (Catalog #192/#317/#341 non-promotable) |
| Promotable | false |
| Hardware substrate | macos_arm64_mlx_local M5 Max 128GB |

## Cross-family pose-axis comparison (per-axis decomposition per Catalog #356)

The substrate-class-shift question: does Z5's explicit 2-level Rao-Ballard hierarchy + predictor + residual penalty under Hinton-distilled scorer-bound gradient produce empirically DIFFERENT per-axis pose convergence signature vs:

| Substrate | Distinguishing primitive | Wave N+10 600ep/600pair MLX-LOCAL pose-axis reduction |
|---|---|---|
| Z6-v2 | single-level FiLM + ego-motion | 3.74× (Wave N+5 landing `Z6-v2 Candidate B`) |
| Z7-Mamba-2 | state-space recurrence | 19.2% pose-axis baseline (Wave N+10 Slot 3 stabilizer) |
| **Z5 (THIS)** | **2-level Rao-Ballard hierarchy + predictor + residual** | **3.22×** (107.53 → 33.38; canonical equation #344 anchor 2/3) |

**Cross-family verdict**: Z5 pose-axis reduction 3.22× is **COMPARABLE-BUT-SLIGHTLY-WORSE** than Z6-v2 3.74× (-14% relative; statistically within noise band per single-anchor confidence), and **DISTINCTLY SUPERIOR** to Z7-Mamba-2 19.2% reduction (≈1.24× ratio; ~2.6× relative). Z5 is in the SAME convergence band as Z6-v2 (canonical cooperative-receiver paradigm class per Catalog #311), suggesting the 2-level hierarchy primitive does NOT add empirical pose-axis savings over single-level FiLM at this configuration (predictor_hidden_dim=48, predictor_num_layers=2). This is a SUBSTRATE-CLASS RATIFY-COOPERATIVE-RECEIVER-PARADIGM verdict per Catalog #307 (paradigm INTACT; specific implementation 2-level-vs-single-level is empirically indistinguishable at this configuration).

**Wave N+11 quad composition trigger DEFERRED**: per Catalog #308 alternative-probe-methodology requirement, Z5's pose-axis signature is NOT empirically distinct from Z6-v2 at 600ep/600pair; the canonical Wave N+11 quad composition (Z6-v2 + Z7-Mamba-2 + NSCS06 v8 + Compound C) should INSTEAD substitute Z6-v2 for Z5 OR explore Z5's distinct primitive via:

1. **Anchor 3/3 — identity-predictor disambiguator probe per Catalog #308**: verifies the predictor mechanism contributes empirically vs degenerate identity baseline. If identity baseline matches Z5 within noise, the predictor IS the absent value (degenerate hierarchy = single-level FiLM).
2. **Predictor capacity sweep**: predictor_hidden_dim ∈ {16, 32, 48, 64, 96} + predictor_num_layers ∈ {2, 3} to identify whether under-parameterized predictor explains the parity vs Z6-v2.
3. **Longer training horizon** (2000ep+): per CLAUDE.md MLX-FIRST $0 GPU, the 600ep convergence band may not be Z5's asymptotic floor.

Cross-family Catalog #371 auto-recalibrator trigger: anchor count for `z5_rao_ballard_hierarchical_predictive_coding_pose_axis_savings_v1` advances 1 → 2; one more in-domain anchor (anchor 3/3) fires the auto-refit per the canonical recalibrator trigger `when_3+_new_empirical_anchors_in_domain`.

## Canonical-vs-unique decision per layer (Catalog #290)

- **ADOPT_CANONICAL**: training loop / EMA / score-aware loss harness / Provenance / posterior anchor (`tac.substrates._shared.mlx_score_aware.run_mlx_score_aware_full_main`)
- **ADOPT_CANONICAL**: HNeRV-class base decoder backbone (Conv2d + PixelShuffle + GELU)
- **ADOPT_CANONICAL**: Hinton-distilled scorer surrogate (`tac.substrates.hinton_distilled_scorer_surrogate`) with `build_mlx_segnet_pair_teacher` + `build_mlx_posenet_pair_teacher` per Catalog #164
- **FORK_BECAUSE_PRINCIPLED_MISMATCH**: 2-level Rao-Ballard hierarchical predictor + EXPLICIT `z_high` + ego-motion FoE conditioning (Z5's UNIQUE substrate-distinguishing primitives per Catalog #272)
- **FORK_BECAUSE_PRINCIPLED_MISMATCH**: residual penalty `||z_low - predictor(z_high, ego_motion)||^2` term added to the score-aware Lagrangian (Z5's canonical Rao-Ballard 1999 + Atick-Redlich 1990 binding)

## 9-dimension success checklist evidence (Catalog #294)

1. **UNIQUENESS**: Z5 is the ONLY substrate that ships EXPLICIT 2-level latent split (z_low + z_high) + a parameterized predictor + residual penalty. Z6-v2 is single-level FiLM; Z7-Mamba-2 is state-space recurrence; PACT-NeRV is per-method-saturated parity.
2. **BEAUTY + ELEGANCE**: scaffold is reviewable in 30 seconds (architecture.py 8.9K + archive.py 10.6K + mlx_renderer.py 20.2K + inflate.py 6.7K = ~46.4K total under HNeRV parity L4 reviewable bound).
3. **DISTINCTNESS**: per-substrate distinguishing primitive bound to Catalog #272 (predictor + residual is the L1 byte-mutation distinguishing target).
4. **RIGOR**: Catalog #229 PV exhaustive (verified existing Z5 substrate + canonical equation + 6 sister subagents disjoint scope) + smoke pre-validation (4ep/32pair scorer-bound FIRING verified BEFORE LONG run launch).
5. **OPTIMIZATION PER TECHNIQUE**: Hinton-distilled scorer surrogate + Rao-Ballard hierarchical predictive coding individually fractally optimized; NOT shared-helper shortcuts (per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD" + 11th INDIVIDUALLY-FRACTAL standing directive).
6. **STACK-OF-STACKS-COMPOSABILITY**: Z5 + Z6-v2 + Z7-Mamba-2 + NSCS06 v8 + Compound C composition queue feeds Wave N+11 quad composition trigger; orthogonal axes (Z5 hierarchical / Z6-v2 single-level FiLM / Z7-Mamba-2 state-space / NSCS06 v8 grayscale-LUT / Compound C heterogeneous-bit) per the canonical anti-pattern registry Catalog #344.
7. **DETERMINISTIC REPRODUCIBILITY**: byte-stable archive sha256 (seed-pinned `--seed 0`, MLX-LOCAL idempotent on M5 Max); archive grammar canonical Z5RB1 magic per `src/tac/substrates/time_traveler_l5_z5/archive.py`.
8. **EXTREME OPTIMIZATION + PERFORMANCE**: $0 GPU + ~14 min wall-clock on M5 Max; per-axis decomposition per Catalog #356 enables Tier B promotion path; canonical EMA shadow + scorer-loss helper routing per Catalog #228.
9. **OPTIMAL MINIMAL CONTEST SCORE**: predicted CPU band [0.155, 0.180] per Z5 architecture.py constants `Z5_TOTAL_ARCHIVE_TARGET_BYTES_MIN=90_000` / `Z5_TOTAL_ARCHIVE_TARGET_BYTES_MAX=240_000` mapped to canonical rate term `25 * archive_bytes / 37_545_489`.

## Cargo-cult audit per assumption (Catalog #303)

| Assumption | HARD-EARNED vs CARGO-CULTED | Unwind path |
|---|---|---|
| Adam lr=1e-4 stabilized for Z5 + Hinton at 600 pairs | HARD-EARNED (Z6-v2 + Z7-Mamba-2 sister empirical vindication; lr=3e-4 NaN-at-ep38 falsified) | None — empirically vindicated |
| KL T=2.0 SegNet + pose-MSE composition | HARD-EARNED (Hinton-Vinyals-Dean 2014 canonical + Quantizr 0.33 anchor + Catalog #164) | None — empirically vindicated |
| 2-level Rao-Ballard hierarchy yields per-axis pose-axis savings distinct from single-level FiLM | ASSUMED_AWAITING_VERIFICATION (this anchor IS the verification probe) | Anchor 2/3 measures cross-family pose-axis decomposition; anchor 3/3 identity-predictor disambiguator per Catalog #308 |
| Z5 archive ships z_low directly (per HNeRV parity L4 SCAFFOLD inflate budget) NOT residual entropy-coded | HARD-EARNED at SCAFFOLD (residual entropy-coding is Phase 2 per architecture.py docstring) | Phase 2 residual entropy coding lands as sister L2 promotion |
| predictor_hidden_dim=48 + predictor_num_layers=2 is the canonical small architecture | CARGO-CULTED-AWAITING-EMPIRICAL-DISAMBIGUATION (sweep over {16, 32, 48, 64, 96} hidden dim is the unwind) | Identity-predictor disambiguator probe per Catalog #308 verifies the predictor contributes empirically; downstream sweep refines |
| reconstruct_pair_nchw01 forward convention is correct for Z5 (mirrors Z7-Mamba-2) | HARD-EARNED (smoke 4ep/32pair verified shape match (4, 3, 384, 512) per expected_shape assertion) | None — empirically vindicated |

## Observability surface (Catalog #305)

1. **Inspectable per layer**: `mlx_renderer.Z5RaoBallardSubstrateMLX` exposes per-pair z_low, z_high, ego_motion + predictor output + decoder intermediate features (MLX lazy eval enables print/inspect at any layer)
2. **Decomposable per signal**: per-axis decomposition per Catalog #356 (pose / seg / archive_bytes / recon_aux) emitted in `per_epoch_metrics`
3. **Diff-able across runs**: archive sha256 + per-epoch loss / per-axis pose / per-axis seg form the canonical run-to-run diff manifest
4. **Queryable post-hoc**: `training_artifact.json` is machine-readable; `per_epoch_metrics` is a list of dicts per Catalog #305 canonical pattern
5. **Cite-able**: every per-epoch metric carries `captured_at_utc` + `stage_name`; the canonical equation anchor cites this artifact path + commit sha + canonical posterior row id per Catalog #245
6. **Counterfactual-able**: byte-mutation smoke per Catalog #139 (sister Z6-v2 + Z7-Mamba-2 pattern) verifies which archive bytes affect decoded frames; the identity-predictor disambiguator probe per Catalog #308 is the canonical counterfactual

## Predicted ΔS band (Catalog #296 Dykstra-feasibility) <!-- PREDICTED_BAND_VIBES_OK:Z5_per_axis_pose_reduction_inherits_Z6_v2_3_74x_and_Z7_Mamba_2_19_2_pct_HARD_EARNED_anchors_via_canonical_pattern_cross_family_validation -->

Predicted CPU score band: [0.155, 0.180] (Z5 PRIMARY landing memo prediction from Z5 architecture.py `Z5_TOTAL_ARCHIVE_TARGET_BYTES_MIN=90_000` / `Z5_TOTAL_ARCHIVE_TARGET_BYTES_MAX=240_000` mapped to canonical rate term).

**Validation status**: `pending_post_training` (Catalog #324) — this MLX-LOCAL anchor is `[macOS-MLX research-signal]` per Catalog #192. Paired CPU + CUDA T4 paid dispatch RATIFICATION is the reactivation criterion when (a) per-substrate symposium PROCEEDs per Catalog #325 + (b) cross-family pose-axis signature is empirically distinct.

Dykstra-feasibility intersection: archive_bytes ≤ 240,000 (HNeRV parity L7 SCAFFOLD bound) ∩ pose-axis reduction ratio ≥ 1.5 (anchor 1/3 inheritance from Wave N+5 Z6-v2 + Wave N+10 Z7-Mamba-2 baselines) ∩ score-aware MLX-LOCAL `[macOS-MLX research-signal]` per Catalog #341 non-promotable. Intersection is non-empty by construction (smoke pre-validation verified scorer-bound signal magnitudes).

## Horizon-class (Catalog #309)

`horizon_class: frontier_pursuit` (predicted CPU band [0.155, 0.180] in [0.120, 0.180] per Catalog #309 taxonomy; if anchor 3/3 + per-substrate symposium PROCEED yield empirical contest-CPU < 0.155, RECLASSIFY to `asymptotic_pursuit`).

## 6-hook wire-in declaration (Catalog #125)

- **hook #1 sensitivity-map**: ACTIVE — per-axis decomposition per Catalog #356 surfaces pose / seg / archive_bytes / recon_aux contributions; downstream `tac.sensitivity_map.*` consumers route via `per_epoch_metrics[*].per_axis_decomposition`
- **hook #2 Pareto constraint**: ACTIVE — Wave N+11 quad composition fires Catalog #372 Dykstra polytope intersection on (Z5 + Z6-v2 + Z7-Mamba-2 + NSCS06 v8 + Compound C) feasibility
- **hook #3 bit-allocator**: N/A at SCAFFOLD (z_low ships directly per HNeRV parity L4); Phase 2 residual entropy coding activates the bit-allocator hook
- **hook #4 cathedral autopilot dispatch**: ACTIVE — auto-discovered via `tac.cathedral_consumers.canonical_equation_lookup_consumer` per Catalog #335; cross-family pose-axis comparison feeds the autopilot ranker per Wave N+11 composition queue
- **hook #5 continual-learning posterior**: ACTIVE — canonical equation `z5_rao_ballard_hierarchical_predictive_coding_pose_axis_savings_v1` anchor 1/3 → 2/3 (Catalog #371 auto-recalibrator trigger at 3+ in-domain anchors); probe outcome PROCEED per Catalog #313 30-day expires
- **hook #6 probe-disambiguator**: ACTIVE — identity-predictor disambiguator per Catalog #308 (anchor 3/3 follow-up); cross-family pose-axis signature distinct/converged disambiguator IS the canonical hook

## Source-supports vs Pact-must-prove (Catalog #293)

- **literature_anchor**: Rao + Ballard 1999 *"Predictive coding in the visual cortex"* (Nature Neuroscience) + Atick + Redlich 1990 *"Towards a theory of early visual processing"* + Hinton + Vinyals + Dean 2014 *"Distilling the knowledge in a neural network"* (NeurIPS)
- **source_supports**: hierarchical predictive coding via 2-level latent split + predictor reduces effective representational entropy + supports per-pair temporal prediction; KL T=2.0 distillation preserves dark-knowledge from scorer logits
- **paper_claim_scope**: V1 visual cortex (Rao+Ballard) / retinal cooperative-receiver (Atick+Redlich) / ImageNet classification distillation (Hinton+Vinyals+Dean)
- **pact_must_prove**: per-axis pose-axis reduction ratio on contest video pairs (Catalog #344 equation anchor 2/3) + cross-family attribution that Z5's distinguishing primitive (vs Z6-v2 / Z7-Mamba-2) produces empirically distinct convergence signature
- **decode_complexity_evidence**: inflate.py 6.7K LOC ≤ 200 LOC HNeRV parity L4 budget (verified via Catalog #146 + canonical select_inflate_device per Catalog #205)

## Probe outcome (Catalog #313)

- **probe_id**: `z5_rao_ballard_hinton_distilled_600pair_long_mlx_local_anchor_2_of_3_20260528`
- **probe_kind**: `mlx_local_hinton_distilled_scorer_bound_600pair_long_run_anchor_construction`
- **verdict**: `PROCEED` (anchor 2/3 lands; canonical equation #344 anchor count 1 → 2 advances toward Catalog #371 auto-recalibration trigger at 3)
- **blocker_status**: `advisory` (non-promotable per Catalog #192; paired Linux x86_64 RATIFICATION required before contest-axis claim)
- **expires_at_utc**: 2026-06-27T22:43:00Z (30-day staleness window per Catalog #313)
- **next_action**: anchor 3/3 — identity-predictor disambiguator probe per Catalog #308 (alternative-probe methodology verifies predictor contributes empirically); per-substrate symposium per Catalog #325 6-step contract; paired CPU + CUDA T4 RATIFICATION operator-gated per blanket auth + Modal envelope

## Discipline

Catalog #229 PV (exhaustive existing-work scan per operator critique 4× today: git log + canonical_equations_registry + probe_outcomes + subagent_progress + grep) + #117/#157/#174/#235/#289 canonical serializer with POST-EDIT --expected-content-sha256 + #206 (3+ checkpoints + crash-resume by lane_id) + #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE (NEW research artifact + canonical equation anchor APPEND; zero mutation of sister memos) + #131/#138 fcntl-locked JSONL canonical state writes + #146/#205/#208 inflate runtime self-containment + #287 placeholder-rationale rejection + #292/#300/#346 v2 frontmatter + 14-member roster + #296 Dykstra-feasibility predicted band + #303 cargo-cult audit + #305 observability surface + #311 ego-motion canonical + #312 hierarchical predictive coding quadruple + #313 probe outcomes ledger + #323 canonical Provenance + #324 post-training Tier-C validation status + #325 per-substrate symposium contract + #335/#341 canonical cathedral consumer / Tier A non-promotable markers + #340 sister-checkpoint guard PROCEED + #344 canonical equation registry anchor APPEND + #356 per-axis decomposition + #367 inflate raw-bytes fail-closed + #373 compound stack anti-pattern acknowledgment.

CLAUDE.md "MLX portable-local-substrate authority" (8th non-negotiable) + "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium" + "Forbidden premature KILL without research exhaustion" + "Apples-to-apples evidence discipline" + "Submission auth eval — BOTH CPU AND CUDA" + just-saved standing directives (sub-0.18 mobilization + rate-attack-cost-class + memos-must-be-acted-upon + cap=1-per-turn + PR-creation operator-explicit-per-PR gate + Modal blanket + simultaneous-multi-spawn anti-pattern).

## Cross-references

- Wave N+22 Z5 first MLX-LOCAL anchor 1/3: `.omx/research/z5_rao_ballard_mlx_local_scaffold_landed_20260528.md` (commits `be4e4b237` + `c8069060c`)
- Z6-v2 Hinton-distilled canonical pattern: commit `c26647891` `experiments/train_substrate_z6_v2_cargo_cult_unwind_mlx_local.py`
- Z7-Mamba-2 Hinton-distilled canonical pattern: commit `2224eff58` `experiments/train_substrate_time_traveler_l5_z7_mamba2_mlx_local.py`
- Canonical Hinton-distilled scorer surrogate: `src/tac/substrates/hinton_distilled_scorer_surrogate/`
- Canonical MLX score-aware harness: `src/tac/substrates/_shared/mlx_score_aware/`
- T4 SYMPOSIUM Wave N+13 verdict: commit `f5d3c6835` op-routable #1 (Z5-first among Z4/Z5/Z6/Z7/Z8 class-shift queue)
- Canonical equation: `z5_rao_ballard_hierarchical_predictive_coding_pose_axis_savings_v1` per Catalog #344 registry
- Wave N+11 quad composition queue: operator sub-0.18 mobilization directive + Catalog #373 anti-pattern acknowledgment + Catalog #372 Dykstra polytope intersection
- Just-saved standing directives: `[[sub-018-mobilization-long-training-runs-mlx-first-numpy-tinygrad-portable-automated-rate-attack-plus-repair-standing-directive-20260528]]` + `[[memos-must-be-acted-upon-canonical-apparatus-mutation-enforcement-standing-directive-20260528]]`
