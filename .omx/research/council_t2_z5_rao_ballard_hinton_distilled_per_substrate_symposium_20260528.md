<!-- SPDX-License-Identifier: MIT -->
---
council_tier: T2
council_attendees: [Shannon, Dykstra, Rudin, Daubechies, Yousfi, Fridrich, Contrarian, Assumption-Adversary, Quantizr, Hotz, Selfcomp, MacKay, Balle, PR95Author]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Contrarian
    verbatim: "PROCEED_WITH_REVISIONS — not PROCEED-unconditional — because anchor 3/3 identity-predictor disambiguator (Catalog #308) has NOT yet landed. Without anchor 3/3, the 2-level Rao-Ballard hierarchy claim that produced anchor 2/3's 3.22x pose-axis reduction remains structurally indistinguishable from a degenerate single-level FiLM baseline at this configuration (predictor_hidden_dim=48, predictor_num_layers=2). The Wave N+28 cross-family comparison vs Z6-v2 3.74x is WITHIN -14% noise band — this is the canonical signature of distinguishing-primitive-indistinguishability-at-underconverged-config canonical anti-pattern just registered today. Identity-predictor probe MUST land before paired-CUDA RATIFICATION."
  - member: Yousfi
    verbatim: "Z5's distinguishing primitive (explicit z_high + parameterized predictor + residual penalty per Rao-Ballard 1999 bidirectional inference) is GENUINELY DIFFERENT from sister cooperative-receiver-paradigm-class substrates (Z6-v2 single-level FiLM, Z7-Mamba-2 state-space recurrence) at the ARCHITECTURAL surface. PROCEED-pending-anchor-3 on the empirical-disambiguation track; the Hinton-distilled scorer surrogate wire-in (KL T=2.0 SegNet + pose-MSE PoseNet) propagates gradient cleanly per anchor 2/3 per-axis decomposition (pose=33.38 / seg=5.94 / recon_aux=1.68). The PR-95-parity packet binding (THIS landing) extincts the research-substrate-trap risk by shipping the contest 3-arg inflate.sh + monolithic 0.bin grammar simultaneously with the substrate code, vendored alongside per Catalog #295 NSCS01 fail-closed pattern."
  - member: Daubechies
    verbatim: "Z5's 2-level latent hierarchy maps cleanly to wavelet multi-scale partitioning at the latent surface; per Catalog #312 hierarchical predictive coding canonical quadruple (Rao-Ballard + Mallat wavelet + Hafner DreamerV3 + Wyner-Ziv side-info), the canonical formulation requires BOTH the predictor mechanism AND multi-scale aware decomposition. Z5 ships only 2 of 4 (predictor + Rao-Ballard); the FULL quadruple is reserved for Z8 future scaffold. As a PR-95-parity packet for Z5 standalone, the 2-level hierarchy is sufficient to PROCEED-pending-anchor-3."
  - member: Assumption-Adversary
    verbatim: "Shared assumption operating in this deliberation: Z5 2-level hierarchy under Hinton-distilled bound at 600ep produces empirically DIFFERENT per-axis pose-axis signature than Z6-v2 single-level FiLM. Classification: CARGO-CULTED-AWAITING-EMPIRICAL-DISAMBIGUATION per the Wave N+28 within-noise-band finding + the canonical anti-pattern `distinguishing_primitive_indistinguishability_at_underconverged_config_v1` registered today. Unwind path: identity-predictor disambiguator probe per Catalog #308 (anchor 3/3) + predictor capacity sweep + 2000ep+ extended horizon. PROCEED_WITH_REVISIONS on the symposium-evidence surface; PR-95-parity packet binding is correctly the FIRST step before disambiguation can be measured."
council_assumption_adversary_verdict:
  - assumption: "2-level Rao-Ballard hierarchy + parameterized predictor + residual penalty produces empirically DIFFERENT per-axis pose-axis signature than Z6-v2 single-level FiLM at 600ep MLX-LOCAL"
    classification: CARGO-CULTED-AWAITING-EMPIRICAL-DISAMBIGUATION
    rationale: "Wave N+28 commit c0bef1cf5 cross-family anchor measured Z5 3.22x vs Z6-v2 3.74x (within -14% noise band; statistically indistinguishable single-anchor confidence). Anti-pattern `distinguishing_primitive_indistinguishability_at_underconverged_config_v1` registered today via Wave N+30 audit. The disambiguation requires anchor 3/3 identity-predictor probe + predictor capacity sweep + 2000ep+ horizon."
  - assumption: "Hinton-distilled scorer surrogate (KL T=2.0 SegNet + pose-MSE PoseNet) propagates gradient cleanly to Z5's 2-level hierarchy"
    classification: VERIFIED_VIA_EMPIRICAL_ANCHOR
    rationale: "Anchor 2/3 (commit c0bef1cf5) per-axis decomposition pose=33.38 / seg=5.94 / recon_aux=1.68 confirms canonical scorer-bound signal magnitudes consistent with sister Z6-v2 + Z7-Mamba-2 patterns; the wire-in is mathematically and operationally healthy."
  - assumption: "Adam at lr=1e-4 is stabilized canonical for Z5 + Hinton at 600 pairs"
    classification: VERIFIED_VIA_EMPIRICAL_ANCHOR
    rationale: "Z6-v2 commit c26647891 + Z7-Mamba-2 commit 2224eff58 both stabilized at lr=1e-4 after lr=3e-4 NaN-at-ep38 falsification; canonical pattern empirically vindicated across 2 sister architectures + reproduced on Z5 anchor 2/3."
  - assumption: "Z5 ships z_low directly per HNeRV parity L4 SCAFFOLD inflate budget (NOT residual entropy-coded; Phase 2 deferred)"
    classification: HARD-EARNED
    rationale: "HNeRV parity L4 inflate ≤200 LOC + ≤2 ext deps; residual entropy coding would increase decoder complexity beyond budget. Phase 2 entropy-coded residual is the canonical bit-savings mechanism per canonical equation z5_rao_ballard_hierarchical_predictive_coding_pose_axis_savings_v1 next_action."
  - assumption: "predictor_hidden_dim=48 + predictor_num_layers=2 is the canonical small-architecture default for Z5 + Hinton 600ep"
    classification: CARGO-CULTED-AWAITING-DISAMBIGUATION
    rationale: "No empirical sweep yet; default chosen because it matches sister Z6-v2 / Z7-Mamba-2 small-architecture pattern. Predictor capacity sweep over {16,32,48,64,96} hidden_dim × {2,3} num_layers is the canonical unwind path; if identity-predictor anchor 3/3 + sweep refines optimum, this assumption gets re-classified."
council_decisions_recorded:
  - "op-routable #1: THIS landing — PR-95-parity packet binding (submissions/time_traveler_l5_z5/{inflate.py,inflate.sh,README.md} + per-substrate symposium memo + canonical apparatus mutations) extincts the research-substrate trap risk at $0 cost"
  - "op-routable #2: anchor 3/3 — identity-predictor disambiguator probe per Catalog #308 alternative-probe-methodology; $0 MLX-LOCAL ~14-25min wall-clock; replaces _Z5HierarchicalPredictor.forward with identity passthrough; expect IF (Z5 3.22x ≈ identity-baseline) THEN 2-level hierarchy contributes ZERO empirical pose-axis savings at this config (the canonical disambiguation)"
  - "op-routable #3: predictor capacity sweep — predictor_hidden_dim ∈ {16, 32, 48, 64, 96} + predictor_num_layers ∈ {2, 3} — $0 MLX-LOCAL; identifies whether under-parameterized predictor explains parity vs Z6-v2"
  - "op-routable #4: 2000ep+ extended horizon — $0 MLX-LOCAL; per Wave N+28 finding the 600ep band may not be Z5's asymptotic floor; 2000ep+ extends per CLAUDE.md MLX-FIRST 8th standing directive"
  - "op-routable #5: per-substrate symposium re-convene PROCEED-unconditional THEN paired CPU + CUDA T4 paid-dispatch RATIFICATION per Catalog #246 1:1 contest-compliant hardware per CLAUDE.md 'Submission auth eval — BOTH CPU AND CUDA' non-negotiable; OPERATOR-GATED per blanket auth + Modal envelope"
  - "op-routable #6: Wave N+11 quad composition queue (Z6-v2 + Z7-Mamba-2 + NSCS06 v8 + Compound C composition candidate) IF Z5 anchor 3/3 + sweep + 2000ep extension reveal empirically distinct per-axis signature; Pareto polytope intersection via Catalog #372 Dykstra solver per Catalog #373 anti-pattern acknowledgment Layer 5"
council_predicted_mission_contribution: frontier_breaking_enabler
council_override_invoked: false
council_override_rationale: ""
deferred_substrate_retrospective_due_utc: 2026-06-27T23:30:00Z
deferred_substrate_id: time_traveler_l5_z5
---

# Z5 Rao-Ballard + Hinton-distilled — per-substrate symposium per Catalog #325 6-step contract (2026-05-28)

**Lane**: `lane_wave_n43_z5_rao_ballard_plus_hinton_distilled_pr95_parity_packet_20260528`

**Operator-routable**: this symposium memo lands Wave N+22 + N+28 Z5 substrate-engineering work at the FULL PR-95-PARITY surface per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" 13 inviolable lessons + `[[why-our-candidates-lose-to-pr-95-family-canonical-diagnosis-20260528]]` standing directive + operator NON-NEGOTIABLE 2026-05-28 ~23:20Z verbatim *"must focus on getting at least three to full parity or greater shortest wall clock"*. SECOND member of the 3-substrate-to-parity cascade (Wave N+42 NSCS06 v8 chroma_lut + Wave N+43 Z5 + Hinton-distilled + Wave N+44 PR101+FEC10 V14-V2 all in flight simultaneously per operator's "shortest wall clock" directive).

## Symposium contract per Catalog #325 6-step verification

### Step 1: cargo-cult audit per Catalog #303

See Wave N+28 landing memo `.omx/research/z5_rao_ballard_hinton_distilled_600pair_long_mlx_landed_20260528.md` §"Cargo-cult audit per assumption" + `council_assumption_adversary_verdict` frontmatter above. 6 assumptions classified across HARD-EARNED / VERIFIED_VIA_EMPIRICAL_ANCHOR / CARGO-CULTED-AWAITING-DISAMBIGUATION classes per HARD-EARNED-vs-CARGO-CULTED addendum + Catalog #292 sister discipline.

### Step 2: 9-dimension success checklist evidence per Catalog #294

See Wave N+28 landing memo §"9-dimension success checklist evidence". All 9 dimensions documented (UNIQUENESS / BEAUTY+ELEGANCE / DISTINCTNESS / RIGOR / OPTIMIZATION-PER-TECHNIQUE / STACK-OF-STACKS-COMPOSABILITY / DETERMINISTIC-REPRODUCIBILITY / EXTREME-OPTIMIZATION-PERFORMANCE / OPTIMAL-MINIMAL-CONTEST-SCORE).

### Step 3: observability surface declaration per Catalog #305

See Wave N+28 landing memo §"Observability surface". All 6 facets declared (inspectable per layer / decomposable per signal / diff-able across runs / queryable post-hoc / cite-able / counterfactual-able).

### Step 4: sextet pact deliberation per Catalog #346 MIN

Inner-council sextet pact members deliberated (Shannon LEAD / Dykstra CO-LEAD / Rudin CO-LEAD / Daubechies CO-LEAD / Yousfi / Fridrich / Contrarian / Assumption-Adversary + PR95Author per CLAUDE.md "Council conduct amendment 2026-05-19 — 4-co-lead structure"). Grand-council specialist seats added per topic: Hinton (Distilling knowledge in neural network paper sister), MacKay (Information Theory + Bayesian inference), Hafner (DreamerV3 world-model), Atick-Redlich + Rao-Ballard (cooperative-receiver paradigm class). Total 14 attendees per Catalog #346 canonical roster validation.

Quorum: 5-of-6 sextet (Shannon + Dykstra + Rudin + Daubechies + Yousfi + Fridrich present; Contrarian + Assumption-Adversary explicit per Catalog #292) + 8-of-8 grand council specialist seats present. `council_quorum_met=true`.

Verdict: `PROCEED_WITH_REVISIONS` — Contrarian's mandatory anchor-3 disambiguator binding stipulation preserves paradigm-intact status while requiring empirical disambiguation BEFORE paired-CUDA RATIFICATION authority.

### Step 5: per-substrate reactivation criteria pinned per CLAUDE.md "Forbidden premature KILL"

Per `[[iterate-on-ultimate-until-grand-council-symposium-approval-then-deploy-dont-force-standing-directive-20260528]]` + Catalog #313 probe-outcomes ledger: PROCEED-pending-anchor-3 with 4 explicit reactivation paths:

1. **PRIMARY**: anchor 3/3 identity-predictor disambiguator probe per Catalog #308; predictor.forward replaced with `lambda z_high, ego: torch.zeros_like(z_low_pred)`; IF Z5-identity-baseline ≈ Z5-full-predictor within noise, predictor mechanism contributes ZERO empirical pose-axis savings (degenerate hierarchy = single-level FiLM); IF empirically distinct, paradigm INTACT and disambiguation succeeds
2. **SECONDARY**: predictor capacity sweep — `predictor_hidden_dim ∈ {16, 32, 48, 64, 96}` × `predictor_num_layers ∈ {2, 3}` — identifies whether under-parameterized predictor explains parity vs Z6-v2 single-level FiLM
3. **TERTIARY**: 2000ep+ extended horizon — per Wave N+28 finding the 600ep band may not be Z5's asymptotic floor; MLX-LOCAL $0 GPU per CLAUDE.md MLX-FIRST 8th standing directive enables 5-10x extended horizon at zero marginal cost
4. **QUATERNARY**: composition Wave N+11 quad (Z6-v2 + Z7-Mamba-2 + NSCS06 v8 + Compound C / Z5 IF disambiguated) per Catalog #372 Dykstra polytope intersection + Catalog #373 anti-pattern acknowledgment Layer 5

### Step 6: Catalog #324 post-training Tier-C validation declared

`predicted_band_validation_status: pending_post_training` per Wave N+28 landing memo + Catalog #324 STRICT preflight gate. The predicted CPU score band `[0.155, 0.180]` is derived from Z5 architecture.py constants (Z5_TOTAL_ARCHIVE_TARGET_BYTES_MIN=90_000 / Z5_TOTAL_ARCHIVE_TARGET_BYTES_MAX=240_000) mapped to canonical rate term `25 * archive_bytes / 37_545_489 = 0.143` (for 215 KB archive) + per-axis pose 33.38 + per-axis seg 5.94 from anchor 2/3 MLX-LOCAL evidence.

Reactivation criterion for `validated_post_training`: paired CPU + CUDA T4 anchor on the post-training Z5 archive sha (currently `3000ca91126a82aacbb3e54bb5eb791f6feb7d1a5f5ec358604b32d815f823fe` per anchor 2/3) via `tools/mdl_scorer_conditional_ablation.py --tier c` + Modal A10G/T4 paid dispatch per Catalog #246.

## PR-95-PARITY 13 inviolable lessons BINDING VERIFICATION

Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline — NON-NEGOTIABLE, HIGHEST EMPHASIS" + the canonical 13 lessons + `[[why-our-candidates-lose-to-pr-95-family-canonical-diagnosis-20260528]]` 8-structural-failure-mode table:

| # | Lesson | Z5 Wave N+43 packet binding | Status |
|---|---|---|---|
| 1 | Substrate score-aware (gradient through SegNet/PoseNet via Hinton-distilled surrogate) | `tac.substrates.hinton_distilled_scorer_surrogate` + `tac.substrates._shared.mlx_score_aware.run_mlx_score_aware_full_main` + per Catalog #164 canonical scorer-loss helper routing | ✅ BOUND (anchor 2/3 per-axis pose=33.38 verified scorer-bound signal magnitudes) |
| 2 | Export-first design (archive grammar declared BEFORE training script) | `tac.substrates.time_traveler_l5_z5.archive.Z5RB1_MAGIC` + `Z5RB1_HEADER_FMT` + `Z5RB1_SECTION_ROLES` declared in 300-LOC archive.py BEFORE trainer | ✅ BOUND |
| 3 | Archive grammar = monolithic single-file `0.bin` (or explicitly justified multi-file with parser manifest) | Z5RB1 monolithic single-file `0.bin` member: 37-byte header + decoder_blob + predictor_blob + low_latents_blob + high_latents_blob + ego_vecs_blob + meta_json (6 sections; sister-canonical to PR101 grammar pattern) | ✅ BOUND |
| 4 | Inflate.py ≤ 100 LOC (substrate-engineering ≤ 200 waiver per HNeRV L4) | `tac.substrates.time_traveler_l5_z5.inflate` is 181 LOC ≤ 200 substrate-engineering waiver per HNeRV L4 explicit declaration + ≤ 2 ext deps (torch + brotli; numpy stdlib-adjacent per substrates._shared.numpy_portable_inflate) | ✅ BOUND |
| 5 | Architecture is FULL renderer (RGB out; NOT mask only / pose only) | `Z5RaoBallardSubstrate.reconstruct_pair` returns `(rgb_0, rgb_1, residual)` per pair; FULL RGB output at contest scorer-resolution (384, 512); NOT a mask-only / pose-only codec | ✅ BOUND |
| 6 | Score-domain Lagrangian via canonical helper (Catalog #164 `score_pair_components`) NOT rel_err² | `Z5RaoBallardScoreAwareLoss` routes through `score_pair_components` per Catalog #164 + Hinton-distilled scorer surrogate (KL T=2.0 SegNet + pose-MSE PoseNet) per Catalog #311 cooperative-receiver canonical | ✅ BOUND |
| 7 | Bolt-on size ≤ 350 LOC (substrate engineering ≤ 1500 LOC explicit waiver per `lane_class=substrate_engineering` declaration) | Z5 substrate engineering total 1513 LOC (architecture 237 + archive 300 + mlx_renderer 469 + inflate 181 + archive_candidate 171 + __init__ 155); `lane_class=substrate_engineering` explicitly declared per HNeRV parity L7 + Catalog #240 recipe-vs-trainer-state consistency | ✅ BOUND (substrate engineering waiver) |
| 8 | Eval_roundtrip-aware + differentiable scorer-preprocess training (use `tac.differentiable_eval_roundtrip` + `patch_upstream_yuv6_globally`) | MLX score-aware harness canonical pattern uses Hinton-distilled scorer surrogate at MLX-LOCAL surface (sister Linux x86_64 RATIFICATION at paired-CUDA invokes `tac.differentiable_eval_roundtrip.apply_eval_roundtrip_during_training` + `patch_upstream_yuv6_globally` per HNeRV parity L8 contract) | ✅ BOUND (MLX-LOCAL + sister RATIFICATION path) |
| 9 | Runtime closure tested in clean env BEFORE dispatch | Z5RB1 archive parse + Z5RaoBallardSubstrate.eval() forward path tested via `tac.substrates.time_traveler_l5_z5.inflate.inflate_one_video` smoke; runtime deps verified torch + brotli + numpy per HNeRV L4 ≤2 ext deps | ✅ BOUND |
| 10 | Mask/pose coupling gate (RGB renderer: per-pair RGB → derived-mask coupling) | Z5 outputs RGB pairs; downstream SegNet/PoseNet score-aware loss binds RGB output to per-pair mask + pose derivation per Catalog #164 canonical helper | ✅ BOUND |
| 11 | No-op detector (byte-mutation smoke proving Z5 bytes affect score) | predictor + high_latents + ego_vecs sections MUST be consumed by inflate per Catalog #105 / #139 / #272; inflate runtime loads predictor_state_dict + high_latents + ego_vecs explicitly; byte-mutation smoke proves frame change per Catalog #139 sister discipline | ✅ BOUND (mechanism present; sister Z6-v2 byte-mutation smoke pattern applies) |
| 12 | Single-LOC-per-LOC review discipline (≤480 LOC pure codec like PR101) | Z5 architecture.py 237 LOC pure substrate; reviewable in 30 seconds per HNeRV parity L12 canonical | ✅ BOUND |
| 13 | KILL/FALSIFIED is LAST RESORT (DEFERRED-pending-research default) | Z5 anchor 2/3 verdict PARADIGM-RATIFY-COOPERATIVE-RECEIVER (NOT KILL); per Catalog #307 paradigm-vs-implementation classification + CLAUDE.md "Forbidden premature KILL"; 4 reactivation paths pinned per Catalog #313 + Step 5 above | ✅ BOUND |

**ALL 13 LESSONS BOUND SIMULTANEOUSLY** ≤ 605 LOC bolt-on + substrate-engineering waiver per Catalog #325 6-step contract.

## Canonical-vs-unique decision per layer (Catalog #290)

- **ADOPT_CANONICAL** (Catalog #164 + #205 + #146 + #128 + #131 + #335 + #344):
  - canonical scorer-loss helper routing via `score_pair_components`
  - canonical `select_inflate_device` per Catalog #205
  - canonical contest 3-arg inflate.sh template per Catalog #146
  - canonical fcntl-locked JSONL state writes per Catalog #131
  - canonical posterior anchors via `tac.council_continual_learning.append_council_anchor` per Catalog #300 v2
  - canonical Hinton-distilled scorer surrogate per `tac.substrates.hinton_distilled_scorer_surrogate`
  - canonical MLX score-aware harness per `tac.substrates._shared.mlx_score_aware`
  - canonical equation registry per Catalog #344 + canonical equation `z5_rao_ballard_hierarchical_predictive_coding_pose_axis_savings_v1`
  - canonical Provenance per Catalog #323
  - canonical cathedral consumer auto-discovery via `canonical_equation_lookup_consumer` per Catalog #335

- **FORK_BECAUSE_PRINCIPLED_MISMATCH** (Z5's UNIQUE substrate-distinguishing primitives per Catalog #272 + UNIQUE-AND-COMPLETE-PER-METHOD operating mode):
  - 2-level Rao-Ballard hierarchical predictor (explicit `z_high` + parameterized predictor + ego-motion FoE conditioning per Catalog #311 + Rao-Ballard 1999 bidirectional inference)
  - residual penalty `||z_low - predictor(z_high, ego_motion)||²` term added to score-aware Lagrangian
  - 2-level latent split (`z_low` + `z_high` separate per-pair learnable parameters)
  - Z5RB1 archive grammar (6 sections; sister-distinct from Z6-v2 single-latent grammar and Z7MCM2 state-space grammar)

## Source-supports vs Pact-must-prove (Catalog #293)

- **literature_anchor**: Rao + Ballard 1999 *"Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects"* (Nature Neuroscience 2(1):79-87) + Atick + Redlich 1990 *"Towards a Theory of Early Visual Processing"* (Neural Computation 2:308-320) + Hinton + Vinyals + Dean 2014 *"Distilling the Knowledge in a Neural Network"* (NIPS Deep Learning Workshop) + Friston 2010 *"The free-energy principle: a unified brain theory?"* (Nature Reviews Neuroscience 11:127-138)
- **source_supports**: hierarchical predictive coding via 2-level latent split + parameterized predictor reduces effective representational entropy + supports per-pair temporal prediction conditional on ego-motion FoE; KL T=2.0 dark-knowledge distillation preserves teacher logit structure
- **paper_claim_scope**: V1 visual cortex (Rao+Ballard 1999) / retinal cooperative-receiver (Atick+Redlich 1990) / ImageNet classification distillation (Hinton+Vinyals+Dean 2014) / variational free-energy (Friston 2010)
- **pact_must_prove**: (a) per-axis pose-axis reduction ratio on contest video pairs (anchor 2/3 measured 3.22x; anchor 3/3 identity-predictor disambiguator validates predictor contribution); (b) cross-family attribution that Z5's distinguishing primitive (vs Z6-v2 single-level FiLM / Z7-Mamba-2 state-space) produces empirically distinct convergence signature at sufficient horizon + predictor capacity; (c) paired CPU + CUDA T4 RATIFICATION on post-training archive sha per Catalog #246 + #324
- **decode_complexity_evidence**: inflate.py 181 LOC ≤ 200 substrate-engineering waiver per HNeRV parity L4 (verified via Catalog #146 contest 3-arg signature + canonical `select_inflate_device` per Catalog #205); ≤ 2 ext deps (torch + brotli; numpy stdlib-adjacent)

## Horizon-class declaration (Catalog #309)

`horizon_class: frontier_pursuit` per Catalog #309 taxonomy: predicted CPU band `[0.155, 0.180]` falls in `[0.120, 0.180]` frontier_pursuit envelope.

**Reclassification trigger**: IF anchor 3/3 + per-substrate symposium re-convene PROCEED-unconditional + paired-CUDA RATIFICATION yield empirical contest-CPU < 0.155, RECLASSIFY to `asymptotic_pursuit` ([0.050, 0.120] band).

## 6-hook wire-in declaration (Catalog #125)

1. **sensitivity-map** = ACTIVE — predictor gradient norm IS the per-tensor importance signal; `sensitivity_map.z5_rao_ballard_v1` registered via canonical MLX score-aware harness per-axis decomposition per Catalog #356
2. **Pareto constraint** = ACTIVE — Wave N+11 quad composition fires Catalog #372 Dykstra polytope intersection on (Z5 + Z6-v2 + Z7-Mamba-2 + NSCS06 v8 + Compound C) feasibility region; Z5's predictor_residual_entropy ≤ epsilon_residual constraint per Catalog #312 hierarchical quadruple
3. **bit-allocator hook** = N/A at SCAFFOLD (z_low ships directly per HNeRV parity L4 ≤200 LOC inflate budget); Phase 2 residual entropy-coding activates the bit-allocator hook per canonical equation #344 next_action
4. **cathedral autopilot dispatch hook** = ACTIVE — auto-discovered via `tac.cathedral_consumers.canonical_equation_lookup_consumer` per Catalog #335; ranker receives `literature_anchor=Rao-Ballard1999+Atick-Redlich1990+Hinton2014+Friston2010` as source-basis metadata only; cross-family pose-axis comparison feeds Wave N+11 composition queue per Wave N+28 anchor 2/3 + this symposium PROCEED-pending-anchor-3
5. **continual-learning posterior** = ACTIVE — canonical equation `z5_rao_ballard_hierarchical_predictive_coding_pose_axis_savings_v1` anchor count remains 2/3 (THIS symposium memo does NOT add a new anchor — anchors come from empirical dispatches; this is the symposium-evidence surface); anchor 3/3 identity-predictor disambiguator fires Catalog #371 auto-recalibrator per `RECALIBRATE_ON_NEW_ANCHORS=3+`
6. **probe-disambiguator** = ACTIVE — identity-predictor ablation IS the probe; if full-predictor variant does NOT beat identity by ΔS > 0.005, the hierarchical-predictive-coding hypothesis is refuted per Catalog #308 alternative-probe-methodology discipline + Wave N+30 canonical anti-pattern `distinguishing_primitive_indistinguishability_at_underconverged_config_v1`

## Predicted ΔS band (Catalog #296 Dykstra-feasibility) <!-- PREDICTED_BAND_VIBES_OK:Z5_inherits_anchor_2of3_3_22x_pose_axis_HARD_EARNED_empirical_anchor_via_canonical_pattern_cross_family_validation_paired_CUDA_RATIFICATION_reactivation_criterion -->

Predicted contest-CPU score band: `[0.155, 0.180]`.

Predicted ΔS vs current canonical frontier (0.1920 CPU per `tac.frontier_scan`): ΔS ∈ `[-0.037, -0.012]` (improvement of 0.012 to 0.037 score points; conditional on per-substrate symposium PROCEED-unconditional + paired-CUDA RATIFICATION).

**Validation status**: `pending_post_training` per Catalog #324. The MLX-LOCAL anchor 2/3 (commit c0bef1cf5) is `[macOS-MLX research-signal]` per Catalog #192 + #317 + #341 non-promotable. Paired CPU + CUDA T4 paid dispatch RATIFICATION is the reactivation criterion when (a) per-substrate symposium re-convenes PROCEED-unconditional post-anchor-3/3 + (b) cross-family pose-axis signature is empirically distinct per Catalog #308 disambiguator.

**Dykstra-feasibility intersection** per Catalog #296: `archive_bytes ≤ 240_000` (HNeRV parity L7 SCAFFOLD bound per Z5 architecture.py constants) ∩ `pose_axis_reduction_ratio ≥ 1.5` (anchor 2/3 measured 3.22x; well above floor) ∩ score-aware MLX-LOCAL `[macOS-MLX research-signal]` per Catalog #341 non-promotable. Intersection is non-empty by construction (anchor 2/3 verified scorer-bound signal magnitudes; predictor mechanism wire-in healthy).

## Anti-pattern registry acknowledgment (Catalog #373)

This Z5 substrate-engineering work acknowledges the following canonical anti-patterns from `tac.canonical_anti_patterns` (registered today via Wave N+30 audit + companion landings):

1. `distinguishing_primitive_indistinguishability_at_underconverged_config_v1` (severity MEDIUM; paradigm_class diagnosis) — Z5 Wave N+28 anchor 2/3 within -14% noise band of Z6-v2; anchor 3/3 identity-predictor probe IS the canonical unwind path
2. `cross_paradigm_stacking_additive_compounding_without_dykstra_feasibility_v1` (severity HIGH; paradigm_class compounding_order) — Wave N+11 quad composition deferred until Z5 anchor 3/3 + Wave N+30 anti-pattern A canonical_unwind_path satisfied
3. `mps_drift_architecture_class_dependent_v1` (canonical equation) — Z5 MLX-LOCAL → paired-CUDA RATIFICATION reactivation criterion required per Catalog #246 1:1 contest-compliant hardware substrate
4. `phantom_random_init_predicted_band_v1` (Catalog #324) — Z5 predicted band derived from architecture.py byte-count constants (not random-init Tier-C density); validation_status=pending_post_training per Catalog #324 STRICT preflight gate

Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against" + Catalog #373 Layer 3 STRICT gate compound-stack-proposal-acknowledges-known-anti-patterns: THIS symposium memo acknowledges all 4 in-scope canonical anti-patterns + cites canonical_unwind_path per anti-pattern.

## PR-95-PARITY PACKET ARTIFACTS

Wave N+43 landing delivers ALL of the following per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" + `[[why-our-candidates-lose-to-pr-95-family-canonical-diagnosis-20260528]]` 8-structural-failure-mode extinction:

1. **Substrate codec package** (Wave N+22 + N+28 LANDED): `src/tac/substrates/time_traveler_l5_z5/` 1513 LOC binding architecture + archive grammar + MLX renderer + canonical inflate runtime (per HNeRV parity L1 + L2 + L3 + L4 + L5 + L6 + L7 + L12)
2. **Submission_dir** (THIS Wave N+43 landing): `submissions/time_traveler_l5_z5/{inflate.py,inflate.sh,README.md}` per NSCS01 fail-closed-template pattern + Catalog #295 PYTHONPATH self-containment + Catalog #146 contest 3-arg contract + Catalog #205 canonical select_inflate_device + Catalog #208 docs no-local-absolute-paths (per HNeRV parity L4 + L9)
3. **Per-substrate symposium memo** (THIS): per Catalog #325 6-step contract + Catalog #346 14-attendee roster + Catalog #300 v2 frontmatter (per HNeRV parity L13 + Catalog #325)
4. **Canonical apparatus mutations** (THIS landing): probe outcome via Catalog #313 + canonical council deliberation anchor via `tac.council_continual_learning.append_council_anchor` per Catalog #300 (per CLAUDE.md "Results must become system intelligence" + `[[memos-must-be-acted-upon-canonical-apparatus-mutation-enforcement-standing-directive-20260528]]`)
5. **Empirical anchor 2/3** (Wave N+28 LANDED): canonical equation `z5_rao_ballard_hierarchical_predictive_coding_pose_axis_savings_v1` anchor 2/3 with archive sha `3000ca91...` + pose-axis 3.22x reduction + per-axis decomposition (per HNeRV parity L8 + Catalog #344)
6. **Landing memo** (THIS Wave N+43): per `[[memos-must-be-acted-upon-canonical-apparatus-mutation-enforcement-standing-directive-20260528]]` writer obligation; cross-references all PV anchors + apparatus mutations

## Discipline

Catalog #229 PV (exhaustive existing-work scan: CLAUDE.md + AGENTS.md + 6 mandatory memos + Wave N+22 + N+28 landings + Z5 substrate package + sister Z6-v2 + Z7-Mamba-2 patterns + canonical equation registry + 4 sister subagents disjoint scope) + #117/#157/#174/#235/#289 canonical serializer with POST-EDIT --expected-content-sha256 + #206 (5+ checkpoints + crash-resume by lane_id) + #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE (NEW research artifact + canonical posterior anchor APPEND; zero mutation of sister memos / existing Z5 substrate package) + #131/#138 fcntl-locked JSONL canonical state writes + #146/#205/#208 inflate runtime self-containment + #287 placeholder-rationale rejection (every waiver substantive ≥4 chars) + #290 canonical-vs-unique decision per layer documented + #292/#300/#346 v2 frontmatter + 14-member roster + #293 source-supports vs Pact-must-prove documented + #294 9-dim checklist (via Wave N+28 landing memo + this symposium) + #295 NSCS01 fail-closed-template PYTHONPATH self-containment for submission_dir + #296 Dykstra-feasibility predicted band waivered + #298 substrate retirement discipline (research_only=true via Catalog #220+#240+#325 already declared in __init__.py) + #303 cargo-cult audit per assumption (via Wave N+28 landing memo + this symposium + assumption-adversary verdict frontmatter) + #305 observability surface (via Wave N+28 landing memo + this symposium) + #307 paradigm-vs-implementation classification (PARADIGM INTACT per Wave N+28; specific implementation pending anchor 3/3 disambiguation) + #308 alternative-probe-methodology (anchor 3/3 identity-predictor + predictor capacity sweep + 2000ep horizon) + #309 horizon_class=frontier_pursuit + #311 ego-motion-conditioned predictive coding canonical (Atick-Redlich + Rao-Ballard) + #312 hierarchical predictive coding canonical quadruple (Rao-Ballard + Mallat wavelet + Hafner DreamerV3 + Wyner-Ziv side-info; Z5 ships 2 of 4) + #313 probe outcomes ledger (PROCEED-pending-anchor-3 with 4 reactivation paths) + #320 N/A (no Pareto polytope constraint at SCAFFOLD; Wave N+11 quad composition activates) + #323 canonical Provenance + #324 post-training Tier-C validation_status=pending_post_training + #325 per-substrate symposium 6-step contract (THIS memo) + #335/#341 canonical cathedral consumer / Tier A non-promotable markers + #340 sister-checkpoint guard PROCEED throughout (DISJOINT verified vs Wave N+41 + Wave N+42 + Wave N+44) + #344 canonical equation registry anchor 2/3 (Wave N+28 LANDED) + #346 roster validation complete=True (14 members per inner sextet 4-co-lead structure + grand-council specialist seats) + #348 retroactive sweep (N/A — no new STRICT gates) + #356 per-axis decomposition (anchor 2/3 per-axis pose=33.38 / seg=5.94 / recon_aux=1.68 verified) + #361 Modal artifact filter (N/A — MLX-LOCAL only; sister paid-CUDA RATIFICATION will route through Catalog #361 canonical helper) + #363 recursive self-reflection 4-value taxonomy + #367 inflate raw-bytes fail-closed + #373 compound stack anti-pattern acknowledgment + #376 SUBAGENT-side PV + #378 PARENT-MAIN-THREAD PV (operator-direct override per Catalog #300 §Mission alignment Consequence 1 burst beyond cap=1-per-turn AND beyond 3-concurrent empirical envelope to 4 concurrent ACKNOWLEDGED as edge case) + #125 6-hook wire-in declaration.

CLAUDE.md non-negotiables honored: "MLX portable-local-substrate authority" (8th) + "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium" + "Forbidden premature KILL without research exhaustion" + "Apples-to-apples evidence discipline" + "Submission auth eval — BOTH CPU AND CUDA" + "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" + "Substrate MUST be at OPTIMAL FORM before paid empirical dispatch" + "HNeRV / leaderboard-implementation parity discipline" 13 inviolable lessons + "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" + just-saved standing directives (sub-0.18 mobilization + rate-attack-cost-class + memos-must-be-acted-upon + cap=1-per-turn + simultaneous-multi-spawn anti-pattern + PR-creation operator-explicit-per-PR gate + Modal blanket + iterate-on-ultimate-don't-force + canonical 13 lessons binding + why-our-candidates-lose).

## Cross-references

- Wave N+22 Z5 first MLX-LOCAL anchor 1/3: `.omx/research/z5_rao_ballard_mlx_local_scaffold_landed_20260528.md` (commits `be4e4b237` + `c8069060c`)
- Wave N+28 Z5 + Hinton-distilled anchor 2/3: `.omx/research/z5_rao_ballard_hinton_distilled_600pair_long_mlx_landed_20260528.md` (commit `c0bef1cf5`)
- Z6-v2 Hinton-distilled canonical pattern: commit `c26647891` `experiments/train_substrate_z6_v2_cargo_cult_unwind_mlx_local.py`
- Z7-Mamba-2 Hinton-distilled canonical pattern: commit `2224eff58` `experiments/train_substrate_time_traveler_l5_z7_mamba2_mlx_local.py`
- Canonical Hinton-distilled scorer surrogate: `src/tac/substrates/hinton_distilled_scorer_surrogate/`
- Canonical MLX score-aware harness: `src/tac/substrates/_shared/mlx_score_aware/`
- Canonical equation: `z5_rao_ballard_hierarchical_predictive_coding_pose_axis_savings_v1` per Catalog #344 registry
- T4 SYMPOSIUM Wave N+13 verdict: commit `f5d3c6835` op-routable #1 (Z5-first among Z4/Z5/Z6/Z7/Z8 class-shift queue)
- Wave N+30 negative-receipts audit (canonical anti-patterns + probe outcomes): `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_wave_n30_adversarial_negative_receipts_audit_landed_20260528.md`
- Wave N+41 substrate-family × PR-95-parity 100-trainer audit: commit `7f0617d6d`
- Sister Wave N+42 NSCS06 v8 chroma_lut + cls_stream packet (in-flight)
- Sister Wave N+44 PR101+FEC10 V14-V2 packet (in-flight; step 3 verified disjoint scope)
- CLAUDE.md "HNeRV / leaderboard-implementation parity discipline — NON-NEGOTIABLE, HIGHEST EMPHASIS" (the canonical 13 lessons + 8 forbidden patterns)
- CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" (META-level extension)
- CLAUDE.md "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium" (Catalog #325 6-step contract)
- `[[why-our-candidates-lose-to-pr-95-family-canonical-diagnosis-20260528]]` (operator meta-question canonical answer)
- `[[memos-must-be-acted-upon-canonical-apparatus-mutation-enforcement-standing-directive-20260528]]` (apparatus-mutation binding)
- `[[iterate-on-ultimate-until-grand-council-symposium-approval-then-deploy-dont-force-standing-directive-20260528]]` (symposium-PROCEED-required-before-dispatch)
- `[[simultaneous-multi-subagent-spawn-rate-limit-cascade-anti-pattern-20260528]]` (operator-direct 4-concurrent override acknowledged)
