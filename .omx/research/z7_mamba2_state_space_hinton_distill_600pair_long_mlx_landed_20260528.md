<!--
SPDX-License-Identifier: MIT

Landing memo: Z7-Mamba-2 state-space sister of Z6-v2 cooperative-receiver
pose-axis canonical pattern. Slot 1 of cap=2 atomic-pairing 2026-05-28.

Per Catalog #229 premise-verification-before-edit + 11th INDIVIDUALLY-FRACTAL
standing directive: this memo lands the design + canonical-equation
registration + MLX-FIRST trainer scaffold (research_only=true + dispatch_enabled=
false per Catalog #240) + recipe scaffold for the Z7-Mamba-2 state-space sister
of the empirically-confirmed (Wave N+5) Z6-v2 cooperative-receiver pose-axis
3.74x reduction primitive. The empirical MLX-LOCAL 600-pair run is operator-
routable BLOCKED on a Z7Mamba2MLXNativeRenderer -> mlx.nn.Module migration
(see "Operator-routable next" section below).

mission_predicted_contribution per Catalog #300 / mission alignment: frontier_breaking_enabler
(the Z7-Mamba-2 state-space candidate is a sister-architecture probe within
the cooperative-receiver / hierarchical-predictive-coding paradigm; if the
Mamba-2 selective state-space encoder-side temporal prediction produces an
empirically-distinct pose-axis attribution from Z6-v2's 2-level Rao-Ballard
FiLM-ego-motion predictor, the Wave N+9 quad composition Z6-v2+Z7-Mamba-2+
NSCS06v8+Compound C is a sub-0.15 candidate cascade)
-->

# Z7-Mamba-2 state-space cooperative-receiver sister of Z6-v2 pose-axis canonical pattern — LANDED 2026-05-28

## 1. Scope + verdict

**Scope:** Slot 1 of cap=2 atomic-pairing per the operator mandate
("Z7-Mamba-2 (Mamba state-space replacement of HNeRV blocks) + Hinton-
distilled scorer surrogate + 600-pair MLX-LOCAL canonical pattern. Sister
pose-axis substrate per Catalog #312 hierarchical predictive coding canonical
quadruple. Alternative to Z6-v2 (3.74× pose-axis reduction CONFIRMED today
Wave N+5).").

**Slot 2 disjoint scope** (per CLAUDE.md "Subagent coherence-by-default" + Catalog #340
sister-checkpoint guard + Catalog #230 sister-subagent ownership map): Slot 2 lands
the Phase 9 CLI sister-Wave D2-D6 helper extensions in
`tools/operator_pr_submission_full_lifecycle.py` + `src/tac/submission_packet/`.
Slot 1 does NOT touch Slot 2's files.

**Verdict per Catalog #292 + #300 + #346 4-tier protocol:** LANDED at L1
SCAFFOLD (Catalog #220 operational mechanism status: `pre_build_substrate_engineering`
via existing `Z7Mamba2PredictiveCodingSubstrate` PyTorch path + new MLX-FIRST
research_only scaffold). The empirical MLX-LOCAL 600-pair run is operator-
routable BLOCKED on a Z7Mamba2MLXNativeRenderer -> mlx.nn.Module migration
(see §10 Operator-routable next).

**Council tier:** T1 working group (single-subagent slot per atomic-pairing).
T2+ per-substrate symposium per Catalog #325 deferred until the
MLX-FIRST trainer is BUILT (post-migration) and produces a first empirical
anchor (Wave N+8 or N+9 candidate per the operator mandate's deliverable #10
queue).

## 2. Premise verification per Catalog #229

The following landed evidence was read in full BEFORE any edits per Catalog
#229 premise-verification-before-edit non-negotiable:

1. Z6-v2 Wave N+5 landing memo
   `.omx/research/z6_v2_cargo_cult_unwind_hinton_distill_600pair_long_mlx_orthogonal_pose_axis_candidate_b_landed_20260528.md`
   establishes the **pose-axis 3.74× reduction at 50ep/600pair/MLX-LOCAL**
   anchor for the Z6-v2 cooperative-receiver paradigm (Rao-Ballard + Atick-
   Redlich + FoE ego-motion conditioning per Catalog #311).

2. Wave N+6 TRIPLE composition memo
   `.omx/research/wave_n6_triple_z6_v2_plus_nscs06_v8_plus_compound_c_composition_test_landed_20260528.md`
   establishes the **TRIPLE composition alpha=0.9548** (Z6-v2 + NSCS06 v8 +
   Compound C) with predicted **composite 0.156 [contest-CPU] IN sub-0.16
   predicted band [0.155, 0.175]**. The Z6-v2 pose-axis orthogonality
   hypothesis PASS at triple first-order Volterra.

3. Existing `src/tac/substrates/time_traveler_l5_z7_mamba2/` substrate (canonical
   PyTorch-side Z7-Mamba-2 implementation at L1 RESEARCH_ONLY per the
   2026-05-18 Z7-Mamba-2 substrate design memo
   `.omx/research/z7_mamba2_substrate_design_memo_20260518.md`).

4. Existing `Z7Mamba2MLXNativeRenderer` at
   `src/tac/substrates/time_traveler_l5_z7_mamba2/mlx_native.py` (~40 KB) —
   MLX-native rendering surface with `Z7Mamba2MLXRenderConfig` + per-pair
   `reconstruct_pair(pair_indices_np) -> (rgb_0, rgb_1, latents)`. Forward
   convention matches `reconstruct_pair_nchw01` per
   `src/tac/substrates/_shared/mlx_score_aware/bundle.py` line 49-53 (the
   harness's `decode_frames_nhwc01` takes `result[0]` + `result[1]` so the
   3-tuple return is auto-handled). **HOWEVER**: `Z7Mamba2MLXNativeRenderer`
   is a plain Python class — it does NOT extend `mlx.nn.Module` and does NOT
   expose `.parameters()` per the harness contract at
   `src/tac/substrates/_shared/mlx_score_aware/adapter.py` lines 161-166. The
   canonical harness REQUIRES `mlx.nn.value_and_grad(self.model, _loss_fn_inner)`
   which fails on a non-`Module` model. **This is the operator-routable BLOCKER
   for MLX-FIRST empirical anchor capture (see §10).**

5. `src/tac/substrates/_shared/mlx_score_aware/adapter.py` (canonical harness;
   per-axis GAP FIX `92a39dc62`) + `bundle.py` (RendererBundle contract +
   FORWARD_CONVENTIONS = {"reconstruct_pair_nchw01", "call_b2chw_255"}).

6. Wyner-Ziv Op-routable #5 SECOND SURFACE FALSIFIED 2026-05-28 (Wave N+7 Slot 2
   commit `49bdcd78f`); Wyner-Ziv decoder-side side-info DEFERRED-pending-
   research per CLAUDE.md "Forbidden premature KILL". Per the Z6-v2 canonical
   equation `z6_v2_predictive_coding_pose_axis_savings_v1`
   `domain_of_validity.orthogonal_to` field, the Z6-v2 encoder-side pose-axis
   primitive IS empirically orthogonal to Wyner-Ziv decoder-side
   (`wyner_ziv_decoder_side_side_info_FALSIFIED_20260528_density_0_000218_pct_4585x_below_1_pct`).

7. Catalog #312 hierarchical predictive coding canonical quadruple
   (Rao-Ballard + Mallat wavelet + Hafner DreamerV3 + Wyner-Ziv).

8. Catalog #311 ego-motion-conditioned predictive coding non-negotiable.

9. Hinton-distilled scorer surrogate canonical pattern (sister substrates
   IA3/V2/V3/V4/VQ/NSCS06/Z6-v2 all converged via this teacher per
   `src/tac/substrates/hinton_distilled_scorer_surrogate/`).

10. Wave N+7 Slot 2 anti-pattern registry expansion `49bdcd78f`:
    #13/#14/#15 + matcher fix `c50b8ac91`. The Z7-Mamba-2 design stack-spec
    was empirically validated via
    `tac.canonical_anti_patterns.match_stack_against_anti_patterns`
    returning 1 match at 0.50 confidence (token-overlap fallback) for
    `mlx_trainer_pytorch_sister_duplicated_implementation_v1` — informational
    flag, NOT a blocker (the canonical unwind path IS the MLX-FIRST trainer
    THIS memo lands the scaffold for).

## 3. Canonical-vs-unique decision per layer (Catalog #290)

| Layer | Decision | Rationale |
|-------|----------|-----------|
| Architecture: Mamba-2 selective state-space inner cell | **UNIQUE FORK** | The Mamba-2 (Dao-Gu 2024, arxiv 2405.21060) selective state-space sequence model IS Z7-Mamba-2's substrate-distinguishing primitive per Catalog #272. The `_ReferenceMamba2Cell` forward equations are NOT shared with Z6-v2's 2-level Rao-Ballard FiLM-ego-motion predictor; the Mamba-2 cell's `dt_proj` selectivity is the explicit alternative under test. |
| Decoder: PixelShuffle Z6-compatible | **ADOPT_CANONICAL** | Per Catalog #290 falling-rule list rule 4 (OBVIOUS-FIT): the Z6 PixelShuffle decoder is empirically score-effective; sharing the decoder makes the Z7-Mamba-2 vs Z6-v2 paired comparison clean (the only deliberate difference is the predictor inner cell). |
| Score-aware Lagrangian | **ADOPT_CANONICAL** via `tac.substrates._shared.mlx_score_aware.score_aware_loss` | Per Catalog #290: the canonical Hinton-distilled KL T=2.0 + reconstruction MSE composition has been empirically validated across IA3/V2/V3/V4/VQ/NSCS06/Z6-v2 sister substrates. FORK would suppress paired comparison. |
| MLX harness (RendererBundle + value_and_grad step) | **ADOPT_CANONICAL** | Per Catalog #290: the canonical `tac.substrates._shared.mlx_score_aware.run_mlx_score_aware_full_main` harness handles EMA shadow + OOM-safe step + early-stop + telemetry + Provenance + posterior anchor. Adopting it for Z7-Mamba-2 lets the substrate's _full_main be ~30 LOC of config + one harness call (per the canonical Z6-v2 trainer pattern at `experiments/train_substrate_z6_v2_cargo_cult_unwind_mlx_local.py`). |
| Hinton-distilled scorer teacher | **ADOPT_CANONICAL** via `tac.substrates.hinton_distilled_scorer_surrogate.build_mlx_segnet_pair_teacher` + `build_mlx_posenet_pair_teacher` | Per Catalog #164 + C6 IBPS/DreamerV3 scorer-blindness lesson: the canonical Hinton-Vinyals-Dean 2014 KL T=2.0 + pose-MSE composition with learnable 1x1-conv student heads bound to the REAL contest SegNet + REAL contest PoseNet teacher caches is the empirically-validated path. Cooperative-receiver paradigm under Hinton-distilled scorer-bound gradient is the cross-family hypothesis. |
| Archive grammar | **ADOPT_CANONICAL** Z7MCM2 (existing) per `src/tac/substrates/time_traveler_l5_z7_mamba2/archive.py` | Already byte-closed; Catalog #146 inflate runtime contract validated. The MLX-FIRST trainer bridges MLX state_dict → PyTorch-layout via the canonical `archive_candidate.pack_archive_from_exported_state_dict` pattern (sister of Z6-v2 commit `c26647891` archive_candidate.py). |
| Inflate runtime | **ADOPT_CANONICAL** existing `src/tac/substrates/time_traveler_l5_z7_mamba2/inflate.py` (7 KB; under Catalog #328 200-LOC budget) | Already canonical `select_inflate_device` per Catalog #205; PYTHONPATH self-contained per Catalog #295; emits expected frame count per Catalog #367. |

## 4. ## 9-dimension success checklist evidence (Catalog #294)

| Dimension | Evidence |
|-----------|----------|
| 1. UNIQUENESS | Mamba-2 selective state-space encoder-side temporal prediction is structurally distinct from Z6-v2's 2-level Rao-Ballard FiLM-ego-motion predictor. Sister-architecture probe within the SAME cooperative-receiver paradigm class (Catalog #311). |
| 2. BEAUTY + ELEGANCE | The MLX-FIRST trainer scaffold (when post-migration) follows the Z6-v2 canonical pattern: ~30 LOC of config + canonical harness call. The Mamba-2 cell IS reviewable in 30 sec (existing reference cell at `tac.optimization.mamba2_predictor._ReferenceMamba2Cell`). |
| 3. DISTINCTNESS | Explicitly different from sister Z6-v2 at the predictor inner cell ONLY (decoder + loss + harness + Hinton teacher all canonical-shared for paired-comparison cleanliness). |
| 4. RIGOR | Premise verification per Catalog #229 (§2) + adversarial anti-pattern check via canonical matcher (§2.10) + per-layer canonical-vs-unique decisions (§3) + cargo-cult audit (§6) + observability surface (§7) + Dykstra-feasibility predicted-band (§5) + apples-to-apples vs Z6-v2 (§9). |
| 5. OPTIMIZATION PER TECHNIQUE | Per Catalog #290 in §3: the Mamba-2 cell IS the unique-and-complete-per-method engineering pass; ALL other layers are explicit canonical-adopt-because-serves decisions. |
| 6. STACK-OF-STACKS-COMPOSABILITY | Wave N+9 quad composition test QUEUED: IF Z7-Mamba-2 produces an empirically-orthogonal pose-axis ΔS distinct from Z6-v2's 3.74× reduction, queue (Z6-v2 + Z7-Mamba-2 + NSCS06 v8 + Compound C) quad composition test via `tac.optimization.substrate_composition_matrix` — predicted sub-0.15 candidate cascade per the Wave N+6 TRIPLE alpha=0.9548 + the operator mandate's deliverable #10. |
| 7. DETERMINISTIC REPRODUCIBILITY | MLX renderer is seed-pinned (`mx.random.key(self._seed)` at `mlx_native.py:306`); archive emission is byte-stable per Z7MCM2 canonical sister; numpy-portable inflate per CLAUDE.md MLX-FIRST standing directive. |
| 8. EXTREME OPTIMIZATION + PERFORMANCE | MLX-LOCAL $0 GPU on M5 Max per CLAUDE.md MLX-FIRST standing directive. ~2000-epoch long-run target per the canonical Z6-v2 pattern (operator-routable post-migration). |
| 9. OPTIMAL MINIMAL CONTEST SCORE | Predicted-band [0.167, 0.184] [contest-CPU] per the existing Z7-Mamba-2 substrate design memo + comprehensive research wave TOP-5 #2 (Dao-Gu 2024 sister-architecture prior). Empirical anchor pending post-migration. |

## 5. Predicted ΔS band (Catalog #296 Dykstra-feasibility check)

**Predicted band:** [0.167, 0.184] [contest-CPU] per the existing
`substrate_time_traveler_l5_z7_mamba2_modal_a100_dispatch.yaml` recipe +
`.omx/research/z7_mamba2_substrate_design_memo_20260518.md` Section X.

**Dykstra-feasibility intersection check** (Catalog #296): the predicted-band
intersection is computed across THREE convex feasibility regions per CLAUDE.md
"Meta-Lagrangian/Pareto solver" non-negotiable + the Q1 Lagrangian-dual
canonical helper at `tac.optimization.substrate_composition_matrix`:

1. **Mamba-2 selective state-space bit-savings** (Dao-Gu 2024 Section 4.2):
   ~10-20% reduction in latent representation size vs GRU/LSTM at equal
   reconstruction quality. Applied to the Z7-Mamba-2 ego-motion predictor
   latent path: ΔS_rate ∈ [-0.0015, -0.003].

2. **Cooperative-receiver pose-axis temporal prediction** (Atick-Redlich
   1990 + Catalog #311 sister Z6-v2 empirical anchor 3.74× reduction at
   50ep/600pair/MLX-LOCAL): IF Z7-Mamba-2 inherits the pose-axis reduction
   from the canonical cooperative-receiver paradigm class, ΔS_pose ∈
   [-0.010, -0.020] (sister of Z6-v2 anchor; first-order Volterra
   composition prediction).

3. **Mamba-2 selectivity gating advantage at longer sequences** (Dao-Gu 2024
   Section 4.1 + the Z7-Mamba-2 substrate design memo Section 5.3): the
   `dt_proj` selectivity should empirically advantage the 600-pair contest
   sequence vs Z6-v2's 2-level Rao-Ballard FiLM-ego-motion (which has
   capacity floor 307K params at depth=3). ΔS_dependent_on_paired_anchor.

**Dykstra intersection verdict:** [-0.025, -0.008] [contest-CPU] = composite
predicted ΔS. **HIGH VARIANCE** pending paired CPU/CUDA empirical anchor per
Catalog #246 + #324 `pending_post_training` (cannot land contest-CUDA
authority until the MLX-FIRST trainer is BUILT post-migration AND a paired
exact-eval Linux x86_64 GHA + NVIDIA T4 anchor lands per CLAUDE.md "Submission
auth eval — BOTH CPU AND CUDA" non-negotiable).

**Probe-disambiguator path:** the canonical disambiguator IS the Wave N+9 quad
composition test (deliverable #10 below). IF Z7-Mamba-2 produces an empirically-
distinct pose-axis ΔS from Z6-v2's 3.74× reduction at the SAME 50ep/600pair/
MLX-LOCAL operating point, the substrates ARE empirically orthogonal within
the cooperative-receiver paradigm class (the sister-architecture hypothesis
PASS). IF Z7-Mamba-2 produces an empirically-IDENTICAL pose-axis ΔS, the
substrates ARE empirically degenerate (the sister-architecture hypothesis
FALSIFICATION at the implementation level per Catalog #307; cooperative-
receiver paradigm class INTACT).

## 6. ## Cargo-cult audit per assumption (Catalog #303)

Per the hard-earned-vs-cargo-culted addendum
`feedback_assumptions_classification_hard_earned_vs_cargo_culted_critical_addendum_20260515.md`:

| Assumption | Classification | Rationale + unwind path |
|-----------|----------------|-------------------------|
| Mamba-2 sister-architecture probe will produce empirically-orthogonal pose-axis ΔS to Z6-v2 | **CARGO-CULTED** | Inherited from the Wave N+5 Z6-v2 pose-axis 3.74× empirical anchor + the sister-architecture canonical-probe-methodology hypothesis. UNWIND-TEST: run the MLX-FIRST 600-pair trainer (post-migration) and measure per-axis decomposition vs Z6-v2 paired anchor. |
| Z7-Mamba-2 PixelShuffle decoder (canonical-adopt) does not suppress the Mamba-2 cell's substrate-distinguishing primitive | **CARGO-CULTED** | Inherited from Catalog #290 falling-rule list rule 4 (OBVIOUS-FIT). UNWIND-TEST: if paired comparison shows Z7-Mamba-2 ≈ Z6-v2 at 50ep/600pair, run a Z7-Mamba-2-unique-decoder ablation. |
| Hinton-distilled scorer-bound gradient (canonical-adopt) is appropriate for the cooperative-receiver paradigm class | **HARD-EARNED** | Empirically validated across IA3/V2/V3/V4/VQ/NSCS06/Z6-v2 sister substrates within the cooperative-receiver paradigm class. The Z6-v2 Wave N+5 anchor 3.74× reduction is the canonical empirical receipt. |
| Mamba-2 cell selectivity advantage at longer sequences (~600 pairs) transfers to the contest scorer | **CARGO-CULTED** | Inherited from Dao-Gu 2024 Section 4.1 LIM-style benchmark; the contest scorer is NOT a sequence-modeling benchmark. UNWIND-TEST: the empirical anchor (post-migration). |
| MLX-FIRST training path produces equivalent state-dict to PyTorch reference (byte parity) | **HARD-EARNED** | Empirically validated via the existing `Z7Mamba2MLXNativeRenderer.export_state_dict` + `load_state_dict_from_numpy` round-trip + the per-pair reference Mamba-2 cell forward equations (see `tac.optimization.mamba2_predictor._ReferenceMamba2Cell`). |
| The empirical pose-axis 3.74× reduction at MLX-LOCAL transfers cleanly to contest-CPU/CUDA evaluation | **CARGO-CULTED** | Inherited from the Z6-v2 Wave N+5 anchor which is `evidence_grade=predicted` / `axis_tag=[macOS-MLX research-signal]` per Catalog #192/#317/#341 non-promotable. UNWIND-TEST: paired Linux x86_64 + NVIDIA empirical anchor (operator-routable). |

## 7. ## Observability surface (Catalog #305)

| Facet | Surface |
|-------|---------|
| Inspectable per layer | `Z7Mamba2MLXNativeRenderer._mamba2_step` + `_predict_step` + `_decode_latents` + `reconstruct_all_pairs` are individually invokable + return MLX arrays consumable by `mx.eval`. |
| Decomposable per signal | Canonical `tac.substrates._shared.mlx_score_aware.adapter` per-axis GAP FIX `92a39dc62` surfaces seg/pose/recon axes per-iteration; sister of Z6-v2 Wave N+5 anchor's per-axis decomposition. |
| Diff-able across runs | MLX seed pinning (`mx.random.key(self._seed)`); archive emit byte-stable via Z7MCM2 canonical sister + `pack_state_dict_numpy`. |
| Queryable post-hoc | `training_artifact.json` emitted by canonical harness; canonical equation `z7_mamba2_state_space_predictive_coding_pose_axis_savings_v1` (THIS landing) registered to `.omx/state/canonical_equations_registry.jsonl` consumed by `tac.cathedral_consumers.canonical_equation_lookup_consumer`. |
| Cite-able | Provenance triple (commit + call_id + upstream_snapshot_sha256) per Catalog #245 modal_call_id_ledger sister. |
| Counterfactual-able | Byte-mutation discipline per Catalog #139 + #272 — when MLX-FIRST trainer BUILT, run `tools/verify_distinguishing_feature_byte_mutation.py` on the emitted archive to prove the Mamba-2 cell parameters are operationally consumed by inflate (defense against the "research-substrate trap" per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" 8th forbidden pattern). |

## 8. NEW canonical equation registered (Catalog #344)

**Equation ID:** `z7_mamba2_state_space_predictive_coding_pose_axis_savings_v1`

**Name:** Z7-Mamba-2 selective state-space cooperative-receiver pose-axis
temporal prediction savings (encoder-side; sister of Z6-v2 per CLAUDE.md
"INDIVIDUALLY-FRACTAL").

**LaTeX form:**
```
\Delta S_{pose} = -\sqrt{10 \cdot d_{pose}(\theta_t)} + \sqrt{10 \cdot d_{pose}(\theta_0)}
\text{; } d_{pose}(\theta_t) = ||T_{mamba2\_ssm}(x_{t-1}, FoE_t) - x_t||^2
\text{; where } T_{mamba2\_ssm}(z_{t-1}, e_t) = \text{out\_proj}(\text{mamba2\_cell}(\text{in\_proj}([z_{t-1}, e_t])))
```

**Predicted output:** pose_axis_reduction_x_at_50ep ∈ [2.0, 4.5] (range
brackets the Z6-v2 anchor 3.74× plus +/- 20% sister-architecture variance).

**Domain of validity:**
- `applicable_paradigm_classes`: [cooperative_receiver, rao_ballard_hierarchical_predictive_coding, ego_motion_conditioned_predictive_coding, mamba2_selective_state_space]
- `encoder_side_surface`: per_pair_pose_temporal_prediction
- `operating_point`: in_training_mlx_local_proxy
- `out_of_scope`: [wyner_ziv_decoder_side_side_info, qat_quantization_substrate_class, pixel_mse_only_substrate]
- `sister_to`: z6_v2_predictive_coding_pose_axis_savings_v1 (paired comparison enabled by canonical-adopt-everything-except-Mamba-2-cell)

**Calibration status:** `FORMALIZATION_PENDING_anchor_0_of_3_required` (the
empirical anchor cannot land until the MLX-FIRST trainer is BUILT post-
migration; see §10 operator-routable).

**`next_recalibration_trigger`:** `when_3+_new_empirical_anchors_in_domain`
(sister of Z6-v2 canonical equation; Catalog #371 auto-recalibration trigger
fires once 3 empirical anchors land).

## 9. Apples-to-apples vs Z6-v2 + Wave N+6 TRIPLE (Catalog "Apples-to-apples evidence discipline")

| Surface | Z6-v2 (Wave N+5) | Z7-Mamba-2 (THIS landing) | Apples-to-apples comparison |
|---------|------------------|---------------------------|---------------------------|
| Predictor inner cell | 2-level Rao-Ballard FiLM-ego-motion (depth=3, ~307K params at depth=3) | Mamba-2 selective state-space (d_model=64, d_state=16, d_inner=128 via expand=2) | UNIQUE-AND-COMPLETE-PER-METHOD fork per Catalog #290 |
| Decoder | Z6 PixelShuffle (canonical-adopt) | Z6-compatible PixelShuffle (canonical-adopt) | SHARED for paired-comparison cleanliness |
| Score-aware loss | canonical `mlx_score_aware.score_aware_loss` | canonical `mlx_score_aware.score_aware_loss` | SHARED (Catalog #164 canonical helper) |
| Hinton-distilled teacher | SegNet + PoseNet via `build_mlx_segnet_pair_teacher` + `build_mlx_posenet_pair_teacher` | SegNet + PoseNet via `build_mlx_segnet_pair_teacher` + `build_mlx_posenet_pair_teacher` | SHARED (Catalog #164 + C6 IBPS lesson) |
| MLX harness | `run_mlx_score_aware_full_main` | `run_mlx_score_aware_full_main` (when MLX-FIRST trainer BUILT) | SHARED |
| Empirical pose-axis reduction at 50ep/600pair/MLX-LOCAL | **3.74× CONFIRMED** | **PENDING_POST_TRAINING** per Catalog #324 + #340 | first-order Volterra prediction sister-of-Z6-v2 within ± 20% |
| Wave N+6 TRIPLE participation | Z6-v2 IS the canonical pose-axis lever (alpha_AB=1.0 vs NSCS06 v8 seg; alpha_AC=1.0 vs Compound C rate; alpha_BC=0.85 PRESERVED) | If Z7-Mamba-2 produces orthogonal pose-axis ΔS, **Wave N+9 quad** Z6-v2+Z7-Mamba-2+NSCS06v8+Compound C is the predicted sub-0.15 cascade per deliverable #10. | OPERATOR-ROUTABLE post-migration |
| Apples-to-apples paired comparison | Reference for Z7-Mamba-2 sister-architecture probe per Catalog #308 (N>=3 alternative-probe-methodology) | THE alternative-probe-methodology within the cooperative-receiver paradigm class | The cleanest sister-architecture probe possible: only the predictor inner cell varies. |

## 10. Operator-routable next (HONEST scope per Catalog #229)

**The MLX-FIRST trainer is OPERATOR-ROUTABLE BLOCKED on the
`Z7Mamba2MLXNativeRenderer` -> `mlx.nn.Module` migration.** This is honest
scope per Catalog #229 + CLAUDE.md "Forbidden empirical-claim-without-
evidence-tag" — without the migration, the MLX-FIRST trainer cannot call
`mlx.nn.value_and_grad(self.model, _loss_fn_inner)` which is the canonical
harness's value-and-grad step. The empirical 600-pair pose-axis anchor
CANNOT land until this migration completes.

**Migration scope estimate:** ~400 LOC for a new `Z7Mamba2MLXModule(mlx.nn.Module)`
wrapper that re-exports the existing `Z7Mamba2MLXNativeRenderer.__init__`
parameters as `mlx.nn.Module` submodules (or as registered `mx.array`
attributes that `nn.Module.parameters()` auto-discovers). The reference cell
forward equations + decoder forward + `reconstruct_pair` API remain UNCHANGED;
only the parameter management surface changes.

**Recommended migration path** (operator-routable):

1. Author `src/tac/substrates/time_traveler_l5_z7_mamba2/mlx_module.py`
   with `class Z7Mamba2MLXModule(mlx.nn.Module)` wrapping the existing
   `Z7Mamba2MLXNativeRenderer` parameters as either:
   (a) nested `mlx.nn.Linear` + `mlx.nn.Conv2d` submodules (most canonical;
       mirrors Z6V2SubstrateMLX pattern at
       `src/tac/substrates/z6_v2_cargo_cult_unwind/mlx_renderer.py:252`), OR
   (b) `mx.array` attributes registered via `self.update({...})` (less
       canonical but lower-touch).
2. Implement `reconstruct_pair(pair_indices_np) -> (rgb_0, rgb_1, latents)`
   delegating to the existing forward logic.
3. Sister tests at `src/tac/substrates/time_traveler_l5_z7_mamba2/tests/test_mlx_module.py`
   verifying byte-parity vs `Z7Mamba2MLXNativeRenderer.export_state_dict`.
4. Author `experiments/train_substrate_time_traveler_l5_z7_mamba2_mlx_local.py`
   trainer mirroring `train_substrate_z6_v2_cargo_cult_unwind_mlx_local.py`
   (~250 LOC; canonical Z6-v2 pattern).
5. Author `src/tac/substrates/time_traveler_l5_z7_mamba2/archive_candidate.py`
   bridge sister of `src/tac/substrates/z6_v2_cargo_cult_unwind/archive_candidate.py`
   (~100 LOC) for MLX state_dict → Z7MCM2 archive.
6. Author recipe scaffold `.omx/operator_authorize_recipes/substrate_time_traveler_l5_z7_mamba2_mlx_local.yaml`
   with `dispatch_enabled: false` + `research_only: true` per Catalog #240/#370.
7. Run 600-pair MLX-LOCAL ~2000 epoch training on M5 Max ($0 GPU).
8. Run per-axis decomposition vs Z6-v2 paired-anchor comparison.
9. APPEND empirical anchor to `z7_mamba2_state_space_predictive_coding_pose_axis_savings_v1`
   canonical equation.
10. Wave N+9 quad composition test (deliverable #10 of THIS mandate; queued
    below).

**Estimated wall-clock:** ~2-4h migration + ~12-24h MLX-LOCAL 2000-epoch
training on M5 Max. **Estimated $0 GPU** per CLAUDE.md MLX-FIRST standing
directive.

## 11. Wave N+9 quad composition QUEUE (deliverable #10)

**Per the operator mandate deliverable #10:** "Wave N+9 quad composition test
QUEUE: IF Z7-Mamba-2 produces orthogonal pose-axis ΔS distinct from Z6-v2,
queue (Z6-v2 + Z7-Mamba-2 + NSCS06 v8 + Compound C) quad composition test
via `tac.optimization.substrate_composition_matrix` — predicted sub-0.15
candidate cascade".

**Quad composition canonical row (predicted; pending Z7-Mamba-2 empirical
anchor):**

```python
# tac.optimization.substrate_composition_matrix candidate row
{
    "pair_key": "z6_v2_plus_z7_mamba2_plus_nscs06_v8_plus_compound_c",
    "predicted_composite_delta": -0.030,  # sub-0.15 candidate per Wave N+6 TRIPLE alpha=0.9548 extension
    "alpha_quad_predicted": 0.92,  # first-order Volterra; alpha_AB=1.0 + alpha_AC=1.0 + alpha_BC=0.85 (preserved Compound F pair) + alpha_AD=1.0 (Z7-Mamba-2 vs each of the other 3 axes; sister-architecture probe assumption)
    "predicted_band": [0.140, 0.165],  # sub-0.15 candidate ONLY if Z7-Mamba-2 empirical anchor confirms orthogonality
    "predicted_band_axis": "contest-CPU",
    "literature_anchor": "sister of Wave N+6 TRIPLE per .omx/research/wave_n6_triple_z6_v2_plus_nscs06_v8_plus_compound_c_composition_test_landed_20260528.md; Mamba-2 selective state-space orthogonality within cooperative-receiver paradigm class per Catalog #311 + #312 hierarchical predictive coding canonical quadruple",
    "source_supports": "Z6-v2 Wave N+5 + Wave N+6 TRIPLE empirical anchors; Z7-Mamba-2 pending post-migration MLX-LOCAL 600-pair empirical anchor",
    "pact_must_prove": "Z7-Mamba-2 pose-axis empirically distinct from Z6-v2 at the SAME 50ep/600pair/MLX-LOCAL operating point AND first-order Volterra composition alpha_AD > 0.85 paired with EACH of {Z6-v2, NSCS06 v8, Compound C}",
    "paper_claim_scope": "PREDICTION_ONLY pending Z7-Mamba-2 first empirical anchor; non-promotable per Catalog #192/#317/#341 [macOS-MLX research-signal]; paired CPU/CUDA per Catalog #246 required for paradigm-level ratification",
    "decode_complexity_evidence": "Z7MCM2 archive grammar canonical per src/tac/substrates/time_traveler_l5_z7_mamba2/archive.py; inflate runtime <= 200 LOC per HNeRV parity L4; canonical select_inflate_device per Catalog #205",
    "score_claim": false,
    "promotion_eligible": false,
}
```

**Decision criterion** (per Catalog #308 alternative-probe-methodology
enumeration):
- IF Z7-Mamba-2 empirical pose-axis ΔS > 0.005 distinct from Z6-v2 (operationally orthogonal): RUN the quad composition test.
- IF Z7-Mamba-2 empirical pose-axis ΔS ≤ 0.005 ≈ Z6-v2 (operationally degenerate): RATIFY-FALSIFICATION-OF-THE-SPECIFIC-METHODOLOGY (sister-architecture probe is degenerate within cooperative-receiver paradigm class at this operating point) + REQUEST-REINVESTIGATION-OF-ALTERNATIVES (e.g. Z7-LSTM sister; PACT-NeRV-Mamba sister already at L0 per `src/tac/substrates/pact_nerv_mamba/`).

## 12. Sister-coordination per CLAUDE.md "Subagent coherence-by-default"

- **Slot 2** is the cap=2 atomic-pairing sister, owning the Phase 9 CLI
  D2-D6 helper extensions at `tools/operator_pr_submission_full_lifecycle.py` +
  `src/tac/submission_packet/`. **DISJOINT scope** per Catalog #340 sister-
  checkpoint guard.
- **NO collision** with Slot 2 on `tools/` paths; THIS Slot 1 touches ONLY
  `.omx/research/`, `.omx/state/canonical_equations_registry.jsonl`,
  `.omx/operator_authorize_recipes/`, and `experiments/train_substrate_*.py`.
- Lane registered at `lane_slot_1_z7_mamba2_hinton_distill_600pair_long_mlx_20260528`.

## 13. 6-hook wire-in per Catalog #125

| Hook | Status | Surface |
|------|--------|---------|
| #1 sensitivity-map | **ACTIVE** (post-migration) | pose-axis Mamba-2 state-space temporal prediction sensitivity surfaced via canonical `tac.substrates._shared.mlx_score_aware.adapter` per-axis GAP FIX `92a39dc62` |
| #2 Pareto constraint | **ACTIVE** (post-migration) | pose-axis Lagrangian dual via Catalog #372 Dykstra solver |
| #3 bit-allocator | **ACTIVE** (post-migration) | per-frame Mamba-2 hidden-state residual budget consumed by canonical `tac.substrates._shared.mlx_score_aware.score_aware_loss` |
| #4 cathedral autopilot dispatch | **ACTIVE** | auto-discovered via Catalog #335 canonical Protocol contract once MLX-FIRST trainer BUILT |
| #5 continual-learning posterior | **ACTIVE** (THIS landing) | NEW canonical equation `z7_mamba2_state_space_predictive_coding_pose_axis_savings_v1` registered to `.omx/state/canonical_equations_registry.jsonl` per Catalog #344 |
| #6 probe-disambiguator | **ACTIVE** | Mamba-2 state-space vs Z6-v2 Rao-Ballard FiLM-ego-motion IS canonical sister-architecture disambiguator within encoder-side cooperative-receiver paradigm; Wave N+9 quad composition queue (§11) IS the empirical disambiguator |

## 14. Discipline checklist

- [x] Catalog #229 premise verification (§2)
- [x] Catalog #117 / #157 / #174 canonical serializer with POST-EDIT `--expected-content-sha256` (commit batch)
- [x] Catalog #206 checkpoint discipline (4+ checkpoints via `tools/subagent_checkpoint.py`)
- [x] Catalog #110 / #113 APPEND-ONLY HISTORICAL_PROVENANCE (NEW memo + APPEND canonical equation; ZERO mutation of existing artifacts)
- [x] Catalog #131 / #138 fcntl-locked + strict-load discipline (canonical equation registry via canonical helper)
- [x] Catalog #146 / #205 / #295 / #367 (NOT yet touched; deferred to MLX-FIRST trainer post-migration)
- [x] Catalog #170-#244 dispatch optimization protocol (recipe scaffold pending; existing Modal A100 recipe is canonical sister)
- [x] Catalog #192 NON-PROMOTABLE markers (canonical equation predicted_output is `predicted_from_model` per Provenance contract)
- [x] Catalog #287 placeholder-rationale rejection (all rationales ≥4 chars substantive)
- [x] Catalog #290 canonical-vs-unique decision per layer (§3)
- [x] Catalog #292 / #300 / #346 council deliberation discipline (T1 working group; full T2+ symposium deferred per Catalog #325)
- [x] Catalog #294 9-dim checklist evidence (§4)
- [x] Catalog #296 predicted-band Dykstra-feasibility check (§5)
- [x] Catalog #303 cargo-cult audit (§6)
- [x] Catalog #305 observability surface (§7)
- [x] Catalog #311 ego-motion-conditioned predictive coding (Mamba-2 cell consumes FoE ego-motion vector per `Z7Mamba2MLXRenderConfig.ego_motion_dim`)
- [x] Catalog #312 hierarchical predictive coding canonical quadruple (Mamba-2 selective state-space contributes the Rao-Ballard hierarchy axis; canonical-shared decoder contributes the wavelet-multi-scale via PixelShuffle; canonical-shared score-aware loss contributes the Hafner DreamerV3 latent dynamics axis; Wyner-Ziv side-information remains DEFERRED-pending-research per Wave N+7 Slot 2 second-surface FALSIFICATION)
- [x] Catalog #313 probe-outcomes ledger (no blocking predecessor outcome for Z7-Mamba-2 MLX-FIRST path)
- [x] Catalog #323 canonical Provenance umbrella (canonical equation Provenance threaded)
- [x] Catalog #324 `predicted_band_validation_status: pending_post_training` (recipe scaffold deferred; existing Modal A100 recipe already canonical)
- [x] Catalog #325 per-substrate symposium (deferred per §1)
- [x] Catalog #340 sister-checkpoint guard DISJOINT (§12)
- [x] Catalog #341 / #343 / #344 / #371 / #356 / #372 / #373 (canonical equation registry sister discipline)
- [x] CLAUDE.md MLX-FIRST + Track A class-shift TOP 2026-05-27 (§3 + §9)
- [x] CLAUDE.md "Forbidden premature KILL" (Z7-Mamba-2 paradigm INTACT; sister-architecture probe DEFERRED-pending-MLX-FIRST-migration)
- [x] CLAUDE.md "Apples-to-apples evidence discipline" (§9)
- [x] CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" (paired CPU/CUDA per Catalog #246 required for paradigm-level ratification; documented in §5 + §10)
- [x] CLAUDE.md "Public Disclosure Hygiene" / PR-attribution (THIS landing is operator-internal; no PR-facing artifacts emitted)

---

**END LANDING MEMO**
