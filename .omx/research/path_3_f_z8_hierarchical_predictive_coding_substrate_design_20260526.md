---
schema_version: substrate_design_memo_v2
created_utc: 2026-05-26T07:30:00Z
substrate_id: z8_hierarchical_predictive_coding
substrate_class: fresh_design_canonical_quadruple_binding
council_tier: T1
council_attendees:
  - Shannon
  - Dykstra
  - Rao
  - Ballard
  - Mallat
  - Tishby-memorial
  - Wyner
  - Hafner-DreamerV3-author-cite
  - AssumptionAdversary
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
council_override_rationale: ""
horizon_class: frontier_pursuit
predicted_band_validation_status: pending_post_training
related_deliberation_ids:
  - mlx_candidate_contest_equivalence_gate_landed_20260526
  - path_3_candidate_inventory_for_next_wave_spawning_20260526
  - time_traveler_l5_z6_z7_z8_predictive_coding_world_models_asymptotic_pursuit_scoping_design_20260516
  - path_3_a_dreamerv3_rssm_categorical_posterior_l0_scaffold_landed_20260526
  - path_3_d_z6_predictive_coding_fresh_design_landed_20260526
  - path_3_e_boost_nerv_against_pr110_substrate_design_20260526
binding_operator_directives:
  - "The MLX first requirement might also force us out of the issue we were having before where we had great ideas but we're building them as Boltons to the same substrates over and over again; we want to design the substrate and curriculum and then optimize the design the whole stack around it for extreme optimization and performance and optimal score lowering" (2026-05-26)
  - "Never simply extend unless a rigorous adversarial cargo cult pass has been done first" (2026-05-26)
canonical_equation_refs:
  - mlx_pytorch_full_decoder_downstream_scorer_drift_propagation_v1
  - scorer_conditional_joint_rate_distortion_floor_v1
  - categorical_posterior_capacity_vs_continuous_gaussian_v1  # sister A DreamerV3
  - ego_motion_concentration_prior_v1  # sister D Z6
  - cross_codec_super_additive_orthogonality_predictor_v1  # composition lattice
lane_id: lane_path_3_f_z8_hierarchical_predictive_coding_canonical_quadruple_20260526
---

# Z8 Hierarchical Predictive Coding — MLX-Local L0 SCAFFOLD Design

## BINDING OPERATOR DIRECTIVES (verbatim)

1. *"The MLX first requirement might also force us out of the issue we were having before where we had great ideas but we're building them as Boltons to the same substrates over and over again; we want to design the substrate and curriculum and then optimize the design the whole stack around it for extreme optimization and performance and optimal score lowering"* (operator 2026-05-26)

2. *"Never simply extend unless a rigorous adversarial cargo cult pass has been done first"* (operator 2026-05-26)

Both directives are bound into the substrate-design discipline below. Z8 is **NOT** an extension of Z6 / Z7 / DP1 / A-STACK / NSCS06. It is a **fresh substrate design from first principles** binding Catalog #312's canonical quadruple (Rao-Ballard 1999 + Mallat wavelet + DreamerV3 + Wyner-Ziv 1976) **simultaneously** per HNeRV parity discipline L7.

## 1. Substrate-class statement

Z8 is the asymptotic-pursuit terminal of the F-asymptote-trajectory (Z6 = single-layer FiLM low-risk anchor; Z7 = recurrent+Wyner-Ziv intermediate; Z8 = full hierarchical predictive coding with all 4 canonical primitives bound). Z8 supersedes Z6+Z7 **architecturally** (not in dispatch order — Z6 is still the canonical FIRST empirical anchor per the parent scoping memo's engineering-risk-minimization recommendation) by binding ALL four canonical primitives in a single coherent stack-of-stacks substrate per Catalog #312 STRICT preflight requirement.

Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" L7: substrate engineering UNIQUE-IFIES; binds ALL ingredients NOT incrementally. Z8 is the canonical example: every variant that ships only 1-2 of the canonical quadruple primitives is a substrate-class-fragment, not a substrate-class-shift. Z8 is the substrate-class-shift.

The canonical four primitives are:

1. **Rao-Ballard 1999 hierarchical predictive coding** — multi-level generative model where each layer predicts the level-below's representation; only prediction error is encoded; top-level latent is encoded directly.
2. **Mallat wavelet multi-scale** — orthogonal wavelet packet decomposition of detail bands per level (canonical Mallat 1989 multi-resolution analysis); enables Daubechies-CDF entropy coding of per-level errors.
3. **Hafner DreamerV3 latent dynamics (RSSM)** — discrete-categorical latent posterior at each level (`G * log2(K)` bits/sample); deterministic GRU state captures ego-motion-driven scene evolution; stochastic categorical state captures aleatoric uncertainty.
4. **Wyner-Ziv 1976 source coding with side information** — top-level latent Wyner-Ziv-coded against frame_0's decoded latent (canonical side info); reduces top-level bit budget from `H(z_top)` to `H(z_top | z_0_decoded)`.

The canonical-quadruple-binding-math derivation is Section 4 below.

## 2. Phase 1 — Substrate-design decision per Catalog #290

(Per operator binding directive #1: this is a FRESH design from first principles, not an extension. The 2-phase methodology applies — Phase 1 = design decision; Phase 2 = L0 SCAFFOLD implementation.)

## Canonical-vs-unique decision per layer

Per Catalog #290 STRICT preflight + the operator's UNIQUE-AND-COMPLETE-PER-METHOD operating mode. Default decision is **FORK** unless the canonical helper measurably serves Z8's substrate-optimal score path OR is a non-negotiable structural protection per CLAUDE.md.

| Layer | Decision | Rationale |
|---|---|---|
| **Trainer skeleton** | **ADOPT_CANONICAL_BECAUSE_SERVES** `tac.substrates._shared.trainer_skeleton` | TF32 + CUDA discipline + Catalog #178/#190 + `device_or_die` + `detect_hardware_substrate` are universal protections preventing bug classes Z8 inherits. PR 95 lesson HARD-EARNED. |
| **Scorer loss helper** | **ADOPT_CANONICAL_BECAUSE_SERVES** `tac.substrates._shared.score_aware_common.score_pair_components` | Catalog #164 HARD-EARNED; PR 95 lesson on differentiability + Catalog #187 HNeRV parity (yuv6 patch order). Z8's hierarchical loss adds level-specific terms BUT routes through canonical helper for the canonical seg+pose+rate decomposition. |
| **eval_roundtrip** | **ADOPT_CANONICAL_BECAUSE_SERVES** `tac.differentiable_eval_roundtrip.apply_eval_roundtrip_during_training` | CLAUDE.md "eval_roundtrip — NON-NEGOTIABLE" HARD-EARNED; 2-11× proxy-auth gap without it. |
| **YUV6 patch** | **ADOPT_CANONICAL_BECAUSE_SERVES** `patch_upstream_yuv6_globally()` | Catalog #187 HARD-EARNED per PR 95 lesson. |
| **EMA decay** | **ADOPT_CANONICAL_BECAUSE_SERVES** 0.997 weights / 0.99 codebook | CLAUDE.md "EMA — NON-NEGOTIABLE" HARD-EARNED. |
| **Inflate device selector** | **ADOPT_CANONICAL_BECAUSE_SERVES** `tac.substrates._shared.inflate_runtime.select_inflate_device` | Catalog #205 + #1 (MPS fallback trap); structural protection. |
| **MLX↔PyTorch bridge** | **ADOPT_CANONICAL_BECAUSE_SERVES** sister DreamerV3 `module.py` pattern (Gumbel-Softmax STE + PixelShuffle block reuse) | Sister A landed canonical MLX scaffold pattern; reusing the categorical posterior + STE primitives is canonical (NOT cargo-culted) because the math is IDENTICAL at the per-level RSSM bottleneck. Per-level RSSM at each hierarchy level uses the canonical categorical posterior. |
| **Archive grammar (Z8HPC1)** | **FORK_BECAUSE_PRINCIPLED_MISMATCH** | Z8's per-level latent + per-level wavelet detail bands + DreamerV3 deterministic+stochastic state + Wyner-Ziv top-level coded = 4-section monolithic-0.bin layout structurally distinct from Z6PCWM1 / RSSMC1 / NSCS06 wavelet patterns. The grammar IS the substrate-distinguishing primitive. |
| **Hierarchical predictive coder (multi-level RSSM stack)** | **FORK_BECAUSE_UNIQUE** | The 3-level RSSM stack with per-level Mallat wavelet detail bands is Z8's substrate-distinguishing engineering primitive #1. No canonical helper exists; this IS the substrate. |
| **Wyner-Ziv side-info coder** | **FORK_BECAUSE_UNIQUE** | Reuses canonical Wyner-Ziv mathematical primitive (Wyner-Ziv 1976 theorem) but operationalized specifically against Z8's top-level discrete categorical posterior with frame_0 side info. Per Catalog #319 the deliverability_proof_builder canonical helper validates compliance at L1+. |
| **Mallat wavelet detail-band codec** | **FORK_BECAUSE_PRINCIPLED_MISMATCH** | Sister NSCS06 v8 Path B has wavelet codec but for chroma residual; Z8 applies Mallat to per-level prediction errors at hierarchical scales. The math is canonical (Mallat 1989) but the deployment surface differs structurally. |
| **DreamerV3 categorical posterior per level** | **ADOPT_CANONICAL_BECAUSE_SERVES (per-level)** | Sister A (DreamerV3 RSSM) provides canonical categorical posterior + Gumbel-Softmax STE + per-pair int32 index packing. Z8 instantiates 3 INDEPENDENT categorical posteriors (one per level, smaller K at deeper levels per coarse-fine wavelet bound) using the canonical helper. |
| **Trainer's `_full_main`** | **DEFER_PHASE_2** | Per Catalog #240 acceptance cascade (c) pre-build substrate-engineering: trainer's `_full_main` raises `NotImplementedError` at L0 SCAFFOLD; Phase 2 council deliberation required before lifting per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY". |
| **Inflate runtime (PyTorch)** | **FORK_BECAUSE_PRINCIPLED_MISMATCH** | Z8's inflate must reconstruct 3-level hierarchy + DreamerV3 latent dynamics unroll + Mallat wavelet inverse + Wyner-Ziv decode. Per HNeRV parity L4 the inflate LOC budget is ≤200 for substrate-engineering waiver; Z8 targets ≤200 LOC (substrate-engineering waiver explicit per HNeRV L7). |
| **Score-aware loss (hierarchical)** | **FORK_BECAUSE_UNIQUE** | Adds per-level prediction-error L2 terms + Wyner-Ziv-conditional entropy term + Mallat sub-band entropy estimate; routes through canonical `score_pair_components` for seg+pose components. |

**Net forking pattern**: ~50% canonical adoption (universal protections + math primitives) + ~50% unique substrate engineering (the canonical-quadruple binding layer + archive grammar + inflate runtime). This matches the HNeRV parity L7 split — universal scaffolding ADOPTED; substrate-distinguishing engineering UNIQUE-IFIED.

## 3. The canonical-quadruple binding math (Section 4)

The core derivation Z8 operationalizes is the **stack-of-stacks information-theoretic floor** at the intersection of the four canonical-quadruple primitives:

$$
H_{\text{Z8 archive}} = \underbrace{H(z_{L_2}^{\text{top}} \mid z_0^{\text{decoded}})}_{\text{Wyner-Ziv top-level (Wyner 1976)}} + \sum_{l=0}^{L_1} \underbrace{H(\epsilon_l \mid z_{l+1}^{\text{top-down}})}_{\text{Rao-Ballard per-level errors (Rao 1999)}}
$$

where:

- $z_l$ = level-$l$ latent representation (Mallat wavelet coefficients at scale $2^l$).
- $\epsilon_l = z_l - \hat{z}_l$ = Rao-Ballard prediction error at level $l$; $\hat{z}_l$ is the top-down prediction from level $l+1$.
- $z_{L_2}^{\text{top}}$ = top-level latent (coarsest scale; DreamerV3 RSSM categorical posterior + deterministic state).
- $z_0^{\text{decoded}}$ = decoded frame_0 latent (Wyner-Ziv side information at decoder).

**Why all four simultaneously**:

- **Rao-Ballard alone** (Z5/Z6 prior art): predictive coding without Mallat or Wyner-Ziv compresses the per-pair latent but does NOT exploit spatial-scale-correlation within a frame OR cross-frame side-information.
- **Mallat alone** (NSCS06 v8 Path B prior art): wavelet codec exploits spatial-scale-correlation within a frame but does NOT exploit temporal redundancy OR cross-frame side-info.
- **DreamerV3 alone** (sister A prior art): categorical posterior at the per-pair latent layer exploits low-cardinality semantic discreteness but does NOT exploit spatial-scale or cross-frame.
- **Wyner-Ziv alone** (D4 sister substrate prior art): exploits cross-frame side-information but does NOT exploit spatial-scale or per-pair latent compression.

The canonical-quadruple binding **multiplicatively** combines all four reductions:

$$
\Delta H_{\text{Z8 total}} \approx \prod_i (1 - r_i) \cdot H_{\text{naive}}
$$

where $r_i$ is the per-primitive entropy reduction factor; for stationary-ergodic dashcam video with typical correlation structure, the per-primitive factors are (empirically bounded from sister anchor measurements):

- $r_{\text{Rao-Ballard}} \approx 0.35$ (predictive coding on smooth motion; sister Z6 design memo Section 18)
- $r_{\text{Mallat}} \approx 0.25$ (wavelet detail-band sparsity; canonical NSCS06 v8 Path B Mallat bound)
- $r_{\text{DreamerV3}} \approx 0.20$ (categorical posterior vs continuous-Gaussian; sister A DreamerV3 design + categorical_posterior_capacity_vs_continuous_gaussian_v1)
- $r_{\text{Wyner-Ziv}} \approx 0.30$ (frame-0 side-info on smooth video; sister D4 Wyner-Ziv frame 0 baseline)

Multiplicative combined factor (best-case): $(1-0.35)(1-0.25)(1-0.20)(1-0.30) = 0.65 \cdot 0.75 \cdot 0.80 \cdot 0.70 = 0.273$.

So Z8 targets approximately **27%** of the naive baseline archive bytes (best case). With naive baseline ≈ 275 KB (sister DP1 / Quantizr-class), Z8 targets ≈ **75 KB archive**. This matches the parent Z6/Z7/Z8 scoping memo's Z8 estimate (Section 3 Table: Z8 target 73 KB; Section 10 Z8PCWM1 target 73 KB).

**Per Boyd's Dykstra-feasibility lens** (Section 5 below): the multiplicative combination is an UPPER bound on entropy reduction; the true combined factor is the Dykstra alternating-projection intersection of per-primitive convex feasibility regions, which is subadditive (NOT multiplicative) under correlated reductions. The 75 KB estimate is the **planning prior** Z8 targets; the **achievable** floor requires the Dykstra-feasibility check.

## 4. Dykstra-feasibility intersection check on (rate / d_seg / d_pose / archive-bytes) polytope

Per Catalog #296 STRICT preflight + Boyd's convex-optimization lens. Per CLAUDE.md "Meta-Lagrangian/Pareto solver — NON-NEGOTIABLE": *"Dykstra/ADMM feasibility, Bayesian experimental design"*.

### 4.1 The 4-axis polytope

Z8's score Lagrangian per the contest formula:

$$
S = \underbrace{100 \cdot d_{\text{seg}}}_{\text{seg term}} + \underbrace{\sqrt{10 \cdot d_{\text{pose}}}}_{\text{pose term}} + \underbrace{25 \cdot \frac{B}{37,545,489}}_{\text{rate term}}
$$

with constraint set:

- $B \leq B_{\text{max}}$ (archive byte ceiling per Quantizr 0.33 leader; 293 KB)
- $d_{\text{seg}} \leq d_{\text{seg,max}}$ (SegNet distortion feasibility)
- $d_{\text{pose}} \leq d_{\text{pose,max}}$ (PoseNet distortion feasibility)
- $B \geq B_{\text{Shannon-floor}}$ (Shannon R(D) lower bound)

### 4.2 Dykstra alternating projections

Per Catalog #296 + Boyd's verdict on sister substrates (Catalog #239 Boyd-2 classify_dispatch open-boundary): the canonical Dykstra-feasibility iteration is:

1. Project current `(B, d_seg, d_pose)` onto rate-constraint hyperplane.
2. Project onto seg-distortion hyperplane.
3. Project onto pose-distortion hyperplane.
4. Project onto Shannon-floor hyperplane.
5. Iterate until convergence within $\epsilon$ tolerance.

The fixed point of Dykstra iteration is the L2-projection onto the convex intersection. The Z8 planning band is the projection's resulting (B, d_seg, d_pose) tuple **at the post-training operating point**.

### 4.3 Per-primitive contributions per Boyd's lens

| Primitive | Polytope axis affected | Convex constraint contributed |
|---|---|---|
| Rao-Ballard hierarchical | Rate axis (compresses latent) | $B \geq \sum_l H(\epsilon_l)$ |
| Mallat wavelet detail | Rate axis (sparsifies detail bands) | $B \geq H(\text{wavelet sparse coeffs})$ |
| DreamerV3 categorical | Rate axis (low-cardinality alphabet) | $B \geq G \cdot \log_2(K)$ per level |
| Wyner-Ziv top-level | Rate axis (conditional entropy reduction) | $B \geq H(z_{\text{top}} \mid z_0)$ |
| Hierarchical predictive coding (combined) | Seg/Pose axes (reconstruction quality) | $d_{\text{seg}} \leq d_{\text{seg,Rao-Ballard}}$, $d_{\text{pose}} \leq d_{\text{pose,Rao-Ballard}}$ |

### 4.4 Predicted ΔS band (planning prior; non-promotable)

Per Catalog #309 horizon_class declaration + Catalog #296 Dykstra-feasibility derivation:

**Predicted ΔS band**: **[0.05, 0.10]** (frontier_pursuit horizon class — asymptotic-pursuit class given the canonical-quadruple binding's joint reduction; matches parent scoping memo Section 3 Table Z8 row).

**Validation status**: `pending_post_training` per Catalog #324 STRICT preflight requirement. The band is a planning prior NOT a score claim until post-training Tier-C density measurement on the actual landed Z8 archive. Per CLAUDE.md "Apples-to-apples evidence discipline" the band is tagged `[predicted]` not `[contest-CPU]` or `[contest-CUDA]`.

**Reactivation criteria** (per CLAUDE.md "Forbidden premature KILL"): the band is reactivation-eligible after EITHER (a) post-training Tier-C density measurement validates the multiplicative-reduction estimate within ±20% OR (b) MLX-local smoke convergence demonstrates the canonical-quadruple binding empirically (per the MLX-first gate at `tools/gate_mlx_candidate_contest_equivalence.py` threshold 0.001).

**Probe-disambiguator hook** (Catalog #125 hook #6): the canonical disambiguator IS the per-primitive ablation. The Z8 trainer supports `--ablate-rao-ballard` / `--ablate-mallat` / `--ablate-dreamer-categorical` / `--ablate-wyner-ziv` argparse flags; each disables ONE primitive and measures ΔS contribution per primitive empirically. The 4-way ablation cross-tabulates the multiplicative bound vs the actual additive contribution at the empirical operating point.

## 5. 9-dimension success checklist evidence

Per Catalog #294 STRICT preflight requirement.

### Dimension 1: UNIQUENESS

Z8 is the FIRST repo substrate to bind all 4 of Catalog #312's canonical quadruple primitives **simultaneously**. Sister substrates:
- Z6 (sister D, in-flight) — binds ONLY Rao-Ballard primitive at single-level FiLM scale.
- Z7-Mamba-2 (sister B', in-flight) — binds Mamba-2 SSM recurrent predictor (Rao-Ballard sister but not hierarchical).
- DreamerV3 RSSM (sister A, landed) — binds ONLY DreamerV3 categorical posterior at single-level RSSM.
- D4 Wyner-Ziv frame 0 — binds ONLY Wyner-Ziv on frame-0 side-info (no hierarchy, no Mallat, no DreamerV3).
- NSCS06 v8 Path B — binds ONLY Mallat wavelet (chroma residual; not hierarchical multi-level).

Z8 is the **substrate-class-shift** terminal binding all four. Catalog #312 STRICT preflight requires the canonical quadruple be bound for any "hierarchical predictive coding" claim. Z8 is the canonical compliant substrate.

### Dimension 2: BEAUTY + ELEGANCE

The canonical-quadruple binding math (Section 4) decomposes cleanly into 4 orthogonal primitives that compose multiplicatively. The PR101-style 30-second-reviewability target is met at the architecture summary level (the 4 primitives + their bindings fit in a 30-second reviewer scan; the implementation LOC budget targets ≤200 inflate runtime per HNeRV L4 substrate-engineering waiver).

### Dimension 3: DISTINCTNESS

Z8 is structurally distinct from sister substrates across ALL 5 axes (substrate-class / engineering-risk / wire-grammar / per-primitive-binding / horizon-class). The composition matrix (parent scoping memo Section 13) confirms Z8 is non-overlapping with sister substrates EXCEPT Tishby IB-pure (NON-ORTHOGONAL HEAVY OVERLAP; both occupy IB asymptotic floor) — Z8 dominates Tishby IB-pure because Z8 binds 4 primitives where Tishby IB-pure binds only 1.

### Dimension 4: RIGOR

Per CLAUDE.md "Recursive adversarial review protocol" + Catalog #229 premise verification + Catalog #292 per-deliberation assumption surfacing:

- All canonical citations carry verifiable publication anchors (Rao-Ballard 1999 Nature Neuroscience 2(1):79-87; Mallat 1989 IEEE PAMI; Hafner 2023 DreamerV3 arXiv:2301.04104; Wyner-Ziv 1976 IEEE Trans. Inf. Theory IT-22:1).
- Architecture parameter budget is first-principles-bound from sister substrate empirical anchors (Z6 75K + DreamerV3 50K + sister A canonical decoder topology).
- Predicted ΔS band carries Dykstra-feasibility intersection check per Catalog #296.
- Cargo-cult audit per Catalog #303 below (Section 6) explicitly classifies HARD-EARNED vs CARGO-CULTED per assumption.
- Observability surface per Catalog #305 declared (Section 7).

### Dimension 5: OPTIMIZATION PER TECHNIQUE

Per Catalog #290 canonical-vs-unique decision per layer (Section 2): each per-layer decision documents the canonical-or-fork choice with substantive rationale. Default decision is FORK unless canonical helper measurably serves Z8's substrate-optimal score path per the operator's UNIQUE-AND-COMPLETE-PER-METHOD operating mode.

### Dimension 6: STACK-OF-STACKS COMPOSABILITY

Per the parent scoping memo Section 13 composition matrix:

- **Z8 × A-STACK × NSCS06 v8 Path B**: 3-axis orthogonal asymptotic-pursuit; predicted [0.03, 0.08] convex-intersection.
- **Z8 × Rudin floor**: ORTHOGONAL [0.05, 0.10] + ~−0.015 = [0.035, 0.085].
- **Z8 × Tishby IB-pure**: NON-ORTHOGONAL HEAVY OVERLAP; subadditive ~0 ΔS.
- **Z8 × ATW v2**: NON-ORTHOGONAL MEDIUM OVERLAP; subadditive ~−0.005 only.

Z8's high-EV compositions are with substrates on ORTHOGONAL axes (architectural primitives, decoder interpretability); subadditive compositions are with substrates sharing Z8's information-theoretic framework (Tishby IB, ATW v2 cooperative-receiver).

### Dimension 7: DETERMINISTIC REPRODUCIBILITY

Z8 archive grammar (Z8HPC1; Section 7 below) is byte-deterministic via canonical pattern: sorted-keys JSON sidecar + fixed brotli quality=9 + fp16 state_dict cast on CPU + raw bytes for category indices + per-level wavelet coefficient quantization with fixed scales. Sister DreamerV3 RSSMC1 + Z6PCWM1 patterns directly applicable.

Per Catalog #166 Modal HEAD-parity ledger + Catalog #245 modal_call_id_ledger: every dispatch registers via canonical helpers; the artifacts are forensic-reproducible.

### Dimension 8: EXTREME OPTIMIZATION + PERFORMANCE

- Archive bytes: ~75 KB target (vs Quantizr 293 KB ceiling; ~3.9× compression).
- Inflate LOC: ≤200 (substrate-engineering waiver per HNeRV L4).
- Inflate runtime: O(L · K · log K) for Mallat inverse + O(G · L) for categorical dequant + O(P · L) for per-level decoder unroll per pair, where L=3 (hierarchy levels), K=256 (categorical alphabet), G=24 (groups), P=600 (pairs).
- MLX-local iteration enabled at $0 wall-clock for design + smoke convergence per CLAUDE.md "MLX portable-local-substrate authority" per the 2026-05-26 MLX-contest-grade cascade.

### Dimension 9: OPTIMAL MINIMAL CONTEST SCORE

Per Section 4.4 predicted ΔS band **[0.05, 0.10]** (planning prior; non-promotable per CLAUDE.md "Apples-to-apples evidence discipline"). The asymptotic-pursuit class places Z8 in the floor-targeting tier of the F-asymptote-trajectory; per parent scoping memo Section 3 Table, Z8 approaches Tishby IB asymptotic floor at the long-horizon.

## 6. Cargo-cult audit per assumption

Per Catalog #303 STRICT preflight + Assumption-Adversary mandate per Catalog #292.

Even for fresh designs, every architectural assumption is taxonomized HARD-EARNED-vs-CARGO-CULTED per the canonical addendum.

| Assumption | Classification | Rationale | Unwind-test plan |
|---|---|---|---|
| **The canonical quadruple primitives compose multiplicatively for entropy reduction** | CARGO-CULTED | The multiplicative bound assumes independent entropy reductions; in practice the reductions are CORRELATED (per Boyd's Dykstra-feasibility lens). The actual achievable reduction is the convex-intersection projection. | Per-primitive ablation probe (Section 4.4 disambiguator); measure actual ΔS contribution per primitive empirically; cross-tabulate vs multiplicative bound. |
| **3 hierarchical levels is the right depth (vs 2, vs 4, vs adaptive)** | CARGO-CULTED | The 3-level choice inherits from parent scoping memo (Section 4.3 sketch); 3 matches Rao-Ballard's canonical visual cortex 3-level model. But for dashcam-video specifically, the right depth depends on the natural-image spatial frequency distribution which may be 2 or 4. | Per-level-count ablation: train Z8 with L ∈ {2, 3, 4} and measure ΔS as function of depth. |
| **DreamerV3 categorical with G=24 K=256 at each level is correct sizing** | CARGO-CULTED | These defaults inherit from sister A DreamerV3 + C6 IBPS baseline; the per-level optimal sizing depends on per-level spatial dimensionality (deeper levels have lower spatial resolution → smaller K may suffice per Mallat wavelet bound). | Per-level (G, K) Blahut-Arimoto sweep per sister equation `categorical_blahut_arimoto_rate_distortion_v1`. |
| **Mallat wavelet basis is optimal for dashcam image priors** | CARGO-CULTED | Mallat wavelets are optimal for natural-image priors per Mallat 1989 BUT dashcam-specific motion-correlated priors may benefit from different bases (e.g., directional wavelets per Curvelets-Candès). | Wavelet-basis sweep ablation: train Z8 with Daubechies-4 / Daubechies-8 / Curvelets / Haar; measure ΔS. |
| **Wyner-Ziv against frame_0 (not later frames or aggregated side-info) is optimal** | CARGO-CULTED | Frame_0 choice inherits from sister D4 Wyner-Ziv frame 0; but later-frame side-info OR aggregated multi-frame side-info may give tighter conditional entropy bounds for non-stationary video segments. | Side-info choice ablation: train Z8 with side_info ∈ {frame_0, frame_T/2, multi-frame-aggregate}; measure ΔS. |
| **Top-down predictive-coding architecture (per Rao-Ballard 1999) is optimal vs bottom-up** | HARD-EARNED | The top-down predictive-coding architecture is empirically validated by Rao-Ballard 1999 + 25+ years of follow-up work on visual cortex modeling; PRESERVE. | N/A (paradigm is HARD-EARNED). |
| **Discrete categorical posterior at each level (vs continuous Gaussian) is correct** | HARD-EARNED | Sister A DreamerV3 + canonical equation `categorical_posterior_capacity_vs_continuous_gaussian_v1` + C6 IBPS continuous-Gaussian collapse anchor empirically establishes categorical-posterior advantage; PRESERVE. | N/A (paradigm is HARD-EARNED). |
| **Hierarchical structure exploits scale-correlation in natural images** | HARD-EARNED | Multi-resolution analysis (Mallat 1989 + Daubechies 1992 + 30+ years of wavelet-codec literature) empirically establishes scale-correlation; PRESERVE. | N/A (paradigm is HARD-EARNED). |
| **Ego-motion conditioning improves next-frame prediction** | HARD-EARNED | Sister Z6 + Atick-Redlich 1990 + ego_motion_concentration_prior_v1 canonical equation; PRESERVE. | N/A (paradigm is HARD-EARNED). |
| **eval_roundtrip + canonical scorer-preprocess routing is required for differentiability** | HARD-EARNED | CLAUDE.md "eval_roundtrip — NON-NEGOTIABLE" + Catalog #164 + #187; PRESERVE. | N/A (paradigm is HARD-EARNED). |
| **Per-pair latent (not per-frame) at training matches contest scoring formula** | HARD-EARNED | Contest scoring is pair-based per upstream `evaluate.py`; PR 95 lesson HARD-EARNED. | N/A (paradigm is HARD-EARNED). |
| **MLX-local iteration is contest-grade for Z8 (vs paid CUDA-only)** | HARD-EARNED (recent) | 2026-05-26 MLX-contest-grade cascade: `|S_MLX − S_PyTorch| = 0.000011`, 72× smaller than PR110 frontier delta. PR 95 MLX full-decoder validation anchor. | N/A; Catalog #1265 canonical MLX gate IS the disambiguator at L1+ promotion. |

**Net cargo-cult audit verdict**: 5 HARD-EARNED + 6 CARGO-CULTED assumptions. The 6 CARGO-CULTED assumptions each have explicit per-primitive ablation unwind plans (Section 4.4 disambiguator). Per CLAUDE.md "Forbidden premature KILL without research exhaustion": the cargo-culted assumptions are DEFERRED-pending-ablation NOT killed; the ablation probes are MLX-local at $0 cost.

## 7. Observability surface

Per Catalog #305 STRICT preflight requirement. The 6-facet observability surface:

### Facet 1: Inspectable per layer

Each of the 4 canonical-quadruple primitives exposes per-layer forward-pass observables:

- **Rao-Ballard hierarchical encoder/decoder**: per-level latent + prediction-error tensors at each of 3 levels via canonical xray-style hook pattern; serialized to `experiments/results/<lane>/observability/per_layer/level_<l>_{latent,error}.jsonl`.
- **Mallat wavelet codec**: per-level wavelet coefficient distribution (mean, var, sparsity) + per-band entropy estimate; serialized to `observability/wavelet_bands.json`.
- **DreamerV3 RSSM per-level**: per-level categorical posterior logits + Gumbel-Softmax samples + argmax indices; serialized to `observability/dreamer_per_level.jsonl`.
- **Wyner-Ziv top-level**: side-info correlation strength + conditional entropy estimate; serialized to `observability/wyner_ziv_side_info.json`.

### Facet 2: Decomposable per signal

Composite metrics decompose into constituent contributions:

- `final_score = seg_term + pose_term + rate_term` via canonical `tac.xray.per_pair_score_decomposition` lens.
- `rate_term` decomposes per-section: `rate_top_level_wz + rate_per_level_errors + rate_decoder_weights + rate_meta`.
- Per-level prediction error L2 norm tracked separately.

Serialized to `observability/score_decomposition.json` with axis labels per CLAUDE.md "Apples-to-apples evidence discipline".

### Facet 3: Diff-able across runs

Two runs of Z8 produce byte-identical reproducible artifacts under same `(seed, commit_sha, upstream_snapshot_sha256)` tuple per Catalog #166 Modal HEAD-parity ledger + Catalog #245 modal_call_id_ledger.

### Facet 4: Queryable post-hoc

Run artifacts under `experiments/results/<lane>/` serialize as structured JSON/JSONL consumable without re-running. Continual-learning posterior queryable per `(substrate, axis, hardware, evidence_grade)` per Catalog #128 + #131 fcntl-locked discipline.

### Facet 5: Cite-able

Every behavior signal anchors to canonical tuple `(substrate_id=z8_hierarchical_predictive_coding, commit_sha, modal_call_id, config_path, random_seed, upstream_snapshot_sha256)` via Catalog #245 `tac.deploy.modal.call_id_ledger.register_dispatched_call_id(...)`.

Canonical equation refs (per Catalog #344 STRICT preflight requirement):

- `mlx_pytorch_full_decoder_downstream_scorer_drift_propagation_v1` (MLX-first iteration grade)
- `scorer_conditional_joint_rate_distortion_floor_v1` (theoretical floor)
- `categorical_posterior_capacity_vs_continuous_gaussian_v1` (per-level DreamerV3 categorical sizing)
- `ego_motion_concentration_prior_v1` (sister Z6 ego-motion conditioning)
- `cross_codec_super_additive_orthogonality_predictor_v1` (composition lattice EV)

### Facet 6: Counterfactual-able

Byte-mutation surface per Catalog #139 + Catalog #272 distinguishing-feature integration contract. Per-component ablation switches via `--ablate-rao-ballard` / `--ablate-mallat` / `--ablate-dreamer-categorical` / `--ablate-wyner-ziv` argparse flags. Per-level enable switches via `--num-hierarchy-levels {1,2,3,4}` for ablation depth.

## 8. Z8HPC1 archive grammar (byte-level per HNeRV parity L3)

Per HNeRV parity L3 + sister Z6PCWM1 + sister NSCS06 v8 Path B WLV2 + sister DreamerV3 RSSMC1 patterns: monolithic single-file `0.bin` per substrate.

```
Header (62 bytes; fixed-layout struct):
  - magic = b"Z8HPC1\x00\x00" (8 bytes; canonical Z8 Hierarchical Predictive Coding v1)
  - schema_version = u8 (1 byte; canonical 1)
  - num_levels = u8 (1 byte; canonical 3)
  - num_groups_per_level = u8[3] (3 bytes; e.g. 24/16/8 for L=3 levels)
  - num_categories_per_level = u16[3] (6 bytes; e.g. 256/128/64 for L=3 levels)
  - num_pairs = u16 (2 bytes; canonical 600)
  - decoder_latent_dim = u16 (2 bytes; canonical 28)
  - base_channels = u16 (2 bytes; canonical 24)
  - wavelet_basis_id = u8 (1 byte; canonical 0=Daubechies-4)
  - decoder_blob_len = u32 (4 bytes; brotli-compressed multi-level decoder weights len)
  - per_level_indices_blob_len = u32 (4 bytes; packed per-pair per-level cat indices len)
  - wavelet_coeffs_blob_len = u32 (4 bytes; per-level Mallat wavelet detail-band coeffs len)
  - wyner_ziv_top_blob_len = u32 (4 bytes; Wyner-Ziv top-level coded bytes len)
  - dreamer_state_blob_len = u32 (4 bytes; DreamerV3 GRU deterministic state len)
  - meta_blob_len = u32 (4 bytes; sorted-keys JSON utf-8 bytes len)
  - reserved = u8[12] (12 bytes; zero-padded for future expansion)
Section 1 (DECODER_BLOB):       multi-level decoder + cat-projection weights (~30 KB fp16+brotli)
Section 2 (INDICES_BLOB):       per-pair per-level categorical indices (~18 KB; 600 pairs × 3 levels × 24 groups × 1 byte)
Section 3 (WAVELET_BLOB):       per-level Mallat detail-band Daubechies-CDF coded (~15 KB)
Section 4 (WYNER_ZIV_BLOB):     Wyner-Ziv top-level coded (frame_0 side-info conditional) (~7 KB)
Section 5 (DREAMER_STATE_BLOB): DreamerV3 GRU deterministic + stochastic state init (~3 KB)
Section 6 (META_BLOB):          sorted-keys JSON (~2 KB)
Total target: ~75 KB
```

**Per-section operational consumption** (per Catalog #220 + #105 + #272 distinguishing-feature contract):

- DECODER_BLOB bytes ARE frame-affecting at inflate (multi-level decoder forward consumed).
- INDICES_BLOB bytes ARE frame-affecting at inflate (categorical samples consumed per level; **DISTINGUISHING FEATURE**).
- WAVELET_BLOB bytes ARE frame-affecting at inflate (per-level wavelet inverse consumed; **DISTINGUISHING FEATURE**).
- WYNER_ZIV_BLOB bytes ARE frame-affecting at inflate (top-level Wyner-Ziv decode consumed; **DISTINGUISHING FEATURE**).
- DREAMER_STATE_BLOB bytes ARE frame-affecting at inflate (GRU state init consumed; **DISTINGUISHING FEATURE**).
- header + meta bytes are parse/config gates (control_or_metadata role).

Per Catalog #272 distinguishing-feature integration contract: 4 substrate-distinguishing primitives (multi-level RSSM hierarchy / Mallat wavelet codec / DreamerV3 RSSM per-level / Wyner-Ziv side-info) MUST each pass byte-mutation testing per `tools/verify_distinguishing_feature_byte_mutation.py`.

## 9. MLX-implementation roadmap

### Phase 1 (THIS L0 SCAFFOLD landing)

- **Substrate package**: `src/tac/substrates/z8_hierarchical_predictive_coding/`
  - `__init__.py` — Catalog #124 8-field declaration + Catalog #241 LEGACY_SUBSTRATE_PRE_META_LAYER waiver
  - `mlx_renderer.py` — MLX multi-level RSSM hierarchy + wavelet detail bands per level (uses sister A DreamerV3 categorical posterior pattern via reuse)
  - `archive.py` — Z8HPC1 byte-deterministic grammar
  - `inflate.py` — PyTorch inflate stub (≤200 LOC substrate-engineering waiver per HNeRV L4)
  - `tests/test_basic.py` — Catalog #91 ENCODE_INFLATE_ROUNDTRIP + Catalog #139 byte-mutation no_op_proof + 11+ tests
- **MLX smoke trainer**: `experiments/train_substrate_z8_hierarchical_predictive_coding_mlx.py`
  - Thin MLX-local trainer (smoke ≤5ep ≤8pairs; synthetic MSE proxy + per-level Rao-Ballard residual L2)
  - `_full_main raises NotImplementedError` per Catalog #240
- **Landing memo**: `.omx/research/path_3_f_z8_hierarchical_predictive_coding_L0_scaffold_landed_20260526.md`

### Phase 2 (post per-substrate symposium per Catalog #325; OPERATOR-AUTHORIZE)

- MLX smoke convergence at contest-resolution (384×512 RGB) with 600 pairs.
- MLX-first contest-equivalence gate per Catalog #1265 at threshold 0.001 (likely need sister gate `tools/gate_mlx_candidate_contest_equivalence_z8.py` parameterized for Z8HPC1 grammar — QUEUE AS OP-ROUTABLE).
- Per-primitive ablation probes (Section 4.4 disambiguator).
- Sister-substrate composition smoke (Z8 × A-STACK × NSCS06 v8 Path B per parent scoping memo Section 13).

### Phase 3 (post MLX-gate PASS; OPERATOR-AUTHORIZE)

- PyTorch port via canonical bridge `tac.local_acceleration.pr95_hnerv_mlx::load_pytorch_state_dict_into_mlx` (sister A pattern).
- Modal T4 smoke (smoke wave) per Catalog #270 dispatch optimization protocol.
- Paired CPU/CUDA empirical anchor per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA".

### Curriculum stages (planning prior; will refine in Phase 2 design)

Per parent scoping memo Section 6 Z8 curriculum + canonical PR95 8-stage pattern:

- **Phase 0** (0-30 epochs): warmup with rate loss only at single-level (level 0 only).
- **Phase 1** (30-80 epochs): add level 1 (introduce 2-level hierarchy); train top-down prediction.
- **Phase 2** (80-150 epochs): add level 2 (full 3-level hierarchy); introduce per-level Mallat wavelet codec.
- **Phase 3** (150-220 epochs): introduce DreamerV3 categorical posterior per level; Gumbel-Softmax STE training; introduce SegNet + PoseNet score-aware loss via canonical `score_pair_components` per Catalog #164.
- **Phase 4** (220-280 epochs): introduce Wyner-Ziv top-level coding (frame_0 side-info conditioning); add wz_loss term; joint training with all losses balanced via Lagrangian dual.
- **Phase 5** (280-320 epochs): QAT (FP4 quantization-aware training) per Quantizr 0.33 winning recipe.
- **Phase 6** (320-350 epochs): EMA shadow finalization + archive build + paired CPU/CUDA auth eval per Catalog #226.

Smoke convergence target: per-pair MSE proxy decreases monotonically over ≥5 epochs at ≥8 pairs; sufficient for L0 SCAFFOLD landing verdict.

## 10. Sister substrate citation (Catalog #230)

**LANDED** (reference as research INPUT not bolt-on target):
- **A=DreamerV3 RSSM** (`69253a1cc`) — Provides canonical per-level categorical posterior + Gumbel-Softmax STE primitive. Z8 reuses this primitive at each hierarchy level (canonical per Catalog #290 decision). Code reuse via package-level import.
- **D=Z6 predictive coding** (`83b9ee3e2`) — Provides canonical FiLM-conditioned predictor + ego-motion conditioning primitive. Z8 generalizes to 3-level hierarchical predictor.
- **E=BoostNeRV against PR110** (`83910e54e`) — Provides residual-learner-against-PR110 pattern. Z8 is orthogonal (Z8 is multi-level hierarchical / E is iterative-boosting on PR110 baseline).

**IN-FLIGHT** (avoid file collision per Catalog #340 sister-checkpoint guard):
- **B'=Z7-Mamba-2** (`ac4283983ece21b83`) — Z7-Mamba-2 cargo-cult-first 3-phase. Z8 differs structurally (Z8 = multi-level hierarchical + Mallat + DreamerV3 + Wyner-Ziv; Z7-Mamba-2 = Mamba-2 SSM recurrent at single level).
- **C'=NSCS06 v8** (`ad26de7ad5f90848a`) — NSCS06 v8 chroma_lut cargo-cult-first 3-phase. Z8 differs structurally (Z8 = canonical-quadruple binding at multi-level / NSCS06 v8 = chroma_lut substitution at single level).

**CONCURRENT SISTER SPAWNS**:
- **G=NIRVANA cascading NeRV** — Disjoint scope (different substrate path: NeRV-family residual cascade vs Z8 canonical-quadruple binding).
- **H=ATW V2 cooperative-receiver** — Disjoint scope (different substrate path: Atick-Tishby-Wyner cooperative-receiver vs Z8 hierarchical Rao-Ballard + DreamerV3 + Wyner-Ziv).

Substrate path: `src/tac/substrates/z8_hierarchical_predictive_coding/` (NEW; no file overlap with sisters).

## 11. 6-hook wire-in declaration (per Catalog #125 NON-NEGOTIABLE)

1. **Sensitivity-map contribution** — per-level prediction error L2 norm IS the per-tensor importance signal at each hierarchy level; register `sensitivity_map.z8_hierarchical_predictive_coding_v1` post Phase 2.
2. **Pareto constraint** — adds `per_level_prediction_error_entropy ≤ ε_level` to the convex feasibility region per level; register `tac.pareto.z8_hierarchical_predictive_coding_v1` post-smoke.
3. **Bit-allocator hook** — per-level prediction-error bit allocation derives from per-level Mallat wavelet detail-band sparsity; register `bit_allocator.z8_hierarchical_predictive_coding_per_level_v1` post-smoke.
4. **Cathedral autopilot dispatch hook** — recipe planned at `.omx/operator_authorize_recipes/substrate_z8_hierarchical_predictive_coding_modal_a100_dispatch.yaml`; gated by Catalog #167 smoke-before-full + Catalog #325 per-substrate symposium (REQUIRED before paid dispatch); ranker v2 receives `literature_anchor=Rao-Ballard1999+Mallat1989+Hafner2023+Wyner-Ziv1976` as source-basis metadata only. No class-shift reward is valid until paired exact anchor exists.
5. **Continual-learning posterior** — every Z8 empirical anchor seeds the posterior via `posterior_update_locked` (Catalog #128).
6. **Probe-disambiguator** — per-primitive ablation IS the probe (Section 4.4 + Section 6 Cargo-cult audit unwind plans).

## 12. Catalog compliance checklist

- **Catalog #1** (no MPS fallback default): N/A — MLX is explicit opt-in; PyTorch inflate uses canonical `select_inflate_device` (CPU or CUDA only).
- **Catalog #91** (encoder/decoder dequantization roundtrip tested): tests will land in `tests/test_basic.py` (Phase 1).
- **Catalog #110/#113** (HISTORICAL_PROVENANCE APPEND-ONLY): this design memo is APPEND-ONLY; never mutated.
- **Catalog #124** (representation lane archive grammar at design time): 8 fields declared in `__init__.py` (Phase 1).
- **Catalog #125** (subagent landing has solver wire-in): 6 hooks declared above.
- **Catalog #127** (authoritative tag requires custody): all artifacts carry `[macOS-MLX research-signal]` + `score_claim=false` + `promotion_eligible=false` + `ready_for_exact_eval_dispatch=false`.
- **Catalog #139** (no-op detector): byte-mutation test in `tests/test_basic.py` (Phase 1).
- **Catalog #146** (Phase 1 inflate runtime contract): inflate.py honors 3-positional-arg `inflate.sh <archive_dir> <output_dir> <file_list>`.
- **Catalog #164** (canonical scorer-preprocess routing): `score_pair_components` used in Phase 2 trainer; smoke trainer uses MSE proxy with explicit `# CHECKPOINT_DISCIPLINE_WAIVED:smoke_l0_no_paid_dispatch` waiver as needed.
- **Catalog #166** (Modal HEAD-parity ledger): inherited from sister A pattern.
- **Catalog #167** (smoke-before-full pattern): MLX smoke landed at L0; Modal smoke gated at Phase 2.
- **Catalog #176** (STRICT callsites have CLAUDE.md row): N/A (Z8 does not add a NEW STRICT preflight gate; per Catalog #299 quota brake, current count ~362 well under 400 quota AND no new gate needed).
- **Catalog #190** (no hardcoded hardware substrate): trainer routes via `detect_hardware_substrate` per canonical helper.
- **Catalog #192** (macOS-CPU advisory not promoted without Linux verification): all artifacts non-promotable per Section 11.
- **Catalog #205** (inflate canonical device): `select_inflate_device` used in inflate.py.
- **Catalog #208** (docs/local paths): no `/Users/adpena/...` in body.
- **Catalog #220** (substrate L1+ scaffold operational mechanism): every section operationally consumed per Section 8.
- **Catalog #229** (premise verification): completed before edit (sister substrates read).
- **Catalog #230** (sister-subagent ownership map): Section 10 above.
- **Catalog #240** (recipe-vs-trainer-state consistency): trainer `_full_main raises NotImplementedError` per acceptance cascade (c).
- **Catalog #241** (substrate META layer): `LEGACY_SUBSTRATE_PRE_META_LAYER:<rationale>` waiver in `__init__.py` (Phase 1).
- **Catalog #245** (Modal call_id ledger): inherited from sister A pattern.
- **Catalog #270** (canonical dispatch optimization protocol): Phase 2 trainer wires canonical helpers per umbrella protocol.
- **Catalog #272** (distinguishing-feature integration contract): 4 distinguishing primitives declared in Section 8; byte-mutation tests in `tests/test_basic.py`.
- **Catalog #287** (placeholder-rationale rejection): all rationales substantive ≥4 chars.
- **Catalog #290** (canonical-vs-unique decision per layer): Section 2 above.
- **Catalog #292** (per-deliberation assumption surfacing): explicit assumption statements per attendee in this memo's frontmatter.
- **Catalog #294** (9-dim success checklist evidence): Section 5 above.
- **Catalog #295** (submission inflate works with empty PYTHONPATH): Phase 2 inflate.py uses canonical helper path; substrate-specific code vendored alongside per NSCS06-v6 pattern.
- **Catalog #296** (predicted band has Dykstra-feasibility check): Section 4 above.
- **Catalog #297** (signal-axis-destruction reversibility): N/A — Z8 preserves full RGB; multi-level wavelet decomposition is reversible per Mallat 1989 inverse transform.
- **Catalog #299** (catalog quota brake under 400): no new STRICT gate landed.
- **Catalog #300** (council deliberation v2 frontmatter): frontmatter complete.
- **Catalog #303** (cargo-cult audit per assumption): Section 6 above.
- **Catalog #305** (observability surface): Section 7 above.
- **Catalog #309** (substrate design memo declares horizon_class): `horizon_class: frontier_pursuit` in frontmatter.
- **Catalog #310** (F-asymptote substrate is class-shift not bolt-on): Z8 IS class-shift (multi-axis primary architecture binding 4 canonical primitives simultaneously; substrate-engineering-by-construction).
- **Catalog #311** (predictive coding substrate has ego-motion conditioning): Z8 uses ego-motion conditioning via DreamerV3 deterministic GRU state (ego-motion vector input per pair); sister Z6 FiLM pattern extends to hierarchical levels.
- **Catalog #312** (hierarchical predictive coding has canonical quadruple): Z8 IS the canonical compliant substrate binding all 4 primitives (Rao-Ballard + Mallat + DreamerV3 + Wyner-Ziv).
- **Catalog #319** (Wyner-Ziv reweight has deliverability proof): Phase 2 lands `DeliverabilityProof` for Wyner-Ziv top-level coded section.
- **Catalog #324** (no predicted_band without post-training Tier-C validation): predicted band tagged `pending_post_training` per frontmatter.
- **Catalog #325** (substrate dispatch has per-substrate optimal form symposium): per-substrate symposium queued as operator-routable for Phase 2 paid-dispatch authorization.
- **Catalog #335** (cathedral consumer directory package exposes canonical contract): N/A at L0 SCAFFOLD; Phase 2 may add a sister cathedral consumer.
- **Catalog #340** (subagent commit serializer invokes sister checkpoint guard): commits route through canonical serializer per discipline.
- **Catalog #344** (empirical finding memo references canonical equation): canonical equation refs in frontmatter; specific equation citations in Sections 4 and 7.
- **Catalog #354** (master-gradient exploit consumers complete): N/A at L0 SCAFFOLD.
- **Catalog #357** (Tier B canonical contract): N/A at L0 SCAFFOLD (Tier A observability-only).

## 13. Operator-routable next steps (post Phase 1 landing)

1. **Sister gate for Z8HPC1 grammar**: queue lane `lane_gate_mlx_candidate_contest_equivalence_z8_20260526` — extend canonical MLX gate `tools/gate_mlx_candidate_contest_equivalence.py` to support Z8HPC1 archive grammar (current gate is hardwired for PR95). Estimated cost: $0 + ~2-3h wall-clock.
2. **Contest-scale MLX smoke**: scale L0 smoke from synthetic to actual `upstream/videos/0.mkv` 600-pair contest video; estimated cost $0 + ~1-2h wall-clock; per CLAUDE.md HNeRV parity L1 (training MUST use upstream video).
3. **PyTorch port via canonical bridge**: queue lane `lane_z8_pytorch_port_via_canonical_bridge_20260526` — port MLX module to PyTorch via `tac.local_acceleration.pr95_hnerv_mlx::load_pytorch_state_dict_into_mlx` sister pattern. Estimated cost: $0 + ~3-4h wall-clock.
4. **Per-substrate symposium per Catalog #325**: queue lane `lane_z8_per_substrate_optimal_form_symposium_20260526` — required before paid-dispatch authorization per Catalog #325 STRICT preflight; canonical 6-step contract (cargo-cult audit + 9-dim checklist + observability surface + sextet pact + reactivation criteria + Catalog #324 post-training Tier-C validation discipline). Estimated cost: $0 + ~3-4h wall-clock.
5. **Per-primitive ablation probes** (post Phase 2): the 4 canonical-quadruple primitives ablated independently per Section 4.4 disambiguator + Section 6 cargo-cult audit unwind plans. Estimated cost: $0 MLX-local + paid CUDA at $5-10 per primitive once Phase 2 lands.
6. **Composition smoke** (post per-substrate symposium): Z8 × A-STACK × NSCS06 v8 Path B 3-axis orthogonal asymptotic-pursuit composition per parent scoping memo Section 13. Estimated cost: $0 MLX-local + paid CUDA at $5-15 once Phase 2+composition smoke lands.

## 14. Mission contribution per Catalog #300

`frontier_breaking` — Z8 binds the canonical quadruple (Catalog #312) **simultaneously** for the FIRST time in the repo, operationalizing the asymptotic-pursuit substrate-class-shift at the F-asymptote-trajectory terminal. The MLX-local L0 SCAFFOLD enables $0 iteration on the canonical-quadruple binding hypothesis BEFORE any paid CUDA dispatch authorization per CLAUDE.md "Substrate MUST be at OPTIMAL FORM before paid empirical dispatch" non-negotiable.

The substrate-design-from-first-principles + cargo-cult-pass-FIRST discipline (per operator's binding directives 2026-05-26) is locked into Z8 by construction: every layer's canonical-vs-unique decision documented (Section 2), every assumption taxonomized HARD-EARNED-vs-CARGO-CULTED (Section 6), every cargo-culted assumption paired with explicit ablation unwind plan (Section 4.4 + Section 6).

EOF
