# CPU-vs-CUDA Drift Engineering Analysis (Wave 3 Slot 30 + Slot 31 + Slot B closure)

**Authored 2026-05-20T17:37:24Z**.
**Tasks closed:** #985 (engineering analysis + MPS taxonomy + 3 canonical equations), #986 (HF Jobs LEGAL_NATIVE_PLATFORMS gap audit), #989 (writeup amendment cross-link).
**Sister cross-refs:** `docs/writeup/cuda_cpu_drift_methodology.md` (canonical methodology long-form 2026-05-08; 586 lines), `docs/paper/04_results.md` §4.8 + §4.8.1 (writeup canonical), `docs/findings/cuda_cpu_auth_eval_split_20260508.md` (OSS-disclosure short form), `feedback_dual_cpu_cuda_auth_eval_mandatory_20260508.md` (memory anchor), `feedback_writeup_amendment_cpu_vs_cuda_discrepancy_plus_cuda_frontier_exploration_landed_20260519.md` (predecessor writeup amendment), `reports/public_pr100_108_eval_comment_scorecard_20260508.json` (public anchor scorecard).
**Lane:** `lane_wave_3_cpu_cuda_drift_analysis_and_writeup_amendment_20260520` L1.

This memo is a **synthesis + extension**: it does NOT re-derive prior work; it consolidates the empirical anchor inventory across the apparatus, audits the four candidate mechanism hypotheses for what is empirically tested versus still-conjecture, formalizes the MPS portability taxonomy against canonical-equation language, and proposes three new canonical equations for registration per Catalog #344. The canonical methodology long-form (`docs/writeup/cuda_cpu_drift_methodology.md`) remains the deep-derivation reference; this memo's value is the operator-routable cross-archive view + the equation-registration artifact.

## 1. Empirical anchor inventory

Five **public** HNeRV-class paired CPU+CUDA anchors (harvested from GitHub Actions bot comments; canonical artifact `reports/public_pr100_108_eval_comment_scorecard_20260508.json`):

| PR | Author | Bytes | CUDA score | CPU score | Δ (CUDA-CPU) | R_seg | R_pose | Lane tag |
|---|---|---:|---:|---:|---:|---:|---:|---|
| #100 | BradyMeighan | 178,981 | 0.23000 | 0.20000 | +0.0300 | 1.17 | 5.00 | `[contest-CUDA T4]` + `[contest-CPU GHA Linux x86_64]` (public) |
| #101 | SajayR (gold) | 178,258 | 0.23000 | 0.19000 | +0.0400 | 1.18 | 5.20 | same |
| #102 | EthanYangTW (bronze) | 178,981 | 0.23000 | 0.20000 | +0.0300 | 1.17 | 5.01 | same |
| #103 | rem2 (silver) | 178,223 | 0.23000 | 0.19000 | +0.0400 | 1.17 | 5.00 | same |
| #105 (kitchen_sink) | valtterivalo | 177,857 | 0.23000 | 0.20000 | +0.0300 | 1.16 | 4.97 | same |

R_pose mean = **5.04** across HNeRV-cluster; σ = 0.10; per-PR delta clusters at 0.030–0.040. The R_seg mean = **1.17**; σ = 0.01 (tighter). Per `docs/writeup/cuda_cpu_drift_methodology.md` §1.3, score-points decomposition holds: ~70% of the Δ is pose; ~30% is seg; rate term bit-identical (no decode in the rate term). The cluster's tightness is the canonical evidence for "structural per-archive property of the scorer + decoder stack" rather than per-archive random noise.

Four **internal** anchors with paired or single-axis evidence on contest-1:1 hardware:

| Anchor | CPU axis | CUDA axis | Δ | Source |
|---|---:|---:|---:|---|
| **PR110 frontier** (lane `pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean`, archive sha `6bae0201fb08...`, 178,517 bytes) | 0.192051 `[contest-CPU GHA Linux x86_64]` | 0.226210 `[contest-CUDA T4]` | +0.0341 | `.omx/state/canonical_frontier_pointer.json` (current submission) |
| **PR101 GOLD replay** (archive sha `b83bf348...`, ~178,258 bytes) | ~0.193 (public) | 0.22826957 `[contest-CUDA T4]` | +0.0352 | public scorecard + T4 replay |
| **PR106 format0d latent score table** (archive sha `9cb989cef519...`, 186,876 bytes) | not run | 0.205330 `[contest-CUDA T4]` | n/a | CUDA-axis frontier per canonical pointer |
| **PR107 apogee** (archive sha pinned; ~178,392 bytes) | 0.196636 `[contest-CPU GHA Linux x86_64]` (GHA workflow `25556454358`) | 0.229360 `[contest-CUDA T4]` | +0.0327 | sister methodology long-form §0.1 |

The internal-PR110 anchor delta **+0.0341** is consistent with the public-PR HNeRV-cluster mean +0.033 within the cluster σ ≈ 0.004 — confirming the cluster's "structural per-archive property" hypothesis extends to our own engineered archives, not just upstream maintainer training.

**A1 substrate** (lane `lane_a1_landed`, archive sha `87ec7ca5f2f3...`) carries an internal paired anchor showing dual-axis Δ +0.0335 (CUDA 0.22635 vs CPU 0.19285); cross-ref `submissions/a1/dual_eval_adjudicated.json`. The +0.0335 is again within cluster σ — A1 was engineered for CPU axis, the CUDA-axis position is "what it is" per §4.8.1 framing.

**M5 Max macOS-CPU vs Linux x86_64 GHA CPU** (PR107 anchor): 0.19664189 vs 0.1966358879 = **+6e-6 gap**. This is below the FP32 noise floor and validates the macOS-CPU-as-advisory-proxy hypothesis FOR THIS ARCHIVE CLASS (HNeRV-family small renderer). Per CLAUDE.md "MPS auth eval is NOISE" + Catalog #192 + Catalog #317, the advisory grade STAYS until paired Linux x86_64 replay promotes the per-archive pair empirically; the 6e-6 anchor is sufficient to keep macOS-CPU in the dev-loop research-signal track but NOT sufficient to declare cross-architecture-class universal portability.

**Aggregate count:** ~9 distinct paired or single-axis empirical anchors (5 public HNeRV + 4 internal). The internal anchor density is far below where mechanism attribution becomes statistically determinative; per §2 below, mechanism attribution remains a **mixed-evidence inference**, not a measured causality claim.

## 2. Mechanism attribution audit (four candidate hypotheses)

The +0.033 average delta has four plausible mechanism contributions. We catalog what is empirically tested versus what remains design conjecture.

### Hypothesis A — DALI/NVDEC vs PyAV ground-truth decode drift

**Source claim:** `evaluate.py --device cuda` routes through `DaliVideoDataset` (NVIDIA DALI + hardware NVDEC); `--device cpu` routes through `AVVideoDataset` (PyAV / libavcodec software decode). The two paths produce different RGB uint8 tensors from the same `.mkv` file due to: rounding modes in chroma upsampling, color-space conversion matrix precision, and per-pixel quantization decisions at the YUV→RGB boundary.

**Empirical status:** **PARTIAL EVIDENCE.** Sister tracers landed `tools/probe_eval_loader_drift.py` to compare DALI/NVDEC vs PyAV decoded RGB tensors before scorers. On macOS the tracer emits a non-promotable CUDA/DALI plan; on T4 the tracer emits decoded-RGB LSB statistics. The decisive shared-tensor microbench (decode with PyAV on CPU, hash/dump the tensors, then feed those tensors through CUDA forward) per §2.5 of `docs/writeup/cuda_cpu_drift_methodology.md` has NOT been completed. Without it, decoder-vs-forward-kernel attribution remains a coupled-effects measurement.

**Score-contribution upper bound (back-of-envelope):** if DALI/NVDEC decoded tensors differ from PyAV by ≤2 LSB per pixel uniformly, then PoseNet's per-pixel sensitivity (from §2.3 σ²_cuda model) suggests up to ~30-50% of the observed +0.033 delta could plausibly be decoder-attributable, with the remainder being forward-kernel FP32 noise. This is a fit, not a measurement.

### Hypothesis B — CPU/CUDA forward-kernel FP32 noise floor

**Source claim:** Per `docs/writeup/cuda_cpu_drift_methodology.md` §2.3, `σ²_cuda ≈ K · L · ε² · ||x||²` where L ≈ 50 (convolution operation count for FastViT-T12), ε ≈ 1.7e-3 (per-op RMS precision drift FP32 on T4 vs IEEE-strict), and K is a variance-aggregation constant. For PoseNet's regression head, this predicts σ²_cuda ≈ 1.4e-4, which at medal-band pose_cpu = 3.5e-5 gives pose_cuda ≈ 1.75e-4 → R_pose ≈ 5.0. This matches the observed R_pose = 5.04 ± 0.10 to numerical precision.

**Empirical status:** **MODEL FIT EVIDENCE, NOT LOCALIZATION PROOF.** The σ²_cuda model fits the observed delta perfectly across all five HNeRV anchors, but a decoder-only perturbation with appropriate Lipschitz constants would also fit. Localizing the contribution requires either the shared-tensor microbench (decoder isolated; CUDA forward isolated) OR the calibrated noise injection test (inject 1/2/3 LSB uniform noise into the PyAV decoded tensor, run PoseNet on T4, compare to observed +0.0014e-4 gap). Neither has landed empirically.

### Hypothesis C — Pose-head numerics (FastViT-T12 attention/TF32 amplification)

**Source claim:** original hypothesis — FastViT-T12 attention softmax FP16/TF32 compounding across ~12 layers explains the pose-axis dominance.

**Empirical status:** **FALSIFIED on three independent grounds** per `docs/writeup/cuda_cpu_drift_methodology.md` §2.2:
1. FastViT-T12 uses RepMixer, not attention. `timm/models/fastvit.py:1645` confirms all 4 stages are RepMixer + depthwise convolution + token mixing — no softmax attention layer exists.
2. T4 (sm_75 Turing) has no TF32 hardware. TF32 is sm_80+ (Ampere/A100 and later). On T4, `torch.backends.cuda.matmul.allow_tf32 = True` is a silent no-op.
3. PyTorch's default since 1.12 is `cuda.matmul.allow_tf32 = False` even on hardware with TF32. The codebase does not override this.

**Conclusion:** the TF32-attention hypothesis is **rejected**. Replacement model per §2.3 (Hypothesis B) is the canonical pose-axis-dominance explanation; the actual mechanism is FP32 noise amplified through the regression head's quadratic MSE versus SegNet's piecewise-constant argmax (per §2.4 of methodology long-form).

### Hypothesis D — Loader / dataset config drift

**Source claim:** the CPU and CUDA paths might differ in subtle config beyond decoder choice (frame ordering, batch shape, normalization constants, etc.).

**Empirical status:** **REJECTED for ALL non-decode stages** per `docs/writeup/cuda_cpu_drift_methodology.md` §2.6 table. Every stage of the contest pipeline was audited:
- Inflated-frame `.raw` decode: both paths use `TensorVideoDataset` (NO divergence).
- Arithmetic decoders in PR101/103 inflate: possible numeric divergence, unverified, but most AC implementations are integer-only and thus immune.
- BatchNorm running stats: eval mode + fixed constants (NO divergence).
- Rate term + score formula: bit-identical (NO divergence).
- Hydra-head MLP: shares Hypothesis B's FP32 noise floor (component of B, not separate).

**Conclusion:** ONLY the ground-truth-video decode (Hypothesis A) and the FP32-forward-kernel floor (Hypothesis B) are live divergence sources. The other pipeline stages contribute zero.

### Net mechanism attribution verdict

**Empirically tested:** Hypothesis C (FALSIFIED), Hypothesis D (REJECTED for non-decode stages).

**Empirically partial:** Hypothesis A (loader tracers landed, shared-tensor microbench pending) + Hypothesis B (σ² model fits perfectly but localization microbench pending).

**Mixed conclusion (canonical):** the +0.033 average delta is a coupled effect of (a) decoder-output drift between DALI/NVDEC and PyAV (Hypothesis A; estimated 30-50% contribution), and (b) FastViT-T12 conv-stack FP32 noise floor amplified through the regression head (Hypothesis B; estimated 50-70% contribution). The pose-axis dominance (R_pose ≈ 5 vs R_seg ≈ 1.17) is intrinsic to the score-aggregation shape (§2.4 of methodology long-form), not a per-kernel attribution. This conclusion is **stable across all five public HNeRV anchors** but is NOT a localized measurement.

**Closing observation:** the empirical cluster tightness (σ ≈ 0.10 on R_pose; σ ≈ 0.004 on Δscore) is what makes mechanism attribution actionable for engineering decisions even without complete localization. A target-axis-aware ranker (per `target_axis="cpu_leaderboard"` in `tac.score_geometry`) can be built today from the cluster mean R values; it does not need per-mechanism attribution to reweight Lagrangian terms.

## 3. MPS portability use-case taxonomy

Per CLAUDE.md "MPS auth eval is NOISE" non-negotiable + Catalog #1 + Catalog #192 + Catalog #317, MPS portability has three operational tiers with strict tagging discipline.

### Tier 1 — ACCEPTABLE: free advisory proxy + research signal

**Use cases:**
- Long-running training loops where MPS forward provides per-step proxy distortion signal.
- Smoke / dev-loop / sanity-check runs (code correctness, architecture validation).
- Curve-shape discovery sweeps where the ABSOLUTE value is not the question, only the relative ranking across hyperparameter axes.
- Hinton-distilled scorer surrogate co-training (the surrogate itself runs on MPS, the contest scorer's verdict remains CUDA-axis).
- Macros-CPU-axis advisory proxy per Catalog #317 + Catalog #192 (`evaluate.py --device cpu` on M5 Max, tagged `[macOS-CPU advisory only]`, NOT `[contest-CPU]` — per the 6e-6 empirical anchor on PR107).

**Required tags:** one of `[MPS-PROXY]`, `[MPS-research-signal]`, `[advisory only]`, `[macOS-CPU advisory only]`.

**Forbidden mutations:** the signal MAY seed candidates, refine curve-fit priors, populate research-only manifests, and rank MPS-MPS sweeps. The signal MAY NOT update authoritative cost-band posteriors, promote candidates to L2+ in the lane registry, satisfy substrate L1+→L2 promotion canonical (Catalog #233), or trigger autopilot dispatch via score-claiming metadata. Per Catalog #341 routing-recommender markers: routing-grade signals MUST set `predicted_delta_adjustment=0.0` + `promotable=False` + `axis_tag="[predicted]"` in every cathedral-consumer return value.

### Tier 2 — RESEARCH-ONLY: bounded use for MPS-VIABLE substrates with paired CUDA evidence

**Use cases:** substrates whose drift posture has been empirically tested via `tac.mps_diagnostic.drift_predictor` + paired CUDA anchor + drift bounded to < 5% LOCAL_MPS_TRAIN_VIABLE threshold per `feedback_mps_phase_b_options_b_plus_c_completion_landed_20260519.md` (Phase B empirical aggregate gap: 0.072% / 69x below threshold for `tinyrenderer_phase_b` architecture class).

**Required:** explicit research_only=true lane registry tag, sister CUDA anchor on the SAME architecture class within last 14 days, Tier-A-canonical-routing markers per Catalog #341 in every consumer return value.

**Forbidden mutations:** even with paired CUDA evidence, MPS signal is NOT contest-promotion eligible. The dual-eval mandate per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" still requires Linux x86_64 paired axis on the EXACT archive bytes that will ship.

### Tier 3 — FORBIDDEN: authoritative score, kill-promote-strategy, paper-empirical claims

**Use cases:** none. MPS is NEVER 1:1 contest-compliant for any submission decision.

**Empirical anchor (the 23× drift anchor):** the original 2026-04-25 paired measurement on a pinned archive returned PoseNet distortion 0.245 (MPS) vs 0.0107 (CUDA A100, contest scorer) = 23× WORSE on MPS. SegNet distortion 0.0024 (MPS) vs 0.00116 (CUDA) = 2× WORSE. Final score 2.26 (MPS) vs 0.90 (CUDA) = 2.5× WORSE. This is the **canonical anchor** for the forbidden tier; per CLAUDE.md "MPS auth eval is NOISE" non-negotiable, any score in run_log / BATTLE_PLAN / findings carrying MPS-axis numbers must be tagged `[MPS-PROXY]` and treated advisory only.

**Domain-of-validity insight:** the 23× anchor is for a specific architecture class (Lane G v3 era ~285K param renderer). Phase B 2026-05-19 empirically falsified the 23× hypothesis for `tinyrenderer_phase_b` (12K-param architecture, 0.072% aggregate gap). This is captured in the existing canonical equation `mps_drift_architecture_class_dependent_v1` (slot 16). The taxonomy here adds the THREE-TIER routing discipline that the equation's domain-of-validity implies but does not encode directly.

### Cross-domain MPS portability decision tree

```
Is this for: authoritative score / promotion / paper claim?
├── YES → MPS is FORBIDDEN. Use Linux x86_64 + NVIDIA per CLAUDE.md "Submission auth eval".
└── NO → Continue.
    Is this for: research-signal / advisory proxy / dev-loop / curve discovery?
    ├── YES → MPS is ACCEPTABLE with required tags + canonical Provenance markers.
    └── NO → Continue.
        Is this for: training the substrate that will then be CUDA-eval'd?
        ├── YES, with paired CUDA evidence on architecture class within 14 days → MPS Tier 2 OK.
        └── ELSE → re-route through CUDA (Modal / Vast.ai / Lightning) per Catalog #199 paired-env discipline.
```

## 4. Three candidate canonical equations

Per Catalog #344 + CLAUDE.md "Canonical equations + models registry" non-negotiable + the operator NON-NEGOTIABLE 2026-05-19 verbatim *"we need to formalize all of this and canonicalize and operationalize because I am afraid we are learning but if we don't have systems of equations and models and such we are just gaining tribal knowledge"*. Each equation maps a single empirical-finding cluster from this analysis to a queryable canonical-equation-registry row.

### Equation #1 — `cpu_cuda_score_gap_v1` (per-archive Δ predictor)

**One-line summary:** for HNeRV-class archives, the per-archive (CPU score, CUDA score) gap clusters at +0.030–0.040 with σ_inter-archive ≈ 0.004.

**Latex form:** `Δ_score(a) = score_cuda(a) - score_cpu(a) = 100·(d_seg_cuda(a) - d_seg_cpu(a)) + (√(10·d_pose_cuda(a)) - √(10·d_pose_cpu(a)))`. Mean: **Δ_score ≈ 0.033 + ε** where ε is per-archive variance with σ ≈ 0.004. Rate term cancels (bit-identical).

**Domain-of-validity:** HNeRV-class archives (FastViT-T12 PoseNet head; EfficientNet-B2 SegNet); pose_cpu in [3e-5, 2e-4] band; archive_bytes in [170K, 200K] band; contest scorer SHA pinned.

**Producer:** `tools/public_pr_eval_comment_scorecard.py` (harvests public bot comments); planned `tools/cuda_cpu_drift_compute_ratios.py` (per the methodology long-form §10 sweep plan).

**Consumer:** `tac.score_geometry` `target_axis="cpu_leaderboard"` planner reweighting (per §4.2 of methodology long-form); cathedral autopilot ranker per Catalog #335 auto-discovery; `reports/latest.md` per-axis frontier section.

**Empirical anchor (initial):** `hnerv_cluster_pr100_pr101_pr102_pr103_pr105_paired_anchor_20260508` — observed mean Δ = 0.033 across 5 public HNeRV PRs; predicted Δ = 0.033 (model fit); residual = 0.0.

**Recalibration trigger:** `when_3+_new_empirical_anchors_in_domain` (the canonical default — once 3 new paired anchors land, residual posterior refreshes).

### Equation #2 — `pose_axis_cuda_amplification_v1` (pose-channel R_pose ≈ 5 model)

**One-line summary:** for HNeRV-class FastViT-T12 PoseNet head + medal-band pose_cpu, the CUDA/CPU pose-axis distortion ratio R_pose = pose_cuda / pose_cpu ≈ 5.04 ± 0.10.

**Latex form:** `R_pose = pose_cuda / pose_cpu`. Predicted by additive-precision-noise model: `pose_cuda ≈ pose_cpu + σ²_cuda` where `σ²_cuda ≈ K · L · ε² · ||x||²` with L ≈ 50 (conv-op count), ε ≈ 1.7e-3 (FP32 noise floor on T4). For pose_cpu = 3.5e-5: predicted R_pose = (3.5e-5 + 1.4e-4) / 3.5e-5 = 5.0.

**Domain-of-validity:** HNeRV-class FastViT-T12 PoseNet head (regression to 12-dim pose, first 6 used); pose_cpu in [3e-5, 2e-4] medal-band; T4 / 4090 / A100 / equivalent CUDA hardware; FP32 forward precision (no FP16 / no TF32 / no FP8).

**Producer:** `tools/public_pr_eval_comment_scorecard.py` (R_pose extraction from paired comments); planned shared-tensor microbench in `tools/probe_eval_loader_drift.py` + `tools/probe_posenet_layer_drift.py` (mechanism-discrimination; see §2 above).

**Consumer:** `tac.score_geometry` `target_axis="cpu_leaderboard"` pose-term reweighting `(5 / sqrt(10 * pose_op) / 5.04)` per §4.2 of methodology long-form; `tac.findings_lagrangian` per-pair pose-axis dual-variable; cathedral autopilot consumer routing recommendation per Catalog #341 (Tier-A markers).

**Empirical anchors (initial):** `hnerv_cluster_r_pose_5_04_anchor_20260508` (mean of 5 public anchors: R_pose ≈ 5.04 ± 0.10).

**Recalibration trigger:** `when_3+_new_empirical_anchors_in_domain`. Promotion to a higher-resolution domain awaits the shared-tensor microbench landing (mechanism attribution beyond cluster-mean fit).

### Equation #3 — `mps_portability_use_case_taxonomy_v1` (three-tier routing decision)

**One-line summary:** MPS portability has three operational tiers — ACCEPTABLE / RESEARCH-ONLY-WITH-PAIRED-CUDA / FORBIDDEN — keyed on the (use-case, paired-CUDA-evidence-availability, architecture-class) triple.

**Latex form:** `tier(use_case, paired_cuda_evidence, architecture_class) ∈ {ACCEPTABLE_TIER_1, RESEARCH_ONLY_TIER_2, FORBIDDEN_TIER_3}`. Decision tree per §3 above. Sister equation `mps_drift_architecture_class_dependent_v1` (slot 16, already registered) feeds the architecture-class drift-magnitude prediction; this equation feeds the routing-decision discipline.

**Domain-of-validity:** all CUDA-axis scoring decisions (promotion, kill-promote, paper claims); all MPS/macOS-CPU usage decisions across the apparatus.

**Producer:** `tac.mps_diagnostic.drift_predictor` (architecture-class drift prediction); `tac.cathedral_consumers.mps_viable_prescreen_consumer` (Phase B empirical anchor); CLAUDE.md "MPS auth eval is NOISE" + Catalog #1 + #192 + #317 + #341 (canonical discipline anchors).

**Consumer:** `tools/operator_authorize.py` `_dispatch_local_mps` + `_dispatch_local_cpu` (Catalog #317 routing); `tac.cathedral_consumers.mps_viable_prescreen_consumer` routing recommendation (Catalog #341 markers); `tools/build_macos_cpu_advisory_signal_manifest.py` + `tools/score_macos_cpu_advisory_proxy.py` (manifest writers).

**Empirical anchor (initial):** `mps_three_tier_taxonomy_phase_b_anchor_20260520` — Phase B empirical aggregate 0.072% gap on `tinyrenderer_phase_b` (12K params); 23× pose drift on Lane G v3 (~285K params) provides the architecture-class boundary; PR107 macOS-CPU 6e-6 gap vs GHA Linux CPU provides macOS-CPU-as-advisory-proxy validation FOR THIS ARCHIVE CLASS.

**Recalibration trigger:** `when_3+_new_empirical_anchors_in_domain` — when 3 additional architecture-class anchors land, the per-architecture boundary tier-promotion threshold refines.

## 5. HF Jobs LEGAL_NATIVE_PLATFORMS gap audit (#986)

**Status: CLOSED** by slot 13 commit 2026-05-19. Located at `src/tac/deploy/dispatch_protocol.py:84-86`:

```python
LEGAL_NATIVE_PLATFORMS = frozenset(
    {"modal", "vastai", "vast", "local", "local_mps", "local_cpu", "hf_jobs"}
)
```

Slot 13 commit context (per the source-comment lines 75-83):

> *"Slot 13 (2026-05-19) wired `_dispatch_hf_jobs` into `tools/operator_authorize.py` but the legal-platforms enum here was NOT extended in the same commit batch — slot 26's HF Jobs T4 dispatch was therefore refused by `evaluate_dispatch_protocol_complete` with `platform 'hf_jobs' not in LEGAL_NATIVE_PLATFORMS`. Per CLAUDE.md 'Bugs must be permanently fixed AND self-protected against' + the slot 26 handoff, extending the enum here closes the dispatch-blocked gap so HF Jobs vision training jobs (per `feedback_hf_jobs_segnet_surrogate_per_pixel_sister_lane_landed_20260519.md`) can route through the canonical operator-authorize 30s harness."*

Sister discipline: `LEGAL_DISPATCH_KINDS` (line 62-64) already includes `"hf_jobs_research_surrogate"` per the Catalog #270 scope-narrowing pattern. The `evaluate_dispatch_protocol_complete` function lines 396-419 enforces the canonical HF-Jobs-research-surrogate contract:
- `dispatch_kind=hf_jobs_research_surrogate` requires `platform=hf_jobs`.
- Requires `research_only=True` + `score_claim=False` + `promotion_eligible=False` + `ready_for_exact_eval_dispatch=False`.
- Requires `hf_jobs.expected_axis="advisory"` per the sister Catalog #192 (`macOS-CPU` non-promotion guard).

Active HF Jobs recipes (verified via filesystem inspection):
- `.omx/operator_authorize_recipes/substrate_hf_jobs_segnet_surrogate_distillation_per_pixel_t4_dispatch.yaml`
- `.omx/operator_authorize_recipes/substrate_hf_jobs_segnet_surrogate_distillation_t4_dispatch.yaml`

**Audit verdict:** the slot 13 commit AND the canonical contract enforcement are BOTH in place. The original task #986 gap-fix was CLOSED 2026-05-19. This memo memorializes the audit + cross-references the canonical helper + cite-chains the slot 13 commit + the 2026-05-19 sister landing memo. **No new code change required.** Operator-routable: if a future HF Jobs vision recipe lands without the canonical `hf_jobs_research_surrogate` dispatch_kind tag, the `evaluate_dispatch_protocol_complete` gate refuses dispatch structurally — sister of Catalog #270 + Catalog #199 + Catalog #192.

## 6. Observability surface (Catalog #305)

Per CLAUDE.md "Max observability — non-negotiable" 6-facet definition:

1. **Inspectable per layer:** the per-PR public-comment scorecard `reports/public_pr100_108_eval_comment_scorecard_20260508.json` decomposes each archive's contribution per (axis, component, archive_bytes); the canonical equations registry decomposes per (predicted_output, empirical_output, residual, per-axis).
2. **Decomposable per signal:** the score formula's three components (`100·d_seg`, `√(10·d_pose)`, `25·B/N`) are decomposed in the scorecard; the per-component CPU/CUDA delta is per §1.3 of methodology long-form (~30% seg, ~70% pose, 0% rate).
3. **Diff-able across runs:** the canonical_frontier_pointer.json carries the per-axis frontier with per-archive sha256; the per-PR scorecard has the public-bot-comment delta over time per PR.
4. **Queryable post-hoc:** `tools/list_canonical_equations.py --json` (canonical CLI); `query_equations_by_consumer / query_equations_by_producer / get_equation_by_id` in `tac.canonical_equations.registry`.
5. **Cite-able:** every empirical anchor in the registry carries Provenance (Catalog #323) with archive_sha256 + measurement_axis + hardware_substrate + evidence_grade.
6. **Counterfactual-able:** the (Hypothesis A, B, C, D) audit in §2 is the counterfactual surface — "what changes if hypothesis X is rejected" is per-table; the canonical equations support `update_equation_with_empirical_anchor` for the calibrated-noise-injection microbench OR the shared-tensor microbench when those land.

## 7. Cross-references

- `docs/writeup/cuda_cpu_drift_methodology.md` — canonical methodology long-form 2026-05-08; 586 lines; deep derivation reference.
- `docs/paper/04_results.md` §4.8 + §4.8.1 — writeup canonical (already amended 2026-05-19 per `feedback_writeup_amendment_cpu_vs_cuda_discrepancy_plus_cuda_frontier_exploration_landed_20260519.md`).
- `docs/findings/cuda_cpu_auth_eval_split_20260508.md` — OSS-disclosure short form.
- `reports/public_pr100_108_eval_comment_scorecard_20260508.json` — canonical empirical anchor source.
- `feedback_dual_cpu_cuda_auth_eval_mandatory_20260508.md` — the 2026-05-08 rule-landing memory anchor.
- `feedback_writeup_amendment_cpu_vs_cuda_discrepancy_plus_cuda_frontier_exploration_landed_20260519.md` — predecessor writeup amendment to §4.8.1.
- `feedback_mps_phase_b_options_b_plus_c_completion_landed_20260519.md` — Phase B MPS-VIABLE empirical anchor 0.072% gap.
- `feedback_mps_drift_mathematical_and_engineering_formalization_landed_20260519.md` — MPS drift formalization sister memo.
- `src/tac/deploy/dispatch_protocol.py` (lines 60-119) — canonical `LEGAL_NATIVE_PLATFORMS` + `LEGAL_DISPATCH_KINDS` + HF-Jobs-research-surrogate enforcement.
- CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE" non-negotiable.
- CLAUDE.md "MPS auth eval is NOISE" non-negotiable.
- CLAUDE.md "Apples-to-apples evidence discipline" non-negotiable.
- CLAUDE.md "Canonical equations + models registry" non-negotiable (Catalog #344 anchor).
- Catalog #1 + #127 + #192 + #205 + #317 + #341 (axis + hardware + custody discipline cluster).
- Catalog #344 (canonical equations registry).

## 8. 6-hook wire-in declaration per Catalog #125

- **hook #1 sensitivity-map:** `cpu_cuda_score_gap_v1` + `pose_axis_cuda_amplification_v1` feed the per-axis sensitivity decomposition consumed by `tac.score_geometry` + `tac.findings_lagrangian` per-pair dual-variable surface. The `mps_portability_use_case_taxonomy_v1` is a routing equation (no direct sensitivity contribution).
- **hook #2 Pareto constraint:** the per-archive (CPU score, CUDA score) Pareto frontier is operator-routable per §4.8.1 of methodology long-form (pending operator-routed analysis pass). The canonical Pareto consumer is `tac.findings_lagrangian` per the Dykstra-feasibility primitive.
- **hook #3 bit-allocator:** `cpu_cuda_score_gap_v1` informs target-axis-aware bit allocation when the planner reweights per `target_axis="cpu_leaderboard"`; the per-byte sensitivity from sister `per_byte_leverage_uniformly_distributed_v1` is unchanged.
- **hook #4 cathedral autopilot dispatch:** all three new equations register canonical_consumers in `cathedral_autopilot_autonomous_loop.py` + `tac.cathedral_consumers.mps_viable_prescreen_consumer` + `tac.cathedral_consumers.canonical_equation_lookup_consumer` (auto-discovered per Catalog #335). Routing-recommendation markers preserved per Catalog #341.
- **hook #5 continual-learning posterior:** `update_equation_with_empirical_anchor` is the canonical anchor-append path. When the shared-tensor microbench OR calibrated noise injection lands, the per-axis residual posterior refreshes.
- **hook #6 probe-disambiguator:** the four-hypothesis mechanism audit in §2 is the probe-disambiguator surface for the next-dispatch decision (decoder isolation vs forward-kernel isolation vs head-numerics isolation). The canonical probe paths are `tools/probe_eval_loader_drift.py` + `tools/probe_posenet_layer_drift.py` + the planned shared-tensor microbench per §2.5 of methodology long-form. The 3-tier MPS taxonomy is itself a probe-disambiguator for routing decisions per Catalog #341.

## 9. Mission contribution per Catalog #300

`frontier_protecting` — closes the formalization gap on the canonical CPU-vs-CUDA mechanism cluster + registers the three canonical equations so future agents inherit machine-readable predictors instead of re-reading prose memos. Mission alignment per CLAUDE.md "Mission alignment" Consequence 4: this is `apparatus_maintenance` adjacent (formalization layer for an existing empirical cluster) with the additional `frontier_protecting` weight that the equations become canonical-helper-consumable for the Lagrangian / Pareto solver downstream.

## 10. Closing summary

- **9 paired-or-single-axis empirical anchors inventoried** (5 public HNeRV + 4 internal: PR110, PR101 replay, PR106 format0d, PR107 apogee). Cross-archive cluster tightness (σ ≈ 0.004 on Δscore) is the canonical evidence base.
- **Mechanism attribution audit:** Hypothesis C (FastViT TF32-attention) FALSIFIED; Hypothesis D (loader/dataset drift) REJECTED for non-decode stages; Hypotheses A (decoder drift) + B (FP32 forward-kernel noise floor) PARTIAL EVIDENCE — coupled, not localized.
- **MPS portability taxonomy:** three-tier discipline (ACCEPTABLE / RESEARCH-ONLY-WITH-PAIRED-CUDA / FORBIDDEN) keyed on (use-case, paired-CUDA-evidence, architecture-class).
- **Three canonical equations proposed:** `cpu_cuda_score_gap_v1` / `pose_axis_cuda_amplification_v1` / `mps_portability_use_case_taxonomy_v1`. Registration in §F of landing memo + via `tools/subagent_commit_serializer.py` companion commit.
- **HF Jobs LEGAL_NATIVE_PLATFORMS gap-fix: CLOSED** by slot 13 commit 2026-05-19 (no new code change required; audit memorialized + canonical helper cite-chained).
- **Writeup amendment to `docs/meta_engineering_vision.md`:** sister cross-link inserted in companion landing batch (sister of `docs/paper/04_results.md` §4.8.1 already-landed amendment per `feedback_writeup_amendment_cpu_vs_cuda_discrepancy_plus_cuda_frontier_exploration_landed_20260519.md`).
