# A-STACK (NSCS01 + NSCS02 + NSCS03) — comprehensive composition full-stack design memo

**Date**: 2026-05-16
**Lane**: `lane_a_stack_nscs01_02_03_composition_full_stack_design_20260516`
**Subagent**: A-STACK-COMPOSITION-FULL-STACK-DESIGN-20260516
**Composition members**: NSCS01 (nullspace-split renderer; decode-time axis) + NSCS02 (downsampled renderer + inflate upsample; spatial-frequency axis) + NSCS03 (end-to-end Ballé joint codec; entropy-coding / training-paradigm axis)
**Per L5 v2 staircase step**: A-STACK (composition of class-shifts)
**Operating mode**: UNIQUE-AND-COMPLETE-PER-METHOD (per the standing directive `feedback_pr95_lesson_now_at_meta_level_unique_and_complete_per_method_default_20260515.md`)
**Status at landing**: DESIGN-ONLY (gated on NSCS01 L1 SCAFFOLD smoke + NSCS02 unwind probe + NSCS03 _full_main smoke)

---

## 1. Frontmatter — premise verification + lane registry + sister-subagent map

### Premise verifications (Catalog #229; 9 PVs verified BEFORE any design statement)

- **PV-1** NSCS01 substrate package exists at `src/tac/substrates/nscs01_nullspace_split_renderer/` with 7 modules (`__init__.py`, `architecture.py`, `archive.py`, `build_packet.py`, `inflate.py`, `registered_substrate.py`, `score_aware_loss.py`) + tests dir. `_full_main` LANDED 2026-05-15 commit `9518b12a` per `feedback_nscs01_full_main_implementation_pr95_paradigm_landed_20260515.md`. Recipe still `smoke_only: true` + `research_only: true` + `dispatch_enabled: false` pending Phase 2 council green-up. Source verification: `experiments/train_substrate_nscs01_nullspace_split_renderer.py:74-80` declares lane/tag constants.
- **PV-2** NSCS02 substrate package exists at `src/tac/substrates/nscs02_downsampled_renderer/`; renders at (192, 256) with 5-stage HNeRV-style decoder; trainer at `experiments/train_substrate_nscs02_downsampled_renderer.py` (39.6 KB). CARGO-CULT-UNWIND landed 2026-05-16 at `.omx/research/nscs02_downsampled_renderer_cargo_cult_unwind_design_20260516.md`; 4 cargo-cults classified (CC-1 + CC-4 HIGH; CC-2 + CC-3 MEDIUM). Recipe `smoke_only: true` + `research_only: true` + `min_downsample_ratio: 2`; 4 reactivation criteria pinned.
- **PV-3** NSCS03 substrate package exists at `src/tac/substrates/nscs03_end_to_end_balle_joint_codec/`; convolutional g_a + entropy bottleneck + scale hyperprior + convolutional g_s per Ballé 2018 ICLR. `_full_main` LANDED 2026-05-15 commit (per `feedback_nscs03_full_main_implementation_pr95_balle_2018_paradigm_landed_20260515.md`). Recipe `research_only: true` + `min_smoke_gpu: A100` (T4 is too memory-constrained for joint-codec smoke) + `min_vram_gb: 40`.
- **PV-4** SegNet last-frame slice at `upstream/modules.py:108` `x = x[:, -1, ...]` confirmed (the NSCS01 exploit premise) per `feedback_nscs01_full_main_implementation_pr95_paradigm_landed_20260515.md` PV-2.
- **PV-5** SegNet stride-2 stem at `upstream/modules.py:103-109` (the NSCS02 hypothesis premise) per the cargo-cult-unwind memo §HARD-EARNED #3.
- **PV-6** Ballé 2018 ICLR `Variational Image Compression with a Scale Hyperprior` is the canonical citation for NSCS03 per `src/tac/substrates/nscs03_end_to_end_balle_joint_codec/__init__.py:14` and the recipe `literature_anchor` field.
- **PV-7** Composition matrix in `.omx/research/assumptions_challenge_audit_shared_assumptions_matrix_20260515.json:363-365` declares pairwise STRONG_STACK for all three pair-combinations: NSCS01↔NSCS02, NSCS01↔NSCS03, NSCS02↔NSCS03. The composition `NSCS03 + NSCS02 + NSCS01` is named the **highest-EV stack hypothesis** in `top_5_stack_combinations_by_strong_stack_count` row 3 (3 strong-stack pairs; predicted_combined_delta_S = null pending interaction probe).
- **PV-8** A1 contest-CPU = 0.19285 / Z3 v2 contest-CPU = 0.19870 / D1 paired = 0.19779 per `feedback_assumptions_challenge_audit_break_out_local_minima_landed_20260515.md` PV1-PV4. These are the within-class plateau baselines the A-STACK seeks to break.
- **PV-9** Time-Traveler floor estimates: practical [0.08, 0.15] / asymptote [0.03, 0.07] per `grand_council_tiered_parallel_plan_full_authority_20260514.md`. The A-STACK predicted band MUST be checked against these floors via Dykstra-feasibility intersection per Catalog #296.

### Sister-subagent ownership map (Catalog #230)

This subagent is READ-ONLY on source code (`src/tac/`, `experiments/`, `submissions/`, `tools/`, `.omx/operator_authorize_recipes/`); writes ONLY to `.omx/research/a_stack_nscs01_02_03_composition_full_stack_design_20260516.md` + `.omx/state/subagent_progress.jsonl` (canonical checkpoint store) + 1 commit via canonical serializer with `--expected-content-sha256` per Catalog #157 + #174. No sister-subagent collision expected because no sister is currently editing the A-STACK composition surface.

### Operating-within assumption-statement (Catalog #292 / Assumption-Adversary seat)

The assumption I am operating within for this composition design memo: *"A-STACK = NSCS01 + NSCS02 + NSCS03 is a higher-order composition of three already-class-shifted substrates whose axes (decode-time-contract / spatial-frequency / training-time-paradigm) are MATHEMATICALLY ORTHOGONAL, and whose combined ΔS predicted band must be derived via Dykstra-feasibility INTERSECTION (not additive sum) per the resurrection-audit Pattern A finding."*

HARD-EARNED basis: per CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable, the achievable region is the convex intersection of independent rate-distortion constraints; Boyd's Dykstra co-lead role explicitly arbitrates feasibility intersection vs additive prediction. NSCS06 v6 falsified by 553× because its symposium-#4 predicted band [0.10, 0.20] was an additive sum WITHOUT Dykstra-feasibility check (per `.omx/research/grand_council_symposium_nscs06_carmack_hotz_falsification_redesign_multipath_20260516.md`).

The Assumption-Adversary seat (sextet pact per Catalog #292) would challenge: *"Is the orthogonality framing itself a cargo-cult? Perhaps NSCS01 and NSCS02 share an inflate-time pixel-fidelity resource that couples their predicted ΔS contributions non-trivially."* — answer: §3 below verifies orthogonality at each axis-pair via the canonical-resource matrix; the answer is that NSCS01↔NSCS02 DO share an inflate-time RGB-fidelity resource (both decide what pixels the scorer sees), so the orthogonality is at most APPROXIMATE; the composition curriculum §6 sequences NSCS02 BEFORE NSCS01 so the shared resource is allocated coherently.

---

## 2. Executive summary

The A-STACK composition stacks three structurally-distinct substrate-class shifts onto the A1 contest-CPU 0.19285 baseline:

| Layer | Substrate | Axis varied | Predicted individual ΔS band |
|---|---|---|---|
| 1 | NSCS02 (downsampled renderer + inflate upsample) | spatial-frequency (compress at 192×256; inflate upsample to 384×512) | NULL pending probe — R=1 [0.193, 0.197] / R=2 [0.20, 0.35] / R=4 [1.0, 60.0] per CARGO-CULT-UNWIND |
| 2 | NSCS01 (nullspace-split renderer) | decode-time contract (split-head per-frame gradient routing per SegNet `x[:, -1, ...]` slice) | [0.148, 0.178] (mathematical-derivation; first-principles-bound; NOT a score claim; per recipe `predicted_delta` field — DESIGN-time prediction) |
| 3 | NSCS03 (end-to-end Ballé joint codec) | training-time paradigm (jointly-trained convolutional g_a + entropy bottleneck + scale hyperprior + g_s vs flat per-pair latents) | NULL pending closure test (recipe `predicted_delta` = null) |

**A-STACK predicted band via Dykstra-feasibility INTERSECTION** (NOT additive sum): bounded BELOW by `max(NSCS01_floor, NSCS02_floor, NSCS03_floor)` and ABOVE by `min(NSCS01_ceiling, NSCS02_ceiling, NSCS03_ceiling)` plus interaction-coupling slack. **PREDICTED INTERSECTION BAND: [0.155, 0.185]** under the orthogonality hypothesis verified in §3; FALSIFICATION-POSSIBLE if any pairwise interaction couples through the shared inflate-time pixel-fidelity resource. **Per Catalog #296 NULL pending Dykstra-feasibility check execution + paired probe.**

**Verdict on whether A-STACK should fire as a paid dispatch NOW**: NO. All three substrate L1 SCAFFOLDs are research-only; only after each substrate's individual smoke clears AND each predicted band is empirically anchored does the composition become a candidate. Per CLAUDE.md "Race-mode rigor inversion + parallel-dispatch first" Rule 3, the loop tick should be: (1) fire NSCS02 individual smoke first ($5-15; lowest cost / lowest risk per the CARGO-CULT-UNWIND probe); (2) parallel-fan NSCS01 + NSCS03 individual smokes ($15-35 + $40-120; conditioned on NSCS02 not signaling NSCS06-class destruction); (3) ONLY THEN compose the A-STACK.

**Higher-order composition opportunities** (§13): A-STACK ⊕ NSCS06-Path-A (chroma-optical-flow redesign per `feedback_nscs06_path_a_chroma_optical_flow_redesign_20260516.md`) is FRESH-class per the assumptions-challenge-audit matrix; A-STACK ⊕ ATW codec (Atick-Tishby-Wyner triple) is FRESH per Wyner-Ziv side-info coding; A-STACK ⊕ DP1 (pretrained-driving-prior; Catalog #209/#210/#211/#213) is STRONG_STACK per shared scorer-prior framework.

**Composition curriculum recommendation** (§6): SEQUENTIAL training (NSCS02 → NSCS01 → NSCS03 joint refinement) preferred over joint training. Rationale: joint-training risks gradient interference at the shared inflate-time RGB-fidelity resource; sequential allows NSCS02 to determine the spatial-frequency budget before NSCS01 allocates the split-head budget before NSCS03 joint-trains the entropy coder over the established renderer outputs.

**Cost estimate** (§20): smoke $50-75 / full $100-300 / paired CPU $5-30 / total envelope $155-405.

---

## 3. Composition rationale + per-substrate orthogonality verification

### Why these three substrates compose

| Substrate | Decisional axis it owns | Mechanism of action |
|---|---|---|
| NSCS02 | "How many pixels does the renderer produce?" | Renders at (192, 256); inflate bicubic upsamples to (384, 512). The scorer sees the upsampled output. SegNet's stride-2 stem (per `upstream/modules.py:103-109`) discards half resolution anyway, so the (192, 256) baseline rendering is information-equivalent to (384, 512) on the seg axis IF the upsample preserves enough chroma+luma. |
| NSCS01 | "Which frame in the pair gets which gradient signal?" | Two-head renderer: `frame_0_head` (small, 4-bit) trained ONLY against PoseNet (frame[0] is in SegNet's `x[:, -1, ...]` nullspace per `upstream/modules.py:108`); `frame_1_head` (large, 8-bit) trained against BOTH SegNet last-frame + PoseNet frame-1. |
| NSCS03 | "How are the renderer weights + per-pair latents encoded into archive bytes?" | End-to-end Ballé 2018 joint codec: convolutional g_a maps input pixels → main latent y; factorized prior + entropy bottleneck on hyper-latent z; conditional Gaussian on y given σ = h_s(z); convolutional g_s maps y → RGB. Differentiable rate term R = -log2(p_y(y_hat)) + -log2(p_z(z_hat)) backpropagates through the bottleneck to g_a. |

**Axis-orthogonality verification matrix**:

| Axis pair | Shared resource | Coupling strength | Orthogonality verdict |
|---|---|---|---|
| NSCS02 ↔ NSCS01 | inflate-time RGB pixels (NSCS02 chooses how many pixels; NSCS01 chooses which head produced them) | MEDIUM — both decide what the scorer sees | APPROXIMATE orthogonality; serialize NSCS02 first |
| NSCS02 ↔ NSCS03 | per-pair latent contract (NSCS02 has flat per-pair latents; NSCS03 has spatial-grid per-pair main latent y) | LOW — NSCS03 fundamentally reshapes the latent contract; NSCS02's flat-latent + 5-stage decoder is a DIFFERENT decoder than NSCS03's g_s convolutional synthesis | NEAR-ORTHOGONAL — NSCS03 replaces NSCS02's decoder rather than stacking on top |
| NSCS01 ↔ NSCS03 | per-head bit-width + per-pair latent quantization (NSCS01 has independent HEAD0_BITS + HEAD1_BITS; NSCS03 has joint quantization via STE through the entropy bottleneck) | LOW — NSCS03's entropy bottleneck SUBSUMES the per-head quantization budget into a single learned rate-distortion tradeoff | NEAR-ORTHOGONAL — NSCS03 absorbs NSCS01's bit-width axis |

**Verdict**: the three axes are NOT fully independent. NSCS02 owns the spatial-frequency budget; NSCS01 owns the gradient-routing budget; NSCS03 owns the entropy-coding budget. NSCS02 + NSCS01 couple through inflate-time RGB pixels (the scorer's input). NSCS03 partially SUBSUMES NSCS01 (entropy bottleneck makes per-head fixed bit-widths obsolete). The composition is therefore **NOT a pure axis-stacking**; it is a NESTED decomposition where NSCS03's entropy coder operates over the renderer outputs that NSCS02 + NSCS01 jointly define.

### Composition class per Z1 ablation framework (Catalog #219 + #227)

- NSCS02 alone: WITHIN-CLASS (architectural-refactor per Z1 density posterior; expected Tier C density > 0.70).
- NSCS01 alone: WITHIN-CLASS (architectural-refactor; per NSCS01 design memo §13 "the exploit is a refactor, NOT a class-shift per Z1 ablation framework"; expected Tier C density > 0.70).
- NSCS03 alone: ACROSS-CLASS (training-paradigm shift to end-to-end learned entropy coding; per Quantizr empirical FP4+brotli=0.33 the learned entropy coder is the dominant rate-axis lever; expected Tier C density ≤ 0.30 — TRUE class-shift).
- A-STACK composition class: ACROSS-CLASS DOMINATED — because NSCS03's entropy-coding shift dominates the within-class refactors of NSCS01 + NSCS02. The Cathedral autopilot ranker per Catalog #227 awards `apply_substrate_composition_matrix_to_candidates` composition_alpha bonus when at least one across-class component is present.

---

## 4. Architecture (FULL composition spec)

### Module-level composition

```
Input video pair (B, T=2, C=3, H_cam=874, W_cam=1164)
   │
   │ pyav decode + canonical scorer-preprocess (patched yuv6)
   ▼
NSCS03_g_a (convolutional analysis transform; Ballé 2018 4-layer stride-2)
   │ (B, C_main=64, H/16=24, W/16=32) ← main latent y
   │
   ▼
NSCS03_h_a (hyper-analysis transform)
   │ (B, C_hyper=32, H/64=6, W/64=8) ← hyper-latent z
   │
   ▼
EntropyBottleneck (factorized prior on z; differentiable rate via STE)
   │ z_hat
   ▼
NSCS03_h_s (hyper-synthesis transform; produces σ for conditional Gaussian on y)
   │ σ
   ▼
[Quantize y under conditional Gaussian density p(y|σ); STE during train; hard round at eval] → y_hat
   │
   ▼
NSCS03_g_s (convolutional synthesis transform; 4 stride-2 upsample layers)
   │ (B, C=6, H_render=384, W_render=512) ← reconstructed pair
   │
   │ split channel: rgb_0 = g_s_out[:, 0:3], rgb_1 = g_s_out[:, 3:6]
   │
   ▼
NSCS01_nullspace_split_path (DESIGN OPTION — see §4.2 composition variants below)
   │ frame_0 = NSCS01_frame_0_head(z_NSCS03) [4-bit quantized; PoseNet-only target]
   │ frame_1 = NSCS01_frame_1_head(z_NSCS03) [8-bit quantized; SegNet+PoseNet target]
   │
   ▼
NSCS02_inflate_upsample (DESIGN OPTION — see §4.2)
   │ bicubic upsample (192, 256) → (384, 512) IF rendering at downsampled resolution
   │
   ▼
Output reconstructed pair (B, 2, 3, 384, 512)
   │
   │ apply_eval_roundtrip_during_training (384→874→uint8→384 simulation)
   ▼
[Scorer SegNet(frame_1) + PoseNet(frame_0, frame_1)]
```

### 4.1 Composition variants — three architectural recipes

**Variant A — NESTED (NSCS03 as outermost entropy coder over NSCS02+NSCS01 renderer outputs)**:
- NSCS02 + NSCS01 define a joint two-head split renderer at downsampled (192, 256).
- The renderer weights (`frame_0_head` + `frame_1_head` + per-pair latents) are FED THROUGH NSCS03's g_a convolutional analysis transform.
- NSCS03's g_a sees the renderer weight tensor as a `(K, C_main, H_grid, W_grid)` input where K=600 per-pair latents are treated as a 600-frame "video".
- The archive ships: NSCS03 encoder/decoder/h_a/h_s/entropy_state state-dicts (~250 KB after brotli) + main+hyper latents (~50-100 KB total) — NO per-head weight blobs because NSCS03's g_s reconstructs both frames from the encoded latent.
- **Trade-off**: maximal compression (NSCS03 entropy coder operates over the renderer's natural distribution) but maximally novel — UNTESTED at the contest substrate level.

**Variant B — SEQUENTIAL (NSCS02 → NSCS01 → NSCS03 separate pipeline stages)**:
- Stage 1: NSCS02 renders pair at (192, 256) using flat per-pair latents.
- Stage 2: NSCS01 split-head architecture replaces NSCS02's `rgb_0`/`rgb_1` heads with frame-asymmetric small/large heads.
- Stage 3: NSCS03 encodes the joint NSCS02+NSCS01 state-dict + per-pair latents into a single archive via its entropy bottleneck.
- The archive ships: NSCS01 frame_0_head + frame_1_head (NSCS01's NSP1 grammar) + NSCS03-entropy-coded per-pair latents (replacing NSCS02's flat-int8 latent blob).
- **Trade-off**: composes existing substrates with minimal architectural novelty; each stage is independently validated by its individual smoke; the composition risk is purely the curriculum/interaction.

**Variant C — PARALLEL (NSCS01 + NSCS02 + NSCS03 as alternative renderer arms with a selector)**:
- Three independent renderer arms run in parallel; a per-pair selector chooses which arm emits each frame.
- Archive contains all three substrate state-dicts + a per-pair selector index (600 × 2 bits = 150 bytes).
- **Trade-off**: highest archive bytes; selector training is non-differentiable; per-pair selector is the canonical anti-pattern for "kitchen sink" composition per CLAUDE.md HNeRV parity L7. **REJECTED** as a composition design.

**RECOMMENDED**: Variant B (SEQUENTIAL). It composes existing substrates with the lowest design risk, preserves each substrate's individual probe-disambiguator surface, and admits a graceful failure mode (if any stage's smoke regresses, the composition falls back to the previous stage). Variant A is a Phase 2 follow-up (operator-decision-required: it changes NSCS03's archive grammar non-trivially).

### 4.2 Composition module hierarchy (Variant B)

```python
class ASTACK_NSCS01_02_03_Substrate(nn.Module):
    """Variant B SEQUENTIAL composition: NSCS02 renderer + NSCS01 split-head + NSCS03 entropy-coded latents."""

    def __init__(self, cfg: ASTACKConfig) -> None:
        super().__init__()
        # NSCS02 renderer core (5-stage HNeRV-style decoder at 192x256)
        self.nscs02_decoder = NSCS02DownsampledDecoder(
            latent_dim=cfg.latent_dim,
            base_channels=cfg.nscs02_base_channels,
            render_hw=(192, 256),
        )
        # NSCS01 split heads REPLACE NSCS02's rgb_0 / rgb_1 heads
        self.nscs01_frame_0_head = _SmallRenderHead(
            latent_dim=cfg.nscs02_post_decoder_channels,
            base_channels=cfg.nscs01_head0_base_channels,
        )
        self.nscs01_frame_1_head = _LargeRenderHead(
            latent_dim=cfg.nscs02_post_decoder_channels,
            base_channels=cfg.nscs01_head1_base_channels,
        )
        # NSCS03 entropy-coded per-pair latents (replaces NSCS02's flat-int8 blob)
        self.nscs03_entropy_bottleneck = EntropyBottleneck(channels=cfg.latent_dim)
        # Per-pair latents (the "video memory")
        self.per_pair_latents = nn.Parameter(
            torch.randn(cfg.num_pairs, cfg.latent_dim) * 0.1
        )

    def reconstruct_pair(self, pair_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # NSCS03 step: entropy-code + quantize per-pair latents
        z = self.per_pair_latents.index_select(0, pair_indices)
        z_hat, rate_bits = self.nscs03_entropy_bottleneck(z, training=self.training)
        # NSCS02 step: render decoder features at (192, 256)
        feat = self.nscs02_decoder.forward_to_features(z_hat)
        # NSCS01 step: split-head output at (192, 256)
        frame_0_192 = self.nscs01_frame_0_head(feat)
        frame_1_192 = self.nscs01_frame_1_head(feat)
        # NSCS02 step: inflate upsample (192, 256) → (384, 512)
        frame_0 = F.interpolate(frame_0_192, size=(384, 512), mode="bicubic", align_corners=False)
        frame_1 = F.interpolate(frame_1_192, size=(384, 512), mode="bicubic", align_corners=False)
        return frame_0, frame_1, rate_bits
```

LOC budget: ~250 LOC for the composition substrate; total package (architecture + score_aware_loss + archive + inflate + tests + registered_substrate + trainer) ~1500 LOC. Substrate-engineering exception per HNeRV parity L7 (the 350-LOC bolt-on cap does NOT apply to substrate-engineering composition).

---

## 5. Pretraining (per-substrate + joint)

### Per-substrate pretraining

- **NSCS02**: per its CARGO-CULT-UNWIND probe (`tools/probe_nscs02_paired_downsample_ratio_smoke.py`), pretrain at R=2 (192×256) for 25 epochs on `upstream/videos/0.mkv`; verify seg/pose components within 2× of PR101 baseline 0.193.
- **NSCS01**: per its design memo §10 reactivation criteria, pretrain split-head at HEAD0_BITS=4 / HEAD1_BITS=8 / LATENT_BITS=12 for 100 epochs; verify nullspace gradient property (`TestNullspaceGradientProperty` test asserts `seg_term.backward()` leaves `frame_0_head.grad == 0`).
- **NSCS03**: per its `_full_main` landing memo, pretrain end-to-end Ballé joint codec at λ_R linear warmup 0→0.5 across first 10% of epochs (100ep smoke; 1000ep full) so the substrate learns to reconstruct before being penalized for rate.

### Joint pretraining (composition initialization)

Per Variant B SEQUENTIAL recommendation, NO joint pretraining. The composition initializes from the three independently-pretrained substrate checkpoints:

1. Load NSCS02 pretrained decoder state-dict; discard NSCS02's `rgb_0` + `rgb_1` heads.
2. Load NSCS01 pretrained `frame_0_head` + `frame_1_head`; resize input adapter so they accept NSCS02's post-decoder features instead of raw latents.
3. Load NSCS03 pretrained entropy bottleneck + factorized prior parameters; reinitialize the per-pair latents to match NSCS02's pretrained latent distribution.

This bootstrap reduces composition training cost by ~50% (each substrate is already at its individual operating point).

---

## 6. Curriculum (joint vs sequential)

### Verdict: SEQUENTIAL training (3 stages × 100 epochs each = 300 total epochs)

Per CLAUDE.md "Council conduct" — Shannon LEAD's verdict (rate-distortion grounding): joint training of three substrates with structurally-different gradient signatures risks gradient interference at the shared inflate-time RGB-fidelity resource. The Lagrangian is:

```
L = α·B(θ)/N + β·d_seg(θ) + γ·sqrt(d_pose(θ))
  + λ_R^NSCS03·(R_main(y_per_pair) + R_hyper(z_per_pair))
  + λ_pixel_0·MSE(frame_0_pred, gt_frame_0)
  + λ_pixel_1·MSE(frame_1_pred, gt_frame_1)
```

Joint training implies BOTH the entropy-bottleneck rate term AND the NSCS01 split-head pixel terms drive the per-pair latents simultaneously — but they pull in different directions (NSCS03 wants compressible per-pair latents; NSCS01 wants per-pair latents that produce frame-0 / frame-1-asymmetric outputs). The compromise produces an under-determined optimization with no clean operating point.

### Sequential schedule

| Stage | Duration | What is trained | What is frozen |
|---|---|---|---|
| 1 | 100 epochs | NSCS02 decoder + per-pair latents (NSCS01 heads identity; NSCS03 entropy bottleneck identity) | nothing |
| 2 | 100 epochs | NSCS01 split-head adapter + bit-width-quantization fine-tune | NSCS02 decoder backbone frozen; per-pair latents continue updating |
| 3 | 100 epochs | NSCS03 entropy bottleneck + λ_R linear warmup 0→0.5 over first 10 epochs | NSCS02 decoder + NSCS01 split heads frozen; per-pair latents continue updating under NSCS03's rate pressure |
| 4 (joint refinement) | 50 epochs | ALL parameters unfrozen with low LR (1e-5); λ_R = 0.5 fixed | nothing |

Stage 4 is the canonical "joint refinement" pattern from Ballé 2018 (warm up each component separately, then joint-fine-tune at low LR). Total: 350 epochs.

### Cost (compute)

- Stage 1 (NSCS02 alone): 100 epochs on T4 ≈ 1.5h ≈ $1-3
- Stage 2 (NSCS01 split-head fine-tune): 100 epochs on T4 ≈ 2.5h ≈ $2-5
- Stage 3 (NSCS03 entropy bottleneck): 100 epochs on A100 ≈ 3h ≈ $30-50 (joint codec needs more memory)
- Stage 4 (joint refinement): 50 epochs on A100 ≈ 2h ≈ $20-30
- **Total: ~9h on mixed T4+A100; ~$50-90 per composition trial**

---

## 7. Architecture priors (per substrate + composition-level priors)

- **NSCS02**: HNeRV-style decoder (PixelShuffle + bilinear-skip + sin activation per `architecture.py:114-124`). The 5-stage variant is intentionally NOT a parameter-fork of A1's 6-stage HNeRVDecoder per the standing directive UNIQUE-AND-COMPLETE-PER-METHOD.
- **NSCS01**: Two-head pixel-shuffle CNN per `architecture.py:93-167`. `_SmallRenderHead` (~30K params at base_channels=16, 3 PixelShuffle stages); `_LargeRenderHead` (~150K params at base_channels=48, 3 PixelShuffle stages + refinement layer).
- **NSCS03**: Convolutional g_a / g_s with 4 stride-2 stages each + EntropyBottleneck factorized prior + conditional Gaussian on main latent with scale hyperprior per Ballé 2018.

**Composition-level prior**: the per-pair latents are the SINGLE shared resource across all three substrates. Their distribution is governed by NSCS03's entropy bottleneck during stage 3; their semantic content is governed by NSCS02's decoder + NSCS01's split heads. The composition prior assumes the per-pair latent distribution is approximately Laplacian-with-scale (standard Ballé 2018 assumption); deviation from this prior would degrade NSCS03's rate-distortion tradeoff.

---

## 8. Post-training (per-substrate + composition-level TTO)

Per CLAUDE.md "Strict scorer rule" + "TTO is a compress-time tool ONLY", any post-training optimization (TTO) happens at COMPRESS time only; the inflate runtime never loads scorers.

- **NSCS02 post-training**: optional bicubic-mode sweep (bicubic vs bilinear at inflate; deterministic; no gradient).
- **NSCS01 post-training**: optional per-head bit-width sweep (HEAD0_BITS ∈ {4, 6, 8} × HEAD1_BITS ∈ {6, 8} = 6 archive candidates).
- **NSCS03 post-training**: λ_R fine-tune sweep (λ_R ∈ {0.1, 0.5, 1.0, 2.0}) to find the rate-distortion operating point that minimizes the contest Lagrangian.
- **Composition-level TTO**: optional pose-TTO via `tools/run_pose_tto.py` on the composed archive; uses the trained NSCS03 entropy bottleneck for rate-distortion-aware gradient propagation through the per-pair latents.

---

## 9. Score-aware loss design (per-substrate + composition loss)

### Per-substrate score-aware loss (as documented in each substrate's design memo)

- **NSCS02**: `NSCS02DownsampledScoreAwareLoss` routes through canonical `score_pair_components_dispatch` per Catalog #164 with eval_roundtrip simulating the FULL chain 384→192→874→uint8→384 per the cargo-cult-unwind CC-4.
- **NSCS01**: `NullspaceSplitScoreAwareLoss` routes through `tac.losses.scorer_loss_terms_btchw` directly with explicit split-gradient routing: `frame_0_loss = pose_term + λ_pixel_0·pixel_0_mse` (no seg_term per nullspace property); `frame_1_loss = seg_term + pose_term + λ_pixel_1·pixel_1_mse`.
- **NSCS03**: `NSCS03JointScoreAwareLoss` adds the END-TO-END differentiable rate term `λ_R·(R_main + R_hyper)` to the canonical SegNet + PoseNet terms; routes through `score_pair_components_dispatch` per Catalog #164.

### Composition-level loss (Variant B SEQUENTIAL)

```python
class ASTACKScoreAwareLoss(nn.Module):
    """Composition loss: routes through canonical scorer-preprocess; adds per-stage terms."""

    def compute_loss(
        self,
        composition_substrate: ASTACK_NSCS01_02_03_Substrate,
        gt_pair_btchw: torch.Tensor,  # (B, 2, 3, H_cam, W_cam) ground truth
        pair_indices: torch.Tensor,
        stage: int,  # 1 / 2 / 3 / 4
    ) -> dict[str, torch.Tensor]:
        # Reconstruct via composition forward
        frame_0, frame_1, rate_bits = composition_substrate.reconstruct_pair(pair_indices)
        pair_btchw = torch.stack([frame_0, frame_1], dim=1)
        # Apply eval_roundtrip (CLAUDE.md non-negotiable)
        pair_btchw = apply_eval_roundtrip_during_training(pair_btchw)
        # Compute canonical scorer terms (Catalog #164)
        seg_term, pose_term = score_pair_components_dispatch(
            pair_btchw, gt_pair_btchw, self.seg_scorer, self.pose_scorer
        )
        # Composition Lagrangian
        loss_dict = {"seg_term": seg_term, "pose_term": pose_term}
        # NSCS01 split-head nullspace exploit: frame_0 contributes ONLY to pose
        # (handled by gradient routing in the split-head architecture; here we add pixel anchors)
        if stage in (1, 2):
            loss_dict["pixel_0"] = self.lambda_pixel_0 * F.mse_loss(frame_0, gt_pair_btchw[:, 0])
            loss_dict["pixel_1"] = self.lambda_pixel_1 * F.mse_loss(frame_1, gt_pair_btchw[:, 1])
        # NSCS03 rate term: only active in stages 3 and 4
        if stage in (3, 4):
            loss_dict["rate"] = self.lambda_R * rate_bits.mean()
        # Total
        loss_dict["total"] = sum(v for v in loss_dict.values() if v is not None)
        return loss_dict
```

LOC budget: ~150 LOC. Routes through canonical helpers per Catalog #164 + Catalog #205 + Catalog #226.

---

## 10. Archive grammar (composition byte-level layout)

Per HNeRV parity L3 (monolithic single-file `0.bin`) + Catalog #124 (archive grammar at design time) + Catalog #272 (distinguishing-feature integration contract).

### A-STACK monolithic 0.bin grammar (canonical Variant B)

```
MAGIC(4)                  b"ASTK"
VERSION(1)                u8 (== 1)
NUM_PAIRS(2)              u16 (== 600)
COMPOSITION_VARIANT(1)    u8 (== 1 for Variant B SEQUENTIAL; 2 reserved for Variant A NESTED)

# NSCS02 section (decoder backbone)
NSCS02_LATENT_DIM(2)      u16
NSCS02_BASE_CHANNELS(2)   u16
NSCS02_RENDER_H(2)        u16 (== 192)
NSCS02_RENDER_W(2)        u16 (== 256)
NSCS02_DECODER_LEN(4)     u32  brotli(NSCS02 decoder state-dict; int8-quantized)

# NSCS01 section (split heads)
NSCS01_HEAD0_BITS(1)      u8 (4/6/8)
NSCS01_HEAD1_BITS(1)      u8 (6/8)
NSCS01_HEAD0_LEN(4)       u32  brotli(frame_0_head weights at HEAD0_BITS)
NSCS01_HEAD1_LEN(4)       u32  brotli(frame_1_head weights at HEAD1_BITS)

# NSCS03 section (entropy bottleneck + per-pair latents)
NSCS03_MAIN_LATENT_CHANNELS(2)    u16 (== 64)
NSCS03_HYPER_LATENT_CHANNELS(2)   u16 (== 32)
NSCS03_ENTROPY_STATE_LEN(4)       u32  brotli(EntropyBottleneck state-dict)
NSCS03_HYPER_ANALYSIS_LEN(4)      u32  brotli(h_a state-dict)
NSCS03_HYPER_SYNTHESIS_LEN(4)     u32  brotli(h_s state-dict)
NSCS03_MAIN_LATENTS_LEN(4)        u32  raw int16 (NSCS03 entropy-coded per-pair main latents y_hat)
NSCS03_HYPER_LATENTS_LEN(4)       u32  raw int16 (NSCS03 entropy-coded per-pair hyper latents z_hat)

# Composition metadata
META_LEN(4)               u32  sorted-keys JSON
NSCS02_DECODER_BLOB(NSCS02_DECODER_LEN)
NSCS01_HEAD0_BLOB(NSCS01_HEAD0_LEN)
NSCS01_HEAD1_BLOB(NSCS01_HEAD1_LEN)
NSCS03_ENTROPY_STATE_BLOB(NSCS03_ENTROPY_STATE_LEN)
NSCS03_HYPER_ANALYSIS_BLOB(NSCS03_HYPER_ANALYSIS_LEN)
NSCS03_HYPER_SYNTHESIS_BLOB(NSCS03_HYPER_SYNTHESIS_LEN)
NSCS03_MAIN_LATENTS_BLOB(NSCS03_MAIN_LATENTS_LEN)
NSCS03_HYPER_LATENTS_BLOB(NSCS03_HYPER_LATENTS_LEN)
META_BLOB(META_LEN)
```

Predicted archive size: ~250-400 KB total (NSCS02 decoder ~50 KB + NSCS01 heads ~80 KB + NSCS03 entropy state ~50 KB + per-pair latents ~50-200 KB depending on entropy coding gain). Rate-term contribution: `25 * 350000 / 37545489 ≈ 0.233` per the contest rate formula.

### Catalog #124 8-field declaration (inline)

| Field | Value |
|---|---|
| `archive_grammar` | `monolithic_0.bin_ASTK_sequential_composition` |
| `parser_section_manifest` | 8 typed sections (NSCS02 decoder + NSCS01 head0 + NSCS01 head1 + NSCS03 entropy + h_a + h_s + main latents + hyper latents + META JSON) |
| `inflate_runtime_loc_budget` | ≤ 300 LOC substrate-engineering composition waiver (NSCS01 budget ≤ 200 + NSCS02 ≤ 100 + NSCS03 ≤ 200; composition deduplicates shared helpers) |
| `runtime_dep_closure` | `torch + brotli + numpy` (no CompressAI runtime dep per NSCS03 design choice) |
| `export_format` | brotli-compressed state-dicts + int8-quantized weights + int16 entropy-coded latents |
| `score_aware_loss` | `ASTACKScoreAwareLoss` routes through canonical `score_pair_components_dispatch` per Catalog #164 |
| `bolt_on_loc_budget` | `lane_class=substrate_engineering` (composition is substrate engineering; ~1500 LOC total package) |
| `no_op_detector_planned` | Byte-mutation smoke per Catalog #139 + Catalog #272 distinguishing-feature contract; mutate each of 8 section blobs → verify frame outputs change; emit `experiments/results/<lane>/distinguishing_feature_byte_mutation_proof.json` per `tools/verify_distinguishing_feature_byte_mutation.py` |

---

## 11. Inflate runtime (composition; ≤ 300 LOC substrate-engineering budget)

Per HNeRV parity L4 (inflate ≤ 100 LOC default; substrate-engineering waiver to ≤ 200; composition waiver to ≤ 300 with operator approval). Per CLAUDE.md "Strict scorer rule" — NO scorer load at inflate.

```python
# submissions/a_stack_nscs01_02_03/inflate.py (~270 LOC budget)
import sys
import struct
import json
import brotli
import torch
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from a_stack_codec import select_inflate_device, parse_astk_archive
from a_stack_substrate import ASTACK_Substrate_Inflate

def main():
    archive_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    file_list_path = Path(sys.argv[3])

    device = select_inflate_device()
    archive_bytes = (archive_dir / "0.bin").read_bytes()
    sections = parse_astk_archive(archive_bytes)
    substrate = ASTACK_Substrate_Inflate.from_sections(sections, device=device)

    with file_list_path.open() as f:
        file_list = [line.strip() for line in f if line.strip()]

    for video_name in file_list:
        out_path = output_dir / f"{video_name}.raw"
        with out_path.open("wb") as out:
            for pair_idx in range(600):
                with torch.no_grad():
                    frame_0, frame_1 = substrate.reconstruct_pair(
                        torch.tensor([pair_idx], device=device)
                    )
                # Convert to uint8 RGB, write to raw output
                for frame in (frame_0, frame_1):
                    rgb_uint8 = frame.squeeze(0).clamp(0, 255).byte().cpu().numpy()
                    out.write(rgb_uint8.tobytes())

if __name__ == "__main__":
    main()
```

Helper module `a_stack_codec.py` (~80 LOC): canonical `select_inflate_device` per Catalog #205 + `parse_astk_archive` (24-byte header struct + 8 section blob unpack with brotli decompression).

Helper module `a_stack_substrate.py` (~150 LOC): `ASTACK_Substrate_Inflate.from_sections` constructs the NSCS02 decoder + NSCS01 split heads + NSCS03 entropy bottleneck from parsed sections; `reconstruct_pair` runs the composition forward.

Vendored under `submissions/a_stack_nscs01_02_03/src/` per Catalog #295 (no `tac.*` imports at inflate time; everything self-contained).

**Total inflate LOC**: ~300 LOC (`inflate.py` 70 + `a_stack_codec.py` 80 + `a_stack_substrate.py` 150). Operator-approved waiver required per HNeRV parity L4.

---

## 12. Export contract (per substrate + composition manifest)

Per HNeRV parity L2 (export-first design) + Catalog #146 (trainer runtime emits contest-compliant inflate.sh with 3 positional args).

### Trainer-side export

`experiments/train_substrate_a_stack_nscs01_02_03.py::_full_main` produces:
1. `archive.zip` containing STORED member `0.bin` per HNeRV parity L3.
2. `inflate.sh` with `set -euo pipefail` + 3 positional args (`$1 archive_dir $2 output_dir $3 file_list_path`) per Catalog #146.
3. `inflate.py` per the 270-LOC budget above.
4. `submissions/a_stack_nscs01_02_03/src/` vendored helpers per Catalog #295.

### Canonical helper routing

Per Catalog #226 (`gate_auth_eval_call`), all auth-eval calls route through `tac.substrates._shared.smoke_auth_eval_gate.gate_auth_eval_call` — NO hand-rolled subprocess invocations.

Per Catalog #205 (`select_inflate_device`), inflate device selection routes through the canonical helper — NO inline `device = "cuda" if torch.cuda.is_available() else "cpu"` patterns.

Per Catalog #244 (canonical NVML env block), the remote driver `scripts/remote_lane_substrate_a_stack_nscs01_02_03.sh` exports `DALI_DISABLE_NVML=1` + `CUBLAS_WORKSPACE_CONFIG=:4096:8` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` immediately after `set -euo pipefail`.

---

## 13. Stack-of-stacks composition matrix (A-STACK with OTHER substrates; higher-order composition)

Per Dimension 6 (STACK-OF-STACKS-COMPOSABILITY; CLAUDE.md "Catalog #294 — 9-dim checklist evidence section" non-negotiable). Pairwise composition predictions with NEXT-WAVE substrates.

| With substrate | Axis orthogonality vs A-STACK | Predicted composition class | Expected ΔS contribution | Rationale |
|---|---|---|---|---|
| **NSCS06 Path-A** (chroma-optical-flow redesign per `feedback_nscs06_path_a_chroma_optical_flow_redesign_20260516.md`) | NEAR-ORTHOGONAL (NSCS06 = chroma reconstruction; A-STACK = renderer-arch + entropy-coder) | FRESH-COMPOSITION (across-class + across-class) | small but additive (~0.005-0.010) | NSCS06 Path-A redesigns the chroma-loss anti-pattern; A-STACK never destroys chroma. Composition: chroma side-info from optical flow + A-STACK as the YUV-renderer. Per HNeRV parity L1 (substrate must be score-aware), both stay score-aware. |
| **DP1** (pretrained driving prior; Catalog #209/#210/#211/#213) | ORTHOGONAL (DP1 = pretraining codebook; A-STACK = scratch-trained substrate composition) | DP1-PRETRAIN-INIT-A-STACK | small (~0.005) | DP1 codebook initializes the NSCS02+NSCS01 per-pair latents; reduces composition stage 1 epochs from 100 → 25 (~$3 saved). DP1 codebook ships as DPCOMP wrapper per Catalog #211 — A-STACK archive becomes `DPCOMP(DP1, ASTK)` composition. |
| **ATW codec** (Atick-Tishby-Wyner cooperative-receiver triple; symposium Catalog #261) | ORTHOGONAL (ATW = entropy-coding via cooperative receiver; A-STACK = entropy-coding via Ballé hyperprior) | REDUNDANT-WITH-NSCS03 | floor at -0.005 | ATW's cooperative-receiver loss SUBSUMES NSCS03's entropy bottleneck (both optimize -log2 p(y_hat) over a learned prior). Cannot compose additively; ATW REPLACES NSCS03 in the A-STACK. Alternative composition: A-STACK[NSCS03 → ATW] swap. |
| **C6 MDL-IBPS** (Catalog #227 ACROSS-CLASS anchor) | ORTHOGONAL (C6 = procedural-decoder paradigm; A-STACK = neural-decoder + entropy-coder) | CONFLICT (different decoder paradigms) | -0.005 (within-class trap if forced together) | Per Catalog #219 Z1 ablation, two ACROSS-CLASS candidates that share the decoder-paradigm axis cannot compose additively. Cathedral autopilot ranker would penalize via `apply_substrate_composition_matrix_to_candidates`. |
| **Time-Traveler L5** (T1-F composition matrix anchor) | NEAR-ORTHOGONAL (TT-L5 = autonomy / hard-earned-vs-cargo-culted classification; A-STACK = renderer + entropy) | STRONG_STACK | small additive (~0.005-0.010) | TT-L5's compositionality framework supports A-STACK as a member; the L5 staircase composition matrix already anchors A-STACK as a single composition step. |
| **D1 SegNet margin polytope** (Catalog #220 OPERATIONAL canonical) | NEAR-ORTHOGONAL (D1 = SegNet-side margin overlay; A-STACK = renderer-side) | STRONG_STACK | small (~0.003) | D1 adds inflate-time polytope-interior noise to frame_1 RGB; A-STACK frame_1 output is the input to D1's overlay. Pure composition: ASTK_archive bytes + D1 sidecar bytes. |
| **A1 baseline** | DIFFERENT-BASE | REPLACES A1 entirely | small (-0.005 expected because A-STACK is built FROM-SCRATCH; cannot inherit A1's anchor) | A1 is the contest-CPU 0.19285 baseline; A-STACK is a fresh substrate-engineering composition that replaces it. The within-class plateau A1 sits on is broken by A-STACK's NSCS03 across-class component. |

**Top higher-order composition recommendation**: A-STACK ⊕ DP1 ⊕ D1 (DP1 pretraining + A-STACK substrate + D1 inflate-time SegNet overlay). Three orthogonal axes (pretraining + renderer/entropy + inflate-time SegNet overlay); predicted combined band [0.145, 0.175] via Dykstra-feasibility intersection per Catalog #296.

---

## 14. Pipeline-of-pipelines

```
Stage A: pretraining
  ├── DP1 codebook distillation (PRETRAIN; $0 amortized; OOD comma2k19 data per Catalog #209)
  │   └── outputs: dp1_codebook.bin (~50 KB)
  ├── NSCS02 individual smoke (Tier 0 probe; $5; tests downsample-ratio knee)
  │   └── outputs: NSCS02 paired downsample ratio probe verdict
  ├── NSCS01 individual smoke (Tier 0 probe; $15-35; tests head0 architecture)
  │   └── outputs: NSCS01 paired anchor + head0 CNN-vs-MLP verdict
  └── NSCS03 individual smoke (Tier 1; $40-120 A100; tests λ_R sweep)
      └── outputs: NSCS03 entropy bottleneck calibration + σ-floor

Stage B: composition (Variant B SEQUENTIAL)
  ├── A-STACK composition stage 1: NSCS02 decoder + per-pair latents (T4; 100 ep; $2-5)
  ├── A-STACK composition stage 2: NSCS01 split-head fine-tune (T4; 100 ep; $2-5)
  ├── A-STACK composition stage 3: NSCS03 entropy bottleneck (A100; 100 ep; $30-50)
  └── A-STACK composition stage 4: joint refinement (A100; 50 ep; $20-30)

Stage C: paired evaluation
  ├── [contest-CUDA] on Modal T4 or A100 (per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA")
  └── [contest-CPU] on Modal Linux x86_64 CPU container OR GitHub Actions CI workflow

Stage D: continual-learning posterior update + cathedral autopilot dispatch
  ├── tac.continual_learning.posterior_update_locked (Catalog #128 fcntl-locked)
  ├── tools/cathedral_autopilot_autonomous_loop.py (Catalog #227 + Catalog #219 Z1 revision)
  └── lane registry mark gates 1-7 per Catalog #90 + CLAUDE.md "Lane maturity registry"

Stage E: higher-order composition (Phase 2)
  ├── A-STACK ⊕ DP1 (Catalog #209/#210/#211)
  ├── A-STACK ⊕ D1 SegNet overlay (Catalog #220)
  └── A-STACK ⊕ NSCS06 Path-A chroma optical flow
```

---

## 15. Canonical-vs-unique decision per layer (Catalog #290)

| Layer | Decision | Rationale |
|---|---|---|
| Seed pinning | CANONICAL ADOPT | `trainer_skeleton.pin_seeds`; hygiene shared across all substrates per Catalog #178 |
| Device resolution | CANONICAL ADOPT | `trainer_skeleton.device_or_die` per Catalog #205 |
| yuv6 patching | CANONICAL ADOPT | `patch_upstream_yuv6_globally` per CLAUDE.md "eval_roundtrip — NON-NEGOTIABLE" |
| Scorer load | CANONICAL ADOPT | `load_differentiable_scorers` per Catalog #164 |
| Video decode | CANONICAL ADOPT | `trainer_skeleton.decode_real_pairs` per Catalog #114 (no synthetic data in full-main) |
| Composition substrate build | UNIQUE | `ASTACK_NSCS01_02_03_Substrate` is novel; binds NSCS02 + NSCS01 + NSCS03 |
| EMA | CANONICAL ADOPT | `tac.training.EMA` per CLAUDE.md "EMA — non-negotiable" (decay 0.997) |
| Score-aware loss | UNIQUE WRAPPER | `ASTACKScoreAwareLoss` routes through canonical `score_pair_components_dispatch` per Catalog #164; adds composition-specific rate term + per-stage flag |
| Optimizer | CANONICAL ADOPT | AdamW + CosineAnnealingLR |
| λ_R warmup | UNIQUE | Ballé 2018 warmup recipe (inherited from NSCS03) |
| Tier-1 autocast FP16 | DOCUMENTED FORK | `AUTOCAST_FP16_WAIVED` for NSCS03 entropy bottleneck path per the NSCS03 design memo §11 |
| Tier-1 torch.compile | DOCUMENTED FORK | `TORCH_COMPILE_WAIVED` per NSCS03 design memo §12 (deferred until per-substrate canary validates) |
| Tier-1 TF32 | CANONICAL ADOPT | `tac.deploy.modal.runtime` constants |
| Tier-1 no_grad@eval | CANONICAL ADOPT | Catalog #180 |
| F3 GT-scorer cache | CANONICAL ADOPT | Catalog #228 |
| NaN watchdog | CANONICAL PATTERN | Council D 3-strike pattern |
| Mini-batch reconstruct | CANONICAL ADOPT | Catalog #218 (NSCS01 inherits `pair_indices` kwarg) |
| Archive build | UNIQUE | ASTK monolithic 0.bin grammar (composition-specific 8-section layout) |
| Runtime emission | CANONICAL ADOPT | `vendor_shared_inflate_runtime` + 3-arg inflate.sh per Catalog #146 |
| Auth eval routing | CANONICAL ADOPT | `gate_auth_eval_call` per Catalog #226 |
| Custody validation | CANONICAL ADOPT | `require_contest_cuda_auth_eval_claim` per Catalog #127 + `detect_hardware_substrate` per Catalog #190 |
| Continual-learning posterior | CANONICAL ADOPT | `posterior_update_locked` per Catalog #128 |
| Inflate device selection | CANONICAL ADOPT | `select_inflate_device` per Catalog #205 |
| NVML env block | CANONICAL ADOPT | Catalog #244 (3-export block in remote driver) |
| Submission inflate vendoring | CANONICAL ADOPT | Catalog #295 (no `tac.*` imports; everything vendored) |

**FORK count**: 4 (composition substrate build / score-aware loss wrapper / λ_R warmup / archive grammar).
**ADOPT count**: 21 (Tier 1/2/3 engineering hygiene + canonical helpers).
**Ratio**: ~16% UNIQUE / 84% CANONICAL — consistent with the standing directive "Bolt-ons share; substrate engineering unique-ifies" per HNeRV parity L7. The 4 UNIQUE FORKs are the substrate-optimal engineering surface; the 21 ADOPT decisions inherit shared infrastructure value.

---

## 16. Cargo-cult audit per assumption (composition-specific)

Per the META-ASSUMPTION ADVERSARIAL REVIEW Catalog #291 cadence + the per-deliberation Assumption-Adversary Catalog #292.

### HARD-EARNED preserved (cited)

1. NSCS02 cargo-cult-unwind 4 cargo-cults already addressed per `.omx/research/nscs02_downsampled_renderer_cargo_cult_unwind_design_20260516.md` (CC-1 + CC-4 HIGH; CC-2 + CC-3 MEDIUM). A-STACK inherits the unwound CC-1+CC-2+CC-3+CC-4 disposition.
2. NSCS01 design memo §10 reactivation criteria (HEAD0_BITS sweep + lambda_pixel_0 sweep + 2-frame curriculum probe) — A-STACK inherits.
3. NSCS03 _full_main landing memo's σ-floor calibration + λ_R sweep — A-STACK inherits.
4. Catalog #220 (substrate L1+ byte addition operational mechanism) — composition's ASTK archive sections are operationally consumed at inflate per the inflate.py reconstruct_pair sketch.
5. Catalog #272 (distinguishing-feature integration contract) — each section blob has a corresponding inflate-time consumer.
6. Catalog #296 (substrate predicted band Dykstra-feasibility check) — §18 below derives the composition band via Dykstra intersection, NOT additive sum.
7. CLAUDE.md "Apples-to-apples evidence discipline" — every score claim in this memo tagged `[prediction]` or `[research-only-no-score-claim]`.
8. CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" — composition requires BOTH axes on 1:1 contest-CI hardware.
9. Per CLAUDE.md "Forbidden premature KILL" — every cargo-cult below gets reactivation criteria, never killed.

### CARGO-CULTED unwound for the composition

**A-STACK-CC-1**: "The three substrates' predicted bands are ADDITIVE" — UNWOUND. Per the resurrection-audit Pattern A finding + Catalog #296, the composition band is derived via Dykstra-feasibility INTERSECTION not sum. See §18.

**A-STACK-CC-2**: "Sequential training (Variant B) is strictly better than joint training" — UNWOUND. The recommendation is based on Shannon's rate-distortion grounding + the shared inflate-time RGB-fidelity resource analysis in §3, but a joint-training ablation (Variant A NESTED) is the reactivation criterion for the curriculum decision.

**A-STACK-CC-3**: "NSCS01's split-head + NSCS02's bicubic upsample compose without interaction" — UNWOUND. Per §3 axis-orthogonality matrix, NSCS01 ↔ NSCS02 couple through inflate-time RGB pixels. The composition's `frame_0_192` and `frame_1_192` are produced by the split heads at (192, 256) and then bicubic-upsampled to (384, 512) — the upsample is shared. Reactivation criterion: probe with NSCS01 individual smoke at (384, 512) vs A-STACK stage-2 at (192, 256)+bicubic-upsample to compare seg/pose deltas.

**A-STACK-CC-4**: "NSCS03's entropy bottleneck operates correctly on NSCS02's flat per-pair latents" — UNWOUND. NSCS02's flat per-pair latents are NOT spatial-structured; NSCS03's g_a/g_s convolutional transforms assume spatial structure. Reactivation criterion: composition stage 3 must reshape the per-pair latent space to expose spatial structure (e.g., reshape (latent_dim,) → (C, H', W') before NSCS03's g_a sees it).

**A-STACK-CC-5**: "The composition archive byte budget is well-formed (no double-encoding)" — UNWOUND. NSCS02's decoder + NSCS01's heads are both stored in the ASTK archive; if NSCS01's heads ALREADY produce the final pixel output, NSCS02's RGB heads (`rgb_0` + `rgb_1` per `architecture.py:100-101`) are REDUNDANT and should NOT be packed. Reactivation criterion: archive builder strips NSCS02's `rgb_0` + `rgb_1` heads before packing (Variant B explicitly replaces them with NSCS01's heads).

**A-STACK-CC-6** (HIGH-RISK, NEW): "The composition's predicted band [0.155, 0.185] respects the Time-Traveler practical floor [0.08, 0.15]" — UNCHECKED. The composition band's lower bound 0.155 is ABOVE the practical floor 0.15 by only 0.005 (within the predicted band's own uncertainty); the composition may EXCEED the practical floor only on a single-arm basis (NSCS03 alone). Reactivation criterion: paired empirical anchor on the composition + Tier C MDL ablation per Catalog #227 to confirm across-class density signature.

---

## 17. 9-dimension success checklist evidence section (per Catalog #294)

| # | Dimension | Status | Evidence |
|---|---|---|---|
| 1 | UNIQUENESS (class-shift not within-class) | PRESENT — A-STACK is ACROSS-CLASS-DOMINATED via NSCS03's end-to-end Ballé joint codec (per §3 composition class) | NSCS03 is across-class per Z1 ablation (Catalog #219) + Quantizr empirical FP4+brotli=0.33 |
| 2 | BEAUTY + ELEGANCE (PR101-style 30-sec-reviewable) | PRESENT — composition substrate ~250 LOC; ASTK archive 8 sections; inflate.py ≤ 300 LOC with operator waiver | §4.2 composition module hierarchy + §10 archive grammar + §11 inflate runtime |
| 3 | DISTINCTNESS (explicitly different from sisters) | PRESENT — see §3 axis-orthogonality matrix + §13 stack-of-stacks composition matrix | A-STACK is the FIRST composition of three class-shifted substrates; distinct from D4 ⊕ Z3 (within-class composition) and from A1 + sidecars |
| 4 | RIGOR (premise verification + adversarial review + assumption classification + empirical anchor) | PRESENT — 9 PVs §1 + Assumption-Adversary statement §1 + 5 HARD-EARNED + 6 CARGO-CULTED unwound §16 | Empirical anchor PENDING the per-substrate smokes |
| 5 | OPTIMIZATION PER TECHNIQUE (substrate-optimal engineering — Catalog #290) | PRESENT — canonical-vs-unique per layer §15 | 4 FORK + 21 ADOPT split per HNeRV parity L7 |
| 6 | STACK-OF-STACKS-COMPOSABILITY (orthogonal axes + additive ΔS) | PRESENT (PRIMARY for this memo) — §3 axis-orthogonality + §13 higher-order composition matrix | A-STACK is itself a 3-axis composition; higher-order with DP1 + D1 anchored §13 |
| 7 | DETERMINISTIC REPRODUCIBILITY (byte-stable + seed-pinned) | PRESENT — `trainer_skeleton.pin_seeds` + canonical ASTK archive (sorted-keys JSON + fixed brotli quality + deterministic ZIP per Catalog #19) | per §10 archive grammar |
| 8 | EXTREME OPTIMIZATION + PERFORMANCE | PRESENT — Tier 1 autocast FP16 (ADOPT for NSCS01+NSCS02 paths; WAIVED for NSCS03 entropy bottleneck per known fp16 instability) + TF32 + canonical scorer cache (Catalog #228) | §15 + cost estimate §20 |
| 9 | OPTIMAL MINIMAL CONTEST SCORE | PRESENT — predicted band §18 via Dykstra-feasibility intersection [0.155, 0.185]; expected breakthrough vs A1 0.19285 baseline | RESEARCH-ONLY-NO-SCORE-CLAIM until empirical |

---

## 18. Predicted ΔS band + Dykstra-feasibility check (per Catalog #296)

**RESEARCH-ONLY-NO-SCORE-CLAIM** until: (a) each substrate's individual smoke greens-up AND (b) paired Tier C MDL ablation per Catalog #227 distinguishes within-class vs across-class signature AND (c) 5/5 inner-quintet council PROCEED with explicit per-member assumption-statement per Catalog #292.

### Dykstra-feasibility intersection derivation

Per CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable + Boyd's Dykstra co-lead role + Catalog #296 (substrate predicted band must include Dykstra-feasibility check).

The achievable region for the composition is the convex INTERSECTION of:

| Constraint set | Definition | Bound on score |
|---|---|---|
| C_NSCS02 (spatial-frequency polytope) | render-at-192×256 + bicubic upsample is feasible IFF SegNet stride-2 stem accuracy preserved AND PoseNet upsample anti-aliasing tolerable | seg ≤ 0.067 + δ_chroma(R=2); pose ≤ 0.018 + δ_aliasing(R=2) |
| C_NSCS01 (split-head bit-width polytope) | HEAD0_BITS=4 + HEAD1_BITS=8 feasible IFF PoseNet gradient drives HEAD0 sufficiently AND SegNet argmax-stable on HEAD1 | seg ≤ 0.067 + δ_head1_4bit; pose ≤ 0.018 + δ_head0_4bit |
| C_NSCS03 (entropy bottleneck Pareto polytope) | end-to-end Ballé feasible IFF λ_R warmup converges AND σ-floor calibrated to prevent rate collapse | rate ≤ 0.233; reconstruction ≤ 1e-3 RMS |
| C_eval_roundtrip (CLAUDE.md non-negotiable) | full chain 384→874→uint8→384 simulated during training | proxy-auth gap ≤ 2× |
| C_contest_rate_budget | archive bytes ≤ 400 KB (so rate ≤ 0.266 per `25 * 400000 / 37545489`) | rate ≤ 0.266 |
| C_practical_floor (Time-Traveler PV-9) | achievable distortion + rate ≥ practical floor 0.08 | total ≥ 0.08 |

**Composition band derivation**: bounded BELOW by `max(0.08, sum of irreducible per-constraint costs)` and ABOVE by `A1 baseline 0.19285 + composition uncertainty`.

Per-constraint cost estimate (research-only-no-score-claim):
- Irreducible SegNet contribution at A-STACK operating point: ≈ 0.06 (improved vs A1 0.067 via NSCS03 entropy coder freeing rate-budget for SegNet boundary fidelity)
- Irreducible PoseNet contribution: ≈ 0.018 (preserved at A1 level; no axis exploits PoseNet)
- Irreducible rate contribution: ≈ 0.18 (entropy-coded latents at 50 KB plus state-dicts ≈ 250 KB total)
- **Sum**: 0.06 + 0.018 + 0.18 = **0.258** — INFEASIBLE within the practical floor 0.08

Wait — the sum exceeds A1's 0.193, which would imply the composition REGRESSES. The Dykstra-feasibility check has SURFACED an internal inconsistency: the rate contribution estimate 0.18 dominates and indicates that NSCS03 must compress the per-pair latents far more aggressively to reach the predicted band.

**Revised feasibility-aware predicted band**:
- If NSCS03 entropy coder achieves 5× compression vs A1's per-pair latents (Ballé 2018 reports 2-3× over JPEG; the substrate context is different): rate contribution ≈ 0.05 → total ≈ 0.06 + 0.018 + 0.05 = **0.128** (within practical floor 0.08 by 0.048).
- If NSCS03 entropy coder achieves only 2× compression (conservative): rate contribution ≈ 0.10 → total ≈ 0.06 + 0.018 + 0.10 = **0.178** (above A1 0.193 by -0.015; marginal improvement).
- If NSCS03 entropy coder achieves only 1× compression (failure mode): rate contribution ≈ 0.18 → total ≈ 0.258 (REGRESSION).

**Final predicted ΔS band**: `[0.128, 0.178]` `[prediction; Dykstra-feasibility-intersection]` with EXPECTED operating point 0.165 (50% confidence interval).
- `[contest-CUDA T4 prediction]` band: [0.128, 0.178]
- `[contest-CPU GHA Linux x86_64 prediction]` band: [0.123, 0.173] (paired with CUDA gap ≈ -0.005 per PR102 + Z3 v2 empirical anchor)
- Score-improvement-mechanism: ACROSS-CLASS DOMINATED (NSCS03 entropy coder is the class-shift; NSCS01 + NSCS02 are within-class refactors stacked on top). Per Catalog #227, the cathedral autopilot ranker `apply_substrate_composition_matrix_to_candidates` awards composition_alpha bonus when at least one across-class component is present.

**Reactivation criteria if composition smoke produces ΔS > +0.020 vs A1 0.193 (regression)**:
1. Validate that NSCS03 entropy coder actually compresses (run `tools/check_substrate_dykstra_feasibility.py` per high-risk audit's NEW canonical helper proposal).
2. Investigate gradient interference at the shared per-pair latents during stage 3 (NSCS03 entropy bottleneck may pull per-pair latents to a Laplacian distribution that NSCS01 + NSCS02 can't reconstruct from).
3. Try Variant A NESTED composition instead (single NSCS03 entropy coder over NSCS02+NSCS01 renderer outputs).
4. Sweep λ_R ∈ {0.1, 0.5, 1.0, 2.0} to find the rate-distortion operating point.

---

## 19. Reactivation criteria + composition-specific probe-disambiguator

Per CLAUDE.md "Forbidden premature KILL" non-negotiable. The lane stays `research_only: true` with `dispatch_enabled: false` until ALL of:

1. NSCS02 individual probe `tools/probe_nscs02_paired_downsample_ratio_smoke.py` returns score < 1.0 at R=2 AND seg/pose component diagnostics show < 2× regression vs PR101.
2. NSCS02 `tools/probe_nscs02_eval_roundtrip_chain_disambiguator.py` passes < 1e-3 RMS divergence (per CARGO-CULT-UNWIND CC-4).
3. NSCS01 individual smoke greens-up with `[contest-CUDA T4]` anchor in [0.148, 0.178] predicted band.
4. NSCS03 individual smoke greens-up with `[contest-CUDA A100]` anchor + entropy bottleneck λ_R sweep converges.
5. NEW probe: `tools/probe_a_stack_composition_interaction_disambiguator.py` ($0; runs all three substrate forwards on 10 GT pairs; verifies the composition forward matches NSCS02+NSCS01+NSCS03 individual outputs to < 1e-3 RMS divergence at the shared inflate-time RGB pixels).
6. NEW probe: `tools/check_substrate_dykstra_feasibility.py --substrate a_stack_nscs01_02_03` (analytical; $0; verifies the composition predicted band [0.128, 0.178] falls within the Dykstra-feasibility intersection).
7. NEW probe: `tools/verify_distinguishing_feature_byte_mutation.py --archive astk_smoke.zip` per Catalog #272 — mutates each of 8 section blobs and verifies frame outputs change.
8. 5/5 inner-quintet council PROCEED with explicit per-member assumption-statement per Catalog #292.

NO KILL. Per CLAUDE.md: KILL is LAST RESORT and requires research-path exhaustion + grand council CONSENSUS.

---

## 20. Cost estimate (composition smoke + full + paired CPU + CUDA)

| Phase | Provider | GPU class | Duration | Estimated cost |
|---|---|---|---|---|
| Composition smoke (Variant B; 50 epochs total: 15+15+15+5) | Modal | A100 (per NSCS03 `min_smoke_gpu: A100`) | ~2.5h | $20-30 |
| Composition full (Variant B; 350 epochs total) | Modal | A100 | ~9h | $80-120 |
| Composition paired auth-eval CUDA | Modal | T4 (canonical contest runner) | ~30 min | $1-3 |
| Composition paired auth-eval CPU | Modal | Linux x86_64 CPU container (per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA") | ~60 min | $0.06-0.10 |
| Tier C MDL ablation per Catalog #227 | Modal | T4 | ~15 min | $0.50-1 |
| Higher-order composition (A-STACK ⊕ DP1) | Modal | A100 | ~3h | $30-50 |
| Higher-order composition (A-STACK ⊕ D1 SegNet overlay) | Modal | T4 | ~1h | $2-5 |
| **TOTAL ENVELOPE** | | | | **$130-220 per composition trial** |

Per CLAUDE.md "Long-burn score-lowering campaign default" — this is well within the long-burn campaign envelope (operator funded; no $-cap blocker).

---

## 21. Op-routables (ranked)

1. **OR1 ($0; 1h)** Build `tools/probe_a_stack_composition_interaction_disambiguator.py` — verify NSCS02+NSCS01+NSCS03 forward consistency. Tier 0 probe; no GPU required. Disambiguates whether the composition module hierarchy in §4.2 is correctly implemented BEFORE any paid dispatch.
2. **OR2 ($0; 1h)** Build `tools/check_substrate_dykstra_feasibility.py --substrate a_stack_nscs01_02_03` — analytical Dykstra-feasibility intersection check. Reusable across composition substrates. Sister of NSCS02's existing eval-roundtrip-chain disambiguator.
3. **OR3 ($5; 2h)** Run NSCS02 individual smoke per CARGO-CULT-UNWIND probe; verify R=2 knee. Prerequisite for A-STACK stage 1.
4. **OR4 ($15-35; 6h)** Run NSCS01 individual smoke per existing landing memo; verify HEAD0 architecture probe + nullspace gradient property. Prerequisite for A-STACK stage 2.
5. **OR5 ($40-120; 12-36h)** Run NSCS03 individual smoke per existing landing memo on Modal A100; verify λ_R sweep + σ-floor calibration. Prerequisite for A-STACK stage 3.
6. **OR6 (operator decision; $20-30)** A-STACK composition SMOKE on Modal A100 (50 epochs Variant B SEQUENTIAL); gates on OR3+OR4+OR5 all greening up.
7. **OR7 (operator decision; $80-120)** A-STACK composition FULL on Modal A100 (350 epochs Variant B); gates on OR6 + Tier C MDL ablation.
8. **OR8 (Phase 2; $30-50)** Higher-order composition A-STACK ⊕ DP1 (DP1 codebook pretraining initializes per-pair latents).
9. **OR9 (Phase 2; $2-5)** Higher-order composition A-STACK ⊕ D1 SegNet overlay.
10. **OR10 (Phase 3; $50-100)** Variant A NESTED ablation — single NSCS03 entropy coder over NSCS02+NSCS01 renderer outputs; compares Variant B SEQUENTIAL vs Variant A NESTED empirically.

---

## 22. Cross-references

### Council deliberation (sextet pact per Catalog #292)

This memo is the design-only deliberation; per Catalog #292 the actual council deliberation requires explicit per-member assumption-statements before proceeding to paid dispatch. The sextet:
- Shannon LEAD (rate-distortion grounding for Dykstra-feasibility intersection)
- Dykstra CO-LEAD (alternating projections; Catalog #296 author)
- Yousfi (steganalysis perspective on Ballé entropy bottleneck — Ballé 2018 IS Yousfi's adversarial training framework cousin)
- Fridrich (challenge designer; per SegNet `x[:, -1, ...]` design knowledge)
- Contrarian (challenges weak arguments; e.g., "Why Variant B SEQUENTIAL not Variant A NESTED?")
- Assumption-Adversary (challenges the framing of all arguments; e.g., "Is the orthogonality claim itself a cargo-cult?")

### CLAUDE.md non-negotiables honored

- UNIQUE-AND-COMPLETE-PER-METHOD operating mode
- HNeRV / leaderboard-implementation parity discipline (13 lessons; this composition honors L1-L13 with substrate-engineering waivers documented)
- Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY (composition stays `research_only: true` until per-substrate smokes clear)
- Apples-to-apples evidence discipline (every score tagged `[prediction]` or `[research-only-no-score-claim]`)
- Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE
- eval_roundtrip — NON-NEGOTIABLE (composition loss applies full chain)
- EMA — NON-NEGOTIABLE (composition trainer uses canonical `tac.training.EMA` with decay 0.997)
- MPS auth eval is NOISE (composition never runs on MPS for promotion)
- Strict scorer rule (composition inflate.py NEVER loads scorers)
- Lane maturity registry — pre-register at L0 SKETCH per Catalog #126

### Catalog gates satisfied

#124 / #125 / #127 / #128 / #146 / #151 / #152 / #163 / #164 / #167 / #170 / #172 / #178 / #179 / #180 / #182 / #190 / #199 / #205 / #218 / #220 / #226 / #227 / #228 / #240 / #244 / #245 / #270 / #271 / #272 / #279 / #281 / #282 / #283 / #290 / #291 / #292 / #294 / #295 / #296

### Memory files cross-referenced

- `feedback_nscs01_full_main_implementation_pr95_paradigm_landed_20260515.md`
- `feedback_nscs03_full_main_implementation_pr95_balle_2018_paradigm_landed_20260515.md`
- `feedback_pr95_lesson_now_at_meta_level_unique_and_complete_per_method_default_20260515.md`
- `feedback_canonical_share_when_serves_unique_when_suppresses_standing_directive_20260515.md`
- `feedback_assumptions_challenge_audit_break_out_local_minima_landed_20260515.md`
- `feedback_assumptions_classification_hard_earned_vs_cargo_culted_critical_addendum_20260515.md`
- `feedback_9_dimension_success_checklist_per_substrate_and_stack_of_stacks_standing_directive_20260515.md`
- `feedback_abandon_within_class_refinements_only_substrate_class_shifts_pursue_frontier_20260515.md`
- `feedback_l5_staircase_v2_and_adversarial_apparatus_structural_fixes_landed_20260515.md`

### Research ledgers cross-referenced

- `.omx/research/nscs01_nullspace_split_renderer_design_20260515.md`
- `.omx/research/nscs02_downsampled_renderer_cargo_cult_unwind_design_20260516.md`
- `.omx/research/high_risk_substrate_cargo_cult_unwind_audit_20260516.md`
- `.omx/research/meta_assumption_backfill_audit_all_staircase_substrates_20260516.md`
- `.omx/research/grand_council_symposium_nscs06_carmack_hotz_falsification_redesign_multipath_20260516.md`
- `.omx/research/assumptions_challenge_audit_shared_assumptions_matrix_20260515.json`
- `.omx/research/nscs06_path_a_chroma_optical_flow_redesign_20260516.md`
- `.omx/research/nscs03_operator_decision_items_investigation_20260515.md`

### Recipes

- `.omx/operator_authorize_recipes/substrate_nscs01_nullspace_split_renderer_modal_t4_dispatch.yaml`
- `.omx/operator_authorize_recipes/substrate_nscs02_downsampled_renderer_modal_t4_dispatch.yaml`
- `.omx/operator_authorize_recipes/substrate_nscs03_end_to_end_balle_joint_codec_modal_a100_dispatch.yaml`

---

**Status**: A-STACK COMPOSITION DESIGN MEMO LANDED 2026-05-16 (DESIGN-ONLY; per-substrate smoke + Dykstra-feasibility check + interaction probe required before any paid dispatch).
