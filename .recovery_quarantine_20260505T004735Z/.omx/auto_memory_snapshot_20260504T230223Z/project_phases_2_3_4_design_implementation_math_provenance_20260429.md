---
name: PHASES 2/3/4 detailed status — design + implementation + math + provenance + research per lane
description: 2026-04-29 PM. Comprehensive scoping of all 15 post-contest lanes (Phase 2 weeks 2-4: 5 lanes; Phase 3 weeks 5-12: 8 lanes; Phase 4 weeks 13-24: integration). Each lane has math foundation, implementation outline, dependency chain, kill criteria, paper section, current state. Goal: optimal lowest theoretical floor (target Phase 4 central 0.12-0.20, moonshot <0.12).
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Status convention

- ❌ NOT STARTED
- 🟡 PARTIAL (module/skeleton exists, not integrated)
- 🟢 READY (tests pass, awaits dispatch)
- 🔵 RUNNING / IN-FLIGHT
- ✅ LANDED + auth-eval verified

---

## PHASE 2 (Weeks 2-4): top-3 STC redesigns + Joint ADMM + wavelet + DARTS-S full

### Lane 9: Full STC boundary codec rebuild ❌

**Status**: codec source `src/tac/stc_boundary_codec.py` exists (510 LOC) but has the structural one-majority-class-plus-exceptions bug found by max-rigor codex (109M exceptions vs 11.8M boundaries on 1200 frames). Current implementation is dead.

**Math**: STC (Filler/Pevny/Fridrich 2011) embeds k bits into n cover pixels with H minimum-cost paths through a syndrome trellis. For our masks, k = number of class transitions ~5% of pixels, n = total pixels (236M for 1200 frames). At boundary_fraction=0.05 the codec stores ~1.8 bytes per coded pixel = 0.71 bpp raw / 0.26 bpp deflated. AV1 monochrome at CRF 50 achieves 0.014 bpp via interframe prediction + 2D context modeling. Gap: 18× worse deflated. Structural rebuild required to:
- Use temporal class-ID prediction (not per-frame independent)
- Replace one-majority-plus-exceptions with connected-component / scanline RLE
- Or: drop STC name, ship as residual codec only (Lane 9-AV1+RES)

**Implementation**:
- Replace `detect_boundary_pixels()` with temporal-residual-aware boundary set (mask = predicted_t1 XOR clean_t)
- Add per-frame `predict_from_prior(t)` = warped class-ids from frame t-1 via low-rank flow
- Encode {flow_params, residual} where residual is sparse class-correction stream
- Boundary fraction becomes adaptive (sparse residuals → low fraction)

**Dependencies**: requires Lane 18 (RAFT/radial pose) for cheap flow estimation OR Lane 11 (wavelet) for multi-scale residual, OR neither (use ego-flow homography).

**Kill criteria** (from codex hard-abandon rule): if 24h smoke does NOT yield total mask layer < 380KB OR scorer-improvement-worth-rate-cost with residual < 100-150KB, abandon.

**Paper section**: "Empirical R(D) frontier — STC boundary coding fails AV1 by 18×"; negative-result chapter.

**Provenance**: project_lane_stc_clean_source_FALSIFIED_20260429.md (FALSIFICATION withdrawn after MPS error); project_stc_redesign_verdict_20260429.md (codex top-1 redesign at 78% endorse).

### Lane 10: Joint ADMM stream allocator (Boyd) ❌

**Status**: not started. Boyd grand council voice flagged it as "not 4-day plan, but feasible under 30-day". Under 6-month budget, FULLY feasible.

**Math**: Multi-objective convex optimization. Variables: (x_seg, x_pose, x_rate) ∈ feasible region defined by archive byte budget. Objective: minimize 100·D_seg(x) + √(10·D_pose(x)) + 25·rate(x)/37545489. ADMM separates rate constraint from distortion via Lagrangian:

    L(x, z, λ) = 100·D_seg(x) + √(10·D_pose(x)) + λ·(rate(z) - rate_target) + (ρ/2)|x - z|²

Iterate: x ← argmin (proximal step on distortion); z ← rate-projection of x; λ ← λ + ρ(rate(z) - rate_target). Converges to KKT point of the rate-distortion frontier.

**Implementation**:
- New module `src/tac/joint_admm.py`: 200-300 LOC
- Variables: byte allocations across {renderer.bin, masks payload, poses, codebooks}
- Per-stream marginal rate functions estimated empirically from prior runs
- Ties into compress-time budget search at end of training

**Dependencies**: needs marginal-rate estimates per stream (run several Lane 1 base trainings at varying byte budgets first).

**Kill criteria**: if ADMM doesn't converge within 1000 iterations OR the achievable point doesn't beat the best independent-allocation result by ≥0.01 score, abandon.

**Paper section**: "Convex frontier of task-aware rate-distortion"; constitutes a methodological contribution.

**Provenance**: codex grand council 22-voice (project_grand_council_final_designs_20260429.md voice 11 Boyd).

### Lane 11: Wavelet residual codec (Mallat) ❌

**Status**: not started. Mallat grand council voice flagged it as "post-deadline paper lane" under 4-day budget. Under 6-month, viable.

**Math**: Wavelet sparsity for boundary positions. 2D-DWT (e.g., Daubechies-4) of class-ID image gives sparse coefficient pyramid. Most coefficients near zero except at boundaries. Encode:
- Lossy approximation at coarsest scale (LL band) — small bit budget
- Detail bands (LH, HL, HH) at multiple scales: encode only top-K coefficients per band via arithmetic coding

For 384×512 masks at 4 levels: scales {192×256, 96×128, 48×64, 24×32}. Storage at K=512 per band per level: ~24KB per frame × 1200 = 28MB lossless, OR ~0.5KB per frame × 1200 = 0.6MB lossy with 95% PSNR.

**Implementation**:
- `src/tac/wavelet_mask_codec.py` — 400-500 LOC
- pyWavelets dependency (already in requirements likely)
- Encoder: DWT → top-K selection → arithmetic-code positions+magnitudes
- Decoder: inverse DWT from quantized coefficients

**Dependencies**: standalone. Runs on raw class-IDs (compress-time SegNet output).

**Kill criteria**: if best wavelet config doesn't beat AV1's 421KB at lossless OR doesn't achieve <2× distortion at 200KB lossy, abandon.

**Paper section**: "Wavelet-domain task-aware compression"; comparison vs AV1 monochrome, vs STC boundary.

**Provenance**: codex grand council voice 14 (Mallat).

### Lane 12: NeRV / Cool-Chic mask codec ❌

**Status**: `src/tac/contrib/coolchic_renderer.py` exists for renderer use (Lane F-V2/V3) but NOT for masks. NeRV-as-mask-codec is fresh.

**Math**: NeRV (Chen et al. 2021, arXiv 2104.05079) overfits a coordinate-MLP to a video. For 1200 frames of 5-class masks at 384×512: train tiny 4-layer MLP with input (t, y, x) → 5-class logits. Quantize weights to int8 → ship 30-50KB weight stream. At inflate, run MLP forward over all (t, y, x) coordinates → reconstruct masks. Bit budget governed by MLP size not mask resolution → potentially 30-100× smaller than AV1.

**Implementation**:
- `src/tac/nerv_mask_codec.py` — 600-1000 LOC
- Training: stochastic gradient descent on cross-entropy vs argmax labels, eval_roundtrip=True for scorer parity
- Quantization: existing `quantization.py` infrastructure (FakeQuantSTE / FakeQuantFP4)
- Decoder: `inflate_renderer_nerv.py` runs MLP over all 384×512×1200 = 236M coords (at ~100M coords/sec on T4 = 2.4s — fits 30-min inflate)

**Dependencies**: standalone. Compress-time only requires GT video → SegNet argmax → train NeRV.

**Kill criteria**: if best NeRV size doesn't fit <80KB total mask payload AT lossless OR <40KB at 5% argmax-disagreement-rate, abandon.

**Paper section**: "Coordinate-network task-aware mask compression"; novel comparison vs AV1/STC/wavelet.

**Provenance**: codex grand council voice 15 (van den Oord) + voice 13 (Filler implicitly via boundary residual). Also codex STC redesign verdict top-2 at 44-46% endorse.

### Lane 13: DARTS-S full sweep ❌ (current 3-config is restricted)

**Status**: `experiments/results/lane_darts_s_v1_a3` running on Vast.ai 4090 with 3 configs (default/wide/deep). Full sweep requires re-architecture exploration across 8-12 configs.

**Math**: Differentiable Architecture Search (Liu et al. 2019). Mixed-operation continuous relaxation:

    o_ij = Σ_k (exp(α_ij,k) / Σ_k' exp(α_ij,k')) · op_k(x)

Train α (architecture parameters) jointly with weights via bi-level optimization. After convergence, discretize: pick top-K operations per edge.

For our renderer: explore {kernel size, channel count, depth, dilation, motion-conditioning topology} simultaneously. Restricted 3-config tests {default, wide, deep} only.

**Implementation**:
- `src/tac/darts_search_space.py` — search space definition (currently exists, restricted)
- Expand to 8-12 configs: {h32, h48, h64, h96, h128} × {depth1, depth2} × {standard, dsconv}
- Or: continuous DARTS with full search space, 200 epochs

**Dependencies**: Modal A10G or Vast.ai 4090 ($2-5 per config × 12 configs = $24-60 budget)

**Kill criteria**: if best DARTS-found architecture doesn't beat current Lane G v3 (1.05) by ≥0.05 at same archive size, abandon arch-search direction.

**Paper section**: "Architecture frontier for task-aware mask rendering"; ablation table.

**Provenance**: project_grand_council_final_designs_20260429.md priority 5 (restricted), project_6month_strategic_plan_20260429.md Phase 2 lane 13 (full sweep).

---

## PHASE 3 (Weeks 5-12): theoretical floor reduction research

### Lane 14: Multi-pass compress optimization ❌

**Math**: Iterate {pre-encode → analyze residual → re-encode with informed prior} until convergence. Each pass uses scorer-margin information from prior pass to allocate bits where score-loss is highest. Compress-time UNLIMITED.

**Implementation**: `experiments/multi_pass_compress.py` — wraps existing compress.sh in a 3-5 iteration outer loop with scorer-margin profiling between passes.

**Dependencies**: requires Lane 19 (SegNet logit-margin boundary fitting) for the analysis step.

**Kill criteria**: if 5-pass doesn't beat 1-pass by ≥0.005 score, abandon.

### Lane 15: Bit-level archive optimizer ❌

**Math**: Gradient search over the archive bit-stream itself. Treat the final archive bytes as a continuous-relaxation latent variable, compute ∂score/∂bytes via finite differences (or learned surrogate), step in the direction of lower score. End-to-end differentiable codec with a discrete bit-stream output via STE.

**Implementation**: `src/tac/bit_level_optimizer.py` — uses `torch.autograd.grad` through a STE-quantized archive representation.

**Dependencies**: requires Lane 14 (multi-pass) infrastructure.

**Kill criteria**: if doesn't beat the canonical archive build by ≥0.001 score, abandon.

### Lane 16: Bayesian MDL/evidence analysis ❌

**Math**: MacKay's MDL (Minimum Description Length) framework. Bayesian model evidence p(D|M) = ∫ p(D|θ, M) p(θ|M) dθ approximated via Laplace's method. Compare candidate codec families by evidence ratio. Gives a principled prior weight for stack composition.

**Implementation**: `experiments/mdl_analysis.py` — compute evidence per lane, produce ranking.

**Dependencies**: requires Lanes 9-15 results.

**Paper section**: "Bayesian model selection across codec families"; major contribution.

### Lane 17: Full IMP 10-20 cycle ❌ (skeleton exists)

**Status**: `experiments/train_imp_cycle.py` exists but only smoke-tested, no full run.

**Math**: Iterative Magnitude Pruning (Frankle & Carbin 2019, lottery-ticket hypothesis). Train → prune bottom-k% → rewind weights to early iteration → retrain. Repeat 10-20 cycles. Sparse subnetworks at 90% sparsity often match dense baseline; archive shrinks by 10×.

**Implementation**: `experiments/train_imp_cycle.py` — already exists, needs full 10-cycle dispatch.

**Dependencies**: needs canonical Lane 1 base checkpoint as cycle 0.

**Kill criteria**: if final 10-cycle network doesn't fit <30KB at <0.05 score regression, abandon.

### Lane 18: RAFT / radial pose preimage lane 🟡

**Status**: `src/tac/raft_pose.py` exists (untracked in git); not integrated. RAFT (Teed & Deng 2020) optical flow estimation.

**Math**: Compute optical flow between consecutive frames at compress time. Store low-rank approximation of flow field (e.g., 32 DCT coefficients per frame ≈ 76KB total per existing `flow_compress.py`). At inflate, warp frame N to frame N+1 using stored flow. Skips encoding redundant per-frame data.

**Implementation**: `src/tac/raft_pose.py` (exists), needs integration with `inflate_renderer.py`.

**Kill criteria**: if flow-warp doesn't reduce mask payload by ≥50KB at <0.005 distortion regression, abandon.

### Lane 19: SegNet logit-margin boundary fitting ❌

**Math**: Compute SegNet logit gradients at compress time (UNLIMITED). Identify pixels where the second-best class is within margin ε of argmax — these are FRAGILE pixels where score-loss is largest if encoding flips them. Allocate compression bits proportional to fragility.

Specifically: for each pixel, compute `margin = top1_logit - top2_logit`. Sort pixels ascending by margin. Encode top-N% (smallest margin = most fragile) with highest fidelity, encode rest with low fidelity (RLE / coarse quantization).

**Implementation**: `src/tac/scorer_margin_codec.py` — 300-500 LOC. Uses upstream/contest_scorer SegNet directly at compress time.

**This is the user's "use auth-eval scorer at compress time" direction operationalized.**

**Kill criteria**: if margin-weighted encoding doesn't beat uniform encoding by ≥0.005 score at same byte budget, abandon.

**Paper section**: "Score-aware compression via gradient margins"; novel contribution.

### Lane 20: Ballé hyperprior residual codec ❌

**Math**: Ballé 2018 entropy bottleneck + scale hyperprior. Latent y is encoded with rate `-log p_y(y|σ)` where σ is the hyperprior estimating the local entropy. The hyperprior itself is quantized + transmitted via factorized prior. Trained end-to-end with rate-distortion loss.

**Implementation**: `src/tac/balle_hyperprior_codec.py` — adapts the Cool-Chic infrastructure.

**Dependencies**: standalone, but learning curves benefit from Lane 19 margin signal as auxiliary loss.

**Kill criteria**: if best hyperprior doesn't beat the deflated archive by ≥0.01 score at same bytes, abandon.

### Lane 21: Decoder systems rewrite ❌

**Math/engineering**: Profile inflate_renderer.py → reduce non-essential overhead. Targets: ffmpeg subprocess startup (~200ms), torch import (~500ms), model deserialize (~100ms), per-frame loop overhead. Goal: reduce 30-min budget to 10-min, leaving 20 min of headroom for richer compute.

**Implementation**: rewrite hot loops in C++ extension or torch.jit.script; vectorize frame batching.

**Kill criteria**: if doesn't shave ≥10 min off inflate, abandon.

---

## PHASE 4 (Weeks 13-24): integration

### Lane 22: Final integration

Pick the BEST {semantic representation, pose representation, residual, entropy coder, quantization, distillation} from Phase 1-3 results. Build the integrated frontier. Score arithmetic per codex math (memory project_6month_strategic_plan):

- 60KB savings → -0.04 score
- SegNet -1e-4 → -0.01 score
- PoseNet -1e-4 (at baseline 0.0005) → -0.007 score

Target: 0.12-0.20 central, <0.12 moonshot.

### Lane 23: Paper reproduction harness

Frozen evaluator container + three independent reproductions + ablation table + public archive scripts. Engineering deliverable.

---

## Phase 1 dispatch protocol (per user mandate)

1. Each lane gets adversarial review (3-clean-pass, math + engineering + scientific rigor)
2. Grand council consensus approval (codex CLI gpt-5.5 xhigh)
3. Greenup process (preflight + tests + smoke)
4. Modal/Vast.ai dispatch with appropriate device/budget
5. Auth-eval on contest-CUDA before any kill/promote decision
6. Result + manifest + paper note saved per landing

## Cross-refs

- project_6month_strategic_plan_20260429.md (Phase 1-4 high level)
- project_grand_council_final_designs_20260429.md (4-day council, valid for Phase 1)
- project_stc_redesign_verdict_20260429.md (top-3 STC redesigns, Lane 9/12)
- /tmp/codex_runs/master_6month_strategic_recheck.log (full transcript)
- /tmp/codex_runs/master_30day_design.log (per-lane math/impl/review)
- src/tac/{stc_boundary_codec,raft_pose,contrib/coolchic_renderer,water_filling_codec,learnable_class_targets}.py (existing modules to reuse)
