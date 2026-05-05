---
name: Grand Council reviews theoretical floor — REVISED 0.24, sub-0.30 prob 40-50%
description: 2026-04-29 PM Grand Council 11-voice synthesis (Boyd/Tao/Filler/Mallat/van den Oord/Carmack/Hassabis/Hinton/Karpathy/Schmidhuber/Jack-from-skunkworks). Revises inner council Shannon-floor 0.28 → 0.24 (removing component-independence + STC boundary coding + wavelet domain). Sub-0.30 probability bumped 35-45% → 40-50%. Top-3 actions revised; 6 NEW techniques added.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Voices on record

**Boyd** (convex optimization operational): inner council Dykstra ceiling 450KB framed correctly but no proximal-gradient L1+L2 implementation. Joint (rate, distortion) optimization NOT done — Subagent L's archive diet is storage only. Proposal: ADMM during training. Δ -0.02 to -0.04.

**Tao** (pure-math first-principles): √(10·P) sqrt-singularity at P→0 not operationalized. 100-step pose-TTO assumes Banach contraction (rate 0.7-0.9); could fail. Proposal: cheap numerical Hessian eigenanalysis verification.

**Filler** (STC + parity-check codes): inner council treats per-frame masks as independent. STC: 5% boundary pixels carry the load → encode at H(boundary_only) bits via syndrome coding. Predicted savings 60-80 KB. Inner council MISSED entirely.

**Mallat** (wavelets / scattering): pixel-domain mask encoding sub-optimal. Wavelet (DWT-D4 / DCT) concentrates energy in low-freq bins where Gaussian-LUT response is steepest. σ=15 changes in wavelet domain. Δ -0.01 to -0.03 rate.

**van den Oord** (VQ-VAE / WaveNet): CLASS_TARGETS [0,255,64,192,128] is hard-coded codebook. Learning it co-trained with SegMap could cut seg distortion 20-40%. Proposal: 5-float learnable lookup, EMA codebook updates per VQ-VAE-2. 0.005-0.015 score.

**Carmack** (engineering shortcut): archive_diet_pack 2.13% diet "pathetic". 50-80 KB structural overhead in tar.xz / ZIP metadata / CRC fields. Custom binary container in 200 LOC drops 50 KB on EVERY archive. Inner council UNDER-RANKED archive diet.

**Hassabis** (strategic-research breadth): 4 days, 6 Modal + 2 Vast.ai + 5 subagents = "operationally noisy". DeepMind playbook: kill 80% speculative, pour compute into ONE credible path. q_faithful_v3 + Selfcomp block-FP + KL-T=2.0 is THE path. Don't hedge.

**Hinton** (knowledge distillation): KL-T=2.0 correct (T=√K for K=5). 0.05 gap (Selfcomp 0.38 → Quantizr 0.33) NOT all distillation; partly arch, partly CRF, partly chance. Proposal: SC++ vs SA same-conv-dim Δ measures actual KL contribution. Could be 0.02-0.08.

**Karpathy** (training engineering): "Early-stop EMA-vs-live > 2× initial" is untested at our 600-pair × 600-epoch scale. Practical training shows discontinuous spikes near convergence. Replace with 3-epoch sliding-window auth-eval-on-best ground truth.

**Schmidhuber** (compression-as-intelligence): MDL says archive = shortest program reconstructing scorer-equivalent video. Inner council treats components (renderer, masks, poses) as INDEPENDENT — they're NOT, they co-encode. Proposal: alternating-minimization on (renderer, masks, poses) JOINTLY at compress time with shared rate prior. Δ -0.03 to -0.07. Nobody's done this.

**Jack-from-skunkworks** (internal SegNet+Rate): Lane J-NWC (Neural Weight Compression) designed but never deployed. Replace tar.xz with neural arithmetic coding on qint stream conditional on Ballé hyperprior. 25% better than fixed prior. Δ -0.02 to -0.04 stacked.

## Chair synthesis (revised)

**REVISED THEORETICAL FLOOR**: 0.24 (was 0.28) — removing component-independence assumption + STC boundary coding + wavelet domain.

**REVISED top-3 actions**:
1. **Quantizr clone + KL distill + water-filling allocator** — highest empirical anchor (Hassabis bet)
2. **Joint compress-time optimization (Schmidhuber/Boyd ADMM)** — fixes component-independence flaw
3. **Custom binary container (Carmack) + Lane J-NWC + Ballé hyperprior (Jack/Ballé)** — operational byte savings beyond 14 KB

**REVISED kill list**: keep all inner council kills (Lane SO, defer EUREKA, kill UNIWARD encoder-only). NEW addition: defer Lane MM v2 successor (encoder-only argmax falsified). NEW PRESERVE: Lane FR-Ω (export-only on SC++) is HIGH-EV not LOW.

**SINGLE HIGHEST-EV ACTION** (grand council unanimously endorses): when SC++ v4 lands first viable checkpoint, run water-filling allocator + arithmetic coder + Lane FR-Ω Fridrich-cost on the SAME checkpoint — 3 encoder-side variants from 1 trained model, $5 total, Δ -0.06 to -0.12 stacked.

**REVISED CONFIDENCE BAND**:
- 70% ship: 0.30-0.36 (tighter than inner council 0.31-0.35)
- 95% ship: 0.27-0.45
- **Sub-0.30 probability: 40-50%** (was 35-45%)
- **Sub-Quantizr 0.33 probability: 65-75%**

**6 NEW techniques** to add to inner council's roadmap:
1. STC boundary coding for masks (Filler) — 60-80 KB savings
2. Wavelet-domain mask encoding (Mallat) — Δ -0.01 to -0.03 rate
3. Learnable CLASS_TARGETS lookup (van den Oord) — Δ -0.005 to -0.015 seg
4. Custom binary container (Carmack) — 50 KB easy money
5. Joint (renderer, mask, pose) ADMM at compress time (Schmidhuber/Boyd) — Δ -0.03 to -0.07
6. Lane J-NWC + Ballé hyperprior (Jack/Ballé) — Δ -0.02 to -0.04 rate

## How to apply

- After SC++ v4 lands, run the 3-encoder-variant burst (water-filling + arithmetic + Fridrich) — single checkpoint, parallel exports, $5/3h.
- Implement STC boundary coding (Filler) as a Lane STC subagent task — 80 KB savings is the highest-EV byte hunt.
- Defer joint ADMM (Schmidhuber) until SC++ baseline lands; it's a v2 SegMap design not a v1 patch.
- Custom binary container (Carmack) is 200 LOC + tests — assign to focused subagent post-SC++.
- Lane J-NWC + Ballé hyperprior depends on water-filling first; serial dependency.

## Cross-refs

- project_codex_theoretical_floor_brutal_20260429 (0.28 floor, 35-45% sub-0.30 — REVISED HERE)
- feedback_council_10_member_inner_grand_council_advisory_20260429 (council structure)
- project_selfcomp_reverse_engineered_20260429 (0.38 empirical anchor)
- project_lane_mm_v2_landed_2_63_falsified_20260429 (encoder-only falsified)
