---
name: Council stacking lanes to ≤0.5 (ideally ≤0.33) — 2026-04-29
description: Skunkworks extreme-rigor session post-Selfcomp-RE. 5 parallel lanes designed: MM/SS/SA/SC++/SO. Total ~$22/14h. Sequencing MM→SA→SC++/SO. Lane SC++ predicted 0.33 sub-Quantizr; Lane SO predicted 0.30 frontier.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
Council session 2026-04-29 ~10:30am after Selfcomp RE (project_selfcomp_reverse_engineered_20260429). Question: design highest-EV stacking lanes for ≤0.5, ideally ≤0.33 to beat Quantizr.

**Verdict**: 5 parallel lanes, ~$22 total / wall-clock 14h. Each lane probes a different Selfcomp paradigm shift; they can stack.

| # | Lane | Cost | Time | Predicted | What it changes |
|---|---|---|---|---|---|
| 1 | MM | $1 | 2h | 0.78 | mask encoding: 3ch discrete → 1ch Gaussian-LUT grayscale |
| 2 | SS | $4 | 8h | 0.55 | single mask + 6-DOF affine warp emits frame1+frame2 |
| 3 | SA | $5 | 12h | 0.45 | full SegMap (94K params) replacing ASYM (287K) |
| 4 | SC++ | $5 | 12h | **0.33** | SA + kl_on_logits(T=2.0) — Quantizr's edge |
| 5 | SO | $7 | 14h | **0.30** | SC + Hessian per-weight sub-1bpw quant |

Sequencing: MM ships first (~2h, validates LUT paradigm). Then SA + SC++ + SO start in parallel. SS starts after SA validates SegMap reproduces in our codebase.

Currently in-flight (q_faithful_v3, sz_phase2_v2, mae_v_v2, lane_w_v2) are NOT obsoleted — they probe orthogonal axes (Quantizr-replica, dilated arch, augmentation, hard-pair rate attack). Continue running them.

**How to apply**:
- Build out SegMap class as `src/tac/segmap_renderer.py` (verbatim from inflate.py: ResidualBlock + SegMap + decode_tensor_payload + reconstruct_weight + create_gaussian_softmax_lut + load_segmap)
- Build self-compression weight encoder as `src/tac/block_fp_weight_codec.py` (writes qint + per-channel exponents in HWOI layout)
- Build mask grayscale-LUT encoder as `src/tac/mask_grayscale_lut_codec.py` (forward: argmax→target_color; inverse: pixel→softmax via Gaussian LUT)
- Each Lane MM/SS/SA/SC++/SO needs its own `scripts/remote_lane_<id>_*.sh` + Modal dispatch via experiments/modal_train_lane.py
- Auth-eval everywhere: every lane must end with contest-CUDA inflate.sh + upstream/evaluate.py

Council members on record:
- Yousfi: MM is one-script change, ship in hours.
- Fridrich: KL distill (T=2.0) is the 0.05 gap between Selfcomp 0.38 and Quantizr 0.33.
- Quantizr: stacking my KL with Selfcomp's affine + Hessian per-weight = sub-0.33 frontier.
- Hotz: affine = pose at 6-DOF, the same 6 numbers wear two hats. Beautiful, replicate immediately.
- Contrarian: AV1 grayscale 30-min CPU budget risk; LUT σ=15 is unverified hyperparameter; arch search beyond clone is mandatory.

Risks logged:
- AV1 grayscale decode CPU budget (30-min cap on T4)
- LUT σ=15 may need empirical tune
- SegMap convergence in our codebase may not match Selfcomp's training infra
- Block-FP encoder is a new binary format requiring its own write+read+test path
