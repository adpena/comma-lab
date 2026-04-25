# Latest Report -- 2026-04-25

## Session Addendum: Cool-Chic/C3 Experimental Prototypes

### Local Smoke Update
- Local 8-frame smoke passed for both `coolchic_renderer_smoke` and `c3_residual_renderer_smoke` after fixing two integration bugs.
- Bugs fixed: full-resolution GT eval-roundtrip reshape, and MPS FP4 QAT parametrization buffers staying on CPU.
- Smoke artifact report: `reports/local_smoke_coolchic_c3_20260425.md`.
- Self-compression smoke passed: uniform int4+LZMA2 exported the Cool-Chic smoke checkpoint at 16,509 B and the C3 residual smoke checkpoint at 16,877 B.
- Deterministic replay is metric-stable but not byte-stable on MPS: max dequantized FP4 delta was `4.58e-05`.

### 32-Frame Trend Update
- Trend artifact report: `reports/local_trend_coolchic_c3_20260425.md`.
- Cool-Chic 20-epoch trend: float/scorer loss moved slightly down, but best FP4 checkpoint stayed at epoch 5 (`93.4409`).
- C3 residual 20-epoch trend: float/scorer loss dropped from `92.3028` to `68.7140`, mainly through SegNet (`0.5399` to `0.2763`), but FP4 evaluation did not preserve the gain.
- Uniform int4+LZMA2 trend exports stayed near 16KB: Cool-Chic `16,295 B`, C3 residual `16,493 B`.
- CPU replay confirms cross-device scorer stability but not tensor stability: MPS scorer `93.6397184`, CPU scorer `93.6397169`, max dequantized weight delta `0.0147`.

### Evidence Tier
| Lane | Status | Evidence |
|------|--------|----------|
| Proven baseline | Trustworthy contest-compliant floor | `proven_baseline`, standard loss, eval roundtrip discipline |
| Cool-Chic-style renderer | Implemented prototype, unpromoted | Forward tests, small shared decoder check, local scorer/QAT smoke, FP4/int4 export smoke |
| C3-style residual renderer | Implemented prototype, unpromoted | Zero-init identity residual test, local scorer/QAT smoke, FP4/int4 export smoke |
| Repo-wide suite | Not clean for unrelated reasons | Scheduler syntax error and Kaggle registry assertions block full green |

### What Changed
- Added a Cool-Chic-style low-complexity renderer lane: learned multi-resolution latent grids plus a tiny shared synthesis decoder.
- Added a C3-style residual lane: base renderer plus zero-initialized coordinate MLP residual head.
- Registered deterministic profile-driven experiment entries: `coolchic_renderer_smoke`, `coolchic_renderer_full`, `c3_residual_renderer_smoke`, `c3_residual_renderer_full`.
- Added seed/determinism metadata to renderer training so experiments can be replayed instead of relying on ad-hoc flags.
- Updated the paper to separate proven archive facts from unpromoted Cool-Chic/C3 research lanes.

### Verification Completed
- Focused renderer/profile tests passed.
- Hardening profile validation passed for the affected profile families.
- Quantization, compliance, FP4 strict roundtrip, adversarial shape, CLI help, ruff, compile, and diff-whitespace checks passed on the changed surface.
- Full `src/tac/tests` remains blocked by unrelated existing failures:
  - `src/tac/tests/test_scheduler_cli.py` has an invalid `remote-gpu = ...` assignment.
  - `src/tac/tests/test_build_kaggle_kernels.py` expects `ASSET_DATASET_REF`, while the module exposes `ASSET_DATASET_SLUG`.
  - `src/tac/tests/test_scheduler_registry.py` compares sorted platform output to an unsorted expected list.

### Scientific Gate Before Promotion
1. Run deterministic smoke training from the named profiles only.
2. Score decoded outputs through eval roundtrip, not raw tensors.
3. Prove every neural artifact is inside `archive.zip`.
4. Measure inflate runtime under the contest budget.
5. Submit or authoritative-eval only after the local proxy artifact is reproducible.

### Council Read
Cool-Chic is the higher-upside base-representation experiment. C3 is safer as a residual codec because it preserves the semantic-mask prior and starts exactly at the base renderer. Neither should displace the proven baseline until it beats it through the full archive path.

---

# Latest Report -- 2026-04-15

## Session 35 Summary: DX Hardening + SegNet Paradigm Shift

### Current Best Scores
| Track | Auth Score | Details |
|-------|-----------|---------|
| TTO v5b (embedding) | **0.41** | 500-step, embedding loss, seg_odd_only |
| TTO v5a (output MSE) | **0.43** | 500-step, first valid TTO with PoseNet gradients |
| Renderer baseline | **0.87** | asym_v5_lagrangian_fixed, ep12600 |

### Paradigm Shift: SegNet Dominates at 77:1 Leverage Ratio

Step curve experiment (Vast.ai RTX 4090, 30 pairs) revealed:
- PoseNet saturates at 100 TTO steps (165.27 -> 0.042, 3970x reduction)
- SegNet contributes 98.7% of remaining score after PoseNet convergence
- 500-step breakthrough: SegNet moves from 0.5036 to 0.3435 (32% reduction)
- All future effort must target SegNet

### Three Breakthroughs Implemented (UNTESTED)

1. **Hinge loss for SegNet** (P0): Logit-margin hinge loss focuses gradient on
   boundary pixels at risk of argmax flip. Ignores already-correct pixels.
   Expected 2-5x faster SegNet convergence.

2. **Two-phase TTO** (P1): Phase 1 (100 steps) = joint PoseNet+SegNet optimization.
   Phase 2 (200+ steps) = SegNet-only on odd frames. Freezes even frames and
   PoseNet after Phase 1, preventing PoseNet regression during SegNet polish.

3. **Latent codes per pair**: Pair-specific learnable vectors for amortized TTO.
   Not yet integrated into deployment.

### Session Commits: 40+
- Hinge loss implementation in tac library
- Two-phase TTO with Phase 2 SegNet-only mode
- simulate_resize default changed to True
- check_vastai.py canonical DX script
- download_modal_tto_frames.py data permanence
- PROVENANCE.md experiment provenance documentation
- Pair difficulty map script (first run this session)
- Vast.ai tto_v6_hinge_phase2 experiment registered

### Council Decisions (Binding)
- Hinge loss approved unanimously (15-0)
- Two-phase TTO approved unanimously (15-0)
- Cosine LR killed (empirically worse than constant)
- SegNet is the binding constraint, all effort must target it

### What's Ready to Deploy
- `tto_v6_hinge_phase2` experiment in Vast.ai registry: combines ALL discoveries
  (embedding loss, hinge loss, two-phase, constant LR, simulate_resize, seg_odd_only)
- `tto_step_curve_hinge` experiment: validates hinge loss improvement curve
- Cost estimate: ~$0.12-0.25 per experiment on RTX 4090

### Vast.ai Budget
- Spent: $0.27 of $24.00 hard cap
- Remaining: $23.73
- All instances destroyed

### Critical Data on Modal Volume
- `asym_v5_lagrangian_fixed/tto_v5a_output_mse/tto_frames.pt` (auth 0.43)
- `asym_v5_lagrangian_fixed/tto_v5b_embedding/tto_frames.pt` (auth 0.41)
- MUST download before Modal access expires

### 18-Day Plan (Deadline: May 3, 2026)
1. Download Modal TTO frames (data permanence)
2. Hinge loss step curve validation
3. Two-phase TTO validation
4. Per-pair difficulty map -> adaptive budget allocation
5. Distillation targets from 500-step TTO
6. Lock final approach by April 21
7. Final submission packaging

---

## Renderer Baseline (reference)
- Track: `robust_current`
- Variant: `asym_v5_lagrangian_fixed`
- Platform: `modal_t4`
- Auth score: **0.87** (seg=0.21, pose=0.56, rate=0.10)
- Checkpoint: `renderer_best.pt` at ep12600
