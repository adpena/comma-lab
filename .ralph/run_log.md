# run log

## 2026-04-25T15:30:00-05:00 — NUCLEAR #1 + FP4 fix: -0.237 rate-term lever staged

**Mask sweep landed (commit bc53d474):**
- AV1mono full CRF=63: 109KB / rate=0.072 / 0.74% pixel disagreement
- AV1mono half CRF=63 + naive frame-doubling: 59KB / 0.96% disagreement
- Current archive (CRF=50): 411KB / 0.281 / 0.22% disagreement
- Pareto winner: CRF=63 full — minimal disagreement at 4× rate-term reduction

**FP4 robustness fix landed (commits 70d754b3, 77379ab2):**
- Three composable improvements to `tac.fp4_quantize`:
  1. RESIDUAL_CODEBOOK (denser near zero) — opt-in via `--fp4-codebook=residual`
  2. robust_scale (p99.5 quantile) — opt-in via `--fp4-robust-scale`
  3. stochastic rounding (training-time dither) — `--fp4-stochastic`
  4. (bonus) ndim<2 buffers no longer quantized at export — closes silent
     train↔export drift on `renderer.freqs` (Linf 16.77 → 0.275)
- Audit on existing C3 trend ckpt (experiments/fp4_roundtrip_audit.py):
  - DEFAULT codebook: 4.75% zero-collapse (1739/36609 small weights die)
  - RESIDUAL codebook: 1.26% zero-collapse — **3.7× fewer dead weights**
- Trend re-test on 32 frames: baseline + fixed both plateau at 93.4409 —
  slice too small to exercise the fix (the audit on the production ckpt
  is the real evidence). Deciding test = scaled run (1200 frames).
- 10/10 new property tests pass (test_fp4_robustness.py).
- Train↔export consistency invariant pinned by 2 round-trip tests.

**Council deliberation (5-member inner: Yousfi/Fridrich/Hotz/Quantizr/Contrarian):**
- 5/5 approved CRF=63 ship after gating e2e measurement
- 5/5 voted RUN E2E NOW with current renderer (not Cool-Chic 16KB)
- 3/5 deferred NUCLEAR #3 warp until A and C measured
- Contrarian VETO: no combined claim until inflate.sh→evaluate.py confirms

**E2E gating archive built (369KB total, rate=0.246 vs current 0.483):**
- Components: dilated h64 renderer (297KB) + masks_crf63.mkv (109KB) + poses (7KB)
- Inflate + evaluate running in background (b2332dj1i)
- Expected: rate-term -0.237 minus SegNet impact from 0.74% disagreement
- Next entry: actual auth score result + decision on shipping

## 2026-04-15T07:00:00Z — CRITICAL BUG: TTO PoseNet gradients were ZERO the entire project

**Discovery timeline**:
- TTO v3 (embedding loss) results came back: PoseNet 0.0172 -> 0.0173 (WORSE after 50 steps)
- Council adversarial review convened to diagnose
- Contrarian: "optimization cannot make its own objective worse. Gradients are broken."
- Hotz traced the call chain: `preprocess_input()` -> `rgb_to_yuv6()` -> `@torch.no_grad()` in upstream `frame_utils.py`
- Training pipeline had a patch for this in its scorer loading path
- TTO pipeline used a different scorer loading path — no patch — zero PoseNet gradients
- 13-0 council vote to fix immediately

**What was broken**: Every TTO run (v1, v2, v3, v4) optimized with SegNet+rate gradients only. PoseNet was invisible to the optimizer. The "PoseNet improvements" in TTO logs were random noise.

**What was NOT broken**: Renderer training (auth=0.87) used a different code path that had the fix. The 0.87 baseline is valid.

**Root cause**: `@torch.no_grad()` decorator on `rgb_to_yuv6()` in upstream `frame_utils.py`. PoseNet calls this in its preprocessing. The decorator severs the gradient tape silently — no error, no warning.

**Projected score impact**: With working PoseNet gradients, TTO should reduce PoseNet from 0.031 to ~0.003 (10x, matching SegNet's proven TTO response). Auth: 0.87 -> ~0.35.

**The funniest part**: The biggest bug in the project was a single line decorator in code we don't own. Weeks of GPU time. The fix is removing (or wrapping around) one `@torch.no_grad()`.

## 2026-04-14T20:00:00Z — FOUR PARALLEL PATHS (council revised)

Path 1: Renderer + TTO — warm-start constrained gen from renderer frames (PoseNet 0.031→<0.005?)
Path 2: Joint pair generator — (mask1,mask2)→(frame1,frame2), no warp (Quantizr's approach)
Path 3: Scorer-space generation (Eureka #1) — optimize in PoseNet/SegNet feature space, invert to RGB
Path 4: Kaggle long training — free T4 hours on proven base config

Eureka #1 is the "perfect steganalysis" path: generate what the scorer SEES, not what humans see.
Artifacts from feature→RGB inversion land in preprocessing null-space, invisible to scorer.
Score projection 0.135 if it works. Quantizr's 0.60 uses neural renderer. This skips the renderer entirely.

bat00 unreachable (Tailscale down after sshd restart). Needs physical intervention.

## 2026-04-14T19:00:00Z — COUPLED + ANNEALING: PoseNet 0.163, snapshot strategy works

Improved coupled optimizer: compress weight annealing (1.0→0.1 after 40%) + PoseNet snapshot.
Before: PoseNet diverged 0.170→0.584. After: PoseNet 0.163 snapshot, divergence delayed to 0.236.
Kaggle mount path FIXED (datasets/owner/slug structure found via debug kernel).
Kaggle raft_only ran for first time ever, base died (different error TBD).
Quantizr deep intel: FP4 custom codebook, mask-as-video, marshal-obfuscated arch.
Council revised: constrained gen FROM NOISE is weak, but TTO ON TOP OF renderer is the killer combo.
  Renderer (PoseNet=0.031) + TTO refinement could push to PoseNet<0.005 → score ~0.52.

## 2026-04-14T18:00:00Z — COUPLED TRAJECTORY: PoseNet converges (independent diverged)

Independent constrained gen: PoseNet DIVERGES 0.17→0.80 in 200 steps (SegNet dominates)
Coupled trajectory (4D-Var): PoseNet CONVERGES 0.39→0.27 in 200 steps, still trending down
  Joint optimization over all frames with single Adam optimizer.
  PoseNet gradient flows through both frames of each pair simultaneously.
  2.5x faster (4.1 vs 1.6 steps/s on MPS). Estimated 1000 steps: 4 min MPS.
  This is the architecture that attacks the 47x PoseNet gap vs Quantizr.
  Kaggle base v5 still RUNNING (first successful Kaggle kernel ever).

## 2026-04-14T16:00:00Z — λ-SWEEP COMPLETE: 0.87 is architectural ceiling

λ_cap sweep from v5-best (auth=0.87): {500→0.90, 750→0.87, 1000→0.87, 1500→0.90, 2000→0.87}
Score plateaus at 0.87 across 3 different λ_cap values. Ceiling is architectural, not optimization.
Quantizr at 0.60 has 47x better PoseNet (0.00066 vs 0.031) — architecture gap, not tuning gap.
Council decision: kill λ-sweep path, pivot to constrained generation (joint frame pair output).
Kaggle still failing: dataset mount race condition persists even with 5-min retry. Need 30+ min wait.

## 2026-04-14T14:30:00Z — v5 CHECKPOINT LANDSCAPE SWEEP (9 auth evals)

Full drift curve from ep12500 to ep16999:
- ep12500: 0.93 | ep12600: **0.87** (BEST) | ep13000: 0.88 | ep13500: 0.96
- ep14000: 0.95 | ep14500: 0.90 | ep15000: 0.88 | ep15500: 1.26 (collapse starts)
- ep16000: 1.09 | ep16500: 1.15 | ep16999: 1.37

Sweet spot: ep12500-15000 (all < 1.00, beating v3). Two local minima at ep12600 (0.87) and ep13000/15000 (0.88).
Collapse onset: ep15500 (SegNet and PoseNet both degrade under weak Lagrangian).
Cost: 9 × $0.29 = $2.61. High information density — maps full Pareto drift trajectory.

## 2026-04-14T13:20:00Z — NEW BEST: asym_v5_lagrangian_fixed renderer_best auth=0.8700

- **auth=0.8700** — 13% improvement over v3 baseline (1.00 -> 0.87)
- Checkpoint: renderer_best.pt at ep12600 (only 200 epochs after resume from v3)
- PoseNet: 0.031 (35% better than v3's 0.048) — the big mover
- SegNet: 0.00217 (held flat vs v3's 0.00210)
- Rate: 0.00401 (identical — same model size)
- Key insight: R2 Lagrangian clamp (λ 10000→1000) on resume created a brief transient
  where the over-constrained v3 model found a better PoseNet basin before drifting
- Late checkpoint ep16999 REGRESSED to 1.37 — weaker constraints let model drift
- Council implication: short-horizon fine-tuning with reduced Lagrangian > long training
- v5 constraints_met ep16999: auth=1.37 (SegNet 0.004, PoseNet 0.075) — drift confirmed

## 2026-04-14T01:00:00Z — PRE-SUPERVISION BASELINE CONFIRMED: asym_v3_longer_tight auth=1.0000

- **auth=1.0000** — BEST RESULT TO DATE on asymmetric warp architecture
- seg=0.002104, pose=0.047917, rate=0.100355
- checkpoint: asym_v3_longer_tight/renderer_best.pt (pre-supervision, ~ep12400)
- CONFIRMED: supervision 7600 epochs starting from 1.0 regressed to 1.79–2.87
- Contrarian verdict UPHELD: supervision harmed the model
- Council diagnostic closed: pre-supervision was at 1.0, NOT broken before supervision
  → The 4 Lagrangian fixes are still valid but the failure is CONFIRMED supervision-caused
- IMMEDIATE ACTION REQUIRED: promote asym_v3_longer_tight/renderer_best.pt as current best
- NOTE: rate=0.100 vs dilated h64 rate=0.046, but total score 1.0 still beats 1.33
  → asym warp trades rate efficiency for much better seg (0.002) and pose (0.048)
- json: asym_v3_longer_tight/auth_eval_renderer_best.json (exists in volume, timestamp 2026-04-13T17:37:26Z)

## 2026-04-14T00:30:00Z — asym_v4_supervised best_proxy_constraints_met AUTH EVAL

- **auth=2.8700** (CONFIRMED: supervised training collapses PoseNet)
- seg=0.3002, pose=2.4707, rate=0.1004
- checkpoint: renderer_epoch16800_constraints_met.pt (selected by best_proxy_constraints_met strategy)
- proxy_score=0.7266 (score_projection, fallback — full_eval_score absent due to OOM)
- platform: Modal T4
- KEY FINDING: SegNet 0.003 = BEST EVER (equal to dilated h=64). PoseNet 2.47 = CATASTROPHIC (16× dilated).
- KEY FINDING: score_projection is NOT a reliable proxy for auth score when PoseNet collapses —
  it was tracking SegNet improvement (×33 better) while completely missing PoseNet collapse (×16 worse).
  score_projection as training proxy = useless for PoseNet-dominated regime.
- COMPARISON TABLE (asym_v4 supervised):
  ep16800 (best proxy): seg=0.003, pose=2.47  → total=2.87 ← best_proxy_constraints_met selected this
  ep19999 (periodic):   seg=0.566, pose=1.12  → total=1.79 ← randomly landed on more balanced epoch
  dilated_h64 baseline: seg=0.006, pose=0.060 → total=1.33 ← current best
- INVALIDATED: both supervised checkpoints are worse than dilated baseline; supervised approach FAILS
- AWAITING: raft_only (Kaggle v6) for valid A/B comparison on same architecture
- saved: /results/asym_v4_supervised/auth_eval_renderer_epoch16800_constraints_met.json

## 2026-04-13T23:00:00Z — Kaggle both kernels RUNNING (v6 datasets, canonical)

- asym_warp_raft_only: kernel v6 RUNNING, dataset v6 (all assets confirmed present)
- constrained_gen_smoke: kernel v5 RUNNING, tac-1.0.3 with P100 CPU fallback
- Dataset v6 = tac-1.0.3 + raft_flow.pt(900MB) + renderer_best_v3.pt + posenet_targets.bin + 0.mkv
- Root cause of prior failures: race condition — kernels ran while dataset was still processing
- Fix: push kernel AFTER dataset listing confirms all files present

## 2026-04-13T22:30:00Z — Kaggle kernel v4 + dataset v4 deployed

- asym_warp_raft_only: kernel v4 running, dataset v4 (tac-1.0.1 + raft_flow.pt + posenet_targets.bin + renderer_best_v3.pt)
- constrained_gen_smoke: kernel v3 running, video path fix in tac-1.0.1
- code_file pattern: train_renderer_fridrich.py IS the code_file — bootstrap preamble injects sys.argv before Click
- pyproject.toml bumped to v1.0.1 (commit 5c994bd1)

## 2026-04-13T21:25:00Z — asym_v4_supervised ep19999 AUTH EVAL

- **auth=1.7900** (REGRESSION from baseline ~1.0 at ep12400)
- seg=0.5664, pose=1.1188, rate=0.1004
- archive=150,715 bytes, 287,019 params
- checkpoint: renderer_epoch19999.pt (periodic, NOT best — best_score=0.6019 never updated = resume floor never beaten)
- platform: Modal T4
- key finding: 7600 epochs of PoseNet+RAFT supervision hurt the model. Both seg and pose regressed vs ep12400.
- hypothesis: supervision disruption phase extended past ep19999. Or supervision signal misaligned with scorer.
- status: NEGATIVE RESULT for supervised path at this epoch. Awaiting raft_only (Kaggle) for A/B comparison.
- saved: /results/asym_v4_supervised/auth_eval_renderer_epoch19999.json

## 2026-04-10T21:30:00-05:00 - promoted floor synchronized

- authoritative promoted floor: **1.33**
- variant: `dilated_h64`
- platform: `modal_a10g`
- evidence: `reports/raw/2026-04-10-dilated-h64-authoritative/robust_current-dilated-h64-authoritative-cpu-report.txt`
- mirrors are now expected to be derived from canonical promoted_result.json

## 2026-04-11T23:46:58 — mask_renderer on modal_a10g ep70

- proxy=None
- Notes: L1 loss 0.031, Phase 1 pretrain, 31s/ep

## 2026-04-11T23:47:00 — dp_sims on modal_a10g ep80

- proxy=None
- Notes: L1 loss 0.018 LEADING, Phase 1, 17.7s/ep

## 2026-04-11T23:47:00 — wavelet on modal_a10g ep110

- proxy=8.2
- Notes: Phase 2 scorer training, 12.9s/ep

## 2026-04-11T23:47:00 — dilated_h64 on modal_a10g ep103

- proxy=1.407
- Notes: Steadily improving from 1.476

## 2026-04-12T00:25:49 — dp_sims on modal_a10g ep100

- proxy=4.54
- PoseNet: 0.75700000
- SegNet: 0.05100000
- Notes: First Phase 2 epoch. FP4 saved 2.3MB. Score will improve rapidly.

## 2026-04-12T00:25:50 — dilated_h64 on modal_a10g ep78

- proxy=1.423
- PoseNet: 0.06700000
- SegNet: 0.03400000
- Notes: Steady improvement from 1.476 start

## 2026-04-12T00:29:21 — dp_sims on modal_a10g ep100

- proxy=4.54
- PoseNet: 1.47200000
- SegNet: 0.00700000
- Rate: 2.262
- Notes: Only 1 Phase 2 epoch. Died at scorer start. SegNet excellent (0.007). Must resume P2 on Lightning.

## 2026-04-12T00:29:21 — dilated_h64 on modal_a10g ep84

- proxy=1.413
- PoseNet: 0.07200000
- SegNet: 0.00600000
- Rate: 0.046
- Notes: 84/2500 epochs. INT8 45KB saved. On track for sub-1.3.

## 2026-04-12T00:33:59 — dp_sims on modal_a10g ep109

- proxy=2.93
- PoseNet: 0.53200000
- SegNet: 0.00600000
- Rate: 2.262
- Notes: 9 Phase 2 epochs. SegNet 0.006 matches best! PoseNet needs training. Trajectory: sub-1.5 by ep 150.

## 2026-04-12T00:33:59 — dilated_h64 on modal_a10g ep99

- proxy=1.4
- PoseNet: 0.07000000
- SegNet: 0.00600000
- Rate: 0.046
- Notes: 99/2500 epochs. On track. Needs 800+ more.

## 2026-04-12T01:17:03 — dp_sims on modal_a10g ep189

- proxy=2.5
- PoseNet: 0.48200000
- SegNet: 0.00300000
- Rate: 2.262
- Notes: FINAL Modal run. SegNet 0.003 TIES Quantizr! PoseNet 480x gap + rate 5.7x gap = entire remaining problem.

## 2026-04-25: DX Hardening + Deployment

### Review Rounds
- 11 consecutive review rounds, 64 issues found and fixed (7 critical)
- Key criticals: gradient flow killed (R1), padding_mode not serialized (R4), inline APG TypeError (R8), brotli DQ (R9)
- 3 consecutive clean passes at critical/important level achieved (greenup)

### Auth Score
- 2.26 [contest-compliant] via contest_eval.py → upstream evaluate.py
- SegNet: 0.238, PoseNet: 1.566, Rate: 0.457
- Float renderer (290KB), masks_crf50 (421KB), optimized poses (7KB), no Brotli

### Deployment
- SHIRAZ: A100 SXM4, focal_ste + curriculum + Fridrich, TTO targets
- WILDE: A100 SXM4, hinge + freeze/unfreeze + Fridrich, GT targets (unplanned A/B)
- GREEN: A100 SXM4, WILDE + use_zoom_flow, TTO targets
- Full auto-pipeline: train → QAT (50+250) → pose TTO (200) → auth eval → bundle
- Estimated cost: ~$15-18 total, within $24 cap

### DX Improvements
- contest_eval.py: single-command e2e eval matching upstream 1:1
- Half-frame mask auto-duplication with hard error on bad count
- All 7 renderer loaders handle padding_mode/use_dilation
- I4LZ decompression integrity check
- Archive rebuilt without Brotli (scorer doesn't have brotli package)
- Environment sanitization in contest_eval (strips INFLATE_TTO etc.)

### Flags for Results Analysis
- FLAG 1: WILDE GT targets vs SHIRAZ TTO targets (A/B test)
- FLAG 2: Phase 1 plateau at 181K params (architecture ceiling?)

## 2026-04-25T00:00:00-05:00 — Cool-Chic/C3 prototype implementation and paper update

- Implemented requested experimental items 1 and 3 as unpromoted lanes:
  - `coolchic_renderer`: learned multi-resolution latent grids plus tiny shared synthesis decoder.
  - `c3_residual_renderer`: base renderer plus zero-initialized coordinate MLP residual head.
- Added deterministic seed plumbing and metadata capture in renderer training.
- Added profile entries and tests for smoke/full Cool-Chic and C3 residual variants.
- Verification completed on the changed surface: focused renderer/profile tests, hardening profile tests, quantization/compliance tests, FP4 strict roundtrip, adversarial shape smoke, CLI help, ruff, compile, and diff whitespace.
- Repo-wide full suite remains non-green for unrelated scheduler/Kaggle blockers; do not represent the full repository as clean until those are handled.
- Paper/report/state updates now separate proven archive evidence from unpromoted prototype evidence.
- Council decision: keep Cool-Chic as the high-upside base-representation lane and C3 as the safer residual lane. No deployment to Vast.ai until deterministic smoke, eval-roundtrip proxy, archive audit, and authoritative evidence exist.

## 2026-04-25T03:45:00-05:00 — Local smoke testing: Cool-Chic/C3/self-compression

- Built an 8-frame local smoke dataset at `experiments/results/local_smoke_coolchic_c3_20260425/data/gt_frames.pt`.
- Fixed two smoke blockers:
  - full-resolution GT eval-roundtrip reshape bug in `train_renderer.py`;
  - MPS FP4 QAT NaNs caused by parametrization buffers not moving to the training device.
- Cool-Chic 2-epoch scorer/QAT smoke passed:
  - params=37,170;
  - FP4 checkpoint=56,525 B;
  - uniform int4+LZMA2=16,509 B.
- C3 residual 2-epoch scorer/QAT smoke passed:
  - params=36,492;
  - FP4 checkpoint=67,743 B;
  - uniform int4+LZMA2=16,877 B.
- Self-compression/mixed-precision exporter smoke passed. Crude `latents8` allocation increased bytes, which is expected; next allocation must be scorer-sensitive.
- Determinism finding: Cool-Chic replay has identical scorer metadata but not byte-identical MPS FP4 output; max dequantized delta `4.58e-05`.

## 2026-04-25T04:00:00-05:00 — 32-frame trend smoke

- Ran 32-frame/20-epoch MPS trend smokes for Cool-Chic and C3 residual.
- Cool-Chic:
  - loss epoch 5 -> 19: `94.3579` -> `93.7085`;
  - best FP4 scorer stayed at `93.4409` from epoch 5;
  - uniform int4+LZMA2 trend export: `16,295 B`.
- C3 residual:
  - loss epoch 5 -> 19: `92.3028` -> `68.7140`;
  - SegNet `0.5399` -> `0.2763`, PoseNet `147.0910` -> `168.9101`;
  - best FP4 scorer stayed at `93.4409` from epoch 5;
  - uniform int4+LZMA2 trend export: `16,493 B`.
- Interpretation: C3 residual has a real float-path learning signal, but the gain is not surviving FP4 evaluation. Next blocker is quantization robustness / mixed precision, not basic architecture viability.
- CPU replay of 8-frame Cool-Chic smoke: scorer-stable vs MPS (`93.6397169` CPU vs `93.6397184` MPS), but tensor delta up to `0.0147`.

## 2026-04-25T14:41Z — Rounds 23-26: SHIRAZ-class bug class hardened

Five rounds of fresh-eyes adversarial review (each with 3-5 parallel reviewers from rotating council perspectives) found 19 distinct bugs in the deployment chain across the validator system, pipeline orchestration, profile loading, archive packaging, and remote sync. Every CRITICAL is now fixed and pinned by tests. Commit chain:

- 3793a32e fix: Round 22-24 — qat_finetune missing arch CLI args, pipeline arch propagation, optimize_poses .pt loader
- (preflight commit) preflight: arity + profile + boolean-flag SHIRAZ guards (R23+R24)
- 04392166 critical: kill ad-hoc deployment forever — canonical pipeline only
- (battleplan) battleplan: R23+R24 postmortem, NUCLEAR experiments today, kill list
- (R25) fix: Round 25 — pipeline --profile, deploy_vastai compress subcommand, full provenance
- (R26) fix: Round 26 — _apply_profile honest semantics, kwarg-filter PipelineConfig, fail loud on typo

Validator state: 23/23 tests pass in `src/tac/tests/test_preflight_arity.py` covering: arity rules A-D, short-form alias indexing, per-scope list_vars isolation, top-level subprocess detection, bash -c wrapper detection, profile load, CLI override semantics, typo detection, unknown-profile failure. preflight_arity + preflight_profiles + check_codebase_drift run clean against the live repo.

Pipeline canonicalization: `pipeline.py compress --profile X --video Y --checkpoint Z` is the single entry point. Profile fills defaults; explicit CLI flags win. PipelineConfig built via kwarg-filter from args so new fields auto-thread. Full provenance (git hash, GPU, PyTorch, platform, timestamp) saved to pipeline_config.json. deploy_vastai.py launch invokes the proper subcommand.

NUCLEAR experiments queued (zero-GPU, run TODAY):
1. Half-frame mask AV1 sweep at 384×512 (CRF 24..48) — int8 overflow already fixed in encode_masks_monochrome (R25 reviewer corrected my battleplan claim)
2. SHIRAZ Phase 3 → immediate auth eval before any v2 deploy
3. Even-frame-warp-from-odd at inflate (Quantizr paradigm) — same architectural change as #1, NOT additive (R25 reviewer caught my double-counting)

