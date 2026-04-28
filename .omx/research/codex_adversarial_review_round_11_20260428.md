# Codex Adversarial Review — Round 11 (2026-04-28)

Target: working tree diff (~22 modified files + 39 untracked, 1863 insertions / 489 deletions)
Verdict: needs-attention (4 findings)

## Finding 1 [HIGH, REAL] — Lane WC `--resume-from renderer.bin` will crash

`scripts/remote_lane_wc_curator_outlier.sh:174-184` extracts `renderer.bin` from Lane A archive into `$ANCHOR_RENDERER`, then Stage 5 passes that raw archive artifact to `train_renderer.py --resume-from`. The training code uses `torch.load(args.resume_from)`, but `renderer.bin` is in ASYM/FP4A binary format with custom magic bytes — `torch.load` rejects it.

**Impact**: Lane WC will fail deploy after spending hours on Stage 2 (feature extraction) + Stage 3 (Curator fitting). Wasted GPU.

**Fix**: Resume from PyTorch fp32 checkpoint, OR add ASYM/FP4A magic-byte detection to `--resume-from`, OR add a preflight that rejects raw renderer artifacts for resume.

## Finding 2 [HIGH, REAL] — Learnable pair/class weights don't actually learn

`src/tac/experiments/train_renderer.py:2236-2245` treats `LearnablePairWeights` and `LearnableClassWeights` as optimizer-trained parameter groups, but the Round 10 rewrite exposes only buffers (`lambda_pair`, `lambda_class`) and the training loop never calls `dual_update`. The advertised `--use-learnable-pair-weights` and `--use-learnable-class-weights` flags add empty optimizer groups OR leave weights fixed while logs claim adaptive behavior.

Also: class weights are never threaded into `_pcw`.

**Impact**: Lane W (hard-pair) and Lane PS (per-class SegNet) would train INERTLY — no actual weight adaptation. Any future Lane W/PS deploy is invalid.

**Fix options**:
- (a) Restore trainable `Parameter` wrappers on `raw` field (keep both Parameter + buffer for backward compat)
- (b) Wire explicit `compute_*_dual_update()` calls using observed losses/distortions, remove empty optimizer param groups
- (c) Add integration test proving weights change during a short training loop

## Finding 3 [HIGH, TRANSIENT] — Lane F-V5 tests import missing modules

`src/tac/tests/test_lane_f_v5_hardware_fp8.py:49-52` imports `tac.quantization_fp8` and `export_hardware_fp8_checkpoint`, but neither exists yet. Also expects profile `f_v5_hardware_fp8_dilated_h64` which `get_profile` rejects.

**Impact**: CI blocker for Check 38 (test-imports-resolve).

**Status**: Lane F-V5 codex CLI session still in flight (codex `019dd406...`). Will land source modules + profile when complete. **Transient — auto-resolves.**

## Finding 4 [MEDIUM, TRANSIENT] — GP deploy script test points at missing script

`src/tac/tests/test_remote_lane_gp_script.py:7-12` hardcodes `scripts/remote_lane_gp_gaussian_process_pose.sh`, which doesn't exist. Every test calls `SCRIPT.read_text()` and fails.

**Status**: 8 council EUREKA deploy scripts codex CLI session still in flight (codex `bl1hcoay8`). Will land that script when complete. **Transient — auto-resolves.**

## Council action

Findings 1 + 2 require immediate fix. Findings 3 + 4 will auto-resolve when in-flight codex sessions land. Dispatch focused codex:rescue subagent for Findings 1 + 2.
