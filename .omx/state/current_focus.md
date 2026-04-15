# Current Focus -- 2026-04-15T24:00:00Z

## Session 36: Wrong Checkpoint Discovery + Hardening

### CRITICAL: Wrong Checkpoint Bug (2026-04-15)

ALL Vast.ai experiments used a 5-epoch smoke-test model (MD5: a9aee326)
instead of the auth=0.87 renderer (MD5: cff8dca4). This was discovered
when pair difficulty map results showed PoseNet mean=158 instead of ~0.017.

**Impact:**
- step_curve_v1/ results: INVALIDATED (absolute numbers meaningless)
- step_curve_cosine/ results: INVALIDATED (absolute numbers meaningless)
- Pair difficulty map v2: CORRECT (re-run with verified checkpoint)
- Hinge loss code: VALID but UNTESTED against correct model
- Two-phase TTO code: VALID but UNTESTED against correct model
- simulate_resize code: VALID but UNTESTED against correct model

**Fix applied:**
- Correct checkpoint stored permanently at:
  `experiments/results/v5_lagrangian_renderer/renderer_best.pt` (MD5: cff8dca4)
- `experiments/results/fridrich_renderer/renderer_best.pt` overwritten with correct
- Checkpoint sanity check added: `src/tac/checkpoint.py`
  - `verify_checkpoint_identity()` -- MD5-based, instant, no GPU needed
  - `verify_checkpoint_quality()` -- 2-pair PoseNet sanity check, < 5 seconds
- All experiment scripts now call `verify_checkpoint_identity()` before main loop:
  - `experiments/pair_difficulty_map.py`
  - `experiments/tto_step_curve.py`
  - `experiments/renderer_tto.py`
- `scripts/check_vastai.py` deploy/run commands verify before upload
- `src/tac/deploy/vastai/client.py` uses canonical checkpoint dir, verifies on upload
- INVALIDATED.md added to both step curve result directories

### Re-Validation Needed
1. **step_curve_v1** -- re-run on Vast.ai with correct checkpoint
2. **step_curve_cosine** -- re-run on Vast.ai with correct checkpoint
3. **tto_step_curve_hinge** -- never ran, needs correct checkpoint
4. **tto_v6_hinge_phase2** -- never ran, needs correct checkpoint

### Completed This Session (prior to checkpoint discovery)
- **Hinge loss** (P0): Implemented in tac, registered as experiment
- **Two-phase TTO** (P1): Implemented (100 steps PoseNet, then SegNet-only)
- **simulate_resize default**: Fixed (now True by default, matching auth eval)
- **Cosine LR**: Empirically worse for TTO (step_curve_cosine -- BUT INVALIDATED)
- **check_vastai.py**: Canonical Vast.ai interaction script
- **Pair difficulty map v2**: CORRECT (600 pairs, verified checkpoint)

## Pair Difficulty Map v2 Results (VALID)

600 pairs analyzed with correct checkpoint + simulate_resize=True:
- PoseNet MSE: mean=0.01726 (matches auth eval)
- SegNet disagree: mean=0.00215 (matches auth eval)
- Data enables adaptive TTO budget allocation

## Scores
- **Renderer baseline**: auth=0.87 (seg=0.21, pose=0.56, rate=0.10)
- **TTO v5a (gradient fix)**: auth=0.43 (first valid TTO with PoseNet gradients)
- **TTO v5b (embedding)**: auth=0.41 (10.8% improvement over v5a)
- **Target**: sub-0.20 auth

## Deadline
- May 3, 2026 (~18 days remaining)
