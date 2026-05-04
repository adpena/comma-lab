---
name: 19 known dead-resolver / dead-import violations as of 2026-04-27 R5
description: 2026-04-27: preflight_dead_resolvers scanner went live in warn-only mode and surfaced 19 real bugs (12 dead resolvers + 7 dead imports) across 6 files. These are the SAME class as the pose_dim / segnet_uncertainty_weighted_loss / uncertainty_loss_floor bugs the codex review caught — every profile that intends to set them is silently masked. Cleanup needed before flipping the scanner to strict.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
**Scanner committed in warn-only mode** so the 19 don't break the in-flight Lane A pipeline. Flip `preflight_dead_resolvers(strict=True, ...)` in `preflight_all` only after every violation below is fixed.

**Run to check current state:**
```
.venv/bin/python -c "from tac.preflight import preflight_dead_resolvers; preflight_dead_resolvers(strict=False, verbose=True)"
```

**Dead resolvers (12, all `src/tac/experiments/train_renderer.py`):**
Profiles set these at lines 341-457 in `src/tac/profiles.py` but `parse_args` never copies them into `args`, so `getattr(args, 'X', DEFAULT)` reads the WRONG default on every run.

| Line | Attr | Profile default |
|---|---|---|
| 1076, 1091, 1122, 1827 | `blend_mode` | "spatial" (NOT "scalar") |
| 1077, 1092, 1123, 1828 | `noise_mode` | "deterministic" |
| 1124, 1829 | `motion_type` | "depth_aware" (NOT "learned_cnn") |
| 1107 | `beta_start` | (diffusion teacher only) |
| 1108 | `beta_end` | (diffusion teacher only) |

**The fix:** add resolvers in parse_args mirroring the pose_dim fix (commit 0746a803 / 46e2ab6d). Pseudocode:
```
for key in ("blend_mode", "noise_mode", "motion_type", "beta_start", "beta_end"):
    if key in profile_dict and not getattr(args, key, None):
        setattr(args, key, profile_dict[key])
```

**Dead imports (7):**

| File:line | Bad import | Notes |
|---|---|---|
| `experiments/test_fp4_quality.py:73` | `extract_masks_from_video` from `tac.camera` | likely renamed |
| `experiments/test_fp4_quality.py:103` | `posenet_forward_pair` from `tac.scorer` | likely renamed |
| `experiments/test_fp4_quality.py:109` | `segnet_disagreement` from `tac.scorer` | likely renamed |
| `experiments/train_distill.py:544, 638` | `uniward_quant_noise_loss` from `tac.losses` | likely renamed; would crash if `cfg.use_variance_noise=True` |
| `src/tac/experiments/benchmark_mlx.py:29` | `_bilinear_upsample_2x` from `tac.mlx_renderer` | private name, likely renamed/removed |
| `src/tac/experiments/train_renderer.py:1554` | `luma_local_variance` from `tac.fridrich` | likely renamed; check fridrich.py for `compute_luma_local_variance` or similar |

**The fix per import:** grep the source module for the name; if it was renamed, update the import; if it was removed, delete the dead reference.

**Why this matters:**
- `blend_mode='scalar'` vs `'spatial'` — different rendering math. Profiles documented spatial blending; runs got scalar. Past "spatial blending didn't help" conclusions are INVALID for the same reason past "FiLM didn't help" was invalid.
- `motion_type='learned_cnn'` vs `'depth_aware'` — entirely different motion module. Same invalidation pattern.
- These resolver bugs are old; many trained checkpoints have the wrong arch. Loading them with the corrected resolver may produce shape mismatches.

**How to apply:**
1. Before promoting any new training result, sanity-check that the actual instantiated arch matches what the profile says.
2. Don't trust historical comparisons that depend on `blend_mode` / `motion_type` distinctions.
3. The cleanup is a single multi-resolver patch — should take ~30 min once Lane A's eval pipeline is unblocked.
4. After cleanup: flip `preflight_dead_resolvers` to strict in `preflight_all`. Add a regression test that asserts the scanner returns [] for the live train_renderer.py.

**Cost of NOT fixing:** every future `train_renderer.py` invocation reads the wrong defaults for blend_mode/motion_type/etc. — same bug class that wasted weeks before pose_dim was caught.
