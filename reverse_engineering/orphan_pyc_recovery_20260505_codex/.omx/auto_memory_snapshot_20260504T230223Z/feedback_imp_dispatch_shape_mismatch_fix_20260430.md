---
name: Lane 17 IMP dispatch — shape mismatch + export API fixes for Lightning Studio
description: 2026-04-30 ~21:30 UTC. Two bugs fixed in scripts/remote_lane_j_imp_iterative_magnitude_pruning.sh to enable Lightning Studio dispatch. Both bugs reproducibly killed Stage 1.5 per-cycle CUDA auth eval at cycle 0. Bug 1: build_renderer(use_zoom_flow=False, pose_dim=6) produces motion.head=[2,32,3,3] but Lane G v3 ASYM anchor saves [6,32,3,3] — shape mismatch on load_state_dict. Bug 2: export_asymmetric_checkpoint_fp4 signature requires positional output_path arg, but script called f.write(export_fn(m)) treating it as bytes-returning. Both fixes applied directly via SSH on Lightning. Need to backport to local repo + commit.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Bug 1: motion.head shape mismatch on auth-smoke rebuild

### Symptom
```
size mismatch for motion.head.weight: copying a param with shape torch.Size([6, 32, 3, 3]) from checkpoint, the shape in current model is torch.Size([2, 32, 3, 3]).
```

### Root cause
- Lane G v3 ASYM anchor (saved by `load_asymmetric_checkpoint`): motion.head=[6, 32, 3, 3]
- `build_renderer(use_zoom_flow=False, pose_dim=6, ...)` produces a legacy `PairGenerator` with motion.head=[2, 32, 3, 3]
- `pose_dim` does NOT control motion.head output channels in the legacy PairGenerator code path
- The 6/4 channel logic at `src/tac/renderer.py:1149` (`motion_output_channels = 4 if use_zoom_flow else 6`) is in `AsymmetricPairGenerator`, NOT in the legacy `PairGenerator`
- `build_renderer(use_zoom_flow=False)` returns the legacy `PairGenerator` with hard-coded `output_channels=2`

### Fix
Replace `build_renderer(...)` calls with `load_asymmetric_checkpoint(ANCHOR_RENDERER)` to get the architecture FROM the binary header (which has motion.head=[6,...]), then overlay the IMP cycle's pruned state_dict.

### Files patched (Lightning Studio /home/zeus/pact)
- `scripts/remote_lane_j_imp_iterative_magnitude_pruning.sh` — line 258 (auth-smoke build) + line 386 (Stage 4 final FP4A export)

### TODO: backport to local /Users/adpena/Projects/pact via subagent_commit_serializer

## Bug 2: export_asymmetric_checkpoint_fp4 signature changed

### Symptom
```
TypeError: export_asymmetric_checkpoint_fp4() missing 1 required positional argument: 'output_path'
```

### Root cause
Function signature is now:
```python
export_asymmetric_checkpoint_fp4(model, output_path, block_size=32, codebook_name='default', robust_scale=False) -> int
```
Returns int (bytes written), NOT bytes. Called as `f.write(export_fn(m))` previously assumed returning bytes.

### Fix
- Old: `with open(path, 'wb') as f: f.write(export_asymmetric_checkpoint_fp4(m))`
- New: `export_asymmetric_checkpoint_fp4(m, path)` (writes directly)
- For ASYM export, similar: `export_asymmetric_checkpoint(model, output_path)` returns int

### Backport TODO
Local script also needs same fix. Check other lane scripts for same pattern.

## Verification (post-patch)

```
[lane-j-imp] saved cycle 0 artifacts → /home/zeus/pact/lane_j_imp_results/cycle_0
[lane-j-imp] DONE in 3.5s
[lane-j-imp] cycle 0 complete: sparsity=0.200026
[lane-j-imp] Stage 1.5: per-cycle CUDA auth eval (cycle 0)
[fp4-export] 288,363 params → 169,993 bytes (4.72 bits/param)
GPU: NVIDIA L40S 100% util / 12GB used
```

Lane 17 IMP cycle 0 successfully exported to FP4A on L40S. Stage 1.5 contest-CUDA auth eval running.

## Cross-refs

- feedback_lightning_ai_ssh_credentials_20260430.md (SSH details)
- feedback_vastai_spot_unreliable_pivot_to_modal_lightning_20260430.md (why we moved IMP to Lightning)
- src/tac/renderer.py:1149 (use_zoom_flow → motion_output_channels mapping)
- src/tac/renderer_export.py:931 (`_MAGIC = b"ASYM"` — load_asymmetric_checkpoint)

## Lessons

- **NEVER use `build_renderer(...)` to reconstruct an arch for state-dict loading from a saved checkpoint.** Always use `load_asymmetric_checkpoint(ANCHOR_BIN)` which reads arch from header.
- API signature drift: `export_*_checkpoint*` family changed from "return bytes" → "write to path + return int". Audit all call sites.
- Lightning Studio L40S has 100% util on Lane G v3 architecture cycle 0 — confirms IMP fits + runs on L40S (no need to switch to H100 for cycle 0; may upgrade for cycle 4-9 if memory pressure).
