---
name: Lane S motion.head shape mismatch on Lane A resume
description: 2026-04-28 Lane S overnight dispatch CRASHED at training launch. Lane A's renderer.bin has motion.head with 6-channel output (matching pose_dim=6 + use_zoom_flow=False); Lane S's self_compress_renderer_full profile builds a model with motion.head expecting 4 channels. Resume from Lane A fails with size_mismatch. Fix: align Lane S profile's motion.head config with Lane A's, OR add a profile override flag for motion_head_out_channels.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
**Bug observed 2026-04-28**:

Lane S (`scripts/remote_lane_s_self_compress.sh`) anchored on Lane A's
`renderer.bin` (296,776 bytes, ASYM FP32, base_ch=36 mid_ch=60
motion_hidden=32 embed_dim=6 depth=1 pose_dim=6 use_zoom_flow=False)
crashed at `model.load_state_dict(model_state)`:

```
RuntimeError: Error(s) in loading state_dict for AsymmetricPairGenerator:
  size mismatch for motion.head.weight:
    copying a param with shape torch.Size([6, 32, 3, 3]) from checkpoint,
    the shape in current model is torch.Size([4, 32, 3, 3]).
  size mismatch for motion.head.bias:
    copying a param with shape torch.Size([6]) from checkpoint,
    the shape in current model is torch.Size([4]).
```

Lane A renderer has 6-channel motion.head (matches pose_dim=6 used during
training); Lane S profile `self_compress_renderer_full` rebuilds the model
with 4-channel motion.head. The mismatch surfaces ONLY when resuming, not
during fresh training.

**Probable cause**:

The `motion.head` output channel count in `AsymmetricPairGenerator` is set
by some config that's different between Lane A and Lane S. Likely
candidates:
- `motion_out_channels` profile field
- Conditional logic in motion module: `2 + pose_dim` vs `2 + pose_dim_internal`
- `use_zoom_flow` toggle that adds zoom channel(s)

Since Lane A used `pose_dim=6, use_zoom_flow=False` and got 6-channel
head, the formula is likely `motion_out = pose_dim` for non-zoom-flow.
Lane S's profile may have set `pose_dim=4` somewhere or used a different
formula.

**How to fix**:

1. Check `src/tac/profiles.py::SELF_COMPRESS_RENDERER_FULL` — does it have
   `pose_dim=6` matching Lane A? If not, add it.
2. Check `src/tac/renderer.py::AsymmetricPairGenerator.__init__` —
   verify the motion.head out-channels formula handles all profile
   permutations consistently with what Lane A produced.
3. Add a regression test that asserts Lane A's checkpoint loads cleanly
   into the Lane S profile-built model BEFORE the remote script dispatches.

**How to apply (preflight)**:

Add a 33rd preflight check `check_resume_state_dict_shape_compat`:
- For every remote script with `--resume-from <path>`, statically verify
  the checkpoint's state_dict shapes match the profile-built model's
  shapes.
- Currently the failure happens AFTER 5+ minutes of mask extraction +
  scorer cache build — costing $0.05 per dead launch.

**Cost of this finding**: $0.10 (Lane S setup ran for ~12 min then crashed).

**Related memories**:
- `feedback_train_renderer_arch_drift` — earlier checkpoint shape mismatches
- `feedback_pipeline_pairgenerator_only` — similar deploy-time arch drift
- `feedback_canonical_remote_bootstraps` — preflight-before-dispatch pattern
