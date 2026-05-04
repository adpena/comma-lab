---
name: Forensic audit of 5 killed/exited Vast.ai lanes (V/V-V2/F-V4/D-V3/J-JBL) — engineering bugs, not paradigm failures
description: Skunkworks council audit 2026-04-28 evening. 4 lanes destroyed too aggressively; 1 exited unexpectedly. Categorized each as ENGINEERING / CONFIGURATION / HOST-SIDE / TRULY-BROKEN. Headline: half-frame failures (V/V-V2/D-V3) are NOT paradigm failures — Quantizr ships half-frame at 0.33. Failures are bugs in (1) channel broadcasting, (2) loss-mode validator allowlist, (3) joint warp-expansion training never properly implemented.
type: project
originSessionId: forensic-audit-20260428
---

## Council audit roster (rotating perspectives)

- **Yousfi**: domain expert. Did Quantizr's half-frame work because it's mathematically necessary, or is the rate gain an illusion?
- **Fridrich**: contest creator. Are we measuring against the right baseline (Lane G v3 1.05)?
- **Hotz**: engineering instinct. Find the BUG, not the philosophy.
- **Quantizr**: adversarial. What did the actual leader do that we DIDN'T?
- **Contrarian**: challenge BOLD claims. Did we kill these lanes with insufficient diagnostic data?

## Lane V (35781781) — Quantizr replica 88K halfframe joint-from-epoch-0

**Crash**: `RuntimeError: input[1, 1, 384, 512] expected 3 channels but got 1` at conv2d. Trained 7.6h then crashed at archive build OR at first scorer eval. ~$1.90 wasted.

**Diagnosis**: ENGINEERING BUG (fixable). The crash is downstream of `RadialZoomWarp.warp_inverse_masks(masks_t1, pair_idx)` which returns `(B, H, W)` integer mask (verified at radial_zoom.py:261). When this 1-channel mask flows into a code path that expects RGB-equivalent 3-channel input, the conv layer (probably the renderer's first stem conv at `nn.Conv2d(in_ch, hidden, ...)`) crashes.

**Specific code defect**: Lane V's profile sets `use_dsconv=True, base_ch=24, mid_ch=32`, and `use_zoom_flow=True`. The renderer's `MaskRenderer` path expects mask class indices that are EMBEDDED via `nn.Embedding(num_classes, embed_dim)` to produce embed_dim-channel features (line 762 in renderer.py). The crash signature `weight of size [32, 3, 3, 3], expected input[1, 1, 384, 512] to have 3 channels` indicates the input is going to a conv expecting 3 channels — this is NOT the embedding path. It's most likely the **HintedRenderer** alpha_map path at renderer.py:944: `alpha_map = torch.sigmoid(self.blend_conv(torch.cat([frame_t1_warped, frame_t1], dim=1)))` where blend_conv was constructed with in_channels=6 (3+3) but receives 2 (1+1) when masks are passed instead of frames. Need actual stack trace from `lane_v_results/train.log` to confirm.

**Proposed fix**: Add channel broadcasting at the warp-output boundary:
```python
# In train_renderer.py training loop after warp_inverse_masks:
mask_t = sim_zoom_warp.warp_inverse_masks(mask_t1, pair_idx_t)
# CHANNEL FIX: mask is (B, H, W) int; downstream expects (B, num_classes, H, W) one-hot
# OR (B, embed_dim, H, W) embedded — never raw 1-channel.
if mask_t.dim() == 3:
    mask_t = mask_t  # the embedding will handle channels — verify path
```

The TRUE fix likely lives in profile inheritance: `QUANTIZR_REPLICA_88K_HALFFRAME` inherits from a base where `use_dsconv=True` may dispatch to a different conv-construction path than `use_dsconv=False`. **Action**: instrument with an in-line `print(f"mask_t.shape={mask_t.shape} expected_in_ch={...}")` in the training loop and re-run with `--epochs 1` smoke.

**Council vote**: 4/5 fixable (Yousfi/Fridrich/Hotz/Quantizr). Contrarian says "kill it, the channel-mismatch class signals deeper coupling — Lane H-V3 is the cleaner rebuild."

## Lane V-V2 (35782055) — annealed half-frame

**Status reported**: "claimed fix but half-frame broken". 

**Diagnosis**: ENGINEERING BUG (same root cause as Lane V). The annealing schedule (`mask_half_sim_prob_anneal: {start_value=0, end_value=1, ramp_start_frac=0.30, ramp_end_frac=0.70}`) is well-formed — verified by reading profiles.py:3007-3023. Profile inherits from QUANTIZR_REPLICA_88K_HALFFRAME so the channel bug propagates verbatim.

**Proposed fix**: Same as Lane V. **Composite fix candidate**: Lane H-V3 supersedes both — built on Lane G v3 anchor (288K renderer, NOT the 88K Quantizr-replica) which has a known-working channel pipeline.

**Council vote**: Subsume into Lane H-V3 design. Annealing schedule is sound and should be reused.

## Lane F-V4 (35782082) — mixed-precision FP4 on Lane A anchor

**Status reported**: "Lane F chapter closed". 

**Diagnosis**: TRULY BROKEN (paradigm-level, not engineering). Per memory `feedback_hardware_quantization_disclosure_20260428` + `project_cosmos_deep_dive_addendum_20260428`: NVFP4 requires Blackwell CC 10.0; RTX 4090 is CC 8.9. ALL Lane F runs (V1=2.73, V2=1.79, V3=1.85, V4=predicted [1.20, 1.50]) were SIMULATED FakeQuantFP4 in FP32 — there is no hardware FP4 on the 4090. Lane F lineage spent $15+ on simulated FP4 with no hardware backing. Lane F-V5 (hardware FP8 via torchao.float8) is the proper rescue; Lane F-V4 was the last gasp.

**Proposed fix**: NONE for V4 line. Pivot to Lane F-V5 (FP8) which is supported on Ada/Lovelace+ (CC >= 8.9). Predicted band [0.95, 1.20] LOWEST RISK. Per `project_cosmos_deep_dive_addendum_20260428`.

**Council vote**: 5/5 close V4 line. Promote Lane F-V5 design which already exists in council notes.

## Lane D-V3 (35782081) — dilated-h64 annealed half-frame + KL fix

**Status reported**: "half-frame broken" — but this is the FIRST attempt at COMBINING the V2 LR fix with the post-bugfix KL weight (0.002 vs 1.0) AND annealed mask_half_sim_prob.

**Diagnosis**: CONFIGURATION BUG (subtle). Lane D-V3 inherits from `DILATED_H64_HALF_FRAME` which sets `mask_half_sim_prob=0.5` static. The V3 override adds `mask_half_sim_prob_anneal: {start=0, end=0.5, ramp=0.30→0.70}`. The training loop reads `current_mask_half_sim_prob` from `mask_half_sim_prob_for_epoch()` (verified at train_renderer.py:2559). 

But the ENDPOINT is 0.5, not 1.0. Quantizr ships ALWAYS-on warp at inflate (every even frame is a warped recon). Training at endpoint=0.5 means HALF the training batches see warped masks; the other half see SegNet-extracted. The trained renderer is a HYBRID — at inflate-time it sees 100% warped masks (since the archive only stores odd frames). **Distribution mismatch between train (50%) and test (100%)**. This is the same class as Lane M-V2 BUG-1 (train/inference mismatch — see `feedback_check_42_train_inference_parity_20260428`). 

**Proposed fix**: Set `end_value=1.0` in the anneal schedule (matching Lane V-V2) AND set static `mask_half_sim_prob=1.0`. The diff:
```python
DILATED_H64_HALF_FRAME_V3_ANNEALED_KLDISTILL = {
    **DILATED_H64_HALF_FRAME,
    "mask_half_sim_prob": 1.0,  # was 0.5 — match inflate-time distribution
    "mask_half_sim_prob_anneal": {
        "start_value": 0.0,
        "end_value": 1.0,  # was 0.5
        "ramp_start_frac": 0.30,
        "ramp_end_frac": 0.70,
    },
    "kl_distill_weight": 0.002,
}
```

**Council vote**: 5/5 fixable. The distribution-mismatch insight applies broadly — should also be a Check 42 extension.

## Lane J-JBL (35781543) — Jaccard + Boundary Label Smoothing

**Status reported**: EXITED unexpectedly despite gate-clearance.

**Diagnosis**: CONFIGURATION BUG (the loss-mode validator). Lane J-JBL profile (`J_JBL_DILATED_H64`) inherits from `DILATED_H64_HALF_FRAME_V3_ANNEALED_KLDISTILL` and overrides `loss_mode="jbl"`. The validator at train_renderer.py:888-897 ALLOWS "jbl". But J_JBL_DILATED_H64 ALSO inherits other knobs from DILATED_H64_HALF_FRAME which sets `loss_mode="focal_ste"` (per the comment at train_renderer.py:884 "every Lane D / Lane G v3 / Lane V / Lane MAE-V profile (all of which inherit loss_mode='focal_ste'"). The override should win, but Python dict-spread order matters: `**DILATED_H64_HALF_FRAME_V3_ANNEALED_KLDISTILL,` THEN `"loss_mode": "jbl",` — that should resolve correctly. 

The actual EXIT is more likely: `combined_jbl_distill_loss` not yet implemented in `tac.losses_jbl` (verified file exists at 13.4K), OR train_renderer.py's loss dispatch never wires up `loss_mode == "jbl"` to call it (the comment says "Lane J-JBL flag" was added "this commit" — may be incomplete).

**Proposed fix**: 
1. Verify `tac.losses_jbl.combined_jbl_distill_loss` exists and is called from train_renderer.py when `args.loss_mode == "jbl"`. Grep:
```bash
grep -rn "combined_jbl_distill\|jaccard_metric_loss\|loss_mode == .jbl" src/tac/
```
2. If missing: stub the function to call `kl_distill_segnet_only` (graceful degradation) AND log `[lane-j-jbl] WARN: loss_mode=jbl not yet wired, falling back to KL distill` so the lane completes with a comparable (not improved) score.
3. Add Check 49: profile loss_mode values must be in train_renderer.py validator allowlist (see Deliverable 3).

**Council vote**: 5/5 fixable. The "EXITED unexpectedly with gate-clearance" pattern is specifically what Check 49 will prevent.

## Top-2 most critical findings

1. **Lane D-V3 distribution mismatch (train=0.5, inflate=1.0) is the same bug class as Lane M-V2 BUG-1.** The fix is one-line in the profile: set `end_value=1.0`. This bug class deserves a Check 42 extension specifically for `mask_half_sim_prob` schedules. The cost of NOT catching: every half-frame retrofit on dilated-h64 has been mismatched.

2. **Lane V channel mismatch is suspicious of the HintedRenderer path.** The `[32, 3, 3, 3]` weight signature points to a conv expecting 3 channels in the alpha_map blending OR the residual path. Lane V uses `use_dsconv=True` which goes through a different conv-construction path than the dilated-h64 baseline. Without the actual stack trace, hypothesis confidence is 60%; need `lane_v_results/train.log` retrieved from the destroyed instance OR a local repro at `--epochs 1`.

## Disposition

| Lane | Verdict | Action |
|------|---------|--------|
| Lane V | ENGINEERING BUG | Subsume into Lane H-V3 (cleaner rebuild on Lane G v3 anchor) |
| Lane V-V2 | ENGINEERING BUG | Subsume into Lane H-V3 |
| Lane F-V4 | TRULY BROKEN | Close. Pivot to Lane F-V5 (hardware FP8) |
| Lane D-V3 | CONFIGURATION BUG | Re-launch with `end_value=1.0` after Lane H-V3 lands |
| Lane J-JBL | CONFIGURATION BUG | Wire up `combined_jbl_distill_loss` OR fallback + Check 49 |

## Cross-references

- `feedback_half_frame_breaks_posenet` (UPDATED 2026-04-28 with joint-training note)
- `project_lane_v_crashed_channel_mismatch_20260428`
- `project_cosmos_deep_dive_addendum_20260428` (Lane F hardware truth)
- `feedback_check_42_train_inference_parity_20260428` (same bug class as Lane D-V3)
- `project_lane_h_v3_revival_design_20260428` (the half-frame revival)
- `project_lane_g_v3_landed_1_05_20260428` (current frontier anchor)
