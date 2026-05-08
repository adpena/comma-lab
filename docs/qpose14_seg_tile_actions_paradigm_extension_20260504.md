# qpose14 seg_tile_actions — sidechannel paradigm at the maximalist extreme (2026-05-04)

## Discovery

Audited `experiments/results/public_pr100_intake_20260504_codex/source/submissions/qpose14_r55_segactions_minp/inflate.py` (1016 LOC, the largest sibling submission). Found a NEW pattern worth cataloguing alongside PR100's sidecar and codex_metric_yshift's sidechannel.

## Layout

qpose14 is a **9-file compound archive**:

```
data_dir/
  model.pt.br             FP4 generator weights (Quantizr-family)
  mask.obu.br             AV1 monochrome mask video
  pose.npy.br             primary 600×6 pose vectors
  pose_q.br               quantized pose-delta supplementary stream
  p                       packed RPK1 container (multi-section)
  color_lut.npy.br        color lookup table
  actuator.npz.br         DCT-actuator coefficients (PR67-family)
  seg_tile_actions.br     per-frame per-tile categorical actions  ← NEW
  smooth_pose.npz.br      pose smoothing side-channel
```

## seg_tile_actions wire format

```python
# inflate.py:696-741
def load_seg_tile_actions_data(data, device):
    raw = brotli.decompress(data)
    records = []
    # 3 layouts auto-detected by length and magic:

    if raw.startswith(b"SG2"):  # SG2-versioned variable-length
        # cursor=3, then varint deltas + tile + action
        # action = raw[cursor]; cursor += 1
        # records.append((frame, tile, action))

    elif len(raw) % 4 == 0:  # 4-byte fixed: u16 frame + u8 tile + u8 action
        for i in range(0, len(raw), 4):
            records.append((u16_LE(raw[i:i+2]), raw[i+2], raw[i+3]))

    elif len(raw) % 5 == 0:  # 5-byte fixed: u16 frame + u16 tile + u8 action
        ...

    by_frame = {}
    for frame, tile, action in records:
        by_frame.setdefault(frame, []).append((tile, action))
    # action picks from seg_tile_action_specs() = list of (Y_delta, U_delta, V_delta) directions
```

The `seg_tile_action_specs()` function defines a fixed CODEBOOK of YUV perturbation directions (e.g. `(1.0, 1.0, 1.0)` = uniform brightness, `(1.0, 0.0, 0.0)` = Y-only, etc.). Each tile/frame pair selects ONE direction from the codebook to apply as a small additive correction.

## Position in the score-aware sidechannel paradigm

This extends the paradigm catalogued in `docs/score_aware_sidechannel_paradigm_20260504.md`:

| Granularity | Implementation | Correction count | Bytes/correction |
|---|---|---:|---:|
| PER-PAIR (1 dim) | PR100 hnerv_lc_v2 sidecar | ≤ 600 | 2 (u8 dim_idx + i8 delta) |
| PER-FRAME (channel) | codex_metric_yshift_av1 SC01 | 1200 | 1 (i8 × step) |
| PER-FRAME (low-rank pixel) | Lane SJ-KL Fisher residual | 1200×K coefs | varint |
| **PER-TILE PER-FRAME** | **qpose14 seg_tile_actions** | **>> 1200** | **3-5 (frame+tile+action)** |

qpose14 is the MAXIMALIST extreme — finer-grained than PR100 by O(N_tiles_per_frame). The trade-off:

- **Larger sidechannel size**: with 600 pairs × ~10-50 tiles/pair = 6,000-30,000 records × 4 bytes = 24-120 KB
- **Better distortion gain**: per-tile correction can address spatially-localized scorer responses that per-pair correction averages out
- **Categorical codebook (not continuous deltas)**: action picks from a fixed direction set, more compressible than free i8 deltas

## Implications

The qpose14 pattern is **3-bit search space PER tile** (~7-8 actions in the codebook) instead of PR100's 8-bit search (i8 = 256 deltas). This makes the brute-force search more tractable per-tile, but the tile count multiplies.

For our PR106-stacking work:
- **NOT directly portable**: PR106 exposes no separate tile, mask, or pose ZIP
  member; the verified HNeRV parser surface is decoder weights plus 28 global
  latent dims per pair, not a per-tile representation.
- **PORTABLE WITH ARCHITECTURE CHANGE**: a future PR106 variant that adds a per-tile residual would unlock this pattern. Out of scope for /loop-tick polish.
- **VALIDATES the paradigm**: 4 independent implementations of "small parameter-sparse correction sidechannels trained against the scorer" is now confirmed, spanning from per-pair (cheap) to per-tile (maximalist). This is the actual Shannon-floor mechanism.

## Decision: NO new lane

This is the second consecutive audit (after Quantizr) that finds an interesting pattern but doesn't translate to a PR106-stacking opportunity. The score-aware sidechannel paradigm extends to a 4th implementation (extending the catalog from 3 to 4); the immediate actionable lead remains the PR100 per-pair sidecar (already proposed as `lane_pr106_latent_sidecar` at L1).

## Cross-refs

- qpose14 inflate source: `experiments/results/public_pr100_intake_20260504_codex/source/submissions/qpose14_r55_segactions_minp/inflate.py:696-741`
- Sister memo (paradigm catalogue): `docs/score_aware_sidechannel_paradigm_20260504.md`
- Sister memo (Quantizr layout): `docs/quantizr_archive_layout_confirmation_20260504.md`
- Lead lane (PR100 sidecar port): `docs/pr100_latent_sidecar_porting_proposal_20260504.md` + lane registry `lane_pr106_latent_sidecar` (L1)
- CLAUDE.md PR67 actuator reference: `actuator.npz.br` is the PR67 DCT-actuator pattern, also documented
