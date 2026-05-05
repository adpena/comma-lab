# codex_metric_yshift LRL1 sidechannel audit — luma-only low-rank pixel residual (2026-05-04)

## Discovery

Audited the second sidechannel in `codex_metric_yshift_av1/inflate.py:625-678` (the LRL1 magic, separate from SC01 covered earlier). Found a NEW pattern — **luma-only low-rank pixel residual at downsampled resolution + bilinear upsample**.

This is **functionally equivalent to Lane SJ-KL** but with three key differences:
1. LUMA-only (Y channel correction, broadcast across RGB)
2. Low spatial resolution basis + upsample at apply time
3. SC01-family layout (single binary blob with magic + header + payload)

## LRL1 wire format

```python
LATENT_LUMA_HEADER = struct.Struct("<4sBBHHHff")
# magic(4) + components(1) + mode_id(1) + width(2) + height(2) + frame_count(2)
#         + coeff_step(4) + basis_step(4)  =  20 bytes total

# Then:
#   basis bytes = K × height × width int8 (each ×basis_step → real)
#   coeff bytes = frame_count × K int8 (each ×coeff_step → real)
```

## LRL1 apply logic

```python
def apply_latent_luma_rgb(frame, latent_luma, frame_idx):
    h, w, _ = frame.shape
    # Bilinear-upsample low-res basis to full frame resolution
    basis = F.interpolate(latent_luma["basis"].unsqueeze(0),
                          size=(h, w), mode="bilinear",
                          align_corners=False).squeeze(0)
    # Project per-frame coefficients onto basis
    coeffs = latent_luma["coeffs"][frame_idx]  # (K,)
    correction = torch.einsum("k,khw->hw", coeffs, basis)  # (H, W)
    # Add to all 3 channels (broadcast: same correction → R, G, B)
    return (frame.float() + correction.view(h, w, 1)).clamp(0, 255).round().to(torch.uint8)
```

## Size analysis

For typical contest geometry:
- low_h × low_w = e.g. 48 × 64 = 3,072 basis pixels
- K = number of components (1 byte field, so up to 255)
- frame_count = 1200

| K | Raw bytes | Brotli'd estimate |
|---:|---:|---:|
| 1 | 3,072 + 1,200 = 4,272 | ~1.5 KB |
| 2 | 6,144 + 2,400 = 8,544 | ~3 KB |
| 4 | 12,288 + 4,800 = 17,088 | ~5 KB |
| 8 | 24,576 + 9,600 = 34,176 | ~10 KB |

For the 0.0005-0.002 score Δ range typical of sidechannels, K=2-4 is the sweet spot (3-5 KB sidechannel cost).

## Position in the score-aware paradigm catalogue

This is **variant #6** in the paradigm:

| # | Implementation | Granularity | Mechanism |
|---:|---|---|---|
| 1 | PR100 hnerv_lc_v2 | per-pair (1 dim) | latent additive correction |
| 2 | codex_metric_yshift Y_SAT | per-frame (2 ch) | Y offset + saturation |
| 3 | codex_metric_yshift Y_SHIFT | per-frame (3 ch) | Y offset + (dy, dx) translation |
| 4 | Lane SJ-KL (in-house) | per-frame (K coefs) | RGB low-rank pixel residual |
| 5 | qpose14 seg_tile_actions | per-tile (codebook) | YUV codebook per tile |
| 6 | codex_metric LRL1 | per-frame (K coefs) | LUMA low-rank pixel residual + upsample |

The relationship to Lane SJ-KL is the most interesting: LRL1 is essentially a *lower-cost variant* of SJ-KL — same underlying mechanism (low-rank pixel basis + per-frame coefs), but operating on LUMA only (1/3 the channel work) at LOW RESOLUTION (much smaller basis).

## Comparison: LRL1 vs Lane SJ-KL

| Property | LRL1 | Lane SJ-KL |
|---|---|---|
| Channel coverage | Y only (broadcast to RGB) | Full RGB |
| Basis resolution | Low (e.g., 48×64) + bilinear upsample | Full (e.g., 384×512) |
| Coefficient quantization | int8 × per-tensor step | per-pair min/max/step |
| Basis source | Trained at compress time | Lanczos top-K Fisher eigenvectors |
| Sidechannel size (K=4) | ~5 KB | ~10-30 KB (full payload) |
| Application | Add to all RGB channels uniformly | Add to RGB independently |

LRL1 is the **lighter-weight sibling** of Lane SJ-KL. For a luma-dominated correction (which most scorer-relevant errors are), LRL1's Y-only restriction is a feature, not a limitation.

## Decision

**TERTIARY PROPOSAL** — defer to council gate. Below the latent-sidecar (lead) and Y-shift (secondary) lanes in priority. Reactivate IF latent-sidecar lands and Y-shift either lands or is discarded; LRL1 is the natural 3rd stack-on.

Predicted gain: -0.001 to -0.003 score Δ for K=2-4 sidechannel size 3-5 KB. Stacks orthogonally with both #1 (latent-space) and #3 (per-frame Y/translate) — operates on per-pixel luma additive correction (different stage, different error mode).

Not pre-registering as a separate lane yet — the catalogue is now fairly comprehensive (6 variants), and pre-registering all of them would dilute the registry. The two pre-registered lanes (lane_pr106_latent_sidecar + lane_pr106_yshift_sidechannel) are the immediate-action pipeline; LRL1 is the natural 3rd if the first two land empirically.

## Cross-refs

- LRL1 source: `experiments/results/public_pr100_intake_20260504_codex/source/submissions/codex_metric_yshift_av1/inflate.py:625-678`
- Lane SJ-KL (sister implementation): `src/tac/sjkl_basis.py`
- Sister memos in paradigm thread:
  - `docs/score_aware_sidechannel_paradigm_20260504.md` (catalogue master)
  - `docs/codex_metric_yshift_audit_20260504.md` (Y-shift = variant #3)
  - `docs/qpose14_seg_tile_actions_paradigm_extension_20260504.md` (variant #5)
  - `docs/pr100_latent_sidecar_porting_proposal_20260504.md` (variant #1, lead lane)
- Pre-registered lanes:
  - `lane_pr106_latent_sidecar` (L1, primary lead)
  - `lane_pr106_yshift_sidechannel` (L1, secondary lead)
