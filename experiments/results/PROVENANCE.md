# Experiment Results Provenance

Last updated: 2026-04-15

## Data Permanence Protocol

**Rule**: Never destroy a compute instance before downloading all results.
**Rule**: Every result file must have a metadata JSON or manifest recording source.
**Rule**: Check Modal volume first (free, highest quality), then regenerate on Vast.ai.

### Recovery Priority

1. **Modal volume** (`tac-asymmetric-results`): Contains 500-step TTO frames (highest quality).
   Download with: `python scripts/download_modal_tto_frames.py`
2. **Vast.ai instances**: Must download results BEFORE destroying.
   Download with: `python scripts/check_vastai.py download <id> <remote> <local>`
3. **Regenerate**: Use exact config from experiment registry (`src/tac/deploy/vastai/experiments.py`).

---

## Canonical Results (Current Best)

### TTO Frames -- Modal Volume (500-step, HIGHEST QUALITY)

| Name | Auth Score | Source | Status | Regenerable |
|------|-----------|--------|--------|-------------|
| `tto_frames/v5a_output_mse/tto_frames.pt` | 0.43 | Modal: `asym_v5_lagrangian_fixed/tto_v5a_output_mse/` | **DOWNLOAD PENDING** | Yes (500-step TTO on renderer_best.pt) |
| `tto_frames/v5b_embedding/tto_frames.pt` | 0.41 | Modal: `asym_v5_lagrangian_fixed/tto_v5b_embedding/` | **DOWNLOAD PENDING** | Yes (500-step TTO on renderer_best.pt) |

Expected shape: `(1200, 384, 512, 3)`, dtype: float32, ~708 MB each.

These frames represent the best TTO results achieved so far. The v5a frames
scored auth 0.43 -- the current best score. Download is urgent before Modal
access expires.

### Step Curve Results -- Vast.ai (COMPLETE, SAVED)

| File | Source | Status |
|------|--------|--------|
| `step_curve_v1/step_curve.json` | Vast.ai RTX 4090, 30 pairs, 8 step counts | Saved |
| `step_curve_cosine/step_curve.json` | Vast.ai RTX 4090, cosine LR schedule | Saved |

### Renderer Checkpoint -- Modal (SAVED)

| File | Source | Status |
|------|--------|--------|
| `fridrich_renderer/renderer_best.pt` | Modal A10G, dilated h=64 | Saved, canonical |
| `fridrich_renderer/config.json` | Training config | Saved |
| `fridrich_renderer/history.json` | Training loss history | Saved |

---

## Superseded / Historical Results

| File | Source | Notes |
|------|--------|-------|
| `renderer_tto_20260414T142644/tto_frames.pt` | Modal (early TTO, pre-gradient fix) | Superseded by v5a |
| `lightning_ep830_best_int8.pt` | Lightning AI T4 | Early checkpoint, before dilated arch |
| `asym_v3_longer_tight/renderer_best.pt` | Modal | Superseded by fridrich_renderer |
| `featmatch/`, `boundary/`, `vp_saliency/` | Local M5 Max | Postfilter experiments (not in submission path) |
| `posenet_sensitivity_v5/` | Vast.ai | Analysis artifact, not a trained model |
| `gradient_rank_analysis.json` | Local | Analysis artifact |

---

## LOST Data (Must Regenerate if Needed)

| What | Was On | Config | Priority |
|------|--------|--------|----------|
| 150-step TTO frames (Vast.ai session) | Vast.ai instance (destroyed) | `tto_step_curve` experiment | LOW -- 500-step Modal frames are strictly better |
| Distillation targets (12/60 batches) | Vast.ai Instance B (destroyed) | `tto_v1` with batch_pairs=10 | MEDIUM -- incomplete, restart on next instance |

The 150-step frames are not worth regenerating: the 500-step frames on Modal
are strictly better (32% SegNet improvement at 500 steps vs plateau at 150).

---

## Regeneration Protocol

To regenerate any TTO frames:

```bash
# 1. Check Modal first (free, has 500-step data)
python scripts/download_modal_tto_frames.py --list

# 2. If not on Modal, create Vast.ai instance
python scripts/check_vastai.py create --experiment tto_v1

# 3. Deploy code + checkpoint
python scripts/check_vastai.py deploy <id>

# 4. Run with EXACT config from registry
python scripts/check_vastai.py run <id> tto_v1

# 5. DOWNLOAD BEFORE DESTROYING
python scripts/check_vastai.py download <id> /workspace/experiments/results/ ./

# 6. Only then destroy
python scripts/check_vastai.py destroy <id>
```

All experiment configs are in `src/tac/deploy/vastai/experiments.py`.
The experiment registry is the single source of truth for reproducibility.
