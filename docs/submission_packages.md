# Two-Submission Strategy

## Submission A: Contest-Compliant (Target: auth < 0.30)

**Philosophy:** Everything in the archive, single forward pass at inflate time.
No scorer weights, no gradient computation, no TTO at inflate time.

### inflate.sh
```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 inflate_renderer.py
```

### inflate_renderer.py (Contest-Compliant)
```python
# Inflate pipeline:
# 1. Load renderer + latent codes + postfilter from archive
# 2. Decode masks.mkv (AV1 5-class mask video)
# 3. For each pair: renderer(masks, pose, latent_code) → postfilter → upscale → write
# Total time: ~2-3 minutes (pure inference, no optimization)
```

### archive.zip contents
| File | Size | Purpose |
|------|------|---------|
| renderer_fp4.bin | ~143KB | FP4 quantized MaskRenderer |
| codebook.bin | ~2KB | Per-layer FP4 codebook + scales |
| latent_codes_int8.bin | ~10KB | 600×16 per-pair latent codes |
| projector.bin | ~0.5KB | Latent→feature projector weights |
| postfilter_int8.bin | ~45KB | Residual correction CNN |
| masks.mkv | ~30-40KB | AV1-encoded 5-class mask video |
| poses.pt | ~15KB | Pre-extracted GT pose vectors (600×6) |
| metadata.json | <1KB | Architecture config, version |
| **TOTAL** | **~250KB** | **Within budget** |

### Rate calculation
```
rate = len(archive.zip) / (1200 * 874 * 1164 * 3)
     ≈ 250000 / 3658243200
     ≈ 0.000068 (negligible)
```

### Expected score breakdown
| Component | Value | Notes |
|-----------|-------|-------|
| SegNet distortion | ~0.001-0.003 | Hinge loss training + postfilter |
| PoseNet distortion | ~0.01-0.05 | FiLM + latent codes + postfilter |
| Rate | ~0.0017 | 25 × rate ≈ 25 × 0.000068 |
| **Total** | **0.25-0.40** | 100×seg + sqrt(10×pose) + 25×rate |

### Kill criteria
- If auth > 0.50 after full pipeline: distillation quality insufficient
- If rate > 0.10: archive too large, reduce latent dim or skip postfilter
- If SegNet > 0.005: postfilter not converging, check hinge loss

---

## Submission B: Aggressive (Target: auth < 0.25)

**Philosophy:** Light TTO at inflate time on the hardest pairs only.
Uses scorer weights that are already present in the upstream evaluation environment.

### inflate.sh
```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ "${INFLATE_TTO:-0}" = "1" ]; then
    python3 inflate_renderer.py --tto-hard-pairs 30 --tto-steps 100
else
    python3 inflate_renderer.py
fi
```

### inflate_renderer.py (with TTO option)
```python
# Inflate pipeline:
# 1. Load renderer + latent codes + postfilter (same as Submission A)
# 2. Forward pass for all 600 pairs
# 3. If --tto-hard-pairs: identify hardest N pairs by postfilter residual magnitude
# 4. For hard pairs only: 100 steps of gradient descent on pixel values
#    - Freeze renderer, optimize the OUTPUT pixels directly
#    - Loss: hinge SegNet + PoseNet MSE (scorers from upstream/models/)
# 5. Upscale all frames → write
# Total time: ~8-12 minutes (forward + light TTO on 30 pairs)
```

### Compliance considerations
- Scorer weights ARE present in the evaluation environment (upstream/models/*.safetensors)
- Contest rules say: "inflate.sh must produce frames within 30 minutes"
- No rule explicitly prohibits using scorer weights at inflate time
- BUT: this may be considered "unsportsmanlike" if organizers rule against it
- **Decision required:** check with organizers before submitting this variant

### archive.zip (same as Submission A)
Same contents — the TTO uses no additional archive data. The scorer weights
come from the evaluation environment, not the archive.

### Expected score breakdown (with TTO)
| Component | Value | Notes |
|-----------|-------|-------|
| SegNet distortion | ~0.0005-0.002 | TTO polishes hardest pairs |
| PoseNet distortion | ~0.005-0.02 | TTO directly minimizes |
| Rate | ~0.0017 | Same archive |
| **Total** | **0.20-0.30** | Improvement from targeted TTO |

### Kill criteria
- If inflate time > 25 min: too many TTO pairs, reduce to 20
- If TTO doesn't improve hard pairs: early-stop at 50 steps
- If compliance ruling says NO: fall back to Submission A

---

## Implementation Timeline

| Day | Action | Deliverable |
|-----|--------|-------------|
| 1 | Distillation Phase 1-2 on Vast.ai | distill_base.pt |
| 2 | FP4 quantization + quality eval | renderer_fp4.bin |
| 3 | Latent code optimization (Exp 1) | latent_codes_int8.bin |
| 4 | Postfilter training (Exp 3) | postfilter_int8.bin |
| 5 | Integration + auth eval | Submission A score |
| 6 | TTO integration + timing test | Submission B score |
| 7 | Buffer / iterate on weakest component | Final submission |

---

## Decision Matrix

| Condition | Action |
|-----------|--------|
| Submission A < 0.30 and B < 0.25 | Submit B (aggressive), with A as backup |
| Submission A < 0.30 and B not significantly better | Submit A (safe) |
| Submission A > 0.40 | Latent codes or postfilter failing — debug |
| Submission A > 0.50 | Distillation failed — fall back to pure TTO submission |
| Compliance ruling against TTO at inflate | Submit A only |
