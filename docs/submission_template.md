# Submission PR Template

Submit to: https://github.com/commaai/comma_video_compression_challenge/pulls

---

## PR Title

`adpena — neural renderer + pose-space TTO`

## PR Body

```markdown
## Submission: adpena

**Download:** [archive.zip](PLACEHOLDER_LINK)

**GPU required:** yes

**Compression script included:** yes

### Approach

Task-aware neural renderer trained against frozen PoseNet/SegNet scorers.
Architecture: asymmetric warp generator (288K params, FP4-quantized to ~170KB)
with per-pair FiLM conditioning vectors optimized at compress time.

The archive contains:
- `renderer.bin` — FP4-quantized asymmetric warp renderer
- `masks.mkv` — AV1-encoded semantic segmentation masks (pre-extracted)
- `optimized_poses.pt` — per-pair conditioning vectors (14.4KB)
- `config.env` — pipeline configuration

At inflate time, the renderer produces frame pairs from masks + poses in a
single forward pass (no scorers, no gradient computation, ~3 min on T4).

### report.txt

```
PLACEHOLDER — paste output of evaluate.py
```
```

---

## Checklist Before Submitting

- [ ] `inflate.sh` runs end-to-end on a fresh T4 instance
- [ ] Output is exactly 3,662,409,600 bytes (1164x874x3x1200)
- [ ] `archive.zip` contains ALL neural network weights used at inflate time
- [ ] No external downloads or network access during inflation
- [ ] Total inflate time < 30 minutes on T4
- [ ] `evaluate.py` produces the score reported in report.txt
- [ ] No modifications to upstream scorer code
- [ ] GPU requirement clearly stated
