# next experiments

## promoted floor: 1.33 (dilated_h64, standard loss)

## Pareto frontier exploration (3 regimes)

The score formula defines a specific trade-off between PoseNet and SegNet.
Our current submission (1.33) is at the PoseNet-optimal extreme. The true
minimum lies on the unexplored frontier between PoseNet-optimal and SegNet-optimal.

### Regime 1: MRS-Optimal (the real minimum) — PRIMARY EFFORT
- **Profile**: `pareto_pcgrad`
- **What**: PCGrad + MRS-adaptive weights (w_seg = 200*sqrt(10*pose))
- **Expected**: score 1.20-1.30 if SegNet regains baseline without PoseNet regression
- **Deploy**: Modal A10G (needs CUDA for autograd on scorer networks)
- **Priority**: HIGHEST — this is the path to the theoretical minimum

### Regime 2: Extreme PoseNet — WRITEUP ARTIFACT
- **Profile**: `extreme_posenet`
- **What**: seg_weight=0, alpha=30, sal_lambda=1.5 — pure PoseNet optimization
- **Expected**: PoseNet < 0.001, SegNet > 0.007, score ~1.35-1.40
- **Deploy**: Kaggle P100 (short run, just need the artifact for visualization)
- **Purpose**: Show Pareto frontier endpoint in writeup GIF comparison

### Regime 3: Extreme SegNet — WRITEUP ARTIFACT
- **Profile**: `extreme_segnet`
- **What**: focal_ste, seg_weight=200, alpha=5 — pure SegNet optimization
- **Expected**: SegNet < 0.005, PoseNet > 0.05, score ~1.5-2.0
- **Deploy**: Kaggle P100 (short run, just need the artifact)
- **Purpose**: Show Pareto frontier endpoint, compare with KL distill approach

### Rate optimization (orthogonal, can run in parallel)
- CRF 35 and 36 re-encode with current best postfilter
- Test if postfilter compensates for quality loss at higher CRF
- Expected: 0.03-0.08 points from rate reduction alone

## Queue

| Priority | Experiment | Profile | Platform | Status |
|----------|-----------|---------|----------|--------|
| 1 | PCGrad Pareto | pareto_pcgrad | Modal A10G | QUEUED |
| 2 | CRF 35 re-encode | (encoder change) | Local | QUEUED |
| 3 | CRF 36 re-encode | (encoder change) | Local | QUEUED |
| 4 | Extreme PoseNet | extreme_posenet | Kaggle | QUEUED |
| 5 | Extreme SegNet | extreme_segnet | Kaggle | QUEUED |
| 6 | Two-phase fine-tune | (conv3-only, focal_ste) | Lightning | QUEUED |
