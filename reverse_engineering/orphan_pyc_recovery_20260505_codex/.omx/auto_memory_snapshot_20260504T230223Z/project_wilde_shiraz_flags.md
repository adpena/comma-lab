---
name: WILDE/SHIRAZ Training Flags
description: Two flags from Yousfi+Fridrich analysis of Phase 1 data. Must check when results return.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Training Runs (launched 2026-04-25)

**SHIRAZ** (94.253.160.53:31938, A100, $0.60/hr)
- Focal STE + curriculum + Fridrich, 1680 epochs
- TTO-optimized target frames (range [0, 184]) — pre-optimized to minimize scorer distortion
- Phase 1 plateau at loss 0.75 by epoch 275

**WILDE** (192.165.134.28:12806, A100, $0.61/hr)
- Hinge loss + freeze/unfreeze + error_boost + Fridrich, 1680 epochs
- Raw GT video frames as targets (range [0, 255]) — NOT TTO-optimized (upload failed, regenerated from video)
- Phase 1 loss 5.8 at epoch 75, converging slower

## Flags to Check When Results Return

**FLAG 1: WILDE GT targets vs SHIRAZ TTO targets**
- WILDE uses raw GT video frames, SHIRAZ uses TTO-optimized frames
- If WILDE ≈ SHIRAZ → GT frames sufficient, TTO preprocessing not essential
- If WILDE << SHIRAZ → TTO frames are essential, every future run needs them
- This is an unplanned but valuable A/B test

**FLAG 2: Phase 1 plateau → architecture ceiling**
- SHIRAZ plateaued at 0.75 pixel loss in Phase 1 (epoch 225→275 barely moved)
- 181K params with depth=1 may have hit representational ceiling for pixel reconstruction
- If Phase 2 SegNet loss < 0.002 in first 200 epochs → architecture is fine
- If Phase 2 SegNet loss plateaus above 0.005 → need more params (back to 288K)
- The scorer loss landscape is different from pixel loss — plateau in Phase 1 doesn't necessarily predict Phase 2

## Analysis Framework for Results

When checkpoints are available:
1. Download both best Phase 2 checkpoints
2. Build archives with float renderer (no QAT yet) — one WILDE, one SHIRAZ
3. Run `contest_eval.py` on BOTH → compare auth scores head-to-head
4. Check SegNet vs PoseNet breakdown — which profile won which scorer?
5. Run QAT on the winner → FP4 archive → contest_eval → final score
6. If both below 1.0, submit the lower one
7. If both above 1.0, analyze Phase 2 logs for what went wrong

## Instance Details

| Experiment | Instance ID | SSH | GPU | $/hr | Pipeline |
|------------|-------------|-----|-----|------|----------|
| SHIRAZ | 35562151 | 94.253.160.53:31938 | A100 SXM4 40GB | $0.62 | train→QAT→pose TTO→auth eval→bundle |
| WILDE | 35562150 | 192.165.134.28:12806 | A100 SXM4 40GB | $0.61 | train→QAT→pose TTO→auth eval→bundle |
| GREEN | 35565885 | TBD (loading) | A100 SXM4 40GB | $0.52 | train→QAT→pose TTO→auth eval→bundle |

All pipelines auto-chain: float training → QAT (50 INT8 + 250 FP4) → pose TTO (200 steps) → auth eval → tar bundle.

## Morning Checklist
```bash
# Check all three
for inst in "94.253.160.53:31938:shiraz" "192.165.134.28:12806:wilde" "GREEN_HOST:GREEN_PORT:green"; do
    IFS=: read host port name <<< "$inst"
    echo "=== $name ==="
    ssh -o StrictHostKeyChecking=no -p $port root@$host "tail -5 /workspace/pact/experiments/results/$name/train.log; echo '---'; tail -3 /workspace/pact/experiments/results/$name/pipeline.log 2>/dev/null || echo 'pipeline not started'" 2>&1 | grep -v "Welcome\|Have fun"
done

# Cost
.venv/bin/vastai show instances

# Download bundles (when ALL DONE appears in pipeline.log)
scp -P 31938 root@94.253.160.53:/workspace/pact/experiments/results/shiraz_bundle.tar.gz experiments/results/
scp -P 12806 root@192.165.134.28:/workspace/pact/experiments/results/wilde_bundle.tar.gz experiments/results/
# GREEN: scp -P PORT root@HOST:/workspace/pact/experiments/results/green_bundle.tar.gz experiments/results/

# DESTROY instances immediately after download
echo "y" | .venv/bin/vastai destroy instance 35562151
echo "y" | .venv/bin/vastai destroy instance 35562150
echo "y" | .venv/bin/vastai destroy instance 35565885
```
