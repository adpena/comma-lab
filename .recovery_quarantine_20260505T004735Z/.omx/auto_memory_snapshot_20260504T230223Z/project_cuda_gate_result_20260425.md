---
name: CUDA Gate Result 2026-04-25 — TRUE BASELINE 0.90
description: First contest-compliant CUDA measurement. Pinned archive (dilated h64 + CRF=50 + poses) scored 0.90 on CUDA A100 (vs 2.26 local MPS). Quantizr gap is 0.57. Sub-0.33 is reachable.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
GATING MEASUREMENT — Contest-compliant CUDA A100, 2026-04-25 21:00:

**Archive:** /tmp/archive.zip = renderer.bin (296,776 B dilated h64) + masks_crf50.mkv (421,054 B) + optimized_poses.bin (7,200 B)
**Path:** auth_eval_renderer.py on A100 cuda → upstream evaluate.py compatible scoring

**RESULTS:**
```
PoseNet distortion:     0.01069286
SegNet distortion:      0.00116135
Archive size:           686,321 bytes
Rate:                   0.01827972

Score breakdown:
  100 * seg            = 0.116
  sqrt(10 * pose)      = 0.327
  25 * rate            = 0.457
─────────────────────────────────
  FINAL SCORE:         0.901
```

**vs MPS local same-archive measurement:** 2.26 (PoseNet 0.245, SegNet 0.0024)

**Drift:**
- PoseNet: **23x worse on MPS** (0.245 → 0.011)
- SegNet: 2x worse on MPS (0.0024 → 0.0012)
- Score: 2.5x worse on MPS

**ALSO RAN: SHIRAZ standalone (renderer 99KB + same masks + same poses):**
- PoseNet: 0.918 — STALE poses for new renderer (poses optimized for old renderer)
- SegNet: 0.0147
- Score: 4.83
- Diagnosis: SHIRAZ needs its own pose TTO; using stale poses replicates conditioning-drift bug from CRF=63 e2e

**STRATEGIC IMPLICATIONS:**
1. We are 0.57 from Quantizr (0.33), not 1.93. Sub-Quantizr is reachable in 8 days.
2. PoseNet needs to drop from 0.011 → ~0.005 (50% improvement) to be competitive on distortion
3. Rate term contribution is 0.457 (51% of score) — mask CRF reduction is highest leverage IF renderer trained against new CRF
4. SegNet is already competitive (0.116 contribution, would need to drop to 0.10 = 14% improvement)
5. SHIRAZ has a small renderer but pose mismatch crippled it — re-TTO required

**NEXT STEPS the council mandates:**
1. Re-TTO SHIRAZ poses against SHIRAZ renderer (compress-time, free)
2. Train new renderer with mask augmentation (can ship CRF=63 then)
3. Wire radial zoom (hardcoded geometry, kills 4.6KB pose drift)
4. Stack the wins, run e2e CUDA, iterate

**ARCHIVE COMPONENTS for the team's record:**
- Pinned dilated h64 renderer: /Users/adpena/Projects/pact/submissions/robust_current/renderer.bin (296,776 B, ASYM v2)
- Pinned CRF=50 masks: /Users/adpena/Projects/pact/submissions/robust_current/masks_crf50.mkv (421,054 B)
- Pinned optimized poses: in current archive.zip + experiments/results/e2e_crf63_gating_20260425/current_archive_contents/
- SHIRAZ renderer (99KB): A100:/workspace/pact/experiments/results/shiraz/renderer_distilled.bin
- SHIRAZ all phase ckpts: A100:/workspace/pact/experiments/results/shiraz/distill_phase[1,2,3]*.pt
