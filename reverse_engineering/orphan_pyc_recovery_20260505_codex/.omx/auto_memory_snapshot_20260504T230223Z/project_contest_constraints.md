---
name: Contest Constraints — Exact Pipeline and Rules (Non-Negotiable)
description: Complete contest evaluation pipeline, time budget, rules, and constraints verified 1:1 from upstream evaluate.sh and eval.yml
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Contest Evaluation Pipeline (from eval.yml and evaluate.sh)

### CI Job: `timeout-minutes: 30` (HARD LIMIT)

```
Step 1: checkout repo (includes models/, videos/)
Step 2: download archive.zip from submission URL
Step 3: install git-lfs, pull LFS files
Step 4: install uv, sync dependencies (cu128 group for T4)
Step 5: install ffmpeg
Step 6: TIMED — uv run bash evaluate.sh --device cuda --submission-dir ./submissions/NAME
```

### evaluate.sh Pipeline (within the 30 min):
```
1. unzip -o archive.zip -d archive/
2. bash inflate.sh archive/ inflated/ video_names.txt
3. verify all .raw files exist
4. python evaluate.py --submission-dir ... --device cuda
```

### Time Budget Breakdown:
- Total: 1800s (30 min)
- Fixed overhead (unzip + uv + verify): ~13s
- evaluate.py scoring (DALI + CUDA, 600 pairs): ~205s
- Safety margin: ~60s
- **Available for inflate: ~1522s (25.4 min)**
- Current renderer inflate: ~186s (12% utilization)

### Runner: `linux-nvidia-t4`
- GPU: T4 (16GB VRAM)
- RAM: 26GB
- CPU fallback runner: 4 cores, 16GB RAM

## Scoring Formula (from evaluate.py line 92):
```
score = 100 * segnet_dist + sqrt(10 * posenet_dist) + 25 * rate
```
- rate = archive.zip file size / sum of all files in videos/ directory
- Lower is better
- 600 pairs (1200 frames, non-overlapping consecutive pairs)

## Rules (from README):
1. **Archive rule**: External libraries OK, but large neural network artifacts must be in archive.zip and count toward compressed size. "This applies to the PoseNet and SegNet."
2. **Submission format**: PR with download link to archive.zip + inflate.sh
3. **Compression script**: Optional (not required in submission)
4. **No time limit on compression** — only inflation + evaluation has 30 min limit

## Our Canonical Interpretation (binding):
- Strict scorer rule: NO loading PoseNet or SegNet at inflate time (would require including ~73MB in archive, destroying rate)
- Masks pre-extracted at compress time, stored in archive as AV1 video
- Renderer loads only from archive (renderer.bin + masks.mkv)
- TTO is compress-time only (unlimited compute, results used as training data)

## Key Constants:
- Video: 0.mkv, 37,545,489 bytes, 1200 frames, 20fps, 60 seconds
- Output: 0.raw, RGB24, 874×1164×3×1200 = 3,050,352,000 bytes
- Camera: AR0231AT, fx=910px, pp=(582,437)@1164×874
- SegNet resolution: 384×512, 5 classes
- PoseNet: evaluates non-overlapping consecutive pairs (600 pairs)
