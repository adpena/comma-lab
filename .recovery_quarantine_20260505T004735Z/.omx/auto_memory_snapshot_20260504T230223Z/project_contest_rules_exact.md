---
name: Contest Rules — Exact Pipeline and Compliance Requirements
description: Exact eval pipeline, rate calculation, neural artifact rule, time limits, submission format
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## Rate Calculation
```python
rate = archive.zip file size / original uncompressed video size
```
archive.zip size = EVERYTHING inside the zip (video + neural weights + any other artifacts).

## Neural Artifact Rule
"External libraries won't count unless they use large artifacts (neural networks, etc.), in which case those artifacts should be included in the archive."
- postfilter_int8.pt (46KB) MUST be inside archive.zip
- Real baseline with weights: ~1.36, not 1.33

## Eval Pipeline (evaluate.sh)
1. archive.zip at submission_dir/archive.zip
2. inflate.sh at submission_dir/inflate.sh
3. Unzip: archive.zip → submission_dir/archive/
4. Call: `bash inflate.sh archive/ inflated/ video_names.txt`
5. Check: inflated/0.raw exists
6. Score: evaluate.py --submission-dir X --uncompressed-dir videos/

## Time Limits
- 30 minutes total for inflate
- CPU path: 4 cores, 16GB RAM
- GPU path: T4, 16GB VRAM, 26GB RAM

## Submission Format
- PR to commaai/comma_video_compression_challenge
- Download link to archive.zip
- inflate.sh (bash, entry point)
- Optional: compress.sh

## Output Format
- inflated/0.raw: raw RGB bytes, 1164×874×3 per frame, 1200 frames
- Total: 3,662,409,600 bytes

**How to apply:** runner.py must replicate evaluate.sh EXACTLY. Our inflate.sh must call inflate_postfilter.py with the 3 positional args the upstream evaluate.sh passes.
