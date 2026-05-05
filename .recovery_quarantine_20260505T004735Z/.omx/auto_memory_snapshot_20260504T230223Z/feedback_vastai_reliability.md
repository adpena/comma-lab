---
name: Vast.ai Reliability Issues — Instances Die Mid-Experiment
description: Vast.ai instances can be killed mid-experiment (OOM, host reclaim, timeout). Always checkpoint. Download results frequently.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
Vast.ai instances are UNRELIABLE for long-running experiments. Discovered during v6 TTO:
- Instance died 3 times during a 60-batch experiment
- Batches 11, 29, and 43 caused crashes (hard pairs with high memory usage)
- The third crash destroyed the instance entirely (SSH refused, instance gone)
- All 43 checkpoints were LOST because results weren't downloaded incrementally

**How to apply:**
- ALWAYS use checkpoint-resume capable scripts (renderer_tto.py has this)
- Download results INCREMENTALLY — after every 10-20 batches, scp checkpoints locally
- For experiments >30 min, download partial results every 15 min
- Never assume an instance will survive the full experiment
- The hinge loss creates memory spikes on hard pairs — may need smaller batch_pairs (5 instead of 10)
- Consider splitting 60-batch experiments into 3×20 runs with intermediate downloads
- Track which batches crash and investigate the specific pairs (these are the hardest pairs — valuable data)

**Session lesson:** We validated hinge loss (25% better) and correct step curve, but lost the v6 TTO frames. The step curves are saved locally (committed). The v6 TTO must be regenerated.
