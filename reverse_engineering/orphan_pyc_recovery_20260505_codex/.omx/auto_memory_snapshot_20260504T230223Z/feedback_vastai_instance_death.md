---
name: Vast.ai instances die without warning — download checkpoints incrementally
description: Both 4090 instances (35480476, 35487297) exited mid-training. Best proxy 0.793 and 0.856 lost. Checkpoints NOT downloaded. ALWAYS download checkpoints incrementally during training.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
On 2026-04-23, both Vast.ai RTX 4090 instances exited during Phase 2 training.
No-Fridrich was at proxy 0.793 (epoch 1225). Fridrich was at 0.856 (epoch 1200).
Both checkpoints lost because we never downloaded them during training.

**Why:** Vast.ai spot instances can be reclaimed by the host at any time.
We had no incremental checkpoint download.

**How to apply — NON-NEGOTIABLE:**
- Set up a cron job or monitor that downloads the latest checkpoint every 30 min
- `scp root@host:/workspace/pact/experiments/results/*/distill_latest.pt .`
- Or use rsync with a local mirror of the results directory
- NEVER assume a Vast.ai instance will survive the full training run
- Consider `--checkpoint-every 100` instead of 200 for more frequent saves
- The `vastai show instances` status changes from "running" to "exited" without warning
