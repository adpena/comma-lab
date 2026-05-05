---
name: Nuclear Run Readiness
description: Everything prepared for the final big training push — H100, precomputed data, multiple loss modes
type: project
---

All infrastructure ready for the "biggest and baddest" final run:
- `modal_nuclear_deploy.py`: H100, 12h timeout, resume-capable, configurable loss mode
- `precompute_for_modal.py`: 7.5GB precomputed data (frames + scorer weights)
- `precomputed/` directory ready to upload to Modal volume
- Pair-aware 6-channel architecture: built, tested, in VARIANTS dict
- KL distill loss: implemented, reviewed, bug-fixed
- Temperature annealing: implemented, smoke-tested

Recommended final config: h=96 + kl_distill + T=5→1 + 5000 epochs on H100.
Estimated cost: ~$54. Expected time: ~11 hours.

User approved spending money on Modal when justified. Plan to run in ~2 weeks.

**Why:** Maximize score before May 3 deadline.
**How to apply:** Validate locally first (50 epochs), then deploy nuclear.
