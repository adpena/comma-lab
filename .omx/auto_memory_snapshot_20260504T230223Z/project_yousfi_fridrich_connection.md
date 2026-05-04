---
name: Yousfi was Fridrich's PhD student — challenge IS inverse steganalysis
description: Yassine Yousfi (challenge creator, Head of ML at comma.ai) did PhD with Jessica Fridrich at Binghamton DDE Lab on EfficientNet steganalysis. SegNet scorer IS a Fridrich-school steganalysis detector.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## The Fridrich-Yousfi Connection (confirmed 2026-04-21)

**Yassine Yousfi**: Head of ML at comma.ai. PhD at Binghamton DDE Lab under Fridrich.
- "Improving EfficientNet for JPEG Steganalysis" (IH&MMSec 2021) — surgical EfficientNet modifications
- "How to Pretrain for Steganalysis" (IH&MMSec 2021)
- Blog: yassineyousfi.github.io ("Overfitting Engineer")
- Self-describes as overfitting expert — the contest is about overfitting to one video

**Jessica Fridrich**: World's foremost steganalysis expert. Binghamton DDE Lab.
- No published "Fridrich warp" technique — that's our internal name
- Current work (2025): Transformers for Pooled Steganalysis, Acquisition Noise Outliers

**Implication**: The SegNet scorer (EfficientNet-B4 U-Net) was designed by someone who
spent their PhD making EfficientNet detect statistical anomalies. The challenge is literally
inverse steganalysis — generate frames that fool a Fridrich-school detector.

**How to apply**: Think about what makes frames "detectable" vs "natural" from a
steganalysis perspective. Class boundary precision, inter-frame consistency, and
feature-level statistics matter more than pixel-level PSNR.
