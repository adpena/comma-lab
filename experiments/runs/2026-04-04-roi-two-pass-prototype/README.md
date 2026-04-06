# 2026-04-04 ROI two-pass prototype

Goal: test a tiny segmentation-guided proxy / fixed-ROI two-pass codec against the promoted 3.54 floor.

Design:
- full-frame base stream compressed more aggressively
- central/middle ROI stream compressed less aggressively
- decoder overlays the ROI onto the upscaled base
- honest because both streams are carried in the archive

This is still a proxy for a true segmentation-guided codec, not the final architecture.
