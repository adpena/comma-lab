---
name: Money Left on the Table
description: All implemented-but-unused and approved-but-unbuilt features as of 2026-04-10 evening
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## Implemented, Never Deployed
1. **LSQ learned quantization** — apply_lsq() in quantization.py, never called from Trainer
2. **SWA** — fixed crash bug, council_v2_adaptive has use_swa=True, no run past ep 2000 yet
3. **Multi-pass inflate** — INFLATE_MULTI_PASS=2 env var ready, never tested for score
4. **PSD + KL distill** — was 1.4018 at ep 64 (FASTER than dilated!), killed by SIGTERM, relaunched
5. **Error replay** — in council_v2_adaptive, adaptive run at ep 38
6. **Proxy correction factors** — implemented, auth eval running for calibration

## Council-Approved, Not Built
7. **Adaptive hard_frame_ratio schedule** — ramp 0.1→0.5 over training
8. **`tac monitor` CLI** — reads telemetry JSONL, live-updating table
9. **Gradient norm telemetry** — log per-component gradient magnitudes

## Key Finding
PSD architecture (PixelShuffle+Dilated, 24x24 RF) + KL distill converged FASTER
than dilated (15x15 RF) + KL distill: scorer 1.4018 at ep 64 vs 1.44. The larger
RF aligns better with SegNet's 512x384 operating resolution.

**PSD + KL distill is relaunched and resuming from ep 69.**
