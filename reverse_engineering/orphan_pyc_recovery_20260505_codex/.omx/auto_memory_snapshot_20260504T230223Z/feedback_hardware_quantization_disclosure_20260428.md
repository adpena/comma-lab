---
name: Hardware-quantization disclosure required for ALL FP4 production paths
description: 2026-04-28 binding rule. Lane F lineage (V1=2.73, V2=1.79, V3=1.85) was simulated FakeQuantFP4 in FP32 — 4090 is CC 8.9, NVFP4 needs Blackwell CC 10.0. Hardware reality NEVER matched simulation, but no preflight/code surfaced this. Check 40 (`check_fp4_production_paths_disclose_hardware`) now STRICT — every production FakeQuantFP4 call needs `# FP4_HARDWARE_DISCLOSED:` or `assert_quantization_hardware_supported`.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## The rule

**Every production code path that instantiates `FakeQuantFP4(...)` or calls `fake_quant_fp4(...)` MUST disclose the hardware reality.** Acceptable disclosure forms (any one):

1. Runtime banner: `print("[SIMULATED-FP4] hardware capability < 10.0 — FP4 simulated via FakeQuantFP4")`
2. Inline comment: `# FP4_HARDWARE_DISCLOSED: <reason>` near the call
3. Programmatic gate: `assert_quantization_hardware_supported("fp4", device, allow_simulation=True)` from `tac.quantization`

## Why

Hardware reality:
- **NVFP4 hardware requires Blackwell (CC 10.0)** — RTX 5090, B100, B200, GB200
- **RTX 4090 is CC 8.9** — supports FP8 (e4m3, e5m2) via `torch.float8_e4m3fn` natively, NOT FP4
- **T4 (CC 7.5)** — supports INT8 hardware accel, NOT FP4 or FP8
- **A100 (CC 8.0)** — supports BF16, FP16, INT8, NOT FP4 or FP8

Lane F lineage spent multiple Vast.ai 4090 dispatches (V1=2.73, V2=1.79, V3=1.85, V4 in flight) running `FakeQuantFP4` simulation in FP32. The 20× PoseNet penalty we attributed to "FP4 architectural hostility" was unverifiable — could be simulation noise, not architectural. **Council made a strategic decision (`project_lane_f_v2_fp4_architectural_bottleneck_20260427`) on data that didn't reflect any real hardware.**

Cosmos deep-dive synthesis (memory `project_cosmos_deep_dive_addendum_20260428`) surfaced the rescue path:
- **FP8 IS hardware-supported on 4090** via `torchao.float8.convert_to_float8_training` OR native PyTorch 2.x `torch.float8_e4m3fn`
- Cosmos RL's `ROWWISE_WITH_GW_HP` recipe: forward + grad_input quantized to FP8, grad_weight kept BF16 — directly addresses PoseNet penalty
- Lane F-V5 (in flight) implements this rescue

## Storage vs inference distinction

The disclosure rule applies to **inference-quantization claims**, not **storage-quantization**.

- **Storage quantization (Lane S/W/Ω self-compress)**: stores weights at low bit-depth in the archive, dequantizes to FP16/FP32 at inference. Hardware-agnostic. NO disclosure required.
- **Inference quantization (Lane F FP4 QAT)**: claims to RUN inference at FP4 precision. Hardware-bound. Disclosure required.

The current `FakeQuantFP4` path produces FP4-bit-packed archives that get unpacked to FP32 at inference. So Lane F was actually doing storage quantization the whole time — but the framing was inference quantization, which obscured the hardware reality.

## Preflight Check 40

`check_fp4_production_paths_disclose_hardware` (in `src/tac/preflight.py`):
- Detects `FakeQuantFP4(...)`, `FakeQuantFP4.apply(...)`, `fake_quant_fp4(...)` instantiations in non-test .py files under `src/tac/` and `experiments/`
- Exempts `src/tac/quantization.py` (defines the simulation primitive itself) and `src/tac/renderer_export.py` (defines the FP4A archive format)
- Currently STRICT in `preflight_all` (3 violations were flagged + fixed: `src/tac/fp4_quantize.py`, `experiments/profile_fp4_layer_sensitivity.py`, `experiments/qat_finetune.py`)
- 9-test regression suite at `src/tac/tests/test_fp4_hardware_disclosure.py`

## What this catches

- Future Lane F-style attempts to claim "FP4 inference" without verifying hardware
- Subagent-generated code that uses FakeQuantFP4 without realizing it's simulated
- Council deliberations that conflate "we trained with FakeQuantFP4" with "we ran on hardware FP4"

## Cross-references
- `project_cosmos_deep_dive_addendum_20260428` — discovery + rescue path
- `project_lane_f_v2_fp4_architectural_bottleneck_20260427` — INVALIDATED conclusion (was simulation, not hardware)
- `feedback_compress_time_unlimited_archive_small_20260428` — orthogonal: storage-quant fine; inference-quant hardware-bound
- `project_hardware_geometry_chroma_full` — full hardware capability matrix (T4, 4090, A100)
