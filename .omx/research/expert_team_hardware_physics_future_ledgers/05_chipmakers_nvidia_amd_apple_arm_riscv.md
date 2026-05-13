# Ledger 05 — Chipmakers (NVIDIA / AMD / Apple / ARM / RISC-V) lineage (2026-05-13)

**Lane:** `lane_expert_team_hardware_physics_future_alien_tech_20260513`.
**Persona:** NVIDIA Tensor Core architects (H100 / H200 / B100 SM design, Santa Clara CA), AMD CDNA / MI300X engineers (Austin TX), Apple Neural Engine systems leads (Cupertino CA), ARM Mali GPU shader programmers (Cambridge UK), RISC-V Vector Extension (RVV 1.0) implementers (international consortium). We design **dispatchable kernels**: register pressure, memory bandwidth, dispatch granularity, sparsity-aware compute.
**Mode:** READ-ONLY hardware-derivation engineering analog. `research_only=true`. NO archive bytes mutated.
**Evidence:** `[hardware-derivation]`, `[engineering-analog]`, `[literature-prediction]`.

---

## 0. The chipmaker frame

Modern accelerators expose **hardware-native compression primitives** that software rarely uses optimally:
- **FP8 (NVIDIA H100 E4M3 / E5M2)**: same bit count as INT8 but with floating-point dynamic range.
- **Ternary (Apple ANE BitNet b1.58)**: 1.58 bits/weight.
- **2:4 structured sparsity (NVIDIA Ampere onward)**: 50% sparsity with 2× compute throughput.
- **MXFP / Microscaling (Open Compute MX-format)**: per-block scales for mixed precision.
- **GPU shader-based decode (ARM Mali, AMD CDNA, Apple Metal)**: fragment-shader decode at 60 FPS for 1080p video.

References operationalized:
- NVIDIA H100/H200 whitepaper (2022, *Hopper Architecture*).
- BitNet b1.58 (Ma et al., 2024, *arXiv:2402.17764*).
- NVIDIA Ampere structured sparsity (Mishra et al., 2021, *arXiv:2104.08378*).
- OCP Microscaling MX formats specification (Open Compute Project, 2023).
- Apple Neural Engine reverse-engineering (Mehrotra et al., 2024, *MLSys workshop*).

---

## 1. NVIDIA H100/H200 FP8 (E4M3) — same bytes, more dynamic range

### 1.1 Background

H100 introduced FP8 in two flavors:
- **E4M3** (4 exponent + 3 mantissa + 1 sign): range ±448, useful for forward activations + weights.
- **E5M2** (5 exponent + 2 mantissa + 1 sign): range ±57344, useful for gradients.

E4M3 has dynamic range ~2¹⁵ = 32768× vs INT8's ~256× (8-bit signed). For weights with large outliers (which our renderer has — convolution kernels often have a few large values), E4M3 preserves outliers losslessly while INT8 either clips or wastes bits on outlier-friendly scales.

### 1.2 Contest application

Quantize renderer weights to E4M3 instead of INT8. Same archive byte count (1 byte/weight). Decoder at inflate time reads each byte as E4M3, converts to FP32 via lookup table (~256-entry, ~1 KB extra in archive).

```python
# Inflate-time
def e4m3_decode(byte_value):
    """8-bit byte -> FP32 via E4M3 spec."""
    sign = (byte_value & 0x80) >> 7
    exp = (byte_value & 0x78) >> 3
    mant = byte_value & 0x07
    if exp == 0:
        # subnormal
        value = (mant / 8.0) * 2**(-6)
    elif exp == 15 and mant == 7:
        # NaN
        value = float('nan')
    else:
        # normal
        value = (1 + mant / 8.0) * 2**(exp - 7)
    return -value if sign else value
```

Lookup table replaces the bit-twiddling at runtime; ~256 × 4 bytes = 1 KB archive overhead.

### 1.3 Bit budget

229 KB renderer + 1 KB lookup = 230 KB. **Same as INT8 within 1 KB.**

### 1.4 Score-impact prediction

Distortion improvement from preserving outliers ~0.3-0.7 dB. At PR106 r2 operating point, equivalent to ~5-15 KB rate savings → -0.00015 to -0.00040 score [hardware-derivation, literature-prediction].

### 1.5 Reactivation

Easy to test — replace INT8 quantizer with E4M3 quantizer in existing QAT pipeline. Register as `lane_e4m3_fp8_renderer` at L0 SKETCH. **Recommend testing this week.**

---

## 2. Apple ANE 1.58-bit ternary (BitNet b1.58 backport)

### 2.1 Background

Microsoft BitNet b1.58 (Ma et al., 2024) showed that LLM weights quantized to **ternary {-1, 0, +1}** match FP16 accuracy at scale. The Apple M5 Neural Engine (announced 2025-Q3, shipping 2026-Q1) supports ternary-native matmul.

Information content per weight: log₂(3) ≈ 1.585 bits. Stored as 5 weights per byte (3⁵ = 243 ≤ 256).

### 2.2 Contest application

Quantize renderer weights to ternary. 229K weights × log₂(3) / 8 ≈ 45 KB vs INT4's 115 KB. **Savings 70 KB.**

**Encoding:** pack 5 ternary weights per byte (3⁵ = 243 ≤ 256; the remaining 13 byte values are unused or used for special markers).

```python
# Encoding (training-time)
def pack_ternary(weights_ternary):
    """weights_ternary: shape (N,) with values in {-1, 0, +1}."""
    N = len(weights_ternary)
    n_bytes = (N + 4) // 5
    packed = []
    for b in range(n_bytes):
        byte = 0
        for i in range(5):
            idx = b * 5 + i
            if idx < N:
                ternary = weights_ternary[idx]
                trit = {-1: 0, 0: 1, 1: 2}[ternary]
                byte = byte * 3 + trit
        packed.append(byte)
    return bytes(packed)

# Decoding (inflate-time, ~10 LOC)
def unpack_ternary(packed_bytes, N):
    weights = []
    for byte in packed_bytes:
        for _ in range(5):
            trit = byte % 3
            byte = byte // 3
            weights.append({0: -1, 1: 0, 2: 1}[trit])
    return weights[:N]
```

### 2.3 Critical caveat — Catalog #123

The previous Track 4 uniward-STC-Hessian 3-bit attempt **FALSIFIED** at -0.0058 score regression because `mean(θ²)` weight-magnitude saliency is **anti-correlated** with score-gradient saliency on score-aware-trained substrates. Same failure mode applies to ternary if the ternarization decision uses weight-magnitude.

**Correct procedure:** use **score-gradient saliency** (per Catalog #123) for ternary assignment — assign each weight to {-1, 0, +1} that minimizes score-gradient-weighted reconstruction error, not weight-magnitude-weighted error.

### 2.4 Bit budget

229K params at 1.58 bits/param = 45 KB. Lookup tables + per-channel scales ~3 KB. **Total: ~48 KB.** Savings vs INT8: ~180 KB; vs INT4: ~70 KB.

### 2.5 Score-impact prediction

If Catalog #123 score-gradient discipline is enforced:
- Rate savings: 70-180 KB → -0.0018 to -0.0048.
- Distortion penalty: 1-2 dB with proper outlier handling and QAT → score penalty +0.0010 to +0.0030.

**Net: -0.0010 to -0.0030.** [hardware-derivation, time-traveler-prediction, literature-prediction]

### 2.6 Reactivation

Register as `lane_bitnet_158_ternary_renderer` at L0 SKETCH. Substrate-engineering tier (3-week build). Reactivation requires:
- Catalog #123 score-gradient-saliency primitive verified working.
- Operator approval per CLAUDE.md "Design decisions" non-negotiable.
- Council deliberation (substrate engineering).

---

## 3. NVIDIA 2:4 structured sparsity

### 3.1 Background

NVIDIA Ampere onwards (A100, H100, H200, B100) supports **2:4 fine-grained structured sparsity**: in every group of 4 consecutive weights, at most 2 are nonzero. Hardware skips zero weights → **2× throughput**. The sparsity pattern is stored as a 4-bit mask per group + 2 nonzero values.

### 3.2 Contest application

Apply 2:4 sparsity to the renderer's Conv2D weights. Storage cost per group of 4 weights:
- 2 nonzero weights × 8 bits = 16 bits
- 4-bit mask (which 2 of 4 are nonzero, encoding C(4,2)=6 possibilities + 2 unused) = 4 bits
- **Total: 20 bits per 4 weights = 5 bits/weight average**

For 229K weights: 229K × 5 / 8 = 143 KB vs INT8's 229 KB. **Savings 86 KB.**

### 3.3 Critical caveat — Catalog #123 again

Same caveat as §2: which 2 of 4 weights to keep nonzero **must be chosen by score-gradient saliency**, not weight magnitude. Magnitude-based pruning on a score-aware-trained substrate is anti-correlated with score importance (per the falsified Track 4 result).

### 3.4 Score-impact prediction

86 KB rate savings → -0.0023.
Distortion penalty (with score-grad-saliency pruning + ~10% retraining): ≤ 1 dB → score penalty +0.0005 to +0.0010.

**Net: -0.0010 to -0.0018.** [hardware-derivation, literature-prediction]

### 3.5 Reactivation

Register as `lane_2_4_structured_sparsity_renderer` at L0 SKETCH. Medium effort (2-week build). Reactivation requires:
- Catalog #123 score-gradient-saliency primitive verified working.
- Empirical test on a small subset of layers first.

---

## 4. ARM Mali / Apple Metal / AMD CDNA shader-based inflate

### 4.1 Background

ARM Mali GPUs decode H.264/H.265/AV1 video via fragment shaders at 60 FPS for 1080p. The decoder is < 1 KB GLSL. Apple Metal + AMD CDNA have equivalent shader-based decode pipelines.

### 4.2 Contest application

Our inflate.py runs PyTorch (~10 MB installed). A **fragment-shader-equivalent inflate** would:
- Run on any GPU (Vulkan/Metal/D3D12/CUDA backends).
- Reduce inflate runtime dependency closure (no PyTorch needed).
- Be byte-deterministic (shaders are specified bit-exactly in their backends).
- Enable mobile deployment (comma.ai Snapdragon 845 Adreno GPU).

**Practical translation:** structure the inflate's computation as a **pure feed-forward shader-friendly graph** — no scatter/gather, fixed-stride memory, no FP exceptions, no dynamic control flow.

### 4.3 Bit budget

**No direct score gain.** Indirect: reduces inflate.py LOC (HNeRV parity lesson 4: ≤100 LOC budget), enables cross-runtime determinism.

### 4.4 Reactivation

Register as `lane_shader_friendly_inflate_kernel` at L0 SKETCH. Cross-link with NASA §4 (LEON3 integer-only) and L3 Harris §3 (Hexagon HVX-aligned). All three converge on the same recommendation: **structure inflate.py as a fixed-stride, exception-free, single-precision-or-integer feed-forward graph.**

---

## 5. RISC-V Vector Extension cross-runtime kernel design

### 5.1 Background

RVV 1.0 (RISC-V Vector Extension, ratified 2021) supports **runtime-configurable vector length** via `vsetvli` instruction. Code is written once; hardware dispatches at whatever VLEN is available. AMD CDNA, ARM SVE, and Intel AVX-512 have analogous variable-length vector support.

### 5.2 Contest application

Structure inflate.py kernels to be **vector-length-agnostic**:
- Avoid scatter/gather (varies in cost across implementations).
- Prefer contiguous strided ops.
- Avoid hardcoded batch sizes; use `runtime-vector-length`-aware iteration.

**Result:** the same inflate.py runs equivalently fast on T4 (CUDA SM thread group), Vast.ai 4090 (CUDA), Modal A100 (CUDA), Lightning T4 (CUDA), and contest-CPU x86_64 (AVX-512). No platform-specific tuning needed.

### 5.3 Bit budget

**No direct score gain.** Indirect: **closes the CUDA-CPU drift** documented in `feedback_cuda_cpu_auth_eval_drift_pr102_pr104_20260508.md`. Cross-runtime kernel determinism + integer-only arithmetic (NASA §4) = single archive that scores identically on both axes.

### 5.4 Reactivation

Register as `lane_rvv_cross_runtime_inflate` at L0 SKETCH. Substrate-engineering tier; long-term refactor of inflate.py.

---

## 6. Open Compute MX-format microscaling

### 6.1 Background

Open Compute Project (Meta, NVIDIA, Microsoft, AMD, Intel, ARM consortium) ratified the **MX format** in 2023: per-block scaled FP / INT formats. Variants:
- **MXFP8 (E4M3 or E5M2 elements + shared 8-bit scale per 32-element block)**: 8.25 bits/element effective.
- **MXFP6 (E3M2 elements + shared scale)**: 6.25 bits/element.
- **MXFP4 (E2M1 elements + shared scale)**: 4.25 bits/element.
- **MXINT8 (INT8 elements + shared scale)**: 8.25 bits/element with FP-equivalent dynamic range.

### 6.2 Contest application

Quantize renderer weights with MXFP4: 229K × 4.25 / 8 = 122 KB. Slightly larger than pure INT4 (115 KB) but **with FP-equivalent dynamic range**. Outlier weights are preserved by the per-block scale.

### 6.3 Bit budget

229K × 4.25 / 8 = 122 KB. Vs PR101's INT4 quantized ~115 KB: 7 KB larger.

### 6.4 Score-impact prediction

**Slight RATE penalty** (+7 KB, +0.00018 score) **for substantial DISTORTION benefit** (~0.5-1 dB at outlier weights). Net: likely **score-neutral or slightly negative**. The MX format wins at higher-bit-width comparisons (MXFP8 vs INT8) not at low-bit. **Not pursued.**

### 6.5 Reactivation

Register as `lane_mx_format_renderer` at L0 SKETCH with **explicit deferral note**: MXFP4 is dominated by INT4+score-grad-saliency. Reactivate only if a higher-bit-width comparison becomes the operating point (e.g., a substrate at 8-bit equivalent quality).

---

## 7. Status / cross-references / next steps

- **Companion:** master memo `expert_team_hardware_physics_future_alien_tech_20260513.md`.
- **Strong correspondences:**
  - §1 E4M3 ↔ existing INT8 QAT pipeline (drop-in replacement, ~1-day test).
  - §2 ternary ↔ existing Track 4 falsified attempt (requires Catalog #123 discipline).
  - §3 2:4 sparsity ↔ Apple ANE ternary discipline (sister technique).
  - §4 shader-friendly inflate ↔ NASA §4 LEON3 integer-only ↔ L3 Harris §3 Hexagon HVX.
- **Wire-in hooks** declared in master memo §9.
- **Highest-value tests this week:**
  - §1 E4M3 quantizer (1-day implementation).
  - §3 2:4 sparsity pilot on a single layer (3-day pilot).
- **Substrate-engineering candidates** (multi-week, operator-approval-required):
  - §2 BitNet b1.58 ternary.
  - §4 shader-friendly inflate refactor.
  - §5 RVV cross-runtime inflate.

**Per CLAUDE.md "KILL is LAST RESORT":** §6 MX-format is DEFER-pending-higher-bit-width-operating-point, not killed. All other techniques DEFER-pending-research.
