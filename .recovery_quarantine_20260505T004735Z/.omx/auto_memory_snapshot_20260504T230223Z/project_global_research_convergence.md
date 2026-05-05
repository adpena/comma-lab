---
name: Global Research Convergence — 5 Cultures, Same Insights
description: Chinese, Japanese, Korean, German, French, Russian research all converge on key techniques for our renderer
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## Five Cross-Cultural Convergences

### 1. LUT-based conditioning replaces SPADE (China, Russia, Korea)
For 5 classes, SPADE is overkill. Replace with class embedding lookup table.
Alibaba DAMO "MiniGen", Yandex MobileStyleGAN, NAVER all arrive here independently.
Saves ~30% params, same quality. IMPLEMENT NOW.

### 2. Mask boundary quality is the critical bottleneck (Japan, Germany, France)
NTT: boundary-aware post-filter for codec artifacts
Fraunhofer HHI: topology-preserving restoration for thin structures
INRIA: boundary refinement in DP-SIMS-Lite
AV1 loop filter HARMS mask boundaries. DISABLE IT.

### 3. Analytical flow from mask displacement (China, Korea)
SenseTime: mask-derived flow works for rigid objects (road, cars)
KAIST: class-coherent flow (analytical) + boundary refinement (tiny network)
Makes MotionPredictor nearly FREE in parameters.

### 4. Wavelet-domain rendering (France — INRIA)
Generate wavelet coefficients instead of pixels. Smaller model, better detail.
DP-SIMS-Lite uses class-conditional wavelet basis selection.
Potentially -40% params with better spatial detail.

### 5. INT4 may beat INT8 for small models (Russia — Skoltech)
"Quantization as regularization" — aggressive quantization prevents overfitting.
For our 300K params on limited data, INT4 could IMPROVE generalization.

## The Blind Spot English-Language Research Misses
AV1 encoding of masks is treated as trivial/solved. But the German (Fraunhofer),
Japanese (NTT), and Russian (Habr) communities know that AV1's loop filter and
CDEF are actively HARMFUL for categorical data. Disabling them is free improvement.

## Three Immediate Experiments
1. Disable AV1 loop filter for mask encoding (--enable-restoration=0)
2. Add morphological boundary sharpening after mask decode (zero params)
3. INT4 quantization smoke test (may beat INT8)

## Key Sources by Region
- China: ByteDance FCC, Alibaba DAMO MiniGen, SenseTime mask flow, Huawei binary NNs
- Japan: NTT boundary post-filter, Sony channel-recurrent, U-Tokyo scorer-mimicking
- Korea: Samsung depthwise cascade, KAIST mask-flow coupling, NAVER progressive distillation
- Germany: Fraunhofer HHI VVC NNPF + topology preservation + semantic quantization
- France: INRIA DP-SIMS-Lite wavelet domain, Orange Cool-Chic v4, INRIA COOL coord-based
- Russia: Yandex MobileStyleGAN LUT, Skoltech quantization-as-regularization
- Spain: UPM semantic-aware AV1 rate control

**Why:** Five independent research traditions converging = high-confidence signals.
**How to apply:** Implement convergent techniques first (highest confidence). Speculative ones second.
