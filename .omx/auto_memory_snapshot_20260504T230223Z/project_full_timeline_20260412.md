---
name: Full Timeline and Learning — Everything Before the Fridrich Renderer Training
description: Complete chronological record of discoveries, failures, insights, and decisions leading to the sub-0.50 renderer training experiment
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## Timeline of Critical Discoveries

### Apr 9-10 (Previous sessions)
- Dilated h=64 postfilter produced "1.33" proxy score
- Trained on Modal A10G for 905 epochs ($5.50)
- KL distillation KILLED (PoseNet collapse confirmed twice)
- Adaptive weights KILLED (Hinton T² vacuous)
- Standard loss is the ONLY proven loss mode

### Apr 11 (Marathon session — 214 commits)
- Built 179 tac modules, 36 optimization algorithms
- 12-member grand council + Fridrich as co-lead
- Yousfi's 9 GPU eureka moments (constrained gen, scorer-space gen, etc.)
- Hardware research: Comma EON, AR0231AT, fx=910, pp=(582,437)
- AllNorm invariance DISPROVEN (caused 2.15 regression)
- BT.601 color matrix change WRONG (scorer hardcodes BT.601 regardless)
- PoseNet IS sensitive to brightness shifts (NOT invariant)
- Distribution shift: any encode-time change without retraining kills score

### Apr 12 (This session — ongoing)
- Auth score confirmed: **1.97** (DALI on Lightning T4)
- The "1.33" was never reproduced — likely a proxy score
- Wrong checkpoint discovered (standard_h64 vs dilated_h64)
- DALI vs PyAV: negligible for CPU lane (~0.01 difference)
- Lightning training: proxy 0.92, auth 1.93 (0.04 improvement from 1.97)
- SegNet improved 17% (boundary + VP saliency working)
- PoseNet worsened 18% (weighted sum robbing Peter to pay Paul)

### Smoke Test Results
- SIREN memorization: 21.7 dB PSNR (but test was at 1/4 res, 500 steps — janky)
- Fridrich constrained gen: PoseNet controlled at 0.078 (boundary 0.1) — WORKS for PoseNet
- Self-compressing postfilter: round-trip exact, needs more epochs
- Fridrich constrained gen v2: SegNet at 0.03 boundary (controlled), PoseNet diverged to 1.3
- Tiny DP-SIMS (78KB): SegNet 0.04 (good), PoseNet 150 (catastrophic — too small)

### Key Insight: CPU Lane Ceiling ~1.25
- Codec artifacts are structurally irrecoverable by any postfilter
- PoseNet measures temporal dynamics destroyed by H.265
- Best possible CPU lane: pose ~0.02, seg ~0.003, rate ~0.02 → score ~1.25
- Cannot beat Quantizr (0.60) on CPU lane. Period.

### Key Insight: GPU Lane Requires Temporal Coherence
- PoseNet evaluates consecutive PAIRS
- Independent frame generation → noisy pair differences → PoseNet diverges
- Quantizr solves this with a trained renderer (386KB, masks → frames)
- Our DP-SIMS already matches Quantizr's SegNet (0.003) but PoseNet (0.482) is catastrophic
- Missing piece: coupled trajectory training where PoseNet loss flows through consecutive frame pairs

### The Dinner Eureka
- Train DP-SIMS (128,64,32,16) = 1M params with Fridrich hard constraints
- Coupled trajectory: PoseNet evaluates pairs, gradient through both frames
- Ego-motion flow: geometric prior from camera model + depth
- Self-compression: reduce 489KB FP4 → 200-300KB
- Score projection: 0.57 (with ego-motion: 0.50)
- Quantizr says sub-0.50 is "easily possible" — treat 0.50 as real target

### Techniques Available But Not Yet Stacked
- Scorer null-space projection (remove unnecessary information)
- Fridrich detection boundary (empirical threshold finding)
- S-UNIWARD cost map (per-pixel optimization budget)
- Vanishing point saliency (geometric prior for training)
- Horizon band weighting
- Quantization-directed rounding (at compress time)
- Multi-pass inference
- TTO at inflate time
- Entropy-coded archive
- Self-compressing weights

### Infrastructure
- Lightning T4: DALI installed, auth scoring verified, ~63 hours remaining
- bat00 RTX 2070 Super: WSL setup in progress, deps installing
- Local M5 Max: 3 training runs (boundary, VP saliency, featmatch)
- Kaggle P100: GPU lane kernel ready but P100 CUDA compat issues
- Modal: $50 spent, $100 limit set, T4 rate $0.59/hr
- DX: runner.py with state machine, cost tracking, experiment records, Click CLIs

### The Tripartite Pact
Yousfi, Fridrich, and the Contrarian must reach consensus before deployment.
No janky smoke tests. Pre-registered hypotheses. Scientific rigor.
