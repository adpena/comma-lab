---
name: GPU Lane Literature — CLADE, Scorer-Aware Loss, Knowledge Distillation
description: Three high-impact techniques to implement: CLADE normalization, scorer-aware loss weighting, KD from teacher
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## Must Do (implement immediately)

### 1. CLADE normalization (per-class gamma/beta at every GroupNorm)
Source: CLADE (TPAMI 2021) — 60x lighter than SPADE, comparable quality
- Current: masks only enter at input via embedding
- Fix: inject per-class (gamma, beta) at EVERY GroupNorm layer
- Cost: ~50 extra params per layer (negligible)
- Prevents information washout through deep network

### 2. Scorer-aware loss weighting
Source: scoring formula analysis
- SegNet only evaluates LAST FRAME argmax → weight SegNet loss on frame_t1 only
- PoseNet evaluates PAIRS → weight PoseNet loss on both frames
- Earlier frames can be optimized purely for PoseNet with zero SegNet penalty
- This directly matches the scorer structure

### 3. INT4/codebook quantization
Source: torchao library, NVFP4 format
- Replace current INT8 (300KB) with codebook INT4 (150KB)
- torchao provides CodebookWeightOnlyConfig
- Halves model size with minimal quality loss

## Should Do (medium priority)

4. Per-class flow in MotionPredictor (road doesn't move, cars do)
5. Knowledge distillation: train 1-2M param teacher, distill to 300K student
6. QAT with FP4 fake-quant during training

## Key Citations for Paper
- CLADE (arxiv.org/abs/2012.04644)
- SPADE (arxiv.org/abs/1903.07291)
- Sandwiched Compression (arxiv.org/abs/2402.05887)
- Cool-Chic (arxiv.org/abs/2402.03179)
- HiFiC (hific.github.io)
- Instance-Adaptive Compression (arxiv.org/abs/2111.10302)

**Why:** These techniques have proven track records and directly apply to our renderer.
**How to apply:** CLADE first (30 min), scorer-aware loss second (15 min), INT4 third (1 hr).
