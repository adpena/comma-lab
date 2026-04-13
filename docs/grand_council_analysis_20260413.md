# Grand Council Analysis — 2026-04-13
## Skunkworks Tripartite Pact: Full State Assessment + Battle Plan

---

## Ground Truth

| Metric | Value | Notes |
|--------|-------|-------|
| **True postfilter score** | ~1.96 | PyAV proxy, reproducible. The 1.33 was an artifact. |
| **Renderer best proxy** | 1.88 | ep4000, (36,60) on Modal T4, 1.8h training |
| **Quantizr (target)** | 0.60 | PR #53, mask2mask paradigm |
| **Days remaining** | 21 | Deadline May 3, 2026 |

## Modal Training Results — First Asymmetric Warp Run

### Run 1: (36,60) medium — COMPLETED + AUTO-KILLED
- **Config**: base_ch=36, mid_ch=60, 287,019 params, FP4 ~150KB
- **GPU**: T4 ($0.59/hr), 1.8 hours wall-clock
- **Epochs**: 4418 total, auto-killed at ep4418, best at ep4000
- **Phase transition**: Phase 1 → Phase 2 at epoch 4000 (40% of 10000)

### Eval Scores (full evaluation):
| Epoch | SegNet | PoseNet | Rate | Score |
|-------|--------|---------|------|-------|
| 3600 | 0.00601 | 0.14947 | 0.003822 | **1.9187** |
| 3800 | 0.00639 | 0.15357 | 0.003822 | 1.9740 |
| 4000 | 0.00624 | 0.13531 | 0.003822 | **1.8828** (BEST) |
| 4200 | 0.01252 | 0.18953 | 0.003822 | 2.7239 (diverged) |
| 4400 | 0.00849 | 0.16062 | 0.003822 | 2.2123 (diverged) |

### Score Decomposition at Best (ep4000):
```
SegNet:  100 × 0.00624 = 0.624  (EXCELLENT — matches postfilter)
PoseNet: sqrt(10 × 0.135) = 1.163 (BAD — 2.4x worse than postfilter)
Rate:    25 × 0.00382  = 0.096  (EXCELLENT — 6x better than postfilter)
Total:                   1.883
```

### Run 2: (24,40) moonshot — ABORTED
Client disconnected after 1 epoch (forgot `--detach` flag).

---

## Root Cause Analysis

### Why training diverged (the Lagrangian explosion):
rho_growth=1.02 compounded: 10 × 1.02^400 = catastrophic. λ_s reached 42,182, λ_p reached 184,738. The penalty overwhelmed the actual task loss. The model couldn't reduce PoseNet fast enough to keep up with exponentially growing penalties.

### Why flow is dead (near-zero magnitude throughout training):
Flow started at 0.008 and never developed. The residual pathway learned to produce corrections during Phase 1 (MSE-dominated). By Phase 2, residual was entrenched and flow was dead. The asymmetric architecture's key advantage — temporal warping — was never activated.

### Why SegNet works but PoseNet doesn't:
SegNet evaluates semantic class labels. The renderer generates from masks, preserving semantics by construction. PoseNet evaluates geometric/pose accuracy which requires precise pixel-level fidelity that the renderer hasn't learned yet.

---

## Council Cross-Examination

### Yousfi (Contest Architect) on the flow problem:
"gate_mean=0.37 means 63% warp, but with flow≈0 the warp is an identity transform. frame_t ≈ 0.63 × frame_t1 + 0.37 × residual. That's a learned blend, not a warp. The architecture's power comes from the WARP exploiting temporal correlation, not from blending."

### Fridrich (Theoretical Framework) responds:
"This is a curriculum design failure, not an architecture failure. Flow never developed because Phase 1 MSE loss can be minimized by the residual alone. The fix: zero out residual for the first 500 epochs, forcing the model to learn temporal correspondence through flow. Only then enable residual as a correction channel."

### Contrarian (Adversarial Reviewer) responds:
"I agree with the flow warmup but challenge the optimism. Even with proper flow, PoseNet needs to drop from 0.135 to 0.006 (22x improvement) to match the postfilter. We haven't proven the architecture can do that. The training-step PoseNet numbers (0.006) use per-batch eval which overfits; the full-eval PoseNet (0.135) is the real number."

### Yousfi responds to Contrarian:
"The training-step vs full-eval gap isn't overfitting — it's a normalization difference. Per-pair vs per-video PoseNet computation. But you're right to be cautious. We need the auth eval number."

---

## Complete Fix List — Council Verdicts

### A. Training Stability (Lagrangian)
| # | Fix | Verdict |
|---|-----|---------|
| A1 | rho_growth 1.02 → 1.005 | **UNANIMOUS YES** |
| A2 | rho_max 10000 → 1000 | **UNANIMOUS YES** |
| A3 | lambda_cap 1e6 → 1e4 (with monitoring) | **YES** |
| A4 | phase1_end=0.25, phase2_end=0.85 | **YES** |
| A5 | Conditional rho updates | **DEFER to v3** |

### B. Architecture/Curriculum (Dead Flow)
| # | Fix | Verdict |
|---|-----|---------|
| B1 | flow_warmup_epochs=500 (zero residual) | **UNANIMOUS YES** |
| B2 | Gradual residual scale (0→1 over 500ep) | **YES** |
| B3 | 3-phase curriculum | **NO — B1+B2 covers this** |
| B4 | Gate init -4.0 | **DEFER until flow develops** |
| B5 | batch_size=16 | **YES, trivially fits T4** |
| B6 | Gradient accumulation | **DEFER until B5 tested** |

### C. Research Roadmap Cross-Reference
| # | Item | Relevance | Verdict |
|---|------|-----------|---------|
| C1 | Ego-motion TTO (implemented) | HIGH — attacks PoseNet at inflate | **PRIORITY 2, $0 cost** |
| C2 | Entropy coding | MEDIUM — 0.02 rate savings | **Polish phase** |
| C3 | AWQ mixed-precision | MEDIUM — quality per byte | **Polish phase** |
| C4 | Scorer resolution round-trip | N/A — renderer already at 384×512 | **N/A** |
| C5 | Odd-frame simplification | HIGH — 50% frames simpler | **Verify and implement** |
| C6 | NeRV positional encoding | HIGH, free (360 bytes) | **Include in next run** |
| C7 | RAFT-lite flow | HIGH if flow develops | **Gated on B1 result** |
| C8 | Knowledge distillation | MEDIUM, 20h GPU | **Week 2** |
| C9 | Depth conditioning | LOW, +100KB rate | **CUT** |
| C10 | Multi-pass inference | HIGH, free at inflate | **Include at eval** |
| C11 | DALI bypass | Already true for renderer | **Already exploited** |
| C12 | YUV null space | N/A for renderer | **N/A** |

### D. Infrastructure (Design Bug)
| # | Fix | Verdict |
|---|-----|---------|
| D1 | Auth eval uses upstream evaluate.py with DALI | **CRITICAL — the whole point of Modal** |
| D2 | Add nvidia-dali-cuda120 to Modal image | **Required for D1** |
| D3 | Apply same fix to Lightning + Kaggle scripts | **YES** |

---

## Training Run v2 Config

Resume from ep4000 checkpoint with:
```
--rho-growth 1.005
--rho-max 1000
--lambda-cap 10000
--phase1-end 0.25
--phase2-end 0.85
--flow-warmup-epochs 500
--residual-scale-ramp-epochs 500
--batch-size 16
--epochs 10000
```

Plus NeRV positional encoding (requires code change).

---

## Score Projections

| Scenario | PoseNet | SegNet | Rate | Score |
|----------|---------|--------|------|-------|
| Current best (ep4000) | 0.135 | 0.006 | 0.004 | 1.88 |
| Fix Lagrangian only | 0.05 | 0.005 | 0.004 | 1.31 |
| Fix Lagrangian + flow | 0.02 | 0.005 | 0.004 | 1.05 |
| + TTO at inflate | 0.006 | 0.005 | 0.004 | 0.84 |
| Best case (match postfilter PoseNet) | 0.002 | 0.005 | 0.004 | 0.74 |
| Quantizr benchmark | ? | ? | ? | 0.60 |

---

## Decision Tree (Pending Auth Eval)

```
Auth eval result (running on Modal):
├── < 1.5  → Architecture works. Apply all fixes. Relaunch.
│           Target: sub-1.0 by April 21.
├── 1.5-2.0 → On par with postfilter. Rate advantage real.
│              Fix Lagrangian. Relaunch.
├── 2.0-3.0 → Pipeline bug or overfitting. Debug first.
└── > 3.0   → Fundamental problem. Fallback to postfilter.
```

---

*Generated by Claude (Anthropic, Opus 4.6) for the comma.ai video compression challenge*
*Skunkworks Council: Yousfi + Fridrich + Contrarian*
*2026-04-13 ~02:30 UTC*
