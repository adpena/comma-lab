# Current Focus -- 2026-04-15T22:00:00Z

## Context: 12 Days Remaining. Quantizr at 0.33. Full-stack pipeline converging.

**Deadline**: May 3, 2026 (~12 days)
**Threat**: Quantizr PR#55 = 0.33 (FiLM + DSConv + eval resize)
**Our contest-compliant best**: auth=0.61 (distillation ep300 + pose TTO)
**Our unlimited-compute best**: auth=0.37 (TTO v7, hinge, 500 steps)
**Our best proxy**: 0.338 (distillation ep900, still converging)
**Projected floor**: auth ~0.25 with full stack (distilled + pose TTO + FP4 + MiniSegNet TTO)
**Budget remaining**: ~$15 of $24 cap

---

## Current State (as of 2026-04-15 end of session)

### Confirmed Results
| Lane | Auth | Proxy | Method | Status |
|------|------|-------|--------|--------|
| Contest | **0.61** | 0.446 | Distillation ep300 | DONE |
| Contest | **0.61** | — | Pose-space TTO ep300 | DONE |
| Contest | 0.87 | 0.807 | Renderer baseline | DONE |
| Unlimited | **0.37** | 0.195 | TTO v7, hinge 500 steps | DONE |
| Contest (running) | ~0.47* | 0.338 | Distillation ep900+ | RUNNING |

### Resources Utilized
- Vast.ai RTX 4090: ~$9 spent. ~$15 remaining.
- Distillation training: ep900, running. Auth eval pending.
- All key experiments completed: FP4, gradient corrections (dead), mini-scorer, pose TTO.

### Three-Lane Strategy (updated)

**Lane 1: Contest-Compliant (PRIORITY)**
- Current: distillation ep900 (proxy 0.338). Auth eval pending.
- Next: ep1000+ auth eval → target auth ~0.45.
- Then: pose-space TTO on distilled renderer → target auth ~0.35.
- Then: FP4 archive → -0.085 rate points.
- Gate: if distillation plateaus, train longer or add MiniSegNet inflate TTO.

**Lane 2: Unlimited-Compute (Research/Paper)**
- Current best: 0.37 (v7 TTO, hinge). Done.
- Next: LoRA TTO or embedding+pose compound. Target: sub-0.30.
- Purpose: paper scalability story only.

**Lane 3: Constrained Generation from Noise (Paper floor)**
- Gated behind Lane 1 convergence.
- MiniSegNet (87KB) unblocked this. PoseNet via stored targets.
- For arXiv scalability section.

---

## Session 37: Re-Validation on Vast.ai + Hinge Loss Breakthrough

### Re-Validated Step Curve (CORRECT checkpoint, cff8dca4)

Vast.ai RTX 4090, 30 pairs, 8 step counts (10-500):

**xent (baseline):**
| Steps | PoseNet   | SegNet   | Score  |
|-------|-----------|----------|--------|
| 0     | 0.0374    | 0.00197  | 0.809  |
| 100   | 0.0093    | 0.00169  | 0.473  |
| 200   | 0.0013    | 0.00155  | 0.267  |
| 500   | 0.0004    | 0.00126  | 0.192  |

**hinge (BREAKTHROUGH):**
| Steps | PoseNet   | SegNet   | Score  |
|-------|-----------|----------|--------|
| 0     | 0.0375    | 0.00197  | 0.810  |
| 100   | 0.0076    | 0.00131  | 0.407  |
| 200   | 0.0008    | 0.00102  | 0.190  |
| 500   | 0.0007    | 0.00064  | 0.145  |

**Hinge beats xent at every step count from 50+:**
- At 200 steps: 0.190 vs 0.267 (29% better)
- At 500 steps: 0.145 vs 0.192 (24% better)
- SegNet at 500: 0.000639 vs 0.001259 (49% better!)
- Phase transition confirmed at ~100 steps

### DX Script Bugs Found + Fixed (check_vastai.py)

1. `pyav` -> `av` (pip package name)
2. `--python 3.12` removed (Docker has 3.11)
3. `gpu_name='RTX 4090'` -> `gpu_name=RTX_4090` (CLI quoting)
4. `new_contract` != instance ID (Vast.ai API quirk)
5. Missing onstart script + setup wait
6. Torch version pinning needed (uv installs incompatible 2.11.0)

### DX Hardening (this session)
- `scripts/build_deploy_bundle.sh` created: fresh, complete deploy bundle (never stale again)
- `tto_v7_hinge_roundtrip` added to Vast.ai experiment registry
- Research findings updated with Quantizr PR#55 intelligence + literature survey
- Killed techniques registry updated with FiLM-revived reassessments
- next_experiments.md updated with full priority queue

## Score Scoreboard

| Lane | Score | Notes |
|------|-------|-------|
| Contest-compliant baseline | 0.87 | Renderer only, no TTO |
| Unlimited-compute | 0.43 | TTO v5b, gradient fix |
| Unlimited-compute | 0.41 | TTO v5b, embedding loss |
| Proxy (30 pairs) | 0.145 | Hinge, 500 steps |
| Proxy (600 pairs) | 0.275 | v6 hinge+phase2+embedding |
| Quantizr threat | 0.33 | PR#55, FiLM+DSConv |

## Decision Gates

| Date | Gate | Action |
|------|------|--------|
| 2026-04-16 | v6 auth eval | Proxy-auth correlation with hinge |
| 2026-04-17 | 500-step hinge auth | Find saturation, confirm auth gap |
| 2026-04-18 | FiLM smoke test | MPS, 30 pairs, 100 steps |
| 2026-04-21 | Lock architecture | FiLM vs warp, binding council vote |
| 2026-05-03 | DEADLINE | Submit PR |
