# Current Focus -- 2026-04-15T18:00:00Z

## Context: 13 Days Remaining. Quantizr at 0.33. Three-Lane Strategy.

**Deadline**: May 3, 2026 (~13 days)
**New threat**: Quantizr PR#55 = 0.33 (FiLM + DSConv + eval resize)
**Our contest-compliant best**: auth=0.87 (renderer baseline)
**Our unlimited-compute best**: auth=0.43 (TTO v5b, gradient fix validated)
**Our best proxy**: 0.275 (TTO v6, hinge+phase2+embedding, 600 pairs, RTX 4090)

---

## Three-Lane Strategy

### Lane 1: Contest-Compliant TTO (PRIORITY)
- TTO runs at compress time (unlimited); inflate = single forward pass
- Hinge loss confirmed 24-49% better than xent from 50+ steps
- v6 proxy=0.275 needs auth eval to confirm
- **Next**: Auth eval of v6 TTO frames (708MB tto_frames.pt downloaded locally)
- **Path to 0.20**: Extend TTO steps to 500 P1 + 500 P2 with hinge

### Lane 2: FiLM Architecture (NEW — Quantizr validated)
- FiLM pose conditioning on renderer feature maps
- Directly addresses DP-SIMS PoseNet failure (temporal coherence via pose conditioning)
- Quantizr at 0.33 proves the paradigm works
- **Gate**: Implement FiLM in inflate_renderer.py, then run smoke test on MPS
- **Target**: FiLM + hinge TTO should reach 0.15-0.20 [contest-compliant]

### Lane 3: Constrained Generation from Noise (Paper / GPU Eureka)
- GPU Eureka projected 0.135. Cool-chic literature validates the paradigm.
- Gated behind Lane 2 architecture work
- For arXiv scalability section; not the submission path

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
