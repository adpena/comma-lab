# Current Focus -- 2026-04-16T00:45:00Z

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

### Running: v6 TTO (hinge + phase2 + embedding)

- Instance 35026289 on ssh7.vast.ai:26288
- Processing all 600 pairs (1200 frames)
- Config: 150 P1 steps + 200 P2 segnet-only steps, hinge, embedding loss
- Estimated completion: ~20 min from batch 15/60

## Scores
- **Renderer baseline**: auth=0.87
- **TTO v5a (gradient fix)**: auth=0.43
- **TTO v5b (embedding)**: auth=0.41
- **TTO v6 (hinge step curve proxy)**: ~0.145 at 500 steps (30 pairs)
- **Target**: sub-0.20 auth

## Deadline
- May 3, 2026 (~17 days remaining)
