# Swarm V1-B0 — PR95-family HiNeRV timing smoke (the B1 gating measurement)

- **Date:** 2026-06-09 (UTC)
- **Agent:** swarm_v1_b0
- **Lane:** `lane_swarm_v1_b0_timing_smoke_20260609`
- **Tool:** `tools/timing_smoke_hinerv_pr95_family.py`
- **Result JSON:** `.omx/research/timing_smoke_hinerv_pr95_family_20260609T031845Z.json`
- **Authority:** `[macOS-MLX research-signal]` — `score_claim=false`,
  `promotion_eligible=false`, `promotable=false`,
  `ready_for_exact_eval_dispatch=false`. This is TIMING + a COST MODEL only.
  **No contest score is claimed.** The score is exact-eval'd later (B2) on
  byte-closed `archive.zip` bytes via `upstream/evaluate.py`.

## Why this exists (MVP-first gating)

Per CLAUDE.md "Long-burn score-lowering campaign default" (a campaign MUST
include "a timing-smoke command that measures seconds/epoch") + "Carmack
MVP-first phasing": before any B1 paid/long dispatch of the PR95-family HiNeRV
600-pair 29650-epoch curriculum, turn the curriculum wall-clock cost from a
GUESS into MEASURED hours on the smallest faithful local MLX surface using
REAL contest pairs (synthetic data FORBIDDEN per CLAUDE.md Catalog #114).

## What was measured

The REAL PR95-family MLX renderer
(`tac.substrates.hi_nerv.mlx_renderer.HinervSubstrateMLX` built from the
default `HinervConfig`: `decoder_channels (48,40,32,24,20,16,12)`, 3-scale
latent pyramid, 7 PixelShuffle upsample blocks + sin activation + bilinear
final resize per CLAUDE.md L18). REAL contest frames decoded from
`upstream/videos/0.mkv` via the canonical `decode_mlx_targets` (official
full-camera -> SegNet/PoseNet bilinear receiver surface at 384x512).

Two surfaces timed in one tool (definitive run: 16 real pairs, batch=16,
2 warmup steps dropped):

| Surface | What | median s/step | **s/epoch (600 pairs)** |
|---|---|---:|---:|
| A | renderer fwd+bwd through score-DOMAIN reconstruction MSE + AdamW step (score-aware EXCLUDED) | 0.057 | **2.18** (lower bound) |
| B | canonical `MlxScoreAwareAdapter.train_step` with REAL gradient-free MLX SegNet+PoseNet teacher caches + canonical learnable student heads (score-aware INCLUDED) | 0.207 | **7.86** (faithful) |

Surface B teacher-cache setup (one SegNet + one PoseNet forward per pair,
gradient-blocked, cached) is a **one-time** 2.6s — NOT a per-epoch cost. The
renderer gradient flows through small learnable student heads distilled toward
the caches; the expensive scorer forwards do not recur per step. This is the
faithful PR95-family training step.

**Surface B is ~3.6x Surface A** — confirming CLAUDE.md's prediction that the
score-aware loss dominates wall-clock. The projection uses **Surface B** (the
faithful number).

## Key findings

1. **Exact param count = 340,802** (torch `HinervSubstrate` == MLX
   `HinervSubstrateMLX`, confirmed match). **This is ABOVE the
   `architecture.py` docstring's "~240K target"** — the
   `(48,40,32,24,20,16,12)` 7-block PixelShuffle decoder is larger than the
   SKETCH comment implies. Still PR95-family-class structure (CLAUDE.md
   L18/L19). FLAG for the design owner: either the docstring should be
   corrected to ~340K or the decoder channels trimmed if a ~240K budget is
   intended. (I did NOT edit `architecture.py` — file owned by sibling swarm
   agents.)

2. **Faithful seconds/epoch (600 pairs, MLX-local) = ~7.86s** (Surface B).
   Renderer-only lower bound = ~2.18s (Surface A).

3. **Full 29650-epoch PR95 curriculum projection (local MLX, $0):** **~64.8 h**
   at the faithful Surface-B rate (~17.6 h at the renderer-only lower bound).

4. **Pragmatic reduced 3000-epoch schedule (local MLX, $0):** **~6.5 h**
   faithful (~1.8 h lower bound).

## Cost model — local MLX ($0) vs paid GPU (advisory bracket)

Paid estimates are a COARSE research-signal bracket: measured MLX s/epoch x a
published MLX<->T4 throughput anchor (0.5x per the CLAUDE.md GPU table
"Local M5 Max MPS ~0.5x T4") x each GPU's `speed_vs_t4`. **The authoritative
paid per-epoch cost is a B1 timing smoke on the actual GPU, never this tool.**

From the faithful Surface-B rate:

| Path | full 29650-ep | reduced 3000-ep |
|---|---:|---:|
| **local MLX (M5 Max)** | ~64.8 h / **$0** | ~6.5 h / **$0** |
| Vast.ai RTX 4090 ($0.25/hr) | ~7.2 h / ~$1.80 | ~0.7 h / ~$0.18 |
| AWS T4 ($0.22/hr) | ~32.4 h / ~$7.12 | ~3.3 h / ~$0.72 |
| Modal T4 ($0.59/hr) | ~32.4 h / ~$19.10 | ~3.3 h / ~$1.93 |
| Modal A100 (~$1.10/hr) | ~5.4 h / ~$5.94 | ~0.6 h / ~$0.60 |

## B1 recommendation (the gated decision)

**`local_mlx_reduced_then_paid_full_if_promising`.**

- The full 29650-ep curriculum (~64.8 h local) exceeds the default 12 h local
  tolerance, so a full local run is impractical as a first B1 step.
- The reduced 3000-ep schedule projects to **~6.5 h local at $0** — a credible
  B1 MVP that requires zero spend and produces a real checkpoint to exact-eval
  at B2.
- **Recommended B1 path:** run the **reduced 3000-ep schedule locally (MLX, $0)**
  as the B1 MVP. Escalate to a **paid full-curriculum campaign** (lane-claim +
  Vast.ai RTX 4090, full-curriculum est ~$1.80 / ~7.2 h — re-measure the real
  per-epoch via a paid timing smoke first) ONLY if the reduced run is promising
  on exact-eval at B2. Vast.ai RTX 4090 is the clear price/performance winner
  for a paid full run (~$1.80 full vs ~$5.94 A100 / ~$19.10 Modal-T4).

## STOP

B0's deliverable is the timing that gates the B1 decision. Per the mission,
**B1 training was NOT started.** The recommendation above routes the B1 launch.

## Wire-in (per CLAUDE.md "Mandatory wire-in" 6-hook)

1. **Sensitivity-map** — N/A (timing tool; no per-axis byte sensitivity produced).
2. **Pareto constraint** — N/A directly; the cost model feeds the campaign
   stop/continue gate (a Pareto-adjacent dispatch-economics constraint).
3. **Bit-allocator hook** — N/A (no per-tensor importance).
4. **Cathedral autopilot dispatch hook** — the cost model + B1 recommendation
   is the input a dispatch ranker needs to schedule B1; consumed by the swarm
   campaign decision, not auto-registered (timing tool, not an archive lane).
5. **Continual-learning posterior** — this memo + the result JSON are the
   durable anchor; the seconds/epoch number reseeds the campaign cost prior.
6. **Probe-disambiguator** — N/A (no 2-way interpretation; the two surfaces
   are lower-bound vs faithful, both reported).

## Reproduce

```bash
.venv/bin/python tools/timing_smoke_hinerv_pr95_family.py \
    --timing-pairs 16 --epochs 4 --score-aware-epochs 2 \
    --batch-pairs 16 --warmup-steps 2
# Surface A only (fast, renderer lower bound):
.venv/bin/python tools/timing_smoke_hinerv_pr95_family.py --skip-score-aware
```

## Caveats / honesty

- MLX numbers are hardware-advisory ONLY (no MPS/CUDA/CPU contest authority).
- The paid bracket is a coarse anchor; B1 must re-measure on the real GPU.
- The reduced 3000-ep schedule is a pragmatic local-feasible default, NOT the
  PR95-faithful 8-stage 29650-ep curriculum (CLAUDE.md L14). A faithful full
  run needs the paid path (or a multi-day local run).
- Surface B uses the canonical adapter + real teachers + canonical student
  heads (`build_learnable_student_head` / `build_learnable_pose_student_head`)
  — NOT a fake stand-in. If the adapter contract shifts, the tool records the
  blocker and Surface A stands as the faithful lower bound.
