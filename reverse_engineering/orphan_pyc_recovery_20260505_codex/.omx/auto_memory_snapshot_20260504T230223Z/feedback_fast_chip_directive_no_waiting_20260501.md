---
name: FAST CHIP DIRECTIVE — H100/A100/RTX 5090 preferred over T4/4090; no time to wait for results
description: 2026-05-01 user mandate. Time pressure to Shannon floor before May 3 deadline means we must use the FASTEST available chips even at higher $/hr. T4 is too slow for iteration cycles; RTX 4090 is acceptable but H100 SXM ($1.80/hr) is the new default for any new dispatch. Wall-clock to a contest-CUDA score is the optimization target, not $.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---

## The directive

User: "make sure we are using fast chips because we don't have time to waste waiting for results"

## Chip ranking (eval throughput / $)

For contest-CUDA archive eval (the bottleneck):
- inflate.sh runs ~5-30 min on T4
- contest_auth_eval upstream evaluate.py runs ~12s on RTX 4090, ~5s on H100

| Chip | Speed vs T4 | $/hr (Vast.ai) | Comment |
|---|---|---|---|
| T4 | 1.0× (baseline) | $0.20 | Too slow for iteration; only used for final T4-equivalent A++ promotion |
| RTX 4090 | ~3-4× | $0.25-0.30 | Current default for OWV3 chains; warm-venv reuse keeps it competitive |
| RTX 5090 | ~4-5× | $0.50-1.00 | Newer; good price/perf |
| A100 SXM4 (40GB) | ~3-4× compute, +VRAM | $0.80-1.50 | Use when VRAM > 24GB needed (Q-FAITHFUL OOM'd on 4090) |
| H100 SXM (80GB) | ~5-8× | $1.80 | NEW DEFAULT for time-critical dispatches |
| H200 | ~6-10× | $2.50+ | Use for heaviest training (NeRV / IMP) |

## Rules

1. **Time-critical dispatches (chain evals, T4 promotion runs, sub-frontier sweeps)**: prefer H100 SXM if available within 2 min boot time. Fallback: RTX 4090 (warm if possible).
2. **Heavy training (NeRV mask codec, IMP cycles, Joint-ADMM iterations)**: prefer H100 SXM 80GB or A100 SXM4 80GB. Avoid H100 PCIE / 40GB if checkpoint loading exceeds VRAM.
3. **Cost-sensitive sweeps (byte-only, build-only)**: stay on RTX 4090 — these don't need GPU compute.
4. **Modal A10G**: $0.59/hr, ~2× T4. Use only if Vast.ai supply is exhausted or if a Modal-specific lane (e.g., `modal_train_lane.py`) requires it.
5. **Lightning T4 / L40S**: ONLY for the final T4-equivalent A++ promotion run on a deploy candidate. NEVER for iteration.

## Vast.ai search filter (canonical)

```bash
.venv/bin/vastai search offers \
    'gpu_name in [H100_SXM,H100_NVL,H100_PCIE,H200] reliability>0.95 disk_space>=80 num_gpus=1 dph<3.0' \
    -o 'dph'
```

If no H100 within budget:
```bash
.venv/bin/vastai search offers \
    'gpu_name=A100_SXM4 reliability>0.95 disk_space>=80 num_gpus=1 dph<2.0' \
    -o 'dph'
```

Last resort:
```bash
.venv/bin/vastai search offers \
    'gpu_name in [RTX_5090,RTX_4090] cuda_vers>=12.4 reliability>0.97 disk_space>=60 num_gpus=1 dph<1.0' \
    -o 'dph'
```

## Cost discipline

User said "no expense will be spared in the time remaining". Translates to:
- Spending $5-10/hr instead of $0.30/hr is FINE if it saves 30 min of wait time
- Don't queue multiple slow dispatches if a single fast one beats the timeline
- Idle a slow instance to launch a fast one — DON'T parallelize on multiple slow chips when one fast chip clears the queue faster

## Session example (2026-05-01 ~13:30Z)

- Vast.ai 35959478 (RTX 4090, $0.26/hr) was the warm box for owv3_0119/0120/0065/0032/0076 chain (5 candidates, ~30 min total)
- For arithmetic-coding lane, launched H100 SXM 35961748 ($1.80/hr) — predicted ~10 min total chain
- Net: $0.30 H100 vs ~$0.13 4090 for same work, but 3× faster wall-clock = 20 min saved

## Cross-refs

- `feedback_500_budget_multiday_arc.md` (budget context — $500 reserve)
- `feedback_compute_budget_hundreds_of_dollars_20260428.md` (operator budget posture)
- `feedback_no_24_dollar_vastai_cap_20260501.md` (no $24 cap)
- `AGENTS.md` "Backend Routing" section (canonical platform priorities)
