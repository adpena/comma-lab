---
name: Lightning.ai Pro account ACTIVATED — SSH credentials + 240 annual credits
description: 2026-04-30 ~10:50 CDT user signed up for Lightning AI Pro $50/mo annual ($20/mo effective) with $240 annual credits. SSH credentials provided. Agents should use this for paradigm-shift training (H100 access), paper supplement / dashboards, and IMP 10-cycle migration from slower Vast.ai 4090.
type: reference
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## SSH credentials

```
ssh s_01knw7wnzbe79wfq5mqqbx1mbz@ssh.lightning.ai
```

User: Alejandro Pena (adpena@gmail.com)
Plan: Pro $50/mo billed annually ($20/mo effective)
Credits: $240/year included
Account dashboard: https://lightning.ai

## What Lightning is for (per CLAUDE.md decision memory)

- **Paper supplement + reproducibility notebook** (top writeup value)
- **Comparison dashboards + viz + graphics** (top writeup value)
- **H100/A100 access** for paradigm-shift training (3-4× wall-clock faster than Vast.ai 4090 on FP4/FP16 ops)
- **Persistent Studio** (no NVDEC roulette, no SCP roundtrip per dispatch)
- Lane 17 IMP 10-cycle migration (3× wall-clock multiplier vs Vast.ai 4090)

## Cost math vs alternatives

| Path | Cost | Wall-clock |
|---|---|---|
| **IMP 10-cycle on Vast.ai 4090** ($0.30/hr × 80h) | $24 | 80h (~3.3 days) |
| **IMP 10-cycle on Lightning H100** ($3.50/hr × 25h) | $87.50 | 25h (~1 day) |
| **Difference** | **+$63** | **−55h saved** |

H100 wins on wall-clock-priority mandate. Cost is paid via Lightning credits ($87.50 of $240 = 36%).

| Workload | Modal-only | Lightning subscribed | Diff |
|---|---|---|---|
| ~170 hr mixed deadline workload | ~$280 | ~$320 ($240 sub + ~$80 overage) | +$40 over Modal |
| But: $2k contest prize | — | — | +$40 = 2% of prize, trivial |

## Bootstrap workflow (for agents)

```bash
# 1. SSH into Studio
ssh s_01knw7wnzbe79wfq5mqqbx1mbz@ssh.lightning.ai

# 2. Inside Studio, install uv + clone repo
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone <repo-url> /workspace/pact
cd /workspace/pact
uv venv
uv pip install -e .

# 3. Verify CUDA + GPU available
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"

# 4. Set TAC_UPSTREAM_DIR + PYTHONPATH
export TAC_UPSTREAM_DIR=/workspace/pact/upstream
export PYTHONPATH=src:upstream:$PWD
```

## Apply for academic tier IN PARALLEL

Per `feedback_lightning_ai_activated_for_writeup_value_20260430.md`:
- Submit https://lightning.ai/docs/team-management/academia/students application
- Research justification: comma.ai video compression contest + paper supplement + Shannon-floor research
- If approved: 80% off Pro = ~$10/mo (vs current $20/mo)
- Ask Lightning support for prorated refund of remaining months

## Lane 17 IMP migration plan

- Currently on Vast.ai 35899275 at cycle 1 of 10
- Let cycle 1 complete on Vast.ai (sunk-cost is small, harvest checkpoint)
- After cycle 1: migrate cycles 2-10 to Lightning H100
- Estimated savings: 50-60 hours wall-clock
- Cost on Lightning H100: ~$80 of $240 credits

## Cross-refs

- feedback_lightning_ai_activated_for_writeup_value_20260430.md
- feedback_no_monetary_commit_20260430.md
- feedback_priority_time_to_floor_with_final_approval_20260430.md
- feedback_full_six_month_plan_aggressive_no_shortcuts_20260430.md
