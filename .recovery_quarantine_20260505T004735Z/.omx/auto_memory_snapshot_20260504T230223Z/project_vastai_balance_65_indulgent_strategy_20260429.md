---
name: Vast.ai $65.59 credits available — indulgent supplementary spend approved
description: 2026-04-29 PM. User confirms Modal is primary lane; Vast.ai is indulgent supplement. Verified balance $65.59 via console.vast.ai API. Spend can be aggressive (use credits up over 4 days to deadline). Council kill list applies to PRIMARY budget allocation, not VAST.AI supplementary.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
**Verified balance**: $65.59 (via direct API call to /api/v0/users/current/, 2026-04-29 PM).

**User mandate**: "we can spend all of it today and tomorrow if we want, we don't care, modal is the primary lane now and we can be somewhat indulgent."

**Spend strategy**:
- Modal handles long-form lanes (12-14h trainings, $5-15 each).
- Vast.ai handles parallel + big-chip + cheap supplementary work.
- $65 / 4 days deadline = ~$16/day spending budget.
- Codex's $150 budget allocation applies to TOTAL Modal+Vast.ai+probes; Vast.ai $65 fits within the 30% archive-diet + 10% probes slice.

**Vast.ai allocation plan**:
- $4.50: Tier 1 — Q-CD-SWEEP 3-config conv-dim sweep on RTX 4090 (codex top-action #1)
- $20: Tier 2 — H100 Self-Compression NN PoC OR pose-TTO 1000-step
- $0.50: Tier 3 — archive-diet validation on 4090 (after Subagent M)
- $20: discretionary big-chip experiments as Modal results inform direction
- $20: reserve for re-dispatches / NVDEC-roulette retry budget

**Vast.ai operational risks** (reminders from feedback memory):
- 85% NVDEC bad-host rate on RTX 4090 (most fixed but still appearing)
- Bash launch-returns-success-before-lane-starts pattern
- Bash harness 144-trap on long-running launcher invocations
- Korea region 4090 base image fails CUDA init (filter geolocation!=KR)
- Use `scripts/launch_lane_on_vastai.py` 5-phase split or `launch_lane_with_retry.py` (auto-retry NVDEC_BAD/SCP_FAIL)
- ALWAYS `--label` instances; track in `.omx/state/vastai_active_instances.json`
- Always `tarball-only parity` (no git pull / reset on remote)
- Heartbeat loop required; verify_vast_instances.py per +30min

Cross-refs:
- feedback_vastai_nvdec_roulette_pivot_to_modal_20260429
- feedback_canonical_parent_shell_launcher_20260428
- feedback_launcher_v5_auto_discovery_tarball_20260428
- feedback_canonical_lane_lifecycle_DECISION_TREE_20260428
- project_codex_theoretical_floor_brutal_20260429 (60/30/10 budget)
