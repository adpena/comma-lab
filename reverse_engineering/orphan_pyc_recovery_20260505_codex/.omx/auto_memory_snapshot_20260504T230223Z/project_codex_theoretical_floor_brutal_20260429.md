---
name: Theoretical floor (codex gpt-5.5 xhigh) — sub-0.3 is 35-45% in 4 days, NOT a EUREKA-lane problem
description: 2026-04-29 PM. Codex CLI rate-distortion analysis. Critical correction — Selfcomp's seg/pose/rate numbers are SCORE CONTRIBUTIONS not raw distortions. Floor is 0.22-0.30 hard, likely 0.24-0.28. Winning path is Quantizr-family conv-dim sweep + archive diet, NOT new EUREKA lanes.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
**Critical correction landed**: Selfcomp's seg=0.122 / pose=0.074 / rate=0.186 are SCORE CONTRIBUTIONS, not raw distortions. Real Selfcomp non-rate distortion = 0.194. Real Quantizr non-rate distortion ≈ 0.135 (293KB archive at rate 0.195).

**Theoretical floor**: 0.22-0.30 hard, most likely 0.24-0.28. Sub-0.30 is **35-45% probable** in 4 days (NOT 8% as my earlier optimistic framing suggested), AND ONLY IF (a) Quantizr-class clone lands AND (b) archive diet works. <15% if next 24h don't produce Quantizr-class local score.

**Rate-budget table** (sub-0.30 requires):
| Archive | Rate term | Required non-rate distortion |
|---|---|---|
| 180KB | 0.120 | < 0.180 |
| 240KB | 0.160 | < 0.140 |
| 279KB (Selfcomp) | 0.186 | < 0.114 |
| 293KB (Quantizr) | 0.195 | < 0.105 |
| 300KB | 0.200 | < 0.100 |

So sub-0.30 at Quantizr size = cut non-rate distortion from 0.135 → 0.105 (22% reduction). OR keep Quantizr distortion + shrink archive to ~240KB.

**Top-3 highest-EV actions (codex-derived)**:
1. **Quantizr-family conv-dim/latent-rate sweep** — $60-90 / 12-18h / predicted Δ -0.02 to -0.06. Sweep at 240/270/293/320 KB targets. Decision rule: 15KB must save 0.01 score or it's net-negative.
2. **Archive diet / packing pass** — $0-10 / 4-8h / Δ -0.015 to -0.045. Manual tensor packing, split entropy streams, zstd tune. Saving 45KB = 0.03 score. **NEW high-EV path not in our existing portfolio.**
3. **Pose residual/affine bolt-on** — $20-40 / 6-12h / Δ -0.01 to -0.03. Bolt-on to proven pipeline.

**Codex kill list (extends council kill list)**:
- Any archive > 350KB (rate alone 0.233, needs distortion < 0.067 = below floor)
- DARTS arch search (too slow, noisy for 4 days)
- Lane AL, FC, PA as PRIMARY lanes (only useful as bolt-ons within 12h)
- mae_v_v2, lane_w_v2, sz_phase2_v2 unless first eval beats Selfcomp's slope
- Adaptive rebalance / high seg weights / PoseNet clamps / large alpha_seg
- Old KL distill loss_mode (overweighted implementation)
- Multi-pass inflate unless beats 15KB→0.01 slope rule

**Budget allocation ($150)**:
- 60% / $90 — Quantizr-family conv-dim + latent-rate sweep
- 30% / $45 — Archive-diet engineering + SH/TR/PD bolt-ons
- 10% / $15 — Cheap probes only (MM v2, one AL checkpoint, no speculative long runs)

**Brutal verdict**: 70% confidence ship target = 0.31-0.35. Best-case ship 0.27-0.30 if Quantizr clone + archive diet land. Sub-0.30 NOT a 70% outcome. The winning path is NOT a new EUREKA lane.

**How to apply**:
- Watch q_faithful_v3 (Quantizr 1:1 replica) + SC++ v3 (Selfcomp+KL) — these are the load-bearing lanes per codex.
- Spawn archive-diet engineering subagent: manual tensor pack, entropy stream split, zstd tune. Aim 45KB+ savings.
- Demote Lane AL/FC/PA to BOLT-ONS only (cheap probes that take <12h to first auth eval).
- Do NOT continue iterating on UNIWARD/FR-Ω/HM-S/DARTS-S/WC-S unless q_faithful or SC++ scores below Selfcomp's slope.
- Re-fetch leaderboard daily through May 3 — competitor moves change the optimal-action ordering.

Cross-refs:
- project_grand_council_brutal_forecast_20260429 (earlier 60/20/8 forecast, now superseded by codex's 35-45% sub-0.30)
- project_selfcomp_reverse_engineered_20260429 (Selfcomp source RE)
- project_council_kill_uniward_20260429
