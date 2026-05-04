---
name: Leaderboard 2026-04-29 — Selfcomp 0.38 #2, Mask2mask 0.60 #3
description: Live leaderboard intel. Two new entries beat us. Selfcomp #2 at 0.38 uses self-compression ~1.017 bpw + PoseNet-affine-learned-image. Mask2mask #3 at 0.60 obfuscated arch. Lane G v3 (1.04) would rank ~4th. Selfcomp validates Lane Ω Hessian-per-weight premise.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
Leaderboard fetched 2026-04-29 ~10am from comma.ai/leaderboard + GitHub PRs:

| Rank | Score | Name | PR | Technique |
|---|---|---|---|---|
| 1 | 0.33 | quantizr | #55 | FiLM CNN 88K + KL-T2 + AV1 |
| 2 | **0.38** | **selfcomp** | **#56** | self-compression ~1.017 bpw + affine-learned-image PoseNet |
| 3 | **0.60** | **mask2mask** | **#53** | "slightly different arch" (obfuscated) |
| 4 | 1.89 | neural_inflate | #49 | |
| 5 | 1.91 | svtav1_dilated_ren | #58 | dilated renderer + AV1 |
| ours | 1.04 | Lane G v3 | not submitted | KL distill + pose TTO retry |

**Score breakdowns**:
- Selfcomp 0.38: seg 0.12 + pose 0.08 + rate 0.19 (rate = 2nd biggest)
- Mask2mask 0.60: seg 0.264 + pose 0.081 + rate 0.257 (seg ≈ rate)

**Why this matters**:
- Selfcomp's 1.017 bpw validates Lane Ω-Hessian-QAT premise (#196 in flight). Per-weight bit allocation to that floor is empirically achievable on this scorer.
- Selfcomp's "PoseNet trained with affine-transformed learned image" is a novel trick: per-pair learned image as compress-time, affine warp as inflate-time-cheap encoder. Inspires NEW Lane LI proposal.
- Mask2mask author refused to publish compress script — competitive secrecy active. Reinforces our strategic-secrecy rule.
- Quantizr at 0.33 is reachable from current 1.04 via stacking Lane W + Lane Ω + smaller renderer.

**How to apply**:
- Prioritize Lane W (rate attack, in flight) and Lane Ω (Hessian per-weight, in flight) — both directly aligned with Selfcomp's gain sources.
- Kill UNIWARD-as-mask-prior (council 2026-04-29 5/5 unanimous, see project_council_kill_uniward_20260429).
- Defer Lane LI (Learned-Image PoseNet trick fork) until Lane W result lands.
- Re-fetch leaderboard daily through May 3 deadline; watch for code releases on PR #56 / #53.
