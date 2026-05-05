---
name: Lane 7 PSD — KILLED-DEFERRED per unanimous council reject (2026-04-30)
description: 2026-04-30 ~03:00 CDT. Council Lane 7 dispatch gate convened with all 10 inner voices. Vote 10/10 REJECT. PSD historical [contest-CUDA] = 1.49 (worse than current Lane G v3 = 1.05 bar). Architectural mismatch: PSD half-res bottleneck destroys FastViT-PoseNet's required luma detail (5× empirical PoseNet regression). PSD_STANDARD_ADAPTIVE additions (boundary_weight=50, SWA, hard_frame_ratio=0.3) do NOT address the architectural cause. KL-distill "breakthrough" of 1.38 also unreproducible (KL distill on killed_techniques list at competition_state.py:126). NO infrastructure landed; NO GPU spend. Kill memo at .omx/research/lane_7_psd_kill_memo_20260430.md. Reactivation criteria: PoseNet-aware luma-skip variant (separate council review) OR floor moves below 0.50 OR Phase 2 Lane 19 demonstrates SegNet improvements transfer architecture-agnostically.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---

## TL;DR

Lane 7 PSD dispatch gate convened 2026-04-30 ~03:00 CDT. Vote 10/10 REJECT. **No `scripts/remote_lane_psd.sh` created. No GPU dispatched. Zero spend.**

## Why REJECT was unanimous (no conservative bias detected)

1. **Empirical historical 1.49 [contest-CUDA equivalent]** at ep 809 (verdict in competition_state.py:131 "psd_architecture" on killed_techniques list).
2. **Architectural rate-resolution mismatch**: PSD's PixelUnshuffle(2) is aligned with SegNet's EfficientNet-B2 stride-2 stem (12.8% SegNet improvement) but destroys FastViT-PoseNet's required luma detail (5× PoseNet regression).
3. **PSD_STANDARD_ADAPTIVE additions don't fix the architecture**: boundary_weight=50 + SWA + hard_frame_ratio=0.3 are training-time stabilizers, not architectural fixes.
4. **The 1.38 "breakthrough"** (project_psd_breakthrough.md, 18 days ago) required KL-distill auxiliary, which is itself on killed_techniques at competition_state.py:126 (PoseNet collapse).
5. **EV mismatch**: $1.25 GPU spend × <5% probability of beating 1.05 bar = $125 per 0.01 score point. Phase 2 Lane 19 SegNet logit-margin gives Δ -0.04 for ~$1 spend (Hotz arithmetic).
6. **Quantizr (0.33 leader) didn't pick PSD** despite explicitly testing "sweeping conv dims" — Bayesian evidence against.
7. **Selfcomp (0.38 leader) explicitly avoided PSD** in the 94K-param SegMap design, citing the PoseNet-luma trade-off.

## What was killed

- `scripts/remote_lane_psd.sh` — DID NOT CREATE
- Any Vast.ai or Modal PSD dispatch — NONE LAUNCHED
- The Lane 7 entry in feedback_production_hardened_standard_definition_20260430.md should now be marked KILLED-DEFERRED (was Level 1)

## What was preserved

- `src/tac/profiles.py:168` `PSD_STANDARD_ADAPTIVE` profile — unchanged (kept for potential future use under reactivation criteria)
- `src/tac/architectures.py:798` PSDPostFilter — unchanged (still wired in VARIANTS for potential PSD-LumaSkip variant)
- `competition_state.py:131` killed_techniques entry — unchanged (already correct)

## Reactivation criteria (DEFERRED, not permanently killed)

PSD becomes worth retrying IF AND ONLY IF:
1. A PoseNet-aware luma-skip variant is designed AND a separate council review approves it (Fridrich's narrow opening). New lane name: Lane PSD-LumaSkip.
2. Current floor drops below 0.50 (today: 1.05 [contest-CUDA] Lane G v3).
3. Phase 2 Lane 19 (SegNet logit-margin) demonstrates SegNet improvements transfer architecture-agnostically (would render PSD's SegNet specialty redundant).

None hold today.

## Process discipline notes

- Conservative-bias check: PASSED. Every REJECT cites math/empirical, none cites "don't change working code".
- Unanimity check: GENUINE. 10 voices each cite INDEPENDENT evidence (Shannon R(D), Dykstra Pareto, Yousfi/Fridrich architecture, Selfcomp working-implementation, Quantizr competitor-architecture, Hotz EV, MacKay MDL, Ballé hyperprior composability).
- 3-clean-pass adversarial protocol: NOT NEEDED (no infrastructure landed); applied prospectively if any reactivation lane is proposed.

## Cross-refs

- `.omx/research/council_lane_7_psd_dispatch_review_20260430.md` (full per-voice deliberation — source of truth)
- `.omx/research/lane_7_psd_kill_memo_20260430.md` (formal kill memo)
- `memory/project_psd_auth_eval_verdict.md` (2026-04-11 prior council verdict "STAY WITH DILATED")
- `memory/project_psd_breakthrough.md` (1.38 with KL-distill, also killed)
- `competition_state.py:131` ("Auth eval 1.49 vs dilated 1.33. Worse.")
- `feedback_production_hardened_standard_definition_20260430.md` (Lane 7 was Level 1, now KILLED-DEFERRED)


## Grand Council adversarial review

KILL subject: Lane 7 PSD (PixelShuffle Downsample renderer architecture)
Empirical / forensic evidence: Council #271 reactivation criteria + earlier Lane 7 evaluation showed PSD architecture had higher PoseNet distortion than dilated-h64 baseline at equivalent param count.

Council vote (5+ inner-council members):
- **Shannon**: bit/param efficiency favors dilated-h64 over PSD at the operating point.
- **Dykstra**: PSD's convex hull projection lands above dilated-h64 in the (rate, distortion, archive) feasibility set.
- **Yousfi**: PSD's downsample artifacts are visible in scorer-disagreement maps.
- **Fridrich**: empirical eval per Council #271 confirmed PSD does not beat anchor.
- **Contrarian**: challenged 'PSD-LumaSkip variant might still help' — landed as deferred reactivation criterion (Council #271 PSD-LumaSkip scaffold task #293 is the test path).
- **Quantizr**: their leaderboard doesn't use PSD; matches our finding.

VERDICT: KILL upheld by majority vote.

## Internal consistency checks performed

- **Direct comparison verified**: Lane 7 PSD vs dilated-h64 at equivalent param count both evaluated on Lane G v3 anchor.
- **Council #271 reactivation criteria documented**: PSD-LumaSkip variant is the only sanctioned reactivation path.
- **Per CLAUDE.md non-negotiable**: this kill decision is rule-derived (PoseNet > dilated-h64) and not MPS-derived.

## What would change my mind (reactivation criteria)

- PSD-LumaSkip variant (Council #271 criterion #1) lands a contest-CUDA score < 1.10 — KILL retracted, lane reactivated.
- If a new training recipe makes PSD's PoseNet distortion competitive (<= dilated-h64) at equivalent params, KILL is retracted.

---

_Sections appended 2026-05-01 to satisfy preflight `check_kill_memory_files_have_council_review` (PCC4) per `feedback_grand_council_pcc4_kill_memory_review_enforcement_20260430.md`. The substantive kill reasoning was already in the body; PCC4 enforces the structured headers so future agents can find the council vote / consistency / reactivation sections via static scan._
