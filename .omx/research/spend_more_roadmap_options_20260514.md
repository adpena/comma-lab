# Spend-more roadmap: budget allocation tiers (2026-05-14)

**Operator directive (verbatim 2026-05-14)**: *"we can spend more to get higher signal"*
**Operator decision**: **Option 4 ACTIVE** (codex-aligned tight portfolio ~$27); **Options 1, 2, 3 DOCUMENTED as deferred roadmap tiers**.

## Tier 0 (ACTIVE, this session): Option 4 — Codex-aligned tight portfolio

**Budget**: ~$27 dispatch + 7-10 days build time
**Predicted ΔS**: -0.040 to -0.077 (sub-0.155 reachable from PR101 0.193 baseline)
**Rationale**: codex did the math; we execute the CUDA-credible dispatches. Validates codex's predictions before committing higher tier.

| Workstream | Cost | Build | Predicted Δ | Status |
|---|---|---|---|---|
| Harvest codex CUDA-in-loop palette sweep | $0 | — | feeds HDM8 selector | gated on codex `fc-01KRK453W4GT5A99XFTMQ3KXMF` harvest |
| HDM8 cumulative wave (selector + YUV6 sublattice + motion-translate + Hadamard/chirp) | $6 | 5-7 days | -0.020 to -0.060 | queued post-DP1-HARDEN-V2 |
| DP1 Phase 2 Scenario A dispatch | $5 | ~30 min | -0.005 to -0.008 | queued post-DP1-HARDEN-V2 |
| D1 SegNet margin polytope encoder | $1 | 3 days | -0.005 to -0.012 | **build subagent dispatched this turn** |
| D4 Wyner-Ziv frame-0 substrate | $15 | 7-10 days | -0.025 to -0.045 | **build subagent dispatched this turn** |

**Total predicted Amdahl-stacked**: -0.040 to -0.077 from PR101 0.193 → 0.116-0.153 (sub-0.155 cleared)

## Tier 1 (deferred): Option 1 — Portfolio bet $50-100

Adds to Tier 0:
- DP1 upgrade to Scenario C ($45 incremental): 500k-1M frames distilled on Modal A100, codebook ~25-50 KB; production-grade codebook for 6+ downstream compositions
- D2 camera-resolution YUV6 nullspace encoder ($1, 3-5 days): extends YUCR orthogonally to D1; exploits 80.7% pixel-direction left-nullspace (deep-math eureka)
- Composition cells dispatch ($10): YUCR × DP1 × HDM8 × D-paths matrix entries empirically tested

**Total**: ~$83 cumulative; predicted -0.060 to -0.100 stacked = sub-0.155 + sub-0.10 zen-floor edge in range

**When to activate**: after Tier 0 dispatches return empirical anchors. Tier 1 doubles the budget for ~30-50% more predicted ΔS.

## Tier 2 (deferred): Option 2 — Maximalist sub-0.10 push ~$300

Adds to Tier 1:
- DP1 Scenario D federated production ($200-300 incremental): 2M+ frames distributed, PR-shippable production-deployment codebook
- A1 × substrate composition matrix exhaustive ($20 incremental): all pairwise combinations dispatched
- Repeated codex review rounds + 5-round skunkworks council greenup before PR submission ($10 incremental for cross-machine variance dispatches)

**Total**: ~$300 cumulative; predicted same band as Tier 1 (-0.060 to -0.100) for contest score BUT adds production-deployment-aligned codebook (~3-week federated infrastructure project)

**When to activate**: after Tier 1 demonstrates the math; production-deployment alignment becomes strategic priority beyond contest.

## Tier 3 (deferred): Option 3 — Vertical scaling only ~$300-500

DP1 Scenario D federated ($300-500); skip D-paths + codex waves; focus entirely on pretrained-codebook robustness.

**When to activate**: only if operator strategically prioritizes production-deployment alignment over contest-score push. Score-side gain bounded by Contrarian's -0.012 verdict.

## Decision tree

```
Tier 0 (active) — validate codex predictions empirically
   ↓
   if anchors land within predicted bands → Tier 1 (portfolio doubling)
   if anchors deliver upper-band ΔS → Tier 2 (maximalist sub-0.10)
   if anchors deliver lower-band ΔS → re-evaluate D-path mechanism
```

## Sub-0.188 gate clearance probability per tier

| Tier | Total cost | Predicted Δ | Final score (PR101 0.193 - Δ) | Sub-0.188 cleared? | Sub-0.155 cleared? | Sub-0.10 in range? |
|---|---|---|---|---|---|---|
| 0 | $27 | -0.040 to -0.077 | 0.116-0.153 | ✅ YES | ✅ if upper-band | ⚠️ tight |
| 1 | $83 | -0.060 to -0.100 | 0.093-0.133 | ✅ YES | ✅ YES | ⚠️ edge |
| 2 | $300 | -0.060 to -0.100 | 0.093-0.133 | ✅ YES | ✅ YES | ⚠️ edge |
| 3 | $300-500 | -0.012 | 0.181 | ⚠️ marginal | ❌ NO | ❌ NO |

## Cross-refs

- `.omx/research/deep_math_geometry_manifolds_synthesis_20260514.md` (D1/D2/D4 derivations, 88 KB)
- `.omx/research/segnet_posenet_frame_exploit_latest_research_20260514_codex.md` (codex HDM8 frame-exploit research, 34 KB)
- `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_yucr_substrate_landed_20260514.md` (YUCR L1)
- `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_dp1_phase_2_landed_20260514.md` (DP1 Phase 2 L1)

Tagged `research_only=true`. NO score claims. NO GPU spend by this memo. All predicted ΔS tagged `[mathematical-derivation]` / `[first-principles-bound]`.
