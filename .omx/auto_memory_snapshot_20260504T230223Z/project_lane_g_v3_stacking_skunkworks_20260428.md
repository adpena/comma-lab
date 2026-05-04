---
name: 2026-04-28 Lane G v3 stacking skunkworks council session — 3 stacks + 3 new lanes + Cycle 1 plan
description: Council deliberation post-Lane-G-v3-1.05 frontier. Wedge attribution shows SegNet=43% of remaining gap (LARGEST, portfolio thin), Rate=36%, PoseNet=20%. 3 candidate stacks (conservative/aggressive/moonshot) with file paths + per-component predicted Δ. 3 NEW lanes proposed (EC-V2, EBR, PRIOR) that don't exist in portfolio. Cycle 1 = $5.80 dispatch (Ω-V2 + EC + SAUG-V2 parallel, 14h).
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Wedge attribution (the strategic finding)

Lane G v3 = 1.05 = pose 0.186 + seg 0.401 + rate 0.462. To get to Quantizr 0.33, need to close 0.72 points. Decomposed against Quantizr's estimated 0.33 = pose ~0.04 + seg ~0.05 + rate ~0.20:

| Wedge | Lane G v3 | Quantizr | Gap | % of total |
|-------|-----------|----------|-----|------------|
| **SegNet** | 0.401 | ~0.05 | **0.31** | **43.2%** |
| **Rate** | 0.462 | ~0.20 | **0.26** | **36.4%** |
| **PoseNet** | 0.186 | ~0.04 | **0.15** | **20.4%** |
| **TOTAL** | 1.05 | 0.33 | 0.72 | 100% |

**Council insight**: Portfolio has been heavy on rate-attack (Lane S/W/Ω/F/I/V/K/SZ — 8 lanes) and pose-attack (Lane G/M/LR/LM/OS/RM/GP/FL/GE — 9 lanes), but THIN on SegNet-attack (only Lane EC + Lane SI variants). Top 2 wedges (SegNet + Rate, 80% of gap) should drive cycle priorities.

## 3 NEW lane proposals (filling SegNet-attack gap)

### Lane EC-V2 — sequential SegNet-flipping with Lagrangian rate cap
- **Premise**: Pareto-dominates Lane EC by greedy water-fill on flip-gain/byte ratio. Sequential allocation: at each step, pick the pixel-flip with highest (Δ_seg / Δ_bytes) ratio until rate cap hit.
- **Predicted band**: [0.85, 1.00] — first sub-1.0 candidate
- **Cost**: $0.50, ~2h
- **Composability**: drop-in replacement for Lane EC. EC-first composition rule applies.

### Lane EBR — entropy-bottleneck rate model (Ballé 2018)
- **Premise**: replace post-hoc quantization with Ballé-2018 entropy-bottleneck end-to-end RD optimization. Joint-train the entropy bottleneck during renderer training. Theoretically optimal RD.
- **Predicted band**: [0.65, 0.95]
- **Cost**: $5, ~12h (full retrain)
- **Composability**: replaces Lane S/W/Ω quantization scheme entirely (mutually exclusive)

### Lane PRIOR — compress-time per-class mask distribution prior
- **Premise**: feed a precomputed per-class mask distribution prior to libsvtav1 as entropy-coder hint. Saves rate on the masks.mkv component without quality loss.
- **Predicted band**: [0.95, 1.05]
- **Cost**: $0.50, ~2h
- **Composability**: orthogonal to all renderer/quant/pose lanes

## 3 candidate stacks (file paths + per-component Δ)

### Stack 1 — Conservative ($5, [0.85, 0.95])
- Lane G v3 baseline (1.05)
- + Lane W (`scripts/remote_lane_w_hard_pair_self_compress.sh`) → -0.05 to -0.10
- + Lane LM-V2 (`scripts/remote_lane_lm_v2_endpoint_tracking.sh`) → -0.005 to -0.010
- + Lane SI-V2 (`scripts/remote_lane_si_v2_learnable_threshold.sh`) → -0.05 to -0.08
- **Total predicted**: [0.85, 0.95]

### Stack 2 — Aggressive ($15, [0.55, 0.75])
- Lane G v3 baseline (1.05)
- + Lane Ω-V2 (`scripts/remote_lane_omega_v2_lagrangian.sh`) → -0.10 to -0.15
- + Lane I-B (Cool-Chic mask codec) → -0.10 to -0.15
- + Lane SAUG-V2 (`scripts/remote_lane_saug_v2.sh`) → -0.05 to -0.10
- + Lane HF (`scripts/remote_lane_hf.sh`) → -0.03 to -0.08
- + Lane EC (`scripts/remote_lane_ec_engineered_corrections.sh`) → -0.05 to -0.10
- **Total predicted**: [0.55, 0.75]

### Stack 3 — Moonshot ($25, [0.20, 0.50])
- Lane V Quantizr replica (`scripts/remote_lane_v_quantizr_replica_88k_halfframe.sh`) — anchor change, predicted [0.50, 1.10]
- + Lane W applied to V's checkpoint
- + Lane SAUG (input perturbation) + Lane SAUG-V2 (noise schedule) — orthogonal axes
- + Lane EC final polish
- OR: Lane SZ replica (`scripts/remote_lane_sz_szabolcs_no_masks.sh`) — predicted [0.30, 0.50]
- **Total predicted**: [0.20, 0.50]

## Cycle 1 dispatch plan ($5.80, ≤14h, parallel 3× 4090)

**Tripartite pact (Yousfi + Fridrich + Contrarian) APPROVED:**

| Order | Lane | Script | Cost | Re-anchored band |
|-------|------|--------|------|------------------|
| 1 | **Lane Ω-V2** | `scripts/remote_lane_omega_v2_lagrangian.sh` | $1.50 | [0.70, 0.95] |
| 2 | **Lane EC** | `scripts/remote_lane_ec_engineered_corrections.sh` | $0.30 | [0.85, 1.05] FIRST SUB-1.0 |
| 3 | **Lane SAUG-V2** | `scripts/remote_lane_saug_v2.sh` | $4.00 | [0.60, 0.90] |

Total: $5.80, ≤14h wallclock.

## 3 transferable patterns (council process improvements)

### Pattern 1 — Re-anchoring rule for pose-TTO-saturated lanes
After a pose-TTO win (Lane G v3 dropped PoseNet from 0.005 to 0.0035), all pose-mechanism lanes (Lane GP, Lane FL, Lane GE, Lane LM-V2, Lane LR-V2) have upper band capped near new frontier — they can't beat what's already optimized. Re-anchor predictions before dispatch.

### Pattern 2 — EC-first composition rule
Engineered corrections (Lane EC, Lane EC-V2) is net-positive for any stack with predicted SegNet ≥ 0.001. Should be the LAST step in any stack composition (compress-time only, doesn't conflict with anything else).

### Pattern 3 — Wedge attribution principle
Always decompose the gap into rate/seg/pose contributions BEFORE choosing next lane. Target the two largest wedges first. Ignore lanes targeting wedges that are already <10% of gap.

## Council critical re-anchoring corrections

- `project_outstanding_work_and_stacks_20260428` Stack A predicted 1.08; actual Lane G v3 = 1.05 (3-point error to GOOD side, validates ±0.05 band methodology).
- `project_lane_taxonomy_stacking_strategy_20260427` predictions mostly valid; pose-TTO lanes need upper-bound tightening per Pattern 1.

## Output

Full synthesis: `/Users/adpena/Projects/pact/.omx/research/lane_g_v3_stacking_skunkworks_20260428.md` (780 lines).

## Cross-references
- `project_lane_g_v3_landed_1_05_20260428` — the new frontier
- `project_outstanding_work_and_stacks_20260428` — TIER 3 portfolio
- `project_cosmos_deep_dive_addendum_20260428` — Lane SAUG-V2 source
- `project_arxiv_2604_24763_tuna2_synthesis_20260428` — composability with T2 lanes
- `project_lane_ec_engineered_corrections_20260428` — Lane EC original
- `feedback_dont_abandon_high_score_lanes_for_stacking_20260428` — keep iterating on V/I/M-V2 too
