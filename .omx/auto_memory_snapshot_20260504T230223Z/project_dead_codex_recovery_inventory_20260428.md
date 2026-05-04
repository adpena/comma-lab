---
name: Dead codex recovery inventory (post 2026-04-28 mass session loss)
description: 2026-04-28 PM 9 codex CLI sessions died (only molt-project codex --yolo alive). Inventory of all dead-subagent work across 25-30 distinct items. Wave 1 respawned 5 (SAUG/MOS/F-V5/HM+CG landed; M-V3 API500 retry). Wave 2 dispatched 6 parallel Claude general-purpose subagents.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## What died (audit confirmed 2026-04-28 PM)

The original task list claimed 9 in-flight codex sessions. Only `codex --yolo` PID 30856 alive — and it's working on **molt** (different repo). All pact-lane sessions gone.

## Full backlog (25+ distinct items)

### Tier 1 — Critical recovery (immediate dispatch)
- **Lane M-V3-clean BUG-1 fix** — train/inference pose-pad parity per skunkworks audit (RETRY after API 500)
- **Lane W learnable weights don't learn** — Round 11 Finding 2 (CRITICAL — invalidates any Lane W deploy)
- **Lane WC resume-from crash** — Round 11 Finding 1 (deploy blocker)
- **Lane EC engineered corrections** — script exists as `scripts/remote_lane_ec_engineered_corrections.sh` (??)
- **Lane EC-V2 Pareto-dominant** — test exists as `test_lane_ec_v2_greedy.py` (??)

### Tier 2 — Cosmos/MAE/Tuna-2/Telescope research lanes (high EV)
- **Lane SAUG-V2 Cosmos** — HighSigmaStrategy 5% sigma redraw (orthogonal to Lyra Lane SAUG)
- **Lane MAE-V** — joint mask-aug from epoch 0
- **Lane HF Telescope** — hyperbolic foveation 4-param learnable
- **Lane HFM** — multi-scale Telescope variant
- **Lane WC Curator** — Cosmos-Embed1-336p outlier soft-DTW scoring
- **Lane T2-DROP** — Tuna-2 masking-based feature learning Tab.6 (+0.6 to +4.2 perception)
- **Lane T2-RATIO** — Tuna-2 seg-weight sweep
- **Lane T2-DUAL** — Tuna-2 dual-encoder methodology
- **Lane TFR** — Cosmos Transfer multi-control encoder
- **Lane CTRL** — multi-control normalization (Σweights ≤ 1.0 rule)
- **Lane CFG** — CFG dropout (text=0.2, video=0.2, action=0.0 from Cosmos)
- **Lane HAAR** — 3D Haar wavelet tokenizer

### Tier 3 — Architectural revival (lower predicted EV but interesting)
- **Lane SG** — re-scoped SegNet protected layers per Lane G v3 wedge attribution
- **Lane LR-V2/V3** — LoRA delta per content
- **Lane V-DMD** — adversarial QAT distillation
- **Lane FP** — FramePack variable-kernel
- **Lane CCW** — canonical-coordinate warping

### Tier 4 — Council EUREKA deploy scripts (8 separate)
- 8 deploy scripts mentioned in run_log "council EUREKA" — all dead

### Anti-arbitrariness gates already landed (commit chain)
- Preflight check 41 (heartbeat loop)
- Preflight check 42 (train/inference pose-projection parity — BUG-1 class extinct)
- Preflight check 43 (launcher tarball anchor parity)
- Round 23-26 fixer: 19 distinct deployment-chain bugs hardened

## Wave 1 outcome (2026-04-28 PM)

| Lane | Status | SHA |
|------|--------|-----|
| SAUG | ✅ landed | 226abded |
| MOS | ✅ landed | 9cdd052b |
| F-V5 hardware FP8 | ✅ landed | 54d29cec |
| HM+CG | ✅ landed | b3b4a978 (HM was already in src/tac/contrib) |
| M-V3-clean | ❌ API 500 — retry | — |

## Wave 2 dispatch (in flight)

6 parallel Claude general-purpose subagents covering:
1. M-V3-clean retry
2. Lane W + Lane WC Round 11 Findings 1+2 fix
3. Lane SAUG-V2 Cosmos + Lane MAE-V
4. Lane HF Telescope hyperbolic foveation
5. Lane EC + EC-V2 engineered corrections
6. Lane SG protected-layers re-scope

## Cross-references
- `feedback_metabugs_round_3_20260428` — 3 metabugs from V5 launcher
- `project_lane_g_v3_landed_1_05_20260428` — current frontier
- `project_lane_m_v2_audit_council_findings_20260428` — BUG-1 audit
- `project_lane_taxonomy_stacking_strategy_20260427` — full lane taxonomy
- `.omx/research/cosmos_mae_2604_telescope_synthesis.md` — research synthesis
- `.omx/research/cosmos_deep_dive_addendum_20260428.md` — Cosmos corrective pass
- `.omx/research/lane_g_v3_stacking_skunkworks_20260428.md` — wedge attribution
- `.omx/research/lane_m_v2_audit_council_20260428.md` — M-V2 audit
- `.omx/research/codex_adversarial_review_round_11_20260428.md` — Round 11 findings
- `.omx/research/arxiv_2604_24763_synthesis.md` — Tuna-2 synthesis
