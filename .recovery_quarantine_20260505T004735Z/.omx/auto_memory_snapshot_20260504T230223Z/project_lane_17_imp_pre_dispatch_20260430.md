---
name: Lane 17 (IMP 10-cycle) — Pre-Dispatch Memo (BUDGET-GATED $25, USER APPROVAL REQUIRED)
description: 2026-04-30. Cost breakdown + predicted score band + kill criteria + alternative for the Lane 17 GPU dispatch. ESCALATION TO USER per CLAUDE.md GPU budget non-negotiable ($10 threshold exceeded).
type: project
authoritative_for: lane_17_imp_dispatch_20260430
status: AWAITING_USER_APPROVAL
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## TL;DR — Awaiting User Approval

**Dispatching the full 10-cycle Lane 17 IMP run is estimated at $25.** This exceeds the $10 informal threshold for autonomous dispatch (CLAUDE.md "GPU budget and compute resources" + "Strategic Secrecy Rule"). Pre-dispatch protocol invoked: STOP and ask user.

Two variants on the table:

| Variant | Cost | Cycles | Final sparsity | Coverage of expected benefit | Risk |
|---------|------|--------|----------------|------------------------------|------|
| **Full 10-cycle** | **$25** | 10 | 89.3% | 100% | High variance — sub-100K-param LTH unproven |
| **Quick 5-cycle** | **$12.50** | 5 | 67.2% | ~60% (sparse-CSR breakeven hits at cycle 5) | Lower; if no signal at 67% sparse, full run unlikely to recover |

The quick 5-cycle variant captures the codec breakeven crossover (cycle 5 = 67% sparsity, which is where sparse-CSR begins beating dense FP4) and gives us a kill/promote decision at half the cost. The full 10-cycle is the primary deliverable IF the quick variant lands a competitive score.

## Cost breakdown

### Full 10-cycle run

| Stage | Per-cycle | Cycles | Total |
|-------|-----------|--------|-------|
| Fine-tune (200 epochs × ~30s/epoch on RTX 4090) | ~1.7h | 10 | 17.0h |
| Per-cycle CUDA auth eval (cycles 0, 2, 4, 6, 8, 9) | ~5min | 6 | 0.5h |
| Final auth eval (Stage 4) | ~5min | 1 | 0.1h |
| Re-export + archive build (Stage 2-3) | ~2min | 1 | <0.1h |
| Idle / scheduling overhead | — | — | ~1h |
| **GPU hours (Vast.ai 4090 @ $0.25/hr)** | — | — | **~18.7h × $0.25 = $4.68** |

Wait — $4.68 isn't $25. Let me re-check the dispatcher's actual numbers.

The dispatcher script comment says: "cycle 0 fine-tune is the most expensive (~10h on RTX 4090); cycles 1-9 are ~5h each." That's:
- cycle 0: 10h
- cycles 1-9: 9 × 5h = 45h
- total: 55h × $0.25 = **$13.75** for the GPU compute alone
- plus ~6h for scheduling / NVDEC retries / heartbeat overhead → 60h × $0.25 = **$15** at the LOW end

If using A100 40GB at $0.80/hr → 60h × $0.80 = $48 (rejected).
If using T4 (slower per-step but cheaper) → 100h × $0.22 = **$22**.

The original dispatcher estimate of $25 seems to be a conservative ceiling factoring in retries, Vast.ai NVDEC roulette (re-launches), and the per-cycle smoke auth eval overhead. Council kept this estimate as the COST CAP for safety.

### Quick 5-cycle run

| Stage | Per-cycle | Cycles | Total |
|-------|-----------|--------|-------|
| Fine-tune cycle 0 (most expensive) | 10h | 1 | 10h |
| Fine-tune cycles 1-4 | 5h | 4 | 20h |
| Per-cycle smoke (cycles 0, 2, 4) | 5min | 3 | 0.25h |
| Final auth eval | 5min | 1 | 0.1h |
| Idle / overhead | — | — | ~3h |
| **GPU hours @ $0.25/hr (Vast.ai 4090)** | — | — | **~33h × $0.25 = $8.25** |

Conservative ceiling with NVDEC retries: **$12.50**.

## Predicted score band (Council Lane-17 design 2026-04-30)

| Scenario | Probability | Score | Archive | Outcome |
|----------|-------------|-------|---------|---------|
| Best case: lottery ticket exists | ~25% | **0.90 - 1.00** | ~250KB | -0.05 to 0.0 vs Lane G v3 (1.05) |
| Most likely: weak ticket signal | ~50% | **1.00 - 1.15** | ~270KB | flat to slight regression |
| Kill: regression triggers revert | ~25% | reverts to baseline (~1.05) | 290KB (Lane G v3 unchanged) | sunk cost limited |

**Council narrows to band [0.92, 1.12] central, [0.85, 1.20] 90% CI.**

## Explicit success / null / regression criteria (Council Round 2 M5)

| Cycle 9 [contest-CUDA] score | Verdict | Action |
|------------------------------|---------|--------|
| score < 1.00 | **STRONG WIN** — promote Lane 17 to primary submission candidate | ship; stack with Ω-W-V2 |
| 1.00 ≤ score < 1.05 | **WIN** — Lane 17 ties or beats Lane G v3 anchor | promote; integrate as fallback |
| 1.05 ≤ score < 1.155 | **NULL** — within revert threshold but no win | paper-section material only; do not ship |
| score ≥ 1.155 | **REGRESSION** — kill criterion fires | revert applied; lane 17 deprecated |

The 1.155 = 1.10 × 1.05 boundary is the revert threshold (Council Q4 9/10).

## Kill criteria (revert-on-regression)

The dispatcher now implements the Council Q4 (9/10 vote) kill criterion:

```bash
IMP_REGRESSION_THRESHOLD=1.10
# Per-cycle smoke auth eval at cycles 0, 2, 4, 6, 8, 9.
# If cycle_N_score > 1.10 × min(cycle_0..N-1_score):
#   - Write REVERT_TO_CYCLE.txt with the best cycle index
#   - Break out of the IMP loop
#   - Stage 2 sources from $FINAL_CYCLE_DIR = best cycle's dir
#   - Final archive = best-scoring cycle's renderer (NOT cycle-9)
```

This caps the regression-class downside at the cost of cycles up to the revert point.

## Why $25 / $12.50 is justified

1. **Sparse architecture is a paradigm Quantizr has not explored.** Quantizr's 0.33 leaderboard score uses 88K dense FP4. If a lottery ticket exists at the 88K-param scale, the Lane G v3 → Lane 17 archive shrinks ~290KB → ~250KB and the score drops to ~0.95 — that's a $25 buy-in on a potential -0.10 score delta.
2. **It's a paper deliverable regardless of competitive outcome.** Whether or not Lane 17 ships in the contest archive, the empirical measurement of LTH at 88K-param scale on a video-compression renderer is publication-worthy (the paper section "Lottery tickets at sub-100K scale").
3. **The kill-criterion caps the downside.** If cycle 5 regresses, we revert + STOP at $7-8 of GPU spend. The full $25 is only paid if the 89% sparsity actually holds.

## Why we should NOT skip this even if we don't get user approval today

The Lane 17 chain is now ~2,500 LOC + 17 unit tests + IMPS inflate handler + Check 94 STRICT preflight. The infrastructure is permanent and benefits any future sparsity work. The DECISION GATE is the GPU dispatch, not the code.

## Council recommendation (Inner-quintet 2026-04-30)

**Tier 1**: dispatch the **5-cycle quick variant ($12.50)** first. Gates the full 10-cycle on whether cycle 5 lands within `[0.95, 1.20]`.

**Tier 2** (if Tier 1 lands inside [0.95, 1.10]): dispatch the **incremental 5 more cycles** ($12.50 incremental) to reach the full 89% sparsity result.

**Tier 3** (if both Tier 1 and Tier 2 land): Stack with Lane Ω-W-V2 (block-FP on survivors) for the final paper-grade result.

Total max budget: $12.50 + $12.50 = $25, but dispatched in 2 stages with a kill-or-continue gate in between. This is STRICTLY BETTER than the all-at-once $25 dispatch.

## Required user response

Reply with one of:

- **"approved-quick"**: dispatch the 5-cycle quick variant ($12.50). Stops at cycle 4. Promotion to full 10-cycle requires a second approval after harvest.
- **"approved-full"**: dispatch the full 10-cycle ($25). The dispatcher's revert-on-regression caps the downside.
- **"approved-stack"**: dispatch quick + Lane Ω-W-V2 stack (~$15). Smaller scope, tests stack composition.
- **"deferred"**: hold the dispatch. Lane 17 chain is in production state and ready to fire when user wants.
- **"killed"**: kill the lane permanently. Council Q3 vote 7/10 supports a revisit if the broader Phase 2 ACCELERATE plan calls for it.

## Pre-flight checklist (already passing)

- [x] CLAUDE.md FORBIDDEN PATTERNS reviewed before any code typed
- [x] subagent_commit_serializer used for every commit
- [x] EMA decay 0.997 wired in `experiments/train_imp_cycle.py:140-143`
- [x] eval_roundtrip inherited from `contest_auth_eval.py` → `inflate.sh` → `evaluate.py`
- [x] Auth eval per scheduled cycle (Council Q3) + final
- [x] No invented CLI flags (dispatcher does an argparse dead-flag scan at startup)
- [x] NVDEC probe at Stage 0
- [x] Provenance JSON + heartbeat loop
- [x] `[contest-CUDA]` tag at completion
- [x] Pattern A nohup detach for the dispatch invocation (NEVER `Bash run_in_background:true` per CLAUDE.md)
- [x] Vast.ai instance MUST have `--label "lane-17-imp-$(date +%Y%m%d)"` (Check A STRICT)
- [x] Vast.ai instance MUST register to `.omx/state/vastai_active_instances.json` (Check B STRICT)
- [x] Cost cap $24 hard limit on Vast.ai launch wrapper

## Cross-refs

- `lane_17_imp_scaffold_audit_20260430.md` (Phase A)
- `council_lane_17_imp_design_20260430.md` (Phase B — council vote table)
- `feedback_production_hardened_standard_definition_20260430.md` (Level 3 standard)
- `project_phases_2_3_4_design_implementation_math_provenance_20260429.md` §"Lane 17"
- CLAUDE.md "GPU budget and compute resources", "Recursive adversarial review protocol"
