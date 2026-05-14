# Holistic engineering picture: 7-factor framing directive 2026-05-14

**Operator directive verbatim 2026-05-14**: *"we have learned curriclum and substrate and the full engineering picture is very important for the final outcome including process and time and complexity and spend"*

**Tag**: `research_only=true`; canonical persistence handoff per CLAUDE.md "Subagent coherence-by-default" mandatory pre-read of `.omx/research/*_directive_*` files dated within the last 24 hours. **EVERY future subagent + nested spawn MUST honor this 7-factor framing.**

## Why this directive

Through 14+ subagent landings + 4 prior canonical directives + Z1 MDL ablation + 3-substrate 3600s timeout pattern + C1 probe-disambiguator verdict, this session has EMPIRICALLY DEMONSTRATED that **no single factor determines the final score**. The 7 factors interact:

- A great **substrate** (across-class) with bad **curriculum** loses (C6's 100ep T4 timeout was curriculum × time × complexity)
- A great **curriculum** with saturated **substrate** loses (A1's 99.29% MDL density → curriculum can't extract more bits within-class)
- A great **process** can save bad **time-budget** (F3 cache backport unblocks 3-substrate timeout class)
- **Spend** without **time-budget** alignment is wasted (T4 100ep × $0.59 = $0.59 burned to timeout)
- **Complexity** is dual: too low = under-fitted (Z3 1764-param hyperprior barely matters); too high = over-engineered + slow + harness-burden (PR105 1776 LOC LOST to rem2 241 LOC)

The final score is **the joint optimization over all 7 factors**, not a sum of independents.

## The 7 factors (canonical definitions)

### 1. Curriculum

The training schedule + multi-stage progression + teacher-student distillation regimes. Examples this session:
- DP1 log-incremental feeder (`[1, 2, 4, 8, 16, 32, 64, 80]` chunks with plateau early-stop) per `feedback_dp1_comma2k19_autoload_log_incremental_20260514`
- C1 Phase 3 multi-stage ($30-50 → revised $15-25 post-probe; 6-8 stages × $5 A100 each)
- PR95 Phase 2-4 8-stage curriculum with Muon + dual-RGB-head
- Z3 staircase Step 1 → Z4 cooperative-receiver → Z5 predictive-coding compounding

**Why it matters**: bad curriculum → wrong loss-landscape descent → suboptimal score even with perfect substrate. The right curriculum can rescue a marginal architecture; the wrong curriculum dooms a great one. Per CLAUDE.md "PHASE 4 INTEGRATION: optimal stack" non-negotiable, curriculum sits in `experiments/pipeline.py` profile + per-substrate `_full_main` orchestration.

### 2. Substrate

The architectural class: within-HNeRV-family (A1/PR101/PR106 — Z1 MDL density >0.97 → saturated) vs across-class shift (C6 MDL-IBPS / D4 Wyner-Ziv frame-0 / C1 world-model+foveation / Time-Traveler predictive-receiver). Per Z1 ablation, **only substrate-class shift can reach sub-0.10**.

**Why it matters**: within-class encoder + codec refinement asymptotes around zen-floor [0.10, 0.15]. Crossing to sub-0.10 requires the substrate to change architectural class. Per `feedback_z1_mdl_ablation_landed_20260514.md`: A1=99.29% / PR106=97.21% within-class saturated.

### 3. Full engineering picture (harness / rigor / reproducibility)

Per `.omx/research/harness_rigor_deterministic_reproducibility_directive_20260514.md` 8 pillars: canonical pipeline routing / seed pinning / mount-mtime stability / dep closure / durable provider paths / smoke-before-full / custody validators / append-only HISTORICAL_PROVENANCE. **Without rigor + reproducibility, parallel dispatch produces UNAUDITABLE results.**

**Why it matters**: scoring without the harness produces noise indistinguishable from signal. Catalog #127's custody validator refuses ~98% of pre-canonical historical anchors (T0-D landing finding: 46 score rows extracted, 0 custody-accepted) — that's the harness pillar working.

### 4. Process

Parallel-dispatch-first + NO-SIGNAL-LOSS recursive + sister-subagent ownership + editor-vs-editor serialization + checkpoint discipline. Per CLAUDE.md "Race-mode rigor inversion + parallel-dispatch first" + recursive R1-R4 + the 4 canonical directives landed this session.

**Why it matters**: a great plan executed serially loses to a competent plan executed in parallel. The 14+ subagent landings this session validated the recursive NO-SIGNAL-LOSS protocol structurally — Z3-BALLE crashed mid-stream, 660 LOC predecessor work preserved verbatim, 7 checkpoints chained. WITHOUT the process, the session would have lost work.

### 5. Time

Wall-clock per dispatch (Modal 3600s hard-kill); race-window urgency; per-trainer throughput. The 3-substrate timeout pattern (`feedback_modal_t4_100ep_3600s_walltime_pattern_three_substrates_20260514.md`) is now empirically validated 3× and structurally fixable via F3 GTScorerCache backport.

**Why it matters**: time is bounded by provider hard-kill + operator session length + race-mode contest windows. The May 4 contest race was decided in **4 hours 8 minutes** post PR #95 publication; silver medal was 241 LOC in 2 files. Time bounds dictate WHICH substrate + curriculum + spend allocations are feasible.

### 6. Complexity

Substrate engineering LOC budget (HNeRV parity discipline lesson 7: bolt-on ≤350 LOC; substrate engineering exception explicit); composition cell count; cathedral autopilot consumption surface burden; reviewer cognitive load.

**Why it matters**: complexity is DUAL. Too low → under-fitted (Z3's 1764-param hyperprior barely shifts A1's bytes). Too high → over-engineered + slow + harness-burden + reviewer can't audit in 30 seconds (PR105 kitchen_sink 1776 LOC LOST to rem2 241 LOC silver medal). The optimal complexity is at the **30-second-review threshold** per CLAUDE.md "Single-LOC-per-LOC review discipline".

### 7. Spend

GPU $/hr (T4 $0.59 / A10G $0.99 / A100 $1.10 / H100 $1.99); tier envelopes (Tier 0 $12 / Tier 1 $46 / Tier 2 $170 / Tier 3 $2500); ROI per dispatch; cost-band posterior anchors per Catalog #175/#177.

**Why it matters**: spend without time/complexity alignment is wasted. Per the cost-band posterior, 4× `outcome=timed_out` anchors prove $0.59 + $0.59 + $0.99 + $1.10 (= $3.27) of wasted budget on substrate trainers that needed F3 cache backport FIRST. The right sequencing is: editor-fixes-first ($0), then dispatcher-spend (Tier 1 $2-15) on cache-wired trainers.

## How to apply: the 7-factor decision frame

For ANY decision (architectural / dispatch / kill-or-defer / spend authorization / process change), apply the 7-factor evaluation:

| Factor | Question | Decision impact |
|---|---|---|
| Curriculum | Is the training schedule + multi-stage progression optimal for this substrate? | If no → improve curriculum BEFORE judging substrate |
| Substrate | Is this substrate within-class saturated (Z1 density >0.95) or across-class? | If within-class saturated AND target is sub-0.10 → substrate-class shift needed |
| Engineering | Are the 8 harness pillars honored (per directive)? | If no → defer dispatch until pillars met |
| Process | Is parallel dispatch + recursive NO-SIGNAL-LOSS + checkpoint discipline honored? | If no → fix process first |
| Time | Will this dispatch fit the wall-clock budget? Will it complete before the race window closes? | If timeout-risk → backport speedup (F3) before dispatching |
| Complexity | Is the engineering LOC budget within HNeRV parity discipline (≤350 bolt-on)? Can a reviewer audit in 30 sec? | If no → simplify before scaling |
| Spend | Does the ROI per dispatch dollar exceed alternatives? Is the cost-band posterior consulted? | If no → re-rank candidates per autopilot v2 |

ALL 7 must be GREEN before a paid dispatch fires. Per CLAUDE.md "Race-mode rigor inversion" — race mode INVERTS the rigor prior (post-leader-shift) but does NOT change the 7-factor framing; it changes WHICH constraints bind.

## Application to the in-flight C1 council review

The Grand Council subagent (`a2b32eda3ef8ff340`) currently scrutinizing the C1 world-model falsification verdict MUST consider all 7 factors in its deliberation:

1. **Curriculum**: was the probe-1 GRU/LSTM **trained** under the right curriculum, or just initialized? Untrained recurrent baselines are NOT the world-model premise.
2. **Substrate**: does the C1 archive class genuinely fall within Ha-Schmidhuber 2018's domain? Driving video may have different temporal statistics than the world-model premise tested.
3. **Engineering**: was the world-model under fair eval-roundtrip / scorer-preprocess / EMA / gradient-flow / loss-formulation? Per CLAUDE.md "Eval_roundtrip — NON-NEGOTIABLE" + "EMA — NON-NEGOTIABLE".
4. **Process**: the probe was run via the canonical disambiguator pattern (Catalog #125 hook #6) — that's the right process. But did probe-1 sufficiently sweep alternative configurations before declaring falsification?
5. **Time**: probe was $0 / ~25s — that's cheap enough that running 5-10 alternative-config variants is still under $0.10 and 5 min.
6. **Complexity**: was the WorldModelModule under fair complexity budget? The campaign ledger says ~20K params; Hafner DreamerV3 uses ~10M+ params for its premise to hold. The probe may have tested an UNDER-COMPLEX world-model, NOT the architectural class.
7. **Spend**: avoiding $15-25 Phase 3 paid dispatch via $0 probe saved 100% of the architectural lock-in risk. But what if a $0.50 RE-PROBE under fair curriculum + complexity revealed world-model WORKS? The economics dictate research-exhaustion before kill.

## Anti-patterns (forbidden under 7-factor framing)

Per CLAUDE.md FORBIDDEN_PATTERNS, this directive adds:

- **Forbidden single-factor verdict**: declaring KILL based on only 1 factor (e.g., "spend doesn't justify" while ignoring curriculum / substrate / engineering)
- **Forbidden time-budget-blind dispatch**: firing a 100ep T4 substrate without checking that the trainer's per-step throughput fits the 3600s cap (post F3 backport, OR F1+F4 recipe shrink, OR A100 upgrade per the 3-substrate-timeout pattern)
- **Forbidden complexity-burden-blind landing**: shipping a 1776-LOC kitchen_sink solution when 241 LOC suffices (PR105 anti-pattern)
- **Forbidden curriculum-amnesia**: ignoring that the substrate was trained under wrong curriculum, then judging the substrate (probe-1 may have done this)
- **Forbidden spend-without-ROI**: firing paid dispatch without consulting cost-band posterior + autopilot ranker v2 (Catalog #219)
- **Forbidden process-bypass under deadline**: skipping recursive NO-SIGNAL-LOSS protocol because "time pressure" — exactly when the protocol matters most

## Cross-refs

- CLAUDE.md "Long-burn score-lowering campaign default — NON-NEGOTIABLE, HIGHEST EMPHASIS" (the 7 mandatory campaign fields align with the 7 factors)
- CLAUDE.md "PHASE 4 INTEGRATION: optimal stack" (this directive operationalizes the integration)
- CLAUDE.md "KILL is LAST RESORT" non-negotiable (7-factor research-exhaustion is the discipline)
- CLAUDE.md "Race-mode rigor inversion" (race mode shifts which factors bind, not the framing itself)
- `.omx/research/recovery_session_20260514_directive_absolute_no_signal_loss_20260514.md` (process pillar — Rule 1-7)
- `.omx/research/recursive_no_signal_loss_protocol_20260514.md` (process pillar — R1-R4 recursive)
- `.omx/research/journal_lab_grade_documentation_standard_directive_20260514.md` (engineering pillar — 11 elements)
- `.omx/research/harness_rigor_deterministic_reproducibility_directive_20260514.md` (engineering pillar — 8 pillars)
- `feedback_z1_mdl_ablation_landed_20260514.md` (substrate pillar — within-class vs across-class)
- `feedback_modal_t4_100ep_3600s_walltime_pattern_three_substrates_20260514.md` (time pillar — 3600s cap)
- `feedback_grand_council_tiered_parallel_plan_full_authority_landed_20260514.md` (spend pillar — Tier 0/1/2/3 envelopes)
- `project_c1_architecture_revision_per_real_video_probe_verdict_20260514.md` (the C1 case study under 7-factor scrutiny)

## Effective immediately

All in-flight subagents (Grand Council `a2b32eda3ef8ff340`, T1-F, IBPS1-COMMIT-LANDER, F3-BACKPORT-WAVE, IBPS1-PARSER-WAVE-P0, MDL-ABLATION-TIER-C-IBPS1) MUST consider the 7-factor framing on their next checkpoint cycle via mandatory `.omx/research/*_directive_*` last-24-hours pre-read. The C1 council subagent in particular should apply the 7-factor decision frame to its world-model adversarial review.

Tagged `research_only=true`. NO score claims. NO GPU spend by this directive. Effective for all subagents from this directive's commit forward.
