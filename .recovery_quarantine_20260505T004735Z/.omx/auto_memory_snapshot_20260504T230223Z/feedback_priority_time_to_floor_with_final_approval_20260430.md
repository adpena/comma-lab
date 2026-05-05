---
name: PRIORITY — drive to Shannon theoretical floor in fastest wall-clock time, final approval before submission
description: 2026-04-30 ~10:05 CDT user mandate: "driving down to theoretical floor in fastest wall clock time possible is the priority within reason" + "must get final approval though". Top objective is time-to-floor. Spawn aggressively in parallel on EV merit (no budget gates per `feedback_no_monetary_commit_20260430`). MUST get explicit human approval before final contest submission / public PR / leaderboard entry — but otherwise dispatch and iterate freely.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## The rule

User: "driving down to theoretical floor in fastest wall clock time possible is the priority within reason" + "must get final approval though"

## Decoded

- **Priority**: time-to-Shannon-floor (0.28 [contest-CUDA])
- **Wall-clock optimization**: parallelize, avoid sequential bottlenecks, dispatch lanes simultaneously where infrastructure allows
- **Within reason**: still EV-rational, don't burn resources on dominated/false-positive lanes, maintain rigor (3-clean-pass, council review, MPS-CUDA discipline)
- **Final approval**: human-in-loop required for contest submission / public disclosure / leaderboard entry / PR-to-upstream — NOT for individual lane dispatches

## How to apply

For each dispatch decision:
1. Predicted band documented? (REQUIRED)
2. Kill criteria documented? (REQUIRED)
3. Pattern A nohup detach? (REQUIRED for >3min)
4. EV positive (predicted improvement OR information value)? (REQUIRED — "within reason" gate)
5. → DISPATCH IMMEDIATELY (no budget approval needed)

For each result:
1. Tag `[contest-CUDA]` / `[empirical:<path>]` / `[derivation]` / `[prediction]` (REQUIRED)
2. Update lane registry maturity gates (REQUIRED)
3. 3-clean-pass adversarial review per landing (REQUIRED)
4. → Move to next experiment

For final submission decisions:
- ANY contest submission, PR-to-comma-ai, leaderboard entry, public disclosure → **PAUSE for explicit user approval**
- The pre-existing CLAUDE.md "Submission PR gate" non-negotiable still applies
- Strategic Secrecy Rule (CLAUDE.md) still applies

## Wall-clock parallelization patterns

Maximize parallel infrastructure utilization:

### Vast.ai 4090 fleet
- Currently 5 instances active
- Can scale to ~10 simultaneous before Vast.ai rate limits
- Each lane can run on its own instance (no contention)

### Modal A10G/T4
- ~$70 credits remaining
- A10G has 22GB shared VRAM (OOM-prone for 21+GB lanes — use Vast.ai for those)
- T4 fits everything but slower

### Local M5 Max MPS
- Free, but MPS is NOISE for strategic decisions per CLAUDE.md
- Use for proxy training, smoke tests, byte-deterministic checks ONLY
- Never tag MPS scores as authoritative

### bat00 RTX 2070S → 3090
- Free, WSL2 GPU passthrough works
- Use for local CUDA training of medium-cost lanes

### Kaggle
- Free T4/P100, 2 sessions max, ~30hr/week budget
- Bonus parallelism for non-deadline lanes

## Time-to-floor escalation ladder (current state)

Per Grand Council #294 verdict + Lane GP forensic #297:

### Tier 1 (next 1-3 days, wall-clock priority)
- Lane PFP16 dispatch ($0.50, predicted -0.005, ZERO distortion)
- Lane Pint12-PCA dispatch ($0.50, predicted +0.00115 marginal, max-abs 0.0025)
- Sensitivity-map module GPU dispatch (Phase 3 #275) — foundational for paradigm shift β
- Ω-W-V3 design + dispatch (PoseNet-sensitivity-weighted layer protection — predicted [1.025, 1.045])
- Lane 12 NeRV contest-CUDA result (in-flight retries)
- Lane 19 logit-margin contest-CUDA result (in-flight retries)
- Lane 8 multi-pass contest-CUDA result (in-flight retries)
- HM-S contest-CUDA result (eval phase, completing soon)
- SC++ V5 recovery dispatch (block_fp_codec tolerance fix)

### Tier 2 (next 3-7 days)
- All-scores forensic hidden-gem recoveries (60% of killed lanes — engineering bugs)
- Lane 17 IMP 10-cycle (~80h running, currently on cycle 1)
- PSD-LumaSkip dispatch IF dispatch council approves
- Phase 3 lane scaffolds → Level 2 (MDL/Bayesian, RAFT/radial pose, bit-level archive)

### Tier 3 (next 1-4 weeks — paradigm shift)
- Mask payload overhaul (paradigm shift α, predicted -0.20 to -0.25 score)
- Sensitivity-aware everything (paradigm shift β, predicted -0.05 to -0.18 indirect)
- Joint score-aware codec stack (paradigm shift γ, predicted -0.015 to -0.05)
- C3 coordinate-MLP residual codec (Phase 3)
- Self-Compressing NN (Phase 3)
- Joint-ADMM 4-stream coordinator deployment
- Lane J-NWC shared-corpus codec build

### Tier 4 (next 1-6 months — sub-Quantizr 0.33)
- Top 6 paradigm shifts stacked optimally
- Multi-pass compress with score-feedback inner loop (Phase 1.5 Lane 8 generalization)
- Custom binary container (saves ~50KB ZIP overhead)
- Bit-level archive optimization
- All Phase 4 integration + paper harness

## Wall-clock bottlenecks to fix immediately

1. **Subagent commit serializer lock contention** — Bug class hardening agent (#296) was supposed to land changed-files-only mode for `tools/preflight_hook.py`. Currently every subagent commit takes 60-120s waiting for preflight; with N>5 concurrent subagents, throughput collapses. **PRIORITY FIX.**

2. **Modal harvester retry-storm** — When the agent hits quota, it enters infinite "completed" notification loop (100+ in last incident). **Need watchdog or single-shot mode.**

3. **NVDEC roulette on Vast.ai 4090** — historical 85% bad-host rate, partially fixed via `launch_lane_with_retry.py --max-retries 5`. Each retry costs ~1-4 min but $0.10 max. **Acceptable but eats wall-clock.**

4. **Subagent quota cap → mid-work death** — Multiple sessions have hit quota cap mid-iteration. Symptom: agents return "You're out of extra usage · resets at <time>". **Need work-checkpoint discipline so re-spawn can resume.**

## Cross-refs

- feedback_no_monetary_commit_20260430.md (the policy change)
- project_grand_council_paradigm_shift_to_shannon_floor_20260430.md (#294 paradigm shifts)
- project_lane_gp_class_forensic_audit_20260430.md (#297 Lane Pint12-PCA recommendation)
- project_quota_incident_2_recovery_state_20260430_1000.md (current recovery state)
- CLAUDE.md "Strategic Secrecy Rule" + "Submission PR gate" (final approval rules)
