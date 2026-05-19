---
review_kind: CATHEDRAL_CONSUMER_WIRE_IN_LANDING
lane_id: lane_mps_prescreen_cathedral_consumer_wire_in_20260519
landing_kind: cathedral_consumer_package_wire_in
ranks_against_canonical_frontier: false
score_claim: false
predicted_mission_contribution: apparatus_maintenance
related_deliberation_ids:
  - meta_assumption_adversarial_review_post_r11_20260519T062526Z
  - mps_phase_b_options_b_plus_c_completion_20260519T062500Z
related_commits:
  - d1d51d1c5  # Slot 2 R11 H1-1+H1-6 fix-wave (invoker callsite wired)
  - c8d51ebb5  # META-ASSUMPTION review post-R11
  - 635c41972  # MPS Phase B Options B+C completion verdict
  - 71960e927  # PoseNet shape-adapter fix unlocking 3rd-component aggregate
---

# MPS-VIABLE pre-screen cathedral consumer wire-in — LANDED

## Authority

Per META-ASSUMPTION review item #2 (`.omx/research/meta_assumption_adversarial_review_post_r11_20260519T062526Z.md`, commit `c8d51ebb5`):

> "Local MPS forward is noisy by 23x and unsafe for promotion" — **HARD-EARNED-NUANCED**: MPS Phase B RE-FIRE empirically demonstrated 0.072% aggregate drift (69× below 5% LOCAL_MPS_TRAIN_VIABLE threshold) on the current archive. The 23× figure from 2026-04-25 is HARD-EARNED for SHIRAZ-era archives + scoring code but does NOT generalize. **Predicted ΔS direct = 0; predicted ΔS indirect (more cycles per $) = -0.005 per week.**

Operator standing directive 2026-05-19 verbatim: *"do everything possible to accelerate dev velocity and save money using local MPS"*.

## Summary

This landing operationalizes the META-ASSUMPTION review's nuanced classification by wiring a new cathedral autopilot consumer that recommends routing advisory-grade substrate experiments to local MPS pre-screen BEFORE paying for Modal/Vast.ai/Lightning dispatch.

The consumer satisfies the canonical `CathedralConsumerContract` per Catalog #335 + auto-activates via Slot 2's R11 H1-1+H1-6 invoker callsite landing (commit `d1d51d1c5`).

## Deliverable

- **NEW package**: `src/tac/cathedral_consumers/mps_viable_prescreen_consumer/__init__.py` (~245 LOC)
- **NEW tests**: `src/tac/tests/test_mps_viable_prescreen_consumer.py` (21 tests, all pass)
- Cumulative `cathedral_consumers/*` count: **22** (was 21 + 1 new)
- Catalog #335 LIVE_COUNT: **0**

## Per-candidate routing logic

The consumer implements a 4-step decision cascade per `consume_candidate(candidate)`:

| Step | Condition | Recommended route | Confidence |
|---|---|---|---|
| 1 | MPS-VIABLE probe SUPERSEDED to blocking verdict (DEFER/KILL/INDEPENDENT) OR probe missing | `paid_cuda_authoritative` | 1.0 |
| 2 | Candidate is promotable contest-axis (`promotion_eligible=True` OR `score_claim_valid=True` + `[contest-CUDA]`/`[contest-CPU]` axis) | `paid_cuda_authoritative` | 1.0 |
| 3 | Candidate is advisory-grade (`[predicted]`/`[advisory only]`/`[macOS-CPU advisory]`/`[MPS-PROXY]`/`[MPS-research-signal]`/`[diagnostic-*]` axis) | `local_mps_prescreen` | 0.9 |
| 4 | Insufficient signal (no axis info) | `none` (cathedral autopilot falls back to default routing) | 0.0 |

`predicted_delta_adjustment` is ALWAYS 0.0 (routing is not a score signal). `promotable` is ALWAYS False (the routing recommendation itself is observability-only per Catalog #287/#323). `axis_tag` is ALWAYS `[predicted]`.

## Catalog #313 probe-outcomes ledger SUPERSEDE auto-fallback

The consumer reads `.omx/state/probe_outcomes.jsonl` at consume-time via `tac.probe_outcomes_ledger.latest_outcome_by_probe_id(MPS_VIABLE_PROBE_ID)`. If the future re-measurement SUPERSEDEs the current PROCEED to a blocking verdict, the next `consume_candidate` call automatically falls back to `paid_cuda_authoritative`. No additional posterior update is required (`update_from_anchor` is intentional NO-OP).

This is the canonical Catalog #313 probe-outcomes ledger consumption pattern: the ledger IS the source-of-truth; consumers query it at decision-time so SUPERSEDE events propagate without manual re-wiring.

## Predicted dev-velocity impact

Per META-ASSUMPTION review item #2 quantification: **5-10× per-cycle acceleration** for substrate experiments whose evidence-grade is advisory-by-design (smoke / proxy / sweep / dev-loop).

Concrete predicted improvement: substrate trainer dev-loop iterations that previously required $0.30-$5 Modal/Vast.ai dispatches for advisory smokes can now run on local MPS at $0 marginal cost. Compounding effect: -0.005 ΔS per week per META-ASSUMPTION quantification.

## Sister-coordination status

| Sister subagent | Status | Coordination |
|---|---|---|
| Slot 2 (R11 H1-1+H1-6 invoker wire-in) | **LANDED** at commit `d1d51d1c5` (sister did this work BEFORE this consumer landed) | This consumer is auto-discovered via the just-landed `discover_and_register_consumers` invocation in `main()` |
| Slot 3 (Catalog #204 cross-driver expansion) | In flight | DISJOINT scope: Slot 3 edits preflight.py + driver scripts; we own NEW consumer package + NEW test file |
| Slot 6 (cheap-signal-first dispatch wave) | In flight | DISJOINT scope: Slot 6 owns Z6 + STC recipe flips + verdicts |

Catalog #314 absorption-pattern avoidance: this landing is NEW package + NEW test file + NEW memo + new memory entry. No edits to preflight.py / CLAUDE.md / cathedral_autopilot_autonomous_loop.py.

## Wave 2C migration recommendation (operator-routable)

The first substrates to migrate to MPS pre-screen routing are those with the highest frequency AND lowest cost-per-dispatch:

1. **C6 IBPS smoke sweeps** — substrate currently in PROCEED_WITH_REVISIONS Phase 2; pre-fix smoke iterations cost $0.30-$2 each on Modal A10G; pre-screen via local MPS would catch 80%+ of pre-fix bugs before paying.
2. **STC v2 smoke + adversarial review iterations** — Modal smoke timeout incidents (e.g. `fc-01KRSVKF9VEESQY2FS33FF4WDM`) burned $1.56 each at rc=25 before training started; local MPS pre-screen would catch driver/recipe/path bugs at $0.
3. **NSCS06 v8 cargo-cult-unwind iterations** — sister subagent currently in NSCS06 Path B reformulation; iterations are advisory-by-design (pre-paradigm-validation); MPS pre-screen at $0 vs Modal T4 $0.30/iteration.
4. **DP1 stacking sweeps** — DP1 codebook composition with sister substrates is advisory until paired Linux contest-CUDA anchor lands; MPS pre-screen accelerates the within-stack hyperparameter search.

## 6-hook wire-in (Catalog #125 non-negotiable)

- **Hook #4 CATHEDRAL_AUTOPILOT_DISPATCH** — PRIMARY: this consumer's routing recommendation is consumed by the cathedral autopilot ranker cascade via `discover_and_register_consumers` (Slot 2 commit `d1d51d1c5`).
- **Hook #6 PROBE_DISAMBIGUATOR** — SECONDARY: the MPS-VIABLE probe outcome IS the canonical disambiguator between MPS-routable and CUDA-required candidates; consumer queries the ledger at consume-time.
- **Hook #1 SENSITIVITY_MAP** — N/A: routing is not a sensitivity signal.
- **Hook #2 PARETO_CONSTRAINT** — N/A: routing does not add Pareto constraints (the routed dispatch will produce its own Pareto-relevant evidence).
- **Hook #3 BIT_ALLOCATOR** — N/A: routing does not affect bit allocation.
- **Hook #5 CONTINUAL_LEARNING_POSTERIOR** — INTENTIONAL NO-OP: the probe-outcomes ledger is the canonical source-of-truth; `update_from_anchor` does not maintain redundant state.

## Discipline contract honored

- **Catalog #314** absorption avoidance: NEW package + NEW test file + NEW memo (no contended-file edits)
- **Catalog #229** premise verification: 6 PVs confirmed pre-edit (probe outcome exists, contract Protocol importable, sister consumer pattern readable, Slot 2 not yet landed at PV time but landed before commit, lane registered, no active dispatch conflicts)
- **Catalog #335** consumer-contract satisfied (validated via `validate_consumer_module(...)`)
- **Catalog #313** probe-outcomes ledger consumption via canonical helper
- **Catalog #287/#323** axis-tagging + Provenance: ALL outputs `[predicted]`, `promotable=False`
- **Catalog #192/#317** non-promotion: MPS routing is advisory ONLY; promotable candidates auto-route to paid CUDA
- **Catalog #125** hook #4 PRIMARY + #6 SECONDARY
- **CLAUDE.md "MPS auth eval is NOISE" non-negotiable**: pre-screen is advisory; authoritative claims still require dual-eval (Linux x86_64 + NVIDIA)
- **Catalog #117/#206** subagent commit serializer + checkpoint discipline

## Lane gates

- ✓ `impl_complete` (package + 21 tests pass)
- ✓ `memory_entry` (this memo + MEMORY.md cross-ref)

Lane `lane_mps_prescreen_cathedral_consumer_wire_in_20260519` L1.

## Budget

$0 (editor only; ~30 min wall-clock).


<!-- # FORMALIZATION_PENDING:pre_framework_memo_dated_2026-05-19_predates_canonical_equations_birthday_registry_population_in_progress_appended_by_strict_flip_enablers_per_operator_blanket_approval_per_claude_md_forbidden_premature_kill_without_research_exhaustion_this_is_DEFER_pending_canonical_equation_backfill_NOT_kill -->
