# Retroactive sweep for PR110-OPT-7 trainer 4-helper canonical wire-in 2026-05-30

Per CLAUDE.md "Catalog #348 retroactive sweep" non-negotiable: every new STRICT
preflight gate (or NEW substrate / trainer modification that touches the 4-helper
canonical wire-in surface) MUST emit a retroactive sweep memo per the 4-field
contract: bug-class symptom signature + pre-fix window + historical
KILL/DEFER/FALSIFY search results + per-finding RE-EVAL priority assignment.

This sweep is for the PR110-OPT-7 trainer wire-in landing (NOT a new STRICT
gate; existing Catalog #226 + #205 + #222 + #270 STRICT gates remain
canonical). The sweep is operator-routable per the Catalog #348 standing
discipline + memory completeness per CLAUDE.md "Bugs must be permanently
fixed AND self-protected against" non-negotiable.

## Bug-class symptom signature

**Bug class**: `substrate_trainer_l1_scaffold_dispatch_emits_archive_but_not_auth_eval_json_v1`

**Symptom signature**:
- Trainer is L1 IMPL_COMPLETE per lane registry
- Trainer's `main()` function emits archive + training_stats.json
- Trainer has ZERO canonical auth_eval helper invocations (`grep -c "gate_auth_eval_call" trainer.py = 0`)
- Trainer has ZERO canonical scorer-loss helper invocations (`grep -c "score_pair_components" trainer.py = 0`)
- `tools/canonical_dispatch_optimization_protocol.py --strict` returns `overall_pass=false` with Tier 1 `canonical_scorer_loss=FAIL` + Tier 3 `canonical_auth_eval_helper=FAIL` + Tier 3 `canonical_inflate_device=FAIL` + Tier 3 `scorer_loader_order_correct=FAIL`
- `tools/local_pre_deploy_check.py --strict` returns `auth_eval_reachability=FAIL` (no reachable auth_eval invocation from entrypoints)
- Recipe at `.omx/operator_authorize_recipes/substrate_*.yaml` has `dispatch_enabled: false` per Catalog #240 because firing it would burn paid Modal/Vast.ai dispatch producing only an archive + no contest-CUDA score evidence

**Empirical anchor**: PR110-OPT-7 L1 PROMOTION landing at commit `1230b3b9c` 2026-05-30
shipped a 319 LOC SMOKE-ONLY trainer per the Phase C landing memo. The trainer was
correctly tagged `research_only=true` per Catalog #240, but a subsequent paired-CUDA
RATIFICATION dispatch attempt 2026-05-30 ~23:00Z DEFERRED at the canonical pre-flight
harness per `feedback_pr110_opt7_l1_paired_cuda_ratification_DEFER_pending_trainer_wire_in_landed_20260530`
because the 4 canonical helpers were not wired into the trainer.

## Pre-fix window

The bug-class symptom signature first manifested at L1 PROMOTION commit `1230b3b9c`
landing 2026-05-30 ~20:52Z and persisted until the wire-in landing at commit
(THIS commit batch) 2026-05-30 ~23:34Z. Pre-fix window = ~2 hours 42 minutes.

This is a SHORT pre-fix window — the predecessor DEFER subagent CORRECTLY identified
the 4 wire-in gaps + filed the canonical operator-routable next-action per CLAUDE.md
"Forbidden premature KILL" + canonical reactivation cascade. NO paid dispatch fired
during the pre-fix window because the canonical pre-flight harness refused at
Catalog #243 + #270.

## Historical KILL/DEFER/FALSIFY search results

Per the canonical search across the `.omx/state/probe_outcomes.jsonl` ledger + the
`~/.claude/projects/-Users-adpena-Projects-pact/memory/` directory:

1. **PR110-OPT-7 L1 PROMOTION** (commit `1230b3b9c`): PROCEED — Phase C MLX-LOCAL
   smoke 7/7 GREEN. Substrate paradigm INTACT per Catalog #307. THIS sweep does
   NOT revise that verdict; the substrate paradigm remains validated.

2. **PR110-OPT-7 L0 SCAFFOLD** (commit `3fd28b5b2`): PROCEED — canonical
   inverse-scorer basis primitive. THIS sweep does NOT revise that verdict.

3. **Yousfi-T1 A+B+C canonical helpers** (commit `3d027ecf9`): PROCEED — three
   canonical pose-axis primitives. THIS sweep does NOT revise that verdict.

4. **alaska canonical patterns** (commit `61a91a48e`): PROCEED — canonical
   inverse-steganalysis color separation primitive. THIS sweep does NOT revise.

5. **PR110-OPT-7 paired-CUDA RATIFICATION DEFER** (probe_id
   `pr110_opt7_l1_paired_cuda_ratification_dispatch_DEFER_pending_trainer_auth_eval_wire_in_20260530`):
   DEFER (predecessor) → SUPERSEDED (THIS landing). Closed via canonical
   `update_probe_outcome(..., event_type=EVENT_SUPERSEDED, ...)` per Catalog #110/#113
   APPEND-ONLY HISTORICAL_PROVENANCE discipline.

6. **Sister substrate L1 SCAFFOLDS** (lane registry grep for L1 substrates with
   similar bug-class signature): per the audit table grep, several sister substrate
   L1 SCAFFOLDS may carry the same canonical-helper-missing gap pattern. THIS sweep
   does NOT mass-flag them — each substrate's dispatch readiness is tracked
   individually via the canonical Catalog #243 + #270 pre-flight harness which
   fires structurally before dispatch.

No historical KILL or FALSIFY verdicts were affected by this wire-in landing.
Per CLAUDE.md "Forbidden premature KILL without research exhaustion": the
predecessor DEFER is a research-deferral that the wire-in landing addresses
canonically, NOT a paradigm kill that requires retroactive re-evaluation.

## Per-finding RE-EVAL priority assignment

| Probe outcome | Pre-fix verdict | Post-fix verdict | RE-EVAL priority | Rationale |
|---|---|---|---|---|
| PR110-OPT-7 L1 PROMOTION | PROCEED | PROCEED | NONE | Substrate paradigm INTACT per Catalog #307; Phase C MLX-LOCAL preserved (re-validated 5/5 + substantive=PASS) |
| PR110-OPT-7 L0 SCAFFOLD | PROCEED | PROCEED | NONE | Upstream primitive; wire-in is downstream |
| Yousfi-T1 A+B+C | PROCEED | PROCEED | NONE | Canonical primitives; wire-in is downstream consumer |
| alaska canonical patterns | PROCEED | PROCEED | NONE | Canonical primitive; wire-in is downstream consumer |
| PR110-OPT-7 paired-CUDA RATIFICATION DEFER | DEFER (blocking) | SUPERSEDED via EVENT_SUPERSEDED | RESOLVED | All 6 reactivation criteria satisfied per CLAUDE.md "Forbidden premature KILL" canonical reactivation cascade |

## 6-hook wire-in declaration per Catalog #125 sister-discipline

- Hook #1 sensitivity-map: ACTIVE — the canonical `score_pair_components` invocation in `_full_main` produces per-pair (seg, pose) score components consumable by downstream sensitivity-map consumers.
- Hook #2 Pareto constraint: N/A at this wire-in landing; Pareto polytope constraints are emitted by the substrate's 5-helper composition (alaska + Yousfi-T1 A+B+C + PR110-OPT-7) per the L1 PROMOTION landing memo.
- Hook #3 bit-allocator: N/A at this wire-in landing; the substrate's OPT7VYT1 archive grammar is byte-frozen per Catalog #146 at the L1 PROMOTION landing.
- Hook #4 cathedral autopilot dispatch: ACTIVE — the canonical `gate_auth_eval_call` invocation in `_full_main` enables auth-eval JSON emission which feeds the canonical posterior + autopilot ranker per Catalog #245 sister discipline.
- Hook #5 continual-learning posterior: ACTIVE — probe outcome registered at `.omx/state/probe_outcomes.jsonl` per Catalog #313 PROCEED 14-day window + predecessor DEFER superseded per Catalog #110/#113 APPEND-ONLY discipline.
- Hook #6 probe-disambiguator: ACTIVE — the canonical 4-helper presence IS the disambiguator between dispatch-ready vs scaffold-only states per the canonical_dispatch_optimization_protocol Tier 1/3 token check.

## Cross-references

- Predecessor DEFER memo: `.omx/research/pr110_opt7_l1_paired_cuda_ratification_DEFER_pending_trainer_wire_in_20260530.md`
- Wire-in landing memo: `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_pr110_opt7_trainer_4_helper_wire_in_landed_20260530.md`
- L1 PROMOTION landing memo: `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_pr110_opt7_l1_promotion_via_yousfi_t1_landed_20260530.md`
- Phase C MLX-LOCAL re-validation artifact: `experiments/results/pr110_opt7_wire_in_phase_c_revalidation_20260530T233412Z/training_stats.json`
- Wire-in tests: `src/tac/tests/test_train_substrate_pr110_opt7_via_yousfi_t1_canonical_4_helper_wire_in.py` (19 tests; all PASS)
- Sister substrate trainer reference: `experiments/train_substrate_a1_plus_lapose.py` (canonical 4-helper wire-in pattern)
- Canonical helper docs: Catalog #205 + #222 + #226 + #270 in CLAUDE.md
