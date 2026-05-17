---
title: "PROBE-OUTCOMES-BAKE-IN — Catalog #313 canonical 4-layer ledger landed (sister of Catalog #245 Modal call_id ledger exemplar)"
date: 2026-05-16
author: PROBE-OUTCOMES-BAKE-IN subagent (probe_outcomes_bake_in_20260516)
lane: lane_probe_outcomes_canonical_ledger_bake_in_20260516
horizon_class: apparatus_maintenance
council_tier: T1
council_predicted_mission_contribution: apparatus_maintenance
council_override_invoked: false
council_attendees: [PROBE-OUTCOMES-BAKE-IN_subagent]
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_decisions_recorded:
  - "Catalog #313 claimed via canonical serializer per Catalog #186 (transactional)"
  - "Layer 1 canonical helper src/tac/probe_outcomes_ledger.py landed (mirrors tac.deploy.modal.call_id_ledger pattern)"
  - "Layer 2 operator-facing CLI + library tools/check_predecessor_probe_outcome.py landed"
  - "Layer 3 STRICT preflight gate check_dispatch_target_has_no_predecessor_adjudicated_outcome landed at strict=True in preflight_all"
  - "Layer 4 runtime wire-in tools/operator_authorize.py::_check_predecessor_probe_outcome between Catalog #152 + #243"
  - "Catalog #292 amended to add Assumption-Adversary standing question 'has this probe / dispatch already been executed and adjudicated?'"
  - "Sister discipline path PROBE_OUTCOMES_LEDGER_PATH registered in Catalog #131 _SHARED_STATE_PATH_MARKERS so direct writes outside canonical helper refuse"
  - "2 historical probe outcomes backfilled (ATW v2 D4 H(latent|scorer_class) → INDEPENDENT + Wunderkind G1 v2 per-pair-dominant SegNet argmax reducer → DEFER)"
related:
  - src/tac/deploy/modal/call_id_ledger.py
  - .omx/research/atw_codec_v2_d4_probe_verdict_20260516_codex.md
  - .omx/research/atw_d4_probe_recipe_disambiguation_20260516.md
  - .omx/research/grand_council_t2_wunderkind_g1_v2_pivot_validation_v3_cpu_competitiveness_20260516.md
council_assumption_adversary_verdict:
  - assumption: "the scattered-evidence-no-queryable-index bug class for probe outcomes is structurally identical to the scattered-call_id bug class Catalog #245 extincted"
    classification: HARD-EARNED
    rationale: "Empirical: 60+ probe-related memos exist at `.omx/research/*probe*.md` with no single queryable index; the ATW v2 D4 verdict + Wunderkind G1 v2 reducer DEFER were both captured but the apparatus could not answer 'has this been adjudicated within the last 30 days for this recipe?' before paid dispatch. The bug class is the same shape as the Modal call_id ledger pre-Catalog-#245 state."
  - assumption: "the 30-day staleness window matches the canonical L1 substrate retirement window per Catalog #298"
    classification: HARD-EARNED
    rationale: "Per CLAUDE.md 'Substrate retirement discipline' non-negotiable: L1 SCAFFOLD substrates are flagged stale after 30 days without dispatch/mark activity. Same default semantically applies to probe-outcome staleness."
  - assumption: "blocking verdicts {INDEPENDENT, KILL, DEFER} are research-deferrals not kills per CLAUDE.md 'Forbidden premature KILL without research exhaustion'"
    classification: HARD-EARNED
    rationale: "Per CLAUDE.md non-negotiable + Catalog #308: every probe verdict that blocks dispatch is the apparatus saying 'this specific reducer/methodology has been adjudicated; alternative reducers warrant trial before class-wide kill.' The gate REFUSES re-dispatch but does NOT mark the lane killed."
---

## TL;DR (60 seconds)

Per operator NON-NEGOTIABLE directive 2026-05-16 (PROBE-OUTCOMES-BAKE-IN
subagent): *"bake in the FULL 4-layer canonical pattern per the Catalog #245
Modal call_id ledger exemplar so probe-disambiguator verdicts are queryable
across sessions and gating dispatch BEFORE we re-run something an existing
adjudicated probe already settled."*

**All 4 layers landed in same commit batch**:

1. **Layer 1 (canonical helper)**: `src/tac/probe_outcomes_ledger.py` (~750 LOC)
   — fcntl-locked JSONL APPEND-ONLY at `.omx/state/probe_outcomes.jsonl`;
   mirrors `tac.deploy.modal.call_id_ledger` (Catalog #245 exemplar).
2. **Layer 2 (operator CLI)**: `tools/check_predecessor_probe_outcome.py`
   — `--recipe` / `--substrate` / `--list-blocking` / `--json` flags;
   exit codes 0/1/2.
3. **Layer 3 (STRICT preflight gate)**: Catalog #313
   `check_dispatch_target_has_no_predecessor_adjudicated_outcome` at
   `src/tac/preflight.py` lines 67899-68190; wired `strict=True` in
   `preflight_all()` at line 2622.
4. **Layer 4 (runtime wire-in)**: `tools/operator_authorize.py::_check_predecessor_probe_outcome`
   between Catalog #152 `_validate_required_input_files` and Catalog #243
   `_run_local_pre_deploy_check` insertion points; paired-env bypass per
   Catalog #199 sister rule.

**Live count at landing: 0** (Catalog #313 STRICT preflight gate). **All
META-meta sister gates clean**: #131 (canonical helper exclusion list +
PROBE_OUTCOMES_LEDGER_PATH in `_SHARED_STATE_PATH_MARKERS`) + #138 (strict-load
sister) + #176 (CLAUDE.md catalog row present) + #118 (no duplicate numbers)
+ #159 (catalog text matches strict value) + #185 (LIVE_COUNT=0 honored) +
#292 (amended with Catalog #313 standing question).

**77/77 dedicated tests pass** across `src/tac/tests/test_probe_outcomes_ledger.py`
(50 tests: schema validation / register happy-path + invalid-input rejections /
update event-type transitions / load_outcomes lenient + strict / quarantine
on corrupt / atomic write / query helpers / 4-proc spawn-pool stress / full
lifecycle / ATW v2 D4 backfill regression / JSONL byte-stable sort_keys) and
`src/tac/tests/test_check_313_dispatch_target_no_predecessor_probe_outcome.py`
(27 tests: live-repo regression / positive blocking-outcome flagged / 3
blocking-verdict-types / negative no-outcome + PROMOTE not-blocking + token-
without-recipe-path / same-line waiver + 3 placeholder rejection variants /
test files excluded / intake clones excluded / strict raises with Catalog
#313 / strict silent on clean / canonical helper self-exempt / multi-violation
aggregation / missing dir + string repo_root / comment lines / only .py + .sh
scanned / ATW v2 D4 end-to-end / orchestrator regression guards).

**2 historical probe outcomes backfilled** (Layer 1 ledger):

1. `atw_v2_d4_h_latent_given_scorer_class_20260516` — substrate
   `atw_codec_v2`; verdict `INDEPENDENT`; MI=0.006385 bits/symbol; threshold
   0.5 MEANINGFUL_CONDITIONING; evidence
   `.omx/research/atw_codec_v2_d4_probe_verdict_20260516_codex.md`; blocking;
   adjudicated_at 2026-05-16T22:47:41Z; expires 2026-06-15T22:47:41Z.
2. `wunderkind_g1_v2_per_pair_dominant_segnet_argmax_reducer_20260516` —
   substrate `wunderkind_g1_v2`; verdict `DEFER`; per Q1 SPLIT-VERDICT in
   `grand_council_t2_wunderkind_g1_v2_pivot_validation_v3_cpu_competitiveness_20260516.md`:
   `RATIFY-FALSIFICATION-OF-THE-SPECIFIC-v2-REDUCER + REQUEST-REINVESTIGATION-
   OF-ALTERNATIVE-REDUCERS-BEFORE-CLASS-WIDE-DEFERRAL` per Catalog #308;
   blocking; adjudicated_at 2026-05-16T00:00:00Z; expires 2026-06-15T00:00:00Z.

## Bug class anchor

Pre-landing: probe-disambiguator outcomes were SCATTERED across many surfaces
with NO single queryable index. The apparatus could not answer the question
"has this substrate / recipe already been adjudicated by a probe within the
last 30 days, and if so what was the verdict?" before paid Modal/Vast.ai
dispatch. Sister subagents could re-fire dispatch on substrates where the
apparatus had ALREADY adjudicated the question — burning paid GPU re-measuring
an answer the system already had.

Empirical anchor: 2026-05-16 ATW v2 D4 H(latent|scorer_class) probe (Codex
`tools/run_atw_v2_d4_probe_from_a1.py` $0.30 CPU smoke 2026-05-16 22:47:41Z)
returned `INDEPENDENT` verdict — MI=0.006385 bits/symbol vs 0.5 MEANINGFUL_
CONDITIONING threshold (2 orders of magnitude below; Wyner-Ziv gain ceiling
fraction 0.000907). The verdict was captured in
`.omx/research/atw_codec_v2_d4_probe_verdict_20260516_codex.md` + sister
`.omx/research/atw_d4_probe_recipe_disambiguation_20260516.md` but the
apparatus had no structural mechanism to refuse a future dispatch wrapper that
targeted `substrate_atw_codec_v2_modal_a100_dispatch.yaml`. The lane's recipe
correctly carries `dispatch_enabled: false` + `research_only: true` via
Catalog #240 (recipe-vs-trainer chain), but that gate fires at the
recipe-vs-trainer-state surface, not at the predecessor-probe-outcome
surface. The two surfaces are orthogonal: the recipe could be flipped to
`dispatch_enabled: true` AND the predecessor blocker would still apply.

## Canonical-vs-unique decision per layer

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" the design
question is: *"What's the OPTIMAL ENGINEERING for THIS specific method to
achieve the lowest score possible given the methods and techniques involved?"*
For an apparatus-maintenance / coherence gate (NOT a substrate codec), the
optimal engineering is to ADOPT the canonical Catalog #245 fcntl-locked JSONL
pattern because the bug class is structurally identical (scattered evidence
with no queryable index).

| Layer | Decision | Rationale |
|---|---|---|
| fcntl-locked JSONL storage | **ADOPT canonical** (Catalog #245 pattern) | Bug class structurally identical; reusing the proven pattern serves rather than suppresses. |
| HISTORICAL_PROVENANCE APPEND-ONLY | **ADOPT canonical** (Catalog #110/#113/#132) | Probe-outcome lifecycle (adjudicated → ratified → operator_override) is the same semantic shape as call_id lifecycle (dispatched → harvested → failed). |
| strict-load helper (`load_outcomes_strict`) | **ADOPT canonical** (Catalog #138 pattern) | Same fail-closed corruption-detection contract; same quarantine semantics. |
| schema fields | **FORK** (probe-specific fields: verdict / metric_name / metric_value / threshold / threshold_token / blocker_status / expires_at_utc) | Probe-outcome semantic is OPERATIONALLY DIFFERENT from call_id semantic (verdict + threshold ≠ rc + score) so the schema must encode the probe-specific contract. |
| event-type taxonomy | **FORK** (5 events: adjudicated / ratified / superseded / expired / operator_override) | Probe-outcome lifecycle has different transitions than Modal call_id lifecycle (dispatched / harvested / failed / stale / manually_terminated). |
| verdict taxonomy | **FORK** (7 verdicts: INDEPENDENT / KILL / DEFER / PROMOTE / PROCEED / PARTIAL / OPERATOR_REVIEW_REQUIRED) | Domain-specific; mirrors the existing probe-disambiguator outputs across the 60+ memos. |
| blocker_status auto-derivation | **FORK** (auto-blocking for {INDEPENDENT, KILL, DEFER}; auto-advisory for others) | Encodes the CLAUDE.md "Forbidden premature KILL" non-negotiable structurally into the helper. |
| expires_at_utc staleness window | **FORK with canonical default** (default = DEFAULT_STALENESS_WINDOW_DAYS = 30) | 30 days matches Catalog #298 L1 substrate retirement window per CLAUDE.md "Substrate retirement discipline" non-negotiable. |
| query API (`query_by_probe_id` / `query_by_substrate` / `query_by_recipe` / `latest_blocking_outcome_by_recipe` / `latest_blocking_outcome_by_substrate`) | **FORK** (probe-specific query helpers) | Three orthogonal lookup keys (probe_id / substrate / recipe_path) are all canonically valid identity surfaces; Modal call_id ledger only needs call_id + lane_id. |
| operator-authorize wire-in pattern | **ADOPT canonical** (Catalog #243 + #271 pattern) | Same insertion point between Catalog #152 + Catalog #243; same paired-env bypass per Catalog #199. |
| Catalog #131 sister-discipline registration | **ADOPT canonical** | PROBE_OUTCOMES_LEDGER_PATH added to `_SHARED_STATE_PATH_MARKERS` + `register_probe_outcome` etc. added to `_BARE_WRITE_CANONICAL_HELPER_CALL_TOKENS`. |
| STRICT preflight gate scope | **FORK with canonical pattern** (scan .py + .sh in tools/scripts/experiments/src/tac for dispatch tokens + recipe path literals; consult ledger via `latest_blocking_outcome_by_recipe`) | The gate's bug class (dispatch wrapper bypasses canonical entry point) is unique to this surface; the scan pattern + waiver semantics mirror sister gates. |

## 9-dimension success checklist evidence

1. **UNIQUENESS (class-shift not within-class)**: NEW CLASS-SHIFT — this gate
   operates at the **apparatus-coherence** surface (preventing re-execution of
   adjudicated probes); orthogonal to all existing dispatch-flow gates
   (#152 required-input / #243 local pre-deploy / #271 codex review / #240
   recipe-vs-trainer-chain). The new shared-state path
   `.omx/state/probe_outcomes.jsonl` is the empirical evidence of the
   class-shift (no prior path served this purpose).
2. **BEAUTY + ELEGANCE (30-sec-reviewable)**: each layer fits the canonical
   pattern: helper module = ~750 LOC mirroring exemplar; CLI = ~230 LOC
   single-file argparse + library API; STRICT gate = ~290 LOC single function
   with docstring; runtime wire-in = ~90 LOC single function. All 4 surfaces
   reviewable in 30 seconds against the Catalog #245 exemplar.
3. **DISTINCTNESS**: clearly differs from sister gates (#152 / #243 / #271 /
   #240) at the bug-class surface (predecessor-probe-outcome ≠ required-input
   / pre-deploy / codex-review / recipe-state).
4. **RIGOR**: premise verification per Catalog #229 (6 PVs before edit:
   Catalog #245 exemplar exists; Catalog #131/#138/#176/#185/#292 sister
   gates exist; operator_authorize.py insertion pattern exists; PreflightError
   pattern exists; lane registry pre-registration discipline; commit
   serializer with --expected-content-sha256). Adversarial-review: this memo
   is being written as the council-style deliberation surface; the
   Assumption-Adversary analysis is in frontmatter.
5. **OPTIMAL ENGINEERING PER TECHNIQUE** (Catalog #290 sister): documented
   per-layer in the canonical-vs-unique-decision table above.
6. **STACK-OF-STACKS COMPOSABILITY**: ADDITIVE with Catalog #245 (Modal call_id
   ledger) + #131 (no bare writes) + #138 (strict-load) + #152 (required-input)
   + #243 (local pre-deploy) + #271 (codex review) + #240 (recipe-vs-trainer
   chain). All sisters fire at orthogonal surfaces; cumulative effect is the
   FULL dispatch-flow protection envelope per CLAUDE.md "Operator gates must
   be wired and used" non-negotiable.
7. **DETERMINISTIC REPRODUCIBILITY**: JSONL byte-stable via
   `json.dumps(sort_keys=True)`; atomic-replace via `.tmp.<uuid12>` +
   `os.replace`; 4-proc spawn-pool concurrent-append regression test pins
   the deterministic semantics.
8. **EXTREME OPTIMIZATION + PERFORMANCE**: single-row append <10ms; query
   helpers O(N) over current ledger (which will stay small — ~1 row per
   probe-adjudication; ~60 probes today + ~10/week growth implies <2 KB/year
   storage; LOCK_TIMEOUT_SECONDS=30s is generous even at 10x fan-out).
   The 30-second budget for `preflight_all()` is preserved.
9. **OPTIMAL MINIMAL CONTEST SCORE**: this gate is INFRASTRUCTURE for
   contest score-lowering — it does not directly lower score, but it
   structurally prevents wasted paid dispatch on already-adjudicated probes,
   redirecting GPU budget toward the canonical research path per Catalog #308
   (alternative-reducer enumeration before class-wide kill).

## Observability surface (Catalog #305)

1. **Inspectable per layer**: every layer is a single .py file with a
   clear API surface; `cat .omx/state/probe_outcomes.jsonl` shows every event.
2. **Decomposable per signal**: every probe-outcome has metric_name +
   metric_value + threshold + threshold_token + verdict + blocker_status;
   future probes that emit composite metrics can attach extra fields per the
   `**extra` kwarg.
3. **Diff-able across runs**: JSONL is line-stable (sort_keys=True per row);
   diff of two ledgers is meaningful at the line level.
4. **Queryable post-hoc**: `tools/check_predecessor_probe_outcome.py
   --list-blocking --json` for machine consumption;
   `query_by_probe_id / query_by_substrate / query_by_recipe /
   latest_blocking_outcome_by_recipe / latest_blocking_outcome_by_substrate`
   for library callers.
5. **Cite-able**: every row carries `evidence_path` (memo) + `adjudicated_at_utc`
   + `agent` + `subagent_id` + `notes`; the canonical helper's `_now_iso`
   stamps `written_at_utc` + `written_pid` + `written_host` so forensic
   reconstruction is byte-stable.
6. **Counterfactual-able**: the 4-proc spawn-pool stress test in
   `test_probe_outcomes_ledger.py::test_concurrent_append_4proc_spawn_pool`
   establishes the counterfactual that the canonical ledger preserves all 20
   concurrent appends; deleting the ledger and re-registering the 2
   backfilled anchors reproduces the live state byte-for-byte.

## Files touched

- `src/tac/probe_outcomes_ledger.py` (NEW; ~750 LOC; canonical helper Layer 1)
- `tools/check_predecessor_probe_outcome.py` (NEW; ~230 LOC; operator CLI Layer 2)
- `src/tac/preflight.py` (EDITED; +~310 LOC; STRICT gate + Catalog #131 sister
  registrations + wire-in in `preflight_all()` — Layer 3)
- `tools/operator_authorize.py` (EDITED; +~90 LOC; runtime hook + call site
  between Catalog #152 + #243 — Layer 4)
- `src/tac/tests/test_probe_outcomes_ledger.py` (NEW; ~530 LOC; 50 tests)
- `src/tac/tests/test_check_313_dispatch_target_no_predecessor_probe_outcome.py`
  (NEW; ~480 LOC; 27 tests)
- `CLAUDE.md` (EDITED; +Catalog #313 row + Catalog #292 amendment)
- `.omx/state/probe_outcomes.jsonl` (NEW; 2 anchors backfilled)
- `.omx/state/lane_registry.json` (EDITED; lane pre-registered at L0)
- `.omx/research/probe_outcomes_canonical_ledger_landed_20260516.md` (THIS FILE)

## 6-hook wire-in per Catalog #125

1. **Sensitivity-map contribution**: N/A — this gate is apparatus-coherence,
   not a sensitivity primitive.
2. **Pareto constraint**: ACTIVE — predecessor-probe-outcome IS a feasibility
   constraint on the dispatch frontier (the apparatus REFUSES to spend on
   already-adjudicated probes; constraint binds at gate-time).
3. **Bit-allocator hook**: N/A — not a per-tensor importance primitive.
4. **Cathedral autopilot dispatch hook**: ACTIVE PRIMARY — the canonical
   `tac.probe_outcomes_ledger.query_blocking_outcomes` is consumable by the
   autopilot ranker as a "do not propose this candidate" filter. Wire-in to
   `tools/cathedral_autopilot_autonomous_loop.py` is operator-routable as a
   follow-up subagent landing (the canonical helper API is stable; consumption
   can be incremental).
5. **Continual-learning posterior update**: ACTIVE — the ledger IS the
   continual-learning posterior for probe-disambiguator outcomes. Every
   adjudicated probe is one update to the posterior.
6. **Probe-disambiguator**: ACTIVE — this gate IS the probe-disambiguator
   wire-in surface; future probe-disambiguator tools at
   `tools/probe_*_disambiguator.py` should call
   `tac.probe_outcomes_ledger.register_probe_outcome` to land their verdicts
   in the canonical ledger.

## Operator-facing usage example

```bash
# After running a new probe-disambiguator and producing a verdict:
.venv/bin/python -c "
from tac.probe_outcomes_ledger import register_probe_outcome
register_probe_outcome(
    probe_id='atw_v3_per_region_class_histogram_reducer_20260520',
    substrate='atw_codec_v2',  # sister attempt with alternative reducer per Catalog #308
    recipe_path='.omx/operator_authorize_recipes/substrate_atw_codec_v2_modal_a100_dispatch.yaml',
    probe_kind='per_region_class_histogram_reducer',
    verdict='PROMOTE',  # alternative reducer found signal
    metric_name='per_region_mutual_information_bits',
    metric_value=0.85,
    threshold=0.5,
    threshold_token='MEANINGFUL_CONDITIONING',
    evidence_path='.omx/research/atw_v3_per_region_class_histogram_verdict_20260520.md',
    next_action='proceed_to_atw_v3_phase_2_dispatch',
    agent='codex',
)
"

# Query before any dispatch:
.venv/bin/python tools/check_predecessor_probe_outcome.py \
    --recipe .omx/operator_authorize_recipes/substrate_atw_codec_v2_modal_a100_dispatch.yaml

# Operator dashboard - all blocking outcomes:
.venv/bin/python tools/check_predecessor_probe_outcome.py --list-blocking

# Operator override (rare; paired-env required per Catalog #199):
OPERATOR_AUTHORIZE_PROBE_PREDECESSOR_BYPASS_VERDICT=1 \
OPERATOR_AUTHORIZE_PROBE_PREDECESSOR_BYPASS_RATIONALE="ratified by T3 council 2026-05-20; fresh evidence per atw_v3 per-region-class-histogram reducer" \
    .venv/bin/python tools/operator_authorize.py \
    --recipe substrate_atw_codec_v2_modal_a100_dispatch.yaml
```

## Cross-references

- Catalog #245 (canonical Modal call_id ledger 4-layer exemplar)
- Catalog #131 (no bare writes to shared state — PROBE_OUTCOMES_LEDGER_PATH registered)
- Catalog #138 (strict-load discipline — `load_outcomes_strict` mirrors)
- Catalog #292 (per-deliberation assumption surfacing — amended with Catalog #313 standing question)
- Catalog #240 (recipe-vs-trainer chain — sister at dispatch-flow surface)
- Catalog #243 (local pre-deploy harness — adjacent insertion point)
- Catalog #271 (codex pre-dispatch review — adjacent insertion point)
- Catalog #167 (smoke-before-full pattern)
- Catalog #199 (paired-env operator bypass discipline — bypass mechanism)
- Catalog #298 (substrate retirement discipline — 30-day staleness window default shared)
- Catalog #308 (kill memos enumerate alternative probe methodologies — referenced in error message)
- CLAUDE.md "Forbidden premature KILL without research exhaustion" non-negotiable
- CLAUDE.md "Operator gates must be wired and used" non-negotiable
- CLAUDE.md "Meta-Lagrangian/Pareto solver — NON-NEGOTIABLE" probe-disambiguator pattern
- CLAUDE.md "Bugs must be permanently fixed AND self-protected against" non-negotiable

## Empirical receipts

- Catalog #313 STRICT preflight gate live count at landing: **0**
- Catalog #131 (canonical helper exclusion + path markers): **0**
- Catalog #176 (CLAUDE.md catalog row): **0**
- Catalog #118 (no duplicate numbers): **0**
- Catalog #159 (catalog text matches strict value): **0**
- Catalog #185 (LIVE_COUNT=0 honored across catalog rows claiming `live count: 0`): **0**
- Catalog #292 (per-deliberation assumption surfacing): **0**
- Tests pass: **50/50** in `test_probe_outcomes_ledger.py` + **27/27** in
  `test_check_313_dispatch_target_no_predecessor_probe_outcome.py` = **77/77**
- Backfilled anchors: **2** (ATW v2 D4 INDEPENDENT + Wunderkind G1 v2 DEFER)
- GPU spend: **$0** (CPU-only landing; no dispatch fired)

## Lane

`lane_probe_outcomes_canonical_ledger_bake_in_20260516` — Level 1 SCAFFOLD
(impl_complete=true; strict_preflight=true; memory_entry=true). The lane
is `lane_class=substrate_engineering` because it lands canonical
infrastructure (not a substrate codec); exempt from Catalog #220 / #272
operational-mechanism requirements by construction.

## Subagent discipline checklist

- [x] Catalog #229 premise verification BEFORE editing (6 PVs confirmed)
- [x] Catalog #186 catalog # claim via canonical serializer (`#313`)
- [x] Catalog #126 lane pre-registered at L0 BEFORE work
- [x] Catalog #206 checkpoint discipline (3 checkpoints written)
- [x] Catalog #230 sister-subagent ownership map honored (Phase 1c-Rudin + Z6 Phase 2 untouched)
- [x] Catalog #248 no conflict markers introduced
- [x] Catalog #185 LIVE_COUNT=0 verified before claiming
- [x] Catalog #176 strict-callsite has CLAUDE.md row landed atomically
- [x] Catalog #159 catalog text says STRICT (matches `strict=True` wire-in)
- [x] 6-hook wire-in declared (4 ACTIVE + 2 N/A with rationale)
- [x] CLAUDE.md "Mission alignment" frontmatter (3 v2 fields present)
- [x] CLAUDE.md "Observability surface" section per Catalog #305

## Next op-routables

1. **Backfill additional historical probe outcomes** — scan the remaining
   ~58 probe-related memos under `.omx/research/*probe*.md` /
   `*verdict*.md` / `*disambig*.md` and register canonical adjudicated rows
   to the ledger. Ambiguous outcomes register with `OPERATOR_REVIEW_REQUIRED`
   rather than guessing per the parent prompt directive.
2. **Cathedral autopilot wire-in** — extend
   `tools/cathedral_autopilot_autonomous_loop.py` to consume
   `tac.probe_outcomes_ledger.query_blocking_outcomes` as a "do not propose"
   filter on the candidate queue. The canonical helper API is stable.
3. **Probe-disambiguator tools should land verdicts via the canonical
   helper** — sister subagents producing
   `tools/probe_<name>_disambiguator.py` should call
   `register_probe_outcome(...)` after writing the verdict memo. Operator-
   facing documentation update needed.
