# Integration Gap Audit (Dimension I) — 2026-05-12

**Pass**: FFF — Wiring + Integration + Arbitrariness (W/I/A) sweep
**Scope**: sibling-tool composition gaps (`tac.cost_band_calibration` ↔ wrappers, `tac.continual_learning` ↔ autopilot, sensitivity_map / pareto / lane_maturity evidence-pointer integrity)
**Author**: subagent (FFF pass), date 2026-05-12

## Methodology

For each known integration contract (canonical sibling-tool pair), grep for:
1. Consumer importing the producer
2. Consumer actually calling the producer's emitting function
3. Producer's output being durable enough to be consumed later (anchor write to disk)

Pairs reviewed:
- `tac.cost_band_calibration.predict()` → operator_authorize wrappers (cost band freshness)
- `tac.continual_learning.posterior_update_locked()` → dispatch harvesters (anchor flow)
- `tac.cost_band_calibration` + `tac.continual_learning` → `tools/cathedral_autopilot*` (autopilot reads posteriors)
- `tac.sensitivity_map.*` → trainers/codecs (sensitivity contribution)
- `tac.pareto_*` → autopilot ranking (Pareto intersection)
- `tools/lane_maturity.py` evidence pointers → on-disk file existence (Catalog #90 covers, but spot-check)

## Headline counts

| Bridge | Producer | Consumer | Status |
|---|---|---|---|
| operator_authorize_phase1_t1_balle wrapper reads cost-band | `tac.cost_band_calibration` | `scripts/operator_authorize_phase1_t1_balle_*.sh` | WIRED (post-OD-CB-1) |
| operator_authorize_kaggle_t1 reads cost-band | `tac.cost_band_calibration` | `scripts/operator_authorize_kaggle_t1_balle_sweep.sh` | WIRED |
| operator_authorize_scpp_stage1 reads cost-band | `tac.cost_band_calibration` | `scripts/operator_authorize_scpp_stage1_anchor_dispatch.sh` | **NOT WIRED** (uses `~$3` literal in docstring) |
| operator_authorize_t10_ib_lagrangian reads cost-band | `tac.cost_band_calibration` | `scripts/operator_authorize_t10_ib_lagrangian_dispatch.sh` | **NOT WIRED** (uses `~$40` literal in docstring) |
| cathedral_autopilot loop reads continual-learning posterior | `tac.continual_learning` | `tools/cathedral_autopilot_autonomous_loop.py` | **DOCSTRING-CLAIM ONLY** (no actual import) |
| cathedral_autopilot loop reads cost-band posterior | `tac.cost_band_calibration` | `tools/cathedral_autopilot_autonomous_loop.py` | **NOT WIRED** |
| cathedral_autopilot reads continual-learning posterior | `tac.continual_learning` | `tools/cathedral_autopilot.py` | **NOT WIRED** |
| cathedral_autopilot reads cost-band posterior | `tac.cost_band_calibration` | `tools/cathedral_autopilot.py` | **NOT WIRED** |
| meta-Lagrangian bridge wires both posteriors | `tac.continual_learning`, `tac.cost_band_calibration` | `tools/cathedral_autopilot_meta_lagrangian_bridge.py` | UNKNOWN — needs inspect |

## Detailed findings

### I-1 (HIGH): autopilot autonomous loop claims posterior wire-in but lacks the import

**File**: `tools/cathedral_autopilot_autonomous_loop.py`

**Docstring (line 70)** says: `:mod:`tac.continual_learning` — posterior consumed and updated by the loop`

**Reality**:
- `grep "from tac.continual_learning\|posterior_update_locked"` → **0 matches** in the loop file
- The constant `POSTERIOR_REWEIGHT = "posterior_reweight"` is defined at line 126 — never referenced elsewhere
- The actual imports list at lines 85-100 includes `tac.optimization.substrate_composition_matrix` etc. but NOT `tac.continual_learning`

**Bug class**: docstring-without-implementation. The autopilot will rank candidates without ever looking at empirical-anchor posteriors, contrary to its stated contract per CLAUDE.md "Subagent coherence-by-default" rule "anti-fragmentation primitive: the unified-Lagrangian action".

**Fix size**: small (10-30 LOC). Add `from tac.continual_learning import load_posterior` + an explicit reweighting step in the ranking loop. **BUT** — this is a design decision per CLAUDE.md "Design decisions — non-negotiable" because the reweighting math (how posterior anchors should reweight predicted Δ-score) is a council-level choice (additive Bayesian update? trust-region tightening? rejection prior?). **Surface for operator decision.**

**Workaround if not landed**: correct the docstring to say "posterior consumed via offline reseed step (operator-invoked)" so future agents don't believe the wire-in exists.

### I-2 (HIGH): two operator_authorize wrappers carry hardcoded $-literal costs without cost-band wire-in

**Files**:
- `scripts/operator_authorize_scpp_stage1_anchor_dispatch.sh:24` — `Cost: ~$3 Modal T4 (3 epochs smoke; conservative)`
- `scripts/operator_authorize_t10_ib_lagrangian_dispatch.sh:23` — `Cost: ~$40 on Modal T4 (8 hours @ ~$5/hr; conservative)`

**Bug class**: the OD-CB-1 + cost-band-calibration landing (commit `5eb355aa`) promoted operator_authorize wrappers to read `tac.cost_band_calibration.predict('modal', 'T4', est_epochs)` rather than embed stale literals. Two wrappers (`phase1_t1_balle`, `kaggle_t1_balle`) got the upgrade. Two wrappers (`scpp_stage1`, `t10_ib_lagrangian`) DID NOT — they still carry hardcoded literals. Per the same CLAUDE.md "F1 HIGH wrapper reads cost-band posterior, supersedes stale $0.59 literal" rationale, the SAME upgrade applies.

**Fix size**: small per wrapper (10-30 LOC each). Each is a council-trivial follow-up to OD-CB-1.

**Surfacing**: this is the kind of bridge that should land in the next cleanup pass. It's NOT in this audit's "land now" set because:
1. SC++ Stage 1 is the active substrate-engineering dispatch; concurrent subagents may have a hand on its wrapper (Subagent 5 lives near this file, and so does Cluster 1 with modal_train_lane work)
2. T10 IB Lagrangian dispatch costs $40 — the calibration data for T10 (long training jobs > 8 h) may not yet be in the posterior (Modal cold-start posterior has 21 anchors but mostly 3-epoch smoke runs)
3. Until the posterior has anchors at T10's wall-clock scale, the band would return wide/conservative — calling out "we updated to use the posterior but the posterior says $20-$60" is honest but less useful than the operator-set $40

**Proposed action**: SURFACE-FOR-OPERATOR-DECISION (do we backport the OD-CB-1 cost-band pattern to these two wrappers now, or wait for T10/SC++ wall-clock anchors to populate the posterior?)

### I-3 (MEDIUM): cathedral_autopilot (top-level, not loop) has no posterior imports

**Files**: `tools/cathedral_autopilot.py` (113.8K, the main autopilot driver)

Confirmed: `grep "cost_band\|continual_learning"` returns **0 matches** in `tools/cathedral_autopilot.py`.

This is sister to I-1 — the autopilot makes ranking decisions without consulting either the cost-band posterior OR the continual-learning anchor posterior. The bridge module `tools/cathedral_autopilot_meta_lagrangian_bridge.py` may delegate the responsibility, but the call chain has not been verified.

**Proposed action**: SURFACE — same design-decision posture as I-1. The bridge module needs an explicit responsibility audit. Defer fix to a future pass that has council authority.

### I-4 (LOW): sensitivity-map wire-in for `iter_layer_pairs`

`src/tac/frozen_bit_quant.py:294 iter_layer_pairs` was flagged in the wiring audit (W-1 #6). It produces an iterator of layer pairs that the bit-allocator could consume. Currently no consumer.

**Proposed action**: DEFERRED-pending-bit-allocator-usecase. Don't delete (CLAUDE.md "KILL is LAST RESORT"); don't force-wire (no clear consumer demand). Surface in next bit-allocator design pass.

### I-5 (NOT-A-BUG): lane_maturity evidence pointers verified by Catalog #90 STRICT

Catalog #90 (`check_lane_registry_consistent`) STRICT covers this end-to-end. Spot-check confirms `.omx/state/lane_registry.json` parses, no duplicate ids, all 7-gate metadata present, and file-path evidence pointers resolve on disk. Caveat: `tools/lane_maturity.py validate` is the canonical authority — re-run before any commit that adds a lane.

## Recommended actions

**LAND NOW (trivial fixes)**: NONE. Every finding is either:
- A design decision (I-1, I-3) requiring operator/council
- Concurrent-subagent-write surface (I-2 touches `scripts/operator_authorize_*.sh`, which Catalog #152 subagent owns)
- Already covered by an existing STRICT check (I-5 via Catalog #90)

**SURFACE FOR OPERATOR DECISION**:
1. I-1: autopilot loop posterior wire-in (math + import; council-level)
2. I-2: backport OD-CB-1 cost-band pattern to `scpp_stage1` + `t10_ib_lagrangian` wrappers
3. I-3: cathedral_autopilot top-level posterior consumption (sister to I-1)

**DEFERRED**:
- I-4: `iter_layer_pairs` consumer-pending

## Wire-in hook declarations (per CLAUDE.md Catalog #125)

1. **Sensitivity-map**: N/A — this audit identifies missing bridges, doesn't add one.
2. **Pareto constraint**: N/A — same.
3. **Bit-allocator**: identified gap (I-4) but no fix lands.
4. **Cathedral autopilot dispatch hook**: relevant — I-1 + I-3 identify the missing wire-in. No fix lands (council-level).
5. **Continual-learning posterior**: relevant — I-1 + I-3 identify the missing reader. No fix lands.
6. **Probe-disambiguator**: N/A — no math arbitration questions.

## References

- `feedback_modal_strategy_reevaluation_post_tier1_engineering_20260512.md` (cost-band rebuild context)
- CLAUDE.md "Subagent coherence-by-default" — the 6-hook unified-Lagrangian wire-in non-negotiable
- CLAUDE.md "Meta-Lagrangian/Pareto solver" — orphan work is a coherence bug
- commit `5eb355aa` — OD-CB-1 cost-band landing (pattern to backport)
- commit `0666720a` — F1 HIGH fix that proved the cost-band wire-in pattern
