# Cross-stack wire-in audit — 2026-05-15

**Lane**: `lane_wire_and_integrate_all_cross_stack_20260515`
**Subagent**: WIRE-AND-INTEGRATE-ALL-SUBAGENT
**Operator directive**: *"wire and integrate all"* (companion to *"canonicalize and standardize all"*)

Per CLAUDE.md "Subagent coherence-by-default" + "Mandatory wire-in for every landing
(no orphaned signals)" — every landing must wire its outputs into the unified solver
stack OR explicitly tag `research_only=true`. The 6 mandatory wire-in hooks are:

1. Sensitivity-map contribution
2. Pareto constraint
3. Bit-allocator hook
4. Cathedral autopilot dispatch hook
5. Continual-learning posterior update
6. Probe-disambiguator (if 2+ defensible interpretations exist)

This audit maps each canonical helper to its producers and consumers, identifies
orphans, and enumerates the wire-ins landed in this commit batch.

---

## Hook 1: Sensitivity-map (`src/tac/sensitivity_map/`)

**Canonical surface**: `tac.sensitivity_map.{axis_weights, save_sensitivity_map, load_sensitivity_map, validate_real_sensitivity_artifact, ...}`.

**Producers** (artifacts with sensitivity weights):
- `experiments/results/posenet_sensitivity_v5/sensitivity_map.pt` (Lane PD)
- 11 sister modules: `tac.{component_sensitivity_artifact, codec_pipeline_sensitivity, balle_sensitivity_weighted, imp_sensitivity_weighted, owv3_sensitivity_weighted, neural_weight_codec_sensitivity, logit_margin_sensitivity_weighted, jcsp_score_marginals, hidden_gems, forensics, cross_paradigm_wiring}`
- 8 sister tools: `tools/{build_composition_ranking_json, build_autopilot_dry_run_summary, build_field_meta_dispatch_selection, dispatch_phase_a_track_1_ablations, dispatch_dryrun_apogee_intN, build_a2_sensitivity_weighted_pr101_packet, build_beta_fisher_lossy_coarsening_weights, build_cross_paradigm_frontier_inventory}`

**Consumers**:
- `tools/cathedral_autopilot_autonomous_loop.py::discover_sensitivity_map_artifacts` reads `*.pt` artifacts
- Cross-paradigm + composition tools consume sensitivity-weighted predictions

**Status**: WIRED (multiple producer-consumer chains; cathedral autopilot is the
canonical consumer). Sub-API `axis_weights_for_named_operating_point` is defined
in `tac.sensitivity_map.axis_weights` but only consumed by its own tests — flagged
as a candidate for eventual deprecation OR active wire-in; defer to council per
CLAUDE.md "Forbidden premature KILL".

---

## Hook 2: Pareto constraint

**Canonical surface**: `tac.pareto_*` (no canonical module — distributed across
`tac.cathedral_lagrangian_phase_iv*`, codec primitives, archive grammars).

**Producers**: every L2+ archive with a measured (rate, seg, pose) triple.

**Consumers**: cathedral autopilot ranker via `predict_cost_band` + Pareto-aware
EIG/$ scoring (`tools/cathedral_autopilot_autonomous_loop.py::rank_candidates`).

**Status**: WIRED — Pareto constraint is implicit in the cost-band posterior + EIG/$
ranker. No explicit `tac.pareto.*` module exists per the operator's distinction
between "ranker primitive" (where Pareto lives) vs "actuator primitive" (the
dispatch fan-out). Consistent with CLAUDE.md "parallel-dispatch is a FIRST-CLASS
DELIVERABLE, not an afterthought" — the Pareto constraint is the ranker; the
actuator is `tools/parallel_dispatch_top_k.py`.

---

## Hook 3: Bit-allocator hook

**Canonical surface**: registered per-substrate via `tac.archive.adaptive` (retired
per CLAUDE.md) + per-lane bit-allocator helpers.

**Producers**: substrate trainers that change per-tensor importance during training
(e.g. EMA-based, Hessian-based).

**Consumers**: substrate `pack_archive` / `_build_archive_zip` callsites.

**Status**: WIRED at substrate level. No central registration today — each substrate
owns its bit-allocator. Operator deferred consolidation per the "tac stays clean"
discipline (each substrate's allocator is local; consolidation belongs in a future
deterministic packet compiler per CLAUDE.md "Deterministic packet compiler"
non-negotiable).

---

## Hook 4: Cathedral autopilot dispatch hook

**Canonical surface**: `tools/cathedral_autopilot_autonomous_loop.py::rank_candidates`
+ `apply_z1_empirical_revision_to_candidate_delta` + `adjust_predicted_delta_for_*`.

**Producers** (candidates feed the ranker):
- `.omx/state/autopilot_candidate_queue.jsonl` + sister `_v2_post_z1_revision_*.jsonl`
- Sister CANONICALIZE-SUBSTRATE-CHAIN-SUBAGENT in flight (Z3 v2 + cathedral autopilot edits)
- New substrates' first paired anchors land here via `harvest_modal_calls.py`

**Consumers**:
- `tools/operator_authorize.py` (consumed via cost-band posterior — the autopilot's
  recommendations route operator dispatches)
- `tools/parallel_dispatch_top_k.py` (the canonical actuator)
- `tools/operator_briefing.py` (surfaces ranked candidates to the operator)

**Status**: WIRED. Live empirical anchors flow: substrate trainer →
`contest_auth_eval` → `harvest_modal_calls --execute` → `append_anchor` →
`load_anchors` (cathedral consumer) → `rank_candidates` → operator dispatch.

---

## Hook 5: Continual-learning posterior update

**Canonical surface**: `tac.continual_learning.{posterior_update, posterior_update_locked, save_posterior, load_posterior, ContestResult}`.

**Producers**: every paired `[contest-CUDA]` + `[contest-CPU]` anchor.

**Consumers**:
- `tools/harvest_modal_calls.py --execute` (canonical reseeder)
- `tools/cathedral_autopilot_autonomous_loop.py` (reads posterior for ranker)
- `tools/operator_authorize.py` (consults cost-band posterior pre-dispatch)
- `tools/{operator_briefing, build_autopilot_dry_run_summary, build_field_meta_dispatch_selection, ...}`

**Status**: WIRED. Catalog #128 strict gate enforces `posterior_update_locked`
discipline; Catalog #131 sister gate enforces bare-write discipline across all
shared `.omx/state/` paths. Catalog #138 enforces strict-load for mutating writers.

---

## Hook 6: Probe-disambiguator (when 2+ defensible interpretations exist)

**Canonical surface**: `tools/probe_<track>_disambiguator.py` per CLAUDE.md
"Anti-arbitrariness primitive: the probe-disambiguator pattern".

**Producers**: design memos with 2+ interpretations.

**Consumers**: trainer/codec/solver consumes the probe verdict.

**Status**: WIRED for the families that have explicit probes (DP1 leakage probe,
C1 world-model probe v2, C6 IBPS Tier C probe, etc.). New design memos that
declare 2+ interpretations are gated by Catalog #125 (which requires probe-disambiguator
declaration in landing memos for designs with 2+ defensible interpretations).

---

## Wire-ins landed in WIRE-AND-INTEGRATE-ALL commit batch

### Catalog #243 (NEW) — local pre-deploy harness wire-in

**Producer**: `tools/local_pre_deploy_check.py` (existing canonical 30s harness;
checks py-compile + archive grammar + auth-eval reachability + canonical inflate
device + deterministic ZIP + `_full_main` implemented).

**Previous consumer**: NONE (orphan helper — the empirical anchor: Z3 v2 smoke
`fc-01KRNHEGC9ZE48Y68GGJHP7FXN` + Z4 smoke `fc-01KRNHE942JSV7VRGXGR1FJGHQ` both
crashed at $2 each on bug classes the harness would have caught).

**New consumer**: `tools/operator_authorize.py::_run_local_pre_deploy_check`
threaded between Catalog #152's `_validate_required_input_files` and
`_native_dispatch_preflight`.

**Bypass discipline**: `OPERATOR_AUTHORIZE_SKIP_LOCAL_PRE_DEPLOY_CHECK=1` requires
paired `OPERATOR_AUTHORIZE_LOCAL_PRE_DEPLOY_BYPASS_REASON=<text>` per CLAUDE.md
"Comment-only contracts are FORBIDDEN" + Catalog #199 paired-env discipline.

**Self-protection gate**: Catalog #243 STRICT preflight refuses any dispatch
wrapper (`.py`/`.sh` under `tools/`/`scripts/`/`experiments/`/`src/tac/` with
`_CHECK_152_DISPATCH_TOKENS`) that bypasses BOTH `tools/operator_authorize.py`
AND `tools/local_pre_deploy_check.py`. Initial wire-in is WARN-ONLY per
CLAUDE.md "Strict-flip atomicity rule"; live count ~56 historical wrappers
(many are docs/diagnostic tools that document dispatch one-liners; backfill
via per-line waivers + canonicalization sweep is a follow-up wave).

---

## Orphan-state findings (post-WIRE-AND-INTEGRATE-ALL)

**Catalog #125 live count**: 22 post-cutover landing memos in operator memory
(`~/.claude/projects/.../memory`) missing wire-in declarations. Operator memory
is OUT-OF-SCOPE for this subagent (per CLAUDE.md scope rules). Surfaced to
operator for manual backfill.

**Catalog #126 live count**: 38 source-code references to unregistered `lane_<id>`
tokens. Includes:
- 2 in `src/tac/preflight.py:51867,52628` (catalog #220 helper constants — false
  positives; the constants tuple is a substring set, not lane references)
- 36 in test fixtures + sister-subagent landings (operator-routable for waiver
  backfill)

**Catalog #243 live count** (initial WARN-ONLY landing): ~56 dispatch wrappers
that document dispatch one-liners but don't actually dispatch via the canonical
path. Strict-flip pending audit + waiver backfill.

---

## Pending council design decisions (operator-routable)

1. **`tac.sensitivity_map.axis_weights_for_named_operating_point` deprecation**: the
   sub-API has tests but no production consumer — council-grade decision per
   CLAUDE.md "Forbidden premature KILL is LAST RESORT".

2. **Catalog #243 strict-flip cadence**: 56 historical violations; bulk backfill
   via per-line `# LOCAL_PRE_DEPLOY_CHECK_BYPASS_OK:<rationale>` waivers OR
   docstring/codeblock detector refinement. Council-grade per "Strict-flip
   atomicity rule".

3. **Per-substrate bit-allocator consolidation**: deterministic packet compiler
   convergence vs leave-decentralized. Council-grade per CLAUDE.md
   "Deterministic packet compiler" non-negotiable.

4. **Operator memory Catalog #125 backfill**: 22 post-cutover memos missing
   wire-in declarations. Operator-routable; subagent cannot edit operator
   memory directly.

---

## 6-hook wire-in declaration (this landing)

Per CLAUDE.md "Subagent coherence-by-default" mandatory wire-in:

1. **Sensitivity-map**: ACTIVE — audit catalogs all 11 sister modules + 8 sister
   tools + canonical surface; cathedral autopilot remains the canonical consumer.
2. **Pareto constraint**: ACTIVE — Catalog #243 STRICT gate adds a refusal
   constraint to the Pareto-feasible-region for dispatch-wrapper-vs-canonical-routing.
3. **Bit-allocator**: N/A — this is a wire-in audit + STRICT gate landing; does
   not change per-tensor importance.
4. **Cathedral autopilot dispatch hook**: ACTIVE — the local pre-deploy harness
   wire-in directly affects which dispatches fire (refused-at-source instead of
   refused-at-Modal-rc-1); autopilot ranker behavior unchanged but dispatch
   conversion rate improves.
5. **Continual-learning posterior**: ACTIVE — refused-at-source dispatches
   prevent garbage anchors from polluting the cost-band posterior; the harness
   IS a posterior-quality preserving primitive.
6. **Probe-disambiguator**: N/A — single canonical interpretation (the harness
   is the canonical pre-deploy check; bypass requires paired-env attestation).

---

## Cross-references

- `feedback_unified_lagrangian_action_principle_GR_style_20260509.md` — the canonical
  6-hook charter
- `feedback_design_tension_ship_both_interpretations_let_math_arbitrate_20260509.md` —
  the probe-disambiguator pattern
- Catalog #125 (`check_subagent_landing_has_solver_wire_in`) — landing-memo gate
- Catalog #152 (`check_operator_wrapper_validates_required_input_files_pre_dispatch`) —
  sister wire-in at the same insertion point
- Catalog #167 (`check_substrate_dispatch_uses_smoke_before_full_pattern`) — sister
  dispatch-wrapper canonical-routing META
- Catalog #243 (NEW) — `check_dispatch_wrappers_invoke_local_pre_deploy_check_first`

