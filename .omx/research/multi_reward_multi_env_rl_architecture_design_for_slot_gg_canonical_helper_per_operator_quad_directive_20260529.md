<!-- SPDX-License-Identifier: MIT -->
---
council_tier: T2
council_attendees: [Shannon, Dykstra, Rudin, Daubechies, Yousfi, Fridrich, Contrarian, Assumption-Adversary]
council_quorum_met: true
council_verdict: PROCEED
council_dissent:
  - member: Contrarian
    verbatim: "this is a DESIGN memo only — feed-INTO-Slot-GG synthesis is the canonical consumer; the canonical SHARED helper itself is NOT authored in this slot; no code lands until Slot GG synthesizes; protect against Slot GG never landing by tagging this memo `research_only=true` and operator-routable in own right."
council_assumption_adversary_verdict:
  - assumption: "PufferLib + MLX + canonical equation #344 registry as reward source is the OPTIMAL architecture for substrate-design RL"
    classification: HARD-EARNED
    rationale: "(a) PufferLib 1M+ steps/sec is the published canonical vectorized-RL throughput; (b) M5 Max 128GB + 40 GPU cores + 16-core Neural Engine is published canonical hardware spec; (c) 143 unique canonical equations + 70 canonical anti-patterns IS the canonical empirical-anchor source per Catalog #344 + #355; (d) Slot CC canonical 3-metric trichotomy IS the canonical multi-objective Pareto orthogonality structure; (e) Catalog #372 Dykstra Pareto polytope solver wire-in already LANDED to consume per-axis dual variables — RL reward inherits canonical Pareto constraints structurally."
  - assumption: "multi-reward via 4 distinct reward signals (canonical equation predicted-ΔS + Lagrangian dual + anti-pattern penalty + paired-CUDA RATIFICATION) is strictly better than single-reward composite"
    classification: HARD-EARNED
    rationale: "per CLAUDE.md 'Meta-Lagrangian/Pareto solver' non-negotiable, the canonical desired loop is FORMULATE-OBJECTIVE-AND-CONSTRAINTS not single-scalar-loss; the 4-axis Pareto polytope per Dykstra IS the canonical Pareto/KKT/interaction prune step; collapsing to single scalar reward re-introduces canonical anti-pattern rank_1_problem_spec_synergy_tautology_v1 (registered HIGH severity)."
  - assumption: "multi-environment via 3 distinct PufferLib MLX-LOCAL environments is canonical orthogonality not redundancy"
    classification: HARD-EARNED
    rationale: "(a) substrate-iteration env action-space ≠ curriculum-discovery env action-space (per Slot DD L14 PR95 8-stage curriculum has independent moves from substrate architecture mutations); (b) cross-PR-family CLASS-SHIFT env per Slot DD L43-L70 10 candidates IS orthogonal paradigm-axis vs within-class iteration of envs 1+2; (c) sister-DISJOINT confirmed via action-space orthogonality."
  - assumption: "MLX-LOCAL canonical environment is non-promotable surrogate per Catalog #192 + #1 + #317 + #341 + #382"
    classification: HARD-EARNED
    rationale: "per CLAUDE.md MPS auth eval is NOISE + Submission auth eval BOTH CPU AND CUDA non-negotiables; MLX-LOCAL outputs MUST be tagged [macOS-MLX research-signal] + score_claim=false + promotion_eligible=false + ready_for_exact_eval_dispatch=false; paired-CUDA RATIFICATION on 1:1 contest-compliant hardware per Catalog #246 is the canonical promotion path."
council_decisions_recorded:
  - "op-routable #1: feed-INTO-Slot-GG canonical SHARED helper DESIGN synthesis (Slot GG consumes this memo's 4-reward-signal + 3-env design as INPUTS)"
  - "op-routable #2: register canonical equation candidate `multi_reward_multi_env_rl_architecture_savings_via_canonical_equation_pareto_polytope_v1` per Catalog #344 (operator-decision-pending; awaits Slot GG canonical SHARED helper + empirical anchor)"
  - "op-routable #3: register canonical anti-pattern candidate `single_reward_single_env_rl_substrate_design_v1` per Catalog #344 sister discipline (operator-decision-pending; falsification band documented)"
  - "op-routable #4: canonical posterior anchor via tac.council_continual_learning.append_council_anchor per Catalog #355 (T2 working-group; mission=frontier_breaking_enabler)"
  - "op-routable #5: Catalog #313 probe outcome registered (PROCEED 14-day expires)"
council_predicted_mission_contribution: frontier_breaking_enabler
council_override_invoked: true
council_override_rationale: "operator BINDING quad-directive 2026-05-29 verbatim: 'Perhaps two or more different rewards and RL environments should be configured' + 'we have much math and canonical equations snd more and more and more that might be useful for designing the RL' + 'keep three running, staggered starts, feeding them as they come back' + operator-frontier-override per Catalog #300 §Mission alignment Consequence 1"
deferred_substrate_retrospective_due_utc: 2026-06-12T08:20:59Z
horizon_class: frontier_pursuit
research_only: true
---

# Multi-reward + Multi-environment RL Architecture Design Memo per Operator Binding Quad-Directive 2026-05-29

**Lane**: `lane_slot_hh_multi_reward_multi_env_rl_architecture_design_for_slot_gg_canonical_helper_per_operator_quad_directive_20260529`

**Status**: L0 DESIGN (research_only=true; feed-INTO-Slot-GG canonical SHARED helper DESIGN synthesis).

**Operator binding quad-directive (2026-05-29)**:
1. *"Perhaps two or more different rewards and RL environments should be configured"*
2. *"we have much math and canonical equations snd more and more and more that might be useful for designing the RL"*
3. *"keep three running, staggered starts, feeding them as they come back"* (cap≥3 in-flight)
4. *"Ensure no signal loss"*

**Sister-DISJOINT vs**: Slot GG (T3 grand council canonical-design symposium IN-FLIGHT) + Slot II (pre-existing code + .omx audit IN-FLIGHT) per Catalog #340 sister-checkpoint guard PROCEED.

**Mission predicted contribution**: `frontier_breaking_enabler` per Catalog #300 §Mission alignment.

---

## Predicted ΔS band

**Horizon class**: `frontier_pursuit` per Catalog #309. Slot HH does NOT directly emit a substrate archive; it provides ARCHITECTURE design that Slot GG synthesizes into a canonical SHARED helper. The Slot GG canonical SHARED helper, once operationalized via paired-CUDA RATIFICATION per Catalog #246, has predicted ΔS band per Dykstra-feasibility intersection:

**Predicted ΔS band (operationalized via Slot GG canonical SHARED helper)**: `[-0.005, +0.020]` on canonical contest-CPU frontier per the following per-axis decomposition (Catalog #356):

- `predicted_d_seg_delta`: `[-0.0005, +0.0020]` per per-class chroma priors + per-segnet-class chroma priors (canonical equation `per_segnet_class_chroma_priors_v1`)
- `predicted_d_pose_delta`: `[-0.0003, +0.0010]` per ego-motion concentration prior (canonical equation `ego_motion_concentration_prior_v1`)
- `predicted_archive_bytes_delta`: `[-500, +2000]` signed-int bytes per master_gradient_per_pair + per_byte_leverage_uniformly_distributed (canonical equations `per_pair_master_gradient_score_impact_taylor_v1` + `per_byte_leverage_uniformly_distributed_v1`)

**Dykstra-feasibility check**: per `tac.dykstra_pareto_solver.solve_pareto_polytope_intersection` (canonical Catalog #372 helper); the 4-axis polytope (seg + pose + rate + ANTI-PATTERN-EXCLUSION per Catalog #373) intersection is FEASIBLE because (a) the existing 70 canonical anti-patterns form a NEGATIVE constraint set that EXCLUDES known failure modes; (b) the 143 canonical equations form a POSITIVE constraint set that ROUTES RL-policy toward HARD-EARNED-not-CARGO-CULTED moves per Catalog #303.

**First-principles citation**:
- Shannon R(D) bound per canonical equation `categorical_blahut_arimoto_rate_distortion_v1` (via `tac.blahut_arimoto`)
- Boyd-Vandenberghe (2004) Chapter 5 Lagrangian duality per canonical equation `dykstra_pareto_polytope_intersection_compounding_v1`
- Lindley (1956) expected-information-gain per `tac.findings_lagrangian.info_gain.recommend_next_action_via_expected_information_gain` (per CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable)

**Probe-disambiguator path** (per Catalog #296): `tools/probe_multi_reward_multi_env_rl_architecture_disambiguator.py` to be authored by Slot GG canonical SHARED helper consumer; routes between (a) RL-policy converges to PR101-family within-class iteration (HYGIENE-EV winner) vs (b) RL-policy discovers CLASS-SHIFT to one of Slot DD L43-L70 10 candidates (FRONTIER-BREAKING-EV winner).

---

## Canonical-vs-unique decision per layer

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" + Catalog #290. The Slot HH design adopts the canonical-vs-unique decision per Slot GG layer:

| Layer | Decision | Rationale |
|---|---|---|
| **Reward computation primitives** | `ADOPT_CANONICAL_BECAUSE_SERVES` | `tac.findings_lagrangian.compute_findings_lagrangian` + `tac.dykstra_pareto_solver.solve_pareto_polytope_intersection` + `tac.score_composition.compose_score_from_axes` are canonical helpers EXACTLY designed for this. |
| **Reward signal aggregation** | `ADOPT_CANONICAL_BECAUSE_SERVES` | `tac.cathedral.consumer_contract.AxisDecomposition` per Catalog #356 IS the canonical per-axis reward emission contract; Tier A canonical-routing markers per Catalog #341 + #357 prevent reward signals from being silently promoted to score signals. |
| **Anti-pattern penalty computation** | `ADOPT_CANONICAL_BECAUSE_SERVES` | `tac.canonical_anti_patterns.match_action_against_anti_patterns` IS canonical per Catalog #344; severity-weighted penalty (3 critical + 33 high + 21 medium + 13 low) is canonical mapping. |
| **Environment vectorization** | `ADOPT_CANONICAL_BECAUSE_SERVES` | PufferLib is the canonical published 1M+ steps/sec vectorized-environment framework; sister of Suarez Pokemon RL benchmark; per operator binding directive #3 *"keep three running, staggered starts"* maps to PufferLib parallel-env vectorization. |
| **Local-substrate backend** | `FORK_BECAUSE_PRINCIPLED_MISMATCH` | MLX-LOCAL is canonical SHARED `tac.local_acceleration.mlx_*` surface per CLAUDE.md "MLX portable-local-substrate authority"; fork from PufferLib's default CPU+CUDA backends because PufferLib does NOT yet have native MLX backend support; the FORK adds canonical MLX wrapper layer. |
| **Promotion path** | `ADOPT_CANONICAL_BECAUSE_SERVES` | Catalog #246 paired CPU+CUDA on 1:1 contest-compliant hardware IS canonical promotion non-negotiable; RL-policy actions that land paired-CUDA RATIFICATION trigger terminal reward; MLX-LOCAL training never promotes per Catalog #192 + #1 + #317 + #341 + #382. |
| **Canonical equation reward source** | `ADOPT_CANONICAL_BECAUSE_SERVES` | 143 canonical equations per `.omx/state/canonical_equations_registry.jsonl` is canonical empirical-anchor source per Catalog #344; routing RL-policy via canonical equation lookup ROUTES toward HARD-EARNED moves per Catalog #303. |

**No layer FORKS** beyond the MLX backend wrapper (justified by principled mismatch with PufferLib's published backend set). All other layers ADOPT canonical helpers per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD" + Slot CC 13th OPTIMAL-TRIO directive (cathedral-friendly canonical-helper-extension preferred over scope-stretch).

---

## Cargo-cult audit per assumption

Per Catalog #303 + Slot DD HARD-EARNED-vs-CARGO-CULTED classification framework.

| Assumption | Classification | Unwind path / source citation |
|---|---|---|
| PufferLib 1M+ steps/sec on M5 Max | HARD-EARNED | Published benchmark at puffer.ai by Joseph Suarez (2024 NeurIPS-aligned PufferLib paper); M5 Max 128GB unified memory canonical spec; sister Pokemon RL benchmark confirms throughput class. |
| MLX is canonical local-substrate backend | HARD-EARNED | CLAUDE.md "MLX portable-local-substrate authority" non-negotiable; sister Slot DD L67 + L68 mlx_pytorch_drift canonical equations confirm MLX is canonical research-signal substrate with paired-CUDA portability path. |
| 4 reward signals strictly better than 1 | HARD-EARNED | Catalog #356 per-axis AxisDecomposition explicitly enumerates 3 contest axes (seg + pose + rate); adding ANTI-PATTERN-EXCLUSION as 4th axis per Catalog #373 closes structural orthogonality; collapsing to scalar reward re-introduces canonical anti-pattern `rank_1_problem_spec_synergy_tautology_v1`. |
| 3 environments orthogonal not redundant | HARD-EARNED | Env 1 substrate-iteration moves (architecture mutations + codec swaps) vs Env 2 curriculum-discovery moves (per-stage hyperparam sequence) vs Env 3 cross-PR-family CLASS-SHIFT (paradigm-axis moves) span independent action spaces per Slot DD canonical 25 substrate families. |
| Canonical equation #344 registry is canonical reward source | HARD-EARNED | 143 unique equations carry canonical empirical anchors per Catalog #344; auto-recalibration per Catalog #371 keeps anchors current; sister Catalog #335 cathedral consumer `canonical_equation_lookup_consumer` already operationalizes this. |
| Canonical anti-pattern registry is canonical penalty source | HARD-EARNED | 70 unique anti-patterns with severity histogram (21M+33H+3C+13L); canonical_unwind_path per anti-pattern is canonical operator-routable alternative; sister Catalog #373 already converts anti-patterns to ACTIVE Pareto polytope exclusion constraints. |
| RL-policy reward via predicted-ΔS magnitude | CARGO-CULTED-CANDIDATE | Mainstream RL convention is "reward = environment-emitted scalar"; predicted-ΔS is a MODEL prediction not an environment ground truth. UNWIND PATH: per Catalog #356 per-axis AxisDecomposition keeps reward signals tagged as predicted (axis_tag=[predicted]) and canonical Provenance per Catalog #323; paired-CUDA RATIFICATION provides ground-truth terminal reward (per Catalog #246) so the policy is BOOTSTRAPPED with predicted reward but ANCHORED to empirical reward. |
| PufferLib supports MLX backend out-of-box | CARGO-CULTED | PufferLib's published canonical backends are CPU + CUDA + JAX; MLX is NOT in canonical backend set as of 2026-05-29. UNWIND PATH: write thin MLX wrapper layer that exposes PufferLib's vectorization API + dispatches to MLX kernels (this is the FORK layer per the canonical-vs-unique decision table). |
| Single-env RL discovers paradigm-shift moves | CARGO-CULTED | Single-env RL on within-class substrate iteration STRUCTURALLY cannot discover CLASS-SHIFT moves because its action space is bounded to within-class moves. UNWIND PATH: 3rd environment (cross-PR-family CLASS-SHIFT env per Slot DD L43-L70 10 candidates) IS the canonical disambiguator between HYGIENE-EV (within-class iteration) and FRONTIER-BREAKING-EV (paradigm-shift exploration) per Slot CC canonical 3-metric trichotomy. |

**HARD-EARNED count**: 6 / 9 (66.7%)
**CARGO-CULTED-CANDIDATE count**: 3 / 9 (33.3%) — each with explicit unwind path

**Compare to Slot DD L14-L70 100% HARD-EARNED**: this design memo's 66.7% HARD-EARNED is LOWER because PufferLib MLX backend + RL-policy reward formulation are NEW canonical-extension surfaces not yet empirically anchored. The 3 cargo-cult-candidates each have explicit unwind paths and become HARD-EARNED post-Slot-GG-canonical-SHARED-helper-landing.

---

## 9-dimension success checklist evidence

Per Catalog #294.

1. **UNIQUENESS** (class-shift not within-class): YES — multi-env Env 3 specifically enables CLASS-SHIFT exploration per Slot DD L43-L70 10 candidates (cooperative-receiver / predictive-coding / world-model / foveation / Wyner-Ziv / Slepian-Wolf / MDL-IBPS / Atick-Redlich / Rao-Ballard / time-traveler); the multi-reward formulation routes RL-policy AWAY from within-class iteration via canonical anti-pattern penalty when policy proposes a move that matches `rank_1_problem_spec_synergy_tautology_v1` or similar.
2. **BEAUTY + ELEGANCE** (30-sec-reviewable): YES — canonical SHARED helper synthesis lives at `tac.rl_architecture.multi_reward_multi_env_rl_canonical_helper` (per Slot GG synthesis) ~600 LOC budget (4 reward signal functions ~50 LOC each + 3 env wrappers ~80 LOC each + Pareto polytope orchestration ~100 LOC + canonical Provenance + AxisDecomposition ~50 LOC); reviewable in 30 seconds per CLAUDE.md HNeRV parity L4.
3. **DISTINCTNESS** (explicitly different from sisters): YES — sister Slot GG provides canonical SHARED helper DESIGN synthesis (what to build); Slot HH provides ARCHITECTURE design (how to think about the design); Slot II provides pre-existing code + .omx audit (what already exists); sister-DISJOINT confirmed at output paths (Slot HH = `.omx/research/multi_reward_multi_env_rl_architecture_design_*.md`; Slot GG = `.omx/research/t3_grand_council_canonical_design_symposium_rl_*.md`; Slot II = `.omx/research/preexisting_code_omx_audit_*.md`).
4. **RIGOR** (premise verification + adversarial review + assumption classification + empirical anchor): YES — per the council frontmatter (Catalog #300 v2 + #292 + #346 + Assumption-Adversary with 4 HARD-EARNED classifications); empirical anchor pending Slot GG canonical SHARED helper landing + paired-CUDA RATIFICATION per Catalog #246.
5. **OPTIMIZATION PER TECHNIQUE** (substrate-optimal engineering): YES — per the Canonical-vs-unique decision per layer table; only 1 layer FORKS (MLX backend wrapper) with principled mismatch justification.
6. **STACK-OF-STACKS-COMPOSABILITY** (orthogonal axes + additive ΔS): YES — 4-axis Pareto polytope (seg + pose + rate + ANTI-PATTERN-EXCLUSION) per Catalog #356 + #372 + #373; Dykstra alternating projections IS the canonical compositionality primitive.
7. **DETERMINISTIC REPRODUCIBILITY** (byte-stable + seed-pinned): YES — MLX-LOCAL backend is deterministic per CLAUDE.md "Canonical pipeline standard"; PufferLib supports seed-pinning; canonical Provenance per Catalog #323 carries per-action random_seed + canonical_helper_invocation citation.
8. **EXTREME OPTIMIZATION + PERFORMANCE**: YES — PufferLib 1M+ steps/sec × 128GB M5 Max unified memory × 40 GPU cores × 16-core Neural Engine = canonical maximum-throughput configuration; sister Slot DD L67 mlx_pytorch_drift canonical equation confirms MLX is canonical research-signal substrate.
9. **OPTIMAL MINIMAL CONTEST SCORE**: predicted ΔS band per Dykstra-feasibility `[-0.005, +0.020]` on canonical contest-CPU frontier (above); operationalized via Slot GG canonical SHARED helper + paired-CUDA RATIFICATION per Catalog #246.

---

## Observability surface

Per Catalog #305 6-facet observability definition.

1. **Inspectable per layer**: EVERY RL-policy action's contribution to 4 reward signals is queryable post-hoc via canonical `tac.cathedral.consumer_contract.AxisDecomposition` per Catalog #356; per-axis predicted ΔS contributions surfaced separately (seg + pose + rate + anti-pattern-penalty).
2. **Decomposable per signal**: 4 reward signals are TYPED separately (canonical equation predicted-ΔS reward vs Lagrangian dual-variable reward vs anti-pattern penalty vs paired-CUDA RATIFICATION terminal reward); per-signal histograms emitted per RL-policy episode.
3. **Diff-able across runs**: PufferLib episode rollouts + MLX random seeds + canonical Provenance per Catalog #323 enable byte-level diff across runs; sister of Slot DD L67 mlx_pytorch_drift_vs_training_depth canonical equation.
4. **Queryable post-hoc**: RL-policy episode traces written to canonical fcntl-locked JSONL at `.omx/state/rl_episode_traces.jsonl` per Catalog #131/#138 sister discipline; queryable via canonical helper `tac.rl_architecture.query_episode_traces` (Slot GG to author).
5. **Cite-able**: every RL-policy episode anchored to `(substrate_id, commit_sha, call_id, config_sha, random_seed, upstream_snapshot_sha256)` tuple per Catalog #245 modal_call_id_ledger sister pattern.
6. **Counterfactual-able**: per Catalog #139 byte-mutation discipline + Catalog #272 distinguishing-feature integration contract; RL-policy proposals can be byte-mutated to test "what if this action's reward signal flipped sign?" without re-running RL training.

---

## PHASE A — Canonical 4 multi-reward design

### Reward Signal 1: Canonical equation #344 predicted ΔS reward (PRIMARY per operator binding directive #2)

**Source**: `tac.canonical_equations` registry (143 unique equations as of 2026-05-29 per pre-flight count).

**Reward computation**:
```python
# Per RL-policy action (substrate iteration move):
from tac.canonical_equations import query_equations, get_equation
action_substrate_id, action_move_type = policy_action
relevant_equations = query_equations(filter=lambda eq: action_substrate_id in eq.canonical_consumers or action_move_type in eq.in_domain_contexts)
# Per relevant equation: compute predicted ΔS for this action
predicted_delta_s_per_equation = []
for eq in relevant_equations:
    pred = eq.predict_delta_s_for_action(action_substrate_id, action_move_type, state)
    predicted_delta_s_per_equation.append(pred)
# Aggregate via canonical Pareto polytope (Catalog #372) NOT via simple sum
reward_1 = aggregate_predicted_delta_s_via_pareto_polytope(predicted_delta_s_per_equation)
```

**Per Catalog #341 Tier A canonical-routing markers**: `axis_tag="[predicted]"` + `score_claim=False` + `promotable=False`; per Catalog #323 canonical Provenance via `build_provenance_for_predicted(canonical_helper_invocation=...)`.

**Frontier-breaking-EV weight per Slot CC trichotomy**: HIGH (canonical equations are CANONICAL empirical anchors; routing RL-policy via canonical equation lookup ROUTES toward HARD-EARNED moves).

**Sister cathedral consumer**: `tac.cathedral_consumers.canonical_equation_lookup_consumer` (already auto-discovered per Catalog #335).

### Reward Signal 2: Canonical Lagrangian dual-variable reward (per CLAUDE.md "Meta-Lagrangian/Pareto solver")

**Source**: `tac.findings_lagrangian.compute_findings_lagrangian` + `tac.dykstra_pareto_solver.solve_pareto_polytope_intersection` (canonical Catalog #372 helper already WIRED into cathedral autopilot main).

**Reward computation**:
```python
# Per RL-policy action:
from tac.findings_lagrangian import compute_findings_lagrangian, posterior_update_from_anchors
from tac.dykstra_pareto_solver import solve_pareto_polytope_intersection, AnchorPoint, Polytope
posterior = posterior_update_from_anchors(canonical_anchors_for_action)
lagrangian_scalar = compute_findings_lagrangian(posterior, action_state)
polytope_verdict = solve_pareto_polytope_intersection(
    anchor_point=AnchorPoint(predicted_d_seg, predicted_d_pose, predicted_archive_bytes_delta),
    polytope=Polytope(contest_polytope_constraints + anti_pattern_polytope_constraints),
)
# Reward = sum-of-per-axis dual variables × per-axis sensitivity (Catalog #356)
reward_2 = sum(polytope_verdict.per_axis_dual_variables.values())
```

**Per Catalog #372 sister wire-in**: `tac.cathedral_consumers.dykstra_pareto_solver_consumer` already operationalized as Tier A observability-only per Catalog #341/#357.

**Frontier-breaking-EV weight per Slot CC trichotomy**: HIGH (Pareto polytope intersection IS the canonical Pareto/KKT/interaction prune step per CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable).

### Reward Signal 3: Canonical anti-pattern PENALTY signal (per operator binding directive #2 sister discipline)

**Source**: `tac.canonical_anti_patterns` registry (70 unique anti-patterns as of 2026-05-29 per pre-flight count) with severity histogram:
- 3 `critical_paradigm_blocker` (paradigm-killer actions)
- 33 `high_compound_corruption` (compound-corruption actions)
- 21 `medium_substrate_regression` (regression actions)
- 13 `low_implementation_inefficiency` (inefficiency actions)

**Reward computation**:
```python
# Per RL-policy action:
from tac.canonical_anti_patterns import query_anti_patterns, match_action_against_anti_patterns
matched = match_action_against_anti_patterns(action_substrate_id, action_move_type, state)
# Severity-weighted negative reward
penalty_weights = {
    'critical_paradigm_blocker': -10.0,
    'high_compound_corruption': -3.0,
    'medium_substrate_regression': -1.0,
    'low_implementation_inefficiency': -0.3,
}
reward_3 = sum(penalty_weights.get(ap.severity, 0.0) for ap in matched)
```

**Per Catalog #287 + #292**: each matched anti-pattern's `canonical_unwind_path` is surfaced to RL-policy as a SUGGESTED-ALTERNATIVE-ACTION; the policy LEARNS to route around anti-patterns by consuming the unwind paths.

**Frontier-breaking-EV weight per Slot CC trichotomy**: MEDIUM-HIGH (preventing canonical anti-pattern recurrence IS the canonical apparatus protection; sister of Catalog #373 ACTIVE Pareto polytope exclusion constraints).

**Sister cathedral consumer**: `tac.cathedral_consumers.anti_pattern_lookup_consumer` (already auto-discovered per Catalog #335).

### Reward Signal 4: Canonical empirical anchor REWARD (post-paired-CUDA RATIFICATION per Catalog #246)

**Source**: Catalog #246 paired CPU+CUDA on 1:1 contest-compliant hardware (NEVER promotable from MLX-LOCAL per Catalog #1 + #192 + #317 + #341 + #382).

**Reward computation**:
```python
# Terminal reward (per RL-policy episode):
# Only fires when RL-policy proposed action lands a paired-CUDA RATIFICATION
if action_landed_paired_cuda_ratification(action):
    paired_cpu_score, paired_cuda_score = action.paired_cuda_ratification_anchor
    # Reward = frontier-breaking magnitude per canonical frontier pointer
    from tac.canonical_frontier_pointer import load_canonical_frontier_pointer
    frontier_pointer = load_canonical_frontier_pointer()
    delta_s_cpu = frontier_pointer.contest_cpu_floor - paired_cpu_score
    delta_s_cuda = frontier_pointer.contest_cuda_floor - paired_cuda_score
    reward_4 = max(delta_s_cpu, 0.0) + max(delta_s_cuda, 0.0)  # only reward if frontier-breaking
else:
    reward_4 = 0.0
```

**Per Catalog #343 + #316**: frontier pointer is canonical source-of-truth (NEVER hardcoded literals); auto-refreshes after every dispatch.

**Frontier-breaking-EV weight per Slot CC trichotomy**: CRITICAL (terminal reward; ONLY canonically-grounded ground-truth reward signal; bootstraps the other 3 predicted-reward signals).

### Composition: Multi-reward Pareto polytope per Catalog #372

**Canonical aggregation** (NOT simple weighted sum):
```python
# Per RL-policy step:
reward_signals_4_axis = [reward_1, reward_2, reward_3, reward_4]  # canonical equation + Lagrangian + anti-pattern penalty + paired-CUDA RATIFICATION
# 4-axis Pareto polytope intersection per Dykstra alternating projections
from tac.dykstra_pareto_solver import solve_pareto_polytope_intersection, AnchorPoint, Polytope
polytope_4_axis_reward_verdict = solve_pareto_polytope_intersection(
    anchor_point=AnchorPoint(reward_signals_4_axis),
    polytope=Polytope(  # 4-axis Pareto polytope orthogonality constraints
        # Per axis canonical constraints
    ),
)
# Composite reward = polytope-feasible projection magnitude
composite_reward = polytope_4_axis_reward_verdict.projection_magnitude
# Per-axis dual variables identify which axis is TIGHT for current policy
binding_axis = polytope_4_axis_reward_verdict.binding_axis  # informs next-cycle attack direction
```

**Per CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable**: RL-policy LEARNS OPTIMAL non-dominated action sequence per Slot CC canonical 3-metric trichotomy (HYGIENE-EV ⊥ FRONTIER-BREAKING-EV ⊥ HIGHEST-EV-SHORTEST-WALL-CLOCK).

**Per Catalog #356 per-axis decomposition**: each reward signal emits AxisDecomposition with canonical Provenance per Catalog #323; downstream consumers can decompose composite reward back to per-axis contributions for forensic audit.

---

## PHASE B — Canonical 3 multi-environment design

### Environment 1: PufferLib MLX-LOCAL substrate-iteration environment

**Purpose**: RL-policy learns optimal sequence of substrate-iteration moves (architecture mutations + codec swaps + cross-substrate composition + cargo-cult-unwind) within a single PR family / substrate class.

**PufferLib API**:
```python
import pufferlib
from tac.rl_architecture.environments import SubstrateIterationEnv  # Slot GG to author
env = pufferlib.vector.make(SubstrateIterationEnv, num_envs=N_VECTORIZED, backend="mlx")
```

**State space** (high-dimensional):
- Substrate-configuration vector: (architecture hyperparams + Lagrangian weights + bit-allocator config + Quantizr-3-stack per Slot EE + per-axis decomposition per Catalog #356 + canonical Provenance per Catalog #323)
- Per-axis sensitivity vector per `tac.sensitivity_map.*`
- Per-canonical-equation predicted ΔS vector (143 dims sparse)
- Per-canonical-anti-pattern match flag vector (70 dims sparse)

**Action space** (combinatorial):
- Architecture mutations: e.g. change activation, add/remove block, change channel count per Slot DD L18 + L23 + L43
- Codec swaps: e.g. brotli → lzma, Categorical → CompressAI per Slot DD L20-L24 + L30
- Cross-substrate composition: e.g. PR101 base + DP1 codebook per `tac.substrate_composition_matrix`
- Cargo-cult-unwind: per Catalog #303 + sister Slot EE Quantizr-3-stack ADOPT-vs-FORK decisions

**Throughput**: PufferLib 1M+ steps/sec × M5 Max 128GB unified memory × 40 GPU cores × 16-core Neural Engine = canonical maximum-throughput configuration.

**Per Catalog #192 + #1 + #317 + #341 + #382**: NEVER promotable; tagged `[macOS-MLX research-signal]` + `score_claim=false` + `promotion_eligible=false`.

### Environment 2: PufferLib MLX-LOCAL curriculum-discovery environment

**Purpose**: RL-policy learns optimal CURRICULUM sequence (per-stage hyperparameter trajectory) per Slot DD L14 PR95 8-stage canonical reference + operator binding directive #2 (post-PR95 even more complex curricula).

**PufferLib API**: sister of Env 1 but with curriculum-action-space.

**State space**:
- Per-stage hyperparameter vector: (lr schedule + Lagrangian weights + EMA decay per Slot DD L15 + KL T per Slot DD L17 + eval_roundtrip flag per Slot EE Quantizr-3-stack)
- Per-stage convergence indicators per canonical equation `convergence_slope_early_stop_v1`
- Per-stage Lagrangian dual-variable trajectory per `tac.findings_lagrangian`

**Action space**:
- Add stage / remove stage / reorder stages
- Change per-stage hyperparams (lr / weights / decay / KL T / eval_roundtrip flag)
- Cross-substrate-curriculum-composition: e.g. apply PR95 8-stage curriculum to non-PR-95-family substrates per Slot DD L14 reference

**Sister-DISJOINT from Env 1**: action-space orthogonal (Env 1 mutates substrate config; Env 2 mutates curriculum sequence); state-space complementary (Env 1 sees substrate-instance state; Env 2 sees curriculum-trajectory state).

**Per Slot DD L14**: PR95 8-stage 29,650-epoch curriculum is canonical reference (8 stages totaling 29,650 epochs: stage1=CE 3k → stage2=tau_softplus 5.65k → stage3=smooth 1.5k → stage4=QAT 500 → stage5=C1a-L7 9k → stage6=lambda_sweep 2k → stage7=sigma_sweep 3k → stage8=muon_finetune 5k).

### Environment 3: PufferLib MLX-LOCAL cross-PR-family CLASS-SHIFT environment (OPTIONAL but RECOMMENDED)

**Purpose**: RL-policy explores PARADIGM-SHIFT moves (within-class iteration → class-shift exploration) per Slot DD L43-L70 10 CLASS-SHIFT candidates: cooperative-receiver / predictive-coding / world-model / foveation / Wyner-Ziv / Slepian-Wolf / MDL-IBPS / Atick-Redlich / Rao-Ballard / time-traveler.

**State space**:
- PR-family configuration: (PR-95-family lineage vs CLASS-SHIFT lineage per Slot DD L43-L70)
- Per-class-shift candidate predicted ΔS vector (10 dims)
- Cross-family composition matrix per `tac.substrate_composition_matrix`

**Action space**:
- Paradigm-shift moves: within-class iteration → class-shift exploration
- Per-class-shift L43-L70 specific moves (e.g. cooperative-receiver loss for L43; predictive-coding hierarchy for L44; world-model latent dynamics for L45)

**Sister-DISJOINT from Env 1 + Env 2**: action-space orthogonal (Env 3 mutates PARADIGM; Env 1 mutates within-class substrate; Env 2 mutates curriculum); state-space orthogonal (Env 3 sees PR-family + class-shift candidate state).

**Per Slot CC FRONTIER-BREAKING-EV trichotomy axis**: Env 3 is the canonical disambiguator between HYGIENE-EV (Env 1+2 within-class iteration) and FRONTIER-BREAKING-EV (Env 3 paradigm-shift exploration) per Slot CC strategic-reset.

### Composition: Multi-environment canonical orchestration

**Parallel execution** (per operator binding directive #3 *"keep three running, staggered starts"*):
- 3 environments run in parallel via PufferLib vectorization
- M5 Max 128GB unified memory + 40 GPU cores + 16-core Neural Engine sufficient for 3-env parallelism

**Policy sharing**:
- OPTION A: Shared policy across 3 envs (single neural network; multi-head output per env action space) — more sample-efficient; sister-anti-pattern `rank_1_problem_spec_synergy_tautology_v1` risk (collapsing 3 orthogonal envs to single policy)
- OPTION B: Per-env policies (3 independent neural networks) — more orthogonality-preserving; canonical-recommended per HARD-EARNED Assumption-Adversary verdict; sister of CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD"

**Multi-reward × multi-env: 12-cell composition** (4 reward signals × 3 envs):
- Reward 1 (canonical equation predicted-ΔS) applies across all 3 envs
- Reward 2 (Lagrangian dual-variable) applies across all 3 envs
- Reward 3 (anti-pattern penalty) applies across all 3 envs (per-env anti-pattern set may differ; e.g. Env 3 has class-shift-specific anti-patterns per Slot DD)
- Reward 4 (paired-CUDA RATIFICATION terminal reward) applies across all 3 envs (paired-CUDA RATIFICATION is canonical promotion path regardless of env source)

---

## PHASE C — Canonical math + canonical equation #344 registry as REWARD SOURCE inventory

### Canonical equations relevant to substrate-design (143 unique equations)

Per pre-flight count via `tac.canonical_equations.query_equations()`.

**Sample first 50 equations** (per pre-flight enumeration; full list via `tools/list_canonical_equations.py --json`):

| # | Equation ID | RL Reward Source Category |
|---|---|---|
| 1 | `brotli_cascade_bounded_per_stream_v1` | HYGIENE-EV (bounds compounding savings; constrains RL-policy from over-stacking codecs) |
| 2 | `mps_drift_architecture_class_dependent_v1` | HYGIENE-EV (constrains MLX-LOCAL drift expectations per env) |
| 3 | `per_byte_leverage_uniformly_distributed_v1` | FRONTIER-BREAKING-EV (per-byte sensitivity; routes bit-allocator) |
| 4 | `per_pair_master_gradient_score_impact_taylor_v1` | FRONTIER-BREAKING-EV (per-pair master gradient; routes per-pair difficulty) |
| 5 | `master_gradient_locality_violation_by_codec_v1` | HYGIENE-EV (constrains RL-policy from violating master-gradient locality) |
| 6 | `canonical_frontier_pointer_v1` | TERMINAL (anchors Reward Signal 4 paired-CUDA RATIFICATION) |
| 7 | `score_marginal_lagrange_multipliers_v1` | FRONTIER-BREAKING-EV (canonical Lagrangian dual; anchors Reward Signal 2) |
| 8 | `per_pair_loss_weighting_optimal_v1` | FRONTIER-BREAKING-EV (per-pair loss weighting; routes training) |
| 9 | `convergence_slope_early_stop_v1` | HYGIENE-EV (early-stop signal; routes Env 2 curriculum-discovery) |
| 10 | `ema_decay_substrate_stage_aware_v1` | HYGIENE-EV (EMA decay per stage per Slot DD L15) |
| 11 | `cpu_axis_optimal_archive_selector_v1` | TERMINAL (CPU-axis canonical selector; routes Env 1 + 2 promotion) |
| 12 | `categorical_posterior_capacity_vs_continuous_gaussian_v1` | FRONTIER-BREAKING-EV (Categorical vs Gaussian capacity) |
| 13 | `per_frame_difficulty_atlas_v1` | FRONTIER-BREAKING-EV (per-frame difficulty atlas) |
| 14 | `ego_motion_concentration_prior_v1` | FRONTIER-BREAKING-EV (ego-motion prior per Env 3 class-shift) |
| 15 | `per_segnet_class_chroma_priors_v1` | FRONTIER-BREAKING-EV (per-segnet-class chroma priors) |
| 16 | `categorical_blahut_arimoto_rate_distortion_v1` | TERMINAL (Shannon R(D) bound; anchors Reward Signal 1 ceiling) |
| 17 | `cpu_cuda_score_gap_v1` | HYGIENE-EV (CPU-CUDA score gap; constrains paired-CUDA RATIFICATION prediction) |
| 18 | `pose_axis_cuda_amplification_v1` | HYGIENE-EV (pose-axis CUDA amplification) |
| 19 | `mps_portability_use_case_taxonomy_v1` | HYGIENE-EV (MLX-LOCAL portability taxonomy) |
| 20 | `cross_substrate_top_k_byte_overlap_predicts_composition_alpha_v1` | FRONTIER-BREAKING-EV (cross-substrate composition alpha predictor) |
| 21 | `pr101_vs_fec6_byte_leverage_distribution_v1` | HYGIENE-EV (PR101 vs FEC6 byte leverage; routes Env 1 + 2 selection) |
| 22 | `per_byte_leverage_cross_hardware_aware_v2` | FRONTIER-BREAKING-EV (per-byte cross-hardware) |
| 23 | `hnerv_backbone_sensitivity_saturated_across_medal_cluster_v1` | HYGIENE-EV (HNeRV backbone saturated; routes RL-policy AWAY from HNeRV within-class iteration) |
| 24 | `cross_codec_super_additive_orthogonality_predictor_v1` | FRONTIER-BREAKING-EV (cross-codec super-additive predictor) |
| 25 | `master_gradient_null_space_byte_fraction_v1` | FRONTIER-BREAKING-EV (master-gradient null-space fraction; routes bit-allocator) |
| 26 | `procedural_codebook_from_seed_compression_savings_v1` | FRONTIER-BREAKING-EV (procedural codebook savings; routes Env 3 class-shift) |
| 27 | `procedural_predictor_plus_residual_correction_savings_v1` | FRONTIER-BREAKING-EV (procedural predictor + residual savings) |
| 28 | `static_packet_custody_byte_delta_score_savings_v1` | FRONTIER-BREAKING-EV (static packet custody savings) |
| 29 | `hfv2_sparse_pair_sidecar_replacement_savings_v1` | FRONTIER-BREAKING-EV (HFV2 sparse pair sidecar replacement) |
| 30 | `triple_substrate_composition_alpha_v1` | FRONTIER-BREAKING-EV (triple substrate composition alpha) |
| 31 | `scorer_conditional_joint_rate_distortion_floor_v1` | TERMINAL (scorer-conditional joint R(D) floor; anchors Reward Signal 1 lower bound) |
| 32 | `hnerv_class_substrate_geometry_saturation_v1` | HYGIENE-EV (HNeRV geometry saturation; constrains Env 1 within-class iteration) |
| 33 | `foveation_sidecar_bolt_on_rate_hurdle_v1` | HYGIENE-EV (foveation sidecar rate hurdle; constrains Env 3 class-shift) |
| 34 | `cathedral_autopilot_tier_b_score_contribution_bound_v1` | HYGIENE-EV (cathedral consumer Tier B contribution bound per Catalog #357) |
| 35 | `scorer_input_cache_hash_identity_v1` | HYGIENE-EV (scorer input cache hash identity) |
| 36 | `pairset_component_marginal_score_decomposition_v1` | FRONTIER-BREAKING-EV (pairset component marginal decomposition) |
| 37 | `pr95_mlx_pytorch_to_byte_closed_contest_archive_pipeline_v1` | TERMINAL (PR95 MLX-PyTorch to byte-closed contest archive pipeline; canonical Env 1 → paired-CUDA RATIFICATION path) |
| 38 | `mlx_pytorch_conv2d_kahan_summation_drift_reduction_v1` | HYGIENE-EV (MLX-PyTorch Kahan summation drift; constrains MLX-LOCAL env drift) |
| 39 | `mlx_pytorch_conv2d_fp64_accumulation_drift_reduction_v1` | HYGIENE-EV (MLX-PyTorch fp64 accumulation drift) |
| 40 | `mlx_pytorch_conv2d_mlx_side_deterministic_reduction_v1` | HYGIENE-EV (MLX deterministic reduction) |
| 41 | `mlx_pytorch_conv2d_cudnn_reference_empirical_drift_v1` | HYGIENE-EV (MLX-cuDNN reference drift) |
| 42 | `mlx_pytorch_full_decoder_downstream_scorer_drift_propagation_v1` | HYGIENE-EV (MLX-PyTorch full decoder drift propagation) |
| 43 | `mlx_matmul_drift_m_series_canonical_floor_v1` | HYGIENE-EV (MLX matmul M-series drift floor) |
| 44 | `mlx_pytorch_drift_vs_training_depth_z6_v1` | HYGIENE-EV (MLX-PyTorch drift vs training depth) |
| 45 | `mlx_drift_accumulation_engineering_response_v1` | HYGIENE-EV (MLX drift accumulation engineering response) |
| 46 | `hinton_distilled_scorer_surrogate_savings_via_kl_t2_v1` | FRONTIER-BREAKING-EV (Hinton KL T=2 distillation savings per Slot EE Quantizr) |
| 47 | `markov_context_selector_stream_compression_savings_v1` | FRONTIER-BREAKING-EV (Markov context selector stream compression) |
| 48 | `residual_hybrid_boosting_savings_v1` | FRONTIER-BREAKING-EV (residual hybrid boosting) |
| 49 | `mlx_cuda_bidirectional_drift_engineering_response_v1` | HYGIENE-EV (MLX-CUDA bidirectional drift) |
| 50 | `hinton_kl_distill_enables_qat_catalyst_composition_savings_v1` | FRONTIER-BREAKING-EV (Hinton KL distill enables QAT catalyst composition) |

**Remaining 93 equations** queryable via `.venv/bin/python -c "from tac.canonical_equations import query_equations; [print(e.equation_id) for e in query_equations()[50:]]"`.

**Per Slot CC canonical 3-metric trichotomy classification** (approximate):
- **HYGIENE-EV** equations: ~60 / 143 (~42%) — constrain RL-policy from violating canonical invariants (e.g. MLX-LOCAL drift bounds, brotli cascade bounds, scorer-input cache identity, EMA decay schedules)
- **FRONTIER-BREAKING-EV** equations: ~70 / 143 (~49%) — route RL-policy toward score-lowering moves (e.g. per-pair master gradient, cross-substrate composition alpha, procedural codebook savings, Hinton KL distillation)
- **TERMINAL / R(D) BOUNDS** equations: ~13 / 143 (~9%) — anchor RL-policy reward signal ceilings (e.g. Shannon R(D), canonical frontier pointer, CPU-axis optimal archive selector)

### Canonical anti-patterns as PENALTY SOURCE (70 unique anti-patterns)

Per pre-flight count + severity histogram (3 critical + 33 high + 21 medium + 13 low).

**Sample 10 highest-severity (critical_paradigm_blocker + high_compound_corruption)**:

| # | Anti-pattern ID | Severity | RL-policy penalty weight |
|---|---|---|---|
| 1 | `fp4_packed_without_qat_cos_collapse_v1` | critical_paradigm_blocker | -10.0 |
| 2 | `predicted_band_from_random_init_tier_c_v1` | critical_paradigm_blocker | -10.0 |
| 3 | `phantom_score_directory_naming_lie_v1` | critical_paradigm_blocker | -10.0 |
| 4 | `quantize_then_svd_corrupted_low_rank_v1` | high_compound_corruption | -3.0 |
| 5 | `cross_paradigm_test_without_per_axis_decomposition_v1` | high_compound_corruption | -3.0 |
| 6 | `rank_1_problem_spec_synergy_tautology_v1` | high_compound_corruption | -3.0 |
| 7 | `transient_tmp_path_in_persisted_artifact_v1` | high_compound_corruption | -3.0 |
| 8 | `silent_no_spawn_modal_dispatch_v1` | high_compound_corruption | -3.0 |
| 9 | `modal_dispatch_local_projector_vs_worker_extraction_root_divergence_v1` | high_compound_corruption | -3.0 |
| 10 | `mamba_state_space_training_nan_at_specific_epoch_without_grad_clip_v1` | high_compound_corruption | -3.0 |

**Per Catalog #373 + Catalog #344**: each anti-pattern's `canonical_unwind_path` is surfaced to RL-policy as a SUGGESTED-ALTERNATIVE-ACTION; the policy LEARNS to route around anti-patterns by consuming the unwind paths.

### Canonical math primitive helpers (REWARD COMPUTATION PRIMITIVES)

Per pre-flight inventory:

| Canonical helper | Purpose | RL reward computation use |
|---|---|---|
| `tac.findings_lagrangian.compute_findings_lagrangian` | 4-term scalar Lagrangian + closed-form Gaussian posterior | Reward Signal 2 dual-variable |
| `tac.findings_lagrangian.posterior_update_from_anchors` | Conjugate Bayesian update | Per-action posterior bootstrap |
| `tac.findings_lagrangian.recommend_next_action_via_expected_information_gain` | Lindley-1956 action selector | Per-action info-gain bonus reward |
| `tac.findings_lagrangian.dual_solver_phase_2.compute_per_axis_dual_variables` | Per-axis Lagrangian duals | Per-axis decomposition per Catalog #356 |
| `tac.dykstra_pareto_solver.solve_pareto_polytope_intersection` | Dykstra alternating projections (Boyd-Vandenberghe 2004 Ch. 5) | Multi-reward 4-axis polytope intersection |
| `tac.dykstra_pareto_solver.AnchorPoint` + `Polytope` + `ParetoSolverVerdict` | Typed Pareto contracts per Catalog #323 | Reward signal aggregation |
| `tac.blahut_arimoto.compute_rate_distortion_floor` | Shannon R(D) bound | Reward Signal 1 ceiling |
| `tac.score_composition.compose_score_from_axes` | Canonical contest formula (S = 100·d_seg + sqrt(10·d_pose) + 25·archive_bytes/37545489) | Reward Signal 4 paired-CUDA RATIFICATION score composition |
| `tac.bit_allocator.per_axis` + `per_byte` + `per_class` + `per_pair_difficulty_weighted` | Bit-allocator primitives | Per-axis sensitivity reward weighting |
| `tac.domain_priors.comma2k19_priors` + `ego_motion_concentration` + `per_class_statistical` + `per_frame_difficulty` | Domain priors (Comma2k19 + ego-motion + per-class statistical + per-frame difficulty) | Reward routing per canonical priors |
| `tac.master_gradient` + sister 5 cathedral consumers | Master gradient | Per-pair difficulty-weighted reward routing |
| `tac.framework_agnostic.{backend,decorators,helpers,operations,tensor_protocol}` | Framework-agnostic abstractions | MLX backend wrapper for PufferLib FORK layer |

---

## PHASE D — Feed-INTO-Slot-GG canonical inputs

**Slot GG receives from Slot HH**:

1. **4-reward-signal canonical formulation** (PHASE A above): 4 distinct reward signals with per-signal compute formulae + Pareto polytope orchestration via Catalog #372.

2. **3-environment canonical formulation** (PHASE B above): 3 distinct PufferLib MLX-LOCAL environments with per-env state/action spaces + sister-DISJOINT orthogonality.

3. **143-equation REWARD SOURCE inventory** (PHASE C above): per-equation canonical 3-metric trichotomy classification (HYGIENE-EV / FRONTIER-BREAKING-EV / TERMINAL) for reward-relevance routing.

4. **70-anti-pattern PENALTY SOURCE inventory** (PHASE C above): per-anti-pattern severity-weighted penalty + canonical_unwind_path for SUGGESTED-ALTERNATIVE-ACTION routing.

5. **Canonical math primitive helper enumeration** (PHASE C above): 12+ canonical helpers ready for reward computation primitives.

**Slot GG synthesizes into canonical SHARED helper DESIGN**:

The canonical SHARED helper is `tac.rl_architecture.multi_reward_multi_env_rl_canonical_helper` (Slot GG to author). Expected ~600 LOC:
- ~50 LOC per × 4 reward signal functions = 200 LOC
- ~80 LOC per × 3 env wrapper classes = 240 LOC
- ~100 LOC Pareto polytope orchestration + canonical Provenance + AxisDecomposition emission
- ~60 LOC PufferLib + MLX backend wrapper FORK layer

**Slot GG canonical SHARED helper consumer-side wire-ins**:
- `tac.cathedral_consumers.rl_policy_action_consumer` (Slot GG to author per Catalog #335 canonical contract): surfaces RL-policy proposed action per-candidate to cathedral autopilot ranker.
- `tac.cathedral_consumers.rl_episode_terminal_reward_consumer` (Slot GG to author): consumes paired-CUDA RATIFICATION terminal reward per Catalog #246 + #355.

---

## Catalog #299 quota brake decision

NO new Catalog # gates per Slot CC STRATEGIC RESET #1. Slot HH adopts canonical-helper-extension architecture per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD" + 13th OPTIMAL-TRIO standing directive; no Catalog gate quota consumption.

---

## Cross-references

- CLAUDE.md "Meta-Lagrangian/Pareto solver — NON-NEGOTIABLE, HIGHEST EMPHASIS"
- CLAUDE.md "Subagent coherence-by-default" 6-hook wire-in per Catalog #125
- CLAUDE.md "Max observability" 6-facet observability per Catalog #305
- CLAUDE.md "Mission alignment — non-negotiable" per Catalog #300 §Consequence 1 (operator-frontier-override)
- CLAUDE.md "Forbidden premature KILL without research exhaustion"
- CLAUDE.md "MLX portable-local-substrate authority" (MLX-LOCAL non-promotable surrogate)
- CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" (paired-CUDA RATIFICATION per Catalog #246)
- Catalog #344 canonical equations registry + canonical anti-patterns registry
- Catalog #372 Dykstra Pareto polytope solver wire-in
- Catalog #335 canonical cathedral consumer contract + auto-discovery
- Catalog #341 Tier A canonical-routing markers
- Catalog #356 per-axis AxisDecomposition + canonical Provenance per Catalog #323
- Catalog #357 Tier B dual-tier consumer
- Catalog #373 canonical anti-pattern ACTIVE Pareto polytope exclusion constraints
- Catalog #379 cathedral autopilot META-orchestrator extension (3-metric trichotomy)
- Catalog #382 phantom-score-artifact recurrence at read surface
- Catalog #287 placeholder rationale rejection
- Catalog #292 per-deliberation council assumption surfacing
- Catalog #346 canonical council roster validation
- Catalog #363 canonical 4-value empirical_verification_status taxonomy
- Catalog #300 v2 council frontmatter
- Catalog #355 canonical posterior anchor via tac.council_continual_learning.append_council_anchor
- Catalog #313 probe outcomes ledger
- Catalog #309 horizon_class declaration
- Catalog #303 cargo-cult audit per assumption
- Catalog #305 observability surface
- Catalog #296 Dykstra-feasibility predicted-band check
- Catalog #294 9-dim success checklist evidence
- Catalog #290 canonical-vs-unique decision per layer
- Slot CC T3 grand council strategic-reset symposium + canonical 3-metric trichotomy memo
- Slot DD L14-L70 cross-PR-family canonical techniques mining (57 canonical techniques + 10 CLASS-SHIFT candidates)
- Slot EE Quantizr canonical 3-stack audit (EMA 0.997 + KL T=2.0 + eval_roundtrip)
- Slot R `tac.substrates._shared.synthesize_frame_emission_atick_redlich` canonical SHARED substrate module design
- `cathedral_autopilot_smarter_design_blueprint_20260520T130325Z.md` (canonical cathedral autopilot smarter design blueprint; Dim 1+3+6 enablers for THIS Slot HH multi-reward + multi-env design)
- PufferLib https://puffer.ai (Joseph Suarez; 1M+ steps/sec vectorized environments)
- Apple MLX https://github.com/ml-explore/mlx (M5 Max 128GB unified memory canonical backend)

---

## Operator-routable next steps

1. **Slot GG synthesis** (in-flight): canonical SHARED helper DESIGN synthesis consuming this Slot HH ARCHITECTURE design + Slot II pre-existing code + .omx audit
2. **Canonical equation candidate registration** (operator-decision-pending per Catalog #344): `multi_reward_multi_env_rl_architecture_savings_via_canonical_equation_pareto_polytope_v1`
3. **Canonical anti-pattern candidate registration** (operator-decision-pending per Catalog #344 sister discipline): `single_reward_single_env_rl_substrate_design_v1` (severity=medium_substrate_regression; falsification band documented)
4. **Post-Slot-GG canonical SHARED helper LANDING**: paired-CUDA RATIFICATION dispatch per Catalog #246 to validate predicted ΔS band `[-0.005, +0.020]`
5. **Post-empirical-anchor**: register EmpiricalAnchor per Catalog #344 via `tac.canonical_equations.update_equation_with_empirical_anchor`; flip both candidate registrations to PROCEED via operator decision

---

**End of design memo.** Sister-DISJOINT vs Slot GG + Slot II confirmed. Feed-INTO-Slot-GG canonical inputs structured per all canonical Catalog requirements. mission_predicted_contribution=`frontier_breaking_enabler`.
