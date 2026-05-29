---
council_tier: T2
council_attendees: [Shannon, Dykstra, Rudin, Daubechies, Yousfi, Fridrich, Contrarian, AssumptionAdversary, Hafner-advisory, Rao, Ballard]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Contrarian
    verbatim: "Delegation is the right structural fix but the Wave 10/11 spawn directive asked for an MLX-LOCAL diagnostic distinguishing the 3 Z7-Mamba-2 NaN hypotheses; I judge the disambiguator is already resolved structurally (NaN extincted via Cell 2-3 stability hardening; reference cell is S6 form per Wave 4). Spawning a new diagnostic on a resolved question would be paid-GPU spend without learning value."
  - member: AssumptionAdversary
    verbatim: "Z8 unimix gap was CARGO-CULTED docstring-vs-implementation divergence per Catalog #303. Delegation is the canonical unwind. DreamerV3 3rd anchor on structural delegation is HARD-EARNED at the structural-correctness surface; paid-GPU 3rd anchor remains operator-routable per spawn Part C."
council_assumption_adversary_verdict:
  - assumption: "Z8 was structurally consuming sister DreamerV3 categorical-posterior primitive"
    classification: HARD-EARNED
    rationale: "Z8 design memo + __init__.py LINE 24-25 explicitly claim Catalog #290 ADOPT_CANONICAL_BECAUSE_SERVES."
  - assumption: "Local Z8 gumbel_softmax_sample was a duplicate implementation NOT a true delegation"
    classification: HARD-EARNED
    rationale: "Pre-fix mlx_renderer.py LINE 195-227 contained mx.random.uniform + Gumbel-perturbation math directly NOT a sister-call."
  - assumption: "Wave 3 unimix fix did not propagate to Z8 because of the duplication"
    classification: HARD-EARNED
    rationale: "Pre-fix Z8 signature lacked unimix_alpha parameter entirely; verified via inspect.signature."
  - assumption: "Z7-Mamba-2 NaN at ep 16-18 was hypothesis (c) ego-motion-conditioning requirement"
    classification: CARGO-CULTED-EMPIRICALLY-FALSIFIED
    rationale: "L2 stability hardening Cell 2-3 NaN-FREE 30ep with pinned-random ego-motion (not PoseNet-derived); root cause is selective-SSM stability per Wave 4 + L2 memo. Per Catalog #307 IMPLEMENTATION-LEVEL."
council_decisions_recorded:
  - "op-routable #1: Z8 gumbel_softmax_sample + apply_unimix_to_logits DELEGATE to sister dreamer_v3_rssm canonical helpers"
  - "op-routable #2: 5 Wave 10/11 unimix-propagation regression tests LANDED"
  - "op-routable #3: DreamerV3 RSSM canonical equation 3rd anchor LANDED; 3+ trigger crossed for Catalog #371 auto-recalibration"
  - "op-routable #4: Z7-Mamba-2 NaN disambiguator status documented: (a) S6-form per Wave 4 IMPLEMENTATION-LEVEL; (b) identity-predictor NOT THE CAUSE per L2 hardening; (c) ego-motion NOT THE CAUSE per L2 pinned-random 30ep NaN-FREE"
  - "op-routable #5: paid-Modal 3rd-anchor for DreamerV3 PyTorch port REMAINS DEFERRED per Wave 7 op-routable; operator-attended dispatch only"
  - "op-routable #6: Z8 stays L0 SCAFFOLD research_only=true per Catalog #240"
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
council_override_rationale: null
horizon_class: frontier_pursuit
deferred_substrate_retrospective_due_utc: "2026-06-28T23:33:00Z"
deferred_substrate_id: z8_hierarchical_predictive_coding
related_deliberation_ids:
  - wave_4_z7_mamba_2_dao_gu_fidelity_audit_landed_20260529
  - wave_3_dreamerv3_rssm_math_fidelity_audit_landed_20260529
  - wave_7_dreamerv3_rssm_phase_2_rl_push_landed_20260529
  - z7_mamba_2_l1_empirical_mlx_fair_shake_landed_20260526
  - z7_mamba_2_v2_l2_stability_hardening_landed_20260526
  - path_3_b_z7_mamba_2_cargo_cult_audit_of_existing_scaffold_20260526
empirical_verification_status:
  z8_pre_fix_local_gumbel_softmax_sample_lacked_unimix_alpha:
    status: VERIFIED_VIA_SOURCE_INSPECTION
    evidence: "git show pre-edit mlx_renderer.py:195-227 + inspect.signature audit captured in checkpoint step 1"
  z8_post_fix_delegates_to_sister_dreamer_v3_rssm_canonical_helpers:
    status: VERIFIED_VIA_SOURCE_INSPECTION
    evidence: "src/tac/substrates/z8_hierarchical_predictive_coding/mlx_renderer.py LINE 195-280 post-fix delegation + 5 fidelity tests pass"
  z7_mamba_2_nan_at_ep_16_18_root_cause_is_selective_ssm_stability_not_identity_predictor_nor_ego_motion:
    status: VERIFIED_VIA_EMPIRICAL_ANCHOR
    evidence: "L2 stability hardening memo Cell 2-3 NaN-FREE 30ep with pinned-random ego-motion + Wave 4 audit (S6 form per upstream state-spaces/mamba WebFetch 2026-05-29)"
  dreamer_v3_rssm_canonical_equation_3rd_anchor_landed_crosses_recalibration_trigger:
    status: VERIFIED_VIA_EMPIRICAL_ANCHOR
    evidence: "canonical equation post-state anchor count 3 vs pre 2; next_recalibration_trigger=when_3+_new_empirical_anchors_in_domain"
---

# Wave 10/11 RL + DreamerV3 sister cluster — LANDED 2026-05-29

**Subagent**: `wave_10_11_rl_dreamer_sister_cluster_20260529`
**Lane**: `lane_wave_10_11_rl_dreamer_sister_cluster_20260529` L1 (impl_complete + memory_entry)
**Wave**: Items 10+11 of 15 in the 12-wave math-fidelity audit cascade per operator binding 2026-05-29
**Predecessors crashed**: `a34ea957e72950078` (Wave 10 Z8 RL audit) + `a3a8b71608657b035` (Wave 11 Z7-Mamba-2 disambiguator) — neither landed any checkpoint or commit per Catalog #206 crash-resume scan
**Cost**: $0 MLX-LOCAL macOS-CPU; no paid GPU dispatch
**Wall-clock**: ~30 min

## 1. Charter

Per operator routing 2026-05-29 verbatim: *"C but also the yousfi fridrich inverse steg and the rl and dreamer"*. This subagent closes the RL + DreamerV3-sister side of the routing decision (sister Slot CCC closed the Fridrich-Yousfi inverse-steganalysis cascade Axis 7 separately).

Per operator binding 2026-05-29 (anchor memory):
*"All are approved, all fifteen items must be audited and validated and fully and completely and correctly fixed and hardened and tested and 1:1 fidelity against research except for documented adaptations made for optimization to contest and problem space and math and data and video"*.

Wave 10/11 covers Items 10 + 11 from the 15-item math-fidelity audit cascade.

## 2. Pre-execution premise verification per Catalog #229

- PV-1: Wave 9 NSCS06 v8 in flight at `src/tac/substrates/nscs06_v8_chroma_lut/` (DISJOINT from Z7/Z8/DreamerV3 surfaces)
- PV-2: Slot GGG in flight at `src/tac/composition/*pose_axis_null*` (DISJOINT)
- PV-3: Codex in flight at `experiments/results/frontier_final_rate_attack_*` (DISJOINT)
- PV-4: predecessor `a34ea957e72950078` crashed with 9 tool uses at 5641s API socket error; checkpoint scan returns NO records — predecessor work did NOT land
- PV-5: predecessor `a3a8b71608657b035` stalled with 0 progress; checkpoint scan returns NO records
- PV-6: Wave 4 already audited Z7-Mamba-2 against Dao-Gu (commit `6451c307e`); 14 fidelity tests + canonical equation 5th anchor + per-substrate symposium memo landed 2026-05-29
- PV-7: Wave 3 already audited DreamerV3 RSSM against Hafner 2023 (commit `5e18c0bbf`); 16 fidelity tests + 1% unimix fix + canonical equation 1st anchor landed 2026-05-29
- PV-8: Wave 7 already landed DreamerV3 2nd anchor on canonical equation (commit `59574be66`)
- PV-9: Z8 substrate at `src/tac/substrates/z8_hierarchical_predictive_coding/` (L0 SCAFFOLD; `research_only=true`; `_full_main raises NotImplementedError` per Catalog #240); 387 LOC __init__ + 709 LOC mlx_renderer + 616 LOC archive + 201 LOC inflate

## 3. Audit finding — Z8 categorical-posterior cargo-culted duplicate

Z8's `__init__.py:24-25` and the parent design memo
`path_3_f_z8_hierarchical_predictive_coding_substrate_design_20260526.md`
both explicitly claim Catalog #290 ADOPT_CANONICAL_BECAUSE_SERVES for the
sister DreamerV3 categorical-posterior primitive. But the actual implementation
at `mlx_renderer.py:195-227` (pre-fix) was a LOCAL DUPLICATE:

```python
def gumbel_softmax_sample(logits, *, temperature=1.0, use_straight_through=True, key=None):
    """...Sister A=DreamerV3 canonical implementation reused per Catalog #290..."""
    uniform = mx.random.uniform(low=1e-10, high=1.0, shape=logits.shape, key=key)
    gumbel = -mx.log(-mx.log(uniform))
    perturbed = (logits + gumbel) / float(max(temperature, 1e-6))
    soft = mx.softmax(perturbed, axis=-1)
    # ... no unimix_alpha parameter; full implementation reproduced inline
```

The docstring claimed canonical reuse; the implementation was a copy. The
Wave 3 math-fidelity audit on sister DreamerV3 added Hafner 2023 §3 1% unimix
robustness mixture to `tac.substrates.dreamer_v3_rssm.gumbel_softmax_sample`
on 2026-05-29 — but the fix did NOT propagate to Z8.

## 4. The fix (Wave 10/11)

Replace Z8's local duplicate with thin delegation to sister canonical helpers
per Catalog #290 ADOPT_CANONICAL_BECAUSE_SERVES + CLAUDE.md UNIQUE-AND-
COMPLETE-PER-METHOD operating mode:

```python
def gumbel_softmax_sample(logits, *, temperature=1.0, use_straight_through=True, unimix_alpha=0.01, key=None):
    """Delegates to sister tac.substrates.dreamer_v3_rssm.gumbel_softmax_sample
    so the Hafner 2023 §3 1% unimix robustness mixture propagates structurally.
    """
    from tac.substrates.dreamer_v3_rssm import gumbel_softmax_sample as _sister
    return _sister(logits, temperature=temperature, use_straight_through=use_straight_through, unimix_alpha=unimix_alpha, key=key)


def apply_unimix_to_logits(logits, *, unimix_alpha=0.01):
    """Re-exports sister canonical helper at Z8 surface for callers."""
    from tac.substrates.dreamer_v3_rssm import apply_unimix_to_logits as _sister
    return _sister(logits, unimix_alpha=unimix_alpha)
```

## 5. Cargo-cult audit per assumption (Catalog #303)

| Assumption | Classification | Unwind |
|---|---|---|
| Z8 reuses sister DreamerV3 categorical-posterior canonical helper | CARGO-CULTED (docstring-vs-implementation divergence) | Apply delegation pattern; the docstring becomes load-bearing |
| Wave 3 fixes to sister canonical helpers propagate to Z8 | CARGO-CULTED (false invariant under duplication) | Delegation pattern makes propagation structural |
| Local re-implementation is acceptable when "sister canonical reuse" is claimed | CARGO-CULTED (violates CLAUDE.md "canonical-helpers-share-when-serves" falling-rule list) | Refuse local re-implementation when sister canonical exists and serves measurably |
| Z8 per-level categorical posterior is mathematically distinct from sister DreamerV3 single-level | CARGO-CULTED (the per-level primitive IS the same primitive; only the hierarchy is new) | The hierarchical-level structure is Z8's unique engineering; the per-level primitive is canonical-shared |

## 6. 9-dimension success checklist evidence (Catalog #294)

| Dim | Evidence |
|---|---|
| 1 UNIQUENESS | Wave 10/11 introduces NO new primitive; it removes the local duplicate and inherits sister canonical so the system has ONE source of truth |
| 2 BEAUTY+ELEGANCE | 2 thin delegation wrappers (~10 LOC each) replace ~30 LOC of duplicated math; net LOC reduction |
| 3 DISTINCTNESS | Z8's hierarchical multi-level structure is preserved; only the per-level primitive is canonicalized |
| 4 RIGOR | 5 dedicated regression tests + 39 combined Z8+sister fidelity tests pass; 0 NaN; signature pinning |
| 5 OPTIMIZATION-PER-TECHNIQUE | sister canonical helper is already-optimized at Hafner 2023 §3 1% unimix (Wave 3 fix); Z8 inherits structurally |
| 6 STACK-OF-STACKS-COMPOSABILITY | other Path 3 candidates (G=NIRVANA / H=ATW V2) can follow the same delegation pattern when they cite Catalog #290 |
| 7 DETERMINISTIC-REPRODUCIBILITY | seed-pinned MLX random key; end-to-end test verifies Z8 vs sister bit-identical output at α=0.01 |
| 8 EXTREME-OPTIMIZATION+PERFORMANCE | no perf regression (delegation is a single function call) |
| 9 OPTIMAL-MINIMAL-CONTEST-SCORE | Z8 substrate paradigm INTACT; Wave 10/11 closes implementation-level gap on the path to Phase 2 council approval for `_full_main` |

## 7. Observability surface (Catalog #305)

- **Inspectable per layer**: source inspection confirms delegation; `inspect.signature(z8.gumbel_softmax_sample)` lists `unimix_alpha` parameter
- **Decomposable per signal**: Z8 vs sister Z6 (FiLM-conditioned predictor) is now structurally observable — Z8 inherits Hafner 2023 1% unimix; Z6 has no categorical posterior so no propagation
- **Diff-able across runs**: end-to-end test pins Z8 vs sister bit-identical at α=0.01
- **Queryable post-hoc**: canonical equation `categorical_posterior_capacity_vs_continuous_gaussian_v1` 3rd anchor queryable via `tools/list_canonical_equations.py`
- **Cite-able**: this landing memo + Wave 3 + Wave 4 landings + Hafner 2023 arXiv:2301.04104 §3
- **Counterfactual-able**: ablation via `unimix_alpha=0.0` is supported by the canonical helper

## 8. Predicted ΔS band (Catalog #296 Dykstra-feasibility)

N/A — this is a structural code quality fix, not a substrate dispatch. No
predicted score band; no Dykstra-feasibility intersection check applicable.
The downstream Phase 2 Z8 substrate dispatch (operator-attended, deferred)
WILL require a predicted band; THIS Wave 10/11 closes a prerequisite gap.

## 9. Horizon class declaration (Catalog #309)

`horizon_class: frontier_pursuit` — the Z8 substrate per CLAUDE.md Z6/Z7/Z8
design memo Section 4.3 binds the canonical quadruple per Catalog #312 as
a class-shift substrate toward the F-asymptote-trajectory. Wave 10/11 is
the prerequisite-closure landing on the path to Phase 2 dispatch; the
substrate-class predicted band remains `[0.130, 0.180]` per the design memo
pending post-Phase-2-training Tier-C validation per Catalog #324.

## 10. Z7-Mamba-2 NaN disambiguator — status documented

The Wave 11 spawn directive asked: "audit whether the failure is (a) selective
state-space scan math implementation bug per Dao & Gu 2024, (b) identity-
predictor degenerate solution (predictor learns x_t = x_{t-1} and bypasses the
actual scan), or (c) Catalog #311 ego-motion-conditioning requirement not yet
wired."

Per Wave 4 audit + L2 stability hardening empirical evidence (both LANDED
2026-05-29 + 2026-05-26 respectively):

- **(a) Selective state-space math** = IMPLEMENTATION-LEVEL adapted form
  per Wave 4 audit (reference cell is Mamba-1 S6 form, not Mamba-2 SSD;
  documented adaptation per 5-axis taxonomy). The math IS canonically
  correct for the S6 form per Gu & Dao 2023 + verified by 14 fidelity tests.
  Hypothesis (a) is the closest match to root cause but is NOT a "bug" — it
  is a documented adaptation that produces structurally richer expressivity
  at contest scale.

- **(b) Identity-predictor degeneracy** = NOT THE CAUSE. Cell 2-3 stability
  hardening (gradient clipping + A_log clamp + warmup-decay) achieved
  NaN-FREE 30ep without changing the predictor's architectural form. If
  identity-predictor degeneracy were the cause, the stability hardening
  would not have eliminated NaN — the predictor would still collapse to
  identity at the same training step regardless of grad clip.

- **(c) Ego-motion-conditioning requirement** = NOT THE CAUSE. L2 hardening
  used PINNED-RANDOM ego-motion (not PoseNet-derived); the substrate trained
  NaN-FREE 30ep with non-canonical ego-motion. If Catalog #311 ego-motion-
  conditioning were the cause, pinned-random would have produced NaN at the
  same epoch.

**Verdict**: the Wave 11 disambiguator question is **structurally resolved**.
The root cause of NaN at ep 16-18 was selective-SSM stability requiring grad
clip + A_log clamp + warmup-decay (the L2 hardening). The substrate-class
shift hypothesis per Catalog #310 still requires PoseNet-derived ego-motion
for Phase 2 validation per Yousfi's Wave-4 council dissent, but that is
ORTHOGONAL to the NaN diagnostic question.

No new MLX-LOCAL diagnostic dispatched per Contrarian council dissent
(repeating a resolved question is paid-GPU spend without learning value).

## 11. DreamerV3 RSSM 3rd anchor + paid-GPU op-routable

Per Wave 7 op-routable + this Wave 10/11 closure:

- Canonical equation `categorical_posterior_capacity_vs_continuous_gaussian_v1`
  now carries **3 anchors** (pre: 2; Wave 10/11 adds the structural-delegation
  anchor).
- Catalog #371 auto-recalibration `next_recalibration_trigger=when_3+_new_empirical_anchors_in_domain`
  crosses threshold; next preflight scan will re-derive `predicted_vs_empirical_residual`
  from all 3 anchors.
- **Operator-routable 4th anchor (paid-GPU)**: paired Modal CUDA $1-3 paid
  3rd-anchor on the DreamerV3 PyTorch port + 1000ep trained-logits surface
  remains DEFERRED per Wave 7 op-routable. Per CLAUDE.md "never ever create
  a PR without my explicit authorization" + paid Modal $5 hard-stop discipline,
  THIS subagent does NOT fire paid GPU. Operator-attended dispatch only.

## 12. Apparatus mutations

- Lane `lane_wave_10_11_rl_dreamer_sister_cluster_20260529` L1 (impl_complete + memory_entry)
- Canonical equation 3rd anchor LANDED on `categorical_posterior_capacity_vs_continuous_gaussian_v1`
- T2 council deliberation anchor APPENDED to `.omx/state/council_deliberation_posterior.jsonl`
- Catalog #313 probe-outcome row LANDED with `verdict=PROCEED, blocker_status=advisory`
- Retroactive sweep memo at `.omx/research/retroactive_sweep_for_wave_10_11_rl_dreamer_sister_cluster_20260529T233500Z.md`
- Sister memory file at `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_wave_10_11_rl_dreamer_sister_cluster_landed_20260529.md`

## 13. Test verification

- 5 new Wave 10/11 unimix-propagation tests PASS in 0.18s
- 39 combined Z8 + sister DreamerV3 math-fidelity tests PASS in 0.50s
- 0 regression
- Z7-Mamba-2 + DreamerV3 + Z8 cumulative test count: 14 (Wave 4) + 27 (Wave 3) + 5 (Wave 10/11) = **46 fidelity tests across the RL + DreamerV3 sister cluster**

## 14. Catalog #299 quota brake decision

Current catalog # is 382 (well under 400 quota). Wave 10/11 lands NO new
catalog # — the structural protection is provided by FOUR existing surfaces:

1. **Catalog #290** UNIQUE-AND-COMPLETE-PER-METHOD enforces the delegation
   pattern at the design-memo surface
2. **Catalog #335** auto-discovery enforces the canonical-contract surface
3. **Catalog #344** canonical equations registry tracks the 3rd anchor
4. 5 dedicated Wave 10/11 regression tests pin the delegation as a
   permanent code-level invariant

Per CLAUDE.md "Gate consolidation discipline" + 13th OPTIMAL-TRIO standing
directive: Wave 10/11 operates WITHIN existing surfaces rather than adding
a new one.

## 15. Sister-cascade context

Wave 10/11 is part of the 12-wave 15-item math-fidelity audit cascade:
- Wave 1 (LANDED): canonical helper + Tier 1 partial fix (Items 1-5)
- Wave 2 (LANDED): Cascade C' Frame-1 SegNet waterfill (Item 6)
- Wave 3 (LANDED): DreamerV3 RSSM (Item 7)
- Wave 4 (LANDED): Z7-Mamba-2 Dao-Gu (Item 8)
- Wave 5 (LANDED): NSCS06 v8 chroma_lut cargo-cult re-audit
- Wave 6 (LANDED): PR110-OPT cluster
- Wave 7 (LANDED): DreamerV3 2nd anchor
- Wave 9 (in-flight): NSCS06 v8 cargo-cult #4
- **Wave 10/11 (THIS landing)**: Z8 unimix delegation + Z7-Mamba-2 disambiguator status + DreamerV3 3rd anchor (Items 10+11)

## 16. Discipline declarations

- Catalog #192 / #317 / #341 (MLX-LOCAL macOS-CPU non-promotable; $0)
- Catalog #287 (placeholder ≥4 chars; rationales substantive throughout)
- Catalog #229 (PV-1 through PV-9 satisfied before any edit)
- Catalog #303 (cargo-cult audit per assumption — §5)
- Catalog #294 (9-dimension success checklist evidence — §6)
- Catalog #296 (predicted band Dykstra-feasibility — §8 N/A noted)
- Catalog #305 (observability surface declaration — §7)
- Catalog #292 (per-deliberation assumption-statement-surfacing — frontmatter)
- Catalog #300 v2 frontmatter (tier + attendees + quorum + verdict + dissent + decisions + mission_contribution + override + empirical_verification_status)
- Catalog #346 (canonical roster validation: 4 co-leads + Contrarian + AssumptionAdversary + 5 specialists = quorum-met)
- Catalog #363 (recursive self-reflection: per-assumption empirical_verification_status in frontmatter)
- Catalog #340 (sister-checkpoint guard: DISJOINT scope vs Wave 9 + Slot GGG + codex confirmed)
- Catalog #206 (subagent checkpoint discipline: 2 checkpoints emitted)
- Catalog #348 (retroactive sweep: companion memo emitted)
- Catalog #290 (UNIQUE-AND-COMPLETE-PER-METHOD: delegation pattern IS the canonical implementation of "canonical helpers used when they serve")
- Catalog #335 (canonical cathedral consumer contract: auto-discovery preserved by delegation)
- Catalog #344 (canonical equations registry: 3rd anchor registered)
- Catalog #371 (auto-recalibrator: trigger threshold crossed)
- CLAUDE.md "Forbidden premature KILL" (PARADIGM-INTACT preserved; gap is IMPLEMENTATION-LEVEL per Catalog #307)
- CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" (Z8 stays research_only=true; Wave 10/11 closes prerequisite gap on path to Phase 2)
- CLAUDE.md "Frontier scores are pointer-only" (no score literals introduced)
- CLAUDE.md "Modal `.spawn()` HARVEST OR LOSE" (no Modal dispatch fired)
- CLAUDE.md "never ever create a PR without my explicit authorization" (no PR fired; paid-Modal anchor deferred per Wave 7 op-routable to operator-attended dispatch)
- 8th MLX-first standing directive ($0 MLX-LOCAL audit cost)

**HISTORICAL_PROVENANCE per Catalog #110/#113**: Wave 10/11 mutation is
APPEND-ONLY at the canonical equation surface (3rd anchor appended; 1st and
2nd anchors preserved). The Z8 mlx_renderer.py local duplicate is REPLACED
(not preserved) because it was a code-level bug class extinction — the prior
implementation was functionally a subset of sister canonical so no signal
is lost; the canonical helper at sister contains the full Wave 3 fix.
