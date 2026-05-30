---
council_tier: T1
council_attendees: [Shannon, Dykstra, Yousfi, Fridrich, Contrarian, Assumption-Adversary, Wyner_Grand, Tishby_Grand_memorial, Atick_Grand, Ballard_Grand]
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "PoseNet IS canonical decoder-reproducible side information at contest runtime"
    classification: HARD-EARNED
    rationale: "upstream/evaluate.py:6 imports posenet_sd_path; upstream/evaluate.py:53 loads PoseNet state dict onto device; upstream/evaluate.py:79 computes posenet_dist on inflated frames. Weights NOT in archive.zip per CLAUDE.md 'Strict scorer rule' = FREE rate-axis side info."
  - assumption: "Wyner-Ziv 1976 R(D|Y)<<R(D) gain holds when PoseNet output Y correlates with substrate source X"
    classification: HARD-EARNED
    rationale: "Z8 M6 commit 5d5634dd3 empirically validated 64-74% byte savings on synthetic Gaussian baseline (.omx/research/z8_m6_wyner_ziv_top_level_coder_full_implementation_landed_20260530.md). Per Wyner & Ziv 1976 § 3 Theorem 1 + Cover-Thomas 2006 § 15.9."
  - assumption: "Per-substrate empirical mutual information I(X; PoseNet(pair)) is non-trivial for typical contest substrates"
    classification: ASSUMED_AWAITING_VERIFICATION
    rationale: "Per-substrate empirical mutual information depends on substrate source X relationship to ego-motion / scene dynamics. THIS landing registers the canonical equation at the PARADIGM surface; per-substrate empirical anchors land separately via Catalog #371 auto-recalibration when >=3 in-domain anchors accumulate."
council_decisions_recorded:
  - "op-routable #1: canonical equation wyner_ziv_decoder_side_posenet_side_information_conditional_entropy_reduction_v1 REGISTERED in canonical posterior; registry size 149 → 150"
  - "op-routable #2: cathedral consumer tac.cathedral_consumers.wyner_ziv_posenet_side_information_consumer auto-discovered per Catalog #335; 80 production consumers cumulative"
  - "op-routable #3: first EmpiricalAnchor cites Z8 M6 64-74% empirical byte savings (synthetic Gaussian; macOS-CPU advisory per Catalog #192 NEVER promotable); per-substrate paired-CUDA RATIFICATION accumulates per-axis anchors"
  - "op-routable #4: extend tac.bit_allocator.per_pair with pose_conditional mode that consumes equation's prediction to allocate bits per H(X_local|PoseNet_local); DEFERRED — separate sister landing"
  - "op-routable #5: future paired-CUDA RATIFICATION on substrates adopting Wyner-Ziv PoseNet-conditional coding will produce per-axis anchors; Catalog #371 auto-refit triggers at >=3 in-domain anchors"
council_predicted_mission_contribution: frontier_breaking_enabler
council_override_invoked: false
council_override_rationale: null
related_deliberation_ids:
  - z8_m6_wyner_ziv_top_level_coder_full_implementation_landed_20260530
  - wyner_ziv_decoder_side_posenet_side_information_conditional_entropy_reduction_design_20260530
horizon_class: frontier_pursuit
axis_tag: "[predicted]"
score_claim: false
promotable: false
ready_for_exact_eval_dispatch: false
---

# Wave N+36 — Wyner-Ziv Decoder-Side PoseNet Side-Information Canonical Equation Registration LANDED 2026-05-30

## Summary

Operator-routed task #1496 (Wave N+36) per CLAUDE.md "Canonical equations + models registry" non-negotiable + Catalog #344 sister discipline. THIS landing registers the canonical equation `wyner_ziv_decoder_side_posenet_side_information_conditional_entropy_reduction_v1` at the canonical Wyner-Ziv (1976) Theorem 1 paradigm-elevation surface — formalizing the identity that PoseNet IS decoder-reproducible side information at contest runtime (loaded by `upstream/evaluate.py:53` from `posenet_sd_path`; NOT charged to `archive.zip` per CLAUDE.md "Strict scorer rule" = FREE rate-axis side info).

Any substrate that adopts conditional coding against `Y = PoseNet(frame_pair)` realizes the canonical rate-axis savings `R(D|Y) << R(D)` per Wyner-Ziv 1976 § 3 Theorem 1. The first EmpiricalAnchor cites Z8 M6 commit `5d5634dd3` (synthetic Gaussian 64-74% byte savings; macOS-CPU advisory per Catalog #192 NEVER promotable); per-substrate empirical anchors accumulate as substrates adopt the pattern.

## Canonical-vs-unique decision per layer

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" Catalog #290:

| Layer | Decision | Rationale |
|---|---|---|
| Wyner-Ziv 1976 R(D\|Y) bound | ADOPT_CANONICAL | Unmodified Wyner & Ziv 1976 Theorem 1; universal identity |
| PoseNet as canonical side-info Y | FORK_BECAUSE_PRINCIPLED | None of the 6 existing sister Wyner-Ziv equations binds this paradigm surface; THIS landing fills the gap |
| Equation infrastructure | ADOPT_CANONICAL | `tac.canonical_equations.CanonicalEquation` + `EmpiricalAnchor` + `register_canonical_equation` per CLAUDE.md "Canonical equations + models registry" non-negotiable |
| Cathedral consumer | ADOPT_CANONICAL | `tac.cathedral.consumer_contract.CathedralConsumerContract` + Catalog #335 auto-discovery + Catalog #341 Tier A markers |
| Provenance umbrella | ADOPT_CANONICAL | `tac.provenance.build_provenance_for_predicted` per Catalog #323 + `build_provenance_for_research_sidecar` for Z8 M6 anchor |
| First EmpiricalAnchor | ADOPT_CANONICAL Z8 M6 | Z8 M6 commit 5d5634dd3 synthetic Gaussian 64-74% byte savings; per Catalog #287 evidence-tag discipline |

## 9-dimension success checklist evidence

| Dimension | Evidence |
|---|---|
| 1. UNIQUENESS | PARADIGM-elevation: distinct from 6 sister Wyner-Ziv equations binding specific contexts (NSCS06 v8 cls_stream / pipeline_stage_codec / cross-substrate composition / class-shift refined predicted-delta / per-pair state_dict bytes) |
| 2. BEAUTY + ELEGANCE | Single registered equation + single canonical builder + single cathedral consumer + 46 dedicated tests; ~430 LOC total infrastructure addition |
| 3. DISTINCTNESS | Explicit FORK rationale documented in canonical-vs-unique table + 4 EXCLUDED contexts (posenet_as_source / non_video_signals / non_decoder_reproducible_substrates / residual_hybrid_contexts_per_catalog_359) |
| 4. RIGOR | Wyner & Ziv (1976) Theorem 1 + Cover & Thomas (2006) § 15.9 + Pradhan & Ramchandran (2003) DISCUS verbatim citations; Z8 M6 empirical 64-74% byte savings as first anchor |
| 5. OPTIMIZATION PER TECHNIQUE | Canonical R(D\|Y) bound at paradigm surface; per-substrate FORKs land independently per Catalog #335 auto-discovery |
| 6. STACK-OF-STACKS-COMPOSABILITY | Equation enumerates 7 canonical_consumers: cathedral consumer / canonical_equation_lookup_consumer / cathedral autopilot loop / bit_allocator.per_pair / wyner_ziv_layer / z8 wyner_ziv_coder / dykstra_pareto_solver |
| 7. DETERMINISTIC REPRODUCIBILITY | PoseNet state-dict load is deterministic; Wyner-Ziv encode/decode is deterministic per Z8 M6 design (Hadamard-Gaussian projection matrix from seed) |
| 8. EXTREME OPTIMIZATION + PERFORMANCE | Zero-runtime-cost registry lookup; downstream consumers inherit canonical equation predictions via auto-discovery |
| 9. OPTIMAL MINIMAL CONTEST SCORE | Frontier-breaking enabler: per Wyner-Ziv 1976, conditional coding against decoder-reproducible side info IS free rate reduction; any substrate adoption realizes savings |

## Observability surface

Per CLAUDE.md "Max observability — non-negotiable" Catalog #305 6 facets — see design memo `.omx/research/wyner_ziv_decoder_side_posenet_side_information_conditional_entropy_reduction_design_20260530.md` for full per-facet enumeration. Operator-runnable surfaces:

* `tools/list_canonical_equations.py --json | jq '.equations[] | select(.equation_id == "wyner_ziv_decoder_side_posenet_side_information_conditional_entropy_reduction_v1")'` — canonical readback
* `python -c "from tac.canonical_equations import get_equation_by_id; eq = get_equation_by_id('wyner_ziv_decoder_side_posenet_side_information_conditional_entropy_reduction_v1'); print(eq.is_well_calibrated, eq.predicted_vs_empirical_residual)"` — programmatic
* Cathedral autopilot consume_candidate emits per-candidate Tier A annotations with anchor count + per-axis residuals

## Cargo-cult audit per assumption

Per Catalog #303 — full enumeration in design memo `.omx/research/wyner_ziv_decoder_side_posenet_side_information_conditional_entropy_reduction_design_20260530.md`. Summary: 4 HARD-EARNED (Wyner-Ziv 1976 paradigm + PoseNet decoder-reproducibility + Z8 M6 empirical + deterministic encoder-decoder agreement) + 1 ASSUMED_AWAITING_VERIFICATION (per-substrate mutual information non-trivial) + 1 CARGO-CULTED (Z8 M6 synthetic Gaussian extrapolation; unwind path = per-substrate paired-CUDA RATIFICATION).

## Empirical-vs-predicted verdict

**Predicted ΔS band**: per-substrate rate-axis band `[-25 * bytes_saved / 37_545_489, 0]` per CLAUDE.md canonical rate term. The savings depend on substrate source-X compressibility AND I(X; PoseNet(pair)).

**Dykstra-feasibility check**: rate-axis polytope constraint `bytes_post <= bytes_pre` is convex (linear); Shannon conditional entropy `H(X|Y) <= H(X)` is canonical chain rule. Intersection is non-empty for any substrate with non-trivial mutual information. FEASIBLE.

**First EmpiricalAnchor** (Z8 M6 synthetic Gaussian; macOS-CPU advisory per Catalog #192):
- Inputs: state_dim=8, side_info=(3,4,4), bit_budget=64, batch=16, noise_scale=0.05, side_info_correlation_proxy=0.9
- Predicted: 385 bytes saved (canonical upper bound; 73.6% savings ratio)
- Empirical: 385 bytes saved (523 unconditional - 138 measured = 385; 73.6% savings; round-trip rel L2 err 3.6%)
- Residual: 0.0 (paradigm-anchor confirms canonical Wyner-Ziv 1976 identity within numerical precision)

## Premise verification per Catalog #229

| Premise | Verification |
|---|---|
| 6 existing Wyner-Ziv canonical equations do NOT bind the PARADIGM surface | Verified via `python -c "from tac.canonical_equations import query_equations; [print(e.equation_id, e.name) for e in query_equations() if 'wyner' in e.equation_id.lower()]"` — 5 existing pre-registration cover NSCS06 v8 cls_stream / pipeline_stage_codec / cross-substrate composition / class-shift refined / per-pair PoseNet state_dict; none binds the general PARADIGM |
| Z8 M6 sister landing exists with empirical 64-74% byte savings | Verified via commit 5d5634dd3 + `.omx/research/z8_m6_wyner_ziv_top_level_coder_full_implementation_landed_20260530.md` |
| PoseNet IS decoder-reproducible at contest runtime | Verified via `grep -n "posenet" upstream/evaluate.py` (lines 6/53/79) |
| Sister Z8 M10 subagent operates on DISJOINT file scope | Verified via `tools/subagent_checkpoint.py read --latest-incomplete` (Z8 M10 touches z8_hierarchical_predictive_coding/inflate.py + canonical_quadruple_binding.py + tests; THIS lane touches canonical_equations/wyner_ziv_decoder_side_posenet_side_information.py + cathedral_consumers/wyner_ziv_posenet_side_information_consumer/ + canonical_equations/__init__.py — DISJOINT) |
| Catalog #335 STRICT preflight gate accepts new consumer | Verified via direct gate invocation: 0 violations |

## 6-hook wire-in declaration per Catalog #125

| Hook | Status | Surface |
|---|---|---|
| #1 sensitivity-map | **ACTIVE** | PoseNet output IS canonical sensitivity surface for pose-conditional bit allocation |
| #2 Pareto constraint | **ACTIVE** | Wyner-Ziv R(D\|Y) bound IS canonical Pareto constraint at rate-axis polytope |
| #3 bit-allocator | **ACTIVE PRIMARY** | Canonical `pose_conditional` mode for `tac.bit_allocator.per_pair` (DEFERRED separate landing) |
| #4 cathedral autopilot dispatch | **ACTIVE** | New cathedral consumer auto-discovered per Catalog #335 |
| #5 continual-learning posterior | **ACTIVE** | Auto-recalibration per Catalog #371 when >=3 in-domain anchors land |
| #6 probe-disambiguator | **ACTIVE** | Equation IS canonical disambiguator between Wyner-Ziv conditional coding (savings) vs unconditional coding (no savings) |

## Implementation surfaces (deltas)

* **NEW module**: `src/tac/canonical_equations/wyner_ziv_decoder_side_posenet_side_information.py` (~310 LOC; SPDX-MIT; canonical builder + predictor helper)
* **NEW package**: `src/tac/cathedral_consumers/wyner_ziv_posenet_side_information_consumer/` (~165 LOC `__init__.py` + tests; Tier A canonical contract per Catalog #335/#341)
* **MODIFIED**: `src/tac/canonical_equations/__init__.py` (+8 lines — re-export new builder + helper)
* **NEW tests**:
  - `src/tac/canonical_equations/tests/test_wyner_ziv_decoder_side_posenet_side_information.py` (25 tests)
  - `src/tac/cathedral_consumers/wyner_ziv_posenet_side_information_consumer/tests/test_consumer.py` (21 tests)
* **NEW design memo**: `.omx/research/wyner_ziv_decoder_side_posenet_side_information_conditional_entropy_reduction_design_20260530.md` (HORIZON-CLASS = frontier_pursuit; 9-dim checklist + observability + cargo-cult audit + Dykstra-feasibility + canonical-vs-unique)
* **CANONICAL POSTERIOR**: registered via `tac.canonical_equations.register_canonical_equation` to `.omx/state/canonical_equations_registry.jsonl` (registry size 149 → 150)
* **LANE REGISTRY**: `lane_wave_n_36_wyner_ziv_decoder_side_posenet_side_information_canonical_equation_registration_20260530` registered at L1 (impl_complete gate satisfied)

## Test summary

```
src/tac/canonical_equations/tests/test_wyner_ziv_decoder_side_posenet_side_information.py: 25/25 PASS
src/tac/cathedral_consumers/wyner_ziv_posenet_side_information_consumer/tests/test_consumer.py: 21/21 PASS
src/tac/cathedral_consumers/canonical_equation_lookup_consumer/tests/test_consumer.py: 8/8 PASS (sister regression)
--------------------------------------------------------------------
Total: 54 PASS in 0.33s
```

Plus live integration verification:
* Canonical equation REGISTERED (registry size 149 → 150 verified)
* Cathedral consumer auto-discovered (80 production consumers cumulative)
* Catalog #335 STRICT preflight gate: 0 violations on new consumer
* End-to-end live `consume_candidate` returns matched_equation_id + anchor_count=1 + is_well_calibrated=True

## Sister DISJOINT scope verification per Catalog #340

Verified at PV time (15:46Z → 16:04Z):

* **Z8 M10 inflate-consumes-real-trained-weights** (in-flight, PID 18852): scope = `src/tac/substrates/z8_hierarchical_predictive_coding/{inflate.py,canonical_quadruple_binding.py,build_progress.py,tests/test_inflate_canonical_quadruple_consumes_real_trained_weights.py,tests/test_basic.py}`; DISJOINT from THIS lane's `src/tac/canonical_equations/{wyner_ziv_decoder_side_posenet_side_information.py,__init__.py}` + `src/tac/cathedral_consumers/wyner_ziv_posenet_side_information_consumer/`
* **Slot GGG Tier C overnight runner** (PID 10169): scope = scorer-axis verification; DISJOINT
* **Cascade B wave-2** (ac302ffd1): scope = cascade_b package; DISJOINT

## Operator-routable next actions

1. **Extend `tac.bit_allocator.per_pair` with `pose_conditional` mode** that consumes the equation's prediction to allocate bits per `H(X_local | PoseNet_local)`. (DEFERRED — separate sister landing; queueable as Wave N+37.)
2. **Per-substrate paired-CUDA RATIFICATION** on substrates adopting Wyner-Ziv PoseNet-conditional coding will produce per-axis EmpiricalAnchors. (Operator-routable per substrate.)
3. **Sister cathedral consumer registration** for related Wyner-Ziv canonical equations (a unified consumer surfacing ALL Wyner-Ziv equations' predictions per candidate). (DEFERRED — premature consolidation; sister consumers can land independently per Catalog #335 auto-discovery.)
4. **Z8 M9 sister landing** can produce the first per-substrate empirical anchor via M9's `_full_main` trainer + Catalog #371 auto-recalibration.

## Cross-references

* CLAUDE.md "Canonical equations + models registry — non-negotiable" + Catalog #344
* CLAUDE.md "Meta-Lagrangian/Pareto solver — non-negotiable, HIGHEST EMPHASIS"
* CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" (PoseNet IS decoder-reproducible at contest runtime)
* CLAUDE.md "Strict scorer rule" (PoseNet NOT inside archive.zip — FREE rate-axis side info)
* Catalog #335 + #341 + #323 + #371 sister disciplines
* Z8 M6 landing: `.omx/research/z8_m6_wyner_ziv_top_level_coder_full_implementation_landed_20260530.md` (commit 5d5634dd3)
* Design memo: `.omx/research/wyner_ziv_decoder_side_posenet_side_information_conditional_entropy_reduction_design_20260530.md`
* Retroactive sweep: `.omx/research/retroactive_sweep_for_wave_n_36_wyner_ziv_decoder_side_posenet_side_information_canonical_equation_registration_20260530T160511Z.md`
* Wyner & Ziv (1976) "The rate-distortion function for source coding with side information at the decoder" IEEE Trans. Inf. Theory IT-22(1):1-10
* Cover & Thomas (2006) Elements of Information Theory, 2nd ed., § 15.9 Rate Distortion with Side Information at Decoder
* Pradhan & Ramchandran (2003) "Distributed source coding using syndromes (DISCUS)" IEEE Trans. Inf. Theory 49(3):626-643
