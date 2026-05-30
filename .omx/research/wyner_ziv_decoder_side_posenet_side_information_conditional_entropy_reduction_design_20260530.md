---
council_tier: T1
council_attendees: [Shannon, Dykstra, Yousfi, Fridrich, Contrarian, Assumption-Adversary, Wyner_Grand, Tishby_Grand_memorial, Atick_Grand, Ballard_Grand]
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "PoseNet output IS decoder-reproducible at contest-runtime because upstream/evaluate.py loads PoseNet and computes posenet_dist on already-inflated frames"
    classification: HARD-EARNED
    rationale: "upstream/evaluate.py:6 imports posenet_sd_path; upstream/evaluate.py:53 loads state dict onto device; upstream/evaluate.py:79 computes posenet_dist from batched ground-truth pose pairs. The PoseNet weights ARE part of the contest runtime stack (NOT inside archive.zip per CLAUDE.md 'Strict scorer rule'). Any decoder that runs `inflate.sh` AND has access to the contest evaluator runtime can compute PoseNet(decoded_frames_pair) without any rate-axis cost — the weights are not charged."
  - assumption: "Wyner-Ziv 1976 R(D|Y) << R(D) gain manifests when source X correlates with PoseNet output Y"
    classification: HARD-EARNED
    rationale: "Z8 M6 empirically validated at 64-74% byte savings vs unconditional zlib baseline on synthetic Gaussian data (commit 5d5634dd3 + .omx/research/z8_m6_wyner_ziv_top_level_coder_full_implementation_landed_20260530.md). The savings ratio increases with side-info correlation per Wyner-Ziv 1976 § 3 Theorem 1 (as I(X;Y) increases, R(D|Y) shrinks)."
  - assumption: "PoseNet output dimensionality at contest scale (12-dim head, 6 used per modules.py) is sufficient as decoder side info"
    classification: ASSUMED_AWAITING_VERIFICATION
    rationale: "PoseNet outputs a 12-dim pose head per upstream/modules.py:177 line 70-ish FastViT-T12 backbone (first 6 used per CLAUDE.md 'Exact scorer architectures'). Whether 6 floats per pair carry sufficient mutual information I(X; PoseNet(pair)) to drive meaningful conditional-entropy reduction on a typical substrate's source X depends on the specific substrate's relationship to ego-motion / scene dynamics. THIS WAVE registers the canonical equation; future EmpiricalAnchors from real substrate dispatches will land per-substrate residuals."
  - assumption: "The canonical contest decoder runtime (upstream/evaluate.py --device cpu/cuda) IS the operationally available decoder for Wyner-Ziv side-info purposes"
    classification: HARD-EARNED
    rationale: "Per CLAUDE.md 'Submission auth eval — BOTH CPU AND CUDA' non-negotiable, every shipped archive runs through upstream/evaluate.py which loads PoseNet weights and computes posenet_dist on the inflated frames. The 'decoder' in the canonical Wyner-Ziv sense IS that runtime."
council_decisions_recorded:
  - "op-routable #1: register canonical equation wyner_ziv_decoder_side_posenet_side_information_conditional_entropy_reduction_v1 per Catalog #344 with first EmpiricalAnchor citing Z8 M6 64-74% empirical savings as the synthetic-Gaussian baseline anchor (HARD-EARNED at the paradigm surface; PENDING-VERIFICATION at the per-substrate contest-deployment surface)"
  - "op-routable #2: land cathedral consumer tac.cathedral_consumers.wyner_ziv_posenet_side_information_consumer per Catalog #335 auto-discovery + Catalog #341 Tier A canonical-routing markers (predicted_delta_adjustment=0.0 + promotable=False + axis_tag=[predicted])"
  - "op-routable #3: future paired-CUDA RATIFICATION on any substrate that adopts Wyner-Ziv PoseNet-conditional coding will produce a second EmpiricalAnchor on the per-substrate axis; the canonical equation accumulates anchors per Catalog #371 auto-recalibration when >=3 in-domain anchors land"
council_predicted_mission_contribution: frontier_breaking_enabler
council_override_invoked: false
council_override_rationale: null
related_deliberation_ids:
  - z8_m6_wyner_ziv_top_level_coder_full_implementation_landed_20260530
  - wyner_ziv_pipeline_stage_codec_decoder_side_canonical_y_savings_v1_registration_anchor
  - wyner_ziv_per_pair_posenet_output_y_pose_axis_savings_v1_registration_anchor
horizon_class: frontier_pursuit
axis_tag: "[predicted]"
score_claim: false
promotable: false
ready_for_exact_eval_dispatch: false
---

# Wyner-Ziv Decoder-Side PoseNet Side-Information Conditional Entropy Reduction — Canonical Equation Registration Design

## Summary

Operator-routed task #1496 (Wave N+36) per CLAUDE.md "Canonical equations + models registry" non-negotiable + Catalog #344 sister discipline. THIS landing registers the canonical equation `wyner_ziv_decoder_side_posenet_side_information_conditional_entropy_reduction_v1` formalizing the canonical Wyner-Ziv (1976) Theorem 1 paradigm-elevation: PoseNet IS canonical decoder-reproducible side information (the contest evaluator `upstream/evaluate.py` loads PoseNet weights and computes per-pair pose distortion on the inflated frames; these weights are NOT charged to `archive.zip` per CLAUDE.md "Strict scorer rule"). Any substrate that adopts conditional coding against PoseNet(frame_pair) as the side-info Y can encode its source X at rate `R(D|Y) << R(D)` per Wyner-Ziv 1976 Theorem 1.

The canonical equation is intentionally at the PARADIGM-ELEVATION surface (general decoder-side PoseNet side-info coding), DISTINCT from the 6 existing sister Wyner-Ziv equations which bind specific application contexts (NSCS06 v8 cls_stream / pipeline_stage_codec / cross-substrate composition / class-shift refined predicted-delta / per-pair PoseNet output Y pose-axis savings). The Z8 M6 empirical 64-74% byte savings (synthetic Gaussian; macOS-local-CPU advisory per Catalog #192 NEVER promotable) lands as the first EmpiricalAnchor at the paradigm-surface, anchoring the registry's residual posterior so future per-substrate empirical anchors can compound.

## Canonical-vs-unique decision per layer

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" Catalog #290:

| Layer | Decision | Rationale |
|---|---|---|
| Wyner-Ziv 1976 R(D\|Y) bound | ADOPT_CANONICAL | The canonical conditional-entropy reduction theorem is the unmodified Wyner & Ziv 1976 Theorem 1. The equation form `R(D|Y) <= R(D)` is canonical and applies universally to any (source X, side-info Y) pair with non-trivial mutual information. |
| PoseNet as side-info Y | FORK_BECAUSE_PRINCIPLED | None of the 6 existing sister Wyner-Ziv equations binds PoseNet as canonical decoder side-info at the paradigm surface. `wyner_ziv_per_pair_posenet_output_y_pose_axis_savings_v1` binds the pose-axis savings sub-case (per-substrate state_dict bytes savings on the pose axis); THIS equation binds the general paradigm. |
| Equation infrastructure | ADOPT_CANONICAL | `tac.canonical_equations.CanonicalEquation` + `EmpiricalAnchor` frozen dataclasses + `register_canonical_equation` fcntl-locked JSONL persistence. Per CLAUDE.md "Canonical equations + models registry" non-negotiable. |
| Cathedral consumer auto-discovery | ADOPT_CANONICAL | `tac.cathedral.consumer_contract.CathedralConsumerContract` + Catalog #335 auto-discovery. The new consumer at `src/tac/cathedral_consumers/wyner_ziv_posenet_side_information_consumer/` follows the canonical Tier A pattern (predicted_delta_adjustment=0.0 + promotable=False + axis_tag=[predicted] per Catalog #341). |
| Provenance umbrella | ADOPT_CANONICAL | `tac.provenance.build_provenance_for_predicted` per Catalog #323. The first EmpiricalAnchor's Provenance carries the canonical PREDICTED grade since the empirical savings come from synthetic Gaussian (macOS-local-CPU advisory per Catalog #192 NEVER promotable). |
| First EmpiricalAnchor | ADOPT_CANONICAL Z8 M6 anchor | The Z8 M6 landing (commit 5d5634dd3) measured 64-74% byte savings on synthetic Gaussian + the canonical Wyner-Ziv 1976 R(D|Y) bound; this is the PARADIGM-anchor (the closed-form mathematical identity is canonical-validated). Per-substrate contest-deployment anchors land separately as Catalog #371 triggers auto-recalibration. |

## 9-dimension success checklist evidence

| Dimension | Evidence |
|---|---|
| 1. UNIQUENESS | PARADIGM-elevation: this is the canonical decoder-side PoseNet side-info equation at the general substrate-applicable surface, distinct from the 6 sister Wyner-Ziv equations which bind specific application contexts. |
| 2. BEAUTY + ELEGANCE | Single registered equation + single canonical builder + single cathedral consumer + first empirical anchor; ~250 LOC total. Builds on existing Z8 M6 infrastructure; zero new substrate code. |
| 3. DISTINCTNESS | Explicit FORK rationale documented (see "Canonical-vs-unique" table). The equation is reusable across substrates (the canonical_consumers list enumerates current + future cathedral consumers). |
| 4. RIGOR | Wyner & Ziv (1976) Theorem 1 cited verbatim; Cover & Thomas (2006) § 15.9 + Pradhan & Ramchandran (2003) DISCUS canonical citations; Z8 M6 empirical 64-74% byte savings (synthetic Gaussian) anchors the residual posterior. |
| 5. OPTIMIZATION PER TECHNIQUE | The canonical Wyner-Ziv 1976 R(D|Y) bound is the canonical formula; no substrate-specific quirks at the paradigm surface. Per-substrate FORKs land separately. |
| 6. STACK-OF-STACKS-COMPOSABILITY | The equation explicitly enumerates `canonical_consumers` (cathedral autopilot ranker / bit-allocator / canonical Wyner-Ziv layer / Z8 hierarchical predictive coding) so downstream stacking inherits the canonical surface. |
| 7. DETERMINISTIC REPRODUCIBILITY | PoseNet weights are deterministic (state-dict loaded from `posenet_sd_path`); deterministic forward pass on canonical YUV6 input → deterministic side-info Y. The Wyner-Ziv encode/decode is deterministic given the same projection seed (sister Z8 M6 design). |
| 8. EXTREME OPTIMIZATION + PERFORMANCE | Apparatus-maintenance value primarily; downstream consumers (bit-allocator + cathedral ranker) inherit the canonical equation prediction at ZERO runtime cost (registry lookup). |
| 9. OPTIMAL MINIMAL CONTEST SCORE | Frontier-breaking enabler: per Wyner-Ziv 1976, conditional coding against decoder-reproducible side-info IS free rate reduction. Any substrate that adopts the pattern realizes the rate-axis savings. |

## Observability surface

Per CLAUDE.md "Max observability — non-negotiable" Catalog #305 6 facets:

* **Inspectable per layer**: canonical equation registered with explicit `domain_of_validity` enumerating `in_domain` substrate-class tokens (rate-axis substrate that admits conditional coding) + `excluded` tokens (substrates that fundamentally cannot use side-info, e.g., pure pose-only substrates where X = PoseNet output is itself the source).
* **Decomposable per signal**: `predicted_vs_empirical_residual` is a per-axis-token mapping; each substrate that lands an empirical anchor records its own per-axis residual.
* **Diff-able across runs**: every empirical anchor carries `inputs` + `predicted_output` + `empirical_output` + `residual` + `measurement_utc` so future anchors can be byte-level diffed against historical anchors via `query_equations_by_consumer`.
* **Queryable post-hoc**: `tools/list_canonical_equations.py --json` + `python -c "from tac.canonical_equations import get_equation_by_id; ..."` are the canonical query surfaces.
* **Cite-able**: every anchor carries `source_artifact` (memo path / commit sha) + `measurement_method` + canonical `Provenance` per Catalog #323. The first anchor cites `.omx/research/z8_m6_wyner_ziv_top_level_coder_full_implementation_landed_20260530.md` + commit `5d5634dd3`.
* **Counterfactual-able**: the canonical Wyner-Ziv 1976 R(D|Y) bound IS the counterfactual: if a substrate adopts the pattern, the predicted rate-axis savings are `R(D) - R(D|Y)`; if the substrate does NOT adopt, the savings are zero. The equation's prediction is structurally counterfactual.

## Cargo-cult audit per assumption

Per Catalog #303 sister discipline:

| Assumption | Classification | Rationale + unwind path |
|---|---|---|
| Wyner-Ziv 1976 Theorem 1 applies to any (X, Y) pair with non-trivial mutual information | HARD-EARNED | Canonical information-theory result; cited verbatim by Cover-Thomas 2006 § 15.9 + Slepian-Wolf 1973. Unwind path = N/A (canonical theorem). |
| PoseNet IS decoder-reproducible at contest runtime | HARD-EARNED | upstream/evaluate.py:53 loads PoseNet state dict; upstream/evaluate.py:79 computes pose distortion on inflated frames. The weights are NOT inside archive.zip per CLAUDE.md "Strict scorer rule" so no rate-axis cost. Unwind path = N/A (verified empirically). |
| 6-dim PoseNet output Y carries sufficient mutual information I(X;Y) on a typical substrate's X | ASSUMED_AWAITING_VERIFICATION | The empirical mutual information depends on the substrate's source X. Per-substrate empirical anchors will land as substrates adopt the pattern. Unwind path: per-substrate paired-CUDA RATIFICATION measures the actual conditional-entropy savings. |
| Z8 M6 synthetic Gaussian 64-74% savings extrapolates to contest substrates | CARGO-CULTED | Synthetic Gaussian was the M6 empirical anchor; real substrate top-state statistics differ. Unwind path: per-substrate empirical anchors land as Catalog #371 auto-recalibration triggers. The first EmpiricalAnchor in THIS equation explicitly tags `measurement_method="synthetic_gaussian_z8_m6_paradigm_anchor"` so the per-substrate axes have separate residual tracks. |
| Wyner-Ziv encoder-decoder agreement on the conditional model is trivial | HARD-EARNED | Z8 M6's deterministic projection matrix W (Hadamard-Gaussian from `(state_dim, side_info_shape, seed)` triple) is the canonical strategy. Sister `tac.codec.wyner_ziv_layer` uses canonical-Y-source enums for the same purpose. Unwind path = N/A. |

## Predicted ΔS band

Per Catalog #296 Dykstra-feasibility requirement:

**Predicted per-substrate rate-axis ΔS band**: `[-25 * (bytes_pre - bytes_post) / 37_545_489, 0]` per CLAUDE.md canonical rate term. The upper bound is 0 (no improvement); the lower bound depends on the substrate's source-X compressibility AND I(X; PoseNet(pair)) per Wyner-Ziv 1976 Theorem 1.

**Dykstra-feasibility check**: the rate-axis polytope constraint `bytes_post <= bytes_pre` is canonically convex (linear). The conditional-entropy reduction `H(X|Y) <= H(X)` (Shannon 1948 + canonical chain rule) is also linear in source vs conditional. The intersection is non-empty (any substrate with non-trivial I(X;Y) admits non-zero savings). FEASIBLE.

**First-principles citation**: Wyner & Ziv (1976) Theorem 1 + Shannon (1948) conditional entropy chain rule + Cover & Thomas (2006) § 15.9 Rate Distortion with Side Information at Decoder.

**Empirical anchor (synthetic Gaussian baseline)**: Z8 M6 measured 64-74% byte savings across noise scales [0.05, 5.00] (the savings ratio decreases as Y becomes less correlated with X per Wyner-Ziv 1976 § 3 Theorem 1; at noise scale 5.00 Y is nearly irrelevant and savings shrink to 64%). Round-trip distortion stays within the canonical R(D|Y) bound at all correlation levels (relative L2 error ~3.6%).

## Domain of validity

**IN-DOMAIN** (the equation's prediction is canonical):
* `decoder_side_posenet_reproducibility = True` (contest-runtime substrates that admit `upstream/evaluate.py` decoder execution)
* `source_signal_kind ∈ {rate-axis bytes, per-pair latent residual, archive-section bytes}` (source X is byte-stream amenable to conditional coding)
* `side_info_correlation = non_trivial` (I(X; PoseNet(pair)) > 0)

**EXCLUDED** (the equation's prediction does NOT apply):
* `posenet_as_source` (when X = PoseNet output itself, side-info is X — degenerate)
* `non_video_signals` (PoseNet is video-specific; non-video sources cannot use video-pose side info)
* `non_decoder_reproducible_substrates` (substrates whose decoder does NOT run the contest evaluator or sister PoseNet-reproducible path)
* `residual_hybrid_contexts_per_catalog_359` (sister anti-pattern per Catalog #359 — equation #26 procedural codebook misapplication taught the lesson that REPLACEMENT-savings equations don't apply to RESIDUAL-CORRECTION-stacking contexts; same here: this equation is the WYNER-ZIV CONDITIONAL-ENTROPY-REDUCTION paradigm, NOT a residual-correction-hybrid)

## Operational mechanism

Per Catalog #220 substrate L1+ scaffold operational mechanism + Catalog #272 distinguishing-feature integration contract:

The equation REGISTRATION is the operational mechanism at the canonical-equations-registry surface. The cathedral consumer (`src/tac/cathedral_consumers/wyner_ziv_posenet_side_information_consumer/`) is the auto-discovered runtime consumer per Catalog #335 — every cathedral autopilot iteration that matches a candidate against this equation surfaces the prediction as an observability-only Tier A annotation per Catalog #341.

## 6-hook wire-in declaration per Catalog #125

| Hook | Status | Surface |
|---|---|---|
| #1 sensitivity-map | **ACTIVE** | PoseNet output IS canonical sensitivity surface for pose-conditional bit allocation; the equation enables sister `tac.sensitivity_map.*` consumers to query the predicted Wyner-Ziv rate-axis savings per substrate. |
| #2 Pareto constraint | **ACTIVE** | Wyner-Ziv R(D\|Y) bound IS canonical Pareto constraint at the rate-axis polytope; sister `tac.dykstra_pareto_solver` can consume the equation's prediction as a polytope-feasibility constraint per Catalog #372. |
| #3 bit-allocator | **ACTIVE PRIMARY** | The canonical bit-allocator extension is `pose_conditional` mode: instead of allocating bits uniformly, allocate per the local conditional entropy `H(X_local | PoseNet_local)`. Sister `tac.bit_allocator.per_pair` can adopt the equation as a per-pair budget allocator. |
| #4 cathedral autopilot dispatch | **ACTIVE** | The new cathedral consumer `wyner_ziv_posenet_side_information_consumer` is auto-discovered per Catalog #335 + emits Tier A markers per Catalog #341 (observability-only). |
| #5 continual-learning posterior | **ACTIVE** | The canonical equation accumulates EmpiricalAnchors per Catalog #371 auto-recalibration; when >=3 in-domain anchors land, the residual posterior auto-refits. |
| #6 probe-disambiguator | **ACTIVE** | The equation IS the canonical disambiguator between "substrate uses Wyner-Ziv conditional coding" (predicted rate-axis savings via R(D|Y) < R(D)) vs "substrate uses unconditional coding" (no savings). |

## HORIZON-CLASS

`frontier_pursuit` — predicted CPU band [0.120, 0.180] per HORIZON-CLASS evaluation axis. The Wyner-Ziv conditional coding paradigm is canonical first-principles; per-substrate adoption is the frontier-pursuit path. PLATEAU-ADJACENT is too cautious (the technique is empirically validated at Z8 M6); ASYMPTOTIC-PURSUIT is too aggressive (sub-0.120 requires multiple compounding paradigm shifts).

## Implementation surfaces

* **NEW module**: `src/tac/canonical_equations/wyner_ziv_decoder_side_posenet_side_information.py` (~280 LOC; SPDX-MIT)
* **NEW package**: `src/tac/cathedral_consumers/wyner_ziv_posenet_side_information_consumer/__init__.py` (~150 LOC; canonical Tier A consumer per Catalog #335 + #341)
* **NEW tests**:
  - `src/tac/canonical_equations/tests/test_wyner_ziv_decoder_side_posenet_side_information.py` (canonical equation + builder + invariants)
  - `src/tac/cathedral_consumers/wyner_ziv_posenet_side_information_consumer/tests/test_consumer.py` (canonical Protocol contract + Tier A markers + hook #4 + hook #5)
* **MODIFIED**: `src/tac/canonical_equations/__init__.py` — re-export the new builder per canonical sister pattern.

## Sister Yousfi-cascade DISJOINT scope verification per Catalog #340

Verified at PV layer 1 via `git status` + `subagent_checkpoint.py read --latest-incomplete` (15:46Z):

* **Z8 M10 inflate-consumes-real-trained-weights** (`abfa0a600` in-flight, PID 16571): scope = `src/tac/substrates/z8_hierarchical_predictive_coding/inflate.py` + `canonical_quadruple_binding.py` + tests; DISJOINT from this lane's NEW canonical_equations + cathedral_consumers package files.
* **Slot GGG Tier C overnight runner** (PID 10169): scope = scorer-axis verification + null-projection; DISJOINT.
* **Cascade B wave-2** (`ac302ffd1`): scope = cascade_b package; DISJOINT.

## Cross-references

* CLAUDE.md "Canonical equations + models registry — non-negotiable" + Catalog #344
* CLAUDE.md "Meta-Lagrangian/Pareto solver — non-negotiable" (typed atom discipline)
* CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" (PoseNet IS decoder-reproducible at contest runtime)
* CLAUDE.md "Strict scorer rule" (PoseNet weights NOT inside archive.zip — free rate-axis side info)
* Catalog #335 (canonical cathedral consumer contract auto-discovery)
* Catalog #341 (Tier A canonical-routing markers)
* Catalog #323 (canonical Provenance umbrella)
* Catalog #371 (auto-recalibrator on >=3 new empirical anchors)
* Z8 M6 landing memo: `.omx/research/z8_m6_wyner_ziv_top_level_coder_full_implementation_landed_20260530.md` (commit 5d5634dd3; sister Wyner-Ziv top-level coder)
* Wyner & Ziv (1976) "The rate-distortion function for source coding with side information at the decoder" IEEE Trans. Inf. Theory IT-22(1):1-10
* Cover & Thomas (2006) Elements of Information Theory, 2nd ed., § 15.9 Rate Distortion with Side Information at Decoder
* Pradhan & Ramchandran (2003) "Distributed source coding using syndromes (DISCUS)" IEEE Trans. Inf. Theory 49(3):626-643

## Operator-routable next actions

1. **Land canonical equation + cathedral consumer in same commit batch** (this landing). Drives canonical equation registry size from 149 → 150.
2. **Extend `tac.bit_allocator.per_pair` with `pose_conditional` mode** that consumes the equation's prediction to allocate bits per `H(X_local | PoseNet_local)`. (DEFERRED — separate sister landing.)
3. **Per-substrate paired-CUDA RATIFICATION** on any substrate that adopts Wyner-Ziv PoseNet-conditional coding will produce a second EmpiricalAnchor on the per-substrate axis. The canonical equation accumulates anchors per Catalog #371 auto-recalibration when >=3 in-domain anchors land. (DEFERRED — operator-routable per substrate.)
4. **Sister cathedral consumer registration** for related canonical equations (e.g., a unified consumer that surfaces ALL Wyner-Ziv equations' predictions per candidate). (DEFERRED — premature consolidation; sister consumers can land independently per Catalog #335 auto-discovery.)
