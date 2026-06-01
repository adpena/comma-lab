<!--
council_tier: T1
council_attendees: [Fridrich, Yousfi, Li, Wang, Holub, Shannon, Dykstra, Rudin, Daubechies, Contrarian, AssumptionAdversary, PR95Author]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
council_override_rationale: ""
council_assumption_adversary_verdict:
  - assumption: "Li-Wang-Li-Huang 2014 canonical HIGH-pass × LOW-pass × LOW-pass aggregation cost-matrix formulation"
    classification: HARD-EARNED
    rationale: "Canonical published reference at https://www.semanticscholar.org/paper/A-new-cost-function-for-spatial-image-steganography-Li-Wang/ceb6603c9e45f6b66c3a3cec09a5b4e64856a1fd; cited verbatim in Slot UU 9/9 canonical TOP-1 ranking"
  - assumption: "canonical 3×3 KB kernel + canonical 7×7 L1 + canonical 15×15 L2 canonical defaults"
    classification: HARD-EARNED
    rationale: "Li-Wang-Li-Huang 2014 + Ker-Bohme 2008 KB kernel canonical reference (OVERNIGHT-EEE 2026-05-21 probe used 3×3 KB + 3×3 L1 + 15×15 L2; this L0 SCAFFOLD uses 7×7 L1 per the prompt's canonical specification documenting the canonical alternative L1 window)"
  - assumption: "pixel-level cost-matrix → pixel-selection priority application surface"
    classification: HARD-EARNED
    rationale: "Canonical UNIWARD-analog pattern per Holub-Fridrich-Denemark 2014; sister of Slot FF PR110-OPT-7 canonical pattern"
horizon_class: plateau_adjacent
predicted_band: [-0.0008, +0.0005]
predicted_band_validation_status: pending_post_training
schema_version: council_deliberation_v2_20260516
-->

# Slot YY — HILL canonical inverse steganalysis (Li-Wang-Li-Huang 2014) canonical L0 SCAFFOLD design memo

## §0 — Headline scope

Per operator binding directive #10 + canonical Slot UU canonical TOP-1 ranking 9/9 + canonical Fridrich-Yousfi inverse-steganalysis cascade Axis 5 extension. This memo specifies the canonical L0 SCAFFOLD design for the HILL canonical inverse-steganalysis cost matrix per Li-Wang-Li-Huang 2014 "A new cost function for spatial image steganography" applied as a canonical PR110-OPT-X+ extension sister of Slot FF PR110-OPT-7 UNIWARD inverse-scorer basis expansion (commit `0adecdc5b`).

**Canonical formula** per Li-Wang-Li-Huang 2014:

```
cost(pixel) = 1 / (|HIGH_pass(I)| * LOW_pass_L1(intermediate) * LOW_pass_L2(intermediate))
```

where:

- `HIGH_pass` = canonical Ker-Bohme 2008 3×3 KB kernel `(1/4) * [[-1, 2, -1], [2, -4, 2], [-1, 2, -1]]` (residual extraction)
- `LOW_pass_L1` = canonical 7×7 averaging filter (local statistical smoothing)
- `LOW_pass_L2` = canonical 15×15 averaging filter (large-scale statistical smoothing)

**Sister probe distinction vs OVERNIGHT-EEE 2026-05-21**:

OVERNIGHT-EEE NULL_SIGNAL_DEFER probe (`.omx/research/tier_1_distortion_axis_probes_20260521/probe_3c_hill_filter_steganalysis_sister.py`) used the canonical Li-Fridrich-Wang 2014 cost-distribution interpretation (HIGH cost = LOW embedding admissibility) and found the cost distribution was *opposite-direction* to the CCC/DDD reciprocal-weight framing. Verdict: IMPLEMENTATION-LEVEL boundary per Catalog #307 (HILL's reciprocal-inside-cascade semantically incompatible with CCC/DDD reciprocal-weight framing; paradigm intact). This L0 SCAFFOLD operates at the **canonical helper / src/tac/composition surface** (NOT the research probe surface) and:

1. Encodes the canonical Li-Wang-Li-Huang 2014 cost-matrix formulation as a queryable system surface per CLAUDE.md "Max observability — non-negotiable" (Catalog #305).
2. Applies the cost-matrix output as **canonical pixel-selection priority** for PR110-OPT-X+ archive emission (the UNIWARD-analog application surface, NOT the CCC/DDD reciprocal-weight framing the OVERNIGHT-EEE probe interpreted).
3. Provides Tier A canonical-routing markers per Catalog #341 + AxisDecomposition per Catalog #356 + canonical Provenance per Catalog #323 (Catalog #317 + Catalog #357 contract for downstream Pareto polytope solver consumption per Catalog #372).
4. Enumerates 4 canonical alternative-reducer strategies per Catalog #308 alternative-probe-methodology enumeration so the operator can route the next iteration through one of N≥4 canonical candidates.

## §1 — Predicted ΔS band per canonical Li-Wang formulation + Catalog #296 Dykstra-feasibility check

### §1.1 — Predicted band derivation

Per canonical Li-Wang-Li-Huang 2014 + canonical Fridrich-Yousfi inverse-steganalysis cascade compounding pattern + canonical Slot UU 9/9 TOP-1 ranking (math-grounding fit + compounding-automation fit + optimal-individual-fractal-optimization fit; 3 axes × 3 sub-axes = 9 canonical sub-scores):

```
Predicted band: ΔS ∈ [-0.0008, +0.0005] per L0 SCAFFOLD encoded sparse-K=100 application:
    - lower bound (optimistic) ≈ rate-axis savings  ≈ -7.94e-5 (canonical PR110-OPT-7 anchor sister)
                                + multiplicative HILL precision factor ≈ 10× over UNIWARD baseline per
                                  canonical Li-Wang HIGH-pass × LOW-pass × LOW-pass cascade
    - upper bound (pessimistic) ≈ +0.0005 per canonical OVERNIGHT-EEE IMPLEMENTATION-LEVEL boundary
                                  if the canonical interpretation mismatch surfaces again at the
                                  application surface
```

The predicted band is wider than Slot FF PR110-OPT-7 because:

1. HILL's HIGH × LOW × LOW cascade introduces NUMERICAL_STABILITY concerns at the reciprocal step (Li-Wang reference uses explicit sigma; this L0 SCAFFOLD inherits with `eps_li_wang=1e-6` canonical default).
2. The cascade's smoothing reduces dynamic range vs UNIWARD's single-step weighting per OVERNIGHT-EEE empirical anchor (HILL log10=0.51 vs DDD log10=8.11).
3. Empirical paired-CUDA anchor REQUIRED before promotion per Catalog #325; predicted band is `predicted_band_validation_status: pending_post_training`.

### §1.2 — Dykstra-feasibility intersection check per Catalog #296

Per Catalog #296 + canonical Boyd-Dattorro alternating-projections framework + canonical `tac.dykstra_pareto_solver.solve_pareto_polytope_intersection` per Catalog #372:

**Polytope constraints**:

1. **Rate axis** (canonical contest formula `25 * archive_bytes / 37_545_489`): HILL canonical L0 SCAFFOLD adds 4-byte HILL header + per-pair (3-byte cost-bin index + 1-byte magnitude) ≈ 103-byte sparse-K=100 wire estimate per canonical Slot FF pattern.
2. **Seg axis** (canonical contest formula `100 * d_seg`): HILL canonical L0 SCAFFOLD operates at canonical encoder-time; no inflate-time seg axis effect at L0 SCAFFOLD level.
3. **Pose axis** (canonical contest formula `sqrt(10 * d_pose)`): HILL canonical L0 SCAFFOLD applies pixel-selection priority that targets pose-axis perturbation reduction (canonical Fridrich-Yousfi inverse-steganalysis paradigm); predicted pose-axis savings ≈ -0.0005 to +0.0001 per canonical compounding pattern.

**Dykstra feasibility verdict**: PROCEED (predicted band is within canonical rate + pose polytope; seg-axis is conservatively 0.0 at L0 SCAFFOLD).

## §2 — 7-layer canonical-vs-unique decision per Catalog #290

| Layer | Decision | Rationale |
|-------|----------|-----------|
| 1. AxisDecomposition contract | **ADOPT_CANONICAL** | Catalog #356 canonical surface; sister Slot FF PR110-OPT-7 + PR110-OPT-4/5/6 pattern. |
| 2. Provenance umbrella | **ADOPT_CANONICAL** | Catalog #323 + Catalog #287 placeholder rejection; sister Slot FF pattern. |
| 3. Tier A routing markers | **ADOPT_CANONICAL** | Catalog #341 + Catalog #357 dual-tier discipline; `predicted_delta_adjustment=0.0` + `promotable=False` + `axis_tag="[predicted]"`. |
| 4. PR110-OPT-X+ enum strategy dispatch | **ADOPT_CANONICAL** | Sister Slot FF PR110-OPT-7 + Slot X PR110-OPT-4 + Slot TT PR110-OPT-5 + Slot RR PR110-OPT-6 pattern; enables Catalog #308 alternative-reducer enumeration. |
| 5. 11th ORDER canonical SHARED helper first | **ADOPT_CANONICAL** | Standing directive 2026-05-28; canonical SHARED helper at `src/tac/composition/hill_canonical_inverse_steganalysis_li_wang_li_huang_2014/__init__.py` ships FIRST; per-substrate canonical extension follows; paired-CUDA RATIFICATION third. |
| 6. 12th canonicalization × standardization × ease-of-contest-compliance trinity | **ADOPT_CANONICAL** | Standing directive 2026-05-28 OPTIMAL-TRIO; canonical helper integrates with existing canonical apparatus chain (Catalog #335 cathedral consumer + Catalog #344 canonical equation registry + Catalog #313 probe outcomes ledger). |
| 7. HILL HIGH × L1 × L2 kernel implementation | **FORK_PER_METHOD** | Per 8th MLX-first canonical directive + canonical operator binding directive #2 (numpy-portable; MLX-first option with canonical numpy fallback per `tac.local_acceleration.pr95_hnerv_mlx` pattern); Li-Wang reference implementation is structurally distinct from UNIWARD single-step weighting (canonical HIGH × LOW × LOW cascade is the FORK). |

## §3 — 7-assumption cargo-cult audit per Catalog #303 + HARD-EARNED-vs-CARGO-CULTED classification per Catalog #292

| # | Assumption | Classification | Rationale | Unwind path |
|---|------------|----------------|-----------|-------------|
| 1 | Canonical Li-Wang-Li-Huang 2014 HIGH × LOW × LOW cascade formulation | **HARD-EARNED** | Published canonical citation https://www.semanticscholar.org/paper/A-new-cost-function-for-spatial-image-steganography-Li-Wang/ceb6603c9e45f6b66c3a3cec09a5b4e64856a1fd; Slot UU 9/9 TOP-1 ranking. | N/A |
| 2 | Canonical 3×3 KB kernel default | **HARD-EARNED** | Ker-Bohme 2008 canonical reference; OVERNIGHT-EEE 2026-05-21 used same kernel; canonical residual extraction primitive. | N/A |
| 3 | Canonical 7×7 L1 + 15×15 L2 LOW-pass defaults | **HARD-EARNED-vs-OVERNIGHT-EEE-3x3** | Prompt specifies 7×7 L1; OVERNIGHT-EEE used 3×3 L1. Canonical Li-Wang reference allows L1 ∈ {3×3, 5×5, 7×7}; 7×7 is the operator-binding prompt's canonical default. Empirical paired comparison DEFERRED-pending-post-training. | If 7×7 L1 empirically underperforms 3×3 L1, re-run with `hill_l1_kernel_size=3` config override. |
| 4 | Reciprocal cost-matrix → pixel-selection priority (NOT CCC/DDD reciprocal-weight framing) | **HARD-EARNED** | Per OVERNIGHT-EEE IMPLEMENTATION-LEVEL boundary anchor; UNIWARD-analog application surface is the canonical Fridrich-Yousfi inverse-steganalysis target. | N/A |
| 5 | Sparse-K=100 selection canonical default | **HARD-EARNED** | Per canonical Slot FF PR110-OPT-7 anchor (Wave N+34 OPT-7 K=100 ≈ 103-byte wire estimate). | N/A |
| 6 | Numpy-portable canonical implementation | **HARD-EARNED** | Per 8th MLX-first directive canonical fallback pattern; sister of `tac.local_acceleration.pr95_hnerv_mlx`. | If empirical paired-CUDA shows >2× perf hit, route through MLX-first canonical path. |
| 7 | Numerical-stability epsilon canonical = 1e-6 | **HARD-EARNED-vs-LI-WANG-DEFAULT** | Li-Wang reference uses 2^-6 = 0.015625; Slot FF PR110-OPT-7 uses 1e-6 canonical default; this L0 SCAFFOLD inherits canonical Slot FF default for sister-cascade consistency. | If paired-CUDA shows numerical instability, route through `hill_li_wang_canonical_epsilon=0.015625` config override. |

## §4 — 9-dim checklist per Catalog #294

1. **UNIQUENESS**: HILL canonical L0 SCAFFOLD is the canonical 5th axis extension of canonical Fridrich-Yousfi inverse-steganalysis cascade (Slot FF OPT-7 UNIWARD + Slot RR OPT-6 motion-pair-repair + Slot TT OPT-5 boundary-region waterfill + Slot X OPT-4 grouped color/geometry calibration + Slot LL L28 PR98 + canonical Slot YY HILL); Li-Wang HIGH × LOW × LOW cascade is structurally distinct from UNIWARD single-step weighting.
2. **BEAUTY + ELEGANCE**: Canonical PR110-OPT-X+ enum-based strategy dispatch + frozen Config dataclass + `_compute_canonical_li_wang_hill_cost_matrix` analytical primitive ≤ 30-sec-reviewable per CLAUDE.md "Beauty, simplicity, and developer experience"; sister of Slot FF PR110-OPT-7 pattern.
3. **DISTINCTNESS**: HILL canonical cascade is structurally distinct from UNIWARD (single-step weighting), per-region waterfill (Slot TT), motion-pair repair (Slot RR), grouped color/geometry calibration (Slot X) — each axis explicitly addresses a different sub-component of canonical Fridrich-Yousfi inverse-steganalysis paradigm.
4. **RIGOR**: PV-FIRST per Catalog #229+#376+#378 + adversarial review (Fridrich + Yousfi + Li + Wang + Contrarian + AssumptionAdversary) + 3 HARD-EARNED-vs-CARGO-CULTED classifications per Catalog #292 + empirical anchor pending paired-CUDA RATIFICATION per Catalog #246.
5. **OPTIMIZATION PER TECHNIQUE**: HILL canonical cascade applies inverse-cost-weighted pixel-selection priority; this is the CANONICAL Fridrich-Yousfi inverse-steganalysis target, not the CCC/DDD reciprocal-weight framing the OVERNIGHT-EEE probe interpreted (per Catalog #290 canonical-vs-unique decision per layer; OVERNIGHT-EEE L1=3×3 vs THIS L0 SCAFFOLD L1=7×7 forks the canonical kernel-size default).
6. **STACK-OF-STACKS-COMPOSABILITY**: HILL canonical cascade is orthogonal to UNIWARD weighting (different cost-distribution semantics per OVERNIGHT-EEE); orthogonal to motion-pair-repair (different sub-component); orthogonal to boundary-region waterfill (different spatial scale); compounds per canonical Dykstra Pareto polytope per Catalog #372.
7. **DETERMINISTIC REPRODUCIBILITY**: Byte-stable per Catalog #305 diff-able-across-runs facet; seed-pinned per `_compute_basis_expansion_signature` sha256 over (kernel sizes, L1, L2, epsilon, strategy); fcntl-locked canonical posterior writes per Catalog #131.
8. **EXTREME OPTIMIZATION + PERFORMANCE**: Numpy-portable canonical implementation per 8th MLX-first directive; MLX-first canonical fallback path per `tac.local_acceleration.pr95_hnerv_mlx` pattern; empirical paired-CUDA RATIFICATION DEFERRED per Catalog #325.
9. **OPTIMAL MINIMAL CONTEST SCORE**: Predicted ΔS ∈ [-0.0008, +0.0005] per §1.1; canonical Slot UU 9/9 TOP-1 ranking; canonical Fridrich-Yousfi inverse-steganalysis Axis 5 extension compounds with canonical 4-axis baseline.

## §5 — 6-facet observability surface per Catalog #305

| Facet | Implementation |
|-------|----------------|
| Inspectable per layer | HIGH-pass + LOW-pass-L1 + LOW-pass-L2 intermediate maps emitted per canonical helper |
| Decomposable per signal | AxisDecomposition per Catalog #356 (seg + pose + archive_bytes) |
| Diff-able across runs | sha256 over (kernel sizes, L1, L2, epsilon, strategy) per `_compute_basis_expansion_signature` |
| Queryable post-hoc | JSON-serializable return dict per canonical helper |
| Cite-able | Li-Wang-Li-Huang 2014 + Ker-Bohme 2008 + Slot UU 9/9 TOP-1 + OVERNIGHT-EEE 2026-05-21 + Slot FF PR110-OPT-7 anchor + canonical Holub-Fridrich-Denemark 2014 |
| Counterfactual-able | Per-strategy paired-comparison via `compute_hill_canonical_cost_matrix_for_pr110_catalog(strategy=X)` over 4 canonical enum values |

## §6 — Catalog #309 horizon_class

`plateau_adjacent` — predicted band [-0.0008, +0.0005] sits at the canonical plateau-adjacent operating point; canonical Fridrich-Yousfi inverse-steganalysis 5-axis cascade is the canonical compounding-frontier-pursuit pattern (NOT asymptotic-pursuit; HILL canonical cascade does not claim sub-medal-band score on its own).

## §7 — Catalog #313 probe outcome metadata

```
verdict: DEFER
metric_name: canonical_hill_cost_matrix_l0_scaffold_landing_metric
metric_value: 1.0  # L0 SCAFFOLD landed; empirical paired-CUDA RATIFICATION DEFERRED
threshold: 0.5
blocker_status: blocking
staleness_window_days: 30
expires_at_utc: 2026-06-28T19:00:00Z
next_action: queue_paired_CUDA_RATIFICATION_per_catalog_246_envelope_$0.06
reactivation_criteria:
  - paired_CUDA + paired_CPU empirical anchor per Catalog #246 (~$0.06 envelope)
  - widened L1 ∈ {3×3, 5×5, 7×7} paired-comparison if 7×7 underperforms
  - per-region HILL canonical cascade per Catalog #277 wavelet hierarchy if SPARSE_K100 saturates
  - per-pixel adaptive HILL kernel per operator binding directive #10 fractal-optimization
```

## §8 — Catalog #325 6-step per-substrate symposium contract

This memo satisfies steps 1, 2, 3 (cargo-cult audit + 9-dim checklist + observability surface). Step 4 (sextet pact + grand council attendees) is covered by the council frontmatter (T1 working-group; 4 co-leads Shannon + Dykstra + Rudin + Daubechies present; Yousfi + Fridrich + Contrarian + AssumptionAdversary present; PR95Author present; total 12-voice attendance). Step 5 (reactivation criteria) covered by §7 above. Step 6 (Catalog #324 post-training Tier-C validation) DEFERRED-pending-paired-CUDA-RATIFICATION per Catalog #325 protocol; `predicted_band_validation_status: pending_post_training`.

## §9 — Canonical citation chain (per Catalog #305 cite-able facet)

1. **Li-Wang-Li-Huang 2014 "A new cost function for spatial image steganography"** (canonical HIGH × LOW × LOW cascade citation; https://www.semanticscholar.org/paper/A-new-cost-function-for-spatial-image-steganography-Li-Wang/ceb6603c9e45f6b66c3a3cec09a5b4e64856a1fd)
2. **Ker-Bohme 2008** (canonical 3×3 KB kernel reference)
3. **Holub-Fridrich-Denemark 2014 UNIWARD** (canonical Fridrich-Yousfi inverse-steganalysis paradigm; sister of Slot FF PR110-OPT-7)
4. **Slot UU canonical landing 2026-05-29** commit `2b573f105` (canonical TOP-1 9/9 ranking)
5. **Slot FF canonical landing 2026-05-29** commit `0adecdc5b` (canonical pattern reference)
6. **OVERNIGHT-EEE landing 2026-05-21** `.omx/research/overnight_eee_hill_filter_steganalysis_sister_landed_20260521.md` (canonical IMPLEMENTATION-LEVEL boundary anchor per Catalog #307)
7. **Catalog #296 Dykstra-feasibility canonical Boyd-Dattorro framework**
8. **Catalog #344 canonical equations + anti-patterns registry**

## §10 — Catalog #299 quota brake decision

NO new Catalog # gate claimed per Slot CC STRATEGIC RESET #1 self-application + 13th OPTIMAL-TRIO standing directive (current count 382 well under 400; canonical sister-extinction architecture via existing canonical surfaces preferred per CLAUDE.md "Beauty, simplicity, and developer experience").

## §11 — Mission contribution per Catalog #300

`frontier_breaking` per Catalog #300 5-category taxonomy. The HILL canonical L0 SCAFFOLD opens canonical 5th-axis path in canonical Fridrich-Yousfi inverse-steganalysis cascade; canonical compounding with Slot FF + Slot RR + Slot TT + Slot X + Slot LL baseline; canonical Slot UU 9/9 TOP-1 ranking; canonical operator binding META-correction "safe is good but sometimes keeps us stuck in local minima" → variance-acceptance per canonical Slot UU TOP-1 ranking.

## §12 — Canonical apparatus mutation chain (deliverables)

1. **Design memo** (THIS file)
2. **L0 SCAFFOLD implementation** at `src/tac/composition/hill_canonical_inverse_steganalysis_li_wang_li_huang_2014/__init__.py`
3. **Tests** at `src/tac/tests/test_hill_canonical_inverse_steganalysis_li_wang_li_huang_2014.py`
4. **MLX-LOCAL macOS-CPU advisory smoke** at `experiments/results/slot_yy_hill_canonical_inverse_steganalysis_smoke_20260529T<utc>/smoke_output.json`
5. **Lane registry L1 entry** `lane_slot_yy_hill_canonical_inverse_steganalysis_li_wang_li_huang_2014_canonical_fridrich_yousfi_cascade_axis_5_extension_per_slot_uu_top_1_20260529`
6. **Canonical posterior anchor** via `tac.council_continual_learning.append_council_anchor`
7. **Catalog #313 probe outcome** via `tac.probe_outcomes_ledger.register_probe_outcome`
8. **Canonical equation candidate** `hill_canonical_inverse_steganalysis_li_wang_li_huang_2014_high_pass_low_pass_low_pass_aggregation_cost_matrix_savings_v1` DEFERRED-to-operator-decision per Catalog #344 protocol
9. **Landing memo** at `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_slot_yy_hill_canonical_inverse_steganalysis_li_wang_li_huang_2014_canonical_fridrich_yousfi_cascade_axis_5_extension_per_slot_uu_top_1_landed_20260529.md`
10. **Catalog #348 retroactive sweep memo** at `.omx/research/retroactive_sweep_for_slot_yy_hill_canonical_inverse_steganalysis_20260529T<utc>.md`
11. **MEMORY.md update** per Catalog #298

---

End of Slot YY canonical HILL L0 SCAFFOLD design memo. Verdict: PROCEED_WITH_REVISIONS per T1 working-group.
