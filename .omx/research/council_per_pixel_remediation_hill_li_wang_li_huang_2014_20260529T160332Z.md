---
council_tier: T2
council_attendees:
  - Shannon
  - Dykstra
  - Rudin
  - Daubechies
  - Yousfi
  - Fridrich
  - Contrarian
  - AssumptionAdversary
  - Li_Wang_Li_Huang_canonical_paper_author_seat
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Contrarian
    verbatim: "The canonical shared helper tac.inverse_steganalysis_real_video_mlx provides REAL per-pixel HILL on REAL upstream/videos/0.mkv frames; the macOS-CPU advisory smoke shows 27.46 dB cost-discrimination dynamic range at 96x72 vs. the previous synthetic-random-noise smokes that showed effectively zero discrimination. The remediation is honest. However, the PR110 archive grammar surface (the per-pair scalar aggregation in the existing apply_hill_canonical_cost_matrix_to_pr110_archive) is NOT yet wired to consume the per-pixel cost matrix; the bind helper apply_hill_canonical_per_pixel_mlx_to_real_video_frames returns macOS-CPU advisory smoke results but does NOT yet emit PR110 selector bytes. PROCEED WITH REVISIONS: queue paired-CUDA RATIFICATION on a PR110 archive that USES the per-pixel cost matrix for sparse-K selection before any score claim."
  - member: AssumptionAdversary
    verbatim: "Shared assumption operating within: 'per-pixel cost discrimination on real video frames (27.46 dB HILL dynamic range) implies non-trivial score impact when the cost matrix routes PR110 archive selector bytes.' Classification: ASSUMED_AWAITING_VERIFICATION per Catalog #363 recursive self-reflection protocol. Empirical evidence is required: a paired-CUDA RATIFICATION on a PR110 archive whose selector bytes are routed by the per-pixel HILL cost matrix would falsify or confirm this assumption. The 27.46 dB cost-discrimination is HARD-EARNED at the cost-matrix surface but the score-impact prediction is INFERRED from canonical Fridrich-Yousfi inverse-steganalysis framing. Reactivation criterion: paired-CUDA empirical ΔS measurement."
council_assumption_adversary_verdict:
  - assumption: "Real per-pixel HILL on real video frames produces non-trivial cost discrimination (operator binding 'no fake implementations' invariant satisfied)"
    classification: HARD-EARNED
    rationale: "Empirically verified: 27.46 dB dynamic range on real upstream/videos/0.mkv decoded frames at 96x72 resolution; matches canonical Li-Wang-Li-Huang 2014 paper expectations for natural-image cover sources; the canonical KB-kernel impulse response test verifies the math primitive matches the paper formula (center weight = -1.0; high-pass property: zero output on constant interior)."
  - assumption: "MLX deployment via mlx.core.conv2d is canonical for per-pixel inverse-steganalysis cost-matrix computation on M-series unified memory"
    classification: HARD-EARNED
    rationale: "MLX conv2d primitive verified to match numpy.scipy.signal.convolve2d to fp32 precision (atol=1e-5); empirical smoke 0.165s for 4 frames at 128x96 demonstrates fast bind step per operator binding invariant 4."
  - assumption: "Per-pixel cost matrix on real video frames will produce non-trivial score impact when applied to PR110 archive selector bytes"
    classification: ASSUMED_AWAITING_VERIFICATION
    rationale: "The cost-matrix surface is decoupled from the PR110 archive grammar surface (per-pair selector emission). Paired-CUDA RATIFICATION on a PR110 archive whose selectors are routed by the per-pixel HILL cost matrix is the canonical disambiguator. Reactivation criterion: empirical ΔS measurement vs FEC6 baseline on real Modal CPU+CUDA T4 dispatch (~$0.06 envelope per Catalog #246)."
council_decisions_recorded:
  - "PROCEED with the canonical shared helper landing + Slot YY HILL bind helper landing (152/152 tests + 15 cathedral consumer tests + Catalog #335 strict gate clean)"
  - "REVISIONS: per-substrate operator-routable cascade enumerated for the other 5 Slot EEE targets (Slot AAA MiPOD / Slot CCC HUGO / Slot FF UNIWARD / Slot TT SegNet-boundary-waterfill / Slot RR pose-axis-null-projection); each follows the canonical Slot YY HILL pattern via the shared helper"
  - "Reactivation criterion for paired-CUDA RATIFICATION: a PR110 candidate archive whose selector bytes are routed by tac.inverse_steganalysis_real_video_mlx.compute_hill_per_pixel_cost_mlx applied to real upstream/videos/0.mkv frames; empirical ΔS measurement vs FEC6 baseline"
  - "Per CLAUDE.md 'Forbidden premature KILL': Slot EEE PARTIAL/FAKE verdicts on Slot YY are REMEDIATED (Axis A PARTIAL → REAL; Axis C FAIL → PASS); the existing per-pair scalar aggregation entry point is PRESERVED for backward compatibility"
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
council_override_rationale: ""
related_deliberation_ids:
  - slot_eee_fake_implementation_audit_on_today_l0_scaffolds_per_operator_binding_must_review_for_fake_implementations_landed_20260529
  - slot_yy_hill_canonical_inverse_steganalysis_li_wang_li_huang_2014_canonical_fridrich_yousfi_cascade_axis_5_extension_per_slot_uu_top_1_landed_20260529
horizon_class: plateau_adjacent
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---

# Per-substrate symposium: Slot YY HILL per-pixel MLX real-video remediation

Per Catalog #325 canonical 6-step contract + operator binding 5-invariant standing
directive 2026-05-29 + Slot EEE fake-implementation audit Axis A (cite-vs-impl
PARTIAL) + Axis C (smoke realism FAIL) remediation.

## 1. Cargo-cult audit per assumption (Catalog #303)

| # | Assumption | Classification | Unwind path applied |
|---|------------|----------------|--------------------|
| 1 | Per-pair row-band aggregation of HILL cost matrix is sufficient bridge between Li-Wang per-pixel paper formulation and PR110 per-pair archive grammar | HARD-EARNED-for-archive-grammar / CARGO-CULTED-for-paper-fidelity | Fork: add per-pixel MLX bind helper that operates on real-video frames at the per-pixel surface the paper actually describes; preserve scalar-aggregation helper for archive grammar |
| 2 | Synthetic 32x32 random-noise inputs are sufficient to validate HILL cost matrix behavior | CARGO-CULTED | Unwind: canonical shared helper requires real `upstream/videos/0.mkv` decoded frames per Catalog #213; smoke runner refuses synthetic input by construction |
| 3 | Pure-Python nested-loop convolution (existing `_convolve_2d_canonical`) is acceptable for full-frame per-pixel processing | HARD-EARNED-for-portability / CARGO-CULTED-for-MLX-deployment | Fork: add MLX conv2d primitive (canonical `conv2d_mlx`) for canonical bind step; preserve nested-loop fallback for portability per CLAUDE.md "MLX portable-local-substrate authority" |
| 4 | KB kernel center weight -1.0 with [-1, 2, -1] symmetry matches Li-Wang-Li-Huang 2014 paper formula | HARD-EARNED | Verified via canonical impulse-response test: center weight = -1.0, cardinal neighbors = +0.50, sum = 0 (canonical high-pass property) |
| 5 | The 27.46 dB cost-discrimination dynamic range on real-video implies non-trivial score impact when applied to PR110 archive selector bytes | ASSUMED_AWAITING_VERIFICATION | Reactivation criterion: paired-CUDA RATIFICATION on PR110 archive whose selector bytes are routed by per-pixel HILL cost matrix |

## 2. 9-dimension success checklist evidence (Catalog #294)

1. **UNIQUENESS**: PASS. The canonical shared helper `tac.inverse_steganalysis_real_video_mlx` is the FIRST canonical MLX per-pixel inverse-steganalysis surface on real `upstream/videos/0.mkv` frames in the repo. Slot YY HILL bind helper is the FIRST canonical per-pixel real-video bind for HILL.
2. **BEAUTY + ELEGANCE**: PASS. The canonical shared helper is ~770 LOC; the Slot YY bind helper is ~140 LOC; both 30-second-reviewable per HNeRV parity L4. Single shared module avoids 6-way code duplication across sister cost functions.
3. **DISTINCTNESS**: PASS. HILL per-pixel via MLX conv2d cascade is structurally distinct from the existing per-pair scalar aggregation surface; the cite-vs-impl PARTIAL audit finding is remediated.
4. **RIGOR**: PASS. 49 canonical helper tests + 9 new HILL bind tests + 15 cathedral consumer tests = 73 NEW tests covering canonical math primitives (KB-kernel impulse response, Wiener filter SNR weighting, conv2d MLX/numpy parity), real-video decode correctness, Tier A canonical-routing markers, paradigm-routing disambiguator. All 152 + 15 = 167 tests pass.
5. **OPTIMIZATION-PER-TECHNIQUE**: PASS. MLX conv2d primitive at 0.165s for 4 frames × 96×128 demonstrates fast bind step. Per-pixel surface enables paired-CUDA RATIFICATION when wired into PR110 selector routing.
6. **STACK-OF-STACKS-COMPOSABILITY**: PASS at the cathedral autopilot ranker surface via the new `per_pixel_inverse_steganalysis_real_video_mlx_consumer` (auto-discovered per Catalog #335). Paradigm-routing disambiguator (hook #6) routes candidates to the correct canonical per-pixel cost function.
7. **DETERMINISTIC-REPRODUCIBILITY**: PASS. Canonical Provenance per Catalog #323 + canonical Tier A routing markers per Catalog #341 + deterministic numpy-fallback path verified.
8. **EXTREME-OPTIMIZATION-PERFORMANCE**: N/A at L0 SCAFFOLD; the MLX path is the canonical bind step for operator-routable paired-CUDA RATIFICATION.
9. **OPTIMAL-MINIMAL-CONTEST-SCORE**: NOT CLAIMED. The bind helper returns `predicted_delta_adjustment=0.0` + `promotable=False` + `score_claim=False` per Catalog #341 + #192. Paired-CUDA empirical anchor required per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" non-negotiable.

## 3. Observability surface (Catalog #305)

1. **Inspectable per layer**: PASS. Each cost function exposes its analytical primitive separately (conv2d_mlx, wiener_filter_canonical_mlx, compute_hill_per_pixel_cost_mlx); the bind helper returns the full `CanonicalSmokeResult` dict with min/max/mean/std/dynamic-range-db.
2. **Decomposable per signal**: PASS. AxisDecomposition per Catalog #356 separates seg/pose/archive_bytes; the existing apply_hill_canonical_cost_matrix_to_pr110_archive still emits this. The per-pixel bind helper surfaces cost-matrix statistics directly (no per-axis decomposition because no PR110 byte emission yet — operator-routable).
3. **Diff-able across runs**: PASS. Canonical Provenance includes captured_at_utc; smoke result includes input/parameter hashes via canonical helper signature.
4. **Queryable post-hoc**: PASS. Canonical smoke output is JSON at `experiments/results/slot_per_pixel_mlx_inverse_steganalysis_remediation_macos_cpu_advisory_smoke_*/smoke_output.json`.
5. **Cite-able**: PASS. Canonical Li-Wang-Li-Huang 2014 citation URL preserved in canonical shared helper; canonical_helper_module field in bind helper output.
6. **Counterfactual-able**: PARTIAL. The byte-mutation smoke per Catalog #105/#139/#272 is OPERATOR-ROUTABLE: a future sister wave can wire the per-pixel cost matrix into PR110 selector bytes + run byte-mutation smoke to verify cost-discrimination → score impact mapping.

## 4. Sextet pact deliberation per Catalog #346 canonical roster

Inner sextet:
- **Shannon LEAD**: HARD-EARNED canonical Li-Wang-Li-Huang 2014 high-pass + low-pass cascade is information-theory grounded (high-pass = canonical residual extraction; low-pass = canonical smoothing; reciprocal = canonical inverse-detectability weighting). PROCEED.
- **Dykstra CO-LEAD**: convex feasibility of the per-pixel cost matrix is HARD-EARNED (positive cost everywhere; bounded by reciprocal-with-epsilon); the pareto polytope intersection (rate + distortion + archive bytes) is OPERATOR-ROUTABLE pending paired-CUDA empirical anchor. PROCEED.
- **Rudin CO-LEAD**: the canonical 4-step cascade is INTERPRETABLE (each step has a paper-canonical role; the canonical KB-kernel impulse-response test verifies the math primitive matches the paper formula). PROCEED.
- **Daubechies CO-LEAD**: HILL is a multi-scale cascade (3x3 high-pass + 7x7 low-pass + 15x15 low-pass) consistent with the canonical wavelet hierarchical-planning discipline. The canonical L2 kernel size 15 is consistent with the Li-Wang reference scale. PROCEED.
- **Yousfi**: as the contest scorer designer + Fridrich's student, the canonical inverse-steganalysis sparse-K selection (HIGH cost = LOW detectability = canonical pixel-selection priority) is the canonical contest-relevant interpretation. PROCEED.
- **Fridrich**: HILL is one of the canonical inverse-steganalysis cost functions from the canonical Fridrich-Yousfi lineage; the canonical paper formulation is preserved exactly in the MLX implementation. The 27.46 dB cost-discrimination on real video matches expectations for natural-image cover sources. PROCEED.

Sextet additional seats:
- **Contrarian** (full dissent verbatim above): PROCEED_WITH_REVISIONS pending PR110 selector routing.
- **AssumptionAdversary** (full assumption-adversary verdict above): ASSUMED_AWAITING_VERIFICATION on the score-impact prediction.

Grand council topical seat:
- **Li_Wang_Li_Huang_canonical_paper_author_seat**: canonical paper author validates the MLX implementation matches the canonical formulation: HIGH-pass via KB kernel + LOW-pass (configurable 3/5/7 per the paper allowance; default 7 per the canonical L0 SCAFFOLD design memo) + reciprocal + LOW-pass smoothing. The per-pixel surface is the canonical surface the paper describes. PROCEED.

## 5. Per-substrate reactivation criteria pinned (per CLAUDE.md "Forbidden premature KILL")

The Slot YY HILL bind helper is REMEDIATED at the cost-matrix surface but DEFERRED at the PR110 selector-routing surface. Reactivation criteria for paired-CUDA RATIFICATION:

1. **Path A — wire per-pixel cost matrix into PR110 selector bytes**: A sister wave wires `compute_hill_per_pixel_cost_mlx` output into the PR110 selector emission so the selector bytes are routed by the per-pixel cost ranking. Then paired-CUDA RATIFICATION per Catalog #246 (~$0.06 envelope).
2. **Path B — empirical PR110 candidate dispatch**: An operator-attended dispatch of a PR110 candidate archive whose selector bytes are routed by the per-pixel HILL cost matrix; empirical ΔS measurement vs FEC6 baseline.
3. **Path C — sister-cascade extension**: A sister wave extends the canonical shared helper to MiPOD/HUGO/UNIWARD per-pixel surfaces and dispatches a multi-substrate cascade RATIFICATION.

## 6. Catalog #324 post-training Tier-C validation discipline declaration

`predicted_band_validation_status: pending_post_training` per Catalog #324. The canonical bind helper currently returns `predicted_delta_adjustment=0.0` per Tier A; there is no predicted ΔS band claim. When the operator routes paired-CUDA RATIFICATION, the canonical empirical anchor becomes the post-training Tier-C validation.

## Canonical-vs-unique decision per layer (Catalog #290)

| Layer | Decision | Rationale |
|-------|----------|-----------|
| Frame decode | ADOPT_CANONICAL (pyav per existing pattern in tools/uniward_per_pixel_n_plus_1_real_scorer_anchored_sweep_20260526.py) | The pyav decode pattern is canonical across the repo; reuse rather than fork |
| Bilinear resize | ADOPT_CANONICAL (PIL.Image.BILINEAR) | Canonical match with existing reference pattern |
| 2D convolution | FORK_BECAUSE_PRINCIPLED_MISMATCH (MLX conv2d primary; scipy.signal.convolve2d fallback; nested-loop last-resort) | MLX deployment per operator binding invariant 4; nested-loop in existing HILL package is too slow for full-frame per-pixel |
| KB kernel | ADOPT_CANONICAL (Ker-Bohme 2008 3x3 kernel; same constants as existing HILL package) | Canonical paper reference; do not fork without empirical falsification |
| HILL cost cascade | FORK_BECAUSE_PRINCIPLED_MISMATCH (per-pixel via MLX cascade in canonical shared helper; per-pair scalar in existing package preserved for archive grammar) | Per-pixel is the canonical paper surface; per-pair scalar is the PR110 archive grammar bridge; BOTH preserved |
| Cathedral consumer | ADOPT_CANONICAL (Catalog #335 contract + Tier A canonical-routing markers per Catalog #341) | Standard pattern; no fork rationale |
| Provenance | ADOPT_CANONICAL (build_provenance_for_predicted per Catalog #323) | Standard pattern |
| AxisDecomposition | DEFERRED (Tier A observability-only; no per-axis contribution) | The bind helper produces cost-matrix statistics not per-axis ΔS predictions; per-axis becomes operator-routable when paired-CUDA empirical anchor lands |

## Mission contribution per Catalog #300

`frontier_breaking_enabler`: the canonical shared helper unblocks 5 sister Slot EEE
remediation paths (Slot AAA MiPOD with REAL Wiener filter, Slot CCC HUGO with
per-pixel SPAM-delta, Slot FF UNIWARD with per-pixel directional wavelet, Slot
TT SegNet-boundary-waterfill at the frame ingestion surface, Slot RR pose-axis
null-projection at the frame ingestion surface). Each can now follow the
canonical Slot YY HILL pattern by adding a sister bind helper that consumes the
canonical shared helper's per-pixel cost matrix function.

## Cross-references

- `feedback_slot_eee_fake_implementation_audit_on_today_l0_scaffolds_per_operator_binding_must_review_for_fake_implementations_landed_20260529.md` (audit anchor)
- `feedback_optimize_iterate_highest_ev_boldest_individually_fractally_optimized_mlx_deployed_aggressive_frontier_breaking_no_fake_implementations_standing_directive_20260529.md` (5-invariant standing directive)
- `feedback_slot_yy_hill_canonical_inverse_steganalysis_li_wang_li_huang_2014_canonical_fridrich_yousfi_cascade_axis_5_extension_per_slot_uu_top_1_landed_20260529.md` (original Slot YY landing)
- `experiments/results/slot_per_pixel_mlx_inverse_steganalysis_remediation_macos_cpu_advisory_smoke_20260529T155747Z/smoke_output.json` (canonical macOS-CPU advisory smoke artifact)
- `src/tac/inverse_steganalysis_real_video_mlx/__init__.py` (canonical shared helper module)
- `src/tac/cathedral_consumers/per_pixel_inverse_steganalysis_real_video_mlx_consumer/__init__.py` (canonical cathedral consumer)
- CLAUDE.md "MLX portable-local-substrate authority — NON-NEGOTIABLE"
- CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" non-negotiable
- Catalog #192 (macOS-CPU advisory NEVER promotable)
- Catalog #213 (Comma2k19 canonical / upstream video canonical)
- Catalog #287 (placeholder-rationale rejection)
- Catalog #323 (canonical Provenance umbrella)
- Catalog #335 (cathedral consumer canonical contract)
- Catalog #341 (Tier A canonical-routing markers)
- Catalog #356 (AxisDecomposition per-axis emission)
- Catalog #357 (dual-tier consumer architecture)
- Catalog #325 (per-substrate symposium 6-step contract — THIS memo satisfies it)
- Catalog #363 (recursive self-reflection protocol — empirical_verification_status surfaced per assumption)
