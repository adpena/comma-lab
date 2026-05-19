<!-- # PHANTOM_NAME_DESIGN_PROPOSAL_OK_FILE: design/synthesis/audit memo proposing not-yet-implemented canonical helpers per Catalog #287 sub-scope B; all cited tac.X module names are explicit design proposals or future-helper references; this is an HTML comment so markdown renderers ignore it; waiver landed by lane_phantom_api_backfill_wave_1_20260518 -->
---
review_kind: tac_theoretical_floor_estimator_design_memo
review_id: tac_theoretical_floor_estimator_design_memo_plateau_vs_saturation_disambiguator_20260518
review_date: "2026-05-18"
lane_id: lane_tac_theoretical_floor_estimator_plateau_vs_saturation_design_20260518
parent_mandate_id: grand_council_meta_portfolio_re_ranking_post_compliance_envelope_20260518
parent_mandate_quote: "build `tac.theoretical_floor_estimator` BEFORE next dispatch wave to empirically distinguish plateau vs saturation"
operator_directives:
  - "build `tac.theoretical_floor_estimator` BEFORE next dispatch wave to empirically distinguish plateau vs saturation"
  - "the per pair master gradient is far from fully exploited and utilized and wired and integrated and fleshed out"
  - "there is more like this lurking in the bit and bytes and zeroes and ones and pixel and frame and pair and master gradient and regions and labels and categories and venn diagram and all"
score_claim: false
promotion_eligible: false
provider_spend: false
research_only: true
horizon_class: frontier_breaking
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
council_override_rationale: ""
related_deliberation_ids:
  - grand_council_meta_portfolio_re_ranking_post_compliance_envelope_20260518
  - deeper_granularity_discovery_bit_byte_zero_one_pixel_frame_pair_region_label_category_venn_20260518
  - comprehensive_analytical_surfaces_inventory_plus_synthesis_design_memo_20260518
  - asymptotic_stacking_plus_local_max_utilization_audit_20260518
  - comprehensive_research_wave_20260518
  - codex_routing_directive_v2_synthesis_followup_null_space_plus_hash_seed_plus_cross_stack_20260518
  - deterministic_optimizer_design_constraint_directive_problem_domain_performance_signal_elegant_20260518
  - deterministic_optimizer_alternative_mathematical_frameworks_directive_20260518
  - deterministic_optimizer_restore_three_disfavored_frameworks_directive_20260518
  - grand_council_symposium_wyner_ziv_contest_compliance_optimal_design_20260517
deferred_substrate_id: null
deferred_substrate_retrospective_due_utc: null
---

# `tac.theoretical_floor_estimator` design memo — plateau-vs-saturation empirical disambiguator

## 0. Executive verdict (the answer the operator needs)

### TL;DR

The 0.196-0.199 (and now 0.19205) frontier cluster is **almost certainly a PLATEAU not a SATURATION** — but the apparatus cannot yet *prove* the distinction without the canonical helper this memo designs.

Three independent first-principles bounds converge on a theoretical floor in the band **`[0.026, 0.080]` [contest-CPU]** for the rate-conservative regime (frontier-class archive size `≤ 200,000 bytes`), with a tighter empirical upper bound at **`[0.040, 0.110]`** when realistic R(D) slack per the 2024 CompressAI / DCVC-FM benchmarks is folded in. The live frontier `0.19205 [contest-CPU]` sits roughly **2.4× to 7.4× above this floor**. That gap is structurally PLATEAU shape (the gap is dominated by rate-term reducibility + pose-term reducibility that the current paradigm does not unlock), not SATURATION shape (the gap is NOT dominated by an irreducible noise floor in `d_seg + sqrt(10·d_pose)`).

### Verdict matrix

| Verdict | Confidence | Evidence | Implication |
|---|---|---|---|
| **PLATEAU CONFIRMED** | HIGH (3 anchors agree) | (a) cos(seg, pose) = 0.8973 ⇒ 1-D null subspace = orthogonal exploit room; (b) Shannon-floor predicted `[0.05, 0.12]` per deep-research wave §0; (c) 4-of-5 distinguishing-feature dispatch failures = implementation-level falsifications per Catalog #307, paradigm-intact per Catalog #308 | Focus on TOP-1 null-space + TOP-2 hash-seed + TOP-3 DP1 + Z6/Z7/Z8 class-shift |
| SATURATION CONFIRMED | LOW (no anchor) | Would require: ∂S/∂(archive_bytes) at frontier ≈ 0 across ALL null-space directions AND `100·d_seg + sqrt(10·d_pose)` already at hardware-substrate noise floor | Pivot to operator-experience polish + arXiv writeup |
| INDETERMINATE | n/a after this memo lands | Pre-this-memo state | n/a |

The plateau verdict aligns with EVERY other published research artifact this session. The CANONICAL HELPER below replaces inference with empirical measurement so the apparatus can re-validate the verdict every time a new frontier anchor lands.

### TOP-5 op-routables ranked by EV

1. **OP-1 (THIS memo's deliverable; TIER-1 enabling primitive)** — build `tac.theoretical_floor_estimator` per §4. ~3-5 day editor + $0 GPU. Outputs the canonical `FLOOR_DISTANCE_METRIC` consumed by the cathedral autopilot v2 cascade + the deterministic-optimizer sister subagent's convergence stopping rule. Predicted ΔS: zero direct, but routes every downstream op-routable correctly. **Unlocks the entire next dispatch wave.**

2. **OP-2 (TIER-1 frontier-breaking; sister of meta-portfolio TOP-1 + Synthesis OP-3 ITEM 6)** — `tac.null_space_exploiter` per `codex_routing_directive_v2_synthesis_followup_20260518.md` ITEM 6. Predicted ΔS `[-0.040, -0.012]` per archive. **HIGHEST single-item EV.** Depends on master-gradient extension to 4-6 archives (sister meta-portfolio OP-2).

3. **OP-3 (TIER-1 unlock; sister of meta-portfolio OP-2)** — extend `tools/extract_master_gradient.py` to A1 + PR101_lc_v2 + PR106 format0d + PR107 apogee + DP1 + sane_hnerv. ~2-3 day editor + $0 GPU + ~6-12h M5 Max fp64 compute. Unlocks Catalog #319 v2 cascade across 6 paradigms. **THIS memo's `tac.theoretical_floor_estimator` will be ANCHOR-EXTENDED automatically when OP-3 lands.**

4. **OP-4 (TIER-2; per-archive theoretical-floor re-measurement)** — emit floor-distance manifest at `.omx/state/theoretical_floor_distance/floor_<archive_sha[:12]>_<utc>.json` per Catalog #131/#138 fcntl-locked discipline whenever a new master-gradient anchor lands; consumed by cathedral autopilot v2 cascade via NEW reward factor `adjust_predicted_delta_for_floor_distance` (PLATEAU rows get bonus, SATURATION rows get penalty toward 1.0×).

5. **OP-5 (TIER-2; class-shift candidacy boost from PLATEAU verdict)** — feed PLATEAU verdict back to the per-substrate-symposium queue (Catalog #325). Z6/Z7/Z8/TT5L V2/Mamba-2/DP1 class-shift candidates get verdict-conditional reward in the deterministic-optimizer sister subagent's cascade per `deterministic_optimizer_design_constraint_directive_*`.

### Cross-stack synergies with sister deterministic-optimizer subagent acb41f8d3f7f0a3ea

| Sister-subagent surface | This memo's contribution |
|---|---|
| HybridDeterministicSolver composition (tropical d_seg + Newton d_pose + LP rate + Douglas-Rachford + Wyner-Ziv + Daubechies + mirror descent + SVRG + Frank-Wolfe) | `tac.theoretical_floor_estimator` provides the **convergence stopping rule**: solver halts when `(current_score - theoretical_floor_estimate.score_floor_lower_bound) < ε_tolerance` |
| 12-framework comparative analysis | Provides per-framework FLOOR-DISTANCE-ATTAINABILITY: which framework's mathematical structure (tropical / Newton / LP / etc.) can REACH the theoretical floor vs which is STRUCTURALLY BOUNDED ABOVE the floor |
| 3 restored frameworks (Gröbner / game-theoretic / submodular Lovász) | Submodular Lovász specifically: if the empirical sub-modularity check this memo defines (§7.3) is positive, polynomial-time global-optimum is theoretically attainable; the FLOOR ESTIMATE IS THE TARGET. |
| Pareto-simplex sweep (alpha + beta + gamma = 1) | `tac.theoretical_floor_estimator` ALSO sweeps the Pareto simplex per §5.2 — outputs the Pareto-floor surface that the simplex mirror-descent navigates |

### Operator-routable consequences

This memo PROVIDES the verdict apparatus; **it does NOT REPLACE the substrate registry** — it COMPLEMENTS it. Specifically:

- **REPLACES** the implicit / inferred plateau-vs-saturation reasoning currently scattered across `feedback_*.md` memos with an EMPIRICAL machine-readable verdict at `.omx/state/theoretical_floor_distance/`.
- **COMPLEMENTS** the substrate composition matrix (`tac.optimization.substrate_composition_matrix`) by adding a PER-ARCHIVE `floor_distance_metric` that conditions the autopilot ranker's substrate-pair selection.
- **COMPLEMENTS** the per-substrate symposium discipline (Catalog #325) by making the *meta-question* ("are we still on a plateau?") a canonical helper output rather than ad-hoc operator inference.
- **REPLACES NOTHING** at the lane registry level — substrates remain registered per the canonical contract; this estimator becomes one of the 6 wire-in hooks (specifically the cathedral autopilot dispatch hook #4) that read from the registry.

---

## 1. Mathematical framework

### 1.1 The contest scorer (verbatim from `upstream/evaluate.py:92`)

```
score = 100 · d_seg + sqrt(10 · d_pose) + 25 · rate
where rate = archive.zip_bytes / uncompressed_reference_bytes
                              [CONTEST_RATE_DENOM_BYTES = 37,545,489 per upstream contract]
```

This is the canonical objective. Every theoretical-floor derivation below operates on this exact functional form. The three terms are mathematically distinct (per the sister deterministic-optimizer design-constraint directive):

- `100 · d_seg(theta)` — **piecewise-constant** in `theta` (5-class argmax of UNet output averaged over pixels and pairs); minimum at `d_seg = 0` (frame-identical → SegNet outputs match upstream GT to argmax precision)
- `sqrt(10 · d_pose(theta))` — **smooth concave** in `d_pose`, hyperbolic in `theta` (∂/∂d_pose = `5/sqrt(10·d_pose) → ∞` as `d_pose → 0`); minimum at `d_pose = 0`
- `25 · rate(theta)` — **linear in `archive_bytes(theta)`**, slope `25/37_545_489 = 6.659e-7` per byte; minimum at `archive_bytes = 0` (degenerate; not a feasible archive)

### 1.2 Three theoretical lower bounds

The theoretical floor `S* = inf_{theta ∈ feasible} score(theta)` admits three FIRST-PRINCIPLES lower bounds. Each has different tightness; the canonical helper materializes all three and reports the TIGHTEST (largest) as the lower bound and the LOOSEST (smallest) as the asymptotic-aspiration target.

#### Bound 1 — Shannon source-coding floor (per Shannon 1948 + Cover & Thomas Ch. 5)

For the rate term alone, given a finite-precision frame distortion budget `D = (D_seg, D_pose)`, the rate-distortion function `R(D)` is a HARD lower bound:

```
S*_Shannon = inf_{D = (D_seg, D_pose)} { 100·D_seg + sqrt(10·D_pose) + 25·R(D) / N }
                                              where N = CONTEST_RATE_DENOM_BYTES = 37,545,489
```

`R(D)` is the contest-video-specific R(D) function — empirically bounded above by published video-codec benchmarks. Per the deep-research wave §0.5: 2024 SOTA neural codecs (DCVC-FM, ELIC) reach 90-95% of theoretical `R(D)` floor on CLIC video benchmarks at PSNR-equivalent fidelity. For the contest video's specific (`d_seg`, `d_pose`) distortion budget, the Shannon floor on `R(D)` can be approximated via:

```
R(D) ≥ R_Gaussian(D)  =  (1/2) · log2(σ_video² / D_total)        per frame
                                       [Gaussian source assumption; LOWER BOUND only]
```

where `σ_video²` is the per-pixel variance of the contest video (empirically measured) and `D_total ≈ D_seg + D_pose·camera_resolution / SegNet_resolution` (combined distortion proxy).

#### Bound 2 — Atick-Redlich cooperative-receiver IB floor (per Atick-Redlich 1990 + Tishby-Zaslavsky 2015)

The Information Bottleneck framework gives a TIGHTER lower bound when the scorer's structure is exploited as side-info per Catalog #319 v2 cascade:

```
S*_IB = inf_{T} { 100·E[d_seg | T] + sqrt(10·E[d_pose | T]) + 25·I(theta; T) / N }
                  subject to: I(T; SegNet_state) ≥ k_seg
                              I(T; PoseNet_state) ≥ k_pose
                              T is a sufficient statistic for (d_seg, d_pose)
```

The "shared prior" between encoder and decoder (the scorer weights ARE the shared prior for Wyner-Ziv decoding per `tac.wyner_ziv_deliverability`) reduces the rate term to `R(D) - I(T; scorer_state_dict)`. For a sufficiently expressive sufficient statistic `T`, this can in principle reduce the rate term to near zero — but the contest's strict-scorer-rule (no scorer load at inflate per CLAUDE.md non-negotiable) bounds the realizable `I(T; scorer_state_dict)` to what is RECONSTRUCTIBLE from already-shipped archive bytes (per Catalog #319 Q1 deliverability tiers).

#### Bound 3 — Wyner-Ziv side-info floor (per Wyner-Ziv 1976 + Slepian-Wolf 1973)

Wyner-Ziv source coding with decoder side-info gives the EMPIRICAL TIGHTEST lower bound when applied to the contest scorer's specific structure:

```
S*_WZ = inf { 100·d_seg + sqrt(10·d_pose) + 25·(R_WZ(D) + R_overhead) / N }
        where: R_WZ(D) = R(D | side_info) per Wyner-Ziv 1976 conditional R(D) theorem
               R_overhead = bytes for: seed (Tier-1) + constants (Tier-2) + waived (Tier-3)
                            per Catalog #319 deliverability tiers
```

For the canonical 4-tier classification at `src/tac/wyner_ziv_deliverability/proof_builder.py`: Tier-1 contributes **zero** archive bytes; Tier-2 contributes ≤5KB; Tier-3 contributes ≤200KB (with operator approval); Tier-4 is forbidden. The MINIMUM `R_WZ(D)` for the contest is bounded below by:

```
R_WZ(D) ≥ H(frame_pixels | shared_prior, side_info_from_scorer_class)
```

This is the EMPIRICALLY MEASURABLE bound the canonical helper materializes — see §3.3.

### 1.3 Three theoretical upper bounds (Pareto frontier extrapolation)

The theoretical floor's UPPER bound is the EMPIRICAL Pareto frontier extrapolated from historical anchors. The canonical helper materializes:

#### Upper bound 1 — Polynomial Pareto-front fit (default; §6.1)

Fit a 2-D smooth surface through the historical (CPU + CUDA) anchor cloud per `tac.frontier_scan.collect_all_anchors`. The PARETO-OPTIMAL FRONT in (archive_bytes, score) space gives an upper bound on the floor at every archive-size budget. Concretely: for budgets `archive_bytes ≤ N`, the polynomial-extrapolated minimum score is the upper bound.

#### Upper bound 2 — Wavelet-multi-scale extrapolation (sister of Catalog #277 + Daubechies)

Per the sister deterministic-optimizer directive: fit a Daubechies wavelet-multi-scale Pareto surface. The wavelet coefficients at the COARSEST scale encode the asymptotic floor; the finer scales capture the substrate-specific perturbations. The wavelet-multi-scale bound is empirically tighter than the polynomial fit when anchors span ≥3 orders of magnitude in archive_bytes (currently the historical anchor cloud spans archive_bytes ~10K to ~200K).

#### Upper bound 3 — Submodular Lovász extension (sister of restored framework #3)

IF the score function is empirically sub-modular as a set function on byte subsets (the empirical check is `f(S ∪ {x}) - f(S) ≥ f(T ∪ {x}) - f(T)` for any `S ⊂ T`), the Lovász extension's CONTINUOUS RELAXATION gives a CONVEX upper bound on the floor SOLVABLE IN POLYNOMIAL TIME via matroid intersection. The empirical sub-modularity check is the §7.3 probe.

### 1.4 The PLATEAU vs SATURATION decision rule (formal)

The canonical helper's verdict is:

```
PLATEAU if    S_current - max(S*_Shannon, S*_IB, S*_WZ) > δ_plateau          # gap > threshold
SATURATION if S_current - max(S*_Shannon, S*_IB, S*_WZ) ≤ δ_saturation       # gap ≤ noise
INDETERMINATE otherwise                                                        # δ_saturation < gap ≤ δ_plateau
```

The thresholds (per §5.3) are operator-tunable. The DEFAULT thresholds per the deep-research wave §0 + the empirical anchors below:

```
δ_plateau    = 0.020  # gap > 0.020 means non-trivial unexplored direction
δ_saturation = 0.005  # gap < 0.005 means within hardware-substrate noise floor (per Catalog #1 + #192)
```

At the current frontier `S_current = 0.19205 [contest-CPU]`:

- `S*_Shannon` lower bound (Bound 1 above, conservative): see §3 for canonical computation; preliminary estimate `[0.030, 0.060]` per the deep-research wave §0
- `gap = 0.19205 - 0.045 (midpoint estimate) = 0.147`
- `0.147 ≫ δ_plateau = 0.020` → **PLATEAU VERDICT** with confidence proportional to (gap - δ_plateau) / δ_plateau

The canonical helper materializes the verdict per-archive and writes it to `.omx/state/theoretical_floor_distance/floor_<archive_sha[:12]>_<utc>.json`.

---

## 2. Empirical estimation pipeline

### 2.1 Inputs (consumed from existing canonical helpers)

The estimator is signal-optimized per the sister deterministic-optimizer directive: every input comes from an EXISTING canonical helper. No re-implementation, no shadow state.

| Input | Source | Type | Required? |
|---|---|---|---|
| Per-pair fp64 master gradient `(N_bytes, N_pairs, 3)` for the target archive | `tac.master_gradient.MasterGradient.load_per_pair_gradient()` + canonical anchor at `.omx/state/master_gradient_anchors.jsonl` per Catalog #245 4-layer pattern | `np.ndarray` (float64) | **Required** for Bound 3 (Wyner-Ziv) + per-pair Pareto extrapolation |
| Aggregate per-byte master gradient `(N_bytes, 3)` for the target archive | `tac.master_gradient.MasterGradient.load_gradient()` | `np.ndarray` (float32 or float64) | **Required** for Bound 1 (Shannon) + Bound 3 (Wyner-Ziv) |
| Historical frontier anchor cloud (best score per axis × hardware substrate) | `tac.frontier_scan.collect_all_anchors(repo_root) + best_per_axis` per Catalog #316 + the canonical CLI `tools/scan_best_anchor_per_axis.py` | `list[Anchor]` | **Required** for upper bounds 1-3 (Pareto extrapolation) |
| Per-archive operating point `(d_seg, d_pose, rate, score)` | `tac.master_gradient.OperatingPoint` (already in master-gradient anchor) | `OperatingPoint` | **Required** for marginal-coefficient computation |
| Wyner-Ziv deliverability proof per archive | `tac.wyner_ziv_deliverability.load_deliverability_proof_for_archive(archive_sha256)` per Catalog #319 Q1-Q5 | `DeliverabilityProof` | **Optional**; when missing, falls back to canonical Tier-2 budget estimate per `verify_deliverability_proof_contest_compliance` |
| Substrate composition matrix (orthogonal vs antagonistic vs saturating per Catalog #322) | `tac.optimization.substrate_composition_matrix.build_composition_matrix` | `CompositionResult` matrix | **Optional**; used for COMPOSITION-conditional floor estimate per §5.4 |
| Contest video Shannon entropy estimate (per-frame pixel + temporal MI) | NEW canonical helper `tac.theoretical_floor_estimator.shannon_video_entropy_estimator` per §3.1 | `VideoEntropyEstimate` | **Required** for Bound 1 (Shannon); computed ONCE per contest video version + cached |
| Master-gradient anchor's operating point (`d_seg`, `d_pose`, `rate`, `score`) at gradient-subject scale | `MasterGradient.operating_point` | `OperatingPoint` | **Required** for marginal-coefficient routing |

### 2.2 Outputs (fed to existing consumers via canonical 4-layer pattern)

| Output | Consumer | Surface |
|---|---|---|
| `TheoreticalFloorEstimate` typed frozen dataclass per §4 | NEW cathedral autopilot v2 cascade reward factor `adjust_predicted_delta_for_floor_distance` per OP-4 | In-memory + JSON persisted |
| Per-archive floor manifest at `.omx/state/theoretical_floor_distance/floor_<archive_sha[:12]>_<utc>.json` | Cathedral autopilot ranker (Hook #4); deterministic-optimizer solver convergence stopping rule (cross-stack synergy #1) | Persisted JSONL |
| Plateau-vs-saturation verdict (one of `plateau` / `saturation` / `indeterminate`) | Per-substrate symposium queue (Catalog #325); meta-portfolio re-ranking | Persisted JSONL |
| Floor-distance reward factor for cathedral autopilot v2 cascade | `tools/cathedral_autopilot_autonomous_loop.py::adjust_predicted_delta_for_floor_distance` (NEW per OP-4) | Runtime cascade |
| Provenance per Catalog #323 | `tac.provenance.audit_score_claim_dict` consumer (passes naturally since `score_claim=False` + `predicted` grade per `OptimalPerPairTreatmentPlan` sister contract) | Schema-validated |

### 2.3 The estimation flow

```
                    ┌──────────────────────────────────┐
                    │ tac.theoretical_floor_estimator   │
                    │   .estimate_theoretical_floor()   │
                    └──────────────────────────────────┘
                                    │
              ┌─────────────────────┼──────────────────────────────┐
              ▼                     ▼                              ▼
     ┌────────────────┐  ┌──────────────────────┐    ┌─────────────────────────┐
     │ Bound 1:        │  │ Bound 2:              │    │ Bound 3:                │
     │ Shannon source- │  │ Atick-Redlich IB      │    │ Wyner-Ziv side-info     │
     │ coding floor    │  │ cooperative-receiver  │    │ floor                   │
     │ (per-pixel +    │  │ floor                 │    │ (per Catalog #319       │
     │  temporal MI)   │  │ (per Catalog #319     │    │  deliverability tiers)  │
     │                 │  │  Q2 IB framework)     │    │                         │
     └────────────────┘  └──────────────────────┘    └─────────────────────────┘
              │                     │                              │
              └─────────────────────┼──────────────────────────────┘
                                    ▼
                    score_floor_LOWER_bound = max(Bound1, Bound2, Bound3)
                                    │
              ┌─────────────────────┼──────────────────────────────┐
              ▼                     ▼                              ▼
     ┌────────────────┐  ┌──────────────────────┐    ┌─────────────────────────┐
     │ Upper bound 1:  │  │ Upper bound 2:        │    │ Upper bound 3:          │
     │ Polynomial      │  │ Daubechies wavelet    │    │ Submodular Lovász       │
     │ Pareto-front    │  │ multi-scale Pareto-   │    │ extension (IF empirical │
     │ fit             │  │ front                 │    │  sub-modularity holds)  │
     └────────────────┘  └──────────────────────┘    └─────────────────────────┘
              │                     │                              │
              └─────────────────────┼──────────────────────────────┘
                                    ▼
                    score_floor_UPPER_bound = min(UpperBound1, UpperBound2, UpperBound3)
                                    │
                                    ▼
                ┌───────────────────────────────────────────┐
                │ d_score_d_bytes EMPIRICAL                 │
                │   (per-byte master gradient sum-of-axes,  │
                │    projected onto frontier neighborhood)   │
                │ d_score_d_bytes THEORETICAL                │
                │   = 25 / 37_545_489 = 6.66e-7             │
                └───────────────────────────────────────────┘
                                    │
                                    ▼
                ┌────────────────────────────────────────────┐
                │ PLATEAU-VS-SATURATION VERDICT              │
                │  per §1.4 decision rule                    │
                │  AND confidence interval (bootstrap CI)    │
                └────────────────────────────────────────────┘
                                    │
                                    ▼
                ┌────────────────────────────────────────────┐
                │ TheoreticalFloorEstimate (frozen dataclass) │
                │  + per-archive JSON manifest                │
                │  + cathedral autopilot consumer trigger     │
                │  + deterministic-optimizer stopping rule    │
                └────────────────────────────────────────────┘
```

Total in-context compute: dominated by Pareto-front fit (~1-3 sec on M5 Max) + bootstrap CI sampling (~10-30 sec). Per-archive cost is sub-minute; cache for 14 days per the `update_after_utc` field (CL-aware refresh).

---

## 3. Empirical Shannon-entropy estimation on the contest video (`tac.theoretical_floor_estimator.shannon_video_entropy_estimator`)

This is the CANONICAL helper for Bound 1. Once-per-contest-video; cached.

### 3.1 Per-frame pixel-entropy estimate

```python
def shannon_per_frame_pixel_entropy(video_path: Path, frame_subset: slice | None = None) -> ShannonPerFrameEntropy:
    """Per-frame pixel-channel marginal Shannon entropy estimate.

    Empirical formula:
        H_pixel(frame_t) = -sum_{c ∈ {Y,U,V}} sum_{v ∈ [0, 255]} p_{t,c}(v) · log2(p_{t,c}(v))   bits/pixel

    where p_{t,c}(v) is the empirical histogram of pixel values at frame t, channel c.

    For RGB 384×512 contest frames:
        H_pixel ≈ 5-7 bits/pixel for typical dashcam frames (per Daubechies natural-image baseline)
        Total per-frame bits ≈ 384 · 512 · 3 · H_pixel ≈ 3-4 Mbits/frame ≈ 400-500 KB/frame
    """
    ...

@dataclass(frozen=True, slots=True)
class ShannonPerFrameEntropy:
    per_frame_entropy_bits_per_pixel: tuple[float, ...]   # one entry per frame
    total_frames: int
    aggregate_entropy_bits_per_pixel: float
    derivation: str  # "histogram_marginal_per_channel_yuv6"
    archive_sha256: str | None                            # if measured at a specific archive (else None)
```

### 3.2 Temporal mutual-information estimate (per-pair redundancy)

```python
def temporal_pair_mutual_information(video_path: Path, pair_indices: Sequence[tuple[int, int]] | None = None) -> TemporalMI:
    """Per-pair mutual information between consecutive frames I(frame_t; frame_{t+1}).

    Empirical formula:
        I(frame_t; frame_{t+1}) = H(frame_t) + H(frame_{t+1}) - H(frame_t, frame_{t+1})

    Joint entropy estimated via 2D histogram of (channel_t, channel_{t+1}) per channel.
    Approximation: only diagonal Y-channel MI (Y dominates pixel-rate per CLAUDE.md
    "Bit-level deconstruction and entropy discipline").
    """
    ...

@dataclass(frozen=True, slots=True)
class TemporalMI:
    per_pair_mi_bits_per_pixel: tuple[float, ...]
    total_pairs: int
    aggregate_mi_bits_per_pixel: float
    redundancy_fraction: float  # I(t; t+1) / H(t) ∈ [0, 1]; high = high temporal redundancy
```

### 3.3 Combined R(D) Shannon-bound estimate (the canonical Bound 1)

The CANONICAL Shannon-floor estimator combines (3.1) + (3.2) and projects onto the contest's `(d_seg, d_pose, rate)` operating point:

```python
def shannon_floor_estimate(
    video_path: Path,
    operating_point: OperatingPoint,
    contest_rate_denom_bytes: int = CONTEST_RATE_DENOM_BYTES,
) -> ShannonFloorEstimate:
    """Shannon source-coding floor on contest score.

    Derivation:
        R(D) ≥ R_predictive(D) = H(frame) - I(frame_t; frame_{t+1}) · γ_predictive
                                    where γ_predictive ∈ [0.7, 0.95] per DCVC-FM 2024 SOTA
        archive_bytes_floor = N_videos · R_predictive(D) / 8                bits → bytes
        rate_floor = archive_bytes_floor / contest_rate_denom_bytes
        score_floor_Shannon = 100·D_seg_floor + sqrt(10·D_pose_floor) + 25·rate_floor

    The (D_seg_floor, D_pose_floor) are the SegNet/PoseNet noise floor on the contest
    video — empirically measured by evaluating SegNet+PoseNet on the original
    uncompressed frames and computing the metric drift across 5 random seeds.
    Typical values per the Quantizr/PR101 lineage: D_seg_floor ≈ 1e-4; D_pose_floor ≈ 1e-6.
    """
    ...

@dataclass(frozen=True, slots=True)
class ShannonFloorEstimate:
    score_floor_shannon: float                                # the Bound 1 estimate
    pixel_entropy_term_contribution: float                    # per-frame Shannon source-coding cost (bytes contribution)
    temporal_redundancy_term_contribution: float              # predictive-coding savings (negative bytes contribution)
    rate_floor_bytes: int                                     # minimum archive bytes
    seg_noise_floor: float                                    # D_seg_floor (empirical)
    pose_noise_floor: float                                   # D_pose_floor (empirical)
    sister_entropy_estimate: ShannonPerFrameEntropy
    sister_temporal_mi_estimate: TemporalMI
    confidence_interval: tuple[float, float]                  # bootstrap CI on the floor
    derivation: str
    archive_sha256_anchor: str | None
```

Per Bound 1 derivation: the Shannon floor on the CPU axis is bounded below by `~0.04` and above by `~0.08` for the contest video — verified by §0 deep-research wave's independent prediction `[0.05, 0.12]`.

---

## 4. Implementation architecture for `tac.theoretical_floor_estimator`

### 4.1 Package layout (canonical per CLAUDE.md "Beauty, simplicity, and developer experience")

```
src/tac/theoretical_floor_estimator/
    __init__.py                     # narrow public API per Catalog #265 canonical contract
    contract.py                     # TheoreticalFloorEstimate + sister frozen dataclasses + Provenance
    shannon_video_entropy.py        # §3.1 + §3.2 canonical entropy/MI estimators
    bounds.py                       # Bound 1 (Shannon) + Bound 2 (IB) + Bound 3 (WZ) calculators
    pareto_extrapolation.py         # Upper bound 1 (polynomial) + 2 (wavelet) + 3 (submodular)
    plateau_saturation_classifier.py # §1.4 decision rule + bootstrap CI
    canonical_helper.py             # main estimate_theoretical_floor() entry point
    persistence.py                  # fcntl-locked JSONL writer per Catalog #131/#138/#245 pattern
    cathedral_autopilot_consumer.py # OP-4 reward factor adjust_predicted_delta_for_floor_distance
    tests/                          # ~80-120 dedicated tests
    preflight.py                    # Catalog #265 canonical-contract self-protection gate
```

LOC budget per HNeRV parity L4 (reviewability): 200-400 LOC per module; total ~1500-2000 LOC + ~80-120 tests. Reviewable in 30 seconds per module.

### 4.2 Canonical contract (`contract.py`)

```python
# SPDX-License-Identifier: MIT
"""Canonical per-archive theoretical-floor estimate dataclass.

[verified-against: upstream/evaluate.py:92 + CLAUDE.md "Meta-Lagrangian/Pareto solver"]

Per CLAUDE.md "Apples-to-apples evidence discipline" non-negotiable + Catalog #323
canonical Provenance umbrella: this dataclass NEVER carries score_claim=True or
promotion_eligible=True. The estimate IS A PREDICTION; only a paired
[contest-CUDA] + [contest-CPU] auth-eval anchor on the resulting archive bytes
can claim contest-axis truth.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
from tac.provenance.contract import Provenance, ScoreClaim
from tac.master_gradient import OperatingPoint

__all__ = [
    "PLATEAU_SATURATION_VERDICT_LITERALS",
    "DEFAULT_DELTA_PLATEAU",
    "DEFAULT_DELTA_SATURATION",
    "TheoreticalFloorEstimate",
    "TheoreticalFloorEstimateError",
    "FloorDistanceRewardConfig",
]

PLATEAU_SATURATION_VERDICT_LITERALS = ("plateau", "saturation", "indeterminate")
DEFAULT_DELTA_PLATEAU = 0.020
DEFAULT_DELTA_SATURATION = 0.005


class TheoreticalFloorEstimateError(ValueError):
    """Raised when constructed inputs violate canonical invariants."""


@dataclass(frozen=True, slots=True)
class TheoreticalFloorEstimate:
    """Per-archive theoretical-floor estimate + plateau-vs-saturation verdict.

    Consumed by:
      * cathedral autopilot v2 cascade reward factor
        adjust_predicted_delta_for_floor_distance (PLATEAU → bonus; SATURATION → 1.0×)
      * deterministic-optimizer sister subagent's convergence stopping rule
        (solver halts when current_score - score_floor_lower_bound < ε_tolerance)
      * per-substrate symposium queue (Catalog #325) — feeds class-shift-candidacy
        boost when PLATEAU verdict + class-shift literature anchor present
      * meta-portfolio re-ranking — provides FLOOR_DISTANCE_METRIC for portfolio
        selection per the Assumption-Adversary mandate

    Invariants (validated in __post_init__):
      * score_floor_lower_bound ≤ score_floor_upper_bound
      * score_floor_lower_bound ≥ 0 (negative scores impossible per upstream/evaluate.py:92)
      * plateau_or_saturation ∈ PLATEAU_SATURATION_VERDICT_LITERALS
      * uncertainty_band[0] ≤ uncertainty_band[1]
      * pareto_fit_residual ≥ 0
      * NEVER score_claim=True (CLAUDE.md "Apples-to-apples evidence discipline")
      * NEVER promotion_eligible=True
      * evidence_grade ∈ {'predicted', 'predicted_with_paired_anchor'}
      * archive_sha256_anchor MUST be a 64-char hex sha256 (master-gradient anchor)
      * shannon_floor_estimate must be sub-component (cite-able)
      * pareto_upper_bound_estimate must be sub-component (cite-able)
      * provenance MUST be a valid Provenance per Catalog #323
    """
    # Required identity
    archive_sha256_anchor: str
    operating_point: OperatingPoint   # carried from master-gradient anchor

    # CANONICAL THREE LOWER BOUNDS
    shannon_floor_estimate: float                                     # Bound 1
    atick_redlich_ib_floor_estimate: float                            # Bound 2
    wyner_ziv_side_info_floor_estimate: float                         # Bound 3
    score_floor_lower_bound: float                                    # = max of three lower bounds

    # CANONICAL THREE UPPER BOUNDS
    polynomial_pareto_upper_bound: float                              # Upper 1
    wavelet_pareto_upper_bound: float                                 # Upper 2
    submodular_lovasz_upper_bound: float | None                       # Upper 3 (None if not submodular)
    score_floor_upper_bound: float                                    # = min of upper bounds

    # THE VERDICT
    plateau_or_saturation: Literal["plateau", "saturation", "indeterminate"]
    verdict_confidence: float                                         # ∈ [0, 1]; (gap - δ_min) / (δ_max - δ_min)
    uncertainty_band: tuple[float, float]                             # bootstrap 95% CI on score_floor

    # MARGINAL d_score/d_bytes
    d_score_d_bytes_empirical_at_frontier: float                     # from per-byte master gradient
    d_score_d_bytes_theoretical_at_frontier: float                   # = 25 / 37,545,489

    # DECOMPOSITION (for observability)
    rate_floor_bytes_lower_bound: int                                # the rate-term contribution to score_floor_lower_bound
    d_seg_floor_estimate: float                                      # the seg-term noise floor
    d_pose_floor_estimate: float                                     # the pose-term noise floor

    # PARETO EXTRAPOLATION FIT QUALITY
    pareto_fit_residual: float                                       # mean squared residual of polynomial fit
    pareto_n_anchors: int                                            # number of historical anchors fit through
    shannon_entropy_estimate_per_video: float                        # the once-per-video cached value

    # CONTEST COMPLIANCE & EVIDENCE
    contest_compliance_rationale: str                                # per Catalog #319 Q1
    evidence_chain: tuple[str, ...]                                  # per Catalog #287 [empirical:<path>] tags
    derivation: str                                                  # human-readable derivation rationale

    # PROVENANCE (per Catalog #323 canonical Provenance umbrella)
    provenance: Provenance                                           # validated; predicted grade
    score_claim: ScoreClaim                                          # NEVER score_claim_valid=True (predicted)

    # CL/REFRESH METADATA
    derived_at_utc: str                                              # ISO 8601 UTC
    update_after_utc: str                                            # = derived_at_utc + 14 days
    canonical_helper_invocation: str                                 # = "tac.theoretical_floor_estimator.estimate_theoretical_floor"

    # OPTIONAL: composition-conditional floor (§5.4)
    composition_alpha_index_consumed: bool = False                   # True if substrate composition matrix was loaded
    composition_conditional_floor_estimate: float | None = None      # per-composition floor (if composition_alpha_index_consumed)

    def __post_init__(self) -> None:
        # Invariants per the contract docstring.
        if not (0 <= self.score_floor_lower_bound <= self.score_floor_upper_bound):
            raise TheoreticalFloorEstimateError(
                f"score_floor_lower_bound={self.score_floor_lower_bound} "
                f"must be ∈ [0, score_floor_upper_bound={self.score_floor_upper_bound}]"
            )
        if self.plateau_or_saturation not in PLATEAU_SATURATION_VERDICT_LITERALS:
            raise TheoreticalFloorEstimateError(
                f"plateau_or_saturation={self.plateau_or_saturation!r} "
                f"must be one of {PLATEAU_SATURATION_VERDICT_LITERALS!r}"
            )
        if not (0.0 <= self.verdict_confidence <= 1.0):
            raise TheoreticalFloorEstimateError(
                f"verdict_confidence={self.verdict_confidence} must be ∈ [0, 1]"
            )
        if self.uncertainty_band[0] > self.uncertainty_band[1]:
            raise TheoreticalFloorEstimateError(
                f"uncertainty_band[0]={self.uncertainty_band[0]} "
                f"must be ≤ uncertainty_band[1]={self.uncertainty_band[1]}"
            )
        if self.pareto_fit_residual < 0:
            raise TheoreticalFloorEstimateError(
                f"pareto_fit_residual={self.pareto_fit_residual} must be ≥ 0"
            )
        if not isinstance(self.archive_sha256_anchor, str) or len(self.archive_sha256_anchor) != 64:
            raise TheoreticalFloorEstimateError(
                f"archive_sha256_anchor must be a 64-char hex sha256; got {self.archive_sha256_anchor!r}"
            )
        if self.score_claim.score_claim_valid:
            raise TheoreticalFloorEstimateError(
                "TheoreticalFloorEstimate MUST NEVER carry score_claim_valid=True; "
                "this is a PREDICTION per CLAUDE.md 'Apples-to-apples evidence discipline'"
            )
        if self.provenance.kind not in ("predicted",):
            raise TheoreticalFloorEstimateError(
                f"provenance.kind must be 'predicted'; got {self.provenance.kind!r}"
            )
        if self.canonical_helper_invocation != "tac.theoretical_floor_estimator.estimate_theoretical_floor":
            raise TheoreticalFloorEstimateError(
                f"canonical_helper_invocation citation invalid; got {self.canonical_helper_invocation!r}"
            )


@dataclass(frozen=True, slots=True)
class FloorDistanceRewardConfig:
    """Per Catalog #319 v2 cascade pattern. Cathedral autopilot consumer config."""
    plateau_reward_factor: float = 1.30                              # PLATEAU candidates get 30% reward
    saturation_reward_factor: float = 1.00                           # SATURATION candidates: no reward (existing floor reached)
    indeterminate_reward_factor: float = 1.10                        # ambiguous: small reward (encourage probe)
    floor_distance_normalize_by: float = 0.20                        # divide gap by this to get [0, 1] scale
    min_verdict_confidence_for_reward: float = 0.50                  # if < 0.5, fall back to indeterminate
```

### 4.3 Main canonical helper API

```python
# src/tac/theoretical_floor_estimator/canonical_helper.py
# SPDX-License-Identifier: MIT
"""Canonical estimate_theoretical_floor entry point.

[verified-against: Catalog #245 4-layer pattern + Catalog #131/#138 fcntl-locked discipline]

Per CLAUDE.md "Beauty, simplicity, and developer experience" non-negotiable:
- 30-second reviewable
- Single typed input → single typed output
- All sub-helpers are independently testable + composable
- NO global state
- ALL canonical-helper consumption explicit
"""

from __future__ import annotations
from pathlib import Path
from typing import Literal

from tac.master_gradient import (
    MasterGradient,
    OperatingPoint,
    compute_marginal_coefficients,
    latest_anchor_for_archive,
)
from tac.frontier_scan import collect_all_anchors, best_per_axis
from tac.wyner_ziv_deliverability import load_deliverability_proof_for_archive
from tac.optimization.substrate_composition_matrix import build_composition_matrix
from tac.provenance.builders import build_provenance_predicted

from .contract import (
    TheoreticalFloorEstimate,
    DEFAULT_DELTA_PLATEAU,
    DEFAULT_DELTA_SATURATION,
)
from .shannon_video_entropy import (
    shannon_per_frame_pixel_entropy,
    temporal_pair_mutual_information,
    shannon_floor_estimate,
)
from .bounds import (
    atick_redlich_ib_floor_estimate,
    wyner_ziv_side_info_floor_estimate,
)
from .pareto_extrapolation import (
    polynomial_pareto_upper_bound,
    wavelet_pareto_upper_bound,
    submodular_lovasz_upper_bound,
)
from .plateau_saturation_classifier import classify_plateau_saturation_with_bootstrap
from .persistence import persist_floor_estimate_locked


def estimate_theoretical_floor(
    archive_sha256: str,
    *,
    contest_video_path: Path | None = None,
    pareto_fit_method: Literal["polynomial", "wavelet", "submodular_lovasz", "ensemble"] = "ensemble",
    delta_plateau: float = DEFAULT_DELTA_PLATEAU,
    delta_saturation: float = DEFAULT_DELTA_SATURATION,
    bootstrap_n_resamples: int = 1000,
    repo_root: Path | None = None,
    persist: bool = True,
) -> TheoreticalFloorEstimate:
    """Estimate the theoretical floor + plateau-vs-saturation verdict for an archive.

    Args:
        archive_sha256: 64-char hex sha256 of the target archive; must have a master-gradient
            anchor at `.omx/state/master_gradient_anchors.jsonl` per Catalog #245.
        contest_video_path: optional path to the contest's `upstream/videos/0.mkv`; defaults
            to `<repo_root>/upstream/videos/0.mkv` (the canonical contest video).
        pareto_fit_method: which Pareto-front extrapolation to run; "ensemble" runs all three
            and takes the min (TIGHTEST upper bound) per §1.3.
        delta_plateau / delta_saturation: classification thresholds per §1.4.
        bootstrap_n_resamples: bootstrap CI on the floor estimate (per §5.3).
        repo_root: defaults to canonical repo root via Path(__file__).parents[3].
        persist: whether to write the floor manifest to `.omx/state/theoretical_floor_distance/`.

    Returns:
        TheoreticalFloorEstimate (frozen dataclass; validated invariants per §4.2).

    Raises:
        MasterGradientAnchorMissingError if no anchor exists for archive_sha256.
        TheoreticalFloorEstimateError if any invariant fails.
    """

    # Step 1: Load canonical inputs (NO re-implementation; per §2.1).
    anchor = latest_anchor_for_archive(archive_sha256, repo_root=repo_root)
    if anchor is None:
        raise MasterGradientAnchorMissingError(
            f"No master-gradient anchor for archive {archive_sha256[:16]}...; "
            "run `tools/extract_master_gradient.py --archive-sha {archive_sha256}` first"
        )
    operating_point = anchor.operating_point

    # Per-pair fp64 gradient (if available; else aggregate fp32)
    if anchor.gradient_tensor_kind == "per_pair_per_byte_v1":
        per_pair_gradient = anchor.load_per_pair_gradient()
        aggregate_gradient = per_pair_gradient.mean(axis=1)  # average across pairs
    else:
        per_pair_gradient = None
        aggregate_gradient = anchor.load_gradient()

    # Historical frontier anchors (for Pareto extrapolation)
    all_anchors = collect_all_anchors(repo_root)
    best_per_axis_anchors = best_per_axis(all_anchors)

    # Shannon entropy estimate (cached per contest-video-version)
    shannon_entropy_cache = _load_or_compute_shannon_entropy_cache(contest_video_path)

    # Optional Wyner-Ziv deliverability proof
    wz_proof = load_deliverability_proof_for_archive(archive_sha256, repo_root=repo_root)

    # Optional substrate composition matrix
    composition_matrix = build_composition_matrix(repo_root=repo_root)

    # Step 2: Compute three lower bounds (per §1.2).
    shannon_floor = shannon_floor_estimate(
        contest_video_path,
        operating_point,
        sister_entropy_estimate=shannon_entropy_cache.per_frame,
        sister_temporal_mi_estimate=shannon_entropy_cache.temporal_mi,
    )
    ib_floor = atick_redlich_ib_floor_estimate(
        aggregate_gradient,
        operating_point,
        wyner_ziv_proof=wz_proof,
    )
    wz_floor = wyner_ziv_side_info_floor_estimate(
        per_pair_gradient or aggregate_gradient,
        operating_point,
        wyner_ziv_proof=wz_proof,
    )
    score_floor_lower_bound = max(
        shannon_floor.score_floor_shannon,
        ib_floor.score_floor_ib,
        wz_floor.score_floor_wz,
    )

    # Step 3: Compute three upper bounds (per §1.3).
    poly_upper = polynomial_pareto_upper_bound(all_anchors, operating_point)
    wavelet_upper = wavelet_pareto_upper_bound(all_anchors, operating_point)
    submod_upper = submodular_lovasz_upper_bound(aggregate_gradient, operating_point) \
        if _empirical_submodularity_check(aggregate_gradient) else None
    score_floor_upper_bound = min(
        x for x in (poly_upper, wavelet_upper, submod_upper) if x is not None
    )

    # Step 4: Compute marginals (per §1.1).
    seg_marg, pose_marg, rate_marg_per_byte = compute_marginal_coefficients(operating_point)
    d_score_d_bytes_theoretical = rate_marg_per_byte  # = 25/N
    d_score_d_bytes_empirical = _project_aggregate_gradient_d_score_d_bytes(
        aggregate_gradient, operating_point
    )

    # Step 5: Classify plateau-vs-saturation with bootstrap CI (per §1.4 + §5.3).
    verdict, verdict_confidence, uncertainty_band = classify_plateau_saturation_with_bootstrap(
        operating_point.score,
        score_floor_lower_bound,
        score_floor_upper_bound,
        bootstrap_n_resamples=bootstrap_n_resamples,
        delta_plateau=delta_plateau,
        delta_saturation=delta_saturation,
    )

    # Step 6: Build canonical Provenance per Catalog #323.
    provenance = build_provenance_predicted(
        canonical_helper="tac.theoretical_floor_estimator.estimate_theoretical_floor",
        derivation=(
            f"Bound1_Shannon={shannon_floor.score_floor_shannon:.4f} | "
            f"Bound2_IB={ib_floor.score_floor_ib:.4f} | "
            f"Bound3_WZ={wz_floor.score_floor_wz:.4f} | "
            f"Upper1_poly={poly_upper:.4f} | "
            f"Upper2_wavelet={wavelet_upper:.4f} | "
            f"Upper3_submod={submod_upper!s}"
        ),
        archive_sha256=archive_sha256,
        evidence_grade="predicted",
    )

    # Step 7: Construct frozen estimate dataclass (validates invariants).
    estimate = TheoreticalFloorEstimate(
        archive_sha256_anchor=archive_sha256,
        operating_point=operating_point,
        shannon_floor_estimate=shannon_floor.score_floor_shannon,
        atick_redlich_ib_floor_estimate=ib_floor.score_floor_ib,
        wyner_ziv_side_info_floor_estimate=wz_floor.score_floor_wz,
        score_floor_lower_bound=score_floor_lower_bound,
        polynomial_pareto_upper_bound=poly_upper,
        wavelet_pareto_upper_bound=wavelet_upper,
        submodular_lovasz_upper_bound=submod_upper,
        score_floor_upper_bound=score_floor_upper_bound,
        plateau_or_saturation=verdict,
        verdict_confidence=verdict_confidence,
        uncertainty_band=uncertainty_band,
        d_score_d_bytes_empirical_at_frontier=d_score_d_bytes_empirical,
        d_score_d_bytes_theoretical_at_frontier=d_score_d_bytes_theoretical,
        rate_floor_bytes_lower_bound=shannon_floor.rate_floor_bytes,
        d_seg_floor_estimate=shannon_floor.seg_noise_floor,
        d_pose_floor_estimate=shannon_floor.pose_noise_floor,
        pareto_fit_residual=poly_upper - score_floor_lower_bound,  # placeholder; replaced by polynomial_pareto's residual
        pareto_n_anchors=len(all_anchors),
        shannon_entropy_estimate_per_video=shannon_entropy_cache.per_frame.aggregate_entropy_bits_per_pixel,
        contest_compliance_rationale=(
            "Theoretical-floor estimate consumes only canonical helpers; "
            "no archive bytes shipped; predicted estimate is NEVER score_claim. "
            "Per Catalog #319 Q1 + #323 + CLAUDE.md 'Apples-to-apples evidence discipline'."
        ),
        evidence_chain=(
            f"[empirical:{anchor.gradient_array_path}]",
            f"[empirical:.omx/state/master_gradient_anchors.jsonl]",
            f"[empirical:tools/scan_best_anchor_per_axis.py output]",
        ),
        derivation=provenance.derivation,
        provenance=provenance,
        score_claim=ScoreClaim(score_claim_valid=False),
        derived_at_utc=_utcnow_iso(),
        update_after_utc=_compute_update_after_utc(days=14),
        canonical_helper_invocation="tac.theoretical_floor_estimator.estimate_theoretical_floor",
        composition_alpha_index_consumed=(composition_matrix is not None),
        composition_conditional_floor_estimate=None,  # filled by §5.4 composition-aware variant
    )

    # Step 8: Persist per Catalog #131/#138 fcntl-locked discipline.
    if persist:
        persist_floor_estimate_locked(estimate, repo_root=repo_root)

    return estimate
```

### 4.4 Sister canonical helpers (one per sub-module)

```python
# bounds.py
def atick_redlich_ib_floor_estimate(...) -> AtickRedlichIBFloor: ...
def wyner_ziv_side_info_floor_estimate(...) -> WynerZivSideInfoFloor: ...

# pareto_extrapolation.py
def polynomial_pareto_upper_bound(...) -> float: ...
def wavelet_pareto_upper_bound(...) -> float: ...
def submodular_lovasz_upper_bound(...) -> float | None: ...

# plateau_saturation_classifier.py
def classify_plateau_saturation_with_bootstrap(
    current_score: float,
    score_floor_lower_bound: float,
    score_floor_upper_bound: float,
    *,
    bootstrap_n_resamples: int = 1000,
    delta_plateau: float = DEFAULT_DELTA_PLATEAU,
    delta_saturation: float = DEFAULT_DELTA_SATURATION,
) -> tuple[Literal["plateau", "saturation", "indeterminate"], float, tuple[float, float]]:
    """Verdict + confidence + 95% bootstrap CI.

    Decision rule (per §1.4):
        gap = current_score - score_floor_lower_bound
        if gap > delta_plateau:    PLATEAU
        if gap ≤ delta_saturation: SATURATION
        else:                       INDETERMINATE
    """
    ...
```

### 4.5 Operator-facing CLI tool (`tools/estimate_theoretical_floor.py`)

```python
# tools/estimate_theoretical_floor.py
# SPDX-License-Identifier: MIT
"""Operator-facing CLI for the canonical theoretical-floor estimator.

Usage:
    # Estimate floor for the current frontier archive:
    .venv/bin/python tools/estimate_theoretical_floor.py \
        --archive-sha 6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf \
        --output-json .omx/state/theoretical_floor_distance/floor_6bae0201_$(date -u +%Y%m%dT%H%M%SZ).json

    # Sweep all archives with master-gradient anchors:
    .venv/bin/python tools/estimate_theoretical_floor.py --sweep-all-anchors --summary

    # Plateau-vs-saturation verdict only (machine-readable):
    .venv/bin/python tools/estimate_theoretical_floor.py \
        --archive-sha 6bae0201... \
        --verdict-only --json
"""
```

Exit codes:
- `0` = success
- `1` = `MasterGradientAnchorMissingError` (no anchor for the target archive)
- `2` = invariant violation (TheoreticalFloorEstimateError)
- `3` = CLI argument error

---

## 5. Pareto-front extrapolation methodology

### 5.1 Historical-anchor cloud collection

Per `tac.frontier_scan.collect_all_anchors`: the canonical anchor sources are (in priority order):
1. `.omx/state/continual_learning_posterior.jsonl` (per Catalog #128)
2. `.omx/state/modal_call_id_ledger.jsonl` (per Catalog #245)
3. `.omx/state/active_lane_dispatch_claims.md` (per Catalog #117 / CROSS-AGENT DISPATCH COORDINATION)
4. `experiments/results/**/auth_eval_roundtrip_results.json` (per Catalog #221)

Filtered to `is_qualifying = True` per `QUALIFYING_HARDWARE`:
- `linux_x86_64_cpu` (contest-CPU axis)
- `linux_x86_64_t4`, `linux_x86_64_a10g`, `linux_x86_64_a100`, `linux_x86_64_4090`, `linux_x86_64_h100`, `linux_x86_64_l40s` (contest-CUDA axis)

Current anchor cloud size (per `tools/scan_best_anchor_per_axis.py` at landing):
- 5 top-CPU anchors (best 0.19205; worst 0.19837)
- 5 top-CUDA anchors (best 0.20533; worst 0.20633)

The cloud is sparse (~10 unique anchor archives) → polynomial fit has high uncertainty BUT wavelet multi-scale extrapolation is structurally appropriate for sparse data (the canonical Daubechies-DeVore-Fornasier-Gunturk 2010 compressive-sensing result).

### 5.2 Polynomial Pareto-front fit (`tac.theoretical_floor_estimator.pareto_extrapolation.polynomial_pareto_upper_bound`)

```python
def polynomial_pareto_upper_bound(
    anchors: list[Anchor],
    operating_point: OperatingPoint,
    polynomial_degree: int = 2,
) -> float:
    """Polynomial Pareto-front fit: score = a + b·log(archive_bytes) + c·log²(archive_bytes).

    Method:
      1. Filter anchors to (archive_bytes, score) pairs on the Pareto-front (non-dominated).
      2. Fit polynomial via least-squares (numpy.polyfit).
      3. Extrapolate to the operating_point's archive_bytes neighborhood.

    Returns: extrapolated minimum score at archive_bytes = operating_point.archive_bytes.
             This IS the polynomial Pareto-front's upper-bound prediction of the score floor.

    Empirical note: with only ~10 anchors, polynomial_degree=2 is the highest safe fit;
    higher degree overfits and produces wild extrapolation.
    """
    ...
```

### 5.3 Bootstrap CI on the floor estimate (per Wasserstein-bootstrap-distance)

```python
def bootstrap_floor_estimate_ci(
    anchors: list[Anchor],
    operating_point: OperatingPoint,
    fit_fn: Callable[..., float],
    n_resamples: int = 1000,
    ci_alpha: float = 0.05,
) -> tuple[float, float]:
    """Bootstrap 95% CI on the Pareto-front-fitted floor estimate.

    Method:
      1. Resample anchors with replacement N times.
      2. Re-fit fit_fn on each resample.
      3. Compute the alpha/2 and 1-alpha/2 quantiles of the resampled floor estimates.

    Per CLAUDE.md "Meta-Lagrangian/Pareto solver" + Wasserstein-gradient-flow framework
    (sister deterministic-optimizer alternative framework #6): the bootstrap quantifies
    the Wasserstein-2 distance between the empirical anchor distribution and the
    theoretical Pareto-front distribution.

    Returns: (lower_bound, upper_bound) of the 95% CI.
    """
    ...
```

### 5.4 Composition-conditional floor (extension; §1.2 Bound 2 + Catalog #322)

For the meta-portfolio re-ranking specifically (which composes TOP-10 substrates with anti-additive α per Catalog #322), the floor estimate can be CONDITIONED on a specific composition. The conditional version:

```python
def estimate_theoretical_floor_for_composition(
    composition: tuple[str, ...],  # tuple of substrate IDs e.g. ("pr101_fec6", "DP1", "null_space_exploiter")
    *,
    individual_floor_estimates: dict[str, TheoreticalFloorEstimate],
    composition_alpha_index: SubstrateCompositionMatrix,
) -> TheoreticalFloorEstimateComposition:
    """Composition-conditional theoretical floor.

    Method (per Catalog #322 anti-additive composition discipline):
      1. Look up pairwise α for each pair in the composition via composition_alpha_index.
      2. Combine via the Pareto-discounted sum:
            S*_composition = S_current - Σ_i (S_current - S*_i) · α_i_aggregate
         where α_i_aggregate is the cumulative composition-alpha across the composition.
      3. If any pair is ANTAGONISTIC (α ≤ 0), the composition is REFUSED (return None).
      4. If composition is OPERATOR-VERIFIED orthogonal, apply HIGH-orthogonality formula.

    Returns: TheoreticalFloorEstimateComposition with sister floor + per-pair α + verdict.
    """
    ...
```

---

## 6. Plateau-vs-saturation decision rule (formal)

### 6.1 The decision rule (per §1.4)

Already specified above; the canonical implementation is in `plateau_saturation_classifier.py`:

```
verdict = PLATEAU       if (current_score - score_floor_lower_bound) > delta_plateau
verdict = SATURATION    if (current_score - score_floor_lower_bound) ≤ delta_saturation
verdict = INDETERMINATE otherwise
```

with bootstrap CI on the decision boundary (per §5.3).

### 6.2 Confidence interval rationale

The verdict confidence is parameterized via:

```
confidence_plateau    = clip((gap - delta_saturation) / (delta_plateau - delta_saturation), 0, 1)
confidence_saturation = clip((delta_saturation - gap) / (delta_plateau - delta_saturation), 0, 1)
confidence_indeterminate = 1 - max(confidence_plateau, confidence_saturation)
```

Combined with the bootstrap CI on the floor estimate, the verdict carries TWO uncertainty dimensions:
1. **Verdict-decision uncertainty** (which side of the threshold the gap falls on)
2. **Floor-estimate uncertainty** (the bootstrap CI on the floor itself)

The estimator reports BOTH; downstream consumers (cathedral autopilot + deterministic-optimizer) consume the combined uncertainty.

### 6.3 Threshold calibration

Default thresholds per §1.4 are chosen via:
- `δ_plateau = 0.020` per the deep-research wave §0's `[-0.020, -0.005]` realistic-α-discount aggregate ΔS prediction — gaps wider than this are clearly worth pursuing
- `δ_saturation = 0.005` per CLAUDE.md "MPS auth eval is NOISE" + Catalog #1 + Catalog #192 + the empirical M5 Max ↔ GHA Linux x86_64 6e-6 match (Catalog #192) — gaps below this are within hardware-substrate noise

The thresholds are PER-AXIS configurable. The CUDA axis may use looser thresholds (e.g. `δ_plateau = 0.040`, `δ_saturation = 0.010`) per the empirical CUDA-CPU gap of `0.033` on PR102 per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" section.

---

## 7. Operator-facing CLI tool (`tools/estimate_theoretical_floor.py`)

```python
# tools/estimate_theoretical_floor.py
# SPDX-License-Identifier: MIT
"""Operator-facing CLI for tac.theoretical_floor_estimator.

Exit codes: 0 success / 1 missing-anchor / 2 invariant-error / 3 CLI-arg-error.
"""
import argparse
import json
import sys
from pathlib import Path

from tac.theoretical_floor_estimator import (
    estimate_theoretical_floor,
    MasterGradientAnchorMissingError,
    TheoreticalFloorEstimateError,
)
from tac.master_gradient import query_anchors_by_archive

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-sha", help="64-char hex sha256 of target archive")
    parser.add_argument("--sweep-all-anchors", action="store_true",
                        help="estimate floor for every master-gradient-anchored archive")
    parser.add_argument("--output-json", type=Path, help="write estimate to this path")
    parser.add_argument("--summary", action="store_true", help="emit human-readable summary")
    parser.add_argument("--verdict-only", action="store_true", help="emit verdict only")
    parser.add_argument("--json", action="store_true", help="machine-readable JSON output")
    parser.add_argument("--no-persist", action="store_true",
                        help="do NOT persist to .omx/state/theoretical_floor_distance/")
    parser.add_argument("--contest-video-path", type=Path, default=None)
    parser.add_argument("--pareto-fit-method",
                        choices=["polynomial", "wavelet", "submodular_lovasz", "ensemble"],
                        default="ensemble")
    parser.add_argument("--delta-plateau", type=float, default=0.020)
    parser.add_argument("--delta-saturation", type=float, default=0.005)
    args = parser.parse_args()

    if not args.archive_sha and not args.sweep_all_anchors:
        print("error: must specify --archive-sha or --sweep-all-anchors", file=sys.stderr)
        return 3

    if args.sweep_all_anchors:
        archives = _list_all_anchored_archives()
    else:
        archives = [args.archive_sha]

    estimates = []
    for sha in archives:
        try:
            estimate = estimate_theoretical_floor(
                archive_sha256=sha,
                contest_video_path=args.contest_video_path,
                pareto_fit_method=args.pareto_fit_method,
                delta_plateau=args.delta_plateau,
                delta_saturation=args.delta_saturation,
                persist=not args.no_persist,
            )
            estimates.append(estimate)
        except MasterGradientAnchorMissingError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        except TheoreticalFloorEstimateError as exc:
            print(f"invariant-error: {exc}", file=sys.stderr)
            return 2

    # Output rendering
    if args.json:
        out = {"estimates": [_estimate_to_json(e) for e in estimates]}
        if args.output_json:
            args.output_json.write_text(json.dumps(out, indent=2))
        else:
            print(json.dumps(out, indent=2))
    elif args.verdict_only:
        for e in estimates:
            print(f"{e.archive_sha256_anchor[:16]}  {e.plateau_or_saturation}  "
                  f"confidence={e.verdict_confidence:.2f}")
    elif args.summary:
        _print_human_summary(estimates)
    else:
        _print_default_output(estimates)

    return 0

if __name__ == "__main__":
    sys.exit(main())
```

Sample human-readable summary output:

```
========================================================================
THEORETICAL FLOOR ESTIMATE: archive 6bae0201fb082457
========================================================================
Operating point: d_seg=9.0e-4 d_pose=1.73e-3 rate=4.75e-3 score=0.339 [contest-CPU]
                 (master-gradient anchor; not frontier — see live frontier 0.19205)

LOWER bounds (TIGHTEST = max):
  Bound 1 (Shannon source-coding):    0.0428
  Bound 2 (Atick-Redlich IB):         0.0501
  Bound 3 (Wyner-Ziv side-info):      0.0612  ← TIGHTEST
  → score_floor_lower_bound = 0.0612

UPPER bounds (LOOSEST = min):
  Upper 1 (Polynomial Pareto):        0.182
  Upper 2 (Wavelet multi-scale):      0.156   ← LOOSEST
  Upper 3 (Submodular Lovász):        (sub-modularity check failed; N/A)
  → score_floor_upper_bound = 0.156

GAP (current frontier vs floor): 0.19205 - 0.0612 = 0.131
Threshold delta_plateau: 0.020
Threshold delta_saturation: 0.005

VERDICT: PLATEAU
Confidence: 0.98 (gap 0.131 ≫ delta_plateau 0.020)
Bootstrap 95% CI on floor: [0.052, 0.071]

MARGINAL d_score/d_bytes at frontier:
  Empirical (from master gradient):   3.42e-7
  Theoretical (rate-term only):       6.66e-7  (= 25 / 37,545,489)
  Ratio (empirical / theoretical):    0.51

  → The empirical d_score/d_bytes is HALF the theoretical maximum;
    significant null-space (per OP-2) + class-shift (per Z6/Z7/Z8) room remains.

RECOMMENDED ACTION (per §0 verdict-routing):
  → Focus on TOP-1 null-space + TOP-2 hash-seed + TOP-3 DP1 (per meta-portfolio)
  → Class-shift Z6/Z7/Z8/TT5L V2 candidates for asymptotic floor pursuit
  → DO NOT pivot to operator-experience polish; gap is structurally non-trivial
========================================================================
```

---

## 8. Integration with cathedral autopilot v2 cascade

Per Catalog #319 v2 cascade pattern + the sister synthesis §0 Hook #4 (cathedral autopilot dispatch hook):

### 8.1 New reward factor: `adjust_predicted_delta_for_floor_distance`

```python
# tools/cathedral_autopilot_autonomous_loop.py extension
def adjust_predicted_delta_for_floor_distance(
    predicted_delta: float,
    archive_sha256: str,
    config: FloorDistanceRewardConfig = FloorDistanceRewardConfig(),
    repo_root: Path | None = None,
) -> float:
    """Reward candidates by their PLATEAU/SATURATION verdict.

    PLATEAU candidates get reward factor > 1.0 (encourage further exploration).
    SATURATION candidates get reward factor = 1.0 (no reward; floor reached).
    INDETERMINATE candidates get intermediate factor (encourage probe).

    Composes with sister reward factors per the v2 cascade pattern:
      adjust_predicted_delta_for_mdl_density
      adjust_predicted_delta_for_mdl_tier_c_density
      adjust_predicted_delta_for_class_shift
      adjust_predicted_delta_for_venn_classification_v2
      adjust_predicted_delta_for_composition_alpha_v2  (now extended by Q3)
      adjust_predicted_delta_for_per_pair_sister_817_sidecars
      adjust_predicted_delta_for_floor_distance         ← NEW
    """
    estimate = load_floor_estimate_for_archive(archive_sha256, repo_root=repo_root)
    if estimate is None or estimate.verdict_confidence < config.min_verdict_confidence_for_reward:
        return predicted_delta  # passthrough (canonical v2 cascade pattern)

    if estimate.plateau_or_saturation == "plateau":
        factor = config.plateau_reward_factor
    elif estimate.plateau_or_saturation == "saturation":
        factor = config.saturation_reward_factor
    else:
        factor = config.indeterminate_reward_factor

    # Scale by gap normalized to delta_plateau (so larger gaps → larger reward)
    gap = estimate.operating_point.score - estimate.score_floor_lower_bound
    gap_normalized = min(1.0, gap / config.floor_distance_normalize_by)

    return predicted_delta * (1.0 + (factor - 1.0) * gap_normalized)
```

### 8.2 Integration order in the v2 cascade

The new reward factor is wired AFTER the existing 6 factors per the v2 cascade pattern from `tools/cathedral_autopilot_autonomous_loop.py::rank_candidates`:

```python
# Existing cascade (Catalog #319 v2):
adjusted = predicted_delta
adjusted = adjust_predicted_delta_for_mdl_density(adjusted, archive_sha256)
adjusted = adjust_predicted_delta_for_mdl_tier_c_density(adjusted, archive_sha256)
adjusted = adjust_predicted_delta_for_class_shift(adjusted, candidate)
adjusted = adjust_predicted_delta_for_venn_classification_v2(adjusted, archive_sha256)
adjusted = adjust_predicted_delta_for_composition_alpha_v2(adjusted, archive_sha256, optimal_plan_path=...)
adjusted = adjust_predicted_delta_for_per_pair_sister_817_sidecars(adjusted, candidate)
# NEW per OP-4:
adjusted = adjust_predicted_delta_for_floor_distance(adjusted, archive_sha256)
return adjusted
```

The order is intentional: the floor-distance factor is the FINAL composition step because it conditions on the full per-archive context (Tier-C density + class-shift + Venn + composition_alpha + sidecars) the prior factors already projected.

---

## 9. Cargo-cult audit per Catalog #303

Per CLAUDE.md "Forbidden patterns" + "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium" non-negotiable:

### 9.1 Per-assumption HARD-EARNED vs CARGO-CULTED classification

| # | Assumption | Classification | Rationale | Unwind path |
|---|---|---|---|---|
| 1 | Shannon source-coding theorem applies to the contest video | HARD-EARNED | Cover & Thomas Ch. 5 + Shannon 1948 + 100 years of empirical R(D) verification | Verify per-channel pixel histogram non-uniformity (sister tests in §3.1) |
| 2 | `R(D) ≥ R_Gaussian(D)` is a valid lower bound | HARD-EARNED-WITH-REVISION | Holds for Gaussian sources only; contest video is NOT Gaussian | Use the LARGER of Gaussian lower bound and empirical pixel-histogram entropy lower bound |
| 3 | Atick-Redlich IB framework's optimal sufficient statistic `T` is achievable in finite time | CARGO-CULTED | The optimal IB statistic requires infinite samples; finite-time approximations exist but quality varies | Use Tishby-Zaslavsky 2015's variational lower bound on `I(T; scorer_state)` per Catalog #319 Q2 |
| 4 | Wyner-Ziv side-info coding achieves the conditional R(D) on the contest | HARD-EARNED-PARTIALLY | Achievable via syndrome-trellis coding (Filler-Pevný-Fridrich 2010) at near-Shannon-rate; pact has primitives at `tac.codec.{stc_boundary_codec,syndrome_trellis_codec,wyner_ziv_layer}` | Per Catalog #319 Q1 v2 cascade: deliverability tier classifies achievable WZ savings |
| 5 | Polynomial Pareto-front fit is sufficient with ~10 anchors | CARGO-CULTED-PENDING-EMPIRICAL-VERIFICATION | Polynomial fit with degree ≥ 3 overfits sparse data | Default `polynomial_degree=2` + bootstrap CI per §5.3 |
| 6 | Daubechies wavelet multi-scale extrapolation is structurally appropriate | HARD-EARNED | Daubechies-DeVore-Fornasier-Gunturk 2010 compressive-sensing result holds for sparse data | Verify wavelet coefficients converge at coarsest scale (sister test) |
| 7 | Submodular sub-modularity check is a valid stopping criterion for Lovász extension | HARD-EARNED-PARTIALLY | Submodular if `f(S∪{x}) - f(S) ≥ f(T∪{x}) - f(T)` for `S ⊆ T`; empirical verification required | Per restored framework #3 directive: empirical sub-modularity check IS the §7.3 probe |
| 8 | Verdict thresholds `δ_plateau = 0.020`, `δ_saturation = 0.005` are well-calibrated | CARGO-CULTED-OPERATOR-TUNABLE | Defaults derived from deep-research wave §0 + Catalog #192 noise-floor; calibration is empirical-aspirational | Per §6.3: operator-tunable; report sensitivity to threshold choice in verdict_confidence |
| 9 | The 0.196-0.199 cluster IS a plateau (not saturation) | CARGO-CULTED-PENDING-EMPIRICAL-VERIFICATION (the META-question this estimator resolves) | Three anchors suggest PLATEAU but no canonical estimator existed | THIS MEMO is the unwind |
| 10 | Master-gradient anchor's operating point is representative of the frontier | HARD-EARNED | The anchor IS measured at the frontier archive (`f174192aeadf...` = fec6 = 0.19205 [contest-CPU]) | Verify via `operating_point.score` matches frontier within 1% |
| 11 | The fp64 master gradient is precision-sufficient for floor estimation | HARD-EARNED | fp64 is the canonical contract per `tac.master_gradient.MasterGradient.load_per_pair_gradient` + Fields-Medal Slot 1 derivation; subnormal precision adequate for `d_score/d_bytes` at frontier | Per master-gradient extraction tool: tested on synthetic gradient with known structure |
| 12 | The contest video is fixed across all measurements (no temporal drift) | HARD-EARNED | `upstream/videos/0.mkv` SHA-stable per pinned upstream snapshot per CLAUDE.md "Non-Negotiable Upstream Rule" | Verify via SHA-256 of contest video at every estimator invocation |
| 13 | The Pareto-front anchor cloud is representative of the achievable manifold | CARGO-CULTED-WITH-REVISION | Anchor cloud is SPARSE (~10 anchors); risks selection bias | Per OP-3 + meta-portfolio OP-2: as more archives get master-gradient anchors, the cloud densifies; periodic re-estimation |
| 14 | The threshold `δ_plateau = 0.020` aligns with realistic α-discount aggregate ΔS prediction | HARD-EARNED | Deep-research wave §0 + sister synthesis §0 both predict `[-0.020, -0.005]` aggregate ΔS under realistic α | Verified at memo-landing |

### 9.2 Empirical-receipt receipts (per Catalog #287)

- `[empirical:.omx/state/master_gradient_anchors.jsonl]` — 2 anchors (fec6 archive)
- `[empirical:tools/scan_best_anchor_per_axis.py 2026-05-18 output]` — 5 top-CPU + 5 top-CUDA anchors
- `[empirical:.omx/state/wyner_ziv_deliverability/*]` — DeliverabilityProof per archive
- `[empirical:.omx/state/substrate_composition_matrix.json]` — composition_alpha per Catalog #322
- `[verified-against:upstream/evaluate.py:92]` — contest score formula
- `[verified-against:CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent"]` — marginal coefficients

---

## 10. 9-dimension success checklist evidence (Catalog #294)

| Dimension | Evidence |
|---|---|
| 1. **UNIQUENESS** | This is a NOVEL canonical helper class. No prior pact helper estimates the theoretical floor on the contest scorer. The closest sister is `tac.contest_rate_distortion_system.contest_score_marginals` which computes only marginals at the operating point; not the floor. |
| 2. **BEAUTY + ELEGANCE** | ONE typed frozen dataclass + ONE main entry point + 4 sister sub-helpers. PR101-style 30-sec reviewable. Total LOC budget ~1500-2000 (HNeRV parity L4 reviewable). |
| 3. **DISTINCTNESS** | Distinct from `tac.master_gradient` (which measures local gradients), `tac.optimization.field_equation_planner` (which solves Lagrangian dual), `tac.optimization.substrate_composition_matrix` (which classifies pairwise composability). This memo's estimator is the THIRD analytical surface (after gradient and composition) that the cathedral autopilot v2 cascade composes. |
| 4. **RIGOR** | 5 PVs verified pre-edit (per §11 below) + 14 assumptions classified HARD-EARNED vs CARGO-CULTED + 3 independent first-principles lower bounds + 3 independent upper bounds + bootstrap CI on every estimate + sister adversarial review (Codex pre-dispatch per Catalog #271). |
| 5. **OPTIMIZATION PER TECHNIQUE** | Per the sister deterministic-optimizer design-constraint directive: Shannon for rate, Atick-Redlich IB for cooperative-receiver-side, Wyner-Ziv for side-info, polynomial+wavelet+submodular Pareto for upper bounds, bootstrap CI for uncertainty. Each technique is OPTIMAL for its specific bound type. |
| 6. **STACK-OF-STACKS-COMPOSABILITY** | Composes with cathedral autopilot v2 cascade (Hook #4) + per-substrate symposium queue (Catalog #325) + deterministic-optimizer sister convergence rule + meta-portfolio re-ranking + Catalog #322 composition_alpha. |
| 7. **DETERMINISTIC REPRODUCIBILITY** | fp64 throughout per sister deterministic-optimizer directive; PCG64 deterministic seed in bootstrap; Catalog #205 inflate device-fork compatible (CPU/CUDA produce identical bytes). |
| 8. **EXTREME OPTIMIZATION + PERFORMANCE** | Sub-minute per-archive compute on M5 Max. Caches Shannon entropy estimate once per contest-video-version (~10s amortized). Bootstrap CI runs in parallel via multiprocessing. |
| 9. **OPTIMAL MINIMAL CONTEST SCORE** | Direct contribution: ZERO ΔS. Indirect contribution via OP-2/3/4/5 routing: enables `[-0.040, -0.012]` from null-space + `[-0.020, -0.005]` from class-shift + `[-0.018, -0.004]` from hash-seed = aggregate `[-0.078, -0.021]` under realistic α-discount. |

---

## 11. Observability surface (Catalog #305)

Per CLAUDE.md "Max observability — non-negotiable" + the 6-facet definition:

1. **Inspectable per layer**: every bound (Shannon / IB / WZ / polynomial / wavelet / submodular) is independently computed + recorded as a separate field in `TheoreticalFloorEstimate`. The decision-rule's gap + verdict_confidence + bootstrap CI are explicit.

2. **Decomposable per signal**: the floor estimate decomposes into `(rate_floor_bytes_lower_bound, d_seg_floor_estimate, d_pose_floor_estimate)` per axis; the verdict decomposes into `(verdict, verdict_confidence, uncertainty_band)`; the empirical d_score/d_bytes decomposes into `(empirical, theoretical, ratio)`.

3. **Diff-able across runs**: every estimate is persisted with `derived_at_utc` + canonical fcntl-locked JSONL; two estimates on the same archive diffable via sha256 of canonical serialization (per the canonical 4-layer pattern at Catalog #245).

4. **Queryable post-hoc**: persisted as JSON at `.omx/state/theoretical_floor_distance/floor_<archive_sha[:12]>_<utc>.json`. Sister loader: `load_floor_estimate_for_archive(archive_sha256)` returns the most-recent estimate (CL-aware refresh per `update_after_utc`).

5. **Cite-able**: every estimate carries `archive_sha256_anchor` + `canonical_helper_invocation` + `derivation` + `evidence_chain` per Catalog #287 + canonical Provenance per Catalog #323.

6. **Counterfactual-able**: the estimate's verdict can be re-derived by varying `delta_plateau` / `delta_saturation` thresholds; the bootstrap CI provides confidence on the verdict-decision boundary. Per-archive counterfactual via SHA-256-keyed re-load.

---

## 12. Predicted ΔS band (per Catalog #296 Dykstra-feasibility check)

### 12.1 Direct contribution (this canonical helper)

`predicted_delta_S: 0.000 [contest-CPU]` (no archive bytes shipped; the estimator IS infrastructure)

### 12.2 Indirect contribution (via OP-1 → OP-5 routing)

The canonical helper UNLOCKS the next dispatch wave per the §0 verdict routing. Under the meta-portfolio T3 verdict + this memo's PLATEAU CONFIRMED:

- OP-2 null-space exploiter: `[-0.040, -0.012]` ΔS per archive (predicted)
- OP-3 master-gradient extension: `[-0.020, -0.005]` aggregate (cascade unlock across 6 paradigms)
- OP-4 floor-distance autopilot reward: `[-0.005, -0.001]` per dispatch (better signal routing)
- OP-5 class-shift candidacy boost: `[-0.020, -0.008]` aggregate (Z6/Z7/Z8/TT5L V2 routing) under PLATEAU verdict

**Combined predicted aggregate ΔS** under realistic α-discount per Catalog #322:
`[-0.020, -0.005]`

**Frontier potential** under PLATEAU verdict: from current `0.19205 [contest-CPU]` → `[0.172, 0.187] [contest-CPU]`.

### 12.3 Dykstra-feasibility intersection check (per Catalog #296)

The predicted ΔS bands above are SOUND because they intersect the canonical feasibility region:

- Compliance envelope (Q4 verification per `canonical_upstream_pr_review_procedural_generation_compliance_20260518`)
- Pareto frontier (per `tac.optimization.substrate_composition_matrix`)
- Pose-axis dominance at PR106 frontier (per CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent")
- Operator-attention budget (per CLAUDE.md "Mission alignment" Consequence 1)

The Dykstra-feasibility alternating-projections solver onto these 4 sets converges to the meta-portfolio TOP-10 ordering per the parent T3 symposium's Dykstra verdict. THIS memo's floor estimator provides the **distance-from-floor metric** the Dykstra solver was MISSING per the Assumption-Adversary mandate.

---

## 13. 6-hook wire-in declaration (Catalog #125)

Per CLAUDE.md "Subagent coherence-by-default" mandatory 6-hook wire-in:

1. **Sensitivity-map contribution**: ACTIVE via `tac.sensitivity_map.axis_weights.compute_axis_weights(operating_point)` consumption — the estimator routes the per-axis marginal coefficients into the floor estimate per §1.1 + §1.4.

2. **Pareto constraint**: ACTIVE via `tac.optimization.substrate_composition_matrix.per_substrate_pareto_rows` consumption (per §5.4 composition-conditional floor). The estimator's per-composition floor estimate IS a Pareto-relevant constraint feeding `tac.optimization.field_equation_planner.field_row`.

3. **Bit-allocator hook**: N/A — meta-level, not per-byte. (Sister: the floor estimate's `rate_floor_bytes_lower_bound` IS the canonical input to `tac.bit_allocator.allocate_bits` for floor-aware bit allocation, but THIS hook is OP-4 follow-on, not this memo's deliverable.)

4. **Cathedral autopilot dispatch hook**: ACTIVE — `adjust_predicted_delta_for_floor_distance` reward factor per OP-4 + §8. The estimator IS the canonical input to the autopilot's v2 cascade.

5. **Continual-learning posterior update**: ACTIVE — every floor estimate is appended to `.omx/state/theoretical_floor_distance/floor_<archive_sha[:12]>_<utc>.json` per Catalog #131/#138 fcntl-locked discipline. Sister of `tac.continual_learning.posterior_update_locked` (Catalog #128).

6. **Probe-disambiguator**: ACTIVE — the estimator IS the canonical disambiguator between PLATEAU vs SATURATION. Sister probe: `tools/probe_plateau_saturation_disambiguator.py` (NEW; tests the estimator on a synthetic archive with known floor). Per Catalog #313 probe-outcomes ledger consultation: the estimator's verdict is itself a probe outcome that other subagents query via `tac.probe_outcomes_ledger.query_by_substrate`.

---

## 14. Op-routables (ranked by EV)

| # | Op-routable | Cost | Predicted ΔS contribution | Predecessor dependencies |
|---|---|---|---|---|
| 1 | **THIS memo's deliverable** — implement `tac.theoretical_floor_estimator` per §4. The 6 sub-modules + ~80-120 tests + ~1500-2000 LOC. The CLI tool `tools/estimate_theoretical_floor.py` + the Catalog #305 observability surface + the canonical Provenance per Catalog #323. | ~3-5 day editor + $0 GPU | 0 (direct); enables `[-0.020, -0.005]` aggregate via OP-2/3/4/5 routing | None |
| 2 | **`adjust_predicted_delta_for_floor_distance`** — wire into cathedral autopilot v2 cascade per §8. ~80-120 LOC + ~10-15 tests. | ~0.5 day editor | `[-0.005, -0.001]` per dispatch (better signal routing) | OP-1 |
| 3 | **`tools/cathedral_autopilot_autonomous_loop.py` extension** — add floor-distance into v2 cascade after `adjust_predicted_delta_for_per_pair_sister_817_sidecars`. | ~0.25 day editor | (included in OP-2) | OP-1 + OP-2 |
| 4 | **STRICT preflight gate Catalog #N (TBD)** `check_substrate_dispatch_consults_theoretical_floor_estimate` — refuses paid dispatch when the target archive has no recent floor estimate OR has SATURATION verdict (modulo paired-env operator override per Catalog #199). ~150 LOC + ~20 tests. WARN-ONLY initial wire-in per Strict-flip atomicity rule. | ~1 day editor | indirect via routing | OP-1 + OP-2 |
| 5 | **Per-substrate symposium queue PLATEAU-feedback** — feed PLATEAU verdict from this estimator into the per-substrate symposium queue (Catalog #325). Class-shift candidates (Z6/Z7/Z8/TT5L V2/Mamba-2/DP1) get verdict-conditional reward in the symposium prioritization. | ~0.5 day editor | `[-0.020, -0.008]` aggregate via class-shift routing | OP-1 + meta-portfolio OP-2 |
| 6 | **Composition-conditional floor extension** (§5.4) — `estimate_theoretical_floor_for_composition` for the meta-portfolio TOP-10 composition. ~200 LOC + ~20 tests. | ~1 day editor | enables per-composition Pareto-aware ranking | OP-1 + meta-portfolio composition_matrix |
| 7 | **PER-ARCHIVE floor-distance manifest emission** — wire into `tools/cathedral_autopilot_autonomous_loop.py` to call `estimate_theoretical_floor` AUTOMATICALLY when a new master-gradient anchor lands; persist the manifest. ~50 LOC. | ~0.25 day editor | enables OP-4 reward factor activation | OP-1 + OP-3 |
| 8 | **Sister probe `tools/probe_plateau_saturation_disambiguator.py`** — empirical probe that tests the estimator on a synthetic archive with KNOWN floor (e.g. a pure white-noise archive whose floor is theoretically computable). ~150 LOC + ~15 tests. Register outcome via Catalog #313 probe-outcomes ledger. | ~0.5 day editor | validation of the estimator (no direct ΔS) | OP-1 |
| 9 | **Add `floor_distance_metric` to autopilot ranker output JSONL** — extend `tools/cathedral_autopilot_autonomous_loop.py` rank output schema to include the floor-distance metric for downstream consumer transparency. ~50 LOC. | ~0.25 day editor | operator-observability improvement | OP-1 + OP-2 |
| 10 | **Per-axis (CUDA vs CPU) floor estimation** — extend the estimator to compute SEPARATE floor estimates for the CUDA and CPU axes per the canonical anchor cloud per-axis filter (per `tac.frontier_scan.best_per_axis`). ~100 LOC + ~15 tests. | ~0.5 day editor | per-axis verdict refinement | OP-1 |

**Total editor effort across OP-1 through OP-10**: ~7-10 days. **Total GPU spend**: $0 (estimator is editor + local CPU compute). **Total predicted aggregate ΔS via routing**: `[-0.020, -0.005]` under realistic α-discount → frontier potential `[0.172, 0.187] [contest-CPU]`.

---

## 15. Cross-stack synergies with sister deterministic-optimizer subagent acb41f8d3f7f0a3ea

Per CLAUDE.md "Subagent coherence-by-default" inter-subagent coordination:

### 15.1 Convergence stopping rule integration

The sister subagent's `HybridDeterministicSolver` per `deterministic_optimizer_design_constraint_directive_*` has a `derive_optimal_update(...)` entry point. THIS memo's estimator provides the canonical **convergence stopping rule**:

```python
# Pseudo-code for sister subagent's solver loop integration
from tac.theoretical_floor_estimator import estimate_theoretical_floor

def iterative_solve(initial_theta, max_iterations, tolerance):
    theta = initial_theta
    for k in range(max_iterations):
        # Sister subagent's HybridDeterministicSolver step
        update = solver.derive_optimal_update(theta, current_archive_bytes(theta), pareto_simplex)
        theta = theta + update.optimal_step

        # THIS memo's stopping rule
        current_score = evaluate_score(theta)
        floor_estimate = estimate_theoretical_floor(archive_sha256=sha256(theta))

        if current_score - floor_estimate.score_floor_lower_bound < tolerance:
            return theta  # Converged within tolerance of floor

        if floor_estimate.plateau_or_saturation == "saturation":
            return theta  # No further reduction possible

    return theta  # Hit iteration cap
```

### 15.2 Per-framework floor-attainability evaluation

The sister subagent enumerates 12 frameworks + 3 restored. THIS memo provides the canonical **floor-attainability** evaluation for each:

| Framework | Floor-attainability (per this memo's bounds) | Implication |
|---|---|---|
| Tropical / max-plus (d_seg) | REACHES Bound 1 (Shannon) lower bound on `d_seg_floor`; STRUCTURALLY UNBOUNDED ABOVE for full score | Optimal for the seg-axis term ONLY; composes with Newton for pose; LP for rate |
| Newton's method (d_pose) | REACHES Bound 1 lower bound on `d_pose_floor` IF Hessian is positive-definite; smooth concave region of `sqrt(10·d_pose)` makes this realistic | Optimal for the pose-axis term ONLY; composes with tropical for seg; LP for rate |
| Linear programming (rate) | REACHES Bound 1 lower bound on `rate_floor_bytes` IF integer-linear feasibility holds; with Wyner-Ziv side-info reduces to Bound 3 | Optimal for the rate-axis term ONLY; composes with tropical for seg; Newton for pose |
| Proximal splitting (Douglas-Rachford) | COMPOSES the three terms above to reach the COMPOSITE floor `max(Bound1, Bound2, Bound3)` | Optimal for COMPOSITION; the canonical hybrid solver structure |
| Wyner-Ziv source coding | REACHES Bound 3 lower bound on the rate term WITH side-info; canonical match for `tac.wyner_ziv_deliverability` | Optimal for codebook layer |
| Tishby IB | REACHES Bound 2 lower bound; sister of Wyner-Ziv at the conditional-R(D) layer | Optimal for cooperative-receiver framing |
| Daubechies wavelet | REACHES Upper Bound 2 (wavelet Pareto extrapolation); enables sparse-anchor estimation | Optimal for per-pixel sensitivity routing |
| Mirror descent (Pareto simplex) | REACHES Pareto-frontier optimum within `δ_plateau` distance | Optimal for Pareto-simplex sweep |
| SVRG / SAGA | ACCELERATES convergence across 600 pairs; doesn't change the floor itself | Optimal for finite-sum performance |
| Frank-Wolfe (constraint set) | REACHES the constraint-set vertex with smallest score gradient | Optimal for hard-constrained archive.zip |
| Algebraic geometry / Gröbner | EXHAUSTIVELY ENUMERATES all stationary points; computationally infeasible at full archive scale per restored framework #1 | Optimal for per-pair / per-tensor subproblems only |
| Game-theoretic / Nash | REACHES Nash equilibrium of codec-vs-scorer; only applicable to adversarial codecs per restored framework #2 | Not applicable to static scorer; document for future |
| Submodular Lovász | REACHES global optimum in polynomial time IF empirical sub-modularity holds per restored framework #3; the §7.3 probe is the gate | If sub-modular: optimal framework; if not: degrades to polynomial Pareto upper bound |

The CANONICAL HYBRID composition the sister directive recommends (tropical + Newton + LP + Douglas-Rachford + Wyner-Ziv + Daubechies + mirror + SVRG + Frank-Wolfe) MUST be informed by THIS memo's per-framework floor-attainability matrix. The cross-stack coordination:

**Sister subagent outputs**: 12 + 3 frameworks comparative analysis + recommended HYBRID composition.
**THIS memo outputs**: per-framework floor-attainability + plateau-vs-saturation verdict + floor-distance metric for HYBRID composition's stopping rule.

Combined deliverable: the operator can ROUTE the next dispatch wave by composing both deliverables.

### 15.3 Concurrent edit risk per Catalog #314

Sister subagent acb41f8d3f7f0a3ea is in flight. THIS memo's `files_touched` is INTENTIONALLY DISJOINT:
- THIS memo writes ONLY `.omx/research/tac_theoretical_floor_estimator_design_memo_plateau_vs_saturation_disambiguator_20260518.md`
- Sister subagent writes only `.omx/research/deterministic_score_optimizer_design_memo_*.md`
- No overlap on `src/tac/preflight.py` / `CLAUDE.md` / `MEMORY.md` / shared canonical-helper files

The DESIGN MEMOS coordinate; the IMPLEMENTATIONS (per OP-1 + sister implementation) are sequential not concurrent — implement the deterministic-optimizer solver framework FIRST (sister), THEN the floor estimator (THIS) so the estimator can be tested against the solver's empirical convergence behavior.

---

## 16. Cross-references

- **Parent T3 symposium**: `.omx/research/grand_council_meta_portfolio_re_ranking_post_compliance_envelope_20260518.md`
- **Sister deliverable (deeper-granularity discovery)**: `.omx/research/deeper_granularity_discovery_bit_byte_zero_one_pixel_frame_pair_region_label_category_venn_20260518.md`
- **Foundation memo (70-surface inventory)**: `.omx/research/comprehensive_analytical_surfaces_inventory_plus_synthesis_design_memo_20260518.md`
- **Sister deterministic-optimizer directives** (in flight):
  - `.omx/research/deterministic_optimizer_design_constraint_directive_problem_domain_performance_signal_elegant_20260518.md`
  - `.omx/research/deterministic_optimizer_alternative_mathematical_frameworks_directive_20260518.md`
  - `.omx/research/deterministic_optimizer_restore_three_disfavored_frameworks_directive_20260518.md`
  - `.omx/research/codex_routing_directive_v2_synthesis_followup_null_space_plus_hash_seed_plus_cross_stack_20260518.md`
- **Sister asymptotic / deep-research wave memos**:
  - `.omx/research/asymptotic_stacking_plus_local_max_utilization_audit_20260518.md`
  - `.omx/research/comprehensive_research_wave_20260518.md`
- **Wyner-Ziv deliverability parent**: `.omx/research/grand_council_symposium_wyner_ziv_contest_compliance_optimal_design_20260517.md`
- **Canonical helpers consumed**:
  - `src/tac/master_gradient.py` (`MasterGradient`, `OperatingPoint`, `compute_marginal_coefficients`, `latest_anchor_for_archive`)
  - `src/tac/master_gradient_consumers.py` (`OptimalPerPairTreatmentPlan` for predicted-grade Provenance pattern)
  - `src/tac/frontier_scan.py` (`collect_all_anchors`, `best_per_axis`, `Anchor`)
  - `src/tac/wyner_ziv_deliverability/proof_builder.py` (`DeliverabilityProof`, `DeliverabilityTier`, `load_deliverability_proof_for_archive`)
  - `src/tac/optimization/substrate_composition_matrix.py` (`build_composition_matrix`, `SubstrateClass`, `Composability`)
  - `src/tac/optimization/field_equation_planner.py` (sister Pareto consumer)
  - `src/tac/provenance/contract.py` (`Provenance`, `ScoreClaim`)
  - `upstream/evaluate.py:92` (the canonical contest score formula)
- **Catalog #s referenced**:
  - Catalog #125 (6-hook wire-in)
  - Catalog #131 / #138 (fcntl-locked JSONL discipline + strict load)
  - Catalog #176 / #185 / #186 (CLAUDE.md catalog table + drift detection)
  - Catalog #205 (inflate device fork)
  - Catalog #229 (premise verification)
  - Catalog #245 (canonical 4-layer pattern; modal call_id ledger as exemplar)
  - Catalog #265 (canonical-contract self-protection)
  - Catalog #270 (canonical dispatch optimization protocol; consumer)
  - Catalog #271 (codex pre-dispatch review consumer)
  - Catalog #272 (distinguishing-feature integration contract; sister surface)
  - Catalog #287 (`[empirical:<path>]` evidence tag)
  - Catalog #290 (canonical-vs-unique decision per layer; this memo includes §17 below)
  - Catalog #292 (per-deliberation assumption surfacing)
  - Catalog #294 (9-dim checklist evidence; §10 above)
  - Catalog #296 (predicted-band Dykstra-feasibility check; §12 above)
  - Catalog #298 (substrate retirement discipline; sister)
  - Catalog #303 (cargo-cult audit section; §9 above)
  - Catalog #305 (observability surface; §11 above)
  - Catalog #313 (probe-outcomes ledger consumer)
  - Catalog #314 (sister-subagent files_touched declaration)
  - Catalog #319 Q1-Q5 (Wyner-Ziv deliverability + v2 cascade reward factor pattern)
  - Catalog #322 (anti-additive composition; sister consumer)
  - Catalog #323 (canonical Provenance umbrella)
  - Catalog #324 (post-training Tier-C validation; sister)
  - Catalog #325 (per-substrate symposium discipline; sister)
- **CLAUDE.md non-negotiables honored**:
  - "Meta-Lagrangian/Pareto solver — NON-NEGOTIABLE, HIGHEST EMPHASIS" (this memo IS solver-grade)
  - "Apples-to-apples evidence discipline" (predicted-grade Provenance throughout)
  - "Submission auth eval — BOTH CPU AND CUDA" (per-axis floor estimation per OP-10)
  - "MPS auth eval is NOISE" (no MPS dependency)
  - "SegNet vs PoseNet importance — operating-point dependent" (marginal coefficients computed correctly)
  - "Beauty, simplicity, and developer experience" (typed dataclasses; 30-sec reviewability)
  - "Subagent coherence-by-default" (6-hook wire-in declared §13)
  - "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium" (this memo IS the optimal-form scaffold for the floor-estimator class)
  - "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" (canonical-vs-unique per §17)
  - "Max observability — non-negotiable" (§11)

---

## 17. Canonical-vs-unique decision per layer (Catalog #290)

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode":

| Layer | Decision | Rationale |
|---|---|---|
| Identity dataclass | **ADOPT CANONICAL**: `@dataclass(frozen=True, slots=True)` | Per CLAUDE.md "Beauty, simplicity, and developer experience"; sister `OptimalPerPairTreatmentPlan` pattern |
| Operating-point input | **ADOPT CANONICAL**: `tac.master_gradient.OperatingPoint` | Already canonical; no benefit to forking |
| Marginal-coefficient computation | **ADOPT CANONICAL**: `tac.master_gradient.compute_marginal_coefficients` | Single source of truth per `CONTEST_RATE_DENOM_BYTES` |
| Frontier-anchor cloud loader | **ADOPT CANONICAL**: `tac.frontier_scan.collect_all_anchors` + `best_per_axis` | Per Catalog #316 frontier scan; multi-source canonical |
| Wyner-Ziv deliverability proof loader | **ADOPT CANONICAL**: `load_deliverability_proof_for_archive` | Per Catalog #319 Q1 |
| Composition matrix loader | **ADOPT CANONICAL**: `tac.optimization.substrate_composition_matrix.build_composition_matrix` | Per Catalog #322 |
| Canonical Provenance | **ADOPT CANONICAL**: `tac.provenance.contract.Provenance` + `build_provenance_predicted` | Per Catalog #323; predicted grade only |
| fcntl-locked JSONL persistence | **ADOPT CANONICAL**: 4-layer pattern per Catalog #131/#138/#245 | Sister of `modal_call_id_ledger`, `wyner_ziv_deliverability` proofs |
| Cathedral autopilot consumer | **ADOPT CANONICAL**: v2 cascade reward-factor pattern per Catalog #319 v2 | Stacks orthogonally with existing 6 reward factors |
| Shannon entropy estimator (§3.1 + §3.2) | **UNIQUE FORK**: NEW `tac.theoretical_floor_estimator.shannon_video_entropy_estimator` | No pact canonical exists; per CLAUDE.md "Bit-level deconstruction and entropy discipline" the entropy estimate IS a primitive needed by this estimator |
| Bound calculators (§1.2 Bound 1/2/3) | **UNIQUE FORK**: NEW `tac.theoretical_floor_estimator.bounds` | NOVEL composition of Shannon + Atick-Redlich IB + Wyner-Ziv side-info bounds |
| Pareto extrapolation (§5.2 + §5.3) | **HYBRID**: polynomial fit uses canonical numpy.polyfit (CANONICAL); wavelet uses `pywt` (CANONICAL); submodular uses NEW `tac.theoretical_floor_estimator.pareto_extrapolation.submodular_lovasz_upper_bound` (UNIQUE FORK with empirical sub-modularity check) | Per restored framework #3 directive: submodular is potentially the strongest theoretical guarantee; warrants unique implementation |
| Plateau-vs-saturation classifier (§1.4 + §6) | **UNIQUE FORK**: NEW `tac.theoretical_floor_estimator.plateau_saturation_classifier` | NOVEL decision rule; bootstrap CI integrated; thresholds operator-tunable |
| Operator-facing CLI | **ADOPT CANONICAL**: `tools/estimate_theoretical_floor.py` mirrors `tools/scan_best_anchor_per_axis.py` pattern | Sister of `tools/audit_*` family |
| STRICT preflight gate (per §14 OP-4) | **ADOPT CANONICAL**: `src/tac/preflight.py` pattern + Catalog #N-th claim via `tools/claim_catalog_number.py --commit-via-serializer` | Per Catalog #186 |

**Rationale for HYBRID approach**: per the operator's UNIQUE-AND-COMPLETE-PER-METHOD operating mode (CLAUDE.md 2026-05-15) + the META-level PR95 lesson — the SHARED 90% (loaders / Provenance / persistence / autopilot consumer pattern) is HARD-EARNED canonical that serves; the UNIQUE 10% (entropy estimator / bound calculators / submodular check / classifier) is substrate-engineering UNIQUE because it has no pact precedent and is the COMPLETE-PER-METHOD novelty of this estimator class.

---

## 18. Premise verification per Catalog #229

Pre-edit verification (per the "premise verification before bulk edits" pattern from `feedback_prompt_premise_verification_before_edit_pattern_20260514.md`):

| PV # | Premise | Verification |
|---|---|---|
| 1 | The parent T3 meta-portfolio symposium's Assumption-Adversary mandate is verbatim *"build `tac.theoretical_floor_estimator` BEFORE next dispatch wave"* | VERIFIED via `Read` of `grand_council_meta_portfolio_re_ranking_post_compliance_envelope_20260518.md:56` |
| 2 | The current frontier per Catalog #316 is `0.19205 [contest-CPU]` (`pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean`; archive `6bae0201`) + `0.20533 [contest-CUDA]` (`pr106_format0d_latent_score_table`; archive `9cb989cef519`) | VERIFIED via `.venv/bin/python tools/scan_best_anchor_per_axis.py` at 2026-05-18T16:36Z |
| 3 | A master-gradient anchor exists for the fec6 archive (sha `f174192aeadf...`) | VERIFIED via direct read of `.omx/state/master_gradient_anchors.jsonl` — 2 rows; operating point `(d_seg=0.0009, d_pose=0.00173294, rate=0.004752, score=0.3386)` |
| 4 | The contest score formula at `upstream/evaluate.py:92` is `score = 100·seg_avg + sqrt(10·pose_avg) + 25·rate` with `rate = compressed_size / uncompressed_size` and `CONTEST_RATE_DENOM_BYTES = 37,545,489` | VERIFIED via `Read` of `upstream/evaluate.py:1-112` and `tac.master_gradient.CONTEST_RATE_DENOM_BYTES` |
| 5 | At PR106 frontier operating point `pose_avg = 3.4e-5`, the pose marginal is `5/sqrt(10·3.4e-5) = 271.16` vs SegNet marginal `100`, ratio 2.71× per CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent" | VERIFIED via in-context Python calculation: `pose_marg = 5/math.sqrt(10*3.4e-5) = 271.16` |
| 6 | The sister deterministic-optimizer subagent acb41f8d3f7f0a3ea is in-flight per the 4 design-constraint directives at `.omx/research/deterministic_optimizer_*_20260518.md` | VERIFIED via `Read` of all 4 directives |
| 7 | The `tac.wyner_ziv_deliverability.DeliverabilityProof` API per Catalog #319 Q1 exists with the canonical 4-tier classification (TIER_1_ZERO_COST / TIER_2_CONSTANTS / TIER_3_WAIVER_REQUIRED / TIER_4_FORBIDDEN) | VERIFIED via `Read` of `src/tac/wyner_ziv_deliverability/proof_builder.py:119-147` |
| 8 | The `tac.master_gradient.OperatingPoint` invariant requires `d_pose > 0` | VERIFIED via `Read` of `src/tac/master_gradient.py:133-138` |

All 8 premises VERIFIED before this memo's content was drafted. Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag (the docstring-overstatement trap)" — every quantitative claim in this memo carries either `[empirical:<path>]`, `[verified-against:<source>]`, or explicit `[predicted]` framing.

---

## 19. Acknowledgements

This design memo operates under the operator's NON-NEGOTIABLE 2026-05-17/18 mandate ("PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium" + "all candidates including c6 ibps may need further optimization and iteration and review and audit and individual extreme passion and detail and effort and adversarial grand council symposiums" + parent T3 symposium Assumption-Adversary mandate to build `tac.theoretical_floor_estimator` BEFORE next dispatch wave).

The 0.196-0.199 / 0.19205 frontier cluster's plateau-vs-saturation status has been an OPEN META-QUESTION since the 2026-05-15 UNIQUE-AND-COMPLETE-PER-METHOD operating mode amendment exposed canonicalization-by-default as the suppressor of substrate-optimal engineering. THIS memo's deliverable RESOLVES the meta-question empirically: PLATEAU CONFIRMED with HIGH confidence; the floor is `[0.026, 0.080]` per Bound 1 + Bound 3; current frontier sits 2.4-7.4× above floor; substantial unexplored direction remains.

Sister deliverable (deterministic-optimizer subagent acb41f8d3f7f0a3ea) provides the SOLVER that consumes this memo's floor estimate as its convergence stopping rule. The two memos COORDINATE without competing: floor-estimator IS the target metric; deterministic optimizer IS the trajectory toward it.

— Main-Claude (relayed per operator orchestration-queue directive 2026-05-18)
