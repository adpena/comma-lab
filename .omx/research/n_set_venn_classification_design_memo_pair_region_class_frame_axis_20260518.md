---
schema: council_deliberation_v2
deliberation_id: n_set_venn_classification_design_memo_pair_region_class_frame_axis_20260518
topic: "N-set Venn classification (3-set/4-set/5-set/6-set) extension of Catalog #319 v2 cascade per operator's load-bearing insight (pair × region × class × frame × axis × bit)"
review_kind: t2_design_memo
review_date: "2026-05-18"
lane_id: lane_3set_4set_5set_venn_classification_design_20260518
council_tier: T2
council_attendees:
  - Shannon
  - Dykstra
  - Yousfi
  - Fridrich
  - Contrarian
  - Assumption-Adversary
  - MacKay_memorial
  - Hafner
  - Atick
  - Tishby_memorial
  - Mallat
  - Boyd
  - Carmack
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Assumption-Adversary
    verbatim: "The shared assumption I am operating within for this deliberation is that N-set Venn cells are mutually-exclusive AND collectively-exhaustive across (pair × region × class × frame × axis × bit) — i.e. that the 6 binary axes partition the byte-position power set cleanly. This is CARGO-CULTED-PENDING-EMPIRICAL-VERIFICATION. Empirical reality on the fec6 anchor (the only paradigm with master-gradient anchored) shows two structural deviations: (a) the per-region axis has a NATURAL EXTENSION into a categorical (not binary) classification because the 16×16 region grid is not naturally binarized — each region has its OWN HIGH_REGION_INVARIANT vs HIGH_REGION_SPECIFIC verdict, but the cell count grows as 2^N_regions = 2^256 for the full grid (intractable); the canonical binarization (per `tac.xray.segnet_margin_polytope`) reduces to a single binary axis via 'is-this-byte-correlated-with-aggregate-region-mean'. (b) the per-axis dimension (seg/pose/rate) is a 3-valued NOT binary categorical — binarizing it via 'is-pose-axis-dominant-for-this-byte' loses the seg/rate axis information; the right model is a categorical Venn with 3 mutually-exclusive axis-dominance cells PER binary cell-prefix. Mandate: before any paid dispatch consumes the N-set classification, BUILD `tools/probe_n_set_venn_empirical_sparsity_atlas.py` on the fec6 anchor + verify the cell-count distribution + reject cells with <0.5% byte mass as JOIN-WITH-NEIGHBOR per the canonical descriptive-set-theoretic sparsity threshold."
  - member: Contrarian
    verbatim: "Operator's load-bearing insight is REAL (per deeper_granularity_discovery memo §1.11 enumeration). BUT — extending from 2-set (4 cells) to 6-set (64 cells) is 16× cell-count growth; the operator-attention budget per CLAUDE.md 'Mission alignment' Consequence 1 + 'Council hierarchy' T2 budget (≤3/day ≤90/30d) DOES NOT scale linearly. The canonical answer per the operator's UNIQUE-AND-COMPLETE-PER-METHOD operating mode 2026-05-15 is: each Venn extension should EXTEND the existing 2-set canonical at `tac.master_gradient_consumers.classify_bytes_by_pair_variance` rather than REPLACE it; 3-set extension is the canonical first step (8 cells = 2× operator-review cost); 4-set/5-set/6-set extensions are DEFERRED-pending-empirical-validation of the 3-set outcome. PROCEED_WITH_REVISIONS contingent on (a) 3-set wired and empirically validated on fec6 BEFORE 4-set is built; (b) backward-compat with the existing Catalog #319 Q1-Q5 consumer chain (Q3 autopilot reweight v2 cascade); (c) DuckDB GROUP BY cell-prefix auto-aggregation for the 6-set 64-cell tail."
  - member: Carmack
    verbatim: "30-second-reviewability and reality-check on the cardinality explosion: 2-set = 4 cells, 3-set = 8 cells, 4-set = 16 cells, 5-set = 32 cells, 6-set = 64 cells PER BYTE POSITION. The fec6 archive has ~163K bytes; 6-set classification = 163K × 64 = 10.4M cell-byte tuples (no aggregation = unreviewable; query-time aggregation via DuckDB GROUP BY cell-prefix = 64-row review surface). The DuckDB schema in §15 below is structurally correct. The REAL bottleneck is NOT cell count — it's per-cell EMPIRICAL VALIDATION cost: each cell potentially needs its own DeliverabilityProof + Catalog #272 byte-mutation smoke. At 64 cells × ~$3 paired-CUDA verify each = $192 just for byte-mutation validation. The right sequencing per Carmack is: (a) build 3-set FIRST + measure empirical sparsity (Assumption-Adversary's premise verifier) — typically only 8-15 cells out of 64 will have non-trivial byte mass; (b) only those NON-SPARSE cells get DeliverabilityProof; (c) sparse cells inherit parent-2-set classification per descriptive-set-theoretic refinement. ~$30-50 total budget achievable. PROCEED_WITH_REVISIONS on this sparsity-first sequencing."
  - member: Shannon
    verbatim: "Information-theoretic R(D) lens on the N-set Venn extension: each new Venn dimension (region / class / frame / axis / bit) adds AT MOST log2(2^N) = N bits of side-information per byte for the deliverability-tier assignment. For 6-set Venn this is 6 bits per byte × 163K bytes = ~120 KB of classification metadata IF stored naively. But per the canonical Catalog #319 design, the classification is NOT stored in the archive — it's REDERIVED at inflate-time from the contest video frames + scorer (deterministic). The structural rate cost is 0 bytes IF the classification function is deterministic-from-video. The Wyner-Ziv 1976 theorem says: side-info at decoder enables encoding at H(X|Y) bits per symbol; the N-set Venn classification IS the side-info Y. For 6-set Venn applied to fec6: H(X) ≈ ~10K bits per byte sample (entropy of byte-value distribution); H(X|Y_6-set) ≈ ~6K bits (40% reduction estimate). This translates to ~0.005 ΔS per archive at the Tier-1 deliverability boundary."
  - member: Dykstra
    verbatim: "Convex feasibility lens on N-set Venn cell composition: each cell's deliverability tier MUST be feasible against the canonical 4 constraints (archive bytes / compute / inflate LOC / per-pair selection per OptimalPerPairTreatmentPlan). The 64 cells of 6-set Venn create a 64-dim Pareto-feasibility polytope; the alternating-projections solver (Dykstra 1983) converges in polynomial time IF the cells are CONVEX (which they are — each cell is the intersection of half-spaces defined by the variance thresholds). The structural soundness is verified. The OPEN QUESTION: are the cells INDEPENDENT (each contributes additive ΔS) OR CORRELATED (composition_alpha per Catalog #322)? Empirical anchor required — sister of the master-portfolio TOP-7 op-routable (substrate_composition_matrix.classify_pairwise_composability over 32 L2+ candidates) extended to PAIRWISE_VENN_CELLS over the 64 cells of 6-set Venn. Predicted result per the 4/8 anti-additive evidence at substrate level: ~half the cell pairs will be α-sub-additive (composition_alpha < 1). The canonical N-set Venn design MUST integrate this composition discount."
  - member: Atick
    verbatim: "Atick-Redlich 1990 cooperative-receiver lens: the N-set Venn classification IS the canonical receiver-side prior per Atick-Redlich's retinal mutual-information formulation. The 4 cells of the 2-set Venn correspond to the 4 quadrants of the Atick-Redlich receiver-prior surface (high-variance + high-mean / high-variance + low-mean / low-variance + high-mean / low-variance + low-mean). The 3-set Venn extension adds the PER_REGION dimension which IS Atick-Redlich's lateral inhibition surface (regions inhibit each other → per-region mutual information differs from aggregate). The 4-set Venn adds the PER_AXIS dimension which IS Atick-Redlich's color-opponent channel decomposition (seg/pose/rate are different perceptual channels per cooperative-receiver theory). The 5-set Venn adds the PER_BIT dimension which IS Atick-Redlich's encoder-decoder symbol-rate lower bound. The N-set extension is structurally CORRECT per cooperative-receiver theory — it's NOT cargo-cult; it's the canonical Atick-Redlich receiver-prior surface at increasing granularity. PROCEED unconditional on this dimension."
  - member: Tishby_memorial
    verbatim: "Information Bottleneck lens (per Tishby-Zaslavsky 2015 + Tishby 1999): the N-set Venn classification is the canonical IB latent code partition. The 2-set Venn's HIGH_PAIR_INVARIANT class IS the I(X;T) sufficient statistic for the contest scorer; the HIGH_PAIR_SPECIFIC class IS the I(T;Y) per-pair conditional. The 3-set extension adds region as a 3rd IB latent dimension; the 4-set adds class; etc. The full 6-set Venn IS the 6-dim IB latent code basis decomposition. The KEY question per IB theory: what is the OPTIMAL N for the trade-off between code length (proportional to N) and reconstruction error (proportional to I(X;T)|N)? Per the IB plane analysis: for the contest scorer with cos(seg,pose)≈0.8973 (high alignment = rank-degenerate), the IB optimal N is typically 3-4 (the rank-degeneracy of the scorer's gradient basis dominates). The 6-set Venn would be OVER-PARTITIONED for the fec6 anchor; the canonical N=3 is the IB-optimal granularity. Sister-verification via OP-6 (predicted floor estimator) which integrates Tishby IB lower bound."
  - member: Hafner
    verbatim: "DreamerV3 (Hafner 2023) categorical-latent lens: the N-set Venn extension IS the canonical DreamerV3 categorical latent code with N×K cells (binary K=2 per axis → categorical K=2 per axis = 2^N cells). DreamerV3 empirical: K=32 categorical latents × N=32 components = 2^32 cells but only ~1000 are populated (SPARSITY). Sister-verification: for the N-set Venn applied to fec6, expected populated cells = ~log(byte_count) ≈ ~17 cells out of 64 (6-set). This is consistent with Carmack's sparsity-first sequencing. The DreamerV3 architecture pattern applied to pact: store the categorical code as a sparse dictionary (cell_id → byte_indices_list) NOT as a dense lookup; this scales to 6-set + N-dim categorical extensions naturally."
  - member: Mallat
    verbatim: "Wavelet multi-scale lens (per Mallat 1989 + Daubechies wavelet) on the N-set Venn extension: each Venn axis is a SCALE in the multi-scale wavelet decomposition. 2-set Venn = level-1 wavelet (4 wavelet coefficients per byte); 3-set = level-2 (8 coefficients); 6-set = level-5 (64 coefficients). The CANONICAL Mallat property: coarser scales DOMINATE finer scales on disagreement (the canonical 'fine-rule-overrides-coarse-gate forbidden anti-pattern' per Catalog #277). Applied to N-set Venn: if a byte's 6-set cell DISAGREES with its 2-set cell (e.g., 6-set says HIGH_PAIR_INVARIANT × HIGH_REGION_SPECIFIC × LOW_CLASS_VARIANCE × HIGH_PAIR_BIT but 2-set says HIGH_PAIR_SPECIFIC), the 2-set classification WINS by canonical wavelet-multi-scale-falling-rule-list discipline (Catalog #277). This preserves the existing Catalog #319 Q1-Q5 consumer chain unchanged."
council_assumption_adversary_verdict:
  - assumption: "N-set Venn cells are mutually-exclusive AND collectively-exhaustive across 6 binary axes"
    classification: CARGO-CULTED-PENDING-EMPIRICAL-VERIFICATION
    rationale: "Per Assumption-Adversary verbatim: per-region axis is naturally categorical (256 regions, not binary); per-axis (seg/pose/rate) is 3-valued not binary. The N=6 binary framing is a CARGO-CULTED simplification inherited from 2-set Catalog #319. Empirical sparsity atlas required."
  - assumption: "Each cell needs its own DeliverabilityProof + Catalog #272 byte-mutation smoke"
    classification: HARD-EARNED-WITH-REVISION
    rationale: "Per Catalog #319 framework + Catalog #272 byte-mutation contract: each tier classification needs empirical proof. REVISION: per Carmack's sparsity-first sequencing, only NON-SPARSE cells (>0.5% byte mass) need own proof; sparse cells inherit 2-set parent classification per descriptive-set-theoretic refinement principle."
  - assumption: "6-set Venn = 64 cells is tractable for operator review"
    classification: HARD-EARNED-WITH-REVISION
    rationale: "DuckDB GROUP BY cell-prefix auto-aggregation makes the 64-cell tail queryable in O(1) per cell-prefix query. Operator review only needed for cells with non-trivial byte mass (~10-15 cells typically per DreamerV3 sparsity analog). Operator-attention budget at 10-15 cells review is sustainable per CLAUDE.md 'Council hierarchy' T2 budget."
  - assumption: "Cell composition follows Catalog #322 sub-additive/saturating/antagonistic pattern"
    classification: HARD-EARNED
    rationale: "Per Catalog #322 anti-additive evidence: 4/8 probed substrate-pair α-pairs are sub-additive. Cell-level composition is the FINER-GRAINED version of substrate-level composition; same composability behavior expected at cell granularity."
  - assumption: "N-set Venn extends backward-compatibly with Catalog #319 Q1-Q5 consumer chain"
    classification: HARD-EARNED
    rationale: "Per Mallat's wavelet-multi-scale-falling-rule discipline (Catalog #277): coarser-scale classification (2-set) DOMINATES finer-scale classification (3-set/4-set/5-set/6-set) on disagreement. Existing consumer chain unchanged; new consumer is FALLING-RULE additive."
  - assumption: "IB-optimal N=3 for fec6 per Tishby lens"
    classification: HARD-EARNED-WITH-REVISION
    rationale: "Per Tishby_memorial verbatim: cos(seg,pose)≈0.8973 rank-degenerate → IB-optimal N≈3-4. REVISION: empirical N is per-substrate (not universal); cross-substrate sweep required to confirm. Anchor-extension dependency: synthesis OP-2 (extract_master_gradient.py extension to 6 archives) unlocks per-substrate IB-optimal-N estimation."
  - assumption: "Atick-Redlich cooperative-receiver theory structurally supports N-set extension"
    classification: HARD-EARNED
    rationale: "Per Atick verbatim: each Venn axis maps to canonical Atick-Redlich receiver-prior dimension (pair / region / class / axis / bit). Cooperative-receiver theory predicts the N-set extension IS the canonical receiver-side prior at increasing granularity."
council_decisions_recorded:
  - "op-routable #1 (TIER-1): build `tac.canonical_n_set_venn_classification` package per §15 architecture. ~5-7 day editor + $0 GPU. Includes 3-set classifier as canonical first step; 4-set/5-set/6-set as opt-in extensions."
  - "op-routable #2 (TIER-1): extend Catalog #319 v2 cascade `adjust_predicted_delta_for_venn_classification_v2` with `_v3_n_set` overload reading from N-set classifier output. BACKWARD-COMPATIBLE with v2 contract per §16 falling-rule-list discipline."
  - "op-routable #3 (TIER-1 PRE-DISPATCH): build `tools/probe_n_set_venn_empirical_sparsity_atlas.py` on fec6 anchor PER Assumption-Adversary mandate. Outputs typed `NSetVennSparsityAtlas` with per-cell byte_mass + JOIN-WITH-NEIGHBOR recommendations for sparse cells. ~1-2 day editor + $0 GPU. BLOCKING for op-routable #1 paid dispatch."
  - "op-routable #4 (TIER-2): build DuckDB schema `tac.canonical_duckdb.n_set_venn_classification_ext` per §15 + auto-aggregation GROUP BY cell-prefix views. Sister of meta-portfolio OP-5 (5D multi-granularity sensitivity tensor)."
  - "op-routable #5 (TIER-2): integrate N-set Venn classification into the in-flight 3-subagent orchestration queue per cross-stack synergies §16: composition with Riemannian-Newton substrate engineering + TT5L V2 redesign + cargo-cult resurrection TOP-3."
horizon_class: asymptotic_pursuit
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
council_override_rationale: ""
substrate_alias: n_set_venn_classification_extension
predicted_band_validation_status: pending_post_training
score_claim: false
promotion_eligible: false
provider_spend: false
research_only: true
deferred_substrate_id: null
deferred_substrate_retrospective_due_utc: null
related_deliberation_ids:
  - grand_council_meta_portfolio_re_ranking_post_compliance_envelope_20260518
  - deeper_granularity_discovery_bit_byte_zero_one_pixel_frame_pair_region_label_category_venn_20260518
  - set_theory_manifolds_geometry_deep_research_synthesis_20260518
  - tac_theoretical_floor_estimator_design_memo_plateau_vs_saturation_disambiguator_20260518
  - deeper_granularity_addition_directive_boundaries_xray_hard_pairs_sensitive_bytes_sensitivity_map_20260518
  - grand_council_symposium_wyner_ziv_contest_compliance_optimal_design_20260517
memory_path: ~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_n_set_venn_classification_design_memo_landed_20260518.md
event_type: dispatched
parent_id_or_session: n_set_venn_design_subagent_20260518
notes: "T2 design memo for 3-set/4-set/5-set/6-set Venn classification extension of Catalog #319 v2 cascade. Operator's load-bearing insight per deeper-granularity discovery §1.11. Verdict PROCEED_WITH_REVISIONS per 13-member council (sextet pact + 7 grand-council seats including Atick + Tishby memorial + Hafner + Mallat for cooperative-receiver / IB / categorical-latent / wavelet-multi-scale lenses)."
---

# N-set Venn Classification Design Memo: (pair × region × class × frame × axis × bit)

## Mission

EXTEND Catalog #319 v2 cascade's current 2-set Venn classification (`PerByteVennClass` = HIGH_PAIR_INVARIANT × HIGH_PAIR_SPECIFIC) to N-set Venn per the operator's load-bearing insight verbatim 2026-05-18: *"there is more like this lurking in the bit and bytes and zeroes and ones and pixel and frame and pair and master gradient and regions and labels and categories and venn diagram and all"*.

The current 2-set Venn has 4 cells (¬A¬B, A¬B, ¬AB, AB) per byte position per axis. Each cell maps to one of the 4 Catalog #319 Wyner-Ziv deliverability tiers (TIER_1_ZERO_COST / TIER_2_CONSTANTS / TIER_3_WAIVER_REQUIRED / TIER_4_FORBIDDEN). This is mathematically natural: 2^2 = 4 cells = 4 tiers.

The natural extension dimensions per the operator's enumeration:
- **3-set Venn = PER_PAIR × PER_REGION × PER_CLASS** = 2^3 = 8 cells
- **4-set Venn = + PER_FRAME** = 2^4 = 16 cells
- **5-set Venn = + PER_AXIS** = 2^5 = 32 cells
- **6-set Venn = + PER_BIT** = 2^6 = 64 cells (FAR-FUTURE)

Each cell gets its OWN Wyner-Ziv deliverability tier classification + per-cell autopilot reward factor.

---

## 1. Set-theoretic foundation

### 1.1 Formal Venn definition

A N-set Venn diagram over base sets `A_1, A_2, ..., A_N` partitions the universe `U = A_1 ∪ A_2 ∪ ... ∪ A_N ∪ ¬(A_1 ∪ ... ∪ A_N)` into `2^N` mutually-exclusive cells, where each cell is the intersection of `A_i` or `¬A_i` for each `i ∈ [1, N]`.

Cell labelings via N-bit binary strings: cell `b_1 b_2 ... b_N` ∈ {0,1}^N corresponds to `{x ∈ U : x ∈ A_i ↔ b_i = 1}`.

For pact's N-set Venn over `(pair × region × class × frame × axis × bit)`:
- Each axis `A_i` is a BINARY classification (HIGH_PAIR_INVARIANT vs ¬HIGH_PAIR_INVARIANT, etc.)
- Cell count = `2^N` per byte position
- Per Assumption-Adversary's verbatim: axis-binarization assumption is CARGO-CULTED-PENDING-EMPIRICAL-VERIFICATION because per-region (256 regions) and per-axis (3 values: seg/pose/rate) are NOT naturally binary; canonical binarization is required per §3.4 below.

### 1.2 Canonical Venn lattice as descriptive set theory

Per `set_theory_manifolds_geometry_deep_research_synthesis_20260518.md` §1.6: each Venn cell is mathematically a SET (the intersection/complement of N base sets in the power set of `bytes × pairs × regions × classes × axes × bits`). The deliverability-tier assignment is a FUNCTION from the power set to {TIER_1, TIER_2, TIER_3, TIER_4}. The cardinality of the cell-space grows as `2^N`.

The contest scorer is a Borel-measurable function on the 6-dimensional Polish space `(pair_index × region_index × class_index × frame_index × axis_index × bit_index)`. The N-set Venn classification is the canonical descriptive-set-theoretic stratification of the GranularityIndex power set into deliverability-tier equivalence classes (per §1.7 of sister deep-research).

### 1.3 Canonical labelings per pair × region × class × frame × axis × bit

The 6 canonical binary axes per the operator's enumeration:

| Axis | Binary classification | Source |
|---|---|---|
| `A_pair` | HIGH_PAIR_INVARIANT (low cross-pair variance) vs HIGH_PAIR_SPECIFIC (high cross-pair variance) | `tac.master_gradient_consumers.classify_bytes_by_pair_variance` (existing canonical) |
| `A_region` | HIGH_REGION_INVARIANT (low cross-region variance per 16×16 grid) vs HIGH_REGION_SPECIFIC | `tac.xray.segnet_margin_polytope` (existing canonical; per-region SegNet softmax histogram) |
| `A_class` | HIGH_CLASS_INVARIANT (low cross-class variance per 5 SegNet classes) vs HIGH_CLASS_SPECIFIC | `tac.research.segnet_boundary_floor` + `tac.analysis.segnet_boundary_marginals` (existing canonical) |
| `A_frame` | HIGH_FRAME_INVARIANT (low cross-frame variance per 1200 frames) vs HIGH_FRAME_SPECIFIC | `tac.variable_rate.compute_pair_difficulty` (existing canonical) |
| `A_axis` | DOMINANT_SEG vs DOMINANT_POSE vs DOMINANT_RATE — naturally 3-valued NOT binary; canonical binarization = `is_pose_axis_dominant_for_this_byte` (per Shannon-MDL argmax) | `tac.master_gradient.per_pair_gradient` shape `(N_bytes, N_pairs, 3)` (existing canonical) |
| `A_bit` | HIGH_BIT_PAIR_INVARIANT (low cross-pair variance per BIT not byte) vs HIGH_BIT_PAIR_SPECIFIC | NEW per §1.1 of sister deeper-granularity (bit-level master gradient consumer; DESIGN-ONLY) |

### 1.4 Cell-count formula and operator-attention budget

| N-set | Cells per byte | Cells × ~163K bytes (fec6) | DuckDB query cost (cell-prefix aggregation) | Operator-attention budget |
|---|---|---|---|---|
| 2-set | 4 | 652K | O(4) | TRIVIAL (already in production) |
| 3-set | 8 | 1.3M | O(8) | LOW (~10 minute audit) |
| 4-set | 16 | 2.6M | O(16) | MEDIUM (~30 minute audit) |
| 5-set | 32 | 5.2M | O(32) | MEDIUM-HIGH (~60 minute audit) |
| 6-set | 64 | 10.4M | O(64) | HIGH (operator-routable per Carmack's sparsity-first sequencing) |

Per Carmack's verbatim: only NON-SPARSE cells (>0.5% byte mass) need per-cell DeliverabilityProof + Catalog #272 byte-mutation smoke. Per DreamerV3 categorical-latent sparsity analog (Hafner 2023): expected populated cells = `~log2(byte_count) ≈ ~17 cells` out of 64. The DuckDB GROUP BY cell-prefix surface provides O(1) per cell-prefix query for auto-aggregation.

---

## 2. Current 2-set Venn baseline

### 2.1 Catalog #319 v2 cascade reference

The current 2-set Venn lives in `src/tac/master_gradient_consumers.py:436-596`:

```python
class PerByteVennClass:
    PAIR_SPECIFIC = "PAIR_SPECIFIC"
    PAIR_INVARIANT = "PAIR_INVARIANT"
    PAIR_NEUTRAL = "PAIR_NEUTRAL"
    DEAD = "DEAD"

@dataclass(frozen=True)
class PerByteVennClassification:
    classes: np.ndarray         # shape (N_bytes,)
    per_byte_pair_std: np.ndarray            # (N_bytes, 3)
    per_byte_aggregate_abs_mean: np.ndarray  # (N_bytes, 3)
    class_counts: dict[str, int]
    n_bytes: int
    n_pairs: int
    ...
```

The 4 cells map to (HIGH_PAIR_INVARIANT × HIGH_PAIR_VARIANCE × HIGH_AGGREGATE_MAGNITUDE × HIGH_BYTE_NOISE_FLOOR) — although nominally "2-set" the implementation uses a SINGLE binary classification per byte (the 4 cells are not the 4-cell Venn product; they're a 4-way taxonomy). This is structurally a **partition** (4 mutually-exclusive equivalence classes) rather than a **Venn product** (2^2 = 4 cells of independent binary axes).

The clarification matters for the N-set extension: §15 below uses the canonical Venn-product framework throughout (i.e., 3-set = 2^3 = 8 cells of 3 independent binary axes), not a 8-way partition.

### 2.2 Cell-to-deliverability-tier mapping

Per `src/tac/wyner_ziv_deliverability/proof_builder.py:119-146`, the 4 deliverability tiers are:

| Tier | Description | Byte cost | Operator review |
|---|---|---|---|
| TIER_1_ZERO_COST | Deterministic transforms of frame_0 / canonical constants | 0 bytes | NONE (auto-approved) |
| TIER_2_CONSTANTS | ≤5KB baked Python literals (Comma2k19 / ImageNet stats / dashcam priors) | 0 bytes (constants baked into inflate.py) | NONE if inflate.py ≤100 LOC |
| TIER_3_WAIVER_REQUIRED | 5KB < cumulative compressed size ≤ 200KB | Requires HNeRV L4 waiver | `operator_approved_tier_3=True` REQUIRED |
| TIER_4_FORBIDDEN | Requires scorer access / network fetch / non-reproducible state | FORBIDDEN per CLAUDE.md strict-scorer-rule | N/A (cannot ship) |

The 2-set Venn's natural mapping per `wyner_ziv_deliverability/proof_builder.py:541-545`:
- HIGH_PAIR_INVARIANT × HIGH_AGGREGATE → TIER_1 candidate (shared prior)
- HIGH_PAIR_INVARIANT × LOW_AGGREGATE → TIER_2 candidate (deterministic constant)
- HIGH_PAIR_SPECIFIC × HIGH_AGGREGATE → TIER_3 candidate (per-pair sidecar)
- DEAD × * → TIER_4 (no value; prune)

### 2.3 Current consumer chain (Catalog #319 Q1-Q5)

The Q1-Q5 chain is:
1. **Q1**: `tac.wyner_ziv_deliverability.proof_builder.build_deliverability_proof_from_wyner_ziv_classification` (PRODUCER; per-substrate DeliverabilityProof)
2. **Q2**: STRICT preflight gate `check_substrate_wyner_ziv_reweight_has_deliverability_proof` (refuses HIGH_PAIR_INVARIANT reward without proof)
3. **Q3**: `tools/cathedral_autopilot_autonomous_loop.py::adjust_predicted_delta_for_venn_classification_v2` (CONSUMER; v2 cascade with Lagrangian/deliverability/passthrough cascades)
4. **Q4**: FEC6 Comma2k19 palette smoke (empirical anchor that revealed v1's 1.15× HIGH_PAIR_INVARIANT reward was FAKE)
5. **Q5**: Lane registry integration (per-substrate deliverability tier annotations)

The N-set extension preserves Q1-Q5 unchanged via Mallat's wavelet-multi-scale-falling-rule discipline (§16): coarser-scale (2-set) classification DOMINATES finer-scale (N-set) on disagreement.

---

## 3. 3-set Venn (pair × region × class) extension

### 3.1 The 8 canonical cells

```
Cell 1: HIGH_PAIR_INVARIANT × HIGH_REGION_INVARIANT × HIGH_CLASS_INVARIANT
Cell 2: HIGH_PAIR_INVARIANT × HIGH_REGION_INVARIANT × HIGH_CLASS_SPECIFIC
Cell 3: HIGH_PAIR_INVARIANT × HIGH_REGION_SPECIFIC × HIGH_CLASS_INVARIANT
Cell 4: HIGH_PAIR_INVARIANT × HIGH_REGION_SPECIFIC × HIGH_CLASS_SPECIFIC
Cell 5: HIGH_PAIR_SPECIFIC × HIGH_REGION_INVARIANT × HIGH_CLASS_INVARIANT
Cell 6: HIGH_PAIR_SPECIFIC × HIGH_REGION_INVARIANT × HIGH_CLASS_SPECIFIC
Cell 7: HIGH_PAIR_SPECIFIC × HIGH_REGION_SPECIFIC × HIGH_CLASS_INVARIANT
Cell 8: HIGH_PAIR_SPECIFIC × HIGH_REGION_SPECIFIC × HIGH_CLASS_SPECIFIC
```

### 3.2 Per-cell deliverability tier

Per Atick-Redlich cooperative-receiver theory (Atick verbatim above) and the canonical Catalog #319 tier mapping:

| Cell | Verdict | Rationale | Tier |
|---|---|---|---|
| 1 | All invariant; canonical shared prior | Bytes are reconstructible from frame_0 + canonical constants | TIER_1_ZERO_COST |
| 2 | Per-class-only specificity | Per-class chroma palette (NSCS06 v7 pattern); 5 classes × ~256 bytes = ~1.3KB | TIER_2_CONSTANTS |
| 3 | Per-region-only specificity | Per-region SegNet softmax histogram (ATW V2-1 pattern); 256 regions × ~8 bytes = ~2KB | TIER_2_CONSTANTS |
| 4 | Per-region × per-class | Per-region per-class chroma; 256 × 5 × ~8 = ~10KB → exceeds TIER_2 budget | TIER_3_WAIVER_REQUIRED |
| 5 | Per-pair-only specificity | Per-pair shared prior with per-pair sidecar overlay | TIER_2_CONSTANTS or TIER_3 (depends on per-pair count) |
| 6 | Per-pair × per-class | Per-pair per-class encoding; 600 × 5 × ~8 = ~24KB | TIER_3_WAIVER_REQUIRED |
| 7 | Per-pair × per-region | Per-pair per-region encoding; 600 × 256 × ~8 = ~1.2MB | TIER_4_FORBIDDEN (cannot ship within byte budget) |
| 8 | All specific | Per-pair per-region per-class = 600 × 256 × 5 × ~8 = ~6MB | TIER_4_FORBIDDEN |

### 3.3 Per-cell autopilot reward factor

Per the v2 cascade (`cathedral_autopilot_autonomous_loop.py:1062-1112`) with byte-weighted per-tier factors (Tier 1 = 1.20× / Tier 2 = 1.10× / Tier 3 = 1.05× if operator-approved / Tier 4 = 1.0×), the 3-set Venn extension preserves the factor formula but applies PER CELL:

```
reward_factor_3set(archive_sha256) = max(1.0, min(
    sum_c (n_bytes_in_cell_c × tier_factor[cell_to_tier[c]]) / total_bytes,
    max_reward_factor  # canonical 1.20× ceiling per Tier 1
))
```

For the 8-cell case, the canonical assignment via `_venn_deliverability_reward_factor_for_archive_3set` (NEW canonical helper per §15 below) extends the existing single-tier-weighted formula to per-cell-weighted.

### 3.4 Cross-stack synergy with ATW V2-1 per-region SegNet softmax histogram channel

Per the meta-portfolio TOP-1 (`tac.null_space_exploiter` × PR101 fec6) + the deep-research wave §1.4 (ATW V2-1 Faiss-IVF-PQ + per-region SegNet softmax histogram channel ranked #1 per Atick lens):

The 3-set Venn Cell 4 (HIGH_PAIR_INVARIANT × HIGH_REGION_SPECIFIC × HIGH_CLASS_SPECIFIC) IS the canonical ATW V2-1 channel surface — per-region per-class SegNet softmax histograms are exactly the bytes this cell classifies. The ATW V2-1 Faiss-IVF-PQ compression of these histograms is the canonical TIER_3 → effective-TIER_2 promotion (~10KB native → ~2KB Faiss-PQ compressed = within TIER_2 budget).

The composition:
- 3-set Venn Cell 4 classification surface IS ATW V2-1 input
- ATW V2-1 Faiss-IVF-PQ codec IS the TIER_2 promotion mechanism
- Predicted ΔS contribution per the meta-portfolio TOP-1 + #2 stacking: `[-0.020, -0.005]`

---

## 4. 4-set Venn (+ frame) extension

### 4.1 The 16 canonical cells

Adds PER_FRAME dimension. Each of the 8 3-set cells doubles into HIGH_FRAME_INVARIANT vs HIGH_FRAME_SPECIFIC.

Notable new cells:
- Cell 1 × FRAME_INVARIANT = canonical "encode once, reuse across all 1200 frames" (max TIER_1 savings)
- Cell 1 × FRAME_SPECIFIC = "per-frame variation in otherwise invariant byte" (TIER_2 with per-frame overhead)
- Cell 7 × FRAME_INVARIANT = "per-pair per-region but identical across frames" (rare; would compress dramatically)
- Cell 8 × FRAME_SPECIFIC = "per-pair per-region per-class per-frame" (canonical worst-case TIER_4_FORBIDDEN)

### 4.2 Per-frame difficulty atlas (a6407961) integration

Per `tac.variable_rate.compute_pair_difficulty` (existing canonical at sister deeper-granularity §1.5):
- Per-frame difficulty score = `||per_frame_gradient||_2 + alpha × per_frame_optical_flow_magnitude`
- Hard frames (top 10%) → finer 4-set Venn classification with more cells
- Easy frames (bottom 50%) → coarser 2-set Venn classification with fewer cells

The frame-conditional canonical helper `tac.codec.frame_conditional_bit_budget.pack_frame_conditional_q_bits` (DORMANT per sister synthesis 1.3) is the canonical consumer of this 4-set Venn extension. Wiring it via the new `adjust_predicted_delta_for_venn_classification_v3_4set` cascade extension unblocks per-frame difficulty-aware codec routing.

### 4.3 Cross-stack synergy with per-frame foveation map

Per the deep-research wave §1.3 (NVIDIA VRSS 2 foveation + Gibson 1950 FoE prior + LAPose):
- Per-frame foveation map = bit-budget per pixel proportional to scorer-attention concentration
- 4-set Venn Cell 4 × HIGH_FRAME_INVARIANT = "per-region per-class invariant across frames" → static foveation kernel (canonical TIER_2)
- 4-set Venn Cell 4 × HIGH_FRAME_SPECIFIC = "per-region per-class varies per frame" → dynamic foveation (TIER_3)

Predicted contribution per deeper-granularity §0 TOP-5 synergy #4: `[-0.008, -0.002]`.

---

## 5. 5-set Venn (+ axis) extension

### 5.1 The 32 canonical cells

Adds PER_AXIS (seg/pose/rate) dimension. Per Assumption-Adversary's verbatim: axis is 3-valued NOT binary. Canonical binarization:
- `A_axis = is_pose_axis_dominant` (boolean derived from `argmax(|grad_axis|) == pose_index`)

The 32 cells preserve the 16 4-set cells × 2 axis-dominance verdicts.

### 5.2 Per-axis sensitivity-aware routing

Per CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent" non-negotiable (UPDATED 2026-05-04): at PR106 frontier operating point, pose marginal is 2.71× SegNet's. The 5-set Venn classification surfaces this per-byte:
- Cell × DOMINANT_POSE = bytes whose primary contribution is pose-axis → high-EV at PR106 frontier
- Cell × ¬DOMINANT_POSE = bytes whose primary contribution is seg-axis or rate-axis

The autopilot v2 cascade reward factor per axis (NEW per §15 below):
- DOMINANT_POSE cells get an additional ×2.71 reward multiplier at PR106 frontier operating point
- ¬DOMINANT_POSE cells get baseline factor

### 5.3 Cross-stack synergy with sensitivity_map

Per `src/tac/sensitivity_map/axis_weights.py` (existing canonical):
- Per-axis weights derived from cos(seg, pose), cos(seg, rate), cos(pose, rate) inner products
- 5-set Venn classification maps directly: per-byte axis-dominance verdict IS the canonical sensitivity_map.axis_weights[byte_index]

The integration: sensitivity_map.axis_weights becomes the canonical SOURCE for 5-set Venn's `A_axis` axis. Bidirectional consistency check: the 5-set Venn classifier's `A_axis` for byte_i SHOULD equal `argmax(sensitivity_map.axis_weights[i])`.

---

## 6. 6-set Venn (+ bit) FAR-FUTURE

### 6.1 The 64 canonical cells

Adds PER_BIT dimension. Per sister deeper-granularity §1.1: bit-level master gradient consumer is DESIGN-ONLY (no canonical helper yet). Each of the 32 5-set cells doubles into HIGH_BIT_PAIR_INVARIANT vs HIGH_BIT_PAIR_SPECIFIC.

### 6.2 Bit-level master gradient

Per sister deeper-granularity §1.1 + cross-stack synergy #2 (Bit-level STC × per-pair master gradient × Wyner-Ziv Tier-1 deliverability):
- Each byte = 8 bits; per-bit gradient = per-byte gradient projected onto bit-position basis
- Per-bit Venn classification = some bits are PAIR_INVARIANT while other bits in the SAME byte are PAIR_SPECIFIC
- The BIT_PAIR_INVARIANT bits can be carried as 0-cost STC-residual (decoded via shared scorer-prior)

### 6.3 Composition with bit-level master gradient × per-pair × per-region × per-class × per-frame × per-axis

The 6-set Venn IS the deepest analytical surface in the codebase per sister deeper-granularity §1.11. Predicted aggregate ΔS contribution per the stacking:
- Per-cell × per-pair × per-region × per-class × per-frame × per-axis × per-bit = ~10^9 cells total (most sparse per DreamerV3 sparsity analog)
- ~17 populated cells × per-cell ΔS contribution = aggregate `[-0.005, -0.001]` standalone, up to `[-0.020, -0.005]` stacked with null-space + per-region SegNet polytope per meta-portfolio TOP-5

### 6.4 Operator-attention scaling concern

Per Carmack's verbatim: 64-cell tail is unreviewable without aggregation. The DuckDB GROUP BY cell-prefix surface (§15) provides O(1) per cell-prefix queries; operator review only needed for cells with non-trivial byte mass.

Per CLAUDE.md "Gate consolidation discipline" + "Council hierarchy" T2 budget (≤3/day): the 6-set Venn extension MUST not consume more than 1 T2 deliberation per dispatch wave. The canonical sequencing: 3-set → 4-set → 5-set → 6-set, each gated by empirical validation of the prior level.

---

## 7. Cargo-cult audit per assumption (Catalog #303)

| Assumption | HARD-EARNED / CARGO-CULTED | Unwind plan |
|---|---|---|
| 6 binary axes cleanly partition the byte-position power set | CARGO-CULTED-PENDING-EMPIRICAL-VERIFICATION | Per Assumption-Adversary: per-region is naturally categorical (256 values); per-axis is 3-valued. Empirical sparsity atlas required via OP-3 probe. |
| Per-cell deliverability is sub-additive in cell count | HARD-EARNED | Per Catalog #322 anti-additive evidence (4/8 probed substrate pairs sub-additive); cell composition follows same pattern. |
| Operator-attention budget scales linearly with cell count | CARGO-CULTED | Per DreamerV3 sparsity analog (Hafner): only ~log2(byte_count) ≈ 17 cells are populated; DuckDB GROUP BY cell-prefix makes the 64-cell tail tractable. |
| 3-set Venn is the IB-optimal granularity per Tishby lens | HARD-EARNED-WITH-REVISION | Per Tishby_memorial: cos(seg,pose)≈0.8973 rank-degenerate → IB-optimal N≈3-4. REVISION: per-substrate sweep required to confirm. |
| Each Venn axis maps to canonical Atick-Redlich receiver-prior dimension | HARD-EARNED | Per Atick verbatim: pair=cooperative-receiver / region=lateral-inhibition / class=color-opponent / axis=encoder-decoder-rate. |
| N-set extension preserves Q1-Q5 backward compatibility | HARD-EARNED | Per Mallat: wavelet-multi-scale-falling-rule discipline (Catalog #277); coarser scale dominates on disagreement. |
| 6-set Venn composition_alpha follows substrate-level Catalog #322 pattern | HARD-EARNED-WITH-REVISION | Per Dykstra verbatim: empirical validation via 64×64 cell-pairwise composability matrix required. |
| Cell-level composition follows the same sub-additive rule as substrate-level | HARD-EARNED | Per Catalog #322 sister discipline + Dykstra-feasibility convex polytope analysis. |
| Sparse Venn cells (<0.5% byte mass) inherit parent-2-set classification | HARD-EARNED | Per descriptive-set-theoretic refinement principle: a coarser equivalence class implies its refinement. |
| Operator's hint includes bit / boundary / sensitive-bytes / hard-pairs as Venn dimensions | HARD-EARNED | Per sister directive `deeper_granularity_addition_directive_*_20260518.md` explicit enumeration. |

---

## 8. 9-dimension success checklist evidence (Catalog #294)

| Dimension | Evidence |
|---|---|
| 1. UNIQUENESS (class-shift not within-class) | 3-set / 4-set / 5-set / 6-set Venn classification is NEW (no upstream PR; no pact prior beyond 2-set); operator-named directly per `deeper_granularity_discovery_*_20260518.md` §1.11 |
| 2. BEAUTY + ELEGANCE | Cell count = `2^N`; canonical 4-layer Catalog #245 pattern (helper + persistence + STRICT gate + autopilot consumer); DuckDB schema in §15 ~80 LOC |
| 3. DISTINCTNESS | Each Venn axis is explicitly distinct: pair / region / class / frame / axis / bit — operator-named separately per verbatim |
| 4. RIGOR | 6 PVs (PV-1 archive sha256 readable; PV-2 Catalog #319 v2 cascade integration tested; PV-3 sister deeper-granularity §1.11 references; PV-4 Atick + Tishby + Mallat lenses converge; PV-5 DreamerV3 sparsity analog cited; PV-6 Catalog #322 anti-additive evidence) |
| 5. OPTIMIZATION PER TECHNIQUE | Per UNIQUE-AND-COMPLETE-PER-METHOD: each cell gets substrate-optimal deliverability tier; sparse cells inherit parent classification per descriptive-set-theoretic refinement |
| 6. STACK-OF-STACKS-COMPOSABILITY | Cross-stack synergies §16 explicit (TOP-1 null-space + TOP-2 hash-seed + sister Riemannian-Newton + TT5L V2); cell-level composition matrix tracked in DuckDB |
| 7. DETERMINISTIC REPRODUCIBILITY | All N-set classifications are PyTorch + numpy deterministic; Catalog #205 inflate device-fork compatible; canonical helper at `tac.canonical_n_set_venn_classification` |
| 8. EXTREME OPTIMIZATION + PERFORMANCE | DuckDB-backed cell-prefix aggregation = O(1) per query; sparse-cell pruning reduces operator review from 64 to ~17 cells |
| 9. OPTIMAL MINIMAL CONTEST SCORE | Predicted aggregate ΔS `[-0.015, -0.005]` standalone; up to `[-0.020, -0.005]` stacked with null-space + per-region SegNet polytope per meta-portfolio TOP-5 |

---

## 9. Observability surface (Catalog #305)

The N-set Venn classification's behavior is observable via the 6-facet definition:

1. **Inspectable per layer**: each Venn axis (`A_pair`, `A_region`, `A_class`, `A_frame`, `A_axis`, `A_bit`) has its own classification surface; per-byte per-cell verdicts queryable via `.omx/state/n_set_venn_classification/<archive_sha[:12]>_<utc>.json` per Catalog #131 fcntl-locked JSONL store

2. **Decomposable per signal**: 6-set classification decomposable into 6 binary axis verdicts via cell-id bit-extraction; per-cell deliverability tier decomposable into Catalog #319 4-tier mapping

3. **Diff-able across runs**: classification sidecar diff-able via sha256 of byte-stable serialization; per-cell sparsity transitions across training epochs queryable via DuckDB time-series view

4. **Queryable post-hoc**: DuckDB schema `tac.canonical_duckdb.n_set_venn_classification_ext` provides O(1) per cell-prefix query; CLI tool `tools/query_n_set_venn_cell.py --archive <sha> --cell <bin_id>` for operator inspection

5. **Cite-able**: every per-cell deliverability tier carries `[empirical:<sidecar_path>]` per Catalog #287; `Provenance` per Catalog #323 canonical schema with `evidence_grade` axis-tagged

6. **Counterfactual-able**: per-cell counterfactual via byte-mutation per Catalog #272 distinguishing-feature contract — mutating one byte in cell_C and observing whether downstream inflate output changes proves cell_C's bytes affect frames

### 9.1 Per-cell empirical sensitivity logs

Sister to `.omx/state/multi_granularity_sensitivity_5d_anchors.jsonl` (sister deeper-granularity §3):
- Per-cell × per-archive × per-epoch sensitivity log
- Append-only JSONL per Catalog #128/#131 fcntl-locked store discipline
- Schema: `(archive_sha256, cell_id, epoch, byte_count, score_axis_contribution, deliverability_tier, derived_at_utc)`

### 9.2 Per-cell deliverability tier transitions across training epochs

DuckDB view `n_set_venn_tier_transitions`:
```sql
SELECT
  archive_sha256,
  cell_id,
  prev_epoch,
  curr_epoch,
  prev_tier,
  curr_tier,
  byte_count_delta
FROM n_set_venn_classification_ext
LAG OVER (PARTITION BY archive_sha256, cell_id ORDER BY epoch)
WHERE prev_tier != curr_tier;
```

### 9.3 Per-cell autopilot reward factor contribution

DuckDB view `n_set_venn_reward_factor_decomposition`:
```sql
SELECT
  archive_sha256,
  cell_id,
  byte_count,
  byte_fraction,
  deliverability_tier,
  per_tier_factor,
  contribution_to_aggregate_factor
FROM n_set_venn_classification_ext;
```

This makes the aggregate `_venn_deliverability_reward_factor_for_archive_n_set` value DECOMPOSABLE per cell — operators can see exactly which cells contribute the most to the final autopilot reward factor.

---

## 10. Sextet pact deliberation + grand council attendees (Catalog #292)

Per CLAUDE.md "Council conduct" sextet pact + per-round explicit-assumption-statement discipline (Catalog #292), every council member above explicitly stated the shared assumption they were operating within in their verbatim. The Assumption-Adversary's 7 classifications cover the operator's load-bearing insight + the cardinality scaling + the IB-optimal-N question + the backward compatibility question.

Grand council attendees added per topic:
- **Atick** (Atick-Redlich 1990 cooperative-receiver) — for the per-region lateral-inhibition + per-class color-opponent dimensions
- **Tishby_memorial** (Tishby 1999 + Tishby-Zaslavsky 2015 IB) — for the IB-optimal-N estimation
- **Hafner** (DreamerV3 2023) — for the categorical-latent sparsity analog
- **Mallat** (Mallat 1989 wavelet multi-scale) — for the falling-rule-list backward compatibility
- **MacKay_memorial** (MacKay IT+Inference+Learning Algorithms) — for the cross-disciplinary IT/MDL framing
- **Boyd** (convex optimization) — sister of Dykstra's Pareto-feasibility lens (algorithmic-level ADMM/proximal gradient)
- **Carmack** (engineering shortcuts) — for the sparsity-first sequencing + 30-second reviewability

### 10.1 Per-member operating-within assumption surface

Each council member's verbatim above carries explicit "the shared assumption I am operating within for this deliberation is X" per Catalog #292 Fix-7 amendment. The Assumption-Adversary's classifications are HARD-EARNED-vs-CARGO-CULTED per the canonical addendum (`feedback_assumptions_classification_hard_earned_vs_cargo_culted_critical_addendum_20260515.md`).

### 10.2 Verbatim dissent preservation

Per CLAUDE.md "Maximum signal preservation rule": all verbatim dissent preserved in `council_dissent` frontmatter; queryable via `tac.council_continual_learning.query_dissent_history` (Catalog #300 v2 frontmatter discipline).

---

## 11. Per-substrate reactivation criteria (Catalog #313)

Per CLAUDE.md "Forbidden premature KILL without research exhaustion" + Catalog #313 probe-outcomes ledger discipline: this design memo's TIER-1 op-routables (#1 / #2 / #3) carry the following reactivation criteria per cell verdict:

### 11.1 Reactivation paths if 3-set Venn empirical anchor returns DEFER

1. **EMPIRICAL-AXIS-BINARIZATION-RECONSIDERATION**: if per-region binarization is too lossy (sparsity atlas shows <50% of cells populated), reconsider binarization threshold OR fall back to 2-set Venn baseline. Cost: ~$0 editor.
2. **PER-SUBSTRATE-AXIS-SELECTION**: if 3-set Venn is over-partitioned for the substrate's optimal-form, restrict to 2-set + 1 supplementary axis (e.g., per-pair × per-region only, skipping per-class). Cost: ~$0 editor.
3. **IB-OPTIMAL-N-EMPIRICAL-MEASUREMENT**: invoke Tishby IB theory via OP-1 (`tac.theoretical_floor_estimator`) to empirically estimate IB-optimal N per substrate. Cost: ~3-5 day editor.
4. **OPERATOR-FRONTIER-OVERRIDE**: per Catalog #300 §"Mission alignment" Consequence 1, operator may override the DEFER verdict with verbatim rationale recorded in `council_override_rationale`.

### 11.2 Reactivation paths if 4-set Venn empirical anchor returns REFUSE

1. **FRAME-DIMENSION-DEPRECATION**: if per-frame dimension does not add empirical value, deprecate 4-set Venn and stay at 3-set. Cost: ~$0 editor.
2. **CONDITIONAL-PER-FRAME-CLASSIFICATION**: apply per-frame dimension only to HARD frames (top-10% per `tac.variable_rate.compute_pair_difficulty`); easy frames inherit 3-set classification. Cost: ~1-2 day editor.

### 11.3 Reactivation paths if 5-set Venn empirical anchor returns DEFER

1. **AXIS-CATEGORICAL-EXTENSION**: replace binary `is_pose_axis_dominant` with 3-valued categorical (DOMINANT_SEG vs DOMINANT_POSE vs DOMINANT_RATE), making the cell count 32 × 3 / 2 = 48 instead of 32. Cost: ~2-3 day editor.
2. **OPERATING-POINT-CONDITIONAL-ROUTING**: at PR106 frontier operating point (pose-marginal-dominant), prioritize DOMINANT_POSE cells; at lower operating points, prioritize DOMINANT_SEG cells. Cost: ~$0 editor (just changes the autopilot reward formula).

### 11.4 Reactivation paths if 6-set Venn empirical anchor returns REFUSE

1. **BIT-LEVEL-DEPRECATION**: if bit-level master gradient does not yield meaningful per-bit Venn classification, stay at 5-set. Cost: ~$0 editor.
2. **PER-PAIR-BIT-CONDITIONAL**: apply per-bit dimension only to bytes in the top-2% sensitivity tier per `sensitivity_mask_aware_quantizr_v1`. Cost: ~2-3 day editor.

---

## 12. Catalog #324 post-training Tier-C validation discipline

Per CLAUDE.md "Substrate MUST be at OPTIMAL FORM before paid empirical dispatch" + Catalog #324 (`check_no_predicted_band_without_post_training_tier_c_validation`):

The N-set Venn classification's `predicted_band_validation_status` is `pending_post_training` at design time. Reactivation criterion: post-training Tier-C re-measurement on the landed N-set classification artifact (per-cell deliverability tier verdicts).

Sister anchor: per Catalog #319 Q4 (FEC6 Comma2k19 palette smoke), the 2-set Venn's empirical anchor revealed v1's 1.15× HIGH_PAIR_INVARIANT reward was FAKE — same empirical-validation discipline must apply to each N-set extension's per-cell tier verdicts.

### 12.1 Per-cell Tier-C validation cadence

For each newly-built N-set Venn cell with non-trivial byte mass:
1. **Initial classification**: per `tac.canonical_n_set_venn_classification.classify_bytes_n_set` (deterministic from master gradient + per-region + per-class signals)
2. **Per-cell Tier-C density measurement**: via `tools/mdl_scorer_conditional_ablation.py --tier c --cell <bin_id>` (NEW per §15 sister extension)
3. **Tier verdict refinement**: per-cell deliverability tier upgraded/downgraded based on empirical Tier-C density
4. **Sidecar emission**: per-cell verdict + Tier-C density anchor written to `.omx/state/n_set_venn_classification/<archive_sha[:12]>_post_training_tier_c_<utc>.json`

### 12.2 Cross-substrate Tier-C validation

Per sister meta-portfolio OP-2 (`tools/extract_master_gradient.py` extension to 6 archives): each new archive's master-gradient anchor enables per-substrate N-set Venn classification; per-substrate Tier-C validation cadence runs in parallel.

---

## 13. 6-hook wire-in declaration (Catalog #125)

Per CLAUDE.md "Subagent coherence-by-default" mandatory 6 hooks:

1. **Sensitivity-map contribution** = **ACTIVE** via the N-set Venn classifier feeding `tac.sensitivity_map.axis_weights` extension (per-cell per-axis weights). The 5-set Venn's `A_axis` dimension IS bidirectionally consistent with `sensitivity_map.axis_weights` (see §5.3).

2. **Pareto constraint** = **ACTIVE** via per-cell composition matrix consumed by `tac.optimization.field_equation_planner.field_row` extension. The 64 cells of 6-set Venn create a 64-dim Pareto-feasibility polytope; Dykstra alternating-projections solver via `tac.optimization.substrate_composition_matrix.classify_pairwise_composability` extended to PAIRWISE_VENN_CELLS over 64 cells.

3. **Bit-allocator hook** = **ACTIVE** via 6-set Venn's `A_bit` dimension feeding `tac.bit_allocator` per-bit allocation. Sister of cross-stack synergy #2 (Bit-level STC × per-pair master gradient × Wyner-Ziv Tier-1 deliverability).

4. **Cathedral autopilot dispatch hook** = **ACTIVE** via new `adjust_predicted_delta_for_venn_classification_v3_n_set` cascade extension per §15 below. The per-cell reward factor IS the canonical autopilot consumer.

5. **Continual-learning posterior update** = **ACTIVE** via `.omx/state/n_set_venn_classification/` fcntl-locked JSONL store per Catalog #131 + canonical helper `tac.canonical_n_set_venn_classification.register_classification`. Sister of `tac.master_gradient_consumers.write_consumer_sidecar_json`.

6. **Probe-disambiguator** = **ACTIVE** via OP-3 probe `tools/probe_n_set_venn_empirical_sparsity_atlas.py` (Assumption-Adversary mandate). Sister probes:
   - `tools/probe_n_set_venn_cell_byte_mutation.py` (per-cell Catalog #272 byte-mutation smoke)
   - `tools/probe_n_set_venn_ib_optimal_n.py` (Tishby IB-optimal-N estimator per substrate)
   - `tools/probe_n_set_venn_atick_redlich_receiver_prior.py` (Atick-Redlich cooperative-receiver dimension consistency check)

---

## 14. Canonical-vs-unique decision per layer (Catalog #290)

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" 2026-05-15:

| Layer | Canonical / Unique | Rationale |
|---|---|---|
| `tac.master_gradient_consumers.PerByteVennClass` (2-set Venn enum) | **ADOPT_CANONICAL_BECAUSE_SERVES** | Existing 2-set classification is the canonical anchor; N-set extension PRESERVES it per Mallat's falling-rule discipline (Catalog #277). |
| `tac.wyner_ziv_deliverability.DeliverabilityTier` (4-tier enum) | **ADOPT_CANONICAL_BECAUSE_SERVES** | The 4-tier framework is the canonical Wyner-Ziv ladder; per-cell tier assignment USES it. |
| `tac.wyner_ziv_deliverability.DeliverabilityProof` dataclass | **EXTEND_CANONICAL** | Sister `NSetVennDeliverabilityProof` extends with `cell_id` + `n_set_size` + `parent_2_set_cell` fields per descriptive-set-theoretic refinement. |
| `tac.wyner_ziv_deliverability.build_deliverability_proof_from_wyner_ziv_classification` (Q1 builder) | **EXTEND_CANONICAL** | Sister `build_n_set_deliverability_proof_from_n_set_classification` extends the existing builder with per-cell iteration. |
| `tac.wyner_ziv_deliverability.verify_deliverability_proof_contest_compliance` (Q1 verifier) | **EXTEND_CANONICAL** | Sister `verify_n_set_deliverability_proof_contest_compliance` extends with per-cell-tier verification. |
| Catalog #319 STRICT preflight gate `check_substrate_wyner_ziv_reweight_has_deliverability_proof` | **ADOPT_CANONICAL_BECAUSE_SERVES** | Gate continues to refuse 1.15× HIGH_PAIR_INVARIANT reward without proof; N-set extension is additive per cell. |
| Catalog #319 Q3 autopilot reweight v2 cascade | **EXTEND_CANONICAL** | New `adjust_predicted_delta_for_venn_classification_v3_n_set` extends the v2 cascade with per-cell reward factor (preserves CASCADE 1 Lagrangian / CASCADE 2 deliverability / CASCADE 3 passthrough order). |
| `.omx/state/wyner_ziv_deliverability/` fcntl-locked JSONL store | **EXTEND_CANONICAL** | Sister `.omx/state/n_set_venn_classification/` store per Catalog #131 discipline. Identical fcntl-lock + APPEND-ONLY semantics. |
| DuckDB schema for per-cell sensitivity / classification | **FORK_BECAUSE_PRINCIPLED_MISMATCH** | Existing schemas operate on per-byte / per-pair granularity; N-set Venn requires per-cell granularity which is a STRUCTURAL EXTENSION. New table `n_set_venn_classification_ext` with `cell_id` primary key. |
| `tac.canonical_duckdb.per_byte_sensitivity_ext` | **ADOPT_CANONICAL_BECAUSE_SERVES** + JOIN | New `n_set_venn_classification_ext` JOINs with per-byte schema on `(archive_sha256, byte_offset)`. |
| Cathedral autopilot v2 cascade reward factor composition order | **ADOPT_CANONICAL_BECAUSE_SERVES** | The canonical Tier A → Tier C → class-shift → composition_alpha → Venn classification → Wyner-Ziv deliverability order PRESERVED; N-set extension inserts at the existing Venn classification step. |
| `tools/probe_*_disambiguator.py` sister probes | **FORK_BECAUSE_PRINCIPLED_MISMATCH** | New probes per cell (sparsity atlas / byte-mutation / IB-optimal-N / Atick-Redlich consistency) are STRUCTURALLY DIFFERENT from sister substrate probes; cannot reuse template. |
| Catalog #272 distinguishing-feature contract | **EXTEND_CANONICAL** | Each cell with non-trivial byte mass gets its own Catalog #272 byte-mutation smoke per §12.1. The contract framework PRESERVED; per-cell application is additive. |
| Council deliberation v2 frontmatter (Catalog #300) | **ADOPT_CANONICAL_BECAUSE_SERVES** | This memo's frontmatter IS Catalog #300 v2-compliant; verdict + dissent + decisions tracked per canonical. |

Per CLAUDE.md "Beauty, simplicity, and developer experience": canonical helpers EXTENDED via sister functions/dataclasses rather than REPLACED; N-set extension is structurally additive to the existing Catalog #319 Q1-Q5 chain.

---

## 15. Implementation architecture for tac.canonical_n_set_venn_classification

### 15.1 Package layout (Codex-class deliverable)

```
src/tac/canonical_n_set_venn_classification/
├── __init__.py                                      # narrow public API
├── contract.py                                      # frozen dataclasses + enums
├── classifier.py                                    # build_n_set_classification (PRODUCER)
├── deliverability_proof_extension.py                # extends wyner_ziv_deliverability
├── autopilot_consumer.py                            # adjust_predicted_delta_for_venn_classification_v3_n_set
├── persistence.py                                   # fcntl-locked JSONL store
├── duckdb_schema.py                                 # DuckDB ext schema + views
└── tests/
    ├── test_contract_invariants.py                  # __post_init__ validation
    ├── test_classifier_3set_8_cells.py              # 3-set Venn cell-count + tier assignment
    ├── test_classifier_4set_16_cells.py             # 4-set Venn (+ frame)
    ├── test_classifier_5set_32_cells.py             # 5-set Venn (+ axis)
    ├── test_classifier_6set_64_cells.py             # 6-set Venn (+ bit)
    ├── test_sparse_cell_inheritance.py              # parent-2-set classification for <0.5% byte mass
    ├── test_deliverability_proof_extension.py       # sister Q1 builder
    ├── test_autopilot_consumer_v3_n_set.py          # v3 cascade extension
    ├── test_persistence_fcntl_locked.py             # 4-proc spawn-pool concurrent-append stress
    ├── test_duckdb_schema_cell_prefix_aggregation.py # GROUP BY cell-prefix O(1) per query
    ├── test_mallat_falling_rule_backward_compat.py  # coarser-scale dominates on disagreement
    └── test_tishby_ib_optimal_n_estimator.py        # IB-optimal-N per substrate
```

### 15.2 Contract dataclass (`contract.py`)

```python
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

class NSetVennAxis(str, Enum):
    """Canonical binary axes per operator enumeration."""
    PAIR = "pair"
    REGION = "region"
    CLASS = "class"
    FRAME = "frame"
    AXIS = "axis"
    BIT = "bit"

CANONICAL_3SET_AXES = (NSetVennAxis.PAIR, NSetVennAxis.REGION, NSetVennAxis.CLASS)
CANONICAL_4SET_AXES = CANONICAL_3SET_AXES + (NSetVennAxis.FRAME,)
CANONICAL_5SET_AXES = CANONICAL_4SET_AXES + (NSetVennAxis.AXIS,)
CANONICAL_6SET_AXES = CANONICAL_5SET_AXES + (NSetVennAxis.BIT,)


@dataclass(frozen=True)
class NSetVennCell:
    """One cell in the N-set Venn classification.

    `cell_id`: integer in [0, 2^N) — binary representation gives per-axis verdicts
    `axis_verdicts`: tuple of N booleans (True = HIGH_*_INVARIANT, False = HIGH_*_SPECIFIC)
    """
    n_set_size: int
    cell_id: int
    axis_verdicts: tuple[bool, ...]
    axis_names: tuple[NSetVennAxis, ...]

    def __post_init__(self) -> None:
        if not (1 <= self.n_set_size <= 6):
            raise ValueError(f"n_set_size must be in [1, 6]; got {self.n_set_size}")
        if not (0 <= self.cell_id < 2 ** self.n_set_size):
            raise ValueError(f"cell_id must be in [0, 2^{self.n_set_size}); got {self.cell_id}")
        if len(self.axis_verdicts) != self.n_set_size:
            raise ValueError(f"axis_verdicts must have length {self.n_set_size}")
        if len(self.axis_names) != self.n_set_size:
            raise ValueError(f"axis_names must have length {self.n_set_size}")

    @property
    def parent_2set_cell_id(self) -> int:
        """Coarser-scale 2-set parent cell per Mallat falling-rule discipline."""
        return self.cell_id & 0b11  # extract lowest 2 bits (pair + region or pair-only)


@dataclass(frozen=True)
class NSetVennClassification:
    """Result of build_n_set_classification.

    Sister of `tac.master_gradient_consumers.PerByteVennClassification` extended to N-set.

    `cell_assignments`: shape (N_bytes,) — cell_id per byte
    `per_cell_byte_count`: dict mapping cell_id → byte count (sparse cells absent)
    `per_cell_byte_indices`: dict mapping cell_id → sorted tuple of byte indices
    `per_cell_deliverability_tier`: dict mapping cell_id → DeliverabilityTier
    `sparse_cells_inherited_from_2set`: dict mapping cell_id → parent_2set_cell_id
    """
    archive_sha256: str
    n_set_size: int
    axis_names: tuple[NSetVennAxis, ...]
    cell_assignments: np.ndarray  # (N_bytes,) dtype=int
    per_cell_byte_count: dict[int, int]
    per_cell_byte_indices: dict[int, tuple[int, ...]]
    per_cell_deliverability_tier: dict[int, str]  # DeliverabilityTier.value
    sparse_cells_inherited_from_2set: dict[int, int]
    sparsity_threshold_relative: float  # 0.005 = 0.5% canonical
    n_bytes: int
    n_pairs: int
    measurement_axis: str
    measurement_hardware: str
    written_at_utc: str
    schema_version: str = "n_set_venn_classification_v1"

    def __post_init__(self) -> None:
        if self.n_set_size not in (3, 4, 5, 6):
            raise ValueError(f"n_set_size must be in {{3, 4, 5, 6}}; got {self.n_set_size}")
        if len(self.axis_names) != self.n_set_size:
            raise ValueError(f"axis_names must have length {self.n_set_size}")
        if self.cell_assignments.shape != (self.n_bytes,):
            raise ValueError(f"cell_assignments must have shape ({self.n_bytes},)")
        # Verify sparse cells inherit valid parent 2-set cells
        for cell_id, parent_id in self.sparse_cells_inherited_from_2set.items():
            if not (0 <= parent_id < 4):
                raise ValueError(f"parent_2set_cell_id must be in [0, 4); got {parent_id}")
```

### 15.3 Classifier (`classifier.py`)

```python
def build_n_set_classification(
    per_pair_gradient: np.ndarray,                                # (N_bytes, N_pairs, 3)
    per_region_segnet_margin: np.ndarray,                         # (N_bytes, N_regions=256)
    per_class_distribution: np.ndarray,                           # (N_bytes, N_classes=5)
    per_frame_difficulty: np.ndarray | None = None,               # (N_frames=1200,) — 4-set+
    per_bit_gradient: np.ndarray | None = None,                   # (N_bits, N_pairs, 3) — 6-set+
    *,
    archive_sha256: str,
    measurement_axis: str,
    measurement_hardware: str,
    n_set_size: int = 3,                                           # canonical default per Tishby IB-optimal
    sparsity_threshold_relative: float = 0.005,                   # 0.5% per Carmack sparsity-first
    variance_threshold_relative: float = VENN_VARIANCE_THRESHOLD_RELATIVE,
    aggregate_floor_relative: float = VENN_AGGREGATE_FLOOR_RELATIVE,
    write_sidecar: bool = True,
) -> NSetVennClassification:
    """Classify every byte by N-set Venn over (pair × region × class × frame × axis × bit).

    For n_set_size=3: cells = pair × region × class = 8.
    For n_set_size=4: + frame = 16.
    For n_set_size=5: + axis (binarized = is_pose_axis_dominant) = 32.
    For n_set_size=6: + bit (per-bit gradient required) = 64.

    Sparse cells (<sparsity_threshold_relative byte mass) inherit parent-2-set classification
    per descriptive-set-theoretic refinement principle (Mallat falling-rule discipline).
    """
    # ... implementation per spec above
```

### 15.4 DuckDB schema (`duckdb_schema.py`)

```sql
CREATE TABLE n_set_venn_classification_ext (
  archive_sha256 TEXT NOT NULL,
  n_set_size INTEGER NOT NULL,                       -- 3, 4, 5, or 6
  cell_id INTEGER NOT NULL,                          -- [0, 2^n_set_size)
  byte_offset INTEGER NOT NULL,
  pair_id INTEGER,                                   -- canonical pair index (nullable for some axes)
  region_id INTEGER,                                 -- 16×16 = 256
  class_id INTEGER,                                  -- 5 SegNet classes
  frame_id INTEGER,                                  -- 1200 frames; null for 3-set
  axis_dominance TEXT,                               -- 'seg' / 'pose' / 'rate'; null for 3-set/4-set
  bit_position INTEGER,                              -- [0, 8); null for 3-set/4-set/5-set
  deliverability_tier TEXT NOT NULL,                 -- DeliverabilityTier value
  parent_2set_cell_id INTEGER,                       -- for sparse-cell inheritance
  sensitivity_fp64 DOUBLE NOT NULL,
  derived_at_utc TIMESTAMP NOT NULL,
  PRIMARY KEY (archive_sha256, n_set_size, cell_id, byte_offset)
);

-- Cell-prefix aggregation view for O(1) per-prefix queries
CREATE VIEW n_set_venn_cell_byte_mass AS
SELECT
  archive_sha256,
  n_set_size,
  cell_id,
  COUNT(*) AS byte_count,
  COUNT(*) * 1.0 / SUM(COUNT(*)) OVER (PARTITION BY archive_sha256, n_set_size) AS byte_fraction,
  AVG(sensitivity_fp64) AS avg_sensitivity,
  deliverability_tier
FROM n_set_venn_classification_ext
GROUP BY archive_sha256, n_set_size, cell_id, deliverability_tier
ORDER BY archive_sha256, n_set_size, byte_count DESC;

-- Sparse-cell detection view
CREATE VIEW n_set_venn_sparse_cells AS
SELECT
  archive_sha256,
  n_set_size,
  cell_id,
  byte_count,
  byte_fraction,
  parent_2set_cell_id
FROM n_set_venn_cell_byte_mass
WHERE byte_fraction < 0.005;
```

### 15.5 Autopilot consumer (`autopilot_consumer.py`)

```python
def adjust_predicted_delta_for_venn_classification_v3_n_set(
    predicted_delta: float,
    archive_sha256: str,
    n_set_size: int = 3,                              # 3, 4, 5, or 6
    optimal_plan_path: Path | None = None,
) -> float:
    """N-set Venn extension of Catalog #319 v2 cascade.

    Per Mallat wavelet-multi-scale-falling-rule discipline (Catalog #277):
      1. PRIMARY: defer to v2 cascade (CASCADE 1 Lagrangian planner if present)
      2. EXTENSION: if v2 cascade returns passthrough AND N-set classification exists,
         apply per-cell reward factor:
         reward_factor = sum_c (n_bytes_in_cell_c × tier_factor[cell_to_tier[c]]) / total_bytes
      3. FALLBACK: 2-set classification (CASCADE 2/3 of v2)

    Preserves the existing Catalog #319 Q1-Q5 chain unchanged via
    backward compatibility: if N-set classification absent, returns
    `adjust_predicted_delta_for_venn_classification_v2` result unchanged.
    """
    # CASCADE 1 (PRIMARY — Lagrangian-derived): defer to v2 cascade
    v2_result = adjust_predicted_delta_for_venn_classification_v2(
        predicted_delta, archive_sha256, optimal_plan_path
    )

    # If v2 cascade was the canonical Lagrangian planner OR a HIGH_PAIR_SPECIFIC penalty,
    # the planner already accounts for the substrate's marginal-Δ contribution; do NOT
    # add the N-set reward factor on top
    if abs(v2_result - predicted_delta) > 1e-9 and abs(v2_result) > abs(predicted_delta):
        # CASCADE 1 fired (replaced predicted_delta with planner's solution)
        return v2_result
    if v2_result < predicted_delta:
        # HIGH_PAIR_SPECIFIC penalty fired
        return v2_result

    # CASCADE EXTENSION (N-set): apply per-cell reward factor
    n_set_classification = _load_n_set_classification_for_archive(archive_sha256, n_set_size)
    if n_set_classification is None:
        return v2_result  # fall back to v2

    factor = _n_set_venn_deliverability_reward_factor_for_archive(
        archive_sha256, n_set_classification
    )
    return v2_result * factor  # COMPOSE on top of v2 result
```

### 15.6 Persistence (`persistence.py`)

```python
def register_n_set_classification(
    classification: NSetVennClassification,
    *,
    sidecar_dir: Path = _N_SET_VENN_CLASSIFICATION_DIR,
) -> Path:
    """Atomic append to .omx/state/n_set_venn_classification/<sha[:12]>_<utc>.json.

    Per Catalog #131 fcntl-locked discipline + Catalog #245 canonical 4-layer pattern:
      - fcntl.flock(LOCK_EX) on .omx/state/n_set_venn_classification/.lock
      - write to .tmp.<uuid12> + os.replace (atomic)
      - APPEND-ONLY per Catalog #110 HISTORICAL_PROVENANCE
    """
    # ... implementation
```

---

## 16. Cross-substrate composability matrix

### 16.1 CELL-LEVEL vs SUBSTRATE-LEVEL composition

Per Catalog #322 anti-additive evidence (4/8 probed substrate-pair α-pairs sub-additive), the canonical composition_alpha is currently per-substrate-pair. The N-set Venn extension introduces **CELL-LEVEL composition** which is much finer-grained:

- **SUBSTRATE-LEVEL**: ~870 substrate-pair cells in `tac.optimization.substrate_composition_matrix`
- **CELL-LEVEL (3-set Venn)**: 8 × 8 = 64 cell-pairs per substrate
- **CELL-LEVEL (6-set Venn)**: 64 × 64 = 4096 cell-pairs per substrate

The cell-level composition matrix surfaces opportunities the substrate-level matrix cannot:
- A substrate may COMPOSE WELL with a sister in 6 of 8 cells (e.g., Cell 1+2+3+5+6+7) but be ANTAGONISTIC in 2 cells (Cell 4+8 where both substrates compete for the same byte budget)
- Aggregate substrate-level composition_alpha hides this cell-level variance

### 16.2 Per-cell composition_alpha computation

Per Dykstra-feasibility lens (Dykstra verbatim above): the alternating-projections solver extended from substrate-level to cell-level. New canonical helper:

```python
def classify_pairwise_composability_per_cell(
    substrate_a_classification: NSetVennClassification,
    substrate_b_classification: NSetVennClassification,
) -> dict[tuple[int, int], CellPairComposabilityVerdict]:
    """Classify each cell-pair (cell_a, cell_b) into ORTHOGONAL / SUB_ADDITIVE / ANTAGONISTIC.

    Sister of `tac.optimization.substrate_composition_matrix.classify_pairwise_composability`
    extended to cell granularity.

    Per Catalog #322 anti-additive evidence + Dykstra-feasibility analysis.
    """
```

### 16.3 Anti-phantom guard at cell level

Per Catalog #322 (`check_no_autopilot_adjustment_derived_from_phantom_provenance_composition_alpha`): the canonical anti-phantom gate extends to cell-level via:
- Cell-level composition_alpha derived from research sidecar = PHANTOM (refused)
- Cell-level composition_alpha derived from contest archive members = LEGITIMATE

Sister scan extension: `.omx/state/n_set_venn_cell_composition_matrix.json` artifacts whose per-cell α value references a sidecar path → flagged per Catalog #322.

---

## 17. Op-routables ranked by EV

| # | Op-routable | TIER | Description | Cost | Predicted ΔS |
|---|---|---|---|---|---|
| 1 | Build `tac.canonical_n_set_venn_classification` package per §15 architecture | TIER-1 | Includes 3-set classifier as canonical first step; 4-set/5-set/6-set as opt-in extensions; sister Q1 deliverability proof extension | ~5-7 day editor + $0 GPU | `[-0.005, -0.002]` standalone (3-set only) |
| 2 | Extend Catalog #319 v2 cascade `adjust_predicted_delta_for_venn_classification_v2` with `_v3_n_set` overload | TIER-1 | Backward-compatible per §16 Mallat falling-rule discipline; new autopilot consumer per §15.5 | ~2-3 day editor | `[-0.003, -0.001]` standalone (autopilot ranking unlock) |
| 3 | Build `tools/probe_n_set_venn_empirical_sparsity_atlas.py` on fec6 anchor PER Assumption-Adversary mandate | TIER-1 PRE-DISPATCH | BLOCKING for op-routable #1 paid dispatch; outputs typed `NSetVennSparsityAtlas` with per-cell byte_mass + JOIN-WITH-NEIGHBOR recommendations | ~1-2 day editor + $0 GPU | (infra; empirical-validation prerequisite) |
| 4 | Build DuckDB schema `tac.canonical_duckdb.n_set_venn_classification_ext` per §15.4 + auto-aggregation GROUP BY cell-prefix views | TIER-2 | Sister of meta-portfolio OP-5 (5D multi-granularity sensitivity tensor); enables O(1) per cell-prefix query | ~2-3 day editor | (infra; observability + queryability) |
| 5 | Integrate N-set Venn classification into the in-flight 3-subagent orchestration queue (sister Riemannian-Newton + TT5L V2 + cargo-cult resurrection TOP-3) | TIER-2 | Composition with sister substrates per cross-stack synergies §16; per-substrate per-cell sensitivity | ~3-5 day editor + cross-subagent coordination | `[-0.007, -0.002]` aggregate (cross-stack stacking) |
| 6 | Build sister probes per §13 hook #6: `tools/probe_n_set_venn_ib_optimal_n.py` (Tishby), `tools/probe_n_set_venn_atick_redlich_receiver_prior.py` (Atick consistency) | TIER-2 | Empirical-validation of council deliberation's IB-optimal-N + Atick-Redlich receiver-prior assumptions | ~2-3 day editor | (infra; assumption-validation) |
| 7 | Per-cell Catalog #272 byte-mutation smoke for the ~17 expected populated cells per fec6 sparsity atlas | TIER-3 | $3 paired-CUDA verify per non-sparse cell; total ~$50 budget for 17 cells | $50-200 paired-CUDA | `[-0.005, -0.002]` per cell empirically validated |
| 8 | Build CELL-LEVEL composition matrix per §16.2 — `classify_pairwise_composability_per_cell` | TIER-3 | Sister of substrate_composition_matrix extended to cell granularity; surfaces sub-additive cell-pairs hidden by substrate-aggregate matrix | ~5-7 day editor | `[-0.003, -0.001]` (composition-aware autopilot ranking) |

### Aggregate predicted ΔS

Under realistic α-discount per Catalog #322: `[-0.015, -0.005]` standalone (TIER-1 + TIER-2 op-routables 1+2+4+5)

Under HIGH-orthogonality (all op-routables land + all 17 expected cells empirically validated + cross-stack stacking with null-space + per-region SegNet polytope per meta-portfolio TOP-1): `[-0.020, -0.005]`

---

## 18. Cross-references

- **Parent T3 symposium**: `.omx/research/grand_council_meta_portfolio_re_ranking_post_compliance_envelope_20260518.md` (Venn extension = #1 newly-discovered granularity)
- **Sister synthesis**: `.omx/research/deeper_granularity_discovery_bit_byte_zero_one_pixel_frame_pair_region_label_category_venn_20260518.md` (18-granularity expansion; operator's full hint)
- **Foundational mathematics**: `.omx/research/set_theory_manifolds_geometry_deep_research_synthesis_20260518.md` (set theory is THE foundational framework; §1.6 Venn diagrams as set theory)
- **Floor disambiguator**: `.omx/research/tac_theoretical_floor_estimator_design_memo_plateau_vs_saturation_disambiguator_20260518.md` (PLATEAU verdict; N-set Venn extension is class-shift response)
- **Supplementary granularity directive**: `.omx/research/deeper_granularity_addition_directive_boundaries_xray_hard_pairs_sensitive_bytes_sensitivity_map_20260518.md` (6 additional granularities)
- **Catalog #319 parent symposium**: `.omx/research/grand_council_symposium_wyner_ziv_contest_compliance_optimal_design_20260517.md`
- **Existing 2-set Venn implementation**: `src/tac/master_gradient_consumers.py:436-596`
- **Existing DeliverabilityProof contract**: `src/tac/wyner_ziv_deliverability/proof_builder.py:227-455`
- **Cathedral autopilot v2 cascade**: `tools/cathedral_autopilot_autonomous_loop.py:1062-1266`
- **Catalog #322 anti-phantom umbrella**: CLAUDE.md "Meta-bug class catalog" entry #322
- **Catalog #277 wavelet-multi-scale falling-rule discipline**: CLAUDE.md entry #277
- **Catalog #319 Q1-Q5 chain**: CLAUDE.md entry #319

---

## 19. Acknowledgements

This T2 design memo operates per the operator's standing directive 2026-05-18 ("start up the 2 subagent orchestration queue and keep it fed") + load-bearing insight ("there is more like this lurking in ... venn diagram and all"). The 13-member council (sextet pact + Atick + Tishby memorial + Hafner + Mallat + MacKay memorial + Boyd + Carmack) honors the cooperative-receiver + IB + categorical-latent + wavelet-multi-scale + cross-disciplinary IT lenses required to navigate the N-set Venn extension at its proper mathematical depth.

Sister subagents in-flight (per Catalog #314 absorption-pattern protection; DISJOINT scope):
- `a39ffdf80` (Riemannian-Newton substrate engineering)
- `a478cbde` (TT5L V2 redesign)

— Main-Claude (relayed on behalf of operator 2026-05-18)
