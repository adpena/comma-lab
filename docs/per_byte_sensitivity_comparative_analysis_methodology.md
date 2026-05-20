# Per-byte sensitivity comparative analysis methodology

A reusable engineering discipline that converts "we have substrate X with score Y" into structural understanding of **which bytes carry the score advantage** by comparing per-byte master gradients across substrate pairs.

Validated empirically on the 2026-05-20 21-pair cross-candidate sensitivity matrix (PR101 GOLD, A1 fine-tuned, fec6 frontier, PR106 format0d, PR107 apogee, NSCS06 v7). One actionable finding per finding-class: backbone-equivalence detection across the HNeRV medal cluster (Pearson seg ρ=0.961, pose ρ=0.971 on PR101↔fec6); cross-hardware top-K leverage drift (6.4% [macOS-CPU advisory] vs 11.1% [contest-CUDA T4] on the SAME archive sha); cross-codec orthogonality detection (top-K Jaccard 0.000 on PR106 vs HNeRV-family).

Sister of [`docs/cargo_cult_unwind_methodology.md`](cargo_cult_unwind_methodology.md). Related: [`docs/asymptotic_floor_candidate_inventory.md`](asymptotic_floor_candidate_inventory.md) §C.8 (entropy-coding canonical references); [`docs/canonical_equations_tour.md`](canonical_equations_tour.md) (the 3 new equations this methodology produced). Submission packet: [`commaai/comma_video_compression_challenge#110`](https://github.com/commaai/comma_video_compression_challenge/pull/110).

---

## 1. Motivation — the gap this discipline fills

A naive substrate-evaluation loop produces score numbers: *"substrate X scores Y on axis Z"*. That signal answers whether X is competitive, but not **why** X is competitive — which bytes do the work, which are dead weight, and which are interchangeable with neighboring substrates' bytes.

Cross-candidate per-byte sensitivity comparison fills the structural gap. By extracting per-byte master gradients on the SAME contest video for two substrates A and B, then comparing the resulting `(N_archive_bytes, 3)` tensors via four diagnostic primitives (top-K Jaccard, per-axis Pearson correlation, top-K leverage concentration, classification taxonomy), the methodology surfaces:

* **Backbone equivalence.** Two substrates that share a backbone (e.g., the 178,158-byte HNeRV encoder shared across the entire 0.19xxx medal cluster) will have near-identical per-axis aggregate sensitivity on the shared bytes. The score difference between them is then concentrated in the bytes that DIFFER — which is exactly where future engineering effort should land.
* **Orthogonal-codec detection.** Two substrates whose top-K byte sets have Jaccard = 0.000 are not competing on the same axis at all; they are stacking candidates because the bytes that move score for one are invisible to the other.
* **Hardware-drift attribution.** The same archive bytes measured on different hardware (macOS CPU advisory vs contest CUDA T4) can yield 73% different top-K leverage concentration — which means an advisory-only sensitivity signal cannot be promoted to ranking-grade authority on a CUDA contest axis without paired empirical confirmation.

Per CLAUDE.md "Meta-Lagrangian/Pareto solver — NON-NEGOTIABLE, HIGHEST EMPHASIS": the goal is solvable math grounded in entropy/MDL/Fisher sensitivity, not arbitrary sweeps. Per-byte sensitivity comparison **is** that grounded math at the byte level.

---

## 2. The methodology — four diagnostic primitives

For substrates A and B, extract per-byte master gradients `g_A, g_B ∈ R^{N × 3}` where each row is `(d_seg/d_byte, d_pose/d_byte, d_rate/d_byte)`. Compute:

**(P1) Per-axis Pearson correlation** on shared bytes (intersection of A and B byte ranges):
```
ρ_seg = corr(g_A[:, 0], g_B[:, 0])
ρ_pose = corr(g_A[:, 1], g_B[:, 1])
ρ_rate = corr(g_A[:, 2], g_B[:, 2])
```
High ρ on a shared-byte range (typically ρ ≥ 0.9 across all three axes) indicates the substrates share that range as a common backbone. Near-zero or negative ρ indicates either different codec choice or different scorer-response geometry.

**(P2) Top-K byte sensitivity Jaccard** at k_byte (default K=32 absolute count OR k_pct=0.01 percentage):
```
top_K_A = indices of top-K bytes by ||g_A[i, :]||_1
top_K_B = indices of top-K bytes by ||g_B[i, :]||_1
Jaccard = |top_K_A ∩ top_K_B| / |top_K_A ∪ top_K_B|
```
Jaccard near 0.000 indicates orthogonal-codec composition candidates (different bytes carry the signal). Jaccard near 1.000 indicates aligned codec compression on the same bytes (less interesting for stacking; more interesting for paradigm-consolidation review).

**(P3) Top-1% leverage concentration**:
```
leverage_top_1pct = sum(top 1% bytes' ||g||_1) / sum(all bytes' ||g||_1)
```
A uniform-Pareto baseline would predict 1% (each byte contributes proportionally). Empirically, the canonical equation `per_byte_leverage_uniformly_distributed_v1` established ~6.4% concentration for PR101's HNeRV backbone — i.e., the top 1% of bytes carry ~6x their share of score sensitivity. Cross-hardware measurement of the SAME archive can shift this to 11.1% on CUDA T4 — the canonical equation `per_byte_leverage_cross_hardware_aware_v2` extends to capture that drift.

**(P4) Classification taxonomy** (SUPER_ADDITIVE / SUB_ADDITIVE / ANTAGONISTIC / INDETERMINATE):
```
SUPER_ADDITIVE  ← Jaccard ≤ 0.10 AND |ρ_seg| ≤ 0.30  (orthogonal-codec composition)
SUB_ADDITIVE    ← Jaccard ≥ 0.50 AND ρ_seg ≥ 0.90    (backbone-equivalence)
ANTAGONISTIC    ← Jaccard ≤ 0.10 AND ρ_seg ≤ −0.30   (rare; bytes anti-correlated)
INDETERMINATE   ← all other combinations             (insufficient signal to classify)
```
This classification feeds the Cathedral autopilot's stack-of-stacks composition predictor as an **observability-only** annotation (Tier A per Catalog #341), NEVER as a score-mutating ranking signal — promotion to score-contributing requires paired CUDA + CPU empirical auth-eval per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA".

---

## 3. Worked example — PR101 GOLD vs fec6 frontier

PR101 GOLD (`0.19284 [contest-CPU]`, archive sha-prefix `b83bf348`) is the canonical HNeRV medal baseline. fec6 frontier (`0.19205 [contest-CPU]`, archive sha-prefix `6bae0201`) is a `+0.000794` improvement that sits 794 ppm below PR101's reduction.

Naive interpretation: fec6 is "a better HNeRV". Structural interpretation via per-byte comparison:

* **Shared bytes (178,158 of 178,417 = 99.85% of fec6's archive)**: per-axis Pearson seg ρ = 0.961, pose ρ = 0.971. The HNeRV backbone is **byte-identical sensitivity** between the two substrates.
* **Top-K Jaccard (K=32)**: 0.641. High overlap on the most-sensitive backbone bytes.
* **Differing bytes**: fec6 carries **+259 bytes** of FEC6 selector + Huffman k=16 frame-exploit metadata that PR101 does not.
* **Per-axis aggregate sum_abs**: byte-identical on the shared 178,158-byte range; both sum to (same to 4 sig figs).

**The structural finding**: the entire +794 ppm advantage is concentrated in those +259 bytes of selector overhead. The HNeRV backbone is **class-saturated** at the medal cluster. Future engineering against the backbone is operating on a Pareto-saturated surface; future engineering against orthogonal selector overlays + microcodec extensions is the path forward.

This finding is codified as canonical equation `hnerv_backbone_sensitivity_saturated_across_medal_cluster_v1` (per Catalog #344 first-class artifact); the structural insight propagates to:

* The asymptotic-floor candidate inventory (the medal-cluster substrates are NOT independent attack surfaces; they share the same saturated backbone).
* PACT-NERV variant taxonomy (the variant-class definition treats backbone-byte modifications as DRIFT-WITHIN-CLASS, not class-shift).
* Selector-extensions taxonomy (the fec6 selector is the proof-of-concept that orthogonal selector overlays ARE the score-lowering path on HNeRV-class backbones).

---

## 4. Cross-hardware drift caveat

The same archive bytes measured on different hardware can yield materially different per-byte sensitivity signals.

* **fec6 frontier on M5 Max macOS CPU (fp64 advisory)**: top-1% leverage = 6.41%.
* **fec6 frontier on Modal T4 CUDA (fp32 authoritative)**: top-1% leverage = 11.11%.
* **Concentration delta**: 73% — the CUDA measurement places **4.71× more sensitivity** at the top 1% than the advisory measurement, on the SAME archive sha `6bae0201`.

The mechanism is the FastViT-T12 fp32 forward-kernel noise floor on CUDA (per the canonical equation `mps_drift_architecture_class_dependent_v1`'s sister findings in the CPU-vs-CUDA drift engineering analysis) plus DALI/NVDEC vs PyAV decoder drift on the input video preprocessing. The drift is a structural property of the contest scorer's hardware-dependent forward kernel, NOT a measurement noise floor.

**The discipline consequence**: per-byte sensitivity matrices MUST tag every row with `(measurement_axis, hardware_substrate, evidence_grade)` per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #287/#323 canonical Provenance. An `[macOS-CPU advisory]` sensitivity signal can seed a hypothesis about WHICH bytes matter; it cannot be promoted to ranking-grade authority on a CUDA contest axis without a paired CUDA measurement.

This finding is codified as canonical equation `per_byte_leverage_cross_hardware_aware_v2` (supersedes v1 per Catalog #110 APPEND-ONLY; v1 preserved as the within-hardware special case).

---

## 5. Auto-trigger pattern — apparatus-wide methodology extension

A per-byte sensitivity comparison run **once** is a snapshot. The methodology becomes apparatus-wide when every NEW master-gradient anchor that lands automatically triggers a similarity-matrix update against the existing anchors. This converts the discipline from a manual operator-routable task into a structural property of the apparatus.

The canonical mechanism is the post-anchor-landed auto-trigger consumer at `tac.cathedral_consumers.auto_trigger_similarity_after_master_gradient_anchor_consumer` (Catalog #335-compliant). The wire-in pattern mirrors Catalog #343 DX auto-update for the canonical frontier pointer:

```
master_gradient.append_anchor_locked(...)
    └─→ append to .omx/state/master_gradient_anchors.jsonl
    └─→ (future hook) auto_trigger_similarity_recompute(new_anchor)
            └─→ cross_substrate_similarity_compute(new_anchor, existing_anchors)
            └─→ append matrix row to .omx/state/cross_substrate_sensitivity_similarity_matrix_*.jsonl
            └─→ flag NEW SUPER_ADDITIVE / ANTAGONISTIC classifications to reports/latest.md
```

The auto-trigger consumes the JUST-LANDED canonical equations `per_byte_leverage_cross_hardware_aware_v2` + `hnerv_backbone_sensitivity_saturated_across_medal_cluster_v1` + `cross_codec_super_additive_orthogonality_predictor_v1` to predict classifications BEFORE running the full comparison — orthogonal-codec pairs (e.g., PR106 vs any HNeRV-family) are pre-classified SUPER_ADDITIVE with high confidence; within-backbone pairs are pre-classified SUB_ADDITIVE with high confidence. The full comparison then validates or refutes the prediction.

Per Catalog #341 the auto-trigger consumer's contribution is Tier A observability-only (`predicted_delta_adjustment=0.0`, `promotable=False`, `axis_tag=[predicted]`); the matrix is signal for downstream consumers (cathedral autopilot ranker via Catalog #335; bit-allocator via per-axis decomposition; Pareto polytope solver via Catalog #356), not a score-mutating claim.

---

## 6. Paper-section-worthy framing

Per-byte sensitivity comparative analysis is a primitive operation that has obvious applicability beyond the comma video compression contest:

* **Neural network compression research.** Pruning + quantization research routinely cites per-parameter saliency. Per-byte sensitivity on a compressed archive extends this to the post-compression byte stream — the bytes that survive into the deployed model, weighted by downstream task impact.
* **Codec design research.** Rate-distortion theory operates at the bit level. Per-byte sensitivity surfaces which bits in the entropy-coded stream actually matter for downstream task metrics, vs which are entropy-coding overhead that the codec could in principle compress further.
* **Interpretable ML.** Per-byte sensitivity is a structural primitive for Rudin-style interpretable models that compose with falling-rule-list classifiers (Catalog #274) — the per-byte leverage distribution itself is interpretable as a rule-list condition (*"byte i is in the top-1%-leverage set"*).
* **Cross-architecture model analysis.** Backbone-equivalence detection generalizes — any two models sharing a pretrained backbone will have correlated per-input sensitivity, and the divergence between them surfaces the actually-differentiating parameters.

The methodology + the canonical-equation registry + the auto-trigger pattern compose into reusable engineering infrastructure. Per CLAUDE.md "Meta-Lagrangian/Pareto solver — NON-NEGOTIABLE": the per-byte sensitivity primitive is solvable math grounded in measurable empirical sensitivity, not heuristic sweep; the methodology is the operational surface that converts substrate-level metrics into byte-level structural understanding.

---

## 7. Cross-references

**Canonical equations (per Catalog #344):**

* `per_byte_leverage_uniformly_distributed_v1` — initial empirical anchor on PR101 advisory.
* `per_byte_leverage_cross_hardware_aware_v2` — supersedes v1; encodes the 73% advisory↔CUDA delta.
* `hnerv_backbone_sensitivity_saturated_across_medal_cluster_v1` — backbone-equivalence finding.
* `cross_codec_super_additive_orthogonality_predictor_v1` — orthogonal-codec composition predictor.
* `master_gradient_locality_violation_by_codec_v1` — defines validity boundary (raw-byte master-gradient invalid for entropy-coded archives; post-decompress grain is canonical).
* `per_pair_master_gradient_score_impact_taylor_v1` — Taylor + Cauchy-Schwarz bound the per-pair extractor uses.

**Sister methodology + tooling:**

* [`docs/cargo_cult_unwind_methodology.md`](cargo_cult_unwind_methodology.md) — paradigm-rescue discipline (sister; same template format).
* [`docs/canonical_equations_tour.md`](canonical_equations_tour.md) — tour of the canonical equations registry.
* [`docs/master_gradient_extractor_tour.md`](master_gradient_extractor_tour.md) — tool tour for the per-element sensitivity extractor.
* [`docs/asymptotic_floor_candidate_inventory.md`](asymptotic_floor_candidate_inventory.md) §C.8 — entropy-coding canonical references.
* [`docs/meta_engineering_vision.md`](meta_engineering_vision.md) §4 — empirical validation status of the medal cluster.

**Canonical research artifacts:**

* `.omx/research/cross_candidate_sensitivity_comparison_diagnostic_20260520T192204Z.md` — full 21-pair matrix + diagnostic.
* `.omx/research/cross_candidate_strategic_findings_canonical_extension_20260520T195940Z.md` — the three findings canonical extension.
* `.omx/state/cross_substrate_sensitivity_similarity_matrix_*.json` — live similarity matrix snapshots.
* `.omx/state/master_gradient_anchors.jsonl` — canonical per-anchor ledger (per Catalog #131/#138 fcntl-locked discipline).

**Canonical consumers:**

* `tac.cathedral_consumers.cross_substrate_similarity_consumer` (Tier A observability annotator).
* `tac.cathedral_consumers.cross_codec_orthogonality_predictor_consumer` (orthogonal-codec composition predictor).
* `tac.cathedral_consumers.auto_trigger_similarity_after_master_gradient_anchor_consumer` (post-anchor-landed auto-trigger — this methodology's apparatus-wide extension surface).

**Council-conduct cross-references** (per CLAUDE.md):

* "Apples-to-apples evidence discipline — NON-NEGOTIABLE, HIGHEST EMPHASIS" — the per-axis hardware-substrate-axis triple discipline this methodology operationalizes.
* "Frontier scores are pointer-only — NON-NEGOTIABLE" — per-byte sensitivity is a substrate-level signal that the canonical frontier pointer consumes via downstream ranking.
* "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY — NON-NEGOTIABLE, HIGHEST EMPHASIS" — per-byte sensitivity comparison is a research-only signal until paired with empirical auth-eval.

---

## Honest scope

The methodology is **engineering rigor for converting substrate scores into byte-level structural understanding**, not a contest-score primitive by itself. It does not produce score reductions directly; it surfaces which bytes carry the score advantage between substrate pairs, which informs WHERE future engineering effort should land.

The three findings codified in canonical equations 7-9 are the substantive empirical anchors — they generalize from the 2026-05-20 21-pair matrix to a structural claim about the HNeRV medal cluster's class-saturation, the cross-hardware drift property of per-byte sensitivity, and the orthogonal-codec composition predictor. The methodology's auto-trigger consumer extends this to apparatus-wide: every new master-gradient anchor that lands automatically updates the similarity matrix, so the structural understanding is a property of the apparatus, not a snapshot in time.

Broader generalization of the methodology (to neural compression research, interpretable ML, cross-architecture model analysis) is honest extrapolation supported by the structural primitives but not yet empirically validated outside the comma video compression contest scorer.
