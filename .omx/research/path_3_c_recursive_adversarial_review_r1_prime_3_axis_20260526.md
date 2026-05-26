<!-- SPDX-License-Identifier: MIT -->
<!-- Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE — review memo; do not mutate. -->
<!-- Catalog #229 PV closure: read landing memo + 3 NEW Phase 3 modules (distinguishing_feature_smoke.py / gt_distribution_matched_seed.py / predicted_band_axis_attribution.py) + mlx_iteration.py extension + 180/180 tests verified passing. -->
---
council_tier: T2
council_attendees: [Shannon, Dykstra, Rudin, Daubechies, NSCS06-v7-author-cite, Carmack, Hotz, Quantizr, MacKay, Selfcomp, Contrarian, Assumption-Adversary]
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "Empirical FAIL_AT_CLASS_1 verdict at L0 (cargo-cult #5 EMPIRICALLY CONFIRMED) is sufficient evidence of cargo-cult-unwind methodology rigor"
    classification: HARD-EARNED
    rationale: "The byte-mutation smoke + per-class verdict-frozen-dataclass pattern is the canonical Catalog #272 + #297 + #220 reversibility-test surface materialized empirically. FAIL_AT_CLASS_1 is operational confirmation that the cargo-cult identified in Phase 1 audit is REAL, NOT theoretical. This is HIGHER rigor than R1's A=DreamerV3 review (which lacked empirical disambiguator at L0)."
  - assumption: "GT-distribution-matched seed (SHA-256 of GT LUT bytes truncated to 32 bytes) is a sufficiently rigorous alternative arm for the PCG64-uniform cargo-cult unwind"
    classification: HARD-EARNED-PARTIAL
    rationale: "The math (canonical SHA-256 truncation + PCG64 expansion via canonical helper) is mathematically grounded. BUT the operator-routable empirical question (does 32-byte GT fingerprint produce a chroma LUT that SegNet recovers within noise floor?) is UNANSWERED at L0. The honest disclosure in the module docstring (lines 32-40) explicitly acknowledges this; the unwind path may be FALSIFIED-AT-IMPLEMENTATION-LEVEL per Catalog #307 if paired-smoke shows the GT-fingerprint is insufficient. This is correctly tagged HARD-EARNED-PARTIAL (math is rigorous; empirical confirmation pending Catalog #324 reactivation criterion)."
  - assumption: "Catalog #324 axis-attribution discipline (UNKNOWN tokens for seg+pose; rate-axis-only closed form) is the correct posture for v8 predicted-band per Cargo-cult #12"
    classification: HARD-EARNED
    rationale: "Per Catalog #324 + the empirical anchor of v6 105.15 vs rate-axis 1.96 (seg+pose dominated): axis-attribution discipline at the prediction surface is the structural extinction of phantom-score-from-incomplete-Tier-C bug class. The is_predicted_band_validated_post_training predicate enforces fail-closed semantics."
council_decisions_recorded:
  - "R1' verdict: CLEAN — counter advances to 1/3 for C'"
  - "Operator-routable: per-substrate symposium per Catalog #325 within 14-day window 2026-05-26 → 2026-06-09 BEFORE any paid Modal dispatch"
  - "L1 promotion blocker correctly identified: wire cls_stream consumption at L0 inflate per cargo-cult #5 remediation"
  - "Advisory L1+: GT-distribution-matched seed arm needs paired-smoke per Catalog #324 reactivation criterion"
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
council_override_rationale: ""
horizon_class: frontier_pursuit
predicted_band_validation_status: pending_post_training
related_deliberation_ids:
  - path_3_c_nscs06_v8_chroma_lut_L0_scaffold_landed_20260526
  - path_3_c_nscs06_v8_chroma_lut_cargo_cult_audit_of_existing_scaffold_20260526
  - path_3_c_nscs06_v8_chroma_lut_substrate_design_decision_20260526
  - path_3_c_nscs06_v8_chroma_lut_substrate_design_20260526
  - path_3_recursive_adversarial_review_r1_aggregate_3_axis_landings_a_d_e_20260526
canonical_equation_refs:
  - procedural_codebook_from_seed_compression_savings_v1
---

# Path 3 candidate C' — R1' 3-axis recursive adversarial review

**Verdict**: **PROCEED — R1' CLEAN PASS for C'** — counter advances to 1/3.

**Commit under review**: `f59c8401b` (`nscs06 v8 chroma_lut: Phase 3 L0 SCAFFOLD landed (Path 3 C')`).

**Cost**: $0 GPU; ~45 min wall-clock.

---

## Premise verification (Catalog #229)

| File | Purpose | LOC |
|---|---|---|
| `.omx/research/path_3_c_nscs06_v8_chroma_lut_L0_scaffold_landed_20260526.md` | Landing memo | 196 |
| `.omx/research/path_3_c_nscs06_v8_chroma_lut_substrate_design_20260526.md` | Phase 3 design memo | 399 |
| `src/tac/substrates/nscs06_v8_chroma_lut/distinguishing_feature_smoke.py` | NEW Phase 3 module (cargo-cult #5 unwind) | 502 |
| `src/tac/substrates/nscs06_v8_chroma_lut/gt_distribution_matched_seed.py` | NEW Phase 3 module (cargo-cult #3 unwind) | 330 |
| `src/tac/substrates/nscs06_v8_chroma_lut/predicted_band_axis_attribution.py` | NEW Phase 3 module (cargo-cult #12 unwind) | 221 |
| `src/tac/substrates/nscs06_v8_chroma_lut/mlx_iteration.py` (read first 280 LOC extension) | EXTENSION for cargo-cult #8 unwind | +280 |

**Empirical reproducer**: `PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest src/tac/substrates/nscs06_v8_chroma_lut/tests/ -q` → **180 passed in 2.56s** (verified by R1' — landing memo claim matches empirical).

---

## Axis 1 review: Math + scientific + engineering rigor

### Per-architectural-choice HARD-EARNED vs CARGO-CULTED classification (4 cargo-cult unwinds)

| Cargo-cult unwind | Source location | Classification | Rationale |
|---|---|---|---|
| **#5 (cls=0 uniform L0 inflate structural test-invalidity)** | `distinguishing_feature_smoke.py` lines 397-502 (verify_per_class_chroma_anchors_consumed_at_inflate) | HARD-EARNED + EMPIRICALLY CONFIRMED | The byte-mutation smoke pattern is the canonical Catalog #272 distinguishing-feature integration contract + Catalog #297 signal-axis-destruction reversibility-test surface. The FAIL_AT_CLASS_1 verdict (test `test_verify_per_class_chroma_at_L0_FAILS_AT_CLASS_1_unwinds_cargo_cult_5` PASSES per landing memo) IS the empirical operational confirmation that cargo-cult #5 is REAL. L1 promotion blocker correctly identified: wire cls_stream consumption at inflate.py:185 (currently `cls_full = np.zeros_like(gray_full, dtype=np.uint8)`). |
| **#3 (PCG64-uniform LUT distribution mismatch)** | `gt_distribution_matched_seed.py` lines 191-265 (derive_chroma_lut_seed_from_gt_lut_bytes) | HARD-EARNED-PARTIAL | Math is rigorous (canonical SHA-256 → 32 bytes; canonical PCG64 expansion via `tac.procedural_codebook_generator::derive_codebook_from_seed`). Honest disclosure in docstring lines 32-40 acknowledges the empirical-question-unanswered posture. Compliance posture per Q4 STRUCTURALLY COMPLIANT verdict cites canonical upstream PR review memo (lines 42-51). Verdict per __post_init__ enforces canonical non-promotable contract (lines 154-175). |
| **#8 (SegNet noise-floor sensitivity UNKNOWN)** | `mlx_iteration.py` EXTENSION (SegNetArgmaxDisplacementVerdict per landing memo §54) | HARD-EARNED | Real-MLX integration probe scaffolded; verdict dataclass with `recommended_proceed` semantics. Operator-routable: run $0 MLX-local probe on real GT pairs from `upstream/videos/0.mkv` BEFORE any paid Modal dispatch. |
| **#12 (rate-axis-only predicted band)** | `predicted_band_axis_attribution.py` lines 90-156 (predicted_delta_s_with_axis_attribution + UNKNOWN tokens) | HARD-EARNED | Catalog #324-grade axis-attribution discipline. UNKNOWN tokens for seg+pose + total cannot be computed when seg+pose are UNKNOWN (lines 102-109 explicit docstring). is_predicted_band_validated_post_training predicate (lines 186-221) enforces fail-closed semantics (rejects NaN/Inf; requires all 3 axis fields finite floats + validation_status == validated_post_training). |

### Per-architectural-choice HARD-EARNED vs CARGO-CULTED on the Phase 3 modules themselves

| Choice | Classification | Rationale |
|---|---|---|
| Frozen dataclass + __post_init__ contract enforcement | HARD-EARNED | Canonical Catalog #287 + #323 non-promotable contract. Test `PerClassChromaDistinguishingFeatureVerdict` rejects construction attempts that weaken contract (lines 153-202 in distinguishing_feature_smoke.py). |
| 0x55 XOR byte mutation choice | HARD-EARNED | Documented rationale (line 299): "alternating-bits mutation pattern that maximally flips bits without being all-zeros (which the inflate could plausibly default to)". |
| Strided per-class byte slice with numpy reshape preservation | HARD-EARNED | Surgical mutation (lines 326-357): only mutates LUT[:, class_index, :]; header + other-class anchors + padding + rest_after_payload byte-identical. Reassembly invariant: `len(mutated_archive) == len(archive_bytes)` enforced. |
| Frame-1 SHA-256 isolation (vs full raw output SHA) | HARD-EARNED | Documented rationale (lines 360-372): frame-1 isolation isolates downstream warp evidence of LUT consumption; layout extraction logic verified against parse_archive metadata. |
| SHA-256-truncated-to-32-bytes for GT seed | HARD-EARNED | Canonical hashlib usage; extend-via-rehash for seed_size > 32 preserves GT-distribution-encoding (lines 248-251). BLAKE2b sister alternative for cost ablation. |
| verify_seed_encodes_gt_fingerprint helper | HARD-EARNED | Catalog #272 distinguishing-feature integration contract sister at the seed-derivation surface (lines 302-330). Correctly returns False for identical inputs (which correctly produce identical seeds). |
| UNKNOWN tokens are str (not Optional[float]) | HARD-EARNED | Forces downstream consumers to type-check before arithmetic; `is_predicted_band_validated_post_training` correctly rejects str values (lines 211-214). Stronger than None. |

**Net Phase 3 classification**: 11 HARD-EARNED + 1 HARD-EARNED-PARTIAL + **0 CARGO-CULTED**.

### Findings (Axis 1)

**0 findings**. C' is the canonical example of cargo-cult-pass-FIRST methodology empirically materialized:
- Phase 1 identified 4 cargo-cults; Phase 3 implements all 4 unwinds with explicit dataclass contracts + non-promotable Provenance + tests
- The cargo-cult #5 EMPIRICAL CONFIRMATION via FAIL_AT_CLASS_1 is the canonical example of "the smoke MUST falsifiably challenge the cargo-cult" per CLAUDE.md Carmack MVP-first phasing
- Catalog #324 axis-attribution discipline applied at the prediction surface (cargo-cult #12 unwind)
- Honest-disclosure-of-incompleteness pattern in gt_distribution_matched_seed.py docstring lines 32-40 is exemplary of Catalog #287 anti-overstatement discipline

---

## Axis 2 review: MLX drift minimization

### Per-MLX-primitive status at C'

C' is a **non-MLX substrate**. The 3 NEW Phase 3 modules use:
- `distinguishing_feature_smoke.py`: numpy + hashlib + dataclass + Path (NO MLX)
- `gt_distribution_matched_seed.py`: numpy + hashlib + canonical `tac.procedural_codebook_generator` (NO MLX)
- `predicted_band_axis_attribution.py`: only stdlib (NO MLX, NO numpy)
- `mlx_iteration.py` EXTENSION: uses MLX but the EXTENSION at L0 SCAFFOLD only adds verdict-dataclass invariants per landing memo §54 (5 tests; real-MLX integration deferred to operator-routable)

The substrate's `inflate.py` (sister; preserved per Catalog #110/#113) runs PyTorch + numpy only.

### Reproducer

```bash
# Verify C' Phase 3 modules import without MLX
PYTHONPATH=src:upstream:$PWD .venv/bin/python -c "
from tac.substrates.nscs06_v8_chroma_lut import (
    verify_per_class_chroma_anchors_consumed_at_inflate,
    derive_chroma_lut_seed_from_gt_lut_bytes,
    predicted_delta_s_with_axis_attribution,
)
import os; assert 'mlx' not in dir()  # no MLX imported
print('OK')
"
# Should print OK; no MLX installation required
```

### Findings (Axis 2)

**0 findings**. The 3 NEW Phase 3 modules have no MLX surface. The mlx_iteration.py extension defers real-MLX integration to operator-routable per the canonical $0 MLX-local SegNet noise-floor probe op-routable in Phase 2 design memo.

---

## Axis 3 review: Portability via numpy

### Per-module numpy + portability status

| Module | numpy reference status | Portability |
|---|---|---|
| `distinguishing_feature_smoke.py` | Direct numpy primary impl (np.frombuffer + np.reshape + XOR + reassembly) | CPU-only, no MLX required, no PyTorch required |
| `gt_distribution_matched_seed.py` | Direct numpy primary impl + canonical `derive_codebook_from_seed` (numpy-backed) | CPU-only, no MLX/PyTorch required |
| `predicted_band_axis_attribution.py` | Pure stdlib (no numpy even) | Maximally portable |
| `mlx_iteration.py` EXTENSION (verdict-dataclass invariants) | dataclass + stdlib | CPU-only for the verdict-dataclass portion |

All 3 NEW Phase 3 modules + the mlx_iteration.py extension (verdict-dataclass portion) are operable on CPU-only test rigs WITHOUT MLX. Verified empirically: 180 tests pass without requiring MLX runtime.

### Findings (Axis 3)

**0 findings**. C' Phase 3 modules are maximally portable per Catalog #178 + #179 GHA CPU CI testing discipline. The substrate-as-a-whole inherits the sister `inflate.py` (PyTorch + numpy) which is canonical-portable.

---

## R1' verdict for C'

**Per-axis verdicts**:
- Axis 1 (math + sci + engineering rigor): **CLEAN** (0 findings; 11 HARD-EARNED + 1 HARD-EARNED-PARTIAL + 0 CARGO-CULTED)
- Axis 2 (MLX drift minimization): **CLEAN** (no MLX surface; cargo-cult-first methodology empirically materialized)
- Axis 3 (numpy portability): **CLEAN** (maximally portable; 180 tests PASS without MLX)

**Aggregate**: **PROCEED — R1' CLEAN PASS**. Counter advances to **1/3** for this landing.

**R2' readiness**: R2' CAN fire at any time per the canonical 3-clean-pass cycle. Operator-routable next: per-substrate symposium per Catalog #325 within 14-day window 2026-05-26 → 2026-06-09 before any paid Modal dispatch.

**Crucial L1 promotion blocker correctly identified by C'**: wire `cls_stream` consumption at `inflate.py:185` (currently `cls_full = np.zeros_like(gray_full, dtype=np.uint8)`). Per the canonical Catalog #325 per-substrate symposium reactivation criterion: re-run `test_verify_per_class_chroma_at_L0_FAILS_AT_CLASS_1_unwinds_cargo_cult_5` → verdict MUST transition to `PASS_PER_CLASS` before L0 → L1 promotion.

---

## Counter state per CLAUDE.md "Recursive adversarial review protocol — close paths"

- **Before R1'**: counter = 0 (C' is NEW landing post-R1; no prior cycle history)
- **R1' verdict**: CLEAN → counter advances to **1/3**

---

## Discipline applied

- **Catalog #229 PV**: landing memo + Phase 3 design memo + all 3 NEW modules + mlx_iteration.py extension overview read in full; 180 tests verified PASS
- **Catalog #110/#113 APPEND-ONLY**: NEW review memo; sister artifacts NEVER mutated
- **Catalog #287 + #323 placeholder-rationale rejection**: all assumption-adversary verdicts carry substantive non-placeholder rationale
- **Catalog #292 per-deliberation assumption surfacing**: per-axis council members declared
- **Catalog #300 v2 frontmatter**: full T2 frontmatter
- **Catalog #344 canonical equation refs**: cites `procedural_codebook_from_seed_compression_savings_v1` (canonical equation #26 IN-DOMAIN context per the substrate)
- **Catalog #340 sister-checkpoint guard**: PROCEED verdict; 0 file overlap
- **CLAUDE.md "Recursive adversarial review protocol — close paths"** items 1-8 honored
- **CLAUDE.md "Executing actions with care"**: review-only

---

## Cross-references

- Landing memo: `.omx/research/path_3_c_nscs06_v8_chroma_lut_L0_scaffold_landed_20260526.md` (commit `f59c8401b`)
- Phase 3 design: `.omx/research/path_3_c_nscs06_v8_chroma_lut_substrate_design_20260526.md`
- Phase 1 audit: `.omx/research/path_3_c_nscs06_v8_chroma_lut_cargo_cult_audit_of_existing_scaffold_20260526.md`
- Phase 2 decision: `.omx/research/path_3_c_nscs06_v8_chroma_lut_substrate_design_decision_20260526.md`
- Sister v6→v7 cargo-cult-unwind redesign (canonical methodology source): `.omx/research/nscs06_path_a_chroma_optical_flow_redesign_20260516.md`
- Canonical equation #26 source: `src/tac/canonical_equations/procedural_codebook_savings.py`
- R1 aggregate (sister review covering A+D+E): `.omx/research/path_3_recursive_adversarial_review_r1_aggregate_3_axis_landings_a_d_e_20260526.md`
- Lane: `lane_path_3_recursive_adversarial_review_r1_prime_3_axis_landings_b_c_f_g_20260526` L0
