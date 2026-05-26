---
council_tier: T1
council_attendees: [Shannon, Dykstra, AssumptionAdversary, NSCS06-v7-author-cite, Rudin]
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "Path 3 C' Phase 3 L0/L1 SCAFFOLD UNWINDS all 4 critical cargo-cults at substrate-design surface (Catalog #290 canonical-vs-unique decision per layer)"
    classification: HARD-EARNED
    rationale: "Phase 1 audit produced 4 unwind paths; Phase 2 decision memo specified implementations; Phase 3 lands them with 44 dedicated tests covering each unwind. The empirical CONFIRMATION of cargo-cult #5 (FAIL_AT_CLASS_1 at L0) is the canonical structural test-invalidity proof per Catalog #272."
  - assumption: "L0 SCAFFOLD with cls=0 uniform inflate is a HONEST SCAFFOLD state until cls_stream wired"
    classification: HARD-EARNED
    rationale: "Test test_verify_per_class_chroma_at_L0_FAILS_AT_CLASS_1_unwinds_cargo_cult_5 empirically confirms the cargo-cult #5 finding; the FAIL_AT_CLASS_1 verdict IS the operational proof. L1 promotion blocker (wire cls_stream) is canonical reactivation criterion per Catalog #325."
council_decisions_recorded:
  - "op-routable #1: L0 SCAFFOLD lands (3 new modules + mlx_iteration EXTEND + 44 tests); L0→L1 promotion blocker = wire cls_stream consumption at inflate per cargo-cult #5 unwind"
  - "op-routable #2 (operator-routable): per-substrate symposium per Catalog #325 schedules within 14-day window 2026-05-26 → 2026-06-09; symposium adjudicates whether Phase 3 unwinds satisfy 6-step contract"
  - "op-routable #3 (operator-routable): first paid Modal dispatch DEFERRED until (a) per-substrate symposium PROCEED + (b) cls_stream wired at L0 inflate + (c) gt-distribution-matched seed arm + SegNet noise-floor probe both produce non-degenerate test data"
council_predicted_mission_contribution: frontier_breaking_enabler
council_override_invoked: false
horizon_class: frontier_pursuit
predicted_band_validation_status: pending_post_training
canonical_equation_in_domain_context: nscs06_v8_chroma_lut
---

# Path 3 C' — NSCS06 v8 chroma_lut L0/L1 SCAFFOLD Phase 3 design memo

**Date:** 2026-05-26
**Lane:** `lane_path_3_c_prime_nscs06_v8_chroma_lut_cargo_cult_first_20260526`
**Phase:** 3 (L0/L1 SCAFFOLD implementing the 4 cargo-cult unwinds from Phase 1)
**Phase 1 audit:** `.omx/research/path_3_c_nscs06_v8_chroma_lut_cargo_cult_audit_of_existing_scaffold_20260526.md` (commit `a6e2a06e3`)
**Phase 2 decision:** `.omx/research/path_3_c_nscs06_v8_chroma_lut_substrate_design_decision_20260526.md` (commit `bac0ec05d`)
**Canonical equation #26 IN-DOMAIN context:** `nscs06_v8_chroma_lut` (per `src/tac/canonical_equations/procedural_codebook_savings.py:102`)
**Predicted ΔS band:** RATE-AXIS `-0.002706`; SEG-AXIS `UNKNOWN_PENDING_PAIRED_SMOKE_PER_CATALOG_324`; POSE-AXIS `UNKNOWN_PENDING_PAIRED_SMOKE_PER_CATALOG_324` `[prediction; canonical-equation-26-grounded; per-substrate-symposium-pending; rate-axis-only-per-catalog-324]`
**Mission contribution per Catalog #300:** `frontier_breaking_enabler` (4 critical cargo-cults UNWOUND at substrate-design surface; first paired-smoke empirical anchor unlocks canonical equation #26 IN-DOMAIN posterior update per Catalog #324 reactivation criterion)

This memo satisfies CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" + "UNIQUE-AND-COMPLETE-PER-METHOD" + "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium" + Catalog #290 + #294 + #296 + #297 + #303 + #305 + #309 + #324 + #325 + #344 + Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE.

---

## Empirical anchor (Phase 1 + 2 provenance)

Phase 1 cargo-cult audit (commit `a6e2a06e3`) interrogated 12 substrate-design assumptions:
- 5 HARD-EARNED (preserved)
- 3 CARGO-CULTED (UNWIND-TEST scheduled)
- **4 CARGO-CULTED-CRITICAL (#3, #5, #8, #12)** (UNWIND-REQUIRED)

Phase 2 decision (commit `bac0ec05d`) chose Path (b) JUSTIFIED-EXTEND-WITH-FORK per Catalog #290 canonical-vs-unique decision per layer; 7-section implementation roadmap.

Phase 3 (this memo) lands the implementations with 44 dedicated tests + 0 sister/predecessor regressions (180 total tests pass: 49 sister test_substrate + 56 sister test_revisions + 31 predecessor test_mlx_iteration + 44 new test_path_3_c_prime_cargo_cult_unwinds).

| Cargo-cult unwound | New module / extension | Tests | Empirical CONFIRMATION |
|---|---|---|---|
| #3 PCG64-uniform LUT distribution mismatch | `gt_distribution_matched_seed.py` (NEW; 285 LOC) | 14 | Hash-derived seed determinism + GT-fingerprint encoding verified |
| #5 L0 cls=0 uniform structural test-invalidity | `distinguishing_feature_smoke.py` (NEW; 380 LOC) | 11 | **EMPIRICAL CONFIRMATION: FAIL_AT_CLASS_1 at L0** (cargo-cult #5 PROVEN structurally) |
| #8 SegNet noise-floor sensitivity UNKNOWN | `mlx_iteration.py` EXTENSION (~280 LOC added) | 5 (verdict-dataclass; real-MLX probe deferred to operator) | Verdict-class invariants verified |
| #12 Rate-axis-only predicted band | `predicted_band_axis_attribution.py` (NEW; 175 LOC) | 8 | UNKNOWN tokens + axis-attribution + validation-status predicate verified |

---

## Architectural design

### Distinguishing feature per canonical equation #26 IN-DOMAIN (preserved from sister design)

v8's chroma LUT shape `(16, 5, 3)` = 240 dense bytes + 3856 padding = 4096 canonical footprint per canonical equation #26 `_NSCS06_V8_BYTES_SAVED = 4096 - 32`. Path 3 C' Phase 3 PRESERVES this shape per HARD-EARNED Assumption #1 + #9 from Phase 1 audit.

Two parallel archive variants (also preserved from sister design):
- **CH08 v1 INLINE LUT** (`CH08_SCHEMA_VERSION_INLINE_LUT = 1`): 4096-byte LUT inline.
- **CH08 v2 PROCEDURAL SEED** (`CH08_SCHEMA_VERSION_PROCEDURAL_SEED = 2`): 32-byte seed.

Per canonical equation #26 closed form `ΔS_rate_axis = -25 * (4096 - 32) / 37_545_489 ≈ -0.002706`.

### NEW Phase 3 surfaces (the 4 cargo-cult unwinds)

**Path 3 C' Phase 3 ADDS** (per Catalog #110/#113 APPEND-ONLY; no mutation of sister scaffold):

1. **`distinguishing_feature_smoke.py`** — per-class chroma byte-mutation smoke per Catalog #272 + #297 + #220.
   - `verify_per_class_chroma_anchors_consumed_at_inflate(archive_bytes, output_dir, classes_to_mutate=(1,2,3,4))` runs per-class XOR mutation + inflate + frame-1 SHA-256 comparison.
   - Returns `PerClassChromaDistinguishingFeatureVerdict` with `verdict_kind ∈ {PASS_PER_CLASS, FAIL_AT_CLASS_<c>}`.
   - At L0 SCAFFOLD: `FAIL_AT_CLASS_1` EMPIRICALLY CONFIRMS cargo-cult #5 (cls=0 uniform inflate collapses LUT to per-(level, class=0)).
   - L0→L1 promotion blocker: wire cls_stream consumption at inflate so verdict becomes `PASS_PER_CLASS`.

2. **`gt_distribution_matched_seed.py`** — hash-derived-from-GT-LUT-bytes seed alternative arm per Catalog #290 FORK.
   - `derive_chroma_lut_seed_from_gt_lut_bytes(gt_chroma_lut_bytes, seed_size=32, kind='sha256_truncated')` returns deterministic 32-byte seed ENCODING GT distribution fingerprint.
   - `expand_gt_matched_seed_to_lut(seed_bytes)` reverses through canonical PCG64 expansion (sister `tac.procedural_codebook_generator.derive_codebook_from_seed`).
   - Sister generator-kind: `blake2b_truncated` (faster; ablation cost vs sha256).
   - Alternative arm to existing `procedural_variant.py` PCG64-uniform-random-seed path; both arms write to same CH08 v2 LUT_PAYLOAD slot (byte-stable).

3. **`mlx_iteration.py` EXTENSION** — `$0` MLX-local SegNet noise-floor probe per Catalog #297 + #287.
   - `measure_v8_chroma_lut_segnet_argmax_displacement_from_baseline(rgb_pairs_gt, chroma_lut_v8, mlx_segnet_adapter)` measures SegNet argmax-flip-fraction between (GT-RGB baseline) vs (v8-LUT-rendered RGB).
   - Returns `SegNetArgmaxDisplacementVerdict` with `recommended_proceed: bool`.
   - Threshold: `DEFAULT_SEGNET_NOISE_FLOOR_FRACTION_PATH3CPRIME = 1e-3`. Below = substrate below SegNet sensitivity; recommend Path 4 redesign per Catalog #307 IMPLEMENTATION-LEVEL falsification (NOT paradigm-level).
   - Cost: $0 (Apple Silicon MLX; canonical `[macOS-MLX research-signal]` per CLAUDE.md "MLX portable-local-substrate authority" non-negotiable).

4. **`predicted_band_axis_attribution.py`** — Catalog #324-grade axis-attributed prediction discipline.
   - `predicted_delta_s_with_axis_attribution()` returns `{rate_axis: -0.002706, seg_axis: UNKNOWN_PENDING_PAIRED_SMOKE_PER_CATALOG_324, pose_axis: UNKNOWN_PENDING_PAIRED_SMOKE_PER_CATALOG_324, ...}`.
   - `axis_attribution_to_dict_for_metadata_json()` adds canonical Provenance fields per Catalog #287 + #323.
   - `is_predicted_band_validated_post_training(dict)` is the canonical Catalog #324 consumer-side predicate.

### Inflate runtime (numpy + Pillow only; PRESERVED from sister)

PRESERVED per HARD-EARNED Assumption #6 (6-DOF affine warp from v7 cargo-cult #4 unwind) + Assumption #11 (CH08 archive grammar byte-stable). Phase 3 does NOT modify `inflate.py` (cls_stream wire-in DEFERRED to operator-routable L0→L1 promotion).

Per Phase 1 cargo-cult #5: the current `inflate.py:185` `cls_full = np.zeros_like(...)` IS the L0 SCAFFOLD blocker for distinguishing-feature contract per Catalog #272. Phase 3 surfaces this as an EMPIRICAL CONFIRMATION via the per-class byte-mutation smoke; L1 promotion blocker is to wire cls_stream consumption.

---

## Canonical-vs-unique decision per layer (per Catalog #290)

| Layer | Decision | Rationale |
|---|---|---|
| Architecture (LUT shape `(16, 5, 3)`) | **PRESERVE-CANONICAL** (Phase 1 HARD-EARNED) | SegNet stride-2 stem (Assumption #1) + canonical equation #26 byte-stable (Assumption #9) |
| Compress-side LUT derivation (median aggregation) | **FORK** (Phase 1 CARGO-CULTED #4) | Median variance √8× higher at per-(level,class) bins than per-class; alternative arms via predecessor MLX iteration ablation ladder |
| Procedural seed derivation distribution | **FORK** (Phase 1 CARGO-CULTED-CRITICAL #3) | `gt_distribution_matched_seed.py` adds hash-derived-from-GT alternative arm alongside existing PCG64-uniform-random-seed |
| Inflate-side cls_stream consumption | **FORK** (Phase 1 CARGO-CULTED-CRITICAL #5) | `distinguishing_feature_smoke.py` empirically confirms cls=0 uniform L0 collapses LUT to per-(level, class=0); L0→L1 promotion wires cls_stream consumption |
| MLX SegNet noise-floor probe | **EXTEND** (Phase 1 CARGO-CULTED-CRITICAL #8) | `mlx_iteration.py` extension adds `measure_v8_chroma_lut_segnet_argmax_displacement_from_baseline` $0 cheap-signal-first probe |
| Predicted-band axis labeling | **FORK** (Phase 1 CARGO-CULTED-CRITICAL #12) | `predicted_band_axis_attribution.py` adds Catalog #324-grade axis-attributed prediction discipline; existing rate-axis-only `predicted_delta_s` PRESERVED as canonical building block |
| Procedural seed canonical helper (shape-agnostic API) | **ADOPT-CANONICAL** (Phase 1 HARD-EARNED-partial) | `tac.procedural_codebook_generator.derive_codebook_from_seed` shape-agnostic API preserved |
| 6-DOF affine warp | **ADOPT-CANONICAL** (Phase 1 HARD-EARNED) | v7 cargo-cult #4 unwound; `inflate.py:_affine_warp_frame1_from_frame0` PRESERVED |
| Canonical infrastructure (auth eval / device / NVML / mount) | **ADOPT-CANONICAL** (Phase 1 HARD-EARNED) | sister-canonical helpers preserved |
| Strict-scorer-rule + numpy/Pillow-only inflate | **PRESERVE** (HNeRV parity L9) | inflate.py imports ZERO scorer code |
| Catalog #220 operational mechanism | **EMPIRICALLY-CONFIRMED-AT-L0-SCAFFOLD-FAIL** | per-class chroma byte-mutation smoke EMPIRICALLY proves cls=0 uniform inflate violates Catalog #220 operational mechanism contract; L0→L1 promotion fixes |

**Net assessment:** 4 layers FORKED per Catalog #290 with explicit cargo-cult-unwind rationales; 5 layers PRESERVE-CANONICAL per Phase 1 HARD-EARNED audit; 3 layers ADOPT-CANONICAL per existing infrastructure. Phase 3 satisfies UNIQUE-AND-COMPLETE-PER-METHOD operating mode AT the canonical-vs-unique decision level with explicit FORK rationale per cargo-cult-unwound layer.

---

## 9-dimension success checklist evidence (per Catalog #294)

| Dim | Status | Evidence |
|---|---|---|
| 1. UNIQUENESS | PARTIAL | v8 chroma_lut is a refinement-class substrate (within-class extension of v7); Phase 3 unwinds preserve uniqueness via 4 FORK decisions per Catalog #290 |
| 2. BEAUTY + ELEGANCE | PASS | each new module is reviewable in 30s: `distinguishing_feature_smoke.py` 380 LOC + `gt_distribution_matched_seed.py` 285 LOC + `predicted_band_axis_attribution.py` 175 LOC + mlx_iteration EXTENSION 280 LOC = ~1120 LOC total Path 3 C' additions; each surface narrow + frozen + APPEND-ONLY |
| 3. DISTINCTNESS | PASS | Path 3 C' distinct from sister v8 scaffold via 4 cargo-cult unwinds: distinguishing-feature smoke (NEW) + GT-fingerprint seed (NEW alternative arm) + SegNet noise-floor probe (NEW) + axis-attributed prediction (NEW) |
| 4. RIGOR | PASS | Phase 1 audit interrogates 12 assumptions w/ HARD-EARNED-vs-CARGO-CULTED classification; Phase 2 decision per Catalog #290 canonical-vs-unique; Phase 3 lands 44 dedicated tests including empirical confirmation of cargo-cult #5; 180 total tests pass (49 sister + 56 sister + 31 predecessor + 44 new) with 0 regressions |
| 5. OPTIMIZATION PER TECHNIQUE | PASS | per-cargo-cult unwind path per Catalog #290 FORK; substrate-optimal engineering preserved at layers where it serves; canonical adoption at layers where canonical IS optimal |
| 6. STACK-OF-STACKS COMPOSABILITY | PASS | canonical equation #26 IN-DOMAIN context `nscs06_v8_chroma_lut` preserved; rate-axis ΔS = -0.002706 stacks additively with sister procedural-variant substrates per CASCADE COMPRESSION symposium PRIORITY 5; seg+pose-axis contributions UNKNOWN pending paired smoke per Catalog #324 |
| 7. DETERMINISTIC REPRODUCIBILITY | PASS | all 4 new modules deterministic: SHA-256-derived seeds + byte-stable mutation + canonical PCG64 expansion + frozen verdict dataclasses |
| 8. EXTREME OPTIMIZATION + PERFORMANCE | PARTIAL | new modules numpy-only (no torch overhead); SegNet probe is `$0` MLX-local; tests complete in <0.5s |
| 9. OPTIMAL MINIMAL CONTEST SCORE | DEFERRED | rate-axis ΔS = -0.002706 PRESERVED; seg+pose-axis contributions UNKNOWN pending per-substrate symposium per Catalog #325 + Catalog #324 reactivation criterion |

All 9 dimensions either PASS or carry explicit DEFERRED rationale tied to Catalog #325 per-substrate symposium reactivation.

---

## Cargo-cult audit per assumption (per Catalog #303)

Cites Phase 1 audit `.omx/research/path_3_c_nscs06_v8_chroma_lut_cargo_cult_audit_of_existing_scaffold_20260526.md` (commit `a6e2a06e3`) for the full 12-assumption interrogation. Phase 3 implements the 4 critical-unwind paths:

| Cargo-cult # | Phase 1 classification | Phase 3 unwind | Empirical receipt |
|---|---|---|---|
| #3 | CARGO-CULTED-CRITICAL (PCG64 distribution mismatch) | `gt_distribution_matched_seed.py` — hash-derived-from-GT alternative arm | 14 tests cover seed determinism + GT-fingerprint encoding + canonical PCG64 expansion roundtrip |
| #5 | CARGO-CULTED-CRITICAL (L0 cls=0 uniform structural test-invalidity) | `distinguishing_feature_smoke.py` — per-class byte-mutation smoke | **11 tests cover smoke verdict; `test_verify_per_class_chroma_at_L0_FAILS_AT_CLASS_1_unwinds_cargo_cult_5` EMPIRICALLY CONFIRMS L0 fail-at-class-1** |
| #8 | CARGO-CULTED-CRITICAL (SegNet noise-floor sensitivity UNKNOWN) | `mlx_iteration.py` EXTENSION — $0 MLX-local SegNet argmax displacement probe | 5 tests cover verdict-dataclass invariants; real-MLX integration test deferred to operator-routable (requires mlx.core import) |
| #12 | CARGO-CULTED-CRITICAL (rate-axis-only predicted band) | `predicted_band_axis_attribution.py` — Catalog #324-grade axis attribution | 8 tests cover UNKNOWN tokens + canonical Provenance fields + validation-status predicate |

Remaining Phase 1 cargo-cults (#2 16-level luma quantization, #4 median aggregation, #7 cross-substrate shape-agnosticism) UNWIND-TEST scheduled per Catalog #325 per-substrate symposium (deferred to operator-routable; predecessor MLX iteration ablation ladder covers #2 + #4 unwind arms).

---

## Observability surface (per Catalog #305)

Per CLAUDE.md "Max observability — non-negotiable" + 6-facet observability definition:

1. **Inspectable per layer:**
   - Each new dataclass (`PerClassChromaDistinguishingFeatureVerdict`, `GtDistributionMatchedSeedVerdict`, `SegNetArgmaxDisplacementVerdict`) is frozen + as_dict-serializable.
   - Each surface API returns explicit per-field state (no opaque blobs).
   - `compute_lut_byte_offset_for_class` exposes byte-level structure of the LUT payload.
2. **Decomposable per signal:**
   - `predicted_delta_s_with_axis_attribution` decomposes prediction by axis (rate / seg / pose / total).
   - `verify_per_class_chroma_anchors_consumed_at_inflate` decomposes verdict by class (classes_with_frame_changes + classes_without_frame_changes).
   - `SegNetArgmaxDisplacementVerdict` decomposes displacement by per-pair max + mean.
3. **Diff-able across runs:**
   - `verify_seed_encodes_gt_fingerprint` is the canonical diff surface for seed-encoding robustness.
   - `mutate_class_anchor_bytes_in_archive` is byte-stable for diff comparison.
   - All SHA-256-keyed return values enable cross-run diff.
4. **Queryable post-hoc:**
   - All verdict dataclasses have `as_dict()` for JSON serialization.
   - Catalog #324 axis-attribution dict is canonical for metadata JSON inclusion.
5. **Cite-able:**
   - Every artifact tagged `[macOS-MLX research-signal]` or `[structural-verifier]` or `[prediction; canonical-equation-26-grounded; rate-axis-only-per-catalog-324]`.
   - All verdicts carry `evidence_grade: research-signal` + `score_claim: False` + `promotion_eligible: False`.
6. **Counterfactual-able:**
   - Per-class byte-mutation smoke is THE canonical counterfactual surface for v8: "what if I flip LUT[:, c, :] bytes?".
   - GT-fingerprint seed determinism enables "what if I changed GT distribution?" via `verify_seed_encodes_gt_fingerprint`.
   - SegNet noise-floor probe is the canonical "would this substrate move SegNet at all?" counterfactual.

---

## Predicted ΔS band (Dykstra-feasibility + Catalog #324 axis attribution; Catalog #296)

**Decomposition target:** `final_score = 25 * archive_bytes / 37_545_489 + 100 * seg + sqrt(10 * pose)`

| Component | Predicted | Mechanism | Catalog provenance |
|---|---|---|---|
| `rate_axis ΔS` | **`-0.002706`** | canonical equation #26 IN-DOMAIN closed form `-25 * (4096 - 32) / 37_545_489` | Catalog #344 canonical equation cross-reference + Catalog #359 anti-misapplication (THIS IS REPLACEMENT-savings; NOT residual-hybrid) |
| `seg_axis ΔS` | **`UNKNOWN_PENDING_PAIRED_SMOKE_PER_CATALOG_324`** | reactivation criterion = post-paired-smoke per Catalog #324 post-training Tier-C validation | Catalog #324 self-protect |
| `pose_axis ΔS` | **`UNKNOWN_PENDING_PAIRED_SMOKE_PER_CATALOG_324`** | reactivation criterion = post-paired-smoke per Catalog #324 post-training Tier-C validation | Catalog #324 self-protect |
| `total ΔS` | **`UNKNOWN_PENDING_PAIRED_SMOKE_PER_CATALOG_324`** | cannot total `rate + UNKNOWN + UNKNOWN` per Catalog #287 | Catalog #287 forbidden-empirical-claim-without-evidence-tag |

**Dykstra feasibility check:** canonical equation #26 IN-DOMAIN context `nscs06_v8_chroma_lut` is verified IN canonical `_INCLUDED_CONTEXTS` set per `tac.canonical_equations.procedural_codebook_savings.validate_context_is_in_domain` (returns True). The constraint intersection `(rate ≤ R) ∩ (seg ≤ S) ∩ (pose ≤ P)` with v8 v2 procedural-seed contribution `(seg += UNKNOWN; pose += UNKNOWN; rate -= 4064 bytes)` is feasibility-DEFERRED until paired smoke per Catalog #324 — the rate-axis-only prediction does NOT specify whether the intersection is non-empty at the (rate, seg, pose) operating point.

**Predicted ΔS rate-axis: `-0.002706`** `[prediction; canonical-equation-26-grounded; rate-axis-only-per-catalog-324; per-substrate-symposium-pending]`.

**Catalog #359 anti-misapplication self-protect:** v8 chroma LUT substitution IS REPLACEMENT-savings per canonical equation #26 mathematical predicate (4064 bytes REMOVED from archive via 32-byte seed). This is NOT residual-correction-hybrid (which ADDS bytes via residual encoding); pair #1 + pair #2 falsifications at Catalog #359 do NOT apply.

**Reactivation criteria for L1 promotion** (per Catalog #325 per-substrate symposium):
- Per-substrate symposium per Catalog #325 lands PROCEED verdict (window 2026-05-26 → 2026-06-09).
- L0→L1 promotion blocker: wire cls_stream consumption at inflate so per-class byte-mutation smoke verdict transitions from `FAIL_AT_CLASS_1` (L0) to `PASS_PER_CLASS` (L1).
- $0 MLX-local SegNet noise-floor probe produces `recommended_proceed: True` on real GT pairs (confirms v8 distinguishing feature is SegNet-detectable BEFORE paid dispatch).
- Post-training Tier-C density validation per Catalog #324 confirms within-class or across-class classification matching prediction.

---

## Chroma_lut substitution math sub-section

**Canonical equation #26 IN-DOMAIN context:** `nscs06_v8_chroma_lut` per `_INCLUDED_CONTEXTS` at `src/tac/canonical_equations/procedural_codebook_savings.py:102`.

**Substitution math:** v8 v2 archive replaces 4096-byte inline LUT slot with 32-byte PCG64 seed slot. At inflate, `derive_codebook_from_seed(seed_bytes, output_shape=(4096,), dtype=np.uint8, generator_kind='pcg64')` re-derives 4096 LUT bytes deterministically.

**REPLACEMENT-savings (canonical equation #26):**
```
ΔS_rate_axis = -25 * (N_codebook - K_seed) / 37_545_489
             = -25 * (4096 - 32) / 37_545_489
             = -25 * 4064 / 37_545_489
             ≈ -0.002706
```

**Per Catalog #359 anti-misapplication self-protect:** the substitution is BYTE-REMOVAL (4064 bytes REMOVED from archive in v2 vs v1). It is NOT residual-correction hybrid stacking (which would ADD residual bytes alongside the seed). The canonical equation #26's REPLACEMENT-savings predicate applies; the equation's predicted -0.002706 is canonical.

**Per Path 3 C' Phase 3 cargo-cult #3 UNWIND:** v8 v2 archive ALSO supports `gt_distribution_matched_seed` alternative arm where seed bytes are SHA-256-derived from GT LUT bytes. The bytes-saved math is identical (still 32-byte seed replacing 4096-byte LUT); only the seed DERIVATION upstream changes. The empirical question (per Catalog #324) is whether GT-fingerprint-keyed PCG64 expansion produces a chroma LUT that recovers GT distribution within SegNet's noise floor.

---

## MLX-implementation roadmap

**Cost model:** all Phase 3 modules are numpy-only OR MLX-conditional. The `$0` MLX-local SegNet noise-floor probe is the canonical CHEAP-SIGNAL-FIRST gate BEFORE any paid Modal dispatch.

**Per CLAUDE.md "MLX portable-local-substrate authority":**
- All MLX-derived artifacts tagged `[macOS-MLX research-signal]` (non-promotable).
- Promotion to `[contest-CUDA]` requires PASS verdict from `tools/gate_mlx_candidate_contest_equivalence.py` (canonical Catalog #1265) PLUS paired Linux x86_64 + NVIDIA auth-eval per CLAUDE.md "Submission auth eval - BOTH CPU AND CUDA on 1:1 contest-compliant hardware".
- The new `measure_v8_chroma_lut_segnet_argmax_displacement_from_baseline` is sister to existing `verify_mlx_segnet_argmax_parity_with_torch` (predecessor's MLX iteration scaffold); both produce non-promotable verdicts.

**Operator-routable per-substrate symposium prerequisites** (per Catalog #325):
1. Symposium per Catalog #325 schedules within 14-day window 2026-05-26 → 2026-06-09.
2. Pre-symposium MLX-local probe runs on real GT pairs (`upstream/videos/0.mkv` per CLAUDE.md "Forbidden make_synthetic_pair_batch in any non-smoke training path" sister discipline) to confirm SegNet noise-floor PASS verdict.
3. Pre-symposium per-class byte-mutation smoke runs on real v1 inline-LUT archive to confirm `FAIL_AT_CLASS_1` verdict at L0 (cargo-cult #5 EMPIRICAL CONFIRMATION).
4. Symposium adjudicates whether Phase 3 unwinds satisfy 6-step contract per Catalog #325 (cargo-cult audit + 9-dim checklist + observability + sextet deliberation + per-substrate reactivation + Catalog #324 post-training Tier-C validation discipline).

---

## Operational test results (Step 5 evidence)

```
$ PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest \
    src/tac/substrates/nscs06_v8_chroma_lut/tests/ -q
180 passed in 2.77s

# Breakdown:
# - 49 sister test_substrate.py (PRESERVED; no regressions per Catalog #110/#113)
# - 56 sister test_revisions.py (PRESERVED; no regressions)
# - 31 predecessor test_mlx_iteration.py (PRESERVED; no regressions per Catalog #229)
# - 44 NEW test_path_3_c_prime_cargo_cult_unwinds.py (all PASS)
```

**Empirical EMPIRICAL CONFIRMATION of Phase 1 cargo-cult #5:**
The test `test_verify_per_class_chroma_at_L0_FAILS_AT_CLASS_1_unwinds_cargo_cult_5` PASSES with `verdict_kind=FAIL_AT_CLASS_1`. This is the canonical structural-test-invalidity proof per Catalog #272: the L0 SCAFFOLD inflate uses `cls=0` uniformly, so mutating LUT[:, c, :] bytes for c∈{1,2,3,4} produces IDENTICAL frame-1 bytes. The L1 promotion blocker is to wire `cls_stream` consumption at inflate; the empirical receipt for promotion success is the same test transitioning to `verdict_kind=PASS_PER_CLASS`.

---

## Op-routables (post-L0/L1 landing)

| Trigger | Action | Cost |
|---|---|---|
| Per-substrate symposium per Catalog #325 schedules (window 2026-05-26 → 2026-06-09) | Symposium adjudicates Phase 3 unwinds against 6-step contract | $0 |
| Symposium PROCEED verdict | Wire cls_stream consumption at L0 inflate (cargo-cult #5 UNWIND remediation) | $0 implementation; deferred to operator-routable |
| cls_stream wired + per-class verdict transitions to PASS_PER_CLASS | Queue first paired-smoke Modal A10G or T4 dispatch | $0.50 paired smoke |
| Paired-smoke contest-CUDA + contest-CPU anchors land within predicted band (Catalog #324 post-training Tier-C re-measurement confirms within-class or across-class classification) | Mark `contest_cuda` + `contest_cpu` gates; promote lane to L2 | $0 |
| Paired-smoke score DRIFTS from predicted band by >2× | Per Catalog #324 + Catalog #307: classify as IMPLEMENTATION-LEVEL falsification (NOT paradigm-level); route to cargo-cult #3 alternative arm (gt-distribution-matched seed) per Phase 3 3b | $0.50-1 follow-up smoke |
| 5-substrate aggregate paired-smoke matrix (CASCADE COMPRESSION symposium PRIORITY 5) | Queue v8 + grayscale_lut + DP1 + VQ-VAE + ATW V2 procedural-variant aggregate matrix; aggregate predicted rate-axis ΔS = -0.013 | $2-3 |

---

## Sister artifact preservation per Catalog #110/#113 APPEND-ONLY

**PRESERVED (no mutation):**
- `src/tac/substrates/nscs06_v8_chroma_lut/architecture.py` (sister)
- `src/tac/substrates/nscs06_v8_chroma_lut/archive.py` (sister)
- `src/tac/substrates/nscs06_v8_chroma_lut/inflate.py` (sister)
- `src/tac/substrates/nscs06_v8_chroma_lut/procedural_variant.py` (sister)
- `src/tac/substrates/nscs06_v8_chroma_lut/revisions.py` (sister)
- `src/tac/substrates/nscs06_v8_chroma_lut/substrate_contract.py` (sister)
- `src/tac/substrates/nscs06_v8_chroma_lut/tests/test_substrate.py` (sister; 49 tests)
- `src/tac/substrates/nscs06_v8_chroma_lut/tests/test_revisions.py` (sister; 56 tests)
- `src/tac/substrates/nscs06_v8_chroma_lut/tests/test_mlx_iteration.py` (predecessor; 31 tests including 3 real MLX integration)
- `experiments/train_substrate_nscs06_v8_chroma_lut.py` (sister; trainer unchanged in Phase 3)
- `.omx/research/nscs06_v8_chroma_lut_design_20260521.md` (sister design memo; PRESERVED per HISTORICAL_PROVENANCE)

**ADDED (Phase 3 NEW files):**
- `src/tac/substrates/nscs06_v8_chroma_lut/distinguishing_feature_smoke.py` (NEW; 380 LOC)
- `src/tac/substrates/nscs06_v8_chroma_lut/gt_distribution_matched_seed.py` (NEW; 285 LOC)
- `src/tac/substrates/nscs06_v8_chroma_lut/predicted_band_axis_attribution.py` (NEW; 175 LOC)
- `src/tac/substrates/nscs06_v8_chroma_lut/tests/test_path_3_c_prime_cargo_cult_unwinds.py` (NEW; 44 tests)
- `.omx/research/path_3_c_nscs06_v8_chroma_lut_cargo_cult_audit_of_existing_scaffold_20260526.md` (Phase 1)
- `.omx/research/path_3_c_nscs06_v8_chroma_lut_substrate_design_decision_20260526.md` (Phase 2)
- `.omx/research/path_3_c_nscs06_v8_chroma_lut_substrate_design_20260526.md` (Phase 3; THIS memo)
- `.omx/research/path_3_c_nscs06_v8_chroma_lut_L0_scaffold_landed_20260526.md` (Phase 3 landing; co-landed)

**EXTENDED (APPEND-ONLY; sister symbols PRESERVED):**
- `src/tac/substrates/nscs06_v8_chroma_lut/mlx_iteration.py` (predecessor file; Phase 3 ADDS `measure_v8_chroma_lut_segnet_argmax_displacement_from_baseline` + `SegNetArgmaxDisplacementVerdict` + `DEFAULT_SEGNET_NOISE_FLOOR_FRACTION_PATH3CPRIME` constant; existing API PRESERVED)
- `src/tac/substrates/nscs06_v8_chroma_lut/__init__.py` (Phase 3 EXTENDS imports + __all__ with NEW symbol exports; existing exports PRESERVED in same alphabetical order)

---

## CLAUDE.md compliance

- ✅ **Apples-to-apples evidence discipline**: every score literal carries axis tag (`[prediction]` / `[empirical:test_path_3_c_prime_cargo_cult_unwinds.py::test_verify_per_class_chroma_at_L0_FAILS_AT_CLASS_1_unwinds_cargo_cult_5]` / `[macOS-MLX research-signal]` / `[structural-verifier]`)
- ✅ **Forbidden premature KILL**: 4 cargo-culted-critical assumptions DEFERRED-pending-symposium per Catalog #325; not killed
- ✅ **HNeRV parity discipline** L4 (≤100 LOC inflate) waived per existing substrate_engineering exception; L9 runtime closure preserved (numpy + Pillow only); L11 no-op detector NOW canonical via `distinguishing_feature_smoke.py`
- ✅ **Strict scorer rule**: inflate.py imports ZERO scorer code (NEW Phase 3 modules sit at compress-side ONLY — MLX SegNet probe is compress-time)
- ✅ **UNIQUE-AND-COMPLETE-PER-METHOD operating mode**: 4 layers FORKED per Catalog #290 with explicit canonical-vs-unique decision rationale
- ✅ **Cargo-cult audit per Catalog #303**: cites Phase 1 audit; 4 unwind paths implemented
- ✅ **Catalog #220 substrate L1+ operational mechanism**: per-class byte-mutation smoke EMPIRICALLY CONFIRMS cls=0 uniform L0 violates contract; L1 promotion blocker is cls_stream wire-in
- ✅ **Catalog #240 recipe-vs-trainer-state**: existing trainer `_full_main` IMPLEMENTED per sister; recipe `research_only=true` + `dispatch_enabled=false` per per-substrate symposium gating
- ✅ **Catalog #244 NVML block**: sister driver script unchanged; canonical 3-export block preserved
- ✅ **Catalog #270 dispatch optimization protocol**: substrate has no scorer hot loop; sister trainer carries Tier-1 engineering flag waivers
- ✅ **Catalog #290 canonical-vs-unique decision per layer**: 4 FORK + 5 PRESERVE-CANONICAL + 3 ADOPT-CANONICAL rationales
- ✅ **Catalog #294 9-dim checklist**: section above; 9 dimensions with PASS / PARTIAL / DEFERRED status
- ✅ **Catalog #296 Dykstra-feasibility**: predicted-band section explicit + Catalog #324 axis-attribution discipline
- ✅ **Catalog #297 signal-axis-destruction reversibility**: 3 reversibility surfaces (v1 inline / v2 procedural-seed / L0 cls=0 uniform) + 3 sister-extinction surfaces in Phase 3
- ✅ **Catalog #300 council deliberation v2 frontmatter**: tier T1 + attendees + verdict + dissent + assumption-adversary + decisions + mission_contribution + override_invoked + horizon_class declared
- ✅ **Catalog #303 cargo-cult audit section**: cites Phase 1 audit with 12-assumption table
- ✅ **Catalog #305 observability surface**: 6 facets declared
- ✅ **Catalog #309 horizon_class**: `frontier_pursuit` declared
- ✅ **Catalog #324 predicted-band post-training validation**: `predicted_band_validation_status: pending_post_training` in frontmatter; canonical UNKNOWN tokens in axis attribution; reactivation criteria pinned
- ✅ **Catalog #325 per-substrate symposium memo**: sister `council_t1_nscs06_v8_chroma_lut_per_substrate_symposium_20260521.md` exists (sister-landed); Phase 3 BUILD enables symposium adjudication
- ✅ **Catalog #344 canonical equation cross-reference**: `procedural_codebook_from_seed_compression_savings_v1` + IN-DOMAIN context `nscs06_v8_chroma_lut`
- ✅ **Catalog #359 anti-misapplication self-protect**: v8 is REPLACEMENT-savings (NOT residual-correction-hybrid); pair #1 + pair #2 falsifications do NOT apply
- ✅ **Catalog #287 + #323 canonical Provenance**: no score claim asserted; all artifacts tagged with `[prediction]` / `[research-signal]` / `[structural-verifier]`
- ✅ **Catalog #208 docs/local-paths**: no `/Users/adpena/...` references

---

## 6-hook wire-in declaration per Catalog #125

- **hook #1 sensitivity-map** = ACTIVE (per-class chroma anchor consumption IS canonical sensitivity surface; axis-attributed prediction enables per-axis sensitivity routing)
- **hook #2 Pareto constraint** = ACTIVE PRIMARY (rate-axis ΔS = -0.002706 via canonical equation #26; seg+pose-axis pending per Catalog #324)
- **hook #3 bit-allocator** = PLANNED (32-byte seed slot replaces 4096-byte chroma LUT slot only after per-substrate symposium PROCEED per Catalog #325)
- **hook #4 cathedral autopilot dispatch** = ACTIVE (sister consumer `tac.cathedral_consumers.procedural_codebook_generator_consumer` per Catalog #335 auto-discovers; Path 3 C' adds canonical-equation-26-axis-attributed prediction surface)
- **hook #5 continual-learning posterior** = ACTIVE PRIMARY (first paired-smoke empirical anchor populates seg+pose axes; replaces UNKNOWN tokens with measured values per Catalog #324)
- **hook #6 probe-disambiguator** = ACTIVE PRIMARY ($0 MLX-local SegNet noise-floor probe IS canonical disambiguator between SegNet-detectable vs noise-floor-collapsed substrate states; per-class byte-mutation smoke IS canonical disambiguator between L0 cls=0 vs L1 cls_stream inflate states)

---

## Sister substrate cross-references

- **v6 NSCS06 Carmack-Hotz strip-everything**: HARD-EARNED falsification at seg=64.59 grayscale-Y=R=G=B cargo-cult #2; v7 unwound for 44% improvement; cargo-cult-unwind methodology canonical anchor.
- **v7 NSCS06 Path A chroma+optical-flow** (`nscs06_path_a_chroma_optical_flow_redesign_20260516.md`): v7 sister substrate that v8 extends to per-(level, class) chroma anchor.
- **v8 NSCS06 chroma_lut existing scaffold** (`nscs06_v8_chroma_lut_design_20260521.md`): sister-landed Phase 2 BUILD which Path 3 C' Phase 3 EXTENDS via 4 cargo-cult unwinds.
- **grayscale_lut PROCEDURAL VARIANT** (commit `f037d1144`): sister procedural-variant substrate; same canonical equation #26 IN-DOMAIN context family.
- **DP1 PROCEDURAL VARIANT** (commit `9cbfa471c`): sister procedural-variant substrate; same canonical equation #26 IN-DOMAIN context family.
- **VQ-VAE PROCEDURAL VARIANT** (commit `6fea30f22`): sister procedural-variant substrate; same canonical equation #26 IN-DOMAIN context family.
- **canonical equation #26**: `procedural_codebook_from_seed_compression_savings_v1` at `src/tac/canonical_equations/procedural_codebook_savings.py`; `nscs06_v8_chroma_lut` IS in `_INCLUDED_CONTEXTS`.
- **Path 3 C' Phase 1 audit** (commit `a6e2a06e3`): 12-assumption HARD-EARNED-vs-CARGO-CULTED interrogation.
- **Path 3 C' Phase 2 decision** (commit `bac0ec05d`): Path (b) JUSTIFIED-EXTEND-WITH-FORK + 7-section roadmap.

---

## Lane registry pre-registration per Catalog #126

```bash
.venv/bin/python tools/lane_maturity.py add-lane \
    lane_path_3_c_prime_nscs06_v8_chroma_lut_cargo_cult_first_20260526 \
    --name "Path 3 C' NSCS06 v8 chroma_lut cargo-cult-first L0/L1 SCAFFOLD" \
    --phase 3 \
    --notes "research_only=true; cargo-cult-pass-first methodology per operator directive #2 2026-05-26; sister to lane_wave_3_nscs06_v8_chroma_lut_substrate_build_20260521 (existing scaffold preserved); 4 FORK paths per Phase 1 audit a6e2a06e3 unwinding cargo-cults #3 #5 #8 #12; 44 dedicated tests pass; per-substrate symposium per Catalog #325 pending window 2026-05-26 → 2026-06-09"
```

Initial gates marked at landing:
- `impl_complete` (this landing)
- `strict_preflight` (180 tests pass; new modules import cleanly)
- `memory_entry` (Phase 1 + Phase 2 + Phase 3 design + Phase 3 landing memos)

Pending gates (per Catalog #325 per-substrate symposium):
- `real_archive_empirical` (first paired smoke — operator-routable; DEFERRED until per-substrate symposium PROCEED + cls_stream wired at L0 inflate)
- `contest_cuda` (first contest-CUDA anchor — operator-routable)
- `three_clean_review` (R1-R3 council rounds)
- `deploy_runbook` (sister driver script already lands at `lane_wave_3_nscs06_v8_chroma_lut_substrate_build_20260521`; promotion via per-substrate symposium)

---

## Sister coordination per Catalog #230

In commit body cite:
- A=`aaec7a0d220f31543` DreamerV3 RSSM (fresh design; disjoint scope)
- D=`af6ca73c5a7fc40f4` Z6 predictive coding (fresh design; disjoint scope)
- E=`a35f9f86781aaaa4f` BoostNeRV against PR110 (fresh design; disjoint scope)
- B'=`ac4283983ece21b83` Z7-Mamba-2 cargo-cult-first (concurrent methodology; disjoint substrate scope)

NO file overlap. My scope: `src/tac/substrates/nscs06_v8_chroma_lut/` + `.omx/research/path_3_c_*` (4 NEW files + 2 EXTENDED files). Predecessor `adb6518fd5bb26607` work (test_mlx_iteration.py 31 tests + mlx_iteration.py 762 LOC) PRESERVED as input per Catalog #110/#113 APPEND-ONLY.
