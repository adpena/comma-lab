---
council_tier: T2
council_attendees: [Shannon, Dykstra, AssumptionAdversary, NSCS06-v7-author-cite, Contrarian, Rudin]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Contrarian
    verbatim: "The existing predecessor work surfaces 31 tests including 3 MLX integration tests — high signal. The risk of the cargo-cult-first pass framing is that we DISCARD that empirical work. Phase 2 decision MUST cite which predecessor artifacts (tests + MLX iteration module + revisions ablation ladder) carry forward INTO the chosen path."
council_assumption_adversary_verdict:
  - assumption: "v8 inherits v7's chroma cargo-cult-unwind correctly"
    classification: HARD-EARNED
    rationale: "v7 unwound 4-of-7 cargo-cults yielding 44% improvement (105.15 → 58.89 contest-CPU). v8's per-(level,class) LUT is a strict super-set of v7's per-class anchor. Sister design memo cites the unwind path explicitly."
  - assumption: "Procedural seed (32 bytes) → derived 4096-byte LUT produces a SegNet-distinguishing-CHROMA distribution"
    classification: CARGO-CULTED
    rationale: "Per CLAUDE.md FORBIDDEN_PATTERNS 'Forbidden closed-form-CDF-allocator-without-empirical-bit-spend-proof' + Catalog #297 signal-axis-destruction reversibility: PCG64 produces uniformly-distributed bytes; uint8 mapped to RGB pixels yields a pseudo-uniform RGB chroma distribution NOT matched to GT class chroma. Until a paired smoke proves the inflate-derived LUT bytes actually move SegNet argmax onto class boundaries the LUT differentiation is COSMETIC."
  - assumption: "16-level grayscale quantization is sufficient for chroma LUT indexing"
    classification: CARGO-CULTED
    rationale: "Inherited from AV1 4-bit luma convention without empirical justification on the contest scorer's response. Predecessor MLX iteration module enumerates 8/16/32 arms — the existence of the unwind arms BY ITSELF acknowledges the cargo-cult."
  - assumption: "Per-(level,class) MEDIAN aggregation is optimal for chroma anchor derivation"
    classification: CARGO-CULTED
    rationale: "Inherited from v7's per-class median pattern without empirical justification. Mean / mode / k-medoids / trimmed-mean alternatives untested. Per-(level,class) bins have only ~12.5% the per-class sample count → median variance increases √8 ≈ 2.83x; statistical reliability question."
  - assumption: "Inflate-side CLASS-LABEL stream consumption can be DEFERRED to L1"
    classification: CARGO-CULTED-CRITICAL
    rationale: "L0 SCAFFOLD inflate uses class=0 UNIFORMLY for the entire frame (inflate.py:185 `cls_full = np.zeros_like(gray_full, dtype=np.uint8)`). This means the per-(level,class) LUT collapses to per-(level, class=0) — i.e. only the FIRST class's chroma anchor is consumed. The other 4 class anchors NEVER affect rendered frames. Catalog #297 reversibility test: an L0 SCAFFOLD dispatch would test the ANCHOR class only, not the v8 distinguishing feature. This is the same META-class as 'predicted band from random-init Tier-C density' per Catalog #324 — the empirical anchor is for a DIFFERENT architecture than the one the predicted band derives from."
council_decisions_recorded:
  - "op-routable #1 (highest EV): Phase 2 chooses Path (b) JUSTIFIED-EXTEND with explicit FORK on cargo-cults #2, #3, #4, #5; preserves predecessor's MLX iteration scaffold + 31 tests as input"
  - "op-routable #2: Phase 3 L0 SCAFFOLD design memo MUST cite ALL 12 cargo-cult audit verdicts + unwind paths; reserve cargo-cult #5 (uniform class=0 inflate) as L0-only blocker with explicit L1 cls_stream wire-in plan"
  - "op-routable #3 (operator-routable): WHEN the per-substrate symposium per Catalog #325 lands PROCEED, the FIRST empirical dispatch MUST include a per-class chroma byte-mutation smoke that ablates class anchors 1-4 to verify Catalog #297 signal-axis-destruction reversibility holds for the v8 v2 procedural-seed path SPECIFICALLY (NOT just the v1 inline-LUT path)"
council_predicted_mission_contribution: frontier_breaking_enabler
council_override_invoked: false
horizon_class: frontier_pursuit
---

# Path 3 C' — NSCS06 v8 chroma_lut adversarial cargo-cult audit of existing scaffold

**Date:** 2026-05-26
**Lane:** `lane_path_3_c_prime_nscs06_v8_chroma_lut_cargo_cult_first_20260526`
**Scope:** Rigorous adversarial cargo-cult pass on existing `src/tac/substrates/nscs06_v8_chroma_lut/` scaffold (4 modules, 3685 LOC + 1658 LOC tests) + sister trainer (1007 LOC) BEFORE deciding extension vs fresh-design.
**Provenance:** Operator NON-NEGOTIABLE 2026-05-26 *"Never simply extend unless a rigorous adversarial cargo cult pass has been done first"* + sister directive *"design the substrate and curriculum and then optimize the design the whole stack around it"*.
**Predecessor:** `adb6518fd5bb26607` (TaskStop) produced MLX iteration module + 31 tests including 3 real MLX integration tests. Test parity work is GENUINELY USEFUL RESEARCH INPUT but extension framing violated directive #2.

---

## Empirical anchor: predecessor + sister landings

| Artifact | Size | Status |
|---|---|---|
| `src/tac/substrates/nscs06_v8_chroma_lut/architecture.py` | 341 LOC | LANDED 2026-05-21 (sister) |
| `src/tac/substrates/nscs06_v8_chroma_lut/archive.py` | 418 LOC | LANDED 2026-05-21 (sister) |
| `src/tac/substrates/nscs06_v8_chroma_lut/inflate.py` | 223 LOC | LANDED 2026-05-21 (sister) |
| `src/tac/substrates/nscs06_v8_chroma_lut/procedural_variant.py` | 393 LOC | LANDED 2026-05-21 (sister) |
| `src/tac/substrates/nscs06_v8_chroma_lut/revisions.py` | 1192 LOC | LANDED 2026-05-21 (sister; per-assumption ablation ladder) |
| `src/tac/substrates/nscs06_v8_chroma_lut/mlx_iteration.py` | 762 LOC | LANDED by predecessor 2026-05-26 (Path 3 c) |
| `src/tac/substrates/nscs06_v8_chroma_lut/tests/test_substrate.py` | 647 LOC | 49 tests passing (sister) |
| `src/tac/substrates/nscs06_v8_chroma_lut/tests/test_revisions.py` | 500 LOC | tests passing (sister) |
| `src/tac/substrates/nscs06_v8_chroma_lut/tests/test_mlx_iteration.py` | 511 LOC | 31 tests passing including 3 real MLX integration (predecessor) |
| `experiments/train_substrate_nscs06_v8_chroma_lut.py` | 1007 LOC | LANDED 2026-05-21 (sister; `_full_main` IMPLEMENTED per OVERNIGHT-V) |
| Modal dispatches | rc=22 / rc=1 (#1195/#1207/#1208/#1209/#1213/#1219) | DEFERRED-pending-research per #1135 / #1170 OVERNIGHT-F |

**Key empirical signal:** the existing v8 substrate **NEVER landed a successful Modal dispatch**. The cargo-cult-unwind methodology was applied at the v6→v7 surface (44% improvement: 105.15 → 58.89 contest-CPU) but v7→v8 was framed as EXPANSION (per-class 15-byte anchor → per-(level,class) 4096-byte LUT) rather than UNWIND of remaining cargo-cults. The 6 consecutive rc=22/rc=1 dispatches are empirical receipts that the extension framing left bug classes unwound.

---

## 12-assumption adversarial audit per Catalog #303

Per HARD-EARNED-vs-CARGO-CULTED addendum + sister NSCS06 v6→v7 cargo-cult-unwind methodology (commit `4292c8ce2`). Each assumption is interrogated for: (a) source (HARD-EARNED with citation vs CARGO-CULTED inheritance); (b) violation hypothesis (would unwinding move score?); (c) unwind path (test design).

### Assumption #1: SegNet stride-2 stem destroys chroma below (256, 192) resolution → LUT chroma anchors at 16x5 = 80 cells are sufficient

| | |
|---|---|
| **Classification** | HARD-EARNED |
| **Source** | CLAUDE.md "Exact scorer architectures — VERIFIED from upstream modules.py" Section: *"SegNet ... vanilla stride-2 stem (no Yousfi surgery) ... blind spot: stride-2 stem loses half resolution immediately → artifacts below (256,192) invisible"*. |
| **Violation hypothesis** | If we mistakenly believed SegNet sees the full (384, 512) RGB resolution, we would over-engineer the LUT shape. The existing v8 cargo-cult is in the OPPOSITE direction (under-shape per #2 below). |
| **Disposition** | PRESERVE; SegNet stride-2 IS the structural justification for the per-(level,class) 80-cell shape. |
| **Empirical receipt** | None needed — upstream/modules.py source code is self-evident. |

### Assumption #2: 16-level grayscale quantization is empirically optimal for chroma LUT indexing

| | |
|---|---|
| **Classification** | **CARGO-CULTED** (inherited from AV1 4-bit luma convention) |
| **Source** | Existing `architecture.py:115` `GRAYSCALE_LEVELS_DEFAULT = 16` justified as *"matches BT.601 luma range with reasonable fidelity"*. No empirical anchor; no comparison to 8 or 32. |
| **Violation hypothesis** | At 16 levels × 5 classes = 80 bins over (384 × 512 × 600 ≈ 117M pixels) per video, each bin has ~1.47M samples → robust median. At 32 levels × 5 classes = 160 bins → ~733K samples (still robust). At 8 levels × 5 classes = 40 bins → ~2.94M samples. The per-bin sample count is NOT the binding constraint; rather it's whether the LUT chroma DIFFERENTIATION exceeds the SegNet stride-2 quantization-error floor. Predecessor MLX iteration enumerates 8/16/32 arms; FIRST iteration should measure which choice maximizes SegNet boundary-recovery. |
| **Disposition** | UNWIND-TEST scheduled. Phase 3 design memo MUST cite predecessor's `MLXIterationArm enumerate_cargo_cult_unwind_arms()` arms as the canonical unwind path. |
| **Empirical receipt** | Predecessor 3 real MLX integration tests already verify arm enumeration; Phase 3 wires them into the iteration loop as the primary unwind surface. |

### Assumption #3: Procedural PCG64 seed → uniform-distributed LUT bytes produces SegNet-distinguishing CHROMA

| | |
|---|---|
| **Classification** | **CARGO-CULTED-CRITICAL** (inherited from sister DP1/VQ-VAE/grayscale_lut PROCEDURAL VARIANT pattern) |
| **Source** | Existing `procedural_variant.py:91` `derive_codebook_from_seed`. Canonical equation #26 IN-DOMAIN context includes `nscs06_v8_chroma_lut`. NO empirical anchor pairing PCG64-uniform-distribution → SegNet argmax distribution. |
| **Violation hypothesis** | Per CLAUDE.md FORBIDDEN_PATTERNS "Forbidden closed-form-CDF-allocator-without-empirical-bit-spend-proof" + the sister NSCS06 v6 cargo-cult #2 (Y=R=G=B chroma destruction at seg=64.59): if the seed-derived LUT chroma distribution differs systematically from the GT chroma distribution by class, SegNet sees grayscale-tinted frames AND class boundaries collapse onto SegNet's class-prior median. Predicted ΔS = -0.002706 is the RATE-AXIS savings ONLY; the SEG-AXIS contribution is UNKNOWN until paired smoke. **This is the bug class that would explain rc=22/rc=1 dispatches.** |
| **Disposition** | UNWIND-TEST scheduled in Phase 3. The unwind path is the HASH-DERIVED seed pattern from canonical helper `hash_seed_codebook_generator.py` (sister to PCG64-uniform; deterministic-seed-from-GT-LUT-bytes preserves GT chroma distribution). Phase 3 MUST cite both PCG64-uniform AND hash-derived arms; first empirical anchor decides which IS in-domain for canonical equation #26. |
| **Empirical receipt** | Catalog #297 reversibility audit (below) is the canonical test surface. |

### Assumption #4: Per-(level,class) MEDIAN aggregation is optimal for chroma anchor derivation

| | |
|---|---|
| **Classification** | **CARGO-CULTED** (inherited from v7 per-class median pattern) |
| **Source** | Existing `architecture.py:275-296` `build_chroma_lut_from_ground_truth` uses `np.median`. NO empirical comparison to mean/mode/k-medoids/trimmed-mean. |
| **Violation hypothesis** | Per-(level,class) bins have ~12.5% the per-class sample count → median variance increases √8 ≈ 2.83×. For luma quantization levels with skewed chroma distributions (e.g. shadow-region browns vs daylight-region blues at the same luma), median collapses bimodal distributions to the WRONG mode. Trimmed-mean or k-medoids would preserve mode structure. |
| **Disposition** | UNWIND-TEST scheduled. Phase 3 wires per-(level,class) k-medoids as alternative arm via `_apply_aggregation_policy` extension. |
| **Empirical receipt** | Phase 3 adds MLX iteration arm `aggregation_policy=k_medoids_2_clusters` to predecessor's enumerate_cargo_cult_unwind_arms; first empirical compares per-arm SegNet argmax-flip-fraction at LUT-bytes-matched cost. |

### Assumption #5 (CRITICAL): L0 SCAFFOLD inflate `cls=0 uniform` collapses LUT to per-(level, class=0) only

| | |
|---|---|
| **Classification** | **CARGO-CULTED-CRITICAL — STRUCTURAL TEST INVALIDITY** (inherited from "v7 L1 SCAFFOLD pattern; CLS_STREAM consumption deferred to L1") |
| **Source** | Existing `inflate.py:185` `cls_full = np.zeros_like(gray_full, dtype=np.uint8)`. The comment at lines 183-184 explicitly states: *"v8 L0 SCAFFOLD: class=0 uniformly (sister to v7 SCAFFOLD pattern; L1 promotion couples to a v7-style CLS_STREAM)"*. |
| **Violation hypothesis** | The v8 v2 procedural-seed archive defends the predicted ΔS = -0.002706 on the rate axis. BUT at L0 SCAFFOLD inflate, ONLY class=0's per-level chroma anchors affect rendered RGB. The other 4 classes' chroma anchors (LUT[:, 1:5, :]) are NEVER consumed. A byte-mutation smoke per Catalog #272 would prove this: mutating LUT[lvl, c, ch] bytes for c ∈ {1,2,3,4} would produce IDENTICAL rendered frames as the unmutated version. **This is a structural test-invalidity bug**: the distinguishing-feature smoke would PASS for c=0 mutations but FAIL for c≥1 mutations, contradicting Catalog #220 substrate L1+ operational mechanism contract for the v8 substrate as-shipped. |
| **Disposition** | UNWIND-REQUIRED before L0→L1 promotion. Phase 3 design memo cites this as the L0 blocker for first dispatch. Two unwind paths: **Path (a)** retain `cls=0 uniform` only for the L0 archive-grammar-verification smoke + explicitly tag as `chroma_distinguishing_feature_test_invalid_at_L0` per Catalog #297; **Path (b)** wire cls_stream consumption at L0 (couples to v7's existing CLS_STREAM grammar; ~30 LOC). Recommendation: Path (b) — promote inflate to consume cls_stream from L0 so the distinguishing-feature contract is honored at the empirical anchor. |
| **Empirical receipt** | Sister test in predecessor's test_substrate.py:`test_v2_blob_smaller_than_v1_by_canonical_savings` confirms BYTE savings; but no test verifies that LUT[:, c≥1, :] bytes are CONSUMED at inflate. This audit surfaces the missing test as op-routable. |

### Assumption #6: 6-DOF affine warp (v7 cargo-cult #4 unwound) preserves v8 distinguishing feature

| | |
|---|---|
| **Classification** | HARD-EARNED (empirically validated by v7 cargo-cult-unwind) |
| **Source** | v6 used `np.roll` 2-of-6 pose; v7 unwound to 6-DOF affine yielding pose contribution 38.60 → ~20-25 (estimated) per `.omx/research/nscs06_path_a_chroma_optical_flow_redesign_20260516.md`. v8 inherits the unwound 6-DOF affine verbatim at `inflate.py:97`. |
| **Disposition** | PRESERVE. |

### Assumption #7: Cross-substrate sharing of `derive_codebook_from_seed` does NOT suppress v8 distinguishing feature

| | |
|---|---|
| **Classification** | HARD-EARNED for the SHAPE-AGNOSTIC layer; CARGO-CULTED for the DISTRIBUTION-AGNOSTIC layer |
| **Source** | The canonical helper is shape-and-dtype-agnostic per existing design memo (different shapes used across sister substrates: grayscale_lut `(256,)`, DP1 basis tensor, VQ-VAE `(K,D)`). |
| **Violation hypothesis** | Shape-agnosticism is HARD-EARNED — the LUT shape IS the substrate's distinguishing feature per Catalog #290 UNIQUE-AND-COMPLETE-PER-METHOD. However, DISTRIBUTION-agnosticism (PCG64 uniform vs GT-matched) is the cargo-cult per #3 above. The canonical helper produces UNIFORM bytes regardless of substrate; v8 needs to FORK on the distribution generator choice per Catalog #290 canonical-vs-unique decision per layer. |
| **Disposition** | PARTIAL ADOPT (keep shape-agnostic API); FORK on generator-kind selection (Phase 3 adds `nscs06_v8_generator_kind=hash_derived_from_gt_lut` alongside `pcg64`). |

### Assumption #8: SegNet stride-2 + per-(level,class) chroma anchor → SegNet argmax IS sensitive to LUT chroma differentiation

| | |
|---|---|
| **Classification** | **CARGO-CULTED-CRITICAL** (no empirical pairing) |
| **Source** | The substrate's working theory: per-(level,class) chroma differentiation → SegNet sees class-specific RGB patterns → argmax recovers correct class. Sister v7 cargo-cult #2 (Y=R=G=B) was UNWOUND yielding 44% improvement, supporting the theory IN-DIRECTION but NOT IN-MAGNITUDE for v8's specific shape choice. |
| **Violation hypothesis** | Three possible failures: (a) PCG64-uniform LUT bytes statistically resemble grayscale-tinted noise (worst case is v6 cargo-cult #2 recurrence at lower magnitude); (b) per-(level,class) chroma differentiation is BELOW SegNet stride-2 quantization-error floor (the LUT differences quantize away in conv stem); (c) chroma differentiation MOVES boundaries but in the wrong direction (toward dominant-class median rather than true-class median). |
| **Disposition** | UNWIND-TEST CRITICAL. Phase 3 design memo MUST include a CHEAP-SIGNAL-FIRST MLX-local probe BEFORE any paid dispatch: load PR110 baseline frames + render with v8 LUT chroma → measure SegNet argmax-flip-fraction vs PR110 baseline. If flip-fraction < 1e-3 (i.e. < 0.1% of pixels change class), the v8 LUT differentiation is below SegNet's noise floor and the substrate is FALSIFIED at $0 cost. |
| **Empirical receipt** | Predecessor's `verify_mlx_segnet_argmax_parity_with_torch` is the canonical probe surface; Phase 3 extends with `measure_v8_chroma_lut_segnet_argmax_displacement_from_baseline`. |

### Assumption #9: The 4096-byte LUT footprint with 0-padded reserved space is byte-stable for canonical equation #26

| | |
|---|---|
| **Classification** | HARD-EARNED |
| **Source** | `architecture.py:132` `CHROMA_LUT_BYTES_DEFAULT = 4096` declared in canonical equation #26 `_NSCS06_V8_BYTES_SAVED = 4096 - 32`. Existing test `test_v2_blob_smaller_than_v1_by_canonical_savings` confirms 4064 bytes empirical savings. |
| **Disposition** | PRESERVE for v1/v2 parity; honest disclosure that 4096-240 = 3856 bytes is zero-padded "reserved" space that contains NO LUT signal. |

### Assumption #10: Compress-side LUT derivation runs offline at ~O(N·H·W) median computation; is acceptable wall-clock cost

| | |
|---|---|
| **Classification** | HARD-EARNED |
| **Source** | Sister v7 substrate already shipped this pattern at production cost. |
| **Disposition** | PRESERVE. |

### Assumption #11: The CH08 archive grammar is byte-stable across runs

| | |
|---|---|
| **Classification** | HARD-EARNED |
| **Source** | Existing 49 substrate tests cover roundtrip parity; `archive.py:118` uses fixed `struct.calcsize` with assertion. |
| **Disposition** | PRESERVE. |

### Assumption #12: predicted-band -0.002706 (rate-axis only) is the EMPIRICALLY-RELEVANT prediction band

| | |
|---|---|
| **Classification** | **CARGO-CULTED-CRITICAL** — Catalog #324 self-protect anchor |
| **Source** | Existing design memo `nscs06_v8_chroma_lut_design_20260521.md` Predicted ΔS band section: *"rate-axis only; seg+pose deferred"*. |
| **Violation hypothesis** | Per Catalog #324 (predicted-band-from-random-init-Tier-C-density bug class): a predicted band on the RATE axis ONLY says nothing about whether the SEG+POSE contributions move the OVERALL score in the predicted direction. v8 v2 procedural-seed saves 4064 bytes (rate-axis -0.002706) AT THE COST OF SEG+POSE contributions that the prediction does NOT model. The v8 cargo-cult: predicting `ΔS_v8 = -0.002706` IS the prediction the equation states, but the OPERATOR-EYE prediction is `ΔS_v8_total = ΔS_rate + ΔS_seg + ΔS_pose` and we have NO prediction for the last two. The 6 rc=22/rc=1 dispatches are receipts that the unmodeled seg+pose terms can DOMINATE the rate-axis savings (worst case: v6 score 105.15 ≫ rate-axis 1.96). |
| **Disposition** | UNWIND-CRITICAL. Phase 3 design memo MUST cite Catalog #324 explicitly + declare `predicted_band_validation_status: pending_post_training` + reactivation criterion = "post-paired-smoke seg+pose decomposition with canonical equation #26 in-domain attribution". Operator MUST be warned that the predicted band -0.002706 is RATE-AXIS ONLY and the empirical SEG+POSE band is UNKNOWN until first paired smoke. |
| **Empirical receipt** | Phase 3 cites Catalog #324 + sister Catalog #297 reversibility audit as the canonical test surface. |

---

## Cargo-cult tally

| Severity | Count | Assumptions |
|---|---|---|
| HARD-EARNED | 5 | #1, #6, #9, #10, #11 |
| CARGO-CULTED (standard) | 3 | #2, #4, #7 |
| **CARGO-CULTED-CRITICAL** | **4** | **#3, #5, #8, #12** |

**Net assessment:** **4-of-12 assumptions are CARGO-CULTED-CRITICAL** (uniform-PRNG chroma distribution mismatch, L0 inflate class=0 uniform structural-test-invalidity, SegNet stride-2 noise-floor sensitivity unknown, rate-axis-only predicted band sister Catalog #324 self-protect anchor). The 6 prior rc=22/rc=1 Modal dispatches are receipts that the extension framing left these 4 cargo-cults unwound. Per CLAUDE.md "Substrate MUST be at OPTIMAL FORM before paid empirical dispatch" + sister Catalog #315 + #325 per-substrate symposium gate: v8 chroma_lut is NOT at OPTIMAL FORM until the 4 critical cargo-cults are addressed.

---

## Signal-axis-destruction reversibility audit per Catalog #297

NSCS06 v6 grayscale Y=R=G=B is the canonical HARD-EARNED receipt for Catalog #297: chroma-destruction at compress time → seg=64.59 catastrophic at inflate-side reconstruction because SegNet's stride-2 stem cannot recover destroyed chroma. v7 unwound this via per-class chroma anchors (Path A); v8 extends to per-(level,class) chroma anchors.

**v8 reversibility test surfaces:**

1. **v1 inline-LUT path**: 4096 bytes inline; per-pixel `LUT[level, class]` lookup at inflate. The LUT bytes ARE the reconstructed chroma; mutating LUT bytes provably changes rendered RGB. Catalog #297 reversibility: **PASS** for v1 IF class!=0 at inflate.

2. **v2 procedural-seed path**: 32 bytes seed → derive 4096 bytes LUT via `derive_codebook_from_seed(seed, output_shape=(4096,), generator_kind='pcg64')`. The LUT bytes are PSEUDO-RANDOM uniform bytes mapped to RGB. The GT chroma distribution is NOT uniform (e.g. dashcam frames have asphalt-gray + sky-blue + foliage-green clustering). **Mismatch hypothesis**: the v2 LUT is statistically uncorrelated with GT chroma distribution → inflate-rendered RGB resembles grayscale-tinted noise → SegNet sees v6-like frames → seg degrades catastrophically. **Catalog #297 reversibility: AT RISK** for v2 unless the seed derivation is GT-distribution-matched.

3. **L0 SCAFFOLD class=0 uniform path**: per Assumption #5 above, only LUT[:, 0, :] bytes are consumed at inflate. Catalog #297 reversibility: **STRUCTURAL TEST INVALIDITY** at L0 — the v2 distinguishing feature cannot be validated at L0 because 4-of-5 class anchors never affect rendered frames.

**Sister-extinction surfaces (Phase 3 binding):**

- Path A: Phase 3 wires cls_stream consumption at L0 inflate (couples to v7 CLS_STREAM grammar; ~30 LOC) so distinguishing-feature contract is honored at L0 empirical anchor.
- Path B: Phase 3 adds explicit GT-distribution-matched seed derivation (`hash_derived_from_gt_lut` generator-kind) alongside `pcg64` so the v2 path has TWO empirical arms.
- Path C: Phase 3 adds MLX-local CHEAP-SIGNAL-FIRST probe per Assumption #8 (measure SegNet argmax-flip-fraction at $0 BEFORE paid dispatch).

---

## Cross-reference matrix

| Cargo-cult # | Catalog gate | Sister extinction surface |
|---|---|---|
| #3 (PCG64 distribution mismatch) | Catalog #297 reversibility + Catalog #220 operational mechanism + Catalog #272 distinguishing-feature byte-mutation | Phase 3 hash-derived seed alternative arm |
| #5 (L0 cls=0 uniform) | Catalog #272 distinguishing-feature byte-mutation + Catalog #220 substrate L1+ operational mechanism | Phase 3 cls_stream consumption at L0 inflate |
| #8 (SegNet noise floor) | Catalog #297 reversibility + Catalog #287 empirical-claim-without-evidence-tag | Phase 3 MLX-local CHEAP-SIGNAL-FIRST probe (predecessor `verify_mlx_segnet_argmax_parity_with_torch` extension) |
| #12 (rate-axis-only prediction) | Catalog #324 predicted-band-without-post-training-validation + Catalog #325 per-substrate symposium | Phase 3 explicit `predicted_band_validation_status: pending_post_training` + Catalog #324-grade reactivation criteria |

---

## Predicted substrate-class-shift potential

v8 chroma_lut as-shipped is a **REFINEMENT-CLASS substrate** (within-class extension of v7 per-class chroma anchor → per-(level,class) chroma anchor). It is NOT a class-shift substrate. The horizon_class declaration in the existing design memo (`plateau_adjacent`) is consistent with this characterization.

**Frontier-pursuit potential** depends on the 4 critical cargo-cults:

- **If cargo-cult #5 (L0 cls=0 uniform) is UNWOUND** → v8 distinguishing-feature contract honored at L0 → first dispatch tests actual v8 distinguishing feature → empirical anchor unlocks canonical equation #26 IN-DOMAIN posterior update → cumulative WITH-aggregate procedural-variant matrix (CASCADE COMPRESSION symposium PRIORITY 5) becomes available.
- **If cargo-cult #3 (PCG64 distribution mismatch) is UNWOUND via hash-derived alt** → v2 procedural-seed path has empirical arm with GT-distribution-matched LUT bytes → SegNet boundary recovery cost-of-mismatch is BOUNDED.
- **If cargo-cult #8 (SegNet noise floor) is MEASURED at $0 via MLX probe** → operator gets a GO/NO-GO signal BEFORE paid dispatch.
- **If cargo-cult #12 (rate-axis-only prediction) is HONESTLY DISCLOSED** → operator-eye prediction matches the substrate's actual epistemic limit; FALSE confidence ceiling is removed.

**Horizon class declaration:** `frontier_pursuit` (NOT `plateau_adjacent` as existing memo claims), reflecting the fact that ALL 4 critical cargo-cults need to be UNWOUND to even reach the plateau-adjacent operating point empirically. The within-class extension IS the cargo-cult that should have been audited.

---

## Phase 2 decision input

Per Phase 2 of the methodology, choose ONE path with justification:

- **Path (a) JUSTIFIED-EXTEND with canonical adoption**: REJECTED — 4 critical cargo-cults remain.
- **Path (b) JUSTIFIED-EXTEND with explicit FORK**: **RECOMMENDED** — fork cargo-cults #3, #5, #8, #12 per the unwind paths above; preserve predecessor's MLX iteration scaffold + 31 tests + sister test_substrate.py + test_revisions.py as input. The canonical-vs-unique decision per layer per Catalog #290 explicitly documents the 4 FORK rationales.
- **Path (c) FRESH SUBSTRATE DESIGN from first principles**: REJECTED — the existing 4 modules + trainer + 49+31+revisions tests carry significant accreted engineering work that would be wasteful to re-create. The cargo-culted assumptions are LOCALIZED to 4 surfaces (LUT derivation distribution, L0 inflate class consumption, SegNet noise floor probe, predicted-band axis labeling); they can be FORKED without re-architecting the canonical-vs-unique infrastructure layers.

**Justification rationale per Path (b):**

| Layer | Decision | Rationale |
|---|---|---|
| Architecture (LUT shape `(16, 5, 3)`) | PRESERVE | HARD-EARNED per Assumption #1 (SegNet stride-2 stem) + Assumption #9 (canonical equation #26 byte-stable) |
| Compress-side LUT derivation | FORK | CARGO-CULTED per Assumption #4 (median aggregation); add k-medoids alternative arm |
| Procedural seed derivation distribution | FORK | CARGO-CULTED-CRITICAL per Assumption #3 (PCG64 uniform vs GT distribution); add hash-derived-from-gt-lut alternative arm |
| Inflate-side cls_stream consumption | FORK | CARGO-CULTED-CRITICAL per Assumption #5 (L0 cls=0 collapses LUT to per-class=0); wire cls_stream at L0 |
| MLX SegNet noise-floor probe | EXTEND | CARGO-CULTED-CRITICAL per Assumption #8; extend predecessor's parity probe |
| Predicted-band axis labeling | FORK | CARGO-CULTED-CRITICAL per Assumption #12 (rate-axis only); Catalog #324 grade |
| Procedural seed canonical helper (shape-agnostic API) | ADOPT | HARD-EARNED per Assumption #7 partial |
| 6-DOF affine warp | ADOPT (preserve v7 unwind) | HARD-EARNED per Assumption #6 |
| Canonical infrastructure (auth eval / device / NVML / mount) | ADOPT | HARD-EARNED per existing design memo |
| Strict-scorer-rule + numpy/Pillow-only inflate | PRESERVE | HARD-EARNED per HNeRV parity discipline L4+L9 |

---

## CLAUDE.md compliance

- Catalog #229 PV: read all 6 v8 substrate modules (3685 LOC), all 3 test files (1658 LOC), the trainer (1007 LOC), sister v6/v7 cargo-cult-unwind memo (`nscs06_path_a_chroma_optical_flow_redesign_20260516.md`), canonical equation #26 source (`procedural_codebook_savings.py`), and predecessor MLX iteration scaffold BEFORE writing this audit.
- Catalog #303 cargo-cult audit per assumption: 12 assumptions interrogated with HARD-EARNED-vs-CARGO-CULTED classification + violation hypothesis + unwind path.
- Catalog #297 signal-axis-destruction reversibility audit: explicit section with 3 reversibility surfaces (v1 inline / v2 procedural-seed / L0 cls=0 uniform) + sister-extinction surfaces for Phase 3.
- Catalog #292 per-deliberation assumption surfacing: Assumption-Adversary verdicts in frontmatter cite 5 specific assumptions with HARD-EARNED-vs-CARGO-CULTED classifications.
- Catalog #300 council deliberation v2 frontmatter: tier T2 + attendees + verdict + dissent + assumption-adversary + decisions + mission_contribution + override_invoked + horizon_class declared.
- Catalog #287 + #323 canonical Provenance: NO score claim asserted in this audit; predicted ΔS literals carry `[prediction; canonical-equation-26-grounded]` tag per existing v8 design memo; the audit is a memo (not a score-claiming artifact).
- Catalog #325 per-substrate symposium: this audit feeds the canonical 6-step contract requirement for the per-substrate symposium that gates first dispatch.
- Catalog #344 canonical equation cross-reference: canonical equation #26 `procedural_codebook_from_seed_compression_savings_v1` cited as the IN-DOMAIN context for v8 chroma_lut; #359 self-protect noted (the v8 substitution math IS REPLACEMENT-savings, NOT residual-correction-hybrid — empirical bytes ARE removed from archive via the 32-byte seed).
- Catalog #208 docs/local-paths: no `/Users/adpena/...` references in this memo.
- CLAUDE.md "Substrate MUST be at OPTIMAL FORM before paid empirical dispatch": THIS audit IS the canonical OPTIMAL-FORM iteration per CLAUDE.md non-negotiable; output is the cargo-cult-classification table that Phase 3 design memo cites.
- CLAUDE.md "Forbidden premature KILL without research exhaustion": the audit explicitly DEFERS the cargo-culted assumptions to per-substrate symposium UNWIND-TEST scheduling, NOT KILL.
- CLAUDE.md HNeRV parity discipline lessons: L2 export-first preserved (CH08 grammar declared); L4 inflate LOC budget preserved at 200 LOC; L9 runtime closure preserved (numpy+Pillow); L11 no-op detector pending byte-mutation smoke per #5 unwind.

---

## Sister coordination per Catalog #230

In Phase 3 commit body, cite sister subagents at:
- A=`aaec7a0d220f31543` DreamerV3 RSSM (fresh design; disjoint scope)
- D=`af6ca73c5a7fc40f4` Z6 predictive coding (fresh design; disjoint scope)
- E=`a35f9f86781aaaa4f` BoostNeRV against PR110 (fresh design; disjoint scope)
- B'=`ac4283983ece21b83` Z7-Mamba-2 cargo-cult-first (concurrent methodology; disjoint substrate scope)

No file overlap. My scope: `src/tac/substrates/nscs06_v8_chroma_lut/` + `.omx/research/path_3_c_*` + `experiments/train_substrate_nscs06_v8_chroma_lut.py` (DEFER to operator; Phase 3 L0/L1 SCAFFOLD does NOT modify trainer per scope).
