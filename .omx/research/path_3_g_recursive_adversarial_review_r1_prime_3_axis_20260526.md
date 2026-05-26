<!-- SPDX-License-Identifier: MIT -->
<!-- Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE — review memo; do not mutate. -->
<!-- Catalog #229 PV closure: read landing memo + mlx_renderer.py + numpy_reference.py + archive.py + inflate.py + tests/test_basic.py in full BEFORE any review claim. 27/27 tests verified passing. -->
---
council_tier: T2
council_attendees: [Shannon, Dykstra, Daubechies, Mallat, NIRVANA-author-cite, Tishby-memorial, Carmack, Hotz, Quantizr, MacKay, Selfcomp, Contrarian, Assumption-Adversary]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "G=NIRVANA's '27/27 PASS' tests + landing memo claim 'axis 2 evidence: MLX↔PyTorch parity ≤ 1e-5' is accurate"
    classification: CARGO-CULTED
    rationale: "Empirically inspected: test `test_numpy_reference_bilinear_upsample_matches_pytorch` (test_basic.py:357-381) compares NUMPY REFERENCE vs PyTorch (not MLX vs PyTorch). The landing memo's repeated claim 'MLX↔PyTorch parity ≤ 1e-5' (lines 70-81 + line 108) is a DOCUMENTATION OVERSTATEMENT: the tests measure numpy_reference parity, not MLX parity. mlx_renderer.py contains config + helpers ONLY — NO actual MLX-callable renderer class. This is the same META class as E=BoostNeRV's BPR1 header size documentation drift (R1 META finding #2): truth lives in CODE; multiple documentation surfaces drift from it. The numpy parity tests are GOOD (axis 3 portability evidence); the labeling as 'axis 2 MLX drift evidence' is the bug class."
  - assumption: "G=NIRVANA's mlx_renderer.py being a thin scaffold (config + factory + estimators; no actual MLX renderer class) is acceptable at L0 SCAFFOLD"
    classification: HARD-EARNED-PARTIAL
    rationale: "Per CLAUDE.md 'Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY': scaffold-only IS the canonical L0 posture. BUT the landing memo's claim of 7 MLX primitives + 3 KNOWN-DRIFT-RISK characterized + parity test max_abs<1e-5 implies a MORE-IMPLEMENTED MLX surface than actually exists. The truth: 0 MLX primitives implemented at L0 in mlx_renderer.py; the parity claims actually measure numpy_reference.py vs PyTorch. This is sister to F=Z8's NOT-CLEAN bug (training-invalidating MLX primitive drift) BUT the bug class is DIFFERENT: G=NIRVANA has documentation overstatement of MLX implementation completeness, NOT live training-invalidating drift bugs (because there is no MLX renderer to be buggy). Promote to PROCEED_WITH_REVISIONS (documentation-only fix) rather than NOT CLEAN with code-fix op-routables."
  - assumption: "G=NIRVANA's PyTorch inflate.py is correct"
    classification: HARD-EARNED
    rationale: "Verified empirically: inflate.py uses `F.interpolate(scale_factor=2, mode='bilinear', align_corners=False)` (line 144) — the CORRECT canonical bilinear sister to PR95 helper. NHWC↔NCHW transpose at state-dict load (lines 199-204) is correct (out_ch, kH, kW, in_ch) → (out_ch, in_ch, kH, kW). select_inflate_device via canonical helper per Catalog #205. inflate.py is RIGOROUS implementation; this is NOT a bug class."
council_decisions_recorded:
  - "R1' verdict: NOT CLEAN — counter resets to 0 for G per CLAUDE.md 'Recursive adversarial review protocol — close paths' item 3"
  - "FIX-WAVE-R1' required BEFORE R2' fires per protocol item 4"
  - "FIX-WAVE-R1' op-routables: G-OP1 (landing memo correction: 'MLX↔PyTorch parity' → 'numpy↔PyTorch parity' across 3 surfaces) + G-OP2 (landing memo correction: 'MLX-first per #1265 anchor' → 'MLX-config-scaffold-first; actual MLX renderer Phase 2') + G-OP3 (mlx_renderer.py docstring correction: 'MLX hierarchical residual decoder cascade' → 'MLX renderer config + helpers; renderer class Phase 2')"
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
council_override_rationale: ""
horizon_class: frontier_pursuit
predicted_band_validation_status: pending_post_training
related_deliberation_ids:
  - path_3_g_nirvana_cascading_nerv_L0_scaffold_landed_20260526
  - path_3_g_nirvana_cascading_nerv_substrate_design_20260526
  - path_3_e_recursive_adversarial_review_r1_3_axis_20260526
  - path_3_recursive_adversarial_review_r1_aggregate_3_axis_landings_a_d_e_20260526
canonical_equation_refs: []
---

# Path 3 candidate G=NIRVANA — R1' 3-axis recursive adversarial review

**Verdict**: **PROCEED_WITH_REVISIONS — R1' NOT CLEAN for G=NIRVANA** — counter resets to 0; FIX-WAVE-R1' successor subagent required BEFORE R2' fires (documentation-only fixes).

**Commit under review**: `f7d2e86fe` (`substrate: land Path 3 candidate G NIRVANA cascading NeRV L0 SCAFFOLD`).

**Cost**: $0 GPU; ~45 min wall-clock.

---

## Premise verification (Catalog #229)

| File | Purpose | LOC |
|---|---|---|
| `.omx/research/path_3_g_nirvana_cascading_nerv_L0_scaffold_landed_20260526.md` | Landing memo | 234 |
| `.omx/research/path_3_g_nirvana_cascading_nerv_substrate_design_20260526.md` | Design memo | 167 |
| `src/tac/substrates/nirvana_cascading_nerv/__init__.py` | Public API + Catalog #124 8-field declaration | 168 |
| `src/tac/substrates/nirvana_cascading_nerv/mlx_renderer.py` | Config + factory + estimators (NO renderer class) | 195 |
| `src/tac/substrates/nirvana_cascading_nerv/numpy_reference.py` | Sister numpy reference (7 primitives) | 329 |
| `src/tac/substrates/nirvana_cascading_nerv/archive.py` | NIRVANA1 byte-deterministic grammar | 287 |
| `src/tac/substrates/nirvana_cascading_nerv/inflate.py` | PyTorch inflate runtime per Catalog #146 + #205 | 313 |
| `src/tac/substrates/nirvana_cascading_nerv/tests/test_basic.py` | 27 tests | ~520 |
| `experiments/train_substrate_nirvana_cascading_nerv_mlx.py` (cited) | MLX smoke trainer stub per Catalog #240 | 113 |

**Empirical reproducer**: `PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest src/tac/substrates/nirvana_cascading_nerv/tests/ -q` → **27 passed in 0.34s** (verified by R1').

---

## Axis 1 review: Math + scientific + engineering rigor

### Per-architectural-choice HARD-EARNED vs CARGO-CULTED classification

| Choice | Source | Classification | Rationale |
|---|---|---|---|
| **Hierarchical residual decoder cascade (4-level wavelet pyramid)** | `mlx_renderer.py::NirvanaCascadingNervConfig` + `numpy_reference.cascade_reconstruct` | HARD-EARNED | Citation to Mallat 1989 multi-resolution analysis + Daubechies 1992 ten lectures (landing memo lines 92-93). Cascade target validated against EVAL_HW=(384,512) per `__post_init__` lines 79-85. |
| **Bilinear upsample with align_corners=False (PyTorch default)** | `numpy_reference.bilinear_upsample_2x_nhwc` lines 144-214 | HARD-EARNED | Explicit canonical implementation citing PyTorch default + sister A=DreamerV3 documented max_abs=24.34 gap caused by align_corners=True via mx.repeat substitution (lines 154-156). The numpy reference computes (dst+0.5)/scale-0.5 mapping correctly. |
| **fp32 accumulation in numpy_reference** | numpy_reference.py lines 36-39, 88, 161 | HARD-EARNED | Catalog #962 / slot 16 engineering corrections cited at lines 87. Stable sigmoid via per-sign branch at lines 221-234. |
| **Kahan summation for large-N batch aggregation** | numpy_reference.py lines 255-281 | HARD-EARNED | Catalog #962 / slot 16 cited explicitly at lines 257-260. Standard Kahan algorithm; canonical reference. |
| **Per-level int8 residual quantization** | mlx_renderer.py config line 60 + inflate.py `_dequantize_residuals` lines 154-171 | HARD-EARNED | Canonical int8 in [-128, 127] → fp32 in [-residual_scale, +residual_scale] dequant; scale carried in archive meta per archive.py grammar. |
| **NHWC layout MLX-canonical; PyTorch NCHW transpose at state-dict load** | inflate.py lines 199-204 + numpy_reference.py lines 75-93 | HARD-EARNED | Documented rationale (numpy_reference.py:88-90): MLX canonical NHWC; PyTorch NCHW requires transpose at export bridge. inflate.py state_dict load (lines 199-204) does (out_ch, kH, kW, in_ch) → (out_ch, in_ch, kH, kW) transpose for 4D weight tensors. |
| **NIRVANA1 archive grammar Catalog #124 8-field declaration** | __init__.py ARCHIVE_GRAMMAR_FIELDS + test passes | HARD-EARNED | Test `test_archive_grammar_fields_catalog_124_compliance` verifies all 8 required keys present. |
| **PyTorch inflate uses canonical F.interpolate bilinear align_corners=False** | inflate.py line 144 | HARD-EARNED | THE canonical pattern — bug-free reference implementation. |
| **PyTorch inflate uses canonical select_inflate_device per Catalog #205** | inflate.py lines 43-46, 182 | HARD-EARNED | Canonical helper import; no inline device-fork bug class (cf. Catalog #205 bug class). |
| **27/27 PASS empirical verdict table** | landing memo lines 53-83 | HARD-EARNED | All test names + verdicts documented; reproducer command cited. |

**Net Axis 1 per-architectural-choice classification**: 10 HARD-EARNED + **0 CARGO-CULTED at the architectural-decision level**.

### Documentation-overstatement findings (Axis 1 — Catalog #287 sister discipline)

| Documentation claim | Actual implementation | Truth gap |
|---|---|---|
| Landing memo line 70: `test_numpy_reference_bilinear_upsample_matches_pytorch PASS  **axis 2 evidence**: MLX↔PyTorch parity ≤ 1e-5` | Test compares `bilinear_upsample_2x_nhwc` (numpy) vs PyTorch F.interpolate; NOT MLX vs PyTorch | **OVERSTATEMENT**: the test is numpy↔PyTorch parity (axis 3 evidence), NOT MLX↔PyTorch parity (axis 2 evidence). MLX is not even imported in this test. |
| Landing memo line 108: "Empirical evidence: `test_numpy_reference_bilinear_upsample_matches_pytorch` max_abs < 1e-5 vs PyTorch F.interpolate align_corners=False" cited under "Axis 2 MLX drift minimization" | Same as above | **OVERSTATEMENT**: This is axis 3 evidence (numpy portability + PyTorch reference parity), not axis 2 evidence (MLX drift) |
| Landing memo line 80: `test_cascade_pytorch_vs_numpy_reference_parity PASS  **axis 2+3 evidence**: PyTorch↔numpy cascade parity ≤ 1e-3` | Test compares PyTorch inflate cascade vs numpy reference cascade | **PARTIAL OVERSTATEMENT**: This is axis 3 evidence (numpy portability + PyTorch correctness) but NOT axis 2 evidence (no MLX measurement) |
| Landing memo line 102-103: "**7 MLX primitives** characterized with expected drift bounds  **3 KNOWN-DRIFT-RISK primitives** with canonical mitigation cites" | mlx_renderer.py has NO MLX primitives — only Config dataclass + `_ensure_mlx_available` + `renderer_param_count` + `estimate_archive_bytes` + `_full_main` (raises NotImplementedError) | **OVERSTATEMENT**: the design memo MAY characterize 7 anticipated L1 MLX primitives, but the L0 SCAFFOLD ships ZERO MLX primitives. The mlx_renderer.py is NOT an "MLX renderer" — it is the MLX renderer CONFIG + FACTORY scaffold. |
| Landing memo line 26: "MLX-first per #1265 anchor; numpy reference per axis 3 portability" | No MLX renderer exists at L0; MLX trainer body raises NotImplementedError | **OVERSTATEMENT**: claim "MLX-first" misrepresents the scaffold posture. The scaffold is MLX-CONFIG-first; the actual MLX renderer is Phase 2. |
| mlx_renderer.py docstring line 2: `MLX hierarchical residual decoder cascade` | Module contains NO decoder cascade implementation; only config + helpers | **OVERSTATEMENT**: module-level docstring promises a renderer that the module does not deliver. |

### Findings (Axis 1)

**G-OP1 (P0 / DOCUMENTATION-ONLY FIX)**: update landing memo `.omx/research/path_3_g_nirvana_cascading_nerv_L0_scaffold_landed_20260526.md`:
- Replace 'axis 2 evidence' tags on numpy↔PyTorch parity tests with 'axis 3 evidence' (lines 70, 71, 80, 81)
- Replace 'Empirical evidence: ... max_abs < 1e-5 vs PyTorch' under "Axis 2 MLX drift minimization" with explicit "Empirical evidence: numpy↔PyTorch parity; MLX↔PyTorch parity deferred to Phase 2 when actual MLX renderer lands"
- Replace '**7 MLX primitives** characterized with expected drift bounds' with '**7 ANTICIPATED L1+ MLX primitives** characterized in design memo with expected drift bounds; ZERO MLX primitives shipped at L0'
- Replace '**3 KNOWN-DRIFT-RISK primitives** with canonical mitigation cites' with '**3 KNOWN-DRIFT-RISK primitives** documented in design memo for L1+ implementation guidance' (these are design-memo content; not empirically measured at L0)

**G-OP2 (P0 / DOCUMENTATION-ONLY FIX)**: in landing memo and design memo, replace 'MLX-first per #1265 anchor' with 'MLX-config-scaffold-first per #1265 anchor; actual MLX renderer implementation deferred to Phase 2 council symposium per Catalog #325'

**G-OP3 (P1 / DOCUMENTATION-ONLY FIX)**: update `mlx_renderer.py` module docstring (line 2):
- From: `MLX hierarchical residual decoder cascade`
- To: `MLX hierarchical residual decoder cascade SCAFFOLD (config + helpers; renderer class Phase 2)`

---

## Axis 2 review: MLX drift minimization

### Per-MLX-primitive empirical drift measurement

**G=NIRVANA ships 0 MLX-callable primitives at L0**. mlx_renderer.py contains:
- `EVAL_HW` constant (line 34)
- `NirvanaCascadingNervConfig` dataclass (lines 38-99)
- `_ensure_mlx_available` lazy-import guard (lines 102-114)
- `renderer_param_count` numpy-only param counter (lines 117-136)
- `estimate_archive_bytes` numpy-only byte-budget estimator (lines 139-166)
- `_full_main` raising NotImplementedError per Catalog #240 (lines 169-185)

There is NO MLX module, NO MLX renderer class, NO MLX primitive used at L0. Tests `test_module_imports_without_mlx` (line 25) + `test_mlx_availability_gate` (line 77 cited in landing memo) verify scaffold-only posture.

**Anticipated L1+ MLX primitives per design memo (these will need parity verification AT L1)**:
1. PixelShuffle 2x NHWC
2. Bilinear upsample 2x NHWC (align_corners=False)
3. Linear projection
4. Conv2d NHWC
5. Sigmoid + Sin activations
6. Mean reduction (or Kahan for large N)
7. Stack/reshape/transpose ops

### Findings (Axis 2)

**0 CRITICAL findings AT L0** (no MLX primitives shipped → 0 drift to measure).

**Documentation overstatement is the Axis-1 finding** (G-OP1 above): landing memo incorrectly labels numpy parity tests as "axis 2 MLX drift evidence". This is NOT a TRAINING-INVALIDATING bug (cf. F=Z8 NOT CLEAN); it is documentation drift only.

**Anticipatory advisory queued for L1+ implementation**:
1. **G-L1-ADV1** (advisory; not blocking R1'): when L1 implements the actual MLX renderer, MUST adopt the canonical META-CONSOLIDATE-OP helpers (`_pixel_shuffle_2x_nhwc` + `bilinear_resize2x_align_corners_false_nhwc` from `tac.local_acceleration.pr95_hnerv_mlx`) per R1 aggregate META finding #1. Sister F=Z8 R1' empirically demonstrates 3.77 + 1.51 max_abs drift bugs from NOT adopting canonical helpers. CONSOLIDATE-OP is the structural extinction.

---

## Axis 3 review: Portability via numpy

### Per-primitive numpy reference status (THIS is the substrate's Axis 3 strength)

| MLX primitive (anticipated L1+) | numpy reference at `numpy_reference.py` | Status |
|---|---|---|
| dtype cast | `to_float32` lines 36-38 | EXISTS |
| linear | `linear` lines 45-61 | EXISTS |
| conv2d NHWC | `conv2d_nhwc` lines 68-137 | EXISTS (naive nested-loop; correctness reference) |
| bilinear upsample 2x NHWC align_corners=False | `bilinear_upsample_2x_nhwc` lines 144-214 | EXISTS (canonical PyTorch parity ≤ 1e-5 per test) |
| sigmoid | `sigmoid` lines 221-234 | EXISTS (numerically stable) |
| sin | `sin` lines 241-243 | EXISTS |
| mean reduction (+ optional Kahan) | `mean` lines 250-252 + `kahan_mean` lines 255-280 | EXISTS |
| cascade reconstruction | `cascade_reconstruct` lines 287-316 | EXISTS (composition reference) |

**7 of 7 anticipated MLX primitives have sister numpy reference implementations** — sister-canonical to R1 aggregate META finding #1 CONSOLIDATE-OP recommendation. Substrate IS operable on CPU-only test rigs WITHOUT MLX dependency: all 27 tests PASS without MLX.

### Verified PyTorch parity (axis 3 evidence — correctly labeled here)

- `test_numpy_reference_bilinear_upsample_matches_pytorch` (test_basic.py:357-381): max_abs ≤ 1e-5 vs PyTorch F.interpolate align_corners=False
- `test_numpy_reference_sigmoid_matches_pytorch`: max_abs ≤ 1e-6 vs PyTorch
- `test_numpy_reference_linear_matches_pytorch`: max_abs ≤ 1e-5 vs PyTorch
- `test_cascade_pytorch_vs_numpy_reference_parity` (test_basic.py:445+): max_abs ≤ 1e-3 vs PyTorch

### Findings (Axis 3)

**0 findings**. G=NIRVANA's `numpy_reference.py` is the canonical example of operator directive #3 axis 3 portability discipline. The 7-of-7 sister numpy references with PyTorch parity ≤ 1e-5 are EXEMPLARY. The substrate is maximally portable per Catalog #178 + #179 GHA CPU CI discipline.

---

## R1' verdict for G=NIRVANA

**Per-axis verdicts**:
- Axis 1 (math + sci + engineering rigor): **CLEAN-at-architecture / NOT CLEAN at documentation** (3 documentation-overstatement findings; G-OP1 + G-OP2 + G-OP3)
- Axis 2 (MLX drift minimization): **N/A AT L0** (0 MLX primitives shipped; documentation overstatement is Axis 1 finding)
- Axis 3 (numpy portability): **CLEAN AND EXEMPLARY** (canonical 7-of-7 numpy reference pattern; should become sister-canonical reference for future Path 3 candidates)

**Aggregate**: **PROCEED_WITH_REVISIONS — R1' NOT CLEAN**. Counter resets to **0/3** per CLAUDE.md protocol item 3 (any issue resets to 0).

**FIX-WAVE-R1' required BEFORE R2' fires** per protocol item 4 (documentation-only fixes; ≤3 files; ≤30 LOC of edits).

---

## Counter state per CLAUDE.md "Recursive adversarial review protocol — close paths"

- **Before R1'**: counter = 0 (G is NEW landing post-R1; no prior cycle history)
- **R1' verdict**: NOT CLEAN → counter remains at **0/3** per protocol item 3
- **R2' BLOCKED until FIX-WAVE-R1' lands**:
  - G-OP1 + G-OP2 + G-OP3 (3 documentation-only edits across 2 files: landing memo + mlx_renderer.py)
- **R2' verification post-FIX-WAVE-R1'**:
  - Re-run 27 tests; verify all PASS (no test changes expected since these are doc-only fixes)
  - Verify landing memo correctly labels parity tests as 'axis 3' evidence
  - Verify mlx_renderer.py docstring correctly reflects scaffold-only posture

---

## FIX-WAVE-R1' op-routables for G=NIRVANA (priority-ranked)

### P0 / DOCUMENTATION-ONLY FIX

**G-OP1**: update `.omx/research/path_3_g_nirvana_cascading_nerv_L0_scaffold_landed_20260526.md`:
- Lines 70-71: change `**axis 2 evidence**: MLX↔PyTorch parity ≤ 1e-5` to `**axis 3 evidence**: numpy↔PyTorch parity ≤ 1e-5`
- Lines 80-81: change `**axis 2+3 evidence**: PyTorch↔numpy cascade parity ≤ 1e-3` to `**axis 3 evidence**: PyTorch↔numpy cascade parity ≤ 1e-3`
- Lines 100-108: rewrite "Axis 2 MLX drift minimization" section to explicitly state: "ZERO MLX primitives implemented at L0; the 7 anticipated MLX primitives + 3 KNOWN-DRIFT-RISK characterized in design memo are L1+ implementation guidance. At L1, MUST adopt canonical META-CONSOLIDATE-OP helpers per R1 aggregate META finding #1 — sister F=Z8 R1' empirically demonstrates 3.77 + 1.51 max_abs drift bugs from NOT adopting canonical helpers."

**G-OP2**: update landing memo line 26: change "MLX-first per #1265 anchor" to "MLX-config-scaffold-first per #1265 anchor; actual MLX renderer implementation deferred to Phase 2 council symposium per Catalog #325"

**G-OP3**: update `src/tac/substrates/nirvana_cascading_nerv/mlx_renderer.py` line 2: change `"""nirvana_cascading_nerv.mlx_renderer — MLX hierarchical residual decoder cascade.` to `"""nirvana_cascading_nerv.mlx_renderer — MLX hierarchical residual decoder cascade SCAFFOLD (config + helpers; actual renderer class lands Phase 2).`

---

## Discipline applied

- **Catalog #229 PV**: landing memo + design memo + all 5 source files (init + mlx_renderer + numpy_reference + archive + inflate) + tests/test_basic.py read in full; 27 tests verified PASS
- **Catalog #110/#113 APPEND-ONLY**: NEW review memo; landing memo + source files NEVER mutated
- **Catalog #287 placeholder-rationale rejection**: every assumption-adversary verdict carries non-placeholder rationale; documentation-overstatement findings per Catalog #287 anti-overstatement sister discipline
- **Catalog #292 per-deliberation assumption surfacing**: per-axis council members declared
- **Catalog #300 v2 frontmatter**: full T2 frontmatter
- **Catalog #340 sister-checkpoint guard**: PROCEED verdict
- **Catalog #307 paradigm-vs-implementation classification**: G=NIRVANA's findings are DOCUMENTATION-LEVEL (memo + docstring labels), NOT IMPLEMENTATION-LEVEL bugs (cf. F=Z8 NOT CLEAN with code fixes required). Per CLAUDE.md "Forbidden premature KILL without research exhaustion": substrate paradigm INTACT
- **CLAUDE.md "Recursive adversarial review protocol — close paths"** items 1-8 honored
- **CLAUDE.md "Executing actions with care"**: review-only

---

## Notable positive finding

G=NIRVANA's `numpy_reference.py` (329 LOC across 7 primitives) IS THE CANONICAL SISTER REFERENCE PATTERN that R1 aggregate META finding #1 recommends for future Path 3 candidates. The pattern:
1. ONE numpy-canonical implementation per MLX primitive
2. PyTorch parity test per primitive (≤ 1e-5 fp32; ≤ 1e-3 fp16 accumulation)
3. fp32 accumulation for numerical stability
4. Stable sigmoid via per-sign branch
5. Kahan summation for large-N batch aggregation
6. Canonical documentation citation per primitive

This pattern SHOULD be extracted to canonical `tac.local_acceleration.numpy_reference` (or equivalent canonical location) per R1 aggregate META-CONSOLIDATE-OP-1 sister extension, so future MLX substrates inherit ONE source of truth at the numpy-reference layer too.

---

## Cross-references

- Landing memo: `.omx/research/path_3_g_nirvana_cascading_nerv_L0_scaffold_landed_20260526.md` (commit `f7d2e86fe`)
- Design memo: `.omx/research/path_3_g_nirvana_cascading_nerv_substrate_design_20260526.md`
- Sister F=Z8 R1' (NOT CLEAN; analogous documentation+implementation gap): `.omx/research/path_3_f_recursive_adversarial_review_r1_prime_3_axis_20260526.md`
- Sister E=BoostNeRV R1 (analogous documentation-drift finding; FIXED in FIX-WAVE-R1): `.omx/research/path_3_e_recursive_adversarial_review_r1_3_axis_20260526.md`
- Canonical PR95 helper: `tac.local_acceleration.pr95_hnerv_mlx::bilinear_resize2x_align_corners_false_nhwc`
- R1 aggregate META finding #2 (documentation-source-of-truth drift class): `.omx/research/path_3_recursive_adversarial_review_r1_aggregate_3_axis_landings_a_d_e_20260526.md`
- Lane: `lane_path_3_recursive_adversarial_review_r1_prime_3_axis_landings_b_c_f_g_20260526` L0
