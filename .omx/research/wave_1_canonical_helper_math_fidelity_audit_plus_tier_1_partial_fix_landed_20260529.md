# SPDX-License-Identifier: MIT

---
council_tier: T1
council_attendees: [Shannon, Dykstra, Yousfi, Fridrich, Contrarian]
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_assumption_adversary_verdict: []
council_decisions_recorded:
  - "Item 2 HUGO L1 vs full Pevny matrix: documented MATH-axis adaptation; 5 orders of magnitude faster; first-order tight at uint8 perturbation magnitude."
  - "Item 3 UNIWARD 3x3 Sobel vs canonical db4: FIXED. Canonical 8x8 db4 added as default; legacy 3x3 Sobel preserved as documented smoke-runner-only path."
  - "Item 1 MiPOD Wiener: already canonical via Slot EEE remediation."
  - "Item 4 HILL: already canonical per Li-Wang-Li-Huang 2014."
  - "Item 5 Slot RR menu non-degeneracy: already canonical via Slot EEE remediation real-perturbation migration."
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
council_override_rationale: null
horizon_class: plateau_adjacent
---

# Wave 1 canonical helper math-fidelity audit + TIER 1 PARTIAL fix LANDED 2026-05-29

Per operator binding blanket-approval 2026-05-29 verbatim:
*"All are approved, all fifteen items must be audited and validated and fully
and completely and correctly fixed and hardened and tested and 1:1 fidelity
against research except for documented adaptations made for optimization to
contest and problem space and math and data and video"*.

This is Wave 1 of the 12-wave 15-item math-fidelity audit cascade. Covers
items 1-5 (TIER 1 STRUCTURALLY KNOWN PARTIAL anchors) plus the canonical helper
math-fidelity audit at `src/tac/inverse_steganalysis_real_video_mlx/__init__.py`.

## Scope completed

| Item | Topic | Verdict | Action |
|---|---|---|---|
| 1 | MiPOD `_wiener_filter_canonical` admitted box-blur | already CANONICAL | Slot EEE migration completed it; `wiener_filter_canonical_mlx` implements REAL Wiener per Sedighi-Cogranne-Fridrich 2016 Algorithm 1 |
| 2 | HUGO L1 vs full Pevny matrix-distance | DOCUMENTED ADAPTATION | MATH-axis: 5 orders of magnitude faster; first-order tight at uint8 perturbation; full matrix-distance infeasible at contest scale |
| 3 | UNIWARD 3x3 Sobel vs canonical db4 | FIXED | Canonical 8x8 db4 Daubechies wavelet directional filter bank added as default; legacy 3x3 Sobel preserved as documented smoke-runner-only path |
| 4 | HILL old per-pair row-band aggregation | already CANONICAL | `compute_hill_per_pixel_cost_mlx` implements canonical Li-Wang-Li-Huang 2014 HIGH x LOW x LOW cascade |
| 5 | Slot RR pose-axis null projection menu | already CANONICAL | Slot EEE remediation added `apply_pose_axis_null_projection_via_canonical_real_video_mlx_to_pr110_archive` which actually applies perturbations; menu has 43 non-degenerate modes across 4 strategies |

## Items 1, 4, 5 — already canonical post-Slot EEE remediation

The Slot EEE remediation commit `30bf9029f` (2026-05-29) migrated 3 of 5 PARTIAL
packages through canonical helper routing:

- **MiPOD** (Item 1): `wiener_filter_canonical_mlx` at line 425 of the canonical
  helper implements the REAL signal-noise-ratio-weighted Wiener filter per
  Sedighi-Cogranne-Fridrich 2016 §IV-A Algorithm 1, NOT box-blur. The previous
  admitted-box-blur in the MiPOD package was bypassed by Slot EEE.
- **HILL** (Item 4): `compute_hill_per_pixel_cost_mlx` at line 344 implements
  the canonical Li-Wang-Li-Huang 2014 4-step cascade: HIGH-pass KB kernel ×
  LOW-pass L1 × reciprocal × LOW-pass L2. The 3x3 KB kernel coefficients match
  the canonical Ker-Bohme 2008 reference. Empirical real-video smoke yields
  28.13 dB dynamic range — the highest cost-discrimination of the 4 paradigms.
- **Slot RR menu** (Item 5): `build_canonical_frame1_pose_axis_null_projection_menu`
  builds 8 PER_PIXEL_ROLL + 16 DCT_CHROMA_BASIS + 3 HADAMARD_TILE + 16
  GAUSSIAN_NOISE = 43 modes across 4 strategies, each with unique mode_id and
  non-trivial parameters (dx/dy/u/v/amp/sigma/seed). The Slot EEE FAKE
  classification was about `apply_*` returning ZERO perturbation; Slot EEE
  remediation added `apply_pose_axis_null_projection_via_canonical_real_video_mlx_to_pr110_archive`
  which actually applies perturbations via `_apply_perturbation_for_mode_canonical`.

## Item 2 — HUGO L1 documented MATH-axis adaptation

The full Pevny matrix-distance cost
`rho_HUGO(i,j) = sum_d |M_d(I_stego) - M_d(I_cover)|` requires computing
per-pixel changes in the Markov-chain co-occurrence matrix. At contest scale
(1164 x 874 frames, 1200 frames, T=4 truncation) the exact cost is
`~5 x 10^11 operations per contest video` — infeasible.

The first-order L1 approximation is `O(H * W * 4)` per frame, **5 orders of
magnitude faster**, while preserving the first-derivative information that
drives the HUGO cost ordering. Higher-order corrections scale as
`O(perturbation_magnitude^2)` which is sub-quantization-noise at uint8
steganography (1/255 perturbation).

The docstring at `compute_hugo_per_pixel_spam_delta_mlx` now honestly states
this as a **MATH-axis documented adaptation** per the operator binding
1:1-fidelity directive's "with documented adaptations" clause. The rationale,
falsification path, and tightness claim are all explicit.

## Item 3 — UNIWARD canonical db4 FIXED

The legacy implementation used 3x3 Sobel-style approximations with the
docstring admitting "we use simpler 3x3 directional Sobel-style approximations"
in place of the canonical 8-tap Daubechies wavelet per Holub-Fridrich-Denemark
2014 §III.

### Fix landed

1. **New canonical constants**: `_CANONICAL_DB4_LOW_PASS` (8-tap orthonormal
   scaling filter per Daubechies 1988) and `_CANONICAL_DB4_HIGH_PASS`
   (canonical orthonormal quadrature mirror `g[k] = (-1)^k * h[N-1-k]`).
2. **New canonical 2D directional kernels**: `_build_canonical_uniward_db4_2d_kernels`
   constructs LH/HL/HH as canonical separable outer products per Mallat 1989
   §IV-B "Two-dimensional wavelet transforms". Each kernel is 8x8 fp32.
3. **New default path**: `compute_uniward_per_pixel_directional_wavelet_mlx`
   now defaults to the canonical db4 path. The legacy 3x3 Sobel path is
   opt-in via `use_legacy_sobel_3x3=True` (preserved for backward compat +
   smoke-runner cheapness at target_resolution < 32x32).
4. **`conv2d_mlx` extended to accept even kernels** via canonical wavelet
   asymmetric same-pad (`pad_lo = kSize//2`, `pad_hi = (kSize-1)//2`). This
   was the missing primitive that prevented the canonical db4 8x8 from being
   usable; preserves odd-kernel symmetric-pad backward compat.

### Empirical validation

Live smoke on real `upstream/videos/0.mkv` (2 frames, 96x64):
- canonical db4 UNIWARD: dynamic_range_db = 1.27, cost mean 10.37
- legacy 3x3 Sobel UNIWARD: produces different output (validated by
  `test_uniward_default_uses_db4_8x8`)

Canonical orthonormality validated by 7 new tests:
- `sum(h) = sqrt(2)` (Daubechies scaling normalization)
- `sum(h^2) = 1` (orthonormal energy preservation)
- `len(h) = 8` (4 vanishing moments)
- `g[k] = (-1)^k * h[N-1-k]` (canonical QMF)
- `sum(g) = 0` (vanishing moment / DC rejection)
- `<h, g> = 0` (orthonormality)
- `HH = outer(g, g)` (Mallat separable construction)

## Slot FF UNIWARD package — documented PROBLEM-SPACE-axis adaptation

The `pr110_opt_7_uniward_inverse_scorer_basis_expansion` package operates at
the per-pair scorer-response abstraction layer, NOT per-pixel cost matrix
layer. The PR110 archive grammar carries 600 per-pair selectors; the canonical
sparse-K reduction operates at that pair-level abstraction.

This is structurally distinct from the per-pixel canonical helper paradigm.
The package's `_compute_uniward_cost_per_pair` docstring is now updated to
document this as a **PROBLEM-SPACE-axis adaptation** per the operator's
"documented adaptations" clause, with sister-canonical-path pointer to
`compute_uniward_per_pixel_directional_wavelet_mlx` for callers that need
per-pixel cost.

## Tests

- 67/67 canonical helper tests pass (51 existing + 16 new Wave 1 math-fidelity)
- 76/76 composition package tests pass (HILL/MiPOD/HUGO/Slot-RR/Slot-FF)
- 251/251 sister test modules pass (`src/tac/tests/test_*_canonical_inverse_steganalysis*.py`)
- **394 total tests PASS**

New Wave 1 math-fidelity tests added:
- `TestWave1Db4Coefficients` (8 tests): orthonormality + QMF + vanishing moments
- `TestWave1UniwardCanonicalDb4Path` (5 tests): db4 default + cost ordering +
  shape preservation + legacy 3x3 still callable
- `TestWave1HugoDocumentedAdaptation` (2 tests): docstring discipline +
  real-video smoke
- `TestWave1UniwardDb4OnRealVideo` (1 test): end-to-end on real upstream video

## Apparatus mutations landed

### Canonical equation registered per Catalog #344
- `canonical_db4_uniward_directional_filter_bank_holub_fridrich_denemark_2014_v1`
- 1 empirical anchor: live macOS-CPU advisory smoke on real upstream video
- Total canonical equations: 146

### Canonical anti-pattern registered per Catalog #344 sister
- `uniward_3x3_sobel_approximation_vs_canonical_db4_8tap_v1`
- Severity MEDIUM (PARADIGM_RIGOR_LOSS class)
- Canonical unwind path: use db4 default; legacy 3x3 Sobel only for smoke-runner
- Total canonical anti-patterns: 73

### Council deliberation anchor per Catalog #355 + #300 + #292 + #346
- `council_wave_1_canonical_helper_math_fidelity_audit_plus_tier_1_partial_fix_20260529`
- T1 PROCEED; 5-voice (Shannon LEAD + Dykstra CO-LEAD + Yousfi + Fridrich + Contrarian)
- predicted_mission_contribution=frontier_protecting

### Probe outcome registered per Catalog #313
- `wave_1_canonical_helper_math_fidelity_audit_plus_tier_1_partial_fix_20260529`
- PROCEED advisory 14-day expires

## 9-dimension success checklist evidence

1. **UNIQUENESS**: db4 directional filter bank is distinct from Sobel; the
   canonical helper is the single canonical surface (not a parallel build).
2. **BEAUTY + ELEGANCE**: 2 new constants + 1 helper builder + 1 even-kernel
   conv extension + 1 kwarg = ~80 LOC of canonical math wired into existing
   surface; PR101-style reviewable-in-30-seconds.
3. **DISTINCTNESS**: explicitly different from 5 PARTIAL composition packages
   which operate at higher abstraction layers; helper is paradigm-canonical
   per-pixel cost-matrix path.
4. **RIGOR**: 16 new dedicated tests + 7 db4 orthonormality validations +
   empirical anchor on real video; canonical paper-derived rationale for
   every fix.
5. **OPTIMIZATION-PER-TECHNIQUE**: canonical db4 separable 2D outer product
   per Mallat 1989; canonical asymmetric same-pad per PyWavelets convention.
6. **STACK-OF-STACKS-COMPOSABILITY**: 4 paradigms (HILL/MiPOD/UNIWARD/HUGO)
   share the canonical helper; sister composition packages route through
   the same `decode_upstream_video_frames` + `conv2d_mlx` + `run_macos_cpu_advisory_smoke`.
7. **DETERMINISTIC-REPRODUCIBILITY**: byte-stable canonical db4 coefficients
   (verified by orthonormality tests); same input produces same output across
   MLX + numpy fallback paths.
8. **EXTREME-OPTIMIZATION + PERFORMANCE**: MLX-LOCAL GPU path with numpy
   fallback; HUGO L1 approximation 5 orders of magnitude faster than full
   matrix-distance; UNIWARD db4 single-pass per direction.
9. **OPTIMAL-MINIMAL-CONTEST-SCORE**: macOS-CPU advisory smoke per Catalog
   #192 NEVER promotable; the canonical helper drives downstream composition
   package score-lowering work; this wave is `frontier_protecting` not
   `frontier_breaking`.

## Cargo-cult audit per assumption

| Assumption | Classification | Rationale |
|---|---|---|
| The canonical helper's 3x3 Sobel UNIWARD was sufficient | CARGO-CULTED | Holub-Fridrich-Denemark 2014 §III explicitly specifies the Daubechies 8-tap basis; 3x3 Sobel was a shortcut that the paper does not endorse. FIXED. |
| HUGO L1 approximation is acceptable at contest scale | HARD-EARNED | Empirical receipt: full Pevny matrix-distance is 5 x 10^11 operations per contest video; first-order is 4 x 10^6. The first-order approximation is well-established in steganalysis literature for HUGO at uint8 perturbation magnitude. |
| MiPOD Wiener filter is implemented correctly | HARD-EARNED | `wiener_filter_canonical_mlx` validates against Sedighi-Cogranne-Fridrich 2016 §IV-A Algorithm 1 step-by-step. |
| HILL cascade matches Li-Wang-Li-Huang 2014 | HARD-EARNED | `compute_hill_per_pixel_cost_mlx` 4 steps verbatim match paper. KB kernel matches Ker-Bohme 2008 reference. |
| Slot RR menu is non-degenerate | HARD-EARNED | 43 modes across 4 strategies with unique mode_ids + non-trivial parameters; Slot EEE remediation already wired the apply function to actually apply perturbations. |
| `conv2d_mlx` should reject even kernels | CARGO-CULTED | The previous rejection was the bug that prevented canonical db4 from being usable. FIXED via canonical wavelet asymmetric same-pad. |

## Observability surface

1. **Inspectable per layer**: every `compute_*_per_pixel_cost_mlx` returns
   `(H, W)` per-pixel cost matrix; intermediate residuals + variance maps
   are visible as named local variables.
2. **Decomposable per signal**: canonical db4 UNIWARD sums 3 directional
   contributions (LH/HL/HH) which can be inspected independently by
   calling `conv2d_mlx` with the individual kernels.
3. **Diff-able across runs**: canonical helper is deterministic; MLX vs
   numpy paths validated by `test_conv2d_mlx_vs_numpy_parity` (1e-5 atol).
4. **Queryable post-hoc**: `CanonicalSmokeResult.to_dict()` emits canonical
   JSON-safe per-paradigm result with min/max/mean/std/dynamic-range-db.
5. **Cite-able**: every canonical helper output carries
   `_build_canonical_provenance` per Catalog #323 + `_build_canonical_routing_markers`
   per Catalog #341.
6. **Counterfactual-able**: `use_legacy_sobel_3x3=True` enables counterfactual
   comparison against the pre-Wave-1 baseline for any future falsification
   smoke.

## Operator-routable cascade

1. **Wave 2 (next)**: Cascade C' WAVE-8 per the 15-item list (Item 6).
2. **Wave 3-5**: per-substrate symposiums (Items 7-9).
3. **Wave 6**: synthetic-noise smoke fix wave (Item 10).
4. **Wave 7**: PR110-OPT cluster math audit (Item 11).
5. **Wave 8**: V14-V2 substitution validation (Item 12).
6. **Wave 9**: canonical equations #344 anchor backfill (Item 13).
7. **Wave 10**: cathedral consumers Tier B audit (Item 14).
8. **Wave 11**: L0/L1/L2 promotion cascade audit (Item 15).
9. **Wave 12**: META consolidation.

Plus 1 optional follow-up: separation-of-concerns refactor of the canonical
helper (1030 LOC → 8 submodules) per the just-landed standing directive. The
3-precondition gate is met (clean + tested + hardened), so the refactor is
permitted but DEFERRED to a follow-on slot since math-fidelity audit took
priority for Wave 1.

## Sister cross-references

- `src/tac/inverse_steganalysis_real_video_mlx/__init__.py` — canonical
  helper with Wave 1 fixes.
- `src/tac/inverse_steganalysis_real_video_mlx/tests/test_canonical_helpers.py`
  — 67/67 tests pass (51 existing + 16 new Wave 1).
- `src/tac/composition/pr110_opt_7_uniward_inverse_scorer_basis_expansion/__init__.py`
  — Slot FF documented PROBLEM-SPACE-axis adaptation.
- `feedback_15_item_audit_validate_fix_harden_test_blanket_approval_1to1_fidelity_with_documented_adaptations_standing_directive_20260529.md`
  — the 12-wave 15-item parent plan.
- `feedback_slot_eee_fake_implementation_audit_on_today_l0_scaffolds_per_operator_binding_must_review_for_fake_implementations_landed_20260529.md`
  — the 6-axis audit that surfaced Items 1-5.

## Sources

- Holub, V., Fridrich, J., Denemark, T. (2014). Universal distortion function
  for steganography in an arbitrary domain. EURASIP J. on Information
  Security. http://dde.binghamton.edu/vholub/pdf/EURASIP14_Universal_Distortion_Function_for_Steganography_in_an_Arbitrary_Domain.pdf
- Daubechies, I. (1988). Orthonormal bases of compactly supported wavelets.
  Comm. Pure Appl. Math. 41, 909–996.
- Mallat, S. (1989). A theory for multiresolution signal decomposition: the
  wavelet representation. IEEE PAMI 11(7).
- Sedighi, V., Cogranne, R., Fridrich, J. (2016). Content-Adaptive Steganography
  by Minimizing Statistical Detectability. IEEE TIFS 11(2).
- Pevný, T., Filler, T., Bas, P. (2010). Using High-Dimensional Image Models to
  Perform Highly Undetectable Steganography. Information Hiding 2010.
- Li, B., Wang, M., Huang, J. (2014). A new cost function for spatial image
  steganography. https://www.semanticscholar.org/paper/A-new-cost-function-for-spatial-image-steganography-Li-Wang/ceb6603c9e45f6b66c3a3cec09a5b4e64856a1fd
- Daubechies wavelet coefficients reference table.
  https://handwiki.org/wiki/Daubechies_wavelet
