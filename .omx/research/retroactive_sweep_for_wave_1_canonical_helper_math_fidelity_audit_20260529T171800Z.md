# Retroactive sweep for Wave 1 canonical helper math-fidelity audit 2026-05-29T17:18:00Z

Per Catalog #348 4-field contract: bug-class symptom signature, pre-fix window,
historical KILL/DEFER/FALSIFY search, per-finding RE-EVAL-priority.

## 1. Bug-class symptom signature

The Wave 1 audit closed two distinct bug classes in the canonical helper at
`src/tac/inverse_steganalysis_real_video_mlx/__init__.py`:

**Bug class A (FIXED via canonical db4 + even-kernel support)**:
- Symptom: `compute_uniward_per_pixel_directional_wavelet_mlx` uses 3x3 Sobel-style
  directional kernels that are NOT the paper-canonical Daubechies 8-tap basis
  per Holub-Fridrich-Denemark 2014 §III.
- Root cause: `conv2d_mlx` rejected even-sized kernels, preventing the
  canonical 8x8 db4 from being usable.
- Fix: 8-tap canonical db4 coefficients added; canonical 2D separable
  outer-product 8x8 directional kernels constructed per Mallat 1989; conv2d
  extended to accept even kernels via canonical wavelet asymmetric same-pad;
  db4 default; legacy 3x3 Sobel preserved as smoke-runner-only path.

**Bug class B (DOCUMENTED ADAPTATION per operator binding)**:
- Symptom: `compute_hugo_per_pixel_spam_delta_mlx` uses L1 approximation of the
  full Pevny matrix-distance cost; previous docstring did not honestly
  document this as an adaptation.
- Root cause: full Pevny matrix-distance is `O(H * W * 4 * (2T+1)^2)` per
  frame, infeasible at contest scale.
- Fix: docstring rewritten to explicitly tag this as a MATH-axis documented
  adaptation per the operator binding 1:1-fidelity directive with substantive
  rationale (5 orders of magnitude faster, first-order tight at uint8
  perturbation magnitude, explicit falsification path).

## 2. Pre-fix window

Pre-fix state:
- `tools/check_canonical_task_status_no_dangling_transitions.py --strict` was
  clean before the wave.
- All 51 canonical helper tests + 76 composition package tests passed before
  the wave (the pre-fix tests did not validate the math-fidelity gap; they
  validated only Python correctness + shape preservation).
- Slot EEE (commit `30bf9029f` 2026-05-29) had already migrated 3 of 5 PARTIAL
  packages through canonical helper routing.

The pre-fix window for both bug classes is **the entire history of the
canonical helper from creation through Slot EEE remediation** because:
- Bug class A (db4 vs Sobel) was present from creation; never empirically
  validated.
- Bug class B (HUGO L1 vs full matrix) was implemented from creation; the
  L1 approximation is correct but the previous docstring did not honestly
  document the adaptation.

## 3. Historical KILL/DEFER/FALSIFY search

Searched `.omx/research/` for memos that may have made KILL/DEFER/FALSIFY
verdicts dependent on either bug class:

**Search 1 - "UNIWARD ... FALSIFIED/KILLED/DEFERRED":**
- `feedback_slot_eee_fake_implementation_audit_on_today_l0_scaffolds_20260529.md`
  classifies Slot FF UNIWARD as PARTIAL not KILLED. RE-EVAL priority NONE
  (Wave 1 closed the underlying gap; PARTIAL classification remains valid for
  the per-pair scorer-response abstraction layer which Slot FF operates at).

**Search 2 - "HUGO ... FALSIFIED/KILLED/DEFERRED":**
- `feedback_slot_eee_fake_implementation_audit_on_today_l0_scaffolds_20260529.md`
  classifies Slot CCC HUGO as PARTIAL not KILLED. RE-EVAL priority NONE
  (Wave 1 documented the L1 adaptation; PARTIAL classification was about
  the docstring honesty, not the math).

**Search 3 - "db4 / Daubechies / wavelet ... FALSIFIED":**
- No historical KILL/DEFER/FALSIFY verdict depended on the canonical db4
  path being absent. The 3x3 Sobel approximation was assumed to be the
  canonical UNIWARD implementation in the canonical helper; no downstream
  consumer was making a verdict that depended on it being db4-canonical.

**Search 4 - "wiener_filter / Sedighi / MiPOD ... FALSIFIED":**
- Slot AAA MiPOD admitted box-blur per Slot EEE audit; Slot EEE remediation
  fixed it. No verdicts pre-Wave-1 were dependent on the previous
  admitted-box-blur being canonical. RE-EVAL priority NONE.

**Search 5 - "Sobel ... FALSIFIED / approximation ... FALSIFIED":**
- No matches.

**Total historical KILL/DEFER/FALSIFY count invalidated by Wave 1**: 0.

The Wave 1 fixes are forward-looking improvements + documented-adaptation
hardening. No historical verdict was based on a Wave-1-falsified premise.

## 4. Per-finding RE-EVAL priority assignment

| Historical verdict | RE-EVAL priority | Rationale |
|---|---|---|
| Slot FF UNIWARD PARTIAL | NONE | PARTIAL classification remains valid for per-pair scorer-response abstraction layer; Wave 1 documented the PROBLEM-SPACE-axis adaptation |
| Slot CCC HUGO PARTIAL | NONE | PARTIAL classification was about docstring honesty; Wave 1 fixed the docstring to honestly document the MATH-axis adaptation |
| Slot AAA MiPOD PARTIAL | NONE | Slot EEE remediation already closed the admitted-box-blur gap; Wave 1 verified the canonical Wiener filter implementation |
| Slot YY HILL PARTIAL | NONE | HILL implementation already canonical per Li-Wang-Li-Huang 2014; Wave 1 verified it |
| Slot RR FAKE | NONE | Slot EEE remediation already closed the ZERO-perturbation gap via `apply_pose_axis_null_projection_via_canonical_real_video_mlx_to_pr110_archive`; Wave 1 verified menu non-degeneracy |

## Cross-references

- Catalog #348: this gate's parent META-meta surface.
- Catalog #110/#113: HISTORICAL_PROVENANCE APPEND-ONLY discipline (this sweep
  is itself APPEND-ONLY; preserves prior verdicts intact).
- Catalog #307: paradigm-vs-implementation classification — Wave 1 fixes are
  IMPLEMENTATION-level enhancements (canonical math fidelity); no paradigms
  were falsified.
- Catalog #344: canonical equations + anti-patterns registry — Wave 1 registered
  1 new canonical equation + 1 new anti-pattern.
- `feedback_15_item_audit_validate_fix_harden_test_blanket_approval_1to1_fidelity_with_documented_adaptations_standing_directive_20260529.md`
  - the 12-wave 15-item parent plan.
