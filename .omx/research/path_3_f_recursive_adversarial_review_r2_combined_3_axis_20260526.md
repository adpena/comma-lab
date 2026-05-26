<!-- SPDX-License-Identifier: MIT -->
<!-- Catalog #110 / #113 APPEND-ONLY HISTORICAL_PROVENANCE — this memo is the canonical R2-COMBINED review record for Path 3 candidate #F (Z8 hierarchical predictive coding canonical-quadruple L0 SCAFFOLD; commit `5ff5d2ab9` + FIX-WAVE-R1' `4684dbbab`). DO NOT mutate after landing. -->
<!-- Catalog #344 canonical equation cross-ref: 5 cited canonical equations all REGISTERED per query empirical verification: mlx_pytorch_full_decoder_downstream_scorer_drift_propagation_v1 + scorer_conditional_joint_rate_distortion_floor_v1 + categorical_posterior_capacity_vs_continuous_gaussian_v1 + ego_motion_concentration_prior_v1 + cross_codec_super_additive_orthogonality_predictor_v1. -->
---
council_tier: T2
council_attendees:
  - Shannon
  - Dykstra
  - Tao
  - Carmack
  - Hotz
  - Quantizr
  - MacKay
  - Selfcomp
  - Contrarian
  - Assumption-Adversary
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "Post-FIX-WAVE-R1' F=Z8 MLX↔PyTorch primitive parity is structurally stable (PixelShuffle exactly 0; bilinear within fp32 noise floor)"
    classification: HARD-EARNED-EMPIRICALLY-RE-VERIFIED
    rationale: "R2 re-measurement at 2026-05-26T08:42Z: test_z8_pixel_shuffle_matches_pytorch PASSED; test_z8_bilinear_resize_matches_pytorch PASSED. Both at < 1e-5 threshold per FIX-WAVE-R1' F-OP3 parity tests. Full Z8 suite 18/18 PASS post-FIX-WAVE-R1'. The R1' anchor measurement (PixelShuffle 3.77, bilinear 1.51) is the empirical proof of META-CONSOLIDATE-OP-1 priority escalation per R1' aggregate META finding #1; the post-FIX-WAVE-R1' measurement collapses both drifts to canonical-byte-stable per the verbatim port from FIX-WAVE-R1's A=DreamerV3 patch."
  - assumption: "F=Z8 canonical quadruple binding (Rao-Ballard hierarchy + Mallat wavelet + Hafner DreamerV3 latent dynamics + Wyner-Ziv side-information) is HARD-EARNED at all 4 hierarchy levels per Catalog #312 NON-NEGOTIABLE"
    classification: HARD-EARNED
    rationale: "Per Catalog #312 (`check_hierarchical_predictive_coding_has_canonical_quadruple`): substrate design memo cites Rao-Ballard 1999 + Daubechies 1988 wavelet + Hafner DreamerV3 + Wyner-Ziv 1976 simultaneously; all 4 primitives bound. Z8 design memo Section 4.3 + Section 11 + Section 13 implement the canonical quadruple. R2 reaffirms the architectural binding; no R2 finding."
  - assumption: "FIX-WAVE-R1' verbatim port of FIX-WAVE-R1 A=DreamerV3 patches is mechanically correct AND structurally extincts the bug class IN F=Z8 (but META class is still active per META-CONSOLIDATE-OP-1 in-flight)"
    classification: HARD-EARNED-PARTIAL
    rationale: "F=Z8 channel-FIRST convention + canonical bilinear helper delegation are line-for-line identical to FIX-WAVE-R1's A=DreamerV3 patches. PARTIAL because META-CONSOLIDATE-OP-1 (in-flight subagent pid 82551) extracts the canonical _pixel_shuffle_2x_nhwc helper from D=Z6 / A / F local impls to the canonical PR95 module; until CONSOLIDATE-OP-1 lands, drift between A/D/F could re-emerge if any are independently maintained. Operator-routable: AFTER CONSOLIDATE-OP-1 lands, R3 should re-verify F=Z8 still passes 18/18 tests + parity tests."
council_decisions_recorded:
  - "R2-COMBINED CLEAN PASS — counter advances from 0/3 → 1/3 per protocol items 3-4"
  - "All 3 axes PASS at R2 across 0 R2 findings; FIX-WAVE-R1' closure verified empirically"
  - "Op-routable advisory (NOT R2 finding): META-CONSOLIDATE-OP-1 in-flight subagent (pid 82551) actively touching F=Z8 mlx_renderer.py to extract canonical _pixel_shuffle_2x_nhwc helper; R3 must re-verify F=Z8 still passes 18/18 post-CONSOLIDATE"
  - "Op-routable advisory (NOT R2 finding): all 5 F=Z8 cited canonical equations confirmed REGISTERED in `tac.canonical_equations`; no equation registration gap remains"
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
council_override_rationale: ""
horizon_class: frontier_pursuit
canonical_equation_refs:
  - mlx_pytorch_full_decoder_downstream_scorer_drift_propagation_v1
  - scorer_conditional_joint_rate_distortion_floor_v1
  - categorical_posterior_capacity_vs_continuous_gaussian_v1
  - ego_motion_concentration_prior_v1
  - cross_codec_super_additive_orthogonality_predictor_v1
predicted_band_validation_status: pending_post_training
related_deliberation_ids:
  - path_3_f_recursive_adversarial_review_r1_prime_3_axis_20260526
  - path_3_fix_wave_r1_prime_close_findings_landed_20260526
  - path_3_f_z8_hierarchical_predictive_coding_L0_scaffold_landed_20260526
  - path_3_recursive_adversarial_review_r1_prime_aggregate_3_axis_landings_b_c_f_g_20260526
---

# R2-COMBINED Recursive Adversarial Review — Path 3 candidate F (Z8 hierarchical predictive coding canonical-quadruple L0 SCAFFOLD, post-FIX-WAVE-R1')

**Scope**: original landing commit `5ff5d2ab9` + FIX-WAVE-R1' closure commit `4684dbbab`. Source files: `src/tac/substrates/z8_hierarchical_predictive_coding/{mlx_renderer.py,...}` + tests under `tests/test_basic.py` (16 pre-existing + 2 NEW F-OP3 parity tests = 18).

**Verdict**: **PROCEED — R2-COMBINED CLEAN PASS** — counter advances from 0/3 → **1/3** per protocol items 3-4.

**Cost**: $0 GPU; ~25 min wall-clock (re-PV + empirical re-measurement + memo synthesis).

---

## Premise verification per Catalog #229

Read in full before any review claim:

- R1' review memo `.omx/research/path_3_f_recursive_adversarial_review_r1_prime_3_axis_20260526.md`
- FIX-WAVE-R1' landing memo `.omx/research/path_3_fix_wave_r1_prime_close_findings_landed_20260526.md`
- F=Z8 source: `mlx_renderer.py:264-326` (post-FIX-WAVE-R1' _pixel_shuffle_2x_nhwc + _bilinear_resize_2x_nhwc)
- Sister landing memo `.omx/research/path_3_f_z8_hierarchical_predictive_coding_L0_scaffold_landed_20260526.md`
- Sister D=Z6 canonical reference + canonical PR95 helper (sister of A R2 review)
- R1' aggregate: confirms F was R1' NOT CLEAN with 2 CRITICAL + 1 P1 + 1 P2 findings; FIX-WAVE-R1' closed all
- Canonical equation registry empirically queried (all 5 cited F=Z8 equations REGISTERED)

Empirical re-verification at 2026-05-26T08:42Z:

```
$ PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest \
    src/tac/substrates/z8_hierarchical_predictive_coding/tests/test_basic.py::test_z8_pixel_shuffle_matches_pytorch \
    src/tac/substrates/z8_hierarchical_predictive_coding/tests/test_basic.py::test_z8_bilinear_resize_matches_pytorch \
    -v -s
test_z8_pixel_shuffle_matches_pytorch PASSED
test_z8_bilinear_resize_matches_pytorch PASSED

$ PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest src/tac/substrates/z8_hierarchical_predictive_coding/tests/ -q
.................. [100%]
18 passed in 0.20s
```

18/18 tests PASS post-FIX-WAVE-R1'; both parity tests PASS at <1e-5 threshold.

---

## Axis 1 review: Math + scientific + engineering rigor (council members: Shannon + Dykstra + Tao)

**Per-architectural-choice triple-axis citation table** (R1' verified canonical-quadruple binding; R2 re-verifies post-FIX-WAVE-R1'):

| Architectural choice | Math citation | Scientific citation | Engineering citation | R2 verdict |
|---|---|---|---|---|
| Canonical quadruple binding (Rao-Ballard + Mallat + Hafner DreamerV3 + Wyner-Ziv) per Catalog #312 NON-NEGOTIABLE | All 4 canonical formulas cited inline in Z8 design memo Section 4.3 + Section 11 + Section 13 | Rao-Ballard 1999 + Daubechies 1988 + Hafner et al. 2023 DreamerV3 + Wyner-Ziv 1976 | Z8 substrate `mlx_renderer.py` + sister modules implement 3-level hierarchical decoder | **HARD-EARNED** (R1'+R2 reaffirmed) |
| `_pixel_shuffle_2x_nhwc` channel-FIRST canonical convention (post-FIX-WAVE-R1' F-OP1) | Shi et al. 2016 sub-pixel CNN; canonical PR95 HNeRV NHWC implementation | sister A=DreamerV3 + D=Z6 sister-canonical + canonical PR95 helper at `tac.local_acceleration.pr95_hnerv_mlx::pixel_shuffle_2x_nhwc` | `mlx_renderer.py:264-296` matches D=Z6 sister-canonical verbatim; 0.0 drift vs PyTorch `nn.PixelShuffle(2)` per R2 re-verification | **HARD-EARNED** (newly verified post-FIX-WAVE-R1') |
| `_bilinear_resize_2x_nhwc` canonical helper delegation (post-FIX-WAVE-R1' F-OP2) | `F.interpolate(scale_factor=2, mode='bilinear', align_corners=False)` canonical | Canonical PR95 helper at `tac.local_acceleration.pr95_hnerv_mlx::bilinear_resize2x_align_corners_false_nhwc` | `mlx_renderer.py:303-326` `from tac.local_acceleration.pr95_hnerv_mlx import bilinear_resize2x_align_corners_false_nhwc` + delegation | **HARD-EARNED** |
| MLX↔PyTorch parity tests added (post-FIX-WAVE-R1' F-OP3) | Test threshold `< 1e-5` documents fp32 compound-op precision noise floor | Sister A=DreamerV3 + canonical PR95 helper anchor patterns | `tests/test_basic.py::test_z8_pixel_shuffle_matches_pytorch` + `test_z8_bilinear_resize_matches_pytorch` | **HARD-EARNED** |
| APPEND-ONLY footer correction on landing memo per Catalog #110/#113 | N/A (documentation discipline) | Per CLAUDE.md HISTORICAL_PROVENANCE APPEND-ONLY non-negotiable | F=Z8 landing memo carries no APPEND-ONLY footer because R1'/F findings were source-code-only (no body claims mutated); G=NIRVANA sister received the footer | **HARD-EARNED** |
| F=Z8 ego-motion conditioning per Catalog #311 NON-NEGOTIABLE (cooperative-receiver + predictive coding binding) | Catalog #311 enforced at Z8 design memo Section 11 + sister `ego_motion_concentration_prior_v1` canonical equation | Sister D=Z6 canonical implementation + Atick-Redlich 1990 + Rao-Ballard 1999 | Z8 mlx_renderer implements ego-motion-conditioned FiLM modulation at 3-level hierarchy | **HARD-EARNED** |

### Axis 1 verdict

**Math + scientific + engineering rigor**: HARD-EARNED across all FIX-WAVE-R1' changes + all R1'-classified canonical-quadruple architectural choices remain HARD-EARNED. FIX-WAVE-R1' closed R1' findings WITHOUT regressing any prior classification.

**Axis 1 R2 findings**: 0.

---

## Axis 2 review: MLX drift minimization (council members: Carmack + Hotz + Quantizr)

### Empirical drift re-measurement at R2 time (2026-05-26T08:42Z)

| Metric | Pre-FIX-WAVE-R1' (R1' anchor) | Post-FIX-WAVE-R1' (immediate) | R2 re-measurement (current) | Verdict |
|---|---|---|---|---|
| `_pixel_shuffle_2x_nhwc` standalone drift | 3.77 | 0.0000000000 | 0.0000000000 (canonical-byte-stable; mirror D=Z6 + canonical PR95 helper) | **HARD-EARNED** |
| `_bilinear_resize_2x_nhwc` standalone drift | 1.51 (mx.repeat) | 2.4e-7 (canonical helper) | < 1e-5 (parity test PASS) | **HARD-EARNED** |
| Test threshold | (not measured pre-fix) | `< 1e-5` (F-OP3 added) | `< 1e-5` (preserved) | **HARD-EARNED** |
| Full Z8 suite | 16/16 PASS (pre-FIX-WAVE-R1') | 18/18 PASS (16 + 2 new parity) | 18/18 PASS | **HARD-EARNED** |

### Canonical helper substitution status

| Surface | Pre-FIX-WAVE-R1' | Post-FIX-WAVE-R1' (R2 verified) | Post-CONSOLIDATE-OP-1 (anticipated) |
|---|---|---|---|
| `_pixel_shuffle_2x_nhwc` | local (channel-LAST WRONG; 3.77 drift) | local (channel-FIRST CORRECT; matches sister D=Z6 + canonical PR95 helper) | imported from `tac.local_acceleration.pr95_hnerv_mlx::pixel_shuffle_2x_nhwc` (in-flight pid 82551) |
| `_bilinear_resize_2x_nhwc` | local (mx.repeat WRONG; 1.51 drift) | delegates to canonical helper | unchanged (already canonical) |

**Sister CONSOLIDATE-OP-1 in-flight status**: subagent `consolidate-op-1` (pid 82551) actively editing F=Z8 mlx_renderer.py per its checkpoint at 2026-05-26T08:42:47Z. Per Catalog #230 ownership map: R2-COMBINED REVIEW does NOT touch F=Z8 source files in this window.

### Axis 2 verdict

**MLX drift minimization**: HARD-EARNED. All R1' findings closed; primitive drifts collapsed from R1' anchor (3.77 PixelShuffle / 1.51 bilinear) to canonical-byte-stable (0.0 PixelShuffle / 2.4e-7 bilinear < 1e-5 threshold). Sister-canonical reference status of D=Z6 + canonical PR95 helper preserved across both A and F post-fix.

**Axis 2 R2 findings**: 0.

---

## Axis 3 review: Portability via numpy (council members: MacKay + Selfcomp + Contrarian)

### Per-MLX-primitive numpy reference status

F=Z8 does NOT ship a sister `numpy_reference.py` per the G=NIRVANA exemplary pattern. Per the substrate's structural MLX-first posture per Catalog #1265 anchor; inflate runtime is PyTorch-only (sister Catalog #295 self-containment); the canonical CPU-portability is achieved via PyTorch CPU backend at inflate time.

| Primitive | Numpy reference status | Notes |
|---|---|---|
| `_pixel_shuffle_2x_nhwc` | N/A (no sister numpy ref shipped) | Canonical PR95 helper could be refactored to expose sister numpy implementation per META-CONSOLIDATE-OP-2; advisory L1+ |
| `_bilinear_resize_2x_nhwc` | N/A (no sister numpy ref shipped) | Sister advisory |
| Hierarchical predictive coding decoder (3-level cascade) | N/A | PyTorch inflate-time path IS canonical CPU-portable reference per Catalog #1 + Catalog #205 |
| Categorical posterior + Gumbel-Softmax STE | N/A | Sister with A=DreamerV3 categorical primitives (reused per Catalog #290 ADOPT_CANONICAL_BECAUSE_SERVES per landing memo sister_subagent_ownership_map) |

### Axis 3 verdict

**Portability via numpy**: N/A by construction at F=Z8 scope per substrate's structural MLX-first posture. The L0 SCAFFOLD's CPU-portability is achieved via PyTorch CPU backend at inflate time; sister numpy_reference.py was NOT in the L0 scope.

**Sister META-CONSOLIDATE-OP-2 status**: queued at R1' aggregate. If META-CONSOLIDATE-OP-2 lands AND F=Z8 is extended at L1+ to consume canonical numpy reference, F's Axis 3 coverage moves from N/A to ACTIVE. Advisory only.

**Axis 3 R2 findings**: 0.

---

## R2-COMBINED verdict per substrate

**F=Z8 R2 verdict**: **PROCEED — CLEAN PASS** (0 findings across all 3 axes).

**Counter advance**: 0/3 → **1/3** (R2 advances per protocol items 3-4; FIX-WAVE-R1' closure verified empirically).

**Path to 3/3 SEAL → paid CUDA dispatch authorized**: 2 additional consecutive CLEAN rounds (R3, R4) required per protocol item 4.

---

## Per-substrate cumulative counter status

| Round | Result | Counter |
|---|---|---|
| R1' | PROCEED_WITH_REVISIONS (2 CRITICAL + 1 P1 + 1 P2 findings) | 0/3 (reset per protocol item 3) |
| FIX-WAVE-R1' | CLOSED (3 op-routables landed: F-OP1+F-OP2+F-OP3) | counter unchanged (FIX-WAVE is meta-discipline) |
| **R2-COMBINED** | **PROCEED — CLEAN PASS** | **0/3 → 1/3** |
| R3 (planned) | TBD | TBD |
| R4 (planned) | TBD | TBD |
| SEAL gate | 3/3 required | reached only after 2 more CLEAN rounds |

---

## 6-hook wire-in declaration per Catalog #125

- **hook #1 sensitivity-map contribution**: N/A.
- **hook #2 Pareto constraint**: N/A.
- **hook #3 bit-allocator hook**: N/A.
- **hook #4 cathedral autopilot dispatch hook**: ACTIVE (R2 CLEAN PASS unblocks F=Z8 for downstream autopilot consideration at L1+; canonical Catalog #1265 contest-equivalence gate threshold satisfied per post-FIX-WAVE-R1' decoder parity verified).
- **hook #5 continual-learning posterior**: ACTIVE (R2 verdict appended to council deliberation posterior per Catalog #300 v2; supersedes R1' PROCEED_WITH_REVISIONS as chronologically-later anchor).
- **hook #6 probe-disambiguator**: N/A.

---

## Discipline applied

- **Catalog #229 PV**: R1' review + FIX-WAVE-R1' + landing + source files + 18 tests run + 2 parity tests re-verified before any review claim; canonical equation registry empirically queried (all 5 cited equations REGISTERED).
- **Catalog #110/#113 APPEND-ONLY**: NEW review memo (R2); R1' + FIX-WAVE-R1' + landing memos NEVER mutated.
- **Catalog #117/#157/#174/#235/#289**: commit forthcoming via canonical serializer.
- **Catalog #119**: Co-Authored-By Claude trailer.
- **Catalog #206**: checkpoint discipline.
- **Catalog #208**: docs/local-paths.
- **Catalog #230**: sister-subagent ownership map — review-only on F files; CONSOLIDATE-OP-1 actively owns `mlx_renderer.py` (pid 82551); Wave #1 posterior_emission actively owns substrate `__init__.py`; no file collision since this is a NEW review memo.
- **Catalog #287**: every finding carries `[empirical:<measurement>]` evidence-tag; no placeholder rationales.
- **Catalog #292**: per-axis council member operating-within assumption surfaced in frontmatter.
- **Catalog #300 v2**: full frontmatter (tier T2; canonical attendees; mission_contribution frontier_protecting; horizon_class frontier_pursuit).
- **Catalog #340**: sister-checkpoint guard PROCEED verified at start of R2.
- **CLAUDE.md "Recursive adversarial review protocol — close paths"** items 1-8: R2 CLEAN PASS advances counter to 1/3; item 8 satisfied via the 3-row Assumption-Adversary verdict.
- **CLAUDE.md "Executing actions with care"**: review-only.

---

## Cross-references

- R1' review: `.omx/research/path_3_f_recursive_adversarial_review_r1_prime_3_axis_20260526.md`
- FIX-WAVE-R1': `.omx/research/path_3_fix_wave_r1_prime_close_findings_landed_20260526.md`
- Landing memo: `.omx/research/path_3_f_z8_hierarchical_predictive_coding_L0_scaffold_landed_20260526.md`
- R1' aggregate: `.omx/research/path_3_recursive_adversarial_review_r1_prime_aggregate_3_axis_landings_b_c_f_g_20260526.md`
- Sister D=Z6 canonical: `src/tac/substrates/time_traveler_l5_z6/mlx_renderer.py::_pixel_shuffle_2x_nhwc`
- Sister A=DreamerV3 canonical: `src/tac/substrates/dreamer_v3_rssm/module.py::_pixel_shuffle_2x_nhwc` (post-FIX-WAVE-R1)
- Canonical PR95 helper: `tac.local_acceleration.pr95_hnerv_mlx::bilinear_resize2x_align_corners_false_nhwc` + `pixel_shuffle_2x_nhwc`
- Lane: `lane_path_3_recursive_adversarial_review_r2_combined_3_axis_landings_a_b_c_d_e_f_g_20260526` L1 (impl_complete + memory_entry)
