<!-- SPDX-License-Identifier: MIT -->
<!-- Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE — R1' AGGREGATE review memo; do not mutate after landing. -->
<!-- Catalog #344 canonical equation cross-ref: cross-substrate META review across 4 sister R1' memos; FIX-WAVE-R1' op-routable queue priority-ranked. -->
---
council_tier: T3
council_attendees:
  - Shannon
  - Dykstra
  - Rudin
  - Daubechies
  - Tao
  - Carmack
  - Hotz
  - Quantizr
  - MacKay
  - Selfcomp
  - Ballé
  - Hassabis
  - PR95Author
  - Contrarian
  - Assumption-Adversary
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "T3 tier is appropriate for an AGGREGATE R1' review spanning 4 sister landings"
    classification: HARD-EARNED
    rationale: "Per CLAUDE.md 'Council hierarchy: 4-tier protocol' Tier elevation triggers T2→T3 trigger (a): the aggregate finding touches a CLAUDE.md non-negotiable (Recursive adversarial review protocol — close paths — items 1-8 + new item 8 assumption-challenge axis). Sister R1 aggregate was T3; consistency. T3 cadence budget ≤3/week is within bounds; this is the 2nd T3 this week (R1 aggregate was 1st)."
  - assumption: "Cross-substrate META findings WARRANT a CONSOLIDATE-OP queued for L1+ rather than blocking R2'"
    classification: HARD-EARNED
    rationale: "F=Z8's TRAINING-INVALIDATING bugs are LINE-FOR-LINE inheritance of pre-FIX-WAVE-R1 A=DreamerV3 bugs. The META class (locally-invented MLX primitives diverge from sister-canonical) was R1's META finding #1; FIX-WAVE-R1 closed A=DreamerV3 instance but did NOT close the META class because the canonical helper (META-CONSOLIDATE-OP-1) is still queued for L1+. F=Z8 is the EMPIRICAL PROOF that the META class continues to recur. CONSOLIDATE-OP must escalate from L1+ advisory to FIX-WAVE-R1' P1 priority."
  - assumption: "FIX-WAVE-R1' successor subagent can land all NOT-CLEAN substrates' fixes in a single commit batch"
    classification: HARD-EARNED
    rationale: "B' = CLEAN (no fixes). C' = CLEAN (no fixes). F=Z8 = 3 op-routables (2 P0 code-fixes + 1 P1 verification; ≤60 LOC across 1 file). G=NIRVANA = 3 P0 documentation-only op-routables (≤30 LOC across 2 files). Total touch surface: ≤4 files; ≤90 LOC of edits. Sister-coherence per Catalog #230 ownership map: zero overlap with the 4 OTHER in-flight Path 3 candidates H/I/J/K + L1-PROMOTION-D-Z6."
council_decisions_recorded:
  - "Aggregate R1' verdict: NOT CLEAN (2 of 4 substrates require FIX-WAVE-R1'; the other 2 (B' + C') pass cleanly)"
  - "Counter resets to 0 per CLAUDE.md 'Recursive adversarial review protocol — close paths' item 3"
  - "FIX-WAVE-R1' successor subagent required BEFORE R2' fires per protocol item 4"
  - "Priority-ranked FIX-WAVE-R1' op-routable queue: see §FIX-WAVE-R1' op-routable queue below"
  - "R2' readiness verdict: BLOCKED until FIX-WAVE-R1' lands and re-runs (a) the 16 F=Z8 tests post-fix + 2 NEW parity tests + re-measure F=Z8 PixelShuffle + bilinear drift + (b) the 27 G=NIRVANA tests post-doc-fix"
  - "META-CONSOLIDATE-OP-1 PRIORITY ESCALATION: from L1+ advisory to FIX-WAVE-R1' P1 priority — F=Z8 is the 2nd substrate to ship this bug class (A=DreamerV3 was 1st; FIX-WAVE-R1 closed A; META class continues to recur)"
  - "CONSOLIDATE-OP-2 NEW PROPOSAL: extract G=NIRVANA's numpy_reference.py 7-primitive canonical pattern to canonical `tac.local_acceleration.numpy_reference` (or equivalent) per axis 3 portability discipline"
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
council_override_rationale: ""
horizon_class: frontier_pursuit
canonical_equation_refs:
  - mlx_pytorch_full_decoder_downstream_scorer_drift_propagation_v1
  - categorical_posterior_capacity_vs_continuous_gaussian_v1
  - procedural_codebook_from_seed_compression_savings_v1
  - categorical_blahut_arimoto_rate_distortion_v1
predicted_band_validation_status: pending_post_training
related_deliberation_ids:
  - path_3_b_recursive_adversarial_review_r1_prime_3_axis_20260526
  - path_3_c_recursive_adversarial_review_r1_prime_3_axis_20260526
  - path_3_f_recursive_adversarial_review_r1_prime_3_axis_20260526
  - path_3_g_recursive_adversarial_review_r1_prime_3_axis_20260526
  - path_3_recursive_adversarial_review_r1_aggregate_3_axis_landings_a_d_e_20260526
  - path_3_fix_wave_r1_close_findings_landed_20260526
---

# R1' Recursive Adversarial Review — AGGREGATE across Path 3 candidates B' + C' + F + G

**Per binding operator directive 2026-05-26**: *"we also need adversarial review against all landing recursive for math and scientific and engineering rigor and for MLX drift minimization and portability via numpy"*

**Per CLAUDE.md "Recursive adversarial review protocol — close paths"**: Round 1' (R1-prime) of 3 consecutive clean-pass cycles required before code is cleared for L1 dispatch authorization. R1' fires AFTER R1 landed (commit `80acd6da3`; aggregate across A+D+E) AFTER FIX-WAVE-R1 closed P0+P1+P2 (commit `a23779a732e7bb056`); R1' covers the 4 NEW landings (B' + C' + F + G) that arrived AFTER R1 fired.

**Aggregate R1' verdict**: **NOT CLEAN — counter resets to 0 per protocol item 3**

**Cost**: $0 GPU; ~3.5h wall-clock (per-substrate review 30-60min each + empirical MLX parity measurement + aggregate synthesis ~30min)

---

## Per-landing R1' verdict summary

| Landing | Commit | Substrate path | R1' Verdict | Counter advance? | Findings count | FIX-WAVE-R1' required? |
|---|---|---|---|---|---|---|
| **B'=Z7-Mamba-2-v2 fresh substrate** | `7a103fdbb` | `src/tac/substrates/z7_mamba2_v2_fresh_substrate/` | **PROCEED — R1' CLEAN PASS** | YES (1/3 advance) | 0 R1' findings; 2 anticipatory L1+ advisories | NO |
| **C'=NSCS06 v8 chroma_lut cargo-cult-first** | `f59c8401b` | `src/tac/substrates/nscs06_v8_chroma_lut/` | **PROCEED — R1' CLEAN PASS** | YES (1/3 advance) | 0 R1' findings | NO |
| **F=Z8 hierarchical predictive coding canonical-quadruple** | `5ff5d2ab9` | `src/tac/substrates/z8_hierarchical_predictive_coding/` | **PROCEED_WITH_REVISIONS** | NO (counter resets) | 2 CRITICAL Axis 2 (MLX drift) + 1 P1 verification + 1 P2 advisory | **YES — 3 op-routables (2 code-fix + 1 test)** |
| **G=NIRVANA cascading NeRV** | `f7d2e86fe` | `src/tac/substrates/nirvana_cascading_nerv/` | **PROCEED_WITH_REVISIONS** | NO (counter resets) | 3 documentation-only findings (label-drift class) | **YES — 3 documentation-only op-routables** |

**Aggregate**: 2/4 substrates require FIX-WAVE-R1' successor. Per CLAUDE.md "Recursive adversarial review protocol — close paths" item 3: a round with ANY issue resets the counter to 0. Per item 4: FIX-WAVE-R1' successor subagent required BEFORE R2' fires.

---

## Cross-substrate META findings

### META finding #1: F=Z8 IS THE EMPIRICAL PROOF that the locally-invented-MLX-primitive bug class RECURS (escalation: R1 META finding #1)

**Class**: When two sister substrates BOTH implement a "canonical" primitive locally instead of routing through a SHARED canonical helper, divergence is inevitable AND RECURS across new landings.

**Empirical anchor (R1' measurement)**:
- F=Z8 `_pixel_shuffle_2x_nhwc` (mlx_renderer.py:274-276): uses channel-LAST convention `(0, 1, 3, 2, 4, 5)` → **3.77 absolute drift vs PyTorch**
- F=Z8 `_bilinear_resize_2x_nhwc` (mlx_renderer.py:282-284): uses `mx.repeat` 2x → **1.51 absolute drift vs PyTorch**
- A=DreamerV3 PRE-FIX-WAVE-R1: same bugs (R1 anchor: 2.4 drift PixelShuffle + 24.34 drift bilinear)
- D=Z6 (canonical sister-canonical reference): 0.0 drift
- Canonical PR95 helper (`tac.local_acceleration.pr95_hnerv_mlx::bilinear_resize2x_align_corners_false_nhwc`): 0.0 drift

The R1 META finding #1 op-routable (#9 META-CONSOLIDATE-OP-1: extract `_pixel_shuffle_2x_nhwc` + general `_bilinear_resize_nhwc` to canonical `tac.local_acceleration.pr95_hnerv_mlx`) was QUEUED for L1+. F=Z8 landed BEFORE the CONSOLIDATE-OP, inheriting the buggy pre-FIX-WAVE-R1 A=DreamerV3 convention. This empirically proves the CONSOLIDATE-OP cannot be deferred — every new MLX substrate that lands BEFORE the canonical helper exists will re-invent the bug locally.

**Op-routable (PRIORITY ESCALATION)**: META-CONSOLIDATE-OP-1 from L1+ advisory → FIX-WAVE-R1' **P1 priority**. The CONSOLIDATE-OP should land alongside F-OP1 + F-OP2 (which currently INLINE the canonical-correct convention into Z8). After CONSOLIDATE-OP lands, F=Z8 + A=DreamerV3 + future MLX substrates ALL import from canonical helper.

### META finding #2: G=NIRVANA's numpy_reference.py IS THE CANONICAL SISTER REFERENCE PATTERN for axis 3 portability (positive META; NEW CONSOLIDATE-OP-2 proposed)

**Class**: A substrate that ships a sister `numpy_reference.py` with 1-to-1 numpy implementations of every anticipated MLX primitive + PyTorch parity tests creates the canonical pattern that ALL future MLX substrates should inherit.

**Empirical anchor**: G=NIRVANA's `numpy_reference.py` (329 LOC across 7 primitives: to_float32 / linear / conv2d_nhwc / bilinear_upsample_2x_nhwc / sigmoid / sin / mean + kahan_mean + cascade_reconstruct) with PyTorch parity tests verified ≤ 1e-5 fp32 / ≤ 1e-3 fp16. Substrate is operable on CPU-only test rigs WITHOUT MLX (27/27 tests PASS without MLX).

Per the operator directive #3 axis 3 portability requirement: G=NIRVANA is the EXEMPLARY pattern. R1' (this aggregate) proposes **NEW CONSOLIDATE-OP-2**: extract G=NIRVANA's pattern to canonical `tac.local_acceleration.numpy_reference` (or equivalent canonical location) so future Path 3 substrates inherit ONE source of truth at the numpy-reference layer.

**Op-routable**: queued for L1+ per CLAUDE.md "consolidate into META layer" standing directive; NOT blocking FIX-WAVE-R1'.

### META finding #3: documentation-overstatement bug class recurs (G=NIRVANA mislabels numpy parity as MLX parity)

**Class**: When a substrate has a numpy reference implementation AND a sister-anticipated MLX implementation that has not yet landed, the documentation labels the numpy parity evidence as MLX parity evidence — overstatement bug class identical to E=BoostNeRV's BPR1 header size drift (R1 META finding #2).

**Empirical anchor**: G=NIRVANA landing memo lines 70, 71, 80, 81, 102-108 repeatedly cite "axis 2 MLX↔PyTorch parity ≤ 1e-5" when the actual test compares numpy_reference vs PyTorch (NO MLX in the test). Sister docstring at `mlx_renderer.py:2` claims "MLX hierarchical residual decoder cascade" when the module contains config + helpers only (no renderer class).

Sister CLAUDE.md anchor: "Comment-only contracts are FORBIDDEN" non-negotiable + Catalog #287 anti-overstatement discipline.

**Op-routable** (P0 FIX-WAVE-R1' documentation-only): 3 string edits across 2 files (landing memo + mlx_renderer.py docstring) per G-OP1 + G-OP2 + G-OP3.

### META finding #4: cargo-cult-pass-FIRST methodology is empirically materialized in B' + C' (positive META)

**Class**: When a substrate's Phase 1 audit explicitly identifies cargo-culted assumptions AND Phase 3 implements unwinds with explicit dataclass contracts + non-promotable Provenance + tests + empirical confirmation, the substrate is EMPIRICALLY HARDER-EARNED than substrates without this discipline.

**Empirical anchor**:
- B' Phase 1 audit identified 8 NEW CC + 2 NEW HARD-EARNED-PARTIAL beyond CC-1..CC-10; Phase 3 implements 16-layer canonical-vs-unique decision table with 8 UNIQUE-FORK + 6 CANONICAL-ADOPT + 1 UNIQUE-IMPL + 1 UNIQUE-DESIGN.
- C' Phase 1 audit identified 4 CARGO-CULTED-CRITICAL; Phase 3 implements all 4 unwinds; cargo-cult #5 EMPIRICALLY CONFIRMED via FAIL_AT_CLASS_1 verdict at L0.

This is the canonical example of operator directive #2 compliance: *"Never simply extend unless a rigorous adversarial cargo cult pass has been done first"*. Future Path 3 candidates should adopt this methodology (cargo-cult-audit-first → design-decision → L0 SCAFFOLD with empirical disambiguator at L0). Sister H=ATW_V2 already shows signs of this discipline per its 3-phase cargo-cult-first methodology in the in-flight queue.

**Op-routable**: None (positive finding); the discipline IS the canonical anchor.

### META finding #5: shared MLX-primitive correctness invariant — D=Z6 + canonical PR95 helper jointly are THE sister-canonical reference (R1 META #3 extended)

**Class**: Across the 7 reviewed substrates (A+D+E from R1 + B'+C'+F+G from R1'), D=Z6's MLX primitives + canonical PR95 helper are the only consistently PyTorch-byte-stable references. Every substrate that does NOT route through D=Z6's convention OR canonical PR95 helper has empirically-measured drift.

**Empirical anchor (R1 + R1' joint measurement)**:

| Substrate | PixelShuffle 2x NHWC max_abs vs PyTorch | Bilinear 2x NHWC max_abs vs PyTorch |
|---|---|---|
| A=DreamerV3 PRE-FIX-WAVE-R1 | 2.40 | 24.34 (mx.repeat) |
| A=DreamerV3 POST-FIX-WAVE-R1 | 0.00 | 0.00 (canonical helper) |
| D=Z6 (sister-canonical reference) | 0.00 | N/A (uses canonical _bilinear_resize_nhwc helper) |
| E=BoostNeRV | N/A (PyTorch-only substrate; no MLX) | N/A |
| B'=Z7-Mamba-2-v2 | N/A at L0 (no MLX primitives shipped) | N/A at L0 |
| C'=NSCS06 v8 chroma_lut | N/A (non-MLX substrate; numpy + PyTorch only) | N/A |
| F=Z8 | **3.77** (NOT CLEAN; sister A pre-fix bug) | **1.51** (NOT CLEAN; mx.repeat bug) |
| G=NIRVANA | N/A at L0 (no MLX renderer shipped; numpy_reference correct) | N/A at L0 |

**No op-routable** beyond what's already in CONSOLIDATE-OP-1 priority escalation; this is a positive META finding (sister-canonical references identified empirically across 7 substrates).

### META finding #6: canonical equation registry empirically queried — equation registrations honored across R1 + R1'

**Class**: Canonical equation citations in landing memos can be EMPIRICALLY VERIFIED via the canonical equation registry. R1' verified all 4 substrates' cited equations:

| Substrate | Cited canonical equation | REGISTERED status |
|---|---|---|
| B' | (none declared in frontmatter; consistent with v1 sister gap) | — |
| C' | `procedural_codebook_from_seed_compression_savings_v1` | ✓ REGISTERED |
| F=Z8 | `mlx_pytorch_full_decoder_downstream_scorer_drift_propagation_v1` | ✓ REGISTERED |
| F=Z8 | `scorer_conditional_joint_rate_distortion_floor_v1` | ✓ REGISTERED |
| F=Z8 | `categorical_posterior_capacity_vs_continuous_gaussian_v1` | ✓ REGISTERED |
| F=Z8 | `ego_motion_concentration_prior_v1` | ✓ REGISTERED |
| F=Z8 | `cross_codec_super_additive_orthogonality_predictor_v1` | ✓ REGISTERED |
| G=NIRVANA | (none declared in frontmatter; advisory: register Mallat wavelet + NeRV-family canonical equations at Phase 2) | — |

**Op-routable** (P2 advisory; not blocking FIX-WAVE-R1'): B' + G=NIRVANA register canonical equations at L1+ per Catalog #344 sister discipline.

### META finding #7: training-invalidating MLX bugs surface ONLY at the PyTorch-export boundary (R1 META #5 reaffirmed by F=Z8)

**Class**: F=Z8's L0 smoke trainer "2537.68 → 2503.03 → 2500.61 monotonic decrease" on SYNTHETIC random targets does NOT reveal the PixelShuffle + bilinear bugs because targets are noise. The bugs surface structurally at L1+ score-aware-loss training where the MLX trainer optimizes against MLX_buggy_decoder while PyTorch inflate uses CORRECT canonical primitives — state-dict-transferred frames at PyTorch-inflate DO NOT MATCH frames MLX trainer observed at convergence.

This is the EXACT mechanism R1 identified for A=DreamerV3. F=Z8's "16/16 PASS" tests are SHAPE + smoke + manifest tests; NONE measure MLX↔PyTorch parity at the decoder forward boundary.

**Op-routable**: F-OP3 (add MLX↔PyTorch parity tests for both primitives per F=Z8 review) PLUS sister test pattern should be REPLICATED at every future MLX substrate's test suite (META-CONSOLIDATE-OP test-template-extraction queued for L1+).

---

## FIX-WAVE-R1' op-routable queue (priority-ranked)

Per CLAUDE.md "Recursive adversarial review protocol — close paths" item 2: all FIX-WAVE-R1' issues land in a successor subagent + are committed BEFORE R2' begins.

### P0 / CRITICAL / TRAINING-INVALIDATING (F=Z8 only)

1. **F-OP1**: rewrite `src/tac/substrates/z8_hierarchical_predictive_coding/mlx_renderer.py::_pixel_shuffle_2x_nhwc` from channel-LAST convention `(B, H, W, 2, 2, out_C)` + transpose `(0, 1, 3, 2, 4, 5)` to channel-FIRST convention `(B, H, W, out_C, 2, 2)` + transpose `(0, 1, 4, 2, 5, 3)` matching SISTER-CANONICAL impl in D=Z6 + canonical PR95 helper + post-FIX-WAVE-R1 A=DreamerV3. Empirical: D=Z6's convention is byte-stable vs PyTorch reference (R1' measurement 0.0 drift).

2. **F-OP2**: replace `src/tac/substrates/z8_hierarchical_predictive_coding/mlx_renderer.py::_bilinear_resize_2x_nhwc` (mx.repeat 2x) with import + usage of canonical `tac.local_acceleration.pr95_hnerv_mlx::bilinear_resize2x_align_corners_false_nhwc`. Empirical: canonical helper is byte-stable vs PyTorch reference.

### P0 / DOCUMENTATION FIX (G=NIRVANA only)

3. **G-OP1**: update landing memo `.omx/research/path_3_g_nirvana_cascading_nerv_L0_scaffold_landed_20260526.md` lines 70-71 + 80-81 + 100-108: replace "axis 2 MLX↔PyTorch parity" labels with "axis 3 numpy↔PyTorch parity" (or explicit "MLX↔PyTorch parity deferred to Phase 2 when actual MLX renderer lands")

4. **G-OP2**: update landing memo line 26: change "MLX-first per #1265 anchor" to "MLX-config-scaffold-first per #1265 anchor; actual MLX renderer implementation deferred to Phase 2 council symposium per Catalog #325"

5. **G-OP3**: update `src/tac/substrates/nirvana_cascading_nerv/mlx_renderer.py` line 2 docstring: change "MLX hierarchical residual decoder cascade" to "MLX hierarchical residual decoder cascade SCAFFOLD (config + helpers; actual renderer class lands Phase 2)"

### P1 / VERIFICATION (F=Z8 only)

6. **F-OP3**: add to `src/tac/substrates/z8_hierarchical_predictive_coding/tests/test_basic.py`: `test_z8_pixel_shuffle_matches_pytorch` + `test_z8_bilinear_resize_matches_pytorch` (both assert max_abs < 1e-5 vs PyTorch reference); mirror A=DreamerV3 post-FIX-WAVE-R1 verification pattern.

### P1 / META-CONSOLIDATE-OP-1 PRIORITY ESCALATION (queued for FIX-WAVE-R1' or immediately after)

7. **META-CONSOLIDATE-OP-1** (ESCALATED from R1 aggregate L1+ advisory): extract `_pixel_shuffle_2x_nhwc` (channel-FIRST convention from D=Z6) + general `_bilinear_resize_nhwc` (D=Z6's impl works for arbitrary target_h, target_w) to canonical `tac.local_acceleration.pr95_hnerv_mlx` per CLAUDE.md "consolidate into META layer" standing directive. After CONSOLIDATE-OP lands: refactor A=DreamerV3 + D=Z6 + F=Z8 (post FIX-WAVE-R1') + future Path 3 candidates (B' L1+, G=NIRVANA Phase 2, etc.) to import from canonical. EMPIRICAL JUSTIFICATION: F=Z8 is the EMPIRICAL PROOF the bug class recurs without the CONSOLIDATE-OP.

### P2 / META-CONSOLIDATE-OP-2 NEW PROPOSAL (advisory; L1+)

8. **META-CONSOLIDATE-OP-2**: extract G=NIRVANA's `numpy_reference.py` 7-primitive canonical pattern to canonical `tac.local_acceleration.numpy_reference` (or equivalent canonical location) per axis 3 portability discipline. After CONSOLIDATE-OP lands: future Path 3 candidates can inherit canonical numpy references rather than re-implementing.

### P2 / ADVISORY (B' + G=NIRVANA L1+)

9. **B'-L1-ADV1**: when L1 implements MLX-native Mamba2V2Cell + Mamba2TemporalDecoder, MUST adopt META-CONSOLIDATE-OP-1 canonical helpers (NOT re-invent locally).
10. **B'-L1-ADV2**: when L1 implements MLX primitives, MUST land sister `numpy_reference.py` per G=NIRVANA canonical pattern (or import canonical per META-CONSOLIDATE-OP-2 if landed).
11. **G-L1-ADV1**: when L1 implements actual MLX renderer, MUST adopt META-CONSOLIDATE-OP-1 canonical helpers (NOT re-invent locally).
12. **B' + G=NIRVANA EQUATION-REGISTRY-OP**: register canonical equations at L1+ per Catalog #344 sister discipline.

### L1+ / TEST-TEMPLATE-EXTRACTION (advisory; deferred)

13. **L1+ TEST-CONSOLIDATE-OP**: extract the MLX↔PyTorch parity test pattern (per F-OP3) to a canonical test fixture/helper at `src/tac/local_acceleration/tests/test_mlx_pytorch_parity.py` so future MLX substrates inherit the test contract rather than re-implementing per-substrate. META class extinction at the test surface.

---

## R2' readiness verdict

**R2' BLOCKED until FIX-WAVE-R1' lands the P0 op-routables:**

1. F-OP1 + F-OP2 + F-OP3 (Z8 code fixes + verification tests)
2. G-OP1 + G-OP2 + G-OP3 (NIRVANA documentation-only fixes across 2 files)

After P0 ops land, verification:
- B': no changes; re-run 40 tests; verify all PASS (sister-coherence preserved)
- C': no changes; re-run 180 tests; verify all PASS (sister-coherence preserved)
- F=Z8: re-run 16 tests + 2 NEW parity tests; verify all 18 PASS; re-measure Z8 PixelShuffle + bilinear drift; verify both max_abs < 1e-5 vs PyTorch
- G=NIRVANA: re-run 27 tests; verify all PASS (no test changes; doc-only fixes); verify landing memo + mlx_renderer.py docstring correctly reflect scaffold-only posture

If verification passes, R2' review subagent can fire per CLAUDE.md "Recursive adversarial review protocol — close paths" — taking different adversarial perspectives per protocol item 1 (rotation across the 4-co-lead structure + sister inner-council voices). Suggested R2' rotation per protocol item 1:
- Shannon LEAD (information theory at canonical-quadruple binding)
- Dykstra CO-LEAD (alternating-projections feasibility on 4-axis polytope)
- Tao + Boyd (math rigor + convex optimization at Phase 2 design boundary)
- Hafner-DreamerV3-author-cite (canonical posterior at Z8 multi-level surface)
- Atick-Redlich + Tishby-memorial (cooperative-receiver lens at Z8 Wyner-Ziv top-level)

P1 + P2 op-routables (F-OP3, META-CONSOLIDATE-OP-1, META-CONSOLIDATE-OP-2) SHOULD land in the same FIX-WAVE-R1' commit batch but META-CONSOLIDATE-OP-1 + CONSOLIDATE-OP-2 + L1+ advisories are NOT strict blockers for R2' (they are verification / amendment / advisory).

---

## Counter state per CLAUDE.md "Recursive adversarial review protocol — close paths"

- **Before R1'**: counter = 0 (4 NEW landings post-R1; no prior cycle history)
- **Per-landing post-R1'**:
  - B' = CLEAN → would advance to 1/3 if aggregated alone
  - C' = CLEAN → would advance to 1/3 if aggregated alone
  - F=Z8 = NOT CLEAN → counter resets to 0
  - G=NIRVANA = NOT CLEAN → counter resets to 0
- **Aggregate post-R1'**: counter = **0/3** (any NOT CLEAN resets the counter per protocol item 3)
- **Post-FIX-WAVE-R1' + R2'-CLEAN**: counter advances to 1/3 for the 2 NOT-CLEAN substrates (F + G); B' + C' already at 1/3 from their R1' clean-pass + can advance to 2/3 jointly with the F + G post-fix verification
- **Post-R3'-CLEAN**: counter advances to 2/3 (or 3/3 if joint with prior)
- **Post-R4'-CLEAN**: counter advances to 3/3 → cycle closes per protocol gate
- **Operator-declared SEAL (D-1, conservative)**: NOT applicable here (counter-advance path is straightforward)

---

## Discipline applied

- **Catalog #229 PV**: 4 landing memos + 4+ source files per substrate read in full; 40+180+16+27 = 263 tests run + verified PASSING before any review claim; empirical MLX↔PyTorch parity measurement on F=Z8 primitives; sister D=Z6 + canonical PR95 helper cross-comparison; canonical equation registry empirically queried
- **Catalog #110/#113 APPEND-ONLY**: 5 NEW memos (4 per-substrate + 1 aggregate); sister landing memos NEVER mutated
- **Catalog #117/#157/#174/#235/#289**: all commits via canonical `tools/subagent_commit_serializer.py` with POST-EDIT `--expected-content-sha256`
- **Catalog #119**: Co-Authored-By Claude trailer
- **Catalog #287**: every finding carries `[empirical:<measurement>]` or `[verified-against:<source>]` evidence-tag; no placeholder rationales (every assumption-adversary verdict ≥4 chars; placeholder `<rationale>` / `<reason>` literals REJECTED)
- **Catalog #208**: docs/local-paths — only relative paths cited
- **Catalog #292**: per-axis council member operating-within assumption surfaced explicitly in frontmatter (all 4 per-substrate memos + this aggregate)
- **Catalog #300 v2**: full frontmatter on all 5 memos (tier T2 per-substrate; T3 aggregate; attendees include canonical 4-co-lead structure Shannon+Dykstra+Rudin+Daubechies per 2026-05-19 amendment + the relevant inner council voices; mission_contribution frontier_protecting; horizon_class frontier_pursuit)
- **Catalog #346**: canonical council roster `validate_council_dispatch_roster` returns complete=True for T3 aggregate (Shannon LEAD + Dykstra CO-LEAD + Rudin CO-LEAD + Daubechies CO-LEAD + Tao + Carmack + Hotz + Quantizr + MacKay + Selfcomp + Ballé + Hassabis + PR95Author + Contrarian + Assumption-Adversary = 15 attendees; ≥12-of-20 grand council quorum honored)
- **Catalog #340**: sister-checkpoint guard PROCEED before any edits; no overlap with the 4 OTHER in-flight Path 3 candidates H/I/J/K + L1-PROMOTION-D-Z6 per Catalog #230 ownership map
- **Catalog #206**: checkpoint discipline (3 checkpoints emitted)
- **Catalog #126**: lane `lane_path_3_recursive_adversarial_review_r1_prime_3_axis_landings_b_c_f_g_20260526` pre-registered
- **Catalog #294 9-dim checklist** (1-9): UNIQUENESS (the 3-axis cross-substrate R1' aggregate review is the 2nd of its kind for Path 3, distinct from R1 aggregate by covering 4 NEW landings); BEAUTY + ELEGANCE (5 memos ≤700 lines each); DISTINCTNESS (NOT a sister review; this is an AGGREGATE synthesis with cross-substrate META findings including META-CONSOLIDATE-OP-1 priority escalation and CONSOLIDATE-OP-2 new proposal); RIGOR (PV + empirical MLX parity measurement + canonical equation verification + sister-canonical cross-comparison); OPTIMIZATION PER TECHNIQUE (per-axis council members rotate per protocol); STACK-OF-STACKS-COMPOSABILITY (aggregate composes 4 per-substrate verdicts + cross-references R1 aggregate); DETERMINISTIC REPRODUCIBILITY (reproducer commands documented per Axis 2); EXTREME OPTIMIZATION + PERFORMANCE (R1' review takes ~3.5h wall-clock vs paid-dispatch cost $0); OPTIMAL MINIMAL CONTEST SCORE (R1' is QUALITY GATE not score-claim; non-promotable per Catalog #287 / #341 Tier A)
- **CLAUDE.md "Recursive adversarial review protocol — close paths"** items 1-8: this aggregate review IS R1'; counter resets to 0; FIX-WAVE-R1' successor subagent required
- **CLAUDE.md "Council conduct" amendment 2026-05-19 4-co-lead structure**: T3 aggregate roster includes all 4 co-leads per Catalog #346 requirement
- **CLAUDE.md "Executing actions with care"**: review-only (NO code modifications); fixes are FIX-WAVE-R1' successor subagent's scope
- **Operator pacing directive 2026-05-26**: "Keep feeding the queue but we need to be mindful not to outpace session rate limits" — efficient token use; combined related reads/edits into single tool calls per the directive

---

## Cross-references

- **R1' per-substrate review memos**:
  - B'=Z7-Mamba-2-v2: `.omx/research/path_3_b_recursive_adversarial_review_r1_prime_3_axis_20260526.md`
  - C'=NSCS06 v8 chroma_lut: `.omx/research/path_3_c_recursive_adversarial_review_r1_prime_3_axis_20260526.md`
  - F=Z8: `.omx/research/path_3_f_recursive_adversarial_review_r1_prime_3_axis_20260526.md`
  - G=NIRVANA: `.omx/research/path_3_g_recursive_adversarial_review_r1_prime_3_axis_20260526.md`
- **Landing memos under review**:
  - B': `.omx/research/path_3_b_z7_mamba_2_L0_scaffold_landed_20260526.md` (commit `7a103fdbb`)
  - C': `.omx/research/path_3_c_nscs06_v8_chroma_lut_L0_scaffold_landed_20260526.md` (commit `f59c8401b`)
  - F=Z8: `.omx/research/path_3_f_z8_hierarchical_predictive_coding_L0_scaffold_landed_20260526.md` (commit `5ff5d2ab9`)
  - G=NIRVANA: `.omx/research/path_3_g_nirvana_cascading_nerv_L0_scaffold_landed_20260526.md` (commit `f7d2e86fe`)
- **Sister R1 aggregate (A+D+E review)**: `.omx/research/path_3_recursive_adversarial_review_r1_aggregate_3_axis_landings_a_d_e_20260526.md`
- **FIX-WAVE-R1 close findings landed** (canonical fix pattern for analogous A=DreamerV3 bugs that F=Z8 inherits): `.omx/research/path_3_fix_wave_r1_close_findings_landed_20260526.md` (commit `a23779a732e7bb056`)
- **Canonical references**:
  - Sister D=Z6 `_pixel_shuffle_2x_nhwc` (canonical correct convention): `src/tac/substrates/time_traveler_l5_z6/mlx_renderer.py::_pixel_shuffle_2x_nhwc` (lines 361-372)
  - Canonical PR95 helper: `tac.local_acceleration.pr95_hnerv_mlx::bilinear_resize2x_align_corners_false_nhwc`
  - Canonical equation registry: `tac.canonical_equations.query_equations()`
- **In-flight sister subagents (DEFERRED per Catalog #230 ownership map; NOT reviewed in this R1'; queued for R1'' when they land)**:
  - H=`aba5069741fc4475b` ATW V2 cooperative-receiver cargo-cult-first
  - I=`a71f2c4404c978f50` V1 Faiss IVF-PQ residual cargo-cult-first
  - J=`abfd5113f1892447c` MDL-IBPS cargo-cult-first
  - K=`a7977f23a7f0f0573` COIN++ implicit neural representation
  - FIX-WAVE-R1=`a23779a732e7bb056` (LANDED before R1'; closed R1 P0+P1+P2)
  - L1-PROMOTION-D-Z6 in-flight (L1 promotion advances substrate readiness not review counter; out of scope)
- **Lane**: `lane_path_3_recursive_adversarial_review_r1_prime_3_axis_landings_b_c_f_g_20260526` L1 (impl_complete + memory_entry pending after commit)

---

## Final aggregate verdict

**PROCEED_WITH_REVISIONS** — R1' NOT CLEAN; counter resets to 0; FIX-WAVE-R1' successor subagent required; R2' BLOCKED until P0 op-routables land and verification passes.

The substrate paradigms (Mamba-2 selective SSM / NSCS06 v8 chroma LUT cargo-cult-unwound / Z8 hierarchical predictive coding canonical-quadruple / NIRVANA hierarchical residual cascade) are HARD-EARNED at the math + scientific + engineering level per Axis 1 across all 4 substrates. The IMPLEMENTATION-LEVEL gaps (2 CRITICAL in F=Z8; 3 documentation-only in G=NIRVANA) are TIGHTLY SCOPED and resolvable in a single FIX-WAVE-R1' commit batch (≤4 files; ≤90 LOC of edits).

Per CLAUDE.md "Forbidden premature KILL without research exhaustion" + Catalog #307 paradigm-vs-implementation classification: these are IMPLEMENTATION-LEVEL findings (F=Z8) and DOCUMENTATION-LEVEL findings (G=NIRVANA) requiring FIX-WAVE-R1', NOT paradigm-level kills. All 4 substrates remain `research_only=true` per their landing-time posture; the L0→L1 promotion path is unblocked by the FIX-WAVE-R1' + R2'-R4' clean-pass cycle.

**Notable wins from R1'**:
1. B' + C' demonstrate the CARGO-CULT-FIRST methodology is empirically materializable + R1' clean-passes naturally → operator directive #2 compliance pays off in review quality.
2. C' demonstrates EMPIRICAL CONFIRMATION at L0 (FAIL_AT_CLASS_1 verdict) of cargo-cult #5 → operator MVP-first phasing pays off in disambiguator quality.
3. G=NIRVANA establishes the CANONICAL SISTER-NUMPY-REFERENCE PATTERN that META-CONSOLIDATE-OP-2 will operationalize for the entire repo.
4. F=Z8 demonstrates EMPIRICAL PROOF that META-CONSOLIDATE-OP-1 cannot be deferred → forces priority escalation of canonical-helper extraction.

Estimated FIX-WAVE-R1' wall-clock: ~45-60min for a successor subagent with PV + canonical-serializer discipline. Estimated R2'-R4' cycle: ~3-4h per round × 3 rounds = 9-12h total to reach SEAL via counter-advance path. The CONSOLIDATE-OPs (META-CONSOLIDATE-OP-1 + CONSOLIDATE-OP-2) add ~2-4h to the L1+ work but are NOT blocking R2'.

**Mission alignment per Catalog #300**: `frontier_protecting` — the R1' review prevents L0→L1 promotion of substrates with TRAINING-INVALIDATING MLX↔PyTorch drift bugs (F=Z8) AND DOCUMENTATION OVERSTATEMENT (G=NIRVANA) (which would silently corrupt L1+ score-aware-loss training and produce phantom-score Modal dispatches per Catalog #313 + #341 sister discipline). Closing FIX-WAVE-R1' + R2'-R4' unblocks the canonical Path 3 substrate-class-shift pursuit at $0 cost. The META-CONSOLIDATE-OPs structurally extinct the bug classes so future Path 3 candidates inherit canonical correctness rather than re-invent the bugs.
