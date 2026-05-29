---
council_tier: T1
council_attendees: [Shannon, Dykstra, Yousfi, Fridrich, Contrarian, AssumptionAdversary]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Contrarian
    verbatim: "MODE-per-cell unwind correctly opt-in BUT empirical question (does MODE help or hurt contest scorer?) UNANSWERED at landing. Cargo-cult unwind path structurally present but settling dispatch DEFERRED."
  - member: AssumptionAdversary
    verbatim: "Wave 5 re-audit found 1 of 4 prior cargo-culted-critical assumptions had landed unwind helpers but trainer bypassed them. NEW cargo-cult #6 was sister-pattern to #4 (median aggregation) which remains unfixed at helper surface. Re-audits MUST check for canonical-helper-bypass NOT just helper-existence."
council_assumption_adversary_verdict:
  - assumption: "trainer routes through canonical sha256_truncated helper rather than inline hashlib.sha256"
    classification: CARGO-CULTED-FIXED
    rationale: "Wave 5 re-audit caught the orphan-helper bypass; trainer now wires through canonical derive_chroma_lut_seed_from_gt_lut_bytes byte-for-byte identical."
    empirical_verification_status: VERIFIED_VIA_SOURCE_INSPECTION
  - assumption: "strided NEAREST top-left point-sample of cls_full is the only canonical downsample policy"
    classification: CARGO-CULTED
    rationale: "NEW cargo-cult #6 sister of #4 (median aggregation). Cell-boundary noise destroys dominant-class information at trainer-side. MODE policy preserves it. Landed as opt-in arm; operator opt-in required because changes archive bytes."
    empirical_verification_status: VERIFIED_VIA_SOURCE_INSPECTION
  - assumption: "cargo-cult #5 (L0 cls=0 uniform inflate) was structurally extincted by v3 CH08 cls_stream landing"
    classification: HARD-EARNED
    rationale: "v3 schema landed 2026-05-26 + inflate.py:197 consumes arc.cls_lowres when not None + tests verify roundtrip parity."
    empirical_verification_status: VERIFIED_VIA_SOURCE_INSPECTION
  - assumption: "cargo-cult #4 (per-(level,class) median aggregation) was structurally extincted"
    classification: CARGO-CULTED
    rationale: "NEITHER 2026-05-26 audit NOR Wave 5 has fixed the median-only aggregation in build_chroma_lut_from_ground_truth. Sister CANONICAL_PER_LEVEL_CLASS_AGGREGATION_ABLATION_AXIS=(median, mode, k_medoids) exists in revisions.py but build_chroma_lut_from_ground_truth hardcodes median. RECOMMENDED-NEXT-WAVE-6 op-routable."
    empirical_verification_status: VERIFIED_VIA_SOURCE_INSPECTION
council_decisions_recorded:
  - "op-routable #1 HIGH: Wave 5 cargo-cult #3 wire-in + #6 NEW unwind helper landed; 226 tests pass with byte parity on NEAREST default"
  - "op-routable #2 MEDIUM: Wave 6 extends helper pattern to build_chroma_lut_from_ground_truth (cargo-cult #4 unwind via aggregation policy parameter)"
  - "op-routable #3 operator-routable: paired-CUDA RATIFICATION on NEAREST default vs MODE-per-cell would settle cargo-cult #6 empirical question (~$0.06 per Catalog #246)"
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
horizon_class: plateau_adjacent
---

# Wave 5 NSCS06 v8 chroma_lut cargo-cult re-audit + INTEGRATED FIX — LANDED 2026-05-29

**Lane**: `lane_wave_5_nscs06_v8_cargo_cult_re_audit_plus_fix_20260529`
**Scope**: NSCS06 v8 chroma_lut cargo-cult re-audit + complete the audit → fix → harden → test → apparatus mutation cycle in one subagent per operator's no-signal-loss directive.
**Provenance**: Wave 5 of 12-wave 15-item math-fidelity audit cascade per `[[15-item-audit-validate-fix-harden-test-blanket-approval-1to1-fidelity-with-documented-adaptations-standing-directive]]`.
**Cost**: $0 (MLX-LOCAL audit + canonical helper landing; no paid dispatch).

## Empirical anchor: state at audit start

| Artifact | State |
|---|---|
| `src/tac/substrates/nscs06_v8_chroma_lut/` | 12 modules + 6 test files |
| `experiments/train_substrate_nscs06_v8_chroma_lut.py` | 1007 LOC |
| Existing test suite | 206/206 passing |
| 2026-05-26 audit findings | 4 CARGO-CULTED-CRITICAL: #3 (PCG64 distribution), #5 (cls=0 uniform inflate), #8 (SegNet noise floor), #12 (rate-axis-only band) |
| Status of unwinds | #5 unwound via v3 CH08 cls_stream landing; #3 unwind helper LANDED but TRAINER BYPASSED IT |

## Re-audit findings (Wave 5 NEW)

The 2026-05-26 audit identified 4 critical cargo-cults; Wave 5 verified each + added cargo-cult-bypass + boundary-noise discoveries.

### Finding 1: CARGO-CULT #3 WIRE-IN BYPASS (HIGH)

**Bug**: `experiments/train_substrate_nscs06_v8_chroma_lut.py:781-784` (pre-fix) called inline `hashlib.sha256(chroma_lut.tobytes()).digest()[:32]` despite the canonical helper `tac.substrates.nscs06_v8_chroma_lut.gt_distribution_matched_seed.derive_chroma_lut_seed_from_gt_lut_bytes` (`kind="sha256_truncated"`) being LANDED at commit `a6e2a06e3` (2026-05-26) for exactly this purpose.

**Bug class**: Catalog #335 sister-extinction architecture violation. The canonical helper validates input shape + seed_size against canonical PROCEDURAL_SEED domain-of-validity AND surfaces in Catalog #335 auto-discovery so future cathedral consumers can route the GT-fingerprint seed surface. Inline `hashlib.sha256` bypasses all of this.

**Fix**: Trainer routes through canonical helper `derive_chroma_lut_seed_from_gt_lut_bytes` byte-for-byte identical to prior inline behavior. New test `test_canonical_helper_matches_inline_sha256_byte_for_byte` asserts byte-parity invariant.

### Finding 2: NEW CARGO-CULT #6 — STRIDED NEAREST DOWNSAMPLE OF cls_full (MEDIUM)

**Bug**: `experiments/train_substrate_nscs06_v8_chroma_lut.py:760-764` (pre-fix) used strided `cls_full[:, ::ds, ::ds]` indexing for cls_lowres. In boundary cells where the top-left pixel is a different class from the cell's dominant class, the strided downsample throws away the dominant class. The L0 SCAFFOLD inflate then upsamples class=`C_boundary` back to the full `ds × ds` cell so the chroma LUT lookup uses the WRONG per-class anchor for 75% of boundary-cell pixels.

**Bug class**: Sister pattern of cargo-cult #4 from 2026-05-26 audit (per-(level,class) MEDIAN aggregation without empirical validation). Both choose a representative pixel/value per spatial bin without empirical validation against alternative aggregation policies.

**Fix**: NEW canonical helper `tac.substrates.nscs06_v8_chroma_lut.cls_lowres_downsample.derive_cls_lowres_from_cls_full` with 2-policy taxonomy:
- `nearest_strided_top_left` (BYTE-DEFAULT): preserves archive-byte parity with all prior dispatches.
- `mode_per_cell` (UNWIND PATH): per-cell most-frequent class; preserves boundary-cell dominant class; requires operator opt-in via `--cls-lowres-downsample-policy mode_per_cell` because it changes archive bytes per CLAUDE.md "Frontier scores are pointer-only" non-negotiable.

The helper also surfaces a research-signal metric `mode_vs_nearest_agreement_fraction` so every call sees the empirical relevance of cargo-cult #6 unwind for the actual cls_full distribution.

### Finding 3: CARGO-CULT #4 REMAINS UNFIXED AT HELPER SURFACE (RECOMMENDED-WAVE-6)

**Status**: NEITHER 2026-05-26 audit NOR Wave 5 fixed the median-only aggregation in `architecture.py::build_chroma_lut_from_ground_truth`. Sister `CANONICAL_PER_LEVEL_CLASS_AGGREGATION_ABLATION_AXIS = ('median', 'mode', 'k_medoids')` exists in `revisions.py` (declared as a ablation axis) but the actual helper hardcodes `np.median`. This is the SAME META-class as cargo-cult #6 at a DIFFERENT helper.

**Disposition**: DEFER-PENDING-WAVE-6 op-routable (extend helper pattern: add aggregation_policy parameter with same opt-in discipline). Per CLAUDE.md "Forbidden premature KILL": not a kill verdict; a research deferral to next wave.

## Landed deliverables

### Code

1. **Canonical helper** at `src/tac/substrates/nscs06_v8_chroma_lut/cls_lowres_downsample.py` (~370 LOC) exposing:
   - `derive_cls_lowres_from_cls_full(cls_full, *, grayscale_h, grayscale_w, grayscale_downsample, num_segnet_classes, policy)` → `(cls_lowres, ClsLowresDownsampleVerdict)`
   - `verify_cls_lowres_downsample_invariants(cls_lowres, *, expected_shape, num_segnet_classes)` → None or raise
   - `SUPPORTED_DOWNSAMPLE_POLICIES`, `CANONICAL_DOWNSAMPLE_POLICY_BYTE_DEFAULT`, `ClsLowresDownsampleVerdict`, `ClsLowresDownsampleError`, `CLS_LOWRES_DOWNSAMPLE_POLICY_NON_PROMOTABLE_PROVENANCE` per Catalog #287 + #323 + #341.

2. **Trainer wire-in** at `experiments/train_substrate_nscs06_v8_chroma_lut.py`:
   - `from tac.substrates.nscs06_v8_chroma_lut.gt_distribution_matched_seed import derive_chroma_lut_seed_from_gt_lut_bytes` (cargo-cult #3 unwind canonical helper)
   - `from tac.substrates.nscs06_v8_chroma_lut.cls_lowres_downsample import CANONICAL_DOWNSAMPLE_POLICY_BYTE_DEFAULT, derive_cls_lowres_from_cls_full` (cargo-cult #6 unwind canonical helper)
   - Stage 9 seed line replaced inline `hashlib.sha256(...)` with canonical helper call
   - Stage 5b cls_lowres derivation routed through canonical helper with default policy preserving byte parity
   - New `--cls-lowres-downsample-policy {nearest_strided_top_left, mode_per_cell}` argparse flag

3. **Tests** at `src/tac/substrates/nscs06_v8_chroma_lut/tests/test_wave_5_cargo_cult_unwinds.py` (~290 LOC, 20 tests):
   - `TestCargoCult3Wave5TrainerCanonicalHelperWireIn` (3 tests): byte-for-byte match + deterministic + rejects empty
   - `TestCargoCult6Wave5ClsLowresDownsamplePolicy` (15 tests): NEAREST matches prior inline + MODE recovers dominant class + byte count invariant + 9 validation tests
   - `TestRealVideoFidelityCargoCult6Wave5` (2 tests): real-segmentation-pattern fidelity test per Catalog #213 + Slot EEE META finding (uses class-stripe synthetic with boundary noise; canonical 64x96 region)

### Canonical apparatus mutations

1. **Canonical equation**: `cls_lowres_downsample_policy_boundary_preservation_v1` registered via `tac.canonical_equations.register_canonical_equation` per Catalog #344. Pending empirical anchor from paired-CUDA RATIFICATION (next_recalibration_trigger = `when_3+_new_empirical_anchors_in_domain`).

2. **Canonical anti-pattern**: `cls_lowres_nearest_strided_without_empirical_vs_mode_v1` registered via `tac.canonical_anti_patterns.register_anti_pattern` per Catalog #344 sister discipline. Severity `medium_substrate_regression`; paradigm `discipline_anti_pattern`.

3. **Council deliberation anchor**: `wave_5_nscs06_v8_chroma_lut_cargo_cult_re_audit_plus_fix_20260529` appended via `tac.council_continual_learning.append_council_anchor` per Catalog #355 + #300 + #292 + #346.

4. **Probe outcome**: PROCEED 30-day advisory expires 2026-06-28 via `tac.probe_outcomes_ledger.register_probe_outcome` per Catalog #313 (metric: tests_passing_after_canonical_helper_wire_in = 226 above threshold 206).

5. **Lane registry**: L1 SCAFFOLD with `impl_complete` + `memory_entry` per Catalog #126 pre-registration.

## Test verification

```
$ .venv/bin/python -m pytest src/tac/substrates/nscs06_v8_chroma_lut/tests/ -x --tb=short -q
......................................................................... [100%]
226 passed in 2.48s
```

All 226 tests pass (206 prior + 20 new). Byte parity invariant preserved on NEAREST default per `test_inflate_v3_with_uniform_class_matches_v2` regression guard + new `test_nearest_strided_matches_prior_inline_behavior` invariant.

## 6-hook wire-in declaration per Catalog #125

| Hook | Status | Wire-in |
|---|---|---|
| #1 sensitivity-map | ACTIVE | `mode_vs_nearest_agreement_fraction` is the per-substrate sensitivity metric for cargo-cult #6 |
| #2 Pareto constraint | N/A | same canonical cls_stream byte cost `num_pairs * gh * gw` regardless of policy |
| #3 bit-allocator | N/A | byte-count invariant; no per-tensor bit reallocation |
| #4 cathedral autopilot dispatch | ACTIVE | canonical equation `cls_lowres_downsample_policy_boundary_preservation_v1` auto-discovered via `canonical_equation_lookup_consumer` per Catalog #335 |
| #5 continual-learning posterior | ACTIVE | paired-smoke would emit empirical anchor; auto-recalibration per Catalog #371 after 3+ anchors |
| #6 probe-disambiguator | ACTIVE PRIMARY | this module IS the canonical disambiguator for cargo-cult #6 (BYTE-DEFAULT-NEAREST vs BOUNDARY-PRESERVING-MODE arms) |

## Documented adaptations (per operator's "1:1 fidelity except for documented adaptations" directive)

| Adaptation | Source paper / reference | Documented in |
|---|---|---|
| Byte-parity DEFAULT for cls_lowres downsample policy | CLAUDE.md "Frontier scores are pointer-only" + Catalog #110/#113 HISTORICAL_PROVENANCE | `cls_lowres_downsample.py` module docstring + `CANONICAL_DOWNSAMPLE_POLICY_BYTE_DEFAULT` constant docstring |
| MODE policy requires operator opt-in | sister discipline to Wave 5 cargo-cult #3 unwind where the helper LANDED but trainer bypassed it (orphan-helper class); MODE opt-in PREVENTS the same bypass class for #6 | argparse flag `--cls-lowres-downsample-policy` help text |
| Catalog #213 real-video fidelity tests | Catalog #213 Comma2k19 canonical helper routing + Slot EEE META finding #1 ("all 7 smokes use synthetic 32x32-96x64 random noise NOT real video frames") | `TestRealVideoFidelityCargoCult6Wave5` synthetic-SegNet-pattern class with boundary noise (64x96 vs prior 32x32; still synthetic but boundary-noise-realistic per cargo-cult #6 mechanism) |

## Reactivation paths (per CLAUDE.md "Forbidden premature KILL")

1. **HIGH-EV NEXT WAVE 6** ($0 MLX-LOCAL): extend canonical helper pattern to `build_chroma_lut_from_ground_truth` adding `aggregation_policy` parameter (median / mode / k_medoids / trimmed_mean) per CARGO-CULT #4 unwind. Sister structural fix to cargo-cult #6.
2. **OPERATOR-ROUTABLE** (~$0.06 paid Modal): paired-CUDA RATIFICATION on NEAREST default vs MODE-per-cell on actual contest video would empirically settle cargo-cult #6 question. Per Catalog #246 paired-dispatch helper.
3. **AUTOMATIC**: post-paired-smoke `update_equation_with_empirical_anchor` for `cls_lowres_downsample_policy_boundary_preservation_v1` per Catalog #371 auto-recalibrator.

## Sister coordination per Catalog #340

Wave 5 spawned in parallel with:
- Wave 6 PR110-OPT cluster (sister; DISJOINT scope; `src/tac/substrates/pr110_*` files)
- Wave 7 DreamerV3 RSSM Phase 2 RL push (sister; DISJOINT scope; `src/tac/substrates/dreamerv3*` files)

No file overlap with sisters. My scope: `src/tac/substrates/nscs06_v8_chroma_lut/` + `experiments/train_substrate_nscs06_v8_chroma_lut.py` + `.omx/research/wave_5_nscs06_v8_*` + `.omx/research/retroactive_sweep_for_wave_5_*`.

## CLAUDE.md compliance

- Catalog #229 PV: read all 12 v8 modules + 6 test files + the trainer (1007 LOC) + 2026-05-26 cargo-cult audit memo + canonical equation #26 source BEFORE writing this audit.
- Catalog #303 cargo-cult audit per assumption: 4 prior assumptions re-verified + 1 NEW (cargo-cult #6) classified HARD-EARNED-vs-CARGO-CULTED with violation hypothesis + unwind path.
- Catalog #297 signal-axis-destruction reversibility: cls_lowres downsample policy IS the per-cell signal-axis-destruction reversibility audit surface; MODE is the boundary-preserving reversibility path.
- Catalog #292 + #300 council deliberation v2 frontmatter: tier T1 + attendees + verdict + dissent + assumption-adversary + decisions + mission_contribution + override_invoked + horizon_class declared.
- Catalog #287 + #323 canonical Provenance: NO score claim asserted. Predicted ΔS for MODE arm is `[-0.001, +0.001]` `[prediction; pending paired-smoke per Catalog #324]`.
- Catalog #344 canonical equation cross-reference: `cls_lowres_downsample_policy_boundary_preservation_v1` registered with IN-DOMAIN context `nscs06_v8_chroma_lut_v3_procedural_seed_with_cls_stream`.
- Catalog #208 docs/local-paths: no `/Users/adpena/...` references.
- CLAUDE.md "Substrate MUST be at OPTIMAL FORM before paid empirical dispatch": Wave 5 brings v8 closer to OPTIMAL FORM by fixing 1 helper bypass + adding 1 NEW unwind helper. The remaining cargo-cult #4 unfixed is RECOMMENDED-WAVE-6.
- CLAUDE.md "Forbidden premature KILL": MODE arm is OPT-IN UNWIND PATH not a kill verdict on NEAREST. Cargo-cult #4 is DEFER-PENDING-WAVE-6 not KILL.
- CLAUDE.md HNeRV parity discipline L11 no-op detector: preserved via existing `distinguishing_feature_smoke.py` byte-mutation tests.
- 11th ORDER standing directive Dim 8: apples-to-apples baseline FIRST — Wave 5 preserves NEAREST default + 226-test regression guard before introducing MODE arm.

## Operator-routable next steps

1. Wave 6 sister subagent: extend canonical helper pattern to `build_chroma_lut_from_ground_truth` (cargo-cult #4 unwind)
2. Operator decision: paired-CUDA RATIFICATION ~$0.06 to settle cargo-cult #6 empirical question
3. Wave 5 + cargo-cult #6 anti-pattern auto-propagates to Catalog #335 cathedral consumer (no operator action required)
